//! Each project's wiki: markdown pages, named by their titles, that the board and agents
//! both read and write. Agents grep them, then read only the lines they need.

use std::collections::BTreeSet;

use regex_lite::{Regex, RegexBuilder};
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Serialize;

use super::{Error, NOW, Result, Store, squash};

/// Made outside the numbered migrations, so this table doesn't take a schema version that a
/// later Kanzo may number differently. Older versions leave the table alone.
pub(super) const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS wiki_pages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        title      TEXT NOT NULL,
        body       TEXT NOT NULL DEFAULT '',
        author     TEXT NOT NULL DEFAULT '',
        version    INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    CREATE UNIQUE INDEX IF NOT EXISTS wiki_pages_by_title ON wiki_pages(project_id, title COLLATE NOCASE);
    CREATE TRIGGER IF NOT EXISTS wiki_pages_insert AFTER INSERT ON wiki_pages BEGIN UPDATE meta SET value = value + 1 WHERE key = 'changes'; END;
    CREATE TRIGGER IF NOT EXISTS wiki_pages_update AFTER UPDATE ON wiki_pages BEGIN UPDATE meta SET value = value + 1 WHERE key = 'changes'; END;
    CREATE TRIGGER IF NOT EXISTS wiki_pages_delete AFTER DELETE ON wiki_pages BEGIN UPDATE meta SET value = value + 1 WHERE key = 'changes'; END;";

const TITLE_MAX: usize = 120;
pub const PAGE_MAX: usize = 512 * 1024;
/// A markdown paragraph is often one long line; grep shows this much of it, around the match.
const LINE_SHOWN: usize = 240;

#[derive(Debug, Clone, Serialize)]
pub struct Page {
    pub id: i64,
    pub project: String,
    pub title: String,
    pub body: String,
    /// Who wrote it last.
    pub author: String,
    /// Goes up with every change, so an editor can tell its copy is out of date.
    pub version: i64,
    pub created_at: String,
    pub updated_at: String,
}

/// A page without its text, for lists.
#[derive(Debug, Clone, Serialize)]
pub struct PageInfo {
    pub id: i64,
    pub project: String,
    pub title: String,
    pub author: String,
    pub version: i64,
    pub updated_at: String,
    pub lines: i64,
}

#[derive(Debug, Default)]
pub struct Grep {
    pub pages: Vec<PageHits>,
    pub matches: usize,
    /// Stopped at the limit: there are more matches than these.
    pub more: bool,
}

#[derive(Debug)]
pub struct PageHits {
    pub project: String,
    pub title: String,
    pub title_matches: bool,
    pub lines: Vec<HitLine>,
}

#[derive(Debug, PartialEq)]
pub struct HitLine {
    /// From 1, as read_wiki_page counts.
    pub number: usize,
    /// Or a line of context around a match.
    pub matched: bool,
    pub text: String,
}

impl Store {
    const PAGE_SELECT: &str = "SELECT w.id, p.slug, w.title, w.body, w.author, w.version, w.created_at, w.updated_at
        FROM wiki_pages w JOIN projects p ON p.id = w.project_id";

    fn page_row(r: &Row) -> rusqlite::Result<Page> {
        Ok(Page {
            id: r.get(0)?,
            project: r.get(1)?,
            title: r.get(2)?,
            body: r.get(3)?,
            author: r.get(4)?,
            version: r.get(5)?,
            created_at: r.get(6)?,
            updated_at: r.get(7)?,
        })
    }

    /// In title order; with `query`, only the pages whose title or text has all its words.
    pub fn list_pages(&self, project_id: Option<i64>, query: &str) -> Result<Vec<PageInfo>> {
        self.add_has_word()?;
        let mut sql = String::from(
            "SELECT w.id, p.slug, w.title, w.author, w.version, w.updated_at,
                    length(w.body) - length(replace(w.body, char(10), '')) + (substr(w.body, -1) NOT IN ('', char(10)))
             FROM wiki_pages w JOIN projects p ON p.id = w.project_id
             WHERE (?1 IS NULL OR w.project_id = ?1)",
        );
        let mut values = vec![project_id.map_or(Value::Null, Value::Integer)];
        for word in query.split_whitespace() {
            values.push(Value::Text(word.to_lowercase()));
            let n = values.len();
            sql.push_str(&format!(" AND (has_word(w.title, ?{n}) OR has_word(w.body, ?{n}))"));
        }
        sql.push_str(" ORDER BY p.slug, w.title COLLATE NOCASE");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(values), |r| {
            Ok(PageInfo {
                id: r.get(0)?,
                project: r.get(1)?,
                title: r.get(2)?,
                author: r.get(3)?,
                version: r.get(4)?,
                updated_at: r.get(5)?,
                lines: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn get_page(&self, id: i64) -> Result<Page> {
        let sql = format!("{} WHERE w.id = ?1", Self::PAGE_SELECT);
        self.conn
            .query_row(&sql, [id], Self::page_row)
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("no wiki page #{id}")))
    }

    /// By title, ignoring case; the error lists the pages there are.
    pub fn find_page(&self, project_id: i64, title: &str) -> Result<Page> {
        let title = squash(title);
        let sql = format!("{} WHERE w.project_id = ?1 AND w.title = ?2 COLLATE NOCASE", Self::PAGE_SELECT);
        if let Some(page) = self.conn.query_row(&sql, params![project_id, title], Self::page_row).optional()? {
            return Ok(page);
        }
        let titles: Vec<String> = self
            .conn
            .prepare("SELECT title FROM wiki_pages WHERE project_id = ?1 ORDER BY title COLLATE NOCASE LIMIT 30")?
            .query_map([project_id], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let hint =
            if titles.is_empty() { "the wiki is empty".to_string() } else { format!("pages: {}", titles.join(", ")) };
        Err(Error::NotFound(format!("no wiki page {title:?} ({hint})")))
    }

    /// Fails if the project already has a page with that title.
    pub fn create_page(&self, project_id: i64, title: &str, body: &str, author: &str) -> Result<Page> {
        let title = clean_page_title(title)?;
        check_body(body)?;
        let id = self.write(|c| {
            if page_id(c, project_id, &title)?.is_some() {
                return Err(Error::Invalid(format!("a wiki page titled {title:?} already exists")));
            }
            insert_page(c, project_id, &title, body, author)
        })?;
        self.get_page(id)
    }

    /// Creates the page, or replaces its text. Says which it did.
    pub fn write_page(&self, project_id: i64, title: &str, body: &str, author: &str) -> Result<(Page, bool)> {
        let title = clean_page_title(title)?;
        check_body(body)?;
        let (id, created) = self.write(|c| match page_id(c, project_id, &title)? {
            Some(id) => {
                c.execute(
                    &format!(
                        "UPDATE wiki_pages SET body = ?1, author = ?2, version = version + 1, updated_at = {NOW}
                         WHERE id = ?3 AND body <> ?1"
                    ),
                    params![body, author, id],
                )?;
                Ok((id, false))
            }
            None => Ok((insert_page(c, project_id, &title, body, author)?, true)),
        })?;
        Ok((self.get_page(id)?, created))
    }

    /// Changes the title, the text or both. With `base`, only if the page is still at that
    /// version, so an edit made from an old copy never undoes someone else's.
    pub fn update_page(
        &self,
        id: i64,
        title: Option<&str>,
        body: Option<&str>,
        author: &str,
        base: Option<i64>,
    ) -> Result<Page> {
        let title = title.map(clean_page_title).transpose()?;
        if let Some(body) = body {
            check_body(body)?;
        }
        self.write(|c| {
            let (project_id, version, old_title, old_body): (i64, i64, String, String) = c
                .query_row("SELECT project_id, version, title, body FROM wiki_pages WHERE id = ?1", [id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })
                .optional()?
                .ok_or_else(|| Error::NotFound(format!("no wiki page #{id}")))?;
            if base.is_some_and(|b| b != version) {
                return Err(Error::Invalid(format!(
                    "{old_title:?} changed since you opened it; copy your text, then edit the newest version"
                )));
            }
            let title = title.unwrap_or_else(|| old_title.clone());
            let body = body.unwrap_or(&old_body);
            if title == old_title && body == old_body {
                return Ok(());
            }
            if let Some(other) = page_id(c, project_id, &title)?
                && other != id
            {
                return Err(Error::Invalid(format!("a wiki page titled {title:?} already exists")));
            }
            c.execute(
                &format!(
                    "UPDATE wiki_pages SET title = ?1, body = ?2, author = ?3, version = version + 1, updated_at = {NOW}
                     WHERE id = ?4"
                ),
                params![title, body, author, id],
            )?;
            Ok(())
        })?;
        self.get_page(id)
    }

    /// Replaces `old`, which must be in the page exactly once unless `all`. Returns the page and
    /// how many places changed.
    pub fn edit_page(
        &self,
        project_id: i64,
        title: &str,
        old: &str,
        new: &str,
        all: bool,
        author: &str,
    ) -> Result<(Page, usize)> {
        let page = self.find_page(project_id, title)?;
        if old.is_empty() {
            return Err(Error::Invalid("old is empty; to replace the whole page, write it".into()));
        }
        let count = page.body.matches(old).count();
        if count == 0 {
            return Err(Error::NotFound(format!(
                "the text isn't in {:?}; read the page again, as spaces and line breaks must match",
                page.title
            )));
        }
        if count > 1 && !all {
            return Err(Error::Invalid(format!(
                "the text is in {:?} {count} times; give more of the text around it, or pass all",
                page.title
            )));
        }
        let body = if all { page.body.replace(old, new) } else { page.body.replacen(old, new, 1) };
        let page = self.update_page(page.id, None, Some(&body), author, Some(page.version))?;
        Ok((page, count))
    }

    pub fn delete_page(&self, id: i64) -> Result<Page> {
        let page = self.get_page(id)?;
        self.conn.execute("DELETE FROM wiki_pages WHERE id = ?1", [id])?;
        Ok(page)
    }

    /// The lines that match `pattern`, with `context` lines around each, in title order. Stops
    /// after `max` matches. Like ripgrep's smart case, ignores case unless the pattern has a capital.
    pub fn grep_pages(&self, project_id: Option<i64>, pattern: &str, context: usize, max: usize) -> Result<Grep> {
        let matcher = Matcher::new(pattern)?;
        let sql = format!(
            "{} WHERE (?1 IS NULL OR w.project_id = ?1) ORDER BY p.slug, w.title COLLATE NOCASE",
            Self::PAGE_SELECT
        );
        let pages: Vec<Page> =
            self.conn.prepare(&sql)?.query_map([project_id], Self::page_row)?.collect::<rusqlite::Result<_>>()?;
        let mut out = Grep::default();
        for page in pages {
            let lines: Vec<&str> = page.body.lines().collect();
            let mut hits: Vec<(usize, usize)> =
                lines.iter().enumerate().filter_map(|(i, l)| matcher.find(l).map(|at| (i, at))).collect();
            let title_matches = matcher.find(&page.title).is_some();
            if hits.is_empty() && !title_matches {
                continue;
            }
            let room = max.saturating_sub(out.matches);
            if room == 0 {
                out.more = true;
                break;
            }
            if hits.len() > room {
                hits.truncate(room);
                out.more = true;
            }
            // A page found by its title alone counts too, or a common word would list every page.
            out.matches += hits.len().max(1);
            let mut shown = BTreeSet::new();
            for &(i, _) in &hits {
                shown.extend(i.saturating_sub(context)..=(i + context).min(lines.len() - 1));
            }
            let lines = shown
                .into_iter()
                .map(|i| {
                    let at = hits.iter().find(|h| h.0 == i).map(|h| h.1);
                    HitLine { number: i + 1, matched: at.is_some(), text: clip(lines[i], at.unwrap_or(0)) }
                })
                .collect();
            out.pages.push(PageHits { project: page.project, title: page.title, title_matches, lines });
        }
        Ok(out)
    }
}

fn page_id(c: &Connection, project_id: i64, title: &str) -> Result<Option<i64>> {
    Ok(c.query_row(
        "SELECT id FROM wiki_pages WHERE project_id = ?1 AND title = ?2 COLLATE NOCASE",
        params![project_id, title],
        |r| r.get(0),
    )
    .optional()?)
}

fn insert_page(c: &Connection, project_id: i64, title: &str, body: &str, author: &str) -> Result<i64> {
    c.execute(
        "INSERT INTO wiki_pages (project_id, title, body, author) VALUES (?1, ?2, ?3, ?4)",
        params![project_id, title, body, author],
    )?;
    Ok(c.last_insert_rowid())
}

fn clean_page_title(title: &str) -> Result<String> {
    let title = squash(title);
    if title.is_empty() {
        return Err(Error::Invalid("wiki page title is empty".into()));
    }
    if title.chars().count() > TITLE_MAX {
        return Err(Error::Invalid(format!("wiki page title is longer than {TITLE_MAX} characters")));
    }
    Ok(title)
}

fn check_body(body: &str) -> Result<()> {
    if body.len() > PAGE_MAX {
        return Err(Error::Invalid(format!("a wiki page holds up to {} KB; split it into pages", PAGE_MAX / 1024)));
    }
    Ok(())
}

/// regex-lite ignores case for ASCII only, so for the rest the pattern and each line are
/// lowercased a character at a time, which keeps match positions in characters the same.
struct Matcher {
    re: Regex,
    fold: bool,
}

impl Matcher {
    fn new(pattern: &str) -> Result<Self> {
        if pattern.is_empty() {
            return Err(Error::Invalid("pattern is empty".into()));
        }
        let fold = !has_capital(pattern);
        let source = if fold { lower(pattern) } else { pattern.to_string() };
        let re = RegexBuilder::new(&source)
            .case_insensitive(fold)
            .build()
            .map_err(|e| Error::Invalid(format!("bad pattern: {e}")))?;
        Ok(Self { re, fold })
    }

    /// The character the first match starts at.
    fn find(&self, line: &str) -> Option<usize> {
        if self.fold && !line.is_ascii() {
            let lowered = lower(line);
            self.re.find(&lowered).map(|m| lowered[..m.start()].chars().count())
        } else {
            self.re.find(line).map(|m| line[..m.start()].chars().count())
        }
    }
}

/// Escapes like `\S` and `\W` don't count as capitals.
fn has_capital(pattern: &str) -> bool {
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            chars.next();
        } else if c.is_uppercase() {
            return true;
        }
    }
    false
}

fn lower(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii() { c } else { c.to_lowercase().next().unwrap_or(c) }).collect()
}

/// At most LINE_SHOWN characters, from a little before the match.
fn clip(line: &str, at: usize) -> String {
    let len = line.chars().count();
    if len <= LINE_SHOWN {
        return line.to_string();
    }
    let start = at.saturating_sub(LINE_SHOWN / 4).min(len - LINE_SHOWN);
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(line.chars().skip(start).take(LINE_SHOWN));
    if start + LINE_SHOWN < len {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, Store, i64) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("kanzo.db"), None).unwrap();
        store.migrate().unwrap();
        let p = store.create_project("api", "API", None).unwrap();
        (dir, store, p.id)
    }

    #[test]
    fn pages_are_found_by_title_whatever_its_case() {
        let (_dir, s, p) = store();
        let (page, created) = s.write_page(p, "  Deploy   steps ", "1. build\n2. ship", "me").unwrap();
        assert!(created);
        assert_eq!(page.title, "Deploy steps");
        assert_eq!(s.find_page(p, "deploy STEPS").unwrap().id, page.id);
        let (again, created) = s.write_page(p, "DEPLOY steps", "1. test\n2. ship", "agent").unwrap();
        assert!(!created);
        assert_eq!((again.id, again.version, again.author.as_str()), (page.id, 2, "agent"));
        assert_eq!(again.title, "Deploy steps", "rewriting keeps the title as it was");
        let err = s.find_page(p, "Nope").unwrap_err().to_string();
        assert!(err.contains("pages: Deploy steps"), "{err}");
    }

    #[test]
    fn writing_the_same_text_changes_nothing() {
        let (_dir, s, p) = store();
        s.write_page(p, "A", "text", "me").unwrap();
        let before = s.changes().unwrap();
        let (page, _) = s.write_page(p, "A", "text", "agent").unwrap();
        assert_eq!((page.version, page.author.as_str()), (1, "me"));
        assert_eq!(s.changes().unwrap(), before);
    }

    #[test]
    fn titles_are_unique_in_a_project_only() {
        let (_dir, s, p) = store();
        let other = s.create_project("web", "", None).unwrap().id;
        s.create_page(p, "Notes", "", "me").unwrap();
        assert!(s.create_page(p, "notes", "", "me").is_err());
        s.create_page(other, "Notes", "", "me").unwrap();
        let b = s.create_page(p, "Other", "", "me").unwrap();
        assert!(s.update_page(b.id, Some("NOTES"), None, "me", None).is_err());
        assert!(s.create_page(p, " ", "", "me").is_err());
    }

    #[test]
    fn an_edit_from_an_old_copy_is_refused() {
        let (_dir, s, p) = store();
        let page = s.create_page(p, "A", "one", "me").unwrap();
        s.update_page(page.id, None, Some("two"), "agent", Some(page.version)).unwrap();
        let err = s.update_page(page.id, None, Some("three"), "me", Some(page.version)).unwrap_err();
        assert!(err.to_string().contains("changed since you opened it"), "{err}");
        assert_eq!(s.get_page(page.id).unwrap().body, "two");
    }

    #[test]
    fn edits_replace_text_found_once_or_everywhere() {
        let (_dir, s, p) = store();
        s.write_page(p, "A", "port 4242\nhost a\nhost b", "me").unwrap();
        let (page, n) = s.edit_page(p, "a", "4242", "8080", false, "agent").unwrap();
        assert_eq!((page.body.as_str(), n), ("port 8080\nhost a\nhost b", 1));
        let err = s.edit_page(p, "A", "host", "server", false, "agent").unwrap_err().to_string();
        assert!(err.contains("2 times"), "{err}");
        let (page, n) = s.edit_page(p, "A", "host", "server", true, "agent").unwrap();
        assert_eq!((page.body.as_str(), n), ("port 8080\nserver a\nserver b", 2));
        assert!(s.edit_page(p, "A", "missing", "x", false, "agent").is_err());
    }

    #[test]
    fn pages_go_with_their_project() {
        let (_dir, s, p) = store();
        s.write_page(p, "A", "x", "me").unwrap();
        s.delete_project("api").unwrap();
        assert!(s.list_pages(None, "").unwrap().is_empty());
    }

    #[test]
    fn lists_count_lines_and_filter_by_words() {
        let (_dir, s, p) = store();
        s.write_page(p, "Setup", "Install Rust\nRun cargo build\n", "me").unwrap();
        s.write_page(p, "Empty", "", "me").unwrap();
        s.write_page(p, "Ünïcode", "ÉCLAIR recipe", "me").unwrap();
        let all = s.list_pages(Some(p), "").unwrap();
        let summary: Vec<(&str, i64)> = all.iter().map(|i| (i.title.as_str(), i.lines)).collect();
        assert_eq!(summary, [("Empty", 0), ("Setup", 2), ("Ünïcode", 1)]);
        let titles = |q| s.list_pages(Some(p), q).unwrap().into_iter().map(|i| i.title).collect::<Vec<_>>();
        assert_eq!(titles("cargo install"), ["Setup"]);
        assert_eq!(titles("éclair"), ["Ünïcode"]);
    }

    #[test]
    fn grep_shows_matching_lines_with_context() {
        let (_dir, s, p) = store();
        s.write_page(p, "Notes", "one\ntwo\nthree\nfour\nfive\nsix\nseven", "me").unwrap();
        s.write_page(p, "Three things", "nothing here", "me").unwrap();
        let g = s.grep_pages(Some(p), "three|six", 1, 100).unwrap();
        let notes = &g.pages[0];
        let shown: Vec<(usize, bool)> = notes.lines.iter().map(|l| (l.number, l.matched)).collect();
        assert_eq!(shown, [(2, false), (3, true), (4, false), (5, false), (6, true), (7, false)]);
        let titled = &g.pages[1];
        assert!(titled.title_matches && titled.lines.is_empty());
        assert_eq!(g.matches, 3);
        assert!(!g.more);
    }

    #[test]
    fn grep_ignores_case_unless_the_pattern_has_a_capital() {
        let (_dir, s, p) = store();
        s.write_page(p, "A", "Deploy on Friday\nПривет мир", "me").unwrap();
        let hits = |pat| s.grep_pages(Some(p), pat, 0, 100).unwrap().matches;
        assert_eq!(hits("deploy"), 1);
        assert_eq!(hits("Deploy"), 1);
        assert_eq!(hits("DEPLOY"), 0);
        assert_eq!(hits(r"\Sriday"), 1, "an escape isn't a capital");
        assert_eq!(hits("привет"), 1, "case folds beyond ASCII");
        assert!(s.grep_pages(Some(p), "(", 0, 100).is_err());
    }

    #[test]
    fn grep_stops_at_the_limit_and_clips_long_lines() {
        let (_dir, s, p) = store();
        let long = format!("{}needle{}", "a".repeat(500), "b".repeat(500));
        s.write_page(p, "A", &format!("x\nx\nx\n{long}"), "me").unwrap();
        let g = s.grep_pages(Some(p), "x", 0, 2).unwrap();
        assert_eq!((g.matches, g.more), (2, true));
        let g = s.grep_pages(Some(p), "needle", 0, 10).unwrap();
        let text = &g.pages[0].lines[0].text;
        assert!(text.contains("needle") && text.starts_with('…') && text.ends_with('…'), "{text}");
        assert!(text.chars().count() <= LINE_SHOWN + 2);
    }
}
