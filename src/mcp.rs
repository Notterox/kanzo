//! Replies are short plain text to save agents' tokens.

use std::collections::HashSet;
use std::fmt::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use anyhow::Result;
use axum::http::request::Parts;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Extensions, Implementation,
    ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::home::Home;
use crate::share::{self, Net};
use crate::store::{self, Error, Grep, Project, Store, Task, TaskPatch, short_time};

/// An old board has thousands of done tasks.
const DONE_LISTED: usize = 20;

/// grep_wiki stops here, so a pattern that matches everything doesn't flood an agent's context.
const GREP_MATCHES: usize = 60;

macro_rules! instructions {
    ($whose:literal, $project:literal) => {
        concat!(
            "The user's kanban board, on ",
            $whose,
            " machine, for all their projects. Tasks look like api#12. ",
            $project,
            " Keep it current unasked: doing when you start a task, comments on what you did, review when you \
             finish or get stuck, url for a PR you opened. On review, comment how to resume this session. Leave done and human tasks to the user. \
             Link files as [name](file:///path). Each project has a wiki for lasting notes: grep_wiki, then read only the lines you need."
        )
    };
}

const INSTRUCTIONS: &str =
    instructions!("this", "Without `project`, Kanzo uses the project linked to this folder's GitHub repo.");
const HTTP_INSTRUCTIONS: &str =
    instructions!("their", "Over HTTP Kanzo can't see your folder, so pass `project` (see list_projects).");

const NO_PROJECTS: &str = "There are no projects yet. Create one with create_project.";

pub async fn serve(home: Home) -> Result<()> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let repos = git_repos(&cwd);
    let images = home.images();
    let ctx = Ctx { home, cwd: Some(cwd), repos, images, host: Default::default(), files: true };
    let server = Kanzo { ctx: Arc::new(ctx), tool_router: Kanzo::tools() };
    serve_stdio(server).await
}

#[cfg(unix)]
use crate::stdio::serve as serve_stdio;

/// Windows can't replace a running program, so there an agent's `kanzo mcp` keeps its version.
#[cfg(not(unix))]
async fn serve_stdio(server: Kanzo) -> Result<()> {
    use rmcp::ServiceExt;
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// `web::protect` checks Host and Origin, including when the board listens on the network,
/// so rmcp's localhost-only list is turned off.
pub fn http_service(home: Home, shutdown: CancellationToken) -> StreamableHttpService<Kanzo, LocalSessionManager> {
    let images = home.images();
    let ctx = Arc::new(Ctx { home, cwd: None, repos: Vec::new(), images, host: Default::default(), files: false });
    let config = StreamableHttpServerConfig::default().disable_allowed_hosts().with_cancellation_token(shutdown);
    // rmcp makes a server per session, and for some requests; the tool list is built once.
    let tool_router = Kanzo::tools();
    StreamableHttpService::new(
        move || Ok(Kanzo { ctx: ctx.clone(), tool_router: tool_router.clone() }),
        Default::default(),
        config,
    )
}

struct Ctx {
    home: Home,
    cwd: Option<PathBuf>,
    repos: Vec<String>,
    images: PathBuf,
    host: tokio::sync::Mutex<Option<share::Upstream>>,
    /// Only `kanzo mcp`, which runs as the agent's own process, reads and writes files for it.
    /// Over HTTP anyone who reaches the board could, so there it never does.
    files: bool,
}

const FILES_HERE_ONLY: &str =
    "file and to_file work only through `kanzo mcp`, on the agent's own computer; pass the text as body instead";

const PAGE_FILE_TYPES: [&str; 3] = ["md", "markdown", "txt"];

impl Ctx {
    /// A file the agent named, for a page to be read from or, `new`, saved to. Kept to text
    /// files in the folder the agent works in, where it writes anyway: whatever an agent is told,
    /// it can't carry a key or a .env file into the wiki, where sync and sharing would take it
    /// further, nor save a page over a startup script. Symlinks count where they lead.
    fn page_path(&self, given: &str, new: bool) -> store::Result<PathBuf> {
        let Some(cwd) = self.cwd.as_ref().filter(|_| self.files) else {
            return Err(Error::Invalid(FILES_HERE_ONLY.into()));
        };
        let root = cwd
            .canonicalize()
            .map_err(|e| Error::Invalid(format!("cannot find the working folder {}: {e}", cwd.display())))?;
        let given = given.trim();
        let path = cwd.join(given);
        let outside = || {
            Error::Invalid(format!(
                "{given} is outside {}, the folder this agent works in; Kanzo reads and writes files only there",
                cwd.display()
            ))
        };
        // Judged by its name first, so a path elsewhere gets the same answer whether or not it
        // exists, and the error can't be used to find out what's on the disk.
        if !lexically_within(&path, cwd) {
            return Err(outside());
        }
        let real = if new {
            if path.symlink_metadata().is_ok() {
                return Err(Error::Invalid(format!(
                    "{given} already exists; to_file only writes new files, so pick another name"
                )));
            }
            let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
                return Err(Error::Invalid(format!("{given} is not a file name")));
            };
            let parent = parent.canonicalize().map_err(|e| Error::Invalid(format!("cannot write {given}: {e}")))?;
            parent.join(name)
        } else {
            path.canonicalize().map_err(|e| Error::NotFound(format!("cannot read {given}: {e}")))?
        };
        if !real.starts_with(&root) {
            return Err(outside());
        }
        let text = real.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        if !text.is_some_and(|e| PAGE_FILE_TYPES.contains(&e.as_str())) {
            return Err(Error::Invalid(format!("{given} is not a .md, .markdown or .txt file")));
        }
        Ok(real)
    }

    fn scope(&self, store: &Store, name: Option<&str>) -> store::Result<Option<Project>> {
        if name.is_some_and(|p| p.trim().eq_ignore_ascii_case("all")) {
            return Ok(None);
        }
        self.project(store, name).map(Some)
    }

    fn project(&self, store: &Store, name: Option<&str>) -> store::Result<Project> {
        if let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) {
            return store.find_project(name);
        }
        for repo in &self.repos {
            if let Some(p) = store.project_by_repo(repo)? {
                return Ok(p);
            }
        }
        let known: Vec<String> = store.list_projects()?.into_iter().map(|p| p.slug).collect();
        let known = known.join(", ");
        let msg = match (self.cwd.as_ref().map(|c| c.display()), self.repos.first(), known.is_empty()) {
            (_, None, true) => NO_PROJECTS.to_string(),
            (None, ..) => {
                format!("Pass `project` (one of: {known}); over HTTP Kanzo cannot see which folder you work in.")
            }
            (Some(cwd), Some(repo), false) => format!(
                "No project is linked to {repo} (the git remote of {cwd}). Pass `project` (one of: {known}), \
                 link this repo to one with link_repo, or create a project for it with create_project and repo \"{repo}\"."
            ),
            (Some(cwd), Some(repo), true) => format!(
                "No project is linked to {repo} (the git remote of {cwd}), and there are no projects yet. \
                 Create one with create_project and repo \"{repo}\"."
            ),
            (Some(cwd), None, false) => {
                format!("Cannot tell the project: {cwd} has no git remote. Pass `project` (one of: {known}).")
            }
        };
        Err(Error::NotFound(msg))
    }
}

/// Whether `path`, with its `..` and `.` worked out but symlinks not followed, is in `dir`.
fn lexically_within(path: &std::path::Path, dir: &std::path::Path) -> bool {
    use std::path::Component;
    let mut parts: Vec<Component> = Vec::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(parts.last(), Some(Component::Normal(_))) {
                    parts.pop();
                } else {
                    return false;
                }
            }
            other => parts.push(other),
        }
    }
    parts.iter().collect::<PathBuf>().starts_with(dir)
}

/// A page's text from a file: UTF-8, with Windows line ends made plain.
fn page_file(path: &std::path::Path) -> store::Result<String> {
    let cannot = |e: std::io::Error| Error::NotFound(format!("cannot read {}: {e}", path.display()));
    let size = std::fs::metadata(path).map_err(cannot)?.len();
    if size > store::PAGE_MAX as u64 {
        return Err(Error::Invalid(format!(
            "{} is {} KB; a wiki page holds up to {} KB",
            path.display(),
            size / 1024,
            store::PAGE_MAX / 1024
        )));
    }
    let text = String::from_utf8(std::fs::read(path).map_err(cannot)?)
        .map_err(|_| Error::Invalid(format!("{} is not UTF-8 text", path.display())))?;
    Ok(text.strip_prefix('\u{feff}').unwrap_or(&text).replace("\r\n", "\n"))
}

fn reply(outcome: Result<String, String>) -> CallToolResult {
    match outcome {
        Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
        Err(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
    }
}

/// Clients on protocol 2026-07-28 name themselves on every request; older ones did once, in `initialize`.
fn caller(context: &RequestContext<RoleServer>) -> Option<Implementation> {
    match context.meta.protocol_version() {
        Some(_) => context.meta.client_info(),
        None => context.peer.peer_info().map(|p| p.client_info.clone()),
    }
}

fn git_repos(dir: &std::path::Path) -> Vec<String> {
    let Ok(text) = crate::home::output(Command::new("git").arg("-C").arg(dir).args(["remote", "-v"])) else {
        return Vec::new();
    };
    // `origin` first, so it wins when several remotes are linked to different projects.
    let mut lines: Vec<&str> = text.lines().collect();
    lines.sort_by_key(|l| !l.starts_with("origin\t"));
    let mut repos = Vec::new();
    for line in lines {
        if let Some(repo) = line.split_whitespace().nth(1).and_then(store::repo_from_url)
            && !repos.contains(&repo)
        {
            repos.push(repo);
        }
    }
    repos
}

#[derive(Clone)]
pub struct Kanzo {
    ctx: Arc<Ctx>,
    tool_router: ToolRouter<Self>,
}

/// Agents are told it is a string, which keeps the tool schemas short; a bare number is accepted too.
#[derive(Deserialize)]
#[serde(untagged)]
enum TaskKey {
    Number(i64),
    Text(String),
}

impl TaskKey {
    fn find(&self, store: &Store) -> store::Result<Task> {
        match self {
            TaskKey::Number(n) => store.get_task(*n),
            TaskKey::Text(s) => store.find_task(s),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct CreateProjectArgs {
    /// Short lowercase id, like api.
    slug: String,
    name: Option<String>,
    /// GitHub owner/repo.
    repo: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ListTasksArgs {
    /// Slug, or "all".
    project: Option<String>,
    /// One column, or "all". Done is left out by default.
    status: Option<String>,
    /// Words, or tag: status: project: title: description: url: comment: id: comments: images: created: updated:
    /// is:human is:waiting has: no:, like tag:bug -status:done. Includes done.
    filter: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct TaskArgs {
    /// Like api#12.
    #[schemars(with = "String")]
    id: TaskKey,
}

#[derive(Deserialize, JsonSchema)]
struct CreateTaskArgs {
    title: String,
    /// Markdown.
    description: Option<String>,
    /// Slug.
    project: Option<String>,
    /// Comma-separated.
    tags: Option<String>,
    /// Column; defaults to todo.
    status: Option<String>,
    /// For the user, not for agents.
    human: Option<bool>,
    /// Tasks it waits on, like api#12.
    waits_on: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct UpdateTaskArgs {
    /// Like api#12.
    #[schemars(with = "String")]
    id: TaskKey,
    title: Option<String>,
    /// Markdown.
    description: Option<String>,
    /// Column, like review.
    status: Option<String>,
    /// Slug to move it to.
    project: Option<String>,
    /// Comma-separated; "" clears.
    tags: Option<String>,
    /// For the user, not for agents.
    human: Option<bool>,
    /// A link; "" clears.
    url: Option<String>,
    /// Tasks it waits on, like api#12; "" clears.
    waits_on: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct AddCommentArgs {
    /// Like api#12.
    #[schemars(with = "String")]
    id: TaskKey,
    /// Markdown.
    body: String,
    /// Defaults to your client's name.
    author: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct LinkRepoArgs {
    /// Slug.
    project: String,
    /// GitHub owner/repo.
    repo: String,
}

#[derive(Deserialize, JsonSchema)]
struct WikiArgs {
    /// Slug, or "all".
    project: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct PageArgs {
    title: String,
    /// Slug.
    project: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ReadPageArgs {
    title: String,
    /// Slug.
    project: Option<String>,
    /// First line, from 1.
    offset: Option<usize>,
    /// How many lines.
    limit: Option<usize>,
    /// Save the page to a new .md file in your working folder instead of showing it, to edit it there.
    to_file: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct WritePageArgs {
    title: String,
    /// Markdown; link pages as [[Title]].
    body: Option<String>,
    /// A .md or .txt file in your working folder to take the text from instead of body; much faster for long text.
    file: Option<String>,
    /// Slug.
    project: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct EditPageArgs {
    title: String,
    /// Exact text now in the page.
    old: String,
    new: String,
    /// Replace every occurrence.
    all: Option<bool>,
    /// Slug.
    project: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct GrepWikiArgs {
    /// Regex; ignores case unless it has a capital.
    pattern: String,
    /// Slug, or "all".
    project: Option<String>,
    /// Lines around each match; default 2.
    context: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
struct NextTaskArgs {
    /// Slug, or "all".
    project: Option<String>,
    /// As in list_tasks, like tag:bug.
    filter: Option<String>,
}

impl Kanzo {
    /// The shared board a joined one sends calls to can't see this computer's files: a file
    /// to write a page from is read here and its text sent instead.
    fn files_here(
        &self,
        tool: &str,
        args: &mut Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<(), String> {
        let Some(args) = args else { return Ok(()) };
        if tool == "read_wiki_page" && args.contains_key("to_file") {
            return Err("to_file doesn't work on a shared board; read the page without it".into());
        }
        if tool == "write_wiki_page"
            && let Some(file) = args.remove("file")
        {
            if args.contains_key("body") {
                return Err("pass body or file, not both".into());
            }
            let path = self.ctx.page_path(file.as_str().unwrap_or_default(), false).map_err(|e| e.to_string())?;
            args.insert("body".into(), page_file(&path).map_err(|e| e.to_string())?.into());
        }
        Ok(())
    }

    /// Drops rmcp's `$schema` line and the `null` of optional fields: agents read this list in
    /// every session, so each byte counts.
    fn tools() -> ToolRouter<Self> {
        let mut router = Self::tool_router();
        for route in router.map.values_mut() {
            let mut schema = (*route.attr.input_schema).clone();
            schema.remove("$schema");
            // Optional fields are typed `["string", "null"]`; leaving them out of
            // `required` already says they're optional.
            if let Some(serde_json::Value::Object(props)) = schema.get_mut("properties") {
                for prop in props.values_mut() {
                    if let Some(kind) = prop.get_mut("type")
                        && let Some(first) = kind.as_array().and_then(|k| k.first()).cloned()
                    {
                        *kind = first;
                    }
                }
            }
            route.attr.input_schema = Arc::new(schema);
        }
        router
    }
}

#[tool_router]
impl Kanzo {
    async fn run(&self, f: impl FnOnce(&Ctx, &Store) -> store::Result<String> + Send + 'static) -> CallToolResult {
        self.run_for(&Extensions::default(), f).await
    }

    async fn run_for(
        &self,
        ext: &Extensions,
        f: impl FnOnce(&Ctx, &Store) -> store::Result<String> + Send + 'static,
    ) -> CallToolResult {
        let header = |name| {
            let value = ext.get::<Parts>()?.headers.get(name)?;
            Some(String::from_utf8_lossy(value.as_bytes()).into_owned())
        };
        let ctx = match header(share::FOLDER_HEADER) {
            Some(folder) => Arc::new(Ctx {
                home: self.ctx.home.clone(),
                cwd: Some(folder.into()),
                repos: header(share::REPOS_HEADER)
                    .unwrap_or_default()
                    .split(',')
                    .filter(|r| !r.is_empty())
                    .map(String::from)
                    .collect(),
                images: header(share::IMAGES_HEADER).map_or_else(|| self.ctx.images.clone(), PathBuf::from),
                host: Default::default(),
                files: false,
            }),
            None => self.ctx.clone(),
        };
        let outcome = tokio::task::spawn_blocking(move || {
            let store = ctx.home.store().map_err(|e| e.to_string())?;
            f(&ctx, &store).map_err(|e| e.to_string())
        })
        .await
        .unwrap_or_else(|e| Err(format!("internal error: {e}")));
        reply(outcome)
    }

    #[tool(description = "List projects with their repos and open task counts.")]
    async fn list_projects(&self, ext: Extensions) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, |ctx, store| {
                let projects = store.list_projects()?;
                if projects.is_empty() {
                    return Ok(NO_PROJECTS.into());
                }
                let here = projects.iter().find(|p| p.repos.iter().any(|r| ctx.repos.contains(r)));
                let mut out = String::new();
                for p in &projects {
                    out.push_str(&p.slug);
                    if p.name != p.slug {
                        let _ = write!(out, " \"{}\"", p.name);
                    }
                    if !p.repos.is_empty() {
                        let _ = write!(out, " ({})", p.repos.join(", "));
                    }
                    let _ = write!(out, ": {} open of {}", p.open, p.total);
                    match p.pages {
                        0 => {}
                        1 => out.push_str(", 1 wiki page"),
                        n => {
                            let _ = write!(out, ", {n} wiki pages");
                        }
                    }
                    if here.is_some_and(|h| h.id == p.id) {
                        out.push_str(" <- current repo");
                    }
                    out.push('\n');
                }
                out.pop();
                Ok(out)
            })
            .await)
    }

    #[tool(description = "Create a project.")]
    async fn create_project(
        &self,
        Parameters(args): Parameters<CreateProjectArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run(move |_, store| {
                let p = store.create_project(&args.slug, args.name.as_deref().unwrap_or(""), args.repo.as_deref())?;
                let repo = p.repos.first().map(|r| format!(", linked to {r}")).unwrap_or_default();
                Ok(format!("Created project {} ({}){repo}.", p.slug, p.name))
            })
            .await)
    }

    #[tool(description = "Link another GitHub repo to a project.")]
    async fn link_repo(&self, Parameters(args): Parameters<LinkRepoArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run(move |_, store| {
                let p = store.link_repo(&args.project, &args.repo)?;
                Ok(format!("Project {} is linked to {}.", p.slug, p.repos.join(", ")))
            })
            .await)
    }

    #[tool(description = "List tasks in board order.")]
    async fn list_tasks(
        &self,
        Parameters(args): Parameters<ListTasksArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let project = ctx.scope(store, args.project.as_deref())?;
                let status = args.status.map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty());
                let only = match status.as_deref() {
                    Some("all") | None => None,
                    Some(s) => {
                        store.check_status(s)?;
                        Some(s)
                    }
                };
                let filter = args.filter.filter(|f| !f.trim().is_empty());
                let keep = match (&filter, &status) {
                    (Some(_), _) => None,
                    (None, None) => Some(0),
                    (None, Some(_)) => Some(DONE_LISTED),
                };
                let (mut tasks, hidden) = store.list_tasks_keeping_done(project.as_ref().map(|p| p.id), only, keep)?;
                if let Some(filter) = &filter {
                    let found: HashSet<i64> = store.search_for_agent(filter)?.into_iter().map(|t| t.id).collect();
                    tasks.retain(|t| found.contains(&t.id));
                }
                let done = store::DONE;
                let properties = store.hidden_properties()?;
                let mut out = String::new();
                let mut people = 0;
                for t in &tasks {
                    if t.human {
                        people += 1;
                    } else {
                        task_line(&mut out, t, &properties);
                        out.push('\n');
                    }
                }
                if out.is_empty() {
                    let scope = project.as_ref().map_or("any project", |p| p.slug.as_str());
                    let _ = writeln!(out, "No tasks in {scope}.");
                }
                if people > 0 && shown(&properties, "human") {
                    let _ = writeln!(
                        out,
                        "({people} human {} not shown: for the user, not for agents)",
                        store::tasks(people)
                    );
                }
                if hidden > 0 && status.is_none() {
                    let _ = writeln!(out, "({hidden} {done} not shown; pass status \"{done}\" or \"all\")");
                } else if hidden > 0 {
                    let _ = writeln!(out, "({hidden} older {done} not shown)");
                }
                out.pop();
                Ok(out)
            })
            .await)
    }

    #[tool(description = "Show a task with its comments and image paths.")]
    async fn get_task(
        &self,
        Parameters(args): Parameters<TaskArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let task = args.id.find(store)?;
                task_detail(ctx, store, &task)
            })
            .await)
    }

    #[tool(description = "Add a task, to todo unless you pass status.")]
    async fn create_task(
        &self,
        Parameters(args): Parameters<CreateTaskArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let CreateTaskArgs { title, description, project, tags, status, human, waits_on } = args;
                let project = ctx.project(store, project.as_deref())?;
                let status = status.map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty());
                let patch = TaskPatch { description, status, tags, human, waits_on, ..Default::default() };
                let t = store.create_task_in(project.id, &title, patch)?;
                Ok(format!("Created {} in {}: {}{}", t.key(), t.status, t.title, waits_note(store, &t)?))
            })
            .await)
    }

    #[tool(description = "Change the fields you pass.")]
    async fn update_task(&self, Parameters(args): Parameters<UpdateTaskArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run(move |_, store| {
                let before = args.id.find(store)?;
                let status = args.status.map(|s| s.trim().to_ascii_lowercase());
                let project = args.project.as_deref().map(str::trim).filter(|p| !p.is_empty());
                let project_id = project.map(|p| store.find_project(p)).transpose()?.map(|p| p.id);
                let after = store.update_task(
                    before.id,
                    TaskPatch {
                        title: args.title,
                        description: args.description,
                        status,
                        tags: args.tags,
                        project_id,
                        human: args.human,
                        url: args.url,
                        waits_on: args.waits_on,
                    },
                )?;
                let was = if before.project != after.project { format!("{} -> ", before.key()) } else { String::new() };
                let moved = if before.status != after.status {
                    format!(", {} -> {}", before.status, after.status)
                } else {
                    String::new()
                };
                Ok(format!("Updated {was}{}{moved}: {}{}", after.key(), after.title, waits_note(store, &after)?))
            })
            .await)
    }

    #[tool(description = "Comment on a task, like to report progress.")]
    async fn add_comment(
        &self,
        Parameters(args): Parameters<AddCommentArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let client = caller(&context).map(|i| i.name).filter(|n| !n.is_empty());
        let author = args.author.filter(|a| !a.trim().is_empty()).or(client).unwrap_or_else(|| "agent".into());
        Ok(self
            .run(move |_, store| {
                let task = args.id.find(store)?;
                store.add_comment(task.id, &author, &args.body, &[])?;
                Ok(format!("Comment added to {} as {author}.", task.key()))
            })
            .await)
    }

    #[tool(description = "Take the top todo task, move it to doing and show it.")]
    async fn next_task(
        &self,
        Parameters(args): Parameters<NextTaskArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let project = ctx.scope(store, args.project.as_deref())?;
                let id = project.as_ref().map(|p| p.id);
                let filter = args.filter.as_deref().unwrap_or("");
                match store.next_task_matching(id, filter)? {
                    Some(task) => Ok(format!("Moved to {}:\n{}", task.status, task_detail(ctx, store, &task)?)),
                    None => {
                        let waiting = store.waiting_in_todo(id, filter)?;
                        let scope = project.map_or("any project".into(), |p| p.slug);
                        Ok(match waiting {
                            0 => format!("Nothing in todo for {scope}."),
                            1 => format!("Nothing ready in todo for {scope}: 1 task waits on others."),
                            n => format!("Nothing ready in todo for {scope}: {n} tasks wait on others."),
                        })
                    }
                }
            })
            .await)
    }

    #[tool(description = "Delete a task with its comments and images. Can't be undone.")]
    async fn delete_task(&self, Parameters(args): Parameters<TaskArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run(move |ctx, store| {
                let task = args.id.find(store)?;
                let properties = store.hidden_properties()?;
                ctx.home.deleting_images(store, store::ImagesOf::Task(task.id), || store.delete_task(task.id))?;
                let count = if shown(&properties, "comments") { task.comments } else { 0 };
                let comments = match count {
                    0 => String::new(),
                    1 => " and its comment".into(),
                    n => format!(" and its {n} comments"),
                };
                Ok(format!("Deleted {}{comments}: {}", task.key(), task.title))
            })
            .await)
    }

    #[tool(description = "List wiki pages with their line counts.")]
    async fn list_wiki_pages(
        &self,
        Parameters(args): Parameters<WikiArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let project = ctx.scope(store, args.project.as_deref())?;
                let pages = store.list_pages(project.as_ref().map(|p| p.id), "")?;
                if pages.is_empty() {
                    let scope = project.map_or("any project".into(), |p| p.slug);
                    return Ok(format!("No wiki pages in {scope}. Add one with write_wiki_page."));
                }
                let mut out = String::new();
                for page in &pages {
                    if project.is_none() {
                        let _ = write!(out, "{}/", page.project);
                    }
                    let lines = if page.lines == 1 { "line" } else { "lines" };
                    let _ = writeln!(
                        out,
                        "{} ({} {lines}, updated {})",
                        page.title,
                        page.lines,
                        short_time(&page.updated_at)
                    );
                }
                out.pop();
                Ok(out)
            })
            .await)
    }

    #[tool(description = "Read a wiki page, or some of its lines, like those grep_wiki found, or save it to a file.")]
    async fn read_wiki_page(
        &self,
        Parameters(args): Parameters<ReadPageArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let project = ctx.project(store, args.project.as_deref())?;
                let page = store.find_page(project.id, &args.title)?;
                if let Some(to) = &args.to_file {
                    use std::io::Write as _;
                    let path = ctx.page_path(to, true)?;
                    let cannot = |e: std::io::Error| Error::Invalid(format!("cannot write {to}: {e}"));
                    // create_new: never over a file that appeared since the check.
                    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(cannot)?;
                    file.write_all(page.body.as_bytes()).map_err(cannot)?;
                    return Ok(format!(
                        "Saved wiki page {} ({} lines) to {}. To save changes made there, call write_wiki_page with file.",
                        page.title,
                        page.body.lines().count(),
                        to.trim()
                    ));
                }
                let lines: Vec<&str> = page.body.lines().collect();
                let first = args.offset.unwrap_or(1).max(1);
                let count = args.limit.unwrap_or(lines.len());
                let shown: Vec<&str> = lines.iter().skip(first - 1).take(count).copied().collect();
                let by = if page.author.is_empty() { String::new() } else { format!(" by {}", page.author) };
                let mut out =
                    format!("{} ({}), updated {}{by}", page.title, page.project, short_time(&page.updated_at));
                if lines.is_empty() {
                    out.push_str("\n\n(empty)");
                } else if shown.is_empty() {
                    let _ = write!(out, "\n\n(it has {} lines)", lines.len());
                } else {
                    if shown.len() < lines.len() {
                        let _ = write!(out, ", lines {}-{} of {}", first, first + shown.len() - 1, lines.len());
                    }
                    let _ = write!(out, "\n\n{}", shown.join("\n"));
                }
                Ok(out)
            })
            .await)
    }

    #[tool(
        description = "Create a wiki page, or replace all its text. For long text, write it to a file and pass file."
    )]
    async fn write_wiki_page(
        &self,
        Parameters(args): Parameters<WritePageArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let author = caller(&context).map(|i| i.name).filter(|n| !n.is_empty()).unwrap_or_else(|| "agent".into());
        Ok(self
            .run_for(&context.extensions, move |ctx, store| {
                let project = ctx.project(store, args.project.as_deref())?;
                let body = match (args.body, args.file) {
                    (Some(_), Some(_)) => return Err(Error::Invalid("pass body or file, not both".into())),
                    (Some(body), None) => body,
                    (None, Some(file)) => page_file(&ctx.page_path(&file, false)?)?,
                    (None, None) => return Err(Error::Invalid("pass the text as body, or as a local file".into())),
                };
                let (page, created) = store.write_page(project.id, &args.title, &body, &author)?;
                let did = if created { "Created" } else { "Saved" };
                Ok(format!("{did} wiki page {} in {} ({} lines).", page.title, page.project, page.body.lines().count()))
            })
            .await)
    }

    #[tool(description = "Replace exact text in a wiki page, without sending the whole page.")]
    async fn edit_wiki_page(
        &self,
        Parameters(args): Parameters<EditPageArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let author = caller(&context).map(|i| i.name).filter(|n| !n.is_empty()).unwrap_or_else(|| "agent".into());
        Ok(self
            .run_for(&context.extensions, move |ctx, store| {
                let project = ctx.project(store, args.project.as_deref())?;
                let all = args.all.unwrap_or(false);
                let (page, n) = store.edit_page(project.id, &args.title, &args.old, &args.new, all, &author)?;
                let places = if n == 1 { String::new() } else { format!(" in {n} places") };
                Ok(format!("Edited wiki page {}{places}.", page.title))
            })
            .await)
    }

    #[tool(description = "Search wiki pages by regex: matching lines with their numbers and a little context.")]
    async fn grep_wiki(
        &self,
        Parameters(args): Parameters<GrepWikiArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let project = ctx.scope(store, args.project.as_deref())?;
                let context = args.context.unwrap_or(2).min(20);
                let grep = store.grep_pages(project.as_ref().map(|p| p.id), &args.pattern, context, GREP_MATCHES)?;
                let scope = project.as_ref().map_or("any project", |p| p.slug.as_str());
                if grep.pages.is_empty() {
                    return Ok(format!("No wiki matches for {:?} in {scope}.", args.pattern));
                }
                Ok(grep_text(&grep, project.is_none()))
            })
            .await)
    }

    #[tool(description = "Delete a wiki page. Can't be undone.")]
    async fn delete_wiki_page(
        &self,
        Parameters(args): Parameters<PageArgs>,
        ext: Extensions,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(self
            .run_for(&ext, move |ctx, store| {
                let project = ctx.project(store, args.project.as_deref())?;
                let page = store.find_page(project.id, &args.title)?;
                store.delete_page(page.id)?;
                Ok(format!("Deleted wiki page {} from {}.", page.title, page.project))
            })
            .await)
    }
}

// Serve the router built once in `tools()`; by default the macro builds a new,
// untrimmed one for every request.
#[tool_handler(router = self.tool_router)]
impl ServerHandler for Kanzo {
    /// Checked on every call, so agents already running switch boards when this computer joins or leaves.
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        // Over HTTP, the board itself sends calls on when it joined another.
        if let Some(cwd) = &self.ctx.cwd
            && let Net::Joined(host) = Net::load(&self.ctx.home)
        {
            let mut upstream = self.ctx.host.lock().await;
            if upstream.as_ref().is_none_or(|u| u.host() != host) {
                *upstream = Some(share::Upstream::new(host, cwd, &self.ctx.repos, &self.ctx.images));
            }
            let client = share::handshake(context.peer.peer_info().as_deref(), &context.meta);
            let upstream = upstream.as_mut().expect("set above");
            let mut arguments = request.arguments;
            if let Err(e) = self.files_here(&request.name, &mut arguments) {
                return Ok(reply(Err(e)).into());
            }
            let outcome = upstream.call(client, &request.name, arguments).await;
            return Ok(reply(outcome).into());
        }
        self.tool_router.call(ToolCallContext::new(self, request, context)).await
    }

    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_tool_list_changed().build())
            .with_server_info(Implementation::new("kanzo", env!("CARGO_PKG_VERSION")))
            .with_instructions(if self.ctx.cwd.is_some() { INSTRUCTIONS } else { HTTP_INSTRUCTIONS })
    }
}

/// Like grep's output: `title:12:text` for a match, `title-13-text` around one, `--` between groups.
fn grep_text(grep: &Grep, all: bool) -> String {
    let mut out = String::new();
    for page in &grep.pages {
        let name = if all { format!("{}/{}", page.project, page.title) } else { page.title.clone() };
        if !out.is_empty() {
            out.push_str("--\n");
        }
        if page.title_matches && page.lines.is_empty() {
            let _ = writeln!(out, "{name}: (the title matches)");
        }
        let mut last = None;
        for line in &page.lines {
            if last.is_some_and(|n| n + 1 != line.number) {
                out.push_str("--\n");
            }
            let sep = if line.matched { ':' } else { '-' };
            let _ = writeln!(out, "{name}{sep}{}{sep}{}", line.number, line.text);
            last = Some(line.number);
        }
    }
    if grep.more {
        let _ = writeln!(out, "(stopped at {} matches; narrow the pattern)", grep.matches);
    }
    out.pop();
    out
}

fn shown(hidden: &[String], property: &str) -> bool {
    !hidden.iter().any(|p| p == property)
}

fn task_line(out: &mut String, t: &Task, hidden: &[String]) {
    let _ = write!(out, "{} [{}] {}", t.key(), t.status, t.title);
    if shown(hidden, "tags") {
        write_tags(out, t);
    }
    if shown(hidden, "waiting") {
        write_waits(out, t);
    }
    match if shown(hidden, "comments") { t.comments } else { 0 } {
        0 => {}
        1 => out.push_str(" (1 comment)"),
        n => {
            let _ = write!(out, " ({n} comments)");
        }
    }
}

fn write_tags(out: &mut String, t: &Task) {
    if !t.tags.is_empty() {
        let _ = write!(out, " {{{}}}", t.tags.join(", "));
    }
}

fn write_waits(out: &mut String, t: &Task) {
    if !t.waiting_on.is_empty() {
        let ids: Vec<String> = t.waiting_on.iter().map(|id| format!("#{id}")).collect();
        let _ = write!(out, " (waits on {})", ids.join(", "));
    }
}

/// Tells an agent that moved or changed a task that it's still waiting; moving one anyway is allowed.
fn waits_note(store: &Store, t: &Task) -> store::Result<String> {
    let mut note = String::new();
    if !t.waiting_on.is_empty() && shown(&store.hidden_properties()?, "waiting") {
        write_waits(&mut note, t);
    }
    Ok(note)
}

fn task_detail(ctx: &Ctx, store: &Store, t: &Task) -> store::Result<String> {
    let hidden = store.hidden_properties()?;
    let comments = if shown(&hidden, "comments") { store.list_comments(t.id)? } else { Vec::new() };
    let images = if shown(&hidden, "images") { store.list_images(t.id)? } else { Vec::new() };
    let dir = &ctx.images;
    let paths = |comment: Option<i64>| -> Vec<String> {
        images.iter().filter(|i| i.comment_id == comment).map(|i| dir.join(&i.file).display().to_string()).collect()
    };
    let (created, updated) = (short_time(&t.created_at), short_time(&t.updated_at));
    let mut out = format!("{} {}", t.key(), t.title);
    if shown(&hidden, "tags") {
        write_tags(&mut out, t);
    }
    let _ = write!(out, "\nstatus: {}", t.status);
    if shown(&hidden, "waiting") {
        write_waits(&mut out, t);
    }
    if t.human && shown(&hidden, "human") {
        out.push_str(", human: for the user, not for agents");
    }
    if shown(&hidden, "created") {
        let _ = write!(out, ", created {created}");
    }
    if updated != created && shown(&hidden, "updated") {
        let _ = write!(out, ", updated {updated}");
    }
    if !t.url.is_empty() && shown(&hidden, "url") {
        let _ = write!(out, "\nurl: {}", t.url);
    }
    out.push('\n');
    if !t.description.is_empty() && shown(&hidden, "description") {
        let _ = writeln!(out, "\n{}", t.description);
    }
    let own = paths(None);
    if !own.is_empty() {
        let _ = writeln!(out, "\nImages: {}", own.join(", "));
    }
    if !comments.is_empty() {
        let _ = writeln!(out, "\nComments ({}):", comments.len());
        for c in &comments {
            let _ = write!(out, "- {}, {}: {}", c.author, short_time(&c.created_at), c.body);
            let attached = paths(Some(c.id));
            if !attached.is_empty() {
                let sep = if c.body.is_empty() { "" } else { " " };
                let _ = write!(out, "{sep}[images: {}]", attached.join(", "));
            }
            out.push('\n');
        }
    }
    out.truncate(out.trim_end().len());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use tokio::io::{
        AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines, ReadHalf, WriteHalf,
    };

    use rmcp::ServiceExt;

    use super::*;

    const STATELESS: &str = "2026-07-28";

    /// The client side of a `kanzo mcp` on stdio.
    struct Agent {
        lines: Lines<BufReader<ReadHalf<DuplexStream>>>,
        write: WriteHalf<DuplexStream>,
    }

    impl Agent {
        fn start(home: Home, folder: &std::path::Path) -> Self {
            let images = home.images();
            let ctx = Ctx {
                home,
                cwd: Some(folder.into()),
                repos: Vec::new(),
                images,
                host: Default::default(),
                files: true,
            };
            let server = Kanzo { ctx: Arc::new(ctx), tool_router: Kanzo::tools() };
            let (ours, theirs) = tokio::io::duplex(1 << 16);
            tokio::spawn(async move { server.serve(tokio::io::split(theirs)).await.unwrap().waiting().await });
            let (read, write) = tokio::io::split(ours);
            Self { lines: BufReader::new(read).lines(), write }
        }

        /// The reply to `message`, or `None` for a notification.
        async fn send(&mut self, message: Value) -> Option<Value> {
            self.write.write_all(format!("{message}\n").as_bytes()).await.unwrap();
            let id = message.get("id")?;
            loop {
                let reply: Value = serde_json::from_str(&self.lines.next_line().await.unwrap().unwrap()).unwrap();
                if &reply["id"] == id {
                    return Some(reply);
                }
            }
        }
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
    }

    fn board() -> (tempfile::TempDir, Home, Task) {
        let dir = tempfile::tempdir().unwrap();
        let home = Home::open(dir.path().to_path_buf()).unwrap();
        let store = home.store().unwrap();
        let project = store.create_project("api", "", None).unwrap();
        let task = store.create_task(project.id, "Ship it", "").unwrap();
        (dir, home, task)
    }

    /// Serves `home`'s MCP over HTTP on a free port, as a shared board does.
    async fn shared(home: Home) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = axum::Router::new().nest_service("/mcp", http_service(home, CancellationToken::new()));
        tokio::spawn(async move { axum::serve(listener, app).await });
        port
    }

    fn authors(home: &Home, task: &Task) -> Vec<String> {
        home.store().unwrap().list_comments(task.id).unwrap().into_iter().map(|c| c.author).collect()
    }

    fn comment(task: &Task, meta: Value) -> Value {
        let args = json!({ "id": task.key(), "body": "done" });
        json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": "add_comment", "arguments": args, "_meta": meta } })
    }

    fn stateless(client: Option<&str>) -> Value {
        let mut meta = json!({
            "io.modelcontextprotocol/protocolVersion": STATELESS,
            "io.modelcontextprotocol/clientCapabilities": {},
        });
        if let Some(name) = client {
            meta["io.modelcontextprotocol/clientInfo"] = json!({ "name": name, "version": "1" });
        }
        meta
    }

    #[test]
    fn agents_see_what_a_task_waits_on_and_why_nothing_is_ready() {
        block_on(async {
            let (dir, home, blocker) = board();
            let mut agent = Agent::start(home.clone(), dir.path());
            let mut call = async |name: &str, args: Value| -> String {
                let params = json!({ "name": name, "arguments": args, "_meta": stateless(None) });
                let message = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params });
                let reply = agent.send(message).await.unwrap();
                reply["result"]["content"][0]["text"].as_str().unwrap().to_string()
            };
            let waiting =
                call("create_task", json!({ "title": "Then this", "project": "api", "waits_on": blocker.key() })).await;
            assert_eq!(waiting, "Created api#2 in todo: Then this (waits on #1)");
            assert!(call("list_tasks", json!({ "project": "api" })).await.contains("Then this (waits on #1)"));
            assert!(call("get_task", json!({ "id": "api#2" })).await.contains("status: todo (waits on #1)"));

            call("update_task", json!({ "id": "api#1", "status": "doing" })).await;
            let next = call("next_task", json!({ "project": "api" })).await;
            assert_eq!(next, "Nothing ready in todo for api: 1 task waits on others.");
            let moved = call("update_task", json!({ "id": "api#2", "status": "doing" })).await;
            assert_eq!(moved, "Updated api#2, todo -> doing: Then this (waits on #1)");
            let next = call("next_task", json!({ "project": "api" })).await;
            assert_eq!(next, "Nothing in todo for api.");

            home.store().unwrap().set_property("waiting", false).unwrap();
            assert!(!call("get_task", json!({ "id": "api#2" })).await.contains("waits on"));
        });
    }

    #[test]
    fn agents_grep_the_wiki_then_read_and_edit_only_what_they_need() {
        block_on(async {
            let (dir, home, _) = board();
            let mut agent = Agent::start(home.clone(), dir.path());
            let mut call = async |name: &str, args: Value| -> String {
                let params = json!({ "name": name, "arguments": args, "_meta": stateless(Some("claude-code")) });
                let message = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params });
                let reply = agent.send(message).await.unwrap();
                reply["result"]["content"][0]["text"].as_str().unwrap().to_string()
            };
            let body = "# Setup\n\nInstall Rust.\nRun `cargo build`.\n\n# Deploy\n\nTag a release.\nThe CI builds it.";
            let made = call("write_wiki_page", json!({ "title": "Runbook", "body": body, "project": "api" })).await;
            assert_eq!(made, "Created wiki page Runbook in api (9 lines).");
            call("write_wiki_page", json!({ "title": "Build notes", "body": "cargo is slow", "project": "api" })).await;

            let grep = call("grep_wiki", json!({ "pattern": "build", "project": "api", "context": 1 })).await;
            assert_eq!(
                grep,
                "Build notes: (the title matches)\n--\nRunbook-3-Install Rust.\nRunbook:4:Run `cargo build`.\nRunbook-5-\n--\nRunbook-8-Tag a release.\nRunbook:9:The CI builds it."
            );
            let everywhere = call("grep_wiki", json!({ "pattern": "Tag", "project": "all", "context": 0 })).await;
            assert_eq!(everywhere, "api/Runbook:8:Tag a release.");

            let read =
                call("read_wiki_page", json!({ "title": "runbook", "project": "api", "offset": 8, "limit": 2 })).await;
            assert!(read.starts_with("Runbook (api), updated "), "{read}");
            assert!(read.contains("by claude-code, lines 8-9 of 9\n\nTag a release.\nThe CI builds it."), "{read}");

            let edited = call(
                "edit_wiki_page",
                json!({ "title": "Runbook", "old": "Tag a release.", "new": "Run `kanzo update`.", "project": "api" }),
            )
            .await;
            assert_eq!(edited, "Edited wiki page Runbook.");
            let page = home.store().unwrap().find_page(1, "Runbook").unwrap();
            assert!(page.body.contains("Run `kanzo update`.\nThe CI"), "{}", page.body);

            let listed = call("list_wiki_pages", json!({ "project": "api" })).await;
            assert!(
                listed.starts_with("Build notes (1 line, updated ") && listed.contains("\nRunbook (9 lines, "),
                "{listed}"
            );
            assert!(call("list_projects", json!({})).await.contains("api: 1 open of 1, 2 wiki pages"));
            let gone = call("delete_wiki_page", json!({ "title": "Build notes", "project": "api" })).await;
            assert_eq!(gone, "Deleted wiki page Build notes from api.");
            let missing = call("read_wiki_page", json!({ "title": "Build notes", "project": "api" })).await;
            assert_eq!(missing, "no wiki page \"Build notes\" (pages: Runbook)");
        });
    }

    #[test]
    fn agents_write_long_pages_from_files_and_save_them_to_files() {
        block_on(async {
            let (dir, home, _) = board();
            let mut agent = Agent::start(home.clone(), dir.path());
            let mut call = async |name: &str, args: Value| -> (bool, String) {
                let params = json!({ "name": name, "arguments": args, "_meta": stateless(None) });
                let message = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params });
                let result = agent.send(message).await.unwrap()["result"].clone();
                (
                    result["isError"].as_bool().unwrap_or(false),
                    result["content"][0]["text"].as_str().unwrap().to_string(),
                )
            };
            let long: String = (1..=2000).map(|i| format!("Line {i} of a long page.\r\n")).collect();
            std::fs::write(dir.path().join("notes.md"), &long).unwrap();
            let (failed, made) =
                call("write_wiki_page", json!({ "title": "Notes", "file": "notes.md", "project": "api" })).await;
            assert!(!failed, "{made}");
            assert_eq!(made, "Created wiki page Notes in api (2000 lines).");
            let page = home.store().unwrap().find_page(1, "Notes").unwrap();
            assert_eq!(page.body, long.replace("\r\n", "\n"), "Windows line ends are made plain");

            let to = json!({ "title": "Notes", "project": "api", "to_file": "copy.md" });
            let (failed, saved) = call("read_wiki_page", to).await;
            assert!(!failed && saved.starts_with("Saved wiki page Notes (2000 lines) to "), "{saved}");
            assert_eq!(std::fs::read_to_string(dir.path().join("copy.md")).unwrap(), page.body);

            let both = json!({ "title": "Notes", "body": "x", "file": "notes.md", "project": "api" });
            let (failed, text) = call("write_wiki_page", both).await;
            assert!(failed && text.contains("not both"), "{text}");
            let (failed, text) =
                call("write_wiki_page", json!({ "title": "Notes", "file": "nope.md", "project": "api" })).await;
            assert!(failed && text.contains("cannot read"), "{text}");
            let (failed, text) = call("write_wiki_page", json!({ "title": "Notes", "project": "api" })).await;
            assert!(failed && text.contains("pass the text"), "{text}");
        });
    }

    #[test]
    fn the_board_over_http_never_reads_or_writes_files() {
        let (dir, home, _) = board();
        let images = home.images();
        let ctx = Ctx {
            home,
            cwd: Some(dir.path().into()),
            repos: Vec::new(),
            images,
            host: Default::default(),
            files: false,
        };
        let err = ctx.page_path("notes.md", false).unwrap_err().to_string();
        assert!(err.contains("only through `kanzo mcp`"), "{err}");
    }

    #[test]
    fn page_files_stay_inside_the_agents_folder_and_are_text() {
        let (dir, home, _) = board();
        let images = home.images();
        let work = dir.path().join("repo");
        std::fs::create_dir_all(work.join("docs")).unwrap();
        std::fs::write(work.join("docs/notes.md"), "notes").unwrap();
        std::fs::write(work.join(".env"), "SECRET=1").unwrap();
        std::fs::write(dir.path().join("outside.md"), "not yours").unwrap();
        let ctx =
            Ctx { home, cwd: Some(work.clone()), repos: Vec::new(), images, host: Default::default(), files: true };
        let refused = |given: &str, new: bool| ctx.page_path(given, new).unwrap_err().to_string();

        assert!(ctx.page_path("docs/notes.md", false).is_ok());
        assert!(ctx.page_path("docs/new.md", true).is_ok());
        assert!(refused("../outside.md", false).contains("outside"));
        let absolute = dir.path().join("outside.md").display().to_string();
        assert!(refused(&absolute, false).contains("outside"));
        assert!(refused(".env", false).contains("not a .md"));
        assert!(refused("docs/notes.md", true).contains("already exists"), "to_file never overwrites");
        assert!(refused("../new.md", true).contains("outside"));
        assert!(refused("missing/new.md", true).contains("cannot write"));
        let nowhere = dir.path().join("no-such-file.md").display().to_string();
        assert!(refused(&nowhere, false).contains("outside"), "a missing file elsewhere is only outside");
        assert!(refused("docs/../../outside.md", false).contains("outside"));
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_counts_where_it_leads() {
        let (dir, home, _) = board();
        let images = home.images();
        let work = dir.path().join("repo");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(dir.path().join("id_rsa"), "key").unwrap();
        std::os::unix::fs::symlink(dir.path().join("id_rsa"), work.join("key.md")).unwrap();
        std::os::unix::fs::symlink(dir.path(), work.join("up")).unwrap();
        let ctx = Ctx { home, cwd: Some(work), repos: Vec::new(), images, host: Default::default(), files: true };
        assert!(ctx.page_path("key.md", false).unwrap_err().to_string().contains("outside"));
        assert!(ctx.page_path("up/new.md", true).unwrap_err().to_string().contains("outside"));
    }

    #[test]
    fn agents_move_a_task_to_another_project() {
        block_on(async {
            let (dir, home, task) = board();
            home.store().unwrap().create_project("web", "", None).unwrap();
            let mut agent = Agent::start(home.clone(), dir.path());
            let mut call = async |name: &str, args: Value| -> String {
                let params = json!({ "name": name, "arguments": args, "_meta": stateless(None) });
                let message = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params });
                let result = &agent.send(message).await.unwrap()["result"];
                let text = result["content"][0]["text"].as_str().unwrap();
                if result["isError"] == true { format!("Error: {text}") } else { text.to_string() }
            };

            let moved = call("update_task", json!({ "id": task.key(), "project": "web", "status": "doing" })).await;
            assert_eq!(moved, "Updated api#1 -> web#1, todo -> doing: Ship it");
            let saved = home.store().unwrap().get_task(task.id).unwrap();
            assert_eq!((saved.key(), saved.status.as_str()), ("web#1".into(), "doing"));
            assert_eq!(call("get_task", json!({ "id": "api#1" })).await, "Error: no task api#1 (#1 is web#1)");

            // An unknown project refuses the whole update, and says which projects there are.
            let args = json!({ "id": "web#1", "project": "nope", "status": "review", "title": "Renamed" });
            assert_eq!(call("update_task", args).await, r#"Error: no project "nope" (projects: api, web)"#);
            let kept = home.store().unwrap().get_task(task.id).unwrap();
            assert_eq!((kept.key(), kept.status.as_str(), kept.title.as_str()), ("web#1".into(), "doing", "Ship it"));

            // Blank, or the project it's already in, changes nothing and says nothing extra.
            assert_eq!(call("update_task", json!({ "id": "web#1", "project": " " })).await, "Updated web#1: Ship it");
            assert_eq!(call("update_task", json!({ "id": "web#1", "project": "WEB" })).await, "Updated web#1: Ship it");
        });
    }

    #[test]
    fn a_joined_board_takes_calls_from_a_client_that_never_initialized() {
        block_on(async {
            let (_host_dir, host, task) = board();
            let port = shared(host.clone()).await;
            let guest_dir = tempfile::tempdir().unwrap();
            std::fs::write(guest_dir.path().join(share::FILE), format!("join 127.0.0.1:{port}\n")).unwrap();
            let mut agent = Agent::start(Home::open(guest_dir.path().to_path_buf()).unwrap(), guest_dir.path());

            let reply = agent.send(comment(&task, stateless(Some("claude-code")))).await.unwrap();
            assert_eq!(reply["result"]["isError"], false, "{reply}");
            assert_eq!(authors(&host, &task), ["claude-code"]);
        });
    }

    #[test]
    fn comments_are_signed_with_the_name_a_client_initialized_with() {
        block_on(async {
            let (dir, home, task) = board();
            let mut agent = Agent::start(home.clone(), dir.path());
            let client = json!({ "name": "codex-mcp-client", "version": "1" });
            let params = json!({ "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": client });
            agent.send(json!({ "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": params })).await;
            agent.send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).await;

            let meta = json!({ "io.modelcontextprotocol/clientInfo": { "name": "from-meta", "version": "1" } });
            agent.send(comment(&task, meta)).await;
            assert_eq!(authors(&home, &task), ["codex-mcp-client"]);
        });
    }

    #[test]
    fn a_comment_over_http_from_a_client_that_gives_no_name_is_signed_agent() {
        block_on(async {
            let (_dir, home, task) = board();
            let port = shared(home.clone()).await;
            let body = comment(&task, stateless(None)).to_string();
            let mut conn = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            let head = format!(
                "POST /mcp HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
                 Accept: application/json, text/event-stream\r\nMCP-Protocol-Version: {STATELESS}\r\n\
                 Mcp-Method: tools/call\r\nMcp-Name: add_comment\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            conn.write_all(format!("{head}{body}").as_bytes()).await.unwrap();
            let mut reply = String::new();
            conn.read_to_string(&mut reply).await.unwrap();

            assert!(reply.contains("Comment added"), "{reply}");
            assert_eq!(authors(&home, &task), ["agent"]);
        });
    }
}
