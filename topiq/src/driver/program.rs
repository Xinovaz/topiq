//! Loading a whole program: the units given, and every unit they import.
//!
//! A unit is analysed against the interfaces of the units it imports, so
//! those must be known first. Starting from the files given, each unit is
//! parsed and its `import` items read; each unit named is looked for on the
//! **unit path** (the directories of the files given, then any added with
//! `--unit-path`): as `NAME.tq`, which is compiled from source in this run,
//! or else `NAME.tcu` or `NAME.tqu`, a classical or quantum unit compiled
//! earlier, whose interface is read from its metadata. A unit found nowhere
//! is left for analysis to report where it is imported. A classical unit
//! importing a quantum unit also depends on the library's `qpu`, which runs
//! circuits, without naming it; a unit compiled earlier depends on every
//! unit its metadata says it uses.
//!
//! # Units of either kind
//!
//! A `#unit any` unit is classical or quantum as it is asked: by
//! `import(kind)`, or `--kind` for one given, then by the kind it prefers,
//! then by the kind of the unit importing it. Its kind is chosen before it is
//! preprocessed, since its `#alias` and `#if` lines read it. One source asked
//! for both kinds is two units, known in the program as `NAME.classical` and
//! `NAME.quantum`; compiled earlier, they are `NAME.tcu` and `NAME.tqu` side
//! by side, and each import finds the one of its kind. Each unit records the
//! units it imports by those names, so that a unit compiled earlier finds the
//! same ones again. A `--kind` naming a unit of fixed kind is `EU08`.
//!
//! The units compiled from source are then analysed in groups, each after
//! the groups it imports from. A group is one unit, or units that import one
//! another: each of those needs the others' interfaces before its own is
//! known, so they are analysed repeatedly, each against what the others
//! offered the time before, until none offers anything new, and then compiled
//! against the interfaces they settled on.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::{Path, PathBuf};

use crate::diag::{Code, Diagnostic};
use crate::meta::Metadata;
use crate::pp::{KindChoice, UnitKind};
use crate::sema::Interface;

use super::graph::Graph;
use super::{KindArgs, Options, Outcome, Session, Stage, back, dependencies, finish, front};

/// How a unit of the program is known.
#[derive(Debug)]
pub enum MemberKind {
    /// Compiled from source in this run.
    Source(Box<Outcome>),
    /// Compiled earlier: known from its object's metadata alone.
    Compiled {
        /// The object file.
        path: PathBuf,
        /// Its metadata.
        metadata: Box<Metadata>,
    },
}

/// One unit of the program.
#[derive(Debug)]
pub struct Member {
    /// The unit's name.
    pub name: String,
    /// How it is known.
    pub kind: MemberKind,
    /// Whether it was named on the command line, rather than found through
    /// an import.
    pub given: bool,
    /// Library units it depends on without naming them: `qpu`, for a
    /// classical unit importing a quantum one, whose circuits it may run.
    pub implied: Vec<String>,
    /// The units it imports and names, each by the name it goes by in the
    /// program, as [`super::identity`] gives it: a `#unit any` unit's carries
    /// the kind it was resolved to.
    pub deps: Vec<String>,
}

impl Member {
    /// A unit compiled from source in this run, depending on nothing
    /// unnamed yet.
    fn source(name: String, out: Outcome, given: bool) -> Member {
        Member {
            name,
            kind: MemberKind::Source(Box::new(out)),
            given,
            implied: Vec::new(),
            deps: Vec::new(),
        }
    }

    /// The unit's kind: classical or quantum.
    pub fn unit_kind(&self) -> Option<UnitKind> {
        match &self.kind {
            MemberKind::Source(out) => out.unit_kind(),
            MemberKind::Compiled { metadata, .. } => {
                Some(if metadata.interface.quantum { UnitKind::Quantum } else { UnitKind::Classical })
            }
        }
    }

    /// The unit's metadata, if it has any yet: always for a compiled unit, and
    /// for a source unit once its analysis has succeeded.
    pub fn metadata(&self) -> Option<&Metadata> {
        match &self.kind {
            MemberKind::Source(out) => out.metadata.as_ref(),
            MemberKind::Compiled { metadata, .. } => Some(metadata),
        }
    }

    /// The translation.
    pub fn outcome(&self) -> Option<&Outcome> {
        match &self.kind {
            MemberKind::Source(out) => Some(out),
            MemberKind::Compiled { .. } => None,
        }
    }
}

/// A loaded program.
#[derive(Debug)]
pub struct Loaded {
    /// Every source file, for rendering diagnostics.
    pub session: Session,
    /// The units.
    pub members: Vec<Member>,
    /// Problems with the program as a whole, such as units that import each
    /// other.
    pub diagnostics: Vec<Diagnostic>,
}

impl Loaded {
    /// Every diagnostic: the units', then the program's.
    pub fn all_diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.members
            .iter()
            .filter_map(Member::outcome)
            .flat_map(|o| o.diagnostics.iter())
            .chain(&self.diagnostics)
    }

    /// Whether anything reported an error.
    pub fn has_errors(&self) -> bool {
        self.all_diagnostics().any(Diagnostic::is_error)
    }
}

/// Why a program could not be loaded, as opposed to faults in it (those are
/// diagnostics).
#[derive(Debug)]
pub enum LoadError {
    /// A file could not be read.
    Io {
        /// The file.
        path: PathBuf,
        /// What went wrong.
        error: io::Error,
    },
    /// A compiled unit's metadata could not be read.
    Metadata {
        /// The object file.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            LoadError::Metadata { path, reason } => {
                write!(f, "{} cannot be used as a compiled unit: {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Whether a path names a compiled unit rather than a source file: a
/// classical unit's object, `.tcu`, or a quantum unit's circuits, `.tqu`.
fn is_object(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "tcu" || e == "tqu")
}

/// Reads a compiled unit's metadata: the section of a `.tcu` object that
/// holds it, or the whole of a `.tqu`, which is its metadata.
fn read_compiled(path: &Path, session: &mut Session) -> Result<Metadata, LoadError> {
    let bytes = std::fs::read(path).map_err(|error| LoadError::Io {
        path: path.to_owned(),
        error,
    })?;
    let unusable = |reason| LoadError::Metadata {
        path: path.to_owned(),
        reason,
    };
    let text = if path.extension().is_some_and(|e| e == "tqu") {
        String::from_utf8(bytes).map_err(|_| unusable("it is not UTF-8 text".to_owned()))?
    } else {
        crate::meta::object::read_section(&bytes).map_err(unusable)?
    };
    crate::meta::decode::decode(&text, session.interner_mut())
        .map_err(|e| unusable(format!("its metadata is damaged: {e}")))
}

/// Loads the program made of `inputs` (source files and compiled units) and
/// everything they import, then analyses and, as far as `options.stage` says,
/// compiles every unit given as source.
///
/// # Errors
///
/// A [`LoadError`] when a file cannot be read, or a compiled unit's metadata
/// cannot be.
pub fn load(inputs: &[PathBuf], unit_path: &[PathBuf], options: Options) -> Result<Loaded, LoadError> {
    load_with(inputs, unit_path, options, &Overlay::new())
}

/// [`load`], with the kinds `--kind` chooses for the `#unit any` units among
/// `inputs`.
///
/// # Errors
///
/// As for [`load`].
pub fn load_as(inputs: &[PathBuf], unit_path: &[PathBuf], options: Options, kinds: &KindArgs) -> Result<Loaded, LoadError> {
    load_from(inputs, unit_path, options, &Overlay::new(), kinds)
}

/// Source text standing in for files, by path, such as an editor's unsaved
/// buffers. A path it holds need not exist on disk.
pub type Overlay = std::collections::HashMap<PathBuf, String>;

/// [`load`], taking each source file `overlay` holds from there rather than
/// from disk.
///
/// # Errors
///
/// As for [`load`].
pub fn load_with(
    inputs: &[PathBuf],
    unit_path: &[PathBuf],
    options: Options,
    overlay: &Overlay,
) -> Result<Loaded, LoadError> {
    load_from(inputs, unit_path, options, overlay, &KindArgs::default())
}

/// [`load_with`] and [`load_as`] together.
fn load_from(
    inputs: &[PathBuf],
    unit_path: &[PathBuf],
    options: Options,
    overlay: &Overlay,
    kinds: &KindArgs,
) -> Result<Loaded, LoadError> {
    let mut session = Session::new();
    let mut program_diags: Vec<Diagnostic> = Vec::new();
    let mut members: Vec<Member> = Vec::new();
    let mut search: Vec<PathBuf> = Vec::new();
    for input in inputs {
        let dir = input.parent().map_or_else(|| PathBuf::from("."), Path::to_owned);
        let dir = if dir.as_os_str().is_empty() { PathBuf::from(".") } else { dir };
        if !search.contains(&dir) {
            search.push(dir);
        }
    }
    search.extend(unit_path.iter().cloned());

    let parse_only = Options {
        stage: Stage::Parse,
        ..options
    };
    let mut queue: VecDeque<usize> = VecDeque::new();
    for input in inputs {
        // a `#unit any` unit given takes the kind `--kind` names for it
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_owned();
        let chosen = kinds.for_unit(&stem);
        let choice = KindChoice {
            forced: chosen,
            fallback: None,
        };
        let member = open(input, &mut session, Options { kind: choice, ..parse_only }, true, overlay)?;
        if let MemberKind::Source(out) = &member.kind
            && let Some(d) = out.directive()
            && d.any.is_none()
            && kinds.names(&stem)
        {
            program_diags.push(
                Diagnostic::new(Code::Eu08)
                    .with_message(format!("`--kind {stem}=…` names a unit of fixed kind"))
                    .at(d.span)
                    .with_note("only a `#unit any` unit takes its kind from the command line")
                    .with_help(format!("leave `{stem}` out of `--kind`, or declare it with `#unit any`")),
            );
        }
        if matches!(member.kind, MemberKind::Source(_)) {
            queue.push_back(members.len());
        }
        members.push(member);
    }
    // what is known of each `#unit any` unit, by its file's name: the kind it
    // prefers, if any; a unit of fixed kind is known by its name alone
    let mut any_units: HashMap<String, Option<UnitKind>> = HashMap::new();
    for m in &members {
        if let Some(prefers) = any_of(m) {
            any_units.insert(super::written(&m.name).to_owned(), prefers);
        }
    }

    // follow imports until every unit named is found, or known to be missing
    // a compiled unit's own imports are followed too: checking its generic
    // bodies here needs their interfaces
    for (i, m) in members.iter().enumerate() {
        if matches!(m.kind, MemberKind::Compiled { .. }) {
            queue.push_back(i);
        }
    }
    loop {
        while let Some(i) = queue.pop_front() {
            if matches!(&members[i].kind, MemberKind::Source(out) if out.has_errors()) {
                continue;
            }
            let importer = members[i].unit_kind();
            let mut resolved = Vec::new();
            for (path, asked) in requests_of(&members, i, session.interner()) {
                // a name that already carries a kind, from a compiled unit's
                // metadata, asks for that kind
                let (path, asked) = match super::unit_of(&path).split_once('.') {
                    Some((_, k)) => {
                        let kind = if k == "quantum" { UnitKind::Quantum } else { UnitKind::Classical };
                        (path[..path.len() - k.len() - 1].to_owned(), Some(kind))
                    }
                    None => (path, asked),
                };
                let stem = super::unit_of(&path).to_owned();
                // the directories it is found in, kept with the name it resolves to
                let dirs = path[..path.len() - stem.len()].to_owned();
                // the name it goes by, if what it is is known already
                let known = |any_units: &HashMap<String, Option<UnitKind>>| match any_units.get(&stem) {
                    Some(&prefers) => asked.or(prefers).or(importer).map(|k| format!("{stem}.{}", k.name())),
                    None => Some(stem.clone()),
                };
                if let Some(name) = known(&any_units)
                    && members.iter().any(|m| m.name == name)
                {
                    resolved.push(format!("{dirs}{name}"));
                    continue;
                }
                // a library unit is found before anything on the unit path
                if let Some(text) = crate::library::source(&path) {
                    let id = session.add(crate::library::path(&path), text);
                    let out = front(&mut session, id, parse_only);
                    queue.push_back(members.len());
                    members.push(Member::source(path.clone(), out, false));
                    resolved.push(path);
                    continue;
                }
                let choice = KindChoice {
                    forced: asked,
                    fallback: importer,
                };
                let Some(file) = find_unit(&search, &path, overlay, asked, importer, &mut session)? else {
                    // missing: analysis reports it where it is imported
                    resolved.push(path.clone());
                    continue;
                };
                let member = open(&file, &mut session, Options { kind: choice, ..parse_only }, false, overlay)?;
                if let Some(prefers) = any_of(&member) {
                    any_units.insert(stem.clone(), prefers);
                }
                let name = member.name.clone();
                if !members.iter().any(|m| m.name == name) {
                    queue.push_back(members.len());
                    members.push(member);
                }
                resolved.push(format!("{dirs}{name}"));
            }
            members[i].deps = resolved;
        }
        // only once every unit is found is it known which are quantum
        let gained = imply_qpu(&mut members);
        if gained.is_empty() {
            break;
        }
        queue.extend(gained);
    }

    let mut loaded = Loaded {
        session,
        members,
        diagnostics: program_diags,
    };
    if options.stage > Stage::Parse {
        let sources = source_graph(&loaded);
        for group in sources.groups() {
            // a unit compiled earlier is analysed already
            if !matches!(loaded.members[group[0]].kind, MemberKind::Source(_)) {
                continue;
            }
            if sources.is_cycle(&group) {
                analyze_cycle(&mut loaded, &group, options);
            } else {
                analyze(&mut loaded, group[0], options);
            }
        }
    }
    Ok(loaded)
}

/// Opens one file of the program, parsing a source file, whose text is
/// `overlay`'s if it holds the path.
fn open(
    path: &Path,
    session: &mut Session,
    options: Options,
    given: bool,
    overlay: &Overlay,
) -> Result<Member, LoadError> {
    if is_object(path) {
        let mut metadata = read_compiled(path, session)?;
        reparse(&mut metadata, session, path)?;
        return Ok(Member {
            name: metadata.unit.clone(),
            kind: MemberKind::Compiled {
                path: path.to_owned(),
                metadata: Box::new(metadata),
            },
            given,
            implied: Vec::new(),
            deps: Vec::new(),
        });
    }
    let loaded = match overlay.get(path) {
        Some(text) => Ok(session.add(path, text)),
        None => session.load(path),
    };
    let id = match loaded {
        Ok(id) => id,
        // a file read but not decoded is a program's fault, reported as one
        Err(error) => match crate::source::splice::DecodeError::of(&error) {
            Some(bad) => {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("<anonymous>").to_owned();
                let out = Outcome {
                    diagnostics: vec![bad.diagnostic(path)],
                    ..Outcome::default()
                };
                return Ok(Member::source(name, out, given));
            }
            None => {
                return Err(LoadError::Io {
                    path: path.to_owned(),
                    error,
                });
            }
        },
    };
    let out = front(session, id, options);
    let name = out.identity(session);
    Ok(Member::source(name, out, given))
}

/// The units the member `i` depends on, by name: those it imports and
/// names, and those it depends on without naming them.
fn deps_of(members: &[Member], i: usize) -> Vec<String> {
    let mut names = members[i].deps.clone();
    names.extend(members[i].implied.iter().cloned());
    names
}

/// What the member `i` asks to be found: each unit it imports and names, as
/// written, with the kind an `import(kind)` asks of it, and those it depends
/// on without naming them.
fn requests_of(members: &[Member], i: usize, interner: &crate::intern::Interner) -> Vec<(String, Option<UnitKind>)> {
    let mut out: Vec<(String, Option<UnitKind>)> = match &members[i].kind {
        MemberKind::Source(out) => {
            // the kind each `import(kind)` asks, by where the import is
            let mut asked: HashMap<crate::span::Span, UnitKind> = HashMap::new();
            if let Some(unit) = &out.unit {
                for item in &unit.items {
                    if let crate::ast::ItemKind::Import { kind: Some(k), .. } = &item.node.kind {
                        match interner.resolve(k.node) {
                            "classical" => asked.insert(item.span, UnitKind::Classical),
                            "quantum" => asked.insert(item.span, UnitKind::Quantum),
                            _ => None,
                        };
                    }
                }
            }
            dependencies(out, super::written(&members[i].name), interner)
                .into_iter()
                .map(|(n, at)| (n, asked.get(&at).copied()))
                .collect()
        }
        // what a compiled unit uses may include a library unit it depends on
        // without naming it; the names it records carry their kinds
        MemberKind::Compiled { metadata, .. } => {
            let mut names: Vec<(String, Option<UnitKind>)> =
                metadata.interface.imports.iter().map(|n| (n.clone(), None)).collect();
            for u in &metadata.uses {
                if !names.iter().any(|(n, _)| *n == u.unit) {
                    names.push((u.unit.clone(), None));
                }
            }
            names
        }
    };
    out.extend(members[i].implied.iter().map(|n| (n.clone(), None)));
    out
}

/// For a `#unit any` member, the kind it prefers, if any; `None` for a unit
/// of fixed kind.
fn any_of(m: &Member) -> Option<Option<UnitKind>> {
    match &m.kind {
        MemberKind::Source(out) => out.directive().and_then(|d| d.any).map(|a| a.preference),
        MemberKind::Compiled { metadata, .. } => metadata.interface.any,
    }
}

/// The file of the unit `path` on the unit path `search`: its source,
/// `NAME.tq`, or else the unit compiled earlier. A `#unit any` unit compiled
/// earlier is `NAME.tcu` as a classical unit and `NAME.tqu` as a quantum one,
/// the kind being the one `asked`, or else the one it prefers, or else the
/// `importer`'s.
///
/// # Errors
///
/// A [`LoadError`] when a compiled unit's metadata cannot be read.
fn find_unit(
    search: &[PathBuf],
    path: &str,
    overlay: &Overlay,
    asked: Option<UnitKind>,
    importer: Option<UnitKind>,
    session: &mut Session,
) -> Result<Option<PathBuf>, LoadError> {
    let exists = |p: &Path| overlay.contains_key(p) || p.is_file();
    if let Some(src) = search.iter().map(|dir| dir.join(format!("{path}.tq"))).find(|p| exists(p)) {
        return Ok(Some(src));
    }
    for dir in search {
        for ext in ["tcu", "tqu"] {
            let p = dir.join(format!("{path}.{ext}"));
            if !p.is_file() {
                continue;
            }
            let Some(prefers) = read_compiled(&p, session)?.interface.any else {
                return Ok(Some(p));
            };
            let kind = asked.or(prefers).or(importer).unwrap_or(UnitKind::Classical);
            let want = dir.join(format!("{path}.{}", if kind == UnitKind::Quantum { "tqu" } else { "tcu" }));
            return Ok(want.is_file().then_some(want));
        }
    }
    Ok(None)
}

/// Whether the member is a quantum unit.
fn is_quantum(m: &Member) -> bool {
    match &m.kind {
        MemberKind::Source(out) => out.unit_kind() == Some(crate::pp::UnitKind::Quantum),
        MemberKind::Compiled { metadata, .. } => metadata.interface.quantum,
    }
}

/// Gives each classical unit compiled from source that imports a quantum
/// unit the library unit `qpu`, which runs circuits, as an implied
/// dependency: the positions of the members that gained it.
fn imply_qpu(members: &mut [Member]) -> Vec<usize> {
    let mut gained = Vec::new();
    for i in 0..members.len() {
        let MemberKind::Source(out) = &members[i].kind else { continue };
        if out.has_errors()
            || out.unit_kind() != Some(crate::pp::UnitKind::Classical)
            || crate::library::source(&members[i].name).is_some()
            || members[i].implied.iter().any(|n| n == "qpu")
        {
            continue;
        }
        let imports_quantum = deps_of(members, i)
            .iter()
            .filter_map(|n| members.iter().find(|m| m.name == super::unit_of(n)))
            .any(is_quantum);
        if imports_quantum {
            members[i].implied.push("qpu".to_owned());
            gained.push(i);
        }
    }
    gained
}

/// Parses again the source a compiled unit carries, so that its generic and
/// constant functions can be checked where they are used.
fn reparse(metadata: &mut Metadata, session: &mut Session, object: &Path) -> Result<(), LoadError> {
    let Some(src) = &metadata.interface.source else {
        return Ok(());
    };
    let id = session.add(PathBuf::from(&src.path), &src.text);
    let mut embeds = crate::pp::MapEmbedResolver {
        files: src.embeds.iter().cloned().collect(),
    };
    let out = super::front_with(
        session,
        id,
        Options {
            stage: Stage::Parse,
            ..Options::default()
        },
        &mut embeds,
    );
    match out.unit {
        Some(unit) if !out.has_errors() => {
            metadata.interface.ast = Some(std::sync::Arc::new(unit));
            Ok(())
        }
        _ => Err(LoadError::Metadata {
            path: object.to_owned(),
            reason: "the source it carries no longer parses".to_owned(),
        }),
    }
}

/// Every unit `names` import, directly or through each other, found among
/// `members`.
fn closure(loaded: &Loaded, names: Vec<String>) -> Vec<&Member> {
    let mut out: Vec<&Member> = Vec::new();
    let mut work = names;
    while let Some(name) = work.pop() {
        let Some(m) = loaded.members.iter().find(|m| m.name == super::unit_of(&name)) else {
            continue;
        };
        if out.iter().any(|o| std::ptr::eq(*o, m)) {
            continue;
        }
        if let Some(meta) = m.metadata() {
            work.extend(meta.interface.imports.iter().cloned());
        }
        out.push(m);
    }
    out
}

/// The source units each imports, by position among the members: directly,
/// or through units compiled earlier, which are analysed already but whose
/// own imports may be compiled in this run.
fn source_deps(loaded: &Loaded) -> Vec<Vec<usize>> {
    let find = |name: &str| loaded.members.iter().position(|o| o.name == super::unit_of(name));
    let members = &loaded.members;
    (0..members.len())
        .map(|i| match &members[i].kind {
            MemberKind::Source(_) => {
                let mut deps = Vec::new();
                let mut seen = Vec::new();
                let mut work: Vec<usize> = deps_of(members, i)
                    .iter()
                    .filter_map(|name| find(name))
                    .collect();
                while let Some(j) = work.pop() {
                    if seen.contains(&j) {
                        continue;
                    }
                    seen.push(j);
                    match &members[j].kind {
                        MemberKind::Source(_) => deps.push(j),
                        MemberKind::Compiled { .. } => {
                            work.extend(deps_of(members, j).iter().filter_map(|n| find(n)));
                        }
                    }
                }
                deps
            }
            MemberKind::Compiled { .. } => Vec::new(),
        })
        .collect()
}

/// The members as a graph of which source unit depends on which, as
/// [`source_deps`] finds them: unit `i` of the graph is member `i`.
fn source_graph(loaded: &Loaded) -> Graph {
    let mut graph = Graph::new();
    for m in &loaded.members {
        graph.push(&m.name);
    }
    for (i, deps) in source_deps(loaded).into_iter().enumerate() {
        for j in deps {
            graph.link(i, j);
        }
    }
    graph
}

/// The program's units and what each depends on, by name, as `tqc emit
/// --stage depgraph` draws it: unit `i` of the graph is member `i`, and a
/// unit named but not found is added after the members. Each is noted as a
/// library unit, a unit compiled earlier, a quantum unit, or one not found.
pub fn graph(loaded: &Loaded) -> Graph {
    let mut graph = Graph::new();
    for m in &loaded.members {
        let i = graph.push(&m.name);
        if crate::library::source(&m.name).is_some() {
            graph.note(i, "library");
        }
        if matches!(m.kind, MemberKind::Compiled { .. }) {
            graph.note(i, "compiled");
        }
        if m.unit_kind() == Some(UnitKind::Quantum) {
            graph.note(i, "quantum");
        }
    }
    for i in 0..loaded.members.len() {
        for name in deps_of(&loaded.members, i) {
            let j = match loaded.members.iter().position(|o| o.name == super::unit_of(&name)) {
                Some(j) => j,
                None => graph.index(&name).unwrap_or_else(|| {
                    let j = graph.push(&name);
                    graph.note(j, "not found");
                    j
                }),
            };
            graph.link(i, j);
        }
    }
    graph
}

/// Analyses, and compiles as far as asked, units that import one another:
/// each against the interfaces they settle on, as [`super::settle`] finds
/// them, which are also what each offers when it is linked.
fn analyze_cycle(loaded: &mut Loaded, group: &[usize], options: Options) {
    // a unit outside the group that one of them imports, which failed, has
    // its own diagnostics to say so
    let mut outside: Vec<Vec<Interface>> = Vec::new();
    for &i in group {
        let in_group = |m: &Member| group.iter().any(|&g| std::ptr::eq(m, &loaded.members[g]));
        let Some(interfaces) = imported(loaded, i, in_group) else { return };
        outside.push(interfaces);
    }
    let mut outs: Vec<&Outcome> = Vec::new();
    for &i in group {
        let MemberKind::Source(out) = &loaded.members[i].kind else { return };
        outs.push(out);
    }
    let settled = super::settle(&loaded.session, &outs, &outside);
    for (k, &i) in group.iter().enumerate() {
        compile_member(loaded, i, options, &settled[k]);
    }
}

/// Analyses, and compiles as far as asked, one source unit, unless a unit
/// it imports failed, whose own diagnostics then say why.
fn analyze(loaded: &mut Loaded, i: usize, options: Options) {
    // a compiled unit this one imports may itself import this one; this
    // unit's own interface is not needed to analyse it
    let this = &loaded.members[i];
    if let Some(interfaces) = imported(loaded, i, |m| std::ptr::eq(m, this)) {
        compile_member(loaded, i, options, &interfaces);
    }
}

/// The interfaces of every unit the source member `i` imports, directly or
/// through the ones it imports (their generic bodies are checked in its
/// analysis, in their own scope), other than those `skip` picks. `None`
/// when `i` failed, or one of them did and so was not analysed: its own
/// diagnostics say why.
fn imported(loaded: &Loaded, i: usize, skip: impl Fn(&Member) -> bool) -> Option<Vec<Interface>> {
    let MemberKind::Source(out) = &loaded.members[i].kind else { return None };
    if out.has_errors() {
        return None;
    }
    let direct = deps_of(&loaded.members, i);
    closure(loaded, direct)
        .into_iter()
        .filter(|m| !skip(m))
        .map(|m| m.metadata().map(|meta| meta.interface.clone()))
        .collect()
}

/// Compiles the source member `i`, parsed already, against `interfaces`.
fn compile_member(loaded: &mut Loaded, i: usize, options: Options, interfaces: &[Interface]) {
    let session = &loaded.session;
    let deps = loaded.members[i].deps.clone();
    let MemberKind::Source(out) = &mut loaded.members[i].kind else {
        return;
    };
    let mut taken = std::mem::take(out.as_mut());
    back(session, &mut taken, options, interfaces);
    // the units it imports, as this program resolved them, so that a
    // `#unit any` unit among them is found as the same kind again
    if let Some(meta) = &mut taken.metadata {
        meta.interface.imports = deps;
    }
    **out = finish(taken, options);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::Code;

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    fn check(inputs: &[PathBuf], extra: &[PathBuf]) -> Loaded {
        load(
            inputs,
            extra,
            Options {
                stage: Stage::Check,
                ..Options::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn an_import_is_found_beside_the_importer_and_analyzed_first() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "geo.tq", "#unit classical\nstruct P { x: i32 }\nfn area(p: *P) -> i32 { p.x }\n");
        let main = write(
            dir.path(),
            "app.tq",
            "#unit classical\nimport geo;\nfn main() -> i32 { let p = geo::P { x: 3 }; geo::area(&p) }\n",
        );
        let loaded = check(&[main], &[]);
        let errors: Vec<String> = loaded.all_diagnostics().map(|d| d.to_string()).collect();
        assert!(!loaded.has_errors(), "{errors:?}");
        let names: Vec<&str> = loaded.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["app", "geo", "core"], "the library's `core` comes with every program");
        assert!(loaded.members.iter().all(|m| m.metadata().is_some()));
    }

    #[test]
    fn the_unit_path_is_searched_after_the_files_directories() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join("lib");
        std::fs::create_dir(&lib).unwrap();
        write(&lib, "util.tq", "#unit classical\nfn one() -> i32 { 1 }\n");
        let main = write(dir.path(), "app.tq", "#unit classical\nimport util;\nfn main() -> i32 { util::one() }\n");
        let loaded = check(std::slice::from_ref(&main), &[]);
        assert!(loaded.all_diagnostics().any(|d| d.code == Code::Es17), "not found without it");
        let loaded = check(&[main], &[lib]);
        assert!(!loaded.has_errors());
    }

    #[test]
    fn an_import_of_several_parts_is_found_in_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("shapes").join("solid");
        std::fs::create_dir_all(&nested).unwrap();
        write(&nested, "cube.tq", "#unit classical\nfn volume(s: i64) -> i64 { s * s * s }\n");
        let main = write(
            dir.path(),
            "app.tq",
            "#unit classical\nimport shapes::solid::cube;\nfn main() -> i32 { cube::volume(2) as i32 }\n",
        );
        let loaded = check(&[main], &[]);
        let errors: Vec<String> = loaded.all_diagnostics().map(|d| d.to_string()).collect();
        assert!(!loaded.has_errors(), "{errors:?}");
        assert!(loaded.members.iter().any(|m| m.name == "cube"));
    }

    #[test]
    fn units_that_import_each_other_are_analyzed_together() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "b.tq",
            "#unit classical\nimport a;\nstruct B { a: a::A }\nfn g(n: i32) -> i32 { if n == 0 { 0 } else { a::f(n - 1) } }\n",
        );
        let a = write(
            dir.path(),
            "a.tq",
            "#unit classical\nimport b;\nstruct A { n: i32 }\nfn f(n: i32) -> i32 { b::g(n) }\nfn h(x: b::B) -> i32 { x.a.n }\n",
        );
        let loaded = check(&[a], &[]);
        let codes: Vec<Code> = loaded.all_diagnostics().map(|d| d.code).collect();
        assert_eq!(codes, [], "each finds what the other offers");
        assert!(loaded.members.iter().all(|m| m.metadata().is_some()), "both were analysed");
        // a name the other does not have is still reported, once
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "b.tq", "#unit classical\nimport a;\nfn g() -> i32 { a::missing() }\n");
        let a = write(dir.path(), "a.tq", "#unit classical\nimport b;\nfn f() -> i32 { b::g() }\n");
        let loaded = check(&[a], &[]);
        let codes: Vec<Code> = loaded.all_diagnostics().map(|d| d.code).collect();
        assert_eq!(codes, [Code::Es04]);
    }

    #[test]
    fn a_unit_whose_import_failed_is_not_analyzed() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "bad.tq", "#unit classical\nfn f() -> i32 { true }\n");
        let main = write(dir.path(), "app.tq", "#unit classical\nimport bad;\nfn main() -> i32 { bad::f() }\n");
        let loaded = check(&[main], &[]);
        let codes: Vec<Code> = loaded.all_diagnostics().map(|d| d.code).collect();
        assert_eq!(codes, [Code::Es06], "only the imported unit's own error");
    }

    #[test]
    fn core_named_in_another_units_body_is_the_librarys() {
        // a generic's body is checked where it is instantiated, and a quantum
        // operator's where it is inlined, each in its own unit's scope
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "pairs.tq", "#unit classical\nfn flip<T>(a: *T, b: *T) { core::swap(a, b); }\n");
        let main = write(
            dir.path(),
            "app.tq",
            "#unit classical\nimport pairs;\nfn main() -> i32 { let x: i32 = 1; let y: i32 = 2; pairs::flip(&x, &y); x }\n",
        );
        let loaded = check(&[main], &[]);
        let errors: Vec<String> = loaded.all_diagnostics().map(|d| d.to_string()).collect();
        assert!(!loaded.has_errors(), "{errors:?}");
        write(dir.path(), "qpairs.tq", "#unit quantum\nfn flip(a: *[qubit; 2], b: *[qubit; 2]) { core::swap(a, b); }\n");
        let main = write(
            dir.path(),
            "circ.tq",
            "#unit quantum\nimport qpairs;\nfn g(a: *[qubit; 2], b: *[qubit; 2]) { qpairs::flip(a, b); }\n",
        );
        let loaded = check(&[main], &[]);
        let errors: Vec<String> = loaded.all_diagnostics().map(|d| d.to_string()).collect();
        assert!(!loaded.has_errors(), "{errors:?}");
    }

    #[test]
    fn the_graph_shows_each_unit_and_what_it_depends_on() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "b.tq", "#unit classical\nimport a;\nfn g() -> i32 { 0 }\n");
        let a = write(dir.path(), "a.tq", "#unit classical\nimport b;\nimport missing;\nfn f() -> i32 { b::g() }\n");
        let loaded = load(&[a], &[], Options::default()).unwrap();
        let graph = graph(&loaded);
        let a = graph.index("a").expect("a");
        let names: Vec<&str> = graph.deps(a).iter().map(|&i| graph.name(i)).collect();
        assert_eq!(names, ["b", "missing", "core"]);
        let text = graph.render(&[a]);
        assert!(text.contains("+-- missing [not found]"), "{text}");
        assert!(text.contains("core [library]"), "{text}");
        assert!(text.contains("+-- a (cycle)"), "{text}");
        assert!(text.ends_with("analysed in order: core, missing, {a, b}\n"), "{text}");
    }
}
