//! The compiler driver: one translation, phase by phase.
//!
//! The pipeline lives in the library rather than in the `tqc` binary so that it
//! can be unit tested and so that a tool embedding this crate gets the same
//! behaviour the command line does.
//!
//! # Order
//!
//! The phases run in a fixed order:
//!
//! - 1 and 2, decoding and line splicing: [`crate::source::splice`];
//! - 3, tokenisation: [`crate::lex`];
//! - 4, preprocessing: [`crate::pp`], then the bracket-marking pass,
//!   [`crate::mark`];
//! - 5, parsing: [`crate::parse`];
//! - 6 to 8, name resolution, constant evaluation and type checking:
//!   [`crate::sema`];
//! - 9, for a quantum unit, circuit lowering and judgement:
//!   [`crate::lower`];
//! - 10, for a classical unit, code generation, when built with the `llvm`
//!   feature, `crate::codegen`.
//!
//! A quantum unit has no object of native code. What it offers (each
//! `[entry]` operator's circuit, and the judgement record of each operator
//! with program linkage) goes into its metadata once its circuits are made,
//! and its metadata is its output, the `.tqu`. It is judged at the least
//! common multiple of its conductor and those of the quantum units it
//! imports, whose judgements meet its own.
//!
//! Linking is not part of this pipeline. It combines the objects of several
//! units into one program, so it runs once per program rather than once per
//! unit; see `crate::link`.
//!
//! # Programs of several units
//!
//! A unit is analysed against the interfaces of the units it imports, so
//! translating one unit can mean first finding and translating others.
//! [`program`] does that: from the files given it follows every `import`
//! along the unit path, reads units compiled earlier from their objects'
//! metadata, and translates the rest in an order where each unit comes after
//! everything it imports (units that import one another being translated
//! together). [`compile`], translating one unit, does the same for the library
//! units it uses, which may depend on one another too. [`graph`] works out
//! that order for both, and draws the graph for `tqc emit --stage depgraph`.
//!
//! # Stopping early
//!
//! Five diagnostics are warnings (an unknown annotation, an unknown pragma,
//! a `match` arm that can never be reached, a judgement that left the checked
//! fragment, and a use of a deprecated item), and every other one
//! stops translation of that unit. The pipeline checks for errors after each
//! phase: running the parser over a token stream the preprocessor could not
//! make sense of produces noise, not information.

#[cfg(feature = "llvm")]
pub mod build;
pub mod graph;
pub mod options;
pub mod program;
pub mod session;

pub use options::{KindArgs, OptLevel, Options, Stage};
pub use session::Session;

use crate::ast::Unit;
use crate::diag::{Code, Diagnostic};
use crate::lex::Token;
use crate::mark::Marked;
use crate::pp::{Preprocessed, UnitDirective, UnitKind};
use crate::span::{SourceId, Span};

/// What one translation produced.
#[derive(Debug, Default)]
pub struct Outcome {
    /// The stage actually reached.
    pub reached: Option<Stage>,
    /// Phase 3's tokens.
    pub tokens: Vec<(Token, Span)>,
    /// Phase 3's doc comments, which the parser gives to declarations.
    pub docs: Vec<crate::lex::DocComment>,
    /// Phase 4's result, including the unit directive and the pragmas.
    pub preprocessed: Option<Preprocessed>,
    /// The marked token stream.
    pub marked: Option<Marked>,
    /// Phase 5's tree.
    pub unit: Option<Unit>,
    /// Analysis, including the typed unit.
    pub analysis: Option<crate::sema::Analysis>,
    /// The unit's metadata.
    pub metadata: Option<crate::meta::Metadata>,
    /// The circuits phase 9 made.
    pub circuits: Vec<crate::lower::Lowered>,
    /// The LLVM IR generated for the unit.
    pub llvm_ir: Option<String>,
    /// The object file generated for the unit.
    pub object: Option<Vec<u8>>,
    /// Everything diagnosed, in phase order.
    pub diagnostics: Vec<Diagnostic>,
    /// The file translated.
    pub source: Option<SourceId>,
    /// Every file `#embed` read, by the path it was named with, and its text.
    pub embeds: Vec<(String, String)>,
    /// Every document `@embed` names, by the path it was named with: the
    /// source it was added to the session as, or why it could not be read.
    pub documents: Vec<(String, Result<SourceId, String>)>,
}

impl Outcome {
    /// Whether any error was reported.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }

    /// How many errors were reported.
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_error()).count()
    }

    /// How many warnings were reported.
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| !d.is_error()).count()
    }

    /// Whether a diagnostic with this identifier was reported.
    pub fn reported(&self, code: Code) -> bool {
        self.diagnostics.iter().any(|d| d.code == code)
    }

    /// Whether the translation succeeded as far as it was asked to go.
    pub fn succeeded(&self, asked: Stage) -> bool {
        !self.has_errors() && self.reached == Some(asked)
    }

    /// The unit's `#unit` directive.
    pub fn directive(&self) -> Option<UnitDirective> {
        self.preprocessed.as_ref().and_then(|p| p.unit)
    }

    /// The name the unit goes by within a program.
    pub fn identity(&self, session: &Session) -> String {
        let stem = self.source.map_or("", |id| session.sources().file(id).unit_name());
        identity(stem, self.directive())
    }

    /// The unit's kind.
    pub fn unit_kind(&self) -> Option<UnitKind> {
        self.directive().map(|u| u.kind)
    }

    /// The conductor the unit declares, or the default of 8.
    pub fn conductor(&self) -> u32 {
        self.preprocessed.as_ref().map_or(8, |p| p.conductor)
    }

    /// The typed unit.
    pub fn tir(&self) -> Option<&crate::tir::Unit> {
        self.analysis.as_ref().and_then(|a| a.tir.as_ref())
    }
}

/// Runs the pipeline over one source file that imports nothing it cannot
/// find among the library units.
pub fn compile(session: &mut Session, id: SourceId, options: Options) -> Outcome {
    compile_with(session, id, options, &[])
}

/// Runs the pipeline over one source file, which may import any unit whose
/// interface is among `imports`.
pub fn compile_with(
    session: &mut Session,
    id: SourceId,
    options: Options,
    imports: &[crate::sema::Interface],
) -> Outcome {
    let mut out = front(session, id, options);
    if out.reached == Some(Stage::Parse) && !out.has_errors() && options.stage > Stage::Parse {
        // the library units it uses are analysed here, if nothing gave them
        let mut all = imports.to_vec();
        let this = out.identity(session);
        for (name, _) in dependencies(&out, &this, session.interner()) {
            if !all.iter().any(|i| i.unit == name)
                && let Some(i) = library_interface(session, &name)
            {
                all.push(i);
            }
        }
        // a classical unit may run the circuits of a quantum unit it
        // imports, which needs `qpu`
        if out.unit_kind() == Some(UnitKind::Classical)
            && all.iter().any(|i| i.quantum)
            && !all.iter().any(|i| i.unit == "qpu")
            && let Some(i) = library_interface(session, "qpu")
        {
            all.push(i);
        }
        back(session, &mut out, options, &all);
    }
    finish(out, options)
}

/// The interface of the library unit `name`, analysing it the first time it
/// is asked for in this session. `None` if this compiler does not provide
/// it, or it failed, which would be a fault in the compiler.
///
/// The library units it depends on, directly or through one another, are
/// analysed with it: each after the units it depends on, and units that
/// depend on one another together, as [`program`] analyses a program's.
pub fn library_interface(session: &mut Session, name: &str) -> Option<crate::sema::Interface> {
    library_interface_from(session, name, &crate::library::source)
}

/// [`library_interface`], with the library's text as `source` gives it: the
/// source of the unit it is given the name of, or `None` for no unit.
fn library_interface_from(session: &mut Session, name: &str, source: &LibrarySource<'_>) -> Option<crate::sema::Interface> {
    if let Some(i) = session.library.get(name) {
        return Some(i.clone());
    }
    source(name)?;
    let (graph, mut outs) = library_graph(session, name, source);
    let check = Options {
        stage: Stage::Check,
        ..Options::default()
    };
    for group in graph.groups() {
        // what each unit of the group takes from library units outside it,
        // all of which are analysed by now
        let outside: Vec<Vec<crate::sema::Interface>> = group
            .iter()
            .map(|&i| {
                graph
                    .deps(i)
                    .iter()
                    .filter(|j| !group.contains(j))
                    .filter_map(|&j| session.library.get(graph.name(j)).cloned())
                    .collect()
            })
            .collect();
        let interfaces = if graph.is_cycle(&group) {
            let members: Vec<&Outcome> = group.iter().map(|&i| &outs[i]).collect();
            settle(session, &members, &outside)
        } else {
            outside
        };
        for (k, &i) in group.iter().enumerate() {
            let mut out = std::mem::take(&mut outs[i]);
            if out.reached == Some(Stage::Parse) && !out.has_errors() {
                back(session, &mut out, check, &interfaces[k]);
            }
            let out = finish(out, check);
            if let Some(meta) = out.metadata {
                session.library.insert(graph.name(i).to_owned(), meta.interface);
            }
        }
    }
    session.library.get(name).cloned()
}

/// The text of a library unit.
type LibrarySource<'s> = dyn Fn(&str) -> Option<&'static str> + 's;

/// The library unit `name`, parsed, with every library unit it depends on,
/// directly or through others, that this session has not analysed yet; and
/// the graph of which depends on which, `name` its first unit. Units the
/// session has analysed already are in the graph too, with nothing left to
/// analyse, so that what others take from them is found through it. A
/// classical unit depending on a quantum one also depends on `qpu`, which
/// runs circuits, as in [`compile_with`].
fn library_graph(session: &mut Session, name: &str, source: &LibrarySource<'_>) -> (graph::Graph, Vec<Outcome>) {
    let parse = Options {
        stage: Stage::Parse,
        ..Options::default()
    };
    let mut graph = graph::Graph::new();
    let mut outs: Vec<Outcome> = Vec::new();
    let parse_unit = |session: &mut Session, graph: &mut graph::Graph, outs: &mut Vec<Outcome>, unit: &str| {
        let i = graph.add(unit);
        let out = match (session.library.contains_key(unit), source(unit)) {
            (false, Some(text)) => {
                let id = session.add(crate::library::path(unit), text);
                front(session, id, parse)
            }
            _ => Outcome::default(),
        };
        outs.push(out);
        i
    };
    parse_unit(session, &mut graph, &mut outs, name);
    let mut next = 0;
    while next < graph.len() {
        let i = next;
        next += 1;
        if outs[i].reached != Some(Stage::Parse) || outs[i].has_errors() {
            continue;
        }
        let mut deps: Vec<String> = dependencies(&outs[i], graph.name(i), session.interner())
            .into_iter()
            .map(|(d, _)| d)
            .filter(|d| source(d).is_some())
            .collect();
        for d in &deps {
            if graph.index(d).is_none() {
                parse_unit(session, &mut graph, &mut outs, d);
            }
        }
        let quantum = |d: &String| match session.library.get(d) {
            Some(iface) => iface.quantum,
            None => graph.index(d).is_some_and(|j| outs[j].unit_kind() == Some(UnitKind::Quantum)),
        };
        if outs[i].unit_kind() == Some(UnitKind::Classical)
            && deps.iter().any(quantum)
            && !deps.iter().any(|d| d == "qpu")
        {
            if graph.index("qpu").is_none() {
                parse_unit(session, &mut graph, &mut outs, "qpu");
            }
            deps.push("qpu".to_owned());
        }
        for d in &deps {
            let j = graph.index(d).expect("added above");
            graph.link(i, j);
        }
    }
    (graph, outs)
}

/// The interfaces each of `group`, units that depend on one another, is
/// compiled against: what the units outside the group that any of them
/// depends on offer, `outside[k]` being those of the `k`-th, and what every
/// unit of the group offers.
///
/// Each needs the others' interfaces before its own is known. So each is
/// analysed against what the others were last found to offer (nothing, the
/// first time), and what it offers is found in turn, the round repeated until
/// no unit offers anything new. Every name one takes from another then
/// resolves. Every unit of the group, the unit itself included, is offered:
/// another's bodies, checked in the unit's analysis in their own scope, may
/// name it. For the same reason each is offered what every other imports
/// from outside the group: another's declarations, opened in the unit's
/// analysis, import those units in turn.
pub fn settle(session: &Session, group: &[&Outcome], outside: &[Vec<crate::sema::Interface>]) -> Vec<Vec<crate::sema::Interface>> {
    let mut beyond: Vec<crate::sema::Interface> = Vec::new();
    for i in outside.iter().flatten() {
        if !beyond.iter().any(|b| b.unit == i.unit) {
            beyond.push(i.clone());
        }
    }
    let offered = |provisional: &[Option<crate::meta::Metadata>]| -> Vec<crate::sema::Interface> {
        let mut interfaces = beyond.clone();
        interfaces.extend(provisional.iter().flatten().map(|m| m.interface.clone()));
        interfaces
    };
    let mut provisional: Vec<Option<crate::meta::Metadata>> = vec![None; group.len()];
    let mut text: Vec<String> = vec![String::new(); group.len()];
    // each round resolves at least one more name one unit takes from
    // another, so a bound on rounds is only a guard
    for _ in 0..(4 * group.len() + 4) {
        let mut changed = false;
        for (k, out) in group.iter().enumerate() {
            let interfaces = offered(&provisional);
            let meta = if out.has_errors() { None } else { provisional_metadata(session, out, &interfaces) };
            let now = meta.as_ref().map(|m| crate::meta::encode::encode(m, session.interner())).unwrap_or_default();
            if now != text[k] {
                changed = true;
                text[k] = now;
            }
            provisional[k] = meta;
        }
        if !changed {
            break;
        }
    }
    (0..group.len()).map(|_| offered(&provisional)).collect()
}

/// The units a parsed unit named `this` depends on: those it imports, the
/// library's `core`, which every unit uses without importing it, and each
/// other library unit it names as `unit::item`, which needs no import
/// either.
pub fn dependencies(out: &Outcome, this: &str, interner: &crate::intern::Interner) -> Vec<(String, Span)> {
    let mut deps = imports_of(out, interner);
    if out.unit.is_some()
        && let Some(pp) = &out.preprocessed
    {
        for pair in pp.tokens.windows(2) {
            let [(Token::Ident(s), at), (next, _)] = pair else {
                continue;
            };
            let name = interner.resolve(*s);
            if *next == Token::Punct(crate::lex::Punct::ColonColon)
                && name != "core"
                && name != this
                && crate::library::source(name).is_some()
                && !deps.iter().any(|(d, _)| d == name)
            {
                deps.push((name.to_owned(), *at));
            }
        }
    }
    if this != "core" && out.unit.is_some() {
        let at = out.directive().map_or_else(Span::synthetic, |d| d.span);
        deps.push(("core".to_owned(), at));
    }
    deps
}

/// The units a parsed unit imports, other than the library's `core`, which
/// is always available: each as the path it is found at on the unit path,
/// `shapes/solid/cube` for `import shapes::solid::cube;`.
pub fn imports_of(out: &Outcome, interner: &crate::intern::Interner) -> Vec<(String, Span)> {
    let Some(unit) = &out.unit else {
        return Vec::new();
    };
    unit.items
        .iter()
        .filter_map(|item| match &item.node.kind {
            crate::ast::ItemKind::Import { path, .. } => {
                let parts: Vec<&str> = path.segments.iter().map(|s| interner.resolve(s.node)).collect();
                let name = parts.join("/");
                (name != "core").then_some((name, item.span))
            }
            _ => None,
        })
        .collect()
}

/// The unit an import path names: its last part, which is the file's name.
pub fn unit_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The name a unit goes by within a program: its file's name `stem`, with,
/// for a `#unit any` unit, the kind it is translated as after a `.`, as
/// `dsp.quantum`. One source can be a unit of each kind, and the two are
/// different units; the `.` cannot occur in a unit's own name.
pub fn identity(stem: &str, directive: Option<UnitDirective>) -> String {
    match directive {
        Some(d) if d.any.is_some() => format!("{stem}.{}", d.kind.name()),
        _ => stem.to_owned(),
    }
}

/// The written name `identity` stands for: `dsp` for `dsp.quantum`.
pub fn written(identity: &str) -> &str {
    identity.split('.').next().unwrap_or(identity)
}

/// Phases 1 to 5 over one source file: everything up to and including the
/// syntax tree. Stops early at an error or at the stage asked for.
pub fn front(session: &mut Session, id: SourceId, options: Options) -> Outcome {
    front_with(session, id, options, &mut crate::pp::FsEmbedResolver)
}

/// [`front`], with `#embed` read through `embed`.
pub fn front_with(
    session: &mut Session,
    id: SourceId,
    options: Options,
    embed: &mut dyn crate::pp::EmbedResolver,
) -> Outcome {
    let mut out = Outcome::default();

    // phases 1 to 3. decoding and splicing already happened when the file was
    // added to the source map
    let spliced = session.sources().file(id).spliced().clone();
    let lexed = crate::lex::lex(id, &spliced, session.interner_mut());
    out.diagnostics.extend(lexed.diagnostics);
    out.tokens = lexed.tokens;
    out.docs = lexed.docs;
    out.reached = Some(Stage::Tokens);
    if out.has_errors() || options.stage == Stage::Tokens {
        return finish(out, options);
    }

    // phase 4
    let mut recording = Recording {
        inner: embed,
        read: Vec::new(),
    };
    let pp = {
        let (sources, interner) = session.split_mut();
        crate::pp::preprocess_as(id, &out.tokens, sources, interner, &mut recording, options.kind)
    };
    // the documents `@embed` names are read now, and become sources of their
    // own, so that a mistake in one is reported at its line
    let from = session.sources().file(id).path().to_owned();
    let mut named: Vec<String> = Vec::new();
    for w in pp.tokens.windows(3) {
        if let [(Token::MacroName(m), _), (Token::Punct(crate::lex::Punct::LParen), _), (Token::Str { value, .. }, _)] = w
            && session.interner().resolve(*m) == "embed"
        {
            let path = session.interner().resolve(*value).to_owned();
            if !named.contains(&path) {
                named.push(path);
            }
        }
    }
    for path in named {
        let read = crate::pp::EmbedResolver::read(&mut recording, &from, &path).map(|text| {
            let full = from.parent().unwrap_or_else(|| std::path::Path::new(".")).join(&path);
            session.add(full, &text)
        });
        out.documents.push((path, read));
    }
    out.embeds = recording.read;
    out.diagnostics.extend(pp.diagnostics.clone());
    let kind = pp.unit.map(|u| u.kind);
    out.preprocessed = Some(pp);
    out.reached = Some(Stage::Preprocess);
    if out.has_errors() || options.stage == Stage::Preprocess {
        return finish(out, options);
    }

    // decide which `[` opens an annotation before parsing sees any of them
    let marked = crate::mark::mark(&out.preprocessed.as_ref().expect("phase 4 ran").tokens);
    out.diagnostics.extend(marked.diagnostics.clone());
    out.marked = Some(marked);
    out.reached = Some(Stage::Mark);
    if out.has_errors() || options.stage == Stage::Mark {
        return finish(out, options);
    }

    // phase 5
    let marked_tokens = &out.marked.as_ref().expect("the marking pass ran").tokens;
    let eoi = crate::parse::eoi_span(id, marked_tokens);
    let parsed = {
        let cx = crate::parse::Cx::new(session.interner_mut()).with_docs(&out.docs);
        crate::parse::parse_unit(cx, kind, marked_tokens, eoi)
    };
    out.diagnostics.extend(parsed.diagnostics);
    out.unit = Some(parsed.unit);
    out.reached = Some(Stage::Parse);
    out.source = Some(id);
    finish(out, options)
}

/// Phases 6 to 10 over a unit [`front`] parsed cleanly: analysis against the
/// interfaces of its imports, then code generation if asked for.
pub fn back(session: &Session, out: &mut Outcome, options: Options, imports: &[crate::sema::Interface]) {
    let id = out.source.expect("the front half ran");
    // reaching here without errors means the directive was read, so the
    // unit's kind and name are known
    let directive = out.directive().expect("a unit that parsed cleanly has a `#unit` directive");
    // a unit may not take a library unit's name, nor one set aside for a
    // later edition's library
    let file = session.sources().file(id);
    let name = file.unit_name();
    let library = crate::library::is_library(name) || crate::library::RESERVED.contains(&name);
    if library && file.path() != crate::library::path(name) {
        out.diagnostics.push(
            Diagnostic::new(Code::Es20)
                .with_message(format!("a unit cannot be called `{name}`: that name belongs to the library"))
                .at(directive.span)
                .with_note(format!(
                    "a unit's name is its file's name, and `{name}` names a library unit, or one set \
                     aside for a later edition"
                ))
                .with_help(format!("rename `{}`", file.name())),
        );
        return;
    }
    let analysis = analysis_of(session, out, imports, options.static_circuits);
    out.diagnostics.extend(analysis.diagnostics.iter().cloned());
    let succeeded = analysis.succeeded();
    out.analysis = Some(analysis);
    out.reached = Some(Stage::Check);
    if succeeded {
        out.metadata = Some(metadata_of(session, out, out.tir().expect("analysis ran")));
    }
    // a quantum unit has no classical code; phase 9 makes its circuits
    let quantum = out.tir().is_some_and(|t| t.quantum);
    if quantum
        && succeeded
        && let Some(tir) = out.tir()
    {
        let pp = out.preprocessed.as_ref();
        let options = crate::lower::Options {
            conductor: meeting_conductor(out.conductor(), imports),
            reuse: pp.is_none_or(|p| p.qalloc == crate::pp::QAlloc::Reuse),
            stabilizer: pp.is_some_and(|p| p.fragment == crate::pp::Fragment::Stabilizer),
        };
        let interner = session.interner();
        let lowering = crate::lower::lower(tir, interner, options, imports);
        let (records, circuits) = crate::lower::publish(tir, interner, &lowering);
        let uses = crate::meta::record_uses(tir, interner, imports);
        out.diagnostics.extend(lowering.diagnostics);
        out.circuits = lowering.circuits;
        if let Some(meta) = &mut out.metadata {
            meta.interface.records = records;
            meta.interface.circuits = circuits;
            meta.uses.extend(uses);
        }
    }
    if out.has_errors() || quantum || options.stage == Stage::Check || options.stage == Stage::Metadata {
        if options.stage == Stage::Metadata && succeeded {
            out.reached = Some(Stage::Metadata);
        }
        return;
    }

    // phase 10
    generate(session, out, options);
}

/// The conductor a quantum unit of conductor `own` is judged at: the least
/// common multiple of its own and those of the quantum units it imports,
/// whose judgements meet its own. Analysis has refused an import whose
/// conductor meets the unit's above the limit, so the multiple is within it.
fn meeting_conductor(own: u32, imports: &[crate::sema::Interface]) -> u32 {
    let limit = crate::diag::Limit::Conductor.value();
    let n = imports
        .iter()
        .filter(|i| i.quantum)
        .fold(own, |n, i| num_integer::lcm(n, i.conductor.max(1)));
    if n <= limit { n } else { own }
}

/// Analyses the parsed unit `out` against `imports`, changing nothing.
fn analysis_of(session: &Session, out: &Outcome, imports: &[crate::sema::Interface], static_circuits: bool) -> crate::sema::Analysis {
    let id = out.source.expect("the front half ran");
    let directive = out.directive().expect("a unit that parsed cleanly has a `#unit` directive");
    let unit = out.unit.as_ref().expect("phase 5 ran");
    let name = out.identity(session);
    let documents: Vec<crate::sema::Document<'_>> = out
        .documents
        .iter()
        .map(|(path, read)| {
            let read = read.clone().map(|sid| (sid, session.sources().file(sid).spliced()));
            (path.clone(), read)
        })
        .collect();
    let input = crate::sema::Input {
        documents: &documents,
        conductor: out.conductor(),
        name: &name,
        source: id,
        directive: directive.span,
        initializer: directive.initializer.map(|n| (n, directive.span)),
        macros: out.preprocessed.as_ref().map(|p| &p.macros),
        imports,
        static_circuits,
    };
    crate::sema::analyze(unit, input, session.interner())
}

/// The metadata of the parsed unit `out`, analysed as `tir`: what it offers
/// its importers, with its source when they need its bodies.
fn metadata_of(session: &Session, out: &Outcome, tir: &crate::tir::Unit) -> crate::meta::Metadata {
    let id = out.source.expect("the front half ran");
    let mut meta = crate::meta::Metadata::of(tir, out.conductor(), session.interner());
    let this = &out.identity(session);
    meta.interface.imports = dependencies(out, this, session.interner()).into_iter().map(|(n, _)| n).collect();
    meta.interface.any = out.directive().and_then(|d| d.any).map(|a| a.preference);
    let unit = out.unit.as_ref().expect("phase 5 ran");
    // a quantum unit's operators are inlined into the circuits of a quantum
    // unit importing it, so their bodies travel with it
    if meta.interface.quantum || carries_bodies(unit, &meta.interface) {
        let file = session.sources().file(id);
        meta.interface.source = Some(crate::sema::interface::Source {
            path: file.path().display().to_string(),
            text: file.original().to_owned(),
            embeds: out.embeds.clone(),
        });
        meta.interface.ast = Some(std::sync::Arc::new(unit.clone()));
    }
    meta
}

/// What the parsed unit `out` would offer its importers, analysed against
/// `imports`: which may not yet say all they will: for units that import
/// one another, each is analysed against what the others offered last time,
/// until nothing changes. Its diagnostics are not kept. `None` if analysis
/// made nothing to offer.
pub fn provisional_metadata(
    session: &Session,
    out: &Outcome,
    imports: &[crate::sema::Interface],
) -> Option<crate::meta::Metadata> {
    let analysis = analysis_of(session, out, imports, false);
    let tir = analysis.tir.as_ref()?;
    Some(metadata_of(session, out, tir))
}

/// Code generation, which needs LLVM.
#[cfg(not(feature = "llvm"))]
fn generate(_session: &Session, out: &mut Outcome, _options: Options) {
    out.diagnostics.push(
        Diagnostic::new(Code::Tq003)
            .with_message("this `tqc` was built without LLVM, so it cannot generate code")
            .with_help("rebuild it with the `llvm` feature, which is on by default"),
    );
}

/// Code generation, which needs LLVM.
#[cfg(feature = "llvm")]
fn generate(session: &Session, out: &mut Outcome, options: Options) {
    let object = options.stage == Stage::Build;
    // only an object carries metadata; the IR printed for a reader does not
    // need it
    let text = out
        .metadata
        .as_ref()
        .filter(|_| object)
        .map(|m| crate::meta::encode::encode(m, session.interner()));
    let result = {
        let unit = out.tir().expect("analysis succeeded");
        crate::codegen::generate(&crate::codegen::Request {
            unit,
            sources: session.sources(),
            interner: session.interner(),
            opt: options.opt_level,
            object,
            metadata: text.as_deref(),
            entry: options.entry,
        })
    };
    match result {
        Ok(g) => {
            out.llvm_ir = Some(g.ir);
            out.object = g.object;
            out.reached = Some(options.stage);
        }
        Err(e) => out.diagnostics.push(
            Diagnostic::new(Code::Tq011)
                .with_message(format!("code generation failed: {e}"))
                .with_note(
                    "this is a fault in the compiler or its LLVM installation, not in the \
                     program, which analysed cleanly",
                ),
        ),
    }
}

/// Whether a unit's importers may need its function bodies: it declares a
/// generic item, whose instances they make, or a constant function, which
/// they evaluate.
fn carries_bodies(unit: &Unit, iface: &crate::sema::Interface) -> bool {
    let generic = unit.items.iter().any(|item| match &item.node.kind {
        crate::ast::ItemKind::Fn(f) => !f.generics.is_empty(),
        crate::ast::ItemKind::Impl { generics, items, .. } => {
            !generics.is_empty() || items.iter().any(|f| !f.generics.is_empty())
        }
        crate::ast::ItemKind::Struct { generics, .. } | crate::ast::ItemKind::Enum { generics, .. } => {
            !generics.is_empty()
        }
        _ => false,
    });
    generic || iface.fns.iter().any(|f| f.constant)
}

/// An `#embed` resolver that remembers what it read, so that the text can
/// travel with the unit's source.
struct Recording<'r> {
    inner: &'r mut dyn crate::pp::EmbedResolver,
    read: Vec<(String, String)>,
}

impl crate::pp::EmbedResolver for Recording<'_> {
    fn read(&mut self, from: &std::path::Path, path: &str) -> Result<String, String> {
        let text = self.inner.read(from, path)?;
        if !self.read.iter().any(|(p, _)| p == path) {
            self.read.push((path.to_owned(), text.clone()));
        }
        Ok(text)
    }
}

/// Applies the warnings-as-errors option, if it is set.
fn finish(mut out: Outcome, options: Options) -> Outcome {
    if options.deny_warnings {
        for d in &mut out.diagnostics {
            d.severity = crate::diag::Severity::Error;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str, stage: Stage) -> (Session, Outcome) {
        let mut session = Session::new();
        let id = session.add("demo.tq", src);
        let out = compile(
            &mut session,
            id,
            Options {
                stage,
                ..Options::default()
            },
        );
        (session, out)
    }

    fn ok(src: &str) -> Outcome {
        let (_, out) = run(src, Stage::Parse);
        assert!(
            !out.has_errors(),
            "{src:?} failed: {:?}",
            out.diagnostics.iter().map(|d| d.to_string()).collect::<Vec<_>>()
        );
        out
    }

    #[test]
    fn documentation_survives_preprocessing_only_with_its_declaration() {
        let out = ok("//! The unit.\n#unit classical\n/// Kept.\nfn a() { }\n#if 0\n/// Dropped.\nfn b() { }\n#endif\nfn c() { }\n");
        let unit = out.unit.expect("parsed");
        assert_eq!(unit.doc.map(|d| d.text).as_deref(), Some("The unit."));
        let docs: Vec<Option<String>> = unit.items.iter().map(|i| i.node.doc.clone().map(|d| d.text)).collect();
        assert_eq!(docs, [Some("Kept.".to_owned()), None], "`c` does not take `b`'s documentation");
    }

    #[test]
    fn a_minimal_classical_unit_compiles_to_a_tree() {
        let out = ok("#unit classical\nfn main() -> i32 { 0 }\n");
        assert_eq!(out.reached, Some(Stage::Parse));
        assert_eq!(out.unit_kind(), Some(UnitKind::Classical));
        let unit = out.unit.unwrap();
        assert_eq!(unit.functions().count(), 1);
        assert!(unit.is_classical());
    }

    #[test]
    fn a_quantum_unit_is_recorded_as_quantum() {
        let out = ok("#unit quantum\nfn f() { }\n");
        assert_eq!(out.unit_kind(), Some(UnitKind::Quantum));
        assert!(out.unit.unwrap().is_quantum());
    }

    #[test]
    fn stopping_early_returns_only_what_that_stage_produced() {
        let src = "#unit classical\nfn main() -> i32 { 0 }\n";
        let (_, toks) = run(src, Stage::Tokens);
        assert_eq!(toks.reached, Some(Stage::Tokens));
        assert!(!toks.tokens.is_empty());
        assert!(toks.preprocessed.is_none());
        assert!(toks.unit.is_none());

        let (_, pp) = run(src, Stage::Preprocess);
        assert_eq!(pp.reached, Some(Stage::Preprocess));
        assert!(pp.preprocessed.is_some());
        assert!(pp.unit.is_none());
    }

    #[test]
    fn macros_are_expanded_before_parsing() {
        let out = ok("#unit classical\n#define LIMIT 64\nlet N: const u32 = LIMIT;\n");
        let unit = out.unit.unwrap();
        assert_eq!(unit.len(), 1, "the #define is not an item");
    }

    #[test]
    fn a_missing_unit_directive_stops_before_parsing() {
        let (_, out) = run("fn main() -> i32 { 0 }\n", Stage::Parse);
        assert!(out.reported(Code::Eu01));
        assert_eq!(
            out.reached,
            Some(Stage::Preprocess),
            "translation stops at the phase that failed"
        );
        assert!(out.unit.is_none());
    }

    #[test]
    fn a_pragma_warning_does_not_stop_translation() {
        // an unknown pragma is a warning and translation
        // continues, which is the whole point of EA03 being a warning
        let out = ok("#unit classical\n#pragma nonesuch(1)\nfn main() -> i32 { 0 }\n");
        assert!(out.reported(Code::Ea03));
        assert_eq!(out.warning_count(), 1);
        assert_eq!(out.error_count(), 0);
        assert_eq!(out.reached, Some(Stage::Parse));
    }

    #[test]
    fn deny_warnings_turns_a_diagnosed_construct_into_a_failure() {
        let mut session = Session::new();
        let id = session.add("demo.tq", "#unit classical\n#pragma nonesuch(1)\n");
        let out = compile(
            &mut session,
            id,
            Options {
                deny_warnings: true,
                ..Options::default()
            },
        );
        assert!(out.has_errors());
    }

    #[test]
    fn the_conductor_and_pragmas_survive_to_the_outcome() {
        let out = ok("#unit quantum\n#pragma conductor(24)\nfn f() { }\n");
        let pp = out.preprocessed.unwrap();
        assert_eq!(pp.conductor, 24);
    }

    #[test]
    fn check_runs_analysis_and_keeps_the_typed_unit() {
        let (_, out) = run(
            "#unit classical\nfn main() -> i32 { let x = 40; x + 2 }\n",
            Stage::Check,
        );
        assert!(out.succeeded(Stage::Check), "{:?}", out.diagnostics);
        let tir = out.tir().expect("analysis ran");
        assert_eq!(tir.name, "demo", "the unit is named after its file");
        assert!(tir.main.is_some());
    }

    #[test]
    fn an_analysis_error_stops_the_pipeline_at_check() {
        let (_, out) = run("#unit classical\nfn main() -> i32 { true }\n", Stage::Build);
        assert!(out.reported(Code::Es06));
        assert_eq!(out.reached, Some(Stage::Check));
        assert!(out.object.is_none());
    }

    #[test]
    fn a_quantum_unit_becomes_circuits_and_no_object() {
        let (_, out) = run("#unit quantum\n[entry]\nfn f() -> bool { let q: [qubit; 1] = prep |1>; let m = measure q; m[0] }\n", Stage::Build);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.metadata.as_ref().is_some_and(|m| m.interface.quantum));
        assert_eq!(out.circuits.len(), 1);
        assert!(out.circuits[0].entry);
        assert_eq!(out.circuits[0].circuit.bits, 1);
        assert!(out.object.is_none(), "a quantum unit has no classical code");
    }

    #[test]
    fn a_lexical_error_stops_before_preprocessing() {
        let (_, out) = run("#unit classical\nlet s = \"unterminated\n", Stage::Parse);
        assert!(out.has_errors());
        assert_eq!(out.reached, Some(Stage::Tokens));
    }

    #[test]
    fn a_syntax_error_is_reported_as_es02() {
        let (_, out) = run("#unit classical\nfn main( { }\n", Stage::Parse);
        assert!(out.has_errors());
        assert!(out.reported(Code::Es02));
    }

    #[test]
    fn succeeded_reports_both_cleanliness_and_progress() {
        let out = ok("#unit classical\nfn main() -> i32 { 0 }\n");
        assert!(out.succeeded(Stage::Parse));
        assert!(!out.succeeded(Stage::Build), "it never reached that far");
    }

    #[test]
    fn library_units_that_depend_on_one_another_are_analyzed_together() {
        // two library units that import each other, beside the real `core`
        let source = |name: &str| match name {
            "ping" => Some(
                "#unit classical\nimport pong;\nstruct P { n: i32 }\n\
                 fn f(n: i32) -> i32 { if n == 0 { 0 } else { pong::g(n - 1) } }\n",
            ),
            "pong" => Some("#unit classical\nimport ping;\nfn g(n: i32) -> i32 { ping::f(n) }\nfn h(p: ping::P) -> i32 { p.n }\n"),
            other => crate::library::source(other),
        };
        let mut session = Session::new();
        let (graph, _) = library_graph(&mut session, "ping", &source);
        let together = graph.groups().into_iter().find(|g| graph.is_cycle(g)).expect("a cycle");
        let mut names: Vec<&str> = together.iter().map(|&i| graph.name(i)).collect();
        names.sort_unstable();
        assert_eq!(names, ["ping", "pong"]);
        // each is analysed once, against the other, and kept for the session
        let mut session = Session::new();
        assert!(library_interface_from(&mut session, "ping", &source).is_some());
        assert!(session.library.contains_key("pong"));
        assert!(session.library.contains_key("core"));
        // a unit using either is analysed against both
        let id = session.add("demo.tq", "#unit classical\nimport ping;\nfn main() -> i32 { ping::f(3) }\n");
        let out = compile(&mut session, id, Options { stage: Stage::Check, ..Options::default() });
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
    }

    #[test]
    fn a_unit_of_a_cycle_sees_what_the_others_import_from_outside() {
        // `pong` makes an instance of `ping`'s generic, which opens `ping`'s
        // declarations and so its import of `tcon`, which `pong` lacks
        let source = |name: &str| match name {
            "ping" => Some(
                "#unit classical\nimport pong;\nimport tcon;\n\
                 fn same<T>(x: T) -> T { x }\nfn f(n: i32) -> i32 { if n == 0 { 0 } else { pong::g(n - 1) } }\n",
            ),
            "pong" => Some("#unit classical\nimport ping;\nfn g(n: i32) -> i32 { ping::same(ping::f(n)) }\n"),
            other => crate::library::source(other),
        };
        let mut session = Session::new();
        assert!(library_interface_from(&mut session, "ping", &source).is_some());
        assert!(session.library.contains_key("pong"), "`pong` analysed cleanly");
    }

    #[test]
    fn a_library_unit_analyzed_already_is_not_analyzed_again() {
        let mut session = Session::new();
        assert!(library_interface(&mut session, "core").is_some());
        let (graph, outs) = library_graph(&mut session, "str", &crate::library::source);
        // `core` is in the graph, for `str` and `tcon` to find, with nothing
        // left to analyse
        let core = graph.index("core").expect("core");
        assert!(outs[core].unit.is_none());
        assert!(library_interface(&mut session, "str").is_some());
    }
}
