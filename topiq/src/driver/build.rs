//! Building a whole program: every unit compiled to an object, then linked.
//!
//! [`super::compile`] translates one unit. A program is several, and some of
//! its checks only make sense for the whole: whether every unit one imports is
//! present and still what it was compiled against, and whether it has exactly
//! one `main`. This module loads the program (the files given and every unit
//! they import, see [`super::program`]), compiles each unit given as source,
//! writes each object as `<unit>.tcu`, checks the program, and hands every
//! object, including those of units compiled earlier, to the linker, with
//! one more made for the whole program: the function that runs the units'
//! initialisers, and the program's descriptor for `module::load`.
//!
//! The same objects make one of four things: an executable started at
//! `main`; a module, a library a program loads while it runs, which needs
//! no `main`; a circuit archive, `.tqar`, of the program's quantum
//! circuits alone; or, for [`test()`], an executable that runs the
//! program's `[test]` functions one at a time, each in a process of its own.
//!
//! A quantum unit's output is its `.tqu`: its metadata, holding the circuit
//! of each `[entry]` operator and the judgement record of each operator
//! with program linkage. An executable or module embeds every such circuit
//! (it is a hybrid image) and a classical unit's handle to one refers to
//! the circuit embedded.
//!
//! Every unit is loaded into one [`Session`], so their spans never collide and
//! a diagnostic about one unit can point into another's file.

use std::io;
use std::path::{Path, PathBuf};

use crate::diag::Diagnostic;
use crate::link::{self, Linked, ToolError};
use crate::meta::Metadata;

use super::program::{self, LoadError, MemberKind};
use super::{OptLevel, Options, Outcome, Session, Stage};

/// How to build a program.
#[derive(Clone, Debug, Default)]
pub struct BuildOptions {
    /// How hard code generation optimises.
    pub opt_level: OptLevel,
    /// Where the objects go.
    pub out_dir: PathBuf,
    /// The executable to produce. `None` means `<first unit>.exe` in
    /// [`BuildOptions::out_dir`].
    pub executable: Option<PathBuf>,
    /// Whether to link at all, or stop at the objects.
    pub link: bool,
    /// A linker to use instead of the installed one.
    pub linker: Option<PathBuf>,
    /// Directories to look for imported units in, after those of the files
    /// given.
    pub unit_path: Vec<PathBuf>,
    /// Whether to build the program to run its `[test]` functions, one per
    /// process, rather than its `main`.
    pub tests: bool,
    /// Whether to build a module, a library a program loads while it runs
    /// with `module::load`, rather than a program. It needs no `main`, and
    /// is written as `<first unit>.dll` unless named.
    pub module: bool,
    /// Whether to write a circuit archive rather than an executable: the
    /// circuits of the program's `[entry]` operators, each with its
    /// OpenQASM 3 and its judgement, as `<first unit>.tqar` unless named. It
    /// needs no `main`, and its circuits must be static.
    pub archive: bool,
    /// The kinds `--kind` chooses for the `#unit any` units given.
    pub kinds: super::KindArgs,
}

/// What building a program produced.
#[derive(Debug)]
pub struct Program {
    /// Every source, for rendering diagnostics.
    pub session: Session,
    /// The translation of each unit compiled from source.
    pub units: Vec<Outcome>,
    /// Each unit compiled earlier, with its object and metadata.
    pub compiled: Vec<(PathBuf, Metadata)>,
    /// Diagnostics about the program as a whole.
    pub diagnostics: Vec<Diagnostic>,
    /// The objects linked.
    pub objects: Vec<PathBuf>,
    /// Each quantum unit's `.tqu`, whose circuits the image embeds.
    pub quantum: Vec<PathBuf>,
    /// The executable.
    pub executable: Option<PathBuf>,
    /// The circuit archive.
    pub archive: Option<PathBuf>,
    /// For a program built to run its tests, each test's name, as
    /// `unit::function`, in the order the executable numbers them.
    pub tests: Vec<String>,
}

impl Program {
    /// Every diagnostic, the units' first, then the program's.
    pub fn all_diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.units
            .iter()
            .flat_map(|u| u.diagnostics.iter())
            .chain(&self.diagnostics)
    }

    /// Whether anything reported an error.
    pub fn has_errors(&self) -> bool {
        self.all_diagnostics().any(Diagnostic::is_error)
    }
}

/// Why a build could not run to completion, as opposed to finding faults in
/// the program (those are diagnostics).
#[derive(Debug)]
pub enum BuildError {
    /// A source could not be read, or an object not written.
    Io {
        /// The file concerned.
        path: PathBuf,
        /// What went wrong.
        error: io::Error,
    },
    /// A compiled unit given, or found on the unit path, could not be used.
    Load(LoadError),
    /// The linker is missing or failed.
    Tool(ToolError),
    /// Making the object that starts the program failed, which is a fault in
    /// the compiler or its LLVM installation.
    Codegen(crate::codegen::CodegenError),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            BuildError::Load(e) => write!(f, "{e}"),
            BuildError::Tool(e) => write!(f, "{e}"),
            BuildError::Codegen(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BuildError {}

/// Makes an I/O failure about `path` a [`BuildError`].
fn io(path: &Path) -> impl FnOnce(io::Error) -> BuildError + '_ {
    move |error| BuildError::Io {
        path: path.to_owned(),
        error,
    }
}

impl From<LoadError> for BuildError {
    fn from(e: LoadError) -> BuildError {
        match e {
            LoadError::Io { path, error } => BuildError::Io { path, error },
            other => BuildError::Load(other),
        }
    }
}

/// Builds a program from its units: source files, and objects of units
/// compiled earlier.
///
/// # Errors
///
/// A [`BuildError`] when a file cannot be read or written, a compiled unit's
/// metadata cannot be read, or the linker cannot do its work. Faults in the
/// program itself are not errors here: they are in the returned
/// [`Program`]'s diagnostics, and no executable is linked.
pub fn build(inputs: &[PathBuf], opts: &BuildOptions) -> Result<Program, BuildError> {
    let options = Options {
        stage: Stage::Build,
        opt_level: opts.opt_level,
        entry: !(opts.tests || opts.module || opts.archive),
        static_circuits: opts.archive,
        ..Options::default()
    };
    let loaded = program::load_as(inputs, &opts.unit_path, options, &opts.kinds)?;
    let first_name = loaded.members.first().map(|m| m.name.clone());
    let mut program = Program {
        session: loaded.session,
        units: Vec::new(),
        compiled: Vec::new(),
        diagnostics: loaded.diagnostics,
        objects: Vec::new(),
        quantum: Vec::new(),
        executable: None,
        archive: None,
        tests: Vec::new(),
    };
    for m in loaded.members {
        match m.kind {
            MemberKind::Source(out) => program.units.push(*out),
            MemberKind::Compiled { path, metadata } => program.compiled.push((path, *metadata)),
        }
    }
    if program.has_errors() {
        return Ok(program);
    }

    for u in &program.units {
        let tir = u.tir().expect("a unit that built has been analysed");
        // a library unit's object is kept apart from the program's own
        let dir = if crate::library::source(&tir.name).is_some() {
            opts.out_dir.join(".topiq-lib")
        } else {
            opts.out_dir.clone()
        };
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        // a quantum unit contributes circuits, not classical code: its
        // `.tqu` is its metadata, which holds them
        if tir.quantum {
            let path = dir.join(format!("{}.tqu", super::written(&tir.name)));
            let meta = u.metadata.as_ref().expect("a unit that built has metadata");
            let text = crate::meta::encode::encode(meta, program.session.interner());
            std::fs::write(&path, text).map_err(io(&path))?;
            program.quantum.push(path);
            continue;
        }
        let path = dir.join(format!("{}.tcu", super::written(&tir.name)));
        let bytes = u.object.as_ref().expect("a unit that built has an object");
        std::fs::write(&path, bytes).map_err(io(&path))?;
        program.objects.push(path);
    }
    for (path, meta) in &program.compiled {
        if meta.interface.quantum {
            program.quantum.push(path.clone());
        } else {
            program.objects.push(path.clone());
        }
    }
    if !opts.link {
        return Ok(program);
    }

    let linked: Vec<Linked<'_>> = program
        .units
        .iter()
        .map(|u| {
            let tir = u.tir().expect("analysed");
            Linked {
                meta: u.metadata.as_ref().expect("a unit that built has metadata"),
                main_span: tir.main.map(|id| tir.func(id).span),
            }
        })
        .chain(program.compiled.iter().map(|(_, m)| Linked {
            meta: m,
            main_span: None,
        }))
        .collect();
    // what is written, unless named: `<first unit>.<extension>`
    let output = |extension: &str| {
        let name = first_name.as_deref().unwrap_or("program");
        opts.executable.clone().unwrap_or_else(|| opts.out_dir.join(format!("{name}.{extension}")))
    };
    let needs_main = !(opts.tests || opts.module || opts.archive);
    let diagnostics = link::check_program_for(&linked, program.session.interner(), needs_main);
    let inits = link::startup_order(&linked).unwrap_or_default();
    program.diagnostics.extend(diagnostics);
    if program.has_errors() {
        return Ok(program);
    }
    if opts.archive {
        let metas: Vec<&Metadata> = linked.iter().map(|l| l.meta).collect();
        let (text, refusals) = link::archive(&metas, program.session.interner());
        program.diagnostics.extend(refusals);
        if program.has_errors() {
            return Ok(program);
        }
        let path = output("tqar");
        std::fs::write(&path, text).map_err(io(&path))?;
        program.archive = Some(path);
        return Ok(program);
    }

    // the initialisers, run in order before `main` or a test, from an object
    // made now that the whole program is known
    let tests = opts.tests.then(|| tests_of(&program));
    if let Some(t) = &tests {
        program.tests = t.iter().map(|(name, _)| name.clone()).collect();
    }
    let image = image_of(&program, opts.module);
    let startup = crate::codegen::startup_object(&inits, tests.as_deref(), &image).map_err(BuildError::Codegen)?;
    let lib = opts.out_dir.join(".topiq-lib");
    std::fs::create_dir_all(&lib).map_err(io(&lib))?;
    let path = lib.join("startup.obj");
    std::fs::write(&path, startup).map_err(io(&path))?;
    program.objects.push(path);

    let exe = output(if opts.module { "dll" } else { "exe" });
    if opts.module {
        link::link_module(&program.objects, &exe, opts.linker.as_deref()).map_err(BuildError::Tool)?;
    } else {
        link::link(&program.objects, &exe, opts.linker.as_deref()).map_err(BuildError::Tool)?;
    }
    program.executable = Some(exe);
    Ok(program)
}

/// What the program or module records about itself for `module::load`: its
/// types with their layout digests, the methods its units add to other
/// units' types, and the circuits of its quantum units' `[entry]`
/// operators, from every unit's metadata; and for a module, the functions
/// its own units offer, library units aside.
fn image_of(program: &Program, module: bool) -> crate::codegen::image::Image {
    use crate::codegen::image::{Image, Symbol, Type};
    use crate::meta::check;
    let interner = program.session.interner();
    let metas = program
        .units
        .iter()
        .filter_map(|u| u.metadata.as_ref())
        .chain(program.compiled.iter().map(|(_, m)| m));
    let mut image = Image {
        module,
        ..Image::default()
    };
    let mut seen = std::collections::HashSet::new();
    for m in metas {
        let iface = &m.interface;
        let types = &iface.types;

        // the types the unit declares, each once
        for (i, a) in types.adts().iter().enumerate() {
            if a.origin.as_deref() != Some(m.unit.as_str()) {
                continue;
            }
            let id = crate::tir::AdtId(i as u32);
            let name = types.qualified(crate::tir::Ty::Adt(id), interner, &m.unit);
            if seen.insert(name.clone()) {
                let digest = check::adt_digest(types, id, &m.unit, interner);
                image.types.push(Type { name, digest });
            }
        }

        // its entry circuits
        for c in &iface.circuits {
            let name = interner.resolve(c.name);
            image.circuits.push(crate::codegen::image::Circuit {
                name: format!("{}::{name}", m.unit),
                sig: c.sig.clone(),
                document: c.document.clone(),
                handle: crate::codegen::mangle::global(&m.unit, name),
            });
        }

        // the methods it adds to other units' types
        for f in &iface.fns {
            let Some(method) = f.method else { continue };
            let owner = types.adt(method.owner);
            if owner.origin.as_deref() == Some(m.unit.as_str()) {
                continue;
            }
            let name = interner.resolve(f.name);
            let shown = if method.operator { format!("${name}") } else { name.to_owned() };
            image.methods.push((types.qualified(crate::tir::Ty::Adt(method.owner), interner, &m.unit), shown));
        }

        // for a module, the plain functions it offers
        if !module || crate::library::source(&m.unit).is_some() {
            continue;
        }
        for f in &iface.fns {
            if f.method.is_some() || f.constant || f.entry || f.linkage != crate::ast::Linkage::Program {
                continue;
            }
            let name = interner.resolve(f.name);
            image.symbols.push(Symbol {
                name: format!("{}::{name}", m.unit),
                sig: types.qualified_signature(&f.params, f.ret, interner, &m.unit),
                symbol: f
                    .symbol
                    .clone()
                    .unwrap_or_else(|| crate::codegen::mangle::function(&m.unit, name)),
            });
        }
    }
    image
}

/// Builds and runs a program whose main unit is `source`, returning its exit
/// status.
///
/// The program runs with the caller's standard streams, so its output (and
/// the line an abort writes) appear as it would when run by hand.
///
/// # Errors
///
/// As [`build`], or if the executable cannot be started.
pub fn run(source: &Path, opts: &BuildOptions) -> Result<(Program, Option<i32>), BuildError> {
    let program = build(&[source.to_owned()], opts)?;
    let status = run_built(&program)?;
    Ok((program, status))
}

/// Runs the executable `program` built, if it built one, giving its exit
/// status: `None` when there is no executable, or when the program was ended
/// by something other than an exit.
///
/// # Errors
///
/// A [`BuildError`] when the executable cannot be started.
pub fn run_built(program: &Program) -> Result<Option<i32>, BuildError> {
    let Some(exe) = &program.executable else {
        return Ok(None);
    };
    let status = std::process::Command::new(exe).status().map_err(io(exe))?;
    Ok(status.code())
}

/// The `[test]` functions of the units compiled from source, library units
/// aside: each one's name as `unit::function`, and its symbol.
fn tests_of(program: &Program) -> Vec<(String, String)> {
    let interner = program.session.interner();
    let mut out = Vec::new();
    for u in &program.units {
        let tir = u.tir().expect("analysed");
        if tir.quantum || crate::library::source(&tir.name).is_some() {
            continue;
        }
        for f in tir.fns.iter().filter(|f| f.attrs.test && f.origin.is_none() && !f.closure) {
            let name = interner.resolve(f.name);
            let symbol = f
                .attrs
                .symbol
                .clone()
                .unwrap_or_else(|| crate::codegen::mangle::function(&tir.name, name));
            out.push((format!("{}::{name}", tir.name), symbol));
        }
    }
    out
}

/// How one test went.
#[derive(Clone, Debug)]
pub struct TestResult {
    /// The test (i.e. `unit::function`).
    pub name: String,
    /// Whether it returned, rather than aborting.
    pub passed: bool,
    /// The status its process exited with.
    pub status: Option<i32>,
    /// What it wrote, to standard output and then standard error.
    pub output: String,
}

/// Builds the program `files` make to run its tests, then runs each test in
/// a process of its own, so that one that aborts fails alone.
///
/// # Errors
///
/// As [`build`], or if the executable cannot be started.
pub fn test(files: &[PathBuf], opts: &BuildOptions) -> Result<(Program, Vec<TestResult>), BuildError> {
    let program = build_tests(files, opts)?;
    let results = run_tests(&program)?;
    Ok((program, results))
}

/// Builds the program made of `files` to run its `[test]` functions, as
/// [`test`](fn@test) does before running them.
///
/// # Errors
///
/// As for [`build`].
pub fn build_tests(files: &[PathBuf], opts: &BuildOptions) -> Result<Program, BuildError> {
    let opts = BuildOptions {
        tests: true,
        link: true,
        ..opts.clone()
    };
    build(files, &opts)
}

/// Runs each `[test]` function of `program`, built by [`build_tests`], in a
/// process of its own; none if it built no executable.
///
/// # Errors
///
/// A [`BuildError`] when the executable cannot be started.
pub fn run_tests(program: &Program) -> Result<Vec<TestResult>, BuildError> {
    let Some(exe) = program.executable.clone() else {
        return Ok(Vec::new());
    };
    let mut results = Vec::new();
    for (i, name) in program.tests.iter().enumerate() {
        let out = std::process::Command::new(&exe).arg(i.to_string()).output().map_err(io(&exe))?;
        let mut output = String::from_utf8_lossy(&out.stdout).into_owned();
        output.push_str(&String::from_utf8_lossy(&out.stderr));
        results.push(TestResult {
            name: name.clone(),
            passed: out.status.success(),
            status: out.status.code(),
            output,
        });
    }
    Ok(results)
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

    fn opts(dir: &Path) -> BuildOptions {
        BuildOptions {
            out_dir: dir.to_owned(),
            link: true,
            ..BuildOptions::default()
        }
    }

    #[test]
    fn each_test_runs_alone_after_the_initializers() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(
            dir.path(),
            "checks.tq",
            "#unit classical(setup)\nlet BASE: i64;\nfn setup() { BASE = 40; }\n\
             [test]\nfn passes() { if BASE + 2 != 42 { panic(\"wrong\"); } }\n\
             [test]\nfn aborts() { panic(\"no\"); }\n\
             [test]\nfn passes_too() { }\n",
        );
        let (program, results) = test(&[src], &opts(dir.path())).unwrap();
        assert!(!program.has_errors(), "a program of tests needs no `main`");
        let got: Vec<(&str, bool)> = results.iter().map(|r| (r.name.as_str(), r.passed)).collect();
        assert_eq!(got, [("checks::passes", true), ("checks::aborts", false), ("checks::passes_too", true)]);
        assert!(results[1].output.contains("no"), "{}", results[1].output);
    }

    #[test]
    fn a_program_builds_runs_and_exits_with_mains_result() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(
            dir.path(),
            "answer.tq",
            "#unit classical\nfn main() -> i32 { 6 * 7 }\n",
        );
        let (program, status) = run(&src, &opts(dir.path())).unwrap();
        assert!(!program.has_errors());
        assert!(dir.path().join("answer.tcu").exists(), "the object is kept");
        assert_eq!(status, Some(42));
    }

    #[test]
    fn a_failed_check_aborts_with_status_101() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(
            dir.path(),
            "boom.tq",
            "#unit classical\nfn f(x: u8) -> u8 { x + 250 }\nfn main() -> i32 { f(10) as i32 }\n",
        );
        let (_, status) = run(&src, &opts(dir.path())).unwrap();
        assert_eq!(status, Some(101));
    }

    #[test]
    fn a_program_without_main_is_not_linked() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(dir.path(), "lib.tq", "#unit classical\nfn f() { }\n");
        let program = build(&[src], &opts(dir.path())).unwrap();
        assert!(program.executable.is_none());
        assert!(program.diagnostics.iter().any(|d| d.code == Code::El05));
    }

    #[test]
    fn a_unit_with_errors_writes_no_object() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(dir.path(), "bad.tq", "#unit classical\nfn main() -> i32 { true }\n");
        let program = build(&[src], &opts(dir.path())).unwrap();
        assert!(program.has_errors());
        assert!(program.objects.is_empty());
        assert!(!dir.path().join("bad.tcu").exists());
    }

    #[test]
    fn stopping_before_the_link_keeps_the_objects() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(dir.path(), "part.tq", "#unit classical\nfn f() -> i32 { 1 }\n");
        let program = build(
            &[src],
            &BuildOptions {
                link: false,
                ..opts(dir.path())
            },
        )
        .unwrap();
        assert!(!program.has_errors(), "no main is only a problem when linking");
        assert_eq!(program.objects.len(), 2, "the unit's, and the library's `core`");
        assert!(program.executable.is_none());
    }

    #[test]
    fn a_missing_source_is_an_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = build(&[dir.path().join("nope.tq")], &opts(dir.path())).unwrap_err();
        assert!(matches!(err, BuildError::Io { .. }));
    }

    #[test]
    fn an_object_carries_metadata_that_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let src = write(dir.path(), "geo.tq", "#unit classical\nstruct P { x: i32 }\nfn f() -> i32 { 1 }\n");
        let program = build(
            &[src],
            &BuildOptions {
                link: false,
                ..opts(dir.path())
            },
        )
        .unwrap();
        let bytes = std::fs::read(&program.objects[0]).unwrap();
        let text = crate::meta::object::read_section(&bytes).unwrap();
        assert!(text.starts_with("Metadata {"), "{text}");
        let mut i = crate::intern::Interner::new();
        let m = crate::meta::decode::decode(&text, &mut i).unwrap();
        assert_eq!(m.unit, "geo");
    }

    #[test]
    fn a_unit_compiled_earlier_is_linked_from_its_object() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join("lib");
        std::fs::create_dir(&lib).unwrap();
        let geo = write(&lib, "geo.tq", "#unit classical\nfn seven() -> i32 { 7 }\n");
        build(
            &[geo],
            &BuildOptions {
                link: false,
                ..opts(&lib)
            },
        )
        .unwrap();
        std::fs::remove_file(lib.join("geo.tq")).unwrap();
        let app = write(dir.path(), "app.tq", "#unit classical\nimport geo;\nfn main() -> i32 { geo::seven() * 6 }\n");
        let (program, status) = run(
            &app,
            &BuildOptions {
                unit_path: vec![lib],
                ..opts(dir.path())
            },
        )
        .unwrap();
        let reported: Vec<String> = program.all_diagnostics().map(|d| d.to_string()).collect();
        assert!(!program.has_errors(), "{reported:?}");
        assert_eq!(program.compiled.len(), 1);
        assert_eq!(status, Some(42));
    }
}
