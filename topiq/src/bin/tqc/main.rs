//! `tqc`, the Topiq compiler driver.
//!
//! A thin front end over [`topiq::driver`]. Everything it does is available as
//! a library function, so a tool embedding the crate behaves the same way.
//!
//! # Exit status
//!
//! A Topiq program that aborts exits with status 101. That is a
//! property of a program Topiq compiles, not of the compiler. `tqc` itself
//! follows the ordinary convention: 0 when the translation it was asked for
//! succeeded, 1 when it did not.

mod cli;

use std::io::{IsTerminal, Write};
use std::process::ExitCode;

use clap::Parser as _;
use topiq::diag::render::{self, Style};
use topiq::diag::{Code, Phase};
use topiq::driver::{KindArgs, Options, Session, Stage, compile};
use topiq::pp::{KindChoice, UnitKind};
use topiq::lex::token::spell;

use cli::{Cli, ColorChoice, Command};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("tqc: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Runs a command, returning whether it succeeded.
fn run(cli: Cli) -> std::io::Result<bool> {
    match cli.command {
        Command::Diagnostics { grep } => {
            list_diagnostics(grep.as_deref());
            Ok(true)
        }
        Command::Doc {
            entry,
            output,
            unit_path,
            kind,
            document_private,
            color,
        } => {
            let Some(kinds) = kind_args(&kind, std::slice::from_ref(&entry)) else { return Ok(false) };
            let options = topiq::doc::DocOptions {
                out_dir: output,
                private: document_private,
            };
            document(&entry, &unit_path, &kinds, &options, color)
        }
        Command::Check {
            files,
            unit_path,
            kind,
            deny_warnings,
            color,
        } => {
            let Some(kinds) = kind_args(&kind, &paths(&files)) else { return Ok(false) };
            translate(&files, &unit_path, &kinds, Stage::Check, deny_warnings, color, None)
        }
        Command::Emit {
            files,
            stage,
            unit_path,
            kind,
            deny_warnings,
            color,
        } => {
            let Some(kinds) = kind_args(&kind, &paths(&files)) else { return Ok(false) };
            translate(&files, &unit_path, &kinds, stage.into(), deny_warnings, color, Some(stage))
        }
        #[cfg(feature = "llvm")]
        Command::Build {
            files,
            unit_path,
            kind,
            output,
            out_dir,
            opt,
            no_link,
            module,
            archive,
            linker,
            color,
        } => {
            let Some(kinds) = kind_args(&kind, &files) else { return Ok(false) };
            build(
                &files,
                topiq::driver::build::BuildOptions {
                    opt_level: opt.into(),
                    out_dir: out_dir.unwrap_or_else(|| std::path::PathBuf::from(".")),
                    executable: output,
                    link: !no_link,
                    linker,
                    unit_path,
                    tests: false,
                    module,
                    archive,
                    kinds,
                },
                color,
            )
        }
        #[cfg(feature = "llvm")]
        Command::Run {
            file,
            unit_path,
            kind,
            opt,
            linker,
            color,
        } => {
            let Some(kinds) = kind_args(&kind, std::slice::from_ref(&file)) else { return Ok(false) };
            run_program(&file, unit_path, kinds, opt.into(), linker, color)
        }
        #[cfg(feature = "llvm")]
        Command::Test {
            files,
            unit_path,
            kind,
            opt,
            linker,
            color,
        } => {
            let Some(kinds) = kind_args(&kind, &files) else { return Ok(false) };
            run_tests(&files, unit_path, kinds, opt.into(), linker, color)
        }
        #[cfg(not(feature = "llvm"))]
        Command::Build { .. } | Command::Run { .. } | Command::Test { .. } => {
            eprintln!("tqc: this build was compiled without LLVM, so it cannot generate code");
            Ok(false)
        }
    }
}

/// The files `files` names.
fn paths(files: &[String]) -> Vec<std::path::PathBuf> {
    files.iter().map(std::path::PathBuf::from).collect()
}

/// The kinds `--kind` gives for the `#unit any` units among `given`, or
/// `None`, having said why, when one is not of the forms `classical`,
/// `quantum` and `NAME=KIND`, or names no unit given.
fn kind_args(values: &[String], given: &[std::path::PathBuf]) -> Option<KindArgs> {
    let kind = |w: &str| match w {
        "classical" => Some(UnitKind::Classical),
        "quantum" => Some(UnitKind::Quantum),
        _ => None,
    };
    let mut args = KindArgs::default();
    for v in values {
        let (name, word) = match v.split_once('=') {
            Some((name, word)) => (Some(name), word),
            None => (None, v.as_str()),
        };
        let Some(k) = kind(word) else {
            eprintln!("tqc: --kind takes `classical` or `quantum`, or `NAME=classical` or `NAME=quantum`, not `{v}`");
            return None;
        };
        match name {
            None if args.all.is_some_and(|a| a != k) => {
                eprintln!("tqc: --kind gives every unit two kinds");
                return None;
            }
            None => args.all = Some(k),
            Some(name) if !given.iter().any(|p| p.file_stem().is_some_and(|s| s == name)) => {
                eprintln!("tqc: --kind names `{name}`, which is not a unit given");
                return None;
            }
            Some(name) => args.named.push((name.to_owned(), k)),
        }
    }
    Some(args)
}

/// `tqc build`: compiles every unit and links the program.
#[cfg(feature = "llvm")]
fn build(
    files: &[std::path::PathBuf],
    opts: topiq::driver::build::BuildOptions,
    color: Option<ColorChoice>,
) -> std::io::Result<bool> {
    let Some(program) = reported(topiq::driver::build::build(files, &opts)) else {
        return Ok(false);
    };
    report(program.all_diagnostics(), program.session.sources(), style_for(color))?;
    if let Some(exe) = program.executable.as_ref().or(program.archive.as_ref()) {
        eprintln!("tqc: built {}", exe.display());
    }
    Ok(!program.has_errors())
}

/// `tqc run`: builds a program in a temporary directory and runs it, then
/// exits with the program's own status.
#[cfg(feature = "llvm")]
fn run_program(
    file: &std::path::Path,
    unit_path: Vec<std::path::PathBuf>,
    kinds: KindArgs,
    opt: topiq::driver::OptLevel,
    linker: Option<std::path::PathBuf>,
    color: Option<ColorChoice>,
) -> std::io::Result<bool> {
    use topiq::driver::build::{BuildOptions, build, run_built};
    let dir = scratch("run")?;
    let opts = BuildOptions {
        opt_level: opt,
        out_dir: dir.clone(),
        link: true,
        linker,
        unit_path,
        kinds,
        ..BuildOptions::default()
    };
    // what translating it found is said before the program runs, so that it
    // does not follow the program's own output
    let Some(program) = reported(build(&[file.to_owned()], &opts)) else {
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(false);
    };
    report(program.all_diagnostics(), program.session.sources(), style_for(color))?;
    let result = run_built(&program);
    let _ = std::fs::remove_dir_all(&dir);
    let Some(status) = reported(result) else {
        return Ok(false);
    };
    match status {
        Some(code) => std::process::exit(code),
        // the program was not built, or it was ended by something other than
        // an exit (which the diagnostics above, or the system, will have said)
        None => Ok(false),
    }
}

/// `tqc test`: builds the program's tests in a temporary directory, runs
/// each, and reports which passed. Succeeds when every test passes.
#[cfg(feature = "llvm")]
fn run_tests(
    files: &[std::path::PathBuf],
    unit_path: Vec<std::path::PathBuf>,
    kinds: KindArgs,
    opt: topiq::driver::OptLevel,
    linker: Option<std::path::PathBuf>,
    color: Option<ColorChoice>,
) -> std::io::Result<bool> {
    use topiq::driver::build::{BuildOptions, build_tests, run_tests};

    // build the tests, and say what translating them found before they run
    let dir = scratch("test")?;
    let opts = BuildOptions {
        opt_level: opt,
        out_dir: dir.clone(),
        linker,
        unit_path,
        kinds,
        ..BuildOptions::default()
    };
    let Some(program) = reported(build_tests(files, &opts)) else {
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(false);
    };
    report(program.all_diagnostics(), program.session.sources(), style_for(color))?;
    if program.executable.is_none() {
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(false);
    }
    let result = run_tests(&program);
    let _ = std::fs::remove_dir_all(&dir);
    let Some(results) = reported(result) else {
        return Ok(false);
    };

    // each test's result, then the tally
    let mut out = std::io::stdout().lock();
    for r in &results {
        if r.passed {
            writeln!(out, "test {} ... ok", r.name)?;
        } else {
            let how = r.status.map_or_else(|| "was ended".to_owned(), |s| format!("exited with {s}"));
            writeln!(out, "test {} ... FAILED ({how})", r.name)?;
            for line in r.output.lines() {
                writeln!(out, "    {line}")?;
            }
        }
    }
    let failed = results.iter().filter(|r| !r.passed).count();
    writeln!(
        out,
        "{} test{}: {} passed, {failed} failed",
        results.len(),
        if results.len() == 1 { "" } else { "s" },
        results.len() - failed
    )?;
    Ok(failed == 0)
}

/// A fresh temporary directory for `tqc {command}` to build in.
#[cfg(feature = "llvm")]
fn scratch(command: &str) -> std::io::Result<std::path::PathBuf> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("tqc-{command}-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// The value, or `None` once the error is printed.
fn reported<T>(result: Result<T, impl std::fmt::Display>) -> Option<T> {
    result.map_err(|e| eprintln!("tqc: {e}")).ok()
}

/// Prints every diagnostic and the summary line, returning how many were
/// errors.
fn report<'d>(
    diagnostics: impl Iterator<Item = &'d topiq::diag::Diagnostic>,
    sources: &topiq::source::SourceMap,
    style: Style,
) -> std::io::Result<usize> {
    let diagnostics: Vec<_> = diagnostics.cloned().collect();
    let mut err = std::io::stderr().lock();
    render::write_all(&diagnostics, sources, style, &mut err)?;
    let errors = diagnostics.iter().filter(|d| d.is_error()).count();
    if let Some(summary) = render::summary(errors, diagnostics.len() - errors) {
        writeln!(err, "tqc: {summary}")?;
    }
    Ok(errors)
}

/// Chooses a rendering style, honouring an explicit request.
fn style_for(choice: Option<ColorChoice>) -> Style {
    match choice {
        Some(ColorChoice::Always) => Style::Color,
        Some(ColorChoice::Never) => Style::Plain,
        Some(ColorChoice::Auto) | None => Style::for_terminal(std::io::stderr().is_terminal()),
    }
}

/// `tqc doc`: parses the program `entry` begins, and every unit it imports,
/// then writes their documentation. Parsing is all it needs, so a unit with
/// a type error is still documented.
fn document(
    entry: &std::path::Path,
    unit_path: &[std::path::PathBuf],
    kinds: &KindArgs,
    options: &topiq::doc::DocOptions,
    color: Option<ColorChoice>,
) -> std::io::Result<bool> {
    let style = style_for(color);
    let load = Options {
        stage: Stage::Parse,
        style,
        ..Options::default()
    };
    let Some(loaded) = reported(topiq::driver::program::load_as(&[entry.to_owned()], unit_path, load, kinds)) else {
        return Ok(false);
    };
    if report(loaded.all_diagnostics(), loaded.session.sources(), style)? > 0 {
        return Ok(false);
    }
    let Some(done) = reported(topiq::doc::document(&loaded, options)) else {
        return Ok(false);
    };
    for w in &done.warnings {
        eprintln!("tqc: warning: {w}");
    }
    eprintln!(
        "tqc: documented {} unit{} in {}",
        done.units,
        if done.units == 1 { "" } else { "s" },
        done.index.display()
    );
    Ok(true)
}

/// Translates every file, printing diagnostics and optionally one phase's
/// output. Up to parsing, each file stands alone; from analysis on, the files
/// and every unit they import are translated together, as one program.
fn translate(
    files: &[String],
    unit_path: &[std::path::PathBuf],
    kinds: &KindArgs,
    stage: Stage,
    deny_warnings: bool,
    color: Option<ColorChoice>,
    emit: Option<cli::EmitStage>,
) -> std::io::Result<bool> {
    let style = style_for(color);
    let options = Options {
        stage,
        style,
        deny_warnings,
        ..Options::default()
    };
    // a dependency graph needs the imports found, which the program does
    if stage > Stage::Parse || matches!(emit, Some(cli::EmitStage::Depgraph)) {
        return translate_program(files, unit_path, kinds, options, emit);
    }

    // up to parsing, each file on its own
    let mut ok = true;
    let mut errors = 0usize;
    let mut warnings = 0usize;

    for path in files {
        // load the file, or report why it cannot be read
        let mut session = Session::new();
        let id = match session.load(path) {
            Ok(id) => id,
            Err(e) => {
                match topiq::source::splice::DecodeError::of(&e) {
                    Some(bad) => {
                        let d = bad.diagnostic(std::path::Path::new(path));
                        let mut err = std::io::stderr().lock();
                        render::write_all(std::slice::from_ref(&d), session.sources(), style, &mut err)?;
                        errors += 1;
                    }
                    None => eprintln!("tqc: cannot read {path}: {e}"),
                }
                ok = false;
                continue;
            }
        };

        // translate it and report what was found
        let stem = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let kind = KindChoice {
            forced: kinds.for_unit(stem),
            fallback: None,
        };
        let out = compile(&mut session, id, Options { kind, ..options });
        errors += out.error_count();
        warnings += out.warning_count();

        let mut err = std::io::stderr().lock();
        render::write_all(&out.diagnostics, session.sources(), style, &mut err)?;

        // print the phase asked for, if the file got that far
        if out.has_errors() {
            ok = false;
            continue;
        }
        if let Some(what) = emit {
            let mut stdout = std::io::stdout().lock();
            emit_stage(&out, &session, what, &mut stdout)?;
        }
    }

    if let Some(summary) = render::summary(errors, warnings) {
        eprintln!("tqc: {summary}");
    }
    Ok(ok)
}

/// Translates the files as one program with every unit they import, printing
/// every unit's diagnostics and, for the files given, one phase's output.
fn translate_program(
    files: &[String],
    unit_path: &[std::path::PathBuf],
    kinds: &KindArgs,
    options: Options,
    emit: Option<cli::EmitStage>,
) -> std::io::Result<bool> {
    let inputs: Vec<std::path::PathBuf> = files.iter().map(std::path::PathBuf::from).collect();
    let Some(loaded) = reported(topiq::driver::program::load_as(&inputs, unit_path, options, kinds)) else {
        return Ok(false);
    };
    if report(loaded.all_diagnostics(), loaded.session.sources(), options.style)? > 0 {
        return Ok(false);
    }
    // one graph for the whole program, below each file given
    if let Some(cli::EmitStage::Depgraph) = emit {
        let given: Vec<usize> = (0..loaded.members.len()).filter(|&i| loaded.members[i].given).collect();
        let drawing = topiq::driver::program::graph(&loaded).render(&given);
        std::io::stdout().lock().write_all(drawing.as_bytes())?;
        return Ok(true);
    }
    if let Some(what) = emit {
        let mut stdout = std::io::stdout().lock();
        for m in loaded.members.iter().filter(|m| m.given) {
            if let Some(out) = m.outcome() {
                emit_stage(out, &loaded.session, what, &mut stdout)?;
            }
        }
    }
    Ok(true)
}

/// Prints what one phase produced.
fn emit_stage(
    out: &topiq::driver::Outcome,
    session: &Session,
    what: cli::EmitStage,
    w: &mut impl Write,
) -> std::io::Result<()> {
    let interner = session.interner();
    if let cli::EmitStage::Qasm = what {
        for l in &out.circuits {
            if l.entry {
                writeln!(w, "// [entry]")?;
            }
            write!(w, "{}", topiq::circuit::qasm::write(&l.circuit))?;
            writeln!(w)?;
        }
        return Ok(());
    }
    match Stage::from(what) {
        Stage::Tokens => {
            for (tok, span) in &out.tokens {
                writeln!(
                    w,
                    "{:<24} {}",
                    render::position(*span, session.sources()),
                    spell(*tok, interner)
                )?;
            }
        }
        Stage::Preprocess => {
            if let Some(pp) = &out.preprocessed {
                if let Some(u) = pp.unit {
                    writeln!(w, "// #unit {}", u.kind.name())?;
                }
                writeln!(w, "// conductor {}", pp.conductor)?;
                writeln!(w, "// fragment {:?}, qalloc {:?}", pp.fragment, pp.qalloc)?;
                let line = pp
                    .tokens
                    .iter()
                    .map(|(t, _)| spell(*t, interner))
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(w, "{line}")?;
            }
        }
        Stage::Mark => {
            if let Some(m) = &out.marked {
                writeln!(w, "// {} annotation group(s)", m.annotation_count())?;
                for (tok, span) in &m.tokens {
                    let marker = if tok.is(topiq::lex::Punct::LBracketAnnot) {
                        "annotation "
                    } else {
                        ""
                    };
                    writeln!(
                        w,
                        "{:<24} {marker}{}",
                        render::position(*span, session.sources()),
                        spell(*tok, interner)
                    )?;
                }
            }
        }
        Stage::Parse => {
            if let Some(u) = &out.unit {
                write!(w, "{}", topiq::ast::print::unit(u, interner))?;
            }
        }
        Stage::Check => {
            if let Some(t) = out.tir() {
                write!(w, "{}", topiq::tir::print::unit(t, interner))?;
            }
        }
        Stage::Metadata => {
            if let Some(m) = &out.metadata {
                write!(w, "{}", topiq::meta::encode::encode(m, interner))?;
            }
        }
        Stage::Llvm => {
            if let Some(ir) = &out.llvm_ir {
                write!(w, "{ir}")?;
            }
        }
        Stage::Build => {}
    }
    Ok(())
}

/// Prints the closed table of diagnostic identifiers.
fn list_diagnostics(grep: Option<&str>) {
    let needle = grep.map(str::to_lowercase);
    for &code in Code::ALL {
        let text = code.message();
        if let Some(n) = &needle
            && !code.id().to_lowercase().contains(n) && !text.to_lowercase().contains(n) {
                continue;
            }
        let phase = match code.phase() {
            Phase::Translation => "translation",
            Phase::Link => "link      ",
            Phase::Runtime => "runtime   ",
        };
        let mark = if code.is_language_defined() { " " } else { "*" };
        println!("{}{mark} {phase}  {text}", code.id());
    }
    if grep.is_none() {
        println!();
        println!("* marks an identifier of this implementation's own; the rest are defined by the language.");
    }
}
