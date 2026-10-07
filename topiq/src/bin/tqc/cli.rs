//! Command-line surface for `tqc`.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use topiq::driver::{OptLevel, Stage};

/// The Topiq compiler.
#[derive(Parser, Debug)]
#[command(
    name = "tqc",
    version,
    about = "The Topiq compiler",
    long_about = "The compiler for Topiq, a hybrid classical/quantum systems \
                  language.\n\n\
                  This build parses the whole language, and compiles the whole \
                  classical language, with the classical library (core, str, dyn, \
                  tcon, la, module, fs and io), to Windows executables, modules and test \
                  runners. A quantum unit is analysed in full and lowered to \
                  circuits, each operator judged exactly, and written as a .tqu; \
                  a program embeds the circuits of the quantum units it uses and \
                  runs them with the qpu library on the sim library's exact \
                  simulator or on a device the program registers, and \
                  --archive writes them alone as a .tqar. What is not yet \
                  translated is reported as TQ003 rather than claiming a success \
                  the build cannot deliver.\n\n\
                  An imported unit is found by its name, beside the files given or \
                  in a directory passed with --unit-path: NAME.tq is compiled from \
                  source, or else NAME.tcu or NAME.tqu, compiled earlier, is used \
                  as it is. A `#unit any` unit takes the kind its importer asks \
                  for, or the one --kind gives a unit named on the command line.\n\n\
                  `tqc doc` writes a program's documentation as a site of HTML \
                  pages, from the `///` and `//!` comments of every unit it uses."
)]
pub struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// The subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Translate a unit through type checking and report diagnostics,
    /// producing no output.
    Check {
        /// The `.tq` files to translate. The units they import are checked
        /// too.
        #[arg(required = true)]
        files: Vec<String>,
        /// A directory to look for imported units in, after those of the
        /// files given. May be repeated.
        #[arg(short = 'L', long = "unit-path", value_name = "DIR")]
        unit_path: Vec<PathBuf>,
        /// The kind of a `#unit any` unit given: `classical` or `quantum` for
        /// every such unit, or `NAME=classical` or `NAME=quantum` for the unit
        /// NAME. May be repeated.
        #[arg(long = "kind", value_name = "KIND|NAME=KIND")]
        kind: Vec<String>,
        /// Treat warnings as errors.
        #[arg(long)]
        deny_warnings: bool,
        /// Force colour on or off; the default follows the destination.
        #[arg(long, value_enum)]
        color: Option<ColorChoice>,
    },
    /// Translate a unit and print what one phase produced.
    Emit {
        /// The `.tq` files to translate.
        #[arg(required = true)]
        files: Vec<String>,
        /// Which phase's output to print.
        #[arg(long, value_enum, default_value_t = EmitStage::Ast)]
        stage: EmitStage,
        /// A directory to look for imported units in, after those of the
        /// files given. May be repeated.
        #[arg(short = 'L', long = "unit-path", value_name = "DIR")]
        unit_path: Vec<PathBuf>,
        /// The kind of a `#unit any` unit given: `classical` or `quantum` for
        /// every such unit, or `NAME=classical` or `NAME=quantum` for the unit
        /// NAME. May be repeated.
        #[arg(long = "kind", value_name = "KIND|NAME=KIND")]
        kind: Vec<String>,
        /// Treat warnings as errors.
        #[arg(long)]
        deny_warnings: bool,
        /// Force colour on or off; the default follows the destination.
        #[arg(long, value_enum)]
        color: Option<ColorChoice>,
    },
    /// Compile each unit to a `.tcu` object, or a quantum unit to a `.tqu`,
    /// and link them into an executable.
    Build {
        /// The units making up the program: `.tq` files to compile, and
        /// `.tcu` and `.tqu` files compiled earlier. The units they import are found
        /// and compiled too.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// A directory to look for imported units in, after those of the
        /// files given. May be repeated.
        #[arg(short = 'L', long = "unit-path", value_name = "DIR")]
        unit_path: Vec<PathBuf>,
        /// The kind of a `#unit any` unit given: `classical` or `quantum` for
        /// every such unit, or `NAME=classical` or `NAME=quantum` for the unit
        /// NAME. May be repeated.
        #[arg(long = "kind", value_name = "KIND|NAME=KIND")]
        kind: Vec<String>,
        /// The executable to write; by default `<first unit>.exe` beside the
        /// objects.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Where to write the objects; by default the current directory.
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Optimisation level. Checks that can fail survive every level.
        #[arg(short = 'O', long = "opt-level", value_enum, default_value_t = Opt::O0)]
        opt: Opt,
        /// Stop after writing the objects.
        #[arg(long)]
        no_link: bool,
        /// Build a module, which a program loads while it runs with
        /// `module::load`, rather than a program: a `.dll`, needing no `main`.
        #[arg(long, conflicts_with = "no_link")]
        module: bool,
        /// Write a circuit archive rather than a program: a `.tqar` of the
        /// circuits of the quantum units' `[entry]` operators, each in
        /// OpenQASM 3 with its judgement, needing no `main`.
        #[arg(long, conflicts_with_all = ["no_link", "module"])]
        archive: bool,
        /// A linker to use instead of the installed Microsoft one.
        #[arg(long)]
        linker: Option<PathBuf>,
        /// Force colour on or off; the default follows the destination.
        #[arg(long, value_enum)]
        color: Option<ColorChoice>,
    },
    /// Build a program in a temporary directory, run it, and exit with its
    /// status.
    Run {
        /// The `.tq` file defining `main`. The units it imports are found and
        /// compiled too.
        file: PathBuf,
        /// A directory to look for imported units in.
        /// May be repeated.
        #[arg(short = 'L', long = "unit-path", value_name = "DIR")]
        unit_path: Vec<PathBuf>,
        /// The kind of a `#unit any` unit given: `classical` or `quantum` for
        /// every such unit, or `NAME=classical` or `NAME=quantum` for the unit
        /// NAME. May be repeated.
        #[arg(long = "kind", value_name = "KIND|NAME=KIND")]
        kind: Vec<String>,
        /// Optimisation level.
        #[arg(short = 'O', long = "opt-level", value_enum, default_value_t = Opt::O0)]
        opt: Opt,
        /// A linker to use instead of the installed Microsoft one.
        #[arg(long)]
        linker: Option<PathBuf>,
        /// Force colour on or off; the default follows the destination.
        #[arg(long, value_enum)]
        color: Option<ColorChoice>,
    },
    /// Build a program's `[test]` functions in a temporary directory and run
    /// each in a process of its own: one that returns passes, one that
    /// aborts fails. Exits with 0 when every test passes.
    Test {
        /// The `.tq` files whose tests to run. The units they import are
        /// found and compiled too; tests are taken from the files compiled
        /// from source.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// A directory to look for imported units in, after those of the
        /// files given. May be repeated.
        #[arg(short = 'L', long = "unit-path", value_name = "DIR")]
        unit_path: Vec<PathBuf>,
        /// The kind of a `#unit any` unit given: `classical` or `quantum` for
        /// every such unit, or `NAME=classical` or `NAME=quantum` for the unit
        /// NAME. May be repeated.
        #[arg(long = "kind", value_name = "KIND|NAME=KIND")]
        kind: Vec<String>,
        /// Optimisation level.
        #[arg(short = 'O', long = "opt-level", value_enum, default_value_t = Opt::O0)]
        opt: Opt,
        /// A linker to use instead of the installed Microsoft one.
        #[arg(long)]
        linker: Option<PathBuf>,
        /// Force colour on or off; the default follows the destination.
        #[arg(long, value_enum)]
        color: Option<ColorChoice>,
    },
    /// Write HTML documentation of a program: every unit it is made of,
    /// the library's included, each declaration with its signature and its
    /// documentation, which `///` and `//!` comments give in Markdown.
    Doc {
        /// The `.tq` file the program begins with. The units it imports are
        /// found and documented too.
        entry: PathBuf,
        /// The directory to write the documentation into.
        #[arg(short, long, value_name = "DIR", default_value = "doc")]
        output: PathBuf,
        /// A directory to look for imported units in.
        /// May be repeated.
        #[arg(short = 'L', long = "unit-path", value_name = "DIR")]
        unit_path: Vec<PathBuf>,
        /// The kind of a `#unit any` entry: `classical` or `quantum`, or
        /// `NAME=classical` or `NAME=quantum`.
        #[arg(long = "kind", value_name = "KIND|NAME=KIND")]
        kind: Vec<String>,
        /// Document what units keep to themselves too: `static`
        /// declarations, and names beginning with `__`.
        #[arg(long)]
        document_private: bool,
        /// Force colour on or off; the default follows the destination.
        #[arg(long, value_enum)]
        color: Option<ColorChoice>,
    },
    /// Print the table of diagnostic identifiers.
    Diagnostics {
        /// Show only identifiers whose text contains this.
        #[arg(long)]
        grep: Option<String>,
    },
}

/// Which phase's output `emit` should print.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum EmitStage {
    /// Phase 3: the token stream.
    Tokens,
    /// Phase 4: the token stream after directives and macro expansion.
    Pp,
    /// The bracket-marking pass.
    Mark,
    /// Phase 5: the syntax tree.
    Ast,
    /// Phases 6 to 8: the typed unit, with every name resolved and every
    /// constant folded.
    Tir,
    /// What the unit offers other units: the metadata its object carries.
    Metadata,
    /// Phase 9: the circuits a quantum unit's operators become, as
    /// OpenQASM 3.
    Qasm,
    /// Phase 10: the LLVM IR generated for the unit.
    Llvm,
    /// The units the files depend on, directly or through others, drawn as
    /// a tree below each file, and the order they are analysed in. Units
    /// that depend on one another are marked, and analysed together.
    Depgraph,
}

impl From<EmitStage> for Stage {
    fn from(s: EmitStage) -> Stage {
        match s {
            EmitStage::Tokens => Stage::Tokens,
            EmitStage::Pp => Stage::Preprocess,
            EmitStage::Mark => Stage::Mark,
            // finding every unit a file depends on needs each one parsed
            EmitStage::Ast | EmitStage::Depgraph => Stage::Parse,
            EmitStage::Tir | EmitStage::Qasm => Stage::Check,
            EmitStage::Metadata => Stage::Metadata,
            EmitStage::Llvm => Stage::Llvm,
        }
    }
}

/// How hard to optimise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Opt {
    /// No optimisation.
    #[value(name = "0")]
    O0,
    /// Light optimisation that keeps compilation quick.
    #[value(name = "1")]
    O1,
    /// The standard optimisation pipeline.
    #[value(name = "2")]
    O2,
    /// The standard pipeline.
    #[value(name = "3")]
    O3,
    /// Optimise for size.
    #[value(name = "s")]
    Os,
    /// Optimise for size above speed.
    #[value(name = "z")]
    Oz,
}

impl From<Opt> for OptLevel {
    fn from(o: Opt) -> OptLevel {
        match o {
            Opt::O0 => OptLevel::O0,
            Opt::O1 => OptLevel::O1,
            Opt::O2 => OptLevel::O2,
            Opt::O3 => OptLevel::O3,
            Opt::Os => OptLevel::Os,
            Opt::Oz => OptLevel::Oz,
        }
    }
}

/// Whether to colour diagnostics.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ColorChoice {
    /// Always colour.
    Always,
    /// Never colour.
    Never,
    /// Colour when writing to a terminal.
    Auto,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_is_well_formed() {
        // clap's own consistency check: duplicate flags, bad defaults and
        // conflicting short options are all caught here
        Cli::command().debug_assert();
    }

    #[test]
    fn emit_stages_map_onto_pipeline_stages() {
        assert_eq!(Stage::from(EmitStage::Tokens), Stage::Tokens);
        assert_eq!(Stage::from(EmitStage::Pp), Stage::Preprocess);
        assert_eq!(Stage::from(EmitStage::Mark), Stage::Mark);
        assert_eq!(Stage::from(EmitStage::Ast), Stage::Parse);
        assert_eq!(Stage::from(EmitStage::Tir), Stage::Check);
        assert_eq!(Stage::from(EmitStage::Metadata), Stage::Metadata);
        assert_eq!(Stage::from(EmitStage::Llvm), Stage::Llvm);
        assert_eq!(Stage::from(EmitStage::Depgraph), Stage::Parse);
    }

    #[test]
    fn the_unit_path_may_be_given_more_than_once() {
        let cli = Cli::try_parse_from(["tqc", "build", "main.tq", "-L", "lib", "--unit-path", "vendor", "geo.tcu"]).unwrap();
        match cli.command {
            Command::Build { files, unit_path, .. } => {
                assert_eq!(files, [PathBuf::from("main.tq"), PathBuf::from("geo.tcu")]);
                assert_eq!(unit_path, [PathBuf::from("lib"), PathBuf::from("vendor")]);
            }
            other => panic!("expected build, got {other:?}"),
        }
    }

    #[test]
    fn build_takes_an_output_an_optimization_level_and_a_linker() {
        let cli = Cli::try_parse_from([
            "tqc", "build", "a.tq", "-o", "app.exe", "-O", "2", "--linker", "lld-link.exe",
        ])
        .unwrap();
        match cli.command {
            Command::Build {
                files,
                output,
                opt,
                linker,
                no_link,
                ..
            } => {
                assert_eq!(files, [PathBuf::from("a.tq")]);
                assert_eq!(output, Some(PathBuf::from("app.exe")));
                assert_eq!(OptLevel::from(opt), OptLevel::O2);
                assert_eq!(linker, Some(PathBuf::from("lld-link.exe")));
                assert!(!no_link);
            }
            other => panic!("expected build, got {other:?}"),
        }
    }

    #[test]
    fn run_takes_one_file_and_defaults_to_no_optimization() {
        let cli = Cli::try_parse_from(["tqc", "run", "fib.tq"]).unwrap();
        match cli.command {
            Command::Run { file, opt, .. } => {
                assert_eq!(file, PathBuf::from("fib.tq"));
                assert_eq!(opt, Opt::O0);
            }
            other => panic!("expected run, got {other:?}"),
        }
        assert!(Cli::try_parse_from(["tqc", "run", "-O", "4", "fib.tq"]).is_err());
    }

    #[test]
    fn every_optimization_level_has_a_flag() {
        let flags = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"];
        for (flag, want) in flags.into_iter().zip(OptLevel::ALL) {
            let cli = Cli::try_parse_from(["tqc", "run", flag, "fib.tq"]).unwrap();
            match cli.command {
                Command::Run { opt, .. } => assert_eq!(OptLevel::from(opt), want, "{flag}"),
                other => panic!("expected run, got {other:?}"),
            }
        }
    }

    #[test]
    fn check_parses_with_files() {
        let cli = Cli::try_parse_from(["tqc", "check", "a.tq", "b.tq"]).unwrap();
        match cli.command {
            Command::Check { files, .. } => assert_eq!(files.len(), 2),
            other => panic!("expected check, got {other:?}"),
        }
    }

    #[test]
    fn emit_defaults_to_the_tree() {
        let cli = Cli::try_parse_from(["tqc", "emit", "a.tq"]).unwrap();
        match cli.command {
            Command::Emit { stage, .. } => {
                assert_eq!(Stage::from(stage), Stage::Parse);
            }
            other => panic!("expected emit, got {other:?}"),
        }
    }

    #[test]
    fn emit_accepts_each_stage_by_name() {
        for name in ["tokens", "pp", "mark", "ast", "tir", "metadata", "qasm", "llvm", "depgraph"] {
            assert!(
                Cli::try_parse_from(["tqc", "emit", "--stage", name, "a.tq"]).is_ok(),
                "--stage {name} should be accepted"
            );
        }
        assert!(Cli::try_parse_from(["tqc", "emit", "--stage=depgraph", "a.tq"]).is_ok());
        assert!(Cli::try_parse_from(["tqc", "emit", "--stage", "asm", "a.tq"]).is_err());
    }

    #[test]
    fn a_command_needs_at_least_one_file() {
        assert!(Cli::try_parse_from(["tqc", "check"]).is_err());
        assert!(Cli::try_parse_from(["tqc", "emit"]).is_err());
    }

    #[test]
    fn doc_takes_an_entry_and_an_output_directory() {
        let cli = Cli::try_parse_from(["tqc", "doc", "app.tq", "-o", "site", "--document-private"]).unwrap();
        match cli.command {
            Command::Doc {
                entry,
                output,
                document_private,
                ..
            } => {
                assert_eq!(entry, PathBuf::from("app.tq"));
                assert_eq!(output, PathBuf::from("site"));
                assert!(document_private);
            }
            other => panic!("expected doc, got {other:?}"),
        }
        match Cli::try_parse_from(["tqc", "doc", "app.tq"]).unwrap().command {
            Command::Doc { output, document_private, .. } => {
                assert_eq!(output, PathBuf::from("doc"), "the default directory");
                assert!(!document_private);
            }
            other => panic!("expected doc, got {other:?}"),
        }
        assert!(Cli::try_parse_from(["tqc", "doc"]).is_err(), "the entry is required");
    }

    #[test]
    fn diagnostics_takes_an_optional_filter() {
        let cli = Cli::try_parse_from(["tqc", "diagnostics", "--grep", "conductor"]).unwrap();
        match cli.command {
            Command::Diagnostics { grep } => assert_eq!(grep.as_deref(), Some("conductor")),
            other => panic!("expected diagnostics, got {other:?}"),
        }
        assert!(Cli::try_parse_from(["tqc", "diagnostics"]).is_ok());
    }
}
