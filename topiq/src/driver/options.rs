//! What to do with a unit, and how far to take it.

use crate::diag::render::Style;

/// How far the pipeline should run.
///
/// Each stage corresponds to a translation phase, so a build that stops early
/// can be described by the phase it reached.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub enum Stage {
    /// Stop after phase 3, tokenisation.
    Tokens,
    /// Stop after phase 4, preprocessing.
    Preprocess,
    /// Stop after the bracket-marking pass, which sits between phases 4 and
    /// 5 and decides which `[` opens an annotation.
    Mark,
    /// Stop after phase 5, parsing.
    #[default]
    Parse,
    /// Stop after phases 6 to 8: name resolution, constant evaluation and
    /// type checking and, for a quantum unit, phase 9 (which makes its
    /// circuits and judges them).
    Check,
    /// Check, then describe what the unit offers other units: the metadata
    /// its object will carry.
    Metadata,
    /// Generate LLVM IR, stopping before machine code.
    Llvm,
    /// Run every phase, producing an object file.
    Build,
}

impl Stage {
    /// The translation phase this stage completes.
    pub fn phase(self) -> u8 {
        match self {
            Stage::Tokens => 3,
            Stage::Preprocess => 4,
            // the marking pass runs between phases 4 and 5 and has no phase
            // number of its own, so it reports the phase it precedes
            Stage::Mark | Stage::Parse => 5,
            Stage::Check | Stage::Metadata => 8,
            Stage::Llvm | Stage::Build => 10,
        }
    }

    /// The name used on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Tokens => "tokens",
            Stage::Preprocess => "pp",
            Stage::Mark => "mark",
            Stage::Parse => "ast",
            Stage::Check => "check",
            Stage::Metadata => "metadata",
            Stage::Llvm => "llvm",
            Stage::Build => "build",
        }
    }

    /// Every stage, in pipeline order.
    pub const ALL: &'static [Stage] = &[
        Stage::Tokens,
        Stage::Preprocess,
        Stage::Mark,
        Stage::Parse,
        Stage::Check,
        Stage::Metadata,
        Stage::Llvm,
        Stage::Build,
    ];

    /// Looks a stage up by its command-line name.
    pub fn from_name(name: &str) -> Option<Stage> {
        Stage::ALL.iter().copied().find(|s| s.name() == name)
    }

    /// Whether this compiler, as built, can reach the stage.
    ///
    /// Code generation needs LLVM, which the `llvm` feature supplies. Without
    /// it everything up to type checking still works.
    pub fn is_implemented(self) -> bool {
        self <= Stage::Metadata || cfg!(feature = "llvm")
    }
}

/// How hard code generation works on the result.
///
/// Optimisation never changes what a program does, including when it aborts:
/// the optimiser may drop a check only where it can prove the check never
/// fires, and a check that can fire always survives.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OptLevel {
    /// No optimisation: the machine code follows the source closely.
    #[default]
    O0,
    /// Light optimisation that keeps compilation quick.
    O1,
    /// The standard optimisation pipeline.
    O2,
    /// The standard pipeline with more inlining and more aggressive loop and
    /// vector transformations.
    O3,
    /// The standard pipeline without the transformations that mostly grow
    /// the code.
    Os,
    /// Size above speed: as `Os`, and smaller still where speed suffers.
    Oz,
}

impl OptLevel {
    /// Every level, for what has to hold at all of them.
    pub const ALL: [OptLevel; 6] = [
        OptLevel::O0,
        OptLevel::O1,
        OptLevel::O2,
        OptLevel::O3,
        OptLevel::Os,
        OptLevel::Oz,
    ];
}

/// How a translation should be run.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// How far to go.
    pub stage: Stage,
    /// How to render diagnostics.
    pub style: Style,
    /// Whether warnings should be treated as errors.
    ///
    /// Off by default: a program may carry unknown annotations and pragmas on
    /// purpose, for other tools, and failing a build over them would make
    /// those features unusable.
    pub deny_warnings: bool,
    /// How hard code generation optimises.
    pub opt_level: OptLevel,
    /// Whether a valid `main` gets the entry point that starts the program
    /// with it. Not in a program built to run its `[test]` functions, nor in
    /// a module, which a program loads and never starts.
    pub entry: bool,
    /// Whether the program is built for a target that runs static circuits
    /// only (a circuit archive) rather than for its own processor: a
    /// target without the capabilities `@target_has` asks about.
    pub static_circuits: bool,
    /// What chooses the kind of a `#unit any` unit being translated: `--kind`
    /// for one given on the command line, or its importer for one found
    /// through an import.
    pub kind: crate::pp::KindChoice,
}

/// The kinds `--kind` chooses for the `#unit any` units given on the command
/// line: one for every such unit, and one for each unit named.
#[derive(Clone, Debug, Default)]
pub struct KindArgs {
    /// `--kind classical` or `--kind quantum`.
    pub all: Option<crate::pp::UnitKind>,
    /// `--kind NAME=classical`, and so on, by the unit's file name.
    pub named: Vec<(String, crate::pp::UnitKind)>,
}

impl KindArgs {
    /// The kind chosen for the unit `stem`: its own, or the one for all.
    pub fn for_unit(&self, stem: &str) -> Option<crate::pp::UnitKind> {
        self.named.iter().find(|(n, _)| n == stem).map(|&(_, k)| k).or(self.all)
    }

    /// Whether the unit `stem` is named on its own.
    pub fn names(&self, stem: &str) -> bool {
        self.named.iter().any(|(n, _)| n == stem)
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            stage: Stage::default(),
            style: Style::Plain,
            deny_warnings: false,
            opt_level: OptLevel::default(),
            entry: true,
            static_circuits: false,
            kind: crate::pp::KindChoice::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_run_in_pipeline_order() {
        assert!(Stage::Tokens < Stage::Preprocess);
        assert!(Stage::Preprocess < Stage::Mark);
        assert!(Stage::Mark < Stage::Parse);
        assert!(Stage::Parse < Stage::Check);
        assert!(Stage::Check < Stage::Metadata);
        assert!(Stage::Metadata < Stage::Llvm);
        assert!(Stage::Llvm < Stage::Build);
    }

    #[test]
    fn stage_names_round_trip_and_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &s in Stage::ALL {
            assert!(seen.insert(s.name()), "duplicate stage name {}", s.name());
            assert_eq!(Stage::from_name(s.name()), Some(s));
        }
        assert_eq!(Stage::from_name("nonesuch"), None);
    }

    #[test]
    fn stages_report_the_phase_they_complete() {
        assert_eq!(Stage::Tokens.phase(), 3);
        assert_eq!(Stage::Preprocess.phase(), 4);
        assert_eq!(Stage::Parse.phase(), 5);
        assert_eq!(Stage::Check.phase(), 8);
        assert_eq!(Stage::Llvm.phase(), 10);
        assert_eq!(Stage::Build.phase(), 10);
    }

    #[test]
    fn analysis_is_always_reachable_and_code_generation_needs_llvm() {
        for s in [
            Stage::Tokens,
            Stage::Preprocess,
            Stage::Mark,
            Stage::Parse,
            Stage::Check,
            Stage::Metadata,
        ] {
            assert!(s.is_implemented(), "{} should be reachable", s.name());
        }
        for s in [Stage::Llvm, Stage::Build] {
            assert_eq!(s.is_implemented(), cfg!(feature = "llvm"), "{}", s.name());
        }
    }

    #[test]
    fn the_default_is_to_parse_without_denying_warnings() {
        // a construct meant for another tool must not fail a build unless the
        // user asks
        let o = Options::default();
        assert_eq!(o.stage, Stage::Parse);
        assert!(!o.deny_warnings);
        assert_eq!(o.opt_level, OptLevel::O0);
    }
}
