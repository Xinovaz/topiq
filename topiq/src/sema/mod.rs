//! Translation phases 6 to 8: name resolution, constant evaluation and type
//! checking.
//!
//! Analysis turns a syntax tree into the [typed intermediate
//! representation](crate::tir): every name resolved, every expression typed,
//! every constant folded. It runs in these steps.
//!
//! 1. **Declare** ([`items`]). Unit scope is unordered, so every declaration's
//!    name is known before anything is looked at, and each `import` is
//!    resolved to the other unit's [`interface`].
//! 2. **Resolve and check** ([`items`], [`body`] and the modules it uses).
//!    Structures and enumerations ([`adt`]), signatures, object types,
//!    initialisers and function bodies are each resolved the first time
//!    something needs them. A body is resolved and typed in one walk, since
//!    block scope is ordered and a name's meaning depends on where it is met;
//!    unsuffixed integer literals are settled at its end ([`infer`]), and
//!    every `match` in it is checked for coverage ([`exhaust`]).
//!    In a quantum unit, each quantum condition used as a value is then
//!    written out as a new qubit holding it ([`qbits::materialize`]).
//! 3. **Moves** ([`moves`]). Every path through each function is followed, to
//!    check that nothing is used after being moved away or before being given
//!    a value, and that no qubit is dropped while it may still be live. In a
//!    quantum unit, [`kinds`] then checks where measured values go.
//! 4. **Copies** ([`copies`]). Each copy of a value whose type defines
//!    `$copy` is written out as a call of it.
//! 5. **Fold** ([`fold`]). Every constant is computed ([`consteval`]) and
//!    substituted, which is also when a constant that is not really constant,
//!    or that would abort, is found.
//!
//! Steps 3 to 5 only run once everything before them is clean. The result is
//! complete only if no step reported an error.
//!
//! Integer arithmetic has one definition, in [`arith`], shared by constant
//! evaluation and by the tests that hold the compiled program to the same
//! rules; layout has one, in [`crate::tir::layout`].
//!
//! # Where each part of the language is checked
//!
//! | module | what |
//! |---|---|
//! | [`items`], [`scope`], [`local_items`] | declarations and what names mean, at unit scope and in blocks |
//! | [`imports`], [`interface`] | imports, and what a unit offers its importers |
//! | [`adt`], [`types`] | structures, enumerations and written types |
//! | [`body`], [`stmt`], [`expr`], [`place`] | bodies, statements, expressions and places |
//! | [`call`], [`path`], [`method`], [`closure`] | calls, paths, methods, operator methods and closures |
//! | [`aggregate`], [`pattern`], [`exhaust`], [`refutable`] | building values and taking them apart |
//! | [`generic`], [`infer`] | generic items and their instances; literal types |
//! | [`iterate`], [`try_op`], [`growable`] | `for`, `?`, and growable arrays |
//! | [`core`], [`library_ops`] | what the compiler provides of the library |
//! | [`macros`], [`embed`], [`typeinfo`] | the builtin macros, `@embed`, and run-time type information |
//! | [`exact`] | `frac`, `cyclo` and `phase` during translation, and linear-algebra operators |
//! | [`attrs`] | `[export]`, `[inline]`, `[test]` and `[deprecated]` |
//! | [`startup`] | unit initialisers |
//! | [`arith`], [`consteval`], [`fold`] | arithmetic, constant evaluation and folding |
//! | [`moves`], [`copies`] | ownership, and `$copy` |
//! | [`quantum`], [`kinds`] | operations on quantum state, and generation-time against outcome values |
//! | [`quon`] | states written in QUON: `prep`, and `@embed` of a QUON document |
//! | [`geometry`], [`qmap`] | covers, gauges, base types, locales and chains; map locales and their two readings |
//! | [`claims`], [`judged`] | what annotations claim of operators, and the macros that ask of their judgements |
//! | [`qpu`] | running a circuit handle, and `circuit::controlled` |
//!
//! Both kinds of unit are analysed the same way. A quantum unit may also hold
//! qubits and act on them, and once analysed it records what a classical
//! unit importing it may use: its types that hold no qubits, its constants,
//! and its `[entry]` operators, which become circuit handles, run with the
//! `qpu` library or called as functions ([`qpu`]). Turning its
//! operators into circuits, and deriving their judgements, is translation
//! phase 9, in [`crate::lower`]; the claims read here are checked there. The
//! one quantum construct analysis does not translate, measuring a quantum
//! enumeration whole outside `match measure`, is reported as `TQ003` where
//! it is written.
//!
//! # Rules
//!
//! - **Unit scope is unordered; block scope is ordered.** A function may be
//!   called above its declaration, but a `let` is visible only after its own
//!   statement. Items a block declares are visible throughout the block.
//! - **`const T` is a type former, not a declaration specifier**, so it
//!   composes with the other formers and `const const T` is `const T`.

pub mod adt;
pub mod aggregate;
pub mod arith;
pub mod attrs;
pub mod body;
pub mod claims;
pub mod call;
pub mod closure;
pub mod consteval;
pub mod copies;
pub mod core;
pub mod embed;
pub mod exact;
pub mod exhaust;
pub mod expr;
pub mod fold;
pub mod generic;
pub mod geometry;
pub mod growable;
pub mod imports;
pub mod infer;
pub mod interface;
pub mod items;
pub mod iterate;
pub mod judged;
pub mod kinds;
pub mod library_ops;
pub mod local_items;
pub mod macros;
pub mod method;
pub mod moves;
pub mod path;
pub mod pattern;
pub mod place;
pub mod qbits;
pub mod qmap;
pub mod qpu;
pub mod quantum;
pub mod quon;
pub mod refutable;
pub mod report;
pub mod scope;
pub mod startup;
pub mod step;
pub mod stmt;
pub mod text;
pub mod try_op;
pub mod typeinfo;
pub mod types;

pub use interface::Interface;

use crate::ast;
use crate::diag::Diagnostic;
use crate::intern::{Interner, Symbol};
use crate::pp::UnitKind;
use crate::span::{SourceId, Span};
use crate::tir;

/// What analysis produced.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// The analysed unit. Present whenever declaration ran, so that it can be
    /// printed; only complete when [`Analysis::succeeded`] holds.
    pub tir: Option<tir::Unit>,
    /// Everything analysis reported.
    pub diagnostics: Vec<Diagnostic>,
}

impl Analysis {
    /// Whether analysis finished without an error, so that the unit is
    /// complete and every constant is folded.
    pub fn succeeded(&self) -> bool {
        self.tir.is_some() && !self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// What analysis needs to know about a unit beyond its syntax tree.
#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    /// The unit's name: its file stem.
    pub name: &'a str,
    /// The file it came from.
    pub source: SourceId,
    /// Where the `#unit` directive is, for diagnostics about the unit as a
    /// whole.
    pub directive: Span,
    /// The unit initialiser the directive names, if it names one, and where
    /// the directive is.
    pub initializer: Option<(Symbol, Span)>,
    /// The interfaces of the units this one may import.
    pub imports: &'a [Interface],
    /// The unit's preprocessor macros, which `@foreach_field` and its kin
    /// expand once per member.
    pub macros: Option<&'a crate::pp::MacroTable>,
    /// The documents `@embed` names.
    pub documents: &'a [Document<'a>],
    /// The unit's conductor.
    pub conductor: u32,
    /// Whether the program is built for a target running static circuits
    /// only, which `@target_has` answers for.
    pub static_circuits: bool,
}

/// A document `@embed` names: the path written, and the document's source
/// and text, or why it could not be read.
pub type Document<'a> = (String, Result<(SourceId, &'a crate::source::Spliced), String>);

/// Runs phases 6 to 8 over one unit.
pub fn analyze(unit: &ast::Unit, input: Input<'_>, interner: &Interner) -> Analysis {
    let quantum = match unit.kind {
        Some(UnitKind::Classical) => false,
        Some(UnitKind::Quantum) => true,
        // without a directive the unit was already rejected
        None => {
            return Analysis {
                tir: None,
                diagnostics: Vec::new(),
            };
        }
    };

    let mut cx = items::UnitCx::new(input.name, input.source, interner, input.imports);
    cx.macros = input.macros;
    cx.documents = input.documents;
    cx.conductor = input.conductor;
    cx.static_circuits = input.static_circuits;
    cx.quantum = quantum;
    cx.declare(unit, input.initializer.is_some());
    if let Some((name, span)) = input.initializer {
        cx.resolve_initializer(name, span);
    }
    cx.check_all();
    cx.check_geometry();
    cx.check_qmaps();
    cx.check_claims(unit);
    cx.check_asserts();
    cx.finish_tables();
    cx.check_new_bodies();
    cx.collect_drops();
    if quantum {
        cx.resolve_adjoints();
        cx.collect_quantum_interface(unit);
    }
    let items::UnitCx {
        unit: mut tir,
        mut diags,
        ..
    } = cx;

    let clean = |d: &[Diagnostic]| !d.iter().any(Diagnostic::is_error);
    if quantum && clean(&diags) {
        qbits::materialize(&mut tir);
    }
    if clean(&diags) {
        moves::check(&tir, interner, &mut diags);
        startup::check(&tir, interner, &mut diags);
        if quantum {
            kinds::check(&tir, &mut diags);
        }
    }
    if clean(&diags) {
        copies::expand(&mut tir);
    }
    if clean(&diags) {
        fold::fold(&mut tir, interner, &mut diags);
    }

    Analysis {
        tir: Some(tir),
        diagnostics: diags,
    }
}

/// Helpers shared by the analysis tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::diag::Code;

    /// Parses and analyses `src` as a classical unit named `test`.
    pub fn analyzed(src: &str) -> (tir::Unit, Vec<Diagnostic>) {
        analyzed_with(src, &[])
    }

    /// Parses and analyses `src` as a quantum unit named `test`.
    pub fn analyzed_quantum(src: &str) -> (tir::Unit, Vec<Diagnostic>) {
        let (u, d, _) = quantum_with_names(src);
        (u, d)
    }

    /// Parses and analyses `src` as a quantum unit named `test`, with the
    /// interner its names are in.
    pub fn quantum_with_names(src: &str) -> (tir::Unit, Vec<Diagnostic>, Interner) {
        let (mut u, i) = crate::parse::testing::unit(src);
        u.kind = Some(UnitKind::Quantum);
        let a = analyze(
            &u,
            Input {
                name: "test",
                source: SourceId(0),
                directive: Span::synthetic(),
                initializer: None,
                macros: None,
                documents: &[],
                conductor: 8,
                static_circuits: false,
                imports: &[],
            },
            &i,
        );
        (a.tir.expect("a quantum unit always collects"), a.diagnostics, i)
    }

    /// The identifiers of everything analysing `src` as a quantum unit
    /// reported, in order.
    pub fn quantum_codes(src: &str) -> Vec<Code> {
        analyzed_quantum(src).1.iter().map(|d| d.code).collect()
    }

    /// Parses and analyses `src`, which may import any of `imports`.
    pub fn analyzed_with(src: &str, imports: &[Interface]) -> (tir::Unit, Vec<Diagnostic>) {
        let (u, i) = crate::parse::testing::unit(src);
        let a = analyze(
            &u,
            Input {
                name: "test",
                source: SourceId(0),
                directive: Span::synthetic(),
                initializer: None,
                macros: None,
                documents: &[],
                conductor: 8,
                static_circuits: false,
                imports,
            },
            &i,
        );
        (a.tir.expect("a classical unit always collects"), a.diagnostics)
    }

    /// Parses and analyses `src` as the unit `name`, interning into
    /// `interner` so that it can import, or be imported by, other units
    /// analysed with the same one.
    pub fn analyzed_in(
        name: &str,
        src: &str,
        imports: &[Interface],
        interner: &mut Interner,
    ) -> (tir::Unit, Vec<Diagnostic>) {
        let u = crate::parse::testing::unit_in(src, interner);
        let a = analyze(
            &u,
            Input {
                name,
                source: SourceId(0),
                directive: Span::synthetic(),
                initializer: None,
                macros: None,
                documents: &[],
                conductor: 8,
                static_circuits: false,
                imports,
            },
            interner,
        );
        (a.tir.expect("a classical unit always collects"), a.diagnostics)
    }

    /// Parses and analyses `src` as the quantum unit `name`, interning into
    /// `interner`, as [`analyzed_in`] does.
    pub fn quantum_in(name: &str, src: &str, imports: &[Interface], interner: &mut Interner) -> (tir::Unit, Vec<Diagnostic>) {
        let mut u = crate::parse::testing::unit_in(src, interner);
        u.kind = Some(UnitKind::Quantum);
        let a = analyze(
            &u,
            Input {
                name,
                source: SourceId(0),
                directive: Span::synthetic(),
                initializer: None,
                macros: None,
                documents: &[],
                conductor: 8,
                static_circuits: false,
                imports,
            },
            interner,
        );
        (a.tir.expect("a quantum unit always collects"), a.diagnostics)
    }

    /// Analyses `src`, which is expected to be clean.
    pub fn check(src: &str) -> tir::Unit {
        let (u, d) = analyzed(src);
        let errors: Vec<String> = d
            .iter()
            .filter(|d| d.is_error())
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect();
        assert!(errors.is_empty(), "{src:?} should analyse cleanly, but:\n{}", errors.join("\n"));
        u
    }

    /// The identifiers of everything analysing `src` reported, in order.
    pub fn codes(src: &str) -> Vec<Code> {
        analyzed(src).1.iter().map(|d| d.code).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{analyzed, check, codes};
    use super::*;
    use crate::diag::Code;

    #[test]
    fn a_small_program_analyzes_to_a_complete_unit() {
        let u = check(
            "fn fib(n: i32) -> i32 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }\n\
             fn main() -> i32 { fib(10) }",
        );
        assert_eq!(u.fns.len(), 2);
        assert_eq!(u.main, Some(tir::FnId(1)));
    }

    #[test]
    fn folding_waits_until_everything_else_is_clean() {
        // with a type error present, the constant is not evaluated at all, so
        // its division by zero is not reported on top
        let (u, d) = analyzed("let X: const i32 = 1 / 0;\nfn f() -> bool { 1 }");
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Es06]);
        assert_eq!(u.globals[0].value, None);
    }

    #[test]
    fn a_quantum_unit_is_analyzed_in_full() {
        let (u, d) = super::testing::analyzed_quantum(
            "struct Reg { q: [qubit; 2] }\nstruct Note { n: u32 }\nlet N: const u32 = 2;\n\
             fn helper(q: *qubit) { }\n[entry]\nfn run(n: u32) -> bool { true }\n",
        );
        assert!(d.is_empty(), "{d:?}");
        assert!(u.quantum);
        assert_eq!(u.entries.len(), 1);
        assert_eq!(u.quantum_only.len(), 2, "the structure holding qubits, and the operator");
    }

    #[test]
    fn a_body_error_in_a_quantum_unit_is_reported() {
        let (_, d) = super::testing::analyzed_quantum("fn f() -> bool { 1 }");
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Es06]);
    }

    #[test]
    fn several_mistakes_are_all_reported() {
        let got = codes("fn f() -> i32 { let a: bool = 1; b }");
        assert_eq!(got, [Code::Es06, Code::Es04]);
    }
}
