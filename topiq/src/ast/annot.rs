//! Annotations: the bracketed declarations that qualify an item.
//!
//! Where other languages introduce attributes with a sigil (e.g. `#[...]`, `@`,
//! `[[...]]`) Topiq uses a bare `[`, and decides from position whether a
//! given one opens an annotation or an array. That decision is made before
//! parsing, by [`crate::mark`].
//!
//! # Arguments
//!
//! An annotation's arguments follow a colon rather than sitting inside
//! parentheses. The reason is legibility when an argument is itself
//! parenthesised:
//!
//! ```text
//! [gauge: fid((|00> + |11>) * isq2)]      // reads
//! [gauge(fid((|00> + |11>) * isq2))]      // does not
//! ```
//!
//! Several annotations stack inside one bracket for the same reason: it keeps
//! a heavily annotated operator to two lines instead of six.
//!
//! # Unknown names are a warning
//!
//! An annotation this compiler does not recognise is a warning (`EA01`), not an
//! error, and is carried through untouched. Annotations are how a program
//! speaks to its toolchain, and a program may carry one meant for a tool that
//! is not this compiler. [`StdAnnotation`] is the closed set of
//! names the language itself defines.

use crate::intern::Symbol;
use crate::quon::ast::QForm;
use crate::span::Spanned;

use super::expr::Expr;
use super::ty::Type;

/// An argument to an annotation.
#[derive(Clone, PartialEq, Debug)]
pub enum AnnArg {
    /// A constant expression (e.g. `[align: 16]`).
    Expr(Box<Spanned<Expr>>),
    /// A type name, as in `[expect: kernel.outcomes = Guess]`.
    Type(Box<Spanned<Type>>),
    /// A QUON state, cover or gauge (e.g. `[cover: fin{ |0>, |1> }]`).
    Quon(Box<QForm>),
    /// `name = value`, as in `[expect: contract = flat]`.
    Named {
        /// The argument's name.
        name: Spanned<Symbol>,
        /// Its value.
        value: Box<Spanned<AnnArg>>,
    },
}

/// One annotation.
#[derive(Clone, PartialEq, Debug)]
pub struct Annotation {
    /// The name. Annotation names live in a name space of their own, so
    /// `[align: 16]` does not collide with a function called `align`, and
    /// `cover` here is an annotation name rather than the keyword.
    pub name: Spanned<Symbol>,
    /// The arguments.
    pub args: Vec<Spanned<AnnArg>>,
}

/// A bracketed group of annotations.
///
/// Groups may be stacked, and several annotations may share one group,
/// separated by `;`. Order carries no meaning unless a particular annotation
/// says otherwise.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct AnnotationGroup {
    /// The annotations in this group.
    pub annotations: Vec<Spanned<Annotation>>,
}

impl AnnotationGroup {
    /// Whether the group contains an annotation with this name.
    pub fn has(&self, name: Symbol) -> bool {
        self.annotations.iter().any(|a| a.node.name.node == name)
    }
}

/// What kind of item an annotation may attach to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    /// An operator: a function in a quantum unit whose signature involves
    /// qubit-bearing types.
    Operator,
    /// A function or an object.
    FunctionOrObject,
    /// A function.
    Function,
    /// A structure or an enumeration.
    StructOrEnum,
    /// A structure.
    Struct,
    /// Any type.
    Type,
    /// A quantum enumeration.
    QuantumEnum,
    /// Any item.
    Item,
    /// An operator or a parameter.
    OperatorOrParam,
    /// An operator or a locale member.
    OperatorOrLocaleMember,
    /// An apply site.
    ApplySite,
}

/// The annotations the language itself defines.
///
/// Closed in the sense that these are the names Topiq gives meaning to. A
/// program may still write others: those warn under `EA01` and are carried
/// through untouched, for whatever tool they were meant for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(missing_docs)]
pub enum StdAnnotation {
    /// `[entry]`: marks an operator as a circuit entry point, so a circuit is
    /// emitted for it.
    Entry,
    /// `[export: "sym"]`: the external symbol name.
    Export,
    /// `[inline]`, `[inline: never]`: an inlining hint.
    Inline,
    /// `[packed]`: removes padding between fields.
    Packed,
    /// `[align: n]`: raises the alignment.
    Align,
    /// `[repr: uN]`: overrides an enumeration's discriminant type.
    Repr,
    /// `[open]`: lets other units add methods to this type.
    Open,
    /// `[typeinfo]`: forces a run-time type description to be emitted.
    Typeinfo,
    /// `[no_typeinfo]`: forbids table emission; `@typeinfo` on such a type is
    /// `EM03`.
    NoTypeinfo,
    /// `[iso: X, Z]`: declares a quantum enumeration an iso-sum, naming the
    /// operators that move between its variants.
    Iso,
    /// `[test]`: tooling.
    Test,
    /// `[deprecated: "msg"]`: tooling.
    Deprecated,
    /// `[tcon: open]`, `[tcon: opaque]`: how this type behaves as a data
    /// schema.
    Tcon,
    /// `[cover: c]`: the set of states a judgement about this operator is
    /// made over.
    Cover,
    /// `[gauge: g]`: the phase convention the judgement is stated against.
    Gauge,
    /// `[expect: …]`: a claim about this operator, checked against the
    /// judgement the compiler derives for it.
    Expect,
    /// `[frame: grouping]`: declares which qubits are considered together
    /// when framing.
    Frame,
    /// `[glue: piecewise]`: the judgement is assembled from pieces checked
    /// separately on each patch of the cover.
    Glue,
    /// `[adjoint: f]`: declares `f` as the adjoint.
    Adjoint,
    /// `[trusted_unitary]`: suppresses the exact unitarity check. One of the
    /// few places a program takes responsibility for something the compiler
    /// would otherwise prove.
    TrustedUnitary,
    /// `[dynamic]`: the operator contains a `lift`, so its circuit is not
    /// fixed until run time.
    Dynamic,
}

impl StdAnnotation {
    /// Every annotation the language defines.
    pub const ALL: &'static [StdAnnotation] = &[
        StdAnnotation::Entry,
        StdAnnotation::Export,
        StdAnnotation::Inline,
        StdAnnotation::Packed,
        StdAnnotation::Align,
        StdAnnotation::Repr,
        StdAnnotation::Open,
        StdAnnotation::Typeinfo,
        StdAnnotation::NoTypeinfo,
        StdAnnotation::Iso,
        StdAnnotation::Test,
        StdAnnotation::Deprecated,
        StdAnnotation::Tcon,
        StdAnnotation::Cover,
        StdAnnotation::Gauge,
        StdAnnotation::Expect,
        StdAnnotation::Frame,
        StdAnnotation::Glue,
        StdAnnotation::Adjoint,
        StdAnnotation::TrustedUnitary,
        StdAnnotation::Dynamic,
    ];

    /// The name as written.
    pub fn name(self) -> &'static str {
        match self {
            StdAnnotation::Entry => "entry",
            StdAnnotation::Export => "export",
            StdAnnotation::Inline => "inline",
            StdAnnotation::Packed => "packed",
            StdAnnotation::Align => "align",
            StdAnnotation::Repr => "repr",
            StdAnnotation::Open => "open",
            StdAnnotation::Typeinfo => "typeinfo",
            StdAnnotation::NoTypeinfo => "no_typeinfo",
            StdAnnotation::Iso => "iso",
            StdAnnotation::Test => "test",
            StdAnnotation::Deprecated => "deprecated",
            StdAnnotation::Tcon => "tcon",
            StdAnnotation::Cover => "cover",
            StdAnnotation::Gauge => "gauge",
            StdAnnotation::Expect => "expect",
            StdAnnotation::Frame => "frame",
            StdAnnotation::Glue => "glue",
            StdAnnotation::Adjoint => "adjoint",
            StdAnnotation::TrustedUnitary => "trusted_unitary",
            StdAnnotation::Dynamic => "dynamic",
        }
    }

    /// What the annotation attaches to.
    pub fn target(self) -> Target {
        match self {
            StdAnnotation::Entry
            | StdAnnotation::Frame
            | StdAnnotation::Glue
            | StdAnnotation::Adjoint
            | StdAnnotation::Dynamic => Target::Operator,
            StdAnnotation::Export => Target::FunctionOrObject,
            StdAnnotation::Inline => Target::Function,
            StdAnnotation::Packed | StdAnnotation::Align | StdAnnotation::Repr => {
                Target::StructOrEnum
            }
            StdAnnotation::Open | StdAnnotation::Typeinfo | StdAnnotation::NoTypeinfo => {
                Target::Type
            }
            StdAnnotation::Iso => Target::QuantumEnum,
            StdAnnotation::Test | StdAnnotation::Deprecated => Target::Item,
            StdAnnotation::Tcon => Target::Struct,
            StdAnnotation::Cover | StdAnnotation::Gauge => Target::OperatorOrParam,
            StdAnnotation::Expect => Target::OperatorOrLocaleMember,
            StdAnnotation::TrustedUnitary => Target::ApplySite,
        }
    }

    /// Whether this annotation's argument is a quantum form rather than an
    /// expression or a type.
    ///
    /// This is the list that decides where a quantum context opens inside an
    /// annotation, and therefore where `|0>` reads as a ket rather than as a
    /// bitwise or followed by a comparison.
    pub fn takes_quon(self) -> bool {
        matches!(
            self,
            StdAnnotation::Cover | StdAnnotation::Gauge | StdAnnotation::Frame
        )
    }

    /// Whether this annotation speaks of operators, circuits or judgements,
    /// which only a quantum unit has.
    pub fn is_quantum(self) -> bool {
        matches!(
            self,
            StdAnnotation::Entry
                | StdAnnotation::Iso
                | StdAnnotation::Cover
                | StdAnnotation::Gauge
                | StdAnnotation::Expect
                | StdAnnotation::Frame
                | StdAnnotation::Glue
                | StdAnnotation::Adjoint
                | StdAnnotation::TrustedUnitary
                | StdAnnotation::Dynamic
        )
    }

    /// Looks an annotation up by name.
    pub fn from_name(name: &str) -> Option<StdAnnotation> {
        StdAnnotation::ALL.iter().copied().find(|a| a.name() == name)
    }
}

impl std::fmt::Display for StdAnnotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::{SourceId, Span};
    use std::collections::HashSet;

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    #[test]
    fn the_standard_annotations_are_named() {
        let names: Vec<&str> = StdAnnotation::ALL.iter().map(|a| a.name()).collect();
        assert_eq!(
            names,
            vec![
                "entry",
                "export",
                "inline",
                "packed",
                "align",
                "repr",
                "open",
                "typeinfo",
                "no_typeinfo",
                "iso",
                "test",
                "deprecated",
                "tcon",
                "cover",
                "gauge",
                "expect",
                "frame",
                "glue",
                "adjoint",
                "trusted_unitary",
                "dynamic",
            ]
        );
    }

    #[test]
    fn names_are_unique_and_round_trip() {
        let mut seen = HashSet::new();
        for &a in StdAnnotation::ALL {
            assert!(seen.insert(a.name()), "duplicate {a}");
            assert_eq!(StdAnnotation::from_name(a.name()), Some(a));
        }
    }

    #[test]
    fn an_unknown_name_is_not_in_the_standard_set() {
        // such a name warns under EA01 rather than failing, so that
        // annotation-driven tooling coexists with conforming compilation
        assert_eq!(StdAnnotation::from_name("derive"), None);
        assert_eq!(StdAnnotation::from_name("Entry"), None);
    }

    #[test]
    fn exactly_three_annotations_take_a_quon_argument() {
        // `cover`, `gauge` and `frame` are the three; this list decides where
        // a ket can be written
        let quon: Vec<&str> = StdAnnotation::ALL
            .iter()
            .filter(|a| a.takes_quon())
            .map(|a| a.name())
            .collect();
        assert_eq!(quon, vec!["cover", "gauge", "frame"]);
    }

    #[test]
    fn judgment_annotations_attach_to_their_targets() {
        assert_eq!(StdAnnotation::Entry.target(), Target::Operator);
        assert_eq!(StdAnnotation::Cover.target(), Target::OperatorOrParam);
        assert_eq!(
            StdAnnotation::Expect.target(),
            Target::OperatorOrLocaleMember
        );
        assert_eq!(StdAnnotation::Iso.target(), Target::QuantumEnum);
        assert_eq!(
            StdAnnotation::TrustedUnitary.target(),
            Target::ApplySite,
            "it suppresses a check at the site, not on a declaration"
        );
    }

    #[test]
    fn a_group_reports_the_annotations_it_holds() {
        let mut i = Interner::new();
        let entry = i.intern("entry");
        let g = AnnotationGroup {
            annotations: vec![sp(Annotation {
                name: sp(entry),
                args: Vec::new(),
            })],
        };
        assert!(g.has(entry));
        assert!(!g.has(i.intern("packed")));
    }

    #[test]
    fn a_group_may_hold_several_annotations() {
        // two annotations in one group: `[packed; align: 8]`
        let mut i = Interner::new();
        let (packed, align) = (i.intern("packed"), i.intern("align"));
        let g = AnnotationGroup {
            annotations: vec![
                sp(Annotation {
                    name: sp(packed),
                    args: Vec::new(),
                }),
                sp(Annotation {
                    name: sp(align),
                    args: vec![sp(AnnArg::Expr(Box::new(sp(Expr::Bool(true)))))],
                }),
            ],
        };
        assert!(g.has(packed) && g.has(align));
        assert_eq!(g.annotations.len(), 2);
    }

    #[test]
    fn a_named_argument_nests_a_value() {
        // a named argument: `[expect: contract = flat]`
        let mut i = Interner::new();
        let arg = AnnArg::Named {
            name: sp(i.intern("contract")),
            value: Box::new(sp(AnnArg::Expr(Box::new(sp(Expr::Bool(true)))))),
        };
        match arg {
            AnnArg::Named { name, .. } => assert_eq!(name.node, i.intern("contract")),
            _ => panic!("expected a named argument"),
        }
    }

    #[test]
    fn an_empty_group_is_the_default() {
        assert!(AnnotationGroup::default().annotations.is_empty());
    }
}
