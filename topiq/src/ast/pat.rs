//! Pattern syntax.
//!
//! Patterns appear in `match` arms, in `let` bindings and in `for` bindings.
//! A `match` on classical values is checked for exhaustiveness, and the
//! pattern of a `let` or `for` must match every value.
//!
//! # Patterns carry no generic arguments
//!
//! A pattern may name a path but never a path with generic arguments. That is
//! what makes `=>` safe to lex as a single token: a pattern can never end with
//! the `>` that closes a generic argument list, so a `=>` after one is never
//! two tokens that need pulling apart. See [`crate::lex::token`] for the `>`
//! rule this plays against.

use crate::intern::Symbol;
use crate::span::Spanned;

use super::expr::Expr;
use super::ty::Path;

/// One field of a structure pattern.
#[derive(Clone, PartialEq, Debug)]
pub struct FieldPat {
    /// The field name.
    pub name: Spanned<Symbol>,
    /// The sub-pattern.
    pub pattern: Option<Spanned<Pattern>>,
}

/// A pattern.
#[derive(Clone, PartialEq, Debug)]
pub enum Pattern {
    /// `_`: matches anything and binds nothing.
    Wildcard,
    /// A literal (e.g. `0` or `'a'` or `true`).
    ///
    /// Held as an [`Expr`] because the grammar reuses the constant forms; only
    /// the literal forms are admissible, which phase 8 checks.
    Literal(Box<Spanned<Expr>>),
    /// A binding.
    Binding(Spanned<Symbol>),
    /// A path naming a unit variant or a constant (e.g. `Guess::Balanced`).
    Path(Path),
    /// `Path(p1, p2, …)`: a tuple-variant pattern.
    TupleStruct {
        /// The variant's name.
        path: Path,
        /// The sub-patterns.
        elements: Vec<Spanned<Pattern>>,
    },
    /// `Path { f: p, … }`: a structure-variant pattern.
    Struct {
        /// The variant's name.
        path: Path,
        /// The field patterns.
        fields: Vec<FieldPat>,
    },
    /// `(p1, p2, …)`: a tuple pattern.
    Tuple(Vec<Spanned<Pattern>>),
}

impl Pattern {
    /// Whether this pattern matches every value, so that an arm using it makes
    /// the ones after it unreachable.
    pub fn is_irrefutable(&self) -> bool {
        match self {
            Pattern::Wildcard | Pattern::Binding(_) => true,
            Pattern::Tuple(ps) => ps.iter().all(|p| p.node.is_irrefutable()),
            _ => false,
        }
    }

    /// Every name this pattern binds, in the order written.
    pub fn bindings(&self) -> Vec<Spanned<Symbol>> {
        let mut out = Vec::new();
        self.collect_bindings(&mut out);
        out
    }

    fn collect_bindings(&self, out: &mut Vec<Spanned<Symbol>>) {
        match self {
            Pattern::Binding(s) => out.push(*s),
            Pattern::Tuple(ps) | Pattern::TupleStruct { elements: ps, .. } => {
                for p in ps {
                    p.node.collect_bindings(out);
                }
            }
            Pattern::Struct { fields, .. } => {
                for f in fields {
                    match &f.pattern {
                        Some(p) => p.node.collect_bindings(out),
                        // the shorthand `Point { x, y }` binds the field name
                        None => out.push(f.name),
                    }
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) | Pattern::Path(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::{SourceId, Span};

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    #[test]
    fn wildcards_and_bindings_are_irrefutable() {
        let mut i = Interner::new();
        assert!(Pattern::Wildcard.is_irrefutable());
        assert!(Pattern::Binding(sp(i.intern("x"))).is_irrefutable());
        assert!(!Pattern::Literal(Box::new(sp(Expr::Bool(true)))).is_irrefutable());
        assert!(!Pattern::Path(Path::single(sp(i.intern("None")))).is_irrefutable());
    }

    #[test]
    fn a_tuple_is_irrefutable_only_if_every_element_is() {
        let mut i = Interner::new();
        let x = Pattern::Binding(sp(i.intern("x")));
        assert!(Pattern::Tuple(vec![sp(x.clone()), sp(Pattern::Wildcard)]).is_irrefutable());
        assert!(
            !Pattern::Tuple(vec![sp(x), sp(Pattern::Literal(Box::new(sp(Expr::Bool(true)))))])
                .is_irrefutable()
        );
    }

    #[test]
    fn bindings_are_collected_in_source_order() {
        let mut i = Interner::new();
        let (a, b) = (i.intern("a"), i.intern("b"));
        let p = Pattern::Tuple(vec![sp(Pattern::Binding(sp(a))), sp(Pattern::Binding(sp(b)))]);
        let names: Vec<Symbol> = p.bindings().iter().map(|s| s.node).collect();
        assert_eq!(names, vec![a, b]);
    }

    #[test]
    fn a_tuple_variant_pattern_binds_its_elements() {
        // the `Some(c)` of `match s as *Circle { Some(c) => Some(c.r), ... }`
        let mut i = Interner::new();
        let c = i.intern("c");
        let p = Pattern::TupleStruct {
            path: Path::single(sp(i.intern("Some"))),
            elements: vec![sp(Pattern::Binding(sp(c)))],
        };
        assert_eq!(p.bindings().len(), 1);
        assert_eq!(p.bindings()[0].node, c);
        assert!(!p.is_irrefutable());
    }

    #[test]
    fn the_field_shorthand_binds_the_field_name() {
        let mut i = Interner::new();
        let x = i.intern("x");
        let p = Pattern::Struct {
            path: Path::single(sp(i.intern("Point"))),
            fields: vec![FieldPat {
                name: sp(x),
                pattern: None,
            }],
        };
        assert_eq!(p.bindings()[0].node, x);
    }

    #[test]
    fn a_field_with_a_sub_pattern_binds_the_sub_patterns_names() {
        let mut i = Interner::new();
        let inner = i.intern("inner");
        let p = Pattern::Struct {
            path: Path::single(sp(i.intern("Point"))),
            fields: vec![FieldPat {
                name: sp(i.intern("x")),
                pattern: Some(sp(Pattern::Binding(sp(inner)))),
            }],
        };
        let names: Vec<Symbol> = p.bindings().iter().map(|s| s.node).collect();
        assert_eq!(names, vec![inner], "the field name is not itself bound");
    }

    #[test]
    fn a_path_pattern_binds_nothing() {
        // `Guess::Balanced` names a variant; it does not introduce a name
        let mut i = Interner::new();
        let p = Pattern::Path(Path {
            segments: vec![sp(i.intern("Guess")), sp(i.intern("Balanced"))],
        });
        assert!(p.bindings().is_empty());
    }
}
