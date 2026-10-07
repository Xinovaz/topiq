//! Type syntax: types, paths and generic arguments.
//!
//! # `const`
//!
//! `const` composes with the other type formers in the ordinary way, and
//! `const const T` is simply `const T`. That is why [`Type::Const`] wraps a
//! type rather than sitting as a flag on a declaration, and why there is no
//! `constexpr` specifier anywhere in this crate.
//!
//! Every former applies to what stands to its **right**, which is the rule that
//! makes the following three distinct and readable:
//!
//! | written | means |
//! |---|---|
//! | `*const T` | a reference to a constant object |
//! | `const *T` | a constant reference to a mutable object |
//! | `const *const T` | both |
//!
//! `constexpr T` is a `const T` whose value must be known during translation.
//! It applies to a whole binding, parameter or result, never behind a
//! reference.
//!
//! # `of` is ordered
//!
//! `A of B` and `B of A` contain exactly the same points, and they are
//! different types whenever their two gauges differ where they overlap. So `of`
//! is **not** commutative and does **not** reduce to an intersection type.
//!
//! [`Type::Restriction`] therefore keeps its two operands in separate fields
//! with separate meanings: the restriction contributes only its predicate, and
//! **the gauge comes from the container**.

use crate::intern::Symbol;
use crate::span::Spanned;

use super::expr::Expr;

/// A path: one or more `::`-separated identifiers.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Path {
    /// The segments.
    pub segments: Vec<Spanned<Symbol>>,
}

impl Path {
    /// A path of one segment.
    pub fn single(name: Spanned<Symbol>) -> Path {
        Path {
            segments: vec![name],
        }
    }

    /// The final segment.
    pub fn last(&self) -> Option<Symbol> {
        self.segments.last().map(|s| s.node)
    }

    /// Whether this path has exactly one segment.
    pub fn is_simple(&self) -> bool {
        self.segments.len() == 1
    }
}

/// A generic argument.
///
/// A compile-time parameter is just one of const type, so an argument is a
/// type or a constant expression, with no third kind.
#[derive(Clone, PartialEq, Debug)]
pub enum TArg {
    /// A type argument.
    Type(Spanned<Type>),
    /// A constant-expression argument (e.g. `[qubit; N]` or `phase::<8>`).
    Const(Spanned<Expr>),
}

/// A type.
#[derive(Clone, PartialEq, Debug)]
pub enum Type {
    /// `const T`: a value that is never assigned. The compiler folds a
    /// `const` binding into its uses whenever it can compute the value
    /// during translation.
    Const(Box<Spanned<Type>>),
    /// `constexpr T`: a `const` value the compiler must be able to compute
    /// during translation, and reports where it cannot.
    Constexpr(Box<Spanned<Type>>),
    /// `*T`: a reference, carried as two words: the address and the
    /// referent's run-time type. It may read and write its referent, unless
    /// that is `const`, as in `*const T`.
    Ref {
        /// The referent type.
        target: Box<Spanned<Type>>,
    },
    /// `[T]`, an owning growable array, or `[T; N]`, a fixed array.
    ///
    /// On `T = qubit` these are a dynamically allocated register and a
    /// fixed-width register respectively.
    Array {
        /// The element type.
        elem: Box<Spanned<Type>>,
        /// The length.
        len: Option<Box<Spanned<Expr>>>,
    },
    /// `(T1, T2, …)`: a tuple.
    Tuple(Vec<Spanned<Type>>),
    /// `fn(T…) -> U`: a function type.
    Fn {
        /// Argument types.
        params: Vec<Spanned<Type>>,
        /// The return type.
        ret: Option<Box<Spanned<Type>>>,
    },
    /// `closure<Sig>`: an owning environment plus a code pointer.
    Closure(Box<Spanned<Type>>),
    /// `circuit<Sig>`: an opaque handle to a circuit, carrying the
    /// judgements derived for it.
    Circuit(Box<Spanned<Type>>),
    /// `qmap<K, V>`: a table that can be looked up coherently.
    Qmap {
        /// The key register type.
        key: Box<Spanned<Type>>,
        /// The common entry type.
        value: Box<Spanned<Type>>,
    },
    /// `dyn`: a value that owns itself and carries its own type.
    Dyn,
    /// `@name(args)` written where a type is: a builtin macro yielding one,
    /// such as `@field_type(T, "x")`. Its arguments are types and strings.
    Macro {
        /// The macro's name.
        name: Spanned<Symbol>,
        /// Its arguments.
        args: Vec<Spanned<crate::ast::expr::MacroArg>>,
    },
    /// `void`, the return type of a function that returns nothing. It has no
    /// values at all, so it is not a unit type in disguise.
    Void,
    /// `A of B`: `B` restricted to those of its points that are also `A`.
    ///
    /// **Ordered and irreducible.** The cover is the intersection of the two
    /// covers, and the gauge is `container`'s; `restricted` contributes only
    /// its predicate.
    Restriction {
        /// `A`: the type whose predicate is taken, and whose gauge is
        /// discarded.
        restricted: Box<Spanned<Type>>,
        /// `B`: the container, whose gauge the restriction carries.
        container: Box<Spanned<Type>>,
    },
    /// `T?`: shorthand for `Opt<T>`.
    Optional(Box<Spanned<Type>>),
    /// A named type.
    Path {
        /// The name.
        path: Path,
        /// Generic arguments, empty when none were written.
        args: Vec<Spanned<TArg>>,
    },
}

impl Type {
    /// Whether this is `const T` or `constexpr T` at the outermost level.
    pub fn is_const(&self) -> bool {
        matches!(self, Type::Const(_) | Type::Constexpr(_))
    }

    /// Whether this is `constexpr T` at the outermost level, among any outer
    /// `const`s.
    pub fn is_constexpr(&self) -> bool {
        let mut t = self;
        loop {
            match t {
                Type::Constexpr(_) => return true,
                Type::Const(inner) => t = &inner.node,
                _ => return false,
            }
        }
    }

    /// Peels every outer `const` and `constexpr`, since `const const T` is
    /// `const T`.
    pub fn peel_const(&self) -> &Type {
        let mut t = self;
        while let Type::Const(inner) | Type::Constexpr(inner) = t {
            t = &inner.node;
        }
        t
    }

    /// Whether the type mentions `qubit` anywhere in its structure.
    ///
    /// The syntactic half of whether a type is quantum: a named type needs
    /// phase 6 to resolve, so it answers `false` here, and `@is_quantum`
    /// answers in full.
    pub fn mentions_qubit(&self, qubit: Symbol) -> bool {
        match self {
            Type::Const(t) | Type::Constexpr(t) | Type::Closure(t) | Type::Circuit(t) | Type::Optional(t) => {
                t.node.mentions_qubit(qubit)
            }
            Type::Ref { target, .. } => target.node.mentions_qubit(qubit),
            Type::Array { elem, .. } => elem.node.mentions_qubit(qubit),
            Type::Tuple(ts) => ts.iter().any(|t| t.node.mentions_qubit(qubit)),
            Type::Fn { params, ret } => {
                params.iter().any(|t| t.node.mentions_qubit(qubit))
                    || ret.as_ref().is_some_and(|t| t.node.mentions_qubit(qubit))
            }
            Type::Qmap { key, value } => {
                key.node.mentions_qubit(qubit) || value.node.mentions_qubit(qubit)
            }
            // what a macro yields is only known once it is expanded
            Type::Macro { .. } => false,
            Type::Restriction {
                restricted,
                container,
            } => restricted.node.mentions_qubit(qubit) || container.node.mentions_qubit(qubit),
            Type::Path { path, args } => {
                path.last() == Some(qubit)
                    || args.iter().any(|a| match &a.node {
                        TArg::Type(t) => t.node.mentions_qubit(qubit),
                        TArg::Const(_) => false,
                    })
            }
            Type::Dyn | Type::Void => false,
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

    fn named(sym: Symbol) -> Type {
        Type::Path {
            path: Path::single(sp(sym)),
            args: Vec::new(),
        }
    }

    #[test]
    fn const_is_a_type_former_that_composes() {
        let mut i = Interner::new();
        let u32_ = i.intern("u32");
        // `*const u32` is a reference to a constant object
        let ref_to_const = Type::Ref {
            target: Box::new(sp(Type::Const(Box::new(sp(named(u32_)))))),
        };
        // `const *u32` is a constant reference to a mutable object
        let const_ref = Type::Const(Box::new(sp(Type::Ref {
            target: Box::new(sp(named(u32_))),
        })));
        assert_ne!(
            ref_to_const, const_ref,
            "a type former applies to what stands to its right, so these two \
             are different types"
        );
        assert!(!ref_to_const.is_const());
        assert!(const_ref.is_const());
    }

    #[test]
    fn double_const_peels_to_one() {
        // `const const T` is the same type as `const T`
        let mut i = Interner::new();
        let t = named(i.intern("u32"));
        let once = Type::Const(Box::new(sp(t.clone())));
        let twice = Type::Const(Box::new(sp(once.clone())));
        assert_eq!(once.peel_const(), &t);
        assert_eq!(twice.peel_const(), &t);
        assert_eq!(once.peel_const(), twice.peel_const());
    }

    #[test]
    fn restriction_is_ordered_and_not_commutative() {
        // `of` is not commutative. a type representation that normalised the
        // operands, or reduced this to an intersection type, would silently
        // conflate two genuinely different types
        let mut i = Interner::new();
        let (a, b) = (named(i.intern("A")), named(i.intern("B")));
        let a_of_b = Type::Restriction {
            restricted: Box::new(sp(a.clone())),
            container: Box::new(sp(b.clone())),
        };
        let b_of_a = Type::Restriction {
            restricted: Box::new(sp(b)),
            container: Box::new(sp(a)),
        };
        assert_ne!(a_of_b, b_of_a);
    }

    #[test]
    fn a_restriction_names_which_operand_carries_the_gauge() {
        // in `A of B` the gauge is B's (the container's). A's gauge is
        // discarded, and A contributes only its predicate
        let mut i = Interner::new();
        let (a, b) = (named(i.intern("A")), named(i.intern("B")));
        let t = Type::Restriction {
            restricted: Box::new(sp(a)),
            container: Box::new(sp(b.clone())),
        };
        match t {
            Type::Restriction { container, .. } => assert_eq!(container.node, b),
            _ => panic!("expected a restriction"),
        }
    }

    #[test]
    fn an_array_distinguishes_the_growable_and_fixed_forms() {
        let mut i = Interner::new();
        let q = i.intern("qubit");
        // `[qubit]` asks the allocator for a register of run-time length
        let growable = Type::Array {
            elem: Box::new(sp(named(q))),
            len: None,
        };
        // `[qubit; N]` is a fixed register, the recommended form
        let fixed = Type::Array {
            elem: Box::new(sp(named(q))),
            len: Some(Box::new(sp(Expr::Path {
                path: Path::single(sp(i.intern("N"))),
                args: Vec::new(),
            }))),
        };
        assert_ne!(growable, fixed);
    }

    #[test]
    fn a_type_is_quantum_when_it_mentions_qubit_anywhere() {
        // however deeply nested the mention is
        let mut i = Interner::new();
        let q = i.intern("qubit");
        let u8_ = i.intern("u8");

        assert!(named(q).mentions_qubit(q));
        assert!(!named(u8_).mentions_qubit(q));

        // nested through every former that can hold a type
        let nested = Type::Tuple(vec![
            sp(named(u8_)),
            sp(Type::Array {
                elem: Box::new(sp(Type::Ref {
                    target: Box::new(sp(named(q))),
                })),
                len: None,
            }),
        ]);
        assert!(nested.mentions_qubit(q));

        // and through a generic argument
        let generic = Type::Path {
            path: Path::single(sp(i.intern("Reg"))),
            args: vec![sp(TArg::Type(sp(named(q))))],
        };
        assert!(generic.mentions_qubit(q));
    }

    #[test]
    fn dyn_and_void_are_never_quantum() {
        // a `dyn` may never hold a value of quantum type, so it can never be
        // quantum itself
        let mut i = Interner::new();
        let q = i.intern("qubit");
        assert!(!Type::Dyn.mentions_qubit(q));
        assert!(!Type::Void.mentions_qubit(q));
    }

    #[test]
    fn a_function_type_is_quantum_if_a_parameter_is() {
        // as in the type of an oracle parameter, `fn(*qubit, *qubit)`
        let mut i = Interner::new();
        let q = i.intern("qubit");
        let t = Type::Fn {
            params: vec![sp(Type::Ref {
                target: Box::new(sp(named(q))),
            })],
            ret: None,
        };
        assert!(t.mentions_qubit(q));
    }

    #[test]
    fn paths_report_their_final_segment() {
        let mut i = Interner::new();
        let p = Path {
            segments: vec![sp(i.intern("qpu")), sp(i.intern("gates"))],
        };
        assert_eq!(p.last(), Some(i.intern("gates")));
        assert!(!p.is_simple());
        assert!(Path::single(sp(i.intern("x"))).is_simple());
    }
}
