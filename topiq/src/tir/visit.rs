//! Walking every expression, block and pattern of a body.
//!
//! Several passes need to reach every node (settling inferred types, folding
//! constants, looking for `match` expressions to check) but only care about a
//! few kinds. A pass implements [`Visit`] or [`VisitMut`], overrides the
//! methods for what it cares about, and calls the matching `walk_*` function
//! to carry on into the children. The walk visits children in evaluation
//! order.

use super::{Arm, Block, Expr, ExprKind, Pat, PatKind, Stmt};

/// A pass that reads a body.
pub trait Visit {
    /// Called for every expression; the default walks into its children.
    fn expr(&mut self, e: &Expr) {
        walk_expr(self, e);
    }

    /// Called for every block; the default walks into its statements.
    fn block(&mut self, b: &Block) {
        walk_block(self, b);
    }

    /// Called for every pattern; the default walks into its sub-patterns.
    fn pat(&mut self, p: &Pat) {
        walk_pat(self, p);
    }
}

/// A pass that may rewrite a body.
pub trait VisitMut {
    /// Called for every expression; the default walks into its children.
    fn expr(&mut self, e: &mut Expr) {
        walk_expr_mut(self, e);
    }

    /// Called for every block; the default walks into its statements.
    fn block(&mut self, b: &mut Block) {
        walk_block_mut(self, b);
    }

    /// Called for every pattern; the default walks into its sub-patterns.
    fn pat(&mut self, p: &mut Pat) {
        walk_pat_mut(self, p);
    }
}

/// Defines the walks for [`Visit`], or with `mut` for [`VisitMut`], which
/// differ only in how they hold what they walk.
macro_rules! walks {
    ($visit:ident, $expr:ident, $block:ident, $pat:ident, $how:literal $(, $mut:tt)?) => {
        #[doc = concat!("Visits an expression's children", $how, ".")]
        pub fn $expr<V: $visit + ?Sized>(v: &mut V, e: &$($mut)? Expr) {
            match &$($mut)? e.kind {
                ExprKind::Const(_)
                | ExprKind::Local(_)
                | ExprKind::Global(_)
                | ExprKind::FnRef(_)
                | ExprKind::ThunkRef(_)
                | ExprKind::Continue { .. } => {}
                ExprKind::Call { args, .. }
                | ExprKind::Intrinsic { args, .. }
                | ExprKind::Quantum { args, .. }
                | ExprKind::ArrayLit(args)
                | ExprKind::Closure { captures: args, .. } => {
                    for a in args {
                        v.expr(a);
                    }
                }
                ExprKind::Growable { array: callee, args, .. } | ExprKind::IndirectCall { callee, args } => {
                    v.expr(callee);
                    for a in args {
                        v.expr(a);
                    }
                }
                ExprKind::Unary { operand: x, .. }
                | ExprKind::Cast { expr: x, .. }
                | ExprKind::Field { base: x, .. }
                | ExprKind::Deref(x)
                | ExprKind::Ref(x)
                | ExprKind::Coerce(x)
                | ExprKind::FnAsClosure(x)
                | ExprKind::Grow(x)
                | ExprKind::ArrayRepeat { elem: x, .. } => v.expr(x),
                ExprKind::Binary { lhs, rhs, .. } | ExprKind::Logical { lhs, rhs, .. } => {
                    v.expr(lhs);
                    v.expr(rhs);
                }
                ExprKind::Index { base, index } => {
                    v.expr(base);
                    v.expr(index);
                }
                ExprKind::StructLit { fields } | ExprKind::Variant { fields, .. } => {
                    for (_, f) in fields {
                        v.expr(f);
                    }
                }
                ExprKind::Assign { place, value } => {
                    v.expr(value);
                    v.expr(place);
                }
                ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => v.block(b),
                ExprKind::If { cond, then, els } => {
                    v.expr(cond);
                    v.block(then);
                    if let Some(e) = els {
                        v.expr(e);
                    }
                }
                ExprKind::Match { scrutinee, arms } => {
                    v.expr(scrutinee);
                    for Arm { pat, body } in arms {
                        v.pat(pat);
                        v.expr(body);
                    }
                }
                ExprKind::While { cond, body, .. } => {
                    v.expr(cond);
                    v.block(body);
                }
                ExprKind::ForRange {
                    start, end, body, ..
                } => {
                    v.expr(start);
                    v.expr(end);
                    v.block(body);
                }
                ExprKind::Break { value, .. } | ExprKind::Return(value) => {
                    if let Some(x) = value {
                        v.expr(x);
                    }
                }
            }
        }

        #[doc = concat!("Visits a block's statements and value", $how, ".")]
        pub fn $block<V: $visit + ?Sized>(v: &mut V, b: &$($mut)? Block) {
            for s in &$($mut)? b.stmts {
                match s {
                    Stmt::Let { init, .. } => {
                        if let Some(e) = init {
                            v.expr(e);
                        }
                    }
                    Stmt::LetPat { pat, init } => {
                        v.pat(pat);
                        v.expr(init);
                    }
                    Stmt::Expr(e) => v.expr(e),
                }
            }
            if let Some(x) = &$($mut)? b.value {
                v.expr(x);
            }
        }

        #[doc = concat!("Visits a pattern's sub-patterns", $how, ".")]
        pub fn $pat<V: $visit + ?Sized>(v: &mut V, p: &$($mut)? Pat) {
            match &$($mut)? p.kind {
                PatKind::Wild | PatKind::Bind(_) | PatKind::BindRef(_) | PatKind::Const(_) => {}
                PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                    for (_, f) in fields {
                        v.pat(f);
                    }
                }
            }
        }
    };
}

walks!(Visit, walk_expr, walk_block, walk_pat, "");
walks!(VisitMut, walk_expr_mut, walk_block_mut, walk_pat_mut, ", mutably", mut);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::{SourceId, Span};
    use crate::tir::{IntTy, LocalId, Ty, Value};

    fn sp() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    fn e(kind: ExprKind) -> Expr {
        Expr {
            kind,
            ty: Ty::Bool,
            span: sp(),
        }
    }

    struct Count {
        exprs: usize,
        pats: usize,
    }

    impl Visit for Count {
        fn expr(&mut self, x: &Expr) {
            self.exprs += 1;
            walk_expr(self, x);
        }
        fn pat(&mut self, p: &Pat) {
            self.pats += 1;
            walk_pat(self, p);
        }
    }

    #[test]
    fn a_match_is_walked_through_its_scrutinee_patterns_and_arms() {
        let one = Expr::constant(Value::Int(1, IntTy::I32), Ty::Int(IntTy::I32), sp());
        let m = e(ExprKind::Match {
            scrutinee: Box::new(e(ExprKind::Local(LocalId(0)))),
            arms: vec![Arm {
                pat: Pat {
                    kind: PatKind::Variant {
                        variant: 0,
                        fields: vec![(
                            0,
                            Pat {
                                kind: PatKind::Wild,
                                ty: Ty::Bool,
                                span: sp(),
                            },
                        )],
                    },
                    ty: Ty::Bool,
                    span: sp(),
                },
                body: one,
            }],
        });
        let mut c = Count { exprs: 0, pats: 0 };
        c.expr(&m);
        assert_eq!(c.exprs, 3, "the match, its scrutinee and one arm body");
        assert_eq!(c.pats, 2, "the variant pattern and its field");
    }

    #[test]
    fn an_assignment_visits_its_value_before_its_place() {
        struct Order(Vec<String>);
        impl Visit for Order {
            fn expr(&mut self, x: &Expr) {
                if let ExprKind::Local(l) = x.kind {
                    self.0.push(format!("local {}", l.0));
                }
                walk_expr(self, x);
            }
        }
        let a = e(ExprKind::Assign {
            place: Box::new(e(ExprKind::Local(LocalId(1)))),
            value: Box::new(e(ExprKind::Local(LocalId(2)))),
        });
        let mut o = Order(vec![]);
        o.expr(&a);
        assert_eq!(o.0, ["local 2", "local 1"]);
    }
}
