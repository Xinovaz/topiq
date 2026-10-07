//! `++` and `--`: an integer place changed by one.
//!
//! `++x` and `--x` give the place's value after the change, and `x++` and
//! `x--` its value before. The place is evaluated once, through a hidden
//! reference to it, and the change is the `+= 1` or `-= 1` it stands for, so
//! an increment past the type's range aborts as any overflow does.
//!
//! # Rules
//!
//! - **Only integers step.** Any other type is `ES22`, since stepping by one
//!   has no single meaning for a float, an exact scalar or a structure.
//! - **The operand is a place** that may be changed, as for assignment:
//!   `ES08` for a value and `EC03` for a constant. `--` is one token, so
//!   `--x` is never a double negation; that is written `-(-x)`.
//! - **Quantum data does not step.** A qubit or a register has no successor
//!   (`EQ18`), and a `quint` steps only by an arithmetic that wraps, which is
//!   written `wrapping_inc` and `wrapping_dec` (`EQ19`).

use crate::ast::{self, StepOp};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Access, BinOp, Block, Expr, ExprKind, IntTy, Stmt, Ty, Value};

use super::body::Checker;

/// Whether evaluating `e` twice does exactly what evaluating it once does:
/// it reads and computes, and calls, assigns, allocates and branches on
/// nothing.
pub(super) fn evaluates_purely(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Const(_) | ExprKind::Local(_) | ExprKind::Global(_) => true,
        ExprKind::Unary { operand: x, .. }
        | ExprKind::Cast { expr: x, .. }
        | ExprKind::Field { base: x, .. }
        | ExprKind::Deref(x)
        | ExprKind::Coerce(x) => evaluates_purely(x),
        ExprKind::Binary { lhs: a, rhs: b, .. }
        | ExprKind::Logical { lhs: a, rhs: b, .. }
        | ExprKind::Index { base: a, index: b } => evaluates_purely(a) && evaluates_purely(b),
        _ => false,
    }
}

impl Checker<'_, '_> {
    /// `++x`, `x++`, `--x` or `x--`.
    pub(super) fn step(&mut self, op: StepOp, post: bool, place: &Spanned<ast::Expr>, span: Span) -> Expr {
        let p = self.expr(place);
        if p.ty == Ty::Never {
            return Checker::error(span);
        }
        let text = op.text();
        if !p.is_place() {
            let mut d = Diagnostic::new(Code::Es08)
                .with_message(format!("`{text}` changes a place, and this is a value"))
                .at(place.span)
                .with_note("a binding, a field or element of one, or what a `*T` reference refers to can be changed");
            if op == StepOp::Dec && !post {
                d = d.with_help("`--` is one operator; for a double negation write `-(-x)`");
            }
            self.report(d);
            return Checker::error(span);
        }
        let t = self.shallow(p.ty);
        if self.types().is_quantum(t) {
            let d = self.quantum_step(op, t, place.span);
            self.report(d);
            return Checker::error(span);
        }
        if !matches!(t, Ty::Int(_) | Ty::Infer(_)) || self.is_floating(t) {
            let what = self.describe(t);
            let by = if self.is_floating(t) { "1.0" } else { "1" };
            let sign = if op == StepOp::Inc { "+" } else { "-" };
            self.report(
                Diagnostic::new(Code::Es22)
                    .with_message(format!("`{text}` steps an integer, not {what}"))
                    .at(place.span)
                    .with_help(format!("write `x {sign}= {by}`")),
            );
            return Checker::error(span);
        }
        let pa = self.place_access(&p);
        if pa.access != Access::Write {
            let d = self.not_mutable(&p, pa, &format!("`{text}` on"), place.span);
            self.report(d);
            return Checker::error(span);
        }

        // `let r = &place;` evaluates the place once
        let target = p.ty;
        let r_ty = self.types_mut().reference(Access::Write, target);
        let r = self.new_local(Symbol::EMPTY, r_ty, false, place.span);
        let reference = Expr {
            kind: ExprKind::Ref(Box::new(p)),
            ty: r_ty,
            span: place.span,
        };
        let through = |at: Span| Expr::deref(Expr::local(r, r_ty, at), target, at);
        let mut stmts = vec![Stmt::Let {
            local: r,
            init: Some(reference),
        }];

        // the value before, kept for `x++`
        let old = post.then(|| {
            let old = self.new_local(Symbol::EMPTY, target, false, place.span);
            stmts.push(Stmt::Let {
                local: old,
                init: Some(through(place.span)),
            });
            old
        });

        // `*r = *r ± 1`, checked for overflow like any sum
        let one = Expr::constant(Value::Int(1, IntTy::I64), self.infer.fresh(), span);
        let bop = if op == StepOp::Inc { BinOp::Add } else { BinOp::Sub };
        let changed = self.binary_checked(bop, through(place.span), one, span);
        stmts.push(Stmt::Expr(Expr {
            kind: ExprKind::Assign {
                place: Box::new(through(place.span)),
                value: Box::new(changed),
            },
            ty: Ty::Void,
            span,
        }));

        let value = match old {
            Some(old) => Expr::local(old, target, span),
            None => through(span),
        };
        Expr {
            ty: target,
            kind: ExprKind::Block(Box::new(Block {
                stmts,
                value: Some(Box::new(value)),
                ty: target,
                span,
            })),
            span,
        }
    }

    /// `place op= value` for a place whose evaluation does something, such as
    /// `a[i++] += 1` or `a[f()] += 1`: the value, then a reference to the
    /// place, each evaluated once, and the place changed through the
    /// reference.
    pub(super) fn assign_once(&mut self, op: BinOp, place: Expr, value: Expr, span: Span) -> Expr {
        let target = place.ty;
        let v = self.new_local(Symbol::EMPTY, value.ty, false, value.span);
        let v_ty = value.ty;
        let v_span = value.span;
        let r_ty = self.types_mut().reference(Access::Write, target);
        let r = self.new_local(Symbol::EMPTY, r_ty, false, place.span);
        let at = place.span;
        let through = |at: Span| Expr::deref(Expr::local(r, r_ty, at), target, at);
        let reference = Expr {
            kind: ExprKind::Ref(Box::new(place)),
            ty: r_ty,
            span: at,
        };
        let changed = self.binary_checked(op, through(at), Expr::local(v, v_ty, v_span), span);
        let stmts = vec![
            // the value first, as in every assignment
            Stmt::Let {
                local: v,
                init: Some(value),
            },
            Stmt::Let {
                local: r,
                init: Some(reference),
            },
            Stmt::Expr(Expr {
                kind: ExprKind::Assign {
                    place: Box::new(through(at)),
                    value: Box::new(changed),
                },
                ty: Ty::Void,
                span,
            }),
        ];
        Expr {
            ty: Ty::Void,
            kind: ExprKind::Block(Box::new(Block {
                stmts,
                value: None,
                ty: Ty::Void,
                span,
            })),
            span,
        }
    }

    /// Why `++` or `--` has no meaning on the quantum type `t`.
    fn quantum_step(&mut self, op: StepOp, t: Ty, at: Span) -> Diagnostic {
        let text = op.text();
        if self.quint_width(t).is_some() {
            let name = if op == StepOp::Inc { "wrapping_inc" } else { "wrapping_dec" };
            return Diagnostic::new(Code::Eq19)
                .with_message(format!("`{text}` on a `quint` would wrap, and Topiq does not wrap silently"))
                .at(at)
                .with_note(
                    "a circuit cannot stop in one branch of a superposition, so arithmetic on qubits \
                     always wraps at 2^N; the classical types abort instead",
                )
                .with_help(format!("write `x = {name}(x);`"));
        }
        let what = self.describe(t);
        Diagnostic::new(Code::Eq18)
            .with_message(format!("`{text}` has no meaning on {what}"))
            .at(at)
            .with_note("a qubit or a register holds bits, not a number; a `quint<N>` holds a number")
            .with_help("to flip a qubit write `t ^= true`")
    }
}
