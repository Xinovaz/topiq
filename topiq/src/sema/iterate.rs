//! `for pattern in iterable { … }`, and ranges as values.
//!
//! What a `for` loop goes through depends on what it is given:
//!
//! - **A range written in place**, `for i in 0..n`, counts, and stops at the
//!   bound without computing a value past it, so a range ending at its type's
//!   maximum does not overflow.
//! - **A reference to an array, or a slice**, gives a reference to each
//!   element in turn: `for x in &xs` sees `*T`, and changes nothing.
//! - **An array**, fixed or growable, gives its elements themselves, moving
//!   each out: the array is used up.
//! - **Anything else** is iterated through its type's operator methods:
//!   `$iter` makes an iterator from it (a value that already has `$next`
//!   is its own), and each `$next` gives `Some(item)` or `None` at the end.
//!   A `Range` value is iterated this way, by the methods `core` gives it.
//!
//! The pattern is any pattern that matches every value, as for `let`.
//!
//! `a..b` and `a..=b` outside a `for` are values of `core`'s `Range<T>`.

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Arm, Block, Expr, ExprKind, IntTy, Intrinsic, Pat, Stmt, Ty, Value};

use super::body::{Checker, LoopFrame, LoopKind};
use super::refutable;
use super::scope::Def;

impl Checker<'_, '_> {
    /// `for pattern in iterable body`.
    pub(super) fn for_loop(
        &mut self,
        pattern: &Spanned<ast::Pattern>,
        iter: &Spanned<ast::Expr>,
        body: &ast::Block,
        span: Span,
    ) -> Expr {
        if let ast::Expr::Range {
            start: Some(start),
            end: Some(end),
            inclusive,
        } = &iter.node
        {
            return self.range_loop(pattern, start, end, *inclusive, iter.span, body, span);
        }
        let e = self.expr(iter);
        let t = self.settled(e.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        // through a reference to an array, or a slice: references to the
        // elements
        let viewed = self
            .types()
            .as_slice(t)
            .map(|(access, elem)| (access, elem, None))
            .or_else(|| {
                let (access, target) = self.types().as_ref(t)?;
                let (elem, n) = self.types().as_array(target)?;
                Some((access, elem, Some(n)))
            });
        if let Some((access, elem, len)) = viewed {
            return self.element_loop(pattern, e, Some(access), elem, len, body, span);
        }
        if let Some((elem, n)) = self.types().as_array(t) {
            return self.element_loop(pattern, e, None, elem, Some(n), body, span);
        }
        if let Ty::Adt(_) = t {
            return self.method_loop(pattern, e, iter.span, body, span);
        }
        if let Ty::Growable(_) = t {
            let d = self.drain(e, iter.span);
            return self.method_loop(pattern, d, iter.span, body, span);
        }
        let what = self.describe(t);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("{what} cannot be gone through with `for`"))
                .at(iter.span)
                .with_note(
                    "`for` goes through a range, an array, a slice, or a value whose type has \
                     `$iter` or `$next`",
                ),
        );
        Checker::error(span)
    }

    /// Binds a `for` loop's pattern to each value, `init`, as the first
    /// statement of its body. The names it binds are visible in the body.
    fn bind_each(&mut self, pattern: &Spanned<ast::Pattern>, ty: Ty) -> Option<Pat> {
        let mut bound = Vec::new();
        let pat = self.pattern(pattern, ty, &mut bound);
        if !refutable::is_irrefutable(&pat, self.types()) {
            self.report(
                Diagnostic::new(Code::Es13)
                    .with_message("this pattern does not match every value, so `for` cannot use it")
                    .at(pattern.span)
                    .with_note("a `for` loop has nowhere to go with a value its pattern does not match")
                    .with_help("bind each value with a name, and `match` on it inside the loop"),
            );
            return None;
        }
        Some(pat)
    }

    /// The loop body.
    fn each_body(&mut self, id: crate::tir::LoopId, body: &ast::Block, span: Span) -> Block {
        self.loops.push(LoopFrame {
            id,
            kind: LoopKind::For,
            value: None,
        });
        let b = self.loop_body_of(body, span, "a `for` loop");
        self.loops.pop();
        b
    }

    /// `for pattern in a..b` and `a..=b`.
    #[allow(clippy::too_many_arguments)]
    fn range_loop(
        &mut self,
        pattern: &Spanned<ast::Pattern>,
        start: &Spanned<ast::Expr>,
        end: &Spanned<ast::Expr>,
        inclusive: bool,
        at: Span,
        body: &ast::Block,
        span: Span,
    ) -> Expr {
        // two ends of one integer type
        let start = self.expr(start);
        let end = self.expr(end);
        let ty = match self.unify(start.ty, end.ty) {
            Ok(t) if matches!(self.shallow(t), Ty::Int(_) | Ty::Infer(_) | Ty::Never) => t,
            _ => {
                let (a, b) = (self.describe(start.ty), self.describe(end.ty));
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("a range needs two integers of one type, but this one runs from {a} to {b}"))
                        .at(at),
                );
                Ty::Never
            }
        };

        // the loop variable; any other pattern is matched against it at the top of the body
        let id = self.fresh_loop();
        self.scopes.enter();
        let (var, first) = match &pattern.node {
            ast::Pattern::Binding(n) => {
                let var = self.new_local(n.node, ty, false, n.span);
                self.scopes.bind(n.node, Def::Local(var));
                (var, None)
            }
            ast::Pattern::Wildcard => (self.new_local(Symbol::EMPTY, ty, false, pattern.span), None),
            _ => {
                let var = self.new_local(Symbol::EMPTY, ty, false, pattern.span);
                let pat = self.bind_each(pattern, ty);
                (var, pat)
            }
        };

        // the body
        let mut b = self.each_body(id, body, span);
        if let Some(pat) = first {
            let init = Expr::local(var, ty, pattern.span);
            b.stmts.insert(0, Stmt::LetPat { pat, init });
        }
        self.scopes.leave();
        Expr {
            kind: ExprKind::ForRange {
                id,
                var,
                start: Box::new(start),
                end: Box::new(end),
                inclusive,
                body: Box::new(b),
            },
            ty: Ty::Void,
            span,
        }
    }

    /// Goes through the elements of an array held in `e` (or of the array or
    /// slice `e` refers to, when `access` says how) by index.
    #[allow(clippy::too_many_arguments)]
    fn element_loop(
        &mut self,
        pattern: &Spanned<ast::Pattern>,
        e: Expr,
        access: Option<crate::tir::Access>,
        elem: Ty,
        len: Option<u64>,
        body: &ast::Block,
        span: Span,
    ) -> Expr {
        let held_ty = e.ty;
        self.scopes.enter();
        let held = self.new_local(Symbol::EMPTY, held_ty, false, e.span);
        let index = self.new_local(Symbol::EMPTY, Ty::USIZE, false, pattern.span);
        let local = |l, ty| Expr::local(l, ty, span);
        // the array itself: held, or what the held reference refers to
        let base = match (access, len) {
            (Some(_), Some(_)) => {
                let target = self.types().as_ref(held_ty).expect("a reference").1;
                Expr::deref(local(held, held_ty), target, span)
            }
            _ => local(held, held_ty),
        };
        let place = Expr {
            kind: ExprKind::Index {
                base: Box::new(base),
                index: Box::new(local(index, Ty::USIZE)),
            },
            ty: elem,
            span,
        };
        // a pattern that takes the element apart looks through the
        // reference, as `match` does; a name binds the reference itself
        let destructures = !matches!(pattern.node, ast::Pattern::Binding(_) | ast::Pattern::Wildcard);
        let (item, item_ty) = match access {
            Some(a) if !destructures => {
                let r = self.types_mut().reference(a, elem);
                (
                    Expr {
                        kind: ExprKind::Ref(Box::new(place)),
                        ty: r,
                        span,
                    },
                    r,
                )
            }
            _ => (place, elem),
        };
        let end = match len {
            Some(n) => Expr::constant(Value::Int(i128::from(n), IntTy::USIZE), Ty::USIZE, span),
            None => Expr::intrinsic(Intrinsic::Len, vec![local(held, held_ty)], Ty::USIZE, span),
        };
        let id = self.fresh_loop();
        let pat = self.bind_each(pattern, item_ty);
        let mut b = self.each_body(id, body, span);
        if let Some(pat) = pat {
            b.stmts.insert(0, Stmt::LetPat { pat, init: item });
        }
        self.scopes.leave();
        let lp = Expr {
            kind: ExprKind::ForRange {
                id,
                var: index,
                start: Box::new(Expr::constant(Value::Int(0, IntTy::USIZE), Ty::USIZE, span)),
                end: Box::new(end),
                inclusive: false,
                body: Box::new(b),
            },
            ty: Ty::Void,
            span,
        };
        Expr::let_then(held, e, lp, Ty::Void, span)
    }

    /// Goes through a value by its type's `$iter` and `$next`.
    fn method_loop(&mut self, pattern: &Spanned<ast::Pattern>, e: Expr, at: Span, body: &ast::Block, span: Span) -> Expr {
        let t = self.shallow(e.ty);
        let iterator = if let Some((entry, label)) = self.operator_method(t, "iter") {
            self.call_method(entry, Some(e), &[], Vec::new(), &label, at)
        } else if self.operator_method(t, "next").is_some() {
            e
        } else {
            let ty = self.show(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{ty}` cannot be gone through with `for`: it defines neither `$iter` nor `$next`"))
                    .at(at)
                    .with_help(format!(
                        "give `{ty}` a `$iter` that makes an iterator, or a `$next(self: *{ty}) -> Opt<…>`"
                    )),
            );
            return Checker::error(span);
        };
        let it_ty = self.settled(iterator.ty);
        if it_ty == Ty::Never {
            return Checker::error(span);
        }
        self.scopes.enter();
        let it = self.new_local(Symbol::EMPTY, it_ty, false, at);
        let Some((entry, label)) = self.operator_method(it_ty, "next") else {
            let ty = self.show(it_ty);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("the iterator `$iter` makes, a `{ty}`, has no `$next`"))
                    .at(at),
            );
            self.scopes.leave();
            return Checker::error(span);
        };
        let recv = Expr::local(it, it_ty, at);
        let next = self.call_method(entry, Some(recv), &[], Vec::new(), &label, at);
        let next_ty = self.settled(next.ty);
        // `$next` gives `Some(item)`, or `None` at the end
        let opt = match next_ty {
            Ty::Adt(id) => {
                let def = self.adt(id).clone();
                let is_opt = self.name(def.name) == "Opt" && def.origin.as_deref().is_none_or(|o| o == "core");
                is_opt.then_some(id)
            }
            _ => None,
        };
        let Some(opt) = opt else {
            if next_ty != Ty::Never {
                let what = self.describe(next_ty);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`$next` should give an `Opt` of the next item, but gives {what}"))
                        .at(at),
                );
            }
            self.scopes.leave();
            return Checker::error(span);
        };
        let def = self.adt(opt).clone();
        let some = def.variant_index(self.interner.get("Some").unwrap_or(Symbol::EMPTY)).unwrap_or(0);
        let none = def.variant_index(self.interner.get("None").unwrap_or(Symbol::EMPTY)).unwrap_or(1);
        let item_ty = def.variants()[some as usize].fields[0].ty;
        let id = self.fresh_loop();
        let pat = self.bind_each(pattern, item_ty);
        let b = self.each_body(id, body, span);
        self.scopes.leave();
        let item_pat = pat.unwrap_or(Pat::wild(item_ty, span));
        let arms = vec![
            Arm {
                pat: Pat::variant(some, vec![(0, item_pat)], next_ty, span),
                body: Expr {
                    ty: b.ty,
                    kind: ExprKind::Block(Box::new(b)),
                    span,
                },
            },
            Arm {
                pat: Pat::variant(none, Vec::new(), next_ty, span),
                body: Expr {
                    kind: ExprKind::Break { target: id, value: None },
                    ty: Ty::Never,
                    span,
                },
            },
        ];
        let step = Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(next),
                arms,
            },
            ty: Ty::Void,
            span,
        };
        let lp = Expr {
            kind: ExprKind::Loop {
                id,
                body: Box::new(Block {
                    stmts: vec![Stmt::Expr(step)],
                    value: None,
                    ty: Ty::Void,
                    span,
                }),
            },
            ty: Ty::Void,
            span,
        };
        Expr::let_then(it, iterator, lp, Ty::Void, span)
    }

    /// `a..b` or `a..=b` as a value: `core`'s `Range`.
    pub(super) fn range_value(
        &mut self,
        start: Option<&Spanned<ast::Expr>>,
        end: Option<&Spanned<ast::Expr>>,
        inclusive: bool,
        span: Span,
    ) -> Expr {
        let (Some(start), Some(end)) = (start, end) else {
            for e in [start, end].into_iter().flatten() {
                self.expr(e);
            }
            let dots = if inclusive { "..=" } else { ".." };
            self.report(
                Diagnostic::new(Code::Es02)
                    .with_message(format!("a range written with `{dots}` has a value on each side"))
                    .at(span)
                    .with_note(
                        "a range is from its start to its end, both written: a range with an end \
                         left open would have no value to stop at or to start from",
                    )
                    .with_help(format!("write both ends, as in `0{dots}n`")),
            );
            return Checker::error(span);
        };

        // two ends of one type
        let s = self.expr(start);
        let e = self.expr(end);
        let t = match self.unify(s.ty, e.ty) {
            Ok(t) => t,
            Err(_) => {
                let (a, b) = (self.describe(s.ty), self.describe(e.ty));
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("a range runs between two values of one type, but these are {a} and {b}"))
                        .at(span),
                );
                return Checker::error(span);
            }
        };

        // a `Range<t>` holding them
        let t = self.settled(t);
        let Some(key) = self.interner.get("Range").and_then(|s| self.cx.generic(s)) else {
            return self.unsupported(span, "ranges as values without the `core` library");
        };
        let Some(id) = self.cx.instantiate_adt(key, vec![crate::tir::Arg::Type(t)], span) else {
            return Checker::error(span);
        };
        let s = self.coerce(s, t, "the start of a range");
        let e = self.coerce(e, t, "the end of a range");
        Expr {
            kind: ExprKind::StructLit {
                fields: vec![(0, s), (1, e), (2, Expr::constant(Value::Bool(inclusive), Ty::Bool, span))],
            },
            ty: Ty::Adt(id),
            span,
        }
    }
}
