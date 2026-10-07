//! Growable arrays: their methods, and going through one by value.
//!
//! A growable array `[T]` owns its elements. What changes its length (`push`,
//! `pop`, `reserve`, `clear` and `truncate`) works on the array where it is,
//! so the array must be a place that may change: a binding that is not a
//! constant, or a field or element reached through a `*T`. A slice `*[T]` of
//! one has none of them.
//! It is a view of the elements with their count fixed in it, and cannot
//! reach the buffer to grow it or record a new length.
//!
//! `len` and `as_ref` look at the array, and `map` and `fold` go through its
//! elements, so all four work on a slice too. `filter` uses the array up: the
//! elements it keeps are moved into the result, and the rest destroyed.
//!
//! The function `map`, `filter` and `fold` take is a function, a closure, or a
//! reference to either. It is given each element by reference, `*T`, or by
//! value if it asks for a `T` that can be copied.
//!
//! `for x in xs` on a `[T]` moves each element out in turn; stopping early
//! destroys the ones not reached. `for x in &xs` sees each by reference and
//! leaves the array as it was.

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Access, Arg, Callee, Expr, ExprKind, GrowOp, Intrinsic, Ty};

use super::body::Checker;

/// The methods of a growable array.
pub const METHODS: [&str; 10] = [
    "len", "push", "pop", "reserve", "clear", "truncate", "as_ref", "map", "filter", "fold",
];

/// What a method does with the array.
enum Kind {
    /// Changes it where it is.
    Change(GrowOp),
    /// Goes through its elements.
    Each,
}

impl Checker<'_, '_> {
    /// `recv.name(args)` on a growable array or a slice.
    pub(super) fn array_method(
        &mut self,
        recv: Expr,
        name: Spanned<Symbol>,
        targs: &[Spanned<ast::TArg>],
        args: &[Spanned<ast::Expr>],
        span: Span,
    ) -> Expr {
        let text = self.name(name.node);
        let t = self.shallow(recv.ty);
        let (elem, owned) = match self.types().as_growable(t) {
            Some(e) => (e, true),
            None => (self.types().as_slice(t).expect("a growable array or a slice").1, false),
        };

        // a method it has, without generic arguments
        let internal = self.in_library() && matches!(text, "__take" | "__forget_front");
        if !METHODS.contains(&text) && !internal {
            self.check_only(args);
            let what = self.describe(t);
            let mut d = Diagnostic::new(Code::Es04)
                .with_message(format!("{what} has no method named `{text}`"))
                .at(name.span);
            d = match super::report::closest(text, METHODS) {
                Some(s) => d.with_help(format!("did you mean `{s}`?")),
                None => d.with_note(format!("its methods are {}", list(&METHODS))),
            };
            self.report(d);
            return Checker::error(span);
        }
        if !targs.is_empty() {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{text}` takes no generic arguments"))
                    .at(span),
            );
            return Checker::error(span);
        }

        // `len` and `as_ref` are done here; the rest change the array or visit each element
        let kind = match text {
            "push" => Kind::Change(GrowOp::Push),
            "pop" => Kind::Change(GrowOp::Pop),
            "reserve" => Kind::Change(GrowOp::Reserve),
            "clear" => Kind::Change(GrowOp::Clear),
            "truncate" => Kind::Change(GrowOp::Truncate),
            "__take" => Kind::Change(GrowOp::Take),
            "__forget_front" => Kind::Change(GrowOp::ForgetFront),
            "len" | "as_ref" => {
                if !args.is_empty() {
                    self.check_only(args);
                    self.report(
                        Diagnostic::new(Code::Es07)
                            .with_message(format!("`{text}` takes no arguments"))
                            .at(span),
                    );
                    return Checker::error(span);
                }
                return if text == "len" {
                    Expr::intrinsic(Intrinsic::Len, vec![recv], Ty::USIZE, span)
                } else if owned {
                    self.reference_to(recv, span)
                } else {
                    recv
                };
            }
            _ => Kind::Each,
        };
        match kind {
            Kind::Change(op) => {
                if !owned {
                    self.check_only(args);
                    let d = through_slice(text, name.span, recv.span);
                    self.report(d);
                    return Checker::error(span);
                }
                self.change(op, recv, elem, args, span)
            }
            Kind::Each => self.each(text, recv, elem, owned, args, span),
        }
    }

    /// `push`, `pop`, `reserve`, `clear`, `truncate`, or one of the library's
    /// own changes, on the array `recv`.
    fn change(&mut self, op: GrowOp, recv: Expr, elem: Ty, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let text = op.name();
        let params: &[Ty] = match op {
            GrowOp::Push => &[elem],
            GrowOp::Reserve | GrowOp::Truncate | GrowOp::Take | GrowOp::ForgetFront => &[Ty::USIZE],
            GrowOp::Pop | GrowOp::Clear => &[],
        };

        // the arguments, as many as it takes, each checked as what it should be
        let mut checked: Vec<Expr> = if args.len() == params.len() {
            args.iter().zip(params).map(|(a, &p)| self.expr_for(a, p)).collect()
        } else {
            args.iter().map(|a| self.expr(a)).collect()
        };
        if checked.len() != params.len() {
            let wanted = match params.len() {
                0 => "no arguments".to_owned(),
                _ if op == GrowOp::Push => "one argument, the element to add".to_owned(),
                _ => "one argument, a count".to_owned(),
            };
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`{text}` takes {wanted}, but this call gives {}", checked.len()))
                    .at(span),
            );
            return Checker::error(span);
        }
        checked = checked
            .into_iter()
            .zip(params)
            .map(|(a, &p)| self.coerce(a, p, &format!("the argument of `{text}`")))
            .collect();

        // an array that can be changed
        if recv.is_place() {
            let pa = self.place_access(&recv);
            if pa.access != Access::Write {
                let d = self.not_mutable(&recv, pa, &format!("`.{text}(…)` on"), span);
                self.report(d);
            }
        }

        // what it gives back
        let ty = match op {
            GrowOp::Pop => match self.opt_of(elem, span) {
                Some(t) => t,
                None => return Checker::error(span),
            },
            GrowOp::Take => elem,
            _ => Ty::Void,
        };
        Expr {
            kind: ExprKind::Growable {
                op,
                array: Box::new(recv),
                args: checked,
            },
            ty,
            span,
        }
    }

    /// `map`, `filter` or `fold`: a call of `core`'s function that does it.
    fn each(&mut self, text: &str, recv: Expr, elem: Ty, owned: bool, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let mut checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
        let fold = text == "fold";
        let wanted = if fold { 2 } else { 1 };
        if checked.len() != wanted {
            let what = if fold {
                "two arguments, the starting value and a function"
            } else {
                "one argument, a function"
            };
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`{text}` takes {what}, but this call gives {}", checked.len()))
                    .at(span),
            );
            return Checker::error(span);
        }
        let f = checked.pop().expect("the function");
        let init = checked.pop();
        let fty = self.settled(f.ty);
        if fty == Ty::Never {
            return Checker::error(span);
        }
        let callable = self.types().as_ref(fty).map_or(fty, |(_, x)| x);
        let sig = matches!(callable, Ty::Fn(_) | Ty::Closure(_))
            .then(|| self.types().as_sig(callable).map(|(p, r)| (p.to_vec(), r)))
            .flatten();
        let by_ref = self.types_mut().reference(Access::Write, elem);
        let shape = |me: &mut Self| {
            let e = me.show(elem);
            match text {
                "map" => format!("`fn(*{e}) -> U`"),
                "filter" => format!("`fn(*{e}) -> bool`"),
                _ => format!("`fn(A, *{e}) -> A`"),
            }
        };
        let Some((params, ret)) = sig.filter(|(p, _)| p.len() == wanted) else {
            let what = self.describe(fty);
            let shape = shape(self);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{text}` takes a function shaped {shape}, but this is {what}"))
                    .at(f.span)
                    .with_note("a function, a closure, or a reference to either will do"),
            );
            return Checker::error(span);
        };
        let item = self.settled(params[wanted - 1]);
        let copy = if item == by_ref {
            false
        } else if item == elem && self.types().is_copyable(elem) {
            true
        } else {
            let (have, e) = (self.describe(item), self.show(elem));
            let mut d = Diagnostic::new(Code::Es06)
                .with_message(format!("the function given to `{text}` takes {have} for each element, but the elements are `{e}`"))
                .at(f.span);
            d = if item == elem {
                d.with_note(format!("`{e}` cannot be copied, so each element is lent to the function"))
                    .with_help(format!("take the element as `*{e}`"))
            } else {
                d.with_help(format!("take the element as `*{e}`, or as `{e}` if it can be copied"))
            };
            self.report(d);
            return Checker::error(span);
        };
        if text == "filter" && self.settled(ret) != Ty::Bool && ret != Ty::Never {
            let what = self.describe(ret);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("the function given to `filter` says whether to keep an element, so it returns a `bool`, but this returns {what}"))
                    .at(f.span),
            );
            return Checker::error(span);
        }
        let helper = format!("__{text}{}", if copy { "_copy" } else { "" });
        match text {
            "filter" => {
                if !owned {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message("`filter` uses up the array it is called on, which a slice does not own")
                            .at(recv.span)
                            .with_help("call it on the growable array itself"),
                    );
                    return Checker::error(span);
                }
                self.core_call(&helper, vec![Arg::Type(elem), Arg::Type(fty)], vec![recv, f], span)
            }
            _ => {
                let view = if owned { self.reference_to(recv, span) } else { recv };
                // what `map` makes, or what `fold` accumulates
                let targs = vec![Arg::Type(elem), Arg::Type(ret), Arg::Type(fty)];
                let mut xs = vec![view];
                xs.extend(init);
                xs.push(f);
                self.core_call(&helper, targs, xs, span)
            }
        }
    }

    /// `for x in xs` on a growable array: through `core`'s `Drain`, which
    /// moves each element out.
    pub(super) fn drain(&mut self, xs: Expr, span: Span) -> Expr {
        let elem = self.types().as_growable(xs.ty).expect("a growable array");
        self.core_call("__drain", vec![Arg::Type(elem)], vec![xs], span)
    }

    /// `Opt<t>`.
    pub(super) fn opt_of(&mut self, t: Ty, span: Span) -> Option<Ty> {
        let Some(key) = self.interner.get("Opt").and_then(|s| self.cx.generic(s)) else {
            self.report(super::report::unsupported(span, "`Opt` without the `core` library"));
            return None;
        };
        self.cx.instantiate_adt(key, vec![Arg::Type(t)], span).map(Ty::Adt)
    }

    /// A call of the instance of `core`'s generic function `name` made with
    /// `targs`, with arguments already checked.
    pub(super) fn core_call(&mut self, name: &str, targs: Vec<Arg>, args: Vec<Expr>, span: Span) -> Expr {
        let Some(key) = self.interner.get(name).and_then(|s| self.cx.core_generic(s)) else {
            return self.unsupported(span, &format!("`{name}` without the `core` library"));
        };
        self.instance_call(key, targs, args, name, span)
    }

    /// A call of the instance of the generic function `key`, written `label`,
    /// made with `targs`, with arguments already checked.
    pub(super) fn instance_call(&mut self, key: super::items::GenericKey, targs: Vec<Arg>, args: Vec<Expr>, label: &str, span: Span) -> Expr {
        let Some(id) = self.cx.instantiate_fn(key, targs, span) else {
            return Checker::error(span);
        };
        let f = self.cx.unit.func(id);
        let (params, ret): (Vec<Ty>, Ty) = (f.param_types().collect(), f.ret);
        let args = args
            .into_iter()
            .zip(&params)
            .map(|(a, &p)| self.coerce(a, p, &format!("an argument of `{label}`")))
            .collect();
        Expr::call(Callee::Fn(id), args, ret, span)
    }
}

/// `push` and the rest called through a slice.
fn through_slice(text: &str, at: Span, slice: Span) -> Diagnostic {
    Diagnostic::new(Code::Ec08)
        .with_message(format!("`{text}` changes an array's length, which cannot be done through a slice"))
        .at(at)
        .also(slice, "this is a slice `*[T]`")
        .with_note(
            "a slice is a view of elements with their count fixed in it: it cannot reach the \
             array's buffer to grow it, or record a new length",
        )
        .with_help(format!(
            "call `{text}` on the growable array itself; a function that must change one takes \
             it by value and returns it"
        ))
}

/// Names in backquotes, joined with commas and a final "and".
fn list(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => quoted.concat(),
    }
}
