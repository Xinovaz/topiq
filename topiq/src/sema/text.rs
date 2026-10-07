//! Text becoming the `core` library's `string`, and a `string` lending its
//! characters.
//!
//! A `string` owns its characters and has operators: `==` compares them, `+`
//! puts two end to end. Text written any other way (a string literal, a
//! `*[char]`, a `[char]`) becomes a `string` wherever one is wanted: as a
//! value, as the operand of a `string`'s operator, on the left of an
//! operator whose right operand is a `string`, so that `"a" + s` and
//! `s == "a"` both read as text, and where a `*const string` is wanted, as
//! a `string` kept until the function returns. A slice's characters are
//! copied, since another owns them; an owned `[char]` is moved.
//!
//! `==` and `!=` between a `string` and plain text compare the characters
//! where they are, making no `string` of the text.
//!
//! The other way, a reference to a `string` is a view of its characters
//! wherever a slice of them is wanted: `print(&s)` prints them, without a
//! copy. A `*const string` gives only a `*const [char]`.
//!
//! Two slices compared with `==` remain two references compared, which asks
//! whether they are the same characters in the same place; a `string` on
//! either side makes it a comparison of text.

use crate::span::Span;
use crate::tir::{Access, BinOp, Callee, Expr, ExprKind, Ty, UnOp};

use super::body::Checker;
use super::imports;
use super::scope::Def;

impl Checker<'_, '_> {
    /// Whether `t` is the `core` library's `string`.
    pub(super) fn is_string(&self, t: Ty) -> bool {
        let Ty::Adt(id) = self.shallow(t) else { return false };
        let def = self.types().adt(id);
        def.origin.as_deref() == Some("core") && self.name(def.name) == "string"
    }

    /// Whether `t` is text that is not a `string`: a slice or growable
    /// array of characters.
    pub(super) fn is_plain_text(&self, t: Ty) -> bool {
        let t = self.shallow(t);
        let types = self.types();
        types.as_slice(t).is_some_and(|(_, e)| e == Ty::Char) || types.as_growable(t) == Some(Ty::Char)
    }

    /// `e` as a `string`, if it is plain text: copied from a slice, moved
    /// from a `[char]`. Anything else is given back as it is.
    pub(super) fn text_to_string(&mut self, e: Expr) -> Result<Expr, Expr> {
        if !self.is_plain_text(e.ty) {
            return Err(e);
        }
        let span = e.span;
        let owned = self.types().as_growable(self.shallow(e.ty)).is_some();
        let name = if owned { "__string_of_chars" } else { "__string_of" };
        Ok(self.core_fn_call(name, vec![e], span))
    }

    /// `l == r` or `l != r` between a `string` and plain text, either way
    /// round: their characters compared where they are, without making a
    /// `string` of the text. Both operands are given back otherwise.
    pub(super) fn text_equality(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Result<Expr, Box<(Expr, Expr)>> {
        let mixed = (self.is_string(l.ty) && self.is_plain_text(r.ty)) || (self.is_plain_text(l.ty) && self.is_string(r.ty));
        if !matches!(op, BinOp::Eq | BinOp::Ne) || !mixed {
            return Err(Box::new((l, r)));
        }
        let (a, b) = (self.chars_of(l), self.chars_of(r));
        let same = self.core_fn_call("__text_eq", vec![a, b], span);
        if op == BinOp::Eq {
            return Ok(same);
        }
        Ok(Expr {
            kind: ExprKind::Unary { op: UnOp::Not, operand: Box::new(same) },
            ty: Ty::Bool,
            span,
        })
    }

    /// The characters of `e`, a `string` or plain text, as a slice: a slice
    /// as it is, and anything else referred to where it is.
    fn chars_of(&mut self, e: Expr) -> Expr {
        let span = e.span;
        if self.types().as_slice(self.shallow(e.ty)).is_some() {
            return e;
        }
        let r = self.reference_to(e, span);
        let want = self.types_mut().slice(Access::Const, Ty::Char);
        self.coerce(r, want, "the characters compared")
    }

    /// A call of `core`'s function `name`, which is not generic, with
    /// `args`, as `core::name(args…)` would be.
    fn core_fn_call(&mut self, name: &str, args: Vec<Expr>, span: Span) -> Expr {
        let found = (|| {
            let core = self.cx.scope.get_unit(self.interner.get("core")?)?;
            imports::import_value(self.cx, core, self.interner.get(name)?, span)
        })();
        let callee = match found {
            Some(Ok(Def::Fn(id))) => Callee::Fn(id),
            Some(Ok(Def::Extern(id))) => Callee::Extern(id),
            Some(Err(d)) => {
                self.report(*d);
                return Checker::error(span);
            }
            _ => return self.unsupported(span, &format!("`core::{name}` without the `core` library")),
        };
        let (params, ret) = match callee {
            Callee::Fn(id) => {
                let f = self.cx.unit.func(id);
                (f.param_types().collect::<Vec<_>>(), f.ret)
            }
            Callee::Extern(id) => {
                let f = self.cx.unit.extern_fn(id);
                (f.params.clone(), f.ret)
            }
        };
        self.cx.use_callee(callee, span);
        let args = args
            .into_iter()
            .zip(params)
            .map(|(a, p)| self.coerce(a, p, &format!("an argument of `core::{name}`")))
            .collect();
        Expr::call(callee, args, ret, span)
    }

    /// Text where a `*const string` is wanted: a `string` made of it, which
    /// the reference refers to until the function returns. `Err(e)` when
    /// `e` is not plain text, or `want` is not that reference.
    pub(super) fn text_to_string_ref(&mut self, e: Expr, want: Ty) -> Result<Expr, Expr> {
        let want = self.shallow(want);
        let Some((Access::Const, target)) = self.types().as_ref(want) else { return Err(e) };
        if !self.is_string(target) {
            return Err(e);
        }
        let s = self.text_to_string(e)?;
        let span = s.span;
        let r = self.reference_to(s, span);
        Ok(self.coerce(r, want, "the string made of the text"))
    }

    /// A reference to a `string` where a slice of characters is wanted: the
    /// view of its characters, `&s.chars`, without a copy. `Err(r)` when `r`
    /// is not such a reference, or `want` could change what `r` may only
    /// read.
    pub(super) fn string_view(&mut self, r: Expr, want: Ty) -> Result<Expr, Expr> {
        let (want, have) = (self.shallow(want), self.shallow(r.ty));
        let Some((wa, Ty::Char)) = self.types().as_slice(want) else { return Err(r) };
        let Some((ra, target)) = self.types().as_ref(have) else { return Err(r) };
        if !self.is_string(target) || !ra.converts_to(wa) {
            return Err(r);
        }
        let span = r.span;
        let chars = self.types_mut().growable(Ty::Char);
        let place = Expr {
            kind: ExprKind::Field { base: Box::new(Expr::deref(r, target, span)), field: 0 },
            ty: chars,
            span,
        };
        let view = self.reference_to(place, span);
        Ok(self.coerce(view, want, "the characters of the string"))
    }

    /// `e` as a `string` where `want` is one, or else as it is.
    pub(super) fn text_for(&mut self, e: Expr, want: Ty) -> Expr {
        if !self.is_string(want) {
            return e;
        }
        self.text_to_string(e).unwrap_or_else(|e| e)
    }
}
