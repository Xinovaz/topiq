//! Places: what can be assigned, what a reference can be taken of, and how
//! fields and elements are reached.
//!
//! # Following a reference
//!
//! `*r` is the place the reference `r` refers to: with `p: *i32`, `*p = 5`
//! stores through it and `*p + 1` reads through it. `&` and `*` undo each
//! other: `*&x` is `x` again, and `&*p` is a reference to the same place as
//! `p`.
//!
//! Field access, indexing, `.len()` and `match` also look through one level
//! of reference by themselves: with `p: *Point`, `p.x` reads the `x` of the
//! `Point` that `p` refers to, and with `xs: *[i64; 4]`, `xs[2]` reads an
//! element. The typed IR writes that step out as a dereference too, so nothing
//! later has to rediscover it. Only one level is looked through; a reference
//! to a reference is followed with `*` first, as in `(*pp).x`.
//!
//! # Who may change a place
//!
//! A place may be assigned when every step to it allows that: the binding it
//! starts from is not a constant, and it is not reached through a `*const`
//! reference or slice. Every binding may be assigned unless its type is
//! `const`, and every `*T` may write what it refers to. `&` of a place that
//! may not change gives a `*const T`. The diagnostic names the
//! step that forbids the change, and what to write instead.
//!
//! # References to values that are not places
//!
//! `&f()` refers to a value that has no storage of its own. It is stored in a
//! temporary that lasts until the function returns, and the reference points
//! there. Taking `&` of a constant gives a `*const` reference, since nothing
//! may change a constant.

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Access, Expr, ExprKind, Ty};

use super::body::Checker;
use super::expr::{Mapped, map_op};
use super::report;

/// The operator method that `op=` calls.
fn assign_method(op: crate::tir::BinOp) -> Option<&'static str> {
    use crate::tir::BinOp as B;
    Some(match op {
        B::Add => "add_assign",
        B::Sub => "sub_assign",
        B::Mul => "mul_assign",
        B::Div => "div_assign",
        B::Rem => "rem_assign",
        B::BitAnd => "and_assign",
        B::BitOr => "or_assign",
        B::BitXor => "xor_assign",
        B::Shl => "shl_assign",
        B::Shr => "shr_assign",
        _ => return None,
    })
}

/// Why a place may or may not change.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Via {
    /// It starts from a binding or object with this name, declared here.
    Binding {
        /// The name.
        name: Symbol,
        /// Where it was declared.
        decl: Span,
    },
    /// It is reached through the reference or slice written here.
    Reference(Span),
    /// It is a temporary value, not storage the program named.
    Temporary,
}

/// What a place permits, and why.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaceAccess {
    /// What may be done to it.
    pub access: Access,
    /// What decided that.
    pub via: Via,
}

impl Checker<'_, '_> {
    /// Looks through one level of reference: a reference becomes the place it
    /// refers to. Anything else is unchanged.
    pub(super) fn auto_deref(&mut self, e: Expr) -> Expr {
        let t = self.shallow(e.ty);
        match self.types().as_ref(t) {
            Some((_, target)) => {
                let span = e.span;
                Expr::deref(e, target, span)
            }
            None => e,
        }
    }

    /// What a place permits.
    pub(super) fn place_access(&self, e: &Expr) -> PlaceAccess {
        let temp = PlaceAccess {
            access: Access::Write,
            via: Via::Temporary,
        };
        match &e.kind {
            ExprKind::Local(l) => {
                let x = &self.locals[l.index()];
                PlaceAccess {
                    access: binding_access(x.constant),
                    via: Via::Binding {
                        name: x.name,
                        decl: x.span,
                    },
                }
            }
            ExprKind::Global(g) => {
                let x = self.cx.unit.global(*g);
                PlaceAccess {
                    access: binding_access(x.constant),
                    via: Via::Binding {
                        name: x.name,
                        decl: x.span,
                    },
                }
            }
            ExprKind::Field { base, .. } if base.is_place() => self.place_access(base),
            ExprKind::Index { base, .. } => match self.types().as_slice(base.ty) {
                Some((access, _)) => PlaceAccess {
                    access,
                    via: Via::Reference(base.span),
                },
                None if base.is_place() => self.place_access(base),
                None => temp,
            },
            ExprKind::Deref(r) => PlaceAccess {
                access: self.types().as_ref(r.ty).map_or(Access::Write, |(a, _)| a),
                via: Via::Reference(r.span),
            },
            _ => temp,
        }
    }

    /// `receiver.name`.
    pub(super) fn field(&mut self, receiver: &Spanned<ast::Expr>, name: Spanned<Symbol>, span: Span) -> Expr {
        let base = self.expr(receiver);
        let base = self.auto_deref(base);
        let bt = self.shallow(base.ty);
        let text = self.name(name.node);
        match bt {
            Ty::Never => Checker::error(span),
            // `t.0`: an element of a tuple, by its position
            Ty::Tuple(_) => {
                let elems = self.types().as_tuple(bt).map(<[Ty]>::to_vec).unwrap_or_default();
                match text.parse::<usize>().ok().filter(|&i| i < elems.len()) {
                    Some(i) => Expr {
                        kind: ExprKind::Field {
                            base: Box::new(base),
                            field: i as u32,
                        },
                        ty: elems[i],
                        span,
                    },
                    None => {
                        let what = self.describe(bt);
                        let d = Diagnostic::new(Code::Es04)
                            .with_message(format!("{what} has no element `.{text}`"))
                            .at(name.span)
                            .with_note(match elems.len() {
                                0 => "the empty tuple has no elements".to_owned(),
                                1 => "its one element is `.0`".to_owned(),
                                n => format!("its elements are `.0` to `.{}`, by position", n - 1),
                            });
                        self.report(d);
                        Checker::error(span)
                    }
                }
            }
            Ty::Adt(id) => {
                let def = self.adt(id).clone();
                let type_name = self.types().adt_name(id, self.interner);
                if !def.is_struct() {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "`{type_name}` is an enumeration, so its fields cannot be read with `.`"
                            ))
                            .at(name.span)
                            .with_note(
                                "which fields a value has depends on its variant, so they are \
                                 reached by matching on it",
                            )
                            .with_help(format!("use `match`, as in `match x {{ {type_name}::V(a) => … }}`")),
                    );
                    return Checker::error(span);
                }
                match def.field_index(name.node) {
                    Some(i) => Expr {
                        kind: ExprKind::Field {
                            base: Box::new(base),
                            field: i,
                        },
                        ty: def.fields()[i as usize].ty,
                        span,
                    },
                    None => {
                        let interner = self.interner;
                        let mut d = Diagnostic::new(Code::Es04)
                            .with_message(format!("`{type_name}` has no field named `{text}`"))
                            .at(name.span);
                        let names = def.fields().iter().map(|f| interner.resolve(f.name));
                        if let Some(s) = report::closest(text, names) {
                            d = d.with_help(format!("did you mean `{s}`?"));
                        }
                        self.report(d);
                        Checker::error(span)
                    }
                }
            }
            Ty::Ref(_) => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("this is a reference to a reference, and `.` looks through only one")
                        .at(receiver.span)
                        .with_help(format!(
                            "follow the outer reference with `*` first: `(*{}).{text}`",
                            self.place_text(&base)
                        )),
                );
                Checker::error(span)
            }
            other => {
                let what = self.describe(other);
                let mut d = Diagnostic::new(Code::Es06)
                    .with_message(format!("{what} has no fields, so `.{text}` names nothing"))
                    .at(name.span);
                if text == "len" && matches!(other, Ty::Array(_) | Ty::Slice(_) | Ty::Growable(_)) {
                    d = d.with_help("the length is a method call: write `.len()`");
                }
                self.report(d);
                Checker::error(span)
            }
        }
    }

    /// `receiver[index]`.
    pub(super) fn index(&mut self, receiver: &Spanned<ast::Expr>, index: &Spanned<ast::Expr>, span: Span) -> Expr {
        // only reading `a[i]` only reads `a`, as in `a[i][j]`; the index
        // itself is an ordinary expression
        let reading = std::mem::take(&mut self.read_index);
        if let Some(e) = self.bare_qmap_index(receiver, span) {
            return e;
        }
        self.read_index = reading && matches!(receiver.node, ast::Expr::Index { .. });
        let base = self.expr(receiver);
        self.read_index = reading;
        let e = self.index_checked(base, receiver.span, index, span);
        self.read_index = false;
        e
    }

    /// `base[index]`, the receiver already checked.
    pub(super) fn index_checked(&mut self, base: Expr, at: Span, index: &Spanned<ast::Expr>, span: Span) -> Expr {
        // a type with `$index`, or a map locale, which is read another way
        let reading = std::mem::take(&mut self.read_index);
        let base = self.auto_deref(base);
        if self.is_structured(base.ty) {
            return self.structured_index(base, index, reading, span);
        }
        if let Ty::Qmap(_) = self.shallow(base.ty) {
            self.expr(index);
            self.report(
                Diagnostic::new(Code::Eq09)
                    .with_message("`t[…]` of a map locale does not say how the map locale is read")
                    .at(span)
                    .with_note(
                        "a map locale is read coherently, applying each entry under the control of its key \
                         without measuring, or classically, measuring the key and taking the entry it held",
                    )
                    .with_help("write `query t[k]` or `measure t[k]`"),
            );
            return Checker::error(span);
        }

        // else an array or slice, by a `usize`
        let i = self.expr(index);
        let i = self.coerce(i, Ty::USIZE, "an index");
        let bt = self.shallow(base.ty);
        let elem = match bt {
            Ty::Never => return Checker::error(span),
            Ty::Array(_) => self.types().as_array(bt).map(|a| a.0),
            Ty::Slice(_) => self.types().as_slice(bt).map(|s| s.1),
            Ty::Growable(_) => self.types().as_growable(bt),
            _ => None,
        };
        let Some(elem) = elem else {
            let what = self.describe(bt);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("{what} cannot be indexed"))
                    .at(at)
                    .with_note("indexing applies to arrays, `[T; N]` and `[T]`, and to slices `*[T]`"),
            );
            return Checker::error(span);
        };
        Expr {
            kind: ExprKind::Index {
                base: Box::new(base),
                index: Box::new(i),
            },
            ty: elem,
            span,
        }
    }

    /// `base[index]` on a structure or enumeration: `*base.$index(index)`,
    /// the place the reference `$index` returns refers to, or
    /// `*base.$index_rd(index)`, which may only be read.
    ///
    /// `$index_rd` is called where the element is only read (`reading`, as
    /// the value of a new `const` binding) and wherever `$index` cannot
    /// serve: `base` is a constant, or reached through a `*const`, or the
    /// type has no `$index`.
    fn structured_index(&mut self, base: Expr, index: &Spanned<ast::Expr>, reading: bool, span: Span) -> Expr {
        let i = self.expr(index);
        let t = self.shallow(base.ty);
        let has = |c: &mut Self, name: &str| c.operator_method(t, name).is_some();
        let constant = base.is_place() && self.place_access(&base).access == Access::Const;
        if constant && has(self, "index") && !has(self, "index_rd") {
            let ty = self.show(t);
            let target = self.place_text(&base);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{target}` is a constant, and `{ty}` has no `$index_rd` to read it by"))
                    .at(span)
                    .with_note("`$index` may change what it indexes, so it is not called on a constant")
                    .with_help(format!("declare `fn $index_rd(self: *const {ty}, …) -> *const …` in `impl {ty} {{ … }}`")),
            );
            return Checker::error(span);
        }
        let name = if has(self, "index_rd") && (reading || constant || !has(self, "index")) {
            "index_rd"
        } else {
            "index"
        };
        let r = self.operator_call(name, "[…]", base, Some(i), span);
        let t = self.shallow(r.ty);
        if t == Ty::Never {
            return r;
        }
        match self.types().as_ref(t) {
            Some((Access::Write, _)) if name == "index_rd" => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("`$index_rd` returns a reference that may write, but it serves reading")
                        .at(span)
                        .with_note("`$index_rd` is called on constants, so what it returns must not change them")
                        .with_help("return `*const T` from `$index_rd`"),
                );
                Checker::error(span)
            }
            Some((_, target)) => Expr::deref(r, target, span),
            None => {
                let what = self.describe(t);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`${name}` returns {what}, but indexing needs a reference to a place"))
                        .at(span)
                        .with_note("`a[i]` is the place `$index` returns a reference to, so it can be read or assigned")
                        .with_help("return `*T` from `$index`, or `*const T` from `$index_rd`"),
                );
                Checker::error(span)
            }
        }
    }

    /// `*operand`: the place a reference refers to.
    ///
    /// What it permits is what the reference permits, so `*p = v` needs `p`
    /// to be a `*T`, not a `*const T`. A reference to a fixed array yields the whole
    /// array. A slice does not: it is a view of several elements with no one
    /// value to name, and is indexed instead.
    pub(super) fn dereference(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Expr {
        let r = self.expr(operand);
        let t = self.shallow(r.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        if let Some((_, Ty::Void)) = self.types().as_ref(t) {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("`*` cannot follow a `*any`: what it refers to has a type known only when the program runs")
                    .at(operand.span)
                    .with_help("downcast it first, as in `match p as *T { Some(t) => …, None => … }`"),
            );
            return Checker::error(span);
        }
        if let Some((_, target)) = self.types().as_ref(t) {
            return Expr::deref(r, target, span);
        }
        let what = self.describe(t);
        let d = if self.types().as_slice(t).is_some() {
            Diagnostic::new(Code::Es06)
                .with_message(format!("`*` cannot follow {what}: a slice refers to several elements, not one value"))
                .at(operand.span)
                .with_help("index it, as in `s[i]`, or ask its length with `.len()`")
        } else {
            Diagnostic::new(Code::Es06)
                .with_message(format!("`*` follows a reference, but this is {what}"))
                .at(operand.span)
                .with_note("`*r` is the place the reference `r` refers to, so there must be a reference to follow")
        };
        self.report(d);
        Checker::error(span)
    }

    /// `&operand`.
    pub(super) fn reference(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Expr {
        let x = self.expr(operand);
        self.reference_to(x, span)
    }

    /// A reference to an already checked expression: to the place it names,
    /// or to a temporary holding its value. It is a `*const T` when the place
    /// is a constant or reached through one, and a `*T` otherwise.
    pub(super) fn reference_to(&mut self, x: Expr, span: Span) -> Expr {
        if x.ty == Ty::Never {
            return Checker::error(span);
        }
        let access = if x.is_place() { self.place_access(&x).access } else { Access::Write };
        let ty = self.types_mut().reference(access, x.ty);
        Expr {
            kind: ExprKind::Ref(Box::new(x)),
            ty,
            span,
        }
    }

    /// `place = value` and `place op= value`.
    pub(super) fn assign(
        &mut self,
        op: Option<ast::BinOp>,
        place: &Spanned<ast::Expr>,
        value: &Spanned<ast::Expr>,
        span: Span,
    ) -> Expr {
        let p = self.expr(place);
        if p.ty != Ty::Never && !p.is_place() {
            self.report(
                Diagnostic::new(Code::Es08)
                    .with_message("this cannot be assigned: it is a value, not a place that holds one")
                    .at(place.span)
                    .with_note(
                        "a binding, a field or element of one, or what a `*T` reference refers \
                         to can be assigned",
                    ),
            );
            let _ = self.expr(value);
            return Checker::error(span);
        }
        let mut ok = p.ty != Ty::Never;
        if ok {
            let pa = self.place_access(&p);
            if pa.access != Access::Write {
                let d = self.not_mutable(&p, pa, "assignment to", place.span);
                self.report(d);
                ok = false;
            }
        }
        let value = match op {
            None => {
                let v = self.expr_for(value, p.ty);
                let target = self.place_text(&p);
                self.coerce(v, p.ty, &format!("the value assigned to `{target}`"))
            }
            Some(aop) => {
                // the parser only forms compound assignment from operators
                // that have one, all of which evaluate both operands
                let Mapped::Binary(bop) = map_op(aop) else {
                    unreachable!("`{aop:?}=` is not an assignment operator");
                };
                // qubits change in place by reversible operations
                if self.cx.quantum && self.types().is_quantum(self.shallow(p.ty)) {
                    let r = self.expr(value);
                    if !ok {
                        return Checker::error(span);
                    }
                    return self.quantum_assign(bop, p, r, span);
                }
                // a structure's `a op= b` is its `$op_assign`, when it has
                // one, which changes `a` in place
                if self.is_structured(p.ty)
                    && let Some(name) = assign_method(bop)
                    && self.operator_method(p.ty, name).is_some()
                {
                    let r = self.expr(value);
                    let text = format!("{}=", bop.text());
                    return self.operator_call(name, &text, p, Some(r), span);
                }
                let r = self.expr(value);
                // a place that does something when evaluated, like
                // `a[i++]`, is evaluated once
                if ok && !super::step::evaluates_purely(&p) {
                    return self.assign_once(bop, p, r, span);
                }
                let read = p.clone();
                self.binary_checked(bop, read, r, span)
            }
        };
        if !ok {
            return Checker::error(span);
        }
        Expr {
            kind: ExprKind::Assign {
                place: Box::new(p),
                value: Box::new(value),
            },
            ty: Ty::Void,
            span,
        }
    }

    /// A place written the way a program would write it, for a message.
    pub(super) fn place_text(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Local(l) => self.name(self.locals[l.index()].name).to_owned(),
            ExprKind::Global(g) => self.name(self.cx.unit.global(*g).name).to_owned(),
            ExprKind::Field { base, field } => {
                let f = match base.ty {
                    Ty::Adt(id) => self
                        .types()
                        .adt(id)
                        .fields()
                        .get(*field as usize)
                        .map_or("?", |f| self.name(f.name)),
                    _ => "?",
                };
                format!("{}.{f}", self.place_text(base))
            }
            ExprKind::Index { base, .. } => format!("{}[…]", self.place_text(base)),
            ExprKind::Deref(r) => self.place_text(r),
            // what `$index` or `$index_rd` gave, `base[…]` as written
            ExprKind::Call { callee: crate::tir::Callee::Fn(f), args }
                if matches!(self.name(self.cx.unit.func(*f).name), "index" | "index_rd") && !args.is_empty() =>
            {
                format!("{}[…]", self.place_text(&args[0]))
            }
            ExprKind::Ref(x) | ExprKind::Coerce(x) => self.place_text(x),
            _ => "…".to_owned(),
        }
    }

    /// Why a place cannot be changed, and what to write instead. `doing` is
    /// what would change it, such as "assignment to".
    pub(super) fn not_mutable(&self, e: &Expr, pa: PlaceAccess, doing: &str, span: Span) -> Diagnostic {
        let target = self.place_text(e);
        match pa.via {
            Via::Binding { name, decl } => {
                let n = self.name(name);
                Diagnostic::new(Code::Ec03)
                    .with_message(format!("{doing} `{target}` would change `{n}`, which is a constant"))
                    .at(span)
                    .also(decl, "declared `const` here")
                    .with_note("a `const` binding keeps the value it is declared with")
                    .with_help(format!("if `{n}` needs to change, drop `const` from its type"))
            }
            Via::Reference(r) => Diagnostic::new(Code::Ec03)
                .with_message(format!(
                    "{doing} `{target}` would change an object reached through a `*const` reference"
                ))
                .at(span)
                .also(r, "this reference is `*const`")
                .with_note("`*const T` refers to a constant, and may only read it")
                .with_help("take a `*T` instead if what it refers to needs to change"),
            Via::Temporary => Diagnostic::new(Code::Es08)
                .with_message(format!("{doing} a temporary value has no effect anyone could see"))
                .at(span),
        }
    }
}

/// What a binding permits: every binding may be assigned, unless its type
/// is `const`.
fn binding_access(constant: bool) -> Access {
    if constant { Access::Const } else { Access::Write }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::{ExprKind, Stmt};

    #[test]
    fn every_binding_but_a_constant_may_be_assigned() {
        check("fn f() { let x = 1; x = 2; }");
        check("fn f(x: i32) { x = 2; }");
        check("fn f() { for i in 0..3 { i = 5; } }");
        assert_eq!(codes("fn f() { let x: const i32 = 1; x = 2; }"), [Code::Ec03]);
    }

    #[test]
    fn an_assignment_has_no_value_so_it_does_not_chain() {
        // `b = 2` is void, and void is not an i32
        assert_eq!(codes("fn f() { let a = 1; let b = 1; a = b = 2; }"), [Code::Es06]);
    }

    #[test]
    fn a_constant_cannot_be_assigned() {
        assert_eq!(codes("let X: const i32 = 1;\nfn f() { X = 2; }"), [Code::Ec03]);
    }

    #[test]
    fn a_field_of_a_binding_may_be_assigned() {
        check("struct P { x: i32 }\nfn f() { let p = P { x: 1 }; p.x = 2; }");
    }

    #[test]
    fn a_referent_is_assigned_through_a_reference() {
        check("struct P { x: i32 }\nfn f(p: *P) { p.x = 2; }");
        assert_eq!(codes("struct P { x: i32 }\nfn f(p: *const P) { p.x = 2; }"), [Code::Ec03]);
        check("fn f(s: *[u8]) { s[0] = 1; }");
        assert_eq!(codes("fn f(s: *const [u8]) { s[0] = 1; }"), [Code::Ec03]);
    }

    #[test]
    fn a_field_is_read_through_a_reference() {
        let u = check("struct P { x: i32 }\nfn f(p: *P) -> i32 { p.x }");
        let v = u.fns[0].body.value.as_ref().unwrap();
        let ExprKind::Field { base, .. } = &v.kind else {
            panic!("{:?}", v.kind)
        };
        assert!(matches!(base.kind, ExprKind::Deref(_)), "the dereference is written out");
    }

    #[test]
    fn an_unknown_field_suggests_a_close_one() {
        let (_, d) = crate::sema::testing::analyzed("struct P { total: i32 }\nfn f(p: P) -> i32 { p.totl }");
        assert_eq!(d[0].code, Code::Es04);
        assert!(d[0].helps.iter().any(|h| h.contains("`total`")), "{:?}", d[0].helps);
    }

    #[test]
    fn an_enumeration_has_no_fields_to_read() {
        assert_eq!(codes("enum E { A { x: i32 } }\nfn f(e: E) -> i32 { e.x }"), [Code::Es06]);
    }

    #[test]
    fn indexing_needs_an_array_or_slice_and_a_usize() {
        check("fn f(a: [i32; 3], i: usize) -> i32 { a[i] + a[0] }");
        check("fn f(s: *[i32]) -> i32 { s[1] }");
        check("fn f(a: *[i32; 3]) -> i32 { a[2] }");
        assert_eq!(codes("fn f(a: [i32; 3], i: i32) -> i32 { a[i] }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: i32) -> i32 { a[0] }"), [Code::Es06]);
    }

    #[test]
    fn a_reference_records_what_it_may_do() {
        check("fn f() { let x = 1; let r: *i64 = &x; *r = 2; }");
        check("fn f() { let x = 1; let r: *const i64 = &x; }");
        assert_eq!(codes("let K: const i32 = 1;\nfn f() -> *i32 { &K }"), [Code::Es06]);
        check("let K: const i32 = 1;\nfn f() -> *const i32 { &K }");
    }

    #[test]
    fn a_reference_to_a_value_needs_no_place() {
        check("fn g() -> i32 { 1 }\nfn f() -> *i32 { &g() }");
    }

    #[test]
    fn a_value_is_not_a_place() {
        assert_eq!(codes("fn g() -> i32 { 1 }\nfn f() { g() = 2; }"), [Code::Es08]);
    }

    #[test]
    fn a_dereference_is_the_place_a_reference_refers_to() {
        check("fn f(p: *i32) -> i32 { *p + 1 }");
        check("fn f(p: *i32) { *p = 5; *p += 1; }");
        check("fn f(p: *[i32; 3]) -> i32 { let a = *p; a[0] }");
        check("fn f(x: i32) -> i32 { *&x }");
        check("fn f(p: **i32) -> i32 { **p }");
        check("fn f(p: **i32) { (**)p = 1; }");
    }

    #[test]
    fn storing_through_a_dereference_needs_a_reference_that_may_write() {
        check("fn f(p: *i32) { *p = 5; }");
        assert_eq!(codes("fn f(p: *const i32) { *p = 5; }"), [Code::Ec03]);
        assert_eq!(codes("fn f(p: **const i32) { **p = 5; }"), [Code::Ec03], "the inner reference only reads");
        check("fn f(p: *i32) -> *i32 { &*p }");
        assert_eq!(codes("fn f(p: *const i32) -> *i32 { &*p }"), [Code::Es06]);
    }

    #[test]
    fn only_a_reference_can_be_followed() {
        assert_eq!(codes("fn f(x: i32) -> i32 { *x }"), [Code::Es06]);
        let (_, d) = crate::sema::testing::analyzed("fn f(s: *[i32]) -> i32 { let a = *s; 0 }");
        assert_eq!(d[0].code, Code::Es06);
        assert!(d[0].message.contains("slice"), "{}", d[0].message);
    }

    #[test]
    fn a_value_cannot_be_moved_out_through_a_dereference() {
        let src = "struct Q { v: i32 }\nfn f(p: *Q) -> Q { *p }";
        assert_eq!(codes(src), [Code::Es12]);
        check("struct Q { v: i32 }\nfn f(p: *Q) -> i32 { (*p).v }");
    }

    #[test]
    fn runs_of_references_undo_each_other() {
        // `(**)(&&)x` takes a reference to a reference to `x` and follows both
        check("fn f(x: i32) -> bool { (**)(&&)x == x }");
        check("fn f(x: i32) { (**)(&&)x = 3; }");
        let u = check("fn f(x: i32) -> **i32 { &&x }");
        assert_eq!(u.types.display(u.fns[0].ret, &crate::intern::Interner::new()).matches('*').count(), 2);
    }

    #[test]
    fn references_compare_by_what_they_refer_to() {
        check("fn f(p: *i32) -> bool { &*p == p }");
        check("fn f(p: *i32, q: *i32) -> bool { p != q }");
        check("fn f(s: *[i32], t: *[i32]) -> bool { s == t }");
        assert_eq!(codes("fn f(p: *i32, q: *u8) -> bool { p == q }"), [Code::Es06]);
        assert_eq!(codes("fn f(p: *i32) -> bool { p == 3 }"), [Code::Es06]);
        assert_eq!(codes("fn f(p: *i32, q: *i32) -> bool { p < q }"), [Code::Es06], "references have no order");
    }

    #[test]
    fn compound_assignment_reads_the_same_place() {
        let u = check("fn f(a: [i32; 2]) { a[1] += 5; }");
        let Stmt::Expr(e) = &u.fns[0].body.stmts[0] else {
            panic!()
        };
        assert!(matches!(e.kind, ExprKind::Assign { .. }));
    }
}
