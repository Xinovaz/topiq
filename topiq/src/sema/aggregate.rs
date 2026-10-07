//! Structure, variant and array values.
//!
//! A structure literal names every field, each exactly once, in any order;
//! the fields are evaluated in the order written, whatever order the
//! structure declares them in. There are no default values: a field left out
//! is an error, not a zero. A variant with fields is built the same way
//! (`Shape::Rect { w: 2, h: 3 }`) or, for a tuple variant, like a call:
//! `Shape::Circle(1)`.
//!
//! An array literal lists its elements, all of one type; `[e; N]` evaluates `e`
//! once and copies it into all `N` elements, so `e`'s type must be one that
//! can be copied.

use crate::ast::{self, FieldInit, Path};
use crate::intern::Symbol;
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{AdtId, Expr, ExprKind, FieldDef, Ty, VariantShape};

use super::items::GenericKey;

use super::body::Checker;
use super::path::Resolution;
use super::report;

/// What a variant is built from: values by position, or fields by name.
#[derive(Clone, Copy)]
pub(super) enum VariantArgs<'e> {
    /// `V(a, b)`.
    Positional(&'e [Spanned<ast::Expr>]),
    /// `V { f: a }`.
    Named(&'e [FieldInit]),
}

impl Checker<'_, '_> {
    /// `want`, when it is an instance of the generic structure `path` names.
    fn instance_named(&mut self, path: &Path, want: Ty) -> Option<AdtId> {
        let Ty::Adt(id) = want else { return None };
        let (key, used) = self.generic_path(&path.segments)?;
        if used != path.segments.len() {
            return None;
        }
        let origin = self.cx.generic_origin(key);
        let def = self.adt(id);
        let same = def.is_struct() && !def.args.is_empty() && def.name == key.name && def.origin == origin;
        same.then_some(id)
    }

    /// `Path { field: value, … }`.
    pub(super) fn struct_lit(&mut self, path: &Path, fields: &[FieldInit], span: Span) -> Expr {
        // a structure literal has nowhere to write generic arguments, so a
        // generic structure's are deduced from the values given for its
        // fields
        if let Some((key, used)) = self.generic_path(&path.segments) {
            match path.segments[used..] {
                [] => return self.generic_struct_lit(key, fields, span),
                [variant] => return self.generic_variant(key, variant, VariantArgs::Named(fields), span),
                _ => {}
            }
        }
        let Some(r) = self.resolve_path(path, &[], span) else {
            for f in fields {
                self.expr(&f.value);
            }
            return Checker::error(span);
        };
        let target = match r {
            Resolution::Adt(id) if self.adt(id).is_struct() => Some((id, None)),
            Resolution::Variant(id, v) => {
                let var = self.adt(id).variants()[v as usize].clone();
                let name = format!("{}::{}", self.types().adt_name(id, self.interner), self.name(var.name));
                match var.shape {
                    VariantShape::Struct => Some((id, Some(v))),
                    VariantShape::Tuple => {
                        let blanks = vec!["…"; var.fields.len()].join(", ");
                        self.report(
                            Diagnostic::new(Code::Es07)
                                .with_message(format!("`{name}` holds its values by position, not by name"))
                                .at(span)
                                .with_help(format!("write `{name}({blanks})`")),
                        );
                        None
                    }
                    VariantShape::Unit => {
                        self.report(
                            Diagnostic::new(Code::Es07)
                                .with_message(format!("`{name}` has no fields"))
                                .at(span)
                                .with_help(format!("write just `{name}`")),
                        );
                        None
                    }
                }
            }
            Resolution::Adt(id) => {
                let name = self.types().adt_name(id, self.interner);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{name}` is an enumeration, so a value of it is one of its variants"))
                        .at(span)
                        .with_help(format!("name the variant, as in `{name}::V {{ … }}`")),
                );
                None
            }
            Resolution::Value(_) => {
                let written = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{written}` is not a structure, so it cannot be built with `{{ … }}`"))
                        .at(span),
                );
                None
            }
        };
        let Some((id, variant)) = target else {
            for f in fields {
                self.expr(&f.value);
            }
            return Checker::error(span);
        };
        self.fields_of(id, variant, fields, None, span)
    }

    /// Checks the field values of a structure literal or a structure variant
    /// against what the type declares. `pre` holds the values already
    /// checked, one per field written, when deducing generic arguments needed
    /// them first, so no value is checked twice.
    fn fields_of(
        &mut self,
        id: AdtId,
        variant: Option<u32>,
        fields: &[FieldInit],
        pre: Option<Vec<Expr>>,
        span: Span,
    ) -> Expr {
        // a value already checked, or else one checked now
        let mut pre = pre.map(Vec::into_iter);
        let mut value = |c: &mut Self, f: &FieldInit, want: Option<Ty>| match pre.as_mut().and_then(Iterator::next) {
            Some(e) => e,
            None => match want {
                Some(t) => c.expr_for(&f.value, t),
                None => c.expr(&f.value),
            },
        };

        // the fields declared, and what the message calls their owner
        let def = self.adt(id).clone();
        let (decl, what): (&[FieldDef], String) = match variant {
            None => (def.fields(), self.types().adt_name(id, self.interner)),
            Some(v) => {
                let var = &def.variants()[v as usize];
                (
                    &var.fields,
                    format!("{}::{}", self.types().adt_name(id, self.interner), self.name(var.name)),
                )
            }
        };

        // each field written, once, and one the type has
        let mut given: Vec<(u32, Expr)> = Vec::with_capacity(fields.len());
        let mut seen: Vec<(u32, Span)> = Vec::new();
        for f in fields {
            let text = self.name(f.name.node);
            let Some(i) = decl.iter().position(|d| d.name == f.name.node).map(|i| i as u32) else {
                let interner = self.interner;
                let mut d = Diagnostic::new(Code::Es04)
                    .with_message(format!("`{what}` has no field named `{text}`"))
                    .at(f.name.span);
                if let Some(s) = report::closest(text, decl.iter().map(|d| interner.resolve(d.name))) {
                    d = d.with_help(format!("did you mean `{s}`?"));
                }
                self.report(d);
                value(self, f, None);
                continue;
            };
            if let Some(&(_, earlier)) = seen.iter().find(|(j, _)| *j == i) {
                let d = Diagnostic::new(Code::Es05)
                    .with_message(format!("the field `{text}` is given a value twice"))
                    .at_with(f.name.span, "given again here")
                    .also(earlier, "first given here")
                    .with_help("remove one of them");
                self.report(d);
                value(self, f, None);
                continue;
            }
            seen.push((i, f.name.span));
            let v = value(self, f, Some(decl[i as usize].ty));
            let v = self.coerce(v, decl[i as usize].ty, &format!("the field `{text}` of `{what}`"));
            given.push((i, v));
        }

        // every field given a value
        let missing: Vec<&str> = decl
            .iter()
            .enumerate()
            .filter(|(i, _)| !seen.iter().any(|(j, _)| *j as usize == *i))
            .map(|(_, d)| self.name(d.name))
            .collect();
        if !missing.is_empty() {
            let list = missing.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>();
            let list = match list.as_slice() {
                [one] => one.clone(),
                [init @ .., last] => format!("{} and {last}", init.join(", ")),
                [] => unreachable!(),
            };
            self.report(
                Diagnostic::new(Code::Es15)
                    .with_message(format!("this `{what}` gives no value for {list}"))
                    .at(span)
                    .with_note("every field must be given a value; there are no defaults")
                    .with_help(format!("add `{}: <value>`", missing[0])),
            );
            return Checker::error(span);
        }

        let kind = match variant {
            None => ExprKind::StructLit { fields: given },
            Some(v) => ExprKind::Variant {
                variant: v,
                fields: given,
            },
        };
        Expr {
            kind,
            ty: Ty::Adt(id),
            span,
        }
    }

    /// `Enum::Variant(args)`.
    pub(super) fn variant_call(&mut self, id: AdtId, v: u32, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let var = self.adt(id).variants()[v as usize].clone();
        let name = format!("{}::{}", self.types().adt_name(id, self.interner), self.name(var.name));
        let checked: Vec<Expr> = args
            .iter()
            .enumerate()
            .map(|(i, a)| match var.fields.get(i) {
                Some(f) => self.expr_for(a, f.ty),
                None => self.expr(a),
            })
            .collect();
        match var.shape {
            VariantShape::Tuple => {}
            VariantShape::Unit => {
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("`{name}` has no fields, so it takes no values"))
                        .at(span)
                        .with_help(format!("write just `{name}`")),
                );
                return Checker::error(span);
            }
            VariantShape::Struct => {
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("`{name}` holds its values by name, not by position"))
                        .at(span)
                        .with_help(format!("write `{name} {{ field: value, … }}`")),
                );
                return Checker::error(span);
            }
        }
        self.positional_variant(id, v, checked, &name, Some(var.span), span)
    }

    /// The variant `v` of `id`, named `name`, holding the values `checked`
    /// by position, each converted to its field's type.
    fn positional_variant(&mut self, id: AdtId, v: u32, checked: Vec<Expr>, name: &str, decl: Option<Span>, span: Span) -> Expr {
        let tys: Vec<Ty> = self.adt(id).variants()[v as usize].fields.iter().map(|f| f.ty).collect();
        if checked.len() != tys.len() {
            let mut d = wrong_count(name, tys.len(), checked.len(), span);
            if let Some(decl) = decl {
                d = d.also(decl, "the variant is declared here");
            }
            self.report(d);
            return Checker::error(span);
        }
        let fields = checked
            .into_iter()
            .zip(tys)
            .enumerate()
            .map(|(i, (e, t))| (i as u32, self.coerce(e, t, &format!("value {} of `{name}`", i + 1))))
            .collect();
        Expr {
            kind: ExprKind::Variant { variant: v, fields },
            ty: Ty::Adt(id),
            span,
        }
    }

    /// `Pair { first: 1, second: true }` for a generic structure: the values
    /// given for the fields say what its arguments are.
    ///
    /// The values are checked once, here, and the instance is made from what
    /// they turned out to be; the whole literal is then checked against that
    /// instance, so a field whose type does not match is reported against a
    /// real structure rather than against a parameter.
    fn generic_struct_lit(&mut self, key: GenericKey, fields: &[FieldInit], span: Span) -> Expr {
        let text = self.cx.generic_label(key);
        let Some(params) = self.cx.generic_params(key).map(<[ast::GenericParam]>::to_vec) else {
            return Checker::error(span);
        };
        let Some(decl) = self.cx.generic_struct_fields(key) else {
            let d = super::generic::needs_arguments(&text, "type", params.len(), span, true);
            self.report(d);
            return Checker::error(span);
        };
        // match each written field against the value given for it
        let names: Vec<Symbol> = params.iter().map(|p| p.name.node).collect();
        let mut subst = super::generic::Subst::new();
        let mut checked = Vec::with_capacity(fields.len());
        for f in fields {
            let e = self.expr(&f.value);
            if let Some(d) = decl.iter().find(|d| d.name.node == f.name.node) {
                let found = self.settled(e.ty);
                super::generic::match_written(self.cx, &d.ty.node, found, &names, &mut subst);
            }
            checked.push(e);
        }
        let mut args = Vec::with_capacity(params.len());
        for p in &params {
            match super::generic::lookup(&subst, p.name.node) {
                Some(a) => args.push(a),
                None => {
                    let d = super::generic::undeduced_field(&text, self.name(p.name.node), span);
                    self.report(d);
                    return Checker::error(span);
                }
            }
        }
        let Some(id) = self.cx.instantiate_adt(key, args, span) else {
            return Checker::error(span);
        };
        self.fields_of(id, None, fields, Some(checked), span)
    }

    /// A variant of a generic enumeration built with the enumeration named
    /// without its arguments: `Maybe::Just(5)` or `Tree::Node { … }`. The
    /// values the variant carries say what the arguments are, as for a
    /// generic structure.
    pub(super) fn generic_variant(
        &mut self,
        key: GenericKey,
        variant: Spanned<Symbol>,
        payload: VariantArgs<'_>,
        span: Span,
    ) -> Expr {
        let label = self.cx.generic_label(key);
        let text = format!("{label}::{}", self.name(variant.node));
        let Some(params) = self.cx.generic_params(key).map(<[ast::GenericParam]>::to_vec) else {
            return Checker::error(span);
        };
        let Some(decl) = self.cx.generic_enum_variant(key, variant.node) else {
            let e = label.clone();
            self.report(
                Diagnostic::new(Code::Es04)
                    .with_message(format!("`{e}` has no variant named `{}`", self.name(variant.node)))
                    .at(variant.span),
            );
            return Checker::error(span);
        };
        let names: Vec<Symbol> = params.iter().map(|p| p.name.node).collect();
        let mut subst = super::generic::Subst::new();
        let mut checked = Vec::new();
        match (payload, &decl.payload) {
            (VariantArgs::Positional(args), Some(ast::VariantPayload::Tuple(tys))) => {
                for (i, a) in args.iter().enumerate() {
                    let e = self.expr(a);
                    if let Some(t) = tys.get(i) {
                        // literals stay unsettled here: where the variant is
                        // used may still say what they are
                        let found = self.resolved(e.ty);
                        super::generic::match_written(self.cx, &t.node, found, &names, &mut subst);
                    }
                    checked.push(e);
                }
            }
            (VariantArgs::Named(fields), Some(ast::VariantPayload::Struct(decl_fields))) => {
                for f in fields {
                    let e = self.expr(&f.value);
                    if let Some(d) = decl_fields.iter().find(|d| d.name.node == f.name.node) {
                        let found = self.settled(e.ty);
                        super::generic::match_written(self.cx, &d.ty.node, found, &names, &mut subst);
                    }
                    checked.push(e);
                }
            }
            _ => {
                let form = match &decl.payload {
                    None => format!("`{text}` carries nothing, so nothing says what the arguments of `{label}` are"),
                    Some(ast::VariantPayload::Tuple(_)) => format!("`{text}` holds its values by position"),
                    Some(ast::VariantPayload::Struct(_)) => format!("`{text}` holds its values by name"),
                };
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(form)
                        .at(span)
                        .with_help(format!("write the arguments, as in `{text}::<…>`, and the values the way the variant holds them")),
                );
                return Checker::error(span);
            }
        }
        let mut args = Vec::with_capacity(params.len());
        for p in &params {
            match super::generic::lookup(&subst, p.name.node) {
                Some(a) => args.push(a),
                // one the values do not mention, such as the `E` of `Ok(5)`,
                // is whatever the variant's context says
                None if matches!(payload, VariantArgs::Positional(_)) && p.ty.is_none() => {
                    args.push(crate::tir::Arg::Type(self.infer.fresh_any()));
                }
                None => {
                    let d = super::generic::undeduced(&text, self.name(p.name.node), span);
                    self.report(d);
                    return Checker::error(span);
                }
            }
        }
        // a value carried by position that is still an unsettled literal
        // leaves the instance open until the variant's context settles it:
        // `Just(3)` passed where a `Maybe<u8>` is wanted is a `Maybe<u8>`
        let open = matches!(payload, VariantArgs::Positional(_))
            && args.iter().any(|a| matches!(*a, crate::tir::Arg::Type(t) if self.types().has_infer(t)));
        if open {
            let arity = match &decl.payload {
                Some(ast::VariantPayload::Tuple(tys)) => tys.len(),
                _ => 0,
            };
            if checked.len() != arity {
                self.report(wrong_count(&text, arity, checked.len(), span));
                return Checker::error(span);
            }
            let index = self
                .cx
                .generic_enum_decl(key)
                .and_then(|vs| vs.iter().position(|x| x.name.node == variant.node))
                .unwrap_or(0) as u32;
            let recorded = args
                .iter()
                .map(|a| match *a {
                    crate::tir::Arg::Type(t) => t,
                    crate::tir::Arg::Const(_) => Ty::Never,
                })
                .collect();
            let ty = self.infer.fresh_instance_with(key.name, self.cx.generic_origin(key), recorded);
            self.pending.push(super::body::Pending {
                ty,
                span,
                what: text,
                args: Some((key, args)),
            });
            let fields = checked.into_iter().enumerate().map(|(i, e)| (i as u32, e)).collect();
            return Expr {
                kind: ExprKind::Variant { variant: index, fields },
                ty,
                span,
            };
        }
        let Some(id) = self.cx.instantiate_adt(key, args, span) else {
            return Checker::error(span);
        };
        let Some(v) = self.adt(id).variant_index(variant.node) else {
            return Checker::error(span);
        };
        match payload {
            VariantArgs::Named(fields) => self.fields_of(id, Some(v), fields, Some(checked), span),
            VariantArgs::Positional(_) => self.positional_variant(id, v, checked, &text, None, span),
        }
    }

    /// `(a, b, …)`, and `()`, which is the one value of the empty tuple.
    ///
    /// A tuple is a structure whose fields are its positions, so it is carried
    /// as a structure literal whose "field" numbers are those positions, and
    /// everything that handles a structure (layout, construction, patterns)
    /// handles a tuple unchanged.
    pub(super) fn tuple_lit(&mut self, items: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let mut fields = Vec::with_capacity(items.len());
        let mut types = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let e = self.expr(item);
            if e.ty == Ty::Void {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("an element of a tuple must hold a value, and this produces none")
                        .at(e.span),
                );
            }
            types.push(e.ty);
            fields.push((i as u32, e));
        }
        let ty = self.types_mut().tuple(types);
        Expr {
            kind: ExprKind::StructLit { fields },
            ty,
            span,
        }
    }

    /// Checks `e`, which is to be delivered where a `want` is wanted. An
    /// array written out passes the element type it should have down to its
    /// elements, so that `[[1], [2, 3]]` written where a `[[i64]]` is wanted
    /// makes each inner array growable before their lengths could disagree.
    /// The caller still converts the result to `want`.
    pub(super) fn expr_for(&mut self, e: &Spanned<ast::Expr>, want: Ty) -> Expr {
        let w = self.shallow(want);
        let elem = self
            .types()
            .as_growable(w)
            .or_else(|| self.types().as_array(w).map(|(t, _)| t));
        match (&e.node, elem) {
            // a document is read as the type wanted here
            (ast::Expr::Macro { name, args }, _) if self.name(name.node) == "embed" => self.embed(args, Some(w), e.span),
            // a number where an exact one is wanted is one
            (_, _) if self.is_exact_number(w) && let Some(x) = self.exact_for(e, w) => x,
            // an array literal where a vector, covector or matrix is wanted is
            // one: its elements, or rows, checked as the vector's or matrix's
            (ast::Expr::Array(_) | ast::Expr::ArrayRepeat { .. }, _)
                if ["vec", "covec", "mat"].into_iter().any(|n| self.core_exact(w, n).is_some()) =>
            {
                let Ty::Adt(id) = self.shallow(w) else { unreachable!("a vector or matrix") };
                let field = self.adt(id).fields()[0].ty;
                let inner = self.expr_for(e, field);
                let inner = self.coerce(inner, field, "the elements");
                Expr {
                    kind: ExprKind::StructLit {
                        fields: vec![(0, inner)],
                    },
                    ty: w,
                    span: e.span,
                }
            }
            // a generic structure's literal wanted as one of its instances is
            // that instance, which says what its fields cannot
            (ast::Expr::StructLit { path, fields }, _) if self.instance_named(path, w).is_some() => {
                let id = self.instance_named(path, w).expect("just found");
                self.fields_of(id, None, fields, None, e.span)
            }
            // a tuple written out passes each element the type it should
            // have, so that `("name", v)` is a `(string, V)` where one is
            // wanted
            (ast::Expr::Tuple(items), _)
                if self.types().as_tuple(w).is_some_and(|ts| ts.len() == items.len()) =>
            {
                let wanted = self.types().as_tuple(w).expect("just found").to_vec();
                let fields: Vec<(u32, Expr)> = items
                    .iter()
                    .zip(&wanted)
                    .enumerate()
                    .map(|(i, (item, &t))| {
                        let x = self.expr_for(item, t);
                        (i as u32, self.coerce(x, t, &format!("element {} of the tuple", i + 1)))
                    })
                    .collect();
                Expr {
                    kind: ExprKind::StructLit { fields },
                    ty: w,
                    span: e.span,
                }
            }
            (ast::Expr::Array(items), Some(elem)) if !items.is_empty() => {
                let out: Vec<Expr> = items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        let x = self.expr_for(item, elem);
                        self.coerce(x, elem, &format!("element {} of the array", i + 1))
                    })
                    .collect();
                let ty = self.types_mut().array(elem, out.len() as u64);
                Expr {
                    kind: ExprKind::ArrayLit(out),
                    ty,
                    span: e.span,
                }
            }
            _ => self.expr(e),
        }
    }

    /// `[a, b, c]`.
    pub(super) fn array_lit(&mut self, items: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let mut out: Vec<Expr> = Vec::with_capacity(items.len());
        let mut elem = Ty::Never;
        for (i, item) in items.iter().enumerate() {
            let e = self.expr(item);
            match self.unify(elem, e.ty) {
                Ok(t) => elem = t,
                Err(_) => {
                    let (a, b) = (self.describe(elem), self.describe(e.ty));
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "the elements of an array all have one type, but element 1 is {a} \
                                 and element {} is {b}",
                                i + 1
                            ))
                            .at_with(e.span, format!("this is {b}")),
                    );
                }
            }
            out.push(e);
        }
        let ty = self.types_mut().array(elem, out.len() as u64);
        Expr {
            kind: ExprKind::ArrayLit(out),
            ty,
            span,
        }
    }

    /// `[e; N]`.
    pub(super) fn array_repeat(&mut self, elem: &Spanned<ast::Expr>, len: &Spanned<ast::Expr>, span: Span) -> Expr {
        let e = self.expr(elem);
        let Some(n) = self.cx.const_usize(len, "the length of an array") else {
            return Checker::error(span);
        };
        if n > 1 && !self.types().is_copyable(self.shallow(e.ty)) {
            let what = self.describe(e.ty);
            self.report(
                Diagnostic::new(Code::Es12)
                    .with_message(format!(
                        "`[value; {n}]` copies its value into every element, but {what} cannot be copied"
                    ))
                    .at(elem.span)
                    .with_note(
                        "a structure or enumeration is moved rather than copied, so one value can \
                         fill only one element",
                    )
                    .with_help("write the elements out one by one, as `[a, b, c]`"),
            );
            return Checker::error(span);
        }
        let ty = self.types_mut().array(e.ty, n);
        Expr {
            kind: ExprKind::ArrayRepeat {
                elem: Box::new(e),
                len: n,
            },
            ty,
            span,
        }
    }
}

/// A variant `name` holding `want` values given `got`.
fn wrong_count(name: &str, want: usize, got: usize, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es07)
        .with_message(format!(
            "`{name}` holds {want} value{}, but {got} {} given",
            if want == 1 { "" } else { "s" },
            if got == 1 { "is" } else { "are" }
        ))
        .at(span)
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::{ExprKind, IntTy, Ty};

    #[test]
    fn a_structure_literal_keeps_the_order_written() {
        let u = check("struct P { x: i32, y: i32 }\nfn f() -> P { P { y: 2, x: 1 } }");
        let v = u.fns[0].body.value.as_ref().unwrap();
        let ExprKind::StructLit { fields } = &v.kind else {
            panic!("{:?}", v.kind)
        };
        let order: Vec<u32> = fields.iter().map(|(i, _)| *i).collect();
        assert_eq!(order, [1, 0], "evaluated in the order written");
    }

    #[test]
    fn every_field_needs_exactly_one_value() {
        assert_eq!(codes("struct P { x: i32, y: i32 }\nfn f() -> P { P { x: 1 } }"), [Code::Es15]);
        assert_eq!(codes("struct P { x: i32 }\nfn f() -> P { P { x: 1, x: 2 } }"), [Code::Es05]);
        assert_eq!(codes("struct P { x: i32 }\nfn f() -> P { P { x: 1, z: 2 } }"), [Code::Es04]);
        assert_eq!(codes("struct P { x: i32 }\nfn f() -> P { P { x: true } }"), [Code::Es06]);
    }

    #[test]
    fn variants_are_built_by_name_or_by_position() {
        check(
            "enum S { Circle(i64), Rect { w: i64, h: i64 } }\n\
             fn f() -> S { S::Circle(1) }\n\
             fn g() -> S { S::Rect { h: 2, w: 3 } }",
        );
        assert_eq!(codes("enum S { C(i64) }\nfn f() -> S { S::C(1, 2) }"), [Code::Es07]);
        assert_eq!(codes("enum S { C(i64) }\nfn f() -> S { S::C { x: 1 } }"), [Code::Es07]);
        assert_eq!(codes("enum S { R { w: i64 } }\nfn f() -> S { S::R(1) }"), [Code::Es07]);
    }

    #[test]
    fn an_array_literal_has_one_element_type() {
        let u = check("fn f() -> [u8; 3] { [1, 2, 3] }");
        assert_eq!(u.types.as_array(u.fns[0].ret), Some((Ty::Int(IntTy::U8), 3)));
        assert_eq!(codes("fn f() { let a = [1, true]; }"), [Code::Es06]);
        check("fn f() -> [bool; 0] { [] }");
    }

    #[test]
    fn a_repeated_element_must_be_copyable() {
        check("fn f() -> [u8; 4] { [0; 4] }");
        assert_eq!(codes("struct P { }\nfn f() -> [P; 2] { [P { }; 2] }"), [Code::Es12]);
        check("struct P { }\nfn f() -> [P; 1] { [P { }; 1] }");
    }
}
