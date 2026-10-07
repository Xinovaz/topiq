//! What a path in an expression names.
//!
//! A one-part path is a binding, a function, an object or a type: the nearest
//! block binding wins, then unit scope, then the `core` library. A longer path
//! starts either with an imported unit (`geometry::area`,
//! `geometry::Shape::Circle`) or with an enumeration, naming one of its
//! variants (`Shape::Circle`). An alias of an enumeration names its variants
//! as the enumeration does, so `fs::Error::NotFound` is `io::Error::NotFound`
//! when `fs` declares `type Error = io::Error;`. Structure fields and methods
//! are never reached with `::`; they use `.`.

use crate::ast::{Path, TArg};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{AdtId, Callee, Expr, ExprKind, Ty, VariantShape};

use super::body::Checker;
use super::core;
use super::imports;
use super::report;
use super::items::GenericKey;
use super::scope::Def;

/// What a path names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolution {
    /// A value: binding, object, function or library function.
    Value(Def),
    /// A structure or enumeration.
    Adt(AdtId),
    /// One variant of an enumeration.
    Variant(AdtId, u32),
}

impl Checker<'_, '_> {
    /// The generic declaration a path starts with, unless its first part is
    /// a binding of the block scopes here, which hides it.
    pub(super) fn generic_path(&mut self, segments: &[Spanned<Symbol>]) -> Option<(GenericKey, usize)> {
        let first = segments.first()?;
        if self.scopes.in_blocks(first.node).is_some() {
            return None;
        }
        self.cx.generic_at(segments)
    }

    /// Resolves a path, reporting a name that is not found.
    ///
    /// Generic arguments make an instance: `sum::<i32>` names the instance of
    /// `sum` made with `i32`, and `Pair<i32, bool>` the structure made with
    /// those.
    pub fn resolve_path(&mut self, path: &Path, args: &[Spanned<TArg>], span: Span) -> Option<Resolution> {
        let r = self.resolve_path_as_written(path, args, span)?;
        match r {
            Resolution::Value(Def::Fn(id)) => self.cx.use_callee(Callee::Fn(id), span),
            Resolution::Value(Def::Extern(id)) => self.cx.use_callee(Callee::Extern(id), span),
            Resolution::Value(Def::Global(id)) => self.cx.use_global(id, span),
            Resolution::Adt(id) | Resolution::Variant(id, _) => self.cx.use_adt(id, span),
            _ => {}
        }
        Some(r)
    }

    /// What `path` names, with no warning of a deprecated item.
    fn resolve_path_as_written(&mut self, path: &Path, args: &[Spanned<TArg>], span: Span) -> Option<Resolution> {
        // a generic declaration, of this unit or an imported one: named with
        // its arguments, or with a variant of it after it and the arguments
        // after that, as in `Maybe::Nothing::<char>`
        if let Some((key, used)) = self.generic_path(&path.segments) {
            match path.segments[used..] {
                [] => return self.generic_instance(key, args, span),
                [variant] => return self.generic_variant_path(key, variant, args, span),
                _ => {}
            }
        }
        match path.segments.as_slice() {
            [name] => {
                if !args.is_empty() {
                    let text = self.name(name.node).to_owned();
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{text}` takes no generic arguments"))
                            .at(span),
                    );
                    return None;
                }
                if let Some(d) = self.resolve(name.node) {
                    return Some(Resolution::Value(d));
                }
                if let Some(t) = self.cx.lookup_type(name.node) {
                    return Some(Resolution::Adt(t));
                }
                let mut d = report::not_found(span, "value", self.name(name.node), self.visible());
                if self.func.is_some_and(|f| self.cx.fn_layers.contains_key(&f)) {
                    d = d.with_note(
                        "a function declared inside a block cannot use the bindings of the function around \
                         it: only a closure captures",
                    );
                }
                self.report(d);
                None
            }
            [first, second] => self.two_part(*first, *second, span),
            [unit, ty, variant] => {
                let Some(u) = self.cx.scope.get_unit(unit.node) else {
                    let d = imports::unknown_unit(self.cx, unit.node, unit.span);
                    self.report(d);
                    return None;
                };
                if let Some(id) = imports::import_type(self.cx, u, ty.node) {
                    return self.variant_of(id, *variant);
                }
                match self.cx.imported_alias(u, ty.node, ty.span) {
                    Some(aliased) => self.aliased_variant(*ty, aliased, *variant),
                    None => {
                        let d = imports::no_such_item(self.cx, u, ty.node, ty.span, "type");
                        self.report(d);
                        None
                    }
                }
            }
            _ => {
                let written = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
                self.report(report::path_too_long(span, &written, 3));
                None
            }
        }
    }

    /// `a::b`: an imported unit's item, a library function, or an
    /// enumeration's variant.
    fn two_part(&mut self, first: Spanned<Symbol>, second: Spanned<Symbol>, span: Span) -> Option<Resolution> {
        if let Some(u) = self.cx.scope.get_unit(first.node) {
            if let Some(v) = imports::import_value(self.cx, u, second.node, span) {
                return match v {
                    Ok(d) => Some(Resolution::Value(d)),
                    Err(d) => {
                        self.report(*d);
                        None
                    }
                };
            }
            if let Some(t) = imports::import_type(self.cx, u, second.node) {
                return Some(Resolution::Adt(t));
            }
            // what the compiler provides on the library's behalf
            if self.name(first.node) == "core"
                && let Some(i) = core::lookup(self.name(second.node))
            {
                return Some(Resolution::Value(Def::Intrinsic(i)));
            }
            let d = imports::no_such_item(self.cx, u, second.node, second.span, "item");
            self.report(d);
            return None;
        }
        if self.name(first.node) == "core" {
            return match core::lookup(self.name(second.node)) {
                Some(i) => Some(Resolution::Value(Def::Intrinsic(i))),
                None => {
                    let mut d = Diagnostic::new(Code::Es04)
                        .with_message(format!(
                            "the `core` library has no function named `{}`",
                            self.name(second.node)
                        ))
                        .at(second.span);
                    if let Some(s) = report::closest(self.name(second.node), core::NAMES) {
                        d = d.with_help(format!("did you mean `core::{s}`?"));
                    }
                    self.report(d);
                    None
                }
            };
        }
        // an alias the unit declares hides a type of the `core` library
        if !self.cx.declares_alias(first.node)
            && let Some(t) = self.cx.lookup_type(first.node)
        {
            return self.variant_of(t, second);
        }
        if let Some(aliased) = self.cx.alias(first.node, first.span) {
            return self.aliased_variant(first, aliased, second);
        }
        let mut d = imports::unknown_unit(self.cx, first.node, first.span);
        if self.resolve(first.node).is_some() {
            d = d.with_help(format!(
                "`{}` is a value; a field is reached with `.`, as in `{}.{}`",
                self.name(first.node),
                self.name(first.node),
                self.name(second.node)
            ));
        }
        self.report(d);
        None
    }

    /// The variant `name` of the type the alias `alias` stands for, which
    /// has variants only when it is an enumeration.
    fn aliased_variant(&mut self, alias: Spanned<Symbol>, aliased: Ty, name: Spanned<Symbol>) -> Option<Resolution> {
        if let Ty::Adt(id) = aliased {
            return self.variant_of(id, name);
        }
        let what = self.describe(aliased);
        self.report(
            Diagnostic::new(Code::Es04)
                .with_message(format!(
                    "`{}` stands for {what}, which has no variants; `{}::{}` names nothing",
                    self.name(alias.node),
                    self.name(alias.node),
                    self.name(name.node)
                ))
                .at(name.span),
        );
        None
    }

    /// The variant `name` of the enumeration `id`.
    pub(super) fn variant_in(&mut self, id: AdtId, name: Spanned<Symbol>) -> Option<Resolution> {
        self.variant_of(id, name)
    }

    /// The variant `name` of the enumeration `id`.
    fn variant_of(&mut self, id: AdtId, name: Spanned<Symbol>) -> Option<Resolution> {
        let interner = self.interner;
        let def = self.adt(id).clone();
        let type_name = self.types().adt_name(id, interner);
        if def.is_struct() {
            self.report(
                Diagnostic::new(Code::Es04)
                    .with_message(format!(
                        "`{type_name}` is a structure, so it has no variants; `{type_name}::{}` names nothing",
                        interner.resolve(name.node)
                    ))
                    .at(name.span)
                    .with_help(format!(
                        "a structure's fields are reached with `.` on a value of it, as in `p.{}`",
                        interner.resolve(name.node)
                    )),
            );
            return None;
        }
        match def.variant_index(name.node) {
            Some(v) => Some(Resolution::Variant(id, v)),
            None => {
                let mut d = Diagnostic::new(Code::Es04)
                    .with_message(format!(
                        "`{type_name}` has no variant named `{}`",
                        interner.resolve(name.node)
                    ))
                    .at(name.span);
                let names = def.variants().iter().map(|v| interner.resolve(v.name));
                if let Some(s) = report::closest(interner.resolve(name.node), names) {
                    d = d.with_help(format!("did you mean `{type_name}::{s}`?"));
                }
                self.report(d);
                None
            }
        }
    }

    /// An instance of a generic declaration named with its arguments. `None`
    /// when something was wrong, which has been reported.
    fn generic_instance(&mut self, key: GenericKey, args: &[Spanned<TArg>], span: Span) -> Option<Resolution> {
        let params = self.cx.generic_params(key)?.to_vec();
        let text = self.cx.generic_label(key);
        if args.is_empty() {
            let kind = if self.cx.is_generic_fn(key) { "function" } else { "type" };
            let d = super::generic::needs_arguments(&text, kind, params.len(), span, true);
            self.report(d);
            return None;
        }
        let resolved = self.cx.generic_args(&text, &params, args, span)?;
        if self.cx.is_generic_fn(key) {
            return self
                .cx
                .instantiate_fn(key, resolved, span)
                .map(|id| Resolution::Value(Def::Fn(id)));
        }
        self.cx.instantiate_adt(key, resolved, span).map(Resolution::Adt)
    }

    /// A variant of a generic enumeration, with the enumeration's arguments
    /// written after it: `Maybe::Nothing::<char>`.
    fn generic_variant_path(
        &mut self,
        key: GenericKey,
        variant: Spanned<Symbol>,
        args: &[Spanned<TArg>],
        span: Span,
    ) -> Option<Resolution> {
        let text = self.cx.generic_label(key);
        let name = self.name(variant.node).to_owned();
        let params = self.cx.generic_params(key).map(<[crate::ast::GenericParam]>::to_vec)?;
        if args.is_empty() {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("nothing here says what the arguments of `{text}` are"))
                    .at(span)
                    .with_note(
                        "a generic enumeration's arguments are worked out from the values a \
                         variant carries, and this one carries none",
                    )
                    .with_help(format!("write them, as in `{text}::{name}::<…>`")),
            );
            return None;
        }
        let resolved = self.cx.generic_args(&text, &params, args, span)?;
        let id = self.cx.instantiate_adt(key, resolved, span)?;
        self.variant_of(id, variant)
    }

    /// A path used as a value: a binding, an object, a constant generic
    /// parameter, or an item that is not a value, which is reported.
    pub fn path_value(&mut self, path: &Path, args: &[Spanned<TArg>], span: Span) -> Expr {
        // a `const` generic parameter is a value while its instance is being
        // checked
        if let [name] = path.segments.as_slice()
            && let Some(crate::tir::Arg::Const(v)) = self.cx.param_arg(name.node) {
                let ty = self.infer.fresh();
                return Expr::constant(crate::tir::Value::Int(v, crate::tir::IntTy::USIZE), ty, span);
            }
        // `i` and `isq2`, where nothing else has the name
        if let [name] = path.segments.as_slice()
            && args.is_empty()
            && matches!(self.name(name.node), "i" | "isq2")
            && self.resolve(name.node).is_none()
            && self.cx.lookup_type(name.node).is_none()
            && let Some(e) = self.exact_name(self.name(name.node), span)
        {
            return e;
        }
        // a variant of a generic enumeration that carries nothing, written
        // without the enumeration's arguments: its type is whichever instance
        // the context wants, settled with the literals
        let variant = match self.prelude_variant(path) {
            Some(v) => Some(v),
            None => self.generic_path(&path.segments).and_then(|(key, used)| match path.segments[used..] {
                [second] => Some((key, second)),
                _ => None,
            }),
        };
        if args.is_empty()
            && let Some((key, second)) = variant
            && let Some(decl) = self.cx.generic_enum_decl(key)
            && let Some(v) = decl.iter().position(|x| x.name.node == second.node)
            && decl[v].payload.is_none()
        {
            let ty = self.infer.fresh_instance(key.name, self.cx.generic_origin(key));
            let what = format!("{}::{}", self.cx.generic_label(key), self.name(second.node));
            self.pending.push(super::body::Pending {
                ty,
                span,
                what,
                args: None,
            });
            return Expr {
                kind: ExprKind::Variant {
                    variant: v as u32,
                    fields: Vec::new(),
                },
                ty,
                span,
            };
        }
        // a type's associated function or method, as a value
        if let Some((entry, label)) = self.associated(path) {
            return match entry.target {
                super::method::MethodTarget::Fn(id) => self.function_value(Callee::Fn(id), path, span),
                super::method::MethodTarget::Extern(id) => self.function_value(Callee::Extern(id), path, span),
                super::method::MethodTarget::Generic(_) => {
                    self.report(
                        Diagnostic::new(Code::Es07)
                            .with_message(format!("`{label}` is generic, so it is only a value once its arguments are known"))
                            .at(span)
                            .with_help(format!("call it, as in `{label}(…)`, and its arguments are worked out")),
                    );
                    Checker::error(span)
                }
            };
        }
        let Some(r) = self.resolve_path(path, args, span) else {
            return Checker::error(span);
        };
        match r {
            Resolution::Value(Def::Local(id)) => Expr::local(id, self.locals[id.index()].ty, span),
            Resolution::Value(Def::Global(id)) => {
                self.cx.global_ty(id);
                Expr {
                    kind: ExprKind::Global(id),
                    ty: self.cx.unit.global(id).ty,
                    span,
                }
            }
            Resolution::Value(Def::Fn(id)) if self.cx.unit.init == Some(id) => {
                let n = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
                let d = self.initializer_used(&n, span);
                self.report(d);
                Checker::error(span)
            }
            Resolution::Value(Def::Fn(id)) => self.function_value(Callee::Fn(id), path, span),
            Resolution::Value(Def::Extern(id)) => self.function_value(Callee::Extern(id), path, span),
            Resolution::Value(Def::Capture(i)) => self.captured(i, span),
            Resolution::Value(Def::Intrinsic(which)) => self.library_function_value(which, path, span),
            Resolution::Adt(id) => {
                let name = self.types().adt_name(id, self.interner);
                let def = self.adt(id);
                let help = if def.is_struct() {
                    format!("a value of it is written `{name} {{ field: value, … }}`")
                } else {
                    format!("a value of it is one of its variants, such as `{name}::…`")
                };
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{name}` is a type, not a value"))
                        .at(span)
                        .with_help(help),
                );
                Checker::error(span)
            }
            Resolution::Variant(id, v) => self.unit_variant(id, v, span),
        }
    }

    /// `Enum::Variant` written as a value: fine for a variant without fields,
    /// and a mistake otherwise, since the fields need values.
    fn unit_variant(&mut self, id: AdtId, v: u32, span: Span) -> Expr {
        let name = self.types().adt_name(id, self.interner);
        let var = self.adt(id).variants()[v as usize].clone();
        let vname = self.name(var.name);
        match var.shape {
            VariantShape::Unit => Expr {
                kind: ExprKind::Variant {
                    variant: v,
                    fields: Vec::new(),
                },
                ty: Ty::Adt(id),
                span,
            },
            VariantShape::Tuple => {
                let blanks = vec!["…"; var.fields.len()].join(", ");
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!(
                            "`{name}::{vname}` holds {} value{}, which must be given",
                            var.fields.len(),
                            if var.fields.len() == 1 { "" } else { "s" }
                        ))
                        .at(span)
                        .with_help(format!("write `{name}::{vname}({blanks})`")),
                );
                Checker::error(span)
            }
            VariantShape::Struct => {
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("`{name}::{vname}` has fields, which must be given"))
                        .at(span)
                        .with_help(format!("write `{name}::{vname} {{ field: value, … }}`")),
                );
                Checker::error(span)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};

    #[test]
    fn a_unit_variant_is_a_value() {
        check("enum Guess { Constant, Balanced }\nfn f() -> Guess { Guess::Balanced }");
    }

    #[test]
    fn a_variant_with_fields_needs_them() {
        assert_eq!(codes("enum S { C(i64) }\nfn f() -> S { S::C }"), [Code::Es07]);
        assert_eq!(codes("enum S { R { w: i64 } }\nfn f() -> S { S::R }"), [Code::Es07]);
    }

    #[test]
    fn a_misspelt_variant_suggests_the_intended_one() {
        let (_, d) = crate::sema::testing::analyzed("enum G { Balanced }\nfn f() -> G { G::Balancd }");
        assert_eq!(d[0].code, Code::Es04);
        assert!(d[0].helps.iter().any(|h| h.contains("G::Balanced")), "{:?}", d[0].helps);
    }

    #[test]
    fn an_alias_of_an_enumeration_names_its_variants() {
        check(
            "enum Guess { Constant, Balanced(i64) }\ntype G = Guess;\n\
             fn f() -> G { G::Constant }\n\
             fn g(x: G) -> i64 { match x { G::Balanced(n) => n, G::Constant => 0 } }",
        );
    }

    #[test]
    fn an_alias_of_anything_else_has_no_variants() {
        let (_, d) = crate::sema::testing::analyzed("type N = i64;\nfn f() -> N { N::Zero }");
        assert_eq!(d[0].code, Code::Es04);
        assert!(d[0].message.contains("has no variants"), "{}", d[0].message);
    }

    #[test]
    fn a_type_is_not_a_value() {
        assert_eq!(codes("struct P { }\nfn f() { let x = P; }"), [Code::Es06]);
    }

    #[test]
    fn a_path_through_a_value_suggests_a_field() {
        let (_, d) = crate::sema::testing::analyzed("fn f(p: i32) { let x = p::q; }");
        assert_eq!(d[0].code, Code::Es04);
        assert!(d[0].helps.iter().any(|h| h.contains("p.q")), "{:?}", d[0].helps);
    }

    #[test]
    fn the_core_library_is_reached_by_its_name() {
        assert_eq!(codes("fn f() { core::prnt(\"x\"); }"), [Code::Es04]);
    }
}
