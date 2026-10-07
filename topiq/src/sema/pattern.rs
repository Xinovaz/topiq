//! `match` and the patterns in its arms.
//!
//! Arms are tried in order, and the first whose pattern matches is chosen.
//! A pattern is one of:
//!
//! - `_`, which matches anything and binds nothing;
//! - a single name, which matches anything and binds it: a bare name always
//!   binds, even if a constant of that name exists, so `n => …` never
//!   compares against some `n` declared elsewhere;
//! - a literal (an integer (negative ones included), `true`, `false`, or a
//!   character), which matches that value;
//! - a path, `Guess::Balanced` or `limits::MAX`, naming a variant without
//!   fields or a constant. A constant of a structure, tuple or enumeration
//!   matches as the pattern its value is written as; one holding a floating
//!   value, text or an array cannot be matched;
//! - `Shape::Circle(r)`, a tuple variant with a pattern for each value;
//! - `Point { x: 0, y }` or `Shape::Rect { w, h: 1 }`, naming fields. A field
//!   written alone, as `y`, binds it; a field not mentioned matches anything.
//!
//! # Matching through a reference
//!
//! A `match` on a reference matches what it refers to, as field access does.
//! Binding part of it by name copies that part out when its type can be
//! copied; a part that cannot be copied is not moved out through the
//! reference but referred to, so the name is a reference to it, permitting
//! what the matched reference permits: in a `match` on a `*Value`,
//! `Value::Str(s)` makes `s` a `*[char]`.
//!
//! Every value must be matched by some arm, which [`super::exhaust`] checks
//! once the types are settled.

use crate::ast::{self, FieldPat, MatchArm, Pattern};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{AdtId, Arm, Expr, ExprKind, Pat, PatKind, Ty, Value, VariantShape};

use super::body::Checker;
use super::path::Resolution;
use super::report;
use super::scope::Def;

impl Checker<'_, '_> {
    /// `match scrutinee { arms }`.
    pub(super) fn match_expr(&mut self, scrutinee: &Spanned<ast::Expr>, arms: &[MatchArm], span: Span) -> Expr {
        let s = self.expr(scrutinee);
        self.match_checked(s, arms, span)
    }

    /// A `match` on a scrutinee already checked.
    pub(super) fn match_checked(&mut self, s: Expr, arms: &[MatchArm], span: Span) -> Expr {
        // a reference is matched through, remembering what it permits
        let st = self.shallow(s.ty);
        let through = self.types().as_ref(st).map(|(access, _)| access);
        let s = self.auto_deref(s);
        let sty = s.ty;

        // each arm in a scope of its own, all giving one type
        let mut out: Vec<Arm> = Vec::with_capacity(arms.len());
        let mut ty: Option<Ty> = None;
        for arm in arms {
            self.scopes.enter();
            let mut bound = Vec::new();
            let outer = std::mem::replace(&mut self.match_through, through);
            let pat = self.pattern(&arm.pattern, sty, &mut bound);
            self.match_through = outer;
            let body = self.expr(&arm.body);
            self.scopes.leave();
            let t = match ty {
                None => body.ty,
                Some(prev) => match self.unify(prev, body.ty) {
                    Ok(t) => t,
                    Err(_) => {
                        let (a, b) = (self.describe(prev), self.describe(body.ty));
                        self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!(
                                    "the arms of this `match` give different types: {a} and {b}"
                                ))
                                .at_with(body.span, format!("this arm gives {b}"))
                                .with_note("whichever arm runs, the `match` has one type, so all arms must agree"),
                        );
                        prev
                    }
                },
            };
            ty = Some(t);
            out.push(Arm { pat, body });
        }
        Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(s),
                arms: out,
            },
            ty: ty.unwrap_or(Ty::Never),
            span,
        }
    }

    /// Checks a pattern against the type of the value it matches, binding its
    /// names in the current scope. `bound` collects the names bound so far in
    /// this arm, so one name cannot be bound twice.
    pub(super) fn pattern(&mut self, p: &Spanned<Pattern>, expected: Ty, bound: &mut Vec<(Symbol, Span)>) -> Pat {
        // matching the stand-in for a value already reported as wrong, a
        // pattern cannot fit or fail to: saying it does not would only
        // repeat the first mistake
        let stand_in = self.shallow(expected) == Ty::Never;
        let before = self.cx.diags.len();
        let pat = self.pattern_as_written(p, expected, bound);
        if stand_in {
            let made = self.cx.diags.split_off(before);
            self.cx.diags.extend(made.into_iter().filter(|d| d.code != Code::Es06));
        }
        pat
    }

    /// [`Checker::pattern`], reporting every pattern that does not fit.
    fn pattern_as_written(&mut self, p: &Spanned<Pattern>, expected: Ty, bound: &mut Vec<(Symbol, Span)>) -> Pat {
        let span = p.span;
        let bad = bad(span);
        match &p.node {
            Pattern::Wildcard => Pat::wild(expected, span),
            // a bare name that names a `core` variant, or a constant of the
            // unit, is that variant or constant; any other binds
            Pattern::Binding(name) => {
                let path = ast::Path::single(*name);
                let names_something = self.prelude_variant(&path).is_some()
                    || matches!(
                        self.cx.lookup_value(name.node),
                        Some(Def::Global(g)) if { self.cx.global_ty(g); self.cx.unit.global(g).constant }
                    ) && self.scopes.in_blocks(name.node).is_none();
                if names_something {
                    return self.pattern(&Spanned::new(Pattern::Path(path), span), expected, bound);
                }
                self.bind(*name, expected, bound)
            }
            Pattern::Literal(e) => self.literal_pattern(e, expected),
            Pattern::Tuple(elems) => self.tuple_pattern(elems, expected, span, bound),
            Pattern::Path(path) => match self.resolve_pattern_path(path, expected, span) {
                Some(Resolution::Variant(id, v)) => {
                    if !self.pattern_type(Ty::Adt(id), expected, span) {
                        return bad;
                    }
                    let var = self.adt(id).variants()[v as usize].clone();
                    if var.shape != VariantShape::Unit {
                        let name = self.variant_name(id, v);
                        let (form, help) = if var.shape == VariantShape::Tuple {
                            ("by position", format!("{name}({})", vec!["_"; var.fields.len()].join(", ")))
                        } else {
                            ("by name", format!("{name} {{ … }}"))
                        };
                        self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!("`{name}` holds values {form}, so its pattern must say how to match them"))
                                .at(span)
                                .with_help(format!("write `{help}`, using `_` for a value you do not need")),
                        );
                        return bad;
                    }
                    Pat::variant(v, Vec::new(), Ty::Adt(id), span)
                }
                Some(Resolution::Value(Def::Global(g))) => self.constant_pattern(g, expected, span),
                Some(_) => {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message("a path in a pattern names a variant without fields, or a constant")
                            .at(span),
                    );
                    bad
                }
                None => bad,
            },
            Pattern::TupleStruct { path, elements } => {
                let Some(Resolution::Variant(id, v)) = self.resolve_pattern_path(path, expected, span) else {
                    self.not_a_variant(path, span);
                    return bad;
                };
                if !self.pattern_type(Ty::Adt(id), expected, span) {
                    return bad;
                }
                let var = self.adt(id).variants()[v as usize].clone();
                let name = self.variant_name(id, v);
                if var.shape != VariantShape::Tuple {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{name}` does not hold values by position"))
                            .at(span)
                            .with_help(if var.shape == VariantShape::Unit {
                                format!("write just `{name}`")
                            } else {
                                format!("name its fields, as in `{name} {{ … }}`")
                            }),
                    );
                    return bad;
                }
                if elements.len() != var.fields.len() {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "`{name}` holds {} value{}, but this pattern matches {}",
                                var.fields.len(),
                                if var.fields.len() == 1 { "" } else { "s" },
                                elements.len()
                            ))
                            .at(span)
                            .with_help("give one pattern per value, using `_` for any you do not need"),
                    );
                    return bad;
                }
                let fields = elements
                    .iter()
                    .zip(&var.fields)
                    .enumerate()
                    .map(|(i, (e, f))| (i as u32, self.pattern(e, f.ty, bound)))
                    .collect();
                Pat::variant(v, fields, Ty::Adt(id), span)
            }
            Pattern::Struct { path, fields } => {
                let (id, variant) = match self.resolve_pattern_path(path, expected, span) {
                    Some(Resolution::Adt(id)) if self.adt(id).is_struct() => (id, None),
                    Some(Resolution::Variant(id, v)) => (id, Some(v)),
                    Some(_) => {
                        self.not_a_variant(path, span);
                        return bad;
                    }
                    None => return bad,
                };
                if !self.pattern_type(Ty::Adt(id), expected, span) {
                    return bad;
                }
                self.fields_pattern(id, variant, fields, span, bound)
            }
        }
    }

    /// A name binding the matched value.
    fn bind(&mut self, name: Spanned<Symbol>, ty: Ty, bound: &mut Vec<(Symbol, Span)>) -> Pat {
        if let Some(&(_, earlier)) = bound.iter().find(|(n, _)| *n == name.node) {
            let d = report::duplicate(name.span, earlier, self.name(name.node), "in one pattern");
            self.report(d);
        }
        bound.push((name.node, name.span));
        // through a reference, a part that cannot be copied is not moved out
        // but referred to
        if let Some(access) = self.match_through {
            let settled = self.shallow(ty);
            if !matches!(settled, Ty::Infer(_)) && !self.types().is_copyable(settled) {
                let rty = self.types_mut().reference(access, settled);
                let local = self.new_local(name.node, rty, false, name.span);
                self.scopes.bind(name.node, Def::Local(local));
                return Pat {
                    kind: PatKind::BindRef(local),
                    ty: settled,
                    span: name.span,
                };
            }
        }
        let local = self.new_local(name.node, ty, false, name.span);
        self.scopes.bind(name.node, Def::Local(local));
        Pat::bind(local, ty, name.span)
    }

    /// `Point { x: p, y }` or `Shape::Rect { w, h: 1 }`.
    fn fields_pattern(
        &mut self,
        id: AdtId,
        variant: Option<u32>,
        fields: &[FieldPat],
        span: Span,
        bound: &mut Vec<(Symbol, Span)>,
    ) -> Pat {
        // the fields declared, and what the message calls their owner
        let def = self.adt(id).clone();
        let (decl, what) = match variant {
            None => (def.fields().to_vec(), self.types().adt_name(id, self.interner)),
            Some(v) => {
                let var = &def.variants()[v as usize];
                if var.shape != VariantShape::Struct {
                    let name = self.variant_name(id, v);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{name}` has no named fields"))
                            .at(span),
                    );
                    return Pat::wild(Ty::Adt(id), span);
                }
                (var.fields.clone(), self.variant_name(id, v))
            }
        };

        // each field matched, once, and one the type has
        let mut out: Vec<(u32, Pat)> = Vec::new();
        for f in fields {
            let text = self.name(f.name.node);
            let Some(i) = decl.iter().position(|d| d.name == f.name.node) else {
                let interner = self.interner;
                let mut d = Diagnostic::new(Code::Es04)
                    .with_message(format!("`{what}` has no field named `{text}`"))
                    .at(f.name.span);
                if let Some(s) = report::closest(text, decl.iter().map(|d| interner.resolve(d.name))) {
                    d = d.with_help(format!("did you mean `{s}`?"));
                }
                self.report(d);
                continue;
            };
            if out.iter().any(|(j, _)| *j as usize == i) {
                self.report(
                    Diagnostic::new(Code::Es05)
                        .with_message(format!("the field `{text}` is matched twice in this pattern"))
                        .at(f.name.span),
                );
                continue;
            }
            let fty = decl[i].ty;
            let sub = match &f.pattern {
                Some(p) => self.pattern(p, fty, bound),
                None => self.bind(f.name, fty, bound), // `x` alone binds `x`
            };
            out.push((i as u32, sub));
        }

        // in declaration order
        out.sort_by_key(|(i, _)| *i);
        let kind = match variant {
            None => PatKind::Struct { fields: out },
            Some(v) => PatKind::Variant { variant: v, fields: out },
        };
        Pat {
            kind,
            ty: Ty::Adt(id),
            span,
        }
    }

    /// A literal pattern: an integer, a boolean or a character.
    fn literal_pattern(&mut self, e: &Spanned<ast::Expr>, expected: Ty) -> Pat {
        let span = e.span;
        let value = match &e.node {
            ast::Expr::Int { raw, base, suffix } => self.int_literal(*raw, base.radix(), *suffix, false, span),
            ast::Expr::Unary {
                op: ast::UnOp::Neg,
                operand,
            } => match &operand.node {
                ast::Expr::Int { raw, base, suffix } => self.int_literal(*raw, base.radix(), *suffix, true, span),
                _ => Checker::error(span),
            },
            ast::Expr::Bool(b) => Expr::constant(Value::Bool(*b), Ty::Bool, span),
            ast::Expr::Char(c) => Expr::constant(Value::Char(*c), Ty::Char, span),
            ast::Expr::Str { .. } => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a string cannot be matched against a pattern")
                        .at(span)
                        .with_note(
                            "a pattern compares against one integer, `bool` or `char` at a time; \
                             a string is a reference to many characters",
                        ),
                );
                Checker::error(span)
            }
            ast::Expr::Float { raw, suffix } => self.float_literal(*raw, *suffix, span),
            _ => Checker::error(span),
        };
        let ExprKind::Const(v) = value.kind else {
            unreachable!("a literal pattern checks to a constant");
        };
        if value.ty == Ty::Never {
            // already reported: stand in with a pattern marked as an error
            return bad(span);
        }
        let ty = match self.unify(value.ty, expected) {
            Ok(t) => t,
            Err(_) => {
                self.report_pattern_mismatch(value.ty, expected, span);
                expected
            }
        };
        Pat {
            kind: PatKind::Const(v),
            ty,
            span,
        }
    }

    /// A constant named in a pattern, which must be an integer, `bool` or
    /// `char`.
    fn constant_pattern(&mut self, g: crate::tir::GlobalId, expected: Ty, span: Span) -> Pat {
        let bad = bad(span);
        self.cx.global_ty(g);
        let global = self.cx.unit.global(g);
        let (ty, constant, name) = (global.ty, global.constant, self.name(global.name));
        if !constant {
            self.report(
                Diagnostic::new(Code::Ec01)
                    .with_message(format!(
                        "`{name}` is not a constant, so its value is not known to compare against"
                    ))
                    .at(span)
                    .with_note("a pattern compares against values known during translation"),
            );
            return bad;
        }
        if !self.pattern_type(ty, expected, span) {
            return bad;
        }
        let Some(v) = self.cx.global_value(g, span) else {
            return bad;
        };
        // a structure, tuple or variant matches as the pattern its value
        // is written as: field by field, down to integers, `bool`s and
        // `char`s
        match self.value_pattern(&v, ty, span) {
            Some(p) => p,
            None => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{name}` cannot be matched against: it holds a value no pattern compares"))
                        .at(span)
                        .with_note(
                            "a pattern compares integers, `bool`s and `char`s, and structures, tuples \
                             and variants made of them; not floating values, text or arrays",
                        )
                        .with_help(format!("compare with `==` in an `if` instead, or match the parts of `{name}` that can be")),
                );
                bad
            }
        }
    }

    /// The pattern that matches exactly `v`, a value of `ty`, if there is
    /// one.
    fn value_pattern(&mut self, v: &Value, ty: Ty, span: Span) -> Option<Pat> {
        let ty = self.shallow(ty);
        let kind = match v {
            Value::Int(..) | Value::Bool(_) | Value::Char(_) => PatKind::Const(v.clone()),
            Value::Struct(parts) => {
                let types: Vec<Ty> = match ty {
                    Ty::Adt(id) => self.adt(id).fields().iter().map(|f| f.ty).collect(),
                    _ => self.types().as_tuple(ty)?.to_vec(),
                };
                PatKind::Struct {
                    fields: self.values_pattern(parts, types, span)?,
                }
            }
            Value::Enum { variant, fields: parts } => {
                let Ty::Adt(id) = ty else { return None };
                let types: Vec<Ty> = self.adt(id).variants()[*variant as usize].fields.iter().map(|f| f.ty).collect();
                PatKind::Variant {
                    variant: *variant,
                    fields: self.values_pattern(parts, types, span)?,
                }
            }
            _ => return None,
        };
        Some(Pat { kind, ty, span })
    }

    /// The patterns that match exactly `parts`, of `types`, numbered in
    /// order, if each has one.
    fn values_pattern(&mut self, parts: &[Value], types: Vec<Ty>, span: Span) -> Option<Vec<(u32, Pat)>> {
        (0u32..)
            .zip(parts.iter().zip(types))
            .map(|(i, (p, t))| Some((i, self.value_pattern(p, t, span)?)))
            .collect()
    }

    /// `(p1, p2, …)`: a pattern for each element, in order, and as many of
    /// them as the tuple has.
    fn tuple_pattern(
        &mut self,
        elems: &[Spanned<Pattern>],
        expected: Ty,
        span: Span,
        bound: &mut Vec<(Symbol, Span)>,
    ) -> Pat {
        let bad = bad(span);
        let expected = self.shallow(expected);
        let Some(types) = self.types().as_tuple(expected).map(<[Ty]>::to_vec) else {
            if expected != Ty::Never {
                let what = self.describe(expected);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("this pattern matches a tuple, but the value matched is {what}"))
                        .at(span),
                );
            }
            return bad;
        };
        if types.len() != elems.len() {
            let shown = self.show(expected);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "`{shown}` has {} element{}, and this pattern gives {}",
                        types.len(),
                        if types.len() == 1 { "" } else { "s" },
                        elems.len()
                    ))
                    .at(span)
                    .with_note("a tuple pattern matches every element, so it has one pattern for each"),
            );
            return bad;
        }
        let fields = elems
            .iter()
            .zip(&types)
            .enumerate()
            .map(|(i, (p, &t))| (i as u32, self.pattern(p, t, bound)))
            .collect();
        Pat {
            kind: PatKind::Struct { fields },
            ty: expected,
            span,
        }
    }

    /// Resolves the path of a pattern. A generic type or its variant is named
    /// without arguments, and means the instance of it being matched.
    fn resolve_pattern_path(&mut self, path: &ast::Path, expected: Ty, span: Span) -> Option<Resolution> {
        // matching needs to know which instance the value is: one still
        // waiting on its literals is made now
        if self.infer.unsettled_instance(expected).is_some() {
            self.settled(expected);
        }
        let expected = self.shallow(expected);
        if let Some((key, variant)) = self.prelude_variant(path) {
            let Some(id) = self.cx.instance_of(key, expected) else {
                let text = format!("{}::{}", self.cx.generic_label(key), self.name(variant.node));
                let what = self.describe(expected);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("this pattern matches `{text}`, but the value matched is {what}"))
                        .at(span),
                );
                return None;
            };
            return self.variant_in(id, variant);
        }
        if let Some((key, used)) = self.generic_path(&path.segments) {
            let Some(id) = self.cx.instance_of(key, expected) else {
                let text = self.cx.generic_label(key);
                let what = self.describe(expected);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("this pattern matches a `{text}`, but the value matched is {what}"))
                        .at(span),
                );
                return None;
            };
            return match path.segments[used..] {
                [] => Some(Resolution::Adt(id)),
                [variant] => self.variant_in(id, variant),
                _ => {
                    let written = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
                    self.report(report::path_too_long(span, &written, 3));
                    None
                }
            };
        }
        self.resolve_path(path, &[], span)
    }

    /// Requires a pattern of type `found` to match values of `expected`.
    fn pattern_type(&mut self, found: Ty, expected: Ty, span: Span) -> bool {
        if self.unify(found, expected).is_ok() {
            return true;
        }
        self.report_pattern_mismatch(found, expected, span);
        false
    }

    fn report_pattern_mismatch(&mut self, found: Ty, expected: Ty, span: Span) {
        let (f, e) = (self.describe(found), self.describe(expected));
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("this pattern matches {f}, but the value matched is {e}"))
                .at(span),
        );
    }

    fn variant_name(&self, id: AdtId, v: u32) -> String {
        let def = self.types().adt(id);
        format!(
            "{}::{}",
            self.types().adt_name(id, self.interner),
            self.name(def.variants()[v as usize].name)
        )
    }

    fn not_a_variant(&mut self, path: &ast::Path, span: Span) {
        let written = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("`{written}` is not a variant or structure that a pattern can take apart"))
                .at(span),
        );
    }
}

/// What an ill-formed pattern becomes: it matches anything, and its type
/// marks it as standing for an error already reported, so the coverage
/// check does not report around it.
fn bad(span: Span) -> Pat {
    Pat::wild(Ty::Never, span)
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};

    const SHAPE: &str = "enum Shape { Empty, Circle(i64), Rect { w: i64, h: i64 } }\n";

    #[test]
    fn every_kind_of_pattern_checks() {
        check(&format!(
            "{SHAPE}fn area(s: Shape) -> i64 {{\n\
                 match s {{\n\
                     Shape::Empty => 0,\n\
                     Shape::Circle(r) => 3 * r * r,\n\
                     Shape::Rect {{ w, h: 1 }} => w,\n\
                     Shape::Rect {{ w, h }} => w * h,\n\
                 }}\n\
             }}"
        ));
    }

    #[test]
    fn literals_match_integers_booleans_and_characters() {
        check("fn f(x: i32) -> i32 { match x { -1 => 0, 0 => 1, n => n } }");
        check("fn f(b: bool) -> i32 { match b { true => 1, false => 0 } }");
        check("fn f(c: char) -> bool { match c { 'a' => true, _ => false } }");
        assert_eq!(codes("fn f(x: i32) -> i32 { match x { true => 0, _ => 1 } }"), [Code::Es06]);
    }

    #[test]
    fn a_match_on_a_reference_matches_its_referent() {
        check(&format!(
            "{SHAPE}fn f(s: *Shape) -> i64 {{ match s {{ Shape::Circle(r) => r, _ => 0 }} }}"
        ));
    }

    #[test]
    fn a_pattern_must_fit_its_variant() {
        let wrong = [
            "fn f(s: Shape) -> i64 { match s { Shape::Circle => 0, _ => 1 } }",
            "fn f(s: Shape) -> i64 { match s { Shape::Circle(a, b) => 0, _ => 1 } }",
            "fn f(s: Shape) -> i64 { match s { Shape::Rect(a, b) => 0, _ => 1 } }",
        ];
        for w in wrong {
            assert_eq!(codes(&format!("{SHAPE}{w}")), [Code::Es06], "{w}");
        }
        assert_eq!(
            codes(&format!("{SHAPE}fn f(s: Shape) -> i64 {{ match s {{ Shape::Rect {{ z }} => 0, _ => 1 }} }}")),
            [Code::Es04]
        );
    }

    #[test]
    fn a_name_is_bound_once_per_pattern() {
        let src = format!("{SHAPE}fn f(s: Shape) -> i64 {{ match s {{ Shape::Rect {{ w: a, h: a }} => a, _ => 0 }} }}");
        assert_eq!(codes(&src), [Code::Es05]);
    }

    #[test]
    fn a_bare_name_that_names_a_constant_matches_it() {
        // `LIMIT` is the constant, so the arm after it is still reached; `n`
        // names nothing, so it binds
        check("let LIMIT: const i32 = 10;\nfn f(x: i32) -> i32 { match x { LIMIT => 1, _ => 0 } }");
        let got = codes("fn f(x: i32) -> i32 { match x { n => n, _ => 0 } }");
        assert_eq!(got, [Code::Es14]);
    }

    #[test]
    fn the_arms_agree_on_a_type() {
        assert_eq!(codes("fn f(b: bool) -> i32 { match b { true => 1, false => false } }"), [Code::Es06]);
    }
}
