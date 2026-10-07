//! `e?`: using the value in an `Opt` or a `Result`, or returning early.
//!
//! On an `Opt`, `e?` is the value `Some` carries, and a `None` makes the
//! enclosing function return `None`. On a `Result`, it is what `Ok` carries,
//! and an `Err(e)` makes the function return `Err(e)`. So the function must
//! return the same kind: an `Opt` of any type for the first, and a `Result`
//! for the second, whose error type is `e`'s, or an enumeration with exactly
//! one variant carrying one value of `e`'s type, which `e` is returned in, as
//! a `tcon::Error` is returned in `dyn::Error::Schema`. It is written out as
//! the `match` it stands for.

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{AdtId, Arm, Expr, ExprKind, Pat, Ty};

use super::body::Checker;

/// Which of the two types `?` works on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Opt,
    Result,
}

impl Checker<'_, '_> {
    /// Whether `id` is the `core` library's `Opt` or `Result`.
    fn try_kind(&self, id: AdtId) -> Option<Kind> {
        let def = self.types().adt(id);
        let is_core = match &def.origin {
            Some(o) => o == "core",
            None => self.cx.unit.name == "core",
        };
        if !is_core || def.args.is_empty() {
            return None;
        }
        match self.name(def.name) {
            "Opt" => Some(Kind::Opt),
            "Result" => Some(Kind::Result),
            _ => None,
        }
    }

    /// `operand?`.
    pub(super) fn try_expr(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Expr {
        let e = self.expr(operand);
        let t = self.settled(e.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        let Some((id, kind)) = (match t {
            Ty::Adt(id) => self.try_kind(id).map(|k| (id, k)),
            _ => None,
        }) else {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`?` works on an `Opt` or a `Result`, but this is {what}"))
                    .at(operand.span)
                    .with_note("`e?` is the value inside `e`, or else returns early with what `e` holds instead"),
            );
            return Checker::error(span);
        };
        let ret = self.settled(self.ret);
        let ret_kind = match ret {
            Ty::Adt(r) => self.try_kind(r).map(|k| (r, k)),
            _ => None,
        };
        let want = match kind {
            Kind::Opt => "an `Opt`",
            Kind::Result => "a `Result`",
        };
        let Some((ret_id, ret_kind)) = ret_kind.filter(|&(_, k)| k == kind) else {
            let what = self.describe(ret);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "`?` here may return early, so the function must return {want}, but it returns {what}"
                    ))
                    .at(span)
                    .with_note(match kind {
                        Kind::Opt => "on a `None`, `?` makes the function return `None`",
                        Kind::Result => "on an `Err(e)`, `?` makes the function return `Err(e)`",
                    })
                    .with_help("handle the other case with `match`, or change what the function returns"),
            );
            return Checker::error(span);
        };
        let _ = ret_kind;
        let def = self.adt(id).clone();
        let (hit, miss) = match kind {
            Kind::Opt => ("Some", "None"),
            Kind::Result => ("Ok", "Err"),
        };
        let index = |name: &str| {
            def.variants()
                .iter()
                .position(|v| self.name(v.name) == name)
                .expect("the variant is declared") as u32
        };
        let (hit_v, miss_v) = (index(hit), index(miss));
        let inner = def.variants()[hit_v as usize].fields[0].ty;
        let value = self.new_local(def.variants()[hit_v as usize].name, inner, false, span);
        let hit_arm = Arm {
            pat: Pat::variant(hit_v, vec![(0, Pat::bind(value, inner, span))], t, span),
            body: Expr::local(value, inner, span),
        };
        // what returns: `None`, or `Err(e)` with the error the value held
        let ret_def = self.adt(ret_id).clone();
        let ret_miss = ret_def
            .variants()
            .iter()
            .position(|v| self.name(v.name) == miss)
            .expect("the variant is declared") as u32;
        let (miss_fields, returned_fields) = match kind {
            Kind::Opt => (Vec::new(), Vec::new()),
            Kind::Result => {
                let err = def.variants()[miss_v as usize].fields[0].ty;
                let wanted = ret_def.variants()[ret_miss as usize].fields[0].ty;
                // an error enumeration with one variant for this error, as
                // `dyn::Error::Schema(tcon::Error)` is, takes it wrapped
                let (err_s, wanted_s) = (self.settled(err), self.settled(wanted));
                let wrap = if err_s == wanted_s { None } else { self.wrapping_variant(wanted_s, err_s) };
                if wrap.is_none() && self.unify(err, wanted).is_err() {
                    let (e, w) = (self.show(err), self.show(wanted));
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "`?` would return this `Err` from a function whose error type is `{w}`, but it holds a `{e}`"
                            ))
                            .at(span)
                            .with_help("convert the error first, as in `e.map_err(…)?`, or return that error type"),
                    );
                    return Checker::error(span);
                }
                let held = self.new_local(def.variants()[miss_v as usize].name, err, false, span);
                let mut returned = Expr::local(held, err, span);
                if let Some(wrap) = wrap {
                    returned = Expr {
                        kind: ExprKind::Variant {
                            variant: wrap,
                            fields: vec![(0, returned)],
                        },
                        ty: wanted_s,
                        span,
                    };
                }
                (vec![(0, Pat::bind(held, err, span))], vec![(0, returned)])
            }
        };
        // on a miss, a return of the function's own miss variant
        let miss_arm = Arm {
            pat: Pat::variant(miss_v, miss_fields, t, span),
            body: Expr {
                kind: ExprKind::Return(Some(Box::new(Expr {
                    kind: ExprKind::Variant {
                        variant: ret_miss,
                        fields: returned_fields,
                    },
                    ty: ret,
                    span,
                }))),
                ty: Ty::Never,
                span,
            },
        };
        Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(e),
                arms: vec![hit_arm, miss_arm],
            },
            ty: inner,
            span,
        }
    }

    /// The variant of the enumeration `outer` that carries exactly one
    /// value, of type `inner`, when it has exactly one such variant.
    fn wrapping_variant(&mut self, outer: Ty, inner: Ty) -> Option<u32> {
        let Ty::Adt(id) = outer else { return None };
        let def = self.adt(id).clone();
        let mut found = None;
        for (i, v) in def.variants().iter().enumerate() {
            if let [f] = v.fields.as_slice()
                && self.settled(f.ty) == inner
            {
                if found.is_some() {
                    return None;
                }
                found = Some(i as u32);
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::driver::{Options, Session, Stage, compile};

    fn codes(src: &str) -> Vec<Code> {
        let mut s = Session::new();
        let id = s.add("t.tq", &format!("#unit classical\n{src}"));
        let out = compile(
            &mut s,
            id,
            Options {
                stage: Stage::Check,
                ..Options::default()
            },
        );
        out.diagnostics.iter().map(|d| d.code).collect()
    }

    #[test]
    fn question_mark_returns_early_with_the_same_kind() {
        assert_eq!(codes("fn f(x: i64?) -> i64? { Some(x? + 1) }"), []);
        assert_eq!(
            codes("fn g(r: Result<i64, bool>) -> Result<i64, bool> { let v = r?; Ok(v * 2) }"),
            []
        );
    }

    #[test]
    fn question_mark_needs_a_function_that_can_return_early() {
        assert_eq!(codes("fn f(x: i64?) -> i64 { x? }"), [Code::Es06]);
        assert_eq!(codes("fn f(x: i64?) -> Result<i64, bool> { Ok(x?) }"), [Code::Es06]);
        assert_eq!(codes("fn f(r: Result<i64, u8>) -> Result<i64, bool> { Ok(r?) }"), [Code::Es06]);
        // an error enumeration with one variant for the error takes it
        let wrap = "enum E { Other, Low(u8) }\n";
        assert_eq!(codes(&format!("{wrap}fn f(r: Result<i64, u8>) -> Result<i64, E> {{ Ok(r?) }}")), []);
        let two = "enum E { A(u8), B(u8) }\n";
        assert_eq!(codes(&format!("{two}fn f(r: Result<i64, u8>) -> Result<i64, E> {{ Ok(r?) }}")), [Code::Es06]);
        assert_eq!(codes("fn f(x: i64) -> i64? { Some(x?) }"), [Code::Es06]);
    }
}
