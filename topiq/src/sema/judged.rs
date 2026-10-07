//! The macros that ask of judgements: `@judgment`, `@kernel`,
//! `@certificate`, `@fragment_of`, `@matrix_of`, `@classify_op` and
//! `@holonomy`, whose values exist once the unit's circuits are generated;
//! and `@classify` and `@bargmann`, which need no circuit and are computed
//! here.
//!
//! A value of the first kind is a placeholder until then
//! ([`crate::tir::Intrinsic::Derived`]). It may be read by an
//! `@static_assert`, which is checked when the value exists, and in an
//! operator's body, as a value known while the circuit is generated, which
//! phase 9 derives before generating any body that reads it; in any other
//! constant it is reported where it is read. The records' types are the
//! `judge` library's, which the unit imports to name them.

use crate::ast::{self, MacroArg};
use crate::diag::{Code, Diagnostic, Limit};

use crate::intern::Symbol;
use crate::quon::ast::QForm;
use crate::quon::eval::Amplitudes;
use crate::span::{Span, Spanned};
use crate::tir::claims::Derived;
use crate::tir::{Arg, Callee, Expr, ExprKind, FnId, Intrinsic, Ty, Value};

use super::body::Checker;
use super::geometry::Kind;
use super::scope::Def;

impl Checker<'_, '_> {
    /// One of the macros that ask of judgements; `None` when `which` is not
    /// one.
    pub(super) fn judgment_macro(&mut self, which: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<Expr> {
        const NAMES: [&str; 9] = [
            "judgment",
            "kernel",
            "certificate",
            "fragment_of",
            "matrix_of",
            "classify_op",
            "holonomy",
            "classify",
            "bargmann",
        ];
        if !NAMES.contains(&which) {
            return None;
        }
        // why a problem gives no value has been reported
        Some(self.judged(which, args, span).unwrap_or_else(|| Checker::error(span)))
    }

    fn judged(&mut self, which: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<Expr> {
        let e = match which {
            "judgment" | "kernel" | "certificate" | "fragment_of" | "matrix_of" => {
                let [a] = args else {
                    return Some(self.macro_error(which, "the name of an operator", span));
                };
                let Some(f) = self.operator_arg(a) else {
                    return Some(self.macro_error(which, "the name of an operator", a.span));
                };
                let (derived, ty) = match which {
                    "judgment" => (Derived::Judgment(f), self.library_type("judge", "Judgment", span)?),
                    "kernel" => (Derived::Kernel(f), self.library_type("judge", "Kernel", span)?),
                    "certificate" => (Derived::Certificate(f), self.library_type("judge", "Certificate", span)?),
                    "fragment_of" => (Derived::FragmentOf(f), self.library_type("judge", "FragmentInfo", span)?),
                    _ => (Derived::MatrixOf(f), self.matrix_type(f, span)?),
                };
                self.derived(derived, ty, span)
            }
            "classify_op" => {
                let [a, b] = args else {
                    return Some(self.macro_error(which, "an operator and a base type", span));
                };
                let Some(f) = self.operator_arg(a) else {
                    return Some(self.macro_error(which, "the name of an operator, first", a.span));
                };
                let Some(name) = arg_name(b) else {
                    return Some(self.macro_error(which, "the name of a base type, second", b.span));
                };
                let base = self.cx.named_geometry(Spanned::new(name, b.span), Kind::Base)?;
                let ty = self.library_type("judge", "Judgment", span)?;
                self.derived(Derived::ClassifyOp(f, base), ty, span)
            }
            "holonomy" => {
                let [c, s] = args else {
                    return Some(self.macro_error(which, "a chain and a point of its first base type", span));
                };
                let Some(name) = arg_name(c) else {
                    return Some(self.macro_error(which, "the name of a chain, first", c.span));
                };
                let chain = self.cx.named_geometry(Spanned::new(name, c.span), Kind::Chain)?;
                let point = self.point_arg(chain, s)?;
                let ty = self.phase_type(span)?;
                self.derived(Derived::Holonomy(chain, point), ty, span)
            }
            "classify" => {
                let [a] = args else {
                    return Some(self.macro_error(which, "a state", span));
                };
                let state = self.state_arg(a)?;
                if !state.is_normalized() {
                    self.report(super::quon::not_normalized(&state, a.span));
                    return Some(Checker::error(span));
                }
                let ty = self.library_type("quon", "Cover", span)?;
                // the smallest cover holding one state is the point itself
                let point = Value::Enum {
                    variant: 0,
                    fields: vec![super::quon::state_value(&state)],
                };
                Expr::constant(point, ty, span)
            }
            "bargmann" => {
                let [a, b, c] = args else {
                    return Some(self.macro_error(which, "three states", span));
                };
                let (a, b, c) = (self.state_arg(a)?, self.state_arg(b)?, self.state_arg(c)?);
                let inner = crate::judge::linalg::inner;
                let z = inner(&a, &b).mul(&inner(&b, &c)).mul(&inner(&c, &a));
                let n = self.cx.conductor;
                let Some(m) = z.arg() else {
                    let d = if z.is_zero() {
                        Diagnostic::new(Code::Es06)
                            .with_message("the Bargmann invariant of these states has no argument: two of them are orthogonal")
                            .at(span)
                            .with_note("the invariant is the argument of ⟨a|b⟩⟨b|c⟩⟨c|a⟩, which is zero here")
                    } else {
                        let suffices = [16u32, 24].into_iter().filter(|&m| m.is_multiple_of(n) && m != n).find(|&m| {
                            z.embed(m).and_then(|w| w.arg()).is_some()
                        });
                        let help = match suffices {
                            Some(m) => format!("add `#pragma conductor({m})` to the unit"),
                            None => "the invariant is no angle of any conforming conductor".to_owned(),
                        };
                        Diagnostic::new(Code::Ej04)
                            .with_message(format!(
                                "the Bargmann invariant of these states is no whole number of {n}-th parts of a turn"
                            ))
                            .at(span)
                            .with_help(help)
                    };
                    self.report(d);
                    return Some(Checker::error(span));
                };
                let ty = self.phase_type(span)?;
                Expr::constant(super::exact::phase_value(u64::from(m)), ty, span)
            }
            _ => return None,
        };
        Some(e)
    }

    fn derived(&mut self, d: Derived, ty: Ty, span: Span) -> Expr {
        let k = self.cx.unit.derived.len() as u32;
        self.cx.unit.derived.push(d);
        Expr::intrinsic(Intrinsic::Derived(k), Vec::new(), ty, span)
    }

    fn macro_error(&mut self, which: &str, wants: &str, span: Span) -> Expr {
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("`@{which}` takes {wants}"))
                .at(span),
        );
        Checker::error(span)
    }

    /// The operator a macro argument names.
    fn operator_arg(&mut self, a: &Spanned<MacroArg>) -> Option<FnId> {
        if let Some(name) = arg_name(a) {
            return match self.cx.lookup_value(name) {
                Some(Def::Fn(f)) => Some(f),
                _ => None,
            };
        }
        // an operator of an imported unit, named by its path, which reads as
        // a type's as well as a value's
        let e = match &a.node {
            MacroArg::Expr(e) if matches!(e.node, ast::Expr::Path { .. }) => e.clone(),
            MacroArg::Type(t) => match &t.node {
                ast::Type::Path { path, args } if args.is_empty() => Spanned::new(
                    ast::Expr::Path {
                        path: path.clone(),
                        args: Vec::new(),
                    },
                    t.span,
                ),
                _ => return None,
            },
            _ => return None,
        };
        match self.expr(&e).kind {
            ExprKind::FnRef(Callee::Fn(f)) => Some(f),
            _ => None,
        }
    }

    /// A state a macro argument writes in QUON, or names as a constant.
    fn state_arg(&mut self, a: &Spanned<MacroArg>) -> Option<Amplitudes> {
        match &a.node {
            MacroArg::Quon(q) => match q.as_ref() {
                QForm::State(s) => self.cx.written_state(s),
                _ => {
                    self.macro_error("…", "a state", a.span);
                    None
                }
            },
            MacroArg::Expr(e) => self.cx.constant_state(e),
            _ => {
                self.macro_error("…", "a state", a.span);
                None
            }
        }
    }

    /// The point of the chain's first cover a macro argument names: a state,
    /// or an index.
    fn point_arg(&mut self, chain: usize, a: &Spanned<MacroArg>) -> Option<usize> {
        if let MacroArg::Expr(e) = &a.node {
            return self.cx.chain_point(chain, e);
        }
        let state = self.state_arg(a)?;
        let first = self.cx.unit.geometry.chains[chain].def.stages.first()?.from;
        let cover = self.cx.unit.geometry.covers[self.cx.unit.geometry.bases[first].def.cover].def.clone();
        let points = cover.points().unwrap_or_default();
        self.cx.point_index(&points, &state, a.span)
    }

    /// The library type `unit::name`, which the unit imports.
    pub(super) fn library_type(&mut self, unit: &str, name: &str, span: Span) -> Option<Ty> {
        let seg = |s: &str| Spanned::new(self.interner.intern_late(s), span);
        let t = Spanned::new(
            ast::Type::Path {
                path: ast::Path {
                    segments: vec![seg(unit), seg(name)],
                },
                args: Vec::new(),
            },
            span,
        );
        match super::types::resolve(self.cx, &t) {
            Ok(r) => Some(r.ty),
            Err(_) => {
                self.report(
                    Diagnostic::new(Code::Es04)
                        .with_message(format!("this gives a `{unit}::{name}`, and the unit does not import `{unit}`"))
                        .at(span)
                        .with_help(format!("add `import {unit};` to the unit")),
                );
                None
            }
        }
    }

    /// `phase<N>` for the unit's conductor.
    fn phase_type(&mut self, span: Span) -> Option<Ty> {
        let key = self.cx.core_generic(self.interner.get("phase")?)?;
        let n = self.cx.conductor;
        self.cx.instantiate_adt(key, vec![Arg::Const(i128::from(n))], span).map(Ty::Adt)
    }

    /// `mat<cyclo<N>, 2^k, 2^k>` for an operator on `k` qubits.
    fn matrix_type(&mut self, f: FnId, span: Span) -> Option<Ty> {
        self.cx.fn_sig(f);
        let params: Vec<Ty> = self.cx.unit.func(f).param_types().collect();
        let types = &self.cx.unit.types;
        let mut k: u32 = 0;
        for t in params {
            let t = types.as_ref(t).map_or(t, |(_, x)| x);
            k += match t {
                Ty::Qubit => 1,
                _ => types.as_array(t).filter(|(e, _)| *e == Ty::Qubit).map_or(0, |(_, n)| n as u32),
            };
        }
        if let Some(d) = Limit::MatrixOfQubits.check(k, span) {
            self.report(d);
            return None;
        }
        let cyclo = self.cyclo_type(u64::from(self.cx.conductor), span)?;
        let key = self.cx.core_generic(self.interner.get("mat")?)?;
        let size = 1i128 << k;
        self.cx
            .instantiate_adt(key, vec![Arg::Type(cyclo), Arg::Const(size), Arg::Const(size)], span)
            .map(Ty::Adt)
    }
}

/// The single name a macro argument is.
fn arg_name(a: &Spanned<MacroArg>) -> Option<Symbol> {
    match &a.node {
        MacroArg::Expr(e) => match &e.node {
            ast::Expr::Path { path, args } if args.is_empty() && path.segments.len() == 1 => Some(path.segments[0].node),
            _ => None,
        },
        MacroArg::Type(t) => match &t.node {
            ast::Type::Path { path, args } if args.is_empty() && path.segments.len() == 1 => Some(path.segments[0].node),
            _ => None,
        },
        _ => None,
    }
}
