//! What the compiler does on the `core` library's behalf.
//!
//! - **`abort(code, message)`** names the line that called it, like any
//!   abort. The line is written when the program is built, so each
//!   identifier's line is prepared there, and a code only known when the
//!   program runs chooses among them.
//! - **`unwrap` and `expect`** on an `Opt` or a `Result` abort with `RA10`
//!   at the line that called them, not at a line inside the library, so they
//!   are written out where they are called.
//! - **The operations the library is built from**, under names beginning
//!   with `__`, which only the library's own units may use: arithmetic that
//!   wraps, clamps or truncates rather than aborting, and the deep copy
//!   `clone` makes, which follows each type's shape.

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{Arm, Expr, ExprKind, Intrinsic, Pat, Ty};

use super::body::Checker;

impl Checker<'_, '_> {
    /// Whether the code being checked is a library unit's, which may use the
    /// operations whose names begin with `__`.
    pub(super) fn in_library(&self) -> bool {
        let unit = self.cx.frame_origin().unwrap_or_else(|| self.cx.unit.name.clone());
        crate::library::source(&unit).is_some()
    }

    /// Whether `t` is the `core` library's enumeration `name`, and if so its
    /// definition.
    pub(super) fn core_enum(&self, t: Ty, name: &str) -> Option<crate::tir::AdtDef> {
        let Ty::Adt(id) = t else { return None };
        let def = self.types().adt(id);
        let in_core = match &def.origin {
            Some(o) => o == "core",
            None => self.cx.unit.name == "core",
        };
        (in_core && self.name(def.name) == name).then(|| def.clone())
    }

    /// `abort(code, message)`.
    pub(super) fn abort_call(&mut self, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let [code, message] = args else {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`abort` takes an identifier and a message, but this call gives {}", args.len()))
                    .at(span)
                    .with_help("write it as `abort(AbortCode::RA10, \"what went wrong\")`"),
            );
            return Checker::error(span);
        };
        let c = self.expr(code);
        let t = self.settled(c.ty);
        let string = self.types_mut().string();
        let m = self.expr(message);
        let m = self.coerce(m, string, "the message of `abort`");
        let Some(def) = self.core_enum(t, "AbortCode") else {
            if t != Ty::Never {
                let what = self.describe(t);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`abort` takes an `AbortCode` first, but this is {what}"))
                        .at(code.span)
                        .with_help("name the identifier, as in `AbortCode::RA06`"),
                );
            }
            return Checker::error(span);
        };
        // the number an identifier's variant stands for: `RA06` is 6
        let numbers: Vec<u8> = def
            .variants()
            .iter()
            .map(|v| self.name(v.name)[2..].parse().unwrap_or(10))
            .collect();
        let aborting = |n: u8, msg: Expr| Expr::intrinsic(Intrinsic::Abort(n), vec![msg], Ty::Never, span);
        // named where it is written: one line to prepare
        if let ExprKind::Variant { variant, .. } = c.kind {
            return aborting(numbers[variant as usize], m);
        }
        // chosen when the program runs: one line for each, and the message
        // evaluated once, before
        let held = self.new_local(crate::intern::Symbol::EMPTY, string, false, span);
        let msg = || Expr::local(held, string, span);
        let arms = numbers
            .iter()
            .enumerate()
            .map(|(i, &n)| Arm {
                pat: Pat::variant(i as u32, Vec::new(), t, span),
                body: aborting(n, msg()),
            })
            .collect();
        let chosen = Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(c),
                arms,
            },
            ty: Ty::Never,
            span,
        };
        Expr::let_then(held, m, chosen, Ty::Never, span)
    }

    /// `x.unwrap()` or `x.expect(message)` on an `Opt` or a `Result`, if
    /// that is what this is: the value inside, or an abort with `RA10` at
    /// this line.
    pub(super) fn unwrap_call(&mut self, recv: &Expr, name: &str, args: &[Spanned<ast::Expr>], span: Span) -> Option<Expr> {
        if !matches!(name, "unwrap" | "expect") {
            return None;
        }
        let t = self.settled(recv.ty);
        let (def, hit, miss, what) = if let Some(d) = self.core_enum(t, "Opt") {
            (d, "Some", "None", "`None`")
        } else if let Some(d) = self.core_enum(t, "Result") {
            (d, "Ok", "Err", "an `Err`")
        } else {
            return None;
        };
        let _ = what;
        let string = self.types_mut().string();
        // `unwrap` reports the identifier's own words; `expect`, the caller's
        let message: Vec<Expr> = match (name, args) {
            ("unwrap", []) => Vec::new(),
            ("expect", [m]) => {
                let m = self.expr(m);
                vec![self.coerce(m, string, "the message of `expect`")]
            }
            _ => {
                self.check_only(args);
                let wants = if name == "unwrap" { "no arguments" } else { "one message" };
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("`{name}` takes {wants}, but this call gives {}", args.len()))
                        .at(span),
                );
                return Some(Checker::error(span));
            }
        };
        let index = |n: &str| {
            def.variants()
                .iter()
                .position(|v| self.name(v.name) == n)
                .expect("the variant is declared") as u32
        };
        let (hit_v, miss_v) = (index(hit), index(miss));
        let inner = def.variants()[hit_v as usize].fields[0].ty;
        let value = self.new_local(def.variants()[hit_v as usize].name, inner, false, span);
        let miss_fields = if miss == "Err" {
            let err = def.variants()[miss_v as usize].fields[0].ty;
            vec![(0, Pat::wild(err, span))]
        } else {
            Vec::new()
        };
        let arms = vec![
            Arm {
                pat: Pat::variant(hit_v, vec![(0, Pat::bind(value, inner, span))], t, span),
                body: Expr::local(value, inner, span),
            },
            Arm {
                pat: Pat::variant(miss_v, miss_fields, t, span),
                body: Expr::intrinsic(Intrinsic::Abort(10), message, Ty::Never, span),
            },
        ];
        Some(Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(recv.clone()),
                arms,
            },
            ty: inner,
            span,
        })
    }

    /// One of the operations the library is built from.
    pub(super) fn library_operation(&mut self, which: Intrinsic, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
        if which == Intrinsic::Clone {
            return self.clone_call(checked, span);
        }
        if which == Intrinsic::Exchange {
            return self.exchange_call(checked, span);
        }
        if which == Intrinsic::Adjoint {
            return self.adjoint_call(checked, span);
        }
        if which == Intrinsic::Controlled {
            return self.functor(which, checked, span);
        }
        if matches!(which, Intrinsic::Gate(_) | Intrinsic::Apply) {
            return self.quantum_operation(which, checked, span);
        }
        if which.is_dyn() {
            return self.dyn_operation(which, checked, span);
        }
        let arity = match which {
            Intrinsic::Overflowing(_) | Intrinsic::Saturating(_) => 2,
            _ => 1,
        };
        if checked.len() != arity {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`{}` takes {arity} operands", which.name()))
                    .at(span),
            );
            return Checker::error(span);
        }
        let t = if arity == 2 {
            match self.unify(checked[0].ty, checked[1].ty) {
                Ok(t) => t,
                Err(_) => {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{}` takes two integers of one type", which.name()))
                            .at(span),
                    );
                    return Checker::error(span);
                }
            }
        } else {
            checked[0].ty
        };
        let t = self.settled(t);
        let numeric = match which {
            Intrinsic::SaturatingAs => matches!(t, Ty::Int(_) | Ty::Float(_)),
            _ => matches!(t, Ty::Int(_)),
        };
        if !numeric && t != Ty::Never {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{}` works on integers, but this is {what}", which.name()))
                    .at(span),
            );
            return Checker::error(span);
        }
        let ty = match which {
            Intrinsic::Overflowing(_) | Intrinsic::OverflowingNeg => self.types_mut().tuple(vec![t, Ty::Bool]),
            Intrinsic::Saturating(_) => t,
            // the type converted to is whatever the context says
            _ => self.infer.fresh_any(),
        };
        Expr::intrinsic(which, checked, ty, span)
    }
}

impl Checker<'_, '_> {
    /// `adjoint(x)`. Of an operator it is the operator that undoes it, which
    /// only a quantum unit has. Of a value it is the value's `$adj`, given
    /// `x` as a receiver is, so `x` may be a value or a reference to one: a
    /// matrix's conjugate transpose, a vector's covector, a complex number's
    /// conjugate. A real number is its own adjoint.
    fn adjoint_call(&mut self, checked: Vec<Expr>, span: Span) -> Expr {
        let Ok([x]) = <[Expr; 1]>::try_from(checked) else {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`adjoint` takes one operand")
                    .at(span)
                    .with_help("write `adjoint(x)`"),
            );
            return Checker::error(span);
        };
        let t = self.settled(x.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        if matches!(t, Ty::Fn(_)) {
            if !self.cx.quantum {
                return self.quantum_in_classical("take the adjoint of an operator", span);
            }
            return self.functor(Intrinsic::Adjoint, vec![x], span);
        }
        let target = match self.types().as_ref(t) {
            Some((_, u)) => self.settled(u),
            None => t,
        };
        let real = matches!(target, Ty::Int(_) | Ty::Float(_)) || self.infer.is_integer_variable(target);
        if real {
            return if target == t { x } else { Expr::deref(x, target, span) };
        }
        if let Some((entry, label)) = self.operator_method(target, "adj") {
            return self.call_method(entry, Some(x), &[], Vec::new(), &label, span);
        }
        let what = self.describe(target);
        let mut d = Diagnostic::new(Code::Es06)
            .with_message(format!("`adjoint` takes an operator, or a value whose type defines `$adj`; not {what}"))
            .at(x.span)
            .with_note(
                "the adjoint of a value is its type's `$adj`: a matrix's conjugate transpose, a \
                 vector's covector, a complex number's conjugate",
            );
        if matches!(target, Ty::Circuit(_)) {
            d = d.with_help(
                "a circuit handle's adjoint is `circuit::adjoint(c)`, which gives a `Result`: a \
                 circuit that measures has none, and that is only known when the program runs",
            );
        } else if matches!(target, Ty::Adt(_)) {
            let ty = self.show(target);
            d = d.with_help(format!("declare `fn $adj(self: *const {ty}) -> …` in `impl {ty} {{ … }}`"));
        }
        self.report(d);
        Checker::error(span)
    }

    /// `__clone(r)`, which `core`'s `clone` is: a deep copy of what `r`
    /// refers to. A value holding a closure anywhere cannot be cloned, since
    /// what a closure's environment holds is not known; the part that is one
    /// is named.
    /// `__exchange(a, b)`: two `*T` of one `T`, whose values change places.
    fn exchange_call(&mut self, checked: Vec<Expr>, span: Span) -> Expr {
        if checked.len() != 2 {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`__exchange` takes two references to values of one type")
                    .at(span),
            );
            return Checker::error(span);
        }
        let (a, b) = (self.settled(checked[0].ty), self.settled(checked[1].ty));
        if a == Ty::Never || b == Ty::Never {
            return Checker::error(span);
        }
        let target = |c: &Self, t: Ty| match c.types().as_ref(t) {
            Some((crate::tir::Access::Write, x)) => Some(x),
            _ => None,
        };
        match (target(self, a), target(self, b)) {
            (Some(x), Some(y)) if x == y => Expr::intrinsic(Intrinsic::Exchange, checked, Ty::Void, span),
            _ => {
                let (ta, tb) = (self.show(a), self.show(b));
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "`__exchange` takes two `*T` of one `T`, but these are `{ta}` and `{tb}`"
                        ))
                        .at(span),
                );
                Checker::error(span)
            }
        }
    }

    fn clone_call(&mut self, mut checked: Vec<Expr>, span: Span) -> Expr {
        let (Some(r), true) = (checked.pop(), checked.is_empty()) else {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`clone` takes one argument, a reference to what it copies")
                    .at(span),
            );
            return Checker::error(span);
        };
        let t = self.settled(r.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        // a slice views a growable array's elements, and its copy is one
        let viewed = self.types().as_slice(t).map(|(_, elem)| elem);
        let target = match viewed {
            Some(elem) => Some(self.types_mut().growable(elem)),
            None => self.types().as_ref(t).map(|(_, x)| x),
        };
        let Some(target) = target else {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`clone` takes a reference to what it copies, but this is {what}"))
                    .at(r.span)
                    .with_help("lend the value, as in `clone(&v)`"),
            );
            return Checker::error(span);
        };
        if self.types().is_quantum(target) {
            let ty = self.show(target);
            self.report(
                Diagnostic::new(Code::Ec09)
                    .with_message(format!("`clone` cannot copy `{ty}`: it holds qubits"))
                    .at(r.span)
                    .with_note("an unknown quantum state cannot be copied, so no type holding a qubit is copyable")
                    .with_help(
                        "where this operator prepared it, `qcopy(&value)` prepares it again; otherwise \
                         `replay` a monic operator that made it",
                    ),
            );
            return Checker::error(span);
        }
        if let Some((part, opaque)) = self.closure_within(target, String::new(), &mut Vec::new()) {
            let ty = self.show(target);
            let what = if opaque { "marked `[tcon: opaque]`" } else { "a closure" };
            let message = if part.is_empty() {
                format!("`clone` cannot copy `{ty}`: it is {what}")
            } else {
                format!("`clone` cannot copy a `{ty}`: its part `{part}` is {what}")
            };
            let d = Diagnostic::new(Code::Ec09).with_message(message).at(r.span);
            let d = if opaque {
                d.with_note("a type marked `[tcon: opaque]` has asked not to be copied part by part")
                    .with_help("give the type a `$copy`, which `clone` uses instead, if a copy of it makes sense")
            } else {
                d.with_note("what a closure's environment holds is not known, so there is no telling how to copy it")
            };
            self.report(d);
            return Checker::error(span);
        }
        Expr::intrinsic(Intrinsic::Clone, vec![r], target, span)
    }

    /// The path to a part of a value of `ty` that `clone` cannot copy,
    /// written from `at`, if there is one, and whether it is a type marked
    /// `[tcon: opaque]` rather than a closure. A type with `$copy` is copied
    /// by it, so its parts are not looked into.
    fn closure_within(&mut self, ty: Ty, at: String, seen: &mut Vec<crate::tir::AdtId>) -> Option<(String, bool)> {
        let join = |at: &str, part: &str| if at.is_empty() { part.to_owned() } else { format!("{at}.{part}") };
        match ty {
            Ty::Closure(_) => Some((at, false)),
            Ty::Adt(id) => {
                if self.types().defines_copy(id) || seen.contains(&id) {
                    return None;
                }
                if self.adt(id).opaque {
                    return Some((at, true));
                }
                seen.push(id);
                let def = self.adt(id).clone();
                let found = match &def.kind {
                    crate::tir::AdtKind::Struct { fields } => fields
                        .iter()
                        .find_map(|f| self.closure_within(f.ty, join(&at, self.name(f.name)), seen)),
                    crate::tir::AdtKind::Enum { variants } => variants.iter().find_map(|v| {
                        v.fields.iter().enumerate().find_map(|(i, f)| {
                            let field = match v.shape {
                                crate::tir::VariantShape::Struct => self.name(f.name).to_owned(),
                                _ => i.to_string(),
                            };
                            let part = format!("{}({field})", self.name(v.name));
                            self.closure_within(f.ty, join(&at, &part), seen)
                        })
                    }),
                };
                seen.pop();
                found
            }
            Ty::Tuple(_) => {
                let elems = self.types().as_tuple(ty).map(<[Ty]>::to_vec).unwrap_or_default();
                elems
                    .into_iter()
                    .enumerate()
                    .find_map(|(i, e)| self.closure_within(e, join(&at, &i.to_string()), seen))
            }
            Ty::Array(_) | Ty::Growable(_) => {
                let elem = self
                    .types()
                    .as_array(ty)
                    .map(|(e, _)| e)
                    .or_else(|| self.types().as_growable(ty))?;
                self.closure_within(elem, format!("{at}[…]"), seen)
            }
            _ => None,
        }
    }
}

impl Checker<'_, '_> {
    /// The operations the `dyn` library is built from.
    fn dyn_operation(&mut self, which: Intrinsic, checked: Vec<Expr>, span: Span) -> Expr {
        let arity = match which {
            Intrinsic::FieldAt | Intrinsic::DynCall => 3,
            Intrinsic::DynStore => 2,
            _ => 1,
        };
        if checked.len() != arity {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`{}` takes {arity} operands", which.name()))
                    .at(span),
            );
            return Checker::error(span);
        }
        let mut args = checked;
        let any = self.types_mut().reference(crate::tir::Access::Write, Ty::Void);
        let info = match self.cx.typeinfo_types(span) {
            Some(t) => self.types_mut().reference(crate::tir::Access::Const, Ty::Adt(t.info)),
            None => return Checker::error(span),
        };
        let (which, ty) = match which {
            Intrinsic::DynOf(_) => {
                let t = self.settled(args[0].ty);
                if t == Ty::Never || !self.cx.need_table(t, span) {
                    return Checker::error(span);
                }
                (Intrinsic::DynOf(t), Ty::Dyn)
            }
            Intrinsic::DynTake => {
                let a = args.remove(0);
                args.push(self.coerce(a, Ty::Dyn, "the value taken apart"));
                (which, self.infer.fresh_any())
            }
            Intrinsic::DynView => {
                let t = self.settled(args[0].ty);
                match self.types().as_ref(t) {
                    Some((access, Ty::Dyn)) => (which, self.types_mut().reference(access, Ty::Void)),
                    _ => {
                        let what = self.describe(t);
                        self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!("`__dyn_view` takes a reference to a `dyn`, but this is {what}"))
                                .at(span),
                        );
                        return Checker::error(span);
                    }
                }
            }
            Intrinsic::FieldAt => {
                let p = self.settled(args[0].ty);
                if !matches!(self.types().as_ref(p), Some((_, Ty::Void))) {
                    let what = self.describe(p);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`__field_at` takes a `*any` first, but this is {what}"))
                            .at(span),
                    );
                    return Checker::error(span);
                }
                let offset = args.remove(1);
                args.insert(1, self.coerce(offset, Ty::USIZE, "the offset"));
                let t = args.remove(2);
                args.push(self.coerce(t, info, "the type of the part"));
                (which, p)
            }
            Intrinsic::DynCall => {
                let slice_of_dyn = self.types_mut().slice(crate::tir::Access::Write, Ty::Dyn);
                let [call, this, rest] = <[Expr; 3]>::try_from(args).expect("three operands");
                args = vec![
                    self.coerce(call, any, "the method's `call` entry"),
                    self.coerce(this, any, "the receiver"),
                    self.coerce(rest, slice_of_dyn, "the arguments"),
                ];
                (which, Ty::Dyn)
            }
            Intrinsic::DynStore => {
                let place = self.types_mut().reference(crate::tir::Access::Write, Ty::Void);
                let [p, v] = <[Expr; 2]>::try_from(args).expect("two operands");
                args = vec![
                    self.coerce(p, place, "the place stored to"),
                    self.coerce(v, Ty::Dyn, "the value stored"),
                ];
                (which, Ty::Void)
            }
            Intrinsic::IsNull => {
                let r = args.remove(0);
                args.push(self.coerce(r, any, "the address"));
                (which, Ty::Bool)
            }
            Intrinsic::CircuitDoc => {
                let t = self.settled(args[0].ty);
                if !matches!(t, Ty::Circuit(_)) {
                    let what = self.describe(t);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`__circuit_doc` takes a circuit handle, but this is {what}"))
                            .at(span),
                    );
                    return Checker::error(span);
                }
                (which, self.types_mut().string())
            }
            // the handle's signature is the one the context wants
            Intrinsic::CircuitOf => {
                let text = self.types_mut().string();
                let d = args.remove(0);
                args.push(self.coerce(d, text, "the circuit's document"));
                (which, self.infer.fresh_any())
            }
            _ => {
                let a = args.remove(0);
                args.push(self.coerce(a, info, "the type to make"));
                (which, Ty::Dyn)
            }
        };
        Expr::intrinsic(which, args, ty, span)
    }
}
