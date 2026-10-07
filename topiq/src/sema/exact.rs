//! The exact scalars during translation.
//!
//! `frac`, `cyclo<N>` and `phase<N>` are `core` types, whose arithmetic is
//! Topiq code when the program runs. During translation the same values are
//! computed here, with [`crate::exact`], and held in the layout `core` gives
//! them, so a constant computed now and a value computed later are the same
//! bytes:
//!
//! - a `frac` is a structure of two integers, numerator and denominator;
//! - an integer is a structure of a sign and a growable array of base-2^32
//!   digits, least significant first, with no zero digit at the top;
//! - a `cyclo<N>` is a structure holding a growable array of `frac`s, its
//!   coefficients on the powers of ζ;
//! - a `phase<N>` is a structure holding its index as a `usize`.
//!
//! This module also gives the names the language provides for exact
//! constants: `i`, `isq2` and `w(k, N)`, each a `cyclo` of the unit's
//! conductor for the first two, of `N` for the third.
//!
//! It holds the rest of what the exact scalars and linear algebra need of
//! analysis:
//!
//! - **Literals.** A number written where a `frac` or `cyclo` is wanted is
//!   that exact number, as is arithmetic over such numbers: `1/3` is one
//!   third, `0.1` one tenth.
//! - **Conversions.** `as f64`, `as f32` and `as cplx<f64>` approximate; an
//!   integer converts to `frac`, `cyclo<N>` and `cplx<T>`.
//! - **Widening.** A `cyclo<M>` becomes a `cyclo<N>` where one is wanted
//!   when M divides N, and in an operator over both the smaller order widens;
//!   neither dividing the other is reported.
//! - **Vectors, covectors and matrices.** An array of exact numbers written
//!   with no type is a `vec` or `mat`. `*` of a matrix or a covector and a
//!   vector is `apply`, of a vector and a covector `outer`, and with a number
//!   `scale`; a covector times a matrix is the covector's `$mul`, as two
//!   matrices are the left one's. Kinds, shapes or elements that do not fit
//!   an operator are reported before any method is chosen, and a product of
//!   two vectors names the inner and outer products `adjoint` writes.

use num_bigint::{BigInt, Sign};

use crate::diag::{Code, Diagnostic};
use crate::exact::{Cyclo, Frac};
use crate::span::Span;
use crate::tir::{Arg, BinOp, Expr, ExprKind, IntTy, Ty, Value};

use super::body::Checker;

/// An integer as `core` holds one.
fn int_value(n: &BigInt) -> Value {
    let (sign, digits) = n.to_u32_digits();
    Value::Struct(vec![
        Value::Bool(sign == Sign::Minus),
        Value::Array(digits.iter().map(|&d| Value::Int(i128::from(d), IntTy::U32)).collect()),
    ])
}

fn int_of(v: &Value) -> Option<BigInt> {
    let Value::Struct(parts) = v else { return None };
    let [Value::Bool(neg), Value::Array(digits)] = parts.as_slice() else {
        return None;
    };
    let mut out = Vec::with_capacity(digits.len());
    for d in digits {
        out.push(u32::try_from(d.as_int()?).ok()?);
    }
    let sign = if *neg { Sign::Minus } else { Sign::Plus };
    Some(BigInt::from_slice(sign, &out))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A `frac` as `core` holds one.
pub fn frac_value(f: &Frac) -> Value {
    Value::Struct(vec![int_value(f.numer()), int_value(f.denom())])
}

/// The `frac` a value `core` holds is.
pub fn frac_of(v: &Value) -> Option<Frac> {
    let Value::Struct(parts) = v else { return None };
    let [n, d] = parts.as_slice() else { return None };
    Frac::checked_new(int_of(n)?, int_of(d)?)
}

/// A `cyclo<N>` as `core` holds one.
pub fn cyclo_value(c: &Cyclo) -> Value {
    Value::Struct(vec![Value::Array(c.coeffs().iter().map(frac_value).collect())])
}

/// The `cyclo<n>` a value `core` holds is.
pub fn cyclo_of(v: &Value, n: u32) -> Option<Cyclo> {
    let Value::Struct(parts) = v else { return None };
    let [Value::Array(coeffs)] = parts.as_slice() else {
        return None;
    };
    let coeffs = coeffs.iter().map(frac_of).collect::<Option<Vec<_>>>()?;
    Some(Cyclo::from_coeffs(n, coeffs))
}

/// A `phase<N>` as `core` holds one.
pub fn phase_value(m: u64) -> Value {
    Value::Struct(vec![Value::Int(i128::from(m), IntTy::USIZE)])
}

/// The index of the `phase` a value `core` holds is.
pub fn phase_of(v: &Value) -> Option<u64> {
    let Value::Struct(parts) = v else { return None };
    let [m] = parts.as_slice() else { return None };
    u64::try_from(m.as_int()?).ok()
}

/// Which of `core`'s linear algebra types a type is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Linear {
    /// `vec<T, N>`, a column: a ket.
    Vec,
    /// `covec<T, N>`, a row: a bra.
    Covec,
    /// `mat<T, R, C>`.
    Mat,
}

/// One of `core`'s operations on exact scalars, which constant evaluation
/// computes itself rather than running its code.
#[derive(Clone, Debug)]
pub struct ExactOp {
    on: On,
    /// The operation's name.
    name: String,
}

#[derive(Clone, Copy, Debug)]
enum On {
    Frac,
    Cyclo(u32),
    Phase(u32),
    /// `__cyclo_widen<M, N>`.
    Widen(u32, u32),
}

impl ExactOp {
    /// Whether this is `$copy`, which gives back what it is given.
    pub fn is_copy(&self) -> bool {
        self.name == "$copy"
    }
}

/// Why an exact operation was not computed.
#[derive(Clone, Copy, Debug)]
pub enum ExactError {
    /// It is not one computed during translation.
    Unsupported,
    /// Computing it aborts, as it would when the program runs.
    Abort(super::arith::Abort),
}

/// The exact operation a call of `callee` is.
pub fn exact_op(unit: &crate::tir::Unit, interner: &crate::intern::Interner, callee: crate::tir::Callee) -> Option<ExactOp> {
    use crate::tir::Callee;
    let (origin, name, method, args) = match callee {
        Callee::Fn(id) => {
            let f = unit.func(id);
            let origin = f.origin.clone().unwrap_or_else(|| unit.name.clone());
            (origin, f.name, f.method, f.args.clone())
        }
        Callee::Extern(id) => {
            let x = unit.extern_fn(id);
            (x.unit.clone(), x.name, x.method, Vec::new())
        }
    };
    if origin != "core" {
        return None;
    }
    let text = interner.resolve(name);
    let Some(m) = method else {
        return match (text, args.as_slice()) {
            ("__cyclo_widen", [Arg::Const(a), Arg::Const(b)]) => Some(ExactOp {
                on: On::Widen(u32::try_from(*a).ok()?, u32::try_from(*b).ok()?),
                name: text.to_owned(),
            }),
            _ => None,
        };
    };
    let def = unit.types.adt(m.owner);
    let n = match def.args.first() {
        Some(Arg::Const(n)) => u32::try_from(*n).ok(),
        _ => None,
    };
    let on = match interner.resolve(def.name) {
        "frac" => On::Frac,
        "cyclo" => On::Cyclo(n?),
        "phase" => On::Phase(n?),
        _ => return None,
    };
    let name = if m.operator { format!("${text}") } else { text.to_owned() };
    Some(ExactOp { on, name })
}

fn int_arg(v: &Value) -> Result<i128, ExactError> {
    v.as_int().ok_or(ExactError::Unsupported)
}

fn frac_arg(v: &Value) -> Result<Frac, ExactError> {
    frac_of(v).ok_or(ExactError::Unsupported)
}

fn cyclo_arg(v: &Value, n: u32) -> Result<Cyclo, ExactError> {
    cyclo_of(v, n).ok_or(ExactError::Unsupported)
}

/// `Ord`'s variant for a comparison: `Lt`, `Eq` or `Gt`, in that order.
fn ord_value(o: std::cmp::Ordering) -> Value {
    let variant = match o {
        std::cmp::Ordering::Less => 0,
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Greater => 2,
    };
    Value::Enum {
        variant,
        fields: Vec::new(),
    }
}

/// An `Opt` as `core` lays one out: `Some` its first variant, `None` its
/// second.
fn opt_value(v: Option<Value>) -> Value {
    match v {
        Some(x) => Value::Enum { variant: 0, fields: vec![x] },
        None => Value::Enum { variant: 1, fields: Vec::new() },
    }
}

/// What `op` computes from these arguments, each a value in `core`'s layout
/// or an integer.
///
/// # Errors
///
/// [`ExactError::Abort`] where the program would abort, and
/// [`ExactError::Unsupported`] for an operation left to the running program.
pub fn eval_exact(op: &ExactOp, args: &[Value]) -> Result<Value, ExactError> {
    use super::arith::Abort;
    let divide_by_zero = ExactError::Abort(Abort::DivideByZero);
    match op.on {
        On::Frac => {
            let f = |i: usize| frac_arg(args.get(i).ok_or(ExactError::Unsupported)?);
            let v = match (op.name.as_str(), args.len()) {
                ("of", 2) => {
                    let (n, d) = (int_arg(&args[0])?, int_arg(&args[1])?);
                    if d == 0 {
                        return Err(ExactError::Abort(Abort::Called(2)));
                    }
                    Frac::checked_new(BigInt::from(n), BigInt::from(d)).ok_or(divide_by_zero)?
                }
                ("from", 1) => Frac::from_bigint(BigInt::from(int_arg(&args[0])?)),
                ("$copy" | "$adj", 1) => f(0)?,
                ("$add", 2) => f(0)? + f(1)?,
                ("$sub", 2) => f(0)? - f(1)?,
                ("$mul", 2) => f(0)? * f(1)?,
                ("$div", 2) => {
                    let d = f(1)?;
                    if d.is_zero() {
                        return Err(ExactError::Abort(Abort::Called(2)));
                    }
                    f(0)? / d
                }
                ("$neg", 1) => -f(0)?,
                ("$eq", 2) => return Ok(Value::Bool(f(0)? == f(1)?)),
                ("$ord", 2) => return Ok(ord_value(f(0)?.cmp(&f(1)?))),
                ("is_zero", 1) => return Ok(Value::Bool(f(0)?.is_zero())),
                _ => return Err(ExactError::Unsupported),
            };
            Ok(frac_value(&v))
        }
        On::Cyclo(n) => {
            let c = |i: usize| cyclo_arg(args.get(i).ok_or(ExactError::Unsupported)?, n);
            let v = match (op.name.as_str(), args.len()) {
                ("zero", 0) => Cyclo::zero(n),
                ("of", 1) => Cyclo::from_frac(n, frac_arg(&args[0])?),
                ("from", 1) => Cyclo::from_frac(n, Frac::from_bigint(BigInt::from(int_arg(&args[0])?))),
                ("zeta", 1) => Cyclo::zeta_pow(n, i64::try_from(int_arg(&args[0])?).map_err(|_| ExactError::Unsupported)?),
                ("$copy", 1) => c(0)?,
                ("$add", 2) => c(0)?.add(&c(1)?),
                ("$sub", 2) => c(0)?.sub(&c(1)?),
                ("$mul", 2) => c(0)?.mul(&c(1)?),
                ("$div", 2) => c(0)?.div(&c(1)?).ok_or(ExactError::Abort(Abort::Called(2)))?,
                ("$neg", 1) => c(0)?.neg(),
                ("$adj", 1) => c(0)?.conj(),
                ("norm2", 1) => c(0)?.norm_sq(),
                ("$eq", 2) => return Ok(Value::Bool(c(0)? == c(1)?)),
                ("is_zero", 1) => return Ok(Value::Bool(c(0)?.is_zero())),
                ("sign", 1) => return Ok(opt_value(crate::exact::sign::real_sign(&c(0)?).map(ord_value))),
                ("arg", 1) => return Ok(opt_value(c(0)?.arg().map(|m| phase_value(u64::from(m))))),
                _ => return Err(ExactError::Unsupported),
            };
            Ok(cyclo_value(&v))
        }
        On::Phase(n) => {
            let n = u64::from(n);
            let p = |i: usize| phase_of(args.get(i).ok_or(ExactError::Unsupported)?).ok_or(ExactError::Unsupported);
            let m = match (op.name.as_str(), args.len()) {
                ("of", 1) => {
                    let k = int_arg(&args[0])?;
                    if k < 0 || k >= i128::from(n) {
                        return Err(ExactError::Abort(Abort::Called(9)));
                    }
                    k as u64
                }
                ("$copy", 1) => p(0)?,
                ("$add", 2) => (p(0)? + p(1)?) % n,
                ("$sub", 2) => (p(0)? + n - p(1)?) % n,
                ("$neg", 1) => (n - p(0)?) % n,
                ("$eq", 2) => return Ok(Value::Bool(p(0)? == p(1)?)),
                ("$ord", 2) => return Ok(ord_value(p(0)?.cmp(&p(1)?))),
                ("index", 1) => return Ok(Value::Int(i128::from(p(0)?), IntTy::USIZE)),
                _ => return Err(ExactError::Unsupported),
            };
            Ok(phase_value(m))
        }
        On::Widen(m, n) => {
            let x = cyclo_arg(args.first().ok_or(ExactError::Unsupported)?, m)?;
            Ok(cyclo_value(&x.embed(n).ok_or(ExactError::Unsupported)?))
        }
    }
}

impl Checker<'_, '_> {
    /// `core`'s `frac`.
    pub(super) fn frac_type(&mut self) -> Option<Ty> {
        let sym = self.interner.get("frac")?;
        self.cx.core_type(sym).map(Ty::Adt)
    }

    /// `core`'s `cyclo<n>`.
    pub(super) fn cyclo_type(&mut self, n: u64, span: Span) -> Option<Ty> {
        let key = self.cx.core_generic(self.interner.get("cyclo")?)?;
        self.cx.instantiate_adt(key, vec![Arg::Const(i128::from(n))], span).map(Ty::Adt)
    }

    /// `e` checked with no type written for it: an array literal whose
    /// innermost elements are exact numbers is a vector of them, and one of
    /// rows of equal length a matrix, as `[[isq2, isq2], [isq2, -isq2]]` is
    /// a `mat<cyclo<8>, 2, 2>`. Any other expression is checked as it is.
    pub(super) fn expr_inferred(&mut self, e: &crate::span::Spanned<crate::ast::Expr>) -> Expr {
        let x = self.expr(e);
        if !matches!(e.node, crate::ast::Expr::Array(_)) {
            return x;
        }
        let t = self.shallow(x.ty);
        let Some((row, r)) = self.types().as_array(t) else { return x };
        let (name, args) = match self.types().as_array(self.shallow(row)) {
            Some((elem, c)) => ("mat", vec![Arg::Type(elem), Arg::Const(i128::from(r)), Arg::Const(i128::from(c))]),
            None => ("vec", vec![Arg::Type(row), Arg::Const(i128::from(r))]),
        };
        let Some(Arg::Type(elem)) = args.first().copied() else { return x };
        let elem = self.shallow(elem);
        let exact = self.is_exact_number(elem) || self.core_exact(elem, "cplx").is_some();
        if !exact {
            return x;
        }
        let Some(key) = self.interner.get(name).and_then(|s| self.cx.core_generic(s)) else {
            return x;
        };
        let Some(id) = self.cx.instantiate_adt(key, args, e.span) else { return x };
        let span = x.span;
        Expr {
            kind: ExprKind::StructLit { fields: vec![(0, x)] },
            ty: Ty::Adt(id),
            span,
        }
    }

    /// Whether `t` is `core`'s generic structure `name`, and if so its first
    /// constant argument: the `N` of `cyclo<N>` or `phase<N>`.
    pub(super) fn core_exact(&mut self, t: Ty, name: &str) -> Option<u64> {
        let Ty::Adt(id) = self.shallow(t) else { return None };
        let def = self.adt(id).clone();
        let in_core = match &def.origin {
            Some(o) => o == "core",
            None => self.cx.unit.name == "core",
        };
        if !in_core || self.name(def.name) != name {
            return None;
        }
        match def.args.first() {
            Some(Arg::Const(n)) => u64::try_from(*n).ok(),
            _ => Some(0),
        }
    }

    /// A `cyclo<n>` constant.
    pub(super) fn cyclo_constant(&mut self, c: &Cyclo, span: Span) -> Option<Expr> {
        let ty = self.cyclo_type(u64::from(c.conductor()), span)?;
        Some(Expr::constant(cyclo_value(c), ty, span))
    }

    /// `i` or `isq2` written where nothing else has the name: the imaginary
    /// unit, or one over the square root of two, in the unit's field.
    pub(super) fn exact_name(&mut self, name: &str, span: Span) -> Option<Expr> {
        let n = self.cx.conductor;
        let c = match name {
            "i" => Cyclo::imaginary_unit(n)?,
            "isq2" => Cyclo::isqrt2(n)?,
            _ => return None,
        };
        self.cyclo_constant(&c, span)
    }

    /// `w(k, N)`: e^(2πik/N) as a `cyclo<N>`, both numbers written as
    /// constants.
    pub(super) fn exact_root(&mut self, k: i128, n: i128, span: Span) -> Option<Expr> {
        let n = u32::try_from(n).ok().filter(|&n| n > 0)?;
        let k = i64::try_from(k).ok()?;
        self.cyclo_constant(&Cyclo::zeta_pow(n, k), span)
    }

    /// `w(k, N)` as written: both are constants, and `N` is positive.
    pub(super) fn exact_root_call(
        &mut self,
        k: &crate::span::Spanned<crate::ast::Expr>,
        n: &crate::span::Spanned<crate::ast::Expr>,
        span: Span,
    ) -> Expr {
        let k_value = self.cx.const_value(k, Ty::Int(IntTy::I64), "the power `k` of `w(k, N)`");
        let n_value = self.cx.const_value(n, Ty::USIZE, "the order `N` of `w(k, N)`");
        let (Some(k_value), Some(n_value)) = (k_value.and_then(|v| v.as_int()), n_value.and_then(|v| v.as_int())) else {
            return Checker::error(span);
        };
        match self.exact_root(k_value, n_value, span) {
            Some(e) => e,
            None => {
                self.report(
                    crate::diag::Diagnostic::new(crate::diag::Code::Es06)
                        .with_message("`w(k, N)` is e^(2πik/N), for which `N` must be at least 1")
                        .at(n.span)
                        .with_note("`N` is the order of the root of unity, and the `cyclo<N>` it belongs to; there is no root of order 0")
                        .with_help("write the order as a positive constant, as in `w(1, 8)` for e^(2πi/8)"),
                );
                Checker::error(span)
            }
        }
    }

    /// `e`, a `cyclo<M>`, as the `cyclo<N>` wanted, when M divides N and so
    /// the one field lies in the other; `Err(e)` otherwise.
    pub(super) fn widen_cyclo(&mut self, e: Expr, want: Ty) -> Result<Expr, Expr> {
        let (Some(m), Some(n)) = (self.core_exact(e.ty, "cyclo"), self.core_exact(want, "cyclo")) else {
            return Err(e);
        };
        if m == n || m == 0 || n % m != 0 {
            return Err(e);
        }
        let span = e.span;
        let r = self.reference_to(e, span);
        Ok(self.core_call("__cyclo_widen", vec![Arg::Const(i128::from(m)), Arg::Const(i128::from(n))], vec![r], span))
    }

    /// The operands of `l op r` where both are `cyclo`s of different orders:
    /// the one whose order divides the other's is widened into the larger
    /// field, where the operator then applies. Neither dividing the other is
    /// reported, and gives `Err`.
    pub(super) fn cyclo_operands(&mut self, l: Expr, r: Expr) -> Result<(Expr, Expr), ()> {
        let (Some(m), Some(n)) = (self.core_exact(l.ty, "cyclo"), self.core_exact(r.ty, "cyclo")) else {
            return Ok((l, r));
        };
        if m == n || m == 0 || n == 0 {
            return Ok((l, r));
        }
        if n % m == 0 {
            let to = r.ty;
            return Ok((self.widen_cyclo(l, to).unwrap_or_else(|l| l), r));
        }
        if m % n == 0 {
            let to = l.ty;
            let r = self.widen_cyclo(r, to).unwrap_or_else(|r| r);
            return Ok((l, r));
        }
        let lcm = m / gcd(m, n) * n;
        let d = Diagnostic::new(Code::Es06)
            .with_message(format!("a `cyclo<{m}>` and a `cyclo<{n}>` cannot be combined as they are"))
            .at_with(l.span, format!("a `cyclo<{m}>`"))
            .also(r.span, format!("a `cyclo<{n}>`"))
            .with_note(format!(
                "a `cyclo<M>` widens to a `cyclo<N>` on its own only when M divides N, so that the one field lies \
                 inside the other; neither of {m} and {n} divides the other"
            ))
            .with_help(format!("give both operands the type `cyclo<{lcm}>`, which holds both, and they widen to it"));
        self.report(d);
        Err(())
    }

    /// `e as T` for an exact scalar `e`: `as f64` and `as f32` give an
    /// approximation, and `as cplx<f64>` one of a `cyclo`. `Err(e)` for any
    /// other conversion.
    pub(super) fn exact_cast(&mut self, e: Expr, target: Ty, span: Span) -> Result<Expr, Expr> {
        let exact = self.is_exact_number(e.ty) || self.core_exact(e.ty, "phase").is_some();
        if !exact {
            return self.integer_as_exact(e, target, span);
        }
        let method = match target {
            Ty::Float(_) => "to_f64",
            _ if self.core_exact(e.ty, "cyclo").is_some() && self.is_cplx_f64(target) => "to_cplx",
            _ => return Err(e),
        };
        let Ty::Adt(id) = self.shallow(e.ty) else { return Err(e) };
        let Some(sym) = self.interner.get(method) else { return Err(e) };
        let Some(entry) = self.cx.method_of_type(id, sym, false) else { return Err(e) };
        let label = format!("{}.{method}", self.show(e.ty));
        let call = self.call_method(entry, Some(e), &[], Vec::new(), &label, span);
        if target == Ty::Float(crate::tir::FloatTy::F32) {
            return Ok(Expr {
                kind: ExprKind::Cast {
                    expr: Box::new(call),
                    to: target,
                },
                ty: target,
                span,
            });
        }
        Ok(call)
    }

    /// `n as T` for an integer `n` and an exact `T`: `frac`, `cyclo<N>` or
    /// `cplx<U>`, the last with `n` as its real part.
    fn integer_as_exact(&mut self, e: Expr, target: Ty, span: Span) -> Result<Expr, Expr> {
        let from = self.shallow(e.ty);
        let integer = matches!(from, Ty::Int(_)) || self.infer.is_integer_variable(from);
        let exact = ["frac", "cyclo", "cplx"].iter().any(|n| self.core_exact(target, n).is_some());
        if !integer || !exact {
            return Err(e);
        }
        let n = if self.shallow(e.ty) == Ty::Int(IntTy::I64) {
            e
        } else {
            let ty = Ty::Int(IntTy::I64);
            let _ = self.unify(e.ty, ty);
            Expr {
                kind: ExprKind::Cast {
                    expr: Box::new(e),
                    to: ty,
                },
                ty,
                span,
            }
        };
        if let Some(c) = self.core_exact(target, "cyclo") {
            return Ok(self.core_call("__cyclo_int", vec![Arg::Const(i128::from(c))], vec![n], span));
        }
        if self.core_exact(target, "cplx").is_some()
            && let Ty::Adt(id) = self.shallow(target)
            && let Some(Arg::Type(u)) = self.adt(id).args.first().copied()
        {
            return Ok(self.core_call("__cplx_int", vec![Arg::Type(u)], vec![n], span));
        }
        if self.core_exact(target, "frac").is_some()
            && let Ty::Adt(id) = self.shallow(target)
            && let Some(sym) = self.interner.get("from")
            && let Some(entry) = self.cx.method_of_type(id, sym, false)
        {
            return Ok(self.call_method(entry, None, &[], vec![n], "frac::from", span));
        }
        Err(n)
    }

    /// `l * r` where either is a vector, covector or matrix and the product
    /// is not an operator method of `l`'s: a matrix or a covector times a
    /// vector is `apply`, a vector times a covector `outer`, and any of them
    /// times a number `scale`, a number written as a literal taken as an
    /// element. `Err` gives the operands back for any other product, such as
    /// two matrices, whose product is the left one's `$mul`.
    pub(super) fn linear_product(&mut self, l: Expr, r: Expr, span: Span) -> Result<Expr, Box<(Expr, Expr)>> {
        match (self.linear_kind(l.ty), self.linear_kind(r.ty)) {
            (Some(Linear::Mat | Linear::Covec), Some(Linear::Vec)) => Ok(self.named_method(l, "apply", r, span)),
            (Some(Linear::Vec), Some(Linear::Covec)) => Ok(self.named_method(l, "outer", r, span)),
            (Some(_), None) => {
                let k = self.as_element(r, l.ty);
                Ok(self.named_method(l, "scale", k, span))
            }
            (None, Some(_)) => {
                let k = self.as_element(l, r.ty);
                Ok(self.named_method(r, "scale", k, span))
            }
            _ => Err(Box::new((l, r))),
        }
    }

    /// Which of `core`'s linear algebra types `t` is, if it is one.
    fn linear_kind(&mut self, t: Ty) -> Option<Linear> {
        [(Linear::Vec, "vec"), (Linear::Covec, "covec"), (Linear::Mat, "mat")]
            .into_iter()
            .find(|&(_, name)| self.core_exact(t, name).is_some())
            .map(|(kind, _)| kind)
    }

    /// The kind, element type and dimensions of a `vec` or `covec` (one) or
    /// `mat` (rows, then columns), a dimension `None` where it is a generic
    /// parameter.
    fn linear_shape(&mut self, t: Ty) -> Option<(Linear, Ty, Vec<Option<i128>>)> {
        let kind = self.linear_kind(t)?;
        let Ty::Adt(id) = self.shallow(t) else { return None };
        let args = self.adt(id).args.clone();
        let Some(Arg::Type(elem)) = args.first().copied() else {
            return None;
        };
        let dims = args[1..]
            .iter()
            .map(|a| match a {
                Arg::Const(n) => Some(*n),
                _ => None,
            })
            .collect();
        Some((kind, elem, dims))
    }

    /// Reports `l op r` where both are vectors, covectors or matrices whose
    /// kinds, shapes or elements do not fit the operator, before the operator
    /// method is called: lending the operand to a method expecting another
    /// shape would otherwise be reported as a failed conversion between
    /// references. Returns whether it reported.
    pub(super) fn linear_mismatch(&mut self, op: BinOp, l: &Expr, r: &Expr) -> bool {
        let (Some((lk, le, ld)), Some((rk, re, rd))) = (self.linear_shape(l.ty), self.linear_shape(r.ty)) else {
            return false;
        };

        // a shape, as the message writes it; a dimension not yet known is `N`
        let shape = |k: Linear, d: &[Option<i128>]| {
            let n = |x: &Option<i128>| x.map_or_else(|| "N".to_owned(), |v| v.to_string());
            match (k, d) {
                (Linear::Vec, [a]) => format!("a vector of {} elements", n(a)),
                (Linear::Covec, [a]) => format!("a covector of {} elements", n(a)),
                (Linear::Mat, [a, b]) => format!("a {}×{} matrix", n(a), n(b)),
                _ => "a vector, covector or matrix".to_owned(),
            }
        };

        // what the operator does, the rule it breaks, if any, and what to do
        // about it: dimensions are fixed by the types, and a kind with no
        // product has an adjoint that has one
        let known = |a: Option<i128>, b: Option<i128>| matches!((a, b), (Some(x), Some(y)) if x != y);
        let dimensions = "check the dimensions in the two types; they are part of the type, so they are fixed when the program is translated";
        let (what, rule, help) = match op {
            BinOp::Mul => match (lk, ld.as_slice(), rk, rd.as_slice()) {
                (Linear::Mat, [_, c], Linear::Vec, [n]) if known(*c, *n) => (
                    "multiplied",
                    "a matrix times a vector needs as many elements in the vector as the matrix has columns",
                    dimensions,
                ),
                (Linear::Mat, [_, c], Linear::Mat, [p, _]) if known(*c, *p) => (
                    "multiplied",
                    "a matrix times a matrix needs as many rows in the right one as the left one has columns",
                    dimensions,
                ),
                (Linear::Covec, [n], Linear::Vec, [m]) if known(*n, *m) => (
                    "multiplied",
                    "a covector times a vector needs as many elements in each",
                    dimensions,
                ),
                (Linear::Covec, [n], Linear::Mat, [p, _]) if known(*n, *p) => (
                    "multiplied",
                    "a covector times a matrix needs as many rows in the matrix as the covector has elements",
                    dimensions,
                ),
                (Linear::Vec, _, Linear::Vec, _) => (
                    "multiplied",
                    "two vectors have no product: a vector is multiplied by a covector, on either side",
                    "write `adjoint(a) * b` for the inner product <a|b>, or `a * adjoint(b)` for the outer product |a><b|",
                ),
                (Linear::Covec, _, Linear::Covec, _) => (
                    "multiplied",
                    "two covectors have no product: a covector is multiplied by a vector or a matrix on its right",
                    "write `a * adjoint(b)` for the inner product of the two vectors they are the adjoints of",
                ),
                (Linear::Vec, _, Linear::Mat, _) => (
                    "multiplied",
                    "a vector stands on the right of a matrix; on its left stands a covector",
                    "write `adjoint(v) * m` for the covector <v| times the matrix",
                ),
                (Linear::Mat, _, Linear::Covec, _) => (
                    "multiplied",
                    "a covector stands on the left of a matrix; on its right stands a vector",
                    "write `m * adjoint(c)` for the matrix times the vector the covector is the adjoint of",
                ),
                _ => ("multiplied", "", ""),
            },
            BinOp::Add | BinOp::Sub | BinOp::Eq | BinOp::Ne => {
                let differ = lk != rk || ld.len() != rd.len() || ld.iter().zip(&rd).any(|(a, b)| known(*a, *b));
                let verb = match op {
                    BinOp::Add => "added",
                    BinOp::Sub => "subtracted",
                    _ => "compared",
                };
                let help = if lk == rk {
                    dimensions
                } else {
                    "a vector and a covector are of different kinds; `adjoint` turns either into the other"
                };
                (verb, if differ { "both operands must be of one kind and shape, element for element" } else { "" }, help)
            }
            _ => return false,
        };

        // kinds or shapes that do not fit, else elements of two types
        if !rule.is_empty() {
            let (ls, rs) = (shape(lk, &ld), shape(rk, &rd));
            let d = Diagnostic::new(Code::Es06)
                .with_message(format!("{ls} and {rs} cannot be {what}"))
                .at_with(l.span, ls)
                .also(r.span, rs)
                .with_note(rule.to_owned())
                .with_help(help.to_owned());
            self.report(d);
            return true;
        }
        if self.unify(le, re).is_err() {
            let (a, b) = (self.show(le), self.show(re));
            let d = Diagnostic::new(Code::Es06)
                .with_message(format!("a vector, covector or matrix of `{a}` and one of `{b}` cannot be {what}"))
                .at_with(l.span, format!("its elements are `{a}`"))
                .also(r.span, format!("its elements are `{b}`"))
                .with_note("the operators on vectors, covectors and matrices take elements of one type; they never convert them")
                .with_help(format!("convert the elements of one operand so both hold `{a}` or both `{b}`"));
            self.report(d);
            return true;
        }
        false
    }

    /// `k`, a number, as an element of the vector or matrix type `of`.
    fn as_element(&mut self, k: Expr, of: Ty) -> Expr {
        let Ty::Adt(id) = self.shallow(of) else { return k };
        let Some(Arg::Type(elem)) = self.adt(id).args.first().copied() else {
            return k;
        };
        let literal = matches!(&k.kind, ExprKind::Const(Value::Int(..))) && self.infer.is_integer_variable(self.shallow(k.ty));
        if literal
            && self.is_exact_number(elem)
            && let ExprKind::Const(Value::Int(n, _)) = &k.kind
        {
            let n = *n;
            if let Some(e) = self.exact_literal(Frac::from_bigint(BigInt::from(n)), elem, k.span) {
                return e;
            }
        }
        k
    }

    /// `recv.name(arg)`, with `arg` lent.
    fn named_method(&mut self, recv: Expr, name: &str, arg: Expr, span: Span) -> Expr {
        let Ty::Adt(id) = self.shallow(recv.ty) else { return Checker::error(span) };
        let Some(entry) = self.interner.get(name).and_then(|s| self.cx.method_of_type(id, s, false)) else {
            return Checker::error(span);
        };
        let label = format!("{}.{name}", self.show(recv.ty));
        let at = arg.span;
        let arg = if self.types().as_ref(self.shallow(arg.ty)).is_some() {
            arg
        } else {
            self.reference_to(arg, at)
        };
        self.call_method(entry, Some(recv), &[], vec![arg], &label, span)
    }

    fn is_cplx_f64(&mut self, t: Ty) -> bool {
        let Ty::Adt(id) = self.shallow(t) else { return false };
        let def = self.adt(id).clone();
        def.origin.as_deref() == Some("core")
            && self.name(def.name) == "cplx"
            && def.args == [Arg::Type(Ty::Float(crate::tir::FloatTy::F64))]
    }

    /// The exact value a number written as a literal denotes, sign and
    /// parentheses included: `0.1` is one tenth, not the nearest `f64`.
    pub(super) fn literal_exact(&self, e: &crate::ast::Expr) -> Option<Frac> {
        use crate::ast::{Expr as E, UnOp};
        match e {
            E::Int { raw, base, suffix: None } => {
                let digits: String = self.name(*raw).chars().filter(|&c| c != '_').collect();
                let n = BigInt::parse_bytes(digits.as_bytes(), base.radix())?;
                Some(Frac::from_bigint(n))
            }
            E::Float { raw, suffix: None } => Frac::parse_decimal(&self.name(*raw).replace('_', "")),
            E::Unary { op: UnOp::Neg, operand } => self.literal_exact(&operand.node).map(|f| -f),
            E::Paren(inner) => self.literal_exact(&inner.node),
            _ => None,
        }
    }

    /// Whether `t` is `frac` or a `cyclo`.
    pub(super) fn is_exact_number(&mut self, t: Ty) -> bool {
        // looked at, not looked up: naming `frac` would bring it into a unit
        // that never uses it
        self.core_exact(t, "frac").is_some() || self.core_exact(t, "cyclo").is_some()
    }

    /// `e` checked where a `frac` or `cyclo` is wanted: a number written as
    /// a literal is one, and so is arithmetic over such numbers, as in
    /// `1/3`. `None` when `e` is neither.
    pub(super) fn exact_for(&mut self, e: &crate::span::Spanned<crate::ast::Expr>, want: Ty) -> Option<Expr> {
        use crate::ast::{BinOp as B, Expr as E};
        if let Some(f) = self.literal_exact(&e.node) {
            return self.exact_literal(f, want, e.span);
        }
        match &e.node {
            E::Binary { op: op @ (B::Add | B::Sub | B::Mul | B::Div), lhs, rhs } => {
                let l = self.expr_for(lhs, want);
                let r = self.expr_for(rhs, want);
                let op = match op {
                    B::Add => crate::tir::BinOp::Add,
                    B::Sub => crate::tir::BinOp::Sub,
                    B::Mul => crate::tir::BinOp::Mul,
                    _ => crate::tir::BinOp::Div,
                };
                Some(self.binary_checked(op, l, r, e.span))
            }
            E::Paren(inner) => self.exact_for(inner, want),
            _ => None,
        }
    }

    /// A number written where a `frac` or `cyclo<N>` is wanted, as one.
    pub(super) fn exact_literal(&mut self, value: Frac, want: Ty, span: Span) -> Option<Expr> {
        if let Some(frac) = self.frac_type()
            && self.shallow(want) == frac
        {
            return Some(Expr::constant(frac_value(&value), frac, span));
        }
        let n = self.core_exact(want, "cyclo")?;
        let n = u32::try_from(n).ok()?;
        self.cyclo_constant(&Cyclo::from_frac(n, value), span)
    }
}
