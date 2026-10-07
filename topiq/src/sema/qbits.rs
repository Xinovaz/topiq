//! Classical operators on quantum data.
//!
//! The bitwise and logical operators applied to qubits are the reversible
//! operations they stand for, so that an algorithm reads the same in a
//! classical and a quantum unit.
//!
//! # Conditions
//!
//! `!`, `&`, `|`, `^`, `&&`, `||`, `==` and `!=` over qubits, handles to
//! qubits and `bool`s make a *quantum condition*, of type `qubit`. Nothing
//! is computed where it is written: an `if` applies its branches controlled
//! on it, and `t ^= c` flips `t` where it holds. `&` means what `&&` does,
//! since a qubit is not evaluated and so nothing is skipped. `==` and `!=`
//! on two references still ask whether they refer to the same place;
//! `*a == *b` compares the qubits.
//!
//! A condition used as a value, as in `let p = a & b;`, is a new qubit
//! holding it: `{ let p: qubit; p ^= a & b; p }`. Its operands are read,
//! not consumed. [`materialize`] writes that out once the unit is checked.
//!
//! # `^=`
//!
//! - **`t ^= c`** on a qubit flips `t` where `c` holds: `t ^= true` is the
//!   bit flip, `t ^= c` the controlled flip, `t ^= a & b` the Toffoli gate.
//!   A condition known when the circuit is generated, or measured, chooses
//!   whether to flip as an `if` would.
//! - **`r ^= s`** on two registers of one length flips each qubit of `r`
//!   where the one of `s` beside it holds; `r ^= k`, for an integer `k`,
//!   flips the qubits where `k` has a one, the first qubit taking the most
//!   significant bit; `r ^= b` for a `[bool; N]` flips where `b` is true.
//! - **The other compound assignments** have no reversible meaning on
//!   qubits and are `EQ18`: `&=` and `|=` lose the bit they overwrite, and
//!   a shift loses the bits it pushes off.
//!
//! Registers combine with `^`, `&`, `|` and `!` qubit by qubit, giving a new
//! register.
//!
//! # `quint<N>`
//!
//! A `quint<N>`, declared by the `gates` library, is a register of `N`
//! qubits read as an unsigned integer, its first qubit the most significant
//! bit, as in a ket. It has 1 to 64 qubits (`ES23`), since measuring it gives
//! the smallest unsigned integer of at least `N` bits.
//!
//! - **An integer placed on one**, by a literal or by `^=`, must fit: a
//!   constant is checked here (`EC10`), and any other value, which is known
//!   while the circuit is generated, there (`EC04`).
//! - **The comparisons** are quantum conditions over the bits: equal where
//!   every bit is, and less where, at the first bit from the most
//!   significant end where the two differ, the left has the zero. The bits
//!   of a constant are folded in, so `x == 5` tests each qubit once.
//! - **Arithmetic wraps at 2^N**, since a circuit cannot stop in one branch
//!   of a superposition, and Topiq does not wrap silently: `+`, `-`, `*`,
//!   their assignments, `++`, `--` and unary `-` are `EQ19`, and are written
//!   `wrapping_add` and the rest. Those names in `core`, given a `quint`, are
//!   the `gates` library's ([`QUINT_COUNTERPARTS`]). The other operand is
//!   lent, not consumed; multiplication is by an odd number, the only kind
//!   with an inverse modulo 2^N (`EQ18` otherwise).
//!
//! # `qcopy`
//!
//! `qcopy(&x)` is a second preparation of the quantum value `x`: circuit
//! generation repeats on fresh qubits what prepared it, and copies its
//! classical parts, each by its type's `$copy` where it has one. Which values
//! that is possible for is decided there (`EJ19`). A type holding qubits does not
//! define `$copy`, since reading a place of it moves its qubits.

use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::Span;
use crate::tir::visit::{self, VisitMut};
use crate::tir::{
    Access, Arg, BinOp, Block, Expr, ExprKind, Gate, Intrinsic, Local, LocalId, LogicalOp, QuantumOp, Stmt, Ty, UnOp,
    Unit,
    Value,
};

use super::body::Checker;

/// Whether `e` is a quantum condition built by an operator, rather than a
/// qubit held somewhere.
pub(super) fn is_condition_node(e: &Expr) -> bool {
    e.ty == Ty::Qubit
        && matches!(
            e.kind,
            ExprKind::Unary { op: UnOp::Not, .. }
                | ExprKind::Logical { .. }
                | ExprKind::Binary {
                    op: BinOp::BitXor,
                    ..
                }
        )
}

/// The `core` functions that, given a `quint`, are the `gates` library's
/// functions of the same name, so that one name serves both kinds of unit.
pub const QUINT_COUNTERPARTS: [&str; 6] =
    ["wrapping_add", "wrapping_sub", "wrapping_mul", "wrapping_neg", "rotate_left", "rotate_right"];

/// The wrapping operation an arithmetic operator on a `quint` is written
/// as, for the operators that have one.
fn wrapping_name(op: BinOp) -> Option<&'static str> {
    match op {
        BinOp::Add => Some("wrapping_add"),
        BinOp::Sub => Some("wrapping_sub"),
        BinOp::Mul => Some("wrapping_mul"),
        _ => None,
    }
}

/// Builds quantum conditions, folding the `bool`s known now so that a
/// comparison with a constant tests only the qubits it must.
struct Cond {
    span: Span,
}

impl Cond {
    fn node(&self, kind: ExprKind) -> Expr {
        Expr {
            kind,
            ty: Ty::Qubit,
            span: self.span,
        }
    }

    fn known(e: &Expr) -> Option<bool> {
        match e.kind {
            ExprKind::Const(Value::Bool(b)) => Some(b),
            _ => None,
        }
    }

    fn constant(&self, b: bool) -> Expr {
        Expr::constant(Value::Bool(b), Ty::Bool, self.span)
    }

    fn not(&self, e: Expr) -> Expr {
        match Self::known(&e) {
            Some(b) => self.constant(!b),
            None => self.node(ExprKind::Unary {
                op: UnOp::Not,
                operand: Box::new(e),
            }),
        }
    }

    fn logical(&self, op: LogicalOp, a: Expr, b: Expr) -> Expr {
        // the value that decides `op` on its own
        let decides = op == LogicalOp::Or;
        match (Self::known(&a), Self::known(&b)) {
            (Some(x), _) if x == decides => self.constant(decides),
            (_, Some(x)) if x == decides => self.constant(decides),
            (Some(_), _) => b,
            (_, Some(_)) => a,
            _ => self.node(ExprKind::Logical {
                op,
                lhs: Box::new(a),
                rhs: Box::new(b),
            }),
        }
    }

    fn and(&self, a: Expr, b: Expr) -> Expr {
        self.logical(LogicalOp::And, a, b)
    }

    fn xor(&self, a: Expr, b: Expr) -> Expr {
        match (Self::known(&a), Self::known(&b)) {
            (Some(x), Some(y)) => self.constant(x ^ y),
            (Some(x), None) => if x { self.not(b) } else { b },
            (None, Some(y)) => if y { self.not(a) } else { a },
            (None, None) => self.node(ExprKind::Binary {
                op: BinOp::BitXor,
                lhs: Box::new(a),
                rhs: Box::new(b),
            }),
        }
    }

    fn eq(&self, a: Expr, b: Expr) -> Expr {
        self.not(self.xor(a, b))
    }

    fn all(&self, es: Vec<Expr>) -> Expr {
        es.into_iter().fold(self.constant(true), |acc, e| self.and(acc, e))
    }

    fn any(&self, es: Vec<Expr>) -> Expr {
        es.into_iter().fold(self.constant(false), |acc, e| self.logical(LogicalOp::Or, acc, e))
    }
}

impl Checker<'_, '_> {
    /// The length `N` of `t` when it is a `[bool; N]`, whose bitwise
    /// operators apply element by element.
    pub(super) fn bool_array(&self, t: Ty) -> Option<u64> {
        match self.types().as_array(self.shallow(t)) {
            Some((Ty::Bool, n)) => Some(n),
            _ => None,
        }
    }

    /// The width `N` of `t` when it is the `gates` library's `quint<N>`.
    pub(super) fn quint_width(&mut self, t: Ty) -> Option<u64> {
        let Ty::Adt(id) = self.shallow(t) else { return None };
        let def = self.adt(id).clone();
        let in_gates = match &def.origin {
            Some(o) => o == "gates",
            None => self.cx.unit.name == "gates",
        };
        if !in_gates || self.name(def.name) != "quint" {
            return None;
        }
        match def.args.first() {
            Some(Arg::Const(n)) => u64::try_from(*n).ok(),
            _ => None,
        }
    }

    /// The width of `t` when it is a `quint<N>` or a handle to one.
    fn quint_of(&mut self, t: Ty) -> Option<u64> {
        let t = self.shallow(t);
        let t = self.types().as_ref(t).map_or(t, |(_, x)| x);
        self.quint_width(t)
    }

    /// The register of the `quint` `e` holds or a handle to it names.
    fn quint_bits(&mut self, e: Expr) -> Expr {
        let span = e.span;
        let t = self.shallow(e.ty);
        let (base, q) = match self.types().as_ref(t) {
            Some((_, q)) => (Expr::deref(e, q, span), q),
            None => (e, t),
        };
        let n = self.quint_width(q).expect("a quint");
        let ty = self.types_mut().array(Ty::Qubit, n);
        Expr {
            kind: ExprKind::Field {
                base: Box::new(base),
                field: 0,
            },
            ty,
            span,
        }
    }

    /// Qubit `i` of the register `bits`.
    fn bit(bits: &Expr, i: u64) -> Expr {
        let span = bits.span;
        Expr {
            kind: ExprKind::Index {
                base: Box::new(bits.clone()),
                index: Box::new(Expr::constant(Value::Int(i128::from(i), crate::tir::IntTy::USIZE), Ty::USIZE, span)),
            },
            ty: Ty::Qubit,
            span,
        }
    }

    /// `l op r` where an operand is a `quint`: `^`, `&` and `|` give a new
    /// one, and the comparisons a quantum condition.
    fn quint_binary(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        let text = op.text();
        if let Some(name) = wrapping_name(op) {
            let d = Self::wraps(text, &format!("{name}(x, …)"), span);
            self.report(d);
            return Checker::error(span);
        }
        let (ln, rn) = (self.quint_of(l.ty), self.quint_of(r.ty));
        match op {
            BinOp::BitXor | BinOp::BitAnd | BinOp::BitOr => {
                let (Some(n), true) = (ln, ln == rn) else {
                    let (a, b) = (self.describe(l.ty), self.describe(r.ty));
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{text}` combines two `quint`s of one width, but these are {a} and {b}"))
                            .at(span),
                    );
                    return Checker::error(span);
                };
                let name = match op {
                    BinOp::BitXor => "__qxor",
                    BinOp::BitAnd => "__qand",
                    _ => "__qor",
                };
                let (l, r) = (self.handle_of(l, span), self.handle_of(r, span));
                self.library_call("gates", name, vec![Arg::Const(i128::from(n))], vec![l, r], span)
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                // the `quint` on the left
                let (op, l, r, n) = match (ln, rn) {
                    (Some(n), _) => (op, l, r, n),
                    (None, Some(n)) => {
                        let mirrored = match op {
                            BinOp::Lt => BinOp::Gt,
                            BinOp::Le => BinOp::Ge,
                            BinOp::Gt => BinOp::Lt,
                            BinOp::Ge => BinOp::Le,
                            other => other,
                        };
                        (mirrored, r, l, n)
                    }
                    (None, None) => unreachable!("one operand is a quint"),
                };
                self.quint_compare(op, l, r, n, span)
            }
            _ => {
                let d = self.irreversible(text, span);
                self.report(d);
                Checker::error(span)
            }
        }
    }

    /// `x op y` for a `quint<n>` `x` and another of its width or an integer
    /// `y`, as a quantum condition over their bits: equal where every bit
    /// is, and less where, at the first bit from the most significant end
    /// where they differ, `x` has the zero.
    fn quint_compare(&mut self, op: BinOp, x: Expr, y: Expr, n: u64, span: Span) -> Expr {
        for side in [&x, &y] {
            if !super::step::evaluates_purely(side) {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{}` on a `quint` reads each of its bits, so this must be a place", op.text()))
                        .at(side.span)
                        .with_help("hold the value in a binding first"),
                );
                return Checker::error(span);
            }
        }
        let xs = self.quint_bits(x);
        // the other side's bit i: a qubit, or a `bool` known now or when the
        // circuit is generated
        let ys: Vec<Expr> = if self.quint_of(y.ty) == Some(n) {
            let ys = self.quint_bits(y);
            (0..n).map(|i| Self::bit(&ys, i)).collect()
        } else {
            if self.quint_of(y.ty).is_some() || !self.shallow(y.ty).is_integer() {
                let (a, b) = (self.describe(xs.ty), self.describe(y.ty));
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "`{}` compares a `quint` with another of its width or an integer, not {a} with {b}",
                            op.text()
                        ))
                        .at(span),
                );
                return Checker::error(span);
            }
            let Some(k) = self.bits_for(y, n, span) else { return Checker::error(span) };
            (0..n).map(|i| self.bit_of(&k, n - 1 - i, span)).collect()
        };
        let xs: Vec<Expr> = (0..n).map(|i| Self::bit(&xs, i)).collect();
        let c = Cond { span };
        let eq: Vec<Expr> = xs.iter().zip(&ys).map(|(a, b)| c.eq(a.clone(), b.clone())).collect();
        let all_eq = || c.all(eq.clone());
        // x < y where, at the first differing bit, x has 0 and y 1
        let less = |xs: &[Expr], ys: &[Expr]| {
            let terms = (0..xs.len()).map(|i| {
                let differs = c.and(c.not(xs[i].clone()), ys[i].clone());
                c.and(c.all(eq[..i].to_vec()), differs)
            });
            c.any(terms.collect())
        };
        match op {
            BinOp::Eq => all_eq(),
            BinOp::Ne => c.not(all_eq()),
            BinOp::Lt => less(&xs, &ys),
            BinOp::Gt => less(&ys, &xs),
            BinOp::Le => c.not(less(&ys, &xs)),
            _ => c.not(less(&xs, &ys)),
        }
    }

    /// Bit `at` of the `u64` `k`, counting from the least significant, as
    /// a `bool`.
    fn bit_of(&mut self, k: &Expr, at: u64, span: Span) -> Expr {
        let u64t = Ty::Int(crate::tir::IntTy::U64);
        if let ExprKind::Const(Value::Int(v, _)) = k.kind {
            return Expr::constant(Value::Bool((v >> at) & 1 == 1), Ty::Bool, span);
        }
        let int = |v: i128| Expr::constant(Value::Int(v, crate::tir::IntTy::U64), u64t, span);
        let shifted = self.binary_checked(BinOp::Shr, k.clone(), int(i128::from(at)), span);
        let low = self.binary_checked(BinOp::BitAnd, shifted, int(1), span);
        self.binary_checked(BinOp::Eq, low, int(1), span)
    }

    /// `qcopy(&x)`: a second preparation of the quantum value `x`, made by
    /// repeating what prepared it, which circuit generation checks it can.
    pub(super) fn qcopy_call(&mut self, args: &[crate::span::Spanned<crate::ast::Expr>], span: Span) -> Expr {
        let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
        if !self.cx.quantum {
            return self.quantum_in_classical("copy quantum state", span);
        }
        let [r] = &checked[..] else {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`qcopy` takes one argument, a reference to what it copies")
                    .at(span),
            );
            return Checker::error(span);
        };
        let t = self.settled(r.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        let Some((_, target)) = self.types().as_ref(t) else {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`qcopy` takes a reference to what it copies, but this is {what}"))
                    .at(r.span)
                    .with_help("lend the value, as in `qcopy(&q)`"),
            );
            return Checker::error(span);
        };
        if !self.types().is_quantum(target) {
            let shown = self.show(target);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`qcopy` copies quantum values, and `{shown}` holds no qubits"))
                    .at(r.span)
                    .with_help("copy a classical value with `clone`"),
            );
            return Checker::error(span);
        }
        let r = checked.into_iter().next().expect("one argument");
        Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Copy,
                args: vec![r],
            },
            ty: target,
            span,
        }
    }

    /// `name(args)`, one of [`QUINT_COUNTERPARTS`], when its first argument
    /// is a `quint`: the `gates` library's operation. The other operand is
    /// lent, not consumed, and an integer one is placed on qubits for the
    /// operation. `Err` gives the arguments back when the first is not a
    /// `quint`.
    pub(super) fn quint_call(&mut self, name: &str, mut args: Vec<Expr>, span: Span) -> Result<Expr, Vec<Expr>> {
        let Some(n) = args.first().and_then(|a| self.quint_width(a.ty)) else {
            return Err(args);
        };
        let targs = vec![Arg::Const(i128::from(n))];
        let unary = matches!(name, "wrapping_neg");
        let want = if unary { 1 } else { 2 };
        if args.len() != want {
            self.report(Checker::wrong_arity(name, want, args.len(), span));
            return Ok(Checker::error(span));
        }
        if unary || name.starts_with("rotate") {
            return Ok(self.library_call("gates", name, targs, args, span));
        }
        let other = args.pop().expect("two arguments");
        let other_ty = self.shallow(other.ty);
        if self.quint_of(other_ty) == Some(n) && name != "wrapping_mul" {
            let lent = self.handle_of(other, span);
            args.push(lent);
            return Ok(self.library_call("gates", name, targs, args, span));
        }
        if other_ty.is_integer() && !self.is_floating(other_ty) {
            if name == "wrapping_mul"
                && let ExprKind::Const(Value::Int(k, _)) = other.kind
                && k % 2 == 0
            {
                self.report(
                    Diagnostic::new(Code::Eq18)
                        .with_message(format!("multiplying a `quint` by {k}, an even number, loses its top bit"))
                        .at(other.span)
                        .with_note(
                            "a product is reversible only when the factor has an inverse modulo 2^N, \
                             which only odd numbers do",
                        ),
                );
                return Ok(Checker::error(span));
            }
            let Some(k) = self.bits_for(other, n, span) else {
                return Ok(Checker::error(span));
            };
            args.push(k);
            let gates = format!("__{name}_k");
            return Ok(self.library_call("gates", &gates, targs, args, span));
        }
        let what = self.describe(other_ty);
        let wanted = if name == "wrapping_mul" { "an odd integer" } else { "another `quint` of its width or an integer" };
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("`{name}` on a `quint<{n}>` takes {wanted}, not {what}"))
                .at(other.span),
        );
        Ok(Checker::error(span))
    }

    /// Why `text` on a `quint` must be written as the wrapping operation
    /// `instead`.
    fn wraps(text: &str, instead: &str, span: Span) -> Diagnostic {
        Diagnostic::new(Code::Eq19)
            .with_message(format!("`{text}` on a `quint` would wrap, and Topiq does not wrap silently"))
            .at(span)
            .with_note(
                "a circuit cannot stop in one branch of a superposition, so arithmetic on qubits always \
                 wraps at 2^N; the classical types abort instead",
            )
            .with_help(format!("write `{instead}`"))
    }

    /// The length of a fixed register `[qubit; N]`, or of one a handle names.
    fn qubit_array(&self, t: Ty) -> Option<u64> {
        let t = self.shallow(t);
        let t = self.types().as_ref(t).map_or(t, |(_, x)| self.shallow(x));
        match self.types().as_array(t) {
            Some((Ty::Qubit, n)) => Some(n),
            _ => None,
        }
    }

    /// A handle to what `e` holds: `e` itself when it is a reference, and
    /// otherwise a reference to it.
    fn handle_of(&mut self, e: Expr, span: Span) -> Expr {
        if self.shallow(e.ty).is_reference() { e } else { self.reference_to(e, span) }
    }

    /// `l op r` where an operand is a qubit or a register, or `Err` with
    /// the operands when neither is.
    pub(super) fn quantum_binary(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Result<Expr, Box<(Expr, Expr)>> {
        let (lt, rt) = (self.shallow(l.ty), self.shallow(r.ty));
        let (lq, rq) = (self.is_quantum_condition(lt), self.is_quantum_condition(rt));
        // two handles compared are two references
        if matches!(op, BinOp::Eq | BinOp::Ne) && lt.is_reference() && rt.is_reference() {
            return Err(Box::new((l, r)));
        }
        if self.quint_of(lt).is_some() || self.quint_of(rt).is_some() {
            return Ok(self.quint_binary(op, l, r, span));
        }
        if self.qubit_array(lt).is_some() || self.qubit_array(rt).is_some() {
            return Ok(self.register_binary(op, l, r, span));
        }
        if !lq && !rq {
            return Err(Box::new((l, r)));
        }
        let text = op.text();
        match op {
            BinOp::BitXor | BinOp::BitAnd | BinOp::BitOr | BinOp::Eq | BinOp::Ne => {}
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{text}` compares numbers, and a qubit holds one bit"))
                        .at(span)
                        .with_help("compare a `quint<N>`, which holds a number"),
                );
                return Ok(Checker::error(span));
            }
            _ => {
                let d = self.irreversible(text, span);
                self.report(d);
                return Ok(Checker::error(span));
            }
        }
        let (Some(l), Some(r)) = (self.condition_operand(l, text), self.condition_operand(r, text)) else {
            return Ok(Checker::error(span));
        };
        let node = |kind| Expr { kind, ty: Ty::Qubit, span };
        let xor = |l: Expr, r: Expr| {
            node(ExprKind::Binary {
                op: BinOp::BitXor,
                lhs: Box::new(l),
                rhs: Box::new(r),
            })
        };
        Ok(match op {
            BinOp::BitAnd | BinOp::BitOr => node(ExprKind::Logical {
                op: if op == BinOp::BitAnd { LogicalOp::And } else { LogicalOp::Or },
                lhs: Box::new(l),
                rhs: Box::new(r),
            }),
            BinOp::Eq => node(ExprKind::Unary {
                op: UnOp::Not,
                operand: Box::new(xor(l, r)),
            }),
            _ => xor(l, r),
        })
    }

    /// One operand of a quantum condition: a qubit, a handle to one, or a
    /// `bool`.
    fn condition_operand(&mut self, e: Expr, text: &str) -> Option<Expr> {
        let t = self.shallow(e.ty);
        if self.is_quantum_condition(t) || matches!(t, Ty::Bool | Ty::Never) {
            return Some(e);
        }
        let what = self.describe(t);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("`{text}` combines a qubit with a qubit or a `bool`, not with {what}"))
                .at(e.span),
        );
        None
    }

    /// `!e` or `-e` where `e` is quantum, or `Err` with the operand when it
    /// is not.
    pub(super) fn quantum_unary(&mut self, op: UnOp, o: Expr, span: Span) -> Result<Expr, Expr> {
        let t = self.shallow(o.ty);
        if let Some(n) = self.quint_of(t) {
            if op == UnOp::Not {
                let a = self.handle_of(o, span);
                return Ok(self.library_call("gates", "__qnot", vec![Arg::Const(i128::from(n))], vec![a], span));
            }
            let d = Self::wraps("-", "x = wrapping_neg(x)", span);
            self.report(d);
            return Ok(Checker::error(span));
        }
        if let Some(n) = self.qubit_array(t) {
            if op == UnOp::Not {
                let r = self.handle_of(o, span);
                return Ok(self.library_call("gates", "__not_new", vec![Arg::Const(i128::from(n))], vec![r], span));
            }
            let d = self.irreversible("-", span);
            self.report(d);
            return Ok(Checker::error(span));
        }
        if !self.is_quantum_condition(t) {
            return Err(o);
        }
        if op == UnOp::Neg {
            let d = self.irreversible("-", span);
            self.report(d);
            return Ok(Checker::error(span));
        }
        Ok(Expr {
            kind: ExprKind::Unary {
                op: UnOp::Not,
                operand: Box::new(o),
            },
            ty: Ty::Qubit,
            span,
        })
    }

    /// `l && r` or `l || r` where an operand is quantum, or `Err` with the
    /// operands when neither is.
    pub(super) fn quantum_logical(&mut self, op: LogicalOp, l: Expr, r: Expr, span: Span) -> Result<Expr, Box<(Expr, Expr)>> {
        let (lt, rt) = (self.shallow(l.ty), self.shallow(r.ty));
        if !self.is_quantum_condition(lt) && !self.is_quantum_condition(rt) {
            return Err(Box::new((l, r)));
        }
        let (Some(l), Some(r)) = (self.condition_operand(l, op.text()), self.condition_operand(r, op.text())) else {
            return Ok(Checker::error(span));
        };
        Ok(Expr {
            kind: ExprKind::Logical {
                op,
                lhs: Box::new(l),
                rhs: Box::new(r),
            },
            ty: Ty::Qubit,
            span,
        })
    }

    /// `l op r` where an operand is a register: a new register, qubit by
    /// qubit, for `^`, `&` and `|`.
    fn register_binary(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        let (ln, rn) = (self.qubit_array(l.ty), self.qubit_array(r.ty));
        let text = op.text();
        let name = match op {
            BinOp::BitXor => "__xor_new",
            BinOp::BitAnd => "__and_new",
            BinOp::BitOr => "__or_new",
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{text}` compares numbers, and a register holds bits"))
                        .at(span)
                        .with_help("compare a `quint<N>`, which holds a number"),
                );
                return Checker::error(span);
            }
            _ => {
                let d = self.irreversible(text, span);
                self.report(d);
                return Checker::error(span);
            }
        };
        let (Some(n), true) = (ln, ln == rn) else {
            let (a, b) = (self.describe(l.ty), self.describe(r.ty));
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "`{text}` combines two registers of one length, qubit by qubit, but these are {a} and {b}"
                    ))
                    .at(span),
            );
            return Checker::error(span);
        };
        let (l, r) = (self.handle_of(l, span), self.handle_of(r, span));
        self.library_call("gates", name, vec![Arg::Const(i128::from(n))], vec![l, r], span)
    }

    /// `place op= value` where the place holds qubits.
    pub(super) fn quantum_assign(&mut self, op: BinOp, place: Expr, value: Expr, span: Span) -> Expr {
        let text = format!("{}=", op.text());
        let t = self.shallow(place.ty);
        if self.quint_width(t).is_some()
            && let Some(name) = wrapping_name(op)
        {
            let d = Self::wraps(&text, &format!("x = {name}(x, …)"), span);
            self.report(d);
            return Checker::error(span);
        }
        if op != BinOp::BitXor {
            let d = self.irreversible(&text, span);
            self.report(d);
            return Checker::error(span);
        }
        let pa = self.place_access(&place);
        if pa.access != Access::Write {
            let d = self.not_mutable(&place, pa, "`^=` on", place.span);
            self.report(d);
            return Checker::error(span);
        }
        if t == Ty::Qubit {
            return self.flip(place, value, span);
        }
        if let Some(n) = self.qubit_array(t) {
            return self.register_xor(place, n, value, span);
        }
        // a `quint` is its register, read as a number
        if let Some(n) = self.quint_width(t) {
            let bits = self.quint_bits(place);
            let value = if self.quint_of(value.ty) == Some(n) { self.quint_bits(value) } else { value };
            return self.register_xor(bits, n, value, span);
        }
        let what = self.describe(t);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("`^=` flips qubits, and {what} is not a qubit or a register of them"))
                .at(place.span),
        );
        Checker::error(span)
    }

    /// `t ^= c` on one qubit.
    fn flip(&mut self, target: Expr, cond: Expr, span: Span) -> Expr {
        let handle = self.reference_to(target, span);
        let ct = self.shallow(cond.ty);
        if self.is_quantum_condition(ct) {
            return Expr {
                kind: ExprKind::Quantum {
                    op: QuantumOp::Flip,
                    args: vec![handle, cond],
                },
                ty: Ty::Void,
                span,
            };
        }
        // a condition known now or measured chooses whether to flip
        let cond = self.condition_checked(cond, "the right operand of `^=` on a qubit");
        let x = Expr::intrinsic(Intrinsic::Gate(Gate::X), vec![handle], Ty::Void, span);
        Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then: Box::new(Block {
                    stmts: vec![Stmt::Expr(x)],
                    value: None,
                    ty: Ty::Void,
                    span,
                }),
                els: None,
            },
            ty: Ty::Void,
            span,
        }
    }

    /// `r ^= v` on a register of `n` qubits: another register of `n`, a
    /// `[bool; n]`, or an integer that fits in `n` bits.
    fn register_xor(&mut self, place: Expr, n: u64, value: Expr, span: Span) -> Expr {
        let vt = self.shallow(value.ty);
        let targs = vec![Arg::Const(i128::from(n))];
        let target = self.reference_to(place, span);
        if self.qubit_array(vt) == Some(n) {
            let source = self.handle_of(value, span);
            return self.library_call("gates", "__xor_reg", targs, vec![target, source], span);
        }
        if self.bool_array(vt) == Some(n) {
            return self.library_call("gates", "__xor_bools", targs, vec![target, value], span);
        }
        if vt.is_integer() && !self.is_floating(vt) {
            let Some(value) = self.bits_for(value, n, span) else {
                return Checker::error(span);
            };
            return self.library_call("gates", "__xor_bits", targs, vec![target, value], span);
        }
        let what = self.describe(vt);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!(
                    "`^=` on a register of {n} qubits takes another register of {n}, a `[bool; {n}]` or an \
                     integer, not {what}"
                ))
                .at(value.span),
        );
        Checker::error(span)
    }

    /// An integer to be placed on `n` qubits, as a `u64`. A constant must fit
    /// in `n` bits (`EC10` otherwise); any other value is known while the
    /// circuit is generated, since a measured one cannot be passed on, and
    /// the library checks it there.
    pub(super) fn bits_for(&mut self, value: Expr, n: u64, span: Span) -> Option<Expr> {
        let t = self.settled(value.ty);
        if let ExprKind::Const(Value::Int(v, _)) = value.kind {
            let fits = v >= 0 && (n >= 64 || v < (1i128 << n));
            if !fits {
                self.report(
                    Diagnostic::new(Code::Ec10)
                        .with_message(format!("{v} does not fit in {n} qubits"))
                        .at(value.span)
                        .with_note(format!(
                            "{n} qubits hold the numbers 0 to 2^{n} - 1; the bits that do not fit would be lost"
                        )),
                );
                return None;
            }
            return Some(Expr::constant(Value::Int(v, crate::tir::IntTy::U64), Ty::Int(crate::tir::IntTy::U64), span));
        }
        if !matches!(t, Ty::Int(_)) {
            return None;
        }
        Some(self.coerce_int(value, crate::tir::IntTy::U64))
    }

    /// A widening of an unsigned integer to `to`.
    fn coerce_int(&mut self, value: Expr, to: crate::tir::IntTy) -> Expr {
        let span = value.span;
        Expr {
            kind: ExprKind::Cast {
                expr: Box::new(value),
                to: Ty::Int(to),
            },
            ty: Ty::Int(to),
            span,
        }
    }

    /// Why `text` has no meaning on quantum data.
    fn irreversible(&self, text: &str, span: Span) -> Diagnostic {
        let note = match text {
            "&=" | "|=" => "it would overwrite each qubit with a value that depends on what it held, losing that",
            "<<=" | ">>=" | "<<" | ">>" => "a shift loses the bits it pushes off",
            _ => "a qubit holds a bit, not a number; a `quint<N>` holds a number",
        };
        Diagnostic::new(Code::Eq18)
            .with_message(format!("`{text}` has no reversible meaning on qubits"))
            .at(span)
            .with_note(note)
            .with_help("compute the result into a fresh qubit with `^=`, as in `out ^= a & b`")
    }
}

/// Writes out each quantum condition used as a value as a new qubit
/// holding it: `{ let p: qubit; p ^= c; p }`.
pub fn materialize(unit: &mut Unit) {
    let handle = unit.types.reference(Access::Write, Ty::Qubit);
    for f in &mut unit.fns {
        let mut m = Materialize {
            locals: &mut f.locals,
            handle,
        };
        m.block(&mut f.body);
    }
}

struct Materialize<'a> {
    locals: &'a mut Vec<Local>,
    handle: Ty,
}

impl Materialize<'_> {
    /// A condition's operands: nested conditions stay as they are, and
    /// anything else is walked as usual.
    fn condition(&mut self, e: &mut Expr) {
        if !is_condition_node(e) {
            return self.expr(e);
        }
        match &mut e.kind {
            ExprKind::Unary { operand, .. } => self.condition(operand),
            ExprKind::Logical { lhs, rhs, .. } | ExprKind::Binary { lhs, rhs, .. } => {
                self.condition(lhs);
                self.condition(rhs);
            }
            _ => {}
        }
    }
}

impl VisitMut for Materialize<'_> {
    fn expr(&mut self, e: &mut Expr) {
        if is_condition_node(e) {
            return self.materialize(e);
        }
        match &mut e.kind {
            ExprKind::If { cond, then, els } if cond.ty == Ty::Qubit => {
                self.condition(cond);
                self.block(then);
                if let Some(x) = els {
                    self.expr(x);
                }
            }
            ExprKind::Quantum {
                op: QuantumOp::Flip,
                args,
            } => {
                self.expr(&mut args[0]);
                self.condition(&mut args[1]);
            }
            _ => visit::walk_expr_mut(self, e),
        }
    }
}

impl Materialize<'_> {
    /// Replaces the condition `e` with a new qubit holding it.
    fn materialize(&mut self, e: &mut Expr) {
        self.condition(e);
        let span = e.span;
        let fresh = LocalId(self.locals.len() as u32);
        self.locals.push(Local {
            name: Symbol::EMPTY,
            ty: Ty::Qubit,
            constant: false,
            constexpr: false,
            aux: false,
            span,
        });
        let cond = std::mem::replace(e, Expr::local(fresh, Ty::Qubit, span));
        let handle = Expr {
            kind: ExprKind::Ref(Box::new(Expr::local(fresh, Ty::Qubit, span))),
            ty: self.handle,
            span,
        };
        let flip = Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Flip,
                args: vec![handle, cond],
            },
            ty: Ty::Void,
            span,
        };
        *e = Expr {
            kind: ExprKind::Block(Box::new(Block {
                stmts: vec![Stmt::Let { local: fresh, init: None }, Stmt::Expr(flip)],
                value: Some(Box::new(Expr::local(fresh, Ty::Qubit, span))),
                ty: Ty::Qubit,
                span,
            })),
            ty: Ty::Qubit,
            span,
        };
    }
}
