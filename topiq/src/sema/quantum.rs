//! The operations on quantum state: `measure` and `match measure`, `lift`,
//! `forget`, `replay`, `**` on registers, quantum conditions, the
//! gates whose operands are checked where they are applied (`rz`, `gphase`
//! and `apply`) and the operators made of operators: `adjoint(f)`,
//! `controlled(f)`, whose first parameter is a handle to the control qubit,
//! and `f.then(g)`, of two operators of one signature taking handles.
//!
//! A quantum unit may hold qubits; a classical unit may not, so each of these
//! written in a classical unit is `EU02`, naming the operation, except
//! `measure` of classical data, which either kind of unit takes (below).
//!
//! # Rules
//!
//! - **Qubits are linear.** Every operation here consumes its operand, so a
//!   register measured or forgotten cannot be used again. What happens to a
//!   quantum value that is never consumed is the business of the moves check
//!   ([`super::moves`]).
//! - **`measure` gives outcome data:** a `qubit` gives a `bool`, and a
//!   register `[qubit; N]` gives `[bool; N]`. A `quint<N>` gives the
//!   smallest unsigned integer of at least `N` bits, its first qubit the
//!   most significant bit. A quantum structure gives a
//!   tuple of its fields in order, each qubit-bearing one measured the same
//!   way and each classical one as it is; one holding a quantum enumeration
//!   is `ES06`, since an enumeration's variant is read by matching. The
//!   result is an *outcome* value, which exists only while the circuit
//!   runs; [`super::kinds`] checks where outcome values may go. A `quint`
//!   among a structure's fields, or an array of them, gives its numbers.
//! - **`measure` of classical data** gives it back, in a unit of either
//!   kind, with a warning (`ES24`): it stands for the quantum data the same
//!   routine holds in a unit of the other kind, so it takes the shape
//!   measuring that would give: a structure a tuple of its fields, each
//!   structure among them a tuple too. Only what quantum data measures to is
//!   taken: a `bool`, an unsigned integer, an array of `bool`s and a
//!   structure, and an enumeration in `match measure`.
//! - **`lift` takes an outcome value** and gives it back as one fixed when
//!   the circuit is generated, so that what follows may depend on it. The
//!   operator containing it becomes a dynamic circuit.
//! - **`a ** b` on registers** is one register holding `a`'s qubits and then
//!   `b`'s; a lone `qubit` counts as a register of one.
//! - **A quantum condition** is a qubit, a handle to one, or `!`, `&`, `|`,
//!   `^`, `&&`, `||`, `==` and `!=` over those and `bool`s, or a comparison
//!   of `quint`s ([`super::qbits`]). That is what an `if` may test without
//!   measuring: its branches are applied controlled on it. It is given type
//!   `qubit`; an `if` or `^=` takes it as it is, and anything else as a new
//!   qubit holding it. A `while` on one is `EQ03`, since how often a loop
//!   runs must be known in advance.
//! - **`match measure e`** on a quantum enumeration matches its variants,
//!   each arm binding the payload that survives the measurement, and the
//!   value of each classical field, which is measured with the tag.
//! - **`apply(u, r)`** takes an exact matrix, of `cyclo<N>` entries, as large
//!   as the register it acts on: `2^k` square for `k` qubits. A floating
//!   matrix is `EQ12`. Whether it is unitary is checked exactly when the
//!   circuit is generated, unless the operator is marked
//!   `[trusted_unitary]`, which makes the program answerable for it.

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{Arg, Expr, ExprKind, Gate, IntTy, Intrinsic, LogicalOp, QuantumOp, Ty};

use super::body::Checker;

/// What measuring a `quint<n>` gives: the smallest unsigned integer of at
/// least `n` bits.
pub fn quint_measured(n: u64) -> IntTy {
    match n {
        0..=8 => IntTy::U8,
        9..=16 => IntTy::U16,
        17..=32 => IntTy::U32,
        _ => IntTy::U64,
    }
}

impl Checker<'_, '_> {
    /// The number of qubits a register type holds: one for `qubit`, `N` for
    /// `[qubit; N]`, and `None` for anything else.
    pub fn register_width(&self, t: Ty) -> Option<u64> {
        match t {
            Ty::Qubit => Some(1),
            _ => match self.types().as_array(t) {
                Some((Ty::Qubit, n)) => Some(n),
                _ => None,
            },
        }
    }

    /// What measuring the quantum structure `t` gives: a tuple of its fields
    /// in order, each qubit-bearing one measured and each classical one as
    /// it is. The error is the name of a field holding a quantum
    /// enumeration, which is not measured whole.
    fn measured_struct(&mut self, t: Ty) -> Result<Ty, String> {
        // a `quint` among the fields gives its number
        if let Some(n) = self.quint_width(t) {
            return Ok(Ty::Int(quint_measured(n)));
        }
        let Ty::Adt(id) = t else { return Ok(t) };
        let fields: Vec<(crate::intern::Symbol, Ty)> = self.adt(id).fields().iter().map(|f| (f.name, f.ty)).collect();
        let mut out = Vec::with_capacity(fields.len());
        for (name, ft) in fields {
            let ft = self.shallow(ft);
            let m = if !self.types().is_quantum(ft) {
                ft
            } else if ft == Ty::Qubit {
                Ty::Bool
            } else if let Some((elem, n)) = self.types().as_array(ft) {
                let e = if elem == Ty::Qubit { Ty::Bool } else { self.measured_struct(elem)? };
                self.types_mut().array(e, n)
            } else if matches!(ft, Ty::Adt(f) if self.adt(f).is_struct()) {
                self.measured_struct(ft)?
            } else {
                return Err(self.name(name).to_owned());
            };
            out.push(m);
        }
        Ok(self.types_mut().tuple(out))
    }

    /// An operation on quantum state written in a classical unit.
    pub(super) fn quantum_in_classical(&mut self, what: &str, span: Span) -> Expr {
        self.report(
            Diagnostic::new(Code::Eu02)
                .with_message(format!("only a quantum unit can {what}"))
                .at(span)
                .with_note(
                    "a unit is classical or quantum in its entirety, and a classical unit holds no \
                     quantum state to act on",
                )
                .with_help("move this into a unit that begins with `#unit quantum`"),
        );
        Checker::error(span)
    }

    /// `measure e`.
    pub fn measure(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Expr {
        if self.cx.quantum
            && let Some(e) = self.lookup(operand, span)
        {
            return e;
        }
        let o = self.expr(operand);
        let t = self.shallow(o.ty);
        if t != Ty::Never && !self.types().is_quantum(t) {
            return self.classical_measure(o, false, span);
        }
        if !self.cx.quantum {
            return self.quantum_in_classical("measure", span);
        }
        self.measured(o, false, span)
    }

    /// `match measure e { … }`: `e` measured, and the arms matched against
    /// what was observed. A quantum enumeration's arms match its variants,
    /// each binding the payload that survives the measurement.
    pub fn match_measure(&mut self, scrutinee: &Spanned<ast::Expr>, arms: &[ast::MatchArm], span: Span) -> Expr {
        let o = self.expr(scrutinee);
        let t = self.shallow(o.ty);
        if t != Ty::Never && !self.types().is_quantum(t) {
            let m = self.classical_measure(o, true, scrutinee.span);
            if self.shallow(m.ty) == Ty::Never {
                return Checker::error(span);
            }
            return self.match_checked(m, arms, span);
        }
        if !self.cx.quantum {
            return self.quantum_in_classical("measure", span);
        }
        let m = self.measured(o, true, scrutinee.span);
        if self.shallow(m.ty) == Ty::Never {
            return Checker::error(span);
        }
        self.match_checked(m, arms, span)
    }

    /// `measure e` on classical data, in a unit of either kind: `e` itself,
    /// in the shape measuring the quantum data it stands for in a unit of the
    /// other kind would give: a structure as a tuple of its fields, each
    /// structure among them a tuple too. It is a warning (`ES24`), since there
    /// is nothing to observe. An enumeration is taken only by `match measure`,
    /// which `variants` says this is, as a quantum one is.
    fn classical_measure(&mut self, o: Expr, variants: bool, span: Span) -> Expr {
        let t = self.settled(o.ty);
        let fits = match t {
            Ty::Bool => true,
            Ty::Int(i) => !i.signed,
            Ty::Adt(id) => self.adt(id).is_struct() || variants,
            _ => self.bool_array(t).is_some() || self.types().as_growable(t) == Some(Ty::Bool),
        };
        if !fits {
            let what = self.describe(t);
            let mut d = Diagnostic::new(Code::Es06)
                .with_message(format!(
                    "`measure` takes qubits, or classical data standing for them: a `bool`, an unsigned \
                     integer, an array of `bool`s or a structure; not {what}"
                ))
                .at(o.span);
            if matches!(t, Ty::Adt(_)) {
                d = d.with_note("an enumeration is measured only by `match measure`, as a quantum one is");
            }
            self.report(d);
            return Checker::error(span);
        }
        self.report(
            Diagnostic::new(Code::Es24)
                .with_message("`measure` on classical data gives it back as it is")
                .at(span)
                .with_note(
                    "there is nothing to observe; it is accepted so that a routine can serve a unit of \
                     either kind, where its data holds qubits in the other",
                )
                .with_help("measure only in the quantum unit, under `#if __UNIT_KIND__ == __QUANTUM__`"),
        );
        self.measured_shape(o)
    }

    /// `o`, a structure made a tuple of its fields, recursively, as measuring
    /// gives a quantum structure; anything else as it is. `o` is evaluated
    /// once, and its fields are moved out of it.
    fn measured_shape(&mut self, o: Expr) -> Expr {
        let t = self.settled(o.ty);
        let Ty::Adt(id) = t else { return o };
        if !self.adt(id).is_struct() {
            return o;
        }
        let span = o.span;
        let (stmts, base) = if super::step::evaluates_purely(&o) {
            (Vec::new(), o)
        } else {
            let held = self.new_local(crate::intern::Symbol::EMPTY, t, false, span);
            (vec![crate::tir::Stmt::Let { local: held, init: Some(o) }], Expr::local(held, t, span))
        };
        let field_types: Vec<Ty> = self.adt(id).fields().iter().map(|f| f.ty).collect();
        let mut parts = Vec::with_capacity(field_types.len());
        let mut types = Vec::with_capacity(field_types.len());
        for (i, fty) in field_types.into_iter().enumerate() {
            let part = Expr {
                kind: ExprKind::Field {
                    base: Box::new(base.clone()),
                    field: i as u32,
                },
                ty: fty,
                span,
            };
            let part = self.measured_shape(part);
            types.push(part.ty);
            parts.push((i as u32, part));
        }
        let ty = self.types_mut().tuple(types);
        let tuple = Expr {
            kind: ExprKind::StructLit { fields: parts },
            ty,
            span,
        };
        if stmts.is_empty() {
            return tuple;
        }
        Expr {
            ty,
            kind: ExprKind::Block(Box::new(crate::tir::Block {
                stmts,
                value: Some(Box::new(tuple)),
                ty,
                span,
            })),
            span,
        }
    }

    /// The measurement of `o`. A quantum enumeration is measured only by
    /// `match measure`, which `variants` says this is.
    pub(super) fn measured(&mut self, o: Expr, variants: bool, span: Span) -> Expr {
        let t = self.shallow(o.ty);
        // a `quint<N>` gives the smallest unsigned integer of at least N bits
        if let Some(n) = self.quint_width(t) {
            let targs = vec![Arg::Const(i128::from(n)), Arg::Type(Ty::Int(quint_measured(n)))];
            return self.library_call("gates", "__measure_quint", targs, vec![o], span);
        }
        let ty = match t {
            Ty::Never => return Checker::error(span),
            Ty::Qubit => Ty::Bool,
            Ty::Adt(id) if variants && self.types().is_quantum(t) && !self.adt(id).is_struct() => t,
            Ty::Adt(id) if self.types().is_quantum(t) && self.adt(id).is_struct() => match self.measured_struct(t) {
                Ok(ty) => ty,
                Err(field) => {
                    let what = self.describe(t);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`measure` cannot measure {what}: it holds a quantum enumeration"))
                            .at(o.span)
                            .with_note(
                                "measuring a structure measures each field; a quantum enumeration's variant \
                                 is read only by matching on it",
                            )
                            .with_help(format!("measure the field `{field}` with `match measure`, and the rest field by field")),
                    );
                    return Checker::error(span);
                }
            },
            _ => match self.types().as_array(t) {
                Some((Ty::Qubit, n)) => self.types_mut().array(Ty::Bool, n),
                // an array of `quint`s gives their numbers
                Some((elem, n)) if let Some(w) = self.quint_width(elem) => {
                    self.types_mut().array(Ty::Int(quint_measured(w)), n)
                }
                _ => match self.types().as_growable(t) {
                    Some(Ty::Qubit) => self.types_mut().growable(Ty::Bool),
                    _ => {
                        if let Ty::Adt(_) = t
                            && self.types().is_quantum(t)
                        {
                            return self.unsupported(
                                span,
                                "measuring a quantum enumeration outside `match measure`",
                            );
                        }
                        let what = self.describe(t);
                        let help = match self.types().as_ref(t) {
                            Some((_, target)) if self.types().is_quantum(target) => {
                                "a reference names qubits without owning them; measure the register \
                                 itself, which gives it up"
                            }
                            _ => "measure a `qubit` or a register `[qubit; N]`",
                        };
                        self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!("`measure` takes qubits, not {what}"))
                                .at(o.span)
                                .with_note(
                                    "measuring consumes the qubits measured and gives what was \
                                     observed: a `bool` for each qubit",
                                )
                                .with_help(help),
                        );
                        return Checker::error(span);
                    }
                },
            },
        };
        Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Measure,
                args: vec![o],
            },
            ty,
            span,
        }
    }

    /// Whether a condition of type `t` is quantum: a qubit, or a handle to
    /// one.
    pub fn is_quantum_condition(&self, t: Ty) -> bool {
        t == Ty::Qubit || self.types().as_ref(t).is_some_and(|(_, x)| x == Ty::Qubit)
    }

    /// The condition of an `if` or `while`: a `bool`, or, in a quantum
    /// unit, a *quantum condition* (a qubit, a handle to one, or `!`, `&&`
    /// and `||` over those and `bool`s). A quantum condition is given type
    /// `qubit`; an `if` on one applies its branches controlled on it.
    pub(super) fn if_condition(&mut self, e: &Spanned<ast::Expr>, role: &str) -> Expr {
        if !self.cx.quantum {
            return self.condition(e, role);
        }
        let span = e.span;
        match &e.node {
            ast::Expr::Paren(inner) => self.if_condition(inner, role),
            ast::Expr::Unary {
                op: ast::UnOp::Not,
                operand,
            } => {
                let o = self.if_condition(operand, role);
                let ty = if self.is_quantum_condition(o.ty) { Ty::Qubit } else { self.shallow(o.ty) };
                Expr {
                    kind: ExprKind::Unary {
                        op: crate::tir::UnOp::Not,
                        operand: Box::new(o),
                    },
                    ty,
                    span,
                }
            }
            ast::Expr::Binary {
                op: op @ (ast::BinOp::AndAnd | ast::BinOp::OrOr),
                lhs,
                rhs,
            } => {
                let l = self.if_condition(lhs, role);
                let r = self.if_condition(rhs, role);
                let quantum = self.is_quantum_condition(l.ty) || self.is_quantum_condition(r.ty);
                let op = if *op == ast::BinOp::AndAnd { LogicalOp::And } else { LogicalOp::Or };
                Expr {
                    kind: ExprKind::Logical {
                        op,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty: if quantum { Ty::Qubit } else { Ty::Bool },
                    span,
                }
            }
            _ => {
                let c = self.expr(e);
                if self.is_quantum_condition(self.shallow(c.ty)) {
                    c
                } else {
                    self.condition_checked(c, role)
                }
            }
        }
    }

    /// `lift e`.
    pub fn lift(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Expr {
        if !self.cx.quantum {
            return self.quantum_in_classical("lift an outcome", span);
        }
        let o = self.expr(operand);
        let ty = o.ty;
        if self.types().is_quantum(self.shallow(ty)) {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("`lift` takes a measured value, not qubits")
                    .at(operand.span)
                    .with_note(
                        "`lift` makes a value observed while the circuit runs available to the \
                         structure of the rest of it",
                    )
                    .with_help("measure the qubits first, and lift what the measurement gives"),
            );
            return Checker::error(span);
        }
        Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Lift,
                args: vec![o],
            },
            ty,
            span,
        }
    }

    /// `forget e;`, or `None` when it is ill-formed.
    pub fn forget(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Option<Expr> {
        if !self.cx.quantum {
            self.quantum_in_classical("forget quantum values", span);
            return None;
        }
        let o = self.expr(operand);
        let t = self.shallow(o.ty);
        if t == Ty::Never {
            return None;
        }
        if !self.types().is_quantum(t) {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`forget` discards qubits, not {what}"))
                    .at(operand.span)
                    .with_note(
                        "a classical value that is no longer needed is simply dropped; `forget` \
                         exists because a quantum one may not be",
                    )
                    .with_help("remove the `forget`"),
            );
            return None;
        }
        Some(Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Forget,
                args: vec![o],
            },
            ty: Ty::Void,
            span,
        })
    }

    /// `a ** b` where `a` is a register: one register of both.
    pub fn register_tensor(&mut self, l: Expr, rhs: &Spanned<ast::Expr>, span: Span) -> Expr {
        let n = self.register_width(self.shallow(l.ty)).expect("the left operand is a register");
        let r = self.expr(rhs);
        let rt = self.shallow(r.ty);
        let Some(m) = self.register_width(rt) else {
            if rt != Ty::Never {
                let what = self.describe(rt);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("a register can be joined only to another register, not to {what}"))
                        .at(rhs.span)
                        .with_note("`**` on registers makes one register holding the qubits of both")
                        .with_help("join a `qubit` or a `[qubit; N]`"),
                );
            }
            return Checker::error(span);
        };
        let ty = self.types_mut().array(Ty::Qubit, n + m);
        Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Tensor,
                args: vec![l, r],
            },
            ty,
            span,
        }
    }
}

impl Checker<'_, '_> {
    /// The number of qubits a handle names: one for `*qubit`, `N` for
    /// `*[qubit; N]`.
    pub fn handle_width(&self, t: Ty) -> Option<u64> {
        let (_, target) = self.types().as_ref(self.shallow(t))?;
        self.register_width(target)
    }

    /// `rz(k, q)`, `gphase(k)` and `apply(u, r)`, whose operands are checked
    /// here rather than against one signature.
    pub(super) fn quantum_operation(&mut self, which: Intrinsic, mut args: Vec<Expr>, span: Span) -> Expr {
        let want = match which {
            Intrinsic::Gate(Gate::Rz) | Intrinsic::Apply => 2,
            _ => 1,
        };
        if args.len() != want {
            let what = match which {
                Intrinsic::Gate(Gate::Rz) => "a phase and a handle to a qubit",
                Intrinsic::Apply => "a matrix and a handle to the register it acts on",
                _ => "a phase",
            };
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`{}` takes {what}", which.name()))
                    .at(span),
            );
            return Checker::error(span);
        }
        if args.iter().any(|a| self.shallow(a.ty) == Ty::Never) {
            return Checker::error(span);
        }
        match which {
            Intrinsic::Gate(_) => {
                if self.core_exact(args[0].ty, "phase").is_none() {
                    let what = self.describe(args[0].ty);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("a rotation is by a `phase<N>`, not {what}"))
                            .at(args[0].span)
                            .with_note(
                                "a gate's angle is exact: a whole number of N-th parts of a turn, \
                                 which is what `phase<N>` holds",
                            )
                            .with_help("write the angle as `phase::<N>::of(k)`"),
                    );
                    return Checker::error(span);
                }
                if which == Intrinsic::Gate(Gate::Rz) && self.handle_width(args[1].ty) != Some(1) {
                    let what = self.describe(args[1].ty);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`rz` rotates one qubit, named by a `*qubit`, not {what}"))
                            .at(args[1].span)
                            .with_help("pass a handle, as in `rz(k, &q)`"),
                    );
                    return Checker::error(span);
                }
            }
            _ => {
                if !self.apply_operands(&args) {
                    return Checker::error(span);
                }
                // whether the operator is marked as vouching for its matrices
                let trusted = Expr::constant(crate::tir::Value::Bool(self.trusted), Ty::Bool, span);
                args.push(trusted);
            }
        }
        Expr::intrinsic(which, args, Ty::Void, span)
    }

    /// `adjoint(f)` and `controlled(f)`: operators made of the operator `f`.
    /// `adjoint(f)` has `f`'s signature; `controlled(f)` takes a handle to
    /// the control qubit first.
    pub(super) fn functor(&mut self, which: Intrinsic, args: Vec<Expr>, span: Span) -> Expr {
        let [f] = <[Expr; 1]>::try_from(args).unwrap_or_else(|a| {
            let n = a.len();
            [Expr::constant(crate::tir::Value::Void, Ty::Never, if n == 0 { span } else { a[0].span })]
        });
        let t = self.shallow(f.ty);
        if t == Ty::Never {
            return Checker::error(span);
        }
        let Some((params, ret)) = self.types().as_sig(t).filter(|_| matches!(t, Ty::Fn(_))).map(|(p, r)| (p.to_vec(), r)) else {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{}` takes an operator, not {what}", which.name()))
                    .at(f.span)
                    .with_help(format!("name an operator, as in `{}(h)`", which.name())),
            );
            return Checker::error(span);
        };
        let ty = match which {
            Intrinsic::Controlled => {
                let control = self.types_mut().reference(crate::tir::Access::Write, Ty::Qubit);
                let params: Vec<Ty> = std::iter::once(control).chain(params).collect();
                self.types_mut().function(params, ret)
            }
            _ => t,
        };
        Expr::intrinsic(which, vec![f], ty, span)
    }

    /// `f.then(g)`: the operator applying `f` and then `g`, which act in place
    /// on the same arguments, so each is of one signature and takes handles.
    pub(super) fn then_call(&mut self, f: Expr, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let [g] = args else {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`then` takes the operator applied second")
                    .at(span)
                    .with_help("write `f.then(g)`"),
            );
            return Checker::error(span);
        };

        // one signature
        let g = self.expr(g);
        let (ft, gt) = (self.shallow(f.ty), self.shallow(g.ty));
        if gt == Ty::Never {
            return Checker::error(span);
        }
        if ft != gt {
            let (a, b) = (self.describe(ft), self.describe(gt));
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`then` joins two operators of one signature, but these are {a} and {b}"))
                    .at(g.span)
                    .with_note("both are applied to the same arguments, one after the other"),
            );
            return Checker::error(span);
        }

        // taking handles, so both act on the same qubits
        let owned = self
            .types()
            .as_sig(ft)
            .is_some_and(|(ps, _)| ps.iter().any(|&p| self.types().is_quantum(p)));
        if owned {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("`then` joins operators that act in place, but these take qubits by value")
                    .at(f.span)
                    .with_note("both are applied to the same arguments, which a value taken by the first would no longer be")
                    .with_help("take the qubits as handles, `*qubit` or `*[qubit; N]`"),
            );
            return Checker::error(span);
        }
        Expr::intrinsic(Intrinsic::Then, vec![f, g], ft, span)
    }

    /// `replay f(args)`: a second preparation of what the monic operator `f`
    /// prepares from generation-time arguments.
    pub fn replay(&mut self, call: &Spanned<ast::Expr>, span: Span) -> Expr {
        if !matches!(call.node, ast::Expr::Call { .. }) {
            self.expr(call);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("`replay` takes a call of an operator")
                    .at(call.span)
                    .with_help("write `replay f(args)`"),
            );
            return Checker::error(span);
        }
        let c = self.expr(call);
        let ty = c.ty;
        Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Replay,
                args: vec![c],
            },
            ty,
            span,
        }
    }

    /// Checks `apply(u, r)`: `u` an exact matrix of `2^k` rows and columns,
    /// and `r` a handle to `k` qubits.
    fn apply_operands(&mut self, args: &[Expr]) -> bool {
        // a handle, and a matrix
        let (m, r) = (&args[0], &args[1]);
        let Some(k) = self.handle_width(r.ty) else {
            let what = self.describe(r.ty);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`apply` acts on the register a handle names, not on {what}"))
                    .at(r.span)
                    .with_help("pass a handle, as in `apply(u, &r)`"),
            );
            return false;
        };
        let shape = self.matrix_shape(m.ty);
        let Some((elem, rows, cols)) = shape else {
            let what = self.describe(m.ty);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`apply` applies a matrix, `mat<cyclo<N>, R, C>`, not {what}"))
                    .at(m.span),
            );
            return false;
        };

        // exact entries
        let floating = matches!(self.shallow(elem), Ty::Float(_))
            || self.core_exact(elem, "cplx").is_some_and(|_| self.cplx_is_floating(elem));
        if floating {
            self.report(
                Diagnostic::new(Code::Eq12)
                    .with_message("`apply` needs an exact matrix, but this one holds floating-point numbers")
                    .at(m.span)
                    .with_note(
                        "what a gate does is checked exactly, and a floating-point entry is already \
                         rounded; the gate it stands for is not the one written",
                    )
                    .with_help("write the entries as `cyclo<N>` values, with `i`, `isq2` and `w(k, N)`"),
            );
            return false;
        }
        if self.core_exact(elem, "cyclo").is_none() {
            let what = self.describe(elem);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`apply` needs a matrix of `cyclo<N>` entries, not of {what}"))
                    .at(m.span),
            );
            return false;
        }

        // square, with a row for each basis state of the qubits
        let side = 1u64.checked_shl(u32::try_from(k).unwrap_or(u32::MAX)).unwrap_or(0);
        if rows != side || cols != side {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "a gate on {k} qubits is a {side}-by-{side} matrix, but this one is {rows}-by-{cols}"
                    ))
                    .at(m.span)
                    .with_note("a matrix acting on k qubits has a row and a column for each of their 2^k basis states"),
            );
            return false;
        }
        true
    }

    /// The element type, rows and columns of a `mat`.
    fn matrix_shape(&mut self, t: Ty) -> Option<(Ty, u64, u64)> {
        self.core_exact(t, "mat")?;
        let Ty::Adt(id) = self.shallow(t) else { return None };
        match self.adt(id).args.as_slice() {
            [Arg::Type(e), Arg::Const(r), Arg::Const(c)] => Some((*e, u64::try_from(*r).ok()?, u64::try_from(*c).ok()?)),
            _ => None,
        }
    }

    /// Whether a `cplx<T>` has floating parts.
    fn cplx_is_floating(&mut self, t: Ty) -> bool {
        let Ty::Adt(id) = self.shallow(t) else { return false };
        matches!(self.adt(id).args.first(), Some(Arg::Type(Ty::Float(_))))
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{analyzed_quantum, quantum_codes};
    use crate::tir::Ty;

    #[test]
    fn measuring_gives_a_bool_for_each_qubit() {
        let (u, d) = analyzed_quantum("fn f(q: qubit) -> bool { measure q }\nfn g(r: [qubit; 3]) -> [bool; 3] { measure r }");
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(u.fns[0].ret, Ty::Bool);
    }

    #[test]
    fn a_prepared_register_is_as_wide_as_its_kets() {
        let (u, d) = analyzed_quantum("fn f() -> [qubit; 3] { prep |010> }");
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(u.types.as_array(u.fns[0].ret), Some((Ty::Qubit, 3)));
        assert_eq!(quantum_codes("fn f() -> [qubit; 2] { prep |0> + |11> }"), [Code::Es06]);
    }

    #[test]
    fn joined_registers_add_their_widths() {
        let (u, d) = analyzed_quantum("fn f(a: qubit, b: [qubit; 2]) -> [qubit; 3] { a ** b }");
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(u.types.as_array(u.fns[0].ret), Some((Ty::Qubit, 3)));
    }

    #[test]
    fn a_register_joined_to_something_else_is_refused() {
        assert_eq!(quantum_codes("fn f(a: qubit) -> bool { let r = a ** 3; true }"), [Code::Es06]);
    }
}
