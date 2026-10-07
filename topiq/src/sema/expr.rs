//! Checking expressions: literals, operators, conversions and control flow.
//!
//! Every expression is resolved and typed here, and turned into its
//! [`crate::tir`] form. Paths live in [`super::path`], calls in
//! [`super::call`], places and references in [`super::place`], structure,
//! variant and array values in [`super::aggregate`], and `match` in
//! [`super::pattern`]. A few forms are rewritten on the way:
//!
//! - **`x op= y` becomes `x = x op y`**, with the read of `x` evaluated before
//!   `y`, so code generation never sees compound assignment.
//! - **`-5` is one negative literal**, not a negation of `5`. That is what lets
//!   `let x: i8 = -128;` stand: `128` alone is not an `i8`, but `-128` is.
//! - **`for i in a..b` becomes a range loop** with its own node, because a
//!   range reaching its type's maximum must stop without computing a successor
//!   that would overflow.
//!
//! # Operand rules
//!
//! No operator converts between types. Both operands of `+`, `==` and the
//! rest must already have the same type; the one exception is the shift
//! count, which may be any integer type, since it is a count rather than a
//! value of the shifted type. Arithmetic, `==` and the orderings apply to the
//! floating types as well, where they follow IEEE 754 and never abort; `&`,
//! `|`, `^`, `!` and the shifts do not, since a floating value has no bits a
//! program may name. `&`, `|` and `^` also apply to `bool`; `==`, `!=` and the
//! orderings also compare `char`s, by their Unicode scalar values, and two
//! references, by what they refer to. Structures and enumerations have no
//! built-in operators: an operator on one calls the operator method its type
//! defines, such as `$add` or `$eq` ([`super::method`]).
//!
//! Two kinds of operand are handled before those rules. A number written as
//! a literal beside an exact scalar (`x * 2`, `1/3` where a `frac` is
//! wanted) is that exact number; and `*` between a matrix or vector and a
//! vector or number is `apply` or `scale` ([`super::exact`]).

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Arg, Arm, BinOp, Block, Expr, ExprKind, FloatTy, IntTy, LogicalOp, Pat, PatKind, Ty, UnOp, Value};

use super::arith;
use super::body::{Checker, LoopFrame, LoopKind};

/// How the checker treats a binary operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OpClass {
    /// `+ - * / %`: two numbers of one type, integer or floating.
    Arithmetic,
    /// `<< >>`: an integer, and a count of any integer type.
    Shift,
    /// `& | ^`: integers of one type, or two bools.
    Bitwise,
    /// `== !=`: two numbers, bools, chars or references of one type.
    Equality,
    /// `< <= > >=`: two numbers or chars of one type.
    Ordering,
}

fn classify(op: BinOp) -> OpClass {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => OpClass::Arithmetic,
        BinOp::Shl | BinOp::Shr => OpClass::Shift,
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => OpClass::Bitwise,
        BinOp::Eq | BinOp::Ne => OpClass::Equality,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => OpClass::Ordering,
    }
}

/// The analysed form of a syntax-tree operator.
pub(super) enum Mapped {
    Binary(BinOp),
    Logical(LogicalOp),
    Tensor,
}

pub(super) fn map_op(op: ast::BinOp) -> Mapped {
    use ast::BinOp as A;
    Mapped::Binary(match op {
        A::Add => BinOp::Add,
        A::Sub => BinOp::Sub,
        A::Mul => BinOp::Mul,
        A::Div => BinOp::Div,
        A::Rem => BinOp::Rem,
        A::Shl => BinOp::Shl,
        A::Shr => BinOp::Shr,
        A::And => BinOp::BitAnd,
        A::Or => BinOp::BitOr,
        A::Xor => BinOp::BitXor,
        A::Eq => BinOp::Eq,
        A::Ne => BinOp::Ne,
        A::Lt => BinOp::Lt,
        A::Le => BinOp::Le,
        A::Gt => BinOp::Gt,
        A::Ge => BinOp::Ge,
        A::AndAnd => return Mapped::Logical(LogicalOp::And),
        A::OrOr => return Mapped::Logical(LogicalOp::Or),
        A::Tensor => return Mapped::Tensor,
    })
}

impl Checker<'_, '_> {
    /// Checks an expression.
    pub fn expr(&mut self, e: &Spanned<ast::Expr>) -> Expr {
        let span = e.span;
        match &e.node {
            ast::Expr::Int { raw, base, suffix } => self.int_literal(*raw, base.radix(), *suffix, false, span),
            ast::Expr::Bool(b) => Expr::constant(Value::Bool(*b), Ty::Bool, span),
            ast::Expr::Char(c) => Expr::constant(Value::Char(*c), Ty::Char, span),
            ast::Expr::Str { value, .. } => {
                let ty = self.types_mut().string();
                Expr::constant(Value::Str(*value), ty, span)
            }
            ast::Expr::Float { raw, suffix } => self.float_literal(*raw, *suffix, span),
            ast::Expr::Path { .. } if self.cx.quantum && let Some(q) = self.qmap_as_value(e) => q,
            ast::Expr::Path { path, args } => self.path_value(path, args, span),
            ast::Expr::Paren(inner) => self.expr(inner),
            ast::Expr::Tuple(items) => self.tuple_lit(items, span),
            ast::Expr::Array(items) => self.array_lit(items, span),
            ast::Expr::ArrayRepeat { elem, len } => self.array_repeat(elem, len, span),
            ast::Expr::StructLit { path, fields } => self.struct_lit(path, fields, span),
            ast::Expr::Block(b) => {
                let b = self.block(b, span);
                Expr {
                    ty: b.ty,
                    kind: ExprKind::Block(Box::new(b)),
                    span,
                }
            }
            ast::Expr::If { cond, then, els } => self.if_expr(cond, then, els.as_deref(), span),
            ast::Expr::Match {
                measuring: true,
                scrutinee,
                arms,
            } => self.match_measure(scrutinee, arms, span),
            ast::Expr::Match { scrutinee, arms, .. } => self.match_expr(scrutinee, arms, span),
            ast::Expr::While { cond, body } => self.while_expr(cond, body, span),
            ast::Expr::Loop { body } => self.loop_expr(body, span),
            ast::Expr::For {
                pattern,
                iter,
                body,
            } => self.for_loop(pattern, iter, body, span),
            ast::Expr::Closure {
                params,
                ret,
                captures,
                body,
            } => self.closure(params, ret.as_deref(), captures, body, span),
            ast::Expr::Measure(operand) => self.measure(operand, span),
            ast::Expr::Lift(operand) => self.lift(operand, span),
            ast::Expr::Prep(arg) => self.prep(arg, span),
            ast::Expr::Query { map, index } if self.cx.quantum => self.query(map, index, span),
            ast::Expr::Replay(call) if self.cx.quantum => self.replay(call, span),
            ast::Expr::Query { .. } | ast::Expr::Replay(_) => {
                self.report(
                    Diagnostic::new(Code::Eu02)
                        .with_message(format!(
                            "only a quantum unit can {}",
                            if matches!(e.node, ast::Expr::Query { .. }) { "query a map locale" } else { "replay an operator" }
                        ))
                        .at(span)
                        .with_note("a classical unit holds no quantum state to act on")
                        .with_help("move this into a unit that begins with `#unit quantum`"),
                );
                Checker::error(span)
            }
            ast::Expr::Macro { name, args } => self.macro_call(*name, args, span),
            ast::Expr::Unary { op, operand } => self.unary(*op, operand, span),
            ast::Expr::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, span),
            ast::Expr::Assign { op, place, value } => self.assign(*op, place, value, span),
            ast::Expr::Step { op, post, place } => self.step(*op, *post, place, span),
            ast::Expr::Range { start, end, inclusive } => {
                self.range_value(start.as_deref(), end.as_deref(), *inclusive, span)
            }
            ast::Expr::Call { callee, args } => self.call(callee, args, span),
            ast::Expr::Index { receiver, index } => self.index(receiver, index, span),
            ast::Expr::Field { receiver, name } => self.field(receiver, *name, span),
            ast::Expr::MethodCall {
                receiver,
                name,
                targs,
                args,
            } => self.method_call(receiver, *name, targs, args, span),
            ast::Expr::Try(e) => self.try_expr(e, span),
            ast::Expr::Cast { expr, ty } => self.cast(expr, ty, span),
        }
    }

    /// An integer literal, negated when `negative`, which is how `-128` is
    /// read as one value rather than as a negation of `128`.
    pub(super) fn int_literal(
        &mut self,
        raw: Symbol,
        radix: u32,
        suffix: Option<crate::lex::IntSuffix>,
        negative: bool,
        span: Span,
    ) -> Expr {
        let Some(v) = arith::literal_value(self.name(raw), radix) else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "this literal is larger than any integer type can hold; the largest is {}",
                        u64::MAX
                    ))
                    .at(span),
            );
            return Checker::error(span);
        };
        let v = if negative { -v } else { v };
        let (ty, placeholder) = match suffix {
            Some(s) => {
                let t = IntTy::from_name(s.text()).expect("every suffix names an integer type");
                (Ty::Int(t), t)
            }
            None => (self.infer.fresh(), IntTy::I64),
        };
        Expr::constant(Value::Int(v, placeholder), ty, span)
    }

    /// A floating literal, `1.0` or `2.5e-3f32`.
    ///
    /// Unlike an integer literal, no value is out of range: one too large for
    /// the type rounds to an infinity and one too small to zero, which is what
    /// IEEE 754 rounding says. The rounding happens once the type is settled,
    /// in [`super::infer`].
    pub(super) fn float_literal(
        &mut self,
        raw: crate::intern::Symbol,
        suffix: Option<crate::lex::FloatSuffix>,
        span: Span,
    ) -> Expr {
        let text = self.name(raw).replace('_', "");
        let Ok(v) = text.parse::<f64>() else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("this is not a floating value any implementation could read")
                    .at(span),
            );
            return Checker::error(span);
        };
        let (ty, placeholder) = match suffix {
            Some(s) => {
                let t = FloatTy::from_name(s.text()).expect("every suffix names a floating type");
                (Ty::Float(t), t)
            }
            None => (self.infer.fresh_float(), FloatTy::F64),
        };
        Expr::constant(Value::Float(placeholder.round(v), placeholder), ty, span)
    }

    fn if_expr(
        &mut self,
        cond: &Spanned<ast::Expr>,
        then: &ast::Block,
        els: Option<&Spanned<ast::Expr>>,
        span: Span,
    ) -> Expr {
        let cond = self.if_condition(cond, "the condition of an `if`");
        let then = self.block(then, span);
        let (els, ty) = match els {
            Some(e) => {
                let e = self.expr(e);
                let ty = match self.unify(then.ty, e.ty) {
                    Ok(t) => t,
                    Err(_) => {
                        let (a, b) = (self.describe(then.ty), self.describe(e.ty));
                        let d = Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "the two branches of this `if` give different types: {a} and {b}"
                            ))
                            .at_with(e.span, format!("this branch gives {b}"))
                            .with_note("whichever branch runs, the `if` has one type, so both must agree");
                        self.report(d);
                        Ty::Never
                    }
                };
                (Some(Box::new(e)), ty)
            }
            None => {
                if !matches!(self.shallow(then.ty), Ty::Void | Ty::Never) {
                    let what = self.describe(then.ty);
                    let d = Diagnostic::new(Code::Es06)
                        .with_message("an `if` without `else` cannot produce a value")
                        .at(span)
                        .with_note(format!(
                            "when the condition is false nothing runs, so there would be no {what} to give"
                        ))
                        .with_help("add an `else` branch, or end the block with `;` to discard the value");
                    self.report(d);
                }
                (None, Ty::Void)
            }
        };
        Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then: Box::new(then),
                els,
            },
            ty,
            span,
        }
    }

    /// A `bool` condition.
    pub(super) fn condition(&mut self, e: &Spanned<ast::Expr>, role: &str) -> Expr {
        let c = self.expr(e);
        self.condition_checked(c, role)
    }

    /// Checks that `c` is a `bool`, as the condition `role` names must be.
    pub(super) fn condition_checked(&mut self, c: Expr, role: &str) -> Expr {
        let found = self.shallow(c.ty);
        if self.is_quantum_condition(found) {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("{role} is a qubit, which only an `if`, a `match` or `^=` can test"))
                    .at(c.span)
                    .with_note(
                        "an `if` on a qubit applies its branch controlled on the qubit, and `t ^= c` \
                         flips `t` where `c` holds, without measuring it; nothing else can take a \
                         qubit as true or false",
                    )
                    .with_help("test it with `if`, or measure it and use the `bool` observed"),
            );
            return Checker::error(c.span);
        }
        if !matches!(found, Ty::Bool | Ty::Never) {
            let d = self
                .mismatch(c.span, role, Ty::Bool, found)
                .with_help("compare explicitly, as in `x != 0`; no value is treated as true or false on its own");
            self.report(d);
        }
        c
    }

    /// Checks a loop body, which must not produce a value.
    pub(super) fn loop_body_of(&mut self, body: &ast::Block, span: Span, what: &str) -> Block {
        let b = self.block(body, span);
        if !matches!(self.shallow(b.ty), Ty::Void | Ty::Never) {
            let at = b.value.as_ref().map_or(span, |v| v.span);
            let d = Diagnostic::new(Code::Es06)
                .with_message(format!("the body of {what} cannot produce a value"))
                .at(at)
                .with_help("end the body with `;` to discard the value");
            self.report(d);
        }
        b
    }

    fn while_expr(&mut self, cond: &Spanned<ast::Expr>, body: &ast::Block, span: Span) -> Expr {
        let cond = self.if_condition(cond, "the condition of a `while` loop");
        if self.is_quantum_condition(self.shallow(cond.ty)) {
            self.report(
                Diagnostic::new(Code::Eq03)
                    .with_message("unbounded quantum control: a `while` loop cannot repeat on a qubit")
                    .at(cond.span)
                    .with_note(
                        "a circuit is a fixed sequence of operations, so how many times a loop runs \
                         must be known when the circuit is generated; a qubit is not true or false \
                         until it is measured",
                    )
                    .with_help("loop a number of times known when the circuit is generated, with `for` over a constant range"),
            );
        }
        let id = self.fresh_loop();
        self.loops.push(LoopFrame {
            id,
            kind: LoopKind::While,
            value: None,
        });
        let body = self.loop_body_of(body, span, "a `while` loop");
        self.loops.pop();
        Expr {
            kind: ExprKind::While {
                id,
                cond: Box::new(cond),
                body: Box::new(body),
            },
            ty: Ty::Void,
            span,
        }
    }

    fn loop_expr(&mut self, body: &ast::Block, span: Span) -> Expr {
        let id = self.fresh_loop();
        self.loops.push(LoopFrame {
            id,
            kind: LoopKind::Loop,
            value: None,
        });
        let body = self.loop_body_of(body, span, "a `loop`");
        let frame = self.loops.pop().expect("pushed above");
        Expr {
            kind: ExprKind::Loop {
                id,
                body: Box::new(body),
            },
            // with no `break`, a `loop` never finishes
            ty: frame.value.unwrap_or(Ty::Never),
            span,
        }
    }

    fn unary(&mut self, op: ast::UnOp, operand: &Spanned<ast::Expr>, span: Span) -> Expr {
        let (op, method, text, wanted) = match op {
            ast::UnOp::Neg => (UnOp::Neg, "neg", "-", "a number"),
            ast::UnOp::Not => (UnOp::Not, "not", "!", "a bool or an integer"),
            ast::UnOp::Ref => return self.reference(operand, span),
            ast::UnOp::Deref => return self.dereference(operand, span),
        };
        if op == UnOp::Neg
            && let ast::Expr::Int { raw, base, suffix } = &operand.node
        {
            return self.int_literal(*raw, base.radix(), *suffix, true, span);
        }
        let o = self.expr(operand);
        let o = if self.cx.quantum {
            match self.quantum_unary(op, o, span) {
                Ok(done) => return done,
                Err(back) => back,
            }
        } else {
            o
        };
        if self.is_structured(o.ty) {
            return self.operator_call(method, text, o, None, span);
        }
        let t = self.shallow(o.ty);
        if op == UnOp::Not
            && let Some(n) = self.bool_array(t)
        {
            return self.library_call("core", "__bools_not", vec![Arg::Const(i128::from(n))], vec![o], span);
        }
        let fits = match op {
            UnOp::Neg => matches!(t, Ty::Int(_) | Ty::Infer(_) | Ty::Never) || self.is_floating(t),
            UnOp::Not => matches!(t, Ty::Int(_) | Ty::Infer(_) | Ty::Bool | Ty::Never),
        };
        if !fits {
            let d = self.operand_kind(o.span, text, "the operand", t, wanted);
            self.report(d);
            return Checker::error(span);
        }
        Expr {
            ty: o.ty,
            kind: ExprKind::Unary { op, operand: Box::new(o) },
            span,
        }
    }

    fn binary(
        &mut self,
        op: ast::BinOp,
        lhs: &Spanned<ast::Expr>,
        rhs: &Spanned<ast::Expr>,
        span: Span,
    ) -> Expr {
        match map_op(op) {
            Mapped::Tensor => {
                let l = self.expr(lhs);
                if self.register_width(self.shallow(l.ty)).is_some() {
                    return self.register_tensor(l, rhs, span);
                }
                if self.is_structured(l.ty) {
                    let r = self.expr(rhs);
                    return self.operator_call("tensor", "**", l, Some(r), span);
                }
                self.expr(rhs);
                if l.ty != Ty::Never {
                    let what = self.describe(l.ty);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`**` is not defined on {what}"))
                            .at(lhs.span)
                            .with_note(
                                "`**` is the tensor product: of vectors and matrices, as the \
                                 Kronecker product, of quantum registers, and of any type that \
                                 defines `$tensor`. It is not exponentiation, which Topiq does not have",
                            )
                            .with_help("for a power, multiply, or write a function that does"),
                    );
                }
                Checker::error(span)
            }
            Mapped::Logical(op) => {
                let role = format!("each operand of `{}`", op.text());
                let (l, r) = if self.cx.quantum {
                    let (l, r) = (self.expr(lhs), self.expr(rhs));
                    match self.quantum_logical(op, l, r, span) {
                        Ok(done) => return done,
                        Err(back) => {
                            let (l, r) = *back;
                            (self.condition_checked(l, &role), self.condition_checked(r, &role))
                        }
                    }
                } else {
                    (self.condition(lhs, &role), self.condition(rhs, &role))
                };
                Expr {
                    kind: ExprKind::Logical {
                        op,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty: Ty::Bool,
                    span,
                }
            }
            Mapped::Binary(op) => {
                // a number written beside an exact one is exact too
                if let Some(f) = self.literal_exact(&lhs.node) {
                    let r = self.expr(rhs);
                    if self.is_exact_number(r.ty)
                        && let Some(l) = self.exact_literal(f, r.ty, lhs.span)
                    {
                        return self.binary_checked(op, l, r, span);
                    }
                    let l = self.expr(lhs);
                    return self.binary_checked(op, l, r, span);
                }
                let l = self.expr(lhs);
                if self.is_exact_number(l.ty)
                    && let Some(f) = self.literal_exact(&rhs.node)
                    && let Some(r) = self.exact_literal(f, l.ty, rhs.span)
                {
                    return self.binary_checked(op, l, r, span);
                }
                let r = self.expr(rhs);
                self.binary_checked(op, l, r, span)
            }
        }
    }

    /// Types a binary operator over already-checked operands.
    pub(super) fn binary_checked(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        // qubits make a quantum condition, and registers combine qubit by
        // qubit
        let (l, r) = if self.cx.quantum {
            match self.quantum_binary(op, l, r, span) {
                Ok(done) => return done,
                Err(back) => *back,
            }
        } else {
            (l, r)
        };
        if self.linear_mismatch(op, &l, &r) {
            return Checker::error(span);
        }
        let Ok((l, r)) = self.cyclo_operands(l, r) else {
            return Checker::error(span);
        };
        // a vector or matrix times a vector or a number
        let (l, r) = if op == BinOp::Mul {
            match self.linear_product(l, r, span) {
                Ok(done) => return done,
                Err(back) => *back,
            }
        } else {
            (l, r)
        };
        // a `string` compared with text compares characters where they are;
        // otherwise text on the left of a `string` is one too, so `"a" + s`
        // is text
        let (l, r) = match self.text_equality(op, l, r, span) {
            Ok(done) => return done,
            Err(back) => *back,
        };
        let l = if self.is_string(r.ty) { self.text_for(l, r.ty) } else { l };
        if self.is_structured(l.ty) {
            return self.structured_binary(op, l, r, span);
        }
        let class = classify(op);
        let (lt, rt) = (self.shallow(l.ty), self.shallow(r.ty));
        // two `[bool; N]` combine element by element
        if class == OpClass::Bitwise
            && let Some(n) = self.bool_array(lt)
            && self.bool_array(rt) == Some(n)
        {
            let name = match op {
                BinOp::BitAnd => "__bools_and",
                BinOp::BitOr => "__bools_or",
                _ => "__bools_xor",
            };
            return self.library_call("core", name, vec![Arg::Const(i128::from(n))], vec![l, r], span);
        }
        let wanted = match class {
            OpClass::Arithmetic => "a number",
            OpClass::Shift => "an integer",
            OpClass::Bitwise => "an integer or a bool",
            OpClass::Equality => "a number, a bool or a char",
            OpClass::Ordering => "a number or a char",
        };
        let accepts = |c: &Self, t: Ty| {
            let is_int = matches!(t, Ty::Int(_) | Ty::Never) || (matches!(t, Ty::Infer(_)) && !c.is_floating(t));
            let is_float = matches!(t, Ty::Never) || c.is_floating(t);
            let is_bool = matches!(t, Ty::Bool | Ty::Never);
            let is_char = matches!(t, Ty::Char | Ty::Never);
            match class {
                OpClass::Arithmetic => is_int || is_float,
                OpClass::Shift => is_int,
                OpClass::Bitwise => is_int || is_bool,
                OpClass::Equality => is_int || is_float || is_bool || is_char,
                OpClass::Ordering => is_int || is_float || is_char,
            }
        };
        if class == OpClass::Equality && lt.is_reference() && rt.is_reference() {
            return self.reference_equality(op, l, r, span);
        }
        for (side, e, t) in [("the left operand", &l, lt), ("the right operand", &r, rt)] {
            if !accepts(self, t) {
                let mut d = self.operand_kind(e.span, op.text(), side, t, wanted);
                if t.is_reference() {
                    d = d.with_note(
                        "a reference is compared only with another reference, which asks whether \
                         both refer to the same place; to use the value it refers to, follow it \
                         with `*`",
                    );
                } else if matches!(t, Ty::Adt(_) | Ty::Array(_)) {
                    d = d.with_note(
                        "an operator whose left operand is a structure or enumeration calls the \
                         method its type defines for it, such as `$eq` or `$add`; a number on the \
                         left takes no structure on the right, and of the arrays only a \
                         `[bool; N]` has operators, `^`, `&`, `|` and `!` with another of its length",
                    );
                }
                self.report(d);
                return Checker::error(span);
            }
        }
        let ty = if class == OpClass::Shift {
            // the count need not match the value's type
            l.ty
        } else {
            match self.unify(l.ty, r.ty) {
                Ok(t) => {
                    if class == OpClass::Equality || class == OpClass::Ordering {
                        Ty::Bool
                    } else {
                        t
                    }
                }
                Err(_) => {
                    let (a, b) = (self.describe(l.ty), self.describe(r.ty));
                    let d = Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "both operands of `{}` must have the same type, but the left is {a} \
                             and the right is {b}",
                            op.text(),
                        ))
                        .at_with(r.span, format!("this is {b}"))
                        .also(l.span, format!("this is {a}"))
                        .with_help("convert one of them with `as`, for example `x as i64`");
                    self.report(d);
                    return Checker::error(span);
                }
            }
        };
        Expr {
            kind: ExprKind::Binary {
                op,
                lhs: Box::new(l),
                rhs: Box::new(r),
            },
            ty,
            span,
        }
    }

    /// A binary operator on a structure or enumeration: a call of its
    /// operator method. `!=` is the negation of `$eq`.
    fn structured_binary(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        let name = match op {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::Rem => "rem",
            BinOp::Shl => "shl",
            BinOp::Shr => "shr",
            BinOp::BitAnd => "and",
            BinOp::BitOr => "or",
            BinOp::BitXor => "xor",
            BinOp::Eq | BinOp::Ne if self.fieldless_enum(l.ty) => return self.variant_equality(op, l, r, span),
            BinOp::Eq | BinOp::Ne => "eq",
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => return self.ordering(op, l, r, span),
        };
        let call = self.operator_call(name, op.text(), l, Some(r), span);
        if call.ty == Ty::Never || !matches!(op, BinOp::Eq | BinOp::Ne) {
            return call;
        }
        let call = self.coerce(call, Ty::Bool, "the result of `$eq`");
        if op == BinOp::Eq {
            return call;
        }
        Expr {
            kind: ExprKind::Unary { op: UnOp::Not, operand: Box::new(call) },
            ty: Ty::Bool,
            span,
        }
    }

    /// Whether `t` is an enumeration whose variants carry nothing and which
    /// defines no `$eq` of its own: one whose `==` compares variants.
    fn fieldless_enum(&mut self, t: Ty) -> bool {
        let Ty::Adt(id) = self.shallow(t) else { return false };
        let def = self.types().adt(id);
        let fieldless = !def.is_struct() && !def.variants().is_empty() && def.variants().iter().all(|v| v.fields.is_empty());
        fieldless && self.operator_method(t, "eq").is_none()
    }

    /// `==` or `!=` on an enumeration whose variants carry nothing: whether
    /// the two are the same variant. Each operand is examined once, the left
    /// first, by a `match` giving its variant's index.
    fn variant_equality(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        let ty = self.shallow(l.ty);
        let r = self.coerce(r, ty, &format!("the right operand of `{}`", op.text()));
        let Ty::Adt(id) = ty else { unreachable!("an enumeration") };
        let count = self.types().adt(id).variants().len() as u32;
        let index = |e: Expr| {
            let at = e.span;
            // the last variant is what is left, as a circuit's choice made
            // while it runs needs an arm that takes anything
            let arms = (0..count)
                .map(|v| Arm {
                    pat: if v + 1 == count {
                        Pat::wild(ty, at)
                    } else {
                        Pat {
                            kind: PatKind::Variant { variant: v, fields: Vec::new() },
                            ty,
                            span: at,
                        }
                    },
                    body: Expr::constant(Value::Int(i128::from(v), IntTy::U32), Ty::Int(IntTy::U32), at),
                })
                .collect();
            Expr {
                kind: ExprKind::Match { scrutinee: Box::new(e), arms },
                ty: Ty::Int(IntTy::U32),
                span: at,
            }
        };
        let (a, b) = (index(l), index(r));
        self.binary_checked(op, a, b, span)
    }

    /// `<`, `<=`, `>` or `>=` on a structure or enumeration: its `$ord`,
    /// which says how the two compare as an `Ord`, tested for the answer the
    /// operator wants.
    fn ordering(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        let call = self.operator_call("ord", op.text(), l, Some(r), span);
        let t = self.settled(call.ty);
        if t == Ty::Never {
            return call;
        }
        let ord = match t {
            Ty::Adt(id) => {
                let def = self.adt(id).clone();
                (self.name(def.name) == "Ord" && def.origin.as_deref().is_none_or(|o| o == "core")).then_some(def)
            }
            _ => None,
        };
        let Some(def) = ord else {
            let what = self.describe(t);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`$ord` should say how two values compare, as an `Ord`, but gives {what}"))
                    .at(span),
            );
            return Checker::error(span);
        };
        let index = |name: &str| {
            def.variants()
                .iter()
                .position(|v| self.name(v.name) == name)
                .expect("`Ord` has `Lt`, `Eq` and `Gt`") as u32
        };
        // the one answer that decides: `a <= b` is "not greater"
        let (decisive, when) = match op {
            BinOp::Lt => (index("Lt"), true),
            BinOp::Le => (index("Gt"), false),
            BinOp::Gt => (index("Gt"), true),
            _ => (index("Lt"), false),
        };
        let arms = vec![
            crate::tir::Arm {
                pat: crate::tir::Pat::variant(decisive, Vec::new(), t, span),
                body: Expr::constant(Value::Bool(when), Ty::Bool, span),
            },
            crate::tir::Arm {
                pat: crate::tir::Pat::wild(t, span),
                body: Expr::constant(Value::Bool(!when), Ty::Bool, span),
            },
        ];
        Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(call),
                arms,
            },
            ty: Ty::Bool,
            span,
        }
    }

    /// Whether a type is floating, counting a literal that has not settled
    /// yet but can only become one.
    pub(super) fn is_floating(&self, t: Ty) -> bool {
        t.is_float() || self.infer.is_floating_variable(t)
    }

    /// `p == q` or `p != q` on two references, which asks whether they refer
    /// to the same place: the same address and, for slices, the same number
    /// of elements. Comparing what they refer to is written `*p == *q`.
    ///
    /// The two may differ in what they permit (a `*T` compares with a
    /// `*const T`) but not in what they refer to.
    fn reference_equality(&mut self, op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
        let target = |c: &Self, t: Ty| {
            let types = c.types();
            types
                .as_ref(t)
                .map(|(_, x)| (false, x))
                .or_else(|| types.as_slice(t).map(|(_, x)| (true, x)))
        };
        let (lt, rt) = (self.shallow(l.ty), self.shallow(r.ty));
        let (Some((ls, lx)), Some((rs, rx))) = (target(self, lt), target(self, rt)) else {
            unreachable!("both operands are references");
        };
        if ls != rs || self.unify(lx, rx).is_err() {
            let (a, b) = (self.describe(l.ty), self.describe(r.ty));
            let d = Diagnostic::new(Code::Es06)
                .with_message(format!(
                    "`{}` compares two references to the same type, but the left is {a} and the \
                     right is {b}",
                    op.text()
                ))
                .at_with(r.span, format!("this is {b}"))
                .also(l.span, format!("this is {a}"))
                .with_note("references to different types never refer to the same place");
            self.report(d);
            return Checker::error(span);
        }
        Expr {
            kind: ExprKind::Binary {
                op,
                lhs: Box::new(l),
                rhs: Box::new(r),
            },
            ty: Ty::Bool,
            span,
        }
    }

    /// `p as *T`: a reference converted to one of another type. Where the
    /// conversion always holds (to `*any`, to a prefix, to weaker access),
    /// it is simply made. Otherwise it is checked when the program runs,
    /// against the table the reference carries, giving `Some(p)` as a `*T`
    /// when what `p` refers to is a `T`, and `None` when it is not.
    fn reference_cast(&mut self, e: Expr, target: Ty, span: Span) -> Expr {
        let from = self.settled(e.ty);
        if from == Ty::Never {
            return Checker::error(span);
        }
        if !from.is_reference() || from.is_slice() {
            let what = self.describe(from);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`as *T` converts a reference, but this is {what}"))
                    .at(e.span)
                    .with_help("take a reference to the value first, with `&`"),
            );
            return Checker::error(span);
        }

        // a conversion that always holds is made now
        if self.unify(from, target).is_ok() {
            return e;
        }
        if self.converts_silently(from, target) {
            return self.coerce(e, target, "the value converted");
        }

        // any other is a downcast, checked when the program runs, which keeps the access
        let (Some((from_access, _)), Some((to_access, referent))) = (self.types().as_ref(from), self.types().as_ref(target)) else {
            return Checker::error(span);
        };
        if !from_access.converts_to(to_access) {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "a downcast keeps what the reference permits, and this one is `{}`, not `{}`",
                        from_access.prefix().trim(),
                        to_access.prefix().trim()
                    ))
                    .at(span)
                    .with_note("`*` converts to `*const`, but never the other way")
                    .with_help(format!("downcast to `{}T` instead", from_access.prefix())),
            );
            return Checker::error(span);
        }
        if referent == Ty::Void || !self.cx.need_table(referent, span) {
            return Checker::error(span);
        }
        let Some(opt) = self.opt_of(target, span) else {
            return Checker::error(span);
        };
        Expr::intrinsic(crate::tir::Intrinsic::Downcast(referent), vec![e], opt, span)
    }

    fn cast(&mut self, expr: &Spanned<ast::Expr>, ty: &Spanned<ast::Type>, span: Span) -> Expr {
        // `@embed("file") as T`: the document read as a `T`
        if let ast::Expr::Macro { name, args } = &expr.node
            && self.name(name.node) == "embed"
        {
            let target = self.resolve_type(ty).ty;
            return self.embed(args, Some(target), span);
        }
        let e = self.expr(expr);
        let target = self.resolve_type(ty).ty;
        if target == Ty::Never {
            return Checker::error(span);
        }
        // an exact scalar's approximation
        let e = match self.exact_cast(e, target, span) {
            Ok(done) => return done,
            Err(e) => e,
        };
        // `d as T`: the value a `dyn` holds, if it is a `T`
        if self.shallow(e.ty) == Ty::Dyn {
            if !self.cx.need_table(target, span) {
                return Checker::error(span);
            }
            let Some(opt) = self.opt_of(target, span) else {
                return Checker::error(span);
            };
            return Expr::intrinsic(crate::tir::Intrinsic::DynAs(target), vec![e], opt, span);
        }
        if target.is_reference() && !target.is_slice() {
            return self.reference_cast(e, target, span);
        }
        if !matches!(target, Ty::Int(_) | Ty::Float(_) | Ty::Char) {
            let what = self.describe(target);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "`as` converts between numbers and characters, and {what} is not one of those"
                    ))
                    .at(ty.span),
            );
            return Checker::error(span);
        }
        let from = self.shallow(e.ty);
        // a literal that has not settled is whichever family it began in
        let numeric = matches!(from, Ty::Int(_) | Ty::Float(_) | Ty::Infer(_));
        match (from, target) {
            (Ty::Never, _) => {}
            _ if numeric && matches!(target, Ty::Int(_) | Ty::Float(_)) => {}
            // a character converts to an integer by its scalar value, and an
            // integer back if it is one: not every number is
            (Ty::Char, Ty::Int(_)) | (Ty::Int(_) | Ty::Infer(_), Ty::Char) => {}
            (Ty::Char, Ty::Char) => {}
            (Ty::Float(_), Ty::Char) => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a floating value cannot become a `char` directly")
                        .at(e.span)
                        .with_help("convert to an integer first, as in `x as u32 as char`"),
                );
                return Checker::error(span);
            }
            (Ty::Bool, _) => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a bool cannot be converted with `as`")
                        .at(e.span)
                        .with_help("choose the values yourself, as in `if flag { 1 } else { 0 }`"),
                );
                return Checker::error(span);
            }
            (other, _) => {
                let d = self.operand_kind(e.span, "as", "the value converted", other, "a number or a char");
                self.report(d);
                return Checker::error(span);
            }
        }
        Expr {
            kind: ExprKind::Cast {
                expr: Box::new(e),
                to: target,
            },
            ty: target,
            span,
        }
    }

    /// An operand of a kind the operator does not accept.
    pub(super) fn operand_kind(&mut self, span: Span, op: &str, which: &str, found: Ty, wanted: &str) -> Diagnostic {
        let found = self.describe(found);
        Diagnostic::new(Code::Es06)
            .with_message(format!("{which} of `{op}` should be {wanted}, but it is {found}"))
            .at(span)
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::{ExprKind, IntTy, Stmt, Ty, Value};

    #[test]
    fn an_unconstrained_literal_is_an_i64() {
        let u = check("fn f() { let x = 5; }");
        assert_eq!(u.fns[0].locals[0].ty, Ty::Int(IntTy::I64));
    }

    #[test]
    fn an_unconstrained_floating_literal_is_an_f64() {
        let u = check("fn f() { let x = 1.5; }");
        assert_eq!(u.fns[0].locals[0].ty, Ty::Float(crate::tir::FloatTy::F64));
        let u = check("fn f() { let x: f32 = 1.5; let y = 2.5f32; }");
        for l in &u.fns[0].locals {
            assert_eq!(l.ty, Ty::Float(crate::tir::FloatTy::F32));
        }
    }

    #[test]
    fn a_floating_literal_rounds_to_its_type() {
        // 0.1 is not a value of either type; each takes its own nearest
        let u = check("fn f() { let a: f32 = 0.1; let b: f64 = 0.1; }");
        let value = |i: usize| match &u.fns[0].body.stmts[i] {
            Stmt::Let { init: Some(v), .. } => match v.kind {
                ExprKind::Const(Value::Float(x, _)) => x,
                ref other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        assert_eq!(value(0), f64::from(0.1f32));
        assert_eq!(value(1), 0.1f64);
    }

    #[test]
    fn the_two_families_of_number_never_mix() {
        assert_eq!(codes("fn f() -> f64 { 1 + 1.5 }"), [Code::Es06]);
        assert_eq!(codes("fn f(x: f64) -> f64 { x + 1 }"), [Code::Es06]);
        assert_eq!(codes("fn f() { let x = 1.5; let y: i32 = x; }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: f32, b: f64) -> bool { a == b }"), [Code::Es06]);
        check("fn f(x: f64) -> f64 { x + 1.0 }");
        check("fn f(x: f32) -> f32 { -x * 2.0 }");
    }

    #[test]
    fn floating_values_have_no_bitwise_operators_and_no_shifts() {
        assert_eq!(codes("fn f(a: f64, b: f64) -> f64 { a & b }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: f64, b: i32) -> f64 { a << b }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: f64) -> f64 { !a }"), [Code::Es06]);
        check("fn f(a: f64, b: f64) -> bool { a < b && a != b }");
        check("fn f(a: f64, b: f64) -> f64 { a % b }");
    }

    #[test]
    fn a_floating_match_needs_a_wildcard() {
        check("fn f(x: f64) -> i32 { match x { 0.0 => 1, _ => 2 } }");
        assert_eq!(codes("fn f(x: f64) -> i32 { match x { 0.0 => 1, 1.0 => 2 } }"), [Code::Es13]);
    }

    #[test]
    fn a_literal_takes_its_type_from_later_use() {
        let u = check("fn f() -> u8 { let x = 5; x }");
        assert_eq!(u.fns[0].locals[0].ty, Ty::Int(IntTy::U8));
    }

    #[test]
    fn a_literal_takes_its_type_from_the_other_operand() {
        let u = check("fn f(a: u16) -> u16 { 1 + a }");
        assert_eq!(u.fns[0].ret, Ty::Int(IntTy::U16));
    }

    #[test]
    fn a_negative_literal_is_one_value() {
        let u = check("fn f() -> i8 { -128 }");
        let v = u.fns[0].body.value.as_ref().unwrap();
        assert!(matches!(v.kind, ExprKind::Const(Value::Int(-128, IntTy::I8))), "{:?}", v.kind);
    }

    #[test]
    fn a_literal_that_does_not_fit_is_reported_with_the_range() {
        assert_eq!(codes("fn f() -> u8 { 300 }"), [Code::Es06]);
        assert_eq!(codes("fn f() -> i8 { -129 }"), [Code::Es06]);
        assert_eq!(codes("fn f() -> u8 { 300u8 }"), [Code::Es06]);
    }

    #[test]
    fn operands_of_different_types_are_refused() {
        assert_eq!(codes("fn f(a: i32, b: i64) -> i32 { a + b }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: u64, b: usize) -> bool { a == b }"), [Code::Es06]);
    }

    #[test]
    fn a_shift_count_may_be_of_any_integer_type() {
        check("fn f(a: u64, n: u8) -> u64 { a << n }");
    }

    #[test]
    fn arithmetic_is_for_integers_and_bitwise_also_for_bools() {
        assert_eq!(codes("fn f() -> bool { true + false }"), [Code::Es06]);
        check("fn f(a: bool, b: bool) -> bool { a & b | a ^ b }");
        assert_eq!(codes("fn f() -> bool { true < false }"), [Code::Es06]);
        check("fn f() -> bool { true == false }");
    }

    #[test]
    fn characters_compare_but_do_not_add() {
        check("fn f(c: char) -> bool { c >= 'a' && c <= 'z' && c != 'q' }");
        assert_eq!(codes("fn f(c: char) -> char { c + 'a' }"), [Code::Es06]);
    }

    #[test]
    fn structures_have_no_operators_yet() {
        assert_eq!(codes("struct P { x: i32 }\nfn f(a: P, b: P) -> bool { a == b }"), [Code::Es06]);
    }

    #[test]
    fn a_condition_must_be_a_bool() {
        assert_eq!(codes("fn f(x: i32) { if x { } }"), [Code::Es06]);
        assert_eq!(codes("fn f(x: i32) { while x { } }"), [Code::Es06]);
        assert_eq!(codes("fn f(x: i32) -> bool { x && true }"), [Code::Es06]);
    }

    #[test]
    fn both_branches_of_an_if_agree() {
        assert_eq!(codes("fn f(c: bool) -> i32 { if c { 1 } else { false } }"), [Code::Es06]);
        check("fn f(c: bool) -> i32 { if c { 1 } else if !c { 2 } else { 3 } }");
    }

    #[test]
    fn an_if_without_else_gives_no_value() {
        assert_eq!(codes("fn f(c: bool) -> i32 { if c { 1 } }"), [Code::Es06, Code::Es06]);
    }

    #[test]
    fn a_loop_takes_the_type_of_its_breaks() {
        let u = check("fn f() -> u32 { let i = 0; loop { i += 1; if i == 10 { break i; } } }");
        assert_eq!(u.fns[0].ret, Ty::Int(IntTy::U32));
    }

    #[test]
    fn a_for_loop_runs_over_an_integer_range() {
        check("fn f() -> u32 { let s = 0u32; for i in 1..=10 { s += i; } s }");
        assert_eq!(codes("fn f() { for i in 1..true { } }"), [Code::Es06]);
    }

    #[test]
    fn compound_assignment_becomes_a_read_and_a_store() {
        let u = check("fn f() { let x = 1; x += 2; }");
        let Stmt::Expr(e) = &u.fns[0].body.stmts[1] else {
            panic!()
        };
        let ExprKind::Assign { value, .. } = &e.kind else {
            panic!()
        };
        assert!(matches!(value.kind, ExprKind::Binary { .. }));
    }

    #[test]
    fn casts_convert_between_numbers_and_characters() {
        check("fn f(x: i64) -> u8 { x as u8 }");
        check("fn f(c: char) -> u32 { c as u32 }");
        check("fn f(x: u32) -> char { x as char }");
        check("fn f(x: i64) -> f64 { x as f64 }");
        check("fn f(x: f64) -> i32 { x as i32 }");
        check("fn f(x: f64) -> f32 { x as f32 }");
        assert_eq!(codes("fn f(x: bool) -> i32 { x as i32 }"), [Code::Es06]);
        assert_eq!(codes("fn f(x: i32) -> bool { x as bool }"), [Code::Es06]);
        assert_eq!(codes("fn f(x: f64) -> char { x as char }"), [Code::Es06]);
    }

    #[test]
    fn unsupported_expressions_say_what_they_are() {
        assert_eq!(codes("fn f() -> bool { let r = 0..3; true }"), [Code::Tq003]);
    }
}
