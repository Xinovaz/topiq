//! Arithmetic in machine code.
//!
//! Each operator is lowered to the plain instruction plus whatever test makes
//! it safe (as `sema::arith` defines it), and a failing test branches to an abort:
//!
//! | operator | test |
//! |---|---|
//! | `+` `-` `*` | LLVM's overflow-reporting intrinsics |
//! | `/` | a zero divisor; for signed types, `MIN / -1` as well |
//! | `%` | a zero divisor. `MIN % -1` is computed as `MIN % 1`, which is `0` (the right answer) without the instruction that would fault on it |
//! | `-x` | overflow, which for an unsigned type means any non-zero `x` |
//! | `<<` `>>` | the count must lie in `0..width`; `<<` must lose no bits |
//! | `as` | the value must lie within the target type |
//!
//! None of these tests is optional, and none is removed at a higher
//! optimisation level unless LLVM can prove it never fails, which leaves the
//! program's behaviour unchanged.
//!
//! Floating operators need none of this: every one of them has an answer. The
//! conversions do, though. `x as T` from a floating type to an integer one
//! truncates toward zero and aborts with `RA04` unless the result is a value
//! of `T`, which rules out a NaN and both infinities; and `n as char` aborts
//! with `RA03` unless the number is a Unicode scalar value, which excludes the
//! surrogate range.

use inkwell::intrinsics::Intrinsic;
use inkwell::values::{BasicValueEnum, FloatValue, IntValue};
use inkwell::{FloatPredicate, IntPredicate};

use crate::diag::Code;
use crate::span::Span;
use crate::tir::{BinOp, FloatTy, IntTy, Ty};

use super::func::{Lowering, ok};
use super::types;

/// Which overflow-reporting intrinsic an operation uses.
#[derive(Clone, Copy)]
enum Overflowing {
    Add,
    Sub,
    Mul,
}

impl<'ctx> Lowering<'ctx, '_> {
    /// Lowers a binary operator over two operands of the given types.
    pub(super) fn binary(
        &mut self,
        op: BinOp,
        lt: Ty,
        rt: Ty,
        l: IntValue<'ctx>,
        r: IntValue<'ctx>,
        span: Span,
    ) -> IntValue<'ctx> {
        let t = match lt {
            Ty::Int(t) => t,
            Ty::Bool => return self.bool_binary(op, l, r),
            // characters only compare, by their scalar values, which are
            // unsigned 32-bit numbers
            Ty::Char => IntTy::U32,
            other => unreachable!("a binary operator over {other:?}"),
        };
        let signed = t.signed;
        match op {
            BinOp::Add => self.overflowing(Overflowing::Add, t, l, r, span),
            BinOp::Sub => self.overflowing(Overflowing::Sub, t, l, r, span),
            BinOp::Mul => self.overflowing(Overflowing::Mul, t, l, r, span),
            BinOp::Div => {
                self.check_divisor(t, r, span);
                if signed {
                    let bad = self.is_min_over_minus_one(t, l, r);
                    self.abort_if(bad, Code::Ra01, span);
                    ok(self.b.build_int_signed_div(l, r, ""))
                } else {
                    ok(self.b.build_int_unsigned_div(l, r, ""))
                }
            }
            BinOp::Rem => {
                self.check_divisor(t, r, span);
                if signed {
                    // `MIN % -1` is 0, and so is `MIN % 1`; substituting the
                    // divisor avoids an instruction that traps on the former
                    let special = self.is_min_over_minus_one(t, l, r);
                    let one = types::const_int(self.cx, t, 1);
                    let divisor = ok(self.b.build_select(special, one, r, "")).into_int_value();
                    ok(self.b.build_int_signed_rem(l, divisor, ""))
                } else {
                    ok(self.b.build_int_unsigned_rem(l, r, ""))
                }
            }
            BinOp::Shl => {
                let Ty::Int(ct) = rt else {
                    unreachable!("a shift count is an integer")
                };
                let n = self.shift_count(t, ct, r, span);
                let shifted = ok(self.b.build_left_shift(l, n, ""));
                // shifting back must recover the value, or bits were lost
                let back = ok(self.b.build_right_shift(shifted, n, signed, ""));
                let lost = ok(self.b.build_int_compare(IntPredicate::NE, back, l, ""));
                self.abort_if(lost, Code::Ra01, span);
                shifted
            }
            BinOp::Shr => {
                let Ty::Int(ct) = rt else {
                    unreachable!("a shift count is an integer")
                };
                let n = self.shift_count(t, ct, r, span);
                ok(self.b.build_right_shift(l, n, signed, ""))
            }
            BinOp::BitAnd => ok(self.b.build_and(l, r, "")),
            BinOp::BitOr => ok(self.b.build_or(l, r, "")),
            BinOp::BitXor => ok(self.b.build_xor(l, r, "")),
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                ok(self.b.build_int_compare(predicate(op, signed), l, r, ""))
            }
        }
    }

    /// Lowers a binary operator over two floating operands.
    ///
    /// Nothing is checked: every floating operation has an answer, and for the
    /// ones that would be an error on integers the answer is an infinity or a
    /// NaN. A comparison is ordered, so it is false when either side is a NaN
    /// (except `!=`, which is true, since a NaN is not equal to anything).
    pub(super) fn float_binary(
        &mut self,
        op: BinOp,
        l: FloatValue<'ctx>,
        r: FloatValue<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        match op {
            BinOp::Add => ok(self.b.build_float_add(l, r, "")).into(),
            BinOp::Sub => ok(self.b.build_float_sub(l, r, "")).into(),
            BinOp::Mul => ok(self.b.build_float_mul(l, r, "")).into(),
            BinOp::Div => ok(self.b.build_float_div(l, r, "")).into(),
            BinOp::Rem => ok(self.b.build_float_rem(l, r, "")).into(),
            BinOp::Eq => self.float_compare(FloatPredicate::OEQ, l, r),
            BinOp::Ne => self.float_compare(FloatPredicate::UNE, l, r),
            BinOp::Lt => self.float_compare(FloatPredicate::OLT, l, r),
            BinOp::Le => self.float_compare(FloatPredicate::OLE, l, r),
            BinOp::Gt => self.float_compare(FloatPredicate::OGT, l, r),
            BinOp::Ge => self.float_compare(FloatPredicate::OGE, l, r),
            other => unreachable!("`{}` on floating values", other.text()),
        }
    }

    fn float_compare(
        &mut self,
        p: FloatPredicate,
        l: FloatValue<'ctx>,
        r: FloatValue<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        ok(self.b.build_float_compare(p, l, r, "")).into()
    }

    /// Lowers `e as T` for every pair of types `as` converts between.
    pub(super) fn convert(
        &mut self,
        from: Ty,
        to: Ty,
        v: BasicValueEnum<'ctx>,
        span: Span,
    ) -> BasicValueEnum<'ctx> {
        match (from, to) {
            // a character converts through its scalar value, which is an
            // unsigned 32-bit number
            (Ty::Int(_) | Ty::Char, Ty::Int(_) | Ty::Char) => {
                let f = if let Ty::Int(t) = from { t } else { IntTy::U32 };
                let t = if let Ty::Int(t) = to { t } else { IntTy::U32 };
                let x = self.cast(f, t, v.into_int_value(), span);
                if to == Ty::Char && from != Ty::Char {
                    self.check_scalar_value(x, span);
                }
                x.into()
            }
            (Ty::Int(f), Ty::Float(t)) => {
                let x = v.into_int_value();
                let target = types::float(self.cx, t);
                if f.signed {
                    ok(self.b.build_signed_int_to_float(x, target, "")).into()
                } else {
                    ok(self.b.build_unsigned_int_to_float(x, target, "")).into()
                }
            }
            (Ty::Float(f), Ty::Float(t)) => {
                let x = v.into_float_value();
                let target = types::float(self.cx, t);
                match (f, t) {
                    (FloatTy::F32, FloatTy::F64) => ok(self.b.build_float_ext(x, target, "")).into(),
                    (FloatTy::F64, FloatTy::F32) => ok(self.b.build_float_trunc(x, target, "")).into(),
                    _ => x.into(),
                }
            }
            (Ty::Float(f), Ty::Int(t)) => self.float_to_int(f, t, v.into_float_value(), span).into(),
            (f, t) => unreachable!("analysis does not convert {f:?} to {t:?}"),
        }
    }

    /// Aborts unless `x`, an unsigned 32-bit number, is a Unicode scalar
    /// value: the surrogate range and everything above `0x10FFFF` is not one.
    fn check_scalar_value(&mut self, x: IntValue<'ctx>, span: Span) {
        let u32t = types::int(self.cx, IntTy::U32);
        let above = ok(self.b.build_int_compare(
            IntPredicate::UGT,
            x,
            u32t.const_int(0x0010_FFFF, false),
            "",
        ));
        self.abort_if(above, Code::Ra03, span);
        // a surrogate is `0xD800..=0xDFFF`, which is `x - 0xD800 < 0x800`
        let offset = ok(self.b.build_int_sub(x, u32t.const_int(0xD800, false), ""));
        let surrogate = ok(self.b.build_int_compare(
            IntPredicate::ULT,
            offset,
            u32t.const_int(0x800, false),
            "",
        ));
        self.abort_if(surrogate, Code::Ra03, span);
    }

    /// `x as T` from a floating type to an integer one: the value truncated
    /// toward zero, or `RA04` when no value of `T` stands for it (a NaN, an
    /// infinity, or a magnitude the type cannot hold).
    fn float_to_int(&mut self, from: FloatTy, to: IntTy, x: FloatValue<'ctx>, span: Span) -> IntValue<'ctx> {
        // widening to `f64` first is exact and leaves one set of bounds to
        // compute rather than two
        let f64t = self.cx.f64_type();
        let x = match from {
            FloatTy::F32 => ok(self.b.build_float_ext(x, f64t, "")),
            FloatTy::F64 => x,
        };
        let truncated = self.call_float_intrinsic("llvm.trunc", x);
        // a NaN is the one value not equal to itself
        let nan = ok(self.b.build_float_compare(FloatPredicate::UNO, truncated, truncated, ""));
        self.abort_if(nan, Code::Ra04, span);
        // the bounds are exact in `f64`: the low one is zero or a power of
        // two, and the high one is the first power of two above the type
        let low = f64t.const_float(to.min() as f64);
        let high = f64t.const_float((to.max() as f64) + 1.0);
        let below = ok(self.b.build_float_compare(FloatPredicate::OLT, truncated, low, ""));
        self.abort_if(below, Code::Ra04, span);
        let above = ok(self.b.build_float_compare(FloatPredicate::OGE, truncated, high, ""));
        self.abort_if(above, Code::Ra04, span);
        let target = types::int(self.cx, to);
        if to.signed {
            ok(self.b.build_float_to_signed_int(truncated, target, ""))
        } else {
            ok(self.b.build_float_to_unsigned_int(truncated, target, ""))
        }
    }

    /// Calls a one-argument floating intrinsic, such as `llvm.trunc`.
    fn call_float_intrinsic(&mut self, name: &str, x: FloatValue<'ctx>) -> FloatValue<'ctx> {
        let f64t = self.cx.f64_type();
        let intrinsic = Intrinsic::find(name).expect("a standard LLVM intrinsic");
        let f = intrinsic
            .get_declaration(self.module, &[f64t.into()])
            .expect("the intrinsic accepts one f64");
        ok(self.b.build_call(f, &[x.into()], ""))
            .try_as_basic_value()
            .basic()
            .expect("the intrinsic returns a value")
            .into_float_value()
    }

    /// The operators that also apply to `bool`.
    fn bool_binary(&mut self, op: BinOp, l: IntValue<'ctx>, r: IntValue<'ctx>) -> IntValue<'ctx> {
        match op {
            BinOp::BitAnd => ok(self.b.build_and(l, r, "")),
            BinOp::BitOr => ok(self.b.build_or(l, r, "")),
            BinOp::BitXor | BinOp::Ne => ok(self.b.build_xor(l, r, "")),
            BinOp::Eq => ok(self.b.build_int_compare(IntPredicate::EQ, l, r, "")),
            other => unreachable!("`{}` on bool", other.text()),
        }
    }

    /// `+`, `-` or `*` through the intrinsic that reports overflow.
    fn overflowing(
        &mut self,
        which: Overflowing,
        t: IntTy,
        l: IntValue<'ctx>,
        r: IntValue<'ctx>,
        span: Span,
    ) -> IntValue<'ctx> {
        let name = format!(
            "llvm.{}{}.with.overflow",
            if t.signed { "s" } else { "u" },
            match which {
                Overflowing::Add => "add",
                Overflowing::Sub => "sub",
                Overflowing::Mul => "mul",
            }
        );
        let intrinsic = Intrinsic::find(&name)
            .unwrap_or_else(|| panic!("LLVM provides {name}"))
            .get_declaration(self.module, &[types::int(self.cx, t).into()])
            .unwrap_or_else(|| panic!("{name} accepts {t}"));
        let pair = ok(self.b.build_call(intrinsic, &[l.into(), r.into()], ""))
            .try_as_basic_value()
            .basic()
            .expect("the intrinsic returns a pair")
            .into_struct_value();
        let value = ok(self.b.build_extract_value(pair, 0, "")).into_int_value();
        let overflowed = ok(self.b.build_extract_value(pair, 1, "")).into_int_value();
        self.abort_if(overflowed, Code::Ra01, span);
        value
    }

    /// Aborts with a zero divisor.
    fn check_divisor(&mut self, t: IntTy, r: IntValue<'ctx>, span: Span) {
        let zero = types::const_int(self.cx, t, 0);
        let bad = ok(self.b.build_int_compare(IntPredicate::EQ, r, zero, ""));
        self.abort_if(bad, Code::Ra02, span);
    }

    /// Whether a signed division is `MIN / -1`, the one whose quotient does
    /// not fit.
    fn is_min_over_minus_one(&mut self, t: IntTy, l: IntValue<'ctx>, r: IntValue<'ctx>) -> IntValue<'ctx> {
        let min = types::const_int(self.cx, t, t.min());
        let minus_one = types::const_int(self.cx, t, -1);
        let is_min = ok(self.b.build_int_compare(IntPredicate::EQ, l, min, ""));
        let is_minus_one = ok(self.b.build_int_compare(IntPredicate::EQ, r, minus_one, ""));
        ok(self.b.build_and(is_min, is_minus_one, ""))
    }

    /// Checks a shift count of type `ct` against the width of `t`, and returns
    /// it converted to `t`, as LLVM requires of a shift's operands.
    fn shift_count(&mut self, t: IntTy, ct: IntTy, count: IntValue<'ctx>, span: Span) -> IntValue<'ctx> {
        // every count type can represent every width, the widest being 64
        let width = types::const_int(self.cx, ct, i128::from(t.bits));
        let too_wide = ok(self.b.build_int_compare(
            if ct.signed {
                IntPredicate::SGE
            } else {
                IntPredicate::UGE
            },
            count,
            width,
            "",
        ));
        let bad = if ct.signed {
            let zero = types::const_int(self.cx, ct, 0);
            let negative = ok(self.b.build_int_compare(IntPredicate::SLT, count, zero, ""));
            ok(self.b.build_or(too_wide, negative, ""))
        } else {
            too_wide
        };
        self.abort_if(bad, Code::Ra05, span);
        // the count is now in `0..width`, so zero-extending or truncating it
        // keeps its value
        ok(self.b.build_int_cast_sign_flag(count, types::int(self.cx, t), false, ""))
    }

    /// `-x`.
    pub(super) fn neg(&mut self, t: IntTy, x: IntValue<'ctx>, span: Span) -> IntValue<'ctx> {
        let zero = types::const_int(self.cx, t, 0);
        if t.signed {
            self.overflowing(Overflowing::Sub, t, zero, x, span)
        } else {
            // only zero has an unsigned negation
            let bad = ok(self.b.build_int_compare(IntPredicate::NE, x, zero, ""));
            self.abort_if(bad, Code::Ra01, span);
            zero
        }
    }

    /// `x as to`, aborting if the value does not fit.
    pub(super) fn cast(&mut self, from: IntTy, to: IntTy, x: IntValue<'ctx>, span: Span) -> IntValue<'ctx> {
        // each bound is only tested where the target's range is narrower, and
        // in that case the bound is representable in the source type
        let mut bad: Option<IntValue<'ctx>> = None;
        if to.min() > from.min() {
            let lo = types::const_int(self.cx, from, to.min());
            let below = ok(self.b.build_int_compare(
                if from.signed {
                    IntPredicate::SLT
                } else {
                    IntPredicate::ULT
                },
                x,
                lo,
                "",
            ));
            bad = Some(below);
        }
        if from.max() > to.max() {
            let hi = types::const_int(self.cx, from, to.max());
            let above = ok(self.b.build_int_compare(
                if from.signed {
                    IntPredicate::SGT
                } else {
                    IntPredicate::UGT
                },
                x,
                hi,
                "",
            ));
            bad = Some(match bad {
                Some(b) => ok(self.b.build_or(b, above, "")),
                None => above,
            });
        }
        if let Some(bad) = bad {
            self.abort_if(bad, Code::Ra03, span);
        }
        ok(self.b.build_int_cast_sign_flag(x, types::int(self.cx, to), from.signed, ""))
    }
}

/// The comparison predicate for an operator on integers of the given
/// signedness.
fn predicate(op: BinOp, signed: bool) -> IntPredicate {
    match (op, signed) {
        (BinOp::Eq, _) => IntPredicate::EQ,
        (BinOp::Ne, _) => IntPredicate::NE,
        (BinOp::Lt, true) => IntPredicate::SLT,
        (BinOp::Lt, false) => IntPredicate::ULT,
        (BinOp::Le, true) => IntPredicate::SLE,
        (BinOp::Le, false) => IntPredicate::ULE,
        (BinOp::Gt, true) => IntPredicate::SGT,
        (BinOp::Gt, false) => IntPredicate::UGT,
        (BinOp::Ge, true) => IntPredicate::SGE,
        (BinOp::Ge, false) => IntPredicate::UGE,
        (other, _) => unreachable!("`{}` is not a comparison", other.text()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_pick_the_signed_or_unsigned_predicate() {
        assert_eq!(predicate(BinOp::Lt, true), IntPredicate::SLT);
        assert_eq!(predicate(BinOp::Lt, false), IntPredicate::ULT);
        assert_eq!(predicate(BinOp::Eq, true), predicate(BinOp::Eq, false));
        assert_eq!(predicate(BinOp::Ge, false), IntPredicate::UGE);
    }
}
