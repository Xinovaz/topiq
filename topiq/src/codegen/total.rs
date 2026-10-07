//! The library's total arithmetic: results wrapped, clamped or truncated, and
//! never an abort.
//!
//! `core`'s `wrapping_*`, `checked_*`, `overflowing_*`, `saturating_*`,
//! `truncate` and `saturating_as` are written in Topiq over these operations,
//! each defined for every input. A division by zero is the one thing they
//! leave to their caller: `core` tests for it first, and here a zero divisor
//! is replaced, so that no instruction that traps is ever run.

use inkwell::intrinsics::Intrinsic as Llvm;
use inkwell::values::{BasicValueEnum, IntValue};
use inkwell::IntPredicate;

use crate::tir::{BinOp, Expr, FloatTy, IntTy, Ty, layout};

use super::func::{Dest, Lowering, ok};
use super::types;

impl<'ctx> Lowering<'ctx, '_> {
    /// `(result, wrapped)` for `op`, or for negation when `op` is `None`,
    /// written into `dest` as the tuple it is.
    pub(super) fn overflowing_op(&mut self, op: Option<BinOp>, args: &[Expr], ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let Ty::Int(t) = args[0].ty else {
            unreachable!("total arithmetic is on integers")
        };
        let a = self.value(&args[0])?.into_int_value();
        let (value, flag) = match op {
            None => {
                let zero = types::const_int(self.cx, t, 0);
                if t.signed {
                    self.with_overflow("sub", t, zero, a)
                } else {
                    let v = ok(self.b.build_int_sub(zero, a, ""));
                    let f = ok(self.b.build_int_compare(IntPredicate::NE, a, zero, ""));
                    (v, f)
                }
            }
            Some(op) => {
                let b = self.value(&args[1])?.into_int_value();
                self.wrapped(op, t, a, b)
            }
        };
        let out = match dest {
            Dest::Mem(a) => a,
            Dest::Value => self.temp(ty),
        };
        let offsets = layout::fields_of(&self.unit.types, ty).offsets;
        self.store_scalar(value.into(), out);
        let at = self.offset(out, offsets[1]);
        self.store_scalar(flag.into(), at);
        None
    }

    /// `a op b` wrapped to `t`'s width, and whether it was wrapped.
    fn wrapped(&mut self, op: BinOp, t: IntTy, a: IntValue<'ctx>, b: IntValue<'ctx>) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let zero = types::const_int(self.cx, t, 0);
        let one = types::const_int(self.cx, t, 1);
        let falsy = self.cx.bool_type().const_zero();
        match op {
            BinOp::Add => self.with_overflow("add", t, a, b),
            BinOp::Sub => self.with_overflow("sub", t, a, b),
            BinOp::Mul => self.with_overflow("mul", t, a, b),
            BinOp::Div | BinOp::Rem => {
                // a zero divisor is the caller's to exclude; one stands in
                // for it here so that nothing traps
                let is_zero = ok(self.b.build_int_compare(IntPredicate::EQ, b, zero, ""));
                let b = ok(self.b.build_select(is_zero, one, b, "")).into_int_value();
                if !t.signed {
                    let v = if op == BinOp::Div {
                        ok(self.b.build_int_unsigned_div(a, b, ""))
                    } else {
                        ok(self.b.build_int_unsigned_rem(a, b, ""))
                    };
                    return (v, falsy);
                }
                // `MIN / -1` is the one quotient that does not fit: it wraps
                // to `MIN`, and `MIN % -1` is 0
                let min = types::const_int(self.cx, t, t.min());
                let minus_one = types::const_int(self.cx, t, -1);
                let is_min = ok(self.b.build_int_compare(IntPredicate::EQ, a, min, ""));
                let is_m1 = ok(self.b.build_int_compare(IntPredicate::EQ, b, minus_one, ""));
                let special = ok(self.b.build_and(is_min, is_m1, ""));
                let safe = ok(self.b.build_select(special, one, b, "")).into_int_value();
                let v = if op == BinOp::Div {
                    let q = ok(self.b.build_int_signed_div(a, safe, ""));
                    ok(self.b.build_select(special, min, q, "")).into_int_value()
                } else {
                    ok(self.b.build_int_signed_rem(a, safe, ""))
                };
                (v, special)
            }
            BinOp::Shl | BinOp::Shr => {
                // the count is taken modulo the width; a count past it is
                // what wrapped
                let bits = types::const_int(self.cx, t, i128::from(t.bits));
                let too_far = ok(self.b.build_int_compare(IntPredicate::UGE, b, bits, ""));
                let mask = types::const_int(self.cx, t, i128::from(t.bits) - 1);
                let n = ok(self.b.build_and(b, mask, ""));
                let v = if op == BinOp::Shl {
                    ok(self.b.build_left_shift(a, n, ""))
                } else {
                    ok(self.b.build_right_shift(a, n, t.signed, ""))
                };
                (v, too_far)
            }
            other => unreachable!("`{}` has no total form", other.text()),
        }
    }

    /// `llvm.{s,u}<op>.with.overflow`: the wrapped result and the flag.
    pub(super) fn with_overflow(&mut self, op: &str, t: IntTy, a: IntValue<'ctx>, b: IntValue<'ctx>) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let name = format!("llvm.{}{op}.with.overflow", if t.signed { "s" } else { "u" });
        let pair = self.call_int_intrinsic(&name, t, &[a, b]).into_struct_value();
        let v = ok(self.b.build_extract_value(pair, 0, "")).into_int_value();
        let f = ok(self.b.build_extract_value(pair, 1, "")).into_int_value();
        (v, f)
    }

    /// Calls an LLVM intrinsic overloaded on one integer type.
    fn call_int_intrinsic(&mut self, name: &str, t: IntTy, args: &[IntValue<'ctx>]) -> BasicValueEnum<'ctx> {
        let f = Llvm::find(name)
            .unwrap_or_else(|| panic!("LLVM provides {name}"))
            .get_declaration(self.module, &[types::int(self.cx, t).into()])
            .unwrap_or_else(|| panic!("{name} accepts {t}"));
        let args: Vec<_> = args.iter().map(|&a| a.into()).collect();
        ok(self.b.build_call(f, &args, ""))
            .try_as_basic_value()
            .basic()
            .expect("the intrinsic returns a value")
    }

    /// `a op b` clamped to `t`'s range.
    pub(super) fn saturating_op(&mut self, op: BinOp, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let Ty::Int(t) = args[0].ty else {
            unreachable!("total arithmetic is on integers")
        };
        let a = self.value(&args[0])?.into_int_value();
        let b = self.value(&args[1])?.into_int_value();
        let s = if t.signed { "s" } else { "u" };
        let v = match op {
            BinOp::Add => self.call_int_intrinsic(&format!("llvm.{s}add.sat"), t, &[a, b]).into_int_value(),
            BinOp::Sub => self.call_int_intrinsic(&format!("llvm.{s}sub.sat"), t, &[a, b]).into_int_value(),
            BinOp::Mul => {
                let (v, over) = self.with_overflow("mul", t, a, b);
                let max = types::const_int(self.cx, t, t.max());
                let bound = if t.signed {
                    // the true product is negative when exactly one operand
                    // is
                    let zero = types::const_int(self.cx, t, 0);
                    let an = ok(self.b.build_int_compare(IntPredicate::SLT, a, zero, ""));
                    let bn = ok(self.b.build_int_compare(IntPredicate::SLT, b, zero, ""));
                    let neg = ok(self.b.build_xor(an, bn, ""));
                    let min = types::const_int(self.cx, t, t.min());
                    ok(self.b.build_select(neg, min, max, "")).into_int_value()
                } else {
                    max
                };
                ok(self.b.build_select(over, bound, v, "")).into_int_value()
            }
            other => unreachable!("`{}` has no saturating form", other.text()),
        };
        Some(v.into())
    }

    /// An integer given the type `to` by keeping its low bits, or widening.
    pub(super) fn truncate_to(&mut self, x: &Expr, to: Ty) -> Option<BasicValueEnum<'ctx>> {
        let (Ty::Int(from), Ty::Int(to)) = (x.ty, to) else {
            unreachable!("truncation is between integer types")
        };
        let v = self.value(x)?.into_int_value();
        let target = types::int(self.cx, to);
        let r = if to.bits < from.bits {
            ok(self.b.build_int_truncate(v, target, ""))
        } else if to.bits > from.bits && from.signed {
            ok(self.b.build_int_s_extend(v, target, ""))
        } else if to.bits > from.bits {
            ok(self.b.build_int_z_extend(v, target, ""))
        } else {
            v
        };
        Some(r.into())
    }

    /// A value converted to the integer type `to`.
    pub(super) fn saturate_to(&mut self, x: &Expr, to: Ty) -> Option<BasicValueEnum<'ctx>> {
        let Ty::Int(to) = to else {
            unreachable!("a saturating conversion gives an integer")
        };
        let target = types::int(self.cx, to);
        match x.ty {
            Ty::Float(f) => {
                let v = self.value(x)?.into_float_value();
                let name = format!("llvm.fpto{}i.sat", if to.signed { "s" } else { "u" });
                let ft: inkwell::types::BasicTypeEnum<'ctx> = match f {
                    FloatTy::F32 => self.cx.f32_type().into(),
                    FloatTy::F64 => self.cx.f64_type().into(),
                };
                let decl = Llvm::find(&name)
                    .unwrap_or_else(|| panic!("LLVM provides {name}"))
                    .get_declaration(self.module, &[target.into(), ft])
                    .unwrap_or_else(|| panic!("{name} accepts these types"));
                let r = ok(self.b.build_call(decl, &[v.into()], ""))
                    .try_as_basic_value()
                    .basic()
                    .expect("a value");
                Some(r)
            }
            Ty::Int(from) => {
                // compared in 128 bits, where every value of both types fits
                let v = self.value(x)?.into_int_value();
                let wide = self.cx.i128_type();
                let w = if from.signed {
                    ok(self.b.build_int_s_extend_or_bit_cast(v, wide, ""))
                } else {
                    ok(self.b.build_int_z_extend_or_bit_cast(v, wide, ""))
                };
                let lo = wide.const_int_arbitrary_precision(&words(to.min()));
                let hi = wide.const_int_arbitrary_precision(&words(to.max()));
                let below = ok(self.b.build_int_compare(IntPredicate::SLT, w, lo, ""));
                let above = ok(self.b.build_int_compare(IntPredicate::SGT, w, hi, ""));
                let w = ok(self.b.build_select(below, lo, w, "")).into_int_value();
                let w = ok(self.b.build_select(above, hi, w, "")).into_int_value();
                Some(ok(self.b.build_int_truncate_or_bit_cast(w, target, "")).into())
            }
            other => unreachable!("a saturating conversion from {other:?}"),
        }
    }
}

/// A 128-bit value as the two 64-bit words LLVM takes, low word first.
fn words(v: i128) -> [u64; 2] {
    let u = v as u128;
    [u as u64, (u >> 64) as u64]
}
