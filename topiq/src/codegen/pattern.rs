//! Lowering `match`.
//!
//! The value matched is examined where it is: a place is read in place, and
//! anything else is first stored in a temporary. Each arm's pattern becomes a
//! chain of tests on that memory (a discriminant compared with a variant's
//! number, a field compared with a literal), any of which, failing, jumps to
//! the next arm. Names the pattern binds are copied into their bindings as the
//! tests pass. Analysis has checked that some arm matches every value, so
//! falling off the last arm cannot happen and is marked unreachable.
//!
//! At `-O2` LLVM turns a chain of comparisons of one discriminant into a
//! jump table.

use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::values::BasicValueEnum;

use crate::tir::{Arm, Expr, LocalId, Pat, PatKind, Ty, layout};

use super::func::{Addr, Dest, Lowering, ok};
use super::types;

impl<'ctx> Lowering<'ctx, '_> {
    /// `match scrutinee { arms }`, delivering its value to `dest`.
    pub(super) fn match_expr(
        &mut self,
        scrutinee: &Expr,
        arms: &[Arm],
        ty: Ty,
        dest: Dest<'ctx>,
    ) -> Option<BasicValueEnum<'ctx>> {
        let place = scrutinee.is_place();
        let s = self.addr(scrutinee)?;
        // an arm that keeps a part of the value that owns something takes
        // the value; a value that is not a place is the match's anyway. then
        // each arm destroys what its pattern does not keep
        let taken = !place || arms.iter().any(|a| self.binds_owned(&a.pat));
        if place && taken {
            self.moved(scrutinee);
        }
        let func = self.func.expect("inside a function");
        let out = self.out_for(ty, dest, "match.value");
        let join = self.cx.append_basic_block(func, "match.end");
        for arm in arms {
            let next = self.cx.append_basic_block(func, "arm.next");
            self.test(&arm.pat, s, next);
            self.enter_scope();
            self.own_bindings(&arm.pat);
            if taken {
                self.drop_unbound(&arm.pat, s);
            }
            self.deliver(out, &arm.body);
            self.leave_scope();
            self.close_branch(join);
            self.b.position_at_end(next);
        }
        // every value matches some arm
        ok(self.b.build_unreachable());
        self.b.position_at_end(join);
        self.result(out)
    }

    /// Gives the local `l` a reference to the value of type `ty` at `a`.
    fn bind_reference(&mut self, l: crate::tir::LocalId, ty: Ty, a: Addr<'ctx>) {
        let Some(slot) = self.locals[l.index()] else { return };
        let rty = self.local_types[l.index()];
        if let Some(r) = self.reference_at(a, ty, rty) {
            self.store_scalar(r, slot);
        }
    }

    /// Binds what a pattern names from the value at `a`, without testing
    /// anything: for `let (a, b) = …`, where analysis has checked that the
    /// pattern matches every value of the type.
    pub(super) fn bind_all(&mut self, p: &Pat, a: Addr<'ctx>) {
        match &p.kind {
            PatKind::Wild | PatKind::Const(_) => {}
            PatKind::BindRef(l) => self.bind_reference(*l, p.ty, a),
            PatKind::Bind(l) => self.bind_value(*l, p.ty, a),
            PatKind::Struct { fields } => {
                let offsets = layout::fields_of(&self.unit.types, p.ty).offsets;
                for (i, f) in fields {
                    let at = self.offset(a, offsets[*i as usize]);
                    self.bind_all(f, at);
                }
            }
            PatKind::Variant { variant, fields } => {
                let Ty::Adt(id) = p.ty else {
                    unreachable!("a variant pattern has an enumeration type");
                };
                let offsets = layout::enumeration(&self.unit.types, id).variants[*variant as usize].clone();
                for (i, f) in fields {
                    let at = self.offset(a, offsets[*i as usize]);
                    self.bind_all(f, at);
                }
            }
        }
    }

    /// Tests the value at `a` against a pattern, jumping to `fail` if it does
    /// not match, and binding what the pattern binds if it does.
    fn test(&mut self, p: &Pat, a: Addr<'ctx>, fail: BasicBlock<'ctx>) {
        match &p.kind {
            PatKind::Wild => {}
            PatKind::BindRef(l) => self.bind_reference(*l, p.ty, a),
            PatKind::Bind(l) => self.bind_value(*l, p.ty, a),
            PatKind::Const(v) => {
                let Some(x) = self.load_scalar(p.ty, a) else { return };
                let c = types::const_scalar(self.cx, v, &mut |_| unreachable!("no string patterns"))
                    .expect("a literal pattern is a scalar");
                // a floating literal matches the value it names and nothing
                // else, so a NaN scrutinee matches no literal at all
                let same = if p.ty.is_float() {
                    ok(self.b.build_float_compare(
                        inkwell::FloatPredicate::OEQ,
                        x.into_float_value(),
                        c.into_float_value(),
                        "matches",
                    ))
                } else {
                    ok(self.b.build_int_compare(IntPredicate::EQ, x.into_int_value(), c.into_int_value(), "matches"))
                };
                self.branch_on(same, fail);
            }
            PatKind::Struct { fields } => {
                let offsets = layout::fields_of(&self.unit.types, p.ty).offsets;
                for (i, f) in fields {
                    let at = self.offset(a, offsets[*i as usize]);
                    self.test(f, at, fail);
                }
            }
            PatKind::Variant { variant, fields } => {
                let Ty::Adt(id) = p.ty else {
                    unreachable!("a variant pattern has an enumeration type");
                };
                let l = layout::enumeration(&self.unit.types, id);
                let tag_ty = types::int(self.cx, l.tag);
                let tag = ok(self.b.build_load(tag_ty, a.ptr, "tag")).into_int_value();
                if let Some(i) = tag.as_instruction() {
                    let _ = i.set_alignment(a.align as u32);
                }
                let expected = types::const_int(self.cx, l.tag, i128::from(*variant));
                let same = ok(self.b.build_int_compare(IntPredicate::EQ, tag, expected, "is.variant"));
                self.branch_on(same, fail);
                let offsets = l.variants[*variant as usize].clone();
                for (i, f) in fields {
                    let at = self.offset(a, offsets[*i as usize]);
                    self.test(f, at, fail);
                }
            }
        }
    }

    /// Binds `l` to a copy of the value of type `ty` at `a`.
    fn bind_value(&mut self, l: LocalId, ty: Ty, a: Addr<'ctx>) {
        if let Some(slot) = self.locals[l.index()] {
            if ty.is_aggregate() {
                self.copy(slot, a, ty);
            } else if let Some(v) = self.load_scalar(ty, a) {
                self.store_scalar(v, slot);
            }
        }
    }

    /// Continues where `cond` holds, jumping to `fail` where it does not.
    fn branch_on(&mut self, cond: inkwell::values::IntValue<'ctx>, fail: BasicBlock<'ctx>) {
        let func = self.func.expect("inside a function");
        let pass = self.cx.append_basic_block(func, "arm.test");
        ok(self.b.build_conditional_branch(cond, pass, fail));
        self.b.position_at_end(pass);
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::testing::{function, ir};

    #[test]
    fn a_variant_arm_tests_the_discriminant() {
        let ir = ir(
            "enum S { A, B(i64) }\n\
             fn f(s: S) -> i64 { match s { S::A => 0, S::B(n) => n } }",
        );
        let f = function(&ir, "f");
        assert!(f.contains("%tag = load i8"), "{f}");
        assert!(f.contains("icmp eq i8"), "{f}");
    }

    #[test]
    fn falling_off_the_last_arm_is_unreachable() {
        let ir = ir("fn f(b: bool) -> i32 { match b { true => 1, false => 0 } }");
        let f = function(&ir, "f");
        assert!(f.contains("unreachable"), "{f}");
    }
}
