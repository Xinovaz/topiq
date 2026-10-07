//! Addresses of places, and moving values in and out of memory.
//!
//! A place's address is computed from the address of what contains it: a
//! field's is the structure's plus the field's offset, an element's is the
//! array's plus the index times the element's size, and a referent's is the
//! first word of the reference. Every index is checked against the length
//! first, and an index that is not less than it aborts with `RA06`. A growable
//! array's elements are found through the address it holds.
//!
//! Each address carries the alignment it is known to have, and every load and
//! store states it. In a `[packed]` structure a field may sit at any byte, and
//! saying so is what makes reading it correct.
//!
//! A value that is not a place (the result of a call, a structure literal)
//! gets an address by being built in a temporary first.

use inkwell::values::{BasicValue, BasicValueEnum, IntValue, PointerValue, StructValue};

use crate::diag::Code;
use crate::tir::{Expr, ExprKind, Ty, layout};

use super::func::{Addr, Lowering, ok};
use super::types;

/// The largest power of two dividing `n`, or `cap` if `n` is 0 or has a
/// larger one.
fn alignment_of_offset(n: u64, cap: u64) -> u64 {
    if n == 0 { cap } else { (1u64 << n.trailing_zeros()).min(cap) }
}

impl<'ctx> Lowering<'ctx, '_> {
    /// The address of a place, or of a temporary holding any other value.
    pub(super) fn addr(&mut self, e: &Expr) -> Option<Addr<'ctx>> {
        match &e.kind {
            ExprKind::Local(l) => self.locals[l.index()],
            ExprKind::Global(g) => Some(self.global_addr(*g)),
            ExprKind::Field { base, field } => {
                let b = self.addr(base)?;
                // a structure's field, or a tuple's element
                let offset = layout::fields_of(&self.unit.types, base.ty).offsets[*field as usize];
                Some(self.offset(b, offset))
            }
            ExprKind::Index { base, index } => self.element(base, index, e),
            ExprKind::Deref(r) => {
                let v = self.value(r)?.into_struct_value();
                let ptr = ok(self.b.build_extract_value(v, 0, "")).into_pointer_value();
                Some(Addr {
                    ptr,
                    align: self.layout(e.ty).align,
                })
            }
            ExprKind::Const(v) if e.ty.is_aggregate() => Some(self.constant_in_memory(v, e.ty)),
            ExprKind::FnRef(c) => Some(Addr {
                ptr: self.fn_slot(super::closure::FnKey::Callee(*c)),
                align: 8,
            }),
            ExprKind::ThunkRef(id) => Some(Addr {
                ptr: self.fn_slot(super::closure::FnKey::Thunk(*id)),
                align: 8,
            }),
            _ => {
                let t = self.temp(e.ty);
                self.store_expr(e, t);
                Some(t)
            }
        }
    }

    /// The address of a value that is only looked at (indexed, measured,
    /// grown) and not used up: a place's own, or, for any other value, a
    /// temporary that keeps it until the function returns, and then
    /// destroys it.
    pub(super) fn held(&mut self, e: &Expr) -> Option<Addr<'ctx>> {
        if e.is_place() {
            return self.addr(e);
        }
        let (t, flag) = self.lasting_temp(e.ty);
        self.store_expr(e, t);
        self.keep_temp(flag);
        Some(t)
    }

    /// The address `offset` bytes past `a`.
    pub(super) fn offset(&self, a: Addr<'ctx>, offset: u64) -> Addr<'ctx> {
        if offset == 0 {
            return a;
        }
        let i8 = self.cx.i8_type();
        let off = self.cx.i64_type().const_int(offset, false);
        // SAFETY: the offset is a field's, within the object at `a`
        let ptr = unsafe { ok(self.b.build_in_bounds_gep(i8, a.ptr, &[off], "")) };
        Addr {
            ptr,
            align: alignment_of_offset(offset, a.align),
        }
    }

    /// The address of `base[index]`.
    fn element(&mut self, base: &Expr, index: &Expr, e: &Expr) -> Option<Addr<'ctx>> {
        let elem = e.ty;
        let stride = self.layout(elem).size;
        let (start, align, len) = match base.ty {
            Ty::Slice(_) => {
                let s = self.value(base)?.into_struct_value();
                let (p, n) = self.slice_parts(s);
                (p, self.layout(elem).align, n)
            }
            // read in place: indexing a growable array looks at it and moves
            // nothing
            Ty::Growable(_) => {
                let at = self.held(base)?;
                let s = self.load_scalar(base.ty, at)?.into_struct_value();
                let (p, n) = self.slice_parts(s);
                // a buffer promises no more than 16
                (p, self.layout(elem).align.min(16), n)
            }
            _ => {
                let b = self.held(base)?;
                let (_, n) = self.unit.types.as_array(base.ty).expect("indexing an array or slice");
                (b.ptr, b.align, self.cx.i64_type().const_int(n, false))
            }
        };
        let i = self.value(index)?.into_int_value();
        let bad = ok(self.b.build_int_compare(inkwell::IntPredicate::UGE, i, len, "out.of.bounds"));
        self.abort_if(bad, Code::Ra06, index.span);
        let i8 = self.cx.i8_type();
        let bytes = ok(self.b.build_int_nuw_mul(i, self.cx.i64_type().const_int(stride, false), ""));
        // SAFETY: the index was just checked to be less than the length
        let ptr = unsafe { ok(self.b.build_in_bounds_gep(i8, start, &[bytes], "")) };
        Some(Addr {
            ptr,
            align: alignment_of_offset(stride, align).min(align),
        })
    }

    /// `p == q` or `p != q` on two references: whether they refer to the same
    /// place (the same address and, for a slice, the same count).
    pub(super) fn reference_eq(&mut self, equal: bool, l: &Expr, r: &Expr) -> Option<IntValue<'ctx>> {
        let a = self.value(l)?.into_struct_value();
        let b = self.value(r)?.into_struct_value();
        let (pa, pb) = (
            ok(self.b.build_extract_value(a, 0, "")).into_pointer_value(),
            ok(self.b.build_extract_value(b, 0, "")).into_pointer_value(),
        );
        let mut same = ok(self.b.build_int_compare(inkwell::IntPredicate::EQ, pa, pb, "same.address"));
        if l.ty.is_slice() {
            let (na, nb) = (self.slice_parts(a).1, self.slice_parts(b).1);
            let n = ok(self.b.build_int_compare(inkwell::IntPredicate::EQ, na, nb, "same.count"));
            same = ok(self.b.build_and(same, n, ""));
        }
        Some(if equal { same } else { ok(self.b.build_not(same, "")) })
    }

    /// The address and count of a slice.
    pub(super) fn slice_parts(&self, s: StructValue<'ctx>) -> (PointerValue<'ctx>, IntValue<'ctx>) {
        let p = ok(self.b.build_extract_value(s, 0, "")).into_pointer_value();
        let n = ok(self.b.build_extract_value(s, 1, "")).into_int_value();
        (p, n)
    }

    /// Reads a value held in registers from memory.
    pub(super) fn load_scalar(&self, ty: Ty, a: Addr<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let t = types::scalar(self.cx, &self.unit.types, ty)?;
        let v = ok(self.b.build_load(t, a.ptr, ""));
        if let Some(i) = v.as_instruction_value() {
            let _ = i.set_alignment(a.align as u32);
        }
        Some(v)
    }

    /// Writes a value held in registers to memory.
    pub(super) fn store_scalar(&self, v: BasicValueEnum<'ctx>, a: Addr<'ctx>) {
        let s = ok(self.b.build_store(a.ptr, v));
        let _ = s.set_alignment(a.align as u32);
    }

    /// Copies a value of `ty` from `src` to `dst`, byte for byte.
    pub(super) fn copy(&self, dst: Addr<'ctx>, src: Addr<'ctx>, ty: Ty) {
        let size = self.layout(ty).size;
        if size == 0 {
            return;
        }
        let n = self.cx.i64_type().const_int(size, false);
        ok(self.b.build_memcpy(dst.ptr, dst.align as u32, src.ptr, src.align as u32, n));
    }

    /// `&x`: the address of a place, or of a temporary holding any
    /// other value, with the second word the reference type calls for: the
    /// count for an array, and otherwise the referent's table of run-time type
    /// information.
    pub(super) fn reference(&mut self, x: &Expr, ty: Ty) -> Option<BasicValueEnum<'ctx>> {
        // a value that is not a place is kept in a temporary for as long as
        // the function runs, and destroyed when it returns
        let a = if x.is_place() {
            self.addr(x)?
        } else {
            let (t, flag) = self.lasting_temp(x.ty);
            self.store_expr(x, t);
            self.keep_temp(flag);
            t
        };
        self.reference_at(a, x.ty, ty)
    }

    /// A reference of type `ty` to the value of type `target` at `a`.
    pub(super) fn reference_at(&mut self, a: Addr<'ctx>, target: Ty, ty: Ty) -> Option<BasicValueEnum<'ctx>> {
        // a growable array is already the address of its elements and their
        // count, which is what a slice of them is
        if matches!(target, Ty::Growable(_)) {
            return self.load_scalar(target, a);
        }
        let v = if types::counts(&self.unit.types, ty) {
            let (_, n) = self.unit.types.as_array(target).expect("a reference to an array");
            let n = self.cx.i64_type().const_int(n, false);
            let fat = types::fat_slice(self.cx).get_undef();
            let fat = ok(self.b.build_insert_value(fat, a.ptr, 0, "")).into_struct_value();
            ok(self.b.build_insert_value(fat, n, 1, "")).into_struct_value()
        } else {
            let table = self.table(target);
            self.pair(a.ptr, table)
        };
        Some(v.as_basic_value_enum())
    }
}

#[cfg(test)]
mod tests {
    use super::alignment_of_offset;
    use crate::codegen::testing::{function, ir};

    #[test]
    fn an_offset_keeps_only_the_alignment_it_can_promise() {
        assert_eq!(alignment_of_offset(0, 8), 8);
        assert_eq!(alignment_of_offset(4, 8), 4);
        assert_eq!(alignment_of_offset(16, 8), 8);
        assert_eq!(alignment_of_offset(1, 4), 1);
    }

    #[test]
    fn a_packed_field_is_read_at_whatever_alignment_it_has() {
        let ir = ir("[packed]\nstruct H { tag: u8, len: u32 }\nfn f(h: H) -> u32 { h.len }");
        let f = function(&ir, "f");
        assert!(f.contains("load i32, ptr") && f.contains(", align 1"), "{f}");
    }

    #[test]
    fn a_field_through_a_reference_reads_the_first_word() {
        let ir = ir("struct P { x: i32, y: i32 }\nfn f(p: *P) -> i32 { p.y }");
        let f = function(&ir, "f");
        assert!(f.contains("extractvalue { ptr, ptr }"), "{f}");
        assert!(f.contains("getelementptr inbounds i8"), "{f}");
    }

    #[test]
    fn a_reference_to_an_array_carries_its_length() {
        let ir = ir("fn g(s: *[u8]) -> usize { s.len() }\nfn f() -> usize { let a = [1u8, 2, 3]; g(&a) }");
        let f = function(&ir, "f");
        assert!(f.contains("insertvalue { ptr, i64 }"), "{f}");
        assert!(f.contains("i64 3, 1"), "{f}");
    }
}
