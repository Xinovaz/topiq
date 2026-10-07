//! Building structures, enumeration values and arrays.
//!
//! A structure literal evaluates its fields in the order they were written,
//! storing each straight into its place in the destination; a variant stores
//! its discriminant, then its fields at the variant's offsets. Nothing is
//! assembled in registers and copied afterwards. `[e; N]` evaluates `e` once
//! into the first element and copies it into the rest.

use inkwell::IntPredicate;

use crate::tir::{Expr, ExprKind, Ty, layout};

use super::func::{Addr, Lowering, ok};
use super::types;

/// Arrays up to this long are filled by copying element by element; longer
/// ones by a loop.
const UNROLL: u64 = 8;

impl<'ctx> Lowering<'ctx, '_> {
    /// Builds a structure literal, variant or array at `a`.
    pub(super) fn construct(&mut self, e: &Expr, a: Addr<'ctx>) {
        match &e.kind {
            ExprKind::StructLit { fields } => {
                // a tuple is built the same way: its elements are fields whose
                // names are their positions
                let offsets = layout::fields_of(&self.unit.types, e.ty).offsets;
                for (i, f) in fields {
                    let at = self.offset(a, offsets[*i as usize]);
                    self.store_expr(f, at);
                }
            }
            ExprKind::Variant { variant, fields } => {
                let Ty::Adt(id) = e.ty else {
                    unreachable!("a variant has an enumeration type");
                };
                let l = layout::enumeration(&self.unit.types, id);
                let tag = types::const_int(self.cx, l.tag, i128::from(*variant));
                self.store_scalar(tag.into(), a);
                let offsets = &l.variants[*variant as usize];
                for (i, f) in fields {
                    let at = self.offset(a, offsets[*i as usize]);
                    self.store_expr(f, at);
                }
            }
            ExprKind::ArrayLit(items) => {
                let (elem, _) = self.unit.types.as_array(e.ty).expect("an array literal has an array type");
                let stride = self.layout(elem).size;
                for (i, x) in items.iter().enumerate() {
                    let at = self.offset(a, i as u64 * stride);
                    self.store_expr(x, at);
                }
            }
            ExprKind::ArrayRepeat { elem, len } => self.repeat(elem, *len, a),
            other => unreachable!("not built in place: {other:?}"),
        }
    }

    /// Another element of `[e; N]`, from the first: a copy of its bytes, or
    /// for a value that owns something, a copy of its own, as `$copy` or
    /// `clone` would make, so that no two elements share what they own.
    fn repeat_one(&mut self, at: Addr<'ctx>, first: Addr<'ctx>, ty: crate::tir::Ty) {
        if self.needs_drop(ty) {
            self.clone_into(at, first, ty);
        } else {
            self.copy(at, first, ty);
        }
    }

    /// `[elem; len]` at `a`.
    fn repeat(&mut self, elem: &Expr, len: u64, a: Addr<'ctx>) {
        if len == 0 {
            // no element to hold it, but it is still evaluated
            let t = self.temp(elem.ty);
            self.store_expr(elem, t);
            return;
        }
        self.store_expr(elem, a);
        let stride = self.layout(elem.ty).size;
        if len <= UNROLL || stride == 0 {
            for k in 1..len {
                let at = self.offset(a, k * stride);
                self.repeat_one(at, a, elem.ty);
            }
            return;
        }
        let func = self.func.expect("inside a function");
        let i64 = self.cx.i64_type();
        let counter = self.named_temp(Ty::USIZE, "repeat.i");
        self.store_scalar(i64.const_int(1, false).into(), counter);
        let top = self.cx.append_basic_block(func, "repeat");
        let end = self.cx.append_basic_block(func, "repeat.end");
        ok(self.b.build_unconditional_branch(top));
        self.b.position_at_end(top);
        let k = self.load_scalar(Ty::USIZE, counter).expect("a usize").into_int_value();
        let bytes = ok(self.b.build_int_nuw_mul(k, i64.const_int(stride, false), ""));
        let i8 = self.cx.i8_type();
        // SAFETY: `k` runs from 1 to `len - 1`, all within the array
        let ptr = unsafe { ok(self.b.build_in_bounds_gep(i8, a.ptr, &[bytes], "")) };
        let at = Addr {
            ptr,
            align: self.layout(elem.ty).align.min(a.align),
        };
        self.repeat_one(at, a, elem.ty);
        let next = ok(self.b.build_int_add(k, i64.const_int(1, false), ""));
        self.store_scalar(next.into(), counter);
        let more = ok(self.b.build_int_compare(IntPredicate::ULT, next, i64.const_int(len, false), ""));
        ok(self.b.build_conditional_branch(more, top, end));
        self.b.position_at_end(end);
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::testing::{function, ir};

    #[test]
    fn a_variant_stores_its_discriminant_then_its_fields() {
        let ir = ir("enum S { A, B(i64) }\nfn f() -> S { S::B(7) }");
        let f = function(&ir, "f");
        assert!(f.contains("store i8 1"), "the discriminant of the second variant:\n{f}");
        assert!(f.contains("store i64 7"), "{f}");
    }

    #[test]
    fn a_long_repeated_array_is_filled_by_a_loop_and_a_short_one_by_copies() {
        let long = ir("fn f() -> [u32; 100] { [7; 100] }");
        assert!(function(&long, "f").contains("repeat:"), "{long}");
        let short = ir("fn f() -> [u32; 3] { [7; 3] }");
        assert!(!function(&short, "f").contains("repeat:"), "{short}");
    }
}
