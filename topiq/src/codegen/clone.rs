//! `clone`: a deep copy of a value.
//!
//! Each type that needs more than its bytes copied gets a cloning function,
//! made the first time it is needed, that writes a copy of the value at one
//! address to another. A type that defines `$copy` is copied by it, as the
//! type says copying it should be done. A growable array gets a buffer of its
//! own, holding a clone of each element. Anything else is cloned part by
//! part: a structure field by field, an enumeration by its variant's fields,
//! a tuple or fixed array element by element. What owns nothing (numbers,
//! references, functions) is copied as it is.
//!
//! Analysis has already refused a closure anywhere in the value: what its
//! environment holds is not known, so it cannot be cloned.

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::span::Span;
use crate::tir::{AdtKind, Expr, Ty, layout};

use super::func::{Addr, Dest, Lowering, ok};

impl<'ctx> Lowering<'ctx, '_> {
    /// `clone(r)`: a copy of what the reference `r` refers to.
    pub(super) fn clone_of(&mut self, r: &Expr, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let v = self.value(r)?.into_struct_value();
        let src = if r.ty.is_slice() {
            // a slice's words are a growable array's: the address of the
            // elements and their count, which is all a copy reads
            let words = self.temp(ty);
            self.store_scalar(v.into(), words);
            words
        } else {
            Addr {
                ptr: ok(self.b.build_extract_value(v, 0, "")).into_pointer_value(),
                align: self.layout(ty).align,
            }
        };
        self.clone_to(dest, src, ty)
    }

    /// Delivers a clone of the value of type `ty` at `src` to `dest`.
    pub(super) fn clone_to(&mut self, dest: Dest<'ctx>, src: Addr<'ctx>, ty: Ty) -> Option<BasicValueEnum<'ctx>> {
        let out = match dest {
            Dest::Mem(a) => a,
            Dest::Value => self.temp(ty),
        };
        self.clone_into(out, src, ty);
        match dest {
            Dest::Mem(_) => None,
            Dest::Value => self.load_scalar(ty, out),
        }
    }

    /// Whether cloning a value of `ty` does more than copy its bytes.
    fn deep(&self, ty: Ty) -> bool {
        self.unit.types.holds_any(ty, &|t| match t {
            Ty::Growable(_) | Ty::Closure(_) => true,
            Ty::Adt(id) => self.unit.copies.contains_key(&id),
            _ => false,
        })
    }

    /// Writes a clone of the value of type `ty` at `src` to `dst`.
    pub(super) fn clone_into(&mut self, dst: Addr<'ctx>, src: Addr<'ctx>, ty: Ty) {
        if self.terminated() {
            return;
        }
        if !self.deep(ty) {
            self.copy(dst, src, ty);
            return;
        }
        let f = self.clone_glue(ty);
        ok(self.b.build_call(f, &[dst.ptr.into(), src.ptr.into()], ""));
    }

    /// The function that clones a value of `ty`, from the address it is given
    /// second to the one it is given first.
    fn clone_glue(&mut self, ty: Ty) -> FunctionValue<'ctx> {
        if let Some(&f) = self.clone_fns.get(&ty) {
            return f;
        }
        let ptr = self.cx.ptr_type(AddressSpace::default());
        let name = format!("clone.{}", self.clone_fns.len());
        let f = self.module.add_function(
            &name,
            self.cx.void_type().fn_type(&[ptr.into(), ptr.into()], false),
            Some(LlvmLinkage::Private),
        );
        // recorded first, so that a type holding itself behind an owning
        // pointer finds it
        self.clone_fns.insert(ty, f);
        let saved = self.enter_fn(f);
        let align = self.layout(ty).align;
        let param = |i: u32| f.get_nth_param(i).expect("the addresses").into_pointer_value();
        let (dst, src) = (Addr { ptr: param(0), align }, Addr { ptr: param(1), align });
        self.clone_parts(dst, src, ty);
        if !self.terminated() {
            ok(self.b.build_return(None));
        }
        self.leave_fn(saved);
        f
    }

    /// The body of a cloning function.
    fn clone_parts(&mut self, dst: Addr<'ctx>, src: Addr<'ctx>, ty: Ty) {
        match ty {
            Ty::Adt(id) if self.unit.copies.contains_key(&id) => {
                let callee = self.unit.copies[&id];
                let (f, sret) = self.callee_function(callee);
                let table = self.table(ty);
                let r = self.pair(src.ptr, table);
                if sret {
                    ok(self.b.build_call(f, &[dst.ptr.into(), r.into()], ""));
                } else if let Some(v) = ok(self.b.build_call(f, &[r.into()], "")).try_as_basic_value().basic() {
                    self.store_scalar(v, dst);
                }
            }
            Ty::Adt(id) => match self.unit.types.adt(id).kind.clone() {
                AdtKind::Struct { fields } => {
                    let offsets = layout::fields_of(&self.unit.types, ty).offsets;
                    for (f, &off) in fields.iter().zip(&offsets) {
                        let (d, s) = (self.offset(dst, off), self.offset(src, off));
                        self.clone_into(d, s, f.ty);
                    }
                }
                AdtKind::Enum { .. } => {
                    // the whole value first, which copies the tag and every
                    // part that needs nothing more; then the variant's parts
                    // that do
                    self.copy(dst, src, ty);
                    self.per_variant(id, src, "clone", Self::deep, |l, fields| {
                        for &(t, off) in fields {
                            let (d, s) = (l.offset(dst, off), l.offset(src, off));
                            l.clone_into(d, s, t);
                        }
                    });
                }
            },
            Ty::Tuple(_) => {
                let elems = self.unit.types.as_tuple(ty).expect("a tuple").to_vec();
                let offsets = layout::fields_of(&self.unit.types, ty).offsets;
                for (&e, &off) in elems.iter().zip(&offsets) {
                    let (d, s) = (self.offset(dst, off), self.offset(src, off));
                    self.clone_into(d, s, e);
                }
            }
            Ty::Array(_) => {
                let (elem, n) = self.unit.types.as_array(ty).expect("an array");
                let size = self.layout(elem).size;
                let n = self.cx.i64_type().const_int(n, false);
                self.clone_each(dst, src, n, elem, size);
            }
            Ty::Growable(_) => {
                let elem = self.unit.types.as_growable(ty).expect("a growable array");
                let (data, len) = self.array_parts(src);
                let func = self.func.expect("inside a function");
                let copy = self.cx.append_basic_block(func, "clone.buffer");
                let empty = self.cx.append_basic_block(func, "clone.empty");
                let done = self.cx.append_basic_block(func, "clone.buffer.done");
                let i64 = self.cx.i64_type();
                let none = ok(self.b.build_int_compare(IntPredicate::EQ, len, i64.const_zero(), ""));
                ok(self.b.build_conditional_branch(none, empty, copy));

                self.b.position_at_end(empty);
                let nothing = self.array_value(self.null(), i64.const_zero());
                self.store_scalar(nothing, dst);
                ok(self.b.build_unconditional_branch(done));

                self.b.position_at_end(copy);
                let size = self.layout(elem).size;
                let bytes = self.bytes_for(len, size, Span::synthetic());
                let block = self.allocate(self.null(), bytes, Span::synthetic());
                self.write_header(block, len);
                let fresh = self.after_header(block);
                let align = self.layout(elem).align.min(16);
                self.clone_each(Addr { ptr: fresh, align }, Addr { ptr: data, align }, len, elem, size);
                let v = self.array_value(fresh, len);
                self.store_scalar(v, dst);
                ok(self.b.build_unconditional_branch(done));

                self.b.position_at_end(done);
            }
            _ => self.copy(dst, src, ty),
        }
    }

    /// Clones `n` elements of type `elem`, `size` bytes apart, from `src` to
    /// `dst`, first to last.
    fn clone_each(&mut self, dst: Addr<'ctx>, src: Addr<'ctx>, n: inkwell::values::IntValue<'ctx>, elem: Ty, size: u64) {
        let func = self.func.expect("inside a function");
        let i64 = self.cx.i64_type();
        let counter = self.named_temp(Ty::USIZE, "clone.i");
        self.store_scalar(i64.const_zero().into(), counter);
        let test = self.cx.append_basic_block(func, "clone.elems");
        let body = self.cx.append_basic_block(func, "clone.elem");
        let done = self.cx.append_basic_block(func, "clone.elems.done");
        ok(self.b.build_unconditional_branch(test));
        self.b.position_at_end(test);
        let i = ok(self.b.build_load(i64, counter.ptr, "")).into_int_value();
        let more = ok(self.b.build_int_compare(IntPredicate::ULT, i, n, ""));
        ok(self.b.build_conditional_branch(more, body, done));
        self.b.position_at_end(body);
        let bytes = ok(self.b.build_int_nuw_mul(i, i64.const_int(size, false), ""));
        let i8 = self.cx.i8_type();
        // SAFETY: `i` is below the count of elements at both addresses
        let (d, s) = unsafe {
            (
                ok(self.b.build_gep(i8, dst.ptr, &[bytes], "")),
                ok(self.b.build_gep(i8, src.ptr, &[bytes], "")),
            )
        };
        let align = self.layout(elem).align.min(dst.align).min(src.align);
        self.clone_into(Addr { ptr: d, align }, Addr { ptr: s, align }, elem);
        let next = ok(self.b.build_int_nuw_add(i, i64.const_int(1, false), ""));
        self.store_scalar(next.into(), counter);
        ok(self.b.build_unconditional_branch(test));
        self.b.position_at_end(done);
    }
}
