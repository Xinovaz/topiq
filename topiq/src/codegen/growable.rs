//! Growable arrays.
//!
//! A growable array is two words, the address of its first element and how
//! many elements it has, the same two words as a slice of them. Its elements
//! live in a buffer from the C runtime's allocator, which begins with a header
//! of two words (the capacity in elements, then a word set aside for the
//! element type's description, written null), so the elements start 16 bytes
//! in. An array that has never held anything has no buffer and a null
//! address.
//!
//! The buffer grows to at least twice its capacity when it is full, so a run
//! of `push`es copies each element a bounded number of times. When the
//! allocator cannot provide the memory the program aborts with `RA07`.
//!
//! The allocator returns addresses that are multiples of 16, so elements
//! aligned to more than that are not kept at their alignment. Every access to
//! an element states at most 16, which keeps the code correct; only the
//! placement the alignment asked for is lost.

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};

use crate::diag::Code;
use crate::span::Span;
use crate::tir::{Expr, GrowOp, Ty, layout};

use super::func::{Addr, Dest, Lowering, ok};
use super::types;

/// The bytes before element 0: the capacity and the element description.
pub(super) const HEADER: u64 = 16;

/// The capacity of a buffer's first allocation.
const FIRST_CAPACITY: u64 = 4;

impl<'ctx> Lowering<'ctx, '_> {
    /// An operation on the growable array `array`.
    pub(super) fn growable_op(
        &mut self,
        op: GrowOp,
        array: &Expr,
        args: &[Expr],
        ty: Ty,
        dest: Dest<'ctx>,
        span: Span,
    ) -> Option<BasicValueEnum<'ctx>> {
        let elem = self.unit.types.as_growable(array.ty).expect("a growable array");
        let at = self.held(array)?;
        match op {
            GrowOp::Push => {
                // the value first, in case making it reads the array
                let v = self.temp(elem);
                self.store_expr(&args[0], v);
                let (_, len) = self.array_parts(at);
                let one = self.cx.i64_type().const_int(1, false);
                let need = ok(self.b.build_int_nuw_add(len, one, ""));
                self.reserve_for(at, need, elem, span);
                let (data, len) = self.array_parts(at);
                let slot = self.element_at(data, len, elem);
                self.copy(slot, v, elem);
                self.set_len(at, need);
                None
            }
            GrowOp::Reserve => {
                let n = self.value(&args[0])?.into_int_value();
                let (_, len) = self.array_parts(at);
                let need = self.checked_add(len, n, span);
                self.reserve_for(at, need, elem, span);
                None
            }
            GrowOp::Pop => self.pop(at, elem, ty, dest),
            GrowOp::Clear => {
                let (data, len) = self.array_parts(at);
                let zero = self.cx.i64_type().const_zero();
                // the length first: should destroying an element read the
                // array, it sees only what is still there
                self.set_len(at, zero);
                self.drop_elements(data, zero, len, elem);
                None
            }
            GrowOp::Truncate => {
                let n = self.value(&args[0])?.into_int_value();
                let (data, len) = self.array_parts(at);
                let shorter = ok(self.b.build_int_compare(IntPredicate::ULT, n, len, ""));
                let keep = ok(self.b.build_select(shorter, n, len, "")).into_int_value();
                self.set_len(at, keep);
                self.drop_elements(data, keep, len, elem);
                None
            }
            GrowOp::Take => {
                let i = self.value(&args[0])?.into_int_value();
                let (data, len) = self.array_parts(at);
                let bad = ok(self.b.build_int_compare(IntPredicate::UGE, i, len, "out.of.bounds"));
                self.abort_if(bad, Code::Ra06, span);
                let src = self.element_at(data, i, elem);
                match dest {
                    Dest::Mem(d) => {
                        self.copy(d, src, elem);
                        None
                    }
                    Dest::Value => self.load_scalar(elem, src),
                }
            }
            GrowOp::ForgetFront => {
                let n = self.value(&args[0])?.into_int_value();
                let (data, len) = self.array_parts(at);
                let rest = ok(self.b.build_int_nuw_sub(len, n, ""));
                let size = self.cx.i64_type().const_int(self.layout(elem).size, false);
                let bytes = ok(self.b.build_int_nuw_mul(rest, size, ""));
                let from = self.element_at(data, n, elem);
                let align = self.layout(elem).align.min(HEADER) as u32;
                ok(self.b.build_memmove(data, align, from.ptr, align, bytes));
                self.set_len(at, rest);
                None
            }
        }
    }

    /// A fixed array's elements moved into a new buffer.
    pub(super) fn grow(&mut self, array: &Expr, span: Span) -> Option<BasicValueEnum<'ctx>> {
        let (elem, n) = self.unit.types.as_array(array.ty).expect("a fixed array");
        let i64 = self.cx.i64_type();
        if n == 0 {
            // nothing to hold, but the array is still evaluated
            let t = self.temp(array.ty);
            self.store_expr(array, t);
            return Some(self.array_value(self.null(), i64.const_zero()));
        }
        let bytes = i64.const_int(HEADER + n * self.layout(elem).size, false);
        let block = self.allocate(self.null(), bytes, span);
        let data = self.after_header(block);
        self.write_header(block, i64.const_int(n, false));
        self.store_expr(
            array,
            Addr {
                ptr: data,
                align: self.layout(elem).align.min(HEADER),
            },
        );
        Some(self.array_value(data, i64.const_int(n, false)))
    }

    /// The body of a growable array's destroying function: its elements,
    /// last first, then its buffer.
    pub(super) fn destroy_growable(&mut self, at: Addr<'ctx>, ty: Ty) {
        let elem = self.unit.types.as_growable(ty).expect("a growable array");
        let (data, len) = self.array_parts(at);
        self.drop_elements(data, self.cx.i64_type().const_zero(), len, elem);
        let func = self.func.expect("inside a function");
        let free = self.cx.append_basic_block(func, "grow.free");
        let done = self.cx.append_basic_block(func, "grow.freed");
        let empty = ok(self.b.build_is_null(data, ""));
        ok(self.b.build_conditional_branch(empty, done, free));
        self.b.position_at_end(free);
        let block = self.header_of(data);
        let f = self.free_fn();
        ok(self.b.build_call(f, &[block.into()], ""));
        ok(self.b.build_unconditional_branch(done));
        self.b.position_at_end(done);
    }

    /// `xs.pop()`: `None`, or the last element as `Some`.
    fn pop(&mut self, at: Addr<'ctx>, elem: Ty, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let Ty::Adt(opt) = ty else {
            unreachable!("`pop` gives an `Opt`")
        };
        let out = match dest {
            Dest::Mem(a) => a,
            Dest::Value => self.temp(ty),
        };

        // the `Opt` it gives, and its two variants
        let l = layout::enumeration(&self.unit.types, opt);
        let variants = match &self.unit.types.adt(opt).kind {
            crate::tir::AdtKind::Enum { variants } => variants.clone(),
            crate::tir::AdtKind::Struct { .. } => unreachable!("`Opt` is an enumeration"),
        };
        let some = variants.iter().position(|v| !v.fields.is_empty()).expect("`Some` holds a value");
        let none = variants.iter().position(|v| v.fields.is_empty()).expect("`None` holds nothing");

        // branch on whether the array is empty
        let func = self.func.expect("inside a function");
        let has = self.cx.append_basic_block(func, "pop.some");
        let empty = self.cx.append_basic_block(func, "pop.none");
        let done = self.cx.append_basic_block(func, "pop.done");
        let (data, len) = self.array_parts(at);
        let zero = self.cx.i64_type().const_zero();
        let is_empty = ok(self.b.build_int_compare(IntPredicate::EQ, len, zero, ""));
        ok(self.b.build_conditional_branch(is_empty, empty, has));

        // empty: `None`
        self.b.position_at_end(empty);
        let tag = types::const_int(self.cx, l.tag, none as i128);
        self.store_scalar(tag.into(), out);
        ok(self.b.build_unconditional_branch(done));

        // otherwise: shorten by one, and `Some` of the element left behind
        self.b.position_at_end(has);
        let one = self.cx.i64_type().const_int(1, false);
        let last = ok(self.b.build_int_nuw_sub(len, one, ""));
        self.set_len(at, last);
        let tag = types::const_int(self.cx, l.tag, some as i128);
        self.store_scalar(tag.into(), out);
        let src = self.element_at(data, last, elem);
        let payload = self.offset(out, l.variants[some][0]);
        self.copy(payload, src, elem);
        ok(self.b.build_unconditional_branch(done));

        self.b.position_at_end(done);
        None
    }

    /// Makes the buffer of the array at `at` hold at least `need` elements.
    fn reserve_for(&mut self, at: Addr<'ctx>, need: IntValue<'ctx>, elem: Ty, span: Span) {
        let func = self.func.expect("inside a function");
        let (data, _) = self.array_parts(at);
        let cap = self.capacity(data);
        let grow = self.cx.append_basic_block(func, "grow");
        let done = self.cx.append_basic_block(func, "grow.done");
        let short = ok(self.b.build_int_compare(IntPredicate::UGT, need, cap, ""));
        ok(self.b.build_conditional_branch(short, grow, done));

        self.b.position_at_end(grow);
        let i64 = self.cx.i64_type();
        // at least twice as many, and never fewer than a handful
        let doubled = ok(self.b.build_int_mul(cap, i64.const_int(2, false), ""));
        let more = ok(self.b.build_int_compare(IntPredicate::UGT, doubled, need, ""));
        let new_cap = ok(self.b.build_select(more, doubled, need, "")).into_int_value();
        let small = ok(self.b.build_int_compare(IntPredicate::ULT, new_cap, i64.const_int(FIRST_CAPACITY, false), ""));
        let new_cap = ok(self.b.build_select(small, i64.const_int(FIRST_CAPACITY, false), new_cap, "")).into_int_value();
        let size = self.layout(elem).size;
        let bytes = self.bytes_for(new_cap, size, span);
        let old = self.block_of(data);
        let block = self.allocate(old, bytes, span);
        self.write_header(block, new_cap);
        let data = self.after_header(block);
        let slot = ok(self.b.build_struct_gep(types::fat_slice(self.cx), at.ptr, 0, ""));
        ok(self.b.build_store(slot, data));
        ok(self.b.build_unconditional_branch(done));

        self.b.position_at_end(done);
    }

    /// `HEADER + n * size`, aborting with `RA07` if it does not fit in a
    /// word: no allocator could provide that much.
    pub(super) fn bytes_for(&mut self, n: IntValue<'ctx>, size: u64, span: Span) -> IntValue<'ctx> {
        let i64 = self.cx.i64_type();
        let (product, over) = self.word_overflowing("mul", n, i64.const_int(size, false));
        let (total, over2) = self.word_overflowing("add", product, i64.const_int(HEADER, false));
        let bad = ok(self.b.build_or(over, over2, ""));
        self.abort_if(bad, Code::Ra07, span);
        total
    }

    /// `a + b`, aborting with `RA07` if the sum does not fit in a word: a
    /// capacity that large could never be allocated.
    fn checked_add(&mut self, a: IntValue<'ctx>, b: IntValue<'ctx>, span: Span) -> IntValue<'ctx> {
        let (sum, over) = self.word_overflowing("add", a, b);
        self.abort_if(over, Code::Ra07, span);
        sum
    }

    /// An unsigned 64-bit operation and whether it overflowed.
    fn word_overflowing(&mut self, what: &str, a: IntValue<'ctx>, b: IntValue<'ctx>) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let t = crate::tir::IntTy::U64;
        self.with_overflow(what, t, a, b)
    }

    /// `realloc(old, bytes)`, aborting with `RA07` if it fails.
    pub(super) fn allocate(&mut self, old: PointerValue<'ctx>, bytes: IntValue<'ctx>, span: Span) -> PointerValue<'ctx> {
        let f = self.realloc_fn();
        let block = ok(self.b.build_call(f, &[old.into(), bytes.into()], ""))
            .try_as_basic_value()
            .basic()
            .expect("realloc returns an address")
            .into_pointer_value();
        let failed = ok(self.b.build_is_null(block, ""));
        self.abort_if(failed, Code::Ra07, span);
        block
    }

    /// The C runtime's `realloc`.
    fn realloc_fn(&self) -> FunctionValue<'ctx> {
        let ptr = self.cx.ptr_type(AddressSpace::default());
        self.module.get_function("realloc").unwrap_or_else(|| {
            self.module.add_function(
                "realloc",
                ptr.fn_type(&[ptr.into(), self.cx.i64_type().into()], false),
                Some(LlvmLinkage::External),
            )
        })
    }

    /// Writes a buffer's header: its capacity, and no element description.
    pub(super) fn write_header(&mut self, block: PointerValue<'ctx>, cap: IntValue<'ctx>) {
        let whole = Addr { ptr: block, align: 8 };
        self.store_scalar(cap.into(), whole);
        let info = self.offset(whole, 8);
        self.store_scalar(self.null().into(), info);
    }

    /// The capacity of the buffer whose elements start at `data`: 0 when
    /// there is none.
    fn capacity(&mut self, data: PointerValue<'ctx>) -> IntValue<'ctx> {
        let func = self.func.expect("inside a function");
        let before = self.b.get_insert_block().expect("inside a block");
        let read = self.cx.append_basic_block(func, "grow.cap");
        let done = self.cx.append_basic_block(func, "grow.cap.done");
        let empty = ok(self.b.build_is_null(data, ""));
        ok(self.b.build_conditional_branch(empty, done, read));
        self.b.position_at_end(read);
        let block = self.header_of(data);
        let i64 = self.cx.i64_type();
        let cap = ok(self.b.build_load(i64, block, "cap")).into_int_value();
        ok(self.b.build_unconditional_branch(done));
        self.b.position_at_end(done);
        let phi = ok(self.b.build_phi(i64, ""));
        phi.add_incoming(&[(&i64.const_zero(), before), (&cap, read)]);
        phi.as_basic_value().into_int_value()
    }

    /// The buffer holding elements that start at `data`, or null when there
    /// is none.
    fn block_of(&mut self, data: PointerValue<'ctx>) -> PointerValue<'ctx> {
        let empty = ok(self.b.build_is_null(data, ""));
        let block = self.header_of(data);
        ok(self.b.build_select(empty, self.null(), block, "")).into_pointer_value()
    }

    /// The start of the buffer whose elements start at `data`.
    fn header_of(&mut self, data: PointerValue<'ctx>) -> PointerValue<'ctx> {
        let back = self.cx.i64_type().const_int(HEADER.wrapping_neg(), true);
        // SAFETY: only used where `data` is at `HEADER` bytes into a buffer,
        // or, in `block_of`, discarded when it is not
        unsafe { ok(self.b.build_gep(self.cx.i8_type(), data, &[back], "")) }
    }

    /// Where the elements of a buffer start.
    pub(super) fn after_header(&mut self, block: PointerValue<'ctx>) -> PointerValue<'ctx> {
        let fwd = self.cx.i64_type().const_int(HEADER, false);
        // SAFETY: every buffer is at least `HEADER` bytes long
        unsafe { ok(self.b.build_in_bounds_gep(self.cx.i8_type(), block, &[fwd], "")) }
    }

    /// The address of element `i`.
    pub(super) fn element_at(&mut self, data: PointerValue<'ctx>, i: IntValue<'ctx>, elem: Ty) -> Addr<'ctx> {
        let l = self.layout(elem);
        let bytes = ok(self.b.build_int_nuw_mul(i, self.cx.i64_type().const_int(l.size, false), ""));
        // SAFETY: callers pass an index within the buffer
        let ptr = unsafe { ok(self.b.build_gep(self.cx.i8_type(), data, &[bytes], "")) };
        Addr {
            ptr,
            align: l.align.min(HEADER),
        }
    }

    /// Destroys the elements from `from` up to `to`, last first.
    fn drop_elements(&mut self, data: PointerValue<'ctx>, from: IntValue<'ctx>, to: IntValue<'ctx>, elem: Ty) {
        if !self.needs_drop(elem) {
            return;
        }
        let func = self.func.expect("inside a function");
        let i64 = self.cx.i64_type();
        let counter = self.named_temp(Ty::USIZE, "drop.i");
        self.store_scalar(to.into(), counter);
        let test = self.cx.append_basic_block(func, "drop.elems");
        let body = self.cx.append_basic_block(func, "drop.elem");
        let done = self.cx.append_basic_block(func, "drop.elems.done");
        ok(self.b.build_unconditional_branch(test));
        self.b.position_at_end(test);
        let i = ok(self.b.build_load(i64, counter.ptr, "")).into_int_value();
        let more = ok(self.b.build_int_compare(IntPredicate::UGT, i, from, ""));
        ok(self.b.build_conditional_branch(more, body, done));
        self.b.position_at_end(body);
        let prev = ok(self.b.build_int_nuw_sub(i, i64.const_int(1, false), ""));
        self.store_scalar(prev.into(), counter);
        let at = self.element_at(data, prev, elem);
        self.drop_at(at, elem);
        ok(self.b.build_unconditional_branch(test));
        self.b.position_at_end(done);
    }

    /// The address of the first element and the count, read from the array
    /// at `at`.
    pub(super) fn array_parts(&mut self, at: Addr<'ctx>) -> (PointerValue<'ctx>, IntValue<'ctx>) {
        let s = ok(self.b.build_load(types::fat_slice(self.cx), at.ptr, "")).into_struct_value();
        self.slice_parts(s)
    }

    /// Stores a new count into the array at `at`.
    fn set_len(&mut self, at: Addr<'ctx>, len: IntValue<'ctx>) {
        let slot = ok(self.b.build_struct_gep(types::fat_slice(self.cx), at.ptr, 1, ""));
        ok(self.b.build_store(slot, len));
    }

    /// A growable array's two words.
    pub(super) fn array_value(&self, data: PointerValue<'ctx>, len: IntValue<'ctx>) -> BasicValueEnum<'ctx> {
        let fat = types::fat_slice(self.cx).get_undef();
        let fat = ok(self.b.build_insert_value(fat, data, 0, "")).into_struct_value();
        ok(self.b.build_insert_value(fat, len, 1, "")).into_struct_value().into()
    }

    /// The null address.
    pub(super) fn null(&self) -> PointerValue<'ctx> {
        self.cx.ptr_type(AddressSpace::default()).const_null()
    }
}
