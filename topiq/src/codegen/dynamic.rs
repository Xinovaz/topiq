//! Dynamic values: `dyn`, and what the `dyn` library is built from.
//!
//! A `dyn` is two words: the address of a buffer holding its value, and the
//! value's table. `dyn::of` moves a value into a fresh buffer; taking the
//! value out moves it back and frees the buffer; destroying a `dyn` runs the
//! destroying function its table records, then frees the buffer.
//!
//! # Calling a method with `dyn` arguments
//!
//! A table lists each method's code, which `invoke` calls with arguments of
//! types the caller wrote. `dyn::call` has only `dyn` values, whose types it
//! learns when the program runs, so each listed method also gets a `call`
//! entry: a small function taking the receiver as `*any` and the arguments as
//! a slice of `dyn`, reading each argument out of its buffer as the type the
//! method takes, calling it, and putting the result in a `dyn` of its own.
//! The library checks the arguments' types against the method's signature
//! first. Arguments are read out by copying, so only a method whose
//! parameters can all be copied gets such an entry; and `dyn::call` has only
//! a shared reference to the value, so only a method taking `*self` does.

use inkwell::IntPredicate;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, PointerValue, StructValue};

use crate::span::Span;
use crate::tir::{Access, Expr, TableMethod, Ty, layout};

use super::func::{Addr, Dest, Lowering, ok};
use super::types;

impl<'ctx> Lowering<'ctx, '_> {
    /// `dyn::of(v)`: `v` moved into a buffer of its own.
    pub(super) fn dyn_of(&mut self, v: &Expr, ty: Ty, span: Span) -> Option<BasicValueEnum<'ctx>> {
        let size = self.layout(ty).size.max(1);
        let block = self.allocate(self.null(), self.cx.i64_type().const_int(size, false), span);
        let align = self.layout(ty).align.min(16);
        self.store_expr(v, Addr { ptr: block, align });
        let table = self.table(ty);
        Some(self.pair(block, table).into())
    }

    /// The value a `dyn` holds.
    pub(super) fn dyn_take(&mut self, d: &Expr, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let v = self.value(d)?.into_struct_value();
        let value = ok(self.b.build_extract_value(v, 0, "value")).into_pointer_value();
        let at = Addr {
            ptr: value,
            align: self.layout(ty).align.min(16),
        };
        let out = match dest {
            Dest::Mem(a) => {
                self.copy(a, at, ty);
                None
            }
            Dest::Value => self.load_scalar(ty, at),
        };
        let free = self.free_fn();
        ok(self.b.build_call(free, &[value.into()], ""));
        out
    }

    /// `d as T`: `Some` of the value a `dyn` holds when its table is `T`'s,
    /// the buffer freed; otherwise `None`, the `dyn` destroyed.
    pub(super) fn dyn_as(&mut self, d: &Expr, target: Ty, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        // compare the table carried with `target`'s
        let v = self.value(d)?.into_struct_value();
        let value = ok(self.b.build_extract_value(v, 0, "value")).into_pointer_value();
        let carried = ok(self.b.build_extract_value(v, 1, "carried")).into_pointer_value();
        let wanted = self.table(target);
        let same = self.same_type(carried, wanted);

        // the `Opt` it gives, and its two variants
        let Ty::Adt(opt) = ty else {
            unreachable!("`as` on a `dyn` gives an `Opt`")
        };
        let out = match dest {
            Dest::Mem(a) => a,
            Dest::Value => self.temp(ty),
        };
        let def = self.unit.types.adt(opt).clone();
        let some = def.variants().iter().position(|v| !v.fields.is_empty()).expect("`Some`");
        let none = def.variants().iter().position(|v| v.fields.is_empty()).expect("`None`");
        let l = layout::enumeration(&self.unit.types, opt);
        let func = self.func.expect("inside a function");
        let yes = self.cx.append_basic_block(func, "dyn.as.some");
        let no = self.cx.append_basic_block(func, "dyn.as.none");
        let done = self.cx.append_basic_block(func, "dyn.as.done");
        ok(self.b.build_conditional_branch(same, yes, no));

        // same: `Some` of the value, moved out of the buffer
        self.b.position_at_end(yes);
        self.store_scalar(types::const_int(self.cx, l.tag, some as i128).into(), out);
        let payload = self.offset(out, l.variants[some][0]);
        let at = Addr {
            ptr: value,
            align: self.layout(target).align.min(16),
        };
        self.copy(payload, at, target);
        let free = self.free_fn();
        ok(self.b.build_call(free, &[value.into()], ""));
        ok(self.b.build_unconditional_branch(done));

        // otherwise: `None`, and the `dyn` destroyed
        self.b.position_at_end(no);
        self.store_scalar(types::const_int(self.cx, l.tag, none as i128).into(), out);
        let held = self.temp(Ty::Dyn);
        self.store_scalar(v.into(), held);
        self.drop_at(held, Ty::Dyn);
        ok(self.b.build_unconditional_branch(done));

        self.b.position_at_end(done);
        None
    }

    /// A reference to a `dyn` as a reference to what it holds.
    pub(super) fn dyn_view(&mut self, r: &Expr) -> Option<BasicValueEnum<'ctx>> {
        let v = self.value(r)?.into_struct_value();
        let slot = ok(self.b.build_extract_value(v, 0, "")).into_pointer_value();
        Some(ok(self.b.build_load(types::fat_ref(self.cx), slot, "held")))
    }

    /// `(p, offset, ty)`: a reference `offset` bytes into what `p` refers
    /// to, carrying the table `ty` refers to.
    pub(super) fn field_at(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let p = self.value(&args[0])?.into_struct_value();
        let offset = self.value(&args[1])?.into_int_value();
        let t = self.value(&args[2])?.into_struct_value();
        let base = ok(self.b.build_extract_value(p, 0, "")).into_pointer_value();
        let table = ok(self.b.build_extract_value(t, 0, "")).into_pointer_value();
        // SAFETY: the library gives only the offsets of fields of the value
        let at = unsafe { ok(self.b.build_gep(self.cx.i8_type(), base, &[offset], "")) };
        Some(self.pair(at, table).into())
    }

    /// `(call, self, args)`: a method's `call` entry called.
    pub(super) fn dyn_call(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let call = self.value(&args[0])?.into_struct_value();
        let this = self.value(&args[1])?;
        let rest = self.value(&args[2])?;
        let code = ok(self.b.build_extract_value(call, 0, "")).into_pointer_value();
        let fat = types::fat_ref(self.cx);
        let fn_type = fat.fn_type(&[fat.into(), types::fat_slice(self.cx).into()], false);
        let r = ok(self.b.build_indirect_call(fn_type, code, &[this.into(), rest.into()], ""));
        r.try_as_basic_value().basic()
    }

    /// A `dyn` holding a value of the type a table describes, all of whose
    /// bytes are zero.
    pub(super) fn dyn_make(&mut self, t: &Expr, span: Span) -> Option<BasicValueEnum<'ctx>> {
        let t = self.value(t)?.into_struct_value();
        let table = ok(self.b.build_extract_value(t, 0, "")).into_pointer_value();
        let i64 = self.cx.i64_type();
        let size = self.table_entry(table, 1, i64.into()).expect("a table is at hand").into_int_value();
        let one = i64.const_int(1, false);
        let small = ok(self.b.build_int_compare(IntPredicate::ULT, size, one, ""));
        let bytes = ok(self.b.build_select(small, one, size, "")).into_int_value();
        let block = self.allocate(self.null(), bytes, span);
        ok(self.b.build_memset(block, 1, self.cx.i8_type().const_zero(), bytes));
        Some(self.pair(block, table).into())
    }

    /// `(p, v)`: the old value `p` refers to destroyed, the value `v` holds
    /// moved in its place, and `v`'s buffer freed. The library has checked
    /// that the two are of one type.
    pub(super) fn dyn_store(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let p = self.value(&args[0])?.into_struct_value();
        let v = self.value(&args[1])?.into_struct_value();
        let place = ok(self.b.build_extract_value(p, 0, "place")).into_pointer_value();
        let table = ok(self.b.build_extract_value(p, 1, "table")).into_pointer_value();
        let buffer = ok(self.b.build_extract_value(v, 0, "buffer")).into_pointer_value();
        let func = self.func.expect("inside a function");
        let glue = self.cx.append_basic_block(func, "store.drop");
        let moved = self.cx.append_basic_block(func, "store.move");
        self.destroy_through(table, place, glue, moved);
        self.b.position_at_end(moved);
        let i64 = self.cx.i64_type();
        let size = self.table_entry(table, 1, i64.into()).expect("a table is at hand").into_int_value();
        ok(self.b.build_memcpy(place, 1, buffer, 1, size));
        let free = self.free_fn();
        ok(self.b.build_call(free, &[buffer.into()], ""));
        None
    }

    /// Whether a reference's address is null.
    pub(super) fn is_null(&mut self, r: &Expr) -> Option<BasicValueEnum<'ctx>> {
        let r = self.value(r)?.into_struct_value();
        let at = ok(self.b.build_extract_value(r, 0, "")).into_pointer_value();
        Some(ok(self.b.build_is_null(at, "")).into())
    }

    /// `@raw_parts(r)`: the reference's address as an integer, then its
    /// length or its table, written as the tuple they make.
    pub(super) fn raw_parts(&mut self, r: &Expr, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let v = self.value(r)?.into_struct_value();
        let at = ok(self.b.build_extract_value(v, 0, "")).into_pointer_value();
        let second = ok(self.b.build_extract_value(v, 1, ""));
        let addr = ok(self.b.build_ptr_to_int(at, self.cx.i64_type(), "address"));
        let out = match dest {
            Dest::Mem(a) => a,
            Dest::Value => self.temp(ty),
        };
        let offsets = layout::fields_of(&self.unit.types, ty).offsets;
        self.store_scalar(addr.into(), out);
        let place = self.offset(out, offsets[1]);
        let second = match self.unit.types.as_tuple(ty).map(|f| f[1]) {
            // a table reference is its address and `TypeInfo`'s own table
            Some(t) if self.unit.types.as_ref(t).is_some() => self.typeinfo_value(second.into_pointer_value()).into(),
            _ => second,
        };
        self.store_scalar(second, place);
        None
    }

    /// `@from_raw_parts(addr, n)`: two words made from an address and a
    /// length or a table's address.
    pub(super) fn assemble_raw_parts(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let addr = self.value(&args[0])?.into_int_value();
        let second = self.value(&args[1])?;
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let at = ok(self.b.build_int_to_ptr(addr, ptr, "at"));
        let (shape, second) = match second {
            BasicValueEnum::StructValue(t) => (types::fat_ref(self.cx), ok(self.b.build_extract_value(t, 0, "table"))),
            n => (types::fat_slice(self.cx), n),
        };
        let fat = ok(self.b.build_insert_value(shape.get_undef(), at, 0, "")).into_struct_value();
        let fat = ok(self.b.build_insert_value(fat, second, 1, "")).into_struct_value();
        Some(fat.into())
    }

    /// `(place, n)`: `n` zeroed elements given to the empty growable array
    /// `place` refers to, whose element type its table records; a reference
    /// to the first.
    pub(super) fn grow_zeroed(&mut self, args: &[Expr], span: Span) -> Option<BasicValueEnum<'ctx>> {
        let place = self.value(&args[0])?.into_struct_value();
        let n = self.value(&args[1])?.into_int_value();
        let at = ok(self.b.build_extract_value(place, 0, "")).into_pointer_value();
        let table = ok(self.b.build_extract_value(place, 1, "")).into_pointer_value();
        let fat = types::fat_ref(self.cx);
        let elem = self.table_entry(table, 10, fat.into()).expect("a table is at hand").into_struct_value();
        let elem = ok(self.b.build_extract_value(elem, 0, "elem")).into_pointer_value();
        let i64 = self.cx.i64_type();
        let size = self.table_entry(elem, 1, i64.into()).expect("a table is at hand").into_int_value();
        let body = ok(self.b.build_int_mul(n, size, ""));
        let header = i64.const_int(super::growable::HEADER, false);
        let bytes = ok(self.b.build_int_add(body, header, ""));
        let block = self.allocate(self.null(), bytes, span);
        ok(self.b.build_memset(block, 1, self.cx.i8_type().const_zero(), bytes));
        self.write_header(block, n);
        let data = self.after_header(block);
        let array = self.array_value(data, n);
        self.store_scalar(array, Addr { ptr: at, align: 8 });
        Some(self.pair(data, elem).into())
    }

    /// `(x, count, buf)`: the C runtime's leading decimal digits of `x`,
    /// and where its decimal point falls.
    pub(super) fn float_digits(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let x = self.value(&args[0])?;
        let count = self.value(&args[1])?;
        let buf = self.value(&args[2])?.into_struct_value();
        let buf = ok(self.b.build_extract_value(buf, 0, "")).into_pointer_value();
        let i32 = self.cx.i32_type();
        let i64 = self.cx.i64_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let f64 = self.cx.f64_type();
        let decpt = self.temp(Ty::Int(crate::tir::IntTy::I32));
        let sign = self.temp(Ty::Int(crate::tir::IntTy::I32));
        let size = i64.const_int(40, false);
        if self.windows() {
            let ty = i32.fn_type(&[ptr.into(), i64.into(), f64.into(), i32.into(), ptr.into(), ptr.into()], false);
            let f = self.runtime_function("_ecvt_s", ty);
            ok(self.b.build_call(f, &[buf.into(), size.into(), x.into(), count.into(), decpt.ptr.into(), sign.ptr.into()], ""));
        } else {
            let ty = i32.fn_type(&[f64.into(), i32.into(), ptr.into(), ptr.into(), ptr.into(), i64.into()], false);
            let f = self.runtime_function("ecvt_r", ty);
            ok(self.b.build_call(f, &[x.into(), count.into(), decpt.ptr.into(), sign.ptr.into(), buf.into(), size.into()], ""));
        }
        Some(ok(self.b.build_load(i32, decpt.ptr, "decpt")))
    }

    /// `(text)`: the C runtime's reading of a zero-ended numeral.
    pub(super) fn parse_float(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let text = self.value(&args[0])?.into_struct_value();
        let (at, _) = self.slice_parts(text);
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let ty = self.cx.f64_type().fn_type(&[ptr.into(), ptr.into()], false);
        let f = self.runtime_function("strtod", ty);
        let r = ok(self.b.build_call(f, &[at.into(), ptr.const_null().into()], "value"));
        r.try_as_basic_value().basic()
    }

    /// `name(x)` for one of the C runtime's functions from `f64` to `f64`.
    pub(super) fn libm(&mut self, name: &str, x: &Expr) -> Option<BasicValueEnum<'ctx>> {
        let x = self.value(x)?;
        let f64 = self.cx.f64_type();
        let f = self.runtime_function(name, f64.fn_type(&[f64.into()], false));
        ok(self.b.build_call(f, &[x.into()], "")).try_as_basic_value().basic()
    }

    /// The system clock: `GetSystemTimePreciseAsFileTime`, which writes the
    /// time as a 64-bit count of 100-nanosecond ticks.
    pub(super) fn clock(&mut self) -> Option<BasicValueEnum<'ctx>> {
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let at = self.temp(Ty::Int(crate::tir::IntTy::U64));
        let ty = self.cx.void_type().fn_type(&[ptr.into()], false);
        let f = self.runtime_function("GetSystemTimePreciseAsFileTime", ty);
        ok(self.b.build_call(f, &[at.ptr.into()], ""));
        Some(ok(self.b.build_load(self.cx.i64_type(), at.ptr, "clock")))
    }

    /// A function of the C runtime.
    pub(super) fn runtime_function(&self, name: &str, ty: inkwell::types::FunctionType<'ctx>) -> FunctionValue<'ctx> {
        self.module
            .get_function(name)
            .unwrap_or_else(|| self.module.add_function(name, ty, Some(LlvmLinkage::External)))
    }

    /// `x`, a reference to a fixed array or a slice, erased to `to`, a
    /// `*any`: its second word, a count, becomes a table. A reference to an
    /// array carries the array type's; a slice's words are put where the
    /// reference can point, and read as a growable array of the elements.
    /// `None` for any other conversion, which changes no words.
    pub(super) fn erase_counted(&mut self, x: &Expr, to: Ty) -> Option<Option<BasicValueEnum<'ctx>>> {
        let types = &self.unit.types;
        let erased = types.as_ref(to).is_some_and(|(_, t)| t == Ty::Void);
        if !erased || !types::counts(types, x.ty) {
            return None;
        }
        let array = types.as_ref(x.ty).map(|(_, a)| a);
        let whole = match array {
            Some(a) => a,
            None => {
                let (_, elem) = types.as_slice(x.ty).expect("a slice");
                types.find_growable(elem).expect("analysis made the growable array type")
            }
        };
        let Some(v) = self.value(x) else { return Some(None) };
        let v = v.into_struct_value();
        let (at, _) = self.slice_parts(v);
        let at = if array.is_some() {
            at
        } else {
            let words = self.temp(whole);
            self.store_scalar(v.into(), words);
            words.ptr
        };
        let table = self.table(whole);
        Some(Some(self.pair(at, table).into()))
    }

    /// Two words, as a reference or a `dyn` is.
    pub(super) fn pair(&self, a: PointerValue<'ctx>, b: PointerValue<'ctx>) -> StructValue<'ctx> {
        let fat = types::fat_ref(self.cx).get_undef();
        let fat = ok(self.b.build_insert_value(fat, a, 0, "")).into_struct_value();
        ok(self.b.build_insert_value(fat, b, 1, "")).into_struct_value()
    }

    /// The `call` entry of a method a table lists: `None` for one with no
    /// receiver, or with a parameter that cannot be copied out of a `dyn`.
    pub(super) fn call_thunk(&mut self, m: &TableMethod, list: &str, index: usize) -> Option<FunctionValue<'ctx>> {
        let (params, ret) = self.unit.types.as_sig(m.sig).map(|(p, r)| (p.to_vec(), r))?;
        let (first, rest) = params.split_first()?;
        // `dyn::call` holds a `*dyn`, so a method taking `*self` or
        // `*const self` can be called that way, and one taking `self` cannot
        if !matches!(self.unit.types.as_ref(*first), Some((Access::Write | Access::Const, Ty::Void))) {
            return None;
        }
        if !rest.iter().all(|&p| self.unit.types.is_copyable(p)) {
            return None;
        }
        let (callee, sret) = self.callee_function(m.callee);
        let fat = types::fat_ref(self.cx);
        let slice = types::fat_slice(self.cx);
        let thunk = self.module.add_function(
            &format!("{list}$call{index}"),
            fat.fn_type(&[fat.into(), slice.into()], false),
            Some(LlvmLinkage::Private),
        );
        let saved = self.enter_fn(thunk);
        let this = thunk.get_nth_param(0).expect("the receiver");
        let args = thunk.get_nth_param(1).expect("the arguments").into_struct_value();
        let (items, _) = self.slice_parts(args);
        let ret_size = self.layout(ret).size.max(1);
        let i64 = self.cx.i64_type();
        let out = (ret != Ty::Void).then(|| self.allocate(self.null(), i64.const_int(ret_size, false), Span::synthetic()));
        let mut values: Vec<BasicMetadataValueEnum<'ctx>> = Vec::new();
        if sret {
            values.push(out.expect("a result").into());
        }
        values.push(this.into());
        for (i, &p) in rest.iter().enumerate() {
            let at = i64.const_int(i as u64, false);
            // SAFETY: the library checked that there are as many arguments
            // as parameters
            let slot = unsafe { ok(self.b.build_gep(fat, items, &[at], "")) };
            let d = ok(self.b.build_load(fat, slot, "")).into_struct_value();
            let value = ok(self.b.build_extract_value(d, 0, "")).into_pointer_value();
            let src = Addr {
                ptr: value,
                align: self.layout(p).align.min(16),
            };
            if p.is_aggregate() {
                // the callee owns a copy for the call
                let copy = self.temp(p);
                self.copy(copy, src, p);
                values.push(copy.ptr.into());
            } else if let Some(v) = self.load_scalar(p, src) {
                values.push(v.into());
            }
        }
        let r = ok(self.b.build_call(callee, &values, ""));
        let result = match out {
            None => {
                let table = self.table(Ty::Void);
                self.pair(self.null(), table)
            }
            Some(block) => {
                if !sret && let Some(v) = r.try_as_basic_value().basic() {
                    self.store_scalar(v, Addr { ptr: block, align: self.layout(ret).align.min(16) });
                }
                let table = self.table(ret);
                self.pair(block, table)
            }
        };
        ok(self.b.build_return(Some(&result)));
        self.leave_fn(saved);
        Some(thunk)
    }
}
