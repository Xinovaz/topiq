//! Reading and writing files.
//!
//! Each operation is one call of the operating system's, made directly: the
//! `fs` library, written in Topiq, turns paths into the form the system takes
//! and the results into its own `Result`s. A failure gives the system's error
//! code negated, so that one `i64` carries either a result or why there is
//! none; the library reads the code.
//!
//! Windows names files in UTF-16, which is why a path arrives as a zero-ended
//! array of `u16`: `CreateFileW` then opens exactly the file a program names,
//! whatever characters its name has.

use inkwell::IntPredicate;
use inkwell::values::{BasicValueEnum, IntValue};

use crate::tir::{Expr, IntTy, Ty};

use super::func::{Lowering, ok};

/// `GENERIC_READ` and `GENERIC_WRITE`: what a handle is opened to do.
const GENERIC_READ: u64 = 0x8000_0000;
const GENERIC_WRITE: u64 = 0x4000_0000;
/// `FILE_SHARE_READ`: others may read the file while it is open.
const SHARE_READ: u64 = 1;
/// `OPEN_EXISTING` and `CREATE_ALWAYS`: open a file that is there, or make
/// one empty whether or not it was.
const OPEN_EXISTING: u64 = 3;
const CREATE_ALWAYS: u64 = 2;
/// `FILE_ATTRIBUTE_NORMAL`.
const NORMAL: u64 = 0x80;

impl<'ctx> Lowering<'ctx, '_> {
    /// `(path, write)`: `CreateFileW`, giving the handle as an integer.
    pub(super) fn file_open(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let path = self.value(&args[0])?.into_struct_value();
        let write = self.value(&args[1])?.into_int_value();
        let (at, _) = self.slice_parts(path);
        let i32 = self.cx.i32_type();
        let i64 = self.cx.i64_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let pick = |b: &inkwell::builder::Builder<'ctx>, yes: u64, no: u64| {
            ok(b.build_select(write, i32.const_int(yes, false), i32.const_int(no, false), ""))
        };
        let access = pick(&self.b, GENERIC_WRITE, GENERIC_READ);
        let disposition = pick(&self.b, CREATE_ALWAYS, OPEN_EXISTING);
        let ty = ptr.fn_type(
            &[ptr.into(), i32.into(), i32.into(), ptr.into(), i32.into(), i32.into(), ptr.into()],
            false,
        );
        let create = self.runtime_function("CreateFileW", ty);
        let handle = ok(self.b.build_call(
            create,
            &[
                at.into(),
                access.into(),
                i32.const_int(SHARE_READ, false).into(),
                ptr.const_null().into(),
                disposition.into(),
                i32.const_int(NORMAL, false).into(),
                ptr.const_null().into(),
            ],
            "handle",
        ))
        .try_as_basic_value()
        .basic()
        .expect("a handle")
        .into_pointer_value();
        let handle = ok(self.b.build_ptr_to_int(handle, i64, "handle"));
        // `INVALID_HANDLE_VALUE` is all ones
        let failed = ok(self.b.build_int_compare(IntPredicate::EQ, handle, i64.const_all_ones(), "failed"));
        Some(self.or_error(failed, handle))
    }

    /// `(handle)`: `GetFileSizeEx`.
    pub(super) fn file_size(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.handle(&args[0])?;
        let i32 = self.cx.i32_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let size = self.temp(Ty::Int(IntTy::I64));
        let f = self.runtime_function("GetFileSizeEx", i32.fn_type(&[ptr.into(), ptr.into()], false));
        let done = self.succeeded(f, &[handle.into(), size.ptr.into()]);
        let n = ok(self.b.build_load(self.cx.i64_type(), size.ptr, "size")).into_int_value();
        let failed = ok(self.b.build_not(done, "failed"));
        Some(self.or_error(failed, n))
    }

    /// `(handle, bytes)`: `ReadFile` or `WriteFile` over the whole of
    /// `bytes`, giving how many were moved.
    pub(super) fn file_transfer(&mut self, args: &[Expr], write: bool) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.handle(&args[0])?;
        let bytes = self.value(&args[1])?.into_struct_value();
        let (at, len) = self.slice_parts(bytes);
        let i32 = self.cx.i32_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let len = ok(self.b.build_int_truncate(len, i32, "len"));
        let moved = self.temp(Ty::Int(IntTy::U32));
        ok(self.b.build_store(moved.ptr, i32.const_zero()));
        let name = if write { "WriteFile" } else { "ReadFile" };
        let ty = i32.fn_type(&[ptr.into(), ptr.into(), i32.into(), ptr.into(), ptr.into()], false);
        let f = self.runtime_function(name, ty);
        let done = self.succeeded(f, &[handle.into(), at.into(), len.into(), moved.ptr.into(), ptr.const_null().into()]);
        let n = ok(self.b.build_load(i32, moved.ptr, "moved")).into_int_value();
        let n = ok(self.b.build_int_z_extend(n, self.cx.i64_type(), "moved"));
        let failed = ok(self.b.build_not(done, "failed"));
        Some(self.or_error(failed, n))
    }

    /// `(handle)`: `CloseHandle`.
    pub(super) fn file_close(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.handle(&args[0])?;
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let f = self.runtime_function("CloseHandle", self.cx.i32_type().fn_type(&[ptr.into()], false));
        ok(self.b.build_call(f, &[handle.into()], ""));
        None
    }

    /// A handle.
    pub(super) fn handle(&mut self, e: &Expr) -> Option<inkwell::values::PointerValue<'ctx>> {
        let n = self.value(e)?.into_int_value();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        Some(ok(self.b.build_int_to_ptr(n, ptr, "handle")))
    }

    /// Calls `f`, which returns a nonzero `BOOL` on success, giving whether
    /// it succeeded.
    pub(super) fn succeeded(
        &mut self,
        f: inkwell::values::FunctionValue<'ctx>,
        args: &[inkwell::values::BasicMetadataValueEnum<'ctx>],
    ) -> IntValue<'ctx> {
        let r = ok(self.b.build_call(f, args, "ok"))
            .try_as_basic_value()
            .basic()
            .expect("a BOOL")
            .into_int_value();
        ok(self.b.build_int_compare(IntPredicate::NE, r, self.cx.i32_type().const_zero(), "ok"))
    }

    /// `value`, or the system's last error code negated when `failed`.
    pub(super) fn or_error(&mut self, failed: IntValue<'ctx>, value: IntValue<'ctx>) -> BasicValueEnum<'ctx> {
        let i64 = self.cx.i64_type();
        let last = self.runtime_function("GetLastError", self.cx.i32_type().fn_type(&[], false));
        let code = ok(self.b.build_call(last, &[], "error"))
            .try_as_basic_value()
            .basic()
            .expect("an error code")
            .into_int_value();
        let code = ok(self.b.build_int_z_extend(code, i64, "error"));
        let negated = ok(self.b.build_int_neg(code, "negated"));
        ok(self.b.build_select(failed, negated, value, "result"))
    }
}
