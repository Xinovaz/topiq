//! The standard streams and the console.
//!
//! Standard input, output and error are handles like a file's, and the
//! `io` library reads and writes them with the calls `fs` uses. Two things
//! need calls of their own: finding the handles, and reading a console. A
//! console gives what is typed as UTF-16 code units, through `ReadConsoleW`;
//! `ReadFile` would give bytes in the console's code page, which are not
//! UTF-8.
//!
//! A console in its usual mode hands nothing over until Enter ends a line,
//! and shows each key as it is typed. A raw read turns line input and echo
//! off for the one call, so a key is read as soon as it is typed and is not
//! shown, and then restores the mode, so the console is left as the
//! program found it. Processed input stays on, so Ctrl+C still ends the
//! program.

use inkwell::values::{BasicValueEnum, IntValue, PointerValue};

use crate::tir::{Expr, IntTy, Ty};

use super::func::{Lowering, ok};

/// `STD_INPUT_HANDLE`, which is -10 as a `DWORD`; output and error are -11
/// and -12.
const STD_INPUT_HANDLE: u64 = 0xFFFF_FFF6;
/// `ENABLE_LINE_INPUT` and `ENABLE_ECHO_INPUT`: the parts of a console's mode
/// that hold keys back until Enter and show them.
const LINE_INPUT: u64 = 0x2;
const ECHO_INPUT: u64 = 0x4;

impl<'ctx> Lowering<'ctx, '_> {
    /// `(which)`: `GetStdHandle`, giving the handle as an integer.
    pub(super) fn std_handle(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let which = self.value(&args[0])?.into_int_value();
        let i32 = self.cx.i32_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let id = ok(self.b.build_int_sub(i32.const_int(STD_INPUT_HANDLE, false), which, "id"));
        let f = self.runtime_function("GetStdHandle", ptr.fn_type(&[i32.into()], false));
        let handle = ok(self.b.build_call(f, &[id.into()], "handle"))
            .try_as_basic_value()
            .basic()
            .expect("a handle")
            .into_pointer_value();
        Some(ok(self.b.build_ptr_to_int(handle, self.cx.i64_type(), "handle")).into())
    }

    /// `(handle)`: whether `GetConsoleMode` succeeds, which it does only on
    /// a console's handle.
    pub(super) fn is_console(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.handle(&args[0])?;
        let (_, console) = self.console_mode(handle);
        Some(console.into())
    }

    /// `(handle, units, raw)`: `ReadConsoleW` into the whole of `units`,
    /// with line input and echo off for the call when `raw`.
    pub(super) fn console_read(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.handle(&args[0])?;
        let units = self.value(&args[1])?.into_struct_value();
        let raw = self.value(&args[2])?.into_int_value();
        let (at, len) = self.slice_parts(units);
        let i32 = self.cx.i32_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());

        let (mode, _) = self.console_mode(handle);
        let keys = ok(self.b.build_and(mode, i32.const_int(!(LINE_INPUT | ECHO_INPUT) & 0xFFFF_FFFF, false), "keys"));
        let reading = ok(self.b.build_select(raw, keys, mode, "reading"));
        let set = self.runtime_function("SetConsoleMode", i32.fn_type(&[ptr.into(), i32.into()], false));
        ok(self.b.build_call(set, &[handle.into(), reading.into()], ""));

        let len = ok(self.b.build_int_truncate(len, i32, "len"));
        let got = self.temp(Ty::Int(IntTy::U32));
        ok(self.b.build_store(got.ptr, i32.const_zero()));
        let ty = i32.fn_type(&[ptr.into(), ptr.into(), i32.into(), ptr.into(), ptr.into()], false);
        let read = self.runtime_function("ReadConsoleW", ty);
        let args = [handle.into(), at.into(), len.into(), got.ptr.into(), ptr.const_null().into()];
        let done = self.succeeded(read, &args);
        // the error is read before restoring the mode can replace it
        let failed = ok(self.b.build_not(done, "failed"));
        let n = ok(self.b.build_load(i32, got.ptr, "got")).into_int_value();
        let n = ok(self.b.build_int_z_extend(n, self.cx.i64_type(), "got"));
        let result = self.or_error(failed, n);
        ok(self.b.build_call(set, &[handle.into(), mode.into()], ""));
        Some(result)
    }

    /// A handle's console mode, or 0, and whether it is a console's.
    fn console_mode(&mut self, handle: PointerValue<'ctx>) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let i32 = self.cx.i32_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let mode = self.temp(Ty::Int(IntTy::U32));
        ok(self.b.build_store(mode.ptr, i32.const_zero()));
        let get = self.runtime_function("GetConsoleMode", i32.fn_type(&[ptr.into(), ptr.into()], false));
        let console = self.succeeded(get, &[handle.into(), mode.ptr.into()]);
        let mode = ok(self.b.build_load(i32, mode.ptr, "mode")).into_int_value();
        (mode, console)
    }
}
