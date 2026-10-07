//! Loading modules.
//!
//! A module is a library the operating system loads into the running
//! program. Each operation here is one call of the system's; the `module`
//! library, written in Topiq, reads the module's descriptor and checks it
//! against the program's own, which [`super::image`] describes.

use inkwell::IntPredicate;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::values::BasicValueEnum;

use crate::tir::Expr;

use super::func::{Lowering, ok};

impl<'ctx> Lowering<'ctx, '_> {
    /// `(path)`: `LoadLibraryW`, giving the handle as an integer, or the
    /// error negated.
    pub(super) fn module_open(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let path = self.value(&args[0])?.into_struct_value();
        let (at, _) = self.slice_parts(path);
        let i64 = self.cx.i64_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let load = self.runtime_function("LoadLibraryW", ptr.fn_type(&[ptr.into()], false));
        let handle = ok(self.b.build_call(load, &[at.into()], "module"))
            .try_as_basic_value()
            .basic()
            .expect("a handle")
            .into_pointer_value();
        let handle = ok(self.b.build_ptr_to_int(handle, i64, "module"));
        let failed = ok(self.b.build_int_compare(IntPredicate::EQ, handle, i64.const_zero(), "failed"));
        Some(self.or_error(failed, handle))
    }

    /// `(handle, name)`: `GetProcAddress`, giving the address or zero.
    pub(super) fn module_find(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.value(&args[0])?.into_int_value();
        let name = self.value(&args[1])?.into_struct_value();
        let (at, _) = self.slice_parts(name);
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let handle = ok(self.b.build_int_to_ptr(handle, ptr, "module"));
        let find = self.runtime_function("GetProcAddress", ptr.fn_type(&[ptr.into(), ptr.into()], false));
        let addr = ok(self.b.build_call(find, &[handle.into(), at.into()], "addr"))
            .try_as_basic_value()
            .basic()
            .expect("an address")
            .into_pointer_value();
        Some(ok(self.b.build_ptr_to_int(addr, self.cx.i64_type(), "addr")).into())
    }

    /// `(handle)`: `FreeLibrary`, giving whether it succeeded.
    pub(super) fn module_close(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let handle = self.value(&args[0])?.into_int_value();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let handle = ok(self.b.build_int_to_ptr(handle, ptr, "module"));
        let i32 = self.cx.i32_type();
        let free = self.runtime_function("FreeLibrary", i32.fn_type(&[ptr.into()], false));
        let r = ok(self.b.build_call(free, &[handle.into()], "freed"))
            .try_as_basic_value()
            .basic()
            .expect("a BOOL")
            .into_int_value();
        Some(ok(self.b.build_int_compare(IntPredicate::NE, r, i32.const_zero(), "freed")).into())
    }

    /// `(address)`: calls a function taking and returning nothing.
    pub(super) fn call_void(&mut self, args: &[Expr]) -> Option<BasicValueEnum<'ctx>> {
        let addr = self.value(&args[0])?.into_int_value();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let f = ok(self.b.build_int_to_ptr(addr, ptr, "code"));
        let ty = self.cx.void_type().fn_type(&[], false);
        ok(self.b.build_indirect_call(ty, f, &[], ""));
        None
    }

    /// The address of the program's descriptor, which the object made when
    /// the program is linked defines.
    pub(super) fn host_descriptor(&mut self) -> Option<BasicValueEnum<'ctx>> {
        let g = self.module.get_global(super::image::HOST).unwrap_or_else(|| {
            let g = self.module.add_global(self.cx.i8_type(), None, super::image::HOST);
            g.set_linkage(LlvmLinkage::External);
            g
        });
        Some(ok(self.b.build_ptr_to_int(g.as_pointer_value(), self.cx.i64_type(), "host")).into())
    }
}
