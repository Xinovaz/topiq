//! What happens when a check fails at run time.
//!
//! An abort writes exactly one line to standard error,
//!
//! ```text
//! topiq: abort RA01 at fib:12: integer arithmetic overflow
//! ```
//!
//! and ends the process with status 101. Nothing else runs: no destructor, no
//! finalisation of persistent objects, no unwinding.
//!
//! # Every message is a constant
//!
//! The identifier, the unit and the line of a failing check are all known
//! when the check is compiled, so each check site passes a complete, already
//! formatted line to one small helper. The helper does no formatting and
//! allocates nothing, which matters, because it runs when the program has
//! just found itself in a state it cannot continue from.
//!
//! The helper is private to each object, so a program built from several
//! units has several copies of a few instructions and no symbol for them to
//! collide over. It is marked `cold` and `noreturn`, which tells the optimiser
//! that the path to it is rarely taken and that nothing follows it.

use std::collections::HashMap;

use inkwell::AddressSpace;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{FunctionValue, GlobalValue};

use crate::diag::Code;

use super::mangle;

/// The process status of an aborted program.
pub const STATUS: u64 = 101;

/// The line an abort writes.
pub fn line(code: Code, unit: &str, line: u32) -> String {
    format!("topiq: abort {code} at {unit}:{line}: {}\n", code.message())
}

/// The C runtime's names for writing to a file descriptor and for ending the
/// process without running exit handlers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Runtime {
    /// `int write(int fd, const void *buf, unsigned count)`.
    pub write: &'static str,
    /// `void _exit(int status)`, which does not return.
    pub exit: &'static str,
}

impl Runtime {
    /// The names on the target described by `triple`.
    pub fn for_triple(triple: &str) -> Runtime {
        if triple.contains("windows") {
            Runtime {
                write: "_write",
                exit: "_exit",
            }
        } else {
            Runtime {
                write: "write",
                exit: "_exit",
            }
        }
    }
}

/// The abort helper of one module, and the messages its check sites pass it.
pub struct Aborts<'ctx> {
    helper: FunctionValue<'ctx>,
    messages: HashMap<String, GlobalValue<'ctx>>,
}

impl<'ctx> Aborts<'ctx> {
    /// Defines the helper in `module`.
    pub fn define(cx: &'ctx Context, module: &Module<'ctx>, rt: Runtime) -> Aborts<'ctx> {
        let i32_ = cx.i32_type();
        let ptr = cx.ptr_type(AddressSpace::default());
        let (write, exit) = runtime_functions(cx, module, rt);

        let helper = module.add_function(
            mangle::ABORT,
            cx.void_type().fn_type(&[ptr.into(), i32_.into()], false),
            Some(Linkage::Internal),
        );
        for name in ["noreturn", "cold", "noinline", "nounwind"] {
            helper.add_attribute(AttributeLoc::Function, enum_attribute(cx, name));
        }

        let b = cx.create_builder();
        let entry = cx.append_basic_block(helper, "entry");
        b.position_at_end(entry);
        let msg = helper.get_nth_param(0).expect("declared with two parameters");
        let len = helper.get_nth_param(1).expect("declared with two parameters");
        let stderr = i32_.const_int(2, false);
        b.build_call(write, &[stderr.into(), msg.into(), len.into()], "")
            .expect("positioned");
        let status = i32_.const_int(STATUS, false);
        b.build_call(exit, &[status.into()], "").expect("positioned");
        b.build_unreachable().expect("positioned");

        Aborts {
            helper,
            messages: HashMap::new(),
        }
    }

    /// The helper to call.
    pub fn helper(&self) -> FunctionValue<'ctx> {
        self.helper
    }

    /// The constant holding `text`, created on first use and shared by every
    /// check site that fails with the same line.
    pub fn message(&mut self, builder: &inkwell::builder::Builder<'ctx>, text: &str) -> GlobalValue<'ctx> {
        if let Some(g) = self.messages.get(text) {
            return *g;
        }
        let name = format!("abort.message.{}", self.messages.len());
        let g = builder
            .build_global_string_ptr(text, &name)
            .expect("positioned");
        self.messages.insert(text.to_owned(), g);
        g
    }
}

/// The C runtime's `write` and `_exit`, declared in `module` unless they are
/// already.
pub fn runtime_functions<'ctx>(
    cx: &'ctx Context,
    module: &Module<'ctx>,
    rt: Runtime,
) -> (FunctionValue<'ctx>, FunctionValue<'ctx>) {
    let i32_ = cx.i32_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let write = module.get_function(rt.write).unwrap_or_else(|| {
        module.add_function(
            rt.write,
            i32_.fn_type(&[i32_.into(), ptr.into(), i32_.into()], false),
            Some(Linkage::External),
        )
    });
    let exit = module.get_function(rt.exit).unwrap_or_else(|| {
        let f = module.add_function(
            rt.exit,
            cx.void_type().fn_type(&[i32_.into()], false),
            Some(Linkage::External),
        );
        f.add_attribute(AttributeLoc::Function, enum_attribute(cx, "noreturn"));
        f
    });
    (write, exit)
}

/// An attribute that takes no argument.
pub fn enum_attribute(cx: &Context, name: &str) -> Attribute {
    let kind = Attribute::get_named_enum_kind_id(name);
    assert_ne!(kind, 0, "LLVM knows the attribute `{name}`");
    cx.create_enum_attribute(kind, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_the_identifier_the_unit_and_the_line() {
        assert_eq!(
            line(Code::Ra01, "fib", 12),
            "topiq: abort RA01 at fib:12: integer arithmetic overflow\n"
        );
        assert_eq!(
            line(Code::Ra02, "calc", 3),
            "topiq: abort RA02 at calc:3: division or remainder by zero\n"
        );
    }

    #[test]
    fn windows_uses_the_underscored_crt_names() {
        let rt = Runtime::for_triple("x86_64-pc-windows-msvc");
        assert_eq!(rt.write, "_write");
        let rt = Runtime::for_triple("x86_64-unknown-linux-gnu");
        assert_eq!(rt.write, "write");
        assert_eq!(rt.exit, "_exit");
    }

    #[test]
    fn the_helper_is_private_and_never_returns() {
        let cx = Context::create();
        let m = cx.create_module("t");
        let a = Aborts::define(&cx, &m, Runtime::for_triple("x86_64-pc-windows-msvc"));
        assert_eq!(a.helper().get_linkage(), Linkage::Internal);
        let noreturn = Attribute::get_named_enum_kind_id("noreturn");
        assert!(
            a.helper()
                .get_enum_attribute(AttributeLoc::Function, noreturn)
                .is_some()
        );
        assert!(m.verify().is_ok(), "{}", m.verify().unwrap_err());
    }

    #[test]
    fn one_message_is_stored_once() {
        let cx = Context::create();
        let m = cx.create_module("t");
        let mut a = Aborts::define(&cx, &m, Runtime::for_triple("x86_64-pc-windows-msvc"));
        // `build_global_string_ptr` needs a positioned builder
        let f = m.add_function("host", cx.void_type().fn_type(&[], false), None);
        let b = cx.create_builder();
        b.position_at_end(cx.append_basic_block(f, "entry"));
        let x = a.message(&b, "one");
        let y = a.message(&b, "one");
        let z = a.message(&b, "two");
        assert_eq!(x, y);
        assert_ne!(x, z);
    }
}
