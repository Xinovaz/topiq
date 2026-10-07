//! Writing text: the code behind `print`, `eprint` and `panic`.
//!
//! A Topiq string is a slice of `char`s (four-byte Unicode scalar values)
//! while standard output and standard error take bytes. `print` therefore
//! encodes each character as UTF-8 into a small buffer on the stack and
//! hands the buffer to the C runtime's `write` whenever it fills, and once at
//! the end. Nothing is allocated, and nothing is buffered beyond the one
//! call: text printed before an abort always appears before the abort's own
//! line. `print` adds no newline of its own.
//!
//! `panic(msg)` aborts with `RA10` like any failed check (one line on
//! standard error, status 101), except that the line ends with the program's
//! own message rather than a fixed one.
//!
//! Both helpers are private to each object, like the abort helper, so several
//! units' copies never collide. Each is defined the first time a unit uses
//! it, so a unit that prints nothing carries neither.

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::attributes::AttributeLoc;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{FunctionValue, IntValue, PointerValue};

use super::abort::{self, Runtime};
use super::func::ok;

/// How many bytes `print` gathers before writing them.
const BUFFER: u64 = 256;

/// The file descriptor of standard output.
pub const STDOUT: u64 = 1;
/// The file descriptor of standard error.
pub const STDERR: u64 = 2;

/// The helpers of one module.
pub struct Text<'ctx> {
    rt: Runtime,
    print: Option<FunctionValue<'ctx>>,
    panic: Option<FunctionValue<'ctx>>,
}

impl<'ctx> Text<'ctx> {
    /// No helpers yet, for a target whose C runtime is `rt`.
    pub fn new(rt: Runtime) -> Text<'ctx> {
        Text {
            rt,
            print: None,
            panic: None,
        }
    }

    /// `print(fd, chars, count)`.
    pub fn print(&mut self, cx: &'ctx Context, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        if let Some(f) = self.print {
            return f;
        }
        let (write, _) = abort::runtime_functions(cx, module, self.rt);
        let f = define_print(cx, module, write);
        self.print = Some(f);
        f
    }

    /// `panic(head, head_len, chars, count)`, which never returns.
    pub fn panic(&mut self, cx: &'ctx Context, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        if let Some(f) = self.panic {
            return f;
        }
        let print = self.print(cx, module);
        let (write, exit) = abort::runtime_functions(cx, module, self.rt);
        let f = define_panic(cx, module, write, exit, print);
        self.panic = Some(f);
        f
    }
}

fn define_print<'ctx>(cx: &'ctx Context, module: &Module<'ctx>, write: FunctionValue<'ctx>) -> FunctionValue<'ctx> {
    let i8 = cx.i8_type();
    let i32 = cx.i32_type();
    let i64 = cx.i64_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let f = module.add_function(
        "topiq$print",
        cx.void_type().fn_type(&[i32.into(), ptr.into(), i64.into()], false),
        Some(Linkage::Internal),
    );
    let fd = f.get_nth_param(0).expect("three parameters").into_int_value();
    let chars = f.get_nth_param(1).expect("three parameters").into_pointer_value();
    let count = f.get_nth_param(2).expect("three parameters").into_int_value();

    let b = cx.create_builder();
    let entry = cx.append_basic_block(f, "entry");
    let test = cx.append_basic_block(f, "test");
    let body = cx.append_basic_block(f, "char");
    let flush = cx.append_basic_block(f, "flush");
    let encode = cx.append_basic_block(f, "encode");
    let one = cx.append_basic_block(f, "one");
    let not_one = cx.append_basic_block(f, "not.one");
    let two = cx.append_basic_block(f, "two");
    let not_two = cx.append_basic_block(f, "not.two");
    let three = cx.append_basic_block(f, "three");
    let four = cx.append_basic_block(f, "four");
    let next = cx.append_basic_block(f, "next");
    let finish = cx.append_basic_block(f, "finish");
    let last = cx.append_basic_block(f, "last");
    let done = cx.append_basic_block(f, "done");

    b.position_at_end(entry);
    let buf = ok(b.build_alloca(i8.array_type(BUFFER as u32), "buf"));
    let i = ok(b.build_alloca(i64, "i"));
    let used = ok(b.build_alloca(i64, "used"));
    ok(b.build_store(i, i64.const_zero()));
    ok(b.build_store(used, i64.const_zero()));
    ok(b.build_unconditional_branch(test));

    // while i < count
    b.position_at_end(test);
    let iv = ok(b.build_load(i64, i, "")).into_int_value();
    let more = ok(b.build_int_compare(IntPredicate::ULT, iv, count, ""));
    ok(b.build_conditional_branch(more, body, finish));

    // make room for the longest encoding, four bytes, first
    b.position_at_end(body);
    let u = ok(b.build_load(i64, used, "")).into_int_value();
    let full = ok(b.build_int_compare(IntPredicate::UGT, u, i64.const_int(BUFFER - 4, false), ""));
    ok(b.build_conditional_branch(full, flush, encode));

    b.position_at_end(flush);
    emit(&b, write, fd, buf, u);
    ok(b.build_store(used, i64.const_zero()));
    ok(b.build_unconditional_branch(encode));

    b.position_at_end(encode);
    let iv = ok(b.build_load(i64, i, "")).into_int_value();
    // SAFETY: the index is below the count, so the address is within the
    // slice the caller passed
    let at = unsafe { ok(b.build_in_bounds_gep(i32, chars, &[iv], "")) };
    let c = ok(b.build_load(i32, at, "c")).into_int_value();
    let lt = |limit: u64, name: &str| ok(b.build_int_compare(IntPredicate::ULT, c, i32.const_int(limit, false), name));
    let small = lt(0x80, "ascii");
    ok(b.build_conditional_branch(small, one, not_one));

    b.position_at_end(not_one);
    let medium = lt(0x800, "two.bytes");
    ok(b.build_conditional_branch(medium, two, not_two));

    b.position_at_end(not_two);
    let large = lt(0x1_0000, "three.bytes");
    ok(b.build_conditional_branch(large, three, four));

    // each encoding: the lead byte, then six bits per continuation byte
    let cont = |b: &Builder<'ctx>, shift: u64| -> IntValue<'ctx> {
        let s = ok(b.build_right_shift(c, i32.const_int(shift, false), false, ""));
        let low = ok(b.build_and(s, i32.const_int(0x3F, false), ""));
        ok(b.build_or(low, i32.const_int(0x80, false), ""))
    };
    let lead = |b: &Builder<'ctx>, shift: u64, marker: u64| -> IntValue<'ctx> {
        let s = ok(b.build_right_shift(c, i32.const_int(shift, false), false, ""));
        ok(b.build_or(s, i32.const_int(marker, false), ""))
    };
    for (block, bytes, marker) in [(one, 1u64, 0), (two, 2, 0xC0), (three, 3, 0xE0), (four, 4, 0xF0)] {
        b.position_at_end(block);
        let out: Vec<IntValue<'ctx>> = if bytes == 1 {
            vec![c]
        } else {
            std::iter::once(lead(&b, 6 * (bytes - 1), marker)).chain((0..bytes - 1).rev().map(|k| cont(&b, 6 * k))).collect()
        };
        let u = ok(b.build_load(i64, used, "")).into_int_value();
        for (k, byte) in out.into_iter().enumerate() {
            let pos = ok(b.build_int_add(u, i64.const_int(k as u64, false), ""));
            // SAFETY: `used` is at most `BUFFER - 4` here, so four bytes fit
            let p = unsafe { ok(b.build_in_bounds_gep(i8, buf, &[pos], "")) };
            let t = ok(b.build_int_truncate(byte, i8, ""));
            ok(b.build_store(p, t));
        }
        let nu = ok(b.build_int_add(u, i64.const_int(bytes, false), ""));
        ok(b.build_store(used, nu));
        ok(b.build_unconditional_branch(next));
    }

    b.position_at_end(next);
    let iv = ok(b.build_load(i64, i, "")).into_int_value();
    let ni = ok(b.build_int_add(iv, i64.const_int(1, false), ""));
    ok(b.build_store(i, ni));
    ok(b.build_unconditional_branch(test));

    b.position_at_end(finish);
    let u = ok(b.build_load(i64, used, "")).into_int_value();
    let any = ok(b.build_int_compare(IntPredicate::NE, u, i64.const_zero(), ""));
    ok(b.build_conditional_branch(any, last, done));

    b.position_at_end(last);
    emit(&b, write, fd, buf, u);
    ok(b.build_unconditional_branch(done));

    b.position_at_end(done);
    ok(b.build_return(None));
    f
}

/// `write(fd, buf, n)`.
fn emit<'ctx>(b: &Builder<'ctx>, write: FunctionValue<'ctx>, fd: IntValue<'ctx>, buf: PointerValue<'ctx>, n: IntValue<'ctx>) {
    let i32 = n.get_type().get_context().i32_type();
    let n32 = ok(b.build_int_truncate(n, i32, ""));
    ok(b.build_call(write, &[fd.into(), buf.into(), n32.into()], ""));
}

fn define_panic<'ctx>(
    cx: &'ctx Context,
    module: &Module<'ctx>,
    write: FunctionValue<'ctx>,
    exit: FunctionValue<'ctx>,
    print: FunctionValue<'ctx>,
) -> FunctionValue<'ctx> {
    let i32 = cx.i32_type();
    let i64 = cx.i64_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let f = module.add_function(
        "topiq$panic",
        cx.void_type().fn_type(&[ptr.into(), i32.into(), ptr.into(), i64.into()], false),
        Some(Linkage::Internal),
    );
    for name in ["noreturn", "cold", "noinline", "nounwind"] {
        f.add_attribute(AttributeLoc::Function, abort::enum_attribute(cx, name));
    }
    let b = cx.create_builder();
    b.position_at_end(cx.append_basic_block(f, "entry"));
    let head = f.get_nth_param(0).expect("four parameters");
    let head_len = f.get_nth_param(1).expect("four parameters");
    let msg = f.get_nth_param(2).expect("four parameters");
    let msg_len = f.get_nth_param(3).expect("four parameters");
    let stderr = i32.const_int(STDERR, false);
    ok(b.build_call(write, &[stderr.into(), head.into(), head_len.into()], ""));
    ok(b.build_call(print, &[stderr.into(), msg.into(), msg_len.into()], ""));
    let newline = ok(b.build_global_string_ptr("\n", "panic.newline"));
    ok(b.build_call(
        write,
        &[stderr.into(), newline.as_pointer_value().into(), i32.const_int(1, false).into()],
        "",
    ));
    ok(b.build_call(exit, &[i32.const_int(abort::STATUS, false).into()], ""));
    ok(b.build_unreachable());
    f
}

/// The start of the line `panic` writes.
pub fn panic_head(unit: &str, line: u32) -> String {
    abort_head(crate::diag::Code::Ra10, unit, line)
}

/// The start of the line an abort with `code` writes, before its message.
pub fn abort_head(code: crate::diag::Code, unit: &str, line: u32) -> String {
    format!("topiq: abort {code} at {unit}:{line}: ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helpers_verify_and_are_defined_once() {
        let cx = Context::create();
        let m = cx.create_module("t");
        let mut t = Text::new(Runtime::for_triple("x86_64-pc-windows-msvc"));
        assert!(m.get_function("topiq$print").is_none(), "nothing until asked");
        let panic = t.panic(&cx, &m);
        assert_eq!(panic.get_linkage(), Linkage::Internal);
        assert_eq!(t.print(&cx, &m), t.print(&cx, &m));
        assert_eq!(t.print(&cx, &m).get_linkage(), Linkage::Internal);
        assert!(m.verify().is_ok(), "{}", m.verify().unwrap_err());
    }

    #[test]
    fn a_panic_line_names_ra10_the_unit_and_the_line() {
        assert_eq!(panic_head("app", 7), "topiq: abort RA10 at app:7: ");
    }
}
