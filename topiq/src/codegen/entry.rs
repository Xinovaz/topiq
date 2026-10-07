//! The C-level entry point that starts a Topiq program.
//!
//! The operating system starts a process in the C runtime, which calls a C
//! function named `main`. The unit that defines the Topiq `main` therefore
//! also gets a small C `main` that starts the program and returns its result,
//! which the C runtime then uses as the process's exit status. Before calling
//! the Topiq `main` it runs every unit's initialiser, through a function made
//! when the program is linked ([`startup`]), since only the linker knows every
//! unit and so the order they run in.
//!
//! Only a `main` that can actually start a program gets one: it must be
//! exported (not `static`), take nothing or the program's arguments, and
//! return an `i32`. Whether a program has exactly one such `main` is a
//! question about the whole program, answered when it is linked, so a `main`
//! that does not qualify is simply left without an entry point here and
//! reported there.
//!
//! A program built to run its tests has a different C `main`, made when it is
//! linked ([`tests`]): with no argument it lists the tests; with a number it
//! runs the initialisers, then that test.
//!
//! # The program's arguments
//!
//! A `main` declared `fn main(argv: *[*[char]]) -> i32` is given the
//! arguments the program was started with, the program's own name first, each
//! as characters. On Windows they are taken from the command line as the
//! system keeps it, in UTF-16, and split the way the system splits it; the
//! surrogate pairs are joined into characters. Elsewhere they are the C
//! runtime's, read as UTF-8. They are made once and never freed: they last as
//! long as the program.
//!
//! # Exactly the bytes printed
//!
//! On Windows the C runtime opens standard output and standard error in text
//! mode, which turns every `\n` written into `\r\n`. The entry point switches
//! both to binary mode before anything runs, so a program's output is exactly
//! the bytes it printed, on every platform.

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage as LlvmLinkage, Module};
use inkwell::values::{FunctionValue, IntValue, PointerValue};

use super::func::ok;
use super::{mangle, types};

pub use crate::meta::main_problem;

/// `_O_BINARY`, the C runtime's flag for untranslated input and output.
const BINARY: u64 = 0x8000;

/// Defines the C `main` calling `topiq_main`. `windows` says whether the
/// target is Windows, whose C runtime translates line endings (which the
/// entry point turns off) and whose command line is UTF-16. `arguments`
/// says whether `topiq_main` takes the program's arguments.
pub fn define<'ctx>(
    cx: &'ctx Context,
    module: &Module<'ctx>,
    topiq_main: FunctionValue<'ctx>,
    windows: bool,
    arguments: bool,
) {
    let i32_ = cx.i32_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let ty = i32_.fn_type(&[i32_.into(), ptr.into()], false);
    let entry = module.add_function(mangle::ENTRY, ty, Some(LlvmLinkage::External));
    let b = cx.create_builder();
    b.position_at_end(cx.append_basic_block(entry, "entry"));

    // on windows, stdout and stderr in binary mode
    if windows {
        let setmode = external(module, "_setmode", i32_.fn_type(&[i32_.into(), i32_.into()], false));
        for fd in [1, 2] {
            ok(b.build_call(
                setmode,
                &[i32_.const_int(fd, false).into(), i32_.const_int(BINARY, false).into()],
                "",
            ));
        }
    }

    // run the units' initialisers, then `topiq_main`, with the arguments if it takes them
    let startup = external(module, mangle::STARTUP, cx.void_type().fn_type(&[], false));
    ok(b.build_call(startup, &[], ""));
    let args = if arguments {
        let c_argc = entry.get_nth_param(0).expect("argc").into_int_value();
        let c_argv = entry.get_nth_param(1).expect("argv").into_pointer_value();
        let helper = arguments_helper(cx, module, windows);
        let v = ok(b.build_call(helper, &[c_argc.into(), c_argv.into()], "argv"))
            .try_as_basic_value()
            .basic()
            .expect("the arguments");
        vec![v.into()]
    } else {
        Vec::new()
    };
    let status = ok(b.build_call(topiq_main, &args, "status"))
        .try_as_basic_value()
        .basic()
        .expect("main returns an i32");
    ok(b.build_return(Some(&status)));
}

/// A function of this module that nothing here defines.
fn external<'ctx>(module: &Module<'ctx>, name: &str, ty: inkwell::types::FunctionType<'ctx>) -> FunctionValue<'ctx> {
    module
        .get_function(name)
        .unwrap_or_else(|| module.add_function(name, ty, Some(LlvmLinkage::External)))
}

/// Defines the function that runs every unit's initialiser, `inits` being
/// their units in the order they run, and the blank table of run-time type
/// information. They are the whole of a small object the linker adds to the
/// program.
pub fn startup<'ctx>(cx: &'ctx Context, module: &Module<'ctx>, inits: &[String]) {
    // the table a reference falls back on when no unit made its referent's:
    // all zeros, so a name with no characters and nothing else
    let blank_ty = cx.i8_type().array_type(256);
    let blank = module.add_global(blank_ty, None, super::typeinfo::BLANK);
    blank.set_initializer(&blank_ty.const_zero());
    blank.set_constant(true);
    blank.set_alignment(8);
    let void = cx.void_type().fn_type(&[], false);
    let f = module.add_function(mangle::STARTUP, void, Some(LlvmLinkage::External));
    let b = cx.create_builder();
    b.position_at_end(cx.append_basic_block(f, "entry"));
    for unit in inits {
        let init = external(module, &mangle::init(unit), void);
        ok(b.build_call(init, &[], ""));
    }
    ok(b.build_return(None));
}

/// Defines the C `main` of a program built to run its tests, in place of the
/// one that runs its `main`. `tests` are each test's name as it is shown and
/// its function's symbol.
///
/// Run with no argument, the program writes the names, one to a line, and
/// exits with 0. Run with a number, it runs the initialisers, then that test,
/// and exits with 0 if the test returns; a test that aborts ends the program
/// with the abort's status, so each test runs in a process of its own. A
/// number naming no test exits with 2.
pub fn tests<'ctx>(cx: &'ctx Context, module: &Module<'ctx>, windows: bool, tests: &[(String, String)]) {
    let i32_ = cx.i32_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let void = cx.void_type().fn_type(&[], false);
    let entry = module.add_function(mangle::ENTRY, i32_.fn_type(&[i32_.into(), ptr.into()], false), Some(LlvmLinkage::External));
    let b = cx.create_builder();
    b.position_at_end(cx.append_basic_block(entry, "entry"));
    if windows {
        let setmode = external(module, "_setmode", i32_.fn_type(&[i32_.into(), i32_.into()], false));
        ok(b.build_call(setmode, &[i32_.const_int(1, false).into(), i32_.const_int(BINARY, false).into()], ""));
    }
    let argc = entry.get_nth_param(0).expect("argc").into_int_value();
    let argv = entry.get_nth_param(1).expect("argv").into_pointer_value();
    let list = cx.append_basic_block(entry, "list");
    let run = cx.append_basic_block(entry, "run");
    let listing = ok(b.build_int_compare(inkwell::IntPredicate::SLT, argc, i32_.const_int(2, false), "listing"));
    ok(b.build_conditional_branch(listing, list, run));

    // no argument: the names
    b.position_at_end(list);
    let puts = external(module, "puts", i32_.fn_type(&[ptr.into()], false));
    for (i, (name, _)) in tests.iter().enumerate() {
        let text = ok(b.build_global_string_ptr(name, &format!("test.name.{i}")));
        ok(b.build_call(puts, &[text.as_pointer_value().into()], ""));
    }
    ok(b.build_return(Some(&i32_.const_zero())));

    // a number: that test, after the initialisers
    b.position_at_end(run);
    let second = unsafe { ok(b.build_gep(ptr, argv, &[cx.i64_type().const_int(1, false)], "arg")) };
    let text = ok(b.build_load(ptr, second, "text")).into_pointer_value();
    let atoi = external(module, "atoi", i32_.fn_type(&[ptr.into()], false));
    let n = ok(b.build_call(atoi, &[text.into()], "n"))
        .try_as_basic_value()
        .basic()
        .expect("atoi returns an int")
        .into_int_value();
    let startup = external(module, mangle::STARTUP, void);
    ok(b.build_call(startup, &[], ""));
    let unknown = cx.append_basic_block(entry, "unknown");
    let cases: Vec<_> = tests
        .iter()
        .enumerate()
        .map(|(i, (_, symbol))| {
            let block = cx.append_basic_block(entry, &format!("test.{i}"));
            let bb = cx.create_builder();
            bb.position_at_end(block);
            ok(bb.build_call(external(module, symbol, void), &[], ""));
            ok(bb.build_return(Some(&i32_.const_zero())));
            (i32_.const_int(i as u64, false), block)
        })
        .collect();
    ok(b.build_switch(n, unknown, &cases));
    b.position_at_end(unknown);
    ok(b.build_return(Some(&i32_.const_int(2, false))));
}

/// The function turning the C runtime's `argc` and `argv` into the program's
/// arguments as `main` takes them: a slice of slices of characters.
fn arguments_helper<'ctx>(cx: &'ctx Context, module: &Module<'ctx>, windows: bool) -> FunctionValue<'ctx> {
    let i32_ = cx.i32_type();
    let i64_ = cx.i64_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let slice = types::fat_slice(cx);
    let f = module.add_function(
        "topiq$$arguments",
        slice.fn_type(&[i32_.into(), ptr.into()], false),
        Some(LlvmLinkage::Private),
    );
    let b = cx.create_builder();
    b.position_at_end(cx.append_basic_block(f, "entry"));
    let malloc = external(module, "malloc", ptr.fn_type(&[i64_.into()], false));

    // the count and the strings, as the system gives them
    let (count, strings) = if windows {
        let command_line = external(module, "GetCommandLineW", ptr.fn_type(&[], false));
        let split = external(module, "CommandLineToArgvW", ptr.fn_type(&[ptr.into(), ptr.into()], false));
        let line = call_ptr(&b, command_line, &[]);
        let n = ok(b.build_alloca(i32_, "argc"));
        let list = call_ptr(&b, split, &[line.into(), n.into()]);
        let n = ok(b.build_load(i32_, n, "")).into_int_value();
        (n, list)
    } else {
        let n = f.get_nth_param(0).expect("argc").into_int_value();
        (n, f.get_nth_param(1).expect("argv").into_pointer_value())
    };
    let count = ok(b.build_int_z_extend(count, i64_, "count"));
    let outer_bytes = ok(b.build_int_mul(count, i64_.const_int(16, false), ""));
    let outer = call_ptr(&b, malloc, &[outer_bytes.into()]);

    // for each argument: its length in code units, a buffer of as many
    // characters, and the characters decoded into it
    let i = ok(b.build_alloca(i64_, "i"));
    ok(b.build_store(i, i64_.const_zero()));
    let head = cx.append_basic_block(f, "arg");
    let body = cx.append_basic_block(f, "arg.body");
    let done = cx.append_basic_block(f, "args.done");
    ok(b.build_unconditional_branch(head));
    b.position_at_end(head);
    let k = ok(b.build_load(i64_, i, "")).into_int_value();
    let more = ok(b.build_int_compare(IntPredicate::ULT, k, count, ""));
    ok(b.build_conditional_branch(more, body, done));

    b.position_at_end(body);
    // SAFETY: `k` is below the count of strings in the list
    let at = unsafe { ok(b.build_gep(ptr, strings, &[k], "")) };
    let text = ok(b.build_load(ptr, at, "text")).into_pointer_value();
    let unit = if windows { cx.i16_type() } else { cx.i8_type() };
    let len = code_units(cx, &b, f, text, unit);
    let size = ok(b.build_int_mul(len, i64_.const_int(4, false), ""));
    let size = ok(b.build_int_add(size, i64_.const_int(4, false), ""));
    let chars = call_ptr(&b, malloc, &[size.into()]);
    let n = if windows {
        decode_utf16(cx, &b, f, text, len, chars)
    } else {
        decode_utf8(cx, &b, f, text, len, chars)
    };
    let arg = slice.get_undef();
    let arg = ok(b.build_insert_value(arg, chars, 0, "")).into_struct_value();
    let arg = ok(b.build_insert_value(arg, n, 1, "")).into_struct_value();
    // SAFETY: `k` is below the count the outer buffer was made for
    let slot = unsafe { ok(b.build_gep(slice, outer, &[k], "")) };
    ok(b.build_store(slot, arg));
    let next = ok(b.build_int_add(k, i64_.const_int(1, false), ""));
    ok(b.build_store(i, next));
    ok(b.build_unconditional_branch(head));

    b.position_at_end(done);
    let all = slice.get_undef();
    let all = ok(b.build_insert_value(all, outer, 0, "")).into_struct_value();
    let all = ok(b.build_insert_value(all, count, 1, "")).into_struct_value();
    ok(b.build_return(Some(&all)));
    f
}

fn call_ptr<'ctx>(
    b: &Builder<'ctx>,
    f: FunctionValue<'ctx>,
    args: &[inkwell::values::BasicMetadataValueEnum<'ctx>],
) -> PointerValue<'ctx> {
    ok(b.build_call(f, args, ""))
        .try_as_basic_value()
        .basic()
        .expect("an address")
        .into_pointer_value()
}

/// The number of code units of type `unit` before the terminating zero.
fn code_units<'ctx>(
    cx: &'ctx Context,
    b: &Builder<'ctx>,
    f: FunctionValue<'ctx>,
    text: PointerValue<'ctx>,
    unit: inkwell::types::IntType<'ctx>,
) -> IntValue<'ctx> {
    let i64_ = cx.i64_type();
    let n = ok(b.build_alloca(i64_, "len"));
    ok(b.build_store(n, i64_.const_zero()));
    let head = cx.append_basic_block(f, "len");
    let step = cx.append_basic_block(f, "len.step");
    let done = cx.append_basic_block(f, "len.done");
    ok(b.build_unconditional_branch(head));
    b.position_at_end(head);
    let k = ok(b.build_load(i64_, n, "")).into_int_value();
    // SAFETY: the string is terminated by a zero, which stops the walk
    let at = unsafe { ok(b.build_gep(unit, text, &[k], "")) };
    let c = ok(b.build_load(unit, at, "")).into_int_value();
    let end = ok(b.build_int_compare(IntPredicate::EQ, c, unit.const_zero(), ""));
    ok(b.build_conditional_branch(end, done, step));
    b.position_at_end(step);
    let next = ok(b.build_int_add(k, i64_.const_int(1, false), ""));
    ok(b.build_store(n, next));
    ok(b.build_unconditional_branch(head));
    b.position_at_end(done);
    ok(b.build_load(i64_, n, "")).into_int_value()
}

/// Decodes `len` UTF-16 code units at `text` into characters at `out`,
/// joining surrogate pairs; returns how many characters there are. A lone
/// surrogate is kept as the code unit it is.
fn decode_utf16<'ctx>(
    cx: &'ctx Context,
    b: &Builder<'ctx>,
    f: FunctionValue<'ctx>,
    text: PointerValue<'ctx>,
    len: IntValue<'ctx>,
    out: PointerValue<'ctx>,
) -> IntValue<'ctx> {
    let i64_ = cx.i64_type();
    let i32_ = cx.i32_type();
    let i16_ = cx.i16_type();
    let (i, n) = (ok(b.build_alloca(i64_, "i")), ok(b.build_alloca(i64_, "n")));
    ok(b.build_store(i, i64_.const_zero()));
    ok(b.build_store(n, i64_.const_zero()));
    let head = cx.append_basic_block(f, "utf16");
    let body = cx.append_basic_block(f, "utf16.unit");
    let pair = cx.append_basic_block(f, "utf16.pair");
    let single = cx.append_basic_block(f, "utf16.single");
    let done = cx.append_basic_block(f, "utf16.done");
    ok(b.build_unconditional_branch(head));

    b.position_at_end(head);
    let k = ok(b.build_load(i64_, i, "")).into_int_value();
    let more = ok(b.build_int_compare(IntPredicate::ULT, k, len, ""));
    ok(b.build_conditional_branch(more, body, done));

    b.position_at_end(body);
    // SAFETY: `k` is below the length
    let at = unsafe { ok(b.build_gep(i16_, text, &[k], "")) };
    let u = ok(b.build_load(i16_, at, "")).into_int_value();
    let u = ok(b.build_int_z_extend(u, i32_, ""));
    let high = ok(b.build_int_compare(IntPredicate::UGE, u, i32_.const_int(0xD800, false), ""));
    let high2 = ok(b.build_int_compare(IntPredicate::ULT, u, i32_.const_int(0xDC00, false), ""));
    let k1 = ok(b.build_int_add(k, i64_.const_int(1, false), ""));
    let room = ok(b.build_int_compare(IntPredicate::ULT, k1, len, ""));
    let is_pair = ok(b.build_and(high, high2, ""));
    let is_pair = ok(b.build_and(is_pair, room, ""));
    ok(b.build_conditional_branch(is_pair, pair, single));

    // a high surrogate followed by anything: joined with the next unit when
    // that is a low surrogate
    b.position_at_end(pair);
    // SAFETY: `k + 1` is below the length
    let at2 = unsafe { ok(b.build_gep(i16_, text, &[k1], "")) };
    let v = ok(b.build_load(i16_, at2, "")).into_int_value();
    let v = ok(b.build_int_z_extend(v, i32_, ""));
    let low = ok(b.build_int_compare(IntPredicate::UGE, v, i32_.const_int(0xDC00, false), ""));
    let low2 = ok(b.build_int_compare(IntPredicate::ULT, v, i32_.const_int(0xE000, false), ""));
    let is_low = ok(b.build_and(low, low2, ""));
    let hi = ok(b.build_int_sub(u, i32_.const_int(0xD800, false), ""));
    let hi = ok(b.build_left_shift(hi, i32_.const_int(10, false), ""));
    let lo = ok(b.build_int_sub(v, i32_.const_int(0xDC00, false), ""));
    let c = ok(b.build_int_add(hi, lo, ""));
    let c = ok(b.build_int_add(c, i32_.const_int(0x10000, false), ""));
    let c = ok(b.build_select(is_low, c, u, "")).into_int_value();
    let step = ok(b.build_select(is_low, i64_.const_int(2, false), i64_.const_int(1, false), "")).into_int_value();
    put(cx, b, out, n, c);
    let next = ok(b.build_int_add(k, step, ""));
    ok(b.build_store(i, next));
    ok(b.build_unconditional_branch(head));

    b.position_at_end(single);
    put(cx, b, out, n, u);
    ok(b.build_store(i, k1));
    ok(b.build_unconditional_branch(head));

    b.position_at_end(done);
    ok(b.build_load(i64_, n, "")).into_int_value()
}

/// Decodes `len` bytes of UTF-8 at `text` into characters at `out`; returns
/// how many characters there are. The bytes are the C runtime's and are
/// taken to be well formed; a sequence cut short by the end is dropped.
fn decode_utf8<'ctx>(
    cx: &'ctx Context,
    b: &Builder<'ctx>,
    f: FunctionValue<'ctx>,
    text: PointerValue<'ctx>,
    len: IntValue<'ctx>,
    out: PointerValue<'ctx>,
) -> IntValue<'ctx> {
    let i64_ = cx.i64_type();
    let i32_ = cx.i32_type();
    let i8_ = cx.i8_type();
    let (i, n) = (ok(b.build_alloca(i64_, "i")), ok(b.build_alloca(i64_, "n")));
    ok(b.build_store(i, i64_.const_zero()));
    ok(b.build_store(n, i64_.const_zero()));
    let head = cx.append_basic_block(f, "utf8");
    let body = cx.append_basic_block(f, "utf8.lead");
    let done = cx.append_basic_block(f, "utf8.done");
    ok(b.build_unconditional_branch(head));

    b.position_at_end(head);
    let k = ok(b.build_load(i64_, i, "")).into_int_value();
    let more = ok(b.build_int_compare(IntPredicate::ULT, k, len, ""));
    ok(b.build_conditional_branch(more, body, done));

    b.position_at_end(body);
    let byte = |b: &Builder<'ctx>, at: IntValue<'ctx>| {
        // SAFETY: callers keep `at` below the length, or read the zero
        // that ends the string
        let p = unsafe { ok(b.build_gep(i8_, text, &[at], "")) };
        let v = ok(b.build_load(i8_, p, "")).into_int_value();
        ok(b.build_int_z_extend(v, i32_, ""))
    };
    let lead = byte(b, k);
    // how many bytes the sequence has, from its lead byte
    let ge = |x: u64| ok(b.build_int_compare(IntPredicate::UGE, lead, i32_.const_int(x, false), ""));
    let (two, three, four) = (ge(0xC0), ge(0xE0), ge(0xF0));
    let one = i64_.const_int(1, false);
    let size = ok(b.build_select(two, i64_.const_int(2, false), one, "")).into_int_value();
    let size = ok(b.build_select(three, i64_.const_int(3, false), size, "")).into_int_value();
    let size = ok(b.build_select(four, i64_.const_int(4, false), size, "")).into_int_value();
    // the lead byte's own bits, then six from each continuation byte. bytes
    // past the sequence are read only up to the terminating zero and
    // discarded by the selects
    let mask = ok(b.build_select(two, i32_.const_int(0x1F, false), i32_.const_int(0x7F, false), "")).into_int_value();
    let mask = ok(b.build_select(three, i32_.const_int(0x0F, false), mask, "")).into_int_value();
    let mask = ok(b.build_select(four, i32_.const_int(0x07, false), mask, "")).into_int_value();
    let mut c = ok(b.build_and(lead, mask, ""));
    for extra in 1..4u64 {
        let has = ok(b.build_int_compare(IntPredicate::UGT, size, i64_.const_int(extra, false), ""));
        let at = ok(b.build_int_add(k, i64_.const_int(extra, false), ""));
        let in_text = ok(b.build_int_compare(IntPredicate::ULT, at, len, ""));
        let safe = ok(b.build_select(in_text, at, len, "")).into_int_value();
        let cont = byte(b, safe);
        let bits = ok(b.build_and(cont, i32_.const_int(0x3F, false), ""));
        let shifted = ok(b.build_left_shift(c, i32_.const_int(6, false), ""));
        let joined = ok(b.build_or(shifted, bits, ""));
        c = ok(b.build_select(has, joined, c, "")).into_int_value();
    }
    put(cx, b, out, n, c);
    let next = ok(b.build_int_add(k, size, ""));
    ok(b.build_store(i, next));
    ok(b.build_unconditional_branch(head));

    b.position_at_end(done);
    ok(b.build_load(i64_, n, "")).into_int_value()
}

/// Appends the character `c` at `out`, counting it in `n`.
fn put<'ctx>(cx: &'ctx Context, b: &Builder<'ctx>, out: PointerValue<'ctx>, n: PointerValue<'ctx>, c: IntValue<'ctx>) {
    let i64_ = cx.i64_type();
    let k = ok(b.build_load(i64_, n, "")).into_int_value();
    // SAFETY: the buffer has room for one character per code unit
    let at = unsafe { ok(b.build_gep(cx.i32_type(), out, &[k], "")) };
    ok(b.build_store(at, c));
    let next = ok(b.build_int_add(k, i64_.const_int(1, false), ""));
    ok(b.build_store(n, next));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_main<'ctx>(cx: &'ctx Context, m: &Module<'ctx>, args: bool) -> bool {
        let slice = types::fat_slice(cx);
        let ty = if args {
            cx.i32_type().fn_type(&[slice.into()], false)
        } else {
            cx.i32_type().fn_type(&[], false)
        };
        let f = m.add_function("topiq$t$main", ty, None);
        let b = cx.create_builder();
        b.position_at_end(cx.append_basic_block(f, "entry"));
        b.build_return(Some(&cx.i32_type().const_int(7, false))).unwrap();
        define(cx, m, f, true, args);
        m.verify().is_ok()
    }

    #[test]
    fn the_entry_point_starts_the_units_then_calls_topiq_main() {
        let cx = Context::create();
        let m = cx.create_module("t");
        assert!(a_main(&cx, &m, false), "{}", m.verify().unwrap_err());
        assert!(m.get_function("main").is_some());
        assert!(m.get_function("_setmode").is_some(), "output is switched to binary");
        assert!(m.get_function(mangle::STARTUP).is_some(), "the initialisers run first");
    }

    #[test]
    fn a_main_taking_arguments_is_given_them() {
        let cx = Context::create();
        let m = cx.create_module("t");
        assert!(a_main(&cx, &m, true), "{}", m.verify().unwrap_err());
        assert!(m.get_function("CommandLineToArgvW").is_some());
    }

    #[test]
    fn startup_runs_each_initializer_in_the_order_given() {
        let cx = Context::create();
        let m = cx.create_module("startup");
        startup(&cx, &m, &["geo".to_owned(), "app".to_owned()]);
        assert!(m.verify().is_ok());
        let ir = m.print_to_string().to_string();
        let (g, a) = (ir.find("topiq$geo$$init").unwrap(), ir.rfind("topiq$app$$init").unwrap());
        assert!(g < a, "{ir}");
    }
}
