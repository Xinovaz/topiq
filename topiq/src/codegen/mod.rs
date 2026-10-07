//! Translation phase 10: code generation.
//!
//! A fully analysed unit (every name resolved, every type settled, every
//! constant folded) is lowered to LLVM IR, optimised if asked, and turned
//! into a relocatable object for the host.
//!
//! | module | what it does |
//! |---|---|
//! | [`types`] | Topiq types and constants as LLVM types and constants |
//! | [`abi`] | how functions pass values to each other |
//! | [`mangle`] | the symbol each definition is given |
//! | [`func`] | functions, statements and control flow |
//! | [`place`] | addresses of places, loads, stores and copies |
//! | [`aggregate`] | structures, variants and arrays built in place |
//! | [`pattern`] | `match` |
//! | [`arith`] | integer operators with their run-time checks |
//! | [`abort`] | the one-line report and exit status of a failed check |
//! | [`text`] | `print`, `eprint` and `panic` |
//! | [`entry`] | the C `main` that starts the program |
//! | [`meta`] | the unit's metadata, placed in its object |
//! | [`closure`] | function values and closures |
//! | [`drop`](mod@drop) | destroying values when their scope ends |
//! | [`clone`] | `clone`'s deep copies |
//! | [`growable`] | growable arrays and their buffers |
//! | [`total`] | the library's arithmetic that never aborts |
//! | [`typeinfo`] | run-time type information |
//! | [`dynamic`] | `dyn`, and the operations the `dyn` library is built from |
//! | [`target`] | LLVM initialisation and the host target machine |
//!
//! Further modules: [`files`], [`input`] and [`modules`] hold the system
//! calls of the `fs`, `io` and `module` libraries, and [`image`] the
//! descriptor each program or module carries for `module::load` to check.
//!
//! # What a unit produces
//!
//! A classical unit becomes a `.tcu`: a relocatable native object carrying a
//! metadata section that describes what the unit offers (its functions'
//! signatures, its types, its constants' values), so another unit can be
//! compiled against it without its source. A quantum unit produces no code.
//!
//! When a program is linked, [`startup_object`] adds one more object: the
//! function running every unit's initialiser before `main`, the program's
//! descriptor, and, for a program built to run its tests, the entry point
//! that runs them.
//!
//! # Arithmetic
//!
//! Integer overflow, division by zero, bad shift counts and out-of-range
//! conversions abort at every optimisation level. Optimisation removes a
//! check only where LLVM proves it can never fail. Wrapping or saturating
//! arithmetic is asked for by name.
//!
//! # Representations
//!
//! These layouts are fixed by the language, so that foreign interfaces and
//! the dynamic facilities agree on them:
//!
//! | type | first word | second word |
//! |---|---|---|
//! | `*T`, `*const T` | address of the referent | `*const TypeInfo` of the referent's dynamic type |
//! | `*any` | address of the referent | `*const TypeInfo` of the referent's dynamic type |
//! | `[T]` | address of element 0 | element count |
//! | `*[T]`, `*[T; N]` | address of element 0 | element count |
//! | `closure<Sig>` | address of the environment | code address |
//! | `dyn` | address of the value | `*const TypeInfo` of the value |
//!
//! A `[T]` buffer carries a two-word header immediately before element 0,
//! aligned to `usize`: the capacity in elements, then a word the layout sets
//! aside for the element type's `*const TypeInfo`, written null; the element
//! type is found through the array type's own table.
//!
//! Tables of run-time type information are made only for types something
//! asks about ([`typeinfo`]); a reference whose referent's table no unit of
//! the program makes carries a blank one.

pub mod abi;
pub mod abort;
pub mod aggregate;
pub mod arith;
pub mod clone;
pub mod closure;
pub mod drop;
pub mod dynamic;
pub mod entry;
pub mod files;
pub mod func;
pub mod growable;
pub mod image;
pub mod input;
pub mod mangle;
pub mod modules;
pub mod meta;
pub mod pattern;
pub mod place;
pub mod target;
pub mod text;
pub mod total;
pub mod typeinfo;
pub mod types;

use std::fmt;

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::FileType;

use crate::driver::OptLevel;
use crate::intern::Interner;
use crate::source::SourceMap;
use crate::tir;

/// What to generate code for, and how.
#[derive(Clone, Copy)]
pub struct Request<'a> {
    /// The analysed unit. It must be complete: analysis reported no error.
    pub unit: &'a tir::Unit,
    /// The sources.
    pub sources: &'a SourceMap,
    /// For the names of definitions.
    pub interner: &'a Interner,
    /// How hard to optimise.
    pub opt: OptLevel,
    /// Whether to produce an object as well as the IR.
    pub object: bool,
    /// The unit's metadata as text.
    pub metadata: Option<&'a str>,
    /// Whether a valid `main` gets the C entry point that starts the
    /// program with it. A program built to run its tests starts elsewhere.
    pub entry: bool,
}

/// What code generation produced.
#[derive(Clone, Debug)]
pub struct Generated {
    /// The LLVM IR of the unit.
    pub ir: String,
    /// The relocatable object.
    pub object: Option<Vec<u8>>,
}

/// Why code generation failed.
///
/// Every one of these is a fault in the compiler or its environment, never in
/// the program: a program that analysed cleanly always has code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodegenError {
    /// No target machine could be built for the host.
    Target(target::TargetError),
    /// LLVM rejected the IR this compiler generated.
    Invalid(String),
    /// The optimisation pipeline failed.
    Passes(String),
    /// Writing the object failed.
    Emit(String),
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::Target(e) => write!(f, "{e}"),
            CodegenError::Invalid(e) => write!(f, "the generated LLVM IR is invalid: {e}"),
            CodegenError::Passes(e) => write!(f, "optimisation failed: {e}"),
            CodegenError::Emit(e) => write!(f, "writing the object failed: {e}"),
        }
    }
}

impl std::error::Error for CodegenError {}

/// Generates code for one unit.
///
/// # Errors
///
/// A [`CodegenError`], which always indicates a bug in the compiler or a
/// problem with its LLVM installation.
pub fn generate(r: &Request<'_>) -> Result<Generated, CodegenError> {
    let level = match r.opt {
        OptLevel::O0 => OptimizationLevel::None,
        OptLevel::O1 => OptimizationLevel::Less,
        // the size levels keep the standard instruction selection; only
        // their pass pipelines aim at size
        OptLevel::O2 | OptLevel::Os | OptLevel::Oz => OptimizationLevel::Default,
        OptLevel::O3 => OptimizationLevel::Aggressive,
    };
    let triple = target::host_triple();
    let machine = target::machine_for(&triple, level).map_err(CodegenError::Target)?;

    let cx = Context::create();
    let module = cx.create_module(&r.unit.name);
    module.set_triple(&triple);
    module.set_data_layout(&machine.get_target_data().get_data_layout());

    let triple_text = triple.as_str().to_string_lossy().into_owned();
    let rt = abort::Runtime::for_triple(&triple_text);
    let mut lowering = func::Lowering::new(&cx, &module, r.unit, r.sources, r.interner, rt);
    lowering.declare();
    lowering.emit_tables();
    lowering.define();
    if r.entry
        && let Some(id) = r.unit.main
        && entry::main_problem(r.unit.func(id), &r.unit.types).is_none()
    {
        let topiq_main = lowering.function(id).expect("a valid main is compiled");
        let arguments = r.unit.func(id).param_types().next().is_some();
        entry::define(&cx, &module, topiq_main, triple_text.contains("windows"), arguments);
    }
    // the program runs the initialiser through a symbol of its own, named
    // after the unit, when it starts
    if let Some(id) = r.unit.init {
        let init = lowering.function(id).expect("the initialiser is compiled");
        let void = cx.void_type().fn_type(&[], false);
        let f = module.add_function(&mangle::init(&r.unit.name), void, Some(inkwell::module::Linkage::External));
        let b = cx.create_builder();
        b.position_at_end(cx.append_basic_block(f, "entry"));
        func::ok(b.build_call(init, &[], ""));
        func::ok(b.build_return(None));
    }
    lowering.emit_weak_directives();
    drop(lowering);
    if let Some(text) = r.metadata {
        meta::embed(&cx, &module, text);
    }

    module
        .verify()
        .map_err(|e| CodegenError::Invalid(e.to_string()))?;
    if let Some(passes) = pipeline(r.opt) {
        module
            .run_passes(passes, &machine, PassBuilderOptions::create())
            .map_err(|e| CodegenError::Passes(e.to_string()))?;
    }
    let ir = module.print_to_string().to_string();
    let object = if r.object {
        let buffer = machine
            .write_to_memory_buffer(&module, FileType::Object)
            .map_err(|e| CodegenError::Emit(e.to_string()))?;
        Some(buffer.as_slice().to_vec())
    } else {
        None
    };
    Ok(Generated { ir, object })
}

/// The LLVM pass pipeline an optimisation level runs, if it runs one.
fn pipeline(opt: OptLevel) -> Option<&'static str> {
    match opt {
        OptLevel::O0 => None,
        OptLevel::O1 => Some("default<O1>"),
        OptLevel::O2 => Some("default<O2>"),
        OptLevel::O3 => Some("default<O3>"),
        OptLevel::Os => Some("default<Os>"),
        OptLevel::Oz => Some("default<Oz>"),
    }
}

/// The object that runs every unit's initialiser before `main`: `inits` are
/// the units that have one, in the order they run, which only the whole
/// program decides. It also holds the image's descriptor, from `image`,
/// which `module::load` checks a module against, and, for a program built
/// to run its tests, the entry point that runs them: `tests` names each
/// test and its symbol.
///
/// # Errors
///
/// A [`CodegenError`], which always indicates a bug in the compiler or a
/// problem with its LLVM installation.
pub fn startup_object(
    inits: &[String],
    tests: Option<&[(String, String)]>,
    image: &image::Image,
) -> Result<Vec<u8>, CodegenError> {
    let triple = target::host_triple();
    let machine = target::machine_for(&triple, OptimizationLevel::None).map_err(CodegenError::Target)?;
    let cx = Context::create();
    let module = cx.create_module("startup");
    module.set_triple(&triple);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    entry::startup(&cx, &module, inits);
    image::define(&cx, &module, image);
    if let Some(tests) = tests {
        let windows = triple.as_str().to_string_lossy().contains("windows");
        entry::tests(&cx, &module, windows, tests);
    }
    module
        .verify()
        .map_err(|e| CodegenError::Invalid(e.to_string()))?;
    let buffer = machine
        .write_to_memory_buffer(&module, FileType::Object)
        .map_err(|e| CodegenError::Emit(e.to_string()))?;
    Ok(buffer.as_slice().to_vec())
}

/// Helpers shared by the code generation tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::driver::{Options, Session, Stage, compile};

    /// Generates code for `src`, a whole unit, through the whole pipeline.
    pub fn generated(src: &str, opt: OptLevel) -> Generated {
        let mut session = Session::new();
        let id = session.add("demo.tq", src);
        let out = compile(
            &mut session,
            id,
            Options {
                stage: Stage::Check,
                ..Options::default()
            },
        );
        assert!(out.succeeded(Stage::Check), "{:?}", out.diagnostics);
        generate(&Request {
            unit: out.tir().unwrap(),
            sources: session.sources(),
            interner: session.interner(),
            opt,
            object: true,
            metadata: None,
            entry: true,
        })
        .expect("code generation succeeds")
    }

    /// The unoptimised IR of a classical unit whose items are `items`.
    pub fn ir(items: &str) -> String {
        generated(&format!("#unit classical\n{items}"), OptLevel::O0).ir
    }

    /// The IR of one function.
    pub fn function<'a>(ir: &'a str, name: &str) -> &'a str {
        let head = format!("@\"topiq$demo${name}\"(");
        let start = ir
            .lines()
            .position(|l| l.starts_with("define") && l.contains(&head))
            .unwrap_or_else(|| panic!("no function {name} in:\n{ir}"));
        let offset: usize = ir.lines().take(start).map(|l| l.len() + 1).sum();
        let rest = &ir[offset..];
        let end = rest.find("\n}\n").map_or(rest.len(), |e| e + 2);
        &rest[..end]
    }
}

#[cfg(test)]
mod tests {
    use super::testing::generated;
    use super::*;

    const FIB: &str = "#unit classical\n\
        fn fib(n: i32) -> i32 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }\n\
        fn main() -> i32 { fib(10) }\n";

    #[test]
    fn a_program_gets_its_functions_and_an_entry_point() {
        let g = generated(FIB, OptLevel::O0);
        assert!(g.ir.contains("define i32 @\"topiq$demo$fib\"(i32"), "{}", g.ir);
        assert!(g.ir.contains("define i32 @main(i32"), "{}", g.ir);
    }

    #[test]
    fn arithmetic_carries_its_checks() {
        let g = generated(FIB, OptLevel::O0);
        assert!(g.ir.contains("llvm.ssub.with.overflow.i32"), "{}", g.ir);
        assert!(g.ir.contains("llvm.sadd.with.overflow.i32"), "{}", g.ir);
        assert!(
            g.ir.contains("topiq: abort RA01 at demo:2: integer arithmetic overflow"),
            "the abort line is a constant: {}",
            g.ir
        );
    }

    #[test]
    fn an_object_is_produced() {
        let g = generated(FIB, OptLevel::O0);
        let obj = g.object.expect("asked for");
        assert!(obj.len() > 64, "a real object, {} bytes", obj.len());
    }

    #[test]
    fn optimization_keeps_checks_it_cannot_discharge() {
        // `a + b` of two unknown values can overflow, so its check survives
        // every pipeline
        for opt in OptLevel::ALL {
            let g = generated(
                "#unit classical\nfn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i32 { 0 }\n",
                opt,
            );
            assert!(g.ir.contains("with.overflow"), "at {opt:?}: {}", g.ir);
            assert!(g.ir.contains("topiq: abort RA01"), "at {opt:?}: {}", g.ir);
        }
    }

    #[test]
    fn every_optimizing_level_runs_its_own_pipeline() {
        assert_eq!(pipeline(OptLevel::O0), None);
        let all: Vec<_> = OptLevel::ALL.into_iter().filter_map(pipeline).collect();
        assert_eq!(all, ["default<O1>", "default<O2>", "default<O3>", "default<Os>", "default<Oz>"]);
    }

    #[test]
    fn a_static_function_is_private_to_its_object() {
        let g = generated(
            "#unit classical\nstatic fn helper() -> i32 { 1 }\nfn main() -> i32 { helper() }\n",
            OptLevel::O0,
        );
        assert!(g.ir.contains("define internal i32 @\"topiq$demo$helper\""), "{}", g.ir);
    }

    #[test]
    fn constants_and_constant_functions_leave_no_code() {
        let g = generated(
            "#unit classical\n\
             let LIMIT: const i32 = 64;\n\
             fn sq(n: constexpr i32) -> constexpr i32 { n * n }\n\
             fn main() -> i32 { sq(LIMIT) - 4096 }\n",
            OptLevel::O0,
        );
        assert!(!g.ir.contains("LIMIT"), "{}", g.ir);
        assert!(!g.ir.contains("$sq"), "{}", g.ir);
        assert!(g.ir.contains("4096"), "{}", g.ir);
    }

    #[test]
    fn a_unit_scope_object_is_initialized_in_the_image() {
        let g = generated(
            "#unit classical\nlet COUNT: u32 = 7;\nfn main() -> i32 { COUNT += 1; 0 }\n",
            OptLevel::O0,
        );
        assert!(g.ir.contains("@\"topiq$demo$COUNT\" = global i32 7"), "{}", g.ir);
    }

    #[test]
    fn every_control_construct_produces_valid_ir() {
        // verification runs inside `generate`; this only has to get there
        generated(
            "#unit classical\n\
             fn f(n: u8) -> u32 {\n\
                 persist let calls: u32 = 0;\n\
                 calls += 1;\n\
                 let t = 0u32;\n\
                 for i in 0..=n { if i % 2 == 0 { continue; } t += i as u32; }\n\
                 let k = 0u32;\n\
                 while k < 3 && t > 0 { k += 1; }\n\
                 let x = loop { if k == 3 { break 5u32; } k += 1; };\n\
                 if t > 100 { return t; }\n\
                 t + x + calls - 1\n\
             }\n\
             fn main() -> i32 { (f(10) - 30) as i32 }\n",
            OptLevel::O0,
        );
    }

    #[test]
    fn a_main_that_cannot_start_a_program_gets_no_entry_point() {
        let g = generated("#unit classical\nfn main() -> u8 { 0 }\n", OptLevel::O0);
        assert!(!g.ir.contains("@main("), "{}", g.ir);
    }
}
