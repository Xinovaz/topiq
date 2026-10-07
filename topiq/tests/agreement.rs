//! Compiled arithmetic agrees with compile-time arithmetic.
//!
//! Integer semantics are defined once, in `topiq::sema::arith`, which the
//! constant evaluator uses directly. Code generation implements the same rules
//! separately, in machine code. This test holds the two together: it computes
//! the result of hundreds of operations with `arith`, generates a program that
//! performs the same operations at run time and compares each result against
//! the expected literal, and runs it at every optimisation level.
//!
//! The operands pass through an identity function per type, so that without
//! optimisation nothing is known about them when the operation runs. With
//! optimisation LLVM sees through the identity and folds, which checks its
//! folding of the generated IR against the same definition.
//!
//! Operations that `arith` says abort are gathered separately, and each runs
//! as its own program, which must abort with the identifier `arith` names.

#![cfg(feature = "llvm")]

use std::fmt::Write as _;
use std::process::Command;

use topiq::driver::OptLevel;
use topiq::driver::build::{BuildOptions, build};
use topiq::sema::arith::{self, Abort};
use topiq::tir::IntTy;

/// A literal for `v`, which the compiler reads as one negative value when
/// negative.
fn lit(v: i128) -> String {
    v.to_string()
}

/// Values worth testing for a type: both ends, around zero, and a middle.
fn values(t: IntTy) -> Vec<i128> {
    let mut v = vec![t.min(), 0, 1, 3, t.max() / 2, t.max()];
    if t.signed {
        v.extend([-1, t.min() + 1, -7]);
    } else {
        v.push(t.max() - 1);
    }
    v
}

/// One operation and what `arith` says it gives.
struct Case {
    /// The expression.
    expr: String,
    /// The result, or why it aborts.
    result: Result<String, Abort>,
    /// A short description for a failure message.
    what: String,
}

fn id(t: IntTy) -> String {
    format!("id_{}", t.name())
}

fn binary_cases(t: IntTy) -> Vec<Case> {
    type Op = fn(IntTy, i128, i128) -> Result<i128, Abort>;
    type Bitwise = fn(i128, i128) -> i128;
    let ops: [(&str, Op); 5] = [
        ("+", arith::add),
        ("-", arith::sub),
        ("*", arith::mul),
        ("/", arith::div),
        ("%", arith::rem),
    ];
    let mut out = Vec::new();
    for (sym, f) in ops {
        for &a in &values(t) {
            for &b in &values(t) {
                out.push(Case {
                    expr: format!("({}({}) {sym} {}({}))", id(t), lit(a), id(t), lit(b)),
                    result: f(t, a, b).map(lit),
                    what: format!("{a}{} {sym} {b}", t),
                });
            }
        }
    }
    let bitwise: [(&str, Bitwise); 3] =
        [("&", arith::bit_and), ("|", arith::bit_or), ("^", arith::bit_xor)];
    for (sym, f) in bitwise {
        for &a in &values(t) {
            for &b in &values(t) {
                out.push(Case {
                    expr: format!("({}({}) {sym} {}({}))", id(t), lit(a), id(t), lit(b)),
                    result: Ok(lit(f(a, b))),
                    what: format!("{a}{} {sym} {b}", t),
                });
            }
        }
    }
    // shift counts are `u32`, a different type from the value, as the
    // language allows
    for &a in &values(t) {
        for n in [0i128, 1, 3, i128::from(t.bits) - 1, i128::from(t.bits)] {
            for (sym, f) in [("<<", arith::shl as Op), (">>", arith::shr as Op)] {
                out.push(Case {
                    expr: format!("({}({}) {sym} id_u32({n}))", id(t), lit(a)),
                    result: f(t, a, n).map(lit),
                    what: format!("{a}{} {sym} {n}", t),
                });
            }
        }
    }
    for &a in &values(t) {
        out.push(Case {
            expr: format!("(-{}({}))", id(t), lit(a)),
            result: arith::neg(t, a).map(lit),
            what: format!("-({a}{})", t),
        });
        out.push(Case {
            expr: format!("(!{}({}))", id(t), lit(a)),
            result: Ok(lit(arith::not(t, a))),
            what: format!("!({a}{})", t),
        });
    }
    out
}

fn cast_cases(from: IntTy) -> Vec<Case> {
    let mut out = Vec::new();
    for to in IntTy::ALL {
        for &a in &values(from) {
            out.push(Case {
                expr: format!("({}({}) as {to})", id(from), lit(a)),
                result: arith::cast(to, a).map(lit),
                what: format!("{a}{from} as {to}"),
            });
        }
    }
    out
}

fn comparison_cases(t: IntTy) -> Vec<Case> {
    let mut out = Vec::new();
    for &a in &values(t) {
        for &b in &values(t) {
            for (sym, r) in [
                ("<", a < b),
                ("<=", a <= b),
                (">", a > b),
                (">=", a >= b),
                ("==", a == b),
                ("!=", a != b),
            ] {
                out.push(Case {
                    expr: format!("({}({}) {sym} {}({}))", id(t), lit(a), id(t), lit(b)),
                    result: Ok(r.to_string()),
                    what: format!("{a}{} {sym} {b}", t),
                });
            }
        }
    }
    out
}

fn all_cases() -> Vec<Case> {
    IntTy::ALL
        .into_iter()
        .flat_map(|t| {
            let mut v = binary_cases(t);
            v.extend(cast_cases(t));
            v.extend(comparison_cases(t));
            v
        })
        .collect()
}

/// The identity functions every case routes its operands through.
fn prelude() -> String {
    let mut s = String::from("#unit classical\n\n");
    for t in IntTy::ALL {
        let _ = writeln!(s, "fn {}(x: {t}) -> {t} {{ x }}", id(t));
    }
    s
}

/// Builds and runs `src`, returning its status and standard error.
fn build_and_run(name: &str, src: &str, opt: OptLevel) -> (Option<i32>, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{name}.tq"));
    std::fs::write(&path, src).unwrap();
    let program = build(
        &[path],
        &BuildOptions {
            opt_level: opt,
            out_dir: dir.path().to_owned(),
            link: true,
            ..BuildOptions::default()
        },
    )
    .unwrap();
    let errors: Vec<String> = program
        .all_diagnostics()
        .map(|d| format!("{}: {}", d.code, d.message))
        .collect();
    assert!(!program.has_errors(), "the generated program did not build:\n{}", errors.join("\n"));
    let out = Command::new(program.executable.unwrap()).output().unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn every_result_agrees_with_the_compile_time_definition() {
    let cases = all_cases();
    let ok: Vec<&Case> = cases.iter().filter(|c| c.result.is_ok()).collect();
    assert!(ok.len() > 1000, "the matrix covers the operators thoroughly: {}", ok.len());

    // one function per hundred cases keeps each function a reasonable size
    // each returns the 1-based index of its first mismatch, or 0
    let mut src = prelude();
    let chunks: Vec<&[&Case]> = ok.chunks(100).collect();
    for (k, chunk) in chunks.iter().enumerate() {
        let _ = writeln!(src, "\nfn part{k}() -> i32 {{");
        for (j, c) in chunk.iter().enumerate() {
            let want = c.result.as_ref().unwrap();
            let _ = writeln!(src, "    if {} != {want} {{ return {}; }}", c.expr, k * 100 + j + 1);
        }
        let _ = writeln!(src, "    0\n}}");
    }
    let _ = writeln!(src, "\nfn main() -> i32 {{");
    for k in 0..chunks.len() {
        let _ = writeln!(src, "    let r{k} = part{k}(); if r{k} != 0 {{ return r{k}; }}");
    }
    let _ = writeln!(src, "    0\n}}");

    for opt in OptLevel::ALL {
        let (status, stderr) = build_and_run("agreement", &src, opt);
        let status = status.expect("the program exits normally");
        assert!(stderr.is_empty(), "at {opt:?}: {stderr}");
        if status != 0 {
            let c = ok[(status - 1) as usize];
            panic!(
                "at {opt:?}, `{}` disagrees with the compile-time result {}",
                c.what,
                c.result.as_ref().unwrap()
            );
        }
    }
}

#[test]
fn every_kind_of_abort_happens_where_the_definition_says() {
    // one representative of each way an operation can abort, each on a
    // different type, each its own program
    let wanted: [(&str, IntTy, Abort); 12] = [
        ("+", IntTy::I8, Abort::Overflow),
        ("-", IntTy::U8, Abort::Overflow),
        ("*", IntTy::I32, Abort::Overflow),
        ("/", IntTy::U64, Abort::DivideByZero),
        ("/", IntTy::I16, Abort::Overflow),
        ("%", IntTy::ISIZE, Abort::DivideByZero),
        ("<<", IntTy::U16, Abort::ShiftCount),
        ("<<", IntTy::I64, Abort::Overflow),
        (">>", IntTy::USIZE, Abort::ShiftCount),
        ("neg", IntTy::I32, Abort::Overflow),
        ("neg", IntTy::U8, Abort::Overflow),
        ("as", IntTy::I64, Abort::Conversion),
    ];
    let cases = all_cases();
    for (sym, t, abort) in wanted {
        let c = cases
            .iter()
            .find(|c| {
                c.result == Err(abort)
                    && match sym {
                        "neg" => c.expr.starts_with(&format!("(-{}(", id(t))),
                        "as" => c.expr.starts_with(&format!("({}(", id(t))) && c.expr.contains(" as "),
                        _ => {
                            c.expr.starts_with(&format!("({}(", id(t)))
                                && c.expr.contains(&format!(" {sym} "))
                        }
                    }
            })
            .unwrap_or_else(|| panic!("the matrix has a {sym} on {t} that aborts with {abort:?}"));
        let mut src = prelude();
        // the operation is on line 14 of the program: after the directive, a
        // blank line and ten identity functions, then `fn main`
        let _ = writeln!(src, "fn main() -> i32 {{");
        let _ = writeln!(src, "    let r = {};", c.expr);
        let _ = writeln!(src, "    0\n}}");
        for opt in OptLevel::ALL {
            let (status, stderr) = build_and_run("aborts", &src, opt);
            let code = abort.code();
            assert_eq!(status, Some(101), "`{}` at {opt:?} should abort; stderr {stderr:?}", c.what);
            assert!(
                stderr.starts_with(&format!("topiq: abort {code} at aborts:14: ")),
                "`{}` at {opt:?}: {stderr:?}",
                c.what
            );
        }
    }
}
