//! The quantum worked examples.
//!
//! Each program in `tests/acceptance/` is one of the language's worked
//! quantum examples as it is written, completed only where the example names
//! something it does not define (an oracle, a table's entries, a base type)
//! and with `@static_assert`s stating what its judgements are, which phase 9
//! checks once they are derived. Its first line says what analysing it
//! reports: `// expect: clean`, or the identifiers of the errors, in order,
//! as `// expect: EJ05`.
//!
//! A copy of a fixture keeps every line of the fixture's code, in order, so
//! that completing an example never changes it. The hybrid Deutsch program is
//! run in full as a program test (`tests/programs/deutsch`).

use std::path::{Path, PathBuf};

use topiq::diag::Code;
use topiq::driver::{Options, Stage};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("acceptance")
}

/// The identifiers of the errors analysing `path` reports.
fn analyzed(path: &Path) -> Vec<Code> {
    let loaded = topiq::driver::program::load(
        &[path.to_owned()],
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
    .unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()));
    let mut codes: Vec<Code> = loaded.diagnostics.iter().filter(|d| d.is_error()).map(|d| d.code).collect();
    for m in &loaded.members {
        if let topiq::driver::program::MemberKind::Source(out) = &m.kind {
            codes.extend(out.diagnostics.iter().filter(|d| d.is_error()).map(|d| d.code));
        }
    }
    codes
}

/// What a program's first line says analysing it reports.
fn expectation(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    let first = text.lines().next().unwrap_or_default();
    let rest = first
        .strip_prefix("// expect: ")
        .unwrap_or_else(|| panic!("{} must start with `// expect: …`", path.display()));
    if rest.trim() == "clean" {
        Vec::new()
    } else {
        rest.split(',').map(|s| s.trim().to_owned()).collect()
    }
}

fn accept(name: &str) {
    let path = dir().join(name);
    let got: Vec<String> = analyzed(&path).iter().map(|c| c.id().to_owned()).collect();
    assert_eq!(got, expectation(&path), "{name}");
}

/// The lines of a program that are code: comments and blank lines left out.
fn code_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or_default().trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Asserts that the copy keeps every line of the fixture's code, in order.
fn keeps(fixture: &str, copy: &str) {
    let original = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(fixture)).unwrap();
    let completed = std::fs::read_to_string(dir().join(copy)).unwrap();
    let mut rest = code_lines(&completed).into_iter();
    for line in code_lines(&original) {
        assert!(rest.any(|l| l == line), "{copy} does not keep `{line}` of {fixture}, in order");
    }
}

#[test]
fn h2_the_t_refusal_with_its_witnesses() {
    accept("h2_t_refusal.tq");
}

#[test]
fn h3_the_grover_iterate_is_cyclic_with_holonomy_pi() {
    keeps("grover.tq", "h3_grover.tq");
    accept("h3_grover.tq");
}

#[test]
fn h4_a_queried_table_is_a_sector_judgment() {
    keeps("table.tq", "h4_table.tq");
    accept("h4_table.tq");
}

#[test]
fn the_locale_and_the_contract_ascriptions() {
    keeps("locale.tq", "locale.tq");
    accept("locale.tq");
}

#[test]
fn every_program_in_the_directory_is_covered_by_a_test() {
    let mut found: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "tq").then(|| p.file_name()?.to_str().map(str::to_owned))?
        })
        .collect();
    found.sort();
    assert_eq!(found, ["h2_t_refusal.tq", "h3_grover.tq", "h4_table.tq", "locale.tq"]);
}
