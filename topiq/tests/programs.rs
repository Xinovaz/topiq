//! Every program in `tests/programs/` is built, linked and run, and must do
//! what its first line says.
//!
//! A program declares its expected behaviour itself, in its first line:
//!
//! ```text
//! // expect: exit 55
//! // expect: abort RA01 at 7
//! ```
//!
//! An `exit` program must finish with that status and write nothing to
//! standard error. An `abort` program must write exactly the one abort line
//! for that identifier and line to standard error, and exit with status 101.
//!
//! Lines after the first of the form `// stdout: text` say what the program
//! prints to standard output, one line of output each, newline included.
//! Without any, it must print nothing there. Lines of the form `// stdin:
//! text` are its standard input in the same way, where `\xHH` is the byte
//! of those two hex digits, so input need not be UTF-8; without any, its
//! standard input is empty.
//!
//! A directory is a program of several units: it is built from its `main.tq`,
//! which says what the program does, and the units that imports are found
//! beside it. An abort in another unit names it: `// expect: abort RA06 at
//! geometry:12`.
//!
//! Each program runs at every optimisation level. The optimised runs are the
//! ones that matter most: they show that no optimiser pipeline removes a check
//! that can fire, or changes what a program computes.

#![cfg(feature = "llvm")]

use std::path::{Path, PathBuf};
use std::io::Write;
use std::process::{Command, Stdio};

use topiq::driver::OptLevel;
use topiq::driver::build::{BuildOptions, build};

/// What a program says it does.
#[derive(Debug, PartialEq, Eq)]
enum Expect {
    /// Finishes with this status, printing nothing.
    Exit(i32),
    /// Aborts with this identifier at this line, of the named unit or else of
    /// the program's main unit.
    Abort {
        code: String,
        unit: Option<String>,
        line: u32,
    },
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("programs")
}

/// Reads the expectation from a program's first line.
fn expectation(path: &Path) -> Expect {
    let text = std::fs::read_to_string(path).unwrap();
    let first = text.lines().next().unwrap_or_default();
    let rest = first
        .strip_prefix("// expect: ")
        .unwrap_or_else(|| panic!("{} must start with `// expect: ...`", path.display()));
    let words: Vec<&str> = rest.split_whitespace().collect();
    match words.as_slice() {
        ["exit", n] => Expect::Exit(n.parse().expect("an exit status")),
        ["abort", code, "at", at] => {
            let (unit, line) = match at.split_once(':') {
                Some((u, l)) => (Some(u.to_owned()), l),
                None => (None, *at),
            };
            Expect::Abort {
                code: (*code).to_owned(),
                unit,
                line: line.parse().expect("a line number"),
            }
        }
        _ => panic!("{}: cannot read the expectation {rest:?}", path.display()),
    }
}

/// What a program says it prints to standard output: each `// stdout: `
/// line, with a newline.
fn expected_stdout(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .skip(1)
        .take_while(|l| l.starts_with("//"))
        .filter_map(|l| l.strip_prefix("// stdout: "))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// What a program says it is given on standard input: each `// stdin: `
/// line, with a newline, its `\xHH` escapes made bytes.
fn given_stdin(path: &Path) -> Vec<u8> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut bytes = Vec::new();
    for line in text.lines().skip(1).take_while(|l| l.starts_with("//")) {
        let Some(line) = line.strip_prefix("// stdin: ") else { continue };
        let mut rest = line;
        while let Some(at) = rest.find("\\x") {
            bytes.extend_from_slice(&rest.as_bytes()[..at]);
            let hex = rest.get(at + 2..at + 4).expect("two hex digits after `\\x`");
            bytes.push(u8::from_str_radix(hex, 16).expect("two hex digits after `\\x`"));
            rest = &rest[at + 4..];
        }
        bytes.extend_from_slice(rest.as_bytes());
        bytes.push(b'\n');
    }
    bytes
}

/// What running a program did.
struct Ran {
    status: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Builds and runs a program at one optimisation level, giving it `stdin`.
fn run(path: &Path, opt: OptLevel, stdin: &[u8]) -> Ran {
    let out = tempfile::tempdir().unwrap();
    let program = build(
        &[path.to_owned()],
        &BuildOptions {
            opt_level: opt,
            out_dir: out.path().to_owned(),
            link: true,
            ..BuildOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let reported: Vec<String> = program
        .all_diagnostics()
        .map(|d| format!("{}: {}", d.code, d.message))
        .collect();
    assert!(
        !program.has_errors(),
        "{} did not build:\n{}",
        path.display(),
        reported.join("\n")
    );
    let exe = program.executable.expect("linked");
    // each program runs where it was built, so a file it writes goes with it
    let mut child = Command::new(&exe)
        .current_dir(out.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the program starts");
    // the input is small enough for the pipe to hold it all, and dropping
    // the pipe ends it
    let mut pipe = child.stdin.take().expect("a piped standard input");
    pipe.write_all(stdin).expect("the input is written");
    drop(pipe);
    let output = child.wait_with_output().expect("the program finishes");
    Ran {
        status: output.status.code(),
        stdout: String::from_utf8(output.stdout).expect("output is UTF-8"),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Runs one program at every level and checks it against its expectation.
fn check(name: &str) {
    let mut path = dir().join(name);
    if path.is_dir() {
        path = path.join("main.tq");
    }
    let want = expectation(&path);
    let want_stdout = expected_stdout(&path);
    let stdin = given_stdin(&path);
    let main_unit = path.file_stem().unwrap().to_string_lossy().into_owned();
    for opt in OptLevel::ALL {
        let Ran { status, stdout, stderr } = run(&path, opt, &stdin);
        assert_eq!(stdout, want_stdout, "{name} at {opt:?}: standard output");
        match &want {
            Expect::Exit(n) => {
                assert_eq!(status, Some(*n), "{name} at {opt:?}: stderr was {stderr:?}");
                assert!(stderr.is_empty(), "{name} at {opt:?} printed {stderr:?}");
            }
            Expect::Abort { code, unit, line } => {
                let unit = unit.as_deref().unwrap_or(&main_unit);
                assert_eq!(status, Some(101), "{name} at {opt:?} should abort, stderr {stderr:?}");
                let prefix = format!("topiq: abort {code} at {unit}:{line}: ");
                assert!(
                    stderr.starts_with(&prefix) && stderr.ends_with('\n') && stderr.lines().count() == 1,
                    "{name} at {opt:?}: expected one line starting {prefix:?}, got {stderr:?}"
                );
            }
        }
    }
}

macro_rules! programs {
    ($($test:ident => $file:literal,)*) => {
        $(
            #[test]
            fn $test() {
                check($file);
            }
        )*

        /// Every program in the directory has a test, so none is silently
        /// skipped.
        #[test]
        fn every_program_in_the_directory_is_run() {
            let mut found: Vec<String> = std::fs::read_dir(dir())
                .unwrap()
                .filter_map(|e| {
                    let p = e.ok()?.path();
                    let program = p.is_dir() || p.extension()? == "tq";
                    program.then(|| p.file_name()?.to_str().map(str::to_owned))?
                })
                .collect();
            found.sort();
            let mut listed = vec![$($file.to_owned()),*];
            listed.sort();
            assert_eq!(found, listed, "tests/programs and this list have drifted apart");
        }
    };
}

programs! {
    recursion => "fib.tq",
    a_while_loop => "gcd.tq",
    an_inclusive_range => "squares.tq",
    shadowing_and_block_scope => "shadowing.tq",
    unit_scope_and_persistent_objects => "objects.tq",
    constants_and_constant_functions => "constants.tq",
    loop_break_continue_and_while => "loops.tq",
    short_circuiting_skips_an_abort => "short_circuit.tq",
    literals_at_the_extremes => "literals.tq",
    a_range_ending_at_the_maximum => "range_to_max.tq",
    min_remainder_minus_one_is_zero => "signed_remainder.tq",
    overflow_aborts => "overflow.tq",
    increments_before_and_after_a_place => "increments.tq",
    an_increment_past_the_maximum_aborts => "increment_overflow.tq",
    arrays_of_bools_combine_element_by_element => "bool_arrays.tq",
    rotation_and_swapping => "rotate_swap.tq",
    strings_compare_join_and_index_as_text => "string_ops.tq",
    operators_of_the_library_types => "library_operators.tq",
    classical_operators_on_quantum_data => "parity",
    a_unit_of_either_kind_imported_as_both => "either",
    division_by_zero_aborts => "divide_by_zero.tq",
    a_narrowing_conversion_aborts => "narrowing.tq",
    a_bad_shift_count_aborts => "shift_count.tq",
    a_shift_that_loses_bits_aborts => "shift_loses_bits.tq",
    unsigned_negation_aborts => "unsigned_negation.tq",
    min_divided_by_minus_one_aborts => "signed_division.tq",
    printing_text_as_utf8 => "hello.tq",
    structures_by_value_and_by_reference => "structs.tq",
    arrays_and_slices => "arrays.tq",
    enumerations_and_match => "enums.tq",
    strings_and_characters => "strings.tq",
    an_index_out_of_bounds_aborts => "bounds.tq",
    panic_aborts_with_its_message => "panic.tq",
    exit_ends_the_program => "exit.tq",
    a_constant_function_agrees_with_its_compiled_twin => "constant_twin.tq",
    layout_follows_the_rules => "layout.tq",
    aggregates_in_the_image => "image.tq",
    importing_types_functions_and_objects => "shapes",
    a_type_handed_on_through_a_middle_unit => "layered",
    following_and_taking_references => "deref.tq",
    floating_arithmetic_and_conversions => "floats.tq",
    tuples_aliases_and_destructuring => "tuples.tq",
    generic_types_and_functions => "generics.tq",
    methods_operators_and_open_types => "methods",
    a_classical_unit_uses_what_a_quantum_unit_offers => "hybrid",
    the_hybrid_deutsch_program_runs => "deutsch",
    circuits_run_on_the_exact_simulator => "running",
    circuits_run_on_a_device_a_program_registers => "backends",
    circuits_made_and_judged_while_running => "algebra",
    registers_grow_as_the_circuit_runs => "registers",
    quantum_variants_hold_classical_data => "variants",
    map_locales_of_states_and_held_as_values => "tables",
    a_supplied_operator_keeps_what_it_kicks_back_through => "kickback",
    the_library_makes_bell_ghz_and_w_states => "entangle",
    the_library_oracle_algorithms_and_kickback_judged => "oracles",
    the_library_grover_search_typed_over_its_orbit => "search",
    the_library_fourier_transform_estimates_and_adds => "fourier",
    the_library_teleports_codes_densely_and_swap_tests => "protocols",
    inner_products_agree_with_the_tests_that_read_them => "overlaps",
    the_library_simon_algorithm_finds_the_secret => "simon",
    a_refusal_read_while_running_has_a_witness => "witness",
    function_values_and_closures => "closures.tq",
    core_optionals_results_ranges_and_iteration => "library.tq",
    total_arithmetic_never_aborts => "total.tq",
    functions_and_closures_passed_to_optional_methods => "optional_methods.tq",
    values_are_destroyed_when_their_scope_ends => "raii.tq",
    growable_arrays_grow_shrink_and_go_through => "growable.tq",
    growable_arrays_destroy_each_element_once => "growable_raii.tq",
    a_growable_array_checks_its_indices => "growable_bounds.tq",
    copies_made_by_the_type_that_defines_them => "copy.tq",
    clones_share_nothing_with_the_original => "clone.tq",
    an_initializer_runs_before_main_and_main_takes_arguments => "initialiser.tq",
    introspection_answers_during_translation => "macros.tq",
    foreach_expands_a_macro_once_per_member => "foreach.tq",
    dynamic_values_answer_by_name_and_check_their_types => "dynamic.tq",
    prefixes_widen_and_downcasts_and_invoke_check => "hierarchy.tq",
    invoke_with_the_wrong_signature_aborts => "invoke_mismatch.tq",
    references_come_apart_into_words_and_back => "raw_parts.tq",
    constants_hold_growable_arrays_and_copy_them_out => "constant_buffers.tq",
    tcon_writes_any_value_and_reads_it_back => "tcon_roundtrip.tq",
    str_compares_splits_joins_parses_and_encodes => "text.tq",
    configuration_is_embedded_copied_at_startup_and_parsed => "configuration",
    exact_scalars_compute_exactly_at_both_times => "exact.tq",
    vectors_and_matrices_multiply_and_invert_exactly => "linear_algebra.tq",
    annotations_rename_hint_and_warn_without_changing_behavior => "annotations",
    files_are_written_and_read_back_as_text => "files.tq",
    structured_constants_match_as_patterns => "constant_patterns.tq",
    methods_added_to_another_units_generic_type => "foreign_generic_methods",
    items_declared_inside_blocks => "block_items.tq",
    functions_specialized_for_their_const_arguments => "specialised.tq",
    units_that_import_each_other => "import_cycle",
    a_type_serializes_itself_field_by_field => "serialisation.tq",
    expect_names_the_line_that_called_it => "unwrap_none.tq",
    abort_with_an_identifier_chosen_at_run_time => "abort_code.tq",
    quon_reads_writes_and_checks_states => "quon_states.tq",
    standard_input_is_one_buffered_stream => "standard_input.tq",
    files_are_streams_read_and_written_in_pieces => "file_streams.tq",
}

#[test]
fn expectations_are_read_from_the_first_line() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("x.tq");
    std::fs::write(&p, "// expect: abort RA02 at 9\n").unwrap();
    assert_eq!(
        expectation(&p),
        Expect::Abort {
            code: "RA02".to_owned(),
            unit: None,
            line: 9
        }
    );
    std::fs::write(&p, "// expect: abort RA06 at geometry:12\n").unwrap();
    assert_eq!(
        expectation(&p),
        Expect::Abort {
            code: "RA06".to_owned(),
            unit: Some("geometry".to_owned()),
            line: 12
        }
    );
    std::fs::write(&p, "// expect: exit -1\n").unwrap();
    assert_eq!(expectation(&p), Expect::Exit(-1));
    std::fs::write(&p, "// expect: exit 0\n// stdout: one\n// stdout: two\n#unit classical\n// stdout: not read\n").unwrap();
    assert_eq!(expected_stdout(&p), "one\ntwo\n");
    std::fs::write(&p, "// expect: exit 0\n// stdin: a\\x41\\xFF\n// stdin: \\x0a\n#unit classical\n").unwrap();
    assert_eq!(given_stdin(&p), b"aA\xFF\n\n\n");
}
