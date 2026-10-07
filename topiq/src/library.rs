//! The standard library's units.
//!
//! A library unit is ordinary Topiq, compiled like any other unit, except
//! that nothing needs to be installed for it: its source is part of the
//! compiler, and a program that uses it has it compiled and linked in. It is
//! found before any directory of the unit path, so a program cannot replace
//! one by accident.
//!
//! `core` is special in one way: every unit uses it without importing it,
//! and sees its items unqualified. Every other library unit is imported, or
//! named with its unit's name in front, like any unit. `gates` is a quantum
//! unit, and a quantum unit importing it names its gates unqualified, as
//! `h(&q)`: they are the vocabulary circuits are written in. `algo`, a
//! quantum unit too, builds entanglement, phase kickback and the standard
//! algorithms from them. `qpu` runs
//! circuits on devices, and a classical unit importing a quantum unit
//! depends on it without naming it; `sim` holds the simulators `qpu` runs
//! circuits on when no other device is asked for; `circuit` makes circuits
//! from circuits while a program runs.
//!
//! What the library needs of the operating system it reaches through
//! functions the compiler provides, whose names begin with `__`. Names that
//! begin with `__` are reserved, so a program can neither declare nor call
//! them.

use std::path::PathBuf;

/// The library units this compiler carries.
const UNITS: [(&str, &str); 15] = [
    ("algo", include_str!("../lib/algo.tq")),
    ("circuit", include_str!("../lib/circuit.tq")),
    ("core", include_str!("../lib/core.tq")),
    ("dyn", include_str!("../lib/dyn.tq")),
    ("fs", include_str!("../lib/fs.tq")),
    ("gates", include_str!("../lib/gates.tq")),
    ("io", include_str!("../lib/io.tq")),
    ("judge", include_str!("../lib/judge.tq")),
    ("la", include_str!("../lib/la.tq")),
    ("module", include_str!("../lib/module.tq")),
    ("qpu", include_str!("../lib/qpu.tq")),
    ("quon", include_str!("../lib/quon.tq")),
    ("sim", include_str!("../lib/sim.tq")),
    ("str", include_str!("../lib/str.tq")),
    ("tcon", include_str!("../lib/tcon.tq")),
];

/// The names of every library unit the language defines, whether or not this
/// compiler provides it yet.
pub const NAMES: [&str; 15] = [
    "core", "str", "dyn", "tcon", "quon", "la", "gates", "algo", "judge", "qpu", "sim", "circuit", "module", "fs", "io",
];

/// Names set aside for library units of later editions, which no unit may
/// have.
pub const RESERVED: [&str; 6] = ["channel", "noise", "tomo", "mps", "time", "thread"];

/// The source of the library unit `name`.
pub fn source(name: &str) -> Option<&'static str> {
    UNITS.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

/// Whether `name` is a library unit the language defines.
pub fn is_library(name: &str) -> bool {
    NAMES.contains(&name)
}

/// The path a library unit's source is shown under in diagnostics.
pub fn path(name: &str) -> PathBuf {
    PathBuf::from(format!("<library>/{name}.tq"))
}

/// Whether a name is reserved for the library's own use: it begins with
/// `__`.
pub fn is_reserved_identifier(name: &str) -> bool {
    name.starts_with("__")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provided_unit_is_a_library_unit() {
        for (name, text) in UNITS {
            assert!(is_library(name));
            assert!(text.contains("#unit classical") || text.contains("#unit quantum"), "{name}");
        }
        assert!(source("core").is_some());
        assert!(source("geometry").is_none());
    }
}
