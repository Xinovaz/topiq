//! Linking: combining the objects of a program's units into an executable or
//! a module.
//!
//! Compilation works one unit at a time; linking is where the program exists
//! as a whole, and where the questions about the whole program are answered
//! from each unit's metadata:
//!
//! - were all of them compiled for this version of the language (`EL11`)?
//! - is every unit one imports present, and does it still offer every
//!   function, object and type the importer was compiled against, with the
//!   same signature (`EL02`), layout (`EL03`) and constant values (`EL12`),
//!   or at all (`EL06`)?
//! - do two units add the same method to one `[open]` type (`EL07`), or
//!   export the same symbol (`EL06`)?
//! - does exactly one unit define a `main` that can start the program
//!   (`EL05`)? A module, or a program built to run its tests, needs none.
//! - is there an order to run the units' initialisers in, each after those
//!   of the units it imports (`EL04`)?
//! - where a quantum operator's judgement crosses from one unit to another:
//!   is its record the one the importer read (`EL03`); do the two units'
//!   conductors meet within the limit (`EL10`); is a claim that depends on a
//!   phase convention published with the cover and gauge it is stated
//!   against, as declarations the importer can name (`EJ08`); and does each
//!   published claim still hold of the record it came with (`EJ03`)?
//!
//! These are language diagnostics, reported like any other, and a program
//! that fails any of them is not linked.
//!
//! A program's quantum circuits can also be written out on their own, as a
//! circuit archive ([`archive`]), for a target that runs static circuits:
//! one that lifts a measured value (`EL08`), or asks for qubits in a number
//! known only as it runs (`EL09`), is refused there.
//!
//! The objects themselves are handed to the platform's linker. A failure
//! there is not a fault in the program (every unit has already compiled
//! cleanly), so it is reported as a [`ToolError`], with the linker's own
//! output attached.

pub mod msvc;

use std::fmt;
use std::path::{Path, PathBuf};

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::meta::{MainDecl, Metadata, Use, UseKind, check};
use crate::span::Span;

/// A failure of the linker or its installation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolError {
    /// No linker was named and none is installed.
    NotFound,
    /// The linker could not be started.
    Spawn {
        /// The program that was run.
        program: PathBuf,
        /// Why it did not start.
        reason: String,
    },
    /// The linker ran and reported failure.
    Failed {
        /// The program that was run.
        program: PathBuf,
        /// Its exit status, if it exited normally.
        status: Option<i32>,
        /// What it printed.
        output: String,
    },
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::NotFound => f.write_str(
                "no linker was found: install Visual Studio or the Visual Studio Build \
                 Tools with the C++ workload, or name a linker with `--linker`",
            ),
            ToolError::Spawn { program, reason } => {
                write!(f, "the linker {} could not be started: {reason}", program.display())
            }
            ToolError::Failed {
                program,
                status,
                output,
            } => {
                write!(f, "the linker {} failed", program.display())?;
                if let Some(s) = status {
                    write!(f, " with status {s}")?;
                }
                if !output.is_empty() {
                    write!(f, ":\n{output}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ToolError {}

/// One unit of a program being linked.
#[derive(Clone, Copy, Debug)]
pub struct Linked<'a> {
    /// Its metadata.
    pub meta: &'a Metadata,
    /// Where its `main` is declared, when it was compiled from source in this
    /// run and so has a source to point into.
    pub main_span: Option<Span>,
}

/// Attaches a label at `span` if there is one.
fn at(d: Diagnostic, span: Option<Span>, label: &str) -> Diagnostic {
    match span {
        Some(s) => d.at_with(s, label),
        None => d,
    }
}

/// Checks what only the whole program can: that its units were compiled for
/// this version of the language, that every unit one imports is present and
/// still offers what the importer was compiled against, and that exactly one
/// unit defines a `main` able to start the program.
pub fn check_program(units: &[Linked<'_>], interner: &Interner) -> Vec<Diagnostic> {
    check_program_for(units, interner, true)
}

/// [`check_program`], with the check for a `main` made only when `needs_main`:
/// a program built to run its tests starts at them instead.
pub fn check_program_for(units: &[Linked<'_>], interner: &Interner, needs_main: bool) -> Vec<Diagnostic> {
    let mut out = Vec::new();

    // each unit compiled for this version, and named once
    for u in units {
        if u.meta.version != crate::LANGUAGE_VERSION {
            out.push(
                Diagnostic::new(Code::El11)
                    .with_message(format!(
                        "`{}` was compiled for version {} of the language, but this program is \
                         linked for version {}",
                        u.meta.unit,
                        u.meta.version,
                        crate::LANGUAGE_VERSION
                    ))
                    .with_note("metadata from another version cannot be trusted to mean the same thing")
                    .with_help(format!("recompile `{}` with this compiler", u.meta.unit)),
            );
        }
    }
    for (i, u) in units.iter().enumerate() {
        if units[..i].iter().any(|o| o.meta.unit == u.meta.unit) {
            out.push(
                Diagnostic::new(Code::El06)
                    .with_message(format!(
                        "two units named `{}` are linked into this program, so every item they \
                         both declare is defined twice",
                        u.meta.unit
                    ))
                    .with_note("a unit is named after its file, so two files of one name clash wherever they live")
                    .with_help("rename one of the files"),
            );
        }
    }

    // then what they use of one another, and what the program as a whole needs
    for u in units {
        for used in &u.meta.uses {
            if let Some(d) = check_use(u.meta, used, units, interner) {
                out.push(d);
            }
        }
    }
    out.extend(check_methods(units, interner));
    out.extend(check_exports(units, interner));
    out.extend(check_records(units, interner));
    if needs_main {
        out.extend(check_main(units));
    }
    if let Err(d) = startup_order(units) {
        out.push(d);
    }
    out
}

/// The units whose initialisers run before `main`, in the order they run:
/// every unit after the units it imports, and units neither of which imports
/// the other in the order of their names.
///
/// # Errors
///
/// `EL04` when units with initialisers import one another in a cycle, so
/// that no unit of the cycle can go first.
pub fn startup_order(units: &[Linked<'_>]) -> Result<Vec<String>, Diagnostic> {
    use std::collections::{BTreeMap, BTreeSet};
    let name = |u: &Linked<'_>| u.meta.unit.clone();
    // what each unit imports, among the units linked
    let present: BTreeSet<String> = units.iter().map(name).collect();
    let mut needs: BTreeMap<String, BTreeSet<String>> = units
        .iter()
        .map(|u| {
            let imports = u
                .meta
                .interface
                .imports
                .iter()
                .map(|p| crate::driver::unit_of(p).to_owned())
                .filter(|i| present.contains(i) && *i != u.meta.unit)
                .collect();
            (name(u), imports)
        })
        .collect();
    // a quantum unit's initialiser runs while its circuits are generated, and
    // has no part in the program's start
    let has_init: BTreeSet<String> = units
        .iter()
        .filter(|u| u.meta.init && !u.meta.interface.quantum)
        .map(name)
        .collect();
    let mut order = Vec::new();
    // each time, the first by name of the units whose imports have all gone
    // before
    let settle = |needs: &mut BTreeMap<String, BTreeSet<String>>, order: &mut Vec<String>| {
        while let Some(next) = needs.iter().find(|(_, n)| n.is_empty()).map(|(u, _)| u.clone()) {
            needs.remove(&next);
            for n in needs.values_mut() {
                n.remove(&next);
            }
            order.push(next);
        }
    };
    settle(&mut needs, &mut order);
    if !needs.is_empty() {
        // what is left imports, directly or not, a cycle. units that only
        // depend on a cycle are set aside until only the cycles remain
        let mut cycles = needs.clone();
        loop {
            let importers: BTreeSet<String> = cycles.values().flatten().cloned().collect();
            let before = cycles.len();
            cycles.retain(|u, _| importers.contains(u));
            if cycles.len() == before {
                break;
            }
        }
        let cyclic: Vec<&String> = cycles.keys().filter(|u| has_init.contains(*u)).collect();
        if !cyclic.is_empty() {
            let all: Vec<String> = cycles.keys().map(|u| format!("`{u}`")).collect();
            return Err(Diagnostic::new(Code::El04)
                .with_message(format!(
                    "the units {} import one another in a cycle, and `{}` has an initialiser, so \
                     there is no order to run the initialisers in",
                    all.join(", "),
                    cyclic[0]
                ))
                .with_note(
                    "a unit's initialiser runs after those of every unit it imports, so that it \
                     finds their objects given their values; in a cycle every unit would have \
                     to go first",
                )
                .with_help("break the cycle, or move the initialisation out of the units in it"));
        }
        // no initialiser in the cycles: they run nothing, so the units after
        // them are ordered as if the cycles had gone first
        for c in cycles.keys() {
            needs.remove(c);
            for n in needs.values_mut() {
                n.remove(c);
            }
        }
        settle(&mut needs, &mut order);
    }
    Ok(order.into_iter().filter(|u| has_init.contains(u)).collect())
}

/// No two units give one type a method of the same name. Within a unit that
/// is refused where it is written; across units (a unit adding a method to
/// another's `[open]` type) only the whole program can see it.
fn check_methods(units: &[Linked<'_>], interner: &Interner) -> Vec<Diagnostic> {
    type Key = (String, crate::intern::Symbol, crate::intern::Symbol, bool);
    let mut defined: Vec<(Key, &str)> = Vec::new();
    let mut out = Vec::new();
    for u in units {
        let iface = &u.meta.interface;
        for f in &iface.fns {
            let Some(m) = f.method else { continue };
            let owner = iface.types.adt(m.owner);
            if !owner.args.is_empty() {
                continue;
            }
            let origin = owner.origin.clone().unwrap_or_else(|| u.meta.unit.clone());
            let key = (origin, owner.name, f.name, m.operator);
            match defined.iter().find(|(k, _)| *k == key) {
                Some((_, first)) => {
                    let (origin, ty, name, operator) = &key;
                    let method = format!("{}{}", if *operator { "$" } else { "" }, interner.resolve(*name));
                    out.push(
                        Diagnostic::new(Code::El07)
                            .with_message(format!(
                                "`{origin}::{}` is given a method `{method}` by both `{first}` and `{}`",
                                interner.resolve(*ty),
                                u.meta.unit
                            ))
                            .with_note(
                                "a type has one method of each name in a whole program, whichever \
                                 unit adds it, so a call of it means one thing everywhere",
                            )
                            .with_help("rename one of the two methods"),
                    );
                }
                None => defined.push((key, u.meta.unit.as_str())),
            }
        }
    }
    out
}

/// Checks one item an importer uses against what its unit now offers.
fn check_use(importer: &Metadata, used: &Use, units: &[Linked<'_>], interner: &Interner) -> Option<Diagnostic> {
    let name = interner.resolve(used.name);
    let full = match &used.owner {
        Some(o) => format!(
            "{}::{}::{}{name}",
            o.unit,
            interner.resolve(o.ty),
            if o.operator { "$" } else { "" }
        ),
        None => format!("{}::{name}", used.unit),
    };
    let recompile = format!(
        "recompile `{}` against the current `{}`",
        importer.unit, used.unit
    );
    let Some(exporter) = units.iter().find(|x| x.meta.unit == used.unit) else {
        return Some(
            Diagnostic::new(Code::El06)
                .with_message(format!(
                    "`{}` uses `{full}`, but no unit `{}` is part of this program",
                    importer.unit, used.unit
                ))
                .with_help(format!(
                    "add `{0}.tq` or `{0}.tcu` to the files being built",
                    used.unit
                )),
        );
    };
    let iface = &exporter.meta.interface;
    let this = exporter.meta.unit.as_str();
    // a `static` item may be used only by a body borrowed from its own unit's
    // source, and only while that source is the one the importer borrowed
    let current = iface.source.as_ref().map(crate::sema::interface::Source::digest);
    let sources = || importer.uses.iter().filter(|u| u.kind == UseKind::Source && u.unit == used.unit);
    let borrowed = sources().any(|u| Some(&u.expect) == current.as_ref());
    // borrowed from a source that has since changed, which is reported on its
    // own: what the old source used says nothing about the new one
    let stale = !borrowed && sources().next().is_some();
    let missing = || {
        Diagnostic::new(Code::El06)
            .with_message(format!(
                "`{}` uses the {} `{full}`, which `{}` no longer offers",
                importer.unit,
                used.kind.noun(),
                used.unit
            ))
            .with_note(format!(
                "`{}` was compiled when `{}` declared it without `static`",
                importer.unit, used.unit
            ))
            .with_help(recompile.clone())
    };
    let changed = |code: Code, what: &str, now: &str| {
        Diagnostic::new(code)
            .with_message(format!(
                "`{}` was compiled against `{full}` as `{}`, but {what} is now `{now}`",
                importer.unit, used.expect
            ))
            .with_note(format!(
                "`{}` changed after `{}` was compiled, so the two disagree about it",
                used.unit, importer.unit
            ))
            .with_help(recompile.clone())
    };
    match used.kind {
        UseKind::Function => {
            let lookup = |any: bool| match &used.owner {
                Some(o) => iface
                    .method(&o.unit, o.ty, used.name, o.operator)
                    .filter(|f| any || f.linkage == crate::ast::Linkage::Program),
                None if any => iface.any_function(used.name),
                None => iface.function(used.name),
            };
            let Some(f) = lookup(borrowed) else {
                if stale && lookup(true).is_some() {
                    return None;
                }
                return Some(missing());
            };
            let now = check::signature_text(&iface.types, &f.params, f.ret, this, interner);
            (now != used.expect).then(|| changed(Code::El02, "its signature", &now))
        }
        UseKind::Object => {
            let found = if borrowed { iface.any_global(used.name) } else { iface.global(used.name) };
            let Some(g) = found else {
                if stale && iface.any_global(used.name).is_some() {
                    return None;
                }
                return Some(missing());
            };
            let now = check::object_text(&iface.types, g.ty, g.constant, this, interner);
            if now != used.expect {
                return Some(changed(Code::El02, "its type", &now));
            }
            let value = g.value.as_ref().map(|v| check::value_digest(v, interner));
            (value != used.value).then(|| {
                Diagnostic::new(Code::El12)
                    .with_message(format!(
                        "`{}` was compiled with an earlier value of the constant `{full}`",
                        importer.unit
                    ))
                    .with_note(format!(
                        "a constant's value is built into every unit that uses it, so `{}` \
                         still holds the value `{}` had when `{}` was compiled",
                        importer.unit, used.unit, importer.unit
                    ))
                    .with_help(recompile.clone())
            })
        }
        UseKind::Source => {
            let (importer, unit) = (&importer.unit, &used.unit);
            // what the importer took from the source: a quantum unit's
            // operators, placed in its circuits, or another unit's generic and
            // constant functions
            let (what, took) = if iface.quantum {
                ("operators".to_owned(), format!("placed `{unit}`'s operators in its circuits"))
            } else {
                (
                    "generic or constant functions".to_owned(),
                    format!("made its own instances of `{unit}`'s generic functions, or evaluated its constant functions,"),
                )
            };
            (current.as_deref() != Some(used.expect.as_str())).then(|| {
                Diagnostic::new(Code::El02)
                    .with_message(format!("`{importer}` was compiled against an earlier version of `{unit}`'s {what}"))
                    .with_note(format!(
                        "`{importer}` {took} from the source `{unit}` had then; that source has changed since"
                    ))
                    .with_help(recompile.clone())
            })
        }
        UseKind::Record => {
            let Some(p) = iface.record(name) else {
                return Some(missing());
            };
            let now = crate::meta::encode::record_digest(p);
            (now != used.expect).then(|| {
                Diagnostic::new(Code::El03)
                    .with_message(format!(
                        "the judgment record of `{full}` changed after `{}` was compiled against it",
                        importer.unit
                    ))
                    .with_note(format!(
                        "`{}` placed `{full}`'s gates in its circuits and read its judgment as they \
                         were; `{}` now publishes another",
                        importer.unit, used.unit
                    ))
                    .with_help(recompile.clone())
            })
        }
        UseKind::Type => {
            let Some(id) = iface.types.find_adt(Some(&used.unit), used.name) else {
                return Some(missing());
            };
            let now = check::adt_digest(&iface.types, id, this, interner);
            (now != used.expect).then(|| {
                Diagnostic::new(Code::El03)
                    .with_message(format!(
                        "the layout of `{full}` changed after `{}` was compiled against it",
                        importer.unit
                    ))
                    .with_note(format!(
                        "`{}` would read and write `{full}` values at the old offsets, and \
                         `{}` at the new ones",
                        importer.unit, used.unit
                    ))
                    .with_help(recompile.clone())
            })
        }
    }
}

/// The quantum operators whose judgements cross a unit boundary: for each
/// unit, each operator of a quantum unit it uses (one whose record it
/// read, or an `[entry]` operator it holds a handle to), with the unit that
/// declares it.
fn crossings<'a>(units: &'a [Linked<'a>], interner: &Interner) -> Vec<(&'a Metadata, &'a Metadata, &'a crate::judge::record::Published)> {
    let mut out = Vec::new();
    for u in units {
        for used in &u.meta.uses {
            if !matches!(used.kind, UseKind::Record | UseKind::Function) || used.owner.is_some() {
                continue;
            }
            let Some(q) = units.iter().find(|x| x.meta.unit == used.unit && x.meta.interface.quantum) else {
                continue;
            };
            if let Some(p) = q.meta.interface.record(interner.resolve(used.name))
                && !out.iter().any(|(i, d, x): &(&Metadata, &Metadata, &crate::judge::record::Published)| {
                    i.unit == u.meta.unit && d.unit == q.meta.unit && x.name == p.name
                })
            {
                out.push((u.meta, q.meta, p));
            }
        }
    }
    out
}

/// A claim as `[expect: …]` writes it.
fn claim_text(c: &crate::judge::record::Claim) -> String {
    use crate::judge::record::Claim;
    match c {
        Claim::Monic => "monic".to_owned(),
        Claim::Unitary => "unitary".to_owned(),
        Claim::Contract(k) => k.to_string(),
        Claim::Frame(None) => "frame(flat)".to_owned(),
        Claim::Frame(Some(k)) => format!("frame(flat({}))", crate::judge::cert::show_phase(*k)),
        Claim::Outcomes(t) => format!("kernel.outcomes = {t}"),
    }
}

/// What the program's units say of the quantum operators whose judgements
/// cross between them: that the conductors the two units are judged at
/// meet within the limit (`EL10`); that a claim whose meaning depends on a
/// phase convention is published with the cover and gauge it is stated
/// against, as declarations an importer can name (`EJ08`); and that each
/// published claim still holds of the record it is published with (`EJ03`),
/// checked from the record, never from the operator's body.
fn check_records(units: &[Linked<'_>], interner: &Interner) -> Vec<Diagnostic> {
    let limit = crate::diag::Limit::Conductor.value();
    let mut out = Vec::new();
    let mut conductors: Vec<(&str, &str)> = Vec::new();
    for (importer, declarer, p) in crossings(units, interner) {
        let full = format!("{}::{}", declarer.unit, p.name);
        let (m, n) = (importer.conductor.max(1), declarer.conductor.max(1));
        let both = num_integer::lcm(m, n);
        if both > limit && !conductors.contains(&(importer.unit.as_str(), declarer.unit.as_str())) {
            conductors.push((importer.unit.as_str(), declarer.unit.as_str()));
            out.push(
                Diagnostic::new(Code::El10)
                    .with_message(format!(
                        "`{}`, of conductor {m}, uses `{full}`, of conductor {n}: their judgments meet at \
                         conductor {both}, above the limit of {limit}",
                        importer.unit
                    ))
                    .with_note(
                        "a judgment crossing between units is checked at the least common multiple of \
                         their conductors, whose phase group holds both units' phases",
                    )
                    .with_help(format!(
                        "declare both units with conductors whose least common multiple is at most {limit}, \
                         such as the same one"
                    )),
            );
        }
        for c in &p.claims {
            if c.needs_gauge() {
                let why = match (&p.cover, &p.gauge) {
                    (None, _) => Some("its cover is not a declaration: it is written out, or not written".to_owned()),
                    (_, None) => Some("its gauge is not a declaration: it is written out, or not written".to_owned()),
                    (Some(d), _) | (_, Some(d)) if !d.program => {
                        Some(format!("`{}` is declared `static`, so no importer can name it", d.name))
                    }
                    _ => None,
                };
                if let Some(why) = why {
                    out.push(
                        Diagnostic::new(Code::Ej08)
                            .with_message(format!(
                                "`{full}` publishes `{}` to `{}`, but {why}",
                                claim_text(c),
                                importer.unit
                            ))
                            .with_note(
                                "whether an operator satisfies this class depends on the choice of gauge, as \
                                 whether a matrix is diagonal depends on the basis; published without its \
                                 cover and gauge it asserts nothing an importer can check",
                            )
                            .with_help(format!(
                                "declare the cover and gauge in `{}` without `static` and name them in \
                                 `[cover:]` and `[gauge:]`, or publish a `stat` or `sector` claim, which \
                                 needs no convention",
                                declarer.unit
                            )),
                    );
                }
            }
            if p.record.fragment.departure.is_none() && c.holds(&p.record, &p.outcomes) == Some(false) {
                out.push(
                    Diagnostic::new(Code::Ej03)
                        .with_message(format!(
                            "`{full}` claims `{}`, but the record `{}` publishes with it does not bear that out",
                            claim_text(c),
                            declarer.unit
                        ))
                        .with_note(format!(
                            "`{}` uses `{full}` on the strength of its published claims, checked against its \
                             record, not its body",
                            importer.unit
                        ))
                        .with_help(format!("recompile `{}`", declarer.unit)),
                );
            }
        }
    }
    out
}

/// The circuit archive of a program whose units' metadata is `units`: the
/// circuit of each `[entry]` operator of its quantum units, library units
/// aside, each with its OpenQASM 3 and its judgement in its document.
///
/// ```text
/// Archive {
///     edition: 2,
///     members: [
///         Member { name: "deutsch::deutsch", sig: "…", qasm: r#"…"#, document: r#"Circuit { … }"# },
///     ],
/// }
/// ```
///
/// An archive holds static circuits, for a target that runs a circuit as it
/// is given: one that lifts a measured value is refused (`EL08`), and so is
/// one that asks for qubits as it runs, in a number known only then
/// (`EL09`). Neither is made static by unrolling: that would be another
/// circuit, with another judgement.
pub fn archive(units: &[&Metadata], interner: &Interner) -> (String, Vec<Diagnostic>) {
    use crate::tcon::write::{self, Doc};
    let mut members = Vec::new();
    let mut out = Vec::new();
    for m in units {
        if !m.interface.quantum || crate::library::source(&m.unit).is_some() {
            continue;
        }
        for c in &m.interface.circuits {
            // a static circuit, or else a diagnostic saying why not
            let full = format!("{}::{}", m.unit, interner.resolve(c.name));
            if c.dynamic {
                out.push(
                    Diagnostic::new(Code::El08)
                        .with_message(format!(
                            "`{full}` lifts a measured value, which a circuit archive cannot hold"
                        ))
                        .with_note(
                            "an archive holds static circuits, for a target that cannot generate more of \
                             a circuit as it runs; unrolling the lift would make another circuit, with \
                             another judgment",
                        )
                        .with_help("build a program, whose processor runs dynamic circuits, rather than an archive"),
                );
            }
            if c.allocates {
                out.push(
                    Diagnostic::new(Code::El09)
                        .with_message(format!(
                            "`{full}` asks for qubits, or grows a register, as it runs, in a number \
                             known only then, which a circuit archive cannot hold"
                        ))
                        .with_note(
                            "an archive holds circuits for a target that allocates every register before \
                             the circuit starts, and names each qubit an operation acts on then",
                        )
                        .with_help(
                            "allocate outside the loop and use a register of fixed length, or build a \
                             program, whose processor allocates registers as it runs, rather than an archive",
                        ),
                );
            }

            // a member for it either way; the diagnostics decide whether the archive is written
            members.push(Doc::typed(
                "Member",
                vec![
                    ("name", Doc::text(&full)),
                    ("sig", Doc::text(&c.sig)),
                    ("qasm", Doc::Text(c.qasm.clone())),
                    ("document", Doc::Text(c.document.clone())),
                ],
            ));
        }
    }
    let doc = Doc::typed(
        "Archive",
        vec![("edition", Doc::Int(u64::from(crate::LANGUAGE_EDITION))), ("members", Doc::Array(members))],
    );
    (write::write(&doc), out)
}

/// No symbol given by `[export]` in two units: each names one definition.
fn check_exports(units: &[Linked<'_>], interner: &Interner) -> Vec<Diagnostic> {
    let mut seen: std::collections::HashMap<&str, (&str, &str)> = std::collections::HashMap::new();
    let mut out = Vec::new();
    for u in units {
        let iface = &u.meta.interface;
        let fns = iface.fns.iter().filter_map(|f| f.symbol.as_deref().map(|s| (s, f.name)));
        let globals = iface.globals.iter().filter_map(|g| g.symbol.as_deref().map(|s| (s, g.name)));
        for (sym, name) in fns.chain(globals) {
            let item = interner.resolve(name);
            match seen.get(sym) {
                Some(&(unit, first)) => out.push(
                    Diagnostic::new(Code::El06)
                        .with_message(format!(
                            "the symbol `{sym}` is exported by `{unit}::{first}` and by `{}::{item}`",
                            u.meta.unit
                        ))
                        .with_note("an exported symbol names one definition in the whole program")
                        .with_help(format!("give `{}::{item}` another symbol in its `[export]`", u.meta.unit)),
                ),
                None => {
                    seen.insert(sym, (u.meta.unit.as_str(), item));
                }
            }
        }
    }
    out
}

/// Exactly one `main` that can start the program.
fn check_main(units: &[Linked<'_>]) -> Vec<Diagnostic> {
    let valid: Vec<&Linked<'_>> = units.iter().filter(|u| u.meta.main == MainDecl::Valid).collect();
    let invalid: Vec<(&Linked<'_>, &str)> = units
        .iter()
        .filter_map(|u| match &u.meta.main {
            MainDecl::Invalid(why) => Some((u, why.as_str())),
            _ => None,
        })
        .collect();
    match valid.as_slice() {
        [_] => Vec::new(),
        [] if !invalid.is_empty() => invalid
            .into_iter()
            .map(|(u, why)| {
                let d = Diagnostic::new(Code::El05).with_message(format!(
                    "the `main` in `{}` cannot start the program: {why}",
                    u.meta.unit
                ));
                at(d, u.main_span, "here")
                    .with_help("declare it as `fn main() -> i32`; its result becomes the exit status")
            })
            .collect(),
        [] => vec![
            Diagnostic::new(Code::El05)
                .with_message("this program has no `main`, so there is nowhere for it to start")
                .with_note(
                    "a program starts by calling `fn main() -> i32` in exactly one of its \
                     units, and exits with the status that call returns",
                )
                .with_help("add `fn main() -> i32 { 0 }` to one of the units"),
        ],
        [first, rest @ ..] => rest
            .iter()
            .map(|u| {
                let d = Diagnostic::new(Code::El05).with_message(format!(
                    "this program has more than one `main`: `{}` and `{}` both define one",
                    first.meta.unit, u.meta.unit
                ));
                let d = at(d, u.main_span, "a second `main`");
                let d = match first.main_span {
                    Some(s) => d.also(s, "the first `main`"),
                    None => d,
                };
                d.with_note("a program starts in exactly one place")
            })
            .collect(),
    }
}

/// Links `objects` into the executable `out`, using `linker` if one is named.
///
/// # Errors
///
/// A [`ToolError`] if no linker is found or it fails.
pub fn link(objects: &[PathBuf], out: &Path, linker: Option<&Path>) -> Result<(), ToolError> {
    msvc::Linker::find(linker)?.link(objects, out)
}

/// Links `objects` into the module `out`, a library a program loads while it
/// runs, using `linker` if one is named.
///
/// # Errors
///
/// A [`ToolError`] if no linker is found or it fails.
pub fn link_module(objects: &[PathBuf], out: &Path, linker: Option<&Path>) -> Result<(), ToolError> {
    msvc::Linker::find(linker)?.link_library(objects, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::testing::metadata_of;

    fn codes(ms: &[&Metadata], i: &Interner) -> Vec<Code> {
        let linked: Vec<Linked<'_>> = ms.iter().map(|m| Linked { meta: m, main_span: None }).collect();
        check_program(&linked, i).iter().map(|d| d.code).collect()
    }

    const GEO: &str = "struct P { x: i32 }\nfn area(p: *P) -> i32 { p.x }\nlet LIMIT: const i32 = 7;";
    const APP: &str = "import geo;\n\
                       fn main() -> i32 { let p = geo::P { x: 2 }; geo::area(&p) + geo::LIMIT }";

    /// `app` compiled against `geo` as `before`, linked with `geo` as `after`.
    fn relink(before: &str, after: &str) -> Vec<Code> {
        let mut i = Interner::new();
        let old = metadata_of("geo", before, &[], &mut i);
        let app = metadata_of("app", APP, std::slice::from_ref(&old.interface), &mut i);
        let new = metadata_of("geo", after, &[], &mut i);
        codes(&[&app, &new], &i)
    }

    #[test]
    fn one_main_is_a_program() {
        let mut i = Interner::new();
        let a = metadata_of("a", "fn main() -> i32 { 0 }", &[], &mut i);
        let b = metadata_of("b", "fn helper() { }", &[], &mut i);
        assert_eq!(codes(&[&a, &b], &i), []);
    }

    #[test]
    fn no_main_is_el05() {
        let mut i = Interner::new();
        let a = metadata_of("a", "fn helper() { }", &[], &mut i);
        assert_eq!(codes(&[&a], &i), [Code::El05]);
    }

    #[test]
    fn a_main_that_cannot_start_the_program_says_why() {
        let mut i = Interner::new();
        let a = metadata_of("a", "fn main() { }", &[], &mut i);
        let d = check_program(&[Linked { meta: &a, main_span: None }], &i);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Code::El05);
        assert!(d[0].message.contains("i32"), "{}", d[0].message);
    }

    #[test]
    fn two_mains_is_el05_naming_both() {
        let mut i = Interner::new();
        let a = metadata_of("a", "fn main() -> i32 { 0 }", &[], &mut i);
        let b = metadata_of("b", "fn main() -> i32 { 1 }", &[], &mut i);
        let d = check_program(
            &[
                Linked { meta: &a, main_span: Some(Span::synthetic()) },
                Linked { meta: &b, main_span: Some(Span::synthetic()) },
            ],
            &i,
        );
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Code::El05);
        assert_eq!(d[0].labels.len(), 2);
        assert!(d[0].message.contains("`a`") && d[0].message.contains("`b`"), "{}", d[0].message);
    }

    #[test]
    fn an_importer_linked_with_what_it_was_compiled_against_is_consistent() {
        assert_eq!(relink(GEO, GEO), []);
    }

    #[test]
    fn a_changed_layout_is_el03() {
        let after = "struct P { x: i32, y: i32 }\nfn area(p: *P) -> i32 { p.x }\nlet LIMIT: const i32 = 7;";
        assert_eq!(relink(GEO, after), [Code::El03]);
    }

    #[test]
    fn a_changed_signature_is_el02() {
        let after = "struct P { x: i32 }\nfn area(p: *P, k: i32) -> i32 { p.x }\nlet LIMIT: const i32 = 7;";
        assert_eq!(relink(GEO, after), [Code::El02]);
    }

    #[test]
    fn a_removed_item_is_el06() {
        let after = "struct P { x: i32 }\nstatic fn area(p: *P) -> i32 { p.x }\nlet LIMIT: const i32 = 7;";
        assert_eq!(relink(GEO, after), [Code::El06]);
    }

    #[test]
    fn a_changed_constant_is_el12() {
        let after = "struct P { x: i32 }\nfn area(p: *P) -> i32 { p.x }\nlet LIMIT: const i32 = 8;";
        assert_eq!(relink(GEO, after), [Code::El12]);
    }

    #[test]
    fn a_missing_unit_is_el06() {
        let mut i = Interner::new();
        let geo = metadata_of("geo", GEO, &[], &mut i);
        let app = metadata_of("app", APP, std::slice::from_ref(&geo.interface), &mut i);
        let got = codes(&[&app], &i);
        assert!(got.iter().all(|c| *c == Code::El06) && !got.is_empty(), "{got:?}");
    }

    #[test]
    fn another_version_or_a_second_unit_of_one_name_is_refused() {
        let mut i = Interner::new();
        let mut a = metadata_of("a", "fn main() -> i32 { 0 }", &[], &mut i);
        a.version = "0.1".to_owned();
        assert_eq!(codes(&[&a], &i), [Code::El11]);
        let b = metadata_of("b", "fn f() { }", &[], &mut i);
        let c = metadata_of("b", "fn g() { }", &[], &mut i);
        let a = metadata_of("a", "fn main() -> i32 { 0 }", &[], &mut i);
        assert_eq!(codes(&[&a, &b, &c], &i), [Code::El06]);
    }

    #[test]
    fn one_symbol_exported_by_two_units_is_el06() {
        let mut i = Interner::new();
        let a = metadata_of("a", "fn main() -> i32 { 0 }\n[export: \"shared\"]\nfn f() { }", &[], &mut i);
        let b = metadata_of("b", "[export: \"shared\"]\nlet X: i32 = 1;", &[], &mut i);
        assert_eq!(codes(&[&a, &b], &i), [Code::El06]);
        let c = metadata_of("c", "[export: \"other\"]\nfn g() { }", &[], &mut i);
        assert_eq!(codes(&[&a, &c], &i), []);
    }

    #[test]
    fn a_program_built_for_its_tests_needs_no_main() {
        let mut i = Interner::new();
        let a = metadata_of("a", "[test]\nfn t() { }", &[], &mut i);
        let linked = [Linked {
            meta: &a,
            main_span: None,
        }];
        assert_eq!(check_program(&linked, &i).iter().map(|d| d.code).collect::<Vec<_>>(), [Code::El05]);
        assert!(check_program_for(&linked, &i, false).is_empty());
    }

    #[test]
    fn a_missing_linker_says_how_to_get_one() {
        assert!(ToolError::NotFound.to_string().contains("--linker"));
    }

    #[test]
    fn a_failed_link_shows_the_linkers_output() {
        let e = ToolError::Failed {
            program: PathBuf::from("link.exe"),
            status: Some(1120),
            output: "unresolved external symbol".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("1120") && s.contains("unresolved external symbol"), "{s}");
    }
}
