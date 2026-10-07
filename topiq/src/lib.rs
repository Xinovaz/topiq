//! # Topiq
//!
//! A hybrid classical/quantum systems language with data-oriented design, RAII
//! resource semantics over both classical memory and quantum state,
//! compile-time introspection in place of inheritance, and a decidable
//! phase-contract judgement system.
//!
//! The crate is a library: each phase of translation is a module that can be
//! driven and tested on its own. The `tqc` binary is a thin driver over it.
//!
//! ## Phases of translation
//!
//! A Topiq source file becomes a program in ten phases, each handing a
//! complete artefact to the next:
//!
//! | phase | what it does | module |
//! |---|---|---|
//! | 1. Decoding | reads the file as UTF-8 | [`source::splice`] |
//! | 2. Line splicing | joins lines ending in a backslash | [`source::splice`] |
//! | 3. Tokenisation | text becomes tokens | [`lex`] |
//! | 4. Preprocessing | directives run; macros expand | [`pp`] |
//! | 5. Parsing | tokens become a syntax tree | [`parse`] |
//! | 6. Name resolution | every name is tied to what it denotes | [`sema`] |
//! | 7. Constant evaluation | what can be computed during translation is | [`sema`] |
//! | 8. Type checking | every expression gets a type | [`sema`] |
//! | 9. Circuit lowering, judgement | quantum operators become circuits, and their claims are derived | [`lower`] |
//! | 10. Code generation | machine code is emitted | `codegen` |
//!
//! Phases 1 to 5 cover the whole language. Phases 6 to 8 and 10 cover the
//! whole classical language and its library (`core`, `str`, `dyn`, `tcon`,
//! `quon`, `judge`, `la`, `qpu`, `sim`, `circuit`, `module` and `fs`) and build a
//! program into an executable, a module loaded while a program runs, or a
//! runner for its tests.
//!
//! A quantum unit goes through phases 6 to 9. It is analysed in full: its
//! qubits, measurements and ancillae, the rule that no live qubit is
//! dropped, and the covers, gauges, base types, locales, chains and map
//! locales it declares, whose geometry [`judge`] decides exactly. The
//! classical operators mean the reversible operations they stand for on
//! qubits, registers and `quint`s, so that an algorithm reads the same in
//! both kinds of unit ([`sema::qbits`]). [`lower`]
//! then turns its operators into [circuits](circuit) of the `gates`
//! library's gates, and each state `prep` prepares into a circuit that
//! prepares it exactly. Each operator's judgement is derived from its
//! circuit, and every claim about it is checked exactly. What is not yet
//! translated is reported as `TQ003` where it is written; [`sema`] lists
//! those constructs.
//!
//! A `#unit any` unit is translated as whichever kind its importer or the
//! command line asks for, with `#alias` choosing its types for that kind
//! ([`pp`]); one source can be a unit of each kind in one program
//! ([`driver::program`]).
//!
//! A quantum unit's output is its `.tqu`: its metadata, holding each
//! `[entry]` operator's circuit, in OpenQASM 3 and as a
//! [document](circuit::document), and the judgement record of each operator
//! with program linkage, which importers read rather than derive again.
//!
//! A classical program runs a quantum unit's circuits through the `qpu`
//! library, on any device registered with it: the `sim` library's simulator,
//! which computes the state exactly and is always there, or a backend the
//! program registers for a device elsewhere. With the
//! `circuit` library it makes new circuits from them as it runs, each
//! carrying a judgement composed as a translation's is.
//!
//! [`mark`] runs between phases 4 and 5 and decides whether each `[` opens an
//! annotation or an array, which the parser cannot decide because the answer
//! changes what it is parsing.
//!
//! ## From source to program
//!
//! Analysis hands code generation a [typed intermediate representation](tir)
//! in which every name is resolved, every type settled and every constant
//! folded. Each unit's object carries its [metadata](meta): what it offers
//! importers, and what it used from the units it imports. A unit compiled
//! earlier is imported and linked from its object alone, and an object out of
//! date with what it imports is refused when the program is linked.
//!
//! `link` checks the judgement records that cross between the program's
//! units, then links its objects into an executable, or into a module that
//! embeds its quantum units' circuits, or writes those circuits alone as a
//! circuit archive, `.tqar`. [`driver`] finds the units a program imports,
//! and builds, runs and tests it.
//!
//! ## Documentation
//!
//! `///` and `//!` comments are documentation, which the lexer keeps beside
//! the tokens and the parser gives to the declaration or unit they precede.
//! [`doc`] writes a program's documentation, in Markdown, as a site of HTML
//! pages covering every unit the program is made of, the library's included.
//!
//! ## Diagnostics
//!
//! Every way a Topiq program can fail carries an identifier, such as `EU01`
//! or `RA07`, so a failure can be looked up and tested against without
//! depending on the message's wording. [`diag::Code`] declares the whole
//! table, the quantum side's identifiers among them, so that no identifier is
//! invented or reused with another meaning.
//!
//! ## Example
//!
//! ```
//! use topiq::source::SourceMap;
//!
//! let mut sources = SourceMap::new();
//! let id = sources.add_text("demo.tq", "#unit classical\nlet X: const u32 = 2;\n");
//! assert_eq!(sources.file(id).unit_name(), "demo");
//! ```

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]

pub mod ast;
pub mod circuit;
#[cfg(feature = "llvm")]
pub mod codegen;
pub mod diag;
pub mod doc;
pub mod driver;
pub mod exact;
pub mod intern;
pub mod judge;
pub mod lex;
pub mod library;
#[cfg(feature = "llvm")]
pub mod link;
pub mod lower;
pub mod mark;
pub mod meta;
pub mod parse;
pub mod pp;
pub mod quon;
pub mod sema;
pub mod source;
pub mod span;
pub mod tcon;
pub mod tir;

/// The edition of the language this implementation targets.
///
/// The value of the predefined macro `__TOPIQ__`, written into every unit's
/// metadata. Linking two units that disagree on it is `EL11`: their records
/// were written for different editions, and cannot be compared.
pub const LANGUAGE_EDITION: u32 = 2;

/// The language version string.
pub const LANGUAGE_VERSION: &str = "0.2";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_edition_and_version_agree() {
        assert_eq!(LANGUAGE_EDITION, 2);
        assert_eq!(LANGUAGE_VERSION, "0.2");
    }
}
