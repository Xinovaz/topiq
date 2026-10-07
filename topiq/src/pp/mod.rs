//! Translation phase 4: running directives and expanding macros.
//!
//! The preprocessor operates on the token stream produced by phase 3, executes
//! directives, expands macros, and hands [`crate::parse`] the final stream.
//!
//! | module | responsibility |
//! |---|---|
//! | [`directive`] | the driver: logical lines, directives, conditionals, pragmas |
//! | [`macros`] | the macro table, and the eight names the compiler predefines |
//! | [`expand`] | C99 expansion with `#`, `##` and `__VA_ARGS__` |
//! | [`hide`] | hide sets, so a macro does not re-expand within itself |
//! | [`eval`] | the small expression language a `#if` condition is written in |

pub mod directive;
pub mod eval;
pub mod expand;
pub mod hide;
pub mod macros;

pub use directive::{
    AnyUnit, EmbedResolver, Fragment, FsEmbedResolver, KindChoice, MapEmbedResolver, Preprocessed, QAlloc,
    UnitDirective, UnitKind, preprocess, preprocess_as,
};
pub use expand::Expander;
pub use hide::Hide;
pub use macros::{Linkage, MacroDef, MacroTable, PpToken};
