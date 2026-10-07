//! Source text: identity, spans, decoding and line splicing.
//!
//! This module owns translation phases 1 and 2, and the coordinates every
//! later phase reports positions in.
//!
//! **Spans are always original-file coordinates.** The lexer scans
//! [`Spliced::text`], which has lost its backslash-newline pairs, but maps
//! every offset back through [`Spliced::to_original`] before building a
//! [`crate::span::Span`]. Nothing downstream knows splicing happened.

pub mod source_map;

pub mod splice;

pub use source_map::{FileKind, SourceFile, SourceMap};

pub use splice::{DecodeError, Spliced};
