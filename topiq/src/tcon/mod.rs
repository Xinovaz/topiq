//! TCON: the notation for writing classical data down.
//!
//! TCON is to Topiq roughly what JSON is to JavaScript, with two differences
//! that matter:
//!
//! - it is **typed**: every document is checked against a Topiq type, which
//!   acts as its schema, so a malformed document is a translation error rather
//!   than a surprise at run time; and
//! - it is **native**: a document is a Topiq expression, parsed by the Topiq
//!   parser.
//!
//! The second is why this module is small. The document grammar reuses the
//! lexer, the path parser and the literal forms of the main grammar instead of
//! defining a second syntax that would then have to be kept in step.
//!
//! [`parse`] reads a document; [`write`](mod@write) writes one, which is how a unit's
//! metadata is stored in its object.

pub mod ast;
pub mod parse;
pub mod write;

pub use ast::{Document, Pair, Scalar, Value};
