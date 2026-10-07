//! Translation phase 3: turning text into tokens.
//!
//! [`token`] defines the token type and the keyword and punctuator tables;
//! [`lexer`] is the scanner.
//!
//! There is no ket token, no glued `>>`, and no exact-scalar token. In each
//! case the lexer would have to decide something it cannot see enough to
//! decide, so it produces the pieces and a later pass that has the context
//! puts them together. [`token`] gives the reason for each.

pub mod lexer;
pub mod token;

pub use lexer::{DocComment, DocKind, Lexed, lex};
pub use token::{FloatSuffix, IntBase, IntSuffix, Keyword, Punct, StrKind, Token};
