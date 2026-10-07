//! Translation phase 3: turning text into tokens.
//!
//! A hand-written cursor over the spliced text left by phase 2, rather than
//! the parser generator: longest-match scanning, raw numeric text and
//! raw-string delimiters are all easier to express directly.
//!
//! # Comments are one space
//!
//! Phase 3 replaces each comment with a single space. That is why a block
//! comment containing a newline does **not** put the following `#` in
//! line-initial position: after replacement there is no newline there any more.
//! A line comment is different, because the newline that ends it survives.
//!
//! # The `#` token
//!
//! A `#` at the start of a line opens a preprocessing directive. Inside a macro
//! body it is an operator instead, and a macro body is never at the start of a
//! line, so a lexer that only emitted line-initial `#` could not lex
//! `#define CAT(a, b) a ## b`. Both are therefore emitted wherever they appear,
//! and [`crate::pp`] makes the decision: it splits the stream into logical
//! lines and can see directly whether a `#` opens one.
//!
//! # Spans are original-file coordinates
//!
//! The cursor walks the spliced text, but every span is mapped back through
//! [`Spliced::to_original`] before it is recorded, so an identifier written
//! across a spliced line still underlines the backslash and the newline.
//!
//! # Doc comments
//!
//! A line comment opened by `///` (but not `////`) documents what follows it,
//! and one opened by `//!` documents the unit. Each is still trivia: it is
//! kept beside the tokens as a [`DocComment`], never among them, so the
//! preprocessor and the grammar never meet one. A doc comment records the
//! start of the token after it, its *anchor*, and the parser gives it to the
//! declaration that token begins. A doc comment whose anchor the
//! preprocessor removes, such as one inside an inactive `#if`, therefore
//! documents nothing.

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::source::Spliced;
use crate::span::{SourceId, Span};

use super::token::{FloatSuffix, IntBase, IntSuffix, Keyword, Punct, StrKind, Token};

/// The result of tokenising one source file.
#[derive(Clone, Debug)]
pub struct Lexed {
    /// Tokens in source order. Trivia is not included: it only separates
    /// tokens and is otherwise insignificant.
    pub tokens: Vec<(Token, Span)>,
    /// Anything the lexer could not make sense of.
    pub diagnostics: Vec<Diagnostic>,
    /// The doc comments.
    pub docs: Vec<DocComment>,
}

/// Which way a doc comment points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DocKind {
    /// `///`, documenting the declaration that follows.
    Outer,
    /// `//!`, documenting the unit.
    Inner,
}

/// One line of documentation: a `///` or `//!` comment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DocComment {
    /// Which way it points.
    pub kind: DocKind,
    /// The text after the marker.
    pub text: String,
    /// The whole comment.
    pub span: Span,
    /// Where the next token starts, in the same file; `None` when no token
    /// follows.
    pub anchor: Option<u32>,
}

impl Lexed {
    /// Whether tokenisation produced any error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }

    /// Just the tokens, dropping spans. For tests and dumps.
    pub fn kinds(&self) -> Vec<Token> {
        self.tokens.iter().map(|(t, _)| *t).collect()
    }
}

/// Tokenises one source file.
///
/// ```
/// use topiq::intern::Interner;
/// use topiq::lex::{lex, Token, Keyword};
/// use topiq::source::Spliced;
/// use topiq::span::SourceId;
///
/// let mut interner = Interner::new();
/// let spliced = Spliced::from_text("let x = 1;");
/// let out = lex(SourceId(0), &spliced, &mut interner);
/// assert!(!out.has_errors());
/// assert_eq!(out.kinds()[0], Token::Kw(Keyword::Let));
/// ```
pub fn lex(source: SourceId, spliced: &Spliced, interner: &Interner) -> Lexed {
    Lexer {
        text: spliced.text(),
        bytes: spliced.text().as_bytes(),
        pos: 0,
        source,
        spliced,
        interner,
        tokens: Vec::new(),
        diagnostics: Vec::new(),
        docs: Vec::new(),
        unanchored: 0,
    }
    .run()
}

struct Lexer<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    source: SourceId,
    spliced: &'a Spliced,
    interner: &'a Interner,
    tokens: Vec<(Token, Span)>,
    diagnostics: Vec<Diagnostic>,
    docs: Vec<DocComment>,
    // the first doc comment still waiting for a token to anchor it
    unanchored: usize,
}

impl<'a> Lexer<'a> {
    fn run(mut self) -> Lexed {
        loop {
            self.skip_trivia();
            let start = self.pos;
            if start >= self.bytes.len() {
                break;
            }
            match self.scan_one() {
                Some(tok) => {
                    let span = self.span_from(start);
                    for doc in &mut self.docs[self.unanchored..] {
                        doc.anchor = Some(span.start);
                    }
                    self.unanchored = self.docs.len();
                    self.tokens.push((tok, span));
                }
                // `scan_one` already reported; make sure we advance so the
                // loop cannot spin on an unrecognised byte
                None if self.pos == start => {
                    self.next_char();
                }
                None => {}
            }
        }
        Lexed {
            tokens: self.tokens,
            diagnostics: self.diagnostics,
            docs: self.docs,
        }
    }

    ////////////////////
    // CURSOR HELPERS //
    ////////////////////

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, n: usize) -> Option<u8> {
        self.bytes.get(self.pos + n).copied()
    }

    fn starts_with(&self, s: &str) -> bool {
        self.text[self.pos..].starts_with(s)
    }

    /// Advances one whole UTF-8 scalar, so the cursor never lands
    /// mid-sequence, and returns it.
    fn next_char(&mut self) -> Option<char> {
        let c = self.text[self.pos..].chars().next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Advances past every byte that satisfies `f`.
    fn eat_while(&mut self, f: impl Fn(u8) -> bool) {
        while self.peek().is_some_and(&f) {
            self.pos += 1;
        }
    }

    fn span_from(&self, start: usize) -> Span {
        self.spliced
            .span(self.source, start as u32, self.pos as u32)
    }

    fn error(&mut self, start: usize, message: impl Into<String>) {
        let span = self.span_from(start);
        self.diagnostics
            .push(Diagnostic::new(Code::Es02).with_message(message).at(span));
    }

    ////////////
    // TRIVIA //
    ////////////

    /// Consumes whitespace and comments.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.pos += 1,
                Some(b'/') if self.peek_at(1) == Some(b'/') => self.skip_line_comment(),
                Some(b'/') if self.peek_at(1) == Some(b'*') => self.skip_block_comment(),
                _ => break,
            }
        }
    }

    fn skip_line_comment(&mut self) {
        // `////` and longer are ordinary comments, such as a box's rules
        let start = self.pos;
        let kind = match (self.peek_at(2), self.peek_at(3)) {
            (Some(b'/'), next) if next != Some(b'/') => Some(DocKind::Outer),
            (Some(b'!'), _) => Some(DocKind::Inner),
            _ => None,
        };
        // a line comment runs to, but not through, its newline
        while self.peek().is_some_and(|c| c != b'\n') {
            self.next_char();
        }
        if let Some(kind) = kind {
            let text = self.text[start + 3..self.pos].trim_end_matches('\r');
            self.docs.push(DocComment {
                kind,
                text: text.to_owned(),
                span: self.span_from(start),
                anchor: None,
            });
        }
    }

    fn skip_block_comment(&mut self) {
        // block comments do not nest: the first `*/` ends one, as in C
        let start = self.pos;
        self.pos += 2;
        loop {
            if self.pos >= self.bytes.len() {
                self.error(start, "unterminated block comment");
                return;
            }
            if self.peek() == Some(b'*') && self.peek_at(1) == Some(b'/') {
                self.pos += 2;
                return;
            }
            // a newline inside a block comment vanishes with it
            self.next_char();
        }
    }

    ////////////
    // TOKENS //
    ////////////

    fn scan_one(&mut self) -> Option<Token> {
        let c = self.peek()?;
        match c {
            b'r' if matches!(self.peek_at(1), Some(b'"')) => self.scan_raw_string(0),
            b'r' if self.peek_at(1) == Some(b'#') && self.peek_at(2) == Some(b'"') => {
                self.scan_raw_string(1)
            }
            b'$' => self.scan_sigil_name(true),
            b'@' => self.scan_sigil_name(false),
            b'"' => self.scan_string(),
            b'\'' => self.scan_char(),
            b'0'..=b'9' => self.scan_number(),
            _ if is_ident_start(c) => Some(self.scan_word()),
            _ => self.scan_punct(),
        }
    }

    /// `$name`, naming the method behind an operator, or `@name`, a
    /// compiler-supplied macro. The symbol holds the name without its sigil.
    fn scan_sigil_name(&mut self, dollar: bool) -> Option<Token> {
        let start = self.pos;
        self.pos += 1;
        let name_start = self.pos;
        self.eat_while(is_ident_continue);
        if self.pos == name_start {
            let sigil = if dollar { '$' } else { '@' };
            self.error(start, format!("`{sigil}` must be followed by a name"));
            return None;
        }
        let sym = self.interner.intern_late(&self.text[name_start..self.pos]);
        Some(if dollar {
            Token::OpName(sym)
        } else {
            Token::MacroName(sym)
        })
    }

    /// An identifier or keyword: `[A-Za-z_][A-Za-z0-9_]*`.
    fn scan_word(&mut self) -> Token {
        let start = self.pos;
        self.eat_while(is_ident_continue);
        let word = &self.text[start..self.pos];
        match word {
            "true" => Token::Bool(true),
            "false" => Token::Bool(false),
            _ => match Keyword::from_text(word) {
                Some(k) => Token::Kw(k),
                None => Token::Ident(self.interner.intern_late(word)),
            },
        }
    }

    /// An integer or floating literal.
    ///
    /// The value is **not** computed and **not** range-checked here. A 24-qubit
    /// ket such as `|111111111111111111111111>` has an `INT` in the middle
    /// whose value exceeds `u64::MAX`, and rejecting it at lex time would make
    /// every wide ket unlexable. Range checking belongs to constant evaluation.
    fn scan_number(&mut self) -> Option<Token> {
        let start = self.pos;

        // after `.`, digits name a tuple's element: `t.0.1` is two field
        // accesses, not `t.` followed by the number `0.1`
        if matches!(self.tokens.last(), Some((Token::Punct(Punct::Dot), _))) {
            self.eat_while(|c| c.is_ascii_digit());
            let raw = self.interner.intern_late(&self.text[start..self.pos]);
            return Some(Token::Int {
                raw,
                base: IntBase::Decimal,
                suffix: None,
            });
        }

        // prefixed bases. a leading zero alone is decimal, so `0110` is 110
        if self.peek() == Some(b'0')
            && let Some(base) = match self.peek_at(1) {
                Some(b'x' | b'X') => Some(IntBase::Hex),
                Some(b'b' | b'B') => Some(IntBase::Binary),
                Some(b'o' | b'O') => Some(IntBase::Octal),
                _ => None,
            } {
                self.pos += 2;
                let digits_start = self.pos;
                self.eat_while(|c| c == b'_' || is_digit_in(c, base));
                if self.pos == digits_start {
                    self.error(start, format!("`{}` needs at least one digit", base.prefix()));
                    return None;
                }
                let raw = self.interner.intern_late(&self.text[digits_start..self.pos]);
                let suffix = self.scan_int_suffix();
                return Some(Token::Int { raw, base, suffix });
            }

        // decimal digits
        self.eat_while(|c| c.is_ascii_digit() || c == b'_');

        // a fraction only if a digit actually follows the dot, so `1..2` is a
        // range and `1.` is an integer followed by `.`
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.peek_at(1), Some(c) if c.is_ascii_digit()) {
            is_float = true;
            self.pos += 1;
            self.eat_while(|c| c.is_ascii_digit() || c == b'_');
        }

        // an exponent
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let mark = self.pos;
            let mut probe = self.pos + 1;
            if matches!(self.bytes.get(probe), Some(b'+' | b'-')) {
                probe += 1;
            }
            if matches!(self.bytes.get(probe), Some(c) if c.is_ascii_digit()) {
                is_float = true;
                self.pos = probe;
                self.eat_while(|c| c.is_ascii_digit());
            } else {
                // not an exponent after all: `1e` is `1` then the word `e`
                self.pos = mark;
            }
        }

        let raw = self.interner.intern_late(&self.text[start..self.pos]);
        if is_float {
            let suffix = if self.eat_word("f32") {
                Some(FloatSuffix::F32)
            } else if self.eat_word("f64") {
                Some(FloatSuffix::F64)
            } else {
                None
            };
            Some(Token::Float { raw, suffix })
        } else {
            let suffix = self.scan_int_suffix();
            Some(Token::Int {
                raw,
                base: IntBase::Decimal,
                suffix,
            })
        }
    }

    fn scan_int_suffix(&mut self) -> Option<IntSuffix> {
        // IntSuffix::ALL is ordered longest-first, so `usize` is tried before
        // any shorter spelling could match a prefix of it
        IntSuffix::ALL.iter().find(|&&s| self.eat_word(s.text())).copied()
    }

    /// Consumes `word` only if it is present *and* not followed by more
    /// identifier characters, so `1u8` takes the suffix but `1u8x` does not.
    fn eat_word(&mut self, word: &str) -> bool {
        if !self.starts_with(word) {
            return false;
        }
        let after = self.pos + word.len();
        if matches!(self.bytes.get(after), Some(&c) if is_ident_continue(c)) {
            return false;
        }
        self.pos = after;
        true
    }

    /// A character literal.
    fn scan_char(&mut self) -> Option<Token> {
        let start = self.pos;
        self.pos += 1;
        let c = match self.peek() {
            None => {
                self.error(start, "unterminated character literal");
                return None;
            }
            Some(b'\'') => {
                self.pos += 1;
                self.error(start, "empty character literal");
                return None;
            }
            Some(b'\\') => self.scan_escape(start)?,
            Some(_) => self.next_char()?,
        };
        if self.peek() != Some(b'\'') {
            self.error(start, "character literal must hold exactly one character");
            return None;
        }
        self.pos += 1;
        Some(Token::Char(c))
    }

    /// A string literal.
    fn scan_string(&mut self) -> Option<Token> {
        let start = self.pos;
        self.pos += 1;
        let mut value = String::new();
        loop {
            match self.peek() {
                None | Some(b'\n') => {
                    self.error(start, "unterminated string literal");
                    return None;
                }
                Some(b'"') => {
                    self.pos += 1;
                    let sym = self.interner.intern_late(&value);
                    return Some(Token::Str {
                        value: sym,
                        kind: StrKind::Normal,
                    });
                }
                Some(b'\\') => value.push(self.scan_escape(start)?),
                Some(_) => value.push(self.next_char()?),
            }
        }
    }

    /// A raw string.
    ///
    /// There are exactly two forms, `r"..."` and `r#"..."#`, so `hashes` is 0
    /// or 1; more would accept programs the language rejects.
    fn scan_raw_string(&mut self, hashes: usize) -> Option<Token> {
        let start = self.pos;
        self.pos += 1 + hashes + 1; // `r`, the hashes, the opening quote
        let body_start = self.pos;
        let closing = format!("\"{}", "#".repeat(hashes));
        loop {
            if self.pos >= self.bytes.len() {
                self.error(start, "unterminated raw string literal");
                return None;
            }
            if self.starts_with(&closing) {
                let sym = self.interner.intern_late(&self.text[body_start..self.pos]);
                self.pos += closing.len();
                return Some(Token::Str {
                    value: sym,
                    kind: StrKind::Raw,
                });
            }
            self.next_char();
        }
    }

    /// One escape sequence, such as `\n` or `\u{1F600}`.
    fn scan_escape(&mut self, literal_start: usize) -> Option<char> {
        let esc_start = self.pos;
        self.pos += 1; // the backslash
        let Some(c) = self.peek() else {
            self.error(literal_start, "unterminated escape sequence");
            return None;
        };
        self.pos += 1;
        match c {
            b'n' => Some('\n'),
            b't' => Some('\t'),
            b'r' => Some('\r'),
            b'0' => Some('\0'),
            b'\\' => Some('\\'),
            b'\'' => Some('\''),
            b'"' => Some('"'),
            b'u' => self.scan_unicode_escape(esc_start),
            other => {
                self.error(
                    esc_start,
                    format!("unknown escape sequence `\\{}`", other as char),
                );
                None
            }
        }
    }

    /// A `\u{...}` escape, whose body is one or more hex digits.
    fn scan_unicode_escape(&mut self, esc_start: usize) -> Option<char> {
        if self.peek() != Some(b'{') {
            self.error(esc_start, "`\\u` must be followed by `{`");
            return None;
        }
        self.pos += 1;
        let digits_start = self.pos;
        self.eat_while(|c| c.is_ascii_hexdigit());
        let digits = &self.text[digits_start..self.pos];
        if self.peek() != Some(b'}') {
            self.error(esc_start, "unterminated `\\u{...}` escape");
            return None;
        }
        self.pos += 1;
        if digits.is_empty() {
            self.error(esc_start, "`\\u{...}` needs at least one hex digit");
            return None;
        }
        let Ok(value) = u32::from_str_radix(digits, 16) else {
            self.error(esc_start, "`\\u{...}` value is too large");
            return None;
        };
        // a `char` is a Unicode scalar value, so surrogates and anything
        // above U+10FFFF are not characters at all
        let c = char::from_u32(value);
        if c.is_none() {
            self.error(esc_start, format!("U+{value:04X} is not a Unicode scalar value"));
        }
        c
    }

    /// A punctuator.
    fn scan_punct(&mut self) -> Option<Token> {
        for &p in Punct::LEXABLE {
            if self.starts_with(p.text()) {
                self.pos += p.text().len();
                return Some(Token::Punct(p));
            }
        }
        let start = self.pos;
        let ch = self.next_char().unwrap_or('\u{fffd}');
        self.error(start, format!("unexpected character `{ch}`"));
        None
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_continue(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn is_digit_in(c: u8, base: IntBase) -> bool {
    match base {
        IntBase::Decimal => c.is_ascii_digit(),
        IntBase::Hex => c.is_ascii_hexdigit(),
        IntBase::Binary => matches!(c, b'0' | b'1'),
        IntBase::Octal => (b'0'..=b'7').contains(&c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Harness {
        interner: Interner,
    }

    impl Harness {
        fn new() -> Harness {
            Harness {
                interner: Interner::new(),
            }
        }

        fn lex(&mut self, src: &str) -> Lexed {
            let spliced = Spliced::from_text(src);
            lex(SourceId(0), &spliced, &self.interner)
        }

        fn kinds(&mut self, src: &str) -> Vec<Token> {
            let out = self.lex(src);
            assert!(
                !out.has_errors(),
                "unexpected diagnostics for {src:?}: {:?}",
                out.diagnostics
                    .iter()
                    .map(|d| d.message.clone())
                    .collect::<Vec<_>>()
            );
            out.kinds()
        }

        fn one(&mut self, src: &str) -> Token {
            let k = self.kinds(src);
            assert_eq!(k.len(), 1, "expected one token from {src:?}, got {k:?}");
            k[0]
        }

        fn sym(&mut self, text: &str) -> crate::intern::Symbol {
            self.interner.intern_late(text)
        }
    }

    fn p(p: Punct) -> Token {
        Token::Punct(p)
    }

    #[test]
    fn lexes_a_simple_binding() {
        let mut h = Harness::new();
        let k = h.kinds("let x = 1;");
        let x = h.sym("x");
        let one = h.sym("1");
        assert_eq!(
            k,
            vec![
                Token::Kw(Keyword::Let),
                Token::Ident(x),
                p(Punct::Eq),
                Token::Int {
                    raw: one,
                    base: IntBase::Decimal,
                    suffix: None
                },
                p(Punct::Semi),
            ]
        );
    }

    #[test]
    fn keywords_are_recognized_but_lookalikes_are_not() {
        let mut h = Harness::new();
        assert_eq!(h.one("measure"), Token::Kw(Keyword::Measure));
        assert_eq!(h.one("qmap"), Token::Kw(Keyword::Qmap));
        let s = h.sym("measured");
        assert_eq!(h.one("measured"), Token::Ident(s));
        let s = h.sym("pub");
        assert_eq!(h.one("pub"), Token::Ident(s), "`pub` is not reserved");
    }

    #[test]
    fn booleans_are_constants_not_keywords() {
        let mut h = Harness::new();
        assert_eq!(h.one("true"), Token::Bool(true));
        assert_eq!(h.one("false"), Token::Bool(false));
    }

    #[test]
    fn operator_method_and_builtin_macro_names() {
        let mut h = Harness::new();
        let add = h.sym("add");
        assert_eq!(h.one("$add"), Token::OpName(add));
        let ti = h.sym("typeinfo");
        assert_eq!(h.one("@typeinfo"), Token::MacroName(ti));
    }

    #[test]
    fn a_bare_sigil_is_an_error() {
        let mut h = Harness::new();
        assert!(h.lex("$ ").has_errors());
        assert!(h.lex("@").has_errors());
    }

    /////////////
    // NUMBERS //
    /////////////

    #[test]
    fn integer_bases_and_raw_text() {
        let mut h = Harness::new();
        for (src, raw, base) in [
            ("42", "42", IntBase::Decimal),
            ("1_000_000", "1_000_000", IntBase::Decimal),
            ("0xFF", "FF", IntBase::Hex),
            ("0b1010", "1010", IntBase::Binary),
            ("0o755", "755", IntBase::Octal),
        ] {
            let expect_raw = h.sym(raw);
            assert_eq!(
                h.one(src),
                Token::Int {
                    raw: expect_raw,
                    base,
                    suffix: None
                },
                "for {src}"
            );
        }
    }

    #[test]
    fn a_leading_zero_is_decimal_not_octal() {
        // octal is spelt `0o755`, so `0110` is one hundred and ten. keeping
        // the raw digits matters for kets: `|0110>` must still read as the
        // four bits that were written
        let mut h = Harness::new();
        let raw = h.sym("0110");
        assert_eq!(
            h.one("0110"),
            Token::Int {
                raw,
                base: IntBase::Decimal,
                suffix: None
            }
        );
    }

    #[test]
    fn integer_suffixes() {
        let mut h = Harness::new();
        let raw = h.sym("42");
        assert_eq!(
            h.one("42u8"),
            Token::Int {
                raw,
                base: IntBase::Decimal,
                suffix: Some(IntSuffix::U8)
            }
        );
        let raw3 = h.sym("3");
        assert_eq!(
            h.one("3usize"),
            Token::Int {
                raw: raw3,
                base: IntBase::Decimal,
                suffix: Some(IntSuffix::Usize)
            },
            "`usize` must win over any shorter spelling"
        );
        let raw7 = h.sym("7");
        assert_eq!(
            h.one("7i64"),
            Token::Int {
                raw: raw7,
                base: IntBase::Decimal,
                suffix: Some(IntSuffix::I64)
            }
        );
    }

    #[test]
    fn a_wide_ket_lexes_because_integers_are_not_range_checked() {
        // this is the regression that raw-text INT exists for. twenty-four
        // ones is a conforming ket at the maximum conductor, and its value is
        // about 1.1e23, far past u64::MAX
        let mut h = Harness::new();
        let digits = "1".repeat(24);
        let src = format!("|{digits}>");
        let out = h.lex(&src);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let raw = h.sym(&digits);
        assert_eq!(
            out.kinds(),
            vec![
                p(Punct::Or),
                Token::Int {
                    raw,
                    base: IntBase::Decimal,
                    suffix: None
                },
                p(Punct::Gt),
            ]
        );
    }

    #[test]
    fn floats_need_a_digit_after_the_point() {
        let mut h = Harness::new();
        let raw = h.sym("1.0");
        assert_eq!(
            h.one("1.0"),
            Token::Float {
                raw,
                suffix: None
            }
        );
        let raw = h.sym("2.5e-3");
        assert_eq!(
            h.one("2.5e-3"),
            Token::Float {
                raw,
                suffix: None
            }
        );
        let raw = h.sym("1.0");
        assert_eq!(
            h.one("1.0f32"),
            Token::Float {
                raw,
                suffix: Some(FloatSuffix::F32)
            }
        );
    }

    #[test]
    fn a_range_is_not_a_float() {
        // `1..2` must be three tokens, or every `for i in 0..n` breaks
        let mut h = Harness::new();
        let k = h.kinds("1..2");
        assert_eq!(k.len(), 3);
        assert!(matches!(k[0], Token::Int { .. }));
        assert_eq!(k[1], p(Punct::DotDot));
        assert!(matches!(k[2], Token::Int { .. }));

        let k = h.kinds("0..=4");
        assert_eq!(k[1], p(Punct::DotDotEq));
    }

    #[test]
    fn a_trailing_dot_is_not_a_float() {
        let mut h = Harness::new();
        let k = h.kinds("1.len()");
        assert!(matches!(k[0], Token::Int { .. }));
        assert_eq!(k[1], p(Punct::Dot));
    }

    #[test]
    fn digits_after_a_dot_are_a_position_not_a_fraction() {
        let mut h = Harness::new();
        let k = h.kinds("t.0.1");
        assert_eq!(k[1], p(Punct::Dot));
        assert!(matches!(k[2], Token::Int { .. }));
        assert_eq!(k[3], p(Punct::Dot));
        assert!(matches!(k[4], Token::Int { .. }));
    }

    #[test]
    fn a_bare_e_is_not_an_exponent() {
        let mut h = Harness::new();
        let k = h.kinds("1e");
        assert_eq!(k.len(), 2, "{k:?}");
        assert!(matches!(k[0], Token::Int { .. }));
        assert!(matches!(k[1], Token::Ident(_)));
    }

    #[test]
    fn a_prefix_with_no_digits_is_an_error() {
        let mut h = Harness::new();
        assert!(h.lex("0x").has_errors());
        assert!(h.lex("0b;").has_errors());
    }

    ///////////////////////
    // THE `>` INVERSION //
    ///////////////////////

    #[test]
    fn greater_than_is_never_glued_by_the_lexer() {
        let mut h = Harness::new();
        assert_eq!(h.kinds(">>"), vec![p(Punct::Gt), p(Punct::Gt)]);
        assert_eq!(h.kinds(">="), vec![p(Punct::Gt), p(Punct::Eq)]);
        assert_eq!(
            h.kinds(">>="),
            vec![p(Punct::Gt), p(Punct::Gt), p(Punct::Eq)]
        );
    }

    #[test]
    fn nested_generics_close_as_two_tokens() {
        let mut h = Harness::new();
        let k = h.kinds("Vec<Vec<T>>");
        assert_eq!(k[k.len() - 2], p(Punct::Gt));
        assert_eq!(k[k.len() - 1], p(Punct::Gt));
    }

    #[test]
    fn a_generic_type_followed_by_assignment_lexes_correctly() {
        // `let v: Vec<i32>=x;`: a maximal-munch `>=` would ruin this
        let mut h = Harness::new();
        let k = h.kinds("Vec<i32>=x");
        assert_eq!(k[3], p(Punct::Gt));
        assert_eq!(k[4], p(Punct::Eq));
    }

    #[test]
    fn a_generic_type_followed_by_an_arrow_lexes_correctly() {
        // `qmap Oracle: Foo<T> -> ...` and the no-space form
        let mut h = Harness::new();
        let k = h.kinds("Foo<T>->x");
        assert_eq!(k[3], p(Punct::Gt));
        assert_eq!(k[4], p(Punct::Arrow));
    }

    #[test]
    fn less_than_still_munches_maximally() {
        let mut h = Harness::new();
        assert_eq!(h.kinds("<<"), vec![p(Punct::Shl)]);
        assert_eq!(h.kinds("<<="), vec![p(Punct::ShlEq)]);
        assert_eq!(h.kinds("<="), vec![p(Punct::Le)]);
    }

    #[test]
    fn the_tensor_operator_munches_before_multiplication() {
        let mut h = Harness::new();
        assert_eq!(h.kinds("**"), vec![p(Punct::StarStar)]);
        assert_eq!(h.kinds("* *"), vec![p(Punct::Star), p(Punct::Star)]);
    }

    #[test]
    fn increments_munch_before_the_sign() {
        let mut h = Harness::new();
        assert_eq!(h.kinds("++"), vec![p(Punct::PlusPlus)]);
        assert_eq!(h.kinds("--"), vec![p(Punct::MinusMinus)]);
        assert_eq!(h.kinds("+="), vec![p(Punct::PlusEq)]);
        // a double negation is spelled apart
        assert_eq!(h.kinds("- -"), vec![p(Punct::Minus), p(Punct::Minus)]);
        assert_eq!(h.kinds("---"), vec![p(Punct::MinusMinus), p(Punct::Minus)]);
    }

    ////////////////////////////
    // STRINGS AND CHARACTERS //
    ////////////////////////////

    #[test]
    fn strings_are_unescaped() {
        let mut h = Harness::new();
        let want = h.sym("a\nb\tc");
        assert_eq!(
            h.one(r#""a\nb\tc""#),
            Token::Str {
                value: want,
                kind: StrKind::Normal
            }
        );
    }

    #[test]
    fn unicode_escapes_decode_to_scalars() {
        let mut h = Harness::new();
        assert_eq!(h.one(r"'\u{1F600}'"), Token::Char('\u{1F600}'));
        assert_eq!(h.one(r"'\n'"), Token::Char('\n'));
        assert_eq!(h.one("'a'"), Token::Char('a'));
    }

    #[test]
    fn a_surrogate_escape_is_rejected() {
        // a `char` is a Unicode scalar value, and D800 is a surrogate
        let mut h = Harness::new();
        assert!(h.lex(r"'\u{D800}'").has_errors());
        assert!(h.lex(r"'\u{110000}'").has_errors());
    }

    #[test]
    fn raw_strings_keep_their_backslashes() {
        let mut h = Harness::new();
        let want = h.sym(r"C:\path\n");
        assert_eq!(
            h.one(r#"r"C:\path\n""#),
            Token::Str {
                value: want,
                kind: StrKind::Raw
            }
        );
    }

    #[test]
    fn hashed_raw_strings_can_hold_a_quote() {
        let mut h = Harness::new();
        let want = h.sym(r#"say "hi""#);
        assert_eq!(
            h.one(r##"r#"say "hi""#"##),
            Token::Str {
                value: want,
                kind: StrKind::Raw
            }
        );
    }

    #[test]
    fn unterminated_literals_are_reported_not_ignored() {
        let mut h = Harness::new();
        assert!(h.lex("\"oops").has_errors());
        assert!(h.lex("\"oops\nmore\"").has_errors(), "a string is one line");
        assert!(h.lex("r\"oops").has_errors());
        assert!(h.lex("'").has_errors());
        assert!(h.lex("''").has_errors());
        assert!(h.lex(r"'\q'").has_errors());
    }

    ////////////
    // TRIVIA //
    ////////////

    #[test]
    fn comments_separate_but_do_not_appear() {
        let mut h = Harness::new();
        let k = h.kinds("let // a comment\n x /* block */ = 1;");
        assert_eq!(k[0], Token::Kw(Keyword::Let));
        assert!(matches!(k[1], Token::Ident(_)));
        assert_eq!(k[2], p(Punct::Eq));
    }

    #[test]
    fn block_comments_do_not_nest() {
        // block comments do not nest, so the first `*/` ends the comment
        let mut h = Harness::new();
        let k = h.kinds("/* outer /* inner */ x");
        assert_eq!(k.len(), 1);
        assert!(matches!(k[0], Token::Ident(_)));
    }

    #[test]
    fn doc_comments_are_kept_beside_the_tokens() {
        let mut h = Harness::new();
        let src = "//! the unit\n/// first\n///second\nfn f() {}";
        let out = h.lex(src);
        assert_eq!(out.kinds()[0], Token::Kw(Keyword::Fn), "a doc comment is not a token");
        let kinds: Vec<DocKind> = out.docs.iter().map(|d| d.kind).collect();
        assert_eq!(kinds, [DocKind::Inner, DocKind::Outer, DocKind::Outer]);
        let texts: Vec<&str> = out.docs.iter().map(|d| d.text.as_str()).collect();
        assert_eq!(texts, [" the unit", " first", "second"]);
        assert_eq!(&src[out.docs[1].span.range()], "/// first");
    }

    #[test]
    fn a_doc_comment_is_anchored_at_the_next_token() {
        let mut h = Harness::new();
        let src = "let a = 1;\n/// on b\n// plain\nlet b = 2;\n/// trailing";
        let out = h.lex(src);
        assert_eq!(out.docs.len(), 2, "a plain comment is not documentation");
        assert_eq!(out.docs[0].anchor, Some(src.find("let b").unwrap() as u32));
        assert_eq!(out.docs[1].anchor, None, "no token follows the last");
    }

    #[test]
    fn four_slashes_are_an_ordinary_comment() {
        let mut h = Harness::new();
        let out = h.lex("////////\n// BOX //\n////////\n//// four\nx");
        assert!(out.docs.is_empty(), "{:?}", out.docs);
        assert!(h.lex("///\nx").docs.len() == 1, "an empty doc line still counts");
    }

    #[test]
    fn a_doc_comment_drops_the_carriage_return() {
        let mut h = Harness::new();
        let out = h.lex("/// crlf\r\nx");
        assert_eq!(out.docs[0].text, " crlf");
    }

    #[test]
    fn an_unterminated_block_comment_is_reported() {
        let mut h = Harness::new();
        assert!(h.lex("/* forever").has_errors());
    }

    ///////////////////////////
    // THE LINE-INITIAL HASH //
    ///////////////////////////

    #[test]
    fn hash_is_a_token_at_the_start_of_a_line() {
        let mut h = Harness::new();
        let k = h.kinds("#unit classical");
        assert_eq!(k[0], p(Punct::Hash));
        assert_eq!(k[1], Token::Ident(h.sym("unit")));
    }

    #[test]
    fn leading_whitespace_still_counts_as_line_initial() {
        // a directive's `#` must be the first thing on its line, with only
        // whitespace allowed before it
        let mut h = Harness::new();
        assert_eq!(h.kinds("   \t#define X 1")[0], p(Punct::Hash));
        assert_eq!(h.kinds("a\n  #define X 1")[1], p(Punct::Hash));
    }

    #[test]
    fn hash_and_hash_hash_lex_anywhere() {
        // a macro body needs both, and a macro body is never
        // line-initial; the preprocessor decides what each one means
        let mut h = Harness::new();
        assert_eq!(h.kinds("a ## b")[1], p(Punct::HashHash));
        assert_eq!(h.kinds("# x")[0], p(Punct::Hash));
        // `##` munches maximally, and spaced hashes stay separate
        assert_eq!(h.kinds("# #"), vec![p(Punct::Hash), p(Punct::Hash)]);
    }

    #[test]
    fn a_hash_mid_line_is_a_token_not_an_error() {
        // whether it is line-initial is the preprocessor's question
        let mut h = Harness::new();
        let out = h.lex("let x = 1; # nope");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.kinds().iter().any(|t| t.is(Punct::Hash)));
    }

    #[test]
    fn a_line_comment_restores_line_initial_position() {
        // the newline ending a line comment is real trivia and survives
        let mut h = Harness::new();
        let k = h.kinds("// note\n#define X 1");
        assert_eq!(k[0], p(Punct::Hash));
    }

    #[test]
    fn a_hash_after_a_block_comment_is_still_lexed() {
        // phase 3 replaces the whole comment, newlines and all, with one space
        // whether the `#` opens a directive is decided later, from line
        // numbers, and here it does not: `a` precedes it on the same line
        let mut h = Harness::new();
        let out = h.lex("a /* \n */ #define X 1");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.kinds()[1], p(Punct::Hash));
    }

    ///////////
    // SPANS //
    ///////////

    #[test]
    fn spans_point_at_the_written_text() {
        let src = "let alpha = 1;";
        let mut h = Harness::new();
        let out = h.lex(src);
        let (_, span) = out.tokens[1];
        assert_eq!(&src[span.range()], "alpha");
    }

    #[test]
    fn spans_are_original_coordinates_across_a_splice() {
        let src = "let al\\\npha = 1;";
        let mut h = Harness::new();
        let spliced = Spliced::from_text(src);
        let out = lex(SourceId(0), &spliced, &h.interner);
        let (tok, span) = out.tokens[1];
        assert_eq!(tok, Token::Ident(h.sym("alpha")));
        assert_eq!(&src[span.range()], "al\\\npha");
    }

    #[test]
    fn adjacent_tokens_report_adjacent_spans() {
        // this is what the `>>` join and ket reconstruction rely on
        let mut h = Harness::new();
        let out = h.lex(">>");
        assert!(out.tokens[0].1.adjacent_to(out.tokens[1].1));

        let out = h.lex("> >");
        assert!(!out.tokens[0].1.adjacent_to(out.tokens[1].1));
    }

    #[test]
    fn multibyte_text_does_not_desynchronize_the_cursor() {
        let mut h = Harness::new();
        let src = "let x = \"\u{3c0}\u{1f642}\"; let y = 2;";
        let out = h.lex(src);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let (_, span) = *out.tokens.last().unwrap();
        assert_eq!(&src[span.range()], ";");
    }

    #[test]
    fn an_unknown_character_advances_the_cursor() {
        // a lexer that failed to advance here would spin forever
        let mut h = Harness::new();
        let out = h.lex("let \u{a3} x");
        assert!(out.has_errors());
        assert_eq!(out.tokens.len(), 2, "the tokens either side still appear");
    }

    #[test]
    fn an_empty_file_lexes_to_nothing() {
        let mut h = Harness::new();
        let out = h.lex("");
        assert!(out.tokens.is_empty());
        assert!(!out.has_errors());
    }

    #[test]
    fn a_file_of_only_trivia_lexes_to_nothing() {
        let mut h = Harness::new();
        let out = h.lex("  // just a note\n\n  /* and a block */  \n");
        assert!(out.tokens.is_empty());
        assert!(!out.has_errors());
    }

    #[test]
    fn lexes_a_complete_operator_signature() {
        // the signature of a quantum operator taking an oracle
        let mut h = Harness::new();
        let out = h.lex(
            "[entry]\n\
             [cover: fin{ |0>, |1> }]\n\
             [expect: kernel.outcomes = Guess]\n\
             fn deutsch(oracle: fn(*qubit, *qubit)) -> Guess { }",
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.tokens.iter().any(|(t, _)| *t == Token::Kw(Keyword::Fn)));
        // at this stage each ket is still three separate tokens
        let gts = out.kinds().iter().filter(|t| t.is(Punct::Gt)).count();
        assert_eq!(gts, 2);
    }
}
