//! String interning.
//!
//! Tokens carry identifier text, literal text and string contents. Keeping that
//! as `&'src str` would tie every token to the lifetime of one source file,
//! which breaks as soon as the preprocessor synthesises text that no file
//! contains (e.g. a `##` paste, a stringised argument, the body of `__FILE__`).
//! Interning to a [`Symbol`] solves that and buys two more things the frontend
//! needs:
//!
//! - **`Token` stays `Copy`.** chumsky's `ValueInput` requires `Token: Clone`,
//!   and `just`/`select!` require `PartialEq`; a `Copy` token makes the parser's
//!   generated code much smaller than one cloning a `String` per lookahead.
//! - **Comparison is one integer compare**, which matters because the parser
//!   compares tokens constantly.
//!
//! # Raw text is preserved verbatim
//!
//! Numeric literals are interned as *written*, not as parsed values. The
//! language admits `1_000_000`, `0xFF`, `0b1010` and `0o755`, and a ket has to
//! tell `|01>` from `|1>` (two qubits against one). A parsed value cannot: both
//! are the number one. See [`crate::lex::lexer`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

/// An interned string.
///
/// A `Symbol` is only meaningful against the [`Interner`] that produced it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Symbol {
    /// The symbol for the empty string.
    pub const EMPTY: Symbol = Symbol(0);

    /// The raw index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({})", self.0)
    }
}

/// A string table mapping text to [`Symbol`] and back.
///
/// ```
/// use topiq::intern::Interner;
///
/// let mut i = Interner::new();
/// let a = i.intern("qubit");
/// let b = i.intern("qubit");
/// assert_eq!(a, b);
/// assert_eq!(i.resolve(a), "qubit");
/// ```
///
/// # Interning through a shared reference
///
/// Every stage may intern, including those that hold the interner shared:
/// analysis makes strings of its own, such as a type's name for `@name_of`,
/// and expands macros, which lexes and parses text it made. So the table
/// sits behind a `RefCell`, and each string is leaked when it is first
/// interned, so that it can be lent out for as long as anyone likes while the
/// table itself is borrowed only for a moment. A compiler's strings last
/// until it exits, and each is kept once however often it is interned.
#[derive(Clone, Debug)]
pub struct Interner {
    table: RefCell<Table>,
}

#[derive(Clone, Debug, Default)]
struct Table {
    /// Text by symbol index.
    texts: Vec<&'static str>,
    /// Symbol by text.
    lookup: HashMap<&'static str, Symbol>,
}

impl Default for Interner {
    fn default() -> Self {
        Interner::new()
    }
}

impl Interner {
    /// A new interner.
    pub fn new() -> Interner {
        let i = Interner {
            table: RefCell::default(),
        };
        let empty = i.intern_late("");
        debug_assert_eq!(empty, Symbol::EMPTY);
        i
    }

    /// Interns `text`, returning its symbol. Interning the same text twice
    /// returns the same symbol.
    pub fn intern(&mut self, text: &str) -> Symbol {
        self.intern_late(text)
    }

    /// Interns `text` where only a shared reference is at hand, as analysis
    /// has. The same text gives the same symbol however it was interned.
    pub fn intern_late(&self, text: &str) -> Symbol {
        if let Some(sym) = self.get(text) {
            return sym;
        }
        let mut t = self.table.borrow_mut();
        let sym = Symbol(t.texts.len() as u32);
        let leaked: &'static str = Box::leak(text.to_owned().into_boxed_str());
        t.texts.push(leaked);
        t.lookup.insert(leaked, sym);
        sym
    }

    /// The text of a symbol.
    ///
    /// # Panics
    ///
    /// Panics if the symbol did not come from this interner. That is a
    /// programming error rather than a source-program error, so it is not a
    /// diagnostic; use [`Interner::try_resolve`] where the origin is uncertain.
    pub fn resolve(&self, sym: Symbol) -> &str {
        self.try_resolve(sym)
            .unwrap_or_else(|| panic!("{sym:?} is not from this interner"))
    }

    /// The text of a symbol, or `None` if it is not from this interner.
    pub fn try_resolve(&self, sym: Symbol) -> Option<&str> {
        self.table.borrow().texts.get(sym.index()).copied()
    }

    /// The symbol for `text` if it has already been interned, without
    /// interning it.
    pub fn get(&self, text: &str) -> Option<Symbol> {
        self.table.borrow().lookup.get(text).copied()
    }

    /// How many distinct strings are held.
    pub fn len(&self) -> usize {
        self.table.borrow().texts.len()
    }

    /// Whether only the pre-seeded empty string is held.
    pub fn is_empty(&self) -> bool {
        self.len() <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_interning_agrees_with_ordinary_interning() {
        let mut i = Interner::new();
        let early = i.intern("geo::P");
        assert_eq!(i.intern_late("geo::P"), early, "already there");
        let late = i.intern_late("[u8; 4]");
        assert_eq!(i.resolve(late), "[u8; 4]");
        assert_eq!(i.intern("[u8; 4]"), late, "found again by the ordinary path");
        assert_ne!(late, i.intern("other"));
    }

    #[test]
    fn interning_is_idempotent() {
        let mut i = Interner::new();
        let a = i.intern("measure");
        let b = i.intern("measure");
        let c = i.intern("measures");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn symbols_resolve_back_to_their_text() {
        let mut i = Interner::new();
        let s = i.intern("$tensor");
        assert_eq!(i.resolve(s), "$tensor");
    }

    #[test]
    fn the_empty_string_is_pre_seeded() {
        let mut i = Interner::new();
        assert_eq!(i.resolve(Symbol::EMPTY), "");
        assert_eq!(i.intern(""), Symbol::EMPTY);
        assert!(i.is_empty());
        assert_eq!(i.len(), 1);
    }

    #[test]
    fn get_does_not_intern() {
        let mut i = Interner::new();
        assert_eq!(i.get("qubit"), None);
        let before = i.len();
        assert_eq!(i.get("qubit"), None);
        assert_eq!(i.len(), before);
        let s = i.intern("qubit");
        assert_eq!(i.get("qubit"), Some(s));
    }

    #[test]
    fn a_foreign_symbol_resolves_to_none_rather_than_panicking() {
        let i = Interner::new();
        let mut j = Interner::new();
        let foreign = j.intern("elsewhere");
        assert_eq!(i.try_resolve(foreign), None);
    }

    #[test]
    #[should_panic(expected = "is not from this interner")]
    fn resolving_a_foreign_symbol_panics_loudly() {
        let i = Interner::new();
        let mut j = Interner::new();
        let foreign = j.intern("elsewhere");
        let _ = i.resolve(foreign);
    }

    #[test]
    fn literal_text_is_kept_exactly_as_written() {
        // a ket distinguishes |01> from |1> (two qubits against one) and
        // digit separators are allowed anywhere; both need the raw spelling
        // rather than a parsed value
        let mut i = Interner::new();
        for raw in ["01", "1", "1_000_000", "0xFF", "0b1010", "0o755", "42u8"] {
            let s = i.intern(raw);
            assert_eq!(i.resolve(s), raw);
        }
        assert_ne!(i.intern("01"), i.intern("1"));
    }

    #[test]
    fn symbols_are_copy_and_cheap_to_compare() {
        let mut i = Interner::new();
        let a = i.intern("x");
        let b = a; // copy, not a move
        assert_eq!(a, b);
        assert_eq!(a.index(), b.index());
    }

    #[test]
    fn indices_are_handed_out_in_order() {
        let mut i = Interner::new();
        assert_eq!(i.intern("").index(), 0);
        assert_eq!(i.intern("a").index(), 1);
        assert_eq!(i.intern("b").index(), 2);
        assert_eq!(i.intern("a").index(), 1);
    }

    #[test]
    fn a_very_long_string_round_trips() {
        // raw string literals and #embed payloads can be large
        let mut i = Interner::new();
        let long = "x".repeat(100_000);
        let s = i.intern(&long);
        assert_eq!(i.resolve(s).len(), 100_000);
    }
}
