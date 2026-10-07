//! The token type.
//!
//! A token is a keyword, an identifier, an operator-method name, a macro name,
//! a constant, a string, or a punctuator.
//!
//! [`Token`] is `Copy + PartialEq`, as the parser generator requires. All
//! text lives in the [`Interner`](crate::intern::Interner) as a
//! [`crate::intern::Symbol`], which keeps the type small enough to copy.
//!
//! # Three things the lexer does not produce
//!
//! Each would mean deciding at lex time without the context to decide.
//!
//! **A ket is never one token.** `|0>` is a basis state in a quantum context
//! and a bitwise or followed by a comparison elsewhere, so the lexer always
//! produces `|`, `0` and `>`, and the quantum parser reassembles a ket where
//! one can appear. The pieces must touch, so `| 0 >` is never a ket. See
//! [`crate::quon::ket`].
//!
//! **`>` is always lexed alone**, so `>>`, `>=` and `>>=` are never formed.
//! Nested generic argument lists close with `>>`, which the parser could not
//! split again; instead it joins touching `>` tokens where a shift or a
//! comparison is meant. This also keeps `let v: Vec<i32>=x;` from lexing a
//! `>=`. The `<` family needs no such rule, since a type cannot begin with
//! `<`.
//!
//! **Exact scalars are not tokens.** `i`, `isq2` and `w(k, N)` are also
//! ordinary identifiers (`i` above all, as a loop counter), so they are
//! recognised from identifiers: by the QUON parser in a quantum context, and
//! by analysis wherever the program declares nothing of that name.

use crate::intern::Symbol;

/// An enum whose variants each have one spelling, written `Variant = "text"`:
/// the enum, `text` giving the spelling, and `Display` writing it. With
/// `all` before it, also `ALL`, every variant in the order declared.
macro_rules! spelled {
    (all $(#[$m:meta])* pub enum $name:ident { $($(#[$d:meta])* $v:ident = $t:literal,)* }) => {
        spelled!($(#[$m])* pub enum $name { $($(#[$d])* $v = $t,)* });

        impl $name {
            /// Every one, in the order declared.
            pub const ALL: &'static [$name] = &[$($name::$v),*];
        }
    };
    ($(#[$m:meta])* pub enum $name:ident { $($(#[$d:meta])* $v:ident = $t:literal,)* }) => {
        $(#[$m])*
        pub enum $name {
            $($(#[$d])* $v,)*
        }

        impl $name {
            /// The spelling.
            pub fn text(self) -> &'static str {
                match self {
                    $($name::$v => $t,)*
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.text())
            }
        }
    };
}

spelled! {
    all
    /// A reserved word.
    ///
    /// `true` and `false` are not here: they are lexed as [`Token::Bool`],
    /// since every position that accepts them accepts a constant.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    pub enum Keyword {
        /// `as`: converts a value to another type.
        As = "as",
        /// `aux`: borrows an ancilla qubit, to be returned in the state it came
        /// in.
        Aux = "aux",
        /// `base`: declares a base type: a register together with the cover and
        /// gauge its judgements are stated against.
        Base = "base",
        /// `break`.
        Break = "break",
        /// `chain`: declares a sequence of judgement stages, whose accumulated
        /// phase can then be claimed.
        Chain = "chain",
        /// `const`: marks a binding that is never assigned, or, behind a
        /// reference, what the reference may only read.
        Const = "const",
        /// `constexpr`: marks a binding, parameter or result whose value is
        /// known during translation, which the compiler checks.
        Constexpr = "constexpr",
        /// `continue`.
        Continue = "continue",
        /// `cover`: declares a set of states that a judgement is made over.
        Cover = "cover",
        /// `else`.
        Else = "else",
        /// `enum`.
        Enum = "enum",
        /// `fn`.
        Fn = "fn",
        /// `for`.
        For = "for",
        /// `forget`: discards a value deliberately, in a way the judgement system
        /// can see and account for.
        Forget = "forget",
        /// `gauge`: declares the frame each point of a cover is measured against.
        Gauge = "gauge",
        /// `if`.
        If = "if",
        /// `impl`.
        Impl = "impl",
        /// `import`.
        Import = "import",
        /// `in`.
        In = "in",
        /// `let`.
        Let = "let",
        /// `lift`: promotes a measurement outcome to something the circuit
        /// generator can branch on.
        Lift = "lift",
        /// `locale`: groups members that share a container and its conventions.
        Locale = "locale",
        /// `loop`.
        Loop = "loop",
        /// `match`.
        Match = "match",
        /// `measure`: collapses quantum state to a classical outcome.
        Measure = "measure",
        /// `of`: restricts one type by another. Not commutative: `A of B` and
        /// `B of A` are different types.
        Of = "of",
        /// `persist`: storage that outlives the run that created it.
        Persist = "persist",
        /// `prep`: prepares a register in a named state.
        Prep = "prep",
        /// `qmap`: a table that can be looked up coherently, without collapsing
        /// the key.
        Qmap = "qmap",
        /// `query`: the coherent lookup itself, as opposed to a measurement.
        Query = "query",
        /// `replay`: prepares a second, independent copy of what an operator
        /// produced, by re-running its construction.
        Replay = "replay",
        /// `return`.
        Return = "return",
        /// `static`: confines a declaration to its own unit. The only linkage
        /// specifier the language has.
        Static = "static",
        /// `struct`.
        Struct = "struct",
        /// `type`: a type alias.
        Type = "type",
        /// `void`.
        Void = "void",
        /// `where`: a condition on a generic's arguments, checked when it is
        /// instantiated. Takes a constant expression, not a trait bound: Topiq has
        /// no traits.
        Where = "where",
        /// `while`.
        While = "while",
        /// `with`: introduces a closure's capture list.
        With = "with",
    }
}

impl Keyword {
    /// Looks a word up, returning `None` for an ordinary identifier.
    pub fn from_text(text: &str) -> Option<Keyword> {
        Keyword::ALL.iter().copied().find(|k| k.text() == text)
    }
}

spelled! {
    /// A punctuator.
    ///
    /// `>>`, `>=` and `>>=` are absent by design; see the module documentation.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    pub enum Punct {
        /// `**`: the tensor product. **Not** exponentiation: Topiq has no
        /// exponentiation operator, and reading this one as a power is the single
        /// easiest way to misread a quantum expression.
        StarStar = "**",
        /// `<<`.
        Shl = "<<",
        /// `<=`.
        Le = "<=",
        /// `==`.
        EqEq = "==",
        /// `!=`.
        Ne = "!=",
        /// `&&`: short-circuiting and, defined on `bool` alone and not
        /// overloadable.
        AndAnd = "&&",
        /// `||`: bool only, and not overloadable.
        OrOr = "||",
        /// `++`: increments a place, before or after reading it.
        PlusPlus = "++",
        /// `--`: decrements a place, before or after reading it. Lexed as one
        /// token, so a double negation is written `-(-x)`.
        MinusMinus = "--",
        /// `+=`.
        PlusEq = "+=",
        /// `-=`.
        MinusEq = "-=",
        /// `*=`.
        StarEq = "*=",
        /// `/=`.
        SlashEq = "/=",
        /// `%=`.
        PercentEq = "%=",
        /// `&=`.
        AndEq = "&=",
        /// `|=`.
        OrEq = "|=",
        /// `^=`.
        CaretEq = "^=",
        /// `<<=`.
        ShlEq = "<<=",
        /// `..=`: a range including its upper bound.
        DotDotEq = "..=",
        /// `..`: a half-open range.
        DotDot = "..",
        /// `->`.
        Arrow = "->",
        /// `=>`.
        FatArrow = "=>",
        /// `::`.
        ColonColon = "::",
        /// `+`.
        Plus = "+",
        /// `-`.
        Minus = "-",
        /// `*`.
        Star = "*",
        /// `/`.
        Slash = "/",
        /// `%`.
        Percent = "%",
        /// `!`.
        Bang = "!",
        /// `&`.
        And = "&",
        /// `|`.
        Or = "|",
        /// `^`.
        Caret = "^",
        /// `<`.
        Lt = "<",
        /// `>`: always alone; see the module documentation.
        Gt = ">",
        /// `=`.
        Eq = "=",
        /// `.`.
        Dot = ".",
        /// `:`.
        Colon = ":",
        /// `;`.
        Semi = ";",
        /// `,`.
        Comma = ",",
        /// `?`: returns early if the operand is an error or absent, and
        /// otherwise yields the value inside it.
        Question = "?",
        /// `(`.
        LParen = "(",
        /// `)`.
        RParen = ")",
        /// `[`: an array type, array literal or index suffix.
        LBracket = "[",
        /// `[` opening an annotation group. Produced by [`crate::mark`], which
        /// decides between the two readings of `[`, and never by the lexer, which
        /// lacks the context to tell them apart.
        LBracketAnnot = "[",
        /// `]`.
        RBracket = "]",
        /// `{`.
        LBrace = "{",
        /// `}`.
        RBrace = "}",
        /// `#`. At the start of a line it introduces a preprocessing directive;
        /// inside a macro body it turns the following parameter into a string
        /// literal. Which of the two it is depends on where it sits, so the lexer
        /// emits it wherever it appears and [`crate::pp`] decides from position.
        Hash = "#",
        /// `##`, which joins the tokens on either side of it into one. Only
        /// meaningful inside a macro body.
        HashHash = "##",
    }
}

impl Punct {
    /// The lexable punctuators.
    ///
    /// The order is the rule: a punctuator is matched by trying the longest
    /// spelling first, so `<<=` is one token rather than `<<` followed by `=`.
    ///
    /// [`Punct::LBracketAnnot`] is excluded: it is produced by the marking pass
    /// rather than by the lexer.
    pub const LEXABLE: &'static [Punct] = &[
        // three characters
        Punct::ShlEq,
        Punct::DotDotEq,
        // two characters
        Punct::StarStar,
        Punct::Shl,
        Punct::Le,
        Punct::EqEq,
        Punct::Ne,
        Punct::AndAnd,
        Punct::OrOr,
        Punct::PlusPlus,
        Punct::MinusMinus,
        Punct::PlusEq,
        Punct::MinusEq,
        Punct::StarEq,
        Punct::SlashEq,
        Punct::PercentEq,
        Punct::AndEq,
        Punct::OrEq,
        Punct::CaretEq,
        Punct::DotDot,
        Punct::Arrow,
        Punct::FatArrow,
        Punct::ColonColon,
        Punct::HashHash,
        // one character
        Punct::Plus,
        Punct::Minus,
        Punct::Star,
        Punct::Slash,
        Punct::Percent,
        Punct::Bang,
        Punct::And,
        Punct::Or,
        Punct::Caret,
        Punct::Lt,
        Punct::Gt,
        Punct::Eq,
        Punct::Dot,
        Punct::Colon,
        Punct::Semi,
        Punct::Comma,
        Punct::Question,
        Punct::LParen,
        Punct::RParen,
        Punct::LBracket,
        Punct::RBracket,
        Punct::LBrace,
        Punct::RBrace,
        Punct::Hash,
    ];

    /// Whether this punctuator opens a bracketing pair.
    pub fn opens(self) -> Option<Punct> {
        match self {
            Punct::LParen => Some(Punct::RParen),
            Punct::LBracket | Punct::LBracketAnnot => Some(Punct::RBracket),
            Punct::LBrace => Some(Punct::RBrace),
            _ => None,
        }
    }

    /// Whether this punctuator closes a bracketing pair.
    pub fn closes(self) -> bool {
        matches!(self, Punct::RParen | Punct::RBracket | Punct::RBrace)
    }
}

/// The base an integer literal was written in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum IntBase {
    /// `42`, `1_000_000`. A leading zero means nothing: octal is spelt
    /// `0o755`, so `0110` is one hundred and ten, not seventy-two.
    Decimal,
    /// `0xFF`.
    Hex,
    /// `0b1010`.
    Binary,
    /// `0o755`.
    Octal,
}

impl IntBase {
    /// The radix.
    pub fn radix(self) -> u32 {
        match self {
            IntBase::Decimal => 10,
            IntBase::Hex => 16,
            IntBase::Binary => 2,
            IntBase::Octal => 8,
        }
    }

    /// The prefix.
    pub fn prefix(self) -> &'static str {
        match self {
            IntBase::Decimal => "",
            IntBase::Hex => "0x",
            IntBase::Binary => "0b",
            IntBase::Octal => "0o",
        }
    }
}

/// An integer literal's type suffix (e.g. `1u8` or `-3i64`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[allow(missing_docs)]
pub enum IntSuffix {
    U8,
    U16,
    U32,
    U64,
    Usize,
    I8,
    I16,
    I32,
    I64,
    Isize,
}

impl IntSuffix {
    /// Every suffix, longest spelling first so that maximal munch does not stop
    /// early: `usize` must be tried before `u8` would ever match a prefix.
    pub const ALL: &'static [IntSuffix] = &[
        IntSuffix::Usize,
        IntSuffix::Isize,
        IntSuffix::U16,
        IntSuffix::U32,
        IntSuffix::U64,
        IntSuffix::I16,
        IntSuffix::I32,
        IntSuffix::I64,
        IntSuffix::U8,
        IntSuffix::I8,
    ];

    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            IntSuffix::U8 => "u8",
            IntSuffix::U16 => "u16",
            IntSuffix::U32 => "u32",
            IntSuffix::U64 => "u64",
            IntSuffix::Usize => "usize",
            IntSuffix::I8 => "i8",
            IntSuffix::I16 => "i16",
            IntSuffix::I32 => "i32",
            IntSuffix::I64 => "i64",
            IntSuffix::Isize => "isize",
        }
    }

    /// Whether the suffixed type is signed.
    pub fn is_signed(self) -> bool {
        self.text().starts_with('i')
    }
}

/// A floating literal's type suffix, as in `1.5f32`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[allow(missing_docs)]
pub enum FloatSuffix {
    F32,
    F64,
}

impl FloatSuffix {
    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            FloatSuffix::F32 => "f32",
            FloatSuffix::F64 => "f64",
        }
    }
}

/// How a string literal was written.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum StrKind {
    /// `"..."`, in which backslash escapes are processed.
    Normal,
    /// `r"..."` or `r#"..."#`; no escapes are processed.
    Raw,
}

/// One lexical token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Token {
    /// A reserved word.
    Kw(Keyword),
    /// An identifier: `[A-Za-z_][A-Za-z0-9_]*`.
    Ident(Symbol),
    /// `$name`, naming the method that implements an operator. The symbol
    /// holds the name without the `$`.
    OpName(Symbol),
    /// `@name`, a compiler-supplied macro. The symbol holds the name without
    /// the `@`.
    MacroName(Symbol),
    /// An integer literal. `raw` is the digits **as written**, without prefix
    /// or suffix and with `_` separators retained.
    Int {
        /// The digits as written.
        raw: Symbol,
        /// The base it was written in.
        base: IntBase,
        /// The type suffix.
        suffix: Option<IntSuffix>,
    },
    /// A floating literal. `raw` is the text without the suffix.
    Float {
        /// The literal as written.
        raw: Symbol,
        /// The type suffix.
        suffix: Option<FloatSuffix>,
    },
    /// A character literal.
    Char(char),
    /// `true` or `false`.
    Bool(bool),
    /// A string literal.
    Str {
        /// The contents.
        value: Symbol,
        /// How it was written.
        kind: StrKind,
    },
    /// A punctuator.
    Punct(Punct),
}

impl Token {
    /// A punctuator token.
    pub const fn p(p: Punct) -> Token {
        Token::Punct(p)
    }

    /// Whether this is the given punctuator.
    pub fn is(self, p: Punct) -> bool {
        self == Token::Punct(p)
    }

    /// Whether this is the given keyword.
    pub fn is_kw(self, k: Keyword) -> bool {
        self == Token::Kw(k)
    }

    /// The identifier symbol.
    pub fn ident(self) -> Option<Symbol> {
        match self {
            Token::Ident(s) => Some(s),
            _ => None,
        }
    }

    /// Whether this token can begin an item, used by
    /// the marking pass and by error recovery to resynchronise.
    pub fn starts_item(self) -> bool {
        matches!(
            self,
            Token::Kw(
                Keyword::Fn
                    | Keyword::Struct
                    | Keyword::Enum
                    | Keyword::Impl
                    | Keyword::Let
                    | Keyword::Type
                    | Keyword::Import
                    | Keyword::Cover
                    | Keyword::Gauge
                    | Keyword::Base
                    | Keyword::Locale
                    | Keyword::Chain
                    | Keyword::Qmap
                    | Keyword::Static
            )
        )
    }

    /// A short description for a diagnostic's "expected ..., found ..." line.
    pub fn describe(self) -> String {
        match self {
            Token::Kw(k) => format!("keyword `{k}`"),
            Token::Ident(_) => "identifier".to_owned(),
            Token::OpName(_) => "operator-method name".to_owned(),
            Token::MacroName(_) => "builtin macro".to_owned(),
            Token::Int { .. } => "integer literal".to_owned(),
            Token::Float { .. } => "floating literal".to_owned(),
            Token::Char(_) => "character literal".to_owned(),
            Token::Bool(b) => format!("`{b}`"),
            Token::Str { .. } => "string literal".to_owned(),
            Token::Punct(p) => format!("`{p}`"),
        }
    }
}

impl std::fmt::Display for Token {
    /// A description rather than a spelling.
    ///
    /// The text of an identifier or a literal lives in the interner, which a
    /// token does not carry, so this prints what [`Token::describe`] says.
    /// [`spell`] is the function that recovers the source text, and it takes
    /// the interner. `chumsky`'s `Rich` error requires this impl.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

/// The source spelling of a token.
///
/// An integer literal recovers its prefix, raw digits and suffix, a string
/// literal is re-escaped, and everything else has one spelling. This is what
/// `#` stringises, what `##` pastes, and what `tqc emit --stage=tokens`
/// prints.
pub fn spell(tok: Token, interner: &crate::intern::Interner) -> String {
    match tok {
        Token::Kw(k) => k.text().to_owned(),
        Token::Ident(s) => interner.resolve(s).to_owned(),
        Token::OpName(s) => format!("${}", interner.resolve(s)),
        Token::MacroName(s) => format!("@{}", interner.resolve(s)),
        Token::Int { raw, base, suffix } => format!(
            "{}{}{}",
            base.prefix(),
            interner.resolve(raw),
            suffix.map_or("", IntSuffix::text)
        ),
        Token::Float { raw, suffix } => format!(
            "{}{}",
            interner.resolve(raw),
            suffix.map_or("", FloatSuffix::text)
        ),
        Token::Char(c) => format!("'{}'", escape(&c.to_string())),
        Token::Bool(b) => b.to_string(),
        Token::Str { value, kind } => match kind {
            StrKind::Normal => format!("\"{}\"", escape(interner.resolve(value))),
            StrKind::Raw => format!("r\"{}\"", interner.resolve(value)),
        },
        Token::Punct(p) => p.text().to_owned(),
    }
}

/// Re-escapes text so it can be written back inside a string or character
/// literal.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\'' => out.push_str("\\'"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn there_are_thirty_nine_keywords_plus_true_and_false() {
        // there are 41 reserved words in all. `true` and `false` are lexed as
        // boolean constants rather than keywords, leaving 39 here
        assert_eq!(Keyword::ALL.len(), 39);
        assert!(Keyword::from_text("true").is_none());
        assert!(Keyword::from_text("false").is_none());
    }

    #[test]
    fn keyword_spellings_are_unique_and_round_trip() {
        let mut seen = HashSet::new();
        for &k in Keyword::ALL {
            assert!(seen.insert(k.text()), "duplicate keyword {k}");
            assert_eq!(Keyword::from_text(k.text()), Some(k));
        }
    }

    #[test]
    fn the_keyword_list_is_complete_and_alphabetical() {
        let expected = "as aux base break chain const constexpr continue cover else enum \
                        fn for forget gauge if impl import in let lift locale loop \
                        match measure of persist prep qmap query replay return \
                        static struct type void where while with";
        let actual: Vec<&str> = Keyword::ALL.iter().map(|k| k.text()).collect();
        assert_eq!(actual.join(" "), expected);
    }

    #[test]
    fn ordinary_words_are_not_keywords() {
        // these name constructs Topiq deliberately does not have (no
        // visibility modifiers, no traits, no inheritance, no manual
        // allocation), so the words stay available to programs
        for word in ["pub", "trait", "impl_for", "virtual", "new", "delete", "mut"] {
            assert!(Keyword::from_text(word).is_none(), "{word} is not reserved");
        }
    }

    #[test]
    fn greater_than_is_never_glued() {
        // two nested generic argument lists close with `>>`, which must come
        // apart again. splitting a token after lexing is not possible, so the
        // lexer never forms these in the first place
        for text in [">>", ">=", ">>="] {
            assert!(
                !Punct::LEXABLE.iter().any(|p| p.text() == text),
                "`{text}` must not be lexable"
            );
        }
        assert!(Punct::LEXABLE.contains(&Punct::Gt));
    }

    #[test]
    fn less_than_is_still_glued() {
        // a generic argument list never opens twice in a row, because a type
        // cannot begin with `<`, so `<<` needs no splitting
        for p in [Punct::Shl, Punct::ShlEq, Punct::Le, Punct::Lt] {
            assert!(Punct::LEXABLE.contains(&p), "{p} should be lexable");
        }
    }

    #[test]
    fn lexable_punctuators_are_ordered_longest_first() {
        // the ordering is what implements longest-match: the lexer walks this
        // list and takes the first spelling that fits
        let lens: Vec<usize> = Punct::LEXABLE.iter().map(|p| p.text().len()).collect();
        assert!(
            lens.windows(2).all(|w| w[0] >= w[1]),
            "LEXABLE is not sorted longest-first: {lens:?}"
        );
    }

    #[test]
    fn lexable_excludes_only_the_marked_bracket() {
        assert!(!Punct::LEXABLE.contains(&Punct::LBracketAnnot));
        // `#` and `##` are lexable everywhere. a macro body needs both, and a
        // macro body is not at the start of a line, so the lexer cannot make
        // the directive-or-operator decision (position does, later)
        assert!(Punct::LEXABLE.contains(&Punct::Hash));
        assert!(Punct::LEXABLE.contains(&Punct::HashHash));
    }

    #[test]
    fn lexable_punctuator_spellings_are_unique() {
        let mut seen = HashSet::new();
        for &p in Punct::LEXABLE {
            assert!(seen.insert(p.text()), "duplicate punctuator {p}");
        }
    }

    #[test]
    fn bracket_pairing_is_declared() {
        assert_eq!(Punct::LParen.opens(), Some(Punct::RParen));
        assert_eq!(Punct::LBracket.opens(), Some(Punct::RBracket));
        assert_eq!(Punct::LBracketAnnot.opens(), Some(Punct::RBracket));
        assert_eq!(Punct::LBrace.opens(), Some(Punct::RBrace));
        assert_eq!(Punct::Plus.opens(), None);
        assert!(Punct::RBrace.closes());
        assert!(!Punct::LBrace.closes());
    }

    #[test]
    fn integer_suffixes_are_ordered_so_maximal_munch_works() {
        // `usize` must be matched before `u8`-style prefixes are considered,
        // or `1usize` would lex as `1u` plus garbage
        let pos = |s: &str| IntSuffix::ALL.iter().position(|x| x.text() == s).unwrap();
        assert!(pos("usize") < pos("u8"));
        assert!(pos("isize") < pos("i8"));
        assert_eq!(IntSuffix::ALL.len(), 10);
    }

    #[test]
    fn integer_suffix_signedness() {
        assert!(IntSuffix::I8.is_signed());
        assert!(IntSuffix::Isize.is_signed());
        assert!(!IntSuffix::U64.is_signed());
    }

    #[test]
    fn bases_carry_their_radix_and_prefix() {
        assert_eq!(IntBase::Decimal.radix(), 10);
        assert_eq!(IntBase::Hex.radix(), 16);
        assert_eq!(IntBase::Binary.radix(), 2);
        assert_eq!(IntBase::Octal.radix(), 8);
        assert_eq!(IntBase::Octal.prefix(), "0o");
        assert_eq!(IntBase::Decimal.prefix(), "");
    }

    #[test]
    fn token_predicates() {
        let t = Token::p(Punct::Semi);
        assert!(t.is(Punct::Semi));
        assert!(!t.is(Punct::Comma));
        assert!(Token::Kw(Keyword::Fn).is_kw(Keyword::Fn));
        assert!(Token::Kw(Keyword::Struct).starts_item());
        assert!(!Token::Kw(Keyword::Return).starts_item());
        assert_eq!(Token::Ident(Symbol::EMPTY).ident(), Some(Symbol::EMPTY));
        assert_eq!(Token::Bool(true).ident(), None);
    }

    #[test]
    fn descriptions_read_well_in_a_diagnostic() {
        assert_eq!(Token::p(Punct::Semi).describe(), "`;`");
        assert_eq!(Token::Kw(Keyword::Fn).describe(), "keyword `fn`");
        assert_eq!(Token::Ident(Symbol::EMPTY).describe(), "identifier");
        assert_eq!(Token::Bool(false).describe(), "`false`");
    }

    #[test]
    fn tokens_are_copy_and_comparable() {
        let a = Token::Int {
            raw: Symbol::EMPTY,
            base: IntBase::Hex,
            suffix: Some(IntSuffix::U8),
        };
        let b = a;
        assert_eq!(a, b);
        assert_ne!(
            a,
            Token::Int {
                raw: Symbol::EMPTY,
                base: IntBase::Decimal,
                suffix: Some(IntSuffix::U8),
            }
        );
    }

    #[test]
    fn there_is_no_exact_token_class() {
        // `i`, `isq2` and `w(k, N)` stay identifiers, so `for i in 0..n` keeps
        // working; the quantum parser and the evaluator recognise them later
        assert_eq!(Token::Ident(Symbol::EMPTY).describe(), "identifier");
    }

    #[test]
    fn spelling_round_trips_the_tokens_that_can() {
        let mut i = crate::intern::Interner::new();
        let raw = i.intern("FF");
        assert_eq!(
            spell(
                Token::Int {
                    raw,
                    base: IntBase::Hex,
                    suffix: Some(IntSuffix::U8)
                },
                &i
            ),
            "0xFFu8"
        );
        let sep = i.intern("1_000");
        assert_eq!(
            spell(
                Token::Int {
                    raw: sep,
                    base: IntBase::Decimal,
                    suffix: None
                },
                &i
            ),
            "1_000",
            "digit separators survive; a ket is reassembled from raw text, so it \
             must be able to tell `|1_0>` from `|10>`"
        );
        let f = i.intern("2.5e-3");
        assert_eq!(
            spell(
                Token::Float {
                    raw: f,
                    suffix: Some(FloatSuffix::F64)
                },
                &i
            ),
            "2.5e-3f64"
        );
        let n = i.intern("x");
        assert_eq!(spell(Token::Ident(n), &i), "x");
        assert_eq!(spell(Token::OpName(n), &i), "$x");
        assert_eq!(spell(Token::MacroName(n), &i), "@x");
        assert_eq!(spell(Token::Kw(Keyword::Fn), &i), "fn");
        assert_eq!(spell(Token::p(Punct::StarStar), &i), "**");
        assert_eq!(spell(Token::Bool(true), &i), "true");
    }

    #[test]
    fn spelling_re_escapes_string_and_character_literals() {
        let mut i = crate::intern::Interner::new();
        let v = i.intern("a\nb\"c");
        assert_eq!(
            spell(
                Token::Str {
                    value: v,
                    kind: StrKind::Normal
                },
                &i
            ),
            "\"a\\nb\\\"c\""
        );
        assert_eq!(spell(Token::Char('\n'), &i), "'\\n'");
        assert_eq!(spell(Token::Char('a'), &i), "'a'");
        assert_eq!(spell(Token::Char('\\'), &i), "'\\\\'");
    }

    #[test]
    fn a_raw_string_keeps_its_contents_unescaped() {
        let mut i = crate::intern::Interner::new();
        let v = i.intern("C:\\path");
        assert_eq!(
            spell(
                Token::Str {
                    value: v,
                    kind: StrKind::Raw
                },
                &i
            ),
            "r\"C:\\path\""
        );
    }

    #[test]
    fn an_annotation_bracket_spells_like_an_ordinary_one() {
        // the marking pass changes how a `[` is read, never how it is spelt
        assert_eq!(Punct::LBracketAnnot.text(), Punct::LBracket.text());
    }

    #[test]
    fn the_tensor_operator_is_not_exponentiation() {
        // Topiq has no exponentiation operator at all. the name in this enum
        // should not tempt anyone into adding one here by mistake
        assert_eq!(Punct::StarStar.text(), "**");
    }
}
