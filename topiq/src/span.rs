//! Byte spans into a source file, and the `Spanned<T>` pairing used throughout
//! the frontend.
//!
//! A [`Span`] carries its [`SourceId`] so that a diagnostic produced deep inside
//! the parser can be rendered against the right file without threading the file
//! identity separately. Offsets are byte offsets into the *original* file, not
//! into the text the lexer actually walked: line splicing deletes the backslash
//! and newline of a continued line, and [`crate::source::splice`] maps back, so
//! a diagnostic always underlines what the author wrote.
//!
//! This module sits at the crate root rather than inside [`crate::source`]
//! because everything depends on it and it depends on nothing: [`crate::diag`],
//! [`crate::lex`] and [`crate::ast`] all need spans without needing the source
//! map.
//!
//! # Two `Span` traits
//!
//! `chumsky::span::Span` and `ariadne::Span` are different traits that happen to
//! share a name, and [`Span`] implements both: chumsky's so it can be a
//! parser's span type, ariadne's so a label can be attached to it. Because both
//! traits declare `start` and `end`, call those through the struct fields rather
//! than as methods to avoid an ambiguity error.

use std::fmt;
use std::ops::Range;

/// Identifies one source file within a [`SourceMap`](crate::source::SourceMap).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SourceId(pub u32);

impl SourceId {
    /// The id used for synthetic text that has no file behind it (macro
    /// expansions built from `##` pastes, predefined macro bodies).
    pub const SYNTHETIC: SourceId = SourceId(u32::MAX);

    /// Whether this id refers to synthesised rather than user-written text.
    pub fn is_synthetic(self) -> bool {
        self == Self::SYNTHETIC
    }

    /// The index into the source map's file table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_synthetic() {
            f.write_str("<synthetic>")
        } else {
            write!(f, "#{}", self.0)
        }
    }
}

/// A half-open byte range `[start, end)` within one source file.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// The file this span points into.
    pub source: SourceId,
    /// Inclusive start byte offset.
    pub start: u32,
    /// Exclusive end byte offset.
    pub end: u32,
}

impl Span {
    /// Constructs a span. `start` must not exceed `end`.
    pub fn new(source: SourceId, start: u32, end: u32) -> Self {
        debug_assert!(start <= end, "span start {start} exceeds end {end}");
        Span { source, start, end }
    }

    /// An empty span at `offset`, used to point at a position rather than a
    /// range (an unexpected end of input, an insertion point).
    pub fn at(source: SourceId, offset: u32) -> Self {
        Span::new(source, offset, offset)
    }

    /// A synthetic span.
    pub fn synthetic() -> Self {
        Span::new(SourceId::SYNTHETIC, 0, 0)
    }

    /// Length in bytes.
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    /// Whether the span covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The smallest span covering both operands.
    ///
    /// Spans in different files cannot be joined; the left operand wins, which
    /// keeps the result pointing somewhere real rather than at a blend of two
    /// files. Callers that care use [`Span::same_file`] first.
    pub fn join(self, other: Span) -> Span {
        if self.source != other.source {
            return self;
        }
        Span::new(
            self.source,
            self.start.min(other.start),
            self.end.max(other.end),
        )
    }

    /// Whether two spans point into the same file.
    pub fn same_file(self, other: Span) -> bool {
        self.source == other.source
    }

    /// Whether `self` ends exactly where `other` begins, with nothing between.
    ///
    /// This is what lets a ket be reassembled from the three separate tokens
    /// `|`, `0`, `>` that the lexer produces. The lexer cannot tell a ket from
    /// a bitwise or followed by a comparison, so the quantum parser checks
    /// that the pieces were written touching: `|0>` is a ket and `| 0 >` is
    /// not.
    pub fn adjacent_to(self, other: Span) -> bool {
        self.source == other.source && self.end == other.start
    }

    /// The byte range.
    pub fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}..{}", self.source, self.start, self.end)
    }
}

impl fmt::Display for Span {
    /// Byte offsets, not line and column.
    ///
    /// Turning a span into a human position needs the source map, so that
    /// belongs to [`crate::diag::render::position`]. This exists because
    /// `chumsky`'s `Rich` error requires its span type to be `Display`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A value paired with the span of the source text it came from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Spanned<T> {
    /// The value.
    pub node: T,
    /// Where it was written.
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Pairs a value with a span.
    pub fn new(node: T, span: Span) -> Self {
        Spanned { node, span }
    }

    /// Applies `f` to the value, keeping the span.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }

    /// Borrows the value, keeping the span.
    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            node: &self.node,
            span: self.span,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: SourceId = SourceId(0);
    const G: SourceId = SourceId(1);

    #[test]
    fn len_and_emptiness() {
        assert_eq!(Span::new(F, 3, 9).len(), 6);
        assert!(Span::at(F, 4).is_empty());
        assert!(!Span::new(F, 4, 5).is_empty());
    }

    #[test]
    fn join_covers_both() {
        let j = Span::new(F, 2, 5).join(Span::new(F, 9, 12));
        assert_eq!(j, Span::new(F, 2, 12));
        // order does not matter
        assert_eq!(Span::new(F, 9, 12).join(Span::new(F, 2, 5)), j);
    }

    #[test]
    fn join_across_files_keeps_the_left_operand() {
        let a = Span::new(F, 2, 5);
        assert_eq!(a.join(Span::new(G, 100, 200)), a);
    }

    #[test]
    fn adjacency_is_the_ket_reconstruction_test() {
        // `|0>` written without spaces: three touching spans
        let bar = Span::new(F, 10, 11);
        let int = Span::new(F, 11, 12);
        let gt = Span::new(F, 12, 13);
        assert!(bar.adjacent_to(int));
        assert!(int.adjacent_to(gt));

        // `| 0 >` has trivia between, so the pieces are not adjacent
        let spaced_int = Span::new(F, 12, 13);
        assert!(!bar.adjacent_to(spaced_int));

        // adjacency never crosses a file boundary
        assert!(!bar.adjacent_to(Span::new(G, 11, 12)));
    }

    #[test]
    fn synthetic_ids_are_recognizable() {
        assert!(Span::synthetic().source.is_synthetic());
        assert!(!F.is_synthetic());
        assert_eq!(SourceId::SYNTHETIC.to_string(), "<synthetic>");
    }

    #[test]
    fn range_slices_the_original_text() {
        let text = "let x = 1;";
        assert_eq!(&text[Span::new(F, 4, 5).range()], "x");
    }

    #[test]
    fn spanned_map_preserves_the_span() {
        let s = Spanned::new(1u8, Span::new(F, 0, 1));
        let t = s.map(|n| n as u32 + 41);
        assert_eq!(t.node, 42);
        assert_eq!(t.span, s.span);
    }
}

impl chumsky::span::Span for Span {
    type Context = SourceId;
    type Offset = usize;

    /// Builds a span from a `chumsky` range, normalising the inverted one a
    /// zero-width match produces.
    ///
    /// Over a token slice whose tokens carry their own spans, `chumsky`
    /// computes the span of a match as *start of the next token* to *end of the
    /// last token consumed*. When a parser matches nothing and whitespace
    /// separates the two tokens, that range runs backwards: in
    /// `fn main() -> i32 { 0 }` an empty match before the `}` yields `21..20`.
    ///
    /// Such a range means "nothing was consumed here", so it becomes an empty
    /// span at the position the parser had reached. The inherent
    /// [`Span::new`] keeps its debug assertion, so a genuinely reversed span
    /// built by this crate's own code is still caught.
    fn new(context: Self::Context, range: std::ops::Range<Self::Offset>) -> Self {
        let start = range.start as u32;
        let end = (range.end as u32).max(start);
        Span::new(context, start, end)
    }

    fn context(&self) -> Self::Context {
        self.source
    }

    fn start(&self) -> Self::Offset {
        self.start as usize
    }

    fn end(&self) -> Self::Offset {
        self.end as usize
    }
}

impl ariadne::Span for Span {
    type SourceId = SourceId;

    fn source(&self) -> &Self::SourceId {
        &self.source
    }

    fn start(&self) -> usize {
        self.start as usize
    }

    fn end(&self) -> usize {
        self.end as usize
    }
}

#[cfg(test)]
mod trait_impl_tests {
    use super::*;

    #[test]
    fn implements_the_chumsky_span_trait() {
        use chumsky::span::Span as _;
        let s = <Span as chumsky::span::Span>::new(SourceId(3), 5..9);
        assert_eq!(s, Span::new(SourceId(3), 5, 9));
        assert_eq!(s.context(), SourceId(3));
        assert_eq!(chumsky::span::Span::start(&s), 5usize);
        assert_eq!(chumsky::span::Span::end(&s), 9usize);
    }

    #[test]
    fn implements_the_ariadne_span_trait() {
        let s = Span::new(SourceId(2), 4, 7);
        assert_eq!(*ariadne::Span::source(&s), SourceId(2));
        assert_eq!(ariadne::Span::start(&s), 4usize);
        assert_eq!(ariadne::Span::end(&s), 7usize);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn chumsky_union_agrees_with_join() {
        use chumsky::span::Span as _;
        let a = Span::new(SourceId(0), 2, 5);
        let b = Span::new(SourceId(0), 9, 12);
        assert_eq!(a.union(b), a.join(b));
    }
}
