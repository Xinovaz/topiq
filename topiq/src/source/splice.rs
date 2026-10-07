//! Translation phases 1 and 2: decoding the file, and joining lines that end
//! in a backslash.
//!
//! Phase 1 decodes the source as UTF-8; a file that is not well-formed UTF-8 is
//! ill-formed. Phase 2 deletes every backslash that immediately precedes a
//! newline, together with that newline.
//!
//! Splicing only ever *removes* bytes, so the mapping from an offset in the
//! spliced text back to an offset in the original file is monotone and is
//! captured by a short table of cuts. Every later phase works on the spliced
//! text but reports spans in original-file coordinates, so a diagnostic inside a
//! spliced line still underlines the bytes the author actually wrote.
//!
//! # Carriage returns
//!
//! CR-LF counts as one newline, so a backslash before it splices too, as in
//! every C-family implementation. Files checked out on Windows end their
//! lines that way.

use crate::span::{SourceId, Span};

/// One deletion performed by line splicing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cut {
    /// Offset in the *spliced* text at which the deleted bytes used to sit.
    spliced: u32,
    /// Total bytes deleted at or before this point.
    removed: u32,
}

/// The result of phases 1 and 2: spliced text plus the map back to the original.
#[derive(Clone, Debug)]
pub struct Spliced {
    /// The text with backslash-newline pairs removed.
    text: String,
    /// Cuts in increasing `spliced` order.
    cuts: Vec<Cut>,
    /// Length of the original, pre-splice text in bytes.
    original_len: u32,
    /// Original-coordinate offset at which each *logical* line begins.
    ///
    /// A logical line is what survives phase 2: the text either side of a
    /// backslash-newline is one line. The preprocessor numbers lines this
    /// way, since a directive occupies one logical line;
    /// [`SourceFile::line_col`](crate::source::SourceFile::line_col) reports
    /// physical lines, which is what a reader counts.
    logical_line_starts: Vec<u32>,
}

/// Why a source file could not be decoded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DecodeError {
    /// Byte offset of the first ill-formed sequence.
    pub offset: u32,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not well-formed UTF-8 (first bad byte at offset {})", self.offset)
    }
}

impl std::error::Error for DecodeError {}

impl DecodeError {
    /// The decode error an I/O error carries, if it is one: what
    /// [`crate::source::SourceMap::load`] gives for a file that is read but
    /// not well-formed UTF-8.
    pub fn of(e: &std::io::Error) -> Option<DecodeError> {
        e.get_ref().and_then(|inner| inner.downcast_ref::<DecodeError>()).copied()
    }

    /// `ES01`, for the file at `path`.
    pub fn diagnostic(self, path: &std::path::Path) -> crate::diag::Diagnostic {
        crate::diag::Diagnostic::new(crate::diag::Code::Es01)
            .with_message(format!(
                "{} is not well-formed UTF-8: the byte at offset {} begins no character",
                path.display(),
                self.offset
            ))
            .with_note("a source file is read as UTF-8 before anything else, and one that is not cannot be read at all")
            .with_help("save the file as UTF-8")
    }
}

impl Spliced {
    /// Runs phases 1 and 2 over raw file bytes.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are not well-formed UTF-8, naming
    /// the offset of the first ill-formed sequence.
    pub fn new(bytes: &[u8]) -> Result<Spliced, DecodeError> {
        let text = std::str::from_utf8(bytes).map_err(|e| DecodeError {
            offset: e.valid_up_to() as u32,
        })?;
        Ok(Self::from_text(text))
    }

    /// Runs phase 2 over text that is already known to be valid UTF-8.
    pub fn from_text(src: &str) -> Spliced {
        let original_len = src.len() as u32;
        let bytes = src.as_bytes();
        let mut text = String::with_capacity(src.len());
        let mut cuts = Vec::new();
        let mut removed = 0u32;
        let mut i = 0usize;
        let mut logical_line_starts = vec![0u32];

        while i < bytes.len() {
            if bytes[i] == b'\\' {
                // backslash-LF, or backslash-CR-LF
                let eaten = match bytes.get(i + 1) {
                    Some(b'\n') => 2,
                    Some(b'\r') if bytes.get(i + 2) == Some(&b'\n') => 3,
                    _ => 0,
                };
                if eaten > 0 {
                    removed += eaten as u32;
                    cuts.push(Cut {
                        spliced: text.len() as u32,
                        removed,
                    });
                    i += eaten;
                    continue;
                }
            }
            // copy one whole UTF-8 scalar so `text` stays valid at every step
            let width = utf8_width(bytes[i]);
            text.push_str(&src[i..i + width]);
            i += width;
            // a newline that survived splicing ends a logical line
            if bytes[i - width] == b'\n' {
                logical_line_starts.push(i as u32);
            }
        }

        Spliced {
            text,
            cuts,
            original_len,
            logical_line_starts,
        }
    }

    /// The spliced text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether any splice actually occurred.
    pub fn is_spliced(&self) -> bool {
        !self.cuts.is_empty()
    }

    /// Total bytes removed by splicing.
    pub fn removed_bytes(&self) -> u32 {
        self.cuts.last().map_or(0, |c| c.removed)
    }

    /// Maps an offset in the spliced text to the corresponding offset in the
    /// original file.
    ///
    /// A spliced offset that lands exactly on a cut maps to the position
    /// *after* the deleted bytes, which is where the continued text begins.
    pub fn to_original(&self, spliced: u32) -> u32 {
        // last cut whose position is at or before `spliced`
        let idx = self.cuts.partition_point(|c| c.spliced <= spliced);
        let removed = if idx == 0 { 0 } else { self.cuts[idx - 1].removed };
        (spliced + removed).min(self.original_len)
    }

    /// The 0-based *logical* line containing an original-coordinate offset.
    ///
    /// A preprocessing directive occupies one logical line, and phase 2 has
    /// already joined lines across a backslash-newline, so this is the
    /// numbering the preprocessor splits on. A physical line count would break
    /// `#define X 1 \` continued onto the next line, reading it as two
    /// directives.
    pub fn logical_line(&self, original_offset: u32) -> u32 {
        self.logical_line_starts
            .partition_point(|&s| s <= original_offset) as u32
            - 1
    }

    /// How many logical lines the file has. Always at least one.
    pub fn logical_line_count(&self) -> u32 {
        self.logical_line_starts.len() as u32
    }

    /// Maps a byte range in the spliced text to a [`Span`] in the original file.
    pub fn span(&self, source: SourceId, start: u32, end: u32) -> Span {
        Span::new(source, self.to_original(start), self.to_original(end))
    }
}

/// Length in bytes of the UTF-8 sequence beginning with `first`.
fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: SourceId = SourceId(0);

    #[test]
    fn rejects_ill_formed_utf8_and_names_the_offset() {
        // 0xFF is never a valid UTF-8 lead byte
        let err = Spliced::new(b"let x = \xFF;").unwrap_err();
        assert_eq!(err.offset, 8);
    }

    #[test]
    fn accepts_multibyte_scalars() {
        let src = "let s = \"\u{3c0}\u{1f642}\";";
        let s = Spliced::new(src.as_bytes()).unwrap();
        assert_eq!(s.text(), src);
        assert!(!s.is_spliced());
    }

    #[test]
    fn splices_backslash_newline() {
        let s = Spliced::from_text("ab\\\ncd");
        assert_eq!(s.text(), "abcd");
        assert!(s.is_spliced());
        assert_eq!(s.removed_bytes(), 2);
    }

    #[test]
    fn splices_backslash_crlf() {
        let s = Spliced::from_text("ab\\\r\ncd");
        assert_eq!(s.text(), "abcd");
        assert_eq!(s.removed_bytes(), 3);
    }

    #[test]
    fn leaves_a_lone_backslash_alone() {
        let s = Spliced::from_text("a\\b\nc");
        assert_eq!(s.text(), "a\\b\nc");
        assert!(!s.is_spliced());
    }

    #[test]
    fn leaves_a_backslash_before_a_bare_cr_alone() {
        // CR not followed by LF is not a newline for splicing purposes
        let s = Spliced::from_text("a\\\rb");
        assert_eq!(s.text(), "a\\\rb");
        assert!(!s.is_spliced());
    }

    #[test]
    fn maps_offsets_back_across_one_cut() {
        //  original: a b \ \n c d      spliced: a b c d
        //  index:    0 1 2  3 4 5               0 1 2 3
        let s = Spliced::from_text("ab\\\ncd");
        assert_eq!(s.to_original(0), 0); // 'a'
        assert_eq!(s.to_original(1), 1); // 'b'
        assert_eq!(s.to_original(2), 4); // 'c', past the deleted pair
        assert_eq!(s.to_original(3), 5); // 'd'
    }

    #[test]
    fn maps_offsets_back_across_several_cuts() {
        let s = Spliced::from_text("a\\\nb\\\nc\\\nd");
        assert_eq!(s.text(), "abcd");
        assert_eq!(s.to_original(0), 0);
        assert_eq!(s.to_original(1), 3);
        assert_eq!(s.to_original(2), 6);
        assert_eq!(s.to_original(3), 9);
    }

    #[test]
    fn spans_land_on_the_original_bytes() {
        let src = "let a\\\nb = 1;";
        let s = Spliced::from_text(src);
        assert_eq!(s.text(), "let ab = 1;");
        // the identifier is `ab` at spliced 4..6; in the original it straddles
        // the deleted pair, so it must cover the backslash and the newline too
        let span = s.span(F, 4, 6);
        assert_eq!(&src[span.range()], "a\\\nb");
    }

    #[test]
    fn identity_map_when_nothing_was_spliced() {
        let s = Spliced::from_text("let x = 1;");
        for i in 0..=10u32 {
            assert_eq!(s.to_original(i), i);
        }
    }

    #[test]
    fn splice_at_the_very_start_and_end() {
        let s = Spliced::from_text("\\\nx");
        assert_eq!(s.text(), "x");
        assert_eq!(s.to_original(0), 2);

        let t = Spliced::from_text("x\\\n");
        assert_eq!(t.text(), "x");
        // one past the end still maps inside the original file
        assert_eq!(t.to_original(1), 3);
    }

    #[test]
    fn logical_lines_ignore_a_spliced_newline() {
        // this is what makes `#define X 1 \` continued onto the next line one
        // directive rather than two
        let src = "#define X 1 \\\n  + 2\nlet y = 3;\n";
        let s = Spliced::from_text(src);
        assert_eq!(s.text(), "#define X 1   + 2\nlet y = 3;\n");
        assert_eq!(s.logical_line_count(), 3, "two lines of text plus the tail");

        // the `+` sits after the splice but on logical line 0
        let plus = src.find('+').unwrap() as u32;
        assert_eq!(s.logical_line(plus), 0);
        // `let` opens logical line 1
        let let_at = src.find("let").unwrap() as u32;
        assert_eq!(s.logical_line(let_at), 1);
    }

    #[test]
    fn logical_lines_count_ordinary_newlines() {
        let s = Spliced::from_text("a\nb\nc");
        assert_eq!(s.logical_line_count(), 3);
        assert_eq!(s.logical_line(0), 0);
        assert_eq!(s.logical_line(2), 1);
        assert_eq!(s.logical_line(4), 2);
    }

    #[test]
    fn logical_lines_handle_crlf() {
        let s = Spliced::from_text("a\r\nb");
        assert_eq!(s.logical_line_count(), 2);
        assert_eq!(s.logical_line(3), 1);
    }

    #[test]
    fn utf8_widths() {
        assert_eq!(utf8_width(b'a'), 1);
        assert_eq!(utf8_width(0xCF), 2); // start of pi
        assert_eq!(utf8_width(0xE2), 3);
        assert_eq!(utf8_width(0xF0), 4); // start of an emoji
    }
}
