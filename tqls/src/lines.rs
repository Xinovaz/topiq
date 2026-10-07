//! Between byte offsets into a file's text, which spans hold, and the line and
//! character an editor counts.
//!
//! An editor counts characters in UTF-16 code units unless it offers UTF-8,
//! which the server then prefers, since spans are byte offsets.

use lsp_types::Position;

/// How the client counts characters within a line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Encoding {
    /// Bytes.
    Utf8,
    /// UTF-16 code units, the protocol's default.
    Utf16,
}

/// A text's line starts.
pub struct Lines<'a> {
    text: &'a str,
    starts: Vec<usize>,
}

impl<'a> Lines<'a> {
    /// Indexes `text`. A line ends after `\n`, so `\r\n` counts once.
    pub fn new(text: &'a str) -> Lines<'a> {
        let mut starts = vec![0];
        starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
        Lines { text, starts }
    }

    /// The position of the byte `offset`.
    pub fn position(&self, offset: usize, enc: Encoding) -> Position {
        let mut offset = offset.min(self.text.len());
        while !self.text.is_char_boundary(offset) {
            offset -= 1;
        }
        let line = self.starts.partition_point(|&s| s <= offset) - 1;
        let before = &self.text[self.starts[line]..offset];
        let character = match enc {
            Encoding::Utf8 => before.len(),
            Encoding::Utf16 => before.encode_utf16().count(),
        };
        Position::new(line as u32, character as u32)
    }

    /// The byte offset of `pos`: past the end of its line, the line's end,
    /// and past the last line, the end of the text.
    pub fn offset(&self, pos: Position, enc: Encoding) -> usize {
        let Some(&start) = self.starts.get(pos.line as usize) else {
            return self.text.len();
        };
        let end = self.starts.get(pos.line as usize + 1).map_or(self.text.len(), |&e| e);
        let line = self.text[start..end].trim_end_matches(['\n', '\r']);
        let mut units = 0;
        for (i, c) in line.char_indices() {
            if units >= pos.character as usize {
                return start + i;
            }
            units += match enc {
                Encoding::Utf8 => c.len_utf8(),
                Encoding::Utf16 => c.len_utf16(),
            };
        }
        start + line.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_round_trip_through_offsets() {
        let text = "let a = 1;\r\nlet β = \"😀\";\nend";
        let lines = Lines::new(text);
        for enc in [Encoding::Utf8, Encoding::Utf16] {
            // a line's terminator has no position of its own
            for (offset, _) in text.char_indices().filter(|(_, c)| *c != '\n' && *c != '\r') {
                let pos = lines.position(offset, enc);
                assert_eq!(lines.offset(pos, enc), offset, "{offset} via {pos:?} in {enc:?}");
            }
        }
    }

    #[test]
    fn utf16_counts_a_surrogate_pair_as_two() {
        let text = "\"😀\"x";
        let lines = Lines::new(text);
        let x = text.find('x').unwrap();
        assert_eq!(lines.position(x, Encoding::Utf16), Position::new(0, 4));
        assert_eq!(lines.position(x, Encoding::Utf8), Position::new(0, 6));
    }

    #[test]
    fn a_position_past_the_line_is_its_end() {
        let lines = Lines::new("ab\r\ncd");
        assert_eq!(lines.offset(Position::new(0, 40), Encoding::Utf16), 2);
        assert_eq!(lines.offset(Position::new(9, 0), Encoding::Utf16), 6);
    }
}
