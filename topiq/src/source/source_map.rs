//! The table of source files a translation knows about.
//!
//! A [`SourceFile`] keeps both the original text and the phase-2 spliced text
//! (see [`crate::source::splice`]). The original is what diagnostics quote and
//! what line and column numbers are computed against; the spliced text is what
//! the lexer scans.
//!
//! A file's **unit name** is its stem, and it is what the predefined macro
//! `__UNIT__` expands to: `geometry.tq` is the unit `geometry`, the name an
//! `import geometry;` resolves against.

use std::path::{Path, PathBuf};

use crate::span::{SourceId, Span};
use super::splice::{DecodeError, Spliced};

/// What kind of document a file holds. The extension decides, because the three
/// have grammars of their own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FileKind {
    /// A `.tq` translation unit.
    Unit,
    /// A `.tcon` document: classical data in Topiq's own notation.
    Tcon,
    /// A `.quon` document: quantum states and the conventions they are
    /// stated against. The whole file is a quantum context, so kets need no
    /// introduction inside it.
    Quon,
    /// Anything else, including text read from stdin.
    Other,
}

impl FileKind {
    /// Classifies a path by extension.
    pub fn of_path(path: &Path) -> FileKind {
        match path.extension().and_then(|e| e.to_str()) {
            Some("tq") => FileKind::Unit,
            Some("tcon") => FileKind::Tcon,
            Some("quon") => FileKind::Quon,
            _ => FileKind::Other,
        }
    }
}

/// One decoded, spliced, line-indexed source file.
#[derive(Clone, Debug)]
pub struct SourceFile {
    id: SourceId,
    path: PathBuf,
    name: String,
    unit_name: String,
    kind: FileKind,
    original: String,
    spliced: Spliced,
    /// Byte offset of the start of each line in `original`. Always non-empty.
    line_starts: Vec<u32>,
}

impl SourceFile {
    /// This file's id within its map.
    pub fn id(&self) -> SourceId {
        self.id
    }

    /// The path as given.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The display name used in diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The unit name.
    pub fn unit_name(&self) -> &str {
        &self.unit_name
    }

    /// What kind of document this is.
    pub fn kind(&self) -> FileKind {
        self.kind
    }

    /// The original, unspliced text, which is what spans index into.
    pub fn original(&self) -> &str {
        &self.original
    }

    /// The phase-2 spliced text.
    pub fn spliced(&self) -> &Spliced {
        &self.spliced
    }

    /// The text a span covers.
    ///
    /// Returns `None` for a span from another file or one that runs past the
    /// end, rather than panicking, so a malformed span degrades a diagnostic
    /// instead of taking the compiler down.
    pub fn slice(&self, span: Span) -> Option<&str> {
        if span.source != self.id {
            return None;
        }
        self.original.get(span.range())
    }

    /// Number of lines. A file always has at least one.
    pub fn line_count(&self) -> u32 {
        self.line_starts.len() as u32
    }

    /// The 1-based line and column of a byte offset in the original text.
    ///
    /// The column counts Unicode scalar values rather than bytes, so a line
    /// containing multibyte characters reports the column a reader would count.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let line_idx = self.line_index(offset);
        let line_start = self.line_starts[line_idx as usize];
        let upto = self
            .original
            .get(line_start as usize..offset.min(self.original.len() as u32) as usize)
            .unwrap_or("");
        (line_idx + 1, upto.chars().count() as u32 + 1)
    }

    /// The 0-based index of the line containing `offset`.
    pub fn line_index(&self, offset: u32) -> u32 {
        // partition_point gives the count of starts <= offset, which is the
        // 1-based line number; a file always has a start at 0 so this is >= 1
        self.line_starts.partition_point(|&s| s <= offset) as u32 - 1
    }

    /// The text of a 0-based line.
    pub fn line_text(&self, line_idx: u32) -> Option<&str> {
        let start = *self.line_starts.get(line_idx as usize)? as usize;
        let end = self
            .line_starts
            .get(line_idx as usize + 1)
            .map_or(self.original.len(), |&s| s as usize);
        Some(self.original[start..end].trim_end_matches(['\n', '\r']))
    }
}

/// Every file participating in a translation.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    /// An empty map.
    pub fn new() -> SourceMap {
        SourceMap::default()
    }

    /// Decodes and splices raw bytes, adding them under `path`.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are not well-formed UTF-8.
    pub fn add(&mut self, path: impl Into<PathBuf>, bytes: &[u8]) -> Result<SourceId, DecodeError> {
        let text = std::str::from_utf8(bytes).map_err(|e| DecodeError {
            offset: e.valid_up_to() as u32,
        })?;
        Ok(self.add_text(path, text))
    }

    /// Adds text that is already known to be valid UTF-8.
    pub fn add_text(&mut self, path: impl Into<PathBuf>, text: &str) -> SourceId {
        let path = path.into();
        let id = SourceId(self.files.len() as u32);
        let unit_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("<anonymous>")
            .to_owned();
        let file = SourceFile {
            id,
            name: path.display().to_string(),
            kind: FileKind::of_path(&path),
            unit_name,
            path,
            line_starts: line_starts(text),
            spliced: Spliced::from_text(text),
            original: text.to_owned(),
        };
        self.files.push(file);
        id
    }

    /// Reads a file from disk, decodes and splices it.
    ///
    /// # Errors
    ///
    /// The I/O error when the file cannot be read; for ill-formed UTF-8, an
    /// [`std::io::ErrorKind::InvalidData`] error carrying the [`DecodeError`],
    /// which [`DecodeError::of`] finds.
    pub fn load(&mut self, path: impl Into<PathBuf>) -> std::io::Result<SourceId> {
        let path = path.into();
        let bytes = std::fs::read(&path)?;
        self.add(path, &bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Looks up a file.
    pub fn get(&self, id: SourceId) -> Option<&SourceFile> {
        self.files.get(id.index())
    }

    /// Looks up a file, panicking if the id is not from this map.
    pub fn file(&self, id: SourceId) -> &SourceFile {
        self.get(id)
            .unwrap_or_else(|| panic!("source id {id} is not in this map"))
    }

    /// Every file, in insertion order.
    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    /// Number of files.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether no files have been added.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The text a span covers.
    pub fn slice(&self, span: Span) -> Option<&str> {
        self.get(span.source)?.slice(span)
    }
}

/// Byte offsets at which each line begins. Always yields at least one entry.
fn line_starts(text: &str) -> Vec<u32> {
    let mut starts = vec![0u32];
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i as u32 + 1);
        }
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_with(text: &str) -> (SourceMap, SourceId) {
        let mut m = SourceMap::new();
        let id = m.add_text("demo.tq", text);
        (m, id)
    }

    #[test]
    fn unit_name_is_the_file_stem() {
        let mut m = SourceMap::new();
        let id = m.add_text("src/geometry.tq", "#unit classical\n");
        assert_eq!(m.file(id).unit_name(), "geometry");
        assert_eq!(m.file(id).kind(), FileKind::Unit);
    }

    #[test]
    fn extension_classifies_the_document_kind() {
        assert_eq!(FileKind::of_path(Path::new("a.tq")), FileKind::Unit);
        assert_eq!(FileKind::of_path(Path::new("a.tcon")), FileKind::Tcon);
        assert_eq!(FileKind::of_path(Path::new("a.quon")), FileKind::Quon);
        assert_eq!(FileKind::of_path(Path::new("a.txt")), FileKind::Other);
        assert_eq!(FileKind::of_path(Path::new("noext")), FileKind::Other);
    }

    #[test]
    fn line_and_column_are_one_based() {
        let (m, id) = map_with("abc\ndefg\nhi");
        let f = m.file(id);
        assert_eq!(f.line_col(0), (1, 1));
        assert_eq!(f.line_col(2), (1, 3));
        assert_eq!(f.line_col(4), (2, 1)); // 'd'
        assert_eq!(f.line_col(9), (3, 1)); // 'h'
        assert_eq!(f.line_count(), 3);
    }

    #[test]
    fn columns_count_scalars_not_bytes() {
        // two 2-byte scalars then an ASCII 'x'
        let (m, id) = map_with("\u{3c0}\u{3c0}x");
        let f = m.file(id);
        // 'x' is at byte offset 4 but column 3
        assert_eq!(f.line_col(4), (1, 3));
    }

    #[test]
    fn line_text_strips_the_terminator() {
        let (m, id) = map_with("one\r\ntwo\nthree");
        let f = m.file(id);
        assert_eq!(f.line_text(0), Some("one"));
        assert_eq!(f.line_text(1), Some("two"));
        assert_eq!(f.line_text(2), Some("three"));
        assert_eq!(f.line_text(3), None);
    }

    #[test]
    fn empty_file_still_has_one_line() {
        let (m, id) = map_with("");
        let f = m.file(id);
        assert_eq!(f.line_count(), 1);
        assert_eq!(f.line_col(0), (1, 1));
        assert_eq!(f.line_text(0), Some(""));
    }

    #[test]
    fn trailing_newline_opens_a_final_empty_line() {
        let (m, id) = map_with("a\n");
        assert_eq!(m.file(id).line_count(), 2);
    }

    #[test]
    fn slicing_respects_file_identity() {
        let mut m = SourceMap::new();
        let a = m.add_text("a.tq", "let x = 1;");
        let b = m.add_text("b.tq", "let y = 2;");
        assert_eq!(m.slice(Span::new(a, 4, 5)), Some("x"));
        assert_eq!(m.slice(Span::new(b, 4, 5)), Some("y"));
        // a span from `a` never reads out of `b`
        assert_eq!(m.file(b).slice(Span::new(a, 4, 5)), None);
    }

    #[test]
    fn out_of_range_spans_degrade_rather_than_panic() {
        let (m, id) = map_with("short");
        assert_eq!(m.slice(Span::new(id, 0, 999)), None);
    }

    #[test]
    fn both_texts_are_kept_when_a_line_is_spliced() {
        let (m, id) = map_with("let a\\\nb = 1;");
        let f = m.file(id);
        assert_eq!(f.spliced().text(), "let ab = 1;");
        assert!(f.original().contains('\\'));
    }

    #[test]
    fn ill_formed_utf8_is_rejected_on_add() {
        let mut m = SourceMap::new();
        assert!(m.add("bad.tq", b"\xFF").is_err());
        assert!(m.is_empty());
    }

    #[test]
    fn ids_are_handed_out_in_order() {
        let mut m = SourceMap::new();
        assert_eq!(m.add_text("a.tq", ""), SourceId(0));
        assert_eq!(m.add_text("b.tq", ""), SourceId(1));
        assert_eq!(m.len(), 2);
        assert!(m.get(SourceId(7)).is_none());
    }
}
