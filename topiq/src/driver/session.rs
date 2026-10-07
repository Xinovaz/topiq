//! The state one translation needs: source files and the string table.
//!
//! A [`Session`] owns the two things every phase reaches for. Keeping them
//! together, and handing out the borrows the phases need through
//! [`Session::split_mut`], is what lets the preprocessor read the source map
//! while it interns into the string table.

use std::path::{Path, PathBuf};

use crate::intern::Interner;
use crate::source::SourceMap;
use crate::span::SourceId;

/// The state of one translation.
#[derive(Debug, Default)]
pub struct Session {
    sources: SourceMap,
    interner: Interner,
    /// The interfaces of the library units analysed in this session, so
    /// each is analysed once however many units use it.
    pub(super) library: std::collections::HashMap<String, crate::sema::Interface>,
}

impl Session {
    /// A session with no files.
    pub fn new() -> Session {
        Session::default()
    }

    /// Adds a file from memory.
    pub fn add(&mut self, path: impl Into<PathBuf>, text: &str) -> SourceId {
        self.sources.add_text(path, text)
    }

    /// Reads a file from disk.
    ///
    /// # Errors
    ///
    /// Propagates the I/O error, and reports ill-formed UTF-8 as
    /// [`std::io::ErrorKind::InvalidData`]: source must be valid UTF-8.
    pub fn load(&mut self, path: impl AsRef<Path>) -> std::io::Result<SourceId> {
        self.sources.load(path.as_ref().to_path_buf())
    }

    /// The source map.
    pub fn sources(&self) -> &SourceMap {
        &self.sources
    }

    /// The string table.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// The string table.
    pub fn interner_mut(&mut self) -> &mut Interner {
        &mut self.interner
    }

    /// Borrows the source map immutably and the string table mutably at once.
    ///
    /// The preprocessor needs exactly this: it reads logical line numbers and
    /// file names out of the source map while interning the text it synthesises
    /// for `#` and `##`. Two separate accessor calls would borrow the whole
    /// session twice.
    pub fn split_mut(&mut self) -> (&SourceMap, &mut Interner) {
        (&self.sources, &mut self.interner)
    }

    /// How many files the session holds.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether the session holds no files.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_session_is_empty() {
        let s = Session::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn adding_a_file_gives_it_an_id_and_a_unit_name() {
        let mut s = Session::new();
        let id = s.add("src/geometry.tq", "#unit classical\n");
        assert_eq!(s.len(), 1);
        assert_eq!(s.sources().file(id).unit_name(), "geometry");
    }

    #[test]
    fn the_split_borrow_gives_both_halves_at_once() {
        let mut s = Session::new();
        let id = s.add("demo.tq", "let x = 1;\n");
        let (sources, interner) = s.split_mut();
        // read from one while writing to the other, which is what phase 4 does
        let name = sources.file(id).unit_name().to_owned();
        let sym = interner.intern(&name);
        assert_eq!(interner.resolve(sym), "demo");
    }

    #[test]
    fn several_files_get_distinct_ids() {
        let mut s = Session::new();
        let a = s.add("a.tq", "");
        let b = s.add("b.tq", "");
        assert_ne!(a, b);
        assert_eq!(s.sources().file(a).unit_name(), "a");
        assert_eq!(s.sources().file(b).unit_name(), "b");
    }

    #[test]
    fn loading_a_missing_file_reports_the_io_error() {
        let mut s = Session::new();
        let err = s.load("no/such/file.tq").unwrap_err();
        assert!(matches!(
            err.kind(),
            std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidInput
        ));
    }

    #[test]
    fn the_interner_persists_across_files() {
        let mut s = Session::new();
        let sym = s.interner_mut().intern("shared");
        s.add("a.tq", "");
        assert_eq!(s.interner().resolve(sym), "shared");
    }
}
