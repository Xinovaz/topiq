//! Checking the program a file belongs to, and what an editor is told of it.
//!
//! A file is checked as the root of a program: the file, every unit it
//! imports, found as `tqc` finds them, and the library. Open documents stand
//! in for the files they edit, saved or not. The check stops after type
//! checking and, for a quantum unit, circuit lowering and judgement, which is
//! everything that can report a diagnostic without building.

use std::path::{Path, PathBuf};

use lsp_types::{
    Diagnostic as LspDiagnostic, DiagnosticRelatedInformation, DiagnosticSeverity, Location, NumberOrString,
    Range,
};
use topiq::diag::{Diagnostic, Severity};
use topiq::driver::program::{self, Loaded, Member, MemberKind, Overlay};
use topiq::driver::{Options, Outcome, Stage};
use topiq::span::{SourceId, Span};

use crate::lines::{Encoding, Lines};

/// A program.
pub struct Checked {
    /// Every unit of the program, and the session holding their text.
    pub loaded: Loaded,
    /// The file the program was checked from.
    pub root: PathBuf,
}

impl Checked {
    /// Checks the program `root` belongs to, taking each file `overlay`
    /// holds from there, and looking for imported units beside `root` and in
    /// `unit_path`.
    ///
    /// # Errors
    ///
    /// Why the program could not be loaded: a file that cannot be read, or a
    /// compiled unit whose metadata cannot be.
    pub fn new(root: &Path, unit_path: &[PathBuf], overlay: &Overlay) -> Result<Checked, String> {
        let options = Options {
            stage: Stage::Check,
            ..Options::default()
        };
        let loaded = program::load_with(&[root.to_owned()], unit_path, options, overlay).map_err(|e| e.to_string())?;
        Ok(Checked {
            loaded,
            root: root.to_owned(),
        })
    }

    /// The file at `path`.
    pub fn file(&self, path: &Path) -> Option<SourceId> {
        let want = key(path);
        let files = self.loaded.session.sources().files();
        files.iter().find(|f| key(f.path()) == want).map(|f| f.id())
    }

    /// The text of the file `id`.
    pub fn text(&self, id: SourceId) -> &str {
        self.loaded.session.sources().file(id).original()
    }

    /// The path of the file `id`, unless it is a library unit's, which is
    /// held in the compiler rather than on disk.
    pub fn path(&self, id: SourceId) -> Option<&Path> {
        let path = self.loaded.session.sources().get(id)?.path();
        (!path.starts_with("<library>")).then_some(path)
    }

    /// What checking the file `id` produced, if it is one of the program's
    /// source units.
    pub fn outcome(&self, id: SourceId) -> Option<&Outcome> {
        self.loaded.members.iter().find_map(|m| match &m.kind {
            MemberKind::Source(out) if out.source == Some(id) => Some(&**out),
            _ => None,
        })
    }

    /// The unit named `name`.
    pub fn member(&self, name: &str) -> Option<&Member> {
        self.loaded.members.iter().find(|m| m.name == name)
    }

    /// Where `span` is, as a file and a range in it, unless it is in no file
    /// on disk.
    pub fn location(&self, span: Span, enc: Encoding) -> Option<(PathBuf, Range)> {
        if span.source.is_synthetic() {
            return None;
        }
        let path = self.path(span.source)?.to_owned();
        Some((path, range(self.text(span.source), span, enc)))
    }

    /// Every diagnostic, by the file it is in. Each source file of the
    /// program that is on disk is listed, with no diagnostics if it has
    /// none, so that an editor clears what an earlier check showed.
    pub fn diagnostics(&self, enc: Encoding) -> Vec<(PathBuf, Vec<LspDiagnostic>)> {
        let mut out: Vec<(PathBuf, Vec<LspDiagnostic>)> = Vec::new();
        for m in &self.loaded.members {
            if let MemberKind::Source(o) = &m.kind
                && let Some(path) = o.source.and_then(|id| self.path(id))
            {
                out.push((path.to_owned(), Vec::new()));
            }
        }
        if !out.iter().any(|(p, _)| key(p) == key(&self.root)) {
            out.push((self.root.clone(), Vec::new()));
        }

        for d in self.loaded.all_diagnostics() {
            // one outside every file on disk is shown at the top of the root
            let (path, range) = match d.primary_span().and_then(|s| self.location(s, enc)) {
                Some(at) => at,
                None => (self.root.clone(), Range::default()),
            };
            let related = d
                .labels
                .iter()
                .filter(|l| !l.primary && !l.message.is_empty())
                .filter_map(|l| {
                    let (path, range) = self.location(l.span, enc)?;
                    Some(DiagnosticRelatedInformation {
                        location: Location::new(crate::url(&path)?, range),
                        message: l.message.clone(),
                    })
                })
                .collect::<Vec<_>>();
            let shown = LspDiagnostic {
                range,
                severity: Some(match d.severity {
                    Severity::Error => DiagnosticSeverity::ERROR,
                    Severity::Warning => DiagnosticSeverity::WARNING,
                }),
                code: Some(NumberOrString::String(d.code.id().to_owned())),
                source: Some("topiq".to_owned()),
                message: message(d),
                related_information: (!related.is_empty()).then_some(related),
                ..LspDiagnostic::default()
            };
            match out.iter_mut().find(|(p, _)| key(p) == key(&path)) {
                Some((_, list)) => list.push(shown),
                None => out.push((path, vec![shown])),
            }
        }
        out
    }
}

/// A diagnostic's text: its message, what its primary label says unless it
/// says only "here", which the underline already does, then its notes and
/// help, as the terminal shows them.
fn message(d: &Diagnostic) -> String {
    let mut text = d.message.clone();
    if let Some(label) = d.labels.iter().find(|l| l.primary)
        && !matches!(label.message.as_str(), "" | "here")
        && label.message != d.message
    {
        text.push('\n');
        text.push_str(&label.message);
    }
    for note in &d.notes {
        text.push_str("\nnote: ");
        text.push_str(note);
    }
    for help in &d.helps {
        text.push_str("\nhelp: ");
        text.push_str(help);
    }
    text
}

/// The range `span` covers in `text`.
pub fn range(text: &str, span: Span, enc: Encoding) -> Range {
    let lines = Lines::new(text);
    Range::new(lines.position(span.start as usize, enc), lines.position(span.end as usize, enc))
}

/// A path as compared with another: Windows paths are the same whatever
/// their case or separators.
pub fn key(path: &Path) -> String {
    let text = path.to_string_lossy();
    if cfg!(windows) {
        text.replace('/', "\\").to_lowercase()
    } else {
        text.into_owned()
    }
}
