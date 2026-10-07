//! Rendering diagnostics with `ariadne`.
//!
//! This is the only module that turns a [`Span`] into an underline, which keeps
//! the coordinate system in one place.
//!
//! # Byte indices
//!
//! `ariadne` interprets span offsets as **character** indices by default. Topiq
//! spans are byte offsets (source is UTF-8, and the lexer works in
//! bytes), so every report is configured with
//! [`ariadne::IndexType::Byte`]. Without it, any file
//! containing a non-ASCII character (a `π` in a comment is enough) underlines
//! the wrong columns.
//!
//! # Why the cache borrows
//!
//! `ariadne::Cache::fetch` takes `&mut self`, so a cache must be owned or
//! uniquely borrowed at render time. [`SourceCache`] borrows the
//! [`crate::source::SourceMap`] immutably and keeps its own
//! interior table of `ariadne::Source` values, so several reports can be written
//! against one map without cloning any file text.

use std::collections::HashMap;
use std::io::Write;

use ariadne::{Color, Config, IndexType, Label as ALabel, Report, ReportKind, Source};

use super::{Diagnostic, Severity};
use crate::source::SourceMap;
use crate::span::{SourceId, Span};

/// An `ariadne` cache over a [`SourceMap`].
///
/// `ariadne` wants a line-indexed `Source` per file and wants to build it once.
/// This type does that lazily, so rendering a diagnostic that touches one file
/// does not index every file in the map.
pub struct SourceCache<'a> {
    map: &'a SourceMap,
    cached: HashMap<SourceId, Source<&'a str>>,
}

impl<'a> SourceCache<'a> {
    /// Wraps a source map for rendering.
    pub fn new(map: &'a SourceMap) -> SourceCache<'a> {
        SourceCache {
            map,
            cached: HashMap::new(),
        }
    }
}

impl<'a> ariadne::Cache<SourceId> for SourceCache<'a> {
    type Storage = &'a str;

    fn fetch(&mut self, id: &SourceId) -> Result<&Source<&'a str>, impl std::fmt::Debug> {
        if !self.cached.contains_key(id) {
            let file = self
                .map
                .get(*id)
                .ok_or_else(|| format!("no source file with id {id}"))?;
            self.cached.insert(*id, Source::from(file.original()));
        }
        // just inserted above if it was missing
        Ok::<_, String>(&self.cached[id])
    }

    fn display<'b>(&self, id: &'b SourceId) -> Option<impl std::fmt::Display + 'b> {
        self.map.get(*id).map(|f| f.name().to_owned())
    }
}

/// How much decoration to apply.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// ANSI colour, for a terminal.
    Color,
    /// Plain text, for a file, a pipe, or a test assertion.
    Plain,
}

impl Style {
    /// Chooses by whether the destination is a terminal.
    pub fn for_terminal(is_terminal: bool) -> Style {
        if is_terminal {
            Style::Color
        } else {
            Style::Plain
        }
    }

    fn config(self) -> Config {
        Config::default()
            .with_index_type(IndexType::Byte)
            .with_color(self == Style::Color)
    }

    fn color(self, c: Color) -> Color {
        if self == Style::Color {
            c
        } else {
            Color::Primary
        }
    }
}

/// Writes one diagnostic as a full report.
///
/// A diagnostic with no labels still renders: it becomes a headline followed by
/// its notes and help, one per line, which is what link-time identifiers such
/// as `EL05` (no `main`) need, since they belong to no single position in any
/// file.
///
/// # Errors
///
/// Propagates any write error from `out`.
pub fn write(
    diag: &Diagnostic,
    sources: &SourceMap,
    style: Style,
    out: &mut impl Write,
) -> std::io::Result<()> {
    let Some(anchor) = diag.primary_span() else {
        writeln!(out, "{diag}")?;
        for note in &diag.notes {
            writeln!(out, "  = note: {note}")?;
        }
        for help in &diag.helps {
            writeln!(out, "  = help: {help}")?;
        }
        return Ok(());
    };

    let kind = match diag.severity {
        Severity::Error => ReportKind::Error,
        Severity::Warning => ReportKind::Warning,
    };

    let mut report = Report::build(kind, anchor)
        .with_config(style.config())
        .with_code(diag.code.id())
        .with_message(&diag.message);

    for label in &diag.labels {
        let color = style.color(if label.primary {
            Color::Red
        } else {
            Color::Blue
        });
        // a label with no text of its own underlines the span and stops
        // there. attaching an empty message instead would draw the connecting
        // arrow and then point at nothing
        let mut alabel = ALabel::new(label.span)
            .with_color(color)
            .with_order(if label.primary { 0 } else { 1 });
        if !label.message.is_empty() {
            alabel = alabel.with_message(&label.message);
        }
        report = report.with_label(alabel);
    }
    for note in &diag.notes {
        report = report.with_note(note);
    }
    for help in &diag.helps {
        report = report.with_help(help);
    }

    report.finish().write(SourceCache::new(sources), out)
}

/// Writes every diagnostic in order.
///
/// # Errors
///
/// Propagates any write error from `out`.
pub fn write_all(
    diags: &[Diagnostic],
    sources: &SourceMap,
    style: Style,
    out: &mut impl Write,
) -> std::io::Result<()> {
    for d in diags {
        write(d, sources, style, out)?;
    }
    Ok(())
}

/// Renders one diagnostic to a plain-text `String`.
///
/// For tests; never emits ANSI escapes.
pub fn to_string(diag: &Diagnostic, sources: &SourceMap) -> String {
    let mut buf = Vec::new();
    // writing to a Vec cannot fail; if ariadne ever does, fall back to the
    // one-line form rather than losing the diagnostic entirely
    match write(diag, sources, Style::Plain, &mut buf) {
        Ok(()) => String::from_utf8_lossy(&buf).into_owned(),
        Err(_) => diag.to_string(),
    }
}

/// The summary line a driver prints after translating a unit.
///
/// Returns `None` when there is nothing to summarise.
pub fn summary(errors: usize, warnings: usize) -> Option<String> {
    match (errors, warnings) {
        (0, 0) => None,
        (0, w) => Some(format!("{} warning{}", w, plural(w))),
        (e, 0) => Some(format!("{} error{}", e, plural(e))),
        (e, w) => Some(format!(
            "{} error{}, {} warning{}",
            e,
            plural(e),
            w,
            plural(w)
        )),
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Reports a span for a caller that only wants a position, not a picture.
///
/// The format is the one a run-time abort prints, `UNIT:LINE`,
/// extended with a column because a translation diagnostic has one.
pub fn position(span: Span, sources: &SourceMap) -> String {
    match sources.get(span.source) {
        Some(f) => {
            let (line, col) = f.line_col(span.start);
            format!("{}:{}:{}", f.name(), line, col)
        }
        None => "<unknown>".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::{Code, Diagnostic, Label};

    fn sources(text: &str) -> (SourceMap, SourceId) {
        let mut m = SourceMap::new();
        let id = m.add_text("demo.tq", text);
        (m, id)
    }

    #[test]
    fn a_report_names_the_identifier_and_underlines_the_span() {
        let (map, id) = sources("let x = 1;\nlet y = 2;\n");
        let d = Diagnostic::new(Code::Ec03)
            .at(Span::new(id, 4, 5))
            .with_help("drop `const` from its type");
        let out = to_string(&d, &map);

        assert!(out.contains("EC03"), "{out}");
        assert!(out.contains("assignment to a place of const type"), "{out}");
        assert!(out.contains("demo.tq"), "{out}");
        assert!(out.contains("drop `const` from its type"), "{out}");
    }

    #[test]
    fn plain_style_emits_no_ansi_escapes() {
        let (map, id) = sources("let x = 1;\n");
        let d = Diagnostic::new(Code::Eu01).at(Span::new(id, 0, 3));
        let out = to_string(&d, &map);
        assert!(!out.contains('\u{1b}'), "unexpected ANSI escape in {out:?}");
    }

    #[test]
    fn byte_indices_survive_a_multibyte_line() {
        // this is the regression the IndexType::Byte configuration exists for
        // the identifier `y` sits at byte offset 14 but character offset 12,
        // because the comment holds two 2-byte scalars
        let text = "let x = 1; // \u{3c0}\u{3c0}\nlet y = 2;\n";
        let (map, id) = sources(text);
        let y = text.find("y").unwrap() as u32;
        assert_eq!(&text[y as usize..y as usize + 1], "y");

        let d = Diagnostic::new(Code::Ec03).at(Span::new(id, y, y + 1));
        let out = to_string(&d, &map);
        // the report must place the caret on line 2, not drift onto line 1
        assert!(out.contains("demo.tq:2:"), "{out}");
    }

    #[test]
    fn a_diagnostic_with_no_span_renders_as_one_line() {
        let (map, _) = sources("");
        let d = Diagnostic::new(Code::El05);
        let out = to_string(&d, &map);
        assert_eq!(out.trim_end(), "error[EL05]: no main, or more than one");
    }

    #[test]
    fn a_diagnostic_with_no_span_keeps_its_notes_and_help() {
        let (map, _) = sources("");
        let d = Diagnostic::new(Code::El03)
            .with_message("the layout of `geo::P` changed")
            .with_note("the two units disagree about it")
            .with_help("recompile `app`");
        let out = to_string(&d, &map);
        assert_eq!(
            out,
            "error[EL03]: the layout of `geo::P` changed\n  = note: the two units disagree about it\n  \
             = help: recompile `app`\n"
        );
    }

    #[test]
    fn a_warning_renders_as_a_warning() {
        let (map, id) = sources("[nonesuch]\nstruct S {}\n");
        let d = Diagnostic::new(Code::Ea01).at(Span::new(id, 1, 9));
        let out = to_string(&d, &map);
        assert!(out.contains("EA01"), "{out}");
        assert!(
            out.to_lowercase().contains("warning"),
            "expected a warning header in {out}"
        );
    }

    #[test]
    fn secondary_labels_are_rendered_too() {
        let (map, id) = sources("fn f() {}\nfn f() {}\n");
        let d = Diagnostic::new(Code::El06)
            .at_with(Span::new(id, 13, 14), "redefined here")
            .also(Span::new(id, 3, 4), "first defined here");
        let out = to_string(&d, &map);
        assert!(out.contains("redefined here"), "{out}");
        assert!(out.contains("first defined here"), "{out}");
    }

    #[test]
    fn spans_in_a_spliced_line_underline_the_original_bytes() {
        // the lexer works on spliced text; spans are original coordinates, so a
        // report on a spliced identifier must still land in the real file
        let (map, id) = sources("let a\\\nb = 1;\n");
        let f = map.file(id);
        let span = f.spliced().span(id, 4, 6);
        assert_eq!(f.slice(span), Some("a\\\nb"));

        let d = Diagnostic::new(Code::Ec03).at(span);
        let out = to_string(&d, &map);
        assert!(out.contains("demo.tq:1:5"), "{out}");
    }

    #[test]
    fn write_all_renders_every_diagnostic() {
        let (map, id) = sources("let x = 1;\n");
        let ds = vec![
            Diagnostic::new(Code::Eu01).at(Span::new(id, 0, 3)),
            Diagnostic::new(Code::Ea01).at(Span::new(id, 4, 5)),
        ];
        let mut buf = Vec::new();
        write_all(&ds, &map, Style::Plain, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("EU01"), "{out}");
        assert!(out.contains("EA01"), "{out}");
    }

    #[test]
    fn an_unknown_source_id_does_not_panic() {
        let (map, _) = sources("let x = 1;\n");
        let d = Diagnostic::new(Code::Eu01).at(Span::new(SourceId(99), 0, 1));
        // rendering may fail, but it must fail gracefully
        let out = to_string(&d, &map);
        assert!(out.contains("EU01"), "{out}");
    }

    #[test]
    fn position_reports_unit_line_and_column() {
        let (map, id) = sources("let x = 1;\nlet y = 2;\n");
        assert_eq!(position(Span::new(id, 15, 16), &map), "demo.tq:2:5");
        assert_eq!(position(Span::new(SourceId(42), 0, 1), &map), "<unknown>");
    }

    #[test]
    fn summary_pluralises() {
        assert_eq!(summary(0, 0), None);
        assert_eq!(summary(1, 0).unwrap(), "1 error");
        assert_eq!(summary(2, 0).unwrap(), "2 errors");
        assert_eq!(summary(0, 1).unwrap(), "1 warning");
        assert_eq!(summary(3, 2).unwrap(), "3 errors, 2 warnings");
    }

    #[test]
    fn style_follows_the_destination() {
        assert_eq!(Style::for_terminal(true), Style::Color);
        assert_eq!(Style::for_terminal(false), Style::Plain);
    }

    #[test]
    fn labels_can_be_built_directly() {
        let (map, id) = sources("let x = 1;\n");
        let d = Diagnostic::new(Code::Ec03).with_label(Label::primary(Span::new(id, 4, 5), "here"));
        assert!(to_string(&d, &map).contains("here"));
    }
}
