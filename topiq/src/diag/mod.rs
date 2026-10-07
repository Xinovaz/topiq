//! Diagnostics: the closed identifier table, the implementation limits, and
//! rendering.
//!
//! # Why diagnostics own their text
//!
//! [`Diagnostic`] holds `String`s rather than borrowing from the
//! [`crate::source::SourceMap`]. This is forced, not stylistic:
//! `ariadne::Cache::fetch` takes `&mut self`, so rendering needs a mutable
//! borrow of the source map. A diagnostic that borrowed from the map could not
//! be alive at that moment.
//!
//! # Structure
//!
//! - [`code`]: every diagnostic identifier the language defines, and the
//!   two `TQ` identifiers of this implementation's own.
//! - [`limits`]: the implementation limits and the `EM05` that guards them.
//! - [`render`]: the one place a [`Span`] becomes an ariadne label.

pub mod code;
pub mod limits;
pub mod render;

pub use code::{Code, Phase, Severity};
pub use limits::{DepthGuard, Limit};

use crate::span::Span;

/// A span with an explanatory message attached, rendered as an underline.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Label {
    /// What to underline.
    pub span: Span,
    /// What to say about it.
    pub message: String,
    /// Whether this is the label the report points at first.
    pub primary: bool,
}

impl Label {
    /// A primary label: the site of the failure.
    pub fn primary(span: Span, message: impl Into<String>) -> Label {
        Label {
            span,
            message: message.into(),
            primary: true,
        }
    }

    /// A secondary label: supporting context, such as the earlier declaration a
    /// duplicate collides with.
    pub fn secondary(span: Span, message: impl Into<String>) -> Label {
        Label {
            span,
            message: message.into(),
            primary: false,
        }
    }
}

/// One diagnostic, ready to render.
///
/// Built by chaining: every method takes and returns `self`.
///
/// ```
/// use topiq::diag::{Code, Diagnostic};
/// use topiq::span::{SourceId, Span};
///
/// let d = Diagnostic::new(Code::Eu01)
///     .at(Span::new(SourceId(0), 0, 4))
///     .with_note("every unit begins with exactly one #unit directive")
///     .with_help("add `#unit classical` or `#unit quantum` as the first line");
/// assert_eq!(d.code, Code::Eu01);
/// assert!(d.is_error());
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnostic {
    /// Which diagnostic this is.
    pub code: Code,
    /// Whether translation continues: the code's own severity, unless
    /// [`Diagnostic::as_warning`] demoted it.
    pub severity: Severity,
    /// The headline. Defaults to the identifier's own message.
    pub message: String,
    /// Underlined spans, in the order they should be shown.
    pub labels: Vec<Label>,
    /// Explanatory notes.
    pub notes: Vec<String>,
    /// Suggested fixes.
    pub helps: Vec<String>,
}

impl Diagnostic {
    /// A diagnostic for `code`, taking its message and severity from the
    /// identifier.
    pub fn new(code: Code) -> Diagnostic {
        Diagnostic {
            code,
            severity: code.severity(),
            message: code.message().to_owned(),
            labels: Vec::new(),
            notes: Vec::new(),
            helps: Vec::new(),
        }
    }

    /// Replaces the headline, for a code whose standard text is too general to
    /// be useful on its own.
    pub fn with_message(mut self, message: impl Into<String>) -> Diagnostic {
        self.message = message.into();
        self
    }

    /// Adds a primary label that says "here"; [`Diagnostic::at_with`] gives
    /// the underline words of its own.
    pub fn at(self, span: Span) -> Diagnostic {
        // "here", not the headline printed just above it. without label text
        // the underline is not drawn at all
        self.with_label(Label::primary(span, "here"))
    }

    /// Adds a primary label with its own message.
    pub fn at_with(self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.with_label(Label::primary(span, message))
    }

    /// Adds a secondary label.
    pub fn also(self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.with_label(Label::secondary(span, message))
    }

    /// Adds a prepared label.
    pub fn with_label(mut self, label: Label) -> Diagnostic {
        self.labels.push(label);
        self
    }

    /// Adds an explanatory note.
    pub fn with_note(mut self, note: impl Into<String>) -> Diagnostic {
        self.notes.push(note.into());
        self
    }

    /// Adds a suggested fix.
    pub fn with_help(mut self, help: impl Into<String>) -> Diagnostic {
        self.helps.push(help.into());
        self
    }

    /// Makes this diagnostic a warning, for the few constructs translation
    /// continues past: an unknown annotation or pragma, an unreachable
    /// `match` arm, a step out of the checked fragment, and a deprecated use.
    pub fn as_warning(mut self) -> Diagnostic {
        self.severity = Severity::Warning;
        self
    }

    /// Whether this stops translation.
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// The span the report points at.
    pub fn primary_span(&self) -> Option<Span> {
        self.labels
            .iter()
            .find(|l| l.primary)
            .or_else(|| self.labels.first())
            .map(|l| l.span)
    }
}

impl std::fmt::Display for Diagnostic {
    /// A one-line rendering, for test assertions and the line a run-time
    /// abort prints.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]: {}",
            self.severity.name(),
            self.code.id(),
            self.message
        )
    }
}

/// Collects diagnostics as translation proceeds.
///
/// The sink counts errors separately from warnings so that a caller can apply
/// the rule that translation continues past a warning, without inspecting
/// every diagnostic.
#[derive(Clone, Default, Debug)]
pub struct DiagnosticSink {
    diagnostics: Vec<Diagnostic>,
    errors: usize,
    warnings: usize,
}

impl DiagnosticSink {
    /// An empty sink.
    pub fn new() -> DiagnosticSink {
        DiagnosticSink::default()
    }

    /// Records a diagnostic.
    pub fn emit(&mut self, d: Diagnostic) {
        match d.severity {
            Severity::Error => self.errors += 1,
            Severity::Warning => self.warnings += 1,
        }
        self.diagnostics.push(d);
    }

    /// Everything recorded, in emission order.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Whether any error was recorded, meaning the program is ill-formed.
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    /// How many errors were recorded.
    pub fn error_count(&self) -> usize {
        self.errors
    }

    /// How many warnings were recorded.
    pub fn warning_count(&self) -> usize {
        self.warnings
    }

    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Whether a diagnostic with this identifier was recorded.
    ///
    /// The natural assertion for a negative test: a fixture is expected to
    /// produce a *named* identifier, not merely to fail.
    pub fn contains(&self, code: Code) -> bool {
        self.diagnostics.iter().any(|d| d.code == code)
    }

    /// Every identifier recorded, in order and with duplicates.
    pub fn codes(&self) -> Vec<Code> {
        self.diagnostics.iter().map(|d| d.code).collect()
    }

    /// Takes the diagnostics out, leaving the sink empty.
    pub fn take(&mut self) -> Vec<Diagnostic> {
        self.errors = 0;
        self.warnings = 0;
        std::mem::take(&mut self.diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SourceId;

    const F: SourceId = SourceId(0);

    fn span(a: u32, b: u32) -> Span {
        Span::new(F, a, b)
    }

    #[test]
    fn a_new_diagnostic_inherits_its_identifiers_text_and_severity() {
        let d = Diagnostic::new(Code::Eu01);
        assert_eq!(d.message, Code::Eu01.message());
        assert_eq!(d.severity, Severity::Error);
        assert!(d.is_error());

        let w = Diagnostic::new(Code::Ea01);
        assert_eq!(w.severity, Severity::Warning);
        assert!(!w.is_error());
    }

    #[test]
    fn at_marks_the_span_without_repeating_the_headline() {
        let d = Diagnostic::new(Code::Eu04).at(span(3, 9));
        assert_eq!(d.labels.len(), 1);
        assert!(d.labels[0].primary);
        assert_eq!(d.primary_span(), Some(span(3, 9)));
        // short, and deliberately not the headline: the headline is printed
        // immediately above the underline
        assert_eq!(d.labels[0].message, "here");
        assert_ne!(d.labels[0].message, Code::Eu04.message());
    }

    #[test]
    fn at_with_keeps_a_label_that_has_something_of_its_own_to_say() {
        let d = Diagnostic::new(Code::Eu04).at_with(span(3, 9), "not a multiple of 8");
        assert_eq!(d.labels[0].message, "not a multiple of 8");
    }

    #[test]
    fn primary_span_prefers_a_primary_label_over_a_secondary_one() {
        let d = Diagnostic::new(Code::El06)
            .also(span(0, 1), "first definition")
            .at_with(span(10, 11), "second definition");
        assert_eq!(d.primary_span(), Some(span(10, 11)));
    }

    #[test]
    fn primary_span_falls_back_to_the_first_label() {
        let d = Diagnostic::new(Code::El06).also(span(4, 5), "here");
        assert_eq!(d.primary_span(), Some(span(4, 5)));
    }

    #[test]
    fn a_diagnostic_with_no_labels_has_no_span() {
        assert_eq!(Diagnostic::new(Code::El05).primary_span(), None);
    }

    #[test]
    fn notes_and_helps_accumulate_in_order() {
        let d = Diagnostic::new(Code::Eu01)
            .with_note("one")
            .with_note("two")
            .with_help("try this");
        assert_eq!(d.notes, ["one", "two"]);
        assert_eq!(d.helps, ["try this"]);
    }

    #[test]
    fn one_line_rendering_names_the_identifier() {
        let d = Diagnostic::new(Code::Eq01);
        assert_eq!(
            d.to_string(),
            "error[EQ01]: quantum value dropped while possibly live"
        );
    }

    #[test]
    fn severity_can_be_relaxed_where_the_spec_says_diagnosed() {
        let d = Diagnostic::new(Code::Em05).as_warning();
        assert!(!d.is_error());
        assert!(d.to_string().starts_with("warning[EM05]"));
    }

    #[test]
    fn the_sink_counts_errors_and_warnings_separately() {
        let mut sink = DiagnosticSink::new();
        assert!(sink.is_empty());
        sink.emit(Diagnostic::new(Code::Eu01));
        sink.emit(Diagnostic::new(Code::Ea01));
        sink.emit(Diagnostic::new(Code::Ea03));
        assert_eq!(sink.error_count(), 1);
        assert_eq!(sink.warning_count(), 2);
        assert!(sink.has_errors());
    }

    #[test]
    fn a_sink_of_warnings_alone_does_not_make_a_program_ill_formed() {
        // an unknown annotation or pragma is a warning, and
        // translation continues
        let mut sink = DiagnosticSink::new();
        sink.emit(Diagnostic::new(Code::Ea01));
        sink.emit(Diagnostic::new(Code::Ea03));
        assert!(!sink.has_errors());
    }

    #[test]
    fn contains_answers_the_question_negative_tests_ask() {
        let mut sink = DiagnosticSink::new();
        sink.emit(Diagnostic::new(Code::Ej04));
        assert!(sink.contains(Code::Ej04));
        assert!(!sink.contains(Code::Ej05));
        assert_eq!(sink.codes(), vec![Code::Ej04]);
    }

    #[test]
    fn take_empties_the_sink_and_resets_the_counts() {
        let mut sink = DiagnosticSink::new();
        sink.emit(Diagnostic::new(Code::Eu01));
        let taken = sink.take();
        assert_eq!(taken.len(), 1);
        assert!(sink.is_empty());
        assert_eq!(sink.error_count(), 0);
        assert!(!sink.has_errors());
    }

    #[test]
    fn labels_record_which_is_primary() {
        let p = Label::primary(span(0, 1), "here");
        let s = Label::secondary(span(2, 3), "there");
        assert!(p.primary);
        assert!(!s.primary);
    }
}
