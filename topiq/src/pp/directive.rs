//! Preprocessing directives.
//!
//! This is the driver for translation phase 4. It splits the token stream into
//! logical lines, executes the directives, expands macros in what remains, and
//! produces the token stream [`crate::parse`] consumes.
//!
//! # Logical lines
//!
//! A directive occupies one logical line, beginning with a `#` that is the
//! first thing on it. *Logical* is the load-bearing word: phase 2 has already
//! joined lines across a backslash-newline, so
//!
//! ```text
//! #define X 1 \
//!           + 2
//! ```
//!
//! is a single directive. Line numbering therefore comes from
//! [`crate::source::Spliced::logical_line`] rather than from the physical line
//! count a reader sees.
//!
//! # `static #define`
//!
//! A macro private to its unit is written `static #define`, which puts a
//! keyword *before* the `#`. It is accepted here as an optional prefix on a
//! directive line, and is meaningful on `#define` and `#alias`.
//!
//! # `#alias`
//!
//! `#alias dit = bool | qubit`, or `#alias dit classical(bool)
//! quantum(qubit)`, is the type alias for the unit's kind, written out as the
//! item `type dit = …;`, so that one routine reads the same in a unit of
//! either kind. The classical side comes first, as `__UNIT_KIND__` counts the
//! kinds (`__CLASSICAL__` is 1, `__QUANTUM__` 2). The side for the other kind
//! is never read as a type, so a classical unit does not see `qubit`; a side
//! left out gives no alias in that kind. Generic parameters follow the name,
//! and `static #alias` gives the alias unit linkage.
//!
//! # There is no `#include`
//!
//! Textual inclusion is left out on purpose. Code reuse is by `import`, which
//! is semantic and typed; data inclusion is by `#embed`, which is typed through
//! the data notations. A unit's interface is what it declares, not whatever a
//! header happened to paste in ahead of it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::{Interner, Symbol};
use crate::lex::{Keyword, Punct, Token, token::spell};
use crate::source::SourceMap;
use crate::span::{SourceId, Span};

use super::eval;
use super::expand::Expander;
use super::macros::{Linkage, MacroDef, MacroTable, PpToken};

/// Whether a unit is classical or quantum, in its entirety: which primitive
/// types it may materialise, which statements it may use, and what it
/// compiles to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnitKind {
    /// `#unit classical`: produces a `.tcu`.
    Classical,
    /// `#unit quantum`: produces a `.tqu`.
    Quantum,
}

impl UnitKind {
    /// The value of the predefined macro `__UNIT_KIND__`: 1 in a
    /// classical unit, 2 in a quantum one.
    pub fn macro_value(self) -> u8 {
        match self {
            UnitKind::Classical => 1,
            UnitKind::Quantum => 2,
        }
    }

    /// The name as written in the directive.
    pub fn name(self) -> &'static str {
        match self {
            UnitKind::Classical => "classical",
            UnitKind::Quantum => "quantum",
        }
    }
}

/// What a `#unit` directive declared.
#[derive(Clone, Copy, Debug)]
pub struct UnitDirective {
    /// Classical or quantum: for `#unit any`, the kind chosen for it.
    pub kind: UnitKind,
    /// The unit initialiser named in `#unit kind(name)`, or the one `#unit
    /// any` names for the kind chosen, if any.
    pub initializer: Option<Symbol>,
    /// For `#unit any`, what else it declared.
    pub any: Option<AnyUnit>,
    /// Where the directive was written.
    pub span: Span,
}

/// What `#unit any` declared beyond the kind chosen for it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AnyUnit {
    /// The kind it prefers when nothing else chooses (e.g. `#unit any(quantum)`).
    pub preference: Option<UnitKind>,
}

/// What chooses the kind of a `#unit any` unit, besides its preference.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct KindChoice {
    /// A kind asked for outright, by `import(kind)` or `--kind`, which comes
    /// before the unit's preference.
    pub forced: Option<UnitKind>,
    /// The kind of the unit importing it.
    pub fallback: Option<UnitKind>,
}

/// The checked fragment a unit intends to stay inside.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Fragment {
    /// `#pragma fragment(finite)`: the checked fragment, and the default.
    #[default]
    Finite,
    /// `#pragma fragment(stabilizer)`: the polynomial-time sub-fragment.
    /// Declaring it means that leaving it is reported, rather than showing up
    /// as an unexplained slowdown in the checker.
    Stabilizer,
}

/// The register-allocation strategy for a unit's circuits.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QAlloc {
    /// `#pragma qalloc(reuse)`: the lowest-numbered free node at each
    /// allocation, so node assignment is a function of the source. The default.
    #[default]
    Reuse,
    /// `#pragma qalloc(linear)`: every register allocated in the unit occupies
    /// distinct nodes for the whole circuit.
    Linear,
}

/// The result of preprocessing one unit.
#[derive(Debug)]
pub struct Preprocessed {
    /// The final token stream.
    pub tokens: Vec<(Token, Span)>,
    /// The unit directive.
    pub unit: Option<UnitDirective>,
    /// The unit's conductor.
    pub conductor: u32,
    /// The fragment the unit declares it stays inside.
    pub fragment: Fragment,
    /// The register-allocation strategy.
    pub qalloc: QAlloc,
    /// The macro table as it stands at the end of the unit.
    pub macros: MacroTable,
    /// Everything diagnosed along the way.
    pub diagnostics: Vec<Diagnostic>,
}

impl Preprocessed {
    /// Whether preprocessing produced any error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// How `#embed` finds a file, so that preprocessing can be tested without disk
/// access.
pub trait EmbedResolver {
    /// Reads the document at `path`, resolved relative to `from`.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason on failure.
    fn read(&mut self, from: &Path, path: &str) -> Result<String, String>;
}

/// Resolves `#embed` against the real filesystem, relative to the including
/// file's directory.
#[derive(Debug, Default)]
pub struct FsEmbedResolver;

impl EmbedResolver for FsEmbedResolver {
    fn read(&mut self, from: &Path, path: &str) -> Result<String, String> {
        let base = from.parent().unwrap_or_else(|| Path::new("."));
        let full: PathBuf = base.join(path);
        std::fs::read_to_string(&full).map_err(|e| format!("{}: {e}", full.display()))
    }
}

/// An `#embed` resolver backed by an in-memory map, for tests.
#[derive(Debug, Default)]
pub struct MapEmbedResolver {
    /// Path to contents.
    pub files: HashMap<String, String>,
}

impl EmbedResolver for MapEmbedResolver {
    fn read(&mut self, _from: &Path, path: &str) -> Result<String, String> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| format!("no such embedded document `{path}`"))
    }
}

/// One frame of the conditional-inclusion stack.
struct Cond {
    /// Whether some branch of this `#if` chain has already been taken.
    taken: bool,
    /// Whether the branch currently open is being emitted.
    active: bool,
    /// Whether `#else` has been seen, so a second one is an error.
    seen_else: bool,
    /// Whether the enclosing region was itself active.
    parent_active: bool,
    /// Where the `#if` was written.
    span: Span,
}

/// Runs translation phase 4 over one file's tokens.
pub fn preprocess(
    source: SourceId,
    tokens: &[(Token, Span)],
    sources: &SourceMap,
    interner: &mut Interner,
    embed: &mut dyn EmbedResolver,
) -> Preprocessed {
    preprocess_as(source, tokens, sources, interner, embed, KindChoice::default())
}

/// [`preprocess`], with the kind of a `#unit any` unit chosen by `choice`.
pub fn preprocess_as(
    source: SourceId,
    tokens: &[(Token, Span)],
    sources: &SourceMap,
    interner: &mut Interner,
    embed: &mut dyn EmbedResolver,
    choice: KindChoice,
) -> Preprocessed {
    let mut pp = Pp {
        choice,
        source,
        sources,
        interner,
        embed,
        table: MacroTable::new(),
        out: Vec::new(),
        diagnostics: Vec::new(),
        conds: Vec::new(),
        unit: None,
        conductor: 8,
        fragment: Fragment::default(),
        qalloc: QAlloc::default(),
        seen_any_token: false,
    };
    pp.run(tokens);
    Preprocessed {
        tokens: pp.out,
        unit: pp.unit,
        conductor: pp.conductor,
        fragment: pp.fragment,
        qalloc: pp.qalloc,
        macros: pp.table,
        diagnostics: pp.diagnostics,
    }
}

struct Pp<'a> {
    choice: KindChoice,
    source: SourceId,
    sources: &'a SourceMap,
    interner: &'a mut Interner,
    embed: &'a mut dyn EmbedResolver,
    table: MacroTable,
    out: Vec<(Token, Span)>,
    diagnostics: Vec<Diagnostic>,
    conds: Vec<Cond>,
    unit: Option<UnitDirective>,
    conductor: u32,
    fragment: Fragment,
    qalloc: QAlloc,
    seen_any_token: bool,
}

impl<'a> Pp<'a> {
    fn run(&mut self, tokens: &[(Token, Span)]) {
        for line in self.split_lines(tokens) {
            self.do_line(&line);
        }
        // a file that never says what it is cannot be translated
        if self.unit.is_none() {
            let span = tokens
                .first()
                .map_or(Span::at(self.source, 0), |(_, s)| *s);
            self.error(
                Code::Eu01,
                "this unit has no #unit directive",
                span,
                &[
                    "every Topiq source file opens by declaring what it is: a `#unit` \
                     directive naming the unit's kind, and optionally its initialiser",
                    "the directive comes before any item, with only comments and \
                     whitespace allowed ahead of it",
                ],
                &[
                    "add `#unit classical` as the first line, or `#unit quantum` if this \
                     unit works with qubits",
                ],
            );
        }
        for c in std::mem::take(&mut self.conds) {
            self.diagnostics.push(
                Diagnostic::new(Code::Es02)
                    .with_message("unterminated #if")
                    .at(c.span)
                    .with_help("add `#endif`"),
            );
        }
    }

    /// Groups tokens into logical lines.
    fn split_lines(&self, tokens: &[(Token, Span)]) -> Vec<Vec<PpToken>> {
        let Some(file) = self.sources.get(self.source) else {
            return vec![tokens.iter().map(|(t, s)| PpToken::new(*t, *s)).collect()];
        };
        let spliced = file.spliced();
        let mut lines: Vec<Vec<PpToken>> = Vec::new();
        let mut current_line: Option<u32> = None;
        for (tok, span) in tokens {
            let line = spliced.logical_line(span.start);
            if current_line != Some(line) {
                lines.push(Vec::new());
                current_line = Some(line);
            }
            lines.last_mut().unwrap().push(PpToken::new(*tok, *span));
        }
        lines
    }

    fn active(&self) -> bool {
        self.conds.last().is_none_or(|c| c.active && c.parent_active)
    }

    fn error(
        &mut self,
        code: Code,
        message: &str,
        span: Span,
        notes: &[&str],
        helps: &[&str],
    ) {
        let mut d = Diagnostic::new(code).with_message(message).at(span);
        for n in notes {
            d = d.with_note(*n);
        }
        for h in helps {
            d = d.with_help(*h);
        }
        self.diagnostics.push(d);
    }

    /// A malformed directive: `ES02` saying `message`.
    fn malformed(&mut self, message: &str, span: Span) {
        self.error(Code::Es02, message, span, &[], &[]);
    }

    /// `toks` with their macros expanded.
    fn expand(&mut self, toks: Vec<PpToken>) -> Vec<PpToken> {
        Expander::new(&mut self.table, self.interner, self.sources, &mut self.diagnostics, self.conductor).expand(toks)
    }

    /// Whether a line is a directive, and where its keyword sits.
    ///
    /// Accepts the `static #define` form as well as a plain
    /// `# directive`.
    fn directive_at(line: &[PpToken]) -> Option<usize> {
        match line.first().map(|t| t.tok) {
            Some(Token::Punct(Punct::Hash)) => Some(1),
            Some(Token::Kw(Keyword::Static))
                if matches!(line.get(1).map(|t| t.tok), Some(Token::Punct(Punct::Hash))) =>
            {
                Some(2)
            }
            _ => None,
        }
    }

    fn do_line(&mut self, line: &[PpToken]) {
        if line.is_empty() {
            return;
        }
        match Self::directive_at(line) {
            Some(kw_at) => self.do_directive(line, kw_at),
            None if self.active() => self.emit_text(line),
            None => {}
        }
    }

    /// Expands and emits an ordinary (non-directive) line.
    fn emit_text(&mut self, line: &[PpToken]) {
        self.seen_any_token = true;
        let expanded = self.expand(line.to_vec());
        self.out.extend(expanded.into_iter().map(|t| (t.tok, t.span)));
    }

    fn do_directive(&mut self, line: &[PpToken], kw_at: usize) {
        let hash_span = line[kw_at - 1].span;
        let is_static = kw_at == 2;
        let name = match line.get(kw_at).map(|t| t.tok) {
            Some(Token::Ident(s)) => self.interner.resolve(s).to_owned(),
            // `#if` and `#else` are ordinary keywords in the language, so they
            // arrive as keyword tokens rather than identifiers
            Some(Token::Kw(k)) => k.text().to_owned(),
            _ => {
                if self.active() {
                    self.malformed("expected a directive name after `#`", hash_span);
                }
                return;
            }
        };
        let rest = &line[kw_at + 1..];

        // the conditional directives run whether or not the region is active,
        // because they are what decides activity
        match name.as_str() {
            "if" => return self.do_if(rest, hash_span),
            "ifdef" => return self.do_ifdef(rest, hash_span, true),
            "ifndef" => return self.do_ifdef(rest, hash_span, false),
            "elif" => return self.do_elif(rest, hash_span),
            "else" => return self.do_else(hash_span),
            "endif" => return self.do_endif(hash_span),
            _ => {}
        }
        if !self.active() {
            return;
        }
        match name.as_str() {
            "unit" => self.do_unit(rest, hash_span),
            "define" => self.do_define(rest, hash_span, is_static),
            "alias" => self.do_alias(rest, hash_span, is_static),
            "undef" => self.do_undef(rest, hash_span),
            "pragma" => self.do_pragma(rest, hash_span),
            "error" => self.do_message(rest, hash_span, true),
            "warning" => self.do_message(rest, hash_span, false),
            "embed" => self.do_embed(rest, hash_span),
            "include" => self.error(
                Code::Es02,
                "Topiq has no `#include` directive",
                hash_span,
                &[
                    "textual inclusion is left out deliberately: a unit's interface comes \
                     from its own declarations, not from whatever a header happened to \
                     paste in",
                ],
                &[
                    "to use another unit's items, write `import <unit>;`",
                    "to pull in data rather than code, write `#embed \"path\"`",
                ],
            ),
            other => self.error(
                Code::Es02,
                &format!("unknown preprocessing directive `#{other}`"),
                hash_span,
                &[
                    "the directives are `#unit`, `#define`, `#undef`, `#alias`, `#if`, \
                     `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`, `#pragma`, `#error`, \
                     `#warning` and `#embed`, and no others",
                ],
                &[],
            ),
        }
    }

    ////////////////////////
    // THE UNIT DIRECTIVE //
    ////////////////////////

    fn do_unit(&mut self, rest: &[PpToken], span: Span) {
        let word = match rest.first().map(|t| t.tok) {
            Some(Token::Ident(s)) => self.interner.resolve(s).to_owned(),
            _ => String::new(),
        };
        let fixed = match word.as_str() {
            "classical" => Some(UnitKind::Classical),
            "quantum" => Some(UnitKind::Quantum),
            "any" => None,
            _ => {
                self.error(
                    Code::Eu01,
                    "a #unit directive names `classical`, `quantum` or `any`",
                    span,
                    &[],
                    &["write `#unit classical` or `#unit quantum`, or `#unit any` for a unit of either kind"],
                );
                return;
            }
        };
        let (kind, initializer, any) = match fixed {
            Some(kind) => match self.fixed_unit(rest, span) {
                Some(init) => (kind, init, None),
                None => return,
            },
            None => match self.any_unit(&rest[1..], span) {
                Some(found) => found,
                None => return,
            },
        };

        if self.unit.is_some() {
            self.error(
                Code::Eu01,
                "duplicate #unit directive",
                span,
                &[
                    "a unit declares itself exactly once; a second directive would leave \
                     it ambiguous which kind and name the unit really has",
                ],
                &["remove this directive, or merge it into the one above"],
            );
            return;
        }
        if self.seen_any_token {
            self.error(
                Code::Eu01,
                "the #unit directive must come before any item",
                span,
                &[
                    "the directive must precede every item in the file; only comments and \
                     whitespace may come before it",
                ],
                &["move this line to the top of the file"],
            );
            return;
        }

        self.unit = Some(UnitDirective {
            kind,
            initializer,
            any,
            span,
        });
        let (file_name, unit_name) = self
            .sources
            .get(self.source)
            .map_or_else(|| ("<unknown>".to_owned(), "<unknown>".to_owned()), |f| (f.name().to_owned(), f.unit_name().to_owned()));
        self.table
            .install_predefined(self.interner, span, crate::LANGUAGE_EDITION, &file_name, &unit_name, kind.macro_value());
    }

    /// The initialiser of `#unit classical(name)` or `#unit quantum(name)`, if
    /// it names one; `None` when the directive is malformed.
    fn fixed_unit(&mut self, rest: &[PpToken], span: Span) -> Option<Option<Symbol>> {
        let mut initializer = None;
        if rest.len() > 1 {
            let ok = rest.len() == 4
                && rest[1].tok.is(Punct::LParen)
                && rest[3].tok.is(Punct::RParen)
                && matches!(rest[2].tok, Token::Ident(_));
            if ok {
                initializer = rest[2].tok.ident();
            } else {
                self.error(
                    Code::Eu01,
                    "malformed #unit directive",
                    span,
                    &[],
                    &["the initialiser form is `#unit classical(name)`"],
                );
                return None;
            }
        }
        Some(initializer)
    }

    /// The kind, initialiser and declaration of `#unit any(ITEMS)`, `items`
    /// being what follows `any`: an optional preference, `classical` or
    /// `quantum`, first, then initialisers labelled `classical(name)`,
    /// `quantum(name)` or `both(name)`. The kind is the one asked for
    /// outright, then the preference, then the importer's; with none, `EU07`.
    /// `None` when the directive is malformed (`EU01`).
    fn any_unit(&mut self, items: &[PpToken], span: Span) -> Option<(UnitKind, Option<Symbol>, Option<AnyUnit>)> {
        let form = "the form is `#unit any(classical, classical(setup), quantum(build))`, every part optional, \
                    or `both(init)` for one initialiser of either kind";
        let mut preference = None;
        // the initialisers for the classical kind, the quantum kind, and both
        let mut named: [Option<Symbol>; 3] = [None; 3];
        if !items.is_empty() {
            let closed = items.len() >= 2
                && items[0].tok.is(Punct::LParen)
                && items[items.len() - 1].tok.is(Punct::RParen);
            if !closed {
                self.error(Code::Eu01, "malformed #unit any directive", span, &[], &[form]);
                return None;
            }
            let inner = &items[1..items.len() - 1];
            for (i, part) in inner.split(|t| t.tok.is(Punct::Comma)).enumerate() {
                let word = |t: &PpToken, interner: &Interner| match t.tok {
                    Token::Ident(s) => interner.resolve(s).to_owned(),
                    _ => String::new(),
                };
                let kind_of = |w: &str| match w {
                    "classical" => Some(UnitKind::Classical),
                    "quantum" => Some(UnitKind::Quantum),
                    _ => None,
                };
                match part {
                    [w] if i == 0 && kind_of(&word(w, self.interner)).is_some() => {
                        preference = kind_of(&word(w, self.interner));
                    }
                    [w, open, name, close]
                        if open.tok.is(Punct::LParen)
                            && close.tok.is(Punct::RParen)
                            && matches!(name.tok, Token::Ident(_)) =>
                    {
                        let label = word(w, self.interner);
                        let slot = match label.as_str() {
                            "classical" => 0,
                            "quantum" => 1,
                            "both" => 2,
                            _ => {
                                self.error(
                                    Code::Eu01,
                                    &format!("`{label}` labels no initialiser of `#unit any`"),
                                    w.span,
                                    &[],
                                    &[form],
                                );
                                return None;
                            }
                        };
                        if named[slot].is_some() {
                            self.error(Code::Eu01, &format!("`#unit any` names its {label} initialiser twice"), w.span, &[], &[form]);
                            return None;
                        }
                        named[slot] = name.tok.ident();
                    }
                    [w] if kind_of(&word(w, self.interner)).is_some() => {
                        self.error(
                            Code::Eu01,
                            "the kind a `#unit any` unit prefers comes first",
                            w.span,
                            &[],
                            &[form],
                        );
                        return None;
                    }
                    _ => {
                        let at = part.first().map_or(span, |t| t.span);
                        self.error(Code::Eu01, "malformed #unit any directive", at, &[], &[form]);
                        return None;
                    }
                }
            }
            if named[2].is_some() && (named[0].is_some() || named[1].is_some()) {
                self.error(
                    Code::Eu01,
                    "`both(…)` names the initialiser for either kind, so it is not written with `classical(…)` or `quantum(…)`",
                    span,
                    &[],
                    &[form],
                );
                return None;
            }
        }
        let any = Some(AnyUnit { preference });
        let Some(kind) = self.choice.forced.or(preference).or(self.choice.fallback) else {
            self.error(
                Code::Eu07,
                "the kind of this `#unit any` unit is not decided",
                span,
                &[
                    "a `#unit any` unit is classical or quantum as whatever builds or imports it chooses; \
                     nothing chose here, and the unit prefers neither",
                ],
                &[
                    "build it with `--kind classical` or `--kind quantum`, give it a preference such as \
                     `#unit any(classical)`, or import it from another unit",
                ],
            );
            // recorded as classical so that nothing else is reported for it
            return Some((UnitKind::Classical, None, any));
        };
        let initializer = match kind {
            UnitKind::Classical => named[0].or(named[2]),
            UnitKind::Quantum => named[1].or(named[2]),
        };
        Some((kind, initializer, any))
    }

    //////////////////
    // KIND ALIASES //
    //////////////////

    /// `#alias NAME GENERICS? = CLASSICAL | QUANTUM`, or with its sides
    /// labelled, `#alias NAME GENERICS? classical(…) quantum(…)`: the type
    /// alias for this unit's kind, written out as the item
    /// `type NAME GENERICS = SIDE;`. A side left empty or out gives no alias in
    /// that kind, and the side for the other kind is never read as a type.
    fn do_alias(&mut self, rest: &[PpToken], span: Span, is_static: bool) {
        let Some(unit) = &self.unit else {
            self.error(
                Code::Eu01,
                "#alias comes after the #unit directive",
                span,
                &["which side of an `#alias` applies depends on the unit's kind, which `#unit` fixes"],
                &["move the `#unit` directive to the top of the file"],
            );
            return;
        };
        let kind = unit.kind;
        if !matches!(rest.first().map(|t| t.tok), Some(Token::Ident(_))) {
            self.malformed("#alias needs the name of the alias", span);
            return;
        }

        // the name and any generic parameters, `<` to its matching `>`
        let mut at = 1;
        if rest.get(1).is_some_and(|t| t.tok.is(Punct::Lt)) {
            let mut depth = 0i32;
            loop {
                let Some(t) = rest.get(at) else {
                    self.malformed("#alias has generic parameters that are not closed with `>`", span);
                    return;
                };
                if t.tok.is(Punct::Lt) {
                    depth += 1;
                } else if t.tok.is(Punct::Gt) {
                    depth -= 1;
                }
                at += 1;
                if depth == 0 {
                    break;
                }
            }
        }
        let (head, body) = rest.split_at(at);

        let sides = if body.first().is_some_and(|t| t.tok.is(Punct::Eq)) {
            self.alias_sides_bar(&body[1..], span)
        } else {
            self.alias_sides_labelled(body, span)
        };
        let Some((classical, quantum)) = sides else { return };
        let side = match kind {
            UnitKind::Classical => classical,
            UnitKind::Quantum => quantum,
        };
        if side.is_empty() {
            return;
        }

        // `[static] type NAME GENERICS = SIDE;`
        self.seen_any_token = true;
        let mut item = Vec::with_capacity(head.len() + side.len() + 4);
        if is_static {
            item.push(PpToken::new(Token::Kw(Keyword::Static), span));
        }
        item.push(PpToken::new(Token::Kw(Keyword::Type), span));
        item.extend_from_slice(head);
        item.push(PpToken::new(Token::Punct(Punct::Eq), span));
        item.extend(side);
        item.push(PpToken::new(Token::Punct(Punct::Semi), span));
        let expanded = self.expand(item);
        self.out.extend(expanded.into_iter().map(|t| (t.tok, t.span)));
    }

    /// The sides of `= CLASSICAL | QUANTUM`, which no type form can confuse,
    /// since no type holds a `|`.
    fn alias_sides_bar(&mut self, body: &[PpToken], span: Span) -> Option<(Vec<PpToken>, Vec<PpToken>)> {
        let bars: Vec<usize> = body.iter().enumerate().filter(|(_, t)| t.tok.is(Punct::Or)).map(|(i, _)| i).collect();
        let [bar] = bars[..] else {
            self.error(
                Code::Es02,
                "#alias with `=` gives two sides separated by one `|`",
                span,
                &["the classical side comes first and the quantum side second, as `__UNIT_KIND__` counts them"],
                &["write `#alias dit = bool | qubit`, or label the sides: `#alias dit classical(bool) quantum(qubit)`"],
            );
            return None;
        };
        Some((body[..bar].to_vec(), body[bar + 1..].to_vec()))
    }

    /// The sides of `classical(…) quantum(…)`, in either order, each written at
    /// most once.
    fn alias_sides_labelled(&mut self, body: &[PpToken], span: Span) -> Option<(Vec<PpToken>, Vec<PpToken>)> {
        let help = "write `#alias dit classical(bool) quantum(qubit)`, or `#alias dit = bool | qubit`";
        let mut classical: Option<Vec<PpToken>> = None;
        let mut quantum: Option<Vec<PpToken>> = None;
        let mut i = 0;
        while i < body.len() {
            let label = match body[i].tok {
                Token::Ident(s) => self.interner.resolve(s).to_owned(),
                _ => String::new(),
            };
            let slot = match label.as_str() {
                "classical" => &mut classical,
                "quantum" => &mut quantum,
                _ => {
                    self.error(
                        Code::Es02,
                        "#alias takes `= CLASSICAL | QUANTUM`, or sides labelled `classical(…)` and `quantum(…)`",
                        body[i].span,
                        &[],
                        &[help],
                    );
                    return None;
                }
            };
            if slot.is_some() {
                self.error(Code::Es02, &format!("#alias gives its {label} side twice"), body[i].span, &[], &[help]);
                return None;
            }
            // the parenthesised side, to its matching `)`
            if !body.get(i + 1).is_some_and(|t| t.tok.is(Punct::LParen)) {
                self.error(Code::Es02, &format!("`{label}` is followed by its side in parentheses"), body[i].span, &[], &[help]);
                return None;
            }
            let mut depth = 0i32;
            let mut j = i + 1;
            loop {
                let Some(t) = body.get(j) else {
                    self.malformed(&format!("the {label} side of #alias is not closed with `)`"), span);
                    return None;
                };
                if t.tok.is(Punct::LParen) {
                    depth += 1;
                } else if t.tok.is(Punct::RParen) {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                j += 1;
            }
            *slot = Some(body[i + 2..j].to_vec());
            i = j + 1;
        }
        if classical.is_none() && quantum.is_none() {
            self.error(Code::Es02, "#alias needs a side for at least one kind", span, &[], &[help]);
            return None;
        }
        Some((classical.unwrap_or_default(), quantum.unwrap_or_default()))
    }

    //////////////////////
    // MACRO DEFINITION //
    //////////////////////

    fn do_define(&mut self, rest: &[PpToken], span: Span, is_static: bool) {
        let Some(Token::Ident(name)) = rest.first().map(|t| t.tok) else {
            self.malformed("#define needs a macro name", span);
            return;
        };
        if super::macros::is_predefined(self.interner.resolve(name)) {
            self.error(
                Code::Ea04,
                &format!(
                    "`{}` is a predefined macro and may not be redefined",
                    self.interner.resolve(name)
                ),
                rest[0].span,
                &[
                    "the compiler supplies six macro names that describe the translation \
                     in progress, and this is one of them; redefining it would make it \
                     mean different things in different parts of the file",
                ],
                &["choose a different name for your macro"],
            );
            return;
        }

        // function-like only if `(` follows the name immediately, with nothing
        // in between: `#define F(x)` takes a parameter, `#define F (x)` does
        // not
        let mut params: Option<Vec<Symbol>> = None;
        let mut variadic = false;
        let mut body_at = 1;
        if rest.len() > 1
            && rest[1].tok.is(Punct::LParen)
            && rest[0].span.adjacent_to(rest[1].span)
        {
            let mut ps = Vec::new();
            let mut i = 2;
            let mut closed = false;
            while i < rest.len() {
                match rest[i].tok {
                    Token::Punct(Punct::RParen) => {
                        closed = true;
                        i += 1;
                        break;
                    }
                    Token::Punct(Punct::Comma) => i += 1,
                    Token::Punct(Punct::DotDot) | Token::Punct(Punct::DotDotEq) => {
                        // `...` lexes as `..` then `.`, or as `..=` when the
                        // next character is `=`; either way the ellipsis ends
                        // the parameter list
                        variadic = true;
                        i += 1;
                        if matches!(rest.get(i).map(|t| t.tok), Some(Token::Punct(Punct::Dot))) {
                            i += 1;
                        }
                    }
                    Token::Ident(s) => {
                        if ps.contains(&s) {
                            self.malformed("duplicate macro parameter name", rest[i].span);
                        }
                        ps.push(s);
                        i += 1;
                    }
                    _ => {
                        self.malformed("unexpected token in a macro parameter list", rest[i].span);
                        i += 1;
                    }
                }
            }
            if !closed {
                self.malformed("unterminated macro parameter list", span);
                return;
            }
            params = Some(ps);
            body_at = i;
        }

        let body: Vec<PpToken> = rest[body_at..].to_vec();
        self.table.define(MacroDef {
            name,
            params,
            variadic,
            body,
            linkage: if is_static { Linkage::Unit } else { Linkage::Program },
            span,
            predefined: false,
        });
    }

    fn do_undef(&mut self, rest: &[PpToken], span: Span) {
        let Some(Token::Ident(name)) = rest.first().map(|t| t.tok) else {
            self.malformed("#undef needs a macro name", span);
            return;
        };
        if super::macros::is_predefined(self.interner.resolve(name)) {
            self.error(
                Code::Ea04,
                &format!(
                    "`{}` is a predefined macro and may not be undefined",
                    self.interner.resolve(name)
                ),
                rest[0].span,
                &[
                    "the compiler supplies six macro names that describe the translation \
                     in progress, and this is one of them; code later in the file is \
                     entitled to rely on it",
                ],
                &[],
            );
            return;
        }
        self.table.undef(name);
    }

    ///////////////////////////
    // CONDITIONAL INCLUSION //
    ///////////////////////////

    fn do_if(&mut self, rest: &[PpToken], span: Span) {
        let parent_active = self.active();
        let taken = parent_active && self.eval_condition(rest, span);
        self.push_cond(taken, parent_active, span);
    }

    fn do_ifdef(&mut self, rest: &[PpToken], span: Span, want: bool) {
        let parent_active = self.active();
        let taken = parent_active
            && match rest.first().map(|t| t.tok) {
                Some(Token::Ident(name)) => self.table.is_defined(name) == want,
                _ => {
                    self.malformed("#ifdef and #ifndef need a macro name", span);
                    false
                }
            };
        self.push_cond(taken, parent_active, span);
    }

    fn push_cond(&mut self, taken: bool, parent_active: bool, span: Span) {
        if self.conds.len() as u32 >= Limit::BlockNesting.value() {
            self.diagnostics
                .push(Limit::BlockNesting.exceeded(span, self.conds.len() as u32 + 1));
        }
        self.conds.push(Cond {
            taken,
            active: taken,
            seen_else: false,
            parent_active,
            span,
        });
    }

    fn do_elif(&mut self, rest: &[PpToken], span: Span) {
        let Some(top) = self.conds.last() else {
            self.malformed("#elif without #if", span);
            return;
        };
        if top.seen_else {
            self.malformed("#elif after #else", span);
            return;
        }
        let (already, parent_active) = (top.taken, top.parent_active);
        let take = !already && parent_active && self.eval_condition(rest, span);
        let top = self.conds.last_mut().unwrap();
        top.active = take;
        top.taken = already || take;
    }

    fn do_else(&mut self, span: Span) {
        let Some(top) = self.conds.last_mut() else {
            self.malformed("#else without #if", span);
            return;
        };
        if top.seen_else {
            let s = top.span;
            self.diagnostics.push(
                Diagnostic::new(Code::Es02)
                    .with_message("a second #else in one #if")
                    .at(span)
                    .also(s, "this #if already has an #else"),
            );
            return;
        }
        top.seen_else = true;
        top.active = !top.taken;
        top.taken = true;
    }

    fn do_endif(&mut self, span: Span) {
        if self.conds.pop().is_none() {
            self.malformed("#endif without #if", span);
        }
    }

    /// Expands macros in a condition, then evaluates it.
    ///
    /// `defined(NAME)` is protected from expansion first, so that a macro named
    /// in it is not expanded away before the question is asked.
    fn eval_condition(&mut self, rest: &[PpToken], span: Span) -> bool {
        let guarded = self.fold_defined(rest);
        let expanded = self.expand(guarded);
        eval::eval(&expanded, &self.table, self.interner, &mut self.diagnostics, span)
    }

    /// Replaces each `defined(NAME)` and `defined NAME` with `1` or `0` before
    /// macro expansion runs over the condition.
    fn fold_defined(&mut self, toks: &[PpToken]) -> Vec<PpToken> {
        let mut out = Vec::with_capacity(toks.len());
        let mut i = 0;
        while i < toks.len() {
            let is_defined = matches!(toks[i].tok, Token::Ident(s)
                if self.interner.resolve(s) == "defined");
            if !is_defined {
                out.push(toks[i].clone());
                i += 1;
                continue;
            }
            let (name, width) = match toks.get(i + 1).map(|t| t.tok) {
                Some(Token::Punct(Punct::LParen)) => match (
                    toks.get(i + 2).map(|t| t.tok),
                    toks.get(i + 3).map(|t| t.tok),
                ) {
                    (Some(Token::Ident(n)), Some(Token::Punct(Punct::RParen))) => (Some(n), 4),
                    _ => (None, 1),
                },
                Some(Token::Ident(n)) => (Some(n), 2),
                _ => (None, 1),
            };
            match name {
                Some(n) => {
                    let value = u64::from(self.table.is_defined(n));
                    out.push(super::macros::int_token(self.interner, value, toks[i].span));
                    i += width;
                }
                None => {
                    out.push(toks[i].clone());
                    i += 1;
                }
            }
        }
        out
    }

    /////////////
    // PRAGMAS //
    /////////////

    fn do_pragma(&mut self, rest: &[PpToken], span: Span) {
        let Some(Token::Ident(name_sym)) = rest.first().map(|t| t.tok) else {
            self.warn_pragma("a #pragma needs a name", span);
            return;
        };
        let name = self.interner.resolve(name_sym).to_owned();
        // every standard pragma takes one parenthesised argument
        let arg = if rest.len() == 4 && rest[1].tok.is(Punct::LParen) && rest[3].tok.is(Punct::RParen)
        {
            Some(rest[2].clone())
        } else {
            None
        };
        match name.as_str() {
            "conductor" => self.do_pragma_conductor(arg, span),
            "fragment" => match self.pragma_word(arg.as_ref()).as_deref() {
                Some("finite") => self.fragment = Fragment::Finite,
                Some("stabilizer") => self.fragment = Fragment::Stabilizer,
                _ => self.warn_pragma("#pragma fragment takes `stabilizer` or `finite`", span),
            },
            "qalloc" => match self.pragma_word(arg.as_ref()).as_deref() {
                Some("reuse") => self.qalloc = QAlloc::Reuse,
                Some("linear") => self.qalloc = QAlloc::Linear,
                _ => self.warn_pragma("#pragma qalloc takes `linear` or `reuse`", span),
            },
            other => self.warn_pragma(
                &format!(
                    "unknown pragma `{other}`; this compiler understands `conductor`, \
                     `fragment` and `qalloc`"
                ),
                span,
            ),
        }
    }

    fn pragma_word(&self, arg: Option<&PpToken>) -> Option<String> {
        match arg.map(|t| t.tok) {
            Some(Token::Ident(s)) => Some(self.interner.resolve(s).to_owned()),
            _ => None,
        }
    }

    fn do_pragma_conductor(&mut self, arg: Option<PpToken>, span: Span) {
        let Some(arg) = arg else {
            self.warn_pragma("#pragma conductor takes one integer argument", span);
            return;
        };
        let Token::Int { raw, base, .. } = arg.tok else {
            // a malformed argument is a warning, not an error: the pragma is
            // ignored and translation carries on at the default conductor
            self.warn_pragma("#pragma conductor takes an integer", span);
            return;
        };
        let text = self.interner.resolve(raw).replace('_', "");
        let Ok(n) = u32::from_str_radix(&text, base.radix()) else {
            self.warn_pragma("the conductor does not fit in a 32-bit integer", span);
            return;
        };
        // an argument that is a perfectly good integer but not a usable
        // conductor is an error rather than a warning: unlike an unknown
        // pragma, this one was understood, and honouring it is impossible
        if n == 0 || n % 8 != 0 {
            self.error(
                Code::Eu04,
                &format!("{n} cannot be used as a conductor: it is not a positive multiple of 8"),
                arg.span,
                &[
                    "the conductor fixes which exact angles this unit can name, as \
                     fractions of a full turn",
                    "a multiple of 8 is required so that a quarter turn and an eighth \
                     turn are both expressible: that is what makes `i` and `1/sqrt(2)` \
                     exact rather than approximate",
                ],
                &["use 8 for the common case, or 16 or 24 if you need finer angles"],
            );
            return;
        }
        if n > Limit::Conductor.value() {
            self.error(
                Code::Eu04,
                &format!(
                    "a conductor of {n} is above the largest this implementation \
                     supports, which is {}",
                    Limit::Conductor.value()
                ),
                arg.span,
                &[
                    "the conductor bounds the arithmetic every exact coefficient in the \
                     unit is checked against, so it cannot grow without limit",
                ],
                &["use 8, 16 or 24"],
            );
            return;
        }
        self.conductor = n;
    }

    /// Warns of an unknown pragma, or a known one whose argument is
    /// malformed. A pragma may be meant for another tool, so it is dropped
    /// and translation goes on.
    fn warn_pragma(&mut self, message: &str, span: Span) {
        self.diagnostics.push(
            Diagnostic::new(Code::Ea03)
                .with_message(message)
                .at(span)
                .with_note(
                    "this pragma is ignored and translation continues; if it was meant \
                     for another tool, that is expected",
                ),
        );
    }

    ////////////////////////////////////
    // `#error`, `#warning`, `#embed` //
    ////////////////////////////////////

    fn do_message(&mut self, rest: &[PpToken], span: Span, fatal: bool) {
        let text = match rest.first().map(|t| t.tok) {
            Some(Token::Str { value, .. }) => self.interner.resolve(value).to_owned(),
            _ => rest
                .iter()
                .map(|t| spell(t.tok, self.interner))
                .collect::<Vec<_>>()
                .join(" "),
        };
        let mut d = Diagnostic::new(Code::Es03).with_message(text).at(span);
        if !fatal {
            d = d.as_warning();
        }
        self.diagnostics.push(d);
    }

    /// `#embed "path"` in directive position splices the document's tokens
    /// into the stream.
    fn do_embed(&mut self, rest: &[PpToken], span: Span) {
        let Some(Token::Str { value, .. }) = rest.first().map(|t| t.tok) else {
            self.error(Code::Es02, "#embed needs a quoted path", span, &[], &["write `#embed \"data.tcon\"`"]);
            return;
        };
        let path = self.interner.resolve(value).to_owned();
        let from = self
            .sources
            .get(self.source)
            .map_or_else(PathBuf::new, |f| f.path().to_owned());
        let text = match self.embed.read(&from, &path) {
            Ok(t) => t,
            Err(why) => {
                self.malformed(&format!("cannot read the embedded document: {why}"), span);
                return;
            }
        };
        // the spliced document is lexed in its own right; its tokens are
        // reported against the including directive, since it has no SourceId
        // of its own in this pass
        let spliced = crate::source::Spliced::from_text(&text);
        let lexed = crate::lex::lex(SourceId::SYNTHETIC, &spliced, self.interner);
        for d in lexed.diagnostics {
            self.diagnostics.push(
                Diagnostic::new(d.code)
                    .with_message(format!("in `{path}`: {}", d.message))
                    .at(span),
            );
        }
        self.out
            .extend(lexed.tokens.into_iter().map(|(t, _)| (t, span)));
        self.seen_any_token = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Spliced;

    struct Fixture {
        interner: Interner,
        sources: SourceMap,
        embed: MapEmbedResolver,
    }

    impl Fixture {
        fn new() -> Fixture {
            Fixture {
                interner: Interner::new(),
                sources: SourceMap::new(),
                embed: MapEmbedResolver::default(),
            }
        }

        fn run(&mut self, src: &str) -> Preprocessed {
            let id = self.sources.add_text("demo.tq", src);
            let file = self.sources.file(id);
            let spliced: Spliced = file.spliced().clone();
            let lexed = crate::lex::lex(id, &spliced, &self.interner);
            assert!(
                lexed.diagnostics.is_empty(),
                "lex errors: {:?}",
                lexed.diagnostics
            );
            preprocess(
                id,
                &lexed.tokens,
                &self.sources,
                &mut self.interner,
                &mut self.embed,
            )
        }

        /// Preprocesses and renders the resulting tokens for a readable
        /// assertion.
        fn text(&mut self, src: &str) -> String {
            let out = self.run(src);
            let rendered = out
                .tokens
                .iter()
                .map(|(t, _)| spell(*t, &self.interner))
                .collect::<Vec<_>>()
                .join(" ");
            assert!(
                !out.has_errors(),
                "unexpected errors for {src:?}: {:?}",
                out.diagnostics
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
            );
            rendered
        }

        fn codes(&mut self, src: &str) -> Vec<Code> {
            self.run(src).diagnostics.iter().map(|d| d.code).collect()
        }
    }

    ////////////////////////
    // THE UNIT DIRECTIVE //
    ////////////////////////

    #[test]
    fn a_classical_unit_directive_is_recorded() {
        let mut f = Fixture::new();
        let out = f.run("#unit classical\nlet x = 1;\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let u = out.unit.unwrap();
        assert_eq!(u.kind, UnitKind::Classical);
        assert!(u.initializer.is_none());
    }

    #[test]
    fn a_unit_initializer_name_is_recorded() {
        let mut f = Fixture::new();
        let out = f.run("#unit classical(setup)\nfn setup() {}\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let u = out.unit.unwrap();
        assert_eq!(u.kind, UnitKind::Classical);
        assert_eq!(
            f.interner.resolve(u.initializer.unwrap()),
            "setup",
            "the name in `#unit classical(...)` is the unit's initialiser"
        );
    }

    #[test]
    fn a_quantum_unit_directive_is_recorded() {
        let mut f = Fixture::new();
        let out = f.run("#unit quantum(build_tables)\n");
        assert_eq!(out.unit.unwrap().kind, UnitKind::Quantum);
    }

    #[test]
    fn a_missing_unit_directive_is_eu01() {
        let mut f = Fixture::new();
        assert!(f.codes("let x = 1;\n").contains(&Code::Eu01));
    }

    #[test]
    fn a_duplicate_unit_directive_is_eu01() {
        let mut f = Fixture::new();
        let codes = f.codes("#unit classical\n#unit quantum\n");
        assert!(codes.contains(&Code::Eu01));
    }

    #[test]
    fn a_misplaced_unit_directive_is_eu01() {
        // the directive comes before any item
        let mut f = Fixture::new();
        let codes = f.codes("let x = 1;\n#unit classical\n");
        assert!(codes.contains(&Code::Eu01));
    }

    #[test]
    fn a_unit_directive_naming_neither_kind_is_eu01() {
        let mut f = Fixture::new();
        assert!(f.codes("#unit hybrid\n").contains(&Code::Eu01));
        assert!(f.codes("#unit classical(\n").contains(&Code::Eu01));
    }

    #[test]
    fn comments_before_the_unit_directive_are_fine() {
        // only comments and whitespace may precede it
        let mut f = Fixture::new();
        let out = f.run("// a note\n\n#unit classical\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.unit.is_some());
    }

    //////////////////
    // KIND ALIASES //
    //////////////////

    #[test]
    fn an_alias_takes_the_side_for_the_units_kind() {
        let mut f = Fixture::new();
        assert_eq!(f.text("#unit classical\n#alias dit = bool | qubit\n"), "type dit = bool ;");
        assert_eq!(f.text("#unit quantum\n#alias dit = bool | qubit\n"), "type dit = qubit ;");
        assert_eq!(
            f.text("#unit quantum\n#alias word = u8 | quint < 8 >\n"),
            "type word = quint < 8 > ;"
        );
    }

    #[test]
    fn labelled_sides_come_in_either_order() {
        let mut f = Fixture::new();
        let src = "#alias dit quantum(qubit) classical(bool)\n";
        assert_eq!(f.text(&format!("#unit classical\n{src}")), "type dit = bool ;");
        assert_eq!(f.text(&format!("#unit quantum\n{src}")), "type dit = qubit ;");
        // a side may hold parentheses of its own
        assert_eq!(
            f.text("#unit classical\n#alias pair classical((u8, bool)) quantum((quint<8>, qubit))\n"),
            "type pair = ( u8 , bool ) ;"
        );
    }

    #[test]
    fn an_alias_keeps_its_generics_and_linkage() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit quantum\nstatic #alias bits<N: const usize> = [bool; N] | [qubit; N]\n"),
            "static type bits < N : const usize > = [ qubit ; N ] ;"
        );
    }

    #[test]
    fn a_side_left_out_gives_no_alias_in_that_kind() {
        let mut f = Fixture::new();
        assert_eq!(f.text("#unit quantum\n#alias dit = bool |\nlet x = 1;\n"), "let x = 1 ;");
        assert_eq!(f.text("#unit classical\n#alias dit quantum(qubit)\nlet x = 1;\n"), "let x = 1 ;");
        assert_eq!(f.text("#unit classical\n#alias dit = bool |\n"), "type dit = bool ;");
    }

    #[test]
    fn an_alias_side_expands_macros() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#define W 4\n#alias reg = [bool; W] | [qubit; W]\n"),
            "type reg = [ bool ; 4 ] ;"
        );
    }

    #[test]
    fn malformed_aliases_are_es02() {
        let mut f = Fixture::new();
        for src in [
            "#alias\n",
            "#alias dit bool\n",
            "#alias dit = bool\n",
            "#alias dit = bool | qubit | u8\n",
            "#alias dit classical(bool) classical(u8)\n",
            "#alias dit classical bool\n",
            "#alias dit classical(bool\n",
            "#alias dit\n",
        ] {
            assert_eq!(f.codes(&format!("#unit classical\n{src}")), [Code::Es02], "{src}");
        }
    }

    #[test]
    fn an_alias_before_the_unit_directive_is_eu01() {
        let mut f = Fixture::new();
        // the directive after it is misplaced too, being after an item
        assert_eq!(f.codes("#alias dit = bool | qubit\n#unit classical\n")[0], Code::Eu01);
    }

    #[test]
    fn the_kind_constants_name_the_kinds() {
        let mut f = Fixture::new();
        assert_eq!(f.text("#unit quantum\nlet k = __UNIT_KIND__ == __QUANTUM__;\n"), "let k = 2 == 2 ;");
        assert_eq!(f.text("#unit classical\nlet k = __CLASSICAL__;\n"), "let k = 1 ;");
        assert_eq!(f.text("#unit classical\n#if __UNIT_KIND__ == __QUANTUM__\nlet q = 1;\n#endif\n"), "");
        assert_eq!(f.codes("#unit classical\n#define __QUANTUM__ 3\n"), [Code::Ea04]);
    }

    ////////////
    // MACROS //
    ////////////

    #[test]
    fn an_object_like_macro_is_defined_and_used() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#define LIMIT 64\nlet n = LIMIT;\n"),
            "let n = 64 ;"
        );
    }

    #[test]
    fn a_function_like_macro_is_defined_and_used() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#define ADD(a,b) a + b\nlet n = ADD(1,2);\n"),
            "let n = 1 + 2 ;"
        );
    }

    #[test]
    fn a_space_before_the_paren_makes_an_object_like_macro() {
        // the `(` must follow the name immediately for the macro to take
        // parameters
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#define F (x) x\nlet n = F;\n"),
            "let n = ( x ) x ;"
        );
    }

    #[test]
    fn undef_removes_a_macro() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#define X 1\n#undef X\nlet n = X;\n"),
            "let n = X ;"
        );
    }

    #[test]
    fn a_directive_continued_over_a_spliced_line_is_one_directive() {
        // this is what logical-line numbering exists for
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#define SUM 1 \\\n  + 2\nlet n = SUM;\n"),
            "let n = 1 + 2 ;"
        );
    }

    #[test]
    fn static_define_gives_a_macro_unit_linkage() {
        // `static #define` keeps a macro private to its unit
        let mut f = Fixture::new();
        let out = f.run("#unit classical\n#define PUB 1\nstatic #define PRIV 2\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let exported: Vec<&str> = out
            .macros
            .exported()
            .iter()
            .map(|d| f.interner.resolve(d.name))
            .collect();
        assert_eq!(exported, vec!["PUB"]);
        assert!(out.macros.is_defined(f.interner.intern("PRIV")));
    }

    #[test]
    fn redefining_a_predefined_macro_is_ea04() {
        let mut f = Fixture::new();
        assert!(
            f.codes("#unit classical\n#define __FILE__ \"x\"\n")
                .contains(&Code::Ea04)
        );
        assert!(
            f.codes("#unit classical\n#undef __LINE__\n")
                .contains(&Code::Ea04)
        );
    }

    #[test]
    fn the_predefined_macros_are_available_after_the_unit_directive() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit quantum\nlet k = __UNIT_KIND__;\nlet u = __UNIT__;\n"),
            "let k = 2 ; let u = \"demo\" ;"
        );
    }

    #[test]
    fn line_reports_the_logical_line_it_appears_on() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\nlet a = __LINE__;\nlet b = __LINE__;\n"),
            "let a = 2 ; let b = 3 ;"
        );
    }

    //////////////////
    // CONDITIONALS //
    //////////////////

    #[test]
    fn an_if_selects_one_branch() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit classical\n#if 1\nlet a = 1;\n#else\nlet b = 2;\n#endif\n"),
            "let a = 1 ;"
        );
        assert_eq!(
            f.text("#unit classical\n#if 0\nlet a = 1;\n#else\nlet b = 2;\n#endif\n"),
            "let b = 2 ;"
        );
    }

    #[test]
    fn elif_chains_take_the_first_true_branch() {
        let mut f = Fixture::new();
        let src = "#unit classical\n\
                   #if 0\nlet a = 1;\n\
                   #elif 1\nlet b = 2;\n\
                   #elif 1\nlet c = 3;\n\
                   #else\nlet d = 4;\n#endif\n";
        assert_eq!(f.text(src), "let b = 2 ;");
    }

    #[test]
    fn ifdef_and_ifndef_consult_the_macro_table() {
        let mut f = Fixture::new();
        let src = "#unit classical\n#define FEATURE 1\n\
                   #ifdef FEATURE\nlet a = 1;\n#endif\n\
                   #ifndef FEATURE\nlet b = 2;\n#endif\n\
                   #ifndef MISSING\nlet c = 3;\n#endif\n";
        assert_eq!(f.text(src), "let a = 1 ; let c = 3 ;");
    }

    #[test]
    fn defined_is_not_expanded_before_it_is_asked() {
        // `defined(X)` must see the name, not X's expansion
        let mut f = Fixture::new();
        let src = "#unit classical\n#define X 0\n\
                   #if defined(X)\nlet yes = 1;\n#else\nlet no = 2;\n#endif\n";
        assert_eq!(f.text(src), "let yes = 1 ;");
    }

    #[test]
    fn macros_are_expanded_inside_a_condition() {
        let mut f = Fixture::new();
        let src = "#unit classical\n#define N 3\n\
                   #if N > 2\nlet big = 1;\n#endif\n";
        assert_eq!(f.text(src), "let big = 1 ;");
    }

    #[test]
    fn nested_conditionals_nest() {
        let mut f = Fixture::new();
        let src = "#unit classical\n\
                   #if 1\n#if 0\nlet a = 1;\n#else\nlet b = 2;\n#endif\n#endif\n";
        assert_eq!(f.text(src), "let b = 2 ;");
    }

    #[test]
    fn an_inactive_outer_region_suppresses_its_inner_branches() {
        let mut f = Fixture::new();
        let src = "#unit classical\n\
                   #if 0\n#if 1\nlet a = 1;\n#else\nlet b = 2;\n#endif\n#endif\nlet c = 3;\n";
        assert_eq!(f.text(src), "let c = 3 ;");
    }

    #[test]
    fn directives_in_an_inactive_region_do_not_run() {
        let mut f = Fixture::new();
        let src = "#unit classical\n#if 0\n#define HIDDEN 1\n#endif\nlet n = HIDDEN;\n";
        assert_eq!(f.text(src), "let n = HIDDEN ;");
    }

    #[test]
    fn an_unbalanced_conditional_is_reported() {
        let mut f = Fixture::new();
        assert!(f.run("#unit classical\n#if 1\nlet a = 1;\n").has_errors());
        assert!(f.run("#unit classical\n#endif\n").has_errors());
        assert!(f.run("#unit classical\n#else\n").has_errors());
        assert!(
            f.run("#unit classical\n#if 1\n#else\n#else\n#endif\n")
                .has_errors()
        );
    }

    /////////////
    // PRAGMAS //
    /////////////

    #[test]
    fn the_conductor_defaults_to_eight() {
        let mut f = Fixture::new();
        assert_eq!(f.run("#unit quantum\n").conductor, 8);
    }

    #[test]
    fn pragma_conductor_sets_the_conductor() {
        let mut f = Fixture::new();
        for n in [8u32, 16, 24] {
            let src = format!("#unit quantum\n#pragma conductor({n})\n");
            let out = f.run(&src);
            assert!(!out.has_errors(), "{:?}", out.diagnostics);
            assert_eq!(out.conductor, n);
        }
    }

    #[test]
    fn a_non_conforming_conductor_is_eu04() {
        // a conductor is a positive multiple of eight, at most 24
        let mut f = Fixture::new();
        for n in [7u32, 12, 0, 32, 64] {
            let src = format!("#unit quantum\n#pragma conductor({n})\n");
            assert!(
                f.codes(&src).contains(&Code::Eu04),
                "conductor {n} should be EU04"
            );
        }
    }

    #[test]
    fn a_malformed_pragma_argument_is_an_ea03_warning_and_is_ignored() {
        // an unknown pragma is dropped with a warning
        let mut f = Fixture::new();
        let out = f.run("#unit quantum\n#pragma conductor(\"eight\")\n");
        assert!(!out.has_errors(), "a malformed pragma is a warning");
        assert!(out.diagnostics.iter().any(|d| d.code == Code::Ea03));
        assert_eq!(out.conductor, 8, "the pragma is ignored");
    }

    #[test]
    fn an_unknown_pragma_is_an_ea03_warning() {
        let mut f = Fixture::new();
        let out = f.run("#unit classical\n#pragma nonesuch(1)\n");
        assert!(!out.has_errors());
        assert!(out.diagnostics.iter().any(|d| d.code == Code::Ea03));
    }

    #[test]
    fn pragma_fragment_and_qalloc_are_recognized() {
        let mut f = Fixture::new();
        let out = f.run("#unit quantum\n#pragma fragment(stabilizer)\n#pragma qalloc(linear)\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.fragment, Fragment::Stabilizer);
        assert_eq!(out.qalloc, QAlloc::Linear);

        let out = f.run("#unit quantum\n");
        assert_eq!(out.fragment, Fragment::Finite, "finite is the default");
        assert_eq!(out.qalloc, QAlloc::Reuse, "reuse is the default");
    }

    #[test]
    fn conductor_expands_to_the_pragma_value() {
        let mut f = Fixture::new();
        assert_eq!(
            f.text("#unit quantum\n#pragma conductor(24)\nlet n = __CONDUCTOR__;\n"),
            "let n = 24 ;"
        );
    }

    ////////////////////////////
    // MESSAGES AND INCLUSION //
    ////////////////////////////

    #[test]
    fn error_and_warning_carry_the_programs_own_text() {
        let mut f = Fixture::new();
        let out = f.run("#unit classical\n#error \"this build is unsupported\"\n");
        assert!(out.has_errors());
        assert_eq!(out.diagnostics[0].code, Code::Es03);
        assert_eq!(out.diagnostics[0].message, "this build is unsupported");

        let out = f.run("#unit classical\n#warning \"deprecated\"\n");
        assert!(!out.has_errors(), "#warning does not stop translation");
        assert_eq!(out.diagnostics[0].message, "deprecated");
    }

    #[test]
    fn a_message_in_an_inactive_branch_does_not_fire() {
        let mut f = Fixture::new();
        let out = f.run("#unit classical\n#if 0\n#error \"boom\"\n#endif\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
    }

    #[test]
    fn embed_splices_a_documents_tokens() {
        // in directive position, `#embed` splices tokens into the stream
        let mut f = Fixture::new();
        f.embed
            .files
            .insert("table.tcon".to_owned(), "1 , 2 , 3".to_owned());
        assert_eq!(
            f.text("#unit classical\nlet t = [\n#embed \"table.tcon\"\n];\n"),
            "let t = [ 1 , 2 , 3 ] ;"
        );
    }

    #[test]
    fn a_missing_embedded_document_is_reported() {
        let mut f = Fixture::new();
        assert!(f.run("#unit classical\n#embed \"gone.tcon\"\n").has_errors());
        assert!(f.run("#unit classical\n#embed nope\n").has_errors());
    }

    #[test]
    fn include_is_rejected_with_an_explanation() {
        // leaving `#include` out is a decision, not an oversight, so the
        // diagnostic names it and points at the two features that replace it
        // rather than calling the directive unknown
        let mut f = Fixture::new();
        let out = f.run("#unit classical\n#include \"other.tq\"\n");
        assert!(out.has_errors());
        assert!(
            out.diagnostics[0].message.contains("no `#include`"),
            "{}",
            out.diagnostics[0].message
        );
        let helps = out.diagnostics[0].helps.join(" ");
        assert!(helps.contains("import"), "{helps}");
        assert!(helps.contains("#embed"), "{helps}");
    }

    #[test]
    fn an_unknown_directive_is_reported() {
        let mut f = Fixture::new();
        let out = f.run("#unit classical\n#nonesuch 1\n");
        assert!(out.has_errors());
    }

    #[test]
    fn a_hash_that_does_not_open_a_line_is_not_a_directive() {
        // it reaches the parser, which has no rule for it; the preprocessor
        // must not mistake it for a directive
        let mut f = Fixture::new();
        let out = f.run("#unit classical\nlet a = 1; # b\n");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.tokens.iter().any(|(t, _)| t.is(Punct::Hash)));
    }

    #[test]
    fn a_worked_example_preprocesses() {
        // the serialiser idiom: a macro that stringises a field name and
        // pastes it onto an accessor
        let mut f = Fixture::new();
        let src = "#unit classical\n\
                   #define EMIT_FIELD(FIELD, NAME, TY, IDX) \
                   tcon::emit_pair(out, NAME, &self.FIELD);\n\
                   fn go() { EMIT_FIELD(x, \"x\", f64, 0) }\n";
        assert_eq!(
            f.text(src),
            "fn go ( ) { tcon :: emit_pair ( out , \"x\" , & self . x ) ; }"
        );
    }
}
