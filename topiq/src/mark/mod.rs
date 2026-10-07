//! Deciding which `[` opens an annotation.
//!
//! Nothing introduces an annotation but the bracket itself, so whether a `[`
//! starts an annotation or an array is decided by **position**, before
//! parsing: the answer changes what the parser is looking at.
//!
//! A `[` opens an annotation group when both of these hold:
//!
//! 1. it stands where an item, a member declaration, a statement-level
//!    declaration or a parameter may begin; and
//! 2. the next token is a name followed by `]`, `:` or `;`.
//!
//! Test 1 is a grammatical question, answered by the context stack in [`ctx`].
//! Test 2 is plain lookahead. Where both hold the `[` is retagged
//! [`Punct::LBracketAnnot`], so the parser reads a token whose meaning is
//! already settled.
//!
//! # Where `EA02` fires
//!
//! In **item position** a `[` that fails test 2 is refused outright, with a
//! diagnostic showing both readings, rather than being quietly read as an
//! array. Elsewhere (in a statement, a member or a parameter position) a
//! failing test 2 falls back to the array reading.
//!
//! Inside a block, `{ [1, 2, 3]; }` is plainly an array. At unit scope an
//! array cannot stand, so `[1, 2, 3];` is a mistaken annotation or nothing,
//! and asking beats guessing.
//!
//! The other positional rule, that a statement beginning with `if`, `match`,
//! `loop`, `while` or `{` ends at its closing `}`, lives in the statement
//! parser.

pub mod ctx;

use crate::diag::{Code, Diagnostic, DepthGuard, Limit};
use crate::lex::{Punct, Token};
use crate::span::Span;

pub use ctx::{Arm, Ctx};

/// The token stream with every `[` decided.
#[derive(Clone, Debug)]
pub struct Marked {
    /// Tokens, with annotation brackets retagged [`Punct::LBracketAnnot`].
    pub tokens: Vec<(Token, Span)>,
    /// Anything diagnosed along the way.
    pub diagnostics: Vec<Diagnostic>,
}

impl Marked {
    /// Whether marking produced any error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }

    /// How many annotation groups were found.
    pub fn annotation_count(&self) -> usize {
        self.tokens
            .iter()
            .filter(|(t, _)| t.is(Punct::LBracketAnnot))
            .count()
    }
}

/// One open bracket.
#[derive(Clone, Copy, Debug)]
struct Frame {
    ctx: Ctx,
    span: Span,
}

/// Runs the marking pass over a preprocessed token stream.
///
/// The input may be unbalanced (the parser reports that properly), so this
/// pass never panics on a stray closer or an unclosed opener.
pub fn mark(tokens: &[(Token, Span)]) -> Marked {
    let mut m = Marker {
        tokens: tokens.to_vec(),
        stack: Vec::new(),
        prev: None,
        just_closed: None,
        arm: Arm::None,
        braces: DepthGuard::new(Limit::BlockNesting),
        parens: DepthGuard::new(Limit::ParenNesting),
        diagnostics: Vec::new(),
    };
    m.run();
    Marked {
        tokens: m.tokens,
        diagnostics: m.diagnostics,
    }
}

struct Marker {
    tokens: Vec<(Token, Span)>,
    stack: Vec<Frame>,
    prev: Option<Token>,
    /// What a closing bracket just popped, so the next token can tell an
    /// annotation group's `]` from an index's.
    just_closed: Option<Ctx>,
    arm: Arm,
    braces: DepthGuard,
    parens: DepthGuard,
    diagnostics: Vec<Diagnostic>,
}

impl Marker {
    fn top(&self) -> Ctx {
        self.stack.last().map_or(Ctx::Unit, |f| f.ctx)
    }

    fn run(&mut self) {
        let mut i = 0;
        while i < self.tokens.len() {
            let (tok, span) = self.tokens[i];
            let admits = self.top().admits_declaration(self.prev, self.just_closed);
            self.just_closed = None;

            match tok {
                Token::Punct(Punct::LBracket) => self.open_bracket(i, span, admits),
                Token::Punct(Punct::LParen) => self.open_paren(i, span),
                Token::Punct(Punct::LBrace) => self.open_brace(span),
                Token::Punct(Punct::RParen | Punct::RBracket | Punct::RBrace) => {
                    self.close(tok);
                }
                _ => {}
            }

            // update the keyword arming that decides what the next bracket is
            self.arm = match Arm::of(tok) {
                Some(a) => a,
                None if self.arm.survives(tok) => self.arm,
                None => Arm::None,
            };
            self.prev = Some(self.tokens[i].0);
            i += 1;
        }

        for frame in std::mem::take(&mut self.stack) {
            self.diagnostics.push(
                Diagnostic::new(Code::Es02)
                    .with_message(format!("unclosed `{}`", opener_of(frame.ctx)))
                    .at(frame.span)
                    .with_help(format!("add `{}`", frame.ctx.closer())),
            );
        }
    }

    /// Decides what one `[` means.
    fn open_bracket(&mut self, i: usize, span: Span, admits: bool) {
        let looks_like_annotation = self.annotation_lookahead(i);
        if admits && looks_like_annotation {
            self.tokens[i].0 = Token::Punct(Punct::LBracketAnnot);
            self.stack.push(Frame {
                ctx: Ctx::AnnotGroup,
                span,
            });
            return;
        }
        if admits && self.top_is_item_position() {
            self.report_ambiguous(span);
        }
        self.stack.push(Frame {
            ctx: Ctx::BracketExpr,
            span,
        });
    }

    /// The lookahead test: the next token is a name followed by `]`, `:` or
    /// `;`.
    ///
    /// A keyword counts as a name here: annotation names have a name space of
    /// their own, and `[cover: c]` and `[gauge: g]` are spelt with reserved
    /// words.
    fn annotation_lookahead(&self, open: usize) -> bool {
        let names = matches!(
            self.tokens.get(open + 1).map(|(t, _)| *t),
            Some(Token::Ident(_) | Token::Kw(_))
        );
        if !names {
            return false;
        }
        matches!(
            self.tokens.get(open + 2).map(|(t, _)| *t),
            Some(Token::Punct(Punct::RBracket | Punct::Colon | Punct::Semi))
        )
    }

    /// Whether the current context is an *item* position (unit scope, an
    /// `impl` body or a `locale` body), which is the only place a `[` that
    /// fails the lookahead is refused outright rather than read as an array.
    fn top_is_item_position(&self) -> bool {
        matches!(self.top(), Ctx::Unit | Ctx::ImplBody | Ctx::LocaleBody)
    }

    fn report_ambiguous(&mut self, span: Span) {
        self.diagnostics.push(
            Diagnostic::new(Code::Ea02)
                .with_message("this `[` could open either an annotation or an array, and it is not clear which was meant")
                .at(span)
                .with_note(
                    "a `[` at the start of an item opens an annotation only when what \
                     follows is a name and then `]`, `:` or `;`, as in `[packed]`, \
                     `[align: 8]` or `[expect: monic; contract = flat]`",
                )
                .with_note(
                    "what follows this one does not have that shape, so it reads just as \
                     easily as an array type or an array literal",
                )
                .with_note(
                    "the compiler will not choose for you here: the two readings produce \
                     different programs, and picking the wrong one would translate \
                     something you did not write",
                )
                .with_help(
                    "if you meant an array, wrap it in parentheses (`([a; 5])`) or move \
                     it inside a function body, where a `[` is always an array",
                )
                .with_help(
                    "if you meant an annotation, check the name and the punctuation after \
                     it: `[name]`, `[name: argument]`",
                ),
        );
    }

    fn open_paren(&mut self, i: usize, span: Span) {
        if let Some(d) = self.parens.enter(span) {
            self.diagnostics.push(d);
        }
        let ctx = if matches!(self.prev, Some(Token::MacroName(_))) {
            // `@macro(...)` arguments are token soup until phase 7
            Ctx::MacroArgs
        } else if self.arm == Arm::Fn {
            // a `fn` group is a parameter list, unless it is a bare `fn (`
            // that no `with` follows, which makes it a function type
            let bare = matches!(self.prev, Some(Token::Kw(crate::lex::Keyword::Fn)));
            if bare && !ctx::is_closure_params(&self.tokens, i) {
                Ctx::TypeList
            } else {
                Ctx::ParamList
            }
        } else {
            Ctx::ExprParen
        };
        self.stack.push(Frame { ctx, span });
    }

    fn open_brace(&mut self, span: Span) {
        if let Some(d) = self.braces.enter(span) {
            self.diagnostics.push(d);
        }
        let ctx = self.arm.brace_ctx();
        self.stack.push(Frame { ctx, span });
    }

    fn close(&mut self, tok: Token) {
        // pop the innermost frame this token could close. a stray closer with
        // nothing to match is ignored here and reported by the parser
        let Some(frame) = self.stack.last().copied() else {
            return;
        };
        if !tok.is(frame.ctx.closer()) {
            return;
        }
        self.stack.pop();
        match frame.ctx.closer() {
            Punct::RBrace => self.braces.leave(),
            Punct::RParen => self.parens.leave(),
            _ => {}
        }
        self.just_closed = Some(frame.ctx);
    }
}

fn opener_of(ctx: Ctx) -> &'static str {
    match ctx.closer() {
        Punct::RBrace => "{",
        Punct::RParen => "(",
        _ => "[",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::lex::token::spell;
    use crate::source::Spliced;
    use crate::span::SourceId;

    struct Fixture {
        interner: Interner,
    }

    impl Fixture {
        fn new() -> Fixture {
            Fixture {
                interner: Interner::new(),
            }
        }

        fn mark(&mut self, src: &str) -> Marked {
            let spliced = Spliced::from_text(src);
            let lexed = crate::lex::lex(SourceId(0), &spliced, &self.interner);
            assert!(lexed.diagnostics.is_empty(), "{:?}", lexed.diagnostics);
            mark(&lexed.tokens)
        }

        /// Renders the marked stream with annotation brackets shown as `@[`,
        /// so an assertion reads at a glance.
        fn shape(&mut self, src: &str) -> String {
            let out = self.mark(src);
            out.tokens
                .iter()
                .map(|(t, _)| {
                    if t.is(Punct::LBracketAnnot) {
                        "@[".to_owned()
                    } else {
                        spell(*t, &self.interner)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn annotations(&mut self, src: &str) -> usize {
            self.mark(src).annotation_count()
        }
    }

    ////////////////////////////
    // ANNOTATIONS RECOGNISED //
    ////////////////////////////

    #[test]
    fn an_annotation_before_an_item_is_recognized() {
        let mut f = Fixture::new();
        assert_eq!(f.shape("[entry]\nfn f() {}"), "@[ entry ] fn f ( ) { }");
    }

    #[test]
    fn an_annotation_with_arguments_is_recognized() {
        // annotations on a structure and on a function
        let mut f = Fixture::new();
        assert_eq!(f.annotations("[export: \"sym\"]\nfn f() {}"), 1);
        assert_eq!(f.annotations("[align: 16]\nstruct S { x: u8 }"), 1);
        assert_eq!(f.annotations("[inline: never]\nfn f() {}"), 1);
    }

    #[test]
    fn stacked_annotation_groups_are_all_recognized() {
        // three groups stacked on one operator
        let mut f = Fixture::new();
        let src = "[entry]\n[cover: Orb]\n[expect: contract = cyc]\nfn iterate(r: u8) {}";
        assert_eq!(f.annotations(src), 3);
    }

    #[test]
    fn several_annotations_in_one_group_are_one_bracket() {
        // several annotations may share one group,
        // separated by `;`
        let mut f = Fixture::new();
        assert_eq!(f.annotations("[packed; align: 8]\nstruct Header { tag: u8 }"), 1);
    }

    #[test]
    fn an_annotation_name_may_be_spelled_like_a_keyword() {
        // annotation names have a name space of their own, and two of the
        // standard annotations are spelt `cover` and `gauge`, which are also
        // reserved words. they still have to mark
        let mut f = Fixture::new();
        assert_eq!(f.annotations("[cover: B2]\nfn f() {}"), 1);
        assert_eq!(f.annotations("[gauge: GB]\nfn f() {}"), 1);
        // and the annotation name does not stop `cover` being a keyword
        // elsewhere: this is a cover *declaration*, not an annotation
        assert_eq!(f.annotations("cover B2 = fin{ x };"), 0);
    }

    #[test]
    fn an_annotation_on_a_structure_field_is_recognized() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("struct S { [align: 4] x: u8, y: u8 }"), 1);
    }

    #[test]
    fn an_annotation_on_a_parameter_is_recognized() {
        // `[cover: c]` may attach to a parameter, not only to an item
        let mut f = Fixture::new();
        assert_eq!(f.annotations("fn f([cover: B] r: u8, x: u8) {}"), 1);
        assert_eq!(f.annotations("fn f(a: u8, [cover: B] r: u8) {}"), 1);
    }

    #[test]
    fn an_annotation_on_a_statement_declaration_is_recognized() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("fn f() { [test] let x = 1; }"), 1);
    }

    #[test]
    fn an_annotation_inside_an_impl_body_is_recognized() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("impl T { [inline] fn m(self: T) {} }"), 1);
    }

    ///////////////////////////////////////
    // BRACKETS THAT ARE NOT ANNOTATIONS //
    ///////////////////////////////////////

    #[test]
    fn an_array_literal_in_an_initializer_is_not_an_annotation() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("let v = [1, 2, 3];"), 0);
        assert_eq!(f.annotations("let v = [x];"), 0);
    }

    #[test]
    fn an_array_type_is_not_an_annotation() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("let q: [qubit; 2] = r;"), 0);
        assert_eq!(f.annotations("fn f(r: [qubit; 4]) {}"), 0);
        assert_eq!(f.annotations("struct Reg { q: [qubit; N] }"), 0);
    }

    #[test]
    fn an_index_suffix_is_not_an_annotation() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("fn f() { let a = xs[i]; }"), 0);
        assert_eq!(f.annotations("fn f() { xs[0] = 1; }"), 0);
    }

    #[test]
    fn a_bracketed_expression_inside_parentheses_is_not_an_annotation() {
        // this is the case a previous-token rule gets wrong: the `(` opens an
        // expression, so no parameter may begin after it
        let mut f = Fixture::new();
        assert_eq!(f.annotations("fn f() { g(([x])); }"), 0);
        assert_eq!(f.annotations("fn f() { let y = ([x]); }"), 0);
        assert_eq!(f.annotations("fn f() { h([x]); }"), 0);
    }

    #[test]
    fn a_bracket_after_a_value_block_is_an_index_not_an_annotation() {
        // `if c { a } else { b } [i]`: the `}` closed a block used as a value
        let mut f = Fixture::new();
        assert_eq!(f.annotations("fn f() { let x = match v { A => w } [i]; }"), 0);
    }

    #[test]
    fn a_bracket_in_macro_arguments_is_not_an_annotation() {
        let mut f = Fixture::new();
        assert_eq!(f.annotations("fn f() { @sizeof([u8; 4]); }"), 0);
    }

    #[test]
    fn a_function_types_bracketed_argument_is_not_an_annotation() {
        // `fn(...)` in type position takes a type-list, where no parameter may
        // begin, so `[u8]` there is an array type
        let mut f = Fixture::new();
        assert_eq!(f.annotations("let f: fn([u8]) -> u8 = g;"), 0);
        assert_eq!(f.annotations("fn deutsch(oracle: fn(*qubit, *qubit)) {}"), 0);
    }

    #[test]
    fn a_closure_parameter_may_still_be_annotated() {
        // the same `fn (`, but a `with` follows, so it is a closure's params
        let mut f = Fixture::new();
        assert_eq!(f.annotations("let c = fn([inline] x: u8) with () { x };"), 1);
    }

    //////////
    // EA02 //
    //////////

    #[test]
    fn a_bracket_in_item_position_that_fails_the_lookahead_is_ea02() {
        let mut f = Fixture::new();
        let out = f.mark("[1, 2, 3]\nfn f() {}");
        assert!(out.diagnostics.iter().any(|d| d.code == Code::Ea02));
        let d = &out.diagnostics[0];
        assert!(!d.helps.is_empty(), "EA02 should suggest a fix");
        assert_eq!(
            d.notes.len(),
            3,
            "and should give the annotation shape, the array reading, and why \
             the compiler will not choose"
        );
    }

    #[test]
    fn ea02_does_not_fire_in_a_statement_position() {
        // only an item position is refused outright. inside a block a `[` is
        // always an array, so an array-literal statement is read as one rather
        // than being called ambiguous
        let mut f = Fixture::new();
        let out = f.mark("fn f() { [1, 2, 3]; }");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.annotation_count(), 0);
    }

    #[test]
    fn a_valid_annotation_in_item_position_does_not_trigger_ea02() {
        let mut f = Fixture::new();
        assert!(!f.mark("[entry]\nfn f() {}").has_errors());
        assert!(!f.mark("[packed]\nstruct S { x: u8 }").has_errors());
    }

    ////////////////////////
    // BALANCE AND LIMITS //
    ////////////////////////

    #[test]
    fn unbalanced_input_is_reported_without_panicking() {
        let mut f = Fixture::new();
        assert!(f.mark("fn f() { let x = 1;").has_errors());
        // a stray closer is left for the parser rather than reported twice
        let out = f.mark("fn f() { } }");
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
    }

    #[test]
    fn deeply_nested_blocks_report_em05() {
        // blocks nest 64 deep
        let mut f = Fixture::new();
        let src = format!("fn f() {}{}", "{".repeat(70), "}".repeat(70));
        let out = f.mark(&src);
        assert!(out.diagnostics.iter().any(|d| d.code == Code::Em05));
    }

    #[test]
    fn deeply_nested_parentheses_report_em05() {
        // parenthesised expressions nest 64 deep
        let mut f = Fixture::new();
        let src = format!("let x = {}1{};", "(".repeat(70), ")".repeat(70));
        let out = f.mark(&src);
        assert!(out.diagnostics.iter().any(|d| d.code == Code::Em05));
    }

    #[test]
    fn ordinary_nesting_is_within_the_limits() {
        let mut f = Fixture::new();
        let src = format!("let x = {}1{};", "(".repeat(60), ")".repeat(60));
        assert!(!f.mark(&src).has_errors());
    }

    /////////////////////
    // WORKED EXAMPLES //
    /////////////////////

    #[test]
    fn the_deutsch_example_marks_correctly() {
        // a quantum operator with stacked annotations
        let mut f = Fixture::new();
        let src = "\
enum Guess { Constant, Balanced }

[entry]
[cover: fin{ x, y }]
[expect: kernel.outcomes = Guess]
fn deutsch(oracle: fn(*qubit, *qubit)) -> Guess {
    let q: [qubit; 1] = prep(z);
    aux let a: qubit;
    x(&a); h(&a);
    let bits = measure q;
    if bits[0] { Guess::Balanced } else { Guess::Constant }
}";
        let out = f.mark(src);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(
            out.annotation_count(),
            3,
            "[entry], [cover: ...] and [expect: ...]"
        );
    }

    #[test]
    fn the_grover_example_marks_correctly() {
        // judgement declarations with covers and gauges
        let mut f = Fixture::new();
        let src = "\
[cover: Orb] [gauge: GOrb]
[expect: contract = cyc]
fn iterate(r: *[qubit; 2]) { oracle(r); diffuse(r); }";
        let out = f.mark(src);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.annotation_count(), 3);
    }

    #[test]
    fn the_prefix_hierarchy_example_has_no_annotations() {
        // a classical unit that is all types and indexing, with no
        // annotations at all
        let mut f = Fixture::new();
        let src = "\
struct Shape  { kind: u8, area: f64 }
struct Circle { kind: u8, area: f64, r: f64 }

fn total(xs: *[*Shape]) -> f64 {
    let acc = 0.0;
    for i in 0..xs.len() { acc = acc + xs[i].area; }
    acc
}";
        let out = f.mark(src);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.annotation_count(), 0);
    }

    #[test]
    fn an_empty_stream_marks_to_nothing() {
        let mut f = Fixture::new();
        let out = f.mark("");
        assert!(out.tokens.is_empty());
        assert!(!out.has_errors());
    }
}
