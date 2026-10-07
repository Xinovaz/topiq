//! Macro expansion.
//!
//! Expansion follows C99 semantics. Arguments are expanded before being
//! substituted, unless `#` or `##` operates on them; a macro is never
//! re-expanded within its own expansion; and a function-like macro is invoked
//! only when `(` follows its name immediately.
//!
//! Non-re-expansion uses hide sets ([`super::hide`]) rather than a "currently
//! expanding" stack, because only hide sets get the function-like case right
//! when an argument was itself partly expanded.
//!
//! # Termination
//!
//! Hide sets prevent self-recursion, but not expansion that grows: two macros
//! that each add a token per round never stop. So the expansion depth limit
//! applies too (`EM05`), backed by a step budget in case the depth measure is
//! fooled.
//!
//! Macros written `@name` expand much later, so [`Token::MacroName`] passes
//! through untouched.

use std::collections::VecDeque;

use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::{Interner, Symbol};
use crate::lex::{Punct, Token, token::spell};
use crate::source::SourceMap;
use crate::span::Span;

use super::hide::Hide;
use super::macros::{MacroDef, MacroTable, PpToken, int_token, str_token};

/// A hard ceiling on expansion steps, so that a pathological input fails with a
/// diagnostic rather than hanging. Well past anything the depth limit allows.
const STEP_BUDGET: usize = 2_000_000;

/// Expands macros in a token sequence.
pub struct Expander<'a> {
    /// The macro table.
    pub table: &'a mut MacroTable,
    /// The string table.
    pub interner: &'a Interner,
    /// Needed to compute `__LINE__` at a use site.
    pub sources: &'a SourceMap,
    /// Where diagnostics go.
    pub diagnostics: &'a mut Vec<Diagnostic>,
    /// The unit's conductor.
    pub conductor: u32,
    reported_depth: bool,
}

impl<'a> Expander<'a> {
    /// Builds an expander.
    pub fn new(
        table: &'a mut MacroTable,
        interner: &'a Interner,
        sources: &'a SourceMap,
        diagnostics: &'a mut Vec<Diagnostic>,
        conductor: u32,
    ) -> Expander<'a> {
        Expander {
            table,
            interner,
            sources,
            diagnostics,
            conductor,
            reported_depth: false,
        }
    }

    /// Fully expands `input`.
    pub fn expand(&mut self, input: Vec<PpToken>) -> Vec<PpToken> {
        let mut work: VecDeque<PpToken> = input.into();
        let mut out: Vec<PpToken> = Vec::new();
        let mut steps = 0usize;

        while let Some(t) = work.pop_front() {
            steps += 1;
            if steps > STEP_BUDGET {
                self.report_depth(t.span, u32::MAX);
                out.push(t);
                out.extend(work);
                break;
            }
            if self.over_depth(&t) {
                out.push(t);
                continue;
            }
            let Token::Ident(name) = t.tok else {
                out.push(t);
                continue;
            };
            if t.hide.contains(name) {
                out.push(t);
                continue;
            }
            let Some(def) = self.table.get(name).cloned() else {
                out.push(t);
                continue;
            };

            if def.predefined
                && let Some(rep) = self.expand_predefined(name, &t) {
                    out.push(rep);
                    continue;
                }

            if def.is_function_like() {
                // without a `(` straight after, the name is an ordinary
                // identifier
                if !matches!(work.front().map(|x| x.tok), Some(Token::Punct(Punct::LParen))) {
                    out.push(t);
                    continue;
                }
                match self.collect_args(&mut work, &def, &t) {
                    Some((args, close_hide)) => {
                        let hide = t.hide.intersect(&close_hide).add(name);
                        let body = self.substitute(&def, &args, &hide, t.span);
                        for tok in body.into_iter().rev() {
                            work.push_front(tok);
                        }
                    }
                    None => out.push(t),
                }
            } else {
                let hide = t.hide.add(name);
                let body: Vec<PpToken> = def
                    .body
                    .iter()
                    .map(|b| PpToken {
                        tok: b.tok,
                        span: t.span,
                        hide: b.hide.union(&hide),
                    })
                    .collect();
                let body = self.apply_pastes(body);
                for tok in body.into_iter().rev() {
                    work.push_front(tok);
                }
            }
        }
        out
    }

    fn over_depth(&mut self, t: &PpToken) -> bool {
        let depth = t.hide.len() as u32;
        if depth <= Limit::MacroDepth.value() {
            return false;
        }
        self.report_depth(t.span, depth);
        true
    }

    fn report_depth(&mut self, span: Span, depth: u32) {
        if self.reported_depth {
            return;
        }
        self.reported_depth = true;
        self.diagnostics
            .push(Limit::MacroDepth.exceeded(span, depth));
    }

    /// The two predefined macros whose value depends on where they are used
    /// rather than where the enclosing macro was defined.
    fn expand_predefined(&mut self, name: Symbol, at: &PpToken) -> Option<PpToken> {
        match self.interner.resolve(name) {
            "__LINE__" => {
                let line = self
                    .sources
                    .get(at.span.source)
                    .map_or(0, |f| f.line_col(at.span.start).0);
                let mut t = int_token(self.interner, u64::from(line), at.span);
                t.hide = at.hide.add(name);
                Some(t)
            }
            "__CONDUCTOR__" => {
                let mut t = int_token(self.interner, u64::from(self.conductor), at.span);
                t.hide = at.hide.add(name);
                Some(t)
            }
            _ => None,
        }
    }

    /// Reads `( arg, arg, ... )` from the front of `work`.
    ///
    /// Returns the unexpanded arguments and the hide set of the closing
    /// parenthesis, which the function-like rule intersects with.
    fn collect_args(
        &mut self,
        work: &mut VecDeque<PpToken>,
        def: &MacroDef,
        at: &PpToken,
    ) -> Option<(Vec<Vec<PpToken>>, Hide)> {
        work.pop_front(); // the `(`
        let mut args: Vec<Vec<PpToken>> = vec![Vec::new()];
        let mut depth = 0usize;
        let close_hide;
        loop {
            let Some(t) = work.pop_front() else {
                self.diagnostics.push(
                    Diagnostic::new(Code::Es02)
                        .with_message(format!(
                            "unterminated argument list for macro `{}`",
                            self.interner.resolve(def.name)
                        ))
                        .at(at.span),
                );
                return None;
            };
            match t.tok {
                Token::Punct(Punct::LParen) => {
                    depth += 1;
                    args.last_mut().unwrap().push(t);
                }
                Token::Punct(Punct::RParen) if depth == 0 => {
                    close_hide = t.hide.clone();
                    break;
                }
                Token::Punct(Punct::RParen) => {
                    depth -= 1;
                    args.last_mut().unwrap().push(t);
                }
                // a comma at the top level separates arguments, unless the
                // macro is variadic and every declared parameter is already
                // filled, in which case it belongs to the tail
                Token::Punct(Punct::Comma)
                    if depth == 0 && !(def.variadic && args.len() > def.arity()) =>
                {
                    args.push(Vec::new());
                }
                _ => args.last_mut().unwrap().push(t),
            }
        }

        // `F()` on a zero-parameter macro yields one empty argument; drop it so
        // the arity check reads naturally
        if args.len() == 1 && args[0].is_empty() && def.arity() == 0 {
            args.clear();
        }
        if !def.accepts(args.len()) {
            self.diagnostics.push(
                Diagnostic::new(Code::Es02)
                    .with_message(format!(
                        "macro `{}` takes {}{} argument{}, but {} were supplied",
                        self.interner.resolve(def.name),
                        if def.variadic { "at least " } else { "" },
                        def.arity(),
                        if def.arity() == 1 { "" } else { "s" },
                        args.len()
                    ))
                    .at(at.span),
            );
            return None;
        }
        Some((args, close_hide))
    }

    /// Substitutes arguments into a function-like macro's body.
    ///
    /// Follows C99: an argument operated on by `#` or adjacent to `##` is used
    /// unexpanded; every other use of a parameter is replaced by the fully
    /// expanded argument.
    fn substitute(
        &mut self,
        def: &MacroDef,
        args: &[Vec<PpToken>],
        hide: &Hide,
        use_span: Span,
    ) -> Vec<PpToken> {
        let params = def.params.clone().unwrap_or_default();
        let va_args = self.interner.intern_late("__VA_ARGS__");

        let arg_for = |sym: Symbol| -> Option<Vec<PpToken>> {
            if let Some(i) = params.iter().position(|p| *p == sym) {
                return Some(args.get(i).cloned().unwrap_or_default());
            }
            if def.variadic && sym == va_args {
                // the tail, with the separating commas restored
                let mut out: Vec<PpToken> = Vec::new();
                for (k, a) in args.iter().skip(params.len()).enumerate() {
                    if k > 0 {
                        out.push(PpToken::new(Token::Punct(Punct::Comma), use_span));
                    }
                    out.extend(a.iter().cloned());
                }
                return Some(out);
            }
            None
        };

        let body = &def.body;
        let mut out: Vec<PpToken> = Vec::new();
        let mut i = 0usize;
        while i < body.len() {
            let cur = &body[i];

            // `#param` stringises the unexpanded argument
            if cur.tok.is(Punct::Hash) {
                if let Some(Token::Ident(sym)) = body.get(i + 1).map(|t| t.tok)
                    && let Some(arg) = arg_for(sym) {
                        let text = self.spell_sequence(&arg);
                        let mut t = str_token(self.interner, &text, use_span);
                        t.hide = hide.clone();
                        out.push(t);
                        i += 2;
                        continue;
                    }
                self.diagnostics.push(
                    Diagnostic::new(Code::Es02)
                        .with_message("`#` in a macro body must be followed by a parameter")
                        .at(use_span),
                );
                i += 1;
                continue;
            }

            let next_is_paste = matches!(
                body.get(i + 1).map(|t| t.tok),
                Some(Token::Punct(Punct::HashHash))
            );
            let prev_was_paste = i >= 1 && body[i - 1].tok.is(Punct::HashHash);

            match cur.tok {
                Token::Ident(sym) if arg_for(sym).is_some() => {
                    let raw = arg_for(sym).unwrap();
                    let replacement = if next_is_paste || prev_was_paste {
                        raw
                    } else {
                        self.expand(raw)
                    };
                    for mut t in replacement {
                        t.span = use_span;
                        t.hide = t.hide.union(hide);
                        out.push(t);
                    }
                }
                _ => out.push(PpToken {
                    tok: cur.tok,
                    span: use_span,
                    hide: cur.hide.union(hide),
                }),
            }
            i += 1;
        }
        self.apply_pastes(out)
    }

    /// Applies every `##` in a substituted body.
    ///
    /// `a ## b` concatenates the spellings and re-lexes; a paste that does not
    /// yield exactly one token is an error rather than a silent split.
    fn apply_pastes(&mut self, tokens: Vec<PpToken>) -> Vec<PpToken> {
        if !tokens.iter().any(|t| t.tok.is(Punct::HashHash)) {
            return tokens;
        }
        let mut out: Vec<PpToken> = Vec::new();
        let mut i = 0usize;
        while i < tokens.len() {
            if tokens[i].tok.is(Punct::HashHash) && !out.is_empty() && i + 1 < tokens.len() {
                let left = out.pop().unwrap();
                let right = tokens[i + 1].clone();
                match self.paste(&left, &right) {
                    Some(t) => out.push(t),
                    None => {
                        out.push(left);
                        out.push(right);
                    }
                }
                i += 2;
            } else {
                out.push(tokens[i].clone());
                i += 1;
            }
        }
        out
    }

    /// Concatenates two tokens and re-lexes the result.
    fn paste(&mut self, left: &PpToken, right: &PpToken) -> Option<PpToken> {
        let text = format!(
            "{}{}",
            spell(left.tok, self.interner),
            spell(right.tok, self.interner)
        );
        let spliced = crate::source::Spliced::from_text(&text);
        let lexed = crate::lex::lex(crate::span::SourceId::SYNTHETIC, &spliced, self.interner);
        if lexed.tokens.len() == 1 && lexed.diagnostics.is_empty() {
            return Some(PpToken {
                tok: lexed.tokens[0].0,
                span: left.span,
                hide: left.hide.intersect(&right.hide),
            });
        }
        self.diagnostics.push(
            Diagnostic::new(Code::Es02)
                .with_message(format!(
                    "pasting these two tokens together gives `{text}`, which is not a \
                     single token"
                ))
                .at(left.span)
                .with_note(
                    "`##` joins the text on either side of it and the result must lex as \
                     exactly one token: `foo ## _bar` gives the identifier `foo_bar`, but \
                     `foo ## +` gives `foo+`, which is two",
                )
                .with_help(
                    "if you wanted the two tokens next to each other rather than joined, \
                     remove the `##`",
                ),
        );
        None
    }

    /// Spells a token sequence for `#`, preserving whether tokens were written
    /// adjacently.
    fn spell_sequence(&self, tokens: &[PpToken]) -> String {
        let mut out = String::new();
        let mut prev: Option<Span> = None;
        for t in tokens {
            if let Some(p) = prev
                && !p.adjacent_to(t.span) {
                    out.push(' ');
                }
            out.push_str(&spell(t.tok, self.interner));
            prev = Some(t.span);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::{Keyword, lex};
    use crate::pp::macros::Linkage;
    use crate::source::{SourceMap, Spliced};
    use crate::span::SourceId;

    /// Builds a table from `#define`-like descriptions, then expands `src`.
    struct Fixture {
        interner: Interner,
        sources: SourceMap,
        table: MacroTable,
        id: SourceId,
    }

    impl Fixture {
        fn new() -> Fixture {
            let mut sources = SourceMap::new();
            let id = sources.add_text("demo.tq", "");
            Fixture {
                interner: Interner::new(),
                sources,
                table: MacroTable::new(),
                id,
            }
        }

        fn tokens(&mut self, text: &str) -> Vec<PpToken> {
            let spliced = Spliced::from_text(text);
            let out = lex(self.id, &spliced, &self.interner);
            assert!(out.diagnostics.is_empty(), "lex errors in {text:?}");
            out.tokens
                .into_iter()
                .map(|(t, s)| PpToken::new(t, s))
                .collect()
        }

        /// `define("F(a,b)", "a + b")` or `define("X", "1")`.
        fn define(&mut self, head: &str, body: &str) {
            let (name, params, variadic) = match head.find('(') {
                None => (head.to_owned(), None, false),
                Some(i) => {
                    let name = head[..i].to_owned();
                    let inner = head[i + 1..head.len() - 1].trim().to_owned();
                    let variadic = inner.ends_with("...");
                    let ps: Vec<Symbol> = inner
                        .split(',')
                        .map(str::trim)
                        .filter(|p| !p.is_empty() && *p != "...")
                        .map(|p| self.interner.intern_late(p))
                        .collect();
                    (name, Some(ps), variadic)
                }
            };
            let name_sym = self.interner.intern_late(&name);
            let body = self.tokens(body);
            self.table.define(MacroDef {
                name: name_sym,
                params,
                variadic,
                body,
                linkage: Linkage::Program,
                span: Span::new(self.id, 0, 1),
                predefined: false,
            });
        }

        fn expand(&mut self, src: &str) -> (Vec<Token>, Vec<Diagnostic>) {
            let input = self.tokens(src);
            let mut diags = Vec::new();
            let out = {
                let mut e = Expander::new(
                    &mut self.table,
                    &self.interner,
                    &self.sources,
                    &mut diags,
                    8,
                );
                e.expand(input)
            };
            (out.into_iter().map(|t| t.tok).collect(), diags)
        }

        /// Expands and renders back to source text, for readable assertions.
        fn text(&mut self, src: &str) -> String {
            let (toks, diags) = self.expand(src);
            assert!(
                diags.is_empty(),
                "unexpected diagnostics: {:?}",
                diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
            );
            toks.iter()
                .map(|t| spell(*t, &self.interner))
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    #[test]
    fn an_object_like_macro_is_replaced() {
        let mut f = Fixture::new();
        f.define("LIMIT", "64");
        assert_eq!(f.text("let n = LIMIT ;"), "let n = 64 ;");
    }

    #[test]
    fn an_undefined_name_is_left_alone() {
        let mut f = Fixture::new();
        assert_eq!(f.text("let n = LIMIT ;"), "let n = LIMIT ;");
    }

    #[test]
    fn macros_expand_recursively() {
        let mut f = Fixture::new();
        f.define("A", "B");
        f.define("B", "C");
        f.define("C", "42");
        assert_eq!(f.text("A"), "42");
    }

    #[test]
    fn a_self_referential_macro_expands_once() {
        // a macro is never re-expanded within its own expansion
        let mut f = Fixture::new();
        f.define("X", "X");
        assert_eq!(f.text("X"), "X");
    }

    #[test]
    fn mutually_referential_macros_terminate() {
        let mut f = Fixture::new();
        f.define("P", "Q");
        f.define("Q", "P");
        assert_eq!(f.text("P"), "P");
    }

    #[test]
    fn a_macro_that_names_itself_inside_a_larger_body_terminates() {
        let mut f = Fixture::new();
        f.define("X", "( X + 1 )");
        assert_eq!(f.text("X"), "( X + 1 )");
    }

    #[test]
    fn a_function_like_macro_substitutes_its_arguments() {
        let mut f = Fixture::new();
        f.define("ADD(a,b)", "a + b");
        assert_eq!(f.text("ADD ( 1 , 2 )"), "1 + 2");
    }

    #[test]
    fn arguments_are_expanded_before_substitution() {
        let mut f = Fixture::new();
        f.define("ONE", "1");
        f.define("ID(x)", "x");
        assert_eq!(f.text("ID ( ONE )"), "1");
    }

    #[test]
    fn a_function_like_macro_without_a_paren_is_an_ordinary_identifier() {
        // without the `(`, the macro is not invoked at all
        let mut f = Fixture::new();
        f.define("F(a)", "a");
        assert_eq!(f.text("let g = F ;"), "let g = F ;");
    }

    #[test]
    fn nested_parentheses_and_commas_inside_arguments_are_respected() {
        let mut f = Fixture::new();
        f.define("FIRST(a,b)", "a");
        assert_eq!(f.text("FIRST ( g ( 1 , 2 ) , 3 )"), "g ( 1 , 2 )");
    }

    #[test]
    fn a_zero_parameter_macro_is_invoked_with_empty_parentheses() {
        let mut f = Fixture::new();
        f.define("NOW()", "42");
        assert_eq!(f.text("NOW ( )"), "42");
    }

    #[test]
    fn the_wrong_number_of_arguments_is_reported() {
        let mut f = Fixture::new();
        f.define("ADD(a,b)", "a + b");
        let (_, diags) = f.expand("ADD ( 1 )");
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("2 arguments"), "{}", diags[0].message);
    }

    #[test]
    fn an_unterminated_argument_list_is_reported() {
        let mut f = Fixture::new();
        f.define("ADD(a,b)", "a + b");
        let (_, diags) = f.expand("ADD ( 1 , 2");
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("unterminated"));
    }

    #[test]
    fn variadic_macros_collect_the_tail_into_va_args() {
        let mut f = Fixture::new();
        f.define("LOG(fmt,...)", "print ( fmt , __VA_ARGS__ )");
        assert_eq!(
            f.text("LOG ( \"x\" , 1 , 2 )"),
            "print ( \"x\" , 1 , 2 )"
        );
    }

    #[test]
    fn a_variadic_macro_accepts_an_empty_tail() {
        let mut f = Fixture::new();
        f.define("LOG(fmt,...)", "print ( fmt )");
        assert_eq!(f.text("LOG ( \"x\" )"), "print ( \"x\" )");
    }

    #[test]
    fn stringize_uses_the_unexpanded_argument() {
        let mut f = Fixture::new();
        f.define("ONE", "1");
        f.define("STR(x)", "# x");
        // the argument is stringised as written, not as expanded
        assert_eq!(f.text("STR ( ONE )"), "\"ONE\"");
    }

    #[test]
    fn stringize_preserves_written_spacing() {
        let mut f = Fixture::new();
        f.define("STR(x)", "# x");
        assert_eq!(f.text("STR ( a + b )"), "\"a + b\"");
        assert_eq!(f.text("STR ( a+b )"), "\"a+b\"");
    }

    #[test]
    fn paste_joins_two_tokens_into_one() {
        let mut f = Fixture::new();
        f.define("CAT(a,b)", "a ## b");
        assert_eq!(f.text("CAT ( foo , bar )"), "foobar");
        assert_eq!(f.text("CAT ( 1 , 2 )"), "12");
    }

    #[test]
    fn paste_uses_unexpanded_operands() {
        let mut f = Fixture::new();
        f.define("ONE", "1");
        f.define("CAT(a,b)", "a ## b");
        assert_eq!(f.text("CAT ( ONE , X )"), "ONEX");
    }

    #[test]
    fn the_result_of_a_paste_is_rescanned() {
        let mut f = Fixture::new();
        f.define("foobar", "99");
        f.define("CAT(a,b)", "a ## b");
        assert_eq!(f.text("CAT ( foo , bar )"), "99");
    }

    #[test]
    fn an_impossible_paste_is_reported_rather_than_split_silently() {
        let mut f = Fixture::new();
        f.define("CAT(a,b)", "a ## b");
        let (_, diags) = f.expand("CAT ( 1 , + )");
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(
            diags[0].message.contains("not a single token"),
            "{}",
            diags[0].message
        );
    }

    #[test]
    fn a_derive_like_pattern_expands() {
        // the intended composition point for generated code: a
        // #define that pastes an impl plus annotations
        let mut f = Fixture::new();
        f.define("EMIT_FIELD(FIELD,NAME,TY,IDX)", "emit_pair ( out , NAME , & self . FIELD )");
        assert_eq!(
            f.text("EMIT_FIELD ( x , \"x\" , f64 , 0 )"),
            "emit_pair ( out , \"x\" , & self . x )"
        );
    }

    #[test]
    fn builtin_at_macros_pass_through_untouched() {
        // `@`-macros are not preprocessor macros and pass through untouched
        let mut f = Fixture::new();
        f.define("T", "u32");
        assert_eq!(f.text("@sizeof ( T )"), "@sizeof ( u32 )");
    }

    #[test]
    fn line_expands_to_the_line_of_the_use_site() {
        let mut f = Fixture::new();
        let mut sources = SourceMap::new();
        let id = sources.add_text("demo.tq", "a\nb\nLINE_HERE\n");
        f.sources = sources;
        f.id = id;
        f.table
            .install_predefined(&mut f.interner, Span::new(id, 0, 1), 2, "demo.tq", "demo", 1);

        // a token on the third line
        let spliced = Spliced::from_text("a\nb\n__LINE__\n");
        let lexed = lex(id, &spliced, &f.interner);
        let input: Vec<PpToken> = lexed
            .tokens
            .into_iter()
            .map(|(t, s)| PpToken::new(t, s))
            .collect();
        let mut diags = Vec::new();
        let out = Expander::new(&mut f.table, &f.interner, &f.sources, &mut diags, 8)
            .expand(input);
        let spelled: Vec<String> = out.iter().map(|t| spell(t.tok, &f.interner)).collect();
        assert_eq!(spelled, vec!["a", "b", "3"]);
    }

    #[test]
    fn conductor_expands_to_the_units_conductor() {
        let mut f = Fixture::new();
        let id = f.id;
        f.table
            .install_predefined(&mut f.interner, Span::new(id, 0, 1), 2, "demo.tq", "demo", 2);
        let input = f.tokens("__CONDUCTOR__");
        let mut diags = Vec::new();
        let out = Expander::new(&mut f.table, &f.interner, &f.sources, &mut diags, 24)
            .expand(input);
        assert_eq!(spell(out[0].tok, &f.interner), "24");
    }

    #[test]
    fn the_static_predefined_macros_expand_from_the_table() {
        let mut f = Fixture::new();
        let id = f.id;
        f.table
            .install_predefined(&mut f.interner, Span::new(id, 0, 1), 2, "demo.tq", "demo", 2);
        assert_eq!(f.text("__TOPIQ__"), "2");
        assert_eq!(f.text("__UNIT__"), "\"demo\"");
        assert_eq!(f.text("__UNIT_KIND__"), "2");
        assert_eq!(f.text("__FILE__"), "\"demo.tq\"");
    }

    #[test]
    fn keywords_are_never_treated_as_macro_names() {
        let mut f = Fixture::new();
        let (toks, _) = f.expand("fn");
        assert_eq!(toks, vec![Token::Kw(Keyword::Fn)]);
    }

    #[test]
    fn a_deeply_nested_expansion_reports_em05_and_stops() {
        // macro expansion may nest 64 deep
        let mut f = Fixture::new();
        for k in 0..80 {
            let body = format!("M{}", k + 1);
            f.define(&format!("M{k}"), &body);
        }
        f.define("M80", "end");
        let (_, diags) = f.expand("M0");
        assert!(
            diags.iter().any(|d| d.code == Code::Em05),
            "expected EM05, got {:?}",
            diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn expansion_terminates_on_a_pathological_pair() {
        // hide sets alone do not bound growth here; the depth limit does
        let mut f = Fixture::new();
        f.define("A(x)", "B ( x )");
        f.define("B(x)", "A ( x )");
        let (_, _diags) = f.expand("A ( 1 )");
        // reaching this line at all is the assertion: it did not hang
    }
}
