//! The context stack that decides where an item, member, statement-level
//! declaration or parameter may begin.
//!
//! This is a *grammatical* question asked by a pass that runs before parsing,
//! so the pass has to carry enough state to answer it on its own. A rule
//! phrased only on the previous token ("a `[` after `{`, `}`, `;`, `(`, `,`
//! or `]`") is not enough: it misreads `([x])`, where the `(` opens a
//! parenthesised expression rather than a parameter list. This stack tells the
//! two apart.
//!
//! # Classifying an open parenthesis
//!
//! A `(` opens a parameter list only when it is reached through `fn`:
//!
//! | written | the `(` is |
//! |---|---|
//! | `fn name(` , `fn name<T>(` , `fn Type.name(` , `fn $op(` | a parameter list |
//! | `fn(` whose matching `)` is followed by `with` | a closure's parameter list |
//! | `fn(` otherwise | a function *type*'s argument list |
//! | anything else | an expression |
//!
//! A function type such as `fn(*qubit, *qubit)` holds types, where no
//! parameter may begin; a closure such as `fn(x: i64) -> i64 with (n) { x + n }`
//! holds parameters. Telling them apart takes a scan to the matching `)` and a
//! look past it.
//!
//! # Closing brackets carry why they were opened
//!
//! Popping a frame has to say what closed, because a `}` that ends an item body
//! returns to item position while a `}` that ended a block expression inside a
//! larger expression does not. Without that, `let x = if c { a } else { b } [i];`
//! misreads its index suffix as an annotation.

use crate::lex::{Keyword, Punct, Token};

/// What kind of bracketed region the pass is inside.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ctx {
    /// Unit scope, outside every bracket. Items may begin here.
    Unit,
    /// A `{ … }` block. Statements may begin here.
    Block,
    /// A `struct` body. Members may begin here.
    StructBody,
    /// An `enum` body. Variants may begin here.
    EnumBody,
    /// An `impl` body. Items may begin here.
    ImplBody,
    /// A `locale` body. Members may begin here.
    LocaleBody,
    /// A `qmap` body. Entries, not members, so nothing may begin here.
    QmapBody,
    /// A `match` body. Arms are patterns, not declarations.
    MatchBody,
    /// A `fn` parameter list. Parameters may begin here.
    ParamList,
    /// A function type's argument list, which holds types rather than
    /// parameters, so nothing may begin here.
    TypeList,
    /// A parenthesised expression, a call's arguments, or a tuple.
    ExprParen,
    /// The arguments of a builtin `@macro(…)`, which are token soup.
    MacroArgs,
    /// A `[ … ]` array type, array literal or index suffix.
    BracketExpr,
    /// A `[ … ]` annotation group.
    AnnotGroup,
}

impl Ctx {
    /// Whether an item, member, statement-level declaration or parameter may
    /// begin immediately after `prev` in this context.
    ///
    /// `closed` says what a closing bracket just popped, where `prev` is one.
    pub fn admits_declaration(self, prev: Option<Token>, closed: Option<Ctx>) -> bool {
        // a `]` that closed an annotation group always leaves a position where
        // the annotated thing begins: `[a] [b] fn f()` is legal, and so is
        // `[packed] struct S {}`
        if closed == Some(Ctx::AnnotGroup) {
            return true;
        }
        let Some(prev) = prev else {
            // the start of the file is item position
            return self == Ctx::Unit;
        };
        match self {
            Ctx::Unit => {
                prev.is(Punct::Semi) || closed.is_some_and(Ctx::is_brace) || prev.is(Punct::RBrace)
            }
            Ctx::Block => {
                prev.is(Punct::LBrace)
                    || prev.is(Punct::Semi)
                    // a `}` returns to statement position only when it closed
                    // something that stands as a statement on its own
                    || closed.is_some_and(Ctx::ends_a_statement)
            }
            Ctx::StructBody | Ctx::EnumBody => {
                prev.is(Punct::LBrace) || prev.is(Punct::Comma) || prev.is(Punct::Semi)
            }
            Ctx::ImplBody | Ctx::LocaleBody => {
                prev.is(Punct::LBrace) || prev.is(Punct::Semi) || prev.is(Punct::RBrace)
            }
            Ctx::ParamList => prev.is(Punct::LParen) || prev.is(Punct::Comma),
            // inside an expression, a type list, a match body, a qmap body, an
            // index, an annotation group or macro soup, nothing may begin
            Ctx::TypeList
            | Ctx::QmapBody
            | Ctx::MatchBody
            | Ctx::ExprParen
            | Ctx::MacroArgs
            | Ctx::BracketExpr
            | Ctx::AnnotGroup => false,
        }
    }

    /// Whether this context is delimited by braces.
    pub fn is_brace(self) -> bool {
        matches!(
            self,
            Ctx::Block
                | Ctx::StructBody
                | Ctx::EnumBody
                | Ctx::ImplBody
                | Ctx::LocaleBody
                | Ctx::QmapBody
                | Ctx::MatchBody
        )
    }

    /// Whether closing this context leaves a position where a statement may
    /// begin.
    ///
    /// A block that stood as a statement does; a block used as a value inside a
    /// larger expression does not, because the expression continues.
    fn ends_a_statement(self) -> bool {
        matches!(self, Ctx::Block | Ctx::ImplBody | Ctx::LocaleBody)
    }

    /// The punctuator that closes this context.
    pub fn closer(self) -> Punct {
        match self {
            c if c.is_brace() => Punct::RBrace,
            Ctx::ParamList | Ctx::TypeList | Ctx::ExprParen | Ctx::MacroArgs => Punct::RParen,
            _ => Punct::RBracket,
        }
    }
}

/// Tracks what an upcoming `{` or `(` will be, from the keyword that armed it.
///
/// Only a handful of keywords change the meaning of the next bracket, so this
/// is a small prefix automaton rather than a parser.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Arm {
    /// Nothing pending.
    #[default]
    None,
    /// A `fn` was seen; the next `(` is a parameter or type list.
    Fn,
    /// A `struct` was seen; the next `{` is a structure body.
    Struct,
    /// An `enum` was seen; the next `{` is an enumeration body.
    Enum,
    /// An `impl` was seen; the next `{` is an impl body.
    Impl,
    /// A `locale` was seen; the next `{` is a locale body.
    Locale,
    /// A `qmap` was seen; the next `{` is a map-locale body.
    Qmap,
    /// A `match` was seen; the next `{` is a match body.
    Match,
}

impl Arm {
    /// The arming that a token introduces.
    pub fn of(tok: Token) -> Option<Arm> {
        Some(match tok {
            Token::Kw(Keyword::Fn) => Arm::Fn,
            Token::Kw(Keyword::Struct) => Arm::Struct,
            Token::Kw(Keyword::Enum) => Arm::Enum,
            Token::Kw(Keyword::Impl) => Arm::Impl,
            Token::Kw(Keyword::Locale) => Arm::Locale,
            Token::Kw(Keyword::Qmap) => Arm::Qmap,
            Token::Kw(Keyword::Match) => Arm::Match,
            _ => return None,
        })
    }

    /// The body context a `{` opens under this arming.
    pub fn brace_ctx(self) -> Ctx {
        match self {
            Arm::Struct => Ctx::StructBody,
            Arm::Enum => Ctx::EnumBody,
            Arm::Impl => Ctx::ImplBody,
            Arm::Locale => Ctx::LocaleBody,
            Arm::Qmap => Ctx::QmapBody,
            Arm::Match => Ctx::MatchBody,
            Arm::None | Arm::Fn => Ctx::Block,
        }
    }

    /// Whether a token between the arming keyword and its bracket keeps the
    /// arming alive.
    ///
    /// Between `struct` and its `{` come a name and optional generics; between
    /// `fn` and its `(` come a name, a path, an operator-method name and
    /// optional generics. A token outside that set means the construct was
    /// something else and the arming is dropped.
    pub fn survives(self, tok: Token) -> bool {
        match self {
            Arm::None => false,
            Arm::Match => !matches!(tok, Token::Punct(Punct::Semi)),
            _ => matches!(
                tok,
                Token::Ident(_)
                    | Token::OpName(_)
                    | Token::Punct(
                        Punct::Lt
                            | Punct::Gt
                            | Punct::Comma
                            | Punct::Colon
                            | Punct::ColonColon
                            | Punct::Dot
                            | Punct::Star
                            | Punct::And
                            | Punct::LBracket
                            | Punct::RBracket
                            | Punct::Semi
                    )
                    | Token::Kw(Keyword::Const)
                    | Token::Int { .. }
            ),
        }
    }
}

/// Finds the index of the `)` matching an opening `(` at `open`.
///
/// Returns `None` if the input is unbalanced, which the marking pass tolerates
/// rather than treating as fatal (the parser will report it properly).
pub fn matching_paren(tokens: &[(Token, crate::span::Span)], open: usize) -> Option<usize> {
    debug_assert!(tokens[open].0.is(Punct::LParen));
    let mut depth = 0usize;
    for (i, (tok, _)) in tokens.iter().enumerate().skip(open) {
        if tok.is(Punct::LParen) {
            depth += 1;
        } else if tok.is(Punct::RParen) {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Whether a `fn (` at `open` opens a closure's parameter list rather than a
/// function type's argument list.
///
/// A closure expression is `fn ( params? ) ret? with ( captures? ) block`, so
/// the token after the matching `)` (skipping the return type) is `with`.
pub fn is_closure_params(tokens: &[(Token, crate::span::Span)], open: usize) -> bool {
    let Some(close) = matching_paren(tokens, open) else {
        return false;
    };
    // skip an optional `-> type` before looking for `with`. the return type
    // cannot itself contain a top-level `with`, so scanning to the first one is
    // safe, but stop at a delimiter that would end the construct
    for (tok, _) in tokens.iter().skip(close + 1) {
        match tok {
            Token::Kw(Keyword::With) => return true,
            Token::Punct(Punct::LBrace | Punct::Semi | Punct::Comma | Punct::RParen) => {
                return false;
            }
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::{Interner, Symbol};

    fn ident() -> Token {
        Token::Ident(Symbol::EMPTY)
    }

    fn p(p: Punct) -> Token {
        Token::Punct(p)
    }

    #[test]
    fn the_start_of_a_file_is_item_position() {
        assert!(Ctx::Unit.admits_declaration(None, None));
        assert!(!Ctx::Block.admits_declaration(None, None));
    }

    #[test]
    fn unit_scope_admits_a_declaration_after_a_semicolon_or_a_body() {
        assert!(Ctx::Unit.admits_declaration(Some(p(Punct::Semi)), None));
        assert!(Ctx::Unit.admits_declaration(Some(p(Punct::RBrace)), Some(Ctx::Block)));
        assert!(!Ctx::Unit.admits_declaration(Some(ident()), None));
        assert!(!Ctx::Unit.admits_declaration(Some(p(Punct::Eq)), None));
    }

    #[test]
    fn a_block_admits_a_statement_after_an_opening_brace_or_a_semicolon() {
        assert!(Ctx::Block.admits_declaration(Some(p(Punct::LBrace)), None));
        assert!(Ctx::Block.admits_declaration(Some(p(Punct::Semi)), None));
        assert!(!Ctx::Block.admits_declaration(Some(ident()), None));
    }

    #[test]
    fn a_closing_brace_returns_to_statement_position_only_for_a_statement_block() {
        // `if c { a } else { b } [i]`: the `}` closed a block used as a value,
        // so the `[` that follows is an index suffix, not an annotation
        assert!(!Ctx::Block.admits_declaration(Some(p(Punct::RBrace)), Some(Ctx::MatchBody)));
        assert!(!Ctx::Block.admits_declaration(Some(p(Punct::RBrace)), Some(Ctx::StructBody)));
        // a plain block does stand as a statement
        assert!(Ctx::Block.admits_declaration(Some(p(Punct::RBrace)), Some(Ctx::Block)));
    }

    #[test]
    fn a_struct_or_enum_body_admits_a_member_after_a_comma() {
        for c in [Ctx::StructBody, Ctx::EnumBody] {
            assert!(c.admits_declaration(Some(p(Punct::LBrace)), None));
            assert!(c.admits_declaration(Some(p(Punct::Comma)), None));
            assert!(!c.admits_declaration(Some(p(Punct::Colon)), None));
        }
    }

    #[test]
    fn a_parameter_list_admits_a_parameter_after_the_paren_or_a_comma() {
        assert!(Ctx::ParamList.admits_declaration(Some(p(Punct::LParen)), None));
        assert!(Ctx::ParamList.admits_declaration(Some(p(Punct::Comma)), None));
        assert!(!Ctx::ParamList.admits_declaration(Some(p(Punct::Colon)), None));
    }

    #[test]
    fn an_expression_context_never_admits_a_declaration() {
        // this is the case a previous-token rule gets wrong: `([x])`
        for c in [
            Ctx::ExprParen,
            Ctx::TypeList,
            Ctx::BracketExpr,
            Ctx::AnnotGroup,
            Ctx::MatchBody,
            Ctx::MacroArgs,
            Ctx::QmapBody,
        ] {
            for prev in [
                p(Punct::LParen),
                p(Punct::Comma),
                p(Punct::LBrace),
                p(Punct::Semi),
            ] {
                assert!(
                    !c.admits_declaration(Some(prev), None),
                    "{c:?} should not admit a declaration after {prev:?}"
                );
            }
        }
    }

    #[test]
    fn closing_an_annotation_group_always_leaves_a_declaration_position() {
        // `[a] [b] fn f()` stacks groups, and `[packed] struct S {}` follows one
        // with an item
        for c in [Ctx::Unit, Ctx::Block, Ctx::StructBody, Ctx::ParamList] {
            assert!(c.admits_declaration(Some(p(Punct::RBracket)), Some(Ctx::AnnotGroup)));
        }
        // but closing an index does not
        assert!(!Ctx::Block.admits_declaration(Some(p(Punct::RBracket)), Some(Ctx::BracketExpr)));
    }

    #[test]
    fn contexts_know_their_closing_punctuator() {
        assert_eq!(Ctx::Block.closer(), Punct::RBrace);
        assert_eq!(Ctx::StructBody.closer(), Punct::RBrace);
        assert_eq!(Ctx::ParamList.closer(), Punct::RParen);
        assert_eq!(Ctx::ExprParen.closer(), Punct::RParen);
        assert_eq!(Ctx::AnnotGroup.closer(), Punct::RBracket);
        assert_eq!(Ctx::BracketExpr.closer(), Punct::RBracket);
    }

    #[test]
    fn arming_keywords_choose_the_body_context() {
        assert_eq!(Arm::of(Token::Kw(Keyword::Struct)), Some(Arm::Struct));
        assert_eq!(Arm::of(Token::Kw(Keyword::Fn)), Some(Arm::Fn));
        assert_eq!(Arm::of(Token::Kw(Keyword::Let)), None);
        assert_eq!(Arm::Struct.brace_ctx(), Ctx::StructBody);
        assert_eq!(Arm::Match.brace_ctx(), Ctx::MatchBody);
        assert_eq!(Arm::Fn.brace_ctx(), Ctx::Block);
        assert_eq!(Arm::None.brace_ctx(), Ctx::Block);
    }

    #[test]
    fn arming_survives_a_name_and_generics_but_not_an_operator() {
        assert!(Arm::Fn.survives(ident()));
        assert!(Arm::Fn.survives(p(Punct::Lt)));
        assert!(Arm::Fn.survives(p(Punct::ColonColon)));
        assert!(Arm::Fn.survives(Token::OpName(Symbol::EMPTY)));
        assert!(!Arm::Fn.survives(p(Punct::Eq)));
        assert!(!Arm::Fn.survives(p(Punct::Plus)));
        assert!(!Arm::None.survives(ident()));
    }

    /////////////////////////////////
    // THE FN-PAREN CLASSIFICATION //
    /////////////////////////////////

    fn lex(src: &str) -> (Vec<(Token, crate::span::Span)>, Interner) {
        let i = Interner::new();
        let spliced = crate::source::Spliced::from_text(src);
        let out = crate::lex::lex(crate::span::SourceId(0), &spliced, &i);
        assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
        (out.tokens, i)
    }

    fn first_paren(toks: &[(Token, crate::span::Span)]) -> usize {
        toks.iter().position(|(t, _)| t.is(Punct::LParen)).unwrap()
    }

    #[test]
    fn matching_paren_finds_the_partner() {
        let (toks, _) = lex("f(a, g(b), c)");
        let open = first_paren(&toks);
        let close = matching_paren(&toks, open).unwrap();
        assert!(toks[close].0.is(Punct::RParen));
        assert_eq!(close, toks.len() - 1, "the outer paren, not the inner one");
    }

    #[test]
    fn matching_paren_tolerates_unbalanced_input() {
        let (toks, _) = lex("f(a, b");
        assert_eq!(matching_paren(&toks, first_paren(&toks)), None);
    }

    #[test]
    fn a_closure_expression_has_parameters() {
        // a closure: `fn(x: i64) -> i64 with (n) { x + n }`
        let (toks, _) = lex("let add = fn(x: i64) -> i64 with (n) { x + n };");
        let open = first_paren(&toks);
        assert!(is_closure_params(&toks, open));
    }

    #[test]
    fn a_closure_with_no_return_type_still_has_parameters() {
        let (toks, _) = lex("fn(x: i64) with (n) { x }");
        assert!(is_closure_params(&toks, first_paren(&toks)));
    }

    #[test]
    fn a_function_type_has_a_type_list_not_parameters() {
        // a function type: `fn deutsch(oracle: fn(*qubit, *qubit)) -> Guess`
        let (toks, _) = lex("fn(*qubit, *qubit)");
        assert!(!is_closure_params(&toks, first_paren(&toks)));

        let (toks, _) = lex("let f: fn(u8) -> u8 = g;");
        let open = first_paren(&toks);
        assert!(!is_closure_params(&toks, open));
    }

    #[test]
    fn a_function_type_inside_a_parameter_is_not_a_closure() {
        let (toks, _) = lex("fn deutsch(oracle: fn(*qubit, *qubit)) -> Guess { }");
        // the second `(` is the function type's argument list
        let opens: Vec<usize> = toks
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| t.is(Punct::LParen))
            .map(|(i, _)| i)
            .collect();
        assert!(!is_closure_params(&toks, opens[1]));
    }
}
