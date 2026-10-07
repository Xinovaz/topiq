//! The expression grammar.
//!
//! # The precedence table is the grammar
//!
//! There are fourteen levels, tightest to loosest: postfix and `as`; unary;
//! `**`; `* / %`; `+ -`; `<< >>`; `&`; `^`; `|`; comparisons; `&&`; `||`;
//! `.. ..=`; assignment. This module is a chain of parsers in that order rather
//! than a precedence-climbing table, for two reasons. The comparison level is
//! **non-associative**, so `a < b < c` needs a real diagnostic rather than a
//! parse that quietly stalls; and the whole chain has to be built twice (see
//! below) in a way that keeps both copies visibly identical.
//!
//! Unlike C, the bitwise operators bind **tighter** than the comparisons, so
//! `a | b == c` groups as `(a | b) == c`.
//!
//! # Two copies of the chain
//!
//! A structure literal cannot appear bare where a `{` would otherwise be taken
//! to open the body of an enclosing `if`, `while`, `for` or `match`. Otherwise
//! `if P { x: 1 } { … }` would have two readings and neither would be obviously
//! right. Where the literal really is wanted, parentheses say so:
//! `if (P { x: 1 }).ok { }`.
//!
//! [`grammar`] is therefore built twice, once unrestricted and once with
//! structure literals excluded. The restriction propagates leftward through the
//! operator chain automatically, because the whole chain is built from the
//! restricted atom. Inside any delimiter (parentheses, brackets, call
//! arguments), the unrestricted handle is used again, which is exactly what
//! "wherever a `{` would otherwise open a body" amounts to.
//!
//! # Continuing a path after a turbofish
//!
//! `phase::<8>::of(4)` needs a path to continue *after* its generic arguments,
//! so `::<8>` is followed by a further `::of` segment. A `"::" name` suffix
//! supplies that.
//!
//! Generic arguments attach to the path as a whole rather than to the segment
//! they followed; analysis places them by the order of the segments.
//!
//! # Generic arguments need a turbofish
//!
//! In expression position generic arguments are written `::<…>` and never
//! `<…>`, so `a < b` is always a comparison and `f::<T>(x)` always an
//! instantiation, with no lookahead and no ambiguity.
//!
//! `f<T>(x)` is a natural slip, so it is recognised (types in angle brackets
//! directly followed by a call cannot be comparisons, which do not chain)
//! and reported with the `::` it needs, then read on as if it had it.
//!
//! # Runs of `*` and `&`
//!
//! `&e` takes a reference to a place and `*e` is the place a reference refers
//! to. `**` and `&&` are single tokens (the tensor product and logical and),
//! but in prefix position, where no binary operator can stand, each is read as
//! two operators: `**p` follows two references and `&&x` takes a reference to
//! a reference. A run may also be written in parentheses, `(**)p` or
//! `(&&&)x`, and means exactly what it means without them; `(**)(&&)x` is
//! `x`. The parenthesised form keeps a run readable as one step, however it
//! is spaced.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::expr::{Capture, FieldInit, MacroArg, MatchArm, Param, PrepArg};
use crate::ast::{BinOp, Block, Expr, Pattern, StepOp, Type, UnOp};
use crate::ast::ty::TArg;
use crate::lex::{Keyword, Punct, Token};
use crate::quon::ast::QState;
use crate::span::{Span, Spanned};

use super::input::{
    Cx, Extra, close_angle, explained, ge, halves, ident, kw, listed, name, punct, shr, shr_assign,
};
use super::ty::path;

/// The operator-and-operand pairs trailing an assignment chain.
///
/// `a = b = c` parses as a flat repetition and is folded from the right, so
/// this is what the fold consumes.
type AssignTail = Vec<(Option<BinOp>, Spanned<Expr>)>;

/// Parses exactly one block form: `if`, `match`, `while`, `loop`, `for`, or a
/// bare block, and nothing after its closing brace.
///
/// Inside an expression a block form is an ordinary operand, so `1 + if c { 2 }
/// else { 3 }` adds. At the start of a statement it is the whole statement: its
/// closing brace ends it, so in
///
/// ```text
/// for i in 0..n { total += i; }
/// (total + 1) as i32
/// ```
///
/// the second line is a new expression, not a call of the loop. The statement
/// parser uses this function directly for that reason, rather than parsing a
/// whole expression that happens to begin with a block form.
///
/// Conditions and scrutinees use `expr_ns`, the handle that excludes bare
/// structure literals, so that the `{` after them opens the body.
pub fn block_form<'t, I, PE, PN, PP, PB>(
    expr: PE,
    expr_ns: PN,
    pat: PP,
    block: PB,
) -> impl Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PN: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PP: Parser<'t, I, Spanned<Pattern>, Extra<'t>> + Clone + 't,
    PB: Parser<'t, I, Block, Extra<'t>> + Clone + 't,
{
    let if_expr = recursive(|if_expr| {
        kw(Keyword::If)
            .ignore_then(expr_ns.clone())
            .then(block.clone())
            .then(
                kw(Keyword::Else)
                    .ignore_then(
                        if_expr
                            .clone()
                            .or(block.clone().map_with(|b, e| Spanned::new(Expr::Block(Box::new(b)), e.span()))),
                    )
                    .or_not(),
            )
            .map_with(|((cond, then), els), e| {
                Spanned::new(
                    Expr::If {
                        cond: Box::new(cond),
                        then: Box::new(then),
                        els: els.map(Box::new),
                    },
                    e.span(),
                )
            })
    });

    // an arm ends at its comma, except that an arm whose body is a block ends
    // at the closing brace and may leave the comma out
    let arm = pat
        .clone()
        .then_ignore(punct(Punct::FatArrow))
        .then(expr)
        .then(punct(Punct::Comma).or_not())
        .map(|((pattern, body), comma)| {
            let block_body = body.node.is_block_form();
            (MatchArm { pattern, body }, comma.is_some() || block_body)
        });
    let match_expr = kw(Keyword::Match)
        .ignore_then(kw(Keyword::Measure).or_not())
        .then(expr_ns.clone())
        .then(
            arm.repeated()
                .collect::<Vec<_>>()
                .try_map(|arms: Vec<(MatchArm, bool)>, span| {
                    // only the last arm may end without one
                    let last = arms.len().saturating_sub(1);
                    match arms.iter().position(|(_, ended)| !ended) {
                        Some(i) if i != last => Err(Rich::custom(
                            span,
                            "an arm whose body is not a block ends with `,`",
                        )),
                        _ => Ok(arms.into_iter().map(|(a, _)| a).collect::<Vec<_>>()),
                    }
                })
                .delimited_by(punct(Punct::LBrace), punct(Punct::RBrace)),
        )
        .map_with(|((measuring, scrutinee), arms), e| {
            Spanned::new(
                Expr::Match {
                    measuring: measuring.is_some(),
                    scrutinee: Box::new(scrutinee),
                    arms,
                },
                e.span(),
            )
        });

    let while_expr = kw(Keyword::While)
        .ignore_then(expr_ns.clone())
        .then(block.clone())
        .map_with(|(cond, body), e| {
            Spanned::new(
                Expr::While {
                    cond: Box::new(cond),
                    body: Box::new(body),
                },
                e.span(),
            )
        });

    let loop_expr = kw(Keyword::Loop)
        .ignore_then(block.clone())
        .map_with(|body, e| Spanned::new(Expr::Loop { body: Box::new(body) }, e.span()));

    let for_expr = kw(Keyword::For)
        .ignore_then(pat)
        .then_ignore(kw(Keyword::In))
        .then(expr_ns)
        .then(block.clone())
        .map_with(|((pattern, iter), body), e| {
            Spanned::new(
                Expr::For {
                    pattern: Box::new(pattern),
                    iter: Box::new(iter),
                    body: Box::new(body),
                },
                e.span(),
            )
        });

    let bare_block = block.map_with(|b, e| Spanned::new(Expr::Block(Box::new(b)), e.span()));

    choice((
        if_expr,
        match_expr,
        while_expr,
        loop_expr,
        for_expr,
        bare_block,
    ))
}

/// Builds an expression parser.
///
/// `no_struct` builds the variant that excludes bare structure literals.
/// `expr` is the unrestricted handle, used inside every delimiter; `expr_ns`
/// is the restricted one, used for the scrutinee of a control construct.
#[allow(clippy::too_many_arguments)]
pub fn grammar<'t, I, PE, PN, PT, PP, PB, PQ>(
    cx: Cx<'t>,
    no_struct: bool,
    expr: PE,
    expr_ns: PN,
    ty: PT,
    pat: PP,
    block: PB,
    qstate: PQ,
) -> impl Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PN: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PT: Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone + 't,
    PP: Parser<'t, I, Spanned<Pattern>, Extra<'t>> + Clone + 't,
    PB: Parser<'t, I, Block, Extra<'t>> + Clone + 't,
    PQ: Parser<'t, I, Spanned<QState>, Extra<'t>> + Clone + 't,
{
    ///////////////////////////////////////
    // GENERIC ARGUMENTS, TURBOFISH ONLY //
    ///////////////////////////////////////

    let targ = super::ty::generic_arg(cx, ty.clone(), expr.clone()).map_with(|a, e| Spanned::new(a, e.span()));
    let turbofish = punct(Punct::ColonColon)
        .ignore_then(
            targ.clone()
                .separated_by(punct(Punct::Comma))
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(punct(Punct::Lt), close_angle()),
        )
        .or_not()
        .map(Option::unwrap_or_default);
    // `f<T>(x)` is neither an instantiation, `<` being a comparison here, nor
    // two comparisons, which do not chain. it is recognised only to say so,
    // and read on as `f::<T>(x)`
    let angled = ty
        .clone()
        .map(TArg::Type)
        .map_with(|a, e| Spanned::new(a, e.span()))
        .separated_by(punct(Punct::Comma))
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(punct(Punct::Lt), close_angle())
        .then_ignore(punct(Punct::LParen).rewind())
        .validate(|args, e, emitter| {
            emitter.emit(explained(
                e.span(),
                "generic arguments in an expression are written `::<…>`, not `<…>`",
                "in an expression `<` is always a comparison, so that `a < b` has one reading",
                "put `::` before the `<`, as in `f::<T>(x)`",
            ));
            args
        });

    //////////////
    // LITERALS //
    //////////////

    let literal = select! {
        Token::Int { raw, base, suffix } = e => Spanned::new(Expr::Int { raw, base, suffix }, e.span()),
        Token::Float { raw, suffix } = e => Spanned::new(Expr::Float { raw, suffix }, e.span()),
        Token::Char(c) = e => Spanned::new(Expr::Char(c), e.span()),
        Token::Bool(b) = e => Spanned::new(Expr::Bool(b), e.span()),
        Token::Str { value, kind } = e => Spanned::new(Expr::Str { value, kind }, e.span()),
    };

    /////////////////////
    // AGGREGATE FORMS //
    /////////////////////

    let args = expr
        .clone()
        .separated_by(punct(Punct::Comma))
        .allow_trailing()
        .collect::<Vec<_>>();

    // `(e)`, `(a, b, …)` and `()`
    let parenthesized = args
        .clone()
        .delimited_by(punct(Punct::LParen), punct(Punct::RParen))
        .map_with(|mut es: Vec<Spanned<Expr>>, e| {
            if es.len() == 1 {
                Spanned::new(Expr::Paren(Box::new(es.remove(0))), e.span())
            } else {
                Spanned::new(Expr::Tuple(es), e.span())
            }
        });

    // `[a, b, …]` and `[e; N]`
    let array = expr
        .clone()
        .then_ignore(punct(Punct::Semi))
        .then(expr.clone())
        .map(|(elem, len)| Expr::ArrayRepeat {
            elem: Box::new(elem),
            len: Box::new(len),
        })
        .or(args.clone().map(Expr::Array))
        .delimited_by(punct(Punct::LBracket), punct(Punct::RBracket))
        .map_with(|a, e| Spanned::new(a, e.span()));

    // `Path { f: v, … }`, which is also how a data document writes an object
    let field_init = ident()
        .then_ignore(punct(Punct::Colon))
        .then(expr.clone())
        .map(|(name, value)| FieldInit { name, value });
    let struct_lit = path(cx)
        .then(listed(field_init, Punct::LBrace, Punct::RBrace))
        .map_with(|(path, fields), e| Spanned::new(Expr::StructLit { path, fields }, e.span()));

    let path_expr = path(cx)
        .then(angled.or(turbofish.clone()))
        .map_with(|(path, args), e| Spanned::new(Expr::Path { path, args }, e.span()));

    ///////////////////
    // CONTROL FORMS //
    ///////////////////

    let block_form = block_form(expr.clone(), expr_ns.clone(), pat.clone(), block.clone());

    //////////////
    // CLOSURES //
    //////////////

    let param = ident()
        .then_ignore(punct(Punct::Colon))
        .then(ty.clone())
        .map(move |(name, ty)| Param {
            annotations: Vec::new(),
            is_self: name.node == cx.self_,
            name,
            ty,
        });
    let capture = punct(Punct::And)
        .or_not()
        .then(ident())
        .map(|(by_ref, name)| Capture {
            name,
            by_ref: by_ref.is_some(),
        });
    let closure = kw(Keyword::Fn)
        .ignore_then(listed(param, Punct::LParen, Punct::RParen))
        .then(punct(Punct::Arrow).ignore_then(ty.clone()).or_not())
        .then_ignore(kw(Keyword::With))
        .then(listed(capture, Punct::LParen, Punct::RParen))
        .then(block.clone())
        .map_with(|(((params, ret), captures), body), e| {
            Spanned::new(
                Expr::Closure {
                    params,
                    ret: ret.map(Box::new),
                    captures,
                    body: Box::new(body),
                },
                e.span(),
            )
        });

    //////////////////////////////
    // COMPILER-SUPPLIED MACROS //
    //////////////////////////////

    // an argument is a string, a type, or an expression. a type is taken only
    // when it is the whole argument, so that `@sizeof(T) == 8` as an
    // argument is the comparison, not the type `@sizeof(T)`
    let arg_end = punct(Punct::Comma).or(punct(Punct::RParen)).rewind();
    // a state in QUON, as `@holonomy(C, |0>)` and `@bargmann(a, b, c)` take:
    // a whole argument that is a state and writes a ket, so that an
    // expression such as `a + b` is never read as one
    let quon_arg = crate::quon::parse::qstate(cx)
        .then_ignore(arg_end.clone())
        .try_map(|s, span| {
            if crate::quon::ast::writes_ket(&s.node) {
                Ok(s)
            } else {
                Err(Rich::custom(span, "not a state"))
            }
        })
        .map_with(|s, e| {
            Spanned::new(
                MacroArg::Quon(Box::new(crate::quon::ast::QForm::State(s))),
                e.span(),
            )
        });
    let macro_arg = choice((
        select! { Token::Str { value, .. } = e => Spanned::new(MacroArg::Str(Spanned::new(value, e.span())), e.span()) }
            .then_ignore(arg_end.clone()),
        ty.clone()
            .then_ignore(arg_end.clone())
            .map_with(|t, e| Spanned::new(MacroArg::Type(t), e.span())),
        quon_arg,
        expr.clone()
            .map_with(|x, e| Spanned::new(MacroArg::Expr(x), e.span())),
    ));
    let macro_call = select! { Token::MacroName(s) = e => Spanned::new(s, e.span()) }
        .then(listed(macro_arg, Punct::LParen, Punct::RParen))
        .map_with(|(name, args), e| Spanned::new(Expr::Macro { name, args }, e.span()));

    ///////////////////////
    // THE QUANTUM FORMS //
    ///////////////////////

    // each takes a unary or postfix operand, which is why they
    // are declared here and applied as prefixes below
    let prep = kw(Keyword::Prep)
        .ignore_then(
            qstate
                .clone()
                .map(|s| PrepArg::State(Box::new(s)))
                .or(expr.clone().map(|e| PrepArg::Expr(Box::new(e)))),
        )
        .map_with(|a, e| Spanned::new(Expr::Prep(a), e.span()));

    //////////////
    // THE ATOM //
    //////////////

    // a structure literal is excluded from the restricted grammar at the
    // top level, so that a `{` after a condition opens the body
    let primary = if no_struct {
        choice((
            literal,
            block_form.clone(),
            closure.clone(),
            macro_call.clone(),
            prep.clone(),
            parenthesized.clone(),
            array.clone(),
            path_expr.clone(),
        ))
        .boxed()
    } else {
        choice((
            literal,
            block_form,
            closure,
            macro_call,
            prep,
            parenthesized,
            array,
            struct_lit,
            path_expr,
        ))
        .boxed()
    };

    //////////////////////
    // POSTFIX SUFFIXES //
    //////////////////////

    // the quantum forms are primaries, so a suffix may follow one:
    // `query Phases[key](t)` calls the query's result. so the postfix layer is
    // recursive, with the quantum forms inside it
    #[derive(Clone)]
    enum Suffix {
        Call(Vec<Spanned<Expr>>, Span),
        Index(Spanned<Expr>, Span),
        Field(Spanned<crate::intern::Symbol>),
        Method(Spanned<crate::intern::Symbol>, Vec<Spanned<TArg>>, Vec<Spanned<Expr>>, Span),
        Turbo(Vec<Spanned<TArg>>, Span),
        /// `:: name`, continuing a path after a turbofish (see below).
        PathSeg(Spanned<crate::intern::Symbol>),
        Try(Span),
        Cast(Spanned<Type>),
        /// `++` or `--` after a place.
        Step(StepOp, Span),
    }

    fn apply(lhs: Spanned<Expr>, suffix: Suffix) -> Spanned<Expr> {
        let start = lhs.span;
        match suffix {
            Suffix::Call(a, s) => Spanned::new(
                Expr::Call {
                    callee: Box::new(lhs),
                    args: a,
                },
                start.join(s),
            ),
            Suffix::Index(i, s) => Spanned::new(
                Expr::Index {
                    receiver: Box::new(lhs),
                    index: Box::new(i),
                },
                start.join(s),
            ),
            Suffix::Field(name) => Spanned::new(
                Expr::Field {
                    receiver: Box::new(lhs),
                    name,
                },
                start.join(name.span),
            ),
            Suffix::Method(name, targs, a, s) => Spanned::new(
                Expr::MethodCall {
                    receiver: Box::new(lhs),
                    name,
                    targs,
                    args: a,
                },
                start.join(s),
            ),
            // a trailing turbofish on a path refines it in place
            Suffix::Turbo(tg, s) => {
                let span = start.join(s);
                match lhs.node {
                    Expr::Path { path, .. } => Spanned::new(Expr::Path { path, args: tg }, span),
                    other => Spanned::new(other, span),
                }
            }
            // `:: name` extends the path. generic arguments already gathered
            // stay attached to the path as a whole; see the note below on
            // `phase::<8>::of`
            Suffix::PathSeg(seg) => {
                let span = start.join(seg.span);
                match lhs.node {
                    Expr::Path { mut path, args } => {
                        path.segments.push(seg);
                        Spanned::new(Expr::Path { path, args }, span)
                    }
                    other => Spanned::new(other, span),
                }
            }
            Suffix::Try(s) => Spanned::new(Expr::Try(Box::new(lhs)), start.join(s)),
            Suffix::Step(op, s) => Spanned::new(
                Expr::Step {
                    op,
                    post: true,
                    place: Box::new(lhs),
                },
                start.join(s),
            ),
            Suffix::Cast(tt) => {
                let span = start.join(tt.span);
                Spanned::new(
                    Expr::Cast {
                        expr: Box::new(lhs),
                        ty: Box::new(tt),
                    },
                    span,
                )
            }
        }
    }

    let call_args = args
        .clone()
        .delimited_by(punct(Punct::LParen), punct(Punct::RParen))
        .map_with(|a, e| Suffix::Call(a, e.span()));
    let index = expr
        .clone()
        .delimited_by(punct(Punct::LBracket), punct(Punct::RBracket))
        .map_with(|i, e| Suffix::Index(i, e.span()));
    // `.0`: a tuple's element, named by its position
    let position = select! {
        Token::Int { raw, base: crate::lex::token::IntBase::Decimal, suffix: None } = e => Suffix::Field(Spanned::new(raw, e.span())),
    };
    let dot = punct(Punct::Dot).ignore_then(position.or(
        ident()
            .then(turbofish.clone())
            .then(
                args.clone()
                    .delimited_by(punct(Punct::LParen), punct(Punct::RParen))
                    .or_not(),
            )
            .map_with(|((nm, targs), call), e| match call {
                Some(a) => Suffix::Method(nm, targs, a, e.span()),
                None => Suffix::Field(nm),
            }),
    ));
    let turbo_only = punct(Punct::ColonColon)
        .ignore_then(
            targ.separated_by(punct(Punct::Comma))
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(punct(Punct::Lt), close_angle()),
        )
        .map_with(|tg, e| Suffix::Turbo(tg, e.span()));
    // `:: name` after a turbofish, for `phase::<8>::of(4)`. it is tried after
    // `turbo_only`, which needs a `<`, so the two never compete
    let path_seg = punct(Punct::ColonColon)
        .ignore_then(name(cx.kws))
        .map(Suffix::PathSeg);
    let try_op = punct(Punct::Question).map(Suffix::Try);
    let cast = kw(Keyword::As).ignore_then(ty.clone()).map(Suffix::Cast);
    let step = choice((
        punct(Punct::PlusPlus).map(|s| Suffix::Step(StepOp::Inc, s)),
        punct(Punct::MinusMinus).map(|s| Suffix::Step(StepOp::Dec, s)),
    ));

    let any_suffix = choice((
        call_args.clone(),
        index,
        dot.clone(),
        turbo_only.clone(),
        path_seg.clone(),
        try_op.clone(),
        cast.clone(),
        step.clone(),
    ));
    // a query's receiver may not end with an index, or it would swallow the
    // key: in `query Phases[key]`, `[key]` is the query's own bracket
    let non_index_suffix = choice((call_args, dot, turbo_only, path_seg, try_op, cast, step));

    let postfix = recursive(move |postfix| {
        let query = kw(Keyword::Query)
            .ignore_then(primary.clone().foldl(non_index_suffix.repeated(), apply))
            .then(
                expr.clone()
                    .delimited_by(punct(Punct::LBracket), punct(Punct::RBracket)),
            )
            .map_with(|(map, index), e| {
                Spanned::new(
                    Expr::Query {
                        map: Box::new(map),
                        index: Box::new(index),
                    },
                    e.span(),
                )
            });

        let replay = kw(Keyword::Replay)
            .then(postfix.clone())
            .map(|(at, x): (Span, Spanned<Expr>)| {
                let span = at.join(x.span);
                Spanned::new(Expr::Replay(Box::new(x)), span)
            });

        choice((query, replay, primary.clone()))
            .foldl(any_suffix.repeated(), apply)
            .boxed()
    });

    //////////////////////////////////////////////////////////////////
    // UNARY PREFIXES, PLUS THE TWO FORMS THAT TAKE A UNARY OPERAND //
    //////////////////////////////////////////////////////////////////

    let unary = recursive(move |unary| {
        // one reference operator as written: `*` or `&`, or a doubled token
        // `**` or `&&`, which is two of them
        let ref_op = choice((
            punct(Punct::Star).map(|s| vec![(UnOp::Deref, s)]),
            punct(Punct::StarStar).map(|s| {
                let (a, b) = halves(s);
                vec![(UnOp::Deref, a), (UnOp::Deref, b)]
            }),
            punct(Punct::And).map(|s| vec![(UnOp::Ref, s)]),
            punct(Punct::AndAnd).map(|s| {
                let (a, b) = halves(s);
                vec![(UnOp::Ref, a), (UnOp::Ref, b)]
            }),
        ));
        // `(&&&)x`, `(**)p`: a parenthesised run of reference operators,
        // meaning exactly what the run means written without parentheses
        let group = punct(Punct::LParen)
            .then(ref_op.clone().repeated().at_least(1).collect::<Vec<_>>())
            .then_ignore(punct(Punct::RParen))
            .map(|(open, runs): (Span, Vec<Vec<(UnOp, Span)>>)| {
                let mut ops: Vec<(UnOp, Span)> = runs.into_iter().flatten().collect();
                // the outermost operator's span starts at the parenthesis
                ops[0].1 = open;
                ops
            });
        let op = choice((
            punct(Punct::Minus).map(|s| vec![(UnOp::Neg, s)]),
            punct(Punct::Bang).map(|s| vec![(UnOp::Not, s)]),
            ref_op,
            group,
        ))
        .then(unary.clone())
        .map(|(ops, operand): (Vec<(UnOp, Span)>, Spanned<Expr>)| {
            ops.into_iter().rev().fold(operand, |operand, (op, at)| {
                let span = at.join(operand.span);
                Spanned::new(
                    Expr::Unary {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                )
            })
        });

        // `++x` and `--x`. `--` is one token, so `--x` is a decrement and
        // never a double negation
        let pre_step = choice((
            punct(Punct::PlusPlus).to(StepOp::Inc),
            punct(Punct::MinusMinus).to(StepOp::Dec),
        ))
        .map_with(|op, e| (op, e.span()))
        .then(unary.clone())
        .map(|((op, at), place): ((StepOp, Span), Spanned<Expr>)| {
            let span = at.join(place.span);
            Spanned::new(
                Expr::Step {
                    op,
                    post: false,
                    place: Box::new(place),
                },
                span,
            )
        });

        // `measure` and `lift` take a unary operand, so they are
        // written here rather than among the primaries
        let measure = kw(Keyword::Measure).then(unary.clone()).map(|(at, x)| {
            let span = at.join(x.span);
            Spanned::new(Expr::Measure(Box::new(x)), span)
        });
        let lift = kw(Keyword::Lift).then(unary.clone()).map(|(at, x)| {
            let span = at.join(x.span);
            Spanned::new(Expr::Lift(Box::new(x)), span)
        });

        choice((op, pre_step, measure, lift, postfix.clone())).boxed()
    });

    //////////////////////////////////////
    // THE BINARY CHAIN, TIGHTEST FIRST //
    //////////////////////////////////////

    // a plain function rather than a closure, so that the parser closures which
    // use it do not have to borrow it and can outlive this builder
    fn fold(lhs: Spanned<Expr>, (op, rhs): (BinOp, Spanned<Expr>)) -> Spanned<Expr> {
        let span = lhs.span.join(rhs.span);
        Spanned::new(
            Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            span,
        )
    }

    // a left-associative level: operands of the level above, joined by `op`
    macro_rules! level {
        ($operand:expr, $op:expr) => {
            $operand.clone().foldl($op.then($operand.clone()).repeated(), fold).boxed()
        };
    }

    let tensor = level!(unary, punct(Punct::StarStar).to(BinOp::Tensor));
    let product = level!(
        tensor,
        choice((
            punct(Punct::Star).to(BinOp::Mul),
            punct(Punct::Slash).to(BinOp::Div),
            punct(Punct::Percent).to(BinOp::Rem),
        ))
    );
    let sum = level!(product, choice((punct(Punct::Plus).to(BinOp::Add), punct(Punct::Minus).to(BinOp::Sub))));
    // `>>` is two adjacent `>` tokens; `<<` is one (see `crate::lex::token`)
    let shift = level!(sum, choice((punct(Punct::Shl).to(BinOp::Shl), shr().to(BinOp::Shr))));
    let bitand = level!(shift, punct(Punct::And).to(BinOp::And));
    let bitxor = level!(bitand, punct(Punct::Caret).to(BinOp::Xor));
    let bitor = level!(bitxor, punct(Punct::Or).to(BinOp::Or));

    // comparisons are non-associative. the tail is a repetition that rejects
    // a second comparison, so `a < b < c` gets a diagnostic saying so rather
    // than an unexpected token
    let cmp_op = choice((
        punct(Punct::EqEq).to(BinOp::Eq),
        punct(Punct::Ne).to(BinOp::Ne),
        punct(Punct::Le).to(BinOp::Le),
        ge().to(BinOp::Ge),
        punct(Punct::Lt).to(BinOp::Lt),
        punct(Punct::Gt).to(BinOp::Gt),
    ));
    let comparison = bitor
        .clone()
        .then(cmp_op.then(bitor.clone()).repeated().collect::<Vec<_>>())
        .try_map(|(head, tail), span| {
            if tail.len() > 1 {
                return Err(Rich::custom(
                    span,
                    "comparison operators are non-associative, so `a < b < c` has no reading",
                ));
            }
            Ok(tail.into_iter().fold(head, fold))
        })
        .boxed();

    let logical_and = level!(comparison, punct(Punct::AndAnd).to(BinOp::AndAnd));
    let logical_or = level!(logical_and, punct(Punct::OrOr).to(BinOp::OrOr));

    // `a..b` and `a..=b`. this is a single optional suffix, so
    // ranges do not chain
    let range = logical_or
        .clone()
        .then(
            choice((
                punct(Punct::DotDotEq).to(true),
                punct(Punct::DotDot).to(false),
            ))
            .then(logical_or.clone())
            .or_not(),
        )
        .map(|(start, rest)| match rest {
            None => start,
            Some((inclusive, end)) => {
                let span = start.span.join(end.span);
                Spanned::new(
                    Expr::Range {
                        start: Some(Box::new(start)),
                        end: Some(Box::new(end)),
                        inclusive,
                    },
                    span,
                )
            }
        })
        .boxed();

    // assignment is right-associative. its operands evaluate right to left,
    // which the tree records rather than the grammar
    let assign_op = choice((
        punct(Punct::Eq).to(None),
        punct(Punct::PlusEq).to(Some(BinOp::Add)),
        punct(Punct::MinusEq).to(Some(BinOp::Sub)),
        punct(Punct::StarEq).to(Some(BinOp::Mul)),
        punct(Punct::SlashEq).to(Some(BinOp::Div)),
        punct(Punct::PercentEq).to(Some(BinOp::Rem)),
        punct(Punct::AndEq).to(Some(BinOp::And)),
        punct(Punct::OrEq).to(Some(BinOp::Or)),
        punct(Punct::CaretEq).to(Some(BinOp::Xor)),
        punct(Punct::ShlEq).to(Some(BinOp::Shl)),
        shr_assign().to(Some(BinOp::Shr)),
    ));

    // assignment is right-associative, so `a = b = c` is `a = (b = c)`. it is
    // written as a flat repetition folded from the right rather than as a
    // recursive parser, which keeps the type inferable
    range
        .clone()
        .then(
            assign_op
                .then(range.clone())
                .repeated()
                .collect::<Vec<_>>(),
        )
        .map(|(first, mut rest): (Spanned<Expr>, AssignTail)| {
            fn assign(place: Spanned<Expr>, op: Option<BinOp>, value: Spanned<Expr>) -> Spanned<Expr> {
                let span = place.span.join(value.span);
                Spanned::new(Expr::Assign { op, place: Box::new(place), value: Box::new(value) }, span)
            }
            // each place takes as its value the assignment to its right
            let Some((mut op, mut value)) = rest.pop() else { return first };
            while let Some((before, place)) = rest.pop() {
                value = assign(place, op, value);
                op = before;
            }
            assign(first, op, value)
        })
}

#[cfg(test)]
mod tests {
    use crate::ast::{BinOp, Expr, UnOp};
    use crate::parse::testing::{dump, errors, expr};

    /// The first line of the dumped tree.
    fn top(src: &str) -> String {
        dump(src).lines().next().unwrap_or_default().to_owned()
    }

    //////////////////////////////////////
    // PRECEDENCE, WHICH IS THE GRAMMAR //
    //////////////////////////////////////

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        assert_eq!(top("1 + 2 * 3"), "binary +");
        assert_eq!(top("1 * 2 + 3"), "binary +");
        assert_eq!(top("(1 + 2) * 3"), "binary *");
    }

    #[test]
    fn the_tensor_operator_binds_tighter_than_multiplication() {
        // `**` sits between unary and `* / %`
        assert_eq!(top("a ** b * c"), "binary *");
        assert_eq!(top("a * b ** c"), "binary *");
    }

    #[test]
    fn shifts_bind_looser_than_addition() {
        assert_eq!(top("a + b << c"), "binary <<");
        assert_eq!(top("a << b + c"), "binary <<");
    }

    #[test]
    fn the_bitwise_operators_run_and_then_xor_then_or() {
        assert_eq!(top("a & b ^ c"), "binary ^");
        assert_eq!(top("a ^ b | c"), "binary |");
        assert_eq!(top("a | b & c"), "binary |");
    }

    #[test]
    fn comparisons_bind_looser_than_the_bitwise_operators() {
        // this is where the table departs from C, which puts the bitwise
        // operators below comparison. here `a | b == c` is `(a | b) == c`
        assert_eq!(top("a | b == c"), "binary ==");
        assert_eq!(top("a & b != c"), "binary !=");
    }

    #[test]
    fn the_logical_operators_are_the_loosest_before_range_and_assignment() {
        assert_eq!(top("a == b && c"), "binary &&");
        assert_eq!(top("a && b || c"), "binary ||");
        assert_eq!(top("a || b .. c"), "range ..");
    }

    #[test]
    fn arithmetic_is_left_associative() {
        // `1 - 2 - 3` is `(1 - 2) - 3`, so the left operand is the inner one
        let d = dump("1 - 2 - 3");
        let lines: Vec<&str> = d.lines().collect();
        assert_eq!(lines[0].trim(), "binary -");
        assert_eq!(lines[1].trim(), "binary -", "the left operand nests");
    }

    #[test]
    fn assignment_is_right_associative() {
        // `a = b = c` is `a = (b = c)`, so the right operand is
        // itself an assignment and the left is the plain place
        let d = dump("a = b = c");
        let lines: Vec<&str> = d.lines().collect();
        assert_eq!(lines[0].trim(), "assign =");
        assert_eq!(lines[1].trim(), "path a", "the place comes first in the tree");
        assert_eq!(
            lines.iter().filter(|l| l.trim() == "assign =").count(),
            2,
            "the right operand should itself be an assignment:\n{d}"
        );
    }

    #[test]
    fn comparisons_are_non_associative() {
        // `a < b < c` is ill-formed, not left-folded
        let e = errors("fn t() { let x = a < b < c; }");
        assert!(
            e.iter().any(|m| m.contains("non-associative")),
            "expected the rule to be named, got {e:?}"
        );
    }

    #[test]
    fn one_comparison_is_fine() {
        assert_eq!(top("a < b"), "binary <");
        assert_eq!(top("a >= b"), "binary >=");
        assert_eq!(top("a >> b"), "binary >>");
    }

    #[test]
    fn compound_assignment_records_its_operator() {
        for (src, want) in [
            ("a += b", "assign +="),
            ("a -= b", "assign -="),
            ("a *= b", "assign *="),
            ("a <<= b", "assign <<="),
            ("a >>= b", "assign >>="),
        ] {
            assert_eq!(top(src), want, "for {src}");
        }
    }

    ///////////////////////
    // UNARY AND POSTFIX //
    ///////////////////////

    #[test]
    fn unary_operators_parse() {
        assert_eq!(top("-a"), "unary Neg");
        assert_eq!(top("!a"), "unary Not");
        assert_eq!(top("&a"), "unary Ref");
        assert_eq!(top("*p"), "unary Deref");
    }

    /// The unary operators at the top of the tree, outermost first, and what
    /// they apply to.
    fn prefix_run(src: &str) -> (Vec<String>, String) {
        let d = dump(src);
        let mut ops = Vec::new();
        for line in d.lines() {
            match line.trim().strip_prefix("unary ") {
                Some(op) => ops.push(op.to_owned()),
                None => return (ops, line.trim().to_owned()),
            }
        }
        (ops, String::new())
    }

    #[test]
    fn a_doubled_token_in_prefix_position_is_two_operators() {
        let run = |src| prefix_run(src).0;
        assert_eq!(run("**p"), ["Deref", "Deref"]);
        assert_eq!(run("&&x"), ["Ref", "Ref"]);
        assert_eq!(run("***p"), ["Deref", "Deref", "Deref"]);
        assert_eq!(prefix_run("*&x"), (vec!["Deref".into(), "Ref".into()], "path x".into()));
    }

    #[test]
    fn a_parenthesized_run_means_the_run_without_parentheses() {
        for (grouped, plain) in [
            ("(**)p", "**p"),
            ("(&&&)x", "&&&x"),
            ("(*)p", "*p"),
            ("(&&)x", "&&x"),
            ("(* &)x", "*&x"),
        ] {
            assert_eq!(prefix_run(grouped), prefix_run(plain), "{grouped} against {plain}");
        }
        let (ops, leaf) = prefix_run("(**)(&&)x");
        assert_eq!(ops, ["Deref", "Deref", "Ref", "Ref"]);
        assert_eq!(leaf, "path x");
    }

    #[test]
    fn a_run_applies_to_the_whole_postfix_chain() {
        // unary binds looser than postfix: `*p.x` follows the reference in the
        // field, and `(*p).x` is written to read a field of what `p` refers to
        assert_eq!(prefix_run("*p.x"), (vec!["Deref".into()], "field .x".into()));
        assert_eq!(prefix_run("(**)a[0]"), (vec!["Deref".into(), "Deref".into()], "index".into()));
        assert_eq!(top("(*p).x"), "field .x");
    }

    #[test]
    fn the_binary_readings_of_the_doubled_tokens_survive() {
        assert_eq!(top("a ** b"), "binary **");
        assert_eq!(top("a && b"), "binary &&");
        assert_eq!(top("a ** **b"), "binary **");
        assert_eq!(top("a && &&b"), "binary &&");
        assert_eq!(top("a * *p"), "binary *");
        assert_eq!(top("(a) * b"), "binary *");
    }

    #[test]
    fn unary_binds_tighter_than_any_binary_operator() {
        assert_eq!(top("-a + b"), "binary +");
        assert_eq!(top("-a * b"), "binary *");
    }

    #[test]
    fn postfix_suffixes_chain_leftwards() {
        assert_eq!(top("f(x)"), "call");
        assert_eq!(top("xs[i]"), "index");
        assert_eq!(top("p.field"), "field .field");
        assert_eq!(top("p.m(x)"), "method .m");
        assert_eq!(top("e?"), "try ?");
        assert_eq!(top("e as u32"), "as u32");
    }

    #[test]
    fn a_postfix_chain_nests_in_the_order_written() {
        // `xs[i].area` is a field access on an index, not the other way round
        assert_eq!(top("xs[i].area"), "field .area");
        assert_eq!(top("f(x).y"), "field .y");
        assert_eq!(top("a.b[c]"), "index");
    }

    #[test]
    fn generic_arguments_need_a_turbofish_in_an_expression() {
        // `a < b` is always a comparison; an instantiation is
        // written `f::<T>(x)`
        assert_eq!(top("f::<u32>(x)"), "call");
        assert_eq!(top("a < b"), "binary <", "no generic reading here");
        assert_eq!(top("tcon::parse::<ServerConfig>(&src)"), "call");
        assert_eq!(top("invoke::<fn(*any) -> f64>(m.addr, p)"), "call");
    }

    #[test]
    fn generic_arguments_without_the_turbofish_are_reported_with_it() {
        let (out, _) = crate::parse::testing::parse("fn f() { let v = id<i64>(3); }\n");
        assert_eq!(out.diagnostics.len(), 1, "{:?}", out.diagnostics);
        assert!(format!("{:?}", out.diagnostics[0]).contains("put `::` before the `<`"));
        // anything else after `<` is a comparison
        assert_eq!(top("a < b && c > d"), "binary &&");
        assert_eq!(top("a < b + 1"), "binary <");
    }

    ///////////////
    // PRIMARIES //
    ///////////////

    #[test]
    fn literals_parse() {
        assert_eq!(top("42"), "int 42");
        assert_eq!(top("1.5"), "float 1.5");
        assert_eq!(top("true"), "bool true");
        assert!(top("'a'").starts_with("char"));
        assert!(top("\"hi\"").starts_with("string"));
    }

    #[test]
    fn aggregates_parse() {
        assert_eq!(top("[1, 2, 3]"), "array");
        assert_eq!(top("[0u32; 4]"), "array-repeat");
        assert_eq!(top("(a, b)"), "tuple");
        assert_eq!(top("P { x: 1, y: 2 }"), "struct-literal P");
    }

    #[test]
    fn a_single_parenthesized_expression_is_transparent_in_the_dump() {
        // `(e)` keeps a node so a span can cover the parentheses, but the dump
        // shows the inner expression, which is what a reader wants
        assert_eq!(top("(1 + 2)"), "binary +");
    }

    #[test]
    fn control_forms_parse_as_expressions() {
        assert_eq!(top("if c { 1 } else { 2 }"), "if");
        assert_eq!(top("match v { A => 1, B => 2 }"), "match");
        assert_eq!(top("loop { }"), "loop");
        assert_eq!(top("while c { }"), "while");
        assert_eq!(top("for i in 0..4 { }"), "for");
        assert_eq!(top("{ }"), "block");
    }

    #[test]
    fn a_structure_literal_is_excluded_from_a_condition() {
        // a bare `{` after a condition opens the body, so the
        // condition here is the bare path and the brace starts the consequent
        let d = dump("if P { 1 } else { 2 }");
        assert!(d.contains("path P"), "{d}");
        assert!(!d.contains("struct-literal"), "{d}");
    }

    #[test]
    fn a_parenthesized_structure_literal_is_admitted_in_a_condition() {
        // the escape hatch: `if (P { x: 1 }).ok { … }`
        let d = dump("if (P { x: 1 }).ok { 1 } else { 2 }");
        assert!(d.contains("struct-literal P"), "{d}");
    }

    #[test]
    fn a_closure_parses_with_its_capture_list() {
        // a closure with its capture list: `fn(x: i64) -> i64 with (n) { x + n }`
        let d = dump("fn(x: i64) -> i64 with (n) { x + n }");
        assert!(d.starts_with("closure (1 params, 1 captures)"), "{d}");
    }

    #[test]
    fn a_capture_may_be_by_reference() {
        let (e, _) = expr("fn() with (&a, b) { 1 }");
        match e {
            Expr::Closure { captures, .. } => {
                assert!(captures[0].by_ref);
                assert!(!captures[1].by_ref);
            }
            other => panic!("expected a closure, got {other:?}"),
        }
    }

    #[test]
    fn a_builtin_macro_call_parses() {
        assert_eq!(top("@sizeof(u32)"), "@sizeof");
        assert_eq!(top("@has_method(T, \"$add\")"), "@has_method");
    }

    ///////////////////////
    // THE QUANTUM FORMS //
    ///////////////////////

    #[test]
    fn measure_takes_a_unary_expression() {
        assert_eq!(top("measure q"), "measure");
        // it reaches through a postfix chain, so this measures `q[0]`
        assert_eq!(top("measure q[0]"), "measure");
    }

    #[test]
    fn lift_and_replay_parse() {
        assert_eq!(top("lift b"), "lift");
        assert_eq!(top("replay f(x)"), "replay");
    }

    #[test]
    fn prep_takes_a_quon_state() {
        // `prep |0>`, where `|0>` is a ket rather than three tokens
        assert_eq!(top("prep |0>"), "prep");
        let d = dump("prep |0>");
        assert!(d.contains("<QUON state>"), "{d}");
    }

    #[test]
    fn query_is_a_primary_so_a_call_may_follow_it() {
        // `query Phases[key](t)` calls the query's result
        assert_eq!(
            top("query Phases[key](t)"),
            "call",
            "the call applies to the query"
        );
        assert_eq!(top("query Phases[key]"), "query");
    }

    #[test]
    fn match_measure_is_recognized() {
        let (e, _) = expr("match measure q { A => 1 }");
        match e {
            Expr::Match { measuring, .. } => assert!(measuring),
            other => panic!("expected a match, got {other:?}"),
        }
        let (e, _) = expr("match q { A => 1 }");
        match e {
            Expr::Match { measuring, .. } => assert!(!measuring),
            other => panic!("expected a match, got {other:?}"),
        }
    }

    ////////////
    // RANGES //
    ////////////

    #[test]
    fn ranges_parse_in_both_forms() {
        assert_eq!(top("0..4"), "range ..");
        assert_eq!(top("0..=4"), "range ..=");
    }

    #[test]
    fn a_range_does_not_chain() {
        // the range suffix is a single optional, so `a..b..c` has no
        // reading
        assert!(!errors("fn t() { let x = a..b..c; }").is_empty());
    }

    ///////////////
    // STRUCTURE //
    ///////////////

    #[test]
    fn the_operator_of_a_binary_node_is_recorded() {
        let (e, _) = expr("a ** b");
        match e {
            Expr::Binary { op, .. } => assert_eq!(op, BinOp::Tensor),
            other => panic!("expected a binary node, got {other:?}"),
        }
    }

    #[test]
    fn a_unary_node_records_its_operator() {
        let (e, _) = expr("&x");
        match e {
            Expr::Unary { op, .. } => assert_eq!(op, UnOp::Ref),
            other => panic!("expected a unary node, got {other:?}"),
        }
    }

    #[test]
    fn malformed_expressions_are_rejected() {
        for src in ["1 +", "f(", "[1, 2", "a."] {
            assert!(
                !errors(&format!("fn t() {{ let x = {src}; }}")).is_empty(),
                "{src:?} should not parse"
            );
        }
    }
}
