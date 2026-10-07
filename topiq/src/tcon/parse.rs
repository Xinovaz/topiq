//! The grammar for classical data documents.
//!
//! A document is exactly one value. Two conveniences go beyond the bare
//! production: comments and trailing commas are permitted, and numbers and
//! strings use Topiq's own lexical rules, including `_` separators and hex,
//! binary and octal bases, so a document writes `0xFF` or `1_000_000` as a
//! source file does. A minus sign before a number makes it negative.
//!
//! Comments come free, since the lexer already treats them as trivia. Trailing
//! commas are written into the array and object productions here.
//!
//! # `null`
//!
//! `null` is not reserved, so it arrives as an ordinary identifier and is
//! recognised by its text. That also means a *field* may be called `null`,
//! which is why the scalar is tried only where a value is expected.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::intern::Symbol;
use crate::lex::{Punct, Token};
use crate::parse::input::{Cx, Extra, ident, listed, name, punct};
use crate::parse::ty::path;
use crate::span::{Span, Spanned};

use super::ast::{Document, Pair, Scalar, Value};

/// Builds the TCON value parser.
pub fn value<'t, I>(cx: Cx<'t>, null: Symbol) -> impl Parser<'t, I, Spanned<Value>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    recursive(move |value| {
        let scalar = select! {
            Token::Int { raw, base, suffix } = e => Spanned::new(Value::Scalar(Scalar::Int { raw, base, suffix }), e.span()),
            Token::Float { raw, suffix } = e => Spanned::new(Value::Scalar(Scalar::Float { raw, suffix }), e.span()),
            Token::Str { value, kind } = e => Spanned::new(Value::Scalar(Scalar::Str { value, kind }), e.span()),
            Token::Char(c) = e => Spanned::new(Value::Scalar(Scalar::Char(c)), e.span()),
            Token::Bool(b) = e => Spanned::new(Value::Scalar(Scalar::Bool(b)), e.span()),
        };

        // `null` is an ordinary identifier rather than a reserved word
        let null_lit = ident().try_map(move |s, span| {
            if s.node == null {
                Ok(Spanned::new(Value::Scalar(Scalar::Null), s.span))
            } else {
                Err(Rich::custom(span, "expected `null`"))
            }
        });

        let array = listed(value.clone(), Punct::LBracket, Punct::RBracket)
            .map_with(|vs, e| Spanned::new(Value::Array(vs), e.span()));

        // a field name may be spelt like a keyword, since field names have a
        // name space of their own
        let pair = name(cx.kws)
            .then_ignore(punct(Punct::Colon))
            .then(value.clone())
            .map(|(name, value)| Pair { name, value });
        let fields = listed(pair, Punct::LBrace, Punct::RBrace);

        let object = fields
            .clone()
            .map_with(|f, e| Spanned::new(Value::Object(f), e.span()));

        // `path { … }` is a typed object; `path ( … )` an enumerated value with
        // a payload; a bare `path` a unit variant
        let named = path(cx)
            .then(
                choice((
                    fields.map(NamedTail::Fields),
                    listed(value.clone(), Punct::LParen, Punct::RParen).map(NamedTail::Payload),
                ))
                .or_not(),
            )
            .map_with(|(p, tail), e| {
                let v = match tail {
                    Some(NamedTail::Fields(fields)) => Value::Typed { path: p, fields },
                    Some(NamedTail::Payload(payload)) => Value::Enum { path: p, payload },
                    None => Value::Enum {
                        path: p,
                        payload: Vec::new(),
                    },
                };
                Spanned::new(v, e.span())
            });

        // a number may be negated, as in a constant expression
        let negative = punct(Punct::Minus)
            .ignore_then(select! {
                Token::Int { raw, base, suffix } = e => Spanned::new(Value::Scalar(Scalar::Int { raw, base, suffix }), e.span()),
                Token::Float { raw, suffix } = e => Spanned::new(Value::Scalar(Scalar::Float { raw, suffix }), e.span()),
            })
            .map_with(|n, e| Spanned::new(Value::Negative(Box::new(n)), e.span()));

        choice((negative, scalar, array, object, null_lit, named))
    })
}

enum NamedTail {
    Fields(Vec<Pair>),
    Payload(Vec<Spanned<Value>>),
}

/// Builds the document parser: exactly one value.
pub fn document<'t, I>(
    cx: Cx<'t>,
    null: Symbol,
) -> impl Parser<'t, I, Document, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    value(cx, null).map(|value| Document { value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::parse::input::{TokenStream, eoi_span, stream};
    use crate::source::Spliced;
    use crate::span::SourceId;

    fn parse(src: &str) -> (Result<Document, Vec<String>>, Interner) {
        let mut interner = Interner::new();
        let spliced = Spliced::from_text(src);
        let lexed = crate::lex::lex(SourceId(0), &spliced, &interner);
        assert!(lexed.diagnostics.is_empty(), "{:?}", lexed.diagnostics);
        let tokens = lexed.tokens;
        let eoi = eoi_span(SourceId(0), &tokens);
        let null = interner.intern("null");
        let cx = Cx::new(&interner);
        let out = document::<TokenStream>(cx, null)
            .parse(stream(&tokens, eoi))
            .into_result()
            .map_err(|es| es.iter().map(ToString::to_string).collect());
        (out, interner)
    }

    fn ok(src: &str) -> Value {
        let (r, _) = parse(src);
        r.unwrap_or_else(|e| panic!("{src:?} did not parse: {e:?}"))
            .value
            .node
    }

    #[test]
    fn scalars_parse() {
        assert!(matches!(ok("42"), Value::Scalar(Scalar::Int { .. })));
        assert!(matches!(ok("1.5"), Value::Scalar(Scalar::Float { .. })));
        assert!(matches!(ok("\"hi\""), Value::Scalar(Scalar::Str { .. })));
        assert!(matches!(ok("'a'"), Value::Scalar(Scalar::Char('a'))));
        assert!(matches!(ok("true"), Value::Scalar(Scalar::Bool(true))));
        assert!(ok("null").is_null());
    }

    #[test]
    fn numbers_use_topiq_lexical_rules() {
        // separators and all three bases are available, as in source
        for src in ["1_000_000", "0xFF", "0b1010", "0o755", "2.5e-3"] {
            assert!(
                matches!(
                    ok(src),
                    Value::Scalar(Scalar::Int { .. } | Scalar::Float { .. })
                ),
                "{src} should be a number"
            );
        }
    }

    #[test]
    fn arrays_parse_and_allow_a_trailing_comma() {
        match ok("[1, 2, 3]") {
            Value::Array(vs) => assert_eq!(vs.len(), 3),
            other => panic!("expected an array, got {other:?}"),
        }
        match ok("[1, 2, 3,]") {
            Value::Array(vs) => assert_eq!(vs.len(), 3),
            other => panic!("expected an array, got {other:?}"),
        }
        match ok("[]") {
            Value::Array(vs) => assert!(vs.is_empty()),
            other => panic!("expected an array, got {other:?}"),
        }
    }

    #[test]
    fn objects_parse_and_allow_a_trailing_comma() {
        match ok("{ host: \"localhost\", port: 8080 }") {
            Value::Object(f) => assert_eq!(f.len(), 2),
            other => panic!("expected an object, got {other:?}"),
        }
        match ok("{ a: 1, }") {
            Value::Object(f) => assert_eq!(f.len(), 1),
            other => panic!("expected an object, got {other:?}"),
        }
    }

    #[test]
    fn a_typed_object_names_its_type() {
        // `Path { field: expr, … }` is a structure literal in source and an
        // object here: the same syntax, which is the point of the format
        match ok("Vec3 { x: 1.0, y: 0.0, z: 0.0 }") {
            Value::Typed { path, fields } => {
                assert!(path.is_simple());
                assert_eq!(fields.len(), 3);
            }
            other => panic!("expected a typed object, got {other:?}"),
        }
    }

    #[test]
    fn enumerated_values_parse_with_and_without_a_payload() {
        match ok("None") {
            Value::Enum { payload, .. } => assert!(payload.is_empty()),
            other => panic!("expected an enumerated value, got {other:?}"),
        }
        match ok("Some(1)") {
            Value::Enum { payload, .. } => assert_eq!(payload.len(), 1),
            other => panic!("expected an enumerated value, got {other:?}"),
        }
        match ok("Ord::Lt") {
            Value::Enum { path, .. } => assert_eq!(path.segments.len(), 2),
            other => panic!("expected an enumerated value, got {other:?}"),
        }
    }

    #[test]
    fn comments_are_permitted() {
        // comments are trivia to the lexer, so allowing them costs nothing
        match ok("{ // the host\n  host: \"x\", /* and the port */ port: 1 }") {
            Value::Object(f) => assert_eq!(f.len(), 2),
            other => panic!("expected an object, got {other:?}"),
        }
    }

    #[test]
    fn nesting_works() {
        let v = ok("{ tls: { cert: \"c\", key: \"k\" }, ports: [1, 2] }");
        let f = v.fields().unwrap();
        assert_eq!(f.len(), 2);
        assert!(matches!(f[0].value.node, Value::Object(_)));
        assert!(matches!(f[1].value.node, Value::Array(_)));
    }

    #[test]
    fn a_worked_configuration_document_parses() {
        // a configuration document read into a `ServerConfig`. the lookup uses
        // the interner the document was parsed with, or the same symbol numbers
        // would name different things
        let (doc, mut i) = parse(
            "ServerConfig { host: \"localhost\", port: 8080u16, \
             tls: TlsConfig { cert: \"a.pem\", key: \"a.key\" } }",
        );
        let v = doc.unwrap().value.node;
        assert!(matches!(v, Value::Typed { .. }));
        assert!(v.field(i.intern("host")).is_some());
        assert!(v.field(i.intern("tls")).is_some());
        assert!(v.field(i.intern("nonesuch")).is_none());
    }

    #[test]
    fn null_may_stand_where_an_optional_is_expected() {
        // whether the schema position really is optional is a later check,
        // which belongs to phase 7; the grammar only has to admit it
        let (doc, mut i) = parse("ServerConfig { host: \"h\", port: 1, tls: null }");
        let v = doc.unwrap().value.node;
        assert!(v.field(i.intern("tls")).unwrap().node.is_null());
        assert!(!v.field(i.intern("host")).unwrap().node.is_null());
    }

    #[test]
    fn a_field_may_be_spelled_like_a_keyword() {
        // field names have a name space of their own
        match ok("{ type: 1, base: 2 }") {
            Value::Object(f) => assert_eq!(f.len(), 2),
            other => panic!("expected an object, got {other:?}"),
        }
    }

    #[test]
    fn a_document_holds_exactly_one_value() {
        let (r, _) = parse("1 2");
        assert!(r.is_err(), "two values are not one document");
        let (r, _) = parse("");
        assert!(r.is_err(), "a document needs a value");
    }

    #[test]
    fn malformed_documents_are_rejected() {
        for src in ["{", "[1,", "{ a }", "{ : 1 }"] {
            let (r, _) = parse(src);
            assert!(r.is_err(), "{src:?} should not parse");
        }
    }
}
