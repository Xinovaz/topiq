//! Syntax nodes for classical data documents.
//!
//! TCON is the constant-expression subset of Topiq's literal syntax, closed as
//! a standalone document format. It is to Topiq what JSON is to JavaScript,
//! with two differences: it is **typed**, every document being checked against
//! a Topiq type as its schema, and it is **native**, a document being an
//! ordinary Topiq expression parsed by the ordinary parser.
//!
//! Both differences show up here. The typed and enumerated forms carry a path
//! (the name of the type or variant the document claims to be) and that name
//! must be reachable and structurally identical, or the document is `ET01`.
//!
//! # `null` is not a universal value
//!
//! `null` is valid only where the schema position has optional type `T?`. It is
//! not a member of every type, the way it is in JSON, so a document writing
//! `null` where the schema has no `T?` is a mismatch rather than a silently
//! absent field.

use crate::intern::Symbol;
use crate::span::Spanned;

use crate::ast::ty::Path;
use crate::lex::{FloatSuffix, IntBase, IntSuffix, StrKind};

/// A scalar value.
///
/// Numbers use Topiq's lexical rules, `_` separators and all four bases
/// included, so an integer keeps its base and raw digits rather than a
/// parsed value.
#[derive(Clone, PartialEq, Debug)]
pub enum Scalar {
    /// An integer.
    Int {
        /// The digits.
        raw: Symbol,
        /// The base.
        base: IntBase,
        /// The type suffix.
        suffix: Option<IntSuffix>,
    },
    /// A floating literal.
    Float {
        /// The literal.
        raw: Symbol,
        /// The type suffix.
        suffix: Option<FloatSuffix>,
    },
    /// A string.
    Str {
        /// The contents.
        value: Symbol,
        /// How it was written.
        kind: StrKind,
    },
    /// A character.
    Char(char),
    /// `true` or `false`.
    Bool(bool),
    /// `null`, valid only where the schema position is optional.
    Null,
}

/// One field of an object.
#[derive(Clone, PartialEq, Debug)]
pub struct Pair {
    /// The field name.
    pub name: Spanned<Symbol>,
    /// Its value.
    pub value: Spanned<Value>,
}

/// A value.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    /// A scalar.
    Scalar(Scalar),
    /// `[ … ]`: checked against `[U; N]` with the length matching, or against
    /// `[U]`.
    Array(Vec<Spanned<Value>>),
    /// `{ … }`: checked against a structure, with every non-optional field
    /// present, and unknown fields ill-formed unless the structure is
    /// `[tcon: open]`.
    Object(Vec<Pair>),
    /// `Path { … }`: an object that names its type, as in
    /// `Vec3 { x: 1.0, y: 0.0, z: 0.0 }`.
    Typed {
        /// The type's name.
        path: Path,
        /// The fields.
        fields: Vec<Pair>,
    },
    /// `Path(…)`: an enumerated value with a tuple payload, or a bare `Path`
    /// for a unit variant.
    Enum {
        /// The variant's name.
        path: Path,
        /// The payload.
        payload: Vec<Spanned<Value>>,
    },
    /// `-n`: a number negated, as a constant expression may write one.
    Negative(Box<Spanned<Value>>),
}

impl Value {
    /// A short noun for a diagnostic.
    pub fn describe(&self) -> &'static str {
        match self {
            Value::Scalar(Scalar::Int { .. }) => "an integer",
            Value::Scalar(Scalar::Float { .. }) => "a floating literal",
            Value::Scalar(Scalar::Str { .. }) => "a string",
            Value::Scalar(Scalar::Char(_)) => "a character",
            Value::Scalar(Scalar::Bool(_)) => "a boolean",
            Value::Scalar(Scalar::Null) => "null",
            Value::Array(_) => "an array",
            Value::Object(_) => "an object",
            Value::Typed { .. } => "a typed object",
            Value::Enum { .. } => "an enumerated value",
            Value::Negative(n) => n.node.describe(),
        }
    }

    /// Whether this is `null`, which needs an optional schema position.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Scalar(Scalar::Null))
    }

    /// The fields of an object or typed object.
    pub fn fields(&self) -> Option<&[Pair]> {
        match self {
            Value::Object(f) | Value::Typed { fields: f, .. } => Some(f),
            _ => None,
        }
    }

    /// Looks a field up by name.
    pub fn field(&self, name: Symbol) -> Option<&Spanned<Value>> {
        self.fields()?
            .iter()
            .find(|p| p.name.node == name)
            .map(|p| &p.value)
    }
}

/// A document.
#[derive(Clone, PartialEq, Debug)]
pub struct Document {
    /// The value.
    pub value: Spanned<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::{SourceId, Span};

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    #[test]
    fn null_is_recognizable_because_it_needs_an_optional_position() {
        // `null` is valid only where the schema position has
        // optional type `T?`, unlike JSON where it inhabits everything
        assert!(Value::Scalar(Scalar::Null).is_null());
        assert!(!Value::Scalar(Scalar::Bool(false)).is_null());
        assert!(!Value::Array(Vec::new()).is_null());
    }

    #[test]
    fn integers_keep_their_base_and_raw_digits() {
        // numbers use Topiq's own lexical rules, including separators and
        // hex, binary and octal bases, so a parsed value would lose detail
        let mut i = Interner::new();
        let v = Value::Scalar(Scalar::Int {
            raw: i.intern("FF"),
            base: IntBase::Hex,
            suffix: Some(IntSuffix::U8),
        });
        match v {
            Value::Scalar(Scalar::Int { raw, base, suffix }) => {
                assert_eq!(i.resolve(raw), "FF");
                assert_eq!(base, IntBase::Hex);
                assert_eq!(suffix, Some(IntSuffix::U8));
            }
            other => panic!("expected an integer, got {other:?}"),
        }
    }

    #[test]
    fn fields_can_be_looked_up_by_name() {
        let mut i = Interner::new();
        let host = i.intern("host");
        let v = Value::Object(vec![Pair {
            name: sp(host),
            value: sp(Value::Scalar(Scalar::Bool(true))),
        }]);
        assert!(v.field(host).is_some());
        assert!(v.field(i.intern("port")).is_none());
        assert_eq!(v.fields().map(<[Pair]>::len), Some(1));
    }

    #[test]
    fn a_typed_object_also_exposes_its_fields() {
        // `Vec3 { x: 1.0 }` is an object that names its type
        let mut i = Interner::new();
        let x = i.intern("x");
        let v = Value::Typed {
            path: Path::single(sp(i.intern("Vec3"))),
            fields: vec![Pair {
                name: sp(x),
                value: sp(Value::Scalar(Scalar::Bool(true))),
            }],
        };
        assert!(v.field(x).is_some());
        assert_eq!(v.describe(), "a typed object");
    }

    #[test]
    fn a_scalar_has_no_fields() {
        assert!(Value::Scalar(Scalar::Bool(true)).fields().is_none());
        assert!(Value::Array(Vec::new()).fields().is_none());
    }

    #[test]
    fn every_value_describes_itself_for_a_diagnostic() {
        let mut i = Interner::new();
        let cases = [
            (Value::Scalar(Scalar::Null), "null"),
            (Value::Scalar(Scalar::Bool(true)), "a boolean"),
            (Value::Scalar(Scalar::Char('a')), "a character"),
            (Value::Array(Vec::new()), "an array"),
            (Value::Object(Vec::new()), "an object"),
            (
                Value::Enum {
                    path: Path::single(sp(i.intern("None"))),
                    payload: Vec::new(),
                },
                "an enumerated value",
            ),
        ];
        for (v, want) in cases {
            assert_eq!(v.describe(), want);
        }
    }
}
