//! Writing classical data as TCON text.
//!
//! The parser in [`super::parse`] reads TCON into syntax nodes that keep
//! spans and interned names; this module goes the other way, from a plain
//! tree a program builds to text the parser reads back. Output is indented one
//! level per nesting, with a trailing comma after every element, which the
//! grammar permits, so that adding an element later changes one line.
//!
//! A [`Doc::Int`] is unsigned: the metadata, the one writer, writes a signed
//! or wider value as a string of its digits.

use std::fmt::Write;

/// A TCON value to be written.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Doc {
    /// A non-negative integer.
    Int(u64),
    /// A string.
    Str(String),
    /// A string of several lines, written as a raw string, `r#"…"#`, so it
    /// reads as it is. One holding what a raw string cannot (`"#`, a
    /// carriage return, or a line ending in a backslash, which reading would
    /// splice) is written as [`Doc::Str`] is.
    Text(String),
    /// A character.
    Char(char),
    /// `true` or `false`.
    Bool(bool),
    /// `[a, b, …]`.
    Array(Vec<Doc>),
    /// `Name { field: value, … }`, or `{ field: value, … }` when the name
    /// is empty.
    Typed(String, Vec<(String, Doc)>),
    /// `Name(a, b, …)`, or a bare `Name` when there is no payload.
    Enum(String, Vec<Doc>),
}

impl Doc {
    /// `Name` with no payload.
    pub fn tag(name: &str) -> Doc {
        Doc::Enum(name.to_owned(), Vec::new())
    }

    /// `Name(payload…)`.
    pub fn call(name: &str, payload: Vec<Doc>) -> Doc {
        Doc::Enum(name.to_owned(), payload)
    }

    /// A string value.
    pub fn text(s: &str) -> Doc {
        Doc::Str(s.to_owned())
    }

    /// `Name { field: value, … }`.
    pub fn typed(name: &str, fields: Vec<(&str, Doc)>) -> Doc {
        Doc::Typed(name.to_owned(), fields.into_iter().map(|(f, v)| (f.to_owned(), v)).collect())
    }

    /// `[a, b, …]`, each item written by `f`.
    pub fn list<T>(items: impl IntoIterator<Item = T>, f: impl FnMut(T) -> Doc) -> Doc {
        Doc::Array(items.into_iter().map(f).collect())
    }
}

/// Writes a value as a TCON document.
pub fn write(doc: &Doc) -> String {
    let mut out = String::new();
    value(&mut out, doc, 0);
    out.push('\n');
    out
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("    ");
    }
}

/// Whether a value is short enough to stay on one line.
fn flat(doc: &Doc) -> bool {
    match doc {
        Doc::Int(_) | Doc::Str(_) | Doc::Char(_) | Doc::Bool(_) => true,
        Doc::Text(_) => false,
        Doc::Array(items) | Doc::Enum(_, items) => items.iter().all(flat) && items.len() <= 4,
        Doc::Typed(_, fields) => fields.is_empty(),
    }
}

fn value(out: &mut String, doc: &Doc, depth: usize) {
    match doc {
        Doc::Int(n) => {
            let _ = write!(out, "{n}");
        }
        Doc::Bool(b) => {
            let _ = write!(out, "{b}");
        }
        Doc::Text(s) if !s.contains("\"#") && !s.contains('\r') && !s.lines().any(|l| l.ends_with('\\')) => {
            out.push_str("r#\"");
            out.push_str(s);
            out.push_str("\"#");
        }
        Doc::Str(s) | Doc::Text(s) => {
            out.push('"');
            for c in s.chars() {
                escape(out, c, '"');
            }
            out.push('"');
        }
        Doc::Char(c) => {
            out.push('\'');
            escape(out, *c, '\'');
            out.push('\'');
        }
        Doc::Array(items) => list(out, "[", "]", items, depth),
        Doc::Enum(name, payload) => {
            out.push_str(name);
            if !payload.is_empty() {
                list(out, "(", ")", payload, depth);
            }
        }
        Doc::Typed(name, fields) => {
            out.push_str(name);
            if !name.is_empty() {
                out.push(' ');
            }
            if fields.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (k, v) in fields {
                indent(out, depth + 1);
                let _ = write!(out, "{k}: ");
                value(out, v, depth + 1);
                out.push_str(",\n");
            }
            indent(out, depth);
            out.push('}');
        }
    }
}

fn list(out: &mut String, open: &str, close: &str, items: &[Doc], depth: usize) {
    if items.iter().all(flat) && items.len() <= 4 {
        out.push_str(open);
        for (i, x) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            value(out, x, depth);
        }
        out.push_str(close);
        return;
    }
    out.push_str(open);
    out.push('\n');
    for x in items {
        indent(out, depth + 1);
        value(out, x, depth + 1);
        out.push_str(",\n");
    }
    indent(out, depth);
    out.push_str(close);
}

/// Writes one character of a string or character literal, escaped where the
/// literal needs it.
fn escape(out: &mut String, c: char, quote: char) {
    match c {
        '\n' => out.push_str("\\n"),
        '\t' => out.push_str("\\t"),
        '\r' => out.push_str("\\r"),
        '\0' => out.push_str("\\0"),
        '\\' => out.push_str("\\\\"),
        c if c == quote => {
            out.push('\\');
            out.push(c);
        }
        c if c.is_control() => {
            let _ = write!(out, "\\u{{{:x}}}", u32::from(c));
        }
        c => out.push(c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::parse::input::{Cx, TokenStream, eoi_span, stream};
    use crate::source::Spliced;
    use crate::span::SourceId;
    use crate::tcon::ast::{Scalar, Value};
    use chumsky::Parser;

    /// Parses TCON text back with the real parser.
    fn read(text: &str) -> (Value, Interner) {
        let mut interner = Interner::new();
        let spliced = Spliced::from_text(text);
        let lexed = crate::lex::lex(SourceId(0), &spliced, &interner);
        assert!(lexed.diagnostics.is_empty(), "{text}: {:?}", lexed.diagnostics);
        let eoi = eoi_span(SourceId(0), &lexed.tokens);
        let null = interner.intern("null");
        let doc = {
            let cx = Cx::new(&interner);
            crate::tcon::parse::document::<TokenStream>(cx, null)
                .parse(stream(&lexed.tokens, eoi))
                .into_result()
                .unwrap_or_else(|e| panic!("{text}\ndid not parse: {e:?}"))
        };
        (doc.value.node, interner)
    }

    #[test]
    fn what_is_written_reads_back() {
        let doc = Doc::Typed(
            "Unit".to_owned(),
            vec![
                ("name".to_owned(), Doc::text("geo\"metry\n")),
                ("size".to_owned(), Doc::Int(18_446_744_073_709_551_615)),
                ("on".to_owned(), Doc::Bool(true)),
                ("mark".to_owned(), Doc::Char('\'')),
                (
                    "items".to_owned(),
                    Doc::Array(vec![Doc::tag("Empty"), Doc::call("Pair", vec![Doc::Int(1), Doc::Int(2)])]),
                ),
                ("nothing".to_owned(), Doc::Typed("Empty".to_owned(), vec![])),
            ],
        );
        let text = write(&doc);
        let (v, i) = read(&text);
        let name = v.field(i.get("name").unwrap()).unwrap();
        match &name.node {
            Value::Scalar(Scalar::Str { value, .. }) => assert_eq!(i.resolve(*value), "geo\"metry\n"),
            other => panic!("{other:?}"),
        }
        let items = v.field(i.get("items").unwrap()).unwrap();
        let Value::Array(xs) = &items.node else { panic!() };
        assert_eq!(xs.len(), 2);
        assert!(matches!(&xs[1].node, Value::Enum { payload, .. } if payload.len() == 2));
        assert!(matches!(
            v.field(i.get("mark").unwrap()).unwrap().node,
            Value::Scalar(Scalar::Char('\''))
        ));
    }

    #[test]
    fn long_lists_go_one_element_per_line() {
        let doc = Doc::Array((0..6).map(Doc::Int).collect());
        let text = write(&doc);
        assert_eq!(text.lines().count(), 8, "{text}");
        let short = write(&Doc::Array(vec![Doc::Int(1), Doc::Int(2)]));
        assert_eq!(short, "[1, 2]\n");
    }

    #[test]
    fn control_characters_are_escaped() {
        let text = write(&Doc::text("a\u{1}b"));
        assert_eq!(text, "\"a\\u{1}b\"\n");
        let (v, i) = read(&text);
        let Value::Scalar(Scalar::Str { value, .. }) = v else { panic!() };
        assert_eq!(i.resolve(value), "a\u{1}b");
    }

    #[test]
    fn text_of_several_lines_is_written_raw_when_it_can_be() {
        let text = |s: &str| {
            let written = write(&Doc::Text(s.to_owned()));
            let (v, i) = read(&written);
            let Value::Scalar(Scalar::Str { value, .. }) = v else { panic!("{written}") };
            (written, i.resolve(value).to_owned())
        };
        let (written, back) = text("one \"quoted\"\ntwo\n");
        assert!(written.starts_with("r#\""), "{written}");
        assert_eq!(back, "one \"quoted\"\ntwo\n");
        // what a raw string cannot hold is escaped instead
        for s in ["ends \"# early", "a line\\\nspliced", "crlf\r\n"] {
            let (written, back) = text(s);
            assert!(written.starts_with('"'), "{written}");
            assert_eq!(back, s);
        }
    }

    #[test]
    fn an_object_without_a_name_is_written_bare() {
        let doc = Doc::Typed(String::new(), vec![("m".to_owned(), Doc::Int(12))]);
        let written = write(&doc);
        assert!(written.starts_with("{\n"), "{written}");
        let (v, _) = read(&written);
        assert!(matches!(v, Value::Object(_)), "{v:?}");
    }
}
