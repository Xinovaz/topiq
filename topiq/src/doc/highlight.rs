//! Topiq text as HTML: each token marked with what it is, and each name
//! that has a page linked to it.
//!
//! Every token is written as its source spells it, so a raw string keeps its
//! hashes and a number its separators. The text between tokens is written in
//! one of two ways ([`Gaps`]): as it stands, comments included, for source
//! pages and examples; or reduced to its line breaks and indentation, for a
//! declaration's signature, where a comment inside the declaration would
//! only repeat its documentation.

use crate::intern::Interner;
use crate::lex::{Punct, Token};
use crate::source::Spliced;
use crate::span::{SourceId, Span};

/// How the text between tokens is written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gaps {
    /// As written, comments included.
    Verbatim,
    /// Comments left out: a gap without a line break is one space, and one
    /// with breaks is a single break followed by the next line's
    /// indentation, less the indentation of the line the text starts on.
    Signature,
}

/// Escapes text for HTML.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Renders `text[from..to]` as HTML, given the tokens of `text`, in order,
/// with spans into it. `link` gives the address of a name from the path it
/// ends: `["geo", "Point"]` for `Point` in `geo::Point`.
pub fn render(
    text: &str,
    tokens: &[(Token, Span)],
    from: u32,
    to: u32,
    gaps: Gaps,
    link: &mut dyn FnMut(&[&str]) -> Option<String>,
) -> String {
    let first = tokens.partition_point(|(_, s)| s.start < from);
    let last = tokens.partition_point(|(_, s)| s.end <= to).max(first);
    let tokens = &tokens[first..last];
    let base = indentation_at(text, from as usize);

    let mut out = String::new();
    let mut at = from as usize;
    let mut path: Vec<&str> = Vec::new();
    let mut directive = false;
    for (k, &(tok, span)) in tokens.iter().enumerate() {
        let gap = &text[at..span.start as usize];
        match gaps {
            Gaps::Verbatim => write_gap(gap, &mut out),
            // nothing before the first token
            Gaps::Signature if k == 0 => {}
            Gaps::Signature => match gap.rfind('\n') {
                Some(nl) => {
                    let indent = &gap[nl + 1..];
                    out.push('\n');
                    out.push_str(indent.get(base.min(indent.len())..).unwrap_or_default());
                }
                None if !gap.is_empty() => out.push(' '),
                None => {}
            },
        }
        let spelled = &text[span.range()];

        // a name continues the path when `::` joins it to the one before
        if let Token::Ident(_) = tok {
            let joined = k >= 2
                && tokens[k - 1].0.is(Punct::ColonColon)
                && matches!(tokens[k - 2].0, Token::Ident(_));
            if !joined {
                path.clear();
            }
            path.push(spelled);
        } else if !tok.is(Punct::ColonColon) {
            path.clear();
        }

        let class = match tok {
            Token::Kw(_) => Some("kw"),
            Token::Bool(_) | Token::Int { .. } | Token::Float { .. } | Token::Char(_) => Some("num"),
            Token::Str { .. } => Some("str"),
            Token::MacroName(_) => Some("macro"),
            Token::OpName(_) => Some("op"),
            Token::Punct(Punct::Hash) if gaps == Gaps::Verbatim => {
                directive = true;
                Some("directive")
            }
            Token::Ident(_) if directive => Some("directive"),
            _ => None,
        };
        if !tok.is(Punct::Hash) {
            directive = false;
        }
        // a name being declared, a field's or a parameter's, is not a use
        let declared = tokens.get(k + 1).is_some_and(|(t, _)| t.is(Punct::Colon))
            || k.checked_sub(1).is_some_and(|p| declares(tokens[p].0));
        let address = match tok {
            Token::Ident(_) if !declared => link(&path),
            _ => None,
        };
        match (address, class) {
            (Some(href), _) => {
                out.push_str(&format!("<a class=\"name\" href=\"{}\">{}</a>", escape(&href), escape(spelled)));
            }
            (None, Some(c)) => out.push_str(&format!("<span class=\"{c}\">{}</span>", escape(spelled))),
            (None, None) => out.push_str(&escape(spelled)),
        }
        at = span.end as usize;
    }
    if gaps == Gaps::Verbatim {
        write_gap(&text[at.min(to as usize)..to as usize], &mut out);
    }
    out
}

/// Renders a snippet of Topiq, such as an example in documentation, with no
/// links. Text that does not lex is escaped as it stands.
pub fn snippet(text: &str, interner: &Interner) -> String {
    let spliced = Spliced::from_text(text);
    let lexed = crate::lex::lex(SourceId(0), &spliced, interner);
    if lexed.has_errors() {
        return escape(text);
    }
    render(text, &lexed.tokens, 0, text.len() as u32, Gaps::Verbatim, &mut |_| None)
}

/// Whether the name after `tok` is one being declared: after a keyword
/// that declares one, or after the `.` of `fn Type.method`.
fn declares(tok: Token) -> bool {
    use crate::lex::Keyword;
    match tok {
        Token::Kw(k) => matches!(
            k,
            Keyword::Fn
                | Keyword::Struct
                | Keyword::Enum
                | Keyword::Let
                | Keyword::Type
                | Keyword::Cover
                | Keyword::Gauge
                | Keyword::Base
                | Keyword::Locale
                | Keyword::Chain
                | Keyword::Qmap
        ),
        Token::Punct(p) => p == Punct::Dot,
        _ => false,
    }
}

/// How much whitespace begins the line holding `offset`.
fn indentation_at(text: &str, offset: usize) -> usize {
    let start = text[..offset.min(text.len())].rfind('\n').map_or(0, |nl| nl + 1);
    text[start..].len() - text[start..].trim_start_matches([' ', '\t']).len()
}

/// Writes text from between tokens: whitespace and comments.
fn write_gap(gap: &str, out: &mut String) {
    let mut rest = gap;
    while !rest.is_empty() {
        let Some(open) = rest.find("//").into_iter().chain(rest.find("/*")).min() else {
            out.push_str(&escape(rest));
            return;
        };
        out.push_str(&escape(&rest[..open]));
        let comment = &rest[open..];
        let len = if comment.starts_with("//") {
            comment.find('\n').unwrap_or(comment.len())
        } else {
            comment[2..].find("*/").map_or(comment.len(), |e| e + 4)
        };
        let body = comment[..len].trim_end_matches('\r');
        let doc = (body.starts_with("///") && !body.starts_with("////")) || body.starts_with("//!");
        let class = if doc { "doc-comment" } else { "comment" };
        out.push_str(&format!("<span class=\"{class}\">{}</span>", escape(body)));
        rest = &comment[body.len()..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexed(src: &str, interner: &Interner) -> Vec<(Token, Span)> {
        let spliced = Spliced::from_text(src);
        crate::lex::lex(SourceId(0), &spliced, interner).tokens
    }

    #[test]
    fn tokens_are_marked_and_comments_kept_verbatim() {
        let i = Interner::new();
        let html = snippet("let x = \"<a>\"; // note\n@sizeof(T)", &i);
        assert_eq!(
            html,
            "<span class=\"kw\">let</span> x = <span class=\"str\">&quot;&lt;a&gt;&quot;</span>; \
             <span class=\"comment\">// note</span>\n<span class=\"macro\">@sizeof</span>(T)"
        );
    }

    #[test]
    fn a_directive_is_marked_as_one() {
        let i = Interner::new();
        let html = snippet("#unit classical\nfn f() { }", &i);
        assert!(html.starts_with("<span class=\"directive\">#</span><span class=\"directive\">unit</span> classical"), "{html}");
    }

    #[test]
    fn a_signature_drops_comments_and_keeps_its_layout() {
        let i = Interner::new();
        let src = "    fn f(\n        /// the count\n        n: u32,   // trailing\n    ) -> u32 { n }";
        let tokens = lexed(src, &i);
        let end = src.find(" {").unwrap() as u32;
        let html = render(src, &tokens, 4, end, Gaps::Signature, &mut |_| None);
        assert_eq!(html, "<span class=\"kw\">fn</span> f(\n    n: u32,\n) -&gt; u32");
    }

    #[test]
    fn a_path_is_linked_segment_by_segment() {
        let i = Interner::new();
        let src = "geo::Point";
        let tokens = lexed(src, &i);
        let mut seen: Vec<String> = Vec::new();
        let html = render(src, &tokens, 0, src.len() as u32, Gaps::Signature, &mut |p| {
            seen.push(p.join("::"));
            (p.len() == 2).then(|| "geo/struct.Point.html".to_owned())
        });
        assert_eq!(seen, ["geo", "geo::Point"]);
        assert_eq!(html, "geo::<a class=\"name\" href=\"geo/struct.Point.html\">Point</a>");
    }

    #[test]
    fn a_name_being_declared_is_not_linked() {
        let i = Interner::new();
        let src = "fn drop(drop: *drop) { }";
        let tokens = lexed(src, &i);
        let end = src.find(" {").unwrap() as u32;
        let html = render(src, &tokens, 0, end, Gaps::Signature, &mut |_| Some("core/fn.drop.html".to_owned()));
        assert_eq!(html.matches("<a ").count(), 1, "only the type is a use: {html}");
        assert!(html.ends_with("*<a class=\"name\" href=\"core/fn.drop.html\">drop</a>)"), "{html}");
    }

    #[test]
    fn text_that_does_not_lex_is_escaped_as_it_stands() {
        let i = Interner::new();
        assert_eq!(snippet("a < \"open", &i), "a &lt; &quot;open");
    }
}
