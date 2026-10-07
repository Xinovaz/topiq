//! Markdown documentation text as HTML.
//!
//! Three things differ from plain Markdown:
//!
//! - **Links to declarations.** A link with no definition whose text is
//!   shaped like a path, such as ``[`geo::Point`]`` or `[Opt::map]`, and a
//!   link whose destination is one, such as `[the point](geo::Point)`, go to
//!   the page that documents it. One that names nothing documented is left
//!   as text and reported.
//! - **Examples.** A code block with no language, or marked `topiq` or `tq`,
//!   is highlighted as Topiq.
//! - **Headings** are one level lower than written, since the page's title
//!   is its only first-level heading, and each has an `id` to link to.
//! - **Markup is text.** HTML written in documentation is shown as written,
//!   since in Topiq's `vec<T>` or `<a|b>` it is far likelier to be notation
//!   than a tag.
//!
//! A declaration's summary, shown in tables and search results, is its first
//! paragraph.

use pulldown_cmark::{BrokenLink, CodeBlockKind, CowStr, Event, HeadingLevel, LinkType, Options, Parser, Tag, TagEnd};

use crate::intern::Interner;

use super::highlight;

/// Documentation rendered three ways.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rendered {
    /// The whole text.
    pub html: String,
    /// The first paragraph's content.
    pub summary: String,
    /// The first paragraph as plain text.
    pub plain: String,
}

/// What rendering needs from the page it is for.
pub struct Context<'a> {
    /// The path from the page to the site's root, such as `../`.
    pub root: &'a str,
    /// The address of a path, relative to the root, or `None` if nothing
    /// documented has it.
    pub resolve: &'a dyn Fn(&[&str]) -> Option<String>,
    /// For highlighting examples.
    pub interner: &'a Interner,
    /// Where a link that names nothing is reported.
    pub warnings: &'a mut Vec<String>,
    /// What the documentation belongs to, for reports.
    pub owner: &'a str,
}

/// Renders documentation text.
pub fn render(text: &str, cx: &mut Context<'_>) -> Rendered {
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_TASKLISTS;
    let (root, resolve) = (cx.root, cx.resolve);
    let mut unresolved: Vec<String> = Vec::new();
    let address = |reference: &str, unresolved: &mut Vec<String>| -> Option<String> {
        let path = super::resolve::link_path(reference)?;
        let segments: Vec<&str> = path.iter().map(String::as_str).collect();
        match resolve(&segments) {
            Some(href) => Some(format!("{root}{href}")),
            None => {
                unresolved.push(reference.to_owned());
                None
            }
        }
    };

    // links with no definition: `[`geo::Point`]`
    let mut broken_unresolved: Vec<String> = Vec::new();
    let callback = |link: BrokenLink<'_>| {
        address(&link.reference, &mut broken_unresolved).map(|href| (CowStr::from(href), CowStr::from("")))
    };
    let events: Vec<Event<'_>> = Parser::new_with_broken_link_callback(text, options, Some(callback)).collect();

    let mut out: Vec<Event<'_>> = Vec::with_capacity(events.len());
    let mut i = 0;
    while i < events.len() {
        match &events[i] {
            // links whose destination is a path: `[the point](geo::Point)`
            Event::Start(Tag::Link { link_type: LinkType::Inline, dest_url, title, id }) => {
                let dest = match super::resolve::link_path(dest_url) {
                    Some(_) => address(dest_url, &mut unresolved).map_or_else(|| dest_url.clone(), CowStr::from),
                    None => dest_url.clone(),
                };
                out.push(Event::Start(Tag::Link {
                    link_type: LinkType::Inline,
                    dest_url: dest,
                    title: title.clone(),
                    id: id.clone(),
                }));
            }
            Event::Start(Tag::Heading { level, classes, attrs, .. }) => {
                let end = events[i..].iter().position(|e| matches!(e, Event::End(TagEnd::Heading(_)))).map_or(events.len(), |p| i + p);
                let words = plain(&events[i + 1..end]);
                out.push(Event::Start(Tag::Heading {
                    level: lower(*level),
                    id: Some(CowStr::from(slug(&words))),
                    classes: classes.clone(),
                    attrs: attrs.clone(),
                }));
            }
            Event::End(TagEnd::Heading(level)) => out.push(Event::End(TagEnd::Heading(lower(*level)))),
            // what looks like markup is text: `<T>` and `<a|b>` are types and
            // brackets, not tags
            Event::Html(t) | Event::InlineHtml(t) => out.push(Event::Text(t.clone())),
            Event::Start(Tag::CodeBlock(kind)) => {
                let end = events[i..].iter().position(|e| matches!(e, Event::End(TagEnd::CodeBlock))).map_or(events.len(), |p| i + p);
                let code: String = events[i + 1..end]
                    .iter()
                    .filter_map(|e| match e {
                        Event::Text(t) => Some(t.as_ref()),
                        _ => None,
                    })
                    .collect();
                let language = match kind {
                    CodeBlockKind::Fenced(info) => info.split([',', ' ']).next().unwrap_or("").to_owned(),
                    CodeBlockKind::Indented => String::new(),
                };
                let html = match language.as_str() {
                    "" | "topiq" | "tq" => {
                        format!("<pre class=\"code\"><code>{}</code></pre>\n", highlight::snippet(&code, cx.interner))
                    }
                    other => format!(
                        "<pre class=\"code\"><code class=\"language-{}\">{}</code></pre>\n",
                        highlight::escape(other),
                        highlight::escape(&code)
                    ),
                };
                out.push(Event::Html(CowStr::from(html)));
                i = end + 1;
                continue;
            }
            other => out.push(other.clone()),
        }
        i += 1;
    }
    for reference in broken_unresolved.into_iter().chain(unresolved) {
        cx.warnings.push(format!("{}: `{}` names nothing documented", cx.owner, reference.trim_matches('`')));
    }

    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, out.iter().cloned());
    let (mut summary, mut words) = (String::new(), String::new());
    if let Some(start) = out.iter().position(|e| matches!(e, Event::Start(Tag::Paragraph))) {
        let end = out[start..].iter().position(|e| matches!(e, Event::End(TagEnd::Paragraph))).map_or(out.len(), |p| start + p);
        pulldown_cmark::html::push_html(&mut summary, out[start + 1..end].iter().cloned());
        words = plain(&out[start + 1..end]);
    }
    Rendered { html, summary, plain: words }
}

/// The text of some events.
fn plain(events: &[Event<'_>]) -> String {
    let mut out = String::new();
    for e in events {
        match e {
            Event::Text(t) | Event::Code(t) => out.push_str(t),
            Event::SoftBreak | Event::HardBreak => out.push(' '),
            _ => {}
        }
    }
    out
}

/// A heading one level lower.
fn lower(level: HeadingLevel) -> HeadingLevel {
    match level {
        HeadingLevel::H1 => HeadingLevel::H2,
        HeadingLevel::H2 => HeadingLevel::H3,
        HeadingLevel::H3 => HeadingLevel::H4,
        HeadingLevel::H4 => HeadingLevel::H5,
        HeadingLevel::H5 | HeadingLevel::H6 => HeadingLevel::H6,
    }
}

/// An `id` made from a heading's words: lowercase, with every run of other
/// characters a single `-`.
fn slug(words: &str) -> String {
    let mut out = String::new();
    for c in words.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(text: &str) -> (Rendered, Vec<String>) {
        let interner = Interner::new();
        let mut warnings = Vec::new();
        let resolve = |p: &[&str]| match p {
            ["geo", "Point"] => Some("geo/struct.Point.html".to_owned()),
            _ => None,
        };
        let mut cx = Context {
            root: "../",
            resolve: &resolve,
            interner: &interner,
            warnings: &mut warnings,
            owner: "app::main",
        };
        let r = render(text, &mut cx);
        (r, warnings)
    }

    #[test]
    fn a_link_to_a_declaration_goes_to_its_page() {
        let (r, w) = rendered("See [`geo::Point`] and [the point](geo::Point).");
        assert!(r.html.contains("<a href=\"../geo/struct.Point.html\"><code>geo::Point</code></a>"), "{}", r.html);
        assert!(r.html.contains("<a href=\"../geo/struct.Point.html\">the point</a>"), "{}", r.html);
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn a_link_naming_nothing_is_reported_and_left_as_text() {
        let (r, w) = rendered("See [`geo::Line`], an array [qubit; 2] and [a site](https://example.com).");
        assert!(!r.html.contains("geo::Line</a>"), "{}", r.html);
        assert!(r.html.contains("[qubit; 2]"), "{}", r.html);
        assert_eq!(w, ["app::main: `geo::Line` names nothing documented"]);
    }

    #[test]
    fn an_example_is_highlighted_and_other_code_escaped() {
        let (r, _) = rendered("```\nlet x = 1;\n```\n\n```sh\na < b\n```");
        assert!(r.html.contains("<pre class=\"code\"><code><span class=\"kw\">let</span> x = <span class=\"num\">1</span>;"), "{}", r.html);
        assert!(r.html.contains("<code class=\"language-sh\">a &lt; b"), "{}", r.html);
    }

    #[test]
    fn markup_is_shown_as_written() {
        let (r, _) = rendered("Takes a vec<T> and gives <a|b>.\n\n<div>block</div>");
        assert!(r.html.contains("vec&lt;T&gt;"), "{}", r.html);
        assert!(r.html.contains("&lt;div&gt;block&lt;/div&gt;"), "{}", r.html);
    }

    #[test]
    fn headings_are_lowered_and_given_ids() {
        let (r, _) = rendered("# Errors\n\nText.");
        assert!(r.html.starts_with("<h2 id=\"errors\">Errors</h2>"), "{}", r.html);
    }

    #[test]
    fn the_summary_is_the_first_paragraph() {
        let (r, _) = rendered("Adds `a`\nand *b*.\n\nMore.");
        assert_eq!(r.summary, "Adds <code>a</code>\nand <em>b</em>.");
        assert_eq!(r.plain, "Adds a and b.");
    }
}
