//! Writing the site: each page through its template, and the files every
//! page shares: the style sheet, the search script and its index, and the
//! fonts.
//!
//! The templates, style sheet, script and fonts are part of the compiler,
//! so a site needs nothing but its own files: every address in it is
//! relative, and it reads the same from a disk as from a server.

use std::io;
use std::path::Path;

use minijinja::{AutoEscape, Environment, Value, context};

use super::model::{Content, Site};

/// The templates.
const TEMPLATES: [(&str, &str); 6] = [
    ("layout.html", include_str!("templates/layout.html")),
    ("macros.html", include_str!("templates/macros.html")),
    ("index.html", include_str!("templates/index.html")),
    ("unit.html", include_str!("templates/unit.html")),
    ("item.html", include_str!("templates/item.html")),
    ("source.html", include_str!("templates/source.html")),
];

/// The files every page shares.
const SHARED: [(&str, &[u8]); 15] = [
    ("static/style.css", include_bytes!("static/style.css")),
    ("static/search.js", include_bytes!("static/search.js")),
    ("static/fonts/OFL.txt", include_bytes!("static/fonts/OFL.txt")),
    ("static/fonts/eb-garamond-latin-normal.woff2", include_bytes!("static/fonts/eb-garamond-latin-normal.woff2")),
    ("static/fonts/eb-garamond-latin-italic.woff2", include_bytes!("static/fonts/eb-garamond-latin-italic.woff2")),
    ("static/fonts/eb-garamond-latin-ext-normal.woff2", include_bytes!("static/fonts/eb-garamond-latin-ext-normal.woff2")),
    ("static/fonts/eb-garamond-latin-ext-italic.woff2", include_bytes!("static/fonts/eb-garamond-latin-ext-italic.woff2")),
    ("static/fonts/eb-garamond-greek-normal.woff2", include_bytes!("static/fonts/eb-garamond-greek-normal.woff2")),
    ("static/fonts/eb-garamond-greek-italic.woff2", include_bytes!("static/fonts/eb-garamond-greek-italic.woff2")),
    ("static/fonts/jetbrains-mono-latin-normal.woff2", include_bytes!("static/fonts/jetbrains-mono-latin-normal.woff2")),
    ("static/fonts/jetbrains-mono-latin-italic.woff2", include_bytes!("static/fonts/jetbrains-mono-latin-italic.woff2")),
    ("static/fonts/jetbrains-mono-latin-ext-normal.woff2", include_bytes!("static/fonts/jetbrains-mono-latin-ext-normal.woff2")),
    ("static/fonts/jetbrains-mono-latin-ext-italic.woff2", include_bytes!("static/fonts/jetbrains-mono-latin-ext-italic.woff2")),
    ("static/fonts/jetbrains-mono-greek-normal.woff2", include_bytes!("static/fonts/jetbrains-mono-greek-normal.woff2")),
    ("static/fonts/jetbrains-mono-greek-italic.woff2", include_bytes!("static/fonts/jetbrains-mono-greek-italic.woff2")),
];

/// Writes the site into `out`, returning how many pages it has.
///
/// # Errors
///
/// When a file cannot be written. A template that fails to render is a
/// fault in the compiler, reported the same way.
pub fn write(site: &Site, out: &Path) -> io::Result<usize> {
    let mut env = Environment::new();
    // escaping leaves `/` alone, which needs none in HTML, so that
    // addresses read as written
    env.set_formatter(|out, state, value| {
        if state.auto_escape() == AutoEscape::Html && !value.is_safe() && !value.is_undefined() && !value.is_none() {
            out.write_str(&super::highlight::escape(&value.to_string())).map_err(minijinja::Error::from)
        } else {
            minijinja::escape_formatter(out, state, value)
        }
    });
    for (name, text) in TEMPLATES {
        env.add_template(name, text).map_err(io::Error::other)?;
    }
    let units = Value::from_serialize(&site.units);
    let fail = |e: minijinja::Error| io::Error::other(format!("{e:#}"));

    let index = env
        .get_template("index.html")
        .and_then(|t| {
            t.render(context! {
                root => "",
                project => &site.project,
                units => &units,
                tree => &site.tree,
            })
        })
        .map_err(fail)?;
    put(out, "index.html", index.as_bytes())?;

    for page in &site.pages {
        let (template, content) = match &page.content {
            Content::Unit(p) => ("unit.html", Value::from_serialize(p)),
            Content::Item(p) => ("item.html", Value::from_serialize(p)),
            Content::Source(p) => ("source.html", Value::from_serialize(p)),
        };
        let html = env
            .get_template(template)
            .and_then(|t| {
                t.render(context! {
                    root => "../",
                    project => &site.project,
                    units => &units,
                    sidebar => Value::from_serialize(&page.sidebar),
                    page => content,
                })
            })
            .map_err(fail)?;
        put(out, &page.path, html.as_bytes())?;
    }

    for (path, bytes) in SHARED {
        put(out, path, bytes)?;
    }
    put(out, "search-index.js", search_index(&site.search).as_bytes())?;
    Ok(site.pages.len() + 1)
}

/// Writes `bytes` to `path` below `out`, making the directories it needs.
fn put(out: &Path, path: &str, bytes: &[u8]) -> io::Result<()> {
    let full = out.join(path);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(full, bytes)
}

/// The search index as a script, which a page loads from a disk as well as
/// from a server, where reading a data file would need one.
fn search_index(entries: &[[String; 5]]) -> String {
    let mut out = String::from("window.TOPIQ_SEARCH = [\n");
    for e in entries {
        let fields: Vec<String> = e.iter().map(|f| js_string(f)).collect();
        out.push_str(&format!("[{}],\n", fields.join(",")));
    }
    out.push_str("];\n");
    out
}

/// Text as a JavaScript string literal, safe inside a page as well.
fn js_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // `</script>` must not end a script it is inside
            '<' => out.push_str("\\u003c"),
            c if (c as u32) < 0x20 || c == '\u{2028}' || c == '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_search_index_is_a_script_that_cannot_break_out() {
        let text = search_index(&[["a\"b".into(), "</script>".into(), "x\ny".into(), String::new(), "\u{2028}".into()]]);
        assert_eq!(text, "window.TOPIQ_SEARCH = [\n[\"a\\\"b\",\"\\u003c/script>\",\"x\\ny\",\"\",\"\\u2028\"],\n];\n");
    }
}
