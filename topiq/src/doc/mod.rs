//! Documentation: a program's units as a site of linked HTML pages.
//!
//! The site covers every unit of a loaded program (the units given and
//! every unit they import, directly or through others, the library's
//! included), so it stands on its own. Each unit has a page, and so does
//! each declaration it offers, showing its signature as written and its
//! documentation, which is Markdown. The root page lists the units and
//! draws which imports which. A unit with source has a page of that too,
//! and every declaration links to its line there.
//!
//! | module | what it does |
//! |---|---|
//! | [`model`] | builds what each page shows from the units' trees |
//! | [`resolve`] | finds the page that documents a name |
//! | [`markdown`] | renders documentation, resolving its links to declarations |
//! | [`highlight`] | renders Topiq text, marking tokens and linking names |
//! | [`render`] | writes the pages through their templates, and the shared files |
//!
//! A program is documented once it parses: its units are not analysed, so
//! a unit with a type error still has its documentation.

pub mod highlight;
pub mod markdown;
pub mod model;
pub mod render;
pub mod resolve;

use std::io;
use std::path::PathBuf;

use crate::driver::program::Loaded;

/// What to document, and where.
#[derive(Clone, Debug, Default)]
pub struct DocOptions {
    /// The directory the site is written into.
    pub out_dir: PathBuf,
    /// Whether to document what units keep to themselves: `static`
    /// declarations, and names beginning with `__`.
    pub private: bool,
}

/// What documenting a program did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// How many units were documented.
    pub units: usize,
    /// How many pages were written.
    pub pages: usize,
    /// Links in documentation that name nothing documented, each with the
    /// declaration it is written on.
    pub warnings: Vec<String>,
    /// The root page.
    pub index: PathBuf,
}

/// Writes the site documenting `loaded`.
///
/// # Errors
///
/// When a file of the site cannot be written.
pub fn document(loaded: &Loaded, options: &DocOptions) -> io::Result<Report> {
    let site = model::build(loaded, options.private);
    let pages = render::write(&site, &options.out_dir)?;
    Ok(Report {
        units: site.units.len(),
        pages,
        warnings: site.warnings,
        index: options.out_dir.join("index.html"),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use super::*;
    use crate::driver::{Options, Stage};

    /// Loads the program `entry` begins as far as documentation needs.
    fn load(entry: &Path) -> Loaded {
        let options = Options {
            stage: Stage::Parse,
            ..Options::default()
        };
        let loaded = crate::driver::program::load(&[entry.to_owned()], &[], options).unwrap();
        let errors: Vec<String> = loaded.all_diagnostics().map(|d| d.to_string()).collect();
        assert!(!loaded.has_errors(), "{errors:?}");
        loaded
    }

    /// Documents the program in `dir` that `app.tq` begins.
    fn site(dir: &Path, private: bool) -> (Report, PathBuf) {
        let out = dir.join("site");
        let loaded = load(&dir.join("app.tq"));
        let report = document(&loaded, &DocOptions { out_dir: out.clone(), private }).unwrap();
        (report, out)
    }

    const GEO: &str = "\
//! Plane geometry.
//!
//! Points are [`Point`]s.
#unit classical

/// A point in the plane.
struct Point {
    /// Across.
    x: i32,
    /// Up.
    y: i32,
}

impl Point {
    /// The squared distance from the origin.
    fn norm(self: *const Point) -> i32 { self.x * self.x + self.y * self.y }
}

/// Which way a turn goes. See [`Point::norm`].
enum Turn {
    /// Anticlockwise.
    Left,
    Right(i32),
}

/// The point halfway between `a` and `b`, which [`Opt`] does not need.
[deprecated: \"use `mid`\"]
fn halfway(a: Point, b: Point) -> Point { Point { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } }

/// Kept inside.
static fn hidden() -> i32 { 0 }
";

    const APP: &str = "\
//! The app. It uses [`geo::Point`] and [`geo::Nowhere`].
#unit classical
import geo as g;

/// Where it starts.
fn main() -> i32 { let p = g::Point { x: 1, y: 2 }; p.norm() }
";

    /// Writes the two units into `dir`.
    fn program(dir: &Path) {
        std::fs::write(dir.join("geo.tq"), GEO).unwrap();
        std::fs::write(dir.join("app.tq"), APP).unwrap();
    }

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[test]
    fn every_declaration_offered_has_a_page() {
        let dir = tempfile::tempdir().unwrap();
        program(dir.path());
        let (report, out) = site(dir.path(), false);
        for page in [
            "index.html",
            "app/index.html",
            "app/fn.main.html",
            "app/source.html",
            "geo/index.html",
            "geo/struct.Point.html",
            "geo/enum.Turn.html",
            "geo/fn.halfway.html",
            "core/index.html",
            "static/style.css",
            "static/search.js",
            "static/fonts/eb-garamond-latin-normal.woff2",
            "search-index.js",
        ] {
            assert!(out.join(page).is_file(), "{page} is missing");
        }
        assert!(!out.join("geo/fn.hidden.html").exists(), "a `static` function is the unit's own");
        assert_eq!(report.units, 3, "app, geo and core");
        assert_eq!(report.warnings, ["app: `geo::Nowhere` names nothing documented"]);
    }

    #[test]
    fn private_declarations_are_documented_when_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        program(dir.path());
        let (_, out) = site(dir.path(), true);
        assert!(out.join("geo/fn.hidden.html").is_file());
    }

    #[test]
    fn a_page_shows_the_signature_documentation_and_members() {
        let dir = tempfile::tempdir().unwrap();
        program(dir.path());
        let (_, out) = site(dir.path(), false);
        let point = read(&out.join("geo/struct.Point.html"));
        assert!(point.contains("<p>A point in the plane.</p>"), "{point}");
        assert!(point.contains("id=\"field.x\""), "{point}");
        assert!(point.contains("<p>Across.</p>"), "{point}");
        assert!(point.contains("id=\"method.norm\""), "{point}");
        assert!(point.contains("The squared distance from the origin."), "{point}");
        assert!(!point.contains("/// Across."), "a comment inside a signature is left out");

        let turn = read(&out.join("geo/enum.Turn.html"));
        assert!(turn.contains("id=\"variant.Left\""), "{turn}");
        assert!(turn.contains("href=\"../geo/struct.Point.html#method.norm\""), "{turn}");

        let halfway = read(&out.join("geo/fn.halfway.html"));
        assert!(halfway.contains("Deprecated: use `mid`"), "{halfway}");
        assert!(halfway.contains("<a class=\"name\" href=\"../geo/struct.Point.html\">Point</a>"), "{halfway}");
        assert!(halfway.contains("href=\"../core/enum.Opt.html\""), "a name alone may be `core`'s: {halfway}");
        assert!(!halfway.contains("Point { x:"), "the body is not part of the signature");

        let app = read(&out.join("app/index.html"));
        assert!(app.contains("href=\"../geo/struct.Point.html\""), "{app}");
        assert!(app.contains("geo</a>"), "the units it imports: {app}");
    }

    #[test]
    fn the_root_page_lists_the_units_and_draws_their_dependencies() {
        let dir = tempfile::tempdir().unwrap();
        program(dir.path());
        let (_, out) = site(dir.path(), false);
        let index = read(&out.join("index.html"));
        assert!(index.contains("<a href=\"app/index.html\">app</a>\n+-- <a href=\"geo/index.html\">geo</a>"), "{index}");
        assert!(index.contains("Plane geometry."), "a unit's summary: {index}");
        let search = read(&out.join("search-index.js"));
        assert!(search.contains("[\"Point\",\"geo\",\"Structure\",\"geo/struct.Point.html\",\"A point in the plane.\"]"), "{search}");
    }

    /// Every address in the site's pages and style sheet, with the file it
    /// is in.
    fn addresses(out: &Path) -> Vec<(PathBuf, String)> {
        let mut found = Vec::new();
        let mut work = vec![out.to_owned()];
        while let Some(dir) = work.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    work.push(path);
                    continue;
                }
                let text = match path.extension().and_then(|e| e.to_str()) {
                    Some("html" | "css") => read(&path),
                    _ => continue,
                };
                for marker in ["href=\"", "src=\"", "url("] {
                    for (at, _) in text.match_indices(marker) {
                        let rest = &text[at + marker.len()..];
                        let end = rest.find(['"', ')']).unwrap();
                        found.push((path.clone(), rest[..end].to_owned()));
                    }
                }
            }
        }
        found
    }

    #[test]
    fn the_site_stands_on_its_own_and_has_no_broken_links() {
        let dir = tempfile::tempdir().unwrap();
        program(dir.path());
        let (_, out) = site(dir.path(), false);
        let all = addresses(&out);
        assert!(all.len() > 50, "the check found the links");
        for (page, address) in all {
            assert!(!address.contains("://"), "{} reaches outside the site: {address}", page.display());
            let (file, anchor) = address.split_once('#').unwrap_or((&address, ""));
            let target = if file.is_empty() { page.clone() } else { page.parent().unwrap().join(file) };
            assert!(target.is_file(), "{} links to {address}, which is missing", page.display());
            if !anchor.is_empty() && target.extension().is_some_and(|e| e == "html") {
                let text = read(&target);
                let ids: BTreeSet<&str> = text
                    .match_indices("id=\"")
                    .map(|(at, _)| {
                        let rest = &text[at + 4..];
                        &rest[..rest.find('"').unwrap()]
                    })
                    .collect();
                assert!(ids.contains(anchor), "{} links to {address}, which has no such anchor", page.display());
            }
        }
    }

    #[test]
    fn the_library_is_documented_from_its_doc_comments() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("app.tq"), "#unit classical\nimport la;\nimport io;\nfn main() -> i32 { 0 }\n").unwrap();
        let (report, out) = site(dir.path(), false);
        let opt = read(&out.join("core/enum.Opt.html"));
        assert!(opt.contains("A value that may be absent."), "{opt}");
        let kron = read(&out.join("la/fn.kron.html"));
        assert!(kron.contains("The Kronecker product"), "{kron}");
        let la = read(&out.join("la/index.html"));
        assert!(la.contains("linear algebra"), "the unit's own documentation: {la}");
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }
}
