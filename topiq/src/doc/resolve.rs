//! Which page documents a name.
//!
//! An [`Index`] records, for every documented unit, the items it documents
//! and the units its imports name. [`Index::resolve`] then finds the page of
//! a path as a unit writes it, by the rules a reader applies: a name alone is
//! the unit's own item or else `core`'s, which every unit sees unqualified;
//! `a::X` is item `X` of the unit `a` names, through an import or by the
//! unit's own name; and `T::m` is the method, variant or field `m` of the
//! type `T`. Addresses are relative to the site's root.
//!
//! Names are found by spelling alone, without analysis, so a local binding
//! that shares an item's name is linked to the item. The generic parameters
//! of the declaration being rendered are passed in and never linked.

use std::collections::HashMap;

/// What one unit documents.
#[derive(Clone, Debug, Default)]
pub struct UnitEntry {
    /// The unit's name in the program, such as `geo` or `dsp.quantum`.
    pub name: String,
    /// The directory of its pages.
    pub dir: String,
    /// Its documented items, by name: the page of each, relative to `dir`,
    /// and the anchors on that page by the member names they document.
    pub items: HashMap<String, ItemEntry>,
    /// The unit each of its imports brings into scope, by the name it goes
    /// by: the alias, or the last segment of the path.
    pub imports: HashMap<String, usize>,
}

/// One documented item's page.
#[derive(Clone, Debug, Default)]
pub struct ItemEntry {
    /// The page.
    pub page: String,
    /// Anchors on the page, by the method, variant or field they document.
    pub anchors: HashMap<String, String>,
}

/// Every documented unit.
#[derive(Clone, Debug, Default)]
pub struct Index {
    /// The units.
    pub units: Vec<UnitEntry>,
}

impl Index {
    /// The unit called `name`: a unit's full name, or the name its file
    /// gives it when only one unit has that.
    pub fn unit(&self, name: &str) -> Option<usize> {
        if let Some(i) = self.units.iter().position(|u| u.name == name) {
            return Some(i);
        }
        let mut found = self.units.iter().enumerate().filter(|(_, u)| crate::driver::written(&u.name) == name);
        match (found.next(), found.next()) {
            (Some((i, _)), None) => Some(i),
            _ => None,
        }
    }

    /// The address of unit `u`'s own page.
    pub fn unit_page(&self, u: usize) -> String {
        format!("{}/index.html", self.units[u].dir)
    }

    /// The address of `path` as unit `from` writes it, unless it names one
    /// of `generics` or nothing documented.
    pub fn resolve(&self, from: usize, path: &[&str], generics: &[&str]) -> Option<String> {
        match path {
            [] => None,
            [name] if generics.contains(name) => None,
            [name] => self
                .item(from, name)
                .or_else(|| self.unit("core").and_then(|core| self.item(core, name)))
                .or_else(|| self.scope_unit(from, name).map(|u| self.unit_page(u))),
            [prefix @ .., last] => {
                let owner = prefix[prefix.len() - 1];
                // `a::X`, where `a` is a unit
                if let Some(u) = self.scope_unit(from, owner)
                    && let Some(href) = self.item(u, last)
                {
                    return Some(href);
                }
                // `T::m`, where `T` is a documented type
                let (u, entry) = self.item_entry(from, prefix, generics)?;
                let anchor = entry.anchors.get(*last)?;
                Some(format!("{}/{}#{anchor}", self.units[u].dir, entry.page))
            }
        }
    }

    /// The unit `name` refers to inside unit `from`: one its imports bring
    /// into scope, or else one of that name.
    fn scope_unit(&self, from: usize, name: &str) -> Option<usize> {
        self.units[from].imports.get(name).copied().or_else(|| self.unit(name))
    }

    /// The address of unit `u`'s item `name`.
    fn item(&self, u: usize, name: &str) -> Option<String> {
        let unit = &self.units[u];
        unit.items.get(name).map(|e| format!("{}/{}", unit.dir, e.page))
    }

    /// The unit and entry of the item `path` names inside unit `from`.
    fn item_entry(&self, from: usize, path: &[&str], generics: &[&str]) -> Option<(usize, &ItemEntry)> {
        let found = |u: usize, name: &str| self.units[u].items.get(name).map(|e| (u, e));
        match path {
            [name] if generics.contains(name) => None,
            [name] => found(from, name).or_else(|| found(self.unit("core")?, name)),
            [.., unit, name] => found(self.scope_unit(from, unit)?, name),
            [] => None,
        }
    }
}

/// The path an intra-documentation link names, with `.` read as `::`, or
/// `None` when the text does not look like one. Backticks around it and
/// `()` after it are dropped, so ``[`geo::area()`]`` names `geo::area`.
pub fn link_path(text: &str) -> Option<Vec<String>> {
    let text = text.trim();
    let text = text.strip_prefix('`').and_then(|t| t.strip_suffix('`')).unwrap_or(text);
    let text = text.strip_suffix("()").unwrap_or(text);
    let segments: Vec<String> = text.split("::").flat_map(|s| s.split('.')).map(str::to_owned).collect();
    let word = |s: &String| {
        let mut chars = s.chars();
        chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    segments.iter().all(word).then_some(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `app`, which imports `geo` as `g`, beside `geo` and `core`.
    fn index() -> Index {
        let entry = |page: &str, anchors: &[(&str, &str)]| ItemEntry {
            page: page.to_owned(),
            anchors: anchors.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect(),
        };
        let unit = |name: &str, items: Vec<(&str, ItemEntry)>, imports: &[(&str, usize)]| UnitEntry {
            name: name.to_owned(),
            dir: name.to_owned(),
            items: items.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
            imports: imports.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
        };
        Index {
            units: vec![
                unit("app", vec![("main", entry("fn.main.html", &[]))], &[("g", 1)]),
                unit(
                    "geo",
                    vec![("Point", entry("struct.Point.html", &[("norm", "method.norm"), ("x", "field.x")]))],
                    &[],
                ),
                unit("core", vec![("Opt", entry("enum.Opt.html", &[("Some", "variant.Some")]))], &[]),
            ],
        }
    }

    #[test]
    fn a_name_alone_is_the_units_own_or_cores() {
        let ix = index();
        assert_eq!(ix.resolve(0, &["main"], &[]).as_deref(), Some("app/fn.main.html"));
        assert_eq!(ix.resolve(0, &["Opt"], &[]).as_deref(), Some("core/enum.Opt.html"));
        assert_eq!(ix.resolve(0, &["Opt"], &["Opt"]), None, "a generic parameter is never linked");
        assert_eq!(ix.resolve(0, &["missing"], &[]), None);
    }

    #[test]
    fn a_unit_is_named_through_an_import_or_by_its_own_name() {
        let ix = index();
        assert_eq!(ix.resolve(0, &["g", "Point"], &[]).as_deref(), Some("geo/struct.Point.html"));
        assert_eq!(ix.resolve(0, &["geo", "Point"], &[]).as_deref(), Some("geo/struct.Point.html"));
        assert_eq!(ix.resolve(0, &["g"], &[]).as_deref(), Some("geo/index.html"));
        assert_eq!(ix.resolve(1, &["g", "Point"], &[]), None, "the alias is `app`'s alone");
    }

    #[test]
    fn a_member_is_an_anchor_on_its_types_page() {
        let ix = index();
        assert_eq!(ix.resolve(0, &["g", "Point", "norm"], &[]).as_deref(), Some("geo/struct.Point.html#method.norm"));
        assert_eq!(ix.resolve(0, &["Opt", "Some"], &[]).as_deref(), Some("core/enum.Opt.html#variant.Some"));
        assert_eq!(ix.resolve(0, &["Opt", "None"], &[]), None);
    }

    #[test]
    fn only_text_shaped_like_a_path_is_a_link() {
        assert_eq!(link_path("`geo::area()`"), Some(vec!["geo".to_owned(), "area".to_owned()]));
        assert_eq!(link_path("Point.norm"), Some(vec!["Point".to_owned(), "norm".to_owned()]));
        assert_eq!(link_path("qubit; 2"), None);
        assert_eq!(link_path("1"), None);
        assert_eq!(link_path("https://example.com"), None);
    }
}
