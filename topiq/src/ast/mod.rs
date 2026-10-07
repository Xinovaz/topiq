//! The abstract syntax tree.
//!
//! One module per syntactic category:
//!
//! | module | what it covers |
//! |---|---|
//! | [`item`] | everything declarable at unit scope: functions, types, judgement declarations |
//! | [`ty`] | types, paths and generic arguments |
//! | [`expr`] | expressions |
//! | [`stmt`] | statements and blocks |
//! | [`pat`] | patterns |
//! | [`annot`] | annotation groups and their arguments |
//!
//! Quantum-notation nodes live in [`crate::quon::ast`] and classical-data nodes
//! in [`crate::tcon::ast`]. Each has a grammar of its own and each is also a
//! standalone document format, so neither belongs in this tree.
//!
//! # Every node carries a span
//!
//! Nodes are wrapped in [`crate::span::Spanned`] rather than carrying
//! a span field, so the tree stays cheap and a node can be moved between
//! positions without dragging a stale location with it.

pub mod annot;
pub mod expr;
pub mod item;
pub mod pat;
pub mod print;
pub mod stmt;
pub mod ty;

pub use annot::{AnnArg, Annotation, AnnotationGroup, StdAnnotation};
pub use expr::{BinOp, Capture, Expr, FieldInit, MacroArg, MatchArm, Param, PrepArg, StepOp, UnOp};
pub use item::{
    ChainStage, Field, FnItem, FnName, GenericParam, Item, ItemKind, Linkage, LocaleMember,
    NameSpace, QmapEntry, Variant, VariantPayload,
};
pub use pat::{FieldPat, Pattern};
pub use stmt::{Block, Flow, LetStmt, Stmt, Storage};
pub use ty::{Path, TArg, Type};

use crate::lex::DocComment;
use crate::pp::UnitKind;
use crate::span::{Span, Spanned};

/// A parsed translation unit: a list of items.
///
/// A unit is classical or quantum in its entirety: there is no mixing within
/// one file. [`Unit::kind`] records which, and a great deal follows from it:
/// which primitive types can be materialised, which statements are allowed,
/// and what the unit compiles to.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Unit {
    /// Classical or quantum, from the `#unit` directive. `None` only when the
    /// directive was missing, which is `EU01`.
    pub kind: Option<UnitKind>,
    /// The items.
    ///
    /// Source order is kept for diagnostics, but unit scope is **not
    /// ordered**: an item may be used above the point where it is declared,
    /// so nothing here needs forward declarations.
    pub items: Vec<Spanned<Item>>,
    /// The unit's documentation: its `//!` comments above the first item.
    pub doc: Option<Doc>,
}

/// Documentation written on a declaration, as Markdown.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Doc {
    /// The comments' text, one line each, without their markers and
    /// without the indentation the lines share.
    pub text: String,
    /// From the first comment to the end of the last.
    pub span: Span,
}

impl Doc {
    /// The documentation the comments give together, or `None` if there
    /// are none.
    ///
    /// The indentation every non-blank line shares is removed, the way a
    /// reader sees the text, so a line indented further (such as one of an
    /// indented code block) keeps the difference.
    pub fn from_comments<'a>(comments: impl IntoIterator<Item = &'a DocComment>) -> Option<Doc> {
        let comments: Vec<&DocComment> = comments.into_iter().collect();
        let (first, last) = (comments.first()?, comments.last()?);
        let indent = comments
            .iter()
            .filter(|c| !c.text.trim().is_empty())
            .map(|c| c.text.len() - c.text.trim_start_matches([' ', '\t']).len())
            .min()
            .unwrap_or(0);
        let lines: Vec<&str> = comments
            .iter()
            .map(|c| c.text.get(indent..).unwrap_or_default().trim_end())
            .collect();
        Some(Doc {
            text: lines.join("\n"),
            span: first.span.join(last.span),
        })
    }
}

impl Unit {
    /// Whether this is a quantum unit.
    pub fn is_quantum(&self) -> bool {
        self.kind == Some(UnitKind::Quantum)
    }

    /// Whether this is a classical unit.
    pub fn is_classical(&self) -> bool {
        self.kind == Some(UnitKind::Classical)
    }

    /// How many items the unit declares.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the unit declares no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Every item of a given kind, as a filter over the item list.
    pub fn items_matching<'a>(
        &'a self,
        f: impl Fn(&ItemKind) -> bool + 'a,
    ) -> impl Iterator<Item = &'a Spanned<Item>> {
        self.items.iter().filter(move |i| f(&i.node.kind))
    }

    /// Every import the unit declares.
    pub fn imports(&self) -> impl Iterator<Item = &Spanned<Item>> {
        self.items_matching(|k| matches!(k, ItemKind::Import { .. }))
    }

    /// Every function or operator the unit defines.
    pub fn functions(&self) -> impl Iterator<Item = &Spanned<Item>> {
        self.items_matching(|k| matches!(k, ItemKind::Fn(_)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::{SourceId, Span};

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    fn item(kind: ItemKind) -> Spanned<Item> {
        sp(Item {
            annotations: Vec::new(),
            linkage: Linkage::Program,
            doc: None,
            kind,
        })
    }

    #[test]
    fn an_empty_unit_has_no_items_and_no_kind() {
        let u = Unit::default();
        assert!(u.is_empty());
        assert_eq!(u.len(), 0);
        assert!(!u.is_quantum() && !u.is_classical());
    }

    #[test]
    fn a_unit_is_classical_or_quantum_in_its_entirety() {
        // a unit is classical or quantum in its entirety
        let q = Unit {
            kind: Some(UnitKind::Quantum),
            items: Vec::new(),
            doc: None,
        };
        assert!(q.is_quantum());
        assert!(!q.is_classical());

        let c = Unit {
            kind: Some(UnitKind::Classical),
            items: Vec::new(),
            doc: None,
        };
        assert!(c.is_classical());
        assert!(!c.is_quantum());
    }

    #[test]
    fn items_can_be_filtered_by_kind() {
        let mut i = Interner::new();
        let u = Unit {
            kind: Some(UnitKind::Classical),
            items: vec![
                item(ItemKind::Import {
                    path: Path::single(sp(i.intern("geometry"))),
                    alias: None,
                    kind: None,
                }),
                item(ItemKind::Fn(Box::new(FnItem {
                    name: FnName::Plain(sp(i.intern("main"))),
                    generics: Vec::new(),
                    params: Vec::new(),
                    ret: None,
                    where_clause: None,
                    body: Block::empty(),
                    signature: Span::synthetic(),
                    doc: None,
                }))),
                item(ItemKind::Struct {
                    name: sp(i.intern("S")),
                    generics: Vec::new(),
                    fields: Vec::new(),
                }),
            ],
            doc: None,
        };
        assert_eq!(u.len(), 3);
        assert_eq!(u.imports().count(), 1);
        assert_eq!(u.functions().count(), 1);
    }

    #[test]
    fn source_order_is_preserved_even_though_unit_scope_is_unordered() {
        // unit scope is unordered for *resolution*; the list
        // still holds source order, which diagnostics need
        let mut i = Interner::new();
        let names = ["c", "a", "b"];
        let u = Unit {
            kind: Some(UnitKind::Classical),
            items: names
                .iter()
                .map(|n| {
                    item(ItemKind::Struct {
                        name: sp(i.intern(n)),
                        generics: Vec::new(),
                        fields: Vec::new(),
                    })
                })
                .collect(),
            doc: None,
        };
        let got: Vec<Option<crate::intern::Symbol>> =
            u.items.iter().map(|it| it.node.kind.declared_name()).collect();
        let want: Vec<Option<crate::intern::Symbol>> =
            names.iter().map(|n| Some(i.intern(n))).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn documentation_loses_only_the_indentation_its_lines_share() {
        let line = |text: &str, at: u32| DocComment {
            kind: crate::lex::DocKind::Outer,
            text: text.to_owned(),
            span: Span::new(SourceId(0), at, at + 3),
            anchor: None,
        };
        let comments = [line(" Adds.", 0), line("", 10), line("     let x = 1;", 20)];
        let doc = Doc::from_comments(&comments).unwrap();
        assert_eq!(doc.text, "Adds.\n\n    let x = 1;");
        assert_eq!(doc.span, Span::new(SourceId(0), 0, 23));
        assert_eq!(Doc::from_comments(&[]), None);
    }
}
