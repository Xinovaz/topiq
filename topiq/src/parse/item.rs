//! The item grammar.
//!
//! An item is a run of annotation groups, an optional `static`, and then the
//! declaration itself. `static` is the only linkage specifier Topiq has: there
//! is no export keyword, because a unit's interface is what it declares and
//! *privacy* is the case worth marking.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::annot::AnnotationGroup;
use crate::ast::item::{
    ChainStage, Field, FnItem, FnName, GenericParam, Item, ItemKind, Linkage, LocaleMember,
    QmapEntry, Variant, VariantPayload,
};
use crate::ast::{Block, Expr, Param, Type};
use crate::lex::{Keyword, Punct, Token};
use crate::quon::parse::{decl_name, qcover, qgauge};
use crate::span::{Span, Spanned};

use super::input::{Cx, Extra, close_angle, ident, kw, listed, name, punct};
use super::ty::path;

/// Builds the item parser.
#[allow(clippy::too_many_arguments)]
pub fn grammar<'t, I, PE, PT, PB, PA>(
    cx: Cx<'t>,
    expr: PE,
    ty: PT,
    block: PB,
    annot: PA,
) -> impl Parser<'t, I, Spanned<Item>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PT: Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone + 't,
    PB: Parser<'t, I, Block, Extra<'t>> + Clone + 't,
    PA: Parser<'t, I, Spanned<AnnotationGroup>, Extra<'t>> + Clone + 't,
{
    let semi = punct(Punct::Semi);
    let annots = annot.clone().repeated().collect::<Vec<_>>();

    // `generics := "<" gparam ("," gparam)* ">"`, where a const generic is a
    // parameter whose type happens to be a const type
    let generics = ident()
        .then(punct(Punct::Colon).ignore_then(ty.clone()).or_not())
        .map(|(name, ty)| GenericParam { name, ty })
        .separated_by(punct(Punct::Comma))
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(punct(Punct::Lt), close_angle())
        .or_not()
        .map(Option::unwrap_or_default);

    // `param := annotation-group* self-param | annotation-group* IDENT ":" type`
    let param = annots
        .clone()
        .then(ident())
        .then_ignore(punct(Punct::Colon))
        .then(ty.clone())
        .map(move |((annotations, name), ty)| Param {
            annotations,
            is_self: name.node == cx.self_,
            name,
            ty,
        });
    let params = listed(param, Punct::LParen, Punct::RParen);

    // a plain name, `Type.method`, or `$operator`. right after `fn` a keyword
    // can only be a name, and such a function is reached as `unit::name`
    let fn_name = choice((
        select! { Token::OpName(s) = e => Spanned::new(s, e.span()) }.map(FnName::Operator),
        path(cx)
            .then_ignore(punct(Punct::Dot))
            .then(ident())
            .map(|(ty, name)| FnName::External { ty, name }),
        name(cx.kws).map(FnName::Plain),
    ));

    // the signature's span is everything before the body
    let fn_item = kw(Keyword::Fn)
        .ignore_then(fn_name)
        .then(generics.clone())
        .then(params.clone())
        .then(punct(Punct::Arrow).ignore_then(ty.clone()).or_not())
        .then(kw(Keyword::Where).ignore_then(expr.clone()).or_not())
        .map_with(|head, e| (head, e.span()))
        .then(block.clone())
        .map(
            |((((((name, generics), params), ret), where_clause), signature), body)| FnItem {
                name,
                generics,
                params,
                ret,
                where_clause,
                body,
                signature,
                doc: None,
            },
        );

    // `field := annotation-group* IDENT ":" type`, with a field name allowed to
    // be spelt like a keyword, having a name space of its own
    let field = annots
        .clone()
        .then(name(cx.kws))
        .then_ignore(punct(Punct::Colon))
        .then(ty.clone())
        .map_with(move |((annotations, name), ty), e| Field {
            annotations,
            doc: cx.outer_docs(e.span().start, name.span.start),
            name,
            ty,
        });
    let fields = listed(field, Punct::LBrace, Punct::RBrace);

    let struct_item = kw(Keyword::Struct)
        .ignore_then(ident())
        .then(generics.clone())
        .then(fields.clone())
        .map(|((name, generics), fields)| ItemKind::Struct {
            name,
            generics,
            fields,
        });

    let variant = name(cx.kws)
        .then(
            choice((
                listed(ty.clone(), Punct::LParen, Punct::RParen).map(VariantPayload::Tuple),
                fields.map(VariantPayload::Struct),
            ))
            .or_not(),
        )
        .map(move |(name, payload)| Variant {
            doc: cx.outer_docs(name.span.start, name.span.start),
            name,
            payload,
        });

    let enum_item = kw(Keyword::Enum)
        .ignore_then(ident())
        .then(generics.clone())
        .then(listed(variant, Punct::LBrace, Punct::RBrace))
        .map(|((name, generics), variants)| ItemKind::Enum {
            name,
            generics,
            variants,
        });

    let impl_item = kw(Keyword::Impl)
        .ignore_then(path(cx))
        .then(generics.clone())
        .then(
            fn_item
                .clone()
                .map_with(move |f, e| FnItem {
                    doc: cx.outer_docs(e.span().start, e.span().start),
                    ..f
                })
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(punct(Punct::LBrace), punct(Punct::RBrace)),
        )
        .map(|((path, generics), items)| ItemKind::Impl {
            path,
            generics,
            items,
        });

    // a unit-scope binding always declares its type, unlike a local one
    let let_item = kw(Keyword::Let)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Colon))
        .then(ty.clone())
        .then(punct(Punct::Eq).ignore_then(expr.clone()).or_not())
        .then_ignore(semi.clone())
        .map(|((name, ty), init)| ItemKind::Let {
            name,
            ty,
            init,
        });

    let type_alias = kw(Keyword::Type)
        .ignore_then(ident())
        .then(generics.clone())
        .then_ignore(punct(Punct::Eq))
        .then(ty.clone())
        .then_ignore(semi.clone())
        .map(|((name, generics), ty)| ItemKind::TypeAlias {
            name,
            generics,
            ty,
        });

    // `import(quantum) dsp;` asks a `#unit any` unit for a kind
    let import_item = kw(Keyword::Import)
        .ignore_then(
            ident()
                .delimited_by(punct(Punct::LParen), punct(Punct::RParen))
                .or_not(),
        )
        .then(path(cx))
        .then(kw(Keyword::As).ignore_then(ident()).or_not())
        .then_ignore(semi.clone())
        .map(|((kind, path), alias)| ItemKind::Import { path, alias, kind });

    // the judgement-system declarations: cover, gauge, base, locale, chain
    // and qmap
    let cover_item = kw(Keyword::Cover)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Eq))
        .then(qcover(cx))
        .then_ignore(semi.clone())
        .map(|(name, cover)| ItemKind::Cover { name, cover });

    let gauge_item = kw(Keyword::Gauge)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Eq))
        .then(qgauge(cx))
        .then_ignore(semi.clone())
        .map(|(name, gauge)| ItemKind::Gauge { name, gauge });

    let base_item = kw(Keyword::Base)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Eq))
        .then(ty.clone())
        .then_ignore(punct(Punct::Comma))
        .then(decl_name())
        .then_ignore(punct(Punct::Comma))
        .then(decl_name())
        .then_ignore(semi.clone())
        .map(|(((name, register), cover), gauge)| ItemKind::Base {
            name,
            register,
            cover,
            gauge,
        });

    // `locale-member := annotation-group* "fn" IDENT "(" params? ")" ret? ";"`
    let locale_member = annots
        .clone()
        .then_ignore(kw(Keyword::Fn))
        .then(ident())
        .then(params.clone())
        .then(punct(Punct::Arrow).ignore_then(ty.clone()).or_not())
        .then_ignore(semi.clone())
        .map_with(move |(((annotations, name), params), ret), e| LocaleMember {
            annotations,
            doc: cx.outer_docs(e.span().start, name.span.start),
            name,
            params,
            ret,
        });

    let locale_item = kw(Keyword::Locale)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Eq))
        .then(ty.clone())
        .then(
            locale_member
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(punct(Punct::LBrace), punct(Punct::RBrace)),
        )
        .map(|((name, ty), members)| ItemKind::Locale { name, ty, members });

    // a chain stage: an operator, then the base types it moves between
    let chain_stage = expr
        .clone()
        .then_ignore(punct(Punct::Colon))
        .then(ident())
        .then_ignore(punct(Punct::Arrow))
        .then(ident())
        .map(|((operator, from), to)| ChainStage { operator, from, to });

    let chain_item = kw(Keyword::Chain)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Eq))
        .then(
            // stages are separated by `;` and the list ends with one, so the
            // separator may not trail or it would eat the terminator
            chain_stage
                .separated_by(semi.clone())
                .at_least(1)
                .collect::<Vec<_>>(),
        )
        .then_ignore(semi.clone())
        .map(|(name, stages)| ItemKind::Chain { name, stages });

    let qmap_entry = expr
        .clone()
        .then_ignore(punct(Punct::Colon))
        .then(expr.clone())
        .map(|(key, value)| QmapEntry { key, value });

    let qmap_item = kw(Keyword::Qmap)
        .ignore_then(ident())
        .then_ignore(punct(Punct::Colon))
        .then(ty.clone())
        .then_ignore(punct(Punct::Arrow))
        .then(ty.clone())
        .then(listed(qmap_entry, Punct::LBrace, Punct::RBrace))
        .map(|(((name, key), entry), entries)| ItemKind::Qmap {
            name,
            key,
            entry,
            entries,
        });

    // `@static_assert(…);` at unit scope, as the chains' examples write it
    // only a `@static_assert` is tried, so that nothing else pays for
    // parsing an expression here
    let assert_item = select! { Token::MacroName(s) if cx.interner.resolve(s) == "static_assert" => () }
        .rewind()
        .ignore_then(expr.clone())
        .then_ignore(punct(Punct::Semi))
        .map(|call| ItemKind::Assert { call })
        .boxed();

    let body = choice((
        fn_item.map(|f| ItemKind::Fn(Box::new(f))),
        struct_item,
        enum_item,
        impl_item,
        let_item,
        type_alias,
        import_item,
        cover_item,
        gauge_item,
        base_item,
        locale_item,
        chain_item,
        qmap_item,
        assert_item,
    ));

    // an item's documentation is anchored anywhere from its first token to
    // its keyword: before, between or after its annotation groups
    annots
        .then(kw(Keyword::Static).or_not())
        .then(body.map_with(|kind, e| (kind, e.span().start)))
        .map_with(move |((annotations, static_), (kind, keyword)), e| {
            let span: Span = e.span();
            Spanned::new(
                Item {
                    annotations,
                    linkage: if static_.is_some() { Linkage::Unit } else { Linkage::Program },
                    doc: cx.outer_docs(span.start, keyword),
                    kind,
                },
                span,
            )
        })
        .labelled("item")
}

#[cfg(test)]
mod tests {
    use crate::ast::item::{FnName, ItemKind, Linkage, VariantPayload};
    use crate::intern::Interner;
    use crate::parse::testing::{errors, unit};

    /// The first item of a unit.
    fn first(src: &str) -> (ItemKind, Interner) {
        let (u, i) = unit(src);
        (u.items.into_iter().next().expect("one item").node.kind, i)
    }

    #[test]
    fn a_function_records_its_parts() {
        let (k, i) = first("fn hypot(a: f64, b: f64) -> f64 { a }");
        match k {
            ItemKind::Fn(f) => {
                assert_eq!(i.resolve(f.name.name()), "hypot");
                assert_eq!(f.params.len(), 2);
                assert!(f.ret.is_some());
                assert!(f.where_clause.is_none());
            }
            other => panic!("expected a function, got {}", other.describe()),
        }
    }

    #[test]
    fn a_where_clause_is_a_constant_expression() {
        // there are no trait bounds; a `where` clause is a constant
        // condition
        let (k, _) = first(
            "fn sum<T>(xs: *[T]) -> T where @has_method(T, \"$add\") { xs[0] }",
        );
        match k {
            ItemKind::Fn(f) => {
                assert_eq!(f.generics.len(), 1);
                assert!(f.where_clause.is_some());
                assert!(
                    f.generics[0].ty.is_none(),
                    "a plain generic parameter carries no bound"
                );
            }
            other => panic!("expected a function, got {}", other.describe()),
        }
    }

    #[test]
    fn a_const_generic_is_a_parameter_of_const_type() {
        // a const generic: `struct Reg<N: const usize> { q: [qubit; N] }`
        let (k, _) = first("struct Reg<N: const usize> { q: [qubit; N] }");
        match k {
            ItemKind::Struct { generics, .. } => {
                assert_eq!(generics.len(), 1);
                assert!(generics[0].ty.as_ref().unwrap().node.is_const());
            }
            other => panic!("expected a structure, got {}", other.describe()),
        }
    }

    #[test]
    fn the_three_function_name_forms_parse() {
        // the three forms a function name may take
        let (k, _) = first("fn plain() { }");
        assert!(matches!(k, ItemKind::Fn(f) if matches!(f.name, FnName::Plain(_))));

        // an external method definition, adding a method from outside the
        // type's own `impl`
        let (k, _) = first("fn Cplx.conj(self: Cplx) -> Cplx { self }");
        assert!(matches!(k, ItemKind::Fn(f) if matches!(f.name, FnName::External { .. })));

        // an operator method
        let (k, _) = first("fn $add(self: T, o: T) -> T { self }");
        assert!(matches!(k, ItemKind::Fn(f) if f.name.is_operator()));
    }

    #[test]
    fn a_self_parameter_is_recognized() {
        // `self` is not a reserved word, so it arrives as an identifier
        let (k, _) = first("fn m(self: *T, x: u8) { }");
        match k {
            ItemKind::Fn(f) => {
                assert!(f.params[0].is_self);
                assert!(!f.params[1].is_self);
            }
            other => panic!("expected a function, got {}", other.describe()),
        }
    }

    #[test]
    fn static_gives_unit_linkage_and_is_the_only_specifier() {
        // there is no export keyword; privacy is what gets marked
        let (u, _) = unit("fn public() { }\nstatic fn private() { }");
        assert_eq!(u.items[0].node.linkage, Linkage::Program);
        assert_eq!(u.items[1].node.linkage, Linkage::Unit);
    }

    #[test]
    fn a_structure_and_its_fields_parse() {
        let (k, i) = first("struct Packet { tag: u8, len: u32, payload: [u8] }");
        match k {
            ItemKind::Struct { fields, .. } => {
                let names: Vec<&str> =
                    fields.iter().map(|f| i.resolve(f.name.node)).collect();
                assert_eq!(names, vec!["tag", "len", "payload"]);
            }
            other => panic!("expected a structure, got {}", other.describe()),
        }
    }

    #[test]
    fn enumerations_parse_in_all_three_variant_shapes() {
        let (k, _) = first("enum E { Unit, Tup(u8, f64), Rec { w: f64 } }");
        match k {
            ItemKind::Enum { variants, .. } => {
                assert!(variants[0].payload.is_none());
                assert!(matches!(
                    variants[1].payload,
                    Some(VariantPayload::Tuple(_))
                ));
                assert!(matches!(
                    variants[2].payload,
                    Some(VariantPayload::Struct(_))
                ));
            }
            other => panic!("expected an enumeration, got {}", other.describe()),
        }
    }

    #[test]
    fn a_quantum_enumeration_is_an_ordinary_enumeration_syntactically() {
        // what makes an enumeration quantum is a qubit-bearing payload, which
        // is a phase-8 question, not a syntactic one
        let (k, _) = first("enum QBit { Zero(qubit), One(qubit) }");
        assert!(matches!(k, ItemKind::Enum { .. }));
    }

    #[test]
    fn an_impl_block_collects_its_methods() {
        let (k, _) = first("impl Cplx { fn re(self: Cplx) -> f64 { 0.0 } fn im(self: Cplx) -> f64 { 0.0 } }");
        match &k {
            ItemKind::Impl { items, .. } => assert_eq!(items.len(), 2),
            other => panic!("expected an impl block, got {}", other.describe()),
        }
        assert_eq!(k.declared_name(), None, "an impl declares no name");
    }

    #[test]
    fn mut_is_an_ordinary_name() {
        // every binding may be assigned, and `*T` may write, so nothing is
        // declared `mut`
        assert!(errors("fn f() { let mut = 1; mut = 2; }").is_empty());
        assert!(!errors("fn f() { let mut x = 0; }").is_empty());
        assert!(!errors("fn f(p: *mut u32) { }").is_empty());
    }

    #[test]
    fn a_unit_scope_binding_always_declares_its_type() {
        // the type is mandatory at unit scope, unlike a local binding
        let (k, _) = first("let VERSION: const u32 = 2;");
        assert!(matches!(k, ItemKind::Let { init: Some(_), .. }));
        let (k, _) = first("let CACHE: [u32];");
        assert!(matches!(k, ItemKind::Let { init: None, .. }));
        assert!(!errors("let X = 1;").is_empty(), "the type is required");
    }

    #[test]
    fn imports_parse_with_and_without_an_alias() {
        let (k, i) = first("import geometry;");
        assert_eq!(k.declared_name().map(|s| i.resolve(s).to_owned()), Some("geometry".to_owned()));
        let (k, i) = first("import qpu::gates as g;");
        assert_eq!(k.declared_name().map(|s| i.resolve(s).to_owned()), Some("g".to_owned()));
    }

    #[test]
    fn a_type_alias_parses() {
        let (k, _) = first("type Square<T, N> = mat<T, N, N>;");
        match k {
            ItemKind::TypeAlias { generics, .. } => assert_eq!(generics.len(), 2),
            other => panic!("expected a type alias, got {}", other.describe()),
        }
    }

    ///////////////////////////////////////
    // THE JUDGEMENT-SYSTEM DECLARATIONS //
    ///////////////////////////////////////

    #[test]
    fn cover_gauge_and_base_declarations_parse() {
        // a cover given by a finite set of states
        let (u, _) = unit(
            "cover B2 = fin{ |00>, |11> };\n\
             gauge GB = fid((|00> + |11>) * isq2);\n\
             base Bell = [qubit; 2], B2, GB;",
        );
        assert_eq!(u.len(), 3);
        assert!(matches!(u.items[0].node.kind, ItemKind::Cover { .. }));
        assert!(matches!(u.items[1].node.kind, ItemKind::Gauge { .. }));
        assert!(matches!(u.items[2].node.kind, ItemKind::Base { .. }));
    }

    #[test]
    fn a_stationary_only_gauge_parses() {
        // `gauge GNone = none;` declares a gauge that fixes no frame at all,
        // which is what a stationary-only cover needs
        let (u, _) = unit("cover UM = span{ |00>, |01>, |10> };\ngauge GN = none;");
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn a_locale_declares_interfaces_not_definitions() {
        // a locale. each member ends with `;` and declares an interface
        // rather than a body
        let (k, _) = first(
            "locale MarkedIn = M of Search {\n\
             [expect: stat(phase::<8>::of(4))]\n\
             fn reflect(r: *[qubit; 2]);\n\
             }",
        );
        match k {
            ItemKind::Locale { members, .. } => {
                assert_eq!(members.len(), 1);
                assert_eq!(members[0].annotations.len(), 1);
                assert_eq!(members[0].params.len(), 1);
            }
            other => panic!("expected a locale, got {}", other.describe()),
        }
    }

    #[test]
    fn a_chain_keeps_its_stages_in_order() {
        // a three-stage pipeline
        let (k, i) = first("chain HXH = h : T01 -> Tpm ; x : Tpm -> Tpm ; h : Tpm -> T01 ;");
        match k {
            ItemKind::Chain { stages, .. } => {
                assert_eq!(stages.len(), 3);
                assert_eq!(i.resolve(stages[0].from.node), "T01");
                assert_eq!(i.resolve(stages[2].to.node), "T01");
                assert_eq!(
                    stages[0].to.node, stages[1].from.node,
                    "each stage's target is the next one's source"
                );
            }
            other => panic!("expected a chain, got {}", other.describe()),
        }
    }

    #[test]
    fn a_single_stage_chain_parses() {
        // a closed chain: `chain Loop = iterate : Torb -> Torb ;`
        let (k, _) = first("chain Loop = iterate : Torb -> Torb ;");
        match k {
            ItemKind::Chain { stages, .. } => assert_eq!(stages.len(), 1),
            other => panic!("expected a chain, got {}", other.describe()),
        }
    }

    #[test]
    fn a_map_locale_parses_with_its_entries() {
        // a map locale, looked up coherently
        let (k, _) = first(
            "qmap Phases: [qubit; 2] -> fn(*[qubit; 1]) { 0: id1, 1: id1, 2: id1, 3: neg1, }",
        );
        match k {
            ItemKind::Qmap { entries, .. } => assert_eq!(entries.len(), 4),
            other => panic!("expected a map locale, got {}", other.describe()),
        }
    }

    /////////////////////////////////////////
    // ANNOTATIONS AND LINKAGE ON ANY ITEM //
    /////////////////////////////////////////

    #[test]
    fn annotation_groups_stack_above_an_item() {
        let (u, _) = unit("[entry]\n[cover: Orb]\n[expect: contract = cyc]\nfn f() { }");
        assert_eq!(u.items[0].node.annotations.len(), 3);
    }

    ///////////////////
    // DOCUMENTATION //
    ///////////////////

    /// The text of a declaration's documentation.
    fn text(doc: &Option<crate::ast::Doc>) -> Option<&str> {
        doc.as_ref().map(|d| d.text.as_str())
    }

    #[test]
    fn an_item_takes_the_doc_comments_before_it() {
        let (u, _) = unit("/// Adds.\n///\n/// Twice.\nfn add() { }\n\n// plain\nfn bare() { }");
        assert_eq!(text(&u.items[0].node.doc), Some("Adds.\n\nTwice."));
        assert_eq!(text(&u.items[1].node.doc), None, "a plain comment documents nothing");
    }

    #[test]
    fn documentation_may_sit_among_the_annotations() {
        let (u, _) = unit("/// One.\n[entry]\n/// Two.\n[cover: Orb]\n/// Three.\nstatic fn f() { }");
        assert_eq!(text(&u.items[0].node.doc), Some("One.\nTwo.\nThree."));
    }

    #[test]
    fn fields_variants_and_methods_have_their_own_documentation() {
        let (u, _) = unit(
            "/// A point.\nstruct P {\n    /// Across.\n    x: i32,\n    y: i32,\n}\n\
             enum E {\n    /// The first.\n    A,\n    B(u8),\n}\n\
             impl P {\n    /// Moves.\n    fn go(self: *P) { }\n}",
        );
        match &u.items[0].node.kind {
            ItemKind::Struct { fields, .. } => {
                assert_eq!(text(&fields[0].doc), Some("Across."));
                assert_eq!(text(&fields[1].doc), None);
            }
            other => panic!("expected a structure, got {}", other.describe()),
        }
        assert_eq!(text(&u.items[0].node.doc), Some("A point."), "a field's doc is not the item's");
        match &u.items[1].node.kind {
            ItemKind::Enum { variants, .. } => {
                assert_eq!(text(&variants[0].doc), Some("The first."));
                assert_eq!(text(&variants[1].doc), None);
            }
            other => panic!("expected an enumeration, got {}", other.describe()),
        }
        match &u.items[2].node.kind {
            ItemKind::Impl { items, .. } => assert_eq!(text(&items[0].doc), Some("Moves.")),
            other => panic!("expected an impl block, got {}", other.describe()),
        }
    }

    #[test]
    fn a_locale_member_has_its_own_documentation() {
        let (k, _) = first("locale L = M of S {\n    /// Reflects.\n    [expect: stat(1)]\n    fn reflect(r: *[qubit; 2]);\n}");
        match k {
            ItemKind::Locale { members, .. } => assert_eq!(text(&members[0].doc), Some("Reflects.")),
            other => panic!("expected a locale, got {}", other.describe()),
        }
    }

    #[test]
    fn a_doc_comment_before_a_statement_is_an_ordinary_comment() {
        assert!(errors("fn f() {\n    /// not a declaration\n    let x = 1;\n}").is_empty());
    }

    #[test]
    fn a_functions_signature_stops_before_its_body() {
        let src = "fn sum<T>(xs: *[T]) -> T where @has_method(T, \"$add\") { xs[0] }";
        let (k, _) = first(src);
        match k {
            ItemKind::Fn(f) => {
                assert_eq!(&src[f.signature.range()], "fn sum<T>(xs: *[T]) -> T where @has_method(T, \"$add\")");
            }
            other => panic!("expected a function, got {}", other.describe()),
        }
    }

    #[test]
    fn malformed_items_are_rejected() {
        for src in [
            "fn f(",
            "struct S { x }",
            "enum E { ",
            "import ;",
            "base B = [qubit; 1];",
            "chain C = ;",
        ] {
            assert!(!errors(src).is_empty(), "{src:?} should not parse");
        }
    }
}
