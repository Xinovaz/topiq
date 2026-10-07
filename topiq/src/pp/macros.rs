//! The macro table.
//!
//! Two things here are Topiq's own rather than inherited from C:
//!
//! - **A macro has linkage.** Program linkage is the default, and `static
//!   #define` makes a macro private. Still, no other unit can use one: an
//!   importer learns what it imports only after its own preprocessing, too
//!   late for a macro to expand. So macros stay out of a unit's metadata.
//! - **The eight predefined names are closed.** `#define` or `#undef` of one is
//!   `EA04`, or a name describing the translation could change meaning
//!   halfway through a file.
//!
//! `@`-prefixed macros are not preprocessor macros: they expand much later,
//! with types known, so the preprocessor carries [`Token::MacroName`] through
//! as an ordinary token.

use std::collections::HashMap;

use crate::intern::{Interner, Symbol};
use crate::lex::Token;
use crate::span::Span;

use super::hide::Hide;

/// A token as it passes through the preprocessor, carrying its hide set.
#[derive(Clone, Debug)]
pub struct PpToken {
    /// The token.
    pub tok: Token,
    /// Where it came from.
    pub span: Span,
    /// The macros that may not expand this token again.
    pub hide: Hide,
}

impl PpToken {
    /// A token with an empty hide set.
    pub fn new(tok: Token, span: Span) -> PpToken {
        PpToken {
            tok,
            span,
            hide: Hide::empty(),
        }
    }
}

/// Whether a macro is exported with its unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Linkage {
    /// The default. No importer sees a macro either way; see the module
    /// documentation.
    Program,
    /// Private to the unit, written `static #define`.
    Unit,
}

/// One macro definition.
#[derive(Clone, Debug)]
pub struct MacroDef {
    /// The macro's name.
    pub name: Symbol,
    /// Parameters, or `None` for an object-like macro.
    ///
    /// The distinction matters at the use site, because expansion requires the
    /// `(` to follow the name immediately for a function-like macro to be
    /// invoked at all.
    pub params: Option<Vec<Symbol>>,
    /// Whether the parameter list ended in `...`, making `__VA_ARGS__`
    /// available in the body.
    pub variadic: bool,
    /// The replacement list.
    pub body: Vec<PpToken>,
    /// Whether the macro is exported.
    pub linkage: Linkage,
    /// Where it was defined.
    pub span: Span,
    /// Whether this is one of the eight predefined names.
    pub predefined: bool,
}

impl MacroDef {
    /// Whether this macro is invoked as `NAME(args)`.
    pub fn is_function_like(&self) -> bool {
        self.params.is_some()
    }

    /// The number of declared parameters.
    pub fn arity(&self) -> usize {
        self.params.as_ref().map_or(0, Vec::len)
    }

    /// Whether `count` supplied arguments match this macro's parameter list.
    ///
    /// A variadic macro accepts its declared parameters or more; a plain one
    /// requires exactly its arity. The single-empty-argument case is how a
    /// zero-parameter macro is called: `F()` supplies one empty argument.
    pub fn accepts(&self, count: usize) -> bool {
        match &self.params {
            None => false,
            Some(ps) if self.variadic => count >= ps.len(),
            Some(ps) => count == ps.len() || (ps.is_empty() && count == 1),
        }
    }
}

/// The eight names the compiler predefines in every unit.
///
/// The list is closed, and a program may not `#define` or `#undef` any of them
/// (`EA04`).
pub const PREDEFINED: &[&str] = &[
    "__TOPIQ__",
    "__FILE__",
    "__LINE__",
    "__UNIT__",
    "__UNIT_KIND__",
    "__CLASSICAL__",
    "__QUANTUM__",
    "__CONDUCTOR__",
];

/// Whether `name` is one of the predefined macro names.
pub fn is_predefined(name: &str) -> bool {
    PREDEFINED.contains(&name)
}

/// The macro table for one unit.
#[derive(Clone, Debug, Default)]
pub struct MacroTable {
    defs: HashMap<Symbol, MacroDef>,
}

impl MacroTable {
    /// An empty table.
    pub fn new() -> MacroTable {
        MacroTable::default()
    }

    /// Defines a macro, returning the definition it replaced if any.
    pub fn define(&mut self, def: MacroDef) -> Option<MacroDef> {
        self.defs.insert(def.name, def)
    }

    /// Removes a macro, returning it if it was defined.
    pub fn undef(&mut self, name: Symbol) -> Option<MacroDef> {
        self.defs.remove(&name)
    }

    /// Looks a macro up.
    pub fn get(&self, name: Symbol) -> Option<&MacroDef> {
        self.defs.get(&name)
    }

    /// Whether a macro is defined, which is what `defined(NAME)` in a `#if`
    /// asks.
    pub fn is_defined(&self, name: Symbol) -> bool {
        self.defs.contains_key(&name)
    }

    /// How many macros are defined.
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    /// Whether no macros are defined.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Every macro with program linkage.
    pub fn exported(&self) -> Vec<&MacroDef> {
        let mut out: Vec<&MacroDef> = self
            .defs
            .values()
            .filter(|d| d.linkage == Linkage::Program && !d.predefined)
            .collect();
        out.sort_by_key(|d| d.name.index());
        out
    }

    /// Installs the eight predefined macros.
    ///
    /// `__LINE__` and `__CONDUCTOR__` have empty bodies, since the expander
    /// computes them where they are used; they are here so that `defined`
    /// sees them and redefining one is `EA04`.
    pub fn install_predefined(
        &mut self,
        interner: &mut Interner,
        span: Span,
        edition: u32,
        file: &str,
        unit: &str,
        unit_kind: u8,
    ) {
        let mut object = |name: &str, body: Vec<PpToken>, interner: &mut Interner| {
            let sym = interner.intern(name);
            self.defs.insert(
                sym,
                MacroDef {
                    name: sym,
                    params: None,
                    variadic: false,
                    body,
                    linkage: Linkage::Unit,
                    span,
                    predefined: true,
                },
            );
        };

        let edition_tok = int_token(interner, edition as u64, span);
        object("__TOPIQ__", vec![edition_tok], interner);

        let file_tok = str_token(interner, file, span);
        object("__FILE__", vec![file_tok], interner);

        let unit_tok = str_token(interner, unit, span);
        object("__UNIT__", vec![unit_tok], interner);

        let kind_tok = int_token(interner, unit_kind as u64, span);
        object("__UNIT_KIND__", vec![kind_tok], interner);

        // what `__UNIT_KIND__` is in each kind of unit, so that a test of it
        // names the kind
        let classical = int_token(interner, 1, span);
        object("__CLASSICAL__", vec![classical], interner);
        let quantum = int_token(interner, 2, span);
        object("__QUANTUM__", vec![quantum], interner);

        // computed at the use site
        object("__LINE__", Vec::new(), interner);
        object("__CONDUCTOR__", Vec::new(), interner);
    }
}

/// Builds an integer-literal token holding `value`.
pub fn int_token(interner: &Interner, value: u64, span: Span) -> PpToken {
    let raw = interner.intern_late(&value.to_string());
    PpToken::new(
        Token::Int {
            raw,
            base: crate::lex::IntBase::Decimal,
            suffix: None,
        },
        span,
    )
}

/// Builds a string-literal token holding `text`.
pub fn str_token(interner: &Interner, text: &str, span: Span) -> PpToken {
    let value = interner.intern_late(text);
    PpToken::new(
        Token::Str {
            value,
            kind: crate::lex::StrKind::Normal,
        },
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SourceId;

    fn span() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    fn def(interner: &mut Interner, name: &str, params: Option<&[&str]>, variadic: bool) -> MacroDef {
        let name_sym = interner.intern(name);
        let params = params.map(|ps| ps.iter().map(|p| interner.intern(p)).collect());
        MacroDef {
            name: name_sym,
            params,
            variadic,
            body: Vec::new(),
            linkage: Linkage::Program,
            span: span(),
            predefined: false,
        }
    }

    #[test]
    fn object_and_function_like_macros_are_distinguished() {
        let mut i = Interner::new();
        assert!(!def(&mut i, "X", None, false).is_function_like());
        assert!(def(&mut i, "F", Some(&["a"]), false).is_function_like());
    }

    #[test]
    fn arity_is_checked_exactly_for_a_plain_macro() {
        let mut i = Interner::new();
        let f = def(&mut i, "F", Some(&["a", "b"]), false);
        assert_eq!(f.arity(), 2);
        assert!(f.accepts(2));
        assert!(!f.accepts(1));
        assert!(!f.accepts(3));
    }

    #[test]
    fn a_zero_parameter_macro_accepts_one_empty_argument() {
        // `#define F() 1` invoked as `F()` supplies a single empty argument
        let mut i = Interner::new();
        let f = def(&mut i, "F", Some(&[]), false);
        assert_eq!(f.arity(), 0);
        assert!(f.accepts(0));
        assert!(f.accepts(1));
        assert!(!f.accepts(2));
    }

    #[test]
    fn a_variadic_macro_accepts_its_parameters_or_more() {
        let mut i = Interner::new();
        let f = def(&mut i, "F", Some(&["a"]), true);
        assert!(!f.accepts(0));
        assert!(f.accepts(1));
        assert!(f.accepts(5));
    }

    #[test]
    fn define_and_undef_round_trip() {
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        let x = i.intern("X");
        assert!(!t.is_defined(x));
        assert!(t.is_empty());

        t.define(def(&mut i, "X", None, false));
        assert!(t.is_defined(x));
        assert_eq!(t.len(), 1);

        assert!(t.undef(x).is_some());
        assert!(!t.is_defined(x));
        assert!(t.undef(x).is_none(), "undef of an undefined macro is a no-op");
    }

    #[test]
    fn redefining_returns_the_previous_definition() {
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        t.define(def(&mut i, "X", None, false));
        let old = t.define(def(&mut i, "X", Some(&["a"]), false));
        assert!(old.is_some());
        assert!(t.get(i.intern("X")).unwrap().is_function_like());
    }

    #[test]
    fn only_program_linkage_macros_are_exported() {
        // `static #define` keeps a macro private to its unit
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        t.define(def(&mut i, "PUBLIC", None, false));
        let mut private = def(&mut i, "PRIVATE", None, false);
        private.linkage = Linkage::Unit;
        t.define(private);

        let exported: Vec<&str> = t
            .exported()
            .iter()
            .map(|d| i.resolve(d.name))
            .collect();
        assert_eq!(exported, vec!["PUBLIC"]);
    }

    #[test]
    fn the_predefined_list_is_closed_and_has_eight_names() {
        assert_eq!(PREDEFINED.len(), 8);
        for name in PREDEFINED {
            assert!(is_predefined(name), "{name} should be predefined");
        }
        assert!(!is_predefined("__NOT_REAL__"));
        assert!(!is_predefined("__FILE"));
    }

    #[test]
    fn installing_the_predefined_macros_defines_all_eight() {
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        t.install_predefined(&mut i, span(), 2, "demo.tq", "demo", 1);
        assert_eq!(t.len(), 8);
        for name in PREDEFINED {
            let sym = i.intern(name);
            assert!(t.is_defined(sym), "{name} should be defined");
            assert!(t.get(sym).unwrap().predefined);
        }
    }

    #[test]
    fn predefined_macros_are_not_exported() {
        // they are supplied by the implementation in every unit, so exporting
        // them would collide at link time
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        t.install_predefined(&mut i, span(), 2, "demo.tq", "demo", 1);
        assert!(t.exported().is_empty());
    }

    #[test]
    fn predefined_macros_have_their_values() {
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        t.install_predefined(&mut i, span(), 2, "src/demo.tq", "demo", 2);

        let body_of = |t: &MacroTable, i: &mut Interner, name: &str| {
            t.get(i.intern(name)).unwrap().body.clone()
        };

        // __TOPIQ__ is the edition, as an integer literal
        match body_of(&t, &mut i, "__TOPIQ__")[0].tok {
            Token::Int { raw, .. } => assert_eq!(i.resolve(raw), "2"),
            other => panic!("__TOPIQ__ should be an integer literal, got {other:?}"),
        }
        // __UNIT_KIND__ is 1 in a classical unit and 2 in a quantum one
        match body_of(&t, &mut i, "__UNIT_KIND__")[0].tok {
            Token::Int { raw, .. } => assert_eq!(i.resolve(raw), "2"),
            other => panic!("__UNIT_KIND__ should be an integer literal, got {other:?}"),
        }
        // __FILE__ is the source file name, as a string literal
        match body_of(&t, &mut i, "__FILE__")[0].tok {
            Token::Str { value, .. } => assert_eq!(i.resolve(value), "src/demo.tq"),
            other => panic!("__FILE__ should be a string literal, got {other:?}"),
        }
        // __UNIT__ is the file stem
        match body_of(&t, &mut i, "__UNIT__")[0].tok {
            Token::Str { value, .. } => assert_eq!(i.resolve(value), "demo"),
            other => panic!("__UNIT__ should be a string literal, got {other:?}"),
        }
    }

    #[test]
    fn line_and_conductor_are_computed_at_the_use_site() {
        // computed where used; held only for `defined` and `EA04`
        let mut i = Interner::new();
        let mut t = MacroTable::new();
        t.install_predefined(&mut i, span(), 2, "demo.tq", "demo", 1);
        assert!(t.get(i.intern("__LINE__")).unwrap().body.is_empty());
        assert!(t.get(i.intern("__CONDUCTOR__")).unwrap().body.is_empty());
    }
}
