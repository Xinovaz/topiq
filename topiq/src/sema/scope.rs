//! Which declaration a name refers to.
//!
//! Two kinds of scope behave differently, and the difference is the point of
//! this module.
//!
//! **Unit scope is unordered.** A function may call another declared further
//! down the file, and a constant may be defined in terms of one declared after
//! it. So unit scope is a plain map, filled completely before any body is
//! looked at.
//!
//! **Block scope is ordered.** A `let` binding is visible from the end of its
//! own statement to the end of its block, not before, and not inside its own
//! initialiser. A later `let` of the same name *shadows* the earlier one for
//! the rest of the block, which is how a value is refined step by step without
//! inventing new names:
//!
//! ```text
//! let n = read();
//! let n = n * 2;      // the `n` on the right is the one above
//! ```
//!
//! Each block is therefore a list searched from its end, inside a stack of
//! blocks searched from the innermost out, with unit scope underneath.
//!
//! # Name spaces
//!
//! Values (functions, objects, bindings) and types are named separately, so
//! a structure `Point` and a function `Point` may coexist. The name of an
//! imported unit belongs to both: `geometry::area` and `geometry::Point` both
//! start with it, so a unit-scope item named `geometry` alongside
//! `import geometry;` would make every such path ambiguous, and is refused.

use std::collections::HashMap;

use crate::intern::Symbol;
use crate::span::Span;
use crate::tir::{AdtId, ExternId, FnId, GlobalId, Intrinsic, LocalId};

/// What a name in the value name space denotes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Def {
    /// A parameter or local binding of the function being checked.
    Local(LocalId),
    /// A unit-scope, persistent or imported object or constant.
    Global(GlobalId),
    /// A function of the unit.
    Fn(FnId),
    /// A function of another unit.
    Extern(ExternId),
    /// An operation the language provides (e.g. `print`).
    Intrinsic(Intrinsic),
    /// Inside a closure's body, what the closure captured, by its position
    /// in the capture list.
    Capture(u32),
}

/// A unit this one imports.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UnitRef(pub u32);

impl UnitRef {
    /// The unit being analysed, named by another unit whose declarations
    /// are open here: one of units that import one another, whose bodies
    /// are checked here and name this one.
    pub const SELF: UnitRef = UnitRef(u32::MAX);
}

/// Why a name could not be declared: something already has it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Taken {
    /// Where the earlier declaration is.
    pub earlier: Span,
}

/// The declarations at unit scope.
#[derive(Clone, Debug, Default)]
pub struct UnitScope {
    values: HashMap<Symbol, (Def, Span)>,
    types: HashMap<Symbol, (AdtId, Span)>,
    units: HashMap<Symbol, (UnitRef, Span)>,
    /// Types the unit declares in a form the compiler does not analyse, by
    /// the kind of declaration. Naming one reports that, rather than claiming
    /// the type does not exist.
    unsupported_types: HashMap<Symbol, &'static str>,
}

impl UnitScope {
    /// An empty unit scope.
    pub fn new() -> UnitScope {
        UnitScope::default()
    }

    /// Declares a value, unless the name is already a value or an imported
    /// unit.
    ///
    /// # Errors
    ///
    /// The earlier declaration's span, if the name was already taken (in
    /// which case the earlier declaration stands).
    pub fn declare(&mut self, name: Symbol, def: Def, span: Span) -> Result<(), Taken> {
        if let Some(&(_, earlier)) = self.values.get(&name) {
            return Err(Taken { earlier });
        }
        if let Some(&(_, earlier)) = self.units.get(&name) {
            return Err(Taken { earlier });
        }
        self.values.insert(name, (def, span));
        Ok(())
    }

    /// Declares a type, unless the name is already a type or an imported
    /// unit.
    ///
    /// # Errors
    ///
    /// As for [`UnitScope::declare`].
    pub fn declare_type(&mut self, name: Symbol, id: AdtId, span: Span) -> Result<(), Taken> {
        if let Some(&(_, earlier)) = self.types.get(&name) {
            return Err(Taken { earlier });
        }
        if let Some(&(_, earlier)) = self.units.get(&name) {
            return Err(Taken { earlier });
        }
        self.types.insert(name, (id, span));
        Ok(())
    }

    /// Declares the name an imported unit is known by, unless anything at
    /// unit scope already has that name.
    ///
    /// # Errors
    ///
    /// As for [`UnitScope::declare`].
    pub fn declare_unit(&mut self, name: Symbol, unit: UnitRef, span: Span) -> Result<(), Taken> {
        let earlier = self
            .values
            .get(&name)
            .map(|&(_, s)| s)
            .or_else(|| self.types.get(&name).map(|&(_, s)| s))
            .or_else(|| self.units.get(&name).map(|&(_, s)| s));
        if let Some(earlier) = earlier {
            return Err(Taken { earlier });
        }
        self.units.insert(name, (unit, span));
        Ok(())
    }

    /// What a name denotes in the value name space at unit scope.
    pub fn get(&self, name: Symbol) -> Option<Def> {
        self.values.get(&name).map(|&(d, _)| d)
    }

    /// The structure or enumeration a name denotes at unit scope.
    pub fn get_type(&self, name: Symbol) -> Option<AdtId> {
        self.types.get(&name).map(|&(t, _)| t)
    }

    /// The imported unit a name denotes.
    pub fn get_unit(&self, name: Symbol) -> Option<UnitRef> {
        self.units.get(&name).map(|&(u, _)| u)
    }

    /// Where a value was declared at unit scope.
    pub fn span_of(&self, name: Symbol) -> Option<Span> {
        self.values.get(&name).map(|&(_, s)| s)
    }

    /// Records a type the unit declares in a form the compiler does not analyse.
    pub fn note_unsupported_type(&mut self, name: Symbol, what: &'static str) {
        self.unsupported_types.insert(name, what);
    }

    /// If `name` is such a type, what kind of declaration introduced it.
    pub fn unsupported_type(&self, name: Symbol) -> Option<&'static str> {
        self.unsupported_types.get(&name).copied()
    }

    /// Every value name declared at unit scope, for suggesting a correction.
    pub fn names(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.values.keys().copied()
    }

    /// Every type name declared at unit scope, for suggesting a correction.
    pub fn type_names(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.types.keys().copied()
    }
}

/// The stack of block scopes inside one function.
#[derive(Clone, Debug, Default)]
pub struct Scopes {
    frames: Vec<Vec<(Symbol, Def)>>,
}

impl Scopes {
    /// No blocks open.
    pub fn new() -> Scopes {
        Scopes::default()
    }

    /// Opens a block.
    pub fn enter(&mut self) {
        self.frames.push(Vec::new());
    }

    /// Closes the innermost block, forgetting its bindings.
    pub fn leave(&mut self) {
        self.frames.pop();
    }

    /// How many blocks are open.
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Binds a name in the innermost block. A name already bound there is
    /// shadowed, not replaced: nothing that resolved to it earlier changes.
    ///
    /// # Panics
    ///
    /// If no block is open, which is a bug in the caller.
    pub fn bind(&mut self, name: Symbol, def: Def) {
        self.frames
            .last_mut()
            .expect("a binding needs an open block")
            .push((name, def));
    }

    /// What a name denotes here: the innermost, latest binding, then unit
    /// scope.
    pub fn resolve(&self, name: Symbol, unit: &UnitScope) -> Option<Def> {
        self.frames
            .iter()
            .rev()
            .flat_map(|f| f.iter().rev())
            .find(|(n, _)| *n == name)
            .map(|&(_, d)| d)
            .or_else(|| unit.get(name))
    }

    /// What a name denotes in the open blocks alone, without unit scope.
    pub fn in_blocks(&self, name: Symbol) -> Option<Def> {
        self.frames
            .iter()
            .rev()
            .flat_map(|f| f.iter().rev())
            .find(|(n, _)| *n == name)
            .map(|&(_, d)| d)
    }

    /// Every name visible here, innermost first, for suggesting a correction.
    pub fn visible<'a>(&'a self, unit: &'a UnitScope) -> impl Iterator<Item = Symbol> + 'a {
        self.frames
            .iter()
            .rev()
            .flat_map(|f| f.iter().rev().map(|&(n, _)| n))
            .chain(unit.names())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::SourceId;

    fn sp(n: u32) -> Span {
        Span::new(SourceId(0), n, n + 1)
    }

    #[test]
    fn unit_scope_keeps_the_first_declaration_and_reports_the_second() {
        let mut i = Interner::new();
        let f = i.intern("f");
        let mut u = UnitScope::new();
        assert!(u.declare(f, Def::Fn(FnId(0)), sp(1)).is_ok());
        assert_eq!(
            u.declare(f, Def::Fn(FnId(1)), sp(9)),
            Err(Taken { earlier: sp(1) })
        );
        assert_eq!(u.get(f), Some(Def::Fn(FnId(0))));
        assert_eq!(u.span_of(f), Some(sp(1)));
    }

    #[test]
    fn a_type_and_a_value_may_share_a_name() {
        let mut i = Interner::new();
        let p = i.intern("Point");
        let mut u = UnitScope::new();
        u.declare_type(p, AdtId(0), sp(0)).unwrap();
        assert!(u.declare(p, Def::Fn(FnId(0)), sp(5)).is_ok());
        assert_eq!(u.get_type(p), Some(AdtId(0)));
        assert!(u.declare_type(p, AdtId(1), sp(7)).is_err());
    }

    #[test]
    fn an_imported_unit_collides_with_anything_of_its_name() {
        let mut i = Interner::new();
        let g = i.intern("geometry");
        let mut u = UnitScope::new();
        u.declare_unit(g, UnitRef(0), sp(0)).unwrap();
        assert!(u.declare(g, Def::Fn(FnId(0)), sp(3)).is_err());
        assert!(u.declare_type(g, AdtId(0), sp(4)).is_err());
        let h = i.intern("h");
        u.declare(h, Def::Fn(FnId(1)), sp(5)).unwrap();
        assert_eq!(u.declare_unit(h, UnitRef(1), sp(6)), Err(Taken { earlier: sp(5) }));
        assert_eq!(u.get_unit(g), Some(UnitRef(0)));
    }

    #[test]
    fn a_later_let_in_the_same_block_shadows_an_earlier_one() {
        let mut i = Interner::new();
        let n = i.intern("n");
        let u = UnitScope::new();
        let mut s = Scopes::new();
        s.enter();
        s.bind(n, Def::Local(LocalId(0)));
        assert_eq!(s.resolve(n, &u), Some(Def::Local(LocalId(0))));
        s.bind(n, Def::Local(LocalId(1)));
        assert_eq!(s.resolve(n, &u), Some(Def::Local(LocalId(1))));
    }

    #[test]
    fn leaving_a_block_uncovers_the_outer_binding() {
        let mut i = Interner::new();
        let x = i.intern("x");
        let u = UnitScope::new();
        let mut s = Scopes::new();
        s.enter();
        s.bind(x, Def::Local(LocalId(0)));
        s.enter();
        s.bind(x, Def::Local(LocalId(1)));
        assert_eq!(s.resolve(x, &u), Some(Def::Local(LocalId(1))));
        s.leave();
        assert_eq!(s.resolve(x, &u), Some(Def::Local(LocalId(0))));
        s.leave();
        assert_eq!(s.resolve(x, &u), None);
        assert_eq!(s.depth(), 0);
    }

    #[test]
    fn a_local_hides_a_unit_scope_declaration_of_the_same_name() {
        let mut i = Interner::new();
        let limit = i.intern("LIMIT");
        let mut u = UnitScope::new();
        u.declare(limit, Def::Global(GlobalId(0)), sp(0)).unwrap();
        let mut s = Scopes::new();
        s.enter();
        assert_eq!(s.resolve(limit, &u), Some(Def::Global(GlobalId(0))));
        s.bind(limit, Def::Local(LocalId(3)));
        assert_eq!(s.resolve(limit, &u), Some(Def::Local(LocalId(3))));
    }

    #[test]
    fn visible_names_include_both_kinds_of_scope() {
        let mut i = Interner::new();
        let (a, b) = (i.intern("a"), i.intern("b"));
        let mut u = UnitScope::new();
        u.declare(b, Def::Fn(FnId(0)), sp(0)).unwrap();
        let mut s = Scopes::new();
        s.enter();
        s.bind(a, Def::Local(LocalId(0)));
        let names: Vec<_> = s.visible(&u).collect();
        assert_eq!(names, vec![a, b]);
    }

    #[test]
    fn an_unsupported_type_is_remembered_by_name() {
        let mut i = Interner::new();
        let alias = i.intern("Alias");
        let mut u = UnitScope::new();
        u.note_unsupported_type(alias, "type aliases");
        assert_eq!(u.unsupported_type(alias), Some("type aliases"));
        assert_eq!(u.unsupported_type(i.intern("Other")), None);
    }
}
