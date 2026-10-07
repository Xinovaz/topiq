//! Hide sets.
//!
//! A macro is never re-expanded within its own expansion: otherwise
//! `#define X X` would not terminate. The rule is made precise by Prosser's
//! algorithm, in which every token carries the set of macro names that may
//! not expand it again:
//!
//! - an **object-like** macro `M` with hide set *HS* produces tokens whose hide
//!   set is *HS* ∪ {`M`};
//! - a **function-like** macro `M` whose name carried *HS* and whose closing
//!   parenthesis carried *HS'* produces tokens whose hide set is
//!   (*HS* ∩ *HS'*) ∪ {`M`}.
//!
//! The intersection in the second rule is what makes `#define f(x) x` behave
//! when an argument itself came from a partially expanded macro; a naive
//! "currently expanding" stack gets that case wrong.
//!
//! The limit on expansion depth is a separate, cruder guard, against runaway
//! expansion that hide sets permit, such as mutually recursive function-like
//! macros that add a token each round.
//!
//! Hide sets are shared immutable linked lists, so copying a token is cheap and
//! the common case (an empty hide set) allocates nothing.

use std::rc::Rc;

use crate::intern::Symbol;

#[derive(Debug)]
struct Node {
    name: Symbol,
    next: Hide,
}

/// The set of macro names that may not expand a token again.
#[derive(Clone, Default, Debug)]
pub struct Hide(Option<Rc<Node>>);

impl Hide {
    /// The empty hide set.
    pub fn empty() -> Hide {
        Hide(None)
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Whether `name` is hidden.
    pub fn contains(&self, name: Symbol) -> bool {
        let mut cur = self.0.as_deref();
        while let Some(node) = cur {
            if node.name == name {
                return true;
            }
            cur = node.next.0.as_deref();
        }
        false
    }

    /// The set with `name` added. Adding a name already present is a no-op, so
    /// the list cannot grow without bound.
    pub fn add(&self, name: Symbol) -> Hide {
        if self.contains(name) {
            return self.clone();
        }
        Hide(Some(Rc::new(Node {
            name,
            next: self.clone(),
        })))
    }

    /// Every name in the set, innermost first.
    pub fn names(&self) -> Vec<Symbol> {
        let mut out = Vec::new();
        let mut cur = self.0.as_deref();
        while let Some(node) = cur {
            out.push(node.name);
            cur = node.next.0.as_deref();
        }
        out
    }

    /// How many names are hidden.
    pub fn len(&self) -> usize {
        self.names().len()
    }

    /// The union of two hide sets.
    pub fn union(&self, other: &Hide) -> Hide {
        let mut out = self.clone();
        for name in other.names() {
            out = out.add(name);
        }
        out
    }

    /// The intersection of two hide sets.
    pub fn intersect(&self, other: &Hide) -> Hide {
        let mut out = Hide::empty();
        for name in self.names() {
            if other.contains(name) {
                out = out.add(name);
            }
        }
        out
    }
}

impl PartialEq for Hide {
    /// Set equality, not list equality: the same names in a different insertion
    /// order compare equal.
    fn eq(&self, other: &Hide) -> bool {
        let (a, b) = (self.names(), other.names());
        a.len() == b.len() && a.iter().all(|n| other.contains(*n)) && b.iter().all(|n| self.contains(*n))
    }
}

impl Eq for Hide {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;

    fn syms(n: usize) -> (Interner, Vec<Symbol>) {
        let mut i = Interner::new();
        let v = (0..n).map(|k| i.intern(&format!("M{k}"))).collect();
        (i, v)
    }

    #[test]
    fn an_empty_set_hides_nothing() {
        let (_, s) = syms(1);
        let h = Hide::empty();
        assert!(h.is_empty());
        assert!(!h.contains(s[0]));
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn adding_makes_a_name_hidden() {
        let (_, s) = syms(2);
        let h = Hide::empty().add(s[0]);
        assert!(h.contains(s[0]));
        assert!(!h.contains(s[1]));
        assert!(!h.is_empty());
    }

    #[test]
    fn adding_is_idempotent_so_the_list_cannot_grow_unboundedly() {
        let (_, s) = syms(1);
        let mut h = Hide::empty();
        for _ in 0..1000 {
            h = h.add(s[0]);
        }
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn the_original_set_is_unchanged_by_adding() {
        let (_, s) = syms(2);
        let a = Hide::empty().add(s[0]);
        let b = a.add(s[1]);
        assert!(!a.contains(s[1]), "sets are persistent, not mutated");
        assert!(b.contains(s[0]) && b.contains(s[1]));
    }

    #[test]
    fn union_collects_both_sides() {
        let (_, s) = syms(3);
        let a = Hide::empty().add(s[0]).add(s[1]);
        let b = Hide::empty().add(s[1]).add(s[2]);
        let u = a.union(&b);
        assert!(u.contains(s[0]) && u.contains(s[1]) && u.contains(s[2]));
        assert_eq!(u.len(), 3, "the shared name is not duplicated");
    }

    #[test]
    fn intersection_keeps_only_shared_names() {
        // the function-like rule: intersect, then add the macro's own name
        let (_, s) = syms(3);
        let a = Hide::empty().add(s[0]).add(s[1]);
        let b = Hide::empty().add(s[1]).add(s[2]);
        let i = a.intersect(&b);
        assert!(i.contains(s[1]));
        assert!(!i.contains(s[0]) && !i.contains(s[2]));
        assert_eq!(i.len(), 1);
    }

    #[test]
    fn intersecting_with_the_empty_set_gives_the_empty_set() {
        let (_, s) = syms(2);
        let a = Hide::empty().add(s[0]).add(s[1]);
        assert!(a.intersect(&Hide::empty()).is_empty());
        assert!(Hide::empty().intersect(&a).is_empty());
    }

    #[test]
    fn equality_ignores_insertion_order() {
        let (_, s) = syms(2);
        let a = Hide::empty().add(s[0]).add(s[1]);
        let b = Hide::empty().add(s[1]).add(s[0]);
        assert_eq!(a, b);
        assert_ne!(a, Hide::empty().add(s[0]));
    }

    #[test]
    fn a_self_referential_macro_is_hidden_from_itself() {
        // `#define X X` must expand to `X` once and then stop
        let (_, s) = syms(1);
        let after_first = Hide::empty().add(s[0]);
        assert!(
            after_first.contains(s[0]),
            "the expansion of X may not be expanded by X again"
        );
    }
}
