//! The types of one unit: its structures and enumerations, and every compound
//! type it mentions.
//!
//! A compound type (`*T`, `*[T]`, `[T; N]`) is stored once, however many
//! times it is written, and named by its [`CompoundId`]. That keeps [`Ty`]
//! small enough to copy freely, and makes "the same type" mean "the same
//! value": `[u8; 4]` written in two places interns to one id, so the two
//! compare equal without a structural walk.
//!
//! Structures and enumerations are *not* interned by shape. Two declarations
//! with identical fields are still two types; what makes a type is its
//! declaration, identified by the unit that declared it and its name.

use std::collections::HashMap;

use crate::intern::{Interner, Symbol};

use super::adt::AdtDef;
use super::ty::{Access, AdtId, Arg, CompoundId, SigId, TupleId, Ty};

/// A compound type's shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Compound {
    /// `*T` or `*const T`.
    Ref {
        /// What the reference permits.
        access: Access,
        /// The referent's type.
        target: Ty,
    },
    /// `*[T]` or `*const [T]`.
    Slice {
        /// What the slice permits.
        access: Access,
        /// The element type.
        elem: Ty,
    },
    /// `[T; N]`.
    Array {
        /// The element type.
        elem: Ty,
        /// The number of elements.
        len: u64,
    },
    /// `[T]`: a growable array that owns its elements.
    Growable {
        /// The element type.
        elem: Ty,
    },
    /// `qmap<[qubit; k], V>`.
    Qmap {
        /// The key register's width.
        key: u64,
        /// The entry type.
        entry: Ty,
    },
}

/// Every type one unit uses beyond the primitives.
#[derive(Clone, Debug, Default)]
pub struct TypeTable {
    compounds: Vec<Compound>,
    index: HashMap<Compound, CompoundId>,
    /// Tuples are interned by their element list, which is why they are held
    /// apart from the fixed-shape compounds.
    tuples: Vec<Vec<Ty>>,
    tuple_index: HashMap<Vec<Ty>, TupleId>,
    /// Function signatures, the parameter types and the return type, shared
    /// by function types and closure types.
    sigs: Vec<(Vec<Ty>, Ty)>,
    sig_index: HashMap<(Vec<Ty>, Ty), SigId>,
    adts: Vec<AdtDef>,
    /// The structures and enumerations that define `$copy`, by the unit that
    /// declares each (`None` for this one) and its name. Every instance of a
    /// generic one is copyable too.
    copies: Vec<(Option<String>, Symbol)>,
}

impl TypeTable {
    /// An empty table.
    pub fn new() -> TypeTable {
        TypeTable::default()
    }

    fn intern(&mut self, c: Compound) -> CompoundId {
        if let Some(&id) = self.index.get(&c) {
            return id;
        }
        let id = CompoundId(self.compounds.len() as u32);
        self.compounds.push(c);
        self.index.insert(c, id);
        id
    }

    /// A reference type. A reference to a fixed array is still a reference,
    /// `*[T; N]`: though it is carried as an address and a count, like a
    /// slice.
    ///
    /// A reference to a growable array is a slice of its elements: the array
    /// is already an address and a count, and nothing done through a
    /// reference may change its length.
    pub fn reference(&mut self, access: Access, target: Ty) -> Ty {
        if let Some(elem) = self.as_growable(target) {
            return self.slice(access, elem);
        }
        Ty::Ref(self.intern(Compound::Ref { access, target }))
    }

    /// A slice type (i.e. `*[T]`).
    pub fn slice(&mut self, access: Access, elem: Ty) -> Ty {
        Ty::Slice(self.intern(Compound::Slice { access, elem }))
    }

    /// A fixed array type (i.e. `[T; N]`).
    pub fn array(&mut self, elem: Ty, len: u64) -> Ty {
        Ty::Array(self.intern(Compound::Array { elem, len }))
    }

    /// A growable array type (i.e. `[T]`).
    pub fn growable(&mut self, elem: Ty) -> Ty {
        Ty::Growable(self.intern(Compound::Growable { elem }))
    }

    /// A map locale type (i.e. `qmap<[qubit; key], entry>`).
    pub fn qmap(&mut self, key: u64, entry: Ty) -> Ty {
        Ty::Qmap(self.intern(Compound::Qmap { key, entry }))
    }

    /// The key register's width and the entry type, if `ty` is a map locale
    /// type.
    pub fn as_qmap(&self, ty: Ty) -> Option<(u64, Ty)> {
        match ty {
            Ty::Qmap(id) => match self.compound(id) {
                Compound::Qmap { key, entry } => Some((key, entry)),
                _ => None,
            },
            _ => None,
        }
    }

    /// The growable array type of `elem`.
    pub fn find_growable(&self, elem: Ty) -> Option<Ty> {
        self.index.get(&Compound::Growable { elem }).map(|&id| Ty::Growable(id))
    }

    /// The element type.
    pub fn as_growable(&self, ty: Ty) -> Option<Ty> {
        match ty {
            Ty::Growable(id) => match self.compound(id) {
                Compound::Growable { elem } => Some(elem),
                _ => None,
            },
            _ => None,
        }
    }

    /// The type of a string literal: `*[char]`.
    pub fn string(&mut self) -> Ty {
        self.slice(Access::Write, Ty::Char)
    }

    /// A tuple type. With no elements this is `()`, the type of the one value
    /// that carries nothing.
    pub fn tuple(&mut self, elems: Vec<Ty>) -> Ty {
        if let Some(&id) = self.tuple_index.get(&elems) {
            return Ty::Tuple(id);
        }
        let id = TupleId(self.tuples.len() as u32);
        self.tuples.push(elems.clone());
        self.tuple_index.insert(elems, id);
        Ty::Tuple(id)
    }

    /// The element types.
    pub fn as_tuple(&self, ty: Ty) -> Option<&[Ty]> {
        match ty {
            Ty::Tuple(id) => Some(&self.tuples[id.index()]),
            _ => None,
        }
    }

    fn sig(&mut self, params: Vec<Ty>, ret: Ty) -> SigId {
        let key = (params, ret);
        if let Some(&id) = self.sig_index.get(&key) {
            return id;
        }
        let id = SigId(self.sigs.len() as u32);
        self.sigs.push(key.clone());
        self.sig_index.insert(key, id);
        id
    }

    /// A function type (i.e. `fn(T…) -> U`).
    pub fn function(&mut self, params: Vec<Ty>, ret: Ty) -> Ty {
        Ty::Fn(self.sig(params, ret))
    }

    /// A closure type (i.e. `closure<fn(T…) -> U>`).
    pub fn closure(&mut self, params: Vec<Ty>, ret: Ty) -> Ty {
        Ty::Closure(self.sig(params, ret))
    }

    /// A circuit handle's type (i.e. `circuit<fn(T…) -> U>`).
    pub fn circuit(&mut self, params: Vec<Ty>, ret: Ty) -> Ty {
        Ty::Circuit(self.sig(params, ret))
    }

    /// The parameter types and return type, if `ty` is a function, closure or
    /// circuit type.
    pub fn as_sig(&self, ty: Ty) -> Option<(&[Ty], Ty)> {
        match ty {
            Ty::Fn(id) | Ty::Closure(id) | Ty::Circuit(id) => {
                let (p, r) = &self.sigs[id.index()];
                Some((p, *r))
            }
            _ => None,
        }
    }

    /// A compound type's shape.
    pub fn compound(&self, id: CompoundId) -> Compound {
        self.compounds[id.index()]
    }

    /// What a reference permits and what it refers to, if `ty` is a reference.
    pub fn as_ref(&self, ty: Ty) -> Option<(Access, Ty)> {
        match ty {
            Ty::Ref(id) => match self.compound(id) {
                Compound::Ref { access, target } => Some((access, target)),
                _ => None,
            },
            _ => None,
        }
    }

    /// What a slice permits and its element type, if `ty` is a slice.
    pub fn as_slice(&self, ty: Ty) -> Option<(Access, Ty)> {
        match ty {
            Ty::Slice(id) => match self.compound(id) {
                Compound::Slice { access, elem } => Some((access, elem)),
                _ => None,
            },
            _ => None,
        }
    }

    /// The element type and length.
    pub fn as_array(&self, ty: Ty) -> Option<(Ty, u64)> {
        match ty {
            Ty::Array(id) => match self.compound(id) {
                Compound::Array { elem, len } => Some((elem, len)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Adds a structure or enumeration, returning its id.
    pub fn add_adt(&mut self, def: AdtDef) -> AdtId {
        let id = AdtId(self.adts.len() as u32);
        self.adts.push(def);
        id
    }

    /// A structure or enumeration.
    pub fn adt(&self, id: AdtId) -> &AdtDef {
        &self.adts[id.index()]
    }

    /// A structure or enumeration.
    pub fn adt_mut(&mut self, id: AdtId) -> &mut AdtDef {
        &mut self.adts[id.index()]
    }

    /// Every structure and enumeration, indexed by [`AdtId`].
    pub fn adts(&self) -> &[AdtDef] {
        &self.adts
    }

    /// The structure or enumeration declared as `name` by `origin` (`None`
    /// meaning this unit), if the table has it.
    pub fn find_adt(&self, origin: Option<&str>, name: crate::intern::Symbol) -> Option<AdtId> {
        self.find_instance(origin, name, &[])
    }

    /// The instance of the generic structure or enumeration `name` of unit
    /// `origin` made with `args`, if the table has it.
    pub fn find_instance(&self, origin: Option<&str>, name: crate::intern::Symbol, args: &[Arg]) -> Option<AdtId> {
        self.adts
            .iter()
            .position(|a| a.name == name && a.origin.as_deref() == origin && a.args == args)
            .map(|i| AdtId(i as u32))
    }

    /// Whether a value of `ty` is copied, rather than moved, when it is used.
    ///
    /// Integers, `bool`, `char` and every reference are copied: they are
    /// plain values that own nothing. A fixed array is copied when its
    /// elements are. A structure or enumeration is moved (using it as a value
    /// hands it on, and the binding it came from may not be used again),
    /// unless its type defines `$copy`, which makes each copy.
    pub fn is_copyable(&self, ty: Ty) -> bool {
        // no-cloning holds by construction: nothing that holds a qubit is
        // copied, whatever methods its type declares
        if self.is_quantum(ty) {
            return false;
        }
        match ty {
            Ty::Int(_) | Ty::Float(_) | Ty::Bool | Ty::Char | Ty::Void | Ty::Never | Ty::Infer(_) => true,
            Ty::Ref(_) | Ty::Slice(_) => true,
            Ty::Array(_) => self.as_array(ty).is_some_and(|(e, _)| self.is_copyable(e)),
            Ty::Tuple(_) => self
                .as_tuple(ty)
                .is_some_and(|es| es.iter().all(|&e| self.is_copyable(e))),
            // a function is its code address, and a circuit handle its
            // operator's name. a closure owns its environment, and its type
            // does not say what that holds
            Ty::Fn(_) | Ty::Circuit(_) | Ty::Qmap(_) => true,
            Ty::Adt(id) => self.defines_copy(id),
            Ty::Closure(_) | Ty::Growable(_) | Ty::Dyn | Ty::Qubit => false,
        }
    }

    /// The name of the `core::TypeKind` variant that `ty` is.
    pub fn kind_name(&self, ty: Ty) -> &'static str {
        match ty {
            Ty::Int(_) | Ty::Float(_) | Ty::Bool | Ty::Char | Ty::Infer(_) => "Scalar",
            Ty::Void | Ty::Never => "Void",
            Ty::Adt(id) if self.adt(id).is_struct() => "Struct",
            Ty::Adt(_) => "Enum",
            Ty::Tuple(_) => "Tuple",
            Ty::Array(_) => "Array",
            Ty::Growable(_) => "DynArray",
            Ty::Ref(_) => "Ref",
            Ty::Slice(_) => "Slice",
            Ty::Fn(_) => "Fn",
            Ty::Closure(_) => "Closure",
            Ty::Circuit(_) => "Circuit",
            Ty::Qmap(_) => "Map",
            Ty::Qubit => "Qubit",
            Ty::Dyn => "Dyn",
        }
    }

    /// Whether a value of `ty` holds a qubit: `qubit` itself, or an array,
    /// tuple, structure or enumeration with one somewhere inside. A
    /// reference holds none: `*qubit` names a qubit without owning it.
    pub fn is_quantum(&self, ty: Ty) -> bool {
        self.holds_any(ty, &|t| t == Ty::Qubit)
    }

    /// Whether `ty`, or something a value of it holds (an element, a field,
    /// a variant's field, however deep) is of a type `leaf` accepts.
    pub fn holds_any(&self, ty: Ty, leaf: &dyn Fn(Ty) -> bool) -> bool {
        self.holds_any_in(ty, leaf, &mut Vec::new())
    }

    /// How many qubits a value of the quantum type `ty` holds: one for a
    /// `qubit`; an array, tuple or structure the sum of its parts, a
    /// classical field of a structure holding none; and a quantum
    /// enumeration a tag register numbering its variants and a payload
    /// register as wide as its widest variant, whose classical fields are
    /// held as basis states ([`TypeTable::basis_bits`]).
    pub fn qubits(&self, ty: Ty) -> u64 {
        match ty {
            Ty::Qubit => 1,
            Ty::Array(_) => self.as_array(ty).map_or(0, |(elem, n)| n * self.qubits(elem)),
            Ty::Tuple(id) => self.tuples[id.index()].iter().map(|&e| self.qubits(e)).sum(),
            Ty::Adt(id) if self.adt(id).is_struct() => self.adt(id).fields().iter().map(|f| self.qubits(f.ty)).sum(),
            Ty::Adt(id) => {
                let (tag, payload) = self.enum_qubits(id);
                tag + payload
            }
            _ => 0,
        }
    }

    /// The widths of a quantum enumeration's tag and payload registers.
    pub fn enum_qubits(&self, id: AdtId) -> (u64, u64) {
        let variants = self.adt(id).variants();
        let k = variants.len() as u64;
        let tag = if k <= 1 { 0 } else { 64 - u64::from((k - 1).leading_zeros()) };
        let payload = variants
            .iter()
            .map(|v| v.fields.iter().map(|f| self.payload_qubits(f.ty)).sum::<u64>())
            .max()
            .unwrap_or(0);
        (tag, payload)
    }

    /// How many qubits a field of type `ty` takes in a quantum
    /// enumeration's payload: its own, or its bits for a classical one.
    pub fn payload_qubits(&self, ty: Ty) -> u64 {
        if self.is_quantum(ty) { self.qubits(ty) } else { self.basis_bits(ty).unwrap_or(0) }
    }

    /// How many qubits a classical value of type `ty` takes held as a basis
    /// state, one per bit, as a quantum enumeration's variant holds its
    /// classical fields: an integer its width, a `bool` one, and an array,
    /// tuple or structure the sum of its parts. `None` for a type with no
    /// such encoding.
    pub fn basis_bits(&self, ty: Ty) -> Option<u64> {
        match ty {
            Ty::Int(i) => Some(u64::from(i.bits)),
            Ty::Bool => Some(1),
            Ty::Array(_) => {
                let (elem, n) = self.as_array(ty)?;
                Some(n * self.basis_bits(elem)?)
            }
            Ty::Tuple(id) => self.tuples[id.index()].iter().map(|&e| self.basis_bits(e)).sum(),
            Ty::Adt(id) if self.adt(id).is_struct() && !self.is_quantum(ty) => {
                self.adt(id).fields().iter().map(|f| self.basis_bits(f.ty)).sum()
            }
            _ => None,
        }
    }

    fn holds_any_in(&self, ty: Ty, leaf: &dyn Fn(Ty) -> bool, seen: &mut Vec<AdtId>) -> bool {
        if leaf(ty) {
            return true;
        }
        match ty {
            Ty::Array(_) | Ty::Growable(_) => match self.compound(Self::compound_id(ty)) {
                Compound::Array { elem, .. } | Compound::Growable { elem } => self.holds_any_in(elem, leaf, seen),
                _ => false,
            },
            Ty::Tuple(id) => self.tuples[id.index()].iter().any(|&e| self.holds_any_in(e, leaf, seen)),
            Ty::Adt(id) => {
                // a type that contains itself does so through something that
                // owns it indirectly; that path is followed once
                if seen.contains(&id) {
                    return false;
                }
                seen.push(id);
                let found = self.adt(id).field_types().any(|t| self.holds_any_in(t, leaf, seen));
                seen.pop();
                found
            }
            _ => false,
        }
    }

    fn compound_id(ty: Ty) -> CompoundId {
        match ty {
            Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) | Ty::Qmap(id) => id,
            _ => unreachable!("only references, slices and arrays are compound"),
        }
    }

    /// Records that the type `name` declared by `origin` defines `$copy`.
    pub fn mark_copyable(&mut self, origin: Option<String>, name: Symbol) {
        if !self.copies.iter().any(|(o, n)| *n == name && *o == origin) {
            self.copies.push((origin, name));
        }
    }

    /// Whether a structure or enumeration defines `$copy`.
    pub fn defines_copy(&self, id: AdtId) -> bool {
        let a = self.adt(id);
        self.copies.iter().any(|(o, n)| *n == a.name && *o == a.origin)
    }

    /// Rebuilds `ty` with `leaf` applied to every part that is not itself a
    /// compound type, re-interning the compounds on the way back up. Used to
    /// replace settled inference variables inside `[_; N]` and the like.
    pub fn map_leaves(&mut self, ty: Ty, leaf: &mut impl FnMut(Ty) -> Ty) -> Ty {
        match ty {
            Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) | Ty::Qmap(id) => match self.compound(id) {
                Compound::Ref { access, target } => {
                    let t = self.map_leaves(target, leaf);
                    self.reference(access, t)
                }
                Compound::Slice { access, elem } => {
                    let e = self.map_leaves(elem, leaf);
                    self.slice(access, e)
                }
                Compound::Array { elem, len } => {
                    let e = self.map_leaves(elem, leaf);
                    self.array(e, len)
                }
                Compound::Growable { elem } => {
                    let e = self.map_leaves(elem, leaf);
                    self.growable(e)
                }
                Compound::Qmap { key, entry } => {
                    let e = self.map_leaves(entry, leaf);
                    self.qmap(key, e)
                }
            },
            Ty::Tuple(id) => {
                let elems = self.tuples[id.index()].clone();
                let mapped: Vec<Ty> = elems.into_iter().map(|e| self.map_leaves(e, leaf)).collect();
                self.tuple(mapped)
            }
            Ty::Fn(id) | Ty::Closure(id) | Ty::Circuit(id) => {
                let (params, ret) = self.sigs[id.index()].clone();
                let params: Vec<Ty> = params.into_iter().map(|p| self.map_leaves(p, leaf)).collect();
                let ret = self.map_leaves(ret, leaf);
                let sig = self.sig(params, ret);
                match ty {
                    Ty::Fn(_) => Ty::Fn(sig),
                    Ty::Closure(_) => Ty::Closure(sig),
                    _ => Ty::Circuit(sig),
                }
            }
            other => leaf(other),
        }
    }

    /// Whether `ty` mentions an undecided integer anywhere.
    pub fn has_infer(&self, ty: Ty) -> bool {
        match ty {
            Ty::Infer(_) => true,
            Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) | Ty::Qmap(id) => match self.compound(id) {
                Compound::Ref { target: t, .. }
                | Compound::Slice { elem: t, .. }
                | Compound::Array { elem: t, .. }
                | Compound::Growable { elem: t }
                | Compound::Qmap { entry: t, .. } => self.has_infer(t),
            },
            Ty::Tuple(id) => self.tuples[id.index()].iter().any(|&e| self.has_infer(e)),
            Ty::Fn(id) | Ty::Closure(id) | Ty::Circuit(id) => {
                let (params, ret) = &self.sigs[id.index()];
                params.iter().any(|&p| self.has_infer(p)) || self.has_infer(*ret)
            }
            _ => false,
        }
    }

    /// A type as a program writes it: `u8`, `*[Point; 4]`, `geometry::Shape`.
    pub fn display(&self, ty: Ty, interner: &Interner) -> String {
        self.show(ty, interner, None)
    }

    /// A type with every structure and enumeration qualified by the unit
    /// that declared it, the table's own ones by `home`. The same type gets
    /// the same text in every unit's table, which is what a symbol naming it
    /// needs.
    pub fn qualified(&self, ty: Ty, interner: &Interner, home: &str) -> String {
        self.show(ty, interner, Some(home))
    }

    /// The function type of these parameters and result, written as
    /// [`TypeTable::qualified`] writes it, whether or not the table holds it.
    pub fn qualified_signature(&self, params: &[Ty], ret: Ty, interner: &Interner, home: &str) -> String {
        self.signature_shown(params, ret, interner, Some(home))
    }

    fn signature_shown(&self, params: &[Ty], ret: Ty, interner: &Interner, home: Option<&str>) -> String {
        let params: Vec<String> = params.iter().map(|&p| self.show(p, interner, home)).collect();
        let ret = match ret {
            Ty::Void => String::new(),
            r => format!(" -> {}", self.show(r, interner, home)),
        };
        format!("fn({}){ret}", params.join(", "))
    }

    fn show(&self, ty: Ty, interner: &Interner, home: Option<&str>) -> String {
        match ty {
            Ty::Int(t) => t.name().to_owned(),
            Ty::Float(t) => t.name().to_owned(),
            Ty::Bool => "bool".to_owned(),
            Ty::Char => "char".to_owned(),
            Ty::Dyn => "dyn".to_owned(),
            Ty::Qubit => "qubit".to_owned(),
            Ty::Void => "void".to_owned(),
            Ty::Never => "never".to_owned(),
            Ty::Infer(_) => "integer".to_owned(),
            Ty::Adt(id) => self.name_of(id, interner, home),
            Ty::Tuple(id) => {
                let parts: Vec<String> = self.tuples[id.index()]
                    .iter()
                    .map(|&e| self.show(e, interner, home))
                    .collect();
                // `(T)` would be the parenthesised type `T`, so a one-element
                // tuple keeps a trailing comma, as it is written
                let tail = if parts.len() == 1 { "," } else { "" };
                format!("({}{tail})", parts.join(", "))
            }
            Ty::Fn(id) | Ty::Closure(id) | Ty::Circuit(id) => {
                let (params, ret) = &self.sigs[id.index()];
                let sig = self.signature_shown(params, *ret, interner, home);
                match ty {
                    Ty::Closure(_) => format!("closure<{sig}>"),
                    Ty::Circuit(_) => format!("circuit<{sig}>"),
                    _ => sig,
                }
            }
            Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) | Ty::Qmap(id) => match self.compound(id) {
                Compound::Ref { access, target } => {
                    format!("{}{}", access.prefix(), self.show(target, interner, home))
                }
                Compound::Slice { access, elem } => {
                    format!("{}[{}]", access.prefix(), self.show(elem, interner, home))
                }
                Compound::Array { elem, len } => format!("[{}; {len}]", self.show(elem, interner, home)),
                Compound::Growable { elem } => format!("[{}]", self.show(elem, interner, home)),
                Compound::Qmap { key, entry } => format!("qmap<[qubit; {key}], {}>", self.show(entry, interner, home)),
            },
        }
    }

    /// A structure's or enumeration's name as a program writes it, with its
    /// unit when another unit declared it and its arguments when it is an
    /// instance: `Point`, `geometry::Shape`, `Pair<i32, bool>`.
    pub fn adt_name(&self, id: AdtId, interner: &Interner) -> String {
        self.name_of(id, interner, None)
    }

    fn name_of(&self, id: AdtId, interner: &Interner, home: Option<&str>) -> String {
        let a = self.adt(id);
        let base = match a.origin.as_deref().or(home) {
            Some(u) => format!("{u}::{}", interner.resolve(a.name)),
            None => interner.resolve(a.name).to_owned(),
        };
        if a.args.is_empty() {
            return base;
        }
        let args: Vec<String> = a
            .args
            .iter()
            .map(|&x| match x {
                Arg::Type(t) => self.show(t, interner, home),
                Arg::Const(v) => v.to_string(),
            })
            .collect();
        format!("{base}<{}>", args.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Linkage;
    use crate::span::Span;
    use crate::tir::adt::{AdtKind, Repr};
    use crate::tir::ty::{InferId, IntTy};

    fn point(i: &mut Interner) -> AdtDef {
        AdtDef {
            args: Vec::new(),
            name: i.intern("Point"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Struct { fields: vec![] },
            repr: Repr::default(),
            span: Span::synthetic(),
        }
    }

    #[test]
    fn a_compound_written_twice_is_one_type() {
        let mut t = TypeTable::new();
        let a = t.array(Ty::Int(IntTy::U8), 4);
        let b = t.array(Ty::Int(IntTy::U8), 4);
        let c = t.array(Ty::Int(IntTy::U8), 5);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(t.as_array(a), Some((Ty::Int(IntTy::U8), 4)));
    }

    #[test]
    fn references_differ_by_what_they_permit() {
        let mut t = TypeTable::new();
        let writes = t.reference(Access::Write, Ty::Bool);
        let reads = t.reference(Access::Const, Ty::Bool);
        assert_ne!(writes, reads);
        assert_eq!(t.as_ref(writes), Some((Access::Write, Ty::Bool)));
        assert_eq!(t.as_slice(writes), None);
    }

    #[test]
    fn types_display_as_written() {
        let mut i = Interner::new();
        let mut t = TypeTable::new();
        let p = t.add_adt(point(&mut i));
        let arr = t.array(Ty::Adt(p), 4);
        let r = t.reference(Access::Write, arr);
        assert_eq!(t.display(r, &i), "*[Point; 4]");
        let s = t.string();
        assert_eq!(t.display(s, &i), "*[char]");
        let c = t.reference(Access::Const, Ty::Int(IntTy::U32));
        assert_eq!(t.display(c, &i), "*const u32");
        let mut imported = point(&mut i);
        imported.origin = Some("geometry".to_owned());
        let g = t.add_adt(imported);
        assert_eq!(t.display(Ty::Adt(g), &i), "geometry::Point");
    }

    #[test]
    fn a_type_is_identified_by_its_origin_and_name() {
        let mut i = Interner::new();
        let mut t = TypeTable::new();
        let local = t.add_adt(point(&mut i));
        let mut other = point(&mut i);
        other.origin = Some("geometry".to_owned());
        let imported = t.add_adt(other);
        let name = i.intern("Point");
        assert_eq!(t.find_adt(None, name), Some(local));
        assert_eq!(t.find_adt(Some("geometry"), name), Some(imported));
        assert_eq!(t.find_adt(Some("elsewhere"), name), None);
    }

    #[test]
    fn plain_values_copy_and_declared_types_move() {
        let mut i = Interner::new();
        let mut t = TypeTable::new();
        let p = t.add_adt(point(&mut i));
        assert!(t.is_copyable(Ty::Int(IntTy::I32)));
        assert!(t.is_copyable(Ty::Char));
        let r = t.reference(Access::Write, Ty::Adt(p));
        assert!(t.is_copyable(r), "a reference owns nothing");
        let ints = t.array(Ty::Bool, 3);
        assert!(t.is_copyable(ints));
        let points = t.array(Ty::Adt(p), 3);
        assert!(!t.is_copyable(points));
        assert!(!t.is_copyable(Ty::Adt(p)));
    }

    #[test]
    fn leaves_can_be_replaced_inside_compounds() {
        let mut t = TypeTable::new();
        let var = Ty::Infer(InferId(0));
        let arr = t.array(var, 2);
        let r = t.reference(Access::Write, arr);
        assert!(t.has_infer(r));
        let settled = t.map_leaves(r, &mut |l| match l {
            Ty::Infer(_) => Ty::Int(IntTy::I64),
            other => other,
        });
        assert!(!t.has_infer(settled));
        let (_, target) = t.as_ref(settled).unwrap();
        assert_eq!(t.as_array(target), Some((Ty::Int(IntTy::I64), 2)));
    }
}
