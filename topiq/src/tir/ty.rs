//! Types as analysis understands them.
//!
//! The syntax tree spells a type however the program wrote it: `u32`, `const
//! u32`, `*[Point; 4]`. Here a type is what the compiler has concluded
//! about a value. Three things differ from the written form.
//!
//! **`const` at the outside is not a type here.** Whether a value is known
//! during translation is a property of the binding or expression that holds
//! it, tracked by analysis and erased once constants are folded. By the time a
//! [`Ty`] reaches code generation, a `const u32` and a `u32` are the same thing.
//! The one place `const` survives is behind a reference: `*const T` points at
//! an object whose value is fixed, which is a different promise from `*T`, so
//! it is recorded as the reference's [`Access`].
//!
//! **A type is a small value.** Structures and enumerations are named by an
//! [`AdtId`], and the compound forms (references, slices and fixed arrays)
//! by a [`CompoundId`] into the unit's [`TypeTable`](super::TypeTable), which
//! stores each distinct compound once. So a [`Ty`] is `Copy`, and two types
//! are the same exactly when they compare equal.
//!
//! **An unsuffixed integer literal starts out undecided.** `let x: u8 = 5;`
//! makes the `5` a `u8`, and `let y = 5;` makes it an `i64`. While a function
//! body is being checked such a literal has type [`Ty::Infer`], and the
//! checker settles every one of them before the body is finished. None
//! survives into a completed [`crate::tir::Unit`].

use std::fmt;

/// The width of `usize` and `isize` on the only target code is generated for,
/// x86-64.
pub const POINTER_BITS: u8 = 64;

/// An integer type: its signedness and width.
///
/// `usize` and `isize` are distinct types from `u64` and `i64` even though
/// they share a width here. Converting between them is still written with
/// `as`, so that a program keeps meaning the same thing on a target where the
/// widths differ.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IntTy {
    /// Whether values may be negative.
    pub signed: bool,
    /// The width in bits: 8, 16, 32 or 64.
    pub bits: u8,
    /// Whether this is `usize` or `isize`.
    pub pointer_sized: bool,
}

impl IntTy {
    /// `i8`.
    pub const I8: IntTy = IntTy::fixed(true, 8);
    /// `i16`.
    pub const I16: IntTy = IntTy::fixed(true, 16);
    /// `i32`.
    pub const I32: IntTy = IntTy::fixed(true, 32);
    /// `i64`, which is also the type of an integer literal nothing constrains.
    pub const I64: IntTy = IntTy::fixed(true, 64);
    /// `isize`.
    pub const ISIZE: IntTy = IntTy {
        signed: true,
        bits: POINTER_BITS,
        pointer_sized: true,
    };
    /// `u8`.
    pub const U8: IntTy = IntTy::fixed(false, 8);
    /// `u16`.
    pub const U16: IntTy = IntTy::fixed(false, 16);
    /// `u32`.
    pub const U32: IntTy = IntTy::fixed(false, 32);
    /// `u64`.
    pub const U64: IntTy = IntTy::fixed(false, 64);
    /// `usize`.
    pub const USIZE: IntTy = IntTy {
        signed: false,
        bits: POINTER_BITS,
        pointer_sized: true,
    };

    /// Every integer type, signed first, narrowest first.
    pub const ALL: [IntTy; 10] = [
        IntTy::I8,
        IntTy::I16,
        IntTy::I32,
        IntTy::I64,
        IntTy::ISIZE,
        IntTy::U8,
        IntTy::U16,
        IntTy::U32,
        IntTy::U64,
        IntTy::USIZE,
    ];

    const fn fixed(signed: bool, bits: u8) -> IntTy {
        IntTy {
            signed,
            bits,
            pointer_sized: false,
        }
    }

    /// The type's name as a program writes it.
    pub fn name(self) -> &'static str {
        match (self.signed, self.bits, self.pointer_sized) {
            (true, _, true) => "isize",
            (false, _, true) => "usize",
            (true, 8, _) => "i8",
            (true, 16, _) => "i16",
            (true, 32, _) => "i32",
            (true, _, _) => "i64",
            (false, 8, _) => "u8",
            (false, 16, _) => "u16",
            (false, 32, _) => "u32",
            (false, _, _) => "u64",
        }
    }

    /// Looks a type up by the name a program writes.
    pub fn from_name(name: &str) -> Option<IntTy> {
        IntTy::ALL.into_iter().find(|t| t.name() == name)
    }

    /// The smallest value of the type.
    pub fn min(self) -> i128 {
        if self.signed {
            -(1i128 << (self.bits - 1))
        } else {
            0
        }
    }

    /// The largest value of the type.
    pub fn max(self) -> i128 {
        if self.signed {
            (1i128 << (self.bits - 1)) - 1
        } else {
            (1i128 << self.bits) - 1
        }
    }

    /// Whether `value` is a value of this type.
    pub fn fits(self, value: i128) -> bool {
        (self.min()..=self.max()).contains(&value)
    }
}

impl fmt::Display for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A floating type (i.e. `f32` or `f64`).
///
/// Floating arithmetic never aborts: it answers with an infinity or a NaN
/// where an integer operation would stop the program. That is why the two
/// families are kept apart here rather than under one "numeric" type.
///
/// [IEEE 754]: https://en.wikipedia.org/wiki/IEEE_754
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatTy {
    /// `f32`: binary32, 24 significant bits.
    F32,
    /// `f64`: binary64, 53 significant bits, and the type of a float literal
    /// nothing constrains.
    F64,
}

impl FloatTy {
    /// Both floating types, narrowest first.
    pub const ALL: [FloatTy; 2] = [FloatTy::F32, FloatTy::F64];

    /// The type's name as a program writes it.
    pub fn name(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }

    /// Looks a type up by the name a program writes.
    pub fn from_name(name: &str) -> Option<FloatTy> {
        FloatTy::ALL.into_iter().find(|t| t.name() == name)
    }

    /// The width in bits.
    pub fn bits(self) -> u8 {
        match self {
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
        }
    }

    /// `x` rounded to this type's precision, as the nearest value it can
    /// hold. Every value of an `f32` is a value of an `f64`, so one `f64`
    /// carries either.
    pub fn round(self, x: f64) -> f64 {
        match self {
            FloatTy::F32 => f64::from(x as f32),
            FloatTy::F64 => x,
        }
    }

    /// A value of this type as a program would write it (the shortest text
    /// that reads back as the same value), or a name for the three values no
    /// literal spells: `inf`, `-inf` and `nan`.
    pub fn text(self, x: f64) -> String {
        if x.is_nan() {
            return "nan".to_owned();
        }
        if x.is_infinite() {
            return if x < 0.0 { "-inf" } else { "inf" }.to_owned();
        }
        let mut s = match self {
            FloatTy::F32 => format!("{}", x as f32),
            FloatTy::F64 => format!("{x}"),
        };
        if !s.contains(['.', 'e', 'E']) {
            s.push_str(".0");
        }
        s
    }
}

impl fmt::Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A literal whose type the checker has not yet settled: an unsuffixed integer
/// or floating literal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InferId(pub u32);

/// A structure or enumeration.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AdtId(pub u32);

impl AdtId {
    /// The index as a `usize`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One generic argument an instance was made with.
///
/// A generic item is not itself a type or a function: `Pair<A, B>` and
/// `sum<T>` become one of each per set of arguments, and the arguments travel
/// with the instance so that two instances of one item are told apart, named
/// in diagnostics, and given distinct symbols.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Arg {
    /// A type argument.
    Type(Ty),
    /// A `const` argument: a number known during translation, such as the `4`
    /// of `Reg<4>`.
    Const(i128),
}

/// A tuple's element list.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TupleId(pub u32);

impl TupleId {
    /// The index as a `usize`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A function signature.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SigId(pub u32);

impl SigId {
    /// The index as a `usize`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A reference, slice or fixed array.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CompoundId(pub u32);

impl CompoundId {
    /// The index as a `usize`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// What a reference permits.
///
/// `*T` may read and write what it refers to; `*const T` refers to a
/// constant, and may only read it. A `*T` converts implicitly to a
/// `*const T`, which promises less, but never the other way, because that
/// would grant a right nothing checked.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Access {
    /// `*T`: reads and writes the referent.
    Write,
    /// `*const T`: reads a constant referent.
    Const,
}

impl Access {
    /// How the reference type is written, up to its referent: `*` or
    /// `*const `.
    pub fn prefix(self) -> &'static str {
        match self {
            Access::Write => "*",
            Access::Const => "*const ",
        }
    }

    /// Whether a reference of this kind converts implicitly to one of `to`.
    pub fn converts_to(self, to: Access) -> bool {
        self == to || self == Access::Write
    }
}

/// A type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    /// One of the integer types.
    Int(IntTy),
    /// `f32` or `f64`.
    Float(FloatTy),
    /// `bool`.
    Bool,
    /// `char`: one Unicode scalar value, stored in four bytes.
    Char,
    /// `void`: the result of something that produces no value. There are no
    /// values of this type, so nothing can be bound to one.
    Void,
    /// The type of an expression that never finishes evaluating (a `return`,
    /// a `break`, a `loop` with no way out). It is accepted wherever any type is
    /// expected, because control never arrives there with a value.
    Never,
    /// A literal whose exact type is not settled yet. Only present while a
    /// function body is being checked.
    Infer(InferId),
    /// A structure or enumeration.
    Adt(AdtId),
    /// `*T` or `*const T`, where `T` is anything but an unsized
    /// array.
    Ref(CompoundId),
    /// `*[T]` or `*const [T]`: a view of some number of `T`s laid
    /// out one after another, carried as their address and their count.
    Slice(CompoundId),
    /// `[T; N]`: exactly `N` values of `T`, stored inline.
    Array(CompoundId),
    /// `(T1, T2, …)`: values of several types stored inline, one after
    /// another, reached by matching rather than by name. The empty tuple `()`
    /// is a type with exactly one value and no bytes, unlike `void`, which
    /// has no values at all.
    Tuple(TupleId),
    /// `fn(T…) -> U`: a function, held as its code's address. A function
    /// declared with `fn` is a value of this type, and `&f` a reference to
    /// it.
    Fn(SigId),
    /// `closure<fn(T…) -> U>`: a function together with the environment it
    /// captured, which it owns. Held as the environment's address and the
    /// code's.
    Closure(SigId),
    /// `[T]`: a growable array that owns its elements, held as the address
    /// of the first and their count. Its storage begins with two words before
    /// the first element: the capacity, then the element type's run-time
    /// type information.
    Growable(CompoundId),
    /// `circuit<fn(T…) -> U>`: a quantum unit's entry operator, as a
    /// classical program holds it. It can be stored and passed about but not
    /// called here: running it submits it to a quantum processor. It is held
    /// as the operator's name, two words like a slice of characters.
    Circuit(SigId),
    /// `qmap<[qubit; k], V>`: a map locale of a quantum unit, as a value. It
    /// exists only while circuits are generated, naming the table, and
    /// occupies no storage.
    Qmap(CompoundId),
    /// `qubit`: one qubit, a linear resource that is moved and never copied.
    /// It exists only in quantum units, where it names a wire of the circuit
    /// being generated; it occupies no classical storage.
    Qubit,
    /// `dyn`: a value of any type, owned, with its type known from the table
    /// it carries. Held as the address of the value, in a buffer of its own,
    /// and the table.
    Dyn,
}

impl Ty {
    /// `usize`, the type of an index and of a length.
    pub const USIZE: Ty = Ty::Int(IntTy::USIZE);

    /// Whether this is an integer type, settled or not.
    ///
    /// An unsettled literal may still turn out to be floating, so this is
    /// "not known to be anything else" rather than a promise.
    pub fn is_integer(self) -> bool {
        matches!(self, Ty::Int(_) | Ty::Infer(_))
    }

    /// Whether this is a floating type.
    pub fn is_float(self) -> bool {
        matches!(self, Ty::Float(_))
    }

    /// The floating type.
    pub fn as_float(self) -> Option<FloatTy> {
        match self {
            Ty::Float(t) => Some(t),
            _ => None,
        }
    }

    /// The integer type.
    pub fn as_int(self) -> Option<IntTy> {
        match self {
            Ty::Int(t) => Some(t),
            _ => None,
        }
    }

    /// Whether a value of this type can be stored in a binding.
    ///
    /// `void` has no values and `never` is never reached, so neither can.
    pub fn is_storable(self) -> bool {
        !matches!(self, Ty::Void | Ty::Never)
    }

    /// Whether values of this type live in memory rather than in registers:
    /// structures, enumerations, fixed arrays and tuples. Code generation
    /// builds such a value in place and copies it by its bytes.
    pub fn is_aggregate(self) -> bool {
        matches!(self, Ty::Adt(_) | Ty::Array(_) | Ty::Tuple(_))
    }

    /// Whether this is a reference of any kind: `*T`, `*[T; N]` or a slice.
    pub fn is_reference(self) -> bool {
        matches!(self, Ty::Ref(_) | Ty::Slice(_))
    }

    /// Whether this is a slice, `*[T]`, whose second word is a count.
    pub fn is_slice(self) -> bool {
        matches!(self, Ty::Slice(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_those_of_twos_complement() {
        assert_eq!(IntTy::I8.min(), -128);
        assert_eq!(IntTy::I8.max(), 127);
        assert_eq!(IntTy::U8.min(), 0);
        assert_eq!(IntTy::U8.max(), 255);
        assert_eq!(IntTy::I64.min(), i128::from(i64::MIN));
        assert_eq!(IntTy::I64.max(), i128::from(i64::MAX));
        assert_eq!(IntTy::U64.max(), i128::from(u64::MAX));
    }

    #[test]
    fn fits_respects_both_ends() {
        assert!(IntTy::I8.fits(-128));
        assert!(!IntTy::I8.fits(-129));
        assert!(IntTy::U16.fits(65535));
        assert!(!IntTy::U16.fits(65536));
        assert!(!IntTy::U32.fits(-1));
    }

    #[test]
    fn every_name_round_trips() {
        for t in IntTy::ALL {
            assert_eq!(IntTy::from_name(t.name()), Some(t), "{t}");
        }
        assert_eq!(IntTy::from_name("i128"), None);
        assert_eq!(IntTy::from_name("bool"), None);
    }

    #[test]
    fn pointer_sized_types_are_distinct_from_their_fixed_twins() {
        assert_ne!(IntTy::USIZE, IntTy::U64);
        assert_ne!(IntTy::ISIZE, IntTy::I64);
        assert_eq!(IntTy::USIZE.bits, IntTy::U64.bits);
        assert_eq!(IntTy::USIZE.max(), IntTy::U64.max());
    }

    #[test]
    fn only_values_can_be_stored() {
        assert!(Ty::Int(IntTy::I32).is_storable());
        assert!(Ty::Bool.is_storable());
        assert!(!Ty::Void.is_storable());
        assert!(!Ty::Never.is_storable());
    }

    #[test]
    fn undecided_integers_count_as_integers() {
        assert!(Ty::Infer(InferId(0)).is_integer());
        assert!(Ty::Int(IntTy::U8).is_integer());
        assert!(!Ty::Bool.is_integer());
        assert_eq!(Ty::Infer(InferId(3)).as_int(), None);
        assert_eq!(Ty::Int(IntTy::U8).as_int(), Some(IntTy::U8));
    }

    #[test]
    fn a_reference_converts_only_to_one_promising_less() {
        assert!(Access::Write.converts_to(Access::Write));
        assert!(Access::Write.converts_to(Access::Const));
        assert!(Access::Const.converts_to(Access::Const));
        assert!(!Access::Const.converts_to(Access::Write));
        assert_eq!(Access::Write.prefix(), "*");
        assert_eq!(Access::Const.prefix(), "*const ");
    }

    #[test]
    fn only_structures_enumerations_and_arrays_live_in_memory() {
        assert!(Ty::Adt(AdtId(0)).is_aggregate());
        assert!(Ty::Array(CompoundId(0)).is_aggregate());
        assert!(!Ty::Ref(CompoundId(0)).is_aggregate());
        assert!(!Ty::Slice(CompoundId(0)).is_aggregate());
        assert!(!Ty::Char.is_aggregate());
    }
}
