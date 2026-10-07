//! The typed intermediate representation: what analysis hands to code
//! generation.
//!
//! The syntax tree records what a program *says*. This records what it
//! *means*, in a form code generation can consume without asking any further
//! questions:
//!
//! - **Every name is resolved.** A variable is a [`LocalId`] or a [`GlobalId`],
//!   a call names a [`Callee`], a field is an index, and a `break` names the
//!   [`LoopId`] it leaves. Scopes and shadowing no longer exist at this level.
//! - **Every expression is typed**, and every type is settled.
//! - **Every constant is folded.** A reference to a `const` binding, and every
//!   call of a constant function, has already been replaced by its value. A
//!   constant has no run-time representation, so none of them reach code
//!   generation.
//! - **What the program left implicit is written out.** Compound assignment
//!   arrives as `x = x + 1`; reading a field through a reference arrives as a
//!   [`ExprKind::Deref`] followed by a [`ExprKind::Field`]; a reference that
//!   converts to a weaker one is wrapped in [`ExprKind::Coerce`].
//!
//! Evaluation order is the order of the fields: operands left to right, a call's
//! arguments left to right, a structure literal's fields in the order they were
//! *written*, and an assignment's value before its place. Code generation and
//! the constant evaluator both rely on that, so neither has to know the rule
//! separately.
//!
//! # Places
//!
//! A *place* is an expression that denotes storage rather than a value: a
//! binding, a unit-scope object, a field of a place, an element of an array or
//! slice, or the referent of a reference. Places are what can be assigned to,
//! have a reference taken, or be read in place (`p.x` reads one field of `p`
//! without copying the rest). [`Expr::is_place`] says which expressions are.
//!
//! [`print`](mod@print) dumps a unit in a stable, readable form for `tqc emit --stage=tir`
//! and for snapshot tests.

pub mod adt;
pub mod claims;
pub mod layout;
pub mod print;
pub mod table;
pub mod ty;
pub mod visit;

pub use adt::{AdtDef, AdtKind, FieldDef, Repr, VariantDef, VariantShape};
pub use table::{Compound, TypeTable};
pub use ty::{Access, AdtId, Arg, CompoundId, FloatTy, InferId, IntTy, POINTER_BITS, SigId, TupleId, Ty};

use crate::ast::Linkage;
use crate::intern::Symbol;
use crate::span::{SourceId, Span};

/// Declares a small `Copy` index type.
macro_rules! id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(pub u32);

        impl $name {
            /// The index as a `usize`.
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id!(
    /// A function of the unit.
    FnId
);
id!(
    /// A unit-duration or persistent object.
    GlobalId
);
id!(
    /// A parameter or local binding.
    LocalId
);
id!(
    /// A loop within one function, so `break` and `continue` can name theirs.
    LoopId
);
id!(
    /// A function of another unit that this one calls, indexing
    /// [`Unit::externs`].
    ExternId
);

/// A value known during translation.
///
/// Integers of every type fit in an `i128`, which keeps constant evaluation
/// free of per-width code paths: a value is checked against its type's range
/// instead. An aggregate's fields are held in declaration order; which type
/// it is travels alongside, in the expression or object that holds it.
// `Eq` and `Hash` are deliberately absent: a floating value compares by
// what IEEE 754 says, under which a NaN equals nothing, not even itself
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    /// An integer and its type.
    Int(i128, IntTy),
    /// A floating value and its type. An `f32` is held as the `f64` of the
    /// same value, which is exact: every `f32` is an `f64`.
    Float(f64, FloatTy),
    /// A boolean.
    Bool(bool),
    /// A character.
    Char(char),
    /// The result of something that produces no value.
    Void,
    /// A string literal's contents: a constant `*[char]` whose characters are
    /// placed in the program image.
    Str(Symbol),
    /// A structure's fields.
    Struct(Vec<Value>),
    /// One variant of an enumeration, with its fields in declaration order.
    Enum {
        /// The variant's index.
        variant: u32,
        /// Its fields.
        fields: Vec<Value>,
    },
    /// A fixed array's elements.
    Array(Vec<Value>),
}

impl Value {
    /// The integer.
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Value::Int(v, _) => Some(*v),
            _ => None,
        }
    }

    /// The boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The type of a scalar value, which the value itself determines. An
    /// aggregate or a string's type needs the unit's type table, so has none
    /// here.
    pub fn scalar_ty(&self) -> Option<Ty> {
        match self {
            Value::Int(_, t) => Some(Ty::Int(*t)),
            Value::Bool(_) => Some(Ty::Bool),
            Value::Char(_) => Some(Ty::Char),
            Value::Void => Some(Ty::Void),
            _ => None,
        }
    }
}

/// One translation unit, analysed.
#[derive(Clone, Debug)]
pub struct Unit {
    /// The unit's name: its file stem.
    pub name: String,
    /// The file it came from.
    pub source: SourceId,
    /// Its structures and enumerations, those it imports, and every compound
    /// type it uses.
    pub types: TypeTable,
    /// Every function, indexed by [`FnId`].
    pub fns: Vec<Fn>,
    /// Every unit-duration and persistent object, and every constant or
    /// object imported from another unit, indexed by [`GlobalId`].
    pub globals: Vec<Global>,
    /// Every function of another unit this one calls, indexed by [`ExternId`].
    pub externs: Vec<ExternFn>,
    /// The function named `main`, if the unit declares one. Whether it has
    /// the right signature and linkage to start a program is checked when the
    /// program is linked.
    pub main: Option<FnId>,
    /// Whether other units may make instances of this unit's generic
    /// functions. Their bodies may call the unit's `static` functions and use
    /// its `static` objects, so those must then be reachable by the linker,
    /// though still not by name.
    pub shares_bodies: bool,
    /// The units whose generic or constant function bodies were checked
    /// here, each with a digest of the source they were read from. If that
    /// source changes, what this unit made from it is out of date.
    pub borrowed: Vec<(String, String)>,
    /// Each structure or enumeration that defines `$drop`, and the function
    /// it is. It runs when a value of the type is destroyed, before the
    /// value's fields are.
    pub drops: std::collections::HashMap<AdtId, Callee>,
    /// The unit initialiser: the function the `#unit` directive names, run
    /// once before `main` to give the unit's objects declared without a
    /// value theirs.
    pub init: Option<FnId>,
    /// Each structure or enumeration that defines `$copy`, and the function
    /// it is. It makes every implicit copy of a value of the type.
    pub copies: std::collections::HashMap<AdtId, Callee>,
    /// Whether this is a quantum unit, whose operators become circuits
    /// rather than code.
    pub quantum: bool,
    /// For a quantum unit, its `[entry]` operators whose signatures a
    /// classical program can hold as circuit handles.
    pub entries: Vec<FnId>,
    /// For a quantum unit, the names of what a classical unit may not use.
    pub quantum_only: Vec<Symbol>,
    /// The types whose run-time type information this unit makes: those it
    /// asks `@typeinfo` of, erases to `*any`, downcasts to, or marks
    /// `[typeinfo]`. The tables of the types theirs mention come with them.
    pub tables: Vec<Ty>,
    /// The `core` library's types a table is made of, once some table is
    /// needed.
    pub typeinfo: Option<TypeInfoTypes>,
    /// For each structure and enumeration whose table this unit makes, the
    /// methods the table lists.
    pub table_methods: std::collections::HashMap<AdtId, Vec<TableMethod>>,
    /// For each structure and enumeration whose table this unit makes, the
    /// annotations written on it: each one's name, and its argument where
    /// that is a single name, as `open` is in `[tcon: open]`.
    pub table_annots: std::collections::HashMap<AdtId, Vec<(Symbol, Option<Symbol>)>>,
    /// The structures and enumerations this unit declares that are marked
    /// `[deprecated: "msg"]`, by name, with their messages.
    pub deprecated_types: Vec<(Symbol, String)>,
    /// For a quantum unit, the geometry it declares: its covers, gauges,
    /// base types, locales, chains and map locales.
    pub geometry: Geometry,
    /// What its annotations claim of each operator that has claims.
    pub claims: std::collections::HashMap<FnId, claims::Claims>,
    /// The values its macros ask of its judgements.
    pub derived: Vec<claims::Derived>,
    /// Its `@static_assert`s whose conditions read judgements.
    pub deferred: Vec<claims::Deferred>,
    /// Its quantum enumerations' `[iso: …]` witnesses.
    pub isos: Vec<claims::Iso>,
    /// Its type aliases that take no generic parameters, each with the type
    /// it stands for. Generic aliases travel with the unit's source, as
    /// generic functions do.
    pub aliases: Vec<Alias>,
}

/// A type alias.
#[derive(Clone, Debug)]
pub struct Alias {
    /// The name as written.
    pub name: Symbol,
    /// The type it stands for.
    pub ty: Ty,
    /// Whether other units may name it: every one not declared `static`.
    pub linkage: crate::ast::Linkage,
}

/// A quantum unit's covers, gauges, base types, locales, chains and map
/// locales, each list indexed by position.
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    /// `cover Name = …;`
    pub covers: Vec<Named<crate::judge::cover::Cover>>,
    /// `gauge Name = …;`
    pub gauges: Vec<Named<crate::judge::cover::Gauge>>,
    /// `base Name = register, cover, gauge;`
    pub bases: Vec<Named<BaseDef>>,
    /// `locale Name = A of B { … }`
    pub locales: Vec<Named<LocaleDef>>,
    /// `chain Name = stage; …;`
    pub chains: Vec<Named<ChainDef>>,
    /// `qmap Name: Key -> Entry { … }`
    pub qmaps: Vec<Named<QmapDef>>,
    /// The covers, gauges and base types declared at unit scope without
    /// `static`, which other units may name, by name and position.
    pub exports: Vec<(Symbol, Exported)>,
}

/// A cover, gauge or base type of a unit's geometry, by its position in its
/// list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exported {
    /// A cover.
    Cover(usize),
    /// A gauge.
    Gauge(usize),
    /// A base type.
    Base(usize),
}

/// A map locale: operators on orthogonal supports, one for each value of a
/// key register.
#[derive(Clone, Debug)]
pub struct QmapDef {
    /// How many qubits the key register has.
    pub key: usize,
    /// The entries' type: an operator taking its qubits through handles.
    pub entry: Ty,
    /// Each key given an entry, and the entry, by key.
    pub entries: Vec<(u64, Expr)>,
}

/// A declaration of the geometry.
#[derive(Clone, Debug)]
pub struct Named<T> {
    /// The name.
    pub name: Symbol,
    /// What it declares.
    pub def: T,
    /// Where the name was written.
    pub span: Span,
}

/// A base type: a cover of a register's states, with a gauge on it.
#[derive(Clone, Copy, Debug)]
pub struct BaseDef {
    /// How many qubits the register has.
    pub width: usize,
    /// Its cover, by position.
    pub cover: usize,
    /// Its gauge, by position.
    pub gauge: usize,
}

/// A restriction `A of B`: the part of the container `B`'s cover that `A`'s
/// predicate cuts out, under `B`'s gauge.
#[derive(Clone, Debug)]
pub struct Region {
    /// The base types restricted.
    pub bases: Vec<usize>,
    /// The cover they leave: the common part of all of theirs.
    pub cover: crate::judge::cover::Cover,
}

/// A locale: a region, and the interfaces that may act on it.
#[derive(Clone, Debug)]
pub struct LocaleDef {
    /// The region.
    pub region: Region,
    /// Each interface: its name, its signature and where it was written.
    pub members: Vec<Named<SigId>>,
}

/// A chain: operators applied one after another, each from one base type to
/// the next.
#[derive(Clone, Debug)]
pub struct ChainDef {
    /// The stages.
    pub stages: Vec<ChainStage>,
    /// Whether it ends at the base type it starts at.
    pub closed: bool,
    /// Its `[expect: holonomy(s) = k]` claims: the point of its first base
    /// type's cover, by index, the holonomy claimed there, and where.
    pub expects: Vec<(usize, crate::exact::Phase, Span)>,
}

/// One stage of a chain.
#[derive(Clone, Debug)]
pub struct ChainStage {
    /// The operator it applies.
    pub op: Expr,
    /// The base type it starts at.
    pub from: usize,
    /// The base type it ends at.
    pub to: usize,
}

/// A method as a type's table lists it.
#[derive(Clone, Copy, Debug)]
pub struct TableMethod {
    /// Its name, with `$` for an operator method.
    pub name: Symbol,
    /// Its signature with the receiver erased to `*any`, or `*const any`, so
    /// that it can be called through a reference whose type is known only
    /// from its table.
    pub sig: Ty,
    /// The function it is.
    pub callee: Callee,
}

/// The `core` library's structures that make up run-time type information.
#[derive(Clone, Copy, Debug)]
pub struct TypeInfoTypes {
    /// `TypeInfo`.
    pub info: AdtId,
    /// `FieldInfo`.
    pub field: AdtId,
    /// `MethodInfo`.
    pub method: AdtId,
    /// `VariantInfo`.
    pub variant: AdtId,
    /// `AnnotInfo`.
    pub annot: AdtId,
    /// `TypeKind`.
    pub kind: AdtId,
}

impl Unit {
    /// An empty unit.
    pub fn new(name: &str, source: SourceId) -> Unit {
        Unit {
            name: name.to_owned(),
            source,
            types: TypeTable::new(),
            fns: Vec::new(),
            globals: Vec::new(),
            externs: Vec::new(),
            main: None,
            shares_bodies: false,
            borrowed: Vec::new(),
            drops: std::collections::HashMap::new(),
            copies: std::collections::HashMap::new(),
            init: None,
            quantum: false,
            entries: Vec::new(),
            quantum_only: Vec::new(),
            tables: Vec::new(),
            typeinfo: None,
            table_methods: std::collections::HashMap::new(),
            table_annots: std::collections::HashMap::new(),
            deprecated_types: Vec::new(),
            geometry: Geometry::default(),
            claims: std::collections::HashMap::new(),
            derived: Vec::new(),
            deferred: Vec::new(),
            isos: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// A function by id.
    pub fn func(&self, id: FnId) -> &Fn {
        &self.fns[id.index()]
    }

    /// An object by id.
    pub fn global(&self, id: GlobalId) -> &Global {
        &self.globals[id.index()]
    }

    /// An imported function by id.
    pub fn extern_fn(&self, id: ExternId) -> &ExternFn {
        &self.externs[id.index()]
    }
}

/// Whether a function has code of its own at run time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FnKind {
    /// An ordinary function.
    Runtime,
    /// A function whose parameters are all `constexpr`, or that has none and a
    /// `constexpr` result. Its arguments exist only during translation, so
    /// every call is evaluated then and replaced by its result. It is never
    /// compiled.
    Constant,
}

/// A function.
#[derive(Clone, Debug)]
pub struct Fn {
    /// The name as declared.
    pub name: Symbol,
    /// The generic arguments this instance was made with, empty for a
    /// function declared without generic parameters.
    pub args: Vec<Arg>,
    /// The unit that declared it, when that is another unit: an instance of
    /// another unit's generic function, made here because this unit uses it,
    /// or another unit's constant function, here to be evaluated. `None` for
    /// the unit's own functions.
    pub origin: Option<String>,
    /// The type it is a method or associated function of, if it is one.
    pub method: Option<MethodOf>,
    /// Whether it is a closure's body, lifted out of the function the closure
    /// was written in. Its first parameter is a reference to the closure's
    /// environment, and it has no name of its own.
    pub closure: bool,
    /// Whether other units may name it.
    pub linkage: Linkage,
    /// Whether it is compiled or only evaluated.
    pub kind: FnKind,
    /// The parameters.
    pub params: Vec<LocalId>,
    /// The return type; `void` when none is written.
    pub ret: Ty,
    /// Every binding in the function, parameters first.
    pub locals: Vec<Local>,
    /// The body.
    pub body: Block,
    /// The function's name as written.
    pub span: Span,
    /// What its annotations ask of it.
    pub attrs: FnAttrs,
}

/// What a function's annotations ask of it.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct FnAttrs {
    /// The symbol `[export: "sym"]` gives it in place of its own.
    pub symbol: Option<String>,
    /// `[inline]` or `[inline: never]`.
    pub inline: Option<Inline>,
    /// Whether it is a `[test]`, which `tqc test` runs.
    pub test: bool,
    /// Whether it is an `[entry]` operator of a quantum unit: one a circuit
    /// is made for, which a classical program holds as a circuit handle.
    pub entry: bool,
    /// Whether it is marked `[trusted_unitary]`: the matrices it gives
    /// `apply` are taken to be unitary without being checked.
    pub trusted_unitary: bool,
    /// The operator `[adjoint: f]` declares its adjoint, which `adjoint`
    /// applies in place of undoing it.
    pub adjoint: Option<FnId>,
    /// Whether it is marked `[dynamic]`: its circuit's structure depends on
    /// a measured value.
    pub dynamic: bool,
    /// The name `[adjoint: f]` gives, until it is resolved.
    pub adjoint_name: Option<(Symbol, Span)>,
    /// The message of `[deprecated: "msg"]`, repeated at each use.
    pub deprecated: Option<String>,
    /// Whether it is kept from other units: a function the compiler made,
    /// such as one standing for a library function used as a value, or one
    /// declared inside a block. Its symbol is its own, never offered.
    pub hidden: bool,
}

/// An inlining hint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Inline {
    /// `[inline]`: inlining it is expected to pay.
    Hint,
    /// `[inline: never]`: it is never inlined.
    Never,
}

impl Fn {
    /// A function of this unit named `name`, declared at `span`, taking
    /// nothing, returning `void`, with an empty body at `body_span`: what
    /// its declaration fills in once resolved.
    pub fn empty(name: Symbol, linkage: Linkage, span: Span, body_span: Span) -> Fn {
        Fn {
            name,
            args: Vec::new(),
            origin: None,
            method: None,
            closure: false,
            linkage,
            kind: FnKind::Runtime,
            params: Vec::new(),
            ret: Ty::Void,
            locals: Vec::new(),
            body: Block {
                stmts: Vec::new(),
                value: None,
                ty: Ty::Void,
                span: body_span,
            },
            span,
            attrs: FnAttrs::default(),
        }
    }

    /// A binding by id.
    pub fn local(&self, id: LocalId) -> &Local {
        &self.locals[id.index()]
    }

    /// The parameter types.
    pub fn param_types(&self) -> impl Iterator<Item = Ty> + '_ {
        self.params.iter().map(|&p| self.local(p).ty)
    }
}

/// A parameter or local binding.
#[derive(Clone, Debug)]
pub struct Local {
    /// The name as written.
    pub name: Symbol,
    /// Its type.
    pub ty: Ty,
    /// Whether its type was `const` or `constexpr`: it is never assigned.
    /// A `const` binding whose value can be computed during translation is
    /// folded into its uses. Every other binding may be assigned.
    pub constant: bool,
    /// Whether its type was `constexpr`: its value must be known during
    /// translation, as a constant function's parameters always are.
    pub constexpr: bool,
    /// Whether it is an ancilla, declared `aux let`: allocated in |0>, and
    /// returned to |0> by uncomputation when its scope ends.
    pub aux: bool,
    /// Where it was declared.
    pub span: Span,
}

/// A function of another unit.
#[derive(Clone, Debug)]
pub struct ExternFn {
    /// The unit that defines it.
    pub unit: String,
    /// Its name there.
    pub name: Symbol,
    /// The type it is a method or associated function of, if it is one.
    pub method: Option<MethodOf>,
    /// Its parameter types.
    pub params: Vec<Ty>,
    /// Its return type.
    pub ret: Ty,
    /// The symbol `[export: "sym"]` gave it, if any.
    pub symbol: Option<String>,
    /// The message of `[deprecated: "msg"]`, repeated at each use.
    pub deprecated: Option<String>,
}

/// What a method or associated function belongs to.
///
/// A method is found through its type, never by name alone, so two types may
/// each have a method `len`, and a type may have both a method `add` and the
/// operator method `$add`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MethodOf {
    /// The type.
    pub owner: AdtId,
    /// Whether it is an operator method, written with `$`.
    pub operator: bool,
    /// Whether it takes `self`, so is called as `value.name(…)`; otherwise it
    /// is an associated function, called as `Type::name(…)`.
    pub receiver: bool,
}

/// Where an object lives and how long.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GlobalKind {
    /// Declared at unit scope.
    Unit,
    /// Declared `persist let` inside the given function: one object for the
    /// whole program, visible only inside that function's block.
    Persist(FnId),
    /// Declared at unit scope by another unit, whose name this is. A constant
    /// arrives with its value; an object is reached through that unit's
    /// symbol.
    Imported(String),
}

/// A unit-duration or persistent object.
///
/// Its initial value is either a constant, placed directly in the image so
/// that no code runs to initialise it, or (for a unit object declared
/// without one) whatever the unit's initialiser assigns before `main` runs.
#[derive(Clone, Debug)]
pub struct Global {
    /// The name as written.
    pub name: Symbol,
    /// Its type.
    pub ty: Ty,
    /// Whether its type was `const`. A constant has no storage: every use has
    /// already been replaced by its value.
    pub constant: bool,
    /// Whether other units may name it.
    pub linkage: Linkage,
    /// Unit scope, persistent, or imported.
    pub kind: GlobalKind,
    /// The initialiser as checked.
    pub init: Option<Expr>,
    /// The initial value.
    pub value: Option<Value>,
    /// Whether it is a unit object given its value by the unit's initialiser,
    /// having none written where it is declared. Until then its storage is
    /// zero.
    pub startup: bool,
    /// Where it was declared.
    pub span: Span,
    /// The symbol `[export: "sym"]` gave it, here or in the unit that
    /// declares it.
    pub symbol: Option<String>,
    /// The message of `[deprecated: "msg"]`, repeated at each use.
    pub deprecated: Option<String>,
}

/// A block: statements and an optional trailing value.
#[derive(Clone, Debug)]
pub struct Block {
    /// The statements.
    pub stmts: Vec<Stmt>,
    /// The trailing expression.
    pub value: Option<Box<Expr>>,
    /// The block's type: the value's, `never` if a statement diverges, or
    /// `void`.
    pub ty: Ty,
    /// Where the block was written.
    pub span: Span,
}

/// A statement.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let`. A binding declared without a value must be assigned before it
    /// is read, which analysis checks on every path.
    Let {
        /// The binding it introduces.
        local: LocalId,
        /// Its initial value, if one was written.
        init: Option<Expr>,
    },
    /// `let (a, b) = e;`: a value taken apart by a pattern that matches
    /// every value of its type, binding what it names.
    LetPat {
        /// The pattern.
        pat: Pat,
        /// The value taken apart.
        init: Expr,
    },
    /// An expression evaluated for its effect.
    Expr(Expr),
}

/// An expression with its type.
#[derive(Clone, Debug)]
pub struct Expr {
    /// What it computes.
    pub kind: ExprKind,
    /// The type of its value.
    pub ty: Ty,
    /// Where it was written; an abort reports this line.
    pub span: Span,
}

impl Expr {
    /// A constant of type `ty`.
    pub fn constant(value: Value, ty: Ty, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Const(value),
            ty,
            span,
        }
    }

    /// A call of `callee` with `args`.
    pub fn call(callee: Callee, args: Vec<Expr>, ty: Ty, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Call { callee, args },
            ty,
            span,
        }
    }

    /// A read of the binding `local`.
    pub fn local(local: LocalId, ty: Ty, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Local(local),
            ty,
            span,
        }
    }

    /// `{ let local = init; then; }`, a block of type `ty` with no value.
    pub fn let_then(local: LocalId, init: Expr, then: Expr, ty: Ty, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Block(Box::new(Block {
                stmts: vec![Stmt::Let { local, init: Some(init) }, Stmt::Expr(then)],
                value: None,
                ty,
                span,
            })),
            ty,
            span,
        }
    }

    /// What the reference `r` refers to, a `ty`.
    pub fn deref(r: Expr, ty: Ty, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Deref(Box::new(r)),
            ty,
            span,
        }
    }

    /// A use of the library function `which` on `args`, giving a `ty`.
    pub fn intrinsic(which: Intrinsic, args: Vec<Expr>, ty: Ty, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Intrinsic { which, args },
            ty,
            span,
        }
    }

    /// Whether this expression denotes storage (see the module
    /// documentation).
    pub fn is_place(&self) -> bool {
        match &self.kind {
            ExprKind::Local(_)
            | ExprKind::Global(_)
            | ExprKind::Deref(_)
            | ExprKind::Index { .. }
            | ExprKind::FnRef(_)
            | ExprKind::ThunkRef(_) => true,
            ExprKind::Field { base, .. } => base.is_place(),
            _ => false,
        }
    }

    /// Visits this expression and everything under it, children before
    /// parents, allowing each to be rewritten.
    pub fn walk_mut(&mut self, f: &mut impl FnMut(&mut Expr)) {
        struct Post<'f, F>(&'f mut F);
        impl<F: FnMut(&mut Expr)> visit::VisitMut for Post<'_, F> {
            fn expr(&mut self, e: &mut Expr) {
                visit::walk_expr_mut(self, e);
                (self.0)(e);
            }
        }
        visit::VisitMut::expr(&mut Post(f), self);
    }
}

impl Block {
    /// Visits every expression in the block, children before parents.
    pub fn walk_mut(&mut self, f: &mut impl FnMut(&mut Expr)) {
        for s in &mut self.stmts {
            match s {
                Stmt::Let { init, .. } => {
                    if let Some(e) = init {
                        e.walk_mut(f);
                    }
                }
                Stmt::LetPat { init, .. } => init.walk_mut(f),
                Stmt::Expr(e) => e.walk_mut(f),
            }
        }
        if let Some(v) = &mut self.value {
            v.walk_mut(f);
        }
    }
}

/// What a call calls.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Callee {
    /// A function of this unit.
    Fn(FnId),
    /// A function of another unit.
    Extern(ExternId),
}

/// An operation the language provides directly rather than through a
/// function with a body.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Intrinsic {
    /// `print(msg)`: writes the characters to standard output, as UTF-8.
    Print,
    /// `eprint(msg)`: writes the characters to standard error.
    Eprint,
    /// `exit(code)`: ends the program with that status. Never returns.
    Exit,
    /// `panic(msg)`: aborts with `RA10` and the message. Never returns.
    Panic,
    /// `xs.len()` on an array or slice: its element count.
    Len,
    /// Aborts with the abort identifier `RAnn` numbered here, reporting the
    /// message. Never returns. `core::abort` becomes this once its
    /// identifier is known, and `unwrap` on a `None` or an `Err` is this with
    /// `RA10`.
    Abort(u8),
    /// `a op b` on two integers of one type, giving the result wrapped to
    /// the type's width and whether it was wrapped, as a `(T, bool)`: for
    /// `+`, `-`, `*`, `/`, `%`, `<<` and `>>`. A shift count is taken modulo
    /// the width. Never aborts; division by zero is the caller's to exclude.
    Overflowing(BinOp),
    /// `-a` wrapped, and whether it was.
    OverflowingNeg,
    /// `a op b` clamped to the type's range, for `+`, `-` and `*`.
    Saturating(BinOp),
    /// An integer converted to another integer type by keeping its low bits,
    /// or widened.
    Truncate,
    /// An integer or floating value converted to an integer type, clamped to
    /// its range; NaN becomes zero.
    SaturatingAs,
    /// A deep copy of what a reference refers to.
    Clone,
    /// The values two references of one type refer to, exchanged byte for
    /// byte, so that nothing is copied or destroyed.
    Exchange,
    /// `@typeinfo(T)`: the address of `T`'s table, as a `*const TypeInfo`.
    TypeInfo(Ty),
    /// `@typeinfo_of(p)`: the table a reference carries.
    TypeInfoOf,
    /// `p as *T` where the conversion is not known to hold: `Some(p)` as a
    /// `*T` when the table `p` carries is `T`'s, and otherwise `None`.
    Downcast(Ty),
    /// The function an address `*void` holds, as a value of the function
    /// type given, once the table the address carries is found to be that
    /// type's; otherwise an abort with `RA11`. What `invoke<Sig>` calls.
    Invoke(Ty),
    /// A value moved into a buffer of its own, as a `dyn` carrying the
    /// table of the type given.
    DynOf(Ty),
    /// `d as T`: `Some` of the value a `dyn` holds when its table is the
    /// given type's, and otherwise `None`, the `dyn` destroyed.
    DynAs(Ty),
    /// The value a `dyn` holds, moved out, its buffer freed. The type is the
    /// call's; the library checks it first.
    DynTake,
    /// A reference to a `dyn` as a reference to what it holds: `*any`.
    DynView,
    /// `(p, offset, ty)`: a reference `offset` bytes into what `p` refers
    /// to, carrying the table `ty`.
    FieldAt,
    /// `(call, self, args)`: a method's `call` entry called with the
    /// receiver and the arguments as `dyn` values, giving its result as one.
    DynCall,
    /// A `dyn` holding a value of the type a table describes, all of whose
    /// bytes are zero.
    DynMake,
    /// `(p, v)`: the value a `dyn` holds moved into the place `p` refers to,
    /// whose old value is destroyed first.
    DynStore,
    /// `(c)`: the document of the circuit a handle refers to, as the
    /// characters the handle is.
    CircuitDoc,
    /// `(doc)`: the circuit handle referring to a circuit's document, which
    /// it is: a handle is the document's characters.
    CircuitOf,
    /// Whether a reference's address is null, as a table's empty entries are.
    IsNull,
    /// `@raw_parts(r)`: a reference's two words as a tuple, its address as
    /// an integer.
    RawParts,
    /// `@from_raw_parts(addr, n)`: a reference made from an address and a
    /// length or a table, which nothing checks.
    FromRawParts,
    /// `(place, n)`: the empty growable array `place` refers to given `n`
    /// elements whose bytes are all zero, and a reference to the first.
    GrowZeroed,
    /// `(x, count, buf)`: the first `count` significant decimal digits of
    /// `x`, rounded, written to `buf` and ended with a zero byte; gives where
    /// the decimal point falls among them.
    FloatDigits,
    /// `(text)`: the floating value a zero-ended decimal numeral denotes,
    /// correctly rounded.
    ParseFloat,
    /// The sine of an angle in radians.
    Sin,
    /// The cosine of an angle in radians.
    Cos,
    /// `(path, write)`: opens the file a zero-ended UTF-16 path names, for
    /// reading, or for writing from empty; gives its handle, or the
    /// system's error code negated.
    FileOpen,
    /// `(handle)`: the size of an open file in bytes, or the error negated.
    FileSize,
    /// `(handle, buf)`: reads into `buf`, giving how many bytes were read,
    /// or the error negated.
    FileRead,
    /// `(handle, data)`: writes `data`, giving how many bytes were written,
    /// or the error negated.
    FileWrite,
    /// `(handle)`: closes an open file.
    FileClose,
    /// `(which)`: the handle of standard input, output or error, for 0, 1
    /// or 2, as the system gives it; 0 or -1 when the program has none.
    StdHandle,
    /// `(handle)`: whether a handle is a console's.
    IsConsole,
    /// `(handle, units, raw)`: reads UTF-16 code units from a console into
    /// `units`, giving how many were read, or the error negated. `raw` reads
    /// what a key types as soon as it is typed, without echoing it; without
    /// it, the console's own mode applies.
    ConsoleRead,
    /// `(path)`: loads the library a zero-ended UTF-16 path names, giving
    /// its handle, or the system's error code negated.
    ModuleOpen,
    /// `(handle, name)`: the address of what a loaded library exports under
    /// a zero-ended name, or zero.
    ModuleFind,
    /// `(handle)`: unloads a library, giving whether that succeeded.
    ModuleClose,
    /// `(address)`: calls the function at an address that takes nothing and
    /// returns nothing.
    CallVoid,
    /// The address of the program's own descriptor, which says what it
    /// holds for a module loaded into it to be checked against.
    HostDescriptor,
    /// The time from the system clock, in 100-nanosecond ticks since 1601:
    /// a different number each run, for seeding a generator.
    Clock,
    /// `(handles…)`, or `(phase, handles…)` for a rotation: a gate of the
    /// standard set applied to the qubits the handles name. Only a quantum
    /// unit applies one; it becomes part of a circuit.
    Gate(Gate),
    /// `(matrix, handle)`: an exact unitary matrix applied as a gate to the
    /// register the handle names.
    Apply,
    /// `(k)`: a handle to the processor's node `k`.
    Node,
    /// `adjoint(f)`: the operator that undoes the operator `f`.
    Adjoint,
    /// `controlled(f)`: the operator applying `f` controlled on the qubit
    /// its first argument names.
    Controlled,
    /// `f.then(g)`: the operator applying `f` and then `g` to the same
    /// arguments.
    Then,
    /// A value derived from the unit's judgements once its circuits are
    /// generated, by position among [`Unit::derived`]: `@judgment(f)`,
    /// `@holonomy(C, s)` and their kin.
    Derived(u32),
}

/// A gate of the standard set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Gate {
    /// The bit flip.
    X,
    /// The bit and phase flip.
    Y,
    /// The phase flip.
    Z,
    /// The Hadamard gate.
    H,
    /// The quarter turn of phase (i.e. `diag(1, i)`).
    S,
    /// Its inverse.
    Sdg,
    /// The eighth turn of phase (i.e. `diag(1, e^(iπ/4))`).
    T,
    /// Its inverse.
    Tdg,
    /// The bit flip of a target.
    Cx,
    /// The phase flip.
    Cz,
    /// The bit flip of a target.
    Ccx,
    /// The exchange of two qubits.
    Swap,
    /// `diag(1, e^(iθ))` for a phase θ of the phase group.
    Rz,
    /// `e^(iθ)` times the identity: a global phase, which becomes a relative
    /// one once the gate is controlled.
    GPhase,
}

impl Gate {
    /// The name the library gives it.
    pub fn name(self) -> &'static str {
        match self {
            Gate::X => "x",
            Gate::Y => "y",
            Gate::Z => "z",
            Gate::H => "h",
            Gate::S => "s",
            Gate::Sdg => "sdag",
            Gate::T => "t",
            Gate::Tdg => "tdag",
            Gate::Cx => "cx",
            Gate::Cz => "cz",
            Gate::Ccx => "ccx",
            Gate::Swap => "swap",
            Gate::Rz => "rz",
            Gate::GPhase => "gphase",
        }
    }

    /// How many qubits it acts on.
    pub fn arity(self) -> usize {
        match self {
            Gate::GPhase => 0,
            Gate::Cx | Gate::Cz | Gate::Swap => 2,
            Gate::Ccx => 3,
            _ => 1,
        }
    }

    /// Whether it takes a phase before its qubits.
    pub fn takes_phase(self) -> bool {
        matches!(self, Gate::Rz | Gate::GPhase)
    }

    /// The name the library's own code reaches it by.
    pub fn intrinsic_name(self) -> &'static str {
        match self {
            Gate::X => "__x",
            Gate::Y => "__y",
            Gate::Z => "__z",
            Gate::H => "__h",
            Gate::S => "__s",
            Gate::Sdg => "__sdag",
            Gate::T => "__t",
            Gate::Tdg => "__tdag",
            Gate::Cx => "__cx",
            Gate::Cz => "__cz",
            Gate::Ccx => "__ccx",
            Gate::Swap => "__swap",
            Gate::Rz => "__rz",
            Gate::GPhase => "__gphase",
        }
    }

    /// Every gate of the standard set.
    pub const ALL: [Gate; 14] = [
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::Cx,
        Gate::Cz,
        Gate::Ccx,
        Gate::Swap,
        Gate::Rz,
        Gate::GPhase,
    ];
}

/// An operation on a growable array.
///
/// The array's buffer, when it has one, starts with a header of two words,
/// the capacity in elements and then the element type's description, and the
/// elements follow. An empty array that has never grown has no buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GrowOp {
    /// `xs.push(v)`: appends `v`, growing the buffer if it is full.
    Push,
    /// `xs.pop()`: removes the last element and gives it as `Some`, or
    /// `None` if there is none.
    Pop,
    /// `xs.reserve(n)`: makes room for at least `n` more elements.
    Reserve,
    /// `xs.clear()`: destroys every element, keeping the buffer.
    Clear,
    /// `xs.truncate(n)`: destroys the elements from `n` on, if there are any.
    Truncate,
    /// Moves the element at an index out, leaving its slot to be forgotten.
    Take,
    /// Forgets the first `n` elements, which have been moved out, and moves
    /// the rest down to take their place.
    ForgetFront,
}

impl GrowOp {
    /// The name a program calls it by.
    pub fn name(self) -> &'static str {
        match self {
            GrowOp::Push => "push",
            GrowOp::Pop => "pop",
            GrowOp::Reserve => "reserve",
            GrowOp::Clear => "clear",
            GrowOp::Truncate => "truncate",
            GrowOp::Take => "__take",
            GrowOp::ForgetFront => "__forget_front",
        }
    }
}

impl Intrinsic {
    /// Whether it works on a `dyn` value or a circuit handle, which exist
    /// only in the running program.
    pub fn is_dyn(self) -> bool {
        matches!(
            self,
            Intrinsic::DynOf(_)
                | Intrinsic::DynTake
                | Intrinsic::DynView
                | Intrinsic::FieldAt
                | Intrinsic::DynCall
                | Intrinsic::DynMake
                | Intrinsic::DynStore
                | Intrinsic::IsNull
                | Intrinsic::CircuitOf
                | Intrinsic::CircuitDoc
        )
    }

    /// The name a program calls it by.
    pub fn name(self) -> &'static str {
        match self {
            Intrinsic::Print => "print",
            Intrinsic::Eprint => "eprint",
            Intrinsic::Exit => "exit",
            Intrinsic::Panic => "panic",
            Intrinsic::Len => "len",
            Intrinsic::Abort(_) => "abort",
            Intrinsic::Overflowing(_) => "__overflowing",
            Intrinsic::OverflowingNeg => "__overflowing_neg",
            Intrinsic::Saturating(_) => "__saturating",
            Intrinsic::Truncate => "__truncate",
            Intrinsic::SaturatingAs => "__saturating_as",
            Intrinsic::Clone => "__clone",
            Intrinsic::Exchange => "__exchange",
            Intrinsic::TypeInfo(_) => "@typeinfo",
            Intrinsic::TypeInfoOf => "@typeinfo_of",
            Intrinsic::Downcast(_) => "as",
            Intrinsic::Invoke(_) => "invoke",
            Intrinsic::DynOf(_) => "__dyn_of",
            Intrinsic::DynAs(_) => "as",
            Intrinsic::DynTake => "__dyn_take",
            Intrinsic::DynView => "__dyn_view",
            Intrinsic::FieldAt => "__field_at",
            Intrinsic::DynCall => "__dyn_call",
            Intrinsic::DynMake => "__dyn_make",
            Intrinsic::DynStore => "__dyn_store",
            Intrinsic::IsNull => "__is_null",
            Intrinsic::CircuitOf => "__circuit_of",
            Intrinsic::CircuitDoc => "__circuit_doc",
            Intrinsic::RawParts => "@raw_parts",
            Intrinsic::FromRawParts => "@from_raw_parts",
            Intrinsic::GrowZeroed => "__grow_zeroed",
            Intrinsic::FloatDigits => "__float_digits",
            Intrinsic::ParseFloat => "__parse_float",
            Intrinsic::Sin => "__sin",
            Intrinsic::Cos => "__cos",
            Intrinsic::FileOpen => "__file_open",
            Intrinsic::FileSize => "__file_size",
            Intrinsic::FileRead => "__file_read",
            Intrinsic::FileWrite => "__file_write",
            Intrinsic::FileClose => "__file_close",
            Intrinsic::StdHandle => "__std_handle",
            Intrinsic::IsConsole => "__is_console",
            Intrinsic::ConsoleRead => "__console_read",
            Intrinsic::ModuleOpen => "__module_open",
            Intrinsic::ModuleFind => "__module_find",
            Intrinsic::ModuleClose => "__module_close",
            Intrinsic::CallVoid => "__call_void",
            Intrinsic::HostDescriptor => "__host_descriptor",
            Intrinsic::Clock => "__clock",
            Intrinsic::Gate(g) => g.intrinsic_name(),
            Intrinsic::Apply => "apply",
            Intrinsic::Node => "@node",
            Intrinsic::Adjoint => "adjoint",
            Intrinsic::Controlled => "controlled",
            Intrinsic::Then => "then",
            Intrinsic::Derived(_) => "@judgment",
        }
    }

    /// Whether this acts on the world outside the program: its input and
    /// output, its ending, files, loaded modules and their code.
    pub fn acts_on_environment(self) -> bool {
        matches!(
            self,
            Intrinsic::Print
                | Intrinsic::Eprint
                | Intrinsic::Exit
                | Intrinsic::FileOpen
                | Intrinsic::FileSize
                | Intrinsic::FileRead
                | Intrinsic::FileWrite
                | Intrinsic::FileClose
                | Intrinsic::StdHandle
                | Intrinsic::IsConsole
                | Intrinsic::ConsoleRead
                | Intrinsic::ModuleOpen
                | Intrinsic::ModuleFind
                | Intrinsic::ModuleClose
                | Intrinsic::CallVoid
                | Intrinsic::HostDescriptor
                | Intrinsic::Clock
        )
    }

    /// Whether a call never returns.
    pub fn diverges(self) -> bool {
        matches!(self, Intrinsic::Exit | Intrinsic::Panic | Intrinsic::Abort(_))
    }
}

/// One arm of a `match`.
#[derive(Clone, Debug)]
pub struct Arm {
    /// What it matches.
    pub pat: Pat,
    /// What it evaluates to.
    pub body: Expr,
}

/// A pattern.
#[derive(Clone, Debug)]
pub struct Pat {
    /// What it tests and binds.
    pub kind: PatKind,
    /// The type of the value it is matched against.
    pub ty: Ty,
    /// Where it was written.
    pub span: Span,
}

impl Pat {
    /// `_`, matched against a value of `ty`.
    pub fn wild(ty: Ty, span: Span) -> Pat {
        Pat {
            kind: PatKind::Wild,
            ty,
            span,
        }
    }

    /// A binding of `local`.
    pub fn bind(local: LocalId, ty: Ty, span: Span) -> Pat {
        Pat {
            kind: PatKind::Bind(local),
            ty,
            span,
        }
    }

    /// The variant `variant` of the enumeration `ty`, its fields matched by
    /// `fields`.
    pub fn variant(variant: u32, fields: Vec<(u32, Pat)>, ty: Ty, span: Span) -> Pat {
        Pat {
            kind: PatKind::Variant { variant, fields },
            ty,
            span,
        }
    }
}

/// What a pattern tests and binds.
#[derive(Clone, Debug)]
pub enum PatKind {
    /// `_`: matches anything, binds nothing.
    Wild,
    /// A name: matches anything and binds it to a new local, copying or
    /// moving the matched value into it.
    Bind(LocalId),
    /// A name, in a `match` through a reference on a part that cannot be
    /// copied: matches anything and binds a reference to it, of the
    /// pattern's type, to a new local.
    BindRef(LocalId),
    /// A literal: matches only that value.
    Const(Value),
    /// A structure's fields. Fields not listed match anything.
    Struct {
        /// Field index and the pattern it must match.
        fields: Vec<(u32, Pat)>,
    },
    /// One variant of an enumeration, and patterns for its fields.
    Variant {
        /// The variant's index.
        variant: u32,
        /// Field index and the pattern it must match.
        fields: Vec<(u32, Pat)>,
    },
}

/// What an expression computes.
#[derive(Clone, Debug)]
pub enum ExprKind {
    /// A value known during translation.
    Const(Value),
    /// Reads a parameter or local.
    Local(LocalId),
    /// Reads a unit-duration, persistent or imported object.
    Global(GlobalId),
    /// Calls a function.
    Call {
        /// The function.
        callee: Callee,
        /// The arguments.
        args: Vec<Expr>,
    },
    /// A function as a value: its code's address, of a function type. It is
    /// also a place (a constant one, holding that address), so that `&f` is
    /// a reference that stays valid for the whole program.
    FnRef(Callee),
    /// The code of a closure that captures nothing, as a plain function of
    /// the closure's signature, which is how such a closure becomes a
    /// `*fn`. Like [`ExprKind::FnRef`], a constant place.
    ThunkRef(FnId),
    /// A reference to a function, `*fn(…)`, as a closure of the same
    /// signature. Its environment is where the reference points, and its
    /// code calls whatever function is there, so nothing is allocated.
    FnAsClosure(Box<Expr>),
    /// Calls a function value or a closure: `callee` has a function or
    /// closure type, and is evaluated before the arguments.
    IndirectCall {
        /// What is called.
        callee: Box<Expr>,
        /// The arguments.
        args: Vec<Expr>,
    },
    /// Makes a closure: its environment holds `captures`, in order (each a
    /// value moved in, or a reference), and `code` is its body, a function
    /// whose first parameter is a reference to the environment.
    Closure {
        /// The body.
        code: FnId,
        /// What it captures.
        captures: Vec<Expr>,
    },
    /// Uses an operation the language provides.
    Intrinsic {
        /// Which one.
        which: Intrinsic,
        /// Its operands, evaluated left to right.
        args: Vec<Expr>,
    },
    /// An operation on a growable array where it is, without moving it.
    Growable {
        /// Which one.
        op: GrowOp,
        /// The array: a place, or a temporary.
        array: Box<Expr>,
        /// The other operands.
        args: Vec<Expr>,
    },
    /// A fixed array's elements moved into a new growable array.
    Grow(Box<Expr>),
    /// `-e` or `!e`.
    Unary {
        /// The operator.
        op: UnOp,
        /// The operand.
        operand: Box<Expr>,
    },
    /// An arithmetic, bitwise, shift or comparison operator. Both operands are
    /// always evaluated, left first.
    Binary {
        /// The operator.
        op: BinOp,
        /// The left operand.
        lhs: Box<Expr>,
        /// The right operand.
        rhs: Box<Expr>,
    },
    /// `&&` or `||`, which evaluate the right operand only if the left does
    /// not already decide the result.
    Logical {
        /// The operator.
        op: LogicalOp,
        /// The left operand.
        lhs: Box<Expr>,
        /// The right operand.
        rhs: Box<Expr>,
    },
    /// `e as T` to an integer type, from an integer or a `char`, which aborts
    /// if the value does not fit.
    Cast {
        /// The value converted.
        expr: Box<Expr>,
        /// The type converted to: an integer, a floating type or `char`.
        to: Ty,
    },
    /// One field of a structure.
    Field {
        /// The structure: a place, read in place, or a value.
        base: Box<Expr>,
        /// The field's index.
        field: u32,
    },
    /// One element of a fixed array or slice, after checking the index is
    /// less than the length.
    Index {
        /// The array (a place or value) or the slice.
        base: Box<Expr>,
        /// The index.
        index: Box<Expr>,
    },
    /// The object a reference refers to.
    Deref(Box<Expr>),
    /// `&e`, a `*T`, or a `*const T` of a constant. A place gives its own
    /// address; any other value is first stored in a temporary that lasts
    /// until the function returns.
    Ref(Box<Expr>),
    /// A structure value, `Point { x: 1, y: 2 }`. The type says which
    /// structure.
    StructLit {
        /// Field index and value, in the order written, which is the order
        /// they are evaluated in.
        fields: Vec<(u32, Expr)>,
    },
    /// An enumeration value. The type says which enumeration.
    Variant {
        /// The variant's index.
        variant: u32,
        /// Field index and value, in the order written.
        fields: Vec<(u32, Expr)>,
    },
    /// `[a, b, c]`.
    ArrayLit(Vec<Expr>),
    /// `[e; N]`: `e` evaluated once and copied into every element.
    ArrayRepeat {
        /// The element.
        elem: Box<Expr>,
        /// How many.
        len: u64,
    },
    /// A reference converted to a type that promises less, such as `*T` to
    /// `*const T`, or `*[T; N]` to `*[T]`. Its representation does not change.
    Coerce(Box<Expr>),
    /// Stores a value into a place. The value is evaluated first.
    Assign {
        /// Where the value goes: a place expression.
        place: Box<Expr>,
        /// The value.
        value: Box<Expr>,
    },
    /// A block used as an expression.
    Block(Box<Block>),
    /// `if cond { … } else …`.
    If {
        /// The condition.
        cond: Box<Expr>,
        /// The branch taken when it holds.
        then: Box<Block>,
        /// The other branch: a block, or another `if`.
        els: Option<Box<Expr>>,
    },
    /// `match scrutinee { … }`. The arms are tried in order; analysis has
    /// checked that some arm matches every value.
    Match {
        /// The value matched: a place, examined in place, or a value.
        scrutinee: Box<Expr>,
        /// The arms.
        arms: Vec<Arm>,
    },
    /// `loop { … }`, left only by `break`, `return` or an abort.
    Loop {
        /// This loop, for its `break`s.
        id: LoopId,
        /// The body.
        body: Box<Block>,
    },
    /// `while cond { … }`.
    While {
        /// This loop.
        id: LoopId,
        /// The condition.
        cond: Box<Expr>,
        /// The body.
        body: Box<Block>,
    },
    /// `for var in start..end` or `start..=end` over integers.
    ///
    /// Kept as its own node rather than lowered to `while`, because a range
    /// ending at its type's maximum must stop without ever computing the
    /// successor of that maximum, which would overflow.
    ForRange {
        /// This loop.
        id: LoopId,
        /// The loop variable.
        var: LocalId,
        /// The first value.
        start: Box<Expr>,
        /// The bound.
        end: Box<Expr>,
        /// Whether the bound itself is included.
        inclusive: bool,
        /// The body.
        body: Box<Block>,
    },
    /// Leaves a loop, optionally with the loop's value.
    Break {
        /// The loop left.
        target: LoopId,
        /// The value.
        value: Option<Box<Expr>>,
    },
    /// Starts the next iteration of a loop.
    Continue {
        /// The loop continued.
        target: LoopId,
    },
    /// Returns from the function.
    Return(Option<Box<Expr>>),
    /// An operation on quantum state.
    /// Each operand is consumed.
    Quantum {
        /// Which operation.
        op: QuantumOp,
        /// Its operands, in the order evaluated.
        args: Vec<Expr>,
    },
}

/// An operation on quantum state.
#[derive(Clone, Debug)]
pub enum QuantumOp {
    /// `measure e`: consumes a qubit or a register and gives the outcome,
    /// `bool` for a qubit and `[bool; N]` for `[qubit; N]`.
    Measure,
    /// `lift e`: an outcome value made generation-time while the circuit
    /// runs, so that what follows may depend on it.
    Lift,
    /// `forget e;`: discards quantum values without an outcome.
    Forget,
    /// `a ** b` on registers: one register holding `a`'s qubits then `b`'s.
    Tensor,
    /// `replay f(args)`: a second preparation of what a monic operator
    /// prepares, its operand the call.
    Replay,
    /// `prep`: allocates a register and prepares a state in it, by the
    /// circuit analysis made for the state.
    Prep(std::sync::Arc<crate::circuit::synth::Prepared>),
    /// `query t[k]`: the map locale `t`, by position among the unit's, read
    /// coherently: an operator applying each entry controlled on the key
    /// register, its operand a handle to the key.
    Query(usize),
    /// `measure t[k]`: the key measured, giving the index it held and the
    /// entry there; its operand the key register.
    Lookup(usize),
    /// A map locale named as a value.
    Table(usize),
    /// `query t[k]` of a map locale held as a value: its operands the value
    /// and a handle to the key.
    QueryAt,
    /// `measure t[k]` of a map locale held as a value: its operands the
    /// value and the key register.
    LookupAt,
    /// `t ^= c`: the qubit a handle names flipped where the quantum
    /// condition `c` holds. Its operands are the handle and the condition,
    /// which is read and not consumed.
    Flip,
    /// `qcopy(&x)`: new qubits prepared as those `x` holds were, by
    /// repeating what prepared them. Its operand is the handle.
    Copy,
}

impl QuantumOp {
    /// The operation as a program writes it.
    pub fn name(&self) -> &'static str {
        match self {
            QuantumOp::Measure => "measure",
            QuantumOp::Lift => "lift",
            QuantumOp::Forget => "forget",
            QuantumOp::Tensor => "**",
            QuantumOp::Replay => "replay",
            QuantumOp::Prep(_) => "prep",
            QuantumOp::Query(_) | QuantumOp::QueryAt => "query",
            QuantumOp::Lookup(_) | QuantumOp::LookupAt => "measure",
            QuantumOp::Table(_) => "a map locale",
            QuantumOp::Flip => "^=",
            QuantumOp::Copy => "qcopy",
        }
    }
}

/// A unary operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnOp {
    /// Negation, which aborts when the result does not fit.
    Neg,
    /// Logical not on `bool`, bitwise complement on integers.
    Not,
}

impl UnOp {
    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
        }
    }
}

/// A binary operator that evaluates both operands.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`, rounding toward zero.
    Div,
    /// `%`, taking the sign of the dividend.
    Rem,
    /// `<<`
    Shl,
    /// `>>`: arithmetic on signed types, logical on unsigned.
    Shr,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl BinOp {
    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
        }
    }

    /// Whether the result is a `bool` comparing the operands.
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    /// Whether this is a shift, whose right operand may be of any integer
    /// type.
    pub fn is_shift(self) -> bool {
        matches!(self, BinOp::Shl | BinOp::Shr)
    }

    /// Whether this is `&`, `|` or `^`, which also apply to `bool`.
    pub fn is_bitwise(self) -> bool {
        matches!(self, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)
    }
}

/// A short-circuiting operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogicalOp {
    /// `&&`
    And,
    /// `||`
    Or,
}

impl LogicalOp {
    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            LogicalOp::And => "&&",
            LogicalOp::Or => "||",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    fn int(v: i128) -> Expr {
        Expr::constant(Value::Int(v, IntTy::I32), Ty::Int(IntTy::I32), sp())
    }

    #[test]
    fn a_scalar_value_knows_its_type() {
        assert_eq!(Value::Int(3, IntTy::U8).scalar_ty(), Some(Ty::Int(IntTy::U8)));
        assert_eq!(Value::Bool(true).scalar_ty(), Some(Ty::Bool));
        assert_eq!(Value::Char('x').scalar_ty(), Some(Ty::Char));
        assert_eq!(Value::Void.scalar_ty(), Some(Ty::Void));
        assert_eq!(Value::Array(vec![]).scalar_ty(), None);
        assert_eq!(Value::Int(3, IntTy::U8).as_int(), Some(3));
        assert_eq!(Value::Bool(false).as_bool(), Some(false));
        assert_eq!(Value::Void.as_int(), None);
    }

    #[test]
    fn walking_visits_children_before_parents() {
        let mut e = Expr {
            kind: ExprKind::Binary {
                op: BinOp::Add,
                lhs: Box::new(int(1)),
                rhs: Box::new(int(2)),
            },
            ty: Ty::Int(IntTy::I32),
            span: sp(),
        };
        let mut order = Vec::new();
        e.walk_mut(&mut |x| {
            order.push(match &x.kind {
                ExprKind::Const(v) => v.as_int().unwrap().to_string(),
                ExprKind::Binary { op, .. } => op.text().to_owned(),
                _ => "?".to_owned(),
            });
        });
        assert_eq!(order, ["1", "2", "+"]);
    }

    #[test]
    fn walking_a_block_reaches_statements_and_its_value() {
        let mut b = Block {
            stmts: vec![
                Stmt::Let {
                    local: LocalId(0),
                    init: Some(int(1)),
                },
                Stmt::Let {
                    local: LocalId(1),
                    init: None,
                },
                Stmt::Expr(int(2)),
            ],
            value: Some(Box::new(int(3))),
            ty: Ty::Int(IntTy::I32),
            span: sp(),
        };
        let mut seen = 0;
        b.walk_mut(&mut |_| seen += 1);
        assert_eq!(seen, 3);
    }

    #[test]
    fn walking_can_rewrite() {
        let mut e = Expr {
            kind: ExprKind::Unary {
                op: UnOp::Neg,
                operand: Box::new(int(5)),
            },
            ty: Ty::Int(IntTy::I32),
            span: sp(),
        };
        e.walk_mut(&mut |x| {
            if let ExprKind::Const(Value::Int(v, t)) = &x.kind {
                x.kind = ExprKind::Const(Value::Int(v * 10, *t));
            }
        });
        let ExprKind::Unary { operand, .. } = &e.kind else {
            panic!()
        };
        assert!(matches!(operand.kind, ExprKind::Const(Value::Int(50, _))));
    }

    #[test]
    fn places_are_storage_and_values_are_not() {
        let local = Expr {
            kind: ExprKind::Local(LocalId(0)),
            ty: Ty::Bool,
            span: sp(),
        };
        assert!(local.is_place());
        let field_of_place = Expr {
            kind: ExprKind::Field {
                base: Box::new(local.clone()),
                field: 0,
            },
            ty: Ty::Bool,
            span: sp(),
        };
        assert!(field_of_place.is_place());
        let field_of_value = Expr {
            kind: ExprKind::Field {
                base: Box::new(int(1)),
                field: 0,
            },
            ty: Ty::Bool,
            span: sp(),
        };
        assert!(!field_of_value.is_place(), "a field of a temporary is not storage");
        assert!(!int(1).is_place());
    }

    #[test]
    fn operator_classes_are_disjoint_where_they_should_be() {
        for op in [BinOp::Add, BinOp::Shl, BinOp::BitAnd, BinOp::Lt] {
            let classes = [op.is_comparison(), op.is_shift(), op.is_bitwise()];
            assert!(classes.iter().filter(|c| **c).count() <= 1, "{op:?}");
        }
        assert!(BinOp::Ge.is_comparison());
        assert!(BinOp::Shr.is_shift());
        assert!(BinOp::BitXor.is_bitwise());
        assert_eq!(LogicalOp::Or.text(), "||");
        assert_eq!(UnOp::Not.text(), "!");
    }

    #[test]
    fn only_exit_and_panic_never_return() {
        assert!(Intrinsic::Exit.diverges());
        assert!(Intrinsic::Panic.diverges());
        assert!(!Intrinsic::Print.diverges());
        assert_eq!(Intrinsic::Len.name(), "len");
    }

    #[test]
    fn ids_index_their_vectors() {
        assert_eq!(FnId(3).index(), 3);
        assert_eq!(LocalId(0).index(), 0);
        assert_eq!(ExternId(2).index(), 2);
    }
}
