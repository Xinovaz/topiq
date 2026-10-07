//! Collecting what a unit declares, and resolving each declaration when it is
//! first needed.
//!
//! Unit scope is unordered: a function may call one declared further down, and
//! a constant may be defined in terms of one declared later. So the first pass
//! over a unit only *declares* names (every function, object, structure and
//! enumeration gets an id and a place in [`UnitScope`]), without looking at
//! any type, body or initialiser.
//!
//! # Resolving on demand
//!
//! Resolving a declaration (turning its written types into [`Ty`]s) can need
//! the *value* of another: `[u8; SIZE]` needs `SIZE`, whose initialiser must
//! be checked and evaluated first, and `SIZE` may itself be `@sizeof(Header)`,
//! which needs `Header`'s fields. No fixed order of passes serves every
//! program, so each declaration is resolved the first time something asks
//! for it, and remembered. A declaration asked for while it is still being
//! resolved depends on itself, and that is reported rather than looped on.
//!
//! [`UnitCx`] holds the unit being built and the state of every declaration;
//! the methods [`UnitCx::adt`], [`UnitCx::fn_sig`], [`UnitCx::global_ty`],
//! [`UnitCx::global_init`] and [`UnitCx::fn_body`] each make sure one piece is
//! done. [`UnitCx::check_all`] then asks for everything, in an order that
//! makes on-demand resolution the exception.
//!
//! Generic declarations are recorded rather than resolved, and each instance
//! is made when first asked for ([`super::generic`]). A function whose
//! parameters mix `const` and ordinary ones is declared as a generic whose
//! constant parameters are its `const` ones, so each set of their values
//! makes an instance.

use std::collections::{HashMap, HashSet};

use crate::ast::{self, AnnotationGroup, FnItem, ItemKind, Linkage, StdAnnotation};
use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::{Interner, Symbol};
use crate::span::{SourceId, Span, Spanned};
use crate::tir::visit::{self, Visit};
use crate::tir::{
    self, AdtDef, AdtId, Arg, Callee, ExprKind, ExternId, FnId, FnKind, GlobalId, GlobalKind,
    Local, LocalId, Ty, Value,
};

use super::adt;
use super::body;
use super::consteval::{Evaluator, Frame};
use super::fold::{self, Need};
use super::generic;
use super::interface::Interface;
use super::report;
use super::scope::{Def, UnitScope};
use super::types;

/// How far resolving one piece of a declaration has got.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Progress {
    /// Not started.
    Pending,
    /// Under way: asking for it now means it depends on itself.
    Active,
    /// Finished.
    Done,
}

pub(super) struct AdtSrc<'a> {
    pub(super) kind: &'a ItemKind,
    pub(super) annotations: &'a [Spanned<AnnotationGroup>],
    pub(super) state: Progress,
    /// The unit whose scope its field types are resolved in.
    pub(super) frame: usize,
    /// What the generic parameters stand for, for an instance.
    pub(super) subst: generic::Subst,
}

struct FnSrc<'a> {
    item: &'a FnItem,
    sig: Progress,
    body: Progress,
    /// The unit whose scope it is resolved in.
    frame: usize,
}

/// A generic declaration: which unit's, and its name there.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GenericKey {
    /// The frame of the unit that declares it.
    pub frame: usize,
    /// Its name.
    pub name: Symbol,
    /// For a generic method, the type it belongs to.
    pub method: Option<super::method::OwnerKey>,
}

impl GenericKey {
    /// The generic declaration `name` of frame `frame`'s unit.
    pub fn plain(frame: usize, name: Symbol) -> GenericKey {
        GenericKey { frame, name, method: None }
    }
}

/// One unit's declarations, as analysis of this unit sees them.
///
/// Frame 0 is the unit being analysed. Each other frame is a unit it imports
/// whose generic or constant function bodies it needs: those bodies are
/// checked here, but in the scope of the unit that wrote them, where a name
/// means what it means there. That unit's ordinary functions, objects and
/// types come from its interface, as for any import; only its generic items,
/// type aliases and constant functions are taken from its syntax tree.
#[derive(Default)]
struct UnitFrame<'a> {
    /// The unit.
    origin: Option<String>,
    /// Its interface, through which its compiled items are reached.
    iface: Option<&'a Interface>,
    generics: HashMap<(Option<super::method::OwnerKey>, Symbol), GenericSrc<'a>>,
    aliases: HashMap<Symbol, AliasSrc<'a>>,
    /// Its unit scope and imports, kept here while another frame is current;
    /// the current frame's are [`UnitCx::scope`] and [`UnitCx::imported`].
    scope: UnitScope,
    imported: Vec<&'a Interface>,
}

/// A generic declaration, kept as syntax until something asks for an
/// instance of it.
pub(super) struct GenericSrc<'a> {
    /// The parameters it takes. A method of a generic type takes its `impl`
    /// block's parameters first, then its own.
    pub(super) generics: Vec<ast::GenericParam>,
    /// The declaration itself: for a method, the `impl` block or function
    /// item it was written in.
    pub(super) item: &'a ItemKind,
    /// For a method, the method.
    pub(super) method: Option<super::method::GenericMethod<'a>>,
    /// Its annotations, for a structure or enumeration.
    pub(super) annotations: &'a [Spanned<AnnotationGroup>],
    /// Its name as written.
    pub(super) name: Spanned<Symbol>,
    /// Whether importers may name it.
    pub(super) linkage: Linkage,
}

/// A type alias, and how far resolving it has got.
struct AliasSrc<'a> {
    ty: &'a Spanned<ast::Type>,
    name: Spanned<Symbol>,
    /// Whether other units may name it: every one not declared `static`.
    linkage: Linkage,
    pub(super) state: Progress,
    /// What it resolves to, once it has.
    resolved: Option<Ty>,
}

struct GlobalSrc<'a> {
    ty: &'a Spanned<ast::Type>,
    init: Option<&'a Spanned<ast::Expr>>,
    ty_state: Progress,
    init_state: Progress,
}

/// The unit being analysed, and how far each of its declarations has got.
pub struct UnitCx<'a> {
    /// For resolving and printing names.
    pub interner: &'a Interner,
    /// The unit's preprocessor macros.
    pub macros: Option<&'a crate::pp::MacroTable>,
    /// The unit's conductor: the N of the field `i` and `isq2` belong to.
    pub conductor: u32,
    /// How deep in circuit handles' signatures, `circuit<fn(…)>`, a type being
    /// resolved is: there a classical unit may name `qubit`, which a handle's
    /// operator takes, though it holds none.
    pub(super) in_circuit: u32,
    /// Whether the program is built for a target running static circuits
    /// only, which lacks the capabilities `@target_has` asks about.
    pub static_circuits: bool,
    /// Whether this is a quantum unit, which may hold qubits and whose
    /// operators become circuits.
    pub quantum: bool,
    /// The documents `@embed` names.
    pub documents: &'a [super::Document<'a>],
    /// The unit being built.
    pub unit: tir::Unit,
    /// Unit scope.
    pub scope: UnitScope,
    /// Everything reported so far.
    pub diags: Vec<Diagnostic>,
    /// The interfaces of every unit this one may import.
    pub(super) available: &'a [Interface],
    /// The units the current frame's unit imports, indexed by
    /// [`super::scope::UnitRef`].
    pub(super) imported: Vec<&'a Interface>,
    /// Imported functions already given an [`ExternId`], by interface and
    /// name.
    pub(super) extern_ids: HashMap<(usize, Symbol), ExternId>,
    /// Imported objects and constants already given a [`GlobalId`].
    pub(super) imported_globals: HashMap<(usize, Symbol), GlobalId>,
    pub(super) adts: HashMap<AdtId, AdtSrc<'a>>,
    /// Every unit whose declarations are in view; see [`UnitFrame`].
    frames: Vec<UnitFrame<'a>>,
    /// The frame names are being looked up in.
    pub(super) frame: usize,
    /// The frame opened for each imported interface.
    frame_of: HashMap<usize, usize>,
    /// The frame of the `core` library, whose items every unit sees when it
    /// declares nothing of the same name.
    prelude: Option<usize>,
    /// The frames of the quantum library units each unit in view imports,
    /// whose operators it names unqualified: the importing unit's frame,
    /// and the library unit's.
    pub(super) opened: Vec<(usize, usize)>,
    /// Every method and associated function in view, by the type it belongs
    /// to and its name.
    pub(super) methods: HashMap<super::method::MethodKey, super::method::MethodEntry>,
    /// Whether `[typeinfo]` (true) or `[no_typeinfo]` (false) is written on a
    /// type this unit declares.
    pub(super) typeinfo_policy: HashMap<AdtId, bool>,
    /// The annotations written on each type this unit declares: each name,
    /// and its argument where that is a single name.
    pub(super) annotation_names: HashMap<AdtId, Vec<(Symbol, Option<Symbol>)>>,
    /// What `[tcon: …]` says of each type this unit declares that has it.
    pub(super) tcon_forms: HashMap<AdtId, super::typeinfo::TconForm>,
    /// The symbols `[export]` has given, and where.
    pub(super) exported: HashMap<String, Span>,
    /// For each library function used as a value, the function of this
    /// unit that stands for it.
    pub(super) library_values: HashMap<crate::tir::Intrinsic, FnId>,
    /// The functions whose parameters mix `const` and ordinary ones, each a
    /// generic made for each set of `const` values it is called with.
    pub(super) specialized: HashSet<GenericKey>,
    /// The items the blocks around the code being checked declare,
    /// outermost first.
    pub(super) local_items: Vec<super::local_items::Layer>,
    /// Each block that declares items, by its address, and its layer.
    pub(super) block_layers: HashMap<usize, super::local_items::Layer>,
    /// For each function declared in a block, the layers around it.
    pub(super) fn_layers: HashMap<FnId, Vec<super::local_items::Layer>>,
    /// For each type declared in a block, the layers around it.
    pub(super) adt_layers: HashMap<AdtId, Vec<super::local_items::Layer>>,
    /// Each item a block declares with annotations, and the layers around it,
    /// whose claims are read with the unit's.
    pub(super) annotated_items: Vec<(&'a ast::Item, Vec<super::local_items::Layer>)>,
    /// How many deprecated items are being checked at once: a use inside one
    /// is not warned about.
    pub(super) quiet_deprecation: u32,
    /// The uses of deprecated items already warned of.
    pub(super) warned_uses: super::attrs::Warned,
    /// The types marked `[deprecated]`, of this unit or an imported one, by
    /// the unit declaring them (`None` for this one) and name.
    pub(super) deprecated_types: HashMap<(Option<String>, Symbol), String>,
    /// Instances already made, so that one set of arguments makes one.
    fn_instances: HashMap<(GenericKey, Vec<Arg>), FnId>,
    adt_instances: HashMap<(GenericKey, Vec<Arg>), AdtId>,
    /// What the parameters of the instance being checked stand for. Empty
    /// outside a generic body.
    pub(super) subst: Vec<generic::Subst>,
    /// How many instances are being made at once, against the limit.
    pub(super) generic_depth: u32,
    fns: Vec<Option<FnSrc<'a>>>,
    globals: HashMap<GlobalId, GlobalSrc<'a>>,
    /// Types already reported as containing themselves.
    cycles: HashSet<AdtId>,
    /// The covers, gauges, base types, locales and chains the unit declares,
    /// by name.
    pub(super) geometry_decls: HashMap<Symbol, super::geometry::Decl<'a>>,
    /// Covers, gauges and base types of other units already made this
    /// unit's, by the declaring unit and name, with their positions in their
    /// lists.
    pub(super) imported_geometry: HashMap<(String, Symbol), usize>,
    /// The map locales the unit declares.
    pub(super) qmap_decls: super::qmap::QmapDecls<'a>,
    /// The `@static_assert`s written at unit scope.
    pub(super) asserts: Vec<&'a Spanned<ast::Expr>>,
}

impl<'a> UnitCx<'a> {
    /// A context for the unit `name`.
    pub fn new(name: &str, source: SourceId, interner: &'a Interner, available: &'a [Interface]) -> UnitCx<'a> {
        UnitCx {
            interner,
            macros: None,
            conductor: 8,
            in_circuit: 0,
            static_circuits: false,
            quantum: false,
            documents: &[],
            unit: tir::Unit::new(name, source),
            scope: UnitScope::new(),
            diags: Vec::new(),
            available,
            imported: Vec::new(),
            extern_ids: HashMap::new(),
            imported_globals: HashMap::new(),
            adts: HashMap::new(),
            frames: vec![UnitFrame::default()],
            frame: 0,
            frame_of: HashMap::new(),
            prelude: None,
            opened: Vec::new(),
            methods: HashMap::new(),
            typeinfo_policy: HashMap::new(),
            annotation_names: HashMap::new(),
            tcon_forms: HashMap::new(),
            exported: HashMap::new(),
            library_values: HashMap::new(),
            specialized: HashSet::new(),
            local_items: Vec::new(),
            block_layers: HashMap::new(),
            fn_layers: HashMap::new(),
            adt_layers: HashMap::new(),
            annotated_items: Vec::new(),
            quiet_deprecation: 0,
            warned_uses: HashSet::new(),
            deprecated_types: available
                .iter()
                .flat_map(|i| i.deprecated_types.iter().map(|(n, m)| ((Some(i.unit.clone()), *n), m.clone())))
                .collect(),
            fn_instances: HashMap::new(),
            adt_instances: HashMap::new(),
            subst: Vec::new(),
            generic_depth: 0,
            fns: Vec::new(),
            globals: HashMap::new(),
            cycles: HashSet::new(),
            geometry_decls: HashMap::new(),
            imported_geometry: HashMap::new(),
            qmap_decls: HashMap::new(),
            asserts: Vec::new(),
        }
    }

    /// Reports a problem.
    pub fn report(&mut self, d: Diagnostic) {
        self.diags.push(d);
    }

    fn name(&self, s: Symbol) -> &'a str {
        self.interner.resolve(s)
    }

    /// Declares every item of the unit. `has_initializer` is whether the unit
    /// directive names a unit initialiser, which could give an object without
    /// an initial value its value.
    pub fn declare(&mut self, unit: &'a ast::Unit, has_initializer: bool) {
        // every unit but `core` itself sees `core`
        let available = self.available;
        if self.unit.name != "core"
            && let Some(core) = available.iter().find(|i| i.unit == "core")
        {
            self.prelude = self.frame_for(core);
            if let Some(sym) = self.interner.get("core") {
                self.imported.push(core);
                let _ = self.scope.declare_unit(sym, super::scope::UnitRef(self.imported.len() as u32 - 1), Span::synthetic());
            }
        }
        for item in &unit.items {
            self.declare_item(item, has_initializer);
        }
        let this = self.unit.name.clone();
        self.declare_library_units(&this);
        // a method is declared once every type it could belong to is
        for item in &unit.items {
            self.declare_methods(item);
        }
    }

    /// Resolves the operator each `[adjoint: f]` names: an operator of this
    /// unit with the same signature as the one it undoes.
    pub fn resolve_adjoints(&mut self) {
        for i in 0..self.unit.fns.len() {
            let f = &self.unit.fns[i];
            let Some((name, at)) = f.attrs.adjoint_name else { continue };
            if f.origin.is_some() {
                continue;
            }
            let sig: (Vec<Ty>, Ty) = (f.param_types().collect(), f.ret);
            let own = self.name(f.name).to_owned();
            let text = self.name(name).to_owned();
            let Some(Def::Fn(g)) = self.scope.get(name) else {
                let d = Diagnostic::new(Code::Es04)
                    .with_message(format!("there is no operator named `{text}` for `[adjoint: {text}]` to name"))
                    .at(at)
                    .with_help("name an operator of this unit");
                self.report(d);
                continue;
            };
            self.fn_sig(g);
            let other = self.unit.func(g);
            if (other.param_types().collect::<Vec<_>>(), other.ret) != sig {
                let d = Diagnostic::new(Code::Es06)
                    .with_message(format!("`{text}` cannot be the adjoint of `{own}`: their signatures differ"))
                    .at(at)
                    .also(other.span, format!("`{text}` is declared here"))
                    .with_note("an adjoint undoes an operator in place, so it takes what the operator takes and gives what it gives");
                self.report(d);
                continue;
            }
            self.unit.fns[i].attrs.adjoint = Some(g);
        }
    }

    /// Records, once a quantum unit is analysed, what a classical unit
    /// importing it may use: its structures and enumerations that hold no
    /// qubits, its constants, and its `[entry]` operators, which a classical
    /// unit holds as circuit handles. Everything else of program linkage is
    /// recorded by name as quantum-only, which a classical importer naming it
    /// is told.
    ///
    /// A generic type is quantum-only too: whether an instance holds qubits
    /// depends on its arguments, which the importer would choose. An entry
    /// operator is usable whatever its signature: a run gives each quantum
    /// parameter a register (`qpu::RegHandle`), and what it returns crosses
    /// as its classical part.
    pub fn collect_quantum_interface(&mut self, unit: &'a ast::Unit) {
        self.unit.quantum = true;
        for item in &unit.items {
            // each name a classical unit may not use is quantum-only
            let (name, usable) = match &item.node.kind {
                ItemKind::Struct { name, generics, .. } | ItemKind::Enum { name, generics, .. } => {
                    let classical = generics.is_empty()
                        && self.scope.get_type(name.node).is_some_and(|id| !self.unit.types.is_quantum(Ty::Adt(id)));
                    (name.node, classical)
                }
                ItemKind::Let { name, .. } => {
                    let constant = matches!(self.scope.get(name.node), Some(Def::Global(id)) if self.unit.global(id).constant);
                    (name.node, constant)
                }
                ItemKind::Fn(f) => {
                    let ast::FnName::Plain(name) = f.name else { continue };
                    match self.scope.get(name.node) {
                        Some(Def::Fn(id)) if self.unit.func(id).attrs.entry => {
                            self.unit.entries.push(id);
                            continue;
                        }
                        _ => (name.node, false),
                    }
                }
                _ => continue,
            };
            if !usable {
                self.unit.quantum_only.push(name);
            }
        }
    }

    /// Finds the unit initialiser the directive names: a function of this
    /// unit taking nothing and returning `void`, which runs once before
    /// `main`. Anything else is reported.
    pub fn resolve_initializer(&mut self, name: Symbol, directive: Span) {
        let text = self.name(name).to_owned();
        let found = match self.scope.get(name) {
            Some(Def::Fn(id)) => {
                self.fn_sig(id);
                let f = self.unit.func(id);
                let shape_ok = f.param_types().next().is_none() && f.ret == Ty::Void && f.kind == tir::FnKind::Runtime;
                if shape_ok { Ok(id) } else { Err(Some(f.span)) }
            }
            _ => Err(None),
        };
        match found {
            Ok(id) => self.unit.init = Some(id),
            Err(decl) => {
                let mut d = Diagnostic::new(Code::Eu05)
                    .with_message(match decl {
                        Some(_) => format!("the unit initialiser `{text}` must take nothing and return nothing"),
                        None => format!("the unit initialiser `{text}` is not a function of this unit"),
                    })
                    .at(directive)
                    .with_note(
                        "the initialiser runs once before `main`, with nothing to give it and \
                         nowhere to return to, so it is declared `fn name()`",
                    )
                    .with_help(format!("declare `fn {text}() {{ … }}` in this unit"));
                if let Some(decl) = decl {
                    d = d.also(decl, "declared here");
                }
                self.report(d);
            }
        }
    }

    fn declare_item(&mut self, item: &'a Spanned<ast::Item>, has_initializer: bool) {
        let span = item.span;
        let it = &item.node;
        match &it.kind {
            ItemKind::Fn(f) if !matches!(f.name, ast::FnName::Plain(_)) => {
                check_annotations(&it.annotations, Target::Function, self.quantum, self.interner, &mut self.diags);
            }
            ItemKind::Fn(f) => {
                check_annotations(&it.annotations, Target::Function, self.quantum, self.interner, &mut self.diags);
                let generic = !f.generics.is_empty();
                let attrs = self.item_attrs(&it.annotations, super::attrs::Subject::function(it.linkage, generic));
                self.declare_block_items(&f.body);
                if self.declare_specialized(f, &it.kind, &it.annotations, it.linkage) {
                    return;
                }
                if generic
                    && let ast::FnName::Plain(name) = &f.name {
                        self.declare_generic(&f.generics, &it.kind, &it.annotations, *name, it.linkage);
                        return;
                    }
                let id = self.declare_fn(f, it.linkage, span);
                self.unit.fns[id.index()].attrs = attrs;
            }
            ItemKind::Let { name, ty, init } => {
                check_annotations(&it.annotations, Target::Object, self.quantum, self.interner, &mut self.diags);
                let constant = ty.node.is_const();
                let attrs = self.item_attrs(
                    &it.annotations,
                    super::attrs::Subject {
                        target: Target::Object,
                        linkage: it.linkage,
                        generic: false,
                        method: false,
                        constant,
                    },
                );
                let id = GlobalId(self.unit.globals.len() as u32);
                self.unit.globals.push(tir::Global {
                    name: name.node,
                    ty: Ty::Never,
                    constant: false,
                    linkage: it.linkage,
                    kind: GlobalKind::Unit,
                    init: None,
                    value: None,
                    startup: init.is_none(),
                    span: name.span,
                    symbol: attrs.symbol,
                    deprecated: attrs.deprecated,
                });
                self.globals.insert(
                    id,
                    GlobalSrc {
                        ty,
                        init: init.as_ref(),
                        ty_state: Progress::Pending,
                        init_state: Progress::Pending,
                    },
                );
                self.declare_value(name, Def::Global(id));
                if let Some(e) = init {
                    self.declare_expr_items(e);
                }
                if init.is_none() && !has_initializer {
                    self.report(no_initializer(name.span, self.name(name.node)));
                }
            }
            ItemKind::Struct { name, generics, .. } | ItemKind::Enum { name, generics, .. } => {
                let is_struct = matches!(it.kind, ItemKind::Struct { .. });
                let subject = super::attrs::Subject {
                    target: if is_struct { Target::Structure } else { Target::Enumeration },
                    linkage: it.linkage,
                    generic: false,
                    method: false,
                    constant: false,
                };
                if let Some(msg) = self.item_attrs(&it.annotations, subject).deprecated {
                    self.unit.deprecated_types.push((name.node, msg.clone()));
                    self.deprecated_types.insert((None, name.node), msg);
                }
                if !generics.is_empty() {
                    self.declare_generic(generics, &it.kind, &it.annotations, *name, it.linkage);
                    return;
                }
                let def = AdtDef::unresolved(name.node, None, it.linkage, Vec::new(), is_struct, name.span);
                let id = self.add_adt(def, &it.kind, &it.annotations, self.frame, Vec::new());
                self.check_reserved(name.node, name.span);
                if let Err(t) = self.scope.declare_type(name.node, id, name.span) {
                    let d = report::duplicate(name.span, t.earlier, self.name(name.node), "at unit scope");
                    self.report(d);
                }
            }
            ItemKind::Import { path, alias, kind } => {
                if it.linkage == Linkage::Unit {
                    self.report(static_without_linkage(span, "an `import`"));
                }
                super::imports::declare(self, path, alias.as_ref(), kind.as_ref(), span);
            }
            ItemKind::TypeAlias { name, generics, ty } => {
                if !generics.is_empty() {
                    self.declare_generic(generics, &it.kind, &it.annotations, *name, it.linkage);
                    return;
                }
                self.check_reserved(name.node, name.span);
                let aliases = &mut self.frames[self.frame].aliases;
                if let Some(earlier) = aliases.get(&name.node) {
                    let d = report::duplicate(name.span, earlier.name.span, self.name(name.node), "at unit scope");
                    self.report(d);
                    return;
                }
                aliases.insert(
                    name.node,
                    AliasSrc {
                        ty,
                        name: *name,
                        linkage: it.linkage,
                        state: Progress::Pending,
                        resolved: None,
                    },
                );
            }
            ItemKind::Impl { .. } => {}
            ItemKind::Assert { call } => self.asserts.push(call),
            ItemKind::Qmap { name, .. } => self.declare_qmap(&it.kind, *name, span),
            ItemKind::Cover { name, .. }
            | ItemKind::Gauge { name, .. }
            | ItemKind::Base { name, .. }
            | ItemKind::Locale { name, .. }
            | ItemKind::Chain { name, .. } => self.declare_geometry(&it.kind, *name, it.linkage, span),
        }
    }

    /// Reports a name a program may not declare: one beginning with `__`,
    /// which the library reserves for its own use.
    pub(super) fn check_reserved(&mut self, name: Symbol, span: Span) {
        let text = self.name(name);
        if self.frame != 0 || !crate::library::is_reserved_identifier(text) || crate::library::source(&self.unit.name).is_some() {
            return;
        }
        self.report(
            Diagnostic::new(Code::Es20)
                .with_message(format!("`{text}` begins with `__`, which is reserved for the library"))
                .at(span)
                .with_note("names beginning with `__` are what the library is built from, so no program may declare one")
                .with_help("choose a name that does not begin with two underscores"),
        );
    }

    fn declare_value(&mut self, name: &Spanned<Symbol>, def: Def) {
        self.check_reserved(name.node, name.span);
        if let Err(t) = self.scope.declare(name.node, def, name.span) {
            let d = report::duplicate(name.span, t.earlier, self.name(name.node), "at unit scope");
            self.report(d);
        }
    }

    /// Keeps a generic declaration as syntax. Nothing of it is resolved until
    /// something asks for an instance, since what its body means depends on
    /// the arguments.
    fn declare_generic(
        &mut self,
        generics: &'a [ast::GenericParam],
        item: &'a ItemKind,
        annotations: &'a [Spanned<AnnotationGroup>],
        name: Spanned<Symbol>,
        linkage: Linkage,
    ) {
        let src = GenericSrc {
            generics: generics.to_vec(),
            item,
            method: None,
            annotations,
            name,
            linkage,
        };
        self.add_generic(None, src);
    }

    /// Declares `f`, whose parameters mix `constexpr` ones with ordinary
    /// ones, as what it is: a function specialised for each set of values of
    /// its `constexpr` parameters, each value known during translation where
    /// it is called. The `constexpr` parameters become constant generic
    /// parameters, so each specialisation is an instance. Returns whether `f` is such a
    /// function.
    fn declare_specialized(
        &mut self,
        f: &'a FnItem,
        item: &'a ItemKind,
        annotations: &'a [Spanned<AnnotationGroup>],
        linkage: Linkage,
    ) -> bool {
        // a plain function with both kinds of parameter, and no generics of its own
        let ast::FnName::Plain(name) = f.name else { return false };
        let is_const = |p: &ast::Param| p.ty.node.is_constexpr();
        let ordinary = f.params.iter().filter(|p| !p.is_self).any(|p| !is_const(p));
        if !ordinary || !f.params.iter().any(is_const) {
            return false;
        }
        if !f.generics.is_empty() {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "the generic function `{}` has `constexpr` parameters beside ordinary ones",
                        self.name(name.node)
                    ))
                    .at(name.span)
                    .with_note(
                        "a function mixing `constexpr` and ordinary parameters is made once for each value \
                         of its `constexpr` ones; a generic one would also be made for each type, and the \
                         two are not combined",
                    )
                    .with_help("make the `constexpr` parameters generic parameters, as in `fn f<N: const usize>(…)`"),
            );
            return true;
        }

        // each integer `constexpr` parameter becomes a constant generic parameter
        let mut generics = f.generics.clone();
        for p in f.params.iter().filter(|p| is_const(p)) {
            let mut inner = &p.ty;
            while let ast::Type::Const(i) | ast::Type::Constexpr(i) = &inner.node {
                inner = i;
            }
            let integer = matches!(&inner.node, ast::Type::Path { path, args }
                if args.is_empty() && path.segments.len() == 1
                    && crate::tir::IntTy::from_name(self.interner.resolve(path.segments[0].node)).is_some());
            if !integer {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "`{}` is `constexpr` beside ordinary parameters, so it must be an integer",
                            self.name(p.name.node)
                        ))
                        .at(p.ty.span)
                        .with_note(
                            "a function whose parameters mix `constexpr` and ordinary ones is made once for \
                             each value its `constexpr` ones are called with, and those values are integers",
                        )
                        .with_help("make every parameter `constexpr`, making a constant function, or none"),
                );
                return true;
            }
            generics.push(ast::GenericParam {
                name: p.name,
                ty: Some(inner.clone()),
            });
        }

        // declared as a generic, marked as specialised
        let src = GenericSrc {
            generics,
            item,
            method: None,
            annotations,
            name,
            linkage,
        };
        if let Some(key) = self.add_generic(None, src) {
            self.specialized.insert(key);
        }
        true
    }

    /// Adds the structure or enumeration `def`, declared by `kind` in frame
    /// `frame`, its fields resolved when first needed.
    pub(super) fn add_adt(
        &mut self,
        def: AdtDef,
        kind: &'a ItemKind,
        annotations: &'a [Spanned<AnnotationGroup>],
        frame: usize,
        subst: generic::Subst,
    ) -> AdtId {
        let id = self.unit.types.add_adt(def);
        let src = AdtSrc {
            kind,
            annotations,
            state: Progress::Pending,
            frame,
            subst,
        };
        self.adts.insert(id, src);
        id
    }

    /// Records a generic declaration, or reports that its name is taken.
    pub(super) fn add_generic(&mut self, method: Option<super::method::OwnerKey>, src: GenericSrc<'a>) -> Option<GenericKey> {
        let name = src.name;
        self.check_reserved(name.node, name.span);
        if let Some(earlier) = self.frames[self.frame].generics.get(&(method, name.node)) {
            let d = report::duplicate(name.span, earlier.name.span, self.name(name.node), "at unit scope");
            self.report(d);
            return None;
        }
        if self.frame == 0 && (matches!(src.item, ItemKind::Fn(_)) || src.method.is_some()) {
            self.unit.shares_bodies = true;
        }
        self.frames[self.frame].generics.insert((method, name.node), src);
        Some(GenericKey {
            frame: self.frame,
            name: name.node,
            method,
        })
    }

    /// Makes `to` the frame names are looked up in, returning the one that
    /// was, to go back to.
    pub(super) fn enter(&mut self, to: usize) -> usize {
        let from = self.frame;
        if to != from {
            std::mem::swap(&mut self.scope, &mut self.frames[from].scope);
            std::mem::swap(&mut self.imported, &mut self.frames[from].imported);
            std::mem::swap(&mut self.scope, &mut self.frames[to].scope);
            std::mem::swap(&mut self.imported, &mut self.frames[to].imported);
            self.frame = to;
        }
        from
    }

    /// Runs `f` in the scope of frame `frame` with the generic parameters
    /// bound as `subst` says (none, for a declaration that is not an
    /// instance), so that an instance being checked when something else is
    /// resolved on demand lends it nothing.
    fn within<T>(&mut self, frame: usize, subst: generic::Subst, f: impl FnOnce(&mut Self) -> T) -> T {
        let back = self.enter(frame);
        self.subst.push(subst);
        // what a block declares is seen only by what that block holds
        let layers = std::mem::take(&mut self.local_items);
        let out = f(self);
        self.local_items = layers;
        self.subst.pop();
        self.enter(back);
        out
    }

    /// The frame for an imported unit whose bodies are needed, opened the
    /// first time: every item of that unit is declared in it, as that unit
    /// declared it. `None` if the interface carries no source.
    pub(super) fn frame_for(&mut self, iface: &'a Interface) -> Option<usize> {
        let at = std::ptr::from_ref(iface) as usize;
        if let Some(&f) = self.frame_of.get(&at) {
            return Some(f);
        }
        let ast = iface.ast.as_deref()?;
        // a library unit's source is the compiler's, and changes only with
        // it, which the metadata's version already records
        if let Some(src) = &iface.source
            && crate::library::source(&iface.unit).is_none()
        {
            self.unit.borrowed.push((iface.unit.clone(), src.digest()));
        }
        let id = self.frames.len();
        self.frames.push(UnitFrame {
            origin: Some(iface.unit.clone()),
            iface: Some(iface),
            ..UnitFrame::default()
        });
        self.frame_of.insert(at, id);
        self.within(id, Vec::new(), |cx| {
            for item in &ast.items {
                cx.declare_foreign(iface, item);
            }
            cx.declare_library_units(&iface.unit);
        });
        Some(id)
    }

    /// Makes every library unit at hand, other than the unit `this` itself,
    /// a unit name of the current scope: a library unit is named as
    /// `unit::item` without being imported. One the scope already imports is
    /// left as it is, as `core` is in the unit's own scope. Another unit's
    /// frame gets `core` here, so that `core::item` in a body checked there,
    /// such as a generic's, names the library's item.
    fn declare_library_units(&mut self, this: &str) {
        for lib in self.available {
            if lib.unit == this
                || crate::library::source(&lib.unit).is_none()
                || self.imported.iter().any(|i| i.unit == lib.unit)
            {
                continue;
            }
            if let Some(sym) = self.interner.get(&lib.unit) {
                self.imported.push(lib);
                let _ = self.scope.declare_unit(sym, super::scope::UnitRef(self.imported.len() as u32 - 1), Span::synthetic());
            }
        }
    }

    /// Declares one item of another unit in its frame. That unit was checked
    /// when it was compiled, so nothing here is reported.
    fn declare_foreign(&mut self, iface: &'a Interface, item: &'a Spanned<ast::Item>) {
        let it = &item.node;
        match &it.kind {
            ItemKind::Fn(f) => {
                let ast::FnName::Plain(name) = &f.name else {
                    self.declare_foreign_methods(iface, item);
                    return;
                };
                // its other functions are its compiled ones, reached through
                // its interface when something names them
                if self.declare_specialized(f, &it.kind, &it.annotations, it.linkage) {
                    self.declare_block_items(&f.body);
                } else if !f.generics.is_empty() {
                    self.declare_generic(&f.generics, &it.kind, &it.annotations, *name, it.linkage);
                    self.declare_block_items(&f.body);
                } else if iface.any_function(name.node).is_some_and(|x| x.constant) || (self.quantum && iface.quantum) {
                    self.declare_block_items(&f.body);
                    // its body is needed to evaluate it
                    let id = self.declare_fn(f, it.linkage, item.span);
                    self.unit.fns[id.index()].origin = Some(iface.unit.clone());
                    self.unit.fns[id.index()].attrs = super::attrs::quiet(&it.annotations, self.interner);
                }
            }
            ItemKind::Struct { name, generics, .. } | ItemKind::Enum { name, generics, .. } if !generics.is_empty() => {
                self.declare_generic(generics, &it.kind, &it.annotations, *name, it.linkage);
            }
            ItemKind::Import { path, alias, kind } => {
                super::imports::declare(self, path, alias.as_ref(), kind.as_ref(), item.span);
            }
            ItemKind::Impl { .. } => self.declare_foreign_methods(iface, item),
            ItemKind::TypeAlias { name, generics, .. } if !generics.is_empty() => {
                self.declare_generic(generics, &it.kind, &it.annotations, *name, it.linkage);
            }
            ItemKind::TypeAlias { name, ty, .. } => {
                self.frames[self.frame].aliases.insert(
                    name.node,
                    AliasSrc {
                        ty,
                        name: *name,
                        linkage: it.linkage,
                        state: Progress::Pending,
                        resolved: None,
                    },
                );
            }
            _ => {}
        }
    }

    /// The structure or enumeration `name` names here: one declared in the
    /// current frame's unit, and otherwise one the `core` library declares.
    pub fn lookup_type(&mut self, name: Symbol) -> Option<AdtId> {
        if let Some(id) = self.local_type(name) {
            return Some(id);
        }
        let frame = self.frame;
        self.frame_type(frame, name, true).or_else(|| {
            let p = self.prelude.filter(|&p| p != frame)?;
            self.frame_type(p, name, false)
        })
    }

    /// Whether `name` is a type alias the current frame's unit declares,
    /// which hides any type of that name the `core` library declares.
    pub fn declares_alias(&self, name: Symbol) -> bool {
        self.frames[self.frame].aliases.contains_key(&name)
    }

    /// What `name` names in the value name space at unit scope here: an item
    /// of the current frame's unit, and otherwise one of the `core` library.
    pub fn lookup_value(&mut self, name: Symbol) -> Option<Def> {
        if let Some(d) = self.local_value(name) {
            return Some(d);
        }
        let frame = self.frame;
        self.frame_value_lazy(frame, name, true)
            .or_else(|| {
                let p = self.prelude.filter(|&p| p != frame)?;
                self.frame_value_lazy(p, name, false)
            })
            .or_else(|| {
                // a unit's code sees what that unit opened
                let opened: Vec<usize> = self.opened.iter().filter(|(by, _)| *by == frame).map(|(_, p)| *p).collect();
                opened.into_iter().find_map(|p| self.frame_value_lazy(p, name, false))
            })
    }

    /// The structure or enumeration `name` of the unit being analysed, named
    /// from another unit's declarations open here.
    pub(super) fn own_type(&mut self, name: Symbol) -> Option<AdtId> {
        self.frame_type(0, name, true)
    }

    /// The function, object or constant `name` of the unit being analysed,
    /// named from another unit's declarations open here.
    pub(super) fn own_value(&mut self, name: Symbol) -> Option<Def> {
        self.frame_value_lazy(0, name, true)
    }

    /// Frame `frame`'s unit scope.
    fn scope_of(&mut self, frame: usize) -> &mut UnitScope {
        if frame == self.frame { &mut self.scope } else { &mut self.frames[frame].scope }
    }

    /// The type `name` in frame `frame`'s unit scope, bringing another unit's
    /// in through its interface the first time it is named. `any` admits a
    /// `static` one, which only that unit's own code may name.
    fn frame_type(&mut self, frame: usize, name: Symbol, any: bool) -> Option<AdtId> {
        if let Some(id) = self.scope_of(frame).get_type(name) {
            return Some(id);
        }
        let iface = self.frames[frame].iface?;
        let found = if any { iface.any_type(name) } else { iface.named_type(name) };
        found?;
        let id = super::imports::foreign_type(self, iface, name)?;
        let _ = self.scope_of(frame).declare_type(name, id, Span::synthetic());
        Some(id)
    }

    /// The value `name` in frame `frame`'s unit scope, as [`Self::frame_type`]
    /// finds types.
    fn frame_value_lazy(&mut self, frame: usize, name: Symbol, any: bool) -> Option<Def> {
        if let Some(d) = self.scope_of(frame).get(name) {
            return Some(d);
        }
        let iface = self.frames[frame].iface?;
        let visible = if any {
            iface.any_function(name).is_some() || iface.any_global(name).is_some()
        } else {
            iface.function(name).is_some() || iface.global(name).is_some()
        };
        if !visible {
            return None;
        }
        let def = super::imports::foreign_value(self, iface, name, Span::synthetic())?;
        let _ = self.scope_of(frame).declare(name, def, Span::synthetic());
        Some(def)
    }

    /// The frame of the `core` library; `core` is its own.
    fn core_frame(&self) -> Option<usize> {
        self.prelude.or((self.unit.name == "core").then_some(0))
    }

    /// The generic enumeration of the `core` library that has a variant
    /// named `name` (`Some` and `None` of `Opt`, `Ok` and `Err` of
    /// `Result`), which every unit may write unqualified.
    pub fn prelude_variant(&self, name: Symbol) -> Option<GenericKey> {
        let p = self.core_frame()?;
        let text = self.name(name);
        if !matches!(text, "Some" | "None" | "Ok" | "Err") {
            return None;
        }
        self.frames[p].generics.iter().find_map(|((m, n), src)| {
            (m.is_none() && matches!(src.item, ItemKind::Enum { variants, .. } if variants.iter().any(|v| v.name.node == name)))
                .then_some(GenericKey::plain(p, *n))
        })
    }

    /// The unit whose declarations are being looked at, `None` for this one.
    pub(super) fn frame_origin(&self) -> Option<String> {
        self.frames[self.frame].origin.clone()
    }

    /// What `name` denotes in the value name space of frame `frame`'s unit
    /// scope.
    pub(super) fn frame_value(&self, frame: usize, name: Symbol) -> Option<Def> {
        let scope = if frame == self.frame { &self.scope } else { &self.frames[frame].scope };
        scope.get(name)
    }

    pub(super) fn generic_src(&self, key: GenericKey) -> Option<&GenericSrc<'a>> {
        self.frames.get(key.frame)?.generics.get(&(key.method, key.name))
    }

    /// The generic declaration `name` of the current frame's unit, if there
    /// is one.
    pub fn generic(&self, name: Symbol) -> Option<GenericKey> {
        let key = GenericKey::plain(self.frame, name);
        if self.generic_src(key).is_some() {
            return Some(key);
        }
        // the `core` library's, which every unit sees, unless the unit
        // declares something of its own by that name, which hides it
        if self.scope.get(name).is_some() || self.scope.get_type(name).is_some() || self.scope.get_unit(name).is_some() {
            return None;
        }
        let visible = |cx: &Self, frame: usize| {
            let key = GenericKey::plain(frame, name);
            cx.generic_src(key).is_some_and(|s| s.linkage == Linkage::Program).then_some(key)
        };
        if let Some(key) = self.prelude.filter(|&p| p != self.frame).and_then(|p| visible(self, p)) {
            return Some(key);
        }
        // the quantum library units the current frame's unit opened
        self.opened.iter().filter(|(by, _)| *by == self.frame).find_map(|&(_, p)| visible(self, p))
    }

    /// The structure or enumeration `name` the `core` library declares.
    pub fn core_type(&mut self, name: Symbol) -> Option<AdtId> {
        let p = self.core_frame()?;
        self.frame_type(p, name, false)
    }

    /// The generic declaration `name` of the `core` library, `static` or not:
    /// what the compiler itself calls to carry out an operation.
    pub fn core_generic(&self, name: Symbol) -> Option<GenericKey> {
        let key = GenericKey::plain(self.prelude.unwrap_or(self.frame), name);
        self.generic_src(key).is_some().then_some(key)
    }

    /// The generic declaration `name` of the imported unit `u`, if it has one
    /// importers may name.
    pub fn imported_generic(&mut self, u: super::scope::UnitRef, name: Symbol) -> Option<GenericKey> {
        if u == super::scope::UnitRef::SELF {
            let key = GenericKey::plain(0, name);
            return self.generic_src(key).is_some().then_some(key);
        }
        let iface = self.imported[u.0 as usize];
        let key = GenericKey::plain(self.frame_for(iface)?, name);
        (self.generic_src(key)?.linkage == Linkage::Program).then_some(key)
    }

    /// The generic declaration a path starts with (`name` of the current
    /// frame's unit, or `unit::name` of an imported one), and how many of the
    /// path's segments name it.
    pub fn generic_at(&mut self, segments: &[Spanned<Symbol>]) -> Option<(GenericKey, usize)> {
        let first = segments.first()?;
        if let Some(key) = self.generic(first.node) {
            return Some((key, 1));
        }
        let second = segments.get(1)?;
        let u = self.scope.get_unit(first.node)?;
        self.imported_generic(u, second.node).map(|key| (key, 2))
    }

    /// A generic declaration's name as a program would write it here:
    /// `sum`, or `geometry::sum` for another unit's.
    pub fn generic_label(&self, key: GenericKey) -> String {
        let name = self.name(key.name);
        match &self.frames[key.frame].origin {
            Some(u) => format!("{u}::{name}"),
            None => name.to_owned(),
        }
    }

    /// The unit that declared a generic.
    pub fn generic_origin(&self, key: GenericKey) -> Option<String> {
        self.frames[key.frame].origin.clone()
    }

    /// The generic parameters of a generic declaration.
    pub fn generic_params(&self, key: GenericKey) -> Option<&[ast::GenericParam]> {
        self.generic_src(key).map(|g| g.generics.as_slice())
    }

    /// The fields a generic structure declares.
    pub fn generic_struct_fields(&self, key: GenericKey) -> Option<&'a [ast::Field]> {
        match self.generic_src(key)?.item {
            ItemKind::Struct { fields, .. } => Some(fields),
            _ => None,
        }
    }

    /// The variants of a generic enumeration.
    pub fn generic_enum_decl(&self, key: GenericKey) -> Option<&'a [ast::Variant]> {
        match self.generic_src(key)?.item {
            ItemKind::Enum { variants, .. } => Some(variants),
            _ => None,
        }
    }

    /// A variant of a generic enumeration.
    pub fn generic_enum_variant(&self, key: GenericKey, variant: Symbol) -> Option<&'a ast::Variant> {
        self.generic_enum_decl(key)?.iter().find(|v| v.name.node == variant)
    }

    /// The instance of the generic declaration `key` that `ty` is, if it is
    /// one. A pattern names a generic type without its arguments; they are
    /// the ones the value being matched has.
    pub fn instance_of(&self, key: GenericKey, ty: Ty) -> Option<AdtId> {
        let Ty::Adt(id) = ty else { return None };
        let def = self.unit.types.adt(id);
        (def.name == key.name && def.origin == self.frames[key.frame].origin && !def.args.is_empty())
            .then_some(id)
    }

    /// The declaration of a generic function.
    pub fn generic_fn_item(&self, key: GenericKey) -> Option<&'a FnItem> {
        let src = self.generic_src(key)?;
        if let Some(m) = &src.method {
            return Some(m.item);
        }
        match src.item {
            ItemKind::Fn(f) => Some(f),
            _ => None,
        }
    }

    /// Whether a generic declaration is a *function*, rather than a type.
    pub fn is_generic_fn(&self, key: GenericKey) -> bool {
        self.generic_fn_item(key).is_some()
    }

    /// The type a generic type alias stands for with `args`, if `key` is
    /// one: what it was written as, with the parameters bound. `Never` when
    /// that was reported as impossible.
    pub fn instantiate_alias(&mut self, key: GenericKey, args: Vec<Arg>, span: Span) -> Option<Ty> {
        let src = self.generic_src(key)?;
        let ItemKind::TypeAlias { ty, .. } = src.item else {
            return None;
        };
        let params = src.generics.clone();
        if self.too_deep(span) {
            return Some(Ty::Never);
        }
        let subst = self.substitution(&params, &args);
        self.generic_depth += 1;
        let resolved = self.within(key.frame, subst, |cx| cx.resolve_or_report(ty));
        self.generic_depth -= 1;
        Some(resolved.ty)
    }

    /// The instance of a generic structure or enumeration made with `args`,
    /// making it if this is the first time it is asked for.
    ///
    /// Returns `None` when `key` is not a generic type, or when making the
    /// instance was reported as impossible.
    pub fn instantiate_adt(&mut self, key: GenericKey, args: Vec<Arg>, span: Span) -> Option<AdtId> {
        if let Some(&id) = self.adt_instances.get(&(key, args.clone())) {
            return Some(id);
        }
        let src = self.generic_src(key)?;
        let (item, annotations, decl, linkage, params) =
            (src.item, src.annotations, src.name, src.linkage, src.generics.clone());
        if src.method.is_some() || !matches!(item, ItemKind::Struct { .. } | ItemKind::Enum { .. }) {
            return None;
        }
        // the same instance may already be here, come through an interface
        let origin = self.frames[key.frame].origin.clone();
        if let Some(id) = self.unit.types.find_instance(origin.as_deref(), key.name, &args) {
            self.adt_instances.insert((key, args), id);
            return Some(id);
        }
        if self.too_deep(span) {
            return None;
        }
        // a `quint` measures to an unsigned integer, so it has 1 to 64 qubits
        let gates = origin.as_deref().unwrap_or(&self.unit.name) == "gates";
        if gates
            && self.name(key.name) == "quint"
            && let Some(Arg::Const(n)) = args.first()
            && !(1..=64).contains(n)
        {
            self.report(
                Diagnostic::new(Code::Es23)
                    .with_message(format!("`quint<{n}>` has no qubits to hold a number in, or more than 64"))
                    .at(span)
                    .with_note("measuring a `quint<N>` gives the smallest unsigned integer of at least N bits, and the widest is `u64`")
                    .with_help("use a width from 1 to 64, or a register `[qubit; N]`"),
            );
            return None;
        }
        let is_struct = matches!(item, ItemKind::Struct { .. });
        let def = AdtDef::unresolved(key.name, origin, linkage, args.clone(), is_struct, decl.span);
        let subst = self.substitution(&params, &args);
        let id = self.add_adt(def, item, annotations, key.frame, subst);
        // recorded before the fields are resolved, so that a type mentioning
        // itself behind a reference finds this instance rather than making
        // another
        self.adt_instances.insert((key, args), id);
        self.generic_depth += 1;
        self.adt(id);
        self.generic_depth -= 1;
        Some(id)
    }

    /// The instance of a generic function made with `args`, making it (which
    /// means checking its body with the parameters bound) if this is the
    /// first time it is asked for.
    pub fn instantiate_fn(&mut self, key: GenericKey, args: Vec<Arg>, span: Span) -> Option<FnId> {
        if let Some(&id) = self.fn_instances.get(&(key, args.clone())) {
            return Some(id);
        }
        if !self.serializable_instance(key, &args, span) {
            return None;
        }
        let f = self.generic_fn_item(key)?;
        let src = self.generic_src(key)?;
        let (decl, linkage, params) = (src.name, src.linkage, src.generics.clone());
        let method = src.method;
        let attrs = super::attrs::quiet(src.annotations, self.interner);
        if self.too_deep(span) {
            return None;
        }
        let label = self.generic_label(key);
        let instance = generic::instance_name(&label, &args, &self.unit.types, self.interner);
        let id = FnId(self.unit.fns.len() as u32);
        self.unit.fns.push(tir::Fn {
            args: args.clone(),
            origin: self.frames[key.frame].origin.clone(),
            attrs,
            ..tir::Fn::empty(key.name, linkage, decl.span, decl.span)
        });
        // an instance is checked here and now, so it has no entry of its own
        // among the declarations `check_all` walks
        self.fns.push(None);
        self.fn_instances.insert((key, args.clone()), id);
        let subst = self.substitution(&params, &args);
        self.generic_depth += 1;
        self.within(key.frame, subst, |cx| {
            if let Some(m) = method {
                cx.unit.fns[id.index()].method = cx.method_of_instance(m, &args, span);
            }
            let quiet = u32::from(cx.quiet_inside(id, key.frame));
            cx.quiet_deprecation += quiet;
            let allowed = cx.where_holds(f, &instance, span);
            cx.resolve_signature(id, f);
            if allowed {
                let before = cx.diags.len();
                body::check_fn(cx, id, f);
                // a mistake inside the body shows only for these arguments,
                // so it says which instance, and where it was asked for
                for d in &mut cx.diags[before..] {
                    if d.is_error() {
                        d.labels.push(crate::diag::Label::secondary(
                            span,
                            format!("in `{instance}`, which is asked for here"),
                        ));
                    }
                }
            }
            cx.quiet_deprecation -= quiet;
        });
        self.generic_depth -= 1;
        Some(id)
    }

    /// Checks a constant expression as a value of `ty`, or of whatever type
    /// it has when `ty` is `None`, and makes sure everything it reads can be
    /// evaluated; `None` once why not has been reported.
    fn checked_const(&mut self, e: &Spanned<ast::Expr>, ty: Option<Ty>, role: &str) -> Option<tir::Expr> {
        let before = self.diags.len();
        let checked = body::check_const_as(self, e, ty, role);
        if checked.ty == Ty::Never || self.diags[before..].iter().any(Diagnostic::is_error) {
            return None;
        }
        if let Err(d) = self.ready(&checked) {
            self.report(*d);
            return None;
        }
        Some(checked)
    }

    /// Evaluates a constant expression checked and ready, reporting why it
    /// gave no value.
    fn evaluate_ready(&mut self, checked: &tir::Expr, role: &str) -> Option<Value> {
        match Evaluator::new(&self.unit, self.interner).top(checked, &mut Frame::empty()) {
            Ok(v) => Some(v),
            Err(err) => {
                let d = fold::describe(err, &Need::Constant(role), &self.unit, self.interner);
                self.report(d);
                None
            }
        }
    }

    /// Checks and evaluates a constant expression of type `ty`. `Err(None)`
    /// when checking it failed, which has been reported; `Err(Some(e))` when
    /// evaluating it did not give a value, which is the caller's to report or
    /// to answer, as `@aborts` does.
    pub fn evaluate_const(
        &mut self,
        e: &Spanned<ast::Expr>,
        ty: Option<Ty>,
        role: &str,
    ) -> Result<Value, Option<super::consteval::EvalError>> {
        let checked = self.checked_const(e, ty, role).ok_or(None)?;
        Evaluator::new(&self.unit, self.interner)
            .top(&checked, &mut Frame::empty())
            .map_err(Some)
    }

    /// Checks and evaluates a constant expression of whatever type it has,
    /// giving the value and the type; `None` once why there is none has been
    /// reported.
    pub fn typed_const(&mut self, e: &Spanned<ast::Expr>, role: &str) -> Option<(Value, Ty)> {
        let checked = self.checked_const(e, None, role)?;
        Some((self.evaluate_ready(&checked, role)?, checked.ty))
    }

    /// Evaluates a constant expression already checked, reporting why it
    /// could not be.
    pub fn evaluate_checked(&mut self, checked: &tir::Expr, role: &str) -> Option<Value> {
        if let Err(d) = self.ready(checked) {
            self.report(*d);
            return None;
        }
        self.evaluate_ready(checked, role)
    }

    /// Evaluates a constant expression and reports why it could not be.
    pub fn const_value(&mut self, e: &Spanned<ast::Expr>, ty: Ty, role: &str) -> Option<Value> {
        let checked = self.checked_const(e, Some(ty), role)?;
        self.evaluate_ready(&checked, role)
    }

    /// Whether the generic function `key`'s `where` clause holds for `args`,
    /// found without reporting anything, for an instance only made if it is
    /// allowed.
    pub(super) fn where_allows(&mut self, key: GenericKey, args: &[Arg], span: Span) -> bool {
        let (Some(f), Some(params)) = (self.generic_fn_item(key), self.generic_params(key).map(<[ast::GenericParam]>::to_vec)) else {
            return false;
        };
        if f.where_clause.is_none() {
            return true;
        }
        let before = self.diags.len();
        let subst = self.substitution(&params, args);
        let holds = self.within(key.frame, subst, |cx| cx.where_holds(f, "", span));
        self.diags.truncate(before);
        holds
    }

    /// Whether an instance's `where` clause holds.
    ///
    /// The clause is a constant boolean expression, checked with the
    /// parameters bound, so it can ask about the arguments. One that is false
    /// makes the instance ill-formed at the place that asked for it, which is
    /// where the diagnostic points; the body is then not checked, since it was
    /// written on the assumption the clause states.
    fn where_holds(&mut self, f: &'a FnItem, instance: &str, span: Span) -> bool {
        let Some(w) = &f.where_clause else {
            return true;
        };
        let Some(value) = self.const_value(w, Ty::Bool, "a `where` clause") else {
            return false;
        };
        let holds = value == Value::Bool(true);
        if !holds {
            self.report(
                Diagnostic::new(Code::Em01)
                    .with_message(format!("`{instance}` is not allowed: its `where` clause is false"))
                    .at(span)
                    .also(w.span, "this is false for these arguments")
                    .with_note("a `where` clause is the condition a generic states for its arguments"),
            );
        }
        holds
    }

    /// What a generic parameter stands for in the instance being checked.
    pub fn param_arg(&self, name: Symbol) -> Option<Arg> {
        self.subst.last().and_then(|s| generic::lookup(s, name))
    }

    fn declare_fn(&mut self, f: &'a FnItem, linkage: Linkage, item_span: Span) -> FnId {
        let ast::FnName::Plain(name) = f.name else {
            unreachable!("a method is declared as a method");
        };
        let id = self.add_fn(f, linkage, name, item_span);
        if self.frame == 0 && self.name(name.node) == "main" && self.unit.main.is_none() {
            self.unit.main = Some(id);
        }
        self.declare_value(&name, Def::Fn(id));
        id
    }

    /// Adds a function that is checked where it is made (a closure's body)
    /// rather than resolved on demand.
    pub(super) fn add_checked_fn(&mut self, f: tir::Fn) -> FnId {
        let id = FnId(self.unit.fns.len() as u32);
        self.unit.fns.push(f);
        self.fns.push(None);
        id
    }

    /// Adds a function to the unit, to be resolved when first needed.
    pub(super) fn add_fn(&mut self, f: &'a FnItem, linkage: Linkage, name: Spanned<Symbol>, item_span: Span) -> FnId {
        let id = FnId(self.unit.fns.len() as u32);
        self.unit.fns.push(tir::Fn::empty(name.node, linkage, name.span, item_span));
        self.fns.push(Some(FnSrc {
            item: f,
            sig: Progress::Pending,
            body: Progress::Pending,
            frame: self.frame,
        }));
        id
    }

    /// Resolves and checks every declaration: types first, then signatures,
    /// then initialisers, then bodies.
    pub fn check_all(&mut self) {
        let mut adts: Vec<AdtId> = self.adts.keys().copied().collect();
        adts.sort();
        for &id in &adts {
            self.adt(id);
        }
        for &id in &adts {
            self.report_self_containment(id);
        }
        for i in 0..self.fns.len() {
            self.fn_sig(FnId(i as u32));
        }
        self.check_tests();
        let mut globals: Vec<GlobalId> = self.globals.keys().copied().collect();
        globals.sort();
        for &id in &globals {
            self.global_ty(id);
        }
        for &id in &globals {
            self.global_init(id);
        }
        self.record_aliases();
        for i in 0..self.fns.len() {
            self.fn_body(FnId(i as u32));
        }
    }

    /// Resolves every type alias the unit itself declares, and records each
    /// with the type it stands for, so that its interface can offer them.
    fn record_aliases(&mut self) {
        let frame = self.frame;
        let mut names: Vec<(Symbol, Span, Linkage)> = self.frames[frame]
            .aliases
            .values()
            .map(|a| (a.name.node, a.name.span, a.linkage))
            .collect();
        names.sort_by_key(|(_, span, _)| (span.start, span.end));
        for (name, span, linkage) in names {
            if let Some(ty) = self.alias(name, span) {
                self.unit.aliases.push(tir::Alias { name, ty, linkage });
            }
        }
    }

    /// Checks the bodies of functions made since everything was checked,
    /// such as the instances of generic methods a type's table lists.
    pub fn check_new_bodies(&mut self) {
        let mut i = 0;
        while i < self.fns.len() {
            self.fn_body(FnId(i as u32));
            i += 1;
        }
    }

    /// Makes sure a structure's or enumeration's fields are resolved.
    pub fn adt(&mut self, id: AdtId) {
        let Some(src) = self.adts.get_mut(&id) else {
            return;
        };
        if src.state != Progress::Pending {
            return;
        }
        src.state = Progress::Active;
        let (kind, annotations, frame, subst) = (src.kind, src.annotations, src.frame, src.subst.clone());
        let layers = self.adt_layers.get(&id).cloned().unwrap_or_default();
        let resolved = self.within(frame, subst, |cx| cx.with_layers(layers, |cx| adt::resolve(cx, id, kind, annotations)));
        let open = annotations.iter().any(|g| {
            g.node
                .annotations
                .iter()
                .any(|a| StdAnnotation::from_name(self.interner.resolve(a.node.name.node)) == Some(StdAnnotation::Open))
        });
        let def = self.unit.types.adt_mut(id);
        def.kind = resolved.0;
        def.repr = resolved.1;
        def.open = open;
        if let Some(src) = self.adts.get_mut(&id) {
            src.state = Progress::Done;
        }
    }

    /// Whether a structure or enumeration is being resolved right now.
    fn adt_active(&self, id: AdtId) -> bool {
        self.adts.get(&id).is_some_and(|s| s.state == Progress::Active)
    }

    /// Reports a structure or enumeration that contains itself by value, once
    /// for every group of types that contain each other.
    fn report_self_containment(&mut self, id: AdtId) {
        if self.cycles.contains(&id) || !tir::layout::contains_itself(&self.unit.types, id) {
            return;
        }
        // everything on the same cycle is covered by this one report
        for other in 0..self.unit.types.adts().len() {
            let other = AdtId(other as u32);
            if tir::layout::contains_itself(&self.unit.types, other) && tir::layout::reaches(&self.unit.types, id, other) {
                self.cycles.insert(other);
            }
        }
        self.cycles.insert(id);
        let def = self.unit.types.adt(id);
        let name = self.unit.types.adt_name(id, self.interner);
        let d = Diagnostic::new(Code::Es16)
            .with_message(format!(
                "`{name}` contains itself, so a value of it would be infinitely large"
            ))
            .at(def.span)
            .with_note(
                "a structure holds its fields inline, and an enumeration holds room for its \
                 largest variant, so a type that contains itself (directly or through other \
                 types) has no finite size",
            )
            .with_help(format!(
                "hold the inner `{name}` through a reference, such as `*{name}`, which is two \
                 words whatever it refers to"
            ));
        self.report(d);
    }

    /// Makes sure every structure and enumeration inside `ty` by value is
    /// resolved, so that its layout can be computed. Returns `false`, having
    /// reported it, if one of them is still being declared (its size would
    /// depend on itself).
    pub fn layout_ready(&mut self, ty: Ty, span: Span) -> bool {
        let mut visiting = Vec::new();
        self.layout_ready_in(ty, span, &mut visiting)
    }

    fn layout_ready_in(&mut self, ty: Ty, span: Span, visiting: &mut Vec<AdtId>) -> bool {
        match ty {
            Ty::Adt(id) => {
                if visiting.contains(&id) || self.adt_active(id) {
                    if self.cycles.insert(id) {
                        let name = self.unit.types.adt_name(id, self.interner);
                        let d = Diagnostic::new(Code::Es16)
                            .with_message(format!(
                                "the size of `{name}` is needed to declare `{name}` itself"
                            ))
                            .at_with(span, "the size is needed here")
                            .also(self.unit.types.adt(id).span, "while declaring this")
                            .with_note(
                                "a type's size is only known once all of its fields are, so \
                                 none of them may depend on that size",
                            );
                        self.report(d);
                    }
                    return false;
                }
                self.adt(id);
                visiting.push(id);
                let inner: Vec<Ty> = self.unit.types.adt(id).field_types().collect();
                let ok = inner.into_iter().all(|t| self.layout_ready_in(t, span, visiting));
                visiting.pop();
                ok && !self.cycles.contains(&id)
            }
            Ty::Array(_) => {
                let (elem, _) = self.unit.types.as_array(ty).expect("an array type");
                self.layout_ready_in(elem, span, visiting)
            }
            _ => true,
        }
    }

    /// Makes sure a function's signature is resolved.
    pub fn fn_sig(&mut self, id: FnId) {
        let Some(Some(src)) = self.fns.get_mut(id.index()) else {
            return;
        };
        if src.sig != Progress::Pending {
            return;
        }
        src.sig = Progress::Active;
        let (f, frame) = (src.item, src.frame);
        let quiet = self.quiet_inside(id, frame);
        let layers = self.fn_layers.get(&id).cloned().unwrap_or_default();
        self.within(frame, Vec::new(), |cx| {
            cx.with_layers(layers, |cx| cx.quietly(quiet, |cx| cx.resolve_signature(id, f)));
        });
        if let Some(Some(src)) = self.fns.get_mut(id.index()) {
            src.sig = Progress::Done;
        }
    }

    /// The parameter and return types of `callee`, its signature resolved
    /// first if it is this unit's.
    pub(super) fn signature(&mut self, callee: Callee) -> (Vec<Ty>, Ty) {
        match callee {
            Callee::Fn(f) => {
                self.fn_sig(f);
                let f = self.unit.func(f);
                (f.param_types().collect(), f.ret)
            }
            Callee::Extern(x) => {
                let x = self.unit.extern_fn(x);
                (x.params.clone(), x.ret)
            }
        }
    }

    /// Resolves the parameter and return types of `id` from the declaration
    /// `f`, which is also how an instance of a generic function gets its
    /// signature (with the parameters bound).
    pub(super) fn resolve_signature(&mut self, id: FnId, f: &'a FnItem) {
        let mut locals: Vec<Local> = Vec::new();
        let mut params = Vec::new();
        for p in &f.params {
            check_annotations(&p.annotations, Target::Parameter, self.quantum, self.interner, &mut self.diags);
            if p.is_self && !self.self_allowed(id, locals.is_empty(), p.name.span) {
                continue;
            }
            // a `constexpr` parameter beside ordinary ones is fixed in each
            // specialisation, where its value is known: not a parameter of
            // the code
            if p.ty.node.is_constexpr() && self.param_arg(p.name.node).is_some() {
                continue;
            }
            self.check_reserved(p.name.node, p.name.span);
            let r = self.resolve_or_report(&p.ty);
            if r.ty == Ty::Void {
                self.report(void_binding(p.ty.span));
            }
            if let Some(earlier) = locals.iter().find(|l| l.name == p.name.node) {
                let d = report::duplicate(
                    p.name.span,
                    earlier.span,
                    self.name(p.name.node),
                    "in one parameter list",
                );
                self.report(d);
            }
            params.push(LocalId(locals.len() as u32));
            locals.push(Local {
                name: p.name.node,
                ty: r.ty,
                constant: r.constant,
                constexpr: r.constexpr,
                aux: false,
                span: p.name.span,
            });
        }
        let ret = match &f.ret {
            None => types::Resolved {
                ty: Ty::Void,
                constant: false,
                constexpr: false,
            },
            Some(t) => self.resolve_or_report(t),
        };
        if ret.constant && !ret.constexpr {
            let at = f.ret.as_ref().map_or(self.unit.func(id).span, |t| t.span);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("a function's result cannot be `const`")
                    .at(at)
                    .with_note("a result is a value, not a binding, so nothing could assign it anyway")
                    .with_help("write `constexpr` if the result is known during translation, or drop `const`"),
            );
        }

        let name = self.unit.func(id).name;
        let name_span = self.unit.func(id).span;
        let n = locals.len();
        let n_const = locals.iter().filter(|l| l.constexpr).count();
        let kind = if (n > 0 && n_const == n) || (n == 0 && ret.constexpr) {
            FnKind::Constant
        } else {
            if n_const > 0 {
                // a plain function mixing them is specialised; a method is
                // found through its type, one function for each name
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "the method `{}` has `constexpr` parameters beside ordinary ones",
                            self.name(name)
                        ))
                        .at(name_span)
                        .with_note(
                            "a function mixing them is made once for each value of its `constexpr` \
                             parameters, but a type has one method of each name, whatever it is called with",
                        )
                        .with_help("make the method's parameters all `constexpr` or none, or write a plain function"),
                );
            } else if ret.constexpr {
                self.report(unkeepable_constant_result(name_span, self.name(name)));
            }
            FnKind::Runtime
        };
        let func = &mut self.unit.fns[id.index()];
        func.params = params;
        func.locals = locals;
        func.ret = ret.ty;
        func.kind = kind;
        self.check_receiver(id);
    }

    /// The type `t` is written as; `None` once why it is none is reported.
    pub(super) fn written_type(&mut self, t: &Spanned<ast::Type>) -> Option<Ty> {
        match types::resolve(self, t) {
            Ok(r) => Some(r.ty),
            Err(d) => {
                self.report(*d);
                None
            }
        }
    }

    fn resolve_or_report(&mut self, t: &Spanned<ast::Type>) -> types::Resolved {
        match types::resolve(self, t) {
            Ok(r) => r,
            Err(d) => {
                self.report(*d);
                types::Resolved {
                    ty: Ty::Never,
                    constant: false,
                    constexpr: false,
                }
            }
        }
    }

    /// Makes sure a unit-scope object's type is resolved.
    pub fn global_ty(&mut self, id: GlobalId) {
        let Some(src) = self.globals.get_mut(&id) else {
            return;
        };
        if src.ty_state != Progress::Pending {
            return;
        }
        src.ty_state = Progress::Active;
        let ty = src.ty;
        let r = self.resolve_or_report(ty);
        if r.ty == Ty::Void {
            self.report(void_binding(ty.span));
        }
        let g = &mut self.unit.globals[id.index()];
        g.ty = r.ty;
        g.constant = r.constant;
        if let Some(src) = self.globals.get_mut(&id) {
            src.ty_state = Progress::Done;
        }
    }

    /// How far checking a unit-scope object's initialiser has got; `Done` for
    /// an object with nothing to check.
    pub fn init_state(&self, id: GlobalId) -> Progress {
        self.globals.get(&id).map_or(Progress::Done, |s| s.init_state)
    }

    /// Makes sure a unit-scope object's initialiser is checked.
    pub fn global_init(&mut self, id: GlobalId) {
        self.global_ty(id);
        let Some(src) = self.globals.get_mut(&id) else {
            return;
        };
        if src.init_state != Progress::Pending {
            return;
        }
        src.init_state = Progress::Active;
        if let Some(init) = src.init {
            let e = body::check_global_init(self, id, init);
            self.unit.globals[id.index()].init = Some(e);
        }
        if let Some(src) = self.globals.get_mut(&id) {
            src.init_state = Progress::Done;
        }
    }

    /// Makes sure a function's body is checked.
    pub fn fn_body(&mut self, id: FnId) {
        self.fn_sig(id);
        let Some(Some(src)) = self.fns.get_mut(id.index()) else {
            return;
        };
        if src.body != Progress::Pending {
            return;
        }
        src.body = Progress::Active;
        let (item, frame) = (src.item, src.frame);
        // a `where` clause on a function that is not generic has nothing to
        // vary with, so it is decided once, here
        let (name, at) = (self.unit.func(id).name, self.unit.func(id).span);
        let text = self.name(name).to_owned();
        let quiet = self.quiet_inside(id, frame);
        let layers = self.fn_layers.get(&id).cloned().unwrap_or_default();
        self.within(frame, Vec::new(), |cx| {
            cx.with_layers(layers, |cx| {
                cx.quietly(quiet, |cx| {
                    if cx.where_holds(item, &text, at) {
                        body::check_fn(cx, id, item);
                    }
                });
            });
        });
        if let Some(Some(src)) = self.fns.get_mut(id.index()) {
            src.body = Progress::Done;
        }
    }

    fn body_state(&self, id: FnId) -> Progress {
        match self.fns.get(id.index()) {
            Some(Some(s)) => s.body,
            _ => Progress::Done,
        }
    }

    /// The value of a constant, computed now if it was not yet. `None` if it
    /// could not be, which has been reported.
    pub fn global_value(&mut self, id: GlobalId, needed_at: Span) -> Option<Value> {
        if let Some(v) = &self.unit.global(id).value {
            return Some(v.clone());
        }
        self.global_init(id);
        let init = self.unit.global(id).init.clone()?;
        if let Err(d) = self.ready(&init) {
            self.report(*d);
            return None;
        }
        let result = Evaluator::new(&self.unit, self.interner).global(id, needed_at);
        match result {
            Ok(v) => {
                self.unit.globals[id.index()].value = Some(v.clone());
                Some(v)
            }
            Err(e) => {
                let name = self.name(self.unit.global(id).name);
                let d = fold::describe(e, &Need::Object(name), &self.unit, self.interner);
                self.report(d);
                None
            }
        }
    }

    /// The type a name stands for.
    ///
    /// An alias is resolved the first time something names it, since it may
    /// name a type declared later, or an array whose length is a constant
    /// declared later still. An alias defined in terms of itself has no type;
    /// that is reported once, and the alias stands in as `never`.
    ///
    /// One the current frame's unit does not declare may be one the `core`
    /// library does, as `string` is.
    pub fn alias(&mut self, name: Symbol, span: Span) -> Option<Ty> {
        let frame = self.frame;
        if !self.frames[frame].aliases.contains_key(&name) {
            let p = self.prelude.filter(|&p| p != frame)?;
            return self.frame_alias(p, name, span);
        }
        let src = self.frames[frame].aliases.get_mut(&name)?;
        match src.state {
            Progress::Done => return src.resolved,
            Progress::Active => {
                // named again while being resolved: a cycle
                let decl = src.name.span;
                src.resolved = Some(Ty::Never);
                src.state = Progress::Done;
                let text = self.name(name).to_owned();
                self.report(
                    Diagnostic::new(Code::Es16)
                        .with_message(format!("the type `{text}` stands for is `{text}` itself"))
                        .at_with(span, "named here")
                        .also(decl, "while working out what this stands for")
                        .with_note("a type alias is another spelling of a type, so it cannot name itself"),
                );
                return Some(Ty::Never);
            }
            Progress::Pending => src.state = Progress::Active,
        }

        // resolve what it stands for, then record it
        let written = src.ty;
        let resolved = match types::resolve(self, written) {
            Ok(r) => {
                if r.constant {
                    self.report(
                        Diagnostic::new(Code::Ec01)
                            .with_message("a type alias cannot be `const`")
                            .at(written.span)
                            .with_note(
                                "whether a value is known during translation belongs to the \
                                 binding that holds it, not to the type's name",
                            ),
                    );
                }
                r.ty
            }
            Err(d) => {
                self.report(*d);
                Ty::Never
            }
        };
        let src = self.frames[frame].aliases.get_mut(&name).expect("just resolved");
        src.state = Progress::Done;
        src.resolved = Some(resolved);
        Some(resolved)
    }

    /// The type alias `name` that frame `frame`'s unit declares for other
    /// units to use, resolved in that unit's own scope.
    fn frame_alias(&mut self, frame: usize, name: Symbol, span: Span) -> Option<Ty> {
        if self.frames[frame].aliases.get(&name)?.linkage != Linkage::Program {
            return None;
        }
        self.within(frame, Vec::new(), |cx| cx.alias(name, span))
    }

    /// The type alias `name` of the imported unit `u`, as `u::name` names it:
    /// one its interface offers, which every alias not declared `static` is.
    pub fn imported_alias(&mut self, u: super::scope::UnitRef, name: Symbol, span: Span) -> Option<Ty> {
        if u == super::scope::UnitRef::SELF {
            return self.alias(name, span);
        }
        let iface = self.imported[u.0 as usize];
        let ty = iface.aliases.iter().find(|a| a.name == name && a.linkage == Linkage::Program)?.ty;
        Some(super::imports::translate(self, iface, ty))
    }

    /// Evaluates a constant `usize`, such as an array length. `role` says what
    /// it is for, as in "the length of this array". `None` if it could not be
    /// evaluated, which has been reported.
    pub fn const_usize(&mut self, e: &Spanned<ast::Expr>, role: &str) -> Option<u64> {
        match self.const_value(e, Ty::USIZE, role)? {
            Value::Int(v, _) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Makes sure everything `e` would need to be evaluated now has been
    /// checked: the initialisers of the constants it reads, and the bodies of
    /// the constant functions it calls, transitively.
    fn ready(&mut self, e: &tir::Expr) -> Result<(), Box<Diagnostic>> {
        let mut work = needs_of_expr(e);
        let mut seen_globals = HashSet::new();
        let mut seen_fns = HashSet::new();
        while let Some(n) = work.pop() {
            match n {
                Needed::Global(g, span) => {
                    if !seen_globals.insert(g) || !self.unit.global(g).constant || self.unit.global(g).value.is_some() {
                        continue;
                    }
                    if self.init_state(g) == Progress::Active {
                        return Err(Box::new(depends_on_itself(self.name(self.unit.global(g).name), span, self.unit.global(g).span)));
                    }
                    self.global_init(g);
                    if let Some(init) = &self.unit.global(g).init {
                        work.extend(needs_of_expr(init));
                    }
                }
                Needed::Fn(f, span) => {
                    if !seen_fns.insert(f) {
                        continue;
                    }
                    self.fn_sig(f);
                    if self.unit.func(f).kind != FnKind::Constant {
                        continue;
                    }
                    if self.body_state(f) == Progress::Active {
                        let name = self.name(self.unit.func(f).name);
                        return Err(Box::new(
                            Diagnostic::new(Code::Ec01)
                                .with_message(format!(
                                    "this constant calls `{name}`, whose own body needs this constant"
                                ))
                                .at_with(span, format!("`{name}` is called here"))
                                .also(self.unit.func(f).span, "while checking this function"),
                        ));
                    }
                    self.fn_body(f);
                    let mut v = Needs(Vec::new());
                    v.block(&self.unit.func(f).body);
                    work.extend(v.0);
                }
            }
        }
        Ok(())
    }
}

/// Something an expression needs checked before it can be evaluated.
enum Needed {
    Global(GlobalId, Span),
    Fn(FnId, Span),
}

struct Needs(Vec<Needed>);

impl Visit for Needs {
    fn expr(&mut self, e: &tir::Expr) {
        match &e.kind {
            ExprKind::Global(g) => self.0.push(Needed::Global(*g, e.span)),
            ExprKind::Call {
                callee: Callee::Fn(f),
                ..
            } => self.0.push(Needed::Fn(*f, e.span)),
            _ => {}
        }
        visit::walk_expr(self, e);
    }
}

fn needs_of_expr(e: &tir::Expr) -> Vec<Needed> {
    let mut v = Needs(Vec::new());
    v.expr(e);
    v.0
}

/// A constant whose value is needed while its own initialiser is checked.
fn depends_on_itself(name: &str, needed: Span, decl: Span) -> Diagnostic {
    Diagnostic::new(Code::Ec01)
        .with_message(format!(
            "the value of `{name}` depends on itself, so it cannot be computed"
        ))
        .at_with(needed, format!("`{name}` is needed here"))
        .also(decl, format!("while computing `{name}` itself"))
}

/// What an annotation is written on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    /// A function.
    Function,
    /// A unit-scope object.
    Object,
    /// A structure.
    Structure,
    /// An enumeration.
    Enumeration,
    /// A structure field.
    Field,
    /// A function parameter.
    Parameter,
    /// A block.
    Block,
}

impl Target {
    fn noun(self) -> &'static str {
        match self {
            Target::Function => "a function",
            Target::Object => "a unit-scope object",
            Target::Structure => "a structure",
            Target::Enumeration => "an enumeration",
            Target::Field => "a field",
            Target::Parameter => "a parameter",
            Target::Block => "a block",
        }
    }
}

/// Checks each annotation's name and placement. An unknown one is ignored
/// with a warning; a quantum one in a classical unit, or one written on
/// something it cannot apply to, is an error. Arguments are checked where
/// each annotation is acted on.
pub fn check_annotations(
    groups: &[Spanned<AnnotationGroup>],
    target: Target,
    quantum: bool,
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) {
    for g in groups {
        for a in &g.node.annotations {
            let name = interner.resolve(a.node.name.node);
            match StdAnnotation::from_name(name) {
                None => diags.push(
                    Diagnostic::new(Code::Ea01)
                        .with_message(format!(
                            "`[{name}]` is not an annotation this compiler knows, so it is ignored"
                        ))
                        .at(a.node.name.span)
                        .with_note(
                            "an unknown annotation is allowed, because a program may carry \
                             annotations meant for other tools",
                        )
                        .with_help("if you meant a standard annotation, check its spelling")
                        .as_warning(),
                ),
                Some(StdAnnotation::Test | StdAnnotation::Inline) if target == Target::Function => {}
                Some(StdAnnotation::Export) if matches!(target, Target::Function | Target::Object) => {}
                Some(StdAnnotation::Deprecated)
                    if matches!(
                        target,
                        Target::Function | Target::Object | Target::Structure | Target::Enumeration
                    ) => {}
                Some(StdAnnotation::Packed | StdAnnotation::Align)
                    if matches!(target, Target::Structure | Target::Enumeration) => {}
                Some(StdAnnotation::Repr) if target == Target::Enumeration => {}
                Some(StdAnnotation::Open | StdAnnotation::Typeinfo | StdAnnotation::NoTypeinfo | StdAnnotation::Tcon)
                    if matches!(target, Target::Structure | Target::Enumeration) => {}
                Some(StdAnnotation::Entry | StdAnnotation::Dynamic | StdAnnotation::Adjoint) if target == Target::Function && quantum => {}
                Some(StdAnnotation::TrustedUnitary) if target == Target::Function && quantum => {}
                // claims about operators, read into their judgements' claims
                Some(StdAnnotation::Cover | StdAnnotation::Gauge | StdAnnotation::Expect | StdAnnotation::Frame | StdAnnotation::Glue)
                    if target == Target::Function && quantum => {}
                Some(StdAnnotation::Cover | StdAnnotation::Gauge) if target == Target::Parameter && quantum => {}
                Some(StdAnnotation::Iso) if target == Target::Enumeration && quantum => {}
                // the annotations about operators and their judgements have
                // nothing to say of a classical unit, which has neither
                Some(s) if !quantum && s.is_quantum() => diags.push(
                    Diagnostic::new(Code::Es18)
                        .with_message(format!("`[{name}]` applies to the operators of a quantum unit"))
                        .at(a.span)
                        .with_note(
                            "a classical unit's functions run as ordinary code; circuits, their \
                             entry points and the claims made about them belong to quantum units",
                        )
                        .with_help(format!(
                            "remove `[{name}]`, or move the operator into a unit that begins with \
                             `#unit quantum`"
                        )),
                ),
                Some(
                    s @ (StdAnnotation::Inline
                    | StdAnnotation::Test
                    | StdAnnotation::Export
                    | StdAnnotation::Deprecated
                    | StdAnnotation::Open
                    | StdAnnotation::Typeinfo
                    | StdAnnotation::NoTypeinfo
                    | StdAnnotation::Tcon
                    | StdAnnotation::Packed
                    | StdAnnotation::Align
                    | StdAnnotation::Repr),
                ) => {
                    let applies = match s {
                        StdAnnotation::Inline | StdAnnotation::Test => "a function",
                        StdAnnotation::Export => "a function or unit-scope object",
                        StdAnnotation::Deprecated => "a function, object, structure or enumeration",
                        StdAnnotation::Repr => "an enumeration",
                        _ => "a structure or enumeration",
                    };
                    diags.push(
                        Diagnostic::new(Code::Es18)
                            .with_message(format!(
                                "`[{name}]` applies to {applies}, not to {}",
                                target.noun()
                            ))
                            .at(a.span)
                            .with_help(format!("remove `[{name}]`, or move it onto {applies}")),
                    );
                }
                Some(_) => diags.push(report::unsupported(
                    a.span,
                    format_args!("the `[{name}]` annotation"),
                )),
            }
        }
    }
}

/// A binding or parameter declared with type `void`.
pub fn void_binding(span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message("nothing can have type void: it has no values to hold")
        .at(span)
        .with_note("`void` is only a return type, for a function that returns nothing")
}

/// A unit-scope binding with no initial value.
fn no_initializer(span: Span, name: &str) -> Diagnostic {
    Diagnostic::new(Code::Ec05)
        .with_message(format!("`{name}` is never given a value"))
        .at(span)
        .with_note(
            "a unit-scope binding needs a value before the program starts: either an \
             initial value written here, or an assignment in the unit's initialiser",
        )
        .with_help(format!("write `= <value>` after the type of `{name}`"))
}

/// `-> const T` on a function whose parameters are not all `const`.
fn unkeepable_constant_result(span: Span, name: &str) -> Diagnostic {
    Diagnostic::new(Code::Ec01)
        .with_message(format!(
            "`{name}` promises a `constexpr` result, but its parameters are only known when \
             the program runs"
        ))
        .at(span)
        .with_note(
            "a result can only be known during translation if everything it is computed \
             from is, so a `constexpr` result needs every parameter to be `constexpr`",
        )
        .with_help("make every parameter `constexpr`, or drop `constexpr` from the return type")
}

/// `static` where there is no linkage to restrict.
pub fn static_without_linkage(span: Span, what: &str) -> Diagnostic {
    Diagnostic::new(Code::El01)
        .with_message(format!("`static` has no meaning on {what}"))
        .at(span)
        .with_note(
            "`static` keeps a unit-scope declaration private to its unit; there is \
             nothing to keep private here",
        )
        .with_help("remove `static`")
}

/// The limit on fields or variants.
pub fn check_count(limit: Limit, count: usize, span: Span) -> Option<Diagnostic> {
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    (!limit.permits(count)).then(|| limit.exceeded(span, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sema::testing::{analyzed, check, codes};

    #[test]
    fn functions_and_bindings_are_collected_with_their_signatures() {
        let (u, d) = analyzed("fn add(a: i32, b: i32) -> i32 { a + b }\nlet LIMIT: const u8 = 9;");
        assert!(d.is_empty(), "{d:?}");
        let f = &u.fns[0];
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.ret, Ty::Int(tir::IntTy::I32));
        assert_eq!(f.kind, FnKind::Runtime);
        let g = &u.globals[0];
        assert!(g.constant);
        assert_eq!(g.ty, Ty::Int(tir::IntTy::U8));
    }

    #[test]
    fn main_is_found() {
        let u = check("fn helper() { }\nfn main() -> i32 { 0 }");
        assert_eq!(u.main, Some(FnId(1)));
    }

    #[test]
    fn a_function_whose_parameters_are_all_const_is_a_constant_function() {
        let u = check("fn mask(n: constexpr u32) -> constexpr u64 { (1u64 << n) - 1 }");
        assert_eq!(u.fns[0].kind, FnKind::Constant);
        let u = check("fn answer() -> constexpr i32 { 42 }");
        assert_eq!(u.fns[0].kind, FnKind::Constant, "no parameters, constant result");
    }

    #[test]
    fn a_constant_result_needs_constant_parameters() {
        assert_eq!(codes("fn f(n: u32) -> constexpr u32 { n }"), [Code::Ec01]);
    }

    #[test]
    fn two_items_of_one_name_are_a_duplicate() {
        assert_eq!(codes("fn f() { }\nfn f() { }"), [Code::Es05]);
        assert_eq!(codes("struct P { }\nenum P { A }"), [Code::Es05]);
        check("struct P { }\nfn P() { }");
    }

    #[test]
    fn a_repeated_parameter_is_a_duplicate() {
        assert_eq!(codes("fn f(a: i32, a: i32) { }"), [Code::Es05]);
    }

    #[test]
    fn a_unit_scope_binding_without_a_value_is_ec05() {
        assert_eq!(codes("let X: i32;"), [Code::Ec05]);
    }

    #[test]
    fn an_impl_for_an_unknown_type_is_reported() {
        assert_eq!(codes("impl P { fn f() { } }"), [Code::Es04]);
    }

    #[test]
    fn a_generic_type_alias_stands_for_its_type_with_the_arguments_bound() {
        let (u, i) = {
            let (u, d) = crate::sema::testing::analyzed(
                "type Twin<T> = (T, T);\n\
                 type Grid<T, N: const usize> = [[T; N]; N];\n\
                 fn f(p: Twin<i32>, g: Grid<u8, 3>) -> i32 { let (a, b) = p; a + b + (g[2][2] as i32) }",
            );
            assert!(d.iter().all(|d| !d.is_error()), "{d:?}");
            (u, Interner::new())
        };
        let shown: Vec<String> = u.fns[0].param_types().map(|t| u.types.display(t, &i)).collect();
        assert_eq!(shown, ["(i32, i32)", "[[u8; 3]; 3]"]);
        assert_eq!(codes("type Twin<T> = (T, T);\nfn f(p: Twin) { }"), [Code::Es06]);
    }

    #[test]
    fn a_generic_declaration_makes_one_instance_per_set_of_arguments() {
        let u = check(
            "struct Pair<A, B> { first: A, second: B }\n\
             fn first<A, B>(p: *Pair<A, B>) -> A { p.first }\n\
             fn main() -> i32 {\n\
                 let p = Pair { first: 1i32, second: true };\n\
                 let q = Pair { first: 'a', second: 2u8 };\n\
                 first(&p) + first(&q) as i32\n\
             }",
        );
        // one structure and one function per set of arguments, and the
        // generic declarations themselves are not among them
        assert_eq!(u.types.adts().len(), 2);
        assert!(u.types.adts().iter().all(|a| a.args.len() == 2));
        assert_eq!(u.fns.iter().filter(|f| !f.args.is_empty()).count(), 2);
    }

    #[test]
    fn an_instance_is_made_once_however_often_it_is_asked_for() {
        let u = check(
            "fn twice<T>(x: T) -> T { x }\n\
             fn main() -> i32 { twice(1i32) + twice(2i32) + twice::<i32>(3) }",
        );
        assert_eq!(u.fns.iter().filter(|f| !f.args.is_empty()).count(), 1);
    }

    #[test]
    fn a_generic_argument_that_nothing_settles_is_reported() {
        // `T` appears only in the return type, so the call cannot say what it is
        assert_eq!(codes("fn make<T>() -> T { }\nfn f() { make(); }"), [Code::Es07]);
        // written out, it is fine
        assert_eq!(codes("fn make<T>(x: T) -> T { x }\nfn f() { make::<i32>(1); }"), []);
    }

    #[test]
    fn a_variant_that_carries_nothing_takes_its_instance_from_its_context() {
        let src = "enum Maybe<T> { Just(T), Nothing }\n\
                   fn get(m: Maybe<i32>) -> i32 { match m { Maybe::Just(x) => x, Maybe::Nothing => 0 } }\n";
        check(&format!("{src}fn f() -> i32 {{ get(Maybe::Nothing) }}"));
        check(&format!("{src}fn f() -> Maybe<u8> {{ Maybe::Nothing }}"));
        check(&format!("{src}fn f() -> i32 {{ let m: Maybe<i32> = Maybe::Nothing; get(m) }}"));
        check(&format!("{src}fn f() -> i32 {{ get(Maybe::Nothing::<i32>) }}"));
        // nothing here says which `Maybe` it is
        assert_eq!(codes(&format!("{src}fn f() {{ let m = Maybe::Nothing; }}")), [Code::Es07]);
        // and a variant that carries a value says it by that value
        check(&format!("{src}fn f() -> i32 {{ get(Maybe::Just(3)) }}"));
    }

    #[test]
    fn a_generic_type_named_without_arguments_is_reported() {
        assert_eq!(codes("struct Pair<A, B> { a: A, b: B }\nfn f(p: Pair) { }"), [Code::Es06]);
    }

    #[test]
    fn a_where_clause_decides_which_instances_are_allowed() {
        let src = "fn at<T, N: const usize>(xs: [T; N]) -> T where N > 0 { xs[0] }
";
        check(&format!("{src}fn f() -> i32 {{ at([1i32, 2]) }}"));
        assert_eq!(codes(&format!("{src}fn f() -> i32 {{ let e: [i32; 0] = []; at(e) }}")), [Code::Em01]);
        assert_eq!(codes("fn f() where 1 > 2 { }"), [Code::Em01]);
        check("fn f() where 2 > 1 { }");
    }

    #[test]
    fn a_generic_that_instantiates_itself_stops_at_the_limit() {
        let d = codes("fn f<T>(x: T) -> i32 { f([x]) }\nfn g() -> i32 { f(1i32) }");
        assert!(d.contains(&Code::Em05), "{d:?}");
    }

    #[test]
    fn a_type_alias_is_another_spelling_of_a_type() {
        let u = check("type Meters = u32;\nfn f(p: Meters) -> Meters { p }");
        assert_eq!(u.fns[0].ret, Ty::Int(crate::tir::IntTy::U32));
        // an alias may name a type or a constant declared further down
        check("type Row = [u8; SIDE];\nlet SIDE: const usize = 2;\nfn f(r: Row) -> u8 { r[0] }");
        check("type Point = P;\nstruct P { x: i32 }\nfn f(p: Point) -> i32 { p.x }");
    }

    #[test]
    fn a_type_alias_that_names_itself_is_reported_once() {
        assert_eq!(codes("type A = B;\ntype B = A;\nfn f(x: A) { }"), [Code::Es16]);
    }

    #[test]
    fn a_static_import_restricts_nothing() {
        let (_, d) = analyzed("static import gates;");
        assert!(d.iter().any(|d| d.code == Code::El01), "{d:?}");
    }

    #[test]
    fn an_unknown_annotation_is_only_a_warning() {
        let (_, d) = analyzed("[my_tool]\nfn f() { }");
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Ea01]);
        assert!(!d[0].is_error());
    }

    #[test]
    fn annotations_acted_on_are_accepted_and_quantum_ones_refused() {
        check("[inline]\nfn f() { }");
        check("[export: \"f_sym\"]\nfn f() { }");
        check("[test]\nfn t() { }\n[deprecated: \"old\"]\nfn g() { }");
        // a classical unit has no operators for a quantum annotation to
        // describe
        assert_eq!(codes("[entry]\nfn f() { }"), [Code::Es18]);
        assert_eq!(codes("[expect: monic]\nfn f() { }"), [Code::Es18]);
    }

    #[test]
    fn a_layout_annotation_on_a_function_is_misplaced() {
        assert_eq!(codes("[packed]\nfn f() { }"), [Code::Es18]);
        assert_eq!(codes("[repr: u16]\nstruct S { }"), [Code::Es18]);
        check("[packed]\nstruct S { a: u8, b: u32 }");
        check("[repr: u16]\nenum E { A, B }");
    }

    #[test]
    fn a_void_parameter_holds_nothing() {
        assert_eq!(codes("fn f(x: void) { }"), [Code::Es06]);
    }

    #[test]
    fn a_type_that_contains_itself_is_reported_once() {
        assert_eq!(codes("struct A { b: B }\nstruct B { a: A }"), [Code::Es16]);
        assert_eq!(codes("struct N { next: N }"), [Code::Es16]);
        check("struct N { next: *N }");
    }

    #[test]
    fn an_array_length_may_name_a_constant_declared_later() {
        let u = check("struct Buf { bytes: [u8; SIZE] }\nlet SIZE: const usize = 2 * 8;");
        let tir::AdtKind::Struct { fields } = &u.types.adt(AdtId(0)).kind else {
            panic!()
        };
        assert_eq!(u.types.as_array(fields[0].ty).map(|a| a.1), Some(16));
    }

    #[test]
    fn a_size_that_depends_on_itself_is_refused() {
        let d = codes("struct S { a: [u8; @sizeof(S)] }");
        assert!(d.contains(&Code::Es16), "{d:?}");
    }
}
