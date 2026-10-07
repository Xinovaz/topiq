//! The closed table of diagnostic identifiers.
//!
//! Every way a program can fail carries a stable identifier, such as `EU01`
//! or `RA07`, so a failure can be looked up, tested against or searched for
//! without depending on its wording.
//!
//! The table is closed. An identifier means one thing for the life of the
//! language: a later release may add identifiers, but may not give one a new
//! meaning, nor report one for a program the language permits. So a test
//! asserting `EQ01` cannot start passing because another check was
//! renumbered.
//!
//! # The whole table
//!
//! Every identifier the language defines is declared here, the quantum
//! side's included, so no check can invent one or reuse one that means
//! something else. [`Code::ALL`] is the list, and the tests at the foot of
//! this file pin its shape.
//!
//! # How identifiers are grouped
//!
//! The prefix says what area of the language failed:
//!
//! | prefix | area |
//! |---|---|
//! | `EU` | unit structure: the `#unit` directive, unit kind, conductor, initialisers |
//! | `EL` | linkage: what one unit publishes and another consumes |
//! | `EC` | constants and constant evaluation |
//! | `EA` | annotations and pragmas |
//! | `EP` | prefix compatibility between types |
//! | `EQ` | quantum semantics: linearity, control, measurement, synthesis |
//! | `EJ` | judgements: covers, gauges, contracts, frames, holonomy |
//! | `ET` | typed data notation |
//! | `ES` | static semantics: syntax, names, types, calls, ownership, `match` |
//! | `EM` | miscellaneous static checks, including the implementation limits |
//! | `RA` | run-time aborts |
//! | `TQ` | this implementation's own (see below) |
//!
//! [`Phase`] says *when* the failure is caught, which is a separate question:
//! most identifiers are caught while translating one unit, a few only once
//! units are linked together, and the `RA` family only while the program runs.
//!
//! # This implementation's own
//!
//! Two failures belong to this compiler rather than to the language. They
//! carry a `TQ` prefix, which cannot collide with the `E` and `RA` families,
//! and [`Code::is_language_defined`] is `false` for them:
//!
//! - **`TQ003`**: a construct the compiler does not translate yet, or code
//!   written without the `core` library, so it never claims a success it
//!   cannot back.
//! - **`TQ011`**: the compiler itself failed.

/// When a failure is detected, and therefore what is still salvageable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Phase {
    /// While translating a single unit. The unit is rejected; other units are
    /// unaffected and may still translate.
    Translation,
    /// While linking, when a check spans two units or depends on what the
    /// target machine can do. Such a check is never quietly demoted to a
    /// run-time test: a program that would fail it does not link.
    Link,
    /// While the program runs. The operation aborts rather than continuing with
    /// a value the language cannot justify.
    Runtime,
}

impl Phase {
    /// A short lowercase name for rendering.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Translation => "translation",
            Phase::Link => "link",
            Phase::Runtime => "runtime",
        }
    }
}

/// Whether a diagnostic stops translation.
///
/// Almost everything is an error. The warnings are an unknown annotation or
/// pragma, which address tools this compiler does not know; an unreachable
/// `match` arm; a use of a deprecated item; and a judgement that left the
/// checked fragment.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Severity {
    /// Translation continues.
    Warning,
    /// The program is rejected.
    Error,
}

impl Severity {
    /// A short lowercase name for rendering.
    pub fn name(self) -> &'static str {
        match self {
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

/// Builds the [`Code`] enum together with its tables, so that the identifier,
/// phase, severity and text of every diagnostic are declared exactly once.
macro_rules! define_codes {
    ($(
        $(#[$attr:meta])*
        $variant:ident = $id:literal, $phase:ident, $sev:ident, $msg:literal;
    )*) => {
        /// A diagnostic identifier.
        ///
        /// See the [module documentation][self] for what the prefixes mean and
        /// why the table is closed.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[non_exhaustive]
        pub enum Code {
            $(
                $(#[$attr])*
                #[doc = ""]
                #[doc = concat!("`", $id, "`: ", $msg)]
                $variant,
            )*
        }

        impl Code {
            /// Every identifier, grouped by area and ordered within each group.
            pub const ALL: &'static [Code] = &[ $( Code::$variant ),* ];

            /// The identifier as it appears in a diagnostic message, such as
            /// `EU01`.
            pub fn id(self) -> &'static str {
                match self { $( Code::$variant => $id ),* }
            }

            /// The headline text: what went wrong, in one clause.
            ///
            /// This is the summary line. The detail a reader needs in order to
            /// fix the program (which name, which limit, what to write
            /// instead) is attached at the point the diagnostic is raised, as
            /// labels, notes and help.
            pub fn message(self) -> &'static str {
                match self { $( Code::$variant => $msg ),* }
            }

            /// When this failure is detected.
            pub fn phase(self) -> Phase {
                match self { $( Code::$variant => Phase::$phase ),* }
            }

            /// Whether translation continues past this diagnostic.
            pub fn severity(self) -> Severity {
                match self { $( Code::$variant => Severity::$sev ),* }
            }
        }
    };
}

define_codes! {
    ////////////////////
    // UNIT STRUCTURE //
    ////////////////////

    /// Every source file opens by declaring what it is, with a `#unit`
    /// directive naming its kind and name. Omitting it, writing it after
    /// something else, or writing a second one all land here.
    Eu01 = "EU01", Translation, Error, "missing, misplaced or duplicate #unit directive";
    /// A classical unit may not declare quantum storage. The unit kind is a
    /// promise about what the unit can contain, and this breaks it.
    Eu02 = "EU02", Translation, Error, "illegal storage declared in a classical unit";
    /// Not every pairing of importing and imported unit kind is allowed; a
    /// classical unit cannot reach into quantum state through the back door of
    /// an import.
    Eu03 = "EU03", Translation, Error, "illegal cross-kind import";
    /// The conductor fixes which exact angles a unit can name. It must be a
    /// positive multiple of 8, so that a half turn and a quarter turn are both
    /// expressible, and it is capped so that arithmetic over it stays bounded.
    Eu04 = "EU04", Translation, Error, "conductor not a positive multiple of 8, or above the implementation limit";
    /// A unit initialiser runs once before anything else uses the unit, so it
    /// must be a function of this unit taking nothing and returning nothing.
    Eu05 = "EU05", Translation, Error, "unit initialiser name does not denote a function of this unit of type fn() -> void";
    /// An initialiser is run by the language, exactly once. Calling it by hand
    /// would run it twice.
    Eu06 = "EU06", Translation, Error, "explicit call of a unit initialiser";
    /// A `#unit any` unit is classical or quantum as whatever builds or
    /// imports it chooses. One nothing chooses, that prefers neither kind, has
    /// no kind to be translated as, and guessing one would not be strict.
    Eu07 = "EU07", Translation, Error, "kind of a `#unit any` unit not decided";
    /// Only a `#unit any` unit takes a kind from its importer or the command
    /// line. Asking a unit of fixed kind for one, even its own, is refused, so
    /// that a change of kind in the unit is noticed where it matters.
    Eu08 = "EU08", Translation, Error, "a kind chosen for a unit that is not `#unit any`";
    /// `static` restricts a declaration to its own unit. Written where there is
    /// no linkage to restrict (on a local, say), it means nothing, and
    /// accepting it silently would hide a misunderstanding. This is the one
    /// linkage diagnostic that can be raised without looking at another unit.
    El01 = "EL01", Translation, Error, "static in a position with no linkage";

    ///////////////////////////////////////
    // CONSTANTS AND CONSTANT EVALUATION //
    ///////////////////////////////////////

    /// Some positions must be computable during translation: array lengths,
    /// initialisers of unit-scope objects and `constexpr` bindings, generic
    /// const arguments. `const` also cannot be applied to a quantum type,
    /// whose state changes as operations act on it.
    Ec01 = "EC01", Translation, Error, "non-constant initialiser where a constant expression is required, or const applied to a quantum type";
    /// A `constexpr` parameter demands an argument the compiler can evaluate,
    /// not merely one that will have a value at run time.
    Ec02 = "EC02", Translation, Error, "non-constant argument supplied to a constexpr parameter";
    /// A `const` place is fixed once. Assigning to it later contradicts the
    /// property the rest of the program is entitled to rely on.
    Ec03 = "EC03", Translation, Error, "assignment to a place of const type";
    /// The expression is constant, but evaluating it would abort (e.g. dividing by
    /// zero, overflowing, indexing past the end). At run time that is a crash;
    /// during translation it is a rejection, and the abort identifier that
    /// would have fired is reported alongside.
    Ec04 = "EC04", Translation, Error, "constant evaluation would abort";
    /// A unit-scope object must get its value from somewhere: either an
    /// initialiser on the declaration, or the unit's initialiser function. With
    /// neither, it would be readable before it was ever written.
    Ec05 = "EC05", Translation, Error, "unit-scope object with no initialiser in a unit declaring no initialiser";
    /// The unit initialiser must assign each unit-duration object exactly once
    /// on every path through it (not zero times on some branch, and not twice).
    Ec06 = "EC06", Translation, Error, "unit initialiser does not assign a unit-duration object exactly once on every path";
    /// The index and the length are both known during translation, and the
    /// index is outside the array. Deferring this to a run-time abort would be
    /// reporting late what is already certain.
    Ec07 = "EC07", Translation, Error, "constant index out of bounds";
    /// Growing or shrinking an array may move it. A reference that does not own
    /// its storage cannot authorise that, because other references to the same
    /// storage would be left dangling.
    Ec08 = "EC08", Translation, Error, "reallocating operation applied through a non-owning array reference";
    /// Cloning duplicates a value. A qubit cannot be duplicated, and an opaque
    /// type has not published enough of itself for a copy to be meaningful.
    Ec09 = "EC09", Translation, Error, "clone of a qubit-bearing or opaque type";
    /// A constant placed on quantum storage, by a `quint` literal or by `^=`,
    /// that needs more qubits than the storage has. The bits that do not fit
    /// would be lost, which is the overflow the classical types abort on.
    Ec10 = "EC10", Translation, Error, "constant wider than the quantum storage it is placed on";

    /////////////////////////////
    // ANNOTATIONS AND PRAGMAS //
    /////////////////////////////

    /// An annotation this compiler does not recognise. A warning, not an error:
    /// annotations are how a program speaks to tooling, and an unknown one is
    /// more likely meant for a tool that is not this compiler than to be a
    /// mistake. It is ignored.
    Ea01 = "EA01", Translation, Warning, "unknown annotation name";
    /// A `[` at the start of an item could open an annotation or an array, and
    /// what follows does not settle it. Guessing would silently translate a
    /// program that means something else, so both readings are shown and the
    /// program is rejected.
    Ea02 = "EA02", Translation, Error, "a [ that could open either an annotation or an array";
    /// A pragma this compiler does not recognise, or one whose argument is
    /// malformed. A warning for the same reason as an unknown annotation: the
    /// pragma is ignored and translation continues.
    Ea03 = "EA03", Translation, Warning, "unknown pragma or malformed pragma argument";
    /// The predefined macro names are supplied by the compiler and describe the
    /// translation in progress. Redefining or undefining one would make them
    /// unreliable everywhere else.
    Ea04 = "EA04", Translation, Error, "#define or #undef of a predefined macro name";
    /// An annotation whose argument is not of the form it takes, such as
    /// `[export: 3]` or an `[export]` symbol a linker cannot carry.
    Ea05 = "EA05", Translation, Error, "annotation argument not of the form it takes";

    //////////////////////////
    // PREFIX COMPATIBILITY //
    //////////////////////////

    /// Two types were claimed to share a common prefix, and their layouts
    /// diverge before the claim runs out, so a read through one of them would
    /// land on a different field of the other.
    Ep01 = "EP01", Translation, Error, "prefix compatibility claimed between types that diverge";
    /// A reference reached an object through a type that is neither the
    /// object's own nor a prefix of it, so what it would read is not what is
    /// there.
    Ep02 = "EP02", Translation, Error, "access through a reference of a type neither the object's nor prefix-compatible with it";

    ///////////////////////
    // QUANTUM SEMANTICS //
    ///////////////////////

    /// A quantum value went out of scope while it might still hold state.
    /// Unlike a classical value, it cannot simply be forgotten: dropping it
    /// while entangled with something else would corrupt that other thing.
    Eq01 = "EQ01", Translation, Error, "quantum value dropped while possibly live";
    /// An ancilla must be returned to the state it was borrowed in, and no
    /// sequence of operations doing so could be found. Uncomputation is not
    /// optional bookkeeping: a dirty ancilla leaks correlation into everything
    /// that later borrows it.
    Eq02 = "EQ02", Translation, Error, "ancilla uncomputation impossible";
    /// A quantum-controlled region must have a statically known extent, because
    /// the controlled version of it has to be built during translation. Control
    /// over an unbounded region cannot be.
    Eq03 = "EQ03", Translation, Error, "unbounded quantum control";
    /// A branch on a quantum value takes every branch at once, so both arms
    /// must be liftable: the same effect shape, no measurement, no classical
    /// side effect, nothing that only makes sense if one arm alone ran.
    Eq04 = "EQ04", Translation, Error, "lifting rule violation in a quantum if or match";
    /// Forgetting a value discards information. Inside a controlled region that
    /// is not a local act: it would decohere the control.
    Eq05 = "EQ05", Translation, Error, "forget in a controlled region";
    /// A measurement outcome is classical, but only in the places the language
    /// can account for: feeding forward into later operations, a kernel, or a
    /// lift. Used elsewhere it would smuggle a timing dependency into code that
    /// looks pure.
    Eq06 = "EQ06", Translation, Error, "outcome value used outside feed-forward, kernel or lift";
    /// A declared pair grouping says which qubits are considered together.
    /// Re-associating it implicitly would change the operator's meaning while
    /// leaving the source looking identical.
    Eq07 = "EQ07", Translation, Error, "implicit re-association of a declared pair grouping";
    /// The caller needs a circuit fixed at translation time, and the operator it
    /// invokes is only determined at run time.
    Eq08 = "EQ08", Translation, Error, "caller requiring a static circuit invokes a dynamic operator";
    /// Reading a map could mean a coherent lookup that leaves the key in
    /// superposition, or a measurement that collapses it. These are different
    /// programs, so the access must say which: write `query` or `measure`.
    Eq09 = "EQ09", Translation, Error, "ambiguous map access: write query or measure";
    /// The state asks for an angle the unit's conductor cannot express exactly.
    /// Approximating it silently would make the program mean something slightly
    /// different from what it says.
    Eq10 = "EQ10", Translation, Error, "state not exactly synthesisable at the unit conductor";
    /// A quantum unit's initialiser runs before the quantum machine is ready, so
    /// it may not touch quantum resources; and computation performed while the
    /// circuit is generated must not have effects that outlive generation.
    Eq11 = "EQ11", Translation, Error, "quantum-unit initialiser touches quantum resources, or generation-time computation with a run-time side effect";
    /// A matrix applied to qubits must be exact. Floating point cannot state
    /// that a matrix is unitary, only that it is nearly so, and nearly unitary
    /// is not unitary.
    Eq12 = "EQ12", Translation, Error, "floating-point matrix in apply";
    /// The matrix is exact but not unitary, so it does not describe anything a
    /// quantum machine can do.
    Eq13 = "EQ13", Translation, Error, "matrix supplied to apply is not unitary";
    /// Code run while a circuit is generated that does not finish within the
    /// number of steps generation allows, as a loop that never ends would.
    Eq14 = "EQ14", Translation, Error, "circuit generation did not finish";
    /// One qubit named twice among the operands of one operation, which no
    /// gate can act on.
    Eq15 = "EQ15", Translation, Error, "one qubit named twice by one operation";
    /// A value known only when the circuit runs (a classical parameter of
    /// an entry operator, or a measured value) used where the circuit's
    /// structure needs a value when it is generated.
    Eq16 = "EQ16", Translation, Error, "run-time value where the circuit's structure needs one";
    /// The adjoint of an operator that has none, because it measures,
    /// forgets or prepares; or a declared adjoint that does not undo its
    /// operator.
    Eq17 = "EQ17", Translation, Error, "adjoint of an operator that has none, or not its adjoint";
    /// An operator applied to quantum data that has no reversible meaning
    /// there: `&=`, `|=`, a shift, `++` on a qubit, or multiplication by an
    /// even number. What it would do discards information, which no circuit
    /// of unitary gates can.
    Eq18 = "EQ18", Translation, Error, "irreversible operator on quantum data";
    /// Arithmetic on a `quint` written with `+`, `-`, `*`, their assignment
    /// forms, `++` or `--`. On qubits arithmetic can only wrap, and Topiq
    /// does not wrap silently, so it is written `wrapping_add` and so on.
    Eq19 = "EQ19", Translation, Error, "quantum arithmetic written without `wrapping_`";
    /// `t ^= e` where `e` reads a qubit of `t`. The flip of `t` would be
    /// controlled on `t` itself, which is not an operation on qubits.
    Eq20 = "EQ20", Translation, Error, "`^=` on quantum data that its operand reads";

    ///////////////////
    //
    // Judgements:
    //   Covers, gauges, contracts, frames, holonomy.
    //
    ///////////////////

    /// The judgement machinery could not determine this field, and something
    /// demands a definite value for it. Unknown never passes: treating "could
    /// not tell" as "fine" is how an unsound claim gets published.
    Ej01 = "EJ01", Translation, Error, "judgment field Unknown where an expectation demands a value";
    /// Replay re-runs an operator's construction, which only makes sense if the
    /// operator is monic and its arguments are fixed when the circuit is built.
    Ej02 = "EJ02", Translation, Error, "replay on a non-monic operator, or with non-generation-time arguments";
    /// The operator's derived contract does not match the one claimed for it.
    /// The witness showing where the two part is attached.
    Ej03 = "EJ03", Translation, Error, "contract expectation failed";
    /// The coefficient names an angle the unit's conductor cannot represent.
    /// The diagnostic names a conductor that would suffice.
    Ej04 = "EJ04", Translation, Error, "QUON coefficient outside the unit conductor";
    /// The operator does not preserve the declared frame. The witness (the
    /// product or entangled state on which it departs) is attached.
    Ej05 = "EJ05", Translation, Error, "frame expectation failed";
    /// Local judgements agreed on each patch of the cover but disagree where
    /// patches overlap, so they cannot be glued into one global claim. The
    /// schedule is attached, so this reports where the seams are rather than
    /// merely that there are some.
    Ej06 = "EJ06", Translation, Error, "glue failed: constants differ across nerve components";
    /// The accumulated phase around a closed chain is not the one claimed. The
    /// stage responsible is named.
    Ej07 = "EJ07", Translation, Error, "holonomy expectation failed";
    /// A single fiducial state was offered as the gauge, but the cover admits no
    /// pregauge built from it: there is no consistent choice of frame across
    /// the cover starting from that one state.
    Ej09 = "EJ09", Translation, Error, "single-fiducial gauge declared on a cover admitting no pregauge";
    /// A gauge is built by projecting each cover point onto the fiducial, which
    /// requires every point to have a component along it. The orthogonal point
    /// is reported.
    Ej10 = "EJ10", Translation, Error, "fiducial condition violated: a cover point is orthogonal to the fiducial";
    /// Restricting one base type by another needs them to overlap and to agree
    /// on width; otherwise the restriction denotes nothing.
    Ej11 = "EJ11", Translation, Error, "restriction of base types with empty overlap or mismatched width";
    /// A locale gathers members that its container exposes. This member is not
    /// one of them.
    Ej12 = "EJ12", Translation, Error, "locale member is not an interface of its container";
    /// An estimated phase comes from measurement: it has error bars. An
    /// expectation is a claim the compiler proves, and a measurement cannot
    /// discharge one.
    Ej13 = "EJ13", Translation, Error, "an estimated phase used to satisfy an expectation";
    /// A state whose amplitudes' squared magnitudes do not sum to one.
    Ej14 = "EJ14", Translation, Error, "state that is not a unit vector";
    /// A cover whose presentation is not well formed: a point or ket given
    /// twice, points of different widths, or code generators that do not
    /// commute or are not independent.
    Ej15 = "EJ15", Translation, Error, "cover whose presentation is not well formed";
    /// A judgement whose derivation left the checked fragment, its affected
    /// fields `Unknown`.
    Ej16 = "EJ16", Translation, Warning, "judgment left the checked fragment";
    /// A step outside the fragment the unit declares with
    /// `#pragma fragment(…)`.
    Ej17 = "EJ17", Translation, Error, "step outside the fragment the unit declares";
    /// An operator whose body reads a judgement that depends on its own
    /// circuit, through the operators whose judgements it reads: the
    /// circuit would be needed before it exists.
    Ej18 = "EJ18", Translation, Error, "judgment read by what it judges";
    /// `qcopy` of a value whose state the operator does not know: one it
    /// was given, one joined to a qubit outside it, or one whose history
    /// measures, forgets or depends on a measured value. A copy is a second
    /// preparation, so the whole preparation must be known and repeatable.
    Ej19 = "EJ19", Translation, Error, "`qcopy` of a state that was not prepared here";

    /////////////////////////
    // TYPED DATA NOTATION //
    /////////////////////////

    /// An embedded data document does not match the type it is being read into.
    Et01 = "ET01", Translation, Error, "TCON schema mismatch";
    /// Serialising writes a value out as data. A qubit cannot be written out, a
    /// reference would not mean the same thing when read back, and an opaque
    /// type has not published its contents.
    Et02 = "ET02", Translation, Error, "attempt to serialise a qubit-bearing, reference or opaque type";

    //////////////////////
    // STATIC SEMANTICS //
    //////////////////////

    /// Decoding is the first thing translation does, and a file that is not
    /// well-formed UTF-8 cannot be decoded.
    Es01 = "ES01", Translation, Error, "source is not well-formed UTF-8";
    /// Text that does not parse, and so never became a program.
    Es02 = "ES02", Translation, Error, "syntax error";
    /// A message the program itself asks for.
    Es03 = "ES03", Translation, Error, "diagnostic requested by #error or #warning";
    /// A name that denotes nothing visible from where it is used: misspelt,
    /// declared in a scope that has already ended, or declared further down the
    /// same block.
    Es04 = "ES04", Translation, Error, "name not found";
    /// Two declarations of one name in a scope where only one can stand. A later
    /// `let` in a block shadows an earlier one and is not this; two functions
    /// of the same name at unit scope are.
    Es05 = "ES05", Translation, Error, "name declared twice in the same scope";
    /// A value of one type where another is required, an operator applied to
    /// operands it does not accept, or a literal that does not fit the type it
    /// was given.
    Es06 = "ES06", Translation, Error, "types do not match";
    /// A call that does not fit what it calls: the wrong number of arguments,
    /// or something that is not a function.
    Es07 = "ES07", Translation, Error, "call does not match its callee";
    /// Assigning to a value that is not a place: the result of a call, or a
    /// temporary. (Changing a constant, or what a `*const` reference refers
    /// to, is EC03.)
    Es08 = "ES08", Translation, Error, "assignment to something that is not a place";
    /// `break` and `continue` only have meaning inside a loop, and `break` with
    /// a value only inside `loop`, whose value it becomes.
    Es09 = "ES09", Translation, Error, "break or continue outside a loop that accepts it";
    /// A structure or enumeration value that was moved (passed on, assigned
    /// elsewhere) used again. It has a new owner, so the old binding no longer
    /// holds anything.
    Es10 = "ES10", Translation, Error, "use of a value after it was moved";
    /// A binding declared without a value, read on some path before any value
    /// was assigned to it.
    Es11 = "ES11", Translation, Error, "use of a binding before it is given a value";
    /// A value that cannot be copied, taken out of a place that must keep it:
    /// an array element, the referent of a reference, or a unit-scope object,
    /// or copied into several places at once.
    Es12 = "ES12", Translation, Error, "value moved out of a place that must keep it, or copied though its type cannot be";
    /// A `match` with a value no arm accepts. The diagnostic names one such
    /// value.
    Es13 = "ES13", Translation, Error, "match does not cover every value";
    /// A `match` arm that can never be chosen, because earlier arms accept
    /// everything it would.
    Es14 = "ES14", Translation, Warning, "match arm can never be reached";
    /// A structure literal that leaves a field without a value. Every field
    /// must be given one; there is no default.
    Es15 = "ES15", Translation, Error, "structure literal is missing a field";
    /// A structure or enumeration that contains itself by value, and so would
    /// be infinitely large.
    Es16 = "ES16", Translation, Error, "type contains itself";
    /// An `import` naming a unit that is neither among the sources being
    /// compiled nor found on the unit path.
    Es17 = "ES17", Translation, Error, "imported unit not found";
    /// A standard annotation written on something it does not apply to, such
    /// as `[packed]` on a function.
    Es18 = "ES18", Translation, Error, "annotation written where it does not apply";
    /// A method added to another unit's type that its unit did not mark
    /// `[open]`, which is what allows other units to add methods to it.
    Es19 = "ES19", Translation, Error, "method added to another unit's type that is not `[open]`";
    /// A name the library reserves: one beginning with `__`, declared by a
    /// program, or a unit given the name of a library unit or one set aside
    /// for later editions.
    Es20 = "ES20", Translation, Error, "name reserved for the library";
    /// A use of an item marked `[deprecated: "msg"]`, repeating its message.
    Es21 = "ES21", Translation, Warning, "use of an item marked deprecated";
    /// `++` or `--` on something that is not an integer. Stepping by one has
    /// a meaning only for integers.
    Es22 = "ES22", Translation, Error, "`++` or `--` on a type that is not an integer";
    /// `quint<N>` with no qubits, or with more than the 64 the widest
    /// unsigned integer can hold when it is measured.
    Es23 = "ES23", Translation, Error, "`quint` width outside 1 to 64";
    /// `measure` of classical data, which gives it back as it is. It is
    /// accepted so that one routine can serve a unit of either kind, and
    /// warned of, since measuring is meant to be guarded to the quantum unit.
    Es24 = "ES24", Translation, Warning, "`measure` on classical data";

    /////////////////////////
    // OTHER STATIC CHECKS //
    /////////////////////////

    /// A generic was instantiated with arguments its `where` clause rejects.
    /// The failed condition is printed with the values that made it false.
    Em01 = "EM01", Translation, Error, "where clause false at instantiation";
    /// An operator-method name introduced with `$` that is not one of the
    /// operators the language defines.
    Em02 = "EM02", Translation, Error, "unknown $ operator-method name";
    /// The type asked for its run-time description has opted out of carrying
    /// one, so there is nothing to return.
    Em03 = "EM03", Translation, Error, "@typeinfo on a `[no_typeinfo]` type";
    /// Target capabilities are a closed list, so that a misspelt capability is
    /// caught rather than quietly reported absent forever.
    Em04 = "EM04", Translation, Error, "@target_has with a capability name outside the closed list";
    /// An implementation limit was exceeded (nesting depth, field count,
    /// register width and the rest). The limit and its value are named. Limits
    /// are enforced rather than quietly exceeded, because the alternative is a
    /// program that translates and then misbehaves.
    Em05 = "EM05", Translation, Error, "an implementation limit exceeded";

    /////////////
    // LINKAGE //
    /////////////

    /// Two units describe the same imported item with different types. One of
    /// them was compiled against a version of the other that no longer exists.
    El02 = "EL02", Link, Error, "two units disagreeing on the declared type of an imported item";
    /// Two units share a type or operator but disagree about its layout or its
    /// judgement record, so code compiled against one would misread the other.
    El03 = "EL03", Link, Error, "two units disagreeing on the layout digest or judgment record of a shared type or operator";
    /// Unit initialisers run in dependency order, which a cycle makes
    /// impossible: each would need the other to have run first.
    El04 = "EL04", Link, Error, "cycle in the import relation among units declaring initialisers";
    /// A program starts in exactly one place.
    El05 = "EL05", Link, Error, "no main, or more than one";
    /// Something is referred to and never defined, or defined twice where only
    /// one definition can be chosen.
    El06 = "EL06", Link, Error, "unresolved item, or two definitions of one item with program linkage";
    /// Two units each add a method of the same name to the same open type, so
    /// the type would have two different methods with one name.
    El07 = "EL07", Link, Error, "two units adding the same external method to one `[open]` type";
    /// The program can reach a lift, and the target cannot perform one. This is
    /// never worked around by unrolling it statically: that would be a different
    /// circuit, silently substituted.
    El08 = "EL08", Link, Error, "reachable lift on a target without dynamic_lifting";
    /// A register whose length is only known at run time, on a target that
    /// requires register widths to be fixed when the circuit is built.
    El09 = "EL09", Link, Error, "register of run-time length on a target without dynamic_registers";
    /// Two units whose judgements meet use conductors that cannot both be
    /// honoured. A conductor that would suffice for both is named.
    El10 = "EL10", Link, Error, "incompatible conductors between units whose judgments meet";
    /// A unit's metadata was written by a compiler for a different edition of
    /// the language, so its records cannot be interpreted with confidence.
    El11 = "EL11", Link, Error, "metadata written against another edition of the language";
    /// A unit was compiled against a constant of another unit whose value has
    /// since changed, so the value it folded in is stale.
    El12 = "EL12", Link, Error, "imported constant changed since the importer was compiled";
    /// A contract that depends on a convention (a cover and a gauge) was
    /// published beyond its unit without them, so no other unit can tell what it
    /// claims. Detected at link time, since it is about what crosses the
    /// boundary.
    Ej08 = "EJ08", Link, Error, "convention-dependent contract published across a unit boundary without its cover and gauge";

    /////////////////////
    // RUN-TIME ABORTS //
    /////////////////////

    /// Integer arithmetic left the range of its type. It aborts rather than
    /// wrapping, so that a program never silently continues with a wrong number.
    Ra01 = "RA01", Runtime, Error, "integer arithmetic overflow";
    /// Division or remainder by zero, which has no value.
    Ra02 = "RA02", Runtime, Error, "division or remainder by zero";
    /// A conversion between integer types whose value does not fit the target.
    Ra03 = "RA03", Runtime, Error, "integer conversion out of range of the target type";
    /// A float being converted to an integer was NaN, infinite, or outside the
    /// integer type's range, none of which has an integer to convert to.
    Ra04 = "RA04", Runtime, Error, "float-to-integer conversion of NaN, infinity or an out-of-range value";
    /// A shift by at least the width of the value.
    Ra05 = "RA05", Runtime, Error, "shift count not less than the width of the shifted type";
    /// An index outside the bounds of what it indexes.
    Ra06 = "RA06", Runtime, Error, "index out of bounds";
    /// Memory could not be obtained for a growing collection.
    Ra07 = "RA07", Runtime, Error, "allocation failure in reserve or push";
    /// A dynamic operation named a field or method that the value's run-time
    /// type description does not list.
    Ra08 = "RA08", Runtime, Error, "dyn operation on a value whose TypeInfo lists no such field or method";
    /// An exact-scalar operation produced an angle outside the conductor. Only
    /// reachable where the operands were not known during translation.
    Ra09 = "RA09", Runtime, Error, "exact-scalar operation outside the conductor at run time";
    /// A deliberate abort: unwrapping an absent value, or a call to panic.
    Ra10 = "RA10", Runtime, Error, "unwrap or expect on Err or None, or core::panic";
    /// A dynamic invocation whose argument or return types do not match the
    /// signature of what was invoked.
    Ra11 = "RA11", Runtime, Error, "invoke signature mismatch";

    ///////////////////////////////
    // THIS IMPLEMENTATION'S OWN //
    ///////////////////////////////

    /// A construct the compiler does not translate: the few quantum
    /// constructs `sema` lists, and code written without the `core` library.
    Tq003 = "TQ003", Translation, Error, "construct this compiler does not translate";
    /// The compiler itself failed: LLVM rejected the code it generated, or the
    /// toolchain it relies on is missing or broken. Always a fault in the
    /// compiler or its installation, never in the program.
    Tq011 = "TQ011", Translation, Error, "internal compiler error";
}

impl Code {
    /// Looks an identifier up by its string form, for example `"EU01"`.
    ///
    /// The comparison is case-sensitive, matching how identifiers are written
    /// in messages.
    pub fn from_id(id: &str) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| c.id() == id)
    }

    /// Whether this identifier is part of the language, meaning the same in
    /// every Topiq compiler, rather than one of this implementation's own.
    pub fn is_language_defined(self) -> bool {
        !self.id().starts_with("TQ")
    }

    /// Whether translation continues past this diagnostic.
    pub fn is_warning(self) -> bool {
        self.severity() == Severity::Warning
    }
}

impl std::fmt::Display for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn identifiers_are_unique() {
        let mut seen = HashSet::new();
        for c in Code::ALL {
            assert!(seen.insert(c.id()), "duplicate identifier {}", c.id());
        }
        assert_eq!(seen.len(), Code::ALL.len());
    }

    #[test]
    fn every_identifier_round_trips_through_from_id() {
        for &c in Code::ALL {
            assert_eq!(Code::from_id(c.id()), Some(c));
        }
        assert_eq!(Code::from_id("EU99"), None);
        assert_eq!(Code::from_id("eu01"), None, "lookup is case-sensitive");
    }

    #[test]
    fn every_message_is_present_and_lowercase() {
        for &c in Code::ALL {
            let m = c.message();
            assert!(!m.is_empty(), "{c} has no message");
            // a message is the clause that follows "error[EU01]: ", so it
            // starts lower case. an acronym such as QUON or TCON is allowed;
            // sentence case (a capital followed by a lower-case letter) is not
            let mut cs = m.chars();
            let (first, second) = (cs.next().unwrap(), cs.next().unwrap_or('X'));
            assert!(
                !(first.is_uppercase() && second.is_lowercase()),
                "{c} message reads as a sentence rather than a clause: {m}"
            );
        }
    }

    #[test]
    fn the_only_warnings_are_the_ignored_constructs_and_three_more() {
        // unknown annotations and pragmas are ignored with a warning, as are
        // an unreachable arm, a deprecated use, and a judgement that left the
        // checked fragment
        let warnings: Vec<_> = Code::ALL
            .iter()
            .copied()
            .filter(|c| c.is_warning())
            .collect();
        assert_eq!(warnings, vec![Code::Ea01, Code::Ea03, Code::Ej16, Code::Es14, Code::Es21, Code::Es24]);
    }

    #[test]
    fn every_runtime_abort_uses_the_ra_prefix() {
        for &c in Code::ALL {
            assert_eq!(
                c.phase() == Phase::Runtime,
                c.id().starts_with("RA"),
                "{c} phase and prefix disagree"
            );
        }
    }

    #[test]
    fn the_translation_and_link_split_is_respected() {
        // `EL01` is about a `static` that restricts nothing, which is visible
        // within one unit. every other linkage check needs a second unit
        assert_eq!(Code::El01.phase(), Phase::Translation);
        for c in [
            Code::El02,
            Code::El03,
            Code::El04,
            Code::El05,
            Code::El06,
            Code::El07,
            Code::El08,
            Code::El09,
            Code::El10,
            Code::El11,
            Code::El12,
            Code::Ej08,
        ] {
            assert_eq!(c.phase(), Phase::Link, "{c} should be link-time");
        }
    }

    #[test]
    fn ej08_is_the_only_judgment_identifier_checked_at_link_time() {
        let link_ej: Vec<_> = Code::ALL
            .iter()
            .copied()
            .filter(|c| c.id().starts_with("EJ") && c.phase() == Phase::Link)
            .collect();
        assert_eq!(link_ej, vec![Code::Ej08]);
    }

    #[test]
    fn the_judgment_family_is_numbered_with_a_gap() {
        // `EJ08` is absent from the translation-time identifiers because it is
        // checked at link time; the gap is deliberate
        let translation_ej: Vec<&str> = Code::ALL
            .iter()
            .copied()
            .filter(|c| c.id().starts_with("EJ") && c.phase() == Phase::Translation)
            .map(|c| c.id())
            .collect();
        assert!(!translation_ej.contains(&"EJ08"));
        assert!(translation_ej.contains(&"EJ07"));
        assert!(translation_ej.contains(&"EJ09"));
    }

    #[test]
    fn only_tq003_and_tq011_are_this_implementations_own() {
        let own: Vec<_> = Code::ALL.iter().copied().filter(|c| !c.is_language_defined()).collect();
        assert_eq!(own, vec![Code::Tq003, Code::Tq011]);
    }

    #[test]
    fn the_language_defined_table_has_the_expected_size() {
        // 8 EU and EL01, 10 EC, 5 EA, 2 EP, 20 EQ, 19 EJ, 2 ET, 24 ES, 5 EM,
        // 11 link EL, 11 RA. a change here is a change to the language
        let defined = Code::ALL.iter().filter(|c| c.is_language_defined()).count();
        assert_eq!(defined, 118);
    }

    #[test]
    fn counts_per_prefix_are_unchanged() {
        let count = |p: &str| {
            Code::ALL
                .iter()
                .filter(|c| c.id().starts_with(p) && c.is_language_defined())
                .count()
        };
        assert_eq!(count("EU"), 8);
        assert_eq!(count("EL"), 12); // EL01 at translation time, EL02..EL12 at link time
        assert_eq!(count("EC"), 10);
        assert_eq!(count("EA"), 5);
        assert_eq!(count("EP"), 2);
        assert_eq!(count("EQ"), 20);
        assert_eq!(count("EJ"), 19); // EJ01..EJ07 and EJ09..EJ19, plus link-time EJ08
        assert_eq!(count("ET"), 2);
        assert_eq!(count("ES"), 24);
        assert_eq!(count("EM"), 5);
        assert_eq!(count("RA"), 11);
    }

    #[test]
    fn phase_and_severity_names_render() {
        assert_eq!(Phase::Translation.name(), "translation");
        assert_eq!(Phase::Link.name(), "link");
        assert_eq!(Phase::Runtime.name(), "runtime");
        assert_eq!(Severity::Error.name(), "error");
        assert_eq!(Severity::Warning.name(), "warning");
        assert_eq!(Code::Eu01.to_string(), "EU01");
    }
}
