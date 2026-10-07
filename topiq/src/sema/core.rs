//! What the compiler provides for the `core` library.
//!
//! Most of `core` is Topiq, in the library unit `core.tq` that every unit
//! sees without importing it. `print`, `eprint`, `exit`, `panic` and `abort`
//! are provided here instead: they talk to the operating system, which no
//! Topiq code can reach directly, and an abort names the line that called it,
//! which a function written in the library could not. So is `adjoint`, whose
//! result type depends on its operand's: of an operator it is the operator
//! that undoes it, and of a value whose type defines `$adj`, that, as a
//! vector's adjoint is a covector. Each is visible unqualified and as
//! `core::name`. A unit may declare its own function of the same name, which
//! then hides the library one for unqualified use; `core::name` still reaches
//! the library's. `print`, `eprint`, `exit` and `panic` are also values of
//! function type, as `let p = print;` makes one.
//!
//! The library's own units also reach the operations they are built from,
//! by names beginning with `__`, which no program may use: arithmetic that
//! wraps, clamps or truncates; run-time type information and dynamic values;
//! float formatting and parsing; and the system calls behind `io`, `fs` and
//! `module`.
//!
//! # Functions that never return
//!
//! `exit` and `panic` end the program, so control never comes back from a
//! call to one. Their calls have type `never`, like `return`, which lets a
//! function end in one without inventing a value to return afterwards:
//!
//! ```text
//! fn get(i: usize) -> i32 {
//!     if i < 3 { i as i32 } else { panic("index out of range") }
//! }
//! ```

use crate::tir::{Access, BinOp, FloatTy, Intrinsic, IntTy, Ty, TypeTable};

/// Every name `core` provides, for suggesting a correction.
pub const NAMES: [&str; 6] = ["print", "eprint", "exit", "panic", "abort", "adjoint"];

/// The library function an unqualified name reaches when nothing in the unit
/// is called that.
pub fn prelude(name: &str) -> Option<Intrinsic> {
    lookup(name)
}

/// The library function `core::name`.
pub fn lookup(name: &str) -> Option<Intrinsic> {
    Some(match name {
        "print" => Intrinsic::Print,
        "eprint" => Intrinsic::Eprint,
        "exit" => Intrinsic::Exit,
        "panic" => Intrinsic::Panic,
        // which abort is only settled where it is called
        "abort" => Intrinsic::Abort(0),
        "adjoint" => Intrinsic::Adjoint,
        _ => return None,
    })
}

/// What a name reaches in a library unit's own code, beyond what every unit
/// sees: the operations the library is built from. Their names begin with
/// `__`, which no program may use.
pub fn internal(name: &str) -> Option<Intrinsic> {
    Some(match name {
        "__overflowing_add" => Intrinsic::Overflowing(BinOp::Add),
        "__overflowing_sub" => Intrinsic::Overflowing(BinOp::Sub),
        "__overflowing_mul" => Intrinsic::Overflowing(BinOp::Mul),
        "__overflowing_div" => Intrinsic::Overflowing(BinOp::Div),
        "__overflowing_rem" => Intrinsic::Overflowing(BinOp::Rem),
        "__overflowing_shl" => Intrinsic::Overflowing(BinOp::Shl),
        "__overflowing_shr" => Intrinsic::Overflowing(BinOp::Shr),
        "__overflowing_neg" => Intrinsic::OverflowingNeg,
        "__saturating_add" => Intrinsic::Saturating(BinOp::Add),
        "__saturating_sub" => Intrinsic::Saturating(BinOp::Sub),
        "__saturating_mul" => Intrinsic::Saturating(BinOp::Mul),
        "__truncate" => Intrinsic::Truncate,
        "__saturating_as" => Intrinsic::SaturatingAs,
        "__clone" => Intrinsic::Clone,
        "__exchange" => Intrinsic::Exchange,
        // the type is the argument's, settled where it is called
        "__dyn_of" => Intrinsic::DynOf(Ty::Never),
        "__dyn_take" => Intrinsic::DynTake,
        "__dyn_view" => Intrinsic::DynView,
        "__field_at" => Intrinsic::FieldAt,
        "__dyn_call" => Intrinsic::DynCall,
        "__dyn_make" => Intrinsic::DynMake,
        "__dyn_store" => Intrinsic::DynStore,
        "__is_null" => Intrinsic::IsNull,
        "__circuit_of" => Intrinsic::CircuitOf,
        "__circuit_doc" => Intrinsic::CircuitDoc,
        "__grow_zeroed" => Intrinsic::GrowZeroed,
        "__float_digits" => Intrinsic::FloatDigits,
        "__parse_float" => Intrinsic::ParseFloat,
        "__sin" => Intrinsic::Sin,
        "__cos" => Intrinsic::Cos,
        "__file_open" => Intrinsic::FileOpen,
        "__file_size" => Intrinsic::FileSize,
        "__file_read" => Intrinsic::FileRead,
        "__file_write" => Intrinsic::FileWrite,
        "__file_close" => Intrinsic::FileClose,
        "__std_handle" => Intrinsic::StdHandle,
        "__is_console" => Intrinsic::IsConsole,
        "__console_read" => Intrinsic::ConsoleRead,
        "__module_open" => Intrinsic::ModuleOpen,
        "__module_find" => Intrinsic::ModuleFind,
        "__module_close" => Intrinsic::ModuleClose,
        "__call_void" => Intrinsic::CallVoid,
        "__host_descriptor" => Intrinsic::HostDescriptor,
        "__clock" => Intrinsic::Clock,
        _ => return crate::tir::Gate::ALL.into_iter().find(|g| g.intrinsic_name() == name).map(Intrinsic::Gate),
    })
}

/// What an unqualified name reaches in a quantum unit when nothing in the
/// unit is called that: `apply`, which applies an exact matrix as a gate, and
/// `controlled`, which makes a new operator of one.
pub fn quantum_prelude(name: &str) -> Option<Intrinsic> {
    Some(match name {
        "apply" => Intrinsic::Apply,
        "controlled" => Intrinsic::Controlled,
        _ => return None,
    })
}

/// Whether an intrinsic's operand types are whatever it is given, checked
/// where it is called, rather than a fixed signature.
pub fn is_generic(which: Intrinsic) -> bool {
    which.is_dyn()
        || matches!(
            which,
            Intrinsic::Overflowing(_)
                | Intrinsic::OverflowingNeg
                | Intrinsic::Saturating(_)
                | Intrinsic::Truncate
                | Intrinsic::SaturatingAs
                | Intrinsic::Clone
                | Intrinsic::Exchange
                | Intrinsic::Gate(crate::tir::Gate::Rz | crate::tir::Gate::GPhase)
                | Intrinsic::Apply
                | Intrinsic::Adjoint
                | Intrinsic::Controlled
                | Intrinsic::Then
        )
}

/// A library function's parameter types and result type.
///
/// # Panics
///
/// For [`Intrinsic::Len`], which is a method, not a function.
pub fn signature(which: Intrinsic, types: &mut TypeTable) -> (Vec<Ty>, Ty) {
    match which {
        // text is only read, so a `*const [char]` will do
        Intrinsic::Print | Intrinsic::Eprint => (vec![types.slice(Access::Const, Ty::Char)], Ty::Void),
        Intrinsic::Exit => (vec![Ty::Int(IntTy::I32)], Ty::Never),
        Intrinsic::Panic | Intrinsic::Abort(_) => (vec![types.slice(Access::Const, Ty::Char)], Ty::Never),
        Intrinsic::Len => panic!("`len` is called as a method"),
        Intrinsic::GrowZeroed => {
            let place = types.reference(Access::Write, Ty::Void);
            (vec![place, Ty::USIZE], place)
        }
        Intrinsic::FloatDigits => {
            let buf = types.array(Ty::Int(IntTy::U8), 40);
            let buf = types.reference(Access::Write, buf);
            (vec![Ty::Float(FloatTy::F64), Ty::Int(IntTy::I32), buf], Ty::Int(IntTy::I32))
        }
        Intrinsic::ParseFloat => {
            let text = types.slice(Access::Write, Ty::Int(IntTy::U8));
            (vec![text], Ty::Float(FloatTy::F64))
        }
        Intrinsic::Sin | Intrinsic::Cos => (vec![Ty::Float(FloatTy::F64)], Ty::Float(FloatTy::F64)),
        Intrinsic::FileOpen => {
            let path = types.slice(Access::Write, Ty::Int(IntTy::U16));
            (vec![path, Ty::Bool], Ty::Int(IntTy::I64))
        }
        Intrinsic::FileSize => (vec![Ty::Int(IntTy::I64)], Ty::Int(IntTy::I64)),
        Intrinsic::FileRead => {
            let buf = types.slice(Access::Write, Ty::Int(IntTy::U8));
            (vec![Ty::Int(IntTy::I64), buf], Ty::Int(IntTy::I64))
        }
        Intrinsic::FileWrite => {
            let data = types.slice(Access::Write, Ty::Int(IntTy::U8));
            (vec![Ty::Int(IntTy::I64), data], Ty::Int(IntTy::I64))
        }
        Intrinsic::FileClose => (vec![Ty::Int(IntTy::I64)], Ty::Void),
        Intrinsic::StdHandle => (vec![Ty::Int(IntTy::U32)], Ty::Int(IntTy::I64)),
        Intrinsic::IsConsole => (vec![Ty::Int(IntTy::I64)], Ty::Bool),
        Intrinsic::ConsoleRead => {
            let units = types.slice(Access::Write, Ty::Int(IntTy::U16));
            (vec![Ty::Int(IntTy::I64), units, Ty::Bool], Ty::Int(IntTy::I64))
        }
        Intrinsic::ModuleOpen => {
            let path = types.slice(Access::Write, Ty::Int(IntTy::U16));
            (vec![path], Ty::Int(IntTy::I64))
        }
        Intrinsic::ModuleFind => {
            let name = types.slice(Access::Write, Ty::Int(IntTy::U8));
            (vec![Ty::Int(IntTy::I64), name], Ty::USIZE)
        }
        Intrinsic::ModuleClose => (vec![Ty::Int(IntTy::I64)], Ty::Bool),
        Intrinsic::CallVoid => (vec![Ty::USIZE], Ty::Void),
        Intrinsic::HostDescriptor => (vec![], Ty::USIZE),
        Intrinsic::Clock => (vec![], Ty::Int(IntTy::U64)),
        Intrinsic::Gate(g) if !g.takes_phase() => {
            let handle = types.reference(Access::Write, Ty::Qubit);
            (vec![handle; g.arity()], Ty::Void)
        }
        Intrinsic::Node => (vec![Ty::USIZE], types.reference(Access::Write, Ty::Qubit)),
        other => panic!("`{}` is checked where it is called", other.name()),
    }
}

/// What each library function does, for a message about it.
pub fn describe(which: Intrinsic) -> &'static str {
    match which {
        Intrinsic::Print => "writes its text to standard output",
        Intrinsic::Eprint => "writes its text to standard error",
        Intrinsic::Exit => "ends the program with the given status",
        Intrinsic::Panic => "aborts the program, reporting its text",
        Intrinsic::Len => "gives the number of elements",
        Intrinsic::Abort(_) => "aborts the program, reporting its identifier and text",
        Intrinsic::Overflowing(_) | Intrinsic::OverflowingNeg => "computes a result wrapped to its type, and whether it wrapped",
        Intrinsic::Saturating(_) => "computes a result clamped to its type's range",
        Intrinsic::Truncate => "keeps an integer's low bits",
        Intrinsic::SaturatingAs => "converts to an integer type, clamping to its range",
        Intrinsic::Clone => "makes a deep copy of what a reference refers to",
        Intrinsic::Exchange => "exchanges the values two references refer to",
        Intrinsic::TypeInfo(_) => "gives a type's run-time type information",
        Intrinsic::TypeInfoOf => "gives the run-time type information a reference carries",
        Intrinsic::Downcast(_) => "checks what a reference refers to against a type",
        Intrinsic::Invoke(_) => "calls a function address after checking its signature",
        Intrinsic::DynOf(_) => "moves a value into a `dyn`",
        Intrinsic::DynAs(_) => "takes the value out of a `dyn` if it is of the type given",
        Intrinsic::DynTake => "moves the value out of a `dyn`",
        Intrinsic::DynView => "refers to the value a `dyn` holds",
        Intrinsic::FieldAt => "refers to a part of what a reference refers to",
        Intrinsic::DynCall => "calls a method with `dyn` arguments",
        Intrinsic::DynMake => "makes a `dyn` of zero bytes",
        Intrinsic::DynStore => "moves the value a `dyn` holds into a place",
        Intrinsic::IsNull => "says whether an address is null",
        Intrinsic::CircuitOf => "makes a circuit handle of a circuit's document",
        Intrinsic::CircuitDoc => "gives the document of the circuit a handle refers to",
        Intrinsic::RawParts => "takes a reference apart into its address and its length or table",
        Intrinsic::FromRawParts => "makes a reference from an address and a length or table",
        Intrinsic::GrowZeroed => "gives an empty growable array elements of zero bytes",
        Intrinsic::FloatDigits => "writes a floating value's leading decimal digits",
        Intrinsic::ParseFloat => "reads a decimal numeral as a floating value",
        Intrinsic::Sin => "gives the sine of an angle in radians",
        Intrinsic::Cos => "gives the cosine of an angle in radians",
        Intrinsic::FileOpen => "opens a file for reading or writing",
        Intrinsic::FileSize => "gives the size of an open file",
        Intrinsic::FileRead => "reads from an open file",
        Intrinsic::FileWrite => "writes to an open file",
        Intrinsic::FileClose => "closes an open file",
        Intrinsic::StdHandle => "gives the handle of a standard stream",
        Intrinsic::IsConsole => "says whether a handle is a console's",
        Intrinsic::ConsoleRead => "reads what is typed at a console",
        Intrinsic::ModuleOpen => "loads a library",
        Intrinsic::ModuleFind => "finds what a loaded library exports",
        Intrinsic::ModuleClose => "unloads a library",
        Intrinsic::CallVoid => "calls the function at an address",
        Intrinsic::HostDescriptor => "gives the address of the program's own descriptor",
        Intrinsic::Clock => "reads the system clock",
        Intrinsic::Gate(_) => "applies a gate of the standard set to the qubits named",
        Intrinsic::Apply => "applies an exact unitary matrix as a gate",
        Intrinsic::Node => "names one of the processor's nodes",
        Intrinsic::Adjoint => "gives the operator that undoes an operator, or a value's adjoint",
        Intrinsic::Controlled => "gives an operator controlled on a qubit",
        Intrinsic::Then => "gives two operators applied one after the other",
        Intrinsic::Derived(_) => "gives what is derived of an operator's judgment",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};

    #[test]
    fn every_name_is_found_both_ways() {
        for n in NAMES {
            assert_eq!(prelude(n), lookup(n));
            assert!(lookup(n).is_some(), "{n}");
        }
        assert_eq!(lookup("len"), None, "len is a method");
    }

    #[test]
    fn printing_takes_a_string() {
        check("fn f() { print(\"hello\\n\"); eprint(\"oops\"); core::print(\"x\"); }");
        assert_eq!(codes("fn f() { print(5); }"), [Code::Es06]);
    }

    #[test]
    fn exit_and_panic_end_a_function_without_a_value() {
        check("fn f(i: i32) -> i32 { if i < 3 { i } else { panic(\"too big\") } }");
        check("fn f() -> i32 { exit(3) }");
    }

    #[test]
    fn a_unit_function_hides_the_library_one() {
        check("fn print(x: i32) -> i32 { x }\nfn f() -> i32 { print(4) }");
        check("fn print(x: i32) { }\nfn f() { core::print(\"still here\"); }");
    }
}
