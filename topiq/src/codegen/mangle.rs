//! The symbol names a unit's definitions are given in its object.
//!
//! Every name is qualified by its unit, so two units may each define a
//! function called `helper` without their symbols colliding: they are
//! different items, and the object file keeps them apart. The `$` separator
//! cannot occur in a Topiq identifier, so a qualified name can never be
//! mistaken for another.
//!
//! One symbol is not qualified: the C-level `main` that the C runtime calls to
//! start the program, which in turn calls the Topiq `main`.

/// The prefix every Topiq symbol carries.
pub const PREFIX: &str = "topiq";

/// The symbol of a function.
pub fn function(unit: &str, name: &str) -> String {
    format!("{PREFIX}${unit}${name}")
}

/// The symbol of one instance of a generic function.
///
/// The arguments are part of the name, written as a program writes them, so
/// that two instances differ and (just as important) the *same* instance
/// made in two units has the same symbol. The linker then keeps one copy.
/// Characters a symbol cannot carry are replaced, which never brings two
/// different argument lists together: `<`, `,` and a space all become `$`,
/// and no Topiq identifier holds a `$`.
pub fn instance(unit: &str, name: &str, args: &str) -> String {
    format!("{PREFIX}${unit}${name}${}", encode(args))
}

/// The symbol of a method or associated function of the type written
/// `owner`: qualified by its unit, with its arguments if it is an instance,
/// with `args` its instance's own arguments, if any.
///
/// A type has one method of each name in a whole program, whichever unit
/// defines it, so the type alone qualifies the name; units that each make the
/// instance of a generic method make the same symbol.
pub fn method(owner: &str, name: &str, args: &str) -> String {
    let base = format!("{PREFIX}${}.{name}", encode(owner));
    if args.is_empty() { base } else { format!("{base}${}", encode(args)) }
}

/// Text a symbol can carry: `<`, `,`, `:` and a space each become `$`, which
/// no Topiq identifier holds, so two different texts stay different.
fn encode(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '$' })
        .collect()
}

/// The symbol of a unit-scope object.
pub fn global(unit: &str, name: &str) -> String {
    format!("{PREFIX}${unit}${name}")
}

/// The symbol of a persistent object. It is private to its function, and two
/// functions (or two blocks of one function) may each declare one of the
/// same name, so the object's index keeps it unique.
pub fn persist(unit: &str, func: &str, name: &str, index: u32) -> String {
    format!("{PREFIX}${unit}${func}${name}$persist{index}")
}

/// The symbol through which the program runs a unit's initialiser. No
/// identifier is empty, so `$$` never occurs in the symbol of anything a
/// program declares.
pub fn init(unit: &str) -> String {
    format!("{PREFIX}${unit}$$init")
}

/// The function that runs every unit's initialiser, in order, before `main`.
/// It is made when the program is linked, since only then are all the units
/// known.
pub const STARTUP: &str = "topiq$$startup";

/// The helper every abort calls. Each object defines its own private copy.
pub const ABORT: &str = "topiq$abort";

/// The entry point the C runtime calls.
pub const ENTRY: &str = "main";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_qualified_by_their_unit() {
        assert_eq!(function("geometry", "area"), "topiq$geometry$area");
        assert_ne!(function("a", "helper"), function("b", "helper"));
    }

    #[test]
    fn persistent_objects_of_one_name_stay_distinct() {
        assert_ne!(persist("u", "f", "n", 0), persist("u", "f", "n", 1));
        assert_ne!(persist("u", "f", "n", 0), persist("u", "g", "n", 0));
    }

    #[test]
    fn the_topiq_main_is_not_the_c_main() {
        assert_ne!(function("app", "main"), ENTRY);
    }
}
