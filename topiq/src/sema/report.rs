//! The diagnostics analysis raises most often, written once.
//!
//! Each builder produces a complete diagnostic: a headline saying what is
//! wrong in terms of the reader's own program, and notes or help saying why
//! and what to write instead. Having them in one place keeps the wording
//! consistent (the same mistake always reads the same way) and keeps the
//! checker's code about checking.

use std::fmt::Display;

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::span::Span;
use crate::tir::{IntTy, Ty, TypeTable};

/// Something the parser accepted that the compiler does not translate: the
/// few quantum constructs `crate::sema` lists, and code written without the
/// `core` library (`TQ003`). The message says "this build cannot compile
/// {what} yet". `what` is a plural noun phrase, such as `"structure
/// declarations"`.
pub fn unsupported(span: Span, what: impl Display) -> Diagnostic {
    Diagnostic::new(Code::Tq003)
        .with_message(format!("this build cannot compile {what} yet"))
        .at(span)
        .with_note(
            "the program is well formed as far as the parser can tell; the rest of \
             the compiler has not caught up with this construct",
        )
}

/// A path of more parts than any name has: a unit, a type in it, and a
/// variant of the type. `most` is how many parts the position allows.
pub fn path_too_long(span: Span, written: &str, most: usize) -> Diagnostic {
    let form = match most {
        2 => "`unit::Type`",
        _ => "`unit::Type::Variant`",
    };
    Diagnostic::new(Code::Es04)
        .with_message(format!("`{written}` names nothing: a path here has at most {most} parts, as in {form}"))
        .at(span)
        .with_note(
            "units do not nest, and neither do types, so a name is at most a unit's item and, for \
             an enumeration, one of its variants; an import names a unit by the last part of its path",
        )
        .with_help("name the unit by what it was imported as, then the item")
}

/// A name that denotes nothing visible here. `kind` is what was expected:
/// `"value"`, `"function"` or `"type"`, and `visible` the names that are in
/// scope, from which a likely intended one is suggested.
pub fn not_found<'a>(
    span: Span,
    kind: &str,
    name: &str,
    visible: impl IntoIterator<Item = &'a str>,
) -> Diagnostic {
    let mut d = Diagnostic::new(Code::Es04)
        .with_message(format!("there is no {kind} named `{name}` here"))
        .at(span);
    if let Some(s) = closest(name, visible) {
        d = d.with_help(format!("did you mean `{s}`?"));
    }
    if kind == "value" {
        d = d.with_note(
            "a `let` binding is visible from the end of its own statement to the end \
             of the block it is in; functions and unit-scope bindings are visible \
             throughout the file",
        );
    }
    d
}

/// Two declarations of one name where only one can stand.
pub fn duplicate(span: Span, earlier: Span, name: &str, where_: &str) -> Diagnostic {
    Diagnostic::new(Code::Es05)
        .with_message(format!("`{name}` is declared twice {where_}"))
        .at_with(span, "declared again here")
        .also(earlier, "first declared here")
        .with_help("rename one of them")
}

/// A value of the wrong type. `role` says what the value is for, such as
/// "the condition of an `if`"; `expected` and `found` are types already in
/// words, from [`describe`].
pub fn mismatch(span: Span, role: &str, expected: &str, found: &str) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!("{role} should be {expected}, but this is {found}"))
        .at_with(span, format!("this is {found}"))
        .with_note(
            "values never change type on their own; where a conversion between \
             integer types is intended, write it with `as`",
        )
}

/// A type in the words a message uses: `an i32`, `a bool`, `a *[char]`,
/// `a Point`, `nothing (void)`.
pub fn describe(t: Ty, types: &TypeTable, interner: &Interner) -> String {
    match t {
        Ty::Int(i) => describe_int(i),
        Ty::Float(f) => format!("an {f}"),
        Ty::Void => "nothing (void)".to_owned(),
        Ty::Never => "something that never produces a value".to_owned(),
        Ty::Infer(_) => "an integer".to_owned(),
        Ty::Array(_) => format!("an array `{}`", types.display(t, interner)),
        Ty::Tuple(_) => match types.as_tuple(t) {
            Some([]) => "the empty tuple `()`".to_owned(),
            _ => format!("a tuple `{}`", types.display(t, interner)),
        },
        Ty::Ref(_) => format!("a reference `{}`", types.display(t, interner)),
        Ty::Slice(_) => format!("a slice `{}`", types.display(t, interner)),
        Ty::Growable(_) => format!("a growable array `{}`", types.display(t, interner)),
        Ty::Fn(_) => format!("a function `{}`", types.display(t, interner)),
        Ty::Closure(_) => format!("a closure `{}`", types.display(t, interner)),
        Ty::Circuit(_) => format!("a circuit handle `{}`", types.display(t, interner)),
        Ty::Qmap(_) => format!("a map locale `{}`", types.display(t, interner)),
        Ty::Dyn => "a dynamic value `dyn`".to_owned(),
        Ty::Qubit => "a `qubit`".to_owned(),
        Ty::Adt(id) => {
            let def = types.adt(id);
            format!("{} `{}`", article(def.noun()), types.adt_name(id, interner))
        }
        Ty::Bool | Ty::Char => format!("a {}", types.display(t, interner)),
    }
}

/// An integer type in words: "an i8", "an i64", "an isize", but "a u32".
pub fn describe_int(t: IntTy) -> String {
    let name = t.name();
    if name.starts_with('i') {
        format!("an {name}")
    } else {
        format!("a {name}")
    }
}

/// `noun` with its indefinite article.
fn article(noun: &str) -> String {
    if noun.starts_with(['a', 'e', 'i', 'o', 'u']) {
        format!("an {noun}")
    } else {
        format!("a {noun}")
    }
}

/// The candidate closest to `name` by edit distance, if any is close enough to
/// be a plausible typo.
pub fn closest<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let limit = (name.chars().count() / 3).max(1);
    candidates
        .into_iter()
        .filter(|c| *c != name)
        .map(|c| (edit_distance(name, c), c))
        .filter(|&(d, _)| d <= limit)
        .min_by_key(|&(d, c)| (d, c))
        .map(|(_, c)| c)
}

/// How many single-character typos separate two strings: insertions,
/// deletions, substitutions, and swaps of two neighbours (`i23` for `i32` is
/// one typo, not two).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    // d[i][j]: the distance between the first i characters of `a` and the
    // first j of `b`
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = best;
        }
    }
    d[a.len()][b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SourceId;
    use crate::tir::IntTy;

    fn sp() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    #[test]
    fn edit_distance_counts_single_character_changes() {
        assert_eq!(edit_distance("count", "count"), 0);
        assert_eq!(edit_distance("count", "cont"), 1);
        assert_eq!(edit_distance("count", "coutn"), 1, "a swap is one typo");
        assert_eq!(edit_distance("i23", "i32"), 1);
        assert_eq!(edit_distance("count", "cuont"), 1);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn a_close_name_is_suggested_and_a_distant_one_is_not() {
        let names = ["counter", "limit", "total"];
        assert_eq!(closest("countr", names), Some("counter"));
        assert_eq!(closest("xyz", names), None);
        assert_eq!(closest("limit", ["limit"]), None, "never suggest the name itself");
    }

    #[test]
    fn a_missing_name_suggests_the_likely_one() {
        let d = not_found(sp(), "value", "totl", ["total", "x"]);
        assert_eq!(d.code, Code::Es04);
        assert!(d.message.contains("`totl`"), "{}", d.message);
        assert!(d.helps.iter().any(|h| h.contains("`total`")), "{:?}", d.helps);
    }

    #[test]
    fn a_mismatch_says_what_was_wanted_and_what_was_found() {
        let d = mismatch(sp(), "the condition of an `if`", "a bool", "an i32");
        assert_eq!(d.code, Code::Es06);
        assert!(d.message.contains("should be a bool"), "{}", d.message);
        assert!(d.message.contains("an i32"), "{}", d.message);
    }

    #[test]
    fn types_read_naturally_in_a_sentence() {
        let mut types = TypeTable::new();
        let i = Interner::new();
        assert_eq!(describe(Ty::Int(IntTy::I8), &types, &i), "an i8");
        assert_eq!(describe(Ty::Int(IntTy::U32), &types, &i), "a u32");
        assert_eq!(describe(Ty::Int(IntTy::USIZE), &types, &i), "a usize");
        assert_eq!(describe(Ty::Bool, &types, &i), "a bool");
        assert_eq!(describe(Ty::Char, &types, &i), "a char");
        let s = types.string();
        assert_eq!(describe(s, &types, &i), "a slice `*[char]`");
        let a = types.array(Ty::Bool, 2);
        assert_eq!(describe(a, &types, &i), "an array `[bool; 2]`");
    }

    #[test]
    fn a_duplicate_points_at_both_declarations() {
        let d = duplicate(sp(), Span::new(SourceId(0), 5, 6), "f", "at unit scope");
        assert_eq!(d.code, Code::Es05);
        assert_eq!(d.labels.len(), 2);
    }

    #[test]
    fn unsupported_constructs_name_themselves() {
        let d = unsupported(sp(), "structure declarations");
        assert_eq!(d.code, Code::Tq003);
        assert!(d.message.contains("structure declarations"), "{}", d.message);
    }
}
