//! The annotations that ask something of a function or object itself, rather
//! than of a type's layout or of a judgement:
//!
//! - **`[export: "sym"]`** gives a function or object the symbol `sym` in the
//!   object file, in place of the qualified one the compiler makes, so that
//!   code written in another language can link against it. The symbol is a
//!   plain C identifier, and one a C program may declare: not `main`, and not
//!   one beginning with `__`, which C sets aside for its implementation. A
//!   generic function has a symbol per instance, a `static` item none that
//!   another object may see, and a constant no storage, so none of them can
//!   be exported.
//! - **`[inline]`** and **`[inline: never]`** are hints to the optimiser:
//!   inlining the function is expected to pay, or it is never inlined. Neither
//!   changes what the program does.
//! - **`[test]`** marks a function that `tqc test` runs. It takes nothing and
//!   returns nothing; it passes if it returns and fails if it aborts.
//! - **`[deprecated: "msg"]`** keeps an item working and warns at each use,
//!   in this unit or an importing one, repeating the message. A use inside a
//!   deprecated item itself is not warned about: it is being retired along
//!   with the item.
//!
//! On a quantum unit's operators it also reads `[entry]`, `[dynamic]`,
//! `[trusted_unitary]` and `[adjoint: f]`.

use std::collections::HashSet;

use crate::ast::{self, AnnArg, AnnotationGroup, Linkage, StdAnnotation};
use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::span::{Span, Spanned};
use crate::tir::{FnAttrs, Inline};

use super::items::{Target, UnitCx};

/// What a group of annotations is written on.
#[derive(Clone, Copy, Debug)]
pub struct Subject {
    /// The kind of item.
    pub target: Target,
    /// Whether other units may name it.
    pub linkage: Linkage,
    /// Whether it is a generic function.
    pub generic: bool,
    /// Whether it is a method.
    pub method: bool,
    /// Whether it is a constant, with no storage of its own.
    pub constant: bool,
}

impl Subject {
    /// A function declared at unit scope.
    pub fn function(linkage: Linkage, generic: bool) -> Subject {
        Subject {
            target: Target::Function,
            linkage,
            generic,
            method: false,
            constant: false,
        }
    }
}

/// What the annotations ask, read without reporting anything: for an
/// instance of a generic function, whose annotations were checked where it
/// was declared.
pub fn quiet(groups: &[Spanned<AnnotationGroup>], interner: &Interner) -> FnAttrs {
    let mut sink = Vec::new();
    let subject = Subject::function(Linkage::Program, true);
    gather(groups, interner, subject, &mut sink)
}

/// What the annotations ask, with a diagnostic in `diags` for each one whose
/// argument is not of the form it takes, or that cannot apply to `subject`.
fn gather(
    groups: &[Spanned<AnnotationGroup>],
    interner: &Interner,
    subject: Subject,
    diags: &mut Vec<Diagnostic>,
) -> FnAttrs {
    let mut out = FnAttrs::default();
    for g in groups {
        for a in &g.node.annotations {
            let name = interner.resolve(a.node.name.node);
            let args = &a.node.args;
            match StdAnnotation::from_name(name) {
                Some(StdAnnotation::Export) if matches!(subject.target, Target::Function | Target::Object) => {
                    out.symbol = export(args, interner, subject, a.span, diags);
                }
                Some(StdAnnotation::Inline) if subject.target == Target::Function => {
                    out.inline = inline(args, interner, a.span, diags);
                }
                Some(StdAnnotation::Test) if subject.target == Target::Function => {
                    out.test = plain_only(
                        Plain {
                            name: "test",
                            applies: "a function that can run on its own",
                            generic: "generic function",
                            note: "`tqc test` calls each test with nothing, so a test is a plain function \
                                   taking nothing and returning nothing",
                        },
                        args,
                        subject,
                        a.span,
                        diags,
                    );
                }
                Some(StdAnnotation::Deprecated) => {
                    out.deprecated = deprecated(args, interner, a.span, diags);
                }
                Some(StdAnnotation::Dynamic) if subject.target == Target::Function => {
                    out.dynamic = true;
                }
                Some(StdAnnotation::TrustedUnitary) if subject.target == Target::Function => {
                    out.trusted_unitary = true;
                }
                Some(StdAnnotation::Adjoint) if subject.target == Target::Function => {
                    // a bare name, which may have parsed as an expression or a type
                    let named = match args.as_slice() {
                        [a] => match &a.node {
                            AnnArg::Expr(e) => match &e.node {
                                crate::ast::Expr::Path { path, args } if args.is_empty() && path.is_simple() => {
                                    path.last().map(|s| (s, e.span))
                                }
                                _ => None,
                            },
                            AnnArg::Type(t) => match &t.node {
                                crate::ast::Type::Path { path, args } if args.is_empty() && path.is_simple() => {
                                    path.last().map(|s| (s, t.span))
                                }
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    };
                    match named {
                        Some(n) => out.adjoint_name = Some(n),
                        None => diags.push(malformed(
                            a.span,
                            "`[adjoint: f]` names the operator that is this one's adjoint",
                            "write the operator's name, as in `[adjoint: undo_prepare]`",
                        )),
                    }
                }
                Some(StdAnnotation::Entry) if subject.target == Target::Function => {
                    out.entry = plain_only(
                        Plain {
                            name: "entry",
                            applies: "an operator with one circuit",
                            generic: "generic operator",
                            note: "an entry operator becomes one circuit, which a classical program \
                                   holds and runs by the operator's name",
                        },
                        args,
                        subject,
                        a.span,
                        diags,
                    );
                }
                _ => {}
            }
        }
    }
    out
}

/// An annotation that takes no argument and applies only to a plain
/// function, and how its refusal reads.
struct Plain {
    /// Its name.
    name: &'static str,
    /// What it applies to.
    applies: &'static str,
    /// A generic function.
    generic: &'static str,
    /// Why it applies only there.
    note: &'static str,
}

/// Whether the annotation `ann`, written at `span`, applies to `subject`,
/// with a diagnostic for an argument given, or for a subject that is not a
/// plain function.
fn plain_only(ann: Plain, args: &[Spanned<AnnArg>], subject: Subject, span: Span, diags: &mut Vec<Diagnostic>) -> bool {
    let name = ann.name;
    if !args.is_empty() {
        diags.push(malformed(span, &format!("`[{name}]` takes no argument"), &format!("write `[{name}]`")));
    }
    if !subject.generic && !subject.method {
        return true;
    }
    diags.push(
        Diagnostic::new(Code::Es18)
            .with_message(format!(
                "`[{name}]` applies to {}, not to a {}",
                ann.applies,
                if subject.method { "method" } else { ann.generic }
            ))
            .at(span)
            .with_note(ann.note)
            .with_help(format!("write a plain `fn` that calls this one, and mark that `[{name}]`")),
    );
    false
}

/// The symbol of `[export: "sym"]`, if it is one that can be given.
fn export(
    args: &[Spanned<AnnArg>],
    interner: &Interner,
    subject: Subject,
    span: Span,
    diags: &mut Vec<Diagnostic>,
) -> Option<String> {
    let Some(text) = one_string(args, interner) else {
        diags.push(malformed(
            span,
            "`[export: \"sym\"]` takes the symbol as a string",
            "write the symbol in quotes, as in `[export: \"area\"]`",
        ));
        return None;
    };

    // what the item is, if it is something no symbol can be given to
    let why_not = if subject.generic {
        Some((
            "a generic function",
            "each instance of a generic function is a function of its own, with a symbol of its own, \
             so no one symbol can name them all",
            "export a plain function that calls the instance wanted",
        ))
    } else if subject.method {
        Some((
            "a method",
            "a method is reached through its type, and its symbol is the type's to give",
            "export a plain function that calls the method",
        ))
    } else if subject.linkage == Linkage::Unit {
        Some((
            "a `static` item",
            "a `static` item is private to its unit, but an exported symbol is visible to every \
             object linked with it",
            "drop `static`, or drop `[export]`",
        ))
    } else if subject.constant {
        Some((
            "a constant",
            "a constant has no storage: every use of it is replaced by its value, so there is \
             nothing for a symbol to name",
            "declare an object without `const` to give it storage and a symbol",
        ))
    } else {
        None
    };
    if let Some((what, note, help)) = why_not {
        diags.push(
            Diagnostic::new(Code::Es18)
                .with_message(format!("`[export]` cannot give {what} a symbol"))
                .at(span)
                .with_note(note)
                .with_help(help),
        );
        return None;
    }

    // what is wrong with the symbol, if anything
    let plain = text.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    let problem = if !plain {
        Some("a symbol is a C identifier: a letter or `_`, then letters, digits and `_`")
    } else if text == "main" {
        Some("`main` is the symbol the C runtime starts the program at, which the compiler defines")
    } else if text.starts_with("__") {
        Some("C sets symbols beginning with `__` aside for its implementation and runtime")
    } else {
        None
    };
    if let Some(note) = problem {
        diags.push(
            Diagnostic::new(Code::Ea05)
                .with_message(format!("`{text}` cannot be an exported symbol"))
                .at(span)
                .with_note(note)
                .with_help("choose another name, as a C program would declare it"),
        );
        return None;
    }
    Some(text)
}

/// The hint of `[inline]` or `[inline: never]`.
fn inline(args: &[Spanned<AnnArg>], interner: &Interner, span: Span, diags: &mut Vec<Diagnostic>) -> Option<Inline> {
    match args {
        [] => Some(Inline::Hint),
        [a] if matches!(&a.node, AnnArg::Expr(e)
            if matches!(&e.node, ast::Expr::Path { path, args }
                if args.is_empty() && path.segments.len() == 1 && interner.resolve(path.segments[0].node) == "never")) =>
        {
            Some(Inline::Never)
        }
        _ => {
            diags.push(malformed(
                span,
                "`[inline]` takes nothing, or `never`",
                "write `[inline]` to ask for inlining, or `[inline: never]` to forbid it",
            ));
            None
        }
    }
}

/// The message of `[deprecated: "msg"]`, empty for `[deprecated]`.
fn deprecated(args: &[Spanned<AnnArg>], interner: &Interner, span: Span, diags: &mut Vec<Diagnostic>) -> Option<String> {
    if args.is_empty() {
        return Some(String::new());
    }
    match one_string(args, interner) {
        Some(msg) => Some(msg),
        None => {
            diags.push(malformed(
                span,
                "`[deprecated: \"msg\"]` takes its message as a string",
                "write the message in quotes, as in `[deprecated: \"use area2 instead\"]`",
            ));
            // still deprecated, with no message to repeat
            Some(String::new())
        }
    }
}

/// The text of the single string literal `args` holds.
fn one_string(args: &[Spanned<AnnArg>], interner: &Interner) -> Option<String> {
    let [a] = args else { return None };
    let AnnArg::Expr(e) = &a.node else { return None };
    match &e.node {
        ast::Expr::Str { value, .. } => Some(interner.resolve(*value).to_owned()),
        _ => None,
    }
}

fn malformed(span: Span, message: &str, help: &str) -> Diagnostic {
    Diagnostic::new(Code::Ea05)
        .with_message(message.to_owned())
        .at(span)
        .with_note("an annotation's argument has the form the annotation defines, and no other")
        .with_help(help.to_owned())
}

impl UnitCx<'_> {
    /// What the annotations on `subject` ask of it, reporting any that are
    /// malformed or cannot apply, and an `[export]` symbol given twice.
    pub fn item_attrs(&mut self, groups: &[Spanned<AnnotationGroup>], subject: Subject) -> FnAttrs {
        let mut diags = Vec::new();
        let attrs = gather(groups, self.interner, subject, &mut diags);
        for d in diags {
            self.report(d);
        }
        if let Some(sym) = &attrs.symbol {
            let at = groups.first().map_or_else(Span::synthetic, |g| g.span);
            if let Some(&earlier) = self.exported.get(sym) {
                self.report(
                    Diagnostic::new(Code::Es05)
                        .with_message(format!("the symbol `{sym}` is exported twice"))
                        .at_with(at, "exported again here")
                        .also(earlier, "first exported here")
                        .with_note("a symbol names one definition in the whole program")
                        .with_help("give one of them another symbol"),
                );
            } else {
                self.exported.insert(sym.clone(), at);
            }
        }
        attrs
    }

    /// Warns if the function `callee`, used at `span`, is deprecated.
    pub fn use_callee(&mut self, callee: crate::tir::Callee, span: Span) {
        let (name, msg) = match callee {
            crate::tir::Callee::Fn(id) => {
                let f = self.unit.func(id);
                (f.name, f.attrs.deprecated.clone())
            }
            crate::tir::Callee::Extern(id) => {
                let x = self.unit.extern_fn(id);
                (x.name, x.deprecated.clone())
            }
        };
        if let Some(msg) = msg {
            let what = self.interner.resolve(name).to_owned();
            self.deprecated_use(&what, &msg, span);
        }
    }

    /// Warns if the object `id`, used at `span`, is deprecated.
    pub fn use_global(&mut self, id: crate::tir::GlobalId, span: Span) {
        let g = self.unit.global(id);
        if let Some(msg) = g.deprecated.clone() {
            let what = self.interner.resolve(g.name).to_owned();
            self.deprecated_use(&what, &msg, span);
        }
    }

    /// Warns if the structure or enumeration `id`, named at `span`, is
    /// deprecated.
    pub fn use_adt(&mut self, id: crate::tir::AdtId, span: Span) {
        if let Some(msg) = self.type_deprecation(id).cloned() {
            let what = self.interner.resolve(self.unit.types.adt(id).name).to_owned();
            self.deprecated_use(&what, &msg, span);
        }
    }

    /// The message of the `[deprecated]` on the structure or enumeration
    /// `id`, if it has one.
    fn type_deprecation(&self, id: crate::tir::AdtId) -> Option<&String> {
        let a = self.unit.types.adt(id);
        let origin = a.origin.clone().filter(|o| *o != self.unit.name);
        self.deprecated_types.get(&(origin, a.name))
    }

    /// Reports each `[test]` function that does not take nothing and return
    /// nothing, which is how `tqc test` calls it.
    pub fn check_tests(&mut self) {
        let wrong: Vec<(String, Span)> = self
            .unit
            .fns
            .iter()
            .filter(|f| f.attrs.test && f.origin.is_none() && (!f.params.is_empty() || f.ret != crate::tir::Ty::Void))
            .map(|f| (self.interner.resolve(f.name).to_owned(), f.span))
            .collect();
        for (name, span) in wrong {
            self.report(
                Diagnostic::new(Code::Es18)
                    .with_message(format!("the test `{name}` must take nothing and return nothing"))
                    .at(span)
                    .with_note(
                        "`tqc test` calls each test with no arguments and nothing to receive a value; \
                         a test passes by returning and fails by aborting",
                    )
                    .with_help(format!(
                        "declare `fn {name}() {{ … }}`, and have it `panic` when a result is wrong"
                    )),
            );
        }
    }

    /// Whether uses inside function `id`, declared in frame `frame`, go
    /// unwarned: it is deprecated itself, or a method of a deprecated type,
    /// or another unit's, whose uses that unit was warned of.
    pub fn quiet_inside(&self, id: crate::tir::FnId, frame: usize) -> bool {
        let f = self.unit.func(id);
        let of_deprecated_type = f.method.is_some_and(|m| self.type_deprecation(m.owner).is_some());
        frame != 0 || f.origin.is_some() || f.attrs.deprecated.is_some() || of_deprecated_type
    }

    /// Runs `f` with uses of deprecated items not warned of, when `quiet`:
    /// while checking an item that is itself deprecated.
    pub fn quietly<T>(&mut self, quiet: bool, f: impl FnOnce(&mut Self) -> T) -> T {
        self.quiet_deprecation += u32::from(quiet);
        let out = f(self);
        self.quiet_deprecation -= u32::from(quiet);
        out
    }

    /// Warns that `what`, marked `[deprecated: msg]`, is used at `span`,
    /// unless the use is inside a deprecated item or was already warned of.
    pub fn deprecated_use(&mut self, what: &str, msg: &str, span: Span) {
        if self.quiet_deprecation > 0 || span.source.is_synthetic() || !self.warned_uses.insert(span) {
            return;
        }
        let mut d = Diagnostic::new(Code::Es21)
            .with_message(format!("`{what}` is deprecated"))
            .at(span)
            .with_note(format!("`{what}` is marked `[deprecated]`, so it may be removed; it still works for now"))
            .as_warning();
        d = if msg.is_empty() {
            d.with_help(format!("stop using `{what}`"))
        } else {
            d.with_help(msg.to_owned())
        };
        self.report(d);
    }
}

/// Uses already warned of, so that an expression checked twice (once for
/// the type it should have and once as it is) warns once.
pub type Warned = HashSet<Span>;
