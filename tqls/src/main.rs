//! `tqls`, the Topiq language server.
//!
//! Speaks the Language Server Protocol over standard input and output. It
//! checks each open file as `tqc check` would, as the root of a program with
//! every unit it imports, and offers:
//!
//! - diagnostics, with their identifiers, notes and help, as the file is
//!   edited, and for the units it imports;
//! - hover: a binding's or expression's type, a function's signature, an
//!   item's declaration and the comment above it;
//! - go to definition, across units;
//! - the file's outline;
//! - completion of keywords, names in scope, a unit's items after `unit::`,
//!   fields and methods after `.`, macros, annotations, directives and units.
//!
//! Units are found beside the file and in the directories of the `unitPath`
//! initialisation option, relative to the workspace.

mod analysis;
mod complete;
mod lines;
mod navigate;
mod server;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::process::ExitCode;

use lsp_types::Url;

fn main() -> ExitCode {
    if let Some(arg) = std::env::args().nth(1) {
        match arg.as_str() {
            "--version" | "-V" => println!("tqls {}", env!("CARGO_PKG_VERSION")),
            _ => {
                println!("tqls {}: the Topiq language server", env!("CARGO_PKG_VERSION"));
                println!("Run with no arguments, it speaks the Language Server Protocol over standard input and output.");
            }
        }
        return ExitCode::SUCCESS;
    }

    // analysis recurses as deeply as a program nests, so it is given room
    let run = std::thread::Builder::new()
        .name("tqls".to_owned())
        .stack_size(64 << 20)
        .spawn(|| server::run().map_err(|e| e.to_string()));
    match run.map(std::thread::JoinHandle::join) {
        Ok(Ok(Ok(()))) => ExitCode::SUCCESS,
        Ok(Ok(Err(why))) => {
            eprintln!("tqls: {why}");
            ExitCode::FAILURE
        }
        Ok(Err(_)) => {
            eprintln!("tqls: the server failed");
            ExitCode::FAILURE
        }
        Err(why) => {
            eprintln!("tqls: cannot start: {why}");
            ExitCode::FAILURE
        }
    }
}

/// The URL of the file at `path`.
fn url(path: &Path) -> Option<Url> {
    Url::from_file_path(path).ok()
}
