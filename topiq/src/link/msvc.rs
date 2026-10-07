//! Driving the Microsoft linker.
//!
//! # Finding it
//!
//! The linker is located through the Visual Studio installation, never
//! through `PATH`. That is not only tidiness: Git for Windows ships a `link`
//! command of its own (a tool for making hard links) and a shell with Git on
//! its `PATH` finds that one first. Locating the linker through the
//! installation also yields the library search path it needs, which a bare
//! `link.exe` on `PATH` would not have.
//!
//! An explicitly named linker, such as `lld-link`, still inherits that search
//! path when a Visual Studio installation can be found, since it needs the
//! same C runtime and system libraries.
//!
//! # What it is given
//!
//! The objects, plus the static C runtime, `kernel32`, and `shell32` for the
//! program's arguments; `/DLL` as well for a module. The runtime is linked
//! statically so that a built program runs on any machine without a matching
//! Visual C++ redistributable installed. A module has a runtime of its own;
//! it shares the program's heap, which the runtime takes from the system.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::ToolError;

/// The target the linker is looked up for.
const TARGET: &str = "x86_64-pc-windows-msvc";

/// The libraries every program links against: the static C runtime, in its
/// three parts, the Windows API it rests on, and the part of the Windows API
/// that splits a command line into the program's arguments.
pub const LIBRARIES: [&str; 5] = ["libcmt.lib", "libucrt.lib", "libvcruntime.lib", "kernel32.lib", "shell32.lib"];

/// A linker ready to run: its program and the environment it needs.
#[derive(Clone, Debug)]
pub struct Linker {
    /// The linker executable.
    pub program: PathBuf,
    /// Environment variables to set, notably the library search path `LIB`.
    pub env: Vec<(OsString, OsString)>,
}

impl Linker {
    /// Finds the linker: `explicit` if given, otherwise the one belonging to
    /// the newest Visual Studio installation.
    ///
    /// # Errors
    ///
    /// [`ToolError::NotFound`] when no linker is named and none is installed.
    pub fn find(explicit: Option<&Path>) -> Result<Linker, ToolError> {
        let installed = installed();
        match (explicit, installed) {
            (Some(p), found) => Ok(Linker {
                program: p.to_owned(),
                env: found.map(|l| l.env).unwrap_or_default(),
            }),
            (None, Some(l)) => Ok(l),
            (None, None) => Err(ToolError::NotFound),
        }
    }

    /// The arguments that link `objects` into the executable `out`.
    pub fn arguments(objects: &[PathBuf], out: &Path) -> Vec<OsString> {
        let mut args: Vec<OsString> = vec![
            "/NOLOGO".into(),
            "/SUBSYSTEM:CONSOLE".into(),
            "/MACHINE:X64".into(),
            {
                let mut o = OsString::from("/OUT:");
                o.push(out);
                o
            },
        ];
        args.extend(objects.iter().map(|p| p.as_os_str().to_owned()));
        args.extend(LIBRARIES.iter().map(OsString::from));
        args
    }

    /// Links `objects` into the executable `out`.
    ///
    /// # Errors
    ///
    /// [`ToolError::Spawn`] if the linker cannot be started, and
    /// [`ToolError::Failed`] with its output if it reports failure.
    pub fn link(&self, objects: &[PathBuf], out: &Path) -> Result<(), ToolError> {
        self.run(Linker::arguments(objects, out))
    }

    /// Links `objects` into the library `out`, which a program loads while
    /// it runs: a module.
    ///
    /// # Errors
    ///
    /// As [`Linker::link`].
    pub fn link_library(&self, objects: &[PathBuf], out: &Path) -> Result<(), ToolError> {
        let mut args = Linker::arguments(objects, out);
        args.insert(1, "/DLL".into());
        self.run(args)
    }

    fn run(&self, args: Vec<OsString>) -> Result<(), ToolError> {
        let mut cmd = Command::new(&self.program);
        cmd.args(args);
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        let output = cmd.output().map_err(|e| ToolError::Spawn {
            program: self.program.clone(),
            reason: e.to_string(),
        })?;
        if output.status.success() {
            return Ok(());
        }
        // the Microsoft linker writes its errors to standard output
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        Err(ToolError::Failed {
            program: self.program.clone(),
            status: output.status.code(),
            output: text.trim().to_owned(),
        })
    }
}

/// The installed Visual Studio linker.
#[cfg(windows)]
fn installed() -> Option<Linker> {
    let tool = find_msvc_tools::find_tool(TARGET, "link.exe")?;
    Some(Linker {
        program: tool.path().to_owned(),
        env: tool.env().into_iter().cloned().collect(),
    })
}

/// There is no Visual Studio installation to find off Windows.
#[cfg(not(windows))]
fn installed() -> Option<Linker> {
    let _ = TARGET;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_name_the_output_the_objects_and_the_runtime() {
        let args = Linker::arguments(&[PathBuf::from("a.tcu"), PathBuf::from("b.tcu")], Path::new("app.exe"));
        let args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert!(args.contains(&"/OUT:app.exe".to_owned()), "{args:?}");
        assert!(args.contains(&"a.tcu".to_owned()));
        assert!(args.contains(&"b.tcu".to_owned()));
        for lib in LIBRARIES {
            assert!(args.contains(&lib.to_owned()), "{lib}");
        }
        assert!(args.contains(&"/SUBSYSTEM:CONSOLE".to_owned()));
    }

    #[test]
    fn an_explicit_linker_is_used_as_given() {
        let l = Linker::find(Some(Path::new("C:/tools/lld-link.exe"))).unwrap();
        assert_eq!(l.program, PathBuf::from("C:/tools/lld-link.exe"));
    }

    #[cfg(windows)]
    #[test]
    fn the_installed_linker_is_microsofts_and_not_gits() {
        // Git for Windows' `link.exe` lives under its `usr\bin`; the real one
        // lives under Visual Studio's tool directory
        let l = Linker::find(None).expect("a Visual Studio linker is installed on this machine");
        let p = l.program.to_string_lossy().to_lowercase();
        assert!(p.ends_with("link.exe"), "{p}");
        assert!(!p.contains("\\git\\"), "found Git's link.exe: {p}");
        assert!(
            l.env.iter().any(|(k, _)| k.eq_ignore_ascii_case("LIB")),
            "the library search path comes with it"
        );
    }
}
