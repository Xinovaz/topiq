//! The message loop: the open documents, when each is checked, and the
//! answers to the client's requests.
//!
//! An edit marks its document for checking, which happens once the client
//! has been quiet for a moment, so that typing is not held up by a check per
//! keystroke. A request about a document with edits not yet checked checks it
//! first. Opening, saving or closing a document checks every open document
//! again, since units import one another.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use lsp_server::{Connection, ErrorCode, Message, Notification, Request, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument, Notification as _,
    PublishDiagnostics,
};
use lsp_types::request::{Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, Request as _};
use lsp_types::{
    CompletionOptions, CompletionParams, CompletionResponse, Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, HoverParams, HoverProviderCapability,
    InitializeParams, InitializeResult, OneOf, PositionEncodingKind, PublishDiagnosticsParams, ServerCapabilities,
    ServerInfo, TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions, TextDocumentSyncSaveOptions,
    Url,
};
use topiq::source::FileKind;

use crate::analysis::{Checked, key};
use crate::lines::{Encoding, Lines};

/// How long the client must be quiet before edited documents are checked.
const QUIET: Duration = Duration::from_millis(250);

/// Runs the server over standard input and output until the client shuts it
/// down.
///
/// # Errors
///
/// A protocol error: the client did not initialise the server, or broke off.
pub fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let (conn, io) = Connection::stdio();
    serve(conn)?;
    io.join()?;
    Ok(())
}

/// Serves the client at the other end of `conn` until it shuts the server
/// down.
///
/// # Errors
///
/// As for [`run`].
pub fn serve(conn: Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (id, params) = conn.initialize_start()?;
    let params: InitializeParams = serde_json::from_value(params)?;

    // bytes if the client counts them, which spans already are
    let utf8 = params
        .capabilities
        .general
        .as_ref()
        .and_then(|g| g.position_encodings.as_ref())
        .is_some_and(|e| e.contains(&PositionEncodingKind::UTF8));
    let enc = if utf8 { Encoding::Utf8 } else { Encoding::Utf16 };

    let capabilities = ServerCapabilities {
        position_encoding: Some(if utf8 { PositionEncodingKind::UTF8 } else { PositionEncodingKind::UTF16 }),
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::FULL),
            save: Some(TextDocumentSyncSaveOptions::Supported(true)),
            ..TextDocumentSyncOptions::default()
        })),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some([".", ":", "@", "[", "#"].map(str::to_owned).to_vec()),
            ..CompletionOptions::default()
        }),
        ..ServerCapabilities::default()
    };
    let result = InitializeResult {
        capabilities,
        server_info: Some(ServerInfo {
            name: "tqls".to_owned(),
            version: Some(env!("CARGO_PKG_VERSION").to_owned()),
        }),
    };
    conn.initialize_finish(id, serde_json::to_value(result)?)?;

    let mut server = Server {
        conn,
        enc,
        unit_path: unit_path(&params),
        docs: HashMap::new(),
        checked: HashMap::new(),
        typed: HashMap::new(),
        dirty: Vec::new(),
        shown: HashMap::new(),
    };
    server.run()
}

/// The directories named by the `unitPath` initialisation option, relative
/// to the first workspace folder.
fn unit_path(params: &InitializeParams) -> Vec<PathBuf> {
    #[allow(deprecated)]
    let root = params
        .workspace_folders
        .as_ref()
        .and_then(|f| f.first())
        .map(|f| f.uri.clone())
        .or_else(|| params.root_uri.clone())
        .and_then(|u| u.to_file_path().ok());
    let Some(dirs) = params
        .initialization_options
        .as_ref()
        .and_then(|o| o.get("unitPath"))
        .and_then(|p| p.as_array())
    else {
        return Vec::new();
    };
    dirs.iter()
        .filter_map(|d| d.as_str())
        .map(|d| match &root {
            Some(root) => root.join(d),
            None => PathBuf::from(d),
        })
        .collect()
}

struct Server {
    conn: Connection,
    enc: Encoding,
    unit_path: Vec<PathBuf>,
    /// The open documents' text.
    docs: HashMap<Url, String>,
    /// The latest check of each open document's program.
    checked: HashMap<Url, Rc<Checked>>,
    /// The latest check of each open document that reached a typed unit,
    /// which completion draws names from while the text does not check.
    typed: HashMap<Url, Rc<Checked>>,
    /// The documents to check.
    dirty: Vec<Url>,
    /// The files each document's check has published diagnostics for.
    shown: HashMap<Url, HashSet<Url>>,
}

impl Server {
    fn run(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        loop {
            let msg = if self.dirty.is_empty() {
                match self.conn.receiver.recv() {
                    Ok(msg) => msg,
                    Err(_) => return Ok(()),
                }
            } else {
                match self.conn.receiver.recv_timeout(QUIET) {
                    Ok(msg) => msg,
                    Err(e) if e.is_timeout() => {
                        for url in std::mem::take(&mut self.dirty) {
                            self.check(&url);
                        }
                        continue;
                    }
                    Err(_) => return Ok(()),
                }
            };
            match msg {
                Message::Request(req) => {
                    if self.conn.handle_shutdown(&req)? {
                        return Ok(());
                    }
                    self.request(req);
                }
                Message::Notification(n) => self.notification(n),
                Message::Response(_) => {}
            }
        }
    }

    fn notification(&mut self, n: Notification) {
        match n.method.as_str() {
            DidOpenTextDocument::METHOD => {
                let Ok(p) = serde_json::from_value::<DidOpenTextDocumentParams>(n.params) else { return };
                self.docs.insert(p.text_document.uri.clone(), p.text_document.text);
                self.check_all(&p.text_document.uri);
            }
            DidChangeTextDocument::METHOD => {
                let Ok(mut p) = serde_json::from_value::<DidChangeTextDocumentParams>(n.params) else { return };
                // the whole text, as the server asks for
                if let Some(change) = p.content_changes.pop() {
                    self.docs.insert(p.text_document.uri.clone(), change.text);
                    self.mark(p.text_document.uri);
                }
            }
            DidSaveTextDocument::METHOD => {
                let Ok(p) = serde_json::from_value::<DidSaveTextDocumentParams>(n.params) else { return };
                self.check_all(&p.text_document.uri);
            }
            DidCloseTextDocument::METHOD => {
                let Ok(p) = serde_json::from_value::<DidCloseTextDocumentParams>(n.params) else { return };
                let url = p.text_document.uri;
                self.docs.remove(&url);
                self.checked.remove(&url);
                self.typed.remove(&url);
                self.dirty.retain(|u| *u != url);
                // what this document's check showed goes, unless an open
                // document shows it itself
                for shown in self.shown.remove(&url).unwrap_or_default() {
                    if !self.docs.contains_key(&shown) {
                        self.publish(shown, Vec::new());
                    }
                }
                let others: Vec<Url> = self.docs.keys().cloned().collect();
                for u in others {
                    self.mark(u);
                }
            }
            _ => {}
        }
    }

    /// Marks `url` for checking.
    fn mark(&mut self, url: Url) {
        if !self.dirty.contains(&url) {
            self.dirty.push(url);
        }
    }

    /// Marks every open document for checking, `first` first.
    fn check_all(&mut self, first: &Url) {
        self.dirty.retain(|u| u != first);
        self.dirty.insert(0, first.clone());
        let others: Vec<Url> = self.docs.keys().filter(|&u| u != first).cloned().collect();
        for u in others {
            self.mark(u);
        }
    }

    /// Checks the program the document `url` belongs to, and publishes what
    /// it found.
    fn check(&mut self, url: &Url) {
        let Ok(path) = url.to_file_path() else { return };
        if !self.docs.contains_key(url) || FileKind::of_path(&path) != FileKind::Unit {
            return;
        }
        let overlay = self
            .docs
            .iter()
            .filter_map(|(u, text)| Some((u.to_file_path().ok()?, text.clone())))
            .collect();
        match Checked::new(&path, &self.unit_path, &overlay) {
            Ok(checked) => {
                let mut now = HashSet::new();
                for (file, diagnostics) in checked.diagnostics(self.enc) {
                    if let Some(u) = self.url_for(&file) {
                        self.publish(u.clone(), diagnostics);
                        now.insert(u);
                    }
                }
                self.forget_shown(url, now);
                let checked = Rc::new(checked);
                let typed = checked
                    .file(&path)
                    .and_then(|id| checked.outcome(id))
                    .is_some_and(|o| o.tir().is_some());
                if typed {
                    self.typed.insert(url.clone(), checked.clone());
                }
                self.checked.insert(url.clone(), checked);
            }
            Err(why) => {
                let d = Diagnostic {
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("topiq".to_owned()),
                    message: format!("the program cannot be loaded: {why}"),
                    ..Diagnostic::default()
                };
                self.publish(url.clone(), vec![d]);
                self.forget_shown(url, HashSet::from([url.clone()]));
                self.checked.remove(url);
            }
        }
    }

    /// Records that `root`'s check shows diagnostics for `now`, clearing
    /// those it showed before for files not among them, unless an open
    /// document shows them itself.
    fn forget_shown(&mut self, root: &Url, now: HashSet<Url>) {
        let before = self.shown.insert(root.clone(), now.clone()).unwrap_or_default();
        for gone in before.difference(&now) {
            if !self.docs.contains_key(gone) {
                self.publish(gone.clone(), Vec::new());
            }
        }
    }

    /// The URL to publish for the file at `path`: an open document's own,
    /// as the client wrote it, else one made from the path.
    fn url_for(&self, path: &std::path::Path) -> Option<Url> {
        let want = key(path);
        let open = self.docs.keys().find(|u| u.to_file_path().is_ok_and(|p| key(&p) == want));
        open.cloned().or_else(|| crate::url(path))
    }

    fn publish(&self, uri: Url, diagnostics: Vec<Diagnostic>) {
        let params = PublishDiagnosticsParams {
            uri,
            diagnostics,
            version: None,
        };
        let n = Notification::new(PublishDiagnostics::METHOD.to_owned(), params);
        let _ = self.conn.sender.send(n.into());
    }

    /// The latest check of the document `url`, made now if it has edits not
    /// yet checked.
    fn fresh(&mut self, url: &Url) -> Option<Rc<Checked>> {
        if let Some(i) = self.dirty.iter().position(|u| u == url) {
            self.dirty.remove(i);
            self.check(url);
        }
        self.checked.get(url).cloned()
    }

    fn request(&mut self, req: Request) {
        let id = req.id.clone();
        let result = match req.method.as_str() {
            HoverRequest::METHOD => self.hover(req.params),
            GotoDefinition::METHOD => self.definition(req.params),
            DocumentSymbolRequest::METHOD => self.symbols(req.params),
            Completion::METHOD => self.completion(req.params),
            other => {
                let r = Response::new_err(id, ErrorCode::MethodNotFound as i32, format!("tqls does not answer `{other}`"));
                let _ = self.conn.sender.send(r.into());
                return;
            }
        };
        let response = match result {
            Ok(value) => Response::new_ok(id, value),
            Err(why) => Response::new_err(id, ErrorCode::InvalidParams as i32, why),
        };
        let _ = self.conn.sender.send(response.into());
    }

    /// The check of `url`, the file it is in, and the byte offset of `pos`.
    fn at(&mut self, url: &Url, pos: lsp_types::Position) -> Option<(Rc<Checked>, topiq::span::SourceId, usize)> {
        let checked = self.fresh(url)?;
        let id = checked.file(&url.to_file_path().ok()?)?;
        let offset = Lines::new(checked.text(id)).offset(pos, self.enc);
        Some((checked, id, offset))
    }

    fn hover(&mut self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let p: HoverParams = serde_json::from_value(params).map_err(|e| e.to_string())?;
        let at = &p.text_document_position_params;
        let hover = self
            .at(&at.text_document.uri, at.position)
            .and_then(|(checked, id, offset)| crate::navigate::hover(&checked, id, offset, self.enc));
        serde_json::to_value(hover).map_err(|e| e.to_string())
    }

    fn definition(&mut self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let p: GotoDefinitionParams = serde_json::from_value(params).map_err(|e| e.to_string())?;
        let at = &p.text_document_position_params;
        let found = self
            .at(&at.text_document.uri, at.position)
            .and_then(|(checked, id, offset)| crate::navigate::definition(&checked, id, offset, self.enc))
            .map(GotoDefinitionResponse::Scalar);
        serde_json::to_value(found).map_err(|e| e.to_string())
    }

    fn symbols(&mut self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let p: DocumentSymbolParams = serde_json::from_value(params).map_err(|e| e.to_string())?;
        let url = &p.text_document.uri;
        let symbols = self
            .fresh(url)
            .and_then(|checked| {
                let id = checked.file(&url.to_file_path().ok()?)?;
                Some(crate::navigate::symbols(&checked, id, self.enc))
            })
            .unwrap_or_default();
        serde_json::to_value(DocumentSymbolResponse::Nested(symbols)).map_err(|e| e.to_string())
    }

    fn completion(&mut self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let p: CompletionParams = serde_json::from_value(params).map_err(|e| e.to_string())?;
        let at = &p.text_document_position;
        let url = &at.text_document.uri;
        let (Some(text), Ok(path)) = (self.docs.get(url), url.to_file_path()) else {
            return Ok(serde_json::Value::Null);
        };
        let offset = Lines::new(text).offset(at.position, self.enc);
        let typed = self.typed.get(url).map(Rc::as_ref);
        let items = crate::complete::complete(typed, text, offset, &path);
        serde_json::to_value(CompletionResponse::Array(items)).map_err(|e| e.to_string())
    }
}
