//! The server driven as a client drives it.

use std::path::Path;
use std::time::Duration;

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use lsp_types::notification::{
    DidChangeTextDocument, DidOpenTextDocument, Exit, Initialized, Notification as _, PublishDiagnostics,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, Initialize, Shutdown,
};
use lsp_types::{
    ClientCapabilities, CompletionParams, CompletionResponse, Diagnostic, DidChangeTextDocumentParams,
    DidOpenTextDocumentParams, DocumentSymbolParams, DocumentSymbolResponse, GeneralClientCapabilities,
    GotoDefinitionParams, GotoDefinitionResponse, HoverContents, HoverParams, InitializeParams, Position,
    PositionEncodingKind, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem,
    TextDocumentPositionParams, Url, VersionedTextDocumentIdentifier,
};

/// A unit importing another.
const SHAPES: &str = "#unit classical
import geo;

// A point in the plane.
struct Point { x: i32, y: i32 }

// How far `p` is from the origin, in steps.
fn steps(p: *Point) -> i32 {
    let dx = p.x;
    dx + p.y
}

fn main() -> i32 {
    let p = Point { x: 1, y: 2 };
    steps(&p) + geo::double(3)
}
";

const GEO: &str = "#unit classical

// Twice `n`.
fn double(n: i32) -> i32 { n * 2 }
";

struct Client {
    conn: Connection,
    server: Option<std::thread::JoinHandle<()>>,
    next: i32,
    dir: tempfile::TempDir,
}

impl Client {
    /// A server, initialised, for a directory holding `geo.tq`, whose client
    /// counts in bytes when `utf8`.
    fn start(utf8: bool) -> Client {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("geo.tq"), GEO).unwrap();
        let (server, conn) = Connection::memory();
        let server = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || crate::server::serve(server).unwrap())
            .unwrap();
        let mut client = Client {
            conn,
            server: Some(server),
            next: 0,
            dir,
        };
        let params = InitializeParams {
            capabilities: ClientCapabilities {
                general: utf8.then(|| GeneralClientCapabilities {
                    position_encodings: Some(vec![PositionEncodingKind::UTF8, PositionEncodingKind::UTF16]),
                    ..GeneralClientCapabilities::default()
                }),
                ..ClientCapabilities::default()
            },
            ..InitializeParams::default()
        };
        let _ = client.request::<Initialize>(params);
        client.notify::<Initialized>(lsp_types::InitializedParams {});
        client
    }

    fn url(&self, name: &str) -> Url {
        Url::from_file_path(self.dir.path().join(name)).unwrap()
    }

    fn notify<N: lsp_types::notification::Notification>(&self, params: N::Params) {
        self.conn.sender.send(Notification::new(N::METHOD.to_owned(), params).into()).unwrap();
    }

    fn request<R: lsp_types::request::Request>(&mut self, params: R::Params) -> R::Result {
        self.next += 1;
        let id = RequestId::from(self.next);
        self.conn.sender.send(Request::new(id.clone(), R::METHOD.to_owned(), params).into()).unwrap();
        loop {
            match self.conn.receiver.recv_timeout(Duration::from_secs(60)).unwrap() {
                Message::Response(r) if r.id == id => {
                    let result = r.response_result.unwrap_or_else(|e| panic!("{e:?}"));
                    return serde_json::from_value(result).unwrap();
                }
                _ => {}
            }
        }
    }

    /// The next diagnostics published for `url`.
    fn diagnostics(&self, url: &Url) -> Vec<Diagnostic> {
        loop {
            if let Message::Notification(n) = self.conn.receiver.recv_timeout(Duration::from_secs(60)).unwrap()
                && n.method == PublishDiagnostics::METHOD
            {
                let p: lsp_types::PublishDiagnosticsParams = serde_json::from_value(n.params).unwrap();
                if p.uri == *url {
                    return p.diagnostics;
                }
            }
        }
    }

    fn open(&self, name: &str, text: &str) -> Url {
        let url = self.url(name);
        self.notify::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem::new(url.clone(), "topiq".to_owned(), 1, text.to_owned()),
        });
        url
    }

    fn at(url: &Url, line: u32, character: u32) -> TextDocumentPositionParams {
        TextDocumentPositionParams::new(TextDocumentIdentifier::new(url.clone()), Position::new(line, character))
    }

    fn hover(&mut self, url: &Url, line: u32, character: u32) -> String {
        let h = self.request::<HoverRequest>(HoverParams {
            text_document_position_params: Client::at(url, line, character),
            work_done_progress_params: Default::default(),
        });
        match h.map(|h| h.contents) {
            Some(HoverContents::Markup(m)) => m.value,
            other => panic!("no hover at {line}:{character}: {other:?}"),
        }
    }

    fn definition(&mut self, url: &Url, line: u32, character: u32) -> (Url, Position) {
        let d = self.request::<GotoDefinition>(GotoDefinitionParams {
            text_document_position_params: Client::at(url, line, character),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        });
        match d {
            Some(GotoDefinitionResponse::Scalar(l)) => (l.uri, l.range.start),
            other => panic!("no definition at {line}:{character}: {other:?}"),
        }
    }

    fn completion(&mut self, url: &Url, line: u32, character: u32) -> Vec<String> {
        let c = self.request::<Completion>(CompletionParams {
            text_document_position: Client::at(url, line, character),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
            context: None,
        });
        match c {
            Some(CompletionResponse::Array(items)) => items.into_iter().map(|i| i.label).collect(),
            other => panic!("no completion at {line}:{character}: {other:?}"),
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.request::<Shutdown>(());
        self.notify::<Exit>(());
        if let Some(server) = self.server.take() {
            server.join().unwrap();
        }
    }
}

fn line_of(text: &str, needle: &str) -> u32 {
    text.lines().position(|l| l.contains(needle)).unwrap() as u32
}

fn column_of(text: &str, needle: &str, within: &str) -> (u32, u32) {
    let line = line_of(text, within);
    let col = text.lines().nth(line as usize).unwrap().find(needle).unwrap() as u32;
    (line, col)
}

#[test]
fn an_unsaved_program_checks_clean_then_reports_what_an_edit_breaks() {
    let mut c = Client::start(false);
    // the file is only in the editor, never written
    let url = c.open("shapes.tq", SHAPES);
    assert!(!Path::new(&url.to_file_path().unwrap()).exists());
    assert_eq!(c.diagnostics(&url), Vec::new());

    let broken = SHAPES.replace("let dx = p.x;", "let dx: bool = p.x;");
    c.notify::<DidChangeTextDocument>(DidChangeTextDocumentParams {
        text_document: VersionedTextDocumentIdentifier::new(url.clone(), 2),
        content_changes: vec![TextDocumentContentChangeEvent {
            range: None,
            range_length: None,
            text: broken.clone(),
        }],
    });
    let found = c.diagnostics(&url);
    let codes: Vec<_> = found.iter().filter_map(|d| d.code.clone()).collect();
    assert!(codes.contains(&lsp_types::NumberOrString::String("ES06".to_owned())), "{found:?}");
    let es06 = found.iter().find(|d| d.code == Some(lsp_types::NumberOrString::String("ES06".to_owned()))).unwrap();
    assert_eq!(es06.range.start.line, line_of(&broken, "let dx: bool"));
    // a request after an edit sees the edit
    let (line, col) = column_of(&broken, "dx", "let dx: bool");
    assert!(c.hover(&url, line, col).contains("let dx: bool"));
}

#[test]
fn hover_shows_types_signatures_and_the_comment_above() {
    let mut c = Client::start(false);
    let url = c.open("shapes.tq", SHAPES);
    let (line, col) = column_of(SHAPES, "dx", "dx + p.y");
    assert!(c.hover(&url, line, col).contains("let dx: i32"));

    let (line, col) = column_of(SHAPES, "steps", "steps(&p)");
    let shown = c.hover(&url, line, col);
    assert!(shown.contains("fn steps(p: *Point) -> i32"), "{shown}");
    assert!(shown.contains("How far `p` is from the origin"), "{shown}");

    let (line, col) = column_of(SHAPES, "x", "let dx = p.x;");
    let col = col + "x = p.".len() as u32;
    assert!(c.hover(&url, line, col).contains("Point.x: i32"));

    let (line, col) = column_of(SHAPES, "double", "geo::double");
    let shown = c.hover(&url, line, col);
    assert!(shown.contains("fn geo::double(i32) -> i32"), "{shown}");
    assert!(shown.contains("Twice"), "{shown}");
}

#[test]
fn definitions_are_found_in_this_unit_and_in_the_units_it_imports() {
    let mut c = Client::start(false);
    let url = c.open("shapes.tq", SHAPES);

    let (line, col) = column_of(SHAPES, "steps", "steps(&p)");
    let (to, at) = c.definition(&url, line, col);
    assert_eq!((to, at.line), (url.clone(), line_of(SHAPES, "fn steps")));

    let (line, col) = column_of(SHAPES, "Point", "let p = Point");
    let (_, at) = c.definition(&url, line, col);
    assert_eq!(at.line, line_of(SHAPES, "struct Point"));

    let (line, col) = column_of(SHAPES, "double", "geo::double");
    let (to, at) = c.definition(&url, line, col);
    assert_eq!((to, at.line), (c.url("geo.tq"), line_of(GEO, "fn double")));

    let (line, col) = column_of(SHAPES, "geo", "import geo");
    let (to, _) = c.definition(&url, line, col);
    assert_eq!(to, c.url("geo.tq"));
}

#[test]
fn the_outline_lists_items_with_their_fields() {
    let mut c = Client::start(false);
    let url = c.open("shapes.tq", SHAPES);
    let symbols = c.request::<DocumentSymbolRequest>(DocumentSymbolParams {
        text_document: TextDocumentIdentifier::new(url),
        work_done_progress_params: Default::default(),
        partial_result_params: Default::default(),
    });
    let Some(DocumentSymbolResponse::Nested(symbols)) = symbols else { panic!("{symbols:?}") };
    let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["geo", "Point", "steps", "main"]);
    let fields: Vec<&str> = symbols[1].children.iter().flatten().map(|s| s.name.as_str()).collect();
    assert_eq!(fields, ["x", "y"]);
}

#[test]
fn completion_offers_fields_after_a_dot_and_items_after_a_unit() {
    let mut c = Client::start(true);
    let url = c.open("shapes.tq", SHAPES);
    assert_eq!(c.diagnostics(&url), Vec::new());

    // `p.` where `p.x` was: the last typed check still knows `p`
    let (line, col) = column_of(SHAPES, "x", "let dx = p.x;");
    let col = col + "x = p.".len() as u32;
    let offered = c.completion(&url, line, col);
    assert!(offered.contains(&"x".to_owned()) && offered.contains(&"y".to_owned()), "{offered:?}");

    let (line, col) = column_of(SHAPES, "double", "geo::double");
    assert!(c.completion(&url, line, col).contains(&"double".to_owned()));

    let (line, _) = column_of(SHAPES, "dx", "dx + p.y");
    let offered = c.completion(&url, line, 4);
    for want in ["dx", "p", "steps", "Point", "match", "Opt"] {
        assert!(offered.contains(&want.to_owned()), "{want} in {offered:?}");
    }
}
