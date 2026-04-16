pub mod summarizer;
mod tree_sitter_outline;

use caw_core::{CawError, CawResult, ContentKind, Stub, StubId};
use sha2::{Digest, Sha256};
use std::path::Path;
use summarizer::{DeterministicSummarizer, Summarizer};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct SourceDocument {
    pub path: String,
    pub content: String,
    pub kind: ContentKind,
    pub mtime_unix_secs: u64,
}

impl SourceDocument {
    /// Load a document from the filesystem, reading content, mtime, and detecting kind
    pub fn from_path(path: &Path) -> CawResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CawError::Io(format!("{}: {}", path.display(), e)))?;

        let metadata = std::fs::metadata(path)
            .map_err(|e| CawError::Io(format!("{}: {}", path.display(), e)))?;

        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let kind = detect_content_kind(path);

        Ok(Self {
            path: path.to_string_lossy().into_owned(),
            content,
            kind,
            mtime_unix_secs: mtime,
        })
    }
}

pub struct IngestionPipeline {
    summarizer: Box<dyn Summarizer>,
}

impl IngestionPipeline {
    pub fn new() -> Self {
        Self {
            summarizer: Box::new(DeterministicSummarizer),
        }
    }

    pub fn with_summarizer(summarizer: Box<dyn Summarizer>) -> Self {
        Self { summarizer }
    }

    pub fn ingest(&self, doc: SourceDocument) -> Stub {
        let outline = extract_outline(doc.kind, &doc.content, &doc.path);
        let summary = self
            .summarizer
            .summarize(&doc.path, &doc.content, doc.kind, &outline)
            .unwrap_or_else(|_| {
                // LLM failure degrades to deterministic
                DeterministicSummarizer
                    .summarize(&doc.path, &doc.content, doc.kind, &outline)
                    .unwrap_or_default()
            });
        let token_estimate = estimate_tokens(&doc.content);
        let content_hash = sha256_hash(&doc.content);

        Stub {
            id: StubId(doc.path.clone()),
            path: doc.path,
            token_estimate,
            kind: doc.kind,
            summary,
            outline,
            content_hash,
            mtime_unix_secs: doc.mtime_unix_secs,
            consolidation_notes: Vec::new(),
        }
    }

    /// Ingest all supported files under a directory
    pub fn ingest_directory(&self, root: &Path) -> CawResult<Vec<(Stub, String)>> {
        let mut results = Vec::new();

        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            if should_skip(path) {
                continue;
            }

            match SourceDocument::from_path(path) {
                Ok(doc) => {
                    let content = doc.content.clone();
                    let stub = self.ingest(doc);
                    results.push((stub, content));
                }
                Err(_) => {
                    // Skip files that can't be read as UTF-8 (binary files, etc.)
                    continue;
                }
            }
        }

        Ok(results)
    }
}

fn extract_outline(kind: ContentKind, content: &str, path: &str) -> Vec<String> {
    match kind {
        ContentKind::Markdown => content
            .lines()
            .filter(|l| l.trim_start().starts_with('#'))
            .map(|l| l.trim_start_matches('#').trim().to_string())
            .filter(|l| !l.is_empty())
            .take(20)
            .collect(),
        ContentKind::Code => {
            let extension = Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            // Tree-sitter gives structural outlines; fall back to string matching
            // for languages without grammar support.
            if let Some(outline) =
                tree_sitter_outline::extract_outline_tree_sitter(content, extension)
            {
                outline
            } else {
                extract_outline_naive(content)
            }
        }
        _ => vec![],
    }
}

/// Fallback outline extraction for languages without tree-sitter grammars.
fn extract_outline_naive(content: &str) -> Vec<String> {
    let mut outline = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("pub fn ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("trait ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("mod ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("export function ")
            || trimmed.starts_with("export class ")
            || trimmed.starts_with("export interface ")
            || trimmed.starts_with("export type ")
        {
            outline.push(trimmed.to_string());
        }

        if outline.len() >= 30 {
            break;
        }
    }
    outline
}

/// Estimate token count using whitespace splitting as a rough
/// subword-tokenizer approximation. More accurate than len/4 for
/// mixed prose/code content.
fn estimate_tokens(content: &str) -> usize {
    content.split_whitespace().count().max(1)
}

fn sha256_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

fn detect_content_kind(path: &Path) -> ContentKind {
    match path.extension().and_then(|e| e.to_str()) {
        Some("md" | "mdx" | "markdown") => ContentKind::Markdown,
        Some(
            "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "c" | "cpp" | "h" | "hpp" | "java"
            | "rb" | "ex" | "exs" | "zig" | "lua" | "sh" | "bash" | "zsh" | "cs" | "swift" | "kt"
            | "scala" | "r" | "R" | "pl" | "pm" | "php",
        ) => ContentKind::Code,
        Some("csv" | "tsv" | "parquet") => ContentKind::Tabular,
        Some("txt" | "log" | "cfg" | "conf" | "ini" | "env") => ContentKind::PlainText,
        Some("toml" | "yaml" | "yml" | "json" | "xml") => ContentKind::PlainText,
        Some("srt" | "vtt") => ContentKind::Transcript,
        _ => ContentKind::Other,
    }
}

/// Skip files that are unlikely to be useful for context recall
fn should_skip(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    // Hidden files and directories
    if name.starts_with('.') {
        return true;
    }

    // Check path components for hidden/build directories
    for component in path.components() {
        if let std::path::Component::Normal(c) = component {
            let s = c.to_str().unwrap_or("");
            if s.starts_with('.')
                || s == "target"
                || s == "node_modules"
                || s == "__pycache__"
                || s == ".git"
                || s == "dist"
                || s == "build"
                || s == "vendor"
            {
                return true;
            }
        }
    }

    // Lock files, binaries, etc.
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some(
            "lock"
                | "png"
                | "jpg"
                | "jpeg"
                | "gif"
                | "ico"
                | "svg"
                | "woff"
                | "woff2"
                | "ttf"
                | "eot"
                | "otf"
                | "zip"
                | "tar"
                | "gz"
                | "bz2"
                | "xz"
                | "exe"
                | "dll"
                | "so"
                | "dylib"
                | "o"
                | "a"
                | "wasm"
                | "pyc"
                | "pyo"
                | "class"
        )
    )
}
