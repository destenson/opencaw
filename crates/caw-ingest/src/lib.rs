pub mod chunking;
pub mod summarizer;
mod tree_sitter_outline;

use caw_core::tokenizer::TiktokenTokenizer;
use caw_core::{CawError, CawResult, ContentKind, Stub, StubId, Tokenizer, WhitespaceTokenizer};
use chunking::{ChunkingConfig, chunk_document, chunk_summary};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use summarizer::{DeterministicSummarizer, Summarizer};
use walkdir::WalkDir;

/// Default tokenizer for the ingestion pipeline.
/// cl100k_base matches GPT-4 / GPT-3.5 and is close enough for Claude's
/// vocabulary that budget accounting stays within a few percent. Whitespace
/// counting (the previous default) systematically underestimates by ~30% for
/// prose and ~50% for code, which corrupts every downstream budget decision.
fn default_tokenizer() -> Arc<dyn Tokenizer> {
    match TiktokenTokenizer::cl100k() {
        Ok(t) => Arc::new(t),
        Err(_) => Arc::new(WhitespaceTokenizer),
    }
}

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
    tokenizer: Arc<dyn Tokenizer>,
    chunking: Option<ChunkingConfig>,
}

impl Default for IngestionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl IngestionPipeline {
    pub fn new() -> Self {
        Self {
            summarizer: Box::new(DeterministicSummarizer),
            tokenizer: default_tokenizer(),
            chunking: Some(ChunkingConfig::default()),
        }
    }

    pub fn with_summarizer(summarizer: Box<dyn Summarizer>) -> Self {
        Self {
            summarizer,
            tokenizer: default_tokenizer(),
            chunking: Some(ChunkingConfig::default()),
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn with_chunking(mut self, config: Option<ChunkingConfig>) -> Self {
        self.chunking = config;
        self
    }

    /// Ingest a document, returning `(stub, embed_text)` pairs. Large files
    /// are split into multiple chunks, each producing its own stub paired
    /// with the chunk's embed text (body + overlap prefix — what gets fed
    /// into the embedder, never persisted). Files below the chunking
    /// threshold produce a single pair whose embed text is the whole doc.
    ///
    /// Callers must not substitute the full document text for a chunk's
    /// embed text: chunk stubs embedded against the full doc all collapse
    /// to one doc-level embedding, making chunk-level retrieval meaningless.
    pub fn ingest(&self, doc: SourceDocument) -> Vec<(Stub, String)> {
        let outline = extract_outline(doc.kind, &doc.content, &doc.path);
        let content_hash = sha256_hash(&doc.content);

        // Token estimate for the single-stub path. When chunking is enabled,
        // chunk_document already computes this (via its cheap upper bound or
        // a real tokenize), so we use its answer and avoid redundant work.
        // When chunking is disabled we fall back to tokenizing directly.
        let mut single_token_estimate: Option<usize> = None;

        if let Some(ref config) = self.chunking {
            let chunks = chunk_document(&doc.content, doc.kind, &outline, config, &self.tokenizer);
            if chunks.len() > 1 {
                return chunks
                    .into_iter()
                    .map(|chunk| {
                        let chunk_hash = sha256_hash(&chunk.content);
                        // Reuse the token count chunk_document already computed.
                        // Previously this re-ran the BPE tokenizer per chunk,
                        // doubling tokenization cost on every chunked file.
                        let chunk_token_estimate = chunk.token_count;
                        let position_summary = chunk_summary(&doc.path, &chunk);
                        let base_summary = self
                            .summarizer
                            .summarize(&doc.path, &chunk.content, doc.kind, &chunk.outline_entries)
                            .unwrap_or_else(|_| {
                                DeterministicSummarizer
                                    .summarize(
                                        &doc.path,
                                        &chunk.content,
                                        doc.kind,
                                        &chunk.outline_entries,
                                    )
                                    .unwrap_or_default()
                            });

                        let summary = if position_summary.is_empty() {
                            base_summary
                        } else if base_summary.is_empty() {
                            position_summary
                        } else {
                            format!("{} — {}", position_summary, base_summary)
                        };

                        let stub = Stub {
                            id: StubId(format!("{}#chunk{}", doc.path, chunk.index)),
                            path: doc.path.clone(),
                            token_estimate: chunk_token_estimate,
                            kind: doc.kind,
                            summary,
                            outline: chunk.outline_entries.clone(),
                            content_hash: chunk_hash,
                            mtime_unix_secs: doc.mtime_unix_secs,
                            byte_offset: chunk.body_start,
                            byte_length: chunk.body_end.saturating_sub(chunk.body_start),
                            consolidation_notes: Vec::new(),
                        };
                        (stub, chunk.content)
                    })
                    .collect();
            }
            // Single chunk returned from chunk_document — grab its precomputed
            // token count so the single-stub path below doesn't re-tokenize.
            single_token_estimate = chunks.into_iter().next().map(|c| c.token_count);
        }

        // Single-stub path: file is small or chunking is disabled
        let summary = self
            .summarizer
            .summarize(&doc.path, &doc.content, doc.kind, &outline)
            .unwrap_or_else(|_| {
                DeterministicSummarizer
                    .summarize(&doc.path, &doc.content, doc.kind, &outline)
                    .unwrap_or_default()
            });
        let token_estimate =
            single_token_estimate.unwrap_or_else(|| self.tokenizer.count_tokens(&doc.content));

        let body_length = doc.content.len() as u64;
        let doc_path = doc.path.clone();
        let doc_content = doc.content;
        vec![(
            Stub {
                id: StubId(doc_path.clone()),
                path: doc_path,
                token_estimate,
                kind: doc.kind,
                summary,
                outline,
                content_hash,
                mtime_unix_secs: doc.mtime_unix_secs,
                byte_offset: 0,
                byte_length: body_length,
                consolidation_notes: Vec::new(),
            },
            doc_content,
        )]
    }

    /// Ingest all supported files under a directory.
    ///
    /// Directory walk is sequential (walkdir isn't cheap to parallelize and
    /// the bottleneck isn't here), but per-file work — read, hash, outline
    /// extraction (tree-sitter), summarization, chunking — runs in parallel
    /// via rayon. On large corpora this is the dominant win; embedding
    /// throughput afterwards is gated by the ONNX runtime's own threading.
    pub fn ingest_directory(&self, root: &Path) -> CawResult<Vec<(Stub, String)>> {
        let paths: Vec<std::path::PathBuf> = WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file() && !should_skip(e.path()))
            .map(|e| e.path().to_path_buf())
            .collect();

        let results: Vec<(Stub, String)> = paths
            .par_iter()
            .filter_map(|path| SourceDocument::from_path(path).ok())
            .flat_map_iter(|doc| self.ingest(doc).into_iter())
            .collect();

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

    // Lock files, binaries, and archives. Archive extensions (zip/tar/gz/…)
    // are in here because package/archive inspection is not yet implemented
    // — so if the corpus contains docs inside archives they're silently
    // dropped. Callers that need visibility into what was skipped should
    // walk paths themselves and flag archives before handing them to the
    // pipeline (see caw-bench-build-index for an example).
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
                | "7z"
                | "rar"
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
