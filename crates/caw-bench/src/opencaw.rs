//! opencaw-on-opencaw workload.
//!
//! Hand-authored Q&A against the opencaw repository itself. Questions
//! target specific facts drawn from the code, README, SCOPE.md, TODO.md,
//! and the design doc. Reference answers are short and concrete so the
//! model-as-judge scoring is stable across runs.

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::workload::{CorpusDoc, Scoring, WorkloadItem};
use caw_core::ContentKind;
use caw_ingest::SourceDocument;
use std::path::Path;

// Embedded fallback so the binary is self-contained when no --qa-file is given.
const QA_JSON_EMBEDDED: &str = include_str!("qa/opencaw_qa.json");

fn load_qa_json(path: Option<&std::path::Path>) -> Result<String> {
    if let Some(p) = path {
        std::fs::read_to_string(p).with_context(|| format!("read qa-file {}", p.display()))
    } else {
        Ok(QA_JSON_EMBEDDED.to_owned())
    }
}

#[derive(Debug, Deserialize)]
struct RawQa {
    id: String,
    question: String,
    reference_answer: String,
    /// Paths (relative to the opencaw repo root) the answer should draw from.
    /// Used to score recall@k.
    expected_paths: Vec<String>,
}

/// Build the opencaw Q&A workload. `repo_root` is the opencaw checkout
/// whose files become the corpus. If `qa_file` is provided the Q&A JSON
/// is loaded from disk at runtime; otherwise the binary's embedded copy
/// is used, keeping the binary self-contained.
pub fn build(repo_root: &Path, qa_file: Option<&Path>) -> Result<Vec<WorkloadItem>> {
    let json = load_qa_json(qa_file)?;
    let raw: Vec<RawQa> = serde_json::from_str(&json).context("parse opencaw_qa.json")?;

    let corpus = load_corpus(repo_root)?;
    if corpus.is_empty() {
        anyhow::bail!(
            "no files found under {} — check that --repo-root points at an opencaw checkout",
            repo_root.display()
        );
    }

    let items = raw
        .into_iter()
        .map(|q| WorkloadItem {
            id: q.id,
            question: q.question,
            corpus: corpus.clone(),
            expected_paths: q.expected_paths,
            scoring: Scoring::JudgeAgainst {
                reference_answer: q.reference_answer,
            },
        })
        .collect();

    Ok(items)
}

/// Walk the repo, loading files we expect the Q&A to ask about. Binary
/// and build-artifact paths are excluded by the ingest pipeline's default
/// skip rules.
fn load_corpus(root: &Path) -> Result<Vec<CorpusDoc>> {
    let entries = walk_source_files(root);
    let mut docs = Vec::new();
    for abs_path in entries {
        let rel = abs_path
            .strip_prefix(root)
            .unwrap_or(&abs_path)
            .to_string_lossy()
            .into_owned();

        let Ok(source) = SourceDocument::from_path(&abs_path) else {
            continue;
        };

        docs.push(CorpusDoc {
            path: rel,
            content: source.content,
            kind: source.kind,
        });
    }
    Ok(docs)
}

/// Recursively walk a directory, returning files that should be indexed
/// for the opencaw corpus. Skips build artifacts, hidden dirs, and
/// large/binary files the design doc wouldn't reference.
fn walk_source_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // Skip build outputs, VCS, editor state, nested workspaces,
            // bench artifacts, and the sysdoc test corpora (not opencaw
            // project source — ingesting 1700+ synthetic markdown files
            // blows up the shared-corpus build for no retrieval signal
            // against the opencaw Q&A).
            if matches!(
                name,
                "target"
                    | ".git"
                    | "node_modules"
                    | ".claude"
                    | "opencaw-corpora"
                    | "bench-results"
                    | "data"
            ) {
                continue;
            }
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if let Some("rs" | "md" | "toml") = path.extension().and_then(|e| e.to_str()) {
                out.push(path)
            }
            // match path.extension().and_then(|e| e.to_str()) {
            //     Some("rs" | "md" | "toml") => out.push(path),
            //     _ => {}
            // }
        }
    }
    out.sort();
    out
}

/// Kind detection fallback for paths we construct without extension info.
fn kind_for_path(p: &str) -> ContentKind {
    if p.ends_with(".md") {
        ContentKind::Markdown
    } else if p.ends_with(".rs") {
        ContentKind::Code
    } else {
        ContentKind::PlainText
    }
}
