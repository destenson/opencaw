//! Sysdoc workload.
//!
//! Runs Q&A against a pre-built index of a snapshotted documentation
//! corpus (typically `/usr/share/doc` via `scripts/snapshot-corpus.sh`).
//! The Q&A JSON is produced by `scripts/generate-qa.py`; `expected_paths`
//! are relative to the corpus root the index was built against.
//!
//! Ingestion is done ahead of time by the `caw-bench-build-index` binary
//! and is not part of the benchmark run. Workload items here carry only
//! the question + scoring metadata — the corpus lives in the prebuilt
//! index, shared across all items in a run.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

use crate::workload::{Scoring, WorkloadItem};

#[derive(Debug, Deserialize)]
struct RawQa {
    id: String,
    question: String,
    reference_answer: String,
    expected_paths: Vec<String>,
}

pub fn build(qa_file: &Path) -> Result<Vec<WorkloadItem>> {
    let qa_text = std::fs::read_to_string(qa_file)
        .with_context(|| format!("read qa-file {}", qa_file.display()))?;
    let raw: Vec<RawQa> = serde_json::from_str(&qa_text).context("parse sysdoc QA JSON")?;

    Ok(raw
        .into_iter()
        .map(|q| WorkloadItem {
            id: q.id,
            question: q.question,
            corpus: Vec::new(),
            expected_paths: q.expected_paths,
            scoring: Scoring::JudgeAgainst {
                reference_answer: q.reference_answer,
            },
        })
        .collect())
}
