//! Code-agent workload.
//!
//! Models the questions a coding agent asks itself mid-task while working in a
//! repository: "what fields does this struct have", "what's the signature of
//! this method", "what trait bounds does this declare", "where is this symbol
//! called". These are the lookups an agent would otherwise satisfy by grepping
//! and reading files — the grep-archaeology that OpenCAW exists to replace.
//!
//! The corpus is a checkout of this repo passed via `--repo-root`. The bench
//! points this at a FROZEN snapshot pinned to the commit that authored the QA
//! file (see `docs/skills/caw-dev/scripts/freeze-codeagent-corpus.sh`), so
//! every needle the QA asks for exists in the corpus by construction and the
//! gauge is reproducible as the live repo drifts — the bench is a frozen
//! gauge, not a dogfood run on the live repo (that is `caw-server`'s job). The
//! distinction from the `opencaw` workload is the question framing (agent
//! info-needs, not project facts) and per-item scoring. Scoring is two-phase
//! for exact code facts (a field name, a default value, a crate): the
//! deterministic needle check is primary — a missing token is a hard 0.0 with
//! no judge — and the judge runs only to confirm a present token is asserted
//! rather than quoted inside a denial (`Scoring::NeedleWithJudgeConfirm`). This
//! keeps the exact-match precision a bare substring gives while closing the two
//! gaps it leaves: crediting a quote-while-denying refusal, and (had we judged
//! everything) crediting a semantic near-miss like `summaries` for
//! `stub_summaries`. Synthesis questions (call-site sets, signatures the model
//! paraphrases) use judge scoring against a reference answer.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

use crate::opencaw;
use crate::workload::{Scoring, WorkloadItem};

// Embedded fallback so the binary is self-contained when no --qa-file is given.
const QA_JSON_EMBEDDED: &str = include_str!("qa/codeagent_qa.json");

#[derive(Debug, Deserialize)]
struct RawQa {
    id: String,
    question: String,
    /// Paths (relative to the repo root) the answer should draw from. Used to
    /// score recall@k. For call-site questions this must list *every* call
    /// site, not just one, or the recall ground truth is wrong.
    expected_paths: Vec<String>,
    /// Two-phase scoring (`Scoring::NeedleWithJudgeConfirm`): the answer must
    /// contain this exact substring (case-insensitive) to score above 0.0, and
    /// when it does the judge confirms the token is asserted rather than quoted
    /// inside a refusal. Use a single distinctive token greedy decoding will
    /// emit verbatim (a field name, a crate, a default value) — never a full
    /// signature or phrase the model will paraphrase.
    #[serde(default)]
    needle: Option<String>,
    /// Judge scoring: used when `needle` is absent. The reference answer the
    /// judge model compares the model's answer against.
    #[serde(default)]
    reference_answer: Option<String>,
}

fn load_qa_json(path: Option<&Path>) -> Result<String> {
    if let Some(p) = path {
        std::fs::read_to_string(p).with_context(|| format!("read qa-file {}", p.display()))
    } else {
        Ok(QA_JSON_EMBEDDED.to_owned())
    }
}

/// Build the code-agent Q&A workload. `repo_root` is the checkout whose files
/// become the corpus. If `qa_file` is provided it is loaded at runtime;
/// otherwise the binary's embedded copy is used.
pub fn build(repo_root: &Path, qa_file: Option<&Path>) -> Result<Vec<WorkloadItem>> {
    let json = load_qa_json(qa_file)?;
    let raw: Vec<RawQa> = serde_json::from_str(&json).context("parse codeagent_qa.json")?;

    let corpus = opencaw::load_corpus(repo_root)?;
    if corpus.is_empty() {
        anyhow::bail!(
            "no files found under {} — check that --repo-root points at the repo checkout",
            repo_root.display()
        );
    }

    let mut items = Vec::with_capacity(raw.len());
    for q in raw {
        let scoring = match (q.needle, q.reference_answer) {
            (Some(needle), _) => Scoring::NeedleWithJudgeConfirm { needle },
            (None, Some(reference_answer)) => Scoring::JudgeAgainst { reference_answer },
            (None, None) => anyhow::bail!(
                "codeagent QA item {} has neither `needle` nor `reference_answer`",
                q.id
            ),
        };
        items.push(WorkloadItem {
            id: q.id,
            question: q.question,
            corpus: corpus.clone(),
            expected_paths: q.expected_paths,
            scoring,
        });
    }

    Ok(items)
}
