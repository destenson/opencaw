use caw_core::ContentKind;
use serde::{Deserialize, Serialize};

/// A single benchmark question paired with the corpus it is to be asked against.
///
/// Each item is self-contained: the corpus is materialized fresh per item so
/// runs are reproducible and side effects (consolidation notes, eviction
/// history) cannot leak between items.
#[derive(Debug, Clone)]
pub struct WorkloadItem {
    pub id: String,
    pub question: String,
    /// Documents to ingest before asking the question. Each doc has a stable
    /// `path` field used for ground-truth recall scoring.
    pub corpus: Vec<CorpusDoc>,
    /// Stable paths the question is *expected* to recall. Used for recall@k.
    pub expected_paths: Vec<String>,
    /// Per-item scoring rule.
    pub scoring: Scoring,
}

#[derive(Debug, Clone)]
pub struct CorpusDoc {
    pub path: String,
    pub content: String,
    pub kind: ContentKind,
}

#[derive(Debug, Clone)]
pub enum Scoring {
    /// Pass if the answer (case-insensitive) contains the needle string.
    /// Cheap, deterministic — used for NIAH-style retrieval tests where the
    /// presence of the token *is* the answer.
    ContainsNeedle { needle: String },
    /// Two-phase scoring for exact-fact items. The needle (case-insensitive)
    /// substring check is the *primary* signal: if the answer does not contain
    /// the exact token, the item scores 0.0 with no judge call — this is exact
    /// and correct on token absence, and avoids the judge crediting a semantic
    /// near-miss (e.g. `summaries` for `stub_summaries`). If the token *is*
    /// present, the judge runs only to confirm the answer *asserts* the fact
    /// rather than quoting it while denying knowledge ("the context does not
    /// state that `Foo`…" mentions the token but is a refusal, not an answer).
    NeedleWithJudgeConfirm { needle: String },
    /// Score by semantic match against a reference answer using a judge model.
    /// Used when the answer is open-ended (e.g., opencaw Q&A) and the model is
    /// expected to paraphrase rather than emit a verbatim token.
    JudgeAgainst { reference_answer: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecallMode {
    /// Full dynamic recall: initial retrieval, multi-pass refinement,
    /// thinking-trace + probe recall, eviction, consolidation.
    On,
    /// Initial retrieval only — single-shot RAG. No probes, no traces, no
    /// multi-pass refinement. Same context budget. Isolates the contribution
    /// of the dynamic recall machinery.
    Off,
}

impl RecallMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::On => "recall_on",
            Self::Off => "recall_off",
        }
    }
}
