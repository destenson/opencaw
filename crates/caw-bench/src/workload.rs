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
    /// Cheap, deterministic — used for NIAH-style retrieval tests.
    ContainsNeedle { needle: String },
    /// Score by semantic match against a reference answer using a judge model.
    /// Used when the answer is open-ended (e.g., opencaw Q&A).
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
