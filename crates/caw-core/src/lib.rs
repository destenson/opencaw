pub mod provenance;
pub mod reindex;
pub mod scheduler;
pub mod tokenizer;

pub use provenance::tokenize_terms;
pub use reindex::{ChannelReindexQueue, NoopReindexQueue, ReindexQueue, ReindexReceiver};
pub use tokenizer::{count_tokens_cl100k, extract_token_range};

use std::{collections::{HashMap, HashSet}, fmt::Display};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type CawResult<T> = Result<T, CawError>;

#[derive(Debug, Error)]
pub enum CawError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("vector store error: {0}")]
    VectorStore(String),
    #[error("io error: {0}")]
    Io(String),
    /// The requested stub's source file has changed or been removed since
    /// indexing. The stub has been marked stale in the store and (if a
    /// reindex queue is attached) enqueued for reingestion. Callers should
    /// treat this as a miss, not a hard failure.
    #[error("stale stub at {path}")]
    StaleStub { path: String },
    /// The model produced degenerate looping output. The `sample` field
    /// contains the first 120 chars of the response for diagnostics.
    #[error("degenerate output from {model}: {sample}...")]
    DegenerateOutput { model: String, sample: String },
    /// An error originating outside the core library (e.g. from an adapter,
    /// HTTP client, or external service). Use this only when no more specific
    /// variant applies.
    #[error("{0}")]
    External(String),
}

impl From<std::io::Error> for CawError {
    fn from(e: std::io::Error) -> Self {
        CawError::Io(e.to_string())
    }
}

impl From<std::fmt::Error> for CawError {
    fn from(e: std::fmt::Error) -> Self {
        CawError::Io(e.to_string())
    }
}

/// Returns `true` if `text` looks like degenerate looping output.
///
/// Two signals, either sufficient:
/// - One word accounts for >70% of all whitespace-separated tokens (catches
///   "wordwordword" or "word word word word").
/// - Unique trigrams are <10% of total trigrams with ≥30 words (catches
///   multi-word loops like "the cat sat the cat sat…").
///
/// Responses shorter than 20 words are never flagged — structured one-liners
/// and short factual answers would produce false positives.

/// Truncate a model response at the first chat-template boundary token.
///
/// Chat models sometimes fail to stop at the configured sentinel and begin
/// generating the next conversation turn. Everything from `<|im_start|>`
/// onward is not part of the model's answer and should be discarded rather
/// than treating the entire response as degenerate.
pub fn truncate_at_chat_boundary(text: &str) -> &str {
    if let Some(pos) = text.find("<|im_start|>") {
        return text[..pos].trim_end();
    }
    // A bare trailing <|im_end|> (nothing after it) is the normal Qwen3/ChatML stop
    // token — not runaway generation. Strip it unconditionally. The <|im_start|> check
    // above already handles the case where the model is generating additional turns.
    if let Some(pos) = text.find("<|im_end|>") {
        return text[..pos].trim_end();
    }
    text
}

/// Strip `[recalled from path]\n...\n[end recall]` blocks from text.
///
/// Models that have seen the provenance injection format sometimes generate
/// fake recall blocks verbatim. These blocks contain repeated code and corrupt
/// degeneracy detection: `detect_loop` fires on the repetitive code content
/// rather than on the actual answer prose. Stripping them before detection
/// avoids false positives from B6-style fake-marker output.
pub fn strip_fake_recall_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("[recalled from ") {
        result.push_str(&remaining[..start]);
        let after_start = &remaining[start..];
        if let Some(end_rel) = after_start.find("[end recall]") {
            let end = start + end_rel + "[end recall]".len();
            remaining = &remaining[end..];
        } else {
            // No closing marker — keep everything from start onward to avoid
            // silently truncating a response that just happened to contain the phrase.
            result.push_str(&remaining[start..]);
            return result;
        }
    }
    result.push_str(remaining);
    result
}

/// Inspect `text` for degenerate looping patterns and return a human-readable
/// description of the first pattern that fires, or `None` if the text looks
/// clean. This companion to `is_looping` is intended for logging: callers
/// that want to know *why* a response was flagged should call this and emit
/// the reason at `warn!` level so subsequent QA runs can diagnose false
/// positives.
pub fn detect_loop(text: &str) -> Option<String> {
    if text.contains("<|im_end|>") {
        return Some("chat_template_token: <|im_end|>".into());
    }
    if text.contains("<|im_start|>") {
        return Some("chat_template_token: <|im_start|>".into());
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 20 {
        return None;
    }

    // Single-word dominance: one word accounts for >70% of tokens.
    let (modal_word, modal_count) = {
        let mut counts = HashMap::new();
        for &w in &words {
            *counts.entry(w).or_insert(0usize) += 1;
        }
        counts.into_iter().max_by_key(|&(_, c)| c).unwrap_or(("", 0))
    };
    let pct = modal_count * 100 / words.len();
    if pct > 70 {
        return Some(format!(
            "word_dominance: '{modal_word}' at {pct}% of {} tokens",
            words.len()
        ));
    }

    // Trigram diversity collapse: unique trigrams < 10% of total with ≥30 words.
    if words.len() >= 30 {
        let total = words.len() - 2;
        let unique: HashSet<[&str; 3]> =
            words.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
        if unique.len() * 10 < total {
            let unique_pct = unique.len() * 100 / total;
            return Some(format!(
                "trigram_collapse: {}/{} unique trigrams ({unique_pct}%)",
                unique.len(),
                total
            ));
        }
    }

    // Sentence-level repetition: any substantial line (>30 chars) appearing ≥3 times.
    let long_lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| l.len() > 30)
        .collect();
    if long_lines.len() >= 6 {
        let mut counts = HashMap::new();
        for &line in &long_lines {
            *counts.entry(line).or_insert(0usize) += 1;
        }
        if let Some((&repeated_line, &count)) = counts.iter().find(|&(_, &c)| c >= 3) {
            let preview: String = repeated_line.chars().take(60).collect();
            let byte_pos = text.find(repeated_line).unwrap_or(0);
            return Some(format!(
                "line_repetition: '{preview}...' ×{count} (first at byte {byte_pos})"
            ));
        }
    }

    None
}

pub fn is_looping(text: &str) -> bool {
    detect_loop(text).is_some()
}

/// `(path, mtime_unix_secs)` pair identifying a source file by name and
/// modification time. Used to skip re-embedding unchanged files at startup.
pub type DocumentId = (String, u64);

/// Set of `DocumentId` pairs representing files already present in the index.
pub type DocumentIdSet = std::collections::HashSet<DocumentId>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StubId(pub String);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContentKind {
    Markdown,
    Code,
    PlainText,
    Tabular,
    Transcript,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Locator {
    pub source: String,
    pub locator: String,
}

impl Locator {
    pub fn full(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            locator: "full".to_string(),
        }
    }

    pub fn line_range(source: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            source: source.into(),
            locator: format!("{}-{}", start, end),
        }
    }

    pub fn heading(source: impl Into<String>, heading_path: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            locator: format!("#{}", heading_path.into()),
        }
    }
}

/// Range selection within content — addresses lines, headings, token windows,
/// or the full document. Used throughout the recall pipeline.
#[derive(Debug, Clone)]
pub enum Range {
    Full,
    Lines { start: usize, end: usize },
    Heading { path: Vec<String> },
    Tokens { start: usize, count: usize },
    Custom(String),
}

impl Range {
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.eq_ignore_ascii_case("full") || s.is_empty() {
            return Self::Full;
        }

        // Heading path: #Section/Subsection
        if let Some(heading) = s.strip_prefix('#') {
            return Self::Heading {
                path: heading.split('/').map(|p| p.trim().to_string()).collect(),
            };
        }

        // Token range: T100:500
        if let Some(value) = s.strip_prefix('T')
            .and_then(|rest| {
                rest.split_once(':')
                    .and_then(|(start, count)| {
                        let start = start.parse().ok()?;
                        let count = count.parse().ok()?;
                        Some(Range::Tokens { start, count })
                    })
            }) {
            return value;
        }

        // Line range with L prefix: L5-L15
        if s.starts_with('L') {
            let stripped = s.replace('L', "");
            if let Some((start, end)) = stripped.split_once('-')
                && let (Ok(s), Ok(e)) = (start.parse(), end.parse())
            {
                return Self::Lines { start: s, end: e };
            }
        }

        // Plain line range: 10-20
        if let Some((start, end)) = s.split_once('-')
            && let (Ok(s), Ok(e)) = (start.parse(), end.parse())
        {
            return Self::Lines { start: s, end: e };
        }

        Self::Custom(s.to_string())
    }

    pub fn to_locator_string(&self) -> String {
        match self {
            Self::Full => "full".to_string(),
            Self::Lines { start, end } => format!("{}-{}", start, end),
            Self::Heading { path } => format!("#{}", path.join("/")),
            Self::Tokens { start, count } => format!("T{}:{}", start, count),
            Self::Custom(s) => s.clone(),
        }
    }

    /// Apply this range to extract a portion of content
    pub fn apply(&self, content: &str) -> String {
        match self {
            Self::Full => content.to_string(),

            Self::Lines { start, end } => {
                let lines: Vec<&str> = content.lines().collect();
                let start_idx = start.saturating_sub(1);
                let end_idx = (*end).min(lines.len());
                if start_idx >= lines.len() {
                    return String::new();
                }
                lines[start_idx..end_idx].join("\n")
            }

            Self::Heading { path } => extract_heading_section(content, path),

            Self::Tokens { start, count } => extract_token_range(content, *start, *count),

            Self::Custom(_) => content.to_string(),
        }
    }
}

/// Extract content under a heading path using level-based matching.
/// Finds the heading matching the last element of the path at the
/// appropriate depth, then captures everything until the next heading
/// at the same or higher level.
fn extract_heading_section(content: &str, path: &[String]) -> String {
    if path.is_empty() {
        return content.to_string();
    }

    let target = &path[path.len() - 1];
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::new();
    let mut in_section = false;
    let mut section_level = 0;

    for line in &lines {
        let trimmed = line.trim_start();
        if let Some(after_hashes) = trimmed.strip_prefix('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            let heading_text = after_hashes.trim_start_matches('#').trim();

            if !in_section && heading_text == target {
                in_section = true;
                section_level = level;
                result.push(*line);
                continue;
            }

            if in_section && level <= section_level {
                break;
            }
        }

        if in_section {
            result.push(*line);
        }
    }

    result.join("\n")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stub {
    pub id: StubId,
    pub path: String,
    pub token_estimate: usize,
    pub kind: ContentKind,
    pub summary: String,
    pub outline: Vec<String>,
    pub content_hash: String,
    pub mtime_unix_secs: u64,
    /// Byte offset of this stub's body in the source file at `path`. For
    /// chunked files this is the chunk's body start (overlap is not counted);
    /// for single-stub files this is 0.
    #[serde(default)]
    pub byte_offset: u64,
    /// Length in bytes of this stub's body in the source file. With
    /// `byte_offset` this defines a half-open range `[offset, offset+length)`
    /// that can be sliced out of the file without reading the whole thing.
    /// Chunks tile the file — offsets are contiguous, nothing is duplicated.
    #[serde(default)]
    pub byte_length: u64,
    #[serde(default)]
    pub consolidation_notes: Vec<ConsolidationNote>,
}

#[derive(Debug, Clone)]
pub struct ScoredStub {
    pub stub: Stub,
    pub score: f32,
}

/// Build a synthetic fragment listing candidate files for the model to choose from.
/// Used when the ambiguity gate fires: rather than loading content or loading nothing,
/// surface the file list so the model can reason about what it needs. Scores are omitted
/// intentionally — they're an internal signal, not useful guidance for the model.
pub fn candidate_list_fragment(hits: &[ScoredStub], threshold: f32) -> RecallFragment {
    // Hits are sorted by score descending; keep only the first (best) chunk per file.
    let mut seen = std::collections::HashSet::new();
    let lines: Vec<String> = hits
        .iter()
        .filter(|h| h.score >= threshold && seen.insert(h.stub.path.clone()))
        .map(|h| {
            if h.stub.summary.is_empty() {
                format!("- {}", h.stub.path)
            } else {
                format!("- {} — {}", h.stub.path, h.stub.summary)
            }
        })
        .collect();

    let content = format!(
        "These files match your query but have not been loaded. \
         Treat this list as discovery metadata, not evidence. \
         Mention the specific files you need if you want them loaded, and do not \
         claim details that are not written explicitly below:\n\n{}",
        lines.join("\n")
    );

    RecallFragment {
        stub_id: StubId("__candidates__".to_string()),
        content: content.clone(),
        locator: Locator {
            source: "search-candidates".to_string(),
            locator: "file-list".to_string(),
        },
        tokens: count_tokens_cl100k(&content),
        mtime_unix_secs: 0,
    }
}

#[derive(Debug, Clone)]
pub struct RecallFragment {
    pub stub_id: StubId,
    pub content: String,
    pub locator: Locator,
    pub tokens: usize,
    pub mtime_unix_secs: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    pub max_total: usize,
    pub reserved_for_prompt: usize,
    pub reserved_for_answer: usize,
}

impl TokenBudget {
    pub fn available_for_workspace(self) -> usize {
        self.max_total
            .saturating_sub(self.reserved_for_prompt + self.reserved_for_answer)
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerInput {
    pub currently_loaded: Vec<RecallFragment>,
    pub candidates: Vec<RecallFragment>,
    pub budget: TokenBudget,
    pub candidate_scores: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SchedulerDecision {
    pub keep: Vec<RecallFragment>,
    pub evicted: Vec<RecallFragment>,
    pub admitted: Vec<RecallFragment>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub supports_tool_calls: bool,
    pub supports_hidden_reasoning: bool,
    pub supports_visible_reasoning: bool,
    /// Adapter owns its sampling loop and can inject content mid-generation
    /// every N tokens without restarting. When true, the orchestrator calls
    /// `generate_passive` instead of `complete`.
    pub supports_passive_injection: bool,
}

/// Signals that drive what to proactively load into the context workspace.
/// This is the actionable output of intent classification — the orchestrator
/// uses these to decide what to fetch before invoking the answer model.
/// Guidance-only signals (how to phrase the answer) live separately in QueryIntent.
#[derive(Debug, Clone, Default)]
pub struct AugmentationSignals {
    pub is_inventory_request: bool,
    pub is_results_request: bool,
    pub is_status_request: bool,
    pub is_next_step_request: bool,
    pub wants_latest_run_only: bool,
    /// When true the query asks for an overview, explanation, or description — weight
    /// documentation sources (`.md` files) above implementation files during retrieval.
    pub wants_explanation: bool,
    /// Model-defined augmentation hints from extra fields (e.g. "needs_git_status").
    pub extra_hints: Vec<String>,
}

impl AugmentationSignals {
    pub fn is_empty(&self) -> bool {
        !self.is_inventory_request
            && !self.is_results_request
            && !self.is_status_request
            && !self.is_next_step_request
            && !self.wants_latest_run_only
            && self.extra_hints.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct QueryIntent {
    pub is_inventory_request: bool,
    pub is_results_request: bool,
    pub is_status_request: bool,
    pub is_next_step_request: bool,
    pub wants_exact_names_or_paths: bool,
    pub wants_numeric_values: bool,
    pub wants_latest_run_only: bool,
    pub wants_comparison: bool,
    pub wants_explanation: bool,
    pub wants_completion_state: bool,
    pub wants_recommended_actions: bool,
    pub needs_grounded_evidence_only: bool,
    pub abstain: bool,
    pub confidence: Option<f32>,
    /// Arbitrary extra signals emitted by the classifier beyond the fixed schema.
    /// Bool-true values and non-empty strings are forwarded as guidance to the answer model.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Display for QueryIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if self.is_inventory_request {
            parts.push("inventory");
        }
        if self.is_results_request {
            parts.push("results");
        }
        if self.is_status_request {
            parts.push("status");
        }
        if self.is_next_step_request {
            parts.push("next_step");
        }
        if self.wants_exact_names_or_paths {
            parts.push("exact_names_or_paths");
        }
        if self.wants_numeric_values {
            parts.push("numeric_values");
        }
        if self.wants_latest_run_only {
            parts.push("latest_run_only");
        }
        if self.wants_comparison {
            parts.push("comparison");
        }
        if self.wants_explanation {
            parts.push("explanation");
        }
        if self.wants_completion_state {
            parts.push("completion_state");
        }
        if self.wants_recommended_actions {
            parts.push("recommended_actions");
        }
        if self.needs_grounded_evidence_only {
            parts.push("grounded_evidence_only");
        }
        if self.abstain {
            parts.push("abstain");
        }
        write!(f, "QueryIntent({})", parts.join(", "))
    }
}

impl QueryIntent {
    /// Extract the context workspace augmentation signals. These drive what to
    /// proactively load before answering — independent of how the answer is phrased.
    pub fn augmentation_signals(&self) -> AugmentationSignals {
        AugmentationSignals {
            is_inventory_request: self.is_inventory_request,
            is_results_request: self.is_results_request,
            is_status_request: self.is_status_request,
            is_next_step_request: self.is_next_step_request,
            wants_latest_run_only: self.wants_latest_run_only,
            wants_explanation: self.wants_explanation,
            extra_hints: self
                .extra
                .iter()
                .filter_map(|(k, v)| {
                    matches!(v, serde_json::Value::Bool(true)).then(|| k.clone())
                })
                .collect(),
        }
    }

    /// Simplified system prompt that only asks for context augmentation signals.
    /// 5 fields instead of 13 — better suited for small models because the task
    /// maps directly to "what should I fetch" rather than mixing retrieval signals
    /// with answer-formatting hints.
    pub fn augmentation_system_prompt() -> &'static str {
        concat!(
            "You are a context-augmentation classifier. ",
            "Given a user query, identify what types of information should be loaded into the context workspace before answering. ",
            "Return exactly one JSON object — no prose, no markdown fences, no explanation.\n\n",
            "- is_inventory_request: the query asks WHAT EXISTS — artifacts, files, a list of runs, or a COUNT of items. ",
            "Examples: 'what benchmarks have been run?', 'how many todos are left?', 'how many open bugs?', 'total remaining items?'\n",
            "- is_results_request: the query asks for METRIC VALUES or data from something that ran — numbers, scores, timings. Example: 'what throughput numbers were reported?'\n",
            "- is_status_request: the query asks about PROGRESS or COMPLETION STATE — what is done, pending, or in-flight. ",
            "Examples: 'is benchmarking finished?', 'how many items remain?', 'what percentage is complete?', 'how much is done?'\n",
            "- is_next_step_request: the query asks what to DO NEXT. Example: 'what should I work on?'\n",
            "- wants_explanation: the query asks WHY or HOW something works, or requests a description/overview of a concept or system. Examples: 'what is X?', 'how does Y work?', 'why should I use Z?'\n",
            "- wants_latest_run_only: the query is scoped to the most recent run or timestamp.\n\n",
            "Schema: {\"is_inventory_request\":bool,\"is_results_request\":bool,\"is_status_request\":bool,\"is_next_step_request\":bool,\"wants_explanation\":bool,\"wants_latest_run_only\":bool}"
        )
    }

    pub fn augmentation_user_prompt(query: &str) -> String {
        format!(
            "Classify the user's query for context augmentation.\n\n\
             Note: count queries ('how many X?', 'total remaining', 'how much is left?') \
             should set is_inventory_request=true and/or is_status_request=true.\n\n\
             User query:\n{query}\n"
        )
    }

    /// Full 13-field system prompt covering both augmentation and guidance signals.
    /// Used for benchmarking alignment/conformity. Production use should prefer
    /// augmentation_system_prompt() for small models.
    pub fn classifier_system_prompt() -> &'static str {
        concat!(
            "You are a query-intent classifier for retrieval planning. ",
            "Return exactly one JSON object — no prose, no markdown fences, no explanation. ",
            "\n\nField definitions:\n",
            "- is_inventory_request: the query asks WHAT EXISTS or HOW MANY EXIST — names, paths, files, a list of artifacts, or a count of items. Examples: 'what benchmarks have been run?', 'how many todos are left?', 'how many open bugs?'\n",
            "- is_results_request: the query asks for METRIC VALUES or data from something that ran — numbers, scores, timings. Example: 'what throughput numbers were reported?'\n",
            "- is_status_request: the query asks about PROGRESS or COMPLETION STATE of work — what is done, pending, or in-flight. Examples: 'is benchmarking finished?', 'how many items remain?', 'how much is done?'\n",
            "- is_next_step_request: the query asks what to DO NEXT. Example: 'what should I work on?'\n",
            "- wants_exact_names_or_paths: the answer requires precise artifact names, file paths, or model identifiers (not paraphrases).\n",
            "- wants_numeric_values: the answer requires exact numbers with units.\n",
            "- wants_latest_run_only: the query is scoped to the most recent run or timestamp.\n",
            "- wants_comparison: the query compares two or more things side by side.\n",
            "- wants_explanation: the query asks WHY or HOW, not just WHAT.\n",
            "- wants_completion_state: the answer must state explicitly whether something is done or still pending.\n",
            "- wants_recommended_actions: the answer should include concrete next actions.\n",
            "- needs_grounded_evidence_only: set true unless the query explicitly invites open-ended opinions or speculation; default to true for factual, status, inventory, and results queries.\n",
            "- abstain: the query is genuinely ambiguous and cannot be routed.\n",
            "- confidence: float in [0,1].\n",
            "\nSchema: {\"is_inventory_request\":bool,\"is_results_request\":bool,\"is_status_request\":bool,\"is_next_step_request\":bool,",
            "\"wants_exact_names_or_paths\":bool,\"wants_numeric_values\":bool,",
            "\"wants_latest_run_only\":bool,\"wants_comparison\":bool,",
            "\"wants_explanation\":bool,\"wants_completion_state\":bool,\"wants_recommended_actions\":bool,\"needs_grounded_evidence_only\":bool,",
            "\"abstain\":bool,\"confidence\":number}"
        )
    }

    pub fn classifier_user_prompt(query: &str) -> String {
        format!("Classify the user's query for retrieval planning.\n\nUser query:\n{query}\n")
    }

    /// Parse a classifier response. Returns the intent and the set of top-level keys
    /// the model actually emitted, so callers can distinguish "absent" from "false".
    pub fn from_classifier_response(raw: &str) -> CawResult<(Self, HashSet<String>)> {
        let json = extract_json_object(raw).ok_or_else(|| {
            CawError::InvalidInput("classifier did not return a JSON object".into())
        })?;
        // Parse via Value first so duplicate keys are silently collapsed to the last value.
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| CawError::InvalidInput(format!("invalid classifier JSON: {e}")))?;
        let emitted_keys: HashSet<String> = value
            .as_object()
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();
        let mut parsed: Self = serde_json::from_value(value)
            .map_err(|e| CawError::InvalidInput(format!("invalid classifier JSON: {e}")))?;
        if let Some(c) = parsed.confidence {
            parsed.confidence = Some(if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 });
        }
        Ok((parsed, emitted_keys))
    }

    pub fn guidance_lines(&self) -> Vec<String> {
        if self.abstain {
            return Vec::new();
        }

        let mut lines = Vec::new();
        if self.is_inventory_request {
            lines.push(
                "For inventory-style questions, list exact names or paths present in recalled evidence before summarizing.".to_string(),
            );
        }
        if self.is_results_request {
            lines.push(
                "For result-oriented questions, report only metrics and values explicitly present in recalled evidence.".to_string(),
            );
        }
        if self.is_status_request {
            lines.push(
                "For status questions, distinguish clearly between completed, pending, and unknown work based on recalled evidence.".to_string(),
            );
        }
        if self.is_next_step_request {
            lines.push(
                "For next-step questions, recommend concrete next actions grounded in recalled evidence rather than generic advice.".to_string(),
            );
        }
        if self.wants_exact_names_or_paths {
            lines.push(
                "Prefer precise artifact names, file names, and paths over paraphrases."
                    .to_string(),
            );
        }
        if self.wants_numeric_values {
            lines.push(
                "Prefer exact numeric values and units when they are available in recalled evidence.".to_string(),
            );
        }
        if self.wants_latest_run_only {
            lines.push(
                "If multiple runs are present, prioritize the newest clearly identified run or timestamped artifact.".to_string(),
            );
        }
        if self.wants_comparison {
            lines.push(
                "Keep compared artifacts separate and label each metric or claim with the corresponding artifact.".to_string(),
            );
        }
        if self.wants_explanation {
            lines.push(
                "Separate direct evidence from inference when explaining causes or tradeoffs."
                    .to_string(),
            );
        }
        if self.wants_completion_state {
            lines.push(
                "State explicitly whether each relevant task or artifact is completed, still pending, or not evidenced; if completion cannot be determined, say what evidence is needed to decide it.".to_string(),
            );
        }
        if self.wants_recommended_actions {
            lines.push(
                "When recommending next actions, tie each action to recalled evidence; if evidence is insufficient, say what planning or status evidence is needed before recommending a next step.".to_string(),
            );
        }
        if self.needs_grounded_evidence_only {
            lines.push(
                "If recalled context lacks direct evidence, do not guess; state what evidence, artifact, or file is needed to fulfill the request.".to_string(),
            );
        }
        for (key, val) in &self.extra {
            match val {
                serde_json::Value::Bool(true) => {
                    lines.push(format!("Additional context: {}.", key.replace('_', " ")));
                }
                serde_json::Value::String(s) if !s.is_empty() => {
                    lines.push(s.clone());
                }
                _ => {}
            }
        }
        lines
    }

    pub fn is_actionable(&self, min_confidence: f32) -> bool {
        !self.abstain && self.confidence.map_or(true, |c| c >= min_confidence)
    }

    /// Returns true if any retrieval-relevant intent flag is set, meaning the
    /// query carries a clear signal worth running a retrieval cycle for.
    /// All-false + short message = casual acknowledgment, not a query.
    pub fn is_substantive(&self) -> bool {
        self.is_inventory_request
            || self.is_results_request
            || self.is_status_request
            || self.is_next_step_request
            || self.wants_exact_names_or_paths
            || self.wants_numeric_values
            || self.wants_latest_run_only
            || self.wants_comparison
            || self.wants_explanation
            || self.wants_completion_state
            || self.wants_recommended_actions
            || self.needs_grounded_evidence_only
    }

    /// Merge multiple classifier votes by strict majority (> half must agree for a field
    /// to be set). Returns the default intent if `votes` is empty.
    pub fn majority_vote(votes: &[QueryIntent]) -> QueryIntent {
        if votes.is_empty() {
            return QueryIntent::default();
        }
        if votes.len() == 1 {
            return votes[0].clone();
        }

        let threshold = votes.len() / 2 + 1;
        let count = |f: fn(&QueryIntent) -> bool| votes.iter().filter(|v| f(v)).count() >= threshold;

        let confidence = {
            let vals: Vec<f32> = votes.iter().filter_map(|v| v.confidence).collect();
            if vals.is_empty() { None } else { Some(vals.iter().sum::<f32>() / vals.len() as f32) }
        };

        // Include extra bool-true keys that the majority agree on.
        let mut extra_counts: HashMap<String, usize> = HashMap::new();
        for vote in votes {
            for (k, v) in &vote.extra {
                if matches!(v, serde_json::Value::Bool(true)) {
                    *extra_counts.entry(k.clone()).or_insert(0) += 1;
                }
            }
        }
        let extra = extra_counts
            .into_iter()
            .filter(|(_, n)| *n >= threshold)
            .map(|(k, _)| (k, serde_json::Value::Bool(true)))
            .collect();

        QueryIntent {
            is_inventory_request:      count(|v| v.is_inventory_request),
            is_results_request:        count(|v| v.is_results_request),
            is_status_request:         count(|v| v.is_status_request),
            is_next_step_request:      count(|v| v.is_next_step_request),
            wants_exact_names_or_paths: count(|v| v.wants_exact_names_or_paths),
            wants_numeric_values:      count(|v| v.wants_numeric_values),
            wants_latest_run_only:     count(|v| v.wants_latest_run_only),
            wants_comparison:          count(|v| v.wants_comparison),
            wants_explanation:         count(|v| v.wants_explanation),
            wants_completion_state:    count(|v| v.wants_completion_state),
            wants_recommended_actions: count(|v| v.wants_recommended_actions),
            needs_grounded_evidence_only: count(|v| v.needs_grounded_evidence_only),
            abstain:                   count(|v| v.abstain),
            confidence,
            extra,
        }
    }
}

pub trait IntentClassifier {
    fn classify(&self, query: &str) -> CawResult<QueryIntent>;
}

#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    pub system: String,
    pub user: String,
    pub workspace_fragments: Vec<RecallFragment>,
    pub workspace_guidance: Vec<String>,
}

/// Actual token usage reported by the model API. Preferred over estimates
/// whenever the adapter can provide it.
#[derive(Debug, Clone, Copy)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub answer: String,
    /// Visible reasoning emitted by the model before the answer, if any.
    /// Populated by adapters that detect `<think>...</think>` blocks.
    /// Useful for thinking-trace recall without the orchestrator having to
    /// re-parse the raw output.
    pub thinking: Option<String>,
    /// Actual token counts from the API response. `None` for adapters that
    /// don't report usage (e.g. ClaudeCode, Mock).
    pub usage: Option<TokenUsage>,
}

/// Split a raw model response into (thinking, answer) by extracting the first
/// `<think>...</think>` block. Returns `(None, raw)` if no block is found.
///
/// If `<think>` is present but `</think>` is absent (context window exceeded
/// mid-reasoning), the partial thinking trace is returned and the answer is
/// whatever preceded `<think>` (usually empty). This prevents storing raw
/// `<think>…` text as the visible answer, which would corrupt session history.
pub fn split_thinking(raw: &str) -> (Option<String>, String) {
    let open = raw.find("<think>");
    let close = raw.find("</think>");
    match (open, close) {
        (Some(o), Some(c)) if c > o => {
            let thinking = raw[o + "<think>".len()..c].trim().to_string();
            let answer = raw[c + "</think>".len()..].trim().to_string();
            // An empty think block is a model formatting artifact (e.g. Qwen3
            // emitting <think>\n\n</think> before the real answer). Treat it as
            // a no-op so callers see a plain response without a thinking trace.
            if thinking.is_empty() {
                (None, answer)
            } else {
                (Some(thinking), answer)
            }
        }
        (Some(o), _) => {
            let thinking = raw[o + "<think>".len()..].trim().to_string();
            let answer = raw[..o].trim().to_string();
            (Some(thinking), answer)
        }
        _ => (None, raw.to_string()),
    }
}

/// Format for inline provenance tags on recalled content.
#[derive(Debug, Clone, Copy)]
pub enum ProvenanceFormat {
    /// XML tags: `<recalled from="path">content</recalled>`
    /// Preferred by Anthropic models which handle XML natively.
    Xml,
    /// Bracket tags: `[recalled from path]\ncontent\n[end recall]`
    /// Preferred by OpenAI-protocol models.
    Bracketed,
}

impl CompletionRequest {
    /// Format workspace fragments with inline provenance for model consumption.
    /// Each fragment is wrapped with source attribution so the model treats
    /// recalled content as quoted material, not its own knowledge.
    pub fn format_workspace(&self, format: ProvenanceFormat) -> String {
        self.format_workspace_with_guidance(format, &[])
    }

    /// Format workspace fragments with optional extra guidance supplied by an
    /// upstream intent classifier or caller-owned policy.
    pub fn format_workspace_with_guidance(
        &self,
        format: ProvenanceFormat,
        extra_guidance: &[&str],
    ) -> String {
        if self.workspace_fragments.is_empty() {
            return String::new();
        }

        // Render every admitted fragment verbatim. Per-source limiting is the
        // producer's responsibility (the orchestrator's admission cap and the
        // proxy's clamp both bound chunks-per-source); the renderer no longer
        // collapses multiple chunks of one file, because different chunks of a
        // multi-chunk document answer different questions — collapsing to the
        // top-scoring chunk silently drops the answer-bearing one.
        let fragments: Vec<String> = self
            .workspace_fragments
            .iter()
            .map(|f| {
                // "full" is the default and adds no information; omit it.
                let source_ref = if f.locator.locator == "full" {
                    f.locator.source.clone()
                } else {
                    format!("{}:{}", f.locator.source, f.locator.locator)
                };
                match format {
                    ProvenanceFormat::Xml => format!(
                        "<recalled from=\"{source_ref}\">\n\
                         {content}\n\
                         </recalled>",
                        content = f.content,
                    ),
                    ProvenanceFormat::Bracketed => format!(
                        "[recalled from {source_ref}]\n\
                         {content}\n\
                         [end recall]",
                        content = f.content,
                    ),
                }
            })
            .collect();

        let inline_guidance: Vec<&str> =
            self.workspace_guidance.iter().map(String::as_str).collect();
        let guidance =
            workspace_guidance(&self.workspace_fragments, &inline_guidance, extra_guidance);

        format!(
            "\n\nRecalled workspace context (each block is verbatim from the cited source — \
             treat as quoted material, not your own knowledge):\n{}\n\n{}",
            guidance,
            fragments.join("\n\n")
        )
    }
}

fn workspace_guidance(
    fragments: &[RecallFragment],
    inline_guidance: &[&str],
    extra_guidance: &[&str],
) -> String {
    let has_candidate_list = fragments.iter().any(|fragment| {
        fragment.locator.source == "search-candidates" || fragment.stub_id.0 == "__candidates__"
    });

    let mut lines = Vec::new();

    // Baseline grounding preference, present whenever a workspace is rendered.
    // Stated as a preference, not exclusivity: when the recalled context is thin
    // or off-topic the model may still draw on its own knowledge, but it must not
    // invent specifics the context doesn't support. Addresses the observed
    // failure where a vague query retrieves the right files yet the model answers
    // from generic priors and confabulates details around them.
    lines.push(
        "- Prefer the recalled context when it bears on the question, and cite the source file. If the context does not cover the question, say so plainly rather than inventing specifics.".to_string(),
    );

    if has_candidate_list {
        lines.push(
            "- Blocks from `search-candidates:file-list` are candidate metadata only. Use them to decide what to load, not as evidence for factual claims.".to_string(),
        );
    }

    for guidance in inline_guidance.iter().chain(extra_guidance.iter()) {
        let trimmed = guidance.trim();
        if !trimmed.is_empty() && !lines.iter().any(|line| line == &format!("- {trimmed}")) {
            lines.push(format!("- {trimmed}"));
        }
    }

    if lines.is_empty() {
        String::new()
    } else {
        format!("\nAnswering hints:\n{}", lines.join("\n"))
    }
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in raw.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if start.is_none() {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    let begin = start?;
                    return Some(&raw[begin..=idx]);
                }
            }
            _ => {}
        }
    }

    None
}

/// Hysteresis thresholds for recall loading/unloading.
/// Single source of truth — all orchestrators and schedulers reference this.
#[derive(Debug, Clone, Copy)]
pub struct RecallThresholds {
    pub load: f32,
    pub unload: f32,
}

impl RecallThresholds {
    pub fn default_hysteresis() -> Self {
        Self {
            load: 0.7,
            unload: 0.55,
        }
    }

    pub fn permissive() -> Self {
        Self {
            load: 0.3,
            unload: 0.2,
        }
    }
}

impl Default for RecallThresholds {
    fn default() -> Self {
        Self::default_hysteresis()
    }
}

/// A note captured during a fragment's time in the workspace.
/// Attached to stubs so future recalls benefit from prior visits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationNote {
    pub content: String,
    pub source: ConsolidationSource,
    pub created_at_secs: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsolidationSource {
    /// Generated when the fragment was evicted from the workspace
    Eviction,
    /// Extracted from model output during an active session
    ModelAnnotation,
}

/// Detected topic overlap between recalled fragments from different sources.
/// High overlap from different files signals potential contradiction worth
/// surfacing for reconciliation.
#[derive(Debug, Clone)]
pub struct TopicOverlap {
    pub stub_a: StubId,
    pub path_a: String,
    pub stub_b: StubId,
    pub path_b: String,
    pub shared_terms: Vec<String>,
    pub overlap_score: f32,
}

/// An annotation emitted by the model about a specific recalled stub
#[derive(Debug, Clone)]
pub struct ModelAnnotation {
    pub stub_id: String,
    pub content: String,
    pub position: usize,
}

pub trait Retriever {
    fn search(&mut self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>>;
    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment>;
    /// Insert a new document. Default no-op for read-only retrievers.
    fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        let _ = (stub, content);
        Ok(())
    }
}

pub trait BudgetScheduler {
    fn schedule(&self, input: SchedulerInput) -> SchedulerDecision;
}

pub trait ProvenanceStore {
    fn record(&mut self, fragment: RecallFragment);
    fn all(&self) -> Vec<RecallFragment>;

    /// Record a fragment with the query context that triggered its recall.
    fn record_with_context(&mut self, fragment: RecallFragment, _query: &str, _turn: usize) {
        self.record(fragment);
    }

    /// Store a consolidation note for a stub (generated on eviction or from model annotations).
    fn record_consolidation(&mut self, _stub_id: StubId, _note: ConsolidationNote) {}

    /// Retrieve consolidation notes for a specific stub.
    fn consolidation_notes_for(&self, _stub_id: &StubId) -> Vec<ConsolidationNote> {
        Vec::new()
    }

    /// Check for topic overlaps between recalled fragments from different sources.
    fn check_topic_overlaps(&self) -> Vec<TopicOverlap> {
        Vec::new()
    }

    /// Format overlap warnings for injection into the next completion.
    fn format_overlap_warnings(&self) -> String {
        String::new()
    }
}

pub trait ModelAdapter {
    fn model_name(&self) -> &str;
    fn capabilities(&self) -> ModelCapabilities;
    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse>;

    /// Drive generation token-by-token with a sliding recall window.
    ///
    /// Every `check_interval` generated tokens, `on_window` is called with the
    /// last `window_size` tokens of generated text (a sliding window, not just
    /// the interval slice). This lets the embedding query span token sequences
    /// that cross interval boundaries.
    ///
    /// If `on_window` returns `Some(content)`, the adapter injects those tokens
    /// directly into the in-flight KV cache before sampling the next token —
    /// no generation restart, no re-encoding of prior output.
    ///
    /// Default: runs `complete` with no injection. Only adapters that own their
    /// sampling loop (e.g. `LlamaCppAdapter`) override this.
    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        let _ = (check_interval, window_size, on_window);
        self.complete(req)
    }

    /// Stream the model's thinking trace, calling `on_step` at each reasoning
    /// step boundary (`\n\n`). Returning `false` from `on_step` stops replay
    /// early so the orchestrator can inject context and restart.
    ///
    /// Default: runs a full completion, splits the thinking block, and replays
    /// each paragraph through `on_step`. Streaming adapters override this to
    /// stop the HTTP response at `</think>` without paying for answer tokens.
    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<CompletionResponse> {
        let response = self.complete(req)?;
        let thinking = response.clone().thinking.unwrap_or_default();
        for step in thinking
            .split("\n\n")
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if !on_step(step)? {
                break;
            }
        }
        Ok(response)
    }

    /// The provenance wrapping format this adapter uses when it renders recalled
    /// workspace fragments into the prompt (via `CompletionRequest::format_workspace`).
    /// Prompt-dump tooling queries this to reproduce exactly the context string
    /// the model receives. Defaults to `Bracketed` (the OpenAI-protocol format);
    /// Anthropic-family adapters override to `Xml`.
    fn provenance_format(&self) -> ProvenanceFormat {
        ProvenanceFormat::Bracketed
    }
}

/// Forward `ModelAdapter` through a boxed trait object so callers that build
/// adapters dynamically (e.g. CLI-driven harnesses choosing between Ollama,
/// vLLM, ClaudeCode, etc.) can hand the box directly to generic consumers
/// like `DynamicRecallOrchestrator<_, _, _, _, M, _>`.
impl<T: ModelAdapter + ?Sized> ModelAdapter for Box<T> {
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        (**self).capabilities()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        (**self).complete(req)
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        (**self).generate_passive(req, check_interval, window_size, on_window)
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<CompletionResponse> {
        (**self).thinking_with_steps(req, on_step)
    }

    fn provenance_format(&self) -> ProvenanceFormat {
        (**self).provenance_format()
    }
}

/// Forward `ModelAdapter` through an `Arc` so a single loaded adapter can be
/// shared between the orchestrator and the intent classifier without loading
/// the model twice.
impl<T: ModelAdapter + ?Sized> ModelAdapter for std::sync::Arc<T> {
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        (**self).capabilities()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        (**self).complete(req)
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        (**self).generate_passive(req, check_interval, window_size, on_window)
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<CompletionResponse> {
        (**self).thinking_with_steps(req, on_step)
    }

    fn provenance_format(&self) -> ProvenanceFormat {
        (**self).provenance_format()
    }
}

/// Embedding generation trait - abstracts different embedding backends.
///
/// Asymmetric models (BGE, E5) produce better results when queries and documents
/// are encoded differently. Override `embed_query`/`embed_document` for these;
/// symmetric models get correct behavior from the default delegation to `embed`.
pub trait EmbeddingProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>>;

    /// Embed texts that will be used as search queries.
    /// Asymmetric models should override to add the appropriate prefix.
    fn embed_query(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    /// Embed texts that will be indexed as documents.
    /// Asymmetric models should override to add the appropriate prefix.
    fn embed_document(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn dimension(&self) -> usize;
    fn provider_name(&self) -> &str;

    /// Optional per-run histogram of padded sequence lengths observed during
    /// embedding. Each entry is `(bucket_max_len, count_of_batches_in_bucket)`
    /// with buckets sorted ascending; a batch with padded_len L is counted in
    /// the smallest bucket whose max >= L. Returns `None` for backends that
    /// don't instrument this (default). Shows what shape TensorRT would see.
    fn seq_len_histogram(&self) -> Option<Vec<(usize, u64)>> {
        None
    }

    /// Optional per-run histogram of individual item (pre-padding) token
    /// lengths. Complements `seq_len_histogram`: batch-max shows what TRT
    /// sees, per-item shows the actual content distribution and therefore
    /// how much compute batch-max padding is wasting.
    fn item_seq_len_histogram(&self) -> Option<Vec<(usize, u64)>> {
        None
    }
}

/// A probe marker emitted by the model to request recall
#[derive(Debug, Clone)]
pub struct ProbeMarker {
    pub content: String,
    pub position: usize,
}

/// A single reasoning step extracted from a model's thinking trace
#[derive(Debug, Clone)]
pub struct ThinkingStep {
    pub content: String,
    pub step_number: usize,
}

/// An explicit line-range reference detected in model output or thinking traces.
/// Carries enough information to load the specific range from the workspace.
#[derive(Debug, Clone)]
pub struct LineReference {
    /// The source as written by the model — may be a partial path like `lib.rs`
    /// or a full path like `./crates/caw-core/src/lib.rs`.
    pub source_hint: String,
    pub start: usize,
    pub end: usize,
}

/// Persistent storage for stubs and their embeddings.
///
/// Content is NOT stored here — stubs carry `(path, byte_offset, byte_length)`
/// and `get_content` reads the body back from the source file on demand.
/// Storing chunk text in the index duplicated the corpus by Nx (one copy per
/// chunk); disk is the cheap source of truth.
///
/// Does NOT include similarity search — that belongs in a proper
/// vector index (HNSW, IVF, etc.), not a row-scan over blobs.
pub trait StubStore {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>) -> CawResult<()>;
    /// Read this stub's body from the source file by slicing
    /// `[byte_offset, byte_offset + byte_length)` out of `path`. Returns an
    /// error if the store has no corpus root configured or the file is
    /// missing/shorter than expected.
    fn get_content(&self, id: &StubId) -> CawResult<String>;
    fn get_stub(&self, id: &StubId) -> CawResult<Stub>;
    /// Look up an existing stub by its content hash. Returns the stub and its
    /// embedding if found, allowing callers to skip re-ingestion and
    /// re-embedding when file content hasn't changed.
    fn get_by_content_hash(&self, hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>>;
    /// Iterate all stored embeddings. Used by vector index implementations
    /// to build their index from persisted data.
    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>>;

    /// Persist a consolidation note for a stub. Called on eviction and
    /// when the model emits annotations. Notes accumulate across sessions,
    /// making stubs richer over time.
    fn save_consolidation(
        &mut self,
        _stub_id: &StubId,
        _note: &ConsolidationNote,
    ) -> CawResult<()> {
        Ok(())
    }

    /// Load all consolidation notes for a stub from persistent storage.
    fn load_consolidation(&self, _stub_id: &StubId) -> CawResult<Vec<ConsolidationNote>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fragment(source: &str, locator: &str, content: &str) -> RecallFragment {
        RecallFragment {
            stub_id: StubId(source.to_string()),
            content: content.to_string(),
            locator: Locator {
                source: source.to_string(),
                locator: locator.to_string(),
            },
            tokens: 0,
            mtime_unix_secs: 0,
        }
    }

    #[test]
    fn candidate_list_marks_metadata_as_non_evidence() {
        let hits = vec![ScoredStub {
            stub: Stub {
                id: StubId("stub-1".to_string()),
                path: "bench-results/throughput/20260419-053329/summary.txt".to_string(),
                token_estimate: 0,
                kind: ContentKind::Markdown,
                summary: "Throughput benchmark summary".to_string(),
                outline: Vec::new(),
                content_hash: String::new(),
                mtime_unix_secs: 0,
                byte_offset: 0,
                byte_length: 0,
                consolidation_notes: Vec::new(),
            },
            score: 0.9,
        }];

        let fragment = candidate_list_fragment(&hits, 0.7);
        assert!(
            fragment
                .content
                .contains("discovery metadata, not evidence")
        );
        assert!(fragment.content.contains("do not claim details"));
    }

    #[test]
    fn workspace_format_marks_candidate_lists_as_metadata() {
        let request = CompletionRequest {
            user: "do you know what benchmarks have been run?".to_string(),
            workspace_fragments: vec![sample_fragment(
                "search-candidates",
                "file-list",
                "- bench-results/throughput/20260419-053329/summary.txt",
            )],
            ..Default::default()
        };

        let formatted = request.format_workspace(ProvenanceFormat::Bracketed);
        assert!(formatted.contains("Answering hints:"));
        assert!(formatted.contains("candidate metadata only"));
    }

    #[test]
    fn workspace_format_includes_explicit_extra_guidance() {
        let request = CompletionRequest {
            user: "can you tell me what the benchmark results were?".to_string(),
            workspace_fragments: vec![sample_fragment(
                "bench-results/throughput/20260419-053329/summary.txt",
                "full",
                "done: 158895 stubs across 27258 files in 392.2s",
            )],
            ..Default::default()
        };

        let formatted = request.format_workspace_with_guidance(
            ProvenanceFormat::Bracketed,
            &[
                "For inventory questions, list exact run names or paths present in recalled text.",
                "For result questions, report only values explicitly present in loaded evidence.",
            ],
        );
        assert!(formatted.contains("list exact run names or paths present in recalled text"));
        assert!(formatted.contains("report only values explicitly present in loaded evidence"));
    }

    #[test]
    fn query_intent_parses_json_from_classifier_output() {
        let (parsed, keys) = QueryIntent::from_classifier_response(
            "```json\n{\"is_inventory_request\":true,\"confidence\":0.82}\n```",
        )
        .unwrap();
        assert!(parsed.is_inventory_request);
        assert_eq!(parsed.confidence, Some(0.82));
        assert!(keys.contains("is_inventory_request"));
        assert!(keys.contains("confidence"));
    }

    #[test]
    fn query_intent_guidance_is_added_to_workspace() {
        let request = CompletionRequest {
            system: String::new(),
            user: "what are the latest results?".to_string(),
            workspace_fragments: vec![sample_fragment("report.json", "full", "latency_ms: 123")],
            workspace_guidance: QueryIntent {
                is_results_request: true,
                wants_numeric_values: true,
                needs_grounded_evidence_only: true,
                confidence: Some(0.9),
                ..Default::default()
            }
            .guidance_lines(),
        };

        let formatted = request.format_workspace(ProvenanceFormat::Bracketed);
        assert!(formatted.contains("report only metrics and values explicitly present"));
        assert!(formatted.contains("Prefer exact numeric values and units"));
        assert!(
            formatted.contains(
                "state what evidence, artifact, or file is needed to fulfill the request"
            )
        );
    }

    #[test]
    fn query_intent_status_guidance_is_added_to_workspace() {
        let request = CompletionRequest {
            system: String::new(),
            user: "is benchmarking all completed?".to_string(),
            workspace_fragments: vec![sample_fragment(
                "TODO.md",
                "full",
                "- benchmark harness: done",
            )],
            workspace_guidance: QueryIntent {
                is_status_request: true,
                wants_completion_state: true,
                needs_grounded_evidence_only: true,
                confidence: Some(0.93),
                ..Default::default()
            }
            .guidance_lines(),
        };

        let formatted = request.format_workspace(ProvenanceFormat::Bracketed);
        assert!(formatted.contains(
            "For status questions, distinguish clearly between completed, pending, and unknown work"
        ));
        assert!(
            formatted
                .contains("State explicitly whether each relevant task or artifact is completed")
        );
        assert!(formatted.contains("say what evidence is needed to decide it"));
    }

    #[test]
    fn query_intent_next_step_guidance_is_added_to_workspace() {
        let request = CompletionRequest {
            system: String::new(),
            user: "what's next?".to_string(),
            workspace_fragments: vec![sample_fragment(
                "TODO.md",
                "full",
                "- benchmark harness: done",
            )],
            workspace_guidance: QueryIntent {
                is_next_step_request: true,
                wants_recommended_actions: true,
                needs_grounded_evidence_only: true,
                confidence: Some(0.88),
                ..Default::default()
            }
            .guidance_lines(),
        };

        let formatted = request.format_workspace(ProvenanceFormat::Bracketed);
        assert!(formatted.contains(
            "For next-step questions, recommend concrete next actions grounded in recalled evidence"
        ));
        assert!(formatted.contains(
            "say what planning or status evidence is needed before recommending a next step"
        ));
    }

    #[test]
    fn every_admitted_chunk_is_rendered_including_same_source() {
        // The renderer no longer collapses multiple chunks of one file: each
        // chunk of a multi-chunk document can hold a different answer, so all
        // admitted fragments are rendered verbatim. Per-source limiting is the
        // producer's job (orchestrator admission / proxy clamp), not the
        // renderer's.
        let request = CompletionRequest {
            system: String::new(),
            user: String::new(),
            workspace_fragments: vec![
                sample_fragment("lib.rs", "1-40", "fn foo() {}"),
                sample_fragment("other.rs", "full", "fn bar() {}"),
                sample_fragment("lib.rs", "80-120", "fn baz() {}"),
                sample_fragment("lib.rs", "200-240", "fn qux() {}"),
            ],
            workspace_guidance: vec![],
        };

        let formatted = request.format_workspace(ProvenanceFormat::Bracketed);

        // All three lib.rs chunks render as full blocks with their content.
        assert!(formatted.contains("[recalled from lib.rs:1-40]"));
        assert!(formatted.contains("fn foo() {}"));
        assert!(formatted.contains("[recalled from lib.rs:80-120]"));
        assert!(formatted.contains("fn baz() {}"));
        assert!(formatted.contains("[recalled from lib.rs:200-240]"));
        assert!(formatted.contains("fn qux() {}"));
        // The old collapse-to-note behavior is gone.
        assert!(!formatted.contains("more section(s) from this file"));
        // A "full" locator still renders without the ":full" noise.
        assert!(formatted.contains("[recalled from other.rs]"));
        assert!(!formatted.contains("[recalled from other.rs:full]"));
        assert!(formatted.contains("fn bar() {}"));
    }
}

/// No-op store for orchestrators that don't need persistent consolidation.
/// Satisfies the `S: StubStore` bound without importing an index crate.
/// `store` is always `None` when using `DynamicRecallOrchestrator::new()`,
/// so none of these methods are reachable at runtime.
impl StubStore for () {
    fn insert(&mut self, _stub: Stub, _embedding: Vec<f32>) -> CawResult<()> {
        Err(CawError::NotFound("no store configured".into()))
    }
    fn get_content(&self, id: &StubId) -> CawResult<String> {
        Err(CawError::NotFound(format!(
            "no store configured (stub {})",
            id.0
        )))
    }
    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        Err(CawError::NotFound(format!(
            "no store configured (stub {})",
            id.0
        )))
    }
    fn get_by_content_hash(&self, _hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>> {
        Ok(None)
    }
    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>> {
        Ok(Vec::new())
    }
}

/// Similarity search over embeddings. Implementations should use an
/// actual indexing structure (HNSW, IVF, etc.), not brute-force scans.
pub trait VectorIndex {
    fn add(&mut self, id: StubId, embedding: Vec<f32>);
    /// Returns (StubId, similarity_score) pairs, highest similarity first.
    fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(StubId, f32)>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Token counting abstraction. Allows swapping between cheap heuristic
/// estimators and real BPE tokenizers depending on accuracy needs.
pub trait Tokenizer: Send + Sync {
    fn count_tokens(&self, text: &str) -> usize;
    fn tokenizer_name(&self) -> &str;
}

/// Fallback tokenizer that splits on whitespace. Fast but diverges
/// significantly from BPE counts, especially on code.
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count().max(1)
    }

    fn tokenizer_name(&self) -> &str {
        "whitespace"
    }
}
