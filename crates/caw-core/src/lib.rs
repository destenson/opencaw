pub mod provenance;
pub mod reindex;
pub mod scheduler;
pub mod tokenizer;

pub use provenance::tokenize_terms;
pub use reindex::{ChannelReindexQueue, NoopReindexQueue, ReindexQueue, ReindexReceiver};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type CawResult<T> = Result<T, CawError>;

#[derive(Debug, Error)]
pub enum CawError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("adapter error: {0}")]
    Adapter(String),
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
}

impl From<std::io::Error> for CawError {
    fn from(e: std::io::Error) -> Self {
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
pub fn is_looping(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 20 {
        return false;
    }

    // Single-word dominance
    let modal_count = {
        let mut counts = std::collections::HashMap::new();
        for &w in &words {
            *counts.entry(w).or_insert(0usize) += 1;
        }
        counts.into_values().max().unwrap_or(0)
    };
    if modal_count * 100 / words.len() > 70 {
        return true;
    }

    // Trigram diversity collapse
    if words.len() >= 30 {
        let total = words.len() - 2;
        let unique: std::collections::HashSet<[&str; 3]> = words
            .windows(3)
            .map(|w| [w[0], w[1], w[2]])
            .collect();
        if unique.len() * 10 < total {
            return true;
        }
    }

    false
}

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
        if let Some(rest) = s.strip_prefix('T')
            && let Some((start, count)) = rest.split_once(':')
            && let (Ok(s), Ok(c)) = (start.parse(), count.parse())
        {
            return Self::Tokens { start: s, count: c };
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

            Self::Tokens { start, count } => {
                let tokens: Vec<&str> = content.split_whitespace().collect();
                tokens
                    .iter()
                    .skip(*start)
                    .take(*count)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" ")
            }

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
         Mention the specific ones you need in order to answer accurately:\n\n{}",
        lines.join("\n")
    );

    RecallFragment {
        stub_id: StubId("__candidates__".to_string()),
        content: content.clone(),
        locator: Locator {
            source: "search-candidates".to_string(),
            locator: "file-list".to_string(),
        },
        tokens: content.split_whitespace().count().max(1),
    }
}

#[derive(Debug, Clone)]
pub struct RecallFragment {
    pub stub_id: StubId,
    pub content: String,
    pub locator: Locator,
    pub tokens: usize,
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

#[derive(Debug, Clone, Copy)]
pub struct ModelCapabilities {
    pub supports_tool_calls: bool,
    pub supports_hidden_reasoning: bool,
    pub supports_visible_reasoning: bool,
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system: String,
    pub user: String,
    pub workspace_fragments: Vec<RecallFragment>,
}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub answer: String,
    /// Visible reasoning emitted by the model before the answer, if any.
    /// Populated by adapters that detect `<think>...</think>` blocks.
    /// Useful for thinking-trace recall without the orchestrator having to
    /// re-parse the raw output.
    pub thinking: Option<String>,
}

/// Split a raw model response into (thinking, answer) by extracting the first
/// `<think>...</think>` block. Returns `(None, raw)` if no block is found.
pub fn split_thinking(raw: &str) -> (Option<String>, String) {
    let open = raw.find("<think>");
    let close = raw.find("</think>");
    match (open, close) {
        (Some(o), Some(c)) if c > o => {
            let thinking = raw[o + "<think>".len()..c].trim().to_string();
            let answer = raw[c + "</think>".len()..].trim().to_string();
            (Some(thinking), answer)
        }
        _ => (None, raw.to_string()),
    }
}

/// Format for inline provenance tags on recalled content.
#[derive(Debug, Clone, Copy)]
pub enum ProvenanceFormat {
    /// XML tags: `<recalled from="path" locator="range">content</recalled>`
    /// Preferred by Anthropic models which handle XML natively.
    Xml,
    /// Bracket tags: `[recalled from path:range]\ncontent\n[end recall]`
    /// Preferred by OpenAI-protocol models.
    Bracketed,
}

impl CompletionRequest {
    /// Format workspace fragments with inline provenance for model consumption.
    /// Each fragment is wrapped with source attribution so the model treats
    /// recalled content as quoted material, not its own knowledge.
    pub fn format_workspace(&self, format: ProvenanceFormat) -> String {
        if self.workspace_fragments.is_empty() {
            return String::new();
        }

        let fragments: Vec<String> = self
            .workspace_fragments
            .iter()
            .map(|f| {
                let source = &f.locator.source;
                let locator = &f.locator.locator;
                match format {
                    ProvenanceFormat::Xml => format!(
                        "<recalled from=\"{source}\" locator=\"{locator}\">\n\
                         {content}\n\
                         </recalled>",
                        content = f.content,
                    ),
                    ProvenanceFormat::Bracketed => format!(
                        "[recalled from {source}:{locator}]\n\
                         {content}\n\
                         [end recall]",
                        content = f.content,
                    ),
                }
            })
            .collect();

        format!(
            "\n\nRecalled workspace context (each block is verbatim from the cited source — \
             treat as quoted material, not your own knowledge):\n\n{}",
            fragments.join("\n\n")
        )
    }
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
            unload: 0.4,
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
    ) -> CawResult<()> {
        let response = self.complete(req)?;
        let thinking = response.thinking.unwrap_or_default();
        for step in thinking
            .split("\n\n")
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if !on_step(step)? {
                break;
            }
        }
        Ok(())
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

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<()> {
        (**self).thinking_with_steps(req, on_step)
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

/// No-op store for orchestrators that don't need persistent consolidation.
/// Satisfies the `S: StubStore` bound without importing an index crate.
/// `store` is always `None` when using `DynamicRecallOrchestrator::new()`,
/// so none of these methods are reachable at runtime.
impl StubStore for () {
    fn insert(&mut self, _stub: Stub, _embedding: Vec<f32>) -> CawResult<()> {
        Err(CawError::NotFound("no store configured".into()))
    }
    fn get_content(&self, id: &StubId) -> CawResult<String> {
        Err(CawError::NotFound(format!("no store configured (stub {})", id.0)))
    }
    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        Err(CawError::NotFound(format!("no store configured (stub {})", id.0)))
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
