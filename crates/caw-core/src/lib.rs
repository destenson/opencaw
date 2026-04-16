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
}

impl From<std::io::Error> for CawError {
    fn from(e: std::io::Error) -> Self {
        CawError::Io(e.to_string())
    }
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
        if let Some(rest) = s.strip_prefix('T') {
            if let Some((start, count)) = rest.split_once(':') {
                if let (Ok(s), Ok(c)) = (start.parse(), count.parse()) {
                    return Self::Tokens { start: s, count: c };
                }
            }
        }

        // Line range with L prefix: L5-L15
        if s.starts_with('L') {
            let stripped = s.replace('L', "");
            if let Some((start, end)) = stripped.split_once('-') {
                if let (Ok(s), Ok(e)) = (start.parse(), end.parse()) {
                    return Self::Lines { start: s, end: e };
                }
            }
        }

        // Plain line range: 10-20
        if let Some((start, end)) = s.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse(), end.parse()) {
                return Self::Lines { start: s, end: e };
            }
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
            let heading_text = after_hashes
                .trim_start_matches('#')
                .trim();

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
}

#[derive(Debug, Clone)]
pub struct ScoredStub {
    pub stub: Stub,
    pub score: f32,
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

pub trait Retriever {
    fn search(&mut self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>>;
    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment>;
}

pub trait BudgetScheduler {
    fn schedule(&self, input: SchedulerInput) -> SchedulerDecision;
}

pub trait ProvenanceStore {
    fn record(&mut self, fragment: RecallFragment);
    fn all(&self) -> Vec<RecallFragment>;
}

pub trait ModelAdapter {
    fn model_name(&self) -> &str;
    fn capabilities(&self) -> ModelCapabilities;
    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse>;
}

/// Embedding generation trait - abstracts different embedding backends
pub trait EmbeddingProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn provider_name(&self) -> &str;
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

/// Vector storage and similarity search trait
pub trait VectorStore {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>, content: String) -> CawResult<()>;
    fn search_by_embedding(&self, query_embedding: &[f32], top_k: usize) -> CawResult<Vec<ScoredStub>>;
    fn get_content(&self, id: &StubId) -> CawResult<String>;
    fn get_stub(&self, id: &StubId) -> CawResult<Stub>;
}
