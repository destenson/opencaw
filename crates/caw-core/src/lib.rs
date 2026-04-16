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

#[derive(Debug, Clone)]
pub enum ContentRange {
    Full,
    Lines { start: usize, end: usize },
    Heading { path: String },
    TokenWindow { start: usize, count: usize },
}

impl ContentRange {
    pub fn parse(range_str: &str) -> Self {
        if range_str == "full" {
            return Self::Full;
        }

        if range_str.starts_with('#') {
            return Self::Heading {
                path: range_str[1..].to_string(),
            };
        }

        if let Some((start, end)) = range_str.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse(), end.parse()) {
                return Self::Lines { start: s, end: e };
            }
        }

        Self::Full
    }

    pub fn apply(&self, content: &str) -> String {
        match self {
            Self::Full => content.to_string(),
            Self::Lines { start, end } => content
                .lines()
                .skip(start.saturating_sub(1))
                .take(end.saturating_sub(*start) + 1)
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Heading { path } => extract_heading_section(content, path),
            Self::TokenWindow { start, count } => {
                // Simple token approximation: split on whitespace
                let tokens: Vec<&str> = content.split_whitespace().collect();
                tokens
                    .iter()
                    .skip(*start)
                    .take(*count)
                    .map(|s| *s)
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }
}

fn extract_heading_section(content: &str, heading_path: &str) -> String {
    let parts: Vec<&str> = heading_path.split('/').collect();
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::new();
    let mut in_section = false;
    let mut current_level = 0;

    for line in lines {
        // Check if this is a heading
        if let Some(stripped) = line.trim_start().strip_prefix('#') {
            let level = line.chars().take_while(|c| *c == '#').count();
            let heading_text = stripped.trim();

            // Check if this matches our target heading
            if parts.iter().any(|p| heading_text.contains(p)) {
                in_section = true;
                current_level = level;
                result.push(line);
                continue;
            }

            // If we're in a section and hit a same-or-higher level heading, stop
            if in_section && level <= current_level {
                break;
            }
        }

        if in_section {
            result.push(line);
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
    /// Generate embeddings for a batch of texts
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>>;
    
    /// Get the dimension of embeddings produced by this provider
    fn dimension(&self) -> usize;
    
    /// Get a human-readable name for this embedding provider
    fn provider_name(&self) -> &str;
}

/// A range selection within recalled content, used by the recall pipeline
/// to address specific portions of a file (lines, headings, token windows).
/// Distinct from ContentRange which handles simpler range parsing for the
/// general retriever interface.
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
        if s.eq_ignore_ascii_case("full") {
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
    /// Insert a stub with its embedding
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>, content: String) -> CawResult<()>;
    
    /// Search for similar stubs given a query embedding
    fn search_by_embedding(&self, query_embedding: &[f32], top_k: usize) -> CawResult<Vec<ScoredStub>>;
    
    /// Get content for a specific stub
    fn get_content(&self, id: &StubId) -> CawResult<String>;
    
    /// Get a stub by ID
    fn get_stub(&self, id: &StubId) -> CawResult<Stub>;
}
