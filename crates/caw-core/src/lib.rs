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
    fn search(&self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>>;
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
