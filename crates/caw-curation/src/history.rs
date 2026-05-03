use caw_core::{CawResult, CompletionRequest, ModelAdapter};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnRole {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurnMetadata {
    pub tool_name: Option<String>,
    /// Tool outputs marked inline_required bypass compression and summarization
    pub inline_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: TurnRole,
    pub content: String,
    pub token_estimate: usize,
    pub timestamp_secs: u64,
    pub metadata: TurnMetadata,
}

impl ConversationTurn {
    pub fn is_tool_output(&self) -> bool {
        self.role == TurnRole::Tool
    }
}

#[derive(Debug, Clone)]
pub struct HistorySummarizerConfig {
    /// Fraction of context budget that triggers summarization (0.0 - 1.0)
    pub trigger_threshold_pct: f32,
    /// Number of most recent turns to always preserve verbatim
    pub retain_recent: usize,
    /// Max tokens the summary itself should target
    pub summary_budget_tokens: usize,
}

impl Default for HistorySummarizerConfig {
    fn default() -> Self {
        Self {
            trigger_threshold_pct: 0.30,
            retain_recent: 3,
            summary_budget_tokens: 1000,
        }
    }
}

pub trait HistorySummarizer {
    /// Compress a sequence of conversation turns into a summary string.
    /// The input slice contains only the turns eligible for summarization
    /// (i.e., the recent turns have already been excluded by the caller).
    fn summarize(&self, turns: &[ConversationTurn]) -> CawResult<String>;
}

/// Uses a ModelAdapter to generate a natural language summary of conversation history.
pub struct LlmHistorySummarizer<'a> {
    adapter: &'a dyn ModelAdapter,
    config: HistorySummarizerConfig,
}

impl<'a> LlmHistorySummarizer<'a> {
    pub fn new_with(adapter: &'a dyn ModelAdapter, config: HistorySummarizerConfig) -> Self {
        Self { adapter, config }
    }
}

impl HistorySummarizer for LlmHistorySummarizer<'_> {
    fn summarize(&self, turns: &[ConversationTurn]) -> CawResult<String> {
        if turns.is_empty() {
            return Ok(String::new());
        }

        let formatted = format_turns_for_summary(turns);

        let prompt = format!(
            "Summarize the following conversation history into a concise summary \
             (target: under {} tokens). Preserve: decisions made, facts established, \
             current task state. Drop: exploratory reasoning that led to dead ends, \
             verbose tool outputs from completed steps.\n\n{}",
            self.config.summary_budget_tokens, formatted
        );

        let req = CompletionRequest {
            // The summary is injected at the top of the context window when a
            // new session or context-reset continues an earlier conversation.
            // The model receiving it has no access to the original turns, so
            // the summary must be self-contained: enough to reconstruct what
            // was decided, what is still open, and what the user was trying to
            // accomplish — not just a description of topics discussed.
            system: "Produce a dense, factual summary of the conversation \
                     history below. The summary will be injected as prior-session \
                     context for a future assistant turn; the original history will \
                     not be available. Preserve: (1) the current task and any \
                     unresolved questions, (2) decisions or conclusions reached, \
                     (3) specific names, paths, numbers, or identifiers that were \
                     referenced, (4) any explicit user preferences or constraints \
                     stated. Omit: exploratory reasoning that was abandoned, \
                     verbose restatements of tool output that was already acted on. \
                     Output the summary only — no preamble, no labels."
                .to_string(),
            user: prompt,
            workspace_fragments: vec![],
            workspace_guidance: Vec::new(),
        };

        let resp = self.adapter.complete(req)?;
        Ok(resp.answer)
    }
}

/// Deterministic fallback summarizer that extracts key sentences without an LLM.
/// Keeps the first and last sentence from each turn, plus any line containing
/// decision-signal words.
pub struct ExtractiveHistorySummarizer;

impl HistorySummarizer for ExtractiveHistorySummarizer {
    fn summarize(&self, turns: &[ConversationTurn]) -> CawResult<String> {
        if turns.is_empty() {
            return Ok(String::new());
        }

        let mut extracted = Vec::new();

        for turn in turns {
            let sentences: Vec<&str> = turn
                .content
                .split(['.', '!', '?'])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if sentences.is_empty() {
                continue;
            }

            let role_prefix = match turn.role {
                TurnRole::User => "[User]",
                TurnRole::Assistant => "[Assistant]",
                TurnRole::System => "[System]",
                TurnRole::Tool => {
                    let name = turn.metadata.tool_name.as_deref().unwrap_or("unknown");
                    &format!("[Tool: {}]", name)
                }
            };

            // Always keep first sentence for context
            extracted.push(format!("{} {}", role_prefix, sentences[0]));

            // Keep sentences containing decision-signal words
            let signals = [
                "decided",
                "conclusion",
                "result",
                "therefore",
                "established",
                "confirmed",
                "found",
                "error",
                "fixed",
                "changed",
                "updated",
                "created",
                "deleted",
                "moved",
                "renamed",
            ];

            for &sentence in &sentences[1..] {
                let lower = sentence.to_lowercase();
                if signals.iter().any(|s| lower.contains(s)) {
                    extracted.push(format!("  - {}", sentence));
                }
            }

            // Keep last sentence if distinct from first
            if sentences.len() > 1 {
                extracted.push(format!("  ... {}", sentences[sentences.len() - 1]));
            }
        }

        Ok(extracted.join("\n"))
    }
}

/// Determine which turns should be summarized vs retained, based on config.
/// Returns (to_summarize, to_retain) slices.
pub fn partition_turns(
    turns: &[ConversationTurn],
    config: &HistorySummarizerConfig,
    context_budget_tokens: usize,
) -> (Vec<ConversationTurn>, Vec<ConversationTurn>) {
    let total_tokens: usize = turns.iter().map(|t| t.token_estimate).sum();
    let threshold = (context_budget_tokens as f32 * config.trigger_threshold_pct) as usize;

    if total_tokens <= threshold || turns.len() <= config.retain_recent {
        return (vec![], turns.to_vec());
    }

    let split_point = turns.len().saturating_sub(config.retain_recent);
    let to_summarize = turns[..split_point].to_vec();
    let to_retain = turns[split_point..].to_vec();
    (to_summarize, to_retain)
}

fn format_turns_for_summary(turns: &[ConversationTurn]) -> String {
    turns
        .iter()
        .map(|t| {
            let role = match t.role {
                TurnRole::User => "User",
                TurnRole::Assistant => "Assistant",
                TurnRole::System => "System",
                TurnRole::Tool => "Tool",
            };
            format!("[{}]: {}", role, t.content)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
