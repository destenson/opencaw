use caw_core::{CawError, CawResult};

use crate::history::{
    ConversationTurn, HistorySummarizer, HistorySummarizerConfig, partition_turns,
};
use crate::prompt_budget::{self, PromptBudgetCheck, SystemPromptBudget};
use crate::tool_output::{CompressedToolOutput, ToolOutputCompressor};

/// Output of running the curation pipeline over a set of inputs.
#[derive(Debug)]
pub struct CurationResult {
    /// System prompt, possibly truncated
    pub system_prompt: String,
    /// Summary of old turns (empty if no summarization was needed)
    pub history_summary: String,
    /// Recent turns preserved verbatim
    pub retained_turns: Vec<ConversationTurn>,
    /// Tool outputs that were compressed (full output stored inside each)
    pub compressed_tool_outputs: Vec<CompressedToolOutput>,
    /// Budget check result for the system prompt
    pub prompt_budget_check: PromptBudgetCheck,
    /// Total tokens saved by curation
    pub tokens_saved: usize,
}

/// Composes history summarization, tool output compression, and prompt budgeting
/// into a single pass over conversation inputs.
pub struct CurationPipeline<'a> {
    history_summarizer: &'a dyn HistorySummarizer,
    history_config: HistorySummarizerConfig,
    tool_compressor: &'a dyn ToolOutputCompressor,
    prompt_budget: SystemPromptBudget,
    context_budget_tokens: usize,
    truncate_on_exceed: bool,
}

pub struct CurationPipelineBuilder<'a> {
    history_summarizer: Option<&'a dyn HistorySummarizer>,
    history_config: HistorySummarizerConfig,
    tool_compressor: Option<&'a dyn ToolOutputCompressor>,
    prompt_budget: SystemPromptBudget,
    context_budget_tokens: usize,
    truncate_on_exceed: bool,
}

impl<'a> CurationPipelineBuilder<'a> {
    pub fn with_context_budget(context_budget_tokens: usize) -> Self {
        Self {
            history_summarizer: None,
            history_config: HistorySummarizerConfig::default(),
            tool_compressor: None,
            // Default: 10% of context budget for system prompt
            prompt_budget: SystemPromptBudget::from_context_window(context_budget_tokens, 0.10),
            context_budget_tokens,
            truncate_on_exceed: false,
        }
    }

    pub fn history_summarizer(mut self, summarizer: &'a dyn HistorySummarizer) -> Self {
        self.history_summarizer = Some(summarizer);
        self
    }

    pub fn history_config(mut self, config: HistorySummarizerConfig) -> Self {
        self.history_config = config;
        self
    }

    pub fn tool_compressor(mut self, compressor: &'a dyn ToolOutputCompressor) -> Self {
        self.tool_compressor = Some(compressor);
        self
    }

    pub fn prompt_budget(mut self, budget: SystemPromptBudget) -> Self {
        self.prompt_budget = budget;
        self
    }

    /// If true, system prompts exceeding the budget are truncated rather than
    /// just flagged. Defaults to false.
    pub fn truncate_on_exceed(mut self, truncate: bool) -> Self {
        self.truncate_on_exceed = truncate;
        self
    }

    pub fn build(self) -> CawResult<CurationPipeline<'a>> {
        let history_summarizer = self
            .history_summarizer
            .ok_or_else(|| CawError::InvalidInput("history_summarizer is required".to_string()))?;
        let tool_compressor = self
            .tool_compressor
            .ok_or_else(|| CawError::InvalidInput("tool_compressor is required".to_string()))?;

        Ok(CurationPipeline {
            history_summarizer,
            history_config: self.history_config,
            tool_compressor,
            prompt_budget: self.prompt_budget,
            context_budget_tokens: self.context_budget_tokens,
            truncate_on_exceed: self.truncate_on_exceed,
        })
    }
}

impl CurationPipeline<'_> {
    /// Run curation over a complete set of conversation inputs.
    pub fn curate(
        &self,
        system_prompt: &str,
        turns: &[ConversationTurn],
    ) -> CawResult<CurationResult> {
        let mut tokens_saved: usize = 0;

        // 1. Check system prompt budget
        let prompt_check = prompt_budget::check_system_prompt(system_prompt, &self.prompt_budget);

        let final_prompt = if self.truncate_on_exceed && prompt_check.is_exceeded() {
            let (truncated, dropped) =
                prompt_budget::truncate_to_budget(system_prompt, &self.prompt_budget);
            tokens_saved += dropped;
            truncated
        } else {
            system_prompt.to_string()
        };

        // 2. Partition history and summarize if over threshold
        let (to_summarize, retained) =
            partition_turns(turns, &self.history_config, self.context_budget_tokens);

        let history_summary = if to_summarize.is_empty() {
            String::new()
        } else {
            let original_tokens: usize = to_summarize.iter().map(|t| t.token_estimate).sum();
            let summary = self.history_summarizer.summarize(&to_summarize)?;
            let summary_tokens = crate::tool_output::estimate_tokens(&summary);
            tokens_saved += original_tokens.saturating_sub(summary_tokens);
            summary
        };

        // 3. Compress tool outputs in the retained turns
        let mut compressed_outputs = Vec::new();
        for turn in &retained {
            if turn.is_tool_output()
                && !turn.metadata.inline_required
                && let Some(compressed) = self.tool_compressor.compress(
                turn.metadata.tool_name.as_deref().unwrap_or("unknown"),
                &turn.content,
                turn.token_estimate,
            )? {
                tokens_saved += compressed
                    .original_tokens
                    .saturating_sub(compressed.summary_tokens);
                compressed_outputs.push(compressed);
            }
        }

        Ok(CurationResult {
            system_prompt: final_prompt,
            history_summary,
            retained_turns: retained,
            compressed_tool_outputs: compressed_outputs,
            prompt_budget_check: prompt_check,
            tokens_saved,
        })
    }
}
