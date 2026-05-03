use caw_core::{CawResult, CompletionRequest, ModelAdapter, Tokenizer, WhitespaceTokenizer};

/// The result of compressing a tool output: a short summary for context,
/// plus the full original stored for later recall.
#[derive(Debug, Clone)]
pub struct CompressedToolOutput {
    pub tool_name: String,
    pub summary: String,
    pub full_output: String,
    pub original_tokens: usize,
    pub summary_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ToolOutputCompressorConfig {
    /// Tool outputs exceeding this token count get compressed
    pub token_threshold: usize,
    /// Max tokens for the compressed summary
    pub summary_budget_tokens: usize,
}

impl Default for ToolOutputCompressorConfig {
    fn default() -> Self {
        Self {
            token_threshold: 500,
            summary_budget_tokens: 200,
        }
    }
}

pub trait ToolOutputCompressor {
    /// Compress a tool output into a summary. Returns None if the output
    /// is below the compression threshold (caller should keep it inline).
    fn compress(
        &self,
        tool_name: &str,
        output: &str,
        token_estimate: usize,
    ) -> CawResult<Option<CompressedToolOutput>>;
}

/// Uses a ModelAdapter to generate intelligent summaries of tool output.
pub struct LlmToolOutputCompressor<'a> {
    adapter: &'a dyn ModelAdapter,
    config: ToolOutputCompressorConfig,
}

impl<'a> LlmToolOutputCompressor<'a> {
    pub fn new_with(adapter: &'a dyn ModelAdapter, config: ToolOutputCompressorConfig) -> Self {
        Self { adapter, config }
    }
}

impl ToolOutputCompressor for LlmToolOutputCompressor<'_> {
    fn compress(
        &self,
        tool_name: &str,
        output: &str,
        token_estimate: usize,
    ) -> CawResult<Option<CompressedToolOutput>> {
        if token_estimate <= self.config.token_threshold {
            return Ok(None);
        }

        let prompt = format!(
            "Summarize this tool output concisely (under {} tokens). \
             Include key findings and counts. The tool was '{}'.\n\n{}",
            self.config.summary_budget_tokens, tool_name, output
        );

        let req = CompletionRequest {
            // The summary replaces the full tool output inline in the context
            // window. The full output is stored for recall, so the summary
            // must preserve the signal a reader needs to decide whether to
            // recall — not a generic description. Prioritize: concrete counts,
            // error/warning totals, key identifiers found or not found, and
            // the actionable outcome.
            system: "Summarize the tool output below. Focus on the actionable \
                     result: what was found or not found, counts that matter, \
                     errors or warnings, and any specific identifiers (file paths, \
                     function names, values) that are likely to be referenced next. \
                     Be terse — a few sentences at most. Output the summary only."
                .to_string(),
            user: prompt,
            workspace_fragments: vec![],
            workspace_guidance: Vec::new(),
        };

        let resp = self.adapter.complete(req)?;
        let summary_tokens = estimate_tokens(&resp.answer);

        Ok(Some(CompressedToolOutput {
            tool_name: tool_name.to_string(),
            summary: resp.answer,
            full_output: output.to_string(),
            original_tokens: token_estimate,
            summary_tokens,
        }))
    }
}

/// Deterministic compressor that extracts the first few and last few lines,
/// plus a line count. No LLM needed.
pub struct ExtractiveToolOutputCompressor {
    config: ToolOutputCompressorConfig,
}

impl ExtractiveToolOutputCompressor {
    pub fn new_with(config: ToolOutputCompressorConfig) -> Self {
        Self { config }
    }
}

impl ToolOutputCompressor for ExtractiveToolOutputCompressor {
    fn compress(
        &self,
        tool_name: &str,
        output: &str,
        token_estimate: usize,
    ) -> CawResult<Option<CompressedToolOutput>> {
        if token_estimate <= self.config.token_threshold {
            return Ok(None);
        }

        let lines: Vec<&str> = output.lines().collect();
        let total_lines = lines.len();

        // Take first 3 and last 2 lines as a representative sample
        let head: Vec<&str> = lines.iter().take(3).copied().collect();
        let tail: Vec<&str> = if total_lines > 5 {
            lines
                .iter()
                .rev()
                .take(2)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        } else {
            vec![]
        };

        let summary = if tail.is_empty() {
            format!(
                "[tool: {}] {} lines of output. Preview:\n{}",
                tool_name,
                total_lines,
                head.join("\n"),
            )
        } else {
            format!(
                "[tool: {}] {} lines of output. Preview:\n{}\n  ... ({} lines omitted) ...\n{}\nFull results available via recall.",
                tool_name,
                total_lines,
                head.join("\n"),
                total_lines.saturating_sub(5),
                tail.join("\n"),
            )
        };

        let summary_tokens = estimate_tokens(&summary);

        Ok(Some(CompressedToolOutput {
            tool_name: tool_name.to_string(),
            summary,
            full_output: output.to_string(),
            original_tokens: token_estimate,
            summary_tokens,
        }))
    }
}

/// Default token estimate using WhitespaceTokenizer. Callers with a real
/// tokenizer should use `tokenizer.count_tokens()` directly instead.
pub fn estimate_tokens(text: &str) -> usize {
    WhitespaceTokenizer.count_tokens(text)
}
