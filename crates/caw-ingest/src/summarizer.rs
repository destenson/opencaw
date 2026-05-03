use caw_core::{CawResult, CompletionRequest, CompletionResponse, ContentKind, ModelAdapter};

pub trait Summarizer: Send + Sync {
    fn summarize(
        &self,
        path: &str,
        content: &str,
        kind: ContentKind,
        outline: &[String],
    ) -> CawResult<String>;
}

/// Extracts summaries using heuristics — first heading + paragraph for markdown,
/// symbol names for code. No LLM calls, always available.
pub struct DeterministicSummarizer;

impl Summarizer for DeterministicSummarizer {
    fn summarize(
        &self,
        _path: &str,
        content: &str,
        kind: ContentKind,
        outline: &[String],
    ) -> CawResult<String> {
        Ok(deterministic_summarize(kind, content, outline))
    }
}

/// Uses a ModelAdapter to generate richer summaries. Sends a truncated
/// content window to keep the summarization call cheap.
pub struct LlmSummarizer {
    adapter: Box<dyn ModelAdapter + Send + Sync>,
    /// Max chars of content to send to the model. Keeps summarization
    /// calls fast and cheap even for large files.
    max_content_chars: usize,
}

impl LlmSummarizer {
    pub fn with_adapter(adapter: Box<dyn ModelAdapter + Send + Sync>) -> Self {
        Self {
            adapter,
            max_content_chars: 2000,
        }
    }

    pub fn max_content_chars(mut self, max: usize) -> Self {
        self.max_content_chars = max;
        self
    }
}

impl Summarizer for LlmSummarizer {
    fn summarize(
        &self,
        path: &str,
        content: &str,
        kind: ContentKind,
        outline: &[String],
    ) -> CawResult<String> {
        let truncated = truncate_chars(content, self.max_content_chars);
        let kind_str = content_kind_label(kind);
        let outline_str = if outline.is_empty() {
            "(none)".to_string()
        } else {
            outline.join(", ")
        };

        let system = concat!(
            "You generate concise file summaries for a context management system. ",
            "Each summary will be shown to an LLM as a stub — it decides whether to ",
            "load the full file based on this summary. Be specific about what the file ",
            "contains and its purpose. 1-3 sentences max.",
        )
        .to_string();

        let user = format!(
            "Summarize this file:\nPath: {path}\nType: {kind_str}\nOutline: {outline_str}\nContent (first {} chars):\n{truncated}",
            self.max_content_chars,
        );

        let resp: CompletionResponse = self.adapter.complete(CompletionRequest {
            system,
            user,
            workspace_fragments: Vec::new(),
            workspace_guidance: Vec::new(),
        })?;

        let answer = resp.answer.trim().to_string();
        if answer.is_empty() {
            // Fall back to deterministic if the model returns nothing useful
            return Ok(deterministic_summarize(kind, content, outline));
        }
        Ok(answer)
    }
}

fn content_kind_label(kind: ContentKind) -> &'static str {
    match kind {
        ContentKind::Markdown => "markdown",
        ContentKind::Code => "code",
        ContentKind::PlainText => "plain text",
        ContentKind::Tabular => "tabular data",
        ContentKind::Transcript => "transcript",
        ContentKind::Other => "other",
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

// ── Deterministic summarization (extracted from the old free functions) ──

fn deterministic_summarize(kind: ContentKind, content: &str, outline: &[String]) -> String {
    match kind {
        ContentKind::Markdown => summarize_markdown(content, outline),
        ContentKind::Code => summarize_code(content, outline),
        _ => summarize_plaintext(content),
    }
}

fn summarize_markdown(content: &str, outline: &[String]) -> String {
    let mut title = String::new();
    let mut first_para = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !first_para.is_empty() {
                break;
            }
            continue;
        }

        if trimmed.starts_with('#') {
            if title.is_empty() {
                title = trimmed.trim_start_matches('#').trim().to_string();
            }
            continue;
        }

        if !title.is_empty() && first_para.is_empty() {
            first_para = trimmed.to_string();
        }
    }

    if !title.is_empty() && !first_para.is_empty() {
        let combined = format!("{}: {}", title, first_para);
        truncate_str(&combined, 200)
    } else if !title.is_empty() {
        if !outline.is_empty() {
            format!("{}; sections: {}", title, outline.join(", "))
        } else {
            title
        }
    } else {
        summarize_plaintext(content)
    }
}

fn summarize_code(content: &str, outline: &[String]) -> String {
    if outline.is_empty() {
        return summarize_plaintext(content);
    }

    let preview: Vec<&String> = outline.iter().take(5).collect();
    let suffix = if outline.len() > 5 {
        format!(" (+{} more)", outline.len() - 5)
    } else {
        String::new()
    };

    format!(
        "Defines: {}{}",
        preview
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        suffix
    )
}

fn summarize_plaintext(content: &str) -> String {
    let first_line = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(empty)");
    truncate_str(first_line, 200)
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 3).collect();
        format!("{}...", truncated)
    }
}
