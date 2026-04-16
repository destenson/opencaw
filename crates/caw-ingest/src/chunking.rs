use caw_core::{ContentKind, Tokenizer};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Files over this token count get chunked
    pub token_threshold: usize,
    /// Target size for each chunk in tokens
    pub target_chunk_tokens: usize,
    /// Overlap between adjacent chunks in tokens (for context continuity)
    pub overlap_tokens: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            token_threshold: 2000,
            target_chunk_tokens: 800,
            overlap_tokens: 100,
        }
    }
}

/// A chunk of a larger document, with its position and the outline entries it contains.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub index: usize,
    pub total_chunks: usize,
    /// Outline entries from the parent document that fall within this chunk
    pub outline_entries: Vec<String>,
}

/// Split a document into chunks based on its content kind.
/// Returns a single chunk wrapping the whole content if the document is
/// below the token threshold.
pub fn chunk_document(
    content: &str,
    kind: ContentKind,
    outline: &[String],
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<Chunk> {
    let token_count = tokenizer.count_tokens(content);
    if token_count <= config.token_threshold {
        return vec![Chunk {
            content: content.to_string(),
            index: 0,
            total_chunks: 1,
            outline_entries: outline.to_vec(),
        }];
    }

    let raw_chunks = match kind {
        ContentKind::Code => chunk_code(content, config, tokenizer),
        ContentKind::Markdown => chunk_markdown(content, config, tokenizer),
        _ => chunk_by_paragraphs(content, config, tokenizer),
    };

    let total = raw_chunks.len();
    raw_chunks
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            let entries = outline_entries_for_chunk(&text, outline);
            Chunk {
                content: text,
                index: i,
                total_chunks: total,
                outline_entries: entries,
            }
        })
        .collect()
}

/// Build a positional summary like "Chunk 2/5 of path: contains `fn foo` and `struct Bar`"
pub fn chunk_summary(path: &str, chunk: &Chunk) -> String {
    if chunk.total_chunks == 1 {
        return String::new();
    }

    let entries_desc = if chunk.outline_entries.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = chunk
            .outline_entries
            .iter()
            .take(4)
            .map(|e| e.as_str())
            .collect();
        format!(": contains {}", names.join(", "))
    };

    format!(
        "Chunk {}/{} of {}{}",
        chunk.index + 1,
        chunk.total_chunks,
        path,
        entries_desc
    )
}

/// For code: split at function/struct/impl boundaries detected by outline entries,
/// falling back to blank-line boundaries.
fn chunk_code(
    content: &str,
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();

    let mut boundary_lines: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if is_code_boundary(trimmed) {
            boundary_lines.push(i);
        }
    }

    if boundary_lines.len() >= 2 {
        split_at_boundaries(&lines, &boundary_lines, config, tokenizer)
    } else {
        let blank_lines: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.trim().is_empty())
            .map(|(i, _)| i)
            .collect();
        if blank_lines.is_empty() {
            split_by_token_count(content, config, tokenizer)
        } else {
            split_at_boundaries(&lines, &blank_lines, config, tokenizer)
        }
    }
}

/// For markdown: split at heading boundaries (## or ### level).
fn chunk_markdown(
    content: &str,
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let heading_lines: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim_start();
            t.starts_with("## ") || t.starts_with("### ")
        })
        .map(|(i, _)| i)
        .collect();

    if heading_lines.is_empty() {
        return split_by_token_count(content, config, tokenizer);
    }

    split_at_boundaries(&lines, &heading_lines, config, tokenizer)
}

/// For plain text and other kinds: split at paragraph boundaries (blank lines),
/// falling back to raw token-count splitting.
fn chunk_by_paragraphs(
    content: &str,
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let blank_lines: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim().is_empty())
        .map(|(i, _)| i)
        .collect();

    if blank_lines.is_empty() {
        return split_by_token_count(content, config, tokenizer);
    }

    split_at_boundaries(&lines, &blank_lines, config, tokenizer)
}

/// Split lines into chunks at the given boundary indices, merging small
/// adjacent sections to stay near the target chunk size, and applying overlap.
fn split_at_boundaries(
    lines: &[&str],
    boundaries: &[usize],
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<String> {
    let mut section_starts: Vec<usize> = vec![0];
    for &b in boundaries {
        if b > 0 && b < lines.len() {
            section_starts.push(b);
        }
    }
    section_starts.dedup();

    let mut sections: Vec<(usize, usize)> = Vec::new();
    for i in 0..section_starts.len() {
        let start = section_starts[i];
        let end = if i + 1 < section_starts.len() {
            section_starts[i + 1]
        } else {
            lines.len()
        };
        sections.push((start, end));
    }

    // Merge small sections until they approach the target size
    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_tokens = 0usize;

    for (start, end) in &sections {
        let section_text = lines[*start..*end].join("\n");
        let section_tokens = tokenizer.count_tokens(&section_text);

        if cur_tokens > 0 && cur_tokens + section_tokens > config.target_chunk_tokens {
            merged.push((cur_start, *start));
            cur_start = *start;
            cur_tokens = section_tokens;
        } else {
            if cur_tokens == 0 {
                cur_start = *start;
            }
            cur_tokens += section_tokens;
        }
    }
    if let Some(&(_, last_end)) = sections.last() {
        merged.push((cur_start, last_end));
    }

    let overlap_lines = token_count_to_line_estimate(config.overlap_tokens);

    merged
        .iter()
        .map(|&(start, end)| {
            let overlap_start = start.saturating_sub(overlap_lines);
            let effective_start = if start == 0 { 0 } else { overlap_start };
            lines[effective_start..end].join("\n")
        })
        .collect()
}

/// Last-resort splitting: cut at roughly target_chunk_tokens boundaries.
fn split_by_token_count(
    content: &str,
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<String> {
    let words: Vec<&str> = content.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < words.len() {
        // Binary-search for how many words fit in the target token budget.
        // The tokenizer may not be 1-word-per-token, so we probe.
        let mut end = (pos + config.target_chunk_tokens).min(words.len());
        let candidate = words[pos..end].join(" ");
        let tokens = tokenizer.count_tokens(&candidate);

        if tokens > config.target_chunk_tokens && end > pos + 1 {
            // Overshoot — shrink until we fit
            while end > pos + 1 {
                end -= 1;
                let candidate = words[pos..end].join(" ");
                if tokenizer.count_tokens(&candidate) <= config.target_chunk_tokens {
                    break;
                }
            }
        }

        chunks.push(words[pos..end].join(" "));
        if end >= words.len() {
            break;
        }
        pos = end.saturating_sub(config.overlap_tokens);
    }

    chunks
}

fn is_code_boundary(trimmed: &str) -> bool {
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("pub trait ")
        || trimmed.starts_with("trait ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("pub mod ")
        || trimmed.starts_with("mod ")
        || trimmed.starts_with("def ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("function ")
        || trimmed.starts_with("export function ")
        || trimmed.starts_with("export class ")
        || trimmed.starts_with("export interface ")
        || trimmed.starts_with("export type ")
}

/// Check which outline entries appear in a chunk's text.
fn outline_entries_for_chunk(chunk_text: &str, outline: &[String]) -> Vec<String> {
    outline
        .iter()
        .filter(|entry| chunk_text.contains(entry.as_str()))
        .cloned()
        .collect()
}

/// Rough conversion: assume ~8 tokens per line on average for code/prose.
fn token_count_to_line_estimate(tokens: usize) -> usize {
    (tokens / 8).max(1)
}
