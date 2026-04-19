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
    /// Target chunk size in characters. Used for greedy merging of sections
    /// without running the BPE tokenizer inside the hot loop. ~3 chars per
    /// cl100k token on English prose, so this approximates
    /// `target_chunk_tokens * 3` when the caller doesn't override it.
    pub target_chunk_chars: usize,
    /// Hard cap on a single emitted chunk's characters. Any natural section
    /// above the cap is force-split at character boundaries. Prevents one
    /// giant section (e.g. a monster changelog entry) from producing a
    /// multi-MB "chunk" that would stall tokenization and waste embedding.
    pub max_chunk_chars: usize,
    /// Overlap between adjacent chunks in characters (prepended to the
    /// start of each chunk after the first, for context continuity).
    pub overlap_chars: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            token_threshold: 2000,
            target_chunk_tokens: 800,
            overlap_tokens: 100,
            // Greedy-chunking lives in char space. Defaults derived from the
            // token defaults via a ~3 chars/token rule of thumb. Precision
            // is not the goal; staying inside the model's 512-token window
            // is, and max_chunk_chars keeps us well under that even on
            // token-dense inputs.
            target_chunk_chars: 2_400,
            max_chunk_chars: 6_000,
            overlap_chars: 300,
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
    /// Token count for this chunk's content, passed through from chunking so
    /// downstream code (see `IngestionPipeline::ingest`) can reuse it instead
    /// of re-running the tokenizer. Approximate for the char-heuristic fast
    /// path (tiny files), exact for chunked paths.
    pub token_count: usize,
}

/// Split a document into chunks based on its content kind.
/// Returns a single chunk wrapping the whole content if the document is
/// below the token threshold.
/// Cheap char-based upper bound on token count. For English-like text, 1
/// cl100k token is ~3.5–4 chars. A /3 estimate is conservative (over-counts
/// tokens) so we never skip chunking when it's actually needed. Orders of
/// magnitude faster than running the BPE tokenizer.
fn cheap_token_upper_bound(content: &str) -> usize {
    content.len() / 3
}

pub fn chunk_document(
    content: &str,
    kind: ContentKind,
    outline: &[String],
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<Chunk> {
    // Below threshold? Ship a single chunk and skip the BPE pass entirely.
    // Precision is not the goal — staying off the tokenizer on small files
    // is. This is the dominant throughput win on corpora full of tiny docs.
    let upper_bound = cheap_token_upper_bound(content);
    if upper_bound <= config.token_threshold {
        return vec![Chunk {
            content: content.to_string(),
            index: 0,
            total_chunks: 1,
            outline_entries: outline.to_vec(),
            token_count: upper_bound,
        }];
    }

    // Over the cheap threshold: chunk. We do NOT tokenize the whole file
    // first — on a 10MB file that alone would dominate wall time. One
    // forward pass over `content` slicing at `\n` boundaries near each
    // char-budget target is enough; structural boundaries (headings, fn
    // defs) would be nicer but aren't worth quadratic section enumeration
    // on corpora where some files are 10MB+.
    //
    // `kind` is intentionally unused here for now — all kinds chunk the
    // same way. Kept in the signature so a future structural variant can
    // branch on it without changing callers.
    let _ = kind;
    let raw_chunks = chunk_by_lines(content, config);

    let total = raw_chunks.len();
    raw_chunks
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            let entries = outline_entries_for_chunk(&text, outline);
            // Per-chunk token_count is still a real tokenize, but each chunk
            // is bounded by `max_chunk_chars` so this is O(cap) not O(file).
            let token_count = tokenizer.count_tokens(&text);
            Chunk {
                content: text,
                index: i,
                total_chunks: total,
                outline_entries: entries,
                token_count,
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

/// Walk `content` once, emitting chunks that are approximately
/// `target_chunk_chars` characters long and always line-aligned.
///
/// Algorithm per chunk:
/// 1. Target cut point is `start + target_chunk_chars`.
/// 2. Scan forward from there for the next `\n`, bounded by
///    `start + max_chunk_chars`.
/// 3. If found, cut just after the `\n`.
/// 4. If not, scan backward from the hard cap for a `\n` and cut there.
/// 5. If there's no `\n` at all in the window (minified JSON, ASCII art),
///    snap the cut to the nearest UTF-8 char boundary before the hard cap
///    and emit that. Rare.
///
/// `\n` is ASCII so byte offsets are always char boundaries when we cut
/// on one. The only UTF-8-sensitive path is the no-newline fallback.
///
/// Linear in content length. No per-section tokenization, no vector of
/// lines, no quadratic section joining. This is the hot path for
/// everything the pipeline ingests.
fn chunk_by_lines(content: &str, config: &ChunkingConfig) -> Vec<String> {
    let target = config.target_chunk_chars.max(1);
    let cap = config.max_chunk_chars.max(target);
    let overlap = config.overlap_chars.min(cap.saturating_sub(1));

    if content.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0usize;
    let total = content.len();

    while start < total {
        // Whole remainder fits — ship and done.
        if total - start <= cap {
            let mut piece = String::new();
            prepend_line_overlap(&mut piece, &content[..start], overlap, chunks.is_empty());
            piece.push_str(&content[start..]);
            chunks.push(piece);
            break;
        }

        let ideal = start + target;
        let hard = (start + cap).min(total);

        // Prefer the first `\n` at-or-after the ideal cut, within the cap.
        let forward_hit = content[ideal..hard].find('\n');
        let cut = if let Some(rel) = forward_hit {
            ideal + rel + 1 // include the newline
        } else if let Some(pos) = content[start..hard].rfind('\n') {
            // Fall back to the last `\n` before the hard cap.
            start + pos + 1
        } else {
            // No line breaks in the entire window. Snap to the previous
            // UTF-8 char boundary at or before `hard`.
            let mut c = hard;
            while c > start && !content.is_char_boundary(c) {
                c -= 1;
            }
            c.max(start + 1)
        };

        let mut piece = String::new();
        prepend_line_overlap(&mut piece, &content[..start], overlap, chunks.is_empty());
        piece.push_str(&content[start..cut]);
        chunks.push(piece);
        start = cut;
    }

    chunks
}

/// Prepend a line-aligned overlap prefix drawn from the tail of
/// `preceding`. `is_first` skips prepending so the first chunk starts at
/// the file's actual beginning. The overlap is sized in characters and
/// snapped back to a line boundary — readers never see a chunk start
/// in the middle of a line.
fn prepend_line_overlap(
    dst: &mut String,
    preceding: &str,
    overlap_chars: usize,
    is_first: bool,
) {
    if is_first || overlap_chars == 0 || preceding.is_empty() {
        return;
    }
    // Walk backward from the end of `preceding` for ~overlap_chars bytes,
    // then snap forward to just after the nearest earlier `\n`.
    let start_byte = preceding.len().saturating_sub(overlap_chars);
    let snap = preceding[..start_byte]
        .rfind('\n')
        .map(|p| p + 1)
        .unwrap_or(0);
    // If the snap goes way further back than we asked for, prefer the
    // requested window (less context is better than loading 500KB overlap).
    let final_start = snap.max(start_byte.saturating_sub(overlap_chars));
    // Always safe: rfind('\n') returns a byte offset at a char boundary,
    // and `final_start` came from start_byte which we already aligned
    // forward by chars if needed below.
    let mut aligned = final_start;
    while aligned < preceding.len() && !preceding.is_char_boundary(aligned) {
        aligned += 1;
    }
    dst.push_str(&preceding[aligned..]);
    if !dst.ends_with('\n') {
        dst.push('\n');
    }
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
