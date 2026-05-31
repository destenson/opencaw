use crate::tree_sitter_outline::ItemSpan;
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
            // Anything that cheap-estimates above this gets chunked. Leave
            // some headroom over target so tiny-over-target files aren't
            // needlessly split.
            token_threshold: 500,
            // Target body size leaves ~100 token slack for the header prefix
            // (path + summary) prepended at embed time and for the char→token
            // estimator's error margin. Max BGE context is 512.
            target_chunk_tokens: 400,
            overlap_tokens: 80,
            // Char-space targets derived from the token targets via a
            // conservative ~3.5 chars/token average. Dense content (code,
            // URLs, minified JSON) can run 2–3 chars/token, so the
            // post-chunking token-count safety pass in `chunk_document`
            // re-splits anything that still exceeds MAX_SAFE_BODY_TOKENS.
            target_chunk_chars: 1_400,
            max_chunk_chars: 1_800,
            overlap_chars: 280,
        }
    }
}

/// Hard ceiling on body token count after all splits. Leaves a small margin
/// below `MAX_SEQ_LEN` (512) for the special tokens the tokenizer adds
/// ([CLS], [SEP]) and for the query prefix prepended at retrieval time on
/// asymmetric models. 500 keeps ~12 tokens of headroom before truncation
/// starts silently dropping content.
pub const MAX_SAFE_BODY_TOKENS: usize = 500;

/// A chunk of a larger document, with its position and the outline entries it contains.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Text fed to the embedder. Includes the overlap prefix drawn from the
    /// tail of the previous chunk (for embedding context continuity) followed
    /// by this chunk's body. NOT stored verbatim — the index records only
    /// `body_start`/`body_end` and re-reads the body from disk on demand.
    pub content: String,
    pub index: usize,
    pub total_chunks: usize,
    /// Byte offset of this chunk's body in the source document. Excludes the
    /// overlap prefix. Chunks tile the source file — adjacent chunks have
    /// `prev.body_end == next.body_start`, with no duplication.
    pub body_start: u64,
    /// Byte offset of the end of this chunk's body in the source document
    /// (exclusive). `body_end - body_start` is the body length in bytes.
    pub body_end: u64,
    /// Outline entries from the parent document that fall within this chunk
    pub outline_entries: Vec<String>,
    /// Token count for `content` (overlap + body), not just the body. Passed
    /// through from chunking so downstream code can reuse it instead of
    /// re-running the tokenizer. Approximate for the char-heuristic fast
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
    item_spans: Option<&[ItemSpan]>,
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<Chunk> {
    // Below threshold? Ship a single chunk and skip the BPE pass entirely.
    // Precision is not the goal — staying off the tokenizer on small files
    // is. This is the dominant throughput win on corpora full of tiny docs.
    // Small files are already concentrated, so the structure-aware header
    // (which only earns its keep when one item's signal is diluted across a
    // large multi-item chunk) is not applied here.
    let upper_bound = cheap_token_upper_bound(content);
    if upper_bound <= config.token_threshold {
        return vec![Chunk {
            content: content.to_string(),
            index: 0,
            total_chunks: 1,
            body_start: 0,
            body_end: content.len() as u64,
            outline_entries: outline.to_vec(),
            token_count: upper_bound,
        }];
    }

    let _ = kind;

    // `(body_start, body_end, embed_text, token_count)` either way. When
    // tree-sitter item spans are available we tile on item boundaries and
    // front-load each chunk with its items' signatures + docstrings so a
    // definitional sentence isn't averaged into the noise of a same-domain
    // corpus. Otherwise we fall back to the line-based char-budget splitter.
    let safe_chunks: Vec<(u64, u64, String, usize)> = match item_spans {
        Some(spans) if !spans.is_empty() => chunk_by_items(content, spans, config, tokenizer),
        _ => {
            // Over the cheap threshold: chunk by lines. We do NOT tokenize the
            // whole file first — on a 10MB file that alone would dominate wall
            // time. One forward pass slicing at `\n` near each char-budget
            // target is enough.
            let raw_chunks = chunk_by_lines(content, config);

            // Safety pass: char-based chunking under-counts tokens on dense
            // inputs (code, URLs, minified data) where chars/token drops below
            // the 3.5 avg the defaults assume. Any chunk whose actual
            // token_count exceeds `MAX_SAFE_BODY_TOKENS` gets split further
            // with tightened bounds, else the embedder silently truncates it.
            let mut safe = Vec::with_capacity(raw_chunks.len());
            for (body_start, body_end, text) in raw_chunks {
                let token_count = tokenizer.count_tokens(&text);
                if token_count <= MAX_SAFE_BODY_TOKENS {
                    safe.push((body_start, body_end, text, token_count));
                    continue;
                }
                safe.extend(split_oversized(
                    content,
                    body_start as usize,
                    body_end as usize,
                    config,
                    tokenizer,
                    0,
                ));
            }
            safe
        }
    };

    let total = safe_chunks.len();
    safe_chunks
        .into_iter()
        .enumerate()
        .map(|(i, (body_start, body_end, text, token_count))| {
            let entries = outline_entries_for_chunk(&text, outline);
            Chunk {
                content: text,
                index: i,
                total_chunks: total,
                body_start,
                body_end,
                outline_entries: entries,
                token_count,
            }
        })
        .collect()
}

/// Recursively split a body range that tokenized over `MAX_SAFE_BODY_TOKENS`.
/// Uses the same line-aware algorithm with halved char bounds. Bottoms out
/// at `depth >= 3` by emitting whatever it has — after 8x tightening a chunk
/// that still over-runs is pathological (base64-encoded binary blobs etc.)
/// and the embedder's truncation becomes the least-bad option.
fn split_oversized(
    content: &str,
    body_start: usize,
    body_end: usize,
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
    depth: u32,
) -> Vec<(u64, u64, String, usize)> {
    let slice = &content[body_start..body_end];
    if depth >= 3 || slice.is_empty() {
        let count = tokenizer.count_tokens(slice);
        return vec![(body_start as u64, body_end as u64, slice.to_string(), count)];
    }
    let tightened = ChunkingConfig {
        target_chunk_chars: (config.target_chunk_chars / 2).max(200),
        max_chunk_chars: (config.max_chunk_chars / 2).max(300),
        // Overlap stays a minor fraction; shrinking it proportionally keeps
        // ratio reasonable without turning into a scanline.
        overlap_chars: (config.overlap_chars / 2).max(80),
        ..config.clone()
    };
    let sub = chunk_by_lines(slice, &tightened);
    let mut out = Vec::with_capacity(sub.len());
    for (rel_start, rel_end, text) in sub {
        let abs_start = body_start + rel_start as usize;
        let abs_end = body_start + rel_end as usize;
        let count = tokenizer.count_tokens(&text);
        if count <= MAX_SAFE_BODY_TOKENS {
            out.push((abs_start as u64, abs_end as u64, text, count));
        } else {
            out.extend(split_oversized(
                content,
                abs_start,
                abs_end,
                &tightened,
                tokenizer,
                depth + 1,
            ));
        }
    }
    out
}

/// Tile `content` on tree-sitter item boundaries. Each chunk covers
/// `[start_i, start_{i+1})`, so the file is tiled without gaps: leading `use`
/// declarations attach to the first chunk, blank lines between items to the
/// preceding one. Adjacent small items merge up to the char budget; an item
/// past the hard cap is line-split via `chunk_by_lines`.
///
/// Each chunk's embed text is prefixed with the signatures + docstrings of the
/// items that begin in it. That prefix is what the embedder and BM25 see; the
/// returned byte range still points at the verbatim body, so served content is
/// unchanged. Returns `(body_start, body_end, embed_text, token_count)`.
fn chunk_by_items(
    content: &str,
    spans: &[ItemSpan],
    config: &ChunkingConfig,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<(u64, u64, String, usize)> {
    let target = config.target_chunk_chars.max(1);
    let cap = config.max_chunk_chars.max(target);
    let total = content.len();

    // Boundaries: open a new chunk at an item start once the span since the
    // current chunk start would exceed `target`. First boundary is 0 so the
    // file preamble (imports) rides with the first item's chunk.
    let mut boundaries: Vec<usize> = vec![0];
    let mut cur_start = 0usize;
    for sp in spans {
        let end = sp.end.min(total);
        if end.saturating_sub(cur_start) > target && sp.start > cur_start {
            boundaries.push(sp.start);
            cur_start = sp.start;
        }
    }
    boundaries.push(total);
    boundaries.dedup();

    let mut out: Vec<(u64, u64, String, usize)> = Vec::with_capacity(boundaries.len());
    for w in boundaries.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a >= b {
            continue;
        }
        let header = build_item_header(spans, a, b);
        let body = &content[a..b];
        let prepend = |text: &str| -> String {
            if header.is_empty() {
                text.to_string()
            } else {
                format!("{header}\n\n{text}")
            }
        };

        // Fast path: tile within the char cap and the prefixed text within the
        // embedder's token ceiling becomes one chunk.
        if b - a <= cap {
            let embed = prepend(body);
            let count = tokenizer.count_tokens(&embed);
            if count <= MAX_SAFE_BODY_TOKENS {
                out.push((a as u64, b as u64, embed, count));
                continue;
            }
        }

        // Oversized item (or a tile that tokenizes hot): line-split the body,
        // keeping the structural header on the first sub-chunk only.
        let subs = chunk_by_lines(body, config);
        for (i, (rel_start, rel_end, text)) in subs.into_iter().enumerate() {
            let abs_start = a + rel_start as usize;
            let abs_end = a + rel_end as usize;
            let embed = if i == 0 { prepend(&text) } else { text };
            let count = tokenizer.count_tokens(&embed);
            if count <= MAX_SAFE_BODY_TOKENS {
                out.push((abs_start as u64, abs_end as u64, embed, count));
            } else {
                out.extend(split_oversized(content, abs_start, abs_end, config, tokenizer, 0));
            }
        }
    }

    out
}

/// Join signatures (and docstrings) of items beginning in `[a, b)` into a
/// concentrated prefix, capped so a tile full of tiny items can't crowd the
/// body out from under the embedder's token ceiling.
fn build_item_header(spans: &[ItemSpan], a: usize, b: usize) -> String {
    const MAX_HEADER_CHARS: usize = 400;
    let mut header = String::new();
    for sp in spans.iter().filter(|sp| sp.start >= a && sp.start < b) {
        let line = if sp.doc.is_empty() {
            sp.signature.clone()
        } else {
            format!("{} — {}", sp.signature, sp.doc)
        };
        if !header.is_empty() && header.len() + line.len() > MAX_HEADER_CHARS {
            break;
        }
        if !header.is_empty() {
            header.push('\n');
        }
        header.push_str(&line);
    }
    header
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
/// Returns `(body_start, body_end, embed_text)` per chunk. `body_start`/
/// `body_end` are byte offsets into `content` (tile the file without gaps or
/// overlap). `embed_text` is what gets fed to the embedder: the overlap
/// prefix (from the tail of the previous chunk) concatenated with the body.
fn chunk_by_lines(content: &str, config: &ChunkingConfig) -> Vec<(u64, u64, String)> {
    let target = config.target_chunk_chars.max(1);
    let cap = config.max_chunk_chars.max(target);
    let overlap = config.overlap_chars.min(cap.saturating_sub(1));

    if content.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<(u64, u64, String)> = Vec::new();
    let mut start = 0usize;
    let total = content.len();

    while start < total {
        // Whole remainder fits — ship and done.
        if total - start <= cap {
            let mut piece = String::new();
            prepend_line_overlap(&mut piece, &content[..start], overlap, chunks.is_empty());
            piece.push_str(&content[start..]);
            chunks.push((start as u64, total as u64, piece));
            break;
        }

        // `target`/`cap` are nominally char counts but we apply them as
        // byte offsets for O(1) slicing. UTF-8 multi-byte chars mean those
        // byte positions may land mid-codepoint, which panics on slice.
        // Snap to the previous char boundary before slicing. Over-counts
        // tokens slightly for multi-byte content; acceptable.
        let ideal = floor_char_boundary(content, (start + target).min(total));
        let hard = floor_char_boundary(content, (start + cap).min(total));

        // Prefer the first `\n` at-or-after the ideal cut, within the cap.
        let forward_hit = content[ideal..hard].find('\n');
        let cut = if let Some(rel) = forward_hit {
            ideal + rel + 1 // include the newline
        } else if let Some(pos) = content[start..hard].rfind('\n') {
            // Fall back to the last `\n` before the hard cap.
            start + pos + 1
        } else {
            // No line breaks in the entire window. `hard` is already at a
            // char boundary, so slicing is safe. Force forward progress.
            hard.max(start + 1)
        };

        let mut piece = String::new();
        prepend_line_overlap(&mut piece, &content[..start], overlap, chunks.is_empty());
        piece.push_str(&content[start..cut]);
        chunks.push((start as u64, cut as u64, piece));
        start = cut;
    }

    chunks
}

/// Prepend a line-aligned overlap prefix drawn from the tail of
/// `preceding`. `is_first` skips prepending so the first chunk starts at
/// the file's actual beginning. The overlap is sized in characters and
/// snapped back to a line boundary — readers never see a chunk start
/// in the middle of a line.
fn prepend_line_overlap(dst: &mut String, preceding: &str, overlap_chars: usize, is_first: bool) {
    if is_first || overlap_chars == 0 || preceding.is_empty() {
        return;
    }
    // Snap the window start to a char boundary before slicing — the raw
    // byte offset can land mid-codepoint on UTF-8 content. Then snap
    // forward to just after the nearest earlier `\n` so the overlap
    // starts at a line boundary.
    let start_byte = floor_char_boundary(preceding, preceding.len().saturating_sub(overlap_chars));
    let snap = preceding[..start_byte]
        .rfind('\n')
        .map(|p| p + 1)
        .unwrap_or(0);
    let final_start = snap.max(floor_char_boundary(
        preceding,
        start_byte.saturating_sub(overlap_chars),
    ));
    dst.push_str(&preceding[final_start..]);
    if !dst.ends_with('\n') {
        dst.push('\n');
    }
}

/// Largest byte offset `<= idx` that falls on a UTF-8 char boundary. Used
/// to make byte-offset arithmetic on char-count budgets safe to slice.
pub(crate) fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Check which outline entries appear in a chunk's text.
fn outline_entries_for_chunk(chunk_text: &str, outline: &[String]) -> Vec<String> {
    outline
        .iter()
        .filter(|entry| {
            // tree-sitter may produce multi-line signatures (wrapped arg lists).
            // A full-string contains check fails when a chunk boundary falls inside
            // the signature — the chunk starts at a wrapped arg line, not the `fn`
            // keyword. Matching the first line correctly attributes the chunk that
            // opens the function even when the closing paren lands in the next chunk.
            let first_line = entry.lines().next().unwrap_or(entry.as_str());
            chunk_text.contains(first_line)
        })
        .cloned()
        .collect()
}
