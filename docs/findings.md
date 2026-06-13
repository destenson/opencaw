# QA Findings — Consolidated (Loops 0001–0021)

Distilled from `qa/recommendations/0001.md` through `0020.md` before artifact
cleanup.

---

## Bugs — Fixed

| Bug | Description                                                                                               | First Seen |
| --- | --------------------------------------------------------------------------------------------------------- | ---------- |
| B6  | Model emits fake `[recalled from ...]` markers in output, mimicking context injection format              | 0006       |
| B11 | Chat-template tokens (`<                                                                                  | im_end     |
| B12 | `<think>…</think>` block stored as the answer instead of being stripped                                   | 0013       |
| B13 | `truncate_at_chat_boundary` left trailing `<                                                              | im_end     |
| B14 | Empty `<think></think>` block (whitespace-only) caused false-positive degenerate                          | 0015       |
| B15 | Session log not written when degenerate error propagated before `write_turn`                              | 0015/0016  |
| B16 | Valid prefix text discarded on degenerate; `String::new()` stored as answer                               | 0016       |
| B17 | Blank answer after `split_thinking` passed through degeneracy check (len < 20 short-circuit)              | 0016       |
| B18 | Degenerate on one turn aborted the entire session instead of continuing to the next question              | 0018       |
| B19 | Degeneracy detection fired false-positives on Qwen3's fake recall blocks (trigram collapse)               | 0020       |
| B20 | Multi-pass refinement loop tracked last completion, not best — overwrote good answer with worse later one | 0020       |
| B21 | Truncated degenerate prefix (~120 chars) injected verbatim as prior-turn history into subsequent sessions | 0020       |
| B22 | `extract_field` bled assistant content into user field in session history                                 | 0020       |

---

## Bugs — Still Open

### B0: First user turn sometimes missing from session log (high)

Session log shows turn 2 as the first entry; the prompt file for turn 1 exists.
Turn 1 was processed but not written. Seen in loop 0008.

### B1: Multi-pass recall corrupts fragment source paths (critical)

Source paths like `[recalled from max_recall_iterations = 3,]` and truncated
`[recalled from ./cr` appear in prompts. Root cause: the workspace formatter
doesn't guard against writing a `[recalled from ...]` header when the path
string is malformed or truncated. Additionally, fragments from the same large
file (e.g. `dynamic.rs`, 34–39 chunks) are re-admitted on every thinking-trace
iteration without deduplication, producing 30+ overlapping windows in a single
prompt. Seen in loops 0003, 0005, 0011, 0012.

### B2: Session log truncated — most turns never written (critical)

Sessions with 13–18 prompt files produce only 1–3 turns in the session `.md`
log. The writer doesn't flush per-turn; it exits silently (on error or session
end) before writing later turns. Makes QA analysis from session logs
systematically incomplete. Active in every loop from 0003–0015. The per-turn
flush fix from 0003 did not fully address it — the degenerate-output error path
exits before calling `write_turn` for the failed turn.

### B3: `.caw/` directory not consistently skipped during indexing (critical)

Session log files (`.caw/session-*.md`) end up in the retrieval index. Model
responses from prior sessions become retrieval-ranked "evidence" for future
queries — a direct feedback loop where hallucinations in session N propagate
into session N+1 with retrieval authority. The `should_skip` predicate exists
but something in the build path bypasses it. Confirmed active in loops
0003–0006; fix was attempted but behavior recurred.

### B4: Search-candidates mode non-functional in a single-turn QA harness (high)

The system prompt tells the model "mention files you want loaded" but there's no
mechanism to honor that request. The model uses discovery metadata as its only
evidence, producing structurally plausible but factually ungrounded answers.
This is an architectural gap — search-candidates is a dialog protocol that
requires a tool-call loop to be useful. In a one-shot QA setting it is actively
worse than loading the top-k candidates directly. Seen in every loop where
`wants_explanation=true`.

### B7/B10: Prior session model responses outcompete workspace stubs (high, partially mitigated)

Model responses from prior sessions are stored in the in-memory embedding pool
and score higher than workspace stubs because they contain all relevant
vocabulary in fluent prose. A `session_history_budget_fraction = 0.25` cap was
applied and helped, but doesn't address the root cause. Model response text
either shouldn't be embedded at all, or needs a hard score ceiling that can't
outrank the top workspace stub. In severe cases (loop 0013), session 3 retrieved
session 2's model output verbatim rather than querying the workspace. Sessions
within the same 30-minute window compound: each session's verbose response
becomes available to the next.

### B9: Consolidation notes nest recursively (medium)

Each eviction appends the prior note's full text as the "topic" of the new note.
After a few eviction cycles the consolidation header for a frequently-evicted
stub is larger than its content. The cap of 2 displayed notes (with "N older
omitted") preserves display budget but loses historical signal. Seen in
loop 0011.

---

## Retrieval Quality — Fixed

### Embedding text is stub metadata, not chunk content — FIXED

`SemanticRetriever::insert` now embeds chunk content as the primary signal
(`path + summary + content`), falling back to metadata-only when content is
empty. The CLI ingest path (`embed_and_insert_batch`) embeds `path + chunk_content`
directly and bypasses the retriever's insert entirely for batched ingest.
The `_content` parameter discard is gone. First identified loop 0001, fixed
alongside the HNSW perf work (commit `d29c679`).

### Outline symbols double-counted in embedding string — FIXED

The CLI ingest path embeds `path + chunk_content` with no outline appended;
`SemanticRetriever::insert` uses `path + summary + content` with no separate
`outline.join()`. The duplication that crowded out tokens in BGE-small's
512-token context is gone. Fixed in `d29c679`.

---

## Retrieval Quality — Persistent Issues

### Documentation stubs served in outline-only mode for explanation queries

README.md chunk 1 is retrieved for "what is opencaw?" but injected as 50-token
section headings only
(`Outline: OpenCAW - Context as Workspace, Architecture, Crates...`). The prose
that answers the question is in chunks 2–4, which aren't retrieved. The
`wants_explanation` flag exists and retrieves the right file but doesn't trigger
full-content injection for documentation stubs. The 1.5× Markdown score boost
gets them retrieved; progressive disclosure (stub→full upgrade on probe) remains
unimplemented. Seen in loops 0015–0016.

### Inventory/count queries retrieve only one chunk of multi-chunk documents

"How many todos are left?" matches one TODO.md chunk (whichever has the highest
BM25/semantic score) while the remaining chunks are never seen.
`is_inventory_request` and `is_status_request` intent flags are correctly
classified but not wired to any retrieval behavior. The document-level overview
stub (section headings + completion state) is in TODO.md as an open item but not
implemented. Consistent failure across loops 0003–0013.

### Session history consumes 38–58% of workspace budget before any stubs load

The 1400-token model response from a prior session scores higher than workspace
stubs for follow-up queries and is admitted first. The B7 fix (compact synthesis
≤50 tokens per prior turn before embedding) is specified but not implemented.
Seen in loops 0009–0013.

### `dynamic.rs` (34–39 chunks) floods retrieval for unrelated queries

Largest single file; its summaries are generic enough to match almost any query.
Without a per-file admission cap or retrieval diversity penalty, it consumes
top-k slots across unrelated sessions. Identified in loops 0012–0016.

### Noise sources indexed and retrieved

Files that should be excluded from retrieval but aren't:

- **`.caw/session-*.md`** — session transcripts (B3 above)
- **`MASCOT.md`** — mascot design criteria, surfaces for architectural queries
- **`codebase-review-report.md`** — generated audit artifact, competes with
  primary docs
- **`CLAUDE.md`** — AI process instructions, surfaces for user-facing queries
- **`scripts/qa.sh`** — contains a CODEBASE LAYOUT stanza that reads as
  authoritative project documentation (with outdated crate names)

### Tail chunks embedded as arbitrary code lines

Last chunk of a large file often has summary = first non-empty code line
remaining = `&mut done_files,` or `let pass = answer.to_lowercase()...`. These
stubs are unreachable by any meaningful semantic query and consume index slots.
The `token_estimate < 10` filter catches 3-token stubs; single-expression tail
chunks with 24–34 tokens escape it.

### Cargo.toml stubs summarized as `[package]`

First chunk summary is the literal first line of the file. A query about "what
embedding library is used" cannot find the relevant Cargo.toml entry. Flagged in
every loop from 0001 onward.

### Convergence detection missing from multi-pass recall loop

The loop runs until budget exhaustion. For "how many todos are left?", the
workspace grew from 22 → 31 → 46 fragments across iterations while answer
quality didn't improve. Stopping when no new unique stubs were admitted in the
last iteration would fix this. Identified in loop 0003, confirmed through 0011.

### Token estimate vs. actual injection discrepancy

`token_estimate` values in the DB are systematically larger than actual injected
token counts (one example: estimated 491, injected 55). The scheduler reserves
budget based on estimates, so the workspace hits its ceiling prematurely and
leaves out stubs that would fit. Identified in loop 0013.

### Intent classifier undertriggers on count/status language

"How many X are left?" should map to `is_inventory_request` +
`is_status_request`. In practice it often classifies as all-false. Count
language ("how many", "how much", "total", "remaining") needs stronger examples
in the classifier prompt/training. Seen in loops 0005–0006.

### Chronic no-engagement stubs have no penalty

Stubs evicted from every session on unrelated queries (e.g., `probe_recall.rs`,
`caw-adapters/src/ollama.rs`) continue to be admitted and immediately evicted in
every subsequent session. There's no mechanism to lower their prior probability
of admission based on eviction history. Consolidation notes accumulate eviction
records but the retriever doesn't use that signal.

### Model self-diagnostic query (loop 0012, 0013) is a useful test pattern

"You're using it right now; is the context helpful? Is it too cluttered?"
reliably surfaces real context quality problems. The model accurately
identified: redundant repeated chunks, `Chunk X/Y` metadata noise, lack of
hierarchical summarization, query-agnostic retrieval mixing high-level docs with
low-level implementation. This should be a permanent fixture in the QA file set.

---

## Script / Harness Issues

- **Review prompt asks Claude to implement at step 7** — then the implementation
  pass runs the identical goal. They interfere with each other and produce lower
  quality in both.
- **Both implementation passes use the identical prompt** — pass 2 has no
  awareness of what pass 1 did; it often duplicates or partially undoes the
  work.
- **Empty Claude output not detected** — loops 0019 and 0021 produced empty
  review output; the script continued silently to implementation with nothing to
  act on.
- **`RUST_LOG="debug"` floods `qa/log.txt`** — thousands of file-discovery lines
  per run bury actual session output and failure signals; the log becomes
  useless for diagnosing failures.
- **`nice to know` in `qa/0001.txt`** — the third question in the first QA file
  is a casual acknowledgment, not a question. The classifier correctly marks it
  all-false but the retrieval still runs and the model fabricates a response
  about "nice-to-have features." One impl pass replaced it, another reverted it.

---

## What Actually Works Well

- Intent classification accuracy is high for well-formed queries
- Model reasoning is sound when context genuinely covers the query — failures
  trace to retrieval, not reasoning
- B6 strip (fake recall markers) works
- B13/B14 (false-positive degenerate from template tokens) fixed and holding
- `wants_explanation` correctly identifies architectural overview queries
- caw-bench stub 0.25× penalty effective — bench internals stopped flooding
  overview sessions after loop 0015
- `search-candidates` mode identifies the right files; the gap is
  content-loading, not ranking
- Answer quality is high when retrieved context matches the query (loops
  0011–0012 positive findings)
