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

> **Reconciliation 2026-06-13.** This list predates the May–June fixes and is partly stale. Verified against current code this pass:
> - **B3 (`.caw` not skipped during indexing) — resolved in the ingest path.** `caw-ingest/src/lib.rs::should_skip` returns `true` for any path component starting with `.caw` (and a session-log filter); `build_index.rs::should_skip` has the same guard.
> - **Session-history compression (B7/B10 root cause) — implemented.** `caw-orchestrator/src/session.rs` compresses prior assistant turns to ≤`MAX_PRIOR_ASSISTANT_CHARS` (300 chars, ~50 tokens) via `compress_assistant` before in-memory embedding; the 0.25 history-budget cap is also live (`dynamic.rs`). The "B7 fix specified but not implemented" persistent-issue entries below are obsolete.
> - **B0/B1/B2 (session-log truncation, path corruption, missing first turn) — NOT re-verified this pass.** They date to the degenerate-output era; many adjacent bugs (B15–B22) were fixed since. Re-run the CLI recall loop and inspect a fresh session log before carrying these forward as real.

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

---

## Measured Retrieval Quality — `caw-bench-graph-eval` (2026-06-13)

Run against `graphify-eval-index.sqlite` (748-chunk code index, BGE/Candle CUDA),
27 golden questions (13 structural, 6 definition, 8 caller). Baseline = semantic
(cosine) only; treatment = cosine seeds + graph-edge neighbor expansion.

**Cosine alone is weak as a top-1 retriever on this single-domain code corpus:**
`recall@1 = 0.222` overall, `0.000` for both definition and caller queries.
The model rarely gets the answer chunk in the *first* slot, though the right
chunk is usually in the admitted pool.

### Grounding coin-flip resolved (2026-06-13): not a retrieval miss, not raw model incapacity

The `honest.log`/`perf-fast.log` generic-textbook eviction answers (FIFO/LRU,
"evicting query intents") looked like a retrieval or model failure. Direct
isolation says otherwise:

- **Retrieval surfaces the right chunks.** `/v1/retrieve` on "What triggers
  eviction…" admits `consolidation.rs` at rank 0 (0.78) and `dynamic.rs` at
  ranks 1/2/5 — the eviction loop and the on-eviction consolidation. Not ranked out.
- **Both models ground when context is injected and the query is precise.**
  Same proxy injection (7 fragments), `llama3.2:3b` and `mistral-small3.2:24b`
  both produced grounded answers citing `record_eviction`, `decayed_score`, and
  the literal `"Evicted (relevance decayed to {:.2})"` note format. The 24b model
  additionally (correctly) flagged that the precise threshold chunk wasn't in the
  injected snippets.
- **The CLI recall engine grounds too**, with the same precise phrasing.

The driver is **query framing + budget clamp**, in two parts:
1. **Vague conceptual phrasing retrieves diffusely and invites confabulation.**
   "how does eviction work in CAW?" (the `honest.log` phrasing) retrieves the real
   `synthesize_eviction_note` signature but the small model wraps it in invented
   conceptual glue ("affective scores", "bias warnings"). The precise pool phrasing
   ("What triggers eviction *of a fragment from the workspace*…") grounds cleanly.
   Nothing in the prompt biases the model toward the injected context over its prior.
2. **The most precise chunk is clamped out by budget, not ranked out.** The
   hysteresis-constant `dynamic.rs` chunks ranked but landed `budget_full` below
   the 2000-token clamp. Raising `--max-tokens` or adding per-file diversity would
   admit them.

Actionable consequences — status after the 2026-06-13 fixes:

- **(a) Grounding framing — DONE.** `caw-core::workspace_guidance` now always emits a
  baseline hint ("prefer the recalled context… if it does not cover the question, say
  so rather than inventing specifics"). Applies to every surface that renders a
  workspace (proxy + orchestrator/CLI). Verified: the vague "how does eviction work in
  CAW?" query, which previously produced generic FIFO/LRU text (`honest.log`) and later
  confabulated "affective scores", now grounds in `dynamic.rs`/`consolidation.rs` and
  honestly hedges on the thresholds it can't see.
- **(b) Budget waste from duplicate sources — DONE (proxy).** The proxy's clamp counted
  tokens per stub while `format_workspace` renders only the first chunk per source, so a
  multi-chunk file (`dynamic.rs`) burned budget for content never injected. The clamp now
  drops same-source duplicates (new `Disposition::DuplicateSource`), mirroring the
  orchestrator's existing `load_fragments` dedup. Measured on the eviction query: distinct
  admitted sources rose 5 → 7 at the same 2000-token ceiling (`runner.rs`, `README.md`
  promoted from `budget_full`). The library orchestrator already deduped at admission, so
  no library change was needed.
- **DEFERRED — surfacing a lower-ranked chunk of an already-admitted file.** The precise
  hysteresis-threshold chunk is itself a lower-ranked `dynamic.rs` chunk; first-per-source
  rendering still collapses it to a note, and showing it risks reintroducing the
  `dynamic.rs`-flooding bug (B1). This is the "insertion order / admission" open question
  in DECISIONS.md and needs a controlled comparison, not a unilateral change. Evidence for
  needing it is thin (precise queries already ground across every path/model).

Aside: the proxy index (`caw-bench-build-index`, 3190 stubs over crates) and the CLI
index (837 stubs over the same crates) chunk differently, so the two surfaces retrieve
differently for the same query — worth unifying or at least documenting.

**Graph expansion is a clear net win for structural/caller queries, neutral-to-negative for definitions:**

| metric | cosine | +graph | Δ |
|---|---|---|---|
| MRR (all) | 0.407 | 0.433 | +0.025 |
| recall@3 (all) | 0.519 | 0.667 | +0.148 |
| recall@5 (all) | 0.593 | 0.704 | +0.111 |
| recall@20 (all) | 0.852 | 1.000 | +0.148 |
| recall@3 (caller) | 0.125 | 0.500 | +0.375 |
| MRR (caller) | 0.145 | 0.274 | +0.128 |
| recall@10 (definition) | 0.833 | 0.500 | **−0.333** |
| recall@10 (all) | 0.778 | 0.741 | −0.037 |

4 of 6 cosine-failed golds were rescued into top-10, all via `calls` edges
(`split_thinking`, `count_tokens`, `ensure_ort_dylib_path`, `truncate_at_chat_boundary`).
`recall@1` is unchanged everywhere — expansion reranks the pool, it never promotes
into the top slot.

Caveats: n=27, single index, golden set authored alongside the feature.

### Correction (2026-06-13): the lift is against the wrong baseline — it does not survive hybrid

The table above is **cosine-only baseline vs. cosine+expansion**. The live retriever
(proxy and CLI) is **hybrid** (BM25 fused with cosine). `caw-bench-graph-eval` now
takes `--baseline hybrid` (BM25 built on the proxy's `path+summary+body` text, fused
at the proxy's 0.6/0.4 weights). Re-measured against the baseline the system actually
uses:

| baseline | merge | MRR (all) | recall@3 (all) | caller recall@3 | verdict |
|---|---|---|---|---|---|
| cosine | adjacent | 0.407 → 0.437 | 0.519 → 0.667 | 0.125 → 0.500 | lift (but wrong baseline) |
| **hybrid** | adjacent | 0.488 → 0.432 | 0.630 → 0.407 | 0.500 → 0.125 | **net loss** |
| **hybrid** | discounted | 0.488 → 0.496 | 0.630 → 0.630 | 0.500 → 0.500 | ~neutral (+0.037 @5–@50) |

Two things changed the picture:
1. **BM25 already captures the rescue.** The hybrid baseline alone lifts caller
   recall@3 from 0.125 to 0.500 — i.e. BM25 finds the symbol-name caller queries that
   cosine missed and graph expansion was rescuing. The graph and BM25 are largely
   redundant for this golden set (whose questions are natural-language paraphrases).
2. **`merge_adjacent` is hostile to a strong baseline.** It inserts every planned
   neighbor right after its seed regardless of the neighbor's score; when gold is
   already at rank 2–3 (via BM25), up to `max_neighbors` chunks get shoved into the
   top band and push gold out of recall@3/@5. That mechanism — not "graph edges are
   useless" — is the −0.222. `merge_discounted` (add neighbors at `seed*0.5*weight`
   and let the existing sort place them) avoids it: it never displaces a high-ranked
   gold, and gives a small deep-recall bump (@5–@20 +0.037) with no definition or
   caller regression.

**Recommendation: do not wire expansion into the live retriever now.** Against the
hybrid baseline the system actually uses, the only non-harmful policy tested
(`discounted`) is marginally positive at ranks 5–20 — below the ~7 fragments the proxy
injects — and does not justify the shipping cost (a `caw-graph-ingest` step in the
build path, the `graphify-out/graph.json` dependency, and edge plumbing through
`AppState`). The earlier "intent-gated definition skip" plan is moot: discounted merge
shows no definition regression, and the proxy has no intent classifier to gate on.

**Door left open (untested):** expansion gated to *weak-baseline* queries (where gold
is deep and BM25 also misses — the population the cosine win came from), or a merge
that only admits a neighbor when it would land inside the injected top-k. Both are
hypotheses, not scheduled work.

`plan_expansion` remains called only by `caw-bench-graph-eval`; nothing in the live
path was changed. The cosine-vs-hybrid bench arm is kept as the artifact that produced
this decision so the question isn't re-litigated from the cosine number alone.

### Identifier-aware BM25 tokenization — shipped (2026-06-13)

The BM25 tokenizer lowercased *before* splitting on non-alphanumeric, so `snake_case`
split but `camelCase`/`PascalCase` did not: `DynamicRecallOrchestrator` stayed one
token, and a paraphrased query ("which struct owns the recall loop") shared no lexical
token with it — a both-retrievers-miss generator. Fixed in `caw-index/src/bm25.rs`:
split identifiers at camelCase and letter↔digit boundaries before lowercasing, keep the
joined form for exact matches, emit the subwords. Default-on; the proxy rebuilds BM25
in-memory from bodies so it lands with no reindex.

Measured lift to the **hybrid baseline** (`--baseline hybrid`, n=27): MRR 0.488→0.525,
recall@3 0.630→0.704, definition recall@3 0.500→0.667, structural recall@3 0.769→0.846.

### Both-miss diagnosis — what's left after the tokenizer fix (2026-06-13)

`caw-bench-graph-eval --diagnose` decomposes each gold ranked past the cutoff into
cosine-rank / BM25-rank / query↔gold token overlap / gold-text size. On the hybrid
baseline it surfaced two distinct mechanisms, neither of which graph expansion would
fix:

1. **The hybrid fusion buries single-half hits — the hybrid is worse than BM25 alone
   for them.** `fuse_hybrid` min-max normalizes each list and always divides by the
   full `0.6+0.4` weight, so a keyword-only hit is capped at `0.4×score` while a
   mediocre chunk present in *both* lists outscores it. Result: BM25 ranks
   `split_thinking` #4 and `count_tokens` #15, but they fuse to #13 and #31. This is a
   real fusion bug with known fixes (divide by the weight of the lists an item actually
   appears in; or rank-based RRF). Untested — the `--diagnose` instrument is now in
   place to measure a fix.
2. **Stale/unreadable bodies fall out of the lexical index.** BM25 build skipped 114 of
   3174 stubs (unreadable bodies — here, index/corpus drift against a prebuilt index).
   The default-hysteresis-load/unload chunks are among them, so those numeric-constant
   definition queries have no lexical fallback and ride on cosine alone (which buries
   them at 12/9). Partly an artifact of testing an old index against current source, but
   the structural point stands: a stub absent from BM25 is invisible to the lexical half.

Caveat throughout: n=27, single golden set authored alongside the graph feature. Treat
the per-mechanism direction as the signal, not the decimals.
