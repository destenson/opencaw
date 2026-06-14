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

1. **The hybrid fusion buries single-half hits — but the "fix" is worse (see below).**
   `fuse_hybrid` min-max normalizes each list and always divides by the full `0.6+0.4`
   weight, so a keyword-only hit is capped at `0.4×score` while a mediocre chunk present
   in *both* lists outscores it. Result: BM25 ranks `split_thinking` #4 and `count_tokens`
   #15, but they fuse to #13 and #31. This looked like a bug; measuring two standard fixes
   showed it is the **price of a net-beneficial agreement reward**, not a defect.
2. **Stale/unreadable bodies fall out of the lexical index.** BM25 build skipped 114 of
   3174 stubs (unreadable bodies — here, index/corpus drift against a prebuilt index).
   The default-hysteresis-load/unload chunks are among them, so those numeric-constant
   definition queries have no lexical fallback and ride on cosine alone (which buries
   them at 12/9). Partly an artifact of testing an old index against current source, but
   the structural point stands: a stub absent from BM25 is invisible to the lexical half.

Caveat throughout: n=27, single golden set authored alongside the graph feature. Treat
the per-mechanism direction as the signal, not the decimals.

### Fusion experiment — negative result, incumbent retained (2026-06-13)

Tried two standard fusions to rescue the single-half burial, measured on the hybrid
baseline (n=27, `caw-bench-graph-eval --baseline hybrid`):

| fusion | MRR (all) | recall@3 (all) | verdict |
|---|---|---|---|
| divide-by-total (incumbent) | ~0.51–0.53 | 0.704 | retained |
| present-weight (÷ weight of lists present) | 0.330 | 0.444 | decisively worse |
| RRF (k=60) | 0.492 | 0.667 | inconclusive-to-worse, not k-swept |

- **present-weight** (divide each item by the weight of the lists it actually appears in)
  is decisively worse: removing the per-item total-weight divisor discards the *agreement
  reward* — items both retrievers rank get a summed bonus — and the top floods with
  single-list items that min-max normalization inflates. The diagnosed "burial" is the
  cost of that reward, and the reward is worth more than the ~2 golds it buries.
- **RRF k=60** is inconclusive-to-slightly-worse. Two reasons it underperforms here:
  embedding-cosine *magnitude* is informative on this corpus and RRF throws it away
  (rank-only); and k=60 is tuned for fusing thousand-item web lists — for a 100-item pool
  where top-10 is what matters it's too flat (which is why definition queries, leaning on
  cosine rank, regressed). A fair-k RRF was **not** swept: at n=27 a few-point win would be
  inside the noise floor (the HashMap tie-order wobble alone is ±~0.01 MRR), i.e.
  overfitting to a self-authored set.

**Decision: keep the incumbent divide-by-total fusion.** The real blocker to adjudicating
fusion variants is eval-set size, not the variant — which is the same gap `scope.md`
already flags ("sweep runs against enough seeds"). Single-half burial is a known, accepted
cost; the door is open for a real fusion change once a larger, independent eval set exists.
Nothing in `caw-server` was touched — the experiment ran entirely in the bench mirror.

## Independent eval set — sysdoc, n=100 chunk-level (2026-06-13)

To escape the n=27 overfit, built an independent golden set on a different corpus and
question style: 100 quote-anchored Q&A over Linux/Debian system docs (`opencaw-corpora`),
converted to chunk-level `path+line` gold (`scripts/sysdoc-qa-to-chunk.py` →
`crates/caw-bench/src/qa/sysdoc_chunk_qa.json`), measured against a curated 2,600-doc
haystack (`scripts/curate-subset-medium.py`, index `target/caw-dev/subset-medium.sqlite`,
43k stubs). Content is post-training-cutoff (Dec 2025 / 2026 CVEs) so the answer model
can't have memorized it. Types: changelog 56, copyright 17, doc 14, readme 8, news 5.

### Headline: hybrid ≫ cosine, validated at n=100

| baseline | MRR | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|---|
| cosine | 0.344 | 0.270 | 0.390 | 0.410 | 0.480 |
| hybrid | 0.525 | 0.380 | 0.600 | 0.690 | 0.840 |

BM25's contribution is even larger here than on the n=27 crates set — these answers are
exact tokens (CVE IDs, version strings, package names) that embeddings blur but lexical
match nails. recall@10 0.48→0.84 is the clearest single number: the hybrid-is-the-right-
baseline decision holds on a 3.7× larger, independent corpus.

Per type (hybrid): **changelogs are the hard category** — recall@1 0.286 vs doc 0.571 /
readme 0.500. A changelog answer is one line among many near-identical version entries
across many package changelogs; that's the genuinely hard discrimination problem.

### Fusion re-adjudication — the n=27 verdict flips on this corpus

The n=27 set said "keep `divide_total`, present_weight is catastrophic (MRR 0.525→0.330)."
On n=100 it reverses. `caw-bench-graph-eval --fusion all` (builds BM25 once, embeds each
query once, evaluates all three modes on identical inputs — so the per-mode *delta* is
apples-to-apples):

| fusion | MRR | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|---|
| divide_total (incumbent) | 0.52–0.53 | 0.38 | 0.61 | 0.71 | 0.84 |
| **present_weight** | **0.57–0.58** | 0.42 | 0.67 | 0.76 | 0.80 |
| rrf (k=60) | 0.46 | 0.36 | 0.49 | 0.61 | 0.61 |

present_weight (divide each item by the weight of the lists it actually appears in, so a
strong single-half hit isn't capped) wins the **top ranks**: MRR +0.045, recall@1 +0.04,
recall@3 +0.06, recall@5 +0.05 — at the cost of recall@10 (−0.04). Deltas stable across
repetitions and 5–6× the GPU-nondeterminism noise floor (~±0.008 MRR; a sort tiebreak was
added for ties but cuBLAS reductions still aren't bit-reproducible, so identical runs
wobble ~0.01 in the absolutes — read the within-run deltas, not the third decimal).

Why the flip: n=27 was hand-authored *paraphrase* questions over code where BM25 added
little, so removing the agreement reward only hurt. n=100 is quote-anchored *fact-lookup*
whose answers **are** exact tokens — the most BM25-favorable style — where many golds are
BM25-strong / cosine-absent, exactly the population present_weight rescues. Corpus and
question-style changed together, so the honest claim is narrow: **present_weight helps
lexical-exact-answer workloads.** Whether live traffic looks like that is open.

### Not shipped to caw-server — it's a surface-dependent trade, the user's call

present_weight is a top-rank-vs-deep-rank trade (wins ≤5, loses ≥10). The **proxy injects
top-k (~7 fragments)** so it would benefit; the **CLI/orchestrator runs multi-pass with
deeper budgets** so it might be hurt. Different surfaces want different answers, and the
sign already flipped between corpora — so this is a deployment decision, not a unilateral
fusion swap. `caw-server::fuse_hybrid` is unchanged; the three modes live behind
`graph_eval --fusion` for continued measurement. Tuning a 4th variant on n=100 would be
the same overfit-to-one-set trap refused on n=27, from the other side.

## Thinking-trace recall is inert on modern Ollama — adapter drops the reasoning channel (2026-06-13)

Attempting the first end-to-end recall-on vs recall-off answer-quality run (the central
claim in `benchmarking.md` line 3, never previously run at scale) surfaced a divergence
between the design (`DECISIONS.md`: thinking-trace-as-retrieval, default on) and the
Ollama adapter as shipped.

**Symptom.** `caw-bench --workload sysdoc` (subset-medium, 43k stubs, answer model
qwen3.5:9b, judge claude-code/haiku, n=2): recall_on and recall_off loaded byte-identical
fragments and produced byte-identical answers. `Δ recall@k = 0`, `Δ mrr = 0`; the only
difference was latency (on 32.9s vs off 19.2s — wasted iteration). The model emitted no
`<think>` text in its answer.

**Root cause (confirmed in `caw-adapters/src/ollama.rs`).** The orchestrator's
thinking-trace recall routes through `thinking_with_steps`, which for a
`supports_visible_reasoning` model (name match: contains "qwen"/"deepseek" — qwen3.5
matches, so the path *was* taken) streams `/api/chat` and scans each chunk's
`message.content` for an inline `<think>…</think>` block. But modern Ollama (0.30.6) with
qwen3.5/3.6 streams reasoning token-by-token in a separate `message.thinking` field with
`message.content` empty until thinking finishes — verified by raw streaming call. The
inline scan never matches, zero steps are collected, the thinking-trace re-query never
fires, and recall_on collapses to recall_off. The request also never sends `"think": true`,
so on these models reasoning may not be emitted at all. `OllamaChatRequest` has no `think`
field; `OllamaStreamToken`/`OllamaChatMessage` have no `thinking` field.

**Consequence for the thesis test.** The recall-on/off sweep cannot test opencaw's
differentiator until the trace is wired through; an n=100 run as-is is structurally
predetermined to show on==off and would only reconfirm the inertness. Sweep is blocked on
the adapter fix.

**Fix scope (confined to `ollama.rs`).** (1) Add `think` to `OllamaChatRequest`, gated —
`think:true` on a non-reasoning model crashes llama-server (GGML_ASSERT, observed on
gemma4). (2) Add `thinking: Option<String>` to the message/stream structs; in
`thinking_with_steps` accumulate `message.thinking` deltas (split on `\n\n` for steps,
stop when `content` begins) as the primary trace, keeping the inline `<think>` scan as a
fallback. (3) In `complete()` prefer `message.thinking` over `split_thinking(content)`.
Gate option: `/api/show` exposes a per-model `thinking` capability (non-heuristic,
replaces the name match) — but it is imperfect (gemma4 advertises `thinking` yet crashes;
phi4-reasoning omits it yet emits inline `<think>`), so the inline fallback must stay and
an advertised-but-crashing model should surface an error, not be papered over.

**Secondary, separate issue.** `cooperative_probes`/`supports_hidden_reasoning` is
hard-coded false for all Ollama models (`adapter_factory.rs`: "enable after verifying with
caw-bench-coop"), so `<probe>` instructions are never injected locally. That is a second
cooperation channel; the trace fix above is independent and unblocks the headline
mechanism on its own.

**Caveat on workload choice.** sysdoc is single-token fact-lookup (CVE IDs, version
strings) — a weak exercise of thinking-trace, which should help most on synthesis/
exploration where reasoning reaches content initial retrieval missed. Once the trace is
wired, the `opencaw` codebase workload is the stronger thesis test; a null on sysdoc alone
would not disprove the thesis.

### Fix landed and validated — thinking-trace recall now fires on modern Ollama (2026-06-13)

Two-part fix (both confined, no cross-crate API change):
- `caw-adapters/src/ollama.rs`: send gated `"think": true` (gate = `/api/show` `thinking`
  capability, cached; model-name match only as offline fallback — sending it to a
  non-reasoning model crashes llama-server), add a `thinking` field to the message/stream
  structs, accumulate `message.thinking` stream deltas in `thinking_with_steps`, and prefer
  `message.thinking` over `split_thinking(content)` in `complete()`. `supports_visible_reasoning`
  now derives from the capability query.
- `caw-orchestrator/src/dynamic.rs`: the `DynamicRecallOrchestrator` recall loop fed
  `last_response.answer` to `process_thinking_trace`. Modern adapters now return the reasoning
  separately in `last_response.thinking`; feed that (fall back to answer when absent). Feeding
  the bare answer retrieved on the model's own output, not its reasoning.

Validation by behavioral isolation at temperature 0 (deterministic), qwen3.5:9b, subset-medium,
n=2: with only the adapter fix (orchestrator still feeding `.answer`), recall_on and recall_off
loaded identical fragment sets (11 vs 11). After also wiring the orchestrator to feed
`.thinking`, recall_on diverged to 16 vs 11 loaded — the thinking-trace re-query now pulls extra
fragments from the reasoning trace. The divergence appeared exactly at the orchestrator wiring
change, isolating the cause. (Note: `caw-bench` installs no tracing subscriber, so RUST_LOG logs
are unavailable for diagnosis — behavioral isolation was the verification.)

On sysdoc n=2 the extra fragments slightly *lowered* the score (Δ answer_score −0.075) — expected:
sysdoc is single-token fact-lookup where more near-identical changelog fragments add confusion,
not synthesis where reaching unretrieved content helps. The mechanism is validated as *wired and
firing*; whether it improves answer quality is the separate question the at-scale sweep measures,
and the `opencaw` codebase (synthesis) workload is the more favorable test than sysdoc.

## At-scale recall-on/off — sysdoc n=40 answer quality (2026-06-13)

First at-scale end-to-end answer-quality run of the central claim, now that the trace fix
makes recall_on non-trivial. qwen3.5:9b answer, haiku judge, subset-medium index, temp 0.

| mode | n | answer_score | recall@k (path) | mrr | ctx_eff | latency |
|---|---|---|---|---|---|---|
| recall_on  | 38 | 0.042 | 0.737 | 0.439 | 0.698 | 23.0s |
| recall_off | 38 | 0.045 | 0.737 | 0.439 | 0.615 | 17.1s |

Δ answer_score = −0.003 (noise; 34/38 items score 0.0 in both modes). **recall_on ≈ recall_off
on sysdoc** — expected: fact-lookup is adverse for thinking-trace, and the bottleneck here is
upstream of orchestration.

**The dominant failure is chunk-level injection precision, not retrieval or recall.** recall@k
is 0.737 but answer_score is ~0.04 — the gold *file* is loaded yet the answer is wrong. Inspecting
gold-retrieved/score-0 items: the model honestly says "the context does not contain X" because the
*injected chunk is the wrong chunk of a multi-chunk changelog*. Example (mintsources 2.3.2 geoIP):
expected `software-properties-common/changelog` was loaded, but the rendered fragment is
"Chunk 2/17 … mintsources (2.3.5)" — the 2.3.2 answer is in a different chunk of the same 17-chunk
file. The retriever surfaces the top-scoring chunk (a recent version); the answer sits in a specific
older-version chunk; first-per-source rendering shows only one chunk per file. Thinking-trace recall
can't rescue this — re-querying reloads the same file's top chunk.

Consequences:
- **Path-level recall@k overstates answer availability on multi-chunk docs.** The 0.84 recall@10
  headline (chunk-level, from graph_eval) and this 0.74 path-level recall both look healthy, but
  neither guarantees the *answer line* is injected.
- The next lever for fact-lookup is chunk/injection precision, not orchestration: surface the
  answer-bearing chunk of an already-loaded file (the DEFERRED item above), changelog-aware chunking
  (version headers as chunk anchors), and budget/first-per-source rendering that currently collapses
  a file to one chunk. These are the open TODO retrieval items, now with direct evidence.
- **Honest grounding validated at n=40:** the model says "not present" rather than confabulating —
  the workspace_guidance grounding hint holds across the set.

This does not test the thesis (sysdoc is adverse); the opencaw synthesis workload is the fair test.

## Per-source cap (1→3): measured effect is workload-dependent (2026-06-13)

Re-ran both sweeps after the per-source chunk-cap fix (commit dc424ad). The fix fired as
intended — avg max-chunks-from-one-source rose 1.00→2.82, total loaded 10.7→17.4 on sysdoc —
but the answer-quality effect splits by workload, the opposite of the sysdoc prediction:

| workload | mode | pre | post | Δ | per-item (Off) |
|---|---|---|---|---|---|
| sysdoc (fact-lookup) | recall_on | 0.042 | 0.015 | −0.027 | improved 1 / regressed 4 / unchanged 32 |
| sysdoc | recall_off | 0.045 | 0.028 | −0.017 | |
| opencaw (synthesis) | recall_on | 0.148 | 0.189 | +0.041 | improved 7 / regressed 2 / unchanged 19 |
| opencaw | recall_off | 0.115 | 0.176 | +0.061 | |

**Why sysdoc regressed — the mechanism-B diagnosis was incomplete.** The earlier check confirmed
the answer is *in the gold file* but not that it ranks in the *top-3 chunks of that file*. For
"mintsources 2.3.2" the cap=3 now admits chunks 2/5/11 (versions 2.3.5, 2.2.9, 2.1.1) — three
*wrong*-version chunks, not the 2.3.2 chunk, which ranks below them. So the cap loads more
near-identical changelog noise without surfacing the answer, and the model hedges harder. The real
fact-lookup bottleneck is chunk-level *ranking* (the answer chunk must rank top-1 of its file), not
the per-source cap — i.e. the open "both-miss / chunking-artifact" TODO item (contextual embeddings,
version-aware chunking), a different fix.

**Why opencaw improved.** On synthesis questions the relevant chunks of a file *are* among the
top-scoring ones, so admitting up to 3 surfaces genuinely useful additional context; 7 of 28 items
improved. This is the thesis-relevant workload and the fix helps it.

Honest read: the per-source cap is correct in principle (one-per-source was over-aggressive and
provably drops content on synthesis), but it is **not** the fact-lookup fix it was hoped to be — it
mildly hurts fact-lookup by dilution. n=30–40, scores low (0.02–0.19), so treat magnitudes as
directional; the *signs* are consistent across both modes within each workload. Recall_on > recall_off
holds on opencaw post-fix (0.189 vs 0.176), preserving the synthesis thesis signal.

## Full-file expansion on explicit reference (2026-06-14)

Follow-up to the per-source cap: the cap mildly hurt fact-lookup because the answer chunk
often ranks below the top-3 chunks of its file, so cap=3 admits wrong-version chunks and
dilutes. Per the project goal — curate context so the model needn't tool-call for on-disk
content — added a budget-bounded full-file escalation driven by the (now-wired) thinking
trace.

Mechanism: when a reasoning step (or the answer) names a file already in the workspace by
its path, `DynamicRecallOrchestrator::process_file_expansion` enumerates that file's chunks
(`Retriever::chunk_ids_for_source`, an indexed `WHERE path=?` on `SqliteStubStore`, O(N)
default elsewhere) and loads them as full bodies, exempt from the per-source cap but still
inside the token budget. It also *upgrades* chunks already loaded as stubs (thinking-trace
recall admits stubs) to full bodies — skipping already-loaded ids would leave the answer in
a summary forever. The model can only name a path it was shown, so candidates are restricted
to currently-loaded sources; matching is literal-path-occurrence (no name heuristic — a file
named differently from its content, e.g. the mintsources entry living in
`software-properties-common/changelog`, is out of scope).

Coupled fix: the multi-pass convergence test was fragment-count based
(`loaded.len() == loaded_before || net_new_tokens < threshold`). A stub→full upgrade is
count-neutral, so the count clause broke the loop *before* the re-completion that would let
the model use the upgraded bodies. Changed convergence to token-based only
(`net_new_tokens < threshold`), which captures both new fragments and stub→full upgrades.

Verified deterministically (not via the noisy full-stack sweep): the bench instrumentation
showed expansion fires, ids match, and the gold file loads as full bodies (sysdoc_001:
3/3 chunks full, ref present in context); a mock-level orchestrator regression test
(`names_file_upgrades_all_chunks_to_full_bodies`) locks the upgrade behavior. Whether this
lifts aggregate answer quality is the at-scale sweep's job — pending. Out-of-scope residuals
seen in the smoke: the gold file never retrieved at all (ranking, mechanism A), and
hard/ambiguous QA items where the content is present but the model mis-extracts.

## Full-file expansion — at-scale result: strong recall_on lift, thesis validated (2026-06-14)

Three-config comparison (qwen3.5:9b answer, claude-code/haiku judge, temp 0; sysdoc n=40,
opencaw n=30). answer_score:

| config | sysdoc on | sysdoc off | opencaw on | opencaw off |
|---|---|---|---|---|
| one-per-source (pre) | 0.042 | 0.045 | 0.148 | 0.115 |
| cap=3 only | 0.015 | 0.028 | 0.189 | 0.176 |
| **cap=3 + expansion** | **0.156** | 0.029 | **0.232** | 0.156 |

**The full-file expansion is the change that makes the multi-pass machinery pay off.** It runs
only in `recall_on`'s refinement loop (the model must name a file in its reasoning), so
`recall_off` is unaffected — and that is exactly the point: it is a genuine differentiator of the
orchestration, not a retrieval tweak that also helps the single-shot control.

- **sysdoc (fact-lookup): recall_on 0.156 vs recall_off 0.029** — a 5.4× on/off gap, up from
  ~parity. Per-item: 9 improved, **0 regressed**; items scoring ≥0.5 went 0 → 6. The escalation
  recovers exactly the failure diagnosed earlier (gold file loaded but answer in an unshown chunk):
  when the model names the file, the whole file loads and the answer becomes available.
- **opencaw (synthesis): recall_on 0.232 vs recall_off 0.156** — a +0.076 gap, the largest yet.
  Per-item 8 improved / 4 regressed (synthesis is noisier).

This is the first strong, consistent end-to-end validation of the core thesis: multi-pass,
thinking-trace-driven recall with eviction/expansion substantially beats single-shot retrieval at
matched budget, on both fact-lookup and synthesis.

**Honest caveats.** (1) `recall@k_on` *dropped* (sysdoc 0.744→0.658, opencaw 0.744→0.613) while
answer_score rose sharply — full-file expansion concentrates the budget on the named file, crowding
out other expected-path files, but the answer is in the file that *is* fully loaded. This is more
evidence that path-level recall@k is a poor proxy; answer_score is the metric that moved the right
way. (2) Absolute scores remain low (0.16–0.23) — a small abliterated 9B model + strict judge; the
*deltas and signs* are the result, not the absolutes. (3) Single run each, n=30–40. (4) The cap
alone (without expansion) still mildly hurts fact-lookup vs one-per-source; expansion is what turns
it into a large net win. (5) The eval loop is slow (~20s/item); see the perf-diagnosis TODO.

## Eval-loop latency — phase decomposition and the judge fix (2026-06-14)

Diagnosed the "~20s/item" before optimizing, per the perf-diagnosis TODO. Added a `TimingAdapter`
decorator in `caw-bench` (wraps the answer and judge adapters, accumulates per-call wall time) so
each item's `latency_ms` splits into `gen` (answer-model completions + count), `judge`, and `other`
(embed + retrieve + orchestration). Measured n=2, recall_on, release build:

| workload | latency | gen | judge | other |
|---|---|---|---|---|
| sysdoc (fact-lookup) | 21.2s | 7.5s ×1.0 | 8.5s | 5.2s* |
| opencaw (synthesis)  | 24.4s | 12.7s ×2.0 | 11.3s | 0.5s |

*item-1 only — the one-time in-memory HNSW build over the 43k-stub index lands inside the first
item's clock; steady-state `other` is ~0.8s. Embedding itself is ~4ms/call (candle), a non-factor.

Findings, two of which overturn the TODO's prior suspicion ordering:
- **The multi-pass loop fires on synthesis (gen ×2.0) but not on fact-lookup (gen ×1.0).** On sysdoc
  the refinement loop converged after the first completion (no expansion/probe admitted new tokens),
  so the "up to 3 completions" cost the TODO feared is workload-dependent — real on opencaw, absent
  on sysdoc.
- **First-run CPU storm was a debug build.** instant-distance building the 43k-node HNSW unoptimized
  pegged all cores for minutes with the GPU idle; release fixed it. (Now a standing rule: benchmark
  release-only.) The HNSW-build cost is amortized across a sweep but would bite an interactive/proxy
  use that rebuilds per process start — `hnsw_index.rs`'s "negligible for low thousands of documents"
  comment no longer holds at 43k.

**Judge anatomy (claude-code/haiku).** The per-item judge was ~8.5s. Decomposed by direct
measurement (timing a trivial-prompt `claude --print` call): ~1s process spawn + ~2s CLI/network
fixed floor + ~5.5s actual haiku inference of the judge prompt. So keeping the CLI process warm or
batching would only remove the ~3s fixed overhead, not the ~5.5s inference — a modest win, not the
lever first assumed. `--bare` (the ~1s minimal-harness path) drops OAuth and demands an
`ANTHROPIC_API_KEY` (it *does* honor `apiKeyHelper` via `--settings`, but that's a billing/auth
change). Keeping OAuth, the available knobs are narrow: added `--no-chrome` (~0.9s) and switched
`--setting-sources user`→`project` (loads no settings layer at the pinned `/tmp` cwd, so the
developer's large global CLAUDE.md isn't injected into every auxiliary call) — ~1–2s saved, auth
intact. There is no individual flag for hooks/LSP/plugin-sync/auto-memory; those bundle only into
`--bare`.

**The decisive fix: groq judge (already wired, no code).** `--judge-adapter groq` already routes
through the adapter-agnostic `judge.rs` at `llama-3.3-70b-versatile`. Measured n=2 sysdoc:
**judge 0.45s vs 8.5s — ~19×**, with valid discriminating verdicts (scores 1.00 / 0.00, coherent
rationales, 344–549ms each). This removes the judge as a meaningful phase (≈3% of latency) and cuts
the sysdoc loop ~21s→~15s. Caveat: groq-70b and haiku score differently, so a groq-judged sweep is
**not directly comparable** to the haiku-judged numbers elsewhere in this doc — switching the judge
re-baselines absolute answer_score. Bench default judge stays claude-code/haiku for comparability;
groq is the fast-iteration option.

**What's left in the loop after the judge fix:** answer-generation (gen) dominates — single ~7.5s
completion on fact-lookup, ~12.7s (×2) on synthesis. UX/product levers there (the judge does not
exist in the proxy/CLI product): cap reasoning `num_predict`, or skip the refinement completion when
the first answer is already grounded (trades against the thesis — measure quality). Plus persisting
the HNSW for interactive startup. n=2 throughout — directional, not precise.

### Negative result: cross-item concurrency does not speed up a sweep on this setup (2026-06-14)

Before building a concurrent runner (the serial `for item { for mode }` loop is the obvious
throughput target for sweeps), tested whether the Ollama server actually parallelizes same-model
requests. It does not, on this single-server two-GPU setup. Quick `api/generate` curls against
`qwen3.5:9b`, num_predict=200, model pre-warmed:

- single call: 2.89s
- 4 concurrent: 9.81s (≈3.4× single; 4× serial would be 11.6s, so only ~18% batching headroom)
- 2 concurrent to `qwen3.5:9b`: 5.37s vs 5.92s serial (~9%)
- 2 concurrent to `qwen3.5:9b` + `qwen3.5:9b-worker2`: 5.28s vs 6.0s serial (~12%)

Two reasons it doesn't pay off: (1) the server serializes same-model requests (NUM_PARALLEL
effectively 1 — 4-concurrent scaled ~linearly with count, not flat); (2) the `-worker2` alias did
**not** load as a second resident instance — `api/ps` showed only one `qwen3.5:9b` (6.7GB) loaded
throughout, so its calls resolved to the same instance and serialized. One 9B decode also leaves
only ~18% spare GPU headroom, so even genuine continuous batching on one device would give well
under 2×. A real ~2× would require a second resident instance pinned to the second GPU, which
conflicts with the recorded "don't run a second Ollama / don't pin GPUs" decision.

Conclusion: cross-item concurrency is **not** the sweep-throughput lever here. Optimization stays on
per-item latency — the gen phase (the dominant cost, helps both sweep and product) and HNSW
persistence (one-time, product-startup win) — which is where the TODO/findings plan already pointed.
Directional (n=1 curl timings), but the ~linear 4-concurrent scaling is unambiguous.

## Gen-phase: capping Ollama `num_predict` — the test and pre-committed bar (2026-06-14)

Testing the DECISIONS.md gen-reduction lever: cap the answer model's per-completion budget
(`num_predict`, default 4096) and check whether it cuts gen time without hurting answers. Pre-committed
bar (DECISIONS.md, not invented here): on matched items (sysdoc, recall_on, temp 0, **groq judge** so
both arms share a judge and the score delta is valid), accept a cap for production only if it cuts
`gen_ms` by ≥20% AND mean `answer_score` drops by ≤0.03 absolute AND no previously-correct item
becomes truncated/empty. Otherwise it ships, if at all, only as an explicit dev-sweep fast mode.

Run via `docs/skills/caw-dev/scripts/bench.sh` (new), n=10, the first 10 sysdoc items.

**Baseline (num_predict=4096 default):** mean_answer_score **0.180**, mean_gen **16.6s** (gen_calls 1.0
— single-pass on fact-lookup; judge 0.35s, other 0.79s, so gen is ~93% of latency). Per-item gen_ms
spans 5.8–26.5s (median 17.2). **Only 2/10 items are correct, and both are long generations:**
item-1 (26.5s, score 1.00) and item-7 (19.4s, score 0.80). The entire 0.180 mean rests on those two,
so losing either drops the mean by ≥0.08 — far past the 0.03 floor. The bar therefore reduces to a
sharp test: *can a cap that cuts mean gen_ms ≥20% avoid truncating items 1 and 7?* Mean gen 16.6s at
the 4096 default also implies most traces already finish well below 4096 tokens, so a cap bites only
the long items — which are exactly the correct ones. Hypothesis: the cap cannot clear the bar on this
workload. (Result pending the cap runs below.)

### Result: cap=1024 fails the bar, and the bar can't be cleanly measured at n=10 (2026-06-14)

`--num-predict 1024`, same 10 items, recall_on, groq judge:

| config | mean gen_ms | mean answer_score |
|---|---|---|
| baseline 4096 | 16628 | 0.180 |
| cap 1024 | 19157 | 0.100 |

Against the bar: gen cut **−15.2% (gen went UP, FAIL)**, score drop **+0.080 (FAIL)**, previously-correct
items degraded **1 (item-7, FAIL)**. But the *why* matters more than the verdict — two findings, the
second methodological and more important:

1. **num_predict=1024 is not binding, so capping there can't reduce gen.** Mean gen_ms rose rather than
   fell, and gen_calls stayed 1.0. A token cap can only shorten or not-bind a generation; it cannot
   lengthen one. So the traces are already mostly **under 1024 tokens**, and the 16.6s baseline gen is
   *slow decode of a moderate-length trace*, not a long one near 4096. Implication: **gen time here is
   decode-throughput-bound (~tokens/sec), not token-count-bound.** Capping tokens is the wrong lever;
   the only caps that would cut ≥20% sit below the typical trace length and would truncate answers.
   The real gen lever is faster inference (a faster/smaller answer model, or a faster serving stack
   such as vLLM), not a shorter budget.
2. **The runs were not reproducible — and the cause is a determinism bug in OUR retrieval, not Ollama.**
   Initially mis-attributed to temp=0 Ollama nondeterminism; that was wrong and is corrected here.
   Diffing the two runs' trace files (which differ *only* in `num_predict`, a flag that cannot touch
   retrieval) showed that for **5 of 10 items the injected fragments (`loaded_paths`) differed**. The
   "degraded" item-7 (0.80→0.00) got a **different set of fragments**, so its answer changed because its
   context changed, not because the model sampled differently. (Item-9's 3× gen_ms swing had *identical*
   injected fragments, so that residual is genuine server-side gen-time variance — secondary.)

   Root cause: hybrid fusion (`caw-index/src/lib.rs`) builds candidates in a `HashMap<StubId,f32>`, then
   stable-sorts by score and truncates to `top_k`. On tied fused scores the stable sort preserves the
   HashMap's randomized iteration order, so a `top_k` cutoff falling among ties admits different stubs
   per process. BM25 (`bm25.rs`) had the identical pattern. **Fixed** by adding a deterministic
   tie-break (`then_with(|| a.id.cmp(b.id))`) at both sort sites. **Verified** by a unit test
   (`bm25::tests::tied_scores_truncate_deterministically`): ten identical-score docs with `top_k=5`
   force truncation among a ten-way tie; the test fails on the pre-fix code (the surviving set varies
   call-to-call, since `search` builds a fresh HashMap with a new random seed each call) and passes
   after. The hybrid-fusion site is the same idiom with the same total-order guarantee.

**Conclusion.** The cap result is *uninterpretable as a quality verdict* — it was measured on a
nondeterministic pipeline. Two separable takeaways survive: (a) gen is **decode-throughput-bound, not
token-count-bound**, so `num_predict` capping is the wrong speed lever (a 1024 cap didn't even reduce
gen); (b) the **eval's run-to-run instability was a real retrieval-determinism bug**, now fixed — which
matters far beyond this experiment, since nondeterministic context injection corrupts every
answer-score measurement and means a user asking the same question twice could get different context.
Once determinism is verified, re-measuring any gen change against the ≤0.03 bar becomes meaningful;
before the fix it was not. Directional (n=10, two runs); the injected-fragment divergence is the robust,
proven part.

## Substrate divergence resolved: spurious stale flags, not a retrieval regression (2026-06-14)

A `caw-bench-graph-eval --diagnose` rerun of the recorded sysdoc n=100 (hybrid recall@10 = 0.84, 2026-06-13) returned **0.37** on the same `subset-medium.sqlite`. Golds resolved fine (no "no stub covers" warnings), so it was a genuine retrieval drop, not a harness fault. Bisected to the index, not the corpus or the eval code: **16,867 of 43,232 stubs were flagged `stale=1`** (including the gold chunks, e.g. `python3-openshot/changelog#chunk0/1/2`) by a since-fixed bug that marked stubs stale on a transient content-read miss. `all_embeddings()` (and the read-only eval path) excludes stale stubs, so ~39% of the corpus — gold chunks among it — never loaded into the HNSW index or BM25, and could not be retrieved.

Repair without rebuilding (the corpus content was never touched): `UPDATE stubs SET stale=0 WHERE stale=1`. After it, the same diagnosis returns **recall@10 0.850 / MRR 0.543** (changelog recall@10 0.125 → 0.839), matching the recorded baseline. The `with_read_only(true)` guard in `graph_eval` already prevents *new* eval runs from re-poisoning the flags; this only cleared the already-persisted damage. Instrument: `docs/skills/caw-dev/scripts/graph-eval.sh` (new) runs this retrieval-only diagnosis with the sysdoc index/QA/source-root and GPU pinning.

## The thesis sweep predates the determinism fix — re-run required before "validated" holds (2026-06-14)

Surfaced during the usefulness/usability/performance assessment. The two findings below, both recorded earlier today, contradict each other and the contradiction was not previously reconciled:

- The headline thesis result ("Full-file expansion — at-scale result", sysdoc recall_on 0.156 vs off 0.029; opencaw 0.232 vs 0.156) was recorded at commit `60c7170`, **2026-06-14 11:21**.
- The retrieval-determinism fix (deterministic tie-break in hybrid fusion + BM25, commit `76180fc`) was made **2026-06-14 13:54** — ~2.5 hours *later* — and its own finding states the pre-fix pipeline injected **different fragments for 5/10 items run-to-run**, that this "corrupts every answer-score measurement", and that re-measuring against the quality bar is "meaningful" only "once determinism is verified".

By the project's own standard, then, the numbers that currently underwrite "thesis validated" were measured on a pipeline since declared non-meaningful for answer-score, and have not been re-confirmed. What survives and what doesn't:

- **The direction is mechanistically robust.** Full-file expansion fires only in `recall_on`'s refinement loop (the model must name a file in its reasoning), so recall_on loads the full named file and recall_off structurally cannot — the *sign* of the on/off gap does not depend on retrieval tie-order.
- **The magnitudes and per-item counts are exactly what the bug scrambles.** The 5.4× gap, "9 improved / 0 regressed", and "items ≥0.5 went 0→6" are per-item answer scores driven by which fragments were injected — the quantity the determinism bug was shown to flip. These are unconfirmed until re-run.

Consequence for "what to work next": the cheapest high-value step is **one confirmatory re-run of the headline sweep on the now-deterministic pipeline** (~25–30 min for an n=40 both-modes sysdoc + n=30 opencaw run via `docs/skills/caw-dev/scripts/bench.sh`), not the gen-phase perf work. The TODO's "gen-perf is the prerequisite for #1" logic applies to repeating a sweep across *many* seeds; it does not apply to a single clean confirmation, which is the gating question for the usefulness verdict and needs the loop only once.

## Bench concurrency now default-on; matches the recorded Ollama ceiling (2026-06-14)

Per the user directive that benchmarking should parallelize by default, `caw-bench` now runs the (item, mode) grid through a capped rayon pool (`--concurrency N`, default 4; `1` restores the serial path). Each task builds its own answer + judge adapters inside the worker (no `dyn ModelAdapter` crosses a thread boundary); the read-only `PrebuiltIndex` (Arc<Mutex<_>> embedder/store/index) is shared and lock-serialized; results are reordered to serial task order before the report is built. Concurrent `runtime.block_on` from multiple rayon threads on the one shared multi-thread tokio runtime works (no deadlock).

Smoke (sysdoc, n=4, recall_on, concurrency 4, groq judge): all four tasks dispatch immediately and run together, scores are unchanged from what a serial run would produce (retrieval is deterministic post-`76180fc`, gen is temp 0), and the per-item ordering in the report is stable. **No wall-clock speedup on this Ollama stack** — exactly the recorded negative result: the four same-model gen calls queue server-side, so per-item `latency_ms` (48–92s here) is inflated by queue-wait, not real per-item cost. Two consequences: (1) under concurrency, per-phase timing *and* the latency aggregate are contention-distorted and must not be read as per-item cost — answer scores and retrieval metrics stay valid; use end-to-end sweep wall time for throughput. (2) The value is architectural: the pool saturates the moment gen moves to a batching serving stack (vLLM/TensorRT), and the groq judge already overlaps. The serial path (`--concurrency 1`) remains for any run that needs the gen/judge/other phase breakdown.
