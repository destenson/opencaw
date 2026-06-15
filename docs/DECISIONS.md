# OpenCAW — Principles and Decisions

**Status:** authoritative reference for implementation decisions. Read `docs/design.md` for the full thesis; read this document before making any significant implementation choice.

---

## What opencaw is

opencaw treats LLM context as a managed workspace, not a container. The core idea is that working-set quality — what is in context and what isn't — is a primary lever on output quality, not just context window size. The system replaces file content with lightweight stubs, recalls full content on demand as the model reasons, and evicts stale material while preserving what was learned.

The differentiating mechanism is using the model's own reasoning trace as the retrieval signal. This sidesteps the need for a separate retrieval query and produces retrieval that tracks what the model is actually thinking about rather than what the user literally typed.

opencaw is a **library** that provides the primitives for context management; applications and deployment targets are built on top of it. Design choices that serve a specific deployment at the expense of generality are wrong.

---

## Invariants

These are properties the system must maintain to function correctly. Any implementation choice that violates an invariant is wrong regardless of how pragmatic it seems in the moment.

**The workspace must reflect the current query.** At any moment, the workspace should contain the content most relevant to what the model is currently reasoning about. Anything that displaces relevant content — session history, noise stubs, runaway retrieval — degrades output quality proportionally. Every subsystem that touches the workspace is responsible for not crowding out content that actually helps.

**Features must run to exist.** A feature behind an opt-in flag doesn't exist in practice. It will never be tested in real workloads, never be measured, and never improve. The only way to know whether something works is to run it. Opt-out flags are fine for measurement (A/B comparisons require a control); opt-in flags mean the feature is effectively unimplemented.

**Retrieval signal must come from meaning, not structure.** Embedding metadata — paths, symbol names, outlines — measures structural similarity. Behavior, purpose, and semantics are in the content. A query about what code does cannot be answered by what it's named.

**Consolidation preserves insight, not events.** The purpose of eviction-time consolidation is to carry forward what was learned from a fragment after it leaves the workspace, so future recalls start with more context than cold. Recording that a fragment was evicted at score 0.52 is not insight. What was relevant about it, what was concluded from it — that is.

**The library does not know your project.** Project-specific configuration — what to skip, what to boost, what counts as noise — belongs in deployment configuration (`.cawignore`, retrieval weights, etc.), not in library code. Hard-coding project paths or heuristics in `should_skip` or retrieval logic is always wrong.

---

## Architectural decisions

These are the significant design choices that define what opencaw is. They are settled. Relitigating them without new evidence wastes time.

### Thinking-trace as retrieval query

The model's own reasoning is the retrieval signal, not a reformulation of the user's query. This works because reasoning traces are more topical and denoised than mixed conversation turns, and step boundaries provide natural trigger points. It also means retrieval tracks the model's actual line of thought rather than the surface form of the question.

The practical consequence: retrieval is triggered at step boundaries, not once per user turn. The orchestrator embeds reasoning steps and matches them against the stub index incrementally.

### Multi-turn orchestration as the baseline

Single-pass retrieval produces a fixed context for the entire response. Multi-turn orchestration — where retrieval can inject new content between generation steps — allows the context to evolve as the model's reasoning evolves. This is the only approach that works for queries that require synthesizing information the model discovers mid-reasoning.

The mid-stream engine-integration path (interrupting generation in-place) is a future optimization, not the current approach. Multi-turn orchestration works with any standard request/response API.

### Stubs replace content; recall materializes it on demand

Files in the workspace are represented as structured stubs until the model's reasoning triggers recall. This keeps the context compact during broad exploration and focuses token budget on content the model is actually reasoning about. The stub carries enough metadata for the model to decide whether to request full content without reading it.

### Hysteresis for eviction

A single threshold produces thrashing: content loaded because it was marginally relevant gets evicted one step later, then re-loaded, then evicted. Two thresholds — load at 0.7, unload at 0.4 — create a band in which content stays once admitted, preventing the oscillation. The gap between them is the stability zone.

### Consolidation as episodic-to-semantic transfer

When a fragment is evicted, the stub absorbs a synthesized note of what was learned during the loaded period. This is analogous to how episodic memory consolidates into semantic memory: the raw experience (the fragment content) is gone, but the derived understanding (what mattered about it) persists and informs future recalls. Without this, every session starts cold regardless of prior work.

The consolidation synthesizer must use an LLM when one is available. A mechanical fallback (recording the eviction score) is not consolidation — it is bookkeeping that gives the appearance of consolidation without the function.

### Session history serves continuity, not retrieval

Prior conversation turns are not workspace stubs. They serve a different purpose — helping the model maintain coherence across a session — and must be handled differently. Model responses use the same vocabulary as current queries and will outrank workspace stubs in similarity search if allowed to compete. Session history is injected as a budget-capped fixed block, not as a retrieval candidate.

Prior session responses (from earlier sessions, not the current one) are compressed before embedding. A 400-token response that outranks workspace stubs is not continuity — it is the current session reading from its own prior output rather than from the actual project.

Concretely in the orchestrator: the `vector_index` field is the session-history index only (this session's turns plus any prior-session stubs from `with_session`). All *corpus* recall — the initial query, probes, and thinking-trace recall — goes through the `retriever`. Trace recall therefore queries the retriever for corpus matches and `vector_index` only for session history, and callers must not pre-populate `vector_index` with the corpus. (An earlier bench wiring left that index empty while trace recall searched only it, which made thinking-trace recall inert against the corpus.)

### Intent classification changes retrieval strategy

The intent classifier exists to adapt retrieval to query type. An inventory query ("how many todos are left?") requires document-level coverage of TODO.md; similarity search will return the single most-matching chunk, which is never sufficient to answer a count. A status query needs the overview documents. An explanation query needs documentation in full-content mode, not outline-only. Classification that doesn't change behavior is waste.

### caw-server exposes a read-only retrieval diagnostic route

`caw-server` was originally intended as a strictly OpenAI-compatible surface (only `/v1/chat/completions`), so that any OpenAI client could point at it unmodified. We added one deliberate exception: `/v1/retrieve`, a read-only route that runs the identical retrieval and returns the ranked candidates as JSON (per-candidate fused score, path, token cost, and disposition: admitted / clamped / budget_full / content_miss), without forwarding to any model.

Rationale: the proxy's only window into retrieval quality was a DEBUG log line and the downstream model answer. When an answer looks wrong you cannot tell from that whether the relevant chunk was ranked out of the candidate pool or ranked in but clamped out by the token budget — the two have opposite fixes. The diagnostic route surfaces that distinction directly. It is read-only, never mutates request flow, and reuses the same `rank_candidates` + clamp logic the chat path uses (via a shared `retrieve_scored`), so what it reports is exactly what the proxy would inject — not a parallel reimplementation that could drift. The `caw-dev` skill's `retrieve.sh` consumes it.

This is a conscious deviation from "pure OpenAI surface," made on explicit request, and scoped to diagnostics. It does not open the door to adding orchestrator/probe/multi-pass endpoints to the proxy — that engine stays in the CLI.

### Prebuilt-index consumers open the store read-only

A consumer that serves or measures over a prebuilt index and has no reingest worker attached must open the `SqliteStubStore` with `with_read_only(true)`. `get_content` marks a stub `stale = 1` (persisted) when it can't read the body — a missing source, a wrong `corpus_root` that makes a present file appear absent, or an mtime mismatch. Without a reingest worker nothing ever clears that flag, and a stale stub is excluded from both retrieval halves (`all_embeddings` and the BM25 build both filter `stale = 0`), so the marking silently and permanently degrades every later run against the shared index.

This bit the bench and CLI: a single run with a misconfigured `corpus_root` poisoned a frozen benchmark index (`subset-medium.sqlite`) with ~26% spurious `stale = 1` flags — the benchmark corpora never change, so they are never legitimately stale, which means any stale flag on them is a marking bug, not a real edit. The read-only flag gates the `mark_path_stale` calls so a transient/misconfigured read can't persist staleness. All four prebuilt-index consumers — `caw-server`, `caw-bench` (`--index` path), `caw-cli` retrieval store, and `graph-eval` — now open read-only. The CLI's separate consolidation store stays writable; an in-process pipeline that legitimately owns a reingest worker (none today) would attach one via `with_reindex_queue` instead of opening read-only.

---

## Implementation principles

When making a choice this document doesn't explicitly cover, apply these in order.

**Prefer measurement over guessing.** When a threshold or configuration value is uncertain, instrument it and measure. The specific values that exist (load=0.7, 20% history cap, 200-token convergence threshold) are starting points based on observed failure modes, not empirically validated optima. Don't change them without data; don't treat them as immutable.

**Default to on.** If a feature improves recall quality or context curation, it runs by default. The cost of a bad default is proportional to how long it runs before anyone notices. The cost of a missing default is that nobody ever notices.

**Library vs. deployment.** Ask whether a choice belongs in the library or in deployment configuration. If it depends on the project, it belongs in configuration. If it applies to every project using opencaw, it belongs in the library.

**Serve the library target.** The library is the primary deployment target. Middleware and engine plugins are future work. A choice that makes the library harder to embed in order to make a future deployment target easier is premature.

**When in doubt, ask.** The first implementation choice is often wrong because no single pass has a complete picture of how all the subsystems interact. Write out the options and their trade-offs and ask rather than picking one. This is not a sign of weakness — it is the correct response to genuine ambiguity.

---

## Settled configuration details

These are specific values and constraints that follow from the architectural decisions above. They are not arbitrary — each has a failure mode it prevents — but they are also not sacred. Change them when measurement supports a different value.

**Session history budget cap: 20%.** Enough for continuity signal; not enough to crowd out workspace stubs for any realistic query.

**Prior session response compression: ≤50 tokens.** A synthesis of what was asked and what was concluded. Not the full response text, which reads like authoritative documentation to the similarity search.

**Multi-pass convergence: stop when an iteration adds <200 tokens or zero new unique stubs.** Either condition means the loop is no longer finding useful content.

**Consolidation note cap: 2 per stub, no nesting.** Notes that grow without bound defeat the purpose. A stub that has been evicted many times gets "Evicted N times; most recent: [summary]" — not an accumulation of all prior notes.

**Retrieval thresholds: load=0.7, unload=0.4, hysteresis floor=0.35.** Starting points pending `HysteresisAnalysis` sweeps. The floor applies to stubs that have been admitted at least twice in a session (indicating persistent relevance).

**Score modifiers:** `.md` stubs get 1.5× for explanation queries (documentation answers those better than implementation); `caw-bench/src/` stubs get 0.5× for non-benchmark queries (0.25× for harness binaries specifically); stubs evicted 3+ times with no engagement get 0.5× (noise attractors that keep consuming admission slots).

**Minimum content filter:** stubs with `summary.trim().len() < 15` are excluded from retrieval. `token_estimate` reflects file size, not how much content the stub actually contributes — don't use it as a content-quality proxy.

**Per-source chunk cap: 3 (was 1).** A multi-chunk document holds different answers in different chunks; a hard one-chunk-per-source rule silently drops the answer-bearing chunk whenever a higher-scoring chunk of the same file is admitted first. Measured: on the sysdoc n=40 answer-quality sweep, ~40% of items (16/40) were losable purely to this — the gold file was loaded but the wrong chunk was injected, and the model honestly reported the answer wasn't present. The cap is now a per-source *count* (`DynamicRecallConfig::max_chunks_per_source`, mirrored by `caw-server::MAX_CHUNKS_PER_SOURCE`), not a ban. Enforced at three coordinated surfaces that previously all assumed one-per-source: the orchestrator's `load_fragments` admission, the proxy's clamp (`Disposition::DuplicateSource` now means "exceeded the cap", not "any duplicate"), and `CompletionRequest::format_workspace`, which no longer collapses to the first chunk per source + a `[+N more]` note but renders every admitted chunk (per-source limiting is the producer's job, not the renderer's). The value trades against the original flooding failure (B1 — a 100+ chunk file flooding the workspace): 3 surfaces the answer chunk of a normal changelog while bounding a pathological file; the token budget remains the hard ceiling. Starting value pending the post-fix re-run; configurable.

**Performance optimization — levers and non-levers (2026-06-14).** After the judge-phase fix (groq judge, ~19×), the eval loop's remaining cost is per-item *generation* plus a one-time in-memory HNSW build. Three decisions:

- **Cross-item concurrency is the default — saturate the system within reason (revised 2026-06-14, user directive).** The bench runs items concurrently with a capped worker count by default; `--concurrency N` overrides, `--concurrency 1` restores the serial path. This *supersedes* the earlier "serial stays default, concurrency opt-in" call. Reason for the change: parallel-by-default is the correct runner architecture — it costs little today and is forward-ready for a gen stack that genuinely batches (vLLM/TensorRT, the primary gen lever), and the groq judge already parallelizes as a remote call. Carry the measured reality honestly: on the current single-Ollama two-GPU setup the *gen* speedup is capped at ~18% because Ollama serializes same-model requests (4-concurrent scaled ~linearly; see `docs/findings.md`, "Negative result: cross-item concurrency"). The user's correction to that finding: a second resident instance does **not** require a second Ollama process or GPU pinning — Ollama already spans both GPUs and loads a *different model name* as a separate instance; the `-worker2` alias failed only because an alias of the same model dedups to one instance. A sweep uses one fixed answer model, so same-model gen still serializes — the ~18% ceiling stands until the gen stack changes — but a local judge or auxiliary model on a different name runs without contending. When concurrency > 1, per-item phase timing (gen/judge/other) is invalid (shared judge counter + wall-time contention) and must be gated to the serial path; aggregate wall-clock and answer scores stay valid. The default cap is a sane small number (overridable), "within reason" so a runaway fan-out can't OOM the GPU or the embedder.
- **Gen reduction is the primary lever** (dominant cost; the only one that also speeds the proxy/CLI *product*, not just the bench): make the hardcoded `num_predict` (currently 4096 in the Ollama adapter) configurable, and optionally skip the refinement completion when the first answer is already grounded. Because gen *is* the thesis mechanism, any gen-reduction change is gated on a **pre-committed quality bar**: on matched items (same workload, same seed, recall_on), accept a change only if it cuts `gen_ms` by ≥20% AND the recall_on `answer_score` aggregate does not drop by more than 0.03 absolute (and no previously-correct item becomes a truncated/empty answer). If quality drops more, it is rejected for production and may ship only as an explicit dev-sweep fast mode. Baseline is re-measured fresh alongside the treatment (the n=2 numbers in findings are too noisy to use as a fixed bar).
- **HNSW persistence is a low-risk product-startup win** (not a meaningful sweep win — the build is one-time, amortized over a sweep). Serialize the built `instant_distance::HnswMap` (serde feature) so process start loads the graph instead of rebuilding from 43k embeddings; the "negligible for low thousands of documents" assumption in `hnsw_index.rs` no longer holds at 43k.
- **Judging is decoupled from generation (two-phase eval, 2026-06-14).** `run_item` no longer judges. The generation phase produces unscored `ItemResult`s (model-judged `JudgeAgainst` items carry `judge_pending = true`; local `ContainsNeedle` scoring stays inline since it's a string check, not a model call). After all generation completes, a single post-gen phase judges every pending item in parallel (each judge call gets its own adapter + timing counter, so per-item `judge_ms` is that call's own latency and stays valid under parallel judging). Rationale: generation never blocks on the judge, and the judge — a remote Groq call that parallelizes freely — fans out at the end instead of serializing one-per-item into the gen loop. `--trace-out` is written during generation, so it persists raw answers *before* scoring (the re-judgeable artifact); the report is built after the judge phase, so its scores are complete. This is step one toward a standalone re-judge path (TODO:122) and the paired-per-item-delta measurement work (TODO:123) — both want answers persisted independently of any one judge run.
- **Standalone re-judge is a `--judge-trace` flag, not a separate binary (2026-06-14).** `caw-bench --judge-trace <trace.jsonl>` short-circuits generation: it loads `ItemResult`s persisted by `--trace-out` and re-scores them against the current judge config, reusing `judge_all` + `build_report` so a re-judged report has the same shape (including paired deltas) as a generated one. A flag on the existing binary — rather than a new `caw-bench-judge` crate — keeps the judge/report wiring in one place and avoids duplicating the CLI surface. Two sub-choices: (1) it re-scores *every* judge-scored item regardless of the persisted `judge_pending` flag, since the point is to re-score (a trace from a completed run has `judge_pending` already cleared); (2) it identifies judge-scored items by a non-empty `reference_answer` rather than persisting the `Scoring` enum — that field is empty exactly for `ContainsNeedle` (local needle scoring, which keeps its inline score) and non-empty for `JudgeAgainst`, so it's a structural property of the record, not an inferred heuristic. This enables scoring the same answers with different or repeated judges to isolate the judge's own contribution to score variance (TODO:122/123), at zero generation cost.
- **Initial-load admission is coverage-first, not greedy score order (2026-06-14).** `load_fragments` admitted fragments in pure score-descending order until the token budget or per-source cap stopped it. That let one multi-chunk source take the head of the workspace and exhaust the budget before a lower-ranked but distinct source was reached — the "gold buried in load order" failure (BUGS "Retrieval", mechanism 1). Fix: `coverage_first_order` reorders the ranked hits so every distinct source's best chunk is admitted before any source's second chunk, then depth follows in score order. It's a stable, deterministic, source-agnostic reorder applied at the initial-load call site (where the source path is still known); the per-source cap and token budget are unchanged. Measured on a deterministic `--only-mode off` sysdoc n=14 run (off-mode does only the initial load, so its loaded set is independent of the answer model): mrr +0.005, recall@k and precision@1 unchanged, 2/14 items improved, none regressed — the reorder changes the loaded set only when flooding would have occurred (12/14 were already source-diverse). Effect is small on sysdoc; the bug was observed on the opencaw workload, where impact should be measured next. Mechanism 2 (thinking-trace re-query drift) is not addressed by this change.

  Measured on opencaw next (2026-06-15), deterministic `--only-mode off` `--concurrency 1`, A/B of HEAD-with-fix vs the same commit's parent over the 29 items both runs scored: precision@1 unchanged (0.207), recall@k −0.017, mrr +0.011. The reorder nudged mrr up on 13/29 items by lifting a gold chunk a rank or two, but promoted gold to rank 1 on none of them (precision@1 flat) and pushed gold out of the top-k window on one item (`qa_028_eviction_trigger`, recall 0.50→0.00 — breadth-first admission can evict a deeper-but-gold chunk of a high-scoring source). **Net: coverage-first is neutral-to-marginal on opencaw with at least one regression — mechanism 1 (load-order flooding) was not the dominant cause of the opencaw recall gap.** The binding constraint is upstream of admission ordering: candidate-set recall@k is only ~0.67 (gold is absent from the retrieved candidates for a third of items, so no admission policy can surface it) and precision@1 is ~0.21. That is a retrieval-ranking-quality problem (embedding / chunking / hybrid weighting), not an admission-ordering problem. The fix is kept (it costs nothing and helps mrr slightly) but is not the path forward; raising candidate recall@k is.

- **The eval/cli measure pure cosine; the dogfood proxy uses hybrid — close that gap first (2026-06-15).** `caw-bench` (and the cli) instantiate `SemanticRetriever` (pure cosine) at `runner.rs:338,390`, while `caw-server` ranks with `HybridRetriever`. Retrieval-only A/B on the sysdoc n=100 chunk gold set (`caw-bench-graph-eval`, no answer model, deterministic) — cosine → hybrid(`divide_total`): recall@1 0.27→0.37, recall@5 0.41→0.71, recall@10 0.48→0.85, recall@20 0.54→0.92. So the opencaw candidate-recall ceiling (~0.67) measured above is largely an artifact of the eval running the weak retriever the product would not use. **Highest-leverage next step: wire bench/cli onto the same hybrid retrieval the proxy already has** (also required by the repo principle that cli/bench/proxy share retrieval code), then re-measure the opencaw sweep before doing any embedding/chunking work. Expected to lift recall@k substantially for free.
- **Do not replace `divide_total` hybrid fusion with RRF (2026-06-15, contradicts a prior plan).** A standing plan (recorded in the retrieval-quality memory note) was to swap the "prototype" `divide_total` min-max fusion for RRF. Same A/B as above, hybrid `divide_total` vs hybrid `rrf`: recall@5 0.71 vs 0.57, recall@10 0.85 vs 0.63, recall@20 0.92 vs 0.66 (they tie at 0.94 only at @50). `divide_total` ranks gold strictly better at every depth that matters. If fusion is revisited, `divide_total` is the baseline to beat — not a problem to be replaced by RRF.
- **Measure initial-load retrieval changes on `--only-mode off`, never `on` (2026-06-14).** A before/after of a retrieval-ordering change run on `--only-mode on` showed a spurious −0.066 mrr "regression" that the deterministic off-mode run did not: in recall-on, the thinking-trace and probe passes load additional fragments driven by the model's generated reasoning, which is nondeterministic (and was run at `--concurrency 4`), so the recall-on loaded set — and its recall@k/mrr/precision@1 — varies run-to-run independent of any code change. recall-off does only the initial load with no answer feedback, so its loaded-set metrics are deterministic and isolate exactly an initial-load change. Compare per-item matched on `item_id`, not aggregate means, since a different item can drop out (degenerate answer) in each run.
- **Multi-seed generation rejected; serial generation is deterministic (verified 2026-06-14).** Considered running the grid across multiple seeds to average out answer-model nondeterminism. Rejected after verifying the premise: two `--concurrency 1` runs of the same items produced bit-identical answers (4/4), confirming the BUGS claim that the serial path is reproducible. At `--concurrency 1` + temperature 0 the answer is deterministic, so repeating with different seeds yields identical output — multi-seed adds no signal there. Nondeterminism only appears at `--concurrency > 1` (batched-matmul FP non-associativity), where you've already given up reproducibility. So the statistical-power levers are: (1) the deterministic serial path for measurement runs, (2) `--judge-trace` to average judge noise, and (3) **more QA items** — between-item variance is the dominant term and only more items shrink its standard error; seeds do not. Larger-n runs are now fast (not free): a groq answer model (`CAW_BENCH_ANSWER=groq` in `bench.sh`, defaulting to the smallest groq chat model) runs ~2–3s/item vs tens of seconds locally, so wall-time is no longer the blocker — but groq bills per token, so prefer the smallest model and bounded `--limit` runs and don't run the full set casually. Two caveats carried honestly: (a) the recall effect is model-specific, so a groq-answer run is for fast instrument iteration and directional large-n reads, while the headline "does recall help" number for the dogfooding target should be pinned to the target model on the no-API-cost local path; (b) a smaller answer model answers worse, which is acceptable for instrument work but not for the headline number.
- **A small answer model is a primary engineering target and the more sensitive QA instrument (2026-06-14).** Two practical reasons to run benches on a small answer model (e.g. groq `llama-3.1-8b-instant`), not just cost: (1) a small, cheap model is a real serving target for the product — good curation is what makes it viable; (2) a small model can't paper over a cluttered or gold-buried workspace the way a large model does (more parametric knowledge, noise-tolerant reasoning), so retrieval-quality regressions show up faster and larger in its recall-on − recall-off numbers. That makes it the better instrument for *catching curation defects*, and means the recall regression (recall-on burying gold, BUGS "Retrieval") degrades small-model output most — which is why fixing it is the priority and small-model benches surface it fastest. This is QA/quality engineering, not validation of the approach.

---

## Open questions

Genuinely unresolved. Present options with trade-offs; don't choose unilaterally.

- **Hysteresis threshold calibration.** The values above are starting points. `HysteresisAnalysis` sweeps against real session data haven't been run. The right values are workload-dependent.
- **Insertion order of recalled content.** Relevance-ranked, reverse-relevance, and stub-order are all viable. Requires controlled comparison.
- **Cooperative vs. transparent mode selection.** Whether to explain the recall mechanism to the model (cooperative) or keep it invisible (transparent) depends on model capability. The heuristic for choosing automatically is unresolved. Default to cooperative.
- **Adaptive chunking strategy.** Token-based is the current approach. Structural boundaries (function/section) may be better for code. Requires a benchmark comparison before changing.

---

## Decision protocol

1. Check `docs/design.md` §11 for settled answers to architectural questions.
2. Check the invariants and architectural decisions above. If a choice violates an invariant, it's wrong. If it contradicts a settled architectural decision, it needs a strong justification.
3. Apply the implementation principles.
4. If still ambiguous: stop, write out the options and trade-offs, and ask. Don't implement a compromise behind a flag.
