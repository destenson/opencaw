# OpenCAW Codebase Review

> **Status: Frozen snapshot (2026-05-02).** A point-in-time review, kept for history. Not maintained — do not read as current state. See [docs/README.md](../README.md) for the live docs index.

**Date:** 2026-05-02
**Branch:** try1
**Reviewer:** codebase-review skill

---

## Executive Summary

OpenCAW is a substantially complete v0.1 Rust library implementing thinking-trace-driven recall with eviction, consolidation, and context curation. The core loop — ingest → embed → retrieve → orchestrate → evict/consolidate — works end-to-end and is covered by a real integration test. The primary gap is that `caw-eval`'s `SessionEvaluator` is never wired into the orchestrator or CLI at runtime, meaning the measurement infrastructure that should drive threshold tuning is built but idle. The next action is wiring the evaluator into the orchestrator's `run_turn` loop so calibration numbers can be collected from actual runs.

---

## Implementation Status by Crate

### caw-core — Working

The shared type layer is solid. `CawError`, `Stub`, `RecallFragment`, `TokenBudget`, `ModelAdapter`, `EmbeddingProvider`, `StubStore`, `VectorIndex`, `Retriever`, and `BudgetScheduler` traits are all defined with appropriate defaults. `CompletionRequest::format_workspace` and `format_workspace_with_guidance` produce correct provenance-tagged output in both XML (Anthropic) and bracketed (OpenAI-protocol) formats.

`QueryIntent` with `majority_vote`, `guidance_lines`, and `from_classifier_response` is well-implemented. The JSON extraction in `extract_json_object` correctly handles string escapes and nested objects.

`AugmentationSignals` is defined and `QueryIntent::augmentation_signals()` produces it, but no caller outside of `caw-core`'s own tests calls `augmentation_signals()` — the orchestrator and CLI only use `guidance_lines()`. The augmentation-to-retrieval wiring described in TODO.md as an open item is genuinely absent.

Notable: `Range::apply` for `Range::Tokens` splits on whitespace rather than BPE tokens, which diverges from the type's name and from what cl100k-tokenized systems would expect. This is not a crash but produces wrong results for token-addressed retrieval.

10 unit tests, all passing.

### caw-ingest — Working

`IngestionPipeline` with cl100k tokenizer, tree-sitter outlines, adaptive chunking, and both deterministic and LLM summarizers is functional. `SourceDocument::from_path` is clean. `chunk_document` implements linear greedy chunking with structural boundary snapping.

No dedicated unit tests. All coverage comes through the end-to-end integration test. `tree_sitter_outline.rs:12` has a bare `.expect("language version mismatch")` — this will panic at startup if the linked tree-sitter grammar ABI version disagrees with the tree-sitter runtime, with no way for the caller to recover.

Ingestion is synchronous and batch-only. The background indexer with lazy fallback from the design doc (TODO.md item) is unimplemented. For the current CLI use pattern this is acceptable.

### caw-index — Working

`SemanticRetriever`, `HybridRetriever`, and `InMemoryIndex` all implement the `Retriever` trait. BM25 is integrated into `HybridRetriever` with min-max normalized fusion and configurable weights. `HnswVectorIndex` and `FlatVectorIndex` both implement `VectorIndex`.

`SqliteStubStore` is production-quality: WAL mode, synchronous=NORMAL, prepared-statement batch inserts, staleness detection with reindex queue integration, and inline migration for schema changes. Content is stored as file byte ranges rather than duplicating text in the DB — a correct design decision.

The `SemanticRetriever::insert` signature silently ignores the `content: String` parameter from the `Retriever::insert` trait default (it recomputes embedding text from stub fields only). `HybridRetriever::insert` does use the content for BM25 indexing. The trait design gives callers no indication that content is used by one but not the other.

3 unit tests (SqliteStubStore staleness paths), all passing.

### caw-adapters — Working

All six adapters compile and implement `ModelAdapter`. The pattern of creating a shared `tokio::Runtime` and using `block_on` is intentional and documented in SCOPE.md — works for library use, would panic inside an existing async context.

`OllamaAdapter::capabilities()` hardcodes `supports_hidden_reasoning: true` for all Ollama models with a TODO comment at line 198. This means all Ollama models get probe/note injection instructions even if they can't follow them. Querying `/api/show` at construction time would give the correct model metadata.

`AnthropicAdapter::capabilities` hardcodes `supports_visible_reasoning: false` for all Claude models. Combined with the orchestrator's gate at `build_system_prompt` (only injects cooperation instructions for visible-reasoning models), this means the full thinking-trace recall path is disabled for Anthropic models. Claude uses probes instead, which requires the model to emit `<probe>` markers explicitly.

`ClaudeCodeAdapter::complete` at line 354 has `.expect("loop exits with final_result set on success path")`. The invariant is correct but `unreachable!()` would be clearer about the intent.

2 unit tests (TracingAdapter), passing.

### caw-orchestrator — Working

`DynamicRecallOrchestrator` implements the full multi-pass loop: initial retrieval, ambiguity gate, candidate list injection, iterative thinking-trace/probe extraction, relevance decay, term-overlap refresh, budget-aware eviction, consolidation note generation, and session history. The implementation matches the design doc closely.

`MechanicalConsolidation` (the default) generates informative but shallow notes. `LlmConsolidation` exists and degrades gracefully to mechanical on empty output but is not the default. SCOPE.md marks richer consolidation notes as "should have" and open.

Degradation monitoring is fully implemented: per-component health tracking, tiered fallback, and probe rate limiting. 9 unit tests cover all three tiers and recovery paths.

`reset_session` requires `P: Default`, which means any custom non-defaultable provenance store can't be reset via this method. Latent ergonomic constraint.

`caw-orchestrator/src/session.rs` implements its own Gregorian date math (`days_to_ymd`, `is_leap`) for session filenames instead of using `chrono`. The result is probably correct but the comment "good enough for filenames, no leap second precision needed" overstates the problem — the actual risk is off-by-one errors on month boundaries.

**Important duplication:** The ambiguity gate, fragment loading, relevance decay, term-overlap refresh, and eviction logic appear in both `DynamicRecallOrchestrator::run_turn` (dynamic.rs:235-373) and the CLI's interactive loop (main.rs:695-900). The two copies have diverged: the orchestrator integrates session history recall; the CLI manages session history via its own parallel `session_content` HashMap. Any bug found in one may not be fixed in the other.

9 unit tests + 1 integration test, all passing.

### caw-curation — Partial

`HistorySummarizer` (extractive + LLM), `ToolOutputCompressor` (extractive + LLM), and `SystemPromptBudget` are implemented. The `CurationPipeline` wires them together. The CLI uses this when `--curate` is passed.

Few-shot management (design doc §8.1) is not implemented — callers can use `Tokenizer::count_tokens` directly but there is no framework-level utility. SCOPE.md marks this "nice to have."

No dedicated unit tests. All coverage comes through the CLI path.

### caw-eval — Built, Not Wired

`SessionEvaluator`, `RecallMetrics`, `FalseRecallMetrics`, `HysteresisAnalysis`, `ContextEfficiency`, and `CooperationMetrics` are all implemented in `caw-eval/src/session.rs` and `metrics.rs`. The data structures and computation logic look correct.

The problem: `caw-eval` is only a dependency of `caw-bench`. `DynamicRecallOrchestrator` has no `caw-eval` dependency and records no eval events. The `caw-bench` runner (`runner.rs:601`) reimplements the false-recall rate heuristic inline with a comment saying it "matches the FalseRecallMetrics default threshold in caw-eval (0.15)" — which confirms the duplication is known. This is the most significant structural gap: the eval infrastructure is built, the event points exist in the orchestrator, but nothing connects them.

No unit tests in this crate.

### caw-bench — Working

The benchmark harness with NIAH, opencaw, and sysdoc workloads is functional. `caw-bench-build-index` for streaming GPU-pipelined index construction is well-engineered. `intent_bench.rs` is fully implemented with per-field TP/FP/TN/FN scoring, ensemble support, and leaderboard display. `caw-bench-tune` aggregates results and emits recommended model configs.

The sweep binary iterates over parameter grids for threshold tuning. No sweep results have been generated yet — the `bench-results/` tree contains individual runs, not calibration sweeps.

No unit tests in this crate.

### caw-transform — Working

`PromptTransformer` handles markdown link and `@path` references. Fenced blocks with `path=` and bare-path regex (from the design doc) are not implemented — acknowledged in TODO.md. `extract_probes`, `extract_thinking_steps`, and `extract_annotations` use `LazyLock<Regex>` with compile-time-verified patterns; the `cap.get(n).unwrap()` calls on match captures are safe because the regex structure guarantees the capture groups.

6 unit tests, all passing.

### caw-server — Working (documentation is stale)

Contrary to the README and SCOPE.md descriptions of "scaffold only," `caw-server/src/lib.rs` is a functional OpenAI-compatible proxy that retrieves context and augments the last user message before forwarding to an upstream model server. It uses Candle + CUDA for embeddings, supports both flat and HNSW retrieval, and handles streaming passthrough correctly.

This is a completed v0 middleware deployment target. The documentation describing it as scaffold is stale and should be updated.

### caw-cli — Working

The CLI is feature-complete for v0.1: ingestion, indexing, intent classification with ensemble support, recall loop, curation, session history, and all adapter types. The `--augmentation-prompt` mode and `--intent-model` ensemble are recent additions.

The CLI implements its own recall loop rather than delegating to `DynamicRecallOrchestrator::run_turn`. This duplication is the primary maintenance liability (see above under caw-orchestrator).

---

## Test Results

All 31 tests pass. Zero failures.

```
caw-adapters:    2 tests (TracingAdapter)
caw-bench:       0 tests
caw-cli:         0 tests
caw-core:       10 tests (QueryIntent, workspace format, candidate list, reindex queue)
caw-curation:    0 tests
caw-eval:        0 tests
caw-index:       3 tests (SqliteStubStore staleness paths)
caw-ingest:      0 tests
caw-orchestrator: 9 unit tests (degradation tiers, recovery, thinking trace, probe recall)
                   1 integration test (end-to-end recall loop with MockAdapter + HashEmbedder)
caw-server:      0 tests
caw-transform:   6 tests (probe/thinking/annotation extraction, range parsing, transformer)
```

The test suite is light on the most algorithmic parts: `HybridRetriever` fusion, `SessionEvaluator` metric computation, `ChunkingConfig` boundary snapping, and `LlmConsolidation` fallback behavior are untested. What exists is sound — the degradation tests cover recovery hysteresis correctly, and the integration test exercises the full admission path.

---

## Code Quality

### Debt markers

From `debt_report.txt` (generated 2026-04-19): 28 total debt markers in source. Key items by file:

- `caw-bench/src/intent.rs:244` — TODO for more varied bench cases with mixed intent signals
- `caw-adapters/src/ollama.rs:198` — TODO to query `/api/models` for per-model capabilities
- `caw-core/src/lib.rs` open items — `AugmentationSignals` not wired to retrieval path

### unwrap/expect in production code

**Safe patterns (not risks):**
- `LazyLock<Regex>` `.unwrap()` calls — pattern is verified at compile time; the unwrap is cosmetic
- `cap.get(n).unwrap()` after a successful regex match — capture group existence is guaranteed by the pattern structure

**Real concerns:**
- `caw-ingest/src/tree_sitter_outline.rs:12` — `.expect("language version mismatch")` panics the process on grammar ABI mismatch with no recovery path
- `caw-core/src/reindex.rs:220-221` — `.unwrap()` on `JoinHandle::join()`; a panicking worker becomes an orchestrator panic rather than a logged error
- `caw-adapters/src/claude_code.rs:354` — `.expect("loop exits with final_result set on success path")`; the invariant is correct but `unreachable!()` would be clearer
- `caw-transform/src/recall.rs:126` — `apply_range(...).unwrap()` called in a context that is not inside `#[test]`; the function returns `Result` so this discards errors in non-test callers

### Structural issues

**CLI/orchestrator recall loop duplication** is the most significant quality issue. The same eviction, relevance decay, term-overlap refresh, ambiguity gate, and fragment loading logic exists in two places. They have visibly diverged (session content handling differs). A regression fixed in one place is a regression waiting to be found in the other.

**`caw-eval` disconnected from runtime.** The bench runner reimplements the false-recall heuristic inline and explicitly references caw-eval's threshold constant in a comment. The wiring is absent; the intent is there.

**`Range::Tokens` uses whitespace split.** The implementation counts whitespace-split words, not BPE tokens. Since the rest of the stack uses cl100k for token accounting, a `Range::Tokens` address refers to a different position than any other token count in the system. No current production path uses this range type, so it's latent rather than active.

**`truncate_str` is duplicated.** Defined identically in `caw-orchestrator/src/dynamic.rs:658` and `caw-orchestrator/src/consolidation.rs:107`. Both are private to their modules.

---

## PRP Status

No `PRPs/` directory. No PRP documents found in the project tree.

---

## Technical Debt — Top Items

**1. CLI/orchestrator recall loop duplication (high impact, medium effort)**
`caw-cli/src/main.rs` implements its own recall loop (~200 lines) rather than delegating to `DynamicRecallOrchestrator::run_turn`. The two loops have diverged in session content handling and candidate list injection. Any enhancement to the orchestrator must be manually mirrored in the CLI. Collapsing the CLI loop into `run_turn` calls removes the duplication and makes every CLI user benefit from orchestrator improvements automatically.

**2. SessionEvaluator not wired to orchestrator (high impact, low effort)**
All the event emission points exist in `DynamicRecallOrchestrator`: recalls in `load_fragments`, evictions in `evict_stale_fragments`, probes in `process_probes`, annotations in `process_annotations`. Adding `evaluator: Option<SessionEvaluator>` to the orchestrator and calling the record methods at existing event points would take roughly an hour and unlock the threshold tuning story that SCOPE.md marks "must have" (blocks usefulness).

**3. AugmentationSignals not wired to retrieval path (medium impact, medium effort)**
`QueryIntent::augmentation_signals()` is defined and the CLI runs the intent classifier on every query. The TODO in TODO.md is accurate: `is_status_request` should bias retrieval toward status/todo docs, `is_results_request` toward bench result files, etc. The mechanism for this (a pre-retrieval augmentation step) doesn't exist. This is a meaningful quality improvement for the "what should I work on" class of queries that the CLI is commonly used for.

**4. Hardcoded Ollama model capabilities (medium impact, low effort)**
`OllamaAdapter::capabilities()` returns `supports_hidden_reasoning: true` for all models. Models that can't follow marker instructions produce probe/note tags in their output instead of using them as intended. A single `/api/show` call at construction time would give actual model metadata. The TODO comment at ollama.rs:198 identifies the fix precisely.

**5. Range::Tokens uses whitespace split (low impact, low effort)**
`Range::apply` for `Range::Tokens` splits on whitespace, not BPE tokens. This should either be renamed `Range::Words` to match the implementation, or fixed to use cl100k. No production path currently generates token-addressed ranges, so this is latent rather than an active bug.

**6. truncate_str duplication (low impact, trivial)**
Identical private function defined in `dynamic.rs:658` and `consolidation.rs:107`. One shared utility in a common module.

**7. timestamp_str Gregorian arithmetic (low impact, low effort)**
`caw-orchestrator/src/session.rs` rolls its own Gregorian date arithmetic for session filenames. The workspace already has `chrono` available through other crates. Using `chrono::Utc::now().format(...)` would be shorter and the correctness would not need to be audited.

**8. caw-server documentation is stale (low impact, trivial)**
README calls caw-server "scaffold only." It is a working OpenAI-compatible retrieval-augmentation proxy. Update the description to accurately reflect what it does and what its current limitations are (synchronous, Candle-only embedder, no multi-pass orchestration).

---

## Recommendation

**Next action:** Wire `SessionEvaluator` into `DynamicRecallOrchestrator::run_turn`. Add `evaluator: Option<SessionEvaluator>` to the orchestrator struct, call `record_recall`/`record_eviction`/`record_probe`/`record_annotation` at the existing event points in `load_fragments`, `evict_stale_fragments`, `process_probes`, and `process_annotations`. Expose a `take_evaluator` method. This is a few dozen lines of wiring, has no behavioral impact, and unlocks the one SCOPE.md "must have" that is unaddressed.

**90-day roadmap:**

Weeks 1-2: Wire `SessionEvaluator` into the orchestrator. Run the opencaw and sysdoc benchmarks in recall-on mode and collect `HysteresisAnalysis` output for the default `load=0.7 / unload=0.4` thresholds.

Weeks 3-4: Use the collected data to evaluate whether the defaults are appropriate for the sysdoc workload. Update `RecallThresholds::default_hysteresis()` if the data supports different values. Document what workloads the defaults target.

Weeks 5-6: Collapse the CLI recall loop into `DynamicRecallOrchestrator::run_turn`. The CLI becomes a thin wrapper: build adapters, call `run_turn`, format the response. This removes the duplication and makes curation pipeline integration cleaner.

Weeks 7-8: Wire `AugmentationSignals` to the retrieval path in the orchestrator. For `is_status_request`, inject a `git status` snapshot. For `is_results_request`, bias retrieval toward paths matching `bench-results/`. For `is_next_step_request`, proactively load TODO.md. The intent classifier is already running; this step acts on its output.

Weeks 9-10: Fix `OllamaAdapter::capabilities()` with a `/api/show` call at construction time. Add appropriate error handling for models that don't respond to the endpoint.

Weeks 11-12: Fix `Range::Tokens`, deduplicate `truncate_str`, migrate `timestamp_str` to chrono, update caw-server documentation. These are cleanup items that should be done before any v0.1 release.
