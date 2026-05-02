# OpenCAW Codebase Review — 2026-05-01

## Executive Summary

OpenCAW is a well-structured Rust library in late-MVP state. The core recall pipeline — ingest, embed, retrieve, orchestrate, curate, evaluate — is implemented and the full workspace test suite passes cleanly. The project's primary gaps are a missing calibration harness for model cooperation, mechanical-quality consolidation notes as the default, and no benchmark sweep data to back the library's threshold defaults. The most important next work is generating actual numbers from the benchmark harness that already exists, not adding new features.

---

## Recent Activity (last 20 commits)

1. `981eb39` Show crate/file/line in trace output
2. `4c724c7` Add trace-level prompt/response logging to all adapters
3. `c9b2070` Add tracing to orchestrator and Ollama streaming path
4. `2dec7ea` cargo fmt
5. `fc8788c` apply cargo clippy suggestions
6. `09089fb` Refactor fixture_docs to use current mtime + improve metadata handling
7. `aa48cb4` Purge "You are a helpful assistant."
8. `eb334b9` Thinking-trace recall loop with Ollama streaming
9. `7d236d3` Update deps to use workspace references for caw-core
10. `895f6ed` Add `thinking` field to CompletionResponse across adapters; implement `split_thinking`
11. `1f62ddf` Add Ollama demo with file ingestion and query orchestration
12. `89ef01f` Enhance Groq demo with recall instrumentation and candidate visibility
13. `c62bb9c` Reset orchestrator state before processing each query in Groq demo
14. `c472947` Add Groq demo example; update dependencies
15. `6e1b784` Fix semantic_demo example
16. `f6f217e` Add history externalization feature
17. `93b45be` Stale-stub detection on recall with reindex queue
18. `c8ed770` caw-server v0: OpenAI-compatible proxy that injects recalled context
19. `c611f08` Update STREAM_EVENT_TIMEOUT and set cwd for ClaudeCode adapter
20. `0cc266e` Increase STREAM_EVENT_TIMEOUT and enhance ClaudeCode streaming

Recent work has focused on observability (tracing), adapter stability (Ollama streaming, ClaudeCode), and demos. The caw-server v0 was added recently and is now a functional proxy, not an empty scaffold as the README describes.

---

## Implementation Status

### Working

- **caw-core**: Complete. All shared types, traits, error types, tokenizer abstractions, provenance stores, scheduling interfaces, and the `CompletionRequest::format_workspace()` formatter with XML/bracketed dual-format support. `split_thinking()`, hysteresis types, `Range::apply()`, byte-range locators all present and functional.

- **caw-ingest**: Complete for v1 scope. SHA256 hashing, `TiktokenTokenizer` defaulting to cl100k, tree-sitter outlines for Rust/Python/JS/TS/Go with naive fallback, adaptive chunking with structural boundary detection, LLM and deterministic summarizers, parallel directory ingestion via rayon. `IngestionPipeline::ingest()` returns `(Stub, embed_text)` pairs with precomputed token counts to avoid double-tokenization.

- **caw-index**: Complete. `SemanticRetriever`, `HybridRetriever` (0.6 semantic / 0.4 BM25), `HnswVectorIndex` (instant-distance), `FlatVectorIndex` (brute-force cosine). `SqliteStubStore` with WAL, batch inserts, consolidation persistence, stale-stub detection with `mtime` mismatch, and optional `ReindexQueue` integration. `BM25Index` with standard K1=1.2/B=0.75 IDF weighting. API embedding provider (OpenAI-shape). FastEmbed (BGE-small/BGE-base), Candle (CUDA), and ONNX providers behind feature flags.

- **caw-transform**: Complete. `PromptTransformer` handles `[text](path)` markdown links and `@path` references. `extract_probes`, `extract_thinking_steps`, `extract_annotations` all functional. Tests cover the main paths.

- **caw-adapters**: Complete for all named targets. Anthropic (Claude Sonnet/Opus), Groq (Llama 70B/8B, Mixtral), Ollama (streaming + non-streaming, thinking-trace extraction), OpenAI-compatible (generic), ClaudeCode (local CLI), MockAdapter. All share a tokio runtime via `create_runtime()` and `new_with(runtime)` constructors. `TracingAdapter` wraps any adapter and writes JSONL request/response logs.

- **caw-orchestrator**: Core `DynamicRecallOrchestrator` is complete. Multi-pass recall (up to `max_recall_iterations`), probe extraction, thinking-trace extraction, relevance decay per step, term-overlap refresh, budget-triggered eviction with consolidation notes, annotation parsing, degradation monitor integration. `DegradationMonitor` with per-component health, tiered fallback, and probe rate limiting is wired up and tested. The older `RecallOrchestrator` and `ProbeRecallOrchestrator` are also present but not actively used.

- **caw-curation**: Complete for v1 scope. `ExtractiveHistorySummarizer` and `LlmHistorySummarizer`, `ExtractiveToolOutputCompressor` and `LlmToolOutputCompressor`, `SystemPromptBudget` with configurable fraction and `check_budget()`/`check_system_prompt()`. `CurationPipeline` + builder compose all three into a single pass. History externalization (session files as recalled documents) implemented in `history.rs`.

- **caw-eval**: Complete. `SessionEvaluator` with builder, recall@k/precision@k, false-recall heuristic (stub-summary vs content term overlap), hysteresis analysis with thrashing detection and threshold adjustment suggestions, context efficiency ratio, and cooperation metrics (probes/turn, annotations/turn, useful-probe %, annotation quality).

- **caw-bench**: Substantive. Three workloads: `niah` (synthetic needle-in-haystack), `opencaw` (Q&A over repo docs), `sysdoc` (pre-built external corpus). `caw-bench-build-index` binary with pipelined GPU embedding (rayon producer + candle consumer), resumable via `(path, mtime)` skip logic, WAL batch inserts, length-bucketed sub-batches. `sweep.rs` binary for parameter sweeps. Per-item JSONL trace output. LLM judge for answer scoring. The harness is the instrument; calibration data from running it is the open gap.

- **caw-server**: Functional v0 proxy (README incorrectly describes it as "scaffold only"). Receives OpenAI-protocol requests, embeds the last user message, retrieves top-k fragments, splices them in with bracketed provenance, and proxies to an upstream server. Supports Flat or HNSW retriever at startup. `build_state`/`build_router` public for integration test mounting.

- **caw-core/reindex**: `ChannelReindexQueue` with condvar-based blocking recv, dedup of pending paths, and `run_worker` helper. `SqliteStubStore` notifies the queue on mtime-mismatch staleness and marks the row stale in DB for durability across restarts.

### Incomplete / Partial

- **caw-orchestrator/consolidation — LlmConsolidation not the default**: `LlmConsolidation` exists and works but `MechanicalConsolidation` remains the default in `DynamicRecallOrchestrator::new()`. The mechanical note format (`"Evicted (relevance decayed to 0.42) during query about '...'"`) records nothing about what the model actually learned. This is the verbatim gap the design doc called out. Acknowledged in `TODO.md` (`caw-orchestrator/src/consolidation.rs:16-33`).

- **caw-transform — missing reference surfaces**: Fenced code blocks with `path=` attribute and bare-path regex matching are described in the design doc (section 3.1) but not implemented. Only `[text](path)` markdown links and `@path` are handled (`caw-transform/src/lib.rs:14-67`). This limits transformer usefulness in code-heavy workspaces.

- **caw-eval/SessionEvaluator — not integrated into orchestrator**: The evaluator is standalone and must be wired up manually by callers. The bench harness does its own accounting. There is no built-in instrumentation path from `DynamicRecallOrchestrator` to `SessionEvaluator`.

- **caw-adapters/OllamaAdapter capabilities — name-based detection**: `ModelCapabilities` are hardcoded via string matching (`self.model.contains("deepseek") || self.model.contains("qwen")` for `supports_visible_reasoning`) at `caw-adapters/src/ollama.rs:127-129`. Ollama's `/api/models` endpoint is not queried. New model names silently get wrong capability flags, changing what cooperation instructions get injected.

- **caw-bench sweep data**: The harness, sweep binary, and sysdoc QA file exist. Actual sweep runs with enough seeds to produce threshold-tuning recommendations have not been conducted. The library's defaults (load=0.7, unload=0.4, decay=0.8) are starting estimates. Notably, `RunnerConfig::default()` in `caw-bench/src/runner.rs:36-50` uses load=0.3 for bench runs, suggesting the library default of 0.7 is already known to be too restrictive for short-text corpora.

### Missing / Not Yet Started

- **Background indexer with lazy fallback**: Ingestion is batch-only and synchronous. The `ReindexQueue` plumbing exists for staleness/reingestion signaling, but there is no persistent background worker or lazy-on-first-reference generation. Corpus must be fully ingested before the first query.

- **Model cooperation calibration harness**: `CooperationMetrics` exists in `caw-eval`. The harness that drives per-model calibration runs — identifying whether each model reliably emits probes and annotations, and automating the choice between cooperative and transparent mode — does not exist.

- **Few-shot token cost utility**: Not implemented. Callers can use the `Tokenizer` trait directly. Low priority per SCOPE.md.

- **Provenance conflict detection beyond Jaccard term overlap**: Contradicting factual assertions, negation patterns, and inconsistent numbers are not detected. Only term co-occurrence is checked (`caw-core/src/provenance.rs:90-133`).

- **Insertion-order experiments harness**: No harness to test relevance-ranked, reverse-relevance, and stub-order insertion sequences.

- **Postgres + pgvector**: Listed in README roadmap as "open". No implementation exists.

---

## Test Results

All tests pass. 0 failures.

```
caw-adapters:      2 passed  (TracingSink, TracingAdapter)
caw-core:          3 passed  (reindex queue: dedup, reenqueue, multi-receiver split)
caw-index:         3 passed  (sqlite: stale-on-missing-file, stale-on-mtime-mismatch, replace-clears-stale)
caw-orchestrator:  9 passed  (degradation: 7 scenarios; probe_recall, thinking_trace: 1 each)
end_to_end:        1 passed  (full pipeline via MockAdapter + SQLite in-memory)
caw-transform:     6 passed  (range parse/apply, extract_probes, transform, extract_stub_references)
caw-server:        0 tests
caw-bench:         0 tests
caw-curation:      0 tests
caw-eval:          0 tests
caw-cli:           0 tests
caw-ingest:        0 tests
```

Coverage gap: caw-curation, caw-eval, caw-ingest, and caw-bench have zero tests. These contain non-trivial logic — hysteresis analysis, history partition thresholds, chunking boundary detection, BM25 scoring. The end-to-end test exercises the full stack via `MockAdapter` and is the primary smoke test, but it does not exercise curation or eval code paths at all.

---

## Technical Debt

### High Priority

1. **`Regex::new(...).unwrap()` compiled on each call** — `caw-transform/src/lib.rs:21,23,98`; `caw-transform/src/recall.rs:6,83`; `caw-orchestrator/src/probe_recall.rs:71,222`. These are literal patterns that are infallible in practice but re-compile the regex on every struct construction or function call. Use `std::sync::LazyLock` to compile once. The `extract_stub_references` function at `caw-transform/src/lib.rs:98` is particularly bad — it compiles a regex on every call and is in a library hot path (called per model output).

2. **`expect("poisoned")` on mutex locks in production path** — `caw-core/src/reindex.rs:85,99,118,127`. A panicking thread poisons these mutexes and causes cascading panics via `.expect()`. The reindex queue is used in the server's live request path. Should recover from poison via `PoisonError::into_inner()`.

3. **`expect(...)` on tiktoken data bundles** — `caw-core/src/tokenizer.rs:16,24,32`. These panic at process startup if the bundled BPE data is corrupted or missing. Since `IngestionPipeline::new()` calls `default_tokenizer()`, the entire ingestion pipeline fails to construct with a panic rather than a `CawResult` error. The `TiktokenTokenizer::cl100k()` et al. should return `CawResult<Self>`.

4. **Synthetic `Stub` construction in thinking-trace recall** — `caw-orchestrator/src/dynamic.rs:258-276`. When a thinking-trace hit comes back from the vector index, a `Stub` with zeroed-out metadata is constructed just to satisfy `load_fragments`'s type signature. The stub fields are unused in that function. `load_fragments` should accept `Vec<(StubId, f32)>` and do its own store lookup, eliminating the fake struct that currently hides a real type mismatch.

### Medium Priority

5. **Stopword list duplicated** — `is_stopword()` appears identically in `caw-core/src/provenance.rs:165-226` and `caw-orchestrator/src/dynamic.rs:501-561`. Future updates must be made in two places. Move to `caw-core` and re-export.

6. **`caw-adapters/src/claude_code.rs:354` — `.expect("loop exits with final_result set on success path")`** — A logic invariant assertion. If the loop is refactored the invariant could silently break. Should be `unreachable!()` with a comment, or restructured so the result is not in an Option at all.

7. **OllamaAdapter capability detection by model name string** — `caw-adapters/src/ollama.rs:127-129`. Model naming is under Ollama's control and changes regularly. `.contains("deepseek")` will fail for models like `deepseek-r1:32b` versus `deepseek-v3` with different reasoning behavior. Add a configuration override or a runtime capability probe.

8. **`#[allow(dead_code)]` in `caw-eval/src/session.rs:39`** — The `RecallEvent` fields `score` and `content_tokens` are stored for future per-recall analytics. The comment explaining rationale is present, which is the right approach, but if the analytics never arrive this becomes permanent dead weight.

### Low Priority

9. **`caw-server/src/lib.rs` hardwires `CandleEmbeddingProvider`** — `build_state` requires CUDA. A server on a CPU-only host fails at startup. FastEmbed would work with no CUDA dependency. Coupling is intentional for v0 scope but undocumented.

10. **`RecallOrchestrator` and `ThinkingTraceOrchestrator` not actively used** — These older orchestrators in `caw-orchestrator/src/lib.rs` and `caw-orchestrator/src/thinking_trace.rs` predate `DynamicRecallOrchestrator` and are not wired into any demo, test, or bench path. If retained they need tests; if superseded they add maintenance surface without benefit.

---

## PRP Status

No PRPs directory exists.

---

## Strategic Recommendation

**Next Action**: Run the benchmark harness across enough seeds and workloads to produce actual recall@k numbers, context efficiency ratios, and threshold-tuning data. The bench binary, sysdoc QA file, sweep binary, and NIAH corpus generator all exist. The single most valuable thing to do right now is produce numbers, because every architectural decision (load threshold 0.7, decay 0.8, top_k 4) is currently a documented guess.

**Justification**: The library's core claim over generic RAG is that thinking-trace-as-retrieval-signal plus eviction/consolidation produces better effective context than naive retrieval. That claim has no empirical backing yet. Without numbers: (a) the default thresholds may be wrong for real workloads — `RunnerConfig::default()` in the bench crate already uses 0.3 as the load threshold rather than the library's 0.7, which is a tacit admission the default is too tight; (b) the cooperation calibration work, consolidation note quality work, and insertion-order experiments are unmotivated without knowing how much each matters in practice. The benchmark infrastructure represents significant engineering investment and is ready to use.

---

## 90-Day Roadmap

**Week 1-2: Generate baseline numbers**

Run `caw-bench` with `opencaw` and `niah` workloads in recall-on vs recall-off mode at several top_k / threshold combinations via the `sweep.rs` binary. Primary deliverable: a table of recall@k, context efficiency, and answer score per configuration. Specifically verify whether load=0.7 is workload-appropriate or whether the bench's hardcoded 0.3 reflects a real problem with the library default.

**Week 3-4: Fix the highest-impact technical debt**

Replace `Regex::new(...).unwrap()` constructions with `LazyLock` statics in `caw-transform` and `caw-orchestrator/src/probe_recall.rs`. Fix the synthetic Stub construction in `DynamicRecallOrchestrator::process_thinking_trace` — change `load_fragments` to accept scored id pairs directly and do its own stub lookup, eliminating the fake struct. Fix tiktoken constructors to return `CawResult` rather than panicking.

**Week 5-8: Model cooperation calibration harness**

Build the calibration harness that runs a fixed set of probing questions through each configured adapter, measures whether the model emits probes and annotations that produce correct recalls, and produces a per-model cooperation score. Wire this into `DynamicRecallOrchestrator` as an optional startup phase that sets `enable_probe_recall` and system prompt injection based on observed cooperation quality rather than the current hardcoded capability flags. This is the blocker for v1 usefulness identified in SCOPE.md.

**Week 9-12: LLM consolidation as default + test coverage for curation/eval**

Switch `DynamicRecallOrchestrator::new()` to use `LlmConsolidation` when an aux adapter is configured (or make `with_llm_consolidation(adapter)` the obvious path and document when to use it). Add integration-level tests for `caw-curation` (history partitioning at threshold, tool output compression trigger, budget check) and `caw-eval` (hysteresis thrashing detection, cooperation metric calculation, context efficiency ratio). These crates are exercised only indirectly via the end-to-end test; targeted tests would catch regressions when eviction/curation logic changes during calibration tuning.
