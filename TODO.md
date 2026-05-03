# TODO

Cross-reference: design doc is `context-as-workspace.md`, scope boundaries are in `SCOPE.md`.

## Retrieval

- [x] **BM25 keyword index**: Standard BM25 scoring (K1=1.2, B=0.75) with IDF weighting.
- [x] **Hybrid retrieval fusion**: Semantic + BM25 with min-max normalized scores, configurable weights (default 0.6/0.4).
- [x] **Asymmetric embeddings**: `EmbeddingProvider` trait has `embed_query()`/`embed_document()` with default delegation to `embed()` for symmetric models. `FastEmbedProvider` overrides both for BGE models (adds `"query: "` / `"passage: "` prefixes). All call sites updated: `SemanticRetriever`, CLI indexing, `DynamicRecallOrchestrator`.

## Ingestion & Indexing

- [x] **File ingestion pipeline**: Reads filesystem, detects content kind, captures mtime, generates stubs.
- [x] **SHA256 content hashing**: Fixed formatting with `format!("{:02x}")`.
- [x] **Summary caching**: `StubStore::get_by_content_hash()` allows callers to skip re-ingestion and re-embedding when file content hasn't changed. CLI ingestion loop checks the store before generating embeddings.
- [x] **LLM-generated summaries**: `LlmSummarizer` in `caw-ingest/src/summarizer.rs` wraps any `ModelAdapter` to produce 1-3 sentence summaries from a truncated content window. Falls back to the deterministic summarizer on empty model output. CLI opts in via `--llm-summarize`.
- [x] **Target-model tokenizer**: `TiktokenTokenizer` in `caw-core/src/tokenizer.rs` supports cl100k / p50k / o200k with `for_model()` auto-selection. `IngestionPipeline::new()` now defaults to cl100k (close enough for GPT-4 and Claude); CLI exposes `--tokenizer` for explicit choice. `WhitespaceTokenizer` remains for callers that want trivial counting.
- [x] **Tree-sitter outlines**: `caw-ingest/src/tree_sitter_outline.rs` extracts structural outlines for Rust, Python, JS, TS, Go. `extract_outline` calls tree-sitter first and falls back to the string-matching heuristic only when no grammar matches the file's extension.
- [x] **Adaptive chunking**: `caw-ingest/src/chunking.rs` splits files over a token threshold (default 2000) at structural boundaries — code items, markdown headings, or paragraph breaks — with configurable target chunk size and overlap. Sub-threshold files pass through as a single chunk.
- [ ] **Background indexer with lazy fallback**: Ingestion is synchronous, single-pass, batch-only. The design doc envisions a background process that continuously indexes workspace files, with on-demand fallback for files referenced before the indexer reaches them.

## Eviction & Consolidation

- [x] **Budget-triggered eviction**: Fires when workspace exceeds 80% of token budget, evicts fragments below relevance floor (0.15), frees to 70%.
- [x] **Relevance decay over reasoning steps**: Each fragment tracks a relevance score in `relevance_scores: HashMap<StubId, f32>`. Scores are initialized from retrieval score on admission, decayed by `relevance_decay_rate` (default 0.8) each reasoning step, and refreshed via term-overlap when the model's output re-engages with the fragment. Eviction uses the hysteresis unload threshold on decayed scores, plus budget enforcement as a hard ceiling. Consolidation notes now include the decayed score at eviction time.
- [ ] **Consolidation note quality**: Eviction-time notes are mechanical strings (`"Evicted during query about '{query}'. Source: {path}"`). The design doc calls for notes that summarize what portions were referenced and what conclusions were drawn — requires either an LLM call or deeper analysis of provenance records.
- [x] **Consolidation persistence**: `StubStore` trait has `save_consolidation()`/`load_consolidation()` with default no-ops. `SqliteStubStore` implements them with a `consolidation_notes` table. `DynamicRecallOrchestrator` accepts an optional store via `with_store()` and persists notes on both eviction and model annotation. Notes survive across sessions.

## Provenance

- [x] **Provenance ledger**: Records recalled fragments, tracks topic terms, detects term-overlap between fragments from different sources (Jaccard >30%).
- [x] **Inline provenance tagging**: `CompletionRequest::format_workspace()` in caw-core wraps every recalled fragment with source attribution (`[recalled from path:locator]` or XML equivalent). All adapters use the shared formatter. The preamble instructs the model to treat recalled content as quoted material. Two format modes: `ProvenanceFormat::Xml` (Anthropic) and `ProvenanceFormat::Bracketed` (OpenAI-protocol models).
- [ ] **Conflict detection beyond term overlap**: Only Jaccard term overlap is implemented. Contradicting assertions, inconsistent numbers, and negation patterns are undetected.

## Mid-Session Annotation

- [x] **Annotation parser**: `extract_annotations()` parses `<note id="stub_id">content</note>` markers from model output.
- [x] **Model instruction for annotations**: `DynamicRecallOrchestrator::build_system_prompt()` injects annotation and probe instructions when the model's capabilities indicate cooperation support. Models with tool call support get `<note>` instructions; models with visible reasoning also get `<probe>` instructions. Injected automatically — callers don't need to craft the prompt.

## Prompt Transformer

- [x] **Markdown link and @path references**: `PromptTransformer` finds `[text](path)` and `@path` references, replaces with formatted stubs.
- [ ] **Additional reference surfaces**: Fenced blocks with `path=` and bare paths matching a regex are described in the design doc but not implemented.

## Orchestration

- [x] **Multi-pass recall loop**: `DynamicRecallOrchestrator` runs iterative recall — initial retrieval, probe/trace extraction, re-retrieval, convergence detection. Up to `max_recall_iterations` (default 3).
- [x] **Probe extraction**: Parses `<probe>...</probe>` markers from model output for automatic recall.
- [x] **Thinking-trace extraction**: Parses `<think>...</think>` blocks and heuristic step boundaries for reasoning models.
- [ ] **Streaming recall**: Multi-pass is request/response per iteration. True streaming (interleave retrieval with token generation mid-response) requires async streaming adapter traits. The multi-pass approach captures most of the value but doesn't match the design doc's mid-reasoning vision.

## Adapters

- [x] **Anthropic adapter**: Claude Sonnet/Opus via blocking reqwest.
- [x] **Groq adapter**: Llama 70B/8B, Mixtral.
- [x] **Ollama adapter**: Local models (DeepSeek R1, Qwen, Llama 3.2).
- [x] **OpenAI-compatible adapter**: Generic adapter for any provider speaking the chat completions protocol (vLLM, Perplexity, HuggingFace Inference Endpoints, ollama.com). Configurable headers and capabilities.
- [ ] **Adapters use sync-wrapped async (`block_on`)**: Works for the library target. Will panic if called from within an async context (e.g., a future server). Acceptable per SCOPE.md — async refactoring deferred to v2.

## Curation Hooks (design doc section 8.1)

Live in `caw-curation`. CLI opts in via `--curate`. Extractive variants avoid
LLM calls entirely; LLM variants route through any configured `ModelAdapter`.

- [x] **History summarization**: `HistorySummarizer` trait with `ExtractiveHistorySummarizer` (keeps first/last sentences per turn) and `LlmHistorySummarizer` (calls a model). `HistorySummarizerConfig` controls trigger threshold (default 30% of budget), retain-recent count (default 3 verbatim), and summary budget. `partition_turns()` selects eligible vs. retained turns.
- [x] **Tool output compression**: `ToolOutputCompressor` trait with extractive + LLM variants. Individual turn metadata (`inline_required`) lets specific tool outputs bypass compression. Config threshold controls when compression fires.
- [x] **System prompt budgeting**: `SystemPromptBudget` with `from_context_window()` constructor and `check_budget()` / `check_system_prompt()` functions returning `Ok` / `Warning` / `Exceeded`. Callers decide whether to truncate or just warn.
- [ ] **Few-shot management**: Not explicit. Callers can use `caw_core::Tokenizer` to measure individual example costs, but no framework-level utility surfaces the delta.

## Measurement (design doc section 10)

Minimal. This is the prerequisite for tuning everything above.

- [x] **Basic eval metrics**: recall@k and precision@k over RecallFragment.
- [x] **SessionEvaluator**: Single-session aggregator that records recalls, evictions, probes, annotations, and turns. Produces recall metrics, false-recall metrics, hysteresis analysis, context efficiency, and cooperation metrics on demand. In `caw-eval/src/session.rs`.
- [x] **False-recall heuristic**: `FalseRecallMetrics::from_observations` scores stub-summary vs recalled-content term overlap. Configurable threshold flags potential misreads for review. Still a heuristic — contradiction detection across recalled fragments is a separate open item.
- [x] **Effective vs nominal context ratio**: `ContextEfficiency::compute` exposes the ratio of content tokens to total workspace tokens (content + stubs + overhead).
- [x] **Hysteresis threshold analysis**: `HysteresisAnalysis` detects thrashing (load-evict-reload within a window) and suggests load/unload threshold adjustments. Still needs a harness that drives it across workloads to produce tuning recommendations rather than per-session diagnostics.
- [ ] **Insertion-order experiments**: Test relevance-ranked vs reverse-relevance vs original-stub order for recalled content placement. No harness wired up yet.
- [ ] **Model cooperation calibration**: `CooperationMetrics` tracks probes-per-turn, annotations-per-turn, useful-probe percentage, and annotation quality. No calibration harness exists yet — per-model benchmarking that drives automatic cooperative-vs-transparent mode selection is still open.
- [x] **End-to-end benchmarking harness**: `caw-bench` binary exercises the full recall loop in recall-on vs recall-off modes at matched context budget. Two workloads: `niah` (procedurally-generated haystack with fabricated needles — guaranteed out-of-training) and `opencaw` (hand-authored Q&A against the repo itself, judged against reference answers). Reports recall@k, precision@k, context efficiency, false-recall rate, latency, and answer score per item with per-mode aggregates. Still open: actually running it across enough seeds/sweeps to produce threshold-tuning recommendations; the harness is the instrument, calibration numbers are a separate exercise.

## Degradation & Monitoring (design doc section 6)

In `caw-orchestrator/src/degradation.rs`. Opt-in via
`DynamicRecallOrchestrator::with_degradation_monitor()`. Without one, the
orchestrator runs at full capability unconditionally.

- [x] **Per-component health checks**: `ComponentHealth` tracks latency and error rate for the embedding service and summary generator independently. `ProbeRateLimiter` tracks probe frequency within a sliding window. Each has configurable thresholds and hysteresis (N consecutive successes to recover).
- [x] **Tiered fallback**: `OperatingTier::{FullRecall, StubsAndToolsOnly, PassThrough}`. `DegradationMonitor::effective_tier()` combines component health into the current tier. Orchestrator skips automatic recall and/or stub generation based on tier. Recovery is automatic when components return to healthy.
- [x] **Probe rate limiting**: `ProbeRateLimiter` with configurable window + max probes. When tripped, the effective tier drops to `StubsAndToolsOnly` until old probes age out of the window.

## Infrastructure

- [ ] **Over-decomposed workspace**: 12 crates for the current codebase size. `caw-provenance`, `caw-eval`, `caw-scheduler` could be modules within larger crates. Not blocking but adds friction. Do not restructure without explicit approval.
- [x] **End-to-end integration test**: `crates/caw-orchestrator/tests/end_to_end.rs` wires ingest → embed → index → retrieve → schedule → complete → provenance through `MockAdapter` and a deterministic hash embedder. Offline, no API keys. Primary smoke test for API-surface breakage.

## Other

- [ ] detect git/repo references in prompts and link to stubs with git status (modified/unmodified), git blame info (author, commit message, date), branch information, and/or git log data, as needed.
