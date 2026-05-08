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
- [ ] **Background indexer with lazy fallback**: Ingestion is synchronous, single-pass, batch-only.
- [ ] **Stale stub detection and re-indexing**: The `StubStore::get_by_content_hash()` mechanism skips re-ingestion when file content hasn't changed, but there is no path that re-ingests files whose content *has* changed. Stubs for frequently-edited files (TODO.md, session logs, SCOPE.md) drift from reality and mislead the model — e.g., TODO.md stubs show items as open when they are completed (observed in QA loop 0003). At session start the CLI should compare stored stubs' content hashes against current file content and queue changed files for re-ingest before answering queries. The design doc envisions a background process that continuously indexes workspace files, with on-demand fallback for files referenced before the indexer reaches them.
- [x] **Fix B3: session-log live-insertion path uses in-memory HNSW only**: `with_session` now calls `session::collect_previous_stubs` which returns `(Stub, embed_text)` pairs without inserting anything into the retriever or SQLite store. Each stub is embedded via `self.embedder` and added to `self.vector_index` + `self.session_content` (in-memory only), identical to how current-session turns are handled. Prior session content can be recalled semantically but never propagates into the persistent store.
- [x] **Minimum-content filter for chunk indexing**: The CLI indexing loop now skips stubs with `token_estimate < 10`. Cargo.toml `[package]`-only chunks (3 tokens) and other low-signal trailing fragments no longer consume index slots. Threshold chosen to catch degenerate chunks while keeping all legitimate content (the next-smallest useful chunk is well above 10 tokens).

## Eviction & Consolidation

- [x] **Budget-triggered eviction**: Fires when workspace exceeds 80% of token budget, evicts fragments below relevance floor (0.15), frees to 70%.
- [x] **Relevance decay over reasoning steps**: Each fragment tracks a relevance score in `relevance_scores: HashMap<StubId, f32>`. Scores are initialized from retrieval score on admission, decayed by `relevance_decay_rate` (default 0.8) each reasoning step, and refreshed via term-overlap when the model's output re-engages with the fragment. Eviction uses the hysteresis unload threshold on decayed scores, plus budget enforcement as a hard ceiling. Consolidation notes now include the decayed score at eviction time.
- [x] **Consolidation note quality**: `LlmConsolidation` synthesizer produces LLM-written notes (conclusions, decisions, key facts). Enabled via `--llm-consolidation`; `MechanicalConsolidation` remains the no-LLM default. Notes are now prepended to fragment content on re-recall so the model sees what was previously learned from the source.
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
- [ ] **Transforming is not a gate**: Remove hard-coded responses from classification/transformer adapters.

## Orchestration

- [x] **Multi-pass recall loop**: `DynamicRecallOrchestrator` runs iterative recall — initial retrieval, probe/trace extraction, re-retrieval, convergence detection. Up to `max_recall_iterations` (default 3).
- [x] **Probe extraction**: Parses `<probe>...</probe>` markers from model output for automatic recall.
- [x] **Thinking-trace extraction**: Parses `<think>...</think>` blocks and heuristic step boundaries for reasoning models.
- [ ] **Streaming recall**: Multi-pass is request/response per iteration. True streaming (interleave retrieval with token generation mid-response) requires async streaming adapter traits. The multi-pass approach captures most of the value but doesn't match the design doc's mid-reasoning vision.
- [ ] **Progressive disclosure: stub-to-full upgrade on probe**: `load_fragments` now defaults to `"stub"` range (summary + outline). When a probe fires on a stub already in the workspace, the fragment should be upgraded from stub to full content in-place rather than being skipped by `loaded_sources`. Requires distinguishing stub-loaded vs full-loaded in the `loaded_sources` set and allowing re-entry for upgrades.

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
- [x] **Model cooperation calibration**: `CooperationMetrics` tracks probes-per-turn, annotations-per-turn, useful-probe percentage, and annotation quality. `caw-bench-coop` runs each model in three modes (baseline/transparent/cooperative), measures recall delta and probe engagement, and emits a `coop-report.json` per model with a `cooperative`/`transparent` recommendation. `caw-bench-tune` accepts `--coop-dir` and includes the recommendations in its config output.
- [x] **End-to-end benchmarking harness**: `caw-bench` binary exercises the full recall loop in recall-on vs recall-off modes at matched context budget. Two workloads: `niah` (procedurally-generated haystack with fabricated needles — guaranteed out-of-training) and `opencaw` (hand-authored Q&A against the repo itself, judged against reference answers). Reports recall@k, precision@k, context efficiency, false-recall rate, latency, and answer score per item with per-mode aggregates. Still open: actually running it across enough seeds/sweeps to produce threshold-tuning recommendations; the harness is the instrument, calibration numbers are a separate exercise.

## Degradation & Monitoring (design doc section 6)

In `caw-orchestrator/src/degradation.rs`. Opt-in via
`DynamicRecallOrchestrator::with_degradation_monitor()`. Without one, the
orchestrator runs at full capability unconditionally.

- [x] **Per-component health checks**: `ComponentHealth` tracks latency and error rate for the embedding service and summary generator independently. `ProbeRateLimiter` tracks probe frequency within a sliding window. Each has configurable thresholds and hysteresis (N consecutive successes to recover).
- [x] **Tiered fallback**: `OperatingTier::{FullRecall, StubsAndToolsOnly, PassThrough}`. `DegradationMonitor::effective_tier()` combines component health into the current tier. Orchestrator skips automatic recall and/or stub generation based on tier. Recovery is automatic when components return to healthy.
- [x] **Probe rate limiting**: `ProbeRateLimiter` with configurable window + max probes. When tripped, the effective tier drops to `StubsAndToolsOnly` until old probes age out of the window.
- [ ] **Continuous improvements**:
  - [ ] Improve the prompt logging to include any metadata that may be helpful for debugging and troubleshooting.
  - [ ] Improve logging with timestamps and component tags to better understand the sequence of events leading to degradation.
  - [ ] Add more detailed error messages and actionable insights in the logs to facilitate faster debugging and resolution of issues.
  - [ ] Implement a notification system to alert developers when degradation is detected, including details about the affected components and potential causes.
  - [ ] Regularly review and analyze degradation incidents to identify common patterns and areas for improvement in the system's robustness and reliability.

## Infrastructure

- [ ] **Over-decomposed workspace**: 12 crates for the current codebase size. `caw-provenance`, `caw-eval`, `caw-scheduler` could be modules within larger crates. Not blocking but adds friction. Do not restructure without explicit approval.
- [x] **End-to-end integration test**: `crates/caw-orchestrator/tests/end_to_end.rs` wires ingest → embed → index → retrieve → schedule → complete → provenance through `MockAdapter` and a deterministic hash embedder. Offline, no API keys. Primary smoke test for API-surface breakage.

## Other

- [ ] **Tool call support**: The design doc describes tool calls as a distinct retrieval signal and a separate path for proactive context injection. Not implemented yet — the current approach is to detect tool-like prompts and inject relevant context reactively, but true tool call support with visible tool output and proactive injection is still open.
- [ ] **Additional prompt reference surfaces**: The current `PromptTransformer` supports markdown links and `@path` references. The design doc also describes fenced blocks with `path=` and bare paths matching a regex. These are not implemented yet.
- [ ] Add scripts for systematically running the `caw-bench` harness across multiple seeds, models, and workloads to produce statistically significant results for tuning the various thresholds. The harness is in place but the actual runs and analysis are still pending.
- [ ] Improve the benchmark prompt suite for intent classification to cover more varied and complex queries, including multi-intent combinations and edge cases. The current set is a starting point but could be expanded to better represent real-world usage.
- [ ] detect git/repo references in prompts and link to stubs with git status (modified/unmodified), git blame info (author, commit message, date), branch information, and/or git log data, as needed.
- [x] Skip retrieval and answer generation entirely when the intent classifier returns all-false and the user message is short (under ~15 tokens). Low-information inputs (casual acknowledgments like "nice to know", "ok", "got it") currently trigger a full retrieval cycle that surfaces lexically similar but contextually wrong content, and the model produces off-topic responses (observed in QA loop 0003 session 050545 turn 3).
- [x] Enforce per-stub deduplication in the workspace during multi-pass recall: once a stub_id is admitted, subsequent retrieval of the same stub_id during the same turn should update relevance score only, not append a second fragment. The current multi-pass path re-admits overlapping windows of the same large file on each thinking-trace step, producing corrupted and repeated context (observed in QA loop 0003 prompt turn-8, 30+ overlapping fragments from one file).
- [x] Add diagnostic logging to the eviction and consolidation path: log how many evictions fired per session, how many consolidation notes were generated, and how many were persisted to the store. Three QA loop sessions (0002, 0003) have produced zero consolidation notes; without this instrumentation it is unknown whether eviction never triggered or the persistence path is broken.
- [ ] Flush session log markdown after every turn (not only at session end). Current behavior truncates session logs when the process exits before final flush — sessions with 18 prompt files show only 3 turns in the session log (observed in QA loop 0003, sessions 050942 and 051313).
- [ ] reconsider parse failures in small models & see if there may be a better way to use them to accomplish the classification goal without strict JSON parsing. The current approach is to look for JSON in the raw output, but that may be brittle. For example, if the model outputs "is_inventory_request: true" without JSON formatting, we could still detect that with a regex or simple string search, and it would be more robust to minor formatting variations.
- [x] Exclude `scripts/` from retrieval indexing (or add a shell-script extension skip rule). `scripts/qa.sh` is surfacing as a top retrieval hit for every query because its CODEBASE LAYOUT section and QA prompt templates mention every crate. This injects stale architecture descriptions into every session. The fix is the same mechanism as the `.caw/` skip: add a path component or extension filter to `should_skip` in `caw-ingest/src/lib.rs`.
- [x] Fix intent classifier prompting to recognize count queries as `is_status_request` or `is_inventory_request`. Queries like "how many todos are left?", "how many open bugs?", "total remaining items?" classify as all-false, disabling intent-driven document loading. Add examples with count language ("how many", "total", "remaining", "count of") to the classifier's few-shot or prompt examples.
- [ ] Fid intent classifier so that it does not act as a gate for retrieval when it fails to recognize the intent. The current approach is to skip retrieval entirely when the classifier returns all-false, but this leads to failure modes where low-information queries (e.g., "ok", "got it") or queries with unrecognized phrasing ("how many todos are left?") bypass retrieval and produce off-topic responses. Instead, the classifier should be used as a signal to bias retrieval and prompt construction, but not as a hard gate that disables retrieval entirely. The user wants the MODEL to respond, not the retrieval system, even if the intent is not recognized. The classifier can still influence the response by tagging the query or adjusting retrieval weights, but it should not prevent relevant context from being retrieved and included in the prompt.
- [ ] Improve stub summarization for trailing chunks that contain only code fragments (closing braces, `#[test]` blocks, partial match arms). The current fallback produces summaries like `| ".git"` or a raw comment line. When a chunk's extracted text is below a minimum semantic content threshold (e.g., contains only punctuation, keywords, or a single code line), fall back to a description derived from the parent file stub's summary rather than summarizing the raw fragment.
- [ ] add more default bench cases with more varied intent combinations, e.g. non-grounded questions, next-step questions that also ask for numeric values ("what's the next step and how long will it take?"), etc.
- [ ] **Intent-driven proactive context injection**: `AugmentationSignals` is defined and extracted from `QueryIntent`, but the orchestrator/CLI does not yet act on it. For `is_status_request` / `is_next_step_request` queries where the top-k hits are fragments of the same document file, load all remaining chunks of that file up to the token budget — this is required for count and status queries against TODO.md, SCOPE.md, etc. to be answerable at all (observed failure in QA loop 0003). Wire augmentation signals into the retrieval path so that e.g. `is_status_request` biases retrieval toward status/todo docs and triggers a `git status` inject, `is_results_request` proactively loads recent bench result files, and `is_inventory_request` injects a file/artifact listing — all before the answer model runs. The goal is that queries like "what's uncommitted?" get the relevant context surfaced automatically without the model needing to invoke tools.
- [ ] Add a `caw-bench-sweep` config for systematically benchmarking more models + workloads + parameter variations.
- [ ] Add a `caw-bench-intent` binary for evaluating small models as query intent classifiers, to help pick a cheap model for routing inventory vs. results vs. comparison vs. explanation queries before the main answer model runs.
- [ ] Consider adding a `caw-bench-probe` binary for evaluating small models as probe responders, to help pick a cheap model for detecting when retrieved context is relevant to the model's current line of reasoning.
- [ ] Deduplicate session history before injection: session history currently arrives as 3–6 overlapping sliding-window fragments per turn, collectively consuming 30–40% of the token budget with heavily repeated content. Replace with a single contiguous fragment (or use `HistorySummarizer` from `caw-curation` to compress history when it exceeds a configurable fraction of the budget). Observed in QA loop 0006 sessions 063242 and 063609.
- [ ] Improve eviction threshold tuning relative to decay rate: with decay_rate=0.8 and unload_threshold=0.5, a fragment admitted at score 0.7 is evicted after 2 reasoning steps with no re-engagement. Documents that are re-admitted multiple times in the same session (indicating persistent relevance) should have their effective eviction floor lowered. Consider tracking a per-fragment "re-admission count" and dampening decay for fragments that the retriever keeps re-surfacing. Run `HysteresisAnalysis` after each QA session to detect thrashing patterns automatically. Observed in QA loop 0006.
- [ ] Improve `MechanicalConsolidation` note format: notes currently record only the eviction score and source path ("Evicted (relevance decayed to 0.52) during query about '...'"). They should also include the evicted stub's summary text so the model can see what topic was evicted, not just that something was evicted. This is low-cost and requires only including `stub.summary` in the mechanical note body. Observed in QA loop 0006 where all 247 consolidation notes are pure bookkeeping with no topic content.
- [ ] Weight documentation stubs above implementation code stubs for `wants_explanation: true` queries. Architecture/overview queries ("what is opencaw?", "what can you do?") consistently retrieve `caw-bench/src/bin/*.rs` implementation files instead of `context-as-workspace.md`, `README.md` chunks, and `SCOPE.md`, because bench files contain the same vocabulary (recall, retrieval, workspace, context). When the intent classifier signals `wants_explanation: true`, apply a score multiplier (e.g., 1.5×) to stubs from `.md` files over stubs from `.rs` files. Observed in QA loop 0006.
- [ ] Detect and reject model output that contains `[recalled from ...]` markers. The provenance format has leaked into model generation — the model produces fake recall blocks as a generation scaffold (observed in QA loop 0006, session 063609 turn 14). Add a post-processing check in `DynamicRecallOrchestrator::run_turn`: if the model's response contains more than N occurrences of the injection marker pattern, flag the response as corrupted and either retry with a rephrased system prompt or strip the markers before logging. Also consider switching to a provenance format that is less likely to be reproduced by the model (UUID-based or XML-namespaced tags).
