# TODO — Honest Status

Cross-reference: design doc is `context-as-workspace.md`, scope boundaries are in `SCOPE.md`.

## Retrieval

- [x] **BM25 keyword index**: Standard BM25 scoring (K1=1.2, B=0.75) with IDF weighting.
- [x] **Hybrid retrieval fusion**: Semantic + BM25 with min-max normalized scores, configurable weights (default 0.6/0.4).
- [x] **Asymmetric embeddings**: `EmbeddingProvider` trait has `embed_query()`/`embed_document()` with default delegation to `embed()` for symmetric models. `FastEmbedProvider` overrides both for BGE models (adds `"query: "` / `"passage: "` prefixes). All call sites updated: `SemanticRetriever`, CLI indexing, `DynamicRecallOrchestrator`.

## Ingestion & Indexing

- [x] **File ingestion pipeline**: Reads filesystem, detects content kind, captures mtime, generates stubs.
- [x] **SHA256 content hashing**: Fixed formatting with `format!("{:02x}")`.
- [x] **Summary caching**: `StubStore::get_by_content_hash()` allows callers to skip re-ingestion and re-embedding when file content hasn't changed. CLI ingestion loop checks the store before generating embeddings.
- [ ] **LLM-generated summaries**: Stub summaries are deterministic extraction (first heading + paragraph for markdown, function list for code). The design doc calls for LLM-generated 1-3 sentence summaries for prose. Deterministic extraction is fine for code but insufficient for prose triage.
- [ ] **Target-model tokenizer**: Token estimation uses `content.split_whitespace().count()` everywhere. The design doc specifies using the target model's actual tokenizer. Whitespace splitting diverges significantly from real token counts, especially for code.
- [ ] **Tree-sitter outlines**: Code outline extraction uses `starts_with("pub fn ")` string matching, Rust-centric with minimal Python/JS support. Misses items inside impl blocks, attributed functions, and most languages. Tree-sitter would give correct, language-agnostic symbol extraction.
- [ ] **Adaptive chunking**: Files are treated as single units regardless of size. Large files should be chunked (token-based threshold at minimum, structural boundaries ideally) with each chunk indexed independently.
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

None implemented. These are the "context quality" half of the framework.

- [ ] **History summarization**: Compress old conversation turns when history exceeds a token threshold. Preserve decisions and facts established; drop dead-end reasoning and verbose tool output from completed steps.
- [ ] **Tool output compression**: Stub verbose tool results using the same stub architecture. Full output stored in index, context receives a summary.
- [ ] **System prompt budgeting**: Measure system prompt token usage, warn or truncate when budget exceeded.
- [ ] **Few-shot management**: Surface token cost of each example. No automatic policy — framework measures, deployment decides.

## Measurement (design doc section 10)

Minimal. This is the prerequisite for tuning everything above.

- [x] **Basic eval metrics**: recall@k and precision@k over RecallFragment.
- [ ] **False-recall rate**: Track provenance-tagged conflicts — cases where recalled content contradicts what the model expected from the stub summary.
- [ ] **Effective vs nominal context ratio**: Measure how much of the context window is doing useful work vs noise.
- [ ] **Hysteresis threshold tuning**: Instrument the load/unload decisions to find optimal thresholds per workload.
- [ ] **Insertion-order experiments**: Test relevance-ranked vs reverse-relevance vs original-stub order for recalled content placement.
- [ ] **Model cooperation calibration**: Per-model benchmarking of probe emission reliability, annotation quality, tool usage effectiveness. Drives automatic cooperative-vs-transparent mode selection.

## Degradation & Monitoring (design doc section 6)

Not implemented.

- [ ] **Per-component health checks**: Embedding service latency/error rate, summary generator queue depth, probe rate monitoring.
- [ ] **Tiered fallback**: Full recall → stubs+tools → pass-through, with automatic recovery when components come back.
- [ ] **Probe rate limiting**: Throttle automatic recall if model emits probes at excessive rate (thrashing or gaming).

## Infrastructure

- [ ] **Over-decomposed workspace**: 11 crates for the current codebase size. `caw-provenance`, `caw-eval`, `caw-scheduler` could be modules within larger crates. Not blocking but adds friction.
