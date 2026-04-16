# TODO — Concerns and Issues

## Critical Bugs

- [x] **caw-ingest sha256 formatting**: Fixed. Now iterates bytes with `format!("{:02x}")`.

- [ ] **Adapters use sync-wrapped async (`block_on`)**: Adapters take `Arc<Runtime>` and call `runtime.block_on()`, avoiding the "create runtime inside runtime" panic. But calling `block_on` from within an async context (e.g., a tokio task in caw-server) will still panic. The server will need either fully async adapter traits or `spawn_blocking` wrappers.

## Architecture

- [ ] **Over-decomposed workspace**: 11 crates for the current codebase size. `caw-provenance`, `caw-eval`, `caw-scheduler` are still small and could be modules. Not blocking, but adds friction.

- [x] **Storage/index responsibility split**: `VectorStore` decomposed into `StubStore` (persistence) + `VectorIndex` (similarity search). SQLite implements `StubStore`; HNSW implements `VectorIndex`.

- [x] **`DynamicRecallOrchestrator::run_turn` sequencing problem**: Fixed with iterative multi-pass recall. After the initial completion, probes/thinking traces are extracted, new fragments loaded, and if the workspace changed, the model re-completes with enriched context. Repeats up to `max_recall_iterations` (default 3) or until convergence.

## Core Capabilities

- [x] **Hybrid retrieval**: Implemented `BM25Index` (standard BM25 scoring with IDF weighting) and `HybridRetriever` that fuses semantic + keyword search using min-max normalized scores with configurable weights (default 0.6/0.4 semantic/keyword).

- [ ] **No streaming recall loop**: Still single-turn request/response per iteration. True streaming (interleaving retrieval with token generation mid-response) remains a future goal. The multi-pass approach gets most of the value for evaluation/testing but doesn't match the design doc's vision of mid-reasoning recall.

- [x] **Eviction with consolidation**: Implemented term-overlap-based relevance scoring for eviction. When workspace is 80%+ full, fragments below `eviction_relevance_floor` (default 0.15) are evicted, freeing space to 70% budget. On eviction, a `ConsolidationNote` is generated recording the query context and source path, attached to the stub via the provenance store.

- [x] **Mid-session annotation**: Models can emit `<note id="stub_id">content</note>` markers. The orchestrator extracts these and records them as `ModelAnnotation` consolidation notes on the corresponding stub. Both eviction-time and mid-session consolidation are available, as the design doc suggested prototyping.

- [x] **Provenance ledger with topic overlap detection**: `ProvenanceLedger` extends `ProvenanceStore` with query context tracking, per-stub consolidation notes, and topic overlap detection. Extracts top-20 non-stopword terms per fragment, computes Jaccard overlap between fragments from different source files, and surfaces overlaps above 30% as warnings injected into the next completion.

## Scaling and Performance

- [x] **Linear-scan vector search**: Fixed. HNSW index via `instant-distance`.

- [ ] **No summary caching**: Ingestion regenerates summaries every run. `StubStore` now persists stubs (including summaries) in SQLite, so the infrastructure for cache-checking on `(content_hash, mtime)` exists — but the ingestion pipeline doesn't check whether a stub already exists before regenerating.

## Remaining Design Doc Items

- [ ] **Streaming recall loop**: The design doc envisions streaming tokens and interleaving retrieval mid-generation. Requires async streaming adapter traits. The multi-pass approach is a practical substitute but not equivalent.

- [ ] **Consolidation note enrichment**: Currently, eviction notes are mechanical ("evicted during query about X"). Richer consolidation — summarizing what portions were referenced and what conclusions drawn — would require an LLM call or deeper analysis of provenance records. Left as a future enhancement.

- [ ] **Provenance conflict detection beyond topic overlap**: The current ledger detects when fragments from different sources have high term overlap (potential contradiction). Deeper conflict detection — contradicting assertions, inconsistent numbers, negation patterns — is marked as a future enhancement per the design doc.

- [ ] **Curation hooks** (design doc section 8.1): Four hooks are specified but none are implemented:
  - [ ] History summarization — compress old conversation turns when history exceeds a token threshold
  - [ ] Tool output compression — stub verbose tool results using the same stub architecture
  - [ ] System prompt budgeting — measure and warn/truncate when system prompts exceed budget
  - [ ] Few-shot management — surface token cost of examples (no automatic policy)

- [ ] **Adaptive chunking** (design doc section 3.1): Large files should be automatically chunked and each chunk indexed independently. The ingestion pipeline currently treats every file as a single unit. Open question: whether the chunk threshold should be token-based, structural (function/section boundaries), or both.

- [ ] **Degradation monitoring** (design doc section 6): Per-component health checks and tiered fallback (full recall → stubs+tools → pass-through) are specified but not implemented. Includes probe rate limiting for models that thrash.

- [ ] **Measurement infrastructure** (design doc section 10): `caw-eval` only has recall@k and precision@k. Hysteresis threshold tuning, insertion-order experiments, and false-recall rate measurement all depend on richer instrumentation that doesn't exist yet. This is a prerequisite for tuning several parameters the design doc explicitly defers to measurement.
