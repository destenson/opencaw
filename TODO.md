# TODO — Concerns and Issues

## Critical Bugs

- [x] **caw-ingest sha256 formatting**: Fixed. Now iterates bytes with `format!("{:02x}")`.

- [ ] **Adapters use sync-wrapped async (`block_on`)**: Adapters now take `Arc<Runtime>` and call `runtime.block_on()`, which avoids the "create runtime inside runtime" panic. But calling `block_on` from within an async context (e.g., a tokio task in caw-server) will still panic. The server will need either fully async adapter traits or `spawn_blocking` wrappers.

## Architecture

- [ ] **Over-decomposed workspace**: 11 crates for the current codebase size. `caw-provenance`, `caw-eval`, `caw-scheduler` are still very small and could be modules. Not blocking, but adds friction.

- [x] **Storage/index responsibility split**: Fixed. `VectorStore` trait was doing double duty (persistence + similarity search). Now cleanly separated into `StubStore` (persistence: stubs, content, embeddings) and `VectorIndex` (similarity search with proper indexing). SQLite implements `StubStore`; HNSW implements `VectorIndex`. This is the right decomposition.

- [ ] **`DynamicRecallOrchestrator::run_turn` sequencing problem**: Still present. The orchestrator completes the full response *before* processing thinking traces and probes. Recalled documents only affect the next turn, not the current one. The thinking-trace recall path now uses `VectorIndex` directly (good), but the fundamental issue is that without streaming, mid-reasoning recall can't happen. The probe/trace processing after completion is essentially pre-loading for a hypothetical next turn.

## Missing Core Capabilities

- [ ] **No hybrid retrieval**: Embedding-only search still the only path. BM25 + dense fusion would improve recall on keyword and exact-match queries.

- [ ] **No streaming recall loop**: Still single-turn request/response throughout. This remains the biggest gap between the design doc's vision and what's implemented.

- [ ] **No mutable stub consolidation**: Not implemented. Stubs are immutable after ingestion.

- [ ] **Eviction is absent**: The `DynamicRecallOrchestrator` removed the empty `evict_low_score_fragments` stub entirely — it now relies purely on budget limits in `load_fragments` to cap growth. There's no mechanism to shrink the workspace based on relevance decay.

- [ ] **No provenance reconciliation**: `InMemoryProvenanceStore` still just records and returns. No conflict detection between stub summaries and recalled content.

## Scaling and Performance

- [x] **Linear-scan vector search**: Fixed. HNSW index implemented via `instant-distance` crate (`HnswVectorIndex`). Builds an immutable graph, rebuilds lazily on next search after inserts. Uses cosine distance. Good for the expected corpus sizes (hundreds to low thousands). The `VectorIndex` trait makes swapping implementations straightforward.

- [ ] **No summary caching**: Ingestion still regenerates summaries every run. The `StubStore` now persists stubs (including summaries) in SQLite, so the infrastructure for cache-checking on `(content_hash, mtime)` exists — but the ingestion pipeline doesn't check whether a stub already exists before regenerating.

## Risk

- [ ] **Commodity work vs. novel work imbalance**: The HNSW index and StubStore/VectorIndex split are good architectural progress. But the distinctive features — streaming recall, mutable stubs, hybrid retrieval, curation policy — still aren't implemented. The next round of work should prioritize these over adding more adapter/provider backends.

- [ ] **Narrowing window for external context management**: Model providers continue building native context management. The value proposition needs demonstration on real workloads.
