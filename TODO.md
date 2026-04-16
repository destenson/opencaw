# TODO — Concerns and Issues

## Critical Bugs

- [ ] **caw-ingest sha256 formatting**: `format!("{:x}", hasher.finalize())` fails because `sha2::Digest::finalize()` returns `GenericArray<u8, U64>`, which doesn't implement `LowerHex`. Blocks compilation.

- [ ] **Adapters use sync-wrapped async (`block_on`)**: Calling `tokio::runtime::block_on` inside an existing tokio runtime panics. This will break when embedding the library into any async application (including the planned caw-server). Needs proper async trait design or at minimum `spawn_blocking`.

## Architecture

- [ ] **Over-decomposed workspace**: 11 crates for ~2400 lines. `caw-provenance` (17 lines), `caw-eval` (37 lines), `caw-scheduler` (81 lines) could be modules within `caw-core`. The crate boundaries impose real costs — feature flag coordination, version pinning, repeated linking — without providing meaningful encapsulation at this scale. Consider consolidating until the codebase actually needs the separation.

- [ ] **`DynamicRecallOrchestrator::run_turn` sequencing problem**: The orchestrator completes the full response *before* processing thinking traces and probes. But the purpose of thinking-trace recall is to load documents *during* reasoning so they influence subsequent steps. Without streaming integration, recalled documents only affect the next turn, not the current one. This makes thinking-trace recall largely ineffective for its stated purpose.

## Missing Core Capabilities

- [ ] **No hybrid retrieval**: Embedding-only search fails on exact-match, keyword, and reasoning-shaped queries. BM25 + dense retrieval fusion is well-understood and would meaningfully improve recall on the hardest, most valuable queries. The design doc acknowledges this gap.

- [ ] **No streaming recall loop**: The entire architecture is single-turn request/response. The novel value proposition — loading documents mid-reasoning based on thinking traces — requires streaming token output with interleaved retrieval. Without this, the system is a conventional RAG pipeline with extra steps.

- [ ] **No mutable stub consolidation**: When a document evicts from the workspace, the stub should absorb what was learned during the loaded period. This is described in the design doc (Section 4) but not implemented. It's one of the most interesting ideas in the project.

- [ ] **Eviction is a no-op**: `evict_low_score_fragments()` in `DynamicRecallOrchestrator` is an empty stub. The system can only grow the workspace until it hits the budget ceiling; it cannot intelligently shrink it. The greedy scheduler in `caw-scheduler` handles admission but doesn't re-score loaded fragments against evolving query context.

- [ ] **No provenance reconciliation**: `InMemoryProvenanceStore` records fragments but nothing checks for conflicts between stub summaries and actual content, or between multiple recalled fragments. The design doc (Section 3) identifies this as a real failure mode.

## Scaling and Performance

- [ ] **Linear-scan vector search in SQLite**: Every query does O(n) cosine similarity computations across all stored embeddings. Acceptable for demos with hundreds of documents, unusable for the thousands-of-files workloads the design doc envisions. Qdrant (stubbed) or an HNSW implementation is needed before real use.

- [ ] **No summary caching**: The design doc specifies caching summaries on `(path, mtime, hash)`, but the ingestion pipeline regenerates summaries on every run. For large or frequently-changing corpora, this is a meaningful cost.

## Risk

- [ ] **Commodity work vs. novel work imbalance**: Most completed code is adapter/provider plumbing (Anthropic, Groq, Ollama, FastEmbed, SQLite). Most "in progress" roadmap items are more of the same (OpenAI adapter, Qdrant store, ONNX embeddings). The features that make this project distinctive — streaming recall, mutable stubs, hybrid retrieval, curation policy — remain unimplemented. There's a gravitational pull toward adding backends instead of solving the hard problems.

- [ ] **Narrowing window for external context management**: Model providers are building native context management (project files, extended thinking, tool use). The strongest counter-argument is that this works across models and enforces curation that providers won't. But the value proposition needs to be demonstrated on real workloads before the window closes.
