# Scope — OpenCAW v1

## What v1 is

A **Rust library** for context-as-workspace management in LLM applications. Embeddable in any application that constructs LLM prompts. The core value proposition: thinking-trace-driven recall with eviction, consolidation, and context curation — not another RAG wrapper.

## v1 deliverables

### Must have (blocks usefulness)
- [x] Measurement primitives (recall quality, false-recall heuristic, hysteresis analysis, context efficiency, cooperation metrics) — see `caw-eval`. Still open: a harness that drives these across real workloads to produce tuning numbers.
- [ ] Model cooperation calibration: per-model benchmarking of probe/annotation/tool protocol compliance, automatic mode selection. Metrics exist; calibration harness does not.
- [x] Curation hooks: history summarization, tool output compression, system prompt budgeting — see `caw-curation`.
- [x] LLM-generated stub summaries for prose: `LlmSummarizer` in `caw-ingest`.
- [x] Summary caching: `StubStore::get_by_content_hash()` lets the CLI skip re-embedding unchanged files.

### Should have (significant quality improvement)
- [ ] Richer consolidation notes: `MechanicalConsolidation` is the default; `LlmConsolidation` exists but the default still emits mechanical strings. Open question whether the LLM variant should be on by default under `--llm-consolidation`.
- [x] Adaptive chunking for large files: `caw-ingest/src/chunking.rs`. Token-threshold splitting with structural boundaries for code and markdown.
- [x] Degradation monitoring and tiered fallback: `caw-orchestrator/src/degradation.rs`. Opt-in via `with_degradation_monitor()`.
- [ ] Provenance conflict detection beyond topic overlap: still only Jaccard. Contradicting assertions and inconsistent numbers are undetected.

### Nice to have (polish)
- [x] Probe rate limiting for models that thrash: `ProbeRateLimiter` in the degradation module.
- [ ] Insertion-order experiments (relevance-ranked vs. reverse-relevance vs. stub-order): no harness wired up.
- [ ] Few-shot token cost surfacing (no automatic policy — just measurement): callers can use the `Tokenizer` trait directly, but no framework-level utility.

## v2 and beyond (no active work in v1)

These are future goals documented in the design doc. Some have placeholder stubs in the codebase to preserve structural intent; these stubs compile but return errors immediately. Don't delete them, but don't invest v1 effort in making them functional.

### Deployment targets beyond library
- **Middleware / proxy**: Transparent LLM API interception. Requires async refactoring, session state over HTTP, concurrent requests. Separate engineering problem from the core framework.
- **Engine plugins** (vLLM, Ollama native, etc.): Deep integration for mid-token recall. Requires the framework to stabilize first.
- **Server** (caw-server): HTTP/gRPC service. Currently an empty scaffold.

### Async and streaming
- **Async trait refactoring**: Only needed for middleware/server. Sync traits are simpler for library consumers.
- **Streaming mid-token recall**: Multi-pass orchestration is the v1 approach. True streaming requires async adapter traits and engine cooperation.

### Additional backends (implemented but feature-gated)
All three compile and function when their Cargo feature is enabled. None are on
by default. They are not v1 focus and the CLI doesn't wire them up, but they
are no longer stubs.
- **Candle embedding provider** (`candle` feature): Loads BERT-family models from HuggingFace Hub (e.g., BGE). Mutually exclusive with `fastembed` because both link `onnxruntime` native lib via different versions of `ort`.
- **ONNX embedding provider** (`onnx` feature): Loads arbitrary ONNX models from a local path with an adjacent `tokenizer.json`. Mutually exclusive with `fastembed`, same native-lib reason.
- **Qdrant vector store** (`qdrant` feature): Full `qdrant_client` integration with payload indexes and stub persistence. SQLite + HNSW remains the default; Qdrant covers higher-scale deployments.

### Additional adapters
- OpenAI (via the generic `OpenAiCompatibleAdapter`), `ClaudeCodeAdapter` for local CLI integration. Four working remote/local adapters plus MockAdapter cover cloud, fast inference, local, and test. vLLM and llama.cpp speak the OpenAI protocol — no new adapter needed.

### Already implemented, included in v1 as-is
- **API embedding provider** (OpenAI functional, Cohere/Voyage stubbed): Works today, useful for v1. No gating needed.
- **BM25 + hybrid retrieval**: Fully implemented and integrated. Part of the v1 retrieval story.
- **HNSW vector index**: Production-ready via instant-distance. Core v1 infrastructure.
- **Multi-pass dynamic orchestration**: Working recall loop with eviction and consolidation. The v1 execution model.

## Scope change protocol

Before adding work that isn't in the "must have" or "should have" lists:
1. Ask: does this serve the library target, or a future deployment target?
2. Ask: does this improve recall quality or context curation, or is it infrastructure for something else?
3. If it's for a future target or doesn't directly improve the core value proposition, it goes on the "future" list, not the backlog.
