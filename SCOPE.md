# Scope — OpenCAW v1

## What v1 is

A **Rust library** for context-as-workspace management in LLM applications. Embeddable in any application that constructs LLM prompts. The core value proposition: thinking-trace-driven recall with eviction, consolidation, and context curation — not another RAG wrapper.

## v1 deliverables

### Must have (blocks usefulness)
- Measurement infrastructure: recall quality, false-recall rate, hysteresis tuning, effective vs. nominal context ratio
- Model cooperation calibration: per-model benchmarking of probe/annotation/tool protocol compliance, automatic mode selection
- Curation hooks: history summarization, tool output compression, system prompt budgeting
- LLM-generated stub summaries for prose (deterministic extraction is fine for code)
- Summary caching in the ingestion pipeline (infrastructure exists in SQLite, not wired up)

### Should have (significant quality improvement)
- Richer consolidation notes (LLM-summarized, not just mechanical "evicted during query about X")
- Adaptive chunking for large files (token-based threshold at minimum)
- Degradation monitoring and tiered fallback (section 6 of design doc)
- Provenance conflict detection beyond topic overlap (contradicting assertions, inconsistent values)

### Nice to have (polish)
- Probe rate limiting for models that thrash
- Insertion-order experiments (relevance-ranked vs. reverse-relevance vs. stub-order)
- Few-shot token cost surfacing (no automatic policy — just measurement)

## v2 and beyond (no active work in v1)

These are future goals documented in the design doc. Some have placeholder stubs in the codebase to preserve structural intent; these stubs compile but return errors immediately. Don't delete them, but don't invest v1 effort in making them functional.

### Deployment targets beyond library
- **Middleware / proxy**: Transparent LLM API interception. Requires async refactoring, session state over HTTP, concurrent requests. Separate engineering problem from the core framework.
- **Engine plugins** (vLLM, Ollama native, etc.): Deep integration for mid-token recall. Requires the framework to stabilize first.
- **Server** (caw-server): HTTP/gRPC service. Currently an empty scaffold.

### Async and streaming
- **Async trait refactoring**: Only needed for middleware/server. Sync traits are simpler for library consumers.
- **Streaming mid-token recall**: Multi-pass orchestration is the v1 approach. True streaming requires async adapter traits and engine cooperation.

### Additional backends (stubs exist, not functional)
- **Candle embedding provider**: Stub returns errors. Has fastembed conflicts to resolve.
- **ONNX embedding provider**: Stub returns errors. Adds ort dependency for marginal gain.
- **Qdrant vector store**: Stub returns errors. SQLite + HNSW covers v1 scale.

### Additional adapters
- OpenAI, vLLM, llama.cpp model adapters. Three working adapters (Anthropic, Groq, Ollama) cover cloud, fast inference, and local. More are additive, not architectural.

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
