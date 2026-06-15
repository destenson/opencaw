# Scope — OpenCAW v0.1

*Status: Living reference — kept current. See [docs/README.md](README.md) for the docs index.*

## What v0.1 is

A **Rust library** for context-as-workspace management in LLM applications. Embeddable in any application that constructs LLM prompts. The core value proposition: thinking-trace-driven recall with eviction, consolidation, and context curation — not another RAG wrapper.

v0.1 ships publicly only when it's genuinely useful and adoptable without friction — ergonomics, docs, and error handling are release-blocking.

## v0.1 deliverables

### Must have (blocks usefulness)
- [x] Measurement primitives (recall quality, false-recall heuristic, hysteresis analysis, context efficiency, cooperation metrics) — see `caw-eval`. Wired into a benchmark harness in `caw-bench` (NIAH + opencaw Q&A + code-agent + sysdoc workloads, recall-on vs recall-off at matched budget). Still open: sweep runs against enough seeds to produce tuning recommendations.
- [ ] Model cooperation calibration: per-model benchmarking of probe/annotation/tool protocol compliance, automatic mode selection. Metrics exist; calibration harness does not.
- [x] Curation hooks: history summarization, tool output compression, system prompt budgeting — see `caw-curation`.
- [x] LLM-generated stub summaries for prose: `LlmSummarizer` in `caw-ingest`.
- [x] Summary caching: `StubStore::get_by_content_hash()` lets the CLI skip re-embedding unchanged files.
- [ ] Serious thought into user interface: ergonomics of user interface, ergonomics of the API, documentation, actionable error handling.

### Should have (significant quality improvement)
- [x] Richer consolidation notes: `LlmConsolidation` runs by default when an LLM adapter is available; `MechanicalConsolidation` is the fallback. Pass `--no-llm-consolidation` to opt out.
- [x] Adaptive chunking for large files: `caw-ingest/src/chunking.rs`. Token-threshold splitting with structural boundaries for code and markdown.
- [x] Degradation monitoring and tiered fallback: `caw-orchestrator/src/degradation.rs`. Always active — not opt-in. `with_degradation_monitor()` is called unconditionally.
- [ ] Provenance conflict detection beyond topic overlap: still only Jaccard. Contradicting assertions and inconsistent numbers are undetected.
- [ ] **Async library surface** (`async` Cargo feature, default-on): an additive async facade over the existing sync traits — it adds an `async fn` orchestrator API, it does not replace or flip the sync traits under the flag. caw-core and caw-orchestrator pull no runtime today, so the feature is a real choice: leave it off to embed the recall engine without tokio, turn it on for the async API. Mechanism is the `spawn_blocking` bridge over the sync `run_turn`. Active v0.1 work, not deferred.
- [ ] **Middleware / proxy that runs the orchestrator** (any OpenAI-compatible 3rd-party client): an `--orchestrate` mode in caw-server where the upstream model becomes the orchestrator's `ModelAdapter` (via `OpenAiCompatibleAdapter`) and the server drives the multi-pass recall loop, instead of the single-shot retrieve→inject→forward path. Buffered response for v0 (one SSE chunk when `stream:true`); the existing single-shot path stays the default so v0 doesn't regress. Active v0.1 work. Targets clients that accept a custom OpenAI base URL (Codex, Cursor, aider); Claude Code's Anthropic protocol needs a separate translation shim, still future.

### Nice to have (polish)
- [x] Probe rate limiting for models that thrash: `ProbeRateLimiter` in the degradation module.
- [ ] Insertion-order experiments (relevance-ranked vs. reverse-relevance vs. stub-order): no harness wired up.
- [ ] Few-shot token cost surfacing (no automatic policy — just measurement): callers can use the `Tokenizer` trait directly, but no framework-level utility.

## v2 and beyond (no active work in v0.1)

These are future goals documented in the design doc. Some have placeholder stubs in the codebase to preserve structural intent; these stubs compile but return errors immediately. Don't delete them, but don't invest v0.1 effort in making them functional.

### Deployment targets beyond library
- **Transparent API interception across protocols**: An always-on proxy that intercepts arbitrary client traffic, including non-OpenAI protocols (notably Claude Code's Anthropic protocol, which needs a translation shim). The OpenAI-compatible orchestrator proxy is in v0.1 scope (see "Should have"); cross-protocol interception is the part that remains future.
- **Engine plugins** (vLLM, Ollama native, etc.): Deep integration for mid-token recall. Requires the framework to stabilize first.
- **Server** (caw-server): Functional OpenAI-compatible retrieval-augmentation proxy. The single-shot retrieve→inject→forward path retrieves context, augments the last user message, and forwards with streaming passthrough; it stays the default. The orchestrator-backed `--orchestrate` mode (multi-pass recall) is active v0.1 work — see "Should have". Uses Candle + CUDA for embeddings.

### Streaming mid-token recall (deferred — design unsettled, not an async-readiness gate)
- Async itself is **not** deferred — the `async` feature and the orchestrator-backed proxy are active v0.1 work (see "Should have").
- What remains future is **token-streaming with mid-stream recall**: the `generate_passive` KV-injection path, where recall fires every N tokens and injects into the live KV cache without restarting generation. This is staged because the passive-injection design is unsettled and only an adapter that owns its sampling loop (e.g. `LlamaCppAdapter`) can implement it — an HTTP upstream can't. Multi-pass orchestration (stop-and-restart on new admissions) remains the approach everywhere else. Track this on its own design, gated on settling passive injection, not on async readiness.

### Additional backends (implemented but feature-gated)
All three compile and function when their Cargo feature is enabled. None are on
by default. They are not v0.1 focus and the CLI doesn't wire them up, but they
are no longer stubs.
- **Candle embedding provider** (`candle` feature): Loads BERT-family models from HuggingFace Hub (e.g., BGE). Mutually exclusive with `fastembed` because both link `onnxruntime` native lib via different versions of `ort`.
- **ONNX embedding provider** (`onnx` feature): Loads arbitrary ONNX models from a local path with an adjacent `tokenizer.json`. Mutually exclusive with `fastembed`, same native-lib reason.
- **Qdrant vector store** (`qdrant` feature): Full `qdrant_client` integration with payload indexes and stub persistence. SQLite + HNSW remains the default; Qdrant covers higher-scale deployments.

### Additional adapters
- OpenAI (via the generic `OpenAiCompatibleAdapter`), `ClaudeCodeAdapter` for local CLI integration. Four working remote/local adapters plus MockAdapter cover cloud, fast inference, local, and test. vLLM and llama.cpp speak the OpenAI protocol — no new adapter needed.

### Already implemented, included in v0.1 as-is
- **API embedding provider** (OpenAI functional, Cohere/Voyage stubbed): Works today, useful for v0.1. No gating needed.
- **BM25 + hybrid retrieval**: Fully implemented and integrated. Part of the v0.1 retrieval story.
- **HNSW vector index**: Production-ready via instant-distance. Core v0.1 infrastructure.
- **Multi-pass dynamic orchestration**: Working recall loop with eviction and consolidation. The v0.1 execution model.

## Scope change protocol

Before adding work that isn't in the "must have" or "should have" lists:
1. Ask: does this serve the library target, or a future deployment target?
2. Ask: does this improve recall quality or context curation, or is it infrastructure for something else?
3. If it's for a future target or doesn't directly improve the core value proposition, it goes on the "future" list, not the backlog.
