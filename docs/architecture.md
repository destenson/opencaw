# Architecture

*Status: Living reference — kept current. See [docs/README.md](README.md) for the docs index.*

OpenCAW treats LLM context as a managed workspace rather than a simple container. The core loop: ingest documents → embed and index them → replace file references in prompts with lightweight stubs → dynamically recall full content as the model reasons about it → evict stale fragments and consolidate what was learned.

## Crates

| Crate | Role |
|---|---|
| **caw-core** | Shared types, traits, tokenizer abstractions |
| **caw-ingest** | Document parsing, adaptive chunking, tree-sitter outlines, summaries |
| **caw-index** | Embedding providers, vector stores, BM25, hybrid retrieval |
| **caw-transform** | Prompt transformer — replaces file references with stubs |
| **caw-adapters** | Model adapters (Anthropic, Groq, Ollama, OpenAI-compat, ClaudeCode, Mock, LlamaCpp) |
| **caw-llama-sys** | FFI bindings to libllama.so (feature-gated; built via pkg-config or `$LLAMA_PATH`) |
| **caw-orchestrator** | `DynamicRecallOrchestrator`, degradation monitor, consolidation |
| **caw-curation** | History summarization, tool output compression, system prompt budgeting |
| **caw-eval** | Recall and cooperation metrics; `SessionEvaluator` records events during turns and is consumed by caw-orchestrator and caw-bench |
| **caw-cli** | Command-line interface |
| **caw-bench** | Benchmark harness (NIAH, opencaw, sysdoc Q&A workloads) and `caw-bench-build-index` |
| **caw-server** | OpenAI-compatible retrieval-augmentation proxy (in progress) |

## Design Principles

1. **Context is a workspace**: quality over quantity — manage what's active, not just accessible
2. **Recall over retrieval**: fragments materialize inline with precise provenance
3. **Budget awareness**: explicit token accounting with reserved bands
4. **Provider independence**: same substrate for cloud APIs and local models
5. **Measurement first**: recall@k, precision, grounding metrics before scaling

## Data Flow

### Indexing (background / offline)

```
Source documents
      │ IngestionPipeline (caw-ingest)
      ▼
 Chunks + outlines + summaries
      │ EmbeddingProvider (caw-index)
      ▼
 SqliteStubStore  ←──────────────────────── corpus on disk
 (stubs + embeddings)
```

### Per-turn: DynamicRecallOrchestrator

```
 User query
      │
      ├─ cross-turn eviction ──────────────────────────────────────────────┐
      │  (decay relevance scores; evict below-threshold fragments)          │
      │                                                                     │
      ├─ Phase 1: initial retrieval                                         │
      │   query → embed → vector search → score vs. load threshold         │
      │   ├─ ≤ max_initial_fragments above threshold: load directly        │
      │   └─ > max_initial_fragments:  surface candidate list              │
      │                                                                     │
      ├─ session history recall (separate in-memory vector index)          │
      │                                                                     │
      ▼                                                                     │
 CompletionRequest                                                          │
 (system prompt + workspace fragments)                                      │
      │                                                                     │
      ▼                                                                     │
 ModelAdapter.complete()  [or generate_passive() if adapter supports it]   │
      │                                                                     │
      ▼                                                                     │
 CompletionResponse  (answer + optional thinking trace)                    │
      │                                                                     │
      ├─ Phase 2: iterative refinement (repeats up to max_recall_iterations)│
      │   ├─ extract thinking-trace steps → embed each → vector search → load
      │   ├─ extract <probe> tags → search → load                          │
      │   ├─ extract path:line-range references → read range → load        │
      │   ├─ extract <note> annotations → persist as consolidation notes   │
      │   ├─ decay + refresh relevance scores                              │
      │   ├─ evict stale / over-budget fragments  ────────────────────────►┘
      │   │   (on eviction: synthesize consolidation note, persist to store)
      │   └─ if new context admitted: re-complete; keep best answer        │
      │      converged when net new tokens < convergence_min_new_tokens    │
      │                                                                     │
      ├─ write turn to session file; embed for cross-turn recall           │
      ├─ record metrics to SessionEvaluator (if attached)                  │
      └─ return best CompletionResponse
```

## Further Reading

- [Origin doc](origin.md) — founding design (frozen): stub-and-recall architecture, eviction policy, curation
- [Scope](scope.md) — v0.1 deliverables and boundaries
- [Decisions and Defaults](DECISIONS.md) — authoritative reference for settled implementation choices, feature defaults, and the decision protocol for ambiguous cases
- [Codebase review](archive/codebase-review.md) — frozen snapshot (2026-05-09): implementation status, structural gaps, debt inventory
