# Architecture

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
| **caw-eval** | `SessionEvaluator` and metrics (recall@k, false-recall, hysteresis, cooperation) |
| **caw-cli** | Command-line interface |
| **caw-bench** | Benchmark harness (NIAH, opencaw, sysdoc Q&A workloads) and `caw-bench-build-index` |
| **caw-server** | OpenAI-compatible retrieval-augmentation proxy |

## Design Principles

1. **Context is a workspace**: quality over quantity — manage what's active, not just accessible
2. **Recall over retrieval**: fragments materialize inline with precise provenance
3. **Budget awareness**: explicit token accounting with reserved bands
4. **Provider independence**: same substrate for cloud APIs and local models
5. **Measurement first**: recall@k, precision, grounding metrics before scaling

## Data Flow

```
                      ┌─────────────────────┐
                      │   Background Indexer │
                      │  (summaries, outlines│
                      │   embeddings, chunks)│
                      └──────────┬──────────┘
                                 │ populates
                                 ▼
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│  User    │──▶│   Prompt     │──▶│  Stub Index   │
│  Prompt  │   │  Transformer │   │  (cache store) │
└──────────┘   └──────┬───────┘   └───────┬───────┘
                      │ stubbed prompt      │ embedding lookup
                      ▼                     │
               ┌──────────────┐             │
               │  Orchestrator │◀────────────┘
               │  (multi-turn) │
               └──────┬───────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   ┌───────────┐ ┌─────────┐ ┌──────────┐
   │  LLM      │ │ Probe   │ │ Tool     │
   │  Generate  │ │ Matcher │ │ Handler  │
   └─────┬─────┘ └────┬────┘ └────┬─────┘
         │            │           │
         │    matched stubs  file content
         │            │           │
         └────────────┴───────────┘
                      │
               ┌──────▼───────┐
               │  Provenance  │
               │  Tagger      │
               └──────┬───────┘
                      │
               ┌──────▼───────┐
               │  Eviction /  │
               │  Consolidation│
               └──────────────┘
```

## Further Reading

- [Design doc](design.md) — thesis, stub-and-recall architecture, eviction policy, curation
- [Scope](scope.md) — v1 deliverables and boundaries
- [Codebase review](codebase-review.md) — implementation status, structural gaps, debt inventory
