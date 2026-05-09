# OpenCAW — Context as Workspace

A Rust library for thinking-trace-driven context management in LLM applications. OpenCAW treats context as a managed workspace: documents are indexed as lightweight stubs, and full content is recalled inline as the model reasons about it — multiplying effective context by 1–2 orders of magnitude at flat compute cost.

This is not another RAG wrapper. The differentiator is thinking-trace-as-retrieval-signal: the model's own reasoning drives what gets loaded, evicted, and consolidated, rather than a retrieval step that runs before inference.

## Quick Start

```bash
cargo build --workspace
cargo run -p caw-cli          # Ollama + recall loop, interactive
cargo test --workspace        # includes end-to-end integration test (no API keys needed)
```

See [docs/getting-started.md](docs/getting-started.md) for adapter setup, environment variables, and a minimal code example.

## Documentation

| | |
|---|---|
| [Architecture](docs/architecture.md) | Crates, data flow, design principles |
| [Getting Started](docs/getting-started.md) | Build, run, environment variables, code example |
| [Adapters](docs/adapters.md) | Anthropic, Groq, Ollama, OpenAI-compat, LlamaCpp, Mock |
| [Embedding Providers](docs/embedding-providers.md) | FastEmbed, API, Candle, ONNX |
| [Storage](docs/storage.md) | SQLite, Qdrant, HNSW, hybrid retrieval |
| [Benchmarking](docs/benchmarking.md) | NIAH, opencaw, sysdoc workloads; sweep harness; intent bench |
| [Design](docs/design.md) | Thesis, stub-and-recall architecture, eviction policy, curation |
| [Scope](docs/scope.md) | v0.1 deliverables, what's in and out, scope change protocol |
| [Bugs](docs/bugs.md) | Known bugs and regression log |
| [Codebase Review](docs/codebase-review.md) | Implementation status, structural gaps, debt inventory |

Development tracking is in [TODO.md](TODO.md).

## License

MIT
