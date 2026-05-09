# Getting Started

## Prerequisites

- Rust toolchain (stable)
- For Ollama adapter: [Ollama](https://ollama.com) running locally
- For Anthropic adapter: `ANTHROPIC_API_KEY` in your environment
- For Groq adapter: `GROQ_API_KEY` in your environment
- For LlamaCpp adapter (feature-gated): libllama installed or `$LLAMA_PATH` set

## Build

```bash
cargo build --workspace
```

Feature flags:

| Flag | Enables |
|---|---|
| `llama` | LlamaCpp native FFI adapter |
| `candle` | Candle CUDA embedding provider |
| `onnx` | ONNX embedding provider |
| `qdrant` | Qdrant vector store |

`candle` and `onnx` are mutually exclusive (both link `onnxruntime` but through different crate versions).

## Run the CLI

```bash
# Default: Ollama intent classifier (llama3.2:3b) + recall loop
cargo run -p caw-cli

# Show the intent classifier output for each query
cargo run -p caw-cli -- --show-intent

# Disable the intent classifier
cargo run -p caw-cli -- --no-intent-classifier

# Use Anthropic for answers
cargo run -p caw-cli -- --adapter anthropic --model claude-sonnet-4-20250514

# LlamaCpp (feature-gated)
cargo run -p caw-cli --features llama -- \
  --adapter llama --model /path/to/model.gguf
```

## Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
# Ollama uses http://localhost:11434 by default
```

## Run Tests

```bash
cargo test --workspace
```

The end-to-end integration test in `crates/caw-orchestrator/tests/end_to_end.rs` uses `MockAdapter` — no API keys or local model needed.

## Minimal Code Example

```rust
use caw_adapters::AnthropicAdapter;
use caw_core::{ContentKind, RecallThresholds};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};
use caw_core::provenance::InMemoryProvenanceStore;

// Build the retrieval stack
let embedder = FastEmbedProvider::bge_small()?;
let dim = embedder.dimension();
let store = SqliteStubStore::in_memory(dim)?;
let index = HnswVectorIndex::new();
let mut retriever = SemanticRetriever::new(embedder, store, index);

// Ingest — cl100k tokenizer, deterministic summaries, adaptive chunking at 2k tokens
let pipeline = IngestionPipeline::new();
let doc = SourceDocument {
    path: "docs/design.md".to_string(),
    content: std::fs::read_to_string("docs/design.md")?,
    kind: ContentKind::Markdown,
    mtime_unix_secs: 0,
};
let content = doc.content.clone();
for stub in pipeline.ingest(doc) {
    retriever.insert(stub, content.clone())?;
}

// Configure the orchestrator
let config = DynamicRecallConfig {
    top_k: 4,
    thresholds: RecallThresholds::default_hysteresis(),
    max_workspace_tokens: 12_000,
    ..Default::default()
};

let trace_embedder = FastEmbedProvider::bge_small()?;
let trace_index = HnswVectorIndex::new();

let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, SqliteStubStore> =
    DynamicRecallOrchestrator::new(
        retriever,
        trace_embedder,
        trace_index,
        InMemoryProvenanceStore::default(),
        AnthropicAdapter::claude_sonnet(),
        config,
    );

let response = orchestrator.run_turn(
    "",
    "How should I optimize context for local models?",
)?;

println!("{}", response.answer);
```

A runnable end-to-end example using `MockAdapter` (no API keys) lives at `crates/caw-orchestrator/tests/end_to_end.rs`.
