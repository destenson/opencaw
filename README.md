# OpenCAW - Context as Workspace

A Rust implementation of context-as-workspace architecture for LLMs, enabling intelligent context management through stub-and-recall patterns with provenance tracking.

## Architecture

OpenCAW treats LLM context as a managed workspace rather than a simple container. It implements:

- **On-demand recall**: Retrieval-based context loading that multiplies effective context by 1-2 orders of magnitude
- **Budget scheduling**: Token-aware admission/eviction policies for optimal workspace utilization
- **Provenance tracking**: Every recalled fragment includes precise source locators for grounding
- **Model-agnostic adapters**: Unified interface for API and local model providers

## Crates

- **caw-core**: Shared types, traits, tokenizer abstractions
- **caw-ingest**: Document parsing, adaptive chunking, tree-sitter outlines, summaries
- **caw-index**: Embedding providers, vector stores, BM25, hybrid retrieval
- **caw-transform**: Prompt transformer — replaces file references with stubs
- **caw-scheduler**: Token budget admission/eviction (greedy)
- **caw-provenance**: In-memory store + provenance ledger with overlap detection
- **caw-adapters**: Model adapters (Anthropic, Groq, Ollama, OpenAI-compat, ClaudeCode, Mock)
- **caw-orchestrator**: `DynamicRecallOrchestrator`, degradation monitor, consolidation
- **caw-curation**: History summarization, tool output compression, system prompt budgeting
- **caw-eval**: `SessionEvaluator` and metrics (recall@k, false-recall, hysteresis, cooperation)
- **caw-cli**: Command-line interface tying it all together
- **caw-bench**: Benchmark harness (NIAH + opencaw Q&A workloads, recall-on vs recall-off)
- **caw-server**: HTTP/gRPC service (scaffold only)

## Embedding Providers

### FastEmbed (MVP - default)
- Rust-native with bundled quantized models
- BGE-small-en-v1.5 (384d), BGE-base-en-v1.5 (768d)
- No external dependencies

```rust
use caw_index::FastEmbedProvider;

let embedder = FastEmbedProvider::bge_small()?;
```

### API-based (OpenAI, Cohere, Voyage)
```rust
use caw_index::ApiEmbeddingProvider;

let embedder = ApiEmbeddingProvider::openai_small()?;
```

### ONNX (feature-gated)
- Custom models via the `ort` crate
- Loads a model file plus adjacent `tokenizer.json`
- Enable with `--features onnx`. Mutually exclusive with `fastembed`.

### Candle (feature-gated)
- BERT-family models downloaded directly from HuggingFace Hub
- Enable with `--features candle`. Mutually exclusive with `fastembed`.

## Stub Storage

Stubs (with embeddings and raw content) are persisted via the `StubStore`
trait. Vector similarity lives in a separate `VectorIndex` so the storage
layer can focus on durability and the index on search performance.

### SQLite (default)
- Single-file database; in-memory variant for tests
- Persists stubs, embeddings, content, and consolidation notes
- Works for datasets up to ~100k stubs without special tuning

```rust
use caw_index::SqliteStubStore;

let store = SqliteStubStore::new("index.db", 384)?;
let store = SqliteStubStore::in_memory(384)?;
```

### Qdrant (feature-gated)
- Full `qdrant_client` integration with payload indexes
- For deployments that outgrow the SQLite store
- Enable with `--features qdrant`

```rust
use caw_index::QdrantStubStore;

let store = QdrantStubStore::local("collection_name")?;
```

### Vector index
- `HnswVectorIndex` via `instant-distance` is the default search layer; pairs
  with either store.

## Supported Adapters

### Anthropic
- Claude Sonnet 4+
- Claude Opus 4+
- Extended thinking (hidden reasoning) support

```rust
use caw_adapters::AnthropicAdapter;

let adapter = AnthropicAdapter::claude_sonnet();
// or
let adapter = AnthropicAdapter::new("api_key", "claude-sonnet-4-20250514");
```

### Groq
- Llama 3.3 70B Versatile
- Llama 3.1 8B Instant
- Mixtral 8x7B

```rust
use caw_adapters::GroqAdapter;

let adapter = GroqAdapter::llama_70b();
// or
let adapter = GroqAdapter::new("api_key", "llama-3.3-70b-versatile");
```

### Ollama (Local)
- Any Ollama-compatible model
- DeepSeek R1 (with visible reasoning)
- Qwen 2.5
- Llama 3.2

```rust
use caw_adapters::OllamaAdapter;

let adapter = OllamaAdapter::deepseek_r1();
// or
let adapter = OllamaAdapter::local("model_name");
// or
let adapter = OllamaAdapter::new("http://custom:11434", "model_name");
```

### OpenAI-compatible
- Generic adapter for any provider speaking the chat completions protocol:
  OpenAI, vLLM, Perplexity, HuggingFace Inference Endpoints, llama.cpp-server
- Configurable headers and per-deployment capability flags

```rust
use caw_adapters::OpenAiCompatibleAdapter;
```

### ClaudeCode
- Local CLI integration for running recall against the Claude Code binary

```rust
use caw_adapters::ClaudeCodeAdapter;
```

### MockAdapter
- Echoes `format_workspace(...)` back into the answer. Use in tests and
  examples — no network, no API keys.

## Quick Start

```bash
# Build the workspace
cargo build --workspace

# Run CLI demo
cargo run -p caw-cli

# Run tests
cargo test --workspace
```

## Environment Variables

```bash
# For Anthropic adapter
export ANTHROPIC_API_KEY="sk-ant-..."

# For Groq adapter
export GROQ_API_KEY="gsk_..."

# Ollama uses default local endpoint (http://localhost:11434)
```

## Usage Example — Dynamic Recall

```rust
use caw_adapters::AnthropicAdapter;
use caw_core::{ContentKind, RecallThresholds};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};
use caw_provenance::InMemoryProvenanceStore;

// Build the retrieval stack
let embedder = FastEmbedProvider::bge_small()?;
let dim = embedder.dimension();
let store = SqliteStubStore::in_memory(dim)?;
let index = HnswVectorIndex::new();
let mut retriever = SemanticRetriever::new(embedder, store, index);

// Ingest — IngestionPipeline::new() defaults to cl100k tokenizer,
// deterministic summaries, and adaptive chunking at 2k tokens.
let pipeline = IngestionPipeline::new();
let doc = SourceDocument {
    path: "context-as-workspace.md".to_string(),
    content: std::fs::read_to_string("context-as-workspace.md")?,
    kind: ContentKind::Markdown,
    mtime_unix_secs: 0,
};
let content = doc.content.clone();
for stub in pipeline.ingest(doc) {
    retriever.insert(stub, content.clone())?;
}

// Configure the orchestrator. DynamicRecallOrchestrator takes a second
// embedder + vector index pair used by thinking-trace and probe recall;
// here we reuse a fresh instance of each.
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
    "You are a helpful assistant.",
    "How should I optimize context for local models?",
)?;

println!("{}", response.answer);
```

A runnable end-to-end example using `MockAdapter` (no API keys needed) lives at
`crates/caw-orchestrator/tests/end_to_end.rs`.

## Design Principles

1. **Context is a workspace**: Quality over quantity - manage what's active, not just accessible
2. **Recall over retrieval**: Fragments materialize inline with precise provenance
3. **Budget awareness**: Explicit token accounting with reserved bands
4. **Provider independence**: Same substrate for cloud APIs and local models
5. **Measurement first**: Recall@k, precision, grounding metrics before scaling

## Roadmap

See `TODO.md` for line-item status and `SCOPE.md` for v1 boundaries.

### Working today
- Core types and trait contracts (`caw-core`)
- Ingestion pipeline with adaptive chunking, tree-sitter outlines, deterministic + LLM summaries, cl100k token estimation (`caw-ingest`)
- Hybrid retrieval: semantic (embeddings) + BM25, min-max normalized fusion
- HNSW vector index via `instant-distance`
- SQLite stub store with consolidation persistence; Qdrant as feature-gated alternative
- FastEmbed (BGE), API (OpenAI/Cohere/Voyage shape), Candle, ONNX embedding providers
- `DynamicRecallOrchestrator`: multi-pass recall with probes, thinking-trace extraction, relevance decay, budget-triggered eviction, and consolidation notes
- Degradation monitoring with tiered fallback and probe rate limiting
- Curation pipeline: history summarization, tool output compression, system prompt budgeting
- Provenance ledger with inline source tagging (XML for Anthropic, bracketed for OpenAI-shape)
- Adapters: Anthropic, Groq, Ollama, OpenAI-compatible (covers vLLM / Perplexity / HF Inference / llama.cpp-server), ClaudeCode, Mock
- Evaluation primitives: `SessionEvaluator` with recall metrics, false-recall heuristic, hysteresis analysis, context efficiency, cooperation metrics
- End-to-end integration test in `crates/caw-orchestrator/tests/end_to_end.rs`

### Open
- Sweep runs of the benchmark harness (caw-bench) across enough seeds and workloads to produce threshold-tuning recommendations and cooperation-calibration numbers
- Richer consolidation notes as default (LLM-synthesized, not mechanical)
- Background indexer with lazy fallback (ingestion is currently synchronous, single-pass)
- Insertion-order experiments (relevance-ranked vs reverse-relevance vs stub-order)
- Provenance conflict detection beyond Jaccard term overlap
- Additional prompt transformer surfaces (fenced blocks with `path=`, bare-path regex)
- Few-shot token cost surfacing utility

### Deferred (v2+, see SCOPE.md)
- Server API (gRPC/HTTP) — `caw-server` is a scaffold
- Async adapter traits / middleware proxy deployment target
- Engine plugins for mid-stream recall (vLLM, Ollama native)
- Streaming recall interleaved with token generation

## License

MIT
