# OpenCAW - Context as Workspace

A Rust implementation of context-as-workspace architecture for LLMs, enabling intelligent context management through stub-and-recall patterns with provenance tracking.

## Architecture

OpenCAW treats LLM context as a managed workspace rather than a simple container. It implements:

- **On-demand recall**: Retrieval-based context loading that multiplies effective context by 1-2 orders of magnitude
- **Budget scheduling**: Token-aware admission/eviction policies for optimal workspace utilization
- **Provenance tracking**: Every recalled fragment includes precise source locators for grounding
- **Model-agnostic adapters**: Unified interface for API and local model providers

## Crates

- **caw-core**: Shared types, traits, and domain models
- **caw-ingest**: Document parsing, chunking, and metadata extraction
- **caw-index**: Hybrid retrieval (dense + BM25 + structural)
- **caw-scheduler**: Token budget management and workspace scheduling
- **caw-provenance**: Source tracking and grounding verification
- **caw-adapters**: Provider adapters (Anthropic, Groq, Ollama, etc.)
- **caw-orchestrator**: Recall loop orchestration
- **caw-eval**: Metrics (recall@k, precision, faithfulness)
- **caw-cli**: Command-line interface
- **caw-server**: HTTP/gRPC service (placeholder)

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

## Usage Example

```rust
use caw_adapters::AnthropicAdapter;
use caw_core::{ContentKind, Stub, StubId, TokenBudget};
use caw_index::InMemoryIndex;
use caw_orchestrator::{OrchestratorConfig, RecallOrchestrator};
use caw_provenance::InMemoryProvenanceStore;
use caw_scheduler::GreedyBudgetScheduler;

// Set up index with documents
let mut index = InMemoryIndex::default();
index.insert(stub, content);

// Configure orchestrator
let config = OrchestratorConfig {
    top_k: 4,
    load_threshold: 0.3,
    budget: TokenBudget {
        max_total: 16_000,
        reserved_for_prompt: 2_000,
        reserved_for_answer: 2_000,
    },
    ..Default::default()
};

// Create orchestrator with adapter
let mut orchestrator = RecallOrchestrator {
    retriever: index,
    scheduler: GreedyBudgetScheduler,
    provenance: InMemoryProvenanceStore::default(),
    adapter: AnthropicAdapter::claude_sonnet(),
    loaded: Vec::new(),
    config,
};

// Run a turn
let response = orchestrator.run_turn(
    "You are a helpful assistant.",
    "How should I optimize context for local models?"
)?;

println!("{}", response.answer);
```

## Design Principles

1. **Context is a workspace**: Quality over quantity - manage what's active, not just accessible
2. **Recall over retrieval**: Fragments materialize inline with precise provenance
3. **Budget awareness**: Explicit token accounting with reserved bands
4. **Provider independence**: Same substrate for cloud APIs and local models
5. **Measurement first**: Recall@k, precision, grounding metrics before scaling

## Roadmap

- [x] Core trait contracts and types
- [x] Basic ingestion pipeline
- [x] In-memory index and retriever
- [x] Greedy budget scheduler
- [x] Provenance tracking
- [x] Anthropic adapter
- [x] Groq adapter
- [x] Ollama adapter
- [ ] Persistent index (SQLite + HNSW)
- [ ] Thinking-trace recall for reasoning models
- [ ] Mutable stub consolidation (memory)
- [ ] Evaluation harness with benchmarks
- [ ] Server API (gRPC/HTTP)
- [ ] OpenAI adapter
- [ ] vLLM adapter
- [ ] llama.cpp adapter

## License

MIT
