# OpenCAW - Context as Workspace

A Rust implementation of context-as-workspace architecture for LLMs, enabling
intelligent context management through stub-and-recall patterns with provenance
tracking.

## Architecture

OpenCAW treats LLM context as a managed workspace rather than a simple
container. It implements:

- **On-demand recall**: Retrieval-based context loading that multiplies
  effective context by 1-2 orders of magnitude
- **Budget scheduling**: Token-aware admission/eviction policies for optimal
  workspace utilization
- **Provenance tracking**: Every recalled fragment includes precise source
  locators for grounding
- **Model-agnostic adapters**: Unified interface for API and local model
  providers

## Crates

- **caw-core**: Shared types, traits, tokenizer abstractions
- **caw-ingest**: Document parsing, adaptive chunking, tree-sitter outlines,
  summaries
- **caw-index**: Embedding providers, vector stores, BM25, hybrid retrieval
- **caw-transform**: Prompt transformer — replaces file references with stubs
- **caw-adapters**: Model adapters (Anthropic, Groq, Ollama, OpenAI-compat,
  ClaudeCode, Mock, LlamaCpp)
- **caw-llama-sys**: FFI bindings to libllama.so (feature-gated; generated
  by bindgen at build time via pkg-config, falling back to `$LLAMA_PATH`)
- **caw-orchestrator**: `DynamicRecallOrchestrator`, degradation monitor,
  consolidation
- **caw-curation**: History summarization, tool output compression, system
  prompt budgeting
- **caw-eval**: `SessionEvaluator` and metrics (recall@k, false-recall,
  hysteresis, cooperation)
- **caw-cli**: Command-line interface tying it all together
- **caw-bench**: Benchmark harness (NIAH, opencaw, and sysdoc Q&A workloads,
  recall-on vs recall-off). Also includes `caw-bench-intent` for benchmarking
  small query-intent classifiers. Ships the `caw-bench-build-index` binary for
  streaming, resumable, GPU-pipelined index construction over large corpora.
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

- BERT-family models downloaded directly from HuggingFace Hub (e.g.
  `BAAI/bge-small-en-v1.5`), with asymmetric query/document prefix handling.
- **CUDA accelerated** — automatically uses `Device::new_cuda(0)` when
  available, falling back to CPU with a log line. Builds with `candle-core`
  et al. compiled with the `cuda` feature.
- Truncates to 512 tokens (BGE's max position) and clips input at 32 KB of
  characters so pathological inputs can't stall the BPE tokenizer.
- Enable with `--features candle`. Used by `caw-bench-build-index` for
  bulk embedding on the GPU.

## Stub Storage

Stubs (with embeddings and raw content) are persisted via the `StubStore` trait.
Vector similarity lives in a separate `VectorIndex` so the storage layer can
focus on durability and the index on search performance.

### SQLite (default)

- Single-file database; in-memory variant for tests (NO FILE CONTENT STORED IN THE DATABASE)
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
- Qwen 3.5
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

### LlamaCpp (feature-gated)

- Native llama.cpp inference via FFI — no HTTP server, no restart overhead
- Owns the sampling loop: implements `generate_passive` for true mid-stream recall injection every N tokens directly into the KV cache
- Any GGUF model; GPU offload via CUDA
- Enable with `--features llama`. Discovers the library via pkg-config; set `LLAMA_PATH` to a custom build directory if pkg-config can't find it.

```rust
use caw_adapters::{LlamaCppAdapter, LlamaCppConfig};

let adapter = LlamaCppAdapter::from_path("model.gguf")?;
// or with explicit config:
let adapter = LlamaCppAdapter::new_with(LlamaCppConfig {
    model_path: "model.gguf".into(),
    n_gpu_layers: -1,   // all layers on GPU
    n_ctx: 8192,
    temperature: 0.7,
    ..Default::default()
})?;
```

```bash
# CLI usage (system install — no env vars needed)
cargo run -p caw-cli --features llama -- \
  --adapter llama --model /path/to/model.gguf
```

### MockAdapter

- Echoes `format_workspace(...)` back into the answer. Use in tests and examples
  — no network, no API keys.

## Quick Start

```bash
# Build the workspace
cargo build --workspace

# Run CLI demo (defaults to an Ollama intent classifier: llama3.2:3b)
cargo run -p caw-cli

# Show the classifier output for each query
cargo run -p caw-cli -- --show-intent

# Disable the intent classifier explicitly
cargo run -p caw-cli -- --no-intent-classifier

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
use caw_core::provenance::InMemoryProvenanceStore;

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
    "",
    "How should I optimize context for local models?",
)?;

println!("{}", response.answer);
```

A runnable end-to-end example using `MockAdapter` (no API keys needed) lives at
`crates/caw-orchestrator/tests/end_to_end.rs`.

## Benchmarking

Workloads live in `caw-bench`:

- **niah** — synthetic needle-in-a-haystack recall, per-item corpus.
- **opencaw** — Q&A over this repo's own docs, per-item corpus.
- **sysdoc** — Q&A against a pre-built index of a snapshotted documentation
  corpus (e.g. `/usr/share/doc`). The QA file carries only the question,
  reference answer, and expected paths; the corpus lives in a sqlite index
  shared across all items in a run.

### Pre-building an index

`caw-bench-build-index` is a streaming, resumable indexer that runs
ingestion on a dedicated rayon pool while a single GPU consumer embeds
in sub-batches:

```bash
cargo run --release -p caw-bench --bin caw-bench-build-index -- \
  --corpus opencaw-corpora/sysdoc \
  --out opencaw-corpora/sysdoc.sqlite
```

Properties worth knowing:

- **Pipelined**: rayon producers ingest files in parallel and push stubs
  into a bounded channel; the main thread pulls batches, embeds on the
  GPU (candle + CUDA), and inserts. CPU and GPU stay busy concurrently.
- **Dedicated thread pool for ingestion** so the HuggingFace tokenizer's
  own rayon use can't deadlock against the producer (they share the
  global pool otherwise).
- **Length-bucketed sub-batches**: each batch is sorted by text length
  before being split into sub-batches, so padding cost tracks the local
  max rather than the batch-wide max.
- **Resumable**: skips `(path, mtime)` pairs already present in the
  target sqlite. Kill it, restart it, pick up where it left off.
- **WAL + prepared-statement batch inserts**: one transaction per batch,
  prepared statements reused across rows. Collapses what was previously
  N×3 fsyncs into a single commit.
- **Greedy char-based chunking, line-aligned**: a single linear pass over
  content, snapping cuts to the nearest `\n` within a target char budget
  with a hard cap. No per-section tokenization; the BPE tokenizer is
  only run once per final bounded chunk (for the stub's `token_estimate`).

### Running a bench

```bash
cargo run --release -p caw-bench --bin caw-bench -- \
  --workload sysdoc \
  --qa-file opencaw-corpora/sysdoc_qa.json \
  --index opencaw-corpora/sysdoc.sqlite \
  --answer-adapter ollama --answer-model qwen3.5:9b \
  --judge-adapter claude-code --judge-model haiku \
  --out bench.json \
  --trace-out bench.jsonl
```

`--trace-out` writes one JSONL line per (item, mode) with the system
prompt, question, reference answer, loaded fragments (content previews
+ source locators), the model's answer, the judge's rationale, and the
full metric vector. Filter failures with `jq`:

```bash
jq 'select(.result.answer_score < 1)' bench.jsonl
```

## Design Principles

1. **Context is a workspace**: Quality over quantity - manage what's active, not
   just accessible
2. **Recall over retrieval**: Fragments materialize inline with precise
   provenance
3. **Budget awareness**: Explicit token accounting with reserved bands
4. **Provider independence**: Same substrate for cloud APIs and local models
5. **Measurement first**: Recall@k, precision, grounding metrics before scaling

## Roadmap

See `TODO.md` for line-item status and `SCOPE.md` for v1 boundaries.

### Working today

- Core types and trait contracts (`caw-core`)
- Ingestion pipeline (`caw-ingest`): greedy char-based line-snapping chunker
  (linear-time, no per-section tokenization), tree-sitter outlines for code,
  deterministic + LLM summaries, cl100k token estimation
- Hybrid retrieval: semantic (embeddings) + BM25, min-max normalized fusion
- HNSW vector index via `instant-distance`
- SQLite stub store with WAL + batch insert (transaction + prepared-statement
  reuse) and consolidation persistence; Qdrant as a feature-gated alternative
- FastEmbed (BGE), API (OpenAI/Cohere/Voyage shape), Candle with CUDA, ONNX
  embedding providers
- `DynamicRecallOrchestrator`: multi-pass recall with probes, thinking-trace
  extraction, relevance decay, budget-triggered eviction, and consolidation
  notes
- Degradation monitoring with tiered fallback and probe rate limiting
- Curation pipeline: history summarization, tool output compression, system
  prompt budgeting
- Provenance ledger with inline source tagging (XML for Anthropic, bracketed for
  OpenAI-shape)
- Adapters: Anthropic, Groq, Ollama, OpenAI-compatible (covers vLLM / Perplexity
  / HF Inference / llama.cpp-server), ClaudeCode, Mock, LlamaCpp (native FFI,
  feature-gated) with passive mid-stream injection — every N tokens the
  orchestrator embeds the window and injects matching stubs directly into the
  KV cache without restarting generation
- Evaluation primitives: `SessionEvaluator` with recall metrics, false-recall
  heuristic, hysteresis analysis, context efficiency, cooperation metrics
- `caw-bench` with NIAH, opencaw, and sysdoc workloads; `caw-bench-build-index`
  for streaming, resumable, pipelined GPU index construction; per-item JSONL
  trace output for failure analysis
- End-to-end integration test in `crates/caw-orchestrator/tests/end_to_end.rs`

### Open

- Sweep runs of the benchmark harness (caw-bench) across enough seeds and
  workloads to produce threshold-tuning recommendations and
  cooperation-calibration numbers
- Richer consolidation notes as default (LLM-synthesized, not mechanical)
- Postgres + pgvector as an alternative `StubStore`, with a docker-compose
  bring-up for local dev (sqlite remains the default for single-process runs)
- Re-embedding of *chunk content* in addition to stub metadata — today's
  stub-level embeddings can't disambiguate questions whose answer lives
  inside a specific paragraph of an otherwise-generic doc
- Document authority is not encoded in retrieval ranking — a sub-crate README
  that densely uses the project name scores alongside (or above) the root
  README that defines what the project is. Flat semantic similarity has no
  concept of document role or hierarchy. Path depth or explicit authority
  signals would need to be added as a scoring factor.
- Vague-query handling — when the ambiguity gate fires (too many candidates
  above threshold), the current fallback is to load nothing and rely on
  probe-driven recall. A better fallback: run a keyword search (ripgrep) over
  the corpus and inject a compact file-list summary — filenames and match counts,
  no content — so the model can see where the term appears and decide what to
  probe. For thinking models this is especially natural: the model will reason
  about the interesting-looking files in its thinking trace, those file names
  appear at step boundaries, and the existing thinking-trace recall pipeline
  picks them up and loads the content without any additional mechanism.
- Insertion-order experiments (relevance-ranked vs reverse-relevance vs
  stub-order)
- Provenance conflict detection beyond Jaccard term overlap
- Additional prompt transformer surfaces (fenced blocks with `path=`, bare-path
  regex)
- Few-shot token cost surfacing utility

### Deferred (v2+, see SCOPE.md)

- Server API (gRPC/HTTP) — `caw-server` is a scaffold
- Async adapter traits / middleware proxy deployment target
- Engine plugins for mid-stream recall (vLLM, Ollama native)
- Streaming recall interleaved with token generation

## License

MIT
