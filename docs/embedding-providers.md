# Embedding Providers

*Status: Living reference — kept current. See [docs/README.md](README.md) for the docs index.*

All providers implement `EmbeddingProvider` from `caw-core`, which exposes `embed_query()` and `embed_document()` for asymmetric encoding. BGE-family models add `"query: "` / `"passage: "` prefixes automatically.

## FastEmbed (default)

Rust-native with bundled quantized models. No external dependencies — the recommended starting point.

```rust
use caw_index::FastEmbedProvider;

let embedder = FastEmbedProvider::bge_small()?;  // BGE-small-en-v1.5, 384d
let embedder = FastEmbedProvider::bge_base()?;   // BGE-base-en-v1.5, 768d
```

## API-based

For OpenAI, Cohere, and Voyage shapes. Functional for OpenAI; Cohere and Voyage are stubbed.

```rust
use caw_index::ApiEmbeddingProvider;

let embedder = ApiEmbeddingProvider::openai_small()?;
```

## Candle (feature-gated)

BERT-family models downloaded from HuggingFace Hub (e.g. `BAAI/bge-small-en-v1.5`). Automatically uses CUDA when available, falls back to CPU with a log line. Enable with `--features candle`.

Used by `caw-bench-build-index` for bulk embedding on the GPU. Mutually exclusive with `fastembed` because both link `onnxruntime` via different crate versions.

Limits: 512-token max position (BGE's limit), 32 KB character clip on input to prevent BPE tokenizer stalls.

## ONNX (feature-gated)

Loads arbitrary ONNX models from a local path with an adjacent `tokenizer.json`. Enable with `--features onnx`. Mutually exclusive with `fastembed`.

```rust
// load from local path; tokenizer.json must be adjacent
let embedder = OnnxEmbeddingProvider::from_path("path/to/model.onnx")?;
```
