# Model Adapters

All adapters implement the `ModelAdapter` trait from `caw-core`. The pattern is synchronous (using `block_on` internally), which works for library use but will panic if called from within an existing async context. Async refactoring is deferred to v2 per [scope.md](scope.md).

## Anthropic

Supports Claude Sonnet 4+ and Claude Opus 4+. Extended thinking (hidden reasoning) is available but `supports_visible_reasoning` is currently hardcoded to `false`, so the thinking-trace recall path uses explicit `<probe>` markers instead.

```rust
use caw_adapters::AnthropicAdapter;

let adapter = AnthropicAdapter::claude_sonnet();
// or
let adapter = AnthropicAdapter::new("api_key", "claude-sonnet-4-20250514");
```

Requires `ANTHROPIC_API_KEY` in the environment (or pass the key directly).

## Groq

```rust
use caw_adapters::GroqAdapter;

let adapter = GroqAdapter::llama_70b();
// or
let adapter = GroqAdapter::new("api_key", "llama-3.3-70b-versatile");
```

Supported models: Llama 3.3 70B Versatile, Llama 3.1 8B Instant, Mixtral 8x7B.

## Ollama (local)

```rust
use caw_adapters::OllamaAdapter;

let adapter = OllamaAdapter::deepseek_r1();
// or
let adapter = OllamaAdapter::local("model_name");
// or with custom endpoint
let adapter = OllamaAdapter::new("http://custom:11434", "model_name");
```

Tested models: DeepSeek R1 (with visible reasoning), Qwen 3.5, Llama 3.2. `capabilities()` currently returns `supports_hidden_reasoning: true` for all Ollama models — this will be fixed by querying `/api/show` at construction time.

## OpenAI-compatible

Generic adapter for any provider speaking the OpenAI chat completions protocol: vLLM, Perplexity, HuggingFace Inference Endpoints, llama.cpp-server, Ollama API endpoint, etc.

```rust
use caw_adapters::OpenAiCompatibleAdapter;
```

Configurable headers and per-deployment capability flags.

## ClaudeCode

Runs recall against the local `claude` CLI binary. Useful for using Claude as a judge in benchmarks without sharing weights with the answer model.

```rust
use caw_adapters::ClaudeCodeAdapter;
```

## LlamaCpp (feature-gated)

Native llama.cpp inference via FFI — no HTTP server, no restart overhead. Owns the sampling loop and implements `generate_passive` for mid-stream recall injection every N tokens directly into the KV cache.

Enable with `--features llama`. Discovers the library via pkg-config; set `LLAMA_PATH` to a custom build directory if pkg-config can't find it.

```rust
use caw_adapters::{LlamaCppAdapter, LlamaCppConfig};

let adapter = LlamaCppAdapter::from_path("model.gguf")?;
// or with explicit config
let adapter = LlamaCppAdapter::new_with(LlamaCppConfig {
    model_path: "model.gguf".into(),
    n_gpu_layers: -1,   // all layers on GPU
    n_ctx: 8192,
    temperature: 0.7,
    ..Default::default()
})?;
```

```bash
cargo run -p caw-cli --features llama -- \
  --adapter llama --model /path/to/model.gguf
```

## MockAdapter

Echoes `format_workspace(...)` back as the answer. Use in tests and examples — no network, no API keys required.
