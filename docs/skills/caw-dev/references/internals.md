# caw-dev internals and gotchas

Non-obvious facts about running and testing OpenCAW locally. Each item below cost real debugging time at least once; consult before improvising.

## GPU / embedder

- `CandleEmbeddingProvider::from_pretrained` (`crates/caw-index/src/embeddings/candle_provider.rs`) hardcodes `Device::new_cuda(0)`. Its CPU fallback fires **only when CUDA is absent**, never on an out-of-memory error during the forward pass. So a busy GPU 0 produces a `CUDA_ERROR_OUT_OF_MEMORY` mid-batch, not a graceful CPU fall-back.
- Workaround used by every script here: `pick-gpu.sh` selects the GPU with the most free VRAM and exports it as `CUDA_VISIBLE_DEVICES`, so the hardcoded `cuda:0` resolves to a GPU with headroom. On this host GPU 0 is the RTX 3090 (usually full of an Ollama model) and GPU 1 is the RTX 5070 Ti.
- BGE attention memory scales as `batch x seq^2`. The bench default `--sub-batch-size 256` with long chunks (seq up to ~800) allocates multi-GB attention tensors and OOMs even with 14 GB free. For the small source corpus, `--batch-size 32 --sub-batch-size 8` is plenty and never OOMs. Throughput is irrelevant here — the corpus is ~1 MB.
- Possible permanent fix (the user asked for it): make `from_pretrained` accept a device index and/or catch a forward-pass OOM and retry on CPU. Until then, the env-var pin is the contract.
- `caw-bench-build-index` does **not** take a `--features` flag; `caw-bench` already enables the `candle`/`onnx` features on `caw-index`. Passing `--features candle` to `cargo run -p caw-bench` errors.

## Corpus / index / path resolution

- Stub paths are stored **relative to `--corpus`** at build time. `caw-server` materializes content via `corpus_root.join(stub.path)`, so `serve.sh`'s `corpus-root` MUST equal the `build-index.sh` `corpus-dir`. Mismatch = every `get_content` fails and the proxy silently forwards unaugmented.
- A single index has a single corpus root. To cover both code and docs with correct resolution, either index a common ancestor (but see the skip-rule caveat) or run two servers.
- `should_skip` in `build_index.rs` skips hidden dirs, `scripts/`, `target/`, `node_modules/`, and binary/archive extensions. It does **not** skip `data/` (~1.6 GB) or `opencaw-corpora/` (~0.5 GB) of system docs. Never point the corpus at the repo root — index `crates/` (code) or `docs/` (prose) instead.
- The build is incremental and resumable (`INSERT OR REPLACE` keyed on path+mtime). Re-running without `--rebuild` only ingests changed/new files. The scripts pass `--rebuild` for a clean snapshot; drop it for incremental updates. There is no file-watcher — the index is stale until rebuilt.

## What the proxy is and isn't

- `caw-server` is single-shot RAG by design. Its own `lib.rs` header: "no orchestrator, no probes, no multi-pass — just retrieve -> inject -> forward." It embeds the last user message, takes top-k by cosine, clamps to `--max-workspace-tokens`, and splices bracketed provenance-tagged fragments onto the end of that message.
- It injects **unconditionally** when retrieval returns anything — no intent gating, no relevance floor beyond top-k + the token cap. Even low-information turns get context stapled on.
- It only augments string `content` on the last `role:"user"` message. Vision/multipart content is passed through untouched.
- The thinking-trace recall engine (multi-pass, probe extraction, relevance decay, budget eviction, consolidation) lives in `DynamicRecallOrchestrator` and is exercised only by `caw-cli` (`run-cli.sh`), not the proxy. Routing the proxy through `DynamicRecallOrchestrator::run_turn` is the open path to making the drop-in proxy run the real engine.

## Verifying injection (don't trust the answer alone)

- With `RUST_LOG=caw_server=debug` (set by `serve.sh`), each augmented request logs `augmented with N fragments (T tokens)`. Absence of that line means retrieval returned nothing or there was no user message — the proxy then forwards unmodified and the model answers from its own weights.
- A strong functional signal: a small model (e.g. `llama3.2:3b`) correctly naming a project-specific symbol or file path it could not know from pretraining. That only happens when injection worked.

## Upstream models (Ollama at :11434)

- Ollama exposes the OpenAI protocol at `http://localhost:11434/v1`; the proxy appends `/chat/completions`.
- For tests prefer a small, fast model (`llama3.2:3b`, `granite4:micro`, `qwen3.5:9b`). Avoid the large local models for smoke tests — they monopolize the 3090 and are slow. Model names are passed straight through to the upstream, so they must be names Ollama actually has (`ollama list`).

## Runtime state

- `serve.sh` writes pid + log to `$ROOT/target/caw-dev/server-<port>.{pid,log}` — under the existing Rust build dir, so it is already gitignored and removed by `cargo clean`, and nothing new lands in the repo. `stop.sh` reads the pidfile (and falls back to `pkill` on the bound port).
