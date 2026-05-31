---
name: caw-dev
description: This skill should be used when working on the OpenCAW project locally — to "stand up the caw-server proxy", "run the opencaw proxy", "build a caw index", "index the repo for recall", "smoke test the proxy", "test context injection", "run the caw-cli recall loop", "start the recall server", or otherwise build, run, and test OpenCAW. Provides scripts that handle the GPU-pinning and batch-size workarounds and verify context injection.
version: 0.1.0
---

# caw-dev

Build, run, and test the OpenCAW retrieval stack locally without rewriting the same bespoke commands and rediscovering the same GPU/batch pitfalls each time. Covers two surfaces: the `caw-server` drop-in OpenAI proxy (single-shot retrieve-inject-forward) and the `caw-cli` recall loop (the full thinking-trace engine).

All scripts live in `scripts/` and are self-locating (they find the repo root via git), so invoke them by absolute or repo-relative path from anywhere. They pin the embedder to the freest GPU automatically — do not prefix them with a manual `CUDA_VISIBLE_DEVICES`.

## When to use which surface

- **Proxy (`caw-server`)** — point any OpenAI-compatible client (aider, Continue, an SDK with `base_url`) at it and repo context is injected automatically. This is plain RAG: top-k retrieve, inject, forward. Good for "use my codebase as context in a normal chat client."
- **CLI (`caw-cli`)** — the real engine: multi-pass recall, probes, relevance decay, eviction, consolidation. Use this when developing or testing the thinking-trace recall path itself, not the proxy.

## Workflow: stand up the proxy

1. **Build an index** over a corpus (pick `crates/` for code or `docs/` for prose — never the repo root; see Gotchas):

   ```bash
   docs/skills/caw-dev/scripts/build-index.sh crates bench-results/opencaw-code-index.sqlite
   ```

   Output is an SQLite index. Re-run after editing files (there is no watcher). The script forces small batches and the freest GPU.

2. **Serve** the proxy. The `corpus-root` MUST equal the corpus passed to `build-index.sh`:

   ```bash
   docs/skills/caw-dev/scripts/serve.sh bench-results/opencaw-code-index.sqlite crates
   #                                     <index>                                <corpus-root> [upstream] [port] [max-tokens]
   ```

   Defaults: upstream `http://localhost:11434/v1` (Ollama), port `8080`, max injected tokens `2000`. The script backgrounds the server, waits for readiness, and prints the log path. Startup includes a compile + candle init + index load, so allow up to ~90s.

3. **Smoke test** that injection actually happens (not just that the model answered):

   ```bash
   docs/skills/caw-dev/scripts/smoke.sh
   #                                     [port] [model] [question]
   ```

   It sends one chat completion and prints both the model's answer and the server-side `augmented with N fragments (T tokens)` line. A small model naming a project-specific symbol/path it could not otherwise know confirms the full path works.

4. **Prove it adds value** (optional) — same question to the proxy vs. straight to the upstream:

   ```bash
   docs/skills/caw-dev/scripts/ab-test.sh   # [port] [upstream] [model] [question]
   ```

   Prints both answers and the injection-proof log line. The direct answer can't name project-specific symbols; the proxied one can.

5. **Stop** when done:

   ```bash
   docs/skills/caw-dev/scripts/stop.sh   # [port], default 8080
   ```

To use the proxy from a real tool: set the client's OpenAI base URL to `http://localhost:8080/v1` and use any model name Ollama has (`ollama list`).

## Workflow: run the recall engine (CLI)

For non-interactive testing with a sane local-only config (no Anthropic key, GPU pinned, index + sessions kept under `target/caw-dev/`), use `test-cli.sh`:

```bash
docs/skills/caw-dev/scripts/test-cli.sh -q "which struct owns the multi-pass recall loop?"
printf 'what is opencaw?\nhow does eviction work?\n' | docs/skills/caw-dev/scripts/test-cli.sh
docs/skills/caw-dev/scripts/test-cli.sh --intent none --model qwen3.5:9b -q "..."
```

For a raw, fully-manual invocation, `run-cli.sh` forwards all arguments straight to `cargo run -p caw-cli` (GPU pinned):

```bash
docs/skills/caw-dev/scripts/run-cli.sh --show-intent
docs/skills/caw-dev/scripts/run-cli.sh --adapter ollama --model qwen3.5:9b
```

Note: `caw-cli`'s default index/session dir is a per-corpus location under `~/.cache/caw/` — a bare run no longer drops `.caw/` into the working directory. `test-cli.sh` pins both under `target/caw-dev/`.

### Exercise the differentiating engine (eviction + consolidation)

`test-cli.sh` disables the LLM consolidation/summarization path (`--no-llm-consolidation`) so it can run without an Anthropic key. To actually test the core novelty — workspace fills past budget, fragments are **evicted**, and each eviction synthesizes an LLM **consolidation note** persisted to the stub store for later recall — use `consolidation-cli.sh`:

```bash
docs/skills/caw-dev/scripts/consolidation-cli.sh
docs/skills/caw-dev/scripts/consolidation-cli.sh --max-tokens 800 -q "how does recall work?" -q "what gets evicted?"
```

It forces eviction with a small `--max-tokens` budget and routes the aux model through the local `claude` CLI (`ClaudeCodeAdapter`, no API key needed — but each eviction spawns one `claude` call costing real tokens, ~$0.01 each with haiku). After the run it prints a proof summary computed from the verbose log: eviction count, consolidation notes persisted, aux LLM calls, and total aux cost. Only `haiku`|`sonnet` are meaningful for `--aux-model` — `build_aux_adapter` (caw-cli `main.rs`) ignores everything else and cannot currently route aux tasks to Ollama.

## Gotchas (read before improvising)

- **GPU OOM is a device-pinning problem, not a memory-shortage problem.** The candle embedder hardcodes `cuda:0` and only falls back to CPU when CUDA is *absent* — never on an OOM. The scripts pin the freest GPU to dodge this. Do not run the raw `cargo` commands without that pin when a large model occupies GPU 0.
- **Batch size matters for indexing.** BGE attention is `batch x seq^2`; the bench default sub-batch of 256 OOMs on long chunks. `build-index.sh` uses small batches deliberately.
- **Corpus root must match.** Stub paths are stored relative to `--corpus`; `serve.sh`'s `corpus-root` must be the same directory or every content fetch fails silently and the proxy forwards unaugmented.
- **Never index the repo root.** The skip rules drop `target/`/hidden/`scripts/` but not `data/` or `opencaw-corpora/` (multi-GB system docs). Index `crates/` or `docs/`.
- **Ignore `bench-results/server-smoke-index.sqlite`** — stale garbage that indexed build artifacts. Build fresh.
- **The proxy is single-shot RAG, not the recall engine**, and it injects unconditionally with no intent gating. The differentiating engine is in the CLI only.

For the full reasoning behind each gotcha, file/line references, the host's GPU layout, and the proxy-vs-engine boundary, read `references/internals.md`.

## Additional resources

- **`references/internals.md`** — detailed internals: embedder device handling, corpus/path-resolution contract, indexer skip rules, what the proxy is and isn't, how to verify injection, upstream model notes, runtime-state layout.
- **`scripts/pick-gpu.sh`** — prints the freest CUDA ordinal (empty = CPU); used by the other scripts.
- **`scripts/build-index.sh`** — build an index with safe batch sizes and GPU pinning.
- **`scripts/serve.sh`** — start the proxy (backgrounded, debug logging, GPU pinned).
- **`scripts/smoke.sh`** — send a test request and verify injection.
- **`scripts/ab-test.sh`** — proxy vs. direct-upstream A/B to prove context changes the answer.
- **`scripts/stop.sh`** — stop a running proxy.
- **`scripts/test-cli.sh`** — drive the caw-cli recall engine non-interactively, local-only config (LLM consolidation OFF).
- **`scripts/consolidation-cli.sh`** — exercise the full engine (eviction + LLM consolidation) and print a computed proof summary; aux model via the local `claude` CLI.
- **`scripts/run-cli.sh`** — raw pass-through to caw-cli, args forwarded.
