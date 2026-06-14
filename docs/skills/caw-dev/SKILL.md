---
name: caw-dev
description: This skill should be used when working on the OpenCAW project locally — to "stand up the caw-server proxy", "run the opencaw proxy", "build a caw index", "index the repo for recall", "smoke test the proxy", "test context injection", "run the caw-cli recall loop", "start the recall server", "run the recall benchmark / eval sweep", "measure recall-on vs recall-off", or otherwise build, run, and test OpenCAW. Provides scripts that handle the GPU-pinning and batch-size workarounds and verify context injection.
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
   docs/skills/caw-dev/scripts/build-index.sh crates target/caw-dev/code-index.sqlite
   ```

   Output is an SQLite index under `target/caw-dev/` (gitignored scratch). Re-run after editing files (there is no watcher). The script forces small batches and the freest GPU.

2. **Serve** the proxy. The `corpus-root` MUST equal the corpus passed to `build-index.sh`:

   ```bash
   docs/skills/caw-dev/scripts/serve.sh target/caw-dev/code-index.sqlite crates
   #                                     <index>                         <corpus-root> [upstream] [port] [max-tokens] [retriever]
   ```

   Defaults: upstream `http://localhost:11434/v1` (Ollama), port `8090`, max injected tokens `2000`, retriever `hybrid`. The script backgrounds the server, waits for readiness, and prints the log path. Startup includes a compile + candle init + index load (hybrid also reads every body once to build BM25 posting lists), so allow up to ~90s.

   The retriever defaults to `hybrid` (BM25 lexical fused with cosine) because pure cosine buries definitional chunks on a single-domain corpus — a query paraphrasing a struct's doc comment can rank that struct at the median of the score band. Pass `flat` as the 6th arg for pure cosine, or `hnsw` for the ANN index. Every request logs the full ranked candidate list with scores at DEBUG (`candidate #N score=… path`), so you can see whether a relevant stub was ranked out vs. clamped out by the token budget.

3. **Smoke test** that injection actually happens (not just that the model answered):

   ```bash
   docs/skills/caw-dev/scripts/smoke.sh
   #                                     [port] [model] [question]
   ```

   With no `question` arg it runs the whole shared query pool (`scripts/queries.txt`), one request per question, and prints each answer with its `augmented with N fragments (T tokens)` line — so one run exercises retrieval across several subsystems instead of the same chunk every time. Pass a `question` to run just that one. A small model naming a project-specific symbol/path it could not otherwise know confirms the full path works.

4. **Inspect the retrieval itself** — see the ranked candidates, not just the model answer:

   ```bash
   docs/skills/caw-dev/scripts/retrieve.sh "which struct owns the multi-pass recall loop?"
   #                                        <query> [port] [--full]
   ```

   Hits the proxy's read-only `/v1/retrieve` route (same retrieval as the chat path, no model call) and prints the full ranked list: fused score, path, token cost, and disposition for each candidate — `admitted` (injected), `clamped` (read but over budget, where the token clamp stopped), `budget_full` (ranked below the clamp, never read), or `content_miss` (unreadable body, usually a corpus-root mismatch). This is how you tell whether a relevant chunk was **ranked out** of the pool or **clamped out** by `--max-tokens` — a distinction smoke.sh can't show. Pass `--full` to also dump the body of each admitted fragment (the exact text that would be injected).

5. **Prove it adds value** (optional) — proxy vs. straight to the upstream:

   ```bash
   docs/skills/caw-dev/scripts/ab-test.sh   # [port] [upstream] [model] [question]
   ```

   Same default: with no `question` it runs an A/B pair for every question in `scripts/queries.txt`; pass one to A/B just that question. Prints both answers and the injection-proof log line. The direct answer can't name project-specific symbols; the proxied one can.

6. **Stop** when done:

   ```bash
   docs/skills/caw-dev/scripts/stop.sh   # [port], default 8090
   ```

To use the proxy from a real tool: set the client's OpenAI base URL to `http://localhost:8090/v1` and use any model name Ollama has (`ollama list`).

## Workflow: run the recall engine (CLI)

For non-interactive testing with a sane local-only config (no Anthropic key, GPU pinned, index + sessions kept under `target/caw-dev/`), use `test-cli.sh`:

```bash
scripts/test-cli.sh -q "which struct owns the multi-pass recall loop?"
printf 'what is opencaw?\nhow does eviction work?\n' | docs/skills/caw-dev/scripts/test-cli.sh
aw-dev/scripts/test-cli.sh --intent none --model llama3.2:3b -q "..."
```

For a raw, fully-manual invocation, `run-cli.sh` forwards all arguments straight to `cargo run -p caw-cli` (GPU pinned):

```bash
scripts/run-cli.sh --show-intent
scripts/run-cli.sh --adapter ollama --model llama3.2:3b
```

Note: `caw-cli`'s default index/session dir is a per-corpus location under `~/.cache/caw/` — a bare run no longer drops `.caw/` into the working directory. `test-cli.sh` pins both under `target/caw-dev/`.

### Exercise the differentiating engine (eviction + consolidation)

`test-cli.sh` disables the LLM consolidation/summarization path (`--no-llm-consolidation`). To actually test the core novelty — workspace fills past budget, fragments are **evicted**, and each eviction synthesizes an LLM **consolidation note** persisted to the stub store for later recall — use `consolidation-cli.sh`:

```bash
scripts/consolidation-cli.sh
scripts/consolidation-cli.sh --max-tokens 800 -q "how does recall work?" -q "what gets evicted?"
scripts/consolidation-cli.sh --aux-adapter claude-code-haiku   # aux via the claude CLI instead
```

It forces eviction with a small `--max-tokens` budget. The aux model (consolidation/summarization/curation) defaults to local Ollama (`--aux-adapter ollama --aux-model llama3.2:3b`), so the whole engine runs locally with no API key. `--aux-adapter` takes the same selectors as `--adapter`: pass `--aux-adapter claude-code-haiku` to route aux through the installed `claude` CLI instead (still no API key, ~$0.01 per eviction). After the run it prints a proof summary computed from the verbose log: eviction count, consolidation notes persisted, and — only for claude-code aux — the claude-code call count and cost.

**Aux must be a non-thinking model.** A thinking aux (`qwen3.x`, `deepseek-r1`, …) routinely returns only a reasoning trace for the consolidation prompt; after `split_thinking` the body is blank, the adapter logs `degenerate output: blank answer`, and eviction falls back to a deterministic templated note instead of real LLM synthesis — silently skipping the path this script exists to demonstrate. It is also ~90–115s per note vs. roughly real-time, enough that the default multi-turn run will not finish inside a normal timeout. `llama3.2:3b` is the validated default.

### Dump exactly what the model sees

To verify the workspace contents, add `--show-prompt` (prints to stderr) and/or `--save-prompt` (one `prompt-{timestamp}-turn-{N}.txt` per model call, beside the session files) to any `run-cli.sh`/`test-cli.sh`/`consolidation-cli.sh` invocation. These work with any adapter and dump the **exact context string** the model receives: the system message with the recalled workspace rendered in the adapter's own provenance format (`Bracketed` for ollama/groq/vllm/llama, `Xml` for anthropic/claude-code), followed by the user message. The recall loop makes several model calls per turn, so each turn produces several dumps — you can watch the workspace grow across the multi-pass loop.

```bash
scripts/run-cli.sh --dir crates --adapter ollama --model llama3.2:3b --max-tokens 1200 --show-prompt
```

## Gotchas (read before improvising)

- **GPU OOM is a device-selection problem, not a memory-shortage problem.** On an OOM the candle embedder dies rather than shrinking — it only falls back to CPU when CUDA is *absent*. Pick a free GPU explicitly. The scripts do this via `pick-gpu.sh` + `CUDA_VISIBLE_DEVICES`; the embedder also reads `CAW_EMBED_DEVICE` (`cpu` | `cuda` | `cuda:N`) as declared config, and an explicit `cuda:N` that can't be opened errors loudly instead of dying later. The two don't compose: the scripts already mask with `CUDA_VISIBLE_DEVICES`, so *inside* a script the only visible device is ordinal 0 — passing `CAW_EMBED_DEVICE=cuda:1` there asks for an ordinal that doesn't exist in the masked view and (correctly) errors. Use one mechanism or the other. Don't run the raw `cargo` commands without selecting a device when a large model occupies GPU 0.
- **Batch size matters for indexing.** BGE attention is `batch x seq^2`; the bench default sub-batch of 256 OOMs on long chunks. `build-index.sh` uses small batches deliberately.
- **Corpus root must match.** Stub paths are stored relative to `--corpus`; `serve.sh`'s `corpus-root` must be the same directory or every content fetch fails silently and the proxy forwards unaugmented.
- **Never index the repo root.** The skip rules drop `target/`/hidden/`scripts/` but not `data/` or `opencaw-corpora/` (multi-GB system docs). Index `crates/` or `docs/`.
- **The proxy is single-shot RAG, not the recall engine**, and it injects unconditionally with no intent gating. The differentiating engine is in the CLI only.

For the full reasoning behind each gotcha, file/line references, the host's GPU layout, and the proxy-vs-engine boundary, read `references/internals.md`.

## Additional resources

- **`references/internals.md`** — detailed internals: embedder device handling, corpus/path-resolution contract, indexer skip rules, what the proxy is and isn't, how to verify injection, upstream model notes, runtime-state layout.
- **`scripts/pick-gpu.sh`** — prints the freest CUDA ordinal (empty = CPU); used by the other scripts.
- **`scripts/build-index.sh`** — build an index with safe batch sizes and GPU pinning.
- **`scripts/serve.sh`** — start the proxy (backgrounded, debug logging, GPU pinned).
- **`scripts/queries.txt`** — shared default query pool (one per line) used by `smoke.sh`, `ab-test.sh`, and `test-cli.sh` when no query is passed. Add lines here to broaden coverage for all three.
- **`scripts/smoke.sh`** — send test requests (whole pool by default) and verify injection.
- **`scripts/retrieve.sh`** — inspect retrieval directly via `/v1/retrieve`: ranked candidates with scores, token cost, and admitted/clamped/budget_full/content_miss disposition (no model call). `--full` dumps admitted bodies.
- **`scripts/ab-test.sh`** — proxy vs. direct-upstream A/B (whole pool by default) to prove context changes the answer.
- **`scripts/stop.sh`** — stop a running proxy.
- **`scripts/bench.sh`** — run the `caw-bench` end-to-end recall harness (recall-on vs recall-off, judge-scored) with the canonical sysdoc index + QA file, GPU pinning, and release build. `bench.sh [sysdoc|opencaw|niah] -- <caw-bench args>` forwards everything after `--` (e.g. `--num-predict`, `--judge-adapter groq`, `--limit`, `--only-mode`, `--out`, `--trace-out`). Don't reconstruct the `caw-bench` invocation by hand.
- **`scripts/test-cli.sh`** — drive the caw-cli recall engine non-interactively, local-only config (LLM consolidation OFF).
- **`scripts/consolidation-cli.sh`** — exercise the full engine (eviction + LLM consolidation) and print a computed proof summary; aux model via the local `claude` CLI.
- **`scripts/run-cli.sh`** — raw pass-through to caw-cli, args forwarded.
