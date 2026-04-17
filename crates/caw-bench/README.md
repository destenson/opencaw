# caw-bench

Benchmark harness for the OpenCAW recall loop. Produces the "recall on vs.
recall off, same model + same context budget" numbers the design doc's
central thesis stands or falls on.

## What it measures

Per-item:

- **answer_score** (0.0-1.0) — did the model answer correctly? For NIAH this
  is a deterministic substring check; for opencaw Q&A it's a cheap judge
  model scoring against a reference answer.
- **recall@k / precision@k** — of the paths that *should* have been recalled
  for this question, how many ended up in the workspace?
- **context_efficiency** — share of the model's actual context window
  (recalled content + system + query + provenance tags) that is recalled
  content. Stubs aren't counted; they live in the index, not the prompt.
- **false_recall_rate** — share of loaded fragments whose stub summary
  shares very few non-stopword terms with the recalled content (heuristic
  from `caw-eval`'s `FalseRecallMetrics`).
- **latency_ms** — wall clock per item.

Aggregates compute mean values per mode plus the on-vs-off deltas.

## Modes

- `recall_on` — full `DynamicRecallOrchestrator`: initial retrieval,
  multi-pass iteration, probes, thinking-trace recall, eviction,
  consolidation.
- `recall_off` — single-shot retrieval only (`max_recall_iterations=0`,
  probes off, traces off). Same model, same budget, same corpus. Isolates
  the contribution of the dynamic recall machinery.

## Backends

Three adapters are wired into the CLI; pick per call via `--answer-adapter`
and `--judge-adapter`:

- `ollama` — local Ollama at `--ollama-url` (default `http://localhost:11434`).
  Default for the answer model. Supply the tag via `--answer-model`
  (e.g. `qwen3.5:9b`, `llama3.2`, `deepseek-v3.1:671b-cloud`).
- `vllm` — vLLM or any OpenAI chat-completions-compatible server at
  `--openai-url`. Reads `OPENAI_COMPATIBLE_API_KEY` from env if set.
- `claude-code` — local `claude` CLI. Default for the judge (cheap haiku,
  doesn't share weights with the answer model so judge bias is reduced).

## Workloads

### `niah` (needle-in-a-haystack)

Procedurally generates bureaucratic project-status filler from a seeded
grammar, inserts a short memo containing a fabricated fact, and asks the
model to retrieve that fact. The filler is out-of-distribution by
construction (fabricated project names, operator handles, regional codes),
so the model can't fall back on training knowledge.

Defaults are sized so the corpus is several times larger than the workspace
budget — forcing the orchestrator's eviction/consolidation machinery to
actually engage when probes bring in additional fragments. Tune with
`--niah-items N --niah-filler P --niah-seed S` and
`--max-workspace-tokens T`.

### `opencaw` (opencaw-on-opencaw)

Hand-authored Q&A drawn from this repository's own code, README, SCOPE,
TODO, and design doc. The corpus is whatever `.rs / .md / .toml` files
`--repo-root` points at (default `.`). Scoring uses the judge adapter to
grade answers against reference strings.

Questions live in `src/qa/opencaw_qa.json` and are embedded into the binary
via `include_str!`. The current set is mostly single-fact lookups — useful
for retrieval@k but doesn't stress multi-pass refinement. Synthesis-style
questions are a separate exercise.

## Usage

```bash
# Smoke test (3 NIAH items, both modes, default Ollama qwen3.5:9b)
cargo run -p caw-bench --release -- --workload niah --niah-items 3 --limit 3

# Full NIAH sweep against a different local model
cargo run -p caw-bench --release -- \
  --workload niah \
  --niah-items 10 \
  --answer-model "deepseek-v3.1:671b-cloud" \
  --out niah_report.json

# vLLM backend
cargo run -p caw-bench --release -- \
  --workload niah \
  --answer-adapter vllm \
  --openai-url http://localhost:8000 \
  --answer-model Qwen/Qwen2.5-7B-Instruct

# opencaw Q&A using claude-code for both answer and judge
cargo run -p caw-bench --release -- \
  --workload opencaw \
  --repo-root . \
  --answer-adapter claude-code --answer-model sonnet \
  --judge-adapter claude-code --judge-model haiku \
  --out opencaw_report.json
```

## Caveats

- Each item runs the orchestrator twice (on + off), and a fresh adapter +
  retriever is built per item to keep runs independent.
- `false_recall_rate` is a heuristic (stub-summary-to-content term
  overlap), not a ground-truth correctness measure. Compare on-vs-off for
  the same workload, not against an absolute pass/fail bar.
- Whether `recall_on` actually diverges from `recall_off` depends on
  whether your model emits `<probe>...</probe>` markers or `<think>` blocks
  the orchestrator can act on. Modern instruct models (qwen3.5,
  llama3.2-instruct, deepseek-r1) get marker instructions injected via
  the system prompt; whether they comply is what the bench measures.
- The 30 hand-authored opencaw Q&A pairs are mostly single-fact lookups.
  They exercise retrieval@k but rarely force multi-pass refinement —
  useful as a baseline, not as a stress test.
