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
# Smoke test (3 NIAH items, both modes, default Ollama huihui_ai/phi4-reasoning-abliterated:3.8b)
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

## Sweeps

`caw-bench-sweep` drives `caw-bench` across a grid of parameters defined in a
TOML config. It writes one report per cell and resumes where it left off if
interrupted or crashed.

```bash
# Smoke test the sweep wiring (2 cells, 2 items each)
cargo build -p caw-bench --release
./target/release/caw-bench-sweep --config crates/caw-bench/sweeps/minimal.toml

# The full roadmap grid (48 cells: 3 models × 4 workload/seed × 2 thresholds × 2 budgets)
./target/release/caw-bench-sweep --config crates/caw-bench/sweeps/default.toml

# See what would run without executing
./target/release/caw-bench-sweep --config … --dry-run

# Force a re-run of every cell (ignore cached reports)
./target/release/caw-bench-sweep --config … --force
```

Output layout (`out_dir` from the config, relative to the config file):

```
bench-results/default/
├── manifest.jsonl         # one line per completed cell {hash, params}
├── 2430eff2b7f9625b/      # cell hash = sha256 of sorted params, truncated
│   ├── params.json        # the exact param set this cell ran
│   └── report.json        # full caw-bench BenchReport for this cell
└── …
```

**Failure model.** A non-zero exit from any cell halts the sweep with the
offending command printed verbatim. Fix the root cause (missing model,
ollama down, timeout too tight) and re-run the same `caw-bench-sweep`
command — cells whose `report.json` already exists are skipped. Cell hashes
cover the full merged parameter set, so editing `fixed` values or an axis
entry invalidates exactly the cells that were affected; unchanged cells
still skip.

Config shape: `fixed` holds flags applied to every cell; `axes.<name>` is
a list of tables where each table is one choice along that axis. The
Cartesian product of all axes is the cell list. Any `caw-bench` flag is
valid in either section — keys convert `snake_case → kebab-case` when
passed to the subprocess.

## Intent Bench

`caw-bench-intent` evaluates small models as strict JSON query-intent
classifiers. This is useful for deciding which cheap model is good enough
to route inventory vs. results vs. comparison vs. explanation queries
before the main answer model runs.

```bash
# Compare a pool of local Ollama models on the built-in intent set
cargo run -p caw-bench --bin caw-bench-intent --release -- \
  --candidate qwen2:0.5b \
  --candidate tinyllama:1.1b \
  --candidate llama3.2:3b \
  --out intent-report.json

# Or discover local small Ollama models automatically and benchmark them
./scripts/intent-bench-ollama.sh
```

The JSON report includes overall exact-match rate, per-field accuracy, and
per-tag exact-match rates so you can spot models that are strong on one
classification family (for example inventory or comparison) even if they
aren't the best overall.

`scripts/intent-bench-ollama.sh` discovers local models from `ollama list`,
keeps only local models at or below `MAX_GB` (default `6`), deduplicates
aliases that share the same model ID, and passes the resulting set to
`caw-bench-intent` as repeated `--candidate` flags. Useful knobs:

- `MAX_GB=3 ./scripts/intent-bench-ollama.sh` to only test very small models.
- `LIMIT=8 ./scripts/intent-bench-ollama.sh` to cap how many discovered models run.
- `NAME_REGEX='^(qwen2|tinyllama|llama3\\.2|granite4)' ./scripts/intent-bench-ollama.sh` to narrow by model name.
- `./scripts/intent-bench-ollama.sh qwen2:0.5b tinyllama:1.1b llama3.2:3b` to bypass discovery and run an explicit set.

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
