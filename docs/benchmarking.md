# Benchmarking

The benchmark harness (`caw-bench`) measures recall-on vs. recall-off at matched context budget — this is the central empirical claim of the project. All binaries live in the `caw-bench` crate.

## Workloads

### `niah` (needle-in-a-haystack)

Procedurally generates bureaucratic project-status filler from a seeded grammar, inserts a short memo with a fabricated fact, and asks the model to retrieve that fact. Filler is out-of-distribution by construction, so the model can't fall back on training knowledge.

The corpus is sized to be several times larger than the workspace budget, forcing eviction and consolidation to engage. Tune with `--niah-items N --niah-filler P --niah-seed S --max-workspace-tokens T`.

### `opencaw` (opencaw-on-opencaw)

Hand-authored Q&A drawn from this repo's own code, README, scope, todo, and design doc. The corpus is whatever `.rs / .md / .toml` files `--repo-root` points at. Scoring uses a judge model to grade answers against reference strings.

The current set is mostly single-fact lookups — useful for retrieval@k but doesn't stress multi-pass refinement.

### `code-agent`

The questions a coding agent asks itself mid-task while working in a repository: struct fields, method signatures, trait bounds, call sites — the lookups it would otherwise satisfy by grepping and reading files. Same repo corpus as `opencaw` (whatever `.rs / .md / .toml` files `--repo-root` points at), but framed as agent info-needs rather than project facts.

Per-item scoring is chosen in the QA JSON: exact code facts (a field name, a default value, a crate) carry a `needle` and are scored with a deterministic case-insensitive substring match, avoiding judge-model noise; synthesis questions (call-site sets, signatures the model paraphrases) carry a `reference_answer` and fall back to judge scoring. Call-site questions must list *every* call site in `expected_paths` or the recall@k ground truth is wrong. No prebuilt index is needed — the repo is ingested in-memory once and shared across items.

### `sysdoc`

Q&A against a pre-built index of a snapshotted documentation corpus (e.g. `/usr/share/doc`). The QA file carries only the question, reference answer, and expected paths; the corpus lives in a sqlite index shared across all items in a run.

## Modes

Both workloads run each item in two modes:

- **`recall_on`** — full `DynamicRecallOrchestrator`: initial retrieval, multi-pass iteration, probes, thinking-trace recall, eviction, consolidation
- **`recall_off`** — single-shot retrieval only (`max_recall_iterations=0`, probes off, traces off). Same model, same budget, same corpus. Isolates the contribution of the dynamic recall machinery.

## Metrics

Per-item:
- **answer_score** (0.0–1.0) — correctness. NIAH: deterministic substring check. opencaw/sysdoc: judge model grading against a reference answer. code-agent: per-item — deterministic substring check for `needle` items, judge grading for `reference_answer` items.
- **recall@k / precision@k** — of the paths that *should* have been recalled, how many ended up in the workspace?
- **context_efficiency** — share of actual context (recalled content + system + query + provenance tags) that is recalled content. Stubs aren't counted.
- **false_recall_rate** — share of loaded fragments whose stub summary shares very few terms with recalled content (heuristic from `caw-eval`).
- **latency_ms** — wall clock per item.

Aggregates: mean per mode + on-vs-off deltas.

## Pre-building an Index

`caw-bench-build-index` is a streaming, resumable indexer. It runs ingestion on a rayon pool while a single GPU consumer embeds in sub-batches:

```bash
cargo run --release -p caw-bench --bin caw-bench-build-index -- \
  --corpus opencaw-corpora/sysdoc \
  --out opencaw-corpora/sysdoc.sqlite
```

Properties:
- **Pipelined**: rayon producers ingest files in parallel → bounded channel → GPU embeds and inserts. CPU and GPU stay busy concurrently.
- **Dedicated thread pool for ingestion** — prevents HuggingFace tokenizer's rayon use from deadlocking against producers.
- **Length-bucketed sub-batches**: sorted by text length before splitting, so padding tracks the local max rather than the batch-wide max.
- **Resumable**: skips `(path, mtime)` pairs already in the target sqlite. Kill it, restart it, pick up where it left off.
- **WAL + prepared-statement batch inserts**: one transaction per batch, prepared statements reused. Collapses N×3 fsyncs into a single commit.

## Running a Benchmark

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

`--trace-out` writes one JSONL line per (item, mode) with the system prompt, question, reference answer, loaded fragments, the model's answer, the judge's rationale, and the full metric vector. Filter failures:

```bash
jq 'select(.result.answer_score < 1)' bench.jsonl
```

### Adapters for benchmarking

| Flag value | Backend |
|---|---|
| `ollama` | Local Ollama at `--ollama-url` (default `http://localhost:11434`) |
| `vllm` | vLLM or any OpenAI-compatible server at `--openai-url` |
| `claude-code` | Local `claude` CLI — default judge (doesn't share weights with answer model) |

## Sweeps

`caw-bench-sweep` drives `caw-bench` across a parameter grid defined in a TOML config. Resumes if interrupted — cells with an existing `report.json` are skipped.

```bash
# Smoke test (2 cells, 2 items each)
cargo build -p caw-bench --release
./target/release/caw-bench-sweep --config crates/caw-bench/sweeps/minimal.toml

# Full roadmap grid (48 cells: 3 models × 4 workload/seed × 2 thresholds × 2 budgets)
./target/release/caw-bench-sweep --config crates/caw-bench/sweeps/default.toml

# Dry run
./target/release/caw-bench-sweep --config … --dry-run

# Force re-run all cells
./target/release/caw-bench-sweep --config … --force
```

Output:
```
bench-results/default/
├── manifest.jsonl              # one line per completed cell {hash, params}
├── 2430eff2b7f9625b/           # cell hash = sha256 of sorted params, truncated
│   ├── params.json
│   └── report.json
└── …
```

Config shape: `fixed` holds flags applied to every cell; `axes.<name>` is a list of tables where each table is one choice along that axis. The Cartesian product of all axes is the cell list. `snake_case` keys convert to `kebab-case` when passed to the subprocess.

## Intent Bench

`caw-bench-intent` evaluates small models as strict JSON query-intent classifiers — useful for picking a cheap routing model before the main answer model runs.

```bash
cargo run -p caw-bench --bin caw-bench-intent --release -- \
  --candidate qwen2:0.5b \
  --candidate tinyllama:1.1b \
  --candidate llama3.2:3b \
  --out intent-report.json

# Auto-discover local small models via ollama list
./scripts/intent-bench-ollama.sh
```

The JSON report includes overall exact-match rate, per-field accuracy, and per-tag exact-match rates. Knobs for `intent-bench-ollama.sh`:

```bash
MAX_GB=3 ./scripts/intent-bench-ollama.sh          # only test models ≤3 GB
LIMIT=8 ./scripts/intent-bench-ollama.sh            # cap discovered model count
NAME_REGEX='^(qwen2|tinyllama)' ./scripts/intent-bench-ollama.sh
./scripts/intent-bench-ollama.sh qwen2:0.5b llama3.2:3b  # bypass discovery
```

## Caveats

- Each item runs the orchestrator twice (on + off). A fresh adapter + retriever is built per item to keep runs independent.
- `false_recall_rate` is a heuristic (stub-summary-to-content term overlap), not ground truth. Compare on-vs-off for the same workload, not against an absolute bar.
- Whether `recall_on` diverges from `recall_off` depends on whether your model emits `<probe>` markers or `<think>` blocks the orchestrator can act on.
- The 30 hand-authored opencaw Q&A pairs are mostly single-fact lookups — useful as a baseline, not as a stress test for multi-pass refinement.
