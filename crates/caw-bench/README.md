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
- **context_efficiency** — share of the total workspace tokens that are
  actual recalled content (as opposed to stubs + prompt + question).
- **false_recall_rate** — share of loaded fragments whose stub summary
  shares very few non-stopword terms with the recalled content (heuristic
  from `caw-eval`'s `FalseRecallMetrics`).
- **latency_ms** — wall clock.

Aggregates compute mean values per mode plus the on-vs-off deltas.

## Modes

- `recall_on` — full `DynamicRecallOrchestrator`: initial retrieval,
  multi-pass iteration, probes, thinking-trace recall, eviction,
  consolidation.
- `recall_off` — single-shot retrieval only (`max_recall_iterations=0`,
  probes off, traces off). Same model, same budget, same corpus. Isolates
  the contribution of the dynamic recall machinery.

## Workloads

### `niah` (needle-in-a-haystack)

Procedurally generates bureaucratic project-status filler from a seeded
grammar, inserts a short memo containing a fabricated fact, and asks the
model to retrieve that fact. The filler is out-of-distribution by
construction (fabricated project names, operator handles, regional codes),
so the model can't fall back on training knowledge.

Flags: `--niah-items N --niah-filler P --niah-seed S`.

### `opencaw` (opencaw-on-opencaw)

Hand-authored Q&A drawn from this repository's own code, README, SCOPE,
TODO, and design doc. The corpus is whatever `.rs / .md / .toml` files
`--repo-root` points at (default `.`). Scoring uses a judge model
(default Haiku) to grade answers against reference strings.

Questions live in `src/qa/opencaw_qa.json` and are embedded into the binary
via `include_str!`. Add or edit pairs there.

## Usage

Uses the local `claude` CLI (Claude Code) for both answering and judging —
no API key required. Pass `--model` shortcuts (`sonnet`, `opus`, `haiku`)
via `--answer-model` and `--judge-model`.

```bash
# Small smoke test (3 NIAH items, both modes)
cargo run -p caw-bench -- --workload niah --niah-items 3 --limit 3

# Full NIAH sweep
cargo run -p caw-bench --release -- \
  --workload niah \
  --niah-items 10 \
  --niah-filler 80 \
  --out niah_report.json

# opencaw Q&A against the current checkout
cargo run -p caw-bench --release -- \
  --workload opencaw \
  --repo-root . \
  --out opencaw_report.json
```

The JSON report contains per-item results plus per-mode aggregates. A
human-readable summary is written to stderr at the end.

## Caveats

- Each item runs the orchestrator twice (on + off), so cost is 2x the
  answer-model tokens per item plus one judge call per item for the
  opencaw workload. NIAH has no judge cost.
- A fresh orchestrator is built per item to keep runs independent. That
  costs a few hundred ms of setup per item (embedder init, HNSW build) —
  predictable but real.
- The default answer model is Sonnet, the default judge is Haiku. Both
  run via the local `claude` CLI (Claude Code). Remote Anthropic API,
  vLLM, and other backends work at the adapter layer but aren't wired to
  the harness CLI yet.
- `false_recall_rate` is a heuristic (stub-summary-to-content term
  overlap), not a ground-truth correctness measure. Interpretation is
  relative — compare on-vs-off for the same workload, not an absolute
  pass/fail bar.
