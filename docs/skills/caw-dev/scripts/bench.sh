#!/usr/bin/env bash
# Run the caw-bench end-to-end recall harness (recall-on vs recall-off at
# matched budget, judge-scored answers) without re-deriving the corpus paths,
# GPU pinning, and release-build flags every time.
#
# Usage: bench.sh [workload] [-- <extra caw-bench args>]
#   workload   sysdoc (default) | opencaw | niah
#
# Everything after `--` is forwarded verbatim to the caw-bench binary, so the
# full flag surface stays available (see `caw-bench --help`). Common ones:
#   --num-predict N      cap the answer model's per-completion budget (gen study)
#   --judge-adapter groq fast judge (~19x vs claude-code; re-baselines absolutes)
#   --only-mode on|off   run a single recall mode
#   --limit N            run only the first N items (directional small-n runs)
#   --out PATH           write the JSON report (default: stdout)
#   --trace-out PATH     per-(item,mode) JSONL with prompts, answers, scores
#
# Two non-obvious choices baked in, matching the other scripts here:
#  - GPU selection: the candle embedder is pinned to the freest GPU via
#    pick-gpu.sh, because it dies on OOM rather than shrinking and GPU 0 is
#    usually full of an Ollama model.
#  - Release build: debug builds make instant-distance's HNSW build over the
#    43k-stub index peg every core for minutes (a recorded gotcha) and distort
#    every latency number, so this always uses --release.
#
# sysdoc requires a prebuilt index + QA file; both default to the canonical
# locations and are overridable with CAW_BENCH_INDEX / CAW_BENCH_QA. The index
# is produced by build-index.sh over the sysdoc corpus subset; see
# scripts/curate-subset-medium.py in the repo root.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKLOAD="${1:-sysdoc}"
if [ "$#" -gt 0 ]; then shift; fi
# Tolerate an explicit `--` separator before the forwarded args.
if [ "${1:-}" = "--" ]; then shift; fi

INDEX="${CAW_BENCH_INDEX:-$ROOT/target/caw-dev/subset-medium.sqlite}"
QA="${CAW_BENCH_QA:-$ROOT/opencaw-corpora/sysdoc_qa.json}"

WORKLOAD_ARGS=(--workload "$WORKLOAD")
if [ "$WORKLOAD" = "sysdoc" ]; then
  if [ ! -f "$INDEX" ]; then
    echo "bench: sysdoc index not found at $INDEX" >&2
    echo "bench: build it first, e.g. build-index.sh <sysdoc-corpus> $INDEX" >&2
    exit 1
  fi
  if [ ! -f "$QA" ]; then
    echo "bench: sysdoc QA file not found at $QA" >&2
    exit 1
  fi
  WORKLOAD_ARGS+=(--index "$INDEX" --qa-file "$QA")
fi

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
if [ -n "$GPU" ]; then
  echo "bench: pinning embedder to GPU $GPU (freest)" >&2
else
  echo "bench: no GPU detected; embedder will run on CPU" >&2
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" cargo run --release --quiet -p caw-bench --bin caw-bench -- \
  "${WORKLOAD_ARGS[@]}" \
  "$@"
