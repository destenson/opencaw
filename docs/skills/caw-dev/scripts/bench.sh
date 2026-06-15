#!/usr/bin/env bash
# Run the caw-bench end-to-end recall harness (recall-on vs recall-off at
# matched budget, judge-scored answers) without re-deriving the corpus paths,
# GPU pinning, and release-build flags every time.
#
# Usage: bench.sh [workload] [-- <extra caw-bench args>]
#   workload   sysdoc (default) | opencaw | code-agent | niah
#
# code-agent: coding-agent mid-task info-needs (signatures, struct fields, trait
#   bounds, call sites) over this repo. Same corpus as opencaw; needle-scored
#   exact code facts plus judge-scored synthesis questions. No prebuilt index
#   needed (the repo is ingested in-memory once and shared across items).
#
# Everything after `--` is forwarded verbatim to the caw-bench binary, so the
# full flag surface stays available (see `caw-bench --help`). Common ones:
#   --num-predict N      cap the answer model's per-completion budget (gen study)
#   --concurrency N      run N (item,mode) tasks in parallel (default 4); 1 =
#                        serial, the only path with valid per-phase timing
#   --only-mode on|off   run a single recall mode
#   --limit N            run only the first N items (directional small-n runs)
#   --out PATH           write the JSON report (default: stdout)
#   --trace-out PATH     per-(item,mode) JSONL with prompts, answers, scores
#
# Three non-obvious choices baked in, matching the other scripts here:
#  - Judge: defaults to groq (~19x faster than claude-code/haiku, judge drops
#    from ~8.5s to ~0.45s per item) because this is a dev-iteration script.
#    Override with CAW_BENCH_JUDGE=claude-code (or pass your own --judge-adapter).
#    Caveat: groq-70b and haiku score differently, so groq-judged absolute
#    answer_scores are NOT comparable to the haiku-judged numbers recorded in
#    docs/findings.md — a within-run recall-on/off delta is still valid (both
#    modes share the judge), but cross-run absolute comparisons need a matched
#    judge. The caw-bench BINARY default stays claude-code for that comparability;
#    this convenience script trades it for speed.
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

# Default the judge to groq for fast dev iteration, unless the caller already
# supplied their own --judge-adapter in the forwarded args.
JUDGE_ARGS=()
if [[ " $* " != *" --judge-adapter "* ]]; then
  JUDGE_ARGS+=(--judge-adapter "${CAW_BENCH_JUDGE:-groq}")
fi

# Optionally run the ANSWER model on a fast remote adapter (e.g. groq) instead
# of the slow local model, for quick iteration on the harness/instrument.
# Local generation dominates wall time (tens of seconds per item); a remote
# answer model cuts that to seconds. Note this is fast but NOT free: groq bills
# per token, so prefer the smallest model and small --limit runs. Defaults to
# the smallest groq model for this reason; override with CAW_BENCH_ANSWER_MODEL.
# The local model stays the overall default because it costs no API money and
# is usually the measurement target (the recall effect is model-specific).
# Skipped if the caller already passed --answer-adapter.
ANSWER_ARGS=()
if [[ " $* " != *" --answer-adapter "* ]] && [ -n "${CAW_BENCH_ANSWER:-}" ]; then
  ANSWER_ARGS+=(--answer-adapter "$CAW_BENCH_ANSWER")
  ANSWER_MODEL="${CAW_BENCH_ANSWER_MODEL:-}"
  if [ -z "$ANSWER_MODEL" ] && [ "$CAW_BENCH_ANSWER" = "groq" ]; then
    ANSWER_MODEL="llama-3.1-8b-instant"  # smallest/cheapest groq chat model
  fi
  if [ -n "$ANSWER_MODEL" ]; then
    ANSWER_ARGS+=(--answer-model "$ANSWER_MODEL")
  fi
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
  "${JUDGE_ARGS[@]}" \
  "${ANSWER_ARGS[@]}" \
  "$@"
