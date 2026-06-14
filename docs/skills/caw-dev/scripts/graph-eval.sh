#!/usr/bin/env bash
# Run the retrieval-only chunk-rank eval (caw-bench-graph-eval) over the sysdoc
# chunk-level gold set, without re-deriving the index / QA / source-root paths.
# This is the deterministic instrument for "where does the gold chunk rank, and
# why does it miss" — pure retrieval, no answer model, so it's fast and is the
# right tool for iterating on chunking/embedding precision (unlike bench.sh,
# which runs the full gen+judge loop).
#
# Usage: graph-eval.sh [-- <extra caw-bench-graph-eval args>]
#   Everything after `--` is forwarded. Common ones:
#     --diagnose             decompose each miss (cosine vs bm25 rank, token
#                            overlap, thin-stub, buried) — the why behind a miss
#     --baseline hybrid|cosine   ranking baseline (default below is hybrid)
#     --fusion divide_total|present_weight|rrf|all
#     --recall-k 1,3,5,10,20,50
#     --miss-cutoff N        a gold ranked past N counts as a miss for --diagnose
#
# Defaults target the sysdoc n=100 chunk set (the independent eval corpus):
#   index        target/caw-dev/subset-medium.sqlite   (CAW_GE_INDEX)
#   questions    crates/caw-bench/src/qa/sysdoc_chunk_qa.json   (CAW_GE_QA)
#   source-root  opencaw-corpora/subset-medium   (CAW_GE_SRC)
# The gold `path`s are relative to source-root; it must be the curate-subset
# dest (scripts/curate-subset-medium.py), which is where the index was built.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--" ]; then shift; fi

INDEX="${CAW_GE_INDEX:-$ROOT/target/caw-dev/subset-medium.sqlite}"
QA="${CAW_GE_QA:-$ROOT/crates/caw-bench/src/qa/sysdoc_chunk_qa.json}"
SRC="${CAW_GE_SRC:-$ROOT/opencaw-corpora/subset-medium}"

for f in "$INDEX" "$QA"; do
  if [ ! -e "$f" ]; then echo "graph-eval: missing $f" >&2; exit 1; fi
done
if [ ! -d "$SRC" ]; then
  echo "graph-eval: source-root $SRC not found (run scripts/curate-subset-medium.py)" >&2
  exit 1
fi

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
if [ -n "$GPU" ]; then
  echo "graph-eval: pinning embedder to GPU $GPU (freest)" >&2
else
  echo "graph-eval: no GPU detected; embedder will run on CPU" >&2
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" cargo run --release --quiet -p caw-bench --bin caw-bench-graph-eval -- \
  --index "$INDEX" \
  --questions "$QA" \
  --source-root "$SRC" \
  --baseline hybrid \
  "$@"
