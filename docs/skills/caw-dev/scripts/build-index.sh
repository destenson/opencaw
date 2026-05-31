#!/usr/bin/env bash
# Build a retrieval index over a corpus directory for caw-server / caw-cli.
#
# Usage: build-index.sh <corpus-dir> <out.sqlite> [batch-size] [sub-batch-size]
#   corpus-dir       directory whose files form the corpus (single root)
#   out.sqlite       output index path (parent dirs created as needed)
#   batch-size       default 32  (small on purpose — see below)
#   sub-batch-size   default 8
#
# Two non-obvious choices baked in:
#  - GPU selection: pinned to the freest GPU via pick-gpu.sh, because the
#    candle embedder hardcodes cuda:0 and won't fall back on OOM.
#  - Tiny batches: BGE attention memory scales as batch x seq^2. The bench
#    default sub-batch of 256 OOMs even with 14 GB free on long chunks. The
#    source tree is small, so throughput is irrelevant; correctness isn't.
#
# corpus-dir / corpus-root contract: stub paths are stored RELATIVE to
# <corpus-dir>. At serve time, caw-server resolves them against its
# --corpus-root, so pass the SAME directory to both. Do NOT point the
# corpus at the repo root: the indexer's skip rules drop hidden/scripts/
# target/node_modules and binaries, but NOT data/ or opencaw-corpora/
# (multi-GB system-doc corpora). Index crates/ or docs/, not the root.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CORPUS="${1:?usage: build-index.sh <corpus-dir> <out.sqlite> [batch] [sub-batch]}"
OUT="${2:?usage: build-index.sh <corpus-dir> <out.sqlite> [batch] [sub-batch]}"
BATCH="${3:-32}"
SUB_BATCH="${4:-8}"

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
if [ -n "$GPU" ]; then
  echo "build-index: pinning embedder to GPU $GPU (freest)" >&2
else
  echo "build-index: no GPU detected; embedder will run on CPU" >&2
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" cargo run --quiet -p caw-bench --bin caw-bench-build-index -- \
  --corpus "$CORPUS" \
  --out "$OUT" \
  --rebuild \
  --backend candle \
  --batch-size "$BATCH" \
  --sub-batch-size "$SUB_BATCH" \
  --log-interval 5
