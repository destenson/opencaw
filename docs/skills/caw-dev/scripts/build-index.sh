#!/usr/bin/env bash
# Build a retrieval index over a corpus directory for caw-server / caw-cli.
#
# Usage: build-index.sh [--no-rebuild] <corpus-dir> <out.sqlite> [batch-size] [sub-batch-size]
#   --no-rebuild     incremental update: keep the existing index and only ingest
#                    new/changed files (by path+mtime). Default is a full rebuild
#                    (deletes the index first). The Rust indexer is incremental by
#                    default; this script forces --rebuild for a clean slate, and
#                    --no-rebuild opts back out. Use it to refresh an index cheaply
#                    (e.g. new Claude Code session docs) instead of re-embedding the
#                    whole corpus. Caveat: incremental does NOT remove stubs for
#                    files deleted from the corpus, and a changed file can leave
#                    orphan stubs if its chunk boundaries shift — use a full
#                    rebuild when the corpus has shrunk or chunking config changed.
#   corpus-dir       directory whose files form the corpus (single root)
#   out.sqlite       output index path (parent dirs created as needed)
#   batch-size       default 2048 (sort window; bigger = better length bucketing)
#   sub-batch-size   default 256  (cap on sub-batch width; see below)
#
# Two non-obvious choices baked in:
#  - GPU selection: pinned to the freest GPU via pick-gpu.sh, because the
#    candle embedder dies on OOM rather than shrinking. (The embedder also
#    accepts CAW_EMBED_DEVICE=cuda:N for explicit selection without masking.)
#  - Adaptive sub-batching: the builder sorts each batch by text length and
#    forms GPU sub-batches by a char budget (EMBED_PADDED_CHAR_BUDGET), so short
#    chunks batch hundreds wide while long chunks batch a few dozen — high
#    throughput without OOM. No need to force a tiny fixed count anymore; a large
#    batch-size just widens the sort window. sub-batch-size only caps the width
#    for floods of very short texts.
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

# Default to a full rebuild (clean slate); --no-rebuild opts into incremental.
REBUILD="--rebuild"
args=()
for a in "$@"; do
  if [ "$a" = "--no-rebuild" ]; then
    REBUILD=""
  else
    args+=("$a")
  fi
done
set -- "${args[@]}"

CORPUS="${1:?usage: build-index.sh [--no-rebuild] <corpus-dir> <out.sqlite> [batch] [sub-batch]}"
OUT="${2:?usage: build-index.sh [--no-rebuild] <corpus-dir> <out.sqlite> [batch] [sub-batch]}"
BATCH="${3:-2048}"
SUB_BATCH="${4:-256}"

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
if [ -n "$GPU" ]; then
  echo "build-index: pinning embedder to GPU $GPU (freest)" >&2
else
  echo "build-index: no GPU detected; embedder will run on CPU" >&2
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" cargo run --release --quiet -p caw-bench --bin caw-bench-build-index -- \
  --corpus "$CORPUS" \
  --out "$OUT" \
  $REBUILD \
  --backend candle \
  --batch-size "$BATCH" \
  --sub-batch-size "$SUB_BATCH" \
  --log-interval 5
