#!/usr/bin/env bash
# Run the caw-cli interactive recall loop — the REAL engine (multi-pass
# recall, probes, eviction, consolidation), unlike the single-shot proxy.
# Use this to develop and test the thinking-trace recall path itself.
#
# Usage: run-cli.sh [extra caw-cli args...]
#   Forwards all args to `cargo run -p caw-cli`. Pin a model/adapter, e.g.:
#     run-cli.sh --show-intent
#     run-cli.sh --adapter ollama --model qwen3.5:9b --show-intent
#   For non-interactive testing, pipe questions on stdin (one per line):
#     printf 'what is opencaw?\nhow does eviction work?\n' | run-cli.sh
#
# GPU is pinned to the freest device for the in-process candle embedder, which
# runs BGE on the GPU (the CLI embedder was switched from FastEmbed/ONNX to
# candle in d29c679), so the pin is load-bearing, not a no-op.
#
# Runs --release: the recall engine is compute-heavy and the debug HNSW graph
# build is ~17x slower (per-turn latency ~33s debug vs ~0.1s release). The first
# release build is a long one-time compile; subsequent runs reuse it.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"

# Keep dev artifacts in-tree under target/caw-dev/ instead of the binary's
# default (~/.cache/caw/…). The bare binary writes to the user cache so an
# end-user `caw` run never drops files into a project; for the dev harness we
# want the index + session history wiped by `cargo clean` alongside everything
# else, matching build-index.sh and serve.sh. Keyed per-corpus (the --dir arg,
# default "crates") so distinct corpora don't share an index. Only injected when
# the caller hasn't already passed --db / --session-dir.
DEV_STATE="$ROOT/target/caw-dev"
corpus="crates"
prev=""
have_db=0
have_sessions=0
for arg in "$@"; do
  [ "$prev" = "--dir" ] && corpus="$arg"
  [ "$arg" = "--db" ] && have_db=1
  [ "$arg" = "--session-dir" ] && have_sessions=1
  prev="$arg"
done
slug="$(printf '%s' "$corpus" | tr -c 'A-Za-z0-9._-' '_')"
extra=()
[ "$have_db" -eq 0 ] && extra+=(--db "$DEV_STATE/cli/$slug/index.db")
[ "$have_sessions" -eq 0 ] && extra+=(--session-dir "$DEV_STATE/cli/$slug/sessions")

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" cargo run --release --quiet -p caw-cli -- "${extra[@]}" "$@"
