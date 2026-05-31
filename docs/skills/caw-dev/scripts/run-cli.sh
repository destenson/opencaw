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
# GPU is pinned to the freest device for any in-process CUDA embedder. The
# default CLI embedder is FastEmbed (CPU ONNX), so this is usually a no-op,
# but it keeps behavior consistent if the CLI is switched to candle.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"

cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES="$GPU" cargo run --quiet -p caw-cli -- "$@"
