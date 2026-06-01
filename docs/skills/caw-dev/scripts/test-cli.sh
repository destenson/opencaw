#!/usr/bin/env bash
# Drive the caw-cli recall engine (the real multi-pass loop) non-interactively
# with a fully-local config — no Anthropic key, no cloud. Questions come from
# args (one per --q) or stdin (one per line); answers + recall logs print to
# the terminal.
#
# Usage:
#   test-cli.sh [--dir DIR] [--model M] [--intent MODE] [-- extra caw-cli args]
#   printf 'q1\nq2\n' | test-cli.sh
#   test-cli.sh -q "what is opencaw?" -q "how does eviction work?"
#
# Defaults chosen for a local-only run:
#   --dir crates                      corpus to ingest
#   --adapter ollama --model llama3.2:3b
#   --no-llm-summarize --no-llm-consolidation   (avoid the haiku aux model / API key)
#   --intent granite4:micro           single fast intent model (MODE=none disables)
#   --db target/caw-dev/cli-index.db  persistent index under the build dir
# The first run ingests + embeds the corpus (CPU FastEmbed) and is slow; the
# --db cache makes subsequent runs fast.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIR="crates"
MODEL="llama3.2:3b"
INTENT="granite4:micro"   # or "none" to disable the classifier
QUESTIONS=()
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --intent) INTENT="$2"; shift 2 ;;
    -q|--q) QUESTIONS+=("$2"); shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "test-cli.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

# Collect questions from stdin if none were passed as -q.
if [ "${#QUESTIONS[@]}" -eq 0 ] && [ ! -t 0 ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && QUESTIONS+=("$line")
  done
fi
if [ "${#QUESTIONS[@]}" -eq 0 ]; then
  QUESTIONS=("In this project, which struct owns the multi-pass recall loop and what file is it in?")
fi

INTENT_ARGS=(--intent-model "$INTENT")
[ "$INTENT" = "none" ] && INTENT_ARGS=(--no-intent-classifier)

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
DB="$ROOT/target/caw-dev/cli-index.db"
mkdir -p "$ROOT/target/caw-dev"

echo "test-cli: dir=$DIR model=$MODEL intent=$INTENT db=$DB questions=${#QUESTIONS[@]}" >&2

cd "$ROOT"
# caw-cli reads queries from stdin, one per line, and exits at EOF.
# --db and --session-dir both point under target/caw-dev so the CLI's default
# .caw/ index + session dirs don't litter the repo root.
printf '%s\n' "${QUESTIONS[@]}" | env CUDA_VISIBLE_DEVICES="$GPU" \
  cargo run --release --quiet -p caw-cli -- \
    --dir "$DIR" \
    --adapter ollama --model "$MODEL" \
    --no-llm-summarize --no-llm-consolidation \
    --db "$DB" \
    --session-dir "$ROOT/target/caw-dev/cli-sessions" \
    --show-intent \
    "${INTENT_ARGS[@]}" \
    "${EXTRA[@]}"
