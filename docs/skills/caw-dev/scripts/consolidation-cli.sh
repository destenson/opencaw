#!/usr/bin/env bash
# End-to-end test of the differentiating engine: the workspace fills past its
# token budget, fragments are EVICTED, and each eviction triggers an
# LLM-synthesized CONSOLIDATION note that is persisted to the stub store so it
# can be recalled on later turns. This is the exact path test-cli.sh disables
# (--no-llm-consolidation); here it is ON.
#
# The aux model (summarization + consolidation) is routed through the local
# `claude` CLI via ClaudeCodeAdapter, so no Anthropic API key is needed — but
# each eviction spawns one `claude` call that costs real tokens (~$0.01 per
# eviction with haiku). The defaults keep the scenario short on purpose.
#
# Usage:
#   consolidation-cli.sh [--dir DIR] [--model M] [--aux-model AUX]
#                        [--max-tokens N] [--max-candidates N]
#                        [-q "question" ...] [-- extra caw-cli args]
#   printf 'q1\nq2\n' | consolidation-cli.sh
#
# Defaults (chosen to force eviction within a few turns):
#   --dir crates                     corpus to ingest (reuses the warm cli-index.db)
#   --model llama3.2:3b              completion adapter (ollama)
#   --aux-model haiku               -> ClaudeCodeAdapter::haiku(); only haiku|sonnet
#                                      are meaningful (see build_aux_adapter)
#   --max-tokens 1200               small workspace so admissions overflow -> eviction
#   --max-candidates 30             wide candidate pool so turns admit a lot
#   --no-intent-classifier          keep the run deterministic/fast
#
# After the run it prints a COMPUTED proof summary parsed from the verbose log:
# eviction count, consolidation notes persisted, aux LLM calls, and total aux
# cost. Interpretation of those numbers is left to the caller — the script only
# reports what it counted.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIR="crates"
MODEL="llama3.2:3b"
AUX_MODEL="haiku"
MAX_TOKENS="1200"
MAX_CANDIDATES="30"
QUESTIONS=()
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --aux-model) AUX_MODEL="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
    -q|--q) QUESTIONS+=("$2"); shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "consolidation-cli.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

# Questions from stdin if none passed as -q. The default set is topically
# coherent (the recall engine itself) so the consolidation notes synthesized on
# eviction are about a single subject rather than noise.
if [ "${#QUESTIONS[@]}" -eq 0 ] && [ ! -t 0 ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && QUESTIONS+=("$line")
  done
fi
if [ "${#QUESTIONS[@]}" -eq 0 ]; then
  QUESTIONS=(
    "How does the dynamic recall loop decide which fragments to admit?"
    "What triggers eviction of a fragment from the workspace?"
    "How are consolidation notes synthesized when fragments are evicted?"
    "Walk me through provenance tagging end to end."
  )
fi

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
mkdir -p "$ROOT/target/caw-dev"
DB="$ROOT/target/caw-dev/cli-index.db"
LOG="$ROOT/target/caw-dev/consolidation-cli.log"

echo "consolidation-cli: dir=$DIR model=$MODEL aux=$AUX_MODEL max_tokens=$MAX_TOKENS turns=${#QUESTIONS[@]}" >&2
echo "consolidation-cli: full verbose log -> $LOG" >&2
echo "consolidation-cli: aux adapter spawns the 'claude' CLI once per eviction (real token cost)" >&2

cd "$ROOT"
# --verbose enables debug-level logs so eviction + consolidation events are
# emitted. We omit --no-llm-consolidation so the LLM consolidation synthesizer
# runs. --no-llm-summarize is kept ON: the warm cli-index.db already has stubs,
# so re-summarizing buys nothing and would add aux cost; flip it off (drop the
# flag) only when building a fresh index where you also want LLM stub summaries.
set +e
printf '%s\n' "${QUESTIONS[@]}" | env CUDA_VISIBLE_DEVICES="$GPU" \
  cargo run --quiet -p caw-cli -- \
    --dir "$DIR" \
    --adapter ollama --model "$MODEL" \
    --aux-model "$AUX_MODEL" \
    --no-llm-summarize \
    --db "$DB" \
    --session-dir "$ROOT/target/caw-dev/cli-sessions" \
    --max-tokens "$MAX_TOKENS" \
    --max-candidates "$MAX_CANDIDATES" \
    --no-intent-classifier \
    --verbose \
    "${EXTRA[@]}" > "$LOG" 2>&1
RC=$?
set -e

# Computed proof summary. Every number below is grepped/summed from the log;
# the script makes no claim about what the numbers mean.
EVICTED=$(grep -c 'fragment evicted' "$LOG" || true)
NOTES_STORE=$(grep -c 'consolidation note persisted to store' "$LOG" || true)
NOTES_MEM=$(grep -c 'consolidation note recorded in memory' "$LOG" || true)
AUX_CALLS=$(grep -c '\[claude-code\] model=' "$LOG" || true)
AUX_COST=$(grep -oE '\[claude-code\] model=[^ ]+ cost=\$[0-9.]+' "$LOG" \
  | grep -oE 'cost=\$[0-9.]+' | sed 's/cost=\$//' \
  | awk '{s+=$1} END {printf "%.4f", s+0}')

echo
echo "================ consolidation proof (from $LOG) ================"
echo "caw-cli exit code:                 $RC"
echo "fragments evicted:                 $EVICTED"
echo "consolidation notes -> store:      $NOTES_STORE"
echo "consolidation notes -> memory:     $NOTES_MEM"
echo "aux (claude-code) LLM calls:       $AUX_CALLS"
echo "aux total cost (USD):              \$$AUX_COST"
echo "per-turn workspace sizes:"
grep -oE '\[workspace: [0-9]+ fragments, ~[0-9]+ tokens\]' "$LOG" | sed 's/^/  /' || true

exit "$RC"
