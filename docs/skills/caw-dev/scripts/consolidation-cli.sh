#!/usr/bin/env bash
# End-to-end test of the differentiating engine: the workspace fills past its
# token budget, fragments are EVICTED, and each eviction triggers an
# LLM-synthesized CONSOLIDATION note that is persisted to the stub store so it
# can be recalled on later turns. This is the exact path test-cli.sh disables
# (--no-llm-consolidation); here it is ON.
#
# The aux model (summarization + consolidation) defaults to a local Ollama
# model, so the whole engine runs locally with no API key. Point it at the
# `claude` CLI instead with `--aux-adapter claude-code-haiku` (still no API key;
# ClaudeCodeAdapter shells out to the installed `claude`, ~$0.01 per eviction).
# The defaults keep the scenario short on purpose.
#
# Aux must be a NON-thinking model. A thinking model (qwen3.x, deepseek-r1, …)
# routinely emits only a reasoning trace for the consolidation prompt; after
# split_thinking the answer body is blank, the adapter logs `degenerate output:
# blank answer`, and eviction falls back to a deterministic templated note
# ("Evicted (relevance decayed …)") instead of real LLM synthesis — so the very
# path this script exists to demonstrate is skipped. It is also far slower
# (~90-115s/note vs ~real-time), enough that the default multi-turn run cannot
# finish inside a normal timeout. llama3.2:3b is the validated default.
#
# Usage:
#   consolidation-cli.sh [--dir DIR] [--model M]
#                        [--aux-adapter A] [--aux-model AUX]
#                        [--max-tokens N] [--max-candidates N]
#                        [-q "question" ...] [-- extra caw-cli args]
#   printf 'q1\nq2\n' | consolidation-cli.sh
#
# Defaults (chosen to force eviction within a few turns):
#   --dir crates                     corpus to ingest (reuses the warm cli-index.db)
#   --model llama3.2:3b              completion adapter (ollama)
#   --aux-adapter ollama             aux runs on the local stack (selectors match
#                                    caw-cli --adapter: ollama, claude-code-haiku, …)
#   --aux-model llama3.2:3b          non-thinking, fast, reliable synthesis (see note above)
#   --max-tokens 1200               small workspace so admissions overflow -> eviction
#   --max-candidates 30             wide candidate pool so turns admit a lot
#   --no-intent-classifier          keep the run deterministic/fast
#
# After the run it prints a COMPUTED proof summary parsed from the verbose log
# and the per-call prompt dumps (--save-prompt, on automatically): eviction
# count, consolidation notes persisted, how many later turns RECALLED a persisted
# note (the full loop, not just persistence), and — only when --aux-adapter is
# claude-code* — the claude-code LLM call count and total cost. Interpretation of
# those numbers is left to the caller — the script only reports what it counted.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIR="crates"
MODEL="llama3.2:3b"
AUX_ADAPTER="ollama"
AUX_MODEL="llama3.2:3b"
MAX_TOKENS="1200"
MAX_CANDIDATES="30"
QUESTIONS=()
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --aux-adapter) AUX_ADAPTER="$2"; shift 2 ;;
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
# This default set is intentionally NOT the shared queries.txt pool: that pool
# spans distinct subsystems, whereas this script needs a topically COHERENT batch
# (all about the recall engine) so the consolidation notes synthesized on
# eviction concern a single subject rather than noise. More turns here = more
# eviction/consolidation events to observe, so the larger the coherent batch the
# better — just keep every question on the recall/eviction/consolidation theme.
if [ "${#QUESTIONS[@]}" -eq 0 ]; then
  QUESTIONS=(
    "How does the dynamic recall loop decide which fragments to admit?"
    "What triggers eviction of a fragment from the workspace?"
    "How are consolidation notes synthesized when fragments are evicted?"
    "Where are consolidation notes persisted, and how are they recalled on later turns?"
    "How does relevance decay affect which fragments survive across turns?"
    "What role does the workspace token budget play in eviction?"
    "Walk me through provenance tagging end to end."
  )
fi

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
mkdir -p "$ROOT/target/caw-dev"
DB="$ROOT/target/caw-dev/cli-index.db"
LOG="$ROOT/target/caw-dev/consolidation-cli.log"

echo "consolidation-cli: dir=$DIR model=$MODEL aux=$AUX_ADAPTER/$AUX_MODEL max_tokens=$MAX_TOKENS turns=${#QUESTIONS[@]}" >&2
echo "consolidation-cli: full verbose log -> $LOG" >&2

cd "$ROOT"
SESSION_DIR="$ROOT/target/caw-dev/cli-sessions"
mkdir -p "$SESSION_DIR"
# Sentinel to identify the prompt dumps THIS run produces (others accumulate in
# the same dir across runs). --save-prompt writes one prompt-{ts}-turn-{N}.txt
# per model call beside the session files; we grep only files newer than this.
RUN_SENTINEL="$SESSION_DIR/.run-sentinel"
touch "$RUN_SENTINEL"

# --verbose enables debug-level logs so eviction + consolidation events are
# emitted. We omit --no-llm-consolidation so the LLM consolidation synthesizer
# runs. --no-llm-summarize is kept ON: the warm cli-index.db already has stubs,
# so re-summarizing buys nothing; flip it off (drop the flag) only when building
# a fresh index where you also want LLM stub summaries.
# --save-prompt dumps the exact workspace per model call: when a previously
# evicted+consolidated stub is re-recalled on a later turn, its content carries a
# "[Prior session notes for this source:" block (dynamic.rs prepends it). That
# block is the ONLY evidence of the full novelty loop — eviction notes being
# RECALLED, not merely persisted — so we count it below.
set +e
printf '%s\n' "${QUESTIONS[@]}" | env CUDA_VISIBLE_DEVICES="$GPU" \
  cargo run --release --quiet -p caw-cli -- \
    --dir "$DIR" \
    --adapter ollama --model "$MODEL" \
    --aux-adapter "$AUX_ADAPTER" --aux-model "$AUX_MODEL" \
    --no-llm-summarize \
    --db "$DB" \
    --session-dir "$SESSION_DIR" \
    --max-tokens "$MAX_TOKENS" \
    --max-candidates "$MAX_CANDIDATES" \
    --no-intent-classifier \
    --save-prompt \
    --verbose \
    "${EXTRA[@]}" > "$LOG" 2>&1
RC=$?
set -e

# Computed proof summary. Every number below is grepped/summed from the log;
# the script makes no claim about what the numbers mean.
EVICTED=$(grep -c 'fragment evicted' "$LOG" || true)
NOTES_STORE=$(grep -c 'consolidation note persisted to store' "$LOG" || true)
NOTES_MEM=$(grep -c 'consolidation note recorded in memory' "$LOG" || true)
# A blank aux answer makes eviction fall back to a deterministic templated note
# instead of real LLM synthesis. The count maps 1:1 to those fallback notes, so
# (NOTES_STORE - DEGEN) is the number of notes that were actually LLM-synthesized.
DEGEN=$(grep -c 'degenerate output' "$LOG" || true)
# Full-loop proof: how many model calls THIS run received a workspace containing
# a recalled consolidation note (the "[Prior session notes…" prepend). Counted
# only over prompt dumps newer than the sentinel so prior runs don't inflate it.
# A persisted note may originate from this run's eviction or an earlier run's —
# either way its presence proves persisted notes are recalled into later turns.
RECALL_MARKER='Prior session notes for this source'
RUN_PROMPTS=$(find "$SESSION_DIR" -name 'prompt-*.txt' -newer "$RUN_SENTINEL" 2>/dev/null)
NOTES_RECALLED=0
if [ -n "$RUN_PROMPTS" ]; then
  NOTES_RECALLED=$(printf '%s\n' "$RUN_PROMPTS" \
    | xargs grep -l "$RECALL_MARKER" 2>/dev/null | wc -l | tr -d ' ')
fi
rm -f "$RUN_SENTINEL"
# Per-call cost is only logged by ClaudeCodeAdapter; Ollama emits no cost line.
CC_CALLS=$(grep -c '\[claude-code\] model=' "$LOG" || true)
# `|| true`: with `set -o pipefail`, the leading grep exits 1 when there are no
# claude-code lines (the default Ollama case), which would otherwise abort the
# script under `set -e` before the summary prints. awk still emits 0.0000.
CC_COST=$(grep -oE '\[claude-code\] model=[^ ]+ cost=\$[0-9.]+' "$LOG" \
  | grep -oE 'cost=\$[0-9.]+' | sed 's/cost=\$//' \
  | awk '{s+=$1} END {printf "%.4f", s+0}' || true)

echo
echo "================ consolidation proof (from $LOG) ================"
echo "caw-cli exit code:                 $RC"
echo "fragments evicted:                 $EVICTED"
echo "consolidation notes -> store:      $NOTES_STORE"
echo "consolidation notes -> memory:     $NOTES_MEM"
echo "degenerate aux outputs (fallback): $DEGEN"
echo "later turns recalling a note:      $NOTES_RECALLED  (full loop: persisted note -> recalled into a later workspace)"
if [ "$CC_CALLS" -gt 0 ]; then
  echo "claude-code aux LLM calls:         $CC_CALLS"
  echo "claude-code aux cost (USD):        \$$CC_COST"
fi
echo "per-turn workspace sizes:"
grep -oE '\[workspace: [0-9]+ fragments, ~[0-9]+ tokens\]' "$LOG" | sed 's/^/  /' || true

exit "$RC"
