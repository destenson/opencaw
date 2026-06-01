#!/usr/bin/env bash
# A/B value proof: send the SAME question to the proxy (context injected) and
# straight to the upstream (no context), and show both answers side by side.
# This is the test that actually demonstrates the proxy adds value — a small
# model that names project-specific symbols/paths only via the proxy is the
# signal that retrieval changed the answer, not just that it answered.
#
# Usage: ab-test.sh [proxy-port] [upstream-base] [model] [question]
#   proxy-port      default 8090
#   upstream-base   default http://localhost:11434/v1 (must match serve.sh's upstream)
#   model           default llama3.2:3b
#   question        if omitted, the whole shared query pool (queries.txt) is run,
#                   one A/B pair per question; pass a question to run only that one
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

PORT="${1:-8090}"
UPSTREAM="${2:-http://localhost:11434/v1}"
MODEL="${3:-llama3.2:3b}"
LOG="$ROOT/target/caw-dev/server-$PORT.log"

# Questions: explicit 4th arg overrides the pool; otherwise read queries.txt
# (skip blanks and '#' comments).
QUESTIONS=()
if [ "${4-}" != "" ]; then
  QUESTIONS=("$4")
else
  while IFS= read -r line; do
    [ -n "$line" ] && [ "${line#\#}" = "$line" ] && QUESTIONS+=("$line")
  done < "$SCRIPT_DIR/queries.txt"
fi

ask() {
  local base="$1" question="$2"
  local q_json
  q_json=$(printf '%s' "$question" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
  curl -s --max-time 240 "$base/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\": \"$MODEL\", \"stream\": false, \"messages\": [{\"role\": \"user\", \"content\": $q_json}]}" \
  | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception as e:
    print("(could not parse response:", e, ")")'
}

echo "ab-test: questions=${#QUESTIONS[@]}  model=$MODEL" >&2
for QUESTION in "${QUESTIONS[@]}"; do
  echo "Q: $QUESTION"
  echo
  echo "================ B: DIRECT upstream (no context) ================"
  ask "$UPSTREAM" "$QUESTION"
  echo
  echo "================ A: via PROXY (context injected) ================"
  ask "http://localhost:$PORT/v1" "$QUESTION"
  echo
  echo "================ injection proof (server log) ================"
  [ -f "$LOG" ] && grep "augmented with" "$LOG" | tail -1 || echo "(no server log at $LOG)"
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo
done
