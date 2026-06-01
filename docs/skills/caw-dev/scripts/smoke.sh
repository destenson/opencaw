#!/usr/bin/env bash
# Send one chat completion through the running proxy and verify that context
# was actually injected (not just that the model answered plausibly).
#
# Usage: smoke.sh [port] [model] [question]
#   port      proxy port (default 8090)
#   model     any Ollama model name (default llama3.2:3b — small + fast)
#   question  the user message. If omitted, the whole shared query pool
#             (queries.txt) is run, one request per question, so a single run
#             exercises retrieval across several subsystems instead of the same
#             chunk every time. Pass a question to run only that one.
#
# Verification: a passing run prints each model answer AND the matching
# server-side "augmented with N fragments (T tokens)" line. A small model naming
# a project-specific symbol/path it could not otherwise know is the signal that
# retrieval + injection worked end to end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

PORT="${1:-8090}"
MODEL="${2:-llama3.2:3b}"
LOG="$ROOT/target/caw-dev/server-$PORT.log"

# Questions: the explicit 3rd arg overrides the pool; otherwise read queries.txt
# (skip blanks and '#' comments).
QUESTIONS=()
if [ "${3-}" != "" ]; then
  QUESTIONS=("$3")
else
  while IFS= read -r line; do
    [ -n "$line" ] && [ "${line#\#}" = "$line" ] && QUESTIONS+=("$line")
  done < "$SCRIPT_DIR/queries.txt"
fi

echo "smoke: POST http://localhost:$PORT/v1/chat/completions  model=$MODEL  questions=${#QUESTIONS[@]}" >&2

ask_one() {
  local question="$1"
  local req resp
  req=$(printf '{"model": %s, "stream": false, "messages": [{"role": "user", "content": %s}]}' \
    "$(printf '%s' "$MODEL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
    "$(printf '%s' "$question" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")
  resp=$(curl -s --max-time 240 "http://localhost:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d "$req")

  echo "===== Q: $question"
  echo "----- model answer -----"
  printf '%s' "$resp" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception as e:
    print("could not parse response:", e)' 2>/dev/null \
    || { echo "raw response:"; printf '%s\n' "$resp"; }

  echo "----- injection proof (server log) -----"
  if [ -f "$LOG" ]; then
    grep "augmented with" "$LOG" | tail -1 || echo "NO augmentation line found — retrieval returned nothing or query had no user message."
  else
    echo "server log not found at $LOG (is the server running? did serve.sh start it on this port?)"
  fi
  echo
}

for q in "${QUESTIONS[@]}"; do
  ask_one "$q"
done
