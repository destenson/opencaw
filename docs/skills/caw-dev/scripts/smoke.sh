#!/usr/bin/env bash
# Send one chat completion through the running proxy and verify that context
# was actually injected (not just that the model answered plausibly).
#
# Usage: smoke.sh [port] [model] [question]
#   port      proxy port (default 8080)
#   model     any Ollama model name (default llama3.2:3b — small + fast)
#   question  the user message (default: a codebase-specific question whose
#             answer requires the injected source)
#
# Verification: a passing run prints the model's answer AND the server-side
# "augmented with N fragments (T tokens)" line. A small model naming a
# project-specific symbol/path it could not otherwise know is the signal
# that retrieval + injection worked end to end.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

PORT="${1:-8080}"
MODEL="${2:-llama3.2:3b}"
QUESTION="${3:-In this Rust project, what does the DynamicRecallOrchestrator do and which source file defines it? Be specific.}"
LOG="$ROOT/target/caw-dev/server-$PORT.log"

echo "smoke: POST http://localhost:$PORT/v1/chat/completions  model=$MODEL" >&2

REQ=$(cat <<JSON
{"model": "$MODEL", "stream": false, "messages": [{"role": "user", "content": $(printf '%s' "$QUESTION" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}]}
JSON
)

RESP=$(curl -s --max-time 240 "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' -d "$REQ")

echo "===== model answer ====="
printf '%s' "$RESP" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception as e:
    print("could not parse response:", e); print(sys.stdin.read() if False else "")' 2>/dev/null \
  || { echo "raw response:"; printf '%s\n' "$RESP"; }

echo "===== injection proof (server log) ====="
if [ -f "$LOG" ]; then
  grep "augmented with" "$LOG" | tail -1 || echo "NO augmentation line found — retrieval returned nothing or query had no user message."
else
  echo "server log not found at $LOG (is the server running? did serve.sh start it on this port?)"
fi
