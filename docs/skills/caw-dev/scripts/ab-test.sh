#!/usr/bin/env bash
# A/B value proof: send the SAME question to the proxy (context injected) and
# straight to the upstream (no context), and show both answers side by side.
# This is the test that actually demonstrates the proxy adds value — a small
# model that names project-specific symbols/paths only via the proxy is the
# signal that retrieval changed the answer, not just that it answered.
#
# Usage: ab-test.sh [proxy-port] [upstream-base] [model] [question]
#   proxy-port      default 8080
#   upstream-base   default http://localhost:11434/v1 (must match serve.sh's upstream)
#   model           default llama3.2:3b
#   question        default: a codebase-specific question
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

PORT="${1:-8080}"
UPSTREAM="${2:-http://localhost:11434/v1}"
MODEL="${3:-llama3.2:3b}"
QUESTION="${4:-In this Rust project, which struct owns the multi-pass recall loop, what file is it in, and name two things it does on eviction?}"

ask() {
  local base="$1"
  local q_json
  q_json=$(printf '%s' "$QUESTION" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
  curl -s --max-time 240 "$base/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\": \"$MODEL\", \"stream\": false, \"messages\": [{\"role\": \"user\", \"content\": $q_json}]}" \
  | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception as e:
    print("(could not parse response:", e, ")")'
}

echo "Q: $QUESTION"
echo
echo "================ B: DIRECT upstream (no context) ================"
ask "$UPSTREAM"
echo
echo "================ A: via PROXY (context injected) ================"
ask "http://localhost:$PORT/v1"
echo
echo "================ injection proof (server log) ================"
LOG="$ROOT/target/caw-dev/server-$PORT.log"
[ -f "$LOG" ] && grep "augmented with" "$LOG" | tail -1 || echo "(no server log at $LOG)"
