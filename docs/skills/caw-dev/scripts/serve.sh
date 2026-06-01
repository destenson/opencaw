#!/usr/bin/env bash
# Start the caw-server drop-in OpenAI proxy in the background, pinned to the
# freest GPU, with debug logging so context injection is verifiable.
#
# Usage: serve.sh <index.sqlite> <corpus-root> [upstream] [port] [max-tokens] [retriever]
#   index.sqlite   index built by build-index.sh
#   corpus-root    MUST match the --corpus passed to build-index.sh
#   upstream       OpenAI-compatible base URL (default Ollama: http://localhost:11434/v1)
#   port           listen port (default 8090)
#   max-tokens     hard cap on injected context (default 2000)
#   retriever      flat | hnsw | hybrid (default hybrid). hybrid fuses BM25
#                  lexical scores with cosine and pays a one-time startup cost
#                  to read every body; flat is pure cosine.
#
# Runtime state (pid + log) lives in $ROOT/target/caw-dev/ — the standard
# Rust throwaway dir, already gitignored and cleaned by `cargo clean`, so no
# new clutter lands in the repo. The debug log prints one line per request:
#   "augmented with N fragments (T tokens)"  <- proof injection happened.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INDEX="${1:?usage: serve.sh <index.sqlite> <corpus-root> [upstream] [port] [max-tokens]}"
CORPUS_ROOT="${2:?usage: serve.sh <index.sqlite> <corpus-root> [upstream] [port] [max-tokens]}"
UPSTREAM="${3:-http://localhost:11434/v1}"
PORT="${4:-8090}"
MAX_TOKENS="${5:-2000}"
RETRIEVER="${6:-hybrid}"

STATE="$ROOT/target/caw-dev"
mkdir -p "$STATE"
LOG="$STATE/server-$PORT.log"
PIDFILE="$STATE/server-$PORT.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "serve: a server is already running on port $PORT (pid $(cat "$PIDFILE")). Run stop.sh $PORT first." >&2
  exit 1
fi

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
echo "serve: GPU='${GPU:-cpu}' index='$INDEX' corpus-root='$CORPUS_ROOT' upstream='$UPSTREAM' port=$PORT retriever='$RETRIEVER'" >&2

cd "$ROOT"
RUST_LOG=caw_server=debug,info CUDA_VISIBLE_DEVICES="$GPU" \
  nohup cargo run --release --quiet -p caw-server --bin caw-server -- \
    --index "$INDEX" \
    --corpus-root "$CORPUS_ROOT" \
    --upstream "$UPSTREAM" \
    --port "$PORT" \
    --max-workspace-tokens "$MAX_TOKENS" \
    --retriever "$RETRIEVER" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"

# Wait for readiness (compile + candle init + index load can take a minute).
for _ in $(seq 1 90); do
  if grep -q "listening on" "$LOG" 2>/dev/null; then
    echo "serve: ready on http://localhost:$PORT/v1  (log: $LOG)"
    grep "caw-server ready" "$LOG" | tail -1
    exit 0
  fi
  if grep -qE "error\[|Error:|panicked" "$LOG" 2>/dev/null; then
    echo "serve: startup failed — last log lines:" >&2
    tail -15 "$LOG" >&2
    exit 1
  fi
  sleep 2
done
echo "serve: timed out waiting for readiness — check $LOG" >&2
exit 1
