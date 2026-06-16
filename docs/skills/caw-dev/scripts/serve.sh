#!/usr/bin/env bash
# Start the caw-server drop-in OpenAI proxy in the background, pinned to the
# freest GPU, with debug logging so context injection is verifiable.
#
# Usage: serve.sh <corpus-root> [upstream] [port] [max-tokens] [retriever] [--index PATH]
#   corpus-root    directory whose files form the corpus (single root). The
#                  index is derived from this and built on demand — there is no
#                  separate build-index.sh step. Stub paths are stored relative
#                  to this dir, so the server resolves bodies against it too.
#   upstream       OpenAI-compatible base URL (default Ollama: http://localhost:11434/v1)
#   port           listen port (default 8090)
#   max-tokens     hard cap on injected context (default 2000)
#   retriever      flat | hnsw | hybrid (default hybrid). hybrid fuses BM25
#                  lexical scores with cosine and pays a one-time startup cost
#                  to read every body; flat is pure cosine.
#   --index PATH   override the derived index path. Default is
#                  $ROOT/target/caw-dev/<corpus-base>-<hash>.sqlite, keyed by the
#                  corpus's ABSOLUTE path so the same corpus maps to the same
#                  index however it is spelled on the command line.
#
# On startup the index is brought up to date before the server launches: a
# missing index is built in full; an existing one is refreshed incrementally
# (only new/changed files, by (path, mtime), are re-embedded). This costs one
# embedder init even when nothing changed — the price of guaranteed freshness
# without a separate build step. Caveat (inherited from incremental build): a
# file deleted from the corpus leaves orphan stubs, and shifted chunk
# boundaries can too; run build-index.sh with a full --rebuild when the corpus
# has shrunk or chunking config changed.
#
# Index path derivation deliberately stays in-tree under target/caw-dev/ (the
# gitignored Rust throwaway, cleaned by `cargo clean`): a skill script must
# never write to the user's home dir, so $HOME/.cache and friends are out.
#
# Runtime state (pid + log) lives in $ROOT/target/caw-dev/ too. The debug log
# prints one line per request:
#   "augmented with N fragments (T tokens)"  <- proof injection happened.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USAGE="usage: serve.sh <corpus-root> [upstream] [port] [max-tokens] [retriever] [--index PATH]"

INDEX_OVERRIDE=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --index) INDEX_OVERRIDE="${2:?--index needs a path}"; shift 2 ;;
    --index=*) INDEX_OVERRIDE="${1#*=}"; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

CORPUS_ROOT="${1:?$USAGE}"
UPSTREAM="${2:-http://localhost:11434/v1}"
PORT="${3:-8090}"
MAX_TOKENS="${4:-2000}"
RETRIEVER="${5:-hybrid}"

STATE="$ROOT/target/caw-dev"
mkdir -p "$STATE"
LOG="$STATE/server-$PORT.log"
PIDFILE="$STATE/server-$PORT.pid"

# Resolve the corpus to an absolute path: the derived index key must be stable
# whether the corpus was given relative or absolute, and build + serve must
# agree on the same root for stub paths to resolve.
CORPUS_ABS="$(realpath "$CORPUS_ROOT" 2>/dev/null)" || {
  echo "serve: corpus root '$CORPUS_ROOT' does not exist" >&2
  exit 1
}

if [ -n "$INDEX_OVERRIDE" ]; then
  INDEX="$INDEX_OVERRIDE"
else
  # Canonical in-tree index: <corpus-base>-<short-hash>.sqlite. The hash of the
  # absolute path keys it per-corpus and stops two corpora that share a
  # basename from colliding onto one index.
  # printf '%s' (no trailing newline) so tr doesn't turn the newline into a
  # stray trailing '-' in the filename.
  CORPUS_BASE="$(printf '%s' "$(basename "$CORPUS_ABS")" | tr -c 'A-Za-z0-9._-' '-')"
  CORPUS_HASH="$(printf '%s' "$CORPUS_ABS" | sha256sum | cut -c1-12)"
  INDEX="$STATE/${CORPUS_BASE}-${CORPUS_HASH}.sqlite"
fi

# Bring the index up to date before serving: missing -> full build, present ->
# incremental refresh. build-index.sh owns GPU pinning and is synchronous, so
# the server only launches once the index is current.
if [ ! -f "$INDEX" ]; then
  echo "serve: no index at $INDEX — building from $CORPUS_ABS" >&2
  bash "$SCRIPT_DIR/build-index.sh" "$CORPUS_ABS" "$INDEX"
else
  echo "serve: refreshing $INDEX incrementally from $CORPUS_ABS" >&2
  bash "$SCRIPT_DIR/build-index.sh" --no-rebuild "$CORPUS_ABS" "$INDEX"
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "serve: a server is already running on port $PORT (pid $(cat "$PIDFILE")). Run stop.sh $PORT first." >&2
  exit 1
fi

GPU="$(bash "$SCRIPT_DIR/pick-gpu.sh")"
echo "serve: GPU='${GPU:-cpu}' index='$INDEX' corpus-root='$CORPUS_ABS' upstream='$UPSTREAM' port=$PORT retriever='$RETRIEVER'" >&2

cd "$ROOT"
RUST_LOG=caw_server=debug,info CUDA_VISIBLE_DEVICES="$GPU" \
  nohup cargo run --release --quiet -p caw-server --bin caw-server -- \
    --index "$INDEX" \
    --corpus-root "$CORPUS_ABS" \
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
