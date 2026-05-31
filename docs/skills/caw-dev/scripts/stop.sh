#!/usr/bin/env bash
# Stop a caw-server started by serve.sh.
#
# Usage: stop.sh [port]   (default port 8090)
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PORT="${1:-8090}"
PIDFILE="$ROOT/target/caw-dev/server-$PORT.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    # The cargo-run wrapper spawns the actual server as a child; kill the group.
    pkill -P "$PID" 2>/dev/null || true
    kill "$PID" 2>/dev/null || true
    echo "stop: terminated pid $PID (port $PORT)"
  fi
  rm -f "$PIDFILE"
fi
# Belt and suspenders: catch a server bound to this port whose pidfile was lost.
pkill -f "caw-server --index.*--port $PORT" 2>/dev/null || true
echo "stop: done (port $PORT)"
