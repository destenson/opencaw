#!/usr/bin/env bash
# Watch the sweep log and emit an alert when a claude CLI call is hung.
# Detection: stat the log's mtime. If the file hasn't grown in STALL_SECS
# while a claude child process is alive, emit one alert per stall event.
set -u
LOG="${LOG:-/home/dennis/src/ai-experiments/opencaw/bench-results/default.log}"
STALL_SECS="${STALL_SECS:-32}"
POLL_SECS="${POLL_SECS:-15}"

alerted=0
while true; do
    if [[ ! -f "$LOG" ]]; then
        sleep "$POLL_SECS"; continue
    fi
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    age=$((now - mtime))
    claude_pid=$(pgrep -f "claude --print --output-format json" 2>/dev/null | head -1)
    if [[ -n "$claude_pid" && "$age" -gt "$STALL_SECS" && "$alerted" != "$claude_pid" ]]; then
        etime=$(ps -p "$claude_pid" -o etime= 2>/dev/null | tr -d ' ')
        echo "STALL: log idle ${age}s, claude PID $claude_pid alive ${etime} — kill -9 $claude_pid to unblock"
        alerted="$claude_pid"
    fi
    # Reset alert after the stuck process dies so a new stall for a new PID alerts again.
    if [[ -z "$claude_pid" ]]; then alerted=0; fi
    sleep "$POLL_SECS"
done
