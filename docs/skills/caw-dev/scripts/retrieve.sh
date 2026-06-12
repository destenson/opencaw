#!/usr/bin/env bash
# Inspect what retrieval actually produces for a query, without sending it to a
# model. Hits the running proxy's read-only /v1/retrieve diagnostic route and
# prints the ranked candidate list: fused score, path, token cost, and
# disposition (admitted / clamped / budget_full / content_miss) for each.
#
# This answers the question smoke.sh can't: when a model answer looks wrong,
# was the relevant chunk ranked out of the pool, or ranked in but clamped out
# by the token budget? smoke.sh shows the answer; this shows the retrieval.
#
# Requires a server already running (serve.sh) — serve once, query many.
#
# Usage: retrieve.sh "<query>" [port] [--full]
#   query   the text to retrieve against (required)
#   port    proxy port (default 8090)
#   --full  also print the body of each admitted fragment (the exact text that
#           would be injected). Omitted by default to keep the ranking scannable.
set -euo pipefail

QUERY="${1:?usage: retrieve.sh \"<query>\" [port] [--full]}"
PORT="${2:-8090}"
FULL=""
[ "${3-}" = "--full" ] && FULL="1"

URL="http://localhost:$PORT/v1/retrieve"
req="$(printf '%s' "$QUERY" | python3 -c 'import json,sys; print(json.dumps({"query": sys.stdin.read()}))')"

resp="$(curl -s --max-time 120 "$URL" -H 'content-type: application/json' -d "$req")" || {
  echo "retrieve: request to $URL failed — is the server running? (serve.sh starts it)" >&2
  exit 1
}

# A non-JSON body means the route returned a plain-text error (e.g. 400/500).
if ! printf '%s' "$resp" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
  echo "retrieve: server did not return JSON (is /v1/retrieve available on this build?):" >&2
  printf '%s\n' "$resp" >&2
  exit 1
fi

FULL="$FULL" python3 - "$resp" <<'PY'
import json, os, sys

d = json.loads(sys.argv[1])
full = os.environ.get("FULL") == "1"

print(f"query:      {d['query']!r}")
print(f"retriever:  {d['retriever']}   max_candidates={d['max_candidates']}   "
      f"max_workspace_tokens={d['max_workspace_tokens']}")
print(f"admitted:   {d['admitted_count']}/{d['candidate_count']} candidates, "
      f"{d['admitted_tokens']} tokens injected")
print()
print(f"{'rank':>4}  {'score':>7}  {'tokens':>6}  {'disposition':<12}  path")
print("-" * 72)
for c in d["candidates"]:
    tok = "" if c["tokens"] is None else str(c["tokens"])
    print(f"{c['rank']:>4}  {c['score']:>7.4f}  {tok:>6}  {c['disposition']:<12}  {c['path']}")

if full:
    print()
    for c in d["candidates"]:
        if c["disposition"] == "admitted" and c["content"]:
            print(f"===== #{c['rank']}  {c['path']}  ({c['tokens']} tokens) =====")
            print(c["content"])
            print()
PY
