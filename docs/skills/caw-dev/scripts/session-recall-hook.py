#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook: inject recalled prior-session content.

Reads the hook's stdin JSON, sends the user's prompt to a running caw-server's
read-only `/v1/retrieve` route, and emits the admitted fragments back to Claude
Code as `additionalContext`. The fragments are the same text the proxy would
have spliced into the model request — here they ride in through the hook channel
instead, so this works regardless of how Claude Code authenticates (it never
touches the model stream).

This is a single-shot retrieval per prompt, not OpenCAW's trace-driven
load/evict/consolidate loop. It is the "augment each prompt with relevant prior
sessions" integration, not the workspace loop. See docs/origin.md for the
distinction; an MCP recall tool is the trace-driven counterpart.

Prerequisite: a caw-server serving an index built from session docs (see
extract-sessions.py -> build-index.sh -> serve.sh). The retrieve endpoint is
the diagnostic route, so the proxy's --upstream does not matter for this hook.

Fail-open contract: any error (server down, malformed response, no admitted
fragments) exits 0 with no stdout, so a prompt is never blocked or polluted by
recall machinery. Set CAW_HOOK_DEBUG=1 to surface diagnostics on stderr (shown
to the developer without affecting the injected context).

Config via environment:
  CAW_RETRIEVE_URL   retrieve endpoint (default http://localhost:8090/v1/retrieve)
  CAW_HOOK_TIMEOUT   request timeout in seconds (default 10)
  CAW_HOOK_DEBUG     if set, write diagnostics to stderr
"""
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_RETRIEVE_URL = "http://localhost:8090/v1/retrieve"
DEFAULT_TIMEOUT_SECS = 10
# Prompts shorter than this carry too little signal to retrieve against; skip
# them rather than inject noise on a one-word "yes" / "continue".
MIN_PROMPT_CHARS = 8
ADMITTED = "admitted"


def debug(msg):
    if os.environ.get("CAW_HOOK_DEBUG"):
        print(f"session-recall-hook: {msg}", file=sys.stderr)


def fetch_candidates(url, query, timeout):
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_context(payload):
    """Render admitted fragments as a single additionalContext block, or None."""
    admitted = [
        c
        for c in payload.get("candidates", [])
        if c.get("disposition") == ADMITTED and c.get("content")
    ]
    if not admitted:
        return None

    parts = [
        "Relevant context recalled from prior Claude Code sessions "
        "(retrieved by OpenCAW; may or may not bear on the current request):",
        "",
    ]
    for c in admitted:
        parts.append(f"--- from {c['path']} (score {c['score']:.3f}) ---")
        parts.append(c["content"].strip())
        parts.append("")
    return "\n".join(parts)


def emit(additional_context):
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": additional_context,
                }
            }
        )
    )


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        debug(f"could not parse hook stdin as JSON: {e}")
        sys.exit(0)

    prompt = (hook_input.get("prompt") or "").strip()
    if len(prompt) < MIN_PROMPT_CHARS:
        debug(f"prompt too short ({len(prompt)} chars); skipping recall")
        sys.exit(0)

    url = os.environ.get("CAW_RETRIEVE_URL", DEFAULT_RETRIEVE_URL)
    try:
        timeout = float(os.environ.get("CAW_HOOK_TIMEOUT", DEFAULT_TIMEOUT_SECS))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECS

    try:
        payload = fetch_candidates(url, prompt, timeout)
    except urllib.error.URLError as e:
        debug(f"retrieve request to {url} failed ({e}); injecting nothing")
        sys.exit(0)
    except (json.JSONDecodeError, ValueError) as e:
        debug(f"retrieve returned non-JSON ({e}); injecting nothing")
        sys.exit(0)

    context = format_context(payload)
    if context is None:
        debug(
            f"no admitted fragments for prompt "
            f"({payload.get('candidate_count', 0)} candidates scored)"
        )
        sys.exit(0)

    debug(
        f"injecting {payload.get('admitted_count')} fragments "
        f"({payload.get('admitted_tokens')} tokens)"
    )
    emit(context)


if __name__ == "__main__":
    main()
