#!/usr/bin/env python3
"""Flatten Claude Code session transcripts (JSONL) into readable per-session
text documents suitable for indexing as an OpenCAW corpus.

This is a FORMAT transform, not a content filter. The transcripts store each
turn as a JSON envelope; indexing them raw would chunk on JSON structure, so a
fragment that matches on conversation prose comes back mostly wrapper text and
wastes the injection budget. Here we emit one Markdown doc per session with
turns on conversation boundaries, so retrieved/injected fragments are prose.

What is emitted: user messages, assistant text, and assistant thinking. What is
skipped: entries with no conversational prose (mode / permission-mode /
file-history-snapshot / attachment / ai-title / last-prompt / system) and, for
now, tool_use / tool_result blocks (the bulk of the bytes and the least
discussion-like; widening to include tool_result is the obvious next step).

Usage:
  extract-sessions.py <projects-dir> [out-dir]

  projects-dir   a ~/.claude/projects/<slug> directory of *.jsonl transcripts
  out-dir        directory to write <date>-<shortid>.md docs into (created).
                 Defaults to a fresh mkdtemp() dir, whose path is printed so it
                 can be passed straight to build-index.sh. Note: the corpus must
                 NOT sit under target/ or any hidden/scripts/node_modules path —
                 build_index's should_skip drops those silently (0 stubs).

                 Pass a STABLE out-dir to get incremental indexing across runs:
                 filenames are deterministic per session and each doc is stamped
                 with its source transcript's mtime, so re-extracting into the
                 same dir leaves unchanged sessions byte- and mtime-identical and
                 the indexer skips re-embedding them. The mkdtemp default is for
                 one-shot extraction; it churns a new dir + paths every run.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Block types within a message.content list that carry conversational prose.
ASSISTANT_PROSE_BLOCKS = ("text", "thinking")


def block_text(block):
    """Return the prose carried by a content block, or None if it carries none."""
    if not isinstance(block, dict):
        return None
    bt = block.get("type")
    if bt in ("text", "thinking"):
        text = block.get("text") or block.get("thinking")
        return text.strip() if isinstance(text, str) and text.strip() else None
    return None


def message_turns(entry):
    """Yield (label, text) turns for one transcript entry, or nothing."""
    role = entry.get("type")
    if role not in ("user", "assistant"):
        return
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return
    content = msg.get("content")

    if isinstance(content, str):
        text = content.strip()
        if text:
            yield ("User" if role == "user" else "Assistant", text)
        return

    if isinstance(content, list):
        for block in content:
            text = block_text(block)
            if text is None:
                continue
            bt = block.get("type")
            if role == "user":
                yield ("User", text)
            elif bt == "thinking":
                yield ("Assistant (thinking)", text)
            else:
                yield ("Assistant", text)


def extract_session(jsonl_path):
    """Parse one transcript file -> (metadata dict, list of (label, text) turns)."""
    meta = {"session": jsonl_path.stem, "date": None, "branch": None}
    turns = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if meta["date"] is None and entry.get("timestamp"):
                meta["date"] = entry["timestamp"][:10]
            if meta["branch"] is None and entry.get("gitBranch"):
                meta["branch"] = entry["gitBranch"]
            turns.extend(message_turns(entry))
    return meta, turns


def render(meta, turns):
    lines = [
        f"# Session {meta['session']}",
        "",
        f"- date: {meta['date'] or 'unknown'}",
        f"- branch: {meta['branch'] or 'unknown'}",
        "",
    ]
    for label, text in turns:
        lines.append(f"## {label}")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    projects_dir = Path(sys.argv[1]).expanduser()
    if len(sys.argv) == 3:
        out_dir = Path(sys.argv[2]).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="caw-sessions-"))

    transcripts = sorted(projects_dir.glob("*.jsonl"))
    if not transcripts:
        sys.exit(f"no *.jsonl transcripts in {projects_dir}")

    written = 0
    skipped_empty = 0
    for path in transcripts:
        meta, turns = extract_session(path)
        if not turns:
            skipped_empty += 1
            continue
        name = f"{meta['date'] or '0000-00-00'}-{meta['session'][:8]}.md"
        out_path = out_dir / name
        out_path.write_text(render(meta, turns))
        # Stamp the extracted doc with the SOURCE transcript's mtime, not the
        # (just-now) write time. The corpus the indexer sees is this derived
        # file, but the thing that actually changes is the source .jsonl. The
        # indexer skips files whose (path, mtime) it has already embedded, so
        # mirroring the source mtime here is what makes re-extraction idempotent:
        # re-run into the same out-dir and a session whose transcript hasn't
        # grown keeps its mtime, gets skipped, and is not re-embedded. (The
        # filename is already deterministic per session, so a stable out-dir
        # gives stable paths.) Without this, every extraction stamps "now" and
        # the indexer re-embeds the whole corpus each time.
        src_mtime = path.stat().st_mtime
        os.utime(out_path, (src_mtime, src_mtime))
        written += 1

    print(f"extract-sessions: {written} docs written to {out_dir} "
          f"({skipped_empty} transcripts had no conversational turns)")


if __name__ == "__main__":
    main()
