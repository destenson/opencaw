#!/usr/bin/env python3
"""
Generate a Q&A workload for caw-bench from a snapshotted doc corpus.

- Reads `<corpus>/manifest.jsonl` produced by snapshot-corpus.sh.
- Filters entries whose source-file mtime is ≥ --since (YYYY-MM-DD).
- Samples files weighted toward Debian-packaging and NEWS/CHANGELOG names,
  which are the content most likely to be post-training-cutoff.
- For each sampled doc, calls the `claude` CLI with a generator prompt and
  asks for ONE Q&A pair anchored by a verbatim quote.
- Rejects outputs whose quote isn't actually in the document (grounding
  check — the LLM hallucinated instead of answering from source).
- Writes a JSON list matching caw-bench's opencaw-workload schema.
- Checkpoints after every successful Q&A so a crash (API error, killed
  terminal) resumes without losing work. Re-run the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROMPT_TEMPLATE = """\
You will read a documentation file and generate ONE question-answer pair \
that could be used to evaluate a retrieval-augmented system.

DOCUMENT (source: {source}):
---
{content}
---

Requirements:
- The question must be answerable ONLY from this document: specific facts, \
version numbers, option names, Debian-specific behavior, precise commands, \
file paths, or configuration details.
- The question must not contain its own answer.
- Avoid generic questions ("what does this package do", "what is the title").
- The reference answer must be a short phrase or single sentence, directly \
supported by the document.
- Output ONLY a JSON object. No markdown fences, no commentary. Schema:
{{"question": "...", "reference_answer": "...", "quote": "<verbatim quote from the document that grounds the answer>"}}
"""


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, type=Path,
                    help="directory containing manifest.jsonl from snapshot-corpus.sh")
    ap.add_argument("--out", required=True, type=Path,
                    help="output qa.json path")
    ap.add_argument("--since", default="2025-06-01",
                    help="mtime cutoff, YYYY-MM-DD (default: 2025-06-01)")
    ap.add_argument("--count", type=int, default=100,
                    help="target number of Q&A pairs (default: 100)")
    ap.add_argument("--model", default="haiku",
                    help="claude model alias (default: haiku)")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed (default: 42)")
    ap.add_argument("--max-bytes", type=int, default=30000,
                    help="max doc bytes sent to the LLM (default: 30000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be sampled and exit")
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-call claude timeout seconds (default: 180)")
    return ap.parse_args()


def load_manifest(corpus_dir: Path) -> list[dict]:
    manifest = corpus_dir / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"manifest not found: {manifest}")
    out = []
    for line in manifest.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def weight_for(entry: dict) -> int:
    """Bias toward content most likely to be post-training-cutoff or
    installation-specific. Debian-packaging files especially — these are
    written by packagers per-release and rarely scraped."""
    base = os.path.basename(entry["source"])
    if "Debian" in base:
        return 3
    if base.startswith(("NEWS", "CHANGELOG", "changelog", "ChangeLog")):
        return 2
    return 1


def filter_eligible(entries: list[dict], cutoff_ts: float) -> list[dict]:
    out = []
    missing = 0
    too_old = 0
    for e in entries:
        try:
            mtime = os.path.getmtime(e["source"])
        except OSError:
            missing += 1
            continue
        if mtime < cutoff_ts:
            too_old += 1
            continue
        e["mtime"] = mtime
        e["weight"] = weight_for(e)
        out.append(e)
    print(f"  {missing} source paths missing (pruned/uninstalled since snapshot)", file=sys.stderr)
    print(f"  {too_old} filtered by mtime cutoff", file=sys.stderr)
    return out


def sample_weighted(entries: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Weighted sample without replacement: inflate by weight then shuffle
    and take unique rels. Cheap and adequate for our pool sizes."""
    pool = []
    for e in entries:
        pool.extend([e] * e["weight"])
    rng.shuffle(pool)
    seen = set()
    out = []
    for e in pool:
        if e["rel"] in seen:
            continue
        seen.add(e["rel"])
        out.append(e)
        if len(out) >= count:
            break
    return out


def run_claude(model: str, prompt: str, timeout: int) -> tuple[str | None, str | None]:
    # No --bare: we want to use the user's existing keychain auth.
    try:
        r = subprocess.run(
            ["claude", "-p", "--model", model, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    except FileNotFoundError:
        return None, "claude CLI not found on PATH"
    if r.returncode != 0:
        # claude writes auth/login errors to stdout, API errors to stderr.
        # Show whichever is non-empty.
        msg = r.stderr.strip() or r.stdout.strip() or "(no output)"
        return None, f"claude exit {r.returncode}: {msg[:400]}"
    return r.stdout.strip(), None


def parse_response(raw: str) -> tuple[dict | None, str | None]:
    # Extract the outermost JSON object even if the model wrapped it in
    # prose or markdown fences despite being told not to.
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None, "no JSON object in response"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"json parse: {e}"
    for k in ("question", "reference_answer", "quote"):
        v = obj.get(k)
        if not isinstance(v, str) or not v.strip():
            return None, f"missing/empty field: {k}"
    return obj, None


def quote_is_grounded(content: str, quote: str) -> bool:
    """Loose containment check — normalize whitespace and compare
    case-insensitive. Rejects hallucinated quotes, tolerates LLM paraphrasing
    of punctuation and line breaks."""
    norm_content = re.sub(r"\s+", " ", content).lower()
    norm_quote = re.sub(r"\s+", " ", quote).lower().strip()
    if len(norm_quote) < 10:
        return False
    return norm_quote in norm_content


def slugify(s: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:limit]


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    corpus = args.corpus.resolve()
    out_path = args.out.resolve()

    try:
        cutoff = datetime.fromisoformat(args.since).timestamp()
    except ValueError as e:
        sys.exit(f"--since must be YYYY-MM-DD: {e}")

    entries = load_manifest(corpus)
    print(f"{len(entries)} manifest entries loaded from {corpus}", file=sys.stderr)

    eligible = filter_eligible(entries, cutoff)
    print(f"{len(eligible)} eligible (mtime ≥ {args.since})", file=sys.stderr)
    if not eligible:
        sys.exit("no eligible entries — loosen --since or check the snapshot")

    # Oversample 2× because some calls will fail or get rejected for
    # ungrounded quotes. Capped at pool size.
    pool_size = min(args.count * 2, len(eligible))
    pool = sample_weighted(eligible, pool_size, rng)
    print(f"sampled pool: {len(pool)} docs", file=sys.stderr)

    if args.dry_run:
        print("\n--- dry run, first 15 samples ---", file=sys.stderr)
        for e in pool[:15]:
            dt = datetime.fromtimestamp(e["mtime"]).date()
            print(f"  w={e['weight']} mtime={dt} {e['rel']}", file=sys.stderr)
        return

    # Resume: reload any existing results so we skip files already processed.
    results: list[dict] = []
    processed_rels: set[str] = set()
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            processed_rels = {r.get("_source_rel") for r in results if r.get("_source_rel")}
            print(f"resume: {len(results)} existing Q&A in {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"resume failed ({e}); starting fresh", file=sys.stderr)
            results = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    attempts = 0
    for entry in pool:
        if len(results) >= args.count:
            break
        if entry["rel"] in processed_rels:
            continue
        attempts += 1
        doc_path = corpus / entry["rel"]
        try:
            raw_bytes = doc_path.read_bytes()[:args.max_bytes]
            content = raw_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  read fail {entry['rel']}: {e}", file=sys.stderr)
            continue
        if len(content.strip()) < 200:
            # Too short to yield a meaningful question.
            continue

        prompt = PROMPT_TEMPLATE.format(source=entry["source"], content=content)
        print(f"[{len(results)+1}/{args.count}] {entry['rel']}", file=sys.stderr)
        t0 = time.time()
        raw, err = run_claude(args.model, prompt, args.timeout)
        dt = time.time() - t0
        if err:
            print(f"  claude error ({dt:.0f}s): {err}", file=sys.stderr)
            continue
        obj, perr = parse_response(raw)
        if perr:
            print(f"  parse error ({dt:.0f}s): {perr}", file=sys.stderr)
            continue
        if not quote_is_grounded(content, obj["quote"]):
            print(f"  quote not grounded ({dt:.0f}s) — skip", file=sys.stderr)
            continue

        qa_id = f"sysdoc_{len(results)+1:03d}_{slugify(entry['rel'])}"
        results.append({
            "id": qa_id,
            "question": obj["question"].strip(),
            "reference_answer": obj["reference_answer"].strip(),
            "expected_paths": [entry["rel"]],
            # Private fields — ignored by caw-bench, useful for debugging.
            "_source_rel": entry["rel"],
            "_source_path": entry["source"],
            "_quote": obj["quote"].strip(),
        })
        # Checkpoint after every successful pair so a crash mid-sweep
        # doesn't lose accumulated work.
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  ok ({dt:.0f}s) — {len(results)}/{args.count}", file=sys.stderr)

    print(f"\n{len(results)} Q&A written to {out_path} after {attempts} attempts",
          file=sys.stderr)
    if len(results) < args.count:
        print(f"WARNING: target {args.count} not reached; re-run to continue "
              f"(resume will pick up from existing output)", file=sys.stderr)


if __name__ == "__main__":
    main()
