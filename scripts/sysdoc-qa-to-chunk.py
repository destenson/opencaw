#!/usr/bin/env python3
"""
Convert the file-level sysdoc Q&A (opencaw-corpora/sysdoc_qa.json) into the
chunk-level golden schema caw-bench-graph-eval consumes.

The source set anchors every answer with a verbatim `_quote`. We resolve that
quote to a 1-indexed line in its source document, which graph_eval then maps to
the covering chunk-stub at measurement time. Matching is whitespace-normalized
and uses the longest quote line, because the generator's stored quote collapses
or reflows the document's original spacing (the naive exact match resolves only
~half; normalized matching resolves all of them).

Output schema (matches crates/caw-bench/src/qa/graph_eval_qa.json):
    {"questions": [{"id", "type", "question", "expected": [{"path", "line"}]}]}

`type` is derived from the document kind (changelog / copyright / readme / doc)
so graph_eval's per-type aggregates show whether retrieval favors one doc family.

Fails loudly (non-zero exit) if any quote cannot be resolved, rather than
silently dropping questions — a dropped gold is a silently weaker eval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces and lowercase."""
    return re.sub(r"\s+", " ", s).strip().lower()


def doc_type(rel: str) -> str:
    low = rel.lower()
    if "changelog" in low:
        return "changelog"
    if "copyright" in low:
        return "copyright"
    if "readme" in low:
        return "readme"
    if "news" in low:
        return "news"
    return "doc"


def normalize_with_linemap(source: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed lowercase text plus, per normalized char, the
    1-indexed source line it came from. Lets a match in the reflowed text map
    back to a line even when the stored quote joined several wrapped lines."""
    chars: list[str] = []
    line_of: list[int] = []
    line = 1
    prev_space = True  # collapse leading whitespace
    for ch in source:
        if ch.isspace():
            if not prev_space:
                chars.append(" ")
                line_of.append(line)
                prev_space = True
            if ch == "\n":
                line += 1
        else:
            chars.append(ch.lower())
            line_of.append(line)
            prev_space = False
    return "".join(chars), line_of


def resolve_line(source: str, quote: str) -> int | None:
    """1-indexed source line anchoring `quote`. Matches the longest quote line
    (then shorter ones, then the whole quote) against the reflow-normalized text
    and maps the match position back to a source line."""
    norm_src, line_of = normalize_with_linemap(source)
    quote_lines = [ln for ln in quote.split("\n") if ln.strip()]
    candidates = sorted(quote_lines, key=len, reverse=True)
    candidates.append(quote)  # whole-quote fallback (maps to its starting line)
    for target in candidates:
        nt = normalize(target)
        if not nt:
            continue
        pos = norm_src.find(nt)
        if pos != -1:
            return line_of[pos]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="opencaw-corpora/sysdoc_qa.json")
    ap.add_argument("--corpus-root", default="opencaw-corpora/sysdoc")
    ap.add_argument("--out", default="crates/caw-bench/src/qa/sysdoc_chunk_qa.json")
    args = ap.parse_args()

    qa = json.loads(Path(args.qa).read_text())
    root = Path(args.corpus_root)

    questions = []
    unresolved = []
    for e in qa:
        rel = e["_source_rel"]
        src_path = root / rel
        if not src_path.exists():
            unresolved.append((e["id"], f"source missing: {src_path}"))
            continue
        source = src_path.read_text(errors="replace")
        line = resolve_line(source, e["_quote"])
        if line is None:
            unresolved.append((e["id"], "quote did not resolve to a line"))
            continue
        questions.append(
            {
                "id": e["id"],
                "type": doc_type(rel),
                "question": e["question"],
                "expected": [{"path": rel, "line": line}],
            }
        )

    if unresolved:
        print(f"ERROR: {len(unresolved)} of {len(qa)} questions did not resolve:", file=sys.stderr)
        for qid, why in unresolved:
            print(f"  {qid}: {why}", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps({"questions": questions}, indent=2) + "\n")
    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    print(f"wrote {len(questions)} chunk-level questions to {args.out}")
    print("by type:", dict(sorted(by_type.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
