#!/usr/bin/env python3
"""
Curate opencaw-corpora/subset-medium: a haystack for the chunk-level sysdoc
retrieval eval. It is the union of every document the golden set references plus
a seeded uniform sample of other sysdoc docs as distractors, copied with their
relative paths preserved so the QA's `expected` paths resolve under the new root.

Indexing all ~27k sysdoc docs would take hours; a few-thousand-doc subset indexes
in minutes while still forcing retrieval to discriminate among many candidates.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="opencaw-corpora/sysdoc/manifest.jsonl")
    ap.add_argument("--qa", default="crates/caw-bench/src/qa/sysdoc_chunk_qa.json")
    ap.add_argument("--src-root", default="opencaw-corpora/sysdoc")
    ap.add_argument("--dest", default="opencaw-corpora/subset-medium")
    ap.add_argument("--distractors", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    referenced = {
        q["expected"][0]["path"]
        for q in json.loads(Path(args.qa).read_text())["questions"]
    }
    all_rels = [json.loads(line)["rel"] for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    pool = sorted(set(all_rels) - referenced)

    rng = random.Random(args.seed)
    distractors = rng.sample(pool, min(args.distractors, len(pool)))
    keep = sorted(referenced | set(distractors))

    src_root = Path(args.src_root)
    dest = Path(args.dest)
    copied = missing = 0
    for rel in keep:
        src = src_root / rel
        if not src.is_file():
            missing += 1
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied += 1

    print(
        f"subset-medium: {copied} docs copied to {dest} "
        f"({len(referenced)} referenced + {len(distractors)} distractors; {missing} missing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
