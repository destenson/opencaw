#!/usr/bin/env bash
# Snapshot the repo at the code-agent QA-authoring commit into the frozen
# bench corpus.
#
# The code-agent QA items (crates/caw-bench/src/qa/codeagent_qa.json) ask the
# model to recall named symbols — a field, a variant, a method signature,
# a trait bound — by exact token. Their ground-truth answers must come from a
# checkout where those symbols exist. The QA file was authored at
# CODEAGENT_CORPUS_PIN, so pinning the corpus to that commit version-locks QA
# and corpus by construction: every needle is present in that tree by
# definition. (The QA file is in fact unchanged from that commit to HEAD, but
# pinning keeps the gauge reproducible as the live repo drifts — the bench is
# a frozen gauge, not a dogfood run. Dogfooding via caw-server still uses the
# live repo; freezing this corpus does not affect it.)
#
# Usage: freeze-codeagent-corpus.sh [output-dir]
#   output-dir  default: $ROOT/opencaw-corpora/codeagent-corpus (gitignored)
#
# Re-freezing: the script refuses to write into a non-empty directory so a
# stale tree from an old pin can't contaminate a new one. Remove the dir
# first (e.g. `rm -rf "$OUT"`) and re-run.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# Pin: the commit that added codeagent_qa.json. Change this only when the QA
# set is intentionally rev'd and the corpus must move with it — and then also
# confirm every needle in the new QA still resolves in the new pin's tree.
CODEAGENT_CORPUS_PIN="b0201a7"

OUT="${1:-$ROOT/opencaw-corpora/codeagent-corpus}"

if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "freeze: $OUT already exists and is non-empty." >&2
  echo "freeze: remove it first to re-freeze (e.g. rm -rf \"$OUT\"), then re-run." >&2
  exit 1
fi

mkdir -p "$OUT"
# git archive emits the tracked tree at the pin only — no .git, no target/
# (both untracked/gitignored), so the corpus is a clean checkout for
# opencaw::load_corpus.
git -C "$ROOT" archive "$CODEAGENT_CORPUS_PIN" | tar -x -C "$OUT"
echo "froze code-agent corpus at $CODEAGENT_CORPUS_PIN -> $OUT"