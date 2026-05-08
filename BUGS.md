# A place to record bugs as they're found

## B0. qa/0001.txt sometimes misses the first user turn (critical)

`.caw0008/session-20260508-055328.md` shows the first user turn "what is opencaw
and why should I use it?" is missing from the session log, even though the
corresponding prompt file `-turn-1.txt` contains that turn. Turn 2 contains the
very next user query, so the first turn is effectively lost. This is a major
issue.

## B1. Multi-pass recall corrupts fragment source paths (critical)

During iterative thinking-trace recall, the `[recalled from ...]` locator field
gets populated with embedded code content instead of the file path. Observed in
`.caw0003/prompt-20260508-050942-turn-8.txt`: entries like
`[recalled from max_recall_iterations = 3,` where code content replaces the
source path. The same prompt also contains `struct S: BudgetScheduler,` as
standalone content — invalid Rust — indicating the fragment boundary detection
is splitting fragments mid-struct-definition and the prefix string from one
fragment is being prepended to adjacent content. Fragments from
`crates/caw-orchestrator/src/lib.rs` are recalled 30+ times in one prompt with
no deduplication, each a slightly different truncated window into the same code.

## B2. Session markdown log truncates after turn 3 (high)

Session log files (`session-20260508-050942.md`, `session-20260508-051313.md`)
record only the first 3 user turns even though the corresponding prompt files
(`-turn-1.txt` through `-turn-18.txt`) show 9+ user turns were processed. The
log writer either exits early or flushes only on process exit (and is killed
before it can flush). QA analysis relying on session `.md` files misses most of
each session's output.

## B3. `.caw/` session log skip filter not working at index build time (medium)

Despite `should_skip` in `caw-ingest/src/lib.rs` filtering paths with a `.caw`
component, session log files from `.caw/session-*.md` appear in the retrieval
index for 0003 (same finding as 0002). The CLI ingestion path or the bench index
builder is apparently not activating the skip correctly — either
`skip_gitignore` is overriding the hidden-dir walk suppression, or `should_skip`
is not being called on the walk results before insertion.

Partially fixed: `build_index.rs::should_skip` now checks all path components
for hidden dirs (`.caw*`) instead of only the filename. The CLI-path issue
(session logs inserted via `SessionFile::load_previous` into the SQLite store)
remains open — this requires a transient insert path that adds only to the
in-memory HNSW index, not the persistent store.

QA 0008 confirms the CLI path is still broken with a concrete case: session
055659 (started 05:56:59) retrieved session 055445 (ended ~05:54:45) as a
top-ranked fragment within 75 seconds of the prior session ending. The model
cited that session's improvised answer to "what can we do to improve
caw-curation?" as authoritative project documentation, attributing B3 as a
curation bug and inventing curation recommendations from the prior session's
output. This is the clearest demonstration of the propagation path: session A
improvises → session A indexed → session B retrieves session A → session A's
improvisation becomes session B's grounded recommendation.

## B5. `scripts/qa.sh` indexed and surfaced as authoritative architecture documentation (medium) — FIXED

`scripts/qa.sh` chunk 6/7 appears as a top-ranked retrieval result for nearly
every query across all three QA loop 0005 sessions — architecture queries,
status queries, metacommentary queries. The chunk contains a CODEBASE LAYOUT
section and QA prompt templates that the retrieval engine treats as architecture
documentation. The CODEBASE LAYOUT in qa.sh is stale (wrong crate names, missing
crates) and injects misleading context. The `.caw/` skip filter handles session
logs but leaves shell scripts in the index. No skip rule currently targets
`scripts/` or shell scripts generally.

Fixed: added `"scripts"` to the path-component skip list in both
`caw-ingest/src/lib.rs::should_skip` and `build_index.rs::should_skip`.

## B4. Search-candidates mode fails silently when model cannot request file loads (low)

The search-candidates presentation instructs the model: "Mention the specific
files you need if you want them loaded." In the QA session harness, there is no
mechanism to honor this request — the model's mention of a file name does not
trigger a follow-up retrieval call. The model is told it can request content but
the infrastructure to fulfill that request doesn't exist in the QA loop, leading
to the model either hallucinating or answering from stub metadata alone.

## B5. Sometimes model generated hallucinated injected context (high)

`.caw0006/prompt-20260508-063609-turn-14.txt` contains evidence that the model
is responding with hallucinated injected content. We need to detect that it's
generating content that looks like injected context and not just hallucinating
an answer. This is a critical failure mode because it means the model is
treating the retrieval substrate as a scratch pad for its own generation, which
can lead to self-reinforcing hallucinations and loss of grounding.

## B6. Model generates fake `[recalled from ...]` markers in output (critical)

`.caw0006/prompt-20260508-063609-turn-14.txt` shows the model's response to
"what can we do to improve on the caw-curation crate?" consisting almost
entirely of lines like
`[recalled from ./crates/caw-orchestrator/src/thinking_trace.rs]` followed by
verbatim Rust code — hundreds of lines of it. This is generated output, not
injected context. The model has learned the `[recalled from ...]` provenance
format from its context and uses it as a generation pattern, effectively
treating the retrieval substrate as a scratch pad for its own output rather than
answering the question.

Consequences:

1. The actual answer is absent or buried.
2. Downstream log consumers (including QA scripts) cannot distinguish
   model-generated recall markers from real injected context.
3. The pattern could propagate: if a session log containing fake markers is
   indexed and later retrieved, the model receives its own fabricated recall
   output as authoritative source material.

Mitigations to consider: use a provenance format the model is unlikely to
generate (e.g., UUID-delimited tags); add a post-processing step that flags
`[recalled from ...]` patterns appearing in model output; detect when a response
is dominated by injection-format text and abort/retry.
