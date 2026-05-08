# A place to record bugs as they're found

## B1. Multi-pass recall corrupts fragment source paths (critical)

During iterative thinking-trace recall, the `[recalled from ...]` locator field gets populated with embedded code content instead of the file path. Observed in `.caw0003/prompt-20260508-050942-turn-8.txt`: entries like `[recalled from max_recall_iterations = 3,` where code content replaces the source path. The same prompt also contains `struct S: BudgetScheduler,` as standalone content — invalid Rust — indicating the fragment boundary detection is splitting fragments mid-struct-definition and the prefix string from one fragment is being prepended to adjacent content. Fragments from `crates/caw-orchestrator/src/lib.rs` are recalled 30+ times in one prompt with no deduplication, each a slightly different truncated window into the same code.

## B2. Session markdown log truncates after turn 3 (high)

Session log files (`session-20260508-050942.md`, `session-20260508-051313.md`) record only the first 3 user turns even though the corresponding prompt files (`-turn-1.txt` through `-turn-18.txt`) show 9+ user turns were processed. The log writer either exits early or flushes only on process exit (and is killed before it can flush). QA analysis relying on session `.md` files misses most of each session's output.

## B3. `.caw/` session log skip filter not working at index build time (medium)

Despite `should_skip` in `caw-ingest/src/lib.rs` filtering paths with a `.caw` component, session log files from `.caw/session-*.md` appear in the retrieval index for 0003 (same finding as 0002). The CLI ingestion path or the bench index builder is apparently not activating the skip correctly — either `skip_gitignore` is overriding the hidden-dir walk suppression, or `should_skip` is not being called on the walk results before insertion.

## B4. Search-candidates mode fails silently when model cannot request file loads (low)

The search-candidates presentation instructs the model: "Mention the specific files you need if you want them loaded." In the QA session harness, there is no mechanism to honor this request — the model's mention of a file name does not trigger a follow-up retrieval call. The model is told it can request content but the infrastructure to fulfill that request doesn't exist in the QA loop, leading to the model either hallucinating or answering from stub metadata alone.
