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

## B6. Model generates fake `[recalled from ...]` markers in output (critical) — PARTIALLY FIXED

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

Partially fixed: `strip_markers()` in `caw-transform` now strips complete
`[recalled from …]…[end recall]` blocks from model output before the answer is
logged or returned. `run_turn` logs a `warn!` when any are stripped, making the
failure visible in session logs. The root cause (model learning the injection
format from context) remains — switching to UUID-delimited or XML-namespaced
tags would eliminate the generation incentive entirely.

## B8. Model produces blank response after identifying an information gap in its thinking trace (high)

Session-074559 turn 3 ("how many todos are left?") shows the model generating 580 lines of correct reasoning — it accurately identified that only chunk 16/22 of TODO.md was retrieved, that chunk contains a narrative description rather than a parseable item list, and that it cannot produce a count from the available evidence. After this sound reasoning the model produced an empty final answer. The session log records a blank response for that turn.

This is distinct from B2 (session log truncation) — the session log for this session correctly captured turns 1 and 2; turn 3's model response was genuinely empty. The model's thinking was not captured as the answer; the answer itself was empty.

The failure mode: exhaustive reasoning about why an answer is impossible can lead to no answer at all. The system prompt does not instruct the model to produce a minimal hedged response when evidence is insufficient.

## B9. Consolidation notes accumulate recursively — outer note includes full text of prior note (medium)

In session-074559 prompt turn-10, the fragment for `caw-orchestrator/src/probe_recall.rs` has a consolidation note whose topic field contains a full prior consolidation note: "Evicted (relevance decayed to 0.51)... Topic: Chunk 1/5... [Prior session notes for this source: - Evicted (relevance decayed to 0.51)... Topic: Chunk 1/5... [Prior session notes for this source: ...]]".

When appending a new eviction note, the implementation is including the prior note (which contains the prior-prior note) as the "Topic" context for the new note. After enough eviction cycles, a stub's consolidation header grows without bound. Observed with 2 levels of nesting; the pattern will continue recursively.

## B10. Session history fragments remain large despite B7 summarization fix (high)

QA 0012 sessions show session-history fragments at 468 tokens and 1092 tokens respectively — far above the ≤300-character (≈50 token) target from the B7 fix. In session 080631 turn 1, session history consumed 1560/2694 workspace tokens (58%) before a single workspace stub was loaded. The B7 fix is marked done in TODO.md, but the actual prompt sizes suggest either: (a) the sessions ran against a build before the fix was active, (b) the fix applies only to model responses but not to full turn blocks injected by `collect_previous_stubs`, or (c) there is a code path that bypasses the compression. The consequence is that B7's echo-chamber effect is still fully present in QA 0012: session 080631 turn 2 gives an answer nearly verbatim identical to session 080400 turn 2, and turn 3's caw-curation query retrieved zero caw-curation stubs because the history fragments consumed the budget. Observed in QA 0012 (recommendations 1, 2).

## B7. Prior session model responses flood context in follow-on sessions (high)

When a new session starts on the same topic as a recent session, `collect_previous_stubs` loads the prior session log into the in-memory HNSW index. The prior session's full model responses (400–450 tokens each) then score as the highest-ranked semantic matches for queries using the same terminology. In QA loop 0009, session 065934's turn-3 context ("how many todos are left?") contained 2500+ tokens of session 065745's model responses — about 40% of the total context budget — with only a single TODO.md chunk retrieved for the actual query.

The fix for B3 correctly prevents prior session stubs from persisting to SQLite, but the in-memory path still loads full model response text into the retrieval pool. Because model responses contain every relevant concept in polished prose, they outcompete workspace stubs for retrieval slots. The effect is that the model answers the current session based on what it said in the prior session rather than from the workspace, creating a self-reinforcing loop: prior answer → indexed → retrieved → used as authoritative context → new answer echoes prior answer.

Distinguish this from intentional session continuity (loading prior user queries as context hints): the problem is the model's verbose *responses*, not the user queries, dominating the context.

QA 0010 quantified the scale: in session 072432 (which ran 13 minutes after sessions 071921 and 072123 on identical questions), the very first retrieval call contained 7 session history fragments totaling ~2713 tokens out of 3375 total workspace tokens — 80% of the initial context budget consumed by prior-session model responses before a single workspace stub was loaded. The effect compounds with each successive session on the same topic: session N retrieves from sessions N-1 and N-2, each of which already retrieved from its predecessors.

## B11. Qwen3 chat-template tokens leaking into session history and re-injected as context (critical)

Observed in QA loop 0013 session 082520, confirmed by grep showing 4 occurrences of the same assistant response sentence in a single prompt file, and the presence of `<|im_end|>`, `<|im_start|>user`, `<|im_start|>assistant`, and `</think>` tokens inline in the prompt's response section.

The Qwen3 adapter is returning the raw model completion string — including chat-template control tokens and the `<think>...</think>` extended-thinking block — rather than just the decoded final answer text. When this raw output is stored as a session turn and later loaded as session history, the history fragment contains these control tokens verbatim. The model then sees what appears to be a second instance of the same Q&A cycle replayed inside its context window, causing it to generate the same answer again.

Effects:
1. The prior assistant response appears 3–4 times in a single prompt, wasting ~3000 tokens.
2. Chat-template control tokens in session logs corrupt automated log parsing.
3. The model's reasoning trace (`<think>...</think>`) is stored as the answer text when the extended-thinking block precedes the final response — causing the session log to record reasoning rather than the answer.

The fix is to strip chat-template tokens and `<think>...</think>` blocks from the raw completion string before constructing `CompletionResponse.answer`. This should apply to all Qwen3-family models (and any future model that emits visible reasoning or chat-format tokens in its completion).

## B12. Qwen3 `<think>` section stored as answer text when extended thinking is active (high)

Related to B11. In session 082520 turn 4, the logged response begins with `<think>` rather than with the answer. The model's internal reasoning (which can be hundreds of tokens) is stored as the turn's answer, making the session log misleading and the history injection actively harmful — subsequent sessions will retrieve the prior reasoning as if it were a factual answer.

Distinct from B11 in that B12 is specifically about the `<think>...</think>` prefix being treated as the answer rather than as a separate artifact to be discarded. The adapter needs to detect whether the model output starts with a thinking block and, if so, extract only the content after `</think>` as the answer.
