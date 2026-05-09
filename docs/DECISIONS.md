# OpenCAW — Design Decisions and Defaults

**Status:** authoritative reference for implementation decisions. When an implementation choice conflicts with this document, fix the implementation. When this document is silent on a choice, see the Decision Protocol at the bottom — the right answer is almost never "pick something and code it up."

---

## How to read this document

Rules without reasoning produce brittle implementations. When you encounter a situation this document doesn't explicitly cover, you should be able to reason from the principles here to a correct answer. The specific values (20% budget cap, 200-token convergence threshold, 2-note cap) are conclusions that follow from reasoning — understanding the reasoning lets you judge whether an edge case falls inside or outside the same logic.

---

## Features must be active by default

The 20 QA loops that produced `qa/recommendations/` almost always found features that existed in the codebase but were never exercised — not because they were broken, but because they were opt-in. `LlmConsolidation`, `SessionEvaluator`, curation, degradation monitoring, `AugmentationSignals`: all built, none running. Every feature gated behind a flag will be disabled in practice, which means it will never be tested, never be measured, and never improve.

If a feature improves recall quality or context curation, it should run on every session. Opt-out flags are acceptable for A/B testing (you need to be able to run without a feature to measure its value). Opt-in flags are not — they mean the feature doesn't exist for any user who didn't read the changelog.

Features that must always be active (may not be exhaustive, this list shouldn't really even exist, but here we are):

| Feature | Notes |
|---|---|
| LLM-generated consolidation notes | `LlmConsolidation` when an LLM is present; `MechanicalConsolidation` only as a fallback when no LLM is available |
| LLM summarization at index time | For prose; deterministic only for code chunks where tree-sitter outline is sufficient |
| Curation pipeline | History summarization, tool output compression — always active, no flag required |
| `SessionEvaluator` | Wired into `DynamicRecallOrchestrator`; fires at every recall, eviction, probe, annotation |
| Degradation monitor | `with_degradation_monitor()` called unconditionally, not via an opt-in builder method |
| `AugmentationSignals` → retrieval | Intent classification results must change retrieval behavior — see below |
| Session history | Always injected, but with a budget cap — see below |
| Consolidation notes persisted | Orchestrator always constructed `with_store()` |
| Multi-pass convergence detection | Always enforced — see below |
| Per-turn session log flush | `write_turn` called before any error propagates |
| Chat-template token stripping | All adapters strip model-specific tokens before storing any response |

---

## Session history must not crowd out workspace stubs

Prior session model responses discuss the same topics as current queries — they use all the same words ("context", "recall", "orchestrator", "opencaw") in fluent prose. They score higher in semantic similarity than raw code stubs because prose is closer to the query format than source code is. Without any constraint, session history floods the workspace before a single stub from the actual project is loaded. The model reads its own prior output rather than the project's current state. Any error or stale claim in a prior response gets recycled as authoritative context in the next session. This compounds: session 3 inherits sessions 1 and 2's errors, and they arrive looking like retrieved evidence.

History fragments are therefore injected as a fixed budget-capped block, not as competitors in the retrieval pool. They may consume at most 20% of the workspace token budget. Retrieval slots are reserved for workspace stubs.

**Prior session responses are compressed before embedding.** The full text of a model response is never re-embedded as retrieval content. Before indexing a prior session turn, replace the model response with a synthesis of ≤50 tokens: what was asked, the main conclusion, any specific facts cited. A 400-token response that outranks every workspace stub is the failure mode this prevents.

**History content includes both sides of each turn.** User query and assistant answer (or compressed synthesis). Injecting only the user side breaks meta-queries ("is the context helpful?") because the model can't evaluate a prior response it can't see.

**Degenerate turns are labeled.** If a turn's answer is a degenerate-response prefix, inject it with a `[partial response — generation failed]` label. Don't inject it as if it were a complete and reliable response.

---

## Multi-pass retrieval must converge

Every file in an active codebase contains terms that match broad queries. "Is opencaw development completed?" matches every Rust file that uses the words "config", "default", or any variable with "mode" in its name. Each retrieval pass finds new files containing some query term and loads them. Without a stopping condition the loop runs until budget exhaustion, loading progressively less relevant material and diluting the workspace with noise. A query about project status ended up with 50 fragments from adapter implementations because they contained the word "development" as a variable name.

Stop the multi-pass loop when either:
1. The iteration admitted zero new unique stubs (by stub-id), or
2. The iteration added fewer than 200 tokens to the workspace.

Log the iteration count at which convergence was detected. If the loop consistently takes 5+ iterations before converging, that is a signal about retrieval precision — not a reason to keep iterating.

---

## Intent classification must change retrieval behavior

The intent classifier runs on every query and correctly identifies query type. If that classification doesn't change what gets loaded or how it's loaded, the classifier is burning compute on a judgment that's immediately discarded. In every QA loop that tested it, `is_inventory_request=true` was computed correctly and then ignored — retrieval loaded the single most similarity-matched chunk of TODO.md (always a single item, never a count), and the model correctly reported that it couldn't answer.

Classification results must change retrieval strategy:

- **`is_inventory_request` or `is_status_request`**: When top-k results are fragments from TODO.md, BUGS.md, or SCOPE.md, load all remaining chunks of that file up to the token budget. Similarity threshold is bypassed for same-file chunks. A count query requires document-level coverage; a similarity-ranked single chunk will never answer it.
- **`is_next_step_request`**: Proactively include TODO.md chunk 1 regardless of similarity score.
- **`is_results_request`**: Bias retrieval toward paths matching `bench-results/`.
- **`wants_explanation`**: Load `.md` stubs in full-content mode, not outline-only. Load the top-ranked documentation stubs directly rather than presenting them as search-candidates for the model to "request." The search-candidates protocol requires a request-fulfillment loop that doesn't exist in most deployments — defaulting to it for explanation queries silently serves the model a file list with no content.
- **All flags false, query under ~15 tokens**: Skip retrieval. The model is acknowledging an input, not asking a question. Running a full retrieval cycle on "nice to know" surfaces semantically adjacent but contextually irrelevant content and the model answers a question nobody asked.

---

## Embed chunk content, not metadata

If you embed a stub's metadata (path, symbol names, outline), similarity measures "does this file's name/structure match the query?" rather than "does this file's content answer the query?". Every chunk from the same large file shares the same symbols and path, so retrieval returns all 42 chunks of `caw-core/src/lib.rs` for any query about core types, flooding top-k with fragments from one file. Queries about what code *does* get no signal because behavior isn't in the symbol names.

The embedding text is the chunk content — actual code or prose — optionally prefixed with path and summary for asymmetric models. For code chunks with no outline entries (tail chunks, closing braces, test boilerplate), embed the actual content rather than the positional header. For config files (`.toml`, `.yaml`, `.json`), embed key names and values from the full file, not just the first line. The `_content: String` parameter in `SemanticRetriever::insert` is used, not silently ignored.

---

## Consolidation must capture insights, not eviction events

`MechanicalConsolidation` writes notes of the form "Evicted (relevance decayed to 0.52) during query about '...'". This tells the model that the stub was dropped and what score it had. The model cannot use this to decide anything — it doesn't know what was relevant, what was concluded, or why the stub mattered. Worse, each eviction appends the prior note as a "prior session notes" field, so after three eviction cycles the note is larger than the original content and consists entirely of nested metadata referencing itself.

`LlmConsolidation` is the default when an LLM is available. Notes record what was learned: what was relevant about the fragment, what was concluded from it, how it relates to the query context. The eviction score is logged separately and not injected into the model's context.

Per-stub notes are capped at 2 entries. When a third eviction would add a note, collapse: "Evicted N times previously; most recent: [one-line summary]". A note must never contain another note as nested content — strip any prior-note text from the topic field before storing.

---

## Model adapter capabilities must reflect reality

Hardcoding `supports_hidden_reasoning: true` for all Ollama models means every Ollama model receives instructions to emit `<probe>` and `<note>` markers. Models that can't follow those instructions produce the markers as literal output prose, contaminating answers with noise that looks like system-injected content. Hardcoding `supports_visible_reasoning: false` for all Anthropic models disables thinking-trace recall for every Claude model regardless of version. Capabilities frozen in library code can never improve without a code change.

- **`OllamaAdapter`**: query `/api/show` at construction time for actual model metadata. Log a warning and default to `false` if the endpoint doesn't respond.
- **`AnthropicAdapter`**: determine `supports_visible_reasoning` from the model ID. Claude models with extended thinking (claude-3-5-sonnet-20241022 and later) support it.
- **All adapters**: strip model-specific chat template tokens before returning `CompletionResponse`. For Qwen3 this includes `<|im_end|>`, `<|im_start|>...`, and `<think>...</think>`. The terminal `<|im_end|>` is a stop token, not content — strip it unconditionally, not only when content follows it.

---

## Degenerate responses are partial results, not failures

When `is_looping()` fires, the response before the loop starts is often a correct and complete answer — the model started correctly and then repeated itself under quantization pressure or token budget constraints. Discarding the entire response loses the answer. Treating degenerate output as a hard session error stops the session immediately, so turns 2 and 3 of a 3-turn session never run even though the model might answer them correctly.

1. Log which check fired (trigram collapse, line repetition, word dominance) and approximately where in the text it triggered.
2. Save the valid prefix — the text before the loop — with a `[response truncated — generation looped]` suffix.
3. Retry at most twice with temperature adjustment or context reduction.
4. Always call `write_turn` before propagating any error.

---

## What belongs in .cawignore vs. should_skip

The library doesn't know what's noise in your project. `should_skip` in library code is for things that are universally harmful regardless of the project using opencaw — specifically, opencaw's own session artifacts (`.caw*/`), which create a retrieval feedback loop in any project. Everything project-specific belongs in `.cawignore`. Hard-coding project paths in `should_skip` is always wrong.

**Library-level skips (in `should_skip`)**: `.caw/` and `.caw[0-9]*/` only.

**This project's `.cawignore`** excludes operational output and files that mislead retrieval without contributing useful information: build output, model cache, benchmark output, QA loop output (`qa/`), build/harness scripts (`scripts/` — these contain stale crate descriptions that outrank real docs), AI assistant instructions (`CLAUDE.md`, `.claude/`), branding copy that incidentally matches every architectural query (`MASCOT.md`).

`docs/bugs.md` and `docs/codebase-review.md` are **not** excluded. They are primary documentation about project status and known issues — exactly the content the model needs to answer status and debugging questions.

**Configuration files** (`.toml`, `.yaml`, `.json`): index whole, do not chunk, summarize with key names and values — not just the first line. A `Cargo.toml` stub must name the package and convey its purpose.

**Minimum content filter**: Stubs with `summary.trim().len() < 15` are excluded from retrieval. `token_estimate` reflects file size, not injected content length — don't use it as a proxy for useful content.

---

## Retrieval thresholds

These are starting points, not settled values. Don't change them without `HysteresisAnalysis` data from real sessions.

| Parameter | Default | Why this value |
|---|---|---|
| Load threshold | 0.7 | High enough to prefer relevance over broad coverage |
| Unload threshold | 0.4 | The gap between 0.7 and 0.4 is the hysteresis band — prevents thrashing near threshold |
| Hysteresis minimum | 0.35 | Fragments admitted twice in a session show persistent relevance; don't evict until truly irrelevant |
| Decay rate | 0.8 per step | Fast enough to evict stale content; slow enough to retain recently-referenced material |

**Score modifiers**: retrieval scores are adjusted before admission to reflect query context.

| Condition | Modifier | Why |
|---|---|---|
| `.md` stub and `wants_explanation=true` | 1.5× | Documentation answers explanation queries better than implementation code |
| `caw-bench/src/bin/` stub, non-benchmark query | 0.25× | Harness binaries reference every project concept but describe the benchmarking infrastructure, not the system |
| `caw-bench/src/` stub, non-benchmark query | 0.5× | Same issue, broader scope |
| Stub evicted N≥3 times with no probe match or annotation | 0.5× | This stub keeps getting loaded and immediately evicted — reduce its score so it stops consuming admission slots |

---

## Open questions

These are genuinely unresolved. Present options with trade-offs and ask rather than choosing unilaterally.

- **Hysteresis threshold calibration**: The defaults above are starting points. `HysteresisAnalysis` sweeps have not been run against enough session data to support different values.
- **Insertion order of recalled content**: Relevance-ranked, reverse-relevance, and stub-order are all viable. Requires controlled comparison to decide.
- **Model capability auto-detection**: The heuristic for choosing cooperative vs. transparent mode (whether to explain the recall substrate to the model) is unresolved. Default to cooperative until there is data.
- **Adaptive chunking strategy**: Token-based chunking is the current approach. Structural boundaries (function/section) may produce better retrieval for code. Don't change without a benchmark comparison.

---

## Decision protocol

When this document is silent on a choice:

1. Check `docs/design.md` section 11. If there is a settled answer, use it.
2. Check `docs/scope.md`. If the choice affects a "must have" or "should have" item, favor the option that most directly improves recall quality or context curation for the library use case.
3. If still ambiguous: stop. Write out the options and their trade-offs. Ask. The first choice is often wrong because no single implementation pass has a complete picture of how the system works together.
4. Never implement a feature behind an opt-in flag as a compromise. A feature is either the correct behavior (on by default) or it isn't ready to ship.
