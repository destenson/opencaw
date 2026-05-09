# OpenCAW — Design Decisions and Defaults

**Status:** authoritative reference for agent implementation decisions. When implementation choices conflict with this document, fix the implementation. When this document is silent on a choice, see the Decision Protocol at the bottom.

---

## Feature Defaults

Every feature listed here must be **on by default**, with no opt-in flag required. Opt-out flags are acceptable for A/B testing and measurement purposes (e.g., `--no-consolidation` to measure the impact of consolidation). Opt-in flags are not acceptable — if a feature is valuable, it should not require the user to discover and enable it.

| Feature | Default state | Notes |
|---|---|---|
| LLM-generated consolidation notes | **ON** | `LlmConsolidation` is the default synthesizer when an LLM adapter is present. `MechanicalConsolidation` is only a fallback when no LLM is available. The `--llm-consolidation` CLI flag is obsolete — remove it. |
| LLM summarization at index build time | **ON** | `LlmSummarizer` for prose; `DeterministicSummarizer` only for code chunks where tree-sitter outline is available and sufficient. |
| Curation pipeline (history summarization, tool output compression) | **ON** | The `--curate` CLI flag is obsolete — remove it. Curation is always active. |
| `SessionEvaluator` wired into `DynamicRecallOrchestrator` | **ON** | Events fired at every recall, eviction, probe, annotation. No separate enable required. |
| Degradation monitor | **ON** | Always constructed via `with_degradation_monitor()`. No opt-in. |
| `AugmentationSignals` wired to retrieval | **ON** | Intent classification results **must** affect retrieval strategy. See Intent Classifier Behavior. |
| Session history injection | **ON with budget cap** | See Session History. |
| Consolidation notes written to store | **ON** | The orchestrator must be constructed `with_store()`. Notes always persist. |
| Convergence detection for multi-pass retrieval | **ON** | See Retrieval Convergence. |
| Per-turn session log flush | **ON** | `write_turn` is called after every turn, before any error propagates. A degenerate response is still written as a turn (with whatever valid prefix exists). |
| Chat-template token stripping | **ON** | Strip `<|im_end|>`, `<|im_start|>...`, and `<think>...</think>` from all model output before storing. |
| Thinking trace extraction | **ON** | `split_thinking` runs on every model response. The extracted answer, not the raw response, is stored. An empty `<think></think>` block is a no-op — the text after `</think>` is the answer. |

---

## Indexing: Skip List

**Do not hard-code project-specific paths in the library.** `.cawignore` exists precisely so each project can declare what it doesn't want indexed. The library's `should_skip` must contain only patterns that are universally correct regardless of the project using opencaw.

### Library-level skips (hard-coded in `should_skip`)

| Pattern | Reason |
|---|---|
| `.caw/` and `.caw[0-9]*/` | Session artifacts written by opencaw itself — indexing them creates a retrieval feedback loop in every project using opencaw |

That is the complete list. Everything else is project-specific and goes in `.cawignore`.

### `.cawignore` for this project

The current `.cawignore` excludes build output, operational artifacts, and files that mislead retrieval without carrying useful information:

```
target/          # build output
.fastembed*/     # model cache
bench-results/   # benchmark output
qa/              # QA loop output directory (session transcripts, raw logs)
scripts/         # build/QA harness scripts (contain stale crate descriptions)
CLAUDE.md        # AI assistant instructions, not project documentation
MASCOT.md        # branding copy, matches every query incidentally
.claude/         # AI assistant configuration
.caw*/           # opencaw's own session artifacts
```

`docs/bugs.md` and `docs/codebase-review.md` are NOT excluded — these are exactly the documentation the model should be able to retrieve to answer questions about project status and known issues.

When adding a new path to this list, put it in `.cawignore` — never in `should_skip`.

### Configuration file handling

`.toml`, `.yaml`, `.json` files: do not skip, do not chunk. Keep whole. Summarize with key names and values, not just the first line. A `Cargo.toml` stub summary must include the package name and purpose derived from the file content, not just `[package]`.

### Minimum content filter

Stubs whose `summary.trim().len() < 15` are excluded from retrieval regardless of `token_estimate`. The `token_estimate` field reflects file size, not injected summary length — do not use it as the minimum-content proxy.

---

## Retrieval Configuration

### Similarity thresholds (defaults, tunable by measurement)

| Parameter | Default | Notes |
|---|---|---|
| Load threshold | 0.7 | Cosine similarity above which a stub is admitted to workspace |
| Unload threshold | 0.4 | Below this, a fragment is evicted |
| Hysteresis minimum | 0.35 | Fragments admitted twice or more in a session are not evicted until they fall below 0.35 |
| Decay rate | 0.8 per step | Relevance multiplied by this each reasoning step with no re-engagement |
| Top-k initial | configurable | Load all stubs above threshold up to max_initial_fragments |

These are starting points. `SessionEvaluator::hysteresis_analysis()` is the tool for tuning them from real session data.

### Per-stub deduplication

Within a single workspace, a stub-id is admitted at most once. Subsequent retrieval of the same stub-id refreshes its relevance score but does not re-inject content. Fragment boundaries must never split a `[recalled from ...]` tag — the formatter validates this before injecting.

### Score modifiers by stub kind

| Condition | Modifier |
|---|---|
| Stub is from a `.md` file and `wants_explanation=true` | 1.5× |
| Stub is from `caw-bench/src/bin/` and query has no benchmark terms | 0.25× |
| Stub is from `caw-bench/src/` generally, non-benchmark query | 0.5× |
| Stub has been evicted N≥3 times with no positive engagement | 0.5× (chronic no-engagement penalty) |

### Documentation loading strategy

For stubs from `.md` files when `wants_explanation=true`:
- Load in **full-content mode**, not outline-only.
- README.md, SCOPE.md, and `context-as-workspace.md` are the canonical explanation sources. When these are the top-ranked candidates for an explanation query, they are loaded directly — not presented as candidates for the model to "request."

---

## Retrieval Convergence

The multi-pass recall loop must stop when either condition is met:

1. The last iteration admitted zero new unique stubs (by stub-id).
2. The last iteration added fewer than 200 tokens to the workspace.

Without convergence, the loop runs until budget exhaustion — retrieving progressively less relevant content and degrading answer quality. Log the iteration at which convergence was detected.

---

## Session History Management

Session history is injected as context but must not crowd out workspace stubs.

**Budget cap**: Session history fragments may consume at most 20% of the total workspace token budget. History fragments are not added to the retrieval pool competing with workspace stubs; they are injected as a fixed header block within their capped budget.

**Compression before indexing**: Prior-session model responses are not indexed verbatim. Before adding a prior session turn to the in-memory embedding index, replace the model response with a compact synthesis of ≤50 tokens: what was asked, the main conclusion, any specific facts cited. The full response text is never re-embedded as retrieval content.

**Content of history fragments**: Each injected turn contains both sides — user query and assistant answer (or the compact synthesis if it's a prior session). Injecting only the user side breaks meta-queries like "is the context helpful?".

**Degenerate turn handling**: If a turn's answer came from a degenerate-response prefix, inject it with a `[partial response — generation failed]` label rather than verbatim, so the model knows the prior answer was unreliable.

---

## Consolidation

`LlmConsolidation` is the default synthesizer when an LLM is available. `MechanicalConsolidation` is only used as a fallback.

**Note cap**: Each stub stores at most 2 consolidation notes. When adding a third, collapse the two oldest into a single line: `"Evicted N times previously; most recent: [query context]"`. A note must never contain another note as nested content — strip any embedded prior-note text from the topic field before storing.

**Note content**: Consolidation notes should record what was learned or concluded from the fragment, not just that it was evicted. The eviction score is logged separately. A useful note: "This stub covers eviction policy and hysteresis thresholds. Retrieved during queries about recall quality and threshold tuning." An useless note: "Evicted (relevance decayed to 0.52)."

**Chronic no-engagement tracking**: After N=3 consecutive evictions from sessions that did not produce a probe match or annotation on this stub, synthesize a note flagging it as low-engagement and apply the 0.5× chronic penalty modifier.

---

## Intent Classifier Behavior

The intent classifier runs on every query and produces `AugmentationSignals`. These signals **must** change retrieval behavior — they are not advisory.

| Signal | Required retrieval action |
|---|---|
| `is_inventory_request=true` or `is_status_request=true` | If top-k results are fragments from TODO.md, BUGS.md, or SCOPE.md, load **all** remaining chunks of that file up to the token budget before running the answer model. Similarity threshold is bypassed for the same-document chunks. |
| `is_next_step_request=true` | Proactively load TODO.md chunk 1 regardless of similarity score. |
| `is_results_request=true` | Bias retrieval toward paths matching `bench-results/`. |
| `wants_explanation=true` | Use full-content mode for `.md` stubs. Apply 1.5× modifier. Load top-ranked documentation stubs directly instead of presenting search-candidates. |
| All flags false + query < 15 tokens | Skip retrieval. Prompt model to ask for clarification. Do not run a full retrieval cycle on an acknowledgment. |

The `search-candidates` presentation mode is only appropriate when: (a) the downstream system can fulfill file-load requests from the model, and (b) the intent does not clearly point to specific files. In the QA harness, `search-candidates` cannot be fulfilled — load top-k stubs directly.

---

## Model Adapter Capabilities

### AnthropicAdapter

`supports_visible_reasoning` must not be hardcoded `false`. Claude models with extended thinking enabled (claude-3-5-sonnet-20241022 and later) support visible reasoning. Detect from model ID or constructor parameter. When `supports_visible_reasoning=true`, the full thinking-trace recall path is active.

### OllamaAdapter

`supports_hidden_reasoning` must not be hardcoded `true`. Query `/api/show` at construction time to get actual model metadata. Log a warning and default to `false` if the endpoint doesn't respond. A model that can't follow marker instructions will produce `<probe>` and `<note>` tags as output prose rather than signaling recall — which contaminates answers.

### Chat template tokens

All adapters must strip model-specific chat template tokens before returning `CompletionResponse`. For Qwen3 this includes `<|im_end|>`, `<|im_start|>...`, `<think>...</think>`. The stop token `<|im_end|>` at the end of a response is a normal termination signal — strip it unconditionally, not only when content follows it.

---

## Degenerate Response Handling

When a model response is flagged as degenerate:

1. **Log the failure** with the triggering check (trigram collapse, line repetition, word dominance) and the approximate position in the text where it fires.
2. **Save the valid prefix** — the text before the loop starts is usually correct. Store it as the turn answer with a `[response truncated — generation looped]` suffix.
3. **Do not hard-exit.** Retry the turn with a temperature adjustment or context reduction. At most 2 retries before recording the best available answer and continuing.
4. **Write the turn** before propagating any error. `write_turn` is always called, even for degenerate turns.

---

## Embedding Text Construction

The embedding text for a stub is the **chunk content** (the actual code or prose), optionally prefixed with path and summary for asymmetric models. It is never constructed from metadata alone.

For code chunks with outline entries: embed the chunk content. Use the summary as a query-side prefix only.

For code chunks with no outline entries (tail chunks): embed the chunk content directly. Do not use the positional header (`"Chunk 20/20 of path — first code line"`) as the embed text.

For `.toml`/`.yaml`/`.json`: embed key names and values from the full file, not just the first line.

The `_content: String` parameter in `SemanticRetriever::insert` is used, not silently ignored.

---

## caw-server Status

`caw-server` is a **functional OpenAI-compatible retrieval-augmentation proxy**. It retrieves context, augments the last user message, and forwards to an upstream model server with streaming passthrough. It uses Candle + CUDA for embeddings. It is not scaffold. It is not stubbed.

Current limitations: synchronous, Candle-only embedder, no multi-pass orchestration. These are v2 items. Do not describe caw-server as incomplete or placeholder.

---

## Open Questions

These are genuinely unresolved and should not be decided unilaterally by an agent. Flag them and ask.

- **Hysteresis threshold tuning**: The 0.7/0.4 defaults are starting points. Actual calibration from `HysteresisAnalysis` runs has not been completed. Do not change the defaults without measurement data.
- **Insertion order of recalled content**: Relevance-ranked, reverse-relevance, or stub-order are all viable. This is an empirical question. Do not pick one arbitrarily.
- **Adaptive chunking boundary strategy**: Whether chunking should be token-based, structural (function/section boundaries), or both is open. The current token-based approach is fine for v0.1.
- **Model capability detection**: The heuristic for choosing cooperative vs. transparent mode (when to tell the model about the recall substrate) is unresolved. Default to cooperative (system prompt explains the mechanism) for now.

---

## Decision Protocol

When this document is silent on a choice, apply in order:

1. Check `docs/design.md` — if it has a settled answer in section 11, that answer applies.
2. Check `docs/scope.md` — if the choice affects a "must have" or "should have" item, choose the option that serves the library target, improves recall quality, or improves context curation.
3. If still ambiguous: **do not invent a solution**. Record the question, the options you see, and the trade-offs of each. Stop and ask. The first choice is often wrong because no single pass has a complete picture of how the system works.
4. Never add a feature behind a flag as a compromise. Either it's the right behavior and it should be the default, or it's wrong and shouldn't be added at all.
