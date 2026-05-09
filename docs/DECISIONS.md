# OpenCAW — Principles and Decisions

**Status:** authoritative reference for implementation decisions. Read `docs/design.md` for the full thesis; read this document before making any significant implementation choice.

---

## What opencaw is

opencaw treats LLM context as a managed workspace, not a container. The core idea is that working-set quality — what is in context and what isn't — is a primary lever on output quality, not just context window size. The system replaces file content with lightweight stubs, recalls full content on demand as the model reasons, and evicts stale material while preserving what was learned.

The differentiating mechanism is using the model's own reasoning trace as the retrieval signal. This sidesteps the need for a separate retrieval query and produces retrieval that tracks what the model is actually thinking about rather than what the user literally typed.

opencaw is a **library** that provides the primitives for context management; applications and deployment targets are built on top of it. Design choices that serve a specific deployment at the expense of generality are wrong.

---

## Invariants

These are properties the system must maintain to function correctly. Any implementation choice that violates an invariant is wrong regardless of how pragmatic it seems in the moment.

**The workspace must reflect the current query.** At any moment, the workspace should contain the content most relevant to what the model is currently reasoning about. Anything that displaces relevant content — session history, noise stubs, runaway retrieval — degrades output quality proportionally. Every subsystem that touches the workspace is responsible for not crowding out content that actually helps.

**Features must run to exist.** A feature behind an opt-in flag doesn't exist in practice. It will never be tested in real workloads, never be measured, and never improve. The only way to know whether something works is to run it. Opt-out flags are fine for measurement (A/B comparisons require a control); opt-in flags mean the feature is effectively unimplemented.

**Retrieval signal must come from meaning, not structure.** Embedding metadata — paths, symbol names, outlines — measures structural similarity. Behavior, purpose, and semantics are in the content. A query about what code does cannot be answered by what it's named.

**Consolidation preserves insight, not events.** The purpose of eviction-time consolidation is to carry forward what was learned from a fragment after it leaves the workspace, so future recalls start with more context than cold. Recording that a fragment was evicted at score 0.52 is not insight. What was relevant about it, what was concluded from it — that is.

**The library does not know your project.** Project-specific configuration — what to skip, what to boost, what counts as noise — belongs in deployment configuration (`.cawignore`, retrieval weights, etc.), not in library code. Hard-coding project paths or heuristics in `should_skip` or retrieval logic is always wrong.

---

## Architectural decisions

These are the significant design choices that define what opencaw is. They are settled. Relitigating them without new evidence wastes time.

### Thinking-trace as retrieval query

The model's own reasoning is the retrieval signal, not a reformulation of the user's query. This works because reasoning traces are more topical and denoised than mixed conversation turns, and step boundaries provide natural trigger points. It also means retrieval tracks the model's actual line of thought rather than the surface form of the question.

The practical consequence: retrieval is triggered at step boundaries, not once per user turn. The orchestrator embeds reasoning steps and matches them against the stub index incrementally.

### Multi-turn orchestration as the baseline

Single-pass retrieval produces a fixed context for the entire response. Multi-turn orchestration — where retrieval can inject new content between generation steps — allows the context to evolve as the model's reasoning evolves. This is the only approach that works for queries that require synthesizing information the model discovers mid-reasoning.

The mid-stream engine-integration path (interrupting generation in-place) is a future optimization, not the current approach. Multi-turn orchestration works with any standard request/response API.

### Stubs replace content; recall materializes it on demand

Files in the workspace are represented as structured stubs until the model's reasoning triggers recall. This keeps the context compact during broad exploration and focuses token budget on content the model is actually reasoning about. The stub carries enough metadata for the model to decide whether to request full content without reading it.

### Hysteresis for eviction

A single threshold produces thrashing: content loaded because it was marginally relevant gets evicted one step later, then re-loaded, then evicted. Two thresholds — load at 0.7, unload at 0.4 — create a band in which content stays once admitted, preventing the oscillation. The gap between them is the stability zone.

### Consolidation as episodic-to-semantic transfer

When a fragment is evicted, the stub absorbs a synthesized note of what was learned during the loaded period. This is analogous to how episodic memory consolidates into semantic memory: the raw experience (the fragment content) is gone, but the derived understanding (what mattered about it) persists and informs future recalls. Without this, every session starts cold regardless of prior work.

The consolidation synthesizer must use an LLM when one is available. A mechanical fallback (recording the eviction score) is not consolidation — it is bookkeeping that gives the appearance of consolidation without the function.

### Session history serves continuity, not retrieval

Prior conversation turns are not workspace stubs. They serve a different purpose — helping the model maintain coherence across a session — and must be handled differently. Model responses use the same vocabulary as current queries and will outrank workspace stubs in similarity search if allowed to compete. Session history is injected as a budget-capped fixed block, not as a retrieval candidate.

Prior session responses (from earlier sessions, not the current one) are compressed before embedding. A 400-token response that outranks workspace stubs is not continuity — it is the current session reading from its own prior output rather than from the actual project.

### Intent classification changes retrieval strategy

The intent classifier exists to adapt retrieval to query type. An inventory query ("how many todos are left?") requires document-level coverage of TODO.md; similarity search will return the single most-matching chunk, which is never sufficient to answer a count. A status query needs the overview documents. An explanation query needs documentation in full-content mode, not outline-only. Classification that doesn't change behavior is waste.

---

## Implementation principles

When making a choice this document doesn't explicitly cover, apply these in order.

**Prefer measurement over guessing.** When a threshold or configuration value is uncertain, instrument it and measure. The specific values that exist (load=0.7, 20% history cap, 200-token convergence threshold) are starting points based on observed failure modes, not empirically validated optima. Don't change them without data; don't treat them as immutable.

**Default to on.** If a feature improves recall quality or context curation, it runs by default. The cost of a bad default is proportional to how long it runs before anyone notices. The cost of a missing default is that nobody ever notices.

**Library vs. deployment.** Ask whether a choice belongs in the library or in deployment configuration. If it depends on the project, it belongs in configuration. If it applies to every project using opencaw, it belongs in the library.

**Serve the library target.** The library is the primary deployment target. Middleware and engine plugins are future work. A choice that makes the library harder to embed in order to make a future deployment target easier is premature.

**When in doubt, ask.** The first implementation choice is often wrong because no single pass has a complete picture of how all the subsystems interact. Write out the options and their trade-offs and ask rather than picking one. This is not a sign of weakness — it is the correct response to genuine ambiguity.

---

## Settled configuration details

These are specific values and constraints that follow from the architectural decisions above. They are not arbitrary — each has a failure mode it prevents — but they are also not sacred. Change them when measurement supports a different value.

**Session history budget cap: 20%.** Enough for continuity signal; not enough to crowd out workspace stubs for any realistic query.

**Prior session response compression: ≤50 tokens.** A synthesis of what was asked and what was concluded. Not the full response text, which reads like authoritative documentation to the similarity search.

**Multi-pass convergence: stop when an iteration adds <200 tokens or zero new unique stubs.** Either condition means the loop is no longer finding useful content.

**Consolidation note cap: 2 per stub, no nesting.** Notes that grow without bound defeat the purpose. A stub that has been evicted many times gets "Evicted N times; most recent: [summary]" — not an accumulation of all prior notes.

**Retrieval thresholds: load=0.7, unload=0.4, hysteresis floor=0.35.** Starting points pending `HysteresisAnalysis` sweeps. The floor applies to stubs that have been admitted at least twice in a session (indicating persistent relevance).

**Score modifiers:** `.md` stubs get 1.5× for explanation queries (documentation answers those better than implementation); `caw-bench/src/` stubs get 0.5× for non-benchmark queries (0.25× for harness binaries specifically); stubs evicted 3+ times with no engagement get 0.5× (noise attractors that keep consuming admission slots).

**Minimum content filter:** stubs with `summary.trim().len() < 15` are excluded from retrieval. `token_estimate` reflects file size, not how much content the stub actually contributes — don't use it as a content-quality proxy.

---

## Open questions

Genuinely unresolved. Present options with trade-offs; don't choose unilaterally.

- **Hysteresis threshold calibration.** The values above are starting points. `HysteresisAnalysis` sweeps against real session data haven't been run. The right values are workload-dependent.
- **Insertion order of recalled content.** Relevance-ranked, reverse-relevance, and stub-order are all viable. Requires controlled comparison.
- **Cooperative vs. transparent mode selection.** Whether to explain the recall mechanism to the model (cooperative) or keep it invisible (transparent) depends on model capability. The heuristic for choosing automatically is unresolved. Default to cooperative.
- **Adaptive chunking strategy.** Token-based is the current approach. Structural boundaries (function/section) may be better for code. Requires a benchmark comparison before changing.

---

## Decision protocol

1. Check `docs/design.md` §11 for settled answers to architectural questions.
2. Check the invariants and architectural decisions above. If a choice violates an invariant, it's wrong. If it contradicts a settled architectural decision, it needs a strong justification.
3. Apply the implementation principles.
4. If still ambiguous: stop, write out the options and trade-offs, and ask. Don't implement a compromise behind a flag.
