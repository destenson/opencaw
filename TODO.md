# TODO

Original Design: [docs/origin.md](docs/origin.md). Scope: [docs/scope.md](docs/scope.md). Settled decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

**For priority and sequence, see [docs/ROADMAP.md](docs/ROADMAP.md)** — this file is the *unordered* backlog; the roadmap carries the ordered critical path and current focus.

The following sections are in no particular order. Do not infer that high priority items are listed first.

## Logging

- Convert all logging to `tracing` with structured fields, and add timestamps. Use `debug` for internal state changes and `info` for user-relevant events (e.g., "retrieved 3 fragments, evicted 2 fragments"). Add component tags to log fields to clarify which part of the system is logging (retriever, orchestrator, probe, etc.). User-facing messages are the only exception, and they should be println!() or eprintln!() so they are visible even with a restrictive log filter.

## Validation & performance

- Dogfood opencaw over this repo as a coding-agent context server (drive a real agent through caw-server against the repo index). The `code-agent` `caw-bench` workload (questions = agent mid-task info needs: signatures, trait bounds, struct fields, call sites; needle + judge scoring) needs expanded QA set and run the recall-on/off sweep.
- Reduce end-to-end eval latency (faster/smaller answer model or vLLM/TensorRT serving stack)

## Retrieval

- Recover "both retrievers miss" queries: contextual embeddings + doc2query at index time
- Graph-neighbor expansion gives +0.000 rank lift on the current sysdoc index (`graph-eval.sh --diagnose --fusion all`): edges are absent or not helping. Revisit edge extraction/population before relying on graph expansion.
- Bench/cli measure pure cosine: both build `SemanticRetriever`, not `HybridRetriever` (BM25 fusion). Decide whether the multi-pass engine should retrieve hybrid like the proxy does, and measure the effect on rank-of-gold.

## Ingestion & Indexing

- Background indexer with lazy fallback
- Stale stub detection and re-indexing: at session start, re-ingest files whose content hash changed
- Add document-level overview stubs for key project docs (headings + completion state), boosted for inventory/status queries
- Faster embedding backend (~10x): ONNX Runtime CUDA EP / TensorRT FP16 (blocked on dlopen-preload of cuDNN)

## Eviction & Consolidation

- Conflict detection beyond term overlap (contradicting assertions, inconsistent numbers, negation)

## Prompt Transformer

- Additional reference surfaces: fenced blocks with `path=`, bare paths matching a regex
- Remove hard-coded responses from classification/transformer adapters (transforming is not a gate)

## Orchestration

- Streaming recall: interleave retrieval with token generation mid-response (needs async streaming adapter traits)
- Progressive disclosure: upgrade a stub already in the workspace to full content in-place when a probe fires on it

## Adapters

- Async adapters: replace sync-wrapped `block_on` (deferred to v2)

## Curation Hooks

- Few-shot management: surface per-example token cost delta
- Curated context surface in the system prompt (authoritative distilled facts/instructions)

## Measurement

- Insertion-order experiments: relevance-ranked vs reverse vs original-stub order for recalled content

## Degradation & Monitoring

- Model-message tracing (full request/response JSONL via `TracingAdapter`) is on by default in caw-cli; opt out with `CAW_NO_TRACE=1`. Remaining: wire caw-server's HTTP passthrough path (no `ModelAdapter` there) and adopt the same sink in caw-bench alongside its per-item `--trace-out`.
- Orchestrator decision-event logging into the same trace stream (retrieval/probe/load/eviction events sharing the `TraceSink`, interleaved with the llm_request/llm_response pairs) — the sink already supports it; the orchestrator has no hook yet.
- Logging: timestamps + component tags on the path to degradation
- Logging: actionable error messages
- Notification system when degradation is detected (affected components, causes)
- Periodic review/analysis of degradation incidents

## Infrastructure

- Over-decomposed workspace (12 crates); consider folding `caw-provenance`/`caw-eval`/`caw-scheduler` — do not restructure without approval

## Intent classifier

- Intent classifier must not gate retrieval; use it to bias retrieval/prompt, never to disable it
- Tolerate classifier parse failures in small models (don't require strict JSON)
- Intent-driven proactive context injection: act on `AugmentationSignals` (status→todo docs + `git status`, results→bench files, inventory→file listing) before the answer model runs
- Redesign `QueryIntent::guidance_lines` — prototype leftover. It maps each intent flag to a fixed hand-written sentence and turns any leftover `extra` bool-true key into a synthesized `Additional context: {key}.` line. Both are canned prose stuffed into the answer prompt: the per-flag sentences are static boilerplate not grounded in the query, and the `extra`-key synthesis dresses an unrecognized model token up as an instruction. Decide what guidance (if any) the answer model should actually receive from intent, rather than emitting templated strings. (Near-miss schema keys like `is_explanation` are now folded onto their real field before this runs, so this is about the design of the guidance layer itself, not the parser.)

## Session history

- Inject session history as a fixed header, not retrieval-slot fragments
- Exclude model response text from the session-history embedding pool (index user queries only)
- Include a compressed assistant answer in session-history fragments, not just the user query
- Deduplicate session history before injection (single contiguous fragment or `HistorySummarizer`)
- Add TTL/expiry for session-derived in-memory HNSW embeddings
- Verify the B7 session-history summarization fix is active on the CLI/QA path

## Retrieval quality (stubs & ranking)

- Replace the prototype hybrid fusion (min-max-weighted-sum in `caw-index` `HybridRetriever`) with a principled fusion — RRF is the candidate; `graph-eval.sh --fusion all` already compares strategies. The current impl is a placeholder per the maintainer, not a tuned baseline.
- Raise stub quality floor beyond `token_estimate` (drop bare single-line code statements)
- Improve stub summaries for trailing code-fragment chunks (fall back to parent file summary)
- Load documentation stubs in full-content mode for `wants_explanation` queries
- Crate-name query boost: when a query names a crate, rank its content-rich stubs first
- Deduplicate overlapping content windows from the same file in multi-pass recall
- Tune eviction threshold vs decay rate; dampen decay for repeatedly re-admitted fragments
- Log `token_estimate` vs actual injected token count per fragment; alert on large discrepancy
- Consider faster/more powerful embedding models

## Robustness

- Detect & mitigate degenerate output with capped retries/fallback instead of failing the turn
- Detect self-referential/meta queries and skip new retrieval; inject the current workspace summary
- Record the eviction-triggering query in mechanical consolidation notes

## Tool & reference support

- Tool call support: visible tool output + proactive context injection
- Detect git/repo references in prompts; link stubs to git status/blame/branch/log
- Git-aware workspace (ambient, not prompt-triggered): make the model implicitly aware of repository state and its changes across a session, so that when `git status` changes (files staged/modified/added, branch switched, commits made) the workspace reflects it without the user having to ask. Distinct from the reactive "detect git references in prompts" item above: this is a background signal feeding the curation loop (e.g. re-ingest/mark-stale changed files, surface a current-state summary like branch + dirty paths). Open questions to resolve before building: where the polling/notification of state change lives (the library is sync today — see Adapters/async), how to expose it without baking project-specific git heuristics into the general-purpose library (likely a generic "external state source" hook the host wires git into), and how a state-change event should interact with eviction/consolidation. Future work, not v0.1.

## Library hygiene

- Remove project-specific paths & heuristics (incl. rust-specific and hardcoded `caw-bench`/`caw-llama-sys` references) from library code

## CLI

- caw-cli should default to writing its index and session history under `<CWD>/.caw/<hash>/` (per-corpus hash of the canonical `--dir` path, preserving "distinct corpora don't share an index"), NOT under `~/.cache/caw/<hash>/` as it does today. This reverses the deliberate choice recorded in `docs/DECISIONS.md:198` and the `default_cache_dir` comment at `crates/caw-cli/src/main.rs:38-43` (whose rationale was "running `caw` in a project never drops a `.caw/` into it"). The user now prefers CWD-local by default so the CLI never writes outside the working tree unless asked. Settled design (decided 2026-06-17, not yet implemented):
  - New default: `<CWD>/.caw/<hash>/{index.db, sessions/}` when the corpus is the CWD (no `--dir`, or `--dir` resolves to CWD).
  - When the corpus is specified separately from CWD (`--dir <elsewhere>`), do NOT default the cache to `<CWD>/.caw/` — the cache location must be specified explicitly (`--db`/`--session-dir`, or the user-dir flag). Rationale: operating on an external corpus from CWD should not silently drop `.caw/` into CWD. Open detail to confirm when picked up: error-and-ask vs. fall back to the user cache dir for the external-corpus case.
  - Opt-in flag restores the current `~/.cache/caw/<hash>` behavior; accept both `--user` and `--global` as aliases for the same opt-in.
  - `--db` / `--session-dir` continue as full path overrides (unchanged).
  - Add `.caw/` to this repo's `.gitignore` so running `caw` here doesn't pollute git.
  - Existing `~/.cache/caw/<hash>` indices are abandoned on the switch (they rebuild on demand).
  - Scope: only `caw-cli` uses `default_cache_dir`; `caw-server`/`serve.sh` and `caw-bench` don't share it, so the change is isolated to the user-facing CLI. Update the `main.rs` comment and reverse the `DECISIONS.md:198` entry when implemented.

## Benchmarking & evaluation

- Scripts to run `caw-bench` across seeds/models/workloads for threshold tuning
- Add a `caw-bench-sweep` config (models × workloads × parameters)
- Add a `caw-bench-intent` binary to evaluate small models as intent classifiers
- Add a `caw-bench-probe` binary to evaluate small models as relevance probes
- Add more default bench cases with varied intent combinations
- Expand the intent-classification benchmark prompt suite (multi-intent, edge cases)
- Expand the QA question set with retrieval-specific queries (symbol lookup, cross-file synthesis, bug investigation)
- QA harness: validate each question before sending to `caw-cli` (skip acknowledgments)
- Eval instrument is too noisy to resolve small effects: with the judge nondeterminism fixed, single-seed n~30 still has a large noise floor (recall-on absolute answer_score swung 0.461→0.338 on identical items between two runs, partly judge, partly answer model). Before optimizing recall, raise statistical power. Done: paired per-item deltas (not diff-of-means), and a re-runnable judge (`--judge-trace`, which scores persisted answers against any judge without regenerating). Remaining: more seeds; a reproducible serial generation path (concurrency>1 isn't bit-reproducible); and actually *use* `--judge-trace` to average several judge passes / pin a judge and quantify the judge's own variance contribution.

## GGUF models to evaluate

- Test `~/models` GGUF models as cheaper intent-classification / relevance-probe adapters

## Diffusion models (exploration — never used in opencaw)

We have never used a diffusion language model in opencaw. A custom `llama.cpp` build at `~/src/llama.cpp` can run the DiffusionGemma model in `~/models/unsloth`. Goal: find where and how diffusion LMs can enhance opencaw, not just slot one in as a drop-in answer model.

Diffusion LMs differ from autoregressive ones in ways that may map onto specific opencaw subsystems — record these as hypotheses to test, not as settled fit:

- **Iterative/parallel denoising** instead of left-to-right token generation. The multi-pass refinement step (orchestrator `dynamic.rs`, the "refinement produced lower-quality answer" path) is itself an iterative-improvement loop — a diffusion model's native refinement may be a better fit there than re-prompting an AR model, or may compose with it.
- **Infilling / bidirectional context.** Consolidation (rewriting/merging evicted stubs into a compact note) and stub-summary generation are constrained-rewrite tasks where bidirectional context could help.
- **Cheap structured classification.** The intent classifier and relevance probe want fast, well-formed short outputs (JSON-ish); a small diffusion model may produce them more reliably than a small AR model that drifts. (Consider fine-tuning for these.)

First steps before committing to anything: (1) stand up the custom `llama.cpp` build as a `caw-bench` adapter (likely via the existing `LlamaCpp` adapter path or an OpenAI-compatible server it exposes — verify which) and confirm it generates at all; (2) measure it on one concrete task with an existing bench (intent classification or relevance probe) so the comparison is apples-to-apples against the current small-model baseline. Don't add diffusion anywhere in the library until a measured task shows it helps.

Kept here (backlog), not in `docs/ROADMAP.md`: this is exploratory and off the current dogfooding critical path. Promote to the roadmap only if a measured result makes it a priority.
