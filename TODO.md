# TODO

Design: [docs/design.md](docs/design.md). Scope: [docs/scope.md](docs/scope.md). Settled decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

The following sections are in no particular order. Do not infer that high priority items are listed first.

## Validation & performance

- Dogfood opencaw over this repo as a coding-agent context server (drive a real agent through caw-server against the repo index). The `code-agent` `caw-bench` workload (questions = agent mid-task info needs: signatures, trait bounds, struct fields, call sites; needle + judge scoring) needs expanded QA set and run the recall-on/off sweep.
- Reduce end-to-end eval latency (faster/smaller answer model or vLLM/TensorRT serving stack)

## Retrieval

- Recover "both retrievers miss" queries: contextual embeddings + doc2query at index time

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

## Session history

- Inject session history as a fixed header, not retrieval-slot fragments
- Exclude model response text from the session-history embedding pool (index user queries only)
- Include a compressed assistant answer in session-history fragments, not just the user query
- Deduplicate session history before injection (single contiguous fragment or `HistorySummarizer`)
- Add TTL/expiry for session-derived in-memory HNSW embeddings
- Verify the B7 session-history summarization fix is active on the CLI/QA path

## Retrieval quality (stubs & ranking)

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

## Library hygiene

- Remove project-specific paths & heuristics (incl. rust-specific and hardcoded `caw-bench`/`caw-llama-sys` references) from library code

## Benchmarking & evaluation

- Scripts to run `caw-bench` across seeds/models/workloads for threshold tuning
- Add a `caw-bench-sweep` config (models × workloads × parameters)
- Add a `caw-bench-intent` binary to evaluate small models as intent classifiers
- Add a `caw-bench-probe` binary to evaluate small models as relevance probes
- Add more default bench cases with varied intent combinations
- Expand the intent-classification benchmark prompt suite (multi-intent, edge cases)
- Expand the QA question set with retrieval-specific queries (symbol lookup, cross-file synthesis, bug investigation)
- QA harness: validate each question before sending to `caw-cli` (skip acknowledgments)

## GGUF models to evaluate

- Test `~/models` GGUF models as cheaper intent-classification / relevance-probe adapters
- Test `~/models/unsloth` diffusion-gemma GGUF via custom `llama.cpp`; consider fine-tuning for intent/probe tasks
