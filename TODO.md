# TODO

Design: [docs/design.md](docs/design.md). Scope: [docs/scope.md](docs/scope.md). Settled decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

The following sections are in no particular order. Do not infer that high priority items are listed first.

## Logging

- Convert all logging to `tracing` with structured fields, and add timestamps. Use `debug` for internal state changes and `info` for user-relevant events (e.g., "retrieved 3 fragments, evicted 2 fragments"). Add component tags to log fields to clarify which part of the system is logging (retriever, orchestrator, probe, etc.). User-facing messages are the only exception, and they should be println!() or eprintln!() so they are visible even with a restrictive log filter.

## Validation & performance

- Dogfood opencaw over this repo as a coding-agent context server (drive a real agent through caw-server against the repo index). The `code-agent` `caw-bench` workload (questions = agent mid-task info needs: signatures, trait bounds, struct fields, call sites; needle + judge scoring) needs expanded QA set and run the recall-on/off sweep.
- Reduce end-to-end eval latency (faster/smaller answer model or vLLM/TensorRT serving stack)

## Retrieval

- Recover "both retrievers miss" queries: contextual embeddings + doc2query at index time
- Thin/empty stubs dominate retrieval misses: `graph-eval.sh --diagnose` (sysdoc n=100) shows many changelog gold chunks have empty stub bodies (`summary=0c body=0tok`) — nothing to embed or match, so they never rank (changelog recall@1=0.25). Fix ingestion so every indexed chunk carries embeddable content, or exclude genuinely-empty chunks. This, not fusion, is the largest single miss cause on sysdoc.
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

## Session history

- Inject session history as a fixed header, not retrieval-slot fragments
- Exclude model response text from the session-history embedding pool (index user queries only)
- Include a compressed assistant answer in session-history fragments, not just the user query
- Deduplicate session history before injection (single contiguous fragment or `HistorySummarizer`)
- Add TTL/expiry for session-derived in-memory HNSW embeddings
- Verify the B7 session-history summarization fix is active on the CLI/QA path

## Retrieval quality (stubs & ranking)

- Replace the prototype hybrid fusion (min-max-weighted-sum in `caw-index` `HybridRetriever`) with a principled fusion — RRF is the candidate; `graph-eval.sh --fusion all` already compares strategies. The current impl is a placeholder per the maintainer, not a tuned baseline.
- Recall regression (recall-on buries gold in load order — the "it used to help, now it doesn't" bug): two mechanisms seen in opencaw traces — (1) `max_chunks_per_source=3` lets 3 chunks of one (often non-gold) file flood the head of the loaded set; (2) thinking-trace re-query drift loads a different set than the raw user query (qa_001/qa_002 load a wrong file's chunks ahead of gold). Some items (qa_003) load gold yet still answer wrong, suggesting the extra ~2× content distracts. Investigate initial-load selectivity and per-source head ordering. (Blocked on deterministic eval to measure fixes.)
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
- Decouple judging from generation: judging does not need to run inline with the answer run. Persist raw answers (already in `--trace-out`) during the run, then judge afterward — in batches, re-runnable against a fixed/pinned judge without regenerating answers. Removes judge latency and nondeterminism from the generation loop and lets the same answers be re-scored by different judges for comparability.
- Answer-model nondeterminism at `--concurrency > 1`: even with `temperature 0` + a fixed `--seed`, batched/concurrent inference is not bit-reproducible (floating-point non-associativity in batched matmuls), so the same item can yield different answers run-to-run. Reproducible measurement currently requires `--concurrency 1` (serial). Investigate a serial generation path for eval, or ollama/llama.cpp batch-determinism options. (Measured 2026-06-14: 23/29 recall-on answers differed across two concurrency-4 runs.)
- **Multi-pass recall-on is non-reproducible even serial + seeded** (the real blocker for measuring recall changes): with `--concurrency 1 --seed 42 --temperature 0`, recall-**off** is now bit-identical across runs (sampling fix works), but recall-**on**'s *loaded set diverges* run-to-run (verified 2026-06-14: qa_002 loaded different files in two identical-config runs). The divergence is upstream of generation — the thinking-trace re-query loads different content, almost certainly because the streaming thinking path (`thinking_with_steps`, `stream:true` with early client-side stop) captures a timing-dependent amount of reasoning before the orchestrator stops the stream. Fix the streaming early-stop to be content-deterministic (stop on a parsed step boundary, not on arrival timing) so the multi-pass trajectory is reproducible. Until then, recall-on answer-quality deltas cannot be measured reproducibly.
  - Streaming-early-stop hypothesis above is wrong: `run_turn` reads the trace from `complete()`'s `thinking`, never calls `thinking_with_steps`. Trace recall was also inert in the bench (searched an empty index); now queries the corpus retriever.
  - Re-measure the live loaded-set divergence now that trace recall queries the corpus.
  - Attribute the remaining divergence to a specific text-driven path (probes / file-expansion / candidate-list mentioned-files) before stabilizing it.
- Wire `seed` into the vllm/`OpenAiCompatibleAdapter` path: `AdapterSpec.seed` is currently honored only by Ollama and Groq; the OpenAI-compatible adapter ignores it, so vllm answer/judge runs aren't reproducible.
- Eval instrument is too noisy to resolve small effects: with the judge nondeterminism fixed, single-seed n~30 still has a large noise floor (recall-on absolute answer_score swung 0.461→0.338 on identical items between two runs, partly judge, partly answer model). Before optimizing recall, raise statistical power: paired per-item deltas (not diff-of-means), more seeds, and/or a pinned/averaged judge.

## GGUF models to evaluate

- Test `~/models` GGUF models as cheaper intent-classification / relevance-probe adapters
- Test `~/models/unsloth` diffusion-gemma GGUF via custom `llama.cpp`; consider fine-tuning for intent/probe tasks
