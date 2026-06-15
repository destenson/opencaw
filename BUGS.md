# Bugs

Open bugs only. Fixed bugs are not tracked here — they live in git history, `docs/DECISIONS.md`, and `docs/archive/findings.md`. Some QA-era findings (loops 0001–0021) are left in `docs/archive/findings.md` rather than carried here: B0/B1/B2 (degenerate-output era, not re-verified — re-check against a fresh session log before treating as real) and B7/B10 (prior-session responses outcompeting stubs — root cause mitigated by the 0.25 history-budget cap).

## Reproducibility

- **Answer-model nondeterminism at `--concurrency > 1`**: even with `temperature 0` + a fixed `--seed`, batched/concurrent inference is not bit-reproducible (floating-point non-associativity in batched matmuls), so the same item can yield different answers run-to-run. Reproducible measurement currently requires `--concurrency 1` (serial). Investigate a serial generation path for eval, or ollama/llama.cpp batch-determinism options. (Measured 2026-06-14: 23/29 recall-on answers differed across two concurrency-4 runs.)
- **`seed` ignored on the vllm / `OpenAiCompatibleAdapter` path**: `AdapterSpec.seed` is honored only by Ollama and Groq; the OpenAI-compatible adapter drops it, so vllm answer/judge runs aren't reproducible.
- **A turn that fails all retries is dropped from the report, silently changing n**: when `run_turn` exhausts its retries (observed 2026-06-15: groq `llama-3.1-8b-instant` echoed the `[recalled from …]` injection scaffold instead of answering, `fake_blocks=2`, all 3 attempts failed), the item is excluded from `summaries`/`items` entirely rather than recorded as a failed/zero row. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh opencaw -- --only-mode off --concurrency 1` — `qa_018_curation_crate` errored and the report carried 29 items, not 30. Effect: item-count differs run-to-run, so a paired A/B over two runs silently compares different item sets unless intersected by hand.

## Retrieval

- **Recall-on buries gold in load order** (recall regression — "it used to help, now it doesn't"): two mechanisms seen in opencaw traces — (1) `max_chunks_per_source=3` lets three chunks of one (often non-gold) file flood the head of the loaded set; (2) thinking-trace re-query drift loads a different set than the raw user query. Some items load gold yet still answer wrong, suggesting the extra content distracts. Investigate initial-load selectivity and per-source head ordering.
- **Search-candidates mode is non-functional one-shot** (was QA B4): the system prompt tells the model to "mention files you want loaded," but nothing honors that request, so the model answers from discovery metadata alone — structurally plausible but ungrounded. It's an architectural gap: search-candidates is a dialog protocol that needs a tool-call loop to be useful; one-shot it is worse than loading the top-k directly. `dynamic.rs` already bypasses it for `wants_explanation` queries (loads stubs directly); the general dead-end remains.

## Consolidation

- **Consolidation notes nest recursively** (was QA B9, medium): each eviction appends the prior note's full text as the new note's "topic," so a frequently-evicted stub's consolidation header grows until it exceeds its own content. The cap of 2 displayed notes preserves display budget but loses historical signal.

## Indexing

- **Indexing batch size is a corpus-dependent magic number that OOMs on long chunks** (medium): `caw-bench/src/runner.rs` fixes the embedding batch at a hand-tuned count, but BGE attention memory scales as `batch × seq_len²`, so a fixed *count* OOMs on long (512-token) chunks while wasting capacity on short ones. `build-index.sh` papers over it with tiny batch sizes chosen to never OOM rather than to fit the workload. Fix: token-budget batching — accumulate texts until `batch_len × max_seq_len_in_batch²` would exceed a configurable budget, then flush — applied to the embedder's internal sub-batching too, not just the bench runner. (Found 2026-05-31.)
