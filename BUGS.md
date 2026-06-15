# Bugs

Open bugs only — symptom, reproduction, and evidence. No cause speculation, no fix proposals; those go in commits, `docs/DECISIONS.md`, or the ROADMAP. Fixed bugs are not tracked here.

Some QA-era findings (loops 0001–0021) remain in `docs/archive/findings.md` rather than here: B0/B1/B2 (degenerate-output era, not re-verified) and B7/B10 (prior-session responses outcompeting stubs).

## Reproducibility

- **Answer-model nondeterminism at `--concurrency > 1`**: with `temperature 0` and a fixed `--seed`, batched/concurrent inference is still not bit-reproducible run-to-run; the same item can yield different answers. Reproducible measurement requires `--concurrency 1`. Measured 2026-06-14: 23/29 recall-on answers differed across two concurrency-4 runs.
- **`seed` ignored on the vllm / `OpenAiCompatibleAdapter` path**: `AdapterSpec.seed` is honored only by Ollama and Groq; the OpenAI-compatible adapter drops it, so vllm answer/judge runs aren't reproducible.
- **A turn that fails all retries is dropped from the report, silently changing n**: when `run_turn` exhausts its retries the item is excluded from `summaries`/`items` entirely rather than recorded as a failed/zero row. Observed 2026-06-15: groq `llama-3.1-8b-instant` echoed the `[recalled from …]` injection scaffold instead of answering (`fake_blocks=2`), all 3 attempts failed. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh opencaw -- --only-mode off --concurrency 1` — `qa_018_curation_crate` errored and the report carried 29 items, not 30.

## Retrieval

- **Recall-on scores lower than recall-off on exact-fact lookups** (found 2026-06-15): on `code-agent` items asking for a specific constant value or method signature, `recall_off` answers correctly by quoting the file body while `recall_on` answers "the context does not contain it" — both report `recall@k=1.0`. In `recall_on` the gold path is resident only as a stub (no body) at the answer turn. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh code-agent -- --concurrency 1`. Evidence (`target/caw-dev/codeagent-sweep.json`): `ca_012_max_chunks_per_source` and `ca_005_vectorindex_search_sig` each score `recall_on`=0.0 / `recall_off`=1.0 with `recall@k`=1.0 in both modes; the `recall_on` answer cites a `:stub` locator. Workspaces measured 1795–1998 tokens against the bench `max_workspace_tokens` default of 2000.

- **`recall@k` reports a hit for a stub-only resident path** (found 2026-06-15): `retrieval_metrics` (`crates/caw-bench/src/runner.rs`) matches loaded paths against expected paths without distinguishing stub residency from body residency, so it reports `recall@k=1.0` for items whose body was evicted before the answer turn. Same repro/evidence as the entry above.

- **Search-candidates mode answers without any file body loaded** (was QA B4): the system prompt instructs the model to "mention files you want loaded," but no code path honors that request. `dynamic.rs` bypasses it for `wants_explanation` queries (loads stubs directly); other queries reach it.

## Consolidation

- **Consolidation notes nest recursively** (was QA B9): each eviction appends the prior note's full text as the new note's "topic," so a frequently-evicted stub's consolidation header grows until it exceeds its own content.

## Eval harness

- **sysdoc bench runs with unreadable bodies — BM25 empty, answers ungrounded** (found 2026-06-15): for the `sysdoc` workload the bench opens the prebuilt index with `corpus_root` defaulted to the opencaw repo root (`cli.repo_root`), but sysdoc stub paths (e.g. `HTML/ca/kcontrol/desktopthemedetails/index.cache`) are relative to the doc root the index was built from, which does not exist under the repo. `store.get_content` fails for every stub; BM25 builds 0 docs and the answer model receives no body text (answer_score ~0.066). bench.sh passes no corpus-root for sysdoc and there is no flag to point it at the sysdoc source tree. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh sysdoc -- --only-mode off --concurrency 1` — log shows `BM25 build: 43232 stubs had no readable stub/body and were skipped` then `built BM25 lexical index: 0 docs`. Evidence: `target/caw-dev/hybrid-off-head.log`.

## Indexing

- **Indexing embedding batch is a fixed count that OOMs on long chunks** (found 2026-05-31): `caw-bench/src/runner.rs` fixes the embedding batch at a constant count; BGE attention memory scales as `batch × seq_len²`, so a fixed count OOMs on long (512-token) chunks. `build-index.sh` uses tiny batch sizes to avoid the OOM.
