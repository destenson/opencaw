# Bugs

Open bugs only — symptom, reproduction, and evidence. No cause speculation, no fix proposals; those go in commits, `docs/DECISIONS.md`, or the ROADMAP. Fixed bugs are not tracked here.

Some QA-era findings (loops 0001–0021) remain in `docs/archive/findings.md` rather than here: B0/B1/B2 (degenerate-output era, not re-verified) and B7/B10 (prior-session responses outcompeting stubs).

See also: `TODO.md` for non-bug work items, and `docs/ROADMAP.md` for planned features, and `docs/README.md` for project documentation.

## Reproducibility

- **Answer-model nondeterminism at `--concurrency > 1`**: with `temperature 0` and a fixed `--seed`, batched/concurrent inference is still not bit-reproducible run-to-run; the same item can yield different answers. Reproducible measurement requires `--concurrency 1`. Measured 2026-06-14: 23/29 recall-on answers differed across two concurrency-4 runs.
- **`seed` ignored on the vllm / `OpenAiCompatibleAdapter` path**: `AdapterSpec.seed` is honored only by Ollama and Groq; the OpenAI-compatible adapter drops it, so vllm answer/judge runs aren't reproducible.
- **A turn that fails all retries is dropped from the report, silently changing n**: when `run_turn` exhausts its retries the item is excluded from `summaries`/`items` entirely rather than recorded as a failed/zero row. Observed 2026-06-15: groq `llama-3.1-8b-instant` echoed the `[recalled from …]` injection scaffold instead of answering (`fake_blocks=2`), all 3 attempts failed. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh opencaw -- --only-mode off --concurrency 1` — `qa_018_curation_crate` errored and the report carried 29 items, not 30.

## Retrieval

- **Explanation/internals query is answered from stub outlines only; a resident stub is never upgraded to its body** (found 2026-06-16): on a `caw-cli` turn where the model (llama3.2:3b) emits no `<probe>` tag and no `path:line` reference, the workspace holds only stub summaries and no body-loading path fires, so the answer is parroted from the stub outline text. Repro: `docs/skills/caw-dev/scripts/test-cli.sh -q "which struct owns the multi-pass recall loop and how does it decide what to evict?"` prints `[workspace: 2 fragments, ~114 tokens]`; add `--save-prompt` (per SKILL.md) and every dumped fragment is a stub summary (an outline naming the struct), not a body. The model names the struct by echoing that outline, but the "how does it decide what to evict" clause is ungrounded because no fragment body is ever made resident.

- **Recall-on scores lower than recall-off on exact-fact lookups under a tight budget** (found 2026-06-15): on `code-agent` items asking for a specific constant value or method signature, at `--max-workspace-tokens 2000` `recall_off` answers correctly while `recall_on` answers "the context does not contain it". Two observed sub-cases. (a) Gold resident only as a stub: `recall_at_k`=0.0 but `stub_recall_at_k`=1.0, answer cites a `:stub` locator (`ca_005_vectorindex_search_sig`, and 7/13 on-mode items in `target/caw-dev/codeagent-on-trace.jsonl`). (b) Gold body resident but the answer-bearing chunk of a large multi-chunk file not admitted: `recall_at_k`=1.0 at file granularity yet the constant is absent (`ca_012_max_chunks_per_source` — 7 of 29 `lib.rs` chunks loaded, not the one with the constant). Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh code-agent -- --concurrency 1 --max-workspace-tokens 2000`. At the 12000 default both sub-cases mostly disappear (`target/caw-dev/codeagent-12k.json`).

- **Search-candidates mode answers without any file body loaded** (was QA B4): the system prompt instructs the model to "mention files you want loaded," but no code path honors that request. `dynamic.rs` bypasses it for `wants_explanation` queries (loads stubs directly); other queries reach it.

## Consolidation

- **Consolidation notes nest recursively** (was QA B9): each eviction appends the prior note's full text as the new note's "topic," so a frequently-evicted stub's consolidation header grows until it exceeds its own content.

## Eval harness

- **sysdoc bench runs with unreadable bodies — BM25 empty, answers ungrounded** (found 2026-06-15): for the `sysdoc` workload the bench opens the prebuilt index with `corpus_root` defaulted to the opencaw repo root (`cli.repo_root`), but sysdoc stub paths (e.g. `HTML/ca/kcontrol/desktopthemedetails/index.cache`) are relative to the doc root the index was built from, which does not exist under the repo. `store.get_content` fails for every stub; BM25 builds 0 docs and the answer model receives no body text (answer_score ~0.066). bench.sh passes no corpus-root for sysdoc and there is no flag to point it at the sysdoc source tree. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh sysdoc -- --only-mode off --concurrency 1` — log shows `BM25 build: 43232 stubs had no readable stub/body and were skipped` then `built BM25 lexical index: 0 docs`. Evidence: `target/caw-dev/hybrid-off-head.log`.

## Indexing

- **Indexing embedding batch is a fixed count that OOMs on long chunks** (found 2026-05-31): `caw-bench/src/runner.rs` fixes the embedding batch at a constant count; BGE attention memory scales as `batch × seq_len²`, so a fixed count OOMs on long (512-token) chunks. `build-index.sh` uses tiny batch sizes to avoid the OOM.

- **Incremental refresh does not heal a partially-deleted file** (found 2026-06-16): the resume skip-list `indexed_paths()` is `SELECT DISTINCT path, mtime_unix_secs FROM stubs WHERE stale = 0` (`crates/caw-index/src/storage/sqlite_store.rs`), keyed per source file, but stubs are per-chunk. If some-but-not-all of a file's chunk rows are missing while ≥1 non-stale chunk for that path remains, the file's `(path, mtime)` is still in the skip-list, so `build-index.sh --no-rebuild` (and therefore every `serve.sh` refresh) skips the file and the missing chunks are never re-embedded; only a full `--rebuild` restores them. A file whose rows are entirely absent, or all `stale=1`, is re-ingested correctly. Repro: build an index over a corpus, then `sqlite3 <index> "DELETE FROM stubs WHERE id=(SELECT id FROM stubs WHERE path='<a-multi-chunk-file>' LIMIT 1);"`, then re-run `build-index.sh --no-rebuild <corpus> <index>` — it logs `nothing to do: all files already indexed` and the total stub count stays one short.
