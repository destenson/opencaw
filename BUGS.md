# Bugs

Open bugs only — symptom, reproduction, and evidence. No cause speculation, no fix proposals; those go in commits, `docs/DECISIONS.md`, or the ROADMAP. Fixed bugs are not tracked here.

Some QA-era findings (loops 0001–0021) remain in `docs/archive/findings.md` rather than here: B0/B1/B2 (degenerate-output era, not re-verified) and B7/B10 (prior-session responses outcompeting stubs).

## Reproducibility

- **Answer-model nondeterminism at `--concurrency > 1`**: with `temperature 0` and a fixed `--seed`, batched/concurrent inference is still not bit-reproducible run-to-run; the same item can yield different answers. Reproducible measurement requires `--concurrency 1`. Measured 2026-06-14: 23/29 recall-on answers differed across two concurrency-4 runs.
- **`seed` ignored on the vllm / `OpenAiCompatibleAdapter` path**: `AdapterSpec.seed` is honored only by Ollama and Groq; the OpenAI-compatible adapter drops it, so vllm answer/judge runs aren't reproducible.
- **A turn that fails all retries is dropped from the report, silently changing n**: when `run_turn` exhausts its retries the item is excluded from `summaries`/`items` entirely rather than recorded as a failed/zero row. Observed 2026-06-15: groq `llama-3.1-8b-instant` echoed the `[recalled from …]` injection scaffold instead of answering (`fake_blocks=2`), all 3 attempts failed. Repro: `CAW_BENCH_ANSWER=groq bash docs/skills/caw-dev/scripts/bench.sh opencaw -- --only-mode off --concurrency 1` — `qa_018_curation_crate` errored and the report carried 29 items, not 30.

## Observability

- **`caw-cli` emits no `tracing` log output; `RUST_LOG` has no effect** (found 2026-06-16): the diagnostic lines `caw-cli` prints on stderr are plain prints (no timestamp, level, target, or `file:line` location), and no `tracing`-formatted event is ever emitted at any `RUST_LOG` setting — so the orchestrator's own load/evict/probe/recall decisions are unreachable. `docs/skills/caw-dev/SKILL.md` documents `RUST_LOG=caw_orchestrator=debug` as the way to see recall decisions and claims an explicit `RUST_LOG` is honored verbatim. Repro: `printf 'what is opencaw?\n' | RUST_LOG=caw_orchestrator=debug,caw_core=debug docs/skills/caw-dev/scripts/run-cli.sh --dir crates --adapter ollama --model llama3.2:3b --no-intent-classifier 2>&1 | grep -E '[a-z_]+\.rs:[0-9]+'` prints nothing — a working subscriber would emit lines carrying a `file.rs:NN` location.

## Intent classification

- **Intent classifier output is dropped to empty and leaks a malformed guidance line** (found 2026-06-16): with `--intent-model granite4:micro`, an explanatory query yields an all-empty structured intent (`[intent] QueryIntent()` on the terminal) while an unrecognized classifier key (`is_explanation`) is rendered into the prompt verbatim as the guidance line `Additional context: is explanation.`. Net effect: the intent is treated as empty so explanation-query handling never fires, and a meaningless line is injected into the model's context. Repro: `docs/skills/caw-dev/scripts/test-cli.sh -q "which struct owns the multi-pass recall loop and how does it decide what to evict?"` prints `[intent] QueryIntent()`; add `--save-prompt` (per SKILL.md) and the dumped prompt's guidance section contains `Additional context: is explanation.`.

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
