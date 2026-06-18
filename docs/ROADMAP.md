# Roadmap — where we are and what's next

**Status:** Living — current focus and sequence. This is the *highest-churn* doc in the project: it is expected to change as work progresses, and it does not need to be stable or perfect. If it contradicts reality, fix it.

This doc answers one question no other doc does: **"where are we, what's next, and why that order?"** It is not a backlog (that's [TODO.md](../TODO.md), deliberately unordered), not a defect log ([BUGS.md](../BUGS.md)), and not settled history ([DECISIONS.md](DECISIONS.md)). It carries *sequence, current position, and the forcing logic* that makes the order non-arbitrary.

The rule that keeps this doc honest: every step is tied to **what it unblocks**, not just "do X then Y." A step ordered only by preference rots into a wishlist. A step ordered by dependency ("Y can't be *evaluated* until X holds") stays meaningful.

---

## Current milestone: dogfood OpenCAW as a coding-agent context server

Drive a real coding agent through `caw-server` against this repo's own index and have it be good enough to use daily: recall surfaces the right content for mid-task information needs (signatures, trait bounds, struct fields, call sites) without burying it, at acceptable latency. "Done" means the product is good enough that we actually use it. The `code-agent` recall-on/off numbers are a QA regression gauge, not evidence — when recall-on answers a lookup wrong, that's a defect to fix.

## Where we are now

**Retrieval wiring done; the trace-driven loop is now exercisable but its net effect on answer quality is not yet a win.** Bench/cli share the proxy's hybrid retriever. A 2026-06-18 re-investigation found that the prior "recall-on and recall-off are indistinguishable" result was not because the surface is clean — it was because the bench was running the answer model non-cooperative, which switches the entire trace-driven loop off, so both modes collapsed to basic-RAG-from-initial-stubs. With cooperation enabled the loop fires and progressive disclosure works mechanically, but it trades one fixed item for two regressed ones (net answer_score slightly down at the 2k study budget). Details below.

- ✓ Judge decoupled from generation (two-phase eval) — `5b45713`; paired per-item deltas — `e309486`; standalone re-judge (`--judge-trace`).
- ✓ Serial path verified deterministic; groq answer model (`CAW_BENCH_ANSWER=groq`, smallest model) runs ~2–3s/item (bills per token).
- ✓ Bench/cli wired onto `HybridRetriever` to match the proxy — `0bb14f5`.
- ✓ Deterministic `code-agent` sweep over this repo (2026-06-15, `--concurrency 1`, groq answer). At the bench default `max_workspace_tokens=2000`: recall-on lost on two exact-fact items (`ca_005`, `ca_012`) because eviction reduced the gold body to a stub before the answer turn. At `--max-workspace-tokens 12000`: both flip to on=off=1.0 and the set goes flat (answer_score 0W/12T/1L, Δ−0.077). The 2k regressions were the eviction-microscope budget, not a product defect. **Caveat added 2026-06-18:** that run's `recall_at_k` used the metric *before* the body/stub split (step 4) landed later the same day, so its `recall_at_k` numbers are not comparable to the current body-recall metric — treat the 2026-06-15 retrieval numbers as path-level, not body-level.
- **2026-06-18 — the loop was inert because the answer model ran non-cooperative (the real blocker, not PD).** The bench builds the orchestrator with `cooperation_mode` unset → `CooperationMode::Auto` (`DynamicRecallConfig::default`). Under `Auto`, the cooperation instructions (`<probe>`, `<note>`, the line-range `path:start-end` syntax, "think out loud") are injected only if the adapter reports a reasoning capability (`dynamic.rs:370-373`); the groq adapter reports none (`openai_compatible.rs:120-121`), so for groq the system prompt is just the base + "always respond" requirement and the model is never told it can ask for more context. With no probes, no line-refs, and the candidate-list mention pass disabled in the bench (`runner.rs:580`), no `LoadMode::Full` load ever fires, so `upgrade_stub_to_body` (progressive disclosure) never runs. Verified by `RUST_LOG=caw_orchestrator::dynamic=debug`: `probes extracted = 0` on every turn at both 2k and 12k. This gating has been in place since 2026-05-03 (`603b896`, `de0778e`), so the 2026-06-15 baseline ran the same way — its body-recall was never actually being tested. `caw-bench-coop` (`54874eb`) confirmed llama-3.1-8b-instant *does* cooperate when instructed (1–2 probes/turn in `cooperative` mode, 0 in `baseline`/`transparent`), so the fix is to run the bench answer model cooperative, not to change the model.
- **2026-06-18 — with cooperation enabled, PD fires and the loop is exercisable, but net answer_score is still slightly negative at 2k.** Setting `cooperation_mode: Cooperative` for the bench recall-on path and re-running the 2k `code-agent` sweep: `recall_at_k` (body-recall) 0.071 → 0.615, `precision@1` 0.071 → 0.462, `mrr` 0.071 → 0.538; 22 progressive-disclosure upgrades fired (was 0). `ca_005` fixed (0.00 → 0.90 — the stub got upgraded and the model got the signature). But `mean_answer_score` went 0.429 → 0.377: cooperation appeared to trade `ca_005` (fixed) for `ca_006` + `ca_008` (both 1.00 → 0.00). **That "trade" was a scoring artifact, not loop behavior** — see the `d33ee97` entry under "Single next action": the off=1.00s were false scores from bare-`contains`/always-judge, so the 0.429→0.377 delta was measured with broken scoring and is retired, not directional. Artifacts in `target/caw-dev/codeagent-{pdisc,coop}-2k.*`. `cooperation_mode: Cooperative` is now committed (`0d03fae`, `runner.rs:610`); it is a bench-only toggle for exercising the loop, not a product default.

---

## The critical path (ordered, with forcing rationale)

### 1. Instrument trust — *first, because a recall change is unfalsifiable without it*

The eval currently can't resolve the effect we'd be optimizing: `answer_score` swung 0.461→0.338 on *identical items* between two runs (TODO "Measurement"), and at `--concurrency > 1` answers aren't reproducible run-to-run (BUGS "Reproducibility"). Optimizing recall against an instrument this noisy is guessing. So:

- **Decouple judge from generation** ✓ — judging no longer blocks or contaminates the generation loop (DECISIONS, two-phase eval).
- **Paired per-item deltas** ◐ — compare on vs off *on the same item*, so item difficulty (usually the dominant variance) cancels and the standard error shrinks. Same point estimate as diff-of-means; far tighter confidence.
- **Standalone re-judge** ✓ — `--judge-trace <trace.jsonl>` scores persisted answers against any judge without regenerating, so judge noise can be isolated and several judge passes averaged (the two-phase split laid the groundwork).
- **Serial path is the deterministic measurement path** ✓ — verified bit-identical across two `--concurrency 1` runs. Multi-seed generation was considered and rejected: at concurrency 1 + temp 0 the answer is deterministic, so seeds add no signal (DECISIONS).
- **Fast (not free) large-n via groq answer model** ✓ — `CAW_BENCH_ANSWER=groq` in `bench.sh` runs ~2–3s/item (vs tens of seconds locally), defaulting to the smallest groq model. Removes wall-time as the blocker on the real power lever (more QA items) — but groq bills per token, so use the smallest model and bounded runs; don't run the full set casually.
- **Larger-n measurement + judge averaging** ☐ — run on the deterministic path, average judge passes via `--judge-trace`, and expand the QA item set. **← next**

### 2. Workspace budget — *first, because the bench's study budget is not a dogfood budget*

The bench pins `max_workspace_tokens=2000` on purpose, to force eviction to fire so the curation machinery is observable (see the comment on `RunnerConfig::default`). A coding agent dogfooding `caw-server` has a much larger context budget. At 2k, eviction throws away gold bodies the answer needs; at 12k it does not. So the server needs a realistic default workspace budget, set independently of the bench's deliberately-tight study setting, before any of the curation behavior below is tuned against a representative workload.

### 3. Progressive disclosure — *implemented; mechanically sound; exercisable only when the answer model runs cooperative*

On a corpus too large to fit (the case OpenCAW exists for), the workspace *will* be under real pressure and stubs *will* be evicted. When a pinpoint query then hits a file resident only as a stub, the loop should re-upgrade that stub to full content in place rather than answer from the stub. Design committed (DECISIONS "Progressive disclosure") and **implemented (`f11146e`, `upgrade_stub_to_body` at `dynamic.rs:1610`)**: the upgrade replaces the early-return in `load_fragments`' already-resident branch (`dynamic.rs:1454`) and the line-reference path (`dynamic.rs:1199`) for `LoadMode::Full`, reads residency from the fragment locator, and respects three invariants (provenance-invertible, query-referenced retention, budget-monotonic).

**Tested 2026-06-18.** Mechanically the upgrade is sound: with the answer model run cooperative, 22 upgrades fired and the flagship case (`ca_005`, gold resident only as a stub) flipped from answer_score 0.00 to 0.90. But two cautions, both important:

1. **PD is inert unless the answer model runs cooperative.** The upgrade only fires under `LoadMode::Full`, and Full loads only come from probe emission, line-references, or the candidate-list mention pass — all three are gated on the cooperation instructions being injected, which under `CooperationMode::Auto` only happens for adapters that report a reasoning capability. The bench's groq answer model reports none, so in the default bench config PD never fires (0 upgrades, gap unchanged). See "Where we are now" for the full chain. The `stub_recall_at_k − recall_at_k` gap is therefore *not* a measure of what PD buys until the loop is actually exercising.
2. **The 2k "enabling the loop is a net loss" reading was a scoring artifact, not loop behavior.** The `ca_006`/`ca_008` 1.00→0.00 "regressions" and the `mean_answer_score` 0.429→0.377 were measured with the pre-`d33ee97` scoring, which over-credited the off answers (a quote-while-denying refusal, a semantic near-miss). With two-phase needle scoring those off answers are correctly 0.0, so the "trade" framing is retired (DECISIONS, "`code-agent` exact-fact scoring"). What remains genuinely open is *what the loop does once it fires* — see `ca_005` on at 12k under "Single next action": the loop hedges instead of answering once the gold body is resident. The degenerate-output abort that used to drop items (and crash `caw-bench-coop` mid-run) is fixed: a degenerate turn is now recorded as a zero-scored row, not dropped (DECISIONS, "A degenerate turn is recorded as a zero row").

### 4. Ranking — *third*

- ✓ `recall@k` now counts only body residency; `stub_recall_at_k` reports the path-level number and the gap is the disclosure headroom (`retrieval_metrics`, report). The metric no longer hides stub-only gold.
- `precision@1`/`mrr` run below recall-off on the `code-agent` sweep while `recall@k` runs above it: recall-on surfaces more gold but ranks it lower. The `divide_total` hybrid fusion is a placeholder (TODO); ranking is where to spend effort once disclosure is settled. (Do **not** swap to RRF: measured worse at every depth ≤20.)

### 5. Latency / generation reduction

Generation dominates wall time. A live agent loop needs it down: configurable `num_predict`; skip the refinement completion when the first answer is already grounded; a batching serve stack (vLLM/TensorRT). Gen-reduction changes are gated on the quality bar in DECISIONS "Performance optimization".

### 6. Dogfood — *the destination*

Drive a real agent through `caw-server` against the repo index at a realistic budget; expand the thin `code-agent` QA set (14 items); run the recall-on/off sweep on real agent information-needs as a standing regression gauge.

---

## Single next action

✓ Done (2026-06-15): server budget 2000→12000 (DECISIONS "Default workspace token budget"); `recall@k` split into body-recall vs `stub_recall_at_k` so the metric no longer counts stub-only gold as a hit (step 4); progressive-disclosure design committed (DECISIONS "Progressive disclosure").

✓ Done (2026-06-18): progressive disclosure **implemented** (`f11146e`) and **tested**. It is mechanically sound but inert under the bench's default `Auto`+non-reasoning-adapter config; with the answer model run cooperative it fires (22 upgrades, `ca_005` fixed). The 2k `mean_answer_score` 0.429→0.377 "regression" reported then was measured with the pre-d33ee97 scoring (see next entry) and is retired, not signal.

✓ Done (2026-06-17, `0d03fae`): cooperation default settled and committed — `runner.rs:610` sets `CooperationMode::Cooperative` for the recall-on path (DECISIONS, "Cooperative vs. transparent mode selection"). The bench always runs the loop now, so recall-on/off sweeps measure the loop rather than basic-RAG-in-both-modes. This is a bench-only toggle for exercising the loop, not a product default.

✓ Done (2026-06-17, `d33ee97`): two-phase needle scoring for `code-agent` exact-fact items (DECISIONS, "`code-agent` exact-fact scoring is two-phase"). This **invalidates the prior "the loop breaks `ca_006`/`ca_008`" reading**: those items' recall-off 1.00s were false scores — a quote-while-denying refusal (`ca_008`, bare `contains` over-credited the token mentioned inside a denial) and a semantic near-miss (`ca_009`, the judge credited `summaries` for the field `stub_summaries`). With corrected scoring, `ca_006`/`ca_008`/`ca_009` off are deterministic 0.0 (needle genuinely absent), which is correct, not a loop regression. Verified on a 12k cooperative sweep: all three scoring-fix targets met.

✓ Done (2026-06-17, `0171592`): fixed an instrument regression the abort-fix `bd79755` introduced — `orchestrator.loaded` was cloned *before* `run_turn` (the loop loads inside the call), so `loaded_paths`/`recall_at_k`/`content_tokens` were zeroed for every item since `bd79755`. Moved the clone to after the match in both `run_item_fresh`/`run_item_shared`. A 12k cooperative code-agent run goes `mean_recall_at_k` 0.0→1.0. Invariant recorded in DECISIONS ("Loaded-set capture invariant"). Any bench result saved between `bd79755` and `0171592` has zeroed recall metrics (generation-side fields were unaffected).

✓ Done (2026-06-17): diagnosed the `ca_005` hedge — it is **generation-side, not recall**. With the instrument fixed, a 12k cooperative recall-on run shows the loop doing its job: the gold (`crates/caw-core/src/lib.rs#chunk54`, the `VectorIndex` trait) is loaded, upgraded to a body via PD, and **resident at the answer turn** (`workspace_frags=32, workspace_tokens=4527`, never evicted). `format_workspace` renders it verbatim as `[recalled from …]` quoted source, so the model receives the exact `fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(StubId, f32)>`. The hedge is the model obeying the `response_requirement` baked into every system prompt (`dynamic.rs:353`): "If retrieved evidence is insufficient, say so explicitly and describe what is missing." The workspace also contains implementors whose `search` takes `query` (not `query_embedding`); the small groq answer model (`llama-3.1-8b-instant`) reads that trait-vs-impl discrepancy as "insufficient / not fully specified" and obeys the instruction — "the full signature… is not provided… additional context is needed" (0.0). The cooperation instructions ("think out loud," emit `<probe>` when "missing something") reinforce the posture. **The loop delivers the gold; the hedge is downstream of it.**

**Direction (2026-06-17, user):** opencaw should be as passive as possible; avoid requiring active participation/compliance of the model and its responses. The `ca_005` hedge is a compliance failure, and the cooperation protocol (`<probe>`, `<note>`, line-refs, "think out loud," the `response_requirement` itself) is the active-compliance machinery that produced it. The core novelty — thinking-trace-as-retrieval-signal — does not inherently require protocol compliance; the signal could be read from the model's *natural* output (mentioned symbols/paths, hedging language) without requiring it to emit tags or follow instructions it may not follow. But the loop today only fires *through* the cooperation instructions (under `CooperationMode::Auto` a non-reasoning adapter gets no injection → inert → basic RAG; the bench forces `Cooperative` to make it fire), so a passive recall path does not yet exist. Building one is the work this direction implies.

**Lever #1 (reframe `response_requirement`) was tried and failed (2026-06-17).** Reworded to "assert what the cited source states; only describe what's missing when the cited source genuinely does not contain the answer." Result: did **not** fix the hedge — `ca_005` still 0.0 (the model looked at `lib.rs:55` and still said "the trait definition is not shown"), and mean `recall_at_k` regressed 1.0→0.8 with one degenerate turn. Reverted and discarded. Negative result is itself signal: **you cannot instruct a small model into compliance by rewording the prompt.** This retires lever #1 and casts doubt on lever #2 (tone down cooperation instructions) for the same reason — if the model won't obey a reworded requirement, it won't obey a toned-down one either. Lever #3 (a stronger answer model asserting from the same quoted source, isolating small-model over-caution from prompt-induced hedging) remains untried and would cost groq tokens.

**Status: this direction is recorded as a goal, not started.** It needs scoping before any implementation — in particular, how passive the recall signal should be (read natural output only / natural output plus silent reactive retrievals / keep the protocol but make it optional) and which surface to build the passive path on first (bench / `caw-server` / `caw-core`). The previous session's attempt to scope it was set aside as too much of a divergence from the current dogfooding work; it is parked here until the user wants to hash it out. Do not begin building passive recall without that scoping conversation.

Adjacent, still gated on the now-trustworthy instrument and independent of the passivity direction: (a) expand the thin `code-agent` QA set (14 items) for gauge power; (b) judge calibration on near-correct signatures — moot for `ca_005` until the model actually answers instead of hedging, which the passivity direction would address.

## How the docs relate

- **ROADMAP.md** (this) — sequence, current position, forcing logic. Changes often.
- **[TODO.md](../TODO.md)** — unordered backlog of concrete work items.
- **[BUGS.md](../BUGS.md)** — open defects with reproductions.
- **[DECISIONS.md](DECISIONS.md)** — settled implementation choices; authoritative when something is ambiguous.
- **[origin.md](origin.md) / [scope.md](scope.md)** — the founding design (frozen) and what v0.1 is/isn't.
