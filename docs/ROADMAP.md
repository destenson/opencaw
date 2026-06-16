# Roadmap — where we are and what's next

**Status:** Living — current focus and sequence. This is the *highest-churn* doc in the project: it is expected to change as work progresses, and it does not need to be stable or perfect. If it contradicts reality, fix it.

This doc answers one question no other doc does: **"where are we, what's next, and why that order?"** It is not a backlog (that's [TODO.md](../TODO.md), deliberately unordered), not a defect log ([BUGS.md](../BUGS.md)), and not settled history ([DECISIONS.md](DECISIONS.md)). It carries *sequence, current position, and the forcing logic* that makes the order non-arbitrary.

The rule that keeps this doc honest: every step is tied to **what it unblocks**, not just "do X then Y." A step ordered only by preference rots into a wishlist. A step ordered by dependency ("Y can't be *evaluated* until X holds") stays meaningful.

---

## Current milestone: dogfood OpenCAW as a coding-agent context server

Drive a real coding agent through `caw-server` against this repo's own index and have it be good enough to use daily: recall surfaces the right content for mid-task information needs (signatures, trait bounds, struct fields, call sites) without burying it, at acceptable latency. "Done" means the product is good enough that we actually use it. The `code-agent` recall-on/off numbers are a QA regression gauge, not evidence — when recall-on answers a lookup wrong, that's a defect to fix.

## Where we are now

**Retrieval wiring done; the dogfood surface is clean at a realistic budget.** Bench/cli now share the proxy's hybrid retriever, and a deterministic `code-agent` sweep over this repo says recall-on and recall-off are indistinguishable on the QA set once the workspace budget is realistic.

- ✓ Judge decoupled from generation (two-phase eval) — `5b45713`; paired per-item deltas — `e309486`; standalone re-judge (`--judge-trace`).
- ✓ Serial path verified deterministic; groq answer model (`CAW_BENCH_ANSWER=groq`, smallest model) runs ~2–3s/item (bills per token).
- ✓ Bench/cli wired onto `HybridRetriever` to match the proxy — `0bb14f5`.
- ✓ Deterministic `code-agent` sweep over this repo (2026-06-15, `--concurrency 1`, groq answer). At the bench default `max_workspace_tokens=2000`: recall-on lost on two exact-fact items (`ca_005`, `ca_012`) because eviction reduced the gold body to a stub before the answer turn. At `--max-workspace-tokens 12000`: both flip to on=off=1.0 and the set goes flat (answer_score 0W/12T/1L, Δ−0.077). The 2k regressions were the eviction-microscope budget, not a product defect.
- Standing observations from that sweep: `recall@k` +0.115 but `precision@1`/`mrr` −0.23 — recall-on surfaces more gold but ranks it lower; `recall@k` counts a stub-only resident path as a hit (BUGS "Retrieval").

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

### 3. Progressive disclosure — *second, the durable fix for genuinely-constrained corpora*

On a corpus too large to fit (the case OpenCAW exists for), the workspace *will* be under real pressure and stubs *will* be evicted. When a pinpoint query then hits a file resident only as a stub, the loop should re-upgrade that stub to full content in place rather than answer from the stub. Design committed (DECISIONS "Progressive disclosure"): the upgrade replaces the early-return in `load_fragments`' already-resident branch (and the line-reference path) for `LoadMode::Full`, reads residency from the fragment locator, and respects three invariants (provenance-invertible, query-referenced retention, budget-monotonic). **Not yet implemented — this is the next code.** The `stub_recall_at_k − recall_at_k` gap on the 2k sweep is the standing reproduction and the measure of what it buys.

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

Next: **implement progressive disclosure** (step 3) per the committed design — turn `load_fragments`' already-resident early-return into a stub→body upgrade for `LoadMode::Full`, honoring the three invariants. Verify with the 2k `code-agent` sweep: the `stub_recall_at_k − recall_at_k` gap should close as stubs get upgraded.

## How the docs relate

- **ROADMAP.md** (this) — sequence, current position, forcing logic. Changes often.
- **[TODO.md](../TODO.md)** — unordered backlog of concrete work items.
- **[BUGS.md](../BUGS.md)** — open defects with reproductions.
- **[DECISIONS.md](DECISIONS.md)** — settled implementation choices; authoritative when something is ambiguous.
- **[origin.md](origin.md) / [scope.md](scope.md)** — the founding design (frozen) and what v0.1 is/isn't.
