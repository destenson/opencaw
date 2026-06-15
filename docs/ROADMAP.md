# Roadmap — where we are and what's next

**Status:** Living — current focus and sequence. This is the *highest-churn* doc in the project: it is expected to change as work progresses, and it does not need to be stable or perfect. If it contradicts reality, fix it.

This doc answers one question no other doc does: **"where are we, what's next, and why that order?"** It is not a backlog (that's [TODO.md](../TODO.md), deliberately unordered), not a defect log ([BUGS.md](../BUGS.md)), and not settled history ([DECISIONS.md](DECISIONS.md)). It carries *sequence, current position, and the forcing logic* that makes the order non-arbitrary.

The rule that keeps this doc honest: every step is tied to **what it unblocks**, not just "do X then Y." A step ordered only by preference rots into a wishlist. A step ordered by dependency ("Y can't be *evaluated* until X holds") stays meaningful.

---

## Current milestone: dogfood OpenCAW as a coding-agent context server

Drive a real coding agent through `caw-server` against this repo's own index, and have recall measurably help the agent answer mid-task information needs (signatures, trait bounds, struct fields, call sites) better than no recall, at acceptable latency. "Done" means: recall-on beats recall-off on the `code-agent` workload by a margin the instrument can resolve, *and* end-to-end latency is low enough that a live agent loop is usable.

## Where we are now

**Instrument-trust phase.** We are making the eval able to resolve a recall change before we try to make recall changes.

- ✓ Judge decoupled from generation (two-phase eval) — `5b45713`.
- ✓ Paired per-item deltas in the report (cancel item difficulty so a real effect is resolvable) — `e309486`.
- ✓ Standalone re-judge of persisted answers (`--judge-trace`): score saved answers against any judge without regenerating.
- ☐ More seeds; reproducible serial generation path; use `--judge-trace` to average/pin judges and quantify judge variance. **← next**

---

## The critical path (ordered, with forcing rationale)

### 1. Instrument trust — *first, because a recall change is unfalsifiable without it*

The eval currently can't resolve the effect we'd be optimizing: `answer_score` swung 0.461→0.338 on *identical items* between two runs (TODO "Measurement"), and at `--concurrency > 1` answers aren't reproducible run-to-run (BUGS "Reproducibility"). Optimizing recall against an instrument this noisy is guessing. So:

- **Decouple judge from generation** ✓ — judging no longer blocks or contaminates the generation loop (DECISIONS, two-phase eval).
- **Paired per-item deltas** ◐ — compare on vs off *on the same item*, so item difficulty (usually the dominant variance) cancels and the standard error shrinks. Same point estimate as diff-of-means; far tighter confidence.
- **Standalone re-judge** ✓ — `--judge-trace <trace.jsonl>` scores persisted answers against any judge without regenerating, so judge noise can be isolated and several judge passes averaged (the two-phase split laid the groundwork).
- **More seeds + serial gen path** ☐ — add seeds and a reproducible serial generation path (concurrency>1 isn't bit-reproducible) for measurement runs.

### 2. Recall quality — *second, because this is the real capability gap but only tellable from noise once (1) holds*

Recall-on currently regresses: it "buries gold in load order" and some items load the gold chunk yet still answer wrong (BUGS "Retrieval"). This is the actual thing standing between us and a useful dogfood — but a fix is indistinguishable from noise until the instrument is trustworthy.

- First, the regression-flavored A/B the evidence points at: did turning thinking-trace recall from inert→active (`983aba4`) make recall-on *worse* than when it was inert? (The "recall-on doesn't win" symptom predates that commit, so this is attribution, not assumption.)
- Then: initial-load selectivity and per-source head ordering — the two named mechanisms (a non-gold file flooding the head via `max_chunks_per_source`; thinking-trace re-query drift loading a different set than the raw query).

### 3. Latency / generation reduction — *third, because a live agent needs speed but a gen change needs a quality bar*

Generation dominates wall time (~60–170s/item; judge is ~0.6s). Dogfooding a live agent loop needs that down. But because generation *is* the thesis mechanism, any gen-reduction change is gated on a pre-committed quality bar (DECISIONS "Performance optimization") — which requires the trustworthy instrument from (1).

- Configurable `num_predict`; skip the refinement completion when the first answer is already grounded; a batching serve stack (vLLM/TensorRT).

### 4. Dogfood — *the destination*

Drive a real agent through `caw-server` against the repo index; expand the `code-agent` QA set; run the recall-on/off sweep on real agent information-needs. This is where qualitative pain becomes the signal — but only after (1)–(3), or we'd be debugging blind.

---

## Single next action

Finish paired per-item deltas (step 1), then the standalone re-judge path — both raise the statistical power needed before touching recall (step 2).

## How the docs relate

- **ROADMAP.md** (this) — sequence, current position, forcing logic. Changes often.
- **[TODO.md](../TODO.md)** — unordered backlog of concrete work items.
- **[BUGS.md](../BUGS.md)** — open defects with reproductions.
- **[DECISIONS.md](DECISIONS.md)** — settled implementation choices; authoritative when something is ambiguous.
- **[design.md](design.md) / [scope.md](scope.md)** — the thesis and what v0.1 is/isn't.
