# Roadmap — where we are and what's next

**Status:** Living — current focus and sequence. This is the *highest-churn* doc in the project: it is expected to change as work progresses, and it does not need to be stable or perfect. If it contradicts reality, fix it.

This doc answers one question no other doc does: **"where are we, what's next, and why that order?"** It is not a backlog (that's [TODO.md](../TODO.md), deliberately unordered), not a defect log ([BUGS.md](../BUGS.md)), and not settled history ([DECISIONS.md](DECISIONS.md)). It carries *sequence, current position, and the forcing logic* that makes the order non-arbitrary.

The rule that keeps this doc honest: every step is tied to **what it unblocks**, not just "do X then Y." A step ordered only by preference rots into a wishlist. A step ordered by dependency ("Y can't be *evaluated* until X holds") stays meaningful.

---

## Current milestone: dogfood OpenCAW as a coding-agent context server

Drive a real coding agent through `caw-server` against this repo's own index and have it be good enough to use daily: recall surfaces the right content for mid-task information needs (signatures, trait bounds, struct fields, call sites) without burying it, at acceptable latency. "Done" means the product is good enough that we actually use it. The `code-agent` recall-on/off numbers are a QA check that retrieval quality is where it should be — a regression gauge, not a proof that the approach works (it does).

## Where we are now

**Fixing retrieval quality — now isolated to candidate-set recall.** The eval is trustworthy enough to use as QA, and a deterministic opencaw A/B has now located the binding constraint.

- ✓ Judge decoupled from generation (two-phase eval) — `5b45713`.
- ✓ Paired per-item deltas in the report (cancel item difficulty so a real effect is resolvable) — `e309486`.
- ✓ Standalone re-judge of persisted answers (`--judge-trace`): score saved answers against any judge without regenerating.
- ✓ Serial path verified deterministic (multi-seed gen rejected as no-signal); groq answer model (`CAW_BENCH_ANSWER=groq`, smallest model) makes large-n runs fast (~2–3s/item) though groq bills per token.
- ✓ Deterministic opencaw A/B of the coverage-first admission fix (`d71c6c4`), `--only-mode off --concurrency 1`, HEAD vs parent over 29 common items: precision@1 flat, recall@k −0.017, mrr +0.011 — **neutral-to-marginal with one regression. Mechanism 1 (load-order flooding) was not the dominant cause.** (2026-06-15; DECISIONS "Initial-load admission".)
- ✓ Located the candidate-recall gap's cause (2026-06-15): **the eval runs the wrong retriever.** `caw-bench`/cli use pure-cosine `SemanticRetriever`; the dogfood proxy uses `HybridRetriever`. Retrieval-only A/B on the sysdoc gold set: cosine→hybrid lifts recall@10 0.48→0.85, recall@20 0.54→0.92. (Also found: RRF is *worse* than the current `divide_total` fusion at every depth ≤20 — the planned RRF swap is contradicted; DECISIONS.)
- ☐ **Wire bench/cli onto the proxy's hybrid retrieval, then re-measure the opencaw sweep.** The candidate-recall ceiling (~0.67) is largely an artifact of the eval running cosine; the better retriever already exists on the proxy. This is also required by the repo principle that cli/bench/proxy share retrieval code. Expected to raise recall@k for free; only after this is embedding/chunking or admission/eviction tuning worth doing. **← next**

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

### 2. Recall quality — *second, because this is the real capability gap but only tellable from noise once (1) holds*

Recall-on currently regresses: it "buries gold in load order" and some items load the gold chunk yet still answer wrong (BUGS "Retrieval"). This is the actual thing standing between us and a useful dogfood — but a fix is indistinguishable from noise until the instrument is trustworthy.

Two deterministic measurements (2026-06-15) have narrowed this to a single first move. (a) Admission ordering is **not** the lever: the coverage-first fix for mechanism 1 was neutral-to-marginal on opencaw, because gold is missing from the candidate set entirely on ~1/3 of items — admission can only reorder what retrieval already surfaced. (b) That missing-candidate ceiling is largely because **the eval runs pure cosine while the proxy runs hybrid**; on the sysdoc gold set, hybrid lifts recall@10 from 0.48 to 0.85. So the work reorders:

- **First, make the eval use the same hybrid retriever as the proxy.** `caw-bench`/cli instantiate `SemanticRetriever` (cosine); `caw-server` uses `HybridRetriever`. Wiring bench/cli onto hybrid is expected to raise candidate recall@k sharply for free, and is required by the repo principle that cli/bench/proxy share retrieval code. Re-run the opencaw recall-on/off sweep afterward — the ~0.67 ceiling should rise.
- **Then, if recall@k is still short, embedding/chunking** — structure-aware chunking already exists on the proxy ingest path; confirm it's on the eval path too. (Do **not** swap `divide_total` fusion for RRF: measured worse at every depth ≤20.)
- **Then, precision@1 / ranking** — once gold is reliably in the candidate set, get it ranked at the top so admission and the answer model see it first.
- Deferred (measured, not the lever): mechanism 1 admission ordering (`d71c6c4`, kept but neutral); mechanism 2 thinking-trace re-query drift — only worth revisiting once candidate recall is high.

### 3. Latency / generation reduction — *third, because a live agent needs speed but a gen change needs a quality bar*

Generation dominates wall time (~60–170s/item; judge is ~0.6s). Dogfooding a live agent loop needs that down. But because generation *is* the thesis mechanism, any gen-reduction change is gated on a pre-committed quality bar (DECISIONS "Performance optimization") — which requires the trustworthy instrument from (1).

- Configurable `num_predict`; skip the refinement completion when the first answer is already grounded; a batching serve stack (vLLM/TensorRT).

### 4. Dogfood — *the destination*

Drive a real agent through `caw-server` against the repo index; expand the `code-agent` QA set; run the recall-on/off sweep on real agent information-needs. This is where qualitative pain becomes the signal — but only after (1)–(3), or we'd be debugging blind.

---

## Single next action

Wire `caw-bench`/cli onto the proxy's `HybridRetriever` (they currently use pure-cosine `SemanticRetriever`), then re-run the opencaw `--only-mode off` sweep. The deterministic A/B showed hybrid lifts recall@10 from 0.48 to 0.85 on the sysdoc gold set, and the opencaw candidate-recall ceiling (~0.67) is largely an artifact of the eval running cosine. The better retriever already exists; the work is making the eval measure it. Measure on the deterministic `--only-mode off` path so the change is attributable.

## How the docs relate

- **ROADMAP.md** (this) — sequence, current position, forcing logic. Changes often.
- **[TODO.md](../TODO.md)** — unordered backlog of concrete work items.
- **[BUGS.md](../BUGS.md)** — open defects with reproductions.
- **[DECISIONS.md](DECISIONS.md)** — settled implementation choices; authoritative when something is ambiguous.
- **[design.md](design.md) / [scope.md](scope.md)** — the thesis and what v0.1 is/isn't.
