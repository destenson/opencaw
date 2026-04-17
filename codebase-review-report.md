# OpenCAW Codebase Review

**Date:** 2026-04-16
**Branch:** `try1`
**Commit:** 2fd9e8e

## Executive Summary

OpenCAW is a **functionally complete v1 library** implementing the context-as-workspace architecture: ingestion, hybrid retrieval, dynamic recall orchestration with eviction/consolidation, provenance tagging, curation, degradation monitoring, and a benchmark harness are all wired. The code is clean (10 debt-marker sites total across the workspace, minimal `unwrap()` outside tests) and the "must-have" SCOPE items are checked off. **The primary remaining gap is empirical, not engineering**: the bench harness exists but has not been swept across enough seeds/models to produce the threshold-tuning and cooperation-calibration numbers the thesis depends on.

**Primary recommendation:** stop building. Run the bench. Sweep `caw-bench` across NIAH + opencaw workloads, several models (Ollama + vLLM + claude-code), and a small grid of hysteresis thresholds; use the results to either validate the architecture or drive the next round of work. Everything else on the backlog (richer consolidation, insertion-order experiments, conflict detection) is speculative until measurement data says which lever matters most.

## Implementation Status

- **Working** — Ingestion pipeline (cl100k tokenizer, tree-sitter outlines, adaptive chunking, deterministic + LLM summaries) — `caw-ingest/`.
- **Working** — Retrieval stack (FastEmbed BGE, HNSW, BM25, hybrid fusion, SQLite stub store with consolidation persistence) — `caw-index/`.
- **Working** — `DynamicRecallOrchestrator` with multi-pass recall, probes, thinking-trace extraction, relevance decay, budget-triggered eviction, mid-session annotations — `caw-orchestrator/src/dynamic.rs`.
- **Working** — Degradation monitor with tiered fallback + probe rate limiting — `caw-orchestrator/src/degradation.rs`.
- **Working** — Curation (history summarization, tool-output compression, system-prompt budgeting, extractive + LLM variants) — `caw-curation/`.
- **Working** — Adapters: Anthropic, Groq, Ollama, OpenAI-compatible, ClaudeCode, Mock — `caw-adapters/`.
- **Working** — Evaluation primitives (`SessionEvaluator`, recall@k, false-recall, hysteresis, context efficiency, cooperation) — `caw-eval/`.
- **Working** — Benchmark harness with NIAH + opencaw Q&A workloads, recall-on vs recall-off — `caw-bench/`.
- **Incomplete** — Consolidation notes are mechanical strings by default; `LlmConsolidation` exists but is opt-in — `caw-orchestrator/` (TODO.md §Eviction).
- **Incomplete** — Provenance conflict detection is Jaccard-only; contradictions/number-mismatch undetected — `caw-provenance/`.
- **Incomplete** — Prompt transformer covers markdown links + `@path` only; fenced `path=` blocks and bare-path regex not wired — `caw-transform/`.
- **Missing** — Background indexer with lazy fallback (ingest is synchronous, single-pass).
- **Missing** — Insertion-order experiment harness; few-shot cost-surfacing utility.
- **Deferred (per SCOPE.md)** — `caw-server` is a `main.rs`-only scaffold; async adapter traits; engine plugins; streaming mid-token recall.

## Code Quality

- Test results: **16 passed / 0 failed** across 4 populated test binaries (23 empty). Tests are sparse by design — user prefers integration over unit tests. The `end_to_end.rs` integration test is the primary smoke test.
- Debt markers (`TODO|FIXME|HACK|for now|temporary`): **10 occurrences** across 4 files — very low. Mostly in `caw-bench/`.
- `unwrap()`/`expect()`/`panic!()` in non-test code: ~28 total, concentrated in `caw-transform` (19) and `caw-core/tokenizer.rs` (3). Worth auditing `caw-transform` but not urgent.
- Uncommitted changes: `caw-bench` tweaks (ollama error surfacing, QA-JSON reshaping) + `.orig.json` artifact. Should be committed or cleaned before sweep runs.
- Docs: README, TODO, SCOPE, and the design doc are mutually consistent and up to date — no drift detected.

## PRP Status

No `PRPs/` directory exists. The project uses `context-as-workspace.md` (design doc) + `SCOPE.md` + `TODO.md` as its planning substrate instead, and they are coherent. No PRPs to validate.

## Recommendation

**Next Action:** execute measurement sweeps via `caw-bench`, not new feature work.

**Justification:**
- **Current capability** — The v1 library is structurally complete and compiles + tests clean. All "must-have" SCOPE items except model-cooperation calibration are done.
- **Gap** — SCOPE.md and TODO.md both explicitly call out: *"sweep runs across enough seeds and workloads to produce threshold-tuning recommendations and cooperation-calibration numbers."* The harness exists. The numbers don't.
- **Impact** — Without bench data, every remaining backlog item (richer consolidation, insertion-order, conflict detection, threshold defaults) is guessing at which knob matters. With bench data, the backlog gets ranked by evidence.

**90-Day Roadmap:**
1. **Week 1-2** — Commit the pending `caw-bench` cleanup. Run NIAH sweep across 3+ models (qwen3.5, deepseek-r1, claude-sonnet via claude-code) at 2-3 seed values. Record recall@k, precision@1, MRR, context_efficiency, false_recall, latency for on vs. off. → Baseline dataset.
2. **Week 3-4** — Run opencaw Q&A sweep. Use results to tune `RecallThresholds` (0.7/0.4 hysteresis defaults may need adjustment). Stand up a minimal cooperation-calibration harness reusing `CooperationMetrics`. → Calibrated defaults + per-model cooperation scores.
3. **Week 5-8** — Address the highest-impact gap the data surfaces. Likely candidates: enable `LlmConsolidation` by default if notes drive measurable recall-quality lift; add synthesis-style questions to `opencaw_qa.json` if single-fact lookups saturate; implement insertion-order experiments if context_efficiency is bottleneck.
4. **Week 9-12** — Depending on (3): either ship v1.0 of the library with calibrated defaults, or tackle the next ranked item (provenance conflict detection beyond Jaccard, or background indexer if ingestion latency shows up in bench numbers).

## Technical Debt Priorities

1. **Mechanical consolidation notes as default** — caps the "semantic memory across sessions" value prop; LLM variant already exists. Effort: low (flip default + add `--llm-consolidation` CLI flag). Impact: medium, but empirically unclear until bench quantifies.
2. **`caw-transform` unwrap density** — 19 `unwrap/expect` calls in a crate that handles user input. Effort: medium (audit + convert to `Result`). Impact: low-medium (crash risk in middleware/server deployment targets; zero impact on library use from trusted call sites).
3. **QA corpus is single-fact-dominated** — README acknowledges this. Adds synthesis-shaped questions to stress multi-pass refinement. Effort: low (author ~10 multi-hop Q&A pairs). Impact: medium (reveals whether multi-pass loop actually helps).
4. **12-crate decomposition** — Flagged in TODO.md but explicitly gated behind "do not restructure without explicit approval." Leave alone.
5. **Uncommitted bench changes** — Commit hygiene, not debt. Resolve before running sweeps so results are reproducible.
