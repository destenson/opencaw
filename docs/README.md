# OpenCAW Documentation

This is the map of the `docs/` tree. Every document falls into one of two kinds, and each carries a `Status:` line at its top so you can tell at a glance whether it is meant to be current:

- **Living reference** — kept up to date; safe to trust as describing the system now.
- **Archive** (`archive/`) — frozen point-in-time records (completed spikes, old snapshots, brainstorms, the QA findings consolidation). Kept for history, **not maintained**, and should not be read as current.

If a living doc contradicts the code, the code wins — fix the doc. The active open-bug log lives at [`../BUGS.md`](../BUGS.md), not in `docs/`.

## Start here

- [Design](design.md) — the thesis: context-as-workspace, on-demand recall, curated working sets. Read this first to understand *why* OpenCAW exists.
- [Scope](scope.md) — what v0.1 is and is not; the scope-change protocol.
- [Roadmap](ROADMAP.md) — where we are, what's next, and *why that order*. Read to answer "where should we go next?"; carries sequence and current focus that TODO/BUGS/DECISIONS deliberately don't.
- [Decisions](DECISIONS.md) — authoritative reference for settled implementation choices. Read before making any significant implementation decision.
- [Getting Started](getting-started.md) — build, run, environment variables, a code example.
- [Glossary](glossary.md) — canonical definitions for terms used across the project and docs.

## Reference (living)

- [Architecture](architecture.md) — crates, data flow, the ingest → index → stub → recall → evict/consolidate loop.
- [Adapters](adapters.md) — model adapters (Anthropic, Groq, Ollama, OpenAI-compatible, LlamaCpp, Mock).
- [Embedding Providers](embedding-providers.md) — FastEmbed, API-based, Candle, ONNX.
- [Storage](storage.md) — stub store and vector index (SQLite, Qdrant, HNSW, hybrid retrieval).
- [Benchmarking](benchmarking.md) — the recall-on vs recall-off harness, workloads, and sweep tooling.

## Archive (frozen — not maintained)

- [Codebase Review](archive/codebase-review.md) — implementation-status snapshot from 2026-05-02.
- [Concept-Vector RAG Guide](archive/concept-vector-rag-guide.md) — completed spike on prefill activations as a semantic index.
- [Graphify Integration Spike](archive/graphify-integration-spike.md) — completed spike on call-graph expansion as a retrieval signal.
- [Findings](archive/findings.md) — closed historical consolidation of QA loops 0001–0021. Judged metrics are retired; read for finding history (including some still-open QA-era bugs) only.
- [Mascot](archive/mascot.md) — early branding/mascot brainstorm.

## Related, outside `docs/`

- [Project README](../README.md) — top-level overview and entry point.
- [BUGS](../BUGS.md) — the active open-bug log.
- [TODO](../TODO.md) — development tracking and open work.
- [Skills](skills/) — the `caw-dev` skill and its scripts for building, running, indexing, and benchmarking the stack.
