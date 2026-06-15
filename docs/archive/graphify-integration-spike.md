# Graphify Integration Spike

> **Status: Frozen spike (2026-06-07).** Completed investigation, kept for history. Not maintained. See [docs/README.md](../README.md) for the live docs index.

Status: spike complete — edge sidecar + ingest + expansion + measurement all built and run over two passes (`caw-index::graph_edges`, `caw-cli` bin `caw-graph-ingest`, `caw-bench` bin `caw-bench-graph-eval`). Verdict in **Result (second pass)**: call-graph expansion with rank-adjacent merge is a real retrieval signal that recovers answers cosine misses, but needs a displacement gate before it goes on by default. Not wired into the orchestrator. Sections marked _(as built)_ record what the implementation does, which in places differs from the original spec.

## Goal

Prove the smallest end-to-end loop that uses [Graphify](https://github.com/safishamsi/graphify)'s deterministic code graph as a retrieval signal inside OpenCAW, then measure whether it improves recall before committing to a real seam. Graphify gives us the one thing `caw-ingest` does not produce today: a cross-file structural graph (call edges, import edges, Leiden communities) over the code, built deterministically from tree-sitter ASTs with no embeddings.

The single question this spike answers: **does graph-neighbor expansion of retrieval results improve chunk rank on code-structure questions, versus embedding+BM25 alone?** This is a direct test of the standing finding that cosine is insufficient on a single-domain corpus — graph edges are a structure signal orthogonal to cosine.

This spike does **not** touch OpenCAW's differentiator (thinking-trace recall, eviction, consolidation). Graphify has no notion of any of that. It strengthens the retrieval substrate (`caw-ingest` + `caw-index`) only. That scoping is deliberate: if the structure signal doesn't help, we learn it cheaply and drop it.

## What Graphify is (grounded, not the marketing)

- **Language/license:** Python, MIT. Built on tree-sitter (28 grammars incl. Rust), NetworkX, Leiden clustering, faster-whisper (A/V — irrelevant here).
- **Code extraction is AST-only, no embeddings, no API key.** `graphify extract` runs fully offline on a code-only corpus. (The LLM only enters for docs/PDF/image passes, which this spike does not use.)
- **Output:** `graphify-out/graph.json` (full graph), `GRAPH_REPORT.md` (highlights), `graph.html` (viz). Nodes carry file + line provenance; edges are typed; nodes are grouped into communities.
- **Query surface:** `graphify query "..."` returns subgraphs with EXTRACTED/INFERRED/AMBIGUOUS confidence tags. **The spike does not use Graphify's query path** — we consume `graph.json` directly and do expansion inside OpenCAW. Graphify is an *extractor*, not a runtime dependency of the recall loop.

## Decisions fixed before this spec

| Decision | Choice | Consequence |
|---|---|---|
| Integration seam | **Spike first** | Build smallest path (extract → edge sidecar → expansion → measure), *then* decide the real seam from data. |
| Boundary | **Subprocess, JSON over the wire** | Shell out to `graphify extract`, read `graph.json`. No PyO3, no FFI, no native reimplementation of extraction. Graphify stays an external tool. |
| Role of the graph | **Retrieval signal, never context payload** | Edges drive neighbor expansion *behind* retrieval. The graph is never dumped into the model's context. This is the opposite of Graphify's own "consult GRAPH_REPORT.md before grep" hook model, which we explicitly reject — it fights "context is a workspace." |
| Corpus | **A real code repo** | Default: OpenCAW itself (Rust, dogfood). The graph helps most on code-structure questions, so the eval must be a code corpus — not the prose/NIAH workloads. |
| Stub mapping | **Overlay edges on existing chunk-stubs** | Do *not* fork ingestion to make one stub per Graphify node. Map each Graphify node to the chunk-stub that covers its source location, and attach edges between chunk-stubs. Keeps `caw-ingest` as the single source of stubs. _(As built: nodes carry a single start line, so mapping is line-containment, not range-overlap — see Constraint.)_ |
| Core changes | **Additive sidecar only** | No change to `Stub`. Edges live in a separate table keyed by `StubId`. `caw-core` is untouched. |

## Constraint discovered during investigation

Graphify's nodes are **file-backed**: each carries a source path and a source location. OpenCAW's `Stub` is *also* file-backed — `caw-core/src/lib.rs:375` defines `Stub { path, byte_offset, byte_length, .. }` and content is reconstructed by slicing `[byte_offset, byte_offset+byte_length)` out of `path` (there is no inline-content store; `caw-core/src/lib.rs:1429`). So Graphify nodes and OpenCAW stubs live in the same coordinate system (a file + a position). **No materialization step is needed** — Graphify nodes drop onto the stub model directly. A non-file corpus (e.g. records in a database) would instead need each record written out to a file before the existing pipeline could index it; Graphify avoids that entirely.

_(As built)_ Two facts about the real `graph.json` (Graphify 0.8.33) shaped the mapping:

- **A node's `source_location` is a single start line (`"L20"`), not a range.** The spec assumed ranges. So a node maps to *the one chunk-stub whose byte range covers the byte offset of that start line* — line-containment, not range-overlap. `caw-graph-ingest` reads each source file once to build a line→byte table, converts each node's start line to a byte offset, and finds the covering stub via `GraphEdgeStore::stub_geometry(path)`. Chunks tile a file contiguously, so a location past the last chunk's end clamps to the last chunk rather than going unmapped.
- **`source_file` is relative to the extract root**, matching how stub paths are stored (e.g. `caw-core/src/lib.rs`). For a graph from `graphify extract crates/`, the ingest reads files under `--source-root crates`. The two coordinate systems line up with no translation.

The remaining mismatch is **granularity**: a Graphify node = one symbol; an OpenCAW stub = one chunk (`caw-ingest/src/chunking.rs`). Many nodes can land on one chunk. For each graph edge, both endpoints are resolved to their covering stubs; the edge is kept only if they land on *different* stubs (intra-chunk edges add nothing for cross-chunk expansion), and stub edges are deduped at `(src, dst, relation)`.

> Whether to eventually carry symbol-level nodes as first-class stubs (finer recall units than chunks) is a **finding**, explicitly out of scope. The spike's job is to surface whether the structure signal is worth that investment at all.

_(As built — validated end-to-end)_ Pure-AST extract of `crates/` (72 code files, no LLM): **1909 nodes, 4733 edges**. Ingesting with the default relation set produced **1027 stub edges** (calls 474, contains 285, method 160, imports_from 57, implements 51) — 953 linking different chunks of the same large file, 74 cross-file. Dropped: 2478 `references` (excluded by default as the vague bucket), 362 intra-chunk self-edges, 332 endpoints in files the test index predated. (Validated against a stale index copy; a freshly matched index is needed for measurement.)

## Components

### 1. Graph extraction (Graphify side, subprocess)

Invoke a **pinned** Graphify version on the corpus:

```
graphify extract <corpus_dir> --out target/caw-dev/graphify-out
```

Code-only run → no API key, fully local, deterministic. Output consumed: `target/caw-dev/graphify-out/graph.json`.

> Supply-chain note (per global policy): pin the Graphify version/commit and review it before first run. It's an external Python tool pulling tree-sitter grammars; treat it like any third-party dependency. Record the pinned ref in the spike's run notes.

**Open question (G1):** Graphify install/version management — pip into a throwaway venv vs. pipx vs. vendored. Default: pinned pipx install, ref recorded.

### 2. Graph ingest + edge sidecar (OpenCAW side, new)

A small new binary, `caw-graph-ingest` (a bin in `caw-cli` — a spike doesn't justify a new crate), that:

1. Reads `graph.json` (nodes with path + line range; typed edges; community ids).
2. Loads the existing OpenCAW index/store for the same corpus (the one built by `caw-bench-build-index`).
3. For each Graphify node, converts its line range to a byte range and finds overlapping chunk-stubs (range-overlap mapping above).
4. Writes a **sidecar edge table** to SQLite alongside the stub store:

   ```sql
   CREATE TABLE stub_edge (
     src_stub_id TEXT NOT NULL,
     dst_stub_id TEXT NOT NULL,
     edge_kind   TEXT NOT NULL,   -- 'call' | 'import' | 'community'
     weight      REAL NOT NULL,   -- 1.0 for call/import; community = shared-membership weight
     PRIMARY KEY (src_stub_id, dst_stub_id, edge_kind)
   );
   ```

   Community membership becomes `community` edges (or a separate `stub_community(stub_id, community_id)` table — decide at build; a membership table is cleaner for "same community" lookups). No change to the `stubs` table or `Stub`.

### 3. Neighbor expansion in retrieval (OpenCAW side, new, behind a flag for A/B)

A post-retrieval step wrapping the existing retriever (`HybridRetriever` / `SemanticRetriever`, `caw-index/src/lib.rs`):

1. Run normal hybrid retrieval → seed `Vec<(StubId, score)>`.
2. For each seed, look up sidecar edges → candidate neighbor stub ids.
3. Admit up to `expansion_cap` neighbors, each scored as `seed_score * edge_discount(edge_kind)` (call/import stronger than community).
4. Merge + dedupe with seeds, keep top-k by merged score.

Exposed as an **opt-out flag for A/B only** (e.g. `--no-graph-expansion`), consistent with the project rule that features are on by default and only A/B toggles are allowed. The expansion step is a pure function `(seeds, edge_lookup) -> expanded` so it's testable in isolation and trivially measured on/off.

### 4. Measurement harness (OpenCAW side, reuse caw-bench)

Reuse the existing benchmark harness pattern (`caw-bench`, recall-on/off at matched budget) with expansion as the toggled variable. Build a small **code-structure question set** against the chosen repo where the answer chunk is known (e.g. "where is X called from", "what does Y depend on", "trace the path from A to B") — these are exactly the questions a call graph should help and pure cosine should miss.

## Measurement plan

A/B is expansion-on vs expansion-off, **same index, same budget, same question set**:

| Metric | What it tells us |
|---|---|
| Chunk rank of the gold stub (expansion on vs off) | The core claim: does walking call/import edges pull the right chunk up the ranking when cosine ranks it low? |
| recall@k delta | Aggregate lift across the code-question set. |
| Edge-kind attribution | Of the cases that improved, which edge type did the work — `call`, `import`, or `community`? Tells us what to keep. |
| Expansion cost | Extra stubs admitted per query, and token cost of those stubs vs. the rank improvement. Structure signal isn't free in context budget. |
| Regression check | Cases where expansion *hurt* (admitted a wrong neighbor that displaced the gold). Honest accounting — neighbor expansion can add noise. |

Honest deliverable: **a measured chunk-rank delta on code-structure questions, with edge-kind attribution and a regression count.** If the delta is flat or negative on OpenCAW's workloads, the conclusion is "structure signal doesn't help here," and the sidecar/expansion code is dropped — that's a successful spike.

## Result (first pass — discounted merge, easy questions)

Fresh BGE-small index of `crates/` (3174 stubs), 1077 stub edges, semantic-only baseline (top-k=30), expansion discount 0.5, cap 8, relations `calls,imports_from,implements,inherits,method,contains`. Golden set: `crates/caw-bench/src/qa/graph_eval_qa.json` (19 questions, 13 structural / 6 definition, gold lines verified against current source). Run via `caw-bench-graph-eval`.

**Aggregate: flat.** recall@1/5/10 unchanged (0.316 / 0.684 / 0.842 overall), MRR 0.517 → 0.519 (+0.002). 1 improvement, 0 regressions across 19 questions.

**The one improvement is the mechanism working as designed.** `s01_callers_count_tokens` ("which functions call `count_tokens_cl100k`") went from **miss → rank 31 via a `calls` edge**: the caller chunk carried none of the query's semantic signal, so cosine missed it in the top-30 entirely, and the graph edge from a high-ranked seed recovered it. That is exactly the case the spike predicted graph expansion would uniquely serve.

**Why aggregate lift is ~0, honestly:**

- The baseline is already strong on this set. Most questions put the answer's symbol name *in* the query, so BGE ranks the gold chunk in the top 1–5 — no headroom. This includes the trait-implementation structural questions (`impl Retriever for ...` literally contains "Retriever"), which turned out to be weaker stressors than hoped.
- Expansion only admits neighbors *below* the seeds (discounted), so it can only change the outcome when the gold chunk is *outside* the baseline top-k. On a 3174-stub corpus with k=30, baseline recall@30 is high, leaving little to rescue. Only `s01` was both out-of-top-k and reachable via an edge.
- `s10_get_content` stayed miss→miss: not reachable from any top-30 seed via the selected relations.

**This is the "underpowered test" risk flagged before the run, now confirmed.** The result does *not* show the graph is useless; it shows that on codebase Q&A where embeddings already rank the answer high, expansion rarely changes the result — and when the answer is semantically invisible (pure caller/trace), it does help. To fairly judge the hypothesis the next pass needs: (a) more "gold chunk lacks the query's terms" questions (pure callers / multi-hop traces), and (b) a top-k sweep — at smaller k the baseline misses more, which is exactly where expansion can act.

## Result (second pass — harder questions + rank-adjacent merge)

Two changes from the first pass. **(1) Harder questions:** added 8 `type: "caller"` questions ("which functions call X"), gold = the cross-file *caller* chunks, phrased so cosine ranks X's definition (the seed) but not the callers. As intended these are hard for the baseline: caller recall@5 = 0.375, MRR 0.145. **(2) Merge policy:** the first pass exposed a design flaw — a neighbor scored `seed × discount` always sorts *below* the pool, so it can never lift a gold already ranked and dumps a recovered miss at the very bottom (the two first-pass rescues landed at rank 101/102, useless). The fix is **rank-adjacent merge**: insert each admitted neighbor immediately after the seed that pulled it in, so a caller inherits its callee-definition's rank. `caw-bench-graph-eval --merge {adjacent,discounted}`; `adjacent` is the default.

**Adjacent merge, `calls` edges only** (the only relation that ever helped — see attribution below):

| group | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| caller (n=8) | 0.125 → **0.625** | 0.375 → **0.750** | 0.625 → **0.875** | 0.145 → 0.282 |
| structural (n=13) | 0.769 → 0.846 | 0.769 → 0.846 | 0.846 → 0.846 | 0.626 → 0.666 |
| definition (n=6) | 0.500 → 0.500 | 0.500 → 0.500 | **0.833 → 0.500** | 0.282 → 0.252 |
| all (n=27) | 0.519 → 0.704 | 0.593 → 0.741 | 0.778 → 0.778 | 0.407 → 0.460 |

**Rescue:** of 6 questions where cosine put the gold outside top-10, adjacent+calls pulled **4 into the top-10** — a full miss → rank 2, miss → 3, 16 → 3, 26 → 3, 65 → 14. The mechanism does the thing.

**Findings:**

- **The signal is real and it is the call graph specifically.** Every single improvement (7/7) came through `calls` edges. `contains`/`method`/`implements`/`imports_from`/`inherits` produced zero wins and only added displacement noise — restricting to `calls` cut regressions 9 → 6 and lifted overall MRR (+0.019 → +0.053). The useful structural signal for retrieval is "who calls whom," not the rest.
- **Big wins exactly where cosine fails.** On the caller set (the population that matters) recall@3 went 5× (0.125 → 0.625) and recall@5 doubled. This is the spike's hypothesis, confirmed.
- **Real cost: displacement of easy mid-rank answers.** Adjacent insertion pushes a rank-6–12 gold down when neighbors of higher-ranked seeds are inserted ahead of it. Definition questions (gold mid-rank, unrelated to call structure) regressed: recall@10 0.833 → 0.500. So unconditional global expansion is *not* a clean win at every k.

**Verdict: confirmed, with a gate required.** Call-graph expansion, merged rank-adjacent, is a strong retrieval signal for structural/caller/trace questions and recovers answers cosine misses entirely. But applied unconditionally it harms lookups where the embedding already had the right chunk. Net recall@3/@5 are clearly up; recall@10 is flat because the caller gains and definition losses cancel. The open problem is **when to expand** — not whether the signal exists.

**Next levers (not yet done):**
- Bound displacement: only insert neighbors after the top-1–2 seeds (a rank-9 gold can't be shoved past by neighbors of rank-5 seeds), and/or cap neighbors-per-seed tighter.
- A gate for *whether* to expand. Note the project's no-heuristics rule: prefer expanding always but bounding displacement, over sniffing "is this a structural question."
- Only then consider wiring expansion into `DynamicRecallOrchestrator` Phase 1. Until the recall@10 regression on easy queries is closed, it should not go on by default.

## File inventory

OpenCAW (this repo):
- `docs/archive/graphify-integration-spike.md` (this file)
- `crates/caw-cli/src/bin/caw-graph-ingest.rs` (graph.json → edge sidecar)
- edge sidecar storage: small module in `caw-index/src/storage/` next to the stub store
- expansion step: pure function in `caw-index` wrapping the retriever; opt-out flag wired in `caw-cli`
- code-structure question set + harness wiring: in `caw-bench`

External:
- Graphify: pinned install, invoked as a subprocess. No code committed here; pinned ref recorded in run notes.

## Out of scope (explicitly)

- Graphify's own `query`/MCP/hook runtime path — we consume `graph.json` only.
- The "consult GRAPH_REPORT.md before grep" coding-agent skill model — rejected; it's a different deployment target and fights the workspace thesis.
- Symbol-level (per-node) stubs as a finer recall unit — a finding, not spike work.
- Docs/PDF/image/A-V passes (the LLM-backed Graphify passes) — code-only here.
- Native Rust reimplementation of graph extraction — subprocess only.
- Incremental / commit-hook rebuild — measure value first, then worry about freshness.
- Anything touching the orchestrator's eviction/consolidation/trace-recall — this spike is retrieval-substrate only.

## Open questions to resolve at build time

- **G1** Graphify install/version pinning (pipx vs venv vs vendored). Default: pinned pipx, ref recorded.
- **G2** Community representation: `community` edges vs. a `stub_community` membership table. Default: membership table (cleaner same-community lookup).
- **G3** Edge-kind discount factors and `expansion_cap`. Default: start `call`/`import` at full seed score, `community` discounted, cap small; let the edge-kind attribution metric tune them.
- **G4** Range-overlap mapping when a Graphify node spans multiple chunks or a chunk holds many nodes. Default: many-to-many overlap, dedupe at the stub-pair level.
- **G5** Corpus + gold question set: OpenCAW repo (dogfood) vs. a larger external repo (e.g. open-design from the demo). Default: OpenCAW repo first; it's small, Rust, and we know the answers.
