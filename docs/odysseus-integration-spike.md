# Odysseus Integration Spike

Status: proposed (spec for review — no code yet)

## Goal

Prove the smallest end-to-end loop between [Odysseus](../../pewdiepie-archdaemon--odysseus) (self-hosted AI workspace, Python/FastAPI) and OpenCAW, then measure it before committing to a production seam. The three standing objectives this spike serves:

- **(a)** Odysseus uses OpenCAW implicitly for context management.
- **(b)** Real usage telemetry from Odysseus improves OpenCAW.
- **(c)** Improved OpenCAW makes Odysseus more efficient and effective.

These form a loop — *use → measure → improve → use better* — so the spike deliberately builds the whole loop thinly rather than any one part deeply.

## Decisions fixed before this spec

From the scoping conversation:

| Decision | Choice | Consequence |
|---|---|---|
| Integration seam | **Spike first** | Build smallest end-to-end path + telemetry, measure, *then* decide the real seam from data. No live Odysseus chat rewiring in this spike. |
| Corpus | **Memory + session history** | OpenCAW manages Odysseus's vector memory + conversation history, not a file/document corpus. |
| Orchestrator | **In the serving path** | The serving binary runs `DynamicRecallOrchestrator` (multi-pass, thinking-trace recall, eviction, consolidation), not single-shot `caw-server` RAG. |
| Telemetry | **caw-eval `SessionEvaluator`** | Emit recall decisions + outcomes in OpenCAW's existing eval format. |
| Orchestrator model | **Mock adapter first** | Prove plumbing + telemetry with zero GPU cost; swap in a real model once the path is green. |

## Constraint discovered during investigation

OpenCAW's `StubStore::get_content` reconstructs a fragment by slicing `[byte_offset, byte_offset+len)` out of the **source file at `path`** (`caw-core/src/lib.rs:1438`). There is no inline-content store. Odysseus memory/session records are **not files** — they live in `memory.json` and `app.db`.

**Spike resolution:** materialize each record to one text file in a corpus directory, then index that directory with the existing pipeline. Zero OpenCAW core changes.

> Whether to add an inline-content store (so OpenCAW can manage non-file, mutable, per-user corpora without materialization) is a **finding for objective (b)** — explicitly out of scope for the spike. The spike's job is to surface whether that investment is justified.

## Components

### 1. Corpus export + materialization (Odysseus side)

A standalone exporter script in Odysseus (or a small Python tool committed there) that reads:

- `data/memory.json` — memory entries: `{id, text, category, timestamp}`.
- `data/app.db` — session messages (chat history), grouped per session.

and writes a corpus directory:

```
<corpus_dir>/
  memory/<memory_id>.txt          # one memory entry per file
  sessions/<session_id>/<n>.txt   # one message (or turn) per file
```

Provenance is the relative path. The first line of each file is a minimal header (`category`, `timestamp`, `session_id`) so it survives into the stub `path` + body; the rest is the record text.

**Open question (O1):** session granularity — one file per message, or one file per user+assistant turn? Default: per turn (matches a recall unit better). Confirm during build.

### 2. Index build (OpenCAW side, existing pipeline)

Reuse `caw-bench-build-index` unchanged:

```
CUDA_VISIBLE_DEVICES=1 cargo run -p caw-bench --bin caw-bench-build-index -- \
  --corpus <corpus_dir> --out target/caw-dev/odysseus-mem-index.sqlite \
  --rebuild --backend candle --batch-size 32 --sub-batch-size 8
```

GPU-pinning / batch-size rationale per the caw-server runbook (BGE OOM on GPU0 / large sub-batches).

### 3. Serving binary `caw-recall-serve` (new, OpenCAW side)

A sibling to `caw-server`, but running the orchestrator instead of single-shot RAG.

- **Crate:** new binary in `caw-cli` or a new `crates/caw-recall-serve` (decide during build; lean toward a bin in an existing crate to avoid a new crate for a spike).
- **HTTP surface (minimal, not OpenAI-compatible — this is a spike harness endpoint, not a drop-in proxy):**

  `POST /recall`
  ```json
  // request
  { "session_id": "string", "query": "string", "k": 8 }
  ```
  ```json
  // response
  {
    "fragments": [
      { "stub_id": "memory/abc.txt", "source": "memory/abc.txt",
        "content": "...", "tokens": 42, "score": 0.71 }
    ],
    "answer": "string (mock adapter output in spike)",
    "iterations": 2,
    "telemetry_ref": "target/caw-dev/telemetry/<session_id>-<turn>.json"
  }
  ```
- **Per request:** build/lookup a `DynamicRecallOrchestrator` for `session_id` (session state persists across turns within a process), attach a fresh-per-session `SessionEvaluator`, call `run_turn(system, query)`, return fragments + answer.
- **Adapter:** `Mock` for the spike. A `--adapter {mock,ollama,openai}` flag reserves the swap; only `mock` is wired in the spike.
- **State:** one orchestrator per `session_id` held in a map behind a mutex (mirrors `caw-server`'s mutex-per-component pattern). Index/store/embedder shared read-only across sessions.

### 4. Telemetry sink (OpenCAW side)

After each `run_turn`, drain the `SessionEvaluator` and serialize to `target/caw-dev/telemetry/<session_id>-<turn>.jsonl`. **Not** `bench-results/` (append-only ground truth — see project memory). One JSON object per turn containing at minimum:

- recall events: `stub_id`, `score`, `content_tokens` per `record_recall_with_content`.
- eviction events, probe events, annotations.
- per-turn token totals: turn / stub / overhead.
- derived: `recall_metrics` (needs expected ids — see O2), `context_efficiency`, `cooperation_metrics`.

**Open question (O2):** ground-truth `expected_ids` for `recall_metrics`. The spike has no labels. Options: (i) skip recall@k, report only efficiency + cooperation + raw events; (ii) use Odysseus's current `_hybrid_retrieve` top-k as a *reference* set to compute agreement (not ground truth). Default: (i) for the spike; (ii) is part of measurement, below.

### 5. Replay harness (Odysseus side)

A script that:
1. runs the exporter (component 1),
2. triggers the index build (component 2),
3. pulls a sample of real user queries from `app.db` (recent chat turns),
4. POSTs each to `/recall`,
5. collects telemetry refs and writes a summary.

No live `chat_processor.py` modification. The existing memory retrieval seam (`src/chat_processor.py:106`, `MemoryVectorStore.search`) is documented as the *future* rewire point but left untouched.

## Measurement plan

For the sampled query set, compare OpenCAW recall against Odysseus's existing `_hybrid_retrieve` (`chat_processor.py:54`):

| Metric | Source | What it tells us |
|---|---|---|
| Fragment agreement | OpenCAW fragments ∩ Odysseus top-k | Does the orchestrator surface the same memories the current system would? Divergence is the interesting signal. |
| Multi-pass lift | `iterations > 1` rate; fragments added after pass 1 | Does thinking-trace recall pull in memories the initial query missed? This is the differentiator's whole claim. |
| Context efficiency | `SessionEvaluator::context_efficiency` | Token cost of the workspace vs. nominal. |
| Cooperation | `SessionEvaluator::cooperation_metrics` | (Limited under Mock adapter; real signal needs a real model — flagged for phase 2.) |

The Mock adapter caps what's measurable: cooperation and multi-pass lift need a model that actually emits thinking traces. The spike's honest deliverable is **the loop runs end-to-end and emits well-formed telemetry**, plus fragment-agreement vs. Odysseus. Real-model measurement is the immediate follow-on.

## File inventory

OpenCAW (this repo):
- `docs/odysseus-integration-spike.md` (this file)
- serving binary: `crates/caw-cli/src/bin/caw-recall-serve.rs` (or new crate — TBD at build)
- telemetry serialization: small module next to the binary; may add a `to_json`/drain helper on `SessionEvaluator` in `caw-eval` if one doesn't exist.

Odysseus (`../../pewdiepie-archdaemon--odysseus`):
- exporter + replay harness: `scripts/opencaw_spike/` (export.py, replay.py).

## Out of scope (explicitly)

- Live Odysseus chat/agent rewiring.
- Inline-content store / mutable per-user corpora in OpenCAW core.
- OpenAI-compatible serving (that's the `caw-server` path; this spike uses the orchestrator).
- Real-model orchestration (Mock first; real model is the next phase).
- Incremental / live reindexing as memory changes.

## Open questions to resolve at build time

- **O1** session export granularity (per-message vs per-turn). Default: per-turn.
- **O2** recall@k ground truth. Default: skip for spike; report agreement vs. Odysseus instead.
- **O3** serving binary placement: bin in `caw-cli` vs. new crate. Default: bin in `caw-cli`.
- **O4** ~~does `SessionEvaluator` already expose a serialize/drain path?~~ **Resolved:** no. `caw-eval` metrics/session structs derive only `Debug, Clone`, recorded events are private, and there is no JSON path. The spike adds (a) `#[derive(serde::Serialize)]` to the metrics structs and the per-turn event records, and (b) a read accessor / snapshot method to expose the recorded events. Small, additive change in `caw-eval`.
