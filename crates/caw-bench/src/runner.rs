use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::{
    count_tokens_cl100k, EmbeddingProvider, ModelAdapter, RecallThresholds, StubId, StubStore,
    VectorIndex,
};
use caw_index::{
    build_bm25_over_store, BM25Index, CandleEmbeddingProvider, FastEmbedProvider, HnswVectorIndex,
    HybridRetriever, SemanticRetriever, SqliteStubStore,
};
use std::sync::Arc;
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{CooperationMode, DynamicRecallConfig, DynamicRecallOrchestrator};

use crate::shared::{ReadOnlyStore, SharedEmbedder, SharedIndex, SharedStore};
use crate::workload::{RecallMode, Scoring, WorkloadItem};

pub struct RunnerConfig {
    pub system_prompt: String,
    pub max_candidates: usize,
    pub max_workspace_tokens: usize,
    pub max_recall_iterations: usize,
    /// Load threshold passed into the orchestrator's hysteresis config.
    /// The library default (0.7) is tuned for real document corpora; short
    /// synthetic text (like NIAH filler) and small code symbols rarely
    /// clear it, so the bench defaults lower to keep workloads live.
    pub load_threshold: f32,
    /// Unload threshold (hysteresis band lower edge).
    pub unload_threshold: f32,
    /// How many total questions to run (trimmed workload); None = run all.
    pub limit: Option<usize>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        // Tight workspace + permissive thresholds: forces the orchestrator's
        // multi-pass / eviction machinery to actually fire when probes
        // bring in additional fragments. With a 12k budget the budget
        // ceiling never bit and recall-on collapsed to recall-off.
        Self {
            system_prompt: "Use the recalled workspace context to answer \
                 the question accurately and concisely. Cite source locators from the recalled \
                 context when they support your answer."
                .to_string(),
            max_candidates: 20,
            max_workspace_tokens: 2_000,
            max_recall_iterations: 3,
            load_threshold: 0.3,
            unload_threshold: 0.2,
            limit: None,
        }
    }
}

/// Ingest `corpus` once into an in-memory sqlite store + HNSW index using
/// the Candle GPU embedding provider. Returns a `PrebuiltIndex` that every
/// item/mode in a bench invocation can share via `run_item_shared`, plus
/// the tempdir backing `corpus_root` (caller must hold it alive for the
/// duration of the run — dropping it deletes the files and invalidates
/// `get_content`).
///
/// This is what the `opencaw` workload uses when no external `--index` is
/// supplied: the repo corpus is identical across items, so re-embedding it
/// per item (the old `run_item_fresh` path) burned the same CPU work N×
/// for no benefit. One upfront build on GPU replaces 4× CPU fastembed runs
/// for the default on/off × 2-item smoke.
pub fn build_in_memory_prebuilt(
    corpus: &[crate::workload::CorpusDoc],
) -> Result<(PrebuiltIndex, tempfile::TempDir)> {
    let started = std::time::Instant::now();

    // Materialize corpus to a tempdir so `SqliteStubStore::get_content` has
    // something to seek into.
    let corpus_tmp = tempfile::tempdir().context("create shared corpus tempdir")?;
    for doc in corpus {
        let full = corpus_tmp.path().join(&doc.path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
        std::fs::write(&full, doc.content.as_bytes())
            .with_context(|| format!("write {}", full.display()))?;
    }

    let mut embedder = CandleEmbeddingProvider::bge_small()
        .context("initialize bge-small (candle) for shared in-memory index")?;
    let dim = embedder.dimension();

    let mut store = SqliteStubStore::in_memory(dim)
        .context("open in-memory sqlite store")?
        .with_corpus_root(corpus_tmp.path().to_path_buf());
    let mut vector_index = HnswVectorIndex::new();
    let mut stub_summaries: HashMap<StubId, String> = HashMap::new();

    // Collect all (stub, embed_text) pairs first, then embed in one batch
    // call so the GPU session gets a full kernel launch per batch instead
    // of one per doc.
    let pipeline = IngestionPipeline::new();
    let mut all_stubs: Vec<caw_core::Stub> = Vec::new();
    let mut all_texts: Vec<String> = Vec::new();
    for doc in corpus {
        let source_doc = SourceDocument {
            path: doc.path.clone(),
            content: doc.content.clone(),
            kind: doc.kind,
            mtime_unix_secs: 0,
        };
        for (stub, _embed_text) in pipeline.ingest(source_doc) {
            let summary_text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
            all_texts.push(summary_text);
            all_stubs.push(stub);
        }
    }

    // Candle's BGE-small forward scales with batch × seq_len². Batch 256
    // at seq 512 OOMed on a 3090 sharing ~7 GB with Ollama's resident
    // model (only ~17 GB free); 128 is 4× the failing footprint, so it
    // fits with margin while keeping throughput reasonable.
    const EMBED_BATCH: usize = 128;
    let total = all_stubs.len();
    let mut offset = 0usize;
    while offset < total {
        let end = (offset + EMBED_BATCH).min(total);
        let batch: Vec<&str> = all_texts[offset..end].iter().map(|s| s.as_str()).collect();
        let embeddings = embedder
            .embed_document(batch)
            .context("batch embed shared corpus")?;
        for (stub, embedding) in all_stubs[offset..end].iter().zip(embeddings.into_iter()) {
            let stub_id = stub.id.clone();
            stub_summaries.insert(stub_id.clone(), stub.summary.clone());
            store
                .insert(stub.clone(), embedding.clone())
                .context("insert stub into shared store")?;
            vector_index.add(stub_id, embedding);
        }
        offset = end;
    }

    eprintln!(
        "built in-memory shared index: {} stubs from {} docs in {:.1}s",
        total,
        corpus.len(),
        started.elapsed().as_secs_f64()
    );

    // Build BM25 over the same stubs so the shared retriever fuses lexical +
    // semantic, matching the proxy. Done after all inserts so `get_content`
    // can slice bodies out of the materialized corpus tempdir.
    let bm25 = build_bm25_over_store(&store, all_stubs.iter().map(|s| s.id.clone()));

    let prebuilt = PrebuiltIndex {
        embedder: SharedEmbedder::new(embedder, "bge-small-en-v1.5"),
        store: SharedStore::new(store),
        index: SharedIndex::new(vector_index),
        bm25: Arc::new(bm25),
        stub_summaries,
    };
    Ok((prebuilt, corpus_tmp))
}

/// A pre-ingested retrieval index, shared across all items in a bench run.
///
/// Built once by `main` (loading a sqlite produced by
/// `caw-bench-build-index`) and passed to `run_item` for workloads whose
/// corpus is known ahead of time. Items then skip ingestion entirely —
/// each one clones the shared handles into a fresh `SemanticRetriever`
/// over the same backing state.
pub struct PrebuiltIndex {
    pub embedder: SharedEmbedder<CandleEmbeddingProvider>,
    pub store: SharedStore<SqliteStubStore>,
    pub index: SharedIndex<HnswVectorIndex>,
    /// Corpus-wide BM25 lexical index, built once at load time and shared
    /// read-only across every per-item retriever. Behind an `Arc` because
    /// building it reads every body — too expensive to repeat per item.
    pub bm25: Arc<BM25Index>,
    /// Stub summaries keyed by id — populated at load time so the false-recall
    /// heuristic can compare recalled content against the summary that
    /// triggered its load.
    pub stub_summaries: HashMap<StubId, String>,
}

/// Result of running a single item in a single mode. Aggregated by the
/// reporter into per-workload and per-mode summaries. Deserializable so a
/// persisted `--trace-out` line can be loaded back and re-judged without
/// regenerating the answer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ItemResult {
    pub item_id: String,
    pub mode: RecallMode,
    pub question: String,
    pub answer: String,
    pub loaded_paths: Vec<String>,
    pub expected_paths: Vec<String>,
    /// recall@k over the final loaded set, counting only *body* fragments —
    /// the fraction of expected paths whose answer-bearing content is resident.
    /// A path resident only as a stub does not count (a stub is a pointer, not
    /// the content).
    pub recall_at_k: f32,
    /// Path-level recall counting stub residency too — the fraction of expected
    /// paths surfaced in the workspace at all, as stub or body. Always ≥
    /// `recall_at_k`; the gap is the progressive-disclosure headroom (paths
    /// surfaced as stubs but never upgraded to full content). Defaults to 0 so
    /// a pre-existing trace without this field still deserializes.
    #[serde(default)]
    pub stub_recall_at_k: f32,
    /// Fraction of the loaded set that's actually relevant (i.e. in
    /// `expected_paths`). Mathematically capped at `min(expected, k) / k`,
    /// so for single-needle NIAH this can never exceed `1/k`. Renamed from
    /// `precision_at_k` to avoid implying answer-quality.
    pub relevance_at_k: f32,
    /// Binary: was the top-1 retrieved fragment in `expected_paths`?
    /// More discriminating than relevance@k for single-needle workloads.
    pub precision_at_1: f32,
    /// Mean reciprocal rank — for each expected path, `1 / (rank + 1)` if
    /// found in `loaded_paths`, 0 otherwise. Averaged across expected
    /// paths. Captures *how high* each needle landed in the ranking.
    pub mrr: f32,
    /// Total tokens in loaded fragments (content).
    pub content_tokens: usize,
    /// Total tokens of stub summaries in the retrieval index (NOT in the
    /// model's context window). Reported per item for the corpus-pool size.
    pub index_pool_tokens: usize,
    /// content / (content + system + query + per-fragment provenance tags).
    /// Higher is better — more of the context window is doing useful work.
    pub context_efficiency: f32,
    /// Per the FalseRecallMetrics heuristic: share of loaded fragments with
    /// low stub-summary-to-content term overlap. Lower is better.
    pub false_recall_rate: f32,
    /// Answer-level pass/fail. 1.0 = correct, 0.0 = incorrect. For
    /// JudgeAgainst scoring this is the judge's 0-1 score.
    pub answer_score: f32,
    /// Free-form judge rationale (JudgeAgainst only; empty for ContainsNeedle).
    pub judge_rationale: String,
    /// True between generation and the post-gen judge phase for items whose
    /// scoring needs a model judge (`JudgeAgainst`). Generation leaves
    /// `answer_score`/`judge_rationale` unset and flags the item here; the
    /// judge phase scores it and clears the flag. `ContainsNeedle` items are
    /// scored inline (a local string check, no model call) and are never
    /// pending. Defaults to false so a pre-existing trace deserializes as
    /// already-judged.
    #[serde(default)]
    pub judge_pending: bool,
    /// Wall time for this item (ms).
    pub latency_ms: u64,
    /// Answer-model generation time (ms) within this item — the sum of all
    /// multi-pass completions. Filled by the bench's `TimingAdapter`; 0 when
    /// timing isn't wired (e.g. direct library callers of `run_item`).
    #[serde(default)]
    pub gen_ms: u64,
    /// Number of answer-model completions for this item (multi-pass count).
    #[serde(default)]
    pub gen_calls: u64,
    /// Judge-model time (ms) for this item. 0 for non-judged scoring.
    #[serde(default)]
    pub judge_ms: u64,
    /// Reference answer from the scoring config (JudgeAgainst only; empty
    /// for ContainsNeedle). Carried through so the trace can show what the
    /// judge was comparing against.
    pub reference_answer: String,
    /// Content of each loaded fragment, in load order, paired with its
    /// source locator. Lets a failure trace show exactly what context the
    /// model had. Truncated per-fragment to keep the JSONL manageable.
    pub loaded_fragments: Vec<LoadedFragment>,
}

/// Compact per-fragment view for trace output. Content is truncated so
/// JSONL lines stay readable; full content is always available via the
/// sqlite store if deeper inspection is needed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadedFragment {
    pub source: String,
    pub locator: String,
    pub tokens: usize,
    /// First ~1 KB of the fragment's content. Enough to see the head of
    /// most changelog entries / doc paragraphs without blowing up the log.
    pub content_preview: String,
}

pub fn run_item(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    answer_adapter: Box<dyn ModelAdapter>,
    prebuilt: Option<&PrebuiltIndex>,
) -> Result<ItemResult> {
    match prebuilt {
        Some(idx) => run_item_shared(item, mode, cfg, answer_adapter, idx),
        None => run_item_fresh(item, mode, cfg, answer_adapter),
    }
}

/// Run one item against a fresh per-item index built from `item.corpus`.
/// Used by workloads whose corpus is small and per-item (opencaw, niah).
fn run_item_fresh(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    answer_adapter: Box<dyn ModelAdapter>,
) -> Result<ItemResult> {
    let started = std::time::Instant::now();

    let mut embedder =
        FastEmbedProvider::bge_small().context("initialize bge-small for retriever")?;
    let dim = embedder.dimension();

    // SqliteStubStore records `(path, byte_offset, byte_length)` and
    // re-reads bodies from `corpus_root.join(path)` on `get_content`.
    // In-memory workloads (niah, opencaw's per-item JSON questions) have
    // no on-disk backing, so materialize the item's corpus into a tempdir
    // and attach it as the store's corpus_root. Tempdir lives for the
    // whole item; returned struct keeps it alive.
    let corpus_tmp = tempfile::tempdir().context("create per-item corpus tempdir")?;
    for doc in &item.corpus {
        let full = corpus_tmp.path().join(&doc.path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
        std::fs::write(&full, doc.content.as_bytes())
            .with_context(|| format!("write {}", full.display()))?;
    }

    let mut store = SqliteStubStore::in_memory(dim)
        .context("open in-memory sqlite store")?
        .with_corpus_root(corpus_tmp.path().to_path_buf());
    let mut vector_index = HnswVectorIndex::new();

    let mut stub_summaries: HashMap<StubId, String> = HashMap::new();

    let pipeline = IngestionPipeline::new();
    for doc in &item.corpus {
        let source_doc = SourceDocument {
            path: doc.path.clone(),
            content: doc.content.clone(),
            kind: doc.kind,
            mtime_unix_secs: 0,
        };
        let stubs = pipeline.ingest(source_doc);
        for (stub, _embed_text) in stubs {
            let stub_summary_text =
                format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
            let embedding = embedder
                .embed_document(vec![stub_summary_text.as_str()])
                .context("embed stub summary")?
                .into_iter()
                .next()
                .context("empty embedding batch")?;
            let stub_id = stub.id.clone();
            stub_summaries.insert(stub_id.clone(), stub.summary.clone());
            store
                .insert(stub.clone(), embedding.clone())
                .context("insert stub into store")?;
            vector_index.add(stub_id, embedding);
        }
    }

    // Hybrid (semantic + BM25) so the per-item path fuses lexical and
    // embedding scores like the proxy. BM25 is built over the just-ingested
    // store; `get_content` slices bodies from the per-item corpus tempdir.
    let bm25 = build_bm25_over_store(&store, stub_summaries.keys().cloned());
    let semantic = SemanticRetriever::new(embedder, store, vector_index);
    let retriever = HybridRetriever::balanced_shared(semantic, Arc::new(bm25));
    // The orchestrator's own embedder + index serve session-history recall only;
    // corpus recall (including thinking-trace recall) goes through the retriever
    // above. A bench item is a single turn with no session history, so the index
    // starts and stays empty.
    let session_embedder =
        FastEmbedProvider::bge_small().context("initialize bge-small for session recall")?;
    let session_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();

    let config = orchestrator_config(mode, cfg);
    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, SqliteStubStore> =
        DynamicRecallOrchestrator::new(
            retriever,
            session_embedder,
            session_index,
            provenance,
            answer_adapter,
            config,
        );

    let response = orchestrator
        .run_turn(&cfg.system_prompt, &item.question, &[], None)
        .context("run_turn failed")?;

    let loaded = orchestrator.loaded.clone();
    finalize_result(
        item,
        mode,
        cfg,
        started,
        &loaded,
        response.answer,
        &stub_summaries,
    )
}

/// Run one item against a shared, prebuilt index. All heavy state
/// (embeddings, store, HNSW) is reused across items.
fn run_item_shared(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    answer_adapter: Box<dyn ModelAdapter>,
    prebuilt: &PrebuiltIndex,
) -> Result<ItemResult> {
    let started = std::time::Instant::now();

    // Retriever store is the read-only view: even if the orchestrator were
    // later wired with `.with_store(...)` or ingestion code crept into a
    // shared-mode path, writes would be silently dropped so the prebuilt
    // sqlite file stays byte-identical across items.
    // Hybrid retrieval over the shared, prebuilt BM25 index (built once in
    // `load_prebuilt_index` / `build_in_memory_prebuilt`). The `Arc` clone is
    // cheap; the index is never mutated on this read-only path.
    let semantic = SemanticRetriever::new(
        prebuilt.embedder.clone(),
        ReadOnlyStore::new(&prebuilt.store),
        prebuilt.index.clone(),
    );
    let retriever = HybridRetriever::balanced_shared(semantic, prebuilt.bm25.clone());

    // The orchestrator's own embedder + index serve session-history recall only;
    // corpus recall (including thinking-trace recall) goes through the retriever
    // above. A bench item is a single turn with no session history, so the index
    // is per-item and stays empty. The embedder can be the shared bge-small
    // instance — the orchestrator serializes its calls.
    let session_embedder = prebuilt.embedder.clone();
    let session_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();

    let config = orchestrator_config(mode, cfg);
    // The `S` (consolidation store) slot stays unset: the orchestrator's
    // `store: Option<S>` defaults to None, so `persist_consolidation` is a
    // no-op. The ReadOnlyStore in the retriever enforces the same invariant
    // defensively — writes to the prebuilt sqlite are dropped in every path.
    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, ReadOnlyStore<SqliteStubStore>> =
        DynamicRecallOrchestrator::new(
            retriever,
            session_embedder,
            session_index,
            provenance,
            answer_adapter,
            config,
        );

    let response = orchestrator
        .run_turn(&cfg.system_prompt, &item.question, &[], None)
        .context("run_turn failed")?;

    let loaded = orchestrator.loaded.clone();
    finalize_result(
        item,
        mode,
        cfg,
        started,
        &loaded,
        response.answer,
        &prebuilt.stub_summaries,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_result(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    started: std::time::Instant,
    loaded: &[caw_core::RecallFragment],
    answer: String,
    stub_summaries: &HashMap<StubId, String>,
) -> Result<ItemResult> {
    let loaded_paths: Vec<String> = loaded.iter().map(|f| f.locator.source.clone()).collect();
    let metrics = retrieval_metrics(loaded, &item.expected_paths);
    let content_tokens: usize = loaded.iter().map(|f| f.tokens).sum();
    let provenance_overhead = loaded.len() * 15;
    let overhead_tokens =
        estimate_tokens(&cfg.system_prompt) + estimate_tokens(&item.question) + provenance_overhead;
    let context_efficiency = if content_tokens + overhead_tokens == 0 {
        0.0
    } else {
        content_tokens as f32 / (content_tokens + overhead_tokens) as f32
    };
    let index_pool_tokens: usize = stub_summaries.values().map(|s| estimate_tokens(s)).sum();
    let false_recall_rate = false_recall_rate_heuristic(loaded, stub_summaries);

    // Model judging (JudgeAgainst) is deferred to the post-generation phase
    // so the generation loop never blocks on a remote judge call. Local
    // needle scoring is free, so it stays inline.
    let (answer_score, judge_rationale, judge_pending) = score_local(&item.scoring, &answer);

    let reference_answer = match &item.scoring {
        Scoring::JudgeAgainst { reference_answer } => reference_answer.clone(),
        _ => String::new(),
    };

    // Truncation keeps the per-item trace line manageable in a JSONL file.
    // 2 KB / fragment captures the head of most relevant chunks; reach for
    // the sqlite store if a reviewer needs the full thing.
    const PREVIEW_CAP: usize = 2048;
    let loaded_fragments: Vec<LoadedFragment> = loaded
        .iter()
        .map(|f| LoadedFragment {
            source: f.locator.source.clone(),
            locator: f.locator.locator.clone(),
            tokens: f.tokens,
            content_preview: truncate_preview(&f.content, PREVIEW_CAP),
        })
        .collect();

    Ok(ItemResult {
        item_id: item.id.clone(),
        mode,
        question: item.question.clone(),
        answer,
        loaded_paths,
        expected_paths: item.expected_paths.clone(),
        recall_at_k: metrics.recall_at_k,
        stub_recall_at_k: metrics.stub_recall_at_k,
        relevance_at_k: metrics.relevance_at_k,
        precision_at_1: metrics.precision_at_1,
        mrr: metrics.mrr,
        content_tokens,
        index_pool_tokens,
        context_efficiency,
        false_recall_rate,
        answer_score,
        judge_rationale,
        judge_pending,
        latency_ms: started.elapsed().as_millis() as u64,
        // Phase timings are populated by the caller (main.rs) from the
        // TimingAdapter counters after run_item returns; runner has no handle
        // to them. Default to 0 so direct library callers still compile.
        gen_ms: 0,
        gen_calls: 0,
        judge_ms: 0,
        reference_answer,
        loaded_fragments,
    })
}

fn truncate_preview(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    let mut end = cap;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 16);
    out.push_str(&s[..end]);
    out.push_str("\n…[truncated]");
    out
}

#[derive(Debug, Clone, Copy)]
struct RetrievalMetrics {
    recall_at_k: f32,
    stub_recall_at_k: f32,
    relevance_at_k: f32,
    precision_at_1: f32,
    mrr: f32,
}

fn orchestrator_config(mode: RecallMode, cfg: &RunnerConfig) -> DynamicRecallConfig {
    let thresholds = RecallThresholds {
        load: cfg.load_threshold,
        unload: cfg.unload_threshold,
    };
    match mode {
        RecallMode::On => DynamicRecallConfig {
            max_candidates: cfg.max_candidates,
            // Disable the ambiguity gate: it's a chat UX feature that shows a
            // candidate list when many files match, expecting the user to name
            // the specific file they want. In a bench, nothing names files and
            // the orchestrator loads nothing, producing 0 recall every time.
            max_initial_fragments: cfg.max_candidates,
            thresholds,
            max_workspace_tokens: cfg.max_workspace_tokens,
            max_recall_iterations: cfg.max_recall_iterations,
            relevance_decay_rate: 0.8,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
            // Run the answer model cooperative: inject the probe/annotation/
            // line-range instructions so the trace-driven loop can actually
            // fire. Under the default `Auto` mode a non-reasoning adapter
            // (groq/openai-compatible) reports no reasoning caps, so injection
            // is skipped and the whole loop is inert — recall_on collapses to
            // basic-RAG-from-initial-stubs, indistinguishable from recall_off.
            // Verified cooperative by caw-bench-coop on llama-3.1-8b-instant.
            cooperation_mode: CooperationMode::Cooperative,
            ..Default::default()
        },
        RecallMode::Off => DynamicRecallConfig {
            max_candidates: cfg.max_candidates,
            max_initial_fragments: cfg.max_candidates,
            thresholds,
            max_workspace_tokens: cfg.max_workspace_tokens,
            // No multi-pass, no probes, no trace recall: this is the
            // "basic RAG" baseline recall is measured against.
            max_recall_iterations: 0,
            relevance_decay_rate: 0.8,
            enable_thinking_trace_recall: false,
            enable_probe_recall: false,
            ..Default::default()
        },
    }
}

/// Expected paths are typically rooted at the corpus (e.g. `pkg/changelog`).
/// Loaded paths come from stubs that may be rooted at the corpus (new
/// indexes) or at the process CWD that built the index (legacy indexes,
/// e.g. `opencaw-corpora/sysdoc/pkg/changelog`). Suffix-match handles both
/// without forcing a rebuild: a loaded path counts as a match when the
/// expected path is a path-component-aligned suffix of it.
fn path_matches(loaded: &str, expected: &str) -> bool {
    if loaded == expected {
        return true;
    }
    // Align on a path separator so `foo/bar.md` doesn't spuriously match
    // `xfoo/bar.md`. Also allow an exact `{expected}` at the end of
    // `{anything/}{expected}`.
    loaded.ends_with(expected)
        && loaded.len() > expected.len()
        && loaded.as_bytes()[loaded.len() - expected.len() - 1] == b'/'
}

/// A stub fragment is a pointer (summary/outline), not answer-bearing content.
/// `read_range(id, "stub")` sets the locator string to "stub"; full/range reads
/// set it to the range. So the locator is an explicit residency marker — no
/// content-sniffing heuristic needed.
fn is_stub_fragment(f: &caw_core::RecallFragment) -> bool {
    f.locator.locator.eq_ignore_ascii_case("stub")
}

fn path_recall(loaded: &[&str], expected: &[String]) -> f32 {
    if expected.is_empty() {
        return 1.0;
    }
    let matched = expected
        .iter()
        .filter(|exp| loaded.iter().any(|l| path_matches(l, exp)))
        .count();
    matched as f32 / expected.len() as f32
}

/// Recall, relevance, precision@1, and mrr are computed over *body* fragments
/// only: a stub holds no answer, so a path resident only as a stub is not
/// recalled in the sense the metric is meant to capture (an earlier version
/// counted stub residency as a hit, reporting recall@k=1.0 for items whose
/// answer body was never loaded). `stub_recall_at_k` keeps the path-level
/// "surfaced at all" number; the gap (stub_recall_at_k − recall_at_k) is the
/// progressive-disclosure headroom — paths surfaced as stubs but never upgraded.
fn retrieval_metrics(loaded: &[caw_core::RecallFragment], expected: &[String]) -> RetrievalMetrics {
    let body_paths: Vec<&str> = loaded
        .iter()
        .filter(|f| !is_stub_fragment(f))
        .map(|f| f.locator.source.as_str())
        .collect();
    let all_paths: Vec<&str> = loaded.iter().map(|f| f.locator.source.as_str()).collect();

    let recall_at_k = path_recall(&body_paths, expected);
    let stub_recall_at_k = path_recall(&all_paths, expected);

    let relevance_at_k = if body_paths.is_empty() {
        0.0
    } else {
        let matched = body_paths
            .iter()
            .filter(|l| expected.iter().any(|exp| path_matches(l, exp)))
            .count();
        matched as f32 / body_paths.len() as f32
    };
    let precision_at_1 = match body_paths.first() {
        Some(top) if expected.iter().any(|exp| path_matches(top, exp)) => 1.0,
        _ => 0.0,
    };
    let mrr = if expected.is_empty() {
        1.0
    } else {
        let total: f32 = expected
            .iter()
            .map(
                |exp| match body_paths.iter().position(|l| path_matches(l, exp)) {
                    Some(rank) => 1.0 / (rank as f32 + 1.0),
                    None => 0.0,
                },
            )
            .sum();
        total / expected.len() as f32
    };
    RetrievalMetrics {
        recall_at_k,
        stub_recall_at_k,
        relevance_at_k,
        precision_at_1,
        mrr,
    }
}

/// Share of loaded fragments with low stub-summary-to-content term overlap.
/// Matches the FalseRecallMetrics default threshold in caw-eval (0.15).
fn false_recall_rate_heuristic(
    loaded: &[caw_core::RecallFragment],
    summaries: &HashMap<StubId, String>,
) -> f32 {
    const THRESHOLD: f32 = 0.15;
    if loaded.is_empty() {
        return 0.0;
    }

    let mut flagged = 0usize;
    let mut evaluated = 0usize;
    for frag in loaded {
        let Some(summary) = summaries.get(&frag.stub_id) else {
            continue;
        };
        let overlap = term_overlap(summary, &frag.content);
        if overlap < THRESHOLD {
            flagged += 1;
        }
        evaluated += 1;
    }
    if evaluated == 0 {
        0.0
    } else {
        flagged as f32 / evaluated as f32
    }
}

fn term_overlap(a: &str, b: &str) -> f32 {
    let terms_a: HashSet<String> = tokenize(a).into_iter().collect();
    let terms_b: HashSet<String> = tokenize(b).into_iter().collect();
    if terms_a.is_empty() || terms_b.is_empty() {
        return 0.0;
    }
    let intersection = terms_a.intersection(&terms_b).count() as f32;
    intersection / terms_a.len().min(terms_b.len()) as f32
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2)
        .map(String::from)
        .collect()
}

fn estimate_tokens(text: &str) -> usize {
    count_tokens_cl100k(text)
}

/// Score what can be scored without a model. Returns
/// `(answer_score, judge_rationale, judge_pending)`.
///
/// `ContainsNeedle` is a local substring check, so it's scored here and is
/// never pending. `JudgeAgainst` needs a model judge, which runs in the
/// post-generation phase — so this leaves the score at 0.0 and marks the
/// item pending. The 0.0 is a placeholder that the judge phase overwrites
/// before any report reads it; `judge_pending` is the authoritative
/// "not yet scored" signal, not the 0.0.
fn score_local(scoring: &Scoring, answer: &str) -> (f32, String, bool) {
    match scoring {
        Scoring::ContainsNeedle { needle } => {
            let pass = answer.to_lowercase().contains(&needle.to_lowercase());
            (if pass { 1.0 } else { 0.0 }, String::new(), false)
        }
        Scoring::JudgeAgainst { .. } => (0.0, String::new(), true),
    }
}
