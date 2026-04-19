use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::{EmbeddingProvider, ModelAdapter, RecallThresholds, StubId, StubStore, VectorIndex};
use caw_index::{
    CandleEmbeddingProvider, FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore,
};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};

use crate::judge::{JudgeVerdict, judge_answer};
use crate::shared::{ReadOnlyStore, SharedEmbedder, SharedIndex, SharedStore};
use crate::workload::{RecallMode, Scoring, WorkloadItem};

pub struct RunnerConfig {
    pub system_prompt: String,
    pub top_k: usize,
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
            system_prompt:
                "You are a helpful assistant. Use the recalled workspace context to answer \
                 the question accurately and concisely. Cite source locators from the recalled \
                 context when they support your answer."
                    .to_string(),
            top_k: 5,
            max_workspace_tokens: 2_000,
            max_recall_iterations: 3,
            load_threshold: 0.3,
            unload_threshold: 0.2,
            limit: None,
        }
    }
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
    /// Stub summaries keyed by id — populated at load time so the false-recall
    /// heuristic can compare recalled content against the summary that
    /// triggered its load.
    pub stub_summaries: HashMap<StubId, String>,
}

/// Result of running a single item in a single mode. Aggregated by the
/// reporter into per-workload and per-mode summaries.
#[derive(Debug, Clone)]
pub struct ItemResult {
    pub item_id: String,
    pub mode: RecallMode,
    pub question: String,
    pub answer: String,
    pub loaded_paths: Vec<String>,
    pub expected_paths: Vec<String>,
    /// recall@k over the final loaded set (after all orchestrator iterations).
    /// Fraction of expected paths that ended up in the loaded set.
    pub recall_at_k: f32,
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
    /// Wall time for this item (ms).
    pub latency_ms: u64,
}

pub fn run_item(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    answer_adapter: Box<dyn ModelAdapter>,
    judge_adapter: &dyn ModelAdapter,
    prebuilt: Option<&PrebuiltIndex>,
) -> Result<ItemResult> {
    match prebuilt {
        Some(idx) => run_item_shared(item, mode, cfg, answer_adapter, judge_adapter, idx),
        None => run_item_fresh(item, mode, cfg, answer_adapter, judge_adapter),
    }
}

/// Run one item against a fresh per-item index built from `item.corpus`.
/// Used by workloads whose corpus is small and per-item (opencaw, niah).
fn run_item_fresh(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    answer_adapter: Box<dyn ModelAdapter>,
    judge_adapter: &dyn ModelAdapter,
) -> Result<ItemResult> {
    let started = std::time::Instant::now();

    let mut embedder =
        FastEmbedProvider::bge_small().context("initialize bge-small for retriever")?;
    let dim = embedder.dimension();
    let mut store = SqliteStubStore::in_memory(dim).context("open in-memory sqlite store")?;
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
        let pairs = pipeline.ingest(source_doc);
        for (stub, chunk_content) in pairs {
            let stub_summary_text = format!(
                "{} {} {}",
                stub.path,
                stub.summary,
                stub.outline.join(" ")
            );
            let embedding = embedder
                .embed_document(vec![stub_summary_text.as_str()])
                .context("embed stub summary")?
                .into_iter()
                .next()
                .context("empty embedding batch")?;
            let stub_id = stub.id.clone();
            stub_summaries.insert(stub_id.clone(), stub.summary.clone());
            store
                .insert(stub.clone(), embedding.clone(), chunk_content)
                .context("insert stub into store")?;
            vector_index.add(stub_id, embedding);
        }
    }

    let retriever = SemanticRetriever::new(embedder, store, vector_index);
    let trace_embedder =
        FastEmbedProvider::bge_small().context("initialize bge-small for trace")?;
    let trace_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();

    let config = orchestrator_config(mode, cfg);
    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, SqliteStubStore> =
        DynamicRecallOrchestrator::new(
            retriever,
            trace_embedder,
            trace_index,
            provenance,
            answer_adapter,
            config,
        );

    let response = orchestrator
        .run_turn(&cfg.system_prompt, &item.question)
        .context("run_turn failed")?;

    let loaded = orchestrator.loaded.clone();
    Ok(finalize_result(
        item,
        mode,
        cfg,
        started,
        &loaded,
        response.answer,
        &stub_summaries,
        judge_adapter,
    )?)
}

/// Run one item against a shared, prebuilt index. All heavy state
/// (embeddings, store, HNSW) is reused across items.
fn run_item_shared(
    item: &WorkloadItem,
    mode: RecallMode,
    cfg: &RunnerConfig,
    answer_adapter: Box<dyn ModelAdapter>,
    judge_adapter: &dyn ModelAdapter,
    prebuilt: &PrebuiltIndex,
) -> Result<ItemResult> {
    let started = std::time::Instant::now();

    // Retriever store is the read-only view: even if the orchestrator were
    // later wired with `.with_store(...)` or ingestion code crept into a
    // shared-mode path, writes would be silently dropped so the prebuilt
    // sqlite file stays byte-identical across items.
    let retriever = SemanticRetriever::new(
        prebuilt.embedder.clone(),
        ReadOnlyStore::new(&prebuilt.store),
        prebuilt.index.clone(),
    );

    // Trace recall uses a distinct embedder + index per turn; the index is
    // always per-item (trace content is per-turn), but the embedder can be
    // the same bge-small instance — the orchestrator serializes its calls.
    let trace_embedder = prebuilt.embedder.clone();
    let trace_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();

    let config = orchestrator_config(mode, cfg);
    // The `S` (consolidation store) slot stays unset: the orchestrator's
    // `store: Option<S>` defaults to None, so `persist_consolidation` is a
    // no-op. The ReadOnlyStore in the retriever enforces the same invariant
    // defensively — writes to the prebuilt sqlite are dropped in every path.
    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, ReadOnlyStore<SqliteStubStore>> =
        DynamicRecallOrchestrator::new(
            retriever,
            trace_embedder,
            trace_index,
            provenance,
            answer_adapter,
            config,
        );

    let response = orchestrator
        .run_turn(&cfg.system_prompt, &item.question)
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
        judge_adapter,
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
    judge_adapter: &dyn ModelAdapter,
) -> Result<ItemResult> {
    let loaded_paths: Vec<String> = loaded.iter().map(|f| f.locator.source.clone()).collect();
    let metrics = retrieval_metrics(&loaded_paths, &item.expected_paths);
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

    let (answer_score, judge_rationale) = score_answer(&item.scoring, &answer, judge_adapter)?;

    Ok(ItemResult {
        item_id: item.id.clone(),
        mode,
        question: item.question.clone(),
        answer,
        loaded_paths,
        expected_paths: item.expected_paths.clone(),
        recall_at_k: metrics.recall_at_k,
        relevance_at_k: metrics.relevance_at_k,
        precision_at_1: metrics.precision_at_1,
        mrr: metrics.mrr,
        content_tokens,
        index_pool_tokens,
        context_efficiency,
        false_recall_rate,
        answer_score,
        judge_rationale,
        latency_ms: started.elapsed().as_millis() as u64,
    })
}

#[derive(Debug, Clone, Copy)]
struct RetrievalMetrics {
    recall_at_k: f32,
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
            top_k: cfg.top_k,
            thresholds,
            max_workspace_tokens: cfg.max_workspace_tokens,
            max_recall_iterations: cfg.max_recall_iterations,
            relevance_decay_rate: 0.8,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
        },
        RecallMode::Off => DynamicRecallConfig {
            top_k: cfg.top_k,
            thresholds,
            max_workspace_tokens: cfg.max_workspace_tokens,
            // No multi-pass, no probes, no trace recall: this is the
            // "basic RAG" baseline the thesis is measured against.
            max_recall_iterations: 0,
            relevance_decay_rate: 0.8,
            enable_thinking_trace_recall: false,
            enable_probe_recall: false,
        },
    }
}

fn retrieval_metrics(loaded: &[String], expected: &[String]) -> RetrievalMetrics {
    let loaded_set: HashSet<&String> = loaded.iter().collect();
    let expected_set: HashSet<&String> = expected.iter().collect();
    let intersection = loaded_set.intersection(&expected_set).count();

    let recall_at_k = if expected.is_empty() {
        1.0
    } else {
        intersection as f32 / expected.len() as f32
    };
    let relevance_at_k = if loaded.is_empty() {
        0.0
    } else {
        intersection as f32 / loaded.len() as f32
    };
    let precision_at_1 = match loaded.first() {
        Some(top) if expected_set.contains(top) => 1.0,
        _ => 0.0,
    };
    let mrr = if expected.is_empty() {
        1.0
    } else {
        let total: f32 = expected
            .iter()
            .map(|exp| match loaded.iter().position(|l| l == exp) {
                Some(rank) => 1.0 / (rank as f32 + 1.0),
                None => 0.0,
            })
            .sum();
        total / expected.len() as f32
    };
    RetrievalMetrics {
        recall_at_k,
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
    text.split_whitespace().count().max(1)
}

fn score_answer(
    scoring: &Scoring,
    answer: &str,
    judge_adapter: &dyn ModelAdapter,
) -> Result<(f32, String)> {
    match scoring {
        Scoring::ContainsNeedle { needle } => {
            let pass = answer.to_lowercase().contains(&needle.to_lowercase());
            Ok((if pass { 1.0 } else { 0.0 }, String::new()))
        }
        Scoring::JudgeAgainst { reference_answer } => {
            let verdict: JudgeVerdict = judge_answer(judge_adapter, answer, reference_answer)
                .context("judge invocation failed")?;
            Ok((verdict.score, verdict.rationale))
        }
    }
}
