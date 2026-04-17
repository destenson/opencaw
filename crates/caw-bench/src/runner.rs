use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use caw_adapters::ClaudeCodeAdapter;
use caw_core::{EmbeddingProvider, RecallThresholds, StubId, StubStore, VectorIndex};
// RecallThresholds is constructed inline in orchestrator_config from
// RunnerConfig's load_threshold / unload_threshold — the type import above
// is kept for the config builder pattern.
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};
use caw_provenance::InMemoryProvenanceStore;

use crate::judge::{JudgeVerdict, judge_answer};
use crate::workload::{RecallMode, Scoring, WorkloadItem};

const DEFAULT_ANSWER_MODEL: &str = "sonnet";
const DEFAULT_JUDGE_MODEL: &str = "haiku";

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
    /// Answering model for the comparison. Same model is used in both
    /// recall-on and recall-off so the delta isolates recall's contribution.
    /// Passed to `claude --model`; typical values: "sonnet", "opus", "haiku".
    pub answer_model: String,
    /// Judge model used for JudgeAgainst scoring. Cheap and separate from
    /// the answering model so judging doesn't bias the comparison.
    pub judge_model: String,
    /// How many total questions to run (trimmed workload); None = run all.
    pub limit: Option<usize>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        // Permissive thresholds match RecallThresholds::permissive() in
        // caw-core. Tuning these per workload is a future calibration
        // exercise; the bench harness is the instrument for that.
        Self {
            system_prompt:
                "You are a helpful assistant. Use the recalled workspace context to answer \
                 the question accurately and concisely. Cite source locators from the recalled \
                 context when they support your answer."
                    .to_string(),
            top_k: 5,
            max_workspace_tokens: 12_000,
            max_recall_iterations: 3,
            load_threshold: 0.3,
            unload_threshold: 0.2,
            answer_model: DEFAULT_ANSWER_MODEL.to_string(),
            judge_model: DEFAULT_JUDGE_MODEL.to_string(),
            limit: None,
        }
    }
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
    pub recall_at_k: f32,
    /// precision@k over the final loaded set.
    pub precision_at_k: f32,
    /// Total tokens in loaded fragments (content).
    pub content_tokens: usize,
    /// Approximate tokens spent on stubs (summaries + outlines) in the workspace.
    pub stub_tokens: usize,
    /// content_tokens / (content_tokens + stub_tokens + overhead). Higher is
    /// better — more of the context window is doing useful work.
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
) -> Result<ItemResult> {
    let started = std::time::Instant::now();

    // Fresh embedder + store + orchestrator per item keeps runs independent.
    // Cost is real (a few hundred ms per item) but predictable and eliminates
    // state leakage between questions.
    let mut embedder =
        FastEmbedProvider::bge_small().context("initialize bge-small for retriever")?;
    let dim = embedder.dimension();
    let mut store = SqliteStubStore::in_memory(dim).context("open in-memory sqlite store")?;
    let mut vector_index = HnswVectorIndex::new();

    // Capture stub summaries as we ingest; the false-recall heuristic needs
    // them to compare against recalled content, but the Retriever trait
    // doesn't expose summaries directly.
    let mut stub_summaries: HashMap<StubId, String> = HashMap::new();

    let pipeline = IngestionPipeline::new();
    let mut stub_tokens: usize = 0;
    for doc in &item.corpus {
        let source_doc = SourceDocument {
            path: doc.path.clone(),
            content: doc.content.clone(),
            kind: doc.kind,
            mtime_unix_secs: 0,
        };
        let stubs = pipeline.ingest(source_doc);
        for stub in stubs {
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
            // Stub-side tokens accounting uses the summary text itself rather
            // than token_estimate (which counts the source content, not the
            // stub representation).
            stub_tokens += estimate_tokens(&stub_summary_text);
            stub_summaries.insert(stub_id.clone(), stub.summary.clone());
            store
                .insert(stub.clone(), embedding.clone(), doc.content.clone())
                .context("insert stub into store")?;
            vector_index.add(stub_id, embedding);
        }
    }

    let retriever = SemanticRetriever::new(embedder, store, vector_index);

    // The orchestrator's trace embedder + index are only exercised when
    // thinking-trace recall fires. In recall-off mode they stay idle.
    let trace_embedder = FastEmbedProvider::bge_small().context("initialize bge-small for trace")?;
    let trace_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();

    let adapter = ClaudeCodeAdapter::builder()
        .model(&cfg.answer_model)
        .build();

    let config = orchestrator_config(mode, cfg);

    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, SqliteStubStore> =
        DynamicRecallOrchestrator::new(
            retriever,
            trace_embedder,
            trace_index,
            provenance,
            adapter,
            config,
        );

    let response = orchestrator
        .run_turn(&cfg.system_prompt, &item.question)
        .context("run_turn failed")?;

    // Snapshot the final loaded set for metric computation.
    let loaded = orchestrator.loaded.clone();
    let loaded_paths: Vec<String> = loaded.iter().map(|f| f.locator.source.clone()).collect();

    let (recall_at_k, precision_at_k) = recall_precision(&loaded_paths, &item.expected_paths);
    let content_tokens: usize = loaded.iter().map(|f| f.tokens).sum();
    let overhead_tokens = estimate_tokens(&cfg.system_prompt) + estimate_tokens(&item.question);
    let context_efficiency = if content_tokens + stub_tokens + overhead_tokens == 0 {
        0.0
    } else {
        content_tokens as f32 / (content_tokens + stub_tokens + overhead_tokens) as f32
    };

    let false_recall_rate = false_recall_rate_heuristic(&loaded, &stub_summaries);

    let (answer_score, judge_rationale) =
        score_answer(&item.scoring, &response.answer, &cfg.judge_model)?;

    Ok(ItemResult {
        item_id: item.id.clone(),
        mode,
        question: item.question.clone(),
        answer: response.answer,
        loaded_paths,
        expected_paths: item.expected_paths.clone(),
        recall_at_k,
        precision_at_k,
        content_tokens,
        stub_tokens,
        context_efficiency,
        false_recall_rate,
        answer_score,
        judge_rationale,
        latency_ms: started.elapsed().as_millis() as u64,
    })
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

fn recall_precision(loaded: &[String], expected: &[String]) -> (f32, f32) {
    let loaded_set: HashSet<&String> = loaded.iter().collect();
    let expected_set: HashSet<&String> = expected.iter().collect();
    let intersection = loaded_set.intersection(&expected_set).count();

    let recall = if expected.is_empty() {
        1.0
    } else {
        intersection as f32 / expected.len() as f32
    };
    let precision = if loaded.is_empty() {
        0.0
    } else {
        intersection as f32 / loaded.len() as f32
    };
    (recall, precision)
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
    judge_model: &str,
) -> Result<(f32, String)> {
    match scoring {
        Scoring::ContainsNeedle { needle } => {
            let pass = answer.to_lowercase().contains(&needle.to_lowercase());
            Ok((if pass { 1.0 } else { 0.0 }, String::new()))
        }
        Scoring::JudgeAgainst { reference_answer } => {
            let verdict: JudgeVerdict = judge_answer(judge_model, answer, reference_answer)
                .context("judge invocation failed")?;
            Ok((verdict.score, verdict.rationale))
        }
    }
}
