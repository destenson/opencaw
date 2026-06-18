//! `caw-bench-coop` — per-model cooperation calibration harness.
//!
//! Runs a Q&A workload in three modes for a given model:
//!
//!   * **baseline** (recall_off): single-shot RAG, no multi-pass
//!   * **transparent** (recall_on, CooperationMode::Transparent): multi-pass
//!     recall with thinking-trace extraction, no probe/annotation instructions
//!   * **cooperative** (recall_on, CooperationMode::Cooperative): multi-pass
//!     recall with probe/annotation instructions injected unconditionally
//!
//! For each mode a `SessionEvaluator` is wired into the orchestrator to
//! capture probe/annotation compliance metrics. Recall@k (path-based)
//! measures whether cooperative mode actually improves retrieval quality for
//! this model.
//!
//! Outputs `{out_dir}/{model_slug}/coop-report.json`.
//!
//! The tune binary (`caw-bench-tune`) reads these reports alongside intent
//! and sweep results to emit a final recommended `cooperation_mode` per model.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;

use caw_bench::adapter_factory::{AdapterKind, AdapterSpec, ModelRole, build};
use caw_bench::opencaw;
use caw_bench::runner::build_in_memory_prebuilt;
use caw_bench::shared::ReadOnlyStore;
use caw_bench::workload::WorkloadItem;
use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::{CawError, ModelAdapter};
use caw_eval::SessionEvaluator;
use caw_index::{HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_orchestrator::dynamic::{
    CooperationMode, DynamicRecallConfig, DynamicRecallOrchestrator,
};
use tracing::warn;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "caw-bench-coop",
    about = "Calibrate per-model probe/annotation compliance"
)]
struct Cli {
    /// Model name. Format depends on --adapter. Defaults to the adapter's
    /// built-in default (e.g. llama-3.3-70b-versatile for groq).
    #[arg(long)]
    model: Option<String>,

    /// Adapter type.
    #[arg(long, value_enum, default_value_t = AdapterKind::Ollama)]
    adapter: AdapterKind,

    /// Ollama base URL.
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// vLLM / OpenAI-compatible base URL.
    #[arg(long, default_value = "http://localhost:8000")]
    openai_url: String,

    /// Sampling temperature. 0.0 = deterministic; higher values add noise but
    /// may reveal cooperation behaviour that greedy decoding suppresses.
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Cap Ollama context window (num_ctx). Useful for reducing VRAM when
    /// running multiple models.
    #[arg(long)]
    num_ctx: Option<u32>,

    /// Path to an opencaw checkout. The whole repo becomes the retrieval corpus.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Custom Q&A JSON file (opencaw format). When omitted uses the embedded copy.
    #[arg(long)]
    qa_file: Option<PathBuf>,

    /// Cap on items to run. Calibration needs far fewer items than a full sweep;
    /// 8–16 is usually sufficient for a stable probe-rate estimate.
    #[arg(long, default_value = "12")]
    limit: usize,

    /// Where to write the report. The binary appends `/{model_slug}/coop-report.json`.
    #[arg(long, default_value = "bench-results/coop")]
    out_dir: PathBuf,

    /// Hysteresis load threshold (same default as the bench runner).
    #[arg(long, default_value = "0.3")]
    load_threshold: f32,

    /// Hysteresis unload threshold.
    #[arg(long, default_value = "0.2")]
    unload_threshold: f32,

    /// Max workspace tokens per turn.
    #[arg(long, default_value = "2000")]
    max_workspace_tokens: usize,
}

// ── Report types ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ModeSummary {
    mean_recall_at_k: f32,
    probes_per_turn: f32,
    useful_probes_pct: f32,
    annotations_per_turn: f32,
    annotation_quality: f32,
    /// Number of items in this mode where the answer model degenerated.
    /// Surfaced so a silently-shrinking n is visible (BUGS "Reproducibility").
    degenerate_count: usize,
}

#[derive(Debug, Serialize)]
struct CoopReport {
    model: String,
    item_count: usize,
    baseline: ModeSummary,
    transparent: ModeSummary,
    cooperative: ModeSummary,
    /// Empirically recommended mode for this model.
    recommendation: String,
    /// Human-readable explanation of the recommendation.
    recommendation_reason: String,
}

// ── Per-item runner ───────────────────────────────────────────────────────────

struct ItemOutcome {
    recall_at_k: f32,
    probes_per_turn: f32,
    useful_probes_pct: f32,
    annotations_per_turn: f32,
    annotation_quality: f32,
    /// True when the answer model degenerated on this item. The cooperation
    /// metrics (probes/annotations emitted before the turn failed) and recall
    /// are still recorded; only the answer text was lost. Surfaced so a
    /// degenerate item is counted, not silently dropped (the prior `?`
    /// aborted the whole run — BUGS "Reproducibility").
    degenerate: bool,
}

fn run_item_coop(
    item: &WorkloadItem,
    coop_mode: CooperationMode,
    recall_on: bool,
    adapter: Box<dyn ModelAdapter>,
    prebuilt: &caw_bench::runner::PrebuiltIndex,
    cfg: &Cli,
) -> Result<ItemOutcome> {
    let retriever = SemanticRetriever::new(
        prebuilt.embedder.clone(),
        ReadOnlyStore::new(&prebuilt.store),
        prebuilt.index.clone(),
    );
    // Orchestrator's own embedder + index are for session-history recall only;
    // corpus recall goes through the retriever. Single-turn item, so empty.
    let session_embedder = prebuilt.embedder.clone();
    let session_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();

    use caw_core::RecallThresholds;
    let thresholds = RecallThresholds {
        load: cfg.load_threshold,
        unload: cfg.unload_threshold,
    };

    // max_initial_fragments == max_candidates disables the ambiguity gate.
    // The gate is a chat UX feature; in a bench the model never emits file
    // paths to exit the candidate-list path, so the gate produces 0 recall.
    let max_candidates = 20usize;
    let config = if recall_on {
        DynamicRecallConfig {
            max_candidates,
            max_initial_fragments: max_candidates,
            thresholds,
            max_workspace_tokens: cfg.max_workspace_tokens,
            max_recall_iterations: 3,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
            cooperation_mode: coop_mode,
            ..Default::default()
        }
    } else {
        DynamicRecallConfig {
            max_candidates,
            max_initial_fragments: max_candidates,
            thresholds,
            max_workspace_tokens: cfg.max_workspace_tokens,
            max_recall_iterations: 0,
            enable_thinking_trace_recall: false,
            enable_probe_recall: false,
            cooperation_mode: CooperationMode::Transparent,
            ..Default::default()
        }
    };

    let evaluator = SessionEvaluator::builder().build();

    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, ReadOnlyStore<SqliteStubStore>> =
        DynamicRecallOrchestrator::new(
            retriever,
            session_embedder,
            session_index,
            provenance,
            adapter,
            config,
        )
        .with_evaluator(evaluator);

    let system = "Use the recalled workspace context to answer the question accurately \
                  and concisely.";
    // A degenerate turn still emitted probes/annotations before the model
    // choked, so the cooperation metrics are worth recording — only the answer
    // text was lost. Catch the degenerate error and mark the outcome instead
    // of `?`-propagating, which used to abort the entire run (BUGS
    // "Reproducibility"). Hard errors (embed/retrieve/network) still
    // propagate: those are infrastructure failures, not model degenerations.
    let degenerate = match orchestrator.run_turn(system, &item.question, &[], None) {
        Ok(_) => false,
        Err(CawError::DegenerateOutput { .. }) => {
            warn!(
                item = %item.id,
                coop_mode = ?coop_mode,
                "degenerate output — recording outcome, not aborting the run"
            );
            true
        }
        Err(e) => return Err(e).context("run_turn failed"),
    };

    let loaded = orchestrator.loaded.clone();
    let coop = orchestrator
        .take_evaluator()
        .expect("evaluator was set")
        .cooperation_metrics();

    let loaded_paths: Vec<String> = loaded.iter().map(|f| f.locator.source.clone()).collect();
    let recall_at_k = recall_at_k(&loaded_paths, &item.expected_paths);

    Ok(ItemOutcome {
        recall_at_k,
        probes_per_turn: coop.probes_per_turn,
        useful_probes_pct: coop.useful_probes_pct,
        annotations_per_turn: coop.annotations_per_turn,
        annotation_quality: coop.annotation_quality,
        degenerate,
    })
}

// ── Scoring helpers ───────────────────────────────────────────────────────────

fn recall_at_k(loaded: &[String], expected: &[String]) -> f32 {
    if expected.is_empty() {
        return 1.0;
    }
    let hits = expected
        .iter()
        .filter(|exp| loaded.iter().any(|l| path_matches(l, exp)))
        .count();
    hits as f32 / expected.len() as f32
}

fn path_matches(loaded: &str, expected: &str) -> bool {
    if loaded == expected {
        return true;
    }
    loaded.ends_with(expected)
        && loaded.len() > expected.len()
        && loaded.as_bytes()[loaded.len() - expected.len() - 1] == b'/'
}

// ── Aggregation ───────────────────────────────────────────────────────────────

fn summarize(outcomes: &[ItemOutcome]) -> ModeSummary {
    if outcomes.is_empty() {
        return ModeSummary {
            mean_recall_at_k: 0.0,
            probes_per_turn: 0.0,
            useful_probes_pct: 0.0,
            annotations_per_turn: 0.0,
            annotation_quality: 0.0,
            degenerate_count: 0,
        };
    }
    let n = outcomes.len() as f32;
    let degenerate_count = outcomes.iter().filter(|o| o.degenerate).count();
    ModeSummary {
        mean_recall_at_k: outcomes.iter().map(|o| o.recall_at_k).sum::<f32>() / n,
        probes_per_turn: outcomes.iter().map(|o| o.probes_per_turn).sum::<f32>() / n,
        useful_probes_pct: outcomes.iter().map(|o| o.useful_probes_pct).sum::<f32>() / n,
        annotations_per_turn: outcomes.iter().map(|o| o.annotations_per_turn).sum::<f32>() / n,
        annotation_quality: outcomes.iter().map(|o| o.annotation_quality).sum::<f32>() / n,
        degenerate_count,
    }
}

fn recommend(
    baseline: &ModeSummary,
    transparent: &ModeSummary,
    cooperative: &ModeSummary,
) -> (String, String) {
    // Primary signal: recall delta vs baseline.
    let coop_delta = cooperative.mean_recall_at_k - baseline.mean_recall_at_k;
    let trans_delta = transparent.mean_recall_at_k - baseline.mean_recall_at_k;

    // Secondary: cooperative mode is only worth its instruction overhead if the
    // model actually emits useful probes (> 40% useful rate) OR annotations
    // (> 0.5 per turn). Below both thresholds the extra prompt tokens are waste.
    let useful_probe_threshold = 0.40;
    let annotation_threshold = 0.50;
    let coop_engages = cooperative.useful_probes_pct > useful_probe_threshold
        || cooperative.annotations_per_turn > annotation_threshold;

    // Cooperative wins if: (a) its recall delta beats transparent by ≥ 0.03
    // and (b) the model actually uses the injected instructions.
    if coop_delta > trans_delta + 0.03 && coop_engages {
        let reason = format!(
            "cooperative mode improved recall by {:+.3} vs baseline (transparent: {:+.3}); \
             useful probe rate {:.0}%, {:.1} annotations/turn",
            coop_delta,
            trans_delta,
            cooperative.useful_probes_pct * 100.0,
            cooperative.annotations_per_turn
        );
        return ("cooperative".to_string(), reason);
    }

    // Transparent wins if its recall delta is better than cooperative.
    if trans_delta >= coop_delta && trans_delta > 0.0 {
        let reason = if !coop_engages {
            format!(
                "model does not engage with probe/annotation instructions \
                 (useful probe rate {:.0}%, {:.1} annotations/turn); \
                 transparent mode gives equivalent recall ({:+.3} vs baseline)",
                cooperative.useful_probes_pct * 100.0,
                cooperative.annotations_per_turn,
                trans_delta,
            )
        } else {
            format!(
                "transparent mode recall ({:+.3} vs baseline) matches or beats cooperative \
                 ({:+.3} vs baseline) — cooperation overhead not justified",
                trans_delta, coop_delta,
            )
        };
        return ("transparent".to_string(), reason);
    }

    // Both modes show minimal delta over baseline. Prefer transparent (less overhead).
    let reason = format!(
        "neither mode improves substantially over baseline (transparent: {:+.3}, \
         cooperative: {:+.3}); defaulting to transparent to avoid instruction overhead",
        trans_delta, coop_delta,
    );
    ("transparent".to_string(), reason)
}

// ── Model slug ────────────────────────────────────────────────────────────────

fn model_slug(model: &str) -> String {
    model
        .replace('/', "_")
        .replace(':', "_")
        .replace('.', "_")
        .replace(' ', "_")
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let runtime = Arc::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build tokio runtime")?,
    );

    // Load workload items.
    let all_items = opencaw::build(&cli.repo_root, cli.qa_file.as_deref())
        .context("load opencaw workload")?;
    let items: Vec<WorkloadItem> = all_items
        .into_iter()
        .take(cli.limit)
        .collect();

    if items.is_empty() {
        anyhow::bail!("workload is empty — check --repo-root and --qa-file");
    }
    let model = cli.model.as_deref()
        .unwrap_or_else(|| cli.adapter.default_model(ModelRole::Answer))
        .to_string();

    eprintln!(
        "caw-bench-coop: {} items × 3 modes for model {}",
        items.len(),
        model
    );

    // Build corpus once — all opencaw items share the same repo corpus.
    let all_corpus: Vec<caw_bench::workload::CorpusDoc> = items[0].corpus.clone();
    let (prebuilt, _corpus_tmp) =
        build_in_memory_prebuilt(&all_corpus).context("build shared index")?;

    let mut baseline_outcomes = Vec::new();
    let mut transparent_outcomes = Vec::new();
    let mut cooperative_outcomes = Vec::new();

    let spec_base = AdapterSpec {
        kind: cli.adapter,
        model: &model,
        ollama_url: &cli.ollama_url,
        openai_url: &cli.openai_url,
        temperature: Some(cli.temperature),
        num_ctx: cli.num_ctx,
        num_predict: None,
        seed: None,
    };

    for (idx, item) in items.iter().enumerate() {
        eprintln!(
            "  [{}/{}] {}",
            idx + 1,
            items.len(),
            item.id
        );

        // Build three fresh adapters — each adapter owns its runtime handle.
        for (label, coop_mode, recall_on) in &[
            ("baseline", CooperationMode::Transparent, false),
            ("transparent", CooperationMode::Transparent, true),
            ("cooperative", CooperationMode::Cooperative, true),
        ] {
            let adapter = build(
                AdapterSpec {
                    kind: spec_base.kind,
                    model: spec_base.model,
                    ollama_url: spec_base.ollama_url,
                    openai_url: spec_base.openai_url,
                    temperature: spec_base.temperature,
                    num_ctx: spec_base.num_ctx,
                    num_predict: spec_base.num_predict,
                    seed: spec_base.seed,
                },
                &runtime,
            )
            .with_context(|| format!("build adapter for {}", model))?;

            let outcome = run_item_coop(item, *coop_mode, *recall_on, adapter, &prebuilt, &cli)
                .with_context(|| format!("item {} mode {}", item.id, label))?;

            eprintln!(
                "      {:12} recall={:.3}  probes/turn={:.2}  useful={:.0}%  ann/turn={:.2}{}",
                label,
                outcome.recall_at_k,
                outcome.probes_per_turn,
                outcome.useful_probes_pct * 100.0,
                outcome.annotations_per_turn,
                if outcome.degenerate { "  ⚠ degenerate" } else { "" },
            );

            match *label {
                "baseline" => baseline_outcomes.push(outcome),
                "transparent" => transparent_outcomes.push(outcome),
                "cooperative" => cooperative_outcomes.push(outcome),
                _ => unreachable!(),
            }
        }
    }

    let baseline = summarize(&baseline_outcomes);
    let transparent = summarize(&transparent_outcomes);
    let cooperative = summarize(&cooperative_outcomes);
    let (recommendation, recommendation_reason) = recommend(&baseline, &transparent, &cooperative);

    let report = CoopReport {
        model: model.clone(),
        item_count: items.len(),
        baseline,
        transparent,
        cooperative,
        recommendation: recommendation.clone(),
        recommendation_reason,
    };

    let slug = model_slug(&model);
    let out_dir = cli.out_dir.join(&slug);
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;
    let out_path = out_dir.join("coop-report.json");
    let json = serde_json::to_string_pretty(&report).context("serialize report")?;
    std::fs::write(&out_path, &json)
        .with_context(|| format!("write {}", out_path.display()))?;

    eprintln!("\nrecommendation: {recommendation}");
    eprintln!("report written to {}", out_path.display());
    println!("{json}");

    Ok(())
}
