use serde::Serialize;

use crate::runner::ItemResult;
use crate::workload::RecallMode;

#[derive(Debug, Serialize)]
pub struct ModeSummary {
    pub mode: String,
    pub item_count: usize,
    /// Mean answer_score across items (0.0-1.0). Primary metric of the
    /// thesis: does recall improve task-level correctness at matched budget.
    pub mean_answer_score: f32,
    pub mean_recall_at_k: f32,
    /// Capped at min(|expected|, k) / k. For single-needle workloads this
    /// is degenerate (max = 1/k); use precision_at_1 or mrr as the
    /// signal instead.
    pub mean_relevance_at_k: f32,
    pub mean_precision_at_1: f32,
    pub mean_mrr: f32,
    pub mean_content_tokens: f32,
    /// Mean tokens of stub summaries in the retrieval INDEX (not in context).
    /// Reported for corpus-pool size analysis.
    pub mean_index_pool_tokens: f32,
    pub mean_context_efficiency: f32,
    pub mean_false_recall_rate: f32,
    pub mean_latency_ms: f32,
    /// Phase breakdown of mean latency (diagnose-before-optimize). gen =
    /// answer-model completions, judge = judge model, other = embed + retrieve
    /// + orchestration (latency − gen − judge). `gen_calls` is the mean number
    /// of completions per item — the multi-pass cost in `recall_on`.
    pub mean_gen_ms: f32,
    pub mean_gen_calls: f32,
    pub mean_judge_ms: f32,
    pub mean_other_ms: f32,
}

/// Paired (recall_on − recall_off) statistics for one metric, computed over
/// items that have a result in *both* modes.
///
/// The `mean_delta` equals the difference of per-mode means — pairing does not
/// move the point estimate. Its value is `std_error`: because each item
/// contributes a single `on − off` difference, item-to-item difficulty (the
/// usual dominant variance) cancels, so the standard error of the paired delta
/// is far smaller than the unpaired `sqrt(var_on/n + var_off/n)`. A small
/// effect that is invisible in the diff-of-means noise floor becomes
/// resolvable here. `wins`/`ties`/`losses` give a distribution-free read of
/// how consistently recall helped, not just the average.
#[derive(Debug, Serialize)]
pub struct PairedDelta {
    pub metric: String,
    /// Number of items present in both modes (the paired sample size).
    pub n_paired: usize,
    pub mean_delta: f32,
    /// Sample standard deviation of the per-item deltas (n−1). 0 when n < 2.
    pub std_delta: f32,
    /// Standard error of `mean_delta` = `std_delta / sqrt(n)`. 0 when n < 2.
    pub std_error: f32,
    /// Items where on > off / on == off / on < off.
    pub wins: usize,
    pub ties: usize,
    pub losses: usize,
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub workload: String,
    pub answer_model: String,
    pub judge_model: String,
    pub summaries: Vec<ModeSummary>,
    /// Per-metric paired (on − off) deltas with standard errors. Empty when a
    /// single mode was run (e.g. `--only-mode`), since pairing needs both.
    pub paired_deltas: Vec<PairedDelta>,
    /// Per-item results so downstream analysis can slice by question class,
    /// latency, or any other axis without re-running.
    pub items: Vec<SerializableItem>,
}

#[derive(Debug, Serialize)]
pub struct SerializableItem {
    pub item_id: String,
    pub mode: String,
    pub question: String,
    pub answer: String,
    pub loaded_paths: Vec<String>,
    pub expected_paths: Vec<String>,
    pub recall_at_k: f32,
    pub relevance_at_k: f32,
    pub precision_at_1: f32,
    pub mrr: f32,
    pub content_tokens: usize,
    pub index_pool_tokens: usize,
    pub context_efficiency: f32,
    pub false_recall_rate: f32,
    pub answer_score: f32,
    pub judge_rationale: String,
    pub latency_ms: u64,
    pub gen_ms: u64,
    pub gen_calls: u64,
    pub judge_ms: u64,
}

impl From<&ItemResult> for SerializableItem {
    fn from(r: &ItemResult) -> Self {
        Self {
            item_id: r.item_id.clone(),
            mode: r.mode.as_str().to_string(),
            question: r.question.clone(),
            answer: r.answer.clone(),
            loaded_paths: r.loaded_paths.clone(),
            expected_paths: r.expected_paths.clone(),
            recall_at_k: r.recall_at_k,
            relevance_at_k: r.relevance_at_k,
            precision_at_1: r.precision_at_1,
            mrr: r.mrr,
            content_tokens: r.content_tokens,
            index_pool_tokens: r.index_pool_tokens,
            context_efficiency: r.context_efficiency,
            false_recall_rate: r.false_recall_rate,
            answer_score: r.answer_score,
            judge_rationale: r.judge_rationale.clone(),
            latency_ms: r.latency_ms,
            gen_ms: r.gen_ms,
            gen_calls: r.gen_calls,
            judge_ms: r.judge_ms,
        }
    }
}

pub fn build_report(
    workload: &str,
    answer_model: &str,
    judge_model: &str,
    results: &[ItemResult],
) -> BenchReport {
    let mut summaries = Vec::new();
    for mode in [RecallMode::On, RecallMode::Off] {
        let filtered: Vec<&ItemResult> = results.iter().filter(|r| r.mode == mode).collect();
        if filtered.is_empty() {
            continue;
        }
        summaries.push(mean_summary(mode, &filtered));
    }

    BenchReport {
        workload: workload.to_string(),
        answer_model: answer_model.to_string(),
        judge_model: judge_model.to_string(),
        summaries,
        paired_deltas: compute_paired_deltas(results),
        items: results.iter().map(SerializableItem::from).collect(),
    }
}

/// Pair each item's recall_on result with its recall_off result (matched on
/// `item_id`) and compute per-metric paired deltas. Items present in only one
/// mode are skipped. Pair order follows first appearance in `results` so the
/// output is deterministic.
fn compute_paired_deltas(results: &[ItemResult]) -> Vec<PairedDelta> {
    use std::collections::{HashMap, HashSet};

    let mut on: HashMap<&str, &ItemResult> = HashMap::new();
    let mut off: HashMap<&str, &ItemResult> = HashMap::new();
    for r in results {
        match r.mode {
            RecallMode::On => {
                on.insert(r.item_id.as_str(), r);
            }
            RecallMode::Off => {
                off.insert(r.item_id.as_str(), r);
            }
        }
    }

    let mut seen: HashSet<&str> = HashSet::new();
    let mut pairs: Vec<(&ItemResult, &ItemResult)> = Vec::new();
    for r in results {
        if !seen.insert(r.item_id.as_str()) {
            continue;
        }
        if let (Some(o), Some(f)) = (on.get(r.item_id.as_str()), off.get(r.item_id.as_str())) {
            pairs.push((o, f));
        }
    }
    if pairs.is_empty() {
        return Vec::new();
    }

    vec![
        paired_delta_for("answer_score", &pairs, |r| r.answer_score),
        paired_delta_for("recall_at_k", &pairs, |r| r.recall_at_k),
        paired_delta_for("precision_at_1", &pairs, |r| r.precision_at_1),
        paired_delta_for("mrr", &pairs, |r| r.mrr),
    ]
}

/// Compute the paired (on − off) statistics for one metric. `pairs` is
/// `(on, off)` per item; `extract` pulls the metric off a result.
fn paired_delta_for(
    metric: &str,
    pairs: &[(&ItemResult, &ItemResult)],
    extract: impl Fn(&ItemResult) -> f32,
) -> PairedDelta {
    let n = pairs.len();
    let deltas: Vec<f32> = pairs
        .iter()
        .map(|(on, off)| extract(on) - extract(off))
        .collect();
    let mean = deltas.iter().sum::<f32>() / n as f32;
    // Sample std (n−1): we're estimating the SE of the mean, not describing a
    // census. Undefined for n < 2, so report 0 there.
    let (std_delta, std_error) = if n < 2 {
        (0.0, 0.0)
    } else {
        let var = deltas.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / (n as f32 - 1.0);
        let std = var.sqrt();
        (std, std / (n as f32).sqrt())
    };
    let wins = deltas.iter().filter(|&&d| d > 0.0).count();
    let losses = deltas.iter().filter(|&&d| d < 0.0).count();
    let ties = n - wins - losses;

    PairedDelta {
        metric: metric.to_string(),
        n_paired: n,
        mean_delta: mean,
        std_delta,
        std_error,
        wins,
        ties,
        losses,
    }
}

fn mean_summary(mode: RecallMode, items: &[&ItemResult]) -> ModeSummary {
    let n = items.len() as f32;
    let sum_answer: f32 = items.iter().map(|r| r.answer_score).sum();
    let sum_recall: f32 = items.iter().map(|r| r.recall_at_k).sum();
    let sum_relevance: f32 = items.iter().map(|r| r.relevance_at_k).sum();
    let sum_p1: f32 = items.iter().map(|r| r.precision_at_1).sum();
    let sum_mrr: f32 = items.iter().map(|r| r.mrr).sum();
    let sum_content: usize = items.iter().map(|r| r.content_tokens).sum();
    let sum_pool: usize = items.iter().map(|r| r.index_pool_tokens).sum();
    let sum_eff: f32 = items.iter().map(|r| r.context_efficiency).sum();
    let sum_fr: f32 = items.iter().map(|r| r.false_recall_rate).sum();
    let sum_latency: u64 = items.iter().map(|r| r.latency_ms).sum();
    let sum_gen_ms: u64 = items.iter().map(|r| r.gen_ms).sum();
    let sum_gen_calls: u64 = items.iter().map(|r| r.gen_calls).sum();
    let sum_judge_ms: u64 = items.iter().map(|r| r.judge_ms).sum();
    let sum_other_ms: u64 = items
        .iter()
        .map(|r| {
            r.latency_ms
                .saturating_sub(r.gen_ms)
                .saturating_sub(r.judge_ms)
        })
        .sum();

    ModeSummary {
        mode: mode.as_str().to_string(),
        item_count: items.len(),
        mean_answer_score: sum_answer / n,
        mean_recall_at_k: sum_recall / n,
        mean_relevance_at_k: sum_relevance / n,
        mean_precision_at_1: sum_p1 / n,
        mean_mrr: sum_mrr / n,
        mean_content_tokens: sum_content as f32 / n,
        mean_index_pool_tokens: sum_pool as f32 / n,
        mean_context_efficiency: sum_eff / n,
        mean_false_recall_rate: sum_fr / n,
        mean_latency_ms: sum_latency as f32 / n,
        mean_gen_ms: sum_gen_ms as f32 / n,
        mean_gen_calls: sum_gen_calls as f32 / n,
        mean_judge_ms: sum_judge_ms as f32 / n,
        mean_other_ms: sum_other_ms as f32 / n,
    }
}

/// Format a human-readable delta between recall-on and recall-off modes.
/// This is the top-line output the thesis either lives or dies on.
pub fn format_summary(report: &BenchReport) -> String {
    let on = report.summaries.iter().find(|s| s.mode == "recall_on");
    let off = report.summaries.iter().find(|s| s.mode == "recall_off");

    let mut out = String::new();
    out.push_str(&format!("=== Workload: {} ===\n", report.workload));
    out.push_str(&format!("Answer model: {}\n", report.answer_model));
    out.push_str(&format!("Judge model:  {}\n\n", report.judge_model));

    for summary in &report.summaries {
        out.push_str(&format!("[{}] n={}\n", summary.mode, summary.item_count));
        out.push_str(&format!(
            "  answer_score:         {:.3}\n",
            summary.mean_answer_score
        ));
        out.push_str(&format!(
            "  recall@k:             {:.3}\n",
            summary.mean_recall_at_k
        ));
        out.push_str(&format!(
            "  precision@1:          {:.3}\n",
            summary.mean_precision_at_1
        ));
        out.push_str(&format!(
            "  mrr:                  {:.3}\n",
            summary.mean_mrr
        ));
        out.push_str(&format!(
            "  relevance@k:          {:.3}  (capped at min(|expected|,k)/k)\n",
            summary.mean_relevance_at_k
        ));
        out.push_str(&format!(
            "  context_efficiency:   {:.3}\n",
            summary.mean_context_efficiency
        ));
        out.push_str(&format!(
            "  false_recall_rate:    {:.3}\n",
            summary.mean_false_recall_rate
        ));
        out.push_str(&format!(
            "  avg content tokens:   {:.0}\n",
            summary.mean_content_tokens
        ));
        out.push_str(&format!(
            "  avg index pool toks:  {:.0}  (corpus stubs in retrieval pool, not in context)\n",
            summary.mean_index_pool_tokens
        ));
        out.push_str(&format!(
            "  avg latency ms:       {:.0}  (gen {:.0}ms x{:.1} + judge {:.0}ms + other {:.0}ms)\n\n",
            summary.mean_latency_ms,
            summary.mean_gen_ms,
            summary.mean_gen_calls,
            summary.mean_judge_ms,
            summary.mean_other_ms,
        ));
    }

    if let (Some(on), Some(off)) = (on, off) {
        out.push_str("=== Delta (on vs off) ===\n");
        out.push_str(&format!(
            "  Δ answer_score:       {:+.3}\n",
            on.mean_answer_score - off.mean_answer_score
        ));
        out.push_str(&format!(
            "  Δ recall@k:           {:+.3}\n",
            on.mean_recall_at_k - off.mean_recall_at_k
        ));
        out.push_str(&format!(
            "  Δ precision@1:        {:+.3}\n",
            on.mean_precision_at_1 - off.mean_precision_at_1
        ));
        out.push_str(&format!(
            "  Δ mrr:                {:+.3}\n",
            on.mean_mrr - off.mean_mrr
        ));
        out.push_str(&format!(
            "  Δ context_efficiency: {:+.3}\n",
            on.mean_context_efficiency - off.mean_context_efficiency
        ));
        out.push_str(&format!(
            "  Δ false_recall_rate:  {:+.3}\n",
            on.mean_false_recall_rate - off.mean_false_recall_rate
        ));
        out.push_str(&format!(
            "  Δ latency ms:         {:+.0}\n",
            on.mean_latency_ms - off.mean_latency_ms
        ));
    }

    if let Some(first) = report.paired_deltas.first() {
        out.push_str(&format!(
            "\n=== Paired delta (on − off, per item; n={}) ===\n",
            first.n_paired
        ));
        out.push_str(
            "  item difficulty cancels — mean ± standard error.\n  \
             W/T/L = items where on>off / on==off / on<off.\n",
        );
        for d in &report.paired_deltas {
            out.push_str(&format!(
                "  Δ {:<14} {:+.3} ± {:.3}  (σ={:.3}, {}W/{}T/{}L)\n",
                d.metric, d.mean_delta, d.std_error, d.std_delta, d.wins, d.ties, d.losses,
            ));
        }
    }

    out
}
