//! `caw-bench-tune` — reads completed bench runs and emits an "ideal" config.
//!
//! Two sources are combined:
//!
//! - **Intent bench** (`--intent-dir`): walks every `*/intent-report.json`
//!   produced by `caw-bench-intent`, aggregates per-model scores across runs,
//!   and selects the best intent classifier.
//!
//! - **Sweep bench** (`--sweep-dir`): walks every `*/report.json` produced by
//!   `caw-bench` / `caw-bench-sweep`, groups by answer model, and selects the
//!   best answer model by absolute recall-on score.
//!
//! Outputs:
//! - TOML config (to `--out-toml`, or stdout) — drop-in values for the CLI
//! - Rust constants file (to `--out-rust`) — `pub const` block for compile-time
//!   baking into the binary
//!
//! ## Scoring (model selection)
//!
//! **Intent classifier**: combined_score = 0.6 × mean(exact_match) + 0.4 × min(exact_match),
//! multiplied by (1 - mean_parse_failure_rate). The min component penalises models
//! that are unreliable across runs — a model that scores 1.0 once and 0.0 the next
//! run is worse than one that scores 0.7 every time.
//!
//! **Answer model**: mean answer_score across all recall-on items across all sweep
//! cells in which that model appeared.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "caw-bench-tune",
    about = "Aggregate bench results and emit a recommended model config"
)]
struct Cli {
    /// Directory containing intent bench run subdirs (each with intent-report.json).
    #[arg(long, default_value = "bench-results/intent")]
    intent_dir: PathBuf,

    /// Directory containing sweep / caw-bench run subdirs (each with report.json).
    #[arg(long, default_value = "bench-results/default")]
    sweep_dir: PathBuf,

    /// Write TOML config to this file. Defaults to stdout.
    #[arg(long)]
    out_toml: Option<PathBuf>,

    /// Write Rust constants to this file. Optional.
    #[arg(long)]
    out_rust: Option<PathBuf>,

    /// Minimum number of bench runs a model must appear in to be eligible.
    /// Useful to skip one-off experiments and only trust models with repeated
    /// evidence. Default 1 (include everything).
    #[arg(long, default_value = "1")]
    min_runs: usize,
}

// ── Report deserialization ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct IntentReport {
    cases: usize,
    summaries: Vec<IntentSummary>,
}

#[derive(Deserialize)]
struct IntentSummary {
    model: String,
    exact_match_rate: f32,
    parse_failures: usize,
}

/// Matches `caw_bench::report::BenchReport` / `ModeSummary`.
#[derive(Deserialize)]
struct SweepReport {
    answer_model: String,
    summaries: Vec<SweepModeSummary>,
}

#[derive(Deserialize)]
struct SweepModeSummary {
    mode: String,
    mean_answer_score: f32,
    item_count: usize,
}

// ── Per-model aggregation ─────────────────────────────────────────────────────

struct IntentModelStats {
    scores: Vec<f32>,
    parse_failure_rates: Vec<f32>,
}

impl IntentModelStats {
    fn new() -> Self {
        Self {
            scores: Vec::new(),
            parse_failure_rates: Vec::new(),
        }
    }

    fn add(&mut self, score: f32, parse_failures: usize, cases: usize) {
        self.scores.push(score);
        let pfr = if cases > 0 {
            parse_failures as f32 / cases as f32
        } else {
            1.0
        };
        self.parse_failure_rates.push(pfr);
    }

    fn mean_score(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f32>() / self.scores.len() as f32
    }

    fn min_score(&self) -> f32 {
        self.scores.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    fn mean_parse_failure_rate(&self) -> f32 {
        if self.parse_failure_rates.is_empty() {
            return 0.0;
        }
        self.parse_failure_rates.iter().sum::<f32>() / self.parse_failure_rates.len() as f32
    }

    fn runs(&self) -> usize {
        self.scores.len()
    }

    /// Combined score weighting both mean performance and consistency (min).
    /// Multiplied by reliability factor (1 - parse_failure_rate) so a model
    /// that often fails to emit valid JSON ranks below one that always does.
    fn combined_score(&self) -> f32 {
        let mean = self.mean_score();
        let min = self.min_score();
        let pfr = self.mean_parse_failure_rate();
        (0.6 * mean + 0.4 * min) * (1.0 - pfr)
    }
}

struct SweepModelStats {
    // Weighted sum of mean_answer_score × item_count, and total item count,
    // so that cells with more items contribute proportionally.
    weighted_score_sum: f64,
    total_items: usize,
}

impl SweepModelStats {
    fn new() -> Self {
        Self {
            weighted_score_sum: 0.0,
            total_items: 0,
        }
    }

    fn add(&mut self, mean_answer_score: f32, item_count: usize) {
        self.weighted_score_sum += mean_answer_score as f64 * item_count as f64;
        self.total_items += item_count;
    }

    fn mean_score(&self) -> f32 {
        if self.total_items == 0 {
            return 0.0;
        }
        (self.weighted_score_sum / self.total_items as f64) as f32
    }
}

// ── Loading ───────────────────────────────────────────────────────────────────

fn load_intent_stats(intent_dir: &Path) -> Result<BTreeMap<String, IntentModelStats>> {
    let mut stats: BTreeMap<String, IntentModelStats> = BTreeMap::new();

    if !intent_dir.exists() {
        return Ok(stats);
    }

    let entries = std::fs::read_dir(intent_dir)
        .with_context(|| format!("reading intent dir {}", intent_dir.display()))?;

    for entry in entries.flatten() {
        let report_path = entry.path().join("intent-report.json");
        if !report_path.exists() {
            continue;
        }

        let raw = std::fs::read_to_string(&report_path)
            .with_context(|| format!("reading {}", report_path.display()))?;
        let report: IntentReport = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", report_path.display()))?;

        for summary in &report.summaries {
            stats
                .entry(summary.model.clone())
                .or_insert_with(IntentModelStats::new)
                .add(
                    summary.exact_match_rate,
                    summary.parse_failures,
                    report.cases,
                );
        }
    }

    Ok(stats)
}

fn load_sweep_stats(sweep_dir: &Path) -> Result<BTreeMap<String, SweepModelStats>> {
    let mut stats: BTreeMap<String, SweepModelStats> = BTreeMap::new();

    if !sweep_dir.exists() {
        return Ok(stats);
    }

    let entries = std::fs::read_dir(sweep_dir)
        .with_context(|| format!("reading sweep dir {}", sweep_dir.display()))?;

    for entry in entries.flatten() {
        let report_path = entry.path().join("report.json");
        if !report_path.exists() {
            continue;
        }

        let raw = std::fs::read_to_string(&report_path)
            .with_context(|| format!("reading {}", report_path.display()))?;
        let report: SweepReport = match serde_json::from_str(&raw) {
            Ok(r) => r,
            Err(_) => continue, // sweep dir may contain intent reports or other JSON
        };

        for summary in &report.summaries {
            if summary.mode != "recall_on" {
                continue; // only recall-on mode for answer model selection
            }
            stats
                .entry(report.answer_model.clone())
                .or_insert_with(SweepModelStats::new)
                .add(summary.mean_answer_score, summary.item_count);
        }
    }

    Ok(stats)
}

// ── Selection ─────────────────────────────────────────────────────────────────

struct IntentWinner {
    model: String,
    combined_score: f32,
    mean_score: f32,
    min_score: f32,
    runs: usize,
    mean_parse_failure_rate: f32,
}

struct SweepWinner {
    model: String,
    mean_answer_score: f32,
}

fn pick_intent_winner(
    stats: &BTreeMap<String, IntentModelStats>,
    min_runs: usize,
) -> Option<IntentWinner> {
    stats
        .iter()
        .filter(|(_, s)| s.runs() >= min_runs)
        .max_by(|(_, a), (_, b)| {
            a.combined_score()
                .partial_cmp(&b.combined_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(model, s)| IntentWinner {
            model: model.clone(),
            combined_score: s.combined_score(),
            mean_score: s.mean_score(),
            min_score: s.min_score(),
            runs: s.runs(),
            mean_parse_failure_rate: s.mean_parse_failure_rate(),
        })
}

fn pick_sweep_winner(stats: &BTreeMap<String, SweepModelStats>) -> Option<SweepWinner> {
    stats
        .iter()
        .max_by(|(_, a), (_, b)| {
            a.mean_score()
                .partial_cmp(&b.mean_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(model, s)| SweepWinner {
            model: model.clone(),
            mean_answer_score: s.mean_score(),
        })
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn render_toml(
    intent_winner: Option<&IntentWinner>,
    intent_stats: &BTreeMap<String, IntentModelStats>,
    sweep_winner: Option<&SweepWinner>,
    sweep_stats: &BTreeMap<String, SweepModelStats>,
    min_runs: usize,
) -> String {
    let today = chrono_today();
    let mut out = String::new();

    out.push_str("# Generated by caw-bench-tune — do not edit by hand.\n");
    out.push_str(&format!("# Generated: {today}\n"));
    out.push_str("# These values can be passed directly as CLI flags or used as defaults.\n\n");

    out.push_str("[models]\n\n");

    // Classifier
    match intent_winner {
        Some(w) => {
            out.push_str(&format!(
                "# Intent classifier — best local model by combined score\n\
                 # (0.6 × mean_exact_match + 0.4 × min_exact_match) × (1 - parse_failure_rate)\n\
                 # score={:.3}  mean={:.3}  min={:.3}  parse_fail={:.1}%  runs={}\n",
                w.combined_score,
                w.mean_score,
                w.min_score,
                w.mean_parse_failure_rate * 100.0,
                w.runs
            ));
            out.push_str(&format!("intent_model = {:?}\n\n", w.model));
        }
        None => {
            out.push_str("# intent_model: no intent bench data found\n");
            out.push_str("# intent_model = \"llama3.2:3b\"  # CLI default\n\n");
        }
    }

    // Answer model
    match sweep_winner {
        Some(w) => {
            out.push_str(&format!(
                "# Answer model — best by absolute recall-on mean_answer_score\n\
                 # score={:.3}\n",
                w.mean_answer_score
            ));
            out.push_str(&format!("# answer_model = {:?}\n\n", w.model));
            out.push_str(
                "# (Set via --model / --adapter at the CLI; no single config key yet)\n\n",
            );
        }
        None => {
            out.push_str("# answer_model: no sweep bench data found\n");
            out.push_str("# Run caw-bench-sweep to generate sweep data.\n\n");
        }
    }

    // Full leaderboard as comments
    if !intent_stats.is_empty() {
        out.push_str(
            "# ── Intent classifier leaderboard ──────────────────────────────────────────\n",
        );
        out.push_str("# Rank  Score   Mean    Min     PFail%  Runs  Model\n");
        let mut ranked: Vec<_> = intent_stats
            .iter()
            .filter(|(_, s)| s.runs() >= min_runs)
            .collect();
        ranked.sort_by(|(_, a), (_, b)| {
            b.combined_score()
                .partial_cmp(&a.combined_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, (model, s)) in ranked.iter().enumerate() {
            out.push_str(&format!(
                "# {:>4}   {:.3}   {:.3}   {:.3}   {:>5.1}%  {:>4}  {}\n",
                i + 1,
                s.combined_score(),
                s.mean_score(),
                s.min_score(),
                s.mean_parse_failure_rate() * 100.0,
                s.runs(),
                model
            ));
        }
        out.push('\n');
    }

    if !sweep_stats.is_empty() {
        out.push_str(
            "# ── Answer model leaderboard (recall-on, absolute score) ───────────────────\n",
        );
        out.push_str("# Rank  Score   Model\n");
        let mut ranked: Vec<_> = sweep_stats.iter().collect();
        ranked.sort_by(|(_, a), (_, b)| {
            b.mean_score()
                .partial_cmp(&a.mean_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, (model, s)) in ranked.iter().enumerate() {
            out.push_str(&format!(
                "# {:>4}   {:.3}   {}\n",
                i + 1,
                s.mean_score(),
                model
            ));
        }
        out.push('\n');
    }

    out
}

fn render_rust(intent_winner: Option<&IntentWinner>, sweep_winner: Option<&SweepWinner>) -> String {
    let today = chrono_today();
    let mut out = String::new();

    out.push_str("// Generated by caw-bench-tune — do not edit by hand.\n");
    out.push_str(&format!("// Generated: {today}\n\n"));

    match intent_winner {
        Some(w) => {
            out.push_str(&format!(
                "// Intent classifier: score={:.3}  mean={:.3}  min={:.3}  runs={}\n",
                w.combined_score, w.mean_score, w.min_score, w.runs
            ));
            out.push_str(&format!(
                "pub const DEFAULT_INTENT_MODEL: &str = {:?};\n\n",
                w.model
            ));
        }
        None => {
            out.push_str("// No intent bench data — using CLI default.\n");
            out.push_str("pub const DEFAULT_INTENT_MODEL: &str = \"llama3.2:3b\";\n\n");
        }
    }

    match sweep_winner {
        Some(w) => {
            out.push_str(&format!(
                "// Answer model: mean_answer_score={:.3} (recall-on)\n",
                w.mean_answer_score
            ));
            out.push_str(&format!(
                "pub const DEFAULT_ANSWER_MODEL: &str = {:?};\n",
                w.model
            ));
        }
        None => {
            out.push_str("// No sweep bench data — no answer model recommendation.\n");
            out.push_str("// pub const DEFAULT_ANSWER_MODEL: &str = \"???\";\n");
        }
    }

    out
}

fn chrono_today() -> String {
    // Use env-based date if available, otherwise fall back to a placeholder.
    // Avoids pulling in chrono just for formatting.
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let intent_stats = load_intent_stats(&cli.intent_dir)
        .with_context(|| format!("loading intent bench from {}", cli.intent_dir.display()))?;
    let sweep_stats = load_sweep_stats(&cli.sweep_dir)
        .with_context(|| format!("loading sweep bench from {}", cli.sweep_dir.display()))?;

    let intent_winner = pick_intent_winner(&intent_stats, cli.min_runs);
    let sweep_winner = pick_sweep_winner(&sweep_stats);

    let toml_output = render_toml(
        intent_winner.as_ref(),
        &intent_stats,
        sweep_winner.as_ref(),
        &sweep_stats,
        cli.min_runs,
    );

    match &cli.out_toml {
        Some(path) => {
            std::fs::write(path, &toml_output)
                .with_context(|| format!("writing TOML to {}", path.display()))?;
            eprintln!("Wrote TOML config to {}", path.display());
        }
        None => print!("{toml_output}"),
    }

    if let Some(path) = &cli.out_rust {
        let rust_output = render_rust(intent_winner.as_ref(), sweep_winner.as_ref());
        std::fs::write(path, &rust_output)
            .with_context(|| format!("writing Rust constants to {}", path.display()))?;
        eprintln!("Wrote Rust constants to {}", path.display());
    }

    Ok(())
}
