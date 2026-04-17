use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use caw_bench::niah::{self, NiahConfig};
use caw_bench::opencaw;
use caw_bench::report::{build_report, format_summary};
use caw_bench::runner::{ItemResult, RunnerConfig, run_item};
use caw_bench::workload::{RecallMode, WorkloadItem};

#[derive(Parser, Debug)]
#[command(name = "caw-bench", about = "Benchmark harness for OpenCAW recall")]
struct Cli {
    /// Workload to run.
    #[arg(long, value_enum, default_value_t = Workload::Niah)]
    workload: Workload,

    /// Path to an opencaw checkout (for the `opencaw` workload).
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Number of NIAH items to generate (only used for `niah`).
    #[arg(long, default_value = "10")]
    niah_items: usize,

    /// Filler paragraphs per NIAH item (controls effective haystack size).
    #[arg(long, default_value = "80")]
    niah_filler: usize,

    /// Seed for NIAH's reproducible RNG.
    #[arg(long, default_value = "12345")]
    niah_seed: u64,

    /// Cap on how many items to actually run (trim the workload from the front).
    #[arg(long)]
    limit: Option<usize>,

    /// Answer model. Use the same model for both recall-on and recall-off runs.
    #[arg(long, default_value = "claude-sonnet-4-5-20250929")]
    answer_model: String,

    /// Judge model for open-ended (JudgeAgainst) scoring.
    #[arg(long, default_value = "claude-haiku-4-5-20251001")]
    judge_model: String,

    /// Max workspace tokens — applied identically to both modes for a
    /// matched-budget comparison.
    #[arg(long, default_value = "12000")]
    max_workspace_tokens: usize,

    /// top-k for retrieval.
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Where to write the JSON report (stdout if omitted).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Only run one mode (useful for debugging). Default: both.
    #[arg(long, value_enum)]
    only_mode: Option<ModeArg>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Workload {
    Niah,
    Opencaw,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArg {
    On,
    Off,
}

impl From<ModeArg> for RecallMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::On => Self::On,
            ModeArg::Off => Self::Off,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut items = build_workload(&cli)?;
    if let Some(limit) = cli.limit {
        items.truncate(limit);
    }
    if items.is_empty() {
        anyhow::bail!("workload produced zero items");
    }

    let cfg = RunnerConfig {
        top_k: cli.top_k,
        max_workspace_tokens: cli.max_workspace_tokens,
        answer_model: cli.answer_model.clone(),
        judge_model: cli.judge_model.clone(),
        limit: cli.limit,
        ..Default::default()
    };

    let runtime = caw_adapters::create_runtime().context("create tokio runtime")?;

    let modes: Vec<RecallMode> = match cli.only_mode {
        Some(m) => vec![m.into()],
        None => vec![RecallMode::On, RecallMode::Off],
    };

    let total = items.len() * modes.len();
    let mut done = 0usize;
    let mut results: Vec<ItemResult> = Vec::with_capacity(total);

    for item in &items {
        for mode in &modes {
            done += 1;
            eprintln!(
                "[{done}/{total}] {} [{}] — {}",
                item.id,
                mode.as_str(),
                truncate(&item.question, 80)
            );
            match run_item(runtime.clone(), item, *mode, &cfg) {
                Ok(result) => {
                    eprintln!(
                        "  score={:.2} recall@k={:.2} ctx_eff={:.2} loaded={}",
                        result.answer_score,
                        result.recall_at_k,
                        result.context_efficiency,
                        result.loaded_paths.len(),
                    );
                    results.push(result);
                }
                Err(e) => {
                    eprintln!("  error: {:#}", e);
                }
            }
        }
    }

    let workload_name = match cli.workload {
        Workload::Niah => "niah",
        Workload::Opencaw => "opencaw",
    };

    let report = build_report(workload_name, &cli.answer_model, &cli.judge_model, &results);
    let json = serde_json::to_string_pretty(&report).context("serialize report")?;

    if let Some(path) = &cli.out {
        std::fs::write(path, &json).with_context(|| format!("write report to {:?}", path))?;
        eprintln!("\nreport written to {}", path.display());
    } else {
        println!("{}", json);
    }

    eprintln!("\n{}", format_summary(&report));

    Ok(())
}

fn build_workload(cli: &Cli) -> Result<Vec<WorkloadItem>> {
    match cli.workload {
        Workload::Niah => {
            let config = NiahConfig {
                filler_paragraphs: cli.niah_filler,
                seed: cli.niah_seed,
                item_count: cli.niah_items,
            };
            Ok(niah::build(&config))
        }
        Workload::Opencaw => opencaw::build(&cli.repo_root).context("build opencaw workload"),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let trimmed: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{}...", trimmed)
    }
}
