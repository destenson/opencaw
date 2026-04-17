use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use caw_bench::adapter_factory::{self, AdapterKind, AdapterSpec};
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
    /// Default chosen so the corpus is several times larger than the
    /// workspace budget — forcing real retrieval competition.
    #[arg(long, default_value = "200")]
    niah_filler: usize,

    /// Seed for NIAH's reproducible RNG.
    #[arg(long, default_value = "12345")]
    niah_seed: u64,

    /// Cap on how many items to actually run (trim the workload from the front).
    #[arg(long)]
    limit: Option<usize>,

    /// Adapter for the answering model.
    #[arg(long, value_enum, default_value_t = AdapterKind::Ollama)]
    answer_adapter: AdapterKind,

    /// Answering model name. Same value used for both recall-on and
    /// recall-off so the delta isolates recall's contribution. Format
    /// depends on `--answer-adapter`: ollama tag (e.g. "qwen3.5:9b"),
    /// vLLM HF id (e.g. "Qwen/Qwen2.5-7B-Instruct"), or claude alias
    /// ("sonnet", "opus", "haiku").
    #[arg(long, default_value = "qwen3.5:9b")]
    answer_model: String,

    /// Adapter for the judge model (JudgeAgainst scoring only).
    #[arg(long, value_enum, default_value_t = AdapterKind::ClaudeCode)]
    judge_adapter: AdapterKind,

    /// Judge model name. Default keeps the judge on a different model
    /// family than the answer to avoid same-model self-agreement bias.
    #[arg(long, default_value = "haiku")]
    judge_model: String,

    /// Ollama base URL (used by both answer and judge if either is `ollama`).
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// vLLM / OpenAI-compatible base URL. Reads OPENAI_COMPATIBLE_API_KEY
    /// from env if present.
    #[arg(long, default_value = "http://localhost:8000")]
    openai_url: String,

    /// Max workspace tokens — applied identically to both modes for a
    /// matched-budget comparison. Tight default forces eviction to engage
    /// when probes bring in additional fragments.
    #[arg(long, default_value = "2000")]
    max_workspace_tokens: usize,

    /// top-k for retrieval.
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Hysteresis load threshold (score above which a candidate is loaded
    /// into the workspace). The library default is 0.7, tuned for real
    /// document corpora; the bench defaults lower because short synthetic
    /// text and small symbols rarely clear 0.7 on BGE-small.
    #[arg(long, default_value = "0.3")]
    load_threshold: f32,

    /// Hysteresis unload threshold (score below which a loaded fragment
    /// becomes an eviction candidate). Must be less than `load_threshold`.
    #[arg(long, default_value = "0.2")]
    unload_threshold: f32,

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

    if cli.unload_threshold >= cli.load_threshold {
        anyhow::bail!(
            "unload_threshold ({}) must be strictly less than load_threshold ({}) for hysteresis to work",
            cli.unload_threshold,
            cli.load_threshold,
        );
    }

    let cfg = RunnerConfig {
        top_k: cli.top_k,
        max_workspace_tokens: cli.max_workspace_tokens,
        load_threshold: cli.load_threshold,
        unload_threshold: cli.unload_threshold,
        limit: cli.limit,
        ..Default::default()
    };

    // Single tokio runtime shared across all adapter constructions. Both
    // OllamaAdapter and OpenAiCompatibleAdapter need it; ClaudeCodeAdapter
    // ignores it.
    let runtime = caw_adapters::create_runtime().context("create tokio runtime")?;

    let judge_adapter = adapter_factory::build(
        AdapterSpec {
            kind: cli.judge_adapter,
            model: &cli.judge_model,
            ollama_url: &cli.ollama_url,
            openai_url: &cli.openai_url,
        },
        &runtime,
    )?;

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

            // Each item gets a fresh answer adapter — the orchestrator
            // takes ownership, and we want no state carried between runs.
            let answer_adapter = match adapter_factory::build(
                AdapterSpec {
                    kind: cli.answer_adapter,
                    model: &cli.answer_model,
                    ollama_url: &cli.ollama_url,
                    openai_url: &cli.openai_url,
                },
                &runtime,
            ) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("  error building answer adapter: {:#}", e);
                    continue;
                }
            };

            match run_item(item, *mode, &cfg, answer_adapter, judge_adapter.as_ref()) {
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
