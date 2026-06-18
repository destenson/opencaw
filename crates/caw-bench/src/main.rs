use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use caw_bench::adapter_factory::{self, AdapterKind, AdapterSpec, ModelRole};
use caw_bench::codeagent;
use caw_bench::niah::{self, NiahConfig};
use caw_bench::opencaw;
use caw_bench::report::{build_report, format_summary};
use caw_bench::runner::{
    ItemResult, PrebuiltIndex, RunnerConfig, build_in_memory_prebuilt, run_item,
};
use caw_bench::shared::{SharedEmbedder, SharedIndex, SharedStore};
use caw_bench::sysdoc;
use caw_bench::workload::{RecallMode, WorkloadItem};
use caw_core::{EmbeddingProvider, StubStore, VectorIndex};
use caw_index::{CandleEmbeddingProvider, HnswVectorIndex, SqliteStubStore};
use std::collections::HashMap;

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

    /// Answering model name. Format depends on `--answer-adapter`.
    /// Defaults to the adapter's built-in default when not specified.
    #[arg(long)]
    answer_model: Option<String>,

    /// Adapter for the judge model (JudgeAgainst scoring only).
    #[arg(long, value_enum, default_value_t = AdapterKind::ClaudeCode)]
    judge_adapter: AdapterKind,

    /// Judge model name. Default keeps the judge on a different model
    /// family than the answer to avoid same-model self-agreement bias.
    /// Defaults to the adapter's built-in default when not specified.
    #[arg(long)]
    judge_model: Option<String>,

    /// Ollama base URL (used by both answer and judge if either is `ollama`).
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// vLLM / OpenAI-compatible base URL. Reads OPENAI_COMPATIBLE_API_KEY
    /// from env if present.
    #[arg(long, default_value = "http://localhost:8000")]
    openai_url: String,

    /// Sampling temperature for the answer model. Default 0.0 means
    /// deterministic greedy decoding — required to isolate framework
    /// signal from sampling noise when comparing recall-on vs recall-off.
    /// Pass a higher value (e.g. 0.7) to study real-world stochastic
    /// behavior; if you do, consider running multiple seeds per item.
    /// Ignored for `claude-code` (CLI doesn't expose temperature in --print).
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Cap the answer model's per-completion generation budget (`num_predict`,
    /// Ollama only). Omitted = adapter default (4096). For a reasoning model
    /// this budget covers the thinking trace plus the answer, so a low cap can
    /// truncate the answer — use only to measure the gen-time/quality trade.
    #[arg(long)]
    num_predict: Option<i32>,

    /// Sampling temperature for the judge model. Defaults to 0.0 so the
    /// judge gives reproducible scores for the same answer/reference pair.
    #[arg(long, default_value = "0.0")]
    judge_temperature: f32,

    /// Sampling seed forwarded to the answer and judge adapters (Ollama, Groq).
    /// With temperature 0 this makes single-request decoding reproducible.
    /// Note: at `--concurrency > 1` the answer model can still vary because
    /// batched inference is not bit-reproducible; use `--concurrency 1` for a
    /// fully reproducible run.
    #[arg(long, default_value = "42")]
    seed: i64,

    /// Max workspace tokens — applied identically to both modes for a
    /// matched-budget comparison. Tight default forces eviction to engage
    /// when probes bring in additional fragments.
    #[arg(long, default_value = "2000")]
    max_workspace_tokens: usize,

    /// Candidate pool size for ANN search; the load threshold controls actual admissions.
    #[arg(long, default_value = "20")]
    max_candidates: usize,

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

    /// Path to a Q&A JSON file. Required for `sysdoc`. For `opencaw`, when
    /// provided the file is loaded at runtime; otherwise the binary's
    /// embedded copy is used.
    #[arg(long)]
    qa_file: Option<PathBuf>,

    /// Pre-built retrieval index (sqlite produced by
    /// `caw-bench-build-index`). Required for the `sysdoc` workload; the
    /// full corpus is ingested once ahead of time and shared across items
    /// so the bench doesn't re-embed on every run.
    #[arg(long)]
    index: Option<PathBuf>,

    /// Where to write the JSON report (stdout if omitted).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Where to write a per-item JSONL trace. One line per (item, mode)
    /// with the system prompt, question, reference answer, loaded
    /// fragments (with content previews and source locators), the model's
    /// answer, the judge's rationale, and the full metrics. Built for
    /// `jq`-filtering failure cases (e.g. `jq 'select(.answer_score < 1)'`)
    /// so you can see exactly what the model had and what it said.
    #[arg(long)]
    trace_out: Option<PathBuf>,

    /// Re-judge a persisted trace instead of generating. Reads a JSONL
    /// written by `--trace-out`, re-scores every judge-scored item against
    /// the current `--judge-adapter`/`--judge-model`/`--judge-temperature`,
    /// and emits a report — without re-running the answer model. Lets the
    /// same answers be scored by different or repeated judges to measure the
    /// judge's own contribution to score variance, at zero generation cost.
    /// When set, all generation flags (workload, index, answer model) are
    /// ignored.
    #[arg(long)]
    judge_trace: Option<PathBuf>,

    /// Only run one mode (useful for debugging). Default: both.
    #[arg(long, value_enum)]
    only_mode: Option<ModeArg>,

    /// Number of (item, mode) tasks to run concurrently. Default saturates
    /// the gen/judge/retrieval overlap within reason; `1` restores the serial
    /// path, which is the only path with valid per-phase (gen/judge/other)
    /// timing. Note Ollama serializes same-model generation, so the *gen*
    /// speedup is bounded (~18% on the current single-server stack) until the
    /// gen serving stack batches; the groq judge and retrieval overlap freely.
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Workload {
    Niah,
    Opencaw,
    /// Coding-agent mid-task info-needs (signatures, struct fields, trait
    /// bounds, call sites) over the repo corpus. Same corpus as `opencaw`,
    /// different question framing and per-item needle/judge scoring.
    CodeAgent,
    Sysdoc,
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
    caw_bench::init_tracing();
    let cli = Cli::parse();

    // Re-judge mode short-circuits generation entirely: load persisted
    // answers and score them. Nothing about the workload, index, or answer
    // model is consulted.
    if let Some(trace_path) = cli.judge_trace.clone() {
        return rejudge_trace(&cli, &trace_path);
    }

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

    // Keep the shared-corpus tempdir alive for the whole bench run. When
    // the opencaw workload runs without an external `--index`, we ingest
    // items[0].corpus once here instead of re-embedding per item. The
    // tempdir backs SqliteStubStore::get_content; dropping it would
    // invalidate recall mid-run.
    let mut _shared_corpus_tmp: Option<tempfile::TempDir> = None;
    let prebuilt = match (cli.workload, cli.index.as_deref()) {
        (Workload::Sysdoc, None) => anyhow::bail!(
            "--index is required for the sysdoc workload; build one with caw-bench-build-index"
        ),
        (_, Some(path)) => Some(load_prebuilt_index(path, cli.repo_root.clone())?),
        (Workload::Opencaw | Workload::CodeAgent, None) => {
            let (idx, tmp) = build_in_memory_prebuilt(&items[0].corpus)
                .context("build shared in-memory index for repo-corpus workload")?;
            _shared_corpus_tmp = Some(tmp);
            Some(idx)
        }
        (_, None) => None,
    };

    let cfg = RunnerConfig {
        max_candidates: cli.max_candidates,
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

    let answer_model = cli.answer_model.as_deref()
        .unwrap_or_else(|| cli.answer_adapter.default_model(ModelRole::Answer))
        .to_string();
    let judge_model = cli.judge_model.as_deref()
        .unwrap_or_else(|| cli.judge_adapter.default_model(ModelRole::Judge))
        .to_string();
    // The judge runs in a dedicated phase after all generation completes
    // (see `judge_all`), so no judge adapter is built here — each judged item
    // gets its own adapter in that phase so the remote judge can fan out.

    let modes: Vec<RecallMode> = match cli.only_mode {
        Some(m) => vec![m.into()],
        None => vec![RecallMode::On, RecallMode::Off],
    };

    let total = items.len() * modes.len();
    let mut done = 0usize;
    let mut results: Vec<ItemResult> = Vec::with_capacity(total);

    // Optional per-item JSONL trace. Opened once; lines append as each item
    // completes so a kill mid-run still leaves a usable partial trace.
    let mut trace_writer: Option<std::io::BufWriter<std::fs::File>> = match &cli.trace_out {
        Some(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).ok();
            }
            match std::fs::File::create(path) {
                Ok(f) => {
                    eprintln!("trace output: {}", path.display());
                    Some(std::io::BufWriter::new(f))
                }
                Err(e) => {
                    eprintln!(
                        "  warning: could not open trace file {}: {}",
                        path.display(),
                        e
                    );
                    None
                }
            }
        }
        None => None,
    };

    if cli.concurrency > 1 {
        run_concurrent(
            &items,
            &modes,
            &cfg,
            &cli,
            &answer_model,
            &runtime,
            prebuilt.as_ref(),
            total,
            &mut results,
            trace_writer.as_mut(),
        )?;
    } else {
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
                let answer_inner = match adapter_factory::build(
                    AdapterSpec {
                        kind: cli.answer_adapter,
                        model: &answer_model,
                        ollama_url: &cli.ollama_url,
                        openai_url: &cli.openai_url,
                        temperature: Some(cli.temperature),
                        num_ctx: None,
                        num_predict: cli.num_predict,
                        seed: Some(cli.seed),
                    },
                    &runtime,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("  error building answer adapter: {:#}", e);
                        continue;
                    }
                };
                // Fresh answer counter per item; judge counter is shared, so diff
                // its snapshot across the call to isolate this item's judge time.
                let answer_counters = caw_bench::timing::PhaseCounters::new();
                let answer_adapter: Box<dyn caw_core::ModelAdapter> = Box::new(
                    caw_bench::timing::TimingAdapter::new(answer_inner, answer_counters.clone()),
                );

                match run_item(item, *mode, &cfg, answer_adapter, prebuilt.as_ref()) {
                    Ok(mut result) => {
                        let (gen_calls, gen_ms) = answer_counters.snapshot();
                        result.gen_ms = gen_ms;
                        result.gen_calls = gen_calls;
                        // judge_ms is filled later, in the post-gen judge phase.
                        let other_ms =
                            result.latency_ms.saturating_sub(result.gen_ms);
                        eprintln!(
                            "  recall@k={:.2} ctx_eff={:.2} loaded={} | \
                            {}ms = gen {}ms(x{}) + other {}ms (judge: pending)",
                            result.recall_at_k,
                            result.context_efficiency,
                            result.loaded_paths.len(),
                            result.latency_ms,
                            result.gen_ms,
                            result.gen_calls,
                            other_ms,
                        );
                        if let Some(writer) = trace_writer.as_mut() {
                            use std::io::Write;
                            let entry = serde_json::json!({
                                "system_prompt": cfg.system_prompt,
                                "result": &result,
                            });
                            if let Ok(line) = serde_json::to_string(&entry) {
                                let _ = writeln!(writer, "{}", line);
                                let _ = writer.flush();
                            }
                        }
                        results.push(result);
                    }
                    Err(e) => {
                        eprintln!("  error: {:#}", e);
                    }
                }
            }
        }
    } // end serial path (concurrency == 1)

    let workload_name = match cli.workload {
        Workload::Niah => "niah",
        Workload::Opencaw => "opencaw",
        Workload::CodeAgent => "codeagent",
        Workload::Sysdoc => "sysdoc",
    };

    // Post-generation judge phase: score every pending item now that all
    // answers exist (JudgeAgainst, plus NeedleWithJudgeConfirm items whose token
    // was found). Generation never blocked on the judge; here the judge calls
    // fan out in parallel (the remote groq judge parallelizes freely).
    judge_all(&mut results, &cli, &judge_model, &runtime)?;

    let report = build_report(workload_name, &answer_model, &judge_model, &results);
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

/// Run the (item, mode) grid concurrently with a capped worker pool.
///
/// Each task builds its own answer and judge adapters inside the worker so no
/// `dyn ModelAdapter` trait object crosses a thread boundary; the read-only
/// `PrebuiltIndex` (Arc<Mutex<_>> embedder/store/index) is shared. Per-phase
/// (gen/judge/other) timing is deliberately NOT collected here — under
/// concurrency the shared judge counter and wall-time contention make it
/// invalid (see docs/DECISIONS.md); aggregate wall-clock and answer scores
/// stay valid. Results are reordered to the serial task order before the
/// report is built so output is deterministic regardless of completion order.
#[allow(clippy::too_many_arguments)]
fn run_concurrent(
    items: &[WorkloadItem],
    modes: &[RecallMode],
    cfg: &RunnerConfig,
    cli: &Cli,
    answer_model: &str,
    runtime: &std::sync::Arc<tokio::runtime::Runtime>,
    prebuilt: Option<&PrebuiltIndex>,
    total: usize,
    results: &mut Vec<ItemResult>,
    mut trace_writer: Option<&mut std::io::BufWriter<std::fs::File>>,
) -> Result<()> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let tasks: Vec<(usize, &WorkloadItem, RecallMode)> = items
        .iter()
        .flat_map(|item| modes.iter().map(move |m| (item, *m)))
        .enumerate()
        .map(|(i, (item, m))| (i, item, m))
        .collect();

    let done = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cli.concurrency)
        .build()
        .context("build bench thread pool")?;

    let mut collected: Vec<(usize, ItemResult)> = pool.install(|| {
        tasks
            .par_iter()
            .filter_map(|&(idx, item, mode)| {
                let answer = match adapter_factory::build(
                    AdapterSpec {
                        kind: cli.answer_adapter,
                        model: answer_model,
                        ollama_url: &cli.ollama_url,
                        openai_url: &cli.openai_url,
                        temperature: Some(cli.temperature),
                        num_ctx: None,
                        num_predict: cli.num_predict,
                        seed: Some(cli.seed),
                    },
                    runtime,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("  error building answer adapter: {:#}", e);
                        return None;
                    }
                };
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[{n}/{total}] {} [{}] — {}",
                    item.id,
                    mode.as_str(),
                    truncate(&item.question, 80)
                );

                match run_item(item, mode, cfg, answer, prebuilt) {
                    Ok(result) => {
                        eprintln!(
                            "  recall@k={:.2} ctx_eff={:.2} loaded={} | {}ms \
                             (concurrent: per-phase timing omitted; judge: pending)",
                            result.recall_at_k,
                            result.context_efficiency,
                            result.loaded_paths.len(),
                            result.latency_ms,
                        );
                        Some((idx, result))
                    }
                    Err(e) => {
                        eprintln!("  error: {:#}", e);
                        None
                    }
                }
            })
            .collect()
    });

    collected.sort_by_key(|(idx, _)| *idx);
    for (_, result) in &collected {
        if let Some(writer) = trace_writer.as_deref_mut() {
            use std::io::Write;
            let entry = serde_json::json!({
                "system_prompt": cfg.system_prompt,
                "result": result,
            });
            if let Ok(line) = serde_json::to_string(&entry) {
                let _ = writeln!(writer, "{}", line);
                let _ = writer.flush();
            }
        }
    }
    results.extend(collected.into_iter().map(|(_, r)| r));
    Ok(())
}

/// Score every item that generation left `judge_pending`, in one parallel
/// phase after all answers exist. Each item builds its own judge
/// adapter wrapped in a per-item `TimingAdapter`, so the recorded `judge_ms`
/// is that call's own latency and stays valid even though calls overlap.
///
/// A judge failure is logged and the item left `judge_pending` rather than
/// aborting the run — one malformed judge response can't discard a whole
/// generation pass (CLAUDE.md: log and continue, don't hide from the dev).
///
/// Fan-out is independent of `--concurrency` (which gates *generation*): the
/// judge is a remote call that parallelizes freely, so even a serial
/// generation run — concurrency 1, used to keep a local answer model from
/// batching — still judges in parallel.
fn judge_all(
    results: &mut [ItemResult],
    cli: &Cli,
    judge_model: &str,
    runtime: &std::sync::Arc<tokio::runtime::Runtime>,
) -> Result<()> {
    use rayon::prelude::*;

    let pending = results.iter().filter(|r| r.judge_pending).count();
    if pending == 0 {
        return Ok(());
    }

    // Remote-judge fan-out floor so a serial-generation run still judges in
    // parallel; honor a larger --concurrency if the user set one. 4 is the
    // default generation concurrency, a known-safe groq fan-out width.
    const MIN_JUDGE_WORKERS: usize = 4;
    let judge_workers = cli.concurrency.max(MIN_JUDGE_WORKERS);
    eprintln!(
        "\njudge phase: scoring {} pending item(s) with {} ({}-way parallel)",
        pending, judge_model, judge_workers
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(judge_workers)
        .build()
        .context("build judge thread pool")?;

    pool.install(|| {
        results.par_iter_mut().for_each(|result| {
            if !result.judge_pending {
                return;
            }
            let judge_inner = match adapter_factory::build(
                AdapterSpec {
                    kind: cli.judge_adapter,
                    model: judge_model,
                    ollama_url: &cli.ollama_url,
                    openai_url: &cli.openai_url,
                    temperature: Some(cli.judge_temperature),
                    num_ctx: None,
                    num_predict: None,
                    seed: Some(cli.seed),
                },
                runtime,
            ) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!(
                        "  judge adapter build failed for {} [{}]: {:#}",
                        result.item_id,
                        result.mode.as_str(),
                        e
                    );
                    return;
                }
            };
            let counters = caw_bench::timing::PhaseCounters::new();
            let judge = caw_bench::timing::TimingAdapter::new(judge_inner, counters.clone());
            match caw_bench::judge::judge_answer(&judge, &result.answer, &result.reference_answer) {
                Ok(verdict) => {
                    let (_, judge_ms) = counters.snapshot();
                    result.answer_score = verdict.score;
                    result.judge_rationale = verdict.rationale;
                    result.judge_ms = judge_ms;
                    result.judge_pending = false;
                }
                Err(e) => {
                    eprintln!(
                        "  judge failed for {} [{}] (left unscored): {:#}",
                        result.item_id,
                        result.mode.as_str(),
                        e
                    );
                }
            }
        });
    });

    let still_pending = results.iter().filter(|r| r.judge_pending).count();
    if still_pending > 0 {
        eprintln!(
            "  warning: {} item(s) remain unscored after the judge phase",
            still_pending
        );
    }
    Ok(())
}

/// Load a `--trace-out` JSONL and re-score every judge-scored item against the
/// current judge config, without regenerating answers (the `--judge-trace`
/// path). Reuses `judge_all` and `build_report`, so a re-judged report has the
/// same shape as a generated one — including paired deltas — but the answers
/// come from the file instead of the model.
fn rejudge_trace(cli: &Cli, trace_path: &std::path::Path) -> Result<()> {
    use std::io::BufRead;

    // A trace line is `{"system_prompt": ..., "result": <ItemResult>}`; serde
    // ignores the fields we don't name here.
    #[derive(serde::Deserialize)]
    struct TraceLine {
        result: ItemResult,
    }

    let file = std::fs::File::open(trace_path)
        .with_context(|| format!("open trace {}", trace_path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut results: Vec<ItemResult> = Vec::new();
    let mut parse_failures = 0usize;
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read trace line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<TraceLine>(&line) {
            Ok(tl) => results.push(tl.result),
            Err(e) => {
                parse_failures += 1;
                eprintln!("  warning: trace line {} did not parse: {}", i + 1, e);
            }
        }
    }
    if results.is_empty() {
        anyhow::bail!(
            "no usable results in {} ({} line(s) failed to parse)",
            trace_path.display(),
            parse_failures
        );
    }

    // Force re-scoring of every judge-scored item. A persisted result records
    // its scoring kind only indirectly: a non-empty `reference_answer` means the
    // item was scored by the model judge (see finalize_result — empty for
    // ContainsNeedle, and for a NeedleWithJudgeConfirm item whose token was
    // absent, which holds a deterministic 0.0 the judge must not override), so
    // re-judging it is meaningful. Those local-scored items keep their inline
    // score. This reads a structural property of the record, not a guess about
    // intent.
    let mut to_judge = 0usize;
    for r in &mut results {
        if !r.reference_answer.is_empty() {
            r.judge_pending = true;
            to_judge += 1;
        }
    }
    eprintln!(
        "re-judging {} of {} loaded item(s) from {}",
        to_judge,
        results.len(),
        trace_path.display()
    );

    let runtime = caw_adapters::create_runtime().context("create tokio runtime")?;
    let judge_model = cli
        .judge_model
        .as_deref()
        .unwrap_or_else(|| cli.judge_adapter.default_model(ModelRole::Judge))
        .to_string();

    judge_all(&mut results, cli, &judge_model, &runtime)?;

    let report = build_report("rejudge", "(persisted answers)", &judge_model, &results);
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
        Workload::Opencaw => {
            opencaw::build(&cli.repo_root, cli.qa_file.as_deref()).context("build opencaw workload")
        }
        Workload::CodeAgent => codeagent::build(&cli.repo_root, cli.qa_file.as_deref())
            .context("build codeagent workload"),
        Workload::Sysdoc => {
            let qa_file = cli.qa_file.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--qa-file is required for the sysdoc workload (e.g. opencaw-corpora/sysdoc_qa.json)"
                )
            })?;
            sysdoc::build(qa_file).context("build sysdoc workload")
        }
    }
}

/// Open a prebuilt retrieval index and wrap it for sharing across items.
/// Called once per bench run when `--index` is supplied.
fn load_prebuilt_index(path: &std::path::Path, corpus_root: PathBuf) -> Result<PrebuiltIndex> {
    let started = std::time::Instant::now();
    let path_str = path.to_string_lossy().into_owned();

    let embedder = CandleEmbeddingProvider::bge_small().context("init bge-small (candle)")?;
    let dim = embedder.dimension();

    // corpus_root is required for `get_content` to resolve the relative
    // paths stored on each stub and slice the body back out of the source
    // file. Without this the retrieval path errors the moment any loaded
    // fragment needs its body expanded.
    //
    // Read-only: the bench consumes a prebuilt index and has no reingest
    // worker, so get_content must not mark rows stale on a missing/
    // misconfigured path — that would persist into the shared index and
    // degrade every later run (mirrors caw-server's prebuilt-index opener).
    let store = SqliteStubStore::new(&path_str, dim)
        .with_context(|| format!("open prebuilt index at {}", path.display()))?
        .with_corpus_root(corpus_root)
        .with_read_only(true);

    let all = store.all_embeddings().context("read all embeddings")?;
    if all.is_empty() {
        anyhow::bail!(
            "prebuilt index at {} is empty — rebuild with caw-bench-build-index",
            path.display()
        );
    }

    // Stub summaries are needed regardless of how the HNSW graph is sourced
    // (the false-recall heuristic compares recalled content against them).
    let mut stub_summaries: HashMap<caw_core::StubId, String> = HashMap::with_capacity(all.len());
    for (stub_id, _embedding) in &all {
        if let Ok(stub) = store.get_stub(stub_id) {
            stub_summaries.insert(stub_id.clone(), stub.summary);
        }
    }

    // The HNSW graph build over the full stub set is the dominant startup
    // cost (several seconds of GPU-idle CPU at 43k stubs). Persist it next to
    // the sqlite as `{index}.hnsw` and reuse it across runs. The len guard
    // catches a sqlite that was rebuilt with a different stub count without
    // its companion being invalidated; `build-index --rebuild` deletes the
    // companion to cover the same-count-different-content case.
    let companion = std::path::PathBuf::from(format!("{path_str}.hnsw"));
    let vector_index = match HnswVectorIndex::load(&companion) {
        Ok(idx) if idx.len() == all.len() => {
            eprintln!(
                "loaded persisted HNSW companion: {} stubs from {} ({:.1}s)",
                idx.len(),
                companion.display(),
                started.elapsed().as_secs_f64()
            );
            idx
        }
        _ => {
            let mut idx = HnswVectorIndex::new();
            for (stub_id, embedding) in &all {
                idx.add(stub_id.clone(), embedding.clone());
            }
            idx.ensure_built();
            match idx.save(&companion) {
                Ok(()) => eprintln!(
                    "built and persisted HNSW companion: {} stubs from {} ({:.1}s)",
                    all.len(),
                    path.display(),
                    started.elapsed().as_secs_f64()
                ),
                Err(e) => eprintln!(
                    "built HNSW ({} stubs, {:.1}s); could not persist companion at {}: {e}",
                    all.len(),
                    started.elapsed().as_secs_f64(),
                    companion.display()
                ),
            }
            idx
        }
    };

    // BM25 over the same stub set so the eval fuses lexical + semantic like
    // the proxy. Built once here and shared read-only across items.
    let bm25 = caw_index::build_bm25_over_store(&store, all.iter().map(|(id, _)| id.clone()));
    eprintln!(
        "built BM25 lexical index: {} docs ({:.1}s total load)",
        bm25.len(),
        started.elapsed().as_secs_f64()
    );

    Ok(PrebuiltIndex {
        embedder: SharedEmbedder::new(embedder, "bge-small-en-v1.5"),
        store: SharedStore::new(store),
        index: SharedIndex::new(vector_index),
        bm25: std::sync::Arc::new(bm25),
        stub_summaries,
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let trimmed: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{}...", trimmed)
    }
}
