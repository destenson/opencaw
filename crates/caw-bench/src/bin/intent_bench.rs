use anyhow::{Context, Result};
use clap::Parser;
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use caw_bench::adapter_factory::{self, AdapterKind, AdapterSpec};
use caw_bench::intent::{
    IntentBenchCase, IntentFieldScore, KNOWN_FIELDS, default_cases, empty_field_scores, score_case,
};
use caw_core::{CompletionRequest, ModelAdapter, QueryIntent};
use serde::Serialize;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(
    name = "caw-bench-intent",
    about = "Benchmark small models for query-intent classification"
)]
struct Cli {
    /// Adapter used for all candidate models in this run.
    #[arg(long, value_enum, default_value_t = AdapterKind::Ollama)]
    adapter: AdapterKind,

    /// Candidate model names to benchmark. Repeat this flag to compare a pool.
    #[arg(long, required = true)]
    candidate: Vec<String>,

    /// Ollama base URL.
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// vLLM / OpenAI-compatible base URL.
    #[arg(long, default_value = "http://localhost:8000")]
    openai_url: String,

    /// Sampling temperature for the classifier. Keep 0.0 for deterministic routing.
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Cap the Ollama context window (num_ctx). Dramatically reduces VRAM for
    /// models with large default contexts (e.g. 32k). Classification prompts
    /// fit well within 4096 tokens; 2048 is sufficient for most cases.
    /// Ignored for non-Ollama adapters.
    #[arg(long)]
    num_ctx: Option<u32>,

    /// Optional JSON output path.
    #[arg(long)]
    out: Option<PathBuf>,

    /// After benchmarking all individual candidates, also run an ensemble that
    /// majority-votes across the top N models (by micro-F1) and adds the result
    /// to the leaderboard. Set to 0 to disable.
    #[arg(long, default_value_t = 3)]
    ensemble: usize,
}

#[derive(Debug, Serialize)]
struct CandidateCaseResult {
    case_id: String,
    query: String,
    tags: Vec<String>,
    predicted: Option<QueryIntent>,
    error: Option<String>,
    /// Raw model response, included only when exact_match is false or a parse error occurred.
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_response: Option<String>,
    exact_match: bool,
    tp: usize,
    fp: usize,
    tn: usize,
    fn_count: usize,
}

#[derive(Debug, Serialize)]
struct CandidateSummary {
    model: String,
    exact_match_rate: f32,
    micro_precision: f32,
    micro_recall: f32,
    micro_f1: f32,
    field_scores: BTreeMap<String, IntentFieldScore>,
    tag_exact_match_rate: BTreeMap<String, f32>,
    parse_failures: usize,
    cases: Vec<CandidateCaseResult>,
}

#[derive(Debug, Serialize)]
struct IntentBenchReport {
    adapter: String,
    cases: usize,
    summaries: Vec<CandidateSummary>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cases = default_cases();
    let runtime = caw_adapters::create_runtime().context("create tokio runtime")?;

    let mut summaries = Vec::new();
    for model in &cli.candidate {
        let adapter = adapter_factory::build(
            AdapterSpec {
                kind: cli.adapter,
                model,
                ollama_url: &cli.ollama_url,
                openai_url: &cli.openai_url,
                temperature: Some(cli.temperature),
                num_ctx: cli.num_ctx,
            },
            &runtime,
        )
        .with_context(|| format!("build adapter for intent candidate {}", model))?;
        summaries.push(run_candidate(model, adapter.as_ref(), &cases)?);
    }

    if cli.ensemble > 0 && summaries.len() > 1 {
        summaries.push(run_ensemble(&summaries, &cases, cli.ensemble));
    }

    let report = IntentBenchReport {
        adapter: format!("{:?}", cli.adapter),
        cases: cases.len(),
        summaries,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = cli.out {
        std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
        print_summary(&report, Some(path.as_path()), &mut io::stdout())?;
    } else {
        println!("{json}");
        print_summary(&report, None, &mut io::stderr())?;
    }
    Ok(())
}

fn print_summary(
    report: &IntentBenchReport,
    out_path: Option<&std::path::Path>,
    writer: &mut dyn Write,
) -> Result<()> {
    let mut summaries = report.summaries.iter().collect::<Vec<_>>();
    summaries.sort_by(compare_summary);

    writeln!(writer)?;
    writeln!(writer, "intent bench summary")?;
    if let Some(path) = out_path {
        writeln!(writer, "report: {}", path.display())?;
    }
    writeln!(writer, "cases: {}", report.cases)?;
    writeln!(writer)?;
    writeln!(writer, "leaderboard")?;

    for (index, summary) in summaries.iter().take(3).enumerate() {
        writeln!(
            writer,
            concat!(
                "{}. {} exact={:.1}% f1={:.3} parse_failures={} ",
                "next_step_f1={:.3} status_f1={:.3} ",
                "inventory_f1={:.3} results_f1={:.3} grounded_f1={:.3}"
            ),
            index + 1,
            summary.model,
            summary.exact_match_rate * 100.0,
            summary.micro_f1,
            summary.parse_failures,
            field_f1(summary, "is_next_step_request"),
            field_f1(summary, "is_status_request"),
            field_f1(summary, "is_inventory_request"),
            field_f1(summary, "is_results_request"),
            field_f1(summary, "needs_grounded_evidence_only"),
        )?;
    }

    let tag_winners = collect_tag_winners(&summaries);
    if !tag_winners.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "tag winners")?;
        for (tag, winners) in tag_winners {
            writeln!(writer, "{}: {}", tag, winners.join(", "))?;
        }
    }

    writeln!(writer)?;
    writeln!(writer, "all models")?;
    for summary in summaries {
        writeln!(writer, "- {}", format_model_line(summary))?;
    }

    Ok(())
}

fn compare_summary(left: &&CandidateSummary, right: &&CandidateSummary) -> std::cmp::Ordering {
    right
        .micro_f1
        .partial_cmp(&left.micro_f1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.parse_failures.cmp(&right.parse_failures))
        .then_with(|| left.model.cmp(&right.model))
}

fn collect_tag_winners(summaries: &[&CandidateSummary]) -> Vec<(String, Vec<String>)> {
    let mut tags = summaries
        .iter()
        .flat_map(|summary| summary.tag_exact_match_rate.keys().cloned())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();

    let mut winners = Vec::new();
    for tag in tags {
        let best_score = summaries
            .iter()
            .map(|summary| summary.tag_exact_match_rate.get(&tag).copied().unwrap_or(0.0))
            .fold(-1.0f32, f32::max);
        if best_score <= 0.0 {
            continue;
        }
        let mut best_models = summaries
            .iter()
            .filter_map(|summary| {
                let score = summary.tag_exact_match_rate.get(&tag).copied().unwrap_or(0.0);
                ((score - best_score).abs() < f32::EPSILON)
                    .then(|| format!("{} ({:.1}%)", summary.model, score * 100.0))
            })
            .collect::<Vec<_>>();
        best_models.sort();
        winners.push((tag, best_models));
    }

    winners
}

fn format_model_line(summary: &CandidateSummary) -> String {
    format!(
        concat!(
            "{}: exact={:.1}% f1={:.3} (P={:.3} R={:.3}) parse_failures={} ",
            "inventory_f1={:.3} results_f1={:.3} status_f1={:.3} grounded_f1={:.3}"
        ),
        summary.model,
        summary.exact_match_rate * 100.0,
        summary.micro_f1,
        summary.micro_precision,
        summary.micro_recall,
        summary.parse_failures,
        field_f1(summary, "is_inventory_request"),
        field_f1(summary, "is_results_request"),
        field_f1(summary, "is_status_request"),
        field_f1(summary, "needs_grounded_evidence_only"),
    )
}

fn field_f1(summary: &CandidateSummary, field: &str) -> f32 {
    summary
        .field_scores
        .get(field)
        .map(|s| s.f1())
        .unwrap_or(0.0)
}

fn micro_metrics(
    total_tp: usize,
    total_fp: usize,
    total_fn: usize,
) -> (f32, f32, f32) {
    let precision = {
        let d = (total_tp + total_fp) as f32;
        if d == 0.0 { 0.0 } else { total_tp as f32 / d }
    };
    let recall = {
        let d = (total_tp + total_fn) as f32;
        if d == 0.0 { 0.0 } else { total_tp as f32 / d }
    };
    let f1 = {
        let d = precision + recall;
        if d == 0.0 { 0.0 } else { 2.0 * precision * recall / d }
    };
    (precision, recall, f1)
}

fn run_candidate(
    model: &str,
    adapter: &dyn ModelAdapter,
    cases: &[IntentBenchCase],
) -> Result<CandidateSummary> {
    let mut exact_matches = 0usize;
    let mut parse_failures = 0usize;
    let mut field_scores = empty_field_scores();
    let mut tag_hits: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut case_results = Vec::new();
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;

    for case in cases {
        let (score, predicted, raw, is_parse_failure) = match classify_case(adapter, case) {
            Ok((predicted, emitted_keys, raw)) => {
                let score = score_case(&case.expected, &predicted, &emitted_keys);
                (score, Some(predicted), raw, false)
            }
            Err((_, raw)) => {
                // Parse failure: treat as all-absent output — FN for every expected-true field.
                let score = score_case(&case.expected, &QueryIntent::default(), &HashSet::new());
                (score, None, raw, true)
            }
        };

        if is_parse_failure {
            parse_failures += 1;
        }
        if score.exact_match {
            exact_matches += 1;
        }

        total_tp += score.tp;
        total_fp += score.fp;
        total_fn += score.fn_count;

        for (field, outcome) in &score.fields {
            if let Some(entry) = field_scores.get_mut(field) {
                entry.record(*outcome);
            }
        }

        for tag in case.tags {
            let entry = tag_hits.entry((*tag).to_string()).or_insert((0, 0));
            entry.1 += 1;
            if score.exact_match {
                entry.0 += 1;
            }
        }

        let error = if is_parse_failure {
            Some(format!("parse failure for {}", case.id))
        } else {
            None
        };

        case_results.push(CandidateCaseResult {
            case_id: case.id.to_string(),
            query: case.query.to_string(),
            tags: case.tags.iter().map(|tag| (*tag).to_string()).collect(),
            predicted,
            error,
            raw_response: if score.exact_match { None } else { Some(raw) },
            exact_match: score.exact_match,
            tp: score.tp,
            fp: score.fp,
            tn: score.tn,
            fn_count: score.fn_count,
        });
    }

    let (micro_precision, micro_recall, micro_f1) = micro_metrics(total_tp, total_fp, total_fn);

    let field_scores_serializable = field_scores
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    let tag_exact_match_rate = tag_hits
        .into_iter()
        .map(|(tag, (hits, total))| {
            let rate = if total == 0 { 0.0 } else { hits as f32 / total as f32 };
            (tag, rate)
        })
        .collect();

    Ok(CandidateSummary {
        model: model.to_string(),
        exact_match_rate: exact_matches as f32 / cases.len() as f32,
        micro_precision,
        micro_recall,
        micro_f1,
        field_scores: field_scores_serializable,
        tag_exact_match_rate,
        parse_failures,
        cases: case_results,
    })
}

/// Build an ensemble summary by majority-voting across all individual summaries.
/// No models are re-invoked — votes come from predictions already collected.
fn run_ensemble(
    summaries: &[CandidateSummary],
    cases: &[IntentBenchCase],
    size: usize,
) -> CandidateSummary {
    use caw_core::QueryIntent;

    let mut ranked: Vec<&CandidateSummary> = summaries.iter().collect();
    ranked.sort_by(|a, b| b.micro_f1.partial_cmp(&a.micro_f1).unwrap_or(std::cmp::Ordering::Equal));
    let pool: Vec<&CandidateSummary> = ranked.into_iter().take(size).collect();

    let label = format!(
        "ensemble-top{}({})",
        pool.len(),
        pool.iter().map(|s| s.model.as_str()).collect::<Vec<_>>().join("+")
    );

    // All known fields are considered "emitted" in the ensemble result because
    // majority_vote produces an explicit true/false for each field.
    let all_fields: HashSet<String> = KNOWN_FIELDS.iter().map(|s| s.to_string()).collect();

    let mut exact_matches = 0usize;
    let mut field_scores = empty_field_scores();
    let mut tag_hits: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut case_results = Vec::new();
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;

    for (case_idx, case) in cases.iter().enumerate() {
        let votes: Vec<QueryIntent> = pool
            .iter()
            .filter_map(|s| s.cases.get(case_idx)?.predicted.clone())
            .collect();

        let (merged, is_all_failed) = if votes.is_empty() {
            (QueryIntent::default(), true)
        } else {
            (QueryIntent::majority_vote(&votes), false)
        };

        let score = score_case(&case.expected, &merged, &all_fields);

        if score.exact_match {
            exact_matches += 1;
        }
        total_tp += score.tp;
        total_fp += score.fp;
        total_fn += score.fn_count;

        for (field, outcome) in &score.fields {
            if let Some(entry) = field_scores.get_mut(field) {
                entry.record(*outcome);
            }
        }
        for tag in case.tags {
            let entry = tag_hits.entry((*tag).to_string()).or_insert((0, 0));
            entry.1 += 1;
            if score.exact_match {
                entry.0 += 1;
            }
        }

        case_results.push(CandidateCaseResult {
            case_id: case.id.to_string(),
            query: case.query.to_string(),
            tags: case.tags.iter().map(|t| (*t).to_string()).collect(),
            predicted: if is_all_failed { None } else { Some(merged) },
            error: if is_all_failed { Some("all models failed".to_string()) } else { None },
            raw_response: None,
            exact_match: score.exact_match,
            tp: score.tp,
            fp: score.fp,
            tn: score.tn,
            fn_count: score.fn_count,
        });
    }

    let (micro_precision, micro_recall, micro_f1) = micro_metrics(total_tp, total_fp, total_fn);

    let field_scores_serializable = field_scores
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    let tag_exact_match_rate = tag_hits
        .into_iter()
        .map(|(tag, (hits, total))| {
            let rate = if total == 0 { 0.0 } else { hits as f32 / total as f32 };
            (tag, rate)
        })
        .collect();

    CandidateSummary {
        model: label,
        exact_match_rate: exact_matches as f32 / cases.len() as f32,
        micro_precision,
        micro_recall,
        micro_f1,
        field_scores: field_scores_serializable,
        tag_exact_match_rate,
        parse_failures: 0,
        cases: case_results,
    }
}

fn classify_case(
    adapter: &dyn ModelAdapter,
    case: &IntentBenchCase,
) -> Result<(QueryIntent, HashSet<String>, String), (anyhow::Error, String)> {
    let response = adapter
        .complete(CompletionRequest {
            system: QueryIntent::classifier_system_prompt().to_string(),
            user: QueryIntent::classifier_user_prompt(case.query),
            workspace_fragments: Vec::new(),
            workspace_guidance: Vec::new(),
        })
        .map_err(|e| (anyhow::anyhow!(e), String::new()))?;
    let raw = response.answer.clone();
    QueryIntent::from_classifier_response(&response.answer)
        .map(|(intent, keys)| (intent, keys, raw.clone()))
        .map_err(|e| {
            (anyhow::anyhow!("invalid classifier response for {}: {e}", case.id), raw)
        })
}
