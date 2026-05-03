use anyhow::{Context, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;

use caw_bench::adapter_factory::{self, AdapterKind, AdapterSpec};
use caw_bench::intent::{IntentBenchCase, default_cases, empty_field_scores, score_case};
use caw_core::{CompletionRequest, ModelAdapter, QueryIntent};
use serde::Serialize;

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

    /// Optional JSON output path.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct CandidateCaseResult {
    case_id: String,
    query: String,
    tags: Vec<String>,
    predicted: Option<QueryIntent>,
    error: Option<String>,
    exact_match: bool,
    correct_fields: usize,
    total_fields: usize,
}

#[derive(Debug, Serialize)]
struct CandidateSummary {
    model: String,
    exact_match_rate: f32,
    field_accuracy: BTreeMap<String, f32>,
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
            },
            &runtime,
        )
        .with_context(|| format!("build adapter for intent candidate {}", model))?;
        summaries.push(run_candidate(model, adapter.as_ref(), &cases)?);
    }

    let report = IntentBenchReport {
        adapter: format!("{:?}", cli.adapter),
        cases: cases.len(),
        summaries,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = cli.out {
        std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
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

    for case in cases {
        match classify_case(adapter, case) {
            Ok(predicted) => {
                let score = score_case(&case.expected, &predicted);
                if score.exact_match {
                    exact_matches += 1;
                }
                for (field, ok) in &score.fields {
                    if let Some(entry) = field_scores.get_mut(field) {
                        entry.total += 1;
                        if *ok {
                            entry.correct += 1;
                        }
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
                    tags: case.tags.iter().map(|tag| (*tag).to_string()).collect(),
                    predicted: Some(predicted),
                    error: None,
                    exact_match: score.exact_match,
                    correct_fields: score.correct_fields,
                    total_fields: score.total_fields,
                });
            }
            Err(error) => {
                parse_failures += 1;
                for entry in field_scores.values_mut() {
                    entry.total += 1;
                }
                for tag in case.tags {
                    let entry = tag_hits.entry((*tag).to_string()).or_insert((0, 0));
                    entry.1 += 1;
                }
                case_results.push(CandidateCaseResult {
                    case_id: case.id.to_string(),
                    query: case.query.to_string(),
                    tags: case.tags.iter().map(|tag| (*tag).to_string()).collect(),
                    predicted: None,
                    error: Some(error.to_string()),
                    exact_match: false,
                    correct_fields: 0,
                    total_fields: field_scores.len(),
                });
            }
        }
    }

    let field_accuracy = field_scores
        .into_iter()
        .map(|(field, score)| {
            let accuracy = if score.total == 0 {
                0.0
            } else {
                score.correct as f32 / score.total as f32
            };
            (field.to_string(), accuracy)
        })
        .collect();
    let tag_exact_match_rate = tag_hits
        .into_iter()
        .map(|(tag, (hits, total))| {
            let accuracy = if total == 0 {
                0.0
            } else {
                hits as f32 / total as f32
            };
            (tag, accuracy)
        })
        .collect();

    Ok(CandidateSummary {
        model: model.to_string(),
        exact_match_rate: exact_matches as f32 / cases.len() as f32,
        field_accuracy,
        tag_exact_match_rate,
        parse_failures,
        cases: case_results,
    })
}

fn classify_case(adapter: &dyn ModelAdapter, case: &IntentBenchCase) -> Result<QueryIntent> {
    let response = adapter.complete(CompletionRequest {
        system: QueryIntent::classifier_system_prompt().to_string(),
        user: QueryIntent::classifier_user_prompt(case.query),
        workspace_fragments: Vec::new(),
        workspace_guidance: Vec::new(),
    })?;
    QueryIntent::from_classifier_response(&response.answer)
        .map_err(|e| anyhow::anyhow!("invalid classifier response for {}: {e}", case.id))
}
