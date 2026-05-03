use caw_core::QueryIntent;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize)]
pub struct IntentBenchCase {
    pub id: &'static str,
    pub query: &'static str,
    pub tags: &'static [&'static str],
    pub expected: QueryIntent,
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentFieldScore {
    pub correct: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentCaseScore {
    pub exact_match: bool,
    pub correct_fields: usize,
    pub total_fields: usize,
    pub fields: BTreeMap<&'static str, bool>,
}

pub fn default_cases() -> Vec<IntentBenchCase> {
    vec![
        IntentBenchCase {
            id: "inventory_bench_runs",
            query: "do you know what benchmarks have been run?",
            tags: &["inventory", "exact", "grounded"],
            expected: QueryIntent {
                is_inventory_request: true,
                wants_exact_names_or_paths: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "results_bench_metrics",
            query: "can you tell me what the benchmark results were?",
            tags: &["results", "numeric", "grounded"],
            expected: QueryIntent {
                is_results_request: true,
                wants_numeric_values: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "latest_summary_path",
            query: "show me the path to the most recent throughput summary file",
            tags: &["inventory", "exact", "latest"],
            expected: QueryIntent {
                is_inventory_request: true,
                wants_exact_names_or_paths: true,
                wants_latest_run_only: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "compare_latency",
            query: "compare the latency of the latest candle and onnx runs",
            tags: &["results", "comparison", "latest", "numeric"],
            expected: QueryIntent {
                is_results_request: true,
                wants_numeric_values: true,
                wants_latest_run_only: true,
                wants_comparison: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "explain_slowdown",
            query: "why is onnx-512-256 slower than candle-512-256?",
            tags: &["results", "comparison", "explanation"],
            expected: QueryIntent {
                is_results_request: true,
                wants_comparison: true,
                wants_explanation: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "summarize_runs",
            query: "summarize the benchmark runs we have so far",
            tags: &["inventory", "results"],
            expected: QueryIntent {
                is_inventory_request: true,
                is_results_request: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "metric_values_only",
            query: "what throughput numbers were reported for onnx-128-64?",
            tags: &["results", "numeric", "exact"],
            expected: QueryIntent {
                is_results_request: true,
                wants_exact_names_or_paths: true,
                wants_numeric_values: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "config_diff_explain",
            query: "what changed between the minimal and default sweep configs?",
            tags: &["comparison", "explanation", "inventory"],
            expected: QueryIntent {
                is_inventory_request: true,
                wants_comparison: true,
                wants_explanation: true,
                needs_grounded_evidence_only: true,
                confidence: 1.0,
                ..Default::default()
            },
        },
    ]
}

pub fn score_case(expected: &QueryIntent, predicted: &QueryIntent) -> IntentCaseScore {
    let mut fields = BTreeMap::new();
    fields.insert(
        "is_inventory_request",
        expected.is_inventory_request == predicted.is_inventory_request,
    );
    fields.insert(
        "is_results_request",
        expected.is_results_request == predicted.is_results_request,
    );
    fields.insert(
        "wants_exact_names_or_paths",
        expected.wants_exact_names_or_paths == predicted.wants_exact_names_or_paths,
    );
    fields.insert(
        "wants_numeric_values",
        expected.wants_numeric_values == predicted.wants_numeric_values,
    );
    fields.insert(
        "wants_latest_run_only",
        expected.wants_latest_run_only == predicted.wants_latest_run_only,
    );
    fields.insert(
        "wants_comparison",
        expected.wants_comparison == predicted.wants_comparison,
    );
    fields.insert(
        "wants_explanation",
        expected.wants_explanation == predicted.wants_explanation,
    );
    fields.insert(
        "needs_grounded_evidence_only",
        expected.needs_grounded_evidence_only == predicted.needs_grounded_evidence_only,
    );
    fields.insert("abstain", expected.abstain == predicted.abstain);

    let correct_fields = fields.values().filter(|ok| **ok).count();
    let total_fields = fields.len();

    IntentCaseScore {
        exact_match: correct_fields == total_fields,
        correct_fields,
        total_fields,
        fields,
    }
}

pub fn empty_field_scores() -> BTreeMap<&'static str, IntentFieldScore> {
    [
        "is_inventory_request",
        "is_results_request",
        "wants_exact_names_or_paths",
        "wants_numeric_values",
        "wants_latest_run_only",
        "wants_comparison",
        "wants_explanation",
        "needs_grounded_evidence_only",
        "abstain",
    ]
    .into_iter()
    .map(|name| {
        (
            name,
            IntentFieldScore {
                correct: 0,
                total: 0,
            },
        )
    })
    .collect()
}
