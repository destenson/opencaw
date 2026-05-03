use caw_core::QueryIntent;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize)]
pub struct IntentBenchCase {
    pub id: &'static str,
    pub query: &'static str,
    pub tags: &'static [&'static str],
    pub expected: QueryIntent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FieldOutcome {
    TP,
    FP,
    TN,
    FN,
}

impl FieldOutcome {
    pub fn is_correct(self) -> bool {
        matches!(self, FieldOutcome::TP | FieldOutcome::TN)
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct IntentFieldScore {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_count: usize,
}

impl IntentFieldScore {
    pub fn precision(&self) -> f32 {
        let denom = (self.tp + self.fp) as f32;
        if denom == 0.0 { 0.0 } else { self.tp as f32 / denom }
    }

    pub fn recall(&self) -> f32 {
        let denom = (self.tp + self.fn_count) as f32;
        if denom == 0.0 { 0.0 } else { self.tp as f32 / denom }
    }

    pub fn f1(&self) -> f32 {
        let p = self.precision();
        let r = self.recall();
        let denom = p + r;
        if denom == 0.0 { 0.0 } else { 2.0 * p * r / denom }
    }

    pub fn accuracy(&self) -> f32 {
        let total = (self.tp + self.fp + self.tn + self.fn_count) as f32;
        if total == 0.0 { 0.0 } else { (self.tp + self.tn) as f32 / total }
    }

    pub fn record(&mut self, outcome: FieldOutcome) {
        match outcome {
            FieldOutcome::TP => self.tp += 1,
            FieldOutcome::FP => self.fp += 1,
            FieldOutcome::TN => self.tn += 1,
            FieldOutcome::FN => self.fn_count += 1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentCaseScore {
    pub exact_match: bool,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_count: usize,
    pub fields: BTreeMap<&'static str, FieldOutcome>,
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
                confidence: Some(1.0),
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
                confidence: Some(1.0),
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
                confidence: Some(1.0),
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
                confidence: Some(1.0),
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
                confidence: Some(1.0),
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "benchmarking_completed",
            query: "is benchmarking all completed?",
            tags: &["status", "completion", "grounded"],
            expected: QueryIntent {
                is_status_request: true,
                wants_completion_state: true,
                needs_grounded_evidence_only: true,
                confidence: Some(1.0),
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "benchmarking_pending",
            query: "what benchmarking work is still pending?",
            tags: &["status", "completion", "inventory", "grounded"],
            expected: QueryIntent {
                is_inventory_request: true,
                is_status_request: true,
                wants_completion_state: true,
                needs_grounded_evidence_only: true,
                confidence: Some(1.0),
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "benchmarking_done_list",
            query: "which benchmarking tasks are done already?",
            tags: &["status", "completion", "inventory", "grounded"],
            expected: QueryIntent {
                is_inventory_request: true,
                is_status_request: true,
                wants_completion_state: true,
                needs_grounded_evidence_only: true,
                confidence: Some(1.0),
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "next_step_benchmarking",
            query: "what's next for benchmarking?",
            tags: &["next-step", "actions", "grounded"],
            expected: QueryIntent {
                is_next_step_request: true,
                wants_recommended_actions: true,
                needs_grounded_evidence_only: true,
                confidence: Some(1.0),
                ..Default::default()
            },
        },
        IntentBenchCase {
            id: "next_step_project",
            query: "what should I work on next?",
            tags: &["next-step", "actions", "grounded"],
            expected: QueryIntent {
                is_next_step_request: true,
                wants_recommended_actions: true,
                needs_grounded_evidence_only: true,
                confidence: Some(1.0),
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
                confidence: Some(1.0),
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
                confidence: Some(1.0),
                ..Default::default()
            },
        },
        // "what changed" asks WHAT, not WHY — wants_explanation requires WHY/HOW.
        IntentBenchCase {
            id: "config_diff_explain",
            query: "what changed between the minimal and default sweep configs?",
            tags: &["comparison", "inventory"],
            expected: QueryIntent {
                is_inventory_request: true,
                wants_comparison: true,
                needs_grounded_evidence_only: true,
                confidence: Some(1.0),
                ..Default::default()
            },
        },
    ]
}

/// Score one classification result.
///
/// `emitted_keys` is the set of field names the model actually included in its JSON.
/// An absent field is treated as an intentional false (sparse output is valid):
/// absent + expected=false → TN, absent + expected=true → FN.
pub fn score_case(
    expected: &QueryIntent,
    predicted: &QueryIntent,
    emitted_keys: &HashSet<String>,
) -> IntentCaseScore {
    let classify = |name: &str, exp: bool, pred: bool| -> FieldOutcome {
        let effective = if emitted_keys.contains(name) { pred } else { false };
        match (exp, effective) {
            (true, true) => FieldOutcome::TP,
            (false, true) => FieldOutcome::FP,
            (false, false) => FieldOutcome::TN,
            (true, false) => FieldOutcome::FN,
        }
    };

    let mut fields = BTreeMap::new();
    fields.insert("is_inventory_request", classify("is_inventory_request", expected.is_inventory_request, predicted.is_inventory_request));
    fields.insert("is_results_request", classify("is_results_request", expected.is_results_request, predicted.is_results_request));
    fields.insert("is_status_request", classify("is_status_request", expected.is_status_request, predicted.is_status_request));
    fields.insert("is_next_step_request", classify("is_next_step_request", expected.is_next_step_request, predicted.is_next_step_request));
    fields.insert("wants_exact_names_or_paths", classify("wants_exact_names_or_paths", expected.wants_exact_names_or_paths, predicted.wants_exact_names_or_paths));
    fields.insert("wants_numeric_values", classify("wants_numeric_values", expected.wants_numeric_values, predicted.wants_numeric_values));
    fields.insert("wants_latest_run_only", classify("wants_latest_run_only", expected.wants_latest_run_only, predicted.wants_latest_run_only));
    fields.insert("wants_comparison", classify("wants_comparison", expected.wants_comparison, predicted.wants_comparison));
    fields.insert("wants_explanation", classify("wants_explanation", expected.wants_explanation, predicted.wants_explanation));
    fields.insert("wants_completion_state", classify("wants_completion_state", expected.wants_completion_state, predicted.wants_completion_state));
    fields.insert("wants_recommended_actions", classify("wants_recommended_actions", expected.wants_recommended_actions, predicted.wants_recommended_actions));
    fields.insert("needs_grounded_evidence_only", classify("needs_grounded_evidence_only", expected.needs_grounded_evidence_only, predicted.needs_grounded_evidence_only));
    fields.insert("abstain", classify("abstain", expected.abstain, predicted.abstain));

    let tp = fields.values().filter(|o| **o == FieldOutcome::TP).count();
    let fp = fields.values().filter(|o| **o == FieldOutcome::FP).count();
    let tn = fields.values().filter(|o| **o == FieldOutcome::TN).count();
    let fn_count = fields.values().filter(|o| **o == FieldOutcome::FN).count();

    IntentCaseScore {
        exact_match: fp == 0 && fn_count == 0,
        tp,
        fp,
        tn,
        fn_count,
        fields,
    }
}

pub const KNOWN_FIELDS: &[&str] = &[
    "is_inventory_request",
    "is_results_request",
    "is_status_request",
    "is_next_step_request",
    "wants_exact_names_or_paths",
    "wants_numeric_values",
    "wants_latest_run_only",
    "wants_comparison",
    "wants_explanation",
    "wants_completion_state",
    "wants_recommended_actions",
    "needs_grounded_evidence_only",
    "abstain",
];

pub fn empty_field_scores() -> BTreeMap<&'static str, IntentFieldScore> {
    KNOWN_FIELDS
        .iter()
        .map(|name| (*name, IntentFieldScore::default()))
        .collect()
}
