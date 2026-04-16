use caw_core::RecallFragment;

#[derive(Debug, Clone, Copy)]
pub struct RecallMetrics {
    pub recall_at_k: f32,
    pub precision_at_k: f32,
}

pub fn evaluate_recall(
    expected_stub_ids: &[String],
    retrieved: &[RecallFragment],
    k: usize,
) -> RecallMetrics {
    let top = retrieved.iter().take(k).collect::<Vec<_>>();
    let hits = top
        .iter()
        .filter(|f| expected_stub_ids.contains(&f.stub_id.0))
        .count();

    let recall = if expected_stub_ids.is_empty() {
        1.0
    } else {
        hits as f32 / expected_stub_ids.len() as f32
    };

    let precision = if top.is_empty() {
        1.0
    } else {
        hits as f32 / top.len() as f32
    };

    RecallMetrics {
        recall_at_k: recall,
        precision_at_k: precision,
    }
}

/// Tracks how often recalled content diverges from what the stub summary implied.
/// "False recall" = the model gets content that doesn't match its expectation
/// from the stub, leading to potential confusion or contradiction.
#[derive(Debug, Clone)]
pub struct FalseRecallMetrics {
    pub total_recalls: usize,
    pub contradictions_detected: usize,
    pub false_recall_rate: f32,
}

/// A single recall event paired with the stub summary that set expectations.
#[derive(Debug, Clone)]
pub struct RecallObservation {
    pub stub_summary: String,
    pub recalled_content: String,
    /// If the model explicitly flagged a conflict, this is true regardless of
    /// term overlap. Model judgment overrides heuristic measurement.
    pub model_flagged_conflict: bool,
}

impl FalseRecallMetrics {
    /// Compute false-recall rate from a set of observations.
    /// Uses term overlap as a heuristic: if the stub summary and recalled content
    /// share very few terms, the recall likely surprised the model.
    pub fn from_observations(observations: &[RecallObservation], overlap_threshold: f32) -> Self {
        if observations.is_empty() {
            return Self {
                total_recalls: 0,
                contradictions_detected: 0,
                false_recall_rate: 0.0,
            };
        }

        let contradictions = observations
            .iter()
            .filter(|obs| {
                obs.model_flagged_conflict
                    || term_overlap(&obs.stub_summary, &obs.recalled_content) < overlap_threshold
            })
            .count();

        Self {
            total_recalls: observations.len(),
            contradictions_detected: contradictions,
            false_recall_rate: contradictions as f32 / observations.len() as f32,
        }
    }
}

/// Measures how much of the context window carries useful content vs overhead.
#[derive(Debug, Clone, Copy)]
pub struct ContextEfficiency {
    pub nominal_tokens: usize,
    pub content_tokens: usize,
    pub stub_tokens: usize,
    pub overhead_tokens: usize,
    pub efficiency_ratio: f32,
}

impl ContextEfficiency {
    pub fn compute(
        nominal_tokens: usize,
        content_tokens: usize,
        stub_tokens: usize,
        overhead_tokens: usize,
    ) -> Self {
        let efficiency_ratio = if nominal_tokens == 0 {
            0.0
        } else {
            content_tokens as f32 / nominal_tokens as f32
        };

        Self {
            nominal_tokens,
            content_tokens,
            stub_tokens,
            overhead_tokens,
            efficiency_ratio,
        }
    }
}

/// Analyzes whether load/unload thresholds are well-calibrated by looking
/// for thrashing — the same fragment cycling in and out rapidly.
#[derive(Debug, Clone)]
pub struct HysteresisAnalysis {
    pub total_loads: usize,
    pub total_evictions: usize,
    pub thrashing_events: usize,
    pub avg_loaded_duration_steps: f32,
    pub suggested_load_threshold: f32,
    pub suggested_unload_threshold: f32,
}

/// Measures how well a model cooperates with the probe/annotation protocol.
/// Useful for deciding whether to run in cooperative vs transparent mode.
#[derive(Debug, Clone)]
pub struct CooperationMetrics {
    pub total_turns: usize,
    pub probes_emitted: usize,
    pub annotations_emitted: usize,
    pub probes_per_turn: f32,
    pub annotations_per_turn: f32,
    /// Fraction of probes that actually triggered a recall above threshold
    pub useful_probes_pct: f32,
    /// Average term overlap between annotations and the content they reference,
    /// as a proxy for annotation quality
    pub annotation_quality: f32,
}

/// Normalized term overlap between two texts. Extracts unique lowercase
/// alphanumeric tokens and returns |intersection| / |union|.
/// Returns 0.0 if both texts are empty.
pub fn term_overlap(a: &str, b: &str) -> f32 {
    let terms_a: std::collections::HashSet<String> = a
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect()
        })
        .filter(|w: &String| !w.is_empty())
        .collect();

    let terms_b: std::collections::HashSet<String> = b
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect()
        })
        .filter(|w: &String| !w.is_empty())
        .collect();

    if terms_a.is_empty() && terms_b.is_empty() {
        return 0.0;
    }

    let intersection = terms_a.intersection(&terms_b).count();
    let union = terms_a.union(&terms_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}
