use std::collections::HashMap;

use caw_core::StubId;

use crate::metrics::{
    ContextEfficiency, CooperationMetrics, FalseRecallMetrics, HysteresisAnalysis,
    RecallObservation, RecallMetrics, term_overlap,
};

/// Accumulates events during an orchestrator session and produces
/// all eval metrics on demand. Designed to be embedded in the orchestrator
/// loop with minimal overhead — each record call is O(1).
pub struct SessionEvaluator {
    recalls: Vec<RecallEvent>,
    evictions: Vec<EvictionEvent>,
    probes: Vec<ProbeEvent>,
    annotations: Vec<AnnotationEvent>,
    turns: Vec<TurnEvent>,

    /// Tracks which fragments are currently loaded and when they were loaded,
    /// keyed by stub id string for fast lookup.
    loaded_at_step: HashMap<String, usize>,

    /// Running token counts for context efficiency
    total_content_tokens: usize,
    total_stub_tokens: usize,
    total_overhead_tokens: usize,

    current_step: usize,

    /// How many steps apart a load-evict-reload cycle must be to count as thrashing
    thrashing_window: usize,

    /// Threshold for the false-recall term overlap heuristic
    false_recall_overlap_threshold: f32,
}

#[derive(Debug, Clone)]
struct RecallEvent {
    stub_id: String,
    score: f32,
    content_tokens: usize,
    step: usize,
    stub_summary: Option<String>,
    recalled_content: Option<String>,
}

#[derive(Debug, Clone)]
struct EvictionEvent {
    stub_id: String,
    step: usize,
}

#[derive(Debug, Clone)]
struct ProbeEvent {
    _query: String,
    matched: bool,
    _score: f32,
}

#[derive(Debug, Clone)]
struct AnnotationEvent {
    stub_id: String,
    content: String,
}

#[derive(Debug, Clone)]
struct TurnEvent {
    _tokens: usize,
}

/// Builder for SessionEvaluator — allows configuring thresholds before use.
pub struct SessionEvaluatorBuilder {
    thrashing_window: usize,
    false_recall_overlap_threshold: f32,
}

impl Default for SessionEvaluatorBuilder {
    fn default() -> Self {
        Self {
            thrashing_window: 5,
            false_recall_overlap_threshold: 0.15,
        }
    }
}

impl SessionEvaluatorBuilder {
    /// How close together (in steps) a load-evict-reload must occur to count as thrashing
    pub fn thrashing_window(mut self, window: usize) -> Self {
        self.thrashing_window = window;
        self
    }

    /// Minimum term overlap between stub summary and recalled content.
    /// Below this, the recall is considered a potential false recall.
    pub fn false_recall_overlap_threshold(mut self, threshold: f32) -> Self {
        self.false_recall_overlap_threshold = threshold;
        self
    }

    pub fn build(self) -> SessionEvaluator {
        SessionEvaluator {
            recalls: Vec::new(),
            evictions: Vec::new(),
            probes: Vec::new(),
            annotations: Vec::new(),
            turns: Vec::new(),
            loaded_at_step: HashMap::new(),
            total_content_tokens: 0,
            total_stub_tokens: 0,
            total_overhead_tokens: 0,
            current_step: 0,
            thrashing_window: self.thrashing_window,
            false_recall_overlap_threshold: self.false_recall_overlap_threshold,
        }
    }
}

impl SessionEvaluator {
    pub fn builder() -> SessionEvaluatorBuilder {
        SessionEvaluatorBuilder::default()
    }

    pub fn record_recall(&mut self, stub_id: &StubId, score: f32, content_tokens: usize) {
        self.recalls.push(RecallEvent {
            stub_id: stub_id.0.clone(),
            score,
            content_tokens,
            step: self.current_step,
            stub_summary: None,
            recalled_content: None,
        });
        self.loaded_at_step
            .insert(stub_id.0.clone(), self.current_step);
        self.total_content_tokens += content_tokens;
    }

    /// Record a recall with the stub summary and actual content, enabling
    /// false-recall detection via term overlap.
    pub fn record_recall_with_content(
        &mut self,
        stub_id: &StubId,
        score: f32,
        content_tokens: usize,
        stub_summary: &str,
        recalled_content: &str,
    ) {
        self.recalls.push(RecallEvent {
            stub_id: stub_id.0.clone(),
            score,
            content_tokens,
            step: self.current_step,
            stub_summary: Some(stub_summary.to_string()),
            recalled_content: Some(recalled_content.to_string()),
        });
        self.loaded_at_step
            .insert(stub_id.0.clone(), self.current_step);
        self.total_content_tokens += content_tokens;
    }

    pub fn record_eviction(&mut self, stub_id: &StubId, step: usize) {
        self.current_step = self.current_step.max(step);
        self.evictions.push(EvictionEvent {
            stub_id: stub_id.0.clone(),
            step,
        });
        self.loaded_at_step.remove(&stub_id.0);
    }

    pub fn record_probe(&mut self, query: &str, matched: bool, score: f32) {
        self.probes.push(ProbeEvent {
            _query: query.to_string(),
            matched,
            _score: score,
        });
    }

    pub fn record_annotation(&mut self, stub_id: &str, content: &str) {
        self.annotations.push(AnnotationEvent {
            stub_id: stub_id.to_string(),
            content: content.to_string(),
        });
    }

    pub fn record_turn(&mut self, turn_tokens: usize) {
        self.turns.push(TurnEvent {
            _tokens: turn_tokens,
        });
        self.current_step += 1;
    }

    pub fn record_stub_tokens(&mut self, tokens: usize) {
        self.total_stub_tokens += tokens;
    }

    pub fn record_overhead_tokens(&mut self, tokens: usize) {
        self.total_overhead_tokens += tokens;
    }

    /// Basic recall@k / precision@k over recorded recall events.
    /// `expected_ids` is the ground-truth set of stub ids that should have been recalled.
    pub fn recall_metrics(&self, expected_ids: &[String], k: usize) -> RecallMetrics {
        let top: Vec<&RecallEvent> = self.recalls.iter().take(k).collect();
        let hits = top
            .iter()
            .filter(|r| expected_ids.contains(&r.stub_id))
            .count();

        let recall = if expected_ids.is_empty() {
            1.0
        } else {
            hits as f32 / expected_ids.len() as f32
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

    pub fn false_recall_metrics(&self) -> FalseRecallMetrics {
        let observations: Vec<RecallObservation> = self
            .recalls
            .iter()
            .filter_map(|r| {
                match (&r.stub_summary, &r.recalled_content) {
                    (Some(summary), Some(content)) => Some(RecallObservation {
                        stub_summary: summary.clone(),
                        recalled_content: content.clone(),
                        model_flagged_conflict: false,
                    }),
                    _ => None,
                }
            })
            .collect();

        FalseRecallMetrics::from_observations(&observations, self.false_recall_overlap_threshold)
    }

    pub fn context_efficiency(&self, nominal_tokens: usize) -> ContextEfficiency {
        ContextEfficiency::compute(
            nominal_tokens,
            self.total_content_tokens,
            self.total_stub_tokens,
            self.total_overhead_tokens,
        )
    }

    pub fn hysteresis_analysis(
        &self,
        current_load_threshold: f32,
        current_unload_threshold: f32,
    ) -> HysteresisAnalysis {
        let total_loads = self.recalls.len();
        let total_evictions = self.evictions.len();

        // Detect thrashing: same fragment loaded, evicted, and reloaded within the window
        let mut thrashing_events = 0usize;
        let mut load_steps: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut evict_steps: HashMap<&str, Vec<usize>> = HashMap::new();

        for r in &self.recalls {
            load_steps.entry(&r.stub_id).or_default().push(r.step);
        }
        for e in &self.evictions {
            evict_steps.entry(&e.stub_id).or_default().push(e.step);
        }

        for (stub_id, loads) in &load_steps {
            if let Some(evicts) = evict_steps.get(stub_id) {
                // For each eviction, check if there's a reload within the window
                for &evict_step in evicts {
                    if loads
                        .iter()
                        .any(|&load_step| load_step > evict_step && load_step - evict_step <= self.thrashing_window)
                    {
                        thrashing_events += 1;
                    }
                }
            }
        }

        // Average loaded duration: for each eviction, how many steps was the fragment loaded?
        let mut durations = Vec::new();
        for e in &self.evictions {
            if let Some(loads) = load_steps.get(e.stub_id.as_str()) {
                // Find the most recent load before this eviction
                if let Some(&load_step) = loads.iter().filter(|&&s| s <= e.step).max() {
                    durations.push((e.step - load_step) as f32);
                }
            }
        }

        let avg_loaded_duration_steps = if durations.is_empty() {
            0.0
        } else {
            durations.iter().sum::<f32>() / durations.len() as f32
        };

        // Suggest wider gap if thrashing is detected.
        // The adjustment is a heuristic starting point — the magnitude should
        // be tuned empirically per workload.
        let gap_adjustment = if thrashing_events > 0 {
            0.05 * (thrashing_events as f32).min(4.0)
        } else {
            0.0
        };

        let suggested_load = (current_load_threshold + gap_adjustment).min(0.95);
        let suggested_unload = (current_unload_threshold - gap_adjustment).max(0.1);

        HysteresisAnalysis {
            total_loads,
            total_evictions,
            thrashing_events,
            avg_loaded_duration_steps,
            suggested_load_threshold: suggested_load,
            suggested_unload_threshold: suggested_unload,
        }
    }

    pub fn cooperation_metrics(&self) -> CooperationMetrics {
        let total_turns = self.turns.len();
        let probes_emitted = self.probes.len();
        let annotations_emitted = self.annotations.len();

        let probes_per_turn = if total_turns == 0 {
            0.0
        } else {
            probes_emitted as f32 / total_turns as f32
        };

        let annotations_per_turn = if total_turns == 0 {
            0.0
        } else {
            annotations_emitted as f32 / total_turns as f32
        };

        let useful_probes = self.probes.iter().filter(|p| p.matched).count();
        let useful_probes_pct = if probes_emitted == 0 {
            0.0
        } else {
            useful_probes as f32 / probes_emitted as f32
        };

        // Annotation quality: average term overlap between each annotation
        // and the recalled content for that stub (if available).
        let mut quality_scores = Vec::new();
        for ann in &self.annotations {
            // Find the most recent recall for this stub to compare against
            if let Some(recall) = self
                .recalls
                .iter()
                .rev()
                .find(|r| r.stub_id == ann.stub_id)
            {
                if let Some(ref content) = recall.recalled_content {
                    quality_scores.push(term_overlap(&ann.content, content));
                }
            }
        }

        let annotation_quality = if quality_scores.is_empty() {
            0.0
        } else {
            quality_scores.iter().sum::<f32>() / quality_scores.len() as f32
        };

        CooperationMetrics {
            total_turns,
            probes_emitted,
            annotations_emitted,
            probes_per_turn,
            annotations_per_turn,
            useful_probes_pct,
            annotation_quality,
        }
    }
}
