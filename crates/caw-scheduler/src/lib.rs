use caw_core::{BudgetScheduler, RecallFragment, SchedulerDecision, SchedulerInput, StubId};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug)]
pub struct GreedyBudgetScheduler {
    load_threshold: f32,
    unload_threshold: f32,
    loaded_scores: RefCell<HashMap<StubId, f32>>,
}

impl GreedyBudgetScheduler {
    pub fn new(load_threshold: f32, unload_threshold: f32) -> Self {
        Self {
            load_threshold,
            unload_threshold,
            loaded_scores: RefCell::new(HashMap::new()),
        }
    }

    pub fn with_hysteresis() -> Self {
        Self::new(0.7, 0.4)
    }
}

impl Default for GreedyBudgetScheduler {
    fn default() -> Self {
        Self::new(0.3, 0.2)
    }
}

impl BudgetScheduler for GreedyBudgetScheduler {
    fn schedule(&self, input: SchedulerInput) -> SchedulerDecision {
        let budget = input.budget.available_for_workspace();
        let originally_loaded = input.currently_loaded.clone();
        let loaded_ids: Vec<StubId> = originally_loaded.iter().map(|f| f.stub_id.clone()).collect();

        let mut selected = Vec::<RecallFragment>::new();
        let mut used = 0usize;

        // Keep currently loaded fragments that pass unload threshold
        for frag in input.currently_loaded {
            let score = self.loaded_scores.borrow().get(&frag.stub_id).copied().unwrap_or(1.0);
            
            // Use unload threshold for hysteresis
            if score >= self.unload_threshold && used + frag.tokens <= budget {
                used += frag.tokens;
                selected.push(frag);
            }
        }

        let mut admitted = Vec::new();
        for (candidate, score) in input.candidates.into_iter().zip(input.candidate_scores.iter()) {
            // Suppress already-loaded files
            if loaded_ids.contains(&candidate.stub_id) {
                continue;
            }
            
            // Use load threshold for new admissions
            if *score >= self.load_threshold && used + candidate.tokens <= budget {
                used += candidate.tokens;
                self.loaded_scores.borrow_mut().insert(candidate.stub_id.clone(), *score);
                admitted.push(candidate.clone());
                selected.push(candidate);
            }
        }

        let selected_ids = selected
            .iter()
            .map(|f| f.stub_id.clone())
            .collect::<Vec<_>>();

        let evicted = originally_loaded
            .iter()
            .filter(|frag| !selected_ids.iter().any(|id| id == &frag.stub_id))
            .cloned()
            .collect::<Vec<_>>();
        
        // Remove evicted from tracking
        for evicted_frag in &evicted {
            self.loaded_scores.borrow_mut().remove(&evicted_frag.stub_id);
        }

        SchedulerDecision {
            keep: selected,
            admitted,
            evicted,
        }
    }
}
