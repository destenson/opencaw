use caw_core::{BudgetScheduler, RecallFragment, SchedulerDecision, SchedulerInput};

#[derive(Debug, Clone, Copy, Default)]
pub struct GreedyBudgetScheduler;

impl BudgetScheduler for GreedyBudgetScheduler {
    fn schedule(&self, input: SchedulerInput) -> SchedulerDecision {
        let budget = input.budget.available_for_workspace();
        let originally_loaded = input.currently_loaded.clone();

        let mut selected = Vec::<RecallFragment>::new();
        let mut used = 0usize;

        for frag in input.currently_loaded {
            if used + frag.tokens <= budget {
                used += frag.tokens;
                selected.push(frag);
            }
        }

        let mut admitted = Vec::new();
        for candidate in input.candidates {
            if used + candidate.tokens <= budget {
                used += candidate.tokens;
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

        SchedulerDecision {
            keep: selected,
            admitted,
            evicted,
        }
    }
}
