use caw_core::{
    BudgetScheduler, CawResult, CompletionRequest, CompletionResponse, ModelAdapter,
    ProvenanceStore, RecallFragment, RecallThresholds, Retriever, SchedulerInput, TokenBudget,
};
use regex::Regex;
use std::sync::LazyLock;

static PROBE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<probe>(.*?)</probe>").unwrap());

/// Orchestrator that implements explicit probe-based recall.
/// Models emit <probe>query text</probe> markers to request content.
pub struct ProbeRecallOrchestrator<R, S, P, M>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub retriever: R,
    pub scheduler: S,
    pub provenance: P,
    pub adapter: M,
    pub loaded: Vec<RecallFragment>,
    pub config: ProbeRecallConfig,
}

#[derive(Debug, Clone)]
pub struct ProbeRecallConfig {
    pub top_k: usize,
    pub thresholds: RecallThresholds,
    pub default_range: String,
    pub budget: TokenBudget,
    pub max_probe_iterations: usize,
}

impl Default for ProbeRecallConfig {
    fn default() -> Self {
        Self {
            top_k: 3,
            thresholds: RecallThresholds::default_hysteresis(),
            default_range: "full".to_string(),
            budget: TokenBudget {
                max_total: 16_000,
                reserved_for_prompt: 2_000,
                reserved_for_answer: 2_000,
            },
            max_probe_iterations: 3,
        }
    }
}

impl<R, S, P, M> ProbeRecallOrchestrator<R, S, P, M>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub fn new(
        retriever: R,
        scheduler: S,
        provenance: P,
        adapter: M,
        config: ProbeRecallConfig,
    ) -> Self {
        Self {
            retriever,
            scheduler,
            provenance,
            adapter,
            loaded: Vec::new(),
            config,
        }
    }

    pub fn run_turn_with_probes(
        &mut self,
        system: &str,
        user: &str,
    ) -> CawResult<CompletionResponse> {
        let mut current_user = user.to_string();
        let mut iteration = 0;

        loop {
            let response = self.adapter.complete(CompletionRequest {
                system: system.to_string(),
                user: current_user.clone(),
                workspace_fragments: self.loaded.clone(),
            })?;

            let probes = self.extract_probes(&response.answer);

            if probes.is_empty() || iteration >= self.config.max_probe_iterations {
                return Ok(response);
            }

            let mut recalled_any = false;
            for probe_query in probes {
                let recalled = self.recall_for_probe(&probe_query)?;
                if !recalled.is_empty() {
                    recalled_any = true;
                }
            }

            if !recalled_any {
                return Ok(response);
            }

            current_user = format!(
                "{}\n\nAssistant (partial): {}\n\nUser: Content has been loaded. Please continue.",
                current_user, response.answer
            );

            iteration += 1;
        }
    }

    fn extract_probes(&self, text: &str) -> Vec<String> {
        PROBE_PATTERN
            .captures_iter(text)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    fn recall_for_probe(&mut self, probe_query: &str) -> CawResult<Vec<RecallFragment>> {
        let hits = self.retriever.search(probe_query, self.config.top_k)?;
        let mut candidates = Vec::new();
        let mut candidate_scores = Vec::new();

        for hit in hits {
            if hit.score >= self.config.thresholds.load {
                let fragment = self
                    .retriever
                    .read_range(&hit.stub.id, &self.config.default_range)?;
                candidate_scores.push(hit.score);
                candidates.push(fragment);
            }
        }

        let decision = self.scheduler.schedule(SchedulerInput {
            currently_loaded: self.loaded.clone(),
            candidates,
            candidate_scores,
            budget: self.config.budget,
        });

        self.loaded = decision.keep.clone();

        for frag in &decision.admitted {
            self.provenance.record(frag.clone());
        }

        Ok(decision.admitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caw_core::{
        CawError, CompletionResponse, ModelCapabilities, RecallFragment, SchedulerDecision,
        ScoredStub, StubId,
    };

    struct MockRetriever;
    impl Retriever for MockRetriever {
        fn search(&mut self, _q: &str, _k: usize) -> CawResult<Vec<ScoredStub>> {
            Ok(vec![])
        }
        fn read_range(&self, id: &StubId, _range: &str) -> CawResult<RecallFragment> {
            Err(CawError::NotFound(id.0.clone()))
        }
    }

    struct MockScheduler;
    impl BudgetScheduler for MockScheduler {
        fn schedule(&self, input: SchedulerInput) -> SchedulerDecision {
            SchedulerDecision {
                keep: input.currently_loaded,
                evicted: vec![],
                admitted: input.candidates,
            }
        }
    }

    struct MockProvenance;
    impl ProvenanceStore for MockProvenance {
        fn record(&mut self, _fragment: RecallFragment) {}
        fn all(&self) -> Vec<RecallFragment> {
            vec![]
        }
    }

    struct MockAdapter;
    impl ModelAdapter for MockAdapter {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                supports_tool_calls: false,
                supports_hidden_reasoning: false,
                supports_visible_reasoning: false,
            }
        }
        fn complete(&self, _req: CompletionRequest) -> CawResult<CompletionResponse> {
            Ok(CompletionResponse {
                answer: String::new(),
                thinking: None,
            })
        }
    }

    #[test]
    fn test_extract_probes() {
        let orchestrator = ProbeRecallOrchestrator {
            retriever: MockRetriever,
            scheduler: MockScheduler,
            provenance: MockProvenance,
            adapter: MockAdapter,
            loaded: Vec::new(),
            config: ProbeRecallConfig::default(),
        };

        let text = "I need to check <probe>configuration settings</probe> and also <probe>error logs</probe>";
        let probes = orchestrator.extract_probes(text);

        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0], "configuration settings");
        assert_eq!(probes[1], "error logs");
    }
}
