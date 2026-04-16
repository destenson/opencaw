use anyhow::Result;
use caw_adapters::MockAdapter;
use caw_core::{ContentKind, Stub, StubId};
use caw_index::InMemoryIndex;
use caw_orchestrator::{OrchestratorConfig, RecallOrchestrator};
use caw_provenance::InMemoryProvenanceStore;
use caw_scheduler::GreedyBudgetScheduler;

fn main() -> Result<()> {
    let mut index = InMemoryIndex::default();
    index.insert(
        Stub {
            id: StubId("context-as-workspace.md".to_string()),
            path: "context-as-workspace.md".to_string(),
            token_estimate: 2200,
            kind: ContentKind::Markdown,
            summary: "Design notes for context-as-workspace architecture".to_string(),
            outline: vec![
                "Thesis".to_string(),
                "Stub-and-Recall".to_string(),
                "Curation".to_string(),
            ],
            content_hash: "dev-hash".to_string(),
            mtime_unix_secs: 0,
        },
        "Context as workspace emphasizes managed working sets and provenance-aware recall."
            .to_string(),
    );

    let mut orchestrator = RecallOrchestrator {
        retriever: index,
        scheduler: GreedyBudgetScheduler,
        provenance: InMemoryProvenanceStore::default(),
        adapter: MockAdapter::new("mock-local-model", true),
        loaded: Vec::new(),
        config: OrchestratorConfig::default(),
    };

    let response = orchestrator.run_turn(
        "You are a helpful assistant.",
        "How should I build context scheduling for local and API models?",
    )?;

    println!("{}", response.answer);
    Ok(())
}
