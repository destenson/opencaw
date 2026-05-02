use anyhow::Result;
use caw_adapters::GroqAdapter;
use caw_core::{ContentKind, EmbeddingProvider, ModelAdapter, RecallThresholds, TokenBudget};
use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::scheduler::GreedyBudgetScheduler;
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::{OrchestratorConfig, RecallOrchestrator};
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn ingest_file(
    pipeline: &IngestionPipeline,
    retriever: &mut SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    corpus_root: &Path,
    rel_path: &str,
    kind: ContentKind,
) -> Result<()> {
    let full_path = corpus_root.join(rel_path);
    let content = std::fs::read_to_string(&full_path)?;
    let mtime = std::fs::metadata(&full_path)?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let stubs = pipeline.ingest(SourceDocument {
        path: rel_path.to_string(),
        content,
        kind,
        mtime_unix_secs: mtime,
    });
    for (stub, _) in stubs {
        retriever.insert(stub)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let runtime = Arc::new(Runtime::new()?);
    let adapter = GroqAdapter::llama_70b(runtime)?;

    println!("OpenCAW Groq Demo\n");
    println!("Model: {}\n", adapter.model_name());

    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    println!("Initializing semantic index...");
    let embedder = FastEmbedProvider::bge_small()?;
    let dimension = embedder.dimension();
    let store = SqliteStubStore::in_memory(dimension)?
        .with_corpus_root(project_root.to_path_buf());
    let index = HnswVectorIndex::new();
    let mut retriever = SemanticRetriever::new(embedder, store, index);

    println!("Ingesting project docs...");
    let pipeline = IngestionPipeline::new();

    for (rel_path, kind) in &[
        ("context-as-workspace.md", ContentKind::Markdown),
        ("README.md", ContentKind::Markdown),
    ] {
        ingest_file(&pipeline, &mut retriever, project_root, rel_path, *kind)?;
        println!("  indexed {}", rel_path);
    }
    println!();

    let config = OrchestratorConfig {
        top_k: 3,
        thresholds: RecallThresholds::default_hysteresis(),
        budget: TokenBudget {
            max_total: 16_000,
            reserved_for_prompt: 2_000,
            reserved_for_answer: 2_000,
        },
        ..Default::default()
    };

    let mut orchestrator = RecallOrchestrator {
        retriever,
        scheduler: GreedyBudgetScheduler::default(),
        provenance: InMemoryProvenanceStore::default(),
        adapter,
        loaded: Vec::new(),
        config,
    };

    let system = "You are a helpful assistant with access to the OpenCAW project documentation. \
                  Answer questions accurately based on the recalled context provided.";

    let queries = [
        "What is the eviction policy and how does consolidation work?",
        "What embedding providers does OpenCAW support and what are their tradeoffs?",
        "How does the degradation model work when the embedding service is unavailable?",
    ];

    for query in &queries {
        println!("Query: {}", query);
        println!("{}", "-".repeat(60));
        let response = orchestrator.run_turn(system, query)?;
        println!("{}\n", response.answer);
    }

    Ok(())
}
