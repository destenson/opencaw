use anyhow::Result;
use caw_adapters::MockAdapter;
use caw_core::{ContentKind, RecallThresholds, TokenBudget};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::{OrchestratorConfig, RecallOrchestrator};
use caw_provenance::InMemoryProvenanceStore;
use caw_scheduler::GreedyBudgetScheduler;

// TODO: allow the user to supply a document or path to query about, instead of hardcoding demo documents and queries.

fn main() -> Result<()> {

    println!("OpenCAW Semantic Retrieval Demo\n");

    println!("Initializing semantic index...");
    let embedder = FastEmbedProvider::bge_small()?;
    let dimension = embedder.dimension();
    let store = SqliteStubStore::in_memory(dimension)?;
    let index = HnswVectorIndex::new();
    let mut retriever = SemanticRetriever::new(embedder, store, index);

    println!("Ingesting documents...");
    let pipeline = IngestionPipeline;

    let docs = vec![
        SourceDocument {
            path: "context-as-workspace.md".to_string(),
            content: "Context as Workspace: On-Demand Recall and Curated Working Sets for LLMs. \
                     LLM context is a workspace, not a container. Effective capability is set by \
                     working-set quality, not total accessible information. The stub-and-recall \
                     architecture uses a prompt transformer to replace file references with structured \
                     stubs containing metadata for triage without reading the full content."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 0,
        },
        SourceDocument {
            path: "rust-embedding-guide.md".to_string(),
            content: "Rust Embedding Guide: Using fastembed, ONNX, and Candle for semantic search. \
                     FastEmbed provides quantized models like BGE and E5 in a single crate. \
                     For vector storage, SQLite works well for persistence, while HNSW provides \
                     fast approximate nearest neighbor search."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 0,
        },
        SourceDocument {
            path: "model-adapters.md".to_string(),
            content: "Model Adapters: OpenCAW supports multiple LLM providers through a unified trait. \
                     Anthropic (Claude Sonnet/Opus) for high-quality reasoning with extended thinking. \
                     Groq for fast inference with Llama and Mixtral models. \
                     Ollama for local deployment with DeepSeek R1 and other open models."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 0,
        },
    ];

    for doc in docs {
        let content = doc.content.clone();
        let stub = pipeline.ingest(doc);
        retriever.insert(stub, content)?;
    }

    println!("Indexed 3 documents\n");

    let config = OrchestratorConfig {
        top_k: 2,
        thresholds: RecallThresholds::permissive(),
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
        adapter: MockAdapter::new("mock-demo", true),
        loaded: Vec::new(),
        config,
    };

    let queries = vec![
        "How does the stub-and-recall architecture work?",
        "What embedding options are available in Rust?",
        "Which model providers are supported?",
    ];

    for query in queries {
        println!("Query: {}", query);
        println!("{}", "=".repeat(60));

        let response = orchestrator.run_turn("You are a helpful assistant.", query)?;

        println!("{}\n", response.answer);
    }

    Ok(())
}
