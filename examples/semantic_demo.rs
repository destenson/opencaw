use anyhow::Result;
use caw_adapters::AnthropicAdapter;
use caw_core::{ContentKind, Stub, StubId, TokenBudget};
use caw_index::{FastEmbedProvider, SemanticRetriever, SqliteVectorStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::{OrchestratorConfig, RecallOrchestrator};
use caw_provenance::InMemoryProvenanceStore;
use caw_scheduler::GreedyBudgetScheduler;

fn main() -> Result<()> {
    println!("OpenCAW Semantic Retrieval Demo\n");

    // Set up semantic retrieval with FastEmbed + SQLite
    println!("Initializing semantic index...");
    let embedder = FastEmbedProvider::bge_small()?;
    let store = SqliteVectorStore::in_memory(embedder.dimension())?;
    let mut retriever = SemanticRetriever::new(embedder, store);

    // Ingest sample documents
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
        },
        SourceDocument {
            path: "rust-embedding-guide.md".to_string(),
            content: "Rust Embedding Guide: Using fastembed, ONNX, and Candle for semantic search. \
                     FastEmbed provides quantized models like BGE and E5 in a single crate. \
                     For vector storage, SQLite works well for MVP, while Qdrant offers production-ready \
                     HNSW with payload support."
                .to_string(),
            kind: ContentKind::Markdown,
        },
        SourceDocument {
            path: "model-adapters.md".to_string(),
            content: "Model Adapters: OpenCAW supports multiple LLM providers through a unified trait. \
                     Anthropic (Claude Sonnet/Opus) for high-quality reasoning with extended thinking. \
                     Groq for fast inference with Llama and Mixtral models. \
                     Ollama for local deployment with DeepSeek R1 and other open models."
                .to_string(),
            kind: ContentKind::Markdown,
        },
    ];

    for doc in docs {
        let stub = pipeline.ingest(doc.clone());
        retriever.insert(stub, doc.content)?;
    }

    println!("Indexed {} documents\n", 3);

    // Set up orchestrator with semantic retrieval
    let config = OrchestratorConfig {
        top_k: 2,
        load_threshold: 0.2,
        budget: TokenBudget {
            max_total: 16_000,
            reserved_for_prompt: 2_000,
            reserved_for_answer: 2_000,
        },
        ..Default::default()
    };

    let mut orchestrator = RecallOrchestrator {
        retriever,
        scheduler: GreedyBudgetScheduler,
        provenance: InMemoryProvenanceStore::default(),
        adapter: AnthropicAdapter::claude_sonnet(),
        loaded: Vec::new(),
        config,
    };

    // Run queries with semantic recall
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
