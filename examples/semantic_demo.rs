use anyhow::Result;
use caw_adapters::MockAdapter;
use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::scheduler::GreedyBudgetScheduler;
use caw_core::{ContentKind, EmbeddingProvider, RecallThresholds, TokenBudget};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::{OrchestratorConfig, RecallOrchestrator};

// TODO: allow the user to supply a document or path to query about, instead of hardcoding demo documents and queries.

fn main() -> Result<()> {
    println!("OpenCAW Semantic Retrieval Demo\n");

    println!("Initializing semantic index...");
    let embedder = FastEmbedProvider::bge_small()?;
    let dimension = embedder.dimension();
    // Write fixture docs to a tempdir so the store's on-disk content path
    // resolves; the store reads body text back from disk via
    // (path, byte_offset, byte_length) rather than persisting the content.
    let corpus_root = tempfile::tempdir()?;
    let store =
        SqliteStubStore::in_memory(dimension)?.with_corpus_root(corpus_root.path().to_path_buf());
    let index = HnswVectorIndex::new();
    let mut retriever = SemanticRetriever::new(embedder, store, index);

    println!("Ingesting documents...");
    let pipeline = IngestionPipeline::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

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
            mtime_unix_secs: now,
        },
        SourceDocument {
            path: "rust-embedding-guide.md".to_string(),
            content: "Rust Embedding Guide: Using fastembed, ONNX, and Candle for semantic search. \
                     FastEmbed provides quantized models like BGE and E5 in a single crate. \
                     For vector storage, SQLite works well for persistence, while HNSW provides \
                     fast approximate nearest neighbor search."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: now,
        },
        SourceDocument {
            path: "model-adapters.md".to_string(),
            content: "Model Adapters: OpenCAW supports multiple LLM providers through a unified trait. \
                     Anthropic (Claude Sonnet/Opus) for high-quality reasoning with extended thinking. \
                     Groq for fast inference with Llama and Mixtral models. \
                     Ollama for local deployment with DeepSeek R1 and other open models."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: now,
        },
    ];

    for doc in docs {
        let full = corpus_root.path().join(&doc.path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, &doc.content)?;
        let stubs = pipeline.ingest(doc);
        for (stub, _embed_text) in stubs {
            retriever.insert(stub)?;
        }
    }

    println!("Indexed 3 documents\n");

    let config = OrchestratorConfig {
        max_candidates: 4,
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

        let response = orchestrator.run_turn("", query)?;

        println!("{}\n", response.answer);
    }

    Ok(())
}
