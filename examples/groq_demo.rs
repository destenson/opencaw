use anyhow::Result;
use caw_adapters::GroqAdapter;
use caw_core::{ContentKind, EmbeddingProvider, ModelAdapter, RecallThresholds, Retriever, TokenBudget};
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

    let system = "Answer questions about the OpenCAW project based on the documentation provided in context. \
                  Be precise and cite specific details from the recalled fragments.";

    let queries = [
        "What is the eviction policy and how does consolidation work?",
        "What embedding providers does OpenCAW support and what are their tradeoffs?",
        "How does the degradation model work when the embedding service is unavailable?",
    ];

    for query in &queries {
        orchestrator.loaded.clear();
        orchestrator.provenance = InMemoryProvenanceStore::default();

        // Search before run_turn to capture candidate stubs and their full token estimates.
        // run_turn does the same search internally; this doubles the embedding work but
        // gives us visibility into what the orchestrator is considering.
        let candidates = orchestrator.retriever.search(query, orchestrator.config.top_k)?;

        println!("Query: {}", query);
        println!("{}", "-".repeat(60));

        let response = orchestrator.run_turn(system, query)?;

        let initial_ids: std::collections::HashSet<&str> =
            candidates.iter().map(|h| h.stub.id.0.as_str()).collect();
        let total_recalled: usize = orchestrator.loaded.iter().map(|f| f.tokens).sum();
        let full_doc_tokens: usize = candidates.iter().map(|h| h.stub.token_estimate).sum();

        println!("\n[context]");
        for hit in &candidates {
            let admitted = orchestrator.loaded.iter().find(|f| f.stub_id.0 == hit.stub.id.0);
            match admitted {
                Some(frag) => println!(
                    "  LOADED  {} (score {:.2})  chunk {} tok / full {} tok",
                    hit.stub.path, hit.score, frag.tokens, hit.stub.token_estimate
                ),
                None => println!(
                    "  skipped {} (score {:.2})  full {} tok",
                    hit.stub.path, hit.score, hit.stub.token_estimate
                ),
            }
        }
        for frag in orchestrator.loaded.iter().filter(|f| !initial_ids.contains(f.stub_id.0.as_str())) {
            println!(
                "  THINK   {} (via thinking-trace)  chunk {} tok",
                frag.locator.source, frag.tokens
            );
        }
        println!(
            "  recalled {} tok — would have been {} tok for initial candidates ({:.0}% saving)",
            total_recalled,
            full_doc_tokens,
            if full_doc_tokens > 0 {
                (1.0 - total_recalled as f64 / full_doc_tokens as f64) * 100.0
            } else {
                0.0
            }
        );
        println!();

        if let Some(ref thinking) = response.thinking {
            println!("[thinking: {} words]\n", thinking.split_whitespace().count());
        }
        println!("{}\n", response.answer);
    }

    Ok(())
}
