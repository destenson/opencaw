use anyhow::{Context, Result};
use caw_adapters::MockAdapter;
use caw_core::{
    CompletionRequest, EmbeddingProvider, ModelAdapter, Retriever,
    StubStore, VectorIndex,
};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::IngestionPipeline;
use caw_orchestrator::dynamic::DynamicRecallConfig;
use clap::Parser;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "caw", about = "Context-as-Workspace: on-demand recall for LLMs")]
struct Cli {
    /// Directory to ingest
    #[arg(short, long)]
    dir: PathBuf,

    /// Model adapter to use
    #[arg(short, long, default_value = "mock")]
    adapter: String,

    /// SQLite database path for persistent index (omit for in-memory)
    #[arg(long)]
    db: Option<PathBuf>,

    /// Top-k results for recall
    #[arg(long, default_value = "4")]
    top_k: usize,

    /// Max workspace tokens
    #[arg(long, default_value = "12000")]
    max_tokens: usize,

    /// System prompt
    #[arg(long, default_value = "You are a helpful assistant with access to recalled documents. Use the recalled context to answer questions accurately.")]
    system: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!("Initializing embedding provider...");
    let mut embedder = FastEmbedProvider::bge_small()
        .context("Failed to initialize embedding provider")?;
    let dimension = embedder.dimension();

    let db_path = cli
        .db
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ":memory:".to_string());

    eprintln!("Opening stub store at: {}", db_path);
    let mut store = SqliteStubStore::new(&db_path, dimension)
        .context("Failed to open SQLite stub store")?;

    let pipeline = IngestionPipeline;
    eprintln!("Ingesting files from: {}", cli.dir.display());

    let documents = pipeline
        .ingest_directory(&cli.dir)
        .context("Failed to ingest directory")?;

    let mut vector_index = HnswVectorIndex::new();
    let mut cached = 0usize;
    let mut ingested = 0usize;

    for (stub, content) in &documents {
        // Check if we already have this exact content indexed
        if let Ok(Some((existing_stub, existing_embedding))) =
            store.get_by_content_hash(&stub.content_hash)
        {
            vector_index.add(existing_stub.id.clone(), existing_embedding);
            cached += 1;
            continue;
        }

        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
        let embeddings = embedder
            .embed_document(vec![text.as_str()])
            .context("Failed to generate embedding")?;

        if let Some(embedding) = embeddings.into_iter().next() {
            store
                .insert(stub.clone(), embedding.clone(), content.clone())
                .context("Failed to insert into stub store")?;
            vector_index.add(stub.id.clone(), embedding);
            ingested += 1;
        }
    }

    eprintln!(
        "Index ready: {} files ({} cached, {} newly ingested).",
        documents.len(), cached, ingested
    );

    let retriever = SemanticRetriever::new(embedder, store, vector_index);

    // Second embedder + separate HNSW index for thinking-trace recall
    let trace_embedder = FastEmbedProvider::bge_small()
        .context("Failed to initialize trace embedder")?;

    // Rebuild a second HNSW index from the same store's embeddings
    let trace_store = SqliteStubStore::new(&db_path, dimension)
        .context("Failed to open trace stub store")?;
    let mut trace_index = HnswVectorIndex::new();
    if let Ok(all_emb) = trace_store.all_embeddings() {
        for (id, emb) in all_emb {
            trace_index.add(id, emb);
        }
    }

    let config = DynamicRecallConfig {
        top_k: cli.top_k,
        max_workspace_tokens: cli.max_tokens,
        ..Default::default()
    };

    let adapter: Box<dyn ModelAdapter> = match cli.adapter.as_str() {
        "mock" => Box::new(MockAdapter::new("mock-local", true)),
        "anthropic" | "claude" => {
            let rt = caw_adapters::create_runtime()?;
            Box::new(
                caw_adapters::AnthropicAdapter::claude_sonnet(rt)
                    .context("Failed to create Anthropic adapter")?,
            )
        }
        "groq" => {
            let rt = caw_adapters::create_runtime()?;
            Box::new(
                caw_adapters::GroqAdapter::llama_70b(rt)
                    .context("Failed to create Groq adapter")?,
            )
        }
        "ollama" => {
            let rt = caw_adapters::create_runtime()?;
            Box::new(caw_adapters::OllamaAdapter::llama3_2(rt))
        }
        "perplexity" => {
            let rt = caw_adapters::create_runtime()?;
            Box::new(
                caw_adapters::OpenAiCompatibleAdapter::perplexity(rt)
                    .context("Failed to create Perplexity adapter")?,
            )
        }
        // vllm://model-name or vllm://host:port/model-name
        s if s.starts_with("vllm://") => {
            let rt = caw_adapters::create_runtime()?;
            let rest = &s["vllm://".len()..];
            let (base_url, model) = if let Some(slash_pos) = rest.rfind('/') {
                let host = &rest[..slash_pos];
                let model = &rest[slash_pos + 1..];
                (format!("http://{}", host), model.to_string())
            } else {
                ("http://localhost:8000".to_string(), rest.to_string())
            };
            Box::new(caw_adapters::OpenAiCompatibleAdapter::vllm_at(base_url, model, rt))
        }
        // Generic openai-compatible: openai://base-url/model-name
        s if s.starts_with("openai://") => {
            let rt = caw_adapters::create_runtime()?;
            let rest = &s["openai://".len()..];
            let slash_pos = rest.rfind('/')
                .context("openai:// format requires: openai://host:port/model-name")?;
            let host = &rest[..slash_pos];
            let model = &rest[slash_pos + 1..];
            let base_url = if host.starts_with("http") {
                host.to_string()
            } else {
                format!("http://{}", host)
            };
            let headers = match std::env::var("OPENAI_COMPATIBLE_API_KEY") {
                Ok(key) => caw_adapters::RequestHeaders::bearer(key),
                Err(_) => caw_adapters::RequestHeaders::new(),
            };
            Box::new(caw_adapters::OpenAiCompatibleAdapter::new_with(
                base_url,
                model,
                headers,
                caw_core::ModelCapabilities {
                    supports_tool_calls: true,
                    supports_hidden_reasoning: false,
                    supports_visible_reasoning: false,
                },
                rt,
            ))
        }
        other => {
            let rt = caw_adapters::create_runtime()?;
            Box::new(caw_adapters::OllamaAdapter::local(other, rt))
        }
    };

    eprintln!("Using adapter: {}", adapter.model_name());
    eprintln!("Enter queries (Ctrl+D to exit):\n");

    run_interactive(retriever, trace_embedder, trace_index, adapter, config, &cli.system)
}

fn run_interactive(
    mut retriever: SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    mut trace_embedder: FastEmbedProvider,
    mut trace_index: HnswVectorIndex,
    adapter: Box<dyn ModelAdapter>,
    config: DynamicRecallConfig,
    system: &str,
) -> Result<()> {
    use caw_core::{RecallFragment, StubId};
    use caw_transform::{extract_probes, extract_thinking_steps};
    use std::collections::HashSet;

    let mut loaded: Vec<RecallFragment> = Vec::new();
    let mut loaded_ids: HashSet<StubId> = HashSet::new();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut query = String::new();
        if stdin.lock().read_line(&mut query)? == 0 {
            break;
        }

        let query = query.trim();
        if query.is_empty() {
            continue;
        }

        let hits = retriever.search(query, config.top_k)?;
        load_fragments(&mut retriever, &hits, &mut loaded, &mut loaded_ids, &config)?;

        let response = adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: query.to_string(),
            workspace_fragments: loaded.clone(),
        })?;

        if config.enable_probe_recall {
            let probes = extract_probes(&response.answer);
            for probe in probes {
                let probe_hits = retriever.search(&probe.content, config.top_k)?;
                load_fragments(
                    &mut retriever,
                    &probe_hits,
                    &mut loaded,
                    &mut loaded_ids,
                    &config,
                )?;
            }
        }

        if config.enable_thinking_trace_recall
            && adapter.capabilities().supports_visible_reasoning
        {
            let steps = extract_thinking_steps(&response.answer);
            for step in steps {
                if step.content.len() < 20 {
                    continue;
                }
                if let Ok(embeddings) = trace_embedder.embed_query(vec![&step.content]) {
                    if let Some(emb) = embeddings.first() {
                        let index_hits = trace_index.search(emb, config.top_k);
                        for (stub_id, score) in index_hits {
                            if loaded_ids.contains(&stub_id) || score < config.thresholds.load {
                                continue;
                            }
                            if let Ok(fragment) = retriever.read_range(&stub_id, "full") {
                                let current_tokens: usize = loaded.iter().map(|f| f.tokens).sum();
                                if current_tokens + fragment.tokens <= config.max_workspace_tokens {
                                    loaded_ids.insert(stub_id);
                                    loaded.push(fragment);
                                }
                            }
                        }
                    }
                }
            }
        }

        println!("\n{}\n", response.answer);

        if !loaded.is_empty() {
            eprintln!(
                "[workspace: {} fragments, ~{} tokens]",
                loaded.len(),
                loaded.iter().map(|f| f.tokens).sum::<usize>()
            );
        }
    }

    Ok(())
}

fn load_fragments(
    retriever: &mut SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    hits: &[caw_core::ScoredStub],
    loaded: &mut Vec<caw_core::RecallFragment>,
    loaded_ids: &mut std::collections::HashSet<caw_core::StubId>,
    config: &DynamicRecallConfig,
) -> Result<()> {
    for hit in hits {
        if loaded_ids.contains(&hit.stub.id) {
            continue;
        }
        if hit.score < config.thresholds.load {
            continue;
        }
        let current_tokens: usize = loaded.iter().map(|f| f.tokens).sum();
        if current_tokens >= config.max_workspace_tokens {
            break;
        }

        let fragment = retriever.read_range(&hit.stub.id, "full")?;
        if current_tokens + fragment.tokens <= config.max_workspace_tokens {
            loaded_ids.insert(hit.stub.id.clone());
            loaded.push(fragment);
        }
    }
    Ok(())
}
