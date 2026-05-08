use anyhow::{Context, Result};
use caw_adapters::MockAdapter;
use caw_core::{
    CompletionRequest, EmbeddingProvider, ModelAdapter, QueryIntent, StubStore, VectorIndex,
    WhitespaceTokenizer, Tokenizer, provenance::InMemoryProvenanceStore,
};
use caw_curation::{
    ConversationTurn, CurationPipelineBuilder, ExtractiveHistorySummarizer,
    ExtractiveToolOutputCompressor, HistorySummarizerConfig, LlmHistorySummarizer,
    LlmToolOutputCompressor, ToolOutputCompressorConfig, TurnMetadata, TurnRole,
};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::summarizer::LlmSummarizer;
use caw_ingest::{DocumentIdSet, IngestionPipeline};
use caw_orchestrator::consolidation::LlmConsolidation;
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};
use clap::Parser;
use rustyline::{error::ReadlineError, DefaultEditor};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

/// Concrete orchestrator type used by the CLI.
type CliOrchestrator = DynamicRecallOrchestrator<
    SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    LazyFastEmbedProvider,
    HnswVectorIndex,
    InMemoryProvenanceStore,
    Arc<dyn ModelAdapter>,
    SqliteStubStore,
>;

/// Wraps FastEmbedProvider to defer model loading until the first embed call.
/// Startup only loads the ONNX model when there are actually new files to embed
/// or the first query arrives — making warm starts (everything cached) near-instant.
struct LazyFastEmbedProvider {
    inner: Option<FastEmbedProvider>,
}

impl LazyFastEmbedProvider {
    fn new() -> Self {
        Self { inner: None }
    }

    fn ensure_loaded(&mut self) -> caw_core::CawResult<&mut FastEmbedProvider> {
        if self.inner.is_none() {
            eprintln!("Loading embedding model...");
            self.inner = Some(FastEmbedProvider::bge_small()?);
        }
        Ok(self.inner.as_mut().unwrap())
    }
}

impl EmbeddingProvider for LazyFastEmbedProvider {
    fn embed(&mut self, texts: Vec<&str>) -> caw_core::CawResult<Vec<Vec<f32>>> {
        self.ensure_loaded()?.embed(texts)
    }
    fn embed_query(&mut self, texts: Vec<&str>) -> caw_core::CawResult<Vec<Vec<f32>>> {
        self.ensure_loaded()?.embed_query(texts)
    }
    fn embed_document(&mut self, texts: Vec<&str>) -> caw_core::CawResult<Vec<Vec<f32>>> {
        self.ensure_loaded()?.embed_document(texts)
    }
    fn dimension(&self) -> usize {
        384 // BGE-small-en-v1.5 is always 384-dimensional
    }
    fn provider_name(&self) -> &str {
        "fastembed-bge-small"
    }
}

#[derive(Parser)]
#[command(
    name = "caw",
    about = "Context-as-Workspace: on-demand recall for LLMs"
)]
struct Cli {
    /// Directory to ingest
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Model adapter to use for completions
    #[arg(short, long, default_value = "mock")]
    adapter: String,

    #[arg(short, long)]
    model: Option<String>,

    /// Adapter used to classify user queries before answer generation.
    #[arg(long, default_value = "ollama")]
    intent_adapter: String,

    /// Small model(s) used for query-intent classification. Repeat the flag to
    /// enable an ensemble: each model is run independently and the results are
    /// merged by majority vote, filtering spurious false positives.
    #[arg(
        long,
        default_values = ["granite4:micro", "huihui_ai/jan-nano-abliterated:latest", "huihui_ai/deepseek-r1-abliterated:latest"],
        num_args = 1..
    )]
    intent_model: Vec<String>,

    /// Disable the query-intent classifier and skip intent-guidance injection.
    #[arg(long, default_value_t = false)]
    no_intent_classifier: bool,

    /// Minimum classifier confidence required before intent guidance is used.
    #[arg(long, default_value = "0.65")]
    intent_confidence: f32,

    /// Print classifier output for each query when intent classification is enabled.
    #[arg(long, default_value_t = false)]
    show_intent: bool,

    /// SQLite database path for persistent index. Defaults to .caw/index.db in the current directory.
    #[arg(long)]
    db: Option<PathBuf>,

    /// Candidate pool size for ANN search; the load threshold controls actual admissions.
    #[arg(long, default_value = "20")]
    max_candidates: usize,

    /// Max workspace tokens
    #[arg(long, default_value = "12000")]
    max_tokens: usize,

    /// System prompt
    #[arg(
        long,
        default_value = "With access to recalled documents, use the recalled context to answer questions accurately."
    )]
    system: String,

    /// Tokenizer for token counting: cl100k (default), whitespace, p50k
    #[arg(long, default_value = "cl100k")]
    tokenizer: String,

    /// Use LLM to generate stub summaries during ingestion (uses aux-model)
    #[arg(long)]
    llm_summarize: bool,

    /// Use LLM to synthesize consolidation notes on eviction (uses aux-model)
    #[arg(long)]
    llm_consolidation: bool,

    /// Model for auxiliary LLM tasks (summarization, consolidation). Default: haiku
    #[arg(long, default_value = "haiku")]
    aux_model: String,

    /// Enable curation pipeline (history summarization + tool output compression)
    #[arg(long)]
    curate: bool,

    /// Directory for session history files. Each run appends to a new file;
    /// previous runs' files are indexed at startup for cross-session recall.
    #[arg(long, default_value = ".caw")]
    session_dir: Option<PathBuf>,

    /// Enable verbose logging for debugging and analysis
    /// This turns on debug-level logs that show internal operations like fragment loading,
    /// probe extraction, thinking trace processing, and curation decisions.
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Whether to respect .gitignore when ingesting files from the specified directory.
    /// By default, .gitignore is respected and ignored files are not ingested. Setting
    /// this flag to false will cause all files to be ingested regardless of .gitignore rules.
    #[arg(long, default_value_t = true)]
    gitignore: bool,

    #[arg(long, default_value_t = false)]
    amnesia: bool,

    /// Context window size in tokens for the llama adapter. Defaults to 8192.
    /// Note: Qwen3 models have a 128K training context; leaving this unset
    /// causes llama.cpp to pre-allocate a KV cache that will OOM on 16GB GPUs.
    #[arg(long)]
    num_ctx: Option<u32>,

    /// Sampling temperature. Lower = more deterministic; higher = more creative.
    /// Applies to the llama adapter; other adapters use their own defaults.
    #[arg(long)]
    temperature: Option<f32>,

    /// Number of model layers to offload to GPU. -1 = all layers (default).
    /// Pass a lower value if you need to split between GPU and CPU RAM.
    /// Llama adapter only.
    #[arg(long)]
    n_gpu_layers: Option<i32>,

    /// Maximum number of tokens to generate per response. Llama adapter only.
    #[arg(long)]
    max_new_tokens: Option<usize>,

    /// Print the full formatted prompt to stderr before each llama generation.
    /// Useful for inspecting exactly what context the model receives.
    #[arg(long, default_value_t = false)]
    show_prompt: bool,

    /// Save the full formatted prompt to a file before each llama generation.
    /// Useful for inspecting exactly what context the model receives.
    #[arg(long, default_value_t = false)]
    save_prompt: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let logstr = ["reqwest", "rustls", "globset", "h2", "hyper", "webpki"]
        .map(|s| format!("{s}=info"))
        .join(",");
    let logstr = if cli.verbose {
        format!("debug,{}", logstr)
    } else {
        logstr
    };
    match std::env::var("RUST_LOG") {
        Ok(existing) => unsafe { std::env::set_var("RUST_LOG", format!("{existing},{}", logstr)) },
        Err(_) => unsafe { std::env::set_var("RUST_LOG", logstr) },
    }

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    eprintln!("Loading embedding model...");
    let mut embedder = FastEmbedProvider::bge_small().map_err(|e| anyhow::anyhow!("{}", e))?;
    let dimension = embedder.dimension();

    let db_path = cli
        .db
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| {
            std::fs::create_dir_all(".caw").ok();
            ".caw/index.db".to_string()
        });

    eprintln!("Opening stub store at: {}", db_path);
    let mut store = SqliteStubStore::new(&db_path, dimension)
        .context("Failed to open SQLite stub store")?
        .with_corpus_root(cli.dir.clone());

    let tokenizer: Arc<dyn Tokenizer> = match cli.tokenizer.as_str() {
        "cl100k" => Arc::new(
            caw_core::tokenizer::TiktokenTokenizer::cl100k()
                .context("Failed to load cl100k_base tokenizer")?,
        ),
        "p50k" => {
            eprintln!("Using p50k_base tokenizer");
            Arc::new(
                caw_core::tokenizer::TiktokenTokenizer::p50k()
                    .context("Failed to load p50k_base tokenizer")?,
            )
        }
        _ => Arc::new(WhitespaceTokenizer),
    };

    let pipeline = if cli.llm_summarize {
        let aux_adapter = build_aux_adapter(&cli.aux_model);
        eprintln!(
            "Using LLM summarizer ({}) for stub generation",
            cli.aux_model
        );
        IngestionPipeline::with_summarizer(Box::new(LlmSummarizer::with_adapter(aux_adapter)))
            .with_tokenizer(tokenizer)
    } else {
        IngestionPipeline::new().with_tokenizer(tokenizer)
    };

    eprintln!("Ingesting files from: {}", cli.dir.display());

    let already_indexed: DocumentIdSet = store
        .indexed_paths()
        .unwrap_or_default()
        .into_iter()
        .collect();

    let (documents, skipped) = pipeline
        .ingest_directory(&cli.dir, !cli.gitignore, &already_indexed)
        .context("Failed to ingest directory")?;

    // Chunks whose content is below these thresholds carry no retrieval signal.
    // token_estimate catches tiny raw files; summary_chars catches stubs where the
    // deterministic summarizer produced a near-empty string (e.g. Cargo.toml stubs
    // whose summary is "[package]" — 9 chars, well above MIN_INDEX_TOKENS but
    // useless as context). Both filters must pass to be indexed.
    const MIN_INDEX_TOKENS: usize = 10;
    const MIN_SUMMARY_CHARS: usize = 15;

    let ingested = documents.len();
    for (stub, embed_text) in documents {
        if stub.token_estimate < MIN_INDEX_TOKENS {
            continue;
        }
        if stub.summary.trim().len() < MIN_SUMMARY_CHARS {
            continue;
        }
        // embed_text is the actual chunk content (or full doc for small files).
        // Prepend the path so the embedder can orient to the source location;
        // the chunking pipeline budgets ~100 tokens of headroom for this prefix.
        // The old approach (path + summary + outline) repeated every symbol name
        // 2-3x and discarded the actual code content entirely.
        let text = format!("{}\n{}", stub.path, embed_text);
        let embeddings = embedder
            .embed_document(vec![text.as_str()])
            .context("Failed to generate embedding")?;
        if let Some(embedding) = embeddings.into_iter().next() {
            store
                .insert(stub, embedding)
                .context("Failed to insert into stub store")?;
        }
    }

    match store.apply_cawignore(&cli.dir) {
        Ok((0, 0)) => {}
        Ok((ignored, cleared)) => {
            if ignored > 0 { eprintln!("{ignored} stubs marked ignored from .cawignore."); }
            if cleared > 0 { eprintln!("{cleared} stubs un-ignored (.cawignore updated)."); }
        }
        Err(e) => eprintln!("Warning: failed to apply .cawignore: {e}"),
    }

    let all_emb = store
        .all_embeddings()
        .context("Failed to load embeddings")?;
    let total_indexed = all_emb.len();

    // vector_index goes into the retriever (workspace search).
    // trace_index goes into the orchestrator (thinking-trace and session recall).
    // Both are populated from the same stored embeddings so either index can
    // serve as a lookup for any stub in the corpus.
    let mut vector_index = HnswVectorIndex::new();
    let mut trace_index = HnswVectorIndex::new();
    for (id, emb) in all_emb {
        vector_index.add(id.clone(), emb.clone());
        trace_index.add(id, emb);
    }

    eprintln!(
        "Index ready: {} indexed ({} new, {} skipped).",
        total_indexed, ingested, skipped,
    );

    let budget = caw_curation::SystemPromptBudget::from_context_window(cli.max_tokens, 0.10);
    let budget_check = caw_curation::check_system_prompt(&cli.system, &budget);
    if budget_check.is_exceeded() {
        eprintln!("WARNING: system prompt exceeds 10% of context budget");
    } else if budget_check.is_warning() {
        eprintln!("NOTE: system prompt is approaching budget limit");
    }

    // Second connection to the same database, used by the orchestrator to
    // persist consolidation notes. The retriever already owns the first
    // connection; SQLite WAL mode allows multiple concurrent readers + one
    // writer safely. Without this, consolidation notes are written only to
    // in-memory provenance and are lost at session end.
    let consolidation_store = SqliteStubStore::new(&db_path, dimension)
        .context("Failed to open consolidation store")?;

    let retriever = SemanticRetriever::new(embedder, store, vector_index);
    let trace_embedder = LazyFastEmbedProvider::new();

    // When using the llama adapter, the total prompt (system + user +
    // workspace fragments) must fit within n_ctx. Reserve 3072 tokens for
    // the system prompt, user message, workspace format headers (not counted
    // in workspace_tokens), and generation headroom. Without this cap the
    // prefill will OOM the KV cache.
    let max_workspace_tokens = if cli.adapter == "llama" {
        let n_ctx = cli.num_ctx.unwrap_or(8192) as usize;
        let cap = n_ctx.saturating_sub(3072);
        if cli.max_tokens > cap {
            eprintln!(
                "NOTE: capping workspace tokens from {} to {} to fit within n_ctx={}",
                cli.max_tokens, cap, n_ctx
            );
        }
        cli.max_tokens.min(cap)
    } else {
        cli.max_tokens
    };

    let config = DynamicRecallConfig {
        max_candidates: cli.max_candidates,
        max_workspace_tokens,
        ..Default::default()
    };

    let raw_adapter = build_completion_adapter(
        &cli.adapter,
        cli.model.as_deref(),
        cli.num_ctx,
        cli.temperature,
        cli.n_gpu_layers,
        cli.max_new_tokens,
    )?;
    let adapter: Arc<dyn ModelAdapter> = if cli.show_prompt {
        let mut show = caw_adapters::ShowPromptAdapter::new(raw_adapter, cli.save_prompt);
        if let Some(sdir) = cli.session_dir.as_deref() {
            show = show.saving_to(sdir.to_path_buf());
        }
        Arc::new(show)
    } else if cli.save_prompt {
        let mut save = caw_adapters::SavePromptAdapter::new(raw_adapter);
        if let Some(sdir) = cli.session_dir.as_deref() {
            save = save.saving_to(sdir.to_path_buf());
        }
        Arc::new(save)
    } else {
        Arc::new(raw_adapter)
    };
    // When the main adapter is llama, reuse it for classification rather than
    // spinning up ollama. The Arc lets both the orchestrator and the classifier
    // share the already-loaded model without a second load.
    let intent_adapters: Vec<Arc<dyn ModelAdapter>> = if cli.no_intent_classifier {
        Vec::new()
    } else if cli.adapter == "llama" {
        vec![Arc::clone(&adapter)]
    } else {
        cli.intent_model
            .iter()
            .map(|model| {
                build_intent_adapter(&cli.intent_adapter, model)
                    .map(|a| Arc::new(a) as Arc<dyn ModelAdapter>)
            })
            .collect::<Result<Vec<_>>>()?
    };

    eprintln!("Using adapter: {}", adapter.model_name());
    match intent_adapters.len() {
        0 => {}
        1 => eprintln!("Using intent classifier: {}", intent_adapters[0].model_name()),
        n => eprintln!(
            "Using intent classifier ensemble ({n} models): {}",
            intent_adapters.iter().map(|a| a.model_name()).collect::<Vec<_>>().join(", ")
        ),
    }

    let provenance = InMemoryProvenanceStore::default();
    let mut orchestrator: CliOrchestrator = DynamicRecallOrchestrator::new(
        retriever,
        trace_embedder,
        trace_index,
        provenance,
        adapter,
        config,
    )
    .with_store(consolidation_store);

    if cli.llm_consolidation {
        eprintln!(
            "Using LLM consolidation ({}) for eviction notes",
            cli.aux_model
        );
        orchestrator = orchestrator.with_consolidation_synthesizer(Box::new(
            LlmConsolidation::with_adapter(build_aux_adapter(&cli.aux_model)),
        ));
    }

    if !cli.amnesia {
        if let Some(sdir) = cli.session_dir.as_deref() {
            orchestrator = orchestrator
                .with_session(sdir)
                .context("Failed to set up session history")?;
            eprintln!("[session] recording to {}", sdir.display());
        }
    }

    if cli.curate {
        eprintln!("Curation pipeline enabled (history summarization + tool output compression)");
    }

    let show_intent = cli.show_intent || cli.verbose;

    eprintln!("Enter queries (Ctrl+D to exit):\n");

    run_interactive(
        &mut orchestrator,
        intent_adapters,
        cli.intent_confidence,
        show_intent,
        &cli.system,
        cli.curate,
        &cli.aux_model,
        cli.max_tokens,
    )
}

fn build_aux_adapter(model: &str) -> Box<dyn ModelAdapter + Send + Sync> {
    match model {
        "sonnet" => Box::new(caw_adapters::ClaudeCodeAdapter::sonnet()),
        _ => Box::new(caw_adapters::ClaudeCodeAdapter::haiku()),
    }
}

fn build_completion_adapter(
    adapter_name: &str,
    model: Option<&str>,
    num_ctx: Option<u32>,
    temperature: Option<f32>,
    n_gpu_layers: Option<i32>,
    max_new_tokens: Option<usize>,
) -> Result<Box<dyn ModelAdapter>> {
    let adapter: Box<dyn ModelAdapter> = match adapter_name {
        "mock" => Box::new(MockAdapter::new("mock-local", true)),
        "anthropic" | "claude" => {
            let rt = caw_adapters::create_runtime()?;
            match model.unwrap_or("sonnet") {
                "opus" => Box::new(caw_adapters::AnthropicAdapter::claude_opus(rt)?),
                _ => Box::new(caw_adapters::AnthropicAdapter::claude_sonnet(rt)?),
            }
        }
        "groq" => {
            let rt = caw_adapters::create_runtime()?;
            match model.unwrap_or("llama-70b") {
                "llama-70b" => Box::new(caw_adapters::GroqAdapter::llama_70b(rt)?),
                m => Box::new(caw_adapters::GroqAdapter::groq_model(m, rt)?),
            }
        }
        #[cfg(feature = "llama")]
        "llama" => {
            let path = model.ok_or_else(|| {
                anyhow::anyhow!("--model <path.gguf> is required for the llama adapter")
            })?;
            let defaults = caw_adapters::LlamaCppConfig::default();
            Box::new(
                caw_adapters::LlamaCppAdapter::new_with(caw_adapters::LlamaCppConfig {
                    model_path: path.to_string(),
                    n_ctx: num_ctx.unwrap_or(defaults.n_ctx),
                    temperature: temperature.unwrap_or(defaults.temperature),
                    n_gpu_layers: n_gpu_layers.unwrap_or(defaults.n_gpu_layers),
                    max_new_tokens: max_new_tokens.unwrap_or(defaults.max_new_tokens),
                    ..defaults
                })
                .context("failed to load llama model")?,
            )
        }
        #[cfg(not(feature = "llama"))]
        "llama" => {
            anyhow::bail!("rebuild caw-cli with --features llama to use the llama adapter")
        }
        "claude-code" => Box::new(caw_adapters::ClaudeCodeAdapter::sonnet()),
        "claude-code-haiku" => Box::new(caw_adapters::ClaudeCodeAdapter::haiku()),
        "ollama" => {
            let rt = caw_adapters::create_runtime()?;
            let adapter = match model.unwrap_or("qwen3.6:35b") {
                "haiku" => caw_adapters::OllamaAdapter::llama3_2(rt),
                m => caw_adapters::OllamaAdapter::local(m, rt),
            };
            let adapter = if let Some(t) = temperature {
                adapter.with_temperature(t)
            } else {
                adapter
            };
            Box::new(adapter)
        }
        "perplexity" => {
            let rt = caw_adapters::create_runtime()?;
            match model.unwrap_or("hermes") {
                "hermes" => Box::new(
                    caw_adapters::OpenAiCompatibleAdapter::perplexity(rt)
                        .context("Failed to create Perplexity adapter")?,
                ),
                m => Box::new(
                    caw_adapters::OpenAiCompatibleAdapter::perplexity_model(m, rt)
                        .context("Failed to create Perplexity adapter with model")?,
                ),
            }
        }
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
            Box::new(caw_adapters::OpenAiCompatibleAdapter::vllm_at(
                base_url, model, rt,
            ))
        }
        s if s.starts_with("openai://") => {
            let rt = caw_adapters::create_runtime()?;
            let rest = &s["openai://".len()..];
            let slash_pos = rest
                .rfind('/')
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
                    ..Default::default()
                },
                rt,
            ))
        }
        other => {
            let rt = caw_adapters::create_runtime()?;
            let m = model.unwrap_or(other);
            let adapter = caw_adapters::OllamaAdapter::local(m, rt);
            let adapter = if let Some(t) = temperature {
                adapter.with_temperature(t)
            } else {
                adapter
            };
            Box::new(adapter)
        }
    };
    Ok(adapter)
}

fn build_intent_adapter(adapter_name: &str, model: &str) -> Result<Box<dyn ModelAdapter>> {
    // Temperature 0 for deterministic JSON output — stochastic sampling at the
    // classifier's default (~0.8) produces formatting variations that cause
    // parse failures and degenerate detection false positives.
    build_completion_adapter(adapter_name, Some(model), None, Some(0.0), None, None)
}

fn classify_query_intent(adapter: &dyn ModelAdapter, query: &str) -> Result<QueryIntent> {
    let response = adapter.complete(CompletionRequest {
        system: QueryIntent::augmentation_system_prompt().to_string(),
        user: QueryIntent::augmentation_user_prompt(query),
        ..Default::default()
    })?;

    QueryIntent::from_classifier_response(&response.answer)
        .map(|(intent, _keys)| intent)
        .map_err(|e| anyhow::anyhow!("intent classifier returned invalid output: {e}"))
}

#[allow(clippy::too_many_arguments)]
fn run_interactive(
    orchestrator: &mut CliOrchestrator,
    intent_adapters: Vec<Arc<dyn ModelAdapter>>,
    intent_confidence: f32,
    show_intent: bool,
    system: &str,
    curate: bool,
    aux_model: &str,
    context_budget: usize,
) -> Result<()> {
    let mut history: Vec<ConversationTurn> = Vec::new();

    let extractive_summarizer = ExtractiveHistorySummarizer;
    let hist_config = HistorySummarizerConfig::default();
    let llm_hist_summarizer;
    let aux_adapter_for_curation;

    let history_summarizer: &dyn caw_curation::HistorySummarizer = if curate {
        aux_adapter_for_curation = build_aux_adapter(aux_model);
        llm_hist_summarizer =
            LlmHistorySummarizer::new_with(aux_adapter_for_curation.as_ref(), hist_config.clone());
        &llm_hist_summarizer
    } else {
        &extractive_summarizer
    };

    let extractive_compressor =
        ExtractiveToolOutputCompressor::new_with(ToolOutputCompressorConfig::default());
    let llm_compressor;
    let aux_adapter_for_compressor;

    let tool_compressor: &dyn caw_curation::ToolOutputCompressor = if curate {
        aux_adapter_for_compressor = build_aux_adapter(aux_model);
        llm_compressor = LlmToolOutputCompressor::new_with(
            aux_adapter_for_compressor.as_ref(),
            ToolOutputCompressorConfig::default(),
        );
        &llm_compressor
    } else {
        &extractive_compressor
    };

    let mut rl = DefaultEditor::new()?;

    loop {
        debug!("> waiting for user input");
        let line = match rl.readline("> ") {
            Ok(l) => l,
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => {
                println!();
                break;
            }
            Err(e) => return Err(e.into()),
        };

        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(query);

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        history.push(ConversationTurn {
            role: TurnRole::User,
            content: query.to_string(),
            token_estimate: caw_curation::estimate_tokens(query),
            timestamp_secs: timestamp,
            metadata: TurnMetadata::default(),
        });

        let effective_system = if curate && history.len() > 4 {
            let pipeline = CurationPipelineBuilder::with_context_budget(context_budget)
                .history_summarizer(history_summarizer)
                .history_config(hist_config.clone())
                .tool_compressor(tool_compressor)
                .build()
                .context("Failed to build curation pipeline")?;

            match pipeline.curate(system, &history) {
                Ok(result) => {
                    if result.tokens_saved > 0 {
                        eprintln!("[curation] saved ~{} tokens", result.tokens_saved);
                    }
                    if !result.history_summary.is_empty() {
                        let mut new_history = vec![ConversationTurn {
                            role: TurnRole::System,
                            content: format!(
                                "Previous conversation summary:\n{}",
                                result.history_summary
                            ),
                            token_estimate: caw_curation::estimate_tokens(&result.history_summary),
                            timestamp_secs: timestamp,
                            metadata: TurnMetadata::default(),
                        }];
                        new_history.extend(result.retained_turns);
                        history = new_history;
                    }
                    result.system_prompt
                }
                Err(e) => {
                    eprintln!("[curation] failed, using raw prompt: {}", e);
                    system.to_string()
                }
            }
        } else {
            system.to_string()
        };

        let query_intent = if intent_adapters.is_empty() {
            None
        } else {
            let mut votes: Vec<(String, QueryIntent)> = Vec::new();
            let mut failed_classifiers: Vec<String> = Vec::new();
            for a in &intent_adapters {
                match classify_query_intent(a.as_ref(), query) {
                    Ok(v) => votes.push((a.model_name().to_string(), v)),
                    Err(e) => {
                        failed_classifiers.push(a.model_name().to_string());
                        eprintln!("[intent] classifier {} failed: {}", a.model_name(), e);
                    }
                }
            }
            if show_intent && votes.len() > 1 {
                for (name, v) in &votes {
                    eprintln!("[intent {name}] {}", v);
                }
            }
            if votes.is_empty() {
                // All classifiers failed — log the fallback so QA sessions can diagnose
                // whether the failure is systematic (e.g. all-degenerate, wrong temperature).
                eprintln!(
                    "[intent] all {} classifier(s) failed — falling back to no-intent (full retrieval, no guidance): {:?}",
                    failed_classifiers.len(),
                    failed_classifiers,
                );
                None
            } else {
                let intents: Vec<QueryIntent> = votes.into_iter().map(|(_, v)| v).collect();
                let merged = QueryIntent::majority_vote(&intents);
                if show_intent {
                    let label = if intent_adapters.len() > 1 { " merged" } else { "" };
                    eprintln!("[intent{label}] {}", merged);
                }
                Some(merged)
            }
        };
        let actionable = query_intent
            .as_ref()
            .filter(|intent| intent.is_actionable(intent_confidence));
        let guidance = actionable
            .map(QueryIntent::guidance_lines)
            .unwrap_or_default();
        let signals = actionable.map(|i| i.augmentation_signals());

        // TODO: This is *ABSOLUTELY WRONG* THE INTENT CLASSIFIER SHOULD BE A SIGNAL, NOT A GATING FACTOR.

        // Skip the retrieval cycle when the intent classifier confirms no substantive
        // query signal AND the message is short. Casual acknowledgments ("nice to know",
        // "got it", "ok") would otherwise surface lexically similar but off-topic fragments
        // and generate a misleading substantive response.
        if let Some(ref intent) = query_intent {
            if !intent.is_substantive() && query.split_whitespace().count() < 15 {
                if show_intent {
                    eprintln!("[intent] short non-substantive input — skipping retrieval");
                }
                // FIXME:  DO NOT HARD CODE RESPONSES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                println!("\nGot it.\n");
                continue;
            }
        }

        let response = match orchestrator.run_turn(&effective_system, query, &guidance, signals.as_ref()) {
            Ok(r) => r,
            Err(caw_core::CawError::DegenerateOutput { sample, .. }) => {
                // The session log already captured the turn (B15 fix). Skip to the
                // next query rather than aborting the entire session.
                eprintln!("[error] degenerate response — continuing to next query (sample: {}...)", &sample[..sample.len().min(60)]);
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        println!("\n{}\n", response.answer);

        history.push(ConversationTurn {
            role: TurnRole::Assistant,
            content: response.answer.clone(),
            token_estimate: caw_curation::estimate_tokens(&response.answer),
            timestamp_secs: timestamp,
            metadata: TurnMetadata::default(),
        });

        if !orchestrator.loaded.is_empty() {
            eprintln!(
                "[workspace: {} fragments, ~{} tokens]",
                orchestrator.loaded.len(),
                orchestrator.loaded.iter().map(|f| f.tokens).sum::<usize>()
            );
        }
    }

    Ok(())
}
