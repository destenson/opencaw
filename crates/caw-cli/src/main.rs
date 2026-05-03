use anyhow::{Context, Result};
use caw_adapters::MockAdapter;
use caw_core::{
    CompletionRequest, EmbeddingProvider, Locator, ModelAdapter, RecallFragment, Retriever,
    ScoredStub, StubId, StubStore, Tokenizer, VectorIndex, WhitespaceTokenizer,
    candidate_list_fragment, count_tokens_cl100k,
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
use caw_orchestrator::dynamic::DynamicRecallConfig;
use caw_orchestrator::session::SessionFile;
use clap::Parser;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

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

    // Load (path, mtime) pairs already in the store so ingest_directory can
    // skip reading unchanged files. A stat syscall per file is much cheaper
    // than reading and hashing it.
    let already_indexed: DocumentIdSet = store
        .indexed_paths()
        .unwrap_or_default()
        .into_iter()
        .collect();

    let (documents, skipped) = pipeline
        .ingest_directory(&cli.dir, !cli.gitignore, &already_indexed)
        .context("Failed to ingest directory")?;

    // All files returned from ingest_directory are new or changed — unchanged
    // files were skipped by the mtime check above.
    let ingested = documents.len();
    for (stub, _embed_text) in documents {
        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
        let embeddings = embedder
            .embed_document(vec![text.as_str()])
            .context("Failed to generate embedding")?;
        if let Some(embedding) = embeddings.into_iter().next() {
            store
                .insert(stub, embedding)
                .context("Failed to insert into stub store")?;
        }
    }

    // Rebuild the HNSW index from all stored embeddings (cached + newly ingested).
    let all_emb = store
        .all_embeddings()
        .context("Failed to load embeddings")?;
    let total_indexed = all_emb.len();
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

    // Check system prompt budget
    let budget = caw_curation::SystemPromptBudget::from_context_window(cli.max_tokens, 0.10);
    let budget_check = caw_curation::check_system_prompt(&cli.system, &budget);
    if budget_check.is_exceeded() {
        eprintln!("WARNING: system prompt exceeds 10% of context budget");
    } else if budget_check.is_warning() {
        eprintln!("NOTE: system prompt is approaching budget limit");
    }

    let retriever = SemanticRetriever::new(embedder, store, vector_index);

    let trace_embedder = LazyFastEmbedProvider::new();

    let config = DynamicRecallConfig {
        max_candidates: cli.max_candidates,
        max_workspace_tokens: cli.max_tokens,
        ..Default::default()
    };

    let adapter: Box<dyn ModelAdapter> =
        build_completion_adapter(&cli.adapter, cli.model.as_deref())?;

    eprintln!("Using adapter: {}", adapter.model_name());

    // LLM consolidation is available for when the DynamicRecallOrchestrator is
    // used directly (via with_consolidation_synthesizer). The manual recall loop
    // below doesn't evict, so it doesn't fire yet.
    let _consolidation: Option<LlmConsolidation> = if cli.llm_consolidation {
        eprintln!(
            "Using LLM consolidation ({}) for eviction notes",
            cli.aux_model
        );
        Some(LlmConsolidation::with_adapter(build_aux_adapter(
            &cli.aux_model,
        )))
    } else {
        None
    };

    if cli.curate {
        eprintln!("Curation pipeline enabled (history summarization + tool output compression)");
    }

    eprintln!("Enter queries (Ctrl+D to exit):\n");

    run_interactive(
        retriever,
        trace_embedder,
        trace_index,
        adapter,
        config,
        &cli.system,
        cli.curate,
        &cli.aux_model,
        cli.max_tokens,
        cli.session_dir.as_deref(),
        cli.amnesia,
        &already_indexed,
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
        "claude-code" => Box::new(caw_adapters::ClaudeCodeAdapter::sonnet()),
        "claude-code-haiku" => Box::new(caw_adapters::ClaudeCodeAdapter::haiku()),
        "ollama" => {
            let rt = caw_adapters::create_runtime()?;
            match model.unwrap_or("qwen3.6:35b") {
                "haiku" => Box::new(caw_adapters::OllamaAdapter::llama3_2(rt)),
                m => Box::new(caw_adapters::OllamaAdapter::local(m, rt)),
            }
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
                },
                rt,
            ))
        }
        other => {
            let rt = caw_adapters::create_runtime()?;
            let m = model.unwrap_or(other);
            Box::new(caw_adapters::OllamaAdapter::local(m, rt))
        }
    };
    Ok(adapter)
}

#[allow(clippy::too_many_arguments)]
fn run_interactive(
    mut retriever: SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    mut trace_embedder: LazyFastEmbedProvider,
    mut trace_index: HnswVectorIndex,
    adapter: Box<dyn ModelAdapter>,
    config: DynamicRecallConfig,
    system: &str,
    curate: bool,
    aux_model: &str,
    context_budget: usize,
    session_dir: Option<&std::path::Path>,
    amnesia: bool,
    already_indexed: &DocumentIdSet,
) -> Result<()> {
    use caw_transform::{extract_probes, extract_thinking_steps};
    use std::collections::{HashMap, HashSet};

    let mut loaded: Vec<RecallFragment> = Vec::new();
    let mut loaded_ids: HashSet<StubId> = HashSet::new();
    let mut relevance_scores: HashMap<StubId, f32> = HashMap::new();

    // Session history: in-memory content map + dedicated vector index.
    let mut session_index = HnswVectorIndex::new();
    let mut session_content: HashMap<StubId, String> = HashMap::new();
    let mut session_file: Option<SessionFile> = None;
    let mut session_turn = 0usize;

    if !amnesia && let Some(sdir) = session_dir {
        std::fs::create_dir_all(sdir).ok();
        let pipeline = caw_ingest::IngestionPipeline::new();
        let file_name = format!("session-{}.md", caw_orchestrator::session::timestamp_str());
        let current_path = sdir.join(&file_name);
        match SessionFile::load_previous(
            sdir,
            &current_path,
            &pipeline,
            &mut retriever,
            &already_indexed,
        ) {
            Ok(n) => eprintln!("[session] loaded {n} stubs from previous sessions"),
            Err(e) => eprintln!("[session] warning: {e}"),
        }
        match SessionFile::create(current_path) {
            Ok(sf) => {
                eprintln!("[session] recording to {}", sdir.display());
                session_file = Some(sf);
            }
            Err(e) => eprintln!("[session] warning: could not create session file: {e}"),
        }
    }
    let mut history: Vec<ConversationTurn> = Vec::new();

    // Set up curation components
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

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        debug!("> waiting for user input");
        print!("> ");
        stdout.flush()?;

        let mut query = String::new();
        if stdin.lock().read_line(&mut query)? == 0 {
            println!();
            break;
        }

        let query = query.trim();
        if query.is_empty() {
            continue;
        }

        // Decay relevance of every loaded fragment and evict those that have
        // fallen below the unload threshold. This prevents stale context from
        // prior turns from crowding out content relevant to the current query.
        let decay_rate = config.relevance_decay_rate;
        let unload_threshold = config.thresholds.unload;
        for score in relevance_scores.values_mut() {
            *score *= decay_rate;
        }
        let evicted: HashSet<StubId> = relevance_scores
            .iter()
            .filter(|(_, s)| **s < unload_threshold)
            .map(|(id, _)| id.clone())
            .collect();
        if !evicted.is_empty() {
            relevance_scores.retain(|id, _| !evicted.contains(id));
            loaded_ids.retain(|id| !evicted.contains(id));
            loaded.retain(|f| !evicted.contains(&f.stub_id));
        }

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

        // Run curation when history is long enough to benefit
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

        let hits = retriever.search(query, config.max_candidates)?;
        let above_threshold = hits
            .iter()
            .filter(|h| h.score >= config.thresholds.load)
            .count();
        // Count distinct file paths with any signal above the unload threshold.
        // When that count exceeds max_initial_fragments, the query matches too
        // many distinct sources to auto-load confidently — show a listing so
        // the model can choose. This catches cases where one chunk scores high
        // (and would be auto-loaded) but several other files are also relevant.
        let distinct_above_unload = hits
            .iter()
            .filter(|h| h.score >= config.thresholds.unload)
            .map(|h| h.stub.path.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let show_listing = above_threshold > config.max_initial_fragments
            || distinct_above_unload > config.max_initial_fragments;
        let candidate_fragment: Option<RecallFragment> = show_listing.then(|| {
            let list_threshold = if above_threshold > config.max_initial_fragments {
                config.thresholds.load
            } else {
                config.thresholds.unload
            };
            candidate_list_fragment(&hits, list_threshold)
        });
        if candidate_fragment.is_none() {
            load_fragments(
                &mut retriever,
                &hits,
                &mut loaded,
                &mut loaded_ids,
                &mut relevance_scores,
                &config,
            )?;
        } else {
            debug!(
                above_threshold,
                distinct_above_unload,
                max_initial = config.max_initial_fragments,
                "query matches multiple sources — surfacing candidate list"
            );
        }

        // Session history search runs alongside workspace search.
        if !session_content.is_empty() {
            if let Ok(embeddings) = trace_embedder.embed_query(vec![query]) {
                if let Some(emb) = embeddings.first() {
                    let session_hits = session_index.search(emb, config.max_candidates);
                    load_session_fragments(
                        session_hits,
                        &session_content,
                        &mut loaded,
                        &mut loaded_ids,
                        &mut relevance_scores,
                        &config,
                    );
                }
            }
        }

        let mut initial_fragments = loaded.clone();
        initial_fragments.extend(candidate_fragment);
        let mut response = adapter.complete(CompletionRequest {
            system: effective_system.clone(),
            user: query.to_string(),
            workspace_fragments: initial_fragments,
        })?;

        // When the candidate list was shown, the model's response may mention
        // specific files by path. Load those files and re-complete so the final
        // answer is grounded in actual content rather than just summaries.
        if above_threshold > config.max_initial_fragments {
            // Check both the answer and the thinking trace — thinking models
            // will mention files in the trace rather than (or in addition to)
            // the visible answer.
            let search_text = match &response.thinking {
                Some(t) => format!("{}\n{}", t, response.answer),
                None => response.answer.clone(),
            };
            let mut seen_paths = std::collections::HashSet::new();
            let mentioned: Vec<ScoredStub> = hits
                .iter()
                .filter(|h| h.score >= config.thresholds.load)
                .filter(|h| {
                    let norm = h.stub.path.trim_start_matches("./");
                    search_text.contains(norm) || search_text.contains(&h.stub.path)
                })
                .filter(|h| seen_paths.insert(h.stub.path.clone()))
                .cloned()
                .collect();
            if !mentioned.is_empty() {
                debug!(
                    count = mentioned.len(),
                    "loading files mentioned in response to candidate list"
                );
                load_fragments(
                    &mut retriever,
                    &mentioned,
                    &mut loaded,
                    &mut loaded_ids,
                    &mut relevance_scores,
                    &config,
                )?;
                response = adapter.complete(CompletionRequest {
                    system: effective_system,
                    user: query.to_string(),
                    workspace_fragments: loaded.clone(),
                })?;
            }
        }

        if config.enable_probe_recall {
            let probes = extract_probes(&response.answer);
            for probe in probes {
                let probe_hits = retriever.search(&probe.content, config.max_candidates)?;
                load_fragments(
                    &mut retriever,
                    &probe_hits,
                    &mut loaded,
                    &mut loaded_ids,
                    &mut relevance_scores,
                    &config,
                )?;
            }
        }

        if config.enable_thinking_trace_recall && adapter.capabilities().supports_visible_reasoning
        {
            let steps = extract_thinking_steps(&response.answer);
            for step in steps {
                if step.content.len() < 20 {
                    continue;
                }
                if let Ok(embeddings) = trace_embedder.embed_query(vec![&step.content])
                    && let Some(emb) = embeddings.first()
                {
                    let index_hits = trace_index.search(emb, config.max_candidates);
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

        println!("\n{}\n", response.answer);

        // Record turn: write to disk for persistence, embed for future recall.
        if let Some(ref mut sf) = session_file {
            session_turn += 1;
            if let Ok(text) = sf.write_turn(session_turn, query, &response.answer) {
                let stub_id = StubId(format!("session-turn-{session_turn}"));
                if let Ok(embeddings) = trace_embedder.embed_document(vec![text.as_str()]) {
                    if let Some(emb) = embeddings.into_iter().next() {
                        session_index.add(stub_id.clone(), emb);
                        session_content.insert(stub_id, text);
                    }
                }
            }
        }

        history.push(ConversationTurn {
            role: TurnRole::Assistant,
            content: response.answer.clone(),
            token_estimate: caw_curation::estimate_tokens(&response.answer),
            timestamp_secs: timestamp,
            metadata: TurnMetadata::default(),
        });

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

fn load_session_fragments(
    hits: Vec<(StubId, f32)>,
    session_content: &std::collections::HashMap<StubId, String>,
    loaded: &mut Vec<RecallFragment>,
    loaded_ids: &mut std::collections::HashSet<StubId>,
    relevance_scores: &mut std::collections::HashMap<StubId, f32>,
    config: &DynamicRecallConfig,
) {
    for (stub_id, score) in hits {
        if loaded_ids.contains(&stub_id) {
            let entry = relevance_scores.entry(stub_id).or_insert(0.0);
            *entry = entry.max(score);
            continue;
        }
        if score < config.thresholds.load {
            continue;
        }
        let current_tokens: usize = loaded.iter().map(|f| f.tokens).sum();
        if current_tokens >= config.max_workspace_tokens {
            break;
        }
        if let Some(content) = session_content.get(&stub_id) {
            let tokens = count_tokens_cl100k(content);
            if current_tokens + tokens <= config.max_workspace_tokens {
                loaded_ids.insert(stub_id.clone());
                relevance_scores.insert(stub_id.clone(), score);
                loaded.push(RecallFragment {
                    stub_id,
                    content: content.clone(),
                    locator: Locator {
                        source: "session history".to_string(),
                        locator: "full".to_string(),
                    },
                    tokens,
                });
            }
        }
    }
}

fn load_fragments(
    retriever: &mut SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    hits: &[caw_core::ScoredStub],
    loaded: &mut Vec<caw_core::RecallFragment>,
    loaded_ids: &mut std::collections::HashSet<caw_core::StubId>,
    relevance_scores: &mut std::collections::HashMap<caw_core::StubId, f32>,
    config: &DynamicRecallConfig,
) -> Result<()> {
    for hit in hits {
        if loaded_ids.contains(&hit.stub.id) {
            // Already in workspace — refresh its score so it isn't evicted
            // prematurely when it's still relevant to the current query.
            let entry = relevance_scores.entry(hit.stub.id.clone()).or_insert(0.0);
            *entry = entry.max(hit.score);
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
            relevance_scores.insert(hit.stub.id.clone(), hit.score);
            loaded.push(fragment);
        }
    }
    Ok(())
}
