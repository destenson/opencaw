use anyhow::{Context, Result};
use caw_adapters::MockAdapter;
use caw_core::{
    candidate_list_fragment, CompletionRequest, EmbeddingProvider, Locator, ModelAdapter,
    RecallFragment, Retriever, ScoredStub, StubId, StubStore, Tokenizer, VectorIndex,
    WhitespaceTokenizer,
};
use caw_orchestrator::session::SessionFile;
use caw_curation::{
    ConversationTurn, CurationPipelineBuilder, ExtractiveHistorySummarizer,
    ExtractiveToolOutputCompressor, HistorySummarizerConfig, LlmHistorySummarizer,
    LlmToolOutputCompressor, ToolOutputCompressorConfig, TurnMetadata, TurnRole,
};
use caw_index::{FastEmbedProvider, HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::IngestionPipeline;
use caw_ingest::summarizer::LlmSummarizer;
use caw_orchestrator::consolidation::LlmConsolidation;
use caw_orchestrator::dynamic::DynamicRecallConfig;
use clap::Parser;
use tracing::debug;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

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

    /// SQLite database path for persistent index (omit for in-memory)
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

    /// Tokenizer for token counting: whitespace (default), cl100k, p50k
    #[arg(long, default_value = "whitespace")]
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

    #[arg(long, default_value_t = true)]
    gitignore: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let logstr = ["reqwest", "rustls", "globset", "h2", "hyper", "webpki"].map(|s| format!("{s}=info")).join(",");
    let logstr = if cli.verbose {
        format!("debug,{}", logstr)
    } else {
        logstr
    };
    match std::env::var("RUST_LOG") {
        Ok(existing) => unsafe {
            std::env::set_var("RUST_LOG", format!("{existing},{}", logstr))
        },
        Err(_) => unsafe {
            std::env::set_var("RUST_LOG", logstr)
        },
    }

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    eprintln!("Initializing embedding provider...");
    let mut embedder =
        FastEmbedProvider::bge_small().context("Failed to initialize embedding provider")?;
    let dimension = embedder.dimension();

    let db_path = cli
        .db
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ":memory:".to_string());

    eprintln!("Opening stub store at: {}", db_path);
    let mut store = SqliteStubStore::new(&db_path, dimension)
        .context("Failed to open SQLite stub store")?
        .with_corpus_root(cli.dir.clone());

    let tokenizer: Arc<dyn Tokenizer> = match cli.tokenizer.as_str() {
        "cl100k" => {
            eprintln!("Using cl100k_base tokenizer");
            Arc::new(
                caw_core::tokenizer::TiktokenTokenizer::cl100k()
                    .context("Failed to load cl100k_base tokenizer")?,
            )
        }
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

    let documents = pipeline
        .ingest_directory(&cli.dir, !cli.gitignore)
        .context("Failed to ingest directory")?;

    let mut vector_index = HnswVectorIndex::new();
    let mut cached = 0usize;
    let mut ingested = 0usize;

    for (stub, _embed_text) in &documents {
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
                .insert(stub.clone(), embedding.clone())
                .context("Failed to insert into stub store")?;
            vector_index.add(stub.id.clone(), embedding);
            ingested += 1;
        }
    }

    eprintln!(
        "Index ready: {} files ({} cached, {} newly ingested).",
        documents.len(),
        cached,
        ingested
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

    let trace_embedder =
        FastEmbedProvider::bge_small().context("Failed to initialize trace embedder")?;

    let trace_store = SqliteStubStore::new(&db_path, dimension)
        .context("Failed to open trace stub store")?
        .with_corpus_root(cli.dir.clone());
    let mut trace_index = HnswVectorIndex::new();
    if let Ok(all_emb) = trace_store.all_embeddings() {
        for (id, emb) in all_emb {
            trace_index.add(id, emb);
        }
    }

    let config = DynamicRecallConfig {
        max_candidates: cli.max_candidates,
        max_workspace_tokens: cli.max_tokens,
        ..Default::default()
    };

    let adapter: Box<dyn ModelAdapter> = build_completion_adapter(&cli.adapter)?;

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
    )
}

fn build_aux_adapter(model: &str) -> Box<dyn ModelAdapter + Send + Sync> {
    match model {
        "sonnet" => Box::new(caw_adapters::ClaudeCodeAdapter::sonnet()),
        _ => Box::new(caw_adapters::ClaudeCodeAdapter::haiku()),
    }
}

fn build_completion_adapter(adapter_name: &str) -> Result<Box<dyn ModelAdapter>> {
    let adapter: Box<dyn ModelAdapter> = match adapter_name {
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
        "claude-code" => Box::new(caw_adapters::ClaudeCodeAdapter::sonnet()),
        "claude-code-haiku" => Box::new(caw_adapters::ClaudeCodeAdapter::haiku()),
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
            Box::new(caw_adapters::OllamaAdapter::local(other, rt))
        }
    };
    Ok(adapter)
}

#[allow(clippy::too_many_arguments)]
fn run_interactive(
    mut retriever: SemanticRetriever<FastEmbedProvider, SqliteStubStore, HnswVectorIndex>,
    mut trace_embedder: FastEmbedProvider,
    mut trace_index: HnswVectorIndex,
    adapter: Box<dyn ModelAdapter>,
    config: DynamicRecallConfig,
    system: &str,
    curate: bool,
    aux_model: &str,
    context_budget: usize,
    session_dir: Option<&std::path::Path>,
) -> Result<()> {
    use caw_transform::{extract_probes, extract_thinking_steps};
    use std::collections::{HashMap, HashSet};

    let mut loaded: Vec<RecallFragment> = Vec::new();
    let mut loaded_ids: HashSet<StubId> = HashSet::new();

    // Session history: in-memory content map + dedicated vector index.
    let mut session_index = HnswVectorIndex::new();
    let mut session_content: HashMap<StubId, String> = HashMap::new();
    let mut session_file: Option<SessionFile> = None;
    let mut session_turn = 0usize;

    if let Some(sdir) = session_dir {
        std::fs::create_dir_all(sdir).ok();
        let pipeline = caw_ingest::IngestionPipeline::new();
        let file_name = format!(
            "session-{}.md",
            caw_orchestrator::session::timestamp_str()
        );
        let current_path = sdir.join(&file_name);
        match SessionFile::load_previous(sdir, &current_path, &pipeline, &mut retriever) {
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
            break;
        }

        let query = query.trim();
        if query.is_empty() {
            continue;
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
        let candidate_fragment: Option<RecallFragment> =
            (above_threshold > config.max_initial_fragments)
                .then(|| candidate_list_fragment(&hits, config.thresholds.load));
        if candidate_fragment.is_none() {
            load_fragments(&mut retriever, &hits, &mut loaded, &mut loaded_ids, &config)?;
        } else {
            debug!(
                above_threshold,
                max_initial = config.max_initial_fragments,
                "query too broad for initial augmentation — surfacing candidate list"
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
            let answer = &response.answer;
            let mut seen_paths = std::collections::HashSet::new();
            let mentioned: Vec<ScoredStub> = hits
                .iter()
                .filter(|h| h.score >= config.thresholds.load)
                .filter(|h| {
                    let norm = h.stub.path.trim_start_matches("./");
                    answer.contains(norm) || answer.contains(&h.stub.path)
                })
                .filter(|h| seen_paths.insert(h.stub.path.clone()))
                .cloned()
                .collect();
            if !mentioned.is_empty() {
                debug!(count = mentioned.len(), "loading files mentioned in response to candidate list");
                load_fragments(&mut retriever, &mentioned, &mut loaded, &mut loaded_ids, &config)?;
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
    config: &DynamicRecallConfig,
) {
    for (stub_id, score) in hits {
        if loaded_ids.contains(&stub_id) || score < config.thresholds.load {
            continue;
        }
        let current_tokens: usize = loaded.iter().map(|f| f.tokens).sum();
        if current_tokens >= config.max_workspace_tokens {
            break;
        }
        if let Some(content) = session_content.get(&stub_id) {
            let tokens = content.split_whitespace().count().max(1);
            if current_tokens + tokens <= config.max_workspace_tokens {
                loaded_ids.insert(stub_id.clone());
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
