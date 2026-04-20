use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

use caw_server::{RetrieverKind, build_router, build_state};

#[derive(Parser, Debug)]
#[command(
    name = "caw-server",
    about = "OpenAI-compatible proxy that augments the last user message with recalled workspace context"
)]
struct Cli {
    /// Prebuilt stub index (sqlite produced by caw-bench-build-index).
    #[arg(long)]
    index: PathBuf,

    /// Filesystem root the index's stub paths resolve against. The store
    /// reads body bytes from `corpus_root.join(stub.path)`, so this must
    /// match whatever directory was passed to build-index.
    #[arg(long)]
    corpus_root: PathBuf,

    /// Upstream chat-completions base URL, e.g.
    /// `http://localhost:11434/v1` for Ollama, `https://api.openai.com/v1`
    /// for OpenAI, `http://localhost:8000/v1` for vLLM.
    #[arg(long)]
    upstream: String,

    /// Port the proxy listens on.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// How many stubs to fetch per request.
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Hard cap on injected recall content (whitespace-split token count).
    #[arg(long, default_value = "2000")]
    max_workspace_tokens: usize,

    /// In-memory vector index to back retrieval. `flat` is a brute-force
    /// cosine scan (no warmup, O(n·d) per query, correct for corpora
    /// under ~1M stubs). `hnsw` is instant-distance with an eager build
    /// at startup — use when the flat scan outgrows your latency budget.
    #[arg(long, value_enum, default_value_t = RetrieverKind::Flat)]
    retriever: RetrieverKind,
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("caw_server=info,info"));
    let subscriber = FmtSubscriber::builder().with_env_filter(filter).finish();
    tracing::subscriber::set_global_default(subscriber).ok();

    let cli = Cli::parse();
    let index_path = cli
        .index
        .to_str()
        .context("--index path is not valid UTF-8")?
        .to_string();

    let state = build_state(
        &index_path,
        cli.corpus_root,
        cli.upstream,
        cli.top_k,
        cli.max_workspace_tokens,
        cli.retriever,
    )?;
    let app = build_router(Arc::new(state));

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!("listening on {addr}");
    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}
