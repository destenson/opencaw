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
    #[arg(long, default_value = "8090")]
    port: u16,

    /// Candidate pool size for ANN search; the load threshold controls actual admissions.
    #[arg(long, default_value = "20")]
    max_candidates: usize,

    /// Hard cap on injected recall content (whitespace-split token count).
    /// Default is just large enough that normal recall is not clamped out; it is
    /// not tuned to any upstream context window. Set it to fit your upstream
    /// model's window (mirrors caw-orchestrator's DEFAULT_MAX_WORKSPACE_TOKENS).
    #[arg(long, default_value = "12000")]
    max_workspace_tokens: usize,

    /// Retrieval backend. `hybrid` (default) fuses BM25 lexical scores with
    /// cosine, which surfaces definitional chunks that pure cosine buries on
    /// a single-domain corpus; it pays a one-time startup cost to read every
    /// body and build posting lists. `flat` is a brute-force cosine scan (no
    /// warmup, O(n·d) per query). `hnsw` is instant-distance with an eager
    /// build at startup — use when the flat scan outgrows your latency budget.
    #[arg(long, value_enum, default_value_t = RetrieverKind::Hybrid)]
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
        cli.max_candidates,
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
