//! Build the retrieval index for a corpus directory.
//!
//! Ingestion is a one-shot preprocessing step, separate from the benchmark
//! run itself. This binary walks a directory, produces stubs via the
//! standard ingestion pipeline, embeds each with bge-small, and persists
//! everything to a sqlite file. The bench then opens that file read-only
//! and shares it across all items.
//!
//! Re-running against an existing index re-ingests — no content-hash
//! incremental update here. The expectation is that you rebuild when the
//! corpus or ingestion logic changes, and otherwise leave the file alone.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use caw_core::{EmbeddingProvider, StubStore};
use caw_index::{FastEmbedProvider, SqliteStubStore};
use caw_ingest::IngestionPipeline;

#[derive(Parser, Debug)]
#[command(name = "caw-bench-build-index", about = "Pre-build a retrieval index for caw-bench")]
struct Cli {
    /// Directory whose files form the corpus.
    #[arg(long)]
    corpus: PathBuf,

    /// Output sqlite path. Parent directory is created if missing.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(parent) = cli.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir {}", parent.display()))?;
        }
    }
    if cli.out.exists() {
        std::fs::remove_file(&cli.out)
            .with_context(|| format!("remove stale index {}", cli.out.display()))?;
    }

    eprintln!("ingesting {}", cli.corpus.display());
    let t_ingest = Instant::now();
    let pipeline = IngestionPipeline::new();
    let stubs_with_content = pipeline
        .ingest_directory(&cli.corpus)
        .context("ingest_directory failed")?;
    eprintln!(
        "  ingested {} stubs in {:.1}s",
        stubs_with_content.len(),
        t_ingest.elapsed().as_secs_f64()
    );

    let mut embedder = FastEmbedProvider::bge_small().context("init bge-small")?;
    let dim = embedder.dimension();
    let out_str = cli.out.to_string_lossy().into_owned();
    let mut store = SqliteStubStore::new(&out_str, dim)
        .with_context(|| format!("open sqlite store at {}", cli.out.display()))?;

    let total = stubs_with_content.len();
    let batch_size: usize = 64;
    let t_embed = Instant::now();
    let mut done = 0usize;

    for chunk in stubs_with_content.chunks(batch_size) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|(stub, _)| {
                format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "))
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = embedder
            .embed_document(refs)
            .context("embed document batch")?;

        if embeddings.len() != chunk.len() {
            anyhow::bail!(
                "embedder returned {} vectors for batch of {}",
                embeddings.len(),
                chunk.len()
            );
        }

        for ((stub, content), embedding) in chunk.iter().zip(embeddings.into_iter()) {
            store
                .insert(stub.clone(), embedding, content.clone())
                .with_context(|| format!("insert stub {}", stub.id.0))?;
        }

        done += chunk.len();
        if done.is_multiple_of(batch_size * 8) || done == total {
            eprintln!(
                "  embedded {}/{} ({:.1}s elapsed)",
                done,
                total,
                t_embed.elapsed().as_secs_f64()
            );
        }
    }

    eprintln!(
        "done: {} stubs indexed into {} ({:.1}s embed)",
        total,
        cli.out.display(),
        t_embed.elapsed().as_secs_f64()
    );

    Ok(())
}
