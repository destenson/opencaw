//! Build the retrieval index for a corpus directory.
//!
//! Streaming, resumable indexer:
//!
//! - **Pipelined**: a rayon producer walks the corpus and ingests files in
//!   parallel, pushing stubs into a bounded channel. A single consumer
//!   thread batches them, runs the GPU embedder, and inserts results into
//!   sqlite. CPU (ingest) and GPU (embed) stay busy concurrently.
//! - **Resumable**: on startup the indexer reads the existing stub table
//!   and skips any (path, mtime) already present. If the process is killed
//!   partway through, the next run picks up roughly where the last one
//!   stopped — at worst a few in-flight files are re-ingested, which is
//!   idempotent because `SqliteStubStore::insert` is INSERT OR REPLACE.
//! - **Incremental commits**: every embedded batch writes straight to
//!   sqlite (no `--rebuild` fight). Progress is durable the moment it
//!   lands; there is no giant "commit at the end" step to lose.
//!
//! Pass `--rebuild` to start fresh (deletes the sqlite and reindexes).

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use caw_core::{EmbeddingProvider, Stub, StubStore};
use caw_index::{CandleEmbeddingProvider, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(name = "caw-bench-build-index", about = "Pre-build a retrieval index for caw-bench")]
struct Cli {
    /// Directory whose files form the corpus.
    #[arg(long)]
    corpus: PathBuf,

    /// Output sqlite path. Parent directory is created if missing.
    #[arg(long)]
    out: PathBuf,

    /// Delete any existing index at `--out` before starting. Without this,
    /// existing (path, mtime) pairs are skipped so interrupted runs resume.
    #[arg(long)]
    rebuild: bool,

    /// Embedding batch size. Larger = better GPU utilization up to VRAM
    /// limits; smaller = lower latency to first durable progress.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Bounded channel capacity between ingestion producers and the
    /// embedder consumer. Sized in stubs, not files. Keeps memory bounded
    /// while still giving the embedder a few batches of slack.
    #[arg(long, default_value_t = 1024)]
    channel_capacity: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(parent) = cli.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir {}", parent.display()))?;
        }
    }
    if cli.rebuild && cli.out.exists() {
        std::fs::remove_file(&cli.out)
            .with_context(|| format!("remove existing index {}", cli.out.display()))?;
        eprintln!("--rebuild: cleared existing index");
    }

    let mut embedder = CandleEmbeddingProvider::bge_small().context("init bge-small (candle)")?;
    let dim = embedder.dimension();
    let out_str = cli.out.to_string_lossy().into_owned();
    let mut store = SqliteStubStore::new(&out_str, dim)
        .with_context(|| format!("open sqlite store at {}", cli.out.display()))?;

    let already: HashSet<(String, u64)> = store
        .indexed_paths()
        .context("read existing index for resume")?
        .into_iter()
        .collect();
    if !already.is_empty() {
        eprintln!("resume: {} (path, mtime) pairs already indexed; skipping", already.len());
    }

    eprintln!("walking {}", cli.corpus.display());
    let mut archive_skipped: usize = 0;
    let paths: Vec<PathBuf> = WalkDir::new(&cli.corpus)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| {
            let p = e.path();
            if is_archive(p) {
                archive_skipped += 1;
                None
            } else {
                Some(p.to_path_buf())
            }
        })
        .collect();

    if archive_skipped > 0 {
        eprintln!(
            "note: skipped {} archive files (.gz/.tar/.bz2/.xz/.zip). Package inspection is \
             not yet implemented — if this corpus contains documentation inside archives, \
             decompress first (see scripts/snapshot-corpus.sh) or wait for archive support.",
            archive_skipped
        );
    }

    // Stat + resume-filter in one pass. mtime mismatches re-ingest the file;
    // `INSERT OR REPLACE` in the store keeps that idempotent.
    let todo: Vec<PathBuf> = paths
        .into_iter()
        .filter(|p| {
            let path_str = p.to_string_lossy().into_owned();
            let mtime = std::fs::metadata(p)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            !already.contains(&(path_str, mtime))
        })
        .collect();

    if todo.is_empty() {
        eprintln!("nothing to do: all files already indexed");
        return Ok(());
    }
    eprintln!("{} files to ingest", todo.len());

    // Pipeline: rayon producers → bounded channel → single GPU consumer.
    // The channel's capacity is the bound on in-flight stubs, which caps
    // memory regardless of how fast ingestion outruns embedding.
    let (tx, rx) = sync_channel::<(Stub, String)>(cli.channel_capacity);
    let pipeline = Arc::new(IngestionPipeline::new());

    let producer_pipeline = pipeline.clone();
    let producer = std::thread::spawn(move || {
        todo.par_iter().for_each(|path| {
            if should_skip(path) {
                return;
            }
            let doc = match SourceDocument::from_path(path) {
                Ok(d) => d,
                Err(_) => return,
            };
            let content = doc.content.clone();
            for stub in producer_pipeline.ingest(doc) {
                // Send error means the consumer exited — nothing else to do
                // but stop producing.
                if tx.send((stub, content.clone())).is_err() {
                    return;
                }
            }
        });
        drop(tx);
    });

    let batch_size = cli.batch_size.max(1);
    let mut buf: Vec<(Stub, String)> = Vec::with_capacity(batch_size);
    let mut done_stubs: usize = 0;
    let mut received: usize = 0;
    let mut done_files: HashSet<String> = HashSet::new();
    let t_start = Instant::now();
    let mut last_log = Instant::now();
    let mut first_item_seen = false;

    for item in rx.iter() {
        received += 1;
        if !first_item_seen {
            eprintln!(
                "  first stub received at {:.1}s",
                t_start.elapsed().as_secs_f64()
            );
            first_item_seen = true;
        }
        buf.push(item);
        if buf.len() >= batch_size {
            let t0 = Instant::now();
            flush_batch(&mut buf, &mut embedder, &mut store, &mut done_files)
                .context("flush batch")?;
            let batch_ms = t0.elapsed().as_millis();
            done_stubs += batch_size;
            if done_stubs == batch_size {
                eprintln!("  first batch flushed in {}ms", batch_ms);
            }
            if last_log.elapsed().as_secs() >= 2 {
                eprintln!(
                    "  {} stubs / {} files in {:.1}s (recv'd {}, last batch {}ms)",
                    done_stubs,
                    done_files.len(),
                    t_start.elapsed().as_secs_f64(),
                    received,
                    batch_ms,
                );
                last_log = Instant::now();
            }
        }
    }

    if !buf.is_empty() {
        let n = buf.len();
        flush_batch(&mut buf, &mut embedder, &mut store, &mut done_files)
            .context("flush final batch")?;
        done_stubs += n;
    }

    producer
        .join()
        .map_err(|_| anyhow::anyhow!("ingestion producer panicked"))?;

    eprintln!(
        "done: {} stubs across {} files in {:.1}s",
        done_stubs,
        done_files.len(),
        t_start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn flush_batch(
    buf: &mut Vec<(Stub, String)>,
    embedder: &mut CandleEmbeddingProvider,
    store: &mut SqliteStubStore,
    done_files: &mut HashSet<String>,
) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }

    let texts: Vec<String> = buf
        .iter()
        .map(|(stub, _)| format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" ")))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let embeddings = embedder
        .embed_document(refs)
        .context("embed batch on GPU")?;

    if embeddings.len() != buf.len() {
        anyhow::bail!(
            "embedder returned {} vectors for batch of {}",
            embeddings.len(),
            buf.len()
        );
    }

    for ((stub, content), embedding) in buf.drain(..).zip(embeddings.into_iter()) {
        done_files.insert(stub.path.clone());
        store
            .insert(stub, embedding, content)
            .context("insert stub")?;
    }
    Ok(())
}

/// Pre-producer filter for files we never want to even try reading. Hidden
/// files and known-uninteresting binaries (images, fonts, compiled objects).
/// Archives are handled separately (see `is_archive`) so they can be counted
/// and surfaced in the run summary rather than silently dropped.
fn should_skip(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.starts_with('.') {
        return true;
    }
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some(
            "lock" | "png" | "jpg" | "jpeg" | "gif" | "ico" | "svg" | "woff" | "woff2" | "ttf"
                | "eot" | "otf" | "exe" | "dll" | "so" | "dylib" | "o" | "a" | "wasm" | "pyc"
                | "pyo" | "class"
        )
    )
}

/// Archive extensions we don't yet unpack. Callers count these so the run
/// summary can make it obvious when a corpus is mostly archives.
fn is_archive(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("zip" | "tar" | "gz" | "bz2" | "xz" | "7z" | "rar")
    )
}
