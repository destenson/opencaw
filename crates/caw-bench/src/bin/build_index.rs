//! Build the retrieval index for a corpus directory.
//!
//! Streaming, resumable indexer:
//!
//! - **Pipelined**: three stages run concurrently — a rayon producer walks
//!   the corpus and ingests files in parallel, N GPU embed workers each
//!   own a CandleEmbeddingProvider and pull stubs from a shared channel to
//!   embed batches (separate CUDA streams → real GPU-level concurrency,
//!   and variable per-batch latency from outlier-length stubs no longer
//!   blocks the pipeline), and a single writer thread takes (stub,
//!   content, embedding) triples off a second channel and inserts them
//!   into sqlite. All three stages can be busy at once.
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
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, sync_channel};
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

    /// Number of parallel GPU embed workers. Each worker owns its own
    /// CandleEmbeddingProvider (one model copy in VRAM per worker, ~250MB
    /// for bge-small) and runs batches on an independent CUDA stream. More
    /// workers = better GPU utilization at the cost of VRAM; set based on
    /// available VRAM and how lumpy batch timings are (quadratic outliers
    /// from long sequences benefit the most from more workers).
    #[arg(long, default_value_t = 4)]
    workers: usize,
}

fn main() -> Result<()> {
    // The HuggingFace `tokenizers` crate uses rayon's global threadpool for
    // parallel batch encoding. Our producer also uses rayon (par_iter over
    // paths), and its workers park on a full channel waiting for the
    // consumer — holding rayon slots while sleeping. That starves the
    // tokenizer, which waits forever for a free rayon worker. Disabling
    // tokenizer parallelism sidesteps the deadlock; we're already parallel
    // across batches via the producer.
    unsafe { std::env::set_var("TOKENIZERS_PARALLELISM", "false"); }

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

    let worker_count = cli.workers.max(1);
    eprintln!("initializing {} embed worker(s)", worker_count);
    let mut embedders: Vec<CandleEmbeddingProvider> = Vec::with_capacity(worker_count);
    for i in 0..worker_count {
        let t0 = Instant::now();
        embedders.push(
            CandleEmbeddingProvider::bge_small()
                .with_context(|| format!("init bge-small worker {}", i))?,
        );
        eprintln!(
            "  worker {} init in {:.1}s",
            i,
            t0.elapsed().as_secs_f64()
        );
    }
    let dim = embedders[0].dimension();
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

    // Three-stage pipeline:
    //   producer → stub_chan → N embed workers → insert_chan → writer
    // Each channel is bounded so a slow stage exerts backpressure instead
    // of ballooning memory. Worker capacity = N batches lets idle workers
    // grab the next batch the moment one frees up.
    let (stub_tx, stub_rx) = sync_channel::<(Stub, String)>(cli.channel_capacity);
    let (insert_tx, insert_rx) =
        sync_channel::<(Stub, String, Vec<f32>)>(worker_count * cli.batch_size);
    let stub_rx = Arc::new(Mutex::new(stub_rx));
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
            for pair in producer_pipeline.ingest(doc) {
                // Each pair is (stub, chunk_content). Send error means all
                // workers exited — stop producing.
                if stub_tx.send(pair).is_err() {
                    return;
                }
            }
        });
        drop(stub_tx);
    });

    let batch_size = cli.batch_size.max(1);
    let t_start = Instant::now();

    // Spawn N embed workers. Each owns one candle provider, pulls batches
    // from the shared stub receiver, embeds on GPU, and forwards results
    // to the writer. Mutex contention on the receiver is cheap (lock is
    // held only for the duration of batch collection, not GPU work).
    let mut worker_handles = Vec::with_capacity(worker_count);
    for (worker_id, mut embedder) in embedders.into_iter().enumerate() {
        let stub_rx = stub_rx.clone();
        let insert_tx = insert_tx.clone();
        let handle = std::thread::spawn(move || -> Result<()> {
            let mut first_batch_logged = false;
            loop {
                let batch = collect_batch(&stub_rx, batch_size);
                if batch.is_empty() {
                    break;
                }
                let batch_len = batch.len();
                let t0 = Instant::now();
                let triples = embed_batch(batch, &mut embedder)
                    .with_context(|| format!("worker {} embed", worker_id))?;
                if !first_batch_logged {
                    eprintln!(
                        "  worker {} first batch of {} in {}ms",
                        worker_id,
                        batch_len,
                        t0.elapsed().as_millis()
                    );
                    first_batch_logged = true;
                }
                for triple in triples {
                    if insert_tx.send(triple).is_err() {
                        return Ok(());
                    }
                }
            }
            Ok(())
        });
        worker_handles.push(handle);
    }
    // Drop our own copy of insert_tx so when all workers finish and drop
    // theirs, the writer loop exits.
    drop(insert_tx);

    // Writer runs on the main thread. Owns the store; serializes all
    // sqlite writes. WAL + synchronous=NORMAL (set in SqliteStubStore::new)
    // keeps this fast enough that a single writer doesn't bottleneck the
    // workers.
    let mut done_stubs: usize = 0;
    let mut done_files: HashSet<String> = HashSet::new();
    let mut last_log = Instant::now();
    let mut first_insert_seen = false;
    for (stub, content, embedding) in insert_rx.iter() {
        if !first_insert_seen {
            eprintln!(
                "  first embedded stub available at {:.1}s",
                t_start.elapsed().as_secs_f64()
            );
            first_insert_seen = true;
        }
        done_files.insert(stub.path.clone());
        store
            .insert(stub, embedding, content)
            .context("insert stub")?;
        done_stubs += 1;
        if last_log.elapsed().as_secs() >= 2 {
            let elapsed = t_start.elapsed().as_secs_f64();
            eprintln!(
                "  {} stubs / {} files in {:.1}s ({:.0} stubs/s)",
                done_stubs,
                done_files.len(),
                elapsed,
                done_stubs as f64 / elapsed,
            );
            last_log = Instant::now();
        }
    }

    for (i, h) in worker_handles.into_iter().enumerate() {
        h.join()
            .map_err(|_| anyhow::anyhow!("embed worker {} panicked", i))?
            .with_context(|| format!("embed worker {} failed", i))?;
    }
    producer
        .join()
        .map_err(|_| anyhow::anyhow!("ingestion producer panicked"))?;

    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!(
        "done: {} stubs across {} files in {:.1}s ({:.0} stubs/s)",
        done_stubs,
        done_files.len(),
        elapsed,
        done_stubs as f64 / elapsed,
    );

    Ok(())
}

/// Pull up to `batch_size` stubs from the shared receiver. Blocks for the
/// first item (so idle workers don't spin), then opportunistically drains
/// whatever's already queued. Returns an empty Vec when the channel is
/// closed and drained, signalling the worker to exit.
fn collect_batch(
    rx: &Mutex<Receiver<(Stub, String)>>,
    batch_size: usize,
) -> Vec<(Stub, String)> {
    let mut batch = Vec::with_capacity(batch_size);
    // Scope the lock so we don't hold it across GPU work.
    let guard = rx.lock().unwrap();
    match guard.recv() {
        Ok(first) => batch.push(first),
        Err(_) => return batch,
    }
    while batch.len() < batch_size {
        match guard.try_recv() {
            Ok(item) => batch.push(item),
            Err(_) => break,
        }
    }
    batch
}

fn embed_batch(
    batch: Vec<(Stub, String)>,
    embedder: &mut CandleEmbeddingProvider,
) -> Result<Vec<(Stub, String, Vec<f32>)>> {
    let texts: Vec<String> = batch
        .iter()
        .map(|(stub, _)| format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" ")))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let embeddings = embedder
        .embed_document(refs)
        .context("embed batch on GPU")?;

    if embeddings.len() != batch.len() {
        anyhow::bail!(
            "embedder returned {} vectors for batch of {}",
            embeddings.len(),
            batch.len()
        );
    }

    Ok(batch
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|((stub, content), embedding)| (stub, content, embedding))
        .collect())
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
