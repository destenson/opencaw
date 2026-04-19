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

    /// GPU sub-batch size. The consumer accumulates `batch_size` stubs,
    /// sorts them by text length, and dispatches them to the embedder in
    /// chunks of `sub_batch_size`. Bucketing by length cuts padding waste:
    /// short texts pad to a short max, long texts to a longer max, instead
    /// of everything padding to the batch-wide max token count.
    #[arg(long, default_value_t = 64)]
    sub_batch_size: usize,
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
    //
    // *** Dedicated threadpool for the producer ***
    //
    // The producer's `par_iter` cannot run on the global rayon pool: the
    // HuggingFace tokenizers crate uses the global pool internally for
    // `encode_batch`, which the consumer calls. If producer rayon workers
    // block on `tx.send` (channel full), they starve tokenizer tasks, and
    // tokenizer waits forever for workers that wait for the consumer —
    // classic closed-loop deadlock. A dedicated pool isolates the two.
    let (tx, rx) = sync_channel::<(Stub, String)>(cli.channel_capacity);
    let pipeline = Arc::new(IngestionPipeline::new());

    let total_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Leave a few cores for the consumer thread, tokenizer's global pool,
    // and sqlite/WAL writes. Producer work is I/O-heavy so it oversubscribes
    // well, but keeping some headroom avoids starving the GPU path.
    let producer_threads = total_cpus.saturating_sub(4).max(4);
    let producer_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(producer_threads)
        .thread_name(|i| format!("ingest-{}", i))
        .build()
        .context("build producer rayon pool")?;
    eprintln!(
        "producer pool: {} threads (tokenizer keeps global pool of {})",
        producer_threads, total_cpus
    );

    let producer_pipeline = pipeline.clone();
    // Rewrite stored paths to be relative to the corpus root so the index
    // is portable (not tied to where the snapshot happens to sit in the
    // filesystem) and matches how downstream QA files reference documents.
    let corpus_root = cli.corpus.clone();
    let producer = std::thread::spawn(move || {
        producer_pool.install(|| {
            todo.par_iter().for_each(|path| {
                if should_skip(path) {
                    return;
                }
                let mut doc = match SourceDocument::from_path(path) {
                    Ok(d) => d,
                    Err(_) => return,
                };
                if let Ok(rel) = path.strip_prefix(&corpus_root) {
                    doc.path = rel.to_string_lossy().into_owned();
                }
                let content = doc.content.clone();
                for stub in producer_pipeline.ingest(doc) {
                    // Send error means the consumer exited — nothing else to
                    // do but stop producing.
                    if tx.send((stub, content.clone())).is_err() {
                        return;
                    }
                }
            });
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

    let sub_batch = cli.sub_batch_size.max(1);

    let mut last_recv_log = Instant::now();
    for item in rx.iter() {
        received += 1;
        if !first_item_seen {
            eprintln!(
                "  first stub received at {:.1}s",
                t_start.elapsed().as_secs_f64()
            );
            first_item_seen = true;
        }
        // Intake pulse: the consumer pulls from the channel, buffering
        // until the batch is full. Without this log a slow batch-fill
        // period looks identical to a hang.
        if last_recv_log.elapsed().as_secs() >= 2 {
            eprintln!(
                "  intake: received={} buf={} (waiting for batch of {})",
                received,
                buf.len() + 1,
                batch_size,
            );
            last_recv_log = Instant::now();
        }
        buf.push(item);
        if buf.len() >= batch_size {
            let t0 = Instant::now();
            let n = buf.len();
            flush_batch(&mut buf, &mut embedder, &mut store, &mut done_files, sub_batch)
                .context("flush batch")?;
            let batch_ms = t0.elapsed().as_millis();
            done_stubs += n;
            if done_stubs == n {
                eprintln!("  first batch flushed in {}ms", batch_ms);
            }
            if last_log.elapsed().as_secs() >= 2 {
                eprintln!(
                    "  {} stubs / {} files in {:.1}s (recv'd {}, last batch {}ms, {:.1} stubs/s)",
                    done_stubs,
                    done_files.len(),
                    t_start.elapsed().as_secs_f64(),
                    received,
                    batch_ms,
                    done_stubs as f64 / t_start.elapsed().as_secs_f64(),
                );
                last_log = Instant::now();
            }
        }
    }

    if !buf.is_empty() {
        let n = buf.len();
        flush_batch(&mut buf, &mut embedder, &mut store, &mut done_files, sub_batch)
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

/// Flush an accumulated batch: build the embedding texts, sort by length
/// to reduce padding waste, embed in sub-batches of `sub_batch_size`, then
/// persist everything in a single transaction.
///
/// Length-bucketing: texts are sorted ascending by char count before the
/// sub-batch split, so each sub-batch contains similar-length texts and
/// pads only to its own local max. Across a bench run this typically cuts
/// forward-pass FLOPs by 2-5x on mixed corpora where summary+outline
/// lengths vary by an order of magnitude.
fn flush_batch(
    buf: &mut Vec<(Stub, String)>,
    embedder: &mut CandleEmbeddingProvider,
    store: &mut SqliteStubStore,
    done_files: &mut HashSet<String>,
    sub_batch_size: usize,
) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }

    let n = buf.len();
    // Build (original_index, text) then sort by text length. We need to
    // invert the sort later to line embeddings back up with buf entries.
    let mut indexed: Vec<(usize, String)> = buf
        .iter()
        .enumerate()
        .map(|(i, (stub, _))| {
            (
                i,
                format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" ")),
            )
        })
        .collect();
    indexed.sort_by_key(|(_, t)| t.len());

    let sub = sub_batch_size.max(1);
    // Per-stub embedding, indexed by original buf position.
    let mut embeddings_by_idx: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    for chunk in indexed.chunks(sub) {
        let refs: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        let vecs = embedder
            .embed_document(refs)
            .context("embed sub-batch on GPU")?;
        if vecs.len() != chunk.len() {
            anyhow::bail!(
                "embedder returned {} vectors for sub-batch of {}",
                vecs.len(),
                chunk.len()
            );
        }
        for ((orig_idx, _), v) in chunk.iter().zip(vecs.into_iter()) {
            embeddings_by_idx[*orig_idx] = Some(v);
        }
    }

    let items: Vec<(Stub, Vec<f32>, String)> = buf
        .drain(..)
        .zip(embeddings_by_idx.into_iter())
        .map(|((stub, content), emb_opt)| {
            let emb = emb_opt.expect("every index should have an embedding");
            done_files.insert(stub.path.clone());
            (stub, emb, content)
        })
        .collect();
    store.insert_batch(items).context("insert_batch")?;
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
