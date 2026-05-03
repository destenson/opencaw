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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use caw_core::{EmbeddingProvider, Stub};
use caw_index::{CandleEmbeddingProvider, OnnxEmbeddingProvider, OnnxVariant, SqliteStubStore};
use caw_ingest::{DocumentId, IngestionPipeline, SourceDocument};
use clap::ValueEnum;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "caw-bench-build-index",
    about = "Pre-build a retrieval index for caw-bench"
)]
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
    /// limits; smaller = lower latency to first durable progress. Must be
    /// >= `sub_batch_size` or the consumer will never emit a full sub-batch.
    #[arg(long, default_value_t = 512)]
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
    ///
    /// Default is deliberately large: candle's per-op sync overhead means
    /// the total embed time is dominated by launching ops, not by their
    /// actual work. A sub-batch of 256 takes roughly the same wall time as
    /// a sub-batch of 64, so 4x more stubs per call ≈ 4x throughput.
    #[arg(long, default_value_t = 256)]
    sub_batch_size: usize,

    /// Embedding backend. `candle` runs BGE-small via candle-transformers
    /// with CUDA (falls back to CPU). `onnx` runs the BGE-small ONNX export
    /// via ONNX Runtime, trying the CUDA execution provider first and
    /// falling back to CPU. ONNX Runtime's single-graph execution is
    /// typically much faster than candle's eager op-by-op eval.
    #[arg(long, value_enum, default_value_t = BackendArg::Candle)]
    backend: BackendArg,

    /// ONNX model variant: `fp32` (default, baseline precision), `fp16`,
    /// `int8`, or `quantized` (dynamic int8). Variants are pulled from
    /// `Xenova/bge-small-en-v1.5`. Ignored when `--backend candle`.
    #[arg(long, value_enum, default_value_t = OnnxVariantArg::Fp32)]
    onnx_variant: OnnxVariantArg,

    /// Seconds between progress logs. `0` disables periodic logging (first-
    /// item/first-batch markers and the final summary still print unless
    /// `--quiet` is set). Diagnostics are always compiled in — this flag
    /// only controls how often they're emitted.
    #[arg(long, default_value_t = 10)]
    log_interval: u64,

    /// Suppress all non-error output, including the startup banner, intake
    /// pulses, periodic progress, first-item/first-batch markers, and the
    /// final summary. Errors still go to stderr. Equivalent to
    /// `--log-interval 0` plus silencing the one-shot messages.
    #[arg(long)]
    quiet: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Candle,
    Onnx,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OnnxVariantArg {
    Fp32,
    Fp16,
    Int8,
    Quantized,
}

impl From<OnnxVariantArg> for OnnxVariant {
    fn from(v: OnnxVariantArg) -> Self {
        match v {
            OnnxVariantArg::Fp32 => OnnxVariant::Fp32,
            OnnxVariantArg::Fp16 => OnnxVariant::Fp16,
            OnnxVariantArg::Int8 => OnnxVariant::Int8,
            OnnxVariantArg::Quantized => OnnxVariant::Quantized,
        }
    }
}

/// Dynamic-dispatch wrapper so `flush_batch` can call the same embed path
/// regardless of which backend was chosen. The per-call vtable cost is
/// negligible next to the embedding work itself.
struct AnyEmbedder(Box<dyn EmbeddingProvider + Send>);

impl AnyEmbedder {
    fn embed_document(&mut self, texts: Vec<&str>) -> caw_core::CawResult<Vec<Vec<f32>>> {
        self.0.embed_document(texts)
    }

    fn dimension(&self) -> usize {
        self.0.dimension()
    }

    fn seq_len_histogram(&self) -> Option<Vec<(usize, u64)>> {
        self.0.seq_len_histogram()
    }

    fn item_seq_len_histogram(&self) -> Option<Vec<(usize, u64)>> {
        self.0.item_seq_len_histogram()
    }
}

fn build_embedder(kind: BackendArg, onnx_variant: OnnxVariantArg) -> Result<AnyEmbedder> {
    match kind {
        BackendArg::Candle => {
            let e = CandleEmbeddingProvider::bge_small().context("init bge-small (candle)")?;
            Ok(AnyEmbedder(Box::new(e)))
        }
        BackendArg::Onnx => {
            let variant: OnnxVariant = onnx_variant.into();
            let e = OnnxEmbeddingProvider::bge_small_variant(variant)
                .with_context(|| format!("init bge-small onnx variant={}", variant.as_str()))?;
            Ok(AnyEmbedder(Box::new(e)))
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(parent) = cli.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create parent dir {}", parent.display()))?;
    }
    // `verbose` gates one-shot informational output; `log_interval_s` gates
    // the periodic progress/intake loops. `--quiet` forces both off; an
    // explicit `--log-interval 0` disables only periodic logging while
    // keeping the startup/summary messages. Errors always print regardless.
    let verbose = !cli.quiet;
    let log_interval_s = if cli.quiet { 0 } else { cli.log_interval };
    macro_rules! vlog {
        ($($arg:tt)*) => { if verbose { eprintln!($($arg)*); } };
    }

    if verbose {
        if log_interval_s == 0 {
            eprintln!(
                "diagnostics: periodic progress logging disabled ({}). \
                 Pass --log-interval N (seconds) to enable.",
                if cli.quiet {
                    "--quiet"
                } else {
                    "--log-interval 0"
                },
            );
        } else {
            eprintln!(
                "diagnostics: logging progress every {}s (override with \
                 --log-interval N, or --quiet to silence)",
                log_interval_s,
            );
        }
    }

    if cli.rebuild && cli.out.exists() {
        std::fs::remove_file(&cli.out)
            .with_context(|| format!("remove existing index {}", cli.out.display()))?;
        vlog!("--rebuild: cleared existing index");
    }

    let mut embedder = build_embedder(cli.backend, cli.onnx_variant)?;
    let dim = embedder.dimension();
    let out_str = cli.out.to_string_lossy().into_owned();
    let mut store = SqliteStubStore::new(&out_str, dim)
        .with_context(|| format!("open sqlite store at {}", cli.out.display()))?;

    let already: HashSet<DocumentId> = store
        .indexed_paths()
        .context("read existing index for resume")?
        .into_iter()
        .collect();
    if !already.is_empty() {
        vlog!(
            "resume: {} (path, mtime) pairs already indexed; skipping",
            already.len()
        );
    }

    vlog!("walking {}", cli.corpus.display());
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
        vlog!(
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
        vlog!("nothing to do: all files already indexed");
        return Ok(());
    }
    let todo_total = todo.len();
    vlog!("{} files to ingest", todo_total);

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
    vlog!(
        "producer pool: {} threads (tokenizer keeps global pool of {})",
        producer_threads,
        total_cpus
    );

    let producer_pipeline = pipeline.clone();
    // Rewrite stored paths to be relative to the corpus root so the index
    // is portable (not tied to where the snapshot happens to sit in the
    // filesystem) and matches how downstream QA files reference documents.
    let corpus_root = cli.corpus.clone();
    // Producer-side instrumentation. `emitted` counts stubs handed to the
    // channel; `send_block_ns` sums wall-clock nanoseconds spent blocked on
    // `tx.send` across all producer threads — high values mean the embedder
    // is the bottleneck and producers are waiting on channel space. Low
    // values with low consumer throughput mean the producer is the
    // bottleneck (ingest + chunk + outline).
    let emitted = Arc::new(AtomicU64::new(0));
    let send_block_ns = Arc::new(AtomicU64::new(0));
    let ingest_ns = Arc::new(AtomicU64::new(0));
    let emitted_p = emitted.clone();
    let send_block_p = send_block_ns.clone();
    let ingest_p = ingest_ns.clone();
    let producer = std::thread::spawn(move || {
        producer_pool.install(|| {
            todo.par_iter().for_each(|path| {
                if should_skip(path) {
                    return;
                }
                let t_ing = Instant::now();
                let mut doc = match SourceDocument::from_path(path) {
                    Ok(d) => d,
                    Err(_) => return,
                };
                if let Ok(rel) = path.strip_prefix(&corpus_root) {
                    doc.path = rel.to_string_lossy().into_owned();
                }
                // `ingest` now pairs each stub with its own chunk's embed
                // text (body + overlap prefix), not the full document. The
                // prior code cloned `doc.content` once per stub and shipped
                // it through the channel, which (a) embedded every chunk of
                // a multi-chunk file against the same doc-level text,
                // collapsing chunk-level retrieval, and (b) persisted N
                // copies of the whole file in the `contents` table — the
                // cause of the 80 GB index on a 516 MB corpus.
                let stubs_and_text = producer_pipeline.ingest(doc);
                ingest_p.fetch_add(t_ing.elapsed().as_nanos() as u64, Ordering::Relaxed);
                for (stub, embed_text) in stubs_and_text {
                    let t_send = Instant::now();
                    // Send error means the consumer exited — nothing else to
                    // do but stop producing.
                    if tx.send((stub, embed_text)).is_err() {
                        return;
                    }
                    send_block_p.fetch_add(t_send.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    emitted_p.fetch_add(1, Ordering::Relaxed);
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

    // Consumer-side cumulative timers. Paired with the producer counters,
    // these partition wall time into ingest / wait-for-stubs / embed / sqlite,
    // which is what you need to tell GPU-bound from CPU-bound from IO-bound.
    let mut recv_block_ns: u64 = 0;
    let mut embed_ns_total: u64 = 0;
    let mut sqlite_ns_total: u64 = 0;
    let mut last_recv_log = Instant::now();
    let mut last_emitted: u64 = 0;
    let mut last_done: usize = 0;
    let mut last_tick = Instant::now();
    loop {
        let t_recv = Instant::now();
        let item = match rx.recv() {
            Ok(it) => it,
            Err(_) => break,
        };
        recv_block_ns += t_recv.elapsed().as_nanos() as u64;
        received += 1;
        if !first_item_seen {
            vlog!(
                "  first stub received at {:.1}s",
                t_start.elapsed().as_secs_f64()
            );
            first_item_seen = true;
        }
        // Intake pulse: the consumer pulls from the channel, buffering
        // until the batch is full. Without this log a slow batch-fill
        // period looks identical to a hang. Gated on log_interval_s so
        // `--quiet` / `--log-interval 0` silences it.
        if log_interval_s > 0 && last_recv_log.elapsed().as_secs() >= log_interval_s {
            vlog!(
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
            let mut stage_embed_ns: u64 = 0;
            let mut stage_sqlite_ns: u64 = 0;
            flush_batch(
                &mut buf,
                &mut embedder,
                &mut store,
                &mut done_files,
                sub_batch,
                &mut stage_embed_ns,
                &mut stage_sqlite_ns,
            )
            .context("flush batch")?;
            let batch_ms = t0.elapsed().as_millis();
            embed_ns_total += stage_embed_ns;
            sqlite_ns_total += stage_sqlite_ns;
            done_stubs += n;
            if done_stubs == n {
                vlog!(
                    "  first batch flushed in {}ms (embed {}ms, sqlite {}ms)",
                    batch_ms,
                    stage_embed_ns / 1_000_000,
                    stage_sqlite_ns / 1_000_000,
                );
            }
            if log_interval_s > 0 && last_log.elapsed().as_secs() >= log_interval_s {
                let done_f = done_files.len();
                let remaining = todo_total.saturating_sub(done_f);
                let elapsed_s = t_start.elapsed().as_secs_f64();
                let rate = done_stubs as f64 / elapsed_s;
                // Instant (since-last-tick) rates — steady-state throughput
                // after warmup is more informative than the since-start mean.
                let emitted_now = emitted.load(Ordering::Relaxed);
                let tick_s = last_tick.elapsed().as_secs_f64().max(1e-3);
                let emit_rate = (emitted_now - last_emitted) as f64 / tick_s;
                let consume_rate = (done_stubs - last_done) as f64 / tick_s;
                last_emitted = emitted_now;
                last_done = done_stubs;
                last_tick = Instant::now();

                let file_rate = done_f as f64 / elapsed_s.max(0.001);
                let eta_s = if file_rate > 0.0 {
                    remaining as f64 / file_rate
                } else {
                    0.0
                };

                // Pipeline-share percentages of consumer wall time. The three
                // numbers should roughly sum to the consumer thread's wall
                // time (the small remainder is loop overhead + logging).
                let wall_ns = (elapsed_s * 1e9) as u64;
                let pct = |n: u64| 100.0 * n as f64 / wall_ns.max(1) as f64;
                let send_block_total = send_block_ns.load(Ordering::Relaxed);
                let ingest_total = ingest_ns.load(Ordering::Relaxed);

                vlog!(
                    "  {} stubs / {} files ({}/{}, {} left) in {:.1}s\n    \
                     rates: emit={:.0}/s consume={:.0}/s mean={:.1}/s   \
                     consumer: recv={:.0}% embed={:.0}% sqlite={:.0}%   \
                     producer: ingest={}s send-block={}s   \
                     last batch {}ms (embed {}ms sqlite {}ms)   eta ~{:.0}s",
                    done_stubs,
                    done_f,
                    done_f,
                    todo_total,
                    remaining,
                    elapsed_s,
                    emit_rate,
                    consume_rate,
                    rate,
                    pct(recv_block_ns),
                    pct(embed_ns_total),
                    pct(sqlite_ns_total),
                    ingest_total / 1_000_000_000,
                    send_block_total / 1_000_000_000,
                    batch_ms,
                    stage_embed_ns / 1_000_000,
                    stage_sqlite_ns / 1_000_000,
                    eta_s,
                );
                last_log = Instant::now();
            }
        }
    }

    if !buf.is_empty() {
        let n = buf.len();
        let mut stage_embed_ns: u64 = 0;
        let mut stage_sqlite_ns: u64 = 0;
        flush_batch(
            &mut buf,
            &mut embedder,
            &mut store,
            &mut done_files,
            sub_batch,
            &mut stage_embed_ns,
            &mut stage_sqlite_ns,
        )
        .context("flush final batch")?;
        embed_ns_total += stage_embed_ns;
        sqlite_ns_total += stage_sqlite_ns;
        done_stubs += n;
    }

    producer
        .join()
        .map_err(|_| anyhow::anyhow!("ingestion producer panicked"))?;

    let total_s = t_start.elapsed().as_secs_f64();
    vlog!(
        "done: {} stubs across {} files in {:.1}s ({:.1} stubs/s)",
        done_stubs,
        done_files.len(),
        total_s,
        done_stubs as f64 / total_s.max(1e-3),
    );
    vlog!(
        "  consumer breakdown: recv-block={:.1}s embed={:.1}s sqlite={:.1}s",
        recv_block_ns as f64 / 1e9,
        embed_ns_total as f64 / 1e9,
        sqlite_ns_total as f64 / 1e9,
    );
    vlog!(
        "  producer breakdown: ingest={:.1}s send-block={:.1}s emitted={} stubs",
        ingest_ns.load(Ordering::Relaxed) as f64 / 1e9,
        send_block_ns.load(Ordering::Relaxed) as f64 / 1e9,
        emitted.load(Ordering::Relaxed),
    );

    // seq_len histograms (when the backend instruments them). Batch-max
    // tells us what shape TRT would see; per-item tells us how much the
    // batching policy is wasting on padding.
    let fmt_hist = |hist: &[(usize, u64)], total: u64| -> String {
        hist.iter()
            .map(|(bucket, count)| {
                let pct = 100.0 * *count as f64 / total.max(1) as f64;
                format!("<={}:{} ({:.1}%)", bucket, count, pct)
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(hist) = embedder.seq_len_histogram() {
        let total: u64 = hist.iter().map(|(_, c)| *c).sum();
        if total > 0 {
            vlog!(
                "  seq_len histogram (per batch, {} batches): {}",
                total,
                fmt_hist(&hist, total)
            );
        }
    }
    if let Some(hist) = embedder.item_seq_len_histogram() {
        let total: u64 = hist.iter().map(|(_, c)| *c).sum();
        if total > 0 {
            vlog!(
                "  seq_len histogram (per item, {} items):    {}",
                total,
                fmt_hist(&hist, total)
            );
        }
    }

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
    embedder: &mut AnyEmbedder,
    store: &mut SqliteStubStore,
    done_files: &mut HashSet<String>,
    sub_batch_size: usize,
    embed_ns_out: &mut u64,
    sqlite_ns_out: &mut u64,
) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }

    let n = buf.len();
    let mut indexed: Vec<(usize, String)> = buf
        .iter()
        .enumerate()
        .map(|(i, (_stub, embed_text))| (i, embed_text.clone()))
        .collect();
    indexed.sort_by_key(|(_, t)| t.len());

    let sub = sub_batch_size.max(1);
    // Per-stub embedding, indexed by original buf position.
    let mut embeddings_by_idx: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    for chunk in indexed.chunks(sub) {
        let refs: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        let t_emb = Instant::now();
        let vecs = embedder
            .embed_document(refs)
            .context("embed sub-batch on GPU")?;
        *embed_ns_out += t_emb.elapsed().as_nanos() as u64;
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

    let items: Vec<(Stub, Vec<f32>)> = buf
        .drain(..)
        .zip(embeddings_by_idx)
        .map(|((stub, _embed_text), emb_opt)| {
            let emb = emb_opt.expect("every index should have an embedding");
            done_files.insert(stub.path.clone());
            (stub, emb)
        })
        .collect();
    let t_sql = Instant::now();
    store.insert_batch(items).context("insert_batch")?;
    *sqlite_ns_out += t_sql.elapsed().as_nanos() as u64;
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
            "lock"
                | "png"
                | "jpg"
                | "jpeg"
                | "gif"
                | "ico"
                | "svg"
                | "woff"
                | "woff2"
                | "ttf"
                | "eot"
                | "otf"
                | "exe"
                | "dll"
                | "so"
                | "dylib"
                | "o"
                | "a"
                | "wasm"
                | "pyc"
                | "pyo"
                | "class"
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
