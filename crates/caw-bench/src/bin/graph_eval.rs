//! Measure whether graph-neighbor expansion improves retrieval rank of the
//! answer chunk, against semantic (BGE) retrieval alone.
//!
//! For each golden question: embed the query, take the top-k semantic hits as
//! seeds, and record the rank of the first gold stub. Then expand the seeds
//! along the `stub_edge` sidecar (`plan_expansion`), merge the admitted
//! neighbors by score, and record the gold rank again. The per-question and
//! aggregate deltas are the spike's deliverable.
//!
//! Baseline is semantic-only on purpose: the hypothesis is that graph edges are
//! orthogonal to cosine, so the cleanest test adds the graph to cosine alone
//! rather than to a cosine+BM25 mix that would muddy attribution.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use caw_core::{EmbeddingProvider, ScoredStub, StubId, StubStore, VectorIndex};
use caw_index::graph_edges::{
    byte_offset_of_line, line_start_offsets, plan_expansion, stub_for_byte, ExpansionConfig,
    GraphEdgeStore, PlannedNeighbor,
};
use caw_index::{
    bm25_tokenize, BM25Index, CandleEmbeddingProvider, HnswVectorIndex, SqliteStubStore,
};
use clap::Parser;
use serde::Deserialize;

const DEFAULT_RELATIONS: &str = "calls,imports_from,implements,inherits,method,contains";

// Mirror caw-server's hybrid fusion weights (crates/caw-server/src/lib.rs) so the
// `hybrid` baseline here matches what the live proxy actually ranks with. The
// cosine-only baseline answers "is the graph orthogonal to embeddings"; the
// hybrid baseline answers the shippable question "does expansion still help once
// BM25 already rescues symbol-name queries".
const HYBRID_SEMANTIC_WEIGHT: f32 = 0.6;
const HYBRID_KEYWORD_WEIGHT: f32 = 0.4;
/// RRF rank-bias constant (standard default). Used only by `--fusion rrf`.
const RRF_K: f32 = 60.0;

#[derive(Parser, Debug)]
#[command(
    name = "caw-bench-graph-eval",
    about = "Measure graph-neighbor expansion lift on retrieval rank"
)]
struct Args {
    /// Prebuilt stub index (SQLite) that also holds the stub_edge sidecar.
    #[arg(long)]
    index: PathBuf,

    /// Golden question set.
    #[arg(long, default_value = "crates/caw-bench/src/qa/graph_eval_qa.json")]
    questions: PathBuf,

    /// Filesystem root that gold `path`s resolve under (crates/-relative paths
    /// live under this), used to turn gold lines into byte offsets.
    #[arg(long, default_value = "crates")]
    source_root: PathBuf,

    /// Baseline retriever the expansion is measured against:
    /// `cosine` is semantic-only (isolates the graph's orthogonality to
    /// embeddings); `hybrid` is BM25 fused with cosine at the proxy's 0.6/0.4
    /// weights (the shippable question — does expansion still help once the
    /// lexical half already rescues symbol-name queries).
    #[arg(long, default_value = "cosine")]
    baseline: String,

    /// Hybrid fusion strategy (only meaningful with `--baseline hybrid`):
    /// `divide_total` is the incumbent (min-max weighted sum / total weight);
    /// `present_weight` divides by the weight of the lists an item appears in
    /// (rescues single-half hits but drops the agreement reward);
    /// `rrf` is reciprocal-rank fusion (rank-based, discards score magnitude).
    #[arg(long, default_value = "divide_total")]
    fusion: String,

    /// Depth of the baseline semantic ranking. Gold rank is measured within
    /// this pool; beyond it counts as a miss. Larger = a truer deep rank.
    #[arg(long, default_value_t = 100)]
    pool: usize,

    /// How many of the top semantic hits feed expansion as seeds. Neighbors are
    /// only reachable from these, so this bounds what expansion can rescue.
    #[arg(long, default_value_t = 30)]
    seed_k: usize,

    /// Edge relations to expand along.
    #[arg(long, default_value = DEFAULT_RELATIONS)]
    relations: String,

    /// Score multiplier applied to a seed's score when scoring its neighbor.
    #[arg(long, default_value_t = 0.5)]
    discount: f32,

    /// Max neighbors admitted per query.
    #[arg(long, default_value_t = 8)]
    max_neighbors: usize,

    /// Merge policy for admitted neighbors:
    /// `adjacent` inserts each neighbor immediately after the seed that pulled
    /// it in (a caller lands next to its callee's definition);
    /// `discounted` re-ranks everything by score (neighbor = seed*discount),
    /// which can only push neighbors below the pool.
    #[arg(long, default_value = "adjacent")]
    merge: String,

    /// Recall@k cutoffs to report (comma-separated).
    #[arg(long, default_value = "1,3,5,10,20,50")]
    recall_k: String,

    /// Emit a per-question decomposition of every baseline miss: where cosine
    /// ranked the gold, where BM25 ranked it, how many query tokens overlap the
    /// gold's indexed text, and whether that text is impoverished. Tells apart a
    /// register mismatch (zero overlap), a fusion problem (one half found it), a
    /// thin-stub problem (empty summary), and a ranking bug (overlap but buried).
    #[arg(long)]
    diagnose: bool,

    /// A gold is a "miss" for the diagnosis when its baseline rank exceeds this.
    #[arg(long, default_value_t = 10)]
    miss_cutoff: usize,
}

/// Why one missed gold stub was not retrieved, decomposed across the two halves
/// of the hybrid retriever plus the gold's own indexed-text quality.
struct MissDiag {
    qid: String,
    qtype: String,
    gold_path: String,
    hybrid_rank: Option<usize>,
    cosine_rank: Option<usize>,
    bm25_rank: Option<usize>,
    query_token_count: usize,
    overlap_count: usize,
    shared_sample: Vec<String>,
    summary_len: usize,
    body_token_count: usize,
}

#[derive(Deserialize)]
struct GoldFile {
    questions: Vec<Question>,
}

#[derive(Deserialize)]
struct Question {
    id: String,
    #[serde(rename = "type")]
    qtype: String,
    question: String,
    expected: Vec<Expected>,
}

#[derive(Deserialize)]
struct Expected {
    path: String,
    line: usize,
}

/// Outcome of one question under both retrieval modes. Rank is 1-indexed;
/// `None` means no gold stub appeared in the considered list.
struct Outcome {
    id: String,
    qtype: String,
    base_rank: Option<usize>,
    exp_rank: Option<usize>,
    /// Edge relation that brought the gold stub in, when expansion improved it.
    via_relation: Option<String>,
}

fn main() -> Result<()> {
    caw_bench::init_tracing();
    let args = Args::parse();
    let relations: Vec<String> = split_csv(&args.relations);
    let recall_ks: Vec<usize> = split_csv(&args.recall_k)
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    // --- load retrieval stack ---
    let mut embedder = CandleEmbeddingProvider::bge_small().context("init bge-small (candle)")?;
    let dim = embedder.dimension();
    // Resolve bodies under the same root as the gold paths (and as the proxy's
    // corpus-root) so the hybrid baseline's BM25 can read content; read-only so a
    // transient miss never marks a stub stale in this prebuilt index.
    let store = SqliteStubStore::new(&args.index.to_string_lossy(), dim)
        .with_context(|| format!("open index {}", args.index.display()))?
        .with_corpus_root(args.source_root.clone())
        .with_read_only(true);
    let all = store.all_embeddings().context("read embeddings")?;
    anyhow::ensure!(!all.is_empty(), "index {} is empty", args.index.display());
    let mut index = HnswVectorIndex::new();
    for (id, emb) in &all {
        index.add(id.clone(), emb.clone());
    }

    let hybrid = match args.baseline.as_str() {
        "cosine" => false,
        "hybrid" => true,
        other => anyhow::bail!("unknown --baseline '{other}' (expected cosine|hybrid)"),
    };
    // Build the lexical index only for the hybrid baseline, on the same
    // `path + summary + body` text caw-server's build_state indexes, so the
    // fused ranking here matches the proxy's.
    let bm25 = if hybrid {
        let mut bm = BM25Index::new();
        let mut missing = 0usize;
        for (id, _emb) in &all {
            let (stub, body) = match (store.get_stub(id), store.get_content(id)) {
                (Ok(s), Ok(b)) => (s, b),
                _ => {
                    missing += 1;
                    continue;
                }
            };
            bm.add(id.clone(), &format!("{} {} {}", stub.path, stub.summary, body));
        }
        eprintln!("hybrid baseline: BM25 over {} docs ({missing} skipped)", bm.len());
        Some(bm)
    } else {
        None
    };

    let edges = GraphEdgeStore::open(&args.index)?;
    eprintln!(
        "loaded {} stubs, {} stub edges",
        all.len(),
        edges.edge_count()?
    );

    let gold_file: GoldFile = serde_json::from_str(
        &std::fs::read_to_string(&args.questions)
            .with_context(|| format!("read {}", args.questions.display()))?,
    )?;

    let cfg = ExpansionConfig {
        relations,
        discount: args.discount,
        max_neighbors: args.max_neighbors,
    };

    // cache of path -> stub geometry so each gold file is read/queried once
    let mut geom_cache: HashMap<String, Vec<u64>> = HashMap::new();

    // `--fusion all` evaluates every fusion mode from a single BM25 build and a
    // single query embedding per question — the expensive parts (reading and
    // tokenizing 43k bodies, GPU embedding) happen once, not once per mode.
    let modes: Vec<String> = if args.baseline == "hybrid" && args.fusion == "all" {
        vec![
            "divide_total".into(),
            "present_weight".into(),
            "rrf".into(),
        ]
    } else {
        vec![args.fusion.clone()]
    };

    let mut outcomes_by_mode: Vec<Vec<Outcome>> = modes.iter().map(|_| Vec::new()).collect();
    let mut miss_diags: Vec<MissDiag> = Vec::new();
    for q in &gold_file.questions {
        // Resolve gold stub ids from (path, line).
        let mut gold: HashSet<String> = HashSet::new();
        for e in &q.expected {
            let line_starts = geom_cache.entry(e.path.clone()).or_insert_with(|| {
                let abs = args.source_root.join(&e.path);
                match std::fs::read(&abs) {
                    Ok(c) => line_start_offsets(&c),
                    Err(err) => {
                        eprintln!("warn: gold path unreadable {}: {}", abs.display(), err);
                        Vec::new()
                    }
                }
            });
            let byte = byte_offset_of_line(line_starts, e.line);
            let geom = edges.stub_geometry(&e.path)?;
            if let Some(stub_id) = stub_for_byte(&geom, byte) {
                gold.insert(stub_id.0);
            } else {
                eprintln!("warn: no stub covers {}:{} for {}", e.path, e.line, q.id);
            }
        }

        // Shared per-question work: embed the query and search each retriever once.
        let qe = embedder.embed_query(vec![q.question.as_str()])?;
        let query_embedding = qe.into_iter().next().context("no query embedding")?;
        let semantic_hits = index.search(&query_embedding, args.pool);
        let bm25_hits: Vec<(StubId, f32)> = match &bm25 {
            Some(bm) => bm.search(&q.question, args.pool),
            None => Vec::new(),
        };

        for (mi, mode) in modes.iter().enumerate() {
            let hits = if bm25.is_some() {
                fuse_hybrid(&semantic_hits, &bm25_hits, args.pool, mode)
            } else {
                semantic_hits.clone()
            };
            let base_ranked: Vec<String> = hits.iter().map(|(id, _)| id.0.clone()).collect();
            let base_rank = first_rank(&base_ranked, &gold);

            // Diagnosis is fusion-independent in its decomposition; emit it once,
            // keyed off the primary (first) mode.
            if args.diagnose && mi == 0 && base_rank.map_or(true, |r| r > args.miss_cutoff) {
                miss_diags.push(diagnose_miss(
                    q,
                    &gold,
                    base_rank,
                    &semantic_hits,
                    &bm25_hits,
                    &store,
                ));
            }

            // --- expansion: admit graph neighbors of the top seed_k seeds ---
            let mut seeds = Vec::with_capacity(args.seed_k.min(hits.len()));
            for (id, score) in hits.iter().take(args.seed_k) {
                if let Ok(stub) = store.get_stub(id) {
                    seeds.push(ScoredStub {
                        stub,
                        score: *score,
                    });
                }
            }
            let seed_ids: Vec<StubId> = seeds.iter().map(|s| s.stub.id.clone()).collect();
            let neighbors = edges.neighbors(&seed_ids, &cfg.relations)?;
            let planned = plan_expansion(&seeds, &neighbors, &cfg);

            let via_by_id: HashMap<&str, &str> = planned
                .iter()
                .map(|p| (p.id.0.as_str(), p.via_relation.as_str()))
                .collect();
            let exp_ranked = match args.merge.as_str() {
                "discounted" => merge_discounted(&hits, &planned),
                _ => merge_adjacent(&base_ranked, &planned),
            };
            let exp_rank = first_rank(&exp_ranked, &gold);

            let via_relation = match (base_rank, exp_rank) {
                (b, Some(er)) if b.map_or(true, |br| er < br) => exp_ranked
                    .get(er - 1)
                    .and_then(|id| via_by_id.get(id.as_str()))
                    .map(|s| s.to_string()),
                _ => None,
            };

            outcomes_by_mode[mi].push(Outcome {
                id: q.id.clone(),
                qtype: q.qtype.clone(),
                base_rank,
                exp_rank,
                via_relation,
            });
        }
    }

    for (mi, mode) in modes.iter().enumerate() {
        println!("\nbaseline retriever: {} (fusion: {})", args.baseline, mode);
        report(&outcomes_by_mode[mi], &recall_ks);
    }
    if args.diagnose {
        print_miss_diagnosis(&miss_diags, args.miss_cutoff);
    }
    Ok(())
}

fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

/// 1-indexed position of the first id in `ranked` that is a gold id.
fn first_rank(ranked: &[String], gold: &HashSet<String>) -> Option<usize> {
    ranked.iter().position(|id| gold.contains(id)).map(|p| p + 1)
}

/// 1-indexed rank of the first gold stub in a `(StubId, score)` list.
fn rank_of_gold(hits: &[(StubId, f32)], gold: &HashSet<String>) -> Option<usize> {
    hits.iter()
        .position(|(id, _)| gold.contains(&id.0))
        .map(|p| p + 1)
}

/// Decompose one baseline miss. Characterizes the gold chunk that best overlaps
/// the query lexically (the most findable one): if even that shares no tokens,
/// the miss is a genuine query↔code register mismatch, not just a low rank.
fn diagnose_miss(
    q: &Question,
    gold: &HashSet<String>,
    hybrid_rank: Option<usize>,
    semantic_hits: &[(StubId, f32)],
    bm25_hits: &[(StubId, f32)],
    store: &SqliteStubStore,
) -> MissDiag {
    let query_tokens: HashSet<String> = bm25_tokenize(&q.question).into_iter().collect();

    let mut gold_path = String::new();
    let mut overlap_count = 0usize;
    let mut shared_sample: Vec<String> = Vec::new();
    let mut summary_len = 0usize;
    let mut body_token_count = 0usize;
    let mut best_overlap: Option<usize> = None;
    for gid in gold {
        let id = StubId(gid.clone());
        let (stub, body) = match (store.get_stub(&id), store.get_content(&id)) {
            (Ok(s), Ok(b)) => (s, b),
            _ => continue,
        };
        let body_tokens = bm25_tokenize(&body);
        let doc_text = format!("{} {} {}", stub.path, stub.summary, body);
        let doc_set: HashSet<String> = bm25_tokenize(&doc_text).into_iter().collect();
        let mut shared: Vec<String> = query_tokens.intersection(&doc_set).cloned().collect();
        shared.sort();
        if best_overlap.is_none_or(|b| shared.len() > b) {
            best_overlap = Some(shared.len());
            overlap_count = shared.len();
            shared_sample = shared.into_iter().take(8).collect();
            summary_len = stub.summary.trim().len();
            body_token_count = body_tokens.len();
            gold_path = stub.path;
        }
    }

    MissDiag {
        qid: q.id.clone(),
        qtype: q.qtype.clone(),
        gold_path,
        hybrid_rank,
        cosine_rank: rank_of_gold(semantic_hits, gold),
        bm25_rank: rank_of_gold(bm25_hits, gold),
        query_token_count: query_tokens.len(),
        overlap_count,
        shared_sample,
        summary_len,
        body_token_count,
    }
}

/// One-line cause label per the decomposition. The half-found cases are the
/// actionable fusion bugs; zero-overlap is the register dead-end that motivated
/// query/document expansion; "buried" with overlap is a scoring problem.
fn miss_cause(d: &MissDiag, cutoff: usize) -> &'static str {
    let found = |r: Option<usize>| r.is_some_and(|r| r <= cutoff);
    if d.summary_len < 15 && d.overlap_count == 0 {
        "thin stub + register dead-end"
    } else if found(d.cosine_rank) && found(d.bm25_rank) {
        "both halves found it — fusion drowned it"
    } else if found(d.bm25_rank) {
        "BM25 found it — cosine + fusion buried it"
    } else if found(d.cosine_rank) {
        "cosine found it — BM25 + fusion buried it"
    } else if d.overlap_count == 0 {
        "register dead-end (no shared tokens)"
    } else {
        "findable but buried (ranking/scoring)"
    }
}

fn print_miss_diagnosis(diags: &[MissDiag], cutoff: usize) {
    println!("\n## both-miss diagnosis (baseline rank > {cutoff})\n");
    if diags.is_empty() {
        println!("  no misses past the cutoff.");
        return;
    }
    for d in diags {
        let fmt = |r: Option<usize>| r.map_or("miss".to_string(), |r| r.to_string());
        println!(
            "{:<34} {:<11} gold={}",
            d.qid, d.qtype, d.gold_path
        );
        println!(
            "    rank: hybrid={:<5} cosine={:<5} bm25={:<5}   cause: {}",
            fmt(d.hybrid_rank),
            fmt(d.cosine_rank),
            fmt(d.bm25_rank),
            miss_cause(d, cutoff),
        );
        println!(
            "    overlap: {}/{} query tokens   gold-text: summary={}c body={}tok   shared={:?}",
            d.overlap_count,
            d.query_token_count,
            d.summary_len,
            d.body_token_count,
            d.shared_sample,
        );
    }
}

/// Fuse the semantic and lexical rankings. `divide_total` mirrors caw-server's
/// live `fuse_hybrid` (the incumbent); `present_weight` and `rrf` are the two
/// alternatives we are re-adjudicating on the larger independent set.
fn fuse_hybrid(
    semantic: &[(StubId, f32)],
    bm25: &[(StubId, f32)],
    top_k: usize,
    mode: &str,
) -> Vec<(StubId, f32)> {
    let mut combined: HashMap<StubId, f32> = HashMap::new();

    if mode == "rrf" {
        // Rank-based: weight / (RRF_K + rank), summed over the lists an item is
        // in. Discards score magnitude entirely.
        for (list, weight) in [
            (semantic, HYBRID_SEMANTIC_WEIGHT),
            (bm25, HYBRID_KEYWORD_WEIGHT),
        ] {
            for (rank, (id, _)) in list.iter().enumerate() {
                *combined.entry(id.clone()).or_default() += weight / (RRF_K + (rank + 1) as f32);
            }
        }
        let mut fused: Vec<(StubId, f32)> = combined.into_iter().collect();
        // Deterministic: score desc, then stub_id asc, so equal-score ties don't
        // ride on HashMap iteration order (which made identical runs wobble ~0.01).
        fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0 .0.cmp(&b.0 .0)));
        fused.truncate(top_k);
        return fused;
    }

    // Min-max normalized weighted sum. The divisor distinguishes the modes:
    // `divide_total` always divides by the full weight (so a single-half hit is
    // capped at its own weight — the agreement reward); `present_weight` divides
    // by the weight of the lists the item actually appears in (rescues single-half
    // hits but discards that reward).
    let mut present: HashMap<StubId, f32> = HashMap::new();
    for (list, weight) in [
        (semantic, HYBRID_SEMANTIC_WEIGHT),
        (bm25, HYBRID_KEYWORD_WEIGHT),
    ] {
        if list.is_empty() {
            continue;
        }
        let max = list.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let min = list.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);
        let range = (max - min).max(f32::EPSILON);
        for (id, score) in list {
            *combined.entry(id.clone()).or_default() += ((score - min) / range) * weight;
            *present.entry(id.clone()).or_default() += weight;
        }
    }
    let total_weight = HYBRID_SEMANTIC_WEIGHT + HYBRID_KEYWORD_WEIGHT;
    let mut fused: Vec<(StubId, f32)> = combined
        .into_iter()
        .map(|(id, score)| {
            let denom = match mode {
                "present_weight" => present
                    .get(&id)
                    .copied()
                    .unwrap_or(total_weight)
                    .max(f32::EPSILON),
                _ => total_weight, // divide_total (incumbent)
            };
            (id, score / denom)
        })
        .collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused.truncate(top_k);
    fused
}

/// Insert each admitted neighbor immediately after the seed that pulled it in,
/// so a structurally-linked chunk inherits its seed's rank rather than a
/// globally-discounted one. A neighbor reachable from several seeds is placed
/// after the highest-ranked (first-seen) seed.
fn merge_adjacent(base_ranked: &[String], planned: &[PlannedNeighbor]) -> Vec<String> {
    let mut by_seed: HashMap<&str, Vec<&PlannedNeighbor>> = HashMap::new();
    for p in planned {
        by_seed.entry(p.via_seed.0.as_str()).or_default().push(p);
    }
    for v in by_seed.values_mut() {
        v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }
    let mut placed: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for id in base_ranked {
        if placed.insert(id.clone()) {
            out.push(id.clone());
        }
        if let Some(ns) = by_seed.get(id.as_str()) {
            for n in ns {
                if placed.insert(n.id.0.clone()) {
                    out.push(n.id.0.clone());
                }
            }
        }
    }
    out
}

/// Re-rank pool + admitted neighbors purely by score. Because a neighbor scores
/// `seed * discount`, this can only place neighbors below the pool — kept for
/// comparison against `adjacent`.
fn merge_discounted(hits: &[(StubId, f32)], planned: &[PlannedNeighbor]) -> Vec<String> {
    let mut merged: Vec<(String, f32)> = hits.iter().map(|(id, s)| (id.0.clone(), *s)).collect();
    for p in planned {
        merged.push((p.id.0.clone(), p.score));
    }
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged.into_iter().map(|(id, _)| id).collect()
}

fn recall_at(rank: Option<usize>, k: usize) -> bool {
    matches!(rank, Some(r) if r <= k)
}

fn mrr(rank: Option<usize>) -> f64 {
    rank.map_or(0.0, |r| 1.0 / r as f64)
}

fn report(outcomes: &[Outcome], recall_ks: &[usize]) {
    println!("\n# graph-expansion retrieval eval\n");
    println!(
        "{:<34} {:<11} {:>5} {:>5} {:>7}  {}",
        "id", "type", "base", "exp", "Δrank", "via"
    );
    for o in outcomes {
        let b = o.base_rank.map_or("miss".into(), |r| r.to_string());
        let e = o.exp_rank.map_or("miss".into(), |r| r.to_string());
        let d = match (o.base_rank, o.exp_rank) {
            (Some(br), Some(er)) => format!("{:+}", br as i64 - er as i64),
            (None, Some(_)) => "+new".into(),
            _ => "·".into(),
        };
        println!(
            "{:<34} {:<11} {:>5} {:>5} {:>7}  {}",
            o.id,
            o.qtype,
            b,
            e,
            d,
            o.via_relation.as_deref().unwrap_or("")
        );
    }

    // Aggregates overall and per distinct type (in first-seen order).
    let mut types: Vec<&str> = Vec::new();
    for o in outcomes {
        if !types.contains(&o.qtype.as_str()) {
            types.push(o.qtype.as_str());
        }
    }
    let mut groups: Vec<(&str, Vec<&Outcome>)> = vec![("all", outcomes.iter().collect())];
    for t in types {
        groups.push((t, outcomes.iter().filter(|o| o.qtype == t).collect()));
    }

    println!("\n## aggregates\n");
    for (name, group) in &groups {
        if group.is_empty() {
            continue;
        }
        let n = group.len() as f64;
        let improved = group
            .iter()
            .filter(|o| match (o.base_rank, o.exp_rank) {
                (Some(b), Some(e)) => e < b,
                (None, Some(_)) => true,
                _ => false,
            })
            .count();
        let regressed = group
            .iter()
            .filter(|o| match (o.base_rank, o.exp_rank) {
                (Some(b), Some(e)) => e > b,
                (Some(_), None) => true,
                _ => false,
            })
            .count();
        let base_mrr: f64 = group.iter().map(|o| mrr(o.base_rank)).sum::<f64>() / n;
        let exp_mrr: f64 = group.iter().map(|o| mrr(o.exp_rank)).sum::<f64>() / n;
        println!(
            "[{}] n={}  improved={}  regressed={}  MRR {:.3} -> {:.3} ({:+.3})",
            name,
            group.len(),
            improved,
            regressed,
            base_mrr,
            exp_mrr,
            exp_mrr - base_mrr
        );
        for k in recall_ks {
            let br = group.iter().filter(|o| recall_at(o.base_rank, *k)).count() as f64 / n;
            let er = group.iter().filter(|o| recall_at(o.exp_rank, *k)).count() as f64 / n;
            println!(
                "    recall@{:<3} {:.3} -> {:.3} ({:+.3})",
                k,
                br,
                er,
                er - br
            );
        }
    }

    // Rescue: of the questions cosine fails (gold not in top-10), how many does
    // expansion lift into the top-10? This is the population expansion targets;
    // the aggregate above is diluted by questions cosine already nails.
    const RESCUE_CUTOFF: usize = 10;
    let failed: Vec<&Outcome> = outcomes
        .iter()
        .filter(|o| !recall_at(o.base_rank, RESCUE_CUTOFF))
        .collect();
    let rescued = failed
        .iter()
        .filter(|o| recall_at(o.exp_rank, RESCUE_CUTOFF))
        .count();
    println!("\n## rescue (gold outside baseline top-{RESCUE_CUTOFF})\n");
    println!(
        "  cosine-failed: {}  rescued-into-top-{}: {}",
        failed.len(),
        RESCUE_CUTOFF,
        rescued
    );
    for o in &failed {
        let b = o.base_rank.map_or("miss".into(), |r| r.to_string());
        let e = o.exp_rank.map_or("miss".into(), |r| r.to_string());
        println!(
            "    {:<34} {:<10} base={:<5} exp={:<5} via={}",
            o.id,
            o.qtype,
            b,
            e,
            o.via_relation.as_deref().unwrap_or("")
        );
    }

    // Edge-kind attribution among improvements.
    let mut attr: HashMap<&str, usize> = HashMap::new();
    for o in outcomes {
        if let Some(rel) = &o.via_relation {
            *attr.entry(rel.as_str()).or_default() += 1;
        }
    }
    if !attr.is_empty() {
        let mut rows: Vec<_> = attr.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n## improvement attribution (edge kind that admitted the gold)\n");
        for (rel, c) in rows {
            println!("  {:<14} {}", rel, c);
        }
    }
}
