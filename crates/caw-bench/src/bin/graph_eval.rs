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
    GraphEdgeStore,
};
use caw_index::{CandleEmbeddingProvider, HnswVectorIndex, SqliteStubStore};
use clap::Parser;
use serde::Deserialize;

const DEFAULT_RELATIONS: &str = "calls,imports_from,implements,inherits,method,contains";

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

    /// Number of semantic seeds retrieved per query.
    #[arg(long, default_value_t = 30)]
    top_k: usize,

    /// Edge relations to expand along.
    #[arg(long, default_value = DEFAULT_RELATIONS)]
    relations: String,

    /// Score multiplier applied to a seed's score when scoring its neighbor.
    #[arg(long, default_value_t = 0.5)]
    discount: f32,

    /// Max neighbors admitted per query.
    #[arg(long, default_value_t = 8)]
    max_neighbors: usize,

    /// Recall@k cutoffs to report (comma-separated).
    #[arg(long, default_value = "1,5,10")]
    recall_k: String,
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
    let args = Args::parse();
    let relations: Vec<String> = split_csv(&args.relations);
    let recall_ks: Vec<usize> = split_csv(&args.recall_k)
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    // --- load retrieval stack ---
    let mut embedder = CandleEmbeddingProvider::bge_small().context("init bge-small (candle)")?;
    let dim = embedder.dimension();
    let store = SqliteStubStore::new(&args.index.to_string_lossy(), dim)
        .with_context(|| format!("open index {}", args.index.display()))?;
    let all = store.all_embeddings().context("read embeddings")?;
    anyhow::ensure!(!all.is_empty(), "index {} is empty", args.index.display());
    let mut index = HnswVectorIndex::new();
    for (id, emb) in &all {
        index.add(id.clone(), emb.clone());
    }
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

    let mut outcomes = Vec::new();
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

        // --- baseline: semantic top-k ---
        let qe = embedder.embed_query(vec![q.question.as_str()])?;
        let query_embedding = qe.into_iter().next().context("no query embedding")?;
        let hits = index.search(&query_embedding, args.top_k);
        let base_ranked: Vec<String> = hits.iter().map(|(id, _)| id.0.clone()).collect();
        let base_rank = first_rank(&base_ranked, &gold);

        // --- expansion: admit graph neighbors of the seeds ---
        let mut seeds = Vec::with_capacity(hits.len());
        for (id, score) in &hits {
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

        // Merge seeds + admitted neighbors by score, descending.
        let mut merged: Vec<(String, f32)> =
            hits.iter().map(|(id, s)| (id.0.clone(), *s)).collect();
        let via_by_id: HashMap<&str, &str> = planned
            .iter()
            .map(|p| (p.id.0.as_str(), p.via_relation.as_str()))
            .collect();
        for p in &planned {
            merged.push((p.id.0.clone(), p.score));
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let exp_ranked: Vec<String> = merged.into_iter().map(|(id, _)| id).collect();
        let exp_rank = first_rank(&exp_ranked, &gold);

        // Attribute an improvement to the relation that admitted the gold stub.
        let via_relation = match (base_rank, exp_rank) {
            (b, Some(er)) if b.map_or(true, |br| er < br) => exp_ranked
                .get(er - 1)
                .and_then(|id| via_by_id.get(id.as_str()))
                .map(|s| s.to_string()),
            _ => None,
        };

        outcomes.push(Outcome {
            id: q.id.clone(),
            qtype: q.qtype.clone(),
            base_rank,
            exp_rank,
            via_relation,
        });
    }

    report(&outcomes, &recall_ks);
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

    // Aggregates overall and per type.
    let groups: [(&str, Vec<&Outcome>); 3] = [
        ("all", outcomes.iter().collect()),
        (
            "structural",
            outcomes.iter().filter(|o| o.qtype == "structural").collect(),
        ),
        (
            "definition",
            outcomes.iter().filter(|o| o.qtype == "definition").collect(),
        ),
    ];

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
