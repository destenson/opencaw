//! Map a Graphify code-structure graph onto an OpenCAW stub index and write the
//! resulting stub-to-stub edges into the index's `stub_edge` sidecar table.
//!
//! Graphify (`graphify extract`) produces `graphify-out/graph.json`: nodes are
//! symbols/files with a `source_file` (relative to the extract root) and a
//! single start line (`"L20"`); edges are typed relationships (`calls`,
//! `imports_from`, ...) between node ids. This tool resolves each node's start
//! line to a byte offset in its source file, finds the chunk-stub whose byte
//! range covers that offset, and emits one stub edge per graph edge whose two
//! endpoints both land on (different) stubs.
//!
//! Graphify-specific parsing lives here, not in the reusable library: the
//! library only knows about generic `StubEdge`s.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use caw_core::StubId;
use caw_index::graph_edges::{
    byte_offset_of_line, line_start_offsets, stub_for_byte, GraphEdgeStore, StubEdge,
};
use clap::Parser;
use serde::Deserialize;
use tracing::{info, warn};

/// Relations imported by default. `references` is excluded: it is by far the
/// largest and vaguest bucket in a Graphify graph, and admitting it would dilute
/// the structural signal. Add it explicitly with `--relations` to measure it.
const DEFAULT_RELATIONS: &str = "calls,imports_from,implements,inherits,method,contains";

#[derive(Parser, Debug)]
#[command(about = "Ingest a Graphify graph.json into an OpenCAW index as stub edges")]
struct Args {
    /// Path to Graphify's graph.json.
    #[arg(long, default_value = "graphify-out/graph.json")]
    graph: PathBuf,

    /// Path to the OpenCAW stub index (SQLite) to write edges into.
    #[arg(long)]
    index: PathBuf,

    /// Filesystem root that graph `source_file` paths are relative to. For a
    /// graph extracted with `graphify extract crates/`, source files resolve
    /// under `crates/`, so pass `--source-root crates`.
    #[arg(long, default_value = ".")]
    source_root: PathBuf,

    /// Comma-separated edge relations to import.
    #[arg(long, default_value = DEFAULT_RELATIONS)]
    relations: String,

    /// Only import edges with confidence EXTRACTED (drop INFERRED/AMBIGUOUS).
    #[arg(long, default_value_t = false)]
    extracted_only: bool,
}

#[derive(Deserialize)]
struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Deserialize)]
struct Node {
    id: String,
    source_file: Option<String>,
    source_location: Option<String>,
}

#[derive(Deserialize)]
struct Edge {
    source: String,
    target: String,
    relation: String,
    #[serde(default)]
    confidence: Option<String>,
    #[serde(default)]
    weight: Option<f64>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let relations: Vec<String> = args
        .relations
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    info!(
        graph = %args.graph.display(),
        index = %args.index.display(),
        source_root = %args.source_root.display(),
        relations = ?relations,
        extracted_only = args.extracted_only,
        "ingesting graphify graph"
    );

    let raw = fs::read_to_string(&args.graph)?;
    let graph: Graph = serde_json::from_str(&raw)?;
    info!(
        nodes = graph.nodes.len(),
        edges = graph.edges.len(),
        "parsed graph.json"
    );

    let mut store = GraphEdgeStore::open(&args.index)?;

    // node id -> (source_file, start line). Nodes without a location can't be
    // placed on a stub; count them so a graph/index mismatch is visible.
    let mut located: HashMap<&str, (&str, usize)> = HashMap::new();
    let mut nodes_no_location = 0usize;
    for n in &graph.nodes {
        match (n.source_file.as_deref(), n.source_location.as_deref().and_then(parse_line)) {
            (Some(file), Some(line)) => {
                located.insert(n.id.as_str(), (file, line));
            }
            _ => nodes_no_location += 1,
        }
    }

    // Group located nodes by file so each source file is read and queried once.
    let mut by_file: HashMap<&str, Vec<(&str, usize)>> = HashMap::new();
    for (id, (file, line)) in &located {
        by_file.entry(file).or_default().push((id, *line));
    }

    // node id -> stub id.
    let mut node_stub: HashMap<&str, StubId> = HashMap::new();
    let mut files_missing_source = 0usize;
    let mut files_missing_in_index = 0usize;
    let mut nodes_unmapped = 0usize;

    for (file, file_nodes) in &by_file {
        let abs = args.source_root.join(file);
        let content = match fs::read(&abs) {
            Ok(c) => c,
            Err(e) => {
                warn!(file, path = %abs.display(), error = %e, "source file unreadable; skipping its nodes");
                files_missing_source += 1;
                continue;
            }
        };
        let line_starts = line_start_offsets(&content);

        let geom = store.stub_geometry(file)?;
        if geom.is_empty() {
            warn!(file, "no stubs for this path in the index; skipping its nodes");
            files_missing_in_index += 1;
            continue;
        }

        for (id, line) in file_nodes {
            let byte = byte_offset_of_line(&line_starts, *line);
            match stub_for_byte(&geom, byte) {
                Some(stub_id) => {
                    node_stub.insert(id, stub_id);
                }
                None => nodes_unmapped += 1,
            }
        }
    }

    info!(
        located = located.len(),
        nodes_no_location,
        mapped = node_stub.len(),
        nodes_unmapped,
        files_missing_source,
        files_missing_in_index,
        "mapped nodes to stubs"
    );

    // Build stub edges, deduped at (src, dst, relation) keeping the max weight.
    let relation_set: std::collections::HashSet<&str> =
        relations.iter().map(|s| s.as_str()).collect();
    let mut deduped: HashMap<(String, String, String), f32> = HashMap::new();
    let mut dropped_relation = 0usize;
    let mut dropped_confidence = 0usize;
    let mut dropped_unmapped = 0usize;
    let mut dropped_self = 0usize;

    for e in &graph.edges {
        if !relation_set.contains(e.relation.as_str()) {
            dropped_relation += 1;
            continue;
        }
        if args.extracted_only && e.confidence.as_deref() != Some("EXTRACTED") {
            dropped_confidence += 1;
            continue;
        }
        let (src, dst) = match (node_stub.get(e.source.as_str()), node_stub.get(e.target.as_str()))
        {
            (Some(s), Some(d)) => (s, d),
            _ => {
                dropped_unmapped += 1;
                continue;
            }
        };
        if src.0 == dst.0 {
            dropped_self += 1;
            continue;
        }
        let weight = e.weight.unwrap_or(1.0) as f32;
        let key = (src.0.clone(), dst.0.clone(), e.relation.clone());
        let slot = deduped.entry(key).or_insert(weight);
        if weight > *slot {
            *slot = weight;
        }
    }

    let edges: Vec<StubEdge> = deduped
        .into_iter()
        .map(|((src, dst, relation), weight)| StubEdge {
            src: StubId(src),
            dst: StubId(dst),
            relation,
            weight,
        })
        .collect();

    let written = store.replace_edges(&edges)?;
    info!(
        written,
        dropped_relation,
        dropped_confidence,
        dropped_unmapped,
        dropped_self,
        "wrote stub edges"
    );

    Ok(())
}

/// Parse a Graphify `source_location` like `"L20"` into a 1-indexed line number.
fn parse_line(loc: &str) -> Option<usize> {
    loc.strip_prefix('L').and_then(|n| n.parse().ok())
}
