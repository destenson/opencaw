//! Sweep driver for `caw-bench`.
//!
//! Reads a TOML config describing fixed params plus a set of axes, computes
//! the Cartesian product, and invokes `caw-bench` once per cell. Each cell's
//! report is written to `<out_dir>/<cell-hash>/report.json` with a sibling
//! `params.json`. A completed cell short-circuits on re-run (resume), so a
//! crash + fix + re-invoke picks up where it left off.
//!
//! Design decisions:
//! - Subprocess per cell, not in-process. Isolates crashes (bad adapter,
//!   OOM, dropped Ollama connection) from the driver itself, and lets the
//!   child inherit the terminal signal group so Ctrl-C tears everything
//!   down.
//! - Fail-fast: first non-zero exit halts the sweep. The offending command
//!   is printed verbatim so the operator can reproduce, fix the root
//!   cause (flag the adapter, add the model, raise a timeout), and resume.
//! - Cell hash covers fixed + chosen axis values, sorted by key. Changing
//!   a fixed value or an axis entry yields new hashes for the affected
//!   cells; unchanged cells still skip on resume.

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "caw-bench-sweep",
    about = "Run caw-bench across a grid of parameters defined in a TOML config"
)]
struct Cli {
    /// Path to the TOML sweep config.
    #[arg(long)]
    config: PathBuf,

    /// Override the `out_dir` key from the config.
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Path to the caw-bench binary. Defaults to a sibling of the current
    /// executable (the same `target/<profile>` directory).
    #[arg(long)]
    caw_bench: Option<PathBuf>,

    /// Print the planned cells and exit without running anything.
    #[arg(long)]
    dry_run: bool,

    /// Re-run every cell even if a report already exists.
    #[arg(long)]
    force: bool,
}

/// A sweep config — `fixed` flags applied everywhere, plus a map of
/// named axes. Each axis holds a list of `Table` entries; one entry per
/// axis is chosen per cell, merged into the fixed params.
///
/// The axis map is a `BTreeMap` so iteration order is deterministic —
/// that matters for cell-hash stability across runs.
#[derive(Debug, Deserialize)]
struct SweepConfig {
    /// Where per-cell output goes. Relative paths resolve against the
    /// config file's parent so the same config works from any cwd.
    out_dir: PathBuf,

    #[serde(default)]
    fixed: toml::Table,

    #[serde(default)]
    axes: BTreeMap<String, Vec<toml::Table>>,
}

/// One concrete parameter assignment to run. Kept as a sorted BTreeMap
/// so JSON serialization is canonical — the hash depends on it.
#[derive(Debug, Serialize)]
struct Cell {
    params: BTreeMap<String, toml::Value>,
}

impl Cell {
    fn hash(&self) -> String {
        let canonical =
            serde_json::to_vec(&self.params).expect("BTreeMap<String, toml::Value> serializes");
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        let digest = hasher.finalize();
        hex_short(&digest)
    }

    /// Render as caw-bench CLI args. Each (key, value) becomes
    /// `--<kebab-key> <value-as-string>`. The caw-bench CLI happens to
    /// have no boolean-valued flags, so we reject booleans loudly rather
    /// than guess at flag-style semantics.
    fn to_args(&self, report_path: &Path) -> Result<Vec<String>> {
        let mut args = Vec::new();
        for (key, value) in &self.params {
            // `out` is controlled by the sweep driver — a cell config
            // can't clobber it.
            if key == "out" {
                continue;
            }
            args.push(format!("--{}", key.replace('_', "-")));
            args.push(toml_scalar_to_string(key, value)?);
        }
        args.push("--out".to_string());
        args.push(report_path.to_string_lossy().into_owned());
        Ok(args)
    }
}

fn hex_short(bytes: &[u8]) -> String {
    // 8 bytes / 16 hex chars — enough to avoid collisions for sweeps up to
    // ~4B cells, short enough to fit comfortably in path names.
    let mut s = String::with_capacity(16);
    for b in &bytes[..8] {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn toml_scalar_to_string(key: &str, value: &toml::Value) -> Result<String> {
    match value {
        toml::Value::String(s) => Ok(s.clone()),
        toml::Value::Integer(i) => Ok(i.to_string()),
        toml::Value::Float(f) => Ok(format!("{f}")),
        toml::Value::Boolean(_) => Err(anyhow!(
            "parameter `{key}` is boolean; caw-bench has no boolean-valued flags. \
             Remove it from the config or pass as `--{key}` via `fixed` only if \
             caw-bench grows a bool flag in the future."
        )),
        other => Err(anyhow!(
            "parameter `{key}` has unsupported TOML type {:?}; only string, \
             integer, and float scalars are accepted",
            other.type_str()
        )),
    }
}

fn plan_cells(cfg: &SweepConfig) -> Vec<Cell> {
    // Axis values are lists; Cartesian product of lists yields cells.
    let axis_names: Vec<&String> = cfg.axes.keys().collect();
    let axis_entries: Vec<&Vec<toml::Table>> = cfg.axes.values().collect();

    // Cold-start product: one empty "base" cell inherits `fixed` params.
    let mut cells: Vec<BTreeMap<String, toml::Value>> = vec![cfg
        .fixed
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()];

    for (name, entries) in axis_names.iter().zip(axis_entries) {
        if entries.is_empty() {
            // An empty axis means "this axis contributes nothing" — drop
            // it rather than erase all cells. Matches TOML intuition
            // when a user comments out every entry.
            eprintln!("sweep: axis `{name}` has no entries, skipping");
            continue;
        }
        let mut next = Vec::with_capacity(cells.len() * entries.len());
        for base in &cells {
            for entry in entries {
                let mut merged = base.clone();
                for (k, v) in entry {
                    // Later axis entries win on collision — documented
                    // as "axes override fixed; later axes override earlier".
                    merged.insert(k.clone(), v.clone());
                }
                next.push(merged);
            }
        }
        cells = next;
    }

    cells.into_iter().map(|params| Cell { params }).collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let config_path = cli.config.canonicalize().with_context(|| {
        format!("sweep config not found: {}", cli.config.display())
    })?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent"))?;
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let cfg: SweepConfig = toml::from_str(&raw)
        .with_context(|| format!("parse {}", config_path.display()))?;

    // Resolve out_dir relative to the config file, unless the CLI override
    // or the config value is already absolute.
    let out_dir = match cli.out_dir.clone().unwrap_or_else(|| cfg.out_dir.clone()) {
        p if p.is_absolute() => p,
        p => config_dir.join(p),
    };
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create out_dir {}", out_dir.display()))?;

    let caw_bench_path = resolve_caw_bench(cli.caw_bench.as_deref())?;

    let cells = plan_cells(&cfg);
    if cells.is_empty() {
        bail!("sweep config produced zero cells (no fixed params and no axes?)");
    }

    eprintln!(
        "sweep: {} cells, out_dir = {}",
        cells.len(),
        out_dir.display()
    );
    eprintln!("sweep: caw-bench = {}", caw_bench_path.display());

    let manifest_path = out_dir.join("manifest.jsonl");
    let mut done = 0usize;
    let mut skipped = 0usize;

    for (idx, cell) in cells.iter().enumerate() {
        let hash = cell.hash();
        let cell_dir = out_dir.join(&hash);
        let report_path = cell_dir.join("report.json");
        let params_path = cell_dir.join("params.json");

        eprintln!(
            "\n[{}/{}] cell {} {}",
            idx + 1,
            cells.len(),
            hash,
            summarize_cell(cell)
        );

        if !cli.force && report_path.exists() {
            eprintln!("  skip — report already at {}", report_path.display());
            skipped += 1;
            continue;
        }

        if cli.dry_run {
            let args = cell.to_args(&report_path)?;
            eprintln!("  would run: {} {}", caw_bench_path.display(), args.join(" "));
            continue;
        }

        std::fs::create_dir_all(&cell_dir)
            .with_context(|| format!("create {}", cell_dir.display()))?;
        let params_json = serde_json::to_string_pretty(&cell.params)
            .context("serialize cell params")?;
        std::fs::write(&params_path, params_json)
            .with_context(|| format!("write {}", params_path.display()))?;

        let args = cell.to_args(&report_path)?;
        eprintln!(
            "  run: {} {}",
            caw_bench_path.display(),
            args.join(" ")
        );

        // Point caw-bench's tracing adapter at a per-cell trace file so
        // every sweep cell gets its own JSONL log. Inherits the rest of
        // the caller's environment, so operators can override via the
        // shell if they want a shared file or none at all.
        let trace_path = cell_dir.join("trace.jsonl");
        let status = Command::new(&caw_bench_path)
            .args(&args)
            .env("CAW_TRACE_FILE", &trace_path)
            .status()
            .with_context(|| {
                format!("spawn {} failed", caw_bench_path.display())
            })?;

        if !status.success() {
            // Leave partial state in place so the operator can inspect
            // cell_dir/params.json to see exactly what was tried. Resume
            // will re-run this cell because report.json is absent.
            bail!(
                "cell {} failed with {:?}. Fix the underlying issue and re-run \
                 the same sweep command — completed cells will be skipped.",
                hash,
                status
            );
        }

        append_manifest(&manifest_path, &hash, cell)?;
        done += 1;
    }

    if cli.dry_run {
        eprintln!("\nsweep: dry-run done, no cells executed");
    } else {
        eprintln!(
            "\nsweep: finished — {} ran, {} skipped (resume hits)",
            done, skipped
        );
    }
    Ok(())
}

fn resolve_caw_bench(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        let canonical = p.canonicalize().with_context(|| {
            format!("--caw-bench path not found: {}", p.display())
        })?;
        return Ok(canonical);
    }
    let current = std::env::current_exe().context("current_exe")?;
    let dir = current
        .parent()
        .ok_or_else(|| anyhow!("current_exe has no parent"))?;
    let exe_name = if cfg!(windows) {
        "caw-bench.exe"
    } else {
        "caw-bench"
    };
    let candidate = dir.join(exe_name);
    if !candidate.exists() {
        bail!(
            "expected caw-bench at {} but it is missing. Build with `cargo build -p caw-bench` \
             or pass --caw-bench explicitly.",
            candidate.display()
        );
    }
    Ok(candidate)
}

fn append_manifest(path: &Path, hash: &str, cell: &Cell) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    #[derive(Serialize)]
    struct Entry<'a> {
        hash: &'a str,
        params: &'a BTreeMap<String, toml::Value>,
    }
    let line = serde_json::to_string(&Entry {
        hash,
        params: &cell.params,
    })
    .context("serialize manifest entry")?;
    writeln!(file, "{}", line).context("append manifest line")?;
    Ok(())
}

fn summarize_cell(cell: &Cell) -> String {
    // Not every param is worth printing — pick the ones that usually
    // differentiate cells. Rest is in params.json.
    let interesting = [
        "workload",
        "answer_model",
        "niah_seed",
        "load_threshold",
        "unload_threshold",
        "max_workspace_tokens",
    ];
    let mut parts = Vec::new();
    for key in interesting {
        if let Some(v) = cell.params.get(key) {
            parts.push(format!("{}={}", key, v));
        }
    }
    parts.join(" ")
}
