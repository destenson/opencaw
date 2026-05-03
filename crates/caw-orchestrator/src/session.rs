use caw_core::{CawResult, ContentKind, Retriever};
use caw_ingest::{DocumentIdSet, IngestionPipeline, SourceDocument};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

pub struct SessionFile {
    path: PathBuf,
    file: Option<File>,
}

impl SessionFile {
    pub fn create(path: PathBuf) -> CawResult<Self> {
        Ok(Self { path, file: None })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Format the turn, append to the session file, and return the formatted text.
    pub fn write_turn(&mut self, turn: usize, user: &str, answer: &str) -> CawResult<String> {
        let text = format!("## Turn {turn}\n\n[User]: {user}\n\n[Assistant]: {answer}\n\n---\n\n");
        let file = self.file.get_or_insert(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .map_err(|e| caw_core::CawError::Io(e.to_string()))?,
        );
        file.write_all(text.as_bytes())
            .map_err(|e| caw_core::CawError::Io(e.to_string()))?;
        file.flush()
            .map_err(|e| caw_core::CawError::Io(e.to_string()))?;
        Ok(text)
    }

    /// Ingest all `.md` files in `session_dir` except `current_path` into `retriever`.
    /// Files whose `(path, mtime)` appears in `already_indexed` are skipped — their stubs
    /// are already in the store and the HNSW was built from `all_embeddings()` at startup.
    /// Returns the number of stubs inserted (0 for already-indexed files).
    pub fn load_previous<R: Retriever>(
        session_dir: &Path,
        current_path: &Path,
        pipeline: &IngestionPipeline,
        retriever: &mut R,
        already_indexed: &DocumentIdSet,
    ) -> CawResult<usize> {
        let entries = match std::fs::read_dir(session_dir) {
            Ok(e) => e,
            Err(e) => {
                warn!(dir = %session_dir.display(), error = %e, "could not read session dir");
                return Ok(0);
            }
        };

        let mut count = 0;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            if p == current_path {
                continue;
            }
            if !already_indexed.is_empty() {
                let path_str = p.to_string_lossy().into_owned();
                let mtime = std::fs::metadata(&p)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if already_indexed.contains(&(path_str, mtime)) {
                    debug!(file = %p.display(), "skipping already-indexed session file");
                    continue;
                }
            }
            match load_session_file(&p, pipeline, retriever) {
                Ok(n) => {
                    debug!(file = %p.display(), stubs = n, "loaded previous session");
                    count += n;
                }
                Err(e) => warn!(file = %p.display(), error = %e, "failed to load session file"),
            }
        }
        Ok(count)
    }
}

fn load_session_file<R: Retriever>(
    path: &Path,
    pipeline: &IngestionPipeline,
    retriever: &mut R,
) -> CawResult<usize> {
    let content =
        std::fs::read_to_string(path).map_err(|e| caw_core::CawError::Io(e.to_string()))?;
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let doc = SourceDocument {
        path: path.to_string_lossy().into_owned(),
        content,
        kind: ContentKind::Markdown,
        mtime_unix_secs: mtime,
    };

    let stubs = pipeline.ingest(doc);
    let count = stubs.len();
    for (stub, embed_text) in stubs {
        retriever.insert(stub, embed_text)?;
    }
    Ok(count)
}

pub fn timestamp_str() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Format as YYYYMMDD-HHMMSS using integer arithmetic
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    // Days since epoch → date (good enough for filenames, no leap second precision needed)
    let days = s / 86400;
    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, mo, d, hour, min, sec)
}

// Minimal Gregorian date from days-since-epoch (1970-01-01).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let mut y = 1970u64;
    let mut rem = days;
    loop {
        let leap = is_leap(y);
        let dy = if leap { 366 } else { 365 };
        if rem < dy {
            break;
        }
        rem -= dy;
        y += 1;
    }
    let months = [
        31u64,
        if is_leap(y) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 1u64;
    for &dm in &months {
        if rem < dm {
            break;
        }
        rem -= dm;
        mo += 1;
    }
    (y, mo, rem + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
