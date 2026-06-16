use caw_core::{CawResult, ContentKind, Stub, StubId};
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
        self.write_text(text)
    }

    /// Write a turn where the model failed to produce a valid answer. The
    /// `sample` is stored for human inspection but is prefixed with a sentinel
    /// so the history-injection path can identify and skip it. Subsequent
    /// sessions will see a one-line placeholder instead of the truncated prose.
    pub fn write_degenerate_turn(&mut self, turn: usize, user: &str, sample: &str) -> CawResult<String> {
        let text = format!(
            "## Turn {turn}\n\n[User]: {user}\n\n[Assistant]: [DEGENERATE] {sample}\n\n---\n\n"
        );
        self.write_text(text)
    }

    fn write_text(&mut self, text: String) -> CawResult<String> {
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

}

/// Maximum characters to retain from a prior-session assistant response when
/// compressing for in-memory HNSW indexing. This targets ≤50 tokens per turn,
/// keeping the embedding signal (topic + key conclusions) while preventing full
/// model responses (~400 tokens each) from outcompeting workspace stubs.
const MAX_PRIOR_ASSISTANT_CHARS: usize = 300;

/// Collect (stub, embed_text) pairs from all previous session `.md` files in
/// `session_dir` without inserting them into any retriever or store. Callers
/// embed and index these in-memory only, so prior session content never leaks
/// into the persistent SQLite store and cannot propagate hallucinations across
/// sessions via the authoritative index path.
///
/// Prior session content is indexed at the turn level with assistant responses
/// compressed to ≤50 tokens. Full model output (~400 tokens/turn) would
/// outcompete workspace stubs on any architecture query because it contains the
/// same vocabulary in polished prose — the compressed version retains the topic
/// signal needed for session continuity without flooding the retrieval pool.
pub fn collect_previous_stubs(
    session_dir: &Path,
    current_path: &Path,
) -> Vec<(Stub, String)> {
    let entries = match std::fs::read_dir(session_dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(dir = %session_dir.display(), error = %e, "could not read session dir");
            return Vec::new();
        }
    };

    let mut result = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if p == current_path {
            continue;
        }
        let content = match std::fs::read_to_string(&p) {
            Ok(c) => c,
            Err(e) => {
                warn!(file = %p.display(), error = %e, "failed to read session file");
                continue;
            }
        };
        let mtime = std::fs::metadata(&p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session")
            .to_string();
        let path_str = p.to_string_lossy().into_owned();

        let turns = parse_session_turns(&content);
        debug!(file = %p.display(), turns = turns.len(), "collected prior session turns");

        for (turn_idx, (user, assistant)) in turns.into_iter().enumerate() {
            // Degenerate turns (None) are represented by a one-line placeholder
            // so subsequent sessions know the model failed without receiving the
            // truncated mid-sentence sample as if it were a real answer.
            let embed_text = match assistant {
                None => format!("User: {user}\nAssistant: [turn skipped — model failed to produce a valid response]"),
                Some(ref a) => {
                    let compressed = compress_assistant(a);
                    if compressed.is_empty() {
                        format!("User: {user}")
                    } else {
                        format!("User: {user}\nAssistant: {compressed}")
                    }
                }
            };
            let token_estimate = embed_text.len() / 4;
            let stub = Stub {
                id: StubId(format!("prior-{stem}:turn-{}", turn_idx + 1)),
                path: path_str.clone(),
                token_estimate,
                kind: ContentKind::Markdown,
                summary: format!(
                    "Prior session turn {}: {}",
                    turn_idx + 1,
                    user.chars().take(80).collect::<String>()
                ),
                outline: Vec::new(),
                content_hash: String::new(),
                mtime_unix_secs: mtime,
                byte_offset: 0,
                byte_length: 0,
                chunk_total: 1,
                consolidation_notes: Vec::new(),
            };
            result.push((stub, embed_text));
        }
    }
    result
}

/// Sentinel prefix written by `write_degenerate_turn` to mark turns where
/// the model failed to produce a valid response. The history-injection path
/// uses this to replace the truncated degenerate sample with a placeholder.
const DEGENERATE_SENTINEL: &str = "[DEGENERATE] ";

/// Parse a session markdown file into (user_query, Option<assistant_response>) pairs.
/// A `None` assistant response indicates a degenerate turn that should be
/// represented as a placeholder in history rather than injected verbatim.
///
/// Session format written by `SessionFile::write_turn` / `write_degenerate_turn`:
///   `## Turn N\n\n[User]: ...\n\n[Assistant]: ...\n\n---\n\n`
fn parse_session_turns(content: &str) -> Vec<(String, Option<String>)> {
    let mut turns = Vec::new();
    for section in content.split("\n## Turn ") {
        if !section.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            continue;
        }
        let user = extract_field(section, "[User]: ");
        let assistant_raw = extract_field(section, "[Assistant]: ");
        if !user.is_empty() {
            let assistant = if assistant_raw.starts_with(DEGENERATE_SENTINEL) {
                None
            } else if assistant_raw.is_empty() {
                None
            } else {
                Some(assistant_raw)
            };
            turns.push((user, assistant));
        }
    }
    turns
}

/// Extract the content after `marker` up to the next field boundary.
///
/// Stops at whichever comes first: the turn separator `\n\n---`, or the start
/// of another `[Role]:` header (`\n\n[`). The second stop prevents user-field
/// extraction from accidentally including the assistant response when the
/// session format is `[User]: ...\n\n[Assistant]: ...\n\n---`.
fn extract_field(section: &str, marker: &str) -> String {
    let start = match section.find(marker) {
        Some(i) => i + marker.len(),
        None => return String::new(),
    };
    let rest = &section[start..];
    let end = [rest.find("\n\n---"), rest.find("\n\n[")]
        .iter()
        .filter_map(|&p| p)
        .min()
        .unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// Compress an assistant response to ≤MAX_PRIOR_ASSISTANT_CHARS characters by
/// keeping complete sentences from the start. Appends "…" when truncated.
fn compress_assistant(text: &str) -> String {
    if text.len() <= MAX_PRIOR_ASSISTANT_CHARS {
        return text.to_string();
    }
    // Try to cut at a sentence boundary within the character budget
    let budget = MAX_PRIOR_ASSISTANT_CHARS.saturating_sub(1); // reserve space for ellipsis
    let truncation_point = text[..budget.min(text.len())]
        .rfind(". ")
        .map(|i| i + 1) // include the period, exclude the space
        .unwrap_or(budget.min(text.len()));
    format!("{}…", text[..truncation_point].trim_end())
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
