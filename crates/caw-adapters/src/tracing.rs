//! Append-only JSONL trace of everything the framework sends to and receives
//! from an LLM.
//!
//! `TraceSink` is a shared, thread-safe handle to one trace file. Hand the
//! same sink to `TracingAdapter` (which logs the raw `CompletionRequest` /
//! `CompletionResponse` pairs) and to the orchestrator (which can log
//! retrieval, probe, load, and eviction events against the same stream).
//! Events are interleaved in chronological order, one JSON object per line.
//!
//! The sink is intentionally dumb: it doesn't own a schema, doesn't validate,
//! doesn't buffer beyond what the OS does. Every line it writes is a
//! `serde_json::Value` — callers decide the shape. A minimum-useful event
//! includes `"t"` (unix millis) and `"event"` (string tag); the helper
//! `TraceSink::log` adds those automatically.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
};

/// Shared append-only sink for JSONL trace events. Clone-to-share; the
/// underlying file handle is behind an `Arc<Mutex<_>>`.
#[derive(Clone)]
pub struct TraceSink {
    inner: Arc<Mutex<File>>,
    seq: Arc<AtomicU64>,
}

impl TraceSink {
    /// Open a trace sink at `path`, creating the file (and parent dirs) if
    /// missing. Writes are appended, so concatenating multiple sessions into
    /// one path is safe.
    pub fn open(path: impl AsRef<Path>) -> CawResult<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(CawError::from)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(CawError::from)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(file)),
            seq: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Open a sink from the `CAW_TRACE_FILE` env var, or `None` if unset.
    /// Convenience for wiring in caw-bench and caw-cli without each one
    /// re-implementing the env-var dance.
    pub fn from_env() -> CawResult<Option<Self>> {
        match std::env::var("CAW_TRACE_FILE") {
            Ok(p) if !p.is_empty() => Self::open(&p).map(Some),
            _ => Ok(None),
        }
    }

    /// Append a JSON event. `"t"` (unix millis) and `"seq"` (monotonic)
    /// are injected automatically; callers add `"event"` and whatever
    /// payload they want.
    ///
    /// On I/O failure the error is swallowed and a line with
    /// `"event":"trace_error"` is attempted. Tracing must never break
    /// the thing being traced — if the sink is wedged, the run continues.
    pub fn log(&self, mut value: serde_json::Value) {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);

        if let serde_json::Value::Object(ref mut map) = value {
            map.insert("t_ms".to_string(), serde_json::json!(t));
            map.insert("seq".to_string(), serde_json::json!(seq));
        }

        let line = match serde_json::to_string(&value) {
            Ok(s) => s,
            Err(_) => return,
        };

        let Ok(mut file) = self.inner.lock() else {
            return;
        };
        let _ = writeln!(file, "{line}");
        let _ = file.flush();
    }
}

/// Wraps any `ModelAdapter`, logging each `complete()` call as a paired
/// `llm_request` + `llm_response` (or `llm_error`) event. Model name,
/// capabilities, and provenance format pass through unchanged.
pub struct TracingAdapter<A: ModelAdapter> {
    inner: A,
    sink: TraceSink,
}

impl<A: ModelAdapter> TracingAdapter<A> {
    pub fn new(inner: A, sink: TraceSink) -> Self {
        Self { inner, sink }
    }

    pub fn inner(&self) -> &A {
        &self.inner
    }
}

impl<A: ModelAdapter> ModelAdapter for TracingAdapter<A> {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.inner.capabilities()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        self.sink.log(serde_json::json!({
            "event": "llm_request",
            "model": self.inner.model_name(),
            "request": request_to_json(&req),
        }));

        let started = SystemTime::now();
        let result = self.inner.complete(req);
        let latency_ms = started.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);

        match &result {
            Ok(resp) => self.sink.log(serde_json::json!({
                "event": "llm_response",
                "model": self.inner.model_name(),
                "latency_ms": latency_ms,
                "answer": resp.answer,
            })),
            Err(e) => self.sink.log(serde_json::json!({
                "event": "llm_error",
                "model": self.inner.model_name(),
                "latency_ms": latency_ms,
                "error": e.to_string(),
            })),
        }

        result
    }
}

/// Serialize `CompletionRequest` without requiring `Serialize` on caw-core's
/// struct. Keeps the core types lean and lets this module evolve the trace
/// shape independently.
fn request_to_json(req: &CompletionRequest) -> serde_json::Value {
    let fragments: Vec<serde_json::Value> = req
        .workspace_fragments
        .iter()
        .map(|f| {
            serde_json::json!({
                "stub_id": f.stub_id.0,
                "content": f.content,
                "locator": {
                    "source": f.locator.source,
                    "locator": f.locator.locator,
                },
                "tokens": f.tokens,
            })
        })
        .collect();

    serde_json::json!({
        "system": req.system,
        "user": req.user,
        "workspace_fragments": fragments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use caw_core::{Locator, RecallFragment, StubId};

    // Ensures the sink serializes events in order and injects t_ms/seq.
    #[test]
    fn sink_appends_ordered_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let sink = TraceSink::open(&path).unwrap();

        sink.log(serde_json::json!({"event": "a", "v": 1}));
        sink.log(serde_json::json!({"event": "b", "v": 2}));

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let a: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let b: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(a["event"], "a");
        assert_eq!(b["event"], "b");
        assert_eq!(a["seq"].as_u64().unwrap() + 1, b["seq"].as_u64().unwrap());
    }

    #[test]
    fn tracing_adapter_logs_request_response_pair() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let sink = TraceSink::open(&path).unwrap();

        let adapter = TracingAdapter::new(crate::MockAdapter::new("mock", false), sink);
        let req = CompletionRequest {
            system: "sys".to_string(),
            user: "hello".to_string(),
            workspace_fragments: vec![RecallFragment {
                stub_id: StubId("s_1".to_string()),
                content: "fragment text".to_string(),
                locator: Locator::full("test.md"),
                tokens: 3,
                mtime_unix_secs: 0,
            }],
            workspace_guidance: Vec::new(),
        };
        let resp = adapter.complete(req).unwrap();
        assert!(resp.answer.contains("hello"));

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let req_line: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let resp_line: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(req_line["event"], "llm_request");
        assert_eq!(req_line["request"]["user"], "hello");
        assert_eq!(
            req_line["request"]["workspace_fragments"][0]["stub_id"],
            "s_1"
        );
        assert_eq!(resp_line["event"], "llm_response");
    }
}
