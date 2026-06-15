pub mod adapter_factory;
pub mod codeagent;
pub mod concept_mistral_q;
pub mod intent;
pub mod judge;
pub mod niah;
pub mod niah_corpus;
pub mod opencaw;
pub mod report;
pub mod runner;
pub mod shared;
pub mod sysdoc;
pub mod timing;
pub mod workload;

/// Install the process-wide tracing subscriber for the bench binaries.
///
/// Logs go to **stderr** so the report (written to stdout) stays clean and
/// machine-parseable when redirected. An explicit `RUST_LOG` is honored
/// verbatim; otherwise the default quiets chatty dependencies and keeps the
/// app at `info`, so per-call diagnostics emitted at `debug!` (e.g. the candle
/// embedder's per-batch timing) are off unless asked for with
/// `RUST_LOG=caw_index=debug`.
///
/// Idempotent: safe to call from any bin's `main`; a second call is a no-op.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = match std::env::var("RUST_LOG") {
        Ok(existing) if !existing.trim().is_empty() => EnvFilter::new(existing),
        _ => {
            let noise = ["reqwest", "rustls", "globset", "h2", "hyper", "webpki"]
                .map(|s| format!("{s}=warn"))
                .join(",");
            EnvFilter::new(format!("info,{noise}"))
        }
    };

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .try_init();
}
