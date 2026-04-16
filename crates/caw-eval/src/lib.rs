pub mod metrics;
pub mod session;

// Re-export the original public API so existing consumers don't break
pub use metrics::{
    ContextEfficiency, CooperationMetrics, FalseRecallMetrics, HysteresisAnalysis,
    RecallObservation, term_overlap,
};
pub use metrics::{RecallMetrics, evaluate_recall};
pub use session::{SessionEvaluator, SessionEvaluatorBuilder};
