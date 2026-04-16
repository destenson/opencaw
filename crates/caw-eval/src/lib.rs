use caw_core::RecallFragment;

#[derive(Debug, Clone, Copy)]
pub struct RecallMetrics {
    pub recall_at_k: f32,
    pub precision_at_k: f32,
}

pub fn evaluate_recall(
    expected_stub_ids: &[String],
    retrieved: &[RecallFragment],
    k: usize,
) -> RecallMetrics {
    let top = retrieved.iter().take(k).collect::<Vec<_>>();
    let hits = top
        .iter()
        .filter(|f| expected_stub_ids.contains(&f.stub_id.0))
        .count();

    let recall = if expected_stub_ids.is_empty() {
        1.0
    } else {
        hits as f32 / expected_stub_ids.len() as f32
    };

    let precision = if top.is_empty() {
        1.0
    } else {
        hits as f32 / top.len() as f32
    };

    RecallMetrics {
        recall_at_k: recall,
        precision_at_k: precision,
    }
}
