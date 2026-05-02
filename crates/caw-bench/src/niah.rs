//! NIAH (needle-in-a-haystack) workload.
//!
//! Produces WorkloadItems where a single paragraph contains a unique,
//! fabricated fact ("the secret authorization code is QUATRAFIN-7719")
//! buried in procedurally-generated filler. The question asks for the
//! fact by its distinctive label. Scoring is deterministic: the model's
//! answer either contains the needle string or it doesn't.

use crate::niah_corpus::{Lcg, generate};
use crate::workload::{Scoring, WorkloadItem};

pub struct NiahConfig {
    /// Number of filler paragraphs around the needle. Larger = harder retrieval.
    pub filler_paragraphs: usize,
    /// Needle position is chosen per-item by the seeded RNG. This sets the
    /// RNG seed base — each item adds its index to this seed so items are
    /// independent but the whole suite is reproducible.
    pub seed: u64,
    /// How many items to produce.
    pub item_count: usize,
}

impl Default for NiahConfig {
    fn default() -> Self {
        Self {
            filler_paragraphs: 80,
            seed: 0xca_0000_0001,
            item_count: 10,
        }
    }
}

pub fn build(config: &NiahConfig) -> Vec<WorkloadItem> {
    // Needle topics deliberately do NOT share vocabulary with filler
    // project / operator / sector names (see niah_corpus::PROJECTS etc.).
    // Otherwise filler paragraphs mentioning the topic beat the needle
    // memo in cosine similarity and the test becomes unanswerable.
    // Needles deliberately avoid credential-shaped framing (authorization
    // codes, clearances, reagent IDs, restricted markings) so that
    // safety-tuned hosted models don't refuse the retrieval task. All
    // topics are benign project metadata: versions, dates, thresholds,
    // frequencies, colors, plate sizes, build IDs.
    let needles: &[(&str, &str, &str)] = &[
        (
            "release_version",
            "QUATRAFIN-7719",
            "the internal release tag for the Bluecapsule documentation snapshot",
        ),
        (
            "migration_date",
            "2028-11-04",
            "the scheduled migration date for the Zephyrglass document archive",
        ),
        (
            "sample_ratio",
            "0.6183",
            "the sampling ratio used by the Starkwood telemetry aggregator",
        ),
        (
            "replica_label",
            "NODE-ALPHA-812",
            "the replica label used by the Portmanteau read cache",
        ),
        (
            "ingest_rate",
            "11.42 units/hour",
            "the steady-state ingest rate for the Quillmark ledger",
        ),
        (
            "region_tag",
            "RX-44-EAST",
            "the region tag assigned to the Nimbus telemetry channel",
        ),
        (
            "checksum_prefix",
            "BLAKE3-8F02C1",
            "the checksum prefix used in the Silvermint content hash field",
        ),
        (
            "delivery_window",
            "ROUTE-M7-SOUTHBOUND",
            "the delivery window label used by the Harrow shipping program",
        ),
        (
            "paint_spec",
            "CLEARANCE-AURIC-3",
            "the paint spec used for the Westgarden enclosure housing",
        ),
        (
            "part_number",
            "REAGENT-NX-47",
            "the part number for the Sandpaperer maintenance kit",
        ),
    ];

    let mut items = Vec::new();
    for i in 0..config.item_count {
        let (label, needle, phrase) = needles[i % needles.len()];
        let seed = config.seed.wrapping_add(i as u64);
        let mut pos_rng = Lcg::new(seed);
        let needle_position = pos_rng.next_range(config.filler_paragraphs.max(1));

        // First line becomes the stub summary under DeterministicSummarizer's
        // plain-text strategy, so it must be the sentence retrieval should
        // match on. Memo boilerplate comes after.
        let needle_memo = format!(
            "This note records {phrase}.\n\n\
             Reference copy for internal documentation.\n\n\
             The value is: {needle}.\n",
        );

        let corpus = generate(
            seed,
            config.filler_paragraphs,
            needle_position,
            &needle_memo,
        );

        items.push(WorkloadItem {
            id: format!("niah_{:03}_{}", i, label),
            question: format!("What is {phrase}? Answer with the exact value."),
            corpus,
            expected_paths: vec!["corpus/memo-needle.txt".to_string()],
            scoring: Scoring::ContainsNeedle {
                needle: needle.to_string(),
            },
        });
    }
    items
}
