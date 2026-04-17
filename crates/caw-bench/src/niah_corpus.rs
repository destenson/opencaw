//! Procedurally-generated filler for needle-in-haystack tests.
//!
//! Rationale: real datasets (Gutenberg, Wikipedia dumps, common Kaggle
//! corpora) are plausibly in every modern model's training set, so using
//! them as "filler" contaminates the test — the model may retrieve the
//! needle from training, not from context. Procedurally-generated text
//! built from a fixed vocabulary and random templates is guaranteed out
//! of distribution and reproducible under a seed.
//!
//! The filler mimics bureaucratic project-status reports: dry, varied,
//! structurally similar to the needle document so retrieval can't trivially
//! distinguish the target by stylistic anomaly. If the caller wants real
//! prose instead, they can supply their own corpus via the harness CLI.

use crate::workload::CorpusDoc;
use caw_core::ContentKind;

/// Deterministic LCG RNG so runs are reproducible without pulling in `rand`.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeef } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants; quality is plenty for template filling.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_range(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.next_range(items.len())]
    }
}

/// Generate `paragraph_count` filler paragraphs plus the needle document,
/// returning them in a stable order with the needle inserted at
/// `needle_position` (clamped to the corpus length). The caller asks a
/// question that only the needle paragraph can answer.
pub fn generate(
    seed: u64,
    paragraph_count: usize,
    needle_position: usize,
    needle_text: &str,
) -> Vec<CorpusDoc> {
    let mut rng = Lcg::new(seed);
    let mut docs: Vec<CorpusDoc> = (0..paragraph_count)
        .map(|i| CorpusDoc {
            path: format!("filler/report-{:04}.txt", i),
            content: filler_paragraph(&mut rng),
            kind: ContentKind::PlainText,
        })
        .collect();

    let needle_doc = CorpusDoc {
        path: "corpus/memo-needle.txt".to_string(),
        content: needle_text.to_string(),
        kind: ContentKind::PlainText,
    };

    let pos = needle_position.min(docs.len());
    docs.insert(pos, needle_doc);
    docs
}

fn filler_paragraph(rng: &mut Lcg) -> String {
    // Each paragraph has 3-6 sentences; each sentence from a template pool.
    let sentence_count = 3 + rng.next_range(4);
    let mut out = String::new();
    for i in 0..sentence_count {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&sentence(rng));
    }
    out
}

fn sentence(rng: &mut Lcg) -> String {
    let template = rng.pick(TEMPLATES);
    let project = rng.pick(PROJECTS);
    let epoch = rng.next_range(999);
    let yield_val = rng.next_range(100);
    let operator = rng.pick(OPERATORS);
    let sector = rng.pick(SECTORS);
    let metric = rng.pick(METRICS);
    let verb = rng.pick(VERBS);

    template
        .replace("{PROJECT}", project)
        .replace("{EPOCH}", &format!("{:03}", epoch))
        .replace("{YIELD}", &format!("0.{:02}", yield_val))
        .replace("{OPERATOR}", operator)
        .replace("{SECTOR}", sector)
        .replace("{METRIC}", metric)
        .replace("{VERB}", verb)
}

const TEMPLATES: &[&str] = &[
    "Project {PROJECT} {VERB} nominal {METRIC} during epoch {EPOCH}.",
    "Operator {OPERATOR} logged {METRIC} readings of {YIELD} in sector {SECTOR}.",
    "The {PROJECT} pipeline {VERB} without incident through checkpoint {EPOCH}.",
    "Sector {SECTOR} reported {YIELD} {METRIC} against the baseline for {PROJECT}.",
    "Epoch {EPOCH} summary: {PROJECT} {VERB}, {METRIC} at {YIELD}, no flags.",
    "Operator {OPERATOR} refreshed {PROJECT} calibration following {METRIC} drift.",
    "Telemetry for {PROJECT} shows {METRIC} holding near {YIELD} across sector {SECTOR}.",
    "Routine audit of sector {SECTOR} confirmed {PROJECT} {VERB} within tolerance.",
    "Checkpoint {EPOCH} closed with {PROJECT} {METRIC} logged at {YIELD}.",
    "Maintenance window for sector {SECTOR} did not affect {PROJECT} {METRIC}.",
    "Operator {OPERATOR} noted {METRIC} variance of {YIELD} during {PROJECT} review.",
    "The {PROJECT} run at epoch {EPOCH} {VERB} with {METRIC} stable in sector {SECTOR}.",
];

const PROJECTS: &[&str] = &[
    "Meridian-7",
    "Quadrat-B",
    "Saltspike",
    "Orbital-Drift-2",
    "Feldspar",
    "Turnstile-9",
    "Beacon-Loop",
    "Ampstrand",
    "Kiln-4",
    "Heliotrope",
    "Sandbed-12",
    "Carbon-Node",
];

const OPERATORS: &[&str] = &[
    "Krill",
    "Marsh",
    "Odrin",
    "Petal",
    "Quilt",
    "Ramsey",
    "Sable",
    "Tiln",
    "Vorn",
    "Wheel",
    "Ystra",
    "Zant",
];

const SECTORS: &[&str] = &[
    "A-1", "A-4", "B-2", "B-9", "C-3", "C-7", "D-5", "D-8", "E-6", "F-11", "G-14", "H-2",
];

const METRICS: &[&str] = &[
    "throughput",
    "retention",
    "calibration offset",
    "saturation",
    "fault rate",
    "buffer occupancy",
    "channel bias",
    "parity delta",
    "drift index",
];

const VERBS: &[&str] = &[
    "held",
    "cycled",
    "settled",
    "stabilized",
    "completed",
    "rebalanced",
    "resumed",
    "tracked",
];
