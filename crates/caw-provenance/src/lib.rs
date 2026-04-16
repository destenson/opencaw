use caw_core::{ProvenanceStore, RecallFragment};

#[derive(Debug, Default, Clone)]
pub struct InMemoryProvenanceStore {
    records: Vec<RecallFragment>,
}

impl ProvenanceStore for InMemoryProvenanceStore {
    fn record(&mut self, fragment: RecallFragment) {
        self.records.push(fragment);
    }

    fn all(&self) -> Vec<RecallFragment> {
        self.records.clone()
    }
}
