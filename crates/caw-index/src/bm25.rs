use caw_core::StubId;
use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// BM25 keyword index for term-frequency-based document retrieval.
/// Complements embedding-based search by handling exact matches,
/// rare terms, and queries where lexical overlap matters more than
/// semantic similarity.
pub struct BM25Index {
    inverted: HashMap<String, Vec<(usize, u32)>>,
    doc_ids: Vec<StubId>,
    doc_lengths: Vec<u32>,
    avg_dl: f32,
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            doc_ids: Vec::new(),
            doc_lengths: Vec::new(),
            avg_dl: 0.0,
        }
    }

    pub fn add(&mut self, id: StubId, text: &str) {
        let tokens = tokenize(text);
        let doc_idx = self.doc_ids.len();
        self.doc_ids.push(id);
        self.doc_lengths.push(tokens.len() as u32);

        let mut tf: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_default() += 1;
        }

        for (term, freq) in tf {
            self.inverted.entry(term).or_default().push((doc_idx, freq));
        }

        let total: u32 = self.doc_lengths.iter().sum();
        self.avg_dl = total as f32 / self.doc_lengths.len().max(1) as f32;
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(StubId, f32)> {
        let query_tokens = tokenize(query);
        let n = self.doc_ids.len() as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.inverted.get(token) {
                let df = postings.len() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_idx, tf) in postings {
                    let dl = self.doc_lengths[doc_idx] as f32;
                    let tf_score = (tf as f32 * (K1 + 1.0))
                        / (tf as f32 + K1 * (1.0 - B + B * dl / self.avg_dl));
                    *scores.entry(doc_idx).or_default() += idf * tf_score;
                }
            }
        }

        let mut results: Vec<_> = scores
            .into_iter()
            .map(|(idx, score)| (self.doc_ids[idx].clone(), score))
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(top_k);
        results
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(String::from)
        .collect()
}
