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

/// Tokenize for lexical matching, splitting identifiers into their subwords.
/// Public so retrieval diagnostics can measure query↔document token overlap with
/// the exact tokenization BM25 indexes and queries with.
///
/// Natural-language queries don't contain the identifier they're asking about:
/// "which struct owns the recall loop" shares no token with a chunk whose only
/// lexical signal is `DynamicRecallOrchestrator`. Splitting that identifier into
/// `dynamic` / `recall` / `orchestrator` lets the question's words match the
/// code's names — the "both retrievers miss on register mismatch" failure mode.
///
/// `snake_case` already splits on the underscore (a non-alphanumeric boundary);
/// the gap was `camelCase`/`PascalCase` and letter↔digit runs, which carry no
/// separator. We split on those boundaries *before* lowercasing (lowercasing
/// first would erase them) and also keep the whole joined token, so an exact
/// identifier in a query still scores its chunk.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let subwords = split_identifier(raw);
        let whole = raw.to_lowercase();
        if whole.len() > 1 {
            out.push(whole);
        }
        // Only emit subwords when the split actually produced more than one
        // piece; a plain word ("eviction") is already covered by `whole`.
        if subwords.len() > 1 {
            for sub in subwords {
                let sub = sub.to_lowercase();
                if sub.len() > 1 {
                    out.push(sub);
                }
            }
        }
    }
    out
}

/// Split a single alphanumeric token at camelCase and letter↔digit boundaries.
/// `DynamicRecallOrchestrator` → [Dynamic, Recall, Orchestrator];
/// `HTTPServer` → [HTTP, Server]; `BM25Index` → [BM, 25, Index]. A token with no
/// internal boundary (a plain word or a bare acronym) returns itself.
fn split_identifier(token: &str) -> Vec<String> {
    let chars: Vec<char> = token.chars().collect();
    let mut segments = Vec::new();
    let mut start = 0;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let next_is_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());
        let boundary =
            // lower/digit → upper: `recallLoop`, `5Lives`
            (!prev.is_uppercase() && cur.is_uppercase())
            // acronym end before a word: `HTTPServer` splits HTTP | Server
            || (prev.is_uppercase() && cur.is_uppercase() && next_is_lower)
            // letter ↔ digit either direction: `BM25`, `25Index`, `llama3`
            || (prev.is_alphabetic() && cur.is_ascii_digit())
            || (prev.is_ascii_digit() && cur.is_alphabetic());
        if boundary {
            segments.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    segments.push(chars[start..].iter().collect());
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_case_splits_and_keeps_words() {
        // underscore is a non-alphanumeric boundary, so each word stands alone;
        // single pieces mean no joined duplicate.
        assert_eq!(tokenize("recall_loop"), vec!["recall", "loop"]);
    }

    #[test]
    fn camel_case_yields_joined_plus_subwords() {
        // The NL query "recall loop" can now match this identifier via `recall`
        // and `loop`, while an exact `RecallLoop` still hits `recallloop`.
        assert_eq!(tokenize("RecallLoop"), vec!["recallloop", "recall", "loop"]);
    }

    #[test]
    fn pascal_multiword_identifier() {
        assert_eq!(
            tokenize("DynamicRecallOrchestrator"),
            vec![
                "dynamicrecallorchestrator",
                "dynamic",
                "recall",
                "orchestrator"
            ]
        );
    }

    #[test]
    fn acronym_run_then_word() {
        assert_eq!(tokenize("HTTPServer"), vec!["httpserver", "http", "server"]);
    }

    #[test]
    fn bare_acronym_stays_whole() {
        assert_eq!(tokenize("HTTP"), vec!["http"]);
    }

    #[test]
    fn letter_digit_boundaries() {
        assert_eq!(tokenize("BM25Index"), vec!["bm25index", "bm", "25", "index"]);
        // The single-char "3" subword is dropped by the len>1 filter (same on the
        // query side, so nothing is lost) — only the joined form and `llama` survive.
        assert_eq!(tokenize("llama3"), vec!["llama3", "llama"]);
    }

    #[test]
    fn plain_word_has_no_subwords() {
        assert_eq!(tokenize("eviction"), vec!["eviction"]);
    }

    #[test]
    fn nl_query_overlaps_identifier_doc() {
        // The whole point: a paraphrased query lexically overlaps a code chunk
        // whose only signal is a PascalCase symbol.
        let q: std::collections::HashSet<_> = tokenize("how does the recall loop evict").into_iter().collect();
        let doc: std::collections::HashSet<_> = tokenize("impl DynamicRecallOrchestrator { fn evict").into_iter().collect();
        assert!(q.contains("recall") && doc.contains("recall"));
        assert!(q.contains("evict") && doc.contains("evict"));
    }
}
