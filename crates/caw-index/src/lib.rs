use caw_core::{CawError, CawResult, Locator, RecallFragment, Retriever, ScoredStub, Stub, StubId};

#[derive(Debug, Default, Clone)]
pub struct InMemoryIndex {
    stubs: Vec<Stub>,
    docs: Vec<(StubId, String)>,
}

impl InMemoryIndex {
    pub fn insert(&mut self, stub: Stub, content: String) {
        self.docs.push((stub.id.clone(), content));
        self.stubs.push(stub);
    }
}

impl Retriever for InMemoryIndex {
    fn search(&self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>> {
        let mut scored = self
            .stubs
            .iter()
            .map(|stub| ScoredStub {
                stub: stub.clone(),
                score: score_query_against_stub(query, stub),
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        Ok(scored)
    }

    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment> {
        let content = self
            .docs
            .iter()
            .find_map(|(stub_id, content)| (stub_id == id).then_some(content))
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        Ok(RecallFragment {
            stub_id: id.clone(),
            content: content.clone(),
            locator: Locator {
                source: id.0.clone(),
                locator: range.to_string(),
            },
            tokens: (content.len() / 4).max(1),
        })
    }
}

fn score_query_against_stub(query: &str, stub: &Stub) -> f32 {
    let q = query.to_ascii_lowercase();
    let mut score = 0.0_f32;

    if stub.path.to_ascii_lowercase().contains(&q) {
        score += 0.8;
    }
    if stub.summary.to_ascii_lowercase().contains(&q) {
        score += 0.6;
    }
    if stub
        .outline
        .iter()
        .any(|line| line.to_ascii_lowercase().contains(&q))
    {
        score += 0.4;
    }

    if score == 0.0 {
        let query_terms = q.split_whitespace().collect::<Vec<_>>();
        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "))
            .to_ascii_lowercase();
        let matched = query_terms.iter().filter(|t| text.contains(*t)).count();
        score += matched as f32 / (query_terms.len().max(1) as f32);
    }

    score
}
