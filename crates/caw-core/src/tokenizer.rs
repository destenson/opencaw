use crate::{CawResult, Tokenizer};

/// BPE tokenizer backed by tiktoken-rs. Gives accurate token counts
/// for OpenAI-family models, and reasonable approximations for Claude
/// (which uses a similar vocabulary size to cl100k_base).
pub struct TiktokenTokenizer {
    bpe: tiktoken_rs::CoreBPE,
    name: &'static str,
}

impl TiktokenTokenizer {
    /// cl100k_base encoding — used by GPT-4, GPT-3.5-turbo, and
    /// close enough for Claude models (which use a similar vocabulary size).
    pub fn cl100k() -> CawResult<Self> {
        Ok(Self {
            bpe: tiktoken_rs::cl100k_base().expect("cl100k_base data should be bundled"),
            name: "cl100k_base",
        })
    }

    /// p50k_base encoding — used by older GPT-3 / Codex models.
    pub fn p50k() -> CawResult<Self> {
        Ok(Self {
            bpe: tiktoken_rs::p50k_base().expect("p50k_base data should be bundled"),
            name: "p50k_base",
        })
    }

    /// o200k_base encoding — used by GPT-4o, o1/o3/o4 series.
    pub fn o200k() -> CawResult<Self> {
        Ok(Self {
            bpe: tiktoken_rs::o200k_base().expect("o200k_base data should be bundled"),
            name: "o200k_base",
        })
    }

    /// Auto-select encoding based on model name. Falls back to cl100k_base
    /// for unrecognized models since it's the most widely applicable.
    pub fn for_model(model_name: &str) -> CawResult<Self> {
        let lower = model_name.to_lowercase();
        if lower.contains("gpt-4o") || lower.contains("o1") || lower.contains("o3") {
            Self::o200k()
        } else if lower.contains("gpt-4") || lower.contains("gpt-3.5") || lower.contains("claude") {
            Self::cl100k()
        } else if lower.contains("davinci") || lower.contains("codex") {
            Self::p50k()
        } else {
            Self::cl100k()
        }
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len().max(1)
    }

    fn tokenizer_name(&self) -> &str {
        self.name
    }
}
