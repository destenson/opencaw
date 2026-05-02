use crate::{CawResult, Tokenizer};
use std::sync::OnceLock;

/// Count tokens using cl100k_base BPE. The tokenizer is initialized once
/// and reused — callers don't need to manage a Tokenizer instance.
/// Falls back to whitespace splitting only if BPE data fails to load (shouldn't happen).
pub fn count_tokens_cl100k(text: &str) -> usize {
    static TOKENIZER: OnceLock<Option<TiktokenTokenizer>> = OnceLock::new();
    let tok = TOKENIZER.get_or_init(|| TiktokenTokenizer::cl100k().ok());
    match tok {
        Some(t) => t.count_tokens(text),
        None => text.split_whitespace().count().max(1),
    }
}

/// BPE tokenizer backed by tiktoken-rs. Gives accurate token counts
/// for OpenAI-family models, and reasonable approximations for Claude
/// (which uses a similar vocabulary size to cl100k_base).
pub struct TiktokenTokenizer {
    bpe: tiktoken_rs::CoreBPE,
    name: &'static str,
}

impl TiktokenTokenizer {
    fn from_bpe(
        result: Result<tiktoken_rs::CoreBPE, impl std::fmt::Display>,
        name: &'static str,
    ) -> CawResult<Self> {
        let bpe = result.map_err(|err| {
            crate::CawError::InvalidInput(format!("failed to load {} tokenizer data: {}", name, err))
        })?;
        Ok(Self { bpe, name })
    }

    /// cl100k_base encoding — used by GPT-4, GPT-3.5-turbo, and
    /// close enough for Claude models (which use a similar vocabulary size).
    pub fn cl100k() -> CawResult<Self> {
        Self::from_bpe(tiktoken_rs::cl100k_base(), "cl100k_base")
    }

    /// p50k_base encoding — used by older GPT-3 / Codex models.
    pub fn p50k() -> CawResult<Self> {
        Self::from_bpe(tiktoken_rs::p50k_base(), "p50k_base")
    }

    /// o200k_base encoding — used by GPT-4o, o1/o3/o4 series.
    pub fn o200k() -> CawResult<Self> {
        Self::from_bpe(tiktoken_rs::o200k_base(), "o200k_base")
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
