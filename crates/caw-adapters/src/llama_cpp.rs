use caw_core::{
    split_thinking, CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter,
    ModelCapabilities, ProvenanceFormat,
};
use caw_llama_sys::*;
use std::collections::VecDeque;
use std::ffi::CString;
use std::sync::Mutex;
use tracing::debug;

/// Configuration for a native llama.cpp inference session.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model_path: String,
    /// Context window size in tokens. 0 = use model default (dangerous: models
    /// with large training contexts like Qwen3's 128K will pre-allocate an
    /// enormous KV cache; prefer an explicit cap unless you have the VRAM).
    pub n_ctx: u32,
    /// GPU layers to offload. -1 = all layers.
    pub n_gpu_layers: i32,
    /// RNG seed. Use 0 for a time-based seed.
    pub seed: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_new_tokens: usize,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            n_ctx: 8192,
            n_gpu_layers: -1,
            seed: 42,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            max_new_tokens: 4096,
        }
    }
}

struct LlamaState {
    model: *mut llama_model,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
}

// The model and context pointers are only accessed while the Mutex is held,
// and llama.cpp's context is not thread-safe — single-threaded access is correct.
unsafe impl Send for LlamaState {}

impl Drop for LlamaState {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
    }
}

pub struct LlamaCppAdapter {
    state: Mutex<LlamaState>,
    config: LlamaCppConfig,
    model_name: String,
    /// True when the model is known to produce `<think>...</think>` blocks.
    /// Detected from the model filename at load time.
    visible_reasoning: bool,
}

impl LlamaCppAdapter {
    /// Load a model and create an inference context.
    ///
    /// Set `LLAMA_PATH` in the environment to point to the llama.cpp build
    /// directory so the dynamic linker can find libllama.so at runtime.
    pub fn new_with(config: LlamaCppConfig) -> CawResult<Self> {
        if config.model_path.is_empty() {
            return Err(CawError::Adapter("model_path is required".into()));
        }

        unsafe { llama_backend_init() };

        let model_path_c = CString::new(config.model_path.as_str())
            .map_err(|e| CawError::Adapter(e.to_string().into()))?;

        let mut model_params = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = config.n_gpu_layers;

        let model =
            unsafe { llama_model_load_from_file(model_path_c.as_ptr(), model_params) };
        if model.is_null() {
            return Err(CawError::Adapter(
                format!("failed to load model: {}", config.model_path).into(),
            ));
        }

        let mut ctx_params = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = config.n_ctx;
        ctx_params.n_batch = 512;
        ctx_params.offload_kqv = true;

        let ctx = unsafe { llama_init_from_model(model, ctx_params) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            return Err(CawError::Adapter("failed to create llama context".into()));
        }

        let vocab = unsafe { llama_model_get_vocab(model) };
        let model_name = extract_model_name(&config.model_path);
        let visible_reasoning = is_thinking_model(&model_name);

        debug!(
            model = %model_name,
            n_ctx = config.n_ctx,
            n_gpu_layers = config.n_gpu_layers,
            visible_reasoning,
            "llama model loaded"
        );

        Ok(Self {
            state: Mutex::new(LlamaState { model, ctx, vocab }),
            config,
            model_name,
            visible_reasoning,
        })
    }

    /// Convenience constructor using a model path and default settings.
    pub fn from_path(model_path: impl Into<String>) -> CawResult<Self> {
        Self::new_with(LlamaCppConfig {
            model_path: model_path.into(),
            ..Default::default()
        })
    }
}

impl ModelAdapter for LlamaCppAdapter {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_passive_injection: true,
            supports_visible_reasoning: self.visible_reasoning,
            ..Default::default()
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CawError::Adapter("llama state mutex poisoned".into()))?;

        let prompt = format_prompt(&state, &req)?;
        let raw = run_generation(&mut state, &self.config, &prompt, usize::MAX, 0, None)?;
        let (thinking, answer) = split_thinking(&raw);
        Ok(CompletionResponse {
            answer,
            thinking,
            usage: None,
        })
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CawError::Adapter("llama state mutex poisoned".into()))?;

        let prompt = format_prompt(&state, &req)?;
        let raw = run_generation(
            &mut state,
            &self.config,
            &prompt,
            check_interval,
            window_size,
            Some(on_window),
        )?;
        let (thinking, answer) = split_thinking(&raw);
        Ok(CompletionResponse {
            answer,
            thinking,
            usage: None,
        })
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<()> {
        if !self.visible_reasoning {
            let response = self.complete(req)?;
            let thinking = response.thinking.unwrap_or_default();
            for step in thinking.split("\n\n").map(str::trim).filter(|s| !s.is_empty()) {
                if !on_step(step)? {
                    return Ok(());
                }
            }
            return Ok(());
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| CawError::Adapter("llama state mutex poisoned".into()))?;
        let prompt = format_prompt(&state, &req)?;
        run_thinking_steps(&mut state, &self.config, &prompt, on_step)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Detect whether a model produces `<think>...</think>` blocks from its name.
fn is_thinking_model(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("think") || lower.contains("qwq") || lower.contains("-r1")
}

fn extract_model_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

/// Apply the model's embedded chat template to produce a prompt string.
fn format_prompt(state: &LlamaState, req: &CompletionRequest) -> CawResult<String> {
    let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);
    let full_user = format!("{}{}", req.user, workspace_context);

    // Build system and user message C strings — must outlive the chat array.
    let system_c =
        CString::new(req.system.as_str()).map_err(|e| CawError::Adapter(e.to_string().into()))?;
    let user_c =
        CString::new(full_user.as_str()).map_err(|e| CawError::Adapter(e.to_string().into()))?;
    let role_system = c"system";
    let role_user = c"user";

    let messages = [
        llama_chat_message {
            role: role_system.as_ptr(),
            content: system_c.as_ptr(),
        },
        llama_chat_message {
            role: role_user.as_ptr(),
            content: user_c.as_ptr(),
        },
    ];

    // Use the model's embedded template (from GGUF metadata) so the prompt
    // matches whatever chat format the model was trained with.
    let tmpl = unsafe { llama_model_chat_template(state.model, std::ptr::null()) };

    // First call: measure required buffer size (returns needed byte count).
    let needed = unsafe {
        llama_chat_apply_template(
            tmpl,
            messages.as_ptr(),
            messages.len(),
            true,
            std::ptr::null_mut(),
            0,
        )
    };
    if needed < 0 {
        return Err(CawError::Adapter("chat template failed".into()));
    }

    let mut buf = vec![0u8; needed as usize + 1];
    let written = unsafe {
        llama_chat_apply_template(
            tmpl,
            messages.as_ptr(),
            messages.len(),
            true,
            buf.as_mut_ptr() as *mut i8,
            buf.len() as i32,
        )
    };
    if written < 0 {
        return Err(CawError::Adapter("chat template write failed".into()));
    }

    buf.truncate(written as usize);
    String::from_utf8(buf).map_err(|e| CawError::Adapter(e.to_string().into()))
}

/// Tokenize a UTF-8 string into llama token IDs.
fn tokenize(vocab: *const llama_vocab, text: &str, add_special: bool) -> CawResult<Vec<i32>> {
    let text_c =
        CString::new(text).map_err(|e| CawError::Adapter(e.to_string().into()))?;
    let text_len = text.len() as i32;

    // Upper-bound: one token per byte is impossible but gives a safe allocation.
    let max_tokens = text.len() + 64;
    let mut tokens = vec![0i32; max_tokens];

    let n = unsafe {
        llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text_len,
            tokens.as_mut_ptr(),
            max_tokens as i32,
            add_special,
            false,
        )
    };

    if n == i32::MIN {
        return Err(CawError::Adapter("tokenize: input too large".into()));
    }
    if n < 0 {
        // Buffer was too small — rare, but allocate exactly what's needed.
        let needed = n.unsigned_abs() as usize;
        tokens.resize(needed, 0);
        let n2 = unsafe {
            llama_tokenize(
                vocab,
                text_c.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                needed as i32,
                add_special,
                false,
            )
        };
        if n2 < 0 {
            return Err(CawError::Adapter("tokenize failed after resize".into()));
        }
        tokens.truncate(n2 as usize);
    } else {
        tokens.truncate(n as usize);
    }

    Ok(tokens)
}

/// Convert a single token to its UTF-8 text piece.
fn token_to_piece(vocab: *const llama_vocab, token: i32) -> String {
    let mut buf = [0u8; 256];
    let n = unsafe {
        llama_token_to_piece(
            vocab,
            token,
            buf.as_mut_ptr() as *mut i8,
            buf.len() as i32,
            0,     // lstrip
            false, // special tokens as text
        )
    };
    if n <= 0 {
        return String::new();
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

/// Core sampling loop shared by `complete` and `generate_passive`.
///
/// When `on_window` is provided, `on_window` is called every `check_interval`
/// tokens with the last `window_size` token pieces as a sliding window. This
/// means consecutive calls overlap, so important token sequences are never
/// split across an interval boundary and missed. If the callback returns
/// `Some(content)`, that content is tokenized and decoded into the KV cache
/// before sampling continues — injecting recalled material without restarting
/// generation.
fn run_generation(
    state: &mut LlamaState,
    config: &LlamaCppConfig,
    prompt: &str,
    check_interval: usize,
    window_size: usize,
    mut on_window: Option<&mut dyn FnMut(&str) -> CawResult<Option<String>>>,
) -> CawResult<String> {
    let ctx = state.ctx;
    let vocab = state.vocab;

    // Clear KV cache from any previous call so position tracking starts fresh.
    unsafe {
        let mem = llama_get_memory(ctx);
        llama_memory_clear(mem, false);
    }

    // Tokenize and prefill the prompt.
    let prompt_tokens = tokenize(vocab, prompt, true)?;
    if prompt_tokens.is_empty() {
        return Err(CawError::Adapter("empty prompt after tokenization".into()));
    }

    let prefill_ret = unsafe {
        llama_decode(
            ctx,
            llama_batch_get_one(
                prompt_tokens.as_ptr() as *mut _,
                prompt_tokens.len() as i32,
            ),
        )
    };
    if prefill_ret != 0 {
        return Err(CawError::Adapter(
            format!("prefill decode failed: {prefill_ret}").into(),
        ));
    }

    // Build sampler chain.
    let smpl = unsafe {
        let sparams = llama_sampler_chain_default_params();
        let smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(config.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(config.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(config.seed));
        smpl
    };

    let mut generated = String::new();
    // Sliding deque: each entry is one decoded token piece. Capped at
    // `window_size` so joining it always yields the last N tokens of output.
    let mut sliding_window: VecDeque<String> = VecDeque::with_capacity(window_size + 1);
    let mut tokens_since_check = 0usize;

    'decode: for _ in 0..config.max_new_tokens {
        let token = unsafe { llama_sampler_sample(smpl, ctx, -1) };

        if unsafe { llama_vocab_is_eog(vocab, token) } {
            break;
        }

        let piece = token_to_piece(vocab, token);
        generated.push_str(&piece);

        if on_window.is_some() {
            sliding_window.push_back(piece.clone());
            if sliding_window.len() > window_size {
                sliding_window.pop_front();
            }
        }

        tokens_since_check += 1;

        unsafe { llama_sampler_accept(smpl, token) };

        // Feed the sampled token back into the KV cache for the next step.
        let token_arr = [token];
        let step_ret = unsafe {
            llama_decode(
                ctx,
                llama_batch_get_one(token_arr.as_ptr() as *mut _, 1),
            )
        };
        if step_ret != 0 {
            debug!(ret = step_ret, "llama_decode step error — stopping");
            break 'decode;
        }

        if tokens_since_check >= check_interval {
            tokens_since_check = 0;
            if let Some(ref mut cb) = on_window {
                let window_text: String = sliding_window.iter().cloned().collect();
                if let Ok(Some(injection)) = cb(&window_text) {
                    // Append the injected text to the visible output so the
                    // caller can see what was materialised inline.
                    generated.push_str(&injection);

                    let inj_tokens = tokenize(vocab, &injection, false)?;
                    if !inj_tokens.is_empty() {
                        let inj_ret = unsafe {
                            llama_decode(
                                ctx,
                                llama_batch_get_one(
                                    inj_tokens.as_ptr() as *mut _,
                                    inj_tokens.len() as i32,
                                ),
                            )
                        };
                        if inj_ret != 0 {
                            debug!(ret = inj_ret, "injection decode error — continuing without inject");
                        }
                    }
                }
            }
        }
    }

    unsafe { llama_sampler_free(smpl) };

    Ok(generated)
}

/// Generate tokens, firing `on_step` at each `\n\n` paragraph boundary.
///
/// The sampling loop stops immediately if `on_step` returns `false` — meaning
/// a recall hit was found and the orchestrator wants to restart with an
/// enriched workspace. We own the loop here, so unlike HTTP-streaming adapters
/// there is no need to collect all steps first and replay; we interrupt in
/// real time as soon as new context is found.
fn run_thinking_steps(
    state: &mut LlamaState,
    config: &LlamaCppConfig,
    prompt: &str,
    on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
) -> CawResult<()> {
    let ctx = state.ctx;
    let vocab = state.vocab;

    unsafe {
        let mem = llama_get_memory(ctx);
        llama_memory_clear(mem, false);
    }

    let prompt_tokens = tokenize(vocab, prompt, true)?;
    if prompt_tokens.is_empty() {
        return Err(CawError::Adapter("empty prompt after tokenization".into()));
    }

    let prefill_ret = unsafe {
        llama_decode(
            ctx,
            llama_batch_get_one(
                prompt_tokens.as_ptr() as *mut _,
                prompt_tokens.len() as i32,
            ),
        )
    };
    if prefill_ret != 0 {
        return Err(CawError::Adapter(
            format!("prefill decode failed: {prefill_ret}").into(),
        ));
    }

    let smpl = unsafe {
        let sparams = llama_sampler_chain_default_params();
        let smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(config.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(config.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(config.seed));
        smpl
    };

    let mut step_buf = String::new();
    let mut n_steps = 0usize;

    'decode: for _ in 0..config.max_new_tokens {
        let token = unsafe { llama_sampler_sample(smpl, ctx, -1) };

        if unsafe { llama_vocab_is_eog(vocab, token) } {
            break;
        }

        let piece = token_to_piece(vocab, token);
        unsafe { llama_sampler_accept(smpl, token) };

        let token_arr = [token];
        let step_ret = unsafe {
            llama_decode(ctx, llama_batch_get_one(token_arr.as_ptr() as *mut _, 1))
        };
        if step_ret != 0 {
            debug!(ret = step_ret, "llama_decode step error — stopping");
            break 'decode;
        }

        step_buf.push_str(&piece);

        // Fire a step at each paragraph boundary and stop if the orchestrator
        // found a recall hit (on_step returns false).
        while let Some(boundary) = step_buf.find("\n\n") {
            let step = step_buf[..boundary].trim().to_string();
            step_buf.drain(..boundary + 2);
            if step.is_empty() {
                continue;
            }
            n_steps += 1;
            debug!(step_preview = &step[..step.len().min(80)], "step");
            if !on_step(&step)? {
                debug!(n_steps, "early stop — recall hit");
                break 'decode;
            }
        }
    }

    unsafe { llama_sampler_free(smpl) };

    Ok(())
}
