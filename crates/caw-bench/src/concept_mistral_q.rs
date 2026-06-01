//! Quantized (GGUF) Mistral decoder, vendored from candle-transformers 0.10.2
//! (`models/quantized_mistral.rs`), reduced to a prefill-only forward that returns the
//! **per-layer last-token hidden state** — the "concept vector" of
//! concept-vector-rag-guide.md.
//!
//! Why vendor: the stock `quantized_mistral::Model::forward` runs the layer loop
//! internally and only returns final-token logits; its layers/embeddings are private
//! and there is no `output_hidden_states` equivalent, so the middle-layer residual
//! stream is unreachable. This copy adds one capability: tap each decoder layer's
//! output. The lm_head is dropped (never needed for retrieval).
//!
//! Built on candle-transformers' *public* quantized primitives (`quantized_nn`,
//! `quantized_var_builder`), so only the forward loop is reimplemented, not the
//! GGUF loading or QMatMul plumbing.
//!
//! Throwaway feasibility probe, not production code.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Activation;
use candle_transformers::models::mistral::Config;
use candle_transformers::quantized_nn::{linear_no_bias, Embedding, Linear, RmsNorm};
use candle_transformers::quantized_var_builder::VarBuilder;
use std::sync::Arc;

fn head_dim(cfg: &Config) -> usize {
    cfg.head_dim
        .unwrap_or(cfg.hidden_size / cfg.num_attention_heads)
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, hd) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b_sz, n_kv_head, n_rep, seq_len, hd))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, hd))
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        let rope_theta = cfg.rope_theta as f32;
        let dim = head_dim(cfg);
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q, k))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    // llama.cpp GGUF tensor names: ffn_gate / ffn_up / ffn_down (flat under blk.N).
    fn new(cfg: &Config, vb: &VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("ffn_gate"))?,
            up_proj: linear_no_bias(h, i, vb.pp("ffn_up"))?,
            down_proj: linear_no_bias(i, h, vb.pp("ffn_down"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    // llama.cpp GGUF tensor names: attn_q / attn_k / attn_v / attn_output (flat under blk.N).
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: &VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let hd = head_dim(cfg);
        Ok(Self {
            q_proj: linear_no_bias(h, num_heads * hd, vb.pp("attn_q"))?,
            k_proj: linear_no_bias(h, num_kv_heads * hd, vb.pp("attn_k"))?,
            v_proj: linear_no_bias(h, num_kv_heads * hd, vb.pp("attn_v"))?,
            o_proj: linear_no_bias(num_heads * hd, h, vb.pp("attn_output"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim: hd,
            rotary_emb,
        })
    }

    /// Single-shot prefill attention (no KV cache, seqlen_offset = 0).
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary_emb.apply(&q, &k)?;
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    // llama.cpp GGUF: attn/ffn projections sit flat under blk.N; the two RMSNorms are
    // attn_norm (pre-attention) and ffn_norm (pre-MLP).
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, vb)?,
            mlp: Mlp::new(cfg, vb)?,
            input_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("ffn_norm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct QConceptModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    device: Device,
}

impl QConceptModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // llama.cpp GGUF: token_embd (embedding) + blk.N.* (layers), all top-level.
        let embed_tokens = Embedding::new(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embd"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, vb.device())?);
        let vb_l = vb.pp("blk");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(rotary_emb.clone(), cfg, &vb_l.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens,
            layers,
            device: vb.device().clone(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Full causal mask (f32; quantized compute runs in f32). The probe's inputs are
    /// far shorter than Mistral's sliding window, so the window never engages.
    fn causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?.reshape((1, 1, seq_len, seq_len))
    }

    /// Prefill `input_ids` (shape `(1, seq_len)`) and return the last-token hidden state
    /// after every decoder layer: a Vec of length `num_layers`, each `(hidden,)` on CPU.
    pub fn concept_vectors_per_layer(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mask = self.causal_mask(seq_len)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mut out = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask)?;
            let last = xs.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
            out.push(last.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
        }
        Ok(out)
    }
}
