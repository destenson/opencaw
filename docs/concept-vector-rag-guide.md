# Probing Concept Vectors for RAG

*A practitioner's guide to using prefill activations as a semantic index.*

---

## Overview

Before a language model generates a single output token, it has already formed a rich internal
representation of the prompt — what it's about, what it intends to say, what knowledge it will
draw on. This guide shows you how to capture and probe that representation (the **concept vector**)
and use it to drive retrieval-augmented generation (RAG) without a separate embedding model or
query-rewriting step.

The model's own prefill is your cheapest, most semantically grounded embedding.

---

## Prerequisites

- A HuggingFace model in safetensors format (not GGUF — you need clean activation access)
- `transformers`, `torch`, `scikit-learn`, `umap-learn`, `numpy`
- Enough VRAM to run the model in inference mode (fp16 or bfloat16 is fine)

Recommended starting models: Qwen2.5-7B-Instruct, Mistral-7B-Instruct, Llama-3.1-8B-Instruct.

---

## Step 1: Capture the concept vector

Run a single forward pass with no generation. Extract the last token's hidden state at each layer.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_concept_vectors(prompt: str, model, tokenizer, device="cuda"):
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(
            **tokens,
            output_hidden_states=True,
            output_attentions=False,   # skip unless you need head-level detail
        )
    # hidden_states: tuple of (batch=1, seq_len, hidden_dim) per layer
    # last token position = -1; that position aggregates the full context
    concept_vectors = torch.stack([
        layer[0, -1, :] for layer in out.hidden_states
    ])  # shape: (n_layers, hidden_dim)
    return concept_vectors.cpu().float()
```

**Why the last token?** In causal (decoder-only) models, each token attends to all prior tokens.
The last token's residual stream is the only position that has "seen" the entire prompt — it's
where the model's full contextual understanding converges before generation begins.

---

## Step 2: Choose which layers to probe

Not all layers are equally informative for semantic content.

| Layer range | What it encodes |
|---|---|
| Early (0–20%) | Syntactic structure, token-level features |
| Middle (20–60%) | Semantic content, entity relationships, topic |
| Late (60–85%) | Task-specific reasoning, response planning |
| Final (85–100%) | Output distribution shaping, style |

For RAG concept extraction, **middle layers are your target**. They capture what the prompt
is *about* without being too entangled with how the model plans to respond.

```python
def middle_layer_vector(concept_vectors):
    n = concept_vectors.shape[0]
    lo, hi = int(n * 0.25), int(n * 0.65)
    # mean-pool across the middle layer range
    return concept_vectors[lo:hi].mean(dim=0)
```

You can also probe each layer independently and pick the one that best separates your domain
categories — see Step 4.

---

## Step 3: Build a calibration corpus

Before you can interpret what the concept vector encodes, you need a set of prompts with known
ground-truth topics/domains. This is your probe training set.

```python
def build_calibration_corpus() -> list[dict]:
    # Replace with actual domain examples from your corpus
    return load_domain_examples()   # returns [{prompt, domain, subtopics}]
```

Collect 50–200 examples per domain. Domains should reflect the categories your RAG index is
organized around — not generic topics, but the actual retrieval partitions you'll use.

Run `get_concept_vectors` on each example and store the middle-layer vector alongside its label.

---

## Step 4: Fit linear probes per domain

A linear probe is a logistic regression trained on hidden states to predict a label. If the probe
achieves high accuracy, the concept is linearly decodable from that layer — meaning it's
explicitly represented in the residual stream.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

def fit_domain_probe(vectors: np.ndarray, labels: list[str]):
    scaler = StandardScaler()
    X = scaler.fit_transform(vectors)
    probe = LogisticRegression(max_iter=1000, C=0.1)
    cv_scores = cross_val_score(probe, X, labels, cv=5)
    probe.fit(X, labels)
    return probe, scaler, cv_scores.mean()
```

**Interpret the CV score:**
- >0.85 — concept is cleanly linearly decodable; probe is reliable
- 0.65–0.85 — concept is present but noisy; consider layer search or PCA first
- <0.65 — try adjacent layers, or the concept isn't separable at this granularity

Fit one probe per domain axis: topic, intent, entity type, required knowledge base.

---

## Step 5: Visualize before you trust

Always visualize before deploying a probe. Unexpected cluster geometry is a signal that your
domain labels don't match what the model actually encodes.

```python
from umap import UMAP
import matplotlib.pyplot as plt

def visualize_concept_space(vectors: np.ndarray, labels: list[str]):
    reducer = UMAP(n_components=2, metric="cosine", random_state=42)
    embedding = reducer.fit_transform(vectors)
    
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color in zip(unique_labels, colors):
        mask = [l == label for l in labels]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            label=label, color=color, alpha=0.7, s=40
        )
    ax.legend()
    plt.tight_layout()
    return fig
```

Look for:
- **Clean separation** — your probes will be reliable
- **Overlapping clusters** — those domains are conflated in the model's representation;
  you may need finer labels or a different layer
- **Outliers** — individual prompts that land far from their cluster; inspect these manually,
  they're often ambiguous prompts where the model's representation diverges from your label

---

## Step 6: Extract RAG routing signals at inference

At inference time, run the prefill, extract the concept vector, and apply your probes to get
domain probability distributions. Use those to route retrieval.

```python
def extract_rag_routing(
    prompt: str,
    model,
    tokenizer,
    probes: dict,   # {domain_axis: (probe, scaler)}
    top_k: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    
    vectors = get_concept_vectors(prompt, model, tokenizer)
    cv = middle_layer_vector(vectors).numpy().reshape(1, -1)
    
    routing = {}
    for axis, (probe, scaler) in probes.items():
        X = scaler.transform(cv)
        probs = probe.predict_proba(X)[0]
        classes = probe.classes_
        top = sorted(zip(classes, probs), key=lambda x: -x[1])[:top_k]
        routing[axis] = top   # e.g. [("biology", 0.82), ("chemistry", 0.11), ...]
    
    return routing
```

The routing signal drives which partitions of your vector store to query — skip low-probability
domains entirely to reduce retrieval noise.

---

## Step 7: Handle relational and multi-domain prompts

A single prompt often spans multiple domains. The concept vector doesn't collapse to one label —
it's a continuous vector, and multiple probe dimensions will fire simultaneously.

Strategy: **threshold rather than argmax**.

```python
def route_to_partitions(routing: dict, threshold: float = 0.25) -> list[str]:
    partitions = []
    for axis, ranked in routing.items():
        for label, prob in ranked:
            if prob >= threshold:
                partitions.append(f"{axis}:{label}")
    return list(set(partitions))
```

For relational prompts ("the interaction between X and Y"), both X and Y domains will appear
above threshold — this is the correct behavior. Retrieve from both partitions and let the
synthesis step reconcile.

---

## Step 8: Attention heads as a complementary signal

If you want finer-grained signals — entity mentions, coreference, temporal scope — enable
`output_attentions=True` and inspect specific heads in the middle layers. Some heads
specialize reliably across model families:

- Heads that attend strongly to proper nouns → entity type signal
- Heads with diagonal patterns → positional/syntactic structure
- Heads with diffuse, uniform attention → topic/background knowledge
- Heads that attend strongly to the first token → global context anchoring

Use attention entropy as a quick filter: low-entropy heads (sharp attention) encode specific
relationships; high-entropy heads encode diffuse semantic content.

```python
def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    # attn_weights: (batch, heads, seq, seq)
    # returns entropy per head at last token position
    w = attn_weights[0, :, -1, :]   # (heads, seq)
    w = w.clamp(min=1e-9)
    return -(w * w.log()).sum(dim=-1)   # (heads,)
```

Low-entropy heads at middle layers are worth probing independently — they often have clean
linear structure for entity and relation classification.

---

## Failure modes to know

**Concept drift across prompt length** — very short prompts produce concept vectors that
don't reliably represent the full intended topic. Add a minimum token threshold (~20 tokens)
or use a fixed preamble to anchor the representation.

**Probe overfit to surface form** — if your calibration corpus uses similar phrasing across
examples within a domain, the probe may be learning lexical patterns rather than semantic
content. Test with paraphrased prompts from held-out examples.

**Layer sensitivity** — different model families have different "sweet spots" for semantic
content. Run the layer search (Step 4) on each new model rather than assuming middle layers
are always optimal.

**Instruction-tuned vs base models** — instruction-tuned models have different residual stream
geometry in the late layers (task-planning representations are stronger). For RAG routing, this
is usually fine or beneficial; for pure semantic similarity tasks, base models may give cleaner
concept vectors.

---

## Summary pipeline

```
prompt
  → single forward pass (prefill only, no generation)
  → extract last-token hidden states across layers
  → mean-pool middle layers → concept vector
  → apply linear probes → domain probability distributions
  → threshold routing → retrieval partition selection
  → retrieve from selected partitions
  → augmented context → generation pass
```

The prefill pass is fast — typically 5–15% of generation latency for a typical prompt length.
You get a semantically grounded routing signal at minimal cost, using the model's own internal
representation rather than a separate embedding model that may have a different semantic space.
