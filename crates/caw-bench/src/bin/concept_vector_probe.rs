//! Feasibility probe: can Mistral's own prefill activations (the "concept vector")
//! beat BGE-small at COSINE retrieval on the OpenCAW sysdoc set?
//!
//! This answers one question before any architecture is committed: does raw
//! `cosine(query_cv, passage_cv)` rank the gold passage competitively against the
//! real production embedder (`CandleEmbeddingProvider::bge_small`)? The guide
//! (concept-vector-rag-guide.md) only ever validates concept vectors through a
//! *trained linear probe*; using them directly for cosine similarity is a stronger,
//! unproven claim. This binary measures exactly that claim and nothing else.
//! Interpretation of the table happens in chat, not here.
//!
//!   cargo run -p caw-bench --features ... --bin caw-bench-concept-probe -- --device 1

use std::collections::BTreeMap;
use std::fs;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::mistral::Config;
use candle_transformers::quantized_var_builder::VarBuilder;
use caw_bench::concept_mistral_q::QConceptModel;
use caw_core::EmbeddingProvider;
use caw_index::{CandleEmbeddingProvider, EmbedDevice};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// GGUF repo + file for the quantized concept model. Q8_0 keeps quantization noise
/// minimal (~8 GB) so the feasibility signal isn't muddied by aggressive quantization.
const DEFAULT_GGUF_REPO: &str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF";
const DEFAULT_GGUF_FILE: &str = "Mistral-7B-Instruct-v0.3-Q8_0.gguf";
/// tokenizer.json + config.json come from the original (unquantized) repo, which the
/// GGUF doesn't carry. Already cached locally from the earlier fp download.
const DEFAULT_TOK_REPO: &str = "mistralai/Mistral-7B-Instruct-v0.3";
const DEFAULT_QA: &str = "opencaw-corpora/sysdoc_qa.json";
/// Cap both sides at the same token budget BGE uses (512), so neither method gets
/// to see more of a passage than the other — keeps the comparison fair.
const MAX_TOKENS: usize = 512;
const RECALL_KS: [usize; 3] = [1, 5, 10];

struct Args {
    gguf_repo: String,
    gguf_file: String,
    tok_repo: String,
    qa: String,
    device: usize,
    dump_disagreements: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        gguf_repo: DEFAULT_GGUF_REPO.to_string(),
        gguf_file: DEFAULT_GGUF_FILE.to_string(),
        tok_repo: DEFAULT_TOK_REPO.to_string(),
        qa: DEFAULT_QA.to_string(),
        device: 0,
        dump_disagreements: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--gguf-repo" => a.gguf_repo = it.next().expect("--gguf-repo needs a value"),
            "--gguf-file" => a.gguf_file = it.next().expect("--gguf-file needs a value"),
            "--tok-repo" => a.tok_repo = it.next().expect("--tok-repo needs a value"),
            "--qa" => a.qa = it.next().expect("--qa needs a value"),
            "--device" => {
                a.device = it.next().expect("--device needs N").parse().expect("device ordinal")
            }
            "--dump-disagreements" => a.dump_disagreements = true,
            other => panic!("unknown arg {other:?}"),
        }
    }
    a
}

#[derive(serde::Deserialize)]
struct RawQa {
    question: String,
    #[serde(rename = "_source_rel")]
    source_rel: String,
    #[serde(rename = "_quote")]
    quote: String,
}

struct Dataset {
    questions: Vec<String>,
    gold_idx: Vec<usize>,     // gold passage position per query
    passage_ids: Vec<String>, // _source_rel, deduped, stable order
    passage_text: Vec<String>,
}

fn load_dataset(path: &str) -> Result<Dataset> {
    let raw: Vec<RawQa> =
        serde_json::from_str(&fs::read_to_string(path).with_context(|| format!("read {path}"))?)
            .context("parse sysdoc QA JSON")?;
    // One passage per source, first occurrence wins. BTreeMap gives a stable order.
    let mut by_source: BTreeMap<String, String> = BTreeMap::new();
    for r in &raw {
        by_source
            .entry(r.source_rel.clone())
            .or_insert_with(|| r.quote.clone());
    }
    let passage_ids: Vec<String> = by_source.keys().cloned().collect();
    let pos: BTreeMap<&String, usize> =
        passage_ids.iter().enumerate().map(|(i, s)| (s, i)).collect();
    let passage_text: Vec<String> = passage_ids.iter().map(|s| by_source[s].clone()).collect();

    let mut questions = Vec::new();
    let mut gold_idx = Vec::new();
    for r in &raw {
        questions.push(r.question.clone());
        gold_idx.push(pos[&r.source_rel]);
    }
    Ok(Dataset {
        questions,
        gold_idx,
        passage_ids,
        passage_text,
    })
}

fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    ids.truncate(MAX_TOKENS);
    Ok(ids)
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Per-layer concept vectors for every text: returns `[text][layer][dim]`, L2-normalized.
fn concept_vectors(
    model: &QConceptModel,
    tok: &Tokenizer,
    device: &Device,
    texts: &[String],
    label: &str,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let mut out = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        let ids = encode(tok, text)?;
        if ids.is_empty() {
            bail!("empty token sequence for {label} #{i}");
        }
        let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let layers = model.concept_vectors_per_layer(&input)?;
        let mut per_layer = Vec::with_capacity(layers.len());
        for t in &layers {
            let mut v = t.to_vec1::<f32>()?;
            l2_normalize(&mut v);
            per_layer.push(v);
        }
        out.push(per_layer);
        if (i + 1) % 20 == 0 {
            println!("  {label}: prefilled {}/{}", i + 1, texts.len());
        }
    }
    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// All-but-the-top whitening (Mu & Viswanath 2018): mean-center, then project out the
/// top-`d` principal directions — the standard parameter-free fix for the anisotropy
/// that makes raw decoder activations non-metric. Fit on `fit` (the union of query +
/// passage vectors for this layer = the feasibility upper bound); the returned closure
/// applies the same transform to any vector and re-normalizes.
fn fit_whitening(fit: &[Vec<f32>], d: usize) -> impl Fn(&[f32]) -> Vec<f32> {
    let n = fit.len();
    let h = fit[0].len();
    let mut mean = vec![0.0f32; h];
    for v in fit {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x / n as f32;
        }
    }
    // Centered samples, then top-d principal directions via the N×N Gram matrix
    // (cheap: N=200 here) using power iteration with deflation.
    let centered: Vec<Vec<f32>> = fit
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(x, m)| x - m).collect())
        .collect();
    let mut gram = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i..n {
            let s = dot(&centered[i], &centered[j]);
            gram[i][j] = s;
            gram[j][i] = s;
        }
    }
    let mut comps: Vec<Vec<f32>> = Vec::with_capacity(d);
    for _ in 0..d {
        let mut u = vec![1.0 / (n as f32).sqrt(); n];
        let mut lambda = 0.0f32;
        for _ in 0..100 {
            let mut w = vec![0.0f32; n];
            for i in 0..n {
                w[i] = dot(&gram[i], &u);
            }
            lambda = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in w.iter_mut() {
                *x /= lambda;
            }
            u = w;
        }
        // Map the sample-space eigenvector u into feature space: vf = Xᵀu / sqrt(λ).
        let mut vf = vec![0.0f32; h];
        for (i, ui) in u.iter().enumerate() {
            for (f, c) in vf.iter_mut().zip(&centered[i]) {
                *f += ui * c;
            }
        }
        let vn = vf.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in vf.iter_mut() {
            *x /= vn;
        }
        // Deflate the Gram matrix so the next iteration finds the next component.
        for i in 0..n {
            for j in 0..n {
                gram[i][j] -= lambda * u[i] * u[j];
            }
        }
        comps.push(vf);
    }
    move |v: &[f32]| {
        let mut x: Vec<f32> = v.iter().zip(&mean).map(|(a, m)| a - m).collect();
        for c in &comps {
            let proj = dot(&x, c);
            for (xi, ci) in x.iter_mut().zip(c) {
                *xi -= proj * ci;
            }
        }
        let norm = x.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
        for xi in x.iter_mut() {
            *xi /= norm;
        }
        x
    }
}

/// sim[q][p] cosine matrix from already-normalized embeddings.
fn sim_matrix(queries: &[Vec<f32>], passages: &[Vec<f32>]) -> Vec<Vec<f32>> {
    queries
        .iter()
        .map(|q| passages.iter().map(|p| dot(q, p)).collect())
        .collect()
}

struct Metrics {
    recall: [f32; 3],
    mrr: f32,
    median_rank: f32,
}

/// rank = number of passages scoring strictly higher than the gold passage (0-based).
fn evaluate(sim: &[Vec<f32>], gold: &[usize]) -> Metrics {
    let mut ranks: Vec<usize> = Vec::with_capacity(gold.len());
    for (q, row) in sim.iter().enumerate() {
        let gold_score = row[gold[q]];
        ranks.push(row.iter().filter(|&&s| s > gold_score).count());
    }
    let n = ranks.len() as f32;
    let recall = RECALL_KS.map(|k| ranks.iter().filter(|&&r| r < k).count() as f32 / n);
    let mrr = ranks.iter().map(|&r| 1.0 / (r as f32 + 1.0)).sum::<f32>() / n;
    let mut sorted = ranks.clone();
    sorted.sort_unstable();
    let median_rank = sorted[sorted.len() / 2] as f32 + 1.0;
    Metrics {
        recall,
        mrr,
        median_rank,
    }
}

fn argmax_top(sim_row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..sim_row.len()).collect();
    idx.sort_by(|&a, &b| sim_row[b].partial_cmp(&sim_row[a]).unwrap());
    idx.truncate(k);
    idx
}

fn print_row(tag: &str, m: &Metrics) {
    println!(
        "{tag:30} {:6.3} {:6.3} {:6.3} {:6.3} {:8.1}",
        m.recall[0], m.recall[1], m.recall[2], m.mrr, m.median_rank
    );
}

fn main() -> Result<()> {
    let args = parse_args();
    let ds = load_dataset(&args.qa)?;
    println!(
        "loaded {} queries over {} candidate passages\n",
        ds.questions.len(),
        ds.passage_ids.len()
    );

    let device = Device::new_cuda(args.device).with_context(|| {
        format!("open CUDA:{} (try a free ordinal)", args.device)
    })?;

    // --- Concept vectors (quantized Mistral prefill), scoped so the model frees VRAM
    //     before BGE loads on the same GPU. ---
    let (cv_queries, cv_passages, n_layers) = {
        let api = Api::new().context("HF Hub init")?;
        let tok_repo = api.model(args.tok_repo.clone());
        let config_path = tok_repo.get("config.json").context("get config.json")?;
        let tok_path = tok_repo.get("tokenizer.json").context("get tokenizer.json")?;
        let gguf_path = api
            .model(args.gguf_repo.clone())
            .get(&args.gguf_file)
            .with_context(|| format!("get GGUF {}/{}", args.gguf_repo, args.gguf_file))?;

        let config: Config = serde_json::from_str(&fs::read_to_string(config_path)?)
            .context("parse config.json into Mistral Config")?;
        let tok = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let vb = VarBuilder::from_gguf(&gguf_path, &device).context("load GGUF")?;
        let model = QConceptModel::new(&config, vb).context("build quantized Mistral concept model")?;
        let n_layers = model.num_layers();
        println!(
            "loaded {} ({} layers, GGUF Q8_0) on CUDA:{}\n",
            args.gguf_file, n_layers, args.device
        );
        let q = concept_vectors(&model, &tok, &device, &ds.questions, "query")?;
        let p = concept_vectors(&model, &tok, &device, &ds.passage_text, "passage")?;
        (q, p, n_layers)
    };

    // --- BGE baseline: the real production embedder, same GPU. ---
    println!("\nencoding BGE baseline (the production embedder)...");
    let mut bge = CandleEmbeddingProvider::bge_small_on(EmbedDevice::Cuda(args.device))
        .context("init bge-small (candle)")?;
    let bge_q = bge.embed_query(ds.questions.iter().map(String::as_str).collect())?;
    let bge_p = bge.embed_document(ds.passage_text.iter().map(String::as_str).collect())?;
    let bge_sim = sim_matrix(&bge_q, &bge_p);
    let bge_m = evaluate(&bge_sim, &ds.gold_idx);

    // --- Per-layer concept-vector sweep. ---
    let mut cv_metrics = Vec::with_capacity(n_layers);
    let mut cv_sims = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let q: Vec<Vec<f32>> = cv_queries.iter().map(|t| t[layer].clone()).collect();
        let p: Vec<Vec<f32>> = cv_passages.iter().map(|t| t[layer].clone()).collect();
        let sim = sim_matrix(&q, &p);
        cv_metrics.push(evaluate(&sim, &ds.gold_idx));
        cv_sims.push(sim);
    }
    let best_layer = (0..n_layers)
        .max_by(|&a, &b| cv_metrics[a].recall[1].partial_cmp(&cv_metrics[b].recall[1]).unwrap())
        .unwrap();

    println!("\n==== RESULTS (gold rank over {} candidates) ====", ds.passage_ids.len());
    println!("{:30} {:>6} {:>6} {:>6} {:>6} {:>8}", "method", "r@1", "r@5", "r@10", "mrr", "medRank");
    print_row(&format!("BGE-small ({})", bge.provider_name()), &bge_m);
    for layer in 0..n_layers {
        let tag = if layer == best_layer {
            format!("cv L{layer}  <-best r@5")
        } else {
            format!("cv L{layer}")
        };
        print_row(&tag, &cv_metrics[layer]);
    }

    // --- Whitening sweep: can all-but-the-top recover a metric from the same vectors? ---
    // Per (layer, d): fit on the union of this layer's query+passage vectors, transform
    // both sides, re-score. Report the best layer for each d.
    const WHITEN_DIMS: [usize; 5] = [1, 2, 4, 8, 16];
    println!("\n==== WHITENED concept vectors (all-but-the-top, fit on union) ====");
    println!("{:30} {:>6} {:>6} {:>6} {:>6} {:>8}", "best layer @ d", "r@1", "r@5", "r@10", "mrr", "medRank");
    let mut whiten_best: Option<(usize, usize, Metrics)> = None;
    for &d in &WHITEN_DIMS {
        let mut best_for_d: Option<(usize, Metrics, Vec<Vec<f32>>)> = None;
        for layer in 0..n_layers {
            let q: Vec<Vec<f32>> = cv_queries.iter().map(|t| t[layer].clone()).collect();
            let p: Vec<Vec<f32>> = cv_passages.iter().map(|t| t[layer].clone()).collect();
            let union: Vec<Vec<f32>> = p.iter().chain(q.iter()).cloned().collect();
            let transform = fit_whitening(&union, d);
            let qw: Vec<Vec<f32>> = q.iter().map(|v| transform(v)).collect();
            let pw: Vec<Vec<f32>> = p.iter().map(|v| transform(v)).collect();
            let sim = sim_matrix(&qw, &pw);
            let m = evaluate(&sim, &ds.gold_idx);
            let better = best_for_d
                .as_ref()
                .map(|(_, bm, _)| m.recall[1] > bm.recall[1])
                .unwrap_or(true);
            if better {
                best_for_d = Some((layer, m, sim));
            }
        }
        let (layer, m, _) = best_for_d.unwrap();
        print_row(&format!("d={d} (L{layer})"), &m);
        if whiten_best
            .as_ref()
            .map(|(_, _, bm)| m.recall[1] > bm.recall[1])
            .unwrap_or(true)
        {
            whiten_best = Some((d, layer, m));
        }
    }
    let (bd, bl, bm) = whiten_best.unwrap();
    println!(
        "\nbest whitened: d={bd} L{bl} -> r@1={:.3} r@5={:.3} (BGE r@1={:.3} r@5={:.3})",
        bm.recall[0], bm.recall[1], bge_m.recall[0], bge_m.recall[1]
    );

    if args.dump_disagreements {
        let cv_sim = &cv_sims[best_layer];
        let disagree: Vec<usize> = (0..ds.questions.len())
            .filter(|&q| argmax_top(&bge_sim[q], 1)[0] != argmax_top(&cv_sim[q], 1)[0])
            .collect();
        println!(
            "\n==== DISAGREEMENTS: {}/{} queries differ on top-1 (BGE vs cv L{best_layer}) ====",
            disagree.len(),
            ds.questions.len()
        );
        for q in disagree {
            let gold = ds.gold_idx[q];
            println!("\nQ{q}: {}", ds.questions[q]);
            println!("  gold passage: {}", ds.passage_ids[gold]);
            for (name, sim) in [("BGE", &bge_sim), ("CV", &cv_sim)] {
                let picks: Vec<String> = argmax_top(&sim[q], 5)
                    .iter()
                    .map(|&p| {
                        let mark = if p == gold { "*GOLD*" } else { "" };
                        format!("{}{mark}", ds.passage_ids[p])
                    })
                    .collect();
                println!("  {name:4} top5: {}", picks.join(", "));
            }
        }
    }
    Ok(())
}
