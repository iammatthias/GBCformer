//! Writes a compiled checkpoint in the formats stock Llama runtimes load: a Hugging
//! Face `model.safetensors` with its `config.json` (`LlamaForCausalLM`), and a GGUF
//! (`llama` architecture, f32) laid out as llama.cpp's HF conversion lays it out.
//!
//! ```sh
//! cargo run --release --example export -- <cpu|cpu-memory|apu|control|ppu|one-model> <dir> [--gguf-only|--hf-only|--layer-shapes [--align N]]
//! ```
//!
//! What a stock runtime needs to compute exactly what `llama::Model::forward` computes:
//! every RMSNorm weight is `llama::GAMMA` and `rms_norm_eps` is `llama::EPS` (both 2^14,
//! `docs/stock-llama.md` §2), no biases, untied embeddings, `cfg.kv_heads` KV heads,
//! `head_dim` `llama::HEAD_DIM` = 64 (not hidden / heads) and `rope_theta` 1e38. The
//! export stores every matrix in f16, which holds each weight exactly
//! (`Model::check_f16`), and keeps the norm weights in f32, as llama.cpp's conversion
//! keeps 1-D tensors. It permutes the GGUF's `attn_q` and `attn_k` rows per head from
//! HF's rotate-half order to llama.cpp's interleaved pairs, as `convert_hf_to_gguf.py`'s
//! `LlamaModel.permute` does, so HF dim `j·32 + i` becomes `2i + j`. The tokenizer is
//! placeholder SentencePiece metadata, because the runtime gets token ids directly.
//!
//! `--layer-shapes` (GGUF only) drops every query head whose `o_proj` columns are all
//! zero and every FFN unit whose gate row, up row or down column is all zero, which
//! changes no value, and writes `head_count` and `feed_forward_length` as per-layer
//! arrays. Stock llama.cpp builds every LLAMA layer at layer 0's shape, so it needs
//! `tools/stock-llama/layer-shapes.patch`, and for the stored runs below
//! `stream-slices.patch` and `stored-runs.patch` after it. A layer keeps at least one head
//! and one unit.
//! It also stores only the LM head's run of rows that are not the fill row
//! (`gguf::HeadRows`, `gbcformer.output.rows` and `.fill`), and names for each stream
//! of an overlay the run of tokens it can decode (`gbcformer.stream.<part>.head`,
//! `gguf::stream_head`), and of each layer's `ffn_up` only the run of columns its units
//! read (`gguf::up_cols`, `gbcformer.ffn_up.cols`), and of the embedding and the head only
//! the prefix of columns that are not zero (`gguf::col_prefixes`,
//! `gbcformer.token_embd.cols` and `gbcformer.output.cols`); llama.cpp then needs
//! `tools/stock-llama/stored-runs.patch` too.
//!
//! `--align N` pads each layer's FFN width up to a multiple of N with all-zero units,
//! which add exactly 0 (silu(0)·0). Metal's f16 `mul_mm` falls to bounds-checked scalar
//! loads when `ffn_down`'s inner dimension, the FFN width, is not a multiple of 32, and
//! its few-row MMA kernel needs a multiple of 64. Head widths are heads × 64 already.
//!
//! The checkpoint name `tiny` writes a checkpoint with the one model's 14 layers at
//! hidden 64 and every weight 1/64, to measure a runtime's fixed cost per pass.

use gbcformer::circuit::Circuit;
use gbcformer::llama::{Config, Model, EPS, GAMMA, HEAD_DIM, ROPE_THETA};
use gbcformer::memory::attn::Params;
use gbcformer::gguf::{f16_bits, interleave, keep, HeadRows, Keep};
use gbcformer::{apu_circuit, control_circuit, one_model, ppu_circuit, sm83};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Positions every stream reaches stay below this, and f32 holds every position below
/// 2^24 exactly.
const MAX_POSITIONS: usize = 1 << 23;

fn build(name: &str) -> (Model, Vec<gbcformer::engine::StreamSpec>) {
    let mut specs = Vec::new();
    let (c, memory): (Circuit, bool) = match name {
        "cpu" => (sm83::build(), false),
        "cpu-memory" => (sm83::build_mem(), true),
        "apu" => (apu_circuit::build(), false),
        "control" => (control_circuit::build(), false),
        "ppu" => (ppu_circuit::circuit::build(), true),
        "one-model" => {
            let (c, s) = one_model::circuit(&one_model::parts());
            specs = s;
            (c, true)
        }
        _ => panic!("unknown checkpoint {name}: cpu, cpu-memory, apu, control, ppu or one-model"),
    };
    (if memory { Model::compile_mem(&c, Params::memory()) } else { Model::compile(&c) }, specs)
}

/// One tensor: the HF name, the GGUF name, its shape outermost first (HF order), and its
/// values row-major in that shape. A matrix is stored in f16, a vector in f32.
struct Tensor<'a> {
    hf: String,
    gguf: String,
    shape: Vec<usize>,
    data: Cow<'a, [f32]>,
}

impl Tensor<'_> {
    fn half(&self) -> bool {
        self.shape.len() > 1
    }
    fn bytes(&self) -> usize {
        self.data.len() * if self.half() { 2 } else { 4 }
    }
}

/// The rows `keep` names of a matrix with rows of `width`, each `block` rows long, then
/// zero blocks up to `to` blocks.
fn rows(w: &[f32], width: usize, block: usize, keep: &[usize], to: usize) -> Vec<f32> {
    let mut out: Vec<f32> = keep.iter().flat_map(|&k| &w[k * block * width..][..block * width]).copied().collect();
    out.resize(to * block * width, 0.0);
    out
}

/// The columns `keep` names, each `block` wide, of a matrix with rows of `width`, then
/// zero columns up to `to` blocks per row.
fn cols(w: &[f32], width: usize, block: usize, keep: &[usize], to: usize) -> Vec<f32> {
    let zeros = (to - keep.len()) * block;
    w.chunks(width).flat_map(|row| keep.iter().flat_map(move |&k| row[k * block..][..block].iter().copied()).chain(std::iter::repeat_n(0.0, zeros))).collect()
}

/// The checkpoint's tensors. `shift` (GGUF only, `gguf::qk_shift`, empty for none)
/// divides each layer's query weights by 2^shift and multiplies its key weights by it,
/// which leaves every logit the same product.
fn tensors<'a>(m: &'a Model, gguf: bool, shapes: Option<&[Keep]>, head: &HeadRows, shift: &[u32]) -> Vec<Tensor<'a>> {
    let up_cols = shapes.map(|s| gbcformer::gguf::up_cols(m, s));
    // The prefix of columns a shaped file stores of the embedding and of the head.
    let (embed_cols, head_cols) = if shapes.is_some() { gbcformer::gguf::col_prefixes(m, head) } else { (m.cfg.hidden, m.cfg.hidden) };
    let prefix = |w: &'a [f32], cols: usize| -> Cow<'a, [f32]> { if cols == m.cfg.hidden { w.into() } else { w.chunks(m.cfg.hidden).flat_map(|r| r[..cols].iter().copied()).collect::<Vec<f32>>().into() } };
    let scaled = |w: Cow<'a, [f32]>, f: f32| -> Cow<'a, [f32]> { if f == 1.0 { w } else { w.iter().map(|x| x * f).collect::<Vec<f32>>().into() } };
    let c = &m.cfg;
    let (d, a, i, v) = (c.hidden, c.attn(), c.intermediate, c.vocab);
    let ones: Cow<'a, [f32]> = vec![GAMMA; d].into();
    let kv = c.kv_dim();
    fn t(hf: String, gguf: String, shape: Vec<usize>, data: Cow<'_, [f32]>) -> Tensor<'_> {
        Tensor { hf, gguf, shape, data }
    }
    let mut out = vec![t("model.embed_tokens.weight".into(), "token_embd.weight".into(), vec![v, embed_cols], prefix(&m.embed, embed_cols))];
    for (n, l) in m.layers.iter().enumerate() {
        let (p, b) = (format!("model.layers.{n}"), format!("blk.{n}"));
        let f = (1u32 << shift.get(n).copied().unwrap_or(0)) as f32;
        let qk = |w: &'a [f32], heads: usize| -> Cow<'a, [f32]> { if gguf { interleave(w, heads, d, false).into() } else { w.into() } };
        type W<'a> = Cow<'a, [f32]>;
        let (q, o, gate, up, down, a, i, up_d): (W<'a>, W<'a>, W<'a>, W<'a>, W<'a>, usize, usize, usize) = match shapes {
            None => (qk(&l.q, c.heads), (&l.o[..]).into(), (&l.gate[..]).into(), (&l.up[..]).into(), (&l.down[..]).into(), a, i, d),
            Some(s) => {
                let (h, u, w) = (&s[n].heads, &s[n].units, s[n].width);
                let (first, count) = up_cols.as_ref().expect("a shaped file's up columns")[n];
                let up: Vec<f32> = rows(&l.up, d, 1, u, w).chunks(d).flat_map(|r| r[first..][..count].iter().copied()).collect();
                (interleave(&rows(&l.q, d, HEAD_DIM, h, h.len()), h.len(), d, false).into(), cols(&l.o, a, HEAD_DIM, h, h.len()).into(), rows(&l.gate, d, 1, u, w).into(), up.into(), cols(&l.down, i, 1, u, w).into(), h.len() * HEAD_DIM, w, count)
            }
        };
        out.push(t(format!("{p}.input_layernorm.weight"), format!("{b}.attn_norm.weight"), vec![d], ones.clone()));
        out.push(t(format!("{p}.self_attn.q_proj.weight"), format!("{b}.attn_q.weight"), vec![a, d], scaled(q, 1.0 / f)));
        out.push(t(format!("{p}.self_attn.k_proj.weight"), format!("{b}.attn_k.weight"), vec![kv, d], scaled(qk(&l.k, c.kv_heads), f)));
        out.push(t(format!("{p}.self_attn.v_proj.weight"), format!("{b}.attn_v.weight"), vec![kv, d], (&l.v[..]).into()));
        out.push(t(format!("{p}.self_attn.o_proj.weight"), format!("{b}.attn_output.weight"), vec![d, a], o));
        out.push(t(format!("{p}.post_attention_layernorm.weight"), format!("{b}.ffn_norm.weight"), vec![d], ones.clone()));
        out.push(t(format!("{p}.mlp.gate_proj.weight"), format!("{b}.ffn_gate.weight"), vec![i, d], gate));
        out.push(t(format!("{p}.mlp.up_proj.weight"), format!("{b}.ffn_up.weight"), vec![i, up_d], up));
        out.push(t(format!("{p}.mlp.down_proj.weight"), format!("{b}.ffn_down.weight"), vec![d, i], down));
    }
    out.push(t("model.norm.weight".into(), "output_norm.weight".into(), vec![d], ones));
    out.push(t("lm_head.weight".into(), "output.weight".into(), vec![head.count, head_cols], prefix(&m.lm_head[head.first * d..][..head.count * d], head_cols)));
    out
}

fn write_tensor(w: &mut impl Write, t: &Tensor<'_>) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(1 << 22);
    for chunk in t.data.chunks(1 << 20) {
        buf.clear();
        if t.half() {
            buf.extend(chunk.iter().flat_map(|&f| f16_bits(f).to_le_bytes()));
        } else {
            buf.extend(chunk.iter().flat_map(|f| f.to_le_bytes()));
        }
        w.write_all(&buf)?;
    }
    Ok(())
}

fn safetensors(m: &Model, path: &Path) -> std::io::Result<()> {
    write_safetensors(&tensors(m, false, None, &HeadRows::whole(m.cfg.vocab), &[]), path)
}

fn write_safetensors(ts: &[Tensor<'_>], path: &Path) -> std::io::Result<()> {
    let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut offset = 0usize;
    for t in ts {
        let bytes = t.bytes();
        let shape: Vec<String> = t.shape.iter().map(|s| s.to_string()).collect();
        let dtype = if t.half() { "F16" } else { "F32" };
        header += &format!(",\"{}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}", t.hf, shape.join(","), offset + bytes);
        offset += bytes;
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for t in ts {
        write_tensor(&mut w, t)?;
    }
    w.flush()
}

fn config_json(m: &Model, name: &str) -> String {
    let c = &m.cfg;
    let slots: Vec<String> = m.slots.iter().map(|s| format!("\"{s}\"")).collect();
    format!(
        r#"{{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": {},
  "intermediate_size": {},
  "num_hidden_layers": {},
  "num_attention_heads": {},
  "num_key_value_heads": {},
  "head_dim": {HEAD_DIM},
  "hidden_act": "silu",
  "max_position_embeddings": {MAX_POSITIONS},
  "rms_norm_eps": {EPS:e},
  "rope_theta": {ROPE_THETA:e},
  "rope_scaling": null,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "mlp_bias": false,
  "pretraining_tp": 1,
  "tie_word_embeddings": false,
  "vocab_size": {},
  "bos_token_id": 1,
  "eos_token_id": 2,
  "torch_dtype": "float16",
  "use_cache": true,
  "gbcformer": {{
    "checkpoint": "{name}",
    "n_inputs": {},
    "n_outputs": {},
    "n_const": {},
    "families": {},
    "slots": [{}]
  }}
}}
"#,
        c.hidden,
        c.intermediate,
        c.layers,
        c.heads,
        c.kv_heads,
        c.vocab,
        c.n_inputs,
        c.n_outputs,
        c.n_const,
        c.families,
        slots.join(", ")
    )
}

/// GGUF v3 metadata values.
#[derive(Clone)]
enum Val {
    U32(u32),
    U32s(Vec<u32>),
    Bool(bool),
    F32(f32),
    Str(String),
    Strs(Vec<String>),
    F32s(Vec<f32>),
    I32s(Vec<i32>),
}

fn gguf_str(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn gguf_val(w: &mut impl Write, v: &Val) -> std::io::Result<()> {
    const U32: u32 = 4;
    const I32: u32 = 5;
    const F32: u32 = 6;
    const BOOL: u32 = 7;
    const STR: u32 = 8;
    const ARR: u32 = 9;
    match v {
        Val::U32(x) => {
            w.write_all(&U32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Val::Bool(x) => {
            w.write_all(&BOOL.to_le_bytes())?;
            w.write_all(&[*x as u8])
        }
        Val::F32(x) => {
            w.write_all(&F32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Val::Str(s) => {
            w.write_all(&STR.to_le_bytes())?;
            gguf_str(w, s)
        }
        Val::Strs(xs) => {
            w.write_all(&ARR.to_le_bytes())?;
            w.write_all(&STR.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            xs.iter().try_for_each(|s| gguf_str(w, s))
        }
        Val::F32s(xs) => {
            w.write_all(&ARR.to_le_bytes())?;
            w.write_all(&F32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            xs.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        Val::U32s(xs) => {
            w.write_all(&ARR.to_le_bytes())?;
            w.write_all(&U32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            xs.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        Val::I32s(xs) => {
            w.write_all(&ARR.to_le_bytes())?;
            w.write_all(&I32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            xs.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
    }
}

/// An overlay checkpoint's streams as metadata a runtime can run one part from
/// (`Model::slice`). For each part it writes the channel prefix the part reads
/// (`embedding_length`), the layers it runs (`block_count`), and per layer the file's
/// query heads it keeps (`heads`, the count and then their indices) and its FFN units as
/// runs of file rows (`ffn`, the run count and then each run's first row and length).
/// Every row the slice keeps reads only the prefix's columns, and everything else adds
/// exactly 0 at the part's positions (`Model::stream`). `head` is the run of tokens the
/// part can decode (`gguf::stream_head`), and `width` the prefix of channels a runtime may
/// cut the part's matmuls to (`gguf::stream_width`). Empty for any other checkpoint.
fn stream_keys(m: &Model, shapes: &[Keep], rows: &HeadRows) -> Vec<(String, Val)> {
    if m.overlay.is_none() {
        return Vec::new();
    }
    let mut kv = Vec::new();
    for p in one_model::Part::ALL {
        let s = m.slice(p as usize, true);
        let at = |xs: &[usize], ys: &[usize]| -> Vec<u32> { ys.iter().filter_map(|y| xs.binary_search(y).ok()).map(|i| i as u32).collect() };
        let (mut heads, mut ffn) = (Vec::new(), Vec::new());
        for (li, (us, hs)) in s.units.iter().zip(&s.heads).enumerate() {
            let h = at(&shapes[li].heads, hs);
            heads.push(h.len() as u32);
            heads.extend(h);
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for r in at(&shapes[li].units, us) {
                match runs.last_mut() {
                    Some((first, len)) if *first + *len == r => *len += 1,
                    _ => runs.push((r, 1)),
                }
            }
            ffn.push(runs.len() as u32);
            ffn.extend(runs.into_iter().flat_map(|(a, b)| [a, b]));
        }
        let k = |f: &str| format!("gbcformer.stream.{}.{f}", p.name());
        kv.push((k("embedding_length"), Val::U32(s.hidden as u32)));
        kv.push((k("block_count"), Val::U32(s.units.len() as u32)));
        kv.push((k("heads"), Val::U32s(heads)));
        kv.push((k("ffn"), Val::U32s(ffn)));
        let (first, count) = gbcformer::gguf::stream_head(m, &s, rows);
        eprintln!("stream {}: decodes tokens {first}..{} ({count} rows of the head's {})", p.name(), first + count, rows.count);
        kv.push((k("head"), Val::U32s(vec![first as u32, count as u32])));
        let width = gbcformer::gguf::stream_width(m, &s, rows);
        eprintln!("stream {}: runs on the first {width} of {} channels (its prefix is {})", p.name(), m.cfg.hidden, s.hidden);
        kv.push((k("width"), Val::U32(width as u32)));
    }
    kv
}

fn gguf(c: &Config, ts: Vec<Tensor<'_>>, shapes: Option<&[Keep]>, name: &str, streams: Vec<(String, Val)>, path: &Path) -> std::io::Result<()> {
    const ALIGN: usize = 32;
    let u = |x: usize| Val::U32(x as u32);
    let per_layer = |n: fn(&Keep) -> usize, all: usize| match shapes {
        Some(s) => Val::U32s(s.iter().map(|k| n(k) as u32).collect()),
        None => u(all),
    };
    let mut kv: Vec<(&str, Val)> = vec![
        ("general.architecture", Val::Str("llama".into())),
        ("general.name", Val::Str(format!("gbcformer-{name}"))),
        ("general.alignment", u(ALIGN)),
        ("general.file_type", u(1)),
        ("llama.vocab_size", u(c.vocab)),
        ("llama.context_length", u(MAX_POSITIONS)),
        ("llama.embedding_length", u(c.hidden)),
        ("llama.block_count", u(c.layers)),
        ("llama.feed_forward_length", per_layer(|k| k.width, c.intermediate)),
        ("llama.attention.head_count", per_layer(|k| k.heads.len(), c.heads)),
        ("llama.attention.head_count_kv", u(c.kv_heads)),
        ("llama.attention.key_length", u(HEAD_DIM)),
        ("llama.attention.value_length", u(HEAD_DIM)),
        ("llama.rope.dimension_count", u(HEAD_DIM)),
        ("llama.rope.freq_base", Val::F32(ROPE_THETA as f32)),
        ("llama.attention.layer_norm_rms_epsilon", Val::F32(EPS)),
        ("tokenizer.ggml.model", Val::Str("llama".into())),
        ("tokenizer.ggml.tokens", Val::Strs((0..c.vocab).map(|t| format!("<gbc{t}>")).collect())),
        ("tokenizer.ggml.scores", Val::F32s(vec![0.0; c.vocab])),
        ("tokenizer.ggml.token_type", Val::I32s(vec![1; c.vocab])),
        ("tokenizer.ggml.bos_token_id", u(1)),
        ("tokenizer.ggml.eos_token_id", u(2)),
        ("tokenizer.ggml.unknown_token_id", u(0)),
        ("tokenizer.ggml.add_bos_token", Val::Bool(false)),
    ];
    kv.extend(streams.iter().map(|(k, v)| (k.as_str(), v.clone())));
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"GGUF")?;
    w.write_all(&3u32.to_le_bytes())?;
    w.write_all(&(ts.len() as u64).to_le_bytes())?;
    w.write_all(&(kv.len() as u64).to_le_bytes())?;
    let mut at = 24usize;
    for (k, v) in &kv {
        let mut buf = Vec::new();
        gguf_str(&mut buf, k)?;
        gguf_val(&mut buf, v)?;
        at += buf.len();
        w.write_all(&buf)?;
    }
    let mut offset = 0usize;
    for t in &ts {
        let mut buf = Vec::new();
        gguf_str(&mut buf, &t.gguf)?;
        buf.extend((t.shape.len() as u32).to_le_bytes());
        // GGUF lists dimensions innermost first.
        for &s in t.shape.iter().rev() {
            buf.extend((s as u64).to_le_bytes());
        }
        buf.extend((t.half() as u32).to_le_bytes()); // GGML_TYPE_F16 = 1, GGML_TYPE_F32 = 0
        buf.extend((offset as u64).to_le_bytes());
        at += buf.len();
        w.write_all(&buf)?;
        offset += t.bytes().div_ceil(ALIGN) * ALIGN;
    }
    w.write_all(&vec![0u8; at.div_ceil(ALIGN) * ALIGN - at])?;
    for t in &ts {
        write_tensor(&mut w, t)?;
        let bytes = t.bytes();
        w.write_all(&vec![0u8; bytes.div_ceil(ALIGN) * ALIGN - bytes])?;
    }
    w.flush()
}

/// The llama.cpp patches the layer-shaped GGUF needs, in the order they apply, whose
/// digests its manifest names.
const PATCHES: [(&str, &str); 3] = [
    ("layer-shapes.patch", include_str!("../tools/stock-llama/layer-shapes.patch")),
    ("stream-slices.patch", include_str!("../tools/stock-llama/stream-slices.patch")),
    ("stored-runs.patch", include_str!("../tools/stock-llama/stored-runs.patch")),
];

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(name), Some(dir)) = (args.first(), args.get(1)) else {
        eprintln!("usage: export <cpu|cpu-memory|apu|control|ppu|one-model|tiny|synth> <dir> [--gguf-only|--hf-only|--layer-shapes [--align N]]");
        std::process::exit(2);
    };
    let flag = |f: &str| args.iter().any(|a| a == f);
    let layer_shapes = flag("--layer-shapes");
    let (hf, gg) = (!flag("--gguf-only") && !layer_shapes, !flag("--hf-only"));
    let dir = Path::new(dir);
    std::fs::create_dir_all(dir)?;
    if name == "tiny" {
        // A checkpoint to measure a runtime's fixed cost per pass, with the one model's 14
        // layers at hidden 64, two heads of 64, intermediate 64, 256 tokens and every
        // weight 1/64.
        let c = Config { hidden: 64, intermediate: 64, layers: 14, heads: 2, kv_heads: 2, vocab: 256, n_inputs: 0, n_outputs: 0, n_const: 0, families: 0 };
        let w = |n: usize| -> Cow<'static, [f32]> { vec![1.0 / 64.0; n].into() };
        let (d, a, i, v) = (c.hidden, c.attn(), c.intermediate, c.vocab);
        let t = |hf: String, gguf: String, shape: Vec<usize>| Tensor { hf, data: w(shape.iter().product()), gguf, shape };
        let tiny = || {
            let mut ts = vec![t("model.embed_tokens.weight".into(), "token_embd.weight".into(), vec![v, d])];
            for n in 0..c.layers {
                for (h, g, shape) in [("input_layernorm", "attn_norm", vec![d]), ("self_attn.q_proj", "attn_q", vec![a, d]), ("self_attn.k_proj", "attn_k", vec![a, d]), ("self_attn.v_proj", "attn_v", vec![a, d]), ("self_attn.o_proj", "attn_output", vec![d, a]), ("post_attention_layernorm", "ffn_norm", vec![d]), ("mlp.gate_proj", "ffn_gate", vec![i, d]), ("mlp.up_proj", "ffn_up", vec![i, d]), ("mlp.down_proj", "ffn_down", vec![d, i])] {
                    ts.push(t(format!("model.layers.{n}.{h}.weight"), format!("blk.{n}.{g}.weight"), shape));
                }
            }
            ts.push(t("model.norm.weight".into(), "output_norm.weight".into(), vec![d]));
            ts.push(t("lm_head.weight".into(), "output.weight".into(), vec![v, d]));
            ts
        };
        write_safetensors(&tiny(), &dir.join("model.safetensors"))?;
        return gguf(&c, tiny(), None, "tiny", Vec::new(), &dir.join("model.gguf"));
    }
    if name == "synth" {
        // A uniform Llama checkpoint of a given shape with every matrix zero, written as
        // the one model's uniform file is: one KV head, f16 matrices, the same metadata.
        // It decodes nothing; `stock-llama shapes` times a runtime's pass over it.
        let n = |i: usize, what: &str| -> usize { args.get(i).and_then(|a| a.parse().ok()).unwrap_or_else(|| panic!("export synth <dir> <hidden> <layers> <ffn> <heads> <vocab>: {what}")) };
        let c = Config { hidden: n(2, "hidden"), layers: n(3, "layers"), intermediate: n(4, "ffn"), heads: n(5, "heads"), kv_heads: 1, vocab: n(6, "vocab"), n_inputs: 0, n_outputs: 0, n_const: 0, families: 0 };
        let (d, a, i, v, kv) = (c.hidden, c.attn(), c.intermediate, c.vocab, c.kv_dim());
        let zeros: &'static [f32] = Box::leak(vec![0.0f32; (v * d).max(i * d).max(a * d)].into_boxed_slice());
        let t = |gguf: String, shape: Vec<usize>| Tensor { hf: String::new(), data: if shape.len() == 1 { vec![1.0; shape[0]].into() } else { zeros[..shape.iter().product()].into() }, gguf, shape };
        let mut ts = vec![t("token_embd.weight".into(), vec![v, d])];
        for l in 0..c.layers {
            for (g, shape) in [("attn_norm", vec![d]), ("attn_q", vec![a, d]), ("attn_k", vec![kv, d]), ("attn_v", vec![kv, d]), ("attn_output", vec![d, a]), ("ffn_norm", vec![d]), ("ffn_gate", vec![i, d]), ("ffn_up", vec![i, d]), ("ffn_down", vec![d, i])] {
                ts.push(t(format!("blk.{l}.{g}.weight"), shape));
            }
        }
        ts.push(t("output_norm.weight".into(), vec![d]));
        ts.push(t("output.weight".into(), vec![v, d]));
        eprintln!("synth: hidden {} layers {} ffn {} heads {} vocab {}: {} values", c.hidden, c.layers, c.intermediate, c.heads, c.vocab, ts.iter().map(|t| t.data.len()).sum::<usize>());
        return gguf(&c, ts, None, "synth", Vec::new(), &dir.join("model.gguf"));
    }
    let t = std::time::Instant::now();
    let (m, specs) = build(name);
    let c = &m.cfg;
    eprintln!("{name}: hidden {} intermediate {} layers {} heads {} vocab {} seq {}: {:.1}M parameters, compiled in {:.1}s", c.hidden, c.intermediate, c.layers, c.heads, c.vocab, c.seq(), c.params() as f64 / 1e6, t.elapsed().as_secs_f64());
    if hf {
        std::fs::write(dir.join("config.json"), config_json(&m, name))?;
        safetensors(&m, &dir.join("model.safetensors"))?;
        eprintln!("wrote {}", dir.join("model.safetensors").display());
    }
    if gg {
        let align = args.iter().position(|a| a == "--align").map_or(1, |p| args[p + 1].parse().expect("--align N"));
        let shapes = layer_shapes.then(|| keep(&m, align));
        if let Some(s) = &shapes {
            let heads: Vec<usize> = s.iter().map(|k| k.heads.len()).collect();
            let units: Vec<usize> = s.iter().map(|k| k.width).collect();
            eprintln!("layer shapes: heads {heads:?} ({} of {}), units {units:?}", heads.iter().sum::<usize>(), c.heads * c.layers);
        }
        let shift = gbcformer::gguf::qk_shift(&m);
        let bounds: Vec<(u32, u32)> = m.attention_bounds().iter().map(|&(q, k)| (q as u32, k as u32)).collect();
        eprintln!("per layer, queries and keys up to {bounds:?} at any position: the file divides the queries by 2^shift and multiplies the keys by it, shift {shift:?}");
        let head = if layer_shapes { HeadRows::of(&m) } else { HeadRows::whole(c.vocab) };
        let ts = tensors(&m, true, shapes.as_deref(), &head, &shift);
        let stored = ts.iter().map(|t| t.data.len()).sum::<usize>();
        eprintln!("gguf: {:.1}M dense parameters ({stored})", stored as f64 / 1e6);
        let mut streams = shapes.as_deref().map_or_else(Vec::new, |s| stream_keys(&m, s, &head));
        if let Some(s) = &shapes {
            assert_eq!(stored, gbcformer::gguf::stored_params(&m, s), "gguf::stored_params counts what the file stores");
            eprintln!("LM head: rows {}..{} of {} stored, every other row {} on channel 0", head.first, head.first + head.count, c.vocab, head.fill);
            streams.push((gbcformer::gguf::HEAD_ROWS_KEY.to_string(), Val::U32s(vec![head.first as u32, head.count as u32])));
            streams.push((gbcformer::gguf::HEAD_FILL_KEY.to_string(), Val::F32(head.fill)));
            let (embed_cols, head_cols) = gbcformer::gguf::col_prefixes(&m, &head);
            eprintln!("token_embd: the first {embed_cols} of {} columns stored; output: the first {head_cols}", c.hidden);
            streams.push((gbcformer::gguf::EMBED_COLS_KEY.to_string(), Val::U32(embed_cols as u32)));
            streams.push((gbcformer::gguf::HEAD_COLS_KEY.to_string(), Val::U32(head_cols as u32)));
            let ups = gbcformer::gguf::up_cols(&m, s);
            eprintln!("ffn_up: per layer, the columns of channels {:?}", ups.iter().map(|&(first, count)| first..first + count).collect::<Vec<_>>());
            streams.push((gbcformer::gguf::UP_COLS_KEY.to_string(), Val::U32s(ups.iter().flat_map(|&(first, count)| [first as u32, count as u32]).collect())));
        }
        streams.push((gbcformer::gguf::QK_SHIFT_KEY.to_string(), Val::U32s(shift)));
        let meta = serde_json::to_string(&gbcformer::gguf::Meta::of(&m, specs.clone())).expect("the metadata serializes");
        streams.push((gbcformer::gguf::META_KEY.to_string(), Val::Str(meta)));
        if name == "one-model" {
            let wiring = serde_json::to_string(&gbcformer::gbc_wiring::wiring()).expect("the wiring serializes");
            let llama_cpp = if layer_shapes { format!("llama.cpp b12070 with tools/stock-llama/ {}", PATCHES.map(|(name, text)| format!("{name} (sha1 {})", gbcformer::wiring::sha1(text))).join(", ")) } else { "llama.cpp b12070, unpatched (uniform layers)".to_string() };
            let manifest = gbcformer::wiring::Manifest { wiring_version: gbcformer::wiring::WIRING_VERSION, wiring_sha1: gbcformer::wiring::sha1(&wiring), tensors_sha1: gbcformer::gguf::digest(&m).0, vocab: m.cfg.vocab, llama_cpp };
            streams.push((gbcformer::wiring::MANIFEST_KEY.to_string(), Val::Str(serde_json::to_string(&manifest).expect("the manifest serializes"))));
            streams.push((gbcformer::wiring::WIRING_KEY.to_string(), Val::Str(wiring)));
        }
        gguf(&m.cfg, ts, shapes.as_deref(), name, streams, &dir.join("model.gguf"))?;
        eprintln!("wrote {}", dir.join("model.gguf").display());
    }
    Ok(())
}
