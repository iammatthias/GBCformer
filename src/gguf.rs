//! The checkpoint as the GGUF file a stock runtime loads (`examples/export.rs`), read
//! back into a [`Model`], so the weights the engine computes are the file's.
//!
//! The file holds the weights only. The compiler supplies what is not a weight (the
//! layout, the slot names, the wire ranges the guards use, the memory spec), and the
//! loader expands each of the file's tensors back to the compiled layout. A
//! `--layer-shapes` file keeps only the query heads and FFN units that are not all zero,
//! padded with zero units, and its `attn_q` and `attn_k` rows are in llama.cpp's
//! interleaved pair order. The loader proves the expansion bit-identical to the
//! compiled tensor before it replaces it, so every weight the engine reads came from the
//! file and equals the compiler's.

use crate::llama::{tok, Model, Slice, CONSTANTS, GAMMA, HEAD_DIM, UNIT};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The query heads and FFN units a layer keeps, by index, and its FFN width: the kept
/// units, then all-zero units up to the width. A layer keeps at least one of each.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Keep {
    pub heads: Vec<usize>,
    pub units: Vec<usize>,
    pub width: usize,
}

/// The metadata key naming `qk_shift`, one value per layer. Layer l's `attn_q` is the
/// checkpoint's divided by 2^shift[l], and its `attn_k` is multiplied by 2^shift[l].
pub const QK_SHIFT_KEY: &str = "gbcformer.qk_shift";

/// The largest finite f16.
const F16_MAX: f32 = 65504.0;

/// Per layer, the power of two a GGUF divides every query weight by and multiplies every
/// key weight by: the least that keeps every query and key entry finite in f16 at every
/// position (`Model::attention_bounds`). Metal's flash attention holds the query in f16,
/// and an overlay's memory head reads, at another part's positions, channels that hold
/// that part's nibbles. Each logit `q·k` is the same product, so a runtime computes the
/// same logits bit for bit. Both factors are exact in f16 (the export asserts every
/// weight), and the keys' RoPE partners (`2^shift · sin`, which an f16 cache does not
/// flush to 0 once shifted) leave the same products below 2^-38 that an f32 cache leaves
/// (`docs/stock-llama.md` §4), which no logit holding a multiple of 1,024 keeps.
pub fn qk_shift(m: &Model) -> Vec<u32> {
    m.attention_bounds()
        .iter()
        .enumerate()
        .map(|(l, &(q, k))| {
            let s = (0..16).find(|&s| q / (1u32 << s) as f32 <= F16_MAX && k * (1u32 << s) as f32 <= F16_MAX);
            s.unwrap_or_else(|| panic!("layer {l}: no power of two keeps the queries (up to {q}) and the keys (up to {k}) finite in f16"))
        })
        .collect()
}

/// Each layer's `Keep`, with every FFN width a multiple of `align`.
pub fn keep(m: &Model, align: usize) -> Vec<Keep> {
    let c = &m.cfg;
    let (d, a, i) = (c.hidden, c.attn(), c.intermediate);
    let zero = |xs: &[f32]| xs.iter().all(|&x| x == 0.0);
    let or_first = |xs: Vec<usize>| if xs.is_empty() { vec![0] } else { xs };
    m.layers
        .iter()
        .map(|l| {
            let heads = or_first((0..c.heads).filter(|h| (0..d).any(|r| !zero(&l.o[r * a + h * HEAD_DIM..][..HEAD_DIM]))).collect());
            let units: Vec<usize> = or_first((0..i).filter(|&u| !zero(&l.gate[u * d..][..d]) && !zero(&l.up[u * d..][..d]) && (0..d).any(|r| l.down[r * i + u] != 0.0)).collect());
            Keep { heads, width: units.len().next_multiple_of(align), units }
        })
        .collect()
}

/// The metadata key naming the rows of the LM head a layer-shaped file stores, as
/// `[first, count]` ([`HeadRows`]). Without it the file stores every row.
pub const HEAD_ROWS_KEY: &str = "gbcformer.output.rows";

/// The metadata key naming the weight every row the file leaves out holds on channel 0,
/// its only weight ([`HeadRows::fill`]).
pub const HEAD_FILL_KEY: &str = "gbcformer.output.fill";

/// Whether the constant channels hold `CONSTANTS` at every token and position: every
/// embedding holds them (as `CONSTANTS · UNIT`, which a norm reads as the constant) and
/// no head or unit of any layer writes them.
fn constants_hold(m: &Model) -> bool {
    let c = &m.cfg;
    let (d, a, i) = (c.hidden, c.attn(), c.intermediate);
    let embedded = m.embed.chunks(d).all(|r| r.iter().zip(CONSTANTS).all(|(&x, k)| x == k * UNIT));
    embedded && m.layers.iter().all(|l| l.o[..CONSTANTS.len() * a].iter().chain(&l.down[..CONSTANTS.len() * i]).all(|&w| w == 0.0))
}

/// The run of LM head rows a layer-shaped file stores: tokens `first..first + count`.
/// Every other token's row is the fill row, `fill` on channel 0 and 0 elsewhere.
///
/// Channel 0 holds the constant 1 at every token and nothing writes it (`of` checks
/// both, `constants_hold`), so a fill row scores exactly `fill` at every position. A row
/// of scores over the run alone therefore has the whole vocabulary's first largest score
/// whenever its own largest is above `fill`, and a runtime that scores only the run
/// checks that (`decides`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadRows {
    pub first: usize,
    pub count: usize,
    pub fill: f32,
}

impl HeadRows {
    /// Every row, for a file without `HEAD_ROWS_KEY`.
    pub fn whole(vocab: usize) -> HeadRows {
        HeadRows { first: 0, count: vocab, fill: 0.0 }
    }

    /// The shortest run that leaves out only rows equal to token 0's, when that row is
    /// a fill row and the constant channels hold; else every row.
    pub fn of(m: &Model) -> HeadRows {
        let (d, v) = (m.cfg.hidden, m.cfg.vocab);
        let row = |t: usize| &m.lm_head[t * d..][..d];
        let fill = row(0);
        let same = |t: usize| row(t).iter().zip(fill).all(|(x, y)| x.to_bits() == y.to_bits());
        let (Some(first), true) = ((0..v).find(|&t| !same(t)), constants_hold(m) && fill[1..].iter().all(|&w| w == 0.0)) else { return HeadRows::whole(v) };
        let last = (0..v).rfind(|&t| !same(t)).expect("a row differs");
        HeadRows { first, count: last + 1 - first, fill: fill[0] }
    }

    /// Whether a row of scores over the run alone decides the whole vocabulary's
    /// choice: its largest score is above every row left out, which scores `fill`.
    pub fn decides(&self, vocab: usize, largest: f32) -> bool {
        self.count == vocab || largest > self.fill
    }

    /// The whole head from the file's run, `width` channels a row.
    fn expand(&self, run: &[f32], width: usize, vocab: usize) -> Vec<f32> {
        let mut full = vec![0f32; vocab * width];
        if self.count < vocab {
            full.chunks_mut(width).for_each(|r| r[0] = self.fill);
        }
        full[self.first * width..][..self.count * width].copy_from_slice(run);
        full
    }
}

/// The run of tokens stream `s` of an overlay checkpoint can decode, inside the stored
/// run `rows`, as `(first, count)`: a runtime that runs the stream's slice may score
/// only these rows of the head.
///
/// A channel holds exactly 0 at every token of the stream's passes when no token a pass
/// holds sets it (a pass holds no record token: no `REC`, `CELL`, `RD` or null), no
/// kept head writes it and every kept unit that writes it is dead. A unit is dead when
/// its `up` reads only such channels, so it multiplies by 0, or when its gate reads only
/// such channels and the constants and the constants leave it at 0 or at most −128,
/// where silu is exactly 0 (`llama::K`). The largest such assignment holds by induction
/// over the layers. Two tokens whose rows agree on every other channel then differ by a
/// constant at every position of the stream, and the one with the lower constant never
/// wins. The run spans the tokens no other beats that way. Both the dead units and the
/// rows' constants read the constant channels, so without `constants_hold` the run is the
/// stored one.
pub fn stream_head(m: &Model, s: &Slice, rows: &HeadRows) -> (usize, usize) {
    let c = &m.cfg;
    let (d, a, inter) = (c.hidden, c.attn(), c.intermediate);
    let nc = CONSTANTS.len();
    if !constants_hold(m) {
        return (rows.first, rows.count);
    }
    let held = tok::rec(c).min(c.vocab);
    let mut zero: Vec<bool> = (0..d).map(|ch| (0..held).all(|t| m.embed[t * d + ch] == 0.0)).collect();
    // Per kept unit: its layer, its index, and the channels it writes.
    let mut units: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (li, (us, hs)) in s.units.iter().zip(&s.heads).enumerate() {
        let l = &m.layers[li];
        for ch in 0..d {
            if hs.iter().any(|&h| l.o[ch * a + h * HEAD_DIM..][..HEAD_DIM].iter().any(|&w| w != 0.0)) {
                zero[ch] = false;
            }
        }
        units.extend(us.iter().map(|&u| (li, u, (0..d).filter(|&ch| l.down[ch * inter + u] != 0.0).collect())));
    }
    loop {
        let dead = |li: usize, u: usize| {
            let (gate, up) = (&m.layers[li].gate[u * d..][..d], &m.layers[li].up[u * d..][..d]);
            let bias: f32 = gate[..nc].iter().zip(CONSTANTS).map(|(w, k)| w * k).sum();
            up.iter().enumerate().all(|(ch, &w)| w == 0.0 || zero[ch]) || gate.iter().enumerate().skip(nc).all(|(ch, &w)| w == 0.0 || zero[ch]) && (bias == 0.0 || bias <= -128.0)
        };
        let live: Vec<usize> = units.iter().filter(|(li, u, _)| !dead(*li, *u)).flat_map(|(_, _, chs)| chs.iter().copied()).filter(|&ch| zero[ch]).collect();
        if live.is_empty() {
            break;
        }
        live.into_iter().for_each(|ch| zero[ch] = false);
    }
    let row = |t: usize| &m.lm_head[t * d..][..d];
    let run = rows.first..rows.first + rows.count;
    let reads: Vec<usize> = (nc..d).filter(|&ch| !zero[ch] && run.clone().any(|t| row(t)[ch] != 0.0)).collect();
    let constant = |t: usize| row(t)[..nc].iter().zip(CONSTANTS).map(|(w, k)| w * k).sum::<f32>();
    let mut best: HashMap<Vec<u32>, f32> = HashMap::new();
    let key = |t: usize| reads.iter().map(|&ch| row(t)[ch].to_bits()).collect::<Vec<u32>>();
    for t in run.clone() {
        let b = best.entry(key(t)).or_insert(f32::NEG_INFINITY);
        *b = b.max(constant(t));
    }
    let wins = |t: &usize| constant(*t) == best[&key(*t)];
    let (first, last) = (run.clone().find(wins).expect("a token wins"), run.rev().find(wins).expect("a token wins"));
    (first, last + 1 - first)
}

/// The prefix of channels a runtime may cut stream `s`'s matmuls to
/// (`gbcformer.stream.<part>.width`): `s.hidden` when every row a runtime's slice runs
/// (the stream's layers, in each the kept heads' query rows and output columns, every key
/// and value row, the kept units' gate and up rows and down columns, and the head's stored
/// rows) is zero past it, else every channel. Such a slice computes the same values on the
/// prefix at any width, and the norm divides by 2^7 at the stream's width as at the
/// checkpoint's (`Model::slice` widens `hidden` until its bound holds, and
/// `Model::stream` is that network).
pub fn stream_width(m: &Model, s: &Slice, head: &HeadRows) -> usize {
    let c = &m.cfg;
    let (d, a, inter, h) = (c.hidden, c.attn(), c.intermediate, s.hidden);
    let reads = |rows: &[f32]| rows.chunks(d).all(|r| r[h..].iter().all(|&w| w == 0.0));
    let layers = s.units.iter().zip(&s.heads).zip(&m.layers).all(|((us, hs), l)| {
        let heads = hs.iter().all(|&fh| reads(&l.q[fh * HEAD_DIM * d..][..HEAD_DIM * d]) && (h..d).all(|ch| l.o[ch * a + fh * HEAD_DIM..][..HEAD_DIM].iter().all(|&w| w == 0.0)));
        let units = us.iter().all(|&u| reads(&l.gate[u * d..][..d]) && reads(&l.up[u * d..][..d]) && (h..d).all(|ch| l.down[ch * inter + u] == 0.0));
        heads && units && reads(&l.k) && reads(&l.v)
    });
    if layers && reads(&m.lm_head[head.first * d..][..head.count * d]) { h } else { d }
}

/// The metadata key naming, per layer, the run of channels whose columns the file's
/// `ffn_up` stores, as `[first, count]` pairs (`up_cols`). Without it the file stores
/// every column.
pub const UP_COLS_KEY: &str = "gbcformer.ffn_up.cols";

/// Every run of columns a layer-shaped file stores starts and ends on a multiple of
/// this, the weights Metal's few-row matmul kernels take a step and the quantum of every
/// stream's width, so a narrowed matmul meets the same kernels the full one does.
pub const COLUMN_QUANTUM: usize = 64;

/// The shortest run of whole quanta that holds every channel in `chs`, as
/// `(first, count)`, within `d` channels; the first quantum when `chs` is empty.
fn quanta(chs: impl Iterator<Item = usize>, d: usize) -> (usize, usize) {
    let (lo, hi) = chs.fold((usize::MAX, 0), |(lo, hi), ch| (lo.min(ch), hi.max(ch + 1)));
    let first = if hi == 0 { 0 } else { lo / COLUMN_QUANTUM * COLUMN_QUANTUM };
    (first, hi.max(1).next_multiple_of(COLUMN_QUANTUM).min(d) - first)
}

/// Per layer, the run of channels the kept units' `up` rows read (`quanta`): every
/// other column of `ffn_up` is zero, and a layer-shaped file leaves it out.
pub fn up_cols(m: &Model, keep: &[Keep]) -> Vec<(usize, usize)> {
    let d = m.cfg.hidden;
    m.layers.iter().zip(keep).map(|(l, k)| quanta(k.units.iter().flat_map(|&u| (0..d).filter(move |&ch| l.up[u * d + ch] != 0.0)), d)).collect()
}

/// The metadata keys naming the prefix of channels whose columns the file's
/// `token_embd` and `output` store (`col_prefixes`). Without one the file stores every
/// column of that tensor.
pub const EMBED_COLS_KEY: &str = "gbcformer.token_embd.cols";
pub const HEAD_COLS_KEY: &str = "gbcformer.output.cols";

/// The prefixes of channels, in whole quanta, that hold every channel an embedding sets
/// and every channel a row of the head's run `head` reads: every later column of
/// `token_embd` and of `output` is zero, and a layer-shaped file leaves it out.
pub fn col_prefixes(m: &Model, head: &HeadRows) -> (usize, usize) {
    let d = m.cfg.hidden;
    let prefix = |w: &[f32]| {
        let (first, count) = quanta((0..d).filter(|&ch| w.chunks(d).any(|r| r[ch] != 0.0)), d);
        first + count
    };
    (prefix(&m.embed), prefix(&m.lm_head[head.first * d..][..head.count * d]))
}

/// Rows of `width` values as rows of `d`, zero past `width`.
fn pad_cols(x: &[f32], width: usize, d: usize) -> Vec<f32> {
    let mut full = vec![0f32; x.len() / width * d];
    for (row, run) in full.chunks_mut(d).zip(x.chunks(width)) {
        row[..width].copy_from_slice(run);
    }
    full
}

/// The parameters a layer-shaped GGUF stores (`export --layer-shapes --align N`): the
/// embedding and the LM head's run of rows (`HeadRows`) at their prefixes of columns
/// (`col_prefixes`), the norms, and each layer at its own shape, `keep`'s heads and FFN
/// width, with the K and V rows every layer keeps and `up` at its run of columns
/// (`up_cols`).
pub fn stored_params(m: &Model, keep: &[Keep]) -> usize {
    let c = &m.cfg;
    let (d, kv) = (c.hidden, c.kv_dim());
    let layers = keep.iter().zip(up_cols(m, keep)).map(|(k, (_, up))| 2 * d + 2 * k.heads.len() * HEAD_DIM * d + 2 * kv * d + k.width * (2 * d + up)).sum::<usize>();
    let head = HeadRows::of(m);
    let (embed_cols, head_cols) = col_prefixes(m, &head);
    c.vocab * embed_cols + head.count * head_cols + d + layers
}

/// HF's rotate-half rows to llama.cpp's interleaved pairs, per head: row `j·32 + i`
/// moves to `2i + j`, exactly `permute(w, n_head, n_head)` in `convert_hf_to_gguf.py`.
/// `back` undoes it.
pub fn interleave(w: &[f32], heads: usize, d: usize, back: bool) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut out = vec![0f32; w.len()];
    for h in 0..heads {
        for j in 0..2 {
            for i in 0..half {
                let (hf, gg) = (h * HEAD_DIM + j * half + i, h * HEAD_DIM + 2 * i + j);
                let (from, to) = if back { (gg, hf) } else { (hf, gg) };
                out[to * d..][..d].copy_from_slice(&w[from * d..][..d]);
            }
        }
    }
    out
}

/// The f32 an f16's bits hold.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = ((h >> 10) & 0x1F, (h & 0x3FF) as f32);
    match e {
        0 => sign * m * (1.0 / 16_777_216.0),
        31 => {
            if m == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + m / 1024.0) * f32::from_bits(((e as i32 - 15 + 127) as u32) << 23),
    }
}

/// The f16 bits of `x` rounded to nearest, ties to even (`llama::f16`).
pub fn f32_to_f16(x: f32) -> u16 {
    let r = crate::llama::f16(x);
    let sign = if r.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = r.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | 0x7C00;
    }
    if a < 6.103_515_6e-5 {
        return sign | (a * 16_777_216.0) as u16;
    }
    let b = a.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127;
    sign | (((e + 15) as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}

/// The f16 bits of `x`, which must be an f16 value exactly (`Model::check_f16`).
pub fn f16_bits(x: f32) -> u16 {
    assert!(x.is_finite() && crate::llama::f16(x) == x, "{x} is not an f16 value");
    f32_to_f16(x)
}

/// One tensor's place in the file: dimensions innermost first, GGML type, byte offset
/// from the data section.
struct Info {
    dims: Vec<usize>,
    kind: u32,
    offset: u64,
}

/// A GGUF v3 file's header: the metadata this loader reads and every tensor's place.
pub struct Gguf {
    file: BufReader<File>,
    data: u64,
    u32s: HashMap<String, Vec<u32>>,
    f32s: HashMap<String, f32>,
    strs: HashMap<String, String>,
    tensors: HashMap<String, Info>,
}

fn rd<const N: usize>(r: &mut impl Read) -> std::io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn rd_u32(r: &mut impl Read) -> std::io::Result<u32> {
    rd::<4>(r).map(u32::from_le_bytes)
}

fn rd_u64(r: &mut impl Read) -> std::io::Result<u64> {
    rd::<8>(r).map(u64::from_le_bytes)
}

fn rd_str(r: &mut impl Read) -> std::io::Result<String> {
    let n = rd_u64(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Reads one metadata value of GGUF type `t`; returns it when it is a u32 or an array
/// of u32, the only kinds the loader needs.
fn rd_val(r: &mut impl Read, t: u32) -> std::io::Result<Option<Vec<u32>>> {
    let width = |t: u32| match t {
        0 | 1 | 7 => Some(1u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    match t {
        4 => Ok(Some(vec![rd_u32(r)?])),
        8 => rd_str(r).map(|_| None),
        9 => {
            let (et, n) = (rd_u32(r)?, rd_u64(r)?);
            if et == 4 {
                return (0..n).map(|_| rd_u32(r)).collect::<Result<Vec<_>, _>>().map(Some);
            }
            for _ in 0..n {
                rd_val(r, et)?;
            }
            Ok(None)
        }
        _ => {
            let w = width(t).ok_or_else(|| std::io::Error::other(format!("GGUF value type {t}")))?;
            std::io::copy(&mut r.take(w), &mut std::io::sink()).map(|_| None)
        }
    }
}

impl Gguf {
    pub fn open(path: &Path) -> std::io::Result<Gguf> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        if &rd::<4>(&mut r)? != b"GGUF" || rd_u32(&mut r)? != 3 {
            return Err(std::io::Error::other(format!("{}: not a GGUF v3 file", path.display())));
        }
        let (n_tensors, n_kv) = (rd_u64(&mut r)?, rd_u64(&mut r)?);
        let (mut u32s, mut f32s, mut strs) = (HashMap::new(), HashMap::new(), HashMap::new());
        let mut align = 32u64;
        for _ in 0..n_kv {
            let key = rd_str(&mut r)?;
            let t = rd_u32(&mut r)?;
            if t == 8 {
                strs.insert(key, rd_str(&mut r)?);
                continue;
            }
            if t == 6 {
                f32s.insert(key, f32::from_le_bytes(rd::<4>(&mut r)?));
                continue;
            }
            if let Some(v) = rd_val(&mut r, t)? {
                if key == "general.alignment" {
                    align = v[0] as u64;
                }
                u32s.insert(key, v);
            }
        }
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = rd_str(&mut r)?;
            let n = rd_u32(&mut r)? as usize;
            let dims = (0..n).map(|_| rd_u64(&mut r).map(|x| x as usize)).collect::<Result<Vec<_>, _>>()?;
            let (kind, offset) = (rd_u32(&mut r)?, rd_u64(&mut r)?);
            tensors.insert(name, Info { dims, kind, offset });
        }
        let at = r.stream_position()?;
        Ok(Gguf { file: r, data: at.div_ceil(align) * align, u32s, f32s, strs, tensors })
    }

    /// Layer `l`'s `QK_SHIFT_KEY` as the factors that undo it, the query's and then the
    /// key's.
    pub fn qk_factors(&self, l: usize) -> (f32, f32) {
        let s = self.u32s.get(QK_SHIFT_KEY).map_or(0, |v| if v.len() == 1 { v[0] } else { v[l] });
        ((1u32 << s) as f32, 1.0 / (1u32 << s) as f32)
    }

    /// The rows of the LM head the file stores (`HEAD_ROWS_KEY`, `HEAD_FILL_KEY`).
    pub fn head_rows(&self, vocab: usize) -> Result<HeadRows, String> {
        let Some(v) = self.u32s.get(HEAD_ROWS_KEY) else { return Ok(HeadRows::whole(vocab)) };
        let (&[first, count], Some(&fill)) = (&v[..], self.f32s.get(HEAD_FILL_KEY)) else { return Err(format!("{HEAD_ROWS_KEY} is not [first, count] with {HEAD_FILL_KEY}")) };
        let rows = HeadRows { first: first as usize, count: count as usize, fill };
        if rows.count == 0 || rows.first + rows.count > vocab {
            return Err(format!("{HEAD_ROWS_KEY}: rows {}..{} of {vocab}", rows.first, rows.first + rows.count));
        }
        Ok(rows)
    }

    /// The prefix of columns the file stores under `key` (`EMBED_COLS_KEY`,
    /// `HEAD_COLS_KEY`), every column of `d` without the key.
    pub fn col_prefix(&self, key: &str, d: usize) -> Result<usize, String> {
        match self.u32s.get(key).map(|v| &v[..]) {
            None => Ok(d),
            Some(&[n]) if n > 0 && n as usize <= d => Ok(n as usize),
            Some(v) => Err(format!("{key}: {v:?} is not a prefix of the {d} channels")),
        }
    }

    /// Each layer's run of `ffn_up` columns (`UP_COLS_KEY`), every column of `d` without the key.
    pub fn up_cols(&self, layers: usize, d: usize) -> Result<Vec<(usize, usize)>, String> {
        let Some(v) = self.u32s.get(UP_COLS_KEY) else { return Ok(vec![(0, d); layers]) };
        let runs: Vec<(usize, usize)> = v.chunks(2).map(|r| (r[0] as usize, r[r.len() - 1] as usize)).collect();
        if v.len() != 2 * layers || runs.iter().any(|&(first, count)| count == 0 || first + count > d) {
            return Err(format!("{UP_COLS_KEY}: not a run of the {d} channels for each of {layers} layers"));
        }
        Ok(runs)
    }

    /// A per-layer metadata array, or a scalar repeated for every layer.
    pub fn per_layer(&self, key: &str, layers: usize) -> Option<Vec<usize>> {
        let v = self.u32s.get(key)?;
        Some(if v.len() == 1 { vec![v[0] as usize; layers] } else { v.iter().map(|&x| x as usize).collect() })
    }

    /// Tensor `name` as f32, row-major with its dimensions outermost first.
    pub fn tensor(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let info = self.tensors.get(name).ok_or_else(|| format!("no tensor {name}"))?;
        let n: usize = info.dims.iter().product();
        let shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        let width = match info.kind {
            0 => 4,
            1 => 2,
            k => return Err(format!("{name}: GGML type {k}, not f32 or f16")),
        };
        let mut bytes = vec![0u8; n * width];
        let at = self.data + info.offset;
        self.file.seek(SeekFrom::Start(at)).and_then(|_| self.file.read_exact(&mut bytes)).map_err(|e| format!("{name}: {e}"))?;
        let data = if width == 2 { bytes.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect() } else { bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
        Ok((shape, data))
    }
}

/// What loading found: the tensors read, the values in the file and in the compiled
/// layout, and how many of the compiled values differed.
#[derive(Debug, Default)]
pub struct Loaded {
    pub tensors: usize,
    pub values: usize,
    pub differ: usize,
    pub file_values: usize,
    /// Bytes read from the file (`load_slice`).
    pub bytes: u64,
}

/// Replaces every weight of `m` (as compiled) with the file's, expanded to the compiled
/// layout, after comparing them bit for bit. Returns `Err` when a tensor's shape does
/// not fit or a value differs, and `m` then holds a mix and must not be used.
pub fn load(m: &mut Model, path: &Path) -> Result<Loaded, String> {
    let shapes = keep(m, 1);
    let out = load_into(m, path, &shapes)?;
    if out.differ > 0 {
        return Err(format!("{} of {} weights differ from the compiled checkpoint's", out.differ, out.values));
    }
    Ok(out)
}

/// Every weight of `m` from the file, each tensor expanded to the layout `shapes` gives
/// (the compiled checkpoint's kept heads and units); `Loaded::differ` counts the values
/// that differ from what `m` held.
fn load_into(m: &mut Model, path: &Path, shapes: &[Keep]) -> Result<Loaded, String> {
    let mut g = Gguf::open(path).map_err(|e| e.to_string())?;
    let c = m.cfg.clone();
    let (d, i, v, kvd) = (c.hidden, c.intermediate, c.vocab, c.kv_dim());
    let heads = g.per_layer("llama.attention.head_count", c.layers).ok_or("no head_count")?;
    let widths = g.per_layer("llama.feed_forward_length", c.layers).ok_or("no feed_forward_length")?;
    let ups = g.up_cols(c.layers, d)?;
    // A file that gives one head count and one FFN width for every layer, the compiled
    // ones, is the uniform export (the one unpatched llama.cpp loads): every head and
    // unit sits at its compiled index, the dead ones as zeros.
    let one = |key: &str, all: usize| g.u32s.get(key).is_some_and(|v| v.len() == 1 && v[0] as usize == all);
    let whole: Vec<Keep>;
    let shapes = if one("llama.attention.head_count", c.heads) && one("llama.feed_forward_length", i) {
        whole = (0..c.layers).map(|_| Keep { heads: (0..c.heads).collect(), units: (0..i).collect(), width: i }).collect();
        &whole[..]
    } else {
        shapes
    };
    let mut out = Loaded::default();
    let put = |g: &mut Gguf, name: &str, want: &[usize], expand: &dyn Fn(&[f32]) -> Result<Vec<f32>, String>, slot: &mut Vec<f32>, out: &mut Loaded| -> Result<(), String> {
        let (shape, data) = g.tensor(name)?;
        if shape != want {
            return Err(format!("{name}: shape {shape:?}, the compiled layout needs {want:?}"));
        }
        out.file_values += data.len();
        let full = expand(&data)?;
        if full.len() != slot.len() {
            return Err(format!("{name}: expands to {} values, the compiled tensor has {}", full.len(), slot.len()));
        }
        out.differ += full.iter().zip(slot.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        out.values += full.len();
        out.tensors += 1;
        *slot = full;
        Ok(())
    };
    let same = |x: &[f32]| Ok(x.to_vec());
    let gamma = |name: String, g: &mut Gguf| -> Result<(), String> {
        let (_, x) = g.tensor(&name)?;
        if x.len() == d && x.iter().all(|&y| y == GAMMA) { Ok(()) } else { Err(format!("{name}: not GAMMA on every channel")) }
    };
    let (embed_cols, head_cols) = (g.col_prefix(EMBED_COLS_KEY, d)?, g.col_prefix(HEAD_COLS_KEY, d)?);
    put(&mut g, "token_embd.weight", &[v, embed_cols], &|x| Ok(pad_cols(x, embed_cols, d)), &mut m.embed, &mut out)?;
    for (n, l) in m.layers.iter_mut().enumerate() {
        let (k, h, w) = (&shapes[n], heads[n], widths[n]);
        let (fq, fk) = g.qk_factors(n);
        if h != k.heads.len() {
            return Err(format!("layer {n}: the file keeps {h} heads, the compiled layer has {} that write", k.heads.len()));
        }
        if w < k.units.len() {
            return Err(format!("layer {n}: the file's FFN width {w} is below the compiled layer's {} live units", k.units.len()));
        }
        // Rows of `block` the file keeps, placed back at their compiled indices; the
        // file's padding rows must be zero.
        let place_rows = |x: &[f32], width: usize, block: usize, at: &[usize], total: usize| -> Result<Vec<f32>, String> {
            let mut full = vec![0f32; total * block * width];
            for (r, &to) in at.iter().enumerate() {
                full[to * block * width..][..block * width].copy_from_slice(&x[r * block * width..][..block * width]);
            }
            if x[at.len() * block * width..].iter().any(|&y| y != 0.0) {
                return Err("a padding row is not zero".into());
            }
            Ok(full)
        };
        let place_cols = |x: &[f32], rows: usize, block: usize, at: &[usize], kept: usize, total: usize| -> Result<Vec<f32>, String> {
            let mut full = vec![0f32; rows * total * block];
            for r in 0..rows {
                let row = &x[r * kept * block..][..kept * block];
                for (j, &to) in at.iter().enumerate() {
                    full[r * total * block + to * block..][..block].copy_from_slice(&row[j * block..][..block]);
                }
                if row[at.len() * block..].iter().any(|&y| y != 0.0) {
                    return Err("a padding column is not zero".into());
                }
            }
            Ok(full)
        };
        let b = format!("blk.{n}");
        gamma(format!("{b}.attn_norm.weight"), &mut g)?;
        gamma(format!("{b}.ffn_norm.weight"), &mut g)?;
        put(&mut g, &format!("{b}.attn_q.weight"), &[h * HEAD_DIM, d], &|x| place_rows(&interleave(x, h, d, true).iter().map(|w| w * fq).collect::<Vec<f32>>(), d, HEAD_DIM, &k.heads, c.heads), &mut l.q, &mut out)?;
        put(&mut g, &format!("{b}.attn_k.weight"), &[kvd, d], &|x| Ok(interleave(x, c.kv_heads, d, true).iter().map(|w| w * fk).collect()), &mut l.k, &mut out)?;
        put(&mut g, &format!("{b}.attn_v.weight"), &[kvd, d], &same, &mut l.v, &mut out)?;
        put(&mut g, &format!("{b}.attn_output.weight"), &[d, h * HEAD_DIM], &|x| place_cols(x, d, HEAD_DIM, &k.heads, h, c.heads), &mut l.o, &mut out)?;
        put(&mut g, &format!("{b}.ffn_gate.weight"), &[w, d], &|x| place_rows(x, d, 1, &k.units, i), &mut l.gate, &mut out)?;
        let (up_first, up_count) = ups[n];
        // The file's run of columns back at its channels, in rows of `d`.
        let widen = |x: Vec<f32>| -> Vec<f32> {
            let mut full = vec![0f32; x.len() / up_count * d];
            for (row, run) in full.chunks_mut(d).zip(x.chunks(up_count)) {
                row[up_first..][..up_count].copy_from_slice(run);
            }
            full
        };
        put(&mut g, &format!("{b}.ffn_up.weight"), &[w, up_count], &|x| place_rows(x, up_count, 1, &k.units, i).map(widen), &mut l.up, &mut out)?;
        put(&mut g, &format!("{b}.ffn_down.weight"), &[d, w], &|x| place_cols(x, d, 1, &k.units, w, i), &mut l.down, &mut out)?;
    }
    gamma("output_norm.weight".into(), &mut g)?;
    let head = g.head_rows(v)?;
    put(&mut g, "output.weight", &[head.count, head_cols], &|x| Ok(head.expand(&pad_cols(x, head_cols, d), d, v)), &mut m.lm_head, &mut out)?;
    Ok(out)
}

impl Gguf {
    /// The values of tensor `name` at each run `(first, len)` of its row-major
    /// elements, read straight from the file, so only those bytes are read; `read`
    /// counts them.
    fn runs(&mut self, name: &str, runs: &[(usize, usize)], read: &mut u64) -> Result<Vec<f32>, String> {
        let info = self.tensors.get(name).ok_or_else(|| format!("no tensor {name}"))?;
        let width = match info.kind {
            0 => 4,
            1 => 2,
            k => return Err(format!("{name}: GGML type {k}, not f32 or f16")),
        };
        let base = self.data + info.offset;
        let f = self.file.get_mut();
        let mut out = Vec::with_capacity(runs.iter().map(|r| r.1).sum());
        let mut bytes = Vec::new();
        for &(first, len) in runs {
            bytes.resize(len * width, 0);
            f.seek(SeekFrom::Start(base + (first * width) as u64)).and_then(|_| f.read_exact(&mut bytes)).map_err(|e| format!("{name}: {e}"))?;
            *read += bytes.len() as u64;
            if width == 2 {
                out.extend(bytes.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))));
            } else {
                out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            }
        }
        Ok(out)
    }

    /// Tensor `name`'s dimensions, outermost first.
    fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        Ok(self.tensors.get(name).ok_or_else(|| format!("no tensor {name}"))?.dims.iter().rev().copied().collect())
    }
}

/// Replaces every weight of an overlay stream's network `m` (`Model::stream`, `s` its
/// slice) with the file's, reading only what the slice keeps: its prefix of columns in
/// every row it keeps, its layers, its heads and units. The checkpoint's own file
/// layout is `keep` (the compiled checkpoint's `gguf::keep`, which a `--layer-shapes`
/// file holds) at `d` channels. The loader applies the stream's folds
/// (`Model::stream`) to what it reads, then compares every value bit for bit with the
/// compiler's, as `load` does. `Loaded::bytes` counts the bytes read.
pub fn load_slice(m: &mut Model, s: &crate::llama::Slice, keep: &[Keep], d: usize, path: &Path) -> Result<Loaded, String> {
    let mut g = Gguf::open(path).map_err(|e| e.to_string())?;
    let c = m.cfg.clone();
    let (h, a, kvd, inter, v) = (c.hidden, c.attn(), c.kv_dim(), c.intermediate, c.vocab);
    let n_layers = keep.len();
    let file_heads = g.per_layer("llama.attention.head_count", n_layers).ok_or("no head_count")?;
    let widths = g.per_layer("llama.feed_forward_length", n_layers).ok_or("no feed_forward_length")?;
    let ups = g.up_cols(n_layers, d)?;
    let mut out = Loaded::default();
    let mut read = 0u64;
    let put = |full: Vec<f32>, slot: &mut Vec<f32>, out: &mut Loaded| {
        out.file_values += full.len();
        out.differ += full.iter().zip(slot.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        out.values += full.len();
        out.tensors += 1;
        *slot = full;
    };
    let prefix_rows = |rows: usize| (0..rows).map(|r| (r * d, h)).collect::<Vec<_>>();
    // The stream's prefix of each row of a tensor the file stores at `cols` columns.
    let prefix_of = |g: &mut Gguf, name: &str, rows: usize, cols: usize, read: &mut u64| -> Result<Vec<f32>, String> {
        if g.shape(name)? != [rows, cols] {
            return Err(format!("{name}: shape {:?}, the file's layout gives {rows} × {cols}", g.shape(name)?));
        }
        let n = cols.min(h);
        Ok(pad_cols(&g.runs(name, &(0..rows).map(|r| (r * cols, n)).collect::<Vec<_>>(), read)?, n, h))
    };
    let (embed_cols, head_cols) = (g.col_prefix(EMBED_COLS_KEY, d)?, g.col_prefix(HEAD_COLS_KEY, d)?);
    let mut x = prefix_of(&mut g, "token_embd.weight", v, embed_cols, &mut read)?;
    for row in x.chunks_mut(h) {
        for &ch in &s.cleared {
            row[ch] = 0.0;
        }
    }
    put(x, &mut m.embed, &mut out);
    for li in 0..c.layers {
        let (k, hf, w) = (&keep[li], file_heads[li], widths[li]);
        let (fq, fk) = g.qk_factors(li);
        let b = format!("blk.{li}");
        for norm in ["attn_norm", "ffn_norm"] {
            let x = g.runs(&format!("{b}.{norm}.weight"), &[(0, h)], &mut read)?;
            if x.iter().any(|&y| y != GAMMA) {
                return Err(format!("{b}.{norm}: not GAMMA on every channel"));
            }
        }
        let l = &mut m.layers[li];
        let mut q = vec![0f32; a * h];
        let mut o = vec![0f32; h * a];
        for (nh, (&fh, &own)) in s.heads[li].iter().zip(&s.own_heads[li]).enumerate() {
            let Ok(pos) = k.heads.binary_search(&fh) else { continue };
            let rows: Vec<(usize, usize)> = (0..HEAD_DIM).map(|t| ((pos * HEAD_DIM + t) * d, h)).collect();
            let mut x = interleave(&g.runs(&format!("{b}.attn_q.weight"), &rows, &mut read)?, 1, h, true);
            x.iter_mut().for_each(|w| *w *= fq);
            if own {
                for &dim in &s.null_dims {
                    x[dim * h..][..h].fill(0.0);
                }
            }
            q[nh * HEAD_DIM * h..][..HEAD_DIM * h].copy_from_slice(&x);
            let cols: Vec<(usize, usize)> = (0..h).map(|ch| (ch * hf * HEAD_DIM + pos * HEAD_DIM, HEAD_DIM)).collect();
            let y = g.runs(&format!("{b}.attn_output.weight"), &cols, &mut read)?;
            for ch in 0..h {
                o[ch * a + nh * HEAD_DIM..][..HEAD_DIM].copy_from_slice(&y[ch * HEAD_DIM..][..HEAD_DIM]);
            }
        }
        put(q, &mut l.q, &mut out);
        put(o, &mut l.o, &mut out);
        let mut kx = interleave(&g.runs(&format!("{b}.attn_k.weight"), &prefix_rows(kvd), &mut read)?, c.kv_heads, h, true);
        kx.iter_mut().for_each(|w| *w *= fk);
        if li > 0 {
            for kv in 0..c.kv_heads {
                for &dim in &s.null_dims {
                    kx[(kv * HEAD_DIM + dim) * h..][..h].fill(0.0);
                }
            }
        }
        // As `Model::stream` does, clear the null channel that the key dims read, and the
        // mode channels, which no key reads (`s.cleared`). The stream holds no null token.
        for row in kx.chunks_mut(h) {
            for &ch in &s.cleared {
                row[ch] = 0.0;
            }
        }
        put(kx, &mut l.k, &mut out);
        let vx = g.runs(&format!("{b}.attn_v.weight"), &prefix_rows(kvd), &mut read)?;
        put(vx, &mut l.v, &mut out);
        // The part's units sit at ascending file positions; the down projection reads
        // the span between the first and the last in every kept row.
        let at: Vec<Option<usize>> = s.units[li].iter().map(|u| k.units.binary_search(u).ok()).collect();
        let (mut gate, mut up, mut down) = (vec![0f32; inter * h], vec![0f32; inter * h], vec![0f32; h * inter]);
        let rows: Vec<(usize, usize)> = at.iter().flatten().map(|&p| (p * d, h)).collect();
        // The file's run of up columns, as far as the prefix reaches.
        let (up_first, up_count) = ups[li];
        let up_n = (up_first + up_count).min(h).saturating_sub(up_first);
        let up_rows: Vec<(usize, usize)> = at.iter().flatten().map(|&p| (p * up_count, up_n)).collect();
        let (gx, ux) = (g.runs(&format!("{b}.ffn_gate.weight"), &rows, &mut read)?, g.runs(&format!("{b}.ffn_up.weight"), &up_rows, &mut read)?);
        for (r, nu) in at.iter().enumerate().filter_map(|(nu, p)| p.map(|_| nu)).enumerate() {
            gate[nu * h..][..h].copy_from_slice(&gx[r * h..][..h]);
            let row = &mut up[nu * h..][..h];
            row[up_first.min(h)..][..up_n].copy_from_slice(&ux[r * up_n..][..up_n]);
            if row[0] == 0.0 && row[s.mode] != 0.0 && row.iter().enumerate().all(|(ch, &w)| w == 0.0 || ch == s.mode) {
                (row[0], row[s.mode]) = (row[s.mode], 0.0);
            }
        }
        if let (Some(lo), Some(hi)) = (at.iter().flatten().min(), at.iter().flatten().max()) {
            let span = hi - lo + 1;
            let y = g.runs(&format!("{b}.ffn_down.weight"), &(0..h).map(|ch| (ch * w + lo, span)).collect::<Vec<_>>(), &mut read)?;
            for ch in 0..h {
                for (nu, p) in at.iter().enumerate() {
                    if let Some(p) = p {
                        down[ch * inter + nu] = y[ch * span + p - lo];
                    }
                }
            }
        }
        put(gate, &mut l.gate, &mut out);
        put(up, &mut l.up, &mut out);
        put(down, &mut l.down, &mut out);
    }
    if g.runs("output_norm.weight", &[(0, h)], &mut read)?.iter().any(|&y| y != GAMMA) {
        return Err("output_norm: not GAMMA on every channel".into());
    }
    let head = g.head_rows(v)?;
    let x = prefix_of(&mut g, "output.weight", head.count, head_cols, &mut read)?;
    put(head.expand(&x, h, v), &mut m.lm_head, &mut out);
    out.file_values -= (v - head.count) * h;
    out.bytes = read;
    if out.differ > 0 {
        return Err(format!("{} of {} weights differ from the compiled stream's", out.differ, out.values));
    }
    Ok(out)
}

/// One SHA-1 over a checkpoint: configuration, slot names, every weight and the unit
/// names, plus the memory spec for a memory checkpoint (`examples/digest.rs`). Returns
/// the digest and the nonzero weights.
pub fn digest(m: &Model) -> (String, u64) {
    let mut h = sha1_smol::Sha1::new();
    let mut nonzeros = 0u64;
    let c = &m.cfg;
    h.update(format!("{} {} {} {} {} {} {} {}\n", c.hidden, c.intermediate, c.layers, c.heads, c.vocab, c.n_inputs, c.n_outputs, c.n_const).as_bytes());
    h.update(m.slots.join(",").as_bytes());
    let mut floats = |v: &[f32]| {
        nonzeros += v.iter().filter(|f| **f != 0.0).count() as u64;
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        h.update(&bytes);
    };
    floats(&m.embed);
    for l in &m.layers {
        for w in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] {
            floats(w);
        }
    }
    floats(&m.lm_head);
    for names in &m.unit_names {
        h.update(names.join(",").as_bytes());
    }
    if let Some(spec) = &m.memory {
        h.update(format!("{} {spec:?}", c.families).as_bytes());
    }
    (h.digest().to_string(), nonzeros)
}

/// `GBCFORMER_GGUF=<model.gguf>`: the file an overlay stream's network (`Model::stream`,
/// slice `s`) reads its weights from, its slice only (`load_slice`).
pub fn slice_from_env(m: &mut Model, s: &crate::llama::Slice, keep: &[Keep], d: usize) {
    let Some(path) = std::env::var_os("GBCFORMER_GGUF") else { return };
    let t = std::time::Instant::now();
    let size = std::fs::metadata(&path).map_or(0, |f| f.len());
    match load_slice(m, s, keep, d, Path::new(&path)) {
        Ok(l) => eprintln!("weights: {} ({} tensors, {} of its {} bytes read, {} values, 0 differ from the compiler's) in {:.1}s", Path::new(&path).display(), l.tensors, l.bytes, size, l.values, t.elapsed().as_secs_f64()),
        Err(e) => panic!("{}: {e}", Path::new(&path).display()),
    }
}

/// `GBCFORMER_GGUF=<model.gguf>`: the file whose weights a one-model run computes.
pub fn from_env(m: &mut Model) {
    let Some(path) = std::env::var_os("GBCFORMER_GGUF") else { return };
    let t = std::time::Instant::now();
    match load(m, Path::new(&path)) {
        Ok(l) => eprintln!("weights: {} ({} tensors, {} values in the file, {} in the compiled layout, 0 differ from the compiler's) in {:.1}s", Path::new(&path).display(), l.tensors, l.file_values, l.values, t.elapsed().as_secs_f64()),
        Err(e) => panic!("{}: {e}", Path::new(&path).display()),
    }
}

/// The metadata key that holds, as JSON, everything besides the weights the fast engine
/// needs to run the checkpoint from the file alone ([`Meta`]).
pub const META_KEY: &str = "gbcformer.meta";

/// What a runtime needs besides the weights: the configuration and slot names, the
/// residual channel layout, every wire's declared range, level and group (the engine's
/// guards), the memory and overlay specs, each layer's kept heads and units (where the
/// file's rows go), and the streams of an overlay checkpoint. All of it is the
/// compiler's output; none of it is read from a circuit at load.
///
/// It also names the compiled checkpoint's digest (`digest`) and holds the unit names
/// that digest hashes, so `open_model` can rebuild the digest from the file alone and
/// refuse a file whose tensors or placement keys do not give the checkpoint it names.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Meta {
    pub cfg: crate::llama::Config,
    pub slots: Vec<String>,
    pub wire_ranges: Vec<(i64, i64)>,
    pub wire_levels: Vec<u32>,
    pub wire_groups: Vec<u32>,
    pub memory: Option<crate::llama::MemorySpec>,
    pub layout: crate::llama::Layout,
    pub overlay: Option<crate::llama::Overlay>,
    pub keep: Vec<Keep>,
    pub streams: Vec<crate::engine::StreamSpec>,
    #[serde(default)]
    pub unit_names: Vec<Vec<String>>,
    #[serde(default)]
    pub tensors_sha1: String,
}

impl Meta {
    pub fn of(m: &Model, streams: Vec<crate::engine::StreamSpec>) -> Meta {
        Meta {
            cfg: m.cfg.clone(),
            slots: m.slots.clone(),
            wire_ranges: m.wire_ranges.clone(),
            wire_levels: m.wire_levels.clone(),
            wire_groups: m.wire_groups.clone(),
            memory: m.memory.clone(),
            layout: m.layout.clone(),
            overlay: m.overlay.clone(),
            keep: keep(m, 1),
            streams,
            unit_names: m.unit_names.clone(),
            tensors_sha1: digest(m).0,
        }
    }
}

/// A string metadata value of the GGUF file in `path`, such as `crate::wiring::WIRING_KEY`.
pub fn read_string(path: &Path, key: &str) -> Result<String, String> {
    let g = Gguf::open(path).map_err(|e| e.to_string())?;
    g.strs.get(key).cloned().ok_or_else(|| format!("{}: no {key}", path.display()))
}

/// The checkpoint in `path`, read from the file alone: [`Meta`] from its metadata, then
/// every weight from its tensors. Returns the model, its streams and what loading read.
///
/// No compiled checkpoint is at hand to compare the weights with, so the rebuilt model's
/// digest must equal the one the file names (`Meta::tensors_sha1`, and the manifest's
/// when the file has one). A placement key that lies (a run of rows or columns that
/// starts elsewhere, another fill) puts weights on other channels or tokens within the
/// same shapes, and the digest is what refuses it.
pub fn open_model(path: &Path) -> Result<(Model, Vec<crate::engine::StreamSpec>, Loaded), String> {
    let g = Gguf::open(path).map_err(|e| e.to_string())?;
    let json = g.strs.get(META_KEY).ok_or_else(|| format!("{}: no {META_KEY}", path.display()))?;
    let meta: Meta = serde_json::from_str(json).map_err(|e| format!("{META_KEY}: {e}"))?;
    let manifest = g.strs.get(crate::wiring::MANIFEST_KEY).map(|json| serde_json::from_str::<crate::wiring::Manifest>(json).map_err(|e| format!("{}: {e}", crate::wiring::MANIFEST_KEY))).transpose()?;
    drop(g);
    let c = meta.cfg.clone();
    let (d, a, kvd, i, v) = (c.hidden, c.attn(), c.kv_dim(), c.intermediate, c.vocab);
    let layer = || crate::llama::Layer { q: vec![0.0; a * d], k: vec![0.0; kvd * d], v: vec![0.0; kvd * d], o: vec![0.0; d * a], gate: vec![0.0; i * d], up: vec![0.0; i * d], down: vec![0.0; d * i] };
    let mut m = Model {
        embed: vec![0.0; v * d],
        layers: (0..c.layers).map(|_| layer()).collect(),
        lm_head: vec![0.0; v * d],
        slots: meta.slots,
        unit_names: meta.unit_names,
        wire_names: Vec::new(),
        wire_ranges: meta.wire_ranges,
        wire_levels: meta.wire_levels,
        wire_groups: meta.wire_groups,
        memory: meta.memory,
        layout: meta.layout,
        tiles: crate::llama::Tiles::F32,
        overlay: meta.overlay,
        cfg: c,
    };
    let loaded = load_into(&mut m, path, &meta.keep)?;
    let t = std::time::Instant::now();
    let rebuilt = digest(&m).0;
    if meta.tensors_sha1.is_empty() {
        return Err(format!("{}: its metadata names no digest to hold the weights to (a file exported by an earlier version); export it again", path.display()));
    }
    for (named, by) in [(Some(&meta.tensors_sha1), META_KEY), (manifest.as_ref().map(|m| &m.tensors_sha1), crate::wiring::MANIFEST_KEY)] {
        if named.is_some_and(|n| *n != rebuilt) {
            return Err(format!("{}: the weights rebuild to the checkpoint {rebuilt}, and {by} names {}: a tensor or a placement key is not the exported one", path.display(), named.unwrap()));
        }
    }
    eprintln!("weights: {} rebuilds to the checkpoint its metadata names, {rebuilt} (hashed in {:.1}s)", path.display(), t.elapsed().as_secs_f64());
    // The names served the digest. The engine reads them only to label its statistics,
    // which a model from the file alone has never had.
    m.unit_names = vec![Vec::new(); m.cfg.layers];
    Ok((m, meta.streams, loaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_bits_are_the_bits_of_every_finite_f16() {
        for h in (0..=u16::MAX).filter(|h| h >> 10 & 0x1F != 31) {
            assert_eq!(f16_bits(f16_to_f32(h)), h, "{h:#06x}");
        }
    }

    #[test]
    fn a_run_of_columns_spans_whole_quanta() {
        assert_eq!(quanta([903, 908].into_iter(), 2560), (896, 64));
        assert_eq!(quanta([0, 905].into_iter(), 2560), (0, 960));
        assert_eq!(quanta([2559].into_iter(), 2560), (2496, 64));
        assert_eq!(quanta([70].into_iter(), 100), (64, 36));
        assert_eq!(quanta(std::iter::empty(), 2560), (0, 64));
    }

    #[test]
    fn the_head_expands_to_its_run_among_fill_rows() {
        let rows = HeadRows { first: 1, count: 2, fill: -1024.0 };
        assert_eq!(rows.expand(&[1.0, 2.0, 3.0, 4.0], 2, 4), [-1024.0, 0.0, 1.0, 2.0, 3.0, 4.0, -1024.0, 0.0]);
        assert!(rows.decides(4, -1023.0));
        assert!(!rows.decides(4, -1024.0));
        assert!(HeadRows::whole(4).decides(4, f32::NEG_INFINITY));
    }

    #[test]
    fn padded_columns_are_zero() {
        assert_eq!(pad_cols(&[1.0, 2.0, 3.0, 4.0], 2, 3), [1.0, 2.0, 0.0, 3.0, 4.0, 0.0]);
    }

    /// A checkpoint of zero weights whose embeddings hold the constants.
    fn blank(d: usize, vocab: usize, layers: usize, inter: usize) -> Model {
        let cfg = crate::llama::Config { hidden: d, intermediate: inter, layers, heads: 1, kv_heads: 1, vocab, n_inputs: 0, n_outputs: 0, n_const: CONSTANTS.len(), families: 0 };
        let (a, kvd) = (cfg.attn(), cfg.kv_dim());
        let layer = || crate::llama::Layer { q: vec![0.0; a * d], k: vec![0.0; kvd * d], v: vec![0.0; kvd * d], o: vec![0.0; d * a], gate: vec![0.0; inter * d], up: vec![0.0; inter * d], down: vec![0.0; d * inter] };
        let mut embed = vec![0.0; vocab * d];
        for row in embed.chunks_mut(d) {
            row.iter_mut().zip(CONSTANTS).for_each(|(x, k)| *x = k * UNIT);
        }
        Model {
            embed,
            layers: (0..layers).map(|_| layer()).collect(),
            lm_head: vec![0.0; vocab * d],
            slots: Vec::new(),
            unit_names: vec![Vec::new(); layers],
            wire_names: Vec::new(),
            wire_ranges: Vec::new(),
            wire_levels: Vec::new(),
            wire_groups: Vec::new(),
            memory: None,
            layout: crate::llama::Layout { n_const: CONSTANTS.len(), seq: 0, wires: 0, wire_ch: Vec::new(), occupants: Vec::new(), hots: false, memory: false, positions: (0, 0, 0), offsets: Vec::new(), ids: Vec::new(), own_outputs: None },
            tiles: crate::llama::Tiles::F32,
            overlay: None,
            cfg,
        }
    }

    /// A slice that keeps every unit of every layer and no head.
    fn slice(m: &Model, hidden: usize) -> Slice {
        let n = m.cfg.layers;
        Slice { hidden, units: vec![(0..m.cfg.intermediate).collect(); n], heads: vec![Vec::new(); n], own_heads: vec![Vec::new(); n], null_dims: Vec::new(), mode: 0, cleared: Vec::new() }
    }

    #[test]
    fn the_head_run_is_the_hull_of_the_rows_that_are_not_the_fill_row() {
        let (d, v) = (4, 8);
        let mut m = blank(d, v, 1, 1);
        m.lm_head.chunks_mut(d).for_each(|r| r[0] = -1024.0);
        m.lm_head[2 * d + 3] = 1.0;
        m.lm_head[5 * d] = 7.0;
        assert_eq!(HeadRows::of(&m), HeadRows { first: 2, count: 4, fill: -1024.0 });
        // A token without the constant, a writer of a constant channel, and a fill row that
        // reads a channel besides the constant each leave no row that scores a constant.
        let mut no_constant = blank(d, v, 1, 1);
        no_constant.lm_head = m.lm_head.clone();
        no_constant.embed[3 * d] = 0.0;
        assert_eq!(HeadRows::of(&no_constant), HeadRows::whole(v));
        let mut written = blank(d, v, 1, 1);
        written.lm_head = m.lm_head.clone();
        written.layers[0].down[0] = 1.0;
        assert_eq!(HeadRows::of(&written), HeadRows::whole(v));
        let mut wide_fill = blank(d, v, 1, 1);
        wide_fill.lm_head = m.lm_head.clone();
        wide_fill.lm_head.chunks_mut(d).for_each(|r| r[2] = 1.0);
        wide_fill.lm_head[2 * d + 2] = 0.0;
        assert_eq!(HeadRows::of(&wide_fill), HeadRows::whole(v));
    }

    #[test]
    fn the_up_run_holds_every_channel_a_kept_unit_reads() {
        let d = 256;
        let mut m = blank(d, 4, 2, 3);
        m.layers[0].up[70] = 1.0;
        m.layers[0].up[2 * d + 130] = 1.0;
        // Unit 1 is not kept, so what it reads does not widen the run.
        m.layers[0].up[d + 250] = 1.0;
        let keep = vec![Keep { heads: vec![0], units: vec![0, 2], width: 64 }, Keep { heads: vec![0], units: vec![0], width: 64 }];
        assert_eq!(up_cols(&m, &keep), [(64, 128), (0, 64)]);
    }

    #[test]
    fn the_column_prefixes_hold_every_channel_the_embedding_sets_and_the_run_reads() {
        let (d, v) = (256, 6);
        let mut m = blank(d, v, 1, 1);
        m.embed[3 * d + 70] = UNIT;
        m.lm_head[2 * d + 5] = 1.0;
        // A row outside the stored run does not widen the head's prefix.
        m.lm_head[5 * d + 200] = 1.0;
        assert_eq!(col_prefixes(&m, &HeadRows { first: 2, count: 2, fill: 0.0 }), (128, 64));
        assert_eq!(col_prefixes(&m, &HeadRows::whole(v)), (128, 256));
    }

    #[test]
    fn a_stream_runs_on_its_prefix_only_when_every_row_it_keeps_is_zero_past_it() {
        let (d, v) = (128, 4);
        let m = blank(d, v, 1, 1);
        let head = HeadRows::whole(v);
        assert_eq!(stream_width(&m, &slice(&m, 64), &head), 64);
        let mut key = blank(d, v, 1, 1);
        key.layers[0].k[100] = 1.0;
        assert_eq!(stream_width(&key, &slice(&key, 64), &head), d);
        let mut down = blank(d, v, 1, 1);
        down.layers[0].down[100] = 1.0;
        assert_eq!(stream_width(&down, &slice(&down, 64), &head), d);
        let mut scored = blank(d, v, 1, 1);
        scored.lm_head[2 * d + 70] = 1.0;
        assert_eq!(stream_width(&scored, &slice(&scored, 64), &head), d);
        // The same row outside the head's stored run is never scored.
        assert_eq!(stream_width(&scored, &slice(&scored, 64), &HeadRows { first: 0, count: 2, fill: 0.0 }), 64);
    }

    /// A stream whose head rows fall into groups by the channels a pass can set: channels
    /// 2 and 3 come from the embeddings, a live unit writes 4, only a dead unit writes 5
    /// and nothing sets 6.
    fn grouped() -> (Model, Slice, HeadRows) {
        let (d, v) = (8, 12);
        let mut m = blank(d, v, 1, 2);
        m.embed[d + 2] = UNIT;
        m.embed[2 * d + 3] = UNIT;
        let l = &mut m.layers[0];
        (l.up[2], l.gate[3], l.down[4 * 2]) = (1.0, 1.0, 1.0);
        (l.up[d + 6], l.gate[d + 3], l.down[5 * 2 + 1]) = (1.0, 1.0, 1.0);
        m.lm_head.chunks_mut(d).for_each(|r| r[0] = -1024.0);
        // Token: (channel, weight) pairs, the constant on channel 0 first.
        let rows: [(usize, &[(usize, f32)]); 8] = [
            (2, &[(0, 5.0), (2, 1.0)]),
            (3, &[(0, 7.0), (2, 1.0)]),
            (4, &[(0, 0.0), (3, 2.0)]),
            (5, &[(0, -3.0), (3, 2.0), (5, 9.0)]),
            (6, &[(0, 0.0), (1, 0.125), (4, 1.0)]),
            (7, &[(0, 3.0), (4, 1.0)]),
            (8, &[(0, 0.0), (2, 2.0)]),
            (9, &[(0, 1.0), (2, 1.0)]),
        ];
        for (t, row) in rows {
            row.iter().for_each(|&(ch, w)| m.lm_head[t * d + ch] = w);
        }
        let s = slice(&m, d);
        (m, s, HeadRows { first: 2, count: 8, fill: -1024.0 })
    }

    /// The first token of the stored run with the largest score at residual `x`.
    fn choice(m: &Model, rows: &HeadRows, x: &[f32]) -> usize {
        let d = m.cfg.hidden;
        let score = |t: usize| m.lm_head[t * d..][..d].iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
        (rows.first..rows.first + rows.count).fold(rows.first, |b, t| if score(t) > score(b) { t } else { b })
    }

    #[test]
    fn a_streams_head_run_holds_the_choice_at_every_residual_its_passes_reach() {
        let (m, s, rows) = grouped();
        // Tokens 3, 4, 6 and 8 each have the largest constant among the rows that agree
        // with theirs on channels 2, 3 and 4, and the run spans them.
        let (first, count) = stream_head(&m, &s, &rows);
        assert_eq!((first, count), (3, 6));
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % 41) as f32 - 20.0
        };
        // What a pass can hold: the constants, anything on the channels its tokens and live
        // units set, and 0 on the rest.
        let mut chosen = std::collections::BTreeSet::new();
        for _ in 0..20_000 {
            let x = [CONSTANTS[0], CONSTANTS[1], next(), next(), next(), 0.0, 0.0, 0.0];
            let t = choice(&m, &rows, &x);
            assert!((first..first + count).contains(&t), "the stored run chooses token {t} at {x:?}, outside the stream's run {first}..{}", first + count);
            chosen.insert(t);
        }
        // Both ends of the run are chosen somewhere, so no shorter run would do.
        assert_eq!((chosen.first(), chosen.last()), (Some(&3), Some(&8)), "{chosen:?}");
    }

    #[test]
    fn without_the_constants_a_streams_head_run_is_the_stored_one() {
        let (mut m, s, rows) = grouped();
        m.embed[0] = 0.0;
        assert_eq!(stream_head(&m, &s, &rows), (rows.first, rows.count));
    }

    /// A header holding only metadata, over an empty file.
    fn header(u32s: &[(&str, &[u32])], f32s: &[(&str, f32)]) -> Gguf {
        let path = std::env::temp_dir().join(format!("gbcformer-gguf-header-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::write(&path, []).unwrap();
        let file = BufReader::new(File::open(&path).unwrap());
        std::fs::remove_file(&path).unwrap();
        Gguf { file, data: 0, u32s: u32s.iter().map(|&(k, v)| (k.to_string(), v.to_vec())).collect(), f32s: f32s.iter().map(|&(k, v)| (k.to_string(), v)).collect(), strs: HashMap::new(), tensors: HashMap::new() }
    }

    #[test]
    fn a_head_run_outside_the_vocabulary_is_refused() {
        let fill = [(HEAD_FILL_KEY, -1024.0)];
        assert_eq!(header(&[], &[]).head_rows(100), Ok(HeadRows::whole(100)));
        assert_eq!(header(&[(HEAD_ROWS_KEY, &[10, 90])], &fill).head_rows(100), Ok(HeadRows { first: 10, count: 90, fill: -1024.0 }));
        for (rows, fill) in [(&[10u32, 91][..], &fill[..]), (&[10, 0], &fill), (&[10], &fill), (&[10, 20, 30], &fill), (&[10, 90], &[])] {
            assert!(header(&[(HEAD_ROWS_KEY, rows)], fill).head_rows(100).is_err(), "{rows:?} with {} fill", fill.len());
        }
    }

    #[test]
    fn a_column_prefix_outside_the_channels_is_refused() {
        assert_eq!(header(&[], &[]).col_prefix(HEAD_COLS_KEY, 128), Ok(128));
        assert_eq!(header(&[(HEAD_COLS_KEY, &[64])], &[]).col_prefix(HEAD_COLS_KEY, 128), Ok(64));
        for cols in [&[0u32][..], &[129], &[64, 64], &[]] {
            assert!(header(&[(EMBED_COLS_KEY, cols)], &[]).col_prefix(EMBED_COLS_KEY, 128).is_err(), "{cols:?}");
        }
    }

    #[test]
    fn an_up_run_outside_the_channels_is_refused() {
        assert_eq!(header(&[], &[]).up_cols(2, 128), Ok(vec![(0, 128); 2]));
        assert_eq!(header(&[(UP_COLS_KEY, &[0, 64, 64, 64])], &[]).up_cols(2, 128), Ok(vec![(0, 64), (64, 64)]));
        for runs in [&[0u32, 64][..], &[0, 64, 64, 65], &[0, 64, 64, 0], &[0, 64, 64], &[0, 64, 64, 64, 0, 64], &[u32::MAX, 64, 0, 64]] {
            assert!(header(&[(UP_COLS_KEY, runs)], &[]).up_cols(2, 128).is_err(), "{runs:?}");
        }
    }

    #[test]
    #[should_panic(expected = "is not an f16 value")]
    fn f16_bits_refuse_a_value_f16_cannot_hold() {
        f16_bits(1.0 + f32::EPSILON);
    }
}
