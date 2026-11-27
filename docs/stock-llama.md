# The one model in a stock Llama runtime, exact in f16

## What runs where

Everything here is llama.cpp b12070 (commit `00f113ec`) on an M1 Max, and the model is the current one model, `ff86dde7`, unless a row names another. The unpatched source appears in two builds. "The default build" uses llama.cpp's own default options, which on a Mac turn on Metal and BLAS (Apple's Accelerate). "The BLAS-off build" adds `-DGGML_METAL=ON -DGGML_BLAS=OFF`. Both cmake lines are in `docs/stock-llama.md`, "What runs where".

| Runtime | Build | File | Status | Checked to |
|---|---|---|---|---|
| Metal, Apple's GPU backend | unpatched source, the default build | the one model as one uniform GGUF (sha1 `2f87678b…`, 1,544,154,592 bytes) | **The claim of record.** Every pass runs every weight of the model | Yellow from the hand-off state to frame 300: all 1,374,538 passes equal to the engine's record, and frame 300 hashes to `cfd31f3d…`. This run is the proof of record (2026-11-25, 15 h 28 min 31 s with the GPU lock; `docs/perf-log.md`, "The proof of record on `ff86dde7`"). Two one-frame runs of Zelda DX on this build and file cover what Yellow's 300 frames do not (2026-11-26): frame 7 with 438 lines lent (21,110 passes) and frame 25 with the pixel stepper (14,982 passes, 3,392 of them the stepper's), every pass equal to the engine's record (`docs/perf-log.md`, "Loans and the stepper in unpatched llama.cpp") |
| Metal | unpatched source, the BLAS-off build | the same file | History, the build of the first proof. Decodes correctly | Yellow through frame 13, all 138,984 passes equal, and no more. The first proof (2026-11-23) ran this build on an older model, `45bfb056`, in that model's uniform file: Yellow's 300 frames, all 1,374,538 passes equal, 71 h 34 min with the GPU lock |
| Mac CPU backend | unpatched source, the default build | the same file | Decodes correctly | Yellow's first 400 passes equal to the record, and no more |
| Mac CPU backend | unpatched source, the BLAS-off build | the same file | **NaN at the first GBC CPU pass** | measured on `45bfb056`, `7b064d51` and `ff86dde7` |
| Mac CPU backend | unpatched source, BLAS off, with `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.4-a+nofp16` | the same file | Decodes correctly | Yellow's first 400 passes equal, and no more. `7b064d51`: 400 passes. Not run on `45bfb056` |
| Metal | three patches on the same source (`layer-shapes.patch`, `stream-slices.patch`, `stored-runs.patch`), BLAS off | the trimmed GGUF | A speed-up only, never the claim. Each pass runs its stream's slice of the model | 3,000 passes of Yellow with every sliced pass also decoded by the whole model, 0 rows differ; 4,000 and 3,500 passes of Yellow and Zelda DX to frame 7 (21,110 passes), all equal to the record. A GBC CPU pass takes 12.0 ms against 35.4 unpatched (measured); the 300 frames are not run (6.4 hours, inferred) |
| Any other backend (x86, CUDA, Vulkan) or llama.cpp version | | | Not run | |

What the proof of record covers, and what it does not:

- It starts at the hand-off state the host sets up without a boot ROM (`boot none, warmup 0` in its checkpoints). It does not run the boot ROM from power-on.
- Every pass runs every weight, but a pass attends its stream's scope of the cache's cells, not every cell (`docs/stock-llama.md` §13, "The scopes narrowed"). Each scoped GBC CPU, PPU and length pass was held to the lines that the engine's record says its heads read. No pass was decoded a second time over every cell.
- Yellow lends no lines in 300 frames, and the pixel stepper's stream has no pass, so the proof of record exercises neither. Two separate runs on the same build and file do, each one frame of Zelda DX after frames the reference interpreter ran (`--warmup`). Frame 7 lends lines: all 21,110 passes equal the engine's record, 438 lines lent and put back in 876 write-record decodes, the 424 cells that the harness compares after the put-back all equal to the cells found, and 2,250 write records, those 876 decodes and 160 reader records each decoded both ways with 0 cells and 0 rows that differ. Frame 25 has the stepper: all 14,982 passes equal, 3,392 of them stepper passes that draw one line. Not run in unpatched llama.cpp: a stepper pass in Yellow, Crystal or Mario DX (their first is at frame 1,154, 3,087 and 1,627), a stepper pass on a DMG on `ff86dde7`, a frame with both loans and stepper passes, and more than one frame of either.
- The harness binary had its library search path changed to the run's folder and was signed again ad hoc. The run's `STATUS` file records the sha1 before and after (the run's folder is local and not in this repository; `docs/perf-log.md`, "The proof of record on `ff86dde7`", quotes it).
- The first proof used the BLAS-off build. On Metal the two builds gave the same rates over 3,000 passes (`docs/perf-log.md`, "Round 14").

The Mac CPU rows differ by build options only: no source patch, and no file or weight differs. With BLAS off, the Arm build sums products in 16-bit floats, and a whole-model pass runs every other stream's units at every position, where some of their sums pass 65,504, the largest 16-bit float (1,863 of the model's 76,143 gate rows can: `cargo run --release --example f16foreign`). Metal sums in 32-bit floats, and so does a build without FP16 vector arithmetic (`+nofp16`). That Accelerate is what makes the default build right is inferred from these runs; which multiplies it takes was not traced. Every Mac CPU run here started from a cache that Metal prefilled.

In every row a pass decodes its stream's own group of tokens (`STOCK_GROUP=own`), never the whole sequence, and that is what both proofs ran. Over the whole sequence the other parts' positions carry those parts' modes over this stream's bytes, values no part computes exactly, and on Metal the rows of a GBC CPU pass then differ from its own group's (measured on `ff86dde7` and on `7b064d51`: 148 rows at the first GBC CPU pass; `docs/stock-llama.md` §12a).

The default build, the build of the proof of record, is the unpatched b12070 source configured with no `GGML_` option at all:

```sh
cmake -S <llama.cpp> -B <build> -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build <build> --config Release --target llama
```

The options passed choose only a release build, shared libraries and which of llama.cpp's programs to leave out. On this Mac the cache then holds `GGML_METAL=ON`, `GGML_BLAS=ON` with vendor Apple, `GGML_ACCELERATE=ON`, `GGML_NATIVE=ON` and `GGML_OPENMP=ON` (the proof folder's `STATUS` lists the cache and each library's sha1).

The BLAS-off build, the build of the first proof, adds `-DGGML_METAL=ON -DGGML_BLAS=OFF` to that line. The supported Mac CPU builds are the default build, and, with BLAS off, the build without FP16 vector arithmetic:

```sh
cmake -S <llama.cpp> -B <build> -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.4-a+nofp16 -DGGML_METAL=ON -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build <build> --config Release --target llama
```

with the harness built against either (`LLAMA_CPP_DIR`, `LLAMA_CPP_BUILD`) and run with `STOCK_GPU_LAYERS=0`. Only the two `GGML_` options before `-DGGML_METAL` differ from the BLAS-off build's line. `tools/stock-llama/f16-sums.c` shows which way a build sums. The sections below are the history.

**Now.** The canonical artifact is the one model's per-layer-shaped GGUF:
the Llama layer set with per-layer shapes, with the LM head, each `ffn_up` and the embedding
stored at their runs of rows and columns (`docs/harness.md` §2). It runs in llama.cpp b12070
with three patches, applied in this order, whose digests the manifest names:
`tools/stock-llama/layer-shapes.patch` lets each layer have its own head count and FFN
width, `stream-slices.patch` runs a pass on its stream's slice, and `stored-runs.patch`
loads the stored runs and runs a stream on its own tokens and width. Before the stored runs the
file needed the first alone. The padded uniform export runs in unpatched llama.cpp on Metal
and is the claim of record ("What runs where", above). That export has run Yellow from the
hand-off to frame 300 in unpatched llama.cpp on Metal with every pass equal to the engine's
record (§13, "The full run on `ff86dde7`, measured"; the export of `45bfb056` did so first).
The sections below are the history of getting there.

Status: built (T1). §9 has the results.
`cargo run --release --example stock -- [rom] [--time] [--census]` printed every number
marked measured (the example has since been removed), and `examples/f16census` lists any circuit step an f16 checkpoint
cannot hold.

## 0. Why the model must change

A stock runtime caches a key and a value for every KV head of every layer for every
cached token, and on Metal and the Neural Engine it multiplies in f16. The one model
(digest `490c6845`) fails on both counts:

| | one model today (measured) |
| - | - |
| config | hidden 4,096, intermediate 4,736, 14 layers, 52 heads, 52 KV heads, vocab 35,253 |
| dense parameters, nonzeros | 1,485.3 M, 1,482,153 |
| KV per cached token (f32) | 14 · 52 · 32 · 2 · 4 = 186,368 B |
| Yellow's cache | 1,131,308 records + 29,444 window + 425 group = 1,161,187 cells: **216 GB** |
| largest weight | 2^23 · 65,408 (gate rows), constants 2^20 in the embedding |
| dense pass per CPU step (`forward_cached`, one core) | 38.4 s with an empty cache, 103.4 s over 4,096 cells, so about 18,500 s over Yellow's full cache (linear in cells; the memory heads' attention); the CPU + memory checkpoint 0.13 s, 0.49 s, about 103 s |

The table's step is a GBC CPU step, and its last figures are for the GBC CPU checkpoint
with memory. Every weight above 65,504 is infinite in f16, which is why Metal decodes
every row as token 0 (a finding of the verification run). Two families of magnitudes
cause it. The normalisation trick puts constants of 2^20 in the embedding and gives
every wire weight a factor of 2^15 · K. The circuits' wide linear forms put 16- to
21-bit adders and comparisons in one level.

## 1. Tiers

- **T1, required.** Every weight, every matmul input and every KV entry is stored in
  f16, with f32 accumulation and f32 elementwise ops. This covers llama.cpp Metal
  `mul_mm` with half tiles, MLX f16, llama.cpp's CPU backend and HF in f16. Exact means
  that every argmax token is equal and every activation is the integer the circuit
  defines, under any summation order.
- **T2, built (§10).** Accumulation and elementwise ops are f16 too, as in Core ML on the
  Neural Engine. §6 states what T2 adds. The Neural Engine's matmul is narrower than
  f16, since a partial sum at or above 2^15 overflows, not only one above 65,504 (§11,
  measured).

## 2. RMSNorm without 2^20 constants

Domination moves from the constant channels to `rms_norm_eps`, and the scale moves into
the norm weight (HF `input_layernorm.weight`, llama.cpp `attn_norm`), which stock Llama
already has:

- The residual unit is `u = 2^-7`, so a wire of value v holds `v·u` in the residual. One
  constant channel holds `u` (the bias input). There is no `n_const`, and hidden is no
  longer a multiple of 1,024.
- `eps = 2^14`, and the norm weight is `γ = 2^14` on every channel.
- `rms = sqrt(mean(x²) + eps)`. Every channel holds |v| ≤ 1 (measured as max |wire| 1 in
  all four circuits), so `mean(x²) ≤ u² = 2^-14`.
  - With an f32 reduce, half an ulp of 2^14 is 2^-10 > 2^-14, so `mean + eps` rounds to
    2^14 in any order. Every partial sum of squares is ≤ d·2^-14 < 1 and exact.
  - With an f16 reduce, half an ulp of 2^14 is 8, and Σx² ≤ d·2^-14 < 1 cannot overflow.
  - `sqrt(2^14) = 2^7` exactly under IEEE sqrt. An approximate rsqrt (Metal fast math)
    that is off by a few f32 ulps still lands on the integer once the normalised value is
    rounded to f16 for the matmul (relative error 2^-22 against f16's 2^-11).
- `h = x · 2^-7 · 2^14 = v` exactly, so the matmul input is the wire's own integer. Both
  orders (`x·rsqrt` then `·γ`, or `x·γ` first) stay normal in f16. At |v| = 1, `x·rsqrt`
  is 2^-14 (f16's smallest normal) and `x·γ` is 2^7. No step forms a subnormal, so a
  runtime that flushes subnormals is safe.

This relies on an f32 or f16 reduction with round-to-nearest, and on `sqrt` being exact
on 2^14 (or within a few ulps, which the f16 rounding absorbs). The census bound that
makes it hold is "every channel in [−1, 1]". `Model::check_f16` will assert it per
channel from the declared ranges, as the norm bound in `docs/one-model.md` §4 does today.

## 3. Integer ranges (T1)

With `h` the wire integers and the constant 1:

| Quantity | Value | f16 | Bound |
| - | - | - | - |
| gate weight on a wire | K·w | exact if |w| ≤ 511 at K = 128 (≤ 9 bits · 2^7) | |w| ≤ 511 |
| gate bias (constant channel) | K(c − θ) + {0, K} | multiples of 128 ≤ 65,408 exact | |c − θ| ≤ 510 |
| unit activation `silu(z)·up` | K·n, n = s − θ + 1 on its live side | exact for n ≤ 511 | live side ≤ 511 |
| up weight | 1 | exact | |
| down weight | coef · u / K = coef · 2^-14 | normal, exact for |coef| ≤ 2047 | |
| down product | n · coef · 2^-7 | exact | |
| residual | v · 2^-7 | exact | |v| ≤ 1 |
| embedding | 0 or u | exact | |
| LM head | ±1 on 12 route channels, constant on the bias | exact, logits ≤ 12 | |
| f32 partial sums of z | multiples of 128 | exact while < 2^31 | any order |

Every step `[s ≥ θ]` can be oriented as itself or as `1 − [−s ≥ 1 − θ]`, so the bound
on activations is `min(max(s) − θ + 1, θ − min(s)) ≤ 511`. `--census` measured these
violations:

| Circuit | Steps | Live side > 511 | |w| > 511 | |c − θ| > 511 | Worst |form − θ| | Families |
| - | - | - | - | - | - | - |
| CPU + memory | 7,551 | 60 | 88 | 93 | 98,304 | `hl+pair`, `sp+e8`, `a16+1`, `pair±1`, `sp±1`, `sp±2` (16-bit adders) |
| APU | 7,176 | 519 | 515 | 630 | 1,048,789 | `inL`, `inR` (mixer sums), `SL`, `SR`, `capL/R`, sweep adders |
| control | 2,153 | 229 | 230 | 277 | 1,921,619 | `hppu*`, `hirq`, `s`, `pause_t` (wide counters and horizons) |
| PPU | 13,071 | 45 | 45 | 45 | 12,727 | `bf0..bf8` |

The APU's and control's forms reach 2^21. With K = 256 their pre-activations reach
2^29, where the f32 sum is exact only because every term is a multiple of 256. Reduced
precision inside AMX loses that exactness, which is the likely cause of the Accelerate
mismatch on APU tokens while control matched (an inference, not checked).

**The fix is in the circuits, not the emitter.** A form `Σ 2^i b_i ≥ θ` over 16 or more
bits cannot have a small live side in one level. Each wide form becomes base-2^8 digits.
A byte-wide partial sum (live side ≤ 510) comes first, its carry or comparison flag at
the next level, and then the high digit with the carry folded in. Each digit boundary on
that path costs one level. `Circuit::legalize_f16` will do this generically for any step
outside the bounds (a pass over `Circuit`, run before `schedule`), so the four builders
change only where a hand split saves a level. The GBC CPU's fetch adders are already
split at bit 7 (unit I) and are not in the table. Expected depth is 9 to 10 for the GBC
CPU, 13 or 14 for the PPU (from 13), and for the APU and control whatever their longest
wide chain needs. The one model's depth is the PPU's.

## 4. Memory without RoPE recency

Today a read finds the newest write by RoPE. The recency amplitude is `M = 1.66·10^7`
logit units, a per-position gap of 256 against logits near 10^8
(`memory::attn::Params`). The query entry `M/scale = 9.4·10^7` is infinite in f16, and
storing the rotated key in f16 (ulp 2^-11 near 1) moves the logit by about
`9.4·10^7 · 2^-12 · 0.177 ≈ 4,000`, fifteen times the gap. No scaling of RoPE recency
survives T1.

Encoding order in the key instead (a host clock input, digits in two dims) fits f16
entries. Address dominance, though, needs the per-bit gap above the whole recency range
and the sum exact in f32 (`22·σκ < 2^24`). The window is then
`W < 2·2^24/(22·589) ≈ 2,590` writes, a compaction about every 6 frames of text box.
Under T2 (`22·σκ ≤ 65,504`, gap 18 after scale) `W ≈ 60`. This is workable at T1 but not at T2.

**Chosen: one cell per 8-byte line.** The runtime cache holds exactly one token per line
of physical memory, at the line's index as its position, and a write replaces the cell.
The first form held one cell per byte (1,131,308 cells for Yellow, nibble-one-hot
values). A later change ("sm83: memory circuit reads and writes whole lines") moved
every read and write to lines, which is what this section and the harness now describe.
§12b's measurements are the byte form's.

- The cell of line `n` (byte addresses `8n..8n+7`) sits at position `n`. Every stream
  group and every record's scratch tokens sit at `P = READ_AT = LINES = 0x108800`
  (`memory::attn::READ_AT`), and a write record's cell token sits at `P + 1` and moves to
  `n` once decoded. A group at P sees every cell (causal, pos ≤ P), so every memory head
  of every stream, foreign parts' heads included, reads exactly one cell.
- The key is the line address alone. It holds ±1 for each of the 21 line-address bits
  (byte address bits 3..23): bits 0..17 on the first halves of pairs 13..30 (dims
  13..30), bits 18..20 on the second halves of pairs 29..31 (dims 61..63), and
  `committed` (+1 on a cell) on dim 31. Every other partner dim is 0 in every query and
  key (`memory::attn::KEY_DIMS`). Until the 24-bit layout the 19 bits sat one per pair on
  pairs 12..30 (`docs/perf-log.md`, "Memory: one key bit per RoPE pair"); pair 12 turns too far at the new `KEY_REACH`, and pairs
  29..31 turn by less than 10^-27 rad there, so a roped entry on either half equals its
  unroped value bit for bit (`key_pairs_turn_by_nothing`). Before that change
  the bits sat on dims 10..20 and 42..49, so pairs 10..17 carried a bit on both halves.
  The query is `±S` per bit with S = 512, and `B = 1,024` on committed
  (`Params::memory()`; S = 1,024 and B = 2,048 before the 24-bit layout). A one-bit
  mismatch and an uncommitted token holding the same line address each score 1,024 less
  than the cell, a gap of `1,024 · 2^-3 = 128` after scale (head dim 64), against the 104
  where f32 `exp` is 0. Every entry and every partial sum is a multiple of 512, so f16
  accumulation is exact too. A column select (`circuit::Query::gated`,
  `docs/hardware-shaped.md` §20) asks with 3S on the bits its open and off lines share.
  The widest logit is then 3S·21 + B = 33,280, a multiple of 512 below f16's 65,504
  (at S = 1,024 it would be 66,560, past it), and a mismatched bit still costs at least 2S. `Params::check` holds both, and
  it evaluates the margins in f32 and with the query and keys rounded to f16, for cells
  at the first, a middle and the last line.
- RoPE turns pair 13 by 3.7·10^-16 rad per position, so a key and a query at most
  `KEY_REACH` = 8,667,138 positions apart (H3's byte positions included) differ by less
  than 3.2·10^-9 rad on every key pair. cos rounds to exactly 1.0 in f32 and sin to 0 in
  f16. A roped key entry is exactly ±1 with a ±0 partner, and a roped query entry is
  exactly ±S, ±3S or B, so every logit is the same multiple of 512 in any summation
  order and at any position. In an f32 cache a partner's product `(q·sin)(k·sin)` is
  below 2^-38, which no sum holding a multiple of 512 keeps
  (`memory::attn::key_pairs_turn_by_nothing`, run by `Params::check`, and the test
  `a_cells_logit_is_one_number_in_any_order`).
- The value is the line's 64 bits, with byte k's bit i in dim `8k + i`. The circuit picks
  the byte a read wants (`circuit::low_match`). A write record spells the whole line
  after the write, so a write is read-modify-write in the model. The step reads the line
  at its physical address and decodes the line with its byte over it.
- **The prefill order the rows' exactness needs** (the standalone cpu-memory
  checkpoint). A record's cell rows are exact when the norm at its positions divides by
  exactly the constants' power of two, which holds only while every channel there stays
  in its range. The circuit runs at a record's positions too, and its column heads
  (`docs/hardware-shaped.md` §20) read VOID's zero line when off. Without VOID's cell an
  off column reads the best other key, which at a record's positions is one of the
  record's own tokens, and a bus adds its columns. A record of line 0 decoded first
  reached squares of 719,567 by layer 5 against a bound of 29,696. This is measured, not
  constructed. The GBC CPU's other wires at a record's positions see gathered garbage too
  and stay in range only in practice, and `DenseMemory::push` holds every record's key
  and value to the formula bit for bit. So the order is an enforced rule.
  `DenseMemory::new` keeps VOID's zero line first, and `push` refuses a record before
  it. The harness's prefill decodes the top cell, then VOID's line, then the fixed
  regions, then the top cell and VOID's line again over them, and `Cache::put` refuses
  any other record decoded before VOID's cell. In the one model the GBC CPU's units are
  off at a record's positions (mode 0, `up` = 0) by construction, and its memory heads
  there attend to the null cell, which the harness decodes before anything else (below).
- **The null cell** (the one model, `docs/perf-log.md`, "The one model in stock llama.cpp"). Outside its part's
  positions an overlay memory head attends to the null cell (`docs/hardware-shaped.md`
  §15.2). The dense pass appends it to every cache as a formula (`llama::null_kv`, with
  `null_k` on the null dims, 0 elsewhere, and value 0). A runtime holds only what it
  decodes, so the null token must be a real token whose own forward pass gives that key
  and value. At first it did not. A key dim is `2·bit − 1`, so the decoded null keyed −1
  on all 19 key dims (line 0, uncommitted), and copy-only layers carried no null key at
  all. Every memory layer's key rows now read the null channel, with `null_k` on the null
  dims and +1 on each key dim, which cancels the −1 there. Only the null token sets the
  channel, so no other key changes, and a stream's slice drops the column with the
  channel (`Model::stream`). `Model::null_cell` runs the token alone at position 0, and
  `the_null_token_decodes_the_null_cell` (`tests/one_model.rs`, ignored, 96 s) holds it
  to `null_kv` at every memory layer bit for bit. The same test shows that the cell is
  needed. Over 2,041 random lines, the whole checkpoint's dense pass equals the control
  stream's slice at all 54 of its output positions with the null cell (the formula's or
  the decoded one) and differs at all 54 without it, since a foreign head off the null
  cell reads a real line into channels the stream's own wires share. The harness decodes
  the null token alone at position 0 of the empty sequence 0 (`Runtime::null`), before
  the top cell. Position 0 is line 0's, a ROM line no write evicts (asserted), and two
  cells may share a position. Every scope and lane copies the null cell, and the saved
  state carries it. The sidecar names it, so the harness does not load a state without
  it.
- **Queries finite in f16** (the one model). Metal's flash attention holds the query in
  f16 (`q_t` is `half` in `fa_f16.metal` and `fa_vec_f16.metal`, and its Q·K sums are
  f32). A larger entry becomes inf, which gives a NaN logit against any key's zero
  partner and then poisons every token that attends it. At another part's positions an
  overlay head's query reads that part's wires. The PPU's layer-4 heads over the GBC
  CPU's nibble channels reach 414,720 (`Model::attention_bounds`, an upper bound over
  every occupant's range, after RoPE). The GGUF divides layer l's `attn_q` by
  2^shift[l] and multiplies its `attn_k` by the same factor (`gguf::qk_shift`, the least
  power that keeps every query and key entry at most 65,504; metadata
  `gbcformer.qk_shift`). The shift is 3 at layer 4 and 0 elsewhere. Layer 0's keys reach
  16,416, so one shift for all layers would not fit. Every product `q·k` is the same and
  both factors are exact in f16 (the export asserts every weight), so the logits are
  bit-identical. The scaled keys' RoPE partners (`8·sin`) are no longer flushed to 0 in
  an f16 cache, which leaves the same partner products below 2^-38 that an f32 cache
  leaves (above), and no logit holding a multiple of 1,024 keeps them. The loaders undo
  the shift (`digest --gguf`: 0 of 594,721,792 differ).
- A write's record decodes after the old cell is evicted, so its cell token is the only
  cell for its line in that pass. The engine's and `DenseMemory`'s oracles also evict
  before they decode the record.
- The host decides which cell to drop. It maps the address the model decoded (the
  `WP0a..c` tokens) to its line and calls the runtime's eviction, the same operation it
  already performs to drop a group. It supplies no byte and no address. Newest write
  wins as a cache rule, not by a logit margin.

The project adopted three rules: one cell per physical address, the
address taken only from the model's decoded tokens, and newest-wins as a cache rule.
Lines keep all three, since the cell's address is the line the decoded
address names.

Each of the runtime's cache operations is a stock llama.cpp call. Sequence `S` is the
machine's, and `llama_memory_seq_rm(mem, seq, p0, p1)` drops the cells of `seq` with
`p0 ≤ pos < p1`. A cell left in no sequence is freed. `memory::dense::Records` builds
both records.

| Event | Operations | Where the address and line come from |
| - | - | - |
| a stream pass (CPU step, PPU line, APU segment, control span) | `llama_decode` of the group, every token at `P`; then `seq_rm(S, P, P + 1)` | none |
| a write to line `n` (CPU write `w`, DMA destination, a HOST byte) | `seq_rm(S, n, n + 1)` drops the old cell; `llama_decode` of the 15 scratch tokens `REC WP0a WP0b WP0c WP1a WP1b WP1c WL0..WL7` at `P` (`WP1` spells 0) and the cell token `CELL` at `P + 1`; `seq_rm(S, P, P + 1)` drops the scratch; `llama_memory_seq_add(mem, S, P + 1, P + 2, n − P − 1)` moves the cell to `n` (the K-shift re-ropes its key by below 4·10^-7 rad on the address pairs) | the `WPw` and `WLw` tokens the model decoded in the step (a DMA write's line is its reader record's decoded line; a HOST record's byte is the host's, labelled); the record repeats those tokens |
| a DMA source read, a PPU palette read | `llama_decode` of the reader record `REC WP0a..c(dest) WP1a..c(source) WL0..7 masks RD0..RD7` (23 tokens) at `P`, the eight `RD` rows read; `seq_rm(S, P, P + 1)` | the control logic's or host's DMA schedule, labelled as today; the `RD` tokens decode the destination's line with the source's byte over it |
| prefill (cartridge load) | per line, the write row above without the first `seq_rm`, its `WP0` spelling the line's first byte; the fixed regions one record per `llama_decode`, the rest in lanes (§12) | the loader's enumeration of the layout |

All streams share one sequence. The APU and control streams see the memory too, so each
of their foreign memory heads reads one cell instead of averaging over the group (the
verification run's finding against `docs/one-model.md` §4). In f16 an average can be
worse than inexact. A foreign unit whose inputs leave their declared range can overflow
to inf, and `0·inf` in the down projection poisons every channel. Two records in one
sequence of one `llama_decode` would see each other's address tokens, which carry the
same key, so the prefill gives each record of a batch its own sequence (§12).

Our oracles (`Engine::oracle`, `DenseMemory::push`) decode a whole record at `READ_AT`
and compare the cell token's key with the formula there. The runtime decodes it at
`P + 1` and shifts it to `n`. In f16 both round to the same ±1 entries on the address
pairs. In an f32 cache they differ by the rotation of one position, so a harness that
compares f32 keys sees that difference, which is not an error.

**Two decisions in the harness (`tools/stock-llama`).**

- The harness does not decode the PPU's, the stepper's or the length's VRAM and OAM
  records as they are kept. Each of those images follows the GBC CPU's write records a
  byte at a time, applied at a line's cut (the start of its mode 3) in the order the GBC
  CPU kept them, and the GBC CPU's record already decoded the line's cell. The harness
  decodes only those streams' palette records.
  The engine draws a line when it closes, at the next line's cut (`Ppu::close`). By that
  pass the GBC CPU has kept a line's worth of later records (OAM DMA bytes, HBlank
  writes), and the shared cells hold them, while the engine's image for the pass does
  not. So the harness keeps each such stream's own lines (`Stream::own`: its image's at
  the insert, then the line of every write record its engine kept there). Before a pass
  it decodes the stream's line into each cell that holds another line, and after the
  pass's last decode it decodes the cell's line back (`Cache::lend`, `Cache::put_back`).
  The pass then attends the records its engine's image held and none kept later.
  Two checks guard it. The mirror check on every such pass, made with the lines lent,
  holds every VRAM, OAM and palette byte of the engine's image to the lines the cells
  were written with. Each cell put back is held to the bytes saved before the loan
  (`Cache::decoded`), byte for byte or but for the sign of a zero entry of an f16 key
  row, which llama.cpp's shift of a new cell to its line can leave as -0. The tolerance
  covers the key rows only, found from the saved sequence's layout (`key_rows`), and
  nothing when the keys are not f16. A cell that differed by such a sign is read again
  after each later shift, and the counts say how many then equalled the saved bytes
  (`swap_signs_cleared`) and how many kept the sign through four shifts
  (`swap_signs_kept`). In llama.cpp (measured, below) a later shift cleared the sign of
  both such cells on `dma_basic` and of 74 of 136 in a frame of Link's Awakening DX, and
  no cell kept it through four shifts. The other 62 left the watch when their line was
  written again (inferred from `Cache::put`, which drops a watched line it rewrites).
  No healthy run leaves a cell uncompared. A line the next decode lends again (a CGB
  line's second pass) carries the first loan's saved bytes to the cell that loan puts
  back. A cell found before its shift, because the GBC CPU's record was the last decode,
  is compared before the shift of the cell put back. A cell put back that an engine's
  record replaces at the next decode was read by nothing and is counted apart.
  Under `STOCK_REFERENCE` the first `MIRROR DIFFERS`, the first cell put back that
  differs and the first one never compared stop the run with exit 1, and `proof.sh`
  fails on any error line in its log.
  `play replay` runs the same harness with no model: each pass decodes the tokens of the
  engine's record, so it checks no token, and it runs the scopes, the loans and the mirror
  check in the run's true order. Measured with it (no GPU): Yellow to frame 300 lends no
  line in any stream (0 of 83,498 PPU passes, 0 of 41,855 length passes; 0 mirror
  differences; frame 300 `cfd31f3d`), so the 300-frame proof does not exercise the swap.
  Link's Awakening DX lends in its first 60 frames (538 PPU passes, 1,132 lines, from
  frame 7; 36 passes lend exactly one line, 18 loans carried), Crystal in 24 PPU passes
  from frame 58, Mario DX in none. The stepper and the length lent in no ROM tried: the
  length's line pass runs right after its image applies the records. On `dma_basic` the
  replay counts what llama.cpp did: 2 passes, 14 lines, 28 swap decodes.
  No following stream's own head can name a cell outside VRAM, OAM and the palettes
  (measured from the scopes of Yellow, Crystal, Link's Awakening DX, Mario DX and
  cgb-acid-hell); the harness prints those lines per stream and counts any other
  stream's record that changes one.
  A cheaper design is known and deferred: give each following stream its own cells,
  shared with sequence 0 while the lines agree and decoded only for a line the GBC CPU
  overwrote first. It costs one decode per stale line instead of two per pass, never
  rewrites a cell the GBC CPU attends, and removes the put-back and the zero-sign
  question. It changes the write path the proof depends on, and Yellow's 300 frames lend
  nothing, so it waits until a run that lends is the target.
  Measured on gbmicrotest `dma_basic`, frame 1, where OAM DMA runs across the first two
  lines: 2 PPU passes with 14 lines lent, 0 mirror differences in 144 checks (2 of 144
  before the swap), all 14 cells put back equal (12 byte for byte, 2 but for zero signs),
  and all 7,214 passes equal to the engine's record. Both passes equalled the record
  before the swap too, so those OAM bytes changed no decoded token of the two lines. A
  later record that changes one would differ from the engine's record, where
  `STOCK_REFERENCE` stops the run (inferred; no ROM in the windows run so far does it).
  Yellow's first 6,000 passes keep no VRAM or OAM record, lend no line and count as before.
  Measured again with the fatal checks (the patched build with
  slices, every run under `STOCK_REFERENCE`, load average 6 to 14):
  - `dma_basic`, unscoped, frame 1. All 7,214 passes equal the record and the frame is the
    engine's `4df32995`. 2 PPU passes lent 14 lines. The 14 cells put back were compared:
    12 byte for byte, 2 but for zero signs, and a later shift cleared both. `key_rows`
    found the key rows of a real cell: 25 layers, 3,200 of its 7,048 bytes. 0 error lines.
  - Link's Awakening DX on CGB, scoped as the proof scopes, frame 7 after `--warmup 6`,
    with its own prefill. All 21,110 passes equal the record and the frame is the engine's
    `d043ab24`. 216 PPU passes lent 438 lines, 28 of them exactly one line, and 14 loans
    were carried to the next pass. 424 cells put back were compared, 0 differ, 0 were
    never compared, and 136 differed by zero signs. 0 error lines. The run stopped and
    resumed twice inside the frame (§14).
  - The same frame with `STOCK_WRITES=check` (over the cache that run
    saved, in three chunks). Each of the 2,250 write records and 876 loan decodes decoded
    over its scope and again over every cell: 3,126 pairs of cells, 0 differ in any byte.
    All 21,110 passes equal the record, and the run's record files equal the first run's
    byte for byte. An earlier attempt in one sitting ran out of its 880 s hold after 122
    lending passes with no difference: the second decode of a write record costs about
    180 ms.
  - A deliberate fault. `STOCK_FAULT_PUT_BACK=3` put line `0x105001`'s cell back with one
    bit of its first byte flipped, in llama.cpp, in the same Zelda frame. The next decode
    printed `SWAP CELL DIFFERS`, the run stopped at `SWAP NOT HELD`, and `play` exited 1.
  Reader records decode over every cell: 284 ms each over the 143,848 cells of the Zelda
  run (160 records) and 43 ms over `dma_basic`'s 16,872 (160 records).
- A prefill record's `WP0` spells the line's first byte address. The cell's key is the
  line address only, so any byte of the line gives the same cell. The first byte is the
  one a loader enumerates.

## 5. KV size: grouped-query attention

HF `num_key_value_heads = 1` (llama.cpp supports it, uniform across layers):

- In layers ≥ 1 every head is a memory head (the plan check confirms it), and once the
  class heads are gone all of them share the address key and the nibble value.
- In layer 0 the input gathers and the three address gathers share one key and one
  value. The value is the four lanes (32 dims), and the address tokens carry their bits
  in lane 0. The key is a code per target on the low-frequency dims. Input groups are
  2-hot over 16 dims (C(16,2) = 120 ≥ 32 groups), and the six address targets are
  one-hot over 6 dims. Query entries are 32 and key entries 32, so a match scores
  `2·1,024·0.177 = 362`, one shared dim scores 181, and the gap is 181. REC scores above
  every input group's query in a record, which holds no input token, as today. Group
  tokens share one position, so a group's four keys are bit-identical and each weighs
  exactly 1/4. These scores are the design's, at head dim 32. The code uses head dim 64,
  where a layer-0 target scores `32·32/8 = 128` and leads by 128
  (`src/llama.rs`), and the memory heads lead by 256 (`src/memory/attn.rs`).
- The host-fed GBC CPU checkpoint keeps one KV head per head. Its opcode one-hot gathers
  need distinct values, and it has no cache.

| | Today | After (T1, f16 KV, MQA) |
| - | - | - |
| KV per cached token | 186,368 B | `L · 2 · 32 · 2` = 1,792 B at 14 layers (1,920 B at 15) |
| cached tokens, Yellow | 1,161,187 | 1,131,308 + one group |
| KV, Yellow | 216 GB | 2.0 GB (2.2 GB at 15 layers) |
| KV, Crystal (2 MB ROM) | 412 GB | 3.9 GB |

f32 KV (llama.cpp's CPU backend) doubles those figures.

## 6. Softmax, SiLU and K

- **Softmax.** One-hot attention needs every other weight to be exactly 0. In f32
  `exp(−104) < 2^-149`, and in f16 `exp(−17.4) < 2^-25`, which rounds to 0. Every gap is
  at least 128 (layer 0) or 256 (memory heads). Weights 1 and 1/4 are exact in f16, and
  each output dim has one nonzero term.
- **SiLU at multiples of K.** For z = nK with n ≥ 1, `σ(z) = 1` needs `e^−K < 2^-24`
  (f32) or `2^-11` (f16), so K ≥ 17 or K ≥ 8. For n ≤ −1, `z·σ(z) = 0` exactly needs
  `e^K` to overflow, so K > 88.7 in f32 and K > 11.1 in f16. Below those, `silu(−K)` is a
  nonzero `−K·e^−K` that an f32 residual would carry. K must be a power of two for the
  down weight `coef·2^-7/K` to be exact. So K = 128 under f32 elementwise (T1), and
  K = 16 would do under f16 elementwise (T2).
- **T2 additions.** f16 partial sums of z are exact while every subset sum is a multiple
  of 128 at most 65,504, which per step means `Σ|w| + |c − θ| + 1 ≤ 511`. The census
  finds 172, 817, 445 and 1,091 steps that violate it in the GBC CPU, APU, control and
  PPU. The PPU's are wide sums of pixel bits, not adders. The down partial sums per
  channel are multiples of 2^-7 and exact while `Σ_units n·|coef| ≤ 2,048`, which a
  staircase of many live steps breaks. So T2 needs the legalisation of §3 at the
  stricter bound, and staircases with one live unit pair. At K = 16 the live-side bound
  becomes 4,094, which relaxes §3 eightfold. The plan is to build T1, then measure the
  T2 census on the legalised circuits and either legalise further or show the step that
  cannot be met.

## 7. Dense size

| | Today | After (projected) |
| - | - | - |
| attention (q, k, v, o) | 14 · 4 · 1,664 · 4,096 = 381.7 M | 14 · (2 · 1,664 + 2 · 32) · d ≈ 194 M |
| hidden | 4,096 (3,471 live wires + 532 shared, rounded to 1,024s) | ≈ 4,064: one constant, nibble channels + 32, class wires − 256 |
| total | 1,485.3 M | ≈ 1,290 M before legalisation adds units or a layer |

Two further levers have lower priority and will be measured later. One is balancing
memory heads across layers. There are 52 heads at the widest layer, while 434 heads over
13 layers average 33, so q and o would shrink by a third, and so would a stock runtime's
attention over the cache, which dominates its pass time. The other is the overlay
(`docs/one-model.md` §3, hidden 2,048).

The line lever stores several bytes per kept token. A 32-dim value holds 4 bytes, which
cuts the cells and the stock runtime's attention over them by 4. It needs a byte-select
level after every dynamic read (the PPU chains 2 to 3), read-modify-write for every
write record, and it leaves no room for the nibble one-hots. KV already fits with one
byte per cell (2 GB), so this lever waits until the runtime's pass time is the target.

## 8. Units, each ending in its gate

1. KV-head plumbing (`Config::kv_heads`, `Layer::repeat_kv`), with every checkpoint
   still at one KV head per head. Gate: every digest unchanged.
2. `Model::check_f16` and an f16-tile dense pass (`forward_cached` with matmul inputs and
   KV rounded to f16, f32 accumulation). This is the oracle for T1, and the test that
   fails on today's checkpoints.
3. Emitter scales (§2, §3: u, eps, γ, K = 128, one constant). Gates: SST on circuit and
   checkpoint, engine equal to dense, and the lockstep hashes.
4. `Circuit::legalize_f16` and each builder's census brought to zero. Gates: each
   circuit's own (SST, `dmg_sound`, `instr_timing`, acid2, lockstep hashes).
5. Memory with nibble values, no class heads, one cell per address, write records and no
   compaction, with the engine index and KV cache following. Gate: every memory gate.
6. One KV head (layer-0 codes). Gates: every gate, and `check_f16` passes on every
   checkpoint.
7. Measurements, digests, `docs/perf-log.md` and CLAUDE.md, plus the llama.cpp harness
   run when it lands.

## 9. Results (measured)

Before is `490c6845` (the weights at the start). After is the APU and
control builders narrowed. KV figures are for a stock runtime, which caches a key and a
value of every KV head at every layer for every cached token.

| | Before | After |
| - | - | - |
| KV heads | 52 (one per head) | 1 |
| KV per cached token | 186,368 B (f32) | 3,840 B f32, 1,920 B f16 |
| cached tokens, Yellow | 1,161,187 (1,131,308 records, the 29,444-write window, one group) | 1,131,743 (one cell per address, one group) |
| KV, Yellow | 216.4 GB | 4.35 GB f32, 2.17 GB f16 |
| KV, Crystal | 412 GB | 8.37 GB f32, 4.19 GB f16 |
| config | hidden 4,096, intermediate 4,736, 14 layers, 52 heads | hidden 4,096, intermediate 4,416, 15 layers, 52 heads |
| dense parameters | 1,485.3 M | 1,311.3 M |
| nonzeros | 1,482,153 | 1,386,384 |
| largest weight magnitude | 2^23 · 65,408 | 65,408 (every weight f16-exact, `Model::check_f16`) |
| dense pass per CPU step, one core, empty cache | 38.4 s | 37.6 s |
| same, 4,096 cells | 103.4 s | 89.4 s |
| same, extrapolated to Yellow's full cache | about 18,500 s | about 14,300 s |

The dense-pass times (per GBC CPU step) ran under different machine load from other
builds, so they show a scale, not a comparison. The pass is our reference pass
in Rust, not llama.cpp's. The harness's per-stream ms per pass are not measured yet.

Per part (`examples/one_model`), the PPU has 14 levels (13 before the f16 splits) and the
GBC CPU 9. The APU has 13 (from 12), because its builder narrowed the mixer, the filter
and the timer wraps. Control has 11 (9 before, 12 after the first split, 11 after its
builder's digit compares). The merged circuit had 3,663 live wires after the splits
(hidden 5,120), and `Circuit::fit` brings them to 3,540 (hidden 4,096). The PPU's palette
heads now fill the least-loaded layers, which takes the widest layer from 64 heads to 52.

**Exactness.** Every checkpoint passes `check_f16` (`tests/stream_audit.rs`). The
f16-tile dense pass rounds every matmul input, query, key, value and softmax weight to
f16. It decodes exactly what the engine decodes on the GBC CPU-memory checkpoint's
random steps (`tests/checkpoint_sst.rs`) and on the APU's random segments. Every
stream's merged circuit, with foreign parts at byte 0 and the PPU at sampled pixel
positions, stays at 0/1 on every node and names only addresses with a cell, over
Yellow's memory after its boot. `tools/gate.sh` passes on this branch, with every frame
hash unchanged.

**T2 (f16 accumulation, the Neural Engine) is not met.** The census counts the steps
whose `Σ|w| + |c − θ| + 1` exceeds 511, which f16 partial sums of the gate cannot hold
exactly in every order. It finds 66 in the GBC CPU, 96 in the GBC CPU-memory checkpoint,
512 in the APU, 180 in control and 1,091 in the PPU (the PPU's are wide sums of pixel
and sprite bits). Of those, 6, 14, 74, 30 and 0 steps also exceed the f16-accumulation
bias bound. Meeting T2 needs every such sum split into partial sums of at most 511 (one
more level each on those paths), and every node's down-projection partial sums kept
within 2,048 units of 2^-7. K = 16 (enough for silu in f16 elementwise) would relax both
eightfold. Nothing in the arithmetic rules T2 out. It is the same legalisation at a
tighter bound, and it is not done.

**What a stock runtime still pays.** Attention over the cache dominates a pass, at about
425 positions × 52 heads × 15 layers × 1.13 M cells × 64 multiply-adds. Three levers cut
it: fewer positions per stream (each stream holds every slot today), fewer query heads,
and the 4-bytes-per-cell lines of §7.


## 10. T2: exact under f16 accumulation

Status: built. Every checkpoint passes `Model::check_f16_acc`, and the f16-accumulation
dense pass (`Tiles::F16Acc`) decodes what the engine decodes. The census runs with
`cargo run --release --example f16census -- [cpu|mem|apu|control|ppu] [-v] [--model] [--path]`.

**The exact condition.** An f16 accumulator, with the products summed in any order (a
tree included), is exact on a row exactly when every subset sum of the row's products
is exactly representable. The products of a row are multiples of one power of two q,
and every multiple of q up to 2^11·q, and below 65,504, is an f16. The largest subset
sum is the larger of two sums: the row's positive products, and its negative products,
each at the inputs that maximise it. So the condition per row is

`max(P, N) ≤ 2^11` (in quanta) and `max(P, N)·q ≤ 65,504`,

with P and N taken over the input channels' ranges (`f16acc::gate_mass`). Inputs of one
one-hot group (the output slot identities, which are the PPU's position wires; a token's
own one-hot; a value token's two nibbles) are 1 one at a time, so a group adds its
largest weight, not the sum of its weights. The earlier bound `Σ|w| + |c − θ| + 1 ≤ 511`
is a weaker form of this condition, because it adds both signs together and ignores the
groups. Measured at the start, the exact census was GBC CPU 12, GBC
CPU-memory 48, APU 244 + 16 down rows, control 93 + 4, and PPU 0 + 0. The PPU's 1,091
violations under the old bound were all wide sums over position wires, which are
one-hot.

| Row | Quantum | Bound | What it means for the circuit |
| - | - | - | - |
| MLP gate | K = 128 | 511 | every step's gate mass ≤ 511 (both of its units) |
| MLP down | 2^-7 | 2,048 | per node: the positive (or negative) unit products ≤ 2,046, leaving room for the residual entry a fused accumulate adds and for a clearing unit |
| q, k, v | per row | 2,048 | the memory query ±1,024 per bit, layer 0's codes: met with large margin |
| o | 2^-7 / GROUP | 2,048 | at most 8 nibble entries per bit, one lane per gathered wire |
| LM head | 1 | 2,048 | at most 12 route channels |

`down_mass` evaluates the steps of one form jointly at both ends of its range, where a
sum of ReLUs with one sign peaks because it is convex. For different forms it uses what
each form needs in order to be positive, either a 0/1 literal whose unfavourable value
alone costs more than the headroom, or a one-hot group member. Forms that need
conflicting values are never live together, so a greedy clique cover counts only the
heaviest form of each clique. The PPU's 320-step sprite muxes rely on this. The
weight-level check rebuilds the same unit forms from the gate rows, using channel ranges
taken from the occupant each channel holds at that layer (`Model::channel_vars`).

**Elementwise ops and attention.** In RMSNorm the sum of squares is below 1, so its f16
rounding never matters, because `mean + eps` rounds to 2^14 either way. `sqrt(2^14)` and
`1/128` are exact, and every intermediate is a normal f16. This relies on an IEEE
`rsqrt` or `sqrt` on a power of two. An approximate rsqrt that is off by one f16 ulp
would break exactness, and the Core ML run (§11) checks it. For SiLU at K = 128,
`exp(−128)` underflows and `exp(128)` overflows in f16, so silu is exactly ReLU on every
pre-activation (multiples of K up to 65,408). Softmax subtracts the max logit. Every gap
is at least 128 (layer 0) or 256 (memory heads), and `exp(−128)` is 0 in f16. The equal
logits of a gather come from bit-identical keys, and Σ exp is an exact small integer, so
the weights are exactly 1 or 1/len. Each value dim has one nonzero product. Logit
partial sums stay within 24,576 (memory) and 2,048 (layer 0), multiples of 1,024 and
32².

**Legalisation (`Circuit::node_at`).** A step that holds in T1 but not T2 goes through
these rewrites in order:

1. `reduce` applies Shannon expansion to the 0/1 terms whose one value decides the step.
   It peels them into a chain of ANDs and ORs around a core step, then rebuilds the
   chain from the core out, giving each literal the least weight that keeps it deciding.
   Binary comparisons get Fibonacci-like weights. A run of ANDs (or ORs) costs one core
   live side per literal, so its literals become one shared AND (OR) wire. It first does
   this only for the literals below the step's top level (no level added), then for any
   of them.
2. `split_step` splits the step into digits at base 256, 64, 16 or 4. The narrow digit
   decides the threshold where the wide one ties, which costs one level.
3. A node whose down row is too heavy gets `reorient`, which turns each step toward the
   nearer end of its form so that its units stay small. If that is not enough, `runs`
   cuts a staircase over one form into runs of consecutive thresholds. Each run is 0/1
   up to a constant, so it becomes its own node, and the node is their sum. This adds no
   level.

`bit_of` and the APU's `sum_mod` reduce their form modulo 2^(k+1), taking each weight to
its residue nearest 0 (`Circuit::modulo`). The legaliser rolls back rejected rewrites
(`mark`, `rollback`), so they leave no dead nodes.

**The oracle.** `Tiles::F16Acc(order)` rounds every product, every partial sum and every
elementwise result (norm, RoPE, SiLU, softmax, residual add) to f16. It sums each row in
input order, positives first, or negatives first, and positives first reaches the
largest partial sums. `tests/stream_audit.rs` checks every checkpoint, the one model
included, with `check_f16_acc`. It runs the APU's and control's dense passes in all three
orders against their engines, and it holds a hand-built gate past 65,504 to failing both
checks while f32 decodes it right. `tests/checkpoint_sst.rs` runs the GBC CPU-memory
checkpoint the same way over its whole cached memory.

## 11. Core ML and the Neural Engine, without Python (measured)

The tooling this section names (`examples/coreml_export.rs` and the Swift package
`tools/coreml`) has since been removed; the measured results below stand as the
record.

Status: built for the host-fed GBC CPU checkpoint (`Model::compile(&sm83::build())`, with
6 layers, hidden 2,048, 111.9 M dense parameters and one pass over 47 tokens). The pass
reaches the Neural Engine and is exact there, with one change to how the gate matmul
runs, described below.

**Route.** Core ML is the only public route to the Neural Engine, and coremltools is
Python, so this work uses neither. Core ML compiles a `.mlmodel` at runtime
(`MLModel.compileModel(at:)`), and a `.mlmodel` is a protobuf whose schema
(coremltools' `mlmodel/format/Model.proto` and `NeuralNetwork.proto`, BSD-3) was read
but not run. `examples/coreml_export.rs` writes the protobuf with a small hand-written
encoder. The output is a NeuralNetwork model (specification version 4, exact array
mapping) with the weights inline as f16 bytes (224 MB). `tools/coreml` (a Swift package)
compiles the model, loads it with `computeUnits` set to `.cpuAndNeuralEngine` or
`.cpuOnly`, never the GPU, and holds it to the dense pass:

```sh
cargo run --release --example coreml_export -- DIR            # cpu.mlmodel, cases.bin
swift run -c release --package-path tools/coreml coreml-run DIR --units ane --plan --time 300
swift run -c release --package-path tools/coreml coreml-run DIR --units cpu
```

The graph is HF Llama, op for op. It uses `embeddingND`, and RMSNorm as `x·x`,
`reduceMean`, `RSQRT` with `epsilon = EPS`, and two multiplies. Every matmul with
weights is a `batchedMatmul`. Attention is reshape, transpose, `q·kᵀ`, scale,
`softmaxND`, `·v`, transpose and `o`, and SiLU is `sigmoid` and two multiplies. The
logits end in `argMax`. The graph leaves RoPE out because every token sits at position
0, where cos is 1 and sin is 0. The exporter asserts this.

**Where it runs.** `MLComputePlan` (macOS 14.4+) names a preferred device for 219 of the
220 layers, and the argmax's squeeze has none. Under `.cpuAndNeuralEngine` it prefers the
Neural Engine for 214 of them. Five run on the Mac CPU: the embedding (supported only on
the Mac CPU and the GPU), layer 0's first three norm ops beside it, and the argmax. This
count is per layer, not per FLOP. The embedding is a lookup, and every matmul except the
embedding runs on the Neural Engine.

**The Neural Engine's matmul overflows at 2^15.** The model as written decodes wrong
there. A probe (`coreml_export DIR --probe`, `coreml-run DIR --probe ROWS`) built from
single matmuls with chosen weights, which the plan places on the Neural Engine, shows
the limit:

| Op on the Neural Engine | Result |
| - | - |
| `1 · 30000`, `0.5 · 65408` | 30,000, 32,704 |
| `1 · 32768`, `16384 + 16384`, `3 · 12288` | inf |
| `32768·1 − 32768·0.5` (result 16,384) | inf |
| `−32512·0.5 + 32768·1` (result 16,512) | inf |
| `65408 · 2^-10` | 63.875 |
| two-input `x·xᵀ`: 181² (rounds to 32,768), 180² + 20² = 32,800 | 32,768, inf |
| elementwise `30000 + 30000`, `32704 + 32704` | 60,000, 65,408 |
| `16384 + 8`, `16384 + 24` (both halfway between f16 values) | 16,400, 16,416 |

A product or partial sum at or above 2^15 in magnitude becomes ±inf, even when the row's
sum is small. Inputs and elementwise ops keep the full f16 range. A matmul result halfway
between two f16 values rounds away from zero, where IEEE round-to-nearest-even gives
16,384 and 16,400. This checkpoint never rounds (every partial sum is on its row's grid
within 2^11 quanta), so the rounding mode does not affect it. A checkpoint whose sums
need rounding would be affected. So on the Neural Engine, T2 needs every partial sum at
most 32,752 (the largest f16 below 2^15), not 65,504. `Model::f16_acc_peaks` gives the
largest partial sum per tensor family for this checkpoint: q 32, k 128, v 4, o 0.039,
down 7.98, lm_head 1,024, and gate 65,408 (K·511). Only the gate passes 2^15. The
two-input `q·kᵀ` has no weights to bound, but every query and key reads only constant
and slot-identity channels (asserted), so all cases share one set of logits, and their
largest magnitude is 1,024.

**The fix.** The exporter runs each gate matmul on W/2 and then multiplies by 2
elementwise, before the sigmoid. Halving is exact, since gate weights are multiples of
K = 128. Every gate partial sum is then at most 32,704, and the doubled gate is at most
65,408, which elementwise f16 holds. The exporter asserts the bound for every family.
`--whole-gate` exports the gate unsplit, which reproduces the failure. This is a graph
change and not a stock Llama weight change, because SiLU is not homogeneous. Moving the
fix into the checkpoint means K = 64 (exact for f16 elementwise, inexact under f32, §6)
or a T2 legalisation bound of 32,752 instead of 65,504. Neither is done.

**Exactness (measured, M1 Max, macOS 27.0.1).** The run covers 2,411 cases. They are the
1,000 random pass-1 states of `tests/checkpoint_sst.rs`, and 1,411 passes the GBC CPU
core asked of the engine across all 500,000 SingleStepTests cases (every 400th pass).
The exporter checks the engine's tokens against the dense pass on every case. The runner
compares the argmax at all 47 positions on every case, checks that the final residual is
on the 2^-7 grid, and compares the residual and logits bit for bit with the f32 dense
pass on 64 cases.

| | Argmax, all positions | Residual off the grid | Residual, logits ≠ dense (64 cases) |
| - | - | - | - |
| Neural Engine, gate split | 2,411 / 2,411 | 0 | 0 of 6.16 M, 0 of 16.3 M |
| CPU only, gate split | 2,411 / 2,411 | 0 | 0, 0 |
| Neural Engine, `--whole-gate` | 1,128 / 2,411 (3,249 output positions wrong) | 18,424 | 6,753, 32,256 (up to 161.6) |
| CPU only, `--whole-gate` | 2,411 / 2,411 | 0 | 0, 0 |

"CPU only" in the table is the Mac CPU (`.cpuOnly`). Per-op taps (`--taps`,
`--tap-layer L`, `coreml-run --diff`) found the first divergence in layer 0's gate
matmul. It returned ±inf where the Mac CPU gave values such as 2,176 and −64,768, while
its input (`mlp_norm.out`) was bit-identical. The ANE's `mean(x²)` also differs from the
Mac CPU's on 723 entries over 20 cases, while `rsqrt` and every normalised value match,
because §2's design lets `EPS` absorb the difference.

**Time per pass (measured, 300 passes after 5 warm-up passes, one input, other
load on the machine).** The Neural Engine median is 5.08 ms (min 4.83), and the Mac CPU
only median is 39.9 ms (min 34.0). The time includes building the input array and
returning the 47 × 5,423 logits and the residual. Streaming 224 MB of f16 weights in
5 ms is about 45 GB/s, so the ANE pass is likely bound by weight bandwidth rather than
by its 10 GFLOP (inferred, not profiled). Loads took 0.1 s, except one ANE load that took
4.9 s (not reproduced). The default export takes about 24 minutes, almost all of it the
dense pass on 2,411 cases. `--random`, `--sst-stride` and `--dense` shrink it.

## 12. In stock runtimes (measured)

`tools/stock-llama` runs the f16 exports in unpatched llama.cpp b12070 on Metal (the BLAS option of that build is not recorded; `run.sh setup` passes `-DGGML_BLAS=OFF` today) with an f16
KV cache and flash attention, and in MLX 36969c09 through its C++ API. The exports
(`examples/export.rs`) have f16 matrices, f32 norm weights of `GAMMA`, `rms_norm_eps`
`EPS` and one KV head. `tools/stock-llama/run.sh` reproduced every number here. The
harness now keeps only `play`, `diff`, `scope` and `split-check` on
llama.cpp: the MLX backend, the `metal-gqa-fold.patch` and `kv-shift-rows.patch` leg, and
the `sst`, `sst-mem`, `groups`, `rows-diff` and `timing` subcommands were removed, with
their `run.sh` steps. They run at that commit, and the results below stand as measured.

The lines model's results come first (§12a). Everything after them, from "The cache
operations as built" on, is the byte form's (one cell per byte, head dim 32, digest of
the stock-f16 branch). It stays as the record it was measured on, and its operations
carry over to lines with a line in place of an address, as §4's table states.

### 12a. Lines (measured)

These runs use the harness of that date and after, on an M1 Max, with llama.cpp
b12070 (stock, and patched with `metal-gqa-fold.patch` and `kv-shift-rows.patch`), f16
KV and flash attention. Load averages are `uptime`'s at the start of each run. Another
llama.cpp `play` held the GPU during the Metal runs.

| Check | Runtime | Load | Result |
| - | - | - | - |
| CPU checkpoint, SST (1 case per opcode file, 245) | llama.cpp CPU (no GPU layers), stock | 1.25 | 0 failed; 277 passes, 7,319 tokens against the engine, 0 differ; 157 logit rows against the f32 dense pass, bit-identical |
| CPU-memory checkpoint, SST (1 per file) over the full cache of its image (11,350 line cells) | llama.cpp CPU, stock | 1.67 | 0 failed; 786 write records, every one through the runtime; 365 reader records probed after the run, 0 differ from the engine's index |
| CPU checkpoint, SST (5 per file, 1,225) | Metal, stock | 3.05 | 0 failed; 1,382 passes, 36,529 tokens against the engine, 0 differ; 373 logit rows against the dense pass, bit-identical; 14.4 ms per pass |
| CPU-memory checkpoint, SST (5 per file) over the full cache (11,350 line cells: 2,120 one per decode, 9,231 in lanes) | Metal, stock | 1.29 (after) | 0 failed; 1,233 passes (42.6 ms each), 3,875 write records (21.3 ms each), 365 probed reader records (33.8 ms each), 0 differ; 2 decodes off P; prefill 74.1 s |
| same | Metal, patched | 1.48 | 0 failed; the same counts; 49.7 ms per pass, 34.0 ms per write record; prefill 56.4 s |
| the cache after the last write, saved by each (`STOCK_SAVE_END`, 32,098,100 bytes) | stock against patched Metal | | byte-identical |

In the table, "CPU checkpoint" is the GBC CPU checkpoint and "llama.cpp CPU" is
llama.cpp's CPU backend. The probe (`sst-mem`'s last step) covers the read path SST does
not exercise, since its steps have no DMA. It reads every 64th line whole
(`dest = source`) and copies a byte of every 61st line over a WRAM, VRAM, OAM or HRAM
line, each as a 23-token reader record at P.

**Sizes (`cargo run --release --example stock`, measured).**

| | One model | CPU-memory checkpoint |
| - | - | - |
| config | hidden 5,120, intermediate 6,464, 15 layers, 22 heads, 1 KV head, head dim 64, vocab 36,299, 449 positions | hidden 2,048, intermediate 1,792, 11 layers, 6 heads, 1 KV head, vocab 8,279, 77 positions |
| dense parameters, nonzeros | 2,087.3 M, 1,485,705 | 175.3 M, 254,273 |
| KV per cached token | 7,680 B f32, 3,840 B f16 | 5,632 B f32, 2,816 B f16 |
| Yellow's cache | 141,415 line cells (+ one group of 449: 141,864), 1.09 GB f32, 0.54 GB f16 | 141,415 (+77), 0.80 GB f32, 0.40 GB f16 |
| attention per pass over Yellow's cache | 0.92 TFLOP for a 77-token group, 5.38 TFLOP for the full 449 | 0.18 TFLOP |

Against the byte form's one model (1,131,743 cells, 1,920 B f16 per token, 2.17 GB), the
lines model has 8.0 times fewer cells and a quarter of the KV bytes, at twice the bytes
per token (head dim 64). Query heads per layer fell from 52 to at most 22.

**Per-stream groups (`stock-llama groups`, `one_model::group`, measured).** A pass of a
stream decodes a group of the checkpoint's 449 positions, every token at `P`. The table
gives the census of the whole sequence by kind, and what each form of the group keeps:

| Stream | Own inputs | Other parts' inputs | Own outputs | The CPU's write slots | Other parts' other outputs | Kept, first form | Kept, same checkpoint | Kept, per-part gathers |
| - | - | - | - | - | - | - | - | - |
| PPU | 8 | 117 | 162 | 22 | 140 | 309 | 189 | 170 |
| CPU | 17 (D0, D1 late) | 108 | 60 (5 request, 22 write) | own | 264 | 185 | 98 | 77 |
| APU | 60 | 65 | 60 | 22 | 242 | 207 | 136 | 120 |
| control | 40 | 85 | 42 | 22 | 260 | 189 | 100 | 82 |

The group's first form kept every input slot, the stream's own outputs and the GBC CPU's
write slots. A stream's decoded rows need the following, which `one_model::group` reads
from the weights and asserts step by step:

- The stream's own slots.
- At layer 0, each head attends to the tokens at its largest logit, each alike, and
  every other token, the cells included, scores at least 128 below and weighs exactly 0
  (`exp(−104)` is below f32's smallest subnormal). Each head's query reads only the
  constant channel, so it is one vector at every position. No key reads a byte, so own
  inputs score alike whatever their bytes, and a decoded output byte scores 0. Group
  tokens share one position and a cell's layer-0 key is zero, so RoPE changes no score.
- Another part's slots hold byte 0 or a mask in the stream's view. Their lane channels
  are zero, so their value vectors are zero, and a gather of them writes 0 whichever of
  them it attends to. One target of such a gather is enough. Dropping all of them would
  leave the head on the tokens that share one of its code dims (128 against the
  targets' 256), which carry the stream's own bytes, or, for a record gather, on an
  average over the group and the cells.
- A gather that mixes own slots with another part's keeps all of them. `W_o` undoes the
  weight 1/len, so attending to fewer targets would scale the gathered bits.
- Later layers hold only memory heads, where every group token loses to the cell its
  query names (§4), so a token dropped there weighed 0 too.

On that checkpoint (one-model digest `c6f09388`) that gives the middle
column. The gathers run in eights over all 125 inputs, so two of them straddle parts
([24, 32) has one GBC CPU slot and seven APU slots, and [80, 88) has five APU and three
control). Each stream keeps one target of every gather of other parts' slots: PPU 16 and
one per record gather (3), GBC CPU 14, APU 9 + 3, and control 10 + 3.

**Per-part gathers with a fallback (one-model digest `c9fb64ad`).** Two compiler
changes apply only to a checkpoint of several parts (`Circuit::parts`, which `concat`
fills), so every other checkpoint's digest is unchanged:

- The gathers group each part's inputs on their own, in 1, 3, 8 and 5 groups. That is 17
  as before, so the head count stays 22.
- Every output slot's mask also keys on dim 31 (the gather pair's dim, unused on the
  shared key), which every gather head's query reads. Targets score 2·64·32/8 = 512 (the
  code dims weigh twice what they did), any mask 64·48/8 = 384, a token sharing one code
  dim 256, and a record gather's targets (the record slots' masks) 896. With the targets
  present the masks lose by 128 and weigh 0, so the whole sequence decodes as before.
  The engine's record of five frames (56,419 events, 3,072,729 decoded tokens) equals
  the previous checkpoint's, with 0 differing. Without the targets a gather attends to
  the stream's own masks, whose values are zero, and writes 0, as the byte-0 targets do.

So a stream's group is its own slots and nothing else. A pass always decodes a masked
position, so there is always a mask to attend to, and the harness asserts this on every
pass. Since then the parts share the input slots (`docs/perf-log.md`,
"Round 13"): the 90 slots are 21 groups, each held whole or not at all by every part, and
a gather serves every part that holds its group. A stream's group is still its own slots
and nothing else, and a gather of a group it does not hold falls back on its masks as
before. In a GBC CPU step's second pass the request slots hold decoded bytes, which no
gather targets and which score 0 at layer 0, so the harness leaves them out of that pass
too (the column counts the first pass).

These runs measured Yellow's 141,415 line cells, with stock b12070 on Metal on an M1 Max:

| Check | Checkpoint | Load | Result |
| - | - | - | - |
| `STOCK_GROUP=check`, 300 passes (both groups decoded, every logit row compared bit for bit) | `c6f09388`, groups 98 / 189 / 136 / 100 | 2.3 to 4.4 | CPU 211, APU 20, control 61, PPU 8 passes: 0 rows differ; 301 events equal the engine's, 0 differ |
| same | `c9fb64ad`, groups 77 / 170 / 120 / 82; cache prefilled again (1,534.6 s, 2,137 one by one and 139,279 in lanes) | 4.8 to 6.1 | the same passes, 0 rows differ; 301 events equal the new engine record, 0 differ |
| `tools/stock-llama/bench.sh`, 300 passes, median of 3 | the first form's groups (185 / 309 / 207 / 189) | 0.7 to 1.7 | 271 hours to frame 300 (CPU 681, PPU 1,081, APU 725, control 674 ms per pass) |
| same | `c6f09388`, groups 98 / 189 / 136 / 100 | 2.4 to 5.5 | **165.6 hours** (CPU 405.9, PPU 690.1, APU 475.9, control 408.1 ms); 903 events compared, 0 errors |
| same | `c9fb64ad`, groups 77 / 170 / 120 / 82 | 3.1 to 14.5 | **154.1 hours** (CPU 377.0, PPU 653.8, APU 443.0, control 373.5 ms); 903 events compared, 0 errors |
| same, with a CPU second pass leaving out its settled request slots (`check` over 600 passes: CPU 76 tokens per pass on average, 0 rows differ, 601 events equal) | `c9fb64ad` | 2.4 to 6.4 | 154.7 hours (CPU 379.1, PPU 655.2, APU 443.0, control 374.2 ms); 0 errors: the same as without, within the runs' spread |

Every bench but the baseline ran under other load (and in the second form's third run under
this checkpoint's own `tools/gate.sh`), so its hours are an upper bound. The second step
bought less than the 3.6 ms per token of §12a's timing table. It bought about 1.4 ms per
token for the GBC CPU (21 tokens, 29 ms) and about 2 for the others. The GBC CPU's group
now sits near the 66-token knee, below which a token costs about 1.5 ms. The GBC CPU's
pass, 981,204 of the 1.37 M passes to frame 300, is therefore now mostly the floor of
about 175 ms over the cache plus the matmuls.


**Timing at Yellow's cells (`timing`, patched Metal, load 1.6 to 1.8, median of 3).**
Filling the 141,415 cells took 82.2 s.

| Tokens | 8 | 66 | 185 (CPU) | 309 (PPU) | 449 (all) |
| - | - | - | - | - | - |
| ms per pass | 185 | 275 | 697 | 1,148 | 1,646 |

The byte form took 8,669 ms for the GBC CPU's 171 tokens over 1,131,743 cells (§12b).

**Yellow in the runtime (`play`, measured).** All four streams ran, every pass through
the runtime, over Yellow's full cache of 141,415 line cells. The harness diffed each
pass's decoded tokens against the engine's record (`run.sh engine`, five frames, 56,423
passes):

| Run | Groups | Prefill | ms per pass (CPU, PPU, APU, control) | Result |
| - | - | - | - | - |
| patched b12070, 300 passes, load 1.3 to 1.7 | full (449) | 1,059.6 s (2,137 one by one, 139,279 in lanes); state saved, 545 MB | 1,617, 1,626, 1,617, 1,593 | 301 events equal, 0 differ, 17,461 decoded tokens; mirror 0 of 10 checks differ |
| stock b12070, 300 passes, load 0.7 to 1.7 | own (185, 309, 207, 189) | state loaded in 0.1 s | 681, 1,081, 725, 674 | the same 301 events equal, 0 differ, 17,461 tokens |

A pass's time is the runtime's decode. The patched run's control figure of 18,971 ms per
pass counted the prefill, which ran inside its first pass. Unpatched and patched builds
decode the reduced groups at the same speed, so the five-frame run uses stock b12070.

**Only the decoded tokens and the masks attend the cells (`Runtime::split_pass`,
measured).** Every group token sits in sequence 0, which holds the cells, and in
sequence 1, which holds only two anchor cells. The anchors are the lowest and highest positions, so the coupled
sequences span the same positions, as stock llama.cpp requires. A token's attention
follows the sequence llama.cpp yields first from its sequence set, which under libc++ is
the one listed second. The decoded tokens and every mask token list 0 second, and the rest list 1 second.
Flash attention skips every block of 8 query rows by 64 cells that is masked whole.
`stock-llama split-check` shows from the weights that the decoded rows cannot change.
Layer 0's query reads only the constant channel, so every token asks layer 0 the same
query. Against it a cell's key scores at least 256 below each part's best group key in
every head that writes (104 is enough for exp to be 0 in f32), so cells weigh 0 at
layer 0 for every token. No key or value past layer 0 reads a channel that an MLP or a
later attention writes. The other tokens' cached keys and values are therefore the same
with or without the cells.

That argument holds in exact arithmetic. The runtime also needs every other token's row to
stay finite, because a key or value reads its channels through the row's norm, which sums
every channel, and `ffn_down` takes its input in f16. One unit above 65504 becomes inf
there, the norm turns the whole row into NaN, and a NaN key poisons every row that
attends it, even at weight 0. This happened on the boot-ROM branch. A mask token the pass
did not decode carries its part's mode, so the GBC CPU part's units and memory heads ran
at that token with no line cell to find. The read bus then took two columns, every bit of
`src` held 2, and `cpu:add.z` at layer 7 reached a gate of 527·128 = 67,456. Its declared
ranges bound that gate at 511·128 = 65,408. The engine and the dense pass never evaluate
that case, so the gates and the step oracle could not see it; the first non-finite tensor
was `ffn_out-7`, found with `STOCK_NANCHECK=1`. That shim callback names the first
non-finite F32 tensor, and the first `ffn_swiglu` value beyond f16 along with its
`ffn_norm` row. (`STOCK_TENSORS=<file>` writes every tensor of one decode, to compare two
backends: `docs/perf-log.md`, "The slices' NaN on Metal".) Now every mask token of a group attends the cells, read or not
(`Stream::decode`), so only input and byte tokens skip them. `split-check` adds a fourth
fact: no embedding but a mask's sets a mode channel. Every overlay memory head at such a
token therefore takes its null term and finds the null cell, which the split keeps. The
wires there hold their ranges, and `Model::f16_acc_peaks` bounds every unit.
`STOCK_SPLIT=0` alone does not leave the split path, because a scoped pass is a split
pass over the scope's sequence. Only `STOCK_SCOPE=0 STOCK_SPLIT=0` runs the plain decode,
where every token attends every cell, as a third party running the GGUF in llama.cpp
would.

After the fix, Yellow ran 4,000 passes in stock b12070 Metal with the one-model GGUF
`8af204bb`, against the engine's record. Both the scoped and split path and the plain
decode gave 4,001 events equal, 0 differing and 300,738 decoded tokens compared, across
all five streams.

That exactness holds for reduced groups only. A full group (`STOCK_GROUP=full`, every
slot of the checkpoint) holds every other part's output slot at its mask, so its part's
mode is set there. The channels at that mask still hold this stream's gathered wires,
which the other part's units read as their own. In the GBC CPU stream's full group, an
APU unit at layer 0 reached 141,312 at an APU mask, `ffn_down` read it as inf, and 2,441
of the GBC CPU's 34 passes' rows were NaN against the reduced group's. This does not
depend on the split or the scope; it also happens with `STOCK_SCOPE=0`. The overlay
argument (`Model::stream`) requires f32 sums wherever a part's units see another part's
wires, and a full group is such a place. The harness now decodes reduced groups unless
`STOCK_GROUP=full` or `check` is set. A third party decodes a stream's reduced group, the
positions `one_model::group` names.

| Run (stock b12070, `tools/stock-llama/bench.sh`, 300 passes, measured before the masks attended the cells) | Load | Cpu, Ppu, Apu, Control ms per pass | Hours to frame 300 | Check |
| - | - | - | - | - |
| own groups (baseline binary, 1 run) | 1.3 to 2.0 | 685.6, 1,093.8, 731.0, 676.2 | 272.9 | 0 errors |
| own groups, split (median of 3) | 1.0 to 2.4 | 450.5, 834.3, 460.6, 339.2 | 178.9 | 0 errors over 903 events; logit rows of all 300 passes against the baseline's: 17,488 rows, 0 differ |

**Scoped attention: a stream's rows attend only the lines its heads can name.** A memory
head gives every cell but the one its query names a weight of exactly 0 (§4). A one-bit
mismatch scores 256 below after scale, and `exp(−256)` is 0 in f16 and f32. A decoded
row therefore computes the same sums in any sequence that holds every line its heads can
name, and llama.cpp's kv_rows gathers only the cells of the batch's sequences. Each scope
is a sequence of its own (ids 2 and up, after the split's sequence 1), copied from
sequence 0 once the cache is filled or loaded. A pass puts every group token in the scope
and in sequence 1. The decoded tokens list the scope second, so they attend it, and the
rest list 1 second, as in the split pass. Each scope also holds the top cell, so its next
position is `P` like sequence 0's, and the split's two anchors are the scope's lowest
cell and the top cell.

- **Which lines.** `Circuit::reach` evaluates the merged circuit over intervals. The
  stream's own inputs are open, every other part's inputs sit at byte 0 (what the
  stream's rows hold, `one_model::group`), position wires are open, and each node's
  interval is cut to its declared range. A step is known when its form's interval lies
  wholly on one side of θ. The memory heads read fixed bytes only where nothing can
  write them, which means the cartridge's ROM lines and the register bytes written
  only at load (`MAPPER`, `MASK0`, `RAMMASK` and `BOOTING`). `BOOTING` joined them with
  the boot ROM: with it open, the fetch at a held PC could come from the boot ROM, its
  opcode is unknown, and every stream's scope is the whole cache, which `scope::CELLS`
  now refuses. Every other byte is open. A head can then name
  exactly the lines that match every address bit its query fixes, so the union of those
  cubes holds every line the stream's rows can name (`stock-llama scope <rom>` prints
  them).
- **The sets for Yellow.** The PPU stream has 2,103 lines (2,091 hold a cell): ROM line
  0, VRAM and OAM, the last HRAM line, VOID's first line, three lines from VOID's last to
  REGS, and the 17 PAL lines. The APU and control streams have 2,091 lines, the same set
  less the 12 lines between OAM and HRAM, which hold no cell, and the two streams share
  one sequence. The GBC CPU stream's own heads translate GBC CPU addresses, so the
  analysis leaves its lines open (everything) and the GBC CPU attends sequence 0.
- **What keeps a scope current.** A write evicts the line's old cell from every
  sequence (`seq_rm(-1, n, n + 1)`), decodes its record in sequence 0 as before, and
  copies the new cell into every scope whose lines hold it
  (`seq_cp(0, scope, n, n + 1)`). No sequence keeps an old line, and every scope gets its new one. The harness
  guards the two fixed sources at every write record. No write lands in ROM, and none
  changes those register bytes (`scope::Fixed::guard`).
- **Exactness, measured.** `STOCK_SCOPE=check` decodes every pass of a scoped stream
  twice, scoped and over every cell, and requires bit-identical logit rows. The bench
  greps its `GROUP ROW DIFFERS` lines. `STOCK_SCOPE=0` turns scoping off.

| Run (stock b12070 with `layer-shapes.patch`, Metal; the H1 to H3 model, `export --layer-shapes --align 64`) | Load | Cpu, Ppu, Apu, Control ms per pass | Hours to frame 300 | Check |
| - | - | - | - | - |
| H3b, the base (the figure from before these changes) | | 212.6, 384.0, 234.7, 219.6 | 86.4 | 0 errors |
| `STOCK_SCOPE=check`, 300 passes | 2.0 | (both decodes) | | every scoped pass of the 300: PPU 8, APU 20, control 60, 0 rows differ; 301 events equal the engine's, 0 differ. The first control pass ran before the cache was filled, as in every run, and was not scoped. |
| PPU, APU and control scoped, `bench.sh` (median of 3) | 2.0 to 3.0 | 212.1, 76.2, 49.2, 44.2 | **63.6** | 0 errors over 903 events |

Scoped, the PPU pass falls by 5.0 times and the APU and control passes by 4.8 and 5.0
times. At 2,091 cells a pass is mostly the dense matmuls of its group.

**The GBC CPU scoped to its mapped banks, which the model names.** The GBC CPU's heads
translate bus addresses through the banking registers, so `reach` leaves their lines
open. Four windows move with the registers (`layout::WINDOWS`: 0000-3FFF under MBC1's
mode, 4000-7FFF, A000-BFFF, and D000-DFFF with its echo F000-FDFF). `layout::translate`
maps every address of a window to the translation of the window's first byte plus the
address's offset in it. It maps every other bus address, read, written or fetched, with
OAM DMA on or off, to a fixed region (`layout::UNBANKED`: WRAM bank 0, both VRAM banks,
OAM, HRAM, FF, VOID, REGS, HOST). The test
`every_address_lands_in_a_window_or_an_unbanked_region` checks that over every bus
address under 200 random register sets of every mapper.

- **The model change.** The memory circuit gained twelve result slots, `MAP0a..c`,
  `MAPRa..c`, `MAPSa..c` and `MAPWa..c`. Each holds the physical address of a window's
  first byte, translated with the registers the step reads and with OAM DMA held off (a
  DMA-time translation lands in FF, which would be stale once DMA ends), spelled in
  three bytes like `WP0a..c`. Each is one `translate::read_address` of a constant
  address, at level 2, below the depth, so the layer count is unchanged.
  `copy_mode_inputs_keep_every_node_in_range` holds the slots to `layout::translate`
  under 300 random bankings. The cpu-memory digest changed from `eddbfd8c` to
  `ffe22414`, and the one model's from `c9fb64ad` to `b03d896c` (2,057.9 M dense, 1.49 M
  nonzeros, sequence from 449 to 461, intermediate from 6,464 to 6,336 after the fit).
  The GBC CPU checkpoint's `6bd416c1` and the others are unchanged. The GBC CPU's group
  is 89 tokens (77 + 12).
- **What the host does, exactly.** After each GBC CPU pass it reads the twelve decoded
  `MAP` tokens as four 22-bit addresses, the way it reads `WP0a..c` for an eviction.
  Before the next GBC CPU pass it puts these lines in the GBC CPU's scope sequence: the
  lines of `[address, address + window length)` for each of the four, the `UNBANKED`
  regions, and the lines `reach` gives for every head that translates no GBC CPU address.
  Those heads are the other parts' heads at their held inputs and the GBC CPU's two REGS
  heads, which name VRAM, OAM, REGS and PAL, 2,087 lines. The host moves the sequence
  (`seq_rm` of the scope, `seq_cp` of each run) only when the addresses changed. A write
  record whose line is a REGS line (from the write's decoded address, as for its
  eviction) can move a window, so it drops the addresses. The next GBC CPU pass then
  attends every cell, as before H4, and names them again. The host decodes no register
  byte and runs no mapper rule. Every byte in the scope comes from a decoded address
  token, a fixed layout region or the weights.
- **How often the fallback runs.** A reference count over Yellow's 300 frames (902,094
  steps, the survey's method) finds 2,895 steps that write a banking register (the
  first at step 80,895), so 0.3% of the 981,204 GBC CPU passes attend every cell. At a
  difference of about 150 ms a pass that adds about 0.1 hours to the projection. The
  bench's first 300 passes come before the first switch and do not see it. Since then
  such a pass decodes first for its `MAP` slots over a few fixed lines and
  no longer attends every cell (§13, "The scopes narrowed").
- **Exactness, measured.** `STOCK_SCOPE=check` also decodes each scoped GBC CPU pass over
  every cell. Because the bench window holds no bank switch, a second check runs the
  memory circuit over all 300 frames (`CircuitMemCore` in a `Session`). It holds every
  line every head queries, at every step, to the scope that the previous step's `MAP`
  slots and the REGS rule give.

Under Yellow's first windows (ROM banks 0 and 1, cartridge RAM disabled so that `MAPS`
names the FF block, and WRAM bank 1) the GBC CPU's scope holds 7,300 lines, 7,271 of
them with a cell.

| Run (B: the `MAP` checkpoint `b03d896c`, its cache prefilled again inside the GPU lock in 955.6 s, its own engine record of five frames, 56,423 events) | Load | Cpu, Ppu, Apu, Control ms per pass | Hours to frame 300 | Check |
| - | - | - | - | - |
| `STOCK_SCOPE=check`, 300 passes | 5.7 to 3.1 (`tools/gate.sh` running) | (both decodes) | | every scoped pass: CPU 210, PPU 8, APU 20, control 60, 0 rows differ; the first CPU pass, before any `MAP`, unscoped; 301 events equal the new engine record, 0 differ |
| all four scoped, `bench.sh` (median of 3) | 3.2 to 4.6 | 53.2, 76.3, 49.3, 44.0 | **20.3** (about 20.4 with the 2,895 fallback passes) | 0 errors over 903 events |
| `examples/map_audit` (since removed): the memory circuit over Yellow's 300 frames, every head's line held to the scope | | | | 902,094 steps (the reference's count), 899,198 scoped, 2,314 moves of the windows, 18,012,920 head queries, every one inside its scope; the 2,896 unscoped steps are the 2,895 after a banking write and the first |

Scoped, the GBC CPU pass falls by 4.0 times, to about the dense floor per token of the
PPU, APU and control passes.

**The overlay one model in the runtime** (one model `31307a65`,
with the null cell and the query shift of §4). Yellow ran 4,000 passes from the hand-off in
llama.cpp, with the 141,415 cells prefilled in 405.4 s, and the harness diffed every
decoded token against the engine's record. All 4,001 events were equal and 0 differed
(GBC CPU 2,863, PPU 102, APU 265, control 771; 274,942 tokens). `bench.sh` measured
GBC CPU 26.1, PPU 26.8, APU 17.7 and control 18.2 ms per pass, which gives 9.2 hours of
passes to frame 300, or about 15 with the write and reader records
(`docs/perf-log.md`, "The one model in stock llama.cpp").

**An 8 MiB cartridge** (one model `8772a661`, mooneye `emulator-only/mbc5/rom_64Mb`,
DMG). The cartridge's image holds 1,057,238 line cells (2,977 MB of f16 cache), so
`stock-llama play` asks llama.cpp for a context of the cells, the palettes' 17 lines,
and 16 positions for each prefill lane plus 1,024. The machine is an M1 Max with
`hw.memsize` 68,719,476,736 bytes. The Metal API reports a recommended working set of
55,662,788,608 bytes and a largest buffer of 41,747,087,360 (read from
`MTLDevice`, not from llama.cpp's log).

| `--lanes` | Context (cells) | Result |
| - | - | - |
| 255 (the recipe of every earlier run) | 1,062,359 | refused: the compute buffers need 152,453,283,776 bytes, and Metal fails a 70,925,238,272-byte buffer (`failed to allocate compute pp buffers`) |
| 32 | 1,058,791 | opens; prefill 6,400 lane records per 64 to 70 s (about 94 a second) |
| 64 | 1,059,303 | opens; prefill 12,800 lane records per 105 to 109 s at first (about 120 a second), slowing to 40 to 60 a second past 600,000; at 972,800 of 1,054,734 records (92%, 14,077 s), Metal's command buffers fail with `kIOGPUCommandBufferCallbackErrorOutOfMemory` and `llama_decode` returns −3. Besides the prefill, the streams decoded one APU and one control event (both equal to the engine's record) and no GBC CPU event |

The batch is 16 tokens a lane (at least 512), so 255 lanes make a 4,080-token batch
and 32 lanes the minimum of 512. That the compute buffer scales with the batch is
inferred from the three results; llama.cpp prints no buffer size when it succeeds. The 64-lane run is the one full attempt (`--max-passes 34000`, enough for the GBC
CPU events 16,837 to 27,382 where the engine's record reads ROM past 2 MiB). It ran
unlocked at `nice -n 19`, load 28 to 104, beside other work. Two other
llama.cpp runs started 8 minutes and about 1 minute before the failure, so the
out-of-memory may be contention for the GPU rather than this context alone
(inferred). No 8 MiB GBC CPU pass has run in llama.cpp on this machine. The next attempt needs a quiet GPU and
`STOCK_STATE`, which saves the prefilled cache once it completes, so that a later failure does not cost the four hours again.

**The full run on the overlay one model** (`31307a65`, measured, 2026-11-06). Yellow ran
from the hand-off to frame 300 with every pass decoded by llama.cpp b12070 on Metal. The build
carried the four patches of `docs/slice-stack.md`, but this GGUF has none of the keys
that switch on all but `layer-shapes.patch`, which only lets each layer have its own
width. The run took 63,704 s (17.7 hours) and reached frame 300 at `cfd31f3d`, the
reference hash. It decoded 913,782 GBC CPU, 83,500 PPU, 243,951 APU and 59,051 control
passes. This run's harness binary predates `STOCK_REFERENCE`, so it did not hold each
pass to the record as it decoded. After the run, `proof.sh` compared its whole record
with the engine's, stream by stream, on each event's kind, tokens, positions and decoded
output. All 1,367,705 events were equal: 981,203 GBC CPU (write and reader records
included), 83,500 PPU, 243,951 APU and 59,051 control. The log is the run's
`proof.log`; it shows no `RECORD DIFFERS` line.

**On the closure weights, bounded** (one model `47ceed72`, layer-shaped GGUF,
measured 2026-11-14). The last full 300-frame llama.cpp proof is the one above,
on an older model. The full proof of the final weights was owed after
phase 2, which changed the weights again. It has since run on `45bfb056`, as the uniform
file in unpatched llama.cpp (§13, "The full run, measured").
On these weights, two bounded runs, each pass held to the engine's record as it decoded
(`STOCK_REFERENCE`), llama.cpp b12070 on Metal:

- **Mealybug `m3_bgp_change`, the marked-line trace.** Stopped at a 100,000-pass budget:
  85,288 stepper, 10,039 GBC CPU, 3,354 control, 913 APU, 261 length and 145 PPU passes, every
  one equal to the reference, 0 differ. Per pass: stepper 76.8 ms, GBC CPU 69.6, control 78.6,
  APU 75.0, PPU 97.7, length 15.1.
- **Yellow from the hand-off toward frame 300** (`tools/stock-llama/proof.sh`), stopped on
  purpose after 20,000 passes (4,855 s with a 966 s prefill of 143,881 cells): 14,999 GBC CPU,
  3,126 control, 1,435 APU, 288 PPU and 152 length passes, all equal to the reference, 0
  differ, frame 1 at `d043ab24`. Per pass: GBC CPU 70.7 ms, APU 431, control 464, PPU 581,
  length 70.5. At these rates the 300 frames (about 1.37 M passes, most of them the APU's,
  control's and PPU's slow ones) take about 59 hours, which is why it stopped.

**The DMG OAM-bug trace** (one model `8d5b2f43`, layer-shaped GGUF, `digest --gguf` manifest
ok; measured 2026-11-17). `play` takes
`--hardware dmg|cgb|auto` now, the main binary's flag, so blargg's `oam_bug` ROMs, whose
headers claim CGB support, run on the DMG where the bug fires. The run is
`rom_singles/2-causes.gb` from its first instruction for 6 whole frames, no warmup:
`STOCK_GPU_LAYERS=99 STOCK_GROUP=own STOCK_REFERENCE=<engine record> stock-llama play
<gguf> <rom> --hardware dmg --frames 6 --lanes 255`, llama.cpp b12070 on Metal, each pass
held to the engine's record as it decoded. The engine's record of the same ROM has 134
OAM rewrite passes in 120 frames (the control stream names a row 134 times): none before
GBC CPU event 39,899, 44 by the end of frame 6, 124 by frame 13 and 10 after that.
A rewrite pass is a GBC CPU pass whose `OB` input is not 0 (the scanned row, bit 0 the
read), decoding the rows it rewrites.

- llama.cpp decoded 64,217 passes: 54,900 GBC CPU, 5,019 APU, 3,146 control, 576 PPU and 576
  length, every one equal to the reference, 0 differ. The stepper has no pass in these frames.
- 44 of the GBC CPU passes are OAM rewrite passes, counted in the run's own record by their
  `OB` token. They sit at the engine's 44 event numbers with the engine's event bytes, and
  each decodes 24 positions.
- `stock-llama diff` of the run's record against the engine's exits 0: 75,766 events equal
  (the 11,549 write records included), 0 differ, 4,920,867 decoded tokens compared. The six
  frame hashes equal the engine's (`b19699a3` for frames 1 to 3, `6872b003` for 4 to 6).
- Time, contended (the 300-frame Yellow proof ran on the same GPU throughout): 10,185 s in
  all, with a 245 s prefill of 13,833 cells. Per pass: GBC CPU 119.6 ms, APU 127.2, control
  126.1, PPU 157.5, length 60.7.

Frames 7 to 13 hold the next 80 rewrite passes and the stepper's 847 passes, about 70,000
passes in all (about 3 hours at these rates); they did not run.


**Each pass on its stream's slice** (one model `8d5b2f43`, measured
2026-11-19; `docs/perf-log.md`, "The llama.cpp pass rate since 2026-11-04"). Until here
llama.cpp ran every layer, head and FFN unit of the checkpoint for every stream's pass,
which the model's growth since `31307a65` had made 3 times as slow. With
`tools/stock-llama/stream-slices.patch` on the tree, the harness names the pass's stream
and the graph runs only what the GGUF's `gbcformer.stream.<part>.*` keys give it, as views
of the same tensors: the GBC CPU's 11 layers, 63 heads and 10,830 of 77,376 units, for
example. The weights and the file are unchanged, records and the prefill still run the
whole model, and `STOCK_STREAMS=check` holds every sliced pass's logit rows to the whole
model's bit for bit. The same change moves `argmax` to vector code, asks for logit rows
only at decoded positions, and decodes write records over the cells a prefill lane sees.
Over three interleaved rounds of 1,200 passes at load 18 to 24, the wall time per pass
falls from 89 to 36 ms and a write record from 201 to 39 ms, which projects the 300 frames
at 16.7 hours against 38.5. No full run has been made on these weights.

**Less of the file and less of each pass** (one model `8d5b2f43`, measured
2026-11-19; `docs/perf-log.md`, "Stored runs, stream widths and smaller groups"). The
layer-shaped file stored three blocks that are zero by construction, and a sliced pass
still read every column, scored every token and decoded masks nobody read. Five changes,
none of which moves a compiled weight, with `tools/stock-llama/stored-runs.patch` on the
tree after the other two:

1. **The head as a run of rows.** The file stores the rows of `output.weight` from the first
   to the last that holds a pattern, tokens 61,423 to 65,792, and llama.cpp scores only
   those. Every row left out is −1024 on channel 0, which holds the constant at every token
   and which nothing writes, so it scores exactly −1024 anywhere. A row over the run makes
   the whole vocabulary's choice whenever its largest score is above −1024, and the harness
   stops at a row where it is not (`cpp::Row::token`). At a row whose route bits are 0 or 1
   the byte 0 token alone scores at least 0. A named stream scores the run of tokens it can
   decode (`gguf::stream_head`): the 256 byte tokens for every stream but the GBC CPU, which
   keeps the families.
2. **`ffn_up` as a run of columns.** A live unit's `up` row is one 1.0 on a mode channel or
   the constant. The file stores each layer's run of whole 64-channel quanta that holds
   them, and `up` reads a contiguous copy of that run of the normed input.
3. **`token_embd` and `output` as a prefix of columns** (960 and 896 of 2,560). The token
   rows are padded with zeros to the model's width, and the head reads a contiguous copy of
   its prefix.
4. **A stream's own width.** Every row a stream's slice runs is zero past the stream's
   prefix of channels, which the export checks (`gguf::stream_width`), so the pass runs on
   that prefix: 2,176 channels for the GBC CPU, 2,048 for the APU and control. The norm
   divides by 2^7 at that width as at 2,560 (`Model::slice`'s bound).
5. **A group without the masks nobody reads.** A reduced group leaves out an own output
   position that holds its mask, is not decoded and is no layer-0 gather's target; the GBC
   CPU's group falls from 144 to 99 tokens. `Stream::reduced` states why no read row
   changes and asserts its premise for every gather of every pass.

The file stores 558.2 M values in place of 1,018.1 M. Each step was built and checked at its
own commit over 1,200 passes: `digest --gguf` expands the file and finds 0 values that
differ; the full program loads every stream's slice from it and reaches `cfd31f3d`;
`STOCK_STREAMS=check` finds 0 logit rows that differ between the slice and the whole model
on the tokens both score; every pass equals the engine's record; and the cells the write
records leave equal the ones the build before this change leaves. `STOCK_MASKS=check` holds step 5's
group to the one with every mask, 0 rows differ. Over five interleaved rounds of 1,200 passes at load 3 to 6, a GBC CPU pass falls from
31.3 to 18.7 ms, a PPU pass from 45.6 to 23.5, an APU pass from 32.4 to 18.4 and a control
pass from 21.9 to 13.1; the wall time per pass falls from 29 to 18 ms, which projects the
300 frames at 9.2 hours against 14.2. The runner's text workload is unchanged (63.3 fps
against 62.7).

Nothing in llama.cpp computes the rows and columns the file leaves out, so they rest on the
export's derivations, on `digest --gguf` and on `STOCK_REFERENCE`. A run of 18,000
passes at the tip reaches frame 1 at `d043ab24` with every pass equal to the record, 844
write records and one reader record. Not exercised: the stepper's slice, which has no pass
in these windows, and any frame past the first.

**The merged model** (one model `45bfb056`, hidden 2,176, measured 2026-11-20;
`docs/perf-log.md`, "Round 10"). The file stores 457.5 M values. On a fresh b12070 clone with
the three patches applied by `run.sh setup`, from a cache state prefilled for this model
(822 s): `STOCK_STREAMS=check` over 6,000 passes, `STOCK_MASKS=check` over 2,400,
`STOCK_GROUP=check` and `STOCK_SCOPE=check` over 1,200 each and `STOCK_WRITES=check` over
844 write records find 0 rows and 0 cells that differ, and 18,000 passes with
`STOCK_STREAMS=require` reach frame 1 at `d043ab24`, every pass equal to the record. A pass
reports whether it ran its slice (`llama_get_logits_stream`), so `require` no longer infers
it. The stepper's slice has now run: 7,855 passes of `m3_bgp_change`, each also on the
whole model, 0 rows differ. A frame of gbmicrotest `dma_basic` adds 160 reader records and
144 PPU passes. Both of those ROMs ran with `STOCK_SCOPE=0`, since `scope::CELLS` refuses
their scopes. A GBC CPU pass takes 17.4 ms, a PPU pass 25.4, an APU pass 18.1, a control
pass 11.1 and a length pass 9.2, which projects the 300 frames at 8.2 hours. Still not
exercised: any frame of Yellow past the first, a bank switch, and a bit check of a reader
record, which decodes one way only.
Since then §13 recounted the patched 300 frames at 11.6 hours, and the unpatched run of §13
decoded all 300 frames, with 2,896 GBC CPU passes unscoped at the start or after a banking
write and 2,314 moves of the windows. The patched build has still run no frame of Yellow
past the second.

### 12b. The byte form (measured)

**The cache operations as built.** Stock llama.cpp refuses a batch whose first position
in a sequence is not that sequence's largest position plus one, so every group and every
record decodes at `Y = seq_pos_max(S) + 1`. That is `P` whenever the HOST block's last
cell exists, which holds after the prefill except while that cell is rewritten, and the
harness counts every decode off `P`. Each row of the §4 table then holds with `Y` in
place of `P`. MLX takes no mask, so there a record's five tokens all sit at `Y + 1`. The
scratch tokens then also see the value token, which changes only the scratch tokens, as
the dense oracle decodes a record at one position. The value token's position and its
turn to `A` are llama.cpp's.

**Prefill.** The fixed regions go one record per decode. The HOST block's last address
goes first (the top cell, so every later decode sits at `P`), then ROM byte 0, VOID, FF,
REGS, HOST, PAL, VRAM and OAM, and then the top cell again over those cells, 17,406
records for Yellow. The other 1,114,238 cells go in lanes. Each record of a batch is its
own sequence holding the fixed cells (`seq_cp`), so with flash attention llama.cpp's
kv_rows attends each lane to its own cells only. A record sees the fixed cells, its
lane's earlier records (other addresses, which no head of the record queries) and
itself. `STOCK_STATE` saves the prefilled sequence (`llama_state_seq_save_file`) and
reloads it for the same cells.

**Exactness.**

| Check | Runtime | Result |
| - | - | - |
| CPU checkpoint, SST (100 cases per opcode file, 50,000) | Metal | 0 failed; 56,396 passes, 1,490,712 tokens against the engine, 0 differ; 3,051 logit rows against the f32 dense pass, bit-identical |
| same | MLX | 0 failed; the same 0 / 0 / bit-identical |
| CPU-memory checkpoint, SST (5 per file, 2,500) over the full cache of its image (90,793 cells) | Metal | 0 failed; 2,511 passes, 9,639 write records, every one through the runtime |

**Where a pass's time goes.** These are Metal timings on an M1 Max with f16 KV, from
`timing` (groups decoded over a cache of the given size, no other GPU load):

| Cells | 8 tokens | 66 tokens | 435 tokens |
| - | - | - | - |
| 1,024 (the weights alone) | 10.8 ms | 35.5 ms | 157 ms |
| 131,072, kv_rows on (default) | 233 ms | 476 ms | 2,293 ms |
| 131,072, kv_rows off | 104 ms | 455 ms | 2,265 ms |
| 32,768, flash attention off | 280 ms | 2,365 ms | |
| 32,768, kv_rows off | 34 ms | 136 ms | |
| MLX, 131,072 | 270 ms | 399 ms | 1,779 ms |
| MLX, 1,131,743 | 1,968 ms | 2,903 ms | 13,580 ms |

The next table is patched b12070 (`metal-gqa-fold.patch` and `kv-shift-rows.patch`) on
an M1 Max with f16 KV, at load averages of 2 to 4, median of 3 after a warm-up. The group
sizes are the reduced per-stream groups below (GBC CPU 171, PPU 295):

| Cells | 8 tokens | 66 tokens | 171 tokens | 295 tokens |
| - | - | - | - | - |
| 131,072 | 157 ms | 512 ms | 1,026 ms | 1,707 ms |
| 1,131,743 (Yellow) | 1,326 ms | 4,408 ms | 8,669 ms | 14,675 ms |

Filling Yellow's 1,131,743 cells took 3,814 s. At 8.7 s per GBC CPU pass, the engine's
five recorded frames (49,816 GBC CPU, 4,061 APU, 2,255 control and 288 PPU passes) would
take about 5.7 days, so the demonstration waits for the 8-byte lines (§7).

**Patches exact.** The GBC CPU-memory checkpoint's cache was saved after its prefill and
every write of 80 SST cases by three builds: stock, patched with only the K-shift patch,
and patched with both. Each patched cache differs from the stock one in 330 bytes, every
one the sign of an f16 zero (+0 against −0), and all 3,726 sampled logit rows are
identical.

- The host is about 1% of a pass. A `sample` of the main thread finds 98.8% in
  `waitUntilCompleted` at 8 tokens and 98.9% at 435, and the KQ mask build at 0.6% and
  0.3%. The mask is never copied (unified memory, shared buffers).
- Metal's flash-attention vec kernel runs one threadgroup per query block and query head,
  and each of the 52 heads reads the one KV head separately (`ikv2 = iq2/(ne02/ne12)` in
  `fa_vec_common.metal`), so small groups pay for streaming the cache 52 times.
- MLX's fused attention has no kernel for head dim 32, so the head dim is padded to 64
  with zeros, which is exact. At the lines model's head dim 64 nothing is padded. Its vector
  kernel refuses query length × GQA factor above 32, so the harness folds the 52 heads
  into query rows of the one KV head, which is exact because they read the same keys. At
  435 tokens it runs at about 7.2 TFLOPS counting the padded work, near the M1 Max's
  peak. At 8 tokens 416 query rows make about 13 query blocks, which leave the GPU idle
  (inferred from the scaling, not profiled).
- The work itself is 435 tokens × 52 heads × 15 layers × 1.13 M cells × 64
  multiply-adds, or 4.9·10^13 FLOP per full group, about 5 s at the M1 Max's peak.

**Per-stream groups.** A stream's group keeps every input slot, its own output slots and
the GBC CPU's write slots (`stock-llama groups`): GBC CPU 171 of 435 positions, PPU 295,
APU 193 and control 175. Every dropped token is another part's output slot holding its
mask token, whose layer-0 key is exactly zero (`groups` evaluates it, and the largest
entry is 0). So it scores 0 against every layer-0 query, which loses to every head's
targets (all kept) by 128. At every later layer its key is uncommitted and loses to the
cell by `B`. Every kept token's residual is therefore the full group's, bit for bit, and
`STOCK_GROUP=check` decodes both and compares the rows. In this form another part's
input slots stayed in the group, and §12a's groups drop them.

## 13. The two tiers in llama.cpp: the claim, and the speed-ups (measured)

This section was written for `45bfb056` and is kept as written, down to "The full run on
`ff86dde7`, measured", which is the current result. Text here and in §14 dated before
2026-11-25 says "the proof of record" and "the run of record" for the first proof, the
run on `45bfb056` in the BLAS-off build.

The one model (digest `45bfb056`) runs in llama.cpp in two ways. They hold the same
weights. Only the first is the claim.

**(a) The claim: unpatched llama.cpp.** llama.cpp b12070 (commit `00f113ec`) with no patch
loads the uniform file, `export one-model <dir> --gguf-only`: every layer at the widest
layer's shape (69 heads, FFN width 10,688, hidden 2,176), every row and column of the
embedding and the LM head present. It is 5,042,980,288 bytes and stores 2,518,195,584
values; `digest --gguf` reads it back with 0 values differing. Every pass runs the whole
model, and every decoded pass is held to the engine's record. The harness
(`tools/stock-llama`) still arranges sequences, scopes, reduced groups and the cache. It
uses only `llama.h` as b12070 ships it.

**(b) The speed-ups: patched llama.cpp.** `layer-shapes.patch`, `stream-slices.patch` and
`stored-runs.patch` let llama.cpp load the trimmed file (921,575,328 bytes, 457,490,560
values), which unpatched llama.cpp refuses at `token_embd.weight`'s shape, and run each
pass on its stream's slice of the model. Each speed-up is held bit for bit to (a) or to
the whole model: the slices by `STOCK_STREAMS=check` inside one process and by the row
hashes below across the two builds.

How a run shows which tier it is:

- The harness looks up the four functions the patches add (`llama_gbcformer_set_stream`,
  `llama_gbcformer_fence`, `llama_get_logits_stream`, `llama_get_logits_cols`) by `dlsym`
  and reads the libllama the process loaded for the five GGUF keys the patched loader
  reads. With none of them it prints `llama.cpp unpatched; whole model every pass` and the
  library's path. With all of them it says patched, and with some it says partly patched.
  `layer-shapes.patch` adds no function and no key, so it leaves no mark in the library:
  the clean tree shows its absence, and the uniform file never reaches its code.
- `STOCK_PURE=1` refuses anything but an unpatched llama.cpp over a file whose manifest
  says `unpatched (uniform layers)`. `STOCK_STREAMS=require` refuses an unpatched one, as
  it must. `proof.sh` with `PROOF_PURE=1` sets the first, and `proof-status.sh` prints the
  mode and the harness's line.
- `nm -gU` on the unpatched build's libraries lists none of the four functions, and
  `strings` finds no `gbcformer.` key. The patched libllama exports all four and holds the
  five keys.

Measured with one binary per build, M1 Max, Metal, f16 KV, flash attention,
Yellow CGB from the hand-off, reduced groups, scoped rows, the prefilled cache loaded
from a saved state, every run held to the engine's record of the same run. The load
average was 3 to 17 and the pure rates did not move with it:

| Check | Result |
|---|---|
| Unpatched, uniform file, through `proof.sh` with `PROOF_PURE=1` | 4,500 passes logged before the 15-minute GPU hold ended the run, every one equal to the record, 0 error lines |
| Rows across the builds (`STOCK_ROWHASH`, `STOCK_ROWHASH_CHECK`) | the patched build's 18,000 passes on their slices wrote 1,388,121 row hashes (frame 1 `d043ab24`); the unpatched run's rows equalled every one it reached, over 300,000, on the tokens both score |
| `STOCK_STREAMS=require` on the unpatched build | refuses |
| `STOCK_PURE=1` on the patched build | refuses |
| Unpatched build over the trimmed file | llama.cpp refuses to load it: `token_embd.weight` has the wrong shape |
| The saved cache under the unpatched build (`STOCK_STATE_CHECK=64`) | of 65 cells decoded again, 9 equal byte for byte, 56 equal but for the sign of zero key entries, 0 differ; the patched build that saved the cache gives the same 9, 56 and 0 |

So 65 evenly spaced cells of the cache the patched build saved over the trimmed file are
the cells the unpatched build decodes, up to the zero signs §12a's loans already
tolerate, and the 4,500 unpatched passes read that loaded cache and equalled the record.
The other 143,816 cells were not compared. On `dma_basic`, whose cache both builds can prefill
within a hold, every cell was: after each build's own prefill and the same 300 passes,
all 16,872 cells of the two caches are equal byte for byte (`state-diff`, §14). The unpatched pass runs here did not prefill:
a proof of record prefills in unpatched llama.cpp itself (`proof.sh` without
`PROOF_STATE`). In this round that prefill ran for 300 s and was not finished:
2,523 cells one by one and the first batch in 102 s, then 32.4 ms per lane record, which
projects to 78 minutes for Yellow's 141,326 (the patched build takes 13.7). The proof of
record ran it to its end in 83 minutes, and its saved cache has the sha1 of the patched
build's ("The full run, measured", below).

Milliseconds per pass, the first 1,200 passes, median of 3 for the unpatched build
(ranges within 0.7 ms), one run each for the patched build, interleaved:

| Stream | Tokens | Unpatched, uniform, whole model | Patched, trimmed, whole model | Patched, trimmed, slices |
|---|---:|---:|---:|---:|
| GBC CPU | 99 | 156.5 | 42.2 | 17.5 |
| PPU | 166 | 163.6 | 37.8 | 25.6 |
| APU | 139 | 124.3 | 32.3 | 18.2 |
| Control | 150 | 128.7 | 33.0 | 11.2 |
| Length | 10 | 28.8 | 9.4 | 9.4 |
| Write record | 16 | 119.1 | 29.7 | 30.0 |

**The full run, projected.** With round 10's method (`docs/perf-log.md`) and the record's counts (925,752 GBC CPU,
83,498 PPU, 251,674 APU, 71,759 control and 41,855 length passes, 69,514 write records),
the 300-frame run in unpatched llama.cpp is 55.6 hours of passes and 2.3 hours of write
records at the rates above. About 12,000 reader records and 870 unscoped GBC CPU passes
were not reached in a hold; at four times the patched build's 278 ms, the write records'
ratio, the readers add about 3.7 hours (a guess). With its own prefill the run is about
63 hours, 2.6 days, against 6.7 hours of passes for the patched slices at the same
run's rates. Every rate is from frame 1's first passes.

**The full run, projected again** (the counts above were short). The
run to frame 1 in unpatched llama.cpp (§14) timed every pass of the frame, one run, load
average 10 to 12:

| Stream | Passes timed | ms per pass | Passes to frame 300 | Hours |
|---|---:|---:|---:|---:|
| GBC CPU | 8,768 | 154.4 | 925,752 | 39.7 |
| PPU | 288 | 163.2 | 83,498 | 3.8 |
| APU | 840 | 124.0 | 251,674 | 8.7 |
| Control | 3,119 | 128.3 | 71,759 | 2.6 |
| Length | 152 | 28.2 | 41,855 | 0.3 |
| Write record | 109 | 116.2 | 119,435 | 3.9 |
| Reader record | 1 | 583 | 46,951 | 7.8 at 600 ms (inferred) |

Frame 1 is 13,167 passes, not 18,000, and it holds one reader record. The model-free
replay of the 300 frames counts what the cache decodes besides the passes: 119,435 write
records and 46,951 reader records. The 69,514 above are the record's keeps, which leave
out the write record that follows each of the GBC CPU's 46,912 reader records
(69,514 + 46,912 is the replay's 116,426 GBC CPU writes), and the "about 12,000" reader
records were a quarter of the count. Yellow's reader records start at frame 12, 160 a
frame, so no unpatched run here reached them. The one in frame 1 took 583 ms, against
278 ms in the patched build. `dma_basic` has 160 in its first 300 passes, over a cache of 16,872
cells: 95.5 ms each unpatched and 43.1 ms patched, a ratio of 2.2. The patched build
takes 284 ms over a cache of Yellow's size (160 records in a Zelda DX frame, §4), so the
unpatched rate there is about 600 ms (inferred from the ratio and the one sample).

Passes 55.0 hours, write records 3.9, reader records 7.8, the prefill 0.9 to 1.3 (98 to
110 s for the fixed cells and 22 to 32 ms a lane record: `dma_basic`'s whole prefill,
§13's first 300 s, and Yellow's first 8,160 lane records through the pure kit's own
`proof.sh`, stopped and continued once): about 68 hours, before each chunk's start and replay (2 to 5 minutes a chunk,
§14). The 2,896 GBC CPU passes that run unscoped after a banking write are counted at
the scoped rate. One was timed.

The patched run by the same counts, at §14's rates to frame 2 (18.0, 25.6, 18.6, 11.3
and 9.6 ms a pass, 25.8 ms a write record, 284 ms a reader record): 6.9 hours of passes,
0.9 of write records, 3.7 of reader records and a 14-minute prefill, 11.6 hours. Round
10's 8.1 hours counted about a quarter of the reader records.

**The full run, measured: the first proof** (2026-11-23; `docs/perf-log.md`, "The first proof", has the
whole account). The unpatched run finished. llama.cpp b12070 at commit `00f113ec…` with no
patch and a clean tree loaded the uniform file (sha1 `ce42b0b1…`, 5,042,980,288 bytes) and
ran the whole model in every pass, on Metal, from the run's own prefill.

- Frame 300 hashes to `cfd31f3d…`, the hash of the gate's one-model Yellow 300.
- 1,374,538 of 1,374,538 decoded passes equalled the engine's record as they decoded.
  After the run, every event of the run's own record equalled the reference's (995,266
  GBC CPU events, then 83,498, 251,674, 71,759 and 41,855). The log holds 0 error lines.
- The prefill took 4,961.6 s for 143,881 line cells. The saved cache has sha1
  `a8b3f911…`. The run's `STATUS` file records the same sha1 for the patched build's cache,
  and 143,848 cells equal with 0 differing under `state-diff`. So the 143,816 cells that
  this section left uncompared above are now compared, by that file's account.
- The run took 13 chunks and 12 resumes (§14) and held the GPU lock for 71 h 34 min. The
  projection was about 68 hours.

| | Count | Measured | Projected above |
|---|---:|---:|---:|
| GBC CPU pass | 925,752 | 163.7 ms | 154.4 ms |
| PPU pass | 83,498 | 164.2 ms | 163.2 ms |
| APU pass | 251,674 | 125.1 ms | 124.0 ms |
| Control pass | 71,759 | 129.9 ms | 128.3 ms |
| Length pass | 41,855 | 27.9 ms | 28.2 ms |
| Write record | 119,435 | 118.0 ms | 116.2 ms |
| Reader record | 46,951 | 587.5 ms | 600 ms (inferred) |

The cause of the GBC CPU pass's extra 9.3 ms is not traced. Three limits apply. Yellow
lends no cells in 300 frames, so the run does not exercise the loans of §4. A resume
replays the host from the engine's record (§14), so each pass was decoded and compared
once, in the chunk that ran it live. No patched full run of `45bfb056` exists, and the
11.6 hours above stay a projection.

Reader records are the lever this count shows. A reader record decodes over every cell
with no scope, 284 ms patched against 26 ms for a write record over its scope. They are
3.7 of the patched run's 11.6 hours and about 7.8 of the unpatched run's 68. Scoping
them as the write records are scoped was not built when the run of record ran. It was built later
("The scopes narrowed", below).

**What would shorten the unpatched run without a patch.**

- Sharing the prefill with tier (b) saves the prefill, 83 minutes in the run of record.
  When this was written the price was 143,816 cells the unpatched build had not decoded.
  The run of record decoded them, and its cache has the patched build's sha1 (by the run's `STATUS` file).
- Tokens are the lever inside the harness: a 10-token length pass takes 28.8 ms and a
  99-token GBC CPU pass 156.5. The groups are already reduced, and round 10 found the 14
  slots still droppable cost 0 tokens. Batching two streams' passes into one decode would
  save at most a pass's fixed cost, under the length pass's 28.8 ms (rough, not built).
- The uniform shape is where the time is. Llama's format makes every layer pay for the
  widest: FFN width 10,688 in all 25 layers where the layers use 77,376 units of the
  267,200 stored (29%), and 69 heads where they use 240 of 1,725 (14%). The FFN is 78% of
  a token's multiplies and the query and output projections 21%. A circuit scheduled to
  level the layers would move the digest. Capping the two widest layers (10,688 and
  8,704) at the next, 6,720, would cut about 29% of the multiplies, near 46 hours
  (estimate). The measured floor for any levelling is the patched build's whole
  model over the trimmed file, which is every layer at its own shape: 14.8 hours of passes
  at the table's rates, and a uniform file would still carry the whole embedding and head.
  Re-levelling the layers (`docs/perf-log.md`, "Re-levelling") made the uniform
  file is hidden 1,920, FFN width 3,584 and 41 heads over the same 25 layers, digest
  `1b581cfa`, and one decode of 99 tokens in unpatched llama.cpp on the Mac CPU backend
  takes 0.41 of the time of this section's file. With the routing units counted as written
  ("Re-levelling, round 2") the width is 3,456, the digest `7b064d51` and the ratio 0.39.
  `7b064d51` has run on Metal in unpatched llama.cpp through frame 13 of Yellow, with the
  narrowed scopes, at 42.8 ms a GBC CPU pass, and projects at about 19 hours for the 300
  frames (`docs/perf-log.md`, "Round 12"). On its stream slices in the patched build it
  gave NaN on Metal until the patch aligned its views of `ffn_down`'s columns
  (`docs/perf-log.md`, "The slices' NaN on Metal"): a GBC CPU pass is 11.4 ms there and
  the 300 frames project at about 6.2 hours. This section's patched figures are
  `45bfb056`'s.
  The next round measured what a pass costs on Metal by model shape and then made
  the parts share the input slots ("Round 13"): digest `ff86dde7`, 26 heads and 27,511
  tokens over the same hidden, width and layers. Frame 1 of Yellow in unpatched llama.cpp
  on Metal: every pass equal, 32.2 ms a GBC CPU pass against `7b064d51`'s 38.2 for the
  same passes, about 15.8 hours for the 300 frames (inferred).
  "Round 14" merged the two. `ff86dde7` through frame 13 of Yellow in
  unpatched llama.cpp on Metal: all 138,984 passes equal, 35.4 ms a GBC CPU pass, about
  15.5 hours for the 300 frames (inferred). The 300 frames then ran in the default build in
  15 h 28 min 31 s, at 34.6 ms a GBC CPU pass (measured; "The full run on `ff86dde7`,
  measured", below). On its slices in the patched build, with the
  patches unchanged: every bounded run equal, 12.0 ms a GBC CPU pass, about 6.4 hours
  (inferred).
- The LM head's 65,804 rows cost about 5% of a pass's multiplies at 77 output rows a pass
  (estimate), and the rows' copy to the host about 3 ms of the 156 (measured, the time
  outside the decode). On Metal a row cost 0.039 ms at 65,804 tokens (measured, "Round
  13, step 0"), and the vocabulary is 27,511 since.
- f16 KV and flash attention are already on, and neither can be traded. Without flash
  attention (`STOCK_FA=0`) the context asks Metal for a 160 GiB graph buffer and fails to
  open (measured). An f32 KV cache was not measured: it cannot load the f16 state, and its
  own prefill does not fit a hold. The rate is bound by the GPU: a load average of 17 left
  it unchanged.
- One uniform file per part would be narrower, but it is not the one model, and the GBC
  CPU and the PPU share one cache.

**The scopes narrowed, and the records gathered over their scope** (measured 2026-11-23; `docs/perf-log.md`, "The harness's scopes narrowed"). Only the
harness changed. No weight moved, the files are the same, and every pass is still held
to the engine's record. Five changes, each with the check that holds it:

1. **A stream's scope holds its own heads' lines.** The scoped sets above came from every
   head of the merged circuit, with every other part's inputs held at byte 0. The one
   model is an overlay since `31307a65`. At a stream's tokens every memory head of another
   part carries the null term and attends the null cell alone (`Model::stream`, the
   construction the patched slices rest on), and every scoped sequence holds the null
   cell. So a stream's rows read the cells with its own heads only, and with the two copy
   heads, which no part owns and which query the address a record gathers. At a pass's
   tokens that address is 0, so they name line 0 (`scope::own`, `scope::OWN_BESIDES`).
   For Yellow the PPU's and the stepper's scopes hold 2,086 cells, the length's 22, and
   the APU's and control's one, line 0's, where each held 2,076 to 2,091 before.
   `STOCK_OWN=0` keeps every head's lines.
2. **The GBC CPU's narrow scope.** A pass that decodes `RP0`, `RP1`, `WP0` and `WP1` (the
   physical address of each of the step's two reads and two writes) and the `MAP` slots
   attends a scope of three kinds of lines:
   - the fixed ones, 472 cells for Yellow: every `layout::UNBANKED` region but WRAM bank
     0 and VRAM (OAM, HRAM, FF, VOID, REGS, HOST, the I/O blocks and the readback
     table), the two `REGS` heads' lines, line 0 and the top cell's;
   - the lines its first heads read, which the harness puts in before the pass from the
     pass's own PC and SP input tokens and the windows the model last named. A first
     head is a GBC CPU head whose address the inputs and the banking registers give,
     whatever any other head reads: the opcode's and the two operands' columns, at PC,
     PC + 1 and PC + 2, and read 1, which reads at SP + 1 in every step
     (`scope::first_addresses`). `scope::bus_lines` gives an address's lines: its
     window's line at the address's offset, WRAM bank 0's line for C000-CFFF and its
     echo, both VRAM banks' lines for 8000-9FFF. An address in A000-BFFF gives the pass
     the windows' scope instead, because a cartridge RAM smaller than its window repeats
     in it and the offset does not give the line;
   - the lines the pass's four decoded addresses name. After the pass the harness reads
     those twelve tokens. A named line outside the scope joins it and the pass decodes
     again, since its read or write of that line found no cell. The addresses must be
     the same after the second decode, or the run stops at `NARROW SCOPE NOT HELD`.
   The scope keeps the 256 lines named most recently beyond the fixed ones
   (`STOCK_CPU_LRU`). A pass after a banking write, when no pass has named the windows
   since, first decodes over the fixed lines for its `MAP` slots alone, which read the
   banking registers' cells, then decodes as any other and must name the same windows.
   Before, such a pass attended every cell. Every other GBC CPU pass (an OAM DMA
   placement, which decodes `WD0` and `WD1` only) keeps the windows' scope.
   `STOCK_CPU_SCOPE=windows` gives every pass the windows' scope.
3. **A reader record decodes over the records' scope and the two lines it names**, its
   destination's and its source's, which join the scope for that decode
   (`Cache::read`). It decoded over every cell. `STOCK_READS=check` decodes it both ways
   and requires bit-identical rows, and `STOCK_READS=0` keeps every cell. The line a
   reader record decodes is now also held to the line the mirror gives, in every run.
4. **A write record's batch sits in its scope's sequence.** Its tokens were in sequence
   0 and in the scope, listed second. llama.cpp then gathered the cells of both sequences
   for the batch, every cell, and masked all but the scope's. The tokens now sit in the
   scope and in the split's sequence, as a scoped pass's do, so the batch gathers the
   scope's cells alone, and the new cell joins sequence 0 afterwards (`Cpp::write`).
5. **A row's choice is found a block at a time** (`cpp::argmax`). The scan for the largest
   score's place went entry by entry from the start of the row to the byte tokens near
   its end.

What the narrow scope rests on, and what holds it (the stops and the start-up sample
are round 12's, from the review of this section; `docs/perf-log.md`, "Round 12"):

- **The wiring.** `scope::named_slots_wiring` asserts on the merged circuit, at every
  start, that the GBC CPU's heads whose query names one line whatever the inputs are
  heads at the banking registers' lines (two on this circuit), that a `MAP` slot depends
  on no memory head but those, and that the four address slots depend besides only on
  first heads: heads whose own address depends on the pass's inputs and the banking
  registers alone. On this circuit those are 33 heads: the columns `OP0c`, `OP1c` and
  `OP2c`, `D1h`, and the four `OBH` heads of the DMG's OAM bug. So once a pass has
  decoded over a scope that holds the first heads' lines, its four addresses are right,
  whatever its reads and writes at those addresses found. A named line outside the
  scope is then a read or a write that found no cell, and the second decode names the
  same addresses.
- **Where the heads read: a sample of the circuit, at every start.** Which lines the
  first heads read was a table of head names written by hand. `scope::sampled_reads`
  replaces it. It evaluates the GBC CPU's own circuit on 2,000 random steps (random
  bytes in every input, a DMA line's addresses and the bus states among them, over the
  cartridge's image with random banking registers) and holds, in each step,
  every line each head reads: a fixed head to its line, a first head to the fixed
  regions or the lines `bus_lines` gives PC, PC + 1, PC + 2 and SP + 1 under the windows
  the step's own `MAP` slots name, and every other head to one of those or to a line
  the step's four address slots name. The run prints what it saw: which address explains
  each first head's reads, and how many steps were drawn. It is a sample. It proves
  nothing about a step it did not draw. On Yellow's image 1,288 of the 2,000 are held:
  the circuit refuses 266 (a declared range or an exclusive case broken), 441 have a
  first address in cartridge RAM, which takes the windows' scope, and in 5 a head names
  a line past the last region, which holds no cell in any cache (random bus states
  reach states the machine does not, and the engine refuses a pass whose head finds no
  cell).
- **What is still by hand.** `first_addresses` (PC, PC + 1, PC + 2, SP + 1) and
  `bus_lines`, which the sample holds the circuit to; which passes take the narrow
  scope (those that decode the `MAP` and the four address slots: the others, an OAM
  DMA placement or a write's second port, decode over the windows' scope and are not
  sampled); and that a step whose head names a line with no cell is no state the
  machine reaches. `bus_lines` is written from `layout::translate`; `scope`'s test
  `every_address_lands_in_its_bus_lines_or_a_small_region` holds it to
  `layout::translate_locked` and `layout::fetch_locked` at every bus address, read,
  written and fetched, with OAM DMA off, on every bus and on each alone, under each
  access lock, the boot ROM mapped or not, under 200 random register sets of every
  mapper. Its first form failed on MBC2's 512-byte RAM, which is the cartridge-RAM case
  above.
- **The windows.** A pass's lines are placed under the windows the last pass named.
  `Stream::write` forgets them at any write record to a banking register's line, and
  nothing else changes them, so a pass that decodes its `MAP` slots must name the
  windows its lines were placed under, and must name windows at all. If not the run
  stops at `NARROW SCOPE NOT HELD` (`WINDOWS SCOPE NOT HELD` for a pass over the
  windows' scope), with or without the engine's record. Before, a pass that named
  other windows only moved the next pass's lines, and a `MAP` slot that decoded no byte
  left the windows of an unconfirmed first decode in place.
- **Under the engine's record** (`STOCK_REFERENCE`, the proof) every scoped pass of
  every stream is held to the lines the record names for its heads. A line with a cell
  that the pass's scope lacks stops the run at `SCOPE MISSES LINE`. Since round 12 the
  record lists the lines the PPU's and the stepper's per-position heads read as well as
  the shared heads' (`Engine::record`); before, the check saw only the shared heads'
  lines for those streams. The APU and control have no memory head, so there is nothing
  to hold for them, and their passes are not counted as held.
- **What that check does and does not show.** It holds the harness's own account of
  each scope (`Cache.scoped`, the lines it takes a sequence to hold) to the lines the
  engine read. `play replay` runs it over the 300 frames with no model: that proves the
  bookkeeping, the order of the joins and the second decodes, and nothing about
  llama.cpp. What holds the account to llama.cpp's sequences is `Cache::audit`: it saves
  a scope's sequence through llama.cpp's own state API, reads the cells' positions back,
  and requires a cell at every line of the account, the null cell, and no other
  (`SCOPE SEQUENCE DIFFERS`). It runs at each pass `STOCK_SCOPE=check` samples.
  Besides it, every decoded token is held to the record, so a sequence that lacked a
  line a pass read would show as a wrong token wherever the lost byte matters.
- `STOCK_SCOPE=check` decodes each scoped pass again over every cell and requires
  bit-identical rows (`STOCK_CHECK_FROM=<pass>` starts it late in a run,
  `STOCK_CHECK_EVERY=<n>` samples every nth pass of the run, the form a long run can
  afford). For a narrow pass it compares the decode over the settled scope. A row that
  differs stops the run in every mode (`CHECK NOT HELD`).
- A reader record must decode the line the mirror gives, in every mode
  (`READ NOT HELD`). Before, that stopped a run only under the engine's record.
- **Without the engine's record** a head whose line the scope lacks reads whatever its
  scores pick among the cells the scope holds, and nothing in the model says so: the
  fault run below decoded token 61542 for 61423. What stands against a misplaced line
  then is the start-up sample, the windows' stop, the mirror's check of reader records
  and the sampled `STOCK_SCOPE=check`. None of them sees every pass, so the harness as
  a runtime without the record is not held pass by pass. The proof runs under the
  record.
- The deliberate faults, each to show a stop: `STOCK_FAULT_SCOPE=<k>` (a line a pass
  names leaves its scope: `SCOPE MISSES LINE` under the record, `CHECK NOT HELD` on the
  rows), `STOCK_FAULT_MAP=<k>` and `STOCK_FAULT_PROBE=<k>` (a window moved: both forms of
  `NARROW SCOPE NOT HELD`), `STOCK_FAULT_NAMED=<k>` (an address moved between a pass's two
  decodes), `STOCK_FAULT_READ=<k>` (`READ NOT HELD`), `STOCK_FAULT_SEQUENCE=<k>` (a cell
  taken out of llama.cpp's sequence behind the account: `SCOPE SEQUENCE DIFFERS`).
  `tools/stock-llama/faults.sh` runs the first five under `play replay`.

**Measured** (2026-11-24; `docs/perf-log.md`, "The harness's scopes narrowed", has every
table and the checks). Unpatched b12070 on Metal, Yellow from the hand-off, the first
1,500 passes, median of three runs a side of the kit's harness ("before") and this
branch's (`45bfb056`'s runs alternate; `1b581cfa`'s ran before, before, after, before,
after, after), every pass held to the engine's record, ms:

| | `1b581cfa` before | `1b581cfa` after | `45bfb056` before | `45bfb056` after |
|---|---:|---:|---:|---:|
| GBC CPU pass | 87.5 | 40.4 | 156.8 | 86.9 |
| PPU pass | 82.3 | 79.3 | 164.4 | 162.0 |
| APU pass | 59.1 | 44.9 | 124.9 | 103.0 |
| Control pass | 61.9 | 46.0 | 129.6 | 104.1 |
| Length pass | 16.7 | 12.8 | 29.0 | 22.1 |
| Write record | 77.6 | 34.8 | 119.8 | 52.4 |
| Reader record (40 probed, one run) | 309.2 | 22.1 | 578.8 | 43.0 |

With the counts of the proof of record these rates project the 300 frames at about 40
hours for `45bfb056`, against about 70 at the old harness's rates today and 71.6 measured,
and at about 19 hours for `1b581cfa`, against about 38 (inferred). The GPU still bounds a
pass. It runs the whole model for every token whatever the scope, so `45bfb056`'s APU
pass over one cell takes 103 ms.

What ran:

- The replay of the 300 frames with no model, over each model's engine record: every
  scoped pass of every stream held each line the record names for its heads (the APU's
  and control's records name none, and the record did not yet list the PPU's
  per-position lines: round 12 changed both, above); 913,778 GBC
  CPU passes over the narrow scope, 11,974 over the windows', 2,896 decoded first for
  the windows, 2,820 decoded twice for a named line, none over every cell.
- Frame 1 in the runtime through `proof.sh` with `PROOF_PURE=1` on both models: all
  13,167 passes equal the record, frame 1 `d043ab24`. On `1b581cfa` play was killed after
  a checkpoint and the run continued, once early and once after event 12,921, when the
  scope held named lines.
- `STOCK_SCOPE=check`, `STOCK_WRITES=check` and `STOCK_READS=check` over the first 400
  passes (`1b581cfa`) and 300 (`45bfb056`), and over the end of frame 1 from pass 12,250
  (`1b581cfa`), where passes name lines beyond the fixed ones and decode twice: 0 rows
  and 0 cells differ.
- The fault: in the replay the run stops at `SCOPE MISSES LINE`, in the runtime at
  `GROUP ROW DIFFERS` and `REFERENCE DIFFERS` on the same token.
- Zelda DX's frame 7 (`--warmup 6`), in the patched build with every write and reader
  record decoded both ways, then in unpatched llama.cpp over `45bfb056`'s uniform file:
  all 21,110 passes equal, three GBC CPU passes decoded first for the windows after a
  banking write, 626 decoded twice, 438 lines lent and put back with 0 cells differing,
  160 reader records.
- gbmicrotest `dma_basic`, 400 passes with the scopes on in unpatched llama.cpp: all
  equal, its 160 reader records and 194 write records decoded both ways with 0 rows and
  0 cells differing. A reader record takes 43.8 ms there against 95.9 over every cell.

No frame of Yellow past the first has run in the runtime with these scopes. Its 300
frames ran in the replay only, so its own bank switches and its OAM DMA from frame 12
have not decoded in llama.cpp with this harness. `find_slot`'s walk after an eviction
(3% of a pass, `docs/perf-log.md`) is not removed. The PPU's scope is not narrowed,
because its heads' lines follow the tile map's bytes and no PPU slot names them.

**The full run on `ff86dde7`, measured: the proof of record** (2026-11-25;
`docs/perf-log.md`, "The proof of record on `ff86dde7`", has the whole account). The
paragraph above is from 2026-11-24. Since then the current model has run all 300 frames
of Yellow in unpatched llama.cpp with these scopes. Its bank switches and its OAM DMA
from frame 12 decoded there: 46,951 reader records, and 2,896 GBC CPU passes decoded
first for the windows.

llama.cpp b12070 at commit `00f113ec…` with no patch, in the default build ("What runs
where", at the top), loaded the uniform file of `ff86dde7` (sha1 `2f87678b…`,
1,544,154,592 bytes: 25 layers at FFN width 3,456 and 26 heads) and ran every weight in
every pass, on Metal, from the hand-off state and the run's own prefill.

- Frame 300 hashes to `cfd31f3d…`, the hash of the engine's record and of the gate's
  one-model Yellow 300.
- 1,374,538 of 1,374,538 decoded passes equalled the engine's record as they decoded.
  After the run, every event of the run's own record equalled the reference's (995,266
  GBC CPU events, then 83,498, 251,674, 71,759 and 41,855). The log holds 0 error lines.
- The prefill took 1,757.6 s. The saved cache has sha1 `aaa64e15…`, the sha1 of the
  cache the BLAS-off build saved over the same file (by the run's `STATUS` file).
- The run took 3 chunks and 2 resumes (§14) and held the GPU lock for 15 h 28 min 31 s.
  The projection was 15.5 hours (`docs/perf-log.md`, "Round 14").

| | Count | Measured | Projected from frame 13 | The first proof |
|---|---:|---:|---:|---:|
| GBC CPU pass | 925,752 | 34.6 ms | 35.4 ms | 163.7 ms |
| PPU pass | 83,498 | 59.3 ms | 59.2 ms | 164.2 ms |
| APU pass | 251,674 | 36.6 ms | 37.4 ms | 125.1 ms |
| Control pass | 71,759 | 37.4 ms | 37.1 ms | 129.9 ms |
| Length pass | 41,855 | 9.6 ms | 9.9 ms | 27.9 ms |
| Write record | 119,435 | 21.6 ms | 21.3 ms | 118.0 ms |
| Reader record | 46,951 | 26.2 ms | 26.2 ms | 587.5 ms |

The limits are the first proof's, and one more. Yellow lends no cells and the stepper
has no pass, so the run exercises neither the loans of §4 nor the stepper's stream. A
resume replays the host from the engine's record (§14), so each pass was decoded and
compared once. The run starts at the hand-off, not at power-on. The added limit is the
scopes. A pass
attends its stream's scope, and no pass or record had a second decode over every cell in
this run, so the scopes are held by the record comparison, by the check that a scope
holds every line the record names, and by the bounded `check` runs listed above. No
patched full run of `ff86dde7` exists, and its 6.4 hours stay a projection.

The cell counts in this file (143,881 line cells, 143,848 cells under `state-diff`, and
the "143,816 cells not compared", which is 143,881 less 65) describe one cache of 143,847
distinct lines and the null cell. The harness's 143,881 counts 17 palette lines three
times (`docs/perf-log.md`, "The cell counts, reconciled").

## 14. The run in chunks: checkpoints and resume (measured)

The unpatched run was projected at about 68 hours and took 71 h 34 min with the GPU lock
(§13). `play` can stop and continue, so the proof
runs in chunks, and a chunk that is stopped or killed loses at most the time since its
last checkpoint.

**What a checkpoint holds.** `STOCK_CHECKPOINT=<dir>` turns checkpoints on. An event is
one call of an engine into the harness: a pass, a write record or a reader record. After
an event, when a checkpoint is due, the harness writes two files:

- the runtime's cache, as `llama_state_seq_save_file` saves sequence 0: every cell's
  position, key rows and value rows (922,354,012 bytes for Yellow);
- a manifest: the event's number, the counts and timings so far, what the run is bound
  to, and `held`, which is what the harness holds that does not depend on the runtime or
  the clock. `held` is each stream's passes, tokens, writes and reader records, its place
  in the engine's record with a digest of every line read, a digest of its own lines,
  the mirror's digest, the GBC CPU's scopes (the windows it holds, and the narrow
  scope's named lines and counts), and the length and digest of every record file the
  engines have written.

A checkpoint does not hold the engines or the machine, and the main crate did not change.

The harness takes a checkpoint only when the cache is quiet (`Cache::quiet`): no cell
waits for its shift and no cell put back waits for its comparison. llama.cpp saves a
cell's new position with its key rows as they were before the shift, and only a write
record leaves a shift waiting, so the harness waits for an event that ends in a pass. A run with
checkpoints therefore decodes exactly what a run without them decodes. In Yellow every
pass ends quiet. In a game that lends lines the checkpoint waits for the next pass that
lends none.

**How a run resumes.** A run that finds a checkpoint in the directory does not load the
engines' state. It runs the host again from frame 0 over `runtime::Resuming`: no model
runs, and each pass decodes the tokens of the engine's record (`STOCK_REFERENCE`,
required). When the replay reaches the checkpoint's event, the harness computes `held`
again and compares it with the manifest's, value by value. Any difference stops the run
at `RESUME DIFFERS`. Then the runtime loads the saved cache, the harness copies the
scoped sequences again from it (a scoped sequence is a set of sequence 0's cells by
position), the counts carry on from the manifest's, and llama.cpp decodes every later
pass. The log labels the replayed frames `REPLAYED from the engine's record, no model
ran`, as `--warmup` labels reference frames. Those passes were decoded and held to the
record in the chunk that first ran them. The checkpoint's counts keep that, and the
record files the replay writes again must match the digest the checkpoint keeps of them.

**What a checkpoint is bound to.** The manifest names the model file (its manifest's
tensor and wiring digests and its size), the harness executable's SHA-1, the ROM's
SHA-1, the libllama the process loaded (path, patched or unpatched, SHA-1), the engine's
record (each file's size) and every setting that changes what the harness decodes. A
run under any other value stops at `RESUME REFUSED` and names the field.

**How the files are written.** `Store::commit` writes the state to a temporary file,
syncs it and renames it. Then it writes the manifest the same way. The manifest names
the state with its size and digest, and its rename is the commit. The two newest
checkpoints stay. A state without a manifest is ignored, and a state that does not match
its manifest is passed over for the checkpoint before it.

**The prefill.** The prefill runs inside the first event, so it has checkpoints of its
own (`checkpoint::Prefill`). The harness splits a lane batch into its decode and the
move of its cells to their lines. Between the two no cell waits for a shift, and there
it saves every sequence of the context with `llama_state_save_file`: the cache so far
and the open lanes. A run that finds that checkpoint loads it, moves the batch's cells
and decodes the batches after it. The first checkpoint after an event removes it. The
2,523 fixed cells that decode one by one before the lanes open have no checkpoint. They
take 75 s in the patched build and 98 s unpatched.

**Stopping.** A run ends at a checkpoint, with exit 0, on SIGTERM or SIGINT, when the
file `STOCK_STOP` names exists, after `STOCK_CHUNK_SECS` seconds, or at `--max-passes`.
It takes the checkpoint at the next quiet event, not at the next timer. The harness
takes no checkpoint at a run's end: by then the engines have written the last event's
record line, which a replay has not when it reaches that event, so a resume of it
stops at `RESUME DIFFERS` (measured, before the harness stopped taking it). A finished
run that is started again continues from its last checkpoint inside the run and decodes
the rest again.

`proof.sh` wraps this. Running the same command again continues the run. `touch
PROOF_WORK/stop`, TERM or INT to the script, or `PROOF_CHUNK_SECS` end a chunk at a
checkpoint, and the GPU lock is released between chunks. `PROOF_FRESH=1` drops the
checkpoints. A chunk that dies with no failed check leaves no exit code, and the next
run continues from the last checkpoint. A failed check in any chunk fails the proof. A
chunk that is refused (`RESUME REFUSED`), whose state does not load, or whose checkpoint
is not written fails no check: it leaves no exit code, and the right command continues.
`proof-status.sh` prints the chunks, the time over all of them, the last checkpoint and
the ETA.

**What a checkpoint costs.** Measured on the patched build, M1 Max, load average 8 to 18.
One checkpoint of Yellow's cache is 922,354,012 bytes and takes 0.5 to 1.2 s: 0.2 to
1.0 s for llama.cpp to write the state, 0.2 s for its digest, and the rest for the
digest of the record lines written since the last one. A run to frame 2 with a
checkpoint every 60 s spent 11.5 s of 1,689 s in 25 checkpoints (0.68%). `proof.sh` saves
every 300 s, which is about 0.3% and loses at most five minutes to a kill. A prefill
checkpoint grows with the prefill, from 45 MB to 800 MB, and takes 0.1 to 0.9 s in the
patched build. In unpatched llama.cpp an event checkpoint takes 0.5 to 1.4 s (11 of
them, 0.38% of the run to frame 1 at a 240 s interval), but a prefill checkpoint takes
8.3 to 8.9 s whatever its size: 5 of them on `dma_basic` and 6 on Yellow, 45 to 132 MB.
The cause is not found, and none was taken late in Yellow's prefill, where the state is
800 MB. At 8.4 s and the 300 s default that is 2.8% of the prefill (inferred for its
later part) and under 0.1% of the run.

The proof of record measured these costs in unpatched llama.cpp (2026-11-23). Its 16 prefill checkpoints
took 8.3 s at 71 MB and rose to 10.4 s at 946 MB, 146.6 s of the prefill's 4,961.6 s
(3.0%). So the time does grow with the size, slowly, and the cause of the 8 s floor is
still not found. Its 840 event checkpoints took 613.0 s. All 856 took 759.6 s of
257,502 s, 0.29% of the run.

A resume costs the start of a process and the replay. At 16,227 passes the run was live
again 103 s after it started: the model's compile, the replayed prefill (24 s) and the
replayed passes. The model-free replay of all 300 frames took 160 s when round 10 ran it
(`play replay`), so a resume late in the run should cost about five minutes with the
digest of the 8.8 GB of record files (inferred, not run). The proof of record's 12 resumes
were live again after 118 s (at event 110,424, frame 9) to 299 s (at event 1,491,513,
frame 278), 2,468 s in all (measured).

**Evidence that a resume changes nothing** (measured; patched build with slices unless
the row says unpatched; every run under `STOCK_REFERENCE`, 0 error lines in each but the
two that must fail). The rows ran three earlier versions of the harness. The current harness differs from those in two ways: the frames done
left `held`, and the checkpoint at a run's end is gone. The "last caches" below are
those end checkpoints. The kits' dry runs on the tip's binaries follow the table.

| Runs compared | Result |
|---|---|
| Yellow to 3,000 passes: stopped at 1,500 and resumed, against straight | records byte for byte equal (3,001 events, 234,911 decoded tokens); `held` equal; the two caches' files have one digest; `state-diff`: 143,848 cells equal byte for byte |
| Yellow to frame 2 through `proof.sh`: five chunks with its own prefill, against one chunk over a cache prefilled in one piece | both exit 0 and reach the record tail; frames 1 and 2 `d043ab24` in both, the engine's hash; 22,792 passes equal the record in each; records byte for byte equal (24,340 events, 1,757,727 decoded tokens); `held` equal; last caches: 143,803 cells equal byte for byte, 45 equal but for the sign of zero key entries, 0 differ |
| The five chunks' ends | the chunk's seconds and the stop file (both inside the prefill, at 56,865 and 117,810 of 141,326 lane records), TERM to the script (after event 13,510), a KILL of `play` (the next chunk continued from event 16,802, the last checkpoint), the last frame |
| That prefill in three pieces against the prefill in one piece (round 10's saved cache) | 143,848 cells equal byte for byte |
| `dma_basic`: the prefill stopped by the stop file and continued, against one piece | records equal; `held` equal; 16,872 cells equal byte for byte |
| Yellow's checkpoint offered to a run of Zelda DX under other settings | `RESUME REFUSED` after 9 s, exit 1; it names the ROM, the record and the settings |
| A checkpoint offered to another build of the harness, all else the same | `RESUME REFUSED` after 14 s, exit 1; it names the harness alone |
| A checkpoint with one `held` value changed (a stream's digest of the record lines read), offered to the run that wrote it (harness at the tip) | `RESUME DIFFERS` when the replay reaches the event, exit 1; it names that one value |
| Zelda DX on CGB, scoped, `--warmup 6`, frame 7, in four chunks: two inside its own prefill, two inside the frame (after events 5,827 and 12,628) | frame 7 `d043ab24`, the engine's hash; all 21,110 passes equal the record; 216 PPU passes lent 438 lines, 28 of them one line, 14 loans carried; 424 cells put back compared, 0 differ |
| Unpatched llama.cpp, the uniform file, `PROOF_PURE=1`: Yellow to frame 1 through `proof.sh` in four chunks of at most 691 s, over the cache the patched build prefilled | exit 0 and the record tail; frame 1 `d043ab24`, the engine's hash; all 13,167 passes equal the record; three stops (the chunk's seconds) and three resumes, after events 4,149, 8,221 and 12,273; 38 min 46 s with the GPU lock, 0.38% of the run in 11 checkpoints |
| Unpatched llama.cpp, `dma_basic` unscoped: its own prefill stopped at 13,770 of 14,350 lane records and continued, then 300 passes; against the patched build's run of the same | all 300 passes equal the record; the two builds' records byte for byte equal; the two builds' caches after those passes: 16,872 cells equal byte for byte |

**The kits' dry runs on the current binaries** (measured):

- Patched, slices, Yellow to frame 1 through `proof.sh` in two chunks of 200 s over a
  saved cache: exit 0, the record tail, frame 1 `d043ab24`, 13,167 passes equal the
  record. A chunk started under a stray `STOCK_SWAP=0` was refused, left no exit code
  and did not fail the proof (measured one commit earlier). The finished run, started
  again as after a kill in the record tail, continued from its last checkpoint (event
  11,520), decoded the rest again and finished with exit 0.
- Unpatched, the uniform file, `PROOF_PURE=1`, Yellow to frame 1 through `proof.sh` in
  four chunks of at most 691 s over the cache the patched build prefilled: exit 0, the
  record tail, frame 1 `d043ab24`, 13,167 passes equal the record, three stops and three
  resumes, 38 min 37 s with the GPU lock, 0.31% of the run in 10 checkpoints. The same
  run on an earlier harness is the row in the table.

The pure kit's own start was run too, in a scratch directory: `PROOF_PURE=1` with no
saved cache through the kit's copy of `proof.sh`. The unpatched prefill of Yellow began,
stopped at 4,080 of 141,326 lane records, continued and stopped again at 8,160. No
unpatched prefill of Yellow had run to its end then. The proof of record's did, in one
chunk (§13).

**The first proof in chunks** (measured, 2026-11-23). The unpatched 300-frame run took
13 chunks. Eleven ended at `PROOF_CHUNK_SECS` (21,600 s), one at a stop request (a pause
of 66 minutes), and the last at frame 300. Each of the 12 resumes
began at a checkpoint that a live chunk saved, no resume printed `RESUME DIFFERS` or
`RESUME REFUSED`, and `proof-status.sh` reports the live events from event 0 to the log's
end with no gap. No chunk was killed, so this run did not exercise a resume after a kill.
Every loan counter is 0, so no checkpoint waited for a pass that lends none.

**The proof of record in chunks** (measured, 2026-11-25). The run on `ff86dde7` took 3
chunks. Two ended at `PROOF_CHUNK_SECS` (21,600 s) and the last at frame 300. The two
resumes reached their first live event in 151 s and 225 s, each at a checkpoint that a
live chunk saved, with no `RESUME DIFFERS` or `RESUME REFUSED` and no gap in the live
events. The 184 checkpoints took 152.3 s of 55,457 s (0.27%). No chunk was killed, and
every loan counter is 0, so this run exercised neither a resume after a kill nor a
checkpoint that waits for a loan.

Both of the kits' dry runs above ran `proof.sh` one commit before its last change, which only replaces a
`head -n 0` that BSD `head` rejects for a stream with no events (the comparison was of
two empty inputs either way). The changed loop was then run on the patched dry run's
records: the same six lines, and `RECORD DIFFERS` once one decoded token was changed.

The 45 cells come from a known difference, not from the resume. A cache saved at
the prefill's end (`STOCK_STATE`) holds its last lane batch before the shift, and the run
that loads it never applies that shift. The run that prefills in its own process does.
The shift changes only the sign of zero key entries: `STOCK_SHIFT_PROBE=1` read 2,720
written cells before and after their shift (`dma_basic`'s prefill and frame 1, and three
of Yellow's). In 2,719 the saved bytes differed only in such signs. The other is line 0,
whose saved bytes hold two cells, its own and the null cell, which the sign test does
not parse (found with a build that names the line, which is not committed). A zero of
either sign adds the same to every score (§12a).

**What this does not cover.** The engines and the machine are rebuilt by replay, so a
resume rests on the host being deterministic over the same record. The run checks that
at the checkpoint's event (`held`) and not before it. The passes before a checkpoint are
held to the record once, in the chunk that decoded them. A later chunk trusts the
checkpoint's counts for them. Row hashes (`STOCK_ROWHASH`) cover only the passes a
process decodes itself. The 2,523 fixed cells of the prefill decode in one stretch.
