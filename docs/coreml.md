# The one model on Core ML (measured)

This is an evaluation of Apple's Core ML as a third runtime for the one-model checkpoint,
on the GPU and on the Neural Engine (ANE), after llama.cpp (`docs/slice-stack.md`) and
MLX (`docs/mlx.md`). It runs the GGUF llama.cpp runs (H11,
`tools/stock-llama/.build-heads/one-model/model.gguf`), from llama.cpp's saved cache. Its
logits are checked against llama.cpp's bit for bit and its tokens against the engine's
record, and its speed is measured with the same bench. The verdict is in §6.

This evaluation ran on the one model as it stood before the overlay
(`docs/hardware-shaped.md` §15), exported with the slice stack of `docs/slice-stack.md`. Its
backend (`coreml.rs`, `mil.rs`, `coreml_shim.m`, the shared `gguf.rs`, `STOCK_BACKEND=coreml` and
`coreml-bench.sh`) was not carried into this tree. `examples/f16peaks.rs`, the last of the
tooling, has since been removed. In this tree
`tests/stream_audit.rs` runs `Model::check_f16` and `Model::check_f16_acc` on the one model,
and `Model::f16_acc_peaks` still returns the peaks the example printed.
The measurements stand as the record of that model.

No Python anywhere: the ML Programs are written by hand from Rust (`mil.rs`, the
protobuf and weight-blob formats of coremltools read, not run), and Core ML is driven
through its Objective-C API (`coreml_shim.m`).

## 1. Feasibility, from the numbers

**The Neural Engine.** It multiplies in f16 and accumulates in f16, and a measured probe
(`docs/stock-llama.md` §11) showed its matmul turns any product or partial sum at or
above 2^15 into ±inf, where IEEE f16 holds up to 65,504. §10 of `docs/stock-llama.md`
already makes every partial sum of every row an exact f16 (T2), so the question is only
which families pass 2^15. `examples/f16peaks.rs` ran
`Model::f16_acc_peaks` on the one model (2,057.9 M dense, the checkpoint the GGUF was
exported from; every row passes, no failure):

| family | largest partial sum (any order) | below 2^15 |
| - | - | - |
| q | 3,072 | yes |
| k | 3,488 | yes |
| v | 4 | yes |
| o | 0.0703 | yes |
| gate | **65,408** (K · 511) | **no** |
| down | 15.15 | yes |
| lm_head | 1,024 | yes |

Only the gate breaks the limit, the value that breaks it is 65,408, and it breaks it by
a factor below 2. The fix is the one the GBC CPU checkpoint used (§11 there): run the gate on
`W/2` and double the result elementwise. Gate weights are multiples of K = 128, so `W/2`
is exact, every halved partial sum is the exact half of the original (at most 32,704 <
32,752, the largest f16 below 2^15), and doubling is an exponent shift (elementwise ops
keep f16's full range). The two activation products have no weights to bound: the memory
heads' `q·kᵀ` is ±1,024 per address bit against ±1 keys, 19 bits plus `committed` at 2,048,
so at most 21,504; layer 0's codes are smaller. The norm, SiLU and softmax arguments of §10
there hold in f16 elementwise, provided the ANE's elementwise ops round as IEEE f16 does. So
**the matmuls of a rescaled export are exact on the ANE by construction**, with one graph
rewrite (the gate split) and no weight change. The elementwise ops are not quite: the
ANE's norm is one ulp low on one channel (§3), so ANE exactness is measured (§5), not
proved.

What the arithmetic does not decide, and was measured (§3, §4): whether the ANE compiles
and keeps the programs (it does with limits), whether its whole-state softmax over every
line loads (it does not: error −14 at 141,440 cells, as on coreml-state2), and its speed.

**The GPU.** By Core ML's documentation the GPU accumulates matmuls in f32 unless
`allowLowPrecisionAccumulationOnGPU` is set (it is not); not verified here. The result
does not depend on it: with the gate split every partial sum is exact in f16 too. One Core ML GPU kernel computes a value product wrong (§3), and
the export works around it.

**State and shapes.** Core ML's stateful models (`MLState`, macOS 15) keep the cache on the
device between predictions, and the host writes rows into a state in place
(`getMultiArrayForStateNamed`). The ANE needs fixed shapes, so every package has fixed
rows and cells, padded (§2). Scoped attention (H4) needs only that a state holds a
scope's cells, which the host fills.

**Earlier attempts.** coreml (merged) ran the host-fed GBC CPU checkpoint bit-exact on the ANE
and found the 2^15 limit. coreml-state and coreml-state2 wrote stateful ML Programs for
the byte-form memory checkpoint and ran two functions over one state exactly on the
ANE; they stopped at the then one model's size (one cell per byte, 1.13 M cells) and at
the ANE refusing a softmax over the whole state (−14), before lines and scopes existed.
Their `mil.rs` writer is the base of this one.

## 2. What was built

`tools/stock-llama` gains a Core ML backend behind the harness's `Runtime` trait,
selected by `STOCK_BACKEND=coreml`, beside llama.cpp and MLX (`docs/mlx.md`). `play`, `bench.sh`, `diff` and `rows-diff` run unchanged on it (`rows-diff` has since been removed from the harness).

- `src/gguf.rs`: the GGUF reader and llama.cpp's saved-state reader, moved out of `mlx.rs`
  so both backends share them.
- `src/mil.rs`: coreml-state2's ML Program writer (spec 9, opset CoreML8, `.mlpackage`),
  with raw-bits weights, identifier-safe names, and sparse weights
  (`constexpr_sparse_to_dense`: the nonzero f16 values and a UInt1 mask, BlobDataType 9,
  MIL UINT1 = 37, from coremltools' `BlobDataType.hpp` and `MIL.proto`).
- `src/coreml.rs`, `stock-llama coreml-export <gguf> <dir>`: one package per part
  (`ppu`, `cpu`, `apu`, `ctl`), plus `cpu-full` and `record`, from the GGUF's own tensors.
  Each pass package is LLAMA as llama.cpp runs it with the slice stack: the layer's own
  head count, one KV head, the part's FFN slice with no up (`silu(gate)` times the 0/1
  claim, before `ffn_down`), the norm as `x / sqrt(mean(x²) + eps) · γ`, the part's run
  of LM-head rows (H11). Fixed shapes: rows 176, 96, 128, 88 (groups 170, 89, 120, 82);
  cells 2,112, 7,424, 2,112, 2,112 (scopes of 2,091 to 7,300 held lines). Inputs: the
  rows' token embeddings (`token_embd` rows, gathered by the host as llama.cpp's
  `get_rows` does), the claim, and one bias row: 0, or −16,384 on a pad cell or a pad row.
  States `kc`, `vc` `[15, C, 64]` f16. Output: the logits of every row.
  - Pad rows repeat the first row's token, so they compute finite values, and no key
    reads them (their bias column).
  - `COREML_SPARSE=1` stores every matrix sparse (§4); `COREML_MULTI=1` writes the four
    parts as functions of one package.
- `stock-llama coreml-time <dir> a,b,...` times parts in turn over zero states,
  `coreml-census` counts live rows and columns, `coreml-cmp` holds two compute units'
  intermediate taps (`COREML_TAPS` at export) to each other.
- The runtime (`CoreMl`): the host keeps the cache, each held line's key and value rows,
  loaded from llama.cpp's state file. A pass fills its package's state with its scope's
  held lines when they changed (a rescope), and a write pushes the new rows into every
  state holding the line. Every row of a pass attends the scope's cells and the group.
  The GBC CPU's unscoped passes (before the model first names its windows) run `cpu-full`
  (141,440 cells) on the GPU, since the ANE refuses it (§1). `COREML_UNITS` picks `ane`,
  `gpu` or `cpu`, for all packages or per part (`cpu:ane,ctl:gpu`).
- Each pass package gets one prediction at load, whose time no pass counts: a package's
  first prediction can carry the end of its load.

## 3. Why the same logits, and what differs from llama.cpp

§10 of `docs/stock-llama.md` makes every product and partial sum an integer f16 holds, in
any order, and §1 above extends it to the ANE's 2^15. What this graph does differently
from llama.cpp, and why each is exact (each premise is asserted by the exporter on the
GGUF's weights before it writes, `premises`):

- **No RoPE.** Every token of a pass sits at one position, and every query and key row of
  every RoPE pair below 10 is zero (asserted). Pairs 10 and up turn by at most 1.3·10^-12
  rad per position, so across the at most 285,841 positions a key and a query differ by
  below 4·10^-7 rad: llama.cpp's rotated entries round to the unrotated f16 values or add
  terms far below every logit gap (128 and up), and keys that tie in llama.cpp
  (bit-identical group keys at one position) tie here. The cells' keys are llama.cpp's
  own rotated bytes.
- **Every row attends the scope.** llama.cpp's split pass lets only the rows read attend
  the cells; here every row does, as the unsplit pass did. The rows read are the same
  either way: no key or value past layer 0 reads a channel that any `ffn_down` or any
  later `attn_output` writes (asserted: none of the 4,286 such channels), and layer 0
  gives every cell weight 0 (`stock-llama split-check`), so every row's later keys and
  values are the same whatever it attends. Attending the cells also keeps every row's
  wires in their declared ranges, which matters under f16 accumulation: a row that
  attends the wrong cells could pass 2^15 in its gate and poison every row through its
  key.
- **Writes.** A write record's cell key and value at every layer depend only on the
  record's embeddings and its layer-0 attention (the same assertion), so `record` runs
  layer 0's attention over the 16 record tokens (the scratch tokens do not see the cell
  token, as at positions Y and Y + 1) and then each layer's `attn_k`, `attn_v` on the cell
  token's layer-1 residual, normalised by each layer's own norm. Its `rms` is exactly 2^7
  both there and at the layer itself. The host stores the raw key: by the RoPE premise it
  scores as llama.cpp's K-shifted one does.
- **The GPU's value product.** A single matmul of the softmax weights over the
  concatenated cell and group values comes back wrong on Core ML's GPU (measured with
  `coreml-cmp` on the first control pass: scores and weights equal to the ANE's, the
  product 0 where the ANE gives 0.5 on 1,371 of 112,640 values, a different set each
  run). The export computes it as two matmuls, cells then group, and adds them, which is
  the same sum in another order; the GPU then matches the ANE on every tap.
- **Padding.** Pad cells hold zero rows with bias −16,384 and pad rows are keys with bias
  −16,384: `exp(−16,384 − m)` is 0 in f16 and f32, so neither changes a softmax.

**The ANE's norm is not exact on every channel.** On the ANE, every layer's norm (layers 0 and
7 tapped) returns 63.96875 on channel 1 of every row where the exact value, and the GPU's, is
64 (`coreml-cmp`, the two dumped control passes, the export as built). So the ANE's
`x / rms · γ` is one ulp low there, which contradicts coreml-state2's finding that
`real_div` is exact on both devices. It reaches nothing tapped: layer 0's gate, its
residual after the FFN, layer 14's residual and the logits are bit-identical on the ANE and
the GPU (the activations differ only in the sign of zeros). Whether any matrix reads channel 1
was not checked, so the error is bounded only by these taps and by §5's 300 passes.

## 4. What limits the Neural Engine

Timed with `coreml-time` (zero states, fixed shapes, median of 4 to 20 after a warm-up;
the ANE is not the GPU, so the proof run does not contend for it, but other
Mac CPU load ran throughout).

| packages, run in turn | ppu (176 × 2,112) | cpu (96 × 7,424) | apu (128 × 2,112) | ctl (88 × 2,112) |
| - | - | - | - | - |
| dense, each alone | 58.4 | 58.4 | 32.1 | 25.3 |
| dense, three of the four (3 of the 4 triples) or two (4 pairs) | 58.0 to 58.4 | 58.6 to 59.3 | 32.2 to 32.7 | 25.4 to 25.8 |
| dense, all four | 92.5 (max 617) | 80.4 (max 609) | 577.1 | 45.7 (max 561) |
| dense, the four as functions of one package (2.2 GB) | 85.9 | 68.8 | 48.3 | 34.5 |
| **sparse, all four** (233 MB in all) | **46.5** | **51.8** | **25.4** | **19.5** |
| Core ML, Mac CPU only, dense | | 63.8 | | |
| `cpu-full` (96 × 141,440): Mac CPU only 702, GPU 129 (contended), ANE refused (−14) | | | | |

Every op of every pass package is placed on the ANE (`COREML_PLAN=1`, `MLComputePlan`).

- **The ANE keeps three of the four dense programs.** All four (2.85 GB of f16 weights)
  thrash: the package that runs after the others reloads, 560 to 620 ms. Any three
  (up to 2.35 GB) stay resident. One multifunction package shares the attention weights
  (2.2 GB on disk) and does not reload, but every pass is 10 to 28 ms slower.
- **Dense, it streams weights at about 20 GB/s.** Per pass, ctl 0.53 GB in 25.3 ms, apu
  0.67 GB in 32.1 ms, ppu 1.07 GB in 58.4 ms: 47 to 55 ms per GB whatever the rows, and
  1.9 to 3.2 TFLOP/s, rising with the rows (inferred: bandwidth-bound, not profiled). The
  GBC CPU pass is 23 ms above its weights' share, its 7,424 cells and 4,176-row head.
- **Sparse weights fix both.** The weights are 0.07% nonzero; stored as nonzeros plus a
  1-bit mask the four packages are 233 MB, all four stay resident, and each pass is 11
  to 23% faster than dense alone. `MLComputePlan` places `constexpr_sparse_to_dense` on
  no device and the ctl load takes 17.7 s, which suggests it is expanded at load; yet the
  sparse packages stay resident where the dense ones thrash, which suggests the ANE keeps
  them compact. Which one holds is not known.
- **The live part is far smaller still.** `coreml-census`: attention, every part, 113.4 M
  dense, 6.3 M in rows and columns that hold a nonzero; the FFN slices 420.1 / 234.0 /
  220.9 / 151.4 M dense against 75.4 / 24.5 / 25.0 / 12.7 M live (PPU, GBC CPU, APU, control).
  A graph that gathers the live channels (not built) would cut the dense work by 5 to 18
  times; whether the ANE runs gathers is not checked.

## 5. Exactness (measured)

Every play starts from H4's saved state (`yellow-cache.state`, 141,415 line cells,
llama.cpp's bytes) and runs 300 passes of Yellow with every stream's reduced group, as
`bench.sh` runs them. "301 events equal" is `diff` against the engine's record. The rows-diffs
compare every pass's logit rows with a llama.cpp H11 dump of the same 300 passes
(`STOCK_ROWS=…:1`, H11's own binary, Metal, contended).

| run | result |
| - | - |
| llama.cpp H11 rows dump | 301 events equal, 0 differ, 19,981 decoded tokens |
| ANE, dense, the first GBC CPU pass through Core ML on the Mac CPU, one matmul for the value product | 301 events equal; rows-diff 300 passes, 20,020 logit rows, **0 differ (max diff 0)** |
| ANE, dense, split value product, `cpu-full` through Core ML on the Mac CPU | 301 events equal; rows-diff **0 of 20,020 differ** |
| ANE, the four parts as one multifunction package | 301 events equal |
| ANE, sparse, `cpu-full` on the GPU | 301 events equal; rows-diff **0 of 20,020 differ** |
| GPU, dense, one matmul for the value product | the first control pass differs (event 0); stopped |
| GPU, dense, split value product | 301 events equal; rows-diff **0 of 20,020 differ** |
| GPU, sparse | 301 events equal; rows-diff **0 of 20,020 differ** |
| every `bench.sh` run of §6 (12 runs) | 0 errors over 12 × 301 events |

The window holds 6 write records (3 GBC CPU, 3 PPU palette), every one through `record`, and
one unscoped GBC CPU pass, through `cpu-full`.

## 6. Speed (measured, contended)

M1 Max, macOS 27, Core ML of the OS, llama.cpp b12070 with the slice stack (H11's own
binary and tree). The llama.cpp frame-300 proof run held the GPU throughout, so every run
here is **contended**: `BENCH_NO_LOCK=1`, the lock never touched. Other builds
drove the load average from 3 to 28 during the window. One `bench.sh` run (300 passes)
of each configuration per round, interleaved, three rounds, so each round's four runs saw
about the same load; hours use each stream's median over the rounds and `bench.sh`'s
passes to frame 300 (GBC CPU 981,204, PPU 83,501, APU 243,952, control 59,052).

| run (contended) | load after each round | Cpu, Ppu, Apu, Control ms per pass, rounds 1 / 2 / 3 | median | hours to frame 300 |
| - | - | - | - | - |
| llama.cpp H11 | 21.3, 6.2, 27.5 | 50.1, 67.2, 46.5, 41.0 / 51.2, 65.0, 38.0, 36.5 / 45.1, 53.0, 37.0, 38.3 | 50.1, 65.0, 38.0, 38.3 | **18.4** |
| Core ML GPU, dense | 10.7, 5.1, 10.6 | 56.6, 72.6, 33.3, 30.7 / 67.5, 74.5, 54.5, 41.9 / 59.0, 72.9, 35.8, 31.7 | 59.0, 72.9, 35.8, 31.7 | 20.7 |
| Core ML ANE, sparse (`cpu-full` on the GPU) | 3.9, 4.5, 21.8 | 63.6, 53.7, 29.8, 21.2 / 84.8, 53.4, 44.7, 68.0 / 64.0, 53.2, 28.9, 20.8 | 64.0, 53.4, 29.8, 21.2 | 21.0 |
| Core ML GPU, sparse | 10.6, 12.5, 7.3 | 52.9, 78.9, 35.2, 30.2 / 61.2, 81.7, 40.6, 33.9 / 49.2, 85.1, 39.5, 30.1 | 52.9, 81.7, 39.5, 30.2 | 19.5 |

Every run: 0 errors over its 301 events. The uncontended H11 reference is 11.0 hours (Cpu
30.0, Ppu 38.7, Apu 23.1, Control 21.5, `docs/slice-stack.md`); no uncontended
Core ML GPU figure exists. Round 2's ANE run (GBC CPU 84.8, control 68.0) is an outlier
against rounds 1 and 3 and against the traced run below; what slowed the ANE then was not
found.

**Where the ANE's GBC CPU pass goes.** A traced play (`COREML_TRACE=1`, sparse, ANE, load 3.2)
gives the GBC CPU 59.0 ms per pass: Core ML's predictions 54.7 ms on average (median 52.2; the
one unscoped pass on the GPU, 589 ms, is 2.8 ms of the average), two state fills of 278 ms
(the full 141,440-cell state, once, and the GBC CPU's scope) 2.6 ms per pass, and about 1.7 ms
of harness (embedding rows, logits to f32, three writes). The fills and the unscoped pass
happen once per run, so over a run to frame 300 the GBC CPU's pass on the ANE tends to its
prediction, about 52 ms, and the PPU, APU and control passes to 46.5 to 52, 25 to 29 and
19.5 to 20 ms. Projected from the steady-state predictions of §4 (sparse, all four in
turn): **17.2 hours** (inferred from `coreml-time`, not a bench). The ANE does not share
the GPU, so its figures are the same contended or not, apart from the Mac CPU load and the
unscoped pass.

**Writes.** Each `record` prediction (16 tokens, layer 0 and 15 key and value
projections) plus the pushes into every state holding the line: the play logs' 6 writes
are inside the per-pass figures above; not measured on their own.

## 7. Verdict

**Exact on both.** Core ML computes the one model exactly on the Neural Engine and on the
GPU: every logit of 300 passes is the bit pattern llama.cpp computes, dense and sparse, and
all 12 bench runs decode the engine's tokens. That is measured, not proved, on the ANE,
whose norm is one ulp low on one channel (§3). It rests on one graph rewrite the numbers
require (the gate on `W/2`, since the gate's partial sums reach 65,408 and
the ANE's matmul overflows at 2^15) and on the T2 design of `docs/stock-llama.md` §10; on
the GPU it needed one workaround for a wrong kernel (the split value product).

**Not faster than llama.cpp.** Under the same contended load, llama.cpp H11 projects 18.4
hours, Core ML GPU sparse 19.5, GPU dense 20.7 and ANE sparse 21.0. MLX's best contended
window was 19.6 against llama.cpp's 21.2 (`docs/mlx.md`), so all three runtimes sit within
the load's spread of each other on the GPU. The ANE's own limits are measured: its GBC CPU
pass is about 52 ms, against llama.cpp's 30.0 uncontended, because the ANE streams dense
weights at about 20 GB/s and the GBC CPU pass attends 7,424 cells with a 4,176-row head; it
keeps only three dense programs resident (four thrash at 600 ms a reload), which sparse
weights (233 MB in all) fix. Its 17.2-hour steady-state projection beats contended
llama.cpp and loses to uncontended llama.cpp's 11.0.

**What it offers.** The ANE is a second engine beside the GPU: on it the APU and control
passes (25 and 19.5 ms) already beat llama.cpp's uncontended 23.1 and 21.5, and it is
untouched by GPU contention. The harness runs one pass at a time, so nothing overlaps the
two today (not built). The weights' sparsity is the larger lever: sparse storage cost
nothing in exactness and took 11 to 23% off every ANE pass, and the census (§4) puts the
live rows and columns at 5 to 18 times below the dense work.

## 8. Gaps

- No uncontended GPU comparison: the proof run held the GPU for the whole window. The ANE
  figures do not depend on it.
- The streams run in turn. ANE and GPU in parallel (say the GBC CPU stream on the GPU, the
  rest on the ANE) needs the harness to issue passes of different streams concurrently.
- The unscoped GBC CPU pass runs on the GPU (the ANE refuses 141,440 cells with −14); a
  chunked softmax (coreml-state2's flash-style blocks, exact when the winner leads by
  more than f16's `exp` range) was not tried.
- The ANE's norm returns 63.96875 for an exact 64 on channel 1 (§3). Harmless over these
  300 passes; not explained or bounded.
- The live-channel graph (§4) was not built, nor checked for ANE gathers.
- Round 2's ANE outlier is unexplained.
- No reader record and no write off `P` in the window, as for llama.cpp and MLX; no
  prefill (the backend loads llama.cpp's state and refuses `prefill_lanes`).
- `cpu-full`'s state is filled with every held line at its first use (278 ms), and a write
  updates it from then on; the full state shape (141,440) is Yellow's, set at export.
- Writes were not timed on their own.
- `tools/stock-llama/coreml-bench.sh` repeats §5 and §6 in one script but was not run end
  to end; the measured rows came from scratch scripts making the same calls.

## 9. Reproducing

```sh
# the harness, linked against H11's llama.cpp tree (Core ML needs no build of its own)
(cd tools/stock-llama && CARGO_TARGET_DIR=target-coreml LLAMA_CPP_DIR=$HEADS/llama.cpp-heads \
    LLAMA_CPP_BUILD=$HEADS/llama.cpp-heads/build-metal cargo build --release)
X=tools/stock-llama/target-coreml/release/stock-llama
# the packages: dense (3.6 GB with cpu-full) and sparse (233 MB, plus cpu-full linked or exported)
$X coreml-export $HEADS/one-model/model.gguf $W/cm
COREML_SPARSE=1 COREML_ONLY=ppu,cpu,apu,ctl,record $X coreml-export $HEADS/one-model/model.gguf $W/cms
ln -s $W/cm/cpu-full.mlpackage $W/cms/
# one play on the ANE, its tokens against the engine record
STOCK_BACKEND=coreml COREML_DIR=$W/cms COREML_UNITS=ane BENCH_WORK=$B tools/stock-llama/bench.sh $X $HEADS/one-model/model.gguf
# the interleaved comparison and the rows-diffs (not run end to end: §6 and §5 came from
# per-run scratch scripts that ran the same bench.sh and rows-diff calls)
BENCH_WORK=$B tools/stock-llama/coreml-bench.sh $HEADS/stock-llama-heads $X $HEADS/one-model/model.gguf $W/cm $W/cms
```

The first prediction of a package compiles it (`<package>c` beside it, reused while newer).
`BENCH_NO_LOCK=1` runs beside whatever holds the GPU; its times are then contended.
