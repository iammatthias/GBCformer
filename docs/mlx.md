# The one model on MLX (measured)

This is an evaluation of Apple's MLX as a second third-party runtime for the one-model
checkpoint. It runs the same GGUF that llama.cpp runs, from the same saved cache. Its
tokens are checked against llama.cpp's and the engine's record, and its speed is measured
with the same bench. The verdict is in §5.

This evaluation ran on the one model as it stood before the overlay
(`docs/hardware-shaped.md` §15), exported with the slice stack of `docs/slice-stack.md`. The backend
built for it, `STOCK_BACKEND=mlx` and `mlx-bench.sh`, was not carried into this tree.
The earlier MLX backend of `docs/stock-llama.md` §12 (`tools/stock-llama/src/mlx.rs` and
`mlx_shim.cpp`, built with `MLX_DIR`) has since been removed. The measurements stand as the record of that model.

## 1. What was built

`tools/stock-llama` gains an MLX backend behind the harness's `Runtime` trait, selected by
`STOCK_BACKEND=mlx`. `play`, `bench.sh`, `diff`, `rows-diff` and `timing` run unchanged on
it (`rows-diff` and `timing` have since been removed from the harness). MLX is tag v0.32.4 (`36969c09`), built from source with its C++ API only: no Python,
no Python bindings. `MLX_DIR` names the install prefix at build time, as before.

- `src/mlx.rs` reads the GGUF itself (`Gguf`: metadata scalars and arrays of every integer
  and float kind, each tensor's raw bytes) and hands every tensor to `src/mlx_shim.cpp` as
  it is in the file: f16 matrices, f32 norm weights, `attn_q` and `attn_k` in llama.cpp's
  interleaved pair order.
- `mlx_shim.cpp` is the LLAMA graph as llama.cpp b12070 runs it with the slice stack
  (`layer-shapes`, `ffn-slices`, `ffn-no-up`, `lm-head-parts`):
  - each layer's own query head count (`attn_q` rows / 64), one KV head, head dim 64;
  - f16 matmul operands with f32 accumulation and an f32 residual (§10 of
    `docs/stock-llama.md` makes this exact whatever operand types ggml's kernels use);
  - RoPE in llama.cpp's interleaved (NORM) mode at each token's position, base from
    `llama.rope.freq_base`;
  - the FFN of the batch's one part (`gbcformer.ffn_route`, as `ffn_part_of` picks it),
    `silu(gate)` with no up projection, the activations multiplied by the per-token 0/1
    route mask before `ffn_down`;
  - at the last layer, only the rows read (llama.cpp's `inp_out_ids`);
  - the LM head over the part's run of rows (`gbcformer.head_first`, `head_count`, H11), or
    every row on a file without them;
  - an f16 KV cache.
- **The cache is one row per line.** Row n of each layer's cache holds line n's key and
  value, so llama.cpp's sequences become lists of lines on the device (`mlx_index`):
  sequence 0 is every held line, and a scope is its runs' held lines. A write replaces row
  n (`slice_update`, which may copy the layer's cache; not checked), and every list that
  holds line n reads the new row. This replaces
  llama.cpp's `seq_rm`, `seq_cp` and `seq_add` bookkeeping without changing what any row
  attends.
- **llama.cpp's masks, positions and K-shift.** A pass puts every token at the next
  position `Y` (the largest held line plus one). The rows read attend the pass's cells
  (sequence 0, or the stream's scope) and the group. The other rows attend only the two
  anchors `Cpp::split_in` copies to `SPLIT` (the lowest line of the sequence and the top
  line) and the group. A write record's scratch tokens sit at `Y` and its cell token at
  `Y + 1`, so the scratch tokens do not see the cell token. The cell's f16 key is turned by
  `n − Y − 1` positions in f32 and stored in f16, as llama.cpp's K-shift does.
- **The saved state is llama.cpp's.** `Runtime::load` parses llama.cpp's sequence state
  file (`llama_state_seq_save_file`, version 4: one stream, each cell's position and
  sequence ids, then per layer the f16 key rows and the untransposed f16 value rows) and
  places each cell's rows at its line. The bench starts from exactly the bytes llama.cpp
  starts from. MLX does not prefill: `prefill_lanes` refuses, and a run without
  `STOCK_STATE` stops there.
- **Levers** (`STOCK_MLX`, a comma list; §4): `hoist` builds the attention mask once per
  batch, `norm` runs `mx::fast::rms_norm`, `compile` fuses the norm, activation and
  residual chains with `mx::compile`, and `split` drops the mask where every token shares
  one position (the rows read attend the cells and the group in one unmasked call, the
  rest attend the anchors and the group in another). The timing-only switches `skip-attn`,
  `skip-ffn`, `skip-head`, `contig` and `nomask` leave an op out or change it, as
  `metal-prof-skip.patch` does; their tokens are garbage.
- `STOCK_MLX_DEVICE=cpu` runs the same graph on MLX's CPU backend, which needs no GPU lock.

## 2. Why the same tokens, and what is checked

§10 of `docs/stock-llama.md` makes every product and every partial sum of every matmul an
integer that f16 and f32 hold exactly, so neither the order of a sum nor a kernel's
choice of tiles can change a bit. §2 there makes every norm divide by exactly 2^7 and
lands the normalised value on an integer once it is rounded to f16, which also covers an
`rsqrt` a few ulps off (the `norm` lever). Softmax weights are exactly 0, 1 or 1/len, and
every value dimension a gather reads has one nonzero product. What MLX does differently
from llama.cpp, and why each is exact:

- **Row order.** The harness puts the rows read first (`Mlx::forward`). Every op but
  attention is per row, and attention sees the batch as a set, so the order changes no
  value. Logit rows come back in batch order.
- **Unclaimed rows.** llama.cpp runs the slice's gate on every row of a ubatch and then
  multiplies the activations of rows no part claims by 0. MLX zeroes those rows before
  the norm, so their gate is exactly 0, and multiplies by the same mask after. Zeroing
  first also keeps a gate above f16's range from becoming `inf · 0`. The two differ at
  most in the sign of a zero added to an unclaimed row, and an unclaimed row is never a
  row read.
- **Masks.** With `split`, the rows read attend the cells and the group in one unmasked
  call, and the other rows attend the two anchors and the group in a second call. These
  are the sets llama.cpp's mask allows, so each softmax runs over the same keys. Writes,
  whose tokens sit at two positions, keep the mask.
- **RoPE.** MLX's kernel forms its angles in its own float arithmetic, not ggml's. Group
  tokens share one position, so their bit-identical keys stay bit-identical and their
  softmax weights stay exact. Cells differ from the query by under 4·10^-7 rad on the
  address pairs (§4 there), which f16 rounds away.

The checks: every pass's decoded tokens against the engine's recorded tokens (`diff`, 301
events per 300-pass play), and every pass's logit rows against llama.cpp's, bit for bit
(`rows-diff`).

## 3. Exactness (measured)

Every play below starts from H4's saved state (`yellow-cache.state`, 141,415 line cells,
llama.cpp's bytes) and runs 300 passes of Yellow with every stream's reduced group, as
`bench.sh` runs them. "301 events equal" is `diff` against the engine's record: every
pass's decoded tokens, every stream. Contention does not touch any of these results.

| check | model | runtime | result |
| - | - | - | - |
| 6 plays, levers none, `hoist`, `norm`, `compile`, all three, none again | H9 | MLX, Metal | each 301 events equal, 0 differ, 19,981 decoded tokens |
| 1 play, levers `hoist,norm,compile,split` | H11 | MLX's CPU backend (`STOCK_MLX_DEVICE=cpu`) | 301 events equal, 0 differ |
| rows-diff of that play's every-pass dump against a CPU-backend play with no levers | H11 | MLX CPU against MLX CPU | 300 passes, 20,020 logit rows, 0 differ (max diff 0) |
| 3 plays, levers none, `hoist,norm,compile`, `hoist,norm,compile,split` | H11 | MLX, Metal | each 301 events equal, 0 differ |
| `bench.sh`, 3 runs MLX plain and 3 runs MLX with all four levers, plus one rows-dump run | H11 | MLX, Metal | 0 errors over 7 × 301 events (mirror checks included) |
| **rows-diff of MLX with all four levers against llama.cpp, every pass dumped** (`STOCK_ROWS=…:1` on both, one `bench.sh` run each, H11's binary and tree for llama.cpp) | H11 | MLX Metal against llama.cpp Metal | **300 passes, 20,020 logit rows, 0 differ (0 logits, max diff 0)** |
| rows-diff of the MLX CPU-backend dump against the same llama.cpp dump | H11 | MLX CPU against llama.cpp Metal | 300 passes, 20,020 rows, 0 differ |

H11 narrows each row to its part's run of LM-head rows (`gbcformer.head_first`,
`head_count`), so both dumps hold the same narrowed columns: 256 for the PPU, APU and
control, 4,176 for the GBC CPU. Every logit is the same bit pattern on both runtimes, and so
on MLX's CPU backend too, whose kernels share nothing with Metal's. That is the
integer exactness of §10 at work: the two runtimes sum in different orders, tile
differently, form RoPE's angles differently, and use different softmax and norm kernels,
and no bit moves.

The window holds 6 write records (3 GBC CPU, 3 PPU palette) and no reader record.

## 4. Speed (measured)

M1 Max, MLX v0.32.4, llama.cpp b12070 with the slice stack. Load is `uptime`'s at the
start of each run. Hours to frame 300 are `bench.sh`'s: each stream's ms per pass times its
passes to frame 300 (GBC CPU 981,204, PPU 83,501, APU 243,952, control 59,052).

**Uncontended, H9 (before the proof run took the GPU).** Single plays of 300 passes,
each in the GPU lock, against H9's `bench.sh` median of 3 from `docs/slice-stack.md`
(load 2.1 to 3.8):

| run | load | Cpu, Ppu, Apu, Control ms per pass | hours to frame 300 |
| - | - | - | - |
| llama.cpp H9, `bench.sh` round 1 and round 2 (`docs/slice-stack.md`) | 2.1 to 3.8 | 35.4, 48.3, 26.1, 24.4 and 35.3, 48.5, 26.2, 24.1 | 12.9 |
| MLX, no levers (first play) | 2.8 | 40.7, 53.6, 28.1, 29.5 | 14.7 |
| MLX, no levers (probe, first and last) | 3.4, 2.5 | 40.9, 53.8, 28.9, 29.7 and 41.0, 54.4, 28.4, 29.5 | 14.8 |
| MLX, `hoist` | 3.3 | 40.7, 53.4, 28.0, 29.4 | 14.7 |
| MLX, `norm` | 3.1 | 39.7, 52.4, 26.9, 28.7 | 14.3 |
| MLX, `compile` | 2.3 | 39.3, 51.9, 26.4, 42.3 (control: one run, unexplained) | 14.2 |
| MLX, `hoist,norm,compile` | 2.5 | 38.7, 50.9, 25.9, 27.6 | 13.9 |

`stock-llama timing` (each stream's group over N cells, every row read, median of 5, one GPU
lock, load 2.8 to 4.6; MLX's figure includes copying the logits into the harness's
buffer, llama.cpp's stops at its own logits buffer: 89 rows of 36,311 floats is 12.9 MB, about
1 ms at memcpy speed, computed, not measured):

| runtime, H9 | cells | PPU 170 | CPU 89 | APU 120 | control 82 |
| - | - | - | - | - | - |
| llama.cpp | 2,091 | 43.9 | 23.0 | 26.2 | 19.7 |
| MLX, no levers | 2,091 | 53.5 | 29.2 | 32.1 | 26.3 |
| MLX, `hoist,norm,compile` | 2,091 | 50.7 | 27.2 | 29.8 | 24.2 |
| llama.cpp | 7,300 | 55.0 | 30.3 | 34.5 | 26.9 |
| MLX, no levers | 7,300 | 67.2 | 40.8 | 44.3 | 37.3 |
| MLX, `hoist,norm,compile` | 7,300 | 64.2 | 38.5 | 41.2 | 34.9 |

Where MLX's pass goes, by ablation (`hoist,norm,compile`, ms saved by leaving each part
out; llama.cpp's H9 ablation from `docs/slice-stack.md` beside it for the GBC CPU's 89 tokens):

| left out | MLX CPU 89, 2,091 | llama.cpp CPU 89, 2,091 | MLX CPU 89, 7,300 | llama.cpp CPU 89, 7,300 | MLX PPU 170, 2,091 |
| - | - | - | - | - | - |
| attention (cell gather, mask, SDPA) | 6.5 | 3.4 | 17.5 | 11.6 | 7.9 |
| FFN slices | 9.1 | 6.8 | 9.0 | 8.0 | 24.0 |
| LM head (H9: all 36,311 rows) | 6.9 | 3.8 | 7.4 | 4.7 | 10.4 |
| what is left: projections, norms, elementwise, floor | 7.8 | 9.0 | 6.8 | 11.0 | 14.2 |

MLX loses on every matmul-shaped op and on attention. At 89 rows its f16 GEMM runs the
FFN slices at about 75% of `mul_mm`'s rate (9.1 against 6.8 ms) and the LM head at about
55% (6.9 against 3.8 ms for the same 36,311 × 5,120 head), and its fused attention over 7,300 cells costs 1.5 times
llama.cpp's flash attention. What is left after all three (the projections, the norms,
the elementwise ops and the per-pass floor) is about even. A probe at 7,300 cells (one
lock, but taken while another job's trap had removed the lock, so only its relative
figures stand) put the array mask at about 4 ms of the GBC CPU pass's attention and the cell
gather at under 1 ms; `split` removes the mask.

**Contended, H11 (the proof run holding the GPU).** MLX GPU work ran beside
the llama.cpp frame-300 proof run, without its lock (`BENCH_NO_LOCK=1`). Every number in
this part was taken that way, labelled contended: the proof run's decodes share the GPU,
so every pass is slower than uncontended and the spread is wider. llama.cpp H11 (its own
binary and tree) and MLX ran back to back, one `bench.sh` run each per round, three
rounds, so each round's three runs saw the same load; the hours use each stream's median
over the three rounds.

| run (contended, proof run on GPU) | load per round | Cpu, Ppu, Apu, Control ms per pass, rounds 1 / 2 / 3 | median | hours to frame 300 |
| - | - | - | - | - |
| llama.cpp H11 | 3.3, 4.4, 4.1 | 56.8, 69.7, 53.3, 44.8 / 55.1, 75.3, 42.1, 38.9 / 56.2, 70.9, 52.6, 38.2 | 56.2, 70.9, 52.6, 38.9 | **21.2** |
| MLX, no levers | 16.6, 4.3, 5.2 | 62.3, 85.7, 52.6, 38.4 / 62.1, 82.8, 45.0, 36.2 / 64.4, 96.9, 44.7, 36.7 | 62.3, 85.7, 45.0, 36.7 | 22.6 |
| MLX, `hoist,norm,compile,split` | 8.5, 3.7, 4.2 | 51.3, 75.4, 45.1, 39.5 / 52.4, 69.5, 44.7, 36.7 / 51.9, 76.7, 45.3, 37.0 | 51.9, 75.4, 45.1, 37.0 | **19.6** |

Every run: 0 errors over its 301 events. The uncontended H11 reference is 11.0 hours (Cpu
30.0, Ppu 38.7, Apu 23.1, Control 21.5, `docs/slice-stack.md`), so contention roughly doubles a pass.

`split` on its own (contended, H11, MLX `hoist,norm,compile` against
`hoist,norm,compile,split`, one `bench.sh` run each per round, three rounds; another job
drove the load to 42 during round 1's second run):

| run (contended, proof run on GPU) | load per round | Cpu, Ppu, Apu, Control ms per pass, rounds 1 / 2 / 3 | median | hours |
| - | - | - | - | - |
| `hoist,norm,compile` | 7.5, 12.4, 10.9 | 57.8, 79.1, 42.3, 34.1 / 60.6, 78.4, 42.9, 33.7 / 61.9, 94.0, 45.9, 33.0 | 60.6, 79.1, 42.9, 33.7 | 21.8 |
| `hoist,norm,compile,split` | 42.4, 33.0, 5.3 | 56.9, 74.9, 51.9, 41.0 / 58.7, 81.0, 47.6, 42.5 / 61.9, 81.5, 52.6, 33.6 | 58.7, 81.0, 51.9, 41.0 | 22.4 |

Under this load the two cannot be told apart: the GBC CPU moves by less than its own
round-to-round spread, and the APU and control rows of the `split` runs sat in the load
spike. The window before it, which ran all three runtimes back to back at load 3.3 to 5.2,
is the one the verdict uses.

**Writes, which the bench leaves out.** The metric counts passes only, and its window holds
6 write records. Their cost from the play logs' `CacheStats` (`write_ns / writes`): MLX
322 ms per write (uncontended, H9, 1.94 s over 6; 320 to 424 ms across the H9 probes),
llama.cpp H11 124 to 127 ms (the proof run's smoke log, 6 and 89 writes). A small sample,
but MLX's writes are about 2.5 times llama.cpp's. Three things in this backend make them
so (inferred from the code, not profiled): a write decodes its 16 tokens over all 141,415
cells; `mlx_store`'s `slice_update` may copy each layer's 36 MB key and value arrays; and
`Mlx::changed` drops the held index and every scope index, so the next pass of each
stream uploads its lines again. A run to frame 300 writes far more often than this window,
so the hours above understate MLX's total more than llama.cpp's.

### The levers, summed up

| lever | what it changes | measured |
| - | - | - |
| `hoist` | the attention mask built once per batch instead of once per layer | uncontended H9: 0.2 ms GBC CPU, within noise |
| `norm` | `mx::fast::rms_norm` (one kernel) for the seven-op norm | uncontended H9: 1.0 to 2.0 ms per stream; exact (rows-diff against llama.cpp, all four levers on) |
| `compile` | `mx::compile` (shapeless) fuses the norm, activation and residual chains | uncontended H9: 1.6 to 2.5 ms on GBC CPU, PPU and APU; control 12.6 ms slower in its one run, unexplained, and 2.1 ms faster with all three |
| all three | | uncontended H9: 13.9 hours against 14.8 with none, against llama.cpp's 12.9 |
| `split` | no array mask where every token shares one position | not resolved: under contention within the spread; `timing` without the mask (relative figure only) put the mask at about 4 ms of a GBC CPU pass |

`mx::compile` of a whole layer was not tried: a layer's shapes change with the scope's
cell count and the stream's group, so each new shape retraces, and the per-pass floor it
would cut is the "what is left" row above, where MLX already matches llama.cpp.

## 5. Verdict

MLX computes the one model exactly: every logit of 300 passes is the bit pattern
llama.cpp computes, on Metal and on MLX's CPU backend, with every lever on.

MLX is not faster. Uncontended on H9 it is 8% slower than llama.cpp with its best measured
levers (13.9 hours against 12.9) and 15% slower without them (14.8). The ablation says
why, and it is MLX's kernels, not the harness: at 82 to 170 rows its f16 GEMM is 55% to
75% as fast as ggml's `mul_mm` on the FFN slices and the LM head, and its fused attention
with an array mask over 7,300 cells costs 1.5 times ggml's flash attention. The floor
around the kernels is no worse. Under the proof run's contention, run back to back with
llama.cpp H11, MLX with all four levers projected 19.6 hours against llama.cpp's 21.2 and
MLX without levers 22.6. That is the only window where MLX leads, and a later contended
window put the same levers at 21.8 to 22.4 hours, so the contended lead is within what
the load moves (inferred: two runtimes sharing one GPU with a third process need not slow
down alike).

Its write records, which the metric leaves out, cost about 2.5 times llama.cpp's in this
backend (§4). So llama.cpp stays the runtime to beat on dense passes. MLX's real offer is elsewhere:
custom Metal kernels inside the graph (`mx::fast::metal_kernel`) with no fork of the
runtime, where llama.cpp needs a ggml op and a patch for each. The weights are 0.07%
nonzero (`docs/modern-speedups.md` §1, §11), so a sparse matmul kernel would cut the work
the ablation shows MLX losing on, the matmuls, by far more than any dense kernel could
(estimate, not built or measured). The project's own engine already evaluates the
weights that way (`docs/modern-speedups.md` §11), so that path competes with the engine,
not with llama.cpp.

## 6. Gaps

- No uncontended H11 comparison: the frame-300 proof run held the GPU from 07:36. The
  contended rows are interleaved and labelled; `tools/stock-llama/mlx-bench.sh` reruns the
  comparison in the lock once it is free.
- Writes cost MLX about 2.5 times llama.cpp's (§4) and the metric leaves them out. The
  fixes are in this backend, not MLX: keep each index when a write leaves the held set as
  it was (an eviction and its rewrite), and check whether `slice_update` copies the cache.
- MLX does not prefill: `prefill_lanes` refuses, so every MLX run starts from llama.cpp's
  saved bytes.
- The window holds no reader record and no write off `P` (the top cell's own rewrite), as
  in llama.cpp's own evidence. The K-shifted key rows MLX writes were never compared with
  llama.cpp's cache bytes; only every logit after them was.
- The backend reads only the slice model's GGUF (`gbcformer.ffn_route`), so `run.sh`'s
  MLX `sst` and `sst-mem` steps, which ran the byte form's safetensors (§12b of
  `docs/stock-llama.md`), are gone.
- The 7,300-cell `nomask` and `contig` timing ran after another job's `bench.sh` freed the
  lock under it (that script sets its `rmdir` trap before it holds the lock); only its
  relative figures are used. `bench.sh` here now sets the trap only once it holds the lock.

## 7. Reproducing

```sh
# MLX v0.32.4 from source, C++ only (about 250 MB installed)
git clone --depth 1 --branch v0.32.4 https://github.com/ml-explore/mlx.git $W/mlx
cmake -S $W/mlx -B $W/mlx/build -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_GGUF=OFF -DCMAKE_INSTALL_PREFIX=$W/mlx-install
cmake --build $W/mlx/build -j 4 && cmake --install $W/mlx/build
# the harness, linked against the slice stack's llama.cpp tree
(cd tools/stock-llama && CARGO_TARGET_DIR=target-mlx LLAMA_CPP_DIR=$SLICES LLAMA_CPP_BUILD=$SLICES/build-metal \
    MLX_DIR=$W/mlx-install cargo build --release)
# one play, its tokens against the engine record
STOCK_BACKEND=mlx STOCK_MLX=hoist,norm,compile,split BENCH_WORK=$W tools/stock-llama/bench.sh \
    tools/stock-llama/target-mlx/release/stock-llama $W/one-model-slices/model.gguf
# the interleaved comparison and the rows-diff
BENCH_WORK=$W tools/stock-llama/mlx-bench.sh $LLAMA_HARNESS tools/stock-llama/target-mlx/release/stock-llama $W/one-model-slices/model.gguf
```

`STOCK_MLX_DEVICE=cpu` runs any of these on MLX's CPU backend; `BENCH_NO_LOCK=1` runs
`bench.sh` beside whatever holds the GPU, and its times must then be labelled contended.
