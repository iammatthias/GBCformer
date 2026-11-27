# The slice stack in llama.cpp (measured)

This page continues `docs/stock-llama.md` §12a. It records the llama.cpp hill climb from
H6 to H12 and the first end-to-end proof run, Yellow from the hand-off state to frame 300 with every
pass decoded by llama.cpp.

Every run here used the one model as it stood before the overlay
(`docs/hardware-shaped.md` §15), not today's checkpoint. The overlay gives each part a
prefix of channels, layers, units and heads, which replaces the per-part FFN slices and
LM-head ranges built here. So the model-side code these runs used was not carried into
this tree: the export flags `--experts`, `--slices`, `--no-up` and `--part-heads`,
`one_model::experts`, `head_range` and `head_check`, the narrowed logit rows in the
harness, and the examples `head_census` and `moe_check`. The harness side legs this page
names have since been removed: the patches below, the
`metal-gqa-fold.patch` and `kv-shift-rows.patch` pair, and the `timing` and `rows-diff`
subcommands. Of the other files:

- The four llama.cpp patches, on b12070 after `layer-shapes.patch`:
  `tools/stock-llama/ffn-slices.patch`, `ffn-no-up.patch`, `lm-head-parts.patch` and
  `metal-prof-skip.patch`. This page is the record of what ran.
- `tools/stock-llama/proof.sh` and `proof-status.sh`. They run the end-to-end check on
  any one-model GGUF the harness loads, and `play`'s `STOCK_REFERENCE` holds every
  decoded pass to the engine's record.

**One expert per part (H6, measured, not kept).** With attention scoped, most of a
pass is the dense FFNs, and each stream's rows need only its own part's units. The
export can write the FFNs as stock llama.cpp's LLAMA mixture of experts (`export
one-model DIR --layer-shapes --align 64 --experts`: `ffn_gate_inp`,
`ffn_{gate,up,down}_exps`, `llama.expert_count` 5, `llama.expert_used_count` 1), split
from the weights by `one_model::experts`. No circuit, `Model` or digest changes: every
cell's and group token's keys and values are the dense model's, so H4's prefilled state
loads as it is, and the export without `--experts` is byte-identical to H3b's GGUF.

- **The router.** Five experts: PPU, GBC CPU, APU, control, and an empty one. Each part's
  output masks score 2,048 on their slot identity channels (the GBC CPU's also on the eight
  `RD` channels), and the constant channel scores 1,024 for the empty expert. Nothing
  writes those channels (asserted), so a token's logits are its embedding's at every
  layer: each of the 36,311 token ids leads by at least 1,024 (`exp(−1024)` is 0), so
  the top-1 weight is exactly 1 (`norm_w`: p/p with p = 1) and the other four experts
  contribute nothing. Tokens per expert: PPU 162, GBC CPU 3,664 (its output masks, the
  reader tokens, and the record families, which carry a write slot's identity), APU 60,
  control 42, empty 32,383 (every input token, `REC`, `CELL`, the output bytes).
- **Which units an expert keeps.** Walking back from the LM head, each channel carries
  what its value must satisfy at the expert's rows: EXACT (the dense pass's value);
  NONNEG or NONPOS (an integer of that sign, which a clearing unit downstream turns into
  the dense pass's 0); INT (an integer); FREE (only the norm reads it). A unit is kept
  when a channel it writes must be EXACT after it, unless its gate is at most 0 at
  every token routed to the expert (over the declared ranges at its layer,
  `Model::channel_vars(layer, true)`, and each token's own channels exactly; a gate is
  a multiple of K there, where SiLU is 0). A clearing unit is kept when its channel
  must be EXACT after it, or must have the sign its occupant lacks. A head writing a
  channel that is not FREE gets an EXACT query, so it writes the dense pass's integer. A
  read makes its channels EXACT, asserted consistent with any sign required there. The
  engine's `Plan::prune` works on virtual channels, one per occupant; this works on the
  dense channels the runtime has, where a reused channel's earlier occupants must be
  cleared exactly or left at a sign the later clear absorbs.
- **Why integers.** SiLU is ReLU only at multiples of K. A head whose query nothing
  keeps attends a mix of lines, and a clearing unit fed that fraction leaves a residue:
  the first exact-looking export (signs without integers) decoded every token right but
  differed from H4 in 540 of 20,020 logit rows by 1.79·10⁻⁴, one route channel off.
- **The norm.** Left-out units leave stale values and heads with unkept queries write
  mixes into channels nothing needs, so every norm's exactness is argued anew: each
  expert's channel intervals are carried forward through the mixture (heads add the
  range of their value mix, kept units their largest activation, a clearing unit maps
  its interval, each point is cut to what its requirement guarantees, layer 0's
  attention to its declared range since it is the dense pass's at every row). The worst
  sum of squares is 38,434 against the limit 16·d = 81,920.
- **Checked on the Mac CPU.** `examples/moe_check` runs each stream's group (its own slots,
  inputs at random bytes, outputs masked) through a sparse f32 forward pass with every
  unit and with each row's expert, and compares every logit bit: 12 cases per stream,
  1,944 PPU, 864 GBC CPU, 720 APU and 504 control decoded rows, 0 differ. Its first run found 9
  APU rows of 600 that decoded differently; `MOE_BISECT=1` named a clearing unit (layer
  1, channel 2,354) left out under a NONPOS requirement its 0/1 occupant could not
  meet, which the keep rule above now covers. It has no cells, so its heads never mix
  lines; the residue above showed only in the runtime. Reader records (`RD` rows) are
  routed and proved like the GBC CPU's masks, but neither check runs one (the bench's 300
  passes hold none).

Widths per layer (units with any weight; the mixture's width is the widest expert,
padded to 64):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | sum (padded) |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| dense (H3b) | 1,746 | 5,778 | 5,468 | 4,953 | 5,537 | 6,306 | 5,907 | 5,890 | 6,110 | 5,528 | 5,296 | 5,296 | 1,666 | 1,227 | 3,252 | 70,528 |
| PPU | 357 | 2,007 | 4,844 | 4,202 | 3,384 | 2,864 | 3,868 | 3,349 | 1,795 | 5,503 | 3,768 | 2,000 | 360 | 992 | 1,322 | |
| CPU | 355 | 2,552 | 3,250 | 4,556 | 1,759 | 1,928 | 2,167 | 1,003 | 1,581 | 0 | 3 | 1,839 | 256 | 25 | 1,189 | |
| APU | 772 | 4,046 | 2,900 | 4,593 | 1,434 | 1,783 | 531 | 883 | 2,145 | 0 | 192 | 594 | 647 | 111 | 511 | |
| control | 1,126 | 1,983 | 2,215 | 4,134 | 1,032 | 1,302 | 343 | 330 | 583 | 25 | 13 | 271 | 403 | 99 | 329 | |
| mixture | 1,152 | 4,096 | 4,864 | 4,608 | 3,392 | 2,880 | 3,904 | 3,392 | 2,176 | 5,504 | 3,776 | 2,048 | 704 | 1,024 | 1,344 | 44,864 |

The GGUF holds 3,931.3 M parameters (five experts at each layer's width), 1,174.9 M
active per token against H3b's 1,568.7 M; the FFN work per token falls to 64%. The PPU
sets the width at 10 of the 15 layers, so every stream's tokens pay the PPU's width:
the floor of this split is the PPU's own total. Splitting the PPU's 160 pixel rows
over more experts (measured on the first lattice) leaves its shared per-line work
whole: the padded sum fell from 43,136 to 37,952 with two PPU experts, 35,328 with
four and 34,368 with eight.

`stock-llama timing` (patched Metal, each stream's group as its pass holds it, load
3.1 to 3.5, median of 5, ms per pass; the mixture rows are the first export, 43,136
units):

| model | cells | PPU 170 | CPU 89 | APU 120 | control 82 |
| - | - | - | - | - | - |
| H3b/H4 dense | 2,091 | 71.8 | 40.6 | 50.0 | 40.2 |
| mixture | 2,091 | 68.3 | 43.7 | 56.9 | 41.8 |
| H3b/H4 dense | 7,300 | 83.1 | 48.0 | 58.3 | 47.4 |
| mixture | 7,300 | 79.4 | 51.5 | 65.8 | 48.9 |
| mixture, every token of a group on one expert | 2,091 | 66.5 | 42.8 | 57.3 | 41.9 |
| dense at the mixture's widths (the GBC CPU expert's units; timing only, not exact) | 2,091 | 54.1 | 31.3 | 37.9 | 30.9 |
| dense, every FFN 64 wide (timing only) | 2,091 | 26.8 | 17.7 | 20.2 | 17.1 |
| mixture, every expert 64 wide (timing only) | 2,091 | 37.7 | 26.4 | 30.3 | 26.0 |

What limits it is the mixture's plumbing, not the units. At width 64 the mixture costs
8.7 ms more per GBC CPU pass than the dense model, about 0.58 ms a layer (the router, the
fused top-k, and three `mul_mat_id`, each with its id map, zero fill and barrier). At
the real widths `kernel_mul_mm_id` does in 17.3 ms the units dense `mul_mm` does in
13.6 ms (0.41 against 0.31 µs per unit for 89 tokens). The 39% fewer units save 9.3 ms
of the GBC CPU pass's 22.9 ms of FFN; the plumbing spends 12.4. A layer pays only where
0.41·w_mixture + 580 µs < 0.31·w_dense: layers 5, 8 and 11 for the GBC CPU, about 1 ms a
pass. That would need a loader patch for mixed dense and expert layers (the LLAMA
graph already branches per layer on `ffn_gate_inp`), for about 2% of the total.

| run (`tools/stock-llama/bench.sh`, median of 3, H4's prefilled state) | load | Cpu, Ppu, Apu, Control ms per pass | hours to frame 300 | check |
| - | - | - | - | - |
| H4 dense, again in this round | 2.6 to 3.9 | 53.3, 76.3, 49.3, 44.3 | 20.4 | 0 errors over 903 events |
| mixture, signs without integers (43,136 units) | 3.1 to 5.0 | 57.8, 74.5, 56.5, 46.8 | 22.1 | 0 errors over 903 events; logit rows against H4: 540 of 20,020 differ |
| mixture (44,864 units) | 1.8 to 4.5 | 58.5, 75.7, 57.3, 47.7 | **22.4** | 0 errors over 903 events; `rows-diff` against H4 over 300 passes (`STOCK_ROWS=…:1`): 20,020 rows, 0 differ; 301 events equal the engine record |

The mixture is exact and slower. Not kept.

**One slice per part (H8, measured, kept).** H6's timing table shows that the FFN width
costs the time, and that the mixture's plumbing (router, top-k, `mul_mat_id`) costs more
than the narrower experts save. Here each part keeps H6's units, but as plain dense
matrices at the part's own width. A ubatch picks one slice, so there is no router.
`export one-model DIR --layer-shapes --align 64 --slices` writes them, and
`tools/stock-llama/ffn-slices.patch` (on top of `layer-shapes.patch`) runs them. No
circuit, `Model` or digest changes: the attention tensors are H4's, so H4's prefilled
state loads as it is. The patched build is stock b12070 with `layer-shapes.patch`, then
`ffn-slices.patch`, built with `run.sh`'s cmake line: H4's tree (`layer-shapes` alone) plus
the slices. Neither tree carries `metal-gqa-fold` or `kv-shift-rows`. `run.sh setup` builds
this stack, with H9's patch on top, as `llama.cpp-slices`.

- **The file.** Per layer and part, `blk.N.ffn_{gate,up,down}.P` holds that part's
  `one_model::experts` units in ascending order, padded with zero units to a multiple of
  64. Where a part keeps no unit (the GBC CPU and APU at layer 9) the tensors are absent. The
  router becomes a table: `gbcformer.ffn_route` is H6's `Experts::route`, one i32 per
  token id (0 to 3 a part, 4 none), and `gbcformer.ffn_parts` is 4. The export asserts
  that the empty expert keeps no unit. Each part's slice is its own contiguous tensor,
  not a view of one wide matrix. A column view of `ffn_down` would be strided, and the
  parts overlap, so one shared order could not give every part a contiguous range
  without copies anyway. The copies cost memory, not time: 100,224 units against H4's
  70,528, 2,024.8 M parameters, a 4.05 GB GGUF.
- **How llama.cpp learns the part.** From the tokens. At graph build the patch reads
  each token's route, and it asserts that a ubatch holds tokens of at most one part.
  Every pass and record decode the harness issues meets that, since H6's routes already
  follow the stream's own slots. The graph input's `can_reuse` compares the part and the
  token count, so a reused graph never runs another part's slice, and `set_input` refills
  the mask on every ubatch. The runtime and `bench.sh` change nothing. A per-context
  setter would also work. But APU and control share one scope sequence, so a
  per-sequence tag cannot name the part, and a setter is one more call that every pass,
  including the GBC CPU's record decodes, would have to get right.
- **Tokens no part claims.** A ubatch also holds tokens routed to none: input slots,
  `REC`, `CELL`. They share the part's matmuls, and the patch multiplies the
  activations, after SiLU·up and before `ffn_down`, by a per-token 0/1 vector (1 when
  the token's route is the ubatch's part). At the last layer the vector goes through the
  same `out_ids` rows as the residual. Those rows then add an f32 ±0, exactly what H6's
  empty expert adds. The activation is a finite f32 there, because the row's input is
  H6's empty-expert row, inside the norm bound. Masking before `ffn_down` keeps an
  activation above f16's range away from the down matmul's f16 operand. So every row
  computes what H6's mixture computes at weight exactly 1: its own part's units, or
  none. H6's requirement walk, dead-gate proofs and norm bound (38,434 of 81,920) carry
  over unchanged. Matmul summation order does not matter, because every product and
  partial sum is an integer the f16 and f32 types hold exactly (§10).

| check | load | result |
| - | - | - |
| `examples/moe_check` with `MOE_GGUF` (the exported file read back, f16 decoded, its route table and slices; each row runs its route's slice or none) against the dense pass, 12 cases per stream | 3.0 (after) | 58 slices read, route equal to `experts()`, each slice's units equal to its expert's; 1,944 PPU, 864 GBC CPU, 720 APU and 504 control decoded rows, 0 logit rows differ in any bit |
| `rows-diff` against H4 over 300 passes (`STOCK_ROWS=…:1` on both sides, one `bench.sh` run each) | 3.2 to 3.7 | 300 passes, 20,020 logit rows, 0 differ (max diff 0); 301 events equal the engine record on both |

`stock-llama timing` (Metal, each model on its own tree as above, each stream's group as its pass holds it, median
of 5, ms per pass; base and slices interleaved in one GPU lock, load 4.5 to 3.5):

| model | cells | PPU 170 | CPU 89 | APU 120 | control 82 |
| - | - | - | - | - | - |
| H4 dense | 2,091 | 72.0 | 40.6 | 50.0 | 40.4 |
| slices | 2,091 | 54.4 | 27.4 | 31.1 | 23.9 |
| H4 dense | 7,300 | 83.3 | 48.3 | 58.5 | 47.2 |
| slices | 7,300 | 66.2 | 34.3 | 38.8 | 30.3 |

At 2,091 cells the GBC CPU pass loses 13.2 ms, close to the 15 ms that its width (22,848 of
70,528 units, 32%) predicts for the 22.9 ms of FFN measured under H6. The PPU keeps 58%
of the units and saves the least.

| run (`tools/stock-llama/bench.sh`, median of 3, H4's prefilled state, base and slices interleaved) | load | Cpu, Ppu, Apu, Control ms per pass | hours to frame 300 | check |
| - | - | - | - | - |
| H4 dense, round 1 | 2.8 to 4.0 | 53.3, 76.5, 49.3, 45.3 | 20.4 | 0 errors over 903 events |
| slices, round 1 | 2.0 to 3.0 | 39.5, 58.7, 30.5, 28.6 | **14.7** | 0 errors over 903 events |
| H4 dense, round 2 | 2.4 to 3.0 | 53.2, 76.2, 49.3, 44.2 | 20.3 | 0 errors over 903 events |
| slices, round 2 | 2.4 to 3.9 | 39.4, 58.8, 30.4, 27.7 | **14.6** | 0 errors over 903 events |

The slices are exact and take 28% off the projection, 20.3 hours to 14.6. Kept. Three
gaps remain. Reader-record (`RD`) rows route to the GBC CPU's slice, but neither the Mac CPU
oracle nor the bench's 300 passes run one, as under H6. A prefill under the slice model
has not run, since H4's state loads. The first control pass, before the cache is
filled, runs like any other part's ubatch, unscoped as in every run.

**No up projection (H9, measured, kept).** Every unit `step` writes has an up row of 1 on
the constant channel and 0 elsewhere (`llama.rs`, `Model::check_f16_acc` asserts it).
After the exact norm the constant reads `UNIT / 2^7 · GAMMA = 2^-7 · 2^-7 · 2^14 = 1`, so
`up · h` is exactly 1 for every unit at every row, and the SwiGLU product
`silu(gate) · up` is `silu(gate) · 1`. `export one-model DIR --layer-shapes --align 64
--slices --no-up` leaves every slice's `ffn_up` out of the file, and
`tools/stock-llama/ffn-no-up.patch` (on top of `ffn-slices.patch`) loads `ffn_up` as
optional and runs `ggml_silu(gate)` where a slice has none, in place of
`ggml_swiglu_split(gate, up)`. The mask multiply and `ffn_down` stay where H8 put them,
and `ffn_down` and `ffn_gate` are byte-identical to H8's. So nothing folds into `ffn_down`:
the constant is 1. No circuit, `Model` or digest changes, and H4's prefilled state loads as
it is. The file is 3.02 GB against H8's 4.05 GB (1,511.7 M parameters against 2,024.8 M).

- **Why the constant is 1 at every row.** The export asserts each premise on the
  compiled weights before it writes (`up_is_one`): every unit with a gate or down weight
  has the up row `e_0`, and every other unit has an all-zero up row; every token id embeds
  channel 0 as `UNIT`; no `o` row and no `ffn_down` row writes channel 0 at any layer. So
  channel 0 holds `UNIT` in the residual of every row at every layer. Every norm divides by
  exactly 2^7 because `rms_norm_eps` dominates the sum of squares (§2), and under the slices
  that holds through H6's norm bound (38,434 of 81,920), which covers the rows no part
  claims too. `GAMMA` and `UNIT` are powers of two, so the norm's output on channel 0 is
  exactly 1 in f32 and in f16.
- **Why dropping the multiply is exact.** `x · 1.0` is `x` in IEEE arithmetic, for every
  finite `x` and for ±0. Metal computes `silu` as `x / (1 + exp(-x))` in both
  `kernel_swiglu` and the unary `kernel_silu` (`ggml-metal/kernels/unary.metal`), the same
  expression on the same f32 gate, and the swiglu kernel then multiplies by `x1`, which was
  exactly 1. Gate pre-activations are multiples of K = 128, where that expression is
  exactly ReLU (§6, §10). The padding units have zero gate and zero down rows: `silu(0) = 0`,
  and their down column adds 0, as before. The rows no part claims are masked by the same
  0/1 vector after the activation. The up matmul was a third of each slice's FFN matmuls,
  and no product or partial sum that remains changes, so §10's per-row bounds hold as they
  did under H8.

| check | load | result |
| - | - | - |
| `export --no-up` premise assertions (`up_is_one`) on the compiled one model | 5.1 | pass: up `e_0` on every unit in use, constant channel embedded as `UNIT` on all 36,311 token ids, never written by a head or a unit |
| `examples/moe_check` with `MOE_GGUF` on the no-up file (an absent `ffn_up` runs silu(gate) times a literal 1, as the patch does) against the dense pass, 12 cases per stream | 4.4 to 3.6 | 58 slices read, route equal to `experts()`; 1,944 PPU, 864 GBC CPU, 720 APU and 504 control decoded rows, 0 logit rows differ in any bit |
| `rows-diff` of the no-up model against H8 over 300 passes (`STOCK_ROWS=…:1` on both sides, one `bench.sh` run each, each on its own binary) | 2.4 to 3.1 | 300 passes, 20,020 logit rows, 0 differ (max diff 0); 301 events equal the engine record on both |

`stock-llama timing` (Metal, each model on its own tree as above, each stream's group as its pass holds it, median
of 5, ms per pass; H8 and no-up interleaved in one GPU lock, each on its own binary, load
2.7 to 3.6):

| model | cells | PPU 170 | CPU 89 | APU 120 | control 82 |
| - | - | - | - | - | - |
| H8 slices | 2,091 | 54.4 | 26.9 | 30.8 | 23.4 |
| no up | 2,091 | 44.0 | 23.0 | 26.2 | 19.8 |
| H8 slices | 7,300 | 65.8 | 34.1 | 38.7 | 30.3 |
| no up | 7,300 | 55.0 | 29.9 | 34.2 | 26.6 |

The PPU saves the most, 10.4 ms at 2,091 cells, since it keeps the most units; the GBC CPU
saves 3.9 ms, close to a third of the FFN work H8's slices left it.

| run (`tools/stock-llama/bench.sh`, median of 3, H4's prefilled state, H8 and no-up interleaved) | load | Cpu, Ppu, Apu, Control ms per pass | hours to frame 300 | check |
| - | - | - | - | - |
| H8 slices, round 1 | 2.8 to 3.1 | 39.6, 59.2, 30.5, 29.3 | 14.7 | 0 errors over 903 events |
| no up, round 1 | 2.9 to 3.8 | 35.4, 48.2, 26.2, 24.4 | **12.9** | 0 errors over 903 events |
| H8 slices, round 2 | 3.6 to 5.4 | 39.6, 59.3, 30.6, 27.9 | 14.7 | 0 errors over 903 events |
| no up, round 2 | 3.1 to 3.8 | 35.3, 48.1, 26.1, 24.8 | **12.9** | 0 errors over 903 events |

The no-up model is exact and takes 12% off the projection, 14.7 hours to 12.9, and it beats
H8 on every stream in both rounds. Each model's two rounds agree within 0.1 ms on the GBC CPU,
PPU and APU; control moved 1.4 ms between H8's rounds and 0.4 ms between the no-up
model's. Kept. H8's gaps carry over
unchanged: no reader-record (`RD`) row runs in either check, and no prefill has run under
the slice model.

**The two older patches on the slice stack (H10, measured, not kept).** H9's tree is
stock b12070 with `layer-shapes`, `ffn-slices` and `ffn-no-up`. `metal-gqa-fold.patch`
and `kv-shift-rows.patch` were written for the byte form (§12b) and were never in a tree
the bench measured. Both apply cleanly on top of H9's three, so H10 tests the five-patch
stack on H9's model file and H4's prefilled state. Before the measurements, a check of
whether the bench's passes even reach the code each patch changes:

- **`kv-shift-rows`** changes only the K-shift, which a `seq_add` leaves pending until the
  next `llama_decode`. The runtime calls `seq_add` only to move a write record's cell to its
  line (`Cpp::write`) and in `prefill_lanes`. The scopes use `seq_cp`, and the bench opens no
  lanes. In the bench's 300 passes the GBC CPU writes 3 lines and the PPU 3 (the play's counts
  line), so the patch can change at most the 6 decodes that follow a write.
- **`metal-gqa-fold`** turns the 22 query heads that share the one KV head into query rows
  of that head. It was written for small groups: stock runs groups below 20 query tokens
  in the vector kernel, which streams the cache once per query head, and the patch turns
  the vector kernel off and runs the heads as rows of the 8×8 matrix kernel instead, which
  streams it once. Every pass group here holds 82 to 170 tokens, so stock already runs the
  8×8 kernel, and what the fold avoided does not happen. It also sets
  `blk_cur = 1` for every block, which turns off the kernel's per-block mask skip: a block
  of 8 query rows by 64 cells that is masked whole is read anyway. The split pass
  (`Runtime::split_pass`) relies on that skip, because only the decoded rows attend the cells.

A probe ran one play of each build (300 passes, single runs, so they only attribute the
bench rows), using the patches' own switches `GGML_METAL_FA_FOLD=0` and
`LLAMA_KV_SHIFT_ROWS=0`:

| run (one play, 300 passes) | load | Cpu, Ppu, Apu, Control ms per pass | check |
| - | - | - | - |
| H9 | 5.6 | 35.2, 48.0, 26.0, 24.1 | 301 events equal, 0 differ |
| H10, both patches | 3.6 | 35.9, 48.3, 27.5, 26.2 | the same |
| H10, fold off (shift rows only) | 4.7 | 35.7, 47.4, 26.0, 25.0 | the same |
| H10, shift rows off (fold only) | 3.9 | 35.8, 48.9, 27.8, 25.4 | the same |
| H9 again | 4.2 | 35.4, 48.5, 26.2, 26.2 | the same |

The fold costs the APU about 1.5 ms per pass. The other differences sit inside the
spread of H9's own two runs (control moved 2.1 ms between them).

| check | load | result |
| - | - | - |
| `rows-diff` of H10 against H9 over 300 passes (`STOCK_ROWS=…:1` on both sides, one `bench.sh` run each, each on its own binary) | 2.4 to 3.6 | 300 passes, 20,020 logit rows, 0 differ (max diff 0); 301 events equal the engine record on both |

`stock-llama timing` (Metal, each stream's group as its pass holds it, median of 5, ms per
pass; H9, H10 and H10 with the fold off interleaved in one GPU lock, load 2.0 to 2.6).
`timing` decodes every row against every cell, so it holds no mask for the skip to use:

| model | cells | PPU 170 | CPU 89 | APU 120 | control 82 |
| - | - | - | - | - | - |
| H9 | 2,091 | 44.1 | 23.0 | 26.3 | 19.8 |
| H10 | 2,091 | 44.1 | 22.8 | 26.6 | 19.8 |
| H10, fold off | 2,091 | 44.4 | 23.0 | 26.7 | 20.6 |
| H9 | 7,300 | 55.3 | 30.0 | 34.3 | 26.7 |
| H10 | 7,300 | 55.6 | 30.0 | 35.9 | 26.9 |
| H10, fold off | 7,300 | 56.5 | 31.1 | 35.3 | 27.1 |

Folding saves nothing even with no mask to skip: GBC CPU 30.0 against 30.0 at 7,300 cells.
Reading the keys once per KV head instead of once per query head does not shorten the
kernel, so at these sizes the attention is bound by its arithmetic, not by reading the
cache.

| run (`tools/stock-llama/bench.sh`, median of 3, H4's prefilled state, H9 and H10 interleaved) | load | Cpu, Ppu, Apu, Control ms per pass | hours to frame 300 | check |
| - | - | - | - | - |
| H9, round 1 | 3.5 | 35.4, 48.3, 26.1, 24.4 | **12.9** | 0 errors over 903 events |
| H10, round 1 | 2.3 to 2.9 | 35.7, 48.2, 27.5, 24.4 | 13.1 | 0 errors over 903 events |
| H9, round 2 | 2.1 to 2.7 | 35.3, 48.5, 26.2, 24.1 | **12.9** | 0 errors over 903 events |
| H10, round 2 | 2.0 to 2.9 | 35.7, 48.3, 27.5, 24.6 | 13.1 | 0 errors over 903 events |

H10 is exact and slower: 13.1 hours against 12.9 in both rounds, with the APU 1.3 to
1.4 ms slower and the GBC CPU 0.3 to 0.4 ms slower. Not kept, so `run.sh` still builds H9's
stack. The shift patch makes no measurable difference in this window. A run to frame 300
writes far more often than these 300 passes do, so it could matter there, but the
frozen metric cannot see it, and only a longer window could decide it. If the fold comes
back, it needs the mask skip kept (each folded block's tokens map to one block of `blk`
when 22 ≥ 8) and the 8×8 kernel would still have to gain from it, which the `timing`
rows above say it does not.

**Where an H9 pass goes (measured by ablation).** `tools/stock-llama/metal-prof-skip.patch`
is for timing only. Under `GGML_METAL_PROF_SKIP` it leaves out the Metal kernels of the ops
it names. An entry is an op (`FLASH_ATTN_EXT`), a tensor-name prefix (`Qcur`), or both
(`MUL_MAT:node` selects the unnamed matmuls, which are the FFN slices' gate and down
projections). `GGML_METAL_PROF_LIST=1` prints every encoded node's op, name and shape,
which is how to find the names. The tokens such a run decodes are garbage, but the time is real.
`stock-llama timing` on H9's model ran in one GPU lock (median of 5, load 2.4 to 3.1).
Each cell below is the time saved by leaving that set out, in ms per pass. The last two
rows are the full pass and the floor with every op left out, which is graph build, input
upload and the mask, command-buffer submit and wait.

| left out | CPU 89, 7,300 cells | CPU 89, 2,091 | PPU 170, 2,091 | PPU 170, 7,300 | APU 120, 2,091 | control 82, 2,091 |
| - | - | - | - | - | - | - |
| flash attention (15 kernels) | 11.6 | 3.4 | 5.6 | 17.3 | 3.8 | 3.4 |
| FFN slices (gate and down, 28 matmuls in the GBC CPU pass) | 8.0 | 6.8 | 19.7 | 20.4 | 7.8 | 5.3 |
| attention projections (Q, K, V, O, 60 matmuls) | 6.6 | 6.3 | 8.2 | 7.4 | 6.6 | 5.0 |
| LM head (`result_output`, 36,311 rows: the sizes table's 36,299 predates the 12 `MAP` slots) | 4.7 | 3.8 | 7.9 | 8.5 | 5.0 | 3.9 |
| norms and elementwise (RMS norm, add, mul, SiLU, RoPE, set and get rows) | 1.8 | 1.4 | 2.2 | 0.7 | 1.2 | 1.4 |
| full pass | 30.7 | 22.9 | 44.3 | 55.5 | 26.2 | 19.9 |
| floor, every op left out | 2.6 | 1.3 | 2.0 | 4.7 | 1.5 | 1.1 |

At 2,091 cells the five savings and the floor add up to the full pass within 1.3 ms. At
7,300 they add up to 3.5 to 4.6 ms more than it, because the encoder runs independent
kernels concurrently and leaving one out frees the GPU for the others. The bench's GBC CPU
pass (35.3 ms) is 32.6 ms inside `llama_decode` and 2.7 ms in the harness around it (the
play's counts line). Its decode spans the whole cache's 141,415 cells under the scope's
mask, which accounts for the 1.9 ms over `timing`'s 30.7 (inferred, not split further).

- The GBC CPU's pass at its scope is 38% attention kernels, 26% FFN, 21% attention
  projections, 15% LM head, 6% elementwise and 8% floor (fractions of the full pass; they
  overlap by the 4.6 ms above).
- The PPU's pass at its scope (2,091 cells) is 44% FFN, then the projections and the LM
  head at 18% each. Its 160 pixel rows share per-pixel weights, which is why the PPU keeps
  the most FFN units.
- The LM head scores every one of the 36,299 tokens at every decoded row, while an output
  position decodes a byte (`tok::output_value`).

The next levers, by estimated gain (estimates from the ablation rows, not measured):

1. **An LM head per part.** Score only the tokens a stream's outputs can decode, as
   H8 split the FFN: a per-part `output.P` beside `ffn_route`. The ceiling is the whole
   LM head row: about 4.2 ms × 981,204 GBC CPU passes, 7.9 × 83,501 PPU, 5.0 × 243,952 APU and
   3.9 × 59,052 control, about 1.7 hours of 12.9 (13%). Exactness needs a proof that no
   left-out token can win the argmax, or the rows compared on the kept columns.
2. **Attention heads per part.** Q and O as per-part slices of the heads each part's
   rows read, which shrinks the projections (5 to 8 ms a pass) and the attention kernels'
   query rows (3.4 to 11.6 ms) in proportion. The gain depends on a head census per part,
   which has not been taken. At half the heads, the GBC CPU alone would save about 9 ms of its
   18.2, about 2.5 hours (a guess until the census runs).
3. **The per-pass floor.** 2.7 ms of harness around the decode plus 2.6 ms of floor inside
   it, about 5.3 ms of the GBC CPU's 35.3, or 1.4 hours at the GBC CPU's pass count. Halving it
   would save about 0.7 hours (a guess: the split between the mask over the full cache, the
   graph build and the harness's own work is not measured).


**One LM head range per part (H11, measured, kept).** H9's LM head scores all 36,311 token
ids at every decoded row, and the ablation above puts it at 3.8 to 8.5 ms a pass. A part's
rows can win only a few of them. Here a ubatch scores one run of rows of `output.weight`,
chosen by the same part as its FFN slice. `export one-model DIR --layer-shapes --align 64
--slices --no-up --part-heads` writes the runs, and `tools/stock-llama/lm-head-parts.patch`
(on top of `ffn-no-up.patch`) runs them. No circuit, `Model` or digest changes, and no tensor
changes: H4's prefilled state loads as it is.

- **What a row can win (`one_model::head_check`, asserted on the compiled weights before the
  export writes).** A logit is `lm_head[t] · h`, where `h` is the final norm's output. Its
  channel 0 is positive at every row, because every token embeds it as `UNIT` and nothing
  writes it. Every LM-head row reads only channel 0 and the 12 route channels. Every token
  without a route pattern (inputs, masks, `REC`, `CELL`, `RD`) has the row `−1024·e_0`. The
  256 byte rows are ±1 on the eight value bits, −1 on the four family bits, and count their
  zero bits on channel 0. So where the family bits are 0, the best byte scores at least
  `8·h_0` whatever the value bits hold, and every patternless token scores `−1024·h_0`. Every
  family token's row differs from its byte twin's (the same value bits) only on the family
  bits and on channel 0, where the twin is higher. So where the family bits are 0, the twin
  scores strictly more. For the PPU, APU and control, nothing writes a family bit at their
  rows: no token embeds one, no head writes one, and no unit of the part's slice writes one
  at any layer. Their family bits are exactly 0, so a byte wins.
- **The runs (`one_model::head_range`).** PPU, APU and control: the 256 output bytes, rows
  32,125 to 32,380. GBC CPU: every token from the first output byte up to `REC`, rows 32,125 to
  36,300 (4,176 rows: the bytes, the 336 output masks and the 14 families × 256). The GBC CPU's
  write slots and its reader records' `RD` rows decode family tokens, so its run holds every
  pattern token, and only patternless tokens are left out. Its route bits are the dense
  pass's 0/1 values (EXACT under H6's walk, since the LM head reads them), so its exact
  pattern scores 12 against their −1024. The masks inside the GBC CPU's run cost 336 rows and
  keep it one run.
- **Why the choice is the same.** The harness decodes a row as its first largest score
  (`argmax`; llama.cpp only returns rows here and samples nothing). Every run is ascending
  and holds the full row's winner, so the first largest score in the run is the first largest
  score over every token id. The winner is also unique: the twin argument leaves a margin of
  at least 1, and the patternless tokens trail by more than 1,000. Every product and partial
  sum of the LM head is an exact integer (§10: at most 12 route channels, quantum 1), so
  which Metal kernel runs a narrower matmul does not change a bit.
- **The file.** `--part-heads` adds two metadata arrays, `gbcformer.head_first` and
  `gbcformer.head_count`, one i32 per part. Its tensor data is byte-identical to H9's file
  (`cmp` from each file's data offset, 3,023.6 MB), so H9's exactness evidence holds for every
  tensor.
- **The patch.** The loader reads the two arrays. The graph scores a ubatch of part p with a
  view of `output.weight`'s rows `[head_first[p], head_first[p] + head_count[p])`, and a
  ubatch no part claims with every row, as before. Graph reuse already requires the same
  part (`ffn-slices.patch`'s `can_reuse`). The context copies `n_outputs × width` floats
  instead of `n_outputs × n_vocab`, strides `llama_get_logits_ith` and `output_reorder` by the
  width, and asserts that every ubatch of a batch scores the same run. A new call,
  `llama_get_logits_cols(ctx, &first)`, returns the width and the first token id.
- **The harness.** A logit row is `cpp::Row { first, v }`: the scores of token ids
  `first..first + v.len()`, decoded by `Row::token`. `build.rs` compiles `shim.c` with
  `-DLLAMA_HAS_LOGITS_COLS` when the tree's `llama.h` declares the call, so the same source
  builds against stock trees, which return full rows.
- **The dump format (`STOCK_ROWS`).** A pass of full rows keeps its header (pass, rows,
  vocab). A pass of narrowed rows has the header (pass, rows, width | 2^63, first), then
  `width` f32 per row. `rows-diff` compares each pair of rows on the narrower row's columns,
  bit for bit. Against a full row it also holds the full row's choice over every token id to
  the narrowed row's, and reports how far the run's best leads every column left out.

| check | load | result |
| - | - | - |
| `head_check` on the compiled one model (`export --part-heads`) | 4.8 | pass |
| tensor data against H9's GGUF (`cmp -i 1092224:1092352`) | | identical |
| `examples/moe_check` with `MOE_GGUF` on the part-heads file (each decoded row's run read from the file and held to `head_range`; the first largest logit over the run against the one over every row, in the sliced and the dense row; the run's lead over every row left out), 12 cases per stream | 3.7 to 8.7 | 1,944 PPU, 864 GBC CPU, 720 APU and 504 control decoded rows: 0 logit rows differ from the dense pass, 0 choose otherwise over the run; the run leads by at least 1 (PPU, APU, control) and 1,036 (GBC CPU) |
| `rows-diff` of H9's full rows against the narrowed rows over 300 passes (`STOCK_ROWS=…:1` on both sides, one `bench.sh` run each, each on its own binary) | 8.0, 4.3 | 300 passes, 20,020 logit rows, 0 differ on the narrowed columns (max diff 0); 0 of 20,020 choose otherwise than H9's full row; the run leads every column left out by at least 1; 301 events equal the engine record on both |

`stock-llama timing` (Metal, each model on its own tree, each stream's group as its pass holds
it, every token's row read, median of 5, ms per pass; H9 and the part heads interleaved in
one GPU lock, load 6.8 to 4.6, falling after another job's spike to 43.6):

| model | cells | PPU 170 | CPU 89 | APU 120 | control 82 |
| - | - | - | - | - | - |
| H9 | 2,091 | 44.2 | 23.2 | 26.4 | 20.1 |
| part heads | 2,091 | 35.9 | 19.5 | 20.9 | 16.0 |
| H9 | 7,300 | 55.3 | 30.3 | 34.5 | 26.8 |
| part heads | 7,300 | 46.2 | 26.2 | 28.5 | 22.4 |

At 7,300 cells the GBC CPU's decode loses 4.1 ms, against the 4.7 ms the ablation gave the LM
head's kernel there. The rest of what the head cost stays in the kernel's 4,176 rows. The
PPU loses 9.1 ms, more than the kernel's 8.5. The likely rest (inferred, not isolated) is the
copy the context no longer makes: 170 rows of 36,311 floats (24.7 MB) from the device buffer
each pass.

| run (`tools/stock-llama/bench.sh`, median of 3, H4's prefilled state, H9 and the part heads interleaved) | load | Cpu, Ppu, Apu, Control ms per pass | hours to frame 300 | check |
| - | - | - | - | - |
| H9, round 1 | 2.6 to 3.0 | 35.1, 48.1, 25.9, 24.6 | 12.8 | 0 errors over 903 events |
| part heads, round 1 | 3.6 to 5.0 | 29.9, 38.6, 23.0, 21.6 | **11.0** | 0 errors over 903 events |
| H9, round 2 | 2.2 to 3.7 | 35.3, 48.4, 26.2, 24.0 | 12.9 | 0 errors over 903 events |
| part heads, round 2 | 2.4 to 43.6 (another job's spike in runs 2 and 3) | 30.0, 38.8, 23.2, 21.4 | **11.0** | 0 errors over 903 events |

The part heads are exact and take 14% off the projection, 12.9 hours to 11.0, and beat H9 on
every stream in both rounds. The part heads' two rounds agree within 0.2 ms on every stream,
the spiked round included; H9's agree within 0.3 ms but for control, which moved 0.6 ms, the
spread H9's own rows show. Kept: `run.sh setup` builds the slice stack with
`lm-head-parts.patch` on top, and `run.sh export` writes `--part-heads`.

Where the bench's saving goes (one play of each binary as `bench.sh` runs it, two
interleaved pairs, load 4.8 to 3.0, ms per pass as the play's counts line splits it; the
pairs agree within 0.6 ms):

| stream | H9 total, inside decode | part heads total, inside decode | saved in decode | saved in the harness |
| - | - | - | - | - |
| CPU | 35.2, 32.6 | 29.9, 28.9 | 3.7 | 1.6 |
| PPU | 48.1, 46.3 | 38.8, 38.0 | 8.3 | 1.0 |
| APU | 26.1, 25.0 | 23.3, 22.5 | 2.5 | 0.3 |
| control | 24.7, 20.4 | 22.1, 18.0 | 2.4 | 0.2 |

The harness's share is the rows themselves: each decoded row was 36,311 floats that
`cpp::LlamaCpp::decode` copied, `runtime::ordered` copied again and `argmax` scanned, and is
now 256 (4,176 for the GBC CPU). The GBC CPU's harness time falls from 2.6 to 1.0 ms a pass, so the
per-pass harness time in lever 3 above was more than half logit rows.

Gaps. H8's two carry over: no reader-record (`RD`) row runs in the bench window or in
`moe_check`, though `RD` rows decode `Line` family tokens, which the GBC CPU's run holds by
`head_check`'s range assertion; and no prefill has run under the slice model. The GBC CPU's run
rests on its route bits being the dense pass's 0/1 values (H6's EXACT requirement), not on a
weight-level bound like the other parts' runs: a route value outside [0, 1] at a GBC CPU row
could in principle score every kept token below −1024, which `moe_check` (lead 1,036) and
`rows-diff` (0 of 20,020 choose otherwise) never saw. A batch whose ubatches score different
runs aborts in the patch instead of decoding; no pass of the harness splits into two ubatches.

**Attention heads per part: the census (H12, census only, not built).** H6's walk already
decides, per part, which heads its rows need: a head writing any channel the part's rows
need as more than FREE gets an EXACT query. `one_model::Experts::heads` now keeps that list
(no other output of the walk changes), and `cargo run --release --example head_census`
prints it against the heads `--layer-shapes` keeps (every head with a nonzero `o_proj`
column). A head a part does not need adds only to channels nothing but the norm reads at
that part's rows.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | sum |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| kept | 20 | 22 | 17 | 2 | 12 | 12 | 15 | 12 | 14 | 12 | 6 | 6 | 5 | 2 | 0 | 157 |
| PPU | 16 | 22 | 17 | 2 | 12 | 12 | 14 | 12 | 12 | 12 | 6 | 6 | 5 | 0 | 0 | 148 (94%) |
| CPU | 19 | 22 | 17 | 2 | 12 | 11 | 11 | 4 | 5 | 0 | 0 | 0 | 0 | 2 | 0 | 105 (67%) |
| APU | 16 | 22 | 17 | 2 | 12 | 9 | 6 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 87 (55%) |
| control | 16 | 22 | 17 | 2 | 12 | 5 | 4 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 82 (52%) |

Layer 14 has no head with an `o_proj` column; the export keeps one there anyway. Layers 1 to 4
need every head for every part, so per-part slices would shrink only layers 0 and 5 to 13.

The estimate (inferred from H9's ablation rows above, not measured). Q and O hold 157 query
heads of 64 rows each, against one KV head of 64 rows for K and V per layer, so they are
about 91% of the projections' rows. If the projections and the attention kernels scale with
the heads a part keeps, a part keeping a fraction f saves about (1 − f) × (0.91 × projections
+ attention kernels):

| stream | f | ablation rows used (ms) | saved per pass (ms) | passes to frame 300 | hours |
| - | - | - | - | - | - |
| CPU | 0.67 | 6.6 + 11.6 (7,300 cells) | 5.8 | 981,204 | 1.6 |
| PPU | 0.94 | 8.2 + 5.6 (2,091) | 0.8 | 83,501 | 0.02 |
| APU | 0.55 | 6.6 + 3.8 (2,091) | 4.4 | 243,952 | 0.30 |
| control | 0.52 | 5.0 + 3.4 (2,091) | 3.8 | 59,052 | 0.06 |

That is about 2.0 hours of H11's 11.0, nearly all of it the GBC CPU's. Three things could make it
smaller. The kernels at a layer of 1 to 5 heads may cost more than their share. Layers 1 to 4,
where every part needs every head, may weigh more than the sum's average. And a part's slice
would need its own head count per layer in the graph, which `layer-shapes.patch` gives per
layer, not per part. Two proof gaps come before any build. The census is conservative: a head
counts as needed when it writes any channel the part needs as more than FREE, even where its
value there is 0. And H6's norm bound adds each head's value interval, so a part that skips a
head adds 0 instead, which must lie inside that bound or the bound must be redone without the
head.

This is `proof.sh` as it stood when this page was written, on a model and patches that no longer exist
(`lm-head-parts.patch` and the others of this page are gone from `tools/stock-llama`). Today
`proof.sh` runs in chunks with checkpoints and resumes, `PROOF_PURE=1` runs unpatched
llama.cpp over the uniform file, and its counts are those of `docs/perf-log.md`, "The proof of
record on `ff86dde7`" (`docs/stock-llama.md` §13 and §14; the script's header).

**The end-to-end run (`tools/stock-llama/proof.sh`).** The bench measures 300 passes. The proof
runs Yellow (CGB) from the hand-off state (no boot ROM) to frame 300 with every component a stream of H11's model, every
pass decoded by llama.cpp (the slice stack with `lm-head-parts.patch`). It does not load H4's
saved state. It prefills the cartridge's 141,415 cells under H11's model, which closes the
"no prefill under the slice model" gap above, and saves that cache in its work directory.

- The reference is the engine's record of the same run: `GBCFORMER_RECORD=<dir>
  stock-llama play none <rom> --frames 300`. It is Mac CPU-only, takes about 6 minutes and
  5.1 GB, and reaches frame 300 at `cfd31f3d89f1…`. It holds 913,782 GBC CPU, 83,500 PPU,
  243,951 APU and 59,051 control decoded passes, plus 67,421 GBC CPU write records (`keep`).
  `bench.sh`'s frozen GBC CPU count of 981,204 adds the write records to the decoded passes, so
  its hours overstate the GBC CPU's share by 7%.
- `STOCK_REFERENCE=<dir>` holds every decoded pass, as it decodes, to that record: the
  sequence, the positions and the tokens. The first difference, or a record that ends
  first, stops the run (`REFERENCE DIFFERS`, `REFERENCE ENDS`). `--lockstep all` runs the
  host references beside it. The run records itself too (`rec-run`), and after the lock is
  released `proof.sh` holds every event of that record, the write records included, to the
  reference by kind, sequence, positions and tokens.
- The log prints each frame's hash with a Unix time, and the counts every 10,000 passes
  (`STOCK_PROGRESS`). A pass's time leaves out the prefill.

```sh
nohup tools/stock-llama/proof.sh > /dev/null 2>&1 &   # takes and holds the GPU lock; log in tools/stock-llama/.build-proof/proof.log
tools/stock-llama/proof-status.sh                     # frame, passes, elapsed, ETA, errors; on completion the hash against cfd31f3d
```

`PROOF_WORK`, `PROOF_STATE`, `PROOF_REFERENCE` and `FRAMES` redirect a run, as the smoke
runs below did.

| smoke run (H11's model and tree, inside the GPU lock) | result |
| - | - |
| H4's saved state, `FRAMES=1`, against the five-frame record with GBC CPU event 2's first token changed | stopped at GBC CPU pass 3: `REFERENCE DIFFERS`, reference 932125, runtime 32125; exit 1 |
| prefill under H11's model, then `FRAMES=1` against the 300-frame record | prefill 555.4 s (2,137 cells one by one, 139,279 in lanes); the saved cache is byte-identical to H4's state (`cmp`, 544,730,976 bytes; the same cells sidecar); 12,083 passes (GBC CPU 8,766, PPU 288, APU 812, control 2,217), every one equal to the record; frame 1 at `d043ab24…`, the record's; the run's own record, write records included, equals the reference's first 12,168 events; exit 0 |

So a prefill under the slice model writes the same cache as H4's, bit for bit, and H4's state
was frame 0 of this run. The full run still prefills, so its log covers boot.
Projected from the reference's pass counts at H11's bench rates (29.9, 38.6, 23.0, 21.6 ms),
the full run takes about 10.4 hours plus the 9-minute prefill (inferred before the run).


**The run (measured).** The proof took the GPU lock on 2026-11-04 and exited 0 after
63,660 s (17.7 hours), with the prefill's 556 s included (141,415 line cells: 2,137 one by
one and 139,279 in lanes). Frame 300 hashes to `cfd31f3d89f1…`, the engine's hash. Every
decoded pass equalled the engine's record as it decoded: 913,782 GBC CPU, 83,500 PPU,
243,951 APU and 59,051 control passes, at 31.1, 38.3, 23.3 and 18.9 ms per pass. After
the lock was released, each stream's whole record, write records included, equalled the
reference event for event (981,203 GBC CPU events, then 83,500, 243,951 and 59,051). The
projection above counts decoded passes only. The GBC CPU's 124,621 write records and
46,912 reader records make up most of the difference: `docs/perf-log.md` ("Projected
frame-300 proof") times them at 4.4 and 2.1 hours.

That run used the model before the overlay and a patched llama.cpp. The 300-frame run in
unpatched llama.cpp on Metal, on `45bfb056`, is in `docs/perf-log.md`, "The proof of
record"; what has run on the weights since is in `docs/stock-llama.md`, "What runs where".
