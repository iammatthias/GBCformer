# Architecture census: where size and per-pass work go

**Snapshot of 2026-11-04.** Every number below was measured or
estimated on that day's tree and is kept as it was, except the 64-channel note in §2, which
was added that night. Later rounds changed the sizes, the head counts and the
rounding. Hidden now rounds to 64, not 1,024 (`docs/hardware-shaped.md` §15.1). On
2026-11-06 `shape` measures the APU at 70.2 M and the PPU at 254.1 M with 57 memory heads,
and control is 43.8 M (`docs/perf-log.md`, "Circuit: ports at rest"). For current
figures, run `cargo run --release --example shape`.

This document explains where the compiled Game Boy spends its parameters and its per-pass
work, and what each candidate redesign would change. It is an explanation, not a plan,
and nothing here is built. Every number is labelled "measured" (this branch, a command in
§9) or "estimated" (arithmetic on measured inputs, with the assumption named), and
older figures cite the document that measured them.

The short answer, in numbers:

- **The one model pays for all four parts on every pass.** Its hidden size, 5,120, is the
  sum of the parts' live wires (4,324) plus 796 shared channels (measured). The GBC CPU,
  which is 72% of all passes and 74% of the llama.cpp hours, compiles to hidden 2,048 on
  its own and needs 1,664 channels before rounding (measured), and its rows still compute
  157 attention heads where its own checkpoint has about 18 (estimated from the measured
  head list).
- **Hidden is rounded up to a multiple of 1,024 although the stock form no longer needs
  it.** Before rounding the parts need PPU 2,329, GBC CPU 1,664, APU 973 and control 823
  channels (measured, `one_model`); compiled they are 3,072, 2,048, 1,024 and 1,024. The
  PPU, the largest dense item, would fall from 922.2 M to about 711 M at 64-channel
  rounding (estimated with `Config::params`).
- **The parts on their own sum to 1,237.6 M dense parameters; the one model is 2,057.9 M
  (both measured).** Nonzeros are nearly the same, 1,154,738 against 1,092,918 (measured,
  layers and LM head). The size problem is padding, not logic, since 99.95% of the one
  model's layer weights are zero.
- **The overlay of `docs/one-model.md` §3 no longer projects to hidden 2,048.** The PPU
  now keeps 1,802 wires live, so the null-token overlay needs 2,602 channels (measured
  projection). That is 3,072 channels and 1,234.7 M dense as `one_model` rounds it, or
  about 2,624 and 1,055 M at 64-channel rounding (estimated), not 685 M.
- **One network can still give each part about its own width.** A *prefix overlay*
  (§5.4) orders the overlay's channels so that every part's wires and the shared
  channels sit in a prefix, and a stream runs only its prefix, its own heads, its own
  FFN slice and its own layers. It projects to 2.8 to 3.6 llama.cpp hours, as fast as
  four separate checkpoints, in one network of about 867 M dense (estimated), and a
  run with one part in weights pays about that part alone.
- **Memory in weights doubled the GBC CPU circuit.** The host-fed GBC CPU has depth 5,
  6,012 units and 1,000 live wires. The GBC CPU with memory has depth 10, 10,959 units and
  1,240 live wires (both measured). Address translation alone is 4,712 units, 43% of the
  circuit (measured).
- **About 70% of the engine's per-pass gate updates serve memory access.** Of 2,087
  updates per GBC CPU pass on the text workload (measured), 702 are the opcode decode from
  the fetched line (a few dozen of them CB fields, lengths and cycle counts that the
  host-fed GBC CPU also has), 312 byte selects, 227 the write-line merge and 210 address
  translation. The GBC CPU's own logic (adders, operand muxes, ALU, output fields) is
  26%.
- **PC's low three bits cause almost a quarter of the updates.** Each changes on about
  0.6 of passes, and every opcode-nibble unit reads them, so `PCL.0`, `PCL.1` and `PCL.2`
  alone cause 486 updates per pass, 23% of the total (measured).

## 1. Two cost models

The checkpoint runs on two kinds of runtime, and they pay for different things. Every
table below keeps them apart.

| | Stock runtimes (llama.cpp, MLX, Core ML) | Our sparse engine (`src/engine`) |
| - | - | - |
| What a pass costs | Dense work per row: projections ∝ hidden × heads × layers, FFN ∝ hidden × units, attention ∝ heads × cells, LM head ∝ hidden × scored rows, plus a fixed per-pass floor | Circuit work per step: channel changes, gate updates, sends, layers walked |
| What zeros cost | Full price | Nothing |
| What a row costs | Every row computes every layer (the CPU pass has 89 rows) | The shared layers run once per pass, then only changed outputs decode |
| Levers that move it | Fewer hidden channels, fewer heads per row, fewer units per part, fewer cells per head, fewer rows | Fewer changed channels, fewer readers per changed channel, fewer levels |
| Levers that do not | Fewer gate updates (zeros are paid anyway) | Hidden size, head count, vocabulary (the engine already prunes each stream to its own units: CPU stream 10,444 units, its own checkpoint 10,461, measured) |

The last row matters most. Shrinking hidden or splitting the one model cannot speed
up our engine, and shrinking the circuit's update count barely moves llama.cpp. Only
circuit changes that remove both units and updates help both runtimes.

## 2. Size per part and of the one model

`cargo run --release --example one_model -- --build` measured these figures on that
tree (the example and `Model::config` have since been removed; `Config::params`; nonzeros are `Engine::nonzeros`, which
counts the layers and the LM head but not the embedding).

| Checkpoint | Depth | Live wires (widest) | Circuit units | Channels before rounding | Hidden | Intermediate | Layers | Heads per layer | Vocab | Dense | Nonzero |
| - | - | - | - | - | - | - | - | - | - | - | - |
| PPU | 14 | 1,802 | 31,073 | 2,329 | 3,072 | 5,504 | 15 | 20 | 6,090 | 922.2 M | 785,126 |
| CPU + memory | 10 | 1,240 | 10,959 | 1,664 | 2,048 | 1,792 | 11 | 6 | 8,291 | 175.3 M | 166,392 |
| APU | 13 | 771 | 9,447 | 973 | 1,024 | 960 | 14 | 8 | 15,736 | 90.1 M | 150,077 |
| control | 11 | 659 | 3,943 | 823 | 1,024 | 512 | 12 | 5 | 10,578 | 50.0 M | 53,143 |
| four checkpoints, summed | | | 55,422 | | | | | | | **1,237.6 M** | **1,154,738** |
| one model (built, digest `b03d896c`) | 14 | 4,324 | | 5,120 | 5,120 | 6,336 | 15 | 22 | 36,311 | **2,057.9 M** | **1,092,918** |

`examples/stock` (since removed) counted 1,485,705 nonzeros for the one model because it adds the
embedding (`docs/stock-llama.md` §12a). The figures in `docs/one-model.md` §3 (PPU
552.9 M, APU 98.7 M, control 44.1 M, 778.3 M summed) predate the f16 legalisation and
the stock form; the table above replaces them.

**The rounding.** "Channels before rounding" is the shared channels plus the widest live
wires (`1024 − Model::wire_budget(c, memory, 1)` plus the live count, printed by
`one_model`). `Model::shape` then rounds hidden up to whole blocks of 1,024
(`d = 1024 * blocks`), which the old normalisation trick needed (`n_const` constants of
2^20, `d = 1024·n_const`) and the stock form's `rms_norm_eps` domination does not
(`docs/stock-llama.md` §2: one constant channel, "hidden is no longer a multiple of
1,024"). The layout already has `n_const = 1` (`CONSTANTS.len()`); the only remaining
`hidden / 1024` is in `one_model`'s overlay projection. Rounding to 64 instead would
cost nothing on the engine and, by `Config::params`, take the PPU from 922.2 M to about
711 M, the GBC CPU from 175.3 M to about 142 M and control from 50.0 M to about 41 M
(estimated). The overlay work checked that nothing needs whole blocks and built the
change, and every checkpoint now rounds to 64 (`docs/hardware-shaped.md` §15.1; measured
PPU 254.1 M, GBC CPU with memory 137.0 M, control 40.6 M).

**How the one model's dimensions combine its parts (measured, `one_model --build`).**

| Dimension | Rule | Value | What it costs the GBC CPU's pass |
| - | - | - | - |
| hidden | sum of the parts' live wires, plus shared channels (constants, 461 slot identities, own bits, the memory block, routes, lanes) | 4,324 + 796 = 5,120 | 2.5 times its own 2,048 in every projection, norm, FFN row and LM-head row |
| layers | the deepest part (the PPU) | 15 | 15 layers against its own 11 |
| heads | the most heads any layer needs (the PPU's memory fetches) | 22 per layer, 157 with weights | 157 heads per row against about 18 in its own checkpoint |
| intermediate | the widest scheduled level | 6,336 | none since H8: each part runs its own FFN slice |
| vocabulary | 257 tokens per input slot, plus outputs and families | 36,311 | none since H11: each part scores its own LM-head range |

**The concatenation also widens each part's FFN slice and head set.** Each part keeps
the units and heads its rows need to stay exact, and in the concatenated model that
includes other parts' units and heads whose channels it shares (inferred, because
`Circuit::fit` places wires of different parts in the same channels, so a foreign head or clearing
unit can write a channel a part later reads).

| Part | Own units (`shape`, routing and clearing included) | Units in its one-model slice (H6, `docs/slice-stack.md`) | Heads its rows need in the one model (H12) | Heads in its own checkpoint |
| - | - | - | - | - |
| CPU | 13,379 (measured) | 22,463 | 105 of 157 | about 18: 6 at layer 0, 12 memory and copy heads (measured list) |
| PPU | about 31,500 (estimated from 31,073 circuit units) | 40,615 | 148 | about 129: 127 memory heads (measured) and its layer-0 gathers |
| APU | 11,473 (measured) | 21,142 | 87 | 8 (layer-0 gathers only; no memory head, measured) |
| control | about 4,200 (estimated) | 14,188 | 82 | 5 (layer-0 gathers only; no memory head, measured) |

The H6 widths come from an earlier one-model digest. They show where the extra units and
heads fall, not today's exact counts.

**Where the parameters sit.** The embedding and the LM head hold `2 · vocab · hidden`
parameters, 371.8 M of the one model's 2,057.9 M (18%, estimated from the formula). The APU's own vocabulary,
15,736, is 60 input slots × 257 tokens; its embedding and LM head are 32.2 M of its
90.1 M. A stock runtime only gathers embedding rows, so this is file size, not time.

## 3. The GBC CPU circuit

Both runtimes spend most of their time on the GBC CPU. It takes 981,204 of 1,367,709
passes to frame 300 (72%) and 8.1 of llama.cpp's 11.0 hours (H11, `docs/slice-stack.md`),
and it accounts for the whole of the engine's 60 fps gap (`docs/harness.md` §9.4).

### 3.1 Static: units, levels and live wires

Measured with `cargo run --release --example levels -- --table cpu` (memory mode, the
form the one model uses) and `--table host` (the host-fed GBC CPU). The families are
grouped into categories by hand from their node names; the command prints the full
family tables.

| Category | Families | Host-fed GBC CPU units | GBC CPU with memory, units | GBC CPU with memory, share |
| - | - | - | - | - |
| Address translation and physical selects | `F` (fetch), `RHL`, `RSP`, `RP`, `RRARE`, `WHL`, `WSP1/2`, `WRARE0`, `WA16`, `WP`, `MAP*`, `RFF*`, `WFFN` | 0 | 4,712 | 43.0% |
| Output field muxes with keep | `A'` .. `L'`, `PC'`, `SP'`, `F'`, `CTL'`, `W0/W1`, `WS`, `RQ` | 3,675 | 1,926 | 17.6% |
| 16-bit address adders | `pc+n`, `sp±n`, `pair±`, `hl+pair`, `sp+e8`, `a16+1` | 1,027 | 1,165 | 10.6% |
| Operand muxes | `src`, `x`, `sel` | 315 | 689 | 6.3% |
| Opcode decode | nibble patterns `OP0.lo*/hi*` (256), one-hots `OP0=xx` (256), CB fields, `len`, `MC`, jumps | 444 (one-hots arrive free as tokens) | 688 | 6.3% |
| Write-line merge | `WL*` (the line after a write) | 0 | 640 | 5.8% |
| ALU | `add`, `sub`, logic, `inc/dec`, `daa`, CB rotate, bit, res | 407 | 611 | 5.6% |
| Byte select from a fetched line | `OP1[i]`, `OP2[i]`, `D0[i]`, `D1[i]` | 0 | 320 | 2.9% |
| Address muxes (`RA`, `rare`) | | 144 | 208 | 1.9% |
| **total** | | **6,012** | **10,959** | |

| | Host-fed GBC CPU | GBC CPU with memory |
| - | - | - |
| depth | 5 | 10 |
| units per level | 941, 744, 1,361, 1,441, 1,525 | 139, 1,405, 988, 1,402, 849, 906, 1,405, 1,404, 1,404, 1,057 |
| live wires per boundary | 672, 842, **1,000**, 797, 594, 201 | 136, 247, 423, 555, 697, 921, 1,223, **1,240**, 1,061, 808, 511 |
| memory heads | 0 | 10 (layers 1, 1, 2, 3, 3, 6, 6, 6, 8, 8), plus 2 copy heads |

**Why an 8-bit GBC CPU needs ten levels.** The host-fed GBC CPU's five levels are the
chip's own logic, which is operand selection, the ALU, flags and one final mux per field
(`src/sm83/mod.rs`). The GBC CPU with memory adds five sequential memory steps around
them: translate the fetch addresses
and read the line (levels 1 to 2), pick the opcode's nibbles from the line and form the
256 one-hots (3 to 4), translate the data address and read (5 to 6), select the byte
(7), and read the written line and merge the byte into it (9 to 10).

**What sets the width.** At the widest boundary (7) the 1,240 live wires include 286
opcode wires (`OP0`: 256 one-hots plus nibbles), 192 line bits from the data heads
(`D0h`, `D0r`, `D1h`), about 200 translated addresses (`RP0`, `WRARE0`, `WSP1`, `WSP2`,
`WHL`, `RRARE`, `WA16`, `MAPS`, `MAPR`), and about 230 adder outputs (measured, `shape
--mem`). The one-hots stay live from level 4 to the output fields at level 9, because
every control signal is a free linear sum of them.

### 3.2 Dynamic: what changes per pass, and what it costs the engine

Measured with
`GBCFORMER_UNITS=1 GBCFORMER_WIRE_CHANGES=<file> CORE=model-mem tools/bench/text.sh`
and `overworld.sh`, one run each, on the `cpu-memory` checkpoint.
The census counts the scalar walk's gate updates; its time per pass (9.2 µs) is the
census build's, not a speed measurement. The one model's GBC CPU stream does 2,097
updates per pass on text (`docs/harness.md` §9.2), so the GBC CPU's own checkpoint is a
faithful stand-in.

| Category | Text: updates per pass | Share | Overworld | Share | Text: wire changes per pass |
| - | - | - | - | - | - |
| Opcode decode (nibble patterns read the line and PC's low bits) | 702 | 33.6% | 649 | 33.8% | 8.9 (+ about 2 one-hot changes) |
| Byte select from a fetched line (`OP1`, `OP2`, `D0`, `D1`) | 312 | 14.9% | 289 | 15.0% | 9.2 |
| Write-line merge | 227 | 10.9% | 177 | 9.2% | 9.5 |
| Address translation and physical selects | 210 | 10.1% | 170 | 8.8% | 13.2 |
| 16-bit address adders | 162 | 7.8% | 182 | 9.5% | 13.2 |
| Output field muxes | 152 | 7.3% | 153 | 8.0% | 12.6 |
| Operand muxes | 144 | 6.9% | 118 | 6.1% | 6.6 |
| ALU | 78 | 3.7% | 93 | 4.8% | 9.1 |
| Routing | 54 | 2.6% | 43 | 2.2% | |
| Address muxes (`RA`, `rare`) | 46 | 2.2% | 47 | 2.4% | 5.4 |
| Memory heads' line bits (head outputs) | | | | | 49.8 |
| Inputs (registers, PC, SP) | | | | | 4.6 |
| **total** | **2,087** | | **1,921** | | **144.5** |

Gate updates per layer, text (measured): 46, 63, **630**, 324, 149, 109, 172, 179, 115,
249, 54. Layer 2 is the opcode decode.

The largest single sources (measured, updates per pass caused by one changed wire):
`PCL.0` 184.6, `PCL.1` 182.7, `PCL.2` 118.6, `RP0.0` 48.2, `pc+1.lo.*` and `pc+2.lo.*`
about 110 each. PC's low three bits pick the opcode's byte within the fetched 8-byte
line, and all 256 nibble-pattern units read them, so every PC increment re-evaluates the
whole decoder. The `pc+1` and `pc+2` low bits do the same for the operand byte selects.

**The three change counts measure different things.** The 144.5 above counts changes of the named wires
in the `GBCFORMER_WIRE_CHANGES` table (wire channels, head line bits included). The same
run's stats line counts 218.3 (changed channel, reading layer) pairs per pass, and
`docs/harness.md` §9.1 counts 211 changes per pass on the one model's GBC CPU stream from its
per-layer change lists, which by that document's description are channel changes as each
layer sees them (inferred; it does not split them). None contradicts another.

The categories are `tools/census/categories.ts` (units and updates) and
`tools/census/changes.ts` (wire changes) over the commands' output.

## 4. The other parts (measured, `levels -- --table ppu|apu|ctl`)

| Part | Largest families (units) | Notes |
| - | - | - |
| PPU, 31,073 units, depth 14 | `bg` 4,074 and `wn` 4,074 (background and window), `si` 3,726 (102,220 weights), `fc` 2,560, `fo` 2,560, `so` 2,160, `pb`/`pw` 1,680 each, `oc` 1,280 (214,320 weights) | Five levels hold about 4,600 units each. Its 160 pixel positions share the per-pixel weights through position wires, which is why it keeps the most FFN units; 1,715 of its 1,802 wire channels hold a memory head's byte at some point |
| APU, 9,447 units, depth 13 | `mix` 936, `lfsr` 789, `RV` (readback) 512, `AL`/`AR` 242 each, sweep 442 | Flat: every level holds 600 to 811 units |
| control, 3,943 units, depth 11 | `pause_t` 209, `RV` 192, `s` 160, `t` 128, horizons `hppu`/`hirq`/`hdma`/`hser` about 470 together | Flat: 384 units per level |

The PPU is 8.4% of llama.cpp's hours (83,501 passes at 38.6 ms) and runs on its own
thread in the engine, so it is not on either critical path today. Its dense size, 922 M
alone, is the largest single item in any per-part design.

## 5. Projections

### 5.1 Method for llama.cpp (estimated)

The H9 ablation measured each component's time per pass: attention kernels, FFN slices,
the Q/K/V/O projections, the LM head (reduced to what H11 left), norms and elementwise
ops, and a floor (graph build, upload, mask, submit). Its passage is "Where an H9 pass goes"
in `docs/slice-stack.md`, and the patch was `tools/stock-llama/metal-prof-skip.patch` (since removed).
A variant's pass is projected as

`floor + harness + k · Σ component · ratio`,

where `ratio` is the variant's dims over the one model's for that component
(attention: heads × (cells + rows); projections: hidden × (heads + layers); FFN: hidden ×
units; LM head: hidden × scored rows; norms: hidden × layers), and `k` makes the formula
reproduce H11's measured decode time at ratio 1 (it absorbs kernel overlap). The floor
and the harness time stay fixed. The range's upper end puts a floor of 20 µs under every
kernel (a guess), because small matmuls do not shrink in proportion. Hours use the
frozen bench pass counts (GBC CPU 981,204, PPU 83,501, APU 243,952, control 59,052). The
script is `tools/census/project.ts` (`bun tools/census/project.ts`).

### 5.2 The candidates

The project prefers one model, with separate checkpoints kept as a
reference point. The design must also keep `--weights` and `Stack` as they are: all
components in weights, any mix, or one part alone with the rest on the host. The last
column says what a run with only the GBC CPU in weights pays.

| Variant | GBC CPU ms | PPU ms | APU ms | control ms | llama.cpp hours to frame 300 | Engine: GBC CPU updates per pass, µs per step | Dense | A run with only the GBC CPU in weights |
| - | - | - | - | - | - | - | - | - |
| H11, the one model today | 29.9 | 38.6 | 23.0 | 21.6 | **11.0** (measured) | 2,097, 4.8 (measured) | 2,057.9 M | the CPU's full pass: hidden 5,120, all 157 heads |
| H12, per-part head slices | 24.5 to 25.3 | 37.8 to 38.0 | 18.3 to 19.4 | 17.4 to 18.2 | 9.1 to 9.4 | unchanged | unchanged | hidden 5,120, 105 heads |
| (a) overlay, hidden 2,624, 15 layers, own heads and units per stream | 8.8 to 11.0 | 19.8 to 21.1 | 5.6 to 8.3 | 6.8 to 9.6 | **3.3 to 4.2** | unchanged | about 1,055 M | hidden 2,624, every layer |
| (a') prefix overlay: each stream runs its own prefix of channels, heads and layers | 7.3 to 9.2 | 18.6 to 20.0 | 3.7 to 6.8 | 5.6 to 8.3 | **2.8 to 3.6** | unchanged | about 867 M | hidden 1,792, 11 layers: about the CPU alone |
| (a') plus the CPU shrink d1 + d2 (§5.6) | 6.4 to 8.1 | 18.6 to 20.0 | 3.7 to 6.8 | 5.6 to 8.3 | **2.5 to 3.3** | about 1,500, about 4.0 | about 867 M | hidden about 1,344, 9 layers |
| (b) per-part dims in one GGUF | as (c) | as (c) | as (c) | as (c) | as (c) | unchanged | as (c) | the CPU's own tensors only |
| (c) four checkpoints, as compiled (reference) | 7.7 to 9.4 | 22.0 to 23.1 | 3.4 to 6.6 | 5.4 to 8.3 | 2.9 to 3.7 | unchanged | 1,237.6 M (1,154,738 nonzero, measured) | the CPU checkpoint alone |
| (c) four checkpoints, 64-channel rounding (reference) | 7.1 to 9.1 | 18.6 to 20.0 | 3.4 to 6.6 | 5.3 to 8.3 | 2.7 to 3.5 | unchanged | about 984 M | the CPU checkpoint alone |

Everything but the H11 row is estimated (`bun tools/census/project.ts`; dense sizes by
`Config::params`). The overlay, prefix and per-part rows assume that H8's FFN slices and
H11's LM-head ranges carry over and that a stream computes only its own heads. In an
overlay that holds by construction, because a foreign head attends to the null token and
writes 0, and a foreign unit's gate stays at most −1. In the one model it needs H12's proof
work. The engine column is unchanged for every layout because the engine already prunes
each stream to its own units (§1); only the circuit changes of §5.6 move it.

None of these touches the per-pass floor. The GBC CPU's 2.6 ms floor and 1.0 ms of
harness are 40% to 50% of its projected 7.3 to 9.2 ms, so below about 2.5 hours the
floor and the pass count decide.

### 5.3 (a) The overlay

This is the design of `docs/one-model.md` §3. Every part's wires share residual
channels, every MLP unit gains a gate term `−BIG·(1 − mode_own)`, clearing units are gated alike, layer-0
inputs must land on the same channels, and every memory head's query gains a term that
sends it to a null mode token in a foreign mode.

- **What it buys.** One network with one residual width, which takes hidden from 5,120
  to 2,602 channels (measured projection; 3,072 as `one_model` rounds it). Each stream
  computes only its own units and heads exactly, with no H6 or H12 walk, because foreign
  work is 0 by construction.
- **What it costs.** Every stream needs a mode token and a gather of it at layer 0, and
  every unit needs one more weight (about 55,000). `Segments::new` must accept a
  clearing unit that reads a mode channel. Every memory head of every part gets a
  null-token term, whose margin must hold in f16 next to the address margins of
  `docs/stock-llama.md` §4.
- **What breaks.** The exactness argument holds if BIG dominates every row's
  largest pre-activation over the declared ranges in every mode (a census like §3 of
  `docs/stock-llama.md`, at K = 128, |gate| ≤ 65,408). Every digest, the GGUF and every
  gate change. Engine speed is unchanged (the per-stream engine folds `mode_own = 1`
  into the bias, as the positional stage does).
- **Subset runs.** `--weights cpu` loads one file and runs only GBC CPU streams, and
  those pay the full width, 2,624, and every layer, 15, which is about 1.6 times the GBC
  CPU's own hidden and 1.4 times its depth. Tensors no selected stream reads stay untouched in the
  mmapped file (llama.cpp maps the GGUF; whether Metal keeps an untouched slice out of
  wired memory is not checked).

### 5.4 (a') The prefix overlay

This is the overlay with its channels, heads and layers ordered so that each part needs
only a prefix of them, plus a runtime that runs each stream on its prefix.

- **The layout.** The shared channels come first (the constant, the shared slot
  identities of `docs/one-model.md` §3's shared-slot variant, own bits, the memory
  block, routes, lanes and the mode channels, 505 + 4 in all), then the wires, laid out
  so that part p's wires fall below its width `h_p`. That width is about 1,749 channels
  for the GBC CPU, about 1,280 for the APU, about 1,168 for control and about 2,311 for
  the PPU (estimated from each part's live wires). The overlay already lets
  wires of different parts share a channel; the prefix rule only says which channels.
  Each part's heads are a contiguous range of each layer's heads, and each part routes at
  its own depth, so the GBC CPU's rows can stop after layer 10.
- **Why truncation is exact.** At a part's rows every channel above its prefix holds 0,
  because foreign units write exactly 0 under the mode gate and foreign heads attend the
  null token, whose value is 0. The norm sums squares over fewer channels, but its result is
  `eps`-dominated (`docs/stock-llama.md` §2: `mean(x²) ≤ 2^-14` against half an ulp of
  2^14), so the divisor changes nothing. Layers after a part's own routing layer hold no
  unit of that part and only foreign heads, so leaving them out leaves its residual as
  it was.
- **In llama.cpp.** It needs a patch in the family of `ffn-slices.patch` that gives each
  ubatch part views of each tensor's first `h_p` input columns (or per-part copies, as H8 copies the FFN
  slices), the part's head range of `wq` and `wo`, its own layer count, and its range of
  the LM head. The one KV head per layer is shared, and a cell's keys and values are the
  same whichever part reads it, so the cache stays one cache. Whether Metal's matmul
  accepts a row-strided view of `src0` at full speed is not checked; the copies avoid
  the question at the cost of file size.
- **Subset runs.** A run with one part in weights decodes only that part's streams on
  their prefixes. The GBC CPU runs at about 1,792 channels and 11 layers, about what the
  GBC CPU checkpoint alone costs (1,664 channels after 64-rounding; §5.2 projects 7.3 to
  9.2 ms against 7.1 to 9.1). The runtime does not read the other parts' tensors.
- **What breaks.** Everything (a) breaks, plus the layout rule (a constraint on
  `Circuit::channels` and `fit`) and per-part early exit, which the dense oracle must
  model too, so the oracle's pass of a stream stops at the stream's depth.

### 5.5 (b) Per-part dims in one GGUF, and (c) four checkpoints

(b) puts four networks in one file, each with its own hidden, heads, layers and
vocabulary. (c) is four files. They run alike, and (c) exists today. The four own
checkpoints pass their gates, and the engine ran them this way before the one model
(`--weights cpu,memory,ppu,apu,control` without `--one-model`).

- **What they buy.** They give the per-part rows of §5.2 and the smallest dense size
  (about 984 M at 64-channel rounding).
- **The cache.** A cell's keys and values come from the reading network's own K and V
  projections, so the GBC CPU and the PPU, which both read memory, need two caches. The
  PPU's holds its VRAM, OAM and palette lines and follows the GBC CPU's writes there
  through PPU record passes (the four-checkpoint engine does this through
  `engine::Memory::video`; the count of those passes in llama.cpp is not measured). The
  prefix overlay keeps one cache.
- **How hard (b) is in llama.cpp.** It is hard, because `n_embd` is a model-wide
  hyperparameter. The residual buffer, every norm, every tensor shape, the KV row width per layer, graph
  reuse and the output buffer assume it. (b) would register four architectures under
  one model and give each its own cache, which is (c) with extra loader work.
- **Subset runs.** Both load and run only the selected networks.
- **What breaks.** The exactness argument is unaffected. Under the ruling above, (c)
  stays a reference point.

### 5.6 (d) Shrinking the GBC CPU circuit

These change the compiler or the GBC CPU builder, so each needs every gate of
`CLAUDE.md` again: SST on circuit and checkpoint, engine equal to the dense pass, the
lockstep hashes, the f16 census, a re-exported and re-verified GGUF. The engine
estimates split the measured 4.8 µs GBC CPU step (`docs/harness.md` §9.4) into the gate
walk, 2.3 µs, scaled with updates; the change-driven loops (`drain_wires`,
`route_layer`, `advance`), 0.97 µs, scaled with wire changes; and the rest, 1.53 µs,
fixed. §9.1 of that
document warns that the walk does not shrink linearly when the cheapest updates go, so
these are optimistic.

**d1. Translation in the memory key.** Today the circuit translates every candidate
address (three fetches, HL, SP, the rare paths, both stack writes, a16, FF00+n, FF00+C)
to a 22-bit physical address in its own parallel copy, which saves a level and spends
4,712 units. Instead, key each cell by its logical line address plus the bank its window
uses, with dims that a cell outside a banked window leaves at 0 and the committed dim
evening out the margins. About 50 units at level 1 compute the effective banks (ROM
bank after MBC1's quirks and masks, SRAM bank, SVBK, VBK) once per pass from `REGS`,
and every head's query is then linear in the address wires and those bank wires.

- **Buys.** It saves about 4,500 units (41% of the circuit), about 200 live wires, one
  or two levels on the fetch and data paths (depth from 10 to about 8), and about 210
  updates and 13.2 wire changes per pass (10% and 9%). The engine's step falls from 4.8
  to about 4.5 µs (estimated).
- **Breaks.** It breaks the key format of `docs/stock-llama.md` §4 (19 line-address bits
  on RoPE pairs 10 to 20). The new key needs about 30 key dims, whose partial sums at
  S = 1,024 pass 2^15, so T2 needs S = 512 and a recheck of its margin. The host's eviction maps a decoded
  (bank, address) to a cell position; that must stay bit concatenation, not a mapper
  rule, or the host computes part of the translation. Banking-register writes, the I/O
  page's VOID cases and DMA-time translation still need a region decode. The scope rule
  (`MAP` slots) changes with the key.

**d2. Select the opcode byte first, then decode it by table lookup.** Today 256
nibble-pattern units each read the fetched line and PC's low bits. Instead, select the
opcode byte from the line as `D0` selects a data byte (128 units), and look up its
control signals with attention. A decode table of 256 (and 256 CB) prefilled tokens is
keyed by the opcode's bits and holds the control signals as values, and a few heads read
it at layer 3. The table sits in the cache like the cartridge's cells, and the
checkpoint's own embedding writes it.

- **Buys.** It saves about 300 to 450 updates per pass (PC's bits then reach 128 select
  units instead of 256 decoders), the 256 one-hot ANDs and the 256 one-hot channels
  (about 200 live wires net), and it adds no level (the head at layer 3 delivers the
  signals where the ANDs do now). The engine's step falls from 4.8 to about 4.4 µs
  alone; with d1, it takes about 1,500 updates, 126 wire changes and 4.0 µs (estimated).
- **Breaks.** It breaks the rule that every control signal is a free linear sum of
  one-hots (`src/sm83/mod.rs`). Signals become wires, so their count must fit the heads'
  64-dim values (not counted; it is the first thing a prototype must measure). A table
  head is exact by the same one-hot margin as a memory head. The project must decide whether
  a prefilled table counts as weights.

**d3. Control wires instead of 256 live one-hots.** Materialise each distinct control
form as a wire at level 5 and let the one-hots die there. It needs no table.

- **Buys.** It saves up to about 200 live wires (the GBC CPU's prefix in §5.4 falls from
  about 1,749 to about 1,550 channels) at a cost of one unit per distinct form. Updates
  stay about neutral (estimated).
- **Breaks.** The exactness argument is unaffected; it adds units, and the digest moves.

**d4. Byte-valued channels.** Every wire is 0/1 today, so a byte costs 8 channels, a
byte mux 8 units per case, a byte select 128 units, and a register change about 4
channel changes. A channel holding an integer 0 to 255 makes a mux case one unit
(`relu(v − 256·(1 − s))` is exact for 0 ≤ v ≤ 255 when the gate is a multiple of K),
makes adders linear (free), and lets one 64-dim head value carry 64 bytes instead of 8,
so a line cell could hold 64 bytes and Yellow's 141,415 cells would fall to about
17,700.

- **Buys.** It gains the most of any lever on both cost models (a guess until
  prototyped). Muxes and selects become about 8 times smaller, the GBC CPU's hidden and
  its wire changes perhaps halve, and attention over cells becomes about 8 times smaller
  in llama.cpp. The engine's step might fall to 2.6 to 3.1 µs, which makes d4 the only
  lever here that reaches text's 3.1 µs budget.
- **Breaks.** It breaks §2 of `docs/stock-llama.md` (every channel in [−1, 1], so the
  norm is exact). With |v| ≤ 255 the residual unit must fall to about 2^-13 and the norm weight
  rise, which f16 weight storage does not hold at 2^20, so the scale moves into the
  next layer's weights; T2 forms subnormals. §3's integer ranges hold (K · 256 = 32,768).
  Bitwise ops, BIT/RES/SET, rotates, half-carry and DAA need bits back: one staircase
  level per extraction (about 30 units per byte in nibbles) or nibble tables. The circuit
  IR (`Σ coef·[form ≥ θ]`) gains a value-node kind, and 64-byte lines make every write a
  64-byte read-modify-write. Every gate and digest rule applies. It is a new compiler,
  not a pass.

**d5. Byte cells for RAM only.** This stores the writable regions (WRAM, VRAM, OAM,
HRAM, about 50 KB) as one cell per byte and ROM as lines. It removes the write-line
merge (640 units, 227 updates) and the data byte selects for RAM, and adds about 44,000
cells (31% more attention in llama.cpp). Reads then need a region select between a line
and a byte. These figures are estimated, and d5 is not recommended before d1 and d2.

**Levers considered and left out.** Merging levels does not help, because the schedule
already balances units per level (`Circuit::schedule`) and data dependence makes the
memory steps sequential. Fewer rows per GBC CPU pass would need its own design. Every
one of the 89 rows computes the whole circuit in llama.cpp, though 72 of them compute
the same values, and a single compute row with the outputs copied by a last attention
layer needs about nine KV heads at that layer to carry 576 bits. Two instructions per pass would halve the
pass count and the floor's share, at about double the depth.

## 6. Ranked recommendation

The ranking follows the project's preference. One model comes first, the per-part selection of
`--weights` stays, and a run with one part in weights costs about that part alone.

1. **The prefix overlay, (a'), as the one-model design.** It is the only one-network
   design whose streams, and whose single-part runs, pay about their own dims: an
   estimated 2.8 to 3.6 llama.cpp hours to frame 300 (from 11.0), the GBC CPU at 7.3 to
   9.2 ms a pass, about 867 M dense and about 1.15 M nonzeros (the one model's 1.09 M
   plus a gate term per unit), one KV cache. Engine µs per GBC CPU step stays 4.8. It
   costs the overlay's compiler work (mode gates, null-token heads, the clearing-unit
   change in `Segments::new`), a channel-layout rule, per-part early exit in the dense
   oracle, and a llama.cpp patch in the family of the slice stack. The risks are the
   null-token margin in f16 beside the address margins, whether row-strided weight views
   run at full speed on Metal (copies avoid it), and the PPU's prefix, which sets the
   network's width, so the network can never be smaller than the PPU.
2. **d1 and d2 on the GBC CPU circuit, in any layout.** They are the only levers that
   move both cost models. With (a') they reach about 2.5 to 3.3 hours, and about 4.0 µs per
   GBC CPU step in the engine (both estimated). Prototype d2 first, since it is the larger engine
   win and needs no memory-key change. The risks are the control-signal count of the
   decode table, the key format's T2 margin for d1, and the eviction rule with
   bank-qualified keys.
3. **Round hidden to 64 channels, whatever the layout.** No exactness argument needs a
   multiple of 1,024 any more (§2). It is worth about a quarter of the PPU's dense size
   and nothing on the engine. It needs a check that no table, census or export assumes
   whole blocks.
4. **Prototype d4 on the host-fed GBC CPU before committing to it.** It is the only
   lever estimated to reach text's 3.1 µs budget in the engine and to cut llama.cpp's
   attention over cells, but it is a new compiler and a new norm argument. The host-fed
   GBC CPU (no memory, depth 5) settles its unit and change counts cheaply.
5. **The plain overlay, (a), only as a step toward (a').** It projects to 3.3 to 4.2
   hours, and a single-part run pays the PPU's width and every layer.
6. **(c) as the reference.** Four checkpoints measure what (a') must match. They exist
   and pass their gates, so exporting and timing them in llama.cpp is the cheapest way
   to turn §5.2's estimates into measurements before building (a'). (b) is not worth
   building, because it is (c) with a harder loader and two caches.

## 7. Open questions for the project

- May a lookup table that the checkpoint's own embedding writes into the cache at load
  (d2's decode table) count as weights, as the cartridge's cells count as data?
- With bank-qualified memory keys (d1), may the host map a decoded (bank, address) to a
  cell position by bit concatenation?
- Is a new number representation (d4) in scope, given that it means a new compiler and a
  re-derived exactness argument?
- In the prefix overlay, is it acceptable that a stream skips the layers after its own
  depth and the channels above its prefix, given that the dense oracle then models each
  stream's own depth rather than one pass of the whole network?

## 8. What this census does not cover

- The llama.cpp projections are not measured. The H11 proof run holds the GPU, and
  nobody has exported and timed the four own checkpoints or any overlay. The estimates
  assume each kernel's time scales with its dims, bounded by a guessed 20 µs per kernel.
- The engine's update census ran on the GBC CPU only (the `cpu-memory` checkpoint, text
  and overworld). The PPU, APU and control update counts per family were not measured; the
  APU prints its own line under `GBCFORMER_UNITS`, which a later census can collect.
- Core ML and MLX are not projected. The same dims argument applies, with the Neural
  Engine's T2 bound from `docs/stock-llama.md` §11.
- The PPU's and control's own dense units are estimated from their circuits.

## 9. Reproducing

The two `one_model` commands are gone with the example, which has since been removed.

```sh
cargo run --release --example one_model                  # per-part shapes and channels before rounding, the one model's, the overlay's (seconds)
cargo run --release --example one_model -- --build       # adds nonzeros and stream plans (about 30 s, compiles the one model)
cargo run --release --example levels -- --table cpu      # units per family and level, live wires; also host, ppu, apu, ctl
cargo run --release --example shape -- --mem             # the cpu-memory checkpoint's heads, layers and live families
GBCFORMER_UNITS=1 GBCFORMER_STATS=1 GBCFORMER_WIRE_CHANGES=/tmp/changes.txt CORE=model-mem tools/bench/text.sh 2> /tmp/text.err
cargo run -q --release --example levels -- --table cpu > /tmp/cpu.txt && bun tools/census/categories.ts static /tmp/cpu.txt
bun tools/census/categories.ts dyn /tmp/text.err         # gate updates per category
bun tools/census/changes.ts /tmp/changes.txt             # wire changes per category
bun tools/census/project.ts                              # §5.2's llama.cpp projections and dense sizes
```
