# Modern speedups that keep every token

Status: **design**. Nothing here is built. Every number carries its label: *measured*
(run on this machine, source named), *computed* (arithmetic on measured numbers, shown),
or *guess* (an estimate with no measurement behind it).

This document answers one question. Which modern inference techniques make the one
checkpoint run much faster in llama.cpp, and later in a custom harness at 60 fps, while
every decoded token stays identical. The baseline is `docs/stock-llama.md` §12a. It runs the
lines model (2,087.3 M dense, 1.49 M nonzero, hidden 5,120, intermediate 6,464, 15
layers, 22 query heads, one KV head, head dim 64) over Yellow's 141,415 line cells, and
`tools/stock-llama/bench.sh`, which projects about 271 hours to frame 300.

Three other efforts run in parallel and are not repeated here. They are H1 (sequence sets so that
only some tokens attend the cells), H2 (dropping foreign input slots and other token
cuts) and H3 (per-layer head and FFN counts). Each section says how it interacts with
them.

## 1. Where the 271 hours go

### What the bench counts

`bench.sh` multiplies each stream's ms per pass by a frozen pass count. The GBC CPU count,
981,204, is not one decode per instruction. A reference profile of the same 300 frames
(method in §1.3) gives 902,094 steps (instructions, HALT wake-ups and interrupt
dispatches). 11,688 of them read an I/O register or IE, bytes the host serves, which the GBC CPU
device takes as late inputs in a second pass, so the GBC CPU stream makes 913,782 group
decodes (measured). That is the figure `docs/one-model.md` §6 reports. The other 67,422
GBC CPU events are write records (computed: 981,204 − 913,782). The five recorded frames
confirm the split. `rec-engine/stream-1-A.jsonl` holds 43,878 `fill` lines, exactly
41,819 steps plus 2,059 late passes in the profile, and 5,937 `keep` lines against 5,955
writing steps (measured). The 18 missing keeps are probably writes to host-served
registers, which keep no cell (guess).

A write record decodes 16 tokens, not 185. At 141,415 cells that costs about 197 ms
(computed, linear between the measured 185 ms at 8 tokens and 275 ms at 66), not the
681 ms the bench charges. So the bench overstates the GBC CPU by about 9 hours:

| Stream | Events to frame 300 | ms per event (measured, stock b12070, own groups) | Hours |
| - | - | - | - |
| CPU groups | 913,782 | 681 | 172.9 |
| CPU write records | 67,422 | 197 (computed) | 3.7 |
| APU | 243,952 | 725 | 49.1 |
| PPU | 83,501 | 1,081 | 25.1 |
| control | 59,052 | 674 | 11.1 |
| **total** | | | **261.8** (the bench says 270.9) |

The GBC CPU is 67% of the time, the APU 19%, the PPU 10% and control 4% (computed).

### Two floors decide what llama.cpp can do

- **Attention over cells.** At 185 tokens a pass takes 697 ms over 141,415 cells and
  94 ms over 1,024 cells (measured, §12a), so 603 ms, 87%, is attention over cells.
  The marginal cost is about 3.2 ms per token per pass at Yellow's cache, and it is
  linear in tokens from 66 to 449 (measured: 275, 697, 1,148, 1,646 ms).
- **The dense matmuls and the per-decode floor.** With 1,024 cells, 185 tokens take
  94 ms. The dense work is about 185 × 2 × 1.9 G ≈ 0.7 TFLOP, so the GPU already runs
  near 7.5 TFLOPS (computed). The pass is compute-bound, almost all of it multiplying
  zeros: 1.49 M of 2,087 M weights are nonzero (0.07%). Eight tokens still take
  13.6 ms (measured). That is the floor of one `llama_decode` with these weights: 4.2 GB
  of f16 streamed at about 400 GB/s is 10.5 ms (computed).

The second floor settles the 60 fps question for llama.cpp. Text-box gameplay runs about
5,350 GBC CPU steps, 1,190 writing steps (one record each), 813 APU, 278 PPU and 197 control
passes per frame (measured profile and §1.1 counts), about 7,800 decodes. At 60 fps that
is about 470,000 decodes a second, or 2.1 µs each. The 13.6 ms floor is about 6,400 times
too slow
(computed), and every decode depends on the one before it. **60 fps lives only in a
custom harness** (§11). The goal for llama.cpp is hours to frame 300, and every
candidate below is ranked on that.

### The instruction mix behind the GBC CPU passes

The method is a `Core` that wraps `reference::Reference` and counts each step by PC, opcode,
reads, writes, and DMA bytes (`Core::copied`), passed to `Session::with_core`, with the
bench scripts' `--press` schedules. Yellow is deterministic under a fixed input script,
so one run per window is exact, not a sample.

| Window | Steps | Steps per frame | Writes | Mean steps between jumps |
| - | - | - | - | - |
| frames 0 to 300 (the bench's metric) | 902,094 | 3,007 | 80,695 | 6.1 |
| text (`text.sh`, frames 2,900 to 3,500) | 3,208,968 | 5,348 | 935,597 | 5.5 |
| overworld (`overworld.sh`, frames 7,200 to 7,800) | 1,179,826 | 1,966 | 252,041 | 5.7 |

The 300-frame metric is dominated by one loop that gameplay never runs:

| Window | Loop | Code | Share of steps | Writes | IME |
| - | - | - | - | - | - |
| frames 0 to 300 | `631F` (bank 0x1C) | `nop; nop; nop; dec de; ld a,d; or e; jr nz` (a delay countdown) | 65.2% | none | on |
| frames 0 to 300 | `1D3C`, `1679`, `1AD2` | memory clears and fills | 8.9% | one per iteration | off |
| frames 0 to 300 | `FF86` | `dec a; jr nz` while OAM DMA runs (the HRAM routine every game copies) | 2.5% | none | off |
| text | `3745` | `ld hl,$C02A; xor a; or [hl]; ...; jr nz` (polls sound state) | 11.0% | none | |
| text | `38EB`, `1B72`, `385F` | button wait, text and tile loops with calls | 15.9% | many | |
| text | `FF86` | OAM DMA wait | 1.5% | none | off |
| overworld | `4A59`, `0B6A` (`ld a,[hli]; ld [de],a; inc de; dec c; jr nz`) | copies | 11.9% | one per iteration | off, on |
| overworld | `FF86` | OAM DMA wait | 4.1% | none | off |

Frames 0 to 60 run about 12,000 steps each, the delay loop at full speed. From frame 61
on, the boot window runs about 725 steps per frame (measured per-frame counts). So any
"fewer passes" gain measured on frame 300 is mostly the delay loop and does not carry to
gameplay. Every candidate below that changes the pass count gives the boot and the
gameplay numbers separately.

## 2. Bank-scoped attention (candidate 1)

**Bottleneck.** Attention over cells, 87% of every pass. Every pass of every stream
attends all 141,415 cells, but a GBC CPU step can only address the banks mapped now, a PPU
line reads only VRAM, OAM and the palettes, and the APU and control streams read memory
only through other parts' heads holding fixed inputs.

**What llama.cpp b12070 does, read in its source.** With one unified KV stream,
`llama_kv_cache::get_n_kv_rows` counts the cells of the batch's sequences, and
`set_input_attn_mask` gathers only those rows (`llama_kv_cells::get_seq_cells`, a bitset
per sequence). So attention costs cells in the batch's sequences, not cells in the
cache. The query side has one limit. `set_input_kq_mask` tests
`cells.seq_has(j, ubatch->seq_id[i][0])`, the token's **first** sequence only. A token
that names several sequences gathers their union of rows but attends only its first. So
a pass cannot name "ROM0 + ROM bank 5 + WRAM bank 2" as three sequences. It needs one
sequence that holds that union.

**Mechanism.**

- Sequence 0 keeps every cell, as now, so no cell is ever freed. No pass decodes into
  it, because any token in sequence 0 pulls all 141,415 rows into the gather.
- Sequence `C` is the GBC CPU's mapped set. It holds ROM bank 0 (2,048 lines), the mapped
  ROMX bank (2,048), both VRAM banks (2,048, because the PPU part's foreign heads read
  tile lines that depend on VRAM content), the mapped SRAM bank (1,024), WRAM bank 0 and
  the mapped WRAM bank (512 each), and the small fixed regions (OAM, HRAM, FF, VOID,
  REGS, PAL, HOST), about 8,300 lines (computed from `memory::layout`). A bank is a
  contiguous range of line positions, so a switch is one `llama_memory_seq_rm(C, p0, p1)`
  and one `llama_memory_seq_cp(0, C, q0, q1)`. Both are stock calls that touch only cell
  metadata, but b12070 scans the cell array to do it, and text boxes switch banks
  about 150 times a frame (measured, writes below `6000`, `FF4F` and `FF70`), so the
  host time per switch needs a measurement.
- Sequence `V` serves the PPU stream (VRAM, OAM, PAL, REGS and the lines the GBC CPU part's
  foreign heads name at held inputs, about 2,100), and the APU and control streams too,
  since their foreign heads are the GBC CPU's and the PPU's at held inputs.
- Each sequence holds the top cell (HOST's last line), so stock llama.cpp's "next
  position" stays at `P`.
- A write evicts the old cell from every sequence with `seq_rm(-1, n, n + 1)`, not
  only from `C`. If sequence 0 kept it, the next bank switch would copy both the old
  and the new cell into `C`, two keys for one address, and the read would split 1/2.
  The record then decodes in the stream's sequence, its cell token is added to
  sequence 0 and every other sequence whose range holds that line with `seq_cp`, and
  it moves to `n` as now.

**Who selects the set.** Bank selection reads state, so the selector matters:

- The PPU, APU and control sets are fixed by the layout and the audit, with no state and
  no selector. `tests/stream_audit.rs` already evaluates every stream's merged circuit
  with held inputs over Yellow's memory. It needs one more output, the set of lines each
  foreign head names, to prove that the set holds them. If a foreign GBC CPU head names a
  banked line at held inputs (the immediate address at ROM `0x0001`), that stream uses
  `C` instead.
- The GBC CPU set changes with ROMB0, ROMB1, RAMB, SVBK and VBK, which are memory bytes at
  `REGS` (`docs/memory-in-weights.md`). If the host decoded them into banks, the host
  would be running the mapper. That is side computation, so it is ruled out.
- **The model emits the set.** The GBC CPU circuit already translates logical addresses
  through the banking registers (`sm83::translate`, one level). Add output slots `MAPr`,
  `MAPs`, `MAPw` and `MAPv`, the physical base of `4000`, `A000`, `D000` and `8000`
  after this step's writes, spelled like `WP0a..c`. The host maps each decoded address
  token to its line range and updates `C` before the next pass. That is the same footing
  as eviction in `docs/stock-llama.md` §4, which the project adopted: "the address only
  from the model's decoded tokens". It goes one step further, from "which cell to drop"
  to "which cells to show", so it needs the same project decision in the same words before it
  ships.

**Why the tokens stay identical.** The circuit translates every GBC CPU address through the
banking registers the step reads, so every line a GBC CPU head names lies in a mapped bank,
which `C` holds by construction. Every excluded cell already scored at least 256 below
the matched cell after scale and had softmax weight exactly 0 in f16 and in f32
(`exp(−256)` underflows). Removing a key whose weight is exactly 0 changes no sum. The
only risk is a stale or wrong `MAP` token, and a missing cell then gives a wrong read,
not an error. Two guards cover it. The engine, which knows every head's address, checks
on recorded passes that each named line lies in the set the `MAP` tokens imply. And the
harness's `STOCK_GROUP=check` pattern carries over. It decodes sampled passes in both `C`
and sequence 0 and require bit-identical rows. `bench.sh`'s diff against `rec-engine`
stays on every run.

**Expected gain (computed).** The cost of cells is linear in cells at each token count
between the 1,024-cell and 141,415-cell measurements, so per pass:

| Stream | Tokens | Cells | ms now (measured) | ms scoped (computed) |
| - | - | - | - | - |
| CPU group | 185 | 8,300 | 681 to 697 | 94 + 603 × 7,276 / 140,391 = 125 |
| CPU write record | 16 | 8,300 | 197 (computed) | 28 |
| PPU | 309 | 2,100 | 1,081 to 1,148 | 153 + 995 × 1,076 / 140,391 = 161 |
| APU | 207 | 2,100 | 725 | 110 |
| control | 189 | 2,100 | 674 | 101 |

Hours to frame 300 fall from 261.8 to 45.1 (computed), **5.8 times**. Scoping only the
PPU, APU and control streams, with no circuit change, gives 261.8 → 189.4 hours. A first
GBC CPU version that scopes only the ROMX bank (all RAM always in `C`, about 12,400 lines,
one `MAPr` slot) gives 143 ms per GBC CPU group instead of 125.

**Interaction.** After scoping, cells cost 31 of 125 ms in a GBC CPU pass. H1 and H3 cut
that 31 ms term, not the 94 ms dense term, so they matter much less once scoping lands.
The reverse holds too. Scoping makes the dense term the next bottleneck (§6).

**Where it runs.** The model emits `MAP`. The host routes sequence membership. The
harness applies the PPU, APU and control sets, which come from the layout and the audit.

**Runtime support.** Stock llama.cpp API (`seq_cp`, `seq_rm`, kv_rows with flash
attention), no patch.

**Cost.** The PPU, APU and control sets take about a day (harness and audit, guess). The `MAP` slots are a GBC CPU
circuit change (four translated constants after writes), the slot ABI, and every GBC CPU
gate, about a week (guess).

**Further.** The tightest set is the lines a pass will read, `line(PC)`, `line(PC+2)`,
and the lines of HL, BC, DE, SP, SP±1, SP−2, `FF00+C`, `FF00+n` and the immediate
address, about 16, emitted by the previous pass. That leaves 94 ms of a 125 ms pass, a
1.3 times gain over the mapped set (computed). A miss there is likely rather than
impossible, and the value lanes have no spare dimension for a hit bit, so wait until the
dense term has shrunk.

## 3. Fewer GBC CPU passes (candidate 2)

**Bottleneck.** The pass count: 913,782 GBC CPU groups to frame 300.

**Generic fusion does not help.** Two instructions in one pass need the second's
fetch address from the first's PC and its operands from the first's results. Depth grows
about k times (9 levels per SM83 step), and every layer with memory heads repeats per
step. A two-step GBC CPU pass needs about 19 layers. The one model runs 15 now, so every
stream would pay 27% more per pass while GBC CPU passes halve. With the GBC CPU at 72% of the
time after scoping (computed), the net is 0.72 × 0.63 + 0.28 × 1.27 = 0.81, a 1.23
times gain for a two-instruction SM83 circuit with hazards between the two steps and
four writes (computed). Multi-token prediction is the same trade under another name.
Straight-line runs average 5.5 to 6.1 instructions, so there is little to fuse anyway.
Not recommended.

**Closed forms for loop idioms do help.** Some loops collapse algebraically, as the
APU's timers already advance by closed forms over a tick count. The GBC CPU circuit
recognises the idiom from the code bytes at PC (two lines of fetch heads) and executes
k iterations in one pass:

| Idiom | Bytes | After k iterations | Limit on k |
| - | - | - | - |
| countdown on a pair, with fillers | `00…00 1B 7A B3 20 F9` and its `0B 78 B1` (BC) variants | DE − k, A = D, flags of `or`, cycles 10k − 1 on exit | with IME on, the cycles left before an interrupt can dispatch |
| OAM DMA wait | `3D 20 FD` in HRAM | A − k, flags of `dec` | none with IME off (measured, always off in the three profiled windows) |
| fill and clear | `22 0D 20 FC`, `36 00 23 0B 78 B1 20 F8` | up to two whole lines written, so two write records instead of 16 | to the end of the second line |
| copy | `2A 12 13 0D 20 FA` | up to two lines read and written | the same |

`k` never has to be the exact maximum. Any k at or below the limit is exact, so the
circuit can use `min(DE, budget >> 4)`, a subtraction and a shift, instead of a
division. For the countdown with IME on, the budget is the M-cycles before an IF bit that IE
enables can rise. The control logic does not compute that today. Its horizon (`Lines`,
`Control::lines`) is the scheduler's bound, the least of the DMA's, the pause's and the
LCD's (a mode-3 start or line 144, at most about 114 M-cycles a line), and it leaves out
the timer. Two choices follow. Using that horizon is safe only with the timer interrupt
masked and folds at most one line per pass. A new control output, "M-cycles until an
enabled interrupt can be requested", folds up to a frame per pass. Either reaches the GBC CPU
as a new input slot. The machine still advances the
other devices in blocks bounded by their horizons (`Machine::advance`), so their passes
are unchanged.

**Expected gain (computed from the profile).**

- Frames 0 to 300. 691,226 steps fold into about 1,500 passes with the new control
  output, or 9,000 to 13,500 with the per-line horizon (the delay loop then takes 84,000 / 11
  ≈ 7,600 passes at the exact limit, 12,000 with `budget >> 4`). GBC CPU groups fall from
  913,782 to about 224,000, or 232,000 to 236,000 (4.1, or 3.9 times), and write records from 67,422 to about 51,000. With scoping, 45.1 → 21.0 or
  21.3 hours.
- Text. The DMA wait is 1.5% and the rest of the hot loops write through calls. About
  1.02 times.
- Overworld. The DMA wait and the copy loops are 16% at most (if `4A59` is a copy idiom,
  which is unverified). At most 1.19 times.

So this candidate is large on the bench's metric and small in play. The bench should
grow a gameplay window (§12) before anyone tunes for frame 300.

**Why the tokens stay identical.** Each idiom's result after k iterations is exactly the
state single-stepping reaches, as long as no interrupt dispatches and no other device
writes what the loop reads in that span. The limit on k guarantees both. The loop has no reads (the
countdown and DMA wait), or reads of lines only the loop writes (fill and copy), and no
dispatch before the budget. The lockstep reference runs the same k instructions and
compares at the fused boundary.

**Where it runs.** The model recognises the idiom and computes the result. The host
schedules as now. The budget reaches the GBC CPU as a model-computed input (the control
stream's), not as a host value. On the four-checkpoint stack, where control runs on the
host, the budget is a host input and must be labelled.

**Runtime support.** Any runtime, since it is a circuit change.

**Cost.** High. `step.rs` defines a step as one instruction, and a fused step breaks
that contract for `Lockstep`, SST (the idioms need multi-step expected states from the
reference) and `interrupt_fuzz`. Each idiom is a recogniser over two code lines plus
its closed form. The DMA wait is the cheapest first idiom. Every game has it, IME is
off, and it has no reads and no writes.

**Elision of fixed points is not permitted without a user decision.** The text window's
`3745` loop repeats an identical step input (same registers, same code bytes, same reads,
no write or DMA in between) 351,062 times (measured, by keying every step on its full
input). Like HALT, the GBC CPU there cannot change until an outside event, and the host
could advance time without calling the model, which `step.rs` already allows for HALT.
But here the host would decide from the model's past outputs that the next outputs
repeat. That is the host supplying an instruction result, which the "the model must
really run the CPU" constraint forbids. It is worth 11% in text boxes, 2.5% at boot and
0% in the overworld (measured). It is listed for the record.

## 4. Batched and speculative verification (candidate 3)

**Mechanism.** The engine drafts the next K passes. llama.cpp decodes K groups in one
batch, each in its own sequence with equal token counts (so `get_n_kv_slices` gives
each its own rows), and accepts the longest prefix whose decoded tokens match the
drafts. Accepted passes are llama.cpp's own outputs for exactly the right inputs, so
exactness holds. A draft write shows its cell only to later groups in the batch. The
record's cell token carries the later groups' sequences, and the old cell leaves those
sequences before the decode.

**Expected gain.** Small. Batching pays only for the per-decode floor, and at 185 tokens
the pass is compute-bound. Time is linear in tokens (66 to 449 tokens, measured), so K
groups in one decode cost about K times one group. Per pass the floor is about 14 to 19
ms of 125 (computed, the 1,024-cell intercept), at most 1.15 times. Batching a write
record with the next group saves about 14 ms per record, 67,422 × 14 ms ≈ 16 minutes
to frame 300 (computed). Only 28% of boot writes are batchable without a chain anyway:
58,030 of 80,695 writes hit the line the previous write hit (measured).

**When to revisit.** If H2 and §6 bring a group under about 25 tokens, a pass becomes
bandwidth-bound (the 4.2 GB weight stream ≈ 10.5 ms ≈ 22 tokens of compute, computed),
and batching K groups then approaches K times.

**Runtime support.** Stock API (multi-sequence batches, equal runs). **Cost.** High. It needs a
drafting loop that follows the machine's scheduler across streams, and rollback.

## 5. Key and value shape (candidate 4)

- **Keys narrower than values.** The memory key uses 20 dimensions (19 line-address
  bits and `committed`), layer 0's codes 22, and the value needs all 64 (a line).
  Halving the key dimension saves at most 25% of attention work. That is about 8 ms of a
  125 ms pass after scoping (computed). Stock LLAMA arch forbids it:
  `src/models/llama.cpp` asserts `n_embd_head == hparams.n_embd_head_k()` with
  `n_embd_head = n_embd_head_v()`, and Metal's flash-attention kernels exist only for
  fixed (DK, DV) pairs. A patch and a kernel buy 6%. Not recommended.
- **KV quantization.** No gain and no exactness. A pass reads about 0.54 GB of KV
  (Yellow's whole f16 cache, §12a), about 1.4 ms at 400 GB/s (computed), and the
  attention is compute-bound. `q8_0` stores `d = amax/127` in f16. For a ±1 key,
  `127 · f16(1/127) ≈ 0.99994`, not 1 (computed). `q4_0` can hold {0, 1} blocks exactly
  but not ±1 in one block. Not recommended.
- **Head dimension.** 64 already holds one 8-byte line per cell. Smaller heads mean more
  cells or a byte-select level after every read. Not recommended.

## 6. Experts are parts (candidate 5)

**Bottleneck.** After scoping, the dense matmuls are 94 of 125 ms in a GBC CPU pass, and the
MLP is 1,489 M of the 1,716 M per-token multiply-adds (87%, computed from the config).
Every token runs every part's units. The engine already skips them. `Plan::prune`
evaluates only a stream's reach (GBC CPU 9,829 units, PPU 8,368, APU 7,999, control 1,696,
`docs/one-model.md` §4).

**Mechanism.** Stock LLAMA arch supports experts (`n_expert`, `ffn_gate_inp`, the
`*_exps` tensors, `build_moe_ffn` with softmax gating and `norm_w = true`). Make each
part an expert at every layer. The expert holds that part's units at that layer, plus a
copy of the shared units, the routing and clearing units. The LLAMA graph loads
`n_ff_shexp` tensors but never adds them, so shared units cannot use a shared expert.
The router reads a part channel that every token carries. Each part has its own input
tokens and slot identities, so the embedding can set it. The router weight on the part
channel is 128. The selected logit leads by 128, `exp(−128)` is 0 in f32, the selected
probability is exactly 1, and `norm_w` gives `1/1 = 1`. With top-2 and two equal leading
logits each weight is exactly 1/2, and doubling that expert's down rows (a power of two)
keeps it exact. That allows a second split (GBC CPU opcode families) later.

**Why the tokens stay identical.** An expert that is not selected contributes exactly 0,
which is the "left-out unit" of `Plan::prune`. Its proof is the one already built and
passing for the engine. It needs the norm bound over declared ranges (4,092 against 262,144) and
every left-out routing unit gated by a token channel the stream does not decode. MoE
moves that proof from "the engine skips them" to "the weights skip them". The check is
the usual one. llama.cpp tokens against the engine's on SST and on the five recorded
frames, and logits bit-identical to the dense GGUF on sampled rows.

**Expected gain (guess, pending widths).** An expert's width is the widest part at each
layer. If `Circuit::schedule` spreads each part over all 15 layers, the per-part average
is 1,311 rows (GBC CPU) or less (computed: 2 rows per unit over 15 layers). With clearing and
routing units, a width around 2,000 is plausible (guess). Then per-token work falls from
1,716 M to 687 M multiply-adds, 2.5 times on the dense term (computed from the guess).
A GBC CPU pass after scoping goes from 125 to about 75 ms (the 10 ms floor does not scale),
and frame 300 from 45.1 to about 25 hours (computed from the guess). The weights a pass
streams shrink the same way, which lowers the floor in §1.

**Toward the overlay.** `docs/one-model.md` §3 plans an overlay to cut hidden from 5,120
to about 2,048. It needs every channel writer to contribute exactly 0 outside its own
part. MoE delivers that for every MLP writer, so only the attention writers remain
(null-token heads). The overlay then scales every dense matmul by about 2.5 again
(projected in `one-model.md`, not measured on the lines model). It is the larger build
and comes second.

**Where it runs.** All in the weights. **Runtime support.** Stock llama.cpp LLAMA arch.
**Cost.** Medium. The emitter writes experts and a router, `Circuit::schedule` gains a
per-part width objective, and `examples/export.rs` writes the MoE tensors.

## 7. Smaller dense weights (candidate 6)

- **Quantized weights, exactly representable.** The exporter writes the GGUF itself, so
  it can choose each `Q8_0` block's scale as a power of two `2^k` with every weight in
  the block equal to `q · 2^k`, `|q| ≤ 127`. Metal's `mul_mm` dequantizes to f16 exactly.
  The CPU backend quantizes activations to `q8` for `q8_0` dot products, which is
  inexact, so the GGUF would become Metal-only. A census over every 32-weight block
  decides which tensors qualify. Gate weights are `128·w` with `|w|` up to 511, so mixed
  blocks fail. The gain is bandwidth only, and the pass is compute-bound at today's group
  sizes, so there is no gain until a group falls under about 22 tokens (computed in §4).
  Revisit after H2 and §6.
- **Weight sharing across layers.** No two levels compute the same function, and sharing
  saves no FLOPs. Not recommended.
- **Hidden and intermediate.** §6 (experts) and the overlay are the levers. Per-layer
  intermediate is H3's.

## 8. Streams concurrently (candidate 7)

- **One decode, four streams.** The GBC CPU, APU, PPU and control streams are mostly
  independent between their sync points (the APU's reads, 0.2 to 0.6 a frame), and a
  batch can hold one group per stream in its own sequence. The pass is compute-bound, so
  the batch costs about the sum of its groups. The saving is the per-decode floor for
  three of four decodes, at most about 15% (computed as in §4).
- **Other compute units.** A second llama.cpp context on the CPU backend for the APU and
  control streams runs concurrently with the GPU, but the CPU backend multiplies 2 GB of
  mostly zeros far slower than Metal (guess, ten times). The Neural Engine runs this
  family exactly (`docs/stock-llama.md` §11), but only through Core ML, not llama.cpp.
  Neither is recommended for the llama.cpp goal. For the custom harness, streams on
  their own cores is how the engine already works.

## 9. The PPU and the APU (candidate 8)

- **APU: several samples per pass.** A segment is capped at the 48 kHz sample period,
  about 22 M-cycles, so APU passes follow samples: 813 per frame against 17,556 / 22 ≈
  798 (computed). Register writes are rare: 1.0 per frame at boot, 3.6 in text, 2.0 in
  the overworld (measured), plus DIV edges at 512 Hz (about 8.5 per frame). A pass over
  K sample periods with K stereo samples at K output positions, with position wires for
  the sample index (the PPU's mechanism), cuts APU passes up to about 60 times. Each
  sample position needs its channel phases at offset `k·Δ`. Over long spans a square
  channel's phase needs `floor(t / period)` with a variable period. That is one level of
  `[t ≥ j·P]` steps per sample position, too wide for f16 until legalised (§3 of
  `stock-llama.md`). With K = 16, about 260 tokens per pass and scoping, APU time falls
  from 7.5 to about 0.6 hours (computed, K and the per-pass cost are guesses). Medium
  gain on frame 300 (about 1.2 times after scoping), high cost.
- **PPU: two CGB passes per line.** The two passes per CGB line (one per byte of RGB555)
  repeat the line's non-pixel tokens. One pass with 320 output positions decodes 469
  tokens instead of 618 per line, about 1.3 times on the PPU, about 2% of the total after
  scoping (computed). Low.
- **PPU: several lines per pass.** Each line must see VRAM and the latches at its own
  mode 3, and the pixel positions do not shrink. No gain in token-passes.

## 10. Per-stream depth

The GBC CPU needs about 10 of the 15 layers (9 levels, layer 0 and the routing layer). The
last layers carry the PPU's levels. A patch that runs a stream's tokens through the
identity for layers where that stream has no unit and no head would cut GBC CPU passes by
about 27% (computed: 4 of 15 layers). It is exact when no unit or head in the skipped
layers writes a channel the stream reads, which the weights can check. It needs a
llama.cpp patch (no stock early exit) and changes nothing for the PPU. Medium gain,
medium cost. Ranked below experts, which cut more work and need no patch.

## 11. The custom harness at 60 fps

The engine (`src/engine/`) is already this harness in all but one respect. It evaluates
the same weights sparsely (0.07% nonzero), once per pass rather than once per position,
and incrementally. It reached 69.5 fps on text boxes, 126 in the overworld and 96 in
battle with every component in the one model (`docs/perf-log.md`, "release candidate",
one model `490c6845`, measured). That was before the f16 and lines checkpoints, and
hidden grew from 4,096 to 5,120. **Its speed on the lines checkpoint is not measured**,
and that is the first experiment for this section.

"Memory reads as attention over the same K/V rows" cannot mean a scan. One GBC CPU pass
over even the scoped set is 17 heads × 8,300 cells × 64 × 2 × 2 ≈ 36 MFLOP (computed),
and at 2 µs per pass that would take 18 TFLOPS. The exact fast path is top-1 by key
equality. A query whose address entries are exactly ±S and whose `committed` entry is
B (a guard on 20 numbers per head), against cells whose keys are exactly ±1 and 1 (a
guard at each write), has one maximiser, the cell whose key equals `sign(q)`. It leads
every other cell by at least 256 after scale, so its softmax weight is exactly 1 and the
head's output is that cell's V row. That is attention, computed by an index, under a
guard. A failed guard falls back to a scan of the scoped set. The engine's records index
and its oracles are this path, so the work left is to state the proof next to the
guard and to keep the fallback.

The harness gains from this document too. The loop idioms (§3) cut its passes by the
same 2% to 19% in play, experts (§6) are the pruning it already does, and bank-scoped
sets (§2) bound its fallback scan.

## 12. Ranked plan

Ranked by gain × confidence ÷ cost. Hours are to frame 300 from the corrected 261.8.

1. **Scope the PPU, APU and control streams to fixed sequences.** Gain 261.8 → 189.4
   hours (computed). High confidence, because the kv_rows path is stock and measured in the
   prefill lanes, and the change is the harness and the audit. Low cost.
   - First experiment. Run `stock-llama timing` (since removed) at 2,100 and 8,300 cells for 185, 207
     and 309 tokens. That confirms the linear-in-cells model behind every scoped number
     here before any code. Then extend the stream audit to list the lines each foreign
     head names at held inputs over Yellow's memory, and build `V` from that list.
   - Exactness check. Decode sampled passes in `V` and in sequence 0 and require
     bit-identical rows (the `STOCK_GROUP=check` pattern), plus `bench.sh`'s diff against
     `rec-engine` and the mirror check with 0 differences.
2. **Scope the GBC CPU to its mapped banks with model-emitted `MAP` tokens.** Gain 189.4 →
   45.1 hours (computed), with ROMX only first (143 ms per pass). High confidence in the
   mechanism, medium in the cost. It needs a project decision on "which cells to show"
   on the same footing as eviction.
   - First experiment. Time `seq_rm` plus `seq_cp` of one 2,048-line bank on the host
     (text boxes switch about 150 times a frame). Then add the ROMX-only `MAPr` slot, run
     the GBC CPU gates (SST, lockstep hashes, `checkpoint_sst`) and `bench.sh`.
   - Exactness check. The engine asserts on every recorded pass that every head's named
     line lies in the set the `MAP` tokens imply, the scoped and full decodes give
     bit-identical rows on sampled passes, and the diff shows 0 differences.
3. **Experts are parts.** Gain about 1.8 times on the total after scoping, 45.1 → about
   25 hours (computed from a guessed width of 2,000). Medium confidence until the widths
   are known. Medium cost, stock LLAMA arch.
   - First experiment. Print each part's units per layer for the lines model from
     `examples/one_model -- --build` (since removed), then reschedule with a per-part width objective
     and recompute the expert width. That settles the gain before the emitter changes.
   - Exactness check. `Plan::prune`'s two conditions hold for every stream (they already
     do), llama.cpp tokens equal the engine's on SST and on the five frames, and logits
     are bit-identical to the dense GGUF on sampled rows.
4. **Measure the engine on the lines checkpoint, then write down the exact-index proof.**
   It decides whether 60 fps already holds on the current checkpoint (it held at
   69.5 fps on `490c6845`, measured). Very low cost.
   - First experiment. `tools/bench/text.sh --weights all --one-model` and `overworld.sh`
     on the lines checkpoint, interleaved with `490c6845`, medians of 3, with a row in
     `docs/perf-log.md`.
   - Exactness check. Unchanged (frame hashes `870aa3fd` and `822059c1`, engine equal to
     the dense pass), plus a test that a query off the ±S lattice takes the scan
     fallback.
5. **Closed-form loop idioms, the OAM DMA wait first.** Gain 3.9 to 4.1 times on GBC CPU passes to
   frame 300 (45.1 → about 21 hours with scoping, computed), 1.02 to 1.19 times in play. High
   confidence in the counts, high cost, since a step stops being one instruction.
   - First experiment. The `3D 20 FD` HRAM idiom (IME off, no reads, no writes), with a
     reference-driven test that runs k instructions per expected state, then the
     countdown with the new control output as a GBC CPU input.
   - Exactness check. `--core lockstep` reaches `6872b003` (DMG 8) and `cfd31f3d` (CGB
     300) with the reference comparing at fused boundaries, plus SST and
     `interrupt_fuzz` extended to the fused steps.

Next after these: the overlay (hidden 5,120 → about 2,048), APU multi-sample passes, and
per-stream depth. Not recommended are generic multi-instruction fusion, narrow keys, KV
quantization, weight sharing, and other compute units for llama.cpp. Needs a user
decision first is fixed-point elision (11% of text-box steps).

**Fix the metric before tuning.** `bench.sh` charges write records a group's time
(about 9 hours too much). Its frame-300 window is 65% one boot delay loop, so it rewards
§3 four times more than play does. Add a text-box window (the `text.sh` frames, its
`--press` schedule) next to frame 300, and charge records at their own ms per decode.

## 13. Reproducing the profile

The profile is a scratch tool outside the repository, a binary crate that depends on
this one with `default-features = false`, wraps `reference::Reference` in a `Core` that
counts per step, and drives `Session::new(rom, None, Hardware::Cgb)` with
`press::PressScript`. It resets its counts at the warm-up frame. A step's input key is
its `Cpu`, its three opcode bytes, IE, IF, the DMA flag and its two data bytes. A
"memo hit" is a key already seen since the last write or DMA byte. It builds in 35 s and
runs a window in under a second. Counted by the same rules, the five recorded frames
match `rec-engine` exactly (§1.1), which is the check that the profile counts what the
runtime decodes.
