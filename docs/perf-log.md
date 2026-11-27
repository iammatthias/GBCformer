# Model-core performance log

Each entry is one change to the engine or the compiled circuit, measured on the same workload.

**Workload.** Yellow boot, 8 frames, 68,159 GBC CPU steps. No HALTs, so every step runs the model. This is the worst case.
- Command: `gbcformer run "<rom>" --frames 8 --core model --no-save`.
- Rows from 9 onward are the median of 3 runs. Earlier rows are single lockstep runs; row 0 ran while a full SST job was also running.
- Machine: Apple M1 Max, macOS 27.0.1, release build.

**Gates that must hold for a change to be kept.**
- The circuit passes SingleStepTests: 500,000 cases, 0 failed.
- The sparse engine produces the same tokens as the dense HF-semantics pass, on random states, under both bidirectional and causal masks.
- Lockstep against the reference interpreter for the full run, with frame hash `9e80d0f7…`.

| # | Change | Time | Steps/s | Kept |
| - | - | - | - | - |
| 0 | Engine v1: sparse rows, position pruning, MLP dedup | 287 s | 240 | baseline |
| 1 | Gate scatter; constant-only rows precomputed; one pass when no reads are requested | 121 s | 560 | yes |
| 2 | Template row: circuit layers once per pass, routing and head per position | 69 s | 990 | yes |
| 3 | Cached q/k/v per token and position; rows built lazily | 30 s | 2.3k | yes |
| 4 | LM-head table over route patterns; shared routing activations | 9.6 s | 7.1k | yes |
| 5 | Attention reused when logits are bit-identical; flat q/k/v cache | 7.35 s | 9.3k | yes |
| 6 | SiLU without `exp` outside (−89, 24); sparse logits; cached slot positions | 6.07 s | 11.2k | yes |
| 7 | Slot bytes in arrays, not string maps; sparse template layers | 5.87 s | 11.6k | yes |
| 8 | Integer-exact mode; marks instead of `contains`; unsorted nonzero list | 5.04 s | 13.5k | yes |
| 9 | Materialise wide mux selectors as wires | 5.9–6.3 s | | no: "keep" gates are usually 1, and as wires they reach every step |
| 10 | Single-unit steps when the form never exceeds θ | 3.98 s | 17.1k | yes; 215M → 137M dense params |
| 11 | CB decode from op1 nibble one-hots | 3.91 s | 17.4k | yes; layer-0 weights −18% |
| 12 | One pair datapath for INC/DEC rr, ADD HL,rr and (HL±); one `pc + len − hb` adder | 2.93 s | 23.2k | yes; 180 fewer wires |
| 13 | Allocation-light template path; routing into 8 route registers (routing layer included in the integer-exact check) | 2.02 s | 33.7k | yes |
| 14 | Layer-0 softmax weights memoised by the ids of the q/k vectors | 1.75 s | 38.9k | yes |
| 15 | Incremental evaluation: scatter only changed channels, re-evaluate only reached units, propagate activation differences | 1.13 s | 60.3k | yes |
| 16 | Sparse matrices as contiguous (index, weight) pairs | 1.06 s | 64.3k | yes |
| 17 | Gate pre-activations accumulated in the persistent state, not copied through scratch | 0.95 s | 71.7k | yes |

From row 18 on, the measure is the gameplay workload described below, in fps and µs per model pass, median of 2–3 runs, model core only. The same gates apply, plus a lockstep run of the gameplay segment ending in hash `5953eb63…`.

| # | Change | Gameplay fps | µs/pass | Kept |
| - | - | - | - | - |
| 18 | Baseline at row 17's engine | 13.0 | | |
| 19 | Integer engine (i32 gates, integer wires), fallback to the dense model, routing on a blank row | 16.7 | 10.2 | yes; the dense fallback is now tested by `forced_fallback_matches_fast_path` |
| 20 | Wide mux selectors as shared wires (row 9 retried; changes now dominate the cost) | 17.0 | 10.0 | yes; 120k → 73k nonzeros |
| 21 | Opcode one-hot carried by the token embedding and gathered by 32 extra layer-0 heads | 16.8 | 10.1 | yes; units per pass 1008 → 734 |
| 22 | Layer-0 input recomputed only for heads whose key's value block changed; attention plans interned; decoded tokens cached per position | 20.5 | 8.3 | yes |
| 23 | Operand isolation for `jr`, `sp + e8`, `a16 + 1`; `pc + len` split at the byte | 25.9 | 6.5 | yes |
| 24 | Isolation for the add/sub ALUs, DAA and `hl + pair`; `jr` split at the byte | 26.8 | 6.3 | yes |
| 25 | Attention plans cached per decoded-position list, checked by key ids, with per-plan reader lists | 28.8 | 5.9 | yes |
| 26 | JR folded into the pc adder: one adder gives the sequential pc, taken JR targets and CALL/RST returns | 30.8 | 5.5 | yes |

| 27 | Value-block ids per head, so changed-head detection compares integers, not 16 floats | 32.7 | 5.2 | yes |
| 28 | Routing: each token's gate contribution to its own units precomputed | 33.1 | 5.1 | yes |
| 29 | Activation differences sent on every gate update (they telescope in integers); no hit marks or stored activations | 36.6 | 4.6 | yes |
| 30 | Exclusive attention plans (one key per head, one key per channel): per-(position, token) wire lists cached, so layer 0 is a diff of two lists per moved position | 43.8 | 3.9 | yes |
| 31 | Changes routed into per-layer buckets by a static reader mask | 49.0 | 3.5 | yes |
| 32 | Per-position change stamps written from the routing layer's callback | 45.4 vs 47.8 (A/B, same conditions) | 3.74 vs 3.55 | no: the extra branch in the routing loop costs more than it saves |

| 33 | JR offset in one level from raw inputs; pc adder split at nibbles | 50.7 vs 48.5 (A/B) | 3.35 | yes |
| 34 | Down projection as flat per-unit (channel, weight) arrays; every unit writes one wire, checked at build | 56.6 vs 51.0 (A/B) | 3.0 | yes |
| 35 | Gate entries packed as 12-bit unit plus 20-bit weight | 53.5 vs 53.4 (A/B) | 3.17 | no: neutral |

From row 36 on, the measure is the text-box workload described below. It is the slowest plateau of live play, and the engine is its entire cost. Columns are fps, µs per model pass and units re-evaluated per fast pass (noise-free, so it leads), median of 3 runs; A/B rows interleave the two binaries because another build job shares the machine. Pass 1 (every output slot decoded) is 80% of passes and 93% of engine time (3.37 µs, 865 units); pass 2 costs 0.97 µs with 58 units. `GBCFORMER_STATS=1` prints the split.

| # | Change | Text-box fps | µs/pass | Units/pass | Kept |
| - | - | - | - | - | - |
| 36 | Layer-0 diff skips moved positions no head reads (the 27 output slots toggle mask/decoded every pass; each cost two lookups into the per-plan wire-list table) | 48.2 vs 46.8 (A/B) | 3.11 vs 3.20 | 705 | yes |
| 37 | Attention plan keyed by a per-token (q, k) class table instead of the roped ids of 47 boxed vectors | 49.3 vs 49.4 (A/B) | 3.04 vs 3.03 | 705 | no: neutral |
| 38 | No per-pass allocation: the moved-position list and the layer-0 change list are reused buffers (the change list grew by realloc from empty) | 53.0 vs 49.3 (A/B) | 2.83 vs 3.04 | 705 | yes |
| 39 | Decoded-position cache validated by two stamps (the pass a routing unit owned by the token last changed, the pass the route totals last changed) instead of comparing 12 route totals and 8 unit stamps | 55.0 vs 53.0 (A/B) | 2.72 vs 2.85 | 705 | yes |
| 40 | Decode loop writes an exact-capacity result (`collect::<Option<Vec>>` lost the size hint and reallocated 4, 8, 16, 32); pass 2 no longer copies its position list | 60.6 vs 55.6 (A/B) | 2.47 vs 2.70 | 705 | yes; overworld 131 fps (was 122) |
| 41 | Step pairs fused: the two units of `relu(K(s-θ+1)) - relu(K(s-θ))` (same gate row, biases one apart, opposite down weights) become one unit with activation clamp(g, 0, 1), flagged in the gate entry. `clamp()` itself was slower than the unfused loop (its two assertion branches, +5%); min/max with a cap array −4%; the flag bit −6% | 63.7 vs 60.6 (A/B) | 2.35 vs 2.50 | 512 (was 705) | yes; overworld 145 fps, 589 units (was 820) |
| 41a | Each unit's output channel and down weight interleaved in one array | 2.33 vs 2.33 (A/B) | | 512 | no: neutral |
| 42 | Staircases fused: a run of units with one gate row and one output wire as one unit with an i8 table over its bounded gate range | 2.44 vs 2.43 µs (A/B, median of 6) | | 416 (was 512) | no: neutral; the dropped updates were adjacent units on shared cache lines, the cheap kind |
| 43 | Head dim 32 and rope_theta 1e38 for the whole checkpoint (the memory heads need them; one head_dim per Llama model). 44 heads replace 68, dense params 171.7M → 187.4M, nonzeros 68036 → 68004 | 62.3 vs 61.4 (A/B, median of 5; ranges 2.31–2.48 vs 2.41–3.50 µs, the 3.50 overlapped a prefill run) | 2.41 vs 2.44 | 512 | yes: text no measurable difference; overworld 149.8 → 144.7 fps, 2.82 → 2.91 µs, slower in 4 of 5 pairs (the fifth was an A outlier at 3.54); load 6–10 from other builds, sides interleaved. Limiter not profiled; the hypothesis is the per-head `HEAD_DIM` loops in `layer0_changes`, which now walk 32 value dims per byte head where 8 are used, and layer 0 is a larger share of the overworld pass |

After row 41, interleaved against the row-35 engine on the same noisy machine: text box 3.18 to 2.32 µs per pass (47.1 to 64.5 fps), overworld 115 to 146 fps, the CGB segment of rows 18–35 169 to 202 fps (937 to 709 units per pass, frame hash `92b7cc1c…` unchanged). Phase timers (diagnostic build, not committed) put a text-box pass at 0.43 µs layer-0 diff, 1.27 µs unit loops, 0.44 µs bucket routing and drains, 0.24 µs decode, about 0.2 µs host; it moves 58 wires, 83 bucket entries and 103 sends per pass.

**Text-box workload** (`tools/bench/text.sh`). Frames 0–2899 run on the reference interpreter; 600 model frames advance Oak's speech with A every 40 frames: 3,208,968 steps, 4,004,423 passes, 17.4 layer-0 changes per pass, frame hash `870aa3fd…`. Steps per frame peak near 6,500, so 60 fps needs about 390k steps/s, or 2.0 µs per pass. `tools/bench/overworld.sh` (walking in Red's room, about 122 fps) guards against regressions.

Rows 19–26 came from per-family counts of the gate weights touched per pass. The compiler records unit names per MLP unit (`Model::unit_names`).

**Gameplay workload.** Frames 2400–2700 cover the main menu into Oak's intro. Frames 0–2399 run on the reference interpreter and are labelled as such (`--warmup 2400`).
- At row 15: 300 frames, 1,429,740 model steps, 11.4 fps in lockstep, zero divergences.
- The final frame hash `5953eb63…` equals a pure-reference run.
- The segment averages 4,766 steps per frame, so real time needs about 285k steps/s.

## Host changes

Host work runs once per M-cycle whichever core executes, so it is measured two ways. `tools/bench/text.sh` (Oak's speech, 600 model frames after a 2900-frame reference warmup) and `tools/bench/overworld.sh` (600 model frames walking) give the model-core cost; a 2900-frame `--core ref` run isolates the host, since the interpreter costs almost nothing. Frame hashes must match the base: text `870aa3fd…`, overworld `822059c1…`.

| Change | text fps | overworld fps | ref-core fps | Kept |
| - | - | - | - | - |
| APU ticked every M-cycle, samples mixed at 48 kHz | 50.1 → 49.9 (interleaved A/B, median of 3) | 118.7–119.2 → 119.9–120.7 (noise) | 2992 → 2343 | yes: 0.093 ms per frame, about 0.5% of a model-core frame |
| Phase 0 device contract: blocks cut at the smallest horizon, signals applied after each block, OAM DMA and HDMA as devices requesting copies, bus as a route table | 63.7 → 63.0 (A/B of the two commits around it, median of 3, load 8–10) | 143.1 → 143.4 (same runs, noise) | text 1848 → 1642, overworld 2511 → 2250 (A/B, median of 3, load 6–9); the two commits around it are flat | kept, then fixed below: 10% of host time, 0.5% of a model-core frame. A `sample` profile put it in `route()` out of line (3% of host samples), `apply_signals`' Vec take and put-back per block and per write, and `Machine::advance` self time 375 → 503 samples: the per-block scalars sat in five cache lines spread over the 230 KB struct |
| Phase 0 host fix: `route` inlined, `apply_signals` returns before the Vec take when no copy was asked for, `Machine` is `repr(C)` with the per-block fields first | 62.4 vs 62.5 (A/B of the build before against the fix, median of 5, load 3–7; B's runs inside A's 61.5–64.3) | 144.5 vs 143.2 (same runs; B below A in all 5 pairs, ranges 143.3–148.6 vs 140.1–146.0) | text 1831 → 1735, overworld 2500 → 2351 (A/B against the build before, median of 5, load 13–19); half the contract's host cost is back, the rest is the per-block prologue (about 25 instructions a block, one block per M-cycle in HALT) and OAM DMA as 160 one-cycle blocks a frame | yes. The remaining 1% on overworld is not host: with `GBCFORMER_STATS=1` the engine's own pass timers account for the gap (overworld 2.98 → 3.02 µs a first pass, text 2.59 → 2.65, single runs) while the residual (wall minus pass time) is flat on text and +20 ms per 600 frames on overworld. The model-core bisect over the four commits of the contract splits that 1–2% three ways at the 3-round noise floor; one candidate is the engine merge's per-pair cache index, `cache[slots.of(t, j)]` (a table load) where `cache[t * seq + j]` was a multiply-add |

## Why the shortcuts are exact

- **Integer-exact mode (row 8).** At build time the engine checks the weights:
  - every gate term is a multiple of 256,
  - every `up` is exactly 1,
  - every down term is an integer,
  - gate sums stay below 2^32, and down sums stay below 2^24.

  At run time it checks that residual values are in {−1, 0, 1} and that no gate falls in SiLU's unsaturated band. If any check fails, the pass takes the ordered general path. Under these conditions f32 never rounds, so summation order cannot change a result.
- **Template path (rows 2–4).** It needs three structural preconditions, all checked when the engine is built:
  - no attention after layer 0,
  - no layer before the last reads the token channels,
  - no layer writes the token channels or the constants.

  Positions whose layer-0 attention outputs differ fall back to the general path.

## Lessons

- What the engine pays for is the weights reachable from active channels, not the static weight count. Row 9 cut static weights by 30% and still ran slower.
- Rebuild all targets before measuring. `cargo build --examples` leaves `target/release/gbcformer` stale, which produced one bad reading during row 9.
- Per-pass allocations cost more than the unit loop they sit beside. Three Vecs and one realloc chain were 10% of samples (rows 38, 40). Look at malloc symbols in a `sample` profile before touching the hot loop.
- Units re-evaluated is noise-free but not a cost model. Dropping 193 adjacent pair units saved 6% (row 41); dropping 96 more staircase units saved nothing (row 42). What costs is the distinct cache lines a pass touches and the per-wire-change bookkeeping, not the update count.
- `clamp()` carries an assertion; in the hot loop `min().max()` was 7% faster than `clamp()` with the same values (row 41).
- Interleave A/B binaries when another build shares the machine. Baselines drifted 3.0 to 3.3 µs within an hour, and only paired rounds separated 3% changes.

## Memory prototypes

Prototype B (`src/memory/delta.rs`), `memory::attn::AttnIndex` and `examples/membench` have since been removed, so the commands of this section no longer run.

Both prototypes hold the flat memory (`Machine::memory_layout`: ROM, cartridge RAM, WRAM, VRAM, OAM, HRAM) and were checked against a byte array with `examples/membench` on two recordings of Yellow's GBC CPU bus traffic in DMG mode (one run each; the dense forms are deterministic and the fast paths are array lookups, so the spread is below the reported precision): the boot and title, `--frames 300` (887,480 steps, 2,723,231 reads at 3.07 per step including the three opcode-fetch reads, 74,598 writes at 0.08 per step, 249 per frame), and the text-box segment of `tools/bench/text.sh`, `--frames 3500 --from 2900` with its presses (1,023,199 steps, 3,387,138 reads at 3.31 per step, 257,155 writes at 0.25 per step, 429 per frame). The flat memory is 1,130,783 bytes (1,048,576 ROM, 82,207 RAM).

| | A: attention cache (`memory::attn`) | B: DeltaNet-style state (`memory::delta`) |
| - | - | - |
| Exactness, recorded traffic | fast path exact over all 6,442,122 accesses of both recordings in lockstep; 1,437 sampled dense reads exact | fast path exact over all accesses; 28,320 sampled dense reads exact |
| Exactness, random stream (40,000 accesses on 20 KB, every read dense) | exact | exact |
| Guard | margins computed in the read's own f32 arithmetic at three positions: 248 logits at the longest (104 makes `exp` exactly 0), plus the window of 29,444 writes between compactions; the dense read asserts every output bit is exactly 0 or 1 | argued, not checked at run time: keys and queries one-hot by construction, β = 1, entries 0/1; a model would need per-token checks like the engine's wire-range checks |
| Compactions in the recordings | 2 (boot), 8 (text box): one per 118 or 69 frames | none needed |
| Dense form measured | one head's attention over the whole cache (a stock forward would also run every other layer on every prefilled token, so the end-to-end dense oracle of gate 2 cannot run at this length; the oracle is per head plus the margin argument) | one head's 128×128 matvec (a stock Gated DeltaNet layer runs all 553 heads per token, about 550× these times; skipping zero-key heads is B's first engine shortcut) |
| Dense read | 12.1 ms over 1.15 M tokens; 51 ms at 2.18 M tokens | 1.3 to 1.9 µs; 2.8 to 10 µs at 2.18 MB |
| Dense write | 0.98 µs (one token appended, RoPE of its key) | 2.7 to 3.0 µs (row read, byte merged, rank-1 update) |
| Footprint at 2.18 MB (Crystal-sized) | 2,179,359 tokens; KV cache 558 MB in f32 at head dim 32 | 1,065 heads; state 70 MB in f32 |
| Loading a 1 MB cartridge (dense form) | 0.37 s for 1.13 M tokens, one head's K and V only | 0.11 s for 553 heads, one delta write per 16-byte row |
| Fast path | 2.2 ns per access (byte array plus window counter) | 2.2 ns per access (byte array) |
| Projected cost on Yellow | 6,074 accesses per frame in the text box: 0.013 ms per frame, 0.1% of a 15 ms model-core frame | 0.013 ms per frame |
| Layer set | Llama attention with RoPE (base 1e38, head dim 32) plus KV-cache truncation at compaction | Gated DeltaNet (Yang et al.), outside the Llama layer set |

The engine path is not measured. The fast paths are standalone byte arrays, and the engine's `Plan` rejects attention after layer 0 and would need a memory-head variant (see the report for E16). What bounds A is f32 precision. Logits are of the order 22·S + M ≈ 10⁸ and the per-position recency gap M·θ·cos(π/6) must stay above their rounding, which caps M·θ and so the window at about 30,000 writes. The margin check found that address bits in RoPE pair 3 fail at 2.2 M positions (a mismatch cross term of 2·S·sin(0.16)); pairs 5 and up hold the angle below 3·10⁻⁶ rad. A's compaction takes the RAM snapshot from the newest token per address, which is the fast path's byte array; a host with only the dense model would need 82,207 attention reads for it.

## Memory in weights (E20, design A integration)

Numbers for the units of `docs/memory-in-weights.md`, each a single run unless noted; the prefill is deterministic and its spread between runs is below the reported precision.

| Unit | Measurement | Yellow (1 MB ROM + 82,207 RAM) | Crystal (2 MB ROM + 82,207 RAM) |
| - | - | - | - |
| B | Prefill, formula path (`examples/prefill`): one record per byte, key and value by `memory::attn::key` | 1,130,783 records in 0.834 s, 0.74 µs per record | 2,179,359 records in 1.21 s, 0.56 µs per record |
| B | KV cache: f32 key and value of the memory head per kept token (256 B) | 289 MB | 558 MB |
| B | Oracle, `llama::forward_cached` (dense pass with the KV cache) on every 4,096th record: key and value bit-identical to the formula | 277 of 277, 120 ms per record | 533 of 533, 218 ms per record |
| B | Oracle reads: the reader record decoded by the dense pass with the cache | 35 of 35 exact, 108 ms per read | 67 of 67 exact, 212 ms per read |

Where the time goes. Single runs on a machine at load 6 to 12 (other builds), so the per-record cost is a range. A 64 KB cartridge (blargg `cpu_instrs`, 147,743 records) prefilled in 0.160 s, which bounds the fixed cost (`Params::check` evaluates 3 × 29,444 keys; the index and KV vectors grow) at 0.16 s, and the marginal cost per record between the three cartridges is 0.36 to 0.69 µs. The cache size does not enter the formula path. The limiter is not profiled; the hypothesis is `rope_at` (32 `powf`, `cos` and `sin` in f64 per token), which unit D can cache per position class. The dense oracle is 4 to 10 times the prototype's 12 ms per dense read because `forward_cached` scores the cache for all four positions of the record (each position's query is the same, but the dense pass does not know that) and allocates the logit and exponent vectors per position; it is the oracle, not a path.

### Units C to F: the memory inside the CPU checkpoint (`--core model-mem`)

Five commits. The memory checkpoint is 313.3M dense parameters, 171,186 nonzeros, hidden 2048, intermediate 3584, 10 layers, 23 heads, 8,004 tokens (the host checkpoint stays at 187.4M, 68,004 nonzeros, byte-identical). The circuit is 1,707 wires, 1,427 nodes, depth 9 (host: 1,599 wires, 927 nodes, depth 5): one level for the register heads, one for the fetch translation, two for the opcode one-hots from fetched bytes (nibble one-hots, then the 256 ANDs), one for the read translation; the ALU and the finals shift up by four. The engine serves 16 index reads per pass (ten registers, three fetches, two data bytes, the copy head).

Correctness runs, all in lockstep with the reference with the audit comparing every fetch, read address and byte, write address and record to the bus (`Core::audit`), single runs:

| Gate | Core | Steps | Records kept | Compactions (cost each) | Host-served steps | Hash |
| - | - | - | - | - | - | - |
| Yellow DMG 8 | lockstep-mem | 68,159 | 12,449 | 0 | 2,059 | `6872b003` |
| Yellow CGB 300 | lockstep-mem | 902,094 | 77,709 | 2 | 11,688 | `cfd31f3d` |
| text.sh (600 frames) | lockstep-mem | 3,208,968 | 928,869 | 31 (one per 19 frames) | 15,400 | `870aa3fd` |
| overworld.sh (600 frames) | lockstep-mem | 1,179,826 | 245,818 | 8 (one per 75 frames), 46.2 ms | 7,585 | `822059c1` |
| Crystal 1,500 (MBC3, 2 MB, 558 MB KV) | lockstep-mem | 6,325,729 | 1,657,402 | 56, 42.6 ms | 53,839 | `4e6a7509` |
| Zelda DX 1,500 | lockstep-mem | 9,055,466 | 1,049,991 | 35, 46.6 ms | 69,406 | `1d00d871` |
| Mario DX 1,500 | lockstep-mem | 6,683,768 | 486,291 | 16, 29.3 ms | 18,757 | `7dec3947` |
| blargg cpu_instrs (MBC1, 64 KB) | model-mem | 28,736,136 | 6,017,992 | 204, 24.4 ms | 24,919 | serial Passed |
| blargg instr_timing (ROM only) | model-mem | 23,463,059 | 47,242 | 1, 25.3 ms | 11,104 | serial Passed |

Host-served steps are the ones whose fetch or read landed on the I/O page, IE or the MBC3 clock (joypad, LY, DIV, STAT polls); the run prints the count. Every other step is one pass; the host checkpoint needed two passes for 24% of gameplay steps.

SST: `build_mem()` 500,000 cases, 0 failed, 2,972 host-served steps, one case skipped because its RAM aliases one cell through echo RAM. The memory engine: 500,000 cases, 0 failed, 502,976 passes all on the fast path, 65 compactions, 260 sampled dense-oracle checks (`GBCFORMER_ORACLE=20000`) exact.

Insertion (the prefill): Yellow's machine state as 1,131,049 records in 0.67–0.69 s (0.6 µs per record by the formula), 290 MB of KV cache at 256 B per token; Crystal 2,179,625 records in 1.20 s, 558 MB.

Compaction: one per 29,444 write records. Yellow writes 0.29 records per step in the text box and 0.21 in the overworld, so a compaction comes every 19 text-box frames and every 75 overworld frames, and costs 42–47 ms on a 1 MB cartridge (115,345 snapshot records at 0.4 µs: the RoPE of each key, 32 `powf`, `cos`, `sin` in f64) and 24–29 ms on smaller ones. That is the hitch to remove before 60 fps: a 46 ms stall every 19 frames in a text box. The copy-mode pass through the model (`REC WP0a WP0b WP0c RD`) is exercised by `tests/memory_dense` and sampled by the oracle; the engine stands in for it with the index read plus the formula.

**model versus model-mem**, `tools/bench/text.sh` and `tools/bench/overworld.sh` with `CORE=model` and `CORE=model-mem` interleaved (A B A B A B), 3 runs each, `GBCFORMER_STATS=1`, load 5.3 before and 5.1 after (the 5-minute average 8 to 10, another A/B and an Arq backup shared the machine):

| Workload | Core | fps (3 runs, median) | µs per pass (median) | Passes | Units per fast pass | Hash |
| - | - | - | - | - | - | - |
| text.sh | model | 64.3, 63.9, 63.9 (63.9) | 2.33, 2.34, 2.35 (2.34) | 4,004,423 (2,413,513 one-pass steps, 795,455 two-pass) | 512 | `870aa3fd` |
| text.sh | model-mem | 17.0, 12.4, 17.0 (17.0) | 10.95, 15.04, 10.94 (10.95) | 3,224,368 (one per step; 15,400 host-served) | 4,314 | `870aa3fd` |
| overworld.sh | model | 146.7, 146.7, 148.8 (146.7) | 2.87, 2.87, 2.83 (2.87) | 1,422,828 | 589 | `822059c1` |
| overworld.sh | model-mem | 42.7, 52.5, 47.9 (47.9) | 11.84, 9.62, 10.56 (10.56) | 1,187,411 (7,585 host-served) | 3,490 | `822059c1` |

Where the time goes (explaining the number). The per-unit cost is unchanged, about 2.5 ns, so the 3.8× (text) and 3.1× (overworld) slowdown is units re-evaluated per pass: 4,314 against 512. The census of gate updates per fast pass by node family (`GBCFORMER_UNITS=1`, text.sh): `WP0` 1,131, `RP0` 718, `WP1` 451, `RA0` 387, `pc+2` 145, `F1` 135, `F0` 134, `F2` 128, `pc+1` 110, `pc+` 93, `op0` 90 (nibble one-hots), `cb*` 75, `op1` 64, `op*` 58, `W0` 54, `jr` 47; `A'` 29, `PC'` 27, `add` 21, `sub` 22. The two write translations and the read translation of the muxed bus addresses, with the request mux, are 2,687 of the 4,314 (62%). The muxed address flips between 0 and an address on every step that writes or reads, and each translation node's steps read the region literals of that address. The fetch translations and the two PC adders add 650, the opcode one-hots from fetched bytes 290. The ALU and final muxes cost what they cost in the host core. Units by layer: [306, 416, 154, 480, 866, 171, 1,259, 610, 9, 43]; 4.5 layer-0 changes per pass against 17.4 (the opcode bytes are no longer inputs).

What would bring model-mem to 60 fps on text.sh (3.1 µs per pass at 3.22M passes): translate the stable address sources (HL, SP−1, SP−2, BC, DE, the immediate) once each and mux the physical results, or hold the bus address when no access uses it, so the translation nodes see a change only when the source changes; both leave every head and guard as they are. Compaction's 41–59 ms hitch (one per 19 text frames) needs the snapshot keys computed without 115,345 f64 RoPE evaluations (one `rope_at` per position class, or incremental compaction).

| # | Change | Overworld µs/pass | Kept |
| - | - | - | - |
| 44 | Layer-0 q, k, v projected through their nonzero cells in `Engine::qkv` (cherry-picked from the profiling run's change). Row 43's hypothesis (per-head `HEAD_DIM` loops in `layer0_changes`) was wrong: phase timers showed those loops ran 36 to 52 times in a 600-frame run (once per cached wire list), while `qkv`'s dense walk over `a = 1408` rows cost 46 to 60 µs per fresh slot, 2,806 slots, 130 to 167 ms of a 4.4 s run (0.9% of the 2.1% that reproduced; the rest below the single-run noise floor) | the base 2.96 → fix 2.88 (142.5 → 146.5 fps), faster in 5 of 5 interleaved pairs, ranges 2.95–3.00 vs 2.87–2.95, load 4.8 → 6.7; base → the next build 2.87 → 2.93, slower 5/5 | yes; bit-identical products (same terms, same order); checkpoint_sst, DMG 8 and CGB 300 hashes, text.sh and overworld.sh hashes unchanged |

### Unit G: one translation per address source, nested page cases, a snapshot keyed on read

Three commits: compaction, translation per source, nested page cases. The circuit is 1,996 wires, 1,716 nodes, depth 9 (unchanged), steps per level [292, 2922, 2106, 752, 802, 1786, 1277, 2512, 40]; the memory checkpoint is 440.5M dense parameters, 204,671 nonzeros, hidden 3072 (the wires passed 2048 channels), intermediate 3264, 10 layers, 23 heads (unit E: 313.3M, 171,186, hidden 2048, intermediate 3584). The host checkpoint is byte-identical: `examples/digest` sha1 `fe67fe79…` before and after, 187.4M parameters, 68,004 nonzeros.

**What changed and why.** The census of unit E put 62% of the gate updates in the translations of the muxed bus addresses (`WP0`, `WP1`, `RP0`) and the request mux `RA0`: the mux output flips between 0 and an address on every step that writes or reads, and every translation step that reads a changed bit updates. Now each stable source is translated once (reads: HL, SP, FF00+C, FF00+n; writes: HL, SP−1, SP−2, FF00+n; `RP1` from SP+1 as before) and the physical results are selected by one wire per source (`translate::select`, a 24-wide mux of the 22 bits, the host flag and the record flag). The sources that come from the instruction bytes or are seldom used (a16, BC, DE) share one bus address (`rare`, `rare0`) that is zero unless one of their opcodes executes, so its translation moves only on those steps. The data heads stay at layer 5 and the depth at 9. HL, SP−1 and SP−2 reach their write translations through copies (`hl1`, `hl2`, `sp1c`, `sp2c`), so those translations sit at level 3 and level 2 keeps the fetch translations only, which holds the intermediate width.

The second change nests the FE/FF-page cases of a translation (`translate::Pages`): one case over a9..a15 carries the bits every address of the two pages shares (14, 15, 17, 21) and passes the low seven bus bits, and OAM, the unusable range, the I/O page, HRAM and IE add only what differs. Before, the IE, HRAM, OAM and unusable conditions read a0..a6 in every bit they set, so a fetch walking a page re-evaluated 17 to 41 steps per changed low bit. The cases of one bit stay disjoint, which `Circuit::eval` asserts and the 20,000-sample test against `layout::translate` exercises. A read translation is 478 steps (was 510), a write 640 (was 744).

The third change is the compaction. The engine's KV cache (`llama::KvCache`) has three segments: the ROM prefix and the writes hold f32 keys and values as pushed; the snapshot between them holds one (address, byte) per record and forms its key by `memory::attn::key` at the record's own index when the dense pass reads it. A compaction is the copy of the writable regions (115,345 bytes) and the dense oracle pays the RoPE once per call. `rope_at` precomputes the 16 denominators `θ^(2t/32)`, which leaves the f64 quotient and so the f32 tables bit-identical (`llama::tests::rope_tables_equal_the_literal_formula`). Compaction also moved to between steps (`Engine::compact_if_full`): before, a compaction triggered inside a step kept that step's record at its pre-compaction position, above every later write for a whole window; the index never showed it and the sampled oracle never hit it.

**Units per fast pass (text.sh, `GBCFORMER_UNITS=1`, single runs; the census is deterministic).**

| Circuit | Units | Largest families |
| - | - | - |
| unit E | 4,314 | WP0 1,131, RP0 718, WP1 451, RA0 387, pc+2 145, F1 135, F0 134, F2 128, pc+1 110, pc+ 93, op0 90, cb* 75 |
| translations per source | 2,458 | rare 195, pc+2 145, F1 135, F0 134, RRARE 132, F2 128, pc+1 110, RFFN 110, WRARE0 105, pc+ 93, op0 90, cb* 75, WHL 54, WFFN 53, WSP1 46 |
| nested page cases | 2,093 | rare 195, pc+2 145, RRARE 113, pc+1 110, pc+ 93, op0 90, F1 85, F0 84, F2 81, WRARE0 81, cb* 75, op1 64, op* 58, W0 54, jr 47, RFFN 46, WHL 32 |
| nested page cases, overworld.sh | 1,938 (unit E: 3,490) | rare 190, pc+2 129, RRARE 113, WRARE0 104, pc+ 98, pc+1 98, op0 88, cb* 71, op1 64, op* 58, W0 56, jr 47, RFFN 44, hl+pair 44, F2 40, F1 40, F0 40 |

The model core's units are unchanged (512 on text.sh, 589 on overworld.sh), because its checkpoint is byte-identical.

**Timing**, `tools/bench/ab.sh` (A B A B A B per workload, 3 runs each, `GBCFORMER_STATS=1`) of unit E's binary (A, the base build) against unit G's (B); load averages 9.3 before, 9.0 after the model-mem block and 9.2 after the model block, with another `model-mem` play window open throughout (the 5-minute average reached 19):

| Workload | Core | A fps (median) | B fps (median) | A µs per pass | B µs per pass | Hash |
| - | - | - | - | - | - | - |
| text.sh | model-mem | 14.8, 15.4, 15.0 (15.0) | 22.2, 19.8, 21.9 (21.9) | 12.56, 12.07, 12.39 (12.39) | 8.38, 9.41, 8.49 (8.49) | `870aa3fd` |
| overworld.sh | model-mem | 44.7, 42.5, 48.7 (44.7) | 62.9, 62.4, 63.2 (62.9) | 11.30, 11.90, 10.38 (11.30) | 8.03, 8.09, 7.99 (8.03) | `822059c1` |
| text.sh | model | 33.8, 58.2, 32.0 (33.8) | 60.0, 59.5, 59.4 (59.5) | 4.43, 2.57, 4.68 (4.43) | 2.50, 2.52, 2.52 (2.52) | `870aa3fd` |
| overworld.sh | model | 140.5, 138.5, 138.9 (138.9) | 142.4, 140.8, 140.2 (140.8) | 3.00, 3.04, 3.04 (3.04) | 2.96, 3.00, 3.01 (3.00) | `822059c1` |

B is faster in every interleaved pair of the model-mem rows. The model core's text.sh A runs 1 and 3 hit a load spike (4.4 to 4.7 µs against A's 2.57 and B's 2.50 to 2.52); its overworld row, free of spikes, has B faster by 1% in 3 of 3 pairs, so the host core is no slower (its checkpoint is byte-identical; the engine changes are the KV cache and `rope_at`, which the host core does not touch per pass).

**Compaction.** 31 compactions in text.sh: 50.7 ms each before (single run, load 8), 0.1 ms each after; insertion 0.67 s to 0.16–0.20 s.

**Explaining the number.** The two sides ran under the same load, interleaved, so the ratio holds: text.sh 12.39 to 8.49 µs per pass for 4,314 to 2,093 units, 1.76 ns per removed unit, the engine's per-unit cost. The remaining 8.49 µs at 2,093 units is about 3.7 µs of units and about 4.8 µs that does not scale with units under this load; the host-served passes (41 positions, 42 units) cost 2.66 to 4.07 µs in the census runs, which bounds the fixed cost per pass from below. `sample` over a text.sh run puts 95% of the samples in `Engine::fill_fast` with everything inlined. A second run of the same source with `fill_fast`'s stages marked `#[inline(never)]` (a scratch build, not committed; 15 s of samples during the model phase, 8.91 µs per 46-position pass, 1.85 µs per 42-unit pass, load 9 to 11) splits them: `Marks::layer` on the circuit layers 56% (the 2,093 units, 2.4 ns each), the routing layer's `changed` callback that walks `plan.owners` per updated routing unit 9%, `memory_reads` 8.5% (16 index reads, about 45 ns each, most of it assembling the 22 query bits from wire terms), the per-position decode in `fill_fast` 8%, `advance`'s per-layer bookkeeping (bucket and mark draining, `route`) 8%, `layer0_changes` 4%, the routing layer's gates 2%; the machine (PPU, APU, bus) is under 2%. Not ruled out: the load (the other play window) inflates both sides by an unknown factor; at unit E's documented load of 5 the base ran 10.95 µs, so B would be near 7.5 µs and 25 fps on text.sh.

**What limits it next.** 60 fps on text.sh is 3.1 µs per pass. The units are about 5 µs of the 8.9 and the fixed cost about 3.3 µs, so both must fall. Units: the fetch path (600: the `pc+1`/`pc+2`/`pc+` adders 348, the three fetch translations 250), the `rare` bus (195, each of its 16 nodes reads the eight `OP0` bits in three steps) with its translations (194), the one-hots from the fetched bytes (287). Fixed cost, by the profile: the owners walk on routing updates (0.8 µs: mark the unit and let the decode check it), the index reads (0.75 µs: precompute each head's query as a channel list once, or read the ten constant-address register heads only when a register record is kept), the decode of 46 positions and the per-layer draining.

### Unit H: hidden 2048 again, and two fixed costs off the pass

Five commits: H1 the level floor, H2 token-channel stamps, H3 head reads, H4 the host core's speed back (a head bucket, owners among mask tokens) and H5 per-layer head lists. The circuit is 1,916 wires, 1,636 nodes, depth 9, steps per level [245, 2874, 2106, 752, 802, 1786, 1277, 2512, 40]; the memory checkpoint is 293.7M dense parameters, 203,941 nonzeros, hidden 2048, intermediate 3264, 10 layers, 23 heads (unit G: 440.5M, 204,671, hidden 3072). The host checkpoint is byte-identical, `examples/digest` sha1 `fe67fe79…`, 187.4M parameters, 68,004 nonzeros.

**What changed and why.** Unit G's 1,996 wires passed the 2,048 channels of hidden 2048 by 73 (the layout is `n_const` + 66 slot identities + 8 own bits + wires + 37 memory channels + 12 routes, and `Layout::used` rounds up to the next 1,024), so the compiler chose hidden 3072. Of those wires, 64 were copies (`hl1`, `hl2`, `sp1c`, `sp2c`) whose only purpose was to push the HL, SP−1 and SP−2 write translations to level 3, off the level the fetch translations fill, and 16 were a second SP+1 adder (`sp+1m`) built in decode beside the datapath's. `Circuit::node_at` takes a level floor: the residual keeps every earlier wire, so a node may sit above the level its inputs force, and the three write translations take floor 3 from their level-0 sources. The datapath reuses decode's SP+1 in memory mode. 80 wires gone, 1,916 left, 7 channels to spare at hidden 2048; the nonzero count falls by 730 (the copies' and the adder's steps).

The other two changes are the two largest fixed costs of unit G's profile. The routing layer's `changed` callback walked `plan.owners` per updated routing unit to stamp `token_changed` for every token owning it; a routing unit of a per-slot output field is owned by its mask token and all 256 of its value tokens, so 44 updated units per pass meant about 11,000 stamps (9%). A token's own units are exactly the units reading one of its nonzero channels, so the callback stamps `chan_pass` on the unit's one or two token channels and the decode cache checks the token's few channels instead (`Template::unit_token_chans`, `token_chans`). And every pass served all 16 memory heads from the index (8.5%). A head's byte is a function of its address and the image, so `MemPlan::head_mask` names the heads whose query reads each channel, those channels carry an extra reader bit that routes their changes to a head bucket, `memory_reads` drains it into a dirty mask, and `Image` counts its changes in a generation. A head that is not dirty and whose byte was read at the current generation keeps its output channels. The ten register heads read only after a record is kept, the fetch and data heads only when their address moved. Index reads per fast pass fall from 16.0 to 6.3 (text.sh) and 5.4 (overworld.sh).

**Units per fast pass (`GBCFORMER_UNITS=1`, single runs, deterministic).** The text.sh workload has 2,082 (unit G 2,093; the 46-position passes 2,092, the 42-unit host-served passes 2.20 µs against 2.75 µs under unit G, single runs at different load), overworld.sh 1,931 (1,938). The families are unit G's: rare 195, pc+2 145, RRARE 113, pc+1 110, pc+ 93, op0 90, F1 85, F0 84, F2 81, WRARE0 81, cb* 75, op1 64, op* 58, W0 54, jr 47, RFFN 46, irq0 44, WHL 32; the copies and `sp+1m` were below the list.

**Timing**, `tools/bench/ab.sh` on `CORE=model-mem` (A B A B A B per workload, 3 runs each, `GBCFORMER_STATS=1`), each commit against the one before it, run after every test job had exited; another `model-mem` play window ran throughout. The 1-minute load is given before and after each A/B. A base to H1 A/B whose overworld block ran beside my own `cargo test` (load 9 to 112; B 18.6 and 22.1 µs) and the first H2 to H3 text block (5-minute load 18; B 10.4 to 12.1 µs against A 7.7 to 11.3) are discarded and were rerun.

| Step | Workload | A fps (median) | B fps (median) | A µs per pass | B µs per pass | Pairs B faster | Load |
| - | - | - | - | - | - | - | - |
| base → H1 | text.sh | 21.5, 21.2, 22.7 (21.5) | 23.0, 22.7, 22.7 (22.7) | 8.64, 8.78, 8.18 (8.64) | 8.10, 8.18, 8.20 (8.18) | 2 of 3, one tie | 32 → 12 |
| base → H1 | overworld.sh | 63.6, 62.8, 58.7 (62.8) | 63.7, 63.4, 63.6 (63.6) | 7.95, 8.04, 8.61 (8.04) | 7.93, 7.97, 7.94 (7.94) | 3 of 3 | 32 → 12 |
| H1 → H2 | text.sh | 22.1, 22.7, 22.3 (22.3) | 21.3, 24.0, 23.9 (23.9) | 8.43, 8.19, 8.34 (8.34) | 8.73, 7.74, 7.78 (7.78) | 2 of 3 | 12 → 8 |
| H1 → H2 | overworld.sh | 62.7, 62.4, 63.4 (62.7) | 65.5, 64.7, 66.6 (65.5) | 8.05, 8.09, 7.97 (8.05) | 7.71, 7.81, 7.58 (7.71) | 3 of 3 | 12 → 8 |
| H2 → H3 | text.sh (4 pairs, rerun) | 24.2, 23.9, 24.2, 24.3 (24.2) | 25.1, 24.6, 25.2, 25.3 (25.2) | 7.68, 7.79, 7.70, 7.65 (7.69) | 7.41, 7.57, 7.38, 7.36 (7.40) | 4 of 4 | 6 to 17 |
| H2 → H3 | overworld.sh | 66.2, 62.0, 66.5 (66.2) | 69.0, 68.9, 69.5 (69.0) | 7.63, 8.15, 7.60 (7.63) | 7.32, 7.34, 7.27 (7.32) | 3 of 3 | 8 → 11 |
| H3 → H3b (head bucket) | text.sh | 25.9, 25.8, 25.4 (25.8) | 26.0, 25.2, 26.0 (26.0) | 7.18, 7.20, 7.33 (7.20) | 7.16, 7.38, 7.16 (7.16) | 2 of 3 | 5 → 4 |
| H3 → H3b | overworld.sh | 71.4, 71.4, 71.8 (71.4) | 72.2, 70.7, 71.6 (71.6) | 7.07, 7.08, 7.04 (7.07) | 7.00, 7.15, 7.05 (7.05) | 2 of 3 | 5 → 4 |
| H3b → H4 (owners) | text.sh | 26.0, 26.0, 26.3 (26.0) | 26.5, 26.1, 26.2 (26.2) | 7.16, 7.16, 7.09 (7.16) | 7.02, 7.13, 7.10 (7.10) | 2 of 3 | 4 → 4 |
| H3b → H4 | overworld.sh | 71.3, 71.6, 72.2 (71.6) | 73.5, 72.7, 72.7 (72.7) | 7.08, 7.06, 7.00 (7.06) | 6.87, 6.96, 6.95 (6.95) | 3 of 3 | 4 → 4 |
| H4 → H5 (per-layer heads) | text.sh | 25.6, 25.6, 25.7 (25.6) | 26.0, 25.9, 26.2 (26.0) | 7.26, 7.27, 7.24 (7.26) | 7.15, 7.18, 7.10 (7.15) | 3 of 3 | 6 → 7 |
| H4 → H5 | overworld.sh | 71.3, 71.5, 70.3 (71.3) | 72.6, 72.2, 72.7 (72.6) | 7.09, 7.06, 7.18 (7.09) | 6.96, 7.00, 6.95 (6.96) | 3 of 3 | 6 → 7 |

Every hash held in every run (`870aa3fd`, `822059c1`). Chaining the medians, text.sh goes 8.64 to 7.15 µs per pass (−17%; 0.947 × 0.933 × 0.962 × 0.994 × 0.992 × 0.985) and overworld.sh 8.04 to 6.96 (−13%); in fps, text.sh 26.0 and overworld.sh 72.6 at load 6 against unit G's 21.9 and 62.9 at load 9 to 19, so the fps ratio also carries the load difference and the µs ratio is the measurement. The absolute numbers of unit G's table (8.49 and 8.03 µs) were taken at that higher load.

**Host core, `CORE=model`, base against the final binary** (the host checkpoint is byte-identical, so this measures the engine changes alone; 3 runs, load 3 to 4): text.sh 2.43, 2.44, 2.43 to 2.47, 2.44, 2.47 µs per pass (61.7 to 60.6 fps; 2 of 3 pairs slower by 0.04 µs, one tie), overworld.sh 2.89, 2.88, 2.91 to 2.89, 2.87, 2.87 (145.7 to 146.7 fps; 2 of 3 faster). Within the noise of one binary against itself in this log (unit G's A runs spanned 2.50 to 2.57 at equal load); counted as no slower. Two earlier states were slower and were fixed before this commit: H3 as committed branched on the memory plan in `route` once per routed change (base 2.53 to 2.83 on text.sh, 3 of 3; 2.98 to 3.01 on overworld.sh), and H2's decode check walked a token's channels per position, which on the host core, where a routing unit has one owner (its mask token), cost more than the owners walk it replaced (base 2.48 to 2.55 after the bucket alone, 3 of 3). H4 is both fixes: the head bucket, and owners restricted to single-channel tokens so the decode check is the one load it was. Base against H5 (3 runs, load 6 to 7): text.sh 2.48, 2.50, 2.49 to 2.47, 2.51, 2.48 (2 of 3 faster), overworld.sh 2.96, 3.00, 2.96 to 2.97, 2.94, 2.99 (1 of 3 faster); the host core's `memory_reads` returns at its first line either way.

**Explaining the number.** H1 removes 11 units per pass (2,093 to 2,082) and is a size change; its 1 to 5% is within the noise between its own A runs (8.18 to 8.78), so it is counted as "no slower", not as a win. H2 and H3 each remove about 0.3 to 0.5 µs per pass, which matches the 9% and 8.5% those two stages took in unit G's profile (8.9 µs then); H3b and H4 add 0.05 to 0.1 µs more on the memory core while returning the host core to its speed, and H5 takes 0.1 µs off `memory_reads` by visiting only a layer's own heads (its 7.2% in the profile below was mostly the ten per-layer walks over all 16 heads). The index-read count is the direct evidence for H3: 16 reads per pass became 6.3, and the ten register heads read only on the 29% of text-box passes that keep a record (the generation moved). Not ruled out: the other play window inflates both sides by an unknown factor, as in unit G.

**What limits it next.** 60 fps on text.sh is 3.1 µs per pass; the pass is 7.15 µs, of which the units are about 5 µs at 2.4 ns each (2,082 of them), so no fixed-cost work reaches the target. The units must fall by two thirds. The families are unit G's and are design-bound: the fetch path (`pc+1`, `pc+2`, `pc+` adders 348 and the three fetch translations 250: a two-level incrementer would halve the adder updates but costs a level the opcode one-hots cannot give), the `rare` bus and its translations (389: `LD A,(a16)` is common in Yellow, so `rare` and `RRARE` move on most steps that use them), the one-hots from the fetched bytes (287: the CB ANDs update on every `OP1` change whether or not the prefix is set). The remaining fixed cost (about 2.2 µs on the 42-unit host-served passes) splits as follows. A `sample` of H4 (10 s during text.sh's model phase, 7.33 µs per pass) puts 93% in `fill_fast` with everything inlined; a scratch copy of the tree with `#[inline(never)]` on `Marks::layer`, `memory_reads`, `advance` and `layer0_changes` (not committed; the same 10 s sample, about 7,100 samples) splits it: `Marks::layer` on the circuit layers 62% (the units), the decode loop in `fill_fast` 9.5%, `advance`'s own draining and routing 7.9%, `memory_reads` 7.2% (before H5: ten calls per pass walked all 16 heads filtering by layer and took the head bucket, with 6.3 reads left), the routing layer's `Marks::layer` and its owners callback 3.5%, `layer0_changes` 3.4%, `mach_absolute_time` 1.3% (the `GBCFORMER_STATS` timers), the machine under 2%. The decode loop's 9.5% is 46 cache checks and about 95 own-unit updates per pass; the draining is ten layers of bucket take, touched-list walk and reader-mask routing. Neither has a single large term left. On size, the layout is 7 channels under 2,048, so the next wire added to the memory circuit returns the checkpoint to hidden 3072 unless a wire goes with it: `RA0` (16) and `W0.a`/`W1.a` (32) are the candidates, both contract changes the host would need to live without.

### Phase 1 prototype: the engine takes a scanline circuit

Prototype of the in-model PPU (`docs/ppu-in-weights.md`, `examples/ppu_proto/`). Two engine checks the CPU checkpoint never exercised stood in the way of its 278-head scanline circuit, and lifting them is the only engine change: `MemPlan` refused more than 32 memory heads (a `u32` dirty mask), now a bitset of `u64` words sized to the heads, kept in `State` beside `head_addr` and `head_gen`; and a layer-0 head slot that another layer fills but layer 0 leaves empty made `plan_attention` refuse every pass (a live head with no keys has no softmax), so a head whose `W_o` is zero now counts as dead, which changes no channel whatever it attends to. `memory_reads` is `#[inline(never)]`, because inlined, its loops cost the host core 3% (table), though the host core never enters them. The host checkpoint is byte-identical (`examples/digest` `fe67fe79`), and `machine::ppu` gained a `Pixel` stage and a `shadow` hook with every frame hash unchanged.

**Timing**, `tools/bench/ab.sh` (A B A B A B per workload, 3 runs each) of the base build (A) against each candidate (B); another `model-mem` play window ran throughout, 1-minute load 4 to 8. Every hash held (`870aa3fd`, `822059c1`).

| Candidate | Core | text.sh A → B µs per pass (median) | overworld.sh A → B | Pairs B slower |
| - | - | - | - | - |
| `ppu.rs` refactor alone | model | 2.49 → 2.47 | 2.94 → 2.91 | 0 of 3, 0 of 3 |
| + engine, dirty heads as `Vec<bool>` | model | 2.50 → 2.57 | 2.95 → 3.05 | 3 of 3, 3 of 3 |
| | model-mem | 7.13 → 7.44 | 6.99 → 7.24 | 3 of 3, 3 of 3 |
| + engine, dirty heads as `u64` words | model | 2.42 → 2.46 | 2.84 → 2.93 | 3 of 3, 3 of 3 |
| | model-mem | 7.18 → 7.45 | 7.09 → 7.33 | 3 of 3, 3 of 3 |
| + `memory_reads` out of line | model | 2.43 → 2.43 | 2.88 → 2.86 | 1 of 3, 1 of 3 |
| | model-mem | 6.89 → 7.05 | 6.79 → 6.90 | 3 of 3, 3 of 3 |
| + `head_dirty` in `State` (committed) | model-mem | 6.94 → 7.01 | 6.83 → 6.92 | 3 of 3, 3 of 3 |

**Explaining the number.** The refactor of `render_line` is free (B faster in 6 of 6 pairs, within noise). The engine change costs the same 3 to 4% on both cores whether the dirty set is a `Vec<bool>` or a word bitset, and the host core never executes either (`memory_reads` returns at its first line without a memory plan), so the cost is code layout in `fill_fast`, where everything is inlined and 95% of the time goes; the perf log saw the same from one branch in `route` (unit H). Taking `memory_reads` out of line returns the host core to a tie (3 of 3 pairs tie or faster on text.sh, B faster 2 of 3 on overworld.sh) and leaves the memory core 1.6 to 2.3% slower; moving `head_dirty` into `State`, which restores `Marks` to the layout `Marks::layer` (the 56% hot loop) reads, leaves 1.0 to 1.3% (3 of 3 pairs each, A runs 6.91 to 6.96 against B 7.01 to 7.04 on text.sh). The remaining difference is the call per layer and the word lookup per head read; it is below the spread unit H counted as "no slower" (1 to 5% between its own A runs) but consistent in sign, so it is recorded as a 1% cost of taking more than 32 heads, not as a tie. Units per fast pass and index reads are unchanged (text.sh with `GBCFORMER_STATS=1`: 2,082 units and 6.3 index reads per fast pass, unit H's numbers, 3,224,368 passes all fast). Not ruled out: the load (4 to 8, the other play window) inflates both sides.

### Per-component stack selection (`--weights`, `--lockstep`)

The binary picked a whole stack by name (`--core ref|model|lockstep|model-mem|lockstep-mem`, one `match` arm per command and core). `src/stack.rs` replaces that with a `Backend` (host, model, lockstep) per `Component` (cpu, memory, ppu, apu, control): `Stack::resolve` applies the dependency table and names the one checkpoint to compile, `Stack::build` compiles it into `AnyCore`, an enum over the built cores (`Reference`, `CheckpointCore`, `MemCheckpointCore`, each with or without `Lockstep`, and an `audit` flag for memory lockstep). The old names stay as aliases. The engine, the circuit and the checkpoint are untouched, so this row checks only that the enum dispatch in `step::step` (one `match` per `request`, `execute`, `served_by_host` and `audit` call) costs nothing measurable.

**Timing**, `tools/bench/ab.sh` (A B A B A B per workload, 3 runs each) of the base build (A) against the stack binary (B), both built from this worktree; 1-minute load 3.8 to 4.5 with another play window open. Every hash held (`870aa3fd`, `822059c1`).

| Core | text.sh A → B µs per pass (runs; median) | overworld.sh A → B | Pairs B slower |
| - | - | - | - |
| model | 2.43, 2.40, 2.41 → 2.44, 2.40, 2.40 (2.41 → 2.40) | 2.90, 2.87, 2.91 → 2.89, 2.90, 2.92 (2.90 → 2.90) | 1 of 3, 2 of 3 |
| model-mem | 7.02, 7.09, 7.06 → 7.04, 7.01, 7.02 (7.06 → 7.02) | 6.97, 6.90, 6.94 → 6.87, 6.93, 6.86 (6.94 → 6.87) | 1 of 3, 1 of 3 |

**Explaining the number.** Every difference is within one binary's own spread (A runs 2.40 to 2.43 and 7.02 to 7.09), with the sign split between pairs, so it is a tie. The dispatch is one predictable branch per call against passes of 2.4 and 7 µs, and the checkpoint, engine and host checkpoint are byte-identical, so no other cost was possible. `model-mem` maps to `--weights cpu,memory --lockstep memory`, which keeps the per-step memory audit the alias always ran; `--weights cpu,memory` without lockstep (audit off) is a new mode and is not compared here.


### Unit I: the circuit reshaped around what changes per step

Six commits: a census by changed wire, translation cases per bit, CB decode by field, the rare bus gated by nibble one-hots with a second data head, fetch adders split at bit 7, and region wires; a seventh, CB (x, y) through OP1's high nibble, was measured, found no faster, and reverted. The circuit is 1,726 wires, 1,438 nodes, depth 9, steps per level [208, 729, 2171, 343, 807, 1143, 1497, 2611, 41]; the memory checkpoint is 266.1M dense parameters, 147,525 nonzeros, hidden 2048, intermediate 2816, 10 layers, 16 memory heads (unit H: 293.7M, 203,941, 1,916 wires, intermediate 3264, 15 heads). The host checkpoint is byte-identical after every commit (`examples/digest` `fe67fe79`, 187.4M, 68,004).

**The lever.** `GBCFORMER_UNITS=1` now also prints the gate updates per fast pass by the *wire that changed* (`Engine::sources`, named through `Model::wire_names`). The engine updates a gate once per changed channel it reads, so a step that reads five changed bits costs five updates; the census by node family said which nodes paid, this says which wires made them pay. Unit H's text.sh census read: `PCL.0` 89, `PCL.1` 85, `PCL.2` 51 (the fetch adders' staircases, where bit k reads every lower bit), `OP0h.*` 304 in all (each fetched bit reached the 48 steps of the `rare` bus, which tested the opcode by its bit pattern), `OP1h.*` 194, `ROMB0.0/1/3` 77 (a bank switch reached every step of the switchable-ROM cases in nine read translations), `rare.14/15` 76 (the bus flipping between 0 and an address). The routing layer's 44 updates had been printed as `irq0`, the first family created. `examples/pchist` (since removed) also counted opcodes and 128-byte fetch crossings: in text.sh's model window 4.5% of steps fetch across a 128-byte block, 7.7% are `LD A,(a16)` or `LD (a16),A`, 15.5% are `LDH`.

**What changed and why**, one commit per accepted change, each gated by the full `cargo test --release` (every SST suite, `memory_dense`, `e01`), `run --frames 8 --core lockstep-mem --hardware dmg` `6872b003`, `run --frames 300 --core lockstep-mem` `cfd31f3d`, text.sh `870aa3fd`, overworld.sh `822059c1` and the host digest:

1. *Translation cases per bit*. A translation's cases need only be disjoint per bit, not partition the domain. The MBC1/2/3 forced-low rule (`max(bank & mask, 1)`) touches only bit 14, yet its `complement()` expansion replicated the seven passed bits and every bank bit across four to eight cases per mapper. Now one case per mapper carries the passed and bank bits and two small cases carry bit 14; ROM0's MBC1 mode-1 bits, the WRAM bank (one main case, four over SVBK), the CRAM bank and the DMA block take the same shape. A read translation is 223 steps (was 478), a write 480 (was 640); a `ROMB0` bit reaches seven steps instead of about 250.
2. *CB decode by field*. Every CB control signal factors through the operation `(x, y)` and the register `z`, so memory mode decodes OP1 into 32 + 8 + 8 one-hots (`cbxy`, `cbz`, `cbw` "writes r") instead of 256 one-hots from 32 nibble one-hots. That is 248 fewer wires, which the layout needed (7 channels were spare at hidden 2048) and the next commits spend. A result field's CB cases carry two gates and its keep case subtracts one cover wire; the cycle count groups the CB table by (HL) and bit. Units stay flat (1,806 to 1,798) yet the pass is 3.4% faster, because the hot loop walks shorter gate rows.
3. *Rare bus by nibble one-hots, a second data head*. The bus tests the opcode by its two nibble one-hots (three terms per step instead of nine), which puts it at level 4 and its translation at level 5, the data heads' layer; so read 0 is two heads, one on the stable sources' selection and one on the rare translation, each landing in VOID (a zero byte) unless its source is selected, and `d0` is their sum. The evidence slots carry `main + rare − VOID`, affine. `rare` 195 to 24 updates per pass. The copy-mode test and the dense reader test now model what the layout guarantees, that the VOID block exists and holds zero; the first cut defaulted the main select to zero when the rare source was selected, which the copy-mode test caught (`RP0b` bit 1 = −1) and the CGB 300 lockstep then confirmed (request diverged at step 87160).
4. *Fetch adders split at bit 7*. `pc+1` and `pc+2` are each the low seven bits and the carry at level 1 and the nine high bits at level 2, reading PC's high bits and the carry; F1 and F2 move to level 3 and their heads to layer 3, which costs no depth because only OP0 feeds the one-hots and every consumer of OP1 and OP2 reads at level 4 or later. `pc+1` 110 to 45, `pc+2` 145 to 57.
5. *Region wires*. A translation may test its regions through six wires one level above its source (`translate::Region::Wired`), so a step reads one wire in place of up to seven address bits and a jump within a region leaves every step but its own passed bit still. F0, RHL, RSP, RP1, WHL, WSP1, WSP2 and WRARE0 take it (F0 44 to 27, WRARE0 59 to 24); F1 and F2 cannot (their heads must stay at layer 3), nor the gated RRARE. WP0 moves to level 8, which widens the intermediate to 2816 (+1.5% dense parameters) while the nonzeros fall 5%.
6. *Reverted: CB (x, y) through OP1's high nibble*. Sixteen nibble nodes between OP1 and the 32 `(x, y)` nodes took 30 units off the pass (`cbxy` 83 to 20) and nothing off the time (below).

**Units per fast pass (`GBCFORMER_UNITS=1`, single runs, deterministic).**

| Commit | text.sh | overworld.sh | Largest families on text.sh |
| - | - | - | - |
| unit H | 2,082 | 1,931 | rare 195, pc+2 145, RRARE 113, pc+1 110, pc+ 93, op0 90, F1 85, F0 84, F2 81, WRARE0 81, cb* 75, op1 64, op* 58, W0 54 |
| cases per bit | 1,806 | 1,764 | rare 195, pc+2 145, pc+1 110, pc+ 93, op0 90, cb* 75, op1 64, WRARE0 59, op* 58, W0 54, RRARE 49, jr 47, F1 44, F0 44, F2 42 |
| CB by field | 1,798 | 1,760 | cbxy 83 and cbw 21 replace cb* 75 and op1 64 |
| rare by nibbles | 1,631 | 1,604 | rare 24, RRARE 69 (its gate is in every step), index reads 6.3 to 6.5 |
| fetch adders split | 1,478 | 1,468 | pc+ 93, op0 90, cbxy 83, RRARE 69, WRARE0 59, W0 58, op* 58, pc+2 57, jr 47, pc+1 45, F1 44, F0 44, F2 42 |
| (cbxy via nibble, reverted) | 1,448 | 1,439 | cbxy 20, op1 33 |
| region wires, final | 1,416 | 1,414 | pc+ 93, op0 90, cbxy 83, RRARE 69, W0 58, op* 58, pc+2 57, jr 47, pc+1 45, F1 44, F2 42, route 42, A' 38, RFFN 32, F0 27, PC' 25, WRARE0 24 |

**Timing**, `tools/bench/ab.sh` on `CORE=model-mem` (A B A B A B per workload, 3 runs each), each commit against the one before it, no tests running; another `model-mem` play window ran throughout. The 1-minute load is given before and after each A/B.

| Step | Workload | A fps (median) | B fps (median) | A µs per pass | B µs per pass | Pairs B faster | Load |
| - | - | - | - | - | - | - | - |
| unit H → cases per bit | text.sh | 26.3, 26.4, 26.2 (26.3) | 28.0, 27.9, 27.9 (27.9) | 7.07, 7.06, 7.09 (7.07) | 6.65, 6.68, 6.66 (6.66) | 3 of 3 | 3.6 → 3.0 |
| | overworld.sh | 72.9, 72.6, 73.1 (72.9) | 75.3, 76.0, 75.7 (75.7) | 6.94, 6.96, 6.92 (6.94) | 6.71, 6.65, 6.68 (6.68) | 3 of 3 | |
| → CB by field | text.sh | 27.8, 27.9, 27.6 (27.8) | 28.8, 28.8, 28.8 (28.8) | 6.69, 6.67, 6.75 (6.69) | 6.46, 6.46, 6.46 (6.46) | 3 of 3 | 3.0 → 3.8 |
| | overworld.sh | 75.4, 75.2, 75.4 (75.4) | 76.6, 77.8, 77.5 (77.5) | 6.70, 6.72, 6.70 (6.70) | 6.60, 6.50, 6.52 (6.52) | 3 of 3 | |
| → rare by nibbles | text.sh | 28.8, 28.8, 28.8 (28.8) | 29.8, 30.0, 29.6 (29.8) | 6.47, 6.46, 6.47 (6.47) | 6.25, 6.20, 6.30 (6.25) | 3 of 3 | 3.8 → 6.8 |
| | overworld.sh | 74.4, 74.6, 74.5 (74.5) | 78.5, 78.3, 77.5 (78.3) | 6.80, 6.77, 6.78 (6.78) | 6.44, 6.45, 6.52 (6.45) | 3 of 3 | |
| → fetch adders split | text.sh | 28.9, 29.0, 29.1 (29.0) | 30.2, 30.1, 30.1 (30.1) | 6.43, 6.42, 6.41 (6.42) | 6.16, 6.19, 6.19 (6.19) | 3 of 3 | 6.8 → 6.6 |
| | overworld.sh | 79.1, 78.6, 78.7 (78.7) | 81.7, 81.5, 81.3 (81.5) | 6.39, 6.43, 6.42 (6.42) | 6.18, 6.20, 6.22 (6.20) | 3 of 3 | |
| → cbxy via nibble (reverted) | text.sh | 31.1, 31.3, 30.9 (31.1) | 30.8, 30.8, 31.0 (30.8) | 5.99, 5.95, 6.02 (5.99) | 6.05, 6.03, 6.01 (6.03) | 0 of 3 (one tie) | 2.8 → 4.8 |
| | overworld.sh | 83.5, 84.0, 83.0 (83.5) | 82.3, 83.5, 83.0 (83.0) | 6.05, 6.01, 6.09 (6.05) | 6.14, 6.05, 6.09 (6.09) | 0 of 3 (one tie) | |
| cbxy via nibble → region wires | text.sh | 30.6, 30.8, 30.9 (30.8) | 31.3, 31.4, 31.3 (31.3) | 6.08, 6.05, 6.03 (6.05) | 5.94, 5.93, 5.95 (5.94) | 3 of 3 | 4.8 → 3.5 |
| | overworld.sh | 82.1, 83.0, 82.8 (82.8) | 84.1, 83.5, 84.6 (84.1) | 6.15, 6.09, 6.10 (6.10) | 6.01, 6.05, 5.97 (6.01) | 3 of 3 | |
| fetch adders split → final (region wires, the nibble commit reverted) | text.sh | 29.8, 30.2, 30.3 (30.2) | 30.7, 30.4, 30.4 (30.4) | 6.25, 6.16, 6.13 (6.16) | 6.07, 6.12, 6.12 (6.12) | 3 of 3 | 4.4 → 8.1 |
| | overworld.sh | 80.4, 79.1, 80.1 (80.1) | 81.5, 81.7, 81.5 (81.5) | 6.29, 6.38, 6.30 (6.30) | 6.20, 6.19, 6.20 (6.20) | 3 of 3 | |

Every hash held in every run. Chaining the medians of the kept steps (unit H, cases per bit, CB by field, rare by nibbles, fetch adders, final), text.sh goes 7.07 to 5.95 µs per pass (−15.8%; 0.942 × 0.966 × 0.966 × 0.964 × 0.994) and overworld.sh 6.94 to 5.88 (−15.3%; 0.963 × 0.973 × 0.951 × 0.966 × 0.984); in fps at the loads measured, text.sh 26.3 to 30.4 and overworld.sh 72.9 to 81.5. The region-wires step measured 1.8% on text.sh against the reverted nibble commit and 0.7% against the fetch-adders commit, under a load that rose from 4.4 to 8.1 during the second A/B; both are 3 of 3 pairs and it is kept as a small win. The absolute µs of one binary moved with the load between A/Bs (the fetch-adders binary ran 6.19, 5.99 and 6.16 in three A/Bs), so the ratios within an A/B are the measurement.

**Host core, `CORE=model`, unit H's binary against the region-wires binary** (byte-identical checkpoint, so this measures the engine's census branch alone; 3 runs, load 3.5 to 3.7): text.sh 2.41, 2.42, 2.46 to 2.43, 2.47, 2.44 µs per pass (62.0 to 61.3 fps; B slower in 2 of 3 pairs by 0.02 to 0.05), overworld.sh 2.90, 2.93, 2.87 to 2.88, 2.86, 2.84 (145.3 to 147.6 fps; B faster 3 of 3). Against the fetch-adders binary earlier (load 6.6 to 5.2): text.sh 2.49 to 2.50 (a tie), overworld.sh 2.96 to 2.94. Mixed in sign and within the 1 to 5% spread unit H counted as no slower.

**Explaining the number.** A clean `GBCFORMER_STATS=1` run of the region-wires binary on text.sh (load 3.5) splits the pass: the 46-position passes cost 5.62 µs at 1,393 units, the 42-unit host-served passes 1.12 µs, so the fixed cost is about 1.1 µs, not the 2.2 µs unit H inferred from a census run (`GBCFORMER_UNITS` itself costs about a nanosecond per update), and the units with the changes that drive them are about 4.5 µs of the 5.6. The per-commit deltas put a removed unit at 1.3 to 1.5 ns (unit G: 1.76), below the 3.2 ns an average unit costs by that split. The units removed were short gate rows (a `rare` step has nine terms, a bank case a dozen), and the pass also pays per *changed wire* (the bucket push, the row walk, the touched-list drain), which the census does not count. Two results fit that reading: the CB-by-field commit was unit-neutral and 3.4% faster (256 fewer units and 248 fewer channels, shorter rows), and the reverted commit took 30 units off and nothing off the time (16 nodes added, each a changed wire on every OP1 change). Not ruled out: the load (2.8 to 6.8, the other play window) inflates both sides by an unknown factor, as in units G and H; the per-change cost is inferred from the deltas, not profiled.

**What limits it next.** The families left on text.sh at 1,416 units are the next-PC adder (`pc+` 93: its low nibble reads PC, the length bits, the HALT-bug bit and the offset), the opcode decode itself (`op0` 90 nibble one-hots and `op*` 58, inherent to a new opcode each step), the CB fields (`cbxy` 83, each node reading five OP1 bits; the nibble form was no faster), the rare translations flipping between VOID and an address (`RRARE` 69, `WRARE0` 24), the output muxes (`W0` 58, `A'` 38, `PC'` 25, `route` 42: a new opcode class flips two selectors per field), the offset `jr` 47 (ten steps per bit; isolating it by one wire pushes `pc+` a level, and `W0.v` reads `pc_seq` for CALL, which would push WM0 past level 9 unless the MBC2 nibble rule folds into the value mux), and the F1/F2 translations (86) that cannot take region wires without their heads moving to layer 4. None is above 7% of the pass. 60 fps on text.sh is 3.1 µs per pass; at 1.1 µs fixed and about 3.2 ns per unit including the changes that drive it, that is about 600 units, so the units must fall by more than half again. The next measurement to take is a count of changed wires per pass beside the units, since the deltas say the pass pays for both; the next design candidates are a decoded-opcode class alongside each ROM byte in the memory (which would gate `rare`, the CB fields and the length bits from a head output instead of a decode), and the fixed cost's decode of 46 positions and ten layers of draining.

### DMA copies as records (unit J)

Every OAM DMA, HDMA and GDMA byte becomes a reader record at the source and a write record at the destination (`docs/memory-in-weights.md` §8a), and the memory audit compares the model's VRAM and OAM with the host's after every step that copied and every writable region at each frame end. Two commits, the audit and the records, and the audit's slice comparison below.

**Timing**, scratch copies of `tools/bench/ab.sh` with the ROM by absolute path, `CORE=model-mem`, A = the base build, B = this work, A B A B A B, 3 runs each. Another `model-mem` play window and its own A/B ran throughout, and a `GBCFORMER_ORACLE=997` run of this work ran on one core.

| B binary | Workload | A fps | B fps | A µs per pass | B µs per pass | Records A → B | Compactions A → B | Load |
| - | - | - | - | - | - | - | - | - |
| records (audit compares byte by byte) | text.sh | 30.2, 30.2, 29.8 (30.2) | 27.6, 29.6, 29.3 (29.3) | 6.16, 6.16, 6.24 (6.16) | 6.73, 6.29, 6.36 (6.36) | 928,869 → 1,024,869 | 31 → 34 | 9.9 → 7.1 |
| | overworld.sh | 81.5, 82.5, 81.7 (81.7) | 78.3, 77.6, 78.3 (78.3) | 6.20, 6.13, 6.18 (6.18) | 6.45, 6.51, 6.46 (6.46) | 245,818 → 341,818 | 8 → 11 | |
| audit compares slices first | text.sh | 28.0, 29.8, 30.0 (29.8) | 29.9, 29.8, 29.8 (29.8) | 6.65, 6.23, 6.20 (6.23) | 6.22, 6.24, 6.23 (6.23) | 928,869 → 1,024,869 | 31 → 34 | 9.1 → 5.6 |
| | overworld.sh | 81.0, 82.1, 82.9 (82.1) | 81.4, 81.4, 82.6 (81.4) | 6.24, 6.16, 6.09 (6.16) | 6.21, 6.21, 6.12 (6.21) | 245,818 → 341,818 | 8 → 11 | |

Every hash held (870aa3fd, 822059c1). Both workloads copy exactly 160 bytes per frame (96,000 over 600 frames, one OAM DMA per frame, no HDMA), so the records grow by 10% on text.sh and 39% on overworld.sh, and compactions by the same share, at 0.1 ms each.

**Explaining the number.** The first B lost 3.2% on text.sh and 4.5% on overworld.sh, 0.2 to 0.3 µs per pass, or about 6.7 and 3.4 µs per copied byte. That is not the records. The audit compared VRAM and OAM (16.5 KB) with a per-byte `find` after every step that copied, about 80 steps per frame during the OAM DMA wait loop, roughly 1.3 MB per frame through a scalar loop. Comparing the slices with `==` (memcmp) first and searching only on a mismatch takes text.sh to a tie and overworld.sh to −0.8% (B slower in 3 of 3 pairs, by 0.03 to 0.05 µs per pass). The remaining cost is the records themselves: per byte, one index read, the `attn::key` formula (`rope_at` forms 16 f64 sin/cos pairs per call, no table), the KV push, and the image write; about 0.3 µs per byte by the overworld delta. Not ruled out: the load (5.6 to 10) and the other runs; the text.sh tie sits inside the spread of the A runs.

**Host core, `CORE=model`**, the base against the final binary (the copy log moved below `hram` so the `repr(C)` hot prefix is as on the base), 3 runs, load 9.6 to 9.3: text.sh 2.75, 2.67, 2.62 to 2.66, 2.63, 2.60 µs per pass (56.2 to 57.0 fps), overworld.sh 3.11, 3.07, 3.10 to 3.08, 3.06, 3.09 (136.2 to 136.9 fps); B faster in 6 of 6 pairs by under 3%, inside the load's spread, so no slower. The host core logs the copies too (160 per frame) and its `copied` hook is a no-op.

**Oracle.** `GBCFORMER_ORACLE=997` on lockstep-mem CGB 300 reached `cfd31f3d` with 186 dense checks (every 997th record's key and value, the reader record of each sampled copy, and the compaction samples), 46,912 copied bytes, none host-served; 2,877 s under load.


### Phase 2 prototype: the APU shadow hook

`apu::Event` and `Apu::shadow` (`docs/apu-in-weights.md`) add three `is_some()` checks to the APU's hot path: one per register write, one per DIV edge and one per sample, none per tick but the edge test the tick already computed. `tools/bench/ab.sh` of the base build (A) against the hook binary (B), `CORE=model`, A B A B A B, 3 runs each; 1-minute load 12.7 before, 10.4 after, other work running. Every hash held (`870aa3fd`, `822059c1`).

| Core | text.sh A → B µs per pass (runs; median) | overworld.sh A → B | Pairs B slower |
| - | - | - | - |
| model | 2.57, 2.52, 2.59 → 2.53, 2.53, 2.57 (2.57 → 2.53) | 3.09, 3.13, 3.13 → 3.12, 3.16, 3.12 (3.13 → 3.12) | 1 of 3, 2 of 3 |

**Explaining the number.** Every difference sits inside one binary's own spread (A runs 2.52 to 2.59 and 3.09 to 3.13), with the sign split between pairs, so it is a tie. The checks are predictable branches on a field that is `None` in every benchmark run, and the APU is a few percent of the host side of a 2.5 µs pass. The checkpoint is byte-identical (`examples/digest` `fe67fe79`).


### Size: the level schedule and the grouped gather

Four commits: the level schedule, four input slots per layer-0 head, the schedule by rounded width and spread with the copy head on the emptiest layer, and shape and projection printouts. The goal was fewer dense parameters with every token unchanged. Dense parameters are `2·vocab·d + layers·(4·a·d + 3·d·i + 2·d) + d`, where `i` is the widest layer's units rounded up to 64 and `a = 32 ·` the most heads in any layer. Before this round both checkpoints paid for one wide level and for layer 0's gather in every layer.

| Checkpoint | Dense | Nonzeros | `i` | Heads (`a`) | Hidden | Layers |
| - | - | - | - | - | - | - |
| host, before | 187.4 M | 68,004 | 2,944 | 36 (1,152) | 2,048 | 6 |
| host, after | 109.5 M | 67,989 | 1,472 | 21 (672) | 2,048 | 6 |
| memory, before | 266.1 M | 147,525 | 2,816 | 23 (736) | 2,048 | 10 |
| memory, after | 153.4 M | 147,510 | 1,536 | 10 (320) | 2,048 | 10 |

**What changed and why.**
1. *Level schedule* (`Circuit::schedule`). A node may sit anywhere from one level above its inputs to one level below its earliest reader. A head's query must be ready by the head's layer and an output by the depth. The residual keeps every wire, so the move changes only which layer computes the node. The pass runs earliest-deadline list scheduling under each width cap from the mean to the current widest. Widths are compared rounded up to 64. Among the narrowest, it keeps the placement whose wires are read in the fewest layers, the original placement included. Host units per level went from [936, 756, 1253, 2901, 23] to [936, 756, 1253, 1455, 1469]. The output muxes at level 4 feed only the routing layer, so half of them move to level 5. Memory went from [354, 772, 2315, 373, 1179, 1529, 1728, 2805, 42] to [354, 1261, 1533, 666, 1179, 1529, 1530, 1532, 1513]. Depth, heads and steps are unchanged, so the nonzeros are too.
2. *Grouped gather.* Layer 0 had one head per input slot, and each used 8 of its 32 value dims. A head now keys on four slots with equal logits (181 each, every other key's weight exactly 0), so each key weighs exactly 1/4. Each slot's byte sits in its own lane of 8 token channels (`Layout::lane`, 32 channels after the routes), and the value is scaled by 4. Groups are powers of two, so every weight is exact. Host layer 0 drops from 20 + 16 one-hot blocks to 5 + 16 heads. Memory drops from 20 + 3 address gathers to 5 + 3, so layer 1's memory heads set `a`. The engine's exclusive plan asks, per value dim, whether at most one key's token can reach that dim (`Engine::v_reach`, from the embeddings and each token's home). It builds a position's wire list from its own key only, so the group heads keep the cached per-token wire lists.
3. *Copy head.* Its query is the gathered address and its byte goes to the routing layer, so any circuit layer after 0 serves. It now sits on the layer with the fewest circuit heads (layer 4), which takes layer 1 from 11 heads to 10.
4. The 15 fewer nonzeros on each checkpoint are the q entries of the 15 gather heads that went away.

**Host digest re-baselined on purpose.** `examples/digest` moved from `fe67fe79` (187.4M) to `1f4c3519fac3b303a88452d88814a072f19da9e8` (109.5M). The levels and layer 0 changed. Every output token is unchanged. `checkpoint_sst` passes and the engine matches the dense pass.

**Gates on the final tree.** `tools/gate.sh` passed in full: build without warnings, wasm32, DMG 8 lockstep `6872b003`, CGB 300 lockstep `cfd31f3d`, `cargo test --release` (circuit and checkpoint SST, `build_mem` SST, the memory engine, `memory_dense`, e01, interrupt fuzz), Crystal `4e6a7509`, Zelda DX `1d00d871`, Mario DX `7dec3947`, text.sh `870aa3fd`, overworld.sh `822059c1`, blargg `cpu_instrs`, `instr_timing`, `dmg_sound` 01 to 04, and dmg-acid2 and cgb-acid2 pixel-exact. `--core lockstep-mem` reached DMG 8 `6872b003` and CGB 300 `cfd31f3d`. Each fast-path run reported 0 dense fallbacks.

The sampled oracle also held. `GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem` ran 21 dense oracle checks, with 0 dense fallbacks and frame `6872b003`. A 300-frame run at `GBCFORMER_ORACLE=50` was stopped after 19 CPU minutes, because each check is a dense pass.

**Timing.** `tools/bench/ab.sh` (A B A B A B per workload) shows µs per pass, as the median of each side's 3 runs. "Pairs B slower" counts the pairs where B was slower. Other play windows and builds shared the machine throughout. The 1-minute load is given for each A/B.

| A → B | Core | text.sh µs | overworld.sh µs | Pairs B slower (text, overworld) | Load |
| - | - | - | - | - | - |
| the base build → schedule | model | 2.58 → 2.62 | 3.01 → 3.12 | 2 of 3, 3 of 3 | 18.5 → 10.2 |
| schedule → grouped gather | model | 2.64 → 2.66 | 3.13 → 3.15 | 2 of 3, 1 of 3 (one tie) | 10.2 → 8.9 |
| base → schedule | model-mem | 6.18 → 6.46 | 6.32 → 6.48 | 3 of 3, 3 of 3 | 8.9 → 8.8 |
| schedule → grouped gather | model-mem | 6.69 → 6.59 | 6.46 → 6.52 | 0 of 3, 3 of 3 | 8.8 → 9.6 |
| the base build → final | model | 2.47 → 2.61 | 2.92 → 3.05 | 3 of 3, 3 of 3 | 6.4 → 7.0 |
| grouped gather → final | model | 2.60 → 2.66 | 3.09 → 3.11 | 3 of 3, 2 of 3 | 7.0 → 8.2 |
| base → final | model-mem | 6.15 → 6.40 | 6.03 → 6.23 | 3 of 3, 3 of 3 | 8.2 → 4.5 |
| grouped gather → final | model-mem | 6.19 → 6.15 | 6.20 → 6.17 | 0 of 3, 0 of 3 | 4.5 → 3.9 |

Every hash held in every run (`870aa3fd`, `822059c1`).

**Explaining the number.** The size cut costs about 4 to 6% per pass on the host core (text 2.47 to 2.61 µs, 60.8 to 57.3 fps) and 3 to 4% on the memory core (6.15 to 6.40, 30.3 to 29.1 fps). The schedule causes it. The grouped gather and the copy head measured a tie or faster on the memory core. The work counted per pass is unchanged. A CGB 300 `GBCFORMER_STATS=1` run gives 1,161 units per fast pass on `model-mem` before and after. Only the per-layer split moved: [97, 19, 110, 120, 207, 228, 166, 187, 5, 22] became [97, 26, 95, 128, 207, 228, 149, 106, 103, 22]. The host gives the same 17.5 layer-0 changes; [73, 214, 156, 204, 3, 8] became [73, 214, 156, 108, 98, 8]. The 46-position passes pay the extra (3.94 to 4.15 µs), and the 41-position host-served passes do not (1.04 to 0.96). That points to work per changed wire per reader layer. Half of the output muxes moved from level 4 to level 5, so a changed selector or one-hot wire now walks gate rows in two layers instead of one. This is inferred, not proven. The static count of (wire, reader layer) pairs rose from 2,736 to 2,908 on the host and from 2,623 to 2,932 on memory. A local search that cut the memory count to 2,876 gave 4.38 µs, against 4.40 for the plain schedule and 4.31 for the base (5 interleaved CGB 300 runs each, load 8.5). It was dropped, so the static proxy does not explain the clock. The schedule keeps a tie-break on that count, which is free. On the host, the final placement ([…, 1455, 1469] against the first schedule's [… 1460, 1464]) measured about 2% slower again (2.60 to 2.66 µs, 3 of 3 pairs), the same sign as the schedule itself. If speed matters more than size on the host core, the grouped gather alone gives about 163.8 M with no schedule. The next measurement is the dynamic count: each changed wire weighted by its reader layers, from `Engine::sources`.

**Re-projected PPU shapes.** See `docs/ppu-in-weights.md` §3a (`ppu_proto --project`).

**What limits size next**, measured on the IR (`shape` prints the units per level and the live wires per layer):
- *Channel reuse.* A wire's channel could be freed after its last reader. At most 931 memory wires are live at once (at layer 5), against 1,726 allocated. Hidden 1,024 needs `1 + 66 + 8 + live + 37 + 12 + 32 ≤ 1,024` (one constant, 66 positions, the own byte, the memory channels, the routes, the lanes), so live ≤ 868, or 876 if lane 0 reuses `own_bit`. That would take the memory checkpoint from 153.4 M to about 77 M. Two things are needed: a schedule that also caps the live count, and an engine that versions a channel per layer. Today `Template::check` allows one writer per channel and `State::x` holds one value per channel. A reused channel also needs one clearing unit per reuse in the writing layer. The host cannot reach hidden 1,024, because its 256 one-hot token channels plus 937 live wires exceed it. For the merged PPU, see §3a.
- *Step splitting.* A node's steps that read only earlier wires could form their own wire one level down. The memory circuit's mean is 1,233 units per level against 1,533 at the widest, so `i` could reach 1,280, about 15.7 M fewer. The host's levels 3 to 5 already average 1,392, so `i` could fall from 1,472 to 1,408 at most, 2.4 M. Each split costs a wire and a looser range on its readers, and `eval_mem`'s per-node 0/1 assertion would need a partial-node kind.
- *Attention.* The host's 16 one-hot gather heads (256 dims per one-hot) now set its `a`. Nibble one-hots would add a level, and the host has no slack for one. The memory checkpoint's 10 constant-address register heads at layer 1 set its `a`: F0's translation needs every banking register by level 2.


### Phase 2: the APU checkpoint as the machine's APU

`--weights apu` (`docs/apu-in-weights.md` §11). Two things touch the paths without the
APU in weights: the machine's APU slot is an enum (the reference or a replacing device),
and every bus read checks FF10–FF3F before syncing the APU; `run` now also drains audio
every frame, as `play` does. Scratch copies of `tools/bench/ab.sh`, A = the base build,
B = this work, A B A B A B, 3 runs each.

| Core | text.sh A → B µs per pass (runs; median) | fps (median) | overworld.sh A → B µs per pass | fps (median) | Load |
| - | - | - | - | - | - |
| `model` | 2.46, 2.50, 2.46 → 2.50, 2.46, 2.47 (2.46 → 2.47) | 60.9 → 60.6 | 2.95, 2.96, 2.96 → 2.94, 2.97, 2.94 (2.96 → 2.94) | 142.4 → 143.5 | 5.0 → 4.9 |
| `model-mem` | 6.07, 6.42, 6.17 → 6.08, 6.13, 6.22 (6.17 → 6.13) | 30.2 → 30.4 | 6.19, 6.23, 6.36 → 6.33, 6.24, 6.31 (6.23 → 6.31) | 81.1 → 80.1 | 5.2 → 8.5 |

**Explaining the number.** It is a tie. Every B median sits inside A's spread, and the sign
splits between workloads. The range check is two compares on a read that already
routes, and the enum match is one predictable branch per block. Every hash held.

**The full stack**, same binary, `--weights cpu,memory` (A) against
`--weights cpu,memory,apu` (B), then the APU thread (A) against the APU inline (B,
`GBCFORMER_APU_THREAD=0`):

| Pair | text.sh A → B fps (runs; median) | overworld.sh A → B fps | Load |
| - | - | - | - |
| APU on the host → APU thread | 30.1, 30.9, 31.6 → 27.9, 29.6, 30.5 (30.9 → 29.6) | 82.8, 84.7, 81.0 → 79.7, 46.5, 79.4 (82.8 → 79.4) | 8.3 → 5.5 |
| APU thread → APU inline | 23.7, 30.5, 30.3 → 22.9, 22.9, 23.2 (30.3 → 22.9) | 70.1, 80.6, 78.7 → 44.4, 38.6, 43.1 (78.7 → 43.1) | 5.5 → 5.9 |

**Explaining the number.** Inline, the APU's 815 passes per frame add about 11 ms per
frame on overworld.sh (1/43.1 − 1/82.8 s) and 10.6 ms on text.sh, close to the
prototype's 9.1 to 9.7 ms (its 3,850 units per pass against 3,896 here). On its own
thread the frame takes about the longer of the CPU side (12 ms on overworld.sh, 33 ms
on text.sh) and the APU, so text.sh hides it entirely and overworld.sh nearly: −4% at
the medians. The low thread runs (46.5, 70.1, 23.7 fps) coincide with other
work on the 10-core machine; that they are single runs between normal ones suggests
the APU thread waiting for a core rather than the protocol (a guess: no profile was
taken of them). Reads cost one pass each and
are rare: 0.61 per frame on text.sh, 0.23 on overworld.sh.

### Size and speed: channel reuse and the changed-pair placement

Three commits: the channel plan (clearing units, virtual channels in the engine, the changed-pair counters), the fit (`Circuit::fit` in `sm83::build_with`, the recorded change tables) and the commit that carries this section (virtual channels numbered by wire, the host left unfitted). The goal was the memory checkpoint at hidden 1,024 and the scheduling round's speed back, with every token unchanged.

| Checkpoint | Dense | Nonzeros | `i` | Heads (`a`) | Hidden | Layers | Live wires at most |
| - | - | - | - | - | - | - | - |
| host, before and after | 109.5 M | 67,989 | 1,472 | 21 (672) | 2,048 | 6 | 937 |
| memory, before | 153.4 M | 147,510 | 1,536 | 10 (320) | 2,048 | 10 | 931 |
| memory, after | 82.6 M | 150,084 | 1,728 | 10 (320) | 1,024 | 10 | 868 |

**What changed and why.**
1. *The measurement first.* `Stats::routed` counts the (changed channel, reading layer) pairs a fast pass walks, each a bucket push and a gate row; `GBCFORMER_UNITS=1 GBCFORMER_WIRE_CHANGES=<file>` writes each wire's changes per pass. With the schedule switched off in a scratch build (all else equal, text.sh, 3 interleaved runs each, load 6 to 9), the memory core walked 167.9 pairs per pass against 189.4 with it, at the same 1,423 units, and ran 6.30 against 6.50 µs per pass; the host core 82.9 against 93.8 pairs, 2.61 against 2.70 µs. The scheduling round's slowdown came with the pairs, about 9 ns each, as the last section inferred.
2. *Channel plan* (`Circuit::channels`). A wire may take the channel of a wire whose last reader is at a layer below the new wire's level. One MLP unit in that window, `−[x ≥ 1]` for a 0/1 wire or `[−x ≥ 1]` for a −1/0 wire, subtracts the dead value, reading only that channel; it goes to the emptiest layer of its window. Wires are placed in level order, each taking the free channel that freed first, so the channel count is the widest live count (interval colouring). A wire never takes the channel of a wire it reads, so a unit that reads the channel it writes is always a clearing unit. The compiler uses the plan only when it saves a block of 1,024 channels, so the host layout is as before.
3. *Virtual channels in the engine* (`Segments`). The engine splits each shared dense channel into one virtual channel per occupant, recognises the clearing units, evaluates them on −1, 0 and 1, and keeps the range where they negate exactly as a per-channel guard, and a pass that leaves it falls back. Clearing units are dropped, every virtual channel has one writer, and the hot loop is unchanged but for the range lookup. Occupants are numbered `hidden + wire`, so the bits of a byte sit together as in the old layout (any numbering is exact). A unit test (`reused_channels_match_the_dense_pass_and_the_circuit`) builds a 1,168-wire chain that fits 1,024 channels only with reuse, with both clearing forms, and compares the fast path with the dense pass and the circuit; with a clearing coefficient of −2 instead of −1 it fails against the circuit.
4. *The placement* (`Circuit::fit`, memory circuit only). A deterministic hill climb over single-node level moves (the moves `schedule` makes) scores, in order: live wires above the channel budget at any boundary (`Model::wire_budget`, 868 at hidden 1,024: 1 constant, 66 slots, 8 own bits, 37 memory, 12 route and 32 lane channels), node units above a level's cap, Σ changes(w) · layers reading w over `src/sm83/changes-memory.txt` (the mean of text.sh and overworld.sh), and the spread of units over levels. Each width cap (the current width and one and two quanta wider) runs up to six rounds; after each, the plan's clearing units are placed and the next round's node caps leave their room a quantum narrower. The clearing units set `i`: every channel is busy at boundary 5, so every wire written after it needs its clearing unit at levels 6 to 9, about 230 at level 6. One round gives `i` 1,856; the rounds give 1,728. `fit` costs about 2 s per build of the memory circuit. The host cannot reach its 677 channels (937 live wires beside its 256 one-hot token channels), and a pairs-only placement of the host measured no faster than the plain schedule (2.67 against 2.66 µs on text.sh), so the host is not fitted and its digest stays `1f4c3519`.

**Gates on the final tree.** `tools/gate.sh` from a scratch copy with absolute paths: build without warnings, wasm32, DMG 8 lockstep `6872b003`, CGB 300 lockstep `cfd31f3d`, the same two on `lockstep-mem`, every test binary run from the main checkout (circuit and checkpoint SST, `build_mem` SST, the memory engine, `memory_dense`, `dma_records`, e01, interrupt fuzz, the library tests), Crystal `4e6a7509`, Zelda DX `1d00d871`, Mario DX `7dec3947`, text.sh `870aa3fd`, overworld.sh `822059c1`, blargg `cpu_instrs`, `instr_timing`, `dmg_sound` 01 to 04, and dmg-acid2 and cgb-acid2 pixel-exact; it passed on the fit and again on the final tree. `GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem` ran 27 dense oracle checks with 0 dense fallbacks and frame `6872b003`. Every A/B run below held `870aa3fd` and `822059c1`.

**Timing.** Scratch copies of `tools/bench/ab.sh` (A B A B A B per workload), µs per pass as the median of each side's 3 runs. "remap" is the channel plan with identity channels on both checkpoints; "fitplain" is the fit's placement with the layout forced plain (hidden 2,048); "full" is the fit; "renum" is the final tree. Other play windows and builds shared the machine.

| A → B | Core | text.sh µs | overworld.sh µs | Pairs B slower (text, overworld) | Load |
| - | - | - | - | - | - |
| the base build → remap | model-mem | 6.13 → 6.13 | 6.28 → 6.62 | 1 of 3 (one tie), 2 of 3 | 11.1 → 11.6 |
| remap → fitplain | model-mem | 6.30 → 6.35 | 6.43 → 6.49 | 2 of 3, 2 of 3 (one tie) | 4.3 → 4.3 |
| fitplain → full | model-mem | 6.37 → 6.43 | 6.48 → 6.51 | 3 of 3, 2 of 3 | 4.2 → 4.3 |
| full → renum | model-mem | 6.41 → 6.36 | 6.51 → 6.46 | 0 of 3 (one tie), 0 of 3 | 6.0 → 4.7 |
| base → renum | model-mem | 6.29 → 6.54 | 6.54 → 6.67 | 3 of 3, 3 of 3 | 4.7 → 7.7 |
| base → remap | model | 2.62 → 2.66 | 3.08 → 3.12 | 3 of 3, 3 of 3 | 7.1 → 6.4 |
| base → renum | model | 2.65 → 2.69 | 3.14 → 3.17 | 3 of 3, 3 of 3 | 7.7 → 7.1 |

`GBCFORMER_STATS=1` text.sh on `model-mem`: 170.8 pairs per pass after `fit` (the base 189.4, −10%), at the same 1,423 units.

**Explaining the number.** On size, the memory checkpoint halves, from 153.4 M to 82.6 M dense parameters (hidden 2,048 to 1,024 at `i` 1,536 to 1,728), with 2,574 more nonzeros (858 reuses, three entries per clearing unit). On speed, the goal was not met. The memory core walks 10% fewer pairs yet runs about 1 to 4% slower than the base; the split A/Bs put about 1% on the placement itself (remap to fitplain, both hidden 2,048) and about 1% on the reuse (fitplain to full), of which renumbering by wire won back about 0.8%. So on this placement the pairs proxy did not predict the clock, though it explained the schedule-on/off pair; the same lesson as the static proxy before it. Not ruled out: which layers hold the busy gate rows (cache footprint per layer), and the load (4 to 12, another play window), which moved one binary's absolute µs by up to 4% between A/Bs. The per-channel range guard is not the cost. A diagnostic build with the old `abs() <= 1` check in its place measured no faster (memory text 6.55 to 6.59 µs, host text 2.62 to 2.63). The host core's checkpoint is byte-identical to the base's, yet the engine measures about 1.5% slower in 3 of 3 pairs; its hot path differs from the base's only by the range lookup (ruled out above) and the per-layer `routed` add, so this is unexplained and inside the 1 to 5% spread earlier sections counted as a tie.

**What limits size next.**
- *Clearing units in the late levels.* Every channel is busy at boundary 5, so the wires written at levels 6 to 9 each need a clearing unit there: the ~190 units between the node width and `i` 1,728, about 5.9 M dense parameters. Re-gathering inputs instead of keeping them live (a layer-L head with layer 0's gather pattern, aliased in the engine to the layer-0 channel) would free about 40 to 60 wires at boundaries 5 to 7 at no `a` cost (layers 2 to 9 hold at most 3 heads against 10), slack for nodes to move earlier and for unused channels late.
- *The host* stays at hidden 2,048: 256 one-hot token channels and 937 live wires against 677 channels.
- *The PPU*: `docs/ppu-in-weights.md` §3a; merged One and Tile land at hidden 2,048 (497 M and 555 M, from 975 M and 1.31 B).

### Phase 3: the timer and the interrupt flags as a checkpoint

`--weights control` (`docs/control-in-weights.md`). Without control in weights, the
machine's timer, IF and IE sit behind a three-arm slot matched inline per access, every
bus read checks the control addresses before syncing, and an empty acknowledgement no
longer touches IF. Scratch copies of `tools/bench/ab.sh`, A = the base build, B = this
branch, A B A B A B, 3 runs each. The final rows are the committed binary; the earlier
rows are intermediate builds (the timer with its own `Signals`, then without).

| Core, build | text.sh A → B µs per pass (runs; median) | fps (median) | overworld.sh A → B µs per pass (runs; median) | fps (median) | Load |
| - | - | - | - | - | - |
| `model`, first build | 2.57, 2.60, 2.56 → 2.58, 2.55, 2.56 (2.57 → 2.56) | 58.4 → 58.4 | 3.10, 3.13, 3.09 → 3.11, 3.07, 3.09 (3.10 → 3.09) | 135.9 → 136.6 | 4.8 → 4.4 |
| `model-mem`, first build | 6.27, 6.27, 6.26 → 6.32, 6.31, 6.34 (6.27 → 6.32) | 29.7 → 29.5 | 6.31, 6.40, 6.41 → 6.36, 6.43, 6.44 (6.40 → 6.43) | 79.0 → 78.6 | 4.4 → 4.4 |
| `model-mem`, second build | 6.32, 6.27, 6.26 → 7.97, 6.34, 6.23 (6.27 → 6.34) | 29.7 → 29.3 | 6.37, 6.37, 6.39 → 6.41, 6.39, 6.38 (6.37 → 6.39) | 79.3 → 79.0 | 4.3 → 4.9 |
| `model`, final | 2.53, 2.53, 2.52 → 2.55, 2.54, 2.55 (2.53 → 2.55) | 59.2 → 58.8 | 3.02, 3.06, 3.02 → 3.05, 3.05, 3.05 (3.02 → 3.05) | 139.4 → 138.2 | 3.9 → 3.2 |
| `model-mem`, final | 6.23, 6.22, 6.23 → 6.24, 6.30, 6.25 (6.23 → 6.25) | 29.9 → 29.8 | 6.27, 6.31, 6.34 → 6.39, 6.32, 6.33 (6.31 → 6.33) | 80.0 → 79.8 | 3.2 → 4.0 |

**Explaining the number.** The plain reading is that the slot costs the existing cores 0.3 to
1% per pass. In the final `model` pair every B text.sh run is 0.01 to 0.03 µs behind
every A run, and overworld.sh is 0.03 µs behind at the median; `model-mem` is behind at
the median in every pair. What the change adds per access is a range check before every
read and a three-arm match where a field access was, and the halted loop pays the match
on every M-cycle. Not profiled.

**With control in weights**, one binary, A against B:

| Stacks | text.sh fps (runs; median) | overworld.sh fps (runs; median) | Load |
| - | - | - | - |
| `cpu` → `cpu,control` | 56.0, 56.7, 56.5 → 55.5, 56.0, 55.9 (56.5 → 55.9) | 132.1, 132.6, 134.6 → 127.8, 129.9, 130.1 (132.6 → 129.9) | 10.1 → 7.4 |
| `cpu,memory,apu` → `cpu,memory,apu,control` | 29.0, 29.3, 28.5 → 28.7, 22.4, 28.4 (29.0 → 28.4) | 77.2, 76.3, 76.2 → 75.2, 73.5, 75.0 (76.3 → 75.0) | 7.4 → 8.1 |

1 to 2%: 157 control passes per frame at 1.96 µs on both workloads, 145 of them raises
of a STAT bit Yellow never clears (it keeps STAT out of IE), less the reference timer's
per-M-cycle ticks. The 22.4 run had other work on the machine. Holding the compiled
device behind a trait object measured text.sh 57.5 to 56.2 and overworld.sh 132.8 to
129.8 in an earlier pair (load 7.0); its own static arm of the slot measured text.sh
better and overworld.sh unchanged, not separated from noise.

### Phase 1: the scanline checkpoint as the machine's renderer

`--weights ppu` (`docs/ppu-in-weights.md` §9). What touches the paths without the PPU in
weights: the engine plans a positional stage only for a checkpoint with position wires
(one more reader bit, still a `u16`), a memory write record checks the video queue, the
machine hands each block's DMA copies to the core before the next block, and the host
PPU builds a `Line` (registers and palettes) per visible line. Scratch copies of
`tools/bench/ab.sh`, A = the base build, B = this work, A B A B A B, 3 runs each.

| Core, build | text.sh A → B µs per pass (runs; median) | fps (median) | overworld.sh A → B µs per pass (runs; median) | fps (median) | Load |
| - | - | - | - | - | - |
| `model`, first build (`u32` reader mask) | 3.64, 2.59, 2.67 → 2.69, 2.70, 2.69 (2.67 → 2.69) | 56.0 → 55.7 | 3.15, 3.23, 4.52 → 3.27, 4.60, 3.19 (3.23 → 3.27) | 130.5 → 128.8 | 19.9 → 11.6 |
| `model-mem`, first build | 6.54, 6.68, 6.46 → 6.93, 6.74, 6.72 (6.54 → 6.74) | 28.5 → 27.6 | 6.60, 6.71, 6.68 → 7.02, 6.82, 6.75 (6.68 → 6.82) | 75.6 → 74.1 | 11.6 → 10.7 |
| `model-mem`, `u16` mask | 6.41, 6.44, 6.42 → 6.57, 6.63, 6.54 (6.42 → 6.57) | 29.0 → 28.3 | 6.44, 6.54, 6.63 → 6.62, 6.75, 6.81 (6.54 → 6.75) | 77.3 → 74.9 | 13.7 → 6.8 |
| `model`, final | 2.63, 2.64, 2.66 → 2.73, 2.65, 2.66 (2.64 → 2.66) | 56.7 → 56.4 | 3.16, 3.18, 3.14 → 3.15, 3.22, 3.21 (3.16 → 3.21) | 133.6 → 131.4 | 7.7 → 6.9 |
| `model-mem`, final | 6.52, 6.51, 6.50 → 6.46, 6.48, 6.46 (6.51 → 6.46) | 28.6 → 28.8 | 6.56, 6.64, 6.56 → 6.56, 6.59, 6.60 (6.56 → 6.59) | 77.0 → 76.7 | 6.9 → 7.2 |
| `model`, final, overworld.sh only, 5 runs | | | 3.16, 3.13, 3.13, 3.10, 3.14 → 3.19, 3.14, 3.17, 3.19, 3.15 (3.13 → 3.17) | 134.7 → 133.1 | 5.4 → 6.7 |

**Explaining the number.** The first build was slower on `model-mem` in 6 of 6 pairs. The
engine's own timer (`GBCFORMER_STATS`) put the whole difference in the pass, at equal
units and changed pairs: 6.19 to 6.35 µs on text.sh. Narrowing the reader mask back to a
`u16` did not move it; what did was the decode loop, which had gained a positional check
and branch on each of the 46 decoded positions (3.5 ns each). A plan without a
positional stage now runs the base's loop unchanged (6.10 to 6.12 µs by the same timer),
and `model-mem` ties: B ahead on every text.sh pair, within 0.05 µs on overworld.sh.
The `model` core on overworld.sh stays behind in 5 of 5 pairs by 0.01 to 0.09 µs
(1.3% at the median); the engine timer puts 0.02 µs of it in the pass (3.25 to 3.27) and
the rest on the host side of a 135 fps workload, where the per-step DMA hand-off and the
per-line `Line` are the new code. Not separated further.

**The full stack**, one binary, `--weights cpu,memory,apu` (A) against
`--weights cpu,memory,apu,ppu` (B), the PPU on its own thread (first build; its PPU
path is the final one):

| Pair | text.sh A → B fps (runs; median) | overworld.sh A → B fps (runs; median) | Load |
| - | - | - | - |
| PPU on the host → PPU checkpoint | 27.7, 21.3, 28.0 → 27.4, 27.5, 27.4 (27.7 → 27.4) | 68.5, 73.9, 75.1 → 40.8, 63.3, 63.0 (73.9 → 63.0) | 10.7 → 8.8 |

Every text.sh run reached `870aa3fd` and every overworld.sh run `822059c1`. On its
thread the checkpoint draws a line in 12 to 28 µs (2 to 4 ms per frame) in these runs,
so text.sh, at 36 ms per frame on the CPU side, hides it. overworld.sh's CPU side takes
13.5 ms per frame, close enough that the PPU thread competes for a core with the GBC CPU
and APU threads on a loaded 10-core machine: −15% at the median, the 40.8 run with
other work running (inferred, not profiled).

### Phase 3, second round: in-place passes and the latched lines

Two changes, one commit each. Instruction counts are from `/usr/bin/time -l`
("instructions retired"), which the load on the machine (8 to 33 from other
builds) does not move; wall-clock pairs are `ab.sh`'s procedure on scratch copies.

**Raises and acknowledgements in place**. A raise or an acknowledgement
runs over an empty span and leaves the M-cycles to the next pass
(`docs/control-in-weights.md` §7). `overworld.sh` with `--weights cpu,control`,
`GBCFORMER_STATS=1`: the control engine's units per pass 483 to 20 and its time per pass
1.96 to 0.41 µs (one run each, load 20 to 30), 157.3 passes per frame either way.

**The control device's lines latched in the machine** (`ControlDevice::lines`, the
machine's `lines` field written after every control call). The first unit's slot had
put a three-arm match behind every `Bus::ie` and `Bus::iflag` (out of line in the
profile: 25 samples each, absent before) and in front of every block's counter and
horizon. Now they are field loads, the latch costs one match per control call, and the
read path's sync check is one range test with the sync itself out of line.

| Binary | `--core ref`, Yellow CGB 600 frames: instructions (cycles) | `--core model`, 40 frames from boot: instructions (cycles) |
| - | - | - |
| before the first unit | 4.83 G (0.85 G) | 36.67 G (6.95 G) |
| the first unit | 5.19 G (1.05 G) | 36.90 G (6.93 G) |
| raises in place | 5.20 G (1.03 G) | |
| this change | 5.00 G (0.87 G) | 36.88 G (6.86 G) |

| Core (A before the first unit → B this change) | text.sh µs per pass (runs; median) | fps (median) | overworld.sh µs per pass (runs; median) | fps (median) | Load |
| - | - | - | - | - | - |
| `model` | 3.12, 4.57, 2.63 → 2.95, 2.60, 2.61 (3.12 → 2.61) | 48.0 → 57.4 | 3.17, 3.14, 3.17 → 3.15, 3.11, 3.13 (3.17 → 3.13) | 133.2 → 134.8 | 15.0 → 9.6 |

**Explaining the number.** The host path recovers 0.20 G of the first unit's 0.36 G
extra instructions per 600 reference frames and 0.18 G of its 0.20 G extra cycles;
the rest is the latch's store and the slot's match per block. On the `model` core
every overworld.sh pair has B ahead by 0.02 to 0.04 µs per pass, so the first unit's
0.3 to 1% is no longer measurable there; text.sh's A runs 1 and 2 overlapped another
build (4.57 µs), so only its third pair is clean, and it is flat. The `model`
core's instruction count is still 0.6% above the binary before the first unit for the same 40 frames while
its cycles are not; the difference sits in the engine's code, which this change does
not touch, and is not explained.

**The rest of the control logic in weights**: the joypad, serial,
OAM DMA and HDMA control and KEY0/KEY1/SVBK join the control checkpoint
(`docs/control-in-weights.md` §13). 762 wires, depth 8, 41.3 M dense, 31,748 nonzeros;
314.6 control passes a frame on text.sh and overworld.sh, 302.0 of them in place.

`ab.sh`'s procedure on scratch copies of `text.sh` and `overworld.sh`, A B A B A B, 3
runs a side, load 6.5 to 7.5. With control in weights A is the first round's binary
(timer and flags in weights, the rest of the control logic on the host)
and B this round's (all of the control logic but the PPU timing in
weights). For the cores without control in weights A is the binary before
the first round.

| Stack or core (A → B) | text.sh fps (runs; median) | overworld.sh fps (runs; median) |
| - | - | - |
| `cpu,control` | 57.4, 56.8, 57.0 → 56.5, 55.9, 54.5 (57.0 → 55.9) | 128.4, 127.1, 128.9 → 127.3, 126.9, 127.9 (128.4 → 127.3) |
| `cpu,memory,apu,control`, `--lockstep all` | 27.9, 28.3, 28.1 → 27.8, 27.7, 28.1 (28.1 → 27.8) | 73.8, 73.3, 73.4 → 73.3, 73.5, 73.3 (73.4 → 73.3) |
| `model` (no control in weights) | 56.7, 56.3, 56.6 → 56.4, 56.6, 56.4 (56.6 → 56.4) | 134.6, 133.6, 135.1 → 133.5, 131.0, 133.9 (134.6 → 133.5) |
| `model-mem` | 28.8, 28.6, 28.7 → 28.6, 28.7, 29.0 (28.7 → 28.7) | 77.4, 77.4, 76.7 → 77.3, 77.3, 77.6 (77.4 → 77.3) |

**Explaining the number.** With control in weights the round costs 0.1 to 0.35 ms a
frame (text.sh 57.0 to 55.9 fps at `cpu,control`, overworld.sh 128.4 to 127.3). That is
the price of moving the joypad, serial, OAM DMA, HDMA and the CGB registers off the
host. It buys a checkpoint 2.4 times larger and 157 more passes a frame, almost all of
them in place: 144 HBlanks, 3 P1 writes, 8 P1 reads, 1 button change and 1 FF46 write.
The control engine's own stats on text.sh (`--weights cpu,control`,
`GBCFORMER_STATS=1`, load 12) account for it: 181 units and 1.12 µs per pass over
314.6 passes, 0.35 ms a frame. Most of those units (77 and 69 in layers 1 and 2) are
the event decodes re-evaluating when consecutive in-place passes switch kind, a raise
then an HBlank; that is the next lever.
The full stack measures flat on overworld.sh (73.4 to 73.3) and 1% slower on text.sh.
On the cores without control in weights, B is behind on overworld.sh's `model` core in
every pair (0.7 to 2%) and level elsewhere. The load-independent count says the host
path is not the cause: 600 reference frames take 4.97 G instructions and 0.847 G
cycles against the binary before the first round's 4.83 G and 0.845 G, and 40 `model` frames take 36.67 G
against 36.68 G. So the `model` gap on overworld.sh is not explained by work done. It
is within what a load of 7 moves a 3-run median (the A runs alone span 133.6 to 135.1).
The first round's 0.3 to 1% per pass on `model` and `model-mem` is no longer
measurable.

### Phase 2: the APU checkpoint smaller and cheaper per pass

`docs/apu-in-weights.md` §12. APU only (`--weights apu`,
`GBCFORMER_APU_THREAD=0`), so the run's work is the APU's passes; instructions retired per
pass as §12 defines them.

| Step | text.sh units | text.sh instructions per pass | Dense | Nonzeros |
| - | - | - | - | - |
| the base build | 3,896 | 152,481 | 145.9 M | 223,371 |
| schedule and fit | 3,896 | 151,486 | 102.7 M | 223,362 |
| noise later wraps summed (fit retargeted) | 3,711 | 146,856 | 102.7 M | 218,857 |
| capacitor low block cap + S | 3,552 | 142,690 | 102.7 M | 217,050 |
| n·P as w0·P + m·P | 3,196 | 131,183 | 102.7 M | 196,292 |
| state packed, two fit targets | 3,196 | 131,543 | 95.0 M | 194,810 |
| 4-bit timer blocks, capacitor cap + S, LFSR gate | 2,802 | 120,652 | 96.0 M | 190,307 |
| wave one-hot position, phases over the count | not taken | 118,756 | 96.2 M | 176,855 |
| channel 2's later wraps gated | 2,639 | 116,105 | 98.7 M | 177,082 |

overworld.sh: 4,397 to 2,836 units, 167,906 to 122,637 instructions per pass. Every hash
and lockstep held (§12).

**Timing.** `ab.sh` APU-only interleaved, 3 runs each, the base (A) against the final tree
(B), 1-minute load 26 to 35 (other builds and play windows): text.sh 12.92 to
10.71 µs per pass (runs 12.81, 13.24, 12.92 to 10.71, 10.24, 11.24), overworld.sh 17.18 to
16.21 (34.39, 17.18, 16.76 to 16.21, 20.26, 15.39). At that load the clock moves a run by
up to 2×, so the instruction count is the result; the text.sh medians agree with it
(−17% against −24%).

**Explaining the number.** Instructions fall less than units (−24% against −32% on
text.sh) because a pass also pays per changed (wire, layer) pair (151 against 163) and a
fixed part per decoded position, which item 5 trims (65 to 60 positions) but the router
pays back (89 to 105 routing units: more output bytes change when fields straddle
bytes). Item 1 is size only (units unchanged; the pairs and −0.65% instructions are the
fit's). The extra two layers cost about 3.6 M dense against the schedule's −43 M.

**One checkpoint for the whole machine** (`--one-model`, `docs/one-model.md` §6): the
four circuits concatenated into one 1,422.1 M checkpoint, each part a stream with a
pruned plan. Instructions retired by the 600 model frames (full run minus a run that
stops at the hand-off), same binary, `--weights all` against `--weights all
--one-model`, load 8 to 22 so wall fps is not used:

| Workload | four checkpoints | one checkpoint |
| - | - | - |
| text.sh | 352.6 G | 373.1 G (+5.8%) |
| overworld.sh | 208.8 G | 218.9 G (+4.8%) |
| text.sh, `--weights cpu,memory` | 254.9 G | 270.6 G (+6.2%) |
| text.sh, `--weights apu` | 78.0 G | 81.1 G (+4.0%) |

Frame hashes `870aa3fd` and `822059c1` on both sides. The streams evaluate the same
units per pass as the four checkpoints; the gap is bookkeeping over the 425-position
sequence. Without `--one-model` the pass path gains two branches on the absent view and the `Engine`
holding its model through an `Arc`, and the engine build is faster (the head table and
the memory-head check no longer walk dense rows).

The cores without `--one-model`, text.sh's 600 model frames against the base build
(built from a `git archive` by the profiling round): `model-mem` 261.7 to 256.8 G,
`model` 143.1 to 142.5 G instructions. No slower.

### The GBC CPU circuit with memory, fan-out round: reads off the wires that change every step

Five commits: CB fields through prefix gates, the JR offset from OP1
alone, step wrapper buffers, translations and decode,
and write sources isolated. Only `src/sm83/` and `examples/readers.rs`
changed; the host checkpoint stays `1f4c3519` after every commit.

**The lever.** `examples/readers` (since removed) predicted the census from the IR: each wire's
recorded changes per pass (`src/sm83/changes-memory.txt`) times the steps that read
it, summed by (source family, reader family) with `--by`, one wire's readers by
name, and the live wires at a boundary with `--live N` (it also lists the dead
wires). On the base it predicted 1,433 updates per pass against 1,416 measured. The
change table is re-recorded after every circuit change, since `Circuit::fit` scores
placements against it.

**Units and pairs per fast pass** (`GBCFORMER_UNITS=1`, deterministic):

| Commit | text.sh units | overworld.sh units | text pairs | overworld pairs | Nonzeros |
| - | - | - | - | - | - |
| the base build | 1,416 | 1,414 | 170.8 | 166.1 | 150,084 |
| CB gates | 1,319 | 1,314 | 170.3 | 166.8 | 149,469 |
| JR offset | 1,295 | 1,291 | 166.4 | 163.0 | 148,765 |
| translations | 1,057 | 1,066 | 157.7 | 154.6 | 144,647 |
| writes | 1,012 | 1,017 | 151.8 | 147.7 | 143,997 |

Every commit is 82.6 M dense parameters at hidden 1,024, intermediate 1,728, 10
layers, depth 9.

**What changed and why.**
1. *CB fields.* The 48 CB field nodes read OP1's bits, which change on most steps,
   so every OP1 change updated up to 32 of them whether or not the prefix was set.
   They read eight gates `OP1.k ∧ prefix`, and the prefix comes from OP0's bits a
   level earlier so the fields keep their level (with the prefix as the 0xCB
   one-hot the depth reached 10).
2. *JR offset.* In memory mode the second fetch is pc + 1 − hb, so under the HALT
   bug OP1 already holds OP0's byte; the offset's OP0 steps went, and later the
   taken test became one wire over the nibble one-hots, so each offset bit is one
   step. `tests/halt_bug_mem` runs SST cases of JR, CALL, RST and loads of every
   length with the bug set against the reference (the SST never set it).
3. *Rare read bus.* Its regions are six wires built from the sources and their
   opcode tests at the bus's own level; with no source selected all are 0 and a
   "no region" case lands in VOID, so the gate every step read is gone (RRARE 69
   to 29).
4. *FF page.* LDH A,(n) and LD A,(C) read through three single-step nodes over
   the byte and the DMA bit; LDH (n),A writes through six (HRAM with its record,
   VBK and SVBK on CGB, VOID, the FF block under DMA).
5. *Fetch path.* The fetch adders split again at bit 3. The fetch translations'
   HRAM and IE tests read one wire "bits 0..6 all set" computed from PC, and F1
   and F2's regions are interval tests over PC's bits 7..15 and the adder's carry
   into bit 7, one level above PC, so a step reads a region wire where it read
   the adder's high bits. The next-pc adder's low nibble splits at bit 2.
6. *Request cases.* They test sums of nibble one-hots instead of 16 field
   one-hots, at the same level.
7. *Writes.* The W0 bus address reads the rare write bus and the isolated FF00+n
   byte, so a fetched byte no longer reaches it; ADD HL,rr reads the pair isolated
   to its opcodes one level after the pair (overworld `hl+pair` 44 to 23).
8. *Dead wires.* Write translations' VOID flags and ROM region wires, the selects'
   unused flag, unused field one-hots.
9. *Wrapper.* `MemCheckpointCore` reuses its position lists, byte buffer and kept
   positions instead of allocating them per pass.

**Instructions retired per model pass**, whole-process `/usr/bin/time -l` minus a
run of the same binary with one model frame (identical warmup, compile and
insertion), divided by the model passes; all threads, machine included:

| Binary | text.sh | overworld.sh |
| - | - | - |
| base | 81,641 | 82,194 |
| writes isolated | 68,171 (−16.5%) | 69,161 (−15.9%) |

**Timing**, a scratch copy of `ab.sh`, A = the base, B = the final commit, A B five times per
workload. Load 10.4 before and 19.6 after, with this work's full gate running on
other cores and other jobs throughout, so the A runs spread 6.99 to 13.09 µs
on text.sh.

| Workload | A fps (median) | B fps (median) | A µs per pass | B µs per pass | Pairs B faster |
| - | - | - | - | - | - |
| text.sh | 26.6, 18.7, 17.8, 22.9, 14.2 (18.7) | 30.9, 26.1, 29.3, 29.4, 27.2 (29.3) | 6.99, 9.97, 10.44, 8.13, 13.09 (9.97) | 6.01, 7.12, 6.34, 6.33, 6.85 (6.34) | 5 of 5 |
| overworld.sh | 59.2, 63.9, 73.6, 73.7, 70.1 (70.1) | 77.1, 68.3, 87.5, 66.3, 77.1 (77.1) | 8.54, 7.91, 6.87, 6.86, 7.21 (7.21) | 6.55, 7.40, 5.78, 7.62, 6.56 (6.56) | 4 of 5 |

Every run held `870aa3fd` and `822059c1`.

**Explaining the number.** The wall-clock medians carry the load: text.sh's A runs
alone span a factor of 1.9, so the 36% text.sh median gap is not the measurement. The
instruction counts are: 13.5 k fewer per pass on text.sh, about 33 instructions per
removed gate update (404 fewer) as the full-stack profile measured for the gate loop,
plus the pairs' bookkeeping and the wrapper. All of it is on the GBC CPU's thread, which
retires about 3.8 instructions per cycle at 2.95 GHz (the full-stack profile), so the
13.5 k are about 1.2 µs per pass; that is an inference from the counters, not a
clean-load measurement, and it is a third of the 3.6 µs the loaded medians show.

**Gates on the final commit.** All SST (circuit, checkpoint, `build_mem`, the memory
engine), engine equal to the dense pass, `memory_dense`, `dma_records`,
`halt_bug_mem`, e01, interrupt fuzz, the library tests; `lockstep-mem` DMG 8
`6872b003` and CGB 300 `cfd31f3d`; Crystal `4e6a7509`, Zelda DX `1d00d871`, Mario
DX `7dec3947` 1,500 frames `lockstep-mem`; `--weights cpu,memory,apu,ppu,control
--lockstep all` CGB 300 `cfd31f3d`; `lockstep` DMG 8 and CGB 300; text.sh and
overworld.sh hashes. `GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem`: 27
dense checks, 0 dense fallbacks, frame `6872b003`.

**Tried and reverted.**
- *The next pc from the fetch adders* (pc + 1 and pc + 2 by length, pc + 3 and the
  JR target from pc isolated to their opcodes): pc + 1 and pc + 2 then lived to
  level 9 and boundary 5 reached 977 live wires, over the 868 hidden 1,024 allows.
- *The JR target as its own adder* over pc and OP1 isolated to JR: text.sh 1,295 but
  `jr` 63 and 90 updates per pass, the cost of the isolated inputs flipping from 0 to
  pc on every JR; the committed offset trim gave the same units at no size cost.
- *Muxed regions on the rare write bus*: predicted updates rose (the a16 source's
  region steps read OP2 on every step).
- *The rare write bus holding BC off its opcodes*: `WRARE0` 23 to 41, BC is a loop
  counter.
- *The pair holding HL off its opcodes*: `hl+pair` up on overworld.sh.
- *Three pair selections* (INC, DEC, ADD HL each its own mux): 998 and 985 units, but
  961 live wires at boundary 5 and hidden 2,048.

**What limits it next.**
- *Boundary 5 is full.* The circuit sits at 868 of 868 live wires there; the 256
  opcode one-hots are live across it (their readers are the final muxes at levels 6
  to 9). Every isolation that lands at level 5 adds its width there, which is what
  pushed the reverted variants to hidden 2,048. The host-served late bytes (OP0, OP1,
  OP2, D1 inputs) are live there too though they never change in five games; folding
  each into its head byte costs a node per bit and a level for OP0 and OP1, so only
  OP2 is cheap (8 channels).
- *The decode.* `op0` (the 32 nibble one-hots, 62 updates) and `op*` (the 256 ANDs,
  62) are the largest families left and near the minimum for a two-level decode of a
  byte that changes every step. The one-hot fan-out into selectors is small (`op*` to
  readers about 20). A decoded class carried by the fetch head itself would remove
  `op0` and move the one-hots a level earlier; it needs compiler and engine support
  (below).
- *The next-pc adder* (`pc+` 73 to 77): pc's low bits, the length bits and the JR
  offset. Its alternatives measured above either break the channel budget or pay
  for isolation transitions.
- *Evidence outputs.* OP0', OP1', OP2', RV0 and RV1 keep the head bytes live to the
  routing layer and are decoded every pass; their bits change on most steps, a share
  of `route` 42 and of the 46-position decode. Whether the audit needs them every pass
  is an ABI decision.
- *Decoded opcode class (a spec for the compiler and engine).* The 32 lane channels
  are reserved in the layout and zero on every memory-family value token, so a
  per-byte feature vector (the two nibble one-hots) can sit in those channels of the
  value-token embedding rows at no channel cost. A second head on F0's query reads
  them through V; `circuit::Head` needs a kind whose `out` is 32 wires, `memory_head`
  its V rows over the lane channels, `KvCache` that head's value from the stored byte
  (as it already forms snapshot keys), and `Engine::memory_reads` a byte to lanes
  table. A table of class records read by its own head was considered and dropped:
  each extra index read costs about 900 instructions, some 27 gate updates.

### Engine hot loops by instructions retired

The GBC CPU thread is instruction-bound (IPC 3.6 to 3.8), so these rows are judged by
instructions retired, read per thread with `thread_selfcounts` over the model phase
(a scratch counter, not committed). The counts repeat to 0.03% between runs and do not
depend on load. Every row keeps the frame hashes (`870aa3fd` text, `822059c1`
overworld) and the gates. "full" is `--weights cpu,memory,apu,ppu,control`. text.sh
runs 3,224,368 GBC CPU steps and overworld.sh 1,187,411. The base is the build before these changes.

| Change | full text.sh: CPU / APU / PPU thread, G instructions (k per CPU step) | full overworld.sh: CPU / APU / PPU | `model` text / overworld | `model-mem` text |
| - | - | - | - | - |
| base | 262.4 / 75.6 / 24.6 (81.4) | 98.5 / 82.9 / 34.1 | 144.5 / 58.1 | 263.2 |
| gate-update loop | 200.0 / 49.7 / 21.1 (62.0) | 75.2 / 53.6 / 26.8 | 116.2 / 46.5 | 201.3 |
| decode by route bit | 194.8 / 49.0 / 21.1 (60.4) | 73.7 / 52.9 / 26.9 | 114.4 / 46.1 | 196.4 |
| layer-0 plan check on moved positions | 189.4 / 47.6 / 20.9 (58.7) | 71.5 / 51.5 / 26.5 | 111.7 / 45.0 | 191.0 |
| per-layer drain | 183.1 / 46.7 / 20.2 (56.8) | 69.4 / 50.5 / 26.0 | 107.6 / 43.3 | 184.4 |
| gate-update loop, round 2 | 165.7 / 44.1 / 19.0 (51.4) | 63.0 / 47.7 / 24.0 | 96.1 / 39.0 | 167.3 |
| decode pattern by OR | 164.8 / 43.9 / 18.8 (51.1) | 62.9 / 47.6 / 23.8 | 95.6 / 38.8 | 166.4 |
| positional drain by slot table | 164.6 / 43.8 / 18.9 (51.0) | 62.6 / 47.5 / 23.6 | 95.4 / 38.8 | 166.2 |
| memory address bits kept incrementally | 162.4 / 43.9 / 16.5 (50.4) | 61.8 / 47.6 / 22.3 | 95.9 / 39.0 | 163.7 |

**Gate-update loop.** `Marks::layer` runs once per layer per pass in every engine. One
gate update cost about 33 instructions with its send (profile of the base). It now
costs about 20: 13 for a plain unit that sends nothing, 15 for a fused pair, and about
9 more for a send. Each step was measured on the full text.sh GBC CPU thread:
- `#[inline(never)]` on the loop, −5.7%. Inlined into `fill_fast`, it reloaded its
  pointers from the stack on every update.
- The census closure chosen outside the loop, −4.6%. With `GBCFORMER_UNITS` unset, the
  loop still tested `fam`'s `Option` on every update.
- Plain units and fused pairs in separate loops, −6.8%, and −15% on the APU thread. A
  pair flag sorts last among a channel's entries, so `Cols` keeps one split point per
  channel, and each loop has a constant cap: `max(g, 0)` or `clamp(g, 0, 1)`.
- Unchecked indexing of the gates, output channels, down weights and marks, −6.8%.
  `IntLayer::new` is now the one constructor. It asserts that every gate entry's unit
  is in the layer and every output channel is below the layer's channel count.
  `Marks::layer` asserts the lengths of `g` and the marks once per call. Together the
  two checks prove every index the loop reads.
- `Cols::split` inlined and the mark arrays' base pointers hoisted, −2.4%.

**Decode by route bit.** The profile of the base guessed that all 46 output positions
decode on every GBC CPU pass. A counter (`positions decoded` in the `GBCFORMER_STATS`
line) says otherwise, with 12.0 of 46 on the GBC CPU, 11.3 of 65 on the APU, 13.4 of 160 on the
PPU and 0.1 of 33 on control. The other positions hit the decoded-token cache. Each
decode still cost about 540 instructions, because it chased `route_index[out[u]]`,
`cap[u]` and `c[u]` per own unit and built the pattern with a branch per route. Each
own unit now carries its route bit, pair flag and down weight (`Own`), and the
pattern is built without branches over all 12 routes. A route outside {0, 1}, or a
pattern past the head table, still refuses the pass. GBC CPU thread −2.6%.

**Layer-0 plan check on moved positions.** `layer0_changes` looked up its attention
plan by walking all 66 positions' (k, q) ids against each recent plan, on every pass.
The engine now remembers which recent plan matched the state's tokens
(`recent_hit`). The next pass checks only its moved positions (4.5 a pass) against that
plan, since the others still hold the tokens it matched, and walks the list as before
on a miss. A match is unique, because a plan is added only when none matches, so the
two lookups agree. The hit is cleared when the diff starts and set only when it
succeeds. GBC CPU thread −2.8%.

**Per-layer drain.** After each layer, `advance` moved every touched channel's
change into its wire, checked the wire's range, and pushed the change to the buckets
of the layers that read it. That loop is now `Marks::drain_wires`, out of line (−1.1%).
The buckets are a fixed array of 32, one per reader bit, so a push needs no bounds
check (−1.4%). All five arrays the loop indexes are cut to the marks' length, so one
check per channel replaces five (−1.5%).

**Gate-update loop, round 2.** Two costs per change and per send were left:
- Each change's split into plain and pair entries read `start[ch]`, `start[ch + 1]`
  and the split point with three bounds checks and two slice checks. `Cols` now keeps
  one `[start, split, end]` span per index, and `split` slices without checks, since
  `from_pairs`, the only constructor, makes every span ascending and inside the
  entries (−4.6%).
- A first send to a channel pushed it onto `touched`, a `Vec::push` that reloaded the
  length and capacity from memory. A channel is pushed only when its mark goes up, and
  no mark comes down inside the loop, so one `reserve(channels())` per call covers
  every push. The loop writes through a pointer and a local count, then sets the length
  (−5.2%).

The profile with stage marks (every 32nd pass) now puts the GBC CPU pass's circuit-layer
updates at 22.9 k instructions for 1,381 updates: 16.6 per update, sends and per-change
work included, against about 33 at the base. A plain update that sends nothing is 12
instructions, a fused pair 15.

**Decode pattern by OR.** The branchless pattern compiled to about 70 vector lane
moves. The pattern is now `(v & 1) << i` per route, and an OR of all twelve totals
guards {0, 1}, since it exceeds 1 exactly when some total is outside {0, 1}. The own-unit
loop also indexes the route array directly instead of `bit % ROUTE_CAP`. GBC CPU thread
−0.5%.

**Positional drain by slot table.** The PPU's positional stage drained each touched
channel by dividing its offset by the block stride to find the position and the
channel within it. A table built with the stage (`Positional::slot`) now gives each
channel's wire index or flattened route index, and the drain is out of line
(`Positional::drain`). The PPU thread is −0.8% on overworld.sh. On text.sh it moves
18.83 to 18.86 G, inside the PPU thread's run-to-run spread of 0.2% (its waits spin a
little), since text lines change few positional wires.

**Memory address bits kept incrementally.** A memory head whose query channels
changed refolded all 22 address bits from the wires before reading its byte: about
900 instructions a read on the GBC CPU and 650 on the PPU, which reads 115 times a line on
text.sh. Every change to a channel a query reads already passes through the head
bucket once, so the engine now keeps each bit's form value (`State::head_bits`) and
adds `w · dv` for each change it drains from that bucket (`MemPlan::bit_terms`, the
bits by channel). A dirty head assembles its address from the 22 values, and any value
outside {0, 1} still refuses the pass. A fresh state starts the values at the bits'
constants, with the wires at 0. The argument needs every query change drained before
the head's layer reads. A head's output can't feed another head's query in the same
layer, because the query reads the occupant before the layer (`seg.read(ch, 2·li)`)
and the output writes the one after it (`seg.write(ch, 2·li)`). The PPU thread is
−12% on text.sh and −6% on overworld.sh, and the GBC CPU's thread −1.3%. The `model` core,
which has no memory heads, retires 0.5% more (95.4 to 95.9 G on text.sh). It never
enters the memory code, so the difference is a guess. The guess is that the larger fresh-state setup
changed what the compiler generated for `advance`. Not measured.

**The eight changes rebased** (one model, the APU round, the GBC CPU
circuit round). The same counters, the base against the rebased branch:

| Stack and workload | CPU thread, G instructions (k per CPU step) | APU thread | PPU thread |
| - | - | - | - |
| full, text.sh | 216.9 (67.3) → 138.7 (43.0), −36% | 56.2 → 34.7, −38% | 23.9 → 16.4, −31% |
| full, overworld.sh | 81.9 (69.0) → 53.0 (44.7), −35% | 59.0 → 36.2, −39% | 33.2 → 22.3, −33% |
| `--weights all --one-model`, text.sh | 234.8 (72.8) → 156.0 (48.4), −34% | 61.1 → 38.9, −36% | 24.8 → 17.3, −30% |
| `--weights all --one-model`, overworld.sh | 89.0 (74.9) → 59.9 (50.5), −33% | 63.8 → 40.2, −37% | 34.2 → 23.2, −32% |
| `model-mem`, text.sh | 217.7 (67.5) → 140.1 (43.5), −36% | | |
| `model`, text.sh / overworld.sh | 143.9 / 57.9 → 95.4 / 38.6, −34% / −33% | | |

Frame hashes `870aa3fd` and `822059c1` on every run.

Wall clock, the base (A) against the rebased branch (B), A B A B A B, both binaries with
the scratch counters, `GBCFORMER_STATS=1`, load 6.4 to 9.0:

| Stack | text.sh fps (runs; median), µs per pass | overworld.sh fps (runs; median), µs per pass |
| - | - | - |
| `model` | 55.3, 55.0, 56.1 → 71.9, 71.4, 72.3 (55.3 → 71.9), 2.71 → 2.08 | 131.5, 130.0, 132.3 → 167.7, 167.8, 166.2 (131.5 → 167.7), 3.21 → 2.51 |
| `model-mem` | 33.1, 32.9, 33.3 → 44.2, 44.5, 44.2 (33.1 → 44.2), 5.63 → 4.21 | 88.1, 88.8, 88.9 → 114.5, 114.8, 115.9 (88.8 → 114.8), 5.69 → 4.40 |
| full | 32.3, 32.4, 32.3 → 43.2, 43.5, 43.7 (32.3 → 43.5), 5.75 → 4.28 | 71.1, 72.2, 71.6 → 93.4, 93.2, 94.0 (71.6 → 93.4), 7.06 → 5.41 |

Every B run is ahead of every A run. The instruction ratio and the wall-clock ratio
agree within a few percent (full text.sh: 0.64 of the instructions, 0.74 of the µs per
pass); the rest is the GBC CPU's thread's waits and host work, which these changes leave
alone. The gates on the rebased tree: `tools/gate.sh` all ok, `--weights all
--lockstep all --one-model` CGB 300 `cfd31f3d` with 41,750 lines equal, the four games
300 frames with all five in weights in lockstep, `instr_timing` (cpu, control),
`dmg_sound` 01 to 04 (apu, control) and acid2 (cpu, memory, ppu) in lockstep, and the
five checkpoint digests unchanged.

### Battle workload (`tools/bench/battle.sh`)

Scripted presses take a power-on Yellow through the intro, Red's room, Oak in the grass and
Oak's lab (Pikachu received, no nickname) to the rival battle. Frames 0 to 14,879 run on the
reference interpreter; the 600 model frames (14,880 to 15,479) are one battle round: the FIGHT
menu, Thundershock and its animation, Eevee's HP bar draining, Tackle, a critical hit and
Pikachu's HP bar. 2,200,525 GBC CPU steps (3,668 per frame; text.sh runs 5,348), 24% of them
two-pass. Frame hash `14d7468e…`, the same on `--core ref` (twice), `model`, `lockstep`,
`model-mem`, `--weights all --lockstep all` and `--weights all --one-model --lockstep all`;
both all-in-weights lockstep runs report 86,400 of 86,400 PPU lines equal to the reference.

Wall clock at the base build, battle.sh and text.sh alternating, 3 runs each,
`GBCFORMER_STATS=1`, load 5.9 to 11.3 (other jobs shared the machine). Every run has the same steps and hash, so the spread (the 29.3 and 76.2
outliers included) is presumably that contention:

| Stack | battle.sh fps (runs; median) | text.sh fps (runs; median) |
| - | - | - |
| `model` | 105.2, 108.3, 76.2 (105.2) | 62.2, 74.6, 71.5 (71.5) |
| `model-mem` | 61.2, 69.8, 66.8 (66.8) | 45.8, 45.9, 43.9 (45.8) |
| `--weights all` | 61.1, 29.3, 59.9 (59.9) | 43.2, 37.1, 42.4 (42.4) |
| `--weights all --one-model` | 47.2, 56.3, 43.3 (47.2) | 38.5, 39.4, 32.1 (38.5) |

Instructions retired per thread over the model phase (the scratch `thread_selfcounts`
counter, not committed), k per GBC CPU step in parentheses:

| Stack and workload | CPU thread, G instructions | APU thread | PPU thread |
| - | - | - | - |
| full, battle.sh | 95.2 (43.3) | 35.9 | 23.7 |
| full, text.sh | 138.7 (43.2) | 34.7 | 16.6 |
| `--one-model`, battle.sh | 107.4 (48.8) | 40.2 | 24.5 |
| `--one-model`, text.sh | 156.1 (48.6) | 38.9 | 17.5 |
| `model`, battle.sh / text.sh | 67.7 (30.8) / 95.4 (29.7) | | |

Battles are lighter than text boxes for the GBC CPU's thread. The cost per step is the same, but
the game halts more (5.0M halted M-cycles against 2.2M), so there are 31% fewer steps per
frame. The PPU thread is the one that grows, at 274k instructions per line against 193k
(+42%), from the battle's sprite and tile churn. The APU thread is flat (it is driven by
sample instants, not by the game). The GBC CPU thread still limits the full stack: 9.06 s of
CPU time for 600 frames, 15.1 ms a frame, against 3.9 s for the APU and 2.3 s for the PPU.

### Memory: the image's undo log only after a mark

`Image::write` logged every overwritten byte for `revert`, and only `mark` (called by
`load_bus`, the SingleStepTests harness) ever cleared the log. A game never calls it,
so the log grew by 8 bytes per memory write for the whole run. The log now starts at
the first `mark` or `revert`; a running game keeps none. One 20,000-frame
`--core model-mem` run of Yellow from boot (the title screen, 63 fps, load about 7),
RSS sampled every 20 s:

| Binary | RSS at 20 s | RSS at 300 s | Maximum RSS | Frame hash |
| - | - | - | - | - |
| before | 645.6 MB | 744.1 MB, rising 6.4 MB per 20 s | 769.4 MB | `61e43d23` |
| after | 643.1 MB | 644.5 MB, flat from 60 s on | 660.0 MB | `61e43d23` |

The SingleStepTests suites that revert (`sm83_sst`, `checkpoint_sst`, `memory_dense`)
pass unchanged.

### Phase 3, third round: the PPU's timing in weights

One commit (`docs/control-in-weights.md` §14). A is the round before, B this commit;
interleaved A B A B A B on text.sh and overworld.sh, medians of 3, load 4.5 to 14.9
(other jobs and a play window). Every hash held in every run.

| Stack or core | text.sh fps A → B | overworld.sh fps A → B |
| - | - | - |
| `model` | 57.4 → 57.3 | 135.3 → 134.1 (3 of 3 pairs slower) |
| `model-mem` | 28.1 → 28.3 | 75.8 → 75.0 |
| `cpu,control` | 54.5 → 49.7 | 123.6 → 103.7 |
| `cpu,memory,apu,ppu,control` | 27.3 → 26.0 | 60.8 → 56.9 |

With control in weights a full span now runs at every visible line's mode-3 and mode-0
start and VBlank's line starts, 308 a frame, at 1,747 units and about 6 µs each (181
units and 1.0 µs in the second round, when 302 of 315 passes ran in place): about 1.5
ms a frame. On `model` the reference path ticks the timing in `Control::advance` and
ends about 145 more blocks a frame at render points; overworld.sh is 0.9% slower, which
that fits but no profile confirms.

### Phase 1 and 3: the host's last PPU work

Three commits: the palette lookup and window counter in the PPU checkpoint, BCPS/OCPS
in control, mode 0 a horizon only where needed, the worker pipelined, then two that
add three folds and the control circuit's level schedule. A is the
binary before this whole round (`docs/control-in-weights.md` §14, §15;
`docs/ppu-in-weights.md` §10); B is the final commit. Interleaved A B A B A B, medians of 3,
load 4.5 to 19 (other jobs and a play window); several runs on each side ran under
bursts of 20+ and are listed. Every hash held in every run.

| Stack or core | text.sh fps A → B (runs) | overworld.sh fps A → B (runs) |
| - | - | - |
| `model` | 55.6 → 54.6 (A 56.6, 55.5, 55.6; B 56.4, 54.6, 48.3) | 132.0 → 130.4 (A 128.9, 132.8, 132.0; B 131.4, 130.4, 128.5) |
| `model`, quieter (load 5 to 9), A B A B | 58.4, 57.6 → 57.6, 57.7 | 137.8, 136.1 → 136.0, 136.8 |
| `model-mem` | 27.9 → 27.8 | 50.4 → 74.8 (A runs at 39.4 and 50.4 under load; clean pair 75.2 → 75.5) |
| `cpu,control` | 22.7 → 49.3 (A runs 2 and 3 under load; clean pair 53.5 → 49.4) | 128.2 → 107.3 |
| `cpu,memory,apu,ppu,control` | 26.5 → 24.6 | 57.9 → 63.8 |

**Instructions retired, `model` (load-independent).** A whole text.sh run (2,900
reference frames, 600 model frames) is 185.27 G to 188.32 G (+1.6%), overworld.sh (7,200
reference frames, 600 model frames) 134.45 G to 141.32 G (+5.1%), two runs each side,
within 0.01%. Overworld's share is mostly reference frames, so the extra work is the
reference path: `Control::advance` ticks the timing and computes its horizon after every
control call, and blocks now end at each render point. The model frames' wall clock
ties (the quieter row). That is the cost the `model` and `model-mem` cores carry; it is
not in the model's passes.

**The full stack.** Overworld.sh is faster (57.9 to 63.8; 64.4 in a run at load 4.5):
the GBC CPU's thread no longer waits for the PPU at VBlank (4.05 ms a frame in the
earlier profile). Text.sh is about 2 fps slower, because its GBC CPU thread carries the
control checkpoint's 308 full spans a frame at 6.3 µs (the game keeps STAT's mode-0
source on, so mode 0 stays a horizon), about 1.7 ms a frame more than before the round.

**Checkpoint hashes.** On the base's five-checkpoint `examples/digest`, the CPU checkpoint
stays `1f4c3519`. The PPU checkpoint (`1fd8c447` on the base) and the control checkpoint
(`a5942d44`) move because their circuits change: the PPU's palette heads, lookup, two
output bytes and window outputs, and the folds; control's PPU timing state, palette
index registers and schedule. Their tokens are held to the references by the lockstep
gates above, not by a digest.

### Integration: the PPU's timing and palettes on the one model

The PPU timing, palette and window round (the eight commits above from "the image's
undo log" on) rebased onto the one-model, APU and CPU-circuit rounds. No code in
`one_model.rs`, `stack.rs` or the engine's streams needed to change: `Circuit::concat`
takes the new PPU and control circuits as built, the PPU stream's `View` follows the
PPU's own slot list (8 inputs, 160 colour outputs, `WLN'` and `WAR'`), and
`PpuCheckpoint` drives two passes a CGB line through the stream as it does through its
own checkpoint. `tests/one_model.rs` now expects 170 PPU stream slots and 409 PPU
memory heads (the PPU's own 410 less the copy head, which the GBC CPU stream holds).

**Digests** (`examples/digest`): cpu `1f4c3519`, cpu-memory `8c463dde` and apu
`ad161b52` unchanged; control `a5942d44` to `d8e1f97f` (44.1 M) and ppu `1fd8c447` to
`ffd69b51` (552.9 M), from the round's circuit changes listed above.

**One checkpoint** (`examples/one_model -- --build`): hidden 4,096, intermediate 4,864,
14 layers, 52 heads, vocabulary 34,480, 1,501.0 M dense, 1,039,106 nonzero (1,422.1 M
before: the PPU grew by 38.1 M and its depth to 14 layers). Four checkpoints summed
778.3 M. Overlay projections: 1,125.8 M, 750.5 M, 684.7 M with shared slots.

**`--weights all --lockstep all --one-model`, 300 frames**, every line equal colour for
colour and no dense fallback in any stream: Yellow `cfd31f3d` (41,750 lines in 83,500
PPU passes, 61,498 control passes), Crystal `ccfe7ad3`, Zelda DX `7b126eb8`, Mario DX
`3e1f4d27`. Also under `--one-model`: `instr_timing` Passed, `dmg_sound` 01-04
`99793856`, `881f3fe7`, `e494d943`, `8e776b75`, and both acid2 images pixel-exact.
Lockstep wall clock at load 8 to 9 with `tools/gate.sh` running beside it, not a speed
measurement: on the tree rebased onto the engine hot-loop round, Yellow 75.6 fps, Crystal 51.4, Zelda DX 35.2, Mario DX 84.9 (one run each). Digests and sizes are the same on both trees.

### Text boxes toward the frame budget: engine instructions per GBC CPU step

The text.sh profile of the base (GBC CPU thread 98% busy) puts the 60 fps budget at about
30.9 k instructions per GBC CPU step. These rows are the GBC CPU's thread's instructions retired
over the model phase (`thread_selfcounts`, a scratch counter, not committed), divided
by the GBC CPU steps: text.sh 3,208,968 and overworld.sh 1,179,826. "four" is `--weights
all`, "one" is `--weights all --one-model`. Every row keeps text.sh `870aa3fd`; overworld.sh (`822059c1`) was counted at the base and the last row.

| Change | text.sh four / one, k per CPU step | overworld.sh four / one |
| - | - | - |
| base | 42.86 / 48.26 | 44.59 / 50.40 |
| view: place only changed tokens, diff only rewritten positions | 42.83 / 46.02 | |
| view: tokens compared four at a time; head table in the stream's tokens | 42.83 / 45.26 | |
| view: keep reads the stream's own tokens | 42.83 / 44.01 | |
| drain: range and readers side by side, one unsigned range compare | 42.61 / 43.93 | |
| gate loop: a pair's send test as one sign test | 41.97 / 43.28 | |
| decode: own units without bounds checks | 41.72 / 42.89 | 43.59 / 45.13 |

The four-checkpoint value on the second row was not counted on its own; the third row's count covers both. Two attempts were reverted: the routing stamp through an unchecked owner table (+0.06 / +0.12 k), and the gate loop unrolled by two (+0.49 / +0.51 k on the GBC CPU's thread, though −0.9 G on the APU thread, whose lists are longer).

**View: place only changed tokens.** A stream's `fill` mapped every stream token into
the checkpoint's sequence, mapped the positions again, and layer 0 then compared all
owned positions and copied all 425 tokens into the state. `View::place` now keeps the
stream tokens it last placed and maps only those that changed; each checkpoint
position it rewrites joins `dirty` once. A position whose token differs from the last
fast pass's was rewritten since, so layer 0 diffs only `dirty`, and a continuing pass
writes only its moved positions into the state. The position map is reused while the
stream passes the same positions. `keep` scans the view's owned positions for the
address tokens, not the whole sequence; the other positions hold byte 0 or a mask.

**View: four at a time, and the stream's own head table.** Most of a stream's 66 tokens
repeat between places, so `place` compares them in groups of four and maps a token
only inside a group that differs. A view engine's head table now holds the stream's
tokens (`NO_TOKEN` for a checkpoint token the stream lacks, which decodes to
`usize::MAX` as `View::back` gives it), so a fast pass needs no map back; the dense
fallback still maps its tokens.

**View: keep reads the stream's own tokens.** `keep` placed the step's tokens into the
checkpoint's sequence (its decoded outputs differ from the last fill's masks), scanned
for the address tokens there, and the next fill's place undid the outputs again. The
records need only the stream's tokens, because a view maps every family token one to one, so
`Engine::records` reads the address and value tokens in the stream's own vocabulary.
The checkpoint's sequence is placed only when the dense oracle runs.

**Drain: one table per wire.** `Marks::drain_wires` read each changed wire's range and
reader bits from two arrays, and with the wires, the marks and the buckets it ran out
of registers and reloaded base pointers from the stack per channel. The plan now keeps
one `Wire` per virtual channel (`Template::wires`, built in `Engine::build` once the
readers are final): the range as its low end and width, so the check is one unsigned
compare, and the reader bits beside it. `Wire::new` asserts a nonempty range. A plan with an empty one, which would have refused every pass, now stops at build (none does).

**Gate loop: the pair test as one sign test.** A scratch count over text.sh (every
engine, all threads) found that 89% of gate updates send nothing (plain units 11%
send, fused pairs 11%), and that every input change is ±1. A pair sends exactly when
its gate crosses from ≤ 0 to ≥ 1 or back, which is a differing sign bit of `g − 1`
before and after: two subtractions, an exclusive or and a branch, where the clamps and
their difference took six. The difference is computed only on the send path. The APU
thread also drops 34.4 to 32.6 G instructions on text.sh.

**Decode: own units without bounds checks.** A decoded position walks its token's own
routing units (91 a GBC CPU pass), and each step checked the unit against the gates and the
route bit against the route array. `Engine::build` now asserts once that every own unit
is below the routing layer's length and every route bit below `ROUTE_CAP`, and
`decode` asserts that the gates have the layer's length, so the walk indexes unchecked.

**Wall clock and gates for the round.** fps, the base (A) against the round (B),
interleaved A B, three runs each, median, load 10 to 22 (other builds): text.sh
four 30.9 to 32.8, one 32.2 to 37.5; overworld.sh four 87.8 to 77.6, one 79.0 to 85.0. At
that load single runs swing by a third, so the instruction counts above are the
evidence. On the round's final commit: `tools/gate.sh` all ok, `--weights all --lockstep all` and
`--weights all --lockstep all --one-model` CGB 300 `cfd31f3d` with 41,750 lines,
243,951 APU passes and 49,964 control passes equal, the dense oracle
(`GBCFORMER_ORACLE=500`, 3 frames, 7 checks) with four checkpoints, and the five
digests unchanged.

### Memory reads: the scatter skip and record keys, without the nibble decode

The engine half of the nibble-lane round, ported alone onto the
one-model the base. Every digest is unchanged: cpu `1f4c3519`, cpu-memory `8c463dde`
(82.6 M), apu `ad161b52`, control `d8e1f97f`, ppu `ffd69b51`.

- **Scatter skip.** A memory head whose byte equals the one it last read skips its
  scatter (`State::head_byte`). Only `memory_reads` writes a head's output channels, so
  they still hold that byte.
- **Record keys.** `KvCache::push_record` keeps a ROM or write record as (kind,
  address, position, byte) in the tail and forms its key and value by the formula when
  read. A kept write therefore costs no RoPE. The dense oracle (`DenseMemory::keep`)
  still pushes the key and value the model computed, through `push`.

**Instructions retired, GBC CPU thread** (`thread_selfcounts` over the model phase,
scratch counter; k per GBC CPU step; `--weights all`, and `--weights all --one-model`):

| Binary | text.sh | overworld.sh | battle.sh | one model text | one model overworld | one model battle |
| - | - | - | - | - | - | - |
| the base build | 44.80 | 49.84 | 45.69 | 50.82 | 56.38 | 51.87 |
| + scatter skip | 44.44 | 49.55 | 45.40 | 50.47 | 56.08 | 51.54 |
| + record keys | 43.95 | 49.12 | 44.92 | 49.97 | 55.65 | 51.09 |

PPU thread on text.sh 43.47 to 41.88 G (the scatter skip, as on the branch); APU
unchanged. Every run keeps `870aa3fd`, `822059c1` and `14d7468e`.

**Timing**, interleaved A/B, the base against both cuts, median of 3 (emulated fps). The
load average was 8 to 18 over the runs, with other gates beside them:

| Workload | base | both cuts |
| - | - | - |
| text.sh `--weights all` | 40.5 | 41.3 |
| text.sh `--one-model` | 36.0 | 37.5 |
| overworld.sh `--weights all` | 74.9 | 67.7 |
| overworld.sh `--one-model` | 73.9 | 78.1 |

The overworld `--weights all` B runs landed at load 18 and 15 against A's 9 and 18,
spreading from 49.7 to 87.8 fps. They do not separate the binaries. The counters above
are the evidence.

**Gates** on the final commit: `tools/gate.sh` 27 of 27 ok. `lockstep-mem` DMG 8 `6872b003`,
CGB 300 `cfd31f3d`, and 1,500 frames of Crystal `4e6a7509`, Zelda DX `1d00d871` and
Mario DX `7dec3947`. `GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem` ran 27
dense checks with 0 dense fallbacks and reached `6872b003`. `--weights all --lockstep
all` CGB 300 reached `cfd31f3d` with 41,750 lines equal, the same under `--one-model`.
battle.sh under `--weights all --lockstep all` reached `14d7468e` with 86,400 lines
equal.

**The nibble decode, tried again on top and not kept.** The fetch head writing the
opcode's nibble one-hots (that round's circuit, `head_decoded`, value lanes, host
nibble hots) costs channel room at boundaries 5 and 6, where the 256 opcode one-hots
live. From `schedule`'s placement, `fit` peaks at 939 live wires against 868 at every
cap, with 6× the iterations too. The creation-level start fits at its own width
(intermediate 2,048, 92.4 M). The same start with the caps at the scheduled width
(1,600 or above) fits at intermediate 1,792: 84.6 M, 2.0 M over the base. Widths of 1,472
and below fit from neither start. Measured on text.sh `--weights all`, over the two
cuts above (43.95 k): 43.76 k at 1,792, 43.65 k at 2,048. That is 0.19 k (0.4 %) for
2.4 % more dense parameters, and too small to show in fps. The branch's analysis holds:
F0 now scatters 40 channels, and its one-hots change about three times a pass in every
layer that reads them. The 256 opcode ANDs still change every step.

### Control: fewer full spans, and a horizon that moves less

`docs/control-in-weights.md` §16. A is the base build, B this round (the circuit of
§16). The control checkpoint runs inline on the GBC CPU's thread. Its full spans went
from 308.4 to 156.0 a frame on all three benches, and its units per pass from about
1,750 to about 360.

**Instructions retired on the GBC CPU's thread over the 600 model frames** (load-independent,
`thread_selfcounts` from a scratch hook, one run each; every hash held):

| Bench | `--weights all` A → B per step | `--one-model` A → B per step |
| - | - | - |
| text.sh | 44.64k → 42.64k (−2.00k) | 50.69k → 48.42k (−2.27k) |
| overworld.sh | 49.75k → 43.97k (−5.78k) | 56.32k → 49.99k (−6.32k) |
| battle.sh | 45.56k → 42.59k (−2.97k) | 51.76k → 48.46k (−3.30k) |

Overworld.sh has the fewest steps a frame (1,966, against 5,348 on text.sh), so the same
saving per frame is the most per step. Per frame the saving is about 10.7 M instructions
on text.sh, 11.4 M on overworld.sh and 10.9 M on battle.sh: about 1 ms a frame at the
profile's 0.52 ms a frame per 1k instructions a step on text.sh.

**Control passes, `--weights control`** (`GBCFORMER_STATS=1`; µs per pass under load 10 to 18):

| Bench | Passes per frame A → B | Units per pass A → B | µs per pass A → B | Control ms a frame A → B |
| - | - | - | - | - |
| text.sh | 338.9 → 186.5 | 1,749 → 358 | 5.66 → 2.25 | 1.92 → 0.42 |
| overworld.sh | 324.4 → 172.0 | 1,832 → 360 | 4.27 → 1.27 | 1.39 → 0.22 |
| battle.sh | 333.9 → 181.5 | 1,769 → 367 | 3.24 → 1.38 | 1.08 → 0.25 |

**Checkpoint hashes** (`examples/digest`): cpu `1f4c3519`, cpu-memory `8c463dde`, apu
`ad161b52` and ppu `ffd69b51` unchanged; control `d8e1f97f` to `c3187a58`, because its
circuit changed: the `IF.1 held` node and the gated STAT cases, the merged reload form,
the longest span in place of the bit-12 candidate, and the horizon less one with one
comparison per pair. The lockstep gates hold its tokens to `machine::control`: under
`--weights all --lockstep all`, battle.sh `14d7468e` with 108,880 control passes equal,
and Yellow CGB 300 under `--one-model` `cfd31f3d` with 59,050 control passes equal
(61,498 before) and all 41,750 lines equal.

**Emulated fps**, interleaved A B A B A B, medians of 3, load 7 to 16 (other jobs,
a play window, `tools/gate.sh` for the first benches); every run held its hash:

| Bench | `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - | - |
| text.sh | 41.7 → 43.9 (A 41.9, 41.7, 41.7; B 43.9, 44.2, 42.7) | 35.8 → 39.5 (A 25.9, 35.8, 37.9; B 39.5, 39.0, 39.5) |
| overworld.sh | 91.4 → 93.2 (A 84.3, 91.4, 92.5; B 92.6, 95.6, 93.2) | 78.6 → 81.2 (A 86.1, 45.5, 78.6; B 88.3, 62.2, 81.2) |
| battle.sh | 61.9 → 65.3 (A 61.9, 62.3, 22.4; B 65.1, 65.3, 65.6) | 53.1 → 56.7 (A 54.2, 53.1, 53.0; B 56.7, 55.4, 56.7) |

The runs at 22.4, 25.9, 45.5 and 62.2 fps ran under load bursts. The instruction
counts above are the evidence; these fps show the same direction on every bench.

### PPU palette reads: held to the host, and what they cost

A GBC CPU read of BGP, OBP0, OBP1, BCPD or OCPD with the PPU in weights drains the PPU
thread's queue (up to 8 lines and the records before them) and waits for the byte. The
worker now reports that wait. Yellow CGB 300 frames, `--weights all` without lockstep,
three runs at load 13 to 16: 17 reads, 2.21, 2.38 and 2.15 ms waited (median 2.21 ms,
about 130 µs a read) against 2.73 to 2.81 s of run, under 0.1%. Crystal reads 3 bytes in
300 frames, Link's Awakening DX and Super Mario Bros. Deluxe none, battle.sh 35 in its
600 measured frames. No game polls the palettes, so the drain stays as it is. The read
is the memory index's byte, held to the host's under `--lockstep ppu` and sampled by
the dense oracle at the memory's cadence (`docs/ppu-in-weights.md` §10.1); a reader
record through the dense pass would cost 3 to 5 s a read. Digests unchanged (cpu
`1f4c3519`, ppu `ffd69b51`).

### Engine caches replaced by producer-pushed state

Two of the engine's memo checks, the decode's per-token stamps and the memory heads'
same-byte skip, are gone. In their place the producers of a change mark what it
reaches. The gate loop's send tests are one sign test each. The rows are the GBC CPU
thread's instructions retired over the model phase (`thread_selfcounts`, a scratch
counter, not committed), k per GBC CPU step: text.sh 3,208,968 steps, overworld.sh
1,179,826, battle.sh 2,200,525. "four" is `--weights all`, "one" is `--weights all
--one-model`. Every run kept `870aa3fd`, `822059c1` and `14d7468e`.

| Change | text four / one | overworld four / one | battle four / one |
| - | - | - | - |
| base | 40.31 / 41.90 | 41.73 / 43.66 | 40.34 / 42.10 |
| decode: stale positions pushed by routing updates | 38.91 / 40.51 | 40.51 / 42.45 | 39.00 / 40.77 |
| gate loop: gates kept as g − 1 | 36.59 / 38.18 | 38.09 / 40.02 | 36.69 / 38.44 |
| memory heads: read on an address move or a write hit | 36.23 / 37.82 | 37.74 / 39.66 | 36.34 / 38.09 |

The round takes 4.07 k (10.1%) off text.sh with four checkpoints and 4.08 k off it with
one model. The APU thread on text.sh goes from 32.38 to 28.25 G, and the PPU thread from
41.58 to 42.09 G (+1.2%). On overworld.sh the PPU thread goes from 59.74 to 58.44 G, and
on battle.sh from 57.17 to 56.32 G.

**Decode: stale positions.** A position's decoded token depends on its token, the
blank-row route totals, its positional route totals and the gates of the routing units
its token reaches. Each routing update used to stamp every owner token of its unit with
the pass number, walking a `Cols` list, at about 67 instructions an update. Each decode
then compared three stamps. `Template::owners` now gives each routing unit the home
positions of the tokens it feeds: tokens with one nonzero channel and one home position,
which are the mask tokens. There are at most two a unit, so each unit has a `[u16; 2]`
entry, padded with the spare index `seq`. The update sets those two `State::stale` flags
without a branch. A change to the route totals marks every position, and a change to a
positional block's route totals marks that block's position. A position decodes again
when it is stale or holds another token. After a decode, its flag takes
`Template::untracked[t]`, so a token that no update can mark decodes on every pass, as a
stamp of `u64::MAX` made it do before. Such tokens have several channels, or any output
position as home. A fresh state starts every position stale. A unit fed by more than two
positions refuses the plan at build, and none is.

**Gate loop: g − 1.** The asm of `Marks::layer` spent four instructions on each send
test: `bic, bic, subs, b.eq` for a plain unit and `sub, sub, eor, tbz` for a pair. The
loop now keeps each gate as g − 1 (`IntLayer::start`). A pair sends when g crosses from
≤ 0 to ≥ 1 or back, which is exactly when the sign bits of the old and new values
differ, so its test is `eor; tbz`. A plain unit sends when g > 0 before or after the
update, which is exactly when the AND of the old and new values has a clear sign bit,
so its test is `and; tbnz`. The plain test needs every update to move g. The gate
entries hold only nonzero weights, and every change list a layer reads carries only
nonzero changes, because each producer skips zeros (the drains, the layer-0 diff, the
memory scatter). Gates stay below 2^23 in magnitude, so g − 1 cannot overflow. The gate
entries now hold the bare unit index: the `PAIR` flag still sorts pairs last when
`IntLayer::new` builds the spans, then it is cleared, which removes the pair loop's
mask. The decode reads `route_g + 1`. The APU thread, with longer lists, drops 11.1% on
text.sh (31.81 to 28.27 G).

**Memory heads: address moves and write hits.** A head read again when its query
channels changed or when the image's generation moved, which any write anywhere does.
It then skipped the scatter when the byte equalled the one it last read. Now
`Engine::write_record`, the only place this engine writes its image, marks the heads
whose `head_addr` is the written address (`State::head_hit`). It advances
`State::seen_gen` only if the state had seen every change before this write. A dirty
head whose recomputed address equals `head_addr` reads only if it was hit. If the
image's generation differs from `seen_gen`, something else changed the image (a new
image, a fill, a revert), and every head reads. A fresh state starts with `seen_gen` at
`u64::MAX`. Only the scatter writes a head's output channels. They start at 0, as
`head_out` does, so `head_out` holds them packed, and the scatter walks only the set bits
of the old byte XOR the new one, adding +1 or −1 to each. The PPU thread, whose heads
read about 115 times a line, retires 1.2% more on text.sh. A write record now compares
every head's address, and the PPU engine follows the CPU's VRAM writes. The GBC CPU thread,
which is the frame budget's limit, retires 0.36 k less a step.

**Emulated fps**, base (A) against this round (B), interleaved A B, medians of 3, load
5 to 9:

| Bench | `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - | - |
| text.sh | 45.8 → 49.9 (A 45.8, 45.0, 45.8; B 50.0, 49.2, 49.9) | 43.2 → 46.8 (A 43.5, 42.4, 43.2; B 46.8, 47.3, 46.2) |
| overworld.sh | 95.9 → 101.5 (A 97.1, 95.9, 95.5; B 101.5, 102.7, 93.9) | 91.0 → 95.8 (A 90.0, 91.0, 92.0; B 96.2, 95.5, 95.8) |
| battle.sh | 62.6 → 72.6 (A 53.4, 62.6, 66.9; B 72.6, 69.6, 73.3) | 62.8 → 67.8 (A 63.0, 62.6, 62.8; B 67.8, 66.8, 68.2) |

**Gates** on this round: `tools/gate.sh` 28 of 28 ok; the engine against the dense pass,
`checkpoint_sst`, `sm83_sst`, `memory_dense`, `dma_records`, `ppu_checkpoint` and
`one_model` (also on the first commit alone); battle.sh under `--weights all --lockstep all`
`14d7468e` with 86,400 lines, 495,276 APU passes and 108,880 control passes equal;
`--weights all --lockstep all --one-model` CGB 300 `cfd31f3d` with 41,750 lines equal;
`GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem`, 27 dense checks, 0 dense
fallbacks, `6872b003`. The six checkpoint digests are unchanged: cpu `1f4c3519`,
cpu-memory `8c463dde`, apu `ad161b52`, control `c3187a58`, ppu `ffd69b51`, one model
`2f3f2bc6`.

**Not done in this round.** The layer-0 `recent` / `recent_hit` attention-plan list
stays. Its four entries exist because the first and second passes alternate position
sets and late-input tokens. Without a bound, a table keyed by (positions, (k, q) ids)
could grow with every distinct id vector, and nothing yet proves the ids per slot take
few values. The engine has no per-pass hash maps: `vector_ids` is read once per new
(slot, token) pair, `interned_ids` only when a plan misses, and the `HashMap` in
`records.rs` only in `load_bus` (SingleStepTests).

### The GBC CPU circuit with memory: fewer gate updates per pass on text boxes

Five commits: CB results selected once, SP adders split,
the immediate isolated for the fields, write translations that test
the ROM area by region wires, and the change table re-recorded. Only
`src/sm83/` changed. The host CPU checkpoint stays `1f4c3519` after every commit.
`cpu-memory` moves from `8c463dde` to `1a48135d` and `one-model` from `2f3f2bc6` to
`72c8d09f` (both base values measured on the base),
because the memory circuit has new nodes and a new layout (below).

**Units per fast pass** (`--core model-mem`, `GBCFORMER_STATS=1`, deterministic) and
**GBC CPU-thread instructions per GBC CPU step** (`--weights all`, `thread_selfcounts`, scratch
counter, single runs; repeated runs of one binary agree within 0.07 k):

| Commit | text.sh units | text.sh k per step | Dense | Nonzeros |
| - | - | - | - | - |
| the base build | 1,012 | 36.44 | 82.6 M | 143,997 |
| CB results once | 952 | 35.83 | 70.8 M | 132,579 |
| SP adders split | 922 | 35.76 | 70.8 M | 131,639 |
| immediate isolated | 901 | 35.22, 35.29 | 72.8 M | 131,655 |
| write regions | 896 | 34.90, 34.91 | 70.8 M | 131,656 |
| change table | 896 | 34.92 | 70.8 M | 131,656 |

Every commit is hidden 1,024, 10 layers. Intermediate 1,728 to 1,344 (1,408 at
the immediate isolated). Boundary 5 stays at 868 live wires of 868.

The base against the final commit, all three workloads (units on `model-mem`; k instructions per
GBC CPU step on the GBC CPU's thread, four checkpoints / one model):

| Workload | Units | k per step, four | k per step, one model |
| - | - | - | - |
| text.sh | 1,012 → 896 | 36.51 → 34.92 (−4.4%) | 38.02 → 36.46 (−4.1%) |
| overworld.sh | 1,017 → 884 | 37.98 → 36.20 (−4.7%) | 39.91 → 38.08 (−4.6%) |
| battle.sh | 998 → 870 | 36.55 → 34.88 (−4.6%) | 38.31 → 36.58 (−4.5%) |

Changed-channel pairs per pass fall with the units: text.sh 151.8 to 144.9, overworld.sh
147.7 to 140.0, battle.sh 149.6 to 142.4.

**What changed and why.**
1. *CB results selected once.* With the CB decode by field, every register field and
   W0's value took 24 CB cases (rotate, res and set by `y`), each gated by a `cbxy`
   wire and the register's `cbz` wire. The operand `x` is the same wires for every
   register, so a change of `x` reached nine fields (`x → A` … `x → W0`, 6.8 updates
   each), and a `cbz` change reached 23 steps per bit of its field. Now one `cbres`
   mux selects the CB value by its operation and each field takes it as one case gated
   by `cbw.r`. A' 38 to 22 and W0 38 to 22 updates per pass, and with fewer field steps
   the intermediate falls from 1,728 to 1,344 (−11.8 M dense parameters).
2. *SP adders split.* `sp−1`, `sp−2`, `sp+1` and `sp+2` were one-level 16-bit adders,
   so a change in SP's low bits (every push, pop, call and return) reached all sixteen
   staircases of all four. They take the fetch adders' shape: bits 0..2 and the carries
   out of bits 2 and 6 at level 1, bits 3..6 and 7..15 at level 2. WSP1, WSP2 and RP1
   keep their levels. `SPL.1` 16.2 to 4.2 predicted updates per pass (130 to 34 steps).
3. *The immediate isolated for the fields.* C', E', L', B', D', H', SP' and PC' read
   the 16-bit immediate for LD rr,d16, JP and CALL. OP1 and OP2 change on most steps
   (rates 0.47 to 0.55), so each changed bit reached five field steps. They now read
   `imm16.ld`, the immediate isolated to those 14 opcodes, placed at level 7 so it adds
   nothing at boundary 5, where OP1 and OP2 are live anyway. −21 units and −0.55 k
   instructions per step: the field rows the updates left are long.
4. *Write translations test the ROM area by region wires.* A write into 0000-7FFF lands
   on a banking register or VOID, and those cases tested `¬a15` and `a14` directly, so
   on a wired translation (WHL, WSP1, WSP2, WRARE0) a change of the address's top bits
   reached every register case (`rare0.15` was read by 19 steps per physical bit). The
   cases now test the `rom0` and `romn` region wires, which wired write translations
   now build too, and read only a13, a12 and a8 directly. −5 units, −0.31 k
   instructions per step, and the intermediate back to 1,344.

**Tried and reverted.** *The ALU operand isolated to the ALU opcodes* (`alu.src`, so
LD r,r' and LD r,d8 no longer reach `add` and `sub`, 39.6 predicted updates): 898
units (−24), but the placement reached depth 10, 903 live wires at boundary 5, hidden
2,048, 11 layers and 130.9 M dense parameters, for −0.04 k instructions per step.

**Explaining the number.** The four changes took 116 units off text.sh and 1.52 k
instructions off the GBC CPU step at four checkpoints, about 13 per unit. The per-commit
ratio varies widely. The SP adders' −0.07 k is within the run-to-run spread, so that
commit is kept on its 30 units and its 940 fewer nonzeros, not on a measured
instruction cut. The immediate took about 26 instructions per unit; the guess is that
the updates it removed sat in the long rows of the late field muxes. The write regions
took 0.38 k for 5 units, about 75 per unit, five times the round's average. That is
not explained. The guess is the narrower intermediate (1,408 back to 1,344) and a
different placement, and the changed-pair line per commit was not taken. It is kept
under the stop rule's size clause; with the immediate it is −26 units and −0.85 k at
the same 70.8 M as the SP-adders commit. The census prediction from
the IR (`examples/readers`) matched the measured units within 2 once the change table
was re-recorded (898 predicted, 896 measured).

**Gates on the final commit.** `tools/gate.sh` 28 of 28 ok, its `cargo test --release` the final
head's full suite. `cargo test --release` also ran on the CB-results tree before the
change table was folded into that commit: every suite ok. `lockstep-mem` DMG 8
`6872b003`, CGB 300 `cfd31f3d`, Crystal `4e6a7509`, Zelda DX `1d00d871` and Mario DX
`7dec3947` at 1,500 frames. battle.sh under `--weights all --lockstep all` reached
`14d7468e`. text.sh, overworld.sh and battle.sh kept `870aa3fd`, `822059c1` and
`14d7468e` in every count above. Digests: cpu `1f4c3519`, cpu-memory `1a48135d`
(70.8 M), apu `ad161b52`, control `c3187a58`, ppu `ffd69b51`, one-model `72c8d09f`.

**Timing**, interleaved A B, three runs each, median emulated fps, A = the base,
B = the final commit, both built with the scratch instruction-counter hooks (not
committed). The 1-minute load was 16.0 at the start and 7.0 at the end, with other
jobs and this work's `tools/gate.sh` running beside it:

| Workload | Four checkpoints | One model |
| - | - | - |
| text.sh | 47.5 → 49.2 | 45.9 → 47.4 |
| overworld.sh | 80.4 → 96.7 | 96.2 → 97.0 |
| battle.sh | 70.2 → 71.8 | 69.1 → 71.4 |

The overworld four-checkpoint A runs spread from 64.5 to 97.7 fps, so that row does
not separate the binaries. The instruction counts above are the evidence; text.sh is
still about 3.9 k instructions per GBC CPU step above the 31 k a 60 fps text box needs.

**What limits it next.** On text.sh at 896 units the largest families are the decode
(`OP0h → op0` 61.5 and `op0 → op*` 58: the nibble one-hots and the 256 ANDs, for a
byte that changes every step), the next-pc adder (`pc+` about 73: PC's low bits, the
length bits and the JR offset), the fetch adders and their translations (`pc+1`, `pc+2`,
F1, F2, about 100 together), the ALU operand into `add` and `sub` (39.6; its
isolation placed at depth 10, and whether that depth is inherent was not diagnosed), and the rare buses (`rare` 24,
`RRARE` 29, `rare0 → WRARE0` about 20). Boundary 5 is full, so any isolation that lands
there forces hidden 2,048. The decode is the largest single block. A fetch head that
writes the opcode's 256 one-hots from the value token (an engine and KV-cache change,
outside `src/sm83`) would remove `op0` and `op*` and put the one-hots a level earlier;
the nibble-lane version of that idea measured 0.4% because the ANDs stayed.

### Engine: the layer-0 plan in the state, stale positions by word, flat buckets

The layer-0 `recent` / `recent_hit` attention-plan list is gone, so the engine keeps no
memo cache. Two data-flow changes cut the GBC CPU's thread's per-pass cost outside the gate
loop. The rows are the GBC CPU's thread's instructions retired over the whole run
(`thread_selfcounts`, a scratch counter, not committed), divided by the model steps, in
k per step. They include the warmup and the compile, so they sit above the last
section's model-phase rows. The base is the build before this round. "four" is `--weights all`, and "one"
is `--weights all --one-model`. Every run kept `870aa3fd`, `822059c1` and `14d7468e`.

| Change | text four / one | overworld four / one | battle four / one |
| - | - | - | - |
| base | 35.69 / 37.32 | 37.64 / 39.59 | 35.84 / 37.62 |
| layer 0: plan in the state, no `recent` | 35.47 / 37.04 | | |
| decode: stale positions walked by word | 35.10 / 36.70 | | |
| buckets: one flat array, a region per reader | 33.90 / 35.53 | 35.81 / 37.77 | 34.05 / 35.81 |

The round takes 1.80 k (5.0%) off text.sh with four checkpoints and 1.79 k with one
model, and about 1.8 k off overworld.sh and battle.sh.

**Layer 0.** Counters over text.sh showed that no moved position changed its key or
query id in any of the five engines. The plan changed only when the GBC CPU's request pass
(46 positions) and result pass (41, a suffix) alternated, about 15,400 times. The state
now keeps the plan with the positions it was built for, as a list and as a flag per
position. The plan holds while every moved position keeps its (k, q) ids and every
decoded position is one whose logits were checked equal. A plan checked for a list holds
for any part of it, so both passes share the plan built for the longer list. A change of
ids or a position outside the list rebuilds the plan. Every caller's position lists
nest (request and result passes, `CheckpointDevice`'s outputs and results), so a run
builds the plan about once. `State::key` and `View::owned` were dead and are removed.

**Decode.** Profiled per pass, the decode loop cost 4.8 k for 12 positions decoded out of
46. Each skipped position paid about 35 instructions, and each decode about 100 plus 21
for each own unit. `State::stale` is now one byte per position, padded to whole words. A
routing update still marks it with one byte store. The decode ANDs eight stale bytes
with eight wanted bytes at once and visits only the set positions in the wanted words.
Layer 0's moved positions mark themselves stale, so the per-position token check is gone.
`State::got` holds each position's last decoded token. A first attempt with u64 bit
words saved 0.6 k in the decode but cost 0.6 k in the routing layer's stamp. The bit set
and clear take more instructions than one byte store.

**Buckets.** `drain_wires` paid about 13 instructions a push into a `Vec` per reader
layer: the bucket address, the length, the capacity check and the store. A pass changes
each channel at most once and routes each change to each reader once. So
`Buckets::new` gives bucket b one cell per channel whose reader mask has bit b, all in
one flat array, and a push is a store plus an end bump. A push past a region's end is
dropped, and the pass is refused as `Refusal::BucketOverflow` and takes the dense pass,
so the bound is checked rather than assumed. No pass in any gate overflowed.

**Tried and reverted.** Comparing the layer-0 tokens eight to a compare saved nothing:
+0.007 k. Keeping each memory head's address as the XOR of its bits' low-bit changes,
plus a count of bits outside {0, 1}, cost 0.04 k more than walking 22 bits on a dirty
head.

**What is left above 1.5 k a pass** (sampled stage profile of this round on its base,
before the rebase onto the memory circuit round, four checkpoints): the gate loop is 14.4 k, the decode 4.4 k (12 decodes, most of it the
own-unit loop at 21 instructions a unit), the memory heads 3.4 k, the drain 3.3 k (about
22 instructions a touched channel), the routing layer 1.6 k (42 updates at about 37) and
layer 0 1.9 k.

**Emulated fps**, the base (A) against the round (B), both with the scratch counter
built in and idle, interleaved A B, medians of 3, load 3.5 to 7.3:

| Bench | `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - | - |
| text.sh | 50.1 → 53.2 (A 50.7, 50.1, 50.1; B 53.4, 53.2, 53.0) | 49.0 → 51.0 (A 49.0, 49.5, 47.2; B 52.0, 51.0, 50.1) |
| overworld.sh | 107.1 → 113.0 (A 107.1, 107.5, 106.7; B 113.0, 112.2, 113.4) | 101.1 → 106.2 (A 100.4, 101.1, 102.1; B 105.4, 107.3, 106.2) |
| battle.sh | 74.7 → 79.0 (A 74.4, 74.7, 75.0; B 78.9, 79.0, 79.5) | 70.1 → 74.0 (A 69.8, 70.1, 70.1; B 74.0, 74.0, 73.6) |

**Gates** on the round's final commit: `tools/gate.sh` 28 of 28 ok, `cargo test --release` included (the
engine against the dense pass, `checkpoint_sst`, `sm83_sst`, `memory_dense`,
`dma_records`, `ppu_checkpoint`, `one_model`); battle.sh under `--weights all --lockstep
all` `14d7468e` with 86,400 lines, 495,276 APU passes and 108,880 control passes equal;
no dense fallback in any engine on text.sh. The digests are the base's: cpu `1f4c3519`,
cpu-memory `1a48135d`, apu `ad161b52`, control `c3187a58`, ppu `ffd69b51`, one model
`72c8d09f`.

### The fetch head delivers the opcode one-hot

Two commits: the one-hot from the value token, and the change table
re-recorded). The GBC CPU circuit with memory no longer decodes the opcode. Before, the fetched
byte went through 32 nibble one-hots and 256 ANDs on every step (`OP0h → op0` 61.5 and
`op0 → op*` 58 gate updates per pass on text.sh). Now the fetch head's value token
carries the byte's one-hot, and the opcode one-hots arrive at the fetch head's layer
with its bits.

**Design.**
- *Class channels.* A memory checkpoint whose circuit has a one-hot head gives every
  memory value token (`RomV`, `SnapV`, `Val`) 256 class channels, `Layout::class(v)`,
  set to 1 at its byte. No layer writes them, so a kept token's class channels at any
  layer are its embedding's.
- *Class heads.* `Circuit::head_hot` adds 256 wires of the head's level to a memory head
  (`Head::hot`). The compiler emits `CLASS_HEADS` (8) more heads on the same layer with
  the record pattern's query and key, each with value dim i reading class channel
  32·block + i and writing the wire of that byte (`llama::class_head`). The softmax
  picks the same token as the byte head, so the eight heads write exactly the byte's
  one-hot.
- *Cache values.* `KvCache` holds one key and value per token as before. A class head's
  value is `class_value` of the stored byte value, and `forward_cached` uses it for the
  heads `MemorySpec::classes` names. `DenseMemory::keep` checks that every class head's
  computed key and value equal that formula on each kept token, so the dense pass still
  proves the cache.
- *Engine.* `MemPlan::check` re-emits each class head from the byte head's recovered
  query and compares every weight, as it does for the byte heads. `memory_reads` then
  moves the one-hot by two channel changes on a new byte (the old byte's channel off,
  the new one's on) from the same index read, so the eight heads cost no extra lookup.
- *Host-served fetches.* A fetch from the I/O page reads VOID's zero byte, which sets
  lane 0, and the host's late OP0 byte is zero unless served. The opcode one-hot is
  `lane[op] + host[op]` for op ≠ 0, where `host[op]` is a nibble one-hot AND over the
  host's bits, and `lane[0] − [host ≠ 0]` for op 0. No game changes the host's byte, so
  those 287 nodes update on no pass.
- *Readers.* The request cases, the rare bus, `jr.taken` and the CB prefix read sums
  of opcode one-hots instead of nibble one-hots, one gate a case instead of two.

**Units and instructions.** Units per fast pass (`--core model-mem`, `GBCFORMER_STATS=1`)
and GBC CPU-thread instructions per GBC CPU step (`thread_selfcounts`, scratch counter, whole
run, as in the last section):

| Workload | Units | k per step, four | k per step, one model |
| - | - | - | - |
| text.sh | 896 → 745 | 33.90 → 32.22 (−1.68) | 35.47 → 33.31 (−2.16) |
| overworld.sh | | 35.82 → 34.13 (−1.69) | 37.70 → 35.62 (−2.08) |
| battle.sh | | 34.04 → 32.39 (−1.66) | 35.76 → 33.67 (−2.09) |

Units by layer on text.sh go from [48, 67, 108, 150, 101, 107, 126, 59, 90, 42] to
[48, 67, 56, 114, 102, 74, 107, 62, 75, 42]. Changed-channel pairs per pass are flat
(144.9 to 145.3), because the one-hot changes two channels a step where the nibbles and ANDs
changed about six, and the case gates read one sum each.

**Size.** The class channels and the host's one-hot need more channels than hidden 1,024
has: boundary 7 holds 1,279 live wires. The memory checkpoint is hidden 2,048,
intermediate 1,472, 10 layers, 10 heads, 149.5 M dense parameters and 146,729 nonzeros
(the base 70.8 M, 131,656). Depth stays 9.

**Explaining the number.** 151 fewer units per pass at about 11 instructions a unit is
1.7 k, the four-checkpoint drop. The one-model drop is larger, 2.1 k; the guess is that
the one model's wider rows made each removed update dearer, which was not measured.

**Timing**, interleaved A B, three runs each, median emulated fps, A = the base build, B =
the final commit, both built with the scratch instruction counter (not committed). The 1-minute load
was 10.4 to 14.8 throughout, with other jobs and this work's dense-oracle run beside
it, so the A runs that fell to 37 and 43 fps are load, not the binary. The counts above are the
evidence.

| Workload | Four checkpoints | One model |
| - | - | - |
| text.sh | 38.6 → 55.6 (A 37.2, 38.6, 45.1; B 55.6, 55.7, 51.1) | 48.0 → 54.1 (A 48.0, 46.8, 48.2; B 54.1, 53.0, 54.7) |
| battle.sh | 72.2 → 78.1 (A 73.2, 43.4, 72.2; B 78.9, 72.6, 78.1) | 71.5 → 78.8 (A 71.5, 71.4, 71.6; B 78.2, 78.8, 78.9) |

On the quiet baseline (55.0 four, 53.9 one on text.sh) the counter drop predicts about
57.9 and 57.4 fps. text.sh is still below 60.

**Gates** on the final commit. `tools/gate.sh` 28 of 28 ok, its `cargo test --release` included
(`sm83_sst` and `checkpoint_sst` on the circuit, the memory circuit and the memory engine,
the engine against the dense pass, `memory_dense`, `dma_records`, `halt_bug_mem`,
`ppu_checkpoint`, `one_model`). `lockstep-mem` DMG 8 `6872b003`, CGB 300 `cfd31f3d`, and
1,500 frames of Crystal `4e6a7509`, Zelda DX `1d00d871` and Mario DX `7dec3947`.
`GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem` ran 27 dense checks with 0 dense
fallbacks and reached `6872b003`. battle.sh under `--weights all --lockstep all` reached
`14d7468e`, the same under `--one-model`. text.sh, overworld.sh and battle.sh kept
`870aa3fd`, `822059c1` and `14d7468e` in every count.

**Digests** (`examples/digest`). cpu `1f4c3519`, apu `ad161b52` and control `c3187a58` are
unchanged. cpu-memory `1a48135d` to `57edc5b5` (149.5 M, the new heads, class channels and
layout). one-model `72c8d09f` to `eb59696c` (1,793.7 M), since it contains the GBC CPU circuit with
memory. ppu `ffd69b51` to `d84c03c5` with no weight changed, because the digest hashes the
`MemorySpec` debug string, which now carries an empty `classes` list.

**What limits it next.** The host fallback's 287 wires are the channel cost that forces
hidden 2,048. Serving a void fetch through the memory itself (the host writes the I/O
byte as a record before the request pass runs again) would remove them and the OP0 to
OP2 late inputs. The largest families left on text.sh are the next-pc adder (`pc+` 73),
the routing layer (42) and the fetch adders and their translations (`pc+1`, `pc+2`,
F1, F2, about 107 together).

### Engine: own route totals kept by the routing update

The decode no longer walks a tracked position's own routing units. Counters over
text.sh (scratch `thread_selfcounts` events, not committed) showed why the stale marks
could not be cut further: 12.05 positions decode per pass, and 99.9% of those decodes
give a different token than the position's last one, so no stale mark is spare. The
blank-row route totals never moved. What each decode paid was its own-unit loop, 7.6
units at about 21 instructions, and every routing update (41.7 a pass) reached exactly
one tracked home and always changed that home's clamped contribution.

So the state now keeps, per position, what its tracked token's own units add to the
blank-row route totals (`State::own_route`, started from `Template::own_start`). A
routing update hands `Marks::layer`'s closure the gate before and after and whether the
unit is a fused pair. Each owner of the unit (`Template::owners`, at most two, the
second skipped unless its down weight is not 0) adds c · (clamp(g' + own) − clamp(g'))
− (clamp(g + own) − clamp(g)) to its home's total, and marks the home stale only when
that is not 0. For a pair the clamp is a sign test of the gate kept as g − 1, so the
pair flag must come from the layer's loop, since computing the cap from the record cost 30
more instructions an update. A token is tracked at its home when it is the only
single-channel token homed there (`Template::tracked`); any other token, or a token away
from its home, still walks its own units. The arithmetic is integer and exact, and the
engine's tokens match the dense pass's in every test.

Rows are the GBC CPU's thread's instructions retired over the whole run divided by the model
steps, k per step, against the base. Every run kept `870aa3fd`, `822059c1` and `14d7468e`.

| Bench | four: base → this | one model: base → this |
| - | - | - |
| text.sh | 32.24 → 31.32 (−0.92) | 33.34 → 32.41 (−0.93) |
| overworld.sh | 34.17 → 33.19 (−0.98) | 35.64 → 34.68 (−0.97) |
| battle.sh | 32.42 → 31.50 (−0.91) | 33.69 → 32.79 (−0.90) |

The stage profile on the old base moved the cost, as planned. The decode loop fell from
about 3.4 k a pass to 1.1 k, and the routing layer rose from 1.6 k to 3.1 k.

**Tried and reverted**, each on text.sh with four checkpoints:

- The gate loop two updates a step with one send branch: +1.54 k. The setup and
  remainder of each span (about 12 instructions, two spans an entry, 104 entries a pass
  at 8.2 updates each) cost more than the 1.5 instructions an update saved.
- The owner's route index flattened to one u16, and the stamp as a `move` closure:
  +0.06 k and +0.18 k.
- Skipping a circuit layer whose bucket is empty: +0.24 k four, −0.04 k one model. Few
  layers are empty, and the change moved codegen more than it saved.
- The routing layer as one span per channel with the pair flag loaded from `cap`:
  +0.55 k.

**What is left** (old base, per pass, four checkpoints): the gate loop about 13.1 k at
854 updates (about 10 instructions an update plus about 20 an entry), the drain 3.4 k
(about 34 a touched channel, 1.29 readers a changed one), the routing layer 3.1 k, the
memory heads 3.2 k, the CPU wrapper after the fill (`sm83/abi.rs`) 1.9 k, layer 0 1.9 k,
the decode loop 1.1 k and the collect of the decoded tokens into a `Vec` 0.7 k.

**Emulated fps**, the base (A) against this change (B), both with the scratch
counter built in, interleaved A B, medians of 3. The machine was loaded (load 8.7 to
15.0) and runs spread by up to 40%, so these rows show no measurable change either way;
the instruction counts above are the evidence. In cycles, the sampled stage profile on
the old base went from 6,900 to 6,485 net cycles per GBC CPU step (−6%). The routing layer's
IPC fell from 3.8 to 2.5, but the decode's cycles fell by more than the routing layer's
rose.

| Bench | `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - | - |
| text.sh | 55.0 → 54.8 (A 55.0, 55.1, 54.6; B 55.1, 54.8, 47.3) | 49.5 → 50.7 (A 39.3, 49.5, 49.6; B 50.7, 51.4, 47.8) |
| overworld.sh | 102.5 → 94.7 (A 85.1, 102.5, 102.7; B 81.5, 94.7, 107.8) | 85.1 → 82.3 (A 90.0, 85.1, 84.9; B 82.3, 78.6, 90.5) |
| battle.sh | 79.3 → 81.5 (A 79.5, 79.3, 58.6; B 81.8, 81.5, 46.2) | |

**Gates** on this change rebased onto the base: `tools/gate.sh` 28 of 28 ok, `cargo test
--release` included (the engine against the dense pass, `checkpoint_sst`, `sm83_sst`,
`memory_dense`, `dma_records`, `ppu_checkpoint`, `one_model`); battle.sh under
`--weights all --lockstep all` `14d7468e` with 86,400 lines, 495,276 APU passes and
108,880 control passes equal. The digests are the base's: cpu `1f4c3519`, cpu-memory
`57edc5b5`, apu `ad161b52`, control `c3187a58`, ppu `d84c03c5`, one model `eb59696c`.

### Host fetches through a HOST record, class channels as wires: hidden 1,024 again

Three commits: the HOST record, the class wires and the
engine plan's token channels (a fix to the class wires that the one model caught). The
memory checkpoint is back at hidden 1,024 with the opcode one-hot kept.

**Design.**
- *The HOST block.* A fetch from an address the host serves (the I/O page, IE, the
  unusable area, the MBC3 clock) translated to VOID and read its zero byte. The fetch
  translations now land such a byte in a 128-byte HOST block (`layout::HOST`,
  `0x22C400`) at the same low seven bits: one region bit (10 for 9) in the fetch
  translation's served cases, nothing more. Data reads still land in VOID, whose zero
  byte the two data heads sum over.
- *The protocol.* The request pass flags a host-served fetch (RQ bits 4 to 6, as before).
  The host then keeps the bus byte as a write record at `HOST | (a & 0x7F)`
  (`MemoryDenoiser::keep_host`, labelled host input, counted in `Stats::host_fetches`)
  and the request pass runs again. The fetch heads always read memory, so the opcode
  one-hot is the class heads' lanes alone. The host fallback (nibble one-hots and ANDs
  over the late OP0 byte, 288 wires) and the OP0 to OP2 late input slots are gone. The
  audit holds every fetched byte to the bus as it did.
- *Class wires.* The 256 class channels were a fixed block beside the wires, live at
  every layer though only the class heads at layer 2 read them. They are now level-0
  wires of the one-hot head (`Head::class`): 1 at a memory value token of that byte, 0
  at every step position. The channel plan clears each one (`−[x ≥ 1]`, exact on 0/1,
  so a kept value token's residual is exact too) after the class heads read it, and
  hands the channel to a later wire.
- *Engine.* The plan's token channels are the channels a token sets, less the class
  channels. Only the memory families set those, the families have no home
  (`Home::Nowhere`) and refuse the fast pass, so a class channel is 0 wherever the fast
  pass runs; the plan refuses a class channel any placed token sets. The first cut
  dropped every channel only the families set, `rd` and `committed` among them; a
  routing unit gated by one then read no token channel, every one-model stream kept it
  and the copy head behind it, and the control stream asked for a memory it does not
  have (the panic in `fill_fast`).

**Size.** Live wires per boundary after the fit at hidden 1,024: [392, 512, 856, 703,
830, 871, 842, 871, 758, 362] against 871 channels (boundaries 5 and 7 full; the class
wires are the 256 at boundaries 0 to 2). At boundary 7 the lanes (`OP0h`, 149 still
read) are the largest family, then the read selection (`sel` 39) and `RP0` (23).

| Checkpoint | the base build | the HOST record | the token channels |
| - | - | - | - |
| cpu-memory | hidden 2,048, intermediate 1,472, 149.5 M, 146,729 nonzeros | hidden 2,048, intermediate 1,280, 134.5 M, 128,410 | hidden 1,024, intermediate 1,280, 67.3 M, 129,667 |
| one-model | 1,793.7 M | | 1,472.7 M |

The circuit is 1,974 wires (1,718 plus the 256 class wires) and 1,198 nodes, depth 9.

**Units and instructions.** Units per fast pass on text.sh (`--core model-mem`,
`GBCFORMER_STATS=1`) 745 to 744, changed-channel pairs 145.3 to 142.6. GBC CPU-thread
instructions per GBC CPU step (`thread_selfcounts`, scratch counter, whole run), the base
against the final commit:

| Workload | k per step, four | k per step, one model |
| - | - | - |
| text.sh | 31.40 → 31.11 (−0.29) | 32.41 → 32.25 (−0.16) |
| overworld.sh | 33.20 → 32.97 (−0.22) | 34.68 → 34.51 (−0.17) |
| battle.sh | 31.58 → 31.31 (−0.27) | 32.79 → 32.55 (−0.24) |

Before the rebase onto engine round 5 the HOST record alone gave the
same counts as both commits (text 32.23 to 31.91 four, 33.33 to 33.06 one model; with the
class wires 31.90 and 33.04), so the drop is the HOST record's and the width is free. No game fetches from a host-served address (0 HOST
records in every run below), so the 288 fallback nodes never updated; the guess for the
1% is the three fewer input slots and the one-hot sums one term shorter in every case
gate that reads them, not measured.

**Timing.** Interleaved A B on text.sh, three runs each, median emulated fps, A = the base build, B = the final commit, both with the scratch counter. Four checkpoints 53.8 to 50.7 (A 53.8, 53.8, 52.1; B 52.8, 50.3, 50.7), one model 52.1 to 51.3 (A 53.1, 51.5, 52.1; B 51.2, 51.3, 51.4); 1-minute load 10.7 to 13.3 with other jobs and this work's gate beside it. Before the rebase the same A B (the base against the head) gave 55.4 to 53.8 four and 54.3 to 54.7 one model at load 8 to 12. B is slower on four of six pairs on the four-checkpoint stack while it retires 0.9% fewer GBC CPU-thread instructions; that disagreement is not explained (the other threads and cache behaviour are unmeasured), so a quiet-machine A B must settle it before this counts as an fps change.

**Gates** on the final commit (rebased onto engine round 5). `tools/gate.sh` 28 of 28 ok, its `cargo test --release` included (`sm83_sst`, `checkpoint_sst` with the memory engine, the engine against the dense pass, `memory_dense`, `dma_records`, `halt_bug_mem`, `one_model`, `ppu_checkpoint`). battle.sh under `--weights all --lockstep all` `14d7468e`, the same under `--one-model`. `GBCFORMER_ORACLE=1000` on 30 DMG frames of `model-mem` ran 27 dense checks and reached `6872b003`. `lockstep-mem` on the class wires (the plan fix after it
changes only which channels count as token channels) DMG 8 `6872b003`, CGB 300 `cfd31f3d`, 1,500 frames of Crystal `4e6a7509`,
Zelda DX `1d00d871` and Mario DX `7dec3947`, every fetched byte and read audited, 0 HOST
records. The memory engine's SingleStepTests run keeps 5,189 HOST records over its
500,000 cases (2,972 host-served steps), 0 failed, 0 fallback passes, and the test now
asserts some case fetched through HOST. text.sh, overworld.sh and battle.sh kept
`870aa3fd`, `822059c1` and `14d7468e` in every count.

**Digests** (`examples/digest`). cpu `1f4c3519`, apu `ad161b52`, control `c3187a58` and
ppu `d84c03c5` are unchanged. cpu-memory `57edc5b5` to `4cdad049` (67.3 M) and one-model
`eb59696c` to `64b3a9c4` (1,472.7 M): the HOST block, the three fewer slots, the class
wires and the hidden 1,024 layout. `src/sm83/changes-memory.txt` is not re-recorded: no
remaining wire changed name or function, and the class wires never change at a step
position.

**What limits it next.** On width, boundaries 5 and 7 hold 871 of 871 channels, so the next
wire added before layer 8 costs hidden 2,048 unless one goes; the lanes' 149 live at
boundary 7 are the largest family there. On speed, the families of the last section are
unchanged (`pc+` 73, routing 42, the fetch adders and translations about 107).

### Engine: cycles per step, the routing walk and the decoded-token buffer

The numbers before "Rebased onto" were measured on the same changes over the earlier base, before the rebase.

This round judges each change by GBC CPU-thread cycles per GBC CPU step as well as instructions
(`thread_selfcounts`, scratch counter, not committed, whole run divided by the model
steps). The reason is that IPC had fallen to about 3.05. Two changes were kept. The base is
the build before this round. "four" is `--weights all`, and "one" is `--weights all --one-model`. Every run
kept `870aa3fd`.

| Change | k instr four / one | k cycles four / one |
| - | - | - |
| base | 31.32 / 32.41 | 10.21 to 10.29 / 10.29 to 10.42 |
| routing layer: one span a channel, owners beside the entries | 31.26 / 32.36 | 9.70 to 9.93 / 9.81 to 9.92 |
| `Denoiser::fill_into` into a kept buffer | 30.53 / 31.58 | 9.61 to 9.79 / 9.74 to 10.00 |

The cycle rows are ranges over two to five interleaved runs at load 6 to 24. With the
machine quiet, base repeats within ±10 k-cycles for four checkpoints and ±50 for one model.

**Where the cycles were.** The xctrace CPU Counters template (CPU Bottlenecks mode) on the
main thread over the model frames measured 45% useful, 18% processing, 15% delivery and
21.5% discarded slots, so mispredicts and the refills after them cost more than memory.
The sampled stage profile put the gate loop at about 52% of the net step cycles (IPC 2.1),
the routing layer at 12% (IPC 2.5) and the drain at 11% (IPC 2.8).

**Routing layer.** Event counters showed 40.8 routing entries and 41.7 updates a pass:
each routing channel feeds about one unit. The two-span walk the circuit layers use
(plain units, then fused pairs) cost two empty-span tests and a loop exit per update.
The stamp loaded each unit's owners by unit, and reloaded the `own_route` and `stale`
bases through the closure's captured references after every store. `Template::route_gate`
is now the routing gate table as `Tagged<RouteTag>`: every entry carries its unit's pair
flag and owners in entry order. `Marks::route_layer` walks one span a channel, with the
bases as locals. A stale mark is now a plain store under `d != 0`, not a load, OR and
store. The routing stage went from 738 to 381 cycles a step (IPC 2.5 to 4.5) at about the
same instructions.

**Decoded tokens.** `Denoiser::fill` returned a new `Vec` every pass. The trait's one
required method is now `fill_into`, which fills a `Vec` the caller keeps. `fill` stays as
a provided wrapper for tests and examples. The memory CPU core, `CheckpointDevice` and the
scanline checkpoint keep their buffers. The change saves 0.74 k instructions a step on
text.sh, more than the 0.45 k the stage profile charged to the collect, so the allocator's
free counted elsewhere.

**Tried and reverted.**

- The routing walk with the entry as one `Entry { u, pair, w, tag }` struct (32 bytes)
  instead of the `(u16, i32, RouteTag)` tuple: +0.1 k instructions, and the cycle gain was
  gone (10.5 to 10.6). This was measured twice and the result is not explained.
- `layer` generic over a `Gate` trait (the circuit's two spans, the routing's one) to share
  the send code: the circuit stage rose from 3,324 to 3,551 cycles and the routing stage
  from 381 to 590. So `route_layer` repeats the send block.
- The circuit layers walked as flagged single spans: +1.8 k instructions and +0.4 k
  cycles. Seven units a channel make the per-update flag test cost more than the second
  loop.
- `CheckpointDevice::decode` filling a kept byte buffer: +0.02 k instructions.

**What is left** (stage profile, cycles per machine step, four checkpoints): the gate loop
about 3,300 at IPC 2.1, the drain about 700, the memory heads about 600 (not this round's
code), and the wrapper after the fill about 200.

**Gates** on the round's last two commits: `tools/gate.sh` 28 of 28 ok, `cargo test
--release` included (the engine against the dense pass, `checkpoint_sst`, `sm83_sst`,
`memory_dense`, `dma_records`, `ppu_checkpoint`, `one_model`). battle.sh under
`--weights all --lockstep all --one-model` reached `14d7468e` on both commits, with 86,400
lines, 495,276 APU passes and 108,880 control passes equal. The digests are the base's: cpu
`1f4c3519`, cpu-memory `57edc5b5`, apu `ad161b52`, control `c3187a58`, ppu `d84c03c5` and
one model `eb59696c`. No weight changed.

**Emulated fps** on text.sh, the base (A) against the round (B). Both binaries had
the scratch counter built in and idle. Runs were interleaved A B, medians of 3. Load was
11 to 19 from other jobs, and the desk window (a `gbcformer play` process) kept
running.

| `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - |
| 54.9 → 58.1 (A 54.2, 54.9, 55.4; B 58.1, 58.2, 58.0) | 53.4 → 56.6 (A 53.4, 44.5, 55.3; B 19.7, 57.5, 56.6) |

text.sh is still below 60. The cycle drop, about 5.5% four and 5% one model, predicts
about 58 to 59 fps on the earlier quiet 56.5 / 55.7 baseline.

**Rebased onto the hidden-1,024 memory checkpoint with HOST records**: the base
against this work, three interleaved runs each at load 10 to 30. Instructions were 31.04 to
30.36 k (four) and 32.19 to 31.47 k (one model). Cycles were 10.48, 10.65, 10.75 to 9.76,
9.91, 9.94 k (four) and 10.21, 10.46, 10.46 to 9.53, 9.68, 9.60 k (one model), medians
−7.0% and −8.2%. `tools/gate.sh` 28 of 28 ok and battle.sh `14d7468e` under `--weights all
--lockstep all --one-model` (all equal) on the rebased head. The digests are the new
the base's: cpu-memory `4cdad049`, one model `64b3a9c4`, the rest unchanged.
### Build settings: thin LTO and one codegen unit; the four-checkpoint gap is the fit's placement

One commit: `lto = "thin"` and `codegen-units = 1` in `[profile.release]`. No
source change, so every checkpoint digest and token stream is unchanged.

**Method.** Each variant built from the same tree in its own target directory, with a scratch
counter (`thread_selfcounts` around the timed `run_frames`, never committed) that gives
the GBC CPU's thread's instructions and cycles; divided by the run's GBC CPU steps. Three rounds,
each bench and mode with the binaries in rotated order, `--weights all` and
`--weights all --one-model`. Process instructions from `/usr/bin/time -l`. One-minute
load 5 to 46 (other jobs, and this work's own profiled pairs during rounds 1 and 2),
so cycles carry a few percent of spread and fps more; instructions repeat to 0.1%.

| Bench, mode | k instr per step: default → thin LTO, 1 unit (fat LTO, 1 unit) | k cycles per step | process G instr |
| - | - | - | - |
| text, four | 30.66 → 30.39 (30.36) | 10.43 → 10.03 (10.19) | 296.4 → 292.6 (292.5) |
| text, one model | 31.75 → 31.46 (31.79) | 10.12 → 9.79 (9.73) | 327.4 → 321.9 (324.4) |
| overworld, four | 31.87 → 31.48 (31.50) | 10.94 → 11.01 (11.08) | 290.1 → 285.6 (285.2) |
| overworld, one model | 33.33 → 32.91 (33.27) | 10.94 → 10.82 (11.20) | 319.1 → 312.9 (314.5) |
| battle, four | 30.67 → 30.37 (30.41) | 9.75 → 9.86 (9.71) | 392.0 → 384.8 (383.3) |
| battle, one model | 31.92 → 31.60 (31.94) | 9.60 → 9.47 (9.48) | 422.3 → 413.3 (414.0) |

Thin LTO with one unit retires 0.9 to 1.3% fewer GBC CPU-thread instructions in all six
cells and 1.2 to 2.1% fewer process instructions; its cycles are lower on text in both
modes (3.8% and 3.3%) and inside the spread elsewhere. Fat LTO matches it on four
checkpoints but not on the one model's GBC CPU thread (31.75 to 31.79), the release artifact,
and builds slower. One triage round on text: thin LTO with the default 16 units gives
nothing (30.65 and 31.78 k, 296.0 and 327.2 G), and one unit without LTO little (30.59
and 31.71 k), so the win needs both. `panic = "abort"` with fat LTO cut another 0.5% of
instructions in that round (30.27 and 31.67 k). It was not taken, because it is inside the cycle
spread, and it turns a worker thread's lockstep divergence into a SIGABRT in place of
the `resume_unwind` re-raise on the GBC CPU's thread (the tests are unaffected: Cargo builds
them with unwinding, and the one `catch_unwind` outside tests is under `#[cfg(test)]`).
`-C target-cpu=apple-m1` is the target's default already (`rustc --print cfg` is
identical), so it changes nothing.

**Paired timing.** The default and thin LTO binaries started together, start order
alternating, three rounds, one-minute load 34 to 47 (other jobs; nothing of this
branch beside them). GBC CPU-thread cycles per step, medians: text four 10.41 to 10.25 k
(−1.6%, 3 of 3 pairs lower), text one model 10.16 to 10.00 k (−1.6%, 3 of 3), overworld
four 11.81 to 11.71 k (−0.9%, 2 of 3). Emulated fps, medians: text four 53.8 to 54.2
(53.0/54.1, 53.9/54.2, 53.8/54.7), text one model 54.8 to 55.6 (24.3/23.9, 54.8/55.6,
55.2/55.7), overworld four 31.7 to 30.1 at that load (48.5/51.3, 25.3/27.9, 31.7/30.1),
which says more about the load than the binaries. Every run kept `870aa3fd` and
`822059c1`.

**Build time** (clean, dependencies included; loads 11 to 62). The binary: 29 s to 75 s
(fat LTO 88 s). `cargo build --release --all-targets`: 32 s to 104 s; after a one-line
edit of `src/lib.rs`, 25 s to 143 s (load 51), and the binary alone 18 s to 59 s. The
gate's own clean build in this worktree took 149 s at load 62. The one unit is the cost:
the library compiles serially, and every test and example links it with LTO.

**The four-checkpoint gap.** The previous section measured the hidden-1,024 layout
slower on four checkpoints at equal instructions. Profiled pairs (the scratch stage
profiler, both binaries started together, start order alternating, text.sh,
`--weights all`), the HOST record (hidden 2,048) against the token channels (hidden 1,024), four of
four pairs: the GBC CPU's thread's cycles 36.27 to 37.91, 36.60 to 37.39, 37.36 to 37.84 and
36.64 to 37.74 G at 114.5 G instructions both, and the units stage (`Marks::layer`
over the circuit layers) 2,979 to 3,174, 3,043 to 3,206, 3,083 to 3,238 and 3,053 to 3,160
cycles per step at 6,775 instructions both. So the gap is real, and it is the units stage.

Plan-only variants of `Segments::renumber` on the head, each against the head in four
pairs (units cycles per step, head first): no renumbering (dense index first, later
occupants from `hidden`) 3,262/3,312, 3,271/3,282, 3,322/3,298, a tie; wire occupants
numbered `2,048 + wire` as at hidden 2,048, 3,263/3,253, 3,261/3,226, 3,275/3,289, a
tie; regrouped by the first layer that reads them, 2 to 3% worse in four of four; the
hottest wires first by `changes-memory.txt`, 2 to 7% worse in four of four. The
creation order the plan uses is the best numbering tried, and packing the busy rows
together costs cycles, so the virtual channel order is not the lever.

The fit is. The head with only the memory circuit's fit budget raised to hidden
2,048's (the class wires kept) recovers the gap in four of four pairs: units 3,227 to
3,118, 3,274 to 3,075, 3,355 to 3,250 and 3,328 to 3,262; GBC CPU thread 2.1 to 2.9% fewer
cycles at 0.2% more instructions. Both run 744 units and 142.6 changed pairs per pass
(the fit's own objective sees no difference); the work sits in other layers (units by
layer [48, 69, 46, 112, 94, 89, 81, 77, 87, 42] at 871 channels, [48, 72, 48, 123, 96,
73, 76, 75, 90, 42] at the 2,048 budget), and the text pass takes 3.04 against 2.94 µs.
With the one model the same change helps the units stage but not the GBC CPU's thread (0.1 to
1.1% more cycles in four of four), and it doubles the cpu-memory checkpoint, so it is
not taken. Why a placement with the same updates costs more cycles (branch history or
memory-level effects) is not measured: `thread_selfcounts` gives no miss or mispredict
counts. The next thread is a fit objective scored in cycles rather than pairs, a
circuit change that moves the cpu-memory and one-model digests.

### Engine: the gate walk stages its sends

The gate loop in `Marks::layer` was about half of the GBC CPU's thread's cycles at IPC 2.1.
This round moves its sends out of the update loop. Two changes were kept. Counts are
GBC CPU-thread instructions and cycles per GBC CPU step from `thread_selfcounts` (scratch
counter, not committed) on text.sh. "four" is `--weights all`, and "one" adds
`--one-model`. Every run kept `870aa3fd`.

**Populations first.** A scratch counter over the gate loop on text.sh (one model, per
GBC CPU step) measured 100.8 changed (channel, layer) pairs, about 700 gate updates, 116
sends and 90.5 first touches of an output channel. Spans are short. Of the plain spans
39.2 had one entry, 14.5 two and 11.3 none; of the pair spans 43.6 had none and 21.0
one. So about one update in six sends, and about one send in five finds its channel
already marked. The drain saw 90.5 touched channels, 22.9 of them summing to 0, and
82% of the nonzero ones had one reader.

**Staged sends**. The walk computes every update's activation change
without a branch, stores the unit and the change packed in one `u64` at index `k`, and
moves `k` on only when the change is not 0. A second loop over the `k` kept entries
does the output-channel mark and the accumulator add. The buffer is `Marks::sends`, a
flat scratch array rewritten by every call and never read across calls. The pair's
activation change became `(old >> 31) - (new >> 31)` on the sign bits. The walk is 16
instructions an update for a plain unit and 13 for a pair, against 10 for the branchy
loop that did not send.

**Branchless first touch**. The send loop stores the channel at the end
of `touched` every time and moves the length on only when the mark was down.

Medians of 3 interleaved runs, thin-LTO profile (the base build), load 11 to 31:

| Binary | k instr four / one | k cycles four / one |
| - | - | - |
| base | 30.12 / 31.23 | 9.48 / 9.25 |
| staged sends | 34.14 / 35.29 | 9.06 / 8.97 |
| branchless first touch | 34.12 / 35.26 | 8.58 / 8.56 |

From base, cycles fell 9.5% (four) and 7.5% (one model) for 13% more instructions.
Before the rebase (default profile, on round 6) the staged sends measured 9.66 to
8.88 k (four) and 9.43 to 8.86 k (one), so the gain holds across the profile change.

**Tried and reverted** (default profile, on round 6, single runs unless noted):

- The whole update branchless in one loop over flagged entries, with the output
  channel, the down weight and the pair flag packed into an 8-byte entry: +15.6 k
  instructions, +0.5 to +0.8 k cycles. The inner loop was 34 instructions.
- The same with the plain and pair spans kept apart (constant caps): +11.3 k
  instructions, +0.2 to +0.6 k cycles.
- The branchy send with the output channel and down weight in the entry (no per-unit
  loads on the send path): +0.77 k instructions (one `asr` an update), cycles within
  noise.
- Prefetching every change's span before the walk and the next change's entries
  during it: +1.5 k instructions, +0.36 to +0.4 k cycles. The span and entry loads are
  not the stall.
- The staged walk over one span a channel with the pair flag in the entry and a
  selected cap: +6.2 k instructions and +0.7 to +0.9 k cycles over the staged sends.
- The drain staged the same way (every touched channel adds its change, the nonzero
  ones are routed after): +1.5 k instructions, +0.3 k cycles.
- Plain units' gates kept as g instead of g − 1 (two adds fewer an update): −0.78 k
  instructions, cycles +0.17 k (four) and +0.03 k (one model), medians of 3. Not kept.
- The memory heads walked from per-layer head bitsets instead of a list: instructions
  within 0.06 k. The list scan was not the cost.

**What is left** (sampled stage profile of the staged sends, one model, per machine
step): the gate loop about 2,900 cycles at IPC 3.2, the drain about 570, the memory
heads about 480, the routing layer about 360 and the decode about 250. Without miss or
mispredict counters the remaining gate-loop stall is not split between loop exits
(two a changed pair) and cache misses.

**Gates** on the branchless first touch: `tools/gate.sh` 28 of 28 ok, `cargo test --release` included.
battle.sh under `--weights all --lockstep all --one-model` reached `14d7468e`, with 86,400
lines, 495,276 APU passes and 108,880 control passes equal. The digests are the base's:
cpu `1f4c3519`, cpu-memory `4cdad049`, apu `ad161b52`, control `c3187a58`, ppu
`d84c03c5` and one model `64b3a9c4`. No weight changed.

**Emulated fps** on text.sh, the base (A) against the branchless first touch (B), thin-LTO profile.
Both binaries had the scratch counter built in and idle. Runs were interleaved A B,
medians of 3, with the machine nearly quiet (load 2 to 7 over the last minute).

| `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - |
| 62.7 → 68.1 (A 62.7, 62.8, 62.6; B 68.2, 68.1, 67.9) | 64.5 → 68.0 (A 64.9, 64.3, 64.5; B 68.0, 68.0, 67.6) |

text.sh now runs above 60 fps with everything in the model, as one checkpoint or four.

### Measurement: text boxes past 60 fps with everything in the model

Headless `tools/bench/*.sh`, M1 Max 64 GB, thin-LTO release build, `--weights all` (four checkpoints) and `--weights all --one-model`, interleaved. An Arq backup kept the one-minute load at 4.7 to 6.9 during the text rounds; each text pair waited for load ≤ 6 before starting.

| Bench | Four checkpoints | One model | Runs | Load |
| - | - | - | - | - |
| text (`870aa3fd`) | 67.4 (38.0, 59.6, 67.4, 68.0, 68.6) | 67.2 (65.9, 67.0, 67.2, 67.7, 68.2) | 5 | 4.7–6.9 |
| overworld (`822059c1`) | 107.8 (87.2, 107.8, 108.2) | 108.0 (90.6, 108.0, 108.1) | 3 | 5–32 |
| battle (`14d7468e`) | 90.1 (89.2, 90.1, 90.9) | 88.7 (86.1, 88.7, 89.6) | 3 | 6–21 |

The 38.0 and 59.6 text runs coincided with backup bursts. The release bar (text.sh at 60 fps or more with the one model on this machine) holds with every one-model run at 65.9 or above. Overworld and battle ran while another project's tests pushed the load to 32, so treat them as lower bounds.

Rerun of overworld and battle at load 4.0 to 4.9, interleaved, medians of 3, same binary:

| Bench | Four checkpoints | One model |
| - | - | - |
| overworld (`822059c1`) | 106.9 (103.3, 106.9, 112.7) | 111.8 (108.4, 111.8, 112.6) |
| battle (`14d7468e`) | 93.5 (92.9, 93.5, 93.8) | 92.1 (91.9, 92.1, 92.3) |

Open: the quiet baseline gave overworld 117.4 (four) and 114.8 (one model) at load 2 to 7. Since then text rose 55.0 to 67.4 and battle 81.5 to 93.5, but overworld did not. Either overworld regressed in one of hidden-1024, round 6, thin LTO or round 7 (round 7 runs about 13% more instructions per step), or the two loads differ enough to explain it. The next section resolves it. Round 7's staged walk slowed the PPU thread, which bounds overworld.

### Overworld: the PPU thread and the staged walk

Overworld did regress, and only at round 7's staged sends. That commit made
the PPU thread the slowest thread on overworld.sh. The GBC CPU thread got faster there, as it
did on text.sh. Text is bound by the GBC CPU's thread, so it gained, and overworld is bound by
the PPU thread, so it lost. The next commit gives the positional stage back the branchy
walk and wins overworld back with text unchanged.

**Per-commit counts.** Each commit was built with the scratch counter (`thread_selfcounts`
per thread, not committed), in its own profile: commits before thin LTO without LTO, and
thin LTO and after with thin LTO and one unit. Each cell is a thread's k instructions / k
cycles over the 600 model frames, divided by the GBC CPU steps (1,179,826 every run). The
palette column is the GBC CPU thread's wait at the 849 palette reads (`ppu: the CPU's thread
waited`). One run per cell, sequential, one-minute load 3.6 to 22. Instructions repeat to
0.1%, and cycles and fps follow the load. Every run kept `822059c1`.

| Commit | mode | CPU thread | PPU thread | APU thread | palette wait | fps |
| - | - | - | - | - | - | - |
| quiet baseline | four | 35.77 / 11.73 | 46.11 / 9.77 | 24.32 / 8.16 | 664 ms | 115.8 |
| quiet baseline | one | 37.76 / 12.88 | 46.25 / 12.42 | 25.97 / 10.88 | 3,019 ms | 63.6 (a load burst, see below) |
| one-hot fetch | four | 34.09 / 10.94 | 46.33 / 9.97 | 24.33 / 8.30 | 862 ms | 117.3 |
| own route totals | four | 33.18 / 10.97 | 46.06 / 10.09 | 23.75 / 8.32 | 894 ms | 116.0 |
| own route totals | one | 34.67 / 11.35 | 46.16 / 10.13 | 25.50 / 9.21 | 843 ms | 113.0 |
| HOST record, hidden 1,024 | four | 32.90 / 11.08 | 46.06 / 9.74 | 23.75 / 8.21 | 787 ms | 118.1 |
| HOST record, hidden 1,024 | one | 34.43 / 10.98 | 46.17 / 9.83 | 25.22 / 8.89 | 834 ms | 117.2 |
| fit over flat arrays | four | 32.91 / 11.22 | 46.07 / 10.02 | 23.75 / 8.33 | 823 ms | 115.9 |
| fit over flat arrays | one | 34.44 / 11.11 | 46.16 / 9.90 | 25.22 / 9.04 | 817 ms | 115.5 |
| round 6 | four | 32.18 / 10.32 | 45.89 / 9.80 | 23.46 / 7.97 | 939 ms | 121.5 |
| round 6 | one | 33.68 / 10.28 | 45.98 / 9.82 | 24.93 / 8.62 | 951 ms | 120.5 |
| thin LTO | four | 31.77 / 10.25 | 45.90 / 9.64 | 23.34 / 7.80 | 913 ms | 123.0 |
| thin LTO | one | 33.25 / 10.11 | 45.98 / 9.68 | 24.78 / 8.47 | 949 ms | 122.3 |
| staged sends | four | 35.81 / 9.57 | **54.54 / 11.56** | 27.82 / 7.66 | 1,665 ms | 111.4 |
| staged sends | one | 37.32 / 9.57 | **54.63 / 11.58** | 29.35 / 8.17 | 1,660 ms | 110.7 |
| branchless first touch | four | 35.78 / 9.17 | 54.46 / 11.57 | 27.85 / 7.37 | 1,761 ms | 112.7 |
| branchless first touch | one | 37.31 / 9.21 | 54.55 / 11.62 | 29.38 / 7.77 | 1,753 ms | 111.7 |

The quiet baseline's one-model run hit a load burst. Its GBC CPU, PPU and APU threads all show about
25% more cycles at the same instructions as the four-checkpoint run, and its palette wait
is 3 s, so its fps measures the load. The one-hot fetch's one-model cell is left out for the
same reason (load 19, 73.5 fps).

**What the counts rule out.** From the quiet baseline to thin LTO the PPU thread holds at 46
k instructions and 9.6 to 10.1 k cycles. The GBC CPU thread falls from 11.7 to 10.25 k cycles,
and overworld rises from 116 to 123 fps. So the hidden-1,024 merge, round 6 and thin LTO
helped overworld or did nothing to it. Over the same span the GBC CPU's thread had the most
cycles per step. At the staged sends the PPU thread rises 18.8% in instructions and 20% in
cycles. The GBC CPU thread falls to 9.57 k, so now the PPU thread has the most cycles. The
bound moves with it: 123 × 10.25 / 11.57 ≈ 109 fps, beside the 111 to 113 measured.
The branchless first touch does not change the PPU thread. The APU thread also gains 19% in instructions
at the staged sends, but its cycles fall a little, and it is the least loaded thread. The
palette wait doubles because the PPU thread falls further behind before each read
drains its queue.

**Why the PPU paid.** Scratch counters in `Marks::layer` (updates and kept sends per
thread, split by stage, head of the base):

| Thread, bench | shared layers: updates, sends | positional stage: updates, sends |
| - | - | - |
| CPU, overworld | 864.8 M, 134.6 M (15.6%) | none |
| APU, overworld | 1,365.8 M, 90.1 M (6.6%) | none |
| PPU, overworld | 203.5 M, 11.2 M (5.5%) | 1,556.6 M, 489.1 M (31.4%) |
| PPU, text | 53.4 M, 8.6 M (16.1%) | 1,037.1 M, 505.2 M (48.7%) |

The PPU's work is the positional stage: 88% of its updates on overworld. Round 7 sized
the staged walk on the GBC CPU's thread, where about one update in six sends. In the
positional stage a third to a half of the updates send. There the store-every-update
walk and its second loop cost more than the branchy walk's mispredicts.

**The change**. `Marks::layer` takes the walk as a `const STAGED: bool`.
`Engine::advance` (the shared circuit layers) passes `true`, and `Positional::advance`
passes `false`, which is the send-in-place loop from before the staged sends (`Marks::walk_branchy`).
No weight changes, and no new state.

Two variants were measured, each against the branchless first touch, twice each, interleaved, at load 5.6
to 7.5. (a) is only the positional stage branchy. (b) is a checkpoint with a positional
stage walking its shared layers branchy too.

| Bench, mode | branchless first touch, CPU / PPU thread | (a) | (b) |
| - | - | - | - |
| overworld, four | 35.79 / 54.48 k instr; 9.28, 9.41 / 11.92, 11.91 k cycles | 35.78 / 46.67; 9.26, 9.31 / 9.93, 10.10 | 35.82 / 45.91; 9.30, 9.34 / 10.02, 9.89 |
| overworld, one | 37.32 / 54.57; 9.44, 9.32 / 11.95, 12.02 | 37.31 / 46.76; 9.36, 9.32 / 10.13, 10.16 | 37.35 / 45.99; 9.29, 9.34 / 9.94, 9.89 |
| text, four | 34.12 / 14.45; 8.61, 8.48 / 3.47, 3.38 | 34.12 / 12.35; 8.48, 8.42 / 2.68, 2.68 | 34.15 / 12.28; 8.47, 8.42 / 2.65, 2.65 |
| text, one | 35.27 / 14.47; 8.55, 8.51 / 3.44, 3.41 | 35.26 / 12.37; 8.43, 8.44 / 2.67, 2.67 | 35.30 / 12.31; 8.48, 8.62 / 2.66, 2.86 |

(a) is kept. (b) gains about 0.1 k PPU cycles more on overworld, but it costs every GBC CPU
step 0.03 k instructions for the arm that tests the plan, and it needs a flag in the
plan. Text's GBC CPU thread is unchanged under (a), at 34.12 k instructions (four) and
35.26 k (one model), with cycles inside the base's spread. The PPU thread on battle.sh is
28.86 to 24.04 k instructions and 6.88 to 5.52 k cycles (four), with the GBC CPU's thread
unchanged (34.16 k).

On overworld the PPU thread still has the most cycles after the change: 10.0 to 10.2 k
against the GBC CPU's thread's 9.3 k.

**Emulated fps**, the branchless first touch (A) against the fix (B). Both binaries had the scratch
counter built in and idle. Runs were interleaved A B, medians of 3, at one-minute load 7
to 25 (other jobs).

| Bench | `--weights all` A → B (runs) | `--one-model` A → B (runs) |
| - | - | - |
| overworld.sh | 104.6 → 117.8 (A 58.7, 109.3, 104.6; B 115.9, 120.4, 117.8) | 105.8 → 117.4 (A 104.7, 105.8, 106.1; B 117.4, 108.5, 121.0) |
| text.sh | 65.6 → 66.7 (A 65.6, 66.8, 65.6; B 66.7, 67.4, 66.7) | 63.8 → 66.2 (A 63.8, 62.4, 65.7; B 65.7, 66.2, 66.3) |
| battle.sh | 87.7 → 92.7 (A 88.7, 87.7, 56.6; B 93.8, 54.9, 92.7) | 87.1 → 91.4 (A 87.1, 88.0, 86.5; B 91.4, 90.4, 91.6) |

In the counted runs at load 4 to 7, overworld ran 107.0 to 110.7 fps on the branchless first touch
against 119.8 to 124.1 on the change. Text shows no measurable change. Its GBC CPU thread
retires the same instructions, its cycles are no worse in 4 of 4 pairs (two runs a side),
and the fps difference is inside the run-to-run spread.

The counted runs of (a) used a scratch binary built before the fix, marked
`walk_branchy` `unsafe` and reworded the doc comment. The function body is identical, and
the gates below ran on the committed source.

**Gates** on the fix. `tools/gate.sh` passed 28 of 28, its `cargo test --release`
included (the engine against the dense pass, `ppu_checkpoint`, `one_model`). battle.sh
under `--weights all --lockstep all --one-model` reached `14d7468e`. In that run 86,400
lines were equal colour for colour, as were 495,276 APU passes, 108,880 control passes
and all 35 palette reads. The change touches only the engine's walk, so no weight changed
and the digests are the base's.

**What limits overworld next.** The PPU thread, at 46.7 k instructions and 10.0 k cycles
per GBC CPU step, with the positional stage carrying 88% of its gate updates. Then the
palette reads. Each of the 849 reads (about 1.4 a frame) waits for the PPU thread to draw
every line and frame queued before it, 1.2 to 1.3 s in all after this change. The read
depends only on the palette records, not on the lines. A read the worker could answer
without drawing first would need the memory index to hold records by when they were
kept, so a line still sees only the records kept before its mode 3. That is not
measured or built.

### CGB double speed

STOP with KEY1 armed now switches speed, in the reference and in weights
(`docs/control-in-weights.md` §17). Zelda DX switches at frame 26 and Mario DX at frame
275 (the first frame counts whose hash differs from the base on the reference); Crystal
and Yellow never switch in 1,500 frames.

**Digests** (`examples/digest`): cpu `1f4c3519` to `a6e818e1` (the GBC CPU circuit reports
STOP in CTL' bit 4; 109.5 M, nonzeros 0.13 M), cpu-memory `4cdad049` to `554edd52`
(67.3 M), control `c3187a58` to `780d33cf` (44.1 M to 54.5 M, 45,785 to 56,779
nonzeros: the speed, the pause, the halves input and the double-speed horizon), one
model `64b3a9c4` to `490c6845` (1,472.7 M to 1,485.3 M dense, 1.48 M nonzeros). apu
`ad161b52` and ppu `d84c03c5` are unchanged.

**Game hashes.** Under `--core lockstep` at 1,500 frames, Zelda DX `1d00d871` to
`99f51381` and Mario DX `7dec3947` to `b083771f`. With all five components in weights in
lockstep at 300 frames, Zelda DX `7b126eb8` to `e2bd7038` (the reference's hash too) and
Mario DX `3e1f4d27` to `ea7f86ca`. Master's frames were the wrong ones. At normal speed
Zelda DX's intro draws a black band over its first lines at frame 300, and Mario DX's
title screen at frame 900 shows wrong palette colours and orange blocks in the clouds.
Both are clean at double speed. Yellow (`cfd31f3d`, `870aa3fd`, `822059c1`, and battle.sh `14d7468e` with every component in weights in lockstep as one model) and
Crystal (`ccfe7ad3`, `4e6a7509`) are unchanged.

**GBC CPU thread per step on text.sh** (`thread_selfcounts`, k instructions / k cycles per
GBC CPU step, the base build against the branch, interleaved, each binary twice):

| Mode | base | this work |
| - | - | - |
| four checkpoints | 34.12 / 8.60, 34.12 / 8.65 | 34.08 / 8.86, 34.08 / 8.57 |
| one model | 35.26 / 8.54, 35.27 / 8.61 | 35.39 / 8.73, 35.39 / 8.55 |

Four checkpoints run 0.13% fewer instructions a step. The one model runs 0.36% more:
the sampled marks put it in the block loop (+46 instructions a step) and decode
(+14). The control stream's units per fast pass rose 358 to 389 (`GBCFORMER_STATS`).
The first cut, which compared the PPU's position once per speed, rose to 463 and cost
0.45%. Cycles a step are within run-to-run noise.

**Emulated fps** on text.sh `--weights all --one-model`, the base build (A) against the
branch (B), both with the scratch counter built in and idle, interleaved A B, three
rounds, one-minute load 1.9 to 2.5: A 68.3, 68.3, 68.4 (median 68.3); B 67.7, 68.1,
68.1 (median 68.1). The one model is 0.3% slower on text boxes, in line with its 0.36%
more instructions a step. Not yet measured: whether the control stream's eight new
slots (six inputs, two results) widen every stream's per-pass work in the shared
checkpoint, which the four-checkpoint run, 0.13% fewer instructions, would not show.

### Measurement: release candidate (double speed and the positional walk in)

Headless, M1 Max 64 GB, thin LTO, interleaved, medians of 3, every run at its frame hash, load 2.3 to 6.0. `tools/gate.sh` 28 of 28 on this commit; `tools/speed-switch.sh` div 0, ly 109, stat 0 pixels off.

| Bench | Four checkpoints | One model |
| - | - | - |
| text (`870aa3fd`) | 69.5 (66.9, 69.5, 69.8) | 69.5 (69.3, 69.5, 69.6) |
| overworld (`822059c1`) | 127.4 (126.7, 127.4, 127.4) | 126.3 (126.2, 126.3, 126.5) |
| battle (`14d7468e`) | 97.1 (96.2, 97.1, 98.1) | 95.9 (95.4, 95.9, 96.0) |

This closes the "Open" note under "Measurement: text boxes past 60 fps": overworld is above its quiet baseline (117.4 / 114.8) again. One model: 1,485.3M dense, 1.48M nonzero (`490c6845`).

### f16 and one KV head: the one model in a stock runtime (`docs/stock-llama.md`)

This is not a speed change. The redesign lets a stock runtime hold the model, exact under
f16 storage with f32 accumulation (llama.cpp's Metal half tiles, MLX, its CPU backend). The
numbers come from `examples/stock`, `examples/one_model` and `examples/digest` (the first
two have since been removed).

| | before (`490c6845`) | after (`9c4c342a`) |
| - | - | - |
| KV per cached token | 186,368 B f32 | 3,840 B f32, 1,920 B f16 |
| Yellow's cache | 1,161,187 tokens, 216.4 GB | 1,131,743 cells, 4.35 GB f32, 2.17 GB f16 |
| dense parameters, nonzeros | 1,485.3 M, 1,482,153 | 1,311.3 M, 1,386,384 |
| layers, heads, KV heads, hidden, intermediate | 14, 52, 52, 4,096, 4,736 | 15, 52, 1, 4,096, 4,416 |

Digests after: cpu `56729ea4` (111.9 M), cpu-memory `aa0e95b4` (114.9 M; the f16 splits
and the AND one-hots leave 1,026 live wires, over hidden 1,024's budget, so hidden is
2,048), apu `24601491` (85.5 M), control `f626b5c8` (46.9 M), ppu `98460969` (484.7 M),
one-model `9c4c342a` (1,311.3 M). Every digest moved, because the scales (UNIT, EPS, GAMMA,
K = 128, two constant channels), the f16 splits, one cell per address, the nibble values
and the shared layer-0 key change every checkpoint's weights.

This entry does not measure engine speed with the new circuits, since correctness came
first. The frame hashes of `tools/gate.sh` are unchanged.


## T2: exact under f16 accumulation (first build)

`docs/stock-llama.md` §10. Every checkpoint passes `Model::check_f16_acc` and
`check_f16`. `tools/gate.sh` passes with 30 of 30 checks, and `tools/bench/battle.sh`
reaches `14d7468e`. This row does not measure engine speed, since correctness came first.

| | T1 (`9c4c342a`) | T2 first build (`7a15db19`) |
| - | - | - |
| depth: cpu-memory, apu, control, ppu | 9, 13, 11, 14 | 10, 15, 14, 14 |
| one model: layers, heads, hidden, intermediate | 15, 52, 4,096, 4,416 | 16, 54, 4,096, 4,800 |
| one model: dense parameters, nonzeros | 1,311.3 M, 1,386,384 | 1,463.3 M, about 1.38 M |

Digests: cpu `d9f82a97` (111.9 M), cpu-memory `98a9c6dd` (119.1 M), apu `526218c0`
(96.2 M), control `896daaa9` (53.2 M), ppu `98460969` (484.7 M, unchanged, because the
PPU already met the T2 condition), one-model `7a15db19` (1,463.3 M). The digit
splits on the APU's noise timer, mixer, filter and capacitor, on control's gated byte
compares and on the GBC CPU's 16-bit increment add levels. The APU's depth of 15 sets
the one model's 16 layers. The next row removes those added levels.

### T2 depth: the levels the f16-accumulation splits cost, won back

This is not a speed change. T2 (every gate's partial sums within 511·K, every down row's
within 2,046 units of 2^-7, `docs/stock-llama.md` §6) cost levels where the compiler split
steps by digits: APU 13 to 15, control 11 to 14, the GBC CPU with memory 9 to 10, and the
one model 15 to 16 layers. Three generic changes in `circuit.rs` and two builder rewrites
remove those levels. The census reports 0, and `check_f16` and `check_f16_acc` pass on
every checkpoint.

- **Reduction on every step f16 cannot hold.** `reduce` ran only on steps already
  f16-legal, so a 16-bit increment's carries took the digit path. It now runs on any
  step that fails `step_ok`, and `staircase` keeps the reduction when it is shallower
  than its byte digits. `reduce` also stops at a constant core. It used to add OR
  literals the core did not depend on, and a lockstep test caught that once reduction ran
  on wide steps.
- **Twin wires (`balance`).** A twin carries a node's signal over the other folded
  constant, so its products fall on the other side of a gate's partial sums, at no
  extra level. `pc+.c8a` (live sides 257 and 258, positive products 513) fits that way.
- **Depth-chosen blocks in `sum_blocked`.** The boundaries keep every block within one
  gate and minimise the last block's level first, then the units.
- **APU mixer and filter.** The mix is `32·(2A − 15B)`, where A is the level wires (at
  most 480) and B the DAC wires (at most 32). The circuit computes `filtered = F + 64A − 480B`
  and `cap' = G + 64A − 480B` side by side, with `F = −(cap >> 8)` and `G = cap + F` taken
  from the input in free levels.
- **Control.** The LY compares are three span cases over LY alone. The PPU horizon's T1
  is muxed per case K, and its compares are made per case, gated by the case, one level
  after the cases. `hirq` reads ¬TIMA' through its sources.

| checkpoint | depth before → after | widest level (units) | dense parameters | digest before → after |
| - | - | - | - | - |
| cpu | 5 → 5 | 1,525 → 1,525 | 111.9 M → 111.9 M | `d9f82a97` → `ae67a69b` |
| cpu-memory | 10 → 9 | 1,056 → 1,148 | 119.1 M → 114.9 M | `98a9c6dd` → `f4d4ce7b` |
| apu | 15 → 13 | 936 → 811 | 96.2 M → 88.2 M | `526218c0` → `1a7dc57d` |
| control | 14 → 11 | 320 → 384 | 53.2 M → 49.2 M | `896daaa9` → `2a099c98` |
| ppu | 14 → 14 | 3,744 → 3,744 | 484.7 M → 484.7 M | `98460969` (unchanged) |
| one model | 15 → 14 (16 → 15 layers) | 4,181 → 3,898 | 1,463.3 M → 1,287.7 M | `7a15db19` → `72ab78bd` |

The "before" column is the build before this entry. The one model is back to the PPU's depth. This entry
does not measure engine speed. Reduction and twins change the GBC CPU checkpoints' units,
so the text.sh and overworld.sh timings need a fresh A/B. Their frame hashes are
unchanged.

## Stock runtime: each stream's group holds only its own slots (hc-tokens)

Workload: `tools/stock-llama/bench.sh`, Yellow with 141,415 line cells, stock llama.cpp
b12070 on Metal, M1 Max, 300 passes from the saved cache, median of 3, with every decoded
token diffed against the engine's record. The projection multiplies each stream's ms
per pass by its passes to frame 300 (CPU 981,204, PPU 83,501, APU 243,952, control
59,052). `docs/stock-llama.md` §12a has the census and the exactness argument.

| change | groups (CPU, PPU, APU, control) | ms per pass (CPU, PPU, APU, control) | hours to frame 300 | load | one-model digest |
| - | - | - | - | - | - |
| baseline: every input slot, own outputs, the CPU's write slots | 185, 309, 207, 189 | 681, 1,081, 725, 674 | 271 | 0.7 to 1.7 | `c6f09388` |
| `one_model::group`: another part's slots only where a layer-0 gather needs them | 98, 189, 136, 100 | 405.9, 690.1, 475.9, 408.1 | 165.6 | 2.4 to 5.5 | `c6f09388` |
| per-part gathers with a fallback on the stream's own masks | 77, 170, 120, 82 | 377.0, 653.8, 443.0, 373.5 | 154.1 | 3.1 to 14.5 | `c9fb64ad` |
| and a CPU second pass's settled request slots left out | 77 (76 on average), 170, 120, 82 | 379.1, 655.2, 443.0, 374.2 | 154.7 | 2.4 to 6.4 | `c9fb64ad` |

The digest moved because the one model's layer-0 gathers now group each part's inputs
on their own and every output mask keys on a fallback dim. Every other checkpoint's
digest is unchanged, and so is the dense size (2,087.3 M, 1.49 M nonzeros). Each step
costs less per token than the 3.6 ms the timing table shows, because the GBC CPU's group
is near the 66-token knee.

## Stock runtime: scoped attention (H4)

Workload as in the hc-tokens entry: `tools/stock-llama/bench.sh`, 300 passes from the
saved cache, median of 3, stock b12070 with `layer-shapes.patch` on Metal, M1 Max, the
GGUF exported with `--layer-shapes --align 64`. `docs/stock-llama.md` §12a has the
mechanism and the exactness argument.

| change | cells a decoded row attends (CPU, PPU, APU, control) | ms per pass (CPU, PPU, APU, control) | hours to frame 300 | load | one-model digest |
| - | - | - | - | - | - |
| base (H1 to H3b) | 141,415 each | 212.6, 384.0, 234.7, 219.6 | 86.4 | | `c9fb64ad` |
| A: PPU, APU and control scoped to the lines their heads can name | 141,415, 2,091, 2,091, 2,091 | 212.1, 76.2, 49.2, 44.2 | 63.6 | 2.0 to 3.0 | `c9fb64ad` |
| B: and the CPU to the windows its `MAP` slots name | 7,271, 2,091, 2,091, 2,091 | 53.2, 76.3, 49.3, 44.0 | 20.3 | 3.2 to 4.6 (`tools/gate.sh` finishing, a CPU audit running) | `b03d896c` |

B's bench window (the first 300 passes) has no bank switch. Over the 300 frames, 2,895
of the 981,204 GBC CPU passes follow a banking write and attend every cell, which adds
about 0.1 hours (computed). This change did not re-record `src/sm83/changes-memory.txt`.
That table weighs `Circuit::fit` for the cpu-memory checkpoint's engine, and the one model
fits with even weights.

## The harness: weights from the GGUF, reads over the cells' K/V rows (`docs/harness.md`)

Workload: `tools/bench/text.sh --weights all --one-model`, frame `870aa3fd`, M1 Max,
`GBCFORMER_STATS=1`, interleaved A, B, A, B, A, B, other load throughout.

| change | fps (3 runs) | median | CPU step, µs | load |
| - | - | - | - | - |
| the build before the harness, as is (1 run) | 1.4 | 1.4 | 70.9 per 72-position pass, 12,600 per 67-position pass | 3.2 |
| A: the build before the harness, with the layer-0 plan fix (query ids, memoised plans) | 24.2, 23.6, 23.6 | 23.6 | 7.7 to 7.9 | 2.8 to 6.2 |
| B: and weights from the GGUF, every memory read over the cells' f16 rows (guarded lookup, scan fallback) | 24.0, 23.8, 23.2 | 23.8 | 7.8 to 8.0 | 2.4 to 6.5 |

A and B show no measurable difference. Before the layer-0 plan fix, the MAP slots made layer 0
replan twice per host-served step, and each replan compared every decoded
position's logits over every key (about 13 ms). Bit-identical queries now skip the
comparison, and the engine memoises plans by the key set (`docs/harness.md` §6).

### Harness milestone 2: write records' rows by layer group (`docs/harness.md` §8.1)

Text and overworld workloads, `--weights all --one-model`, `GBCFORMER_GGUF`, interleaved
A (milestone 1), B, three rounds each, `GBCFORMER_STATS=1`.

| workload | A fps | B fps | median A, B | CPU step A, B (µs) | load |
| - | - | - | - | - | - |
| text | 23.5, 23.6, 23.8 | 30.4, 30.0, 30.6 | 23.6, **30.4** | 7.87, 6.13 | 2.6 to 3.2 |
| overworld | 29.4, 29.7, 29.2 | 30.1, 30.7, 30.1 | 29.4, **30.1** | 17.2, 16.8 (counts the PPU wait) | 2.4 to 3.0 |

The GBC CPU step fell 1.7 µs of the 2.5 µs the write records cost. Layers 1 to 13 have
equal rows (equal weights), so one group computes and stores them, and layer 0's scores
come from a table. The PPU thread bounds overworld (108 µs a pass).

### Harness milestone 2: the PPU's positional stage in four parts (`docs/harness.md` §8.2)

Same setup; A is §8.1's binary.

| workload | A fps | B fps | median A, B | PPU pass A, B (µs) | load |
| - | - | - | - | - | - |
| overworld | 30.5, 30.3, 29.1 | 68.8, 69.2, 67.8 | 30.3, **68.8** | 107.7, 36.7 | 3.1 to 5.3 |
| text | 30.5, 31.2, 30.5 | 30.7, 30.9, 30.9 | 30.5, **30.9** | 101.9, 36.1 | 3.6 to 5.6 |

The 160 positions' units run in four parts of 40 positions, three of them on helper
threads. The GBC CPU stream's thread still bounds text.

### Harness milestone 2: memory reads from the row's derived line (`docs/harness.md` §8.3)

Text, A is §8.2's binary: 30.7, 31.2, 31.1 against 32.4, 32.2, 32.3 fps (median 31.1 to
**32.3**), GBC CPU step 5.97 to 5.75 µs, load 3.3 to 4.2. The harness makes a read's key
comparison and its 0/1 test once, when it writes the row.

### Harness milestone 2: the write-record memo (`docs/harness.md` §8.4)

A is §8.3's binary. Text: 32.4, 32.1, 32.5 against 35.0, 34.7, 35.1 fps (median 32.4 to
**35.0**), GBC CPU step 5.74 to 5.31 µs, load 3.5 to 4.9. Overworld: 71.6, 71.5, 71.7
against 73.5, 73.5, 73.9 (71.6 to **73.5**), load 2.3 to 3.1. A record whose tokens (byte
address and line) equal a memo entry's takes that entry's rows, which the harness built
from the weights for those tokens. The harness rebuilds and compares every 64th hit.
888,081 of text's 1,024,869 GBC CPU write records hit.

Rejected after it: unrolling the gate walk's loops by two. One pilot pair on text gave
35.4 against 33.8 fps (5.26 against 5.51 µs a step), and the change was reverted
(`docs/harness.md` §8.3).

### Harness milestone 2 against milestone 1 (`docs/harness.md` §8.6)

A milestone 1, B milestone 2, interleaved, three rounds, `GBCFORMER_STATS=1`.
Text: 24.3, 24.2, 24.3 against 36.2, 36.0, 35.4 fps (median 24.3 to **36.0**), GBC CPU
step 7.67 to 5.17 µs, load 1.9 to 3.1. Overworld: 29.7, 29.7, 29.7 against 72.8, 72.2,
72.8 (29.7 to **72.8**), load 2.1 to 2.8. The GBC CPU's thread bounds text (the engine
takes 4.56 µs of a 5.17 µs step, the gate walk about 3.0 µs of it), and the PPU is 35%
busy.

### Harness milestone 3: the gate walk in blocks (`docs/harness.md` §9.2)

A milestone 2, B the blocks, interleaved, three rounds, `GBCFORMER_STATS=1`. Text: 34.9,
34.9, 34.8 against 38.8, 38.7, 38.7 fps (median 34.9 to **38.7**), GBC CPU step 5.33 to
4.80 µs, shared layers 4.32 to 3.78 µs, load 3.4 to 4.6. Overworld: 70.4, 69.3, 70.0
against 72.1, 73.0, 73.7 (70.0 to **73.0**), load 3.8 to 5.7. The engine's plan renumbers each
circuit layer's writing units so that units with the same inputs are adjacent. Each
channel's readers then form blocks of consecutive units, which the walk updates four gates
at a time with NEON and left-packs into the send list. It updates the rest one at a time.
A pass makes the same 2,097 gate updates. Rejected on the way (§9.2's table): compaction
alone, one-lane vectors for every reader, aligned windows, a flat work list, touching the
next change's data ahead, bitmap-staged sends, and exact tables for units with identical
inputs (36.3 against 38.4 fps).

### Harness milestone 3: the write-record memo per engine (`docs/harness.md` §9.3)

A is §9.2's binary. B starts the memo empty, grows it with its misses from 2^10 to at most
2^16 entries, and leaves it out of the cartridge's insertion. Text: 37.9, 38.0, 32.8
against 37.9, 37.4, 38.3 fps (median 37.9 both), load 4.0 to 5.2. Overworld: 72.6, 73.4,
64.6 against 73.3, 72.1, 72.1 (72.6 against 72.1), load 4.3 to 5.5. Speed does not change,
and the PPU stream's memo is 4 MB instead of 70. Rebuilding the rows on every hit (a pilot,
two runs) costs about 10% on text (38.4, 37.6 against 34.1, 34.7 fps), so the harness
still rebuilds every 64th hit and compares the tokens on every hit.

### Hardware-shaped GBC CPU, unit 1: word-line selects (C4, `docs/hardware-shaped.md` §2.6)

Every pick of a byte from a fetched line reads one word line (the one-hot of the access's
low three address bits) in place of the bits. Census, `GBCFORMER_UNITS=1
CORE=model-mem`, one run each: text went from 2,091 to **1,498** gate updates per fast
pass and overworld from 1,921 to 1,372. By category on text, byte picks went from 312 to
153, opcode decode from 702 to 348 and operand muxes from 144 to 55. The circuit went from
10,959 to 11,063 units at depth 10, and the widest live wires from 1,240 to 1,265. Both
frames are unchanged (`870aa3fd`, `822059c1`).

A is the build before this unit with its own GGUF (`b03d896c`, re-exported because the file
kept earlier had an older layout), and B is this unit with its GGUF (`466f7c72`):
every component in weights, one model, `GBCFORMER_GGUF`, `GBCFORMER_STATS=1`,
interleaved, three rounds. The machine was also running a backup at about 330%
Mac CPU, a Core ML evaluation and the llama.cpp proof run on the GPU, and load spikes
slowed two A runs.

| workload | side | fps per run | median | CPU µs per pass | shared layers, µs | load (1 min) |
| - | - | - | - | - | - | - |
| text | A | 18.3, 38.8, 38.1 | 38.1 | 10.15, 4.80, 4.89 | 6.56, 3.77, 3.85 | 9.5 to 24.3 |
| text | B | 41.6, 37.9, 41.9 | **41.6** | 4.47, 4.91, 4.44 | 3.46, 3.74, 3.42 | 12.8 to 31.0 |
| overworld | A | 73.6, 9.6, 74.6 | 73.6 | 6.87, 52.47, 6.78 | | 12.3 to 21.3 |
| overworld | B | 74.7, 75.3, 73.6 | **74.7** | 6.77, 6.71, 6.87 | | 8.7 to 31.2 |

The one model's GBC CPU stream went from 2,087 to 1,480 updates a pass. The shared layers
fell only about 0.4 µs for 607 fewer updates, because the removed updates were the
decoder's and the picks' runs of identical readers, the cheapest ones the block walk makes.

Checks: `tools/gate.sh` is green (every step `ok`, the one-model Yellow 300 included),
`examples/f16census all` finds 0 violations on every circuit, and `digest --gguf` on the
new file finds 0 of 1,571,461,120 values differ. `examples/cells_state` (since removed) needed a llama.cpp
cache state of the new file, which needs the GPU, so it was not run. Digests: cpu-memory
`ffe22414` to `b63c3759` (175.3 M), one model `b03d896c` to `466f7c72` (2,057.9 M to
2,043.2 M, intermediate 6,336 to 6,272); cpu `6bd416c1`, apu `ebeeb39c`, control
`254acbba` and ppu `6ee4beb0` are unchanged. `src/sm83/changes-memory.txt` was
re-recorded (text and overworld, averaged).

### Hardware-shaped GBC CPU, unit 3: the banks once per pass (`docs/hardware-shaped.md` §2.6)

The four MAP translations become the effective banks (`Regs::banked`), and every read
translation but the opcode fetch's takes its window's bank bits from them. Banking F0 too
moved its head from layer 2 to 3 (depth 11, 12 layers). That variant was measured and
not kept. The circuit went from 11,063 to 9,517 units at depth 10, the widest live wires
from 1,265 to 1,292, and cpu-memory from 175.3 M to 162.3 M dense (intermediate 1,792 to
1,600). Census, `GBCFORMER_UNITS=1 CORE=model-mem`, one run each: text went from 1,498 to
1,470 gate updates per fast pass (translation 210 to 183), and overworld from 1,372 to
1,351. Both frames are unchanged. The re-recorded `src/sm83/changes-memory.txt` (text and
overworld, averaged) is identical to the table before.

A is the unit before with its GGUF (`466f7c72`), and B is this unit with its GGUF (`02b8ca6d`):
every component in weights, one model, `GBCFORMER_GGUF`, `GBCFORMER_STATS=1`,
interleaved, three rounds. The machine was quiet, since other jobs waited on the
shared lock.

| workload | side | fps per run | median | CPU µs per pass | shared layers, µs | load (1 min) |
| - | - | - | - | - | - | - |
| text | A | 41.4, 41.7, 41.7 | 41.7 | 4.50, 4.46, 4.47 | 3.47, 3.45, 3.45 | 3.6 to 19.9 |
| text | B | 41.3, 41.4, 41.6 | **41.4** | 4.51, 4.50, 4.47 | 3.49, 3.48, 3.45 | 4.1 to 12.3 |
| overworld | A | 75.0, 75.7, 75.3 | 75.3 | 6.74, 6.67, 6.71 | | 3.8 to 8.8 |
| overworld | B | 74.8, 75.2, 75.9 | **75.2** | 6.76, 6.72, 6.66 | | 3.4 to 6.4 |

Speed does not change. The one model's GBC CPU stream goes from 1,480 to 1,452 updates a
pass, under 0.02 µs at the walk's cost. The unit reduces size, not time.

The DMA placement fix (`docs/memory-in-weights.md` §8a) was measured separately. A is
this unit without it, B with it, both on B's GGUF, text only.
41.9, 41.5, 38.7 against 41.4, 41.6, 41.5 fps (median 41.5 both; a load spike slowed A's
third run to 4.81 µs a pass), load 3.4 to 4.5. The fix costs nothing.

Checks: `cargo test --release` (16 binaries, SST 500,000 on circuit and checkpoint
included) and `tools/gate.sh` are green, with every step `ok` and the one-model Yellow 300
included. Yellow, Crystal, Zelda DX and Mario DX 300 with all five in weights and in
lockstep placed 46,912, 46,272, 44,800 and 82,112 DMA bytes from the emitted banks. Each
byte matched the specification, none was refused, and the host served none.
`examples/f16census all` finds 0 violations, and `digest --gguf` on the new file finds 0
of 1,546,885,120 values differ. Digests: cpu-memory `b63c3759` to `438b8783` (162.3 M,
0.23 M nonzeros), one model `466f7c72` to `02b8ca6d` (2,043.2 M, 1.49 M to 1.47 M
nonzeros); cpu `6bd416c1`, apu `ebeeb39c`, control `254acbba` and ppu `6ee4beb0` are
unchanged.

## PU shape: the palette as one read per pixel (`docs/hardware-shaped.md` §13.1)

A the build before the PU work, B its unit 1, interleaved, three rounds, `--weights all
--one-model`, `GBCFORMER_STATS=1`, load 3.5 to 8.5. Text: 37.5, 37.9, 37.6 against 39.0,
39.1, 37.9 fps (median 37.6 to **39.0**), PPU line 72.1 to 25.7 µs. Overworld: 74.7, 74.5,
72.4 against 99.9, 99.6, 99.9 (74.5 to **99.6**), PPU line 71.5 to 30.3 µs. The PPU's
`out` nodes read FLG bit 2 at every position, so the second pass of each CGB line
re-evaluated 157,440 units. Each pixel now reads its palette line through one positional
head and picks its byte in 84 units a position. PPU units per pass on overworld
(`--weights cpu,memory,ppu`): 166,834 to 20,755. Digests: PPU `6ee4beb0` to `f5a04864`,
one model `b03d896c` to `7aee0707`, the rest unchanged.

## PU shape: each pixel fetches its own tile (`docs/hardware-shaped.md` §13.2)

A unit 1, B P1, interleaved, three rounds, `--weights all --one-model`, load 6 to 11.
Text: 38.5, 37.7, 38.0 against 36.6, 36.2, 37.3 fps (median 38.0 to **36.6**), PPU line
26.6 to 39.6 µs. Overworld: 97.5, 95.6, 96.8 against 93.5, 95.2, 94.3 (96.8 to **94.3**),
PPU line 32.5 to 36.5 µs. This size change costs the engine time. The PPU's units per
pass rise from 20,755 to 35,233 (the fetch now runs per position), and the one model falls
from 2,043.2 M to 1,327.8 M dense. Digests: PPU `768a060c`, one model `a4521464`.

## PU shape: object pixels per row, the OAM scan's count, the fetcher's two bytes (`docs/hardware-shaped.md` §13.3)

Interleaved, three rounds each, `--weights all --one-model`, load 3 to 12. P1 against
the final tree: text 36.8, 36.9, 37.8 against 38.0, 38.2, 37.8 fps (median 36.9 to **38.0**);
overworld 94.4, 93.0, 97.1 against 95.5, 93.6, 96.8 (94.4 to **95.5**). The build before the PU work
against the final tree: text 38.5, 37.4, 39.5 against 37.6, 36.6, 38.0 (38.5 to
**37.6**); overworld 75.4, 76.0, 75.5 against 97.2, 96.7, 96.9 (75.5 to **96.9**); PPU
line 69.3 to 31.3 µs on text, 70.3 to 36.4 on overworld. PPU units per pass on overworld
(`cpu,memory,ppu`): P1 35,233, P2 and U3 50,309, the two bytes 42,687. The PPU checkpoint
is 338.8 M dense (from 922.2 M before the PU work), and the one model 1,259.1 M (from 2,057.9 M).
Digests: PPU `1f904a28`, one model `a5fdac3a`.

## PU shape: the high-pass filter as presentation (A4, `docs/hardware-shaped.md` §13.5)

A the build before A4, B A4, interleaved, three rounds, `--weights all --one-model`, load 5.0 to
5.8. Text: 38.2, 38.4, 38.6 against 39.6, 39.9, 40.0 fps (median 38.4 to **39.9**).
Overworld: 97.8, 98.0, 98.2 against 99.0, 100.0, 100.1 (98.0 to **100.0**). B won every
pair. Why a cheaper APU stream speeds the GBC CPU's thread was not profiled. The inferred
cause is shorter waits at FF10-FF3F reads and at the frame's audio. APU updates per
segment (`GBCFORMER_UNITS=1`, `--weights apu`): text 2,818 to 1,904, overworld 3,026 to
2,043 (estimated −758). The APU checkpoint goes from 14 to 13 layers and from 90.1 M to
80.2 M dense (150,077 to 135,817 nonzero), and its widest live boundary from 771 to 697.
The one model goes from 1,259.1 M to 1,235.4 M. Digests: APU `ebeeb39c` to `110ecc5d`,
one model `a5fdac3a` to `2d6085d7`, the rest unchanged. They moved because the capacitors
left the APU's state and circuit, following the project's decision that the filter belongs to
the board, not the chip. `run --wav` is byte-identical before and after, on the host and
in weights.

## Shape: the GBC CPU and PU work merged (`docs/hardware-shaped.md` §14)

The merge joins the GBC CPU work (`docs/hardware-shaped.md` §2.6) and the PU work (§13). The two
change disjoint circuits, so every per-part checkpoint equals its parent's. They
are cpu `6bd416c1` (115.8 M, 0.13 M nonzeros), cpu-memory `438b8783` (162.3 M, 0.23 M) as on
the GBC CPU work, apu `110ecc5d` (80.2 M, 0.29 M), control `254acbba` (50.0 M, 0.17 M) and
ppu `1f904a28` (338.8 M, 0.38 M) as in the PU work. Only the one model is new. It is
`52f41ae4`, 1,235.4 M dense, 0.97 M nonzeros (hidden 4,096, intermediate 4,544, 14 layers;
the PU work's `2d6085d7` has the same shape and 0.99 M nonzeros). Because cpu-memory and the
APU are byte-identical to the parents that recorded their tables, the merge re-records
neither `src/sm83/changes-memory.txt` nor `src/apu_circuit/changes.txt`. The f16 census
reads 0 violations for every checkpoint. The GGUF (`--layer-shapes --align 64`, 843.0 M
values, 1.69 GB) loads with 0 of 1,235,312,640 values differing (`digest --gguf`).
`cargo test --release` and `tools/gate.sh` pass in full, and `CORE=lockstep-mem` text and
overworld reach `870aa3fd` and `822059c1` with no divergence.

A is the GBC CPU work (one model `02b8ca6d`), B the PU work (`2d6085d7`) and C the merge
(`52f41ae4`). Each binary was rebuilt at its head, checked by its digest, and run with
its own GGUF through `GBCFORMER_GGUF`: `--weights all --one-model`, `GBCFORMER_STATS=1`,
interleaved A, B, C, three rounds, load 2.6 to 7.8.

| | A GBC CPU work | B PU work | C merged |
| - | - | - | - |
| Text fps (three rounds) | 42.6, 42.5, 42.3 | 40.0, 39.3, 40.0 | 42.8, 42.5, 42.6 |
| Text median | 42.5 | 40.0 | **42.6** |
| Overworld fps | 77.9, 76.5, 70.2 | 101.2, 100.0, 100.4 | 107.4, 106.4, 106.6 |
| Overworld median | 76.5 | 100.4 | **106.6** |
| GGUF load | 4.2 to 4.7 s | 2.3 to 2.4 s | 2.3 s |
| Units per fast pass, text: cpu, apu, control, ppu | 1,452, 2,816, 399, 163,947 | 2,087, 1,903, 399, 32,371 | 1,452, 1,903, 399, 32,371 |
| Units per fast pass, overworld | 1,335, 3,025, 389, 166,834 | 1,925, 2,044, 389, 42,687 | 1,335, 2,044, 389, 42,687 |
| PPU line, text, overworld | 68.8, 71.1 µs | 31.4, 36.5 µs | 30.4, 36.2 µs |
| CPU pass, text, overworld (round 2) | 4.38, 6.61 µs | 4.73, 5.05 µs | 4.38, 4.75 µs |

Each stream's update count is its parent's (round 2's files). The GBC CPU stream takes
the GBC CPU work's 1,452 and 1,335, and the APU and PPU streams take the PU work's. The merge is
faster than both parents on overworld (+6% over B, +39% over A) and matches A on text
(+0.2%, inside the run-to-run spread). The text GBC CPU pass equals A's, likely because
the GBC CPU's thread bounds text (inferred). On overworld the GBC CPU pass is 4.75 µs
against A's 6.61 and B's 5.05. The likely cause is less contention, from a cheaper PPU
stream than A's and the cheaper GBC CPU stream than B's (inferred, not profiled).

## Overlay unit 1: hidden in steps of 64 (`docs/hardware-shaped.md` §15.1)

`Model::shape` rounds hidden up to `HIDDEN_QUANTUM` = 64 instead of whole blocks of
1,024, and a channel plan reuses dead wires' channels whenever that narrows hidden. The
one model's channel fit (`one_model::circuit`) steps its target down 64 channels at a
time from the scheduled width until a placement no longer fits, in place of trying
blocks of 1,024. No circuit's levels change, so this unit re-records no change table. Only
one test fixture and the `one_model` example's projection assumed 1,024. The engine
indexes channels as u16, `int::norm_bound` takes the actual width (its 16·d bound holds
above 273 channels), the GGUF export writes any width, and Metal's f16 `mul_mm` asks for
an inner dimension in multiples of 32.

| Checkpoint | Hidden | Intermediate | Layers | Dense | Nonzero | Digest |
| - | - | - | - | - | - | - |
| cpu (host-fed) | 2,048 → 1,408 | 1,600 | 6 | 115.8 M → 79.6 M | 0.13 M | `6bd416c1` → `3c11d62b` |
| cpu-memory | 2,048 → 1,728 | 1,600 | 11 | 162.3 M → 137.0 M | 0.23 M | `438b8783` → `1645a346` |
| apu | 1,024 → 896 | 896 | 13 | 80.2 M → 70.2 M | 0.29 M | `110ecc5d` → `83c6b259` |
| control | 1,024 → 832 | 512 | 12 | 50.0 M → 40.6 M | 0.17 M | `254acbba` → `f2676ee8` |
| ppu | 2,048 → 1,536 | 2,752 | 14 | 338.8 M → 254.1 M | 0.38 M | `1f904a28` → `a1ee149e` |
| one model | 4,096 → 3,776 | 4,544 | 14 | 1,235.4 M → 1,138.9 M | 0.97 M | `52f41ae4` → `c21b3cac` |

The four parts' own checkpoints now sum to 501.9 M (from 631.3 M). The one model's
GGUF (`--layer-shapes --align 64`) holds 780.0 M values (from 843.0 M), 1.56 GB, and
loads with 0 of 1,138,803,840 values differing. `cargo test --release`, `tools/gate.sh`
(every line ok) and `CORE=lockstep-mem` text and overworld (`870aa3fd`, `822059c1`)
pass, and `examples/f16census all` reads 0 violations for every circuit.

A is the merged build with its GGUF, and B is this unit with its GGUF:
interleaved, three rounds, `GBCFORMER_STATS=1`, load 2.9 to 15.3 (other jobs).

| | A | B |
| - | - | - |
| Text, `--weights all --one-model` | 42.2, 41.0, 42.0 (median 42.0) | 41.3, 41.9, 41.8 (41.8) |
| Overworld, all | 101.6, 104.9, 100.8 (101.6) | 100.3, 103.3, 103.3 (103.3) |
| Text, `--weights cpu,memory --one-model` | 43.7, 43.9, 44.4 (43.9) | 43.1, 42.4, 42.9 (42.9) |
| Overworld, cpu,memory | 115.8, 115.3, 115.3 (115.3) | 112.7, 116.4, 116.2 (116.2) |
| GGUF load | 2.2 to 2.9 s | 2.2 to 2.4 s |

The engine's cost does not depend on hidden (`docs/architecture-census.md` §1), so no
fps change was expected, and the differences are inside the rounds' spread. The
single-part rows still compile and load the whole one model. Unit 3 addresses that cost.

## Overlay unit 2: parts share residual channels (`docs/hardware-shaped.md` §15.2)

Every part's wires take the channels its own plan gives them, overlaid on the other
parts'. Its units read its mode channel as `up`, its memory heads attend to a null token
outside its positions, and each part keeps its own levels. Hidden goes from 3,776 to
2,112 and intermediate from 4,544 to 5,440, with 14 layers, 22 heads and vocabulary
34,764. The one model goes from **1,138.9 M to 716.5 M dense**, 0.97 M nonzeros, digest
`c21b3cac` to `e4615c82`, and every per-part digest is unchanged. The GGUF
(`--layer-shapes --align 64`) holds 432.2 M values and loads with 0 of 716,441,088
differing. `cargo test --release` (the f16 census over each row's own part), the
ignored `the_apu_and_control_streams_equal_the_merged_dense_pass` (3 passes each, 184 s),
`tools/gate.sh` (every line ok, Yellow 300 as one model `cfd31f3d`) and
`CORE=lockstep-mem` text and overworld pass, and `f16census all` finds 0 violations.

A is the merged build with its GGUF, and B is this unit with its GGUF: interleaved, three rounds,
load 4.8 to 9.1.

| | A | B |
| - | - | - |
| Text, `--weights all --one-model` | 41.5, 42.2, 42.0 (median 42.0) | 42.0, 42.0, 42.0 (42.0) |
| Overworld, all | 104.8, 104.7, 104.5 (104.7) | 104.4, 104.8, 104.5 (104.5) |
| Text, `--weights cpu,memory --one-model` | 43.8, 44.0, 44.2 (44.0) | 43.8, 42.9, 43.8 (43.8) |
| Overworld, cpu,memory | 116.1, 117.4, 115.8 (116.1) | 115.4, 115.7, 116.4 (115.7) |
| GGUF load | 2.5 to 2.7 s | 1.2 to 1.3 s |
| CPU pass, text | 4.41 to 4.48 µs | 4.43 µs |

The first build ran text at 33.1 fps (5.63 µs a GBC CPU pass). Its mask tokens carried
the mode channel, so the engine's owner tables tracked none of them, and every pass
decoded 72 positions instead of 14.5 (`GBCFORMER_STATS=1`). The stream's network now
clears the unread mode and null channels from its embedding, and the row above is that
build. The GBC CPU stream reads 217.1 (changed channel, reading layer) pairs a pass
against 209.8, because its own levels replace the merged fit's (inferred cause). The
difference is inside the spread.

## Overlay unit 3: each stream runs its own prefix (`docs/hardware-shaped.md` §15.3)

An overlay's layout puts the wires last, routes each part at its own depth, and keeps
each part's units and heads contiguous. Every stream's engine plans and falls back on its
own network (`Model::stream`), which `gguf::load_slice` reads from the GGUF, and a run
builds only the streams it selects. The one model keeps its size (716.5 M, hidden 2,112,
intermediate 5,440, 14 layers, 22 heads, 0.97 M nonzeros). Its digest goes from
`e4615c82` to `6a17e689`, and every per-part digest is unchanged. The GGUF holds 432.2 M
values, 0 of 716,441,088 differing. `cargo test --release`, the ignored
`every_slice_is_the_checkpoint_at_its_positions` (273 s), `tools/gate.sh` (every line ok)
and `CORE=lockstep-mem` text and overworld pass, and `f16census all` finds 0 violations.

Each stream's network: GBC CPU hidden 2,112, 11 layers, 279.2 M; PPU 1,792, 14 layers,
399.2 M; APU 1,536, 13 layers, 225.7 M; control 1,472, 12 layers, 166.8 M. A run reads
only its streams' slices from the 865.5 MB file: the GBC CPU's 460.8 MB (79 tensors, 0.4
to 0.8 s), the PPU's 433.7 MB, the APU's 327.6 MB.

A is the merged build with its GGUF, and B is this unit with its GGUF: interleaved, three rounds,
load 2.0 to 3.9.

| | A | B |
| - | - | - |
| Text, `--weights all --one-model` | 42.1, 41.9, 40.6 (median 41.9) | 42.3, 42.4, 41.3 (42.3) |
| Overworld, all | 106.0, 105.4, 103.6 (105.4) | 108.8, 108.5, 105.0 (108.5) |
| Text, `--weights cpu,memory --one-model` | 42.5, 42.9, 44.4 (42.9) | 42.8, 43.1, 44.2 (43.1) |
| Overworld, cpu,memory | 113.1, 112.2, 113.3 (113.1) | 116.4, 118.3, 118.1 (118.1) |
| CPU pass, overworld | 4.77 to 4.88 µs | 4.64 to 4.81 µs |
| Weights load | 2.2 to 2.7 s (whole file) | 0.3 to 0.8 s per stream (its slice) |

Overworld gains 3% with every component in weights and 4% with the GBC CPU alone, and the
GBC CPU pass is 2.7% to 4.5% cheaper there. The plans are the same as unit 2's, so the
likely cause is the engines' smaller dense networks in memory (inferred, not profiled).
Text moves inside the spread.

After the unit, the slices were measured as dense oracles and mixed subsets as one model.
`GBCFORMER_ORACLE=1000` ran on Yellow 300 with every component in weights, in lockstep,
as one model, with weights from the GGUF. The GBC CPU stream's network ran 166 dense
oracle checks over the 141,398-cell cache, and the PPU stream's network checked 1 palette read
through its reader record (its reader routing kept at the GBC CPU's layer 10, its own
routing at 13). All were equal. `GBCFORMER_ORACLE` now reaches the PPU checkpoint too
(`Stack::build_ppu`), which it never did before. `--one-model` with `cpu,memory,ppu`,
`cpu,memory,apu` (Yellow 300, `cfd31f3d`) and `cpu,memory,ppu,control` (both acid2 images
pixel-exact) pass in lockstep.

## Prototype: byte-valued channels on the host-fed GBC CPU (`docs/byte-channels.md`)

The first nibble prototype, behind `GBCFORMER_D4=1`; the default checkpoint is unchanged (`6bd416c1`).
`GBCFORMER_STATS=1 CORE=model tools/bench/text.sh`, one binary, base and byte runs
interleaved three rounds, load 6 to 11 from other jobs. Frame hash `870aa3fd`
on every run, 0 dense fallback passes.

| | 0/1 circuit | Byte channels |
| - | - | - |
| Text fps | 84.4, 85.2, 85.0 (median 85.0) | 122.5, 121.6, 120.3 (median **121.6**) |
| µs per pass | 1.78, 1.76, 1.76 | 1.22, 1.23, 1.25 |
| First pass of a step, µs | 1.86, 1.84, 1.85 | 1.19, 1.20, 1.22 |
| Shared layers, µs | 1.18, 1.17, 1.17 | 0.64, 0.64, 0.66 |
| Units per fast pass (first pass, second) | 502 (616, 42) | 166 (202, 18) |
| (changed channel, reading layer) pairs | 95.8 | 55.4 |

Layer 0 (0.28 µs) and decode (0.08 µs) are equal on both sides, so the shared layers'
walk is the whole difference. The branch keeps this as a prototype, and the doc has the
projection for the GBC CPU with memory.

## d4 on the GBC CPU with memory: nibble channels, timing runs (`docs/byte-channels.md`)

The memory circuit's nibble version, behind `GBCFORMER_D4=1`, carries each byte as two nibble
channels (0..15). The GBC CPU with memory goes from 9,517 to 6,489 units and from 162.3 M
to 149.4 M dense, with 98,368 nonzeros, depth 10 (11 layers). The one model with every
part (`export one-model --layer-shapes --align 64` under `GBCFORMER_D4=1`) is 1,202.4 M
dense, 0.92 M nonzeros, digest `d259dcc9`, and its GGUF holds 797.3 M values and loads
with 0 of 1,202,282,496 differing. Census runs (`GBCFORMER_UNITS=1`, `CORE=model-mem`,
slower than timing runs): text went from 1,476 to 1,137 units a pass, and overworld from
1,360 to 1,028.

Timing runs: A is the merged build with its GGUF, and B is the nibble memory circuit with
`GBCFORMER_D4=1` and its GGUF: `GBCFORMER_STATS=1`, interleaved, three rounds, load 1.4
to 3.4. In memory mode a step is one pass except for the 0.5% the host serves (3,224,368
passes for 3,208,968 steps on text), so µs per pass equals µs per step to within 0.5%.

| | A | B |
| - | - | - |
| Text, `--weights all --one-model`, fps | 39.7, 39.5, 42.8 (median 39.7) | 47.0, 47.2, 47.5 (**47.2**) |
| Text, CPU µs per pass | 4.69, 4.72, 4.35 (4.69) | 3.96, 3.94, 3.92 (**3.94**) |
| Overworld, all, fps | 109.8, 109.2, 108.9 (109.2) | 118.0, 120.5, 120.2 (120.2) |
| Overworld, CPU µs per pass | 4.60, 4.63, 4.64 (4.63) | 4.28, 4.19, 4.21 (4.21) |
| Text, `--weights cpu,memory --one-model`, fps | 44.9, 45.2, 44.8 (44.9) | 49.8, 49.7, 50.0 (49.8) |
| Text, cpu,memory, µs per pass | 4.14, 4.11, 4.15 (4.14) | 3.74, 3.74, 3.72 (3.74) |
| Overworld, cpu,memory, fps | 120.4, 119.7, 119.6 (119.7) | 133.3, 132.6, 133.1 (133.1) |
| Overworld, cpu,memory, µs per pass | 4.20, 4.22, 4.22 (4.22) | 3.79, 3.81, 3.80 (3.80) |
| Weights load (whole file) | 1.9 to 2.1 s | 1.9 s |

Text gains 19% with every component in weights and 11% with the GBC CPU alone. The
GBC CPU pass is 16% cheaper (4.69 to 3.94 µs) with every component and 10% cheaper
(4.14 to 3.74) alone. Every run reaches frame hashes `870aa3fd` and `822059c1`. The text step is
still above the 3.1 µs that 60 fps needs. On B's text runs the GBC CPU's shared layers
take 2.94 µs of a 3.36 µs pass, layer 0 0.21 µs and decode 0.17 µs (`GBCFORMER_STATS=1`).

### Unit 3: the opcode decode as nibble values, measured and not kept

In the GBC CPU with memory, each nibble one-hot picked and matched its nibble in one
node, with one step per line byte, so a changed line bit reached 16 steps and a moved
word line 32. The unit picks each nibble's value with one ramp per line byte (`value_mux`
gated by the word lines) and makes each one-hot `[value == n]` a level later. Circuit and
checkpoint pass SST 500,000 of 500,000, the engine equals the dense pass,
`CORE=lockstep-mem` text and overworld reach `870aa3fd` and `822059c1`, and the f16
census finds 0 violations. Depth goes from 10 to 11 (12 layers) and intermediate from
1,408 to 1,280, at 150.4 M dense and 97,068 nonzeros.

On `CORE=model-mem` text (census mode) the `OP0` family fell from 288 to 127 updates a
pass and the units from 1,132 to 971. The (changed channel, reading layer) pairs rose
from 196.6 to 200.3, though, and in timing runs the GBC CPU pass got slower. A is
the nibble memory circuit without the unit and B is with it, in one binary (the unit behind a scratch
switch), each with its own GGUF: every component in weights as one model, interleaved,
three rounds, load 2.2 to 6.3.

| | A | B |
| - | - | - |
| Text fps | 46.9, 47.4, 46.9 (median 46.9) | 45.6, 45.7, 45.8 (45.7) |
| Text, CPU µs per pass | 3.97, 3.93, 3.97 (3.97) | 4.08, 4.07, 4.07 (4.07) |
| Overworld fps | 119.2, 118.7, 118.9 (118.9) | 115.8, 116.3, 115.9 (115.9) |
| Overworld, CPU µs per pass | 4.24, 4.26, 4.25 (4.25) | 4.36, 4.34, 4.36 (4.36) |

A first A/B against the nibble memory circuit's own binary gave the same order (text 3.94 against 4.04 µs).
With every component the one model's pairs rose more, from 190.7 to 211.1 a pass, and
its shared layers took 2.95 against 3.07 µs. The 162 fewer units saved less than the 20
more pairs and the extra layer cost (inferred from the stage timings; at about 9 ns a
pair, the estimate in "The measurement first" above, the pairs alone account for
0.18 µs). Re-recording the wire-change table for both decodes (`GBCFORMER_WIRE_CHANGES`,
averaged over text and overworld) changed no level and no pair, because the fit places
d4's nibble wires the same either way. The unit is reverted.

## d4 the default, on the prefix overlay (`docs/byte-channels.md`)

The nibble merge puts the nibble memory circuit onto the overlay and then makes the nibble
circuits the default (`GBCFORMER_BITS=1` selects the 0/1 ones). The merge needed two
re-derivations, both of bounds that assumed every channel in [−1, 1]. First, the norm's
bound now sets a floor on hidden (`llama::norm_squares`). The host-fed checkpoint goes
from its used channels rounded to 64, 1,088, to 1,600, and the memory checkpoint from
1,536 to 1,856. Second, `foreign_score` reads each channel's occupants' ranges, so the
null term grows to 9 dims of 30,720 against a null key of 4 (from 6 dims of 31,744
against 1). The f16 census checks an overlay head's query rows over their own part's
ranges, as it already checked an overlay unit's gate. The wire-change table was
re-recorded on the nibble circuit (text and overworld averaged), and no digest moved.

| Checkpoint | Dense | Nonzero | Digest |
| - | - | - | - |
| cpu (host-fed) | 79.6 M → 70.4 M (hidden 1,600, 7 layers) | 0.13 M → 31,735 | `3c11d62b` → `87bb33f8` |
| cpu-memory | 137.0 M → 135.3 M (hidden 1,856, 11 layers) | 0.23 M → 98,368 | `1645a346` → `6ee28082` |
| one model | 716.5 M → **634.6 M** (hidden 2,112 → 1,856) | 0.97 M → 0.92 M | `6a17e689` → `e872e6c9` |

apu `83c6b259`, control `f2676ee8` and ppu `a1ee149e` are unchanged, and with
`GBCFORMER_BITS=1` every digest is the overlay's. The one model's GGUF (`--layer-shapes
--align 64`) holds 360.6 M values (from 432.2 M), 0 of 634,588,672 differing. The GBC CPU
stream's slice is hidden 1,856, 233.6 M dense (from 2,112 and 279.2 M).

A is the overlay with its GGUF, and B is this work with its GGUF: `GBCFORMER_STATS=1`,
interleaved, three rounds, load 1.8 to 20.7 (other jobs; round 2 of text ran at
15.7 to 20.7).

| | A | B |
| - | - | - |
| Text, `--weights all --one-model`, fps | 43.6, 44.2, 42.5 (median 43.6) | 47.6, 48.5, 47.9 (**47.9**) |
| Text, CPU µs per pass | 4.27, 4.21, 4.38 (4.27) | 3.91, 3.83, 3.89 (**3.89**) |
| Overworld, all, fps | 110.3, 112.7, 108.7 (110.3) | 122.5, 123.5, 120.0 (122.5) |
| Overworld, CPU µs per pass | 4.58, 4.48, 4.65 (4.58) | 4.12, 4.09, 4.21 (4.12) |
| Text, `--weights cpu,memory --one-model`, fps | 43.9, 45.8, 45.6 (45.6) | 49.8, 51.2, 51.0 (51.0) |
| Text, cpu,memory, µs per pass | 4.24, 4.06, 4.08 (4.08) | 3.74, 3.63, 3.65 (3.65) |
| Overworld, cpu,memory, fps | 122.0, 122.0, 122.2 (122.0) | 136.7, 135.8, 131.4 (135.8) |
| Overworld, cpu,memory, µs per pass | 4.14, 4.14, 4.13 (4.14) | 3.70, 3.72, 3.85 (3.72) |
| Weights load, per stream slice | 0.2 to 0.7 s | 0.2 to 0.5 s |

Text gains 10% with every component in weights and 12% with the GBC CPU alone, and
overworld gains 11%. The two sides' ranges do not overlap on any row. The GBC CPU stream
re-evaluates 1,118 units a pass against 1,459 and walks 196.6 pairs against 217.1, and
its shared layers take 2.84 against 3.29 µs (text, round 2). Every run reaches frame
hashes `870aa3fd` and `822059c1`. Text still needs 3.1 µs a step for 60 fps, and the step
is 3.89.

## Engine: the walk's pair shapes in their own buckets

**The measurement first.** This is a `sample` of the GBC CPU's thread on text.sh
(`--weights all --one-model`, weights from the GGUF, against the base). The engine's pass
(`fill_into`) is 85% of the thread. The 0.58 µs a step outside it splits into the write
records' `Cells::put` (5%, of which 3.3% builds the rows of the 13.6% of records the full
65,536-entry memo misses), the memory core's own decode, `set_inputs` and evidence loops
(4%), the machine (3.3%, with the control checkpoint's passes at 1.9% and the PPU and APU
hand-offs at 1.5%) and, only under `GBCFORMER_STATS=1`, the stage clocks (3 to 4%). Text
runs 45.5, 45.7, 45.1 fps with the clocks and 46.7, 47.8, 48.2 without (interleaved).
Inside the pass, the gate walk (`Marks::blocks`) takes 46%, `drain_wires` 12%,
`memory_reads` 8% (inlined into `advance`, with 49.9 bit flips of the heads' lines a
pass at about 4.5 ns each, and 46 ns for the head bucket), `route_layer` 6% and layer 0
3%.

Per-layer counters (a scratch build reading `cntvct_el0` around each stage) show layer 1
walking 2.6 pairs and 43 updates in 79 ns, and layer 9 walking 16 pairs and 35 updates in
116 ns. A pair costs far more than its updates. By shape, a text pass's 196.6 pairs are
59.6 with one lone plain entry, 4.9 with one lone pair, about 43 with one block and no
lone entry (one to nine vectors), and the rest mixed. Every pair paid for the head lookup
and three loops whose trip counts change with every change.

**Single bucket** (H4). A channel whose readers in a circuit layer are one gate entry
routes its changes to that layer's single bucket (`Template::single_bucket`, reader bit
`n + 3 + li`). `Blocks::single` stores the entry as one word (unit, pair flag, weight),
and the walk updates it with no head, block or lone loop, staging its send with the
others. The census path walks the bucket with the scalar `layer`.
`block_walk_sends_what_the_scalar_walk_sends` now gives the random layers single-reader
channels and checks the bucket's sends and gates against the scalar walk's (it fails with
the pair flag inverted).

| text.sh, `--weights all --one-model`, `GBCFORMER_STATS=1` | fps | CPU engine pass, µs |
| - | - | - |
| A, the base (load 3.6 to 7.8) | 48.2, 47.1, 48.0 (median 48.0) | 3.28, 3.36, 3.29 |
| B single bucket | 50.5, 50.5, 50.7 (**50.5**) | 3.10, 3.10, 3.08 |

The pass walks the same 196.6 pairs and 1,118 updates, so the walk's shape alone saves
0.19 µs. Every run reaches frame hash `870aa3fd`, and lockstep-mem DMG 8 reaches
`6872b003`. No weight changed.

**Block bucket** (H5). A channel whose readers in a layer are exactly one block and no
lone entry routes to the layer's block bucket (`Template::block_bucket`, bit
`2n + 3 + li`; `Readers` and the bucket masks widen to u64 for the three buckets a
layer). `Blocks::one_block` stores the block's index and the channel's entry count, and
the walk runs the block with no head and no lone loops.

| text.sh, as above | fps | CPU engine pass, µs |
| - | - | - |
| A single bucket (load 3.1 to 10.0) | 50.2, 50.6, 50.5 (median 50.5) | 3.12, 3.10, 3.10 |
| B block bucket | 51.8, 52.1, 51.9 (**51.9**) | 3.01, 2.99, 3.00 |

**Against the base.** A is the base and B this commit, interleaved, three rounds,
`GBCFORMER_STATS=1` unless noted, load 3.4 to 11.8 from other jobs. A's round 3
on text and round 2 on overworld ran under a load spike.

| | A | B |
| - | - | - |
| Text fps | 47.9, 48.0, 34.6 (median 47.9) | 51.6, 52.0, 49.0 (**51.6**) |
| Text, CPU µs per pass | 3.89, 3.88, 5.38 (3.89) | 3.61, 3.58, 3.79 (**3.61**) |
| Text, CPU engine pass, µs | 3.30, 3.29, 4.27 | 3.01, 2.99, 3.19 |
| Overworld fps | 114.1, 67.5, 115.5 (114.1) | 118.7, 120.3, 121.9 (**120.3**) |
| Overworld, CPU µs per pass | 4.43, 7.48, 4.37 (4.43) | 4.26, 4.20, 4.15 (**4.20**) |

Every run reaches frame hashes `870aa3fd` and `822059c1`. A's medians match the base
measured at load 2.5 to 3 (48.3 fps, 3.26 µs engine, H3's A side), so the spikes did not
set them. Text still needs 3.1 µs a step for 60 fps. The step is 3.61 with the clocks and
about 3.4 without them, since the clocks cost about 0.2 µs (base text runs 45.5, 45.7,
45.1 fps with them and 46.7, 47.8, 48.2 without, interleaved). The kept changes moved no
pair and no update. The per-pair cost came from the walk's loop shapes (mispredicts), and
a pair now costs about 2 ns, not the 9 estimated before the block walk.

**Tried and not kept.** Each change ran interleaved against its predecessor, three rounds.

| Change | Before → after | Why |
| - | - | - |
| The block walk's sends read one packed (out, c) word per unit | text 47.6 → 47.8 fps, engine 3.33 → 3.32 µs | inside noise: a send's two loads are not its cost |
| Circuit: OP1 and OP2 gated to the opcode's length (`OP1g`, `OP2g` at level 5, an AND on the length sums) for the fields, operands and fetched-byte evidence; addresses read the raw bytes | quiet (load 2.5 to 3): 48.3 → 48.6 fps, engine 3.26 → 3.24 µs; pairs 196.6 → 186.2, updates 1,118 → 1,112, depth 10 | 10.4 fewer pairs bought 0.02 µs: about 2 ns a pair now, against the 9 estimated before the block walk. Not worth a digest. Re-recording the change table moved no placement |
| `Buckets::route` stores the first two reader bits branch-free (a spare cell takes the unkept store) | 51.9 → 49.9 fps | two stores and a dependent end update per change cost more than the loop exit they remove |
| The single bucket gathers its table entries before moving the gates | 51.6 → 51.5 | the single walk is not bound by its table loads |
| A lone bucket: two or more lone entries and no block, walked as one run | 51.6 → 51.9, load 7 to 34 | inside noise; about 20 pairs a pass in that class |

**What limits the step now.** These shares come from the per-layer counters above and
`sample`. The walk is still about 45% of the GBC CPU's thread, and its general bucket
keeps the ~90 pairs a pass whose shapes vary (several blocks, blocks with lone entries,
lone runs). `drain_wires` takes 11%, about 3.5 ns for each of ~130 changed channels. The
memory heads' line flips (~50 a pass, each a routed change) and the head bucket take
0.3 µs, and routing 6%. Layer 0, the decode and the memory core's per-position loops take
about 0.45 µs, the write records 0.2 µs, and the machine and the control pass 0.12 µs. No
single piece outside the walk reaches the 0.5 µs that text needs. The remaining options
are fewer changed channels a step (the fetch heads' 31 line-bit flips a pass, with three
heads reading mostly the same line) and the walk's remaining shapes.

## Engine: a channel's block vectors as one run

**The measurement first** (on the base, text, every component in weights, one model,
GGUF). Without `GBCFORMER_STATS` text runs 53.2 fps, about 3.50 µs a step against the
3.1 that 60 fps needs. A `sample` of the GBC CPU's thread without the clocks gives the
vector kernel `block` 21% (the block bucket 11%, the general bucket 10%), the general
bucket's lone loops 5%, the send loop 7%, `drain_wires` 11%, `memory_reads` 9%,
`route_layer` 7%, the write records 4.5% (their row builds 2.6%), and the machine and the
control pass 4%. Instruments' CPU Counters template (`xctrace`, 5 s attached) splits the
thread's issue slots into useful 55%, backend stalls 24%, discarded (mispredicted) 17%
and frontend 4%. A scratch census of the walk's shapes finds that the general bucket
takes 50.7 changes a pass, 40% with no block, 25% with one and 25% with two, with 2.5
lone plain and 0.4 lone pair entries each. The block bucket takes 44.6, of 1 (28%), 2
(24%), 4 (30%), 8 (6%) or 13 (4%) vectors.

**The change.** The general bucket walked each changed channel's blocks, then each
block's vectors, in two loops whose trip counts change with every change.
`Blocks::vectors` stores every block's vectors in order, so a channel's vectors form one
run that one loop walks (`Blocks::flat`). `IntLayer::with_blocks` checks the run against
the blocks. No weight, pair or update changed.

A is the base and B this commit, interleaved, three rounds, with each run started below
load 5 (load 2.9 to 3.9).

| | A | B |
| - | - | - |
| Text fps, no stats clocks | 52.7, 52.6, 52.9 (median 52.7) | 52.5, 53.3, 53.8 (**53.3**) |
| Text fps, `GBCFORMER_STATS=1` | 51.4, 51.3, 51.6 (51.4) | 51.9, 51.9, 52.0 (51.9) |
| Text, CPU µs per pass | 3.62, 3.62, 3.61 (3.62) | 3.59, 3.58, 3.58 (**3.58**) |
| Text, CPU engine pass, µs | 3.04, 3.05, 3.04 | 3.01, 3.01, 3.00 |
| Overworld fps, no stats clocks | 130.9, 131.2, 131.2 (131.2) | 131.5, 132.0, 132.2 (132.0) |
| Overworld fps, `GBCFORMER_STATS=1` | 127.9, 127.8, 125.4 (127.8) | 128.5, 125.7, 130.5 (128.5) |
| Overworld, CPU µs per pass | 3.95, 3.95, 4.03 (3.95) | 3.93, 4.02, 3.87 (3.93) |

Every run reaches frame hashes `870aa3fd` and `822059c1`, and `cargo test --release` and
`tools/gate.sh` are all ok. Text gains 1.1% without the clocks, and 53.3 fps is about
3.49 µs a step.

**Tried and not kept.** Each change ran interleaved against its predecessor, three rounds.

| Change | Before → after | Why |
| - | - | - |
| The lone plain and pair entries in one loop, the activation picked by the unit's pair mask | text 53.8 → 53.1 fps (no clocks), 52.0 → 51.3 | the mask load and both formulas per entry cost more than the pair loop's exit (0 or 1 pair in 99% of changes) |
| Every activation as (g + 1) clamped to [0, cap] in the vector kernel and one lone loop | 53.0 → 51.3 (no clocks), 49.7 → 49.0 | the min after the max lengthens each lane's chain to the send mask; the shift and select ran beside it |
| The kernel counts its kept lanes by a 16-entry table, not `count_ones` | 52.1 → 51.8 (no clocks), 51.6 → 52.1 | noise: the count is off the loop-carried path |
| The block walk's send loop pushes a channel when its sum is 0, with no mark | 54.6 → 54.4 (no clocks), 52.9 → 52.8 | no change: the mark's load and store are not on the critical path |
| The write-record memo at 2^18 entries | 138,955 → 135,437 rows built | the misses are new (address, line) pairs, not capacity: no size or key of a per-record memo removes them |
| Circuit: even and odd fetch heads over a 16-byte window (analysed on the census, not built) | | the opcode decode reads both lines, so every flip of either head still reaches the 16 nibble steps; only the OP1 and OP2 heads' single-entry flips (20 a pass) go, while each pick's word lines double |

**What limits the step now.** The kernel variants show that the vector update is bound
by its lanes' latency to the send mask, not by its instruction count. The remaining loop
shapes (the general bucket's vectors and two lone loops, the block bucket's 1, 2 or 4
vectors) cost what removing them adds back. The walk is about half the engine pass.
`drain_wires` takes 11%, `memory_reads` 9%, routing 7%, layer 0 and the decode about 10%,
and the write records 4.5% (compulsory builds). Each is under the 11% that 60 fps on text
still needs.

## Circuit: ports at rest, output enable, one record a line, the line-timing schedule

The fetch round tuned the engine to its limit, so gains now need fewer changed channels or
fewer passes. This work makes four hardware-shaped changes. Each ran interleaved
against its predecessor, three rounds, every run started below load 5, text and overworld
with every component in weights as one model, and weights from each side's own GGUF.
Every run reaches frame hashes `870aa3fd` and `822059c1`. The decision log was kept
locally and is not in the repository.

**Two costs measured on the way.** They set what a circuit change must beat.

- A wire change in the shared layers costs about 9 ns. In the held-address trial below,
  10.3 fewer head line-bit changes a pass cut the shared layers by 0.09 µs.
- A moved input position costs about 70 ns in layer 0. In the same trial, 0.7 more moved
  positions a pass added 0.05 µs to layer 0. A change that adds state as input slots pays
  this every time the state moves.

**H1, ports at rest** (`docs/hardware-shaped.md` §16). A reference-core replay of the
two workloads reproduced the recorded head table to 0.01 and ranked the idle policies.
Held addresses (RP0 and WP0 carried back as six input slots) removed the most head
changes but lost on the layer-0 cost above. Text was flat (54.4 to 54.4 fps without
clocks), overworld lost 1.2%, the GBC CPU checkpoint grew from 135.3 to 146.3 M dense, and
the one model's hidden from 1,856 to 1,920. That policy was rejected. The kept policy rests
write 0 on VOID and write 1 on SP−2, and gates each write's word lines by its enable. Wire
changes on text went from 106.2 to 102.5 a pass and pairs from 196.6 to 191.0, with no
shape change except 96 nonzeros. The change table was re-recorded, and it moved no
placement.

| H1 (A the base, B kept) | A | B |
| - | - | - |
| Text fps, no stats clocks | 54.4, 54.3, 54.4 (54.4) | 54.9, 54.9, 54.9 (**54.9**) |
| Text fps, `GBCFORMER_STATS=1` | 52.1, 52.9, 51.2 (52.1) | 53.4, 53.8, 53.0 (53.4) |
| Text, CPU µs per pass (engine) | 3.57, 3.52, 3.64 (2.93 to 3.06) | 3.49, 3.46, 3.51 (2.88 to 2.95) |
| Overworld fps, no stats clocks | 133.8, 133.7, 134.2 (133.8) | 132.8, 133.0, 133.0 (133.0) |
| Overworld fps, stats | 131.0, 131.2, 130.9 (131.0) | 131.1, 130.5, 130.5 (130.5) |
| Overworld, CPU µs per pass (engine) | 3.86 (2.89, 2.90) | 3.86, 3.87 (2.88) |

Overworld's GBC CPU pass is equal and its engine 0.01 µs faster, so the 0.6% loss comes
from outside the GBC CPU stream.

**H2, output enable.** The result slots that only the audit reads (`OP0' OP1' OP2'
RP1a..c RV0 RV1`) leave every pass when neither the GBC CPU nor the memory runs in
lockstep (`MemCheckpointCore::without_audit_slots`). They keep the mask token, so the
sequence the model sees is the full decode's. A pass decodes 64 positions in place of
72. No weight changes. The step's three opcode reads stay, because the memory core uses
them for the host's fetch records and the audit.

| H2 | A (H1) | B |
| - | - | - |
| Text fps, no stats clocks | 55.2, 55.3, 55.4 (55.3) | 56.0, 55.0, 55.9 (**55.9**) |
| Text fps, stats | 53.6, 53.4, 53.3 (53.4) | 54.2, 54.3, 54.3 (54.3) |
| Text, CPU µs per pass | 3.47, 3.49, 3.49 | 3.43, 3.43, 3.43 |
| Overworld fps, no stats clocks | 132.8, 132.7, 134.3 (132.8) | 133.8, 135.5, 136.0 (135.5) |
| Overworld fps, stats | 130.5, 131.2, 131.4 (131.2) | 131.8, 132.8, 133.3 (132.8) |

**H5, one record a line** (`docs/hardware-shaped.md` §16.6). Write 0 keeps no record
when write 1 shares its line. Text keeps 877,196 records where it kept 1,024,869.

| H5 (load 2.8 to 4.6) | A (H2) | B |
| - | - | - |
| Text fps, no stats clocks | 54.5, 55.7, 54.3 (54.5) | 56.1, 55.9, 55.5 (**55.9**) |
| Text fps, stats | 53.1, 53.6, 52.7 (53.1) | 53.6, 53.6, 53.9 (53.6) |
| Text, CPU µs per pass | 3.50, 3.47, 3.53 | 3.47, 3.47, 3.45 |
| Overworld fps, no stats clocks | 134.5, 132.2, 128.9 (132.2) | 132.9, 130.9, 128.5 (130.9) |
| Overworld fps, stats | 123.4, 129.3, 126.0 (126.0) | 126.5, 132.1, 132.3 (132.1) |

Overworld is inside the load's noise both ways.

**H4, the line-timing schedule** (`docs/control-in-weights.md` §19), adopted by the
project. Where only the PPU's mode-3 starts can need a control pass before line 144, a pass
schedules up to eight of them, and the host counts down to each and calls the renderer
with the line the model named. Control on text goes from 186.5 passes a frame at
2.19 µs to 58.4 at 3.72 µs (128.1 lines a frame drawn at a scheduled start), and from
0.41 to 0.22 ms a frame on the GBC CPU's thread. The control checkpoint keeps depth 11
and 12 layers. A first cut that made the horizon the schedule's end added two levels, so
the host instead reads the next pass as the start after the last scheduled one. Hidden
goes from 832 to 896, and the checkpoint from 40.6 to 43.8 M dense.

| H4 | A (H5) | B |
| - | - | - |
| Text fps, no stats clocks | 55.7, 55.7, 55.6 (55.7) | 56.3, 56.2, 56.4 (**56.3**) |
| Text fps, stats | 54.0, 53.1, 53.4 (53.4) | 54.4, 54.4, 53.9 (54.4) |
| Text, CPU µs per pass | 3.44, 3.50, 3.48 | 3.42, 3.42, 3.45 |
| Overworld fps, no stats clocks | 135.4, 134.8, 135.6 (135.4) | 135.5, 122.0, 134.4 (134.4) |
| Overworld fps, stats | 133.6, 133.1, 131.9 (133.1) | 133.6, 134.1, 133.9 (133.9) |

**Where the branch ends.** Each median above is against its own predecessor in one
sitting. The sittings drifted by about a frame a second (the same base binary ran text at
54.4 and 55.3), so the gains do not simply add. In the last sitting text runs **56.3 fps
without the clocks** (about 3.30 µs a step against the 3.1 that 60 fps needs) and 54.4
with them (3.42 µs a pass, engine 2.92), and overworld runs 134.4 and 133.9. A `sample`
of the GBC CPU's thread at the end (text, every component, one model) gives the gate walk
(`Marks::blocks`) 47%, `advance` with the memory heads 13%, `drain_wires` 12%, routing
7%, layer 0 4%, the decode 4%, the write records 3%, and the step's three opcode reads
0.25% (kept, because the host's fetch records and the audit use them). The step is the
engine's walk over about 190 changed (channel, layer) pairs and 1,100 updates. At about
9 ns a changed wire, 60 fps needs some 20 fewer changed channels a pass than today's
~100, and most of those are the fetch path (the three fetch heads' 31 line-bit flips) and
the opcode decode.

**Digests** (`examples/digest`). cpu-memory goes from `6ee28082` to `b74a665b` (H1)
to `9c89f0dc` (H5; 135.3 M dense, 98,368 to 98,472 nonzeros).
control goes from `f2676ee8` to `1ff09cda` (H4, 40.6 to 43.8 M). The one model goes from
`e872e6c9` to `5aa84cb2` (H1) to `05d800f6` (H5) to `bb85df19` (H4, 634.6 to 634.7 M).
cpu `87bb33f8`, apu `83c6b259` and ppu `a1ee149e` are unchanged. With
`GBCFORMER_BITS=1`, cpu-memory goes from `1645a346` to `88fdb25e`, and the one model from
`6a17e689` to `52b4cf60`. The one model's GGUF (`export one-model --layer-shapes --align 64`,
kept locally) holds 363.4 M values, 0 of 634,633,216 differing
(`digest --gguf`), and `f16census all` finds 0 violations. Both change tables were
re-recorded (text and overworld averaged, `CORE=model-mem`).

**Gates on the head.** `tools/gate.sh` is all ok (30 checks: the lockstep hashes, Crystal,
Zelda DX and Mario DX 1,500 frames, every phase gate, Yellow 300 with every component as
one model), and `cargo test --release` passes. The commits before the head were built
with `cargo check` and share the head's gate, and each has its own A/B above. Also
checked: Yellow CGB 300 with every component as one model and no lockstep (the path H2
takes) reaches `cfd31f3d`, and `GBCFORMER_ORACLE=1000` on 30 DMG frames of cpu,memory
passes (26 dense checks, `6872b003`).

**Not built: H3, byte-wide RAM writes.** After H1 the write-line heads still flip 7.9
times a pass on text (`WL0h` 6.71, `WL1h` 1.16), above the 3 below which H3 would not be
worth building. Holding the last line would have reached 3.75, but it costs a moved input
slot. Byte cells for RAM are the write path's remaining structural option, and they are a
large change (two key granularities, a bigger cache, every digest), so they wait for a
ruling.

**Against the base.** A is the base and B the round's last commit, interleaved, three
rounds, each run started below load 5, load 3.0 to 4.8 from other jobs.

| | A | B |
| - | - | - |
| Text fps, no stats clocks | 53.0, 53.1, 53.3 (53.1) | 55.4, 54.6, 56.3 (**55.4**) |
| Text fps, `GBCFORMER_STATS=1` | 52.1, 51.4, 51.3 (51.4) | 53.0, 52.8, 53.7 (53.0) |
| Text, CPU µs per pass (engine) | 3.57, 3.62, 3.63 (2.98 to 3.03) | 3.51, 3.52, 3.47 (2.95 to 3.00) |
| Overworld fps, no stats clocks | 131.9, 132.6, 127.3 (131.9) | 134.6, 131.0, 130.5 (131.0) |
| Overworld fps, stats | 124.1, 125.4, 130.0 (125.4) | 129.0, 133.0, 130.9 (130.9) |
| Overworld, CPU µs per pass | 4.07, 4.03, 3.89 | 3.92, 3.80, 3.86 |

Text gains 4.3% without the clocks, and 55.4 fps is about 3.36 µs a step, against the 3.1
that 60 fps needs. Overworld is inside the load's noise without the clocks and gains 4.4%
with them.

## Study: the release candidate's text speed against today's

The earlier design (the release candidate's line) ran text at 69.5 fps
with every component in weights as one model, and the new design runs it at 55.4
without the stats clocks. Both were built and run on this machine today, interleaved,
three rounds, `--weights all --one-model`, the earlier design on its compiled weights and the new design on
its GGUF. Each side also got census runs (`GBCFORMER_UNITS=1`)
and `sample` profiles of the machine's thread (7 s of model frames, no clocks).

| Per pass, text | the earlier design | the new design |
| - | - | - |
| fps, `GBCFORMER_STATS=1` (load 3.2 to 4.5) | 67.6, 66.9, 66.7 (66.9) | 52.2, 52.0, 52.2 (52.2) |
| fps, no clocks (load 3.4 to 4.2; the earlier design's first run hit a load spike, 40.7, so two usable runs) | 66.8, 67.9 | 56.3, 52.9, 51.3 (52.9) |
| µs per pass (engine pass) | 2.78 (2.41) | 3.57 (3.02) |
| gate updates | 746 | 1,112 |
| (changed channel, reading layer) pairs | 141.1 | 191.2 |
| wire changes (`GBCFORMER_WIRE_CHANGES`) | 81.1 | 102.5 |
| positions (decoded a pass) | 46 (12.1) | 64 (9.8) |
| layer-0 changes | 4.5 | 6.7 |
| layers the stream walks | 14 | 11 (prefix overlay) |
| memory reads a pass | 3.6 from the byte image | 2.1 cell-row lookups |

The 69.5 reproduces as about 67 (within 4%). Master has no stage clocks, so the profiles
split the step. Each figure below is a profile share times µs a pass (no clocks).

| Stage | the earlier design µs (share) | the new design µs (share) | the new design − the earlier design |
| - | - | - | - |
| gate walk | 1.36 (48.9%) | 1.64 (46.6%) | +0.28 |
| pass loop, memory heads, fill | 0.50 (18.0%) | 0.68 (19.3%) | +0.18 |
| `drain_wires` | 0.32 (11.6%) | 0.41 (11.5%) | +0.09 |
| routing | 0.19 (6.8%) | 0.22 (6.2%) | +0.03 |
| layer 0, decode | 0.22 | 0.28 | +0.06 |
| write records and their K/V rows | 0.02 (0.6%) | 0.11 (3.1%) | +0.09 |
| the rest | 0.17 | 0.19 | +0.02 |

The engine lost nothing. The new design's walk costs 1.47 ns an update against the earlier design's 1.82
(the block walk and buckets came after the earlier design), and every stage's share is the same on
both sides. The gap is the work a pass does, and the memory's form sets that work.

| Source (census, updates and changes a pass) | Kind | Why |
| - | - | - |
| Master's opcode one-hot from eight class heads, and every read from the engine's byte image | flaw (excluded) | the one-hot was a 256-dim value per cached byte, which forces `kv_heads = heads` (52 KV heads, 216 GB of KV for Yellow, `docs/stock-llama.md` §0), and the image answered reads whose keys used RoPE recency, infinite in f16: no stock runtime could compute either from the weights |
| What replaced them: the opcode decoded from the fetched line, `OP0` 288 and `OP0w` 13 updates a pass | the price of a real cache | under one KV head a cell's value is its line's 64 bits, so the decode picks the opcode's nibbles from the line |
| The operand picks and the fetch heads' line bits: 30.8 line-bit flips a pass against the earlier design's 13.7 byte bits | the price of lines | lines are what make the cache stock-sized (141,415 cells against 1.13 M); recovered in part below (F1) |
| The write line: `WL0h`, `WL1h` 7.9 flips, the merge about 60 updates, 16 line positions against the earlier design's two byte slots | the price of lines | the earlier design wrote a byte into its index; a line cell needs the whole line. H3 (byte cells for RAM) is the lever |
| Write records' K/V rows, 0.09 µs | the price of a real cache | the earlier design pushed a byte into an f32 index; the new design builds the rows llama.cpp's f16 cache holds |
| 16-bit adders: the earlier design `pc+` 131 updates, the new design 41 | the new design ahead | nibble channels |
| Layers: the earlier design's stream walks 14, the new design's 11 | the new design ahead | the prefix overlay |

Most of the 14 fps went to turning the memory into something a stock runtime can compute.
The flaws are the image index and the class heads, and they are not ported. The rest is
what replaced them, the line form and the rows of a real cache. The engine lost no
efficiency, so nothing ports from the earlier design, and the round goes to cuts in today's design.
The decision log was kept locally and is not in the repository.

## Circuit: the operand bytes from the open row (F1)

A RAM's open row serves every byte of it (`docs/hardware-shaped.md` §18). The opcode head
reads PC's line. OP1 and OP2 are bytes of that line unless they cross into the next line,
which one head reads at line(PC) + 1 (`FNh`). Only that line's first two bytes are ever
picked, so only they take channels (`Circuit::unread`). The one line that straddles a host
boundary is FFF8..FFFF, whose last byte is IE. A third head (`FXh`) reads the line that
FFFF's fetch lands in, which is constant except under OAM DMA. The F1 and F2 translations
and the `OP1h` and `OP2h` heads go. `sm83::tests::fetch_bytes_come_from_their_lines`
checks every fetched byte and host flag against `layout::fetch` at PC FFE0..000F, the
bank edges and 200 random line ends, with and without the HALT bug and OAM DMA
(SingleStepTests reaches FFFF as OP1 in six cases, as OP2 in none, and never under DMA).
Both change tables were re-recorded.

| Census, text | the new design | F1 |
| - | - | - |
| wire changes a pass | 102.5 (`OP1h` 10.65, `OP2h` 9.74) | 83.2 (`FNh` 2.61) |
| (changed channel, reading layer) pairs | 191.2 | 183.6 |
| gate updates | 1,112 | 1,085 |

The timing runs ran under the machine's background load (other jobs, load 6 to
13), without waiting for a quiet machine, as asked. A is the new design with
its GGUF and B is F1 with its GGUF, interleaved, three rounds.

| | A | B |
| - | - | - |
| Text fps, `GBCFORMER_STATS=1` (load 7.3 to 10.6) | 52.9, 49.5, 51.1 (51.1) | 53.4, 54.4, 51.8 (**53.4**) |
| Text µs per pass (engine) | 3.52, 3.76, 3.64 (3.00 to 3.19) | 3.48, 3.42, 3.59 (2.91 to 3.05) |
| Text fps, no clocks (load 9.1 to 13.1) | 52.7, 50.7, 53.4 (52.7) | 54.1, 53.5, 53.6 (**53.6**) |
| Overworld fps, stats (load 6.0 to 10.4) | 126.2, 127.2, 123.7 (126.2) | 66.9, 128.8, 90.8 (90.8) |
| Overworld fps, no clocks (load 5.5 to 8.1) | 104.5, 81.4, 134.4 (104.5) | 114.7, 80.4, 135.2 (114.7) |

Text gains 4.5% with the clocks and 1.7% without. The PPU's thread bounds overworld, which
swings from 67 to 135 under this load on both sides. Each round's pair agrees except one,
and the GBC CPU's pass is equal or lower (3.92 to 4.08 µs against 3.97 to 4.08 on the
quiet runs). Every run reaches frame hashes `870aa3fd` and `822059c1`.

Sizes: cpu-memory is 135.3 M dense (hidden 1,856, 11 layers, depth 10), with nonzeros
going from 98,472 to 96,779. The one model is 634.7 M, its digest goes from `bb85df19` to
`4b0d4405`, and its GGUF holds 361.7 M values (from 363.4 M), 0 of 634,633,216 differing
(`digest --gguf`).

Digests (`examples/digest`): cpu-memory `9c89f0dc` to `32d229e8`, one model `bb85df19` to
`4b0d4405`; with `GBCFORMER_BITS=1`, cpu-memory `88fdb25e` to `bc1123fc` (140.6 M) and one
model `52b4cf60` to `4770c53a`; cpu `87bb33f8`, apu `83c6b259`, control `1ff09cda` and ppu
`a1ee149e` unchanged. `f16census all` finds 0 T1 and 0 T2 violations on every circuit. The
GGUF was kept locally. Checks on the commit: `sm83_sst` (both
memory circuits), `checkpoint_sst` (the engine equals the dense pass, exact in f16 and in
f16 sums), the fetch test above, DMG `lockstep-mem` 8 frames `6872b003` and CGB
`lockstep-mem` 300 frames `cfd31f3d`.

## Host: the threads that run passes at the user-interactive QoS (Q1)

A process launched from a shell runs at `QOS_CLASS_DEFAULT` (0x15, measured with
`qos_class_self`), the class of the background builds it shares the machine with. The APU
and PPU workers already asked for `QOS_CLASS_USER_INTERACTIVE`. The machine's thread,
which runs the GBC CPU's passes and limits text, did not, and neither did the positional
stage's helper threads. `qos::interactive` is now the one function that sets the class,
and `main`, `frontend::play`'s caller (a library caller may run the game on its own thread),
both workers and every pool helper call it. The change affects scheduling only, and no
weight, token or pass changes.

A is F1 and B is Q1, on the same GGUF, interleaved, three rounds, under the machine's
load.

| | A | B |
| - | - | - |
| Text fps, `GBCFORMER_STATS=1` | 37.7 (load 47 → 31), 53.6, 53.7 (load 27 → 11) | 52.6 (load 31 → 27), 54.2, 53.0 |
| Text fps, no clocks (load 10 → 4.7) | 55.5, 55.8, 54.7 (55.5) | 54.5, 56.2, 55.4 (55.4) |
| Overworld fps, stats (load 7.4 → 2.9) | 130.4, 132.8, 130.5 (130.5) | 134.8, 134.5, 135.7 (**134.8**) |
| Overworld fps, no clocks (load 4.8 → 3.8) | 135.2, 129.0, 135.8 (135.2) | 135.0, 136.2, 133.4 (135.0) |

Text is flat at load 4.7 to 27. The one pair at load 27 to 47 ran 37.7 against 52.6, but
its A side met the higher load, so it is not a clean comparison. Overworld gains 3.3% with
the clocks in all three pairs and is flat without. The change is kept, because it shows
no cost, and the project's target is the speed under background load, where the machine's
thread has to keep its pace. Every run reaches frame hashes `870aa3fd` and `822059c1`.

## Tried in the open-row round and not kept

| Change | Before → after (load) | Why |
| - | - | - |
| F2: the operand word lines read the opcode's word line for pc & 7 in place of pc's three low bits (a separate build) | text 53.9 → 53.4 fps with the clocks, 55.0 → 55.0 without (4.8 to 13.8); updates 1,085 → 1,033, pairs 183.6 → 183.9; overworld 127.4 → 132.1 without the clocks (5.3 to 8.9), 126.5 → 129.5 with them, one B run in a dip | the 52 updates were one-entry steps, the walk's cheapest; no text gain for a digest |
| H3 probe: byte write records, no write-line heads, a scratch host merge (a measurement only) | text 54.9 → 56.1 with the clocks, 57.0 → 58.0 without (2.9 to 6.4); engine 2.84 to 2.91 → 2.73 to 2.76 µs; pairs 183.6 → 173.7 | the host merge is engine work standing in for the model, so the probe cannot be kept; full H3 waits for a ruling (`docs/hardware-shaped.md` §19) |

**What limits the step now** (F1's census, text, one model). The gate walk is still about
half of the machine's thread, over 183.6 pairs and 1,085 updates a pass. The largest
single item is the opcode decode from the line (`OP0` 288 updates), since each line-bit
change of the opcode head reaches its nibble's 16 pattern nodes, and each word-line
change reaches 32. The operand bits' consumers come next (`OP1[*]` about 110 updates a
pass, `OP2[*]` about 50), then the write path that H3 would change (`WL0h` 6.71 and
`WL1h` 1.16 line-bit changes, about 60 updates in the merge, 16 line positions). The
fetch heads' flips are down from 30.8 to 13.0 a pass. Text needs about 3.1 µs a step for
60 fps. The head runs about 3.3 without the clocks at load 3 to 6, and about 3.45 at load
6 to 13.

**Against the base.** A is the new design with its GGUF and B the head
F1 with its GGUF, interleaved, three rounds. The machine was quiet for this
sitting, load 1.9 to 3.5.

| | A | B |
| - | - | - |
| Text fps, no clocks | 56.8, 56.6, 57.2 (56.8) | 57.8, 57.3, 57.6 (**57.6**, about 3.23 µs a step) |
| Text fps, `GBCFORMER_STATS=1` | 55.0, 54.6, 54.5 (54.6) | 55.4, 55.8, 55.6 (55.6) |
| Text µs per pass (engine) | 3.39 to 3.42 (2.88 to 2.91) | 3.34 to 3.36 (2.85 to 2.87) |
| Overworld fps, no clocks | 135.6, 135.8, 136.8 (135.8) | 137.5, 134.6, 136.8 (136.8) |
| Overworld fps, stats | 132.8, 134.6, 133.9 (133.9) | 135.0, 133.8, 133.2 (133.8) |

Under the background load of the F1 sitting (6 to 13), text ran 53.6 fps without the
clocks and 53.4 with them. 60 fps on text needs about 3.1 µs a step. The head is at 3.23
on a quiet machine and about 3.45 under that load.

**Gates on F1.** `tools/gate.sh` is all ok (30 checks: the lockstep hashes,
Crystal, Zelda DX and Mario DX 1,500 frames, every phase gate, Yellow 300 with every
component as one model `cfd31f3d`), and `cargo test --release` passes.

## Circuit: column select, the byte picked in attention

The logic picked the opcode out of PC's 8-byte line with 32 nibble one-hots and one step
per line byte, and each one-hot reached every line-bit flip of the head (`OP0` 288
updates a pass on text, the largest family). A RAM's column decoder makes that pick in
the memory, and attention can too once the column is in the query. There is one head per
(line, lane), whose query is the line when its word line is 1 and VOID's zero line when
it is 0, and every column of a byte drives one bus (`docs/hardware-shaped.md` §20). The
fetch's three bytes and read 0's byte are now buses, and `OP0h`, `FNh`, `FXh`, `D0h` and
`D0r` go. Depth, layers and the one model's hidden are unchanged.

**The census first.** A reference replay of text over 600 frames shows the opcode byte
flipping 3.91 bits a step, OP1 4.04 and OP2 3.85, and PC's low three bits changing 0.985
times a step. The projection for the opcode alone was OP0 going from 288 to 121 updates,
and the built circuit measured 120.

**The engine had to change before the circuit paid off.** Built on the old head path, the
opcode columns lost. A is F1 with its GGUF and B the opcode columns, interleaved,
three rounds, load 2.0 to 7.0.

| H1, opcode columns on the old head path | A | B |
| - | - | - |
| Text fps, `GBCFORMER_STATS=1` | 55.6, 55.6, 56.1 (55.6) | 51.3, 51.4, 51.6 (51.4) |
| Text engine pass, µs | 2.83 to 2.86 | 3.11 to 3.14 |
| Gate updates, pairs | 1,085, 183.6 | 913, 175.8 |

A timing probe around `Engine::memory_reads` per layer put the opcode layer at 80 ns
without the opcode columns and 341 ns with them. Every column walked all 64 bits of its
read where 8 have a channel, the eight columns recomputed one address each, and a pass
made two more index lookups. Four engine changes, each exact or guarded
(`docs/hardware-shaped.md` §20.4), took it to 132 ns with every fetch column in place:

- A head diffs only its live bits.
- The columns of a row share its address unit and read `word ? row : off` where the
  declared ranges prove the forms 0/1.
- The head loop walks the layer's dirty and hit bits, and a pass reads each (row group,
  line) once.
- A bus diffs the sum of its drivers' lane bytes, and drivers that overlap refuse the
  pass.

`Model::slice` now widens a stream's slice until the norm's bound holds. The first export
of the operand columns left the GBC CPU's prefix at 1,792, and the engine refused the plan
(squares 29,631 against 28,672), so every pass ran dense. `tests/one_model.rs` now asserts
every stream's fast path.

**Against the base.** A is F1 with its GGUF, B the operand columns with the
engine changes, and C is B plus read 0 by column select. They ran interleaved, three
rounds, with every component in weights as one model and weights from each side's GGUF.
Every run reaches frame hashes `870aa3fd` and `822059c1`.

| | A | B | C (kept) |
| - | - | - | - |
| Text fps, no stats clocks (load 3.4 to 4.8) | 56.9, 56.7, 55.6 (56.7) | 59.2, 59.5, 59.9 (59.5) | 59.5, 59.0, 59.9 (**59.5**) |
| Text fps, `GBCFORMER_STATS=1` (load 3.6 to 5.5) | 54.9, 55.2, 55.4 (55.2) | 56.5, 58.0, 57.6 (57.6) | 57.1, 57.6, 57.0 (57.1) |
| Text, µs per pass (engine) | 3.36 to 3.39 (2.88 to 2.90) | 3.21 to 3.29 (2.73 to 2.79) | 3.23 to 3.26 (2.74 to 2.78) |
| Overworld fps, no stats clocks (load 1.9 to 3.1) | 135.6, 135.3, 137.2 (135.6) | 136.5, 137.5, 137.1 (137.1) | 137.4, 137.3, 138.0 (**137.4**) |
| Overworld fps, stats (load 2.2 to 4.1) | 132.6, 133.1, 135.3 (133.1) | 135.0, 135.7, 134.2 (135.0) | 135.4, 134.5, 135.3 (135.3) |
| Overworld, engine pass µs | 2.82 to 2.85 | 2.79 to 2.81 | 2.78 to 2.79 |
| One model, dense | 634.7 M | 619.7 M | **594.8 M** |

Text gains 4.9% without the clocks (59.5 fps, about 3.14 µs a step against the 3.1 that
60 fps needs), and overworld gains 1.3%. Read 0 by column select is equal on text, 0.2%
faster on overworld, and smaller, so it is kept. The one engine-only pair (the live-bit
mask on the base circuit) was flat, at 2.97 against 3.00 µs.

**Census on the head** (text, every component, one model, `GBCFORMER_UNITS=1`). Wire
changes go from 83.2 to 64.4 a pass, (changed channel, reading layer) pairs from 183.6 to
154.6, gate updates from 1,080 to 813, and memory head reads from 1.7 to 2.8 (each a row
group's line once a pass). `OP0` goes from 288 to 120 (its one-hots come from four bus
bits, then the ANDs), and the operand picks are gone.

**Sizes.** cpu-memory has 20 heads a layer in place of 6 (the columns are at layers 2, 3
and 5). It goes from 135.3 to 164.1 M dense, intermediate from 1,408 to 1,280, and
nonzeros from 96,779 to 93,177 (0.18 M). The one model keeps hidden 1,856, 14 layers and
22 heads a layer, and goes from 634.7 to **594.8 M** dense (intermediate 5,504 to 4,992).
Its GGUF holds 366.3 M values (kept locally,
`--layer-shapes --align 64`), 0 of 594,721,792 differing (`digest --gguf`).
`f16census all` finds 0 T1 and 0 T2 violations on every circuit.

**Digests** (`examples/digest`). cpu-memory goes from `32d229e8` to `6e7a3e6c`, and the one
model from `4b0d4405` to `6eb1b6ae`. With `GBCFORMER_BITS=1`, cpu-memory goes from
`bc1123fc` to `7650fd2d` (145.5 M), and the one model from `4770c53a` to `85b2a4bf`
(641.1 M). cpu `87bb33f8`, apu `83c6b259`, control `1ff09cda` and ppu `a1ee149e` are
unchanged. Both change tables were re-recorded (text and overworld averaged,
`CORE=model-mem`).

**VOID's cell first.** A column off its line reads VOID's line. In a cache without that
cell, the column instead gets the best other key, at a record's own positions the
record's tokens, and on a bus the columns add. The memory-dense test pushed line 0 before
VOID. Its record's residual passed the norm's bound by layer 5 (squares 719,567 against
29,696, from 4,693 after layer 3), the cell's K/V came out scaled by 0.99979, and two
memory heads disagreed. The test already stated the rule that a query never names a line
without a cell, and the rule now holds from the first record. The test pushes VOID's line
first, and the stock harness's prefill decodes it right after the top cell
(`tools/stock-llama/src/cache.rs`, not run this round). The engine builds cells by
formula and is not affected. In the one model, a record's positions run no unit or head
of the GBC CPU's part, since its heads read the null token (inferred from
`docs/hardware-shaped.md` §15.2).

**Gates on the head.** `tools/gate.sh` is all ok (30 checks: the lockstep hashes,
Crystal, Zelda DX and Mario DX 1,500 frames, every phase gate, Yellow 300 with every
component as one model `cfd31f3d`). `cargo test --release` passes (every test binary,
`--no-fail-fast`), and so does `every_slice_is_the_checkpoint_at_its_positions` (ignored,
230 s). `GBCFORMER_ORACLE=1000` on Yellow CGB 300 with every component in weights and in
lockstep as one model from the GGUF reaches `cfd31f3d`, with 165 dense oracle checks on
the GBC CPU's stream and 1 on the PPU's palette read, all equal. `CORE=model-mem` keeps
its fast path. The standalone checkpoint's channel reuse leaves read 0's row forms
unproven as 0/1, so its columns take the sign path, each its own unit.

**The head against the base under the machine's background load.** A is F1 with
F1's GGUF and B the head with its GGUF, `decode/final/model.gguf`, interleaved, three
rounds, load 4.8 to 9.0 from other jobs.

| | A | B |
| - | - | - |
| Text fps, no stats clocks (load 5.2 to 6.3) | 55.8, 55.8, 55.9 (55.8) | 58.1, 58.1, 58.3 (**58.1**, about 3.22 µs a step) |
| Text fps, `GBCFORMER_STATS=1` (load 4.8 to 6.6) | 54.3, 54.4, 54.1 (54.3) | 56.4, 56.2, 56.1 (56.2) |
| Text, µs per pass (engine) | 3.42 to 3.44 (2.92 to 2.94) | 3.30 to 3.32 (2.80 to 2.82) |
| Overworld fps, no stats clocks (load 4.9 to 6.9) | 131.4, 133.1, 130.9 (131.4) | 132.4, 133.7, 134.1 (**133.7**) |
| Overworld fps, stats (load 5.8 to 9.0) | 129.2, 129.9, 132.1 (129.9) | 132.4, 131.1, 130.9 (131.1) |
| Overworld, µs per pass (engine) | 3.82 to 3.91 (2.86 to 2.91) | 3.82 to 3.86 (2.81 to 2.83) |

Text +4.1% without the clocks under load 5 to 6, +4.9% at load 3.4 to 4.8 (59.5 fps,
about 3.14 µs a step). 60 fps needs about 3.1: the head is short of it by about 1% on a
lightly loaded machine and about 3% at load 5 to 6.

**What limits the step now** (the census above). The gate walk over 154.6 pairs and 813
updates a pass is still the largest share. Its largest families are the opcode decode
(`OP0` 120, the nibble one-hots and the 256 ANDs), the operand word lines (`OP1w` and
`OP2w` at 30 each, the columns' gates), the rare read path (`RRARE` 26, `rare` 24, `D0rw`
20) and the write path (`WS` 21, `WL0h` 6.71 and `WL1h` 1.16 line-bit flips, the merge).
Wire changes are 64.4 a pass, of which `OP0` makes 9.5 (the opcode's one-hots), `WL0h`
6.7, `sel` 5.3, OP1 and OP2 7.9 (the buses) and PCL 3.8. Memory head reads are 2.8 a pass.

Three commits: the engine and IR support alone (every digest measured
unchanged, cpu-memory `32d229e8`, one model `4b0d4405`), the fetch columns (which
share this head's gate, and whose tree's `tests/one_model.rs` still counts the old
twelve heads) and the head (read 0 by column select, the tables, the docs).

## Memory: one key bit per RoPE pair, and the prefill order

**The key layout** (`docs/hardware-shaped.md` §21). The 19 line-address bits sat on dims
10..20 and 42..49. rotate_half pairs dim t with t + 32, so pairs 10..17 held a bit on
both halves, and a roped entry was ±1 ∓ sin(p·θ), off by up to about 6·10⁻⁶ and different
at every position. Now bit i sits on the first half of pair 12 + i (dims 12..30), with
the commit bit on dim 31, and every partner dim is 0 in every query and key. Pair t turns
by 10^(−1.1875·t) rad a position. From pair 12 on, `KEY_REACH` (8·LINES + 2 = 2,278,530,
every byte position H3 could use) turns it by under 2^-25 rad, so cos is exactly 1.0 in
f32 and sin rounds to 0 in f16. Every roped key entry is exactly ±1 with a ±0 partner,
every roped query entry is ±S, ±3S or B, and every logit is the same multiple of 1,024 in
any summation order. `memory::attn::key_pairs_turn_by_nothing` checks this, and
`Params::check` runs it; `a_cells_logit_is_one_number_in_any_order` checks eight positions
of one line in three orders, in f32 and f16. The overlay's null target moves from dims
22..30 to both halves of pairs 11..8, where it needs only its margin. Head dim stays 64.
This removes H3's blocker (b).

The projection before building said that no unit or channel changes, so the engine (cells
by formula) is speed neutral, and every checkpoint with memory heads moves its digest. In
the measured runs, A is the `decode/final` binary and GGUF and B this work with its GGUF, interleaved, three
rounds, every component in weights as one model. Every run reaches frame `870aa3fd`, and
both sides walk 154.6 pairs.

| | A | B |
| - | - | - |
| Text fps, no stats clocks (load 2.5 to 3.3) | 60.2, 57.9, 58.1 (58.1) | 59.7, 58.1, 58.2 (**58.2**) |
| Text fps, `GBCFORMER_STATS=1` (load 2.5 to 3.6) | 57.9, 56.4, 58.6 (57.9) | 56.8, 58.3, 58.6 (58.3) |
| Text, engine pass µs | 2.69 to 2.79 | 2.69 to 2.77 |
| Text fps, no stats clocks, two more rounds (load 1.2 to 2.8) | 60.2, 59.9 | 60.4, 60.2 |
| Overworld fps, no stats clocks (load 2.4 to 4.3) | 136.7, 137.3, 137.6 (137.3) | 137.1, 137.2, 137.2 (**137.2**) |
| Overworld fps, `GBCFORMER_STATS=1` (load 2.0 to 2.5) | 136.0, 135.5, 135.4 (135.5) | 134.9, 134.9, 134.6 (134.9) |
| Overworld, µs per pass (engine) | 3.71 to 3.73 (2.77 to 2.79) | 3.74 to 3.75 (2.78 to 2.81) |

Speed is flat, as projected (overworld with clocks is 0.4% lower, within the engine's 2.77
to 2.81 spread). Text reaches 60 fps without clocks only below load 3. It runs 60.2 to
60.4 at load 1.2 to 2.8 and 58.1 to 58.2 at 2.5 to 3.3, so the head stays about 1 to 3%
short at load 3 to 8.

**Digests** (`examples/digest`). cpu-memory goes from `6e7a3e6c` to `851a553c`, ppu from
`a1ee149e` to `bac20544`, and the one model from `6eb1b6ae` to `6c006c2a` (594.8 M, sizes
unchanged). With `GBCFORMER_BITS=1`, cpu-memory goes from `7650fd2d` to `0ab21595`, and
the one model from `85b2a4bf` to `5f1033aa`. cpu `87bb33f8`, apu `83c6b259` and control
`1ff09cda` are unchanged. The one model's GGUF (kept locally,
`--layer-shapes --align 64`, 366.3 M values) reads back with 0 of 594,721,792 differing
(`digest --gguf`), and `f16census all` finds 0 T1 and 0 T2 violations on every circuit.
Both change tables were re-recorded (text and overworld, `CORE=model-mem`), and they are
identical, as they must be, since no wire moved.

**The prefill order** (the exactness debt the column heads left, `docs/stock-llama.md`
§4). A record's cell rows are exact only while every channel at its positions stays in
range. An off column reads VOID's zero line, and without VOID's cell it reads the
record's own tokens, and a bus adds its columns. That bound is measured, not
constructed. The GBC CPU's other wires at a record's positions see gathered garbage and
stay in range only in practice, and `DenseMemory::push` checks every record's key and
value against the formula bit for bit. Making the bound hold by construction in the
standalone checkpoint would need the overlay's mode gating there (a mode channel, a null
target, `up` = mode on every unit), which this round did not build. So the order is now
a rule the code enforces. `DenseMemory::new` keeps VOID's zero line first, and `push`
refuses any record before it (`tests/memory_dense.rs` pushes VOID first in both tests).
The harness's prefill decodes the top cell, VOID's line and the fixed regions, then the
top cell and VOID's line again over them (2,121 one by one), and `Cache::put` refuses any
other record before VOID's cell.

**The stock harness, run.** An existing llama.cpp build was on disk, the 18-hour proof's
`.build-heads/llama.cpp-heads` (b12070 with `layer-shapes.patch`, Metal) in the
other worktree. The harness at this work builds against it in 30 s.

- `sst-mem` (the cpu-memory checkpoint, this work's GGUF, one SST case per opcode
  file, Metal, load 5.3) ran 245 cases, and 0 failed. It made 780 write records. Of 365 reader
  records probed after the run, 0 differ from the engine's index. The prefill took 41.3 s
  (2,121 one by one, 9,231 in lanes), and a pass took 27.4 ms. The run checks tokens and
  reads, not K rows, so a sub-ulp key residue would not show here. The row-level proof is
  `key_pairs_turn_by_nothing` and `DenseMemory::push`'s bit-exact check.
- `play` (the one model, `decode/final` GGUF, Yellow, 300 passes, before this work's
  edits) decoded 141,415 cells in the prefill in 404 s, and then `Circuit::reach`
  panicked on the PPU's scope (`ppu:sx0: address bit 3 ranges over [0, 20]`). With
  `STOCK_SCOPE=0` the control stream's first pass decoded wrong tokens (`control: a
  schedule of 255 starts from 256 with the horizon 0`). With `STOCK_GROUP=full` as well,
  the engine panicked at `engine/run.rs:1347` (index −1). No stock run has used an
  overlay model, since the proof's worktree did not contain the overlay
  work.
- The stock cache holds no null cell. The dense pass appends `null_kv` to every cache
  (`Model::residual_cached`), but the harness never decodes `tok::null`. So in llama.cpp
  a memory head off its part's positions, and every GBC CPU head at a record's positions,
  has only cells to attend to. The earlier entry's "a record's positions run no unit or
  head of the GBC CPU's part" holds for the dense pass, not for a stock cache. A decoded
  null would not reproduce `null_kv` either. `memory_head` puts −1 on every key dim's
  constant term, so the decoded null carries line 0 uncommitted on the key dims, and `N`
  was sized against a key that scores 0 there. Fixing that is a model-side change (the
  null's key rows or `N`), and together with the scope derivation's panic it now blocks
  the stock proof, ahead of the prefill order. Withholding `null_kv` from the dense pass
  did not fail `every_slice_is_the_checkpoint_at_its_positions` (227.7 s), because its
  caches hold too few cells to show the effect. So this run neither shows nor rules out
  the null cell as the cause of the first wrong control pass.

**Gates on this work.** `tools/gate.sh` is all ok (30 checks, Yellow 300 with every
component as one model `cfd31f3d`). `cargo test --release` passes (every test binary), as
do `every_slice_is_the_checkpoint_at_its_positions` (ignored, 227.5 s), `checkpoint_sst`
and `memory_dense`. `GBCFORMER_ORACLE=1000` on Yellow CGB 300 with every component in
weights and in lockstep as one model from the GGUF reaches `cfd31f3d`, with 165 dense
oracle checks on the GBC CPU's stream and 1 on the PPU's palette read, all equal. The
census (text, one model, `GBCFORMER_UNITS=1`) shows 154.6 pairs a pass, unchanged.

**H3 not built.** Its blocker (b) is gone, but H3 is a format change. It touches record
tokens, positions, `KvCache`, the engine's cell rows and all-eight guard, `DenseMemory`,
the PPU's image, the DMA, HOST and palette records, and the harness, 264 references
across ten files. Its proof is llama.cpp over the byte cells, which the harness cannot
run on an overlay model today, and its +4 to 5% is a projection from a probe that used a
host merge.

## The one model in stock llama.cpp

No stock run had used an overlay model (`docs/hardware-shaped.md` §15.2). The key-bit round's
short `play` (Yellow, 300 passes, llama.cpp b12070 with `layer-shapes.patch` on Metal)
failed three ways. `Circuit::reach` panicked on the PPU (`ppu:sx0: address bit 3 ranges
over [0, 20]`), with `STOCK_SCOPE=0` the control stream's first pass decoded wrong tokens,
and with `STOCK_GROUP=full` too the engine panicked on them. This work fixes three
causes.

1. **No null cell in the cache** (confirmed). The dense pass appends the null cell as a
   formula (`llama::null_kv`). llama.cpp holds only cells it decodes, and the harness
   never decoded the null token, so another part's memory head at a stream's positions
   attended a real line and wrote it into channels the stream's own wires share. Had the
   harness decoded the token, the cell would still have been wrong. A key dim is
   `2·bit − 1`, so the decoded null keyed −1 on all 19 key dims (line 0, uncommitted,
   which the null term `N` was never sized against), and the two copy-only layers held no
   null key. `the_null_token_decodes_the_null_cell` (`tests/one_model.rs`, ignored, 96 s)
   runs the whole checkpoint's dense pass against the control stream's slice over 2,041
   random lines. With no null cell all 54 control output positions differ, and with the
   formula's null cell 0 differ. The decoded null was wrong at 10 of 10 memory layers
   before the fix and at 0 after. **Fix, in the weights.** Every memory layer's key rows
   read the null channel, with `null_k` on the null dims and +1 on each key dim
   (cancelling the −1). Only the null token sets the channel, and a stream's slice drops
   the column, so every stream and every other checkpoint is unchanged. **In the
   harness.** `Runtime::null` decodes the token alone at position 0 of the empty sequence
   0 before anything else (line 0's position, ROM, never evicted, asserted), and every
   scope and lane copies it.
2. **Queries past f16 in flash attention** (a bound, not yet hit, since no GBC CPU pass
   ran in the failed play). Metal's flash attention keeps the query in f16 (`q_t` =
   `half`), and an entry above 65,504 becomes inf. Against a zero key partner, that inf
   gives a NaN logit, which then corrupts every token attending it. `Model::attention_bounds`
   bounds every query and key entry at any position over every occupant's range, after
   RoPE. The PPU's layer-4 heads at the GBC CPU's positions, over its nibble channels,
   reach 414,720, and layer 0's keys reach 16,416, so one shift for every layer cannot
   fit. **Fix, in the GGUF.** Layer l's `attn_q` is divided by 2^shift[l] and its `attn_k`
   multiplied by it (`gguf::qk_shift`, the least power that keeps both at most 65,504;
   metadata `gbcformer.qk_shift`). The shift is 3 at layer 4 and 0 elsewhere. Every logit
   is the same product of f16-exact factors, and the loaders undo the shift
   (`digest --gguf` 0 differ).
3. **`Circuit::reach`'s assert** expected each query bit's interval inside [−1, 2], and
   under interval arithmetic a sum of exclusive terms exceeds that interval. The bit is the sign
   `[form ≥ 1]` whatever the value, so a sign known at both ends is known, and the assert
   is gone. The scopes are what they were (PPU 2,118 lines named, 2,091 with a cell; APU
   and control 2,075; the GBC CPU 7,300 under its first windows).

Cause 1 explains both the control stream's wrong tokens and the engine's panic. With the
null cell decoded, the first control pass, which runs before the prefill, is right.

**Stock short run** (llama.cpp-heads b12070 with `layer-shapes.patch`, Metal, one model
`31307a65`, GGUF kept locally, `STOCK_GROUP=own`, scoped,
load 2.2 to 1.1). The prefill decoded 141,415 line cells in 405.4 s (2,138 one by one
with the null cell, 139,279 in lanes). Then 4,000 passes ran, every one in the runtime,
diffed against the engine's record: GBC CPU 2,863 events, PPU 102, APU 265, control 771,
**4,001 events equal, 0 differ, 274,942 decoded tokens**. The mirror found 0 of 117
checks differ, and 3 decodes were off P. `split-check` holds with the null cell (the
group's best layer-0 key beats any cell's, the null's included, by 384 in every writing
head).

**Projected frame-300 proof** (`tools/stock-llama/bench.sh`, 300 passes from the saved
cache, median of 3, load 2.3 to 1.3). A pass takes 26.1 ms on the GBC CPU stream, 26.8
on the PPU, 17.7 on the APU and 18.2 on control, which gives **9.2 hours** by the bench's
pass counts. This model's engine record to frame 300 matches those counts: 981,203 CPU,
83,500 PPU, 243,951 APU and 59,051 control events. The bench counts pass time only. The
2026-11-05 proof, on the older model, also spent 4.4 h in 124,753 write records and 2.1 h
in 46,929 reader records. At this run's 106 ms per write record and the proof's 164 ms per
reader record, that adds about 5.8 h, so the run should take about 15 hours plus the
7-minute prefill.

**Digests.** The one model goes from `6c006c2a` to `31307a65` (594.8 M), and with
`GBCFORMER_BITS=1` from `5f1033aa` to `fe00f7a1`. cpu `87bb33f8`, cpu-memory `851a553c`,
apu `83c6b259`, control `1ff09cda` and ppu `bac20544` are unchanged.

**Gates.** The build has 0 warnings, `cargo test
--release` is ok, and `tools/gate.sh` reports 30 ok and 0 FAIL. The ignored one-model
tests (the slice test and the null-cell test) are ok in 235.6 s, and `f16census all`
finds 0 T1 and 0 T2 on every circuit. `GBCFORMER_ORACLE=1000` on Yellow 300 as one model
from the GGUF reaches `cfd31f3d` with 165 + 1 dense checks equal, and the stats census
on text reaches `870aa3fd`. Both change tables were re-recorded (text and overworld,
`CORE=model-mem`) and are identical, since no wire moved.

**Speed A/B** (text, every component in weights as one model, no stats clocks,
interleaved, three rounds; A is the `keys/final` binary and GGUF, B this work's). A ran
60.3, 59.3, 60.3 (60.3) and B 60.3, 60.3, 60.4 (**60.3**) fps at load 1.2 to 3.0, and
every run reached frame `870aa3fd`. Speed is flat, because the engine runs each stream's
slice, which drops the null column.

## Host and engine: the machine thread's last few percent

The profile (`sample`, 9 s of text model frames, every component in weights as one model,
no clocks, the machine's thread, 3,362 samples) gives the engine's pass 84.6%. Inside it,
the gate walk takes 41%, `memory_reads` 14.3%, `drain_wires` 9%, `route_layer` 5%, layer
0 5% and the decode stage 4%. Outside it, `MemCheckpointCore::decode`'s two loops take
4.0%, `fill_into` and the view 1.8%, the write records 4.5% (`Cells::build` on memo misses
2.7%), the inline control checkpoint 1.3%, the APU read wait 0.6% and the DMA copies 0.5%.
A census of `memory_reads` (temporary counters) counts, per pass, 7.2 layer calls, 14.1
query-channel changes, 16.8 dirty heads on 6.6 distinct address units (51 heads, 26
units), 6.9 address moves, 2.8 cell reads, 6.1 column-byte changes and 3.25 bus moves.

| Change | Before → after (load) | Verdict |
| - | - | - |
| H1: `memory_reads` scans each address unit once after its bits move (a stale bit per unit, the per-head 19-bit scan cached; a test held the cached bounds to the scan) | engine 2.72 to 2.74 → 2.78 µs, text no clocks 59.7 → 58.3 (2.3 to 3.6); as a per-channel unit mask 2.73 → 2.80 to 2.82 µs | rejected: the loop is latency and branch bound, the redundant scans were not its cost |
| `#[inline(never)]` on `memory_reads` (its doc comment says it is kept out of `advance`; the attribute is not there) | engine 2.71 to 2.75 → 2.73 to 2.75 µs, 57.7 → 57.3 fps with clocks (1.8 to 5.0) | flat, not kept |
| H2: the host decode loop reads each decoded token's byte with the token ranges loaded once (`tok::Bytes`, held to `tok::byte` over every token by a test) and writes the masks as one offset | text no clocks 57.6 → **57.9** (five rounds, 1.7 to 6.9; B ahead in 4 of 5), with clocks 56.2 → 57.1, µs per pass 3.28 to 3.35 → 3.25 to 3.28 | kept |
| H3: a stream's view names each new position list with a process-unique id (`View::positions_id`); the decode stage's want list and layer 0's plan list compare ids, and the 64-entry list only when an id is 0 or new | text no clocks 57.9, 57.5, 57.6, 57.9 (57.75) → 58.2, 58.4, 58.2, 58.5 (**58.3**) (3.1 to 4.1; B ahead in 4 of 4), with clocks 58.2 → 58.6, engine 2.72 to 2.73 → 2.69 to 2.71 µs | kept |

H3's guard works as follows. The id changes whenever the stream's list differs from the
last one placed, and a stream list maps to one checkpoint list (`View::at` is fixed), so
equal ids mean equal lists. Id 0 (no view, or before the first place) compares the lists
as before.

Also tried and not kept, each below the A/B's floor of about 0.5%: skipping `exp` below
`−EXP_ZERO` in `RowBuilder::group_rows` (the row build's `exp` is 0.5% of samples), the
`Memo` copy on a memo hit, the third list compare in `View::place` (its caller's list
carries no id), and the inline control pass (1%).

**Base against head.** A is the overlay round and B this round, both on the one model's
GGUF `31307a65`, interleaved, three rounds. Every run reaches frame `870aa3fd`.

| Text | A | B |
| - | - | - |
| fps, no clocks (load 1.0 to 2.5) | 59.7, 59.5, 59.5 (59.5) | 60.5, 61.0, 60.7 (**60.7**) |
| fps, `GBCFORMER_STATS=1` (load 2.2 to 4.0) | 58.0, 57.4, 55.7 (57.4) | 58.5, 58.8, 57.7 (58.5) |
| µs per pass (engine pass) | 3.21 to 3.34 (2.71 to 2.81) | 3.17 to 3.23 (2.70 to 2.74) |

The head gains 2.0% without clocks and 1.9% with them. At load 3.1 to 4.1 (H3's rounds)
the head ran 58.2 to 58.5 without clocks, so it does not reach 61 at load 3 to 8.

**What limits the step now** (the head's profile, the machine's thread). The gate walk
takes 41% (about 1.3 µs over 154.6 pairs and 813 updates). `memory_reads` takes 13.6%,
and it is bound by latency and branches. H1 showed that its redundant scans were not the
cost, and its routing of bus flips is ordinary wire changes. `drain_wires` takes 9%,
`route_layer` 6%, layer 0 4.5% and the decode stage 4%. Outside the engine, about
0.45 µs a pass is spread over a dozen items of 1% or less (the write records 4%, with the
row build on memo misses 2%, the decode loop 2.6%, the view 1.8%, the inline control
pass 1%, the APU read wait 0.8%). 60 fps on text needs about 3.1 µs a step, and the head
takes about 3.07 µs at load 1 to 2.5 and about 3.19 at load 3 to 4. The next gain has to
come from the circuit (fewer pairs in the walk) or the memory path's shape, not the host.

**Gates on the head.** `tools/gate.sh` reports 30 ok and 0 FAIL (the lockstep hashes,
every phase gate, Yellow 300 with every component as one model `cfd31f3d`), and
`cargo test --release` reports 176 passed and 0 failed. No weight moved. The digests
(`examples/digest`) cpu `87bb33f8`, cpu-memory `851a553c`, apu `83c6b259`, control
`1ff09cda`, ppu `bac20544` and one model `31307a65` (594.8 M) are all unchanged. The
decision log was kept locally and is not in the repository.

## The step oracle: sampled steps of every stream against the dense pass

An outside review found that the dense oracle (`GBCFORMER_ORACLE`) checked only memory
records, never a step's decoded tokens, so the one model's GBC CPU and PPU streams over
the shared cache had no engine-against-dense token check. `GBCFORMER_STEP_ORACLE` now
runs every Nth fast pass of each engine, the first included, through the dense pass of
the engine's own network over the same tokens and the same KV cache, and stops at the
first decoded token that differs (`docs/one-model.md` §4). The cadence is one count for
every engine or one per component, and it counts passes, not steps.

**The dense pass, threaded.** The first probe (Yellow CGB 300 as one model, cadence
50,000) ran serially and was stopped after about 9.5 minutes. `sample` put 86% of the
machine's thread in `residual_with_null`'s memory heads, a dot product per cached cell
per position. The dense pass now splits its matmul rows and its attention rows across
threads above 2^26 multiply-adds (`llama::each_row`), and reads each layer's cached keys
once instead of once per position and head. Each row is computed alone, so the result is
bit for bit the serial pass's. The engine's tokens equalled the dense pass's at every
check below, and the digests are unchanged.

| Run (load) | Passes checked (GBC CPU, APU, control, PPU) | Time |
| - | - | - |
| cadence 50,000, threaded (about 30) | 19, 5, 2, 2; 18.5, 2.4, 1.7 and 4.1 s a check | 387 s |
| `tools/gate.sh`'s cadence `cpu=100000,apu=25000,control=6000,ppu=8400`, keys read once (26 to 29) | 10, 10, 10, 10 | 187 s |
| the same run without the oracle (2 to 3) | none | 19 s |

All three reach `cfd31f3d`. With a token flipped in the comparison (a temporary hook,
not committed), each stream's first check stopped its run with exit 101, the APU's
and the PPU's from their own threads. For example, the GBC CPU's message was "step
oracle: fast pass 1 at position 281 (cpu:RA0L) decoded token 4369, the dense pass 4368".

**The slice test** (`every_slice_is_the_checkpoint_at_its_positions`) compiles the
engines once for the APU and control streams (the only two it covers) and takes each pass's tokens from the residual it already
computed. It went from 230 s for 2 passes a stream to 40 s, and runs 8 passes for each
in 133 s (load 20 to 26), so it is no longer ignored.

The threading also shortened the full `cargo test --release` binaries that run dense
passes. At load 24 to 27 on the head, `ppu_checkpoint` took 23.1 s (CLAUDE.md said about
3 minutes), `checkpoint_sst` 73.6 s, `sm83_sst` 85.2 s and `one_model` 164.8 s with the
slice test included. The whole suite took 661 s: 178 passed, 0 failed, 1 ignored (the
null-cell test).

**Gates on the head** (with these docs). `tools/gate.sh` reports 31 ok and 0
FAIL in 927 s at load 23 to 30, the new step-oracle line included (`cfd31f3d`, 10, 10, 10,
10). The build has 0 warnings. No weight moved: the digests cpu `87bb33f8`, cpu-memory
`851a553c`, apu `83c6b259`, control `1ff09cda`, ppu `bac20544` and one model
`31307a65` (594.8 M) are unchanged.

## PPU: the display registers in weights

LCDC, SCY, SCX, WY and WX become display bytes in the PPU checkpoint's image, read by
one constant head on BGP's line, and the line's pass takes only `LY WLN FLG`
(`docs/ppu-in-weights.md` §12). A CPU read of one of them comes from the checkpoint.

**Size** (`examples/shape --ppu`). Depth 13 and 14 layers as before. Dense parameters go
from 254.1 M to 252.9 M and nonzeros from 318,898 to 317,621, since five input slots
fewer drop 1,285 tokens from the vocabulary. The widest layer holds 21 heads instead of
20.

**Digests.** ppu `bac20544` to `e368efef`, the one model `31307a65` to `39f44e8d`
(593.3 M), and with `GBCFORMER_BITS=1` from `fe00f7a1` to `d2e97766`. cpu `87bb33f8`,
cpu-memory `851a553c`, apu `83c6b259` and control `1ff09cda` are unchanged.

**Gates.** The build has 0 warnings.
`tools/gate.sh` passed every line but `cargo test`, whose failures were the PPU stream's
counts in `tests/one_model.rs` and `tests/stream_audit.rs` (170 to 165 slots, 57 to 58
heads, its inputs in 2 layer-0 gathers instead of 1), updated in the next commit; those
two binaries then passed. Every hash held: Yellow, Crystal, Zelda DX and Mario DX 300 frames with
all five in weights and in lockstep (`cfd31f3d`, `ccfe7ad3`, `e2bd7038`, `ea7f86ca`),
Yellow as one model with and without the step oracle, both acid2 images with the PPU
in weights. The ignored null-cell test is ok, `f16census all` finds 0 T1 and 0 T2, the
one-model GGUF (`--layer-shapes --align 64`) has the compiled digest, and
`GBCFORMER_ORACLE=1000` on Yellow 300 as one model from that GGUF reaches `cfd31f3d`
with 165 + 1 dense checks equal. In that run Yellow made 39 display byte reads and the
CPU's thread waited 8.97 ms for them over 300 frames.

**Speed A/B** (every component in weights as one model, interleaved, three rounds; A is
the base, B this change). text.sh A 60.1, 60.4, 58.5 (median 60.1) and B 60.7, 59.9,
58.3 (59.9); overworld.sh A 136.0, 135.5, 130.3 (135.5) and B 138.0, 134.6, 130.7
(134.6), at load 2.6 to 14.7 rising through round 3. Every run reached `870aa3fd` or
`822059c1`. The differences (-0.3% and -0.7% at the median) are inside the
round-to-round spread of 3 to 4%.

**Stock short run** (b12070 with `layer-shapes.patch`, Metal, one model `39f44e8d`, the
GGUF above, `STOCK_GROUP=own`, load 2.5 to 4.1). The prefill decoded the cells from
scratch, then 4,000 passes ran in the runtime, diffed against the engine's record: GBC CPU
2,863 events, PPU 102, APU 265, control 771, **4,001 events equal, 0 differ, 274,942
decoded tokens**. The PPU stream's 102 events now decode its three inputs.

## Control: the MBC3 cartridge clock in weights

The MBC3 clock moved from `machine::cart` into the control logic and its checkpoint
(`docs/control-in-weights.md` §20). The control checkpoint grows from 43.77 M to 66.33 M
dense parameters and from 56,241 to 66,540 nonzeros. Depth (11), layers (12) and the
f16 census (0 violations) are unchanged. Clock events are passes in place, at the clock's
bus accesses and once a wall-clock second, so they add a handful of passes per run: on
Crystal 300 frames, 51 writes, 7 reads and 4 wall times beside 134,881 control passes.
No speed was measured. Only the control and one-model weights move: `examples/digest`
gives control `bf493acf` and one model `c6544608` (594.8 M to 615.1 M), with cpu
`87bb33f8`, cpu-memory `851a553c`, apu `83c6b259` and ppu `bac20544` unchanged.

**After the display registers.** The clock's commits were
cherry-picked onto the display registers' work. There the one model is `baa801d1`
(613.6 M, 0.96 M nonzeros) and with `GBCFORMER_BITS=1` `28fe3bf8`; control is
`bf493acf`, ppu `e368efef`, and cpu, cpu-memory and apu are unchanged.
The gates: `tools/gate.sh` every line
ok (`cargo test` included), the ignored null-cell test ok, `f16census all` 0 T1 and 0 T2,
the GGUF's digest equal to the compiled one (0 tensors differ), and
`GBCFORMER_ORACLE=1000` on Yellow 300 as one model from the GGUF at `cfd31f3d` with
165 + 1 dense checks equal.

**Speed A/B** (every component in weights as one model, interleaved, three rounds; A the
display registers' binary, B with the clock; load 10 to 31 from other jobs). text.sh A
58.5, 58.5, 57.8 (median 58.5), B 58.1, 57.8, 58.1 (58.1); overworld.sh A 127.8, 121.3,
129.1 (127.8), B 131.6, 127.3, 128.2 (128.2). Every run reached `870aa3fd` or
`822059c1`. The clock costs nothing measurable: Yellow has no clock, and its control
passes carry the wider circuit's unchanged channels only.

**Stock short run** (b12070 with `layer-shapes.patch`, Metal, one model `baa801d1`,
`STOCK_GROUP=own`, load 8 to 31). 4,000 passes from boot in the runtime against the
engine's record: GBC CPU 2,863 events, PPU 102, APU 265, control 771, **4,001 events
equal, 0 differ, 281,110 decoded tokens** (the control stream's 771 events now decode
its clock slots too).

## Control: DMA transfers as model outputs

Every OAM DMA byte and HDMA block, and every stall, is now a control output
(`docs/control-in-weights.md` §21): one pass per OAM DMA M-cycle, one block per pass, and a
16-bit stall output. The control checkpoint shrinks from 66.33 M to 55.98 M dense
parameters and from 66,540 to 62,287 nonzeros (hidden 1,024 to 896, intermediate 640 to
576, depth 11 and 7 heads unchanged), since the general-purpose adders went. Digests:
control `a39e22da`, one model `50e24cc2`; cpu `87bb33f8`, cpu-memory `851a553c`, apu
`83c6b259` and ppu `e368efef` unchanged.

| Workload | Before (fps, median of 3) | After | Change | Control passes a frame |
| - | - | - | - | - |
| text.sh `--weights all --one-model`, interleaved, 6 rounds (3 each order), load 8 to 29 | 57.65 | 56.35 | −2.3% | 58.4 → 216.4 |

The cost is Yellow's one OAM DMA transfer a frame, 160 more control passes. It is a little
over the 2% the brief allowed; `docs/control-in-weights.md` §21.4 sketches the alternative.

**Per-byte OAM DMA.** `tools/gate.sh` every line
ok, the null-cell test ok, `f16census all`
0 T1 and 0 T2, the GGUF's digest equal to the compiled `50e24cc2` (0 tensors differ),
`GBCFORMER_ORACLE=1000` on Yellow 300 as one model from the GGUF at `cfd31f3d` with 165
+ 1 dense checks equal, and `GBCFORMER_BITS=1` one model `1bf213bc`. The stock short run
(4,000 passes from boot) has 4,001 events equal and 0 differ. The A/B against the clock's
binary (one model, interleaved, three rounds, load 4 to 33): text.sh 60.8, 58.4, 57.7
(median 58.4) against 59.0, 55.7, 56.0 (56.0, -4.1%); overworld.sh 139.0, 131.7, 132.2
(132.2) against 130.8, 127.6, 129.4 (129.4, -2.1%). That is over the 2% line, since the
control model runs 160 more passes a frame, so the OAM DMA moves to line runs next.

## Control: OAM DMA in line runs

A pass moves a run of OAM DMA bytes up to the end of an 8-byte OAM line instead of one
byte (`docs/control-in-weights.md` §21.5). Control 55.98 M to 59.98 M dense, 62,287 to
63,296 nonzeros; digests control `d9dcad71`, one model `07373770`.

| Workload (`--weights all --one-model`, 3 rounds rotated, load 3.7 to 12.1) | Before §21 | One byte a pass | Line runs |
| - | - | - | - |
| text.sh (fps, median) | 57.1 | 55.0 | 57.0 |
| overworld.sh | 133.2 | 131.4 | 130.7 |
| control passes a frame, text and overworld | 58.4, 44.0 | 216.4, 202.0 | 77.3, 62.9 |

Text recovers to within 0.2% of the base; overworld's rounds overlap for all three.

## I/O dispatch from the memory model

The memory model's translation now names the device of every host-served access in the
high bits of the address it emits, and the host dispatches by `layout::port`
(`docs/host-in-weights.md` §3 step 2). The I/O page's port table costs units only when
an instruction addresses the page: Wired translations read the low bits gated by the
I/O page, LDH A,(n) and LD A,(C) read through a gated FF-page translation on the rare
heads, and LDH (n),A gates its low bits too. A first build that kept LDH A,(n) on the
stable reads re-evaluated 1,087 units a pass on text.sh, its port table reading the
fetched byte every step.

Units re-evaluated per fast pass, `CORE=model-mem GBCFORMER_STATS=1`, one run each
(load 60 to 70, other runs alongside; µs are contended):

| workload | circuit | before | after |
| - | - | - | - |
| text.sh | nibble | 817 (3.28 µs) | 837 (3.37 µs) |
| overworld.sh | nibble | 777 (3.53 µs) | 772 (3.54 µs) |
| text.sh | 0/1 | 1,223 | 1,243 |
| overworld.sh | 0/1 | 1,156 | 1,152 |

The memory checkpoint keeps hidden 1,856, intermediate 1,280 and 11 layers (164.1 M);
nonzeros go from 93,177 to 100,541. The 0/1 memory checkpoint is 145.5 M with 146,837
nonzeros. The layout grows by the I/O blocks and the readback table (TOTAL `0x22D400`),
which moves the readers' positions. `examples/digest`: cpu-memory `096c5372` (0/1
`060e022f`), one model `c715f225` (615.1 M to 628.6 M; 0/1 `c19997df`); cpu `87bb33f8`,
apu `83c6b259`, control `bf493acf` and ppu `e368efef` unchanged. No perf A/B was run.

The one model grows because its memory stream's widest MLP level grows: the stream's
view reports intermediate 1,024 to 1,088 (0/1: 1,408 to 1,472) at the same hidden, while
the standalone memory checkpoint stays inside its 1,280 quantum. SingleStepTests: the
host and memory circuits, the 0/1 circuits and both checkpoints pass all cases they run
with 0 failed. The memory cores skip 573 of 500,000: one aliased echo cell, as before,
and 572 cases that read a byte from an unmapped I/O register or the unusable area,
where the flat test bus holds a byte no console returns and the model keeps its
constant. The harness now loads only the bytes a case reads from memory, so write-only
bytes and opcode bytes the host serves no longer refuse a case.

## The line-run DMA and the I/O dispatch together

With the OAM DMA line runs and the I/O dispatch (and the commits
before it) both in, the digests are cpu `87bb33f8`, cpu-memory `096c5372`,
apu `83c6b259`, control `d9dcad71`, ppu `e368efef` and one model `e83d0c16` (628.6 M,
0.97 M nonzeros). The gates:
`tools/gate.sh` 31 ok and 0 FAIL, the null-cell test ok, `f16census all` 0 T1 and 0 T2,
the GGUF's digest equal to the compiled one (0 tensors differ), and `GBCFORMER_ORACLE=1000`
on Yellow 300 as one model from the GGUF at `cfd31f3d` with 165 + 1 dense checks equal.

The stock harness's prefill assumed the HOST block ends the layout; the I/O ports and
the readback table now follow it, so the top cell is the table's last line and the
fixed decodes include both blocks. The stock short run then has 4,001 events equal and 0
differ (284,194 decoded tokens).

## Clocks in weights: the frame sequencer's edges, ticks, halves, the HALT wake and the schedule

The control checkpoint emits the M-cycles to the frame sequencer's next edge, the drop a
DIV reset causes and the HALT wake bit, and turns M-cycles into halves itself; the APU
checkpoint takes M-cycles and the speed and keeps the tick's phase and a waiting step;
the host counts down to the scheduled starts by the emitted period
(`docs/control-in-weights.md` §22, `docs/apu-in-weights.md` §14). Control 55.98 M to
62.19 M dense (hidden 896 to 960, intermediate 576 to 640), APU 70.2 M to 72.4 M
(intermediate 896 to 960), the one model 613.6 M unchanged. Digests: apu `bf499b44`,
control `14f16e93`, one model `8642a540` (`ce1c039d` with `GBCFORMER_BITS=1`); cpu
`87bb33f8`, cpu-memory `851a553c` and ppu `e368efef` unchanged.

| Workload | Before (fps, median of 6) | After | Change | Passes a frame (control, APU) |
| - | - | - | - | - |
| text.sh `--weights all --one-model`, interleaved, 6 rounds (3 each order), load 10.6 to 4.8 | 55.55 | 55.7 | +0.3%, within the spread | 216.4, 816.2 both sides |

The table is the clock unit's own A/B, against the build before it, before the line runs.

**Merged, with the line runs and the I/O dispatch.** Control is
1,785 wires, 62.19 M dense and 63,681 nonzeros (from the line runs' 59.98 M and 63,296:
intermediate 576 to 640, hidden 960 unchanged), the APU 72.4 M as above, the one model
628.6 M dense and 0.97 M nonzeros. Digests: control `7fef034b`, apu `bf499b44`, one model
`ba60eff1` (`9642b3ae` with `GBCFORMER_BITS=1`); cpu `87bb33f8`, cpu-memory `096c5372`
(`060e022f`) and ppu `e368efef` unchanged. `cargo test --release` passes in full (SST
500,000 on the circuit and the checkpoint, `tests/oam_dma_runs.rs`, the edge, tick, wake
and DMA mutants), `f16census all` reports 0 violations, and the gates hold: DMG 8
`6872b003`, CGB 300 `cfd31f3d`, the four games with all five in weights and in lockstep
`cfd31f3d` `ccfe7ad3` `e2bd7038` `ea7f86ca`, Yellow 300 as one model `cfd31f3d` with and
without the step oracle (10, 10, 11, 10 passes checked), `dmg_sound` 01-04 with apu and
control `99793856` `881f3fe7` `e494d943` `8e776b75`, and `instr_timing` with cpu and
control Passed. The merged binary has not been timed.

## Host in weights: the merged work

All four units on one branch: the display registers, the MBC3 clock, the DMA transfers
in line runs and blocks, the I/O dispatch, and the clocks, counters and HALT wake.
Digests: cpu `87bb33f8`, cpu-memory `096c5372`, apu `bf499b44`, control `7fef034b`, ppu
`e368efef`, one model `ba60eff1` (628.6 M dense, 0.97 M nonzeros; `GBCFORMER_BITS=1`
`9642b3ae`).

**Gates.** `tools/gate.sh` 31 ok and 0
FAIL, `cargo test --release` included, with SST 500,000 on the circuit and the
checkpoint. The null-cell test ok, `f16census all` 0 T1 and 0 T2, the one-model GGUF
(`--layer-shapes --align 64`) with the compiled digest and 0 tensors differing, and
`GBCFORMER_ORACLE=1000` on Yellow 300 as one model from that GGUF at `cfd31f3d` with
165 + 1 dense checks equal. The stock short run (b12070 with `layer-shapes.patch`,
Metal, 4,000 passes from boot) has 4,001 events equal and 0 differ (285,736 decoded
tokens).

**Speed A/B** against the clock unit's binary (one model, interleaved, two sets of three
rounds; load 8 to 89 from other jobs). text.sh A 59.1, 60.7, 58.0, 57.5, 58.9, 59.0
(median 58.95) and B 59.6, 56.9, 57.5, 58.4, 59.9, 59.3 (58.85, -0.2%); overworld.sh A
133.3, 127.0, 130.8, 132.1, 138.0, 137.5 (132.7) and B 138.0, 131.1, 130.3, 137.0, 136.2,
129.2 (133.65, +0.7%). Every run reached `870aa3fd` or `822059c1`. Flat; text sits near
59 fps on this loaded machine on both sides.

## Hardware accuracy (docs/hardware-tests.md)

Outside test ROMs judged by their own signals, and the fixes they drove: the DMG wave RAM
rule (reference and APU checkpoint), CGB compatibility mode (reference and PPU
checkpoint), boot ROM execution from a file you supply, LY's early increment (reference and
control checkpoint), a serial cable, and mode-3 length, the VRAM and OAM locks and
ordinary STOP behind `--accuracy` until the control checkpoint has them. Digests: cpu
`87bb33f8`, cpu-memory `096c5372` (both unchanged), apu `fcba971a` (84.5 M dense and 152,034 nonzeros by
`shape --apu`, from 72.4 M and 137,587; the engine re-evaluates 1,239 units a pass where
it did 1,062), control `89b07a8d`, ppu `8b7fc08b` (252.9 M, unchanged
dense, 0.37 M nonzeros), one model `08d8f497` (632.9 M dense, from 628.6 M).

**Hash moved.** Zelda DX CGB 300 with all five in weights goes from `e2bd7038` to
`f10b2b4a`: the game reads LY in a line's last M-cycle, where LY now reads the next line
(daid's `speed_switch_timing_ly` is pixel-exact with it, blargg `oam_bug` 1 passes).
Every other gate hash holds; Zelda DX 1,500 lockstep keeps `99f51381`.

**Gates**: `tools/gate.sh` 35 ok and 0 FAIL, including `cargo test`, the
speed-switch screenshots now all pixel-exact, and the new boot section (Nintendo's CGB
boot ROM and SameBoy's two, run with the GBC CPU, APU and control in weights). The
one-model GGUF (`--layer-shapes --align 64`) loads with the compiled digest and 0 values
differing.

**Speed A/B** against the host-in-weights work before this merge (a later commit
changed only labels), interleaved, three rounds each, load 4 to 16 from other jobs.
- text.sh (`--core model`; the GBC CPU path has not changed since): A 113.6,
  113.5, 112.5 (median 113.5), B 111.2, 112.8, 109.9 (111.2): -2.0%, ranges almost
  touching. Every run reached `870aa3fd`.
- text.sh as one model (`--weights all --one-model`): A 59.7, 60.1, 60.6 (60.1), B 60.4,
  60.4, 60.8 (60.4): no measurable difference.
- Yellow 300 with all five in weights: A 132.2, 129.6, 137.8 fps (132.2), B 125.7, 125.1,
  130.5 (125.7): -4.9%. The PPU's line goes from 25.99, 26.82, 23.88 µs (25.99) to 29.52,
  30.10, 27.64 (29.52), +14%: the compatibility palette head and its 8 steps per output
  bit run at every position of every line, in every mode. A first version that looked the
  shade up per colour and register cost 58.5 µs per line (93.5 fps); computing the shade
  beside the pixel brought it here. Every run reached `cfd31f3d`.

## Host rules

Four units on top of the hardware-accuracy work (`docs/host-in-weights.md`
§5). Saves, observer reads of RAM and the shadow copy come from the model's memory. Every
control event runs over the span before it. The inference core sits apart behind a
`gbc` feature, with the checkpoint's metadata in the GGUF. Control emits `HR`, `DD` and
`CL`. The first three leave every digest at the base's: control `89b07a8d`, one model
`54cb5b10` (`GBCFORMER_BITS=1` `f99b3ec5`). The fourth gives control `5788496a` and one
model `2d125e4e` (`b13ecd53`), 647.9 M dense and 1.04 M nonzeros, with three more
control output slots. cpu `87bb33f8`, cpu-memory `096c5372` (`060e022f`), apu
`a625dcc9` and ppu `cfc69538` are unchanged.

**Gates.** `tools/gate.sh` 32 ok, with `cargo
test` failing on two tests that fail at the base too:
`machine::tests::a_speed_switch_follows_the_documented_timings` (LY at tick 113 of line
144) and `tests/stream_audit.rs`'s APU slot count (110 against 108). `f16census all`
gives 0 violations. The GGUF (`--layer-shapes --align 64`) has the compiled digest with
0 values differing. Yellow 300 as one model from the GGUF alone (`GBCFORMER_GGUF_ONLY`,
no circuit compiled) reaches `cfd31f3d`. The stock short run (b12070 with
`layer-shapes.patch`, Metal, 4,000 passes from boot) has 4,001 events equal and 0
differ (288,314 decoded tokens). With every component in weights as one model and no
lockstep, Yellow, Crystal, Zelda DX and Mario DX reach `cfd31f3d`, `ccfe7ad3`,
`f10b2b4a` and `ea7f86ca` over 300 frames, with no access OAM DMA blocked and no
palette data port read.

| Workload | base (fps, median of 6) | host rules | Change | Control passes |
| - | - | - | - | - |
| text.sh `--weights all --one-model`, interleaved A B B A, 3 rounds, load 4.7 to 19.5 | 55.0 (58.1, 55.2, 54.8, 54.8, 54.6, 55.3) | 55.15 (56.8, 56.6, 55.4, 54.7, 54.9, 54.3) | +0.3%, within the spread | 46,375 and 46,373 |

Running every control event over its span cost no passes: an in-place pass was a pass
already, and the span it now carries changes only its inputs. Every run reached
`870aa3fd`.

**Merged with the hardware-accuracy work** (its APU at 84.5 M and its PPU change). Digests:
apu `fcba971a`, control `5788496a`, ppu `8b7fc08b`, one model `a78e1b63` (632.9 M dense,
0.99 M nonzeros; `GBCFORMER_BITS=1` `71480330`); cpu `87bb33f8` and cpu-memory
`096c5372` (`060e022f`) unchanged. `tools/gate.sh`
34 ok and 0 FAIL, with the inference core's build alone among them; `cargo test --release
--no-fail-fast` passes in all 19 test binaries; `f16census all` 0 violations; the GGUF
with the compiled digest and 0 values differing; Yellow 300 from the GGUF alone at
`cfd31f3d`; the four games as one model without lockstep at `cfd31f3d`, `ccfe7ad3`,
`f10b2b4a` and `ea7f86ca`, with no blocked access and no palette data read; the stock short run 4,001
events equal and 0 differ (288,314 decoded tokens). The two supplied boot ROM lines of
`gate.sh` were skipped (`cgb_boot.bin` was not linked in this worktree) and ran apart:
Yellow 400 from the CGB boot ROM at `0db60d9b` and dmg-acid2 through it at `9c7eee04`.

## The memory checkpoint's accesses at their M-cycle (U3)

The memory circuit translates each data access under the bus state at its own M-cycle
(`docs/hardware-tests.md` §4.1). The host predicts that state from the control model's
spans and its new `DY` output, and asks again only when an access's state differs from
the step's start. Measured with all five components in weights, against the build
(U2 plus the host rules' DMA-line fix):

| Workload | Steps | Steps asked again | Memory passes (result passes) | Control passes per frame | Lookahead control passes |
| - | - | - | - | - | - |
| Yellow 300, `--lockstep all`, before | 902,091 | none | 11,401 result | 215.2 | none |
| Yellow 300, `--lockstep all`, U3 | 902,091 | 0 | 11,401 result | 215.2 | 314 (1.05 per frame) |
| text.sh, before | 3,208,966 | none | 3,248,876 (14,800 result) | 77.3 | none |
| text.sh, U3 | 3,208,966 | 0 | 3,248,876 (14,800 result) | 77.3 | 763 (1.27 per frame) |

A lookahead pass is a control pass over the span to an access's M-cycle whose state is
dropped. It is not in the pass count. The first build ran a pass in place before it, which put text.sh, split or as one model, at 78.6 control passes per frame. 307 of Yellow's 312 in-place passes still left the access past the spans, so it was dropped. Crystal 300 asks 204 steps again, and Zelda DX 66.
Every run reached its old hash: Yellow `cfd31f3d` (one model too), Crystal `ccfe7ad3`,
Zelda DX `f10b2b4a`, Mario DX `ea7f86ca`, text.sh `870aa3fd`. No timing A/B was run.

Digests: cpu-memory `65bc34e9` (186.5 M dense, 0.23 M nonzeros), control `b38390a6`
(88.2 M, 0.24 M), one model `59dea7e8` (788.8 M, 1.07 M). `GBCFORMER_BITS=1` gives
cpu-memory `633b5fcf` and one model `629bddb4`. apu `131ef0af` and ppu `caa99f33` are
the inputs' (U3 moves neither). cpu is `bfd4c294`, moved by U2's `RC`/`WC` outputs.
`src/sm83/changes-memory.txt` was not re-recorded for the two new input slots.

## Host rules: the boot ROM in weights, the length stream, and text's remaining gap

Workload: text.sh and overworld.sh with every component in weights as one model. A is
the merged tip, B the commit named in each row. Rows are medians of six interleaved runs (three
A B B A rounds) taken through the heavy-lock queue, with other unlocked work
running alongside; load is shown.

| B | Workload | A (fps) | B (fps) | Change | Load |
| - | - | - | - | - | - |
| boot as a bank-0 image (DMA inputs held, tables re-recorded) | text | 58.7 | 53.3 | −9.2% (one round) | 3-7 |
| boot as a bank-0 image | overworld | 130.9 | 107.8 | −17.6% (one round) | 3-4 |
| the length stream | text | 57.0 | 53.1 | −6.8% | 9-11 |
| the length stream | overworld | 118.4 | 115.5 | −2.4% | 17-46 |

What moved, by mechanism:
- **The boot ROM.** The first design carved the boot ranges out of bank 0 by address. That
  cost 35 gate updates on every GBC CPU pass, boot or not (text 830 → 870 → 905 units a
  pass). As a 16 KB bank image behind MAP0 it costs nothing after FF50.
- **The DMA line inputs.** A GBC CPU step zeroed DSL..DDH, so each DMA line's pass and the
  step after it re-ran the DP and DQ translations: 36 gate updates a pass on average. Now
  0. `dma_inputs_reach_only_the_dma_slots` proves the inputs reach only DP and DQ.
- **The mode-3 length.** The control logic asks for a line's length at every pass past
  M-cycle 63 of a visible line while STAT's mode-0 interrupt is on. In text those were
  FF00 reads (5,692 asks), FF44 (3,157) and FF41 (499). In overworld they were DIV reads
  (1,068 asks, 1,107 ms) and FF00 (4,852). Each waited for the PPU thread's backlog,
  about 5.7 lines even at a batch of 1, because halts issue lines in bursts. The length
  stream (`ppu_circuit::circuit::build_length`, `Part::Length`) answers at hand-over in
  0.95 µs a line. The waits went 0.29 s → 0.6 ms (text) and 1.2 s → 0 (overworld).
- **The hurry hint**, which flushed the PPU's batch on pass-drawn lines, measured nothing
  and is gone.

**Text's remaining 0.79 s per 600 frames (stats against the merged tip), for the
speed work.** The GBC CPU's memory pass is 876 units against 830, +0.42 s. Gate updates a
pass by node family against the merged tip, from `GBCFORMER_UNITS=1 CORE=model-mem` on
text.sh:

| Family | Updates a pass | Rule |
| - | - | - |
| WD0, WD1 | 14 | second write ports: 5 equality steps over the write's 22 physical bits |
| WSP1, WSP2, WHL, WFFN (+3 each) and WRARE0 (+2) | 14 | KEY0/FF50 register cases (7 of them, measured by removing the cases) and the write paths' other growth |
| LKQ | 6 | the GBC CPU's access locks (the lock query) |
| F0, FN | 5 | flag outputs' growth since the merged tip |
| FP0-2, FPpc | 5 | fetch ports |

The rest:
- **Control** is +0.11 s: 944 → 1,021 units and 71 → 82 decoded positions since
  hardware accuracy. Layer 0 rises 1.17 → 2.22 µs a pass from the late `ML` input's replans
  (0.34 a pass). After U3, control is 16 layers.
- **The length stream** is +0.08 s.
- **Host time** is about 0.16 s, unattributed. A 5 s `sample` of the GBC CPU thread found
  85% in the memory core's engine passes and no host frame above 2%.

Folding WD into the existing translation cases is worth at most its 14 updates, about
0.04 s. The merged tip itself runs text at 57 fps, so 60 fps needs an engine-wide speedup,
not more trimming of rules.

## Control: the step boundary decides when an HDMA block lands (review P2)

The host used to hold every `Copy::Block` back while a GBC CPU step ran. The control
logic now decides it: a new input `IN` (a step runs) and `CL` bit 7 (this pass's block
lands at the step's end), `Signals::held` on the host (`docs/hardware-tests.md` §3.9).
Every game kept its hash with all five components in weights and in lockstep: Yellow
`cfd31f3d` (one model too), Crystal `ccfe7ad3`, Zelda DX `f10b2b4a`, Mario DX `ea7f86ca`.
Digests: control `35082739` (91.2 M dense, 0.24 M nonzeros; was `b38390a6`, 88.2 M), one
model `cdb93ad1` (794.0 M, 1.08 M; was `59dea7e8`, 788.8 M), and with `GBCFORMER_BITS=1`
the one model `aa873c39` (was `629bddb4`). cpu, cpu-memory, apu and ppu are unchanged. The control stream gains a slot (142) and a fallback gather (9).

## The wire-change tables re-recorded for the access timing memory circuit

`src/sm83/changes-memory.txt` and `changes-memory-bits.txt` re-recorded with
`GBCFORMER_UNITS=1 GBCFORMER_WIRE_CHANGES=<file>` on text.sh and overworld.sh,
`CORE=model-mem`, and averaged per wire. The tables now cover `AS0`, `AS1`, `RC` and
`WC`: 1,283 wires (was 1,204) and 1,500 in the 0/1 circuit (was 1,422). The fit places
every wire as before: cpu-memory stays `65bc34e9` (`633b5fcf` with `GBCFORMER_BITS=1`)
and the one model `cdb93ad1` (`aa873c39`).

## The minimal runner at the full program's speed

The runner (`tools/runner`, `docs/minimal-runner.md`) runs the same passes as the full
program (equal pass records), so its gap was in how it drives them. Text and overworld
benches, the runner from the bench's hand-off snapshot, interleaved through the heavy
queue, medians of three; the load column is each bench's range.

| Change (runner) | Load | text full / runner | overworld full / runner |
| - | - | - | - |
| four streams, before | 8-15 | 50.0 / 37.4 | 100.9 / 61.0 |
| all of the below but QoS | 35-88 | 48.0 / 12.2 | 88.3 / 29.1 |
| all but the non-blocking audio drain | 35-88 | 48.0 / 43.7 | 88.3 / 72.8 |
| all but batched worker calls | 35-88 | 48.0 / 42.7 | 88.3 / 84.3 |
| QoS, drain, batching | 35-88 | 48.0 / 44.7 | 88.3 / 96.9 |
| + no per-step allocation or name lookup | 34-113 | 49.7 / 46.5 | 105.5 / 101.5 |
| + reused outcome buffers, Copy request, merged advances | 24-55 | 49.3 / 48.9 | 112.4 / 109.9 |

The first three rows after the baseline leave one change out of the full set. QoS dominates: the
runner's main thread waits in under 1 % of its samples, so its speed is the core time macOS
gives it, and at the default class a loaded machine gives it a fraction. The last row is
within 1 % (text) and 2 % (overworld) of the full program, inside the runner's own
round-to-round spread; it is not shown faster. Profiles (`sample`, 15 s on text) put 89.5 %
of the full program's main-thread samples and 84.6 % of the runner's (before the last two
rows) in the engine; the difference was the runner's per-step allocations and name lookups.

## Review fixes P2, P5 and P4: text.sh as one model

`tools/bench/text.sh --weights cpu,memory,apu,ppu,control --one-model`, no lockstep,
the base against the head (P2 with its stall, P5 with its guard, P4's test),
interleaved base then head, three pairs, each pair one `heavy-queue.sh` ticket. The
one-minute load was 12 to 19 (other work).

| Side | Emulated fps, runs | Median | Steps | Memory result passes | Control passes |
| - | - | - | - | - | - |
| base | 38.8, 48.4, 47.9 | 47.9 | 3,208,966 | 14,800 | 46,400 |
| head | 47.5, 46.8, 47.8 | 47.5 | 3,208,966 | 14,800 | 46,400 |

No measurable difference: the medians differ by less than the base's range. Both sides
did the same work, with equal steps and passes, and every run reached `870aa3fd`. Staging
the records moves when each `keep` runs, not how many records are kept. The limiter was
not profiled for this pair.

## GBC CPU, OAM DMA, timer, serial and LY timing (A2)

The rules of `docs/hardware-tests.md` §3.11, in the reference and in the control, memory
and GBC CPU checkpoints. Their state and slot changes:

- Control: a 32nd state byte `FL` (bit 0 the DMA restart, bit 1 TIMA's reload M-cycle),
  `ST` bit 2 (STAT's LY=LYC bit, held while the LCD is off), and two outputs, `DB` (OAM
  DMA's buses) and `DZ` (the M-cycles before a fresh transfer blocks). `DI` counts 254
  and 255 before byte 0.
- Memory: a 25th input slot `DB`, the transfer's buses and those blocked at the step's
  start; each data access's translation reads its own bus's blocking, and the FF page
  translations no longer read OAM DMA.
- GBC CPU: the dispatch's vector after the high byte's push (`SP` 0 and `IF & PCH`), HALT
  with IME on and an interrupt pending leaving PC on the HALT, and the operand count in
  `RC` and `WC` bit 7. A core that takes its bytes from the bus reads operand k at
  M-cycle k.

Every lockstep game holds its hash with all five in weights, each equal to the
reference's at 300 frames, with no dense fallback: Yellow `cfd31f3d` (one model too),
Crystal `ccfe7ad3`, Zelda DX `f10b2b4a`, Mario DX `ea7f86ca`. Gate 3 holds with the CPU
checkpoint on the host bus (`6872b003` DMG 8, `cfd31f3d` CGB 300), and so do text.sh
(`870aa3fd`) and overworld.sh (`822059c1`). No timed A/B was run.

Digests: cpu `fac7fab8` (68.3 M dense, 0.09 M nonzeros), cpu-memory `f99ec3c5` (187.5 M,
0.23 M), control `72889c63` (94.4 M, 0.25 M), one model `d9816438` (796.1 M, 1.09 M);
apu `131ef0af` and ppu `caa99f33` unchanged. `GBCFORMER_BITS=1`: cpu `f816d219`,
cpu-memory `dae7e314`, one model `0032d771`. `src/sm83/changes-memory.txt` and
`changes-memory-bits.txt` were not re-recorded for the `DB` input.

Merged with the access timing work (the runner and the host rules), the digests are cpu
`fac7fab8` (unchanged), cpu-memory `e22c7272` (187.5 M, 0.24 M), control `16c8ed35`
(94.4 M, 0.25 M) and one model `2011aa71` (919.9 M, 1.17 M, five streams);
`GBCFORMER_BITS=1`: cpu-memory `5cbe314d`, one model `14c7af62`. The shared access loop
(`transport::settle_accesses`) no longer marks write 1 busy when write 0 writes FF46. The
runner feeds the memory stream's `DB` from the control stream's last pass
(`wiring::Part::Emitted`) and blocks from `DZ` (`Spans::busy_from`), as the full
program does. The four games and Yellow as one model hold the hashes above, with no
dense fallback, and so do gate 3 and both benches. The reference census holds at 213 of
322.

Merged again with the access timing work (the 24-bit layout and its 21-bit line key):
cpu-memory `b6571351`, ppu `b9696877` (the layout's), one model `a2d2d764` (919.9 M,
1.17 M); cpu `fac7fab8` and control `16c8ed35` as above. `GBCFORMER_BITS=1`: cpu-memory
`d3816de4`, one model `b92ddb13`.

Operand fetches at their M-cycles in the memory checkpoint (`docs/hardware-tests.md`
§3.11): a 26th input `FS`, two fetch slots in the shared access loop, the next line's
translation unblocked and each operand's columns held off by its own state's buses. The
six fetch ROMs (twelve cases) go from FAIL to PASS with the GBC CPU, memory, PPU and
control in weights and in lockstep. The cost is in the nibble memory circuit: cpu-memory
`97752cae` goes from 187.5 to 192.4 M dense (intermediate 1,536 to 1,600 as the fit
lands at 1,216 units on three levels instead of 1,078 on five; 257 more tokens for the
new slot; hidden 1,856, depth 10 and 20 heads unchanged), the one model `8772a661` from
919.9 to 927.2 M. The 0/1 circuit gets smaller: cpu-memory `a135b934` from 196.2 to
190.0 M (hidden 1,792 to 1,728), the one model `a33e67b1`. cpu, apu, control and ppu
are unchanged. The four lockstep games and Yellow as one model hold their hashes with
all five in weights, and so do gate 3, text.sh and overworld.sh.

## The runner merge in llama.cpp: 4,000 passes

The runner merged into the access timing work, with the held HDMA stall carried
into the shared transport (`Signals::held_stall`). The one model is `dc902b72` (911.5 M
dense, 1.16 M nonzeros). Its GGUF (`--gguf-only --layer-shapes --align 64`) gives the
compiled digest with 0 values differing. The stock short run used llama.cpp b12070 with
Metal, `STOCK_GPU_LAYERS=99 STOCK_GROUP=own`, `--lanes 255` and 4,000 passes from boot.
It was held to the engine's record (`stock-llama play none --frames 3`). The result was
**4,001 events equal, 0 differ, 300,738 decoded tokens**. That is 2,624 GBC CPU events,
92 PPU, 46 length, 244 APU and 995 control. Play took 27.5 min of wall time, unlocked at
`nice -n 19` with a load of 99 to 146. It is a correctness run, so it took no heavy-queue
ticket.

## MBC5 at 8 MiB: the 24-bit layout

**Why.** mooneye `emulator-only/mbc5/rom_32Mb` and `rom_64Mb` (DMG) were the census's
two `REFUSED` cases. MBC5 addresses 8 MiB of ROM with a 9-bit bank number, and the
22-bit layout held only 2 MiB.

**What changed.**

- **The layout.** `memory::layout` puts ROM at 0..8 MiB. Every region above it moves up
  by `0x60_0000` (CRAM `0x80_0000` ... BOOT `0x84_0000`). The low bits are unchanged,
  and the region bit is 23 instead of 21, so the translation's bit tables change only
  in that bit. `ADDR_BITS` is 24.
- **The ninth bank bit.** A fifteenth register, `MASK1`, holds the ROM mask's bit 8 and
  is written at load. MBC5's bank bit 8 comes from `ROMB1` bit 0 under it.
- **The key.** The line key is 21 bits, against 19 before.
  - Pair 12 turns too far at the new `KEY_REACH` (8,667,138 positions). Bits 0..17 sit
    on the first halves of RoPE pairs 13..30, and bits 18..20 on the second halves of
    pairs 29..31, which turn by less than 10⁻²⁷ rad.
  - `key_pairs_turn_by_nothing` checks two things at five positions up to `KEY_REACH`.
    Every roped query and key entry on a key dim must equal its unroped value bit for
    bit. Every unused partner must stay within the product bound.
- **The query scales.** With 21 bits, a column select's 3S on every bit would reach
  66,560 at S = 1,024, which is past f16's range. `Params::memory()` takes S = 512 and
  B = 1,024: the widest logit is then 33,280, and the gaps are 128 logits against
  f32's 104.
- **Per-cartridge state.** Cells are per cartridge already (`Image::has_line`), so a
  small ROM's prefill is unchanged: Yellow inserts 143,830 cells on both sides.
  - A stock runtime's position P (`READ_AT = LINES`) grows from `0x48800` to
    `0x108800`. `stock-llama play` sized its context by P, so llama.cpp failed to
    allocate a 156 GB compute buffer for 1,088,496 cells. It now sizes the context by
    the cells the cartridge's image holds plus the palettes' lines (Yellow: 143,830 plus
    17), with the same room for lane records and a group as before. The cells keep
    their lines' positions up to P: llama.cpp limits a context's cell count, not its
    positions.

**Digests.**

| Checkpoint | Before | After | Dense, nonzeros |
| - | - | - | - |
| cpu-memory | `bf75311f` | `828e6e78` | 186.5 M, 0.23 M (unchanged) |
| ppu | `caa99f33` | `b9696877` | 278.4 M, 0.40 M (unchanged) |
| one model | `dc902b72` | `73b7b9d9` | 911.5 M, 1.16 M (unchanged) |
| cpu-memory, `GBCFORMER_BITS=1` | `4d3ce37f` | `e42c490a` | 177.8 M to 188.3 M, 0.28 M |
| one model, `GBCFORMER_BITS=1` | `eaf5ba8b` | `922413f2` | 972.4 M, 1.21 M |

cpu, apu and control are unchanged (`bfd4c294`, `131ef0af`, `0d5b559c`). The cpu-memory,
ppu and one-model digests move because every memory head's key and query rows move.

**Measured cost: none measurable.**

- **Work.** `GBCFORMER_STATS=1` on text.sh as one model gives identical units per pass
  on every engine: GBC CPU 880, 68 and 38; APU 2,637; control 1,124 and 154; PPU 42,346.
- **The GBC CPU and memory alone.** text.sh `--weights cpu,memory` with stats, four
  interleaved pairs, each one `heavy-queue.sh` ticket, load 41 to 43. Per pass, base
  3.39, 3.27, 3.25 and 3.22 µs (median 3.26) against head 3.35, 3.32, 3.16 and 3.16
  (median 3.24). The fps medians were 48.15 and 48.5.
- **Five as one model.** text.sh `--weights cpu,memory,apu,ppu,control --one-model`, no
  lockstep, ten interleaved pairs, every run at `870aa3fd`.

| Pairs | Load | Base fps | Head fps |
| - | - | - | - |
| 1 to 6 (base first in 1 to 3) | 47 to 112 | 42.3, 50.4, 47.1, 49.2, 47.0, 48.7 | 35.3, 49.0, 41.2, 47.6, 44.3, 47.2 |
| 7 to 10 (alternating) | 49 to 58 | 48.9, 43.6, 48.0, 47.6 | 48.4, 44.1, 47.5, 47.4 |
| median of 10 | | 47.8 | 47.3 |

The first six pairs ran as the machine's load swung from 112 to 47. A stats run in
those minutes slowed the APU, whose circuit did not change, by 19% at equal units. The
four pairs at steady load differ by −0.5 to +0.5 fps. The key width is therefore a
fixed property of the layout, not a per-cartridge choice: two compiled widths would buy
nothing measurable.

**Gates.**

- **The two refused cases.** With `cpu,memory,ppu,control` in weights and all in
  lockstep, all eight mooneye mbc5 ROMs pass on DMG (`hwtests --filter mbc5`, Fibonacci
  at frame 14), the 4 MiB and 8 MiB ROMs among them.
- **Yellow.** Yellow 300 reaches `cfd31f3d` with all five in lockstep, and again as one
  model.
- **llama.cpp.** The stock short run on `73b7b9d9` used the same settings as
  the runner merge's. Its GGUF has 0 values differing. The result was **4,001 events equal,
  0 differ, 300,738 decoded tokens**: 2,624 GBC CPU events, 92 PPU, 46 length, 244 APU
  and 995 control. Play took 25.9 min at a load of 42 to 58, unlocked.

The digests above are the 24-bit layout's, against the runner merge. A2's merge on top
moves cpu-memory and the one model again, for its own reasons (`b6571351`
and `a2d2d764` as its merge records).

## battle.sh's frame hash: 14d7468e to 9a57a127

battle.sh's final frame moved twice upstream of the PPU FIFO, and neither move was
recorded. Measured with `tools/bench/battle.sh --weights none` and the builds already in
the worktrees, at load 60 to 100:

| Build | Frame hash |
| - | - |
| main (2026-11-24) | `14d7468e` |
| host in weights, merged | `14d7468e` |
| hardware accuracy (the hardware rules: STOP, mode-3 length, locks, LY's early increment) | `40e6a1f8` |
| access timing (every access at its M-cycle) | `9a57a127` |
| host rules, PPU FIFO | `9a57a127` |

The hardware accuracy work is one commit with every hardware rule, so which of its rules moved the frame
is not separated; the variable mode-3 length and LY's early increment are the likely ones,
as LY's early increment moved Zelda DX's CGB 300 frame (`docs/hardware-tests.md` §3.8).
That is inferred, not traced. The second move came with access timing, between the hardware rules
and access timing. Both changes are behaviour the outside tests asked for, so `9a57a127` is the
gate hash now (`CLAUDE.md`). The PPU FIFO moves no frame: the PPU FIFO and its base give the
same hash.

## PPU FIFO step 1: the checkpoint's quiet lines follow the FIFO

Workload: text.sh and overworld.sh with every component in weights as one model
(`--weights all --one-model`). A is the host rules, B the PPU FIFO (the scanline
and length circuits' X 0 and window-grid length, CGB objects fetched with OBJ_EN clear, the
labelled mode-3 refusal and the run's counts). Six runs a side, interleaved A B B A over
three rounds, through the heavy-lock queue with other work alongside (load 73 to
120, contended).

| Workload | A median (fps) | B median (fps) | Change | Frame |
| - | - | - | - | - |
| text.sh | 48.05 | 48.25 | +0.4% | `870aa3fd` |
| overworld.sh | 110.15 | 109.45 | -0.6% | `822059c1` |

Both are inside the runs' spread (text 46.4 to 49.4, overworld 104.6 to 112.7): a tie. The
step adds no pass and no per-pass work on the CPU's thread; the PPU checkpoint's circuit
grew in its shared stages only. Digests: ppu `da338c86` (299.9 M dense, 0.40 M nonzeros),
one model `b4ec41d1` (967.3 M, 1.17 M); cpu, cpu-memory, apu and control unchanged. In
stock llama.cpp b12070 Metal, Yellow ran 4,000 passes with this GGUF against the engine's
record: 4,001 events equal, 0 differing, 300,553 decoded tokens compared.

On the merged tip the battery holds: `tools/gate.sh` gives 42 ok and 0 FAIL,
`f16census all` gives 0 violations, and `digest --gguf` gives 0 differing. The stock
short run on `a2d2d764` (919.9 M dense, 1.17 M nonzeros), with the same settings, has
**4,001 events equal, 0 differ, 303,723 decoded tokens**. Play took 27.4 min at a load
of 67 to 96.

## The operand-fetch merge in llama.cpp: 4,000 passes

A2's memory half of the operand-fetch timing gives the one model `8772a661`
(927.2 M dense, 1.18 M nonzeros). Its GGUF (`--gguf-only --layer-shapes --align 64`)
gives the compiled digest with 0 values differing. The stock short run used the
settings of the runner merge's and the 24-bit layout's: b12070 Metal, `STOCK_GPU_LAYERS=99
STOCK_GROUP=own`, `--lanes 255`, 4,000 passes from boot, held to the engine's record.
The result was **4,001 events equal, 0 differ, 303,723 decoded tokens**: 2,624 GBC CPU
events, 92 PPU, 46 length, 244 APU and 995 control. Play took 25.5 min at a load of 25
to 69, unlocked at `nice -n 19`.

The MBC5 entry's cost rests on the GBC CPU's thread. The PPU runs on its own thread,
with pool helpers. Text.sh as one model with stats, four interleaved pairs (each a
`heavy-queue.sh` ticket, load 21 to 35) and a fifth pair while a stock build ran:

| | Base, the runner merge | Head, the 24-bit layout |
| - | - | - |
| PPU µs per line | 46.2, 67.5, 91.7, 46.1, 101.5 | 85.4, 85.3, 105.1, 118.7, 77.1 |
| APU µs per pass (no change to its circuit) | 5.72, 6.54, 8.38, 5.82 | 7.13, 7.34, 8.12, 8.92 |

The PPU's units are equal (42,346 a pass). Its time swings by 2x within one side, and
the APU, whose weights did not move, slows on the same runs. A 6 s `sample` of each
side's model frames shows the same functions in the same order and proportions
(`Marks::blocks`, the pool, `Positional::advance_part`, `Engine::advance`), with no new
frame in the head. The run that was sampled, the fifth pair, has the head faster (77.1
against 101.5 µs a line) at equal fps (35.8 and 35.3). Contention between the PPU's
helpers and the other work is the likely limiter. That is inferred, not
profiled under a quiet machine. The fps of the four pairs differs by 0.7 to 3.9 in the
base's favour.

An 8 MiB cartridge holds 1,057,238 cells (`rom_64Mb`, 2,977 MB of f16 cache in a stock
runtime). That is about the context that failed to allocate before the play fix.
llama.cpp on this machine has run no cartridge that large. Its compute buffer grows
with the context, so it would likely fail the same way (inferred).

## PPU FIFO: the window's edge cases in the checkpoint

No speed measurement: this records the digests and sizes the change moves
(`docs/ppu-in-weights.md` §14.1a). The line circuit and the length circuit compute WX 0 to
6 and 166, the DMG's pixel inserted at WX with the window off, the WX 166 carry (one more
FLG bit, `WAR'` bit 1) and the CGB's WX glitch at WX 0 and 247 to 252.

| Checkpoint | Before | After |
| - | - | - |
| ppu | `84e854e6`, 14 layers, hidden 1792, 299.9 M dense, 354,578 nonzeros | `6f77b44e`, 14 layers, hidden 1984, 337.3 M, 377,005 |
| ppu length | 13 layers, 98.7 M, 139,344 nonzeros | 14 layers, 105.7 M, 148,926 |
| one model | `13878ff8` | `7902814e`, 1,118.1 M, 1.21 M nonzeros |

cpu, cpu-memory, apu and control were not compared against a run of the base in this
run; no file in their build changed. f16census `all`: 0 violations.

## The runner and access timing merge, on the host rules: the one-model A/B

`tools/bench/text.sh` and `overworld.sh` with `--weights all --one-model`, A, the host rules,
against B, the merge, three rounds of A B B A per workload, each round one heavy-queue job.
The verify of the next integration tip was building at the same time, so the load ran
from 31 to 181 (contended). Every run reached its hash (`870aa3fd`, `822059c1`) with 0
dense fallbacks.

| Workload | A, host rules (per-round means) | B, the merge | Median A | Median B | Change |
|---|---|---|---:|---:|---:|
| text | 48.7, 49.5, 48.6 | 47.65, 47.7, 47.35 | 48.7 | 47.65 | −2.2% |
| overworld | 99.25, 106.75, 101.05 | 102.25, 103.85, 101.0 | 101.05 | 102.25 | +1.2% |

B is slower on text in every round, by 1.0 to 1.8 fps. That is small but consistent.
Overworld moves within its round-to-round noise (99 to 109 fps). The cause of the text
difference is not measured. The candidates are the per-access timing B brings (each data
access at its M-cycle, the held HDMA blocks) and the shared transport. The access timing work's
own A/B (47.9 against 47.5) measured only part of that range.
Speed does not block merges in this phase, so this is recorded only.

## The APU as SameBoy's port: what the accuracy costs

The APU checkpoint now follows SameBoy v1.0.5's `Core/apu.c` (`docs/apu-in-weights.md`
§15), and SameSuite's 63 APU cases pass in weights under lockstep. The circuit grew from
depth 12 (72.4 M dense, 137,587 nonzeros, 13 layers) to depth 24 (168.3 M, 194,786, 25
layers). One pass costs more, and a game makes a few more passes, because the frame
sequencer's rises are now events.

Measured on this change against the host rules (A), median of 3, interleaved, with the
machine shared with other jobs (load given per pair). Every run kept its frame hash
(`870aa3fd`, `822059c1`) and had 0 passes on the dense fallback.

| Workload | Load | A | B |
| - | - | - | - |
| text.sh, one model, all five in weights | 11-81 | 49.8 fps | 49.4 fps |
| overworld.sh, one model, all five in weights | 11-81 | 111.8 fps | 105.4 fps |
| text.sh, `--weights apu`, APU on the GBC CPU's thread | 111-152 | 216.0 fps | 145.9 fps |
| overworld.sh, `--weights apu`, APU on the GBC CPU's thread | 111-152 | 217.2 fps | 156.8 fps |

On text the one-model difference is within the spread of the runs (A 49.6 to 50.5, B
48.5 to 49.8). On overworld B's spread (93.4 to 112.2) covers A's median. The APU runs on
its own thread there, so its passes mostly stay off the GBC CPU's critical path. Alone
and inline, the APU is 1.39 times slower (`GBCFORMER_STATS=1`, overworld):

| | A | B |
| - | - | - |
| Passes | 488,599 | 504,028 |
| Units a pass | 2,846 | 3,154 |
| (changed channel, reading layer) pairs a pass | 172.8 | 240.9 |
| µs a pass: layer 0, shared layers, decode | 0.82, 3.73, 0.14 | 0.87, 5.53, 0.16 |
| µs a pass in all | 4.73 | 6.60 |

The shared layers take the difference: 25 layers against 13 and 40% more changed pairs.

### Depth: what running the events beside the cycles reached

This change gave a frame-sequencer event a segment of its own with no cycles, so the events
and the cycles both read the post-write state instead of running one after the other.
The aim was a shallower circuit. The depth went from 23 to 24 (24 to 25 layers), not
down:

- The events left the critical path, but the cycles still read the write's result,
  so the path is still write, then cycles.
- Two stages now sit on that path. The envelope steps a counted fall at double speed
  leaves waiting run before the cycles (`events::waiting`). A mux per field picks the
  event's state or the cycles' state.

The gain of that commit was a cheaper pass, not a shallower one. The duty bit is computed
once, write-only values read gated copies, and the noise counter is block-summed.

The one model is 1,333.0 M dense with the APU at 168.3 M and 25 layers, against about
910 M before the port. I have not measured each stage's depth separately, so the split
of the 24 levels among write, waiting stage, cycles and merge is not known yet. Trimming
is left to the optimization phase.

## The FIFO step 1 merge (access timing and PPU FIFO step 1 on the runner and access timing merge): the one-model A/B

This is the same method as the A/B above, with A the runner and access timing merge and B the FIFO step 1 merge.
The verify of the APU-port merge was building and running beside it, at load 17 to 51
(contended). Every run reached its hash (`870aa3fd`, `822059c1`) with 0 dense fallbacks.

| Workload | A, the earlier merge (per-round means) | B, the FIFO step 1 merge | Median A | Median B | Change |
|---|---|---|---:|---:|---:|
| text | 43.4, 45.25, 44.95 | 44.95, 43.85, 44.4 | 44.95 | 44.4 | −1.2% |
| overworld | 87.8, 93.35, 94.2 | 78.9, 93.55, 91.7 | 93.35 | 91.7 | −1.8% |

On text, B is lower in two rounds of three, and A's first round holds a 41.5 outlier.
Overworld's first round ran both of B's runs at 78.9 under the same load as A's 82.4 and
93.2. Both changes are inside the spread between rounds, so the data shows no clear
regression. Two possible costs are known but not profiled: the PPU checkpoint grew from
278.4M to 299.9M dense, with a length circuit of depth 12 instead of 10 (FIFO step 1),
and every GBC CPU step gained the `FS` access byte and two fetch slots (operand fetches).
A's text median here (44.95) is below its own 47.65 in the earlier A/B. Equal binaries
moving 6% between runs bounds what one contended A/B can resolve.

## The APU-port merge (the APU as SameBoy's port, on the FIFO step 1 merge): the one-model A/B

This is the same method as the A/B above, with A the FIFO step 1 merge and B the APU-port merge. It ran
after the APU-port merge's verify finished, at load 24 to 83 (contended; the overworld rounds ran
at 65 to 83). Every run reached its hash (`870aa3fd`, `822059c1`) with 0 dense fallbacks.

| Workload | A, the FIFO step 1 merge (per-round means) | B, the APU-port merge | Median A | Median B | Change |
|---|---|---|---:|---:|---:|
| text | 47.65, 47.5, 46.75 | 46.95, 45.6, 48.15 | 47.5 | 46.95 | −1.2% |
| overworld | 109.25, 110.4, 109.85 | 109.75, 109.3, 109.25 | 109.85 | 109.3 | −0.5% |

Both changes are inside the spread between rounds: B leads one text round of three and
one overworld round of three. The APU checkpoint grew (one model 985.2M to 1,387.2M
dense), but the APU runs on its own thread and these workloads do not wait on it, so the
GBC CPU's thread sets the rate (inferred, not profiled). The same FIFO step 1 binary gave
44.4 text and 91.7 overworld in the previous A/B and 47.5 and 109.85 here, under higher
load. Equal binaries moving up to 20% between runs again bounds what one contended
A/B can resolve; the cause of that swing is not traced.

## The transport's decisions as model outputs

Workload: text.sh and overworld.sh with every component in weights as one model
(`--weights all --one-model`). A is the base (the runner with
the integration of that day), B is this work's code (the stop owed as a total with `SQ` in, `NE`,
`RT`, the run's tie bit, the clock's binary digits, every pass's copies landed at once, a
cut always a pass). Six runs a side, interleaved A B B A over three rounds, through the
heavy-lock queue with other work alongside (load 51 to 119, contended).

| Workload | A median (fps) | B median (fps) | Change | Frame |
| - | - | - | - | - |
| text.sh | 48.35 | 48.45 | +0.2% | `870aa3fd` |
| overworld.sh | 105.75 | 106.25 | +0.5% | `822059c1` |

Both are inside the runs' spread (text 45.9 to 48.7, overworld 96.6 to 110.7): a tie. The
passes did not change where both sides ran the same case. With every component in weights
and in lockstep, Yellow, Zelda DX and Mario DX 300, Yellow 300 from the boot ROM, daid's
`stop_instr` on DMG and CGB and `oam_dma_restart` ran the same control passes on A and B
(Yellow 64,744, Zelda DX 114,306, Mario DX 66,021, from the boot ROM 610,331, STOP 20,889 and
20,546, `oam_dma_restart` 30,324 with one cut), every frame equal. Crystal ran 142,907 and
142,908: its clock read the host's wall clock (no `GBCFORMER_TIME`), so where a second
fell differed between the runs; with the scripted clock it is equal (below). The runner
proofs' five pass records have the same line counts on A and B for host-fetch, both
`cgbpal_m3` ROMs, `oam_dma_restart`, `oam_dma_start`, `oam_dma_timing` and
`oam_dma/basic`, the cases A ran before its verify was stopped. The clock's digits cost nothing
on a clock that ticks once a second: Crystal 600 frames with `GBCFORMER_TIME=1700000000`
hands the wall time over 9 times and runs 308,450 control passes on both. Digests: control
`615bc5e3` (90.7 M dense, from `16c8ed35` at 94.4 M: six fewer inputs), one model `5c21d371`
(977.3 M dense, 1.17 M nonzeros, from `3de7574a` at 977.4 M, 1.19 M); cpu, cpu-memory, apu
and ppu unchanged.

## Operand fetches take the access locks: the fetch mask in the model

`transport.rs` gave the operand fetches only OAM DMA's bit (`if locks { all } else { busy }`),
so the memory checkpoint fetched each operand under the step-start locks while the
reference reads it at its M-cycle (SameBoy v1.0.5 `cycle_read`, no exception for PC). The
memory circuit now reads `FS` bits 0 and 1, holds an operand's columns off where its own
locks hold its line (`F{k}L`), and translates `FN`, `FP1` and `FP2` unlocked. WC bit 3 is the
operands' lock flag (it was "write 0 starts OAM DMA", which no host read), and
`transport::Asked::masks` carries each access's bits. With WC bit 3 set and the host's
locks lapsed, the host asks the control logic for them before the opcode fetch.

Digests: cpu-memory `36dcca3f` to `ce478a0e` and the one model `f80b39d9` to `8b7b1b1d`
(both 192.4 M and 1391.0 M dense, unchanged nonzeros); with `GBCFORMER_BITS=1` cpu-memory
`93dc9a17` to `719e23aa` and the one model `f2fb030e` to `f0c1520f` (1474.9 M to 1432.8 M
dense). cpu, apu, control and ppu unchanged. `f16census all`: 0 violations. The GGUF
gives the compiled digest, 0 values differing. llama.cpp b12070 Metal, 4,000 passes from
boot held to the engine's record: 4,001 events equal, 0 differ, 310,383 decoded tokens.

Mooneye acceptance in weights (`cpu,memory,ppu,control`, lockstep all), against the base
against the branch: 222 pass and 16 fail on both, every line equal, no guard stop. Yellow,
Crystal, Zelda DX and Mario DX 300 with all five in lockstep and Yellow as one model hold
`cfd31f3d`, `ccfe7ad3`, `f10b2b4a` (base equal), `ea7f86ca` and `cfd31f3d`. Not timed: a
step pays an extra request only when PC's line or the next is VRAM or the FE page.

## The APU's segment end in the model

At double speed a counted frame-sequencer fall leaves the envelope steps for the next
M-cycle's start, and the segment of that M-cycle must end after it. Both cutters
(`apu_circuit::ApuModel` and the runner's `segments.rs`) decided that end by a rule of
their own (a counted fall at the fast speed), which also cut where no step waits: with the APU off, or at a skipped event. The APU pass now emits `SE`, the M-cycles of the
next segment after which it ends: `pend'` bit 1, so 1 exactly where steps wait and 0
elsewhere, with no new node (the APU circuit keeps 3,009 nodes, depth 24). Both programs
count `SD` and `SE` down in `transport::SegmentClock`. `ApuLockstep` holds `SE` to the
reference's `steps_wait` at every M-cycle; a mutant emitting no end and one emitting the
old host rule are both caught by that comparison (`the_lockstep_catches_an_end_the_model_did_not_emit`).

Digests, against the base: apu `65a54694` (from `063ba0cb`; 168.3 M dense,
0.39 M nonzeros, unchanged), one model `94063c04` (from `f80b39d9`; 1,391.0 M dense,
1.25 M nonzeros); cpu `fac7fab8`, cpu-memory `36dcca3f`, control `dc3d5f53` and ppu
`c3a13d11` unchanged. The APU stream has 138 slots (`tests/stream_audit.rs`, from 137).
No perf A/B was run (phase 1: correctness only). Passes with every component in weights,
in lockstep, as one model: Yellow 300 251,674 APU engine passes (with and without `--one-model`), Zelda DX 300 254,132.
In Zelda DX 300 (the engine's APU record, read through a FIFO), 2,563 of 254,132 passes
emitted `SE` = 1 and the rest 0, so the double-speed end is exercised on a real run.

**Checks** (a frozen detached worktree). SameSuite APU
63 of 63 applicable with `--weights apu --lockstep apu`. blargg sound 26 of 26 applicable
(2 whole ROMs, 24 singles) the same way. dmg_sound 01 to 04 with
`apu,control` in weights and `--lockstep all` at `99793856`, `881f3fe7`, `e494d943`,
`8e776b75`. Yellow 300 with `--weights all --lockstep all` at `cfd31f3d`, with and without
`--one-model`, and Zelda DX 300 as one model at `f10b2b4a`, 0 dense fallbacks on every
stream. The runner's proof on Zelda DX 300 (double speed) from the GGUF: five pass records
equal (APU 254,133 lines), frames, audio and save equal, no dense fallback. `digest --gguf`
0 values differing; `f16census all --model` 0 violations.
`cargo test --release`: 163 library tests and all 19 integration test binaries pass (SST and
the checkpoint's SST among them); the runner's tests pass except `every_branch_is_reviewed`,
whose one unreviewed line is access timing's (`let mask = if locks ...`), not this work's.
In stock llama.cpp b12070 Metal (`STOCK_GROUP=own`, under the GPU lock, load 48), Yellow ran
4,000 passes with this GGUF against the engine's record: 4,001 events equal, 0 differ,
310,636 decoded tokens compared (APU 253 events, 70 decoded tokens each with `SE`; Yellow's first 4,000 passes run at normal speed, so llama.cpp decoded `SE` only as 0).

## The APU's event split and halves in the model

Two more host rules of the APU's segment cutter, found by the audit above, move into the
model, and one is shown to be the model's already.

- **An event's segment.** Both cutters ran each frame-sequencer event as a pass of its own
  with no M-cycles, so the host decided that the events precede their M-cycle's cycles.
  The event now opens a segment that takes the M-cycles after it, like a write. Its pass
  runs the write and the event only: every node that counts cycles (`n.b`, `D>0`) folds
  `− fall − rise` into its own threshold, so they read D as 0 at no extra level, and the
  sample clock reads gated copies of D and the sample flag. The pass emits `DL` = D, the
  M-cycles it left, which `transport::Cutter` counts again with the countdowns that pass
  emitted, and then delivers the close that ran it once. The other choice, a circuit that
  runs the event and then the cycles in series, would save one pass per event (about 5,100
  of Yellow's 251,674) at the cost of the events phase's depth on every pass; the split
  keeps the pass count and depth 24.
- **A change of speed.** Both cutters ran an extra pass at each change of speed, because
  `SD` counted M-cycles at the last pass's speed. A half (2 dots) adds 375 units to the
  sample clock at either speed, so `SD` now counts halves (the double-speed formula, one
  branch fewer), and the host counts it down by each M-cycle's halves, the rate the
  control logic emits (`Signals::rate`, the runner's `spans.rate()`), until reached or
  passed, as the frame time is.
- **Where an edge lands.** Already the model's: control emits each edge's distance (`FE`,
  `FR`, the drop), `Spans::advance` reports an edge only at the cycle the count reaches, and
  `ControlLockstep` catches an edge one M-cycle late or early. No change.

`transport::Cutter` is now the one cutter both programs run (`ApuModel` and the runner's
segments stream); the runner's copy of the cut is gone. `ApuLockstep` holds the M-cycles
each emitted end falls after to those after which the reference's `steps_wait` holds, by
cycle index, since the reference is ahead of the model while a segment is open. Mutants
caught: no end, the old end rule, and no cycles left (`the_lockstep_catches_cycles_the_model_did_not_leave`);
`an_event_s_pass_leaves_its_cycles` checks the split under `CircuitSegments`, and
`the_cycles_a_pass_leaves_are_counted_again` and `a_left_sample_is_reached_again` check
the cutter against a scripted model.

The APU circuit: 3,016 nodes (from 3,009), depth 24, 168.3 M dense, 0.39 M nonzeros. The
APU stream has 139 slots (from 138). Digests: apu `d8d3a06e` (from `65a54694`), one model
`01d787da` (from `94063c04`); cpu `fac7fab8`, cpu-memory `36dcca3f`, control `dc3d5f53` and
ppu `c3a13d11` unchanged.

**Pass counts** (APU engine passes, every component in weights and in lockstep, 300
frames; before is the previous commit): Yellow 251,674 before and after, with and without
`--one-model`; Zelda DX as one model 254,132 before, 254,131 after: the one speed switch's
extra pass is gone. The runner proof's APU record on Zelda DX: 254,133 lines before,
254,132 after.

**Checks** (a frozen detached worktree, load 2 to 60).
SameSuite APU 63 of 63 and blargg sound 26 of 26 applicable with `--weights apu --lockstep
apu`. dmg_sound 01 to 04 with `apu,control` and `--lockstep all` at `99793856`,
`881f3fe7`, `e494d943`, `8e776b75`. Yellow 300 at `cfd31f3d` with and without
`--one-model`, Zelda DX 300 as one model at `f10b2b4a`, 0 dense fallbacks. The runner's proof
on Zelda DX 300: five records, frames, audio and save equal, no fallback. `cargo test
--release`: 165 library tests and every integration binary pass. The runner's tests pass
but `every_branch_is_reviewed`, whose one open line is access timing's. `digest --gguf` 0
differ; `f16census all --model` 0 violations. Stock llama.cpp b12070 Metal, 4,000 passes of
Yellow against the engine's record: 4,001 events equal, 0 differ, 310,889 decoded tokens
compared.

## The step's lock asks in one transport function (after the APU cut merge)

Two lock asks open a step. The first applies when WC bit 3 is set (PC's line or the next
reaches VRAM or the FE page) and the host's copy of the locks has lapsed: a pass gives the
locks and the step asks again. The second is RQ bit 7, a pass for the locks the request
asks for. Each was written in `src/step.rs` and again in the runner's scheduler, and only
the full program's copy was tested. Both programs now call `transport::open_step` through
the `Opening` trait, and share `access_mask` (the busy bit always, the locks by the
request's own flag) and `start_state`. Mutants, each red:
- the lapsed branch removed from `open_step`: `tests/fetch_locks.rs` fails on DMG and
  CGB at frame 0;
- the runner's `Opening::fetch_reads_locks` returning false: the runner's proof on the
  fetch-locks ROM (`examples/fetch_locks_rom.rs`, the test's program) fails on both
  consoles, with the steps record at 232,873 lines against 233,129.

The memory circuits' wire-change tables were re-recorded on this circuit (text.sh and
overworld.sh, `CORE=model-mem`, averaged per wire): 1,306 wires (was 1,283) and 1,523 in
the 0/1 circuit (was 1,500), now covering `F1N`, `F2N` and `DB`. The fit places every wire
as before. Digests: cpu `fac7fab8`, cpu-memory `ce478a0e`, apu `65a54694`,
control `dc3d5f53`, ppu `c3a13d11`, and the one model `b2a06e13` (1,391.0 M dense, 1.25 M
nonzeros). The one model combines this work's cpu-memory with the APU cut's apu (`8b7b1b1d`
here before the merge, `94063c04` on the APU cut). With `GBCFORMER_BITS=1` the digests are
cpu-memory `719e23aa` and one model `e2627efe` (1,432.8 M dense).

**Checks** (a frozen detached worktree):
- `digest --gguf`: 0 values differing.
- `f16census all`: 0 violations.
- The runner's tests with `GBCFORMER_GGUF` set pass, `every_branch_is_reviewed` among them.
- The runner's proof passes on four runs, with five records equal, frames and audio equal,
  and 0 dense fallbacks: Yellow 300 (`cfd31f3d`), `oam_dma_restart` 120 (`7b7dbe09`), and
  fetch-locks 24 on DMG (`2817dca1`) and on CGB (`d043ab24`).
- Yellow 300 with `--weights all --lockstep all` reaches `cfd31f3d`, with and without
  `--one-model`, with 0 dense fallbacks on every stream.
- Mooneye acceptance with `--weights all --lockstep all`, three shards: 111 pass, 8 fail
  and 31 NA. Every line equals the previous census.
- `cargo test --release --no-fail-fast` (frozen, the code as at the checks above): 23 test
  binaries, 220 passed, 0 failed, 1 ignored.
- `tools/gate.sh`: 42 ok, 0 FAIL, 0 skip, as at the APU-port merge's verify (DMG 8
  `6872b003`, CGB 300 `cfd31f3d`, the four games with all five in weights, the one model
  with the step oracle, the boot ROMs and the runner's four proofs).
- The census, 448 cases each: in reference 296 pass, 26 fail; with `--weights all
  --lockstep all` (three shards) 273 pass, 49 fail; 0 refused. Every line equals the
  APU-port merge verify's census.

llama.cpp on this machine has run no cartridge that large. Measured since: at the
recipe's 255 lanes its compute buffers ask 152 GB and Metal refuses them; at 32 or 64
lanes the context opens, and the 64-lane prefill ran out of GPU memory at 92% after
3.9 hours, so no 8 MiB pass has run (`docs/stock-llama.md` §12a, "An 8 MiB
cartridge").

## The 8 MiB layout's high key bits under the oracles

Line key bits 18 and 19 sit on the second halves of RoPE pairs 29 and 30, and only ROM
past 2 MiB sets them. Bit 20 sits on pair 31's second half and is the region bit of
every line above ROM, so every RAM, register and VOID cell sets it, Yellow's included.
The step oracle now counts, per run, the fast passes that read ROM past 2 MiB and the
checked passes that read a line with each high bit set. It also takes `rom=N`, which
checks every Nth pass that reads ROM past 2 MiB besides the uniform cadence: a uniform
1-in-15,000 sample would land on none of them.

Both runs used mooneye `emulator-only/mbc5` on DMG, 16 frames, `--weights
cpu,memory,ppu,control --lockstep all`, unlocked at `nice -n 19`, load 120 to 140.

| ROM | Cells | `GBCFORMER_STEP_ORACLE`, `GBCFORMER_ORACLE` | Step checks (s each) | On bit 18, 19, 20 | Record checks | Wall |
| - | - | - | - | - | - | - |
| `rom_32Mb` (4 MiB) | 532,950 | `cpu=15000,rom=16`, 2000 | 26 (42.7) | 16, 0, 26 | 8 | 19.2 min |
| `rom_64Mb` (8 MiB) | 1,057,238 | `cpu=30000,rom=32`, 4000 | 13 (91.1) | 8, 0, 13 | 4 | 20.6 min |

Every checked token equals the dense pass's. Both runs reach frame `002271da`, the
mooneye pass screen, with 142,857 GBC CPU passes and no dense fallback. The
counters' build holds the GBC CPU gates: `--core lockstep` reaches `6872b003` (DMG, 8
frames) and `cfd31f3d` (CGB, 300 frames).

- **Bit 18.** Both ROMs read ROM past 2 MiB in 256 GBC CPU passes, and each of those
  reads sets bit 18. The checks cover 16 of them on `rom_32Mb` and 8 on `rom_64Mb`.
- **Bit 19: no live coverage.** No pass of either ROM reads ROM past 4 MiB, and
  `rom_64Mb` at 60 frames reads none either (393,996 steps). The cause is the test
  itself. mooneye's `common/harness/mbc5_rom.s` (upstream `605e3a07`) loops the bank
  number from 0x000 while it is below 0x100 and writes `ROMB1 = high byte | 0xFE`, so
  it selects banks 0 to 255 on every ROM size although `rom_64Mb.s` lists 512 expected
  banks. The engine's record agrees: over its 158,145 GBC CPU passes, every head read of
  the REGS line holds ROMB1 at 0 or 254 (bit 0 clear) and MASK1 at 1. Both ROMs therefore run the
  same 136,508 steps to the same frame. MBC5's bank bit 8 is held by
  `circuit_translation_equals_the_specification` (circuit against `layout::translate`,
  random ROMB1 at ROMs up to 8 MiB), by `fixed_layout_translation_matches_the_host_on_every_mapper`,
  which now includes a 512-bank MBC5 cartridge and selects bank 0x123 (specification
  against the host at every address), and by the host's `mbc5_ninth_bank_bit`. The
  key's bit 19 is held by `reader_records_read_the_cache_through_the_model`, whose
  dense pass now reads bytes at `0x20_0005`, `0x40_0003` and `0x7F_FFFE` against the
  byte array, through the copy heads; no test reads it through the GBC CPU's read
  heads. No run with the model in lockstep selects bank 256 or above, so the mbc5
  gate's `rom_64Mb` pass covers ROM up to 4 MiB only.
- **Bit 20.** Every check reads a cell with bit 20 set (VOID, the line a column head
  off its word line reads, is one).
- **The record oracle.** It checks written lines and copy readers. Nothing writes ROM,
  and neither ROM runs a DMA from ROM, so 0 of its checks touch bit 18 or 19 (counted:
  0 copies from ROM past 2 MiB).

The previous run of `rom_64Mb` (`cpu=3000`, `GBCFORMER_ORACLE=50`, its
binary from before A2's operand fetches, a 187.5 M memory checkpoint) finished
cleanly: frame `002271da`, 48 step checks and 306 record checks, every one equal. Its
binary has no high-bit counter, so it says nothing about bits 18 and 19.

## The wire-change tables re-recorded for the 24-bit layout and the operand fetches

`src/sm83/changes-memory.txt` and `changes-memory-bits.txt` re-recorded with
`GBCFORMER_UNITS=1 GBCFORMER_WIRE_CHANGES=<file>` on text.sh and overworld.sh,
`CORE=model-mem`, and averaged per wire (a wire one workload lacks counts 0 there).
Text.sh reached `870aa3fd` and overworld.sh `822059c1` in both circuits. The tables now
name the 24-bit layout's address wires (`WP0.24` and `RHL.23` where they were `WP0.22`
and `RHL.21`) and A2's `FS`, `DB`, `F1N`, `F2N` and `irq.ie*` wires: 1,302 wires (was
1,283) and 1,519 in the 0/1 circuit (was 1,500). The fit places every wire as before.
No digest moves: cpu `fac7fab8`, cpu-memory `97752cae`, apu `131ef0af`, control
`16c8ed35`, ppu `b9696877` and the one model `8772a661`; with `GBCFORMER_BITS=1` cpu
`f816d219`, cpu-memory `a135b934` and the one model `a33e67b1`. With no checkpoint
moved, the battery has nothing new to hold.

## A3: the HALT wake, LCD-on, SCX rounding, HDMA, MBC1M, line 144 and FF72-FF75

`docs/hardware-tests.md` §3.12 has the rules and their sources. They move five digests:

| Checkpoint | Before (A2) | After | Cause |
|---|---|---|---|
| cpu | `fac7fab8` | `76c89995` | the wake's `MC` case gone (no M-cycle of its own) |
| cpu-memory | `97752cae` | `3ed3a3d6` | the same, MBC1M's cases, VBK in compatibility mode, FF72-FF75's port class |
| control | `16c8ed35` | `2a7e69d5` | the wake's wait (`FL` bit 2), line 0 after LCD-on (`ST` bit 1), HDMA in mode 0, line 144's STAT on CGB, FF72-FF75 (`UR0`-`UR2`, `FL` bits 4-6) |
| ppu | `b9696877` | `58c49913` | SCX's fine scroll rounded up in `ML` |
| one-model | `8772a661` (927.2 M, 1.18 M nonzeros) | `b0da307c` (930.4 M, 1.19 M nonzeros) | all of the above |

The APU's `131ef0af` holds. With `GBCFORMER_BITS=1`: cpu `db09696f`, cpu-memory
`583e6fcf`, one-model `e54e024f`. `f16census all` gives 0 violations. The one model's
GGUF (`--gguf-only --layer-shapes --align 64`) gives the compiled digest with 0 values
differing. The control stream's depth holds at 16 layers. Its slots go from 146 to 152
and its fallback gathers from 8 to 10 (`tests/stream_audit.rs`).

No gate hash moves: Yellow `cfd31f3d`, Crystal `ccfe7ad3`, Zelda DX `f10b2b4a`, Mario DX
`ea7f86ca` and Yellow as one model, 300 frames with all five in weights and in lockstep.

The stock short run used the settings of the operand-fetch merge's: b12070 Metal, `STOCK_GPU_LAYERS=99
STOCK_GROUP=own`, `--lanes 255`, 4,000 passes from boot, held to the engine's record.
It gave **4,001 events equal, 0 differ, 306,708 decoded tokens**: 2,624 GBC CPU events,
92 PPU, 46 length, 244 APU and 995 control. Play took 44 min at a load of 12 to 30,
unlocked at `nice -n 19`. A first attempt at 14:12 stopped with Metal out of memory while the
in-weights census ran beside it and other jobs shared the GPU (the cause inferred,
not traced); the rerun, the same GGUF and settings, is the result.

The hardware-test census is 224 of 322 in the reference (`after-a3.tsv`) and in weights
(`in-weights-a3.tsv`, the GBC CPU, memory, PPU and control in weights and in lockstep),
case for case equal. gbmicrotest on the DMG reference goes from 363 to 386 of 513.

## battle.sh's frame since `14d7468e` (bisected on the reference)

The FIFO step 1 merge reached `d3534543` with `tools/bench/battle.sh
--weights all --lockstep all`, which no doc recorded. The docs held `14d7468e`, and
the PPU FIFO's perf-log held `9a57a127`. Each candidate below was built at its commit and
ran battle.sh with `--weights none` (the reference alone):

| Commit | Frame | What it changed |
|---|---|---|
| the start of access timing | `40e6a1f8` | moved before access timing; not bisected |
| access timing on by default | `9a57a127` | access timing on by default; the mode-2 source and double-speed reads retuned |
| the runner merge and A2's first commits | `9a57a127` | no change |
| A2's HALT-wake change | `6370aaa5` | HALT leaving in the step it wakes, mode 2 at the line start (both reverted by the next row), serial's end, LY=LYC with the LCD off, boot state |
| A2's HALT-wake fix and A2's merge | `d3534543` | HALT's wake cycle and mode 2's early rise back; line 153's LY and LY=LYC; the reload-cycle and DMA-restart fixes |

The rule that moves it is line 153's LY: LY reads 0 through line 153, on a CGB from its
second M-cycle (`Lcd::read`; gbmicrotest `line_153_ly_*`). On the HALT-wake fix
each of its rules and the HALT-wake change's was undone alone, and battle.sh's reference frame
measured:

| Undone at the HALT-wake fix | Frame |
|---|---|
| LY reading 0 on line 153 | `9a57a127` |
| a serial transfer ending at the counter's eighth bit-8 edge | `d3534543` |
| SC 0x7E after the CGB boot ROM | `d3534543` |
| HALT with IME on and an interrupt pending staying on the HALT | `d3534543` |
| the LY=LYC comparison's line 153 and last-M-cycle positions | `d3534543` |
| line 153's last M-cycle reading mode 0 | `d3534543` |
| OAM DMA's restart kept through the two start M-cycles | `d3534543` |

So line 153's LY alone moves `9a57a127` to `d3534543`: the battle reads LY in line 153.
The HALT-wake change's `6370aaa5` comes from its HALT wake and mode-2 rise, which the fix
put back, so that frame never reached a merge.
Before them, access timing on by default moved it from `40e6a1f8` to `9a57a127`. The move
from `14d7468e` to `40e6a1f8` predates access timing.

On A3's tip (with the access timing work merged) the reference alone gives
`d3534543`. `--core model` gives the same. All five components in weights and in
lockstep as one model give the same too, with 0 dense fallbacks. text.sh `870aa3fd`
and overworld.sh `822059c1` hold on the reference alone.

### A3's A/B: all five in weights as one model, the start of access timing against A3

`tools/bench/text.sh` and `overworld.sh` with `--weights cpu,memory,apu,ppu,control
--one-model`, three interleaved pairs in one `heavy-queue.sh` ticket (16:22, load 14 to
27, with A3's in-weights census running three jobs beside it, so contended):

| Workload | Base | Head | Median change |
|---|---|---|---|
| text.sh (`870aa3fd` both) | 50.2, 50.1, 50.4 | 48.4, 48.4, 46.7 | 50.2 → 48.4 fps, −3.6% |
| overworld.sh (`822059c1` both) | 96.1, 99.9, 92.0 | 111.4, 111.0, 108.4 | 96.1 → 111.0 fps, +15.5% |

The ranges do not overlap on either workload. The base is the start of access timing,
so the change covers everything since then (U3's accesses at their M-cycle, A2's
operand fetches and timing rules, the 24-bit layout and A3), not A3 alone. One
`GBCFORMER_STATS=1` overworld run per side shows the same work: the same components in
weights, 1,212,011 against 1,212,135 GBC CPU passes, 488,594 against 488,599 APU passes,
and 37,233 against 38,465 control passes. The base's control passes wait 1,580.6 ms in
all for the renderer's mode-3 length; the head's wait 0.43 ms. That gap is most of the
wall-time difference (6.65 s against 5.88 s), so the overworld gain is the length
answered without waiting (measured on one pair, under the same contention). The text
slowdown is not traced.

## PPU FIFO green: the stepper, the judge and the merges

The pixel FIFO build (the pixel FIFO, the judge in the length
checkpoint, the stepper as the one model's sixth stream, the window cases, the
APU-port merge and A3) plus a test fix and docs.
`docs/ppu-in-weights.md` §14.2b and §14.2c have the design.

**What the verify of the pixel FIFO build measured.** gate.sh passed every line but `cargo test`.
The one failing test was `machine::ppu::tests::the_fifo_draws_a_quiet_line_as_the_line_renderer_did`:
it compared the FIFO's unrounded dots with `Ppu::mode3_dots`, which A3 rounds SCX's fine
scroll up to 4 dots in, and failed on 285 of 3,000 lines. It compares
`fifo::mode0_dots` of the FIFO's dots now. The one model's 67,061 tokens put
`REC`, `CELL`, the reader tokens and the upper families above 65,535; the engine's 65,536
limit applies to channels only since the pixel FIFO build. `the_cpu_stream_keeps_records_whose_tokens_are_above_65535`
runs the GBC CPU stream on SST's `LD (HL),A` (1,000 cases) with the record oracle and the
step oracle on its dense pass. `cargo test --release` after the test fix passes every binary.
The commit changes tests only, so the rest of the pixel FIFO build's verify carries over:

- digests: cpu `76c89995`, cpu-memory `ef29face`, apu `063ba0cb`, control `63002b2e`,
  ppu `3db883bb`, ppu-stepper `26205bab`, one-model `f70df183`; `GBCFORMER_BITS=1` cpu
  `db09696f`, cpu-memory `1a75e61c`, one-model `dc4524c4`. `digest --gguf` gives the
  compiled one-model digest with 0 values differing. `f16census all`: 0 violations.
- every gate.sh line other than `cargo test`, among them Yellow, Crystal, Zelda DX and Mario
  DX 300 with all five in lockstep, Yellow as one model with the step oracle, the boot
  ROM runs and the minimal runner.
- the three benches with all five in weights and in lockstep at `870aa3fd`, `822059c1`
  and `d3534543`, 0 dense fallbacks.
- llama.cpp b12070 Metal, 4,000 passes each: Yellow from boot, 4,001 events equal, 0
  differ; Mealybug `m3_bgp_change`, 3,999 equal, 0 differ, 3,943 of them stepper passes.
- the census at 310 of 322 in the reference and in weights, case for case
  (`docs/hardware-tests.md` §2.2).

**Sizes.**

| Checkpoint | Dense | Nonzeros | Shape |
|---|---:|---:|---|
| ppu (scanline) | 337.3M | 0.43M | 14 layers, hidden 1,984, intermediate 2,880 |
| ppu stepper | 482.9M | 0.62M | 9 layers, hidden 2,176, intermediate 5,312 |
| one model | 3,178.5M | 1.84M | 25 layers, hidden 2,752, intermediate 10,624, 69 heads, 67,061 tokens, 746 positions |
| one model on the APU-port merge | 1,387.2M | 1.26M | 25 layers, hidden 2,176, intermediate 5,376, 44 heads, 45,131 tokens, 576 positions |

The stepper's stream is what widens the one model. Its 9 layers overlay the one model's
first 9, and its units add to theirs: those layers go from 1,472-5,376 units to
2,560-10,624, and the widest, layer 6, from 5,248 to 10,624 (`export --layer-shapes`
on both). Hidden goes from 2,176 to 2,752. This is left for the optimization phase.

**Stepped lines.** Lines the stepper drew in each bench's 600 model frames, with all
five in lockstep: text 5 (4,385 passes, 141 ms), overworld 0, battle 11 (9,757 passes,
297 ms). `run`'s `stepped_lines` field reads 8, 20 and 31: it counts the machine's marked
lines from frame 0, so it includes the reference's warmup frames.

### The A/B: all five in weights as one model, the APU-port merge against the pixel FIFO build (contended; superseded by the quiet A/B below)

`tools/bench/text.sh`, `overworld.sh` and `battle.sh` with `--weights all --one-model`,
three rounds of A B B A per workload, in one `heavy-queue.sh` ticket from 07:45. A is
the APU-port merge's verified binary, B the pixel FIFO build
(the test fix changes no code outside tests). Other jobs ran beside it, so the load
went from 9 to 64 (contended). Every run reached its hash (`870aa3fd`, `822059c1`,
`d3534543`) with no nonzero dense-fallback count.

| Workload | A, APU-port merge (per-round means) | B, pixel FIFO build | Median A | Median B | Change |
|---|---|---|---:|---:|---:|
| text | 48.4, 47.75, 48.4 | 45.7, 47.05, 47.25 | 48.4 | 47.05 | −2.8% |
| overworld | 108.85, 111.9, 112.0 | 110.8, 110.6, 106.75 | 111.9 | 110.6 | −1.2% |
| battle | 68.7, 68.35, 64.2 | 64.2, 65.75, 63.8 | 68.35 | 64.2 | −6.1% |

B is slower in every round on text, by 0.7 to 2.7 fps, and on battle, by 0.4 to 4.5 fps;
battle's round 3 is the closest because A's second run there fell to 60.7 at load 64. Overworld moves inside its
round-to-round spread. The cause is not profiled. Two candidates are measured only in
part. First, the stepper's own time: 141 ms of text's run and 297 ms of battle's, in the
lockstep runs above, against 15.05 s and 13.12 s of wall time in those runs, so 0.9% and 2.3% if it
all waits on the GBC CPU's thread (inferred). Second, the judge's pass per display write on
the machine's thread (217 writes on text, 166 on battle, 0.6 µs a pass), which is too
small to explain it. Speed does not block merges in this phase, so this is recorded only.

### The quieter A/B: the same comparison with no other job in the queue

The same three benches with `--weights all --one-model`, three rounds of A B B A per
workload, in one `heavy-queue.sh` ticket from 08:34 to 09:19, every run under `nice -n 19`.
A is the APU-port merge's verified binary, B the build
that `cargo test` produced (the tip adds docs only). The ticket waited for the
1-minute load to drop below 6 before the first run. The load ranged from 2 to 14: 3.5 to 8
for most runs, 14 for overworld's round 3 (Arq's backup and Rex, outside the queue), 2 to 3
for battle's last rounds. Every run exited 0, reached its hash (`870aa3fd`, `822059c1`,
`d3534543`), printed its dense-fallback lines (8 for A, 9 for B, the ninth the stepper's)
and none was nonzero. In emulated fps:

| Workload | A runs | B runs | Median of 6, A (range) | Median of 6, B (range) | Change |
|---|---|---|---:|---:|---:|
| text | 49.3, 49.9, 49.6, 50.2, 50.4, 50.3 | 48.8, 48.7, 48.6, 48.7, 49.4, 49.4 | 50.05 (49.3-50.4) | 48.75 (48.6-49.4) | −2.6% |
| overworld | 116.5, 113.4, 119.3, 108.9, 111.0, 113.7 | 114.1, 113.4, 114.0, 112.7, 114.1, 111.2 | 113.55 (108.9-119.3) | 113.7 (111.2-114.1) | none measurable |
| battle | 72.3, 72.8, 71.0, 73.0, 73.0, 72.7 | 67.6, 67.4, 66.7, 69.8, 69.7, 69.6 | 72.75 (71.0-73.0) | 68.6 (66.7-69.8) | −5.7% |

The median of the 3 per-round means gives the same picture: text 49.9 against 48.75
(−2.3%), overworld 114.1 against 113.35 (−0.7%, inside A's 9% spread), battle 72.55
against 68.25 (−5.9%). On text and battle the ranges do not overlap, and B is slower in
every round. The contended table above had the same sign and size.

One `GBCFORMER_STATS=1` run per side per workload followed in the same ticket (not in the
medians; the text pair ran at 49.3 and 49.2 fps, so it shows no gap). Per stream, A then B:

| Stream | text | overworld | battle |
|---|---|---|---|
| GBC CPU, µs per pass | 3.74, 3.75 | 4.19, 4.25 | 3.75, 3.93 |
| APU, µs per pass | 6.75, 6.69 | 6.45, 6.42 | 6.58, 6.57 |
| control, µs per pass (the main pass kind) | 5.96, 6.23 | 5.87, 6.38 | 5.59, 6.07 |
| PPU scanline, µs per line | 33.75, 33.41 | 38.08, 39.96 | 37.41, 37.67 |
| PPU length, µs per pass | 0.98, 1.27 | 0.92, 1.14 | 0.94, 1.18 |
| stepper (B only): lines, passes, time | 5, 4,385, 190 ms | 0, 0, 0 ms | 11, 9,757, 347 ms |

The cause is not resolved. The wall-time gap from the 6-run medians is 0.32 s on text and
0.50 s on battle. On battle the GBC CPU stream's rise (0.18 µs over 2,249,592 passes) is
about 0.40 s, and the stepper's time is 347 ms; both cannot add into 0.50 s, so either the
stepper's time is inside the CPU stream's per-pass figure or it runs off the critical path,
and these runs do not say which. On text the CPU stream is flat (about 0.03 s) beside the
stepper's 190 ms. Control (+0.3 to +0.5 µs over about 40k passes, plus one more pass a
frame) and the length stream (+0.2 to +0.3 µs over about 86k passes) add about 40 ms
together. Overworld, with no stepped line, is consistent with the stepper being the cost
but cannot exclude a gap of text's size. A profile of battle with the stepper's thread
named would settle it; this is left for the optimization phase.
## Integration: the fetch mask merged with the APU cut, verified

The transport rules integration (the transport rules, the APU cut) merged with the fetch mask and
integration's docs; the code differs from the fetch mask by the APU cut's last unit. Both
sides' transport tests are kept (the cutter's and `open_step`'s). One queue ticket, frozen
detached worktree, every step in sequence:

- Digests: cpu `fac7fab8`, cpu-memory `ce478a0e`, apu `d8d3a06e`, control `dc3d5f53`, ppu
  `c3a13d11` (each equal to the side that last moved it), one model `0028a21f` (1,391.0 M
  dense, 1.25 M nonzeros); with `GBCFORMER_BITS=1` cpu `f816d219`, cpu-memory `719e23aa`, one
  model `c13378fe` (1,432.8 M dense, 1.29 M nonzeros). `digest --gguf` 0 values differ;
  `f16census all` 0 violations; Yellow 300 from the GGUF alone `cfd31f3d`, 0 dense fallbacks.
- The runner's ratchet (`every_branch_is_reviewed`) green with no class (c) row; runner
  tests 3, 5 and 15 pass. `cargo test --release --no-fail-fast`: 23 binaries ok, 0 failed
  (SST 500,000 cases on the circuit, bit, memory and bit memory circuits and on both
  checkpoints, 0 failed, 0 dense fallback). `tools/gate.sh` 42 ok, 0 FAIL.
- With every component in weights: the four games 300 as one model and in lockstep at
  `cfd31f3d`, `ccfe7ad3`, `f10b2b4a`, `ea7f86ca`; DMG 8 `6872b003`; text `870aa3fd`,
  overworld `822059c1`, battle `d3534543` in lockstep, 0 refusals; the boot ROM runs
  (`0db60d9b`, `9c7eee04`, SameBoy's `0db60d9b` and `52bc2177`); daid `stop_instr` 3 of 3 in
  lockstep. 0 dense fallbacks on every run.
- The census, 448 cases: 296 pass in reference, 273 with all five in weights and in
  lockstep (49 fail, 0 refused), every pass of the APU-port merge's census kept, none new.

## Integration: the finished pixel FIFO merged onto the integration

The finished pixel FIFO (the pixel FIFO reference, the stepper stream, the length stream's judge,
access timing A3) merged with the integration (the transport rules, the APU cut, the fetch mask).
Where both sides changed one function, each side's change is kept: HDMA's first block in
mode 0 and a cancel's read-back count with the stop owed in `Lines::owed`; A3's HALT-wake
wait (`waking`, FL bit 2) and FF72-FF75 (`UR0`-`UR2`) beside the transport's control slots
(`SQ0`, `SQ1`, `RT`, `NE`); the runner's stepper rows (`Rows`) threaded through `Reach` and
`Landing` beside `open_step`; `proof.sh`'s FIFO records with the stepper's `stream-5-LY`.
The stepper's 36 new branching lines in the runner are listed in `tools/runner/tests/branches.txt`,
all class a.

Sizes and digests: cpu `76c89995` (68.3 M dense, 0.09 M nonzeros),
cpu-memory `4d336582` (192.4 M, 0.24 M), apu `d8d3a06e` (168.3 M, 0.39 M), control `aa9dc582`
(101.4 M, 0.24 M), ppu `3db883bb` (337.3 M, 0.43 M), ppu-stepper `26205bab` (482.9 M, 0.62 M),
one model `69b4f0ef` (3,170.0 M dense, 1.82 M nonzeros, six streams, 65,523 tokens); with
`GBCFORMER_BITS=1` cpu `db09696f`, cpu-memory `1e9ad752`, one model `8bfbc401` (3,209.6 M,
1.87 M). The one model is 8.5 M dense and 1,538 tokens smaller than the finished pixel FIFO's
`f70df183` (3,178.5 M, 67,061 tokens): the transport's rules took six control inputs off
(257 tokens each) and added four outputs. apu and ppu-stepper equal their own sides'.
`tests/stream_audit.rs`: control 150 slots (148 at the base, six off, two and six on),
length 10, stepper 158, control's fallback gathers 8, as measured.

Checks, one queue ticket in a frozen detached worktree, every step in sequence: the
runner's ratchet green with no class (c) row; `digest --gguf` 0 values differ; `f16census all`
0 violations (the stepper depth 8, 2,086 nodes); Yellow 300 from the GGUF alone `cfd31f3d`.
`cargo test --release --no-fail-fast`: 23 binaries ok, 1 failed (below); SST 500,000 cases on
the circuit, bit, memory and bit memory circuits and both checkpoints, 0 failed, 0 dense
fallback. `tools/gate.sh`: 44 ok, 1 FAIL (its `cargo test` line, the same test). With every
component in weights: the four games 300 as one model and in lockstep at `cfd31f3d`,
`ccfe7ad3`, `f10b2b4a`, `ea7f86ca`; DMG 8 `6872b003`; text `870aa3fd`, overworld `822059c1`,
battle `d3534543` in lockstep with 0 refusals; the boot ROM runs at `0db60d9b`, `9c7eee04`
and SameBoy's `0db60d9b`, `52bc2177`; daid `stop_instr` 3 of 3; 0 dense fallbacks on every
run. The census, 448 cases: 310 pass in reference and 310 with all five in weights and in
lockstep (12 fail, 0 refused), every pass of the finished pixel FIFO's census kept and 14 reference
and 37 in-weights passes more than the fetch-mask integration's (the union of both parents, no regression).
The runner's proofs on Yellow from the GGUF, six streams recorded on both sides, every pass
record, frame, sample and save equal, no dense fallback: frame 0 to 300 (`cfd31f3d`), from
power-on to 300 (`0db60d9b`), warmup 2900 to 3100 with presses (`7cb69988`).

The failed test, `the_cpu_stream_keeps_records_whose_tokens_are_above_65535`, stopped at its
premise ("the one model holds 65523 tokens") before its SST, record-oracle and step-oracle
checks. The size is the model's: at most 2^16 tokens no record token exceeds 65,535. The fix
makes the premise an either/or and runs the body in both cases. Rerun alone at the fix (its diff from the merge is that test and CLAUDE.md), one ticket, frozen worktree: `cargo test --release --no-fail-fast` 24 binaries ok, 0 failed, 0 dense fallback; the test reports 65,523 tokens and runs 1,000 cases, 0 failed, 46 fast passes checked by the step oracle, every token equal. Not covered now: a record token above 65,535 (the model has none; one more input slot brings the check back).

## Last 12: the OAM corruption bug in weights, HDMA through a HALT, STAT edges

What changed: the GBC CPU circuits emit the OAM corruption bug's events (reads, writes,
idle M-cycles, fetches), the memory checkpoint runs an OAM pass at each one the control
logic's row read names, the control logic holds an HBlank transfer through a HALT and
passes at the STAT edges a span used to hide (`docs/hardware-tests.md` §3.14-§3.16).

Digests (`verify.sh`, frozen worktree): cpu `188ee397` (68.3 M dense, 0.09 M nonzeros),
cpu-memory `6acacbed` (205.2 M, 0.25 M), apu `d8d3a06e` (168.3 M, 0.39 M), control
`ccd90001` (106.4 M, 0.24 M), ppu `1f3118cc` (337.3 M, 0.43 M), ppu-stepper `75fcd4ef`
(482.9 M, 0.62 M), one model `afed13bc` (3,184.8 M dense, 1.83 M nonzeros); with
`GBCFORMER_BITS=1` cpu `5a9e1d6a`, cpu-memory `2e1f32d7`, one model `e7b08c84` (3,237.6 M,
1.88 M). `f16census`: the GBC CPU circuit at depth 6 (642 nodes), the memory circuit at
depth 10 (1,713 nodes, 7,361 steps).

Checks, one queue ticket: ratchet green, no class (c) row; `digest --gguf` 0 values differ;
Yellow 300 from the GGUF alone `cfd31f3d`; `cargo test --release --no-fail-fast` 24 binaries
ok, SST 500,000 cases on every circuit and both checkpoints with 0 failed and 0 dense
fallback; `tools/gate.sh` 47 ok, 0 FAIL, the runner's proofs among them (Yellow 300, the
warmed-up text snapshot, Crystal from a snapshot, m3_bgp_change, Yellow from power-on, and
on DMG blargg `2-causes` and the fetch-locks ROM with 134 and 88 OAM bug passes on both
sides); every game, boot and bench run at its hash in lockstep and as one model with 0
dense fallbacks and 0 refusals; the census 318 of 321 in both columns, no regression
against the merge or its follow-up.

A/B, frames per second, every component in weights as one model, A = the
integration's build, B = the STAT-edges build, three rounds of A B B A per workload in one queue ticket
(load 1.7-4.2), median of six, every frame hash equal and 0 fallbacks:

| Workload | A | B | B / A |
|---|---:|---:|---:|
| text.sh | 50.15 | 48.45 | 0.966 |
| overworld.sh | 119.2 | 116.5 | 0.977 |
| battle.sh | 69.05 | 66.95 | 0.970 |

Every B run is slower than every A run in its workload, so the drop is real at this load;
its cause is not measured (measure-only round; inferred candidates: the memory circuit's
OAM pass slots and four more heads on every step, and the control circuit's added nodes).

## Last 12: a DMG's read and write locks

What changed: `Lcd::locks` gives reads' and writes' locks apart, the control circuit's `LK`
carries both pairs for two segments with spans to a DMG's M-cycles 19 and 113, and the
transport hands each access the pair of its kind (`docs/hardware-tests.md` §3.17).

Digests (`verify.sh`): control `28e40a08` (101.4 M dense, 0.25 M nonzeros) and the one
model `76524538` (3,184.8 M, 1.83 M; with `GBCFORMER_BITS=1` `c7908db7`, 3,224.4 M) moved;
cpu `188ee397`, cpu-memory `6acacbed`, apu, ppu and ppu-stepper are bit-identical to
the STAT-edges build's. Checks, one ticket: ratchet green; `digest --gguf` 0 differ; `cargo test` 24
binaries ok, SST 0 failed, 0 dense fallback; `tools/gate.sh` 47 ok, 0 FAIL; every game, boot
and bench run at its hash with 0 fallbacks and 0 refusals; the census 320 of 321 in both
columns, mooneye `ppu/lcdon_timing-GS` and `lcdon_write_timing-GS` new, no regression.
gbmicrotest 387 to 391 of 513 (a probe reading FF82 after 30 frames on DMG, not
committed). The runner's proofs on both lcdon ROMs on DMG, 60 frames: six streams, every
pass record, frame and sample equal (`docs/minimal-runner.md`).

A/B, frames per second, all five in weights as one model, A = the STAT-edges build, B =
the DMG locks build, three rounds of A B B A in one queue ticket (load 1.2-4.7), median of six,
every frame hash equal and 0 fallbacks. The CGB benches cannot move by the locks, which
are the modes' there; the DMG run is Yellow 600 frames on `--hardware dmg` (`9292faef`),
where the new segments ask for the locks more often:

| Workload | A | B | B / A | Ranges |
|---|---:|---:|---:|---|
| text.sh (CGB) | 48.10 | 47.75 | 0.993 | 46.6-48.8 both |
| overworld.sh (CGB) | 115.6 | 115.45 | 0.999 | 111.8-116.4, 114.9-115.7 |
| battle.sh (CGB) | 67.1 | 66.7 | 0.994 | 66.2-67.2, 66.2-67.1 |
| Yellow 600, DMG | 122.85 | 121.7 | 0.991 | 122.2-123.3, 121.4-123.0 |

The CGB rows are within the run-to-run spread; DMG is about 1% slower, its ranges
overlapping (not chased).

## A CGB's mode 0 at the dot, and `hdma_timing-C`

Two units, each verified by `verify.sh` from a frozen worktree (docs/hardware-tests.md
§3.14, docs/control-in-weights.md §25, §26).

- **Unit 1: a CGB's mode 0 at the dot.** Lengths published in dots less 128 by
  the PPU checkpoint, its length model and the stepper; the control checkpoint places the
  tick's and a read's mode 0 by halves; an LCD turned on at double speed starts at M-cycle
  1; a speed switch moves the edges at once. Digests: control `4fb028df` (106.4 M dense,
  from 101.4 M), ppu `79706b58`, ppu-stepper `4fdab521`, one model `290df8df` (3,184.8 M,
  1.85 M nonzeros; with `GBCFORMER_BITS=1` `2ee6dc9c`, 3,237.6 M); cpu `188ee397`,
  cpu-memory `6acacbed` and apu unchanged. Checks: ratchet green; `digest --gguf` equal;
  `cargo test` 25 binaries ok, SST 0 failed, 0 dense fallback; `tools/gate.sh` 52 ok, 0
  FAIL; every game, boot and bench run at its hash, 0 fallbacks, 0 refusals; the census 320
  of 321 in both columns, no regression from the DMG locks build; the runner's three proofs ok.
- **Unit 2: the HBlank request at its own edge, HDMA5 counted at the step's
  end, a run's extra M-cycle.** Digests: control `fb83af92` (115.8 M dense, one level
  deeper), one model `42c17824` (3,184.8 M, 1.85 M; bits `477ae03b`, 3,237.6 M); every
  other checkpoint unchanged. Checks as unit 1's, with `tools/gate.sh` 52 ok and 1 FAIL,
  `cargo test`: `tests/one_model.rs` pins each stream's depth and control's went from 16
  to 17 layers. That expectation was updated (the test alone) and the test rerun
  in its own ticket: 3 passed, the vocabulary 65,804 tokens. The census 321 of 321 in both
  columns (`hdma_timing-C` new, no regression from the DMG locks build or unit 1).

A/B, frames per second, all five in weights as one model, three rounds of A B B A in one
queue ticket per unit (load 1.3-10.2 in unit 1's text rounds, 1.5-5.0 otherwise), median
of six, every frame hash equal and 0 fallbacks. Unit 1: A = the DMG locks build, B =
unit 1. Unit 2: A = unit 1, B = unit 2.

| Workload | Unit 1 A | B | B / A | Unit 2 A | B | B / A |
|---|---:|---:|---:|---:|---:|---:|
| text.sh | 48.45 | 48.15 | 0.994 | 48.35 | 48.1 | 0.995 |
| overworld.sh | 116.05 | 116.2 | 1.001 | 115.4 | 114.85 | 0.995 |
| battle.sh | 66.45 | 66.3 | 0.998 | 67.0 | 66.55 | 0.993 |

Ranges: unit 1 text 47.5-48.5 against 46.7-48.7, overworld 115.1-116.3 against
115.5-116.3, battle 63.2-67.2 against 65.5-67.0; unit 2 text 46.8-48.7 against 45.9-48.3,
overworld 114.2-116.5 against 113.7-115.4, battle 66.4-67.1 against 66.3-66.9. Every
median is within its rounds' spread; unit 2's control checkpoint is one layer deeper,
which matches a half-percent cost but is not shown apart from the noise (not chased).

## Phase 2 step zero

Review 3's debts and one canonical model. Digests, compiled (`examples/digest`), against the
integration before this work:

| Checkpoint | Before | After | Why |
|---|---|---|---|
| cpu | `188ee397` (68.3 M) | `bebf5d93` (70.4 M) | the HALT bug's fetched stream: the host-fed circuit's immediates are OP1 and OP2 with or without the bug |
| cpu-memory | `6acacbed` | `6acacbed` | the memory circuit already fetched operand 1 under its own state |
| control | `9e87dbb9` (110.6 M) | `6e225675` (115.8 M) | HDMA's held run-off bit `HE` |
| one model | `47ceed72` | `eda1ea0b` | control's `HE`; 3,184.8 M uniform, 1,093.4 M stored layer-shaped (1,093,292,544 in the matrices), 1.85 M nonzeros |
| apu, ppu, ppu-stepper | `d8d3a06e`, `79706b58`, `4fdab521` | unchanged | |

The 0/1 GBC CPU variant (`GBCFORMER_BITS`) is removed; its last digests were cpu
`5a9e1d6a`, cpu-memory `2e1f32d7`, one model `5ace73d5` (3,237.6 M), before this work.
CLAUDE.md's digest list before this work, kept here as the history it was:

> cpu 188ee397…, cpu-memory 6acacbed…, apu d8d3a06e…, control 9e87dbb9… (110.6 M dense), ppu 79706b58…, ppu-stepper 4fdab521… (482.9 M dense, 0.62 M nonzeros), one-model 47ceed72… (3,184.8 M dense, 1.85 M nonzeros); with `GBCFORMER_BITS=1` cpu 5a9e1d6a…, cpu-memory 2e1f32d7…, one-model 5ace73d5… (3,237.6 M dense); when one HBlank block a line across the gap between STAT's mode 0 and the request (`DN`, `docs/control-in-weights.md` §27) moved only control and the one model; before it control fb83af92… (115.8 M dense), one-model 42c17824… (with `GBCFORMER_BITS=1` 477ae03b…) as of the HDMA round (unit 2), where the HBlank request's edge, HDMA5 counted at the step's end and a run's extra M-cycle (`docs/control-in-weights.md` §26) moved only control and the one model; before it control 4fb028df…, one-model 290df8df… (with `GBCFORMER_BITS=1` 2ee6dc9c…) as of unit 1, where a CGB's mode 0 at the dot (lengths published in dots, §25) moved control, ppu, ppu-stepper and the one model; before that control 28e40a08…, ppu 1f3118cc…, ppu-stepper 75fcd4ef…, one-model 76524538… (with `GBCFORMER_BITS=1` c7908db7…) as of the LCD-on round, where a DMG's read and write locks (`LK` bits 0-7, `docs/hardware-tests.md` §3.17) moved only control and the one model; before that control ccd90001…, one-model afed13bc… (with `GBCFORMER_BITS=1` e7b08c84…) as of the interrupt round, where the OAM corruption bug in weights moved cpu (the idle M-cycles' FE events, `MC` bits 3-6), cpu-memory (the fetches', reads' and writes' FE events and the OAM pass) and the one model; the HALT's HBlank rule and the STAT edges inside a span moved control; the CGB compatibility-mode I/O classes (`layout::io_code`, `docs/hardware-tests.md` §3.15) moved ppu and ppu-stepper (inferred: the only change under them, not bisected); before it cpu 76c89995…, cpu-memory 4d336582…, control aa9dc582…, ppu 3db883bb…, ppu-stepper 26205bab…, one-model 69b4f0ef… (3,170.0 M dense, 1.82 M nonzeros, 65,523 tokens, six streams), with `GBCFORMER_BITS=1` cpu db09696f…, cpu-memory 1e9ad752…, one-model 8bfbc401… (3,209.6 M dense), as of the FIFO integration, the merge of the finished pixel FIFO (the pixel FIFO, the stepper stream, the judge, access timing A3) into the transport rules integration (the transport rules, the APU cut, the fetch mask); on the finished pixel FIFO cpu 76c89995…, cpu-memory ef29face…, apu 063ba0cb…, control 63002b2e…, ppu 3db883bb…, ppu-stepper 26205bab… (482.9 M dense, 0.62 M nonzeros), one-model f70df183… (3,178.5 M dense, 1.84 M nonzeros), after the stepper, the judge and the merges of the APU port and access timing's A3, which moved every digest (not attributed per checkpoint); on the transport rules integration cpu fac7fab8…, cpu-memory ce478a0e…, apu d8d3a06e…, control dc3d5f53…, ppu c3a13d11…, one-model 0028a21f… (1,391.0 M dense, 1.25 M nonzeros); the fetch mask moved cpu-memory and the one model when operand fetches took the access locks at their own M-cycle (`docs/perf-log.md`, "Operand fetches take the access locks"; before it cpu-memory 36dcca3f…, one-model 94063c04…), and the APU cut moved only apu and the one model, when the APU pass emitted `DL` and counted `SD` in halves ("The APU's event split and halves in the model"; before it apu 65a54694…) and before that when it emitted its segment end `SE` ("The APU's segment end in the model"; before it, apu 063ba0cb…, one-model f80b39d9…); the transport rules moved control (from 0ac0de8f…) and the one model; earlier cpu-memory 36dcca3f…, apu 063ba0cb…, control 0ac0de8f…, one-model 77eec1e2… (1,387.2 M dense, 1.26 M nonzeros, five streams) as of the APU-port merge, where the APU port (the APU as SameBoy's port, `docs/hardware-tests.md` §3.12) moved apu, cpu-memory, control (the rise outputs `FR0`, `FR1`) and the one model; before it, cpu-memory 97752cae…, apu 131ef0af…, control 16c8ed35…, one-model 4decc24a… (985.2 M dense, 1.19 M nonzeros), where PPU FIFO step 1 moved ppu and the one model (`docs/perf-log.md`, "PPU FIFO step 1"; before it ppu b9696877…, one-model 8772a661…, 927.2 M dense, 1.18 M nonzeros, as of A2's operand fetches in the memory checkpoint (`FS`, `docs/perf-log.md`, "GBC CPU, OAM DMA, timer, serial and LY timing"; before them cpu-memory b6571351…, one-model a2d2d764…); A2's A2 moved cpu (the dispatch's vector, HALT staying on itself, the operand count in `RC`/`WC` bit 7), cpu-memory (also the `DB` input and per-bus DMA blocking), control (`FL`, `ST` bit 2, `DB`, `DZ`) and the one model, `docs/hardware-tests.md` §3.11; before it cpu bfd4c294…, cpu-memory 828e6e78…, control 0d5b559c…, one-model 73b7b9d9… with the 24-bit layout, whose 21-bit line key moved cpu-memory, ppu and the one model (`docs/perf-log.md`, "MBC5 at 8 MiB: the 24-bit layout"; before it, after merging the runner, cpu-memory bf75311f…, ppu caa99f33…, one-model dc902b72…); with `GBCFORMER_BITS=1` the 0/1 GBC CPU's cpu f816d219…, cpu-memory 719e23aa…, one-model c13378fe… (1,432.8 M dense; before the fetch mask and the APU cut cpu-memory 93dc9a17…; earlier a135b934… and 9d685cbe…) (before PPU FIFO step 1 a33e67b1…; before the operand fetches d3816de4… and b92ddb13…; before A2 b1b4317a…, e42c490a… and 922413f2…; before the 24-bit layout 4d3ce37f… and eaf5ba8b…); access timing moved cpu (its RC/WC outputs), cpu-memory, control and the one model; the host rules moved cpu-memory (the boot ROM bank, KEY0/FF50) and the one model (the length stream); the key-bit round moved every checkpoint with memory heads (cpu-memory, ppu, the one model) when each key bit took its own RoPE pair (`docs/perf-log.md`, "One key bit per RoPE pair"); the overlay round moved only the one model, whose key rows now read the null channel so the null token decodes the null cell ("The one model in stock llama.cpp"); the DMA unit moved control and the one model when every DMA byte and stall became a control output, `docs/control-in-weights.md` §21; the host-in-weights work moved cpu-memory and the one model when the memory model's translation took over the I/O dispatch, and control when OAM DMA moved in line runs, §21.5; the clock unit moved apu, control and the one model when the clocks and the HALT wake became model outputs, §22; hardware accuracy moved apu (the DMG wave RAM rule), ppu (compatibility mode), control (LY's early increment) and the one model, `docs/hardware-tests.md`

What moved the sizes (measured, `export` and `shape`): the cpu circuit lost 26 nodes and 28
wires (642 to 616, 1,344 to 1,316) with the HALT bug's muxes, but `Circuit::schedule` levels
it with a widest MLP of 305 units where it had 287, so the intermediate width rounds from
576 to 640 and the checkpoint grows from 68.3 M to 70.4 M (hidden 1,600, 7 layers, both).
Control's `HE` wire takes hidden from 1,344 to 1,408 (17 layers, intermediate 640, both),
110.6 M to 115.8 M. That `HE` alone moved the one model is inferred from the circuits that
changed (control and the host-fed cpu, which the one model does not hold); the one model's
uniform shape and stored count are unchanged (3,184.8 M; 1,093,432,896 stored, which
`export`'s own count matches).

## The GBC CPU stream's replay bench, layer 0's merged diff, the stream's own channels

**The measuring tool first.** Light-lane wall times moved by 20% and more under other
load, so a change worth a few percent could not be ranked. Two tools now measure
without that noise.

- `engine::counters::thread` reads the calling thread's CPU time and, on macOS, its
  instructions and cycles (`thread_selfcounts`). Under `GBCFORMER_STATS=1` each engine
  reports its thread's counters per pass over the run, and every 1,024th pass of each
  kind is bracketed by them. The bracket includes its two syscalls, about 2,000 cycles,
  so the per-kind figures are an upper bound; the per-run figure is not affected.
- `GBCFORMER_REPLAY_OUT=<file>` records the GBC CPU stream's passes and write records,
  from `GBCFORMER_REPLAY_SKIP` passes on, with the memory image but no ROM or boot ROM
  byte. `cargo run --release --example replay -- <file> <rom> [--reps K] [--variants a,b]`
  runs them through a fresh CPU stream and holds every pass's decoded tokens to the
  recording. `--variants` interleaves runs of `Engine::variant` in one process, which an
  experiment reads.

The text trace (1,000,000 passes from pass 300,000) walks 972 units, 169.1 pairs and 2.9
head reads a pass, against the full run's 936, 164.8 and 2.8. Instructions repeat to 0.01%.
Cycles moved +1.2% and −2.7% between processes (load, and code layout in a rebuilt
binary), and about 1% between repetitions of one process. Changes are ranked by the
in-process interleave, which resolves about 0.5%.

Profile of the replay (`sample`, 6,859 samples): the gate walk 46%, `memory_reads` 13%,
`drain_wires` 10%, `route_layer` 7%, layer 0 5%, the decode 4%, the view's placement 3%,
the write records 3%. The memory share is spread over about 40 lines, the largest the
bus flips' routing (2%) and the head bucket's drain (2%). Per pass the heads compute 16.9
addresses, of which 10.1 equal the last.

**The census.** Under `GBCFORMER_UNITS`, `Plan::demand` counts gate updates that no
position the pass decodes can reach: 3.5 circuit and 12.9 routing updates of about 975 per
GBC CPU pass on text. A per-pass-kind plan would save at most 1.7% of updates.

**Kept.**

| Change | Replay, cycles a pass | Instructions | Note |
| - | - | - | - |
| Layer 0's exclusive diff merges each moved position's sorted wire lists (no marks) | 10,334 → 10,152 (−1.8%) | 40,411 → 40,071 | median of 8, interleaved, ranges disjoint |
| `Plan::compact`: each plan over only the virtual channels it reads or writes | 10,340 → 10,100 (−2.3%) | 39,600 both | every pair lower |

The compaction: the one model's GBC CPU stream uses 2,578 of its 13,826 virtual channels.
Each circuit layer's per-channel walk tables took 553 KB, and the 2,529 (layer, channel)
pairs text touches over the trace spread over them; L1D is 128 KB. `Engine::build`
compacts every plan; a plan with a positional stage keeps its numbering. The APU's thread
gains most (light lane, contended: 27,001 to 22,638 cycles a pass).

**Not yet measured in the heavy queue.** The heavy queue held the lock from 02:58 to past
10:24 for other tickets, so no paired fps A/B ran. A queued ticket
holds it: the base against this work, median of 6 interleaved on text,
overworld and battle, digests before and after, the engine, one_model and checkpoint_sst
tests, and the machine thread's counters against the base. Light lane (contended, ranking
only): text with `GBCFORMER_STATS=1` 42.0 fps and 13,106 machine-thread cycles per GBC
CPU pass with the merged diff alone, then 47.3 fps and 11,722 cycles with the compaction
as well (−10.6%); the replay measured −2.3% for the compaction, so most of the run's gain
is inferred to come from the streams sharing the caches, which the replay does not see.

Checks on the final code (light lane): text `870aa3fd`, overworld `822059c1`, battle
`d3534543`, every stream 0 dense fallbacks; Yellow 300 `--weights all --lockstep all
--one-model` `cfd31f3d`; `--core lockstep --hardware dmg` 8 frames `6872b003`; text with
`GBCFORMER_STEP_ORACLE=cpu=400000,apu=100000,control=20000,ppu=40000` checked 27 passes on
six streams, every token equal; the `GBCFORMER_UNITS` census tables identical to the base;
`cargo check --lib --no-default-features`, native and wasm32.

**Tried and not kept.** Each ran interleaved in one replay process, median of 8.

| Change | Before → after | Why |
| - | - | - |
| The walk's send buffers sized once a bucket (n changes × the layer's largest lanes), no check a change | 10,051 → 10,106 cycles; 39,652 → 39,243 instructions | fewer instructions, more cycles: the checks were off the critical path |
| The view's group compare as an OR of XORs | 10,106 → 10,075 (−0.3%) | the compiler already emitted the same compare |
| After the compaction, channels numbered by the stage that writes them | 10,097 → 10,084 (−0.1%); 39,331 → 39,464 instructions | creation order already groups a stage's channels |
| Each address unit's address and range counts kept in the head-bucket drain, a head marked only when they move | 10,174 → 10,436 (+2.6%); 39,460 → 41,285 instructions | the drain's per-term bookkeeping costs more than the recomputes it saves |

**What is left.** The replayed GBC CPU pass is about 10,100 cycles; the light-lane text run
puts the machine thread near 11,700 cycles per GBC CPU pass, and 60 fps on text needs
about 9,500 (inferred from 3.08 µs a step at 3.1 GHz). The walk is 46% of the replayed pass at
about 19 instructions an update and an IPC near 3.9, and its loop shapes held through
this round. Locality is the lever this round found: the compaction cut no instruction.
The next ideas, by their bounds: per-layer channel indices for the walk tables (each
layer still sizes its head and bucket tables by all 2,578 channels; bound unmeasured), a
guarded restriction of the plan to the tokens decoded so far (12.9 of about 40 routing
updates a pass reach no decoded position; at most about 2%), and a faster hash for layer
0's plan memo (0.6% of samples).


## Fewer gate updates per GBC CPU step: the fetch path's PC tests

The pass census first (light lane, contended, every component in weights as one model): on
text.sh the 3.21 M GBC CPU step passes are about 78% of the main thread and control's 47.5 k
passes 3.1% (0.41 s). Pokémon Yellow flags no FE page access in 600 frames of text, overworld
or battle (0 FF0A row queries, 0 OAM rewrite passes), so eliding CGB's row query would remove
nothing there. Control's 17th layer is one true dependency chain (its ASAP depth equals its
scheduled depth: `halves.12`, `LY=153`, `LYC hit'`, the mode-0 pass due, `to line 144`, `H`),
worth about 0.1 fps. The wire-change table re-recorded on step zero was identical.

What changed, all in the memory circuit (`src/sm83/decode.rs`, `d4.rs`), each the same function
(SST, and two new `sm83` tests at the FE page whose mutants fail):

- The OAM bug's PC + k page tests (a dispatch's and JR's idle M-cycles, operands 1 and 2) had a
  carry staircase over PC's low byte each, and a taken JR's target a pair over PC's low byte and
  the offset. PC + k's page is now PC's high byte or the next line's (`F1N`, `F2N`), and the JR
  target's page the next pc's high byte (PCH, the low byte's carry, the offset's sign).
- The operand word lines (`OP1w`, `OP2w`) and `F1N`/`F2N` read PC's offset one-hot (`OP0w`)
  instead of PC's low three bits as patterns.
- `rare`, `ffn` and the rare translation's regions read their opcodes' one-hot wires instead of
  the nibble one-hots, which every opcode with low nibble A or high nibble F flipped.
- Read 0's bus address adds the rare bus and `ffn`, zero unless their opcodes run, instead of
  selecting them again.

Gate updates per fast pass (`GBCFORMER_UNITS=1`, `CORE=model-mem`, exact counts): text.sh
916, 895, 841, 815, 807 after each change; overworld.sh 785 to 686. Memory checkpoint shape
unchanged (11 layers, hidden 1,856, intermediate 1,792); nonzeros 132,829 to 132,442. The
wire-change table re-recorded after (`OP0w.*` and `IDLE.pc1.n` new); the fit places the circuit
as before (inferred from equal update counts and nonzeros).

Digests (`examples/digest`): cpu-memory `6acacbed` to `17a8b93a`, one model `eda1ea0b` to
`54cdf97a` (1,093,432,896 to 1,092,904,512 stored layer-shaped, 3,184.8 M uniform); cpu
`bebf5d93`, apu, control, ppu and ppu-stepper unchanged.

A/B, frames per second, all five in weights as one model, A = step zero, B =
the model speed work, both built in one queue ticket with the same command, three rounds of A B B A
(1-minute load 2.6-4.6 at the end of each workload's rounds, 6.0 just before them), median of six, every frame hash equal, 0 dense fallbacks, each side's memory
nonzeros printed (188,794 and 188,407):

| Workload | A | B | B / A | Ranges |
|---|---:|---:|---:|---|
| text.sh | 48.1 | 49.35 | 1.026 | 47.8-48.4, 49.0-49.5 |
| overworld.sh | 113.05 | 116.3 | 1.029 | 112.6-114.8, 115.7-118.1 |
| battle.sh | 65.75 | 67.8 | 1.031 | 65.3-66.0, 67.2-68.3 |

Every B run is faster than every A run in its workload. One `GBCFORMER_STATS=1` text run each
(not counted): the step pass 936 to 824 units and 3.04 to 2.93 µs, 164.8 to 157.3 (changed
channel, reading layer) pairs, 3.94 to 3.83 µs per step pass of wall time. The units fell 12% and
the pass 3.6%: the shared layers went from 2.63 to 2.51 µs while layer 0 (0.22 µs) and decode
(0.14 to 0.15 µs) stayed, and the per-step work outside the pass (about 0.9 µs of the 3.83) is
untouched.

## Phase 2: a smaller one model

This section describes the size work with the stepper's delay. The delay was reverted on the combined tip; the sizes, digest and streams of the tip without it are in "Phase 2 combined" below.

Two changes to the one model, each moving only its digest (`eda1ea0b` to `92815f7d`); cpu
`bebf5d93`, cpu-memory `6acacbed`, apu `d8d3a06e`, control `6e225675`, ppu `79706b58` and
ppu-stepper `4fdab521` are unchanged. `examples/shape --one` prints the one model by part and
layer: what sets hidden, the FFN units per layer and part, the clearing units, the norm floor
and the stored size.

| Commit | Change | Hidden | Intermediate | Heads | Uniform | Stored (align 64) |
|---|---|---:|---:|---:|---:|---:|
| step zero | base | 2,752 | 10,688 | 69 | 3,184.8 M | 1,093.4 M |
| identity channels | one identity channel per input gather group (239 to 38 channels) | 2,560 | 10,688 | 69 | 2,962.6 M | 1,017.1 M |
| the stepper's delay | the stepper's nodes and heads delayed past the PPU's widest layers (a search over the delay) | 2,560 | 6,016 | 44 | 1,860.8 M | 1,017.0 M |

- **Identities.** Hidden was the shared front, 1,111 channels of which 769 are slot
  identities, plus the stepper's 1,621 own wire channels. Every slot of a layer-0 gather
  group keys alike (its group's two code dims), so one identity channel per group gives
  bit-identical keys; the embedding rows, the key columns and REC's row move (`Layout::ids`).
- **The stepper's delay.** Each part keeps its own levels, so the stepper's sort (5,280
  units at its level 7) landed on the PPU's widest layer. `one_model::parts` delays the
  stepper (`Circuit::delay`) by the layers whose merged checkpoint has the fewest uniform
  parameters (`Model::config`, 0.4 s per candidate). The stored size barely moves: moving
  units between layers changes only the padding.
- **Reverted: overlay channels opened up to the widest part** (reverted in
  the stepper's delay). Opening every part's channels up to the stepper's before reusing one cut
  clearing units from 11,768 to 8,102 and stored size by 20 M, but widened every stream's
  slice to 2,560, a cost each part pays when it runs alone in a stock runtime.
- **Tried and dropped: `Circuit::fit` on the stepper** at channel targets 1,589 down to
  1,321 (uniform change weights) found no placement at any target; its live wires stay
  1,621 and hidden 2,560 (2,496 needs at most 1,586).

Each stream's own network, as the run prints it, base then tip:

| Stream | Hidden | Layers | Dense |
|---|---|---|---|
| GBC CPU with memory | 2,368, 2,176 | 11, 11 | 486.7 M, 447.3 M |
| APU | 2,240, 2,048 | 25, 25 | 635.4 M, 600.6 M |
| control | 2,240, 2,048 | 17, 17 | 492.3 M, 456.8 M |
| PPU scanline | 2,624, 2,368 | 14, 14 | 766.3 M, 697.9 M |
| PPU length | 1,728, 1,536 | 15, 15 | 414.9 M, 368.8 M |
| PPU stepper | 2,752, 2,560 | 9, 20 | 839.4 M, 1,323.3 M |

The stepper's stream grows: it now runs 20 layers, and a stock runtime pays that per pass and
per cached token. Its engine time is unchanged (text 190.8 ms, battle 371.4 ms, light lane).
The delay is a choice of size against that stream's depth; the uniform size by delay, measured
before the revert: 0 layers 2,876.6 M, 8 layers 1,934.5 M, 12 layers 1,836.2 M, 14 layers
1,823.9 M.

Checks, one heavy-queue ticket with the stepper's delay: `cargo test --release` of the lib (185) and
`one_model`, `stream_audit`, `ppu_stepper`, `ppu_checkpoint`, `ppu_length`, `ppu_lines` and
`boot_overlay`, all passed (`the_null_token_decodes_the_null_cell` is ignored, as before); the
layer-shaped export, 1,016,983,040 stored, and `digest --gguf` with 0 values differing and the
manifest agreeing; `GBCFORMER_GGUF_ONLY` Yellow 300 at `cfd31f3d`; the runner's tests against
that file; Yellow 300 with all five in lockstep as one model at `cfd31f3d`, and again with the
step oracle (5 streams checked); `m3_bgp_change` 4 with the step oracle on all six streams and
3 record oracle checks; dmg-acid2 and cgb-acid2 pixel-exact as one model in lockstep; text,
overworld and battle in lockstep as one model at `870aa3fd`, `822059c1` and `d3534543`; the
runner proofs on `m3_bgp_change` 4 (`STREAMS=6`, every pass record equal, `revised_before`
432 and `taken_after` 428 on both sides) and Yellow 300; `same-passes.sh` on `m3_bgp_change`;
`slice_full` over Yellow 60 and `m3_bgp_change` 2, every stream's sampled events equal to the
whole checkpoint's. No run had a nonzero dense fallback. SST was not rerun: the standalone
digests are unchanged and no part's circuit changed (the delay moves only levels in the merged
circuit).

**The A/B.** text and overworld with `--weights all --one-model`, three rounds of base, new,
tip, tip, new, base, in the same ticket from 17:28 to 18:09. base is step zero
built in that ticket, new the build with the reverted change, tip the size work with the delay.
Every run reached its hash with 0 dense fallbacks; the 1-minute load was 2.5 to 13.4.

| Workload | base, median of 6 (range) | new | tip | tip against base |
|---|---:|---:|---:|---:|
| text | 47.45 (46.9-48.1) | 47.6 (46.5-48.0) | 47.95 (47.7-48.4) | +1.1% |
| overworld | 113.2 (112.8-113.9) | 113.45 (111.2-114.3) | 114.65 (112.9-115.8) | +1.3% |

The tip's ranges overlap the base's on both workloads, so this is no measurable difference,
and the narrower model is not slower. The limiter was not profiled in this ticket.

## Phase 2 combined: the four branches on one tip

The combined tip is step zero with the infrastructure, engine speed, model speed and
size work on top. The stepper's delay from the size work is reverted: it
deepened the stepper stream from 9 to 20 layers, and each part must stay selectable at its own
cost. The section above describes the size work with the delay; this one is the tip without it.
The figures below come from this build's `digest`, `shape --one`, the export and a Yellow
300 run.

Digests: cpu `bebf5d93`, cpu-memory `17a8b93a`, apu `d8d3a06e`, control `6e225675`, ppu
`79706b58`, ppu-stepper `4fdab521`, the one model `e9618832`. Only the one model is new
against the branches' own digests.

The one model: hidden 2,560, intermediate 10,688, 25 layers, 69 heads, vocabulary 65,804,
seq 769 (239 inputs, 530 outputs), 568 slot identity channels. The layer-shaped GGUF
(align 64) stores 1,016.7 M parameters: 1,016,524,800 in its weight matrices and 130,560 in
its norms, 1.85 M nonzero. The padded uniform checkpoint is 2,962.6 M dense. Without the
delay the stepper's widest level (5,280 units) lands on layer 6 with the PPU's 2,861, so
layer 6 sets intermediate at 10,688.

Each stream's own network, as the run prints it:

| Stream | Hidden | Layers | Dense |
|---|---:|---:|---:|
| GBC CPU with memory | 2,176 | 11 | 447.3 M |
| APU | 2,048 | 25 | 580.9 M |
| control | 2,048 | 17 | 450.1 M |
| PPU scanline | 2,368 | 14 | 691.5 M |
| PPU length | 1,536 | 15 | 368.8 M |
| PPU stepper | 2,560 | 9 | 780.8 M |

**The A/B.** text, overworld and battle with `--weights all --one-model`, three rounds of A B B
A in one heavy-queue ticket (21:44 to 22:32, 1-minute load 3.0 to 5.6). A is step zero,
built in that ticket with `cargo build --release -j 6 --all-targets`, B the same
build at this tip less docs. Every run reached its hash with 0 dense fallbacks.

| Workload | A, median of 6 (range) | B, median of 6 (range) | B against A |
|---|---:|---:|---:|
| text | 47.50 (47.1-48.0) | 50.55 (50.2-51.2) | +6.4% |
| overworld | 111.5 (109.6-113.0) | 115.85 (108.3-119.0) | +3.9% |
| battle | 65.95 (65.6-66.7) | 70.40 (66.9-70.9) | +6.7% |

text and battle are faster, with disjoint ranges. On overworld the ranges overlap, so it shows
no measurable difference. The limiter is the GBC CPU stream's pass on the machine thread. With
`GBCFORMER_STATS=1` its 69-position pass is 3.12 µs (936 units) on A and 2.85 µs (824 units) on
B. B's thread counters give 11,423 cycles (3.67 µs CPU) per GBC CPU step for the whole machine
thread. The sampled passes account for 10,236 of those cycles, so 1,187 lie outside the pass.
About 358 of them are the control engine's passes and about 77 the PPU length engine's, both of
which run on that thread. The sampled pass includes two counter reads, so the in-pass share is
an upper bound. By the wall timers the work outside the pass is 0.87 µs per step.

## The block walk's checks once a call

The GBC CPU stream's pass on the replay bench (text trace, 1,000,000 passes from pass
300,000, recorded at the base) was 9,117 to 9,208 cycles and 36,443 instructions. A
sample of it put the shared layers' block walk at 44%, `advance` with the memory heads
inline at 17%, `drain_wires` at 11% and the routing layer at 7%; inside the walk the
vector bodies held 43% of its samples and the send loop 17%. A census of the separate
CPU-and-memory checkpoint on text (exact counts): 119 (channel, layer) pairs, 779 gate
entries on 484 distinct units, 221 vectors at 2.99 entries, 117 lone entries, 108 sends.

**Kept.** `Marks::blocks` checks its send buffers once a call, against (changes x (the
largest channel's lanes + LANES)), instead of once a change, and reads the channel heads
(`Blocks::flat_unchecked`) and the pack index unchecked; a table gives each lane mask's
popcount. The buckets hold only plan channels (layer 0's changes, the memory heads'
outputs, `drain_wires`), which an assert ties to the layer's tables. Every digest equal.

| Measure | Base | Tip | Change |
| - | - | - | - |
| Replay, in-process interleave (V2 against the same binary's walk) | 9,989 / 10,076 cycles | 9,803 / 9,855 | −1.9% / −2.2% |
| Replay, binary against binary (5 alternated pairs, median of 6 each) | 9,168–9,485 | 8,904–9,223 | about −2.5%; 36,445 → 34,366 instructions (−5.7%) |
| Text, the CPU stream's thread under `GBCFORMER_STATS` | 11,130 cycles, 44,829 instructions | 10,874, 42,757 | −2.3%, −4.6% |
| Text fps, median of 6 (ABBA x3) | 51.25 (50.8–51.8) | 52.55 (51.8–53.1) | +2.5%, ranges touch |
| Overworld fps | 118.75 (116.0–119.9) | 118.75 (117.4–119.7) | none |
| Battle fps | 69.1 (68.5–69.5) | 70.65 (70.3–71.8) | +2.2%, ranges disjoint |

Load 3 to 6 in the queue, one battle B run at 34. Hashes 870aa3fd, 822059c1, d3534543 on
both sides, every stream 0 dense fallbacks; Yellow 300 all in weights, in lockstep, as
one model `cfd31f3d`; DMG 8 `6872b003`; step oracle on text over six streams, every token
equal; lib tests 186 passed, checkpoint_sst 8 passed.

**Tried and dropped**, ranked in one replay process (median of 6):

| Change | Effect | Why |
| - | - | - |
| Head address from its 21 bit forms by NEON | −0.1% | the scalar loop was cheap |
| A vector that sends nothing skips the pack on a branch | +4% | mispredicts |
| Skip circuit layers whose buckets are empty | +0.5% | the empty calls were cheap |
| A local channel index per layer (buckets carry it) | +1.3% | one more load per reader bit |
| Six other block rules (fewer low-fill vectors, full vectors only, blocks aligned to 4 units, every run a block) | +0.1% to +7.9% | the current rule is the best of seven; alignment did not help, so partial store forwarding was not the limit |
| The routing layer without data branches | +3.1% | its branches were well predicted |
| One evaluation per dirty aligned group of 4 (not built) | estimated even | 198 groups a pass against a break-even near 190 |

Instruction cuts turned into cycles at under half (V2: −5.4% instructions, −2%
cycles), and the in-binary experiment plumbing alone cost 9% cycles for 2.8% more
instructions (inferred: code layout). The pass runs near 3.8 instructions a cycle; what
limits it beyond instruction count is not identified.


## The pass's floor: a cost model of the GBC CPU pass, and its fixed work cut

Round 4b removed 12% of the GBC CPU step's units and saved 3.6% of its cycles. This round measured
what the pass's cost is proportional to, then cut what does not depend on the step.

**The tool.** `examples/replay --model N` (N prime) brackets every pass of the trace once by the
thread's counters, each phase modulo N in a replay of its own, and fits cycles and instructions by
least squares (`engine::ols`) against what the pass did. It brackets one stage of every Nth pass, the
stages in turn, and does both for a pass run a second time on the same tokens, in which nothing
changed. The brackets live in a second copy of the pass (`fill_stream_as::<true>`, `engine::probe`)
that runs only while a stage is armed; the plain pass tests the probe once. A stage's instructions are
exact. Its cycles are not: a bracket's two system calls cost a real pass about 47 cycles beyond the
empty bracket, so `sample` on the plain replay gives the cycles.

**The model** (text trace, 1,000,000 passes of the one model's GBC CPU stream on the base; ticket of
2026-11-16 09:22).

- A pass is 8,437 cycles and 33,580 instructions (5% and 95%: 4,321 and 15,828 cycles). A pass in
  which nothing changed is 932 cycles and 5,905 instructions: 11.0% and 17.6% of the mean.
- That floor, in instructions: the gate walk called on empty buckets 1,219 (122 a layer), the view's
  comparison of every token 1,191, the drain 868, the memory heads 776, the decoded tokens' copy 720,
  the layer loop's glue 543, the decode loop 245.
- Cycles by `sample`: gate walk 43.4%, memory heads 14.4%, `drain_wires` 11.1%, routing layer 7.3%,
  decode and copy 4.9%, layer 0 4.7%, view 4.2%, layer loop and scatter 3.4%.
- One layer's gate walk costs 147 instructions a call, 24.6 a pair, 44.7 a block vector, 18.6 a lone
  entry, and nothing a unit beyond those (R2 0.98). The mean vector holds 2.99 live entries of 4. A
  unit removed from a vector that stays saves nothing, which is why round 4b's units moved cycles so
  little.
- A layer-0 replan costs 40,159 instructions and about 11,000 cycles. It happens on every result pass
  after host-served bytes and on the step after it (0.0077 a pass, 1.0% of the pass). What in a
  replan costs that much is not established: the profile puts about as many samples on the plan
  memo's key build and hash as the fit gives replans in all.
- Instructions fit what the pass did to R2 0.996 (units replaced by vectors, lone entries, touched
  channels and dirty heads); cycles to 0.82. The full model, with every
  table, was kept locally.

**Kept.** The first five were ranked as `Engine::variant` bits interleaved in one replay process
(median of 8, two processes) before they became the one path. That experiment binary's unchanged pass
ran 6.5% to 7.6% slower than the base's, so the in-process figures rank the changes and are not gains
over the base.

| Change | In-process cycles | Instructions |
| - | - | - |
| Memory heads leave a layer at once when the head bucket is empty, the image is as last seen and no head of the layer is dirty or hit; an output flip no longer marks its readers directly (its route fills the head bucket) | −0.8%, −1.1% | −587 |
| The layer loop's bucket slices without the range check, `drain_wires` in place | −1.1%, −1.5% | −548 |
| A nibble head's route pattern four routes at a time (`value_pattern`) | −5.2%, −5.4% | −291 |
| The decoded tokens copied out under one bound | −0.2%, −0.1% | −337 |
| Layer 0's wire lists dense, a moved position's change one subtraction a channel | −1.1%, −1.5% | −132 |
| The view compares the inputs one by one and the outputs whole | not ranked in-process | |
| A pass between two exclusive attention plans takes the wire lists' difference | inert: 0 of 108,290 replans | |

The route pattern's figure is specific to the experiment binary (decode is 3% of the base's pass).
The copy is below 1% alone and is kept as part of the floor. The plan switch never ran, and why is not
established (a plan that is not exclusive, or a replan that returns the same plan). It is in the
verified tip and should come out, with the step oracle's check of such passes.

**Dropped.**

| Change | Measured | Why |
| - | - | - |
| A change's first 1, 2 or 3 readers routed without a branch on its mask | +1.7%, +4.5%, +8.7% cycles; +1,749 to +5,078 instructions | the 5.4% on `drain_wires`' route line is the pushes themselves |
| The view's tokens compared eight at a time as an OR of differences | −0.7%, −0.4%; −109 instructions | the fold stayed scalar |
| A memo of attention plans by decoded-position list | 0 | the replans are plan changes, not list changes |
| Stage cycles as differences of prefixes of one pass | the bracketed copy ran 1,015 cycles slower, stages came out negative | removed from the bench |
**Measured, binary against binary** (one heavy-queue ticket, 2026-11-16 12:36 to 14:01; against the base,
the tip built the same way, both built with `cargo build --release`; every digest equal, one model `54f80459`).

| | base | tip | change |
| - | -: | -: | -: |
| Replay, cycles a pass (median of 6, three alternated pairs) | 8,825, 8,820, 8,789 | 8,548, 8,553, 8,612 | −3.0% |
| Replay, instructions a pass | 34,640 | 33,120 | −4.4% |
| The pass alone, bracketed (cycles, instructions) | 8,437, 33,580 | 8,161, 32,112 | −3.3%, −4.4% |
| A pass in which nothing changed (cycles, instructions) | 932, 5,905 | 901, 5,268 | −3.3%, −10.8% |
| Machine thread on text, per GBC CPU pass (`GBCFORMER_STATS=1`, one run each) | 10,789 cycles, 43,144 instructions | 10,414, 41,681 | −3.5%, −3.4% |
| text, fps (median of 6, range; three ABBA rounds) | 52.3 (51.7 to 52.7) | 53.6 (51.6 to 54.8) | +2.5% |
| overworld, fps | 117.9 (115.5 to 121.3) | 119.9 (113.0 to 121.2) | +1.7% |
| battle, fps | 72.6 (71.5 to 73.4) | 74.4 (70.8 to 75.4) | +2.5% |

The 1-minute load was 3.2 to 7.7 through the fps runs (other light-lane jobs), so the fps
ranges overlap; the thread's own counters and the replay are the firmer figures. The aim was 10% of
the machine thread's cycles; this is 3.5%. What limits the rest: the instructions removed (1,468 a
pass) were the cheap ones, and the walk, the memory heads' real work and the routing layer did not
move.

By stage, instructions a pass before and after: memory heads 6,256 to 5,898, drain 3,897 to 3,721,
decode 1,532 to 1,229, copy out 720 to 375, layer 0 1,257 to 1,178, the layer loop's glue 525 to 53,
gate walk 14,634 to 14,661, routing 2,772 to 2,780. **The view went the wrong way, 1,321 to 1,534**:
the whole-slice compare of the outputs (`memcmp` over about 2 KB) costs more instructions than the
four-at-a-time compare it replaced. It should be reverted to the old `place_tokens`.

Every run reached its hash with 0 dense fallbacks: text `870aa3fd`, overworld `822059c1`, battle
`d3534543`; Yellow 300 `--weights all --lockstep all --one-model` `cfd31f3d`; `--core lockstep
--hardware dmg` 8 frames `6872b003`; text with `GBCFORMER_STEP_ORACLE=cpu=100000,apu=50000,
control=10000,ppu=20000`, six streams checked, every token equal. The replay held all 1,000,000
passes' tokens to the recording in every repetition. `cargo test --release --lib` 189 passed;
`cargo test --release --test checkpoint_sst` 8 passed.

**Open leads, by the model.** The view's compare of the whole sequence every pass (1,200 to 1,500
instructions, 4% of the pass's cycles) needs the device to say which tokens it wrote, which is a
change to `Denoiser::fill_into`'s contract. The replans (1.0%) need their cost located first: the plan
memo's key build and hash, or the mix of every head. The memory heads read 214 instructions a dirty
or hit head (16.9 a pass). The gate walk's 147 instructions a call are paid 10 times a pass whatever
the buckets hold.

**A trim** (one commit on the floor work) reverts the view's compare to the four-at-a-time
`place_tokens` and removes the plan switch with its step-oracle check. It is not measured in the heavy
queue. Light lane, separate checkpoints (the view and the replans do not occur there): text `870aa3fd`
with 0 dense fallbacks, 48,090 machine-thread instructions a pass against the tip's 48,074, the step
oracle's 9 passes equal, the engine's lib tests and the wasm32 check passed.

## Round 5b: a profile-guided build, and the cells' rows on a worker thread

Base: the units round. Two questions, measured apart: how the binary is built, and
which thread writes a write record's K/V rows. Neither
touches a weight: all seven checkpoint digests are the base's in every build below
(`cpu bebf5d93`, `cpu-memory 79a6f8c5`, `apu d8d3a06e`, `control 6e225675`, `ppu 79706b58`,
`one-model 54f80459`, `ppu-stepper 4fdab521`).

The heavy lane was shared with a light lane all day. The 1-minute load during the timed
runs was 2.3 to 13.3, and every arm shows two speeds of the machine. Each fps figure below
is a median with its range, from runs taken in A B .. B A order.

### The build

One source (whose `src/` is the base's) built five ways in one ticket. Each
build ran the replay bench (the GBC CPU stream's passes recorded from text, 1,000,000
passes, 4 repetitions, twice) and text as one model with `GBCFORMER_STATS=1`, three
rounds in rotated order. Every run reached `870aa3fd` with 0 dense fallbacks.

| Build | Replay, cycles a pass (two medians) | Text fps, stats on (median of 3) | Machine thread, cycles a pass | Instructions a pass |
|---|---:|---:|---:|---:|
| plain (thin LTO, 1 codegen unit) | 8,783, 8,689 | 51.6 | 10,792 | 43,143 |
| fat LTO | 9,409, 9,408 | 49.0 | 11,372 | 44,333 |
| `panic = "abort"` | 8,801, 8,761 | 51.8 | 10,746 | 43,072 |
| profile-guided | 8,450, 8,330 | 53.7 | 10,331 | 41,683 |
| profile-guided, fat LTO | 9,080, 8,995 | 50.2 | 11,099 | 43,776 |

- **`-C target-cpu=native`** was not built. `rustc --print target-cpus` names `apple-m1`
  as the host's CPU, and `--print cfg` lists the same 29 target features with and without
  the flag. `aarch64-apple-darwin` already targets it.
- **Fat LTO** costs 5.4% of the machine thread's cycles on the one model, as "Build
  settings" found before.
- **`panic = "abort"`** gives 0.4% of the cycles, under the 1% bar. It would also turn a
  worker thread's lockstep divergence into a SIGABRT, since the APU's, the PPU's and the
  cell rows' workers hand their panic to the machine thread with `resume_unwind`.
- **Profile-guided optimisation** gives 4.3% of the machine thread's cycles and 3.4% of
  its instructions on text, and 4.0% of the pass's cycles on the replay. The compile at
  the hand-off is no slower (43.4 s against 45.1 s).
- The allocator was left alone, since allocation is 0.4% of the machine thread. No
  `#[cold]` or `#[inline]` was added by hand. The pass's hot path is the walk in
  `engine/run.rs`, which the kernel round was rewriting, and the profile-guided build places
  and inlines by counts.

**`tools/build-pgo.sh`** is the profile-guided build. It builds the binary with
`-Cprofile-generate` into `target/pgo/gen`, deletes the counts that the build's own build
scripts and proc-macros wrote, runs text, overworld and battle as one model on that
binary, merges the counts with the toolchain's `llvm-profdata` (`rustup component add
llvm-tools`), builds again with `-Cprofile-use` into `target/pgo/use`, and holds the new
binary's text frame to the instrumented one's. It uses no Python and no cargo plugin. The
plain `cargo build --release` stays the default, and the tests, the gate and every other
row of this log use it. An fps figure from the profile-guided build must say so. The profile
is taken on the three workloads the benches time, so those figures are the build's best
case.

**The paired runs** (tickets 2 and 3): the plain build T and the
profile-guided build P of one source, as one model, with the base's plain build B in the
same rounds (B T P .. P T B, three rounds a workload). Every run reached its hash
(`870aa3fd`, `822059c1`, `d3534543`) with 0 dense fallbacks.

| Workload | T, plain | P, profile-guided | P over T, medians | P over T, neighbouring runs |
|---|---:|---:|---:|---:|
| text | 53.2 (51.1-54.0) | 55.75 (55.5-56.6) | +4.8% | +5.5% |
| overworld | 122.6 (118.1-123.5) | 123.9 (120.9-127.2) | +1.1% | +2.2% |
| battle | 73.2 (69.0-74.1) | 76.55 (73.0-77.9) | +4.6% | +5.2% |

On text the two ranges do not overlap. On overworld and battle they do, and the
neighbouring-run column is the better figure there. It is the mean over the six places
where a P run and a T run are next to each other. Text's limiter is the machine thread, whose
cycles a pass the profile cuts (10,725 to 10,234 in one stats run each).

**Gates on the profile-guided binary:** the seven digests, DMG 8 in lockstep
(`6872b003`), and Yellow 300 with every component in weights and in lockstep as one model
(`cfd31f3d`).

**Two profiles.** The profile runs are threaded, so their counts differ from run to run
and two builds are not bit-identical. A second build from nothing at the same commit has
the same size (6,053,160 bytes; both profiles 2,062,856 bytes) and a different sha1. Text,
P P2 P2 P twice, gave 56.05 (53.1-56.8) and 56.3 (56.0-56.6). The build is reproducible in
speed, not in bytes.

**Restarting the profile at the hand-off** was tried and reverted.
The instrumented build zeroed its counts once the stack was built, so the profile held
only the frames in weights and its top counts became `Marks::blocks`, `advance` and
`fill_fast` in place of the compiler's loops. Text 55.55 fps (54.9-56.5) against 55.75
for the whole-run profile, and the compile at the hand-off, now outside the profile, took
47.6 s against 43.5 s.

### The cells' rows

The brief: per GBC CPU step about 450 cycles of the machine thread go to write records
(the image, the dense oracle's KV, the video queue, the rows built, placed and memoised)
and to the PPU and APU hand-offs. Which of that does the machine thread's next pass not
need?

**What a read needs.** A memory head's lookup (`Cells::serve`) reads the line's row
index, whether the row's key is the line's canonical key, and the line its value row
spells. Those are decided when the record is kept, and a memo miss needs the rows built
from the weights to decide them. The next pass also needs the image and the heads the
write hit. It needs none of the rest: the 1 KB of rows copied into the cell, the memo's
own copy of them, the second build of the sampled memo check, and the dense oracle's KV
cell.

**Bounds first, by switching work off.** Text with separate checkpoints in the light
lane (contended), one process, the experiment on in frames 1 and 2 of every 4, the
machine thread's cycles per GBC CPU step on against off. A/A reads 0.9989 to 1.0016.

| Switched off | On / off |
|---|---:|
| the dense KV update | 0.9947 |
| the K/V row copy of a rewritten cell | 0.9955 |
| both | 0.9903 |
| both, the memo row read on a hit, the check build and the memo row insert (the memo split into keys, facts and rows first) | 0.9843, 0.9821 |
| the above and the build on a miss, the guard skipped (a bound only) | 0.9728, 0.9739 |
| memo keys searched in their own array against in the 1 KB entries | 0.9994, 1.0013 |

Everything a read does not need is worth at most 1.7%, and the build on a miss about
another 1%. The build cannot move. Of 57,323 rows built on a miss, 44% are read within 3
lookups of the build (about one pass) and 62% within 8, and none is overwritten unread, so
a worker would be waited for or beaten to most of them.

**What moved**. `Cells` keeps, per row and per memo entry, the
facts a lookup reads (`cells::Facts`), decided at the write on the calling thread as
before. The rows themselves are behind `rows::Store`. In a session the GBC CPU's stream
defers them (`Cells::defer`, where `Stack` builds the core). `put` leaves one op per
record, and a worker thread applies them in record order, 512 at a time, the sampled
check's second build with them. Whatever reads the rows waits for the worker first: the
full scan after a failed guard, `GBCFORMER_SCAN=1`, `rows_by_line`, a memo grow. The dense
KV stays on the machine thread and finds its cell by a checked hint before the binary
search (`KvCache::set`; p3 measured this hint at 0.9967 and dropped it as under 1% alone).

This is inference management, not a hardware rule. The same rows reach the same cells,
on another core, later. One behaviour changes. A sampled memo check that fails now stops
the run at a later hand-over, at `Session::settle` (before the run reports) or when the
cells are dropped, where it stopped at the write before. How much later is bounded by how
far the worker may fall behind (`BEHIND` batches in `src/engine/rows.rs`): fewer than
(`BEHIND` + 2) × 512 records, which is the rest of the check's batch, the `BEHIND`
batches accepted before the worker has run it, and the batch whose hand-over raises it.
The tickets below ran with `BEHIND` 64, so 33,792 records, not the 512 this section first
claimed. "After review 5" below lowers it to 4, so 3,072.
`GBCFORMER_ROWS_THREAD=0` keeps the rows on the machine thread. SST, the replay bench, the
PPU's streams, the minimal runner and wasm32 never defer.

**What it measured.** Two rules, each written before its ticket ran. Before ticket 3:
keep the worker if one binary's machine-thread cycles a pass fall by 1% with the worker
on and fps is not lower, and revert it if neither moves. Ticket 3 showed neither, so by
that rule's letter the worker was to be reverted. But ticket 3's runs spread 3%, which
cannot show 1% either way. So a second rule was written before ticket 4: keep the worker
if the replay's minimum cycles a pass fall by 108 (1% of 10,800) in every on/off pair and
wall time is not higher. Ticket 4 met it in all three pairs.

| Measurement | Result |
|---|---|
| The switch above on the built change, the off side running the rows in place on rows taken back from the worker (light lane) | rows deferred 0.9894, 0.9881; the hint 0.9959; both 0.9840, 0.9823 (A/A 1.0003, 1.0008) |
| The replay bench, one binary, the stream's rows deferred or not, on off off on on off, 6 repetitions each, load 2.0 to 2.6 (ticket 4) | minimum cycles a pass 8,604, 8,544, 8,557 on against 8,758, 8,742, 8,768 off: the three pairs fall by 154, 197 and 212. Wall 2,678 to 2,696 ns a pass on against 2,728 to 2,749 off. Instructions 34,468 against 34,671 |
| Text as one model, one binary, the worker on and off (`GBCFORMER_ROWS_THREAD=0`), stats on, on off off on, three rounds, load 4.6 to 7.6 (ticket 3) | cycles a pass on 10,746 to 11,059 (median 10,946), off 10,734 to 11,210 (median 10,854); instructions 43,023 against 43,192; fps 50.95 (49.5-51.5) against 50.9 (50.4-51.9) |
| The tip's plain build T against the base's B, paired (tickets 2 and 3) | text 53.2 (51.1-54.0) against 52.85 (52.0-53.2); overworld 122.6 (118.1-123.5) against 120.0 (118.1-124.5); battle 73.2 (69.0-74.1) against 70.7 (68.1-73.4) |
| The machine thread waiting for the worker (scratch counters, two text runs) | 1,715 hand-overs a run, none waited, the queue never deeper than one batch, 0.12 ms in the hand-over of 11.7 s |
| The worker's QoS, utility or user-interactive against default (the switch, light lane) | 1.0004, 1.0003 and 1.0013, 1.0002 (A/A 1.0028, 0.9994); left at the default |

The replay is the measurement that resolves it. On a quiet machine every pair falls by
1.4% to 2.0% of the machine thread's cycles, and wall time falls with them. Its counted
repetitions run on a memo that the uncounted first one filled, so nearly every record is
a hit, where text misses on about 6% of records and each miss hands 1 KB to the batch.
The in-process switch ran text's own misses and read 1.2%. The worker is worth between
the two. The whole text runs do not show it. They spread 3% under the load they ran in,
the two sides' medians and ranges overlap, and only the instructions move (0.4%). So the
worker is kept on the replay and the second rule, and its fps gain on text is unproven,
since 1.7% of the machine thread would be about 0.9 fps, inside these ranges. Re-run the
on/off pair on a quiet machine before quoting an fps gain for it.

**Checks.** `tests/cell_rows.rs` holds one stream of 9,000 writes to the same rows in
every cell, deferred and at once, across memo hits, sampled checks, a grow and replaced
entries, and holds the full scan and the lookup of each written line to the image. Three
mutants fail it: the worker skips a hit, a hit names the wrong memo entry, and the
worker's check reads the wrong entry (the worker's panic reaches the test thread). The
first stream missed the skipped hit, because its hits rewrote the line the cell already
held, so the stream now returns each line to an earlier line. No gate reads K/V rows
otherwise, since lookups read the facts. `GBCFORMER_SCAN=1` on two DMG frames checks
39,772 lookups against the full scan over the deferred rows, 0 differences. CGB 300 in
lockstep-mem with `GBCFORMER_ORACLE=5000` reaches `cfd31f3d` with 36 dense oracle checks.
The wasm32 `cargo check` is clean.

Ticket 2's gates ran on the plain build with the worker on: the seven digests equal to
the base's, the lib tests (186), `cell_rows`, `checkpoint_sst` (8), `dma_records`, DMG 8
in lockstep (`6872b003`), CGB 300 in lockstep-mem (`cfd31f3d`), Yellow 300 with every
component in weights and in lockstep as one model (`cfd31f3d`), and text with the step
oracle (9 passes checked, `870aa3fd`). No run had a nonzero dense fallback. The tickets
ran at the tip's source plus the instrumented-build hook that
a later commit reverts (compiled out of every build measured here).

**After review 5.** The worker's allowed lag, `BEHIND`, went from
64 batches to 4, so a failed check stops the run fewer than 3,072 records after the write
where it was 33,792. The count that decides it is not a timing: with a counter at each
hand-over (light lane, separate checkpoints, `CORE=model-mem`), text made 1,715 hand-overs
and overworld 591, none waited, and none found a batch queued or being applied
(`870aa3fd`, `822059c1`). So the worker's gain above is not touched; no fps was measured
for this change. Two paths that could lose a failed check are closed. Dropping the cells
after the worker had failed, with no hand-over since, skipped the settle and dropped the
worker's panic; the drop now raises it (`engine::rows` has the test, which fails on the old
drop). `play` settled only in that drop, after its last save was written; it now settles
before the save, as `run` does before it reports.

**Not built.** A records worker for the build on a miss (the read-back counts above). The
PPU length pass a line (60 cycles a step) and the PPU and APU hand-offs (68) are needed at
their M-cycle or already are hand-offs. The video queue stays where it is, since the PPU
takes its records at each line in the order the machine thread kept them.


## The pass places what the device changed

The floor model put 229 cycles of every GBC CPU pass on the view comparing the stream's whole sequence
with its last placement, 129 on copying 69 decoded tokens out when 11.6 were decoded, and 1% on layer-0
replans whose cost was not located. This round changes the contract between a device and its
checkpoint so that neither side handles what did not change, and locates the replans.

**The contract.** A device's tokens sit in a `Sequence` (`src/sequence.rs`). Its tokens are private
and `Sequence::set` is the only write. `set` does nothing for an equal token and records the position
otherwise. `Denoiser::fill_seq(&mut Sequence, &Positions)` is one pass. The engine's view places only
the recorded positions, and the engine writes a decoded token to the sequence's store only at the
positions whose decode ran (`Sequence::fresh`). The device keeps each slot's last byte and converts
the fresh positions alone. `Positions` is a fixed list with an id, so the view and the plan find a
list by id. The memory core, the generic `CheckpointDevice` (the APU's, control's and the plain GBC
CPU's), both PPU checkpoints and the stepper pass through it. `fill_into` stays for a slice: the
tests and the minimal runner use it, and the engine then compares the whole sequence as before.
Every pass still runs in full. A position's token is kept across passes only where the engine's stale
bits already kept it, and the device keeps bytes, not requests or outcomes.

The devices no longer write a pass's decoded tokens into the sequence and mask them again before the
next pass (138 stores a GBC CPU step). Every output holds its mask. The one exception is kept as it
was: a result pass reads the request's decoded tokens at their slots, set through the handle and
masked again after it. A write record reads the finished step's tokens from a buffer built for it
(`MemCheckpointCore::finished`). The tokens the GBC CPU stream's passes read are unchanged: the replay
trace recorded with the base and with the tip is the same 38,003,633 bytes. The other streams' are
unchanged by construction (their devices set the same tokens before each pass); no recording of them
was compared.

**What holds it.**

- In every build, the structure: no code outside `Sequence` can write a token without the record, and
  the sequence's stamp (its id and its pass count) names its state at its last pass. The view places
  by the change list only when the stamp it kept is the sequence's stamp before the pass. A placement
  from a slice (`View::place_tokens`, which a recorded `keep` also calls) clears the stamp. The engine
  writes only the decoded positions back only when its last pass was a fast pass of this sequence at
  this stamp. After a slice pass, a dense fallback or a delegated pass, the next pass marks every
  position stale and decodes it again.
- Under the step oracle (`GBCFORMER_STEP_ORACLE`, any cadence), in debug builds and in the library's
  unit tests, every
  pass also compares the whole sequence with the placement and every wanted position's stored token
  with the engine's, and stops at a difference. The oracle's dense pass is compared with the tokens
  as the device reads them from the store.
- Not held in a plain release run: a fault in `Sequence` itself or in the engine's stale bits would
  leave an old token in the store, and lockstep and the frame hashes catch it only when the token
  differs. The engine skipped the decode of a position that is not stale before this round too. What
  is new is that the copy of such a position is skipped as well.

Tests: `sequence_passes_decode_what_slice_passes_decode` (300 passes through a view against slice
passes, across a slice pass on the same engine and five dense fallbacks),
`an_unrecorded_change_stops_the_pass` and `a_store_out_of_step_stops_the_pass` (`engine/tests.rs`).

**The replans, located.** `GBCFORMER_STATS=1` now counts each replan's cause and brackets its two
halves by the thread's counters. On text (one model) the GBC CPU stream replans 29,601 times in
3,249,234 passes and control's stream 18,583 times. Every one but each stream's first is caused by a
moved output position: the request slots hold decoded tokens during a result pass and their masks
again in the next pass. The memo holds every one, and every one returns the plan already in place. So
a replan's work was all in finding that out. The base built the memo's key from every position of
the one model's sequence and hashed it with SipHash, 40,159 instructions and about 11,000 cycles
(the floor model's figure). A view's other positions never change token, so the key now holds the
stream's own positions, is built by index in a kept buffer, and is hashed eight bytes a step.
Separate checkpoints do not replan at all (1 replan a stream on the same run). The reason is inferred,
not measured: only the one model's output masks key layer 0 (`one_model::group`).

| Change | Measured |
| - | - |
| The view places the positions the device changed; the engine writes back the positions it decoded again | replay, one process, the sequence path against the whole-slice path: 7,831 against 7,985 cycles a pass, 29,902 against 30,852 instructions (median of 6) |
| The memory core and the generic device keep each slot's byte and stop the write and mask of every output | light lane, separate checkpoints (no view), machine thread a GBC CPU pass, with the row above: 39,266 to 37,932 instructions |
| `read_ports` reads the two read addresses, not a whole `decode_evidence`; the slot decoders check their slice's length once; a write record's finished step is rebuilt at its inputs and one pass's positions | same run: 37,932 to 37,579 instructions |
| A view keeps one positions id per named list | in the binary-against-binary figures below |
| A replan's memo key over the stream's own positions, by index, hashed eight bytes a step | a result pass after host-served bytes: 65,478 to 39,679 instructions and 22,436 to 15,965 cycles (sampled, 22 passes a side, each with one bracket); a replan's key and lookup now 2,690 cycles and the rest of its layer 0 1,247, each with a bracket's read |
| A pass that decodes half its positions or more writes every position back in one loop | the PPU's thread, instructions a pass: 210,249 at the base, 214,533 with a list entry per decoded pixel, 211,402 with this |

**Dropped.** The decoded positions kept as the decode's stale-and-wanted words and enumerated after it,
instead of a list entry per decode: 198 instructions a GBC CPU pass more (light lane, separate
checkpoints, 37,784 against 37,586), and no gain on the PPU's thread.

**Not built.** Masks at the request slots during a result pass would remove the cause of every replan
(about 4,000 cycles each with the brackets; 0.009 a GBC CPU pass on its own stream and 0.006 on
control's). It changes the tokens a result pass reads, and so the pass records the minimal runner and the
stock runtime's proofs compare. The decoded tokens are equal either way (an output reads no output).
The minimal runner still passes slices (`fill_into`), so its passes compare the whole sequence.

**Measured, binary against binary** (heavy-queue ticket of 2026-11-16 21:05; against the base, the tip
built the same way, both built with `cargo build --release`; all seven digests equal, one model `8d5b2f43`;
trace: 1,000,000 passes of the GBC CPU stream from pass 300,000 of text, recorded with the base).

| | base | tip | change |
| - | -: | -: | -: |
| Replay, cycles a pass (median of 6, three alternated pairs) | 8,116, 8,091, 8,064 | 7,851, 7,810, 7,814 | −3.3% (−265) |
| Replay, instructions a pass | 30,805 | 29,902 | −2.9% (−903) |
| Machine thread on text, a GBC CPU pass (`GBCFORMER_STATS=1`, A B B A) | 10,059 and 10,011 cycles; 39,334 instructions | 9,697 and 9,591 cycles; 36,446 instructions | −3.9% (−391 cycles), −7.3% (−2,888 instructions) |
| The PPU's thread, a PPU pass (same runs) | 56,091 and 56,740 cycles; 210,249 instructions | 57,302 and 56,767 cycles; 211,402 instructions | +1.1%, +0.5% |
| The APU's thread, an APU pass (same runs) | 20,758 and 20,795 cycles; 64,667 instructions | 20,755 and 20,468 cycles; 62,899 instructions | −0.8%, −2.7% |
| text, fps (median of 6, range; three ABBA rounds) | 56.9 (55.8 to 57.3) | 59.9 (58.9 to 60.4) | +5.3% |
| overworld, fps | 127.1 (125.6 to 129.1) | 128.3 (126.4 to 129.8) | no measurable difference |
| battle, fps | 76.6 (75.7 to 77.4) | 79.9 (79.2 to 80.3) | +4.2% |

The 1-minute load was 3.4 to 11.3 through the fps runs (other light-lane jobs); the sides
alternate. The machine thread's instructions are the same to 4 a pass between runs of one binary, so
the 2,888 is exact. Its cycles come from two runs a side, 48 and 106 apart: the four pairings of a
base run with a tip run give 314 to 468 fewer cycles, and 391 is their mean. The base read 56.9 fps
on text in this ticket and 55.8 in the one three hours before. The round's brief gave about
53.7 for this base; why it reads higher here is not established. The text fps gain is larger
than the thread's cycles give (3.9% fewer cycles is 4.1% more frames; the runs with the counters on
read 55.4 and 55.3 fps against 57.0 and 57.5, which is 3.4%). The rest of the 5.3% is not explained by
a measurement: the thread's counters are read only with the stats on, and the stats cost the base
1.5 fps and the tip 2.6. The PPU's thread
pays 0.5% more instructions a pass: its 163 positions decode nearly all each pass, so the store saves
it nothing and the device reads each byte through one more index.

Where the machine thread's 2,888 instructions went is not split by a measurement in the one model.
Separate checkpoints, which have no view and no replans, lose 1,650 of 39,266 (light lane), so the
device glue and the store are about that much and the view and the replans the rest (inferred).

Every run reached its hash with 0 dense fallbacks: text `870aa3fd`, overworld `822059c1`, battle
`d3534543`; Yellow 300 `--weights all --lockstep all --one-model` `cfd31f3d`; `--core lockstep
--hardware dmg` 8 frames `6872b003`. Text under the step oracle on every stream
(`cpu=400000,apu=200000,control=100000,ppu=100000` at this tip, and
`cpu=100000,apu=50000,control=10000,ppu=20000`): six streams checked, every token equal,
and with it every pass of every stream compared its whole sequence with the placement and its store
with the engine's tokens. The replay held all 1,000,000 passes' tokens to the recording in every
repetition, by both paths. `cargo test --release --lib` 198 passed; `cargo test --release --test
checkpoint_sst` 8 passed.

The passes the benchmarks seldom run, in a third ticket (`cargo test --release --features testing`,
the tests named): `host_fetch` (the fetch-port pass), `halt_bug_mem`, `halt_bug_locks`, `fetch_locks`,
`dma_records`, `oam_dma_runs`, `memory_dense`, `battery_saves`, `boot_overlay`, `interrupt_fuzz`,
`serial_link`, `apu_worker`, `ppu_checkpoint`, `ppu_dma`, `ppu_length`, `ppu_lines`, `ppu_stepper`,
`one_model`, `stream_audit` and `step_oracle_faults` all passed. Yellow 300 in lockstep ran 7
second-port passes. The memory checkpoint's OAM pass had no test through the engine; the new
`the_checkpoint_oam_pass_rewrites_what_the_circuit_does` holds it to the circuit's with the step
oracle on each pass, and `the_step_oracle_catches_one_changed_token_in_a_sequence_s_store` holds the
oracle to the store a device reads.

**Open.** The replans' cause stands (see "Not built"). On the machine thread outside the GBC CPU
stream's passes, the control stream's 57,576 passes at about 16,000 cycles each are about 285 cycles
a GBC CPU pass and the length stream's 86,617 at about 1,800 are about 50 (sampled figures less one
bracket); the rest of the thread's 9,644 is the host's code around a step, which this round did not
profile in the one model.

## Round 6a: the drain and the routing layer

Base: round 5 combined. The brief: the cost model puts `drain_wires` at 11% of the GBC
CPU pass and the routing layer at 7%; find why a touched channel costs 36 instructions to drain
and a routing update 68, and take 300 cycles off the machine thread's pass. The 300 were not
reached. The machine thread's pass lost 194 cycles by the means of two runs a side (the two pairs
give 271 and 117), and the replayed pass 119 by the paired median. The routing layer gave all of
it and the drain gave nothing.

**Why 36 and 68.** A census on text (separate cpu-memory checkpoint, light lane) and the
disassembly of both functions:

- The drain is 9 instructions a touched channel, 12 more when its change is not 0, and 14 a
  push. A pass touches 75.9 channels; 16.4 of them (22%) end with no change; the others have
  one reader (35.3), two (17.4), three (6.2) or four (0.5), which is 91.3 pushes. Nothing in it
  is slack: a load each of the mark, the sum, the wire and its range, and per push the
  bucket's end and limit, the cell and the new end.
- The routing layer is about 30 instructions a changed channel and 40 an update. A pass moves
  34.5 routing channels and 36.6 units, 14.9 of them fused pairs. No update reached a second
  owner, every one moved its owner's total, and 0.25 a pass sent on the blank row.

**Kept: the routing update.**

1. The blank row's activation change is taken once. It is what the unit sends, and an owner's
   total moves by c times (its change with the owner's token minus the blank change). The gate
   entry carries one owner and a flag for a second (24 bytes, was 36), and a channel's entries
   are read under one bound.
2. A linear unit skips the clamps. Most routing units only move a field into a route channel:
   with their owner's token the gate stays where the clamp is the identity, and on the blank
   row it stays at 0. `Owner::linear` proves that for a unit with one owner from the span its
   wires can add to its gate. Such a unit's update is `gate += w·dx`, `total += c·w·dx` and the
   stale mark. The proof reads the ranges the pass guards: `Template::check` narrows
   `seg.range` to the declared range for the wires of units that are linear over their
   declared ranges (as `Positional::split` does for the wires its pruning reads), and
   `Template::index` proves each unit again from `seg.range`. Linear units among those with an
   owner, counted with separate checkpoints (the one model's evidence is the 767 fewer replay
   instructions a pass): 87 of 91 in the CPU checkpoint; 46 of 46, 49 of 49, 505 of 583, 557 of 557, 606 of
   619 and 623 of 623 in the other plans of a run with every component in weights.

**Measured** (heavy queue, 2026-11-16 23:15; the base, R1 = change 1 alone, the tip
plain builds; every digest equal to the base's, one model `8d5b2f43`).

| | base | R1 | tip | tip against base |
| - | -: | -: | -: | -: |
| Replay, cycles a pass (median of 6, three alternated rounds) | 7,744, 7,768, 7,659 | 7,654, 7,631, 7,640 | 7,611, 7,649, 7,588 | −134, −119, −71 by round; −119 (−1.5%) |
| Replay, instructions a pass | 29,675 | 29,444 | 28,678 | −997 (−3.4%) |
| Machine thread on text, cycles a pass (`GBCFORMER_STATS=1`, A R B B R A) | 9,757, 9,648 | 9,621, 9,652 | 9,486, 9,531 | −271 and −117 by pair; −194 (−2.0%) by the means |
| Machine thread on text, instructions a pass | 38,032 | 37,805 | 37,063 | −969 (−2.5%) |
| text, fps (median of 6, range; three ABBA rounds) | 59.35 (58.9 to 59.7) | | 60.1 (59.2 to 60.5) | +1.3%, ranges overlap |
| overworld, fps | 128.65 (123.1 to 131.3) | | 129.75 (128.4 to 130.2) | +0.9%, ranges overlap |
| battle, fps | 79.8 (78.3 to 80.7) | | 81.95 (79.4 to 82.1) | +2.7%, ranges overlap |

The pass changes the same 134.9 (channel, reading layer) pairs and decodes the same 11.2
positions on both sides, so the stale marks are the same. An earlier ticket the same evening
gave R1 alone 9,565 and 9,540 against 9,419 and 9,442 cycles (−122); in this one it is −66. The
machine thread's own figure moves by about 100 cycles between tickets, so the split between the
two changes is not resolved. Change 2 alone, the tip against R1: −43, +18 and −52 cycles by replay
round (0.6%, under the 1% bar) and −135 and −121 on the machine thread (1.3%). It is its own
commit and adds a proof, so it can be left out without change 1.

The 1-minute load was 2.4 to 5.2 through the fps runs. Text's median is over 60 and its range is
not: one of the six tip runs read 59.2. Every one of the 36 fps runs and the 6 stats runs reached
its hash with 0 dense fallbacks: text `870aa3fd`, overworld `822059c1`, battle `d3534543`.

**Gates on the tip** (the commit under this log entry): the seven digests equal to the
base's; Yellow 300 with every component in weights, in lockstep, as one model `cfd31f3d` with 0
dense fallbacks; `--core lockstep --hardware dmg` 8 frames `6872b003`; text with
`GBCFORMER_STEP_ORACLE=cpu=200000,apu=100000,control=20000,ppu=40000`, six streams checked,
every token equal, `870aa3fd`; `cargo test --release --lib` 201 passed; `cargo test --release
--test checkpoint_sst` 8 passed; the replay held all 1,000,000 passes' tokens in every
repetition; the wasm32 `cargo check` is clean. Change 2's narrower guard has run only on
Yellow here. The other games, acid2, `dmg_sound` and `instr_timing` are the gate's, and a wire
outside its declared range would show there as dense fallbacks. The R1 binary measured was built
from the same sources as the first commit before its tests and two comment lines were added.

**Guards.** Nothing a pass refuses changed, except that the wires linear units read are now
held to their declared ranges, which is narrower. Each new test fails on its mutant at the
assertion the mutant predicts: the owner's part without the blank term, no second owner and no
stale mark fail `routing_update_keeps_the_own_totals_and_sends_what_the_scalar_walk_sends`
(own route totals differ; home 3 moved and is not stale); the bucket without its overflow flag
fails `a_full_bucket_drops_the_push_and_says_so`; `Owner::linear` without its blank-row term
fails `a_linear_owner_moves_by_its_weight_over_the_whole_span` (gate −9, own 8, wires 0 to 9).
With every owned unit called linear, Yellow with every component in weights and in lockstep
does not finish 12 frames in 500 s, where the tip takes seconds; what stops it was not read
(inferred: refused passes on the dense path). DMG 8 on the CPU checkpoint alone does not see
that mutant (inferred: its 4 units that are not linear did not move in 8 frames).

**Tried and dropped.** Light lane, separate cpu-memory checkpoint, text; machine-thread cycles
a pass are contended there (the same binary reads about ±100) and instructions are exact.

| Change | Instructions a pass | Cycles | Why |
| - | -: | - | - |
| Lazy routing: no own route totals, each decode sums its token's own units | +1,468 | +3.5% | 60 own units a pass for 11.2 decoded positions cost more than the 36.6 updates |
| Bucket push without a test (a spare cell a bucket, the end clamped to it, overflow found at the clear and at the tail) | +251 | none | the overflow branch was free |
| Drain: one push a channel whatever it holds (a zero change and no reader go to a sink bucket) | +851 | none | as round 5a's D1, now with a push no longer than the branching one |
| Drain: two such pushes, 47 instructions a channel in a straight line | +1,390 | +1% to +4% | the drain is not bound by its two data branches |
| Routing sends staged as `layer` stages them | −10 | none | the walk kept its spills |
| Routing update without the branch on the pair flag | +461 | none | the branch is predicted, as the kernel round found |

A microbenchmark of the push (200M pushes, `rustc -O`): through the bucket's end in memory 2.19
ns a push to one bucket against 0.82 ns through a register, 1.58 against 0.83 over two buckets,
1.21 against 0.82 over four. A chain of pushes to one bucket costs about 7 cycles a link, but a
drain call spreads about 12 pushes over about 5 buckets, so the chain does not bound it
(inferred).

**What bounds the rest.** The drain's cost is its pushes and its touched channels, 91 and 76 a
pass. Five engine-side variants of the drain over two rounds lost (round 5a's three and the
three here that touch it), and its instructions are loads and stores it needs. Fewer pushes or
fewer touched channels is a circuit change. The 22% of touched channels that end with no change
are a guess at one: sends that cancel within a node, not measured. The routing layer is now about 30 instructions a changed channel and about 20 fewer an
update than it was (767 fewer a pass over 36.6 updates); what is left is the bucket hop itself, a push in the drain and a table lookup here for
each of 34.5 channels.

## Round 6b: the minimal runner over 60 fps on text

Base: round 5 combined. The full program read 62.6 to 63.4 fps on text, but the
minimal runner (`tools/runner`, `docs/minimal-runner.md`) was last timed at 48.9, four rounds of
engine work earlier, and had none of the host-side changes since. This round measures it and
ports what it lacked. Every change is in `tools/runner`; `git diff` of the base against this work over `src examples
Cargo.toml Cargo.lock` is empty, so no digest can move and the full program's binary is the
base's. The `digest` example was not run; the GGUF exported at the base carries the one
model's tensors sha1 `8d5b2f43`, the digest the base's verify printed.

**Measured first.** At the base the runner read 58.0 to 59.2 fps on text (6 runs), against
the full program's 63.1 to 63.7 in the same rounds. The engine's gains since the last timing
had reached it, since it links the same engine. What it lacked was the host's side.

**Ported, one commit each.**

1. Sequences. `Seq` holds a `sequence::Sequence` and every pass goes through
   `Denoiser::fill_seq`, as the full program's devices do since the view round. The engine places the
   positions set since the last pass and writes back the positions it decoded again, where
   `fill_into` compared the whole sequence and copied every decoded token. Every output holds
   its mask token. A result pass (the steps and the spans streams) reads the request's decoded
   tokens at their slots, placed for that pass alone. A write record reads the finished step
   from a buffer built for it. A request sets each input once. The tokens each pass reads are
   the ones it read before, which the pass records below hold.
2. The rows worker. With `--threads 1` the steps stream's cells defer their K/V
   rows (`Cells::defer`). The runner settles them before it writes a save and before it reads
   its clock and reports. `--threads 0` keeps them on the calling thread.
3. A step's buffers. Four small vectors a step allocated (its composed inputs, the
   first request's access bytes, and the late bytes of the request and of the result pass) are
   kept between steps.

Nothing else was ported. After the three commits the timings below do not tell the two
programs apart, and one `sample` pair on text (8 s each, the runner with step buffers kept and the full
program) shows the same functions on top of the stack in nearly the same order, the
engine's block walk first, and the runner's own code (`gbcformer_runner::run`) on top of the
stack in 130 samples against the walk's 2,019.

**Timings.** Plain builds (`cargo build --release`), 600 frames, the runner from the bench's
hand-off snapshot with the bench's presses. The full program ran with every component in
weights as one model, from the same GGUF as the runner (`GBCFORMER_GGUF_ONLY`) in the paired
rounds, and compiling its weights in two more runs. Ticket 1 (2026-11-17 09:05, load 2.7 to
6.5) alternated the base, each commit and the full program, three rounds of A B C D G G D C B A:

| text, fps | median of 6 | range |
| - | -: | - |
| runner, base | 58.6 | 58.0 to 59.2 |
| runner, sequences | 61.1 | 59.9 to 61.4 |
| runner, rows worker | 61.95 | 61.6 to 62.7 |
| runner, buffers | 63.4 | 62.3 to 64.3 |
| full program, from the GGUF | 63.3 | 63.1 to 63.7 |
| full program, compiled (2 runs) | | 63.3, 63.4 |

Ticket 2 (10:06, after the proofs below; load 2.1 to 4.4) alternated the base's runner, the
tip's and the full program, three rounds of A B G G B A:

| 600 frames, fps | runner, base | runner, buffers | full program, GGUF | full program, compiled (2 runs) |
| - | -: | -: | -: | - |
| text | 59.3 (58.8 to 59.7) | 63.2 (62.1 to 63.8) | 63.55 (63.3 to 63.9) | 63.1, 63.2 |
| overworld | 128.05 (126.8 to 130.5) | 131.95 (131.0 to 133.4) | 131.3 (130.9 to 133.8) | 131.5, 131.6 |

All 12 text runs of the tip's runner are over 60 (lowest 62.1). The runner and the full
program are not told apart: their ranges overlap on both workloads in both tickets. Sequences
are worth about 2.5 fps on text, and the range of each later commit overlaps the one before
it or nearly does, so the split between the rows worker and the buffers is not resolved. One
figure is not explained: in ticket 1 the full program from the GGUF read 121.9 to 126.3 on
overworld against 131.4 and 132.7 compiled, and in ticket 2 it read 130.9 to 133.8 against
131.5 and 131.6. Every timed run reached its frame (`870aa3fd`, `822059c1`) with 0 dense
fallbacks.

**Gates** (ticket 2, the runner with buffers, the one model's GGUF exported at the base with
`--layer-shapes --align 64`). The runner's tests with the GGUF: 24 passed, the slot-name scan
included. Every runner proof `tools/gate.sh` runs, with its flags, each with every pass record
equal to the full program's on all six streams, frames, audio and saves equal, and no dense
fallback, panic or refusal: Yellow 300 (`995267 83499 251675 71760 41856 1` record lines),
the same on one thread (`--threads 0`, rows in place), Yellow from the text snapshot 2900 to
3100, Crystal from a snapshot past frame 200's end, Mealybug `m3_bgp_change` (the stepper
drawing every line), the display read forwarded from the record queue, the COMPAT record with a
row open, blargg `oam_bug` 2-causes on DMG, fetch-locks on DMG, the HALT bug across a lock edge,
mooneye `lcdon_timing-GS` and `lcdon_write_timing-GS` on DMG, Mealybug `hdma_during_halt-C` and
`hdma_timing-C`, and Yellow 300 from power-on. In ticket 1 the text-snapshot proof also passed
with the runner built with debug assertions, where the engine holds every pass's change list
and decoded store to the whole sequence.

Not run: `same-passes.sh` and `slice_full`, which run the full program alone, and the rest of
`tools/gate.sh`; no file they build from changed.

**The ratchet.** `tools/runner/tests/branches.txt` went from 497 rows to 501: 446 to 449 of
class a, 42 to 43 guards, 9 of class b, none of class c. The rows that left named lines the
port changed or removed (the slice decode's count check among them). The rows added are the
changed lines, two start-up assertions on the wiring's shape that the sequence path relies on,
and `if o.threads {` for the rows worker. No literal, term or slot name was added to the lists
in `tests/quarantine.rs`.

## Round 6c: two evidence gaps in the runner port, closed

Base: round 5 combined. A review of Round 6b found no defect and two gaps in what its
proofs exercised. This round closes both with runs and keeps the two checks in `tools/gate.sh`.
The diff of `src examples Cargo.toml Cargo.lock` against the base is empty, so no digest can move. One
ticket, 2026-11-17 11:52 to 12:29, load 2.1 to 9.5, the one model's GGUF
exported from the same tree with `--layer-shapes --align 64` (2,041,927,328 bytes). No fps was
measured; the seconds below are run times under that load, given to size the gate.

**W1, the placement after a write record.** In a recording run `Engine::keep` places the
record's tokens in the view (`src/engine/records.rs:271-279`, under the `record` feature and
only with a recorder), and `View::place_tokens` drops the placement stamp
(`src/engine/view.rs:161`). The next `fill_seq` is then not synced (`src/engine/run.rs:1537`)
and places the whole sequence (`run.rs:1586-1589`). Without the feature, or without
`GBCFORMER_RECORD`, `keep` does not touch the view. The stamp has four writers (`view.rs:105`
at construction, `view.rs:161`, `run.rs:1584` and `run.rs:1588`) and `place_tokens` three
callers (`records.rs:277`, `run.rs:1587` for a pass that is not synced, `run.rs:1590` for
`fill_into`, which the runner no longer calls). So in a plain build every pass of a stream
after its first is placed by its change list (`run.rs:1582-1585`, `View::place_changed`), the
pass after a keep included. With debug assertions `Engine::verifies` is true (`run.rs:1557`):
`place_changed` then compares the whole sequence with the placement and stops at a token the
change list missed (`view.rs:151-153`), and the pass compares every decoded position's stored
token with the engine's (`run.rs:1378-1382`). Every proof of Round 6b was a recording run, so
none ran those checks on the pass after a keep.

The run: the runner built plain (`cargo build --release`, no `record` feature) with
`CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true`; the build log shows `debug-assertions=on` and no
`record` feature on both the `gbcformer` and the `gbcformer_runner` crates. With `--threads 1`
and no `GBCFORMER_RECORD` (`PLAIN=1` in `tools/runner/proof.sh`, which also requires that no
record file was written), its outputs against the full program's from the recording proof of
the same case:

| Case | Frame hashes, one a frame | `audio.raw` | Saves | Steps stream |
| - | - | - | - | - |
| Yellow 300 | 300 equal (last `cfd31f3d`) | 964,376 bytes equal | none on either side | 925,752 passes, all fast |
| Yellow from the text snapshot, frames 2900 to 3100, the bench's presses | 200 equal (last `7cb69988`) | 642,920 bytes equal | 3 of 32,768 bytes, equal | 961,060 passes, all fast |

No assertion fired, and no stream took a dense fallback. Yellow 300 writes no save, so its save
comparison is empty. The recording proofs of the same four cases (Yellow 300, the
text snapshot, fetch-locks on DMG, `m3_bgp_change`) passed first, every pass record equal.

**W2, the rows the worker writes.** A lookup reads the calling thread's facts, not the K/V rows
(`src/engine/cells.rs:650-665`), so with `--threads 1` the pass records held the worker's rows
only where a scan read them, and the runner did not print how often. The runner now prints,
for each stream it reports (all but the stepper, whose cells no worker writes), its cell reads by lookup, by full scan, and the lookups the scan also checked
`SCAN=1` in `proof.sh` sets `GBCFORMER_SCAN=1` for the runner alone, under which
`Cells::read` runs every lookup again as the softmax over the stored rows and stops at a
difference (`cells.rs:640-646`); the rows are settled before each read
(`src/engine/rows.rs:283-285`). The proof requires the steps stream's checked count to be above
0 and equal to its lookups. All with `--threads 1`:

| Case | Proof | Steps stream's lookups, each also read through the rows | Run time |
| - | - | -: | -: |
| Mealybug `m3_bgp_change`, 4 frames | every check, pass records equal on six streams | 112,997 | 192 s |
| fetch-locks on DMG, 24 frames | every check, pass records equal, 76 OAM bug passes on both sides | 498,687 | 227 s |
| Yellow from the text snapshot, 1 frame (runner alone, the plain build with debug assertions) | frame hash equal to the full program's | 14,032 | 56 s |
| the same, 8 frames | 8 frame hashes equal | 88,984 | 365 s |

The whole 200 frames of the text snapshot under the scan project to about 11,200 s (56 s a
frame), so that proof was not run; the 8 frames are the part of it that fit. The lines and
length streams' cells, which no worker writes, were scanned too (`m3_bgp_change`: 3,284 and
3,678 lookups, all checked).

**In the gate.** `tools/gate.sh` now runs both: the plain build with debug assertions on Yellow
300 and on the text snapshot (43 s and 42 s here, plus one more build of the runner; the full
side is the recording proof's, `REUSE=1`), and `m3_bgp_change` again under the scan (192 s).
`RUNNER_CHECKED_BIN` hands the gate a build made elsewhere. The gate as a whole was not run.

**Hygiene.** The three loops that set every input to 0 in the length, lines and stepper
streams' constructors are gone: `Seq::new` starts every input at byte 0. Their
three class-a rows left `tools/runner/tests/branches.txt`, now 498 rows (446 a, 9 b, 43
guards); the scan count's print and the proof's modes add none. The runner's tests with the
GGUF: 24 passed. `wiring::Steps::unread` stays: the field is part of the wiring table the GGUF
carries, so dropping it changes `src/` and every exported file.

## The llama.cpp pass rate since 2026-11-04, stream slices and three harness fixes

The 300-frame Yellow proof in llama.cpp took 17.7 h on 2026-11-04 and on 2026-11-06.
With the harness before this work (one model `8d5b2f43`) it projected to 38 to 40 h after the scope fix. This entry
says where the time went and takes back what the harness can take back without changing a
weight. Nothing under `src/` changed and the GGUF is the same file.

**The two earlier runs are not one baseline.** The 2026-11-04 run (31.1, 38.3, 23.3 and
18.9 ms per GBC CPU, PPU, APU and control pass) used the model from before the overlay, with
per-part FFN slices, no up projection and per-part LM head ranges (`docs/slice-stack.md`,
H8, H9, H11). The 2026-11-06 run used the overlay model `31307a65` with only
`layer-shapes.patch` live (`docs/stock-llama.md` §12a), so llama.cpp ran the whole model for
every pass, at 26.1, 26.8, 17.7 and 18.2 ms in `bench.sh`. The second is today's ancestor.
`proof.sh` and `bench.sh` have not changed since that run, and the llama.cpp build is the
same Release Metal build with `layer-shapes.patch` alone. No ms-per-pass row was recorded
between 2026-11-06 and 2026-11-14, where the GBC CPU pass first shows at 70.7 ms
(`47ceed72`).

**What a pass computes, then and now** (read from the GGUFs' headers and the docs, not
measured):

| | `31307a65` (2026-11-06) | `8d5b2f43` (now) |
| - | - | - |
| Hidden, layers | 1,856, 14 | 2,560, 25 |
| Values in the layer-shaped GGUF | 366.3 M | 1,018.1 M |
| FFN units stored, summed over layers | not recorded | 77,376 |
| FFN multiply-adds per token (three matrices) | about 155 M (inferred: the file less its embeddings and attention) | 594 M |
| Vocabulary (LM head multiply-adds per decoded row) | 36,311 before the overlay | 65,804 (168 M) |
| Tokens per GBC CPU pass | not recorded (89 on the model before the overlay, `docs/slice-stack.md`) | 144 |

The FFN work of a GBC CPU pass grew about 6 times if the group then held about 89 tokens
(inferred: 3.8 times per token, 1.6 times the tokens), and the measured pass 3 times.
llama.cpp still ran every stored unit, head and
layer for every stream, while the engine runs each stream's slice. The GGUF already names
those slices (`gbcformer.stream.<part>.*`, `examples/export.rs` `stream_keys`), and the
harness never read them:

| Stream | Layers of 25 | FFN units of 77,376 | Query heads of 240 | Decoded rows per pass |
| - | -: | -: | -: | -: |
| GBC CPU | 11 | 10,830 (14.0%) | 63 | 72 |
| PPU | 14 | 17,597 (22.7%) | 68 | 163 |
| APU | 25 | 18,372 (23.7%) | 14 | 71 |
| control | 17 | 9,341 (12.1%) | 13 | 93 |
| length | 15 | 6,424 (8.3%) | 38 | 5 |
| stepper | 9 | 16,541 (21.4%) | 53 | no pass in this window |

**Where the wall time went** (measured, one 5 s `sample` of the harness's main thread on the
whole model at load about 30, a run that is not reported). 45% was the wait for the GPU.
38% was `argmax` over the logit rows, one comparison at a time over 65,804 logits for each
of about 72 rows, and it ran after the pass's timer stopped, so no `ms per pass` line ever
showed it. 9% was llama.cpp's host side of a decode (146 of its 299 samples in
`llama_kv_cache::find_slot`), and 5% copying rows. The killed proof shows the same gap
over 850,000 passes: 143,851 s elapsed against 124,393 s inside passes, writes and reads
and 949 s of prefill, which leaves 21.8 ms per pass outside every counter, 5.1 h by then.

**Four changes**, all in `tools/stock-llama`:

1. **`stream-slices.patch`** (on b12070 after `layer-shapes.patch`; `run.sh setup` applies
   both). The harness names the pass's stream before each decode (`shim_stream`, found with
   `dlsym`, so a build without the patch runs the whole model as before). The graph then
   runs the stream's `block_count` layers, its heads and its FFN units, each as a view of
   the layer's own matrices: rows of `attn_q`, `ffn_gate` and `ffn_up`, strided runs of
   columns of `attn_output` and `ffn_down`. Every unit left out has its `up` row on another
   part's mode channel, which is 0 at every token of the stream's group, and every head left
   out attends the null cell (`Model::stream`). Records and the prefill still run the whole
   model, so the cells' rows are the saved state's. One sliced pass decoded a wrong token:
   6 output rows through the GBC CPU's last `ffn_down` run, which starts at unit 2,047,
   gave another position's token, while the whole model on the same 6 rows was right. Metal
   sends a matmul of 3 to 16 rows to its few-row kernels, so the bound of 16 and the cause
   (an unaligned f16 matrix load from a view of columns) are inferred from the dispatch
   code, not shown; the length stream's sliced passes of 5 and 10 rows had passed the bit
   check. The patch now keeps every sliced matmul at the batch's full row count: it picks
   the output rows after the last layer, and a batch of at most 16 tokens (the length
   stream's) runs the whole model.
2. **`argmax` in vector code** (`cpp::argmax`): 16 running maxima, then the first index of
   the largest. A test holds it to the one-comparison version over 20,000 random rows with
   NaN, infinities, signed zeros and ties. The pass's timer now covers it.
3. **Logit rows only at decoded positions.** Every mask of a group still attends the cells
   (`split-check` fact 4), but only the decoded positions ask for a row. A second GBC CPU
   pass decodes 6 positions and used to compute and copy rows for every mask.
4. **Write records over the prefill lanes' cells** (`Cache::scope_records`, 2,521 cells for
   Yellow). A lane's record already decodes over exactly those cells, and the whole saved
   cache was built that way. A write record over every cell cost about 200 ms.

**Checks.** `STOCK_STREAMS=check` decodes every pass by the slice and by the whole model
and compares every logit bit: 400 passes of each of three builds (FFN slices, with heads,
with all four changes), 0 rows differ, every pass equal to the engine's record.
`STOCK_WRITES=check` decodes a write record both ways and compares the two cells' saved
bytes (`llama_state_seq_save_file` of the one cell, 7,048 bytes: its position and every
layer's key and value row). Over 18,000 passes from the saved cache it checked 844 write
records, 820 of them the GBC CPU's, and 0 cells differ; every pass of that run equals the
engine's record (13,176 GBC CPU, 3,124 control, 1,260 APU, 288 PPU, 152 length) and frame 1
hashes to `d043ab24`. That run replaces each checked cell with the whole-cache one, so the
same 18,000 passes ran again at the defaults (`STOCK_STREAMS=require`), where later passes
read the scoped cells: every pass equal, frame 1 at `d043ab24` after 393 s, 569 s in all
at load 6, the 844 write records at 27.9 ms each. `STOCK_SCOPE=check` over 300 passes with
all four changes decoded 297 scoped, sliced passes again by the whole model over every
cell, 0 rows differ. Every timed run below ran with `STOCK_REFERENCE`, so each of its
passes equals the engine's record: 3,600 passes a side. Not checked: the stepper's slice
(it has no pass in these windows), a reader record past the first, a bank switch, and any
frame past the first; the PPU's slice was bit-checked on 24 passes.

**Measured** (Yellow from the saved cache `8d5b2f43`, 1,200 passes a run, three rounds, the
three sides interleaved inside one hold of the GPU lock a round, order rotated, `nice 19`,
a full verify and other work on the machine throughout; A is the earlier harness on
the unpatched build, B the slices alone, C all four changes; ms per pass as the counts line
prints it, which leaves `argmax` out on A and B and holds it on C; wall is elapsed time over
passes 200 to 1,200 of the mix, 65% GBC CPU, 25% control, 6% APU, 2% PPU, 2% length):

| Side | Load, rounds 1 to 3 | GBC CPU | PPU | APU | Control | Length | Wall per pass | Write record |
| - | - | - | - | - | - | - | - | - |
| A, before | 18 to 19, 18 to 20, 5 to 6 | 76.5, 77.0, 70.0 (**76.5**) | 68.7, 69.7, 58.7 (68.7) | 51.4, 53.2, 46.6 (51.4) | 53.6, 53.6, 47.5 (53.6) | 16.3, 17.4, 11.5 (16.3) | 89, 90, 80 (**89**) | 201.3, 203.7, 194.7 (201.3) |
| B, slices | 18 to 24, 20 to 11, 11 to 6 | 41.3, 33.9, 33.7 (33.9) | 53.0, 42.9, 42.7 (42.9) | 36.4, 30.7, 30.7 (30.7) | 26.8, 20.0, 19.9 (20.0) | 10.6, 7.0, 6.5 (7.0) | 59, 49, 48 (49) | as A |
| C, all four | 24 to 22, 22 to 18, 6 | 38.0, 37.8, 31.4 (**37.8**) | 51.0, 51.2, 46.4 (51.0) | 38.5, 38.6, 32.5 (38.5) | 29.3, 29.8, 22.1 (29.3) | 15.1, 15.9, 11.8 (15.1) | 37, 36, 29 (**36**) | 40.6, 39.0, 32.0 (39.0) |

The wall time per pass falls from 89 to 36 ms, 2.5 times, and a write record from 201 to
39 ms. The slices take 40 ms of the 53 and the harness fixes 13. What bounds a pass now is
still the GPU's matmuls: the stream's own FFN, and an LM head that scores all 65,804 tokens
at every decoded row, about as many multiply-adds as the GBC CPU's sliced FFN (72 rows of
168 M against 144 tokens of 83 M; counted, not profiled). Load moves every side: A is 12%
slower at load 18 to 20 than at 5 to 6, and C 25%, since less of C's pass is GPU work.

**Projected to frame 300** (the reference's passes: 925,752 GBC CPU, 83,498 PPU, 251,674
APU, 71,759 control, 41,855 length; about 49,600 write and 14,900 reader records, inferred
by scaling the killed proof's 39,198 and 11,741 at 731,803 GBC CPU passes to the
reference's 925,752; the 16-minute prefill):

| | Passes | Outside the counters | Write records | Reader records (319 ms each in the killed proof, unchanged) | Prefill | Total |
| - | - | - | - | - | - | - |
| A at load 18 to 20 (the medians) | 26.1 h | 8.0 h (21 ms a pass) | 2.8 h | 1.3 h | 0.3 h | **38.5 h** |
| C at load 18 to 24 (the medians) | 14.4 h | 0.2 h (0.4 ms a pass) | 0.5 h | 1.3 h | 0.3 h | **16.7 h** |
| C at load 5 to 6 (round 3 alone, one run) | 12.0 h | 0.2 h | 0.4 h | 1.3 h | 0.3 h | **14.2 h** |

"Outside the counters" is the wall time per pass less the mix's weighted ms per pass, taken
as the same for every stream (inferred; on A it is mostly `argmax`, which grows with the
decoded rows). The elapsed time prints in whole seconds, so it resolves 1 ms a pass.

The rates are the first 1,200 passes of boot. The killed proof's GBC CPU rate over 731,803
passes (77.7 ms) sits inside A's range here, so the window is taken as representative; that
is inferred. No full run was made.

On a llama.cpp build without `stream-slices.patch` the other three changes still apply.
That case was not timed; from A's rows it comes to about 29 h (26.1 h of whole-model
passes, about 1 h of vector `argmax`, 0.5 h of writes, 1.3 h of reads, the prefill;
inferred). `play` now says for each stream whether its passes run the slice or the whole
model, and `proof.sh` runs with `STOCK_STREAMS=require`, which refuses a build that cannot.

**Not done, with what each would give** (estimates from the counts above, none measured):

- An LM head range per stream, as H11 had: 2.5 to 3.5 h. It needs the proof that no token
  outside the range can win, which lived in `src/` (`one_model::head_range`), or a stream
  key in the GGUF, so it changes the export.
- No up projection for the sliced units (H9): about 1 h. The slices' `up` rows read the
  mode channel, not the constant, so it needs the fold `Model::stream` applies.
- Reader records over the GBC CPU's scope: about 1.1 h of the 1.3. The scope must hold the
  record's source and destination lines, which the windows do not guarantee.
- The rows' copy out of llama.cpp (19 MB a GBC CPU pass) and `find_slot`, 3% and 5% of
  the main thread in a sample of C: about 1 h together.
- A proof that resumes from a frame boundary, so it can run overnight only: no hours
  saved, but no daytime hours taken.

## Round 9: the parts share the output identity channels

One change to the one model's layout. It moves only the one model's digest, `8d5b2f43` to
`45bfb056`. cpu `bebf5d93`, cpu-memory `836fb6de`, apu `d8d3a06e`, control `6e225675`, ppu
`79706b58` and ppu-stepper `4fdab521` are unchanged (`examples/digest` on the tip). The reason
is size: 367 residual channels that every row, token and stream paid for held one output
slot's identity each, and an identity only has to tell a part's own slots apart.

| | Identity channels | Wire base | Hidden | Uniform | Stored (align 64) | Stored (align 1) | Embedding and LM head |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 568 (38 input groups, 530 output slots) | 910 | 2,560 | 2,962.6 M | 1,018.1 M | 1,011.2 M | 336.9 M |
| tip | 201 (38 input groups, 163 shared) | 543 | 2,176 | 2,518.2 M | 865.4 M (865,410,432) | 859.5 M | 286.4 M |

Measured with `examples/shape --one` and `examples/digest`; the base row is the census run of
the same example on the base. Stored size falls 152.7 M (15.0%). Layers (25), heads (69 a
layer at most), FFN units and the vocabulary (65,804) are unchanged: the per-layer kept units
and heads that `shape --one` prints are equal line for line, and the nonzeros stay 1.85 M.

**The change.** A part's k-th output slot holds the k-th output identity channel
(`Layout::identities`), so the six parts (163, 119, 71, 93, 5 and 79 output slots) share the
PPU's 163. Within a part the identities are distinct, so a part's rows read what they read
before. `docs/one-model.md` §4 ("The parts share the output identity channels") has the
argument for every row that reads one, and
`tests/one_model.rs` (`every_reader_of_an_output_identity_is_gated_by_its_part`) reads it
from the compiled weights: 4,223 MLP units, each with `up` on its part's mode channel alone;
no memory head query; 7 layer-0 key rows (the fallback dim and the record gathers' six code
dims); no value, `up` or LM head row; no writer.

**The engine.** A stream's network has no mode gate (`Model::stream` folds it), so another
part's output mask that carries one of the stream's identities reaches the stream's routing
units. The stream's layout now names its own output slots (`Layout::own_outputs`, read from
the masks' mode channels). `Template::check` tracks a routing unit's owner among those slots
only, and the positional stage maps an identity channel to a slot among them
(`Layout::slot_of_id`). Without the first, a routing unit would have up to six owners where
the table holds two (inferred from the code; the refused plan was not run).

**Each stream's slice**, base then tip (`shape --one`):

| Stream | Prefix | Slice hidden |
|---|---|---|
| PPU scanline | 2,361, 1,994 | 2,368, 2,048 |
| GBC CPU with memory | 2,132, 1,765 | 2,176, 1,920 |
| APU | 1,989, 1,622 | 2,048, 1,664 |
| control | 2,029, 1,662 | 2,048, 1,664 |
| PPU length | 1,484, 1,117 | 1,536, 1,152 |
| PPU stepper | 2,531, 2,164 | 2,560, 2,176 |

The GBC CPU's prefix rounds to 1,792, but its slice is 1,920: `Model::slice` widens a slice
until the squares its nibble wires can hold stay below 16 times its width, and at 1,792 they
do not. The base's 2,176 had that room already. The whole checkpoint's norm floor does not
bind (floor 0 in `shape --one`).

**Gates on the tip** (heavy lane, load 4 to 30, 2026-11-19):

- `cargo build --release --all-targets`: no warnings.
- Yellow CGB 300, all five components in weights as one model, in lockstep: `cfd31f3d`, 0
  dense fallbacks. The same run with the step oracle
  (`cpu=100000,apu=25000,control=6000,ppu=8400`): `cfd31f3d`, 10, 11, 12, 10 and 5 fast
  passes checked.
- `cargo test --release --features testing`, in two tickets. The first ran on the tree
  before the commit and stopped at `tests/one_model.rs`: the 16 test files before it passed
  (the 208 library tests among them), and in it the slice test, the stream test and the
  record test passed, while the new test passed every check of the weights and failed its
  own closing count, which expected a head query to read an identity where none does. The
  second, on the tip with the count corrected, ran that file and the nine after it
  (`ppu_checkpoint`, `ppu_dma`, `ppu_length`, `ppu_lines`, `ppu_stepper`, `serial_link`,
  `sm83_sst`, `step_oracle_faults`, `stream_audit`): 10 files ok, 0 failed. `stream_audit`
  holds the one model's f16 checks (`check_f16`, `check_f16_acc`) and each stream's group.
  The doc tests did not run in either ticket.
- The base's digests (`examples/digest` built at the base): the six above equal, one model
  `8d5b2f43`. `digest --gguf` on each side's layer-shaped export: 0 values differ from the
  compiled checkpoint, tensors sha1 `45bfb056` on the tip, and the wiring's sha1 is
  `81ccc726` on both, so the wiring table did not move. The file went from 2,041,927,328 to
  1,736,448,928 bytes.
- The runner's own tests on the tip's GGUF: 24 passed.
- The engine's counts on the text bench, every component in weights as one model from each
  side's GGUF (`GBCFORMER_UNITS=1 GBCFORMER_STATS=1`, not timed): for all five streams the
  passes, the units re-evaluated per pass (GBC CPU 715, APU 3,020, control 1,369, PPU 42,459,
  length 165), the changed (channel, reading layer) pairs (134.9, 254.8, 167.0, 15.9, 4.1),
  the units by layer and the positions decoded are equal on both sides, with 0 dense
  fallbacks. The plans do the same work; only channel numbers differ.

**Timing** (600 frames of the text bench from the frame-2,900 hand-off, both sides built
with `cargo build --release`, each from its own GGUF and its own snapshot; three ABBA rounds
in one heavy ticket, load 3.9 to 6.7; every run reached `870aa3fd` with 0 dense fallbacks):

| Text, 600 frames | Base (fps) | Tip (fps) |
|---|---|---|
| Runner (`tools/runner`) | 62.9 (62.6 to 63.0) | 64.1 (63.1 to 64.3) |
| Full program, one model from the GGUF | 61.55 (61.2 to 62.3) | 61.65 (60.7 to 61.8) |

Medians of six with ranges. The runner stays above 60 fps in every run and is not slower: its
tip median is 1.9% above the base's and the two ranges do not overlap. The full program shows
no measurable difference. The engine re-evaluates the same units on both sides, so the
runner's gain is not fewer units; its cause was not profiled (a guess: narrower rows and
tables in the cache). llama.cpp was not run; each of its passes multiplies rows 12% to 25%
narrower per stream (the slice widths above), which should lower its ms per pass (inferred).

**Stale after this commit.** Every exported file of the one model, the llama.cpp proof's
reference record and saved cache states (kept locally), and the ms-per-pass rows
measured on `8d5b2f43`. Hand-off snapshots depend on the wiring, which did
not move. `README.md` still gives the old size.

**The full gate** (`GATE_NO_TEST=1 GGUF=<the tip's export> tools/gate.sh`, one heavy ticket of
2 h 03 min, load 4 to 6, started on the tip; the commits after it change `docs/` only): 64
ok, 0 FAIL, 2 skips (the test suite, run in the two tickets above, and Nintendo's DMG boot
ROM, which is not in the repository). Its ok, FAIL and skip lines are the base verify's line
for line, every hash included. Among them, the runs that hold the whole checkpoint's mode
gate at every part's positions: "every stream against the whole checkpoint over real runs:
2038 tokens equal" (`examples/slice_full.rs`) and the step oracle on all six streams on
Mealybug `m3_bgp_change` (3, 1, 3, 1, 37 and 1 passes checked, 3 records through the dense
record oracle).

## Stored runs, stream widths and smaller groups: less of the file, less of each llama.cpp pass

No compiled digest moves; the weights are `8d5b2f43`'s. The layer-shaped GGUF, its
manifest's patch list, a third llama.cpp patch and the harness change.

The layer-shaped one model stored 1,018,129,920 values, 1,848,483 of them nonzero. The
census of its tensors found three blocks that are zero by construction and stored whole, and
two places where a sliced pass computed what no row needs. Five steps, in order, each
proven before the next:

| Step | Change | Stored values | File bytes |
| - | - | -: | -: |
| 0 | base | 1,018,129,920 | 2,041,927,328 |
| 1 | `output.weight` as its run of rows, tokens 61,423 to 65,792 (4,370 of 65,804) | 860,858,880 | 1,727,385,792 |
| 2 | each `ffn_up` as its run of columns: channels 896 to 959, and 0 to 959 in layer 10 | 670,767,616 | 1,347,203,520 |
| 3 | `token_embd` as its first 960 columns and `output` as its first 896 | **558,209,536** | 1,122,087,424 |
| 4 | a stream's pass on its own width of channels | 558,209,536 | 1,122,087,680 |
| 5 | a reduced group without the own masks a pass does not decode | 558,209,536 | 1,122,087,680 |

The file stores 45.2% fewer values. Hidden (2,560), layers (25), heads (240), FFN units
(77,376) and vocabulary (65,804) are unchanged: the weights are the same.

Every run and prefix is derived from the tensors at export (`gguf::HeadRows`,
`gguf::up_cols`, `gguf::col_prefixes`, `gguf::stream_head`, `gguf::stream_width`), named in a
metadata key and read by the loader and by `tools/stock-llama/stored-runs.patch`
(`docs/harness.md` §2 lists the keys). A file without a key loads as before, and the
`8d5b2f43` file still passes `digest --gguf`. `gguf::stored_params` counts the file from the
same derivations, and the export asserts that it equals the summed tensor lengths.

**Why each step is exact.**

1. Every row the file leaves out holds −1024 on channel 0 and nothing else. Channel 0 holds
   the constant at every token and no head or unit writes it (`constants_hold`), so such a
   row scores exactly −1024 at every position. A row of scores over the run alone makes the
   whole vocabulary's choice when its largest score is above −1024, and the harness stops at
   a row where it is not (`cpp::Row::token`); none did. The export also names, per stream,
   the run of tokens it can decode: two tokens whose rows agree on every channel the stream
   can make nonzero differ by a constant, and the lower never wins. Five streams (PPU, APU,
   control, length, stepper) cannot write the family routes, so only their 256 byte tokens
   can win; the GBC CPU keeps all 4,370. A stream's head is a view of rows, and like every
   sliced matmul it keeps every batch row; the output rows are picked from the scores.
2. Each live unit's `up` row is one 1.0 on a mode channel or the constant. `up` reads a
   contiguous copy of the layer's run of the normed input; every other step of the FFN is
   `build_ffn`'s.
3. The token rows are padded with zeros to the model's width where `build_inp_embd` already
   pads for a wider input, and the head reads a contiguous copy of its prefix.
4. The export checks that every row a stream's slice runs is zero past the stream's prefix:
   the kept heads' query rows and output columns, every key and value row, the kept units'
   gate, up and down, and the head's stored rows. The widths are 2,368 (PPU), 2,176 (GBC
   CPU), 2,048 (APU, control), 1,536 (length) and 2,560 (stepper). The norm divides by 2^7
   at that width as at 2,560: `Model::slice` widens the prefix until its bound holds, and
   `tests/one_model.rs` holds the engine's stream networks, which have that width, to the
   whole checkpoint.
5. Past layer 0 a group token attends only the cell its query names, so a mask left out
   reaches the other tokens through layer 0 alone, where every head gives it exactly zero
   weight while one of the tokens the head attends stays. `Stream::reduced` asserts that for
   every gather of every pass. Only the GBC CPU has such masks: its group falls from 144 to
   99 tokens a pass in this window.

Where a matmul reads fewer channels it reads a contiguous copy, never a strided view, and
every stored run starts and ends on a multiple of 64, the K step of Metal's few-row kernels.

**Checks, per step** (measured; each step built at its own commit: export, then llama.cpp
with that commit's patch, then the harness; Yellow from the saved cache of `8d5b2f43`,
1,200 passes: 778 GBC CPU, 298 control, 76 APU, 26 PPU, 22 length; load 3 to 6):

| Check | Steps 1 to 4 | Step 5 |
| - | - | - |
| `digest --gguf`: the file expanded and every value held to the compiler's | `manifest ok`, 0 of 2,962,452,480 differ, at every step | as step 4 |
| the full program with every stream's slice read from the file (`gguf::load_slice`), Yellow 300 in lockstep | 6 streams loaded, 0 differ, `cfd31f3d`, at every step | as step 4 |
| `STOCK_STREAMS=check`: every pass by the slice and by the whole model, the rows held bit for bit on the tokens both score, and to the same choice | 1,178 sliced passes (the length stream's 22 passes of 10 tokens run the whole model both ways), 0 rows differ, at every step | 2,356 (both groups), 0 rows differ |
| `STOCK_REFERENCE`: every decoded pass against the engine's record | 1,200 equal, in the check run and in the default run, at every step | the same |
| the cells 26 write records leave (a hash of each cell's saved bytes, `STOCK_CELLS`) against the same harness on the base's llama.cpp and file | equal, at every step | equal |
| `STOCK_MASKS=check`: the group with every own mask against the one without the undecoded, rows bit for bit | | 1,200 passes, 0 rows differ |
| passes that fell to the 16-token fence | | only the length stream's |

The cells check sees the embedding's padding (step 3). It cannot see `ffn_up`: a cell's keys
and values past layer 0 read no channel an MLP writes (`split-check` fact 3).

`STOCK_GROUP=check` (the full group against the reduced one) over 300 passes flagged 60
rows, all the length stream's, each with a largest difference of 0, and the same 60 with the
masks kept. Its full group of 769 tokens runs the slice, whose head scores 256 tokens, and
its reduced group of 10 tokens runs the whole model, which scores 4,370; the check required
the same run. It now holds the rows on the tokens both score (`Row::agrees`). The other
streams' full groups gave bit-identical rows: with the slices, another part's units no
longer run at its masks, so the f16 overflow §12a describes for a full group did not occur
in this window.

**Pass rate, per step** (measured, one run of 1,200 passes each in the default mode, in
commit order and not interleaved, load 3 to 6; ms per pass as the counts line prints it;
the base row is step 1's harness on the base's llama.cpp build and file):

| | GBC CPU | PPU | APU | Control | Length | Write record |
| - | -: | -: | -: | -: | -: | -: |
| base | 31.2 | 45.5 | 32.2 | 21.8 | 11.9 | 33.3 |
| 1, head rows | 24.8 | 27.2 | 23.9 | 16.0 | 10.5 | 33.7 |
| 2, up columns | 23.7 | 25.4 | 22.2 | 15.2 | 9.7 | 30.9 |
| 3, column prefixes | 22.8 | 24.7 | 21.9 | 15.1 | 9.4 | 30.8 |
| 4, stream widths | 22.1 | 23.5 | 18.4 | 13.1 | 9.5 | 30.8 |
| 5, fewer masks | 18.8 | 23.6 | 18.5 | 13.1 | 9.5 | 30.8 |

The head's run takes the most: the five byte-only streams score 256 rows where they scored
65,804, and the GBC CPU 4,370. The stream widths help the narrow streams (APU and control
at 2,048) more than the GBC CPU at 2,176. Step 5 moves only the GBC CPU.

**Pass rate, A against B** (measured; A is the base's harness and llama.cpp build on the
`8d5b2f43` file as it was, B the tip's on the new file; Yellow from the saved cache, 1,200
passes a run, five rounds with the order alternating (A B, B A, ...) inside one hold of the
GPU lock a round, `nice 19`, load 3.1 to 5.8; every pass of every run held to the engine's
record, 0 error lines in the ten runs; ms per pass as the counts line prints it, which holds
`argmax` on both sides; wall is elapsed time over passes 200 to 1,200, stamped in whole
seconds, so it resolves 1 ms):

| | GBC CPU | PPU | APU | Control | Length | Wall per pass | Write record |
| - | - | - | - | - | - | - | - |
| A | 31.3, 31.2, 31.3, 31.2, 31.3 (**31.3**) | 46.3, 45.5, 45.6, 45.6, 45.7 (45.6) | 32.5, 32.5, 32.3, 32.4, 32.4 (32.4) | 22.0, 21.9, 21.9, 21.8, 21.9 (21.9) | 11.8, 12.0, 11.9, 11.8, 11.9 (11.9) | 29.0 in every round (**29**) | 32.2, 32.2, 32.1, 32.1, 32.1 (32.1) |
| B | 18.7, 18.7, 18.7, 18.8, 18.7 (**18.7**) | 23.6, 23.5, 23.5, 23.5, 23.5 (23.5) | 18.4, 18.4, 18.4, 18.5, 18.4 (18.4) | 13.1 in every round (13.1) | 9.4, 9.5, 9.4, 9.5, 9.5 (9.5) | 18.0 in every round (**18**) | 30.1, 30.0, 30.0, 30.1, 29.9 (30.0) |

A pass is faster on every stream, and the gap is many times the spread between rounds: the
GBC CPU's pass takes 40% less, the PPU's 48%, the APU's 43%, control's 40%, the length's
20%, and the wall time per pass 38%. Of B's 18.7 ms a GBC CPU pass, 18.2 ms are inside
`llama_decode` (A: 28.4 of 31.3; the 2.9 ms outside are mostly rows of 65,804 logits copied
and scanned, inferred). What bounds the 18.2 ms was not profiled.

**Projected to frame 300** (inferred from the medians above with the method and the counts
of the section before: 925,752 GBC CPU, 83,498 PPU, 251,674 APU, 71,759 control and 41,855
length passes; about 49,600 write records at the measured rate; 14,900 reader records at
the killed proof's 319 ms, not measured here; 0.2 h outside the counters, which these runs
resolve no better; the 16-minute prefill):

| | Passes | Write records | Reader records | Outside the counters | Prefill | Total |
| - | - | - | - | - | - | - |
| A | 11.9 h | 0.4 h | 1.3 h | 0.2 h | 0.3 h | **14.2 h** |
| B | 7.0 h | 0.4 h | 1.3 h | 0.2 h | 0.3 h | **9.2 h** |

The window is the first 1,200 passes of boot, taken as representative as before; no full
run was made. The 18,000-pass run below has other rates (a GBC CPU pass 19.1 ms, a PPU pass
23.7, an APU pass 18.7, a control pass 13.2, a length pass 9.7, a write record 25.9 ms) and
projects B at 7.1 h of passes and 0.4 h of write records, the same 9.2 h (inferred).

**The runner** (`tools/runner`, text workload: the hand-off snapshot at frame 2,900, 600
frames with the bench's presses; A is the base's runner on the `8d5b2f43` file, B the
tip's on the new file, each from a snapshot the tip's full program made from that side's
file; three ABBA rounds, load 5.0 to 6.0; every run reached `870aa3fd` with 0 dense
fallbacks): A 62.6, 62.8, 63.0, 62.1, 62.0, 62.7 fps (median 62.7); B 63.7, 64.0, 63.5,
62.8, 63.0, 63.0 (median 63.3). The runner loads the same weights (`gguf::open_model`), so no difference was
expected; B is not slower and stays above 60. The runner's 24
tests pass on the new file.

**Further checks at the tip.** The seven compiled digests are unchanged (`cpu` `bebf5d93`,
`cpu-memory` `836fb6de`, `apu` `d8d3a06e`, `control` `6e225675`, `ppu` `79706b58`,
`one-model` `8d5b2f43`, `ppu-stepper` `4fdab521`). The three committed patches, applied in
`run.sh setup`'s order to b12070's own seven files, give byte for byte the tree that was
built. `STOCK_GROUP=check` over 300 passes: 0 rows differ. The tip's llama.cpp build and
harness on the key-less `8d5b2f43` file, `STOCK_STREAMS=check` over 300 passes: 288 sliced
passes, 0 rows differ, every pass equal to the record. Through frame 1 at the tip: 18,000
passes, every one equal to the record (13,176 GBC CPU, 3,124 control, 1,260 APU, 288 PPU,
152 length), 844 write records at 25.9 ms, one reader record, and frame 1 at `d043ab24`
after 234 s (393 s in the section before); the wall time is 19.4 ms a pass over passes 200
to 18,000, and only the length stream's 152 passes fell to the 16-token fence.

**Not checked.** The stepper has no pass in any window here, so its slice, its width (2,560,
the whole) and its 256-token run have not run in llama.cpp. The PPU's slice was bit-checked
on 26 passes a step. One reader record ran, in the 18,000-pass run; a reader record's rows decode `Line` family tokens through
the whole model's head, which now scores the stored 4,370 rows. No run past frame 1, no
bank switch, and no full proof on the new file: the saved cache state and the proof build
(kept locally) are the base's, and a proof of this work needs the new
export and a llama.cpp built with the three patches.

**On the other lane's layout** (a scratch merge, not committed: the identity lane, hidden
2,176, one model `45bfb056`, with this work's `src/gguf.rs`, `examples/` and
`tools/stock-llama`; measured). The export derives every run again: the embedding's first
576 columns, the head's first 512, `ffn_up` at channels 512 to 575 (0 to 575 in layer 10),
the same head rows and stream runs, and widths of 2,048 (PPU), 1,920 (GBC CPU), 1,664 (APU,
control), 1,152 (length) and 2,176 (stepper). It stores 457,490,560 values; `digest --gguf`
is `manifest ok`, and the full program loads the six slices from the file (0 differ) and
reaches `cfd31f3d`. Nothing ran in llama.cpp on that layout: no saved cache exists for that
digest.

## Round 10: the merged model, checked in the full program and in llama.cpp

Round 10 merges the identity lane (the parts share the output identity channels) and the stored-runs lane (the
file stores runs of the head, the embedding and `ffn_up`; a stream runs on its own width).
Neither lane had built or run the merge. This section is the merge's own record, measured
2026-11-20 on the M1 Max. The timed runs (llama.cpp and the runner) ran at load 2.5 to 6;
the full program's checks ran at load 4 to 19 and report no timing. No compiled digest
moved in this round.

**The file.** `export one-model --gguf-only --layer-shapes --align 64` stores 457,490,560
values (457,379,584 in the weight matrices, 110,976 in the 51 norms) in 921,575,328 bytes,
against 865.4 M on the identity lane and 558.2 M on the stored-runs lane. Hidden is 2,176. The embedding keeps
its first 576 columns (37.9 M values) and the LM head 4,370 of 65,804 rows at 512 columns
(2.2 M). `ffn_up` keeps channels 512 to 575, and 0 to 575 in layer 10. The streams run on
2,048 (PPU), 1,920 (GBC CPU), 1,664 (APU, control), 1,152 (length) and 2,176 (stepper)
channels. The GBC CPU scores all 4,370 stored rows and every other stream 256.
`digest --gguf` prints `manifest ok` with 0 values differing, and the seven compiled digests
are `cpu` `bebf5d93`, `cpu-memory` `836fb6de`, `apu` `d8d3a06e`, `control` `6e225675`, `ppu`
`79706b58`, `one-model` `45bfb056` and `ppu-stepper` `4fdab521`, as the lanes left them.

**What changed on the branch after the merge.**
- `gguf::open_model`, the file-alone path of the runner and `GBCFORMER_GGUF_ONLY`, rebuilds
  the checkpoint's digest from the expanded tensors and refuses a file unless it equals the
  one the file's metadata names. A placement key that names another run of the same size
  used to load. The metadata now carries the digest and the unit names it hashes, which adds
  965,024 bytes to the file. Hashing the one model takes 12.5 s at every load of the runner
  (measured, six loads). A file exported before this change is refused.
- Whether a pass ran its stream's slice is now llama.cpp's own report
  (`llama_get_logits_stream`), and the fence of 16 tokens is one exported constant
  (`llama_gbcformer_fence`). A pass that names a stream and runs the whole model above the
  fence stops the run in every mode.
- `stored-runs.patch` bounds each up run in 64 bits.
- A record's sequence that holds a token no record holds stops the dense pass and the
  harness (`tok::stray`). `Model::slice` asserts the norm's bound where its widening stops.
- `STOCK_SCOPE=check` decoded its second pass on the whole model, so every row of a stream
  that scores 256 tokens counted as differing against rows of 4,370. It now runs the same
  network both ways.
- New tests: the derivations of the stored runs on hand-built checkpoints, the loader's
  refusals, a moved run key in an exported GBC CPU file, and the pinned counts of what reads
  an output identity (4,223 units, 0 head queries, 7 layer-0 key rows).

**In the full program** (heavy lane, one ticket). `tests/one_model.rs` 4 passed and 1
ignored, `tests/stream_audit.rs` 4 passed (it carries `check_f16` and `check_f16_acc` on the
one model), `tests/gguf_file.rs`, `memory_dense`, `dma_records`, `cell_rows` and
`ppu_stepper` passed, the `gguf` unit tests 14 passed, and the runner's 24 tests passed on the
new file. Yellow CGB 300 with every component in weights, in lockstep, as one model, every
stream's slice read from the file with 0 values differing, under the step oracle
(`cpu=100000,apu=25000,control=6000,ppu=8400`), reaches `cfd31f3d` with 0 dense fallbacks
and 10, 11, 12, 10 and 5 passes checked. The full gate and the full `cargo test` did not run
in this round.

**In llama.cpp** (b12070 from a fresh clone, the three patches applied by
`tools/stock-llama/run.sh <dir> <rom> setup`; the seven patched files equal the tree the
patches were written from, byte for byte; Metal, f16 KV, flash attention, one hold of the
GPU a run). The engine's 300-frame record at this tip is byte for byte the record of
`8d5b2f43` (`cmp` of the six files), so the token space did not move. The prefill of Yellow's 143,881 cells took
822 s and its state was saved for this model.

| Run on Yellow from the saved state | Passes | What was compared | Differ |
| - | -: | - | -: |
| `STOCK_STREAMS=check STOCK_WRITES=check` | 6,000 | 5,923 sliced passes against the whole model (3,924 GBC CPU, 1,487 control, 376 APU, 136 PPU); 26 write cells | 0 |
| `STOCK_MASKS=check` | 2,400 | 2,400 groups with and without the undecoded masks | 0 |
| `STOCK_GROUP=check` | 1,200 | 1,200 reduced groups against the full group | 0 |
| `STOCK_SCOPE=check` | 1,200 | 1,197 scoped passes against every cell | 0 |
| `STOCK_WRITES=check STOCK_STREAMS=require` | 18,000 | 844 write cells over the records' scope against every cell | 0 |
| `STOCK_STREAMS=require` | 18,000 | every pass against the record; frame 1 `d043ab24` in 215 s | 0 |

Every pass of every run above also equalled the engine's record (`STOCK_REFERENCE`). Only
the length stream's passes (10 tokens, 5 decoded) fell to the fence, as on the stored-runs lane.

One run carried every switch a pass can combine (`STOCK_STREAMS=check STOCK_GROUP=check
STOCK_WRITES=check STOCK_SCOPE=check`, 300 passes before its hold ended). Its group check
found 0 rows differing and every pass equalled the record. Its slice and scope counts are
not evidence. Under `STOCK_GROUP=check` they also compare the full group on the whole
model, which `docs/stock-llama.md` §12a says is not exact in llama.cpp; that this is why
its slice rows differed is inferred, not isolated. The run also predates the
`STOCK_SCOPE=check` fix. Each check therefore ran in its own hold.

**Two windows the Yellow runs do not reach.** Both ROMs are DMG and their scopes exceed the
4,096-cell bound (`scope::CELLS` refuses them: the PPU's scope names 12,775 and 16,871 cells,
with the GBC CPU's data heads left open, as the stored-runs lane's harness also reports for them), so
both ran with `STOCK_SCOPE=0`, each from its
own prefill (80 s and 101 s).
- **The stepper.** Mealybug `m3_bgp_change`, `--warmup 1 --frames 2`, 8,000 passes under
  `STOCK_STREAMS=check`: 7,855 stepper passes, each on its slice and again on the whole
  model, 0 rows differ, every pass equal to the record. Until this run the stepper's slice
  had no pass in llama.cpp.
- **Reader records and the PPU.** gbmicrotest `dma_basic`, frame 1 whole (7,214 passes)
  under `STOCK_STREAMS=check`: 160 reader records, each equal to the engine's index (the
  assert in `engine/records.rs`), 166 write records, 144 PPU passes and 5,900 GBC CPU
  passes with 0 slice rows differing, every pass equal to the record, and frame 1 at the
  engine's `4df32995`. A reader record decodes one way only, on the whole model over every
  cell, so there is no second decode to hold its rows to. Two of the 144 PPU passes printed
  `MIRROR DIFFERS` (27 and 15 OAM addresses): the cache already held OAM bytes of the DMA
  that the PPU's own image had not applied yet. Both passes still equalled the record. The
  Yellow runs show none.
  This was a defect of the harness, fixed after these runs (`docs/stock-llama.md` §4, "Two
  decisions in the harness"): the engine draws a line a line after its image's cut, and
  the shared cells held the GBC CPU's records kept in between. A pass now lends the
  stream's own lines to those cells. The same frame under `STOCK_STREAMS=require` then
  shows 0 of 144 mirror checks differing, 2 PPU passes with 14 lines lent, and all 7,214
  passes equal to the record (load 7 to 19, beside another verify).

No 2-row decode was found. The length's passes decode 5 positions of 10 tokens (all 41,855
in the record), a write record's batch is 16 tokens with one output row, a reader record's
is 23 tokens with eight, and the null cell is one token. The GBC CPU's passes decode 6, 67
or 74 positions (the record's first 20,000 events).

**The 14 slots on the record identities.** With shared identities a record gather's targets
include each stream's own slots on output indices 40 to 53, so the harness keeps those
masks in every group. Measured over 1,200 passes, tokens per pass are 166 (PPU), 139 (APU),
150 (control), 99 (GBC CPU) and 10 (length), the same as on the stored-runs lane, because each of
those streams decodes every output slot in every pass. Nothing was dropped and the cost is
0 tokens.

**Pass rate** (three runs of 1,200 passes, `STOCK_STREAMS=require`, load 3.7 to 4.5; ms per
pass, median and range):

| | GBC CPU | PPU | APU | Control | Length | Wall |
| - | -: | -: | -: | -: | -: | -: |
| the stored-runs lane on `8d5b2f43` (its section) | 18.7 | 23.5 | 18.4 | 13.1 | 9.5 | 18 |
| round 10 on `45bfb056` | 17.4 (17.4 to 17.4) | 25.4 (25.4 to 25.5) | 18.1 (18.1 to 18.2) | 11.1 (11.1 to 11.2) | 9.2 (9.2 to 9.3) | 16.0 |

The two rows were not interleaved, so the differences are indicative. The PPU's 8% rise
rests on 26 passes a run and has no profiled cause. Over the 18,000-pass run the rates are
17.7, 25.5, 18.3, 11.2 and 9.4 ms, a write record takes 25.5 ms (844) and the one reader
record 278 ms.

**Projected to frame 300: 8.2 hours** (inferred). The record gives 925,752 GBC CPU, 83,498
PPU, 251,674 APU, 71,759 control and 41,855 length passes and 69,514 write records. At the
18,000-pass run's rates the passes take 24,318 s and the write records 1,773 s. About
12,000 reader records (11,967 DMA lines in the engine's run) at 278 ms add 3,336 s, and
about 870 GBC CPU passes that run unscoped after a banking write add about 91 s at 122 ms
(the scope check's cost of a pass over every cell). That is 29,518 s. The reader rate is
one sample, and every rate is from frame 1; the later frames' rates are not measured on
this model.

**The proof's script path.** `tools/stock-llama/proof.sh`, started as the proof will be
(the kit's harness, file, record and saved state) in a work directory of its own, ran to
2,000 passes and had its child stopped. Every pass equalled the record, the script logged
the exit, released the GPU lock and wrote `proof.exit`, and `proof-status.sh` listed the
six streams against the record's counts. The 300-frame proof was not started.

**The runner** (text workload, 600 frames, three ABBA rounds, load 3.9 to 5.7, every run
`870aa3fd` with 0 dense fallbacks and the same pass counts). The base's runner on its file:
62.3, 62.4, 62.5, 62.6, 62.6, 63.2, median 62.55 fps. The tip's on the merged file: 63.6,
63.8, 64.2, 64.2, 64.4, 64.7, median 64.2 fps. Not slower and above 60. The 2.6% has no
profiled cause, so no speedup is claimed.

**Not checked.** No frame past the first on Yellow, no bank switch and no full proof. The
stepper's and the reader records' windows ran unscoped. A reader record has no bit check.
`tok::stray` has no test of its own. `tools/gate.sh` and the full `cargo test` are the
verify's.

The full proof has run since, in unpatched llama.cpp. The next section records it.

## The first proof: Yellow 300 in unpatched llama.cpp (2026-11-23)

This run was "the proof of record" until 2026-11-25, and sections written before then call
it that. The proof of record is now the run on `ff86dde7` in llama.cpp's default build
("The proof of record on `ff86dde7`", at the end of this file). This section is unchanged
below this note.

The run that Round 10 left unstarted has finished. Unpatched llama.cpp decoded every pass
of Pokémon Yellow to frame 300 on the one model, and every pass equalled the
engine's record. The run starts at the hand-off, frame 0 with no boot ROM (the command has no
`--boot`), which is what "from boot" means in `proof.sh` and in these docs. The run's folder is local and
untracked. Every number below is from its `work/proof.log`, `final-status.txt`, `STATUS`
and `driver.log`, measured unless a line says otherwise. No compiled digest moved, and no
code changed for this section.

**What ran.**
- The model is the one model, digest `45bfb056`, as the uniform file: sha1 `ce42b0b1…`,
  5,042,980,288 bytes (hashed again after the run).
- The runtime is llama.cpp b12070, commit `00f113ec…`, with no patch. `git status --short`
  in its tree printed 0 lines before and after the run. The harness printed `llama.cpp
  unpatched; whole model every pass` in each of the 13 chunks. The settings were Metal,
  `STOCK_PURE=1`, `STOCK_STREAMS=0` and `STOCK_GROUP=own`.
- The harness is `tools/stock-llama` built from the source of that day, with its library search path
  changed to the run's folder and the binary signed again ad hoc, and `proof.sh` and
  `proof-status.sh` are the ones of that day (both recorded in `STATUS`, not rebuilt
  here). The host ran with every component in weights and in lockstep (`in weights: cpu,
  memory, ppu, apu, control; on the host: none`).
- The run prefilled its own cache in unpatched llama.cpp: 143,881 line cells, 2,523 one by
  one and 141,326 in lanes, in 4,961.6 s (83 minutes). The saved cache has sha1
  `a8b3f911…` (hashed again after the run). `STATUS` records that the patched build's
  saved cache has the same sha1 and that `state-diff` found 143,848 cells equal and 0
  differing. That comparison was not repeated for this section.

**What was compared.**
- Frame 300 hashes to `cfd31f3d89f111fe6fd2fd8f36a1fe4971aae642`, the hash of the gate's
  one-model Yellow 300.
- 1,374,538 of 1,374,538 decoded passes equalled the engine's record as they decoded
  (`STOCK_REFERENCE`): 925,752 GBC CPU, 83,498 PPU, 251,674 APU, 71,759 control and
  41,855 length. The stepper has no pass in this run.
- After `play` exited, `proof.sh` held the run's own record to the reference, event for
  event: 995,266 GBC CPU events (the 925,752 passes and the 69,514 write records the
  engine keeps), 83,498, 251,674, 71,759 and 41,855, all equal.
- The log holds 0 lines with `DIFFERS`, `REFUSED`, `error` or `panicked`.

**Chunks.** The run took 13 chunks and 12 resumes. Eleven chunks ended at their 21,600 s.
Chunk 4 ended on a stop request at 16:12:23 on 2026-11-21, a planned pause, and
chunk 5 took the GPU lock at 17:18:28, 66 minutes later. Chunk 13 ran to the end.
`proof-status.sh` reports that each resume began at a checkpoint a live chunk saved and
that the live events run from event 0 to the log's end with no gap. The 13 chunks held
the GPU lock for 71 h 34 min. From the start (2026-11-20 19:26:09) to `proof.sh`'s exit
(2026-11-23 20:23:54) is 72 h 58 min.

**Rates and time.**

| | Count | ms each | Hours (count × rate, computed) | Projected (`docs/stock-llama.md` §13) |
| - | -: | -: | -: | -: |
| GBC CPU pass | 925,752 | 163.7 | 42.1 | 154.4 ms, 39.7 h |
| PPU pass | 83,498 | 164.2 | 3.8 | 163.2 ms, 3.8 h |
| APU pass | 251,674 | 125.1 | 8.7 | 124.0 ms, 8.7 h |
| Control pass | 71,759 | 129.9 | 2.6 | 128.3 ms, 2.6 h |
| Length pass | 41,855 | 27.9 | 0.3 | 28.2 ms, 0.3 h |
| Write record | 119,435 | 118.0 | 3.9 | 116.2 ms, 3.9 h |
| Reader record | 46,951 | 587.5 | 7.7 | 600 ms (inferred), 7.8 h |
| Prefill | 1 | | 1.4 | 0.9 to 1.3 h |
| Resume, to the first live event | 12 | 118 to 299 s | 0.7 | 2 to 5 minutes each |
| Checkpoint | 856 | | 0.2 | about 0.3% |

The write and reader rates are the cache's timers over their counts (14,097.6 s and
27,585.8 s). The rows sum to 71.4 hours against the 71.6 hours with the lock. The sum is
computed, and the rows are not shown to be disjoint. The projection was about 68 hours
before the resumes. Most of the difference is the GBC CPU pass, 163.7 ms against the
154.4 ms that frame 1 gave, and its cause is not traced. Inside llama.cpp's decode a GBC
CPU pass took 160.5 ms.

The 856 checkpoints took 759.6 s of 257,502 s (0.29%). Sixteen of them are the
prefill's, which took 8.3 s at 71 MB and rose to 10.4 s at 946 MB, 146.6 s in all (3.0%
of the prefill). The other 840 took 613.0 s.

**What this run does not show.**
- The cache decoded 119,435 write records and 46,951 reader records, the counts the
  model-free replay predicted (`docs/stock-llama.md` §13). Only the 69,514 kept write
  records are events in the record comparison. The other write records and the reader
  records are held by their counts. A later pass that reads a wrong cell would differ from
  the record, which is inferred and not a check of each cell. The log shows
  `write_checks: 0`, so no write record's cells had a second decode in this run.
- Yellow lends no cells in 300 frames. Every loan and swap counter is 0, so the run does
  not exercise the swap path. The patched build's `dma_basic` and Zelda DX runs cover it
  (`docs/stock-llama.md` §4 and §14).
- Each pass was decoded and held to the record once, in the chunk that ran it live. A
  resume runs the host again from frame 0 with every earlier pass answered from the
  engine's record and no model (`docs/stock-llama.md` §14). Chunk 13 went live after event
  1,491,513, in frame 278. So the process that printed frame 300's hash decoded the rest of frame
  278 and frames 279 to 300 itself, and replayed everything before that event. The total of 1,374,538 equal passes
  carries the earlier chunks' counts through the checkpoints.
- The replay writes the run's record files again, and each checkpoint keeps a digest of
  them that the resume must match. So the comparison of the whole record after the run
  reads lines the last process wrote. For events before 1,491,513 those lines come from
  the replay, and the digests tie them to the lines the earlier chunks wrote.
- The lockstep counters at the end of the log are the last process's, so they include
  replayed passes.
- A reader record decodes one way only. It has no second decode to compare its rows with,
  as in Round 10.
- No patched full run of `45bfb056` exists. The patched 300 frames stay projected at 11.6
  hours (`docs/stock-llama.md` §13).
- A smaller re-levelled model exists on a branch and is not part of this proof. In
  unpatched llama.cpp it has run frame 1 only (13,167 passes equal, frame 1 `d043ab24`,
  100.5 ms a GBC CPU pass; reported by that branch, not checked here).

## The harness's scopes narrowed: the unpatched pass without a model change (2026-11-24)

The proof of record took 71.6 hours, and a pass in unpatched llama.cpp runs the whole
uniform model whatever the harness does. This round changes only `tools/stock-llama`. No
weight moves, every digest is the same, and every decoded pass is still held to the
engine's record. Over the first 1,500 passes of Yellow a GBC CPU pass falls from 87.5 to
40.4 ms on the re-levelled model `1b581cfa` and from 156.8 to 86.9 ms on `45bfb056`,
the model of the proof. The 300 frames project at about 19 hours for `1b581cfa`
(from about 38 with the old harness at today's rates) and at about 40 hours for
`45bfb056` (71.6 measured). `docs/stock-llama.md` §13, "The scopes narrowed", states
what the harness now does and what holds it.

**Where the time was (measured before any change).** The kit's harness, `1b581cfa`,
Yellow from the hand-off over its saved cache of 143,848 cells, unpatched b12070 on
Metal, load average 12 to 14, one run each:

| Stream | Scoped cells | ms per pass, scoped (1,500 passes) | ms per pass over every cell (`STOCK_SCOPE=0`, 100 passes) | ms per cell |
|---|---:|---:|---:|---:|
| GBC CPU | 7,655 | 88.1 | 950.9 | 0.00633 |
| APU | 2,076 | 59.6 | 898.9 | 0.00592 |
| Control | 2,076 | 62.9 | 1,076.7 | 0.00715 |
| Length | 2,076 | 16.9 | 253.6 | 0.00167 |

The cost is linear in the cells a pass attends, as an earlier fit said. The fit
gave 0.0077 ms per cell for the GBC CPU's 99 tokens, and the measured slope is 18% lower.
At that slope 48 of the GBC CPU's 88 ms were attention over its 7,655 cells, and a GBC
CPU pass over no cell would take 39.6 ms (inferred from the two points). The narrow
scope's pass measures 40.4 ms.

The fit's other term, 13 ms that follow the cache's length at the same scope, did not
reproduce. It came from 100.5 ms a GBC CPU pass in the kit's frame-1 run of 2026-11-21
against 87.9 on a 12,809-cell cache. Today the same harness takes 86.9 to 88.1 ms over
Yellow's cache at the same 7,655 scoped cells (three runs of 1,500 passes), so the
earlier run was slower throughout for a reason this round did not find.

A profile of the same run (`sample`, 10 s, a run that is not reported) put 5,800 of the
main thread's 6,600 samples (88%) in Metal's `waitUntilCompleted`, so the GPU bounds the
pass. Of the rest, 389 samples (5.9%, about 5 ms a pass) are in the harness's `argmax`,
163 (2.5%) in llama.cpp's `find_slot`, 77 in the rows' copy and 56 in `seq_rm` and
`seq_cp`.

**The changes** (each is in `docs/stock-llama.md` §13).

1. A stream's scope holds its own heads' lines. The APU's and control's scopes fall
   from 2,076 cells to one, the length's to 22, and the PPU's stays near 2,086.
2. A GBC CPU pass that decodes the addresses of its reads and writes attends a narrow
   scope: 472 fixed cells, the lines its first heads read (the bytes at PC, PC + 1,
   PC + 2 and SP + 1), and the lines the pass names, with the 256 named most recently
   kept. A pass that names a line outside it decodes again.
3. A pass after a banking write decodes first for its `MAP` slots over the fixed lines.
   It attended every cell before.
4. A reader record decodes over the records' scope and its two lines.
5. A write record's batch sits in its scope's sequence, so llama.cpp gathers the scope's
   cells for it. Before, its tokens were also in sequence 0, and llama.cpp gathered
   every cell and masked all but the scope's.
6. `argmax` scanned each row entry by entry for the place of its largest score, which
   sits near the row's end. It now skips the blocks that do not hold it.

**Measured** (unpatched b12070 on Metal, `STOCK_PURE=1`, Yellow from the hand-off, the
same saved cache and the same passes on both sides, every pass held to the engine's
record, load average 3 to 16). "Before" is the kit's harness (the re-levelling's for `1b581cfa`,
the proof's for `45bfb056`), "after" this work's.

The first 1,500 passes, median of 3 runs a side, ms (range in brackets). `45bfb056`'s runs
alternate (before, after, three times over 24 minutes). `1b581cfa`'s do not: they ran in the
order before, before, after, before, after, after over 31 minutes, so a drift in the machine's
load between the first two and the last two is not ruled out for that pair:

| | Tokens | `1b581cfa` before | `1b581cfa` after | `45bfb056` before | `45bfb056` after |
|---|---:|---:|---:|---:|---:|
| GBC CPU pass (973) | 99 | 87.5 (86.9 to 88.1) | 40.4 (39.5 to 40.5) | 156.8 (156.7 to 157.2) | 86.9 (86.4 to 87.0) |
| PPU pass (34) | 166 | 82.3 (80.9 to 83.4) | 79.3 (78.2 to 80.2) | 164.4 (163.7 to 166.7) | 162.0 (161.1 to 162.1) |
| APU pass (94) | 139 | 59.1 (58.4 to 59.7) | 44.9 (44.3 to 45.2) | 124.9 (124.9 to 125.7) | 103.0 (102.7 to 103.1) |
| Control pass (373) | 150 | 61.9 (61.4 to 62.7) | 46.0 (44.9 to 46.0) | 129.6 (129.3 to 129.9) | 104.1 (103.9 to 104.2) |
| Length pass (26) | 10 | 16.7 (16.7 to 17.0) | 12.8 (12.1 to 13.0) | 29.0 (29.0 to 29.6) | 22.1 (22.1 to 22.4) |
| Write record (26) | 16 | 77.6 (76.0 to 77.7) | 34.8 (34.7 to 35.6) | 119.8 (118.9 to 121.2) | 52.4 (51.8 to 53.0) |
| All 1,500 passes, s | | 112.5 (112.0 to 112.9) | 61.5 (60.4 to 61.5) | 214.7 (214.5 to 215.0) | 137.1 (136.7 to 137.2) |

These "after" runs used the harness before the branch's last code commit,
which does not yet put the line at SP + 1 in before the pass. One `1b581cfa` run at the tip gives 39.5,
78.0, 44.1, 44.9, 12.5 and 33.8 ms, inside the ranges, and frame 1 below is at the tip
for both models.

A run with `STOCK_CPU_SCOPE=windows` on `1b581cfa` (one run) separates the steps: the
GBC CPU pass stays at 86.4 ms while the APU's falls to 45.0 and the write record to
35.7, so the narrow scope is what moves the GBC CPU pass. On both models the 26 cells
the write records left have the saved bytes the kit's harness left (`STOCK_CELLS`, the
logs equal).

`45bfb056` gains less because its fixed cost is higher. Its uniform file runs every
layer at FFN width 10,688, so its APU pass over one cell still takes 103 ms where
`1b581cfa`'s takes 45.

Reader records, 40 of evenly spaced lines read whole after the prefill
(`STOCK_READ_PROBE=40`, one run a side), ms each:

| | Over every cell (`STOCK_READS=0`) | Over the records' scope and their line |
|---|---:|---:|
| `1b581cfa` | 309.2 | 22.1 |
| `45bfb056` | 578.8 | 43.0 |

The proof of record measured 587.5 ms over its 46,951 reader records.

All of frame 1 (13,167 passes) through `proof.sh`, one run a side, ms, the "after" runs
at the tip. The `1b581cfa` "before" is the kit's run of 2026-11-21, which was
slower than today's runs throughout (above). The `45bfb056` "before" is the frame-1 run
of `docs/stock-llama.md` §13. An earlier `1b581cfa` run of this work, before a
later fix, gave 38.5, 77.4, 43.0, 43.9 and 11.4 ms.

| | Count in frame 1 | `1b581cfa` before | `1b581cfa` after | `45bfb056` before | `45bfb056` after |
|---|---:|---:|---:|---:|---:|
| GBC CPU pass | 8,768 | 100.5 | 39.6 | 154.4 | 86.5 |
| PPU pass | 288 | 95.6 | 77.9 | 163.2 | 161.5 |
| APU pass | 840 | 69.8 | 44.2 | 124.0 | 102.7 |
| Control pass | 3,119 | 73.4 | 45.0 | 128.3 | 103.7 |
| Length pass | 152 | 18.9 | 12.5 | 28.2 | 22.3 |
| Write record | 109 | 82.7 | 28.6 | 116.2 | 47.4 |
| Reader record | 1 | 354.9 | 23.4 | 583 | 47.0 |

In each "after" run 11 GBC CPU passes decoded twice (a named line outside the scope) and
one decoded first for the windows. Their time is inside the GBC CPU's rate. `45bfb056`'s
frame 1 took 1,239.9 s in one chunk. The frame-1 run of §14 held the GPU for 38 min 37 s
in four chunks.

**The 300 frames, projected** (inferred: the rates of the first 1,500 passes above times
the counts of the proof of record, 925,752 GBC CPU, 83,498 PPU, 251,674 APU, 71,759
control and 41,855 length passes, 119,435 write records and 46,951 reader records;
hours). The last column is what the proof of record measured.

| | `1b581cfa` before | `1b581cfa` after | `45bfb056` before | `45bfb056` after | `45bfb056`, the run of record |
|---|---:|---:|---:|---:|---:|
| GBC CPU passes | 22.5 | 10.4 | 40.3 | 22.3 | 42.1 |
| PPU passes | 1.9 | 1.8 | 3.8 | 3.8 | 3.8 |
| APU passes | 4.1 | 3.1 | 8.7 | 7.2 | 8.7 |
| Control passes | 1.2 | 0.9 | 2.6 | 2.1 | 2.6 |
| Length passes | 0.2 | 0.15 | 0.34 | 0.26 | 0.3 |
| Write records | 2.6 | 1.15 | 4.0 | 1.7 | 3.9 |
| Reader records | 4.0 | 0.3 | 7.5 | 0.56 | 7.7 |
| GBC CPU passes after a banking write, and second decodes | 0.7 | 0.06 | 1.0 | 0.14 | in the first row |
| The shift after a write record | 0.2 | 0.2 | 0.3 | 0.3 | in the rows above |
| Prefill | 0.7 | 0.7 | 1.4 | 1.4 | 1.4 |
| Resumes and checkpoints | | | | | 0.9 |
| **Sum** | **38.2** | **18.9** | **70.0** | **39.7** | **71.6** |

How the rows that 1,500 passes do not time are counted:

- Reader records use the probe's rates. Yellow's start at frame 12.
- After a banking write the old harness decoded the next GBC CPU pass over every cell:
  2,896 passes in the 300 frames, each 950.9 ms on `1b581cfa` (56 passes, above) and
  1,446.6 ms on `45bfb056` (30 passes with the kit's harness and `STOCK_SCOPE=0`), counted
  here as the time beyond a scoped pass. The new harness decodes such a pass twice over
  the narrow scope, and 2,820 other passes twice for a named line (the replay's counts).
- The pass that follows a write record carries llama.cpp's shift of the new cell to its
  line, one pass over the whole KV buffer. In `1b581cfa`'s frame 1 the 86 such passes
  took 45.8 ms each against 39.6 for a GBC CPU pass, about 6 ms a write record
  (inferred from the two averages; the 86 are not all GBC CPU passes). In `45bfb056`'s
  frame 1 they took 95.3 ms against 86.5, about 9 ms.

The projection from frame 1's rates instead gives 18.3 hours for `1b581cfa` and 39.4
for `45bfb056`. The run of record ran 6% over its own frame-1 projection, and the
"before" column here is within 2% of what it measured less its resumes and checkpoints.

The proof of record's GBC CPU pass took 163.7 ms over the run against 154.4 in frame 1,
and its cause was not traced. The passes after a banking write account for 4.0 ms of the
9.3 (2,896 passes at 1,446.6 ms over 925,752 passes) and the shifts for about 1.1 (119,435
at about 9 ms). Both are inferred from this round's rates, and the other 4 ms are still
not traced.

**Checks** (all measured; "equal" means equal to the engine's record as decoded).

| Check | Result |
|---|---|
| `play replay` (no model) over the 300 frames of each model's engine record (this row's "each scope holding every named line" counts the APU and control, whose records name none, and predates the PPU's per-position lines in the record: "Round 12"): the scopes, the joins and the second decodes in the run's order, every scoped pass's scope held to the lines the record names for its heads | frame 300 `cfd31f3d`; 925,752 GBC CPU, 83,498 PPU, 251,673 APU, 71,758 control and 41,855 length passes scoped, each scope holding every named line (the first APU and control passes run before the cache exists); 913,778 GBC CPU passes over the narrow scope and 11,974 (the OAM DMA placements) over the windows'; 2,896 decoded first for the windows; 0 unscoped; 46,951 reader records, 0 off the mirror. At the tip, on both records: 3,664 lines joined, 2,820 passes decoded twice. Before SP + 1 went in first: 3,659 and 3,067 |
| `1b581cfa` in the runtime: the first 1,500 passes four times, 9,000 passes once | every pass equal, 0 error lines. Two earlier runs stopped at a panic: the runtime allowed six scoped sequences and the harness now makes seven. The limit is eight |
| `1b581cfa`, `STOCK_SCOPE=check STOCK_WRITES=check STOCK_READS=check`, the first 400 passes, twice | each scoped pass also over every cell: GBC CPU 254, control 98, APU 25, length 13, PPU 8, 0 rows differ; 26 write records both ways, 0 cells differ; 12 probed reader records both ways, 0 rows differ |
| `1b581cfa`, the same check over the end of frame 1 (`STOCK_CHECK_FROM=12250`), where passes name lines beyond the fixed ones | 13,100 passes, all equal; from pass 12,250 each scoped pass also over every cell: GBC CPU 686, control 88, APU 65, PPU 8, length 3, 0 rows differ; the run's 10 passes that decoded twice are among them; 99 write records both ways, 0 cells differ; the frame's reader record both ways, 0 rows differ |
| `45bfb056`, the same check, the first 300 passes | GBC CPU 187, control 74, APU 19, length 12, PPU 6, 0 rows differ; 26 write records, 0 cells differ; 12 probed reader records, 0 rows differ |
| `1b581cfa`, frame 1 through `proof.sh` with `PROOF_PURE=1`, play killed (KILL) after a checkpoint, then continued: once early (after event 1,421) and once late at the tip (after event 12,921, 12,858 passes, when the scope holds named lines and passes have decoded twice) | each time the second chunk replayed to the checkpoint's event, held what the checkpoint held and went live; exit 0; frame 1 `d043ab24`, the engine's hash; all 13,167 passes equal; the run's record equals the reference's, stream by stream |
| `45bfb056`, frame 1 through `proof.sh` with `PROOF_PURE=1`, at the tip, in one chunk (the kill came after the run had ended) | exit 0; frame 1 `d043ab24`; all 13,167 passes equal; the run's record equals the reference's; 21 checkpoints |
| The fault in the replay, `STOCK_FAULT_SCOPE=5000` | GBC CPU pass 8,270 decodes again without line `0x104000`, which it names; the run stops at `SCOPE MISSES LINE`, exit 1 |
| The fault in the runtime, `1b581cfa`, `STOCK_FAULT_SCOPE=8000` | GBC CPU pass 8,270 decodes again without line `0x104000`: `GROUP ROW DIFFERS` at row 39 against every cell (token 61542 for 61423), then `REFERENCE DIFFERS` at the same token, and the run stops. The 8,270 GBC CPU passes before it are equal |
| gbmicrotest `dma_basic`, 400 passes, unpatched, `45bfb056`, at the tip, with the scopes on (`scope::CELLS` refused them for this ROM before, `docs/stock-llama.md` §12a), `STOCK_READS=check STOCK_WRITES=check` | every pass equal (307 GBC CPU, 38 APU, 42 control, 7 length, 6 PPU), every scope holding every named line; 160 reader records both ways, 0 rows differ, 0 off the mirror; 194 write records both ways, 0 cells differ. A reader record takes 43.8 ms over its scope and 95.9 ms over all 16,905 cells (160 each, one run a side) |
| Zelda DX frame 7 (`--warmup 6`), which switches banks and lends lines, at the tip: in the patched build on its slices with `STOCK_WRITES=check STOCK_READS=check` and its own prefill, then in unpatched llama.cpp over `45bfb056`'s uniform file and the cache that prefill saved | both: exit 0, frame 7 `d043ab24`, all 21,110 passes equal (19,491 GBC CPU, 238 PPU, 845 APU, 416 control, 120 length), every scope holding every named line; 19,450 GBC CPU passes over the narrow scope, 626 decoded twice, 4 decoded first for the windows (the first pass and three after a banking write); 216 PPU passes lent 438 lines, 424 cells put back compared, 0 differ; 160 reader records, 0 off the mirror. Patched: 3,126 write records (876 of them loans) and the 160 reader records both ways, 0 cells and 0 rows differ; 13.2 ms a GBC CPU pass. Unpatched: 94.4 ms a GBC CPU pass, 44.8 a write record, 49.7 a reader record, 2,172 s for the frame |
| `scope::named_slots_wiring`, at every start | the `MAP` slots depend on the two `REGS` heads alone, and the four address slots besides on 33 first heads (`OP0c`, `OP1c`, `OP2c`, `D1h`, `OBH`) |
| Offline, over `1b581cfa`'s record of the 300 frames (`heads`) | in all 913,778 passes that decode the address slots, every line a GBC CPU head names is a fixed one, a line of the bytes at PC, or a line `RP0`, `RP1`, `WP0` or `WP1` names, and `RP1`'s line is the line of SP + 1; 149 of the 11,974 OAM DMA placements name other lines, which is why they keep the windows' scope |
| `scope`'s test `every_address_lands_in_its_bus_lines_or_a_small_region` | 200 random register sets of every mapper, every bus address, read, written and fetched, OAM DMA off, on every bus and on each alone, each access lock, the boot ROM mapped or not. Its first form failed on MBC2's 512-byte RAM, which repeats in A000-BFFF, so an address there keeps the windows' scope |

**What bounds a pass now.** A profile of the new harness on `45bfb056` (`sample`, 10 s,
a run that is not reported) puts 5,188 of the main thread's 5,928 samples (87.5%) in
`waitUntilCompleted`. The GPU runs the whole model for every token whatever the scope,
and the APU's 139 tokens take 103 ms over one cell. `find_slot` has 198 samples (3.3%),
`argmax` 186 (3.1%, down from 5.9%) and the rows' copy 88.

`find_slot` walks the KV buffer from its head for free slots. A write record's eviction
leaves a hole and moves the head down to it, and each later decode walks from the hole
to the free slots at the top. Listing the cell token first, so that the new cell takes
the hole, does not work, because llama.cpp refuses a batch whose positions decrease in a
sequence. This round leaves the walk. Moving the new cell into the hole through the
state API would remove most walks (not built, and not measured).

**Not done, not verified.**

- No frame of Yellow past the first has run in the runtime with these scopes. The 300
  frames ran in the replay, which runs no model. Zelda DX's frame 7 gives the narrow
  scope three passes after a banking write, 626 second decodes, loans and reader records
  in unpatched llama.cpp. Yellow's own bank switches and its OAM DMA from frame 12 have
  not decoded in the runtime with this harness.
- That every GBC CPU head of a pass reads a fixed line, a first head's line or a line
  one of the four address slots names is not derived from the circuit. The engine's
  record holds it for every pass under `STOCK_REFERENCE`, and it holds over the 913,778
  passes of Yellow's record. A run without `STOCK_REFERENCE` has the rule alone.
- Where each first head reads (`scope::FIRST_HEADS`) is written from `sm83::decode` by
  hand. The wiring assert checks which heads are first heads, not that `OP1c` reads at
  PC + 1. The checks above hold it on the runs they cover.
- The projection takes the rates of 1,500 passes for 300 frames.
- The kill late in a frame ran on `1b581cfa` only.
- `find_slot`'s walk is not removed. About 4 ms of the run of record's 9.3 ms are not
  traced, nor why the kit's run of 2026-11-21 was slower than today's.
- The PPU's scope is not narrowed. Its heads' lines follow the tile map's bytes, and no
  PPU slot names them.
- In the patched build only Zelda DX's frame 7 ran with this harness. The patched 300
  frames of `docs/stock-llama.md` §13 are not projected again.
- The stepper's stream has no pass in any of these runs or in the replay, so it has
  never decoded over its own heads' scope.
- No run had the boot ROM mapped under the narrow scope. `dma_basic` names window 0 at
  address 0. Only the `bus_lines` test covers that mapping.
- `stock-llama scope <rom>` still prints every head's lines for each stream, the sets
  `play` used before this work.
- The scopes went in as two commits after the `argmax` one, not one commit a change.

## Re-levelling: the PPU's parts fill the layers the others leave

One change to the one model's circuit. It moves only the one model's digest, `45bfb056` to
`1b581cfa`; cpu `bebf5d93`, cpu-memory `836fb6de`, apu `d8d3a06e`, control `6e225675`, ppu
`79706b58` and ppu-stepper `4fdab521` are unchanged (`examples/digest`). The reason is the
claim of record, the uniform file in unpatched llama.cpp b12070 (`docs/stock-llama.md`
§13): there every layer stores and computes the widest layer's MLP and the widest layer's
heads, so a pass costs the layer count times the widest layer. Measured 2026-11-21 on the
M1 Max, beside the 300-frame proof of `45bfb056` on the GPU (load 4 to 26): no timing here
is quiet, and none used the GPU.

**What sets the widest layer.** A layer's MLP units are the sum over the parts placed
there, and so are its heads; only hidden is the maximum over the parts (`docs/one-model.md`
§3, "Why units do not overlay"). Each part kept the levels its own build gave it, so all
six started at layer 0 and the short ones piled up: layer 6 held 10,634 units (the stepper
5,280, the scanline PPU 2,861, the GBC CPU 1,288, the APU 679, control 496, the length 30)
and layer 1 held 69 heads (the stepper 25, the PPU 21, the length 21, the GBC CPU 2), while
layers 17 to 24 held the APU's 557 to 684 units alone and no head.

**The room** (`cargo run --release --example room`, from the circuits alone, 20 s). The
model holds 78,574 units by `Model::census` (nodes, clearing units, routing units, reader
units), 185 heads past layer 0 and 41 gathers at layer 0.

| Part | Nodes | Node units | Heads | Depth (its critical path) | Node units with no slack in its own depth |
|---|---:|---:|---:|---:|---:|
| scanline PPU | 2,347 | 12,040 | 61 | 13 | 9,264 |
| GBC CPU with memory | 1,752 | 8,313 | 53 | 10 | 3,168 |
| APU | 3,016 | 14,675 | 0 | 24 | 462 |
| control | 1,900 | 6,815 | 0 | 16 | 613 |
| PPU length | 1,237 | 4,497 | 31 | 14 | 2,535 |
| PPU stepper | 2,088 | 13,887 | 38 | 8 | 8,606 |

Every part's depth is its critical path, so no part can be shallower, and the APU's 24
levels set the 25 layers. Lower bounds on the widest layer, whatever the placement: the
units confined to a window of layers (a node's earliest level to its latest) need that
window's width, and every unit is confined to the whole model.

| Bound | Layers | Width at least | Heads at least | A token's multiplies against today's |
|---|---:|---:|---:|---:|
| (a) nodes moved within their slack, every head and every part's depth where it is | 25 | 8,941 | 69 | 87% (at width 8,960) |
| (b) every node and head free | 25 | 3,143 | 41 | 36.5% |
| | 26 | 3,023 | 41 | 37.0% |
| | 28 | 2,807 | 41 | 37.7% |
| | 32 | 2,456 | 41 | 40.1% |
| | 40 | 1,965 | 41 | 44.2% |

A token's multiplies in one layer are `hidden · (3 · width + 128 · heads + 128)`, counted at
today's hidden 2,176. Moving nodes inside the parts' own depths gains 13%. Stretching the
short parts over the layers the APU already needs gains almost everything, and a 26th
layer loses: the units' total is fixed, so layers times width cannot fall below it, and
each layer adds its heads. The heads' floor is layer 0's 41 input gathers (38 groups of
input slots and the 3 record gathers), which no placement moves.

**The change.** `src/level.rs` holds a circuit's order (`Graph`: a node above the wires it
reads, a head at or above its query and a layer below its readers, the heads of one bus
together) and packs the free parts' nodes and heads into the room the pinned parts leave,
from the last layer down (`Graph::pack`). A level takes the nodes at their earliest level
whatever the room, then nodes within three levels of their earliest, then the node that
adds the fewest live wires, so hidden does not grow. `one_model::circuit` packs the
scanline PPU to 20 levels, its length to 20 and the stepper to 24 under 3,520 units and 39
heads a layer, with 1,313 live wires a part where it can (`one_model::PLACEMENT`); the GBC CPU, the APU and control keep their own
levels, so their streams are as they were. `Model::shape` now counts a shallower part's
routing units before it places the clearing units, which used to pile onto those layers.
`examples/room -- --scan` lists the placements by cost.

Two limits of the engine shaped the placement. The positional stage held 15 circuit layers
of the PPU's stream in 32-bit reader masks; they are 64 bits now (`engine::Readers`). A
stream of more than 20 circuit layers runs without the template's reader buckets
(`Template::single_bucket`), as the APU's always has, so the PPU and its length stop at 20
and only the stepper goes past (Yellow's first 300 frames never run it, and the text
bench steps 5 lines).

**Old against new** (`examples/shape --one`, `examples/export`, `examples/digest --gguf`).

| | `45bfb056` (round 10) | `1b581cfa` (this work) |
|---|---:|---:|
| Hidden | 2,176 | 1,920 |
| Layers | 25 | 25 |
| Widest layer's FFN (the uniform width) | 10,688 | 3,584 |
| Widest layer's heads (the uniform heads) | 69 | 41 |
| A token's multiplies in the uniform layers | 2,231.7 M | 774.1 M (34.7%) |
| Uniform file: values, bytes | 2,518,195,584, 5,042,980,288 | 1,026,929,280, 2,060,428,032 |
| Layer-shaped file: values, bytes | 457,490,560, 921,575,328 | 408,010,112, 822,595,168 |
| Nonzeros | 1.85 M | 1.85 M |
| Widest live wires: PPU, GBC CPU, APU, control, length, stepper | 1,451, 1,222, 1,079, 1,119, 574, 1,621 | 1,358, 1,222, 1,079, 1,119, 611, 1,339 |
| Clearing units by part | 3,600, 1,201, 2,481, 1,237, 1,247, 2,059 | 3,693, 1,201, 2,481, 1,237, 1,210, 2,341 |

Hidden fell because the stepper set it (543 shared channels and its 1,621 live wires) and
the packer keeps a part's live wires down. 1,920 is the GBC CPU's floor: its slice is that
wide on both sides (`shape --one`), widened past its 1,765 channels for the norm's bound
(`Model::slice`; round 9 measured its squares at 30,502 against 16 · 1,920 = 30,720, and its
channel plan did not move, so the margin is inferred unchanged). The slice now has no wider
checkpoint to grow into: `Model::shape` does not know this floor, so a placement that lands
below 1,920, or a GBC CPU with about 218 more squares, stops in `Model::slice`'s assert. The
wiring's sha1 is `81ccc726` on both, so hand-off snapshots carry over.

| Stream's slice: hidden, layers, widest units, widest heads | `45bfb056` | `1b581cfa` |
|---|---|---|
| scanline PPU | 2,048, 14, 2,861, 21 | 1,920, 21, 1,620, 21 |
| GBC CPU with memory | 1,920, 11, 1,316, 20 | 1,920, 11, 1,316, 20 |
| APU | 1,664, 25, 913, 12 | 1,664, 25, 880, 12 |
| control | 1,664, 17, 631, 11 | 1,664, 17, 631, 11 |
| PPU length | 1,152, 15, 1,442, 21 | 1,216, 21, 981, 21 |
| PPU stepper | 2,176, 9, 5,280, 25 | 1,920, 25, 2,855, 21 |

The new widths by layer (kept units): 1,610, 2,494, 3,194, 3,243, 3,243, 3,246, 3,330,
3,446, 3,446, 3,446, 2,984, 3,445, 3,538, 3,537, 3,537, 3,537, 3,165, 2,197, 3,123, 3,397,
2,725, 3,396, 3,173, 3,203 and 1,152, 76,807 in all (76,469 before: 338 more clearing units).
Heads by layer (`Model::census`: layer 0's gathers, then the circuit's heads, with the two
copy heads at layers 3 and 5): 41, 23, 29, 1, 20, 1, 28, 13, 2, 0, 0, 0, 31, 4, 2, 15, 7, 0,
1, 2, 0, 3, 3, 0 and 0. The layer-shaped file stores one head in a layer that has none,
which is the 1 `shape --one` prints for those seven layers.

**Gates** (heavy lane, one at a time, load 4 to 26 beside the proof).

- `cargo build --release --all-targets`: no warnings.
- `examples/digest`: the six per-part digests above, one model `1b581cfa`.
- `digest --gguf` on both export forms: `manifest ok`, 0 values differing from the compiled
  checkpoint, the uniform file's manifest `llama.cpp b12070, unpatched (uniform layers)`.
- Yellow CGB 300, all five components in weights as one model, in lockstep: `cfd31f3d`, 0
  dense fallbacks. With the step oracle (`cpu=100000,apu=25000,control=6000,ppu=8400`):
  `cfd31f3d`, 10, 11, 12, 10 and 5 fast passes checked, the base's counts.
- Mealybug `m3_bgp_change`, 4 frames, the step oracle on all six streams: 3, 1, 3, 1, 37 and 1
  passes checked, 3 records through the dense record oracle, the base's counts. The stepper
  draws every line there, so this is the run that holds its 25-layer stream to its dense pass.
- The runner's own tests on the tip's GGUF: 24 passed.
- `tools/gate.sh --quick`: 6 ok (DMG 8 `6872b003`, CGB 300 `cfd31f3d`).
- `cargo test --release --features testing`: 27 of 28 result lines ok on the first run. The
  one failure was a literal in `tests/one_model.rs`, the PPU stream's 14 layers, now 21; with
  it corrected the file passes whole, its ignored test included (5 passed, 642 s).
  `every_slice_is_the_checkpoint_at_its_positions` holds the APU's and control's slices to
  the whole checkpoint's dense pass, `the_null_token_decodes_the_null_cell` the whole
  checkpoint over a cache with the null cell, and `stream_audit` the f16 checks and each
  stream's group.
- The stepper's stream, 9 layers before and 25 now, on `m3_bgp_change` for 6 frames with
  every component in weights as one model, no lockstep, two runs a side (a ballpark):
  1,230,372 passes in 33.71 and 33.71 s on the base, 34.28 and 34.08 s on the tip (1.4%
  slower, 27.4 to 27.8 µs a pass), frame `c66c92d4` on both, 0 dense fallbacks.
- The full gate (`GATE_NO_TEST=1 GGUF=<the tip's trimmed export> tools/gate.sh`, one heavy
  ticket of 2 h 01 min): 64 ok, 0 FAIL, 2 skips (the test suite, run above, and Nintendo's
  DMG boot ROM, which is not in the repository), the base verify's counts. Among them, the
  run that holds each stream's slice to the whole checkpoint, which is what an unpatched
  runtime computes: "every stream against the whole checkpoint over real runs: 2038 tokens
  equal" (`examples/slice_full.rs`); the four games with all five components in weights
  (`cfd31f3d`, `ccfe7ad3`, `f10b2b4a`, `ea7f86ca`); the step oracle on 134 OAM rewrite
  passes; and the runner's 17 record-equal legs on the tip's GGUF.

**The engine's work is unchanged** (text bench, every component in weights as one model from
each side's GGUF, `GBCFORMER_UNITS=1`, not timed): the passes, the units re-evaluated per
pass (GBC CPU 715, APU 3,020, control 1,369, PPU 42,459, length 165) and the positions decoded
are equal on both sides, with 0 dense fallbacks. The GBC CPU's, the APU's and control's
changed (channel, reading layer) pairs are equal (134.9, 254.8, 167.0); the PPU's went from
15.9 to 19.8 a pass and the length's from 4.1 to 6.2, since their wires are read in more layers.

**The runner** (text workload, 600 frames from the frame-2,900 hand-off, each side from its
own trimmed GGUF and its own snapshot, three ABBA rounds in one heavy ticket, load 4.8 to
8.0 with the proof on the GPU; every run `870aa3fd` with 0 dense fallbacks):

| Text, 600 frames | Base (fps) | Tip (fps) |
|---|---|---|
| Runner (`tools/runner`) | 62.3 (61.4 to 62.8) | 62.3 (62.1 to 62.3) |
| Full program, one model from the GGUF | 60.65 (60.5 to 61.0) | 60.95 (60.7 to 61.3) |

Medians of six with ranges. No measurable difference on either: the runner's medians are
equal and the full program's ranges overlap. Both stay above 60 in every run. The machine
thread's work is the GBC CPU's and the length's passes, and the GBC CPU's stream did not
move. The numbers are contended (round 10's quiet runner was 64.2), so they compare the two
sides and are not the runner's speed.

**Unpatched llama.cpp on the Mac CPU backend** (b12070 with no patch, built with
`-DGGML_METAL=OFF`; one decode of 99 tokens at position 0, every row asked for, no cache
cells, 6 threads, f16 KV, flash attention; three interleaved rounds of 5 decodes, load 6):

| File | 99 tokens, ms (the rounds' medians) | 10 tokens, ms |
|---|---|---|
| `45bfb056` uniform, 5.04 GB | 553.1, 566.0, 578.0 | 82.9, 82.2, 81.7 |
| `1b581cfa` uniform, 2.06 GB | 226.1, 231.0, 232.3 | 34.0, 34.2, 34.1 |
| ratio of the medians | 0.408 | 0.415 |

This is a ratio of the layers' cost only, on another backend: the tokens are arbitrary ids,
nothing is decoded against a record, and no cache cell is attended, so the part of a Metal
pass that the cells cost is not in it. The count of multiplies with the LM head's 99 rows
predicts 0.379. The 10-token ratio matches the 99-token one, so the fixed part of a decode
scales with the layers too.

**The 300 frames, projected (estimates; nothing ran on Metal).** The measured Metal baseline
is `docs/stock-llama.md` §13's second table: 55.0 hours of passes, 3.9 of write records,
about 7.8 of reader records and about 1.1 of prefill, 68 hours.

- By the Mac CPU ratio alone, 0.408 on every line: 22.4 + 1.6 + 3.2 + 0.5, about 28 hours.
- By the multiplies, a pass's time taken as linear through the two shapes measured on Metal
  (the unpatched file and the patched build's whole model over the trimmed file): GBC CPU
  154.4 to about 61 ms, PPU 163.2 to 59, APU 124.0 to 48, control 128.3 to 49, length 28.2
  to 13, a write record 116.2 to 44, a reader record 600 to 337: about 28 hours with the
  prefill.
- If the time a GBC CPU or PPU pass spends above the APU's rate per token (66 ms of the GBC
  CPU's 154) and the whole of a write or reader record are the cells' attention, they
  scale with the heads alone, 41 of 69: GBC CPU about 75 ms, write record 69, reader record
  356, about 33 hours.

So about 28 to 33 hours against 68. What the estimates leave out: Metal's cost per operation
at the narrower shapes, the cells' share of a pass (unmeasured: the range above), and the
patched tier, whose PPU and length passes now run 21 layers where they ran 14 and 15 (not
measured; at the same rate per layer they add about 0.3 of its 11.6 hours).

**The Metal run, when the GPU is free.** With `K` a directory that holds the harness built
at this tip against the unpatched tree (`tools/stock-llama`, `LLAMA_CPP_DIR` and
`LLAMA_CPP_BUILD` naming that tree and its build) as `bin/stock-llama`, and the uniform file
(`export one-model $K/gguf --gguf-only`), from the repository root (not run: no run was
made with this harness or this file on Metal, and no pass of `1b581cfa` has been decoded by
llama.cpp against a record on any backend):

```sh
GBCFORMER_RECORD=$K/rec-engine-300 $K/bin/stock-llama play none pokemon-yellow.gbc --frames 300
PROOF_PURE=1 FRAMES=1 PROOF_WORK=$K/work PROOF_BIN=$K/bin/stock-llama PROOF_MODEL=$K/gguf/model.gguf \
  PROOF_REFERENCE=$K/rec-engine-300 tools/stock-llama/proof.sh
PROOF_REFERENCE=$K/rec-engine-300 tools/stock-llama/proof-status.sh $K/work
```

The first line records the engine's passes to frame 300 with no model in the runtime (no
GPU; the play must not run past its record, so the record is the whole run's, as the
round-10 dry run's was). The second prefills the
cache in unpatched llama.cpp and decodes frame 1's 13,167 passes against that record under
the GPU lock; its log gives each stream's ms per pass, which replace the estimates above.
The frame must be `d043ab24`.

**What a second round could gain** (counted in a token's multiplies per layer, 16,128 hidden
rows now: 10,752 for the FFN, 5,248 for the heads, 128 for the key and value).

- The heads: 41 is layer 0's input gathers, one head per group of eight input slots per part
  (38) and the 3 record gathers. A gather shared by the parts (its query the group's code
  times the part's mode) would need 12 and 3, the widest part's: 15 heads, 21% fewer
  multiplies, and 63% less of whatever the cells' attention costs.
- The layers: the APU's 24 levels set the 25 layers, and every layer costs its heads. An APU
  as deep as control (16 levels) would give 17 layers of width about 4,544: 20% fewer
  multiplies, 34% with the shared gathers.
- The width: the bound is 3,143 and the placement reaches 3,538. The packer reserves each
  part's routing units by `routing_units`' upper count (four family bits an output, which
  only record slots have: 1,950 for the PPU's 1,298), so the routing layers 17 and 20 end
  at 2,197 and 2,725. Exact counts and a second pass over the clearing units are worth
  about 7%.
- The LM head: 65,804 rows at hidden for every decoded row, about 11% of a GBC CPU pass's
  multiplies now (estimate). The vocabulary is 93% input tokens, and shared input slots
  (`docs/one-model.md` §3) would cut it to about a third.
- Hidden: 1,920 is the GBC CPU slice's floor for the norm's bound, and the PPU and the
  stepper hold 1,358 and 1,339 live wires over 543 shared channels.
- The length's depth: with the length at 22 levels the scan reaches width 3,520, 34.3% of
  the base's multiplies against this placement's 34.7% (`room -- --scan 20,23 20,24
  20,22`). The placement keeps the length at 20 for the engine's reader buckets, a
  judgment: their worth to the length's passes was not measured (the stepper without
  them is 1.4% slower, the ballpark above).

## Re-levelling, round 2: the review's gaps closed

The pins in this section are superseded. The shape below (widest MLP 3,584, the census's
widest layer 3,538, at most 31 memory heads a layer, the length's hidden 1,216) is what the test held when
this section was written. The next section moves the one model, and
`the_one_models_shape_is_pinned` now holds 3,456, 3,424, 29 and 1,152. The "do not pass" asserts named below were implied
by the equalities before them, and "Round 12" replaces them.

An independent review of the re-levelling found no defect and these gaps in what held it.
Each is closed by a test or an assert, and no digest moved while closing them
(`examples/digest -- --check tools/digests.txt`: 8 checkpoints equal). Measured 2026-11-21
on the M1 Max beside the 300-frame proof of `45bfb056` on the GPU (load 4 to 18); nothing
here used the GPU.

**Every part's slice against the whole checkpoint**
(`every_parts_slice_is_the_checkpoint_at_its_positions`, `tests/one_model.rs`). An
unpatched runtime computes the whole checkpoint at every position, and the test that held
a slice to it covered the APU and control only. The new test runs all six parts: one pass
each on random input bytes the part's own circuit accepts, over a cache of every line the
circuit reads there and 1,024 other lines of a random memory (ROM, each RAM, OAM and the
palettes random; the registers and constant blocks a cartridge's). At each output position
the residual is bit-identical on the slice's channels, 0 above them, the tokens are equal,
and each token is the byte the part's own circuit gives.

| Part | Slice: hidden, layers | Lines read | Output positions equal | With no null cell, positions that differ |
|---|---|---:|---:|---:|
| scanline PPU | 1,920, 21 | 88 | 163 | 163 |
| GBC CPU | 1,920, 11 | 10 | 119 | 119 |
| APU | 1,664, 25 | 0 | 71 | 71 |
| control | 1,664, 17 | 0 | 93 | 93 |
| PPU length | 1,216, 21 | 21 | 5 | 5 |
| PPU stepper | 1,920, 25 | 27 | 79 | 79 |

The last column is the test's own proof that it can fail: the same pass with no null cell
lets the other parts' heads read a line of the cache, and the test asserts that every part
then differs. The review's open item, 14 of the GBC CPU's 119 outputs that did not match
its own circuit in the reviewer's scratch test, was that test's decoding: the 14 are the
record slots (6 address bytes and 8 line bytes), which decode to family tokens, and
`tok::output_value` reads only byte tokens. Read with `tok::byte`, all 119 are the
circuit's, over nonzero memory. The test asserts the count of family tokens, 14 for the GBC
CPU and 0 for the others. The circuit refuses some random inputs by panicking, which the
test catches; the test harness captures those lines, so `tools/gate.sh`'s search for
`panicked` in the suite's output stays clean (0 such lines from this test in the full run).

**A cell's rows at every memory head**
(`a_write_records_cell_is_the_caches_at_every_memory_head`). The stream's record oracle
runs the GBC CPU's slice, 11 layers, and the memory heads now sit at layers 1 to 8, 12 to
16, 18, 19, 21 and 22 (the circuit's 183, with the two copy heads at 3 and 5). The test
runs the whole checkpoint's dense pass over 27 write records (VOID's line, the registers,
ROM lines that set the key's high bits, each RAM, a line written twice, 12 random VRAM
lines) and holds the cell token's key and value at each of the 185 heads to the formula's
key and the line's value, bit for bit, which is what `KvCache` hands the dense pass for a
cell. It then holds the rows the engine's `RowBuilder` computes from the whole
checkpoint's weights, at each of the 17 memory layers (3 row groups), to the same key at
the line's position and the same value in f16. Both comparisons also assert that the next
line's key and another line's value do not pass.

**The shape, pinned** (`the_one_models_shape_is_pinned`). Hidden 1,920, widest MLP 3,584,
41 heads, 25 layers; the census's widest layer 3,538 units, 41 gathers, at most 31 memory
heads a layer; each stream's (hidden, layers): (1,920, 21), (1,920, 11), (1,664, 25),
(1,664, 17), (1,216, 21), (1,920, 25). The literals the placement used are derived or held
where they are used: the head cap is layer 0's gathers less the copy heads
(`one_model::head_room`, 39), the PPU's and the length's depth is `engine::BUCKET_LAYERS`
(20, from the reader mask's 64 bits), and the 1,920 floor of `room -- --scan` is
`one_model::CPU_SLICE_HIDDEN`, which the test holds to the GBC CPU's compiled slice.

**The packer's caps.** `Graph::pack` took a layer's room with `saturating_sub`, so a pinned
item or a forced one that went past the cap left no trace. The room is signed now and goes
below zero (`level::Room`), `level` returns how far the fullest layer ended over in units
and in heads (`level::Over`), and `one_model::circuit` asserts the canonical placement
ends 0 over both. The caps stay soft by design, and `Room`'s doc says why: `room -- --scan`
tries caps below what fits, and a unit's cost charges its part's clearing units in
proportion, an estimate (the compiled widest layer is 3,538 under a cap of 3,520). The
packer's decisions are unchanged: the digest is.

**The packer's tests** (`src/level.rs`, 8 tests). The three tests asserted
`values == before` on a circuit evaluated in creation order, which no placement can
change. They now assert each node's level and the reported overflow, and five cases are
new: a node with no later level lands at its earliest and the layer ends 6 units over; a
part at its live-wire limit keeps its nodes for lower levels (limits of 6, 7 and 8 give
three different placements); two heads on one bus are one item and move as one; a pinned
part's head takes its layer's room; and a re-levelled overlay of two parts, compiled,
decodes each part's circuit at all 256 input bytes at the part's own output position, with
the other part's input at 0. An overlay's parts share residual channels, so one part's
input lands on the other's wires: the test also asserts that the chain's input `0x07`
makes the wide part's position decode 4, which is why a stream holds the other parts'
inputs at 0. That test needed an overlay without memory to compile, which it did not: the
copy heads' layers and the record depth were computed for every circuit, and the null
token's index assumed the special tokens. Four expectations were wrong on a first run and
were re-derived by hand from the packer's rule (it fills from the last level down, taking
the node that adds the fewest live wires): which of six equal nodes sits highest, where
`a` lands at a limit of 7, that both positions of the overlay decode a part's byte (only
the part's own does), and the count of distinct outputs (64, not 65).

**The rest.** `examples/digest -- --check tools/digests.txt` holds every digest to a
pinned file and `tools/gate.sh` runs it, so a placement that moves a part checkpoint
fails the gate; the PPU length's own checkpoint joined the list (`c42658f1`).
`one_model::PLACEMENT` is a struct with named fields and `Graph::pack` takes a
`level::Request`. `README.md` gave `45bfb056`'s sizes. The heads-by-layer list above gave
the layer-shaped file's stored heads, one in a layer that has none; it now gives the
census. A comment in `src/engine/run.rs` still said the positional masks were u32.

**Gate.** `cargo test --release --features testing`, every file, one heavy ticket (46 min
at load 6 to 18): 28 result lines, 27 ok; the one failure was the count of distinct
outputs above, a literal of the new overlay test, and the packer's 8 tests pass with it
corrected. `tests/one_model.rs` passes whole in 914 s (7 passed, 1 ignored).

## Re-levelling, round 2: routing units counted as written, width 3,584 to 3,456

One change to the one model. It moves only the one model's digest, `1b581cfa` to
`7b064d51`; the other seven checkpoints are unchanged (`examples/digest -- --check
tools/digests.txt`, which failed on the one model alone before its line was updated).
Measured 2026-11-21 on the M1 Max beside the 300-frame proof of `45bfb056` on the GPU
(load 4 to 19): no timing here is quiet, and none used the GPU.

**The lever.** The packer reserved each layer's routing units by an upper count: four
family units for every output, where only a record slot has any and rarely four. Counted
as the compiler writes them (`llama::routed_units`, `reader_units`) the circuit before it is
placed holds 76,469 units, not 78,574 (the placed model the compiler writes holds 76,802,
the sum of `shape --one`'s layers: placing adds clearing units), and the routing layers 10, 16, 20 and 24 have that room.
`Model::census` counts an overlay exactly now, and `compile_with` asserts that the units
it writes equal the census layer by layer, so the shape the placement is packed by is the
compiled one. The per-part checkpoints keep the upper count, since their clearing units
are placed by it, and so keep their digests.

**The placement** (`ROOM_CAPS=3400:3640:8 ROOM_LIVES=1279:1377:2 room -- --scan 20 24 20`,
1,550 placements in 6 minutes). Width is not what binds: hidden is. A lower cap spreads
the PPU's nodes over more layers, which keeps more of its wires live, and past 1,377 live
wires (1,920 less the 543 channels the parts share) hidden goes to 1,984 and costs more
than the width saved. Of the 1,550, 1,071 end within their caps; one reaches width 3,456
at hidden 1,920 (3,480 units and 1,371 live wires a part, `one_model::PLACEMENT`), and the
next 29 by cost are 3,520 at hidden 1,920. A coarser scan (caps by 16, live limits by 8)
found width 3,456 fourteen times, each at hidden 1,984. A scan below a cap of 3,400 found
none that ends within its caps (its log keeps the result, "0 placements fit", and not the
ranges scanned, so no count is given here). The placement has no room left in
hidden: the scanline PPU keeps 1,377 wires live and the stepper 1,376. The bound with
every node free is 3,059 units (3,072) at 25 layers; the packer ends 12% over it with the
GBC CPU, the APU and control held where they are.

**Old against new** (`examples/shape --one`, `examples/export`, `examples/digest --gguf`).

| | `1b581cfa` | `7b064d51` |
|---|---:|---:|
| Hidden | 1,920 | 1,920 |
| Layers | 25 | 25 |
| Widest layer's FFN (the uniform width) | 3,584 (widest layer 3,538) | 3,456 (3,424) |
| Widest layer's heads (the uniform heads) | 41 | 41 |
| A token's rows in one layer (FFN, heads, key and value) | 16,128 | 15,744 (97.6%) |
| Uniform file: values, bytes | 1,026,929,280, 2,060,428,032 | 1,008,497,280, 2,023,564,128 |
| Layer-shaped file: values, bytes | 408,010,112, 822,595,168 | 408,489,344, 823,553,728 |
| Nonzeros | 1.85 M | 1.85 M |
| Widest live wires: PPU, GBC CPU, APU, control, length, stepper | 1,358, 1,222, 1,079, 1,119, 611, 1,339 | 1,377, 1,222, 1,079, 1,119, 560, 1,376 |
| Clearing units by part | 3,693, 1,201, 2,481, 1,237, 1,210, 2,341 | 3,674, 1,201, 2,481, 1,237, 1,261, 2,304 |
| Each stream's slice (hidden, layers): PPU, GBC CPU, APU, control, length, stepper | (1,920, 21), (1,920, 11), (1,664, 25), (1,664, 17), (1,216, 21), (1,920, 25) | the same but the length's (1,152, 21) |
| Memory heads' layers | 1 to 8, 12 to 16, 18, 19, 21, 22 | 1 to 8, 11 to 16, 19, 21, 22 |

The layer-shaped file stores 479,232 more values (0.12%): the scanline PPU's and the
stepper's prefixes are 19 and 37 channels wider. The wiring's sha1 is `81ccc726` on both,
so hand-off snapshots carry over. The widths by layer (kept units): 1,610, 2,494, 3,143,
3,143, 3,181, 3,183, 3,289, 3,406, 3,424, 3,423, 3,423, 3,314, 3,313, 3,313, 3,411, 3,411,
3,411, 2,692, 2,694, 3,308, 3,308, 3,308, 3,224, 3,224 and 1,152. Heads by layer
(`Model::census`): 41, 23, 29, 1, 20, 1, 27, 12, 4, 0, 0, 10, 22, 1, 3, 12, 11, 0, 0, 3, 0,
3, 3, 0 and 0.

**Gates** (heavy lane, one at a time).

- `cargo build --release --all-targets`: no warnings.
- `the_one_models_shape_is_pinned` with the old literals fails on the new model, as it
  should: `left: (1920, 3456, 41, 25), right: (1920, 3584, 41, 25)`. With the new pins it
  passes.
- `cargo test --release --features testing`: 28 result lines, 28 ok, 0 lines with
  `panicked` (42 min at load 4 to 19). `tests/one_model.rs`'s new tests on the new model:
  every part's slice equal to the whole checkpoint at 163, 119, 71, 93, 5 and 79 output
  positions, and all of them differing with no null cell; 27 write records' cells held at
  185 heads and 17 memory layers.
- `examples/digest -- --check tools/digests.txt`: 8 checkpoints equal.
- `digest --gguf` on both export forms: `manifest ok`, the uniform file's manifest
  `llama.cpp b12070, unpatched (uniform layers)`.
- `tools/gate.sh --quick`: 6 ok (DMG 8 `6872b003`, CGB 300 `cfd31f3d`).
- Yellow CGB 300, all five components in weights as one model, in lockstep, with the step
  oracle (`cpu=100000,apu=25000,control=6000,ppu=8400`): `cfd31f3d`, 0 dense fallbacks,
  10, 11, 12, 10 and 5 fast passes checked, the base's counts.
- Mealybug `m3_bgp_change`, 4 frames, the step oracle on all six streams: `c66c92d4`, 3, 1,
  3, 1, 37 and 1 passes checked, 3 records through the dense record oracle, the base's
  counts.
- The runner's own tests on the new GGUF: 24 passed.
- The full gate at the tip (`GATE_NO_TEST=1 GGUF=<the tip's trimmed export> tools/gate.sh`,
  one heavy ticket of 2 h 00 min): 65 ok, 0 FAIL, 2 skips (the test suite,
  run above, and Nintendo's DMG boot ROM, which is not in the repository). Among them the
  digests against `tools/digests.txt`; "every stream against the whole checkpoint over
  real runs: 2038 tokens equal"; the four games with all five components in weights
  (`cfd31f3d`, `ccfe7ad3`, `f10b2b4a`, `ea7f86ca`); Yellow 300 as one model with the step
  oracle; and the runner's 17 record-equal legs on the tip's GGUF. The code under the
  gate ran at the tip less documents; the commits after it change documents only.
- `the_null_token_decodes_the_null_cell`, ignored by default: ok (77 s).

**The engine's work** (text bench, every component in weights as one model from each
side's GGUF, `GBCFORMER_UNITS=1`, not timed): the passes, the units re-evaluated per pass
(GBC CPU 715, APU 3,020, control 1,369, PPU 42,459, length 165) and the positions decoded
are equal on both sides, with 0 dense fallbacks. The PPU's changed (channel, reading
layer) pairs went from 19.8 to 20.5 a pass and the length's from 6.2 to 6.1.

**The runner and the full program** (text workload, 600 frames from the frame-2,900
hand-off, each side from its own trimmed GGUF and its own snapshot, three ABBA rounds in
one heavy ticket, load 4 to 11 with the proof on the GPU; every run `870aa3fd` with 0
dense fallbacks; medians of six with ranges):

| Text, 600 frames | `1b581cfa` (fps) | `7b064d51` (fps) |
|---|---|---|
| Runner (`tools/runner`) | 61.95 (61.7 to 62.7) | 62.15 (61.0 to 62.4) |
| Full program, one model from the GGUF | 61.1 (60.2 to 62.0) | 60.85 (59.3 to 61.5) |

The runner shows no measurable difference and stays above 60 in every run. The full
program's new side was lower in five of the six pairs, and under 60 twice (59.5 and 59.3,
in the round where the old side ran 60.3 and 60.2). A second ticket ran each binary on
each file to find which it follows (three rounds of the four in ABBA order, load 4):

| Full program, text, median of 6 (range) | old file | new file |
|---|---|---|
| old binary (Part A's tree) | 61.75 (60.9 to 62.3) | 61.8 (61.6 to 62.2) |
| new binary (this commit) | 61.55 (61.1 to 61.7) | 61.3 (60.9 to 61.5) |

With the old binary the new file runs as the old one does, and the GBC CPU's passes take
the same time (2.96 to 3.03 µs on the old file, 2.97 to 3.00 on the new). So the file does
not cost the full program anything measurable. The new build is 0.2 to 0.5 fps lower in
the medians on either file, inside the runs' range of 1.4. The engine's source is the same in both builds, which
points at the build's layout and not at work added; that is inferred, since nothing
profiled the two binaries. These numbers are contended
and compare the sides; they are not the program's speed.

**Unpatched llama.cpp on the Mac CPU backend** (b12070 with no patch, built with
`-DGGML_METAL=OFF`; one decode of 99 tokens at position 0, every row asked for, no cache
cells, 6 threads, f16 KV, flash attention; medians of 5 decodes, interleaved, load 4 to 6):

| File | 99 tokens, ms, first ticket's three rounds | 99 tokens, ms, second ticket (the runs' medians) | 10 tokens, ms |
|---|---|---|---|
| `45bfb056` uniform, 5.04 GB | 570.2, 723.6, 741.3 | 582.9, 589.2, 588.7 | 92.0, 94.7, 92.2 |
| `1b581cfa` uniform, 2.06 GB | 235.8, 299.8, 300.5 | 231.8, 244.3, 244.6, 240.7, 236.6, 232.3 | 33.87, 33.82, 33.86 |
| `7b064d51` uniform, 2.02 GB | 266.1, 291.2, 289.1 | 223.9, 231.2, 231.9, 231.7, 225.5, 224.0 | 33.61, 33.68, 33.51 |
| new against old | 1.129, 0.971, 0.962 | 0.957 (228.3 against 238.7) | 0.992 |
| new against `45bfb056` | 0.467, 0.402, 0.390 | 0.388 | 0.36 |

The first ticket's first round is an outlier on the new file (its five decodes ran from
225 to 273 ms) and its later rounds ran a quarter slower on every file, so the second
ticket is the one to read: the new file takes 0.96 of the old one's time at 99 tokens
and 0.99 at 10 tokens. The count of multiplies with the LM head's 99 rows allows 0.980,
so the 0.96 is partly noise in the old file's runs (231.8 to 244.6 ms), and the
projection below uses 0.98. As in round 1 this is a ratio of the layers' cost only, on another backend: the
tokens are arbitrary ids, nothing is decoded against a record, and no cache cell is
attended.

**The 300 frames, projected (an estimate; this work ran nothing on Metal).** Round 1 put
`1b581cfa` at about 28 to 33 hours against `45bfb056`'s measured 68, from the Mac CPU
ratio and the count of multiplies. While this round ran, frame 1 of `1b581cfa` was decoded
in unpatched llama.cpp on Metal outside this work (its log, read here and not made
here: frame 1 `d043ab24`, 13,167 passes, the prefill 2,631 s). Its rates a pass are GBC CPU
100.5 ms, PPU 95.6, APU 69.8, control 73.4 and length 18.9, against 154.4, 163.2, 124.0,
128.3 and 28.2 on `45bfb056`: 0.56 to 0.67 of the old file's, 0.63 over frame 1's mix of
passes, where the Mac CPU ratio said 0.41. So what a pass costs on Metal is mostly not
the layers' multiplies. At those rates the 68 hours' lines come to about 34.5 hours of
passes, 2.8 of write records (82.7 ms against 116.2), 4.6 of reader records (355 ms
against 600, one record) and 0.7 of prefill: about 43 hours for `1b581cfa`. This change
takes 2.4% of a token's multiplies (2.0% of a 99-token decode with its LM head rows; 1
to 4% measured on the Mac CPU) and leaves the heads at 41, so `7b064d51` is at most 2%
under that: about 42 to 43 hours. No pass of `7b064d51` has been decoded by llama.cpp
against a record on any backend.
The Metal run's commands are the ones under "The Metal run, when the GPU is free" above,
with the harness built at this tip and this tip's uniform file; frame 1 must still be
`d043ab24`.

**The other levers, bounded and not built** (`examples/room`, the circuits alone; a token's
multiplies in one layer are `hidden · (3 · width + 128 · heads + 128)`, so at hidden 1,920
the model is 25 layers of 15,744 rows now: 10,368 FFN, 5,248 heads, 128 key and value).

| Lever | Bound | A token's rows, 25 x 15,744 = 393,600 now | Why not built this round |
|---|---|---:|---|
| Heads: gathers shared by the parts | 41 to 20 with the GBC CPU where it is, 16 with its heads re-levelled | 326,400 (-17.1%), 313,600 (-20.3%) | the layout, the slices and the engine's layer 0 change |
| Layers: an APU of 16 levels | 25 to 17 layers at width 4,544 or more | 323,136 (-17.9%) at the bound | the APU's circuit, and its digest |
| Both | 17 layers, width 4,544, 16 heads | 268,736 (-31.7%) | |
| LM head: input slots shared by the parts | vocabulary 65,804 to 24,684 | about 7% of a GBC CPU pass's multiplies (99 tokens; 6.9% at 74 rows decoded, 7.2% at `docs/stock-llama.md` §13's 77) | the same redesign as the shared gathers |

Round 1 put the width lever at about 7%. With the GBC CPU, the APU and control held where
they are and hidden at 1,920 it was 2.4%, built above; the rest of the 12% over the bound
needs those three parts re-levelled too, or a packer that holds hidden as a limit and
not as a preference.

- **Heads.** Layer 0 holds 41 gathers: one per group of eight input slots of each part
  (2, 5, 9, 8, 2 and 12 for the scanline PPU, the GBC CPU, the APU, control, the length and
  the stepper) and the 3 record gathers. Shared by the parts they would be the widest
  part's 12 and the 3. Past layer 0 the floor is not 15: the GBC CPU's largest bus is 16
  heads, which sit in one layer, and the GBC CPU as it is placed holds 20 heads at layer 4
  and 17 at layer 6. What a shared gather needs: its query names group h of whichever part
  the position belongs to (the mode channels in place of the constant), and its output
  projection is one per head, so lane i of group h must land on the same residual channel
  in every part. Today each part plans its wires on its own (`Circuit::overlay_channels`),
  so the parts' input wires sit at unrelated channels, and an input bit a part does not
  read is a channel another part uses. The channel plan, `Model::slice` (a slice clears the
  mode channel its queries would read), the layer-shaped file's head prefixes, the
  engine's layer 0 and the runner all follow. The risk is hidden: aligned input channels
  that a part does not read are live in it at layer 0, and the scanline PPU has none to
  spare (1,377 of 1,377).
  The packer as it is does not hold a head cap below the gathers' either (`room -- --try
  3480 <heads> ppu=20,stp=24,len=20[,cpu=10] 1371`): at caps of 20, 18 and 14, with the GBC
  CPU held or re-levelled, it ends 24 to 30 heads over, with 44 heads at layer 1. Hidden
  and width hold there (1,920 and 3,456). So 20 and 16 are floors by
  arithmetic, and a packer that spreads those heads is part of the lever.
- **Layers.** The APU's critical path is one chain, 24 levels with no slack: the sweep's
  calculation and overflow (levels 1 to 11), the square channel's stop, wrap and index (12
  to 17), the sample (18 to 20) and the mix (21 to 24) (`room -- --path apu`). Levels 4 to
  13 of it hold 29 units, so the chain itself is thin; but 4,420 of the APU's 14,675 node
  units have less than 8 levels of slack, and every chain through them must shorten for 16
  levels. Seventeen layers cannot be narrower than 76,469 / 17, 4,499 units, and the
  packer ends 12% over its bound today, so the gain is between 11% and 18% of the
  multiplies before the heads. It moves the APU's own checkpoint (`d8d3a06e`), so every APU
  gate runs again.
- **The LM head.** Each decoded row scores 65,804 tokens at hidden, 126.3 M multiplies
  against a token's 755.7 M in the layers. 61,423 of them are the input slots' own byte
  tokens (239 slots of 257). An unpatched runtime cannot score fewer rows than the
  vocabulary, so the lever is the vocabulary: input slots shared by the parts (79, the
  stepper's) would make it 24,684. A stream's pass would then hold the whole of each
  group it uses, so every gather still weighs its eight keys 1/8 each.

## Round 12: the two reviews closed, the re-levelled model under the narrowed scopes, Yellow to frame 13 in unpatched llama.cpp (2026-11-24)

Round 12 merges the re-levelling (the re-levelled one model, `7b064d51`) into the narrowed scopes
(the harness's narrowed scopes) and closes the findings of the two independent reviews.
No digest moves: `digest --check` holds all eight to `tools/digests.txt`, and the part
checkpoints are the re-levelling's. Three things changed outside the harness: an assert in
`one_model::placed`, the test file, and the lines `Engine::record` lists for a pass.

What a user of the proof gets: the re-levelled model has now run in unpatched llama.cpp
b12070 on Metal from the hand-off through frame 13 of Yellow, 138,984 passes, every one
equal to the engine's record, in four chunks of at most 45 minutes with three resumes.
What it does not get: the patched build's stream slices, which give NaN on Metal with
this model (below). With the slices off the patched build runs it.

**The run** (measured; M1 Max, load average 3 to 5, `PROOF_PURE=1 FRAMES=13
PROOF_CHUNK_SECS=2700` through `proof.sh`, the uniform file, its own
prefill, every pass under `STOCK_REFERENCE`).

- Exit 0, 0 error lines, frame 13 `d043ab24` (the engine's), every stream's record equal
  to the reference's. 122,303 GBC CPU, 1,120 PPU, 10,908 APU, 4,084 control and 569 length
  passes; 19,563 write records; 321 reader records, none off the mirror.
- It covers what frame 1 does not: 28 passes after a banking write that decoded first
  for the windows, 2,398 passes that decoded again for a named line, 2,567 lines joining
  the narrow scope of 256 (so the least recently named leave), and the OAM DMA's reader
  records from frame 12 (320 of the 321).
- Four chunks held the GPU lock 2,702, 2,701, 2,701 and 1,447 s, 2 h 39 min in all, the
  prefill's 2,581 s in the first. Each resume replayed the host to its checkpoint in 94
  to 118 s and went live at a checkpoint an earlier chunk saved. 32 checkpoints took
  0.72% of the time.

| | Count to frame 13 | ms each (measured) | Count in 300 frames | Hours in 300 frames (inferred) |
|---|---:|---:|---:|---:|
| GBC CPU pass (its second decodes inside) | 122,303 | 42.8 | 925,752 | 11.0 |
| PPU pass | 1,120 | 77.7 | 83,498 | 1.8 |
| APU pass | 10,908 | 43.5 | 251,674 | 3.0 |
| Control pass | 4,084 | 43.4 | 71,759 | 0.9 |
| Length pass | 569 | 11.4 | 41,855 | 0.13 |
| Write record | 19,563 | 27.0 | 119,435 | 0.9 |
| Reader record | 321 | 28.2 | 46,951 | 0.37 |
| Prefill | | 2,581 s | | 0.72 |
| **Sum, with 0.72% for checkpoints** | | | | **19.0** |

The projection takes thirteen frames' rates for 300 and the proof of record's counts. The
decode after a write record carries llama.cpp's shift of the new cell (19,261 decodes at
47.5 ms against 42.8), which is inside the pass rates. The GBC CPU's 42.8 ms is above
frame 1's 39.4 (the first 3,000 passes alone, same day) . The second decodes and the shifts of frames 2 to 13 would account for it (inferred from
their counts, not traced). No stepper pass ran: Yellow has none.

**The patched build's stream slices give NaN on Metal with the re-levelled model.** With
the trimmed file of `7b064d51` in llama.cpp b12070 with the three patches and each pass on
its stream's slice (`STOCK_STREAMS=require`, the patched kit's setting), the first decoded
row is NaN and the harness stops (`cpp::Row::token`: "its largest score NaN"). The two verify legs
(4,000 and 3,500 passes) and Zelda DX to frame 7 each stopped there after about 70 s.
What it follows (measured, the first 300 passes of Yellow over D1's saved cache, each
under `STOCK_REFERENCE`): the patched build runs the trimmed file with the whole model
every pass on Metal (`STOCK_STREAMS=0`: all 300 equal), the uniform file likewise, and
the trimmed file on its stream slices on the Mac CPU backend (`STOCK_GPU_LAYERS=0`: all 300
equal). Only the stream slices on Metal give NaN, in every group form tried. So the
file's loader (layer shapes, stored runs) is not at fault, and the fault lies in
`stream-slices.patch` on Metal with this model's slices. `STOCK_STREAMS=check` names the
pass (measured): the run's first APU pass decodes on its slice, and the first control
pass, the next, has NaN in all 93 of its rows on its slice against finite rows on the
whole model. Why was not traced in this round. It is now: a run of `ffn_down` columns that
starts on an odd unit, fixed in the patch ("The slices' NaN on Metal", below). The guess
made here, the slice's 1,664 channels and 17 layers, was wrong.

With the slices off, the patched build's legs hold on this model (measured, load 3 to 5):

| Leg, patched llama.cpp, the trimmed file, `STOCK_STREAMS=0` | Result |
|---|---|
| Yellow, 4,000 passes, the scopes and the split on | all equal; `diff` of the run's record against the engine's: 4,001 events equal, 313,445 tokens. GBC CPU 21.3 ms, PPU 34.6, APU 24.6, control 24.8, length 8.2 |
| Yellow, 3,500 passes, `STOCK_SCOPE=0 STOCK_SPLIT=0` (every token over every cell) | all equal; `diff`: 3,501 events equal, 274,258 tokens. GBC CPU 394.7 ms |
| Zelda DX frame 7 (`--warmup 6`), its own prefill (755.7 s), `STOCK_WRITES=check STOCK_READS=check` | exit 0, frame 7 `d043ab24`, all 21,110 passes equal (19,491 GBC CPU, 238 PPU, 845 APU, 416 control, 120 length); 19,450 narrow passes, 626 decoded twice, 4 first for the windows; 216 PPU passes lent 438 lines, 876 loan records, 424 cells put back and compared, 0 differ, 0 unchecked; 3,126 write records both ways, 0 cells differ; 160 reader records both ways, 0 rows differ, 0 off the mirror |

These are the first loans, bank switches and reader records of the re-levelled model in
a llama.cpp runtime. They ran in the patched build, not the unpatched one.
the re-levelling had run this model in llama.cpp only as timing decodes of the uniform file on
the Mac CPU backend, not on Metal and not in the patched build, and the gate has no
llama.cpp leg, so nothing held this before. The file itself reads back right: `digest --gguf`
expands it tensor by tensor to the compiled checkpoint, and the runner and the full
program run from it. The unpatched claim does not depend on the patched build.

**The scope review's findings** (harness only; each stop below was seen to fire).

| Finding | What was done | How it was seen to fail |
|---|---|---|
| A misplaced first-head line decodes a wrong token unseen without the record | Four additions, below. What remains: without `STOCK_REFERENCE` no check sees every pass, and the docs now say so (`docs/stock-llama.md` §13) | |
| a. The windows | A pass that decodes `MAP` must name the windows its lines were placed under, and must name windows at all (`NARROW SCOPE NOT HELD`, `WINDOWS SCOPE NOT HELD`). An unconfirmed first decode's windows no longer stay when the last `MAP` slot is no byte | `STOCK_FAULT_MAP=200`: stops at GBC CPU pass 201 in the replay and in unpatched llama.cpp with no record. `STOCK_FAULT_PROBE=0`: stops at pass 0 in the replay |
| b. `READ DIFFERS` | `READ NOT HELD` stops in every mode, the start-up probe included; so does a row that differs between two decodes a check compares (`CHECK NOT HELD`) | `STOCK_FAULT_READ=0`: stops in the replay, and in llama.cpp with no record at the probe's first reader record |
| c. The name table `FIRST_HEADS` | Gone. At start-up `scope::sampled_reads` evaluates the GBC CPU's own circuit on 2,000 random steps and holds every head's line to what the narrow scope holds: 1,288 steps held on Yellow's image (266 refused by the circuit, 441 in cartridge RAM, 5 with a head at a line no cache holds). The two fixed heads must be the banking registers'. Still by hand: `first_addresses`, `bus_lines`, which passes are narrow, and that a head at a line with no cell is no reachable state | Three unit tests: without SP + 1 among the addresses the sample fails on read 1's head; a head that is not a first head, taken for one, fails it (the test requires that at least one of the other heads does); a `MAP` or address slot wired to the wrong head fails the wiring assert |
| d. A sampled `STOCK_SCOPE=check` | `STOCK_CHECK_EVERY=<n>`; each sampled pass also reads its sequence back from llama.cpp and holds its cells to the harness's account (`Cache::audit`) | `STOCK_FAULT_SEQUENCE=300`: `SCOPE SEQUENCE DIFFERS` at the next sampled pass, 484 cells against 485. The audit's first real run stopped on its own mistake (it counted lines without a cell), since fixed |
| `SCOPE MISSES LINE` counted the APU and control as held | A stream whose record names no line is not counted. The PPU's per-position heads were missing from the record's lines: `Engine::record` now lists them, and the replay of the 300 frames holds all 83,498 PPU passes with them | The replay's counts: APU and control 0 held, PPU 83,498 |
| `argmax` | Tests for a maximum past index 256 and for ties either side of a block's edge | A mutant that skips the first block fails both `argmax` tests |
| Counters | Sliced, fenced and check counts are of passes; the shift line says decodes | |
| Fault injection for both `NARROW SCOPE NOT HELD` stops | `STOCK_FAULT_NAMED`, `STOCK_FAULT_PROBE`, `STOCK_FAULT_MAP`; `tools/stock-llama/faults.sh` runs six replays (one clean, five faults) and checks each stop | `faults.sh`: 6 ok |
| Docs | The `1b581cfa` runs were not interleaved (above, corrected in place); what "held every line" covers | |

The sampled check's cost (measured, unpatched, the first 3,000 passes of Yellow,
`STOCK_CHECK_EVERY=100`, load 3 to 4): 29 passes checked, 0 rows differ, 29 sequences read
back at 3.1 ms each. A checked GBC CPU pass costs 0.94 s more (16 passes, 15.1 s), an APU
pass 1.02 s (11 passes). The decodes took 148.0 s against 120.5 s for the same 3,000
passes without the check (one run each), 23% more at one pass in 100, so about 2.3% at
one in 1,000 (inferred).

One mistake of this round, caught by the replay: a first form sent every pass that
carries a DMA line's address to the windows' scope, on the premise that a step holds
those inputs at 0. They hold the last DMA line's addresses from Yellow's frame 12 on, and
the replay showed 99,574 narrow passes where there had been 913,778. The sample passes
with those inputs random, so the premise went.

**The re-levelling review's findings.**

| Finding | What was done | How it was seen to fail |
|---|---|---|
| One added wire compiles hidden 1,984 silently | `one_model::placed` asserts the compiled hidden is `CPU_SLICE_HIDDEN` beside the `Over` assert | `a_placement_that_widens_hidden_is_refused`: a placement that keeps two more wires live fits its caps (`room --scan` lists it at hidden 1,984) and panics with "compiles a checkpoint of hidden 1984, not 1920". It stands in for an added wire: no circuit was changed |
| Vacuous asserts | The census-against-cfg compare is gone (both come from `shape_census`). The two "also passes as another line" asserts were implied by the equalities before them; in their place the key formula must tell a line from the next, and a value from its complement's, in f16 | Not mutated: these hold the oracle's formula, not the model |
| Doc numbers | 76,469 is the circuit before it is placed; the compiled model holds 76,802 (the sum of `shape --one`'s 25 layers, measured). "950 tried" is dropped: its log keeps "0 placements fit" and no ranges. The first round-2 section points to the second for its pins | |

**Full-program and runner fps.**
Text bench, old (round 10, one model `45bfb056`) against new (round 12, one model
`7b064d51`), each from its own trimmed GGUF (`GBCFORMER_GGUF_ONLY`) and its own hand-off
snapshot, three ABBA rounds, six runs a side, every run frame `870aa3fd` with 0 dense
fallbacks (measured, in one heavy-queue ticket, load average 3.8 to 6.1):

| | Old, median (range) | New, median (range) |
|---|---:|---:|
| Runner, fps | 63.5 (62.5 to 63.8) | 63.75 (63.4 to 64.0) |
| Full program, emulated fps | 62.2 (61.9 to 62.7) | 62.8 (62.2 to 63.3) |

The runner is above 60 in every run, and so is the full program. The 59.3 and 59.5 the
re-levelling round saw for the new model's full program came beside the 300-frame proof
on the GPU (load averages 4.9, 6 and 10 over 1, 5 and 15 minutes); they did not recur
here. The two sides differ by less than their ranges, so this shows no loss
and claims no gain.

**Gates** (measured; the tests and the gate ran near the tip, and `src`,
`tests`, `examples` and `tools/runner` are unchanged from the tests' tree to the tip).

- `cargo test --release --features testing`: 28 files ok, 0 failed. `tools/gate.sh` full:
  65 ok, 0 FAIL, 2 skip (the test step, run before it, and Nintendo's DMG boot ROM, which
  is absent). The ignored one-model test passes.
- `digest --check`: 8 checkpoints equal to `tools/digests.txt` (cpu `bebf5d93`, cpu-memory
  `836fb6de`, apu `d8d3a06e`, control `6e225675`, ppu `79706b58`, one-model `7b064d51`,
  ppu-stepper `4fdab521`, ppu-length `c42658f1`). `digest --gguf` on both exports:
  "manifest ok", 0 values differ; the uniform file's manifest says "llama.cpp b12070,
  unpatched (uniform layers)".
- The harness: 32 unit tests; the replay of the 300 frames with no model ends at frame 300
  `cfd31f3d` with 913,778 narrow passes, 2,896 first decodes for the windows, 2,820 second
  decodes, none over every cell, and no stop; the replays of Zelda DX to frame 7 and of a
  DMG boot likewise.

**The boot ROM under the narrow scope** (measured, unpatched): dmg-acid2 on a DMG from
power-on with SameBoy's DMG boot ROM mapped (`play --boot`), its own prefill of 12,809
cells, the first 4,000 passes: all equal (3,593 GBC CPU passes, every one over the narrow
scope, 150 decoded twice; 400 APU; 7 control), 1,197 write records, 0 error lines. These
passes are the boot ROM's VRAM clear loop, so window 0 is the boot ROM's lines throughout
and nothing here unmaps it.

**Not done, not verified.**

- The patched build's stream slices on this model: NaN on Metal, not traced and not
  fixed in this round. Traced and fixed since ("The slices' NaN on Metal", below, which
  has this model's patched rates and projection).
- Zelda DX in unpatched llama.cpp on this model. Its loans ran in the patched build with
  the whole model every pass.
- Yellow past frame 13 in the runtime on this model. The 300 frames ran in the replay.
- D1 ran on an earlier harness. The tip's harness differs in `Cache::audit` alone
 , which runs only under `STOCK_SCOPE=check`. The kits hold the tip's binary;
  the short legs after D1 used it.
- The engine's record in the kits was written by the harness built at the tests' tree
  with the first form of this round's harness. `play none` runs no harness check.
- Without the engine's record the harness is not held pass by pass (above).
- The sample draws random bytes, so it also draws states the machine does not reach,
  and it skips what the circuit refuses. It does not weigh the steps a game runs.
- No loans ran in unpatched llama.cpp on this model: Yellow lends none through frame 13.
- The stepper's stream has no pass in any of these runs.

## The slices' NaN on Metal: a run of `ffn_down` columns that starts on an odd weight (2026-11-24)

"Round 12" left the patched build's stream slices giving NaN on Metal with `7b064d51`.
The cause is in `stream-slices.patch`, not in the model or the file, and the patch now
avoids it. The patched build runs this model on its slices again: the two verify legs
and Zelda DX to frame 7 hold every pass, and the 300 frames project at about 6.2 hours
(inferred, below) against 19.0 unpatched.

**The cause** (measured unless marked). The control stream's seventh layer runs FFN units
2,311 to 2,790 of the layer: 480 units from an odd one. The patch took `ffn_down`'s
columns of a run as a view of the f16 matrix that starts at the run's first unit, so each
row of this view started 4,622 bytes into the layer's row, 2 bytes past a multiple of 4.
On Metal the product of that view with the layer's activations is wrong and finite:
largest value 1 where the Mac CPU backend's is 2^-7. Both backends hold the same
activations bit for bit (`ffn_swiglu-7`). Before that product the two dumps differ only
in the rotated queries and keys, which hold the same largest value and sum, and in four
nodes ahead of the embedding (`node_1`, `node_3`, `node_4` and a `SET_ROWS`). Every
residual, FFN and attention output of layers 0 to 6 has the same hash. One layer
later the wrong residual drives a gate to 2.4 million, the activations pass f16's 65,504,
and the next `ffn_down` product holds 154,752 values that are not finite. The harness saw
only that last step, as "largest score NaN".

Metal's matmul kernel reads a weight row in blocks of 16 (a `half4x4`) when the row's
width is a multiple of 32, and by single elements otherwise (`kernel_mul_mm`,
`FC_mul_mm_bc_inp`); the Metal Shading Language aligns a `half4x4` to 8 bytes. That the
block load is what misreads is inferred from the kernel's source and from these facts:
among the streams that decode more than 16 tokens, this is the only run of this model
that is a multiple of 32 wide and starts off a multiple of 4 units (the APU's 768 units
from unit 1,644 read right, and so do the runs of other widths from odd units), the old
model's control stream ran 480 units from unit 4,678 (an even one, 4 bytes past a
multiple of 8) and read right, and with the view moved onto a multiple of 4 units the
product equals the Mac CPU's bit for bit. Whether a start 4 bytes
past a multiple of 8 is safe on every Metal device is not known. The patch does not rest
on it.

How it was found: `STOCK_TENSORS=<file>` (new, in the shim) writes one line per tensor of
one decode, with its count of values that are not finite, its largest magnitude, its sum
and a hash. The control stream's first pass was dumped on Metal and on the Mac CPU
backend, and `tools/stock-llama/tensors-diff.sh` pairs the two files' tensors: the first
whose largest value differs, past the embedding, is that product.
`tools/stock-llama/runs.ts` then listed every stream's runs from the GGUF's keys.

**The fix** (`stream-slices.patch`, with `stored-runs.patch` rebased on it; no change to
the model, the file's tensors or the exporter). `mm_cols` is the one place a run of
columns meets a matmul, for `ffn_down` and for the attention output. It starts the view
on the 4 weights at or before the run's first unit, and pads the activations with that
many zero rows in front, which add exactly 0. Three asserts hold what the alignment
rests on: a sliced matrix's row is a multiple of 4 weights long (the exporter's
`--align 64` gives that for `ffn_down`), and so is a head. The stream's FFN is written
out (`ffn_cols`: up, gate, `ggml_swiglu_split`, then `mm_cols`) where it called
`build_ffn`, because the pad sits between the activation and `ffn_down`. The whole
model's path is as it was, and the unpatched build has none of this.

A/B in one build of llama.cpp, with a switch that left the view where it was (the
switch is not in the patch): off, NaN at the control stream's first pass; on, 40 passes
equal. On the build from the patches as committed, the control stream's first pass has
269 f32 tensors, and 247 of them have the Mac CPU backend's hash, the layer-7 product
and the logits among them. The 22 that differ are the four nodes ahead of the embedding
and the rotated queries and keys (equal in largest value and sum) with the views of
them, which the two backends list in different order; the attention outputs computed
from them have the same hash. The three patches apply in
order to a clean b12070 and give the tree that was built.

**The legs** (measured; patched llama.cpp b12070 with the new patches, the trimmed file
of `7b064d51`, `STOCK_STREAMS=require`, Metal, every pass under `STOCK_REFERENCE`; the
the final harness; load 5 to 11, other jobs on the
machine). The four sliced legs ran three times as the harness changed, with the same
counts each time; these are the last:

| Leg | Result |
|---|---|
| Yellow, 600 passes, `STOCK_STREAMS=check` | 585 sliced passes each decoded by the whole model too, 0 rows differ; all 600 equal the record |
| Yellow, 4,000 passes, the scopes and the split on | all equal; `diff` of the run's record against the engine's: 4,001 events equal, 313,445 tokens |
| Yellow, 3,500 passes, `STOCK_SCOPE=0 STOCK_SPLIT=0` | all equal; `diff`: 3,501 events equal, 274,258 tokens. GBC CPU 131.3 ms |
| Zelda DX frame 7 (`--warmup 6`), the saved cache, `STOCK_WRITES=check STOCK_READS=check` | exit 0, frame 7 `d043ab24`, all 21,110 passes equal (19,491 GBC CPU, 238 PPU, 845 APU, 416 control, 120 length), 20,990 of them sliced; 19,450 narrow passes, 626 decoded twice, 4 first for the windows; 216 PPU passes lent 438 lines, 876 loan records, 424 cells put back and compared, 0 differ, 0 unchecked; 3,126 write records both ways, 0 cells differ; 160 reader records both ways, 0 rows differ. The probe held 27 sliced passes to the whole model, 0 rows differ. The frame took 948 s against 1,936 s with the whole model every pass ("Round 12") |
| The Mac CPU backend (`STOCK_GPU_LAYERS=0`), Yellow's first 300 passes, `STOCK_SLICE_EVERY=0` | all 300 equal the record, 288 sliced |
| Unpatched llama.cpp, the uniform file, Yellow through frame 1 (an earlier harness) | exit 0, frame 1 `d043ab24`, all 13,167 passes equal the record (8,768 GBC CPU, 3,119 control, 840 APU, 288 PPU, 152 length), 0 error lines; GBC CPU 39.3 ms ("Round 12": 39.4) |
| Unpatched llama.cpp, the uniform file, Yellow's first 3,000 passes (the final harness) | all 3,000 equal the record, 0 error lines; GBC CPU 39.2 ms |

The unpatched path runs no code of the patch. The harness changed in two places it does
run (`shim_decode`'s dump hook, off without `STOCK_TENSORS`, and the probe below, which
needs a sliced pass), so the last two rows ran. The 32 harness tests pass.

**A wrong slice need not be NaN, and the harness now looks.** The first wrong tensor was
finite. NaN came a layer later, from values that happened to leave f16. A smaller misread
would give finite scores and could decode a wrong token. Under `STOCK_REFERENCE` every
pass is held to the engine's record, and `STOCK_STREAMS=check` holds every sliced pass to
the whole model, so both stop at it. `STOCK_STREAMS=require` without the record had
nothing but the NaN stop. Now, in every mode, the first sliced pass of each form (stream,
count of tokens) is also decoded by the whole model, and then one sliced pass in
`STOCK_SLICE_EVERY` (default 1,000): the kernels a pass runs follow from its stream and
its token count, so the probe covers each such form once. It compares only the pass's
output rows, so it sees a misread run of columns only when that run's activations are
nonzero in that pass and the error reaches a compared row. A row that differs stops the
run (`CHECK NOT HELD`). Seen to stop: with the unaligned view switched
back on, `require` stops at the control stream's first pass with 93 rows named, with the
record and without it. Its limit: a misread that leaves a form's first pass right and a
later one wrong waits for the sample. Its cost is not measurable here: 9 whole-model
decodes in 4,000 passes, and the GBC CPU pass is 11.2 to 11.4 ms with it, as it was
before it (11.3).

The probe found one more thing at once. On the Mac CPU backend the whole model, in the
patched build and in the unpatched one alike, gives NaN at the first GBC CPU pass (`STOCK_STREAMS=0`: "largest score NaN"), with
the old patches and the new alike, while the pass's slice decodes the record's tokens.
`STOCK_NANCHECK` names the first tensor that is not finite: `ffn_gate-1`, at unit 2,068,
which is the length stream's, at the pass's first token. So on that backend a unit of
another part overflows at a GBC CPU token (traced since: "The Mac CPU backend's NaN",
below; Metal's whole model is right there, 384 passes under `STOCK_STREAMS=check`). "Round 12" ran the Mac CPU
backend on slices only, so it did not meet this. The probe compares the slice with that
whole model and stops, rightly by its rule and wrongly for the slice. So
`STOCK_SLICE_EVERY=0` turns the probe off, the first pass of a form included, `play`
prints that it is off, and the Mac CPU backend's row above ran that way.

**Rates and the projection.** Three runs of Yellow's first 4,000 passes on the slices
(measured; load average 7.6 to 8.6 with other builds running, so contended;
all 4,000 passes equal the record in each run, 3,946 sliced; an earlier harness,
and a fourth run on the final harness gave 11.3, 22.3, 17.3, 11.7 and 8.1 ms):

| | ms each, median of 3 (range) | Whole model, "Round 12" | Count in 300 frames | Hours in 300 frames (inferred) |
|---|---:|---:|---:|---:|
| GBC CPU pass | 11.4 (11.2 to 11.4) | 21.3 | 925,752 | 2.93 |
| PPU pass | 22.7 (22.2 to 22.9) | 34.6 | 83,498 | 0.53 |
| APU pass | 17.4 (17.2 to 17.5) | 24.6 | 251,674 | 1.22 |
| Control pass | 11.7 (11.6 to 11.8) | 24.8 | 71,759 | 0.23 |
| Length pass (10 tokens, the whole model below the fence) | 8.4 (8.1 to 9.7) | 8.2 | 41,855 | 0.10 |
| Write record (26 in each run) | 19.3 (19.3 to 25.3) | | 119,435 | 0.64 |
| Reader record | 20.2 (inferred) | | 46,951 | 0.26 |
| Prefill | 756 s (Zelda DX's, "Round 12") | | | 0.21 |
| **Sum, with 0.72% for checkpoints** | | | | **6.2** |

The projection takes the first 4,000 passes' rates for all 300 frames and the proof of
record's counts. No reader record ran alone on the slices' build: 20.2 ms is the
unpatched 28.2 scaled by the write records' ratio, 19.3 to 27.0. Frames 2 to 13 cost the
unpatched GBC CPU pass 9% more than frame 1 (second decodes and shifts, "Round 12"), and
Zelda DX's 19,491 GBC CPU passes here took 12.2 ms each, which would make the sum 6.4
hours. Of the GBC CPU pass's 11.4 ms, 10.8 are inside `llama_decode`; what bounds it there was
not profiled. `45bfb056`'s patched projection
was 11.6 hours, with reader records over every cell (3.7 of them) and a GBC CPU pass of
18.0 ms.

**Not done.**

- No full patched run of this model. The longest is Zelda DX's 21,110 passes.
- The fence stays. With the views aligned and the fence set to 0 in a test build, the
  length stream's 12 passes of 10 tokens in Yellow's first 300 passes ran on their slice
  and equalled the whole model. That suggests the fence's first failure (six rows through
  a run from unit 2,047, an odd one) was this same load, but 12 passes do not show it.
- The whole model on the Mac CPU backend (above), in any build, was not traced here. It
  is since ("The Mac CPU backend's NaN", below).
- The kits (`kit-patched`) still hold the old build. The trimmed file's tensors are
  unchanged; its manifest names the old patches' sha1s, which a new export would update.
- The Metal kernel itself is not patched or reported upstream. A smaller reproducer than
  the control pass was not built.
- A tree with `stream-slices.patch` and without `stored-runs.patch` was compiled and
  never run: no file here has the stream keys without the stored runs.
- The stepper's stream has no sliced pass on this model: Yellow and Zelda DX's frame 7
  have none. Its runs go through the same `mm_cols`.
- The run that shows the probe stopping uses a build of llama.cpp with a switch that
  leaves the view unaligned. That switch is not in the patch, so the tree's own tools
  cannot repeat it.

## The Mac CPU backend's NaN: f16 sums at another part's positions

The slices' NaN fix, from the NaN probe. No change to the model, the files or the digests.
(Read "a default build" in this section as "a build with `-DGGML_BLAS=OFF`", which every
build here was. llama.cpp's own default on a Mac leaves BLAS on, and that build runs the
whole model on the Mac CPU backend: "Round 14".)
For anyone running the one model on llama.cpp's Mac CPU backend
(`STOCK_GPU_LAYERS=0`): a default build cannot run the whole model there, patched or
not. That holds for both models tried, `7b064d51` and `45bfb056`. A build flag makes it run. Metal is not affected, and neither are
the slices on either backend.

**What happens** (measured on this work, Yellow from each kit's prefilled state, the
first GBC CPU pass).

| Build, file | Result |
|---|---|
| Unpatched b12070, the uniform file of `7b064d51` | `ffn_gate-1` holds −inf at unit 2,068, token 0; every row of the pass is 65,804 NaN |
| Unpatched b12070, the uniform file of `45bfb056` (the proof of record's kit, read only) | `ffn_gate-0` holds inf at unit 2,585; every row of the pass is NaN |
| Patched, the trimmed file of `7b064d51`, `STOCK_STREAMS=require` | the probe's whole-model decode differs from the slice in 74 rows and the run stops. The previous round's log names the same tensor and unit as the first row ("The slices' NaN on Metal") |

So the fault is older than the re-levelling and has nothing to do with the patches. The
proof of record ran on Metal, where the whole model is right: the unpatched build's
first 3,000 passes equal the record again on this work's harness.

**The cause.** Layer 1's unit 2,068 is `len:vis14`, a length-stream unit that compares
the line with a sprite's row. Its gate reads channels 756 to 763 with weights −128 to
−16,384. At a length position those channels hold bits. At a GBC CPU position the same
channels hold the GBC CPU's wires, and channel 763 holds the nibble `D#lo`, which was 15.
`STOCK_NANCHECK` now prints the row behind a gate that is not finite: 4 nonzero products,
`2176·1 + 128·1 + 2048·1 − 16384·15`, exact sum −241,408. One product alone, −245,760,
is past f16's 65,504.

ggml's CPU backend multiplies an f16 matrix by converting the activations to f16 and
summing the products in f16 lanes when the compiler has ARM's FP16 vector arithmetic
(`vfmaq_f16`, in `ggml_vec_dot_f16` and in llamafile's `tinyBLAS`), which Apple silicon
has. A lane past 65,504 is inf, SiLU of −inf is NaN, and the unit's `up` of 0 does not
remove it. Metal sums in f32, where the gate is −241,408, SiLU gives −0 and `up` gives 0.

Three measurements hold this, none inferred:

- `tools/stock-llama/f16-sums.c`, 40 lines against a ggml build: one row of 64 weights
  of 16,352 times inputs of 1 (sum 1,046,528) gives inf with f16 weights and the exact
  sum with f32 weights, at 1 token and at 16. A row that sums to 0 gives NaN.
- The same unpatched source built without FP16 vector arithmetic
  (`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.4-a+nofp16`) gives the exact sums in
  that program, and the whole model on the Mac CPU backend then decodes Yellow's first
  400 passes equal to the record (254 GBC CPU, 99 control, 26 APU, 13 length, 8 PPU).
  The two builds differ by that architecture flag (it also replaces `-mcpu=native`), and
  the program shows that the accumulator is what differs for this product.
- The patched tree built the same way runs the slices with the probe on: 400 passes
  equal, 7 sliced passes also decoded by the whole model, 0 rows differ. With the
  default build the probe stops at the first GBC CPU pass (74 rows), as before.

This is the limit `docs/hardware-shaped.md` §15 states ("What stock runtimes must
know"): a runtime that runs every unit at every position needs f32 sums.
`Model::check_f16_acc` holds each part's rows over that part's own ranges only. What was
missing was the knowledge that llama.cpp's default Mac CPU build is such a runtime.

**How many rows** (`cargo run --release --example f16foreign`, static, over each
channel's declared ranges at each other part's positions; "can pass", not "does").
2,027 of the 76,143 part-owned gate rows can pass 65,504 at another part's positions,
in 23 of the 25 layers. 1,980 of them can at GBC CPU positions, where channels hold
nibbles up to 15 (worst 524,160). 273 can at some other part's positions, where the
channels hold bits that are not the one-hot groups the row's own part has there (worst
105,344). None of the 5,421 part-owned query rows can, after the GGUF's shift. `-v`
names every row, and `--unit <layer>:<unit>` prints a row's channels with each part's
occupants.

The census covers matmul rows only. The gate is the first tensor that overflows, and
`STOCK_NANCHECK` stops there, so nothing later was observed on the default build. One
later place is known from the source and not measured: with an f16 KV cache the CPU
backend's flash attention dots each query with its key through the same f16 lanes
(`kq_vec_dot`), and an overlay head's null term is 9 dims of up to 30,720 against a
null key of 4 (`docs/hardware-shaped.md` §15.4). So a remedy that only keeps the gates
finite is not shown to make the pass decode. The build without FP16 vector arithmetic
sums everything in f32, and that is the one remedy seen to work.

**What changed on this work** (harness and patch only).

| Change | Seen |
|---|---|
| `STOCK_NANCHECK` prints the products behind a gate that is not finite | the line above |
| The probe's whole-model decode must report that it ran the whole model (`SLICE PROBE NOT WHOLE`) | 9 probes pass it on Metal; the stop itself is not reachable today and was not seen to fire |
| `stream-slices.patch` asserts the GGUF's alignment is a multiple of 8 | the trimmed file loads (alignment 32); the assert was not seen to fire |
| `STOCK_TENSORS` beside `STOCK_NANCHECK` runs both; a file that cannot be written stops the run; `first` is `-` when every value is finite | one run wrote 784 tensor lines and both `NANCHECK` lines; `STOCK_TENSORS: cannot write …`, exit 1 |
| A malformed `STOCK_SLICE_EVERY` is a plain message, exit 2 | `STOCK_SLICE_EVERY=often is not a count of passes` |

Legs on this harness (measured, load 4 to 8, the rates contended): patched Metal,
Yellow's first 4,000 passes on the slices, all equal the record, 9 probes, 0 rows
differ, GBC CPU 11.2 ms. Unpatched Metal, 3,000 passes, all equal, GBC CPU 39.3 ms. The
harness tests pass (32). The three patches apply in order to a clean b12070 with
`git apply --check`, and `layer-shapes.patch` is unchanged.

**`STOCK_SLICE_EVERY=0`** is still needed on a default Mac CPU build, because the probe
compares the slice with a whole model that build cannot run. It is not needed on a build
without FP16 vector arithmetic.

**Not done.**

- No fix in the model. Bounding what a unit reads at another part's positions moves the
  one model's digest, and the design and its cost were handed back first.
- The census is static. Which of the 2,027 rows overflow on real passes was not counted;
  the runs name one unit each.
- `45bfb056` was not run on a build without FP16 vector arithmetic.
- The Mac CPU rates of the two builds were not compared. The build without FP16 vector
  arithmetic ran a whole-model GBC CPU pass in 488.7 ms and a sliced one in 68.0 ms
  (one contended run each).
- x86 builds of ggml were not run. By the source, AVX builds widen f16 to f32 before
  they sum.

## Round 13, step 0: what an unpatched llama.cpp pass costs on Metal, by model shape (2026-11-24)

Round 2 ranked the unbuilt levers (heads, layers, vocabulary) by a token's multiplies, and
the multiply count had already mispredicted Metal once. This section measures the pass
itself at each shape before anything is designed. No model changed here: all eight
digests are round 12's.

**The tools.** `examples/export -- synth <dir> <hidden> <layers> <ffn> <heads> <vocab>`
writes a uniform Llama GGUF of that shape with every matrix zero, through the uniform
export's own writer (one KV head, f16 matrices, the same metadata). `stock-llama shapes
<model.gguf> --shapes NAME=TOKENS:ATTENDING:ROWS:CELLS,...` decodes groups of that shape
through the pass's own path (`Cpp::scope`, `Runtime::scoped_pass`): TOKENS tokens at one
position, ATTENDING of them attending a scoped sequence of CELLS cells, ROWS of those
reaching the LM head. The shapes decode in turn, 5 discarded rounds and then 30 or 40, and
each prints its median. Settings are the proof's: unpatched b12070, Metal, GPU layers 99,
f16 KV, flash attention, `--lanes 255`.

**The streams' shapes** (measured, `play replay` of Yellow's first 3 frames, which now
prints them; scope cells from the frame-13 run's log):

| Stream | Tokens | Attending the cells | Logit rows | Scope cells |
|---|---:|---:|---:|---:|
| GBC CPU | 100 | 73 | 73 | about 600 (472 and the lines named lately) |
| APU | 139 | 71 | 71 | 1 |
| control | 150 | 93 | 93 | 1 |
| PPU | 166 | 163 | 163 | 2,086 |
| length | 10 | 5 | 5 | 22 |

**The bench is the pass** (measured, one GPU hold of 3.5 minutes, load 5 to 7; ms a pass,
median of 40).

| | GBC CPU | APU | control | PPU | length |
|---|---:|---:|---:|---:|---:|
| The frame-13 run (`play`, inside the runtime's decode; "Round 12") | 41.1 (37.6 in its first 704 passes) | 42.2 | 41.8 | 75.1 | 10.9 |
| `shapes`, the real file, that run's saved cache of 143,881 cells | 38.2 | 40.5 | 41.0 | 73.7 | 10.3 |
| `shapes`, the real file, 4,096 cells decoded here | 37.9 | 40.5 | 40.8 | 73.9 | 10.2 |
| `shapes`, the synthetic baseline (1,920, 25, 3,456, 41, 65,804), 4,096 cells | 37.6 | 39.8 | 40.3 | 73.2 | 10.0 |

The synthetic file runs within 2% of the real one, and a cache of 4,096 cells within 2% of
the real cache, so the grid below uses synthetic files over 4,096 cells. The frame-13
run's GBC CPU rate is 8% above the bench because its passes include second decodes and
the decodes that shift a new cell (47.5 ms each, "Round 12"); its first 704 passes ran at
the bench's rate.

**What a pass's time is made of** (measured on the real file and cache; every token
attending, 2 cells): 10 tokens 11.0 ms, 50 tokens 22.7, 100 tokens 35.5, 150 tokens 43.9,
200 tokens 58.6. So a pass pays about 8 ms whatever it holds and about 0.25 ms a token.
100 tokens with one logit row take 31.7 ms against 35.5 with 100 rows: the LM head is
0.039 ms a row, 2.8 ms of a GBC CPU pass. 100 attending tokens over 2,086 cells take 50.1
ms against 35.5 over 2: the PPU's cells cost about as much as its layers' share of them.

**The grid** (measured, one GPU hold of 6.3 minutes, load 9 to 15 from other jobs' CPU
work, and a second of 2.8 minutes at load 5 for the points the build below can reach,
whose base ran 37.56 ms against the first hold's 37.58; each point generated, timed and
deleted, the list run forward and then backward; ms a pass, the mean of the two runs'
medians of 30, which differ by at most 1.5% but for h16's 2.5%). The five streams' columns
are their own shapes; "100 tokens" and "150 tokens" are groups of that many tokens, all
attending 2 cells and all with a logit row.

| Point (hidden, layers, FFN, heads, vocabulary) | GBC CPU | APU | control | PPU | length | 100 tokens | 150 tokens | GBC CPU hours of 300 frames | 300 frames, hours | Against base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base (1,920, 25, 3,456, 41, 65,804) | 37.58 | 39.97 | 40.40 | 73.50 | 9.91 | 34.70 | 42.81 | 11.0 | 19.0 | 100% |
| heads 29 | 33.27 | 36.21 | 36.75 | 63.05 | 9.03 | 31.55 | 39.19 | 9.7 | 16.9 | 89% |
| heads 20 | 30.88 | 34.05 | 34.14 | 55.52 | 8.36 | 29.81 | 36.52 | 9.0 | 15.6 | 82% |
| heads 16 | 30.33 | 32.72 | 33.01 | 51.14 | 8.12 | 29.42 | 35.42 | 8.9 | 15.2 | 80% |
| layers 17 | 27.08 | 28.69 | 29.03 | 52.75 | 7.38 | 25.52 | 31.50 | 7.9 | 13.7 | 72% |
| layers 17, FFN 4,544 | 30.14 | 32.92 | 33.45 | 57.77 | 8.07 | 28.59 | 35.90 | 8.8 | 15.3 | 80% |
| vocabulary 24,684 | 35.56 | 37.81 | 38.23 | 69.33 | 9.33 | 32.05 | 39.39 | 10.4 | 17.9 | 94% |
| hidden 1,664 | 35.39 | 37.06 | 37.61 | 67.81 | 9.04 | 32.37 | 39.75 | 10.4 | 17.7 | 93% |
| hidden 2,176 | 41.88 | 44.61 | 45.23 | 79.46 | 10.57 | 39.02 | 47.86 | 12.3 | 21.0 | 110% |
| FFN 3,072 | 35.47 | 37.88 | 38.44 | 70.51 | 9.55 | 32.52 | 40.80 | 10.4 | 18.0 | 95% |
| heads 29, vocabulary 24,684 | 31.26 | 34.17 | 34.62 | 58.79 | 8.52 | 28.88 | 35.81 | 9.2 | 15.8 | 83% |
| heads 20, vocabulary 24,684 | 28.89 | 32.11 | 32.02 | 51.72 | 7.80 | 27.21 | 33.17 | 8.5 | 14.6 | 77% |
| all (1,920, 17, 4,544, 16, 24,684) | 23.12 | 25.87 | 26.09 | 38.02 | 6.33 | 22.21 | 27.31 | 6.8 | 11.6 | 61% |
| floor (1,920, 25, 64, 1, 65,804) | 12.29 | 11.81 | 12.06 | 18.55 | 3.92 | 11.87 | 14.24 | 3.6 | 6.1 | 32% |
| heads 26 (second hold) | 32.66 | 35.59 | 35.98 | 59.78 | 8.70 | 31.14 | 38.42 | 9.6 | 16.5 | 87% |
| heads 26, vocabulary 27,511 (second hold) | 31.14 | 34.12 | 34.44 | 56.55 | 8.52 | 29.03 | 35.62 | 9.1 | 15.7 | 83% |
| heads 25, vocabulary 27,511 (second hold) | 30.94 | 33.91 | 34.16 | 55.67 | 8.43 | 28.86 | 35.41 | 9.1 | 15.6 | 82% |
| layers 17, FFN 5,056 (second hold) | 31.70 | 34.62 | 35.09 | 60.16 | 8.45 | 30.20 | 37.68 | 9.3 | 16.1 | 85% |
| layers 17, FFN 5,056, heads 26, vocabulary 27,511 (second hold) | 26.18 | 29.52 | 29.75 | 46.28 | 6.90 | 24.95 | 31.14 | 7.7 | 13.2 | 70% |

The hours are inferred: each stream's rate in the frame-13 run times the point's ratio to
the base at that stream's shape, times the proof of record's counts (925,752, 251,674,
71,759, 83,498 and 41,855 passes). The write records (119,435 at 27.0 ms), the reader
records (46,951 at 28.2 ms) and the prefill (2,581 s) are scaled by the point's ratio at 10
tokens, since each is a decode of a short group; that scaling is a guess, and those lines
are 2.0 of the 19.0 hours. Checkpoints add 0.72% as in "Round 12".

**What the table says.**

- Metal follows the layers' work more closely than round 1's Metal comparison suggested.
  Heads 41 to 20 cut 17.8% of a GBC CPU pass where the multiplies say 17.1%; 17 layers of
  FFN 4,544 cut 19.8% where they say 17.9%; the vocabulary cuts 5.4% where they say
  about 7%. The floor point shows why the earlier ratio missed: a third of a pass (12.3 of
  37.6 ms) is not the FFN's or the heads' multiplies. It is the 25 layers' fixed work, the
  embedding, the norms and the LM head.
- Heads pay more for the PPU than the multiplies say: its 163 rows attend 2,086 cells
  with every head, so 41 to 20 heads cut 24.5% of its pass.
- Hidden is a strong lever in both directions (1,664: 6% less, 2,176: 12% more), and it
  is not available: the GBC CPU's slice sets 1,920 through the norm's bound.
- The order by what a lever pays against what it needs: shared input slots first (the
  heads and the vocabulary in one redesign: 83% at the 26 heads and 27,511 tokens the
  build below reaches, 77% at heads 20 and 24,684), then the layers (80% at the bound's
  width of 4,544 and 85% at 5,056, the 12% over its bound the packer ends at today, with
  the APU's circuit reworked; 70% with the shared slots too), then the FFN width (95% at
  the packer's bound of 3,072).

## Round 13: the parts share the input slots

One change to the one model, for the unpatched runtime: layer 0 holds 24 gathers where it
held 41, a layer holds at most 26 heads, and the vocabulary is 27,511 tokens where it was
65,804. Step 0 above puts those two levers first.

**The design.** Each part's inputs had their own slots: 239 input slots, 257 tokens each,
and a gather per group of each part's slots (38 groups and the 3 record gathers). A pass
decodes one part, so no two parts ever need a slot at once. Now the parts hold the same
slots:

- A part's k-th input without nibble wires holds input slot k (`Circuit::concat`,
  `Circuit::parts`). The stepper is the widest, 79 slots.
- An input with nibble wires holds a slot no other part holds, after those: the GBC CPU's
  11, slots 79 to 89. A gather writes a nibble wire into a channel no clearing unit
  empties (`Circuit::pooled_channels` frees only wires of range 0/1 or −1/0), so every
  other holder of such a slot would keep two foreign channels for good. Shared by index
  that is 22 channels in the stepper (`examples/inputs`), 1,398 with its 1,376 wires,
  and hidden 1,920 has 1,394 beside the 526 shared channels. So the vocabulary is 90
  slots' 27,511 tokens and not round 2's 24,684.
- The input slots are grouped where the set of holders changes (`Model::groups`): 21
  groups, each held whole or not at all by every part, so a stream's pass still carries
  its own inputs and nothing else, and every gather of a group it holds weighs each slot
  1/len.

**The invariants, and what holds each.**

| Invariant | Held by |
|---|---|
| Every holder of slot s reads bit i in channel 8·s + i of its own wires, the channel the one gather writes | `Circuit::overlay_channels` pins the wires; `compile_with` asserts every other holder's wire has the first holder's channel |
| A holder that reads no node from a slot's bit still counts the bit's channel as occupied | `overlay_channels` pins every input bit wire, read or not. (This row first named an assert that a bit is "read by every holder or by none". It compared `Circuit::unread`, which marks head line bits only, so it could not fire; "Round 14" removed it. Of the 1,912 input bit wires 109 have no reader and 95 slot bits are read by some holders only, the review's count on the compiled checkpoint; all 1,912 sit at the first holder's channel.) |
| A slot a part does not hold is byte 0 in its stream, so the gather writes 0 and the slot's channels are free for the part's other wires | the stream's view holds every position it does not own at byte 0 or the mask; `Model::stream` asserts the held tokens' layer-0 values are 0 |
| A stream's slice keeps the gathers of the groups its part holds, and no other | `Overlay::holders`, read by `Model::slice` |
| No layer holds more heads than the cap, layer 0 included | `one_model::HEADS` (26) is the packer's cap, and `placed` asserts layer 0's gathers fit it |
| Only the one model moves | no wire is pinned outside an overlay, and a one-part circuit's groups are as they were; `digest --check` |

The head cap was layer 0's gathers less the copy heads (39). It is now a constant of the
placement: `examples/room -- --try 3480 <heads> ppu=20,stp=24,len=20 1371` ends within its
caps at 26 heads and 2 over at 25, with width 3,456 and hidden 1,920 unchanged (measured,
from the circuits alone).

**What it compiles to** (measured: `examples/shape --one`, `examples/export`, `examples/digest`).
Only the one model moves, `7b064d51` to `ff86dde7`; `digest --check` holds the other seven
to `tools/digests.txt`.

| | `7b064d51` (round 12) | `ff86dde7` (this work) |
|---|---:|---:|
| Hidden, layers, widest layer's FFN | 1,920, 25, 3,456 | the same |
| Widest layer's heads (the uniform heads) | 41 | 26 |
| Layer 0's gathers | 41 (38 groups and the 3 record gathers) | 24 (21 and 3) |
| Input slots, sequence positions | 239, 769 | 90, 620 |
| Vocabulary | 65,804 | 27,511 |
| Uniform file: values, bytes | 1,008,497,280, 2,023,564,128 | 769,292,160, 1,544,154,592 |
| Layer-shaped file: values, bytes | 408,489,344, 823,553,728 | 382,008,896, 769,593,632 |
| Nonzeros | 1.85 M | 1.39 M |
| Channels the parts share | 543 | 526 (17 fewer group identities) |
| Widest live wires: PPU, GBC CPU, APU, control, length, stepper | 1,377, 1,222, 1,079, 1,119, 560, 1,376 | the same |
| Each stream's slice (hidden, layers): PPU, GBC CPU, APU, control, length, stepper | (1,920, 21), (1,920, 11), (1,664, 25), (1,664, 17), (1,152, 21), (1,920, 25) | the same but the length's (1,088, 21) |
| Input gathers a stream holds: PPU, GBC CPU, APU, control, length, stepper | 2, 5, 9, 8, 2, 12 | 2, 9, 15, 12, 3, 18 |
| Heads by layer (`export --layer-shapes`) | 41, 23, 29, 1, 20, 1, 27, 12, 4, 0, 0, 10, 22, 1, 3, 12, 11, 0, 0, 3, 0, 3, 3, 0, 0 (`Model::census`) | 24, 26, 26, 1, 20, 1, 26, 12, 4, 1, 1, 10, 22, 1, 3, 12, 11, 1, 1, 3, 1, 3, 3, 1, 1 (the file stores one head in a layer that has none) |

Hidden has 17 channels to spare now (the scanline PPU's 1,377 wires and the 526 shared
are 1,903), where it had none. A stream holds more gathers than before, since a group
ends wherever the set of holders changes: the GBC CPU's 27 inputs are 9 groups, not 5.
The wiring's sha1 is `81ccc726` on both, so the wiring table carries over; a hand-off
snapshot names the tensors' digest and is made again.

No engine code changed. The engine plans each stream from its own network
(`Model::stream`) and maps its slots through `StreamSpec::at`, which now gives every
part's inputs the shared slots; the runner and the harness read the same specs from the
GGUF.

**The guards, seen to fail** (measured, `src/level.rs`'s
`a_relevelled_overlay_decodes_the_circuits_bytes`, two parts that hold one input slot).
With the second holder's wires pinned one bit off, the compiler stops: "wide:X: bit 0
sits in another channel than slot 0's gather writes". With that assert also gone, the
overlay decodes another byte than the part's circuit at input `0x07` (8 against 4). The
test itself changed with the design: it gave each part its own input slot and held the
other at 0, and it now gives both parts the one slot.

**Tests whose premise moved.**

- `the_cpu_stream_keeps_records_whose_tokens_are_above_65535` asserted that the
  vocabulary passes 2^16, so that record tokens above 65,535 are exercised. At 27,511
  tokens no token is that large. The test keeps its records and oracles under the name
  `the_cpu_stream_keeps_records_through_the_oracles`, and nothing exercises a token
  above 65,535 any more.
- `a_placement_that_widens_hidden_is_refused` stood in for one added wire with a
  placement that keeps 2 more wires live. With 17 channels to spare that compiles 1,920,
  so the stand-in keeps 18 more (the stepper 1,402 wires, hidden 1,984, refused by name).
- `every_stream_decodes_only_its_own_slots` counted, per part, the gathers that fall
  back on the GBC CPU's masks. It now counts for every part the gathers its group holds
  and the ones that fall back (2 and 19, 9 and 12, 15 and 6, 12 and 9, 3 and 18, 18 and
  3), and that none is held in part.
- `every_stream_reads_one_cell_per_head_and_stays_integral` draws 200 random APU states.
  Drawn in the merged circuit's slot order instead of the APU's own input order, the
  merged circuit's evaluation stopped at `apu:env2_e.1 = -1 outside [0, 1]` (measured).
  With the bytes drawn in the APU's own order again, the bytes the test drew before, it
  passes, so the stop follows the draw and not the shared slots; that the APU's own
  circuit refuses that state is inferred, since the draw was not replayed on it. The
  test's comment says the APU "takes any input bytes", which this draw contradicts.

**Unpatched llama.cpp on Metal, Yellow frame 1** (measured; `PROOF_PURE=1 FRAMES=1
PROOF_CHUNK_SECS=2700` through `proof.sh`, the uniform file of
`ff86dde7`, its own prefill, every pass under `STOCK_REFERENCE`, load 4 to 6). Before the
GPU run the harness's replay of the engine's new record ran 2 frames clean, with no
model.

- Exit 0, 0 error lines, frame 1 `d043ab24` (the engine's; that hash is the blank screen's,
  which the record holds for frames 1 to 81, so the evidence is the equal passes), all 13,167 passes equal the
  engine's record: 8,768 GBC CPU, 288 PPU, 840 APU, 3,119 control and 152 length passes,
  109 write records (85 in the GBC CPU's stream and 8 each in the PPU's, the length's and
  the stepper's, the count lines of `proof.log`) and 1 reader record, none off the mirror.
- One chunk held the GPU lock 37 min 51 s, the prefill's 1,754 s in it (`7b064d51`'s took
  2,581 s for the same 143,881 cells).

The same 13,167 passes on both models, each in its own run (`7b064d51`'s from the
frame-13 run's log at frame 1, "Round 12"; ms each):

| | `7b064d51` | `ff86dde7` | Ratio | Step 0's ratio for heads 26 and 27,511 tokens |
|---|---:|---:|---:|---:|
| GBC CPU pass | 38.2 | 32.2 | 0.843 | 0.829 |
| PPU pass | 76.8 | 59.1 | 0.770 | 0.770 |
| APU pass | 42.4 | 36.8 | 0.868 | 0.857 |
| Control pass | 43.1 | 37.1 | 0.861 | 0.854 |
| Length pass | 11.5 | 9.7 | 0.843 | 0.860 |
| Every pass (513.4 s against 440.8 s) | | | 0.859 | |
| Write record (109) | 28.2 | 22.8 | 0.810 | |
| Reader record (1) | 21.9 | 21.0 | 0.96 | |
| Prefill | 2,581 s | 1,754 s | 0.680 | |

The synthetic grid predicted each stream's ratio within 1.7 points. The two runs were
not interleaved: they ran 11 hours apart on the same machine at loads of 3 to 6.

**The 300 frames, projected (inferred).** "Round 12"'s rates through frame 13 times the
ratios above, times the proof of record's counts: 9.28 hours of GBC CPU passes (36.1 ms),
1.39 of PPU, 2.64 of APU, 0.74 of control, 0.11 of length, 0.73 of write records, 0.35
of reader records and 0.49 of prefill, 15.8 hours with the checkpoints' 0.72%, against
19.0 for `7b064d51`: 83%. Frame 1 holds no loan, no bank switch, no OAM DMA and one
reader record; "Round 12"'s frames 2 to 13 did, on the old model, and nothing past frame
1 has run in a llama.cpp runtime on this one.

**The runner and the full program** (measured; text workload, 600 frames from the
frame-2,900 hand-off, old round 12 with `7b064d51` against this work with
`ff86dde7`, each from its own trimmed GGUF and its own hand-off snapshot, three ABBA rounds
in one heavy ticket, load 4 to 7; every run `870aa3fd` with 0 dense fallbacks; medians of
six with ranges):

| Text, 600 frames | `7b064d51` (fps) | `ff86dde7` (fps) |
|---|---|---|
| Runner (`tools/runner`) | 63.0 (62.0 to 63.5) | 63.8 (63.3 to 64.3) |
| Full program, one model from the GGUF | 62.0 (61.8 to 62.3) | 61.7 (60.9 to 62.2) |

Every run of both programs is above 60. The runner's new side was the faster in each of
its six pairs, by 0.3 to 1.3 fps, and its ranges overlap the old side's by 0.2. The full
program's new side was the slower in five of six pairs, by 0.3 to 0.9, with a median 0.3
lower inside both ranges. So the runner shows no loss, and the full program at most a
small one that this bench does not separate from its spread. The stages
of a fast pass are the same on both sides to 0.05 µs (GBC CPU: layer 0 0.11 µs, shared
layers 2.04 and 2.05 µs; APU layer 0 0.63 and 0.66 µs, control 1.34 and 1.37), so the
extra gathers a stream holds cost the engine nothing measurable.

**Gates** (measured, heavy lane, one at a time; the tests and the gate ran near the tip, and from the tests' tree
to the tip `src`, `tests`, `tools` and `Cargo.toml` are unchanged
and `examples/room.rs` gained one printed line).

- `cargo build --release --all-targets --features testing`, the harness and the runner: no
  warnings.
- `cargo test --release --features testing`: 28 files ok, 0 failed, 0 lines with
  `panicked` (26 min at load 6 to 14). `tests/one_model.rs`: 8 passed (513 s), every
  part's slice equal to the whole checkpoint at its positions among them; the ignored
  `the_null_token_decodes_the_null_cell`: ok (61 s).
- `examples/digest -- --check tools/digests.txt`: 8 checkpoints equal, with the one
  model's line moved to `ff86dde7`; before the line moved it failed on the one model alone.
- `digest --gguf` on both export forms: "manifest ok", 0 values differ (the uniform file
  769,194,240 matrix values, the trimmed 381,910,976); the uniform file's manifest says
  "llama.cpp b12070, unpatched (uniform layers)".
- The harness's 32 unit tests and the runner's 24, on the new GGUF.
- `tools/gate.sh` full (`GATE_NO_TEST=1 GGUF=<the tip's trimmed export>`, one heavy ticket
  of 1 h 40 min): 65 ok, 0 FAIL, 2 skips (the test suite, run before it, and Nintendo's
  DMG boot ROM, which is not in the repository). Among them DMG 8 `6872b003` and CGB 300
  `cfd31f3d`; the four games with all five components in weights (`cfd31f3d`,
  `ccfe7ad3`, `f10b2b4a`, `ea7f86ca`); Yellow 300 as one model, and again with the step
  oracle on every stream; Mealybug `m3_bgp_change` with the step oracle on all six
  streams; both acid2 images; and the runner's record-equal legs on the tip's GGUF.
- The engine's record of Yellow to frame 300 on the new model (`play none`, no GPU):
  frame 300 `cfd31f3d`.

**What was not built, and the bound each would reach** (the hours are inferred from step
0's grid as above; the grid's ratios held within 1.7 points on the lever that was built).

| Lever | Bound | What it needs | Why not built |
|---|---|---|---|
| Layers 25 to 17 | 85% alone at FFN 5,056 (the packer's 12% over the bound of 4,499), 80% at 4,544; with the shared slots 70%, 13.2 hours | the APU's critical path, one chain of 24 levels, cut to 16 | the APU's circuit and its digest, and every APU gate again; see the probe below |
| Layers 25 to 22 | 94% of this work's model at FFN 3,968 (measured below) | the APU at 21 levels, which the probe reached, and a placement at 22 layers | the packer does not hold one: see the probe below |
| Heads 26 to 24 | about 1.3% (heads 25 against 26 measured 0.6%) | a packer that holds the cap: layer 0's 24 gathers are the floor | `room --try` ends 2 heads over at 25 and 20 over at 24 (two 21-head buses at layer 1). No head's latest level forces it there (`room` now prints each part's heads by latest level: the PPU's first 21 can sit in layers 1 to 8, the length's in 1 to 7, the stepper's in 1 to 17), so the limit is the packer's top-down order, not the circuit |
| Vocabulary 27,511 to 24,684 | about 0.4% of a GBC CPU pass (2,827 fewer scores a row) | a clearing unit for a nibble channel, or 4 more channels: the stepper would hold 22 foreign nibble channels and hidden has 18 of them | costs more than it returns |
| FFN 3,456 to 3,072 | 95% | the GBC CPU, the APU and control re-levelled too | with control free the packer ends 39 heads over (`room --try 3288 26 ppu=20,stp=24,len=20,ctl=24 1371`: width 3,392, 65 heads at a layer) |
| Hidden 1,920 to 1,664 | 93% | the GBC CPU's norm bound | its slice sets 1,920 |

**The layers lever, probed and stopped** (the APU depth probe, one commit on this
round's tip, not gated and not merged).

Step 0 had no point between 17 and 25 layers, so a third GPU hold (2.2 minutes, load 4 to
7, the same method) measured them from this work's shape, each at about 10% over the
units' average a layer as today's 3,456 is:

| Point (heads 26, vocabulary 27,511, hidden 1,920) | GBC CPU | APU | control | PPU | length | Passes of 300 frames against base |
|---|---:|---:|---:|---:|---:|---:|
| base: 25 layers, FFN 3,456 | 31.30 | 34.28 | 34.57 | 56.45 | 8.53 | 100% |
| 24 layers, FFN 3,648 | 30.58 | 33.69 | 33.98 | 55.12 | 8.16 | 97.8% |
| 23 layers, FFN 3,776 | 29.95 | 33.47 | 33.77 | 54.20 | 8.12 | 96.2% |
| 22 layers, FFN 3,968 | 29.38 | 32.95 | 33.08 | 52.82 | 7.87 | 94.4% |
| 22 layers, FFN 3,456 (the width cannot stay: 76,802 units need 3,650 a layer) | 27.54 | 30.12 | 30.49 | 49.87 | 7.43 | 88.0% |

A layer is worth about 2% at the width it costs. Three come off the end of the APU's path
without touching what it computes: the mix read each channel's sample after three
single-level selections (the cycles' `sel`, the event's merge, the playing gate), and it
can read the cases themselves as literals of its AND terms, one wire per case
(`apu_circuit::circuit::mix`). Measured on the probe: `room -- --path apu` gives 21
levels where it gave 24, and the APU's 10 lib tests pass, the random register traffic
against the reference on DMG and CGB among them. The mix's sums grow from 46 to 142
units a side and the APU's widest live wires from 1,079 to 1,205.

What stopped it is the placement. At 22 layers `room -- --try <cap> 26
ppu=20,stp=21,len=20 1371` ends 18 to 39 heads over at every cap from 3,640 to 3,960 (44
to 65 heads at layer 1), with hidden 1,984 at most of them and widths of 4,160 to 6,976.
A pre-pass that gives every head group a layer before the nodes are packed (tried at
five aims between a group's earliest and latest layer, then discarded) holds the 26
heads from the middle aim on, and then the nodes have no slack left: 1,800 to 3,700
units over, hidden 2,048 to 2,304. So 22 layers need a packer that places heads and
nodes together, which is the same packer the 24-head cap needs. With it, the probe's
commit is the APU's part of the lever; without it the model would cost more than three
layers return. No checkpoint was compiled on the probe branch and no digest recorded.

**What in the brief and in round 2 turned out wrong.**

- "Shared gathers need the layout, the slices, the engine's layer 0 and the runner."
  They needed the circuit's concat, the groups, a pin in the channel plan and the
  slices' head lists. No engine, runner or harness code changed: all three place a
  stream's tokens through `StreamSpec::at`.
- "The risk is hidden: aligned input channels a part does not read are live in it."
  Every input bit wire is pinned to its slot's channel in every holder, read or not, so
  no bit channel is foreign anywhere. This item first said every part reads all 8 bits of
  every slot it holds, which is false: 109 of the 1,912 input bit wires have no reader
  (`examples/inputs` counts every input bit as read, because it asks `Circuit::unread`,
  which marks head line bits only; "Round 14"). Identity channels were already one per group, so 17
  fewer groups freed 17 channels, and hidden gained room where round 2 had none.
- "Vocabulary 24,684." The GBC CPU's 22 nibble wires do not fit beside the stepper's
  wires, so the vocabulary is 27,511.
- "The packer does not hold a head cap below 29." It holds 26 at the same width and
  hidden.
- "The multiply count mispredicts Metal." For these levers the grid's ratios are within
  about 2 points of the multiplies' ratios. What the multiplies miss is the third of a
  pass that is none of the FFN's or the heads' work, which is why whole-model ratios
  were off before.
- "Validate the baseline against 42.8 ms." That figure holds second decodes and the
  decodes that shift a cell. A plain GBC CPU pass is 37.6 to 38.2 ms on `7b064d51`.
- A pass's time is not proportional to its tokens: about 8 ms whatever it holds, then
  about 0.25 ms a token.

**For the patched path** (not run here; another branch repairs its stream slices). The
trimmed file of `ff86dde7` stores other head lists at layer 0 than `7b064d51`'s: a
stream's slice keeps the gathers of the groups its part holds (`Overlay::holders`), 2 to
18 of the 21 and the 3 record gathers. `gbcformer.stream.<part>.heads` carries them as
before, so the patches read them unchanged; whether they run is not known, since
`ff86dde7` has never run in the patched build. `tools/stock-llama/src/play.rs` gained one
line in `Stream::decode` and four counters (the shapes each stream prints), which may
conflict with that branch's edits there.

## Round 14: the shared-slots model through every llama.cpp path, with the slice fix and the review folded in (2026-11-25)

Since this section was written the 300 frames ran on `ff86dde7` in the default build, the same
day ("The proof of record on `ff86dde7`", below). Where this section says the 300 frames
were not started, or says "the proof of record" for the BLAS-off run on `45bfb056`, read it
as of that morning. This section is unchanged below this note.

Round 14 merges round 13 (the one model `ff86dde7`, whose parts share the input
slots) with the slices' NaN fix (the slices' alignment fix, the slice probe, the Mac CPU backend's
build option). "Round 13" left `ff86dde7` with one frame in unpatched llama.cpp and no
run in the patched build. It has now run in unpatched llama.cpp on Metal through frame 13
of Yellow, in the patched build on its slices with the three patches unchanged, and on
the Mac CPU backend with the build option. No digest moved: `digest --check` holds all
eight, and nothing under `src/` that a pass runs differs from round 13.

Every llama.cpp number below is from an M1 Max with other jobs running (load average 3
to 10, given per leg), so each rate is contended. Every pass in every leg was held to the
engine's record as it decoded (`STOCK_REFERENCE`), except where a fault was injected.

**The merge.** The conflicts were in three docs. `tools/stock-llama/src/play.rs` merged
without one: the probe's extra whole-model decode goes through `Cache::pass` and is not in
round 13's four shape counters. At the merged tree: `cargo test --release --features
testing` 28 files ok, the harness's 32 tests, the runner's tests, both exports with
"manifest ok" and 0 values differing. The uniform file is byte for byte round 13's
(sha1 `2f87678b`, 1,544,154,592 bytes). The trimmed file's manifest names this work's
patches, so its sha1 is new (`5dce2fc1`, 769,593,632 bytes; its tensors are the same).

**Do the patches need a change for the new head lists? No** (measured for the four
streams that ran sliced; the stepper's slice has no pass in these games, and the length's
passes sit under the fence). The three patches
of this work apply in order to a clean b12070 (`00f113ec`; 7 files, 442 lines added)
and that build ran every leg below. `tools/stock-llama/runs.ts` now reads the heads key as
the patch's loader does and gives, for every run of columns, the view `mm_cols` reads
(the run plus its lead back to a multiple of 4 weights) and the kernel Metal takes for
that width. On `ff86dde7`'s trimmed file:

- 50 head runs. Each is a whole number of 64-weight heads from a multiple of 64, so none
  has a lead, and all 50 are block loads (a multiple of 32 wide). `7b064d51` had 47 of
  the same form.
- No FFN run's view is a multiple of 32 wide, or 32 to 63 wide and a multiple of 4. So
  on this model no FFN view, padded or not, goes through the block-load or the half4
  kernel: all take element loads. `7b064d51` had one, unpadded (the APU's 768 units from
  unit 1,644). The new model's nearest are 768 units from unit 1,641 and 480 from 2,311,
  whose padded views are 769 and 483 wide.

So this model, like the last, gives no real test of a padded view through those two
kernels. The legs below show the padded views read right through the element loads and
the head runs through the block loads. A later model with such a run would be its first
test; `runs.ts` lists them, and the probe and `STOCK_STREAMS=check` would hold it.

**The patched build on Metal** (measured; b12070 with the three patches, the trimmed file,
the harness of that day, load 3 to 9):

| Leg | Result |
|---|---|
| Yellow, 3,000 passes, `STOCK_STREAMS=check` | all 3,000 equal the record. 2,957 sliced passes (1,958 GBC CPU, 742 control, 189 APU, 68 PPU) each decoded by the whole model too: 0 rows differ. The length's 43 passes have 10 tokens, under the runtime's fence of 16, so they ran the whole model and were not sliced. No stepper pass: Yellow has none |
| Yellow, 4,000 passes, `STOCK_STREAMS=require`, three runs | all equal in each; `diff` of the first run's record against the engine's: 4,001 events equal, 313,445 tokens. The probe held 9 sliced passes to the whole model, 0 rows differ |
| Yellow, 3,500 passes, `STOCK_SCOPE=0 STOCK_SPLIT=0` | all equal; `diff`: 3,501 events equal, 274,258 tokens. GBC CPU 135.3 ms |
| Zelda DX frame 7 (`--warmup 6`), its own prefill (716.9 s), `STOCK_WRITES=check STOCK_READS=check` | exit 0, all 21,110 passes equal (19,491 GBC CPU, 238 PPU, 845 APU, 416 control, 120 length), 20,990 sliced; 19,450 narrow passes, 626 decoded again, 4 first for the windows; 876 loan records, 424 cells put back and compared, 0 differ, 0 unchecked; 3,126 write records both ways, 0 cells differ; 160 reader records both ways, 0 rows differ; the probe held 27 sliced passes, 0 rows differ. The frame took 934 s |

Rates on the slices, the three 4,000-pass runs (they agree to 0.1 ms), and the 300 frames
projected from them with the proof of record's counts (inferred):

| | ms each | `7b064d51` ("The slices' NaN on Metal") | Count in 300 frames | Hours (inferred) |
|---|---:|---:|---:|---:|
| GBC CPU pass | 12.0 | 11.4 | 925,752 | 3.09 |
| PPU pass | 27.8 | 22.7 | 83,498 | 0.64 |
| APU pass | 17.6 | 17.4 | 251,674 | 1.23 |
| Control pass | 11.7 | 11.7 | 71,759 | 0.23 |
| Length pass (the whole model, under the fence) | 8.2 | 8.4 | 41,855 | 0.10 |
| Write record (26 in each run) | 19.2 | 19.3 | 119,435 | 0.64 |
| Reader record | 20.2 (inferred, as there) | | 46,951 | 0.26 |
| Prefill | 717 s (Zelda DX's, above) | 756 s | | 0.20 |
| **Sum, with 0.72% for checkpoints** | | | | **6.4** |

The shared slots do not speed the patched path. The sum is 6.4 hours against `7b064d51`'s
6.2: the PPU's sliced pass is 22% slower and the GBC CPU's 5%. The two models' rates were
taken a day apart at different loads, not interleaved, so the 5% is within what load can
move; the PPU's 5.1 ms is not, by the three runs' agreement. Why the PPU's slice costs
more was not traced. (A guess: a slice keeps every layer-0 gather of the groups its part
holds, and the shared slots' groups are cut where the holders change.)

**Unpatched llama.cpp on Metal, Yellow through frame 13** (measured; `PROOF_PURE=1
FRAMES=13` through `proof.sh`, the uniform file, the whole model every pass, the harness
of that day, load 3 to 6). Frame 1 holds no bank switch and one reader record; frames 2 to 13 hold Yellow's bank switches, the second decodes
and the OAM DMA's reader records.

- Exit 0, 0 error lines. All 138,984 passes equal the engine's record: 122,303 GBC CPU,
  1,120 PPU, 10,908 APU, 4,084 control and 569 length. The run's record equals the
  reference's stream by stream (141,492 events in the GBC CPU's). Frame 13's hash is the
  engine's, `d043ab24`, which is the blank screen's (frames 1 to 81), so the evidence is
  the passes.
- 19,563 write records, 321 reader records (the OAM DMA's from frame 12), none off the
  mirror; 2,398 GBC CPU passes decoded again after naming a line outside the narrow
  scope, 28 decoded first for the windows, 2 moves of the windows, 2,567 lines joined the
  scope. No loan: Yellow lends none.
- Three chunks held the GPU lock 2,702 s, 2,701 s and 295 s, 1 h 35 min in all, none over
  an hour. Each of the two resumes replayed the host to its checkpoint in 90 and 102 s.
  19 checkpoints took 0.38% of the time.
- The cache was loaded, not prefilled: it is the one round 13's unpatched binary
  prefilled on this same file in 1,754 s (`PROOF_STATE`). The frame-1 run of "Round 13"
  could not be continued, since a finished run is refused and so is another binary's
  checkpoint.

| | Count to frame 13 | ms each (measured) | `7b064d51` ("Round 12") | Count in 300 frames | Hours (inferred) |
|---|---:|---:|---:|---:|---:|
| GBC CPU pass (its second decodes inside) | 122,303 | 35.4 | 42.8 | 925,752 | 9.10 |
| PPU pass | 1,120 | 59.2 | 77.7 | 83,498 | 1.37 |
| APU pass | 10,908 | 37.4 | 43.5 | 251,674 | 2.61 |
| Control pass | 4,084 | 37.1 | 43.4 | 71,759 | 0.74 |
| Length pass | 569 | 9.9 | 11.4 | 41,855 | 0.12 |
| Write record | 19,563 | 21.3 | 27.0 | 119,435 | 0.71 |
| Reader record | 321 | 26.2 | 28.2 | 46,951 | 0.34 |
| Prefill | | 1,754 s ("Round 13") | 2,581 s | | 0.49 |
| **Sum, with 0.38% for checkpoints** | | | | | **15.5** |

The projection takes thirteen frames' rates for 300 and the proof of record's counts.
"Round 13" inferred 15.8 hours from frame 1 and the old model's ratios; the measured
rates give 15.5, 82% of `7b064d51`'s 19.0. The 300 frames were not started.

**The sampled audit and the faults** (measured, the same harness and file). Unpatched on
Metal, `STOCK_SCOPE=check STOCK_CHECK_EVERY=100`, 3,000 passes: all equal, 29 passes also
decoded over every cell with 0 rows differing, and 29 scoped sequences read back from
llama.cpp and held to the cache's account (3.3 ms each). `STOCK_FAULT_SEQUENCE=300` on the
same build stopped at `SCOPE SEQUENCE DIFFERS` (484 cells against 485 lines).
`tools/stock-llama/faults.sh`: the clean replay ended with exit 0 and each of the five
faults (scope, map, probe, named, read) stopped at its named check.

**A pass holds its stream's own group, and a whole-sequence GBC CPU pass is not exact on
Metal** (measured). The review of the shared slots found that a whole-sequence dense pass
of the GBC CPU stream now puts 4.3·10^5 in the residual at the other parts' positions,
past f16's 65,504, where `7b064d51` stayed near 9. `STOCK_GROUP=check` decodes each pass
both ways, its own group and the whole sequence, and requires equal rows. On unpatched
Metal it stopped at the first GBC CPU pass on `ff86dde7`: 148 rows differ, the whole
sequence's rows decoding token 0 where the own group's are the record's. The APU's 1 pass
and control's 2 before it agree. The same leg on `7b064d51` (the "Round 12" kit) stopped
at the same pass with the same 148 rows. So this is not new with the shared slots, on
Metal it predates them, and `play.rs`' header already said a full group is not exact in
llama.cpp. The own group is unaffected: every one of the passes above ran it and equalled
the record. The claim of record is therefore stated with it ("What runs where" in the
README): a pass decodes its stream's own group, and the 300-frame proof ran exactly that.

**The Mac CPU backend** (measured; `STOCK_GPU_LAYERS=0`, the unpatched source, the uniform
file).

- `cargo run --release --example f16foreign`: 1,863 of 76,143 part-owned gate rows and 0
  of 5,490 query rows can pass 65,504 at another part's positions. `7b064d51` had 2,027
  of 76,143 and 0 of 5,421. Fewer rows, but the largest sums at the GBC CPU's positions
  rose: the stepper's rows from 524,160 to 981,120, control's from 409,472 to 953,344,
  the APU's from 522,368 to 753,536, the PPU's from 490,624 to 522,240. At every other
  pair of parts the worst sum is equal or lower. Both models are far past f16 there, so sharing did not change which builds work.
- Built with `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.4-a+nofp16` (the full line is
  in `docs/stock-llama.md`, "What runs where"; the source tree had no changed file):
  Yellow's first 400 passes all equal the record (254 GBC CPU, 99 control, 26 APU, 13
  length, 8 PPU). A GBC CPU pass takes 419.7 ms, control 536.7, APU 497.2, PPU 804.5,
  length 107.5 (one run, load 7 to 10).
- The proof's build on the Mac CPU backend: NaN at the first GBC CPU pass, as on the two
  models before. `STOCK_NANCHECK` names `ffn_gate-0`, unit 1,469, whose exact sum is
  −352,768.
- **"The default build" was the wrong name for that build.** Every llama.cpp build in
  this repository's runs, the proof of record's included, was configured with
  `-DGGML_BLAS=OFF`, and llama.cpp's own default on a Mac leaves BLAS (Accelerate) on
  (`GGML_BLAS_DEFAULT`). "The Mac CPU backend's NaN" above and `play.rs` called the
  BLAS-off build the default one. The unpatched source built with llama.cpp's own
  defaults (only the tools and tests left out) was run here for the first time: on Metal
  Yellow's first 3,000 passes all equal the record (GBC CPU 32.3 ms), and on the Mac CPU
  backend the first 400 passes all equal the record with no NaN (GBC CPU 308.5 ms,
  control 339.4, APU 329.9, PPU 571.0, length 97.9; one run, load 3 to 6). So the Mac CPU
  backend's NaN belongs to BLAS-off builds. That Accelerate's f32 multiply is why is
  inferred; which multiplies it takes was not traced, and 400 passes is all that ran.

**The review of the shared input slots** (round 13, an independent
reader). It found no wrong result and these gaps, each closed here without moving a
digest:

| Finding | What changed |
|---|---|
| `tests/stream_audit.rs` gave the other holders of a slot byte 0, and the checkpoint gathers a slot once | Each random state is evaluated a second time with every holder seeing the part's byte. The part's own outputs must be the first evaluation's, and a refusal must name another part's node. Of the APU's 199 random states 173 run to the end and 26 are refused at another part's node, with 62 queried lines that hold no cell. Those values sit at positions only a whole-sequence pass holds (above). The same premise is corrected in `scope::cubes`' and `View`'s comments |
| The assert "a slot's bit is read by every holder or by none" compared `Circuit::unread`, which marks head line bits only | Removed. Every input bit wire is pinned, read or not; the guard is `compile_with`'s assert that every holder's wire has the gather's channel. "Round 13"'s two sentences are corrected in place (109 of 1,912 input bit wires have no reader) |
| `placed` did not hold the compiled head count, copy heads included, to the cap | `assert_eq!(cfg.heads, placement.heads)` beside the hidden assert. It was not seen to fail |
| Nothing exercises a token above 65,535 since the vocabulary fell | A unit test maps a view's tokens above 65,535 and back on a 300-slot `Config` (`a_views_tokens_map_above_65535_and_back`) |
| "The APU's circuit takes any input bytes" | It refuses about 1 random state in 600; the comment says so |
| Nits | "Frame 1 `d043ab24`" is the blank screen's hash (above). The 109 write records of the frame-1 run are 85 in the GBC CPU's stream and 8 each in three others. `examples/shape --one` counts layer 0's gathers by holder |

**The layers lever is closed for this round, not built.** Its bound, from "Round 13": the
APU reached 21 levels on the APU depth probe, but the packer does not hold a placement
at 22 layers (it ends over the head cap and widens hidden), and 22 layers at FFN 3,968
measured 94.4% of a pass, about 6%. That needs a packer that places heads and units
together. Nothing here changes that.

**The full program and the runner, text bench** (measured; 600 frames from the frame-2,900
hand-off, round 12 with `7b064d51` against this work with `ff86dde7`, each
from its own trimmed GGUF and its own snapshot, three ABBA rounds, load 3.4 to 5.9, every
run at frame `870aa3fd` with 0 dense fallbacks):

| Text, 600 frames | `7b064d51` (fps) | `ff86dde7` (fps) |
|---|---:|---:|
| Runner (`tools/runner`) | 63.7 (63.4 to 64.2) | 63.6 (63.0 to 64.4) |
| Full program, one model from the GGUF | 62.35 (61.0 to 62.7) | 62.25 (61.2 to 62.7) |

All 24 runs are above 60; the lowest is 61.0. "Round 13" measured the full program lower
on the new model in 5 of 6 pairs (61.7 against 62.0). That does not hold here: it is
lower in 2 of the 6 pairs and higher in 4, and the medians differ by 0.1. Nor does that
round's runner gain (63.8 against 63.0): the runner's medians differ by 0.1 the other
way. On this machine at this load the two models are not told apart by this bench.

**Gate.** `tools/gate.sh` in full, `cargo test` included: 66 ok, 0 FAIL, 1
skip (Nintendo's DMG boot ROM is absent; SameBoy's stands in). The ignored one-model test
(`the_null_token_decodes_the_null_cell`) passes. Both exports made again at that tree
are byte for byte the files the legs ran, and `digest --gguf` says "manifest ok" on both.

**Kits** (outside the repository, two directories kept locally,
the unpatched kit and the patched kit). Each holds the harness binary, its GGUF, the engine's record to frame
300 (hard links to round 13's record, which this harness replays clean and every leg
was held to), `proof.sh`, `proof-status.sh`, `faults.sh`, a `STATUS` with the sha1 of the
binary, of each llama.cpp library and of the file, and a `COMMAND` with the line that
starts the 300 frames in a fresh work directory. Neither was started. A third,
the default-build kit, is described next.

**The default-build kit** (kept locally beside the other two; measured, one
machine, load 3 to 7). It is `kit-pure` with one difference: the harness links unpatched
llama.cpp b12070 built with llama.cpp's own defaults, so BLAS (Accelerate) is on. The
model file is the same hard link (`2f87678b`), and so is the engine's record. Which of
the two builds the claim of record names is not decided here, and "What runs where" is
unchanged.

- The build. The configure line passes only the build type and the switches that choose
  which targets are built: `-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF`. The cache
  then holds `GGML_METAL=ON`, `GGML_BLAS=ON`, `GGML_ACCELERATE=ON`,
  `GGML_BLAS_VENDOR=Apple`, `GGML_NATIVE=ON`, `CMAKE_BUILD_TYPE=Release`. The build
  directory, kept locally beside the kits, is the one the binary's rpath names. Harness binary
  `7eff5f0d`, `libllama` `e358c095`; `STATUS` lists every library's sha1,
  `libggml-blas` included.
- The source tree. `git -C` fails in the unpatched checkout: its `.git/refs` is gone
  (the directory changed at midnight on 2026-11-25; that the scratch directory's cleaner
  removed it is inferred). The earlier default build above ran after that, so its log's
  "source dirty files: 0" counted the output of a failed command and checked
  nothing. Here the tree was held to commit `00f113ec` two ways before the build. A copy
  of its own `.git` with `refs/` restored gives HEAD `00f113ec`, tag b12070 and an
  empty `git status`. A fresh index read from that commit in another clone's object
  store, which hashes all 4,029 tracked files, gives an empty `git status --ignored`.
- Metal, the first 3,000 passes of Yellow under `STOCK_REFERENCE`, on the cache
  round 13's BLAS-off binary prefilled: all equal the record, 0 error lines. It ran twice, with
  `kit-pure`'s binary between the two on the same cache, which also decoded all 3,000
  equal. The 26 write records took 27.9 ms each on the default build and 27.8 on
  `kit-pure`'s. One run of the BLAS-off build and two of the default.

  | ms per pass, first 3,000 passes | Default build | BLAS off (`kit-pure`) | Default build again | Passes |
  |---|---:|---:|---:|---:|
  | GBC CPU | 32.3 | 32.3 | 32.4 | 1,958 |
  | PPU | 59.4 | 59.3 | 59.3 | 68 |
  | APU | 36.9 | 36.8 | 36.9 | 189 |
  | Control | 37.3 | 37.3 | 37.3 | 742 |
  | Length | 10.7 | 10.7 | 10.7 | 43 |

  The two builds are not told apart on Metal: the largest gap is 0.1 ms, which is also
  the gap between the default build's two runs. The 35.4 ms of the frame-13 run against
  32.3 here is the window (thirteen frames against the first 3,000 passes), not the
  build, since the BLAS-off binary also gives 32.3 over this window.
- Metal, the prefill by this binary with no state loaded: 143,881 line cells in 1,754.2 s
  (round 13's BLAS-off binary took 1,754.5 s), then the first 3,000 passes over that cache, all
  equal, 0 error lines (GBC CPU 32.2 ms). The saved cache (sha1 `aaa64e15`, 922,354,012
  bytes) is byte for byte the one round 13's BLAS-off binary saved. It stays in the kit as
  `state/`, labelled; `COMMAND` does not load it and prefills in its own work directory.
- `faults.sh` with this binary: the clean replay ended with exit 0 and no error line, and
  each of the five faults (scope, map, probe, named, read) stopped at its named check.
- The Mac CPU backend (`STOCK_GPU_LAYERS=0`, `STOCK_NANCHECK=1`), on the cache Metal
  prefilled: the first 400 passes all equal the record, no NaN (GBC CPU 308.9 ms, control
  341.3, APU 334.4, PPU 574.0, length 99.8; one run).
- The 300 frames were not started. The projection is `kit-pure`'s 15.5 hours (inferred):
  the per-pass rates over the same window and the prefill are equal on the two builds,
  and no rate of this build past 3,000 passes was measured. The GPU lock was held 2,418 s
  for these checks.
- If the scratch directory is cleaned (inferred: empty directories of the unpatched
  checkout and of older rounds went missing at midnight on 2026-11-25), a run of 15 hours
  kept there is exposed. Whether a file a running proof needs would be removed was not
  tested.

**Not done, not checked.**

- The 300 frames on `ff86dde7`, in either build. Both figures above are projections.
- A padded view through Metal's block-load or half4 kernels: this model has no such run.
- A sliced length pass (10 tokens, under the fence) and any stepper pass in llama.cpp.
- The prefill by this work's BLAS-off binary: the frame-13 run loaded the cache
  round 13's binary prefilled on the same file. The default-build binary prefilled it
  again and saved the same bytes (above). The patched build prefilled Zelda DX's cache
  itself.
- Loans in unpatched llama.cpp on this model: Yellow lends none through frame 13, and
  Zelda DX ran in the patched build only.
- The flagged Mac CPU build on `45bfb056`, and more than 400 passes of any Mac CPU build
  on any model. Each Mac CPU run started from a cache Metal prefilled.
- More than 3,000 passes on Metal of a build with llama.cpp's own defaults (BLAS on). The
  proof of record and every other Metal run used `-DGGML_BLAS=OFF`.
- Any backend but Metal and the Mac CPU's, and any llama.cpp version but b12070.
- Why the PPU's sliced pass is 5 ms slower than `7b064d51`'s.
- The engine's record was not written again by this work's harness. It replays the
  first 2 frames of round 13's record clean.
- The new head-count assert in `placed` was not seen to fail.

## The proof of record on `ff86dde7`: Yellow 300 in unpatched llama.cpp, the default build (2026-11-25)

The current one model, `ff86dde7`, ran Yellow from the hand-off state to frame 300 in
unpatched llama.cpp b12070 on Metal, built with llama.cpp's own default options. Every
decoded pass equalled the engine's record, and frame 300 hashes to `cfd31f3d…`. The run
held the GPU lock for 15 h 28 min 31 s. It replaces the run of 2026-11-23 on `45bfb056`
("The first proof", above) as the claim of record.

The run's folder is local to the machine that ran it, which git does not
track. Every number below is from its `work/proof.log`, `final-status.txt`, `STATUS`,
`driver.log` and last checkpoint, measured unless a line says otherwise. The hashes of
the binary, the model file, `libllama` and the saved cache were taken again for this
section and equal `STATUS`'s. No compiled digest moved, and no code changed for this
section.

**What ran.**
- The model is the one model, digest `ff86dde7`, as the uniform file: sha1 `2f87678b…`,
  1,544,154,592 bytes. The last checkpoint names the tensors' digest `ff86dde7…` and the
  wiring's `81ccc726…`.
- The runtime is llama.cpp b12070, commit `00f113ec…`, with no patch, in the default
  build: the cmake line passes no `GGML_` option, and the cache holds `GGML_METAL=ON`,
  `GGML_BLAS=ON` with vendor Apple and `GGML_ACCELERATE=ON` (`STATUS`; `libllama` sha1
  `e358c095…`). The first proof's build had `-DGGML_BLAS=OFF`. `STATUS` records how the
  source tree was held to the commit before the build, two ways, since `git -C` fails in
  that checkout. The harness printed `llama.cpp unpatched; whole model every pass` in
  each of the 3 chunks. The settings were Metal (`STOCK_GPU_LAYERS=99`), `STOCK_PURE=1`,
  `STOCK_STREAMS=0` and `STOCK_GROUP=own`.
- The harness is `tools/stock-llama` built from the source of that day, with its library search path
  changed to the run's folder and the binary signed again ad hoc: sha1 `7eff5f0d…`
  before, `d8677352…` after, and the checkpoints name the second. The libraries are
  byte-for-byte copies of the default build's (`cmp`, repeated for this section). By
  `STATUS`'s account `libggml` still searches the scratch build's folder first for its
  backends, so which of the two identical copies of `libggml-cpu`, `libggml-blas` and
  `libggml-metal` was loaded was not traced.
- The run started at the hand-off state: the checkpoints' settings read `hardware Cgb,
  boot none, warmup 0`. It did not run the boot ROM. The host ran with every component in
  weights and in lockstep.
- The engine's record is a copy of round 13's record of Yellow to frame 300, byte-identical to the kit's (`cmp` at set-up, by `STATUS`).
  `STATUS` states that nothing the engine runs under `src/` differs between that tree and
  this one. That statement was not checked again here.
- The run prefilled its own cache in unpatched llama.cpp in 1,757.6 s (29 minutes; the
  first proof's prefill took 4,961.6 s). The saved cache is 922,354,012 bytes with sha1
  `aaa64e15…`, the sha1 `STATUS` gives for the cache that the kit's binary and
  round 13's BLAS-off binary each saved.

**What was compared.**
- Frame 300 hashes to `cfd31f3d89f111fe6fd2fd8f36a1fe4971aae642`. The engine's record of
  the same run has that hash (`record-engine.log`), and so does the gate's one-model
  Yellow 300.
- 1,374,538 of 1,374,538 decoded passes equalled the engine's record as they decoded
  (`STOCK_REFERENCE`): 925,752 GBC CPU, 83,498 PPU, 251,674 APU, 71,759 control and
  41,855 length. The stepper has no pass in this run.
- After `play` exited, `proof.sh` held the run's own record to the reference, event for
  event: 995,266 GBC CPU events (the 925,752 passes and the 69,514 write records the
  engine keeps), 83,498, 251,674, 71,759 and 41,855, all equal, and 0 for the stepper.
- The log holds 0 lines that match `proof-status.sh`'s `ERR` pattern (`REFERENCE
  DIFFERS`, `SCOPE MISSES LINE`, `RESUME DIFFERS`, `LEDGER GAP`, `panicked` and the
  rest), and 0 with `RESUME REFUSED`, `RESUME FAILED` or `CHECKPOINT NOT SAVED`.
- Every scoped pass that the engine's record names lines for was held to them: the
  scope it decoded over held every line its heads read (925,752 GBC CPU, 83,498 PPU and
  41,855 length passes; the log counts 0 for the APU and the control logic, whose rows
  attend one cell each).

**Chunks.** The run took 3 chunks and 2 resumes. Chunks 1 and 2 ended at their 21,600 s,
and chunk 3 ran to the end. `proof-status.sh` reports that each resume began at a
checkpoint a live chunk saved and that the live events run from event 0 to the log's end
with no gap. The lock was taken at 07:19:04, 13:19:05 and 19:19:07 on 2026-11-25 and
`play` last exited at 22:47:35, which gives the 15 h 28 min 31 s. `proof.sh` exited at
23:02:11 after the record comparison, 15 h 43 min after the first lock. The 1-minute load
at the three locks was 3.65, 2.64 and 3.43.

**Rates and time.**

| | Count | ms each | Hours (count × rate, computed) | Projected ("Round 14") | The first proof |
| - | -: | -: | -: | -: | -: |
| GBC CPU pass | 925,752 | 34.6 | 8.90 | 35.4 ms, 9.10 h | 163.7 ms |
| PPU pass | 83,498 | 59.3 | 1.38 | 59.2 ms, 1.37 h | 164.2 ms |
| APU pass | 251,674 | 36.6 | 2.56 | 37.4 ms, 2.61 h | 125.1 ms |
| Control pass | 71,759 | 37.4 | 0.75 | 37.1 ms, 0.74 h | 129.9 ms |
| Length pass | 41,855 | 9.6 | 0.11 | 9.9 ms, 0.12 h | 27.9 ms |
| Write record | 119,435 | 21.6 | 0.72 | 21.3 ms, 0.71 h | 118.0 ms |
| Reader record | 46,951 | 26.2 | 0.34 | 26.2 ms, 0.34 h | 587.5 ms |
| Prefill | 1 | 1,757.6 s | 0.49 | 1,754 s, 0.49 h | 4,961.6 s |
| Resume, to the first live event | 2 | 151 s and 225 s | 0.10 | not counted | 118 to 299 s |
| Checkpoint | 184 | | 0.04 | 0.38% | 856, 0.29% |
| **With the GPU lock** | | | **15.48** (rows: 15.39) | **15.5** | 71 h 34 min |

The pass rates are `final-status.txt`'s, each stream's time over its passes. The write
and reader rates are the cache's timers over their counts (2,583.8 s and 1,230.0 s). The
rows sum to 15.39 hours against the 15.48 hours with the lock. The sum is computed, and
the rows are not shown to be disjoint. Inside llama.cpp's decode a GBC CPU pass took
33.6 ms. The 184 checkpoints took 152.3 s of 55,457 s (0.27%).

The projection of 15.5 hours took thirteen frames' rates and the first proof's counts,
and the run took 15.48. The counts did not move: the same 1,374,538 passes,
119,435 write records and 46,951 reader records.

The run took 22% of the first proof's time with the lock (4.6 times faster). Three things
changed between the two, and this run does not separate them. The model is smaller
(uniform width 3,456 with 26 heads in a 1.54 GB file, against width 10,688 with 69 heads
in 5.04 GB). The harness's scopes are narrower and it gathers reader records over their
scope ("The harness's scopes narrowed"). The build has BLAS on, which on Metal measured
the same as BLAS off over 3,000 passes ("Round 14").

**The cell counts, reconciled.** Four counts of Yellow's cache appear in these docs. All
four describe the same cache.

| Count | What it counts | Where it prints |
| -: | - | - |
| 143,830 | the line cells of the GBC CPU's image | the engine's `cells in the cache` |
| 143,847 | the distinct lines the llama.cpp cache holds: those 143,830 and the 17 palette lines (`PAL`, 136 bytes) that only the PPU's images hold | a checkpoint's `mirror` count, and 2,521 fixed cells plus 141,326 lane records |
| 143,848 | the cells of a saved cache: the 143,847 and the null cell | `state-diff` |
| 143,881 | the harness's list before it merges duplicates: the 143,830 and the 17 palette lines once each for the PPU, the length and the stepper streams | the prefill's `line cells`, `CacheStats.cells` |

So the prefill's "143,881 line cells" overstates the cells it decodes by 34. It decodes
143,847 lines, two of them twice (the "2,523 one by one" are 2,521 cells and the second
decodes of the top cell and VOID's). The harness's text was left as it is, since the
count is part of the saved cache's sidecar. The split is read from
`tools/stock-llama/src/play.rs` (`prefill`) and `cache.rs` (`decode_cells`), and the
arithmetic agrees: 143,830 + 3 × 17 = 143,881 and 143,830 + 17 = 2,521 + 141,326. That the
three images are the PPU's, the length's and the stepper's is read from the log, where those
three streams' scope lines name the palettes.

**Tokens in a GBC CPU pass: 99 or 100.** Both are the harness's mean, rounded, over
different windows. The decodes of the GBC CPU's groups average 99.3 tokens over frame 1,
and the passes average 100.2 over the run (92,607,619 tokens in
924,054 passes at the last checkpoint). The log prints 99 through frame 1 and 100 or 101
after it.

**What this run does not show.**
- It starts at the hand-off state, not at power-on, and it ran one game.
- Every pass ran every weight of the model, but not over every cell. A pass attends its
  stream's scope of the cache. 913,778 of the 925,752 GBC CPU passes ran over the narrow
  scope, 2,820 were decoded again after naming a line outside it, and 2,896 were decoded
  first for the windows alone. Those second decodes are the 5,716 by which the cache's
  1,380,254 decodes exceed the passes. No pass, write record or reader record had a
  second decode over every cell to compare with (`scope check 0 passes`, `write_checks:
  0`, `0 also over every cell`). What holds the scopes here is the comparison with the
  engine's record, the lines check above, and the bounded audits of "The harness's scopes
  narrowed" and "Round 14".
- The cache decoded 119,435 write records and 46,951 reader records. Only the 69,514
  kept write records are events in the record comparison. The others are held by their
  counts and by the passes that read their cells, which is inferred and not a check of
  each cell.
- Yellow lends no lines in 300 frames. Every loan and swap counter is 0. The stepper's
  stream has 0 passes and the length model marked 0 lines.
- Each pass was decoded and held to the record once, in the chunk that ran it live. A
  resume runs the host again from frame 0 with every earlier pass answered from the
  engine's record and no model. Chunk 3 went live after event 1,200,735, in frame 181
  (1,089,349 passes). So the process that printed frame 300's hash decoded the rest of
  frame 181 and frames 182 to 300 itself and replayed everything before that event. The
  lockstep counters at the end of the log are that process's, so they include replayed
  passes.
- The build's objects came from ccache (`STATUS`), and the binary that ran is the
  re-signed copy. The Mac CPU backend, the patched build and every other backend are
  outside this run.

**Two lines of the folder to read with care.** `STATUS` says the run was launched
at 07:17 and gives 15 h 45 min of wall time. The folder was set up at 07:17, the log's
first line and first lock are at 07:19:04, and from there to `proof.sh`'s exit is 15 h
43 min. The `prefill:` line in `final-status.txt` is chunk 3's replay ("REPLAYED, no
model ran ... in 23.2s"). The prefill that ran in llama.cpp is line 25 of `proof.log`.

## Loans and the stepper in unpatched llama.cpp, the default build (2026-11-26)

The proof of record exercises neither a loan nor a pixel-stepper pass, because Yellow's
300 frames have none. Two one-frame runs of Zelda DX now cover both in unpatched
llama.cpp b12070 on Metal, with the proof's binary and file: the harness sha1
`7eff5f0d…` (the kit's copy, before the proof's re-signing), the default build, the
uniform file `2f87678b…`, `STOCK_PURE=1`, `STOCK_STREAMS=0`, `STOCK_GROUP=own`. Every
pass ran the whole model and was held to an engine record written for the run by the
same binary (`play none`, then `STOCK_REFERENCE`). No code changed, and
`digest -- --check tools/digests.txt` gives 8 equal digests on the tip. Every number is
measured unless a line says otherwise. The load was 2.8 to 3.9 during the runs.

**Loans: Zelda DX, frame 7 (`--warmup 6 --frames 7`, `STOCK_WRITES=check STOCK_READS=check`).**
- The run exited 0 with 0 error lines, and frame 7 hashes to the engine's `d043ab24…`.
  Frame 25 below has the same hash, so the pass comparison carries the claim, not the
  frame hash.
  The harness printed `llama.cpp unpatched; whole model every pass` in both chunks.
- All 21,110 decoded passes equal the record: 19,491 GBC CPU, 238 PPU, 845 APU, 416
  control and 120 length. The stepper has no pass in this frame. After `play` exited,
  the run's own record equalled the reference event for event (21,572, 238, 845, 416,
  120 and 0 events).
- 216 PPU passes lent 438 of the stream's lines. Each loan is two write-record decodes,
  one to lend the line and one to put the cell's line back (`swaps: 876`). The harness
  compared 424 cells after the put-back with the cells found: 0 differ, 0 left
  unchecked. The other 14 were lent again before any pass read them (`swap_relent: 14`). The replay with no
  model had predicted the same 876 and 424.
- 3,126 write-record decodes, the 2,250 write records and the 876 of the loans, were
  each done both ways, over the records' scope and over every cell, with 0 cells that
  differ. 160 reader records were decoded both ways with 0 rows
  that differ, and none decoded another line than the mirror's.
- Rates: GBC CPU 35.5 ms a pass, PPU 66.1, APU 37.2, control 37.4, length 16.3. With its second decode over every cell, a write record took 513 ms (1,153.9 s for
  2,250), a lend or a put-back 517 ms (453.2 s for 876) and a reader record 304 ms. The frame took 2,423 s after the prefill (inferred from the two
  chunks' clocks), against 934 s for the patched build's run of the same frame
  ("Round 14").
- The run held the GPU lock twice, for 2,711 s and 1,612 s. The first chunk prefilled
  the cache in 1,755.6 s and decoded 6,713 passes. The second resumed from the
  checkpoint after event 8,149 and decoded the rest.

**The stepper's first pass in each gate ROM.** With no input, the reference marks no
line through frame 300 of Yellow (`stepped_lines=0`, frame hash `cfd31f3d…`). A search
with `run --core ref` over 20,000 frames, confirmed by an engine record of the one
frame, finds the first marked line here:

| ROM | First frame with a stepper pass | The write the length stream's judge marked | Stepper passes in that frame |
|---|---:|---|---:|
| Zelda DX | 25 | SCY ← 253 on line 64, 142 dots after mode 3 began | 3,392 |
| Yellow | 1,154 | SCY ← 254 on line 36, 8 dots after | 1,754 |
| Mario DX | 1,627 | SCY ← 112 on line 7, 4 dots after | 847 |
| Crystal | 3,087 | SCX ← 112 on line 10, 36 dots after | 1,970 |
| cgb-acid2, dmg-acid2 | none through frame 20,000 | | |

In each case the game writes a scroll register while the PPU draws a line, early enough
to change it, so the scanline pass cannot draw that line. The register, the value and
the time are the length pass's `MWI`, `MWV` and `MWH` in the engine's record, with `MJ`
bit 0 set. The dots are the machine's count, `MWH` in 2-dot halves. The same frames hold other writes during mode 3 that the judge passes, since
each lands after the line's last pixel, carries the value the line started with, or
targets a register a CGB line does not read then (WY, BGP, OBP0, OBP1) (9 in
Zelda's frame, 108 in Yellow's, 134 in Crystal's, 1 in Mario's).

**The stepper: Zelda DX, frame 25 (`--warmup 24 --frames 25`).**
- The run exited 0 with 0 error lines in one hold of the GPU lock, 2,416 s, of which the
  prefill took 1,755.4 s. Frame 25 hashes to the engine's `d043ab24…`.
- All 14,982 decoded passes equal the record: 3,392 stepper, 9,891 GBC CPU, 190 PPU, 848
  APU, 245 control and 107 length. The 3,392 stepper passes draw one line and revise 4
  lengths. Each was held to the lines the record's heads name, and the run's own record
  equals the reference's 3,392 stepper events.
- A stepper pass is 158 tokens and took 46.7 ms (GBC CPU 35.4, PPU 59.3, APU 36.8,
  control 38.7, length 9.7).
- This frame lends no line (`swaps: 0`), and the run used the proof's settings, with
  no second decode of write or reader records.

**Not run.** A stepper pass in Yellow, Crystal or Mario DX. From these rates and each
frame's replay counts, Yellow's frame 1,154 would hold the lock about 37 minutes and
Mario DX's frame 1,627 about 34 (inferred). Crystal's would take about 63 minutes,
because its cache has 274,953 cells to prefill against Zelda's 143,881 (inferred). Also
not run: a stepper pass on a DMG on `ff86dde7`, a frame with both loans and stepper passes, and more
than one frame of either.

Logs: `gpu/ll-zelda7-unpatched.log`, `gpu/ll-zelda25-stepper-unpatched.log`,
`gpu/ll-search.out` and `gpu/ll-records.out` in a local folder of round 14,
which git does not track.
