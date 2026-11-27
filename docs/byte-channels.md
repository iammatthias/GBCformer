# Nibble-valued channels (d4)

**Now**, the nibble circuits are the only GBC CPU. The 0/1 circuits this
page measures them against (`build_bits`, `build_mem_bits`, `GBCFORMER_BITS`, `f16census
bits|bitsmem`) were removed; the commands below that name them are history.

Nibble channels became the default GBC CPU circuits when the nibble work was merged
onto the overlay (`docs/hardware-shaped.md` §15). `sm83::build` and `sm83::build_mem` give the nibble circuits
(`src/sm83/d4.rs`). At the time of these measurements `GBCFORMER_BITS=1` selected the 0/1
circuits (`sm83::build_bits`, `build_mem_bits`), whose checkpoints kept the overlay's
digests; setting it now fails with a message. The APU, control and PPU circuits are 0/1
as before. Every number is labelled measured (a command in §8),
estimated (arithmetic on measured inputs, with the assumption named) or guess.

The short answer:

- **The step of the GBC CPU with memory is 9% cheaper in timing runs (measured).** With
  every component in weights as one model and weights from the GGUF, text runs at 47.9
  fps and the GBC CPU pass costs 3.89 µs, against overlay's 43.6 fps and 4.27 (§4). The
  first A/B measured 3.94 µs against the 0/1 circuit's 4.69, 16% cheaper (§4.1). Neither is the
  3.1 µs that 60 fps on text needs (§7).
- **The checkpoints are smaller.** The GBC CPU with memory has 6,489 circuit units against
  9,517, and its own checkpoint has 135.3 M dense parameters against 137.0 M (0.19 M
  nonzeros against 0.23 M, over every tensor as `digest` counts them). The host-fed GBC CPU
  has 70.4 M dense against 79.6 M, and the one model 634.6 M dense against 716.5 M (§6).
- **Exact by construction, and f16-legal.** §3 re-derives every node, the norm, the gather
  and the LM head. Both GBC CPU circuits read 0 T1 and 0 T2 violations in the f16 census,
  and the checkpoints pass `check_f16` and `check_f16_acc`.

## 1. What a nibble channel is

With 0/1 wires a byte costs 8 channels, a byte selection costs 8 units per case, and a
16-bit add costs a staircase per output bit. A nibble channel holds an integer 0..15, so a
byte is two channels:

- **Inputs.** `Circuit::input_nibbles` adds two level-0 wires per input byte, `NAME#lo`
  and `NAME#hi`, beside its bits. Layer 0 already gathers the byte's 8 bits into 8 value
  dims of its head. A nibble wire is four of those dims written into one channel with
  `W_o` weights `1, 2, 4, 8 · UNIT` (`llama.rs`). It costs no level and no unit.
- **Ramps.** A node may be a ramp (`Node::ramp`), where each step contributes
  `coef · max(0, lin − θ + 1)`, the emitter's first unit alone.
- **Muxes.** `value_mux` selects among nibble values by 0/1 gates, with one ramp unit per
  case, `max(0, v + m·(Σ gates − n))`, where m is the largest value v's terms allow. A
  case is 0 unless all its gates are 1. A byte selection is two such nodes.
- **Sums.** A nibble array's value is affine in its nibbles (`Σ 16^i·n_i`), so an add
  costs only its carries. Each carry is one step on a nibble or byte form (`d4::add`,
  carry lookahead per byte).
- **Outputs.** `Circuit::output_nibbles` outputs a field as two nibble forms. Every field
  of a checkpoint with a nibble output routes as two nibbles into routes 0 and 1. A nibble
  routes as one ramp gated by its slot, and a bit field as its bits weighted 1, 2, 4 and 8.

The table lists what costs units and what is free.

| Operation | 0/1 circuit | Nibble channels |
| - | - | - |
| byte from its bits, nibble sums | free | free |
| byte selection, n cases | 8n units | 2n units |
| `a + b + c`, any constant offset | 8 staircases | the carries: 2 steps a byte |
| 16-bit `x + k` | 16 staircases | 4 carries |
| zero flag of a sum | 1 node | 1 node on the byte form |
| AND, OR, XOR, right shifts, BIT | per bit | per bit, then free as nibbles |
| RES, INC, DEC, left shifts | per bit | from the nibbles |

## 2. The circuits (`src/sm83/d4.rs`)

The decode is the 0/1 circuit's (`decode::decode`), with the opcode and CB one-hots, the
interrupt logic, the conditions and the lengths. In memory mode the circuit also keeps the
fetch, the request address, every address translation and the memory heads' queries as
bits, because a memory key is an address's bits (`docs/stock-llama.md` §4). The datapath,
the ALU and the output fields are rebuilt on nibbles:

- **Adders as carries.** `pc + len − hb + e8`, `sp ± 1`, `sp ± 2`, `pair ± 1`,
  `hl + pair`, `sp + e8` and `a16 + 1` are their carries plus affine nibbles. In memory
  mode the circuit translates the stack writes' addresses and `a16 + 1`, so they stay bits
  there.
- **The ALU.** ADD and SUB are a byte form with a carry step, the zero flag is one node
  over it, and the half carries are steps on the nibbles. AND, OR and XOR stay per bit and
  enter A' as nibbles. INC and DEC are nibble adds. DAA keeps the 0/1 corrections and adds
  them to A's nibbles.
- **Operands.** The circuit selects operands as bits (the logic ops, the shifts) and as
  nibbles (sums, loads, results). In memory mode the (HL) operand is read 0's byte, and the
  selection absorbs its pick.
- **Fields.** A', B'..L', SPL', SPH', PCL', PCH', RA0, RA1, the write addresses and the
  written bytes are nibble muxes with "keep" as the complement case. F', CTL', RQ, WQ,
  MC and ACK stay bit fields, and so do the write records' old bytes of a line.

| | Host-fed, 0/1 | Host-fed, nibbles | Memory, 0/1 | Memory, nibbles |
| - | - | - | - | - |
| Nodes | 1,012 | 598 | 1,821 | 1,332 |
| Circuit units | 6,012 | 1,591 | 9,517 | 6,489 |
| Depth (layers) | 5 (6) | 6 (7) | 10 (11) | 10 (11) |
| Own checkpoint, hidden | 1,408 | 1,600 | 1,728 | 1,856 |
| Own checkpoint, dense | 79.6 M | 70.4 M | 137.0 M | 135.3 M |
| Nonzeros, every tensor (`digest`) | 0.13 M | 0.09 M | 0.23 M | 0.19 M |

## 3. Exactness: what changed, and its re-derivation

### 3.1 RMSNorm

`UNIT = 2^-7` and `EPS = 2^14` make every norm divide by exactly 2^7 while the
residual's squares sum below `16·d` in f32 (`docs/stock-llama.md` §2,
`engine::int::norm_bound`). A nibble channel adds at most 225 where a 0/1 channel adds
1, so the compiler checks the bound over every channel's occupants
(`llama::norm_squares`, which counts each wire channel at its occupants' largest square,
a nibble route at 225, every other channel at 1, and the constants). The compiler widens
hidden when the bound needs more channels than the circuit uses, since each channel past
the wires adds at most 1 to the squares and 16 to the bound. The host-fed checkpoint uses
at most 1,088 channels (rounded to 64) and the norm takes 1,600. The memory checkpoint
uses at most 1,536 and the norm takes 1,856 (measured, `examples/shape`). `EPS` and
`GAMMA` are unchanged, so the GGUF loader's `GAMMA` check holds and f16 sums (T2) keep the
same bound (half an ulp of 2^14 in f16 is 8, which `16·d` stays under per channel).

### 3.2 Layer 0's gather

A nibble wire's gather is the exact sum `Σ 2^i · bit_i` of four of one key's value
dims, and the softmax weights stay exactly 1 or 1/len. The engine's layer-0 diff accepts a
gathered value when it is an integer with `|v| ≤ 15` (`GATHER_MAX`), and the range guard
below keeps the wire to its declared range.

### 3.3 The engine's range guard and integer bounds

`Template::check` guards a channel that no other wire shares at its wire's declared
range. A shared channel keeps its clearing's range, and a nibble wire never gives up its
channel, because one clearing unit undoes a 0/1 wire but not a nibble. `IntLayer`'s 2^23
and 2^24 checks bound each gate and down sum over its inputs' declared ranges, so the
integer form is proven on the nibble channels, not on [−1, 1].

### 3.4 The LM head

Token v with nibbles (l, h) scores `2l·L − l² + 2h·H − h²` against the two nibble routes
(L, H). That is `L² + H² − (L − l)² − (H − h)²`, so the token whose nibbles equal the
routes wins by at least 1. The weights are `2l, 2h ≤ 30` on the routes, the squares split
over the two constant channels (1 and 64), and the family bits of a per-slot output score
as the bit head's do. The logits stay under 2,048, exact under f16 sums. The engine
detects the nibble head and decodes the two routes by table.

### 3.5 The one model (the prefix overlay, `docs/hardware-shaped.md` §15)

Two of the overlay's bounds assumed every channel in [−1, 1]. With the GBC CPU's nibble
wires overlaid on other parts' channels, this section re-derives both:

- **The null term.** Outside its part's positions a memory head attends to the null token
  because its null term `N` beats every key its query can score over another part's
  channel values (`foreign_score`). That score now reads each channel's occupants' largest
  magnitude, so a PPU head's query over a channel that the GBC CPU's nibble wire occupies
  at a GBC CPU position sees up to 15. The null term grows from 6 dims of 31,744 to 9 dims
  of 30,720, against a null key of 4 (the power of two that keeps it within the 9 null
  dims, with each weight still an f16). The null logit was already beyond f16 and exists
  only in a runtime without per-part slices (`docs/hardware-shaped.md` §15.3).
- **The norm of each stream's slice.** The GBC CPU's prefix is the whole hidden, so its
  slice must meet the bound above. The other parts' slices hold only 0/1 wires.

### 3.6 What did not change

These are unchanged: K = 128, the step construction `relu(K(s−θ+1)) − relu(K(s−θ))`,
integer weights scaled by powers of two, the gather's equal logits, the causal-mask
equivalence, `EPS` and `GAMMA`.

## 4. Measurements

### 4.1 Timing runs, every component in weights as one model

Each run is `GBCFORMER_STATS=1 tools/bench/text.sh --weights all --one-model` (and the
overworld script), with weights from each side's GGUF, interleaved over three rounds,
reporting the median. A memory-mode step is one pass except for the 0.5% of steps the host
serves, so µs per pass is µs per step to within 0.5%.

The first A/B compares the nibble circuits with `GBCFORMER_D4=1` against the 0/1
circuits of `docs/hardware-shaped.md` §14, at load 1.4 to 3.4.

| | 0/1 circuits | nibble circuits |
| - | - | - |
| Text fps | 39.7 | **47.2** |
| Text, GBC CPU µs per pass | 4.69 | **3.94** |
| Overworld fps | 109.2 | 120.2 |
| Overworld, GBC CPU µs per pass | 4.63 | 4.21 |
| Text, `--weights cpu,memory --one-model`, fps | 44.9 | 49.8 |
| Text, cpu,memory, µs per pass | 4.14 | 3.74 |

The second compares d4 as the default against the overlay before it, at
load 1.8 to 20.7 (other jobs on the machine).

| | overlay | d4 default |
| - | - | - |
| Text fps | 43.6, 44.2, 42.5 (median 43.6) | 47.6, 48.5, 47.9 (**47.9**) |
| Text, GBC CPU µs per pass | 4.27, 4.21, 4.38 (4.27) | 3.91, 3.83, 3.89 (**3.89**) |
| Overworld fps | 110.3, 112.7, 108.7 (110.3) | 122.5, 123.5, 120.0 (122.5) |
| Overworld, GBC CPU µs per pass | 4.58, 4.48, 4.65 (4.58) | 4.12, 4.09, 4.21 (4.12) |
| Text, `--weights cpu,memory --one-model`, fps | 43.9, 45.8, 45.6 (45.6) | 49.8, 51.2, 51.0 (51.0) |
| Text, cpu,memory, µs per pass | 4.24, 4.06, 4.08 (4.08) | 3.74, 3.63, 3.65 (3.65) |
| Overworld, cpu,memory, fps | 122.0, 122.0, 122.2 (122.0) | 136.7, 135.8, 131.4 (135.8) |
| Overworld, cpu,memory, µs per pass | 4.14, 4.14, 4.13 (4.14) | 3.70, 3.72, 3.85 (3.72) |
| Weights load per stream (its slice) | 0.2 to 0.7 s | 0.2 to 0.5 s |

On text the GBC CPU stream re-evaluates 1,118 units a pass against 1,459 and walks 196.6
(changed channel, reading layer) pairs against 217.1. Its shared layers take 2.84 µs
against 3.29, layer 0 0.19 against 0.17 and decode 0.17 against 0.14 (round 2,
`GBCFORMER_STATS=1`). Every run gives frame hashes `870aa3fd` and `822059c1`.

### 4.2 Update census (`GBCFORMER_UNITS=1`, `CORE=model-mem`, slower than timing runs)

| | 0/1 | Nibbles |
| - | - | - |
| GBC CPU with memory, text: units a pass | 1,470 | 1,132 |
| (changed channel, reading layer) pairs | 217.1 | 196.6 |
| Shared layers, µs | 6.69 | 5.51 |
| GBC CPU with memory, overworld: units a pass | 1,351 | 1,021 |
| Pairs | 192.1 | 174.8 |
| Host-fed GBC CPU, text: units a pass | 502 | 219 |
| Pairs | 95.8 | 79.6 |
| Host-fed shared layers, µs | 2.36 | 1.40 |

On the GBC CPU with memory, the decode (`OP0`, 288 updates a pass on text), the address
translations and the memory heads' queries read bits because a memory key is made of
bits, and they are most of what is left (the `GBCFORMER_UNITS=1` census by node family).

### 4.3 The opcode decode as nibble values (measured, not kept)

This variant picked each opcode nibble as a value with one ramp per line byte and made its
one-hots `[value == n]` a level later. `OP0` fell from 288 to 127 updates a pass and the
units from 1,132 to 971, but the circuit gained a layer (depth 11), and the one model's
GBC CPU pairs rose from 190.7 to 211.1. In timing runs the GBC CPU pass went from 3.97 to
4.07 µs on text and from 4.25 to 4.36 on overworld (interleaved, median of 3,
`docs/perf-log.md`). It is not in the circuit.

## 5. Gates (measured)

| Gate | Result |
| - | - |
| Circuits against SST (`tests/sm83_sst.rs`: the nibble and the 0/1 circuits, host-fed and memory) | 500,000 cases each, 0 failed |
| Checkpoints against SST (`tests/checkpoint_sst.rs`) | host-fed 500,000, 0 failed, 564,030 passes all fast; memory 500,000, 0 failed, 502,976 passes all fast |
| Engine against the dense pass (`engine_matches_dense_pass`, `forced_fallback_matches_fast_path`, the memory engine with its cache) | pass |
| `cargo test --release` | every test file and the unit tests pass, `stream_audit`'s f16 audit of the one model included |
| `tools/gate.sh` | every line ok (30), rc 0: lockstep DMG 8 `6872b003` and CGB 300 `cfd31f3d`, the three other games, text and overworld, blargg, both acid2 images, the speed switch, and every phase in weights in lockstep, Yellow 300 as one model `cfd31f3d` |
| `CORE=lockstep-mem` text and overworld | `870aa3fd`, `822059c1`, no divergence |
| `f16census all`, `bits`, `bitsmem` | 0 T1, 0 T2 violations on every circuit |
| One model's GGUF (`export one-model --layer-shapes --align 64`), `digest --gguf` | 360.6 M values, 0 of 634,588,672 differ |
| `every_slice_is_the_checkpoint_at_its_positions` (ignored, 241 s) | pass |
| `GBCFORMER_ORACLE=1000` Yellow 300, every component in weights and in lockstep, one model from the GGUF | `cfd31f3d`; 166 dense oracle checks on the GBC CPU stream, 1 on the PPU's palette read, all equal |
| Wire-change table re-recorded (`changes-memory.txt`, text and overworld averaged) | no digest moved: the fit places the nibble wires the same either way |

Some checks were not covered by default. `tests/checkpoint_sst.rs` and the engine's tests
compile `sm83::build` and `build_mem`, so only `GBCFORMER_BITS=1 cargo test --release`
held the 0/1 checkpoints to SST and to the dense pass (the 0/1 circuits stayed in
`tests/sm83_sst.rs` and `f16census bits|bitsmem` until they were removed). One check
was not run:
`tools/stock-llama` (the llama.cpp proof) against the new GGUF. The nibble head, the ramp
units, hidden 1,856 and the null key of 4 pass the f16 census, which is only the
compile-time claim.

## 6. Sizes

| Checkpoint | Hidden | Intermediate | Layers | Dense | Nonzero, layers and LM head (`shape`) | Digest (0/1 then, under `GBCFORMER_BITS=1`) |
| - | - | - | - | - | - | - |
| cpu (host-fed) | 1,600 | 640 | 7 | 70.4 M | 31,735 | `87bb33f8` (`3c11d62b`, 79.6 M) |
| cpu-memory | 1,856 | 1,408 | 11 | 135.3 M | 98,368 | `6ee28082` (`1645a346`, 137.0 M) |
| apu, control, ppu | unchanged | | | 70.2 M, 40.6 M, 254.1 M | | `83c6b259`, `f2676ee8`, `a1ee149e` |
| one model | 1,856 | 5,504 | 14 | **634.6 M** | 0.92 M | `e872e6c9` (`6a17e689`, 716.5 M) |

The one model's hidden is the shared channels plus the GBC CPU's wires, 789 + 1,050 + 4 +
1 rounded to 1,856, which the norm's floor also allows (the GBC CPU stream's bound reads
29,695 against 29,696). The GBC CPU stream's slice has hidden 1,856, 11 layers, 233.6 M
dense and 125,587 nonzeros (from 2,112 and 279.2 M). The GGUF holds 360.6 M values, 722 MB
(from 432.2 M, 865.5 MB), and a run reads only its streams' slices.

## 7. What remains to 60 fps on text

Text needs 3.1 µs a GBC CPU step, and the default reaches 3.89 in timing runs (measured,
§4.1), so 0.8 µs remain, a fifth of the step. The remaining time is in these places,
measured on round 2 of the A/B unless labelled:

- **The shared layers, 2.84 µs**, for 1,118 units and 196.6 pairs a pass. At the
  per-pair cost of about 9 ns measured for the GBC CPU with memory ("The measurement
  first", `docs/perf-log.md`), the pairs alone are about 1.8 µs of it (estimated). The
  nibble decode (§4.3) showed that removing units at the price of a layer and of pairs
  does not pay, so the next lever has to remove pairs, not only updates.
- **The parts that must read bits.** These are the opcode decode (`OP0`, 288 updates a
  pass), the fetch adders `pc+1` and `pc+2` (48), the word lines of the three fetches and
  the data reads (`OP1w`, `OP2w`, `D0w`, `D0rw`, 92), the translations (`RRARE`, `F0` to
  `F2`, `RP0`, `WHL`, about 100) and the write records (census mode, §4.2). They are the
  memory key's bits (`docs/stock-llama.md` §4).
- **Outside the engine pass, about 0.6 µs a pass.** That is 3.83 µs a pass less 3.25 for
  the engine's decoding pass. The host, the memory index and the other streams'
  contention take it (inferred).

Each option is a guess until it is built and measured. One is a decode whose one-hots
stay at their level (the pick and the match in one node over nibble values, which needs a
different match than `[value == n]`). Another is word lines and fetch addresses that
change fewer channels a step, since PC's low bits change on every step and reach the most
pairs (`PCL.0` and `PCL.1`, 38 and 36 updates a pass). The others are byte cells for RAM
(census d5) for the write-line merge, and engine work on the per-pair cost.

## 8. Reproducing

The commands as they ran then. The 0/1 circuits, `GBCFORMER_BITS`, `f16census bits|bitsmem`,
`GBCFORMER_TRACE` and `examples/byte_census` have since been removed.

```sh
cargo build --release --all-targets
cargo test --release --test sm83_sst                 # both GBC CPU circuits against SST
cargo test --release --test checkpoint_sst           # both checkpoints against SST, the engine against the dense pass
cargo run --release --example f16census -- all       # T1 and T2 for every circuit
cargo run --release --example shape -- --mem         # units, hidden and nonzeros of the memory checkpoint
cargo run --release --example digest
cargo run --release --example export -- one-model <dir> --layer-shapes --align 64
GBCFORMER_GGUF=<dir>/model.gguf GBCFORMER_STATS=1 tools/bench/text.sh --weights all --one-model
GBCFORMER_UNITS=1 GBCFORMER_STATS=1 CORE=model-mem tools/bench/text.sh   # the update census
```

The machine is an Apple M1 Max, with a release build.
