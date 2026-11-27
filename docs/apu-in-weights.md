# APU in weights: one segment per pass, the samples as the sequence

This document puts the APU, waveform generation and mixing included, into Llama
weights (`CLAUDE.md`, "Each component in weights, and its gate"). Sections 1 to 10 are step 1 of that work, a
prototype that decided the APU's shape by numbers before any build. They cover the
quantum and why it is exact, the circuits (square 2 alone, then all four channels plus
the mix and the high-pass filter), their lockstep against `machine::apu`, their size
and their time on the engine, and the decision. The prototype was `examples/apu_proto/`.
Section 11 is the build that replaced it, the four-channel circuit in `src/apu_circuit/`
compiled to its own checkpoint and run as the machine's APU (`--weights apu`).

## 1. What the reference computes

`machine::apu` advances one M-cycle per `Apu::tick`, called by `Machine::advance` for
every M-cycle of every block, halted and HDMA cycles included. Each tick:

1. On a falling edge of system-counter bit 12 (DIV bit 4), the frame sequencer runs its
   next step (0 to 7), with length clocks on even steps, the sweep on steps 2 and 6 and
   the envelopes on step 7. The edge comes every 2,048 M-cycles, 8.57 per frame.
2. Each channel timer loses 4 T-cycles and reloads its period in a `while timer <= 0`
   loop, so one tick can wrap more than once when the period is shorter than 4
   T-cycles.
3. `sample_phase += 192,000`. When it reaches 4,194,304 it drops by that much and
   `push_sample` mixes one stereo sample.

The sample clock does not divide evenly. A sample comes every 21 or 22 M-cycles, 375
samples per 8,192 M-cycles, 803.67 per frame. `push_sample` maps each channel's 0-15
level to `2·level − 15` when its DAC is on, sums the channels NR51 routes to each side,
multiplies by NR50's volume + 1 and by 32, and runs a high-pass filter per side:
`filtered = in − (cap >> 8)`, `cap += filtered`, the sample is `filtered` clamped to
i16. Since §13 the filter is the host's presentation. `push_sample` stops at the mix
times the volume (`Mix`), and `HighPass` in `Machine::take_samples` multiplies by 32
and filters, whatever computes the APU.

The CPU changes the APU only through register writes, and those land between steps,
before the step's M-cycles advance. A write has side effects beyond the stored byte:
NRx4 bit 7 triggers, NRx1 loads the length counter, NRx2 can turn the DAC off, NR10 can
turn channel 1 off, NR52 powers the APU on or off, and a wave-RAM write while channel 3
plays lands on the byte being played.

## 2. The quantum: a segment

The host cuts the APU's M-cycle stream into **segments** at three kinds of event, all
of which the new `apu::Event` reports to a shadow (`Session::set_apu_shadow`):

- a register write starts a segment (applied first);
- a DIV edge starts a segment (applied after a write that starts the same segment,
  before any tick);
- a sample ends a segment.

So a segment is at most one write, at most one frame-sequencer step, then `D` M-cycles
of channel ticks (`D ≤ 22`), then at most one sample. One pass of the checkpoint runs
one segment. Its inputs are the APU state the previous pass produced plus four event
bytes (`WA` the write's address, `WV` its value, `EV` the flags write, edge and sample,
`D` the M-cycle count); its outputs are the new state and the sample.

**Why the segment is exact.** Inside a segment nothing happens but timer ticks, so
every field but the timers is the write's and the step's effect, computed once. The
timers have closed forms over `D` ticks:

- A timer at `u0` that loses `per` units per tick and reloads `P` wraps
  `n = #{j ≥ 0 : u0 + j·P ≤ per·D}` times, the `j = 0` term gated on `D ≥ 1`. This
  counts the reference's `while` loop exactly, including the reset state `u0 = 0` and
  periods shorter than a tick. At `P = 1` and `u0 = 0`, one tick gives `n = 2`, as the
  loop does. The timer after the ticks is `u0 − per·D + n·P`. Each test
  `per·D − u0 − j·P ≥ 0` is one threshold step, since `j` is a constant.
- Squares count in M-cycles, with `per = 1`, `P = 2048 − freq` and `n ≤ 23`. The duty
  phase moves `n mod 8`.
- The wave counts in units of 2 T-cycles (the period `2·(2048 − freq)` is even and the
  timer stays even), with `per = 2` and `n ≤ 45`. The position moves `n mod 32` and the sample register
  reads the nibble at the new position when `n ≥ 1`.
- Noise counts in M-cycles, with `P = (div == 0 ? 2 : 4·div) << shift`, frozen at shifts
  14 and 15. `P` is a product of two fields, so the wraps after the first exist only for
  the (div, shift) pairs with `P ≤ 22`, and each such wrap is one step per pair. The
  15-bit LFSR advanced `n` steps is `Mⁿ·lfsr` over GF(2) (the narrow mode is a second
  matrix); bit i is a parity of a subset of the old bits, one staircase per (n, mode).
- Channel 1's sweep changes the period at the frame-sequencer step. The DIV edge is a
  segment boundary for that reason. The step runs before the segment's ticks, so the
  ticks use one period.

`apu_proto` derives `D` from `sample_phase` (each tick adds 192,000, so the ticks since
the segment began are the difference divided by 192,000, checked to be whole), and it
checks that every timer is a whole number of units, every capacitor fits 24 bits, and
no segment exceeds 22 M-cycles.

**Passes per frame.** A frame has 803.6 samples, plus one pass per write and per DIV edge
that does not fall at a segment start. That gives 813 per frame on Yellow's boot, 824.3 on `text.sh`'s
presses (12.6 APU writes per frame, 9.5 of them NR32), 812 to 830 elsewhere (§4).

**Two inputs come from the host**, as IE and IF do for the CPU: the sample instants
(from `sample_phase`) and the DIV edges (from the timer). Both are Phase 3 control
logic. Moving `sample_phase` into the APU state costs one 22-bit adder and a
"ticks until the next sample" output; the DIV edge belongs to the timer in weights.

**Why not several samples per pass.** The high-pass filter is a floor recurrence:
sample k + 1 reads `cap >> 8` of the capacitor sample k wrote. N samples in one pass
need N filter stages in series, one level each, so the depth grows with N and the
engine's 14 circuit layers bound N to a handful. Samples as residual channels would
also repeat the timer and mix logic N times in distinct wires, the shape the PPU
prototype measured as quadratic in dense size (`docs/ppu-in-weights.md` §3). This
shape was not built, and the argument is the recurrence. Since §13 the recurrence runs on
the host, so this argument no longer applies, though the timer and mix repetition still
does. The samples are therefore the sequence, one pass per sample, each a new position
processed by the same weights, as the GBC CPU's steps are.

## 3. The circuits

Both circuits are built in `examples/apu_proto/` from `circuit.rs` primitives plus four
helpers in `build.rs`: `sum_mod` (bits of a weighted sum modulo 2^n, one staircase per
bit over the low weights), `sum_blocked` (the same in blocks, a level per block),
`sel` (a field keeps its value unless one of mutually exclusive cases loads another),
and `kill` (a bit cleared by cases that may overlap, by inclusion-exclusion).

**Square 2 plus the mix** (`square.rs`, `--ch2`) computes channel 2's write effects,
length, envelope and timer, the mix and the filter. Channels 1, 3 and 4 enter the mixer as the
reference's levels and DAC bits, three input bytes labelled as host-supplied; only
channel 2 is computed.

**All four channels** (`apu4.rs`, the default) carry a state of 68 fields
(`fields.rs`) packed into 60 bytes. They are the frame-sequencer step, power, NR50, NR51, both
capacitors, each channel's timer, phase or position, length, envelope or level,
frequency, channel 1's sweep (pace, direction, step, timer, enabled, negated, shadow
frequency), channel 3's 16 bytes of wave RAM and sample register, and channel 4's LFSR,
NR42 and NR43.

The levels, as `Circuit` assigns them:

| Level | Logic |
| - | - |
| 1 | which write (`WA` against each register, gated by power; DMG length loads while off; wave RAM); the values a write can load (`64 − v`, `256 − v`, `2048 − freq'` for a trigger, the noise period of the current NR43); channel 1's trigger overflow shift |
| 2 | every field after the write: one `sel` per field, power-off as one more case. 2,113 steps, the widest level |
| 3 | frame-sequencer conditions on the post-write state (length, envelope, sweep timer); the sweep's `shadow ± (shadow >> step)`; the square and wave wrap tests |
| 4 | the frame sequencer's effects (length, volume, envelope timer, step, sweep shadow and frequency); wrap counts; the noise wraps (after the sweep can no longer touch them) |
| 5 to 6 | the sweep's second overflow; phases and positions; the wave sample register (a 32-way nibble select); `n·freq` products; the noise LFSR (`Mⁿ`) |
| 7 | per side, the mix wires: `[route ∧ NR50 bit ∧ on ∧ level bit ∧ (duty, phase)]`, the volume multiply folded in as one literal, the duty waveform as one step per high (duty, phase) |
| 8 | per side, the filter input, a 16-bit sum of the mix wires |
| 9 | the sample `in − (cap >> 8)` and the capacitor's low 12 bits, both gated by the sample flag; the timers after the ticks (blocked) |
| 10 | the capacitor's high 12 bits |

Two compile bounds shaped the arithmetic. `engine::IntLayer` refuses a unit whose
Σ|weight| reaches 2^23 and a wire whose down sum reaches 2^24 (the dense pass is f32).
A negative weight modulo 2^(k+1) is close to 2^(k+1), so `sum_mod` writes `−w·x` as
`w·(1 − x) − w`; and the 24-bit capacitor and the 20-bit noise timer are summed in
blocks (12 and 10 bits). Two first builds hit the down-sum bound (the channel 2 filter's negative weights, then the noise timer) and compiled with
the fast path off; no engine change was needed once the circuit respected them.

| Circuit | Inputs, outputs | Wires | Nodes | Steps | Depth | Steps per level |
| - | - | - | - | - | - | - |
| square 2 + mix | 23, 20 | 665 | 481 | 1,517 | 11 | 123, 350, 103, 155, 76, 113, 167, 52, 108, 138, 132 |
| four channels + mix | 64, 64 | 2,104 | 1,592 | 7,055 | 10 | 353, 2113, 397, 789, 752, 1067, 609, 647, 167, 161 |

The largest four-channel families by steps are the mix wires 936, wave RAM after the write
640, LFSR 597, length and envelope 574, square timers 564, filter inputs 253 per side,
sweep shifts 240 (`apu_proto --families`).

## 4. Exactness (measured)

`apu_proto <rom> --frames N` installs the shadow, runs every segment through the
circuit (`Circuit::eval`) on the state its previous segment produced, and compares the
whole state and the sample with the reference after every segment; the first
difference stops the run and names the fields that differ. `--engine` also runs the
compiled checkpoint and holds its tokens to the circuit's; `--engine-only` lets the
checkpoint alone answer and holds it to the reference.

| Circuit | Run | Answered by | Segments equal to the reference |
| - | - | - | - |
| square 2 + mix | Yellow CGB, frames 0–1200, first segmentation | circuit | 967,247 (964,378 samples) |
| square 2 + mix | Yellow CGB, `text.sh` presses, frames 0–3500, DIV edges inside segments (the first segmentation, where the edge commutes with channel 2's ticks) | circuit and engine | 2,855,013 |
| square 2 + mix | Yellow CGB, `text.sh` presses, frames 0–3500, DIV edges as boundaries | engine | 2,884,996 |
| four channels | Yellow CGB, frames 0–300 | circuit | 243,950 |
| four channels | Yellow CGB, `text.sh` presses, frames 0–3500 | circuit | 2,884,996 (2,812,773 samples) |
| four channels | Yellow CGB, `text.sh` presses, frames 0–3500 | engine | 2,884,996 |
| four channels | Yellow CGB, `overworld.sh` presses, frames 0–7800 | engine | 6,388,326 (6,268,468 samples) |
| four channels | Yellow DMG, frames 0–1200 | engine | 977,513 |
| four channels | blargg `dmg_sound` 01–04, DMG, 2,000 frames each | circuit; engine | 1,659,451; 1,625,201; 1,625,252; 1,624,564, each run both ways |

The four-channel circuit runs predate §6's choice of 6-bit blocks for the square and
wave timer sums; the engine runs use the final circuit. Both compute the same sums.

The blargg ROMs report on screen; every run reached the gate's frame hash (`99793856`,
`881f3fe7`, `e494d943`, `8e776b75`), which is evidence only together with the lockstep.
The CPU reads NR52 and the length counters from the reference, and the lockstep shows
that the circuit's state equals the reference's after every segment, so a core that
served the reads from the circuit would read the same bytes.

On `text.sh` the runs exercised writes to every mapped register from FF10 to FF3F
(19 to NR10, 33,391 to NR32, 3,409 to NR50, 190 to each wave-RAM byte),
seven NR52 writes, and the intro's sweep and noise. On blargg they exercised power cycles, the length
counter's extra clock, trigger corner cases and sweep overflow on DMG. Paths none of
them reached are exact by construction only. These are noise periods of 2 M-cycles (11 wraps a
segment), square frequency 2047 (23 wraps), a trigger and a DIV edge in one segment
with channel 1's sweep firing.

## 5. Size (measured)

| Circuit | Hidden | Intermediate | Layers | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - |
| square 2 + mix | 1,024 | 384 | 12 | 23 | 6,187 | 63.0 M | 49,534 |
| four channels + mix | 3,072 | 2,176 | 11 | 64 | 16,768 | 600.5 M | 215,244 |

The heads are layer 0's input gathers, one per input slot. Neither checkpoint has a
memory head. The four-channel 600.5 M splits into embeddings and LM head
(2 · 16,768 · 3,072 = 103.0 M), attention (11 · 4 · 2,048 · 3,072 = 276.8 M) and MLPs
(11 · 3 · 3,072 · 2,176 = 220.6 M). Attention is the largest part and does nothing past
layer 0, because `Config` gives every layer the same head count, so 64 gathers cost 64 heads in
all 11 layers. Hidden 3,072 is the 2,104 wires plus 128 positions and the fixed
channels (`llama::Layout::used`).

Projections (inferred from `Config::params`, not compiled):

| Variant | `d` | Heads | `i` | Layers | Vocab | Dense parameters |
| - | - | - | - | - | - | - |
| APU alone, as measured | 3,072 | 64 | 2,176 | 11 | 16,768 | 600.5 M |
| APU alone, layer 0 gathering 4 slots per head (an emitter change: a byte is 8 of a head's 32 value dims) | 3,072 | 16 | 2,176 | 11 | 16,768 | about 393 M |
| APU merged with the CPU and memory (unit I: 1,726 wires, hidden 2,048, intermediate 2,816, 10 layers, 266.1 M, about 23 heads inferred from its parameter count) | 5,120 | 87 | about 3,600 | 11 | about 24,500 | about 1.5 B |
| merged, with 4-slot gathers | 5,120 | 39 | about 3,600 | 11 | about 24,500 | about 1.1 B |

For the merged rows, `d` is `Layout::used` of both circuits' wires and positions
(2,041 + 2,104 + 128) rounded up to 1,024; `i` is the widest level of the two circuits'
summed steps level by level (CPU 2,611 at level 8 plus APU 647: 3,258 steps) times the
1.08 units per step both checkpoints compile to; layer 0's heads add. Nonzeros add:
147,525 + 215,244 = about 363 k.

## 6. Time (measured)

The engine runs every segment as one pass on the template path, incrementally from the
previous segment, every pass held to the reference. The table has clean runs (no census), each a
single run with other work on the machine, with the 1-minute load.

| Circuit | Workload | Passes per frame | Units per pass | µs per pass | ms per frame | Load |
| - | - | - | - | - | - | - |
| four channels | `text.sh` presses, frames 0–3500 | 824.3 | 3,649 | 11.00 | 9.1 | 10.0 |
| four channels | `overworld.sh` presses, frames 0–7800 | 819.0 | 3,850 | 11.81 | 9.7 | 8.5 |
| four channels | Yellow DMG, frames 0–1200 | 814.6 | 2,573 | 7.94 | 6.5 | 8.2 |
| square 2 + mix | `text.sh` presses, frames 0–3500 | 824.3 | 1,449 | 4.65 | 3.8 | 8.5 |

Units per pass are deterministic; the µs are not. Across these runs the same
four-channel checkpoint measured 7.94 (DMG boot) to 11.81 µs (overworld) per pass in clean runs and up to 13.7 with the census on. The load ran from 8 to 17,
the census (`GBCFORMER_UNITS=1`) adds about a nanosecond per update, and boot frames
change fewer units than music does (1,459 per pass over Yellow's first 120 frames,
3,649 over `text.sh`).

**Where the units go** (`text.sh`, census run, gate updates per pass by node family). The
noise timer's low block has 264, the wave `n·freq` products 244, each capacitor's low block 225,
each filter input 161, the wave position 118, each sample 97, the wave sample register
91, the square timers 63 to 88 per block, the LFSR 80, the mix wires 35 per (side,
channel). By layer they are 105, 138, 337, 329, 288, 764, 256, 548, 622, 228, 33. The timers and the filter
change on every pass by construction. A timer moves by `D` every segment and the
capacitor moves every sample while the input is not constant. Their staircases read
every lower bit of the sum, so a changed low bit updates every higher bit's steps.
Blocking the square and wave timer sums at 6 bits took 7% of the units off at the same
depth (2,743 to 2,556 on frames 0 to 1200); smaller blocks for the noise timer and the
capacitor took off 10% more but cost an eleventh level, +45 M dense parameters.

**Against the budget.** The APU thread has about 16.7 ms per frame. Four channels with
the mix take 6.5 to 9.7 ms per frame at these loads, 40 to 58% of the budget. On the CPU's thread
they would add those 6.5 to 9.7 ms to a frame that unit I's CPU pass (5.95 µs at 5,350 steps
per frame on `text.sh`, `docs/perf-log.md` Unit I) already holds at 32 ms.

## 7. Placement: state as tokens, writes as events, no shared cache

The prototype carries everything the APU holds as state tokens, the registers the CPU
writes included, and feeds each write to the pass as an event. That is not a
convenience but what the hardware asks for:

- A write's effect is an event, not a stored value. Two identical NR24 writes trigger
  twice; a record that holds "the newest byte at FF19" cannot tell the APU that a
  second trigger happened. The APU needs the write stream either way.
- Wave RAM's address depends on APU state. While channel 3 plays, every FF3x address
  reaches the byte being played. A memory record would need the CPU's address
  translation to read channel 3's position.
- NR52's status bits, the length counters and the envelope volume are APU state the
  APU changes itself.

With state as tokens and writes as events, the APU needs no memory heads (§5's
checkpoints have none). **Same checkpoint is then a choice, not a requirement.** The
PPU must share the CPU's KV cache to read VRAM and OAM (`docs/ppu-in-weights.md` §5);
the APU only needs the CPU thread's stream of FF10-FF3F writes, each with its M-cycle
time, over a channel, as the PPU thread receives VRAM writes. Its own checkpoint keeps
it at 600 M (393 M with 4-slot gathers) instead of growing the CPU's to 1.1 to 1.5 B,
and its thread runs its own `Engine` state, incremental from segment to segment.

Two bridges this leaves to Phase 3:

1. **CPU reads of the APU.** NR52's status, wave RAM and the NRx read masks come from
   APU state. In two threads, a CPU read of FF10-FF3F must see the APU caught up to
   that M-cycle. Either the APU thread publishes its readable bytes after each segment
   and the CPU thread waits for the segment that covers the read, or those reads stay
   with the host's shadow until a synchronisation design exists. How often Yellow
   reads these registers was not measured.
2. **The sample clock and the DIV edge** (§2), host inputs today.

## 8. Decision

**The numbers are good. Extend to all four channels (done) and take this shape to the
build.**

- **Quantum.** The quantum is the segment (a write, a DIV edge, up to 22 M-cycles of
  ticks, a sample), exact by closed forms, at 812 to 830 passes per frame.
- **Samples as the sequence, one per pass.** The filter's recurrence forces this, so the
  wide shape is out.
- **Four channels as wires in one pass.** This gives 2,104 wires, depth 10, 600.5 M dense,
  215,244 nonzeros, 6.5 to 9.7 ms per frame. Channels as separate passes (one circuit
  shared by four passes) were not built. A pass that switches channels changes every
  state input, so nearly every unit re-evaluates, and the pass count would quadruple
  for a model about a third the width (inferred).
- **Placement.** The APU gets its own checkpoint on its own thread, fed the CPU's APU
  writes, with no shared cache.
- **Engine.** No change is needed. One emitter change is worth making before the build,
  layer 0 gathering several input slots per head, which takes the standalone APU from
  600 M to about 393 M and helps every checkpoint with many input slots.

The next units of work, in order, are the CPU-read bridge (§7), the 4-slot gather in the
emitter, and the build of `apu4` as a core (`CheckpointDevice` beside `apu.rs` in
`lockstep::Pair`, the same lockstep the prototype runs).

## 9. Gates

The source change is `apu::Event`, `Apu::shadow` and `Session::set_apu_shadow`, with the
channel structs and their fields made public for the shadow to read. Run on the committed
tree with `tools/gate.sh`'s steps; the worktree holds no ROMs or `vendor/`, so the gate
ran from a copy of the script with absolute paths, and `cargo test --release` ran as
`cargo test --release --no-run` with each test binary started from the main checkout,
where `vendor/sm83/v1` lives.

| Gate | Result |
| - | - |
| build, no warnings; wasm32 check of the library | ok |
| `cargo test --release` (7 test binaries: lib 110, `sm83_sst` 3, `checkpoint_sst` 5, `memory_dense` 2, `e01_primitives` 2, `interrupt_fuzz` 3, bin 5) | ok |
| `run --frames 8 --core lockstep --hardware dmg` / `--frames 300 --core lockstep` | `6872b003` / `cfd31f3d` |
| Crystal, Zelda DX, Mario DX, 1,500 frames lockstep | `4e6a7509`, `1d00d871`, `7dec3947` |
| text.sh, overworld.sh | `870aa3fd`, `822059c1` |
| blargg `cpu_instrs`, `instr_timing`, `dmg_sound` 01–04 | Passed; `99793856`, `881f3fe7`, `e494d943`, `8e776b75` |
| dmg-acid2, cgb-acid2 | pixel-exact |
| host checkpoint (`examples/digest`) | `fe67fe79`, unchanged |
| `model` core no slower (ab.sh against the build before, 3 runs, load 12.7 to 10.4) | text.sh 2.57 → 2.53 µs per pass, overworld.sh 3.13 → 3.12: a tie (perf-log, "Phase 2 prototype") |

## 10. Reproducing

The prototype's example, its square-2 circuit, the block-width knobs and the shadow
hook were removed by the build (§11), which runs the same four-channel lockstep through
`run`. The prototype's commands, kept as the record (they no longer run):

```sh
cargo build --release --all-targets
./target/release/examples/apu_proto pokemon-yellow.gbc --frames 3500 [--press ...]          # four channels, circuit
./target/release/examples/apu_proto pokemon-yellow.gbc --frames 3500 --engine-only [--press ...]
./target/release/examples/apu_proto pokemon-yellow.gbc --frames 3500 --ch2 --engine [--press ...]
./target/release/examples/apu_proto vendor/blargg/dmg_sound/01-registers.gb --frames 2000 --hardware dmg
./target/release/examples/apu_proto pokemon-yellow.gbc --frames 1 --families                # circuit statistics
APU_T2_BLOCK=4 APU_T4_BLOCK=5 APU_CAP_BLOCK=6 ./target/release/examples/apu_proto ...      # the §6 block study
```

The presses are `text.sh`'s (`--press F:a:4 --press F+20:down:4` for F = 2300, 2340, …,
3500) and `overworld.sh`'s.

## 11. The build: the APU checkpoint as the machine's APU

`--weights apu` (alone, or with `cpu` and `memory`) compiles the four-channel circuit
into its own checkpoint and hands the machine's APU to it at the hand-off; `--lockstep
apu` runs `machine::apu` beside it. Every sample the speaker plays in `play` and every
sample `run --wav` writes then comes from the model, and so does every byte a CPU read
of FF10-FF3F returns. Two inputs stay with the host, labelled in every run. They are the sample
instants (the 48 kHz phase) and the DIV edges (the timer's counter), as IE and IF do
for the CPU. Both are Phase 3 control logic.

### 11.1 Shape

`src/apu_circuit/` holds the circuit (`circuit.rs`, `build.rs` and `fields.rs`, moved
from the prototype with the block widths fixed at 6, 10 and 12 and no environment
knobs), the slot ABI and the devices (`mod.rs`) and the thread (`worker.rs`).

- **ABI.** `Apu4` implements `checkpoint::SlotAbi` with 65 inputs (the 60 state bytes,
  then `WA WV EV D RA`), no request slots, no late slots, and 65 results (the next state,
  `SL0 SL1 SR0 SR1`, `RV`). `CheckpointDevice<Apu4, Engine>` runs it through the generic
  protocol. `needs_late` is always false, so every segment is one pass, settled in pass 1.
- **Device.** `ApuModel<E: Segments>` implements `Device` and `AudioDevice`. Per M-cycle
  it follows `Apu::tick`. A falling edge of DIV bit 12 joins the open segment or, after
  ticks or another edge, closes it first. The tick follows, and then the sample clock,
  which closes the segment with a sample. A register write closes the open segment and starts
  the next. It carries the DIV bit across blocks as `apu.rs` does, so the edge a DIV
  reset causes is seen. At the hand-off it takes the reference's state, sample phase,
  DIV bit and undrained samples.
- **Lockstep.** `Pair` compares reads and `Signals`, and the APU raises none, so it would
  prove nothing about the state. `ApuLockstep` ticks the reference one M-cycle at a time
  between the model's edge and sample events and, after every pass, compares the whole
  60-byte state (naming the fields that differ), the sample, and every read's byte.
- **Machine.** The APU slot holds the reference or the `AudioDevice` that replaced it
  (`ApuSlot`); `Session::with_stack` installs the device. The shadow hook
  (`apu::Event`, `Apu::shadow`, `Session::set_apu_shadow`) and `examples/apu_proto` are
  gone; the device lockstep replaces them.

### 11.2 The read bridge

The CPU's reads of FF10-FF3F come from the model's state at the M-cycle of the read.
Every bus read of that range (`Machine::read_synced`, also used by DMA sources) calls
`AudioDevice::sync(addr)` first, which closes the open segment with `RA` set to the
read's address. One pass runs the ticks so far and returns `RV`, the byte the read
sees. `RV` is in weights, as one node per bit over the post-segment fields, with the
register read masks, NR52's power and channel status, the unmapped addresses' 0xFF,
and wave RAM through channel 3's position while it plays (`ram[pos3 >> 1]`) or by
address while it does not. It lands at level 5, adding 448 steps to that level and no
new level. `Device::read` without a sync at that M-cycle panics, so no read can be
served from stale state.

How often the CPU reads these registers, measured over the 600 model frames of each
workload, is 0.61 reads per frame on `text.sh`, 0.23 on `overworld.sh`, 0.28 over Yellow's
first 1,200 frames, and 6.06 per frame on blargg `dmg_sound` 01 (which polls NR52 and the
length counters). The bridge costs one pass per read, under 0.1% of the segments on
Yellow.

### 11.3 Placement: its own checkpoint, on its own thread

The APU needs no memory head (§7), so its checkpoint is separate from the CPU side's.
On the CPU's thread it cost overworld.sh half its speed (82.8 fps without the APU, 43.1
with it inline, medians below), because the APU's 815 passes per frame add about 11 ms
to a frame the CPU side holds at 12 ms. `worker::Worker` therefore runs
the device on a thread of its own. The machine's thread queues writes and advances in
batches (256 commands or 1,024 M-cycles; consecutive advances merge, so a halted stretch
is one command) and keeps running up to 36 batches (about two frames) ahead. It waits
for the worker only at a read of FF10-FF3F; samples come back as the worker mixes them,
and `Session::finish_audio` (which `run` calls inside the timed span) waits for the
rest. A lockstep divergence panics on the APU thread and the machine's thread re-raises
it at its next send or wait, so the first divergence still stops the run (tested by
`a_divergence_on_the_worker_stops_the_caller`, and by a deliberate NR52-mask mutation
that stopped blargg 01 at pass 85,472 with exit 101). `GBCFORMER_APU_THREAD=0` keeps the
APU inline; wasm32 always does. Audio in `play` may reach the speaker up to two frames
late under the worker.

The full stack, measured (`tools/bench/ab.sh`'s procedure on scratch copies of
`text.sh` and `overworld.sh`, 600 model frames each, interleaved, 3 runs per side, one
binary; 1-minute load 8.3 → 5.5 for the first pair of rows, 5.5 → 5.9 for the second):

| Stack | text.sh fps (runs; median) | overworld.sh fps (runs; median) |
| - | - | - |
| `cpu,memory` (APU on the host) | 30.1, 30.9, 31.6 (30.9) | 82.8, 84.7, 81.0 (82.8) |
| `cpu,memory,apu`, APU thread | 27.9, 29.6, 30.5 (29.6) | 79.7, 46.5, 79.4 (79.4) |
| `cpu,memory,apu`, APU thread (second pair) | 23.7, 30.5, 30.3 (30.3) | 70.1, 80.6, 78.7 (78.7) |
| `cpu,memory,apu`, APU inline (`GBCFORMER_APU_THREAD=0`) | 22.9, 22.9, 23.2 (22.9) | 44.4, 38.6, 43.1 (43.1) |

With its own thread the APU costs 2 to 4% of the CPU side's fps (−1.3 and −3.4 fps at
the medians) instead of 26% on `text.sh` and 48% on `overworld.sh` inline. With two
threads a frame takes about max(CPU side, APU) plus the waits at reads. The CPU side
holds 12 ms per frame on overworld.sh and 33 ms on text.sh, and the APU about 11 ms
(the inline difference, 1/43.1 − 1/82.8 s). So overworld.sh is the workload where the
two are closest and a busy core shows. The low runs (46.5, 70.1, 23.7) came
with other runs on the machine, most likely the APU thread waiting for a core
(inferred, not profiled). The wall time per CPU pass rises 4% at the medians (from 6.03 to 6.29 µs on
text.sh, from 6.10 to 6.37 on overworld.sh). The cause is the queueing on the CPU's
thread, the waits at reads and the tail `finish_audio` waits for, not separated here.
Against the 16.7 ms frame budget the APU alone fits with room, at 11 ms on overworld.sh.

### 11.4 Size (measured)

`cargo run --release --example shape -- --apu` reports 2,153 wires, 1,633 nodes, depth
10 and steps per level 386, 2,113, 397, 789, 1,200, 1,067, 673, 647, 167, 161. The
checkpoint has hidden 3,072, intermediate 2,176, 11 layers, 65 heads, vocab 17,026,
606.4 M dense parameters and 219,387 nonzeros, with the integer fast path on and no
memory head. Against the
prototype (600.5 M, 215,244), the readback adds one input slot (one layer-0 gather
head, which `Config` repeats in all 11 layers, 4.3 M), 258 vocabulary tokens (1.6 M)
and 4,143 nonzeros.

With layer 0 gathering 4 slots per head (the emitter change another unit is making in
`llama.rs`, not used here), 65 slots fit in 17 heads, attention is 11 · 4 · 544 · 3,072 = 73.5 M
instead of 281.2 M, and the checkpoint is about 399 M dense (inferred from `Config::params`, not
compiled).

### 11.5 Gates (measured)

| Gate | Result |
| - | - |
| blargg `dmg_sound` 01–04, `--weights apu` (the model alone generates the sound and answers every read) | `99793856`, `881f3fe7`, `e494d943`, `8e776b75` |
| the same, `--lockstep apu` | the same hashes; 1,671,562, 1,647,622, 1,654,858 and 1,634,451 passes equal to the reference |
| `text.sh`, `--weights cpu,memory,apu --lockstep all`, 600 frames | `870aa3fd`; 489,745 APU passes equal |
| `overworld.sh`, the same | `822059c1`; 488,590 APU passes equal |
| Yellow DMG 8 / CGB 300, `--weights cpu,apu --lockstep all` | `6872b003` / `cfd31f3d` |
| Yellow CGB 300 / DMG 1,200, `--weights apu --lockstep apu` | `cfd31f3d` / `c941c0f4`, no divergence |
| Yellow frames 0–1,200, `--weights cpu,memory,apu --wav` against `--weights none --wav` | byte-identical WAV (964,379 samples) |
| unit tests: random register traffic through the circuit in lockstep (DMG and CGB, 6,000 events each), reads, the worker's order and its divergence | ok |

The rest of the suite on the final binary, from scratch copies of `tools/gate.sh` with
absolute paths (the worktree holds no ROMs or `vendor/`; each test binary ran from the
main checkout) gave these results: build without warnings and the wasm32 check of the
library ok;
`cargo test --release` ok in all 8 binaries (lib 115, `sm83_sst` 3, `checkpoint_sst` 5,
`memory_dense` 2, `e01_primitives` 2, `interrupt_fuzz` 3, `dma_records` 1, bin 5);
`--core lockstep` DMG 8 `6872b003` and CGB 300 `cfd31f3d`; Crystal, Zelda DX and Mario DX
1,500 frames lockstep `4e6a7509`, `1d00d871`, `7dec3947`; `text.sh` `870aa3fd` and
`overworld.sh` `822059c1` on the `model` core; blargg `cpu_instrs` and `instr_timing`
Passed and `dmg_sound` 01-04 on the host the same four hashes; dmg-acid2 and cgb-acid2
pixel-exact; the CPU checkpoint's digest `fe67fe79`, unchanged. The `model` and
`model-mem` cores are no slower (`docs/perf-log.md`, "Phase 2: the APU checkpoint").

### 11.6 Merging into one checkpoint

The long-term goal is one model for the whole Game Boy. The ABI keeps that open. The
APU's slots are a plain `SlotAbi` with no late inputs, and its circuit shares no wire
with the CPU's. What a merge would take:

1. **One quantum for two clocks.** A CPU pass is one step of 1 to 6 M-cycles and does not
   run while the CPU halts; an APU pass is a segment of up to 22 M-cycles that ends at a
   sample and runs through HALT. A merged sequence would interleave two kinds of pass,
   CPU steps and APU segments, over the same weights. The engine's incremental
   evaluation makes that affordable only because the circuits are disjoint. An APU pass
   changes only APU inputs, so only APU units re-evaluate (3,900 per pass on `text.sh`),
   and a CPU pass leaves them alone. Running the APU inside the CPU's steps instead
   (segments cut at every step) would multiply its passes by six on `text.sh`.
2. **One slot layout.** `llama.rs` lays out one slot list per checkpoint, so the merged
   checkpoint concatenates the CPU side's inputs and results with the APU's 65 and 65,
   and the residual stream holds both circuits' wires. Hidden is about 5,120, intermediate
   is the widest level of the summed steps, and the size is about 1.5 B dense today and
   1.1 B with 4-slot gathers (§5, inferred). The memory checkpoint's KV cache must not take APU passes as
   write records.
3. **One thread.** One engine state cannot run on two threads, so a merge gives up the
   worker's overlap, and overworld.sh would fall from about 80 fps to the inline figure
   above, unless the engine learns to evaluate the two wire ranges concurrently.
4. **The bridge in the sequence.** Merged, the read that `sync` closes is a pass in the
   same sequence as the CPU step that reads it. Today the host still copies `RV` into
   the CPU's late data slot, as it serves every I/O-page read; with the I/O page in
   weights (Phase 3) the CPU could read the APU's `RV` position by attention, which is
   the payoff that would justify the merge.

## 12. Smaller and cheaper per pass (measured)

This section is a hillclimb on two numbers, the work per pass (gate updates and instructions retired,
both independent of the machine's load) and the dense size. The segment, the ABI's
meaning and the lockstep are unchanged; every gate of §11.5 held on the final tree.

| | Before | After |
| - | - | - |
| Dense parameters, nonzeros | 145.9 M, 223,371 | 98.7 M, 177,082 |
| Hidden, intermediate, layers, heads, vocab | 1,024, 2,560, 11, 17, 17,026 | 1,024, 1,024, 13, 15, 15,736 |
| Input slots, decoded positions | 65, 65 | 60, 60 |
| text.sh units, instructions per pass | 3,896, 152,481 | 2,639, 116,105 |
| overworld.sh units, instructions per pass | 4,397, 167,906 | 2,836, 122,637 |

Instructions per pass are the process's retired instructions over the 600 model frames,
minus a one-model-frame control run (warmup and compile), minus the same difference with
the APU on the host, over the passes (`/usr/bin/time -l`; repeatable to 0.02%). Passes
per frame are unchanged (816.2 on text.sh, 814.3 on overworld.sh), so text.sh's APU
thread goes from 124.5 M to 94.8 M instructions per frame (−24%) and overworld.sh's from
136.7 M to 99.9 M (−27%).

What changed, in the order it landed:

1. **Schedule and fit.** The circuit was never scheduled, so layer 1's write muxes (2,114
   units plus about 400 clearing units) set intermediate 2,560. `schedule` alone overran
   the 845 wire channels of hidden 1,024 (850 live); `fit` toward a recorded change table
   (`changes.txt`, the mean of text.sh and overworld.sh) fits them. `fit` moves nothing
   while the wires already fit, so the build tries channel targets 1 and 16 below the
   scheduled live count and keeps the narrowest MLP, then the fewest changed pairs.
2. **The noise's later wraps summed first.** At most one (div, shift) case is live, so
   the 33 later-wrap wires sum to at most 22, five bits in the timer instead of 33 terms.
3. **The capacitor is cap + S.** S is computed anyway; the old form added the mix, the
   capacitor and `−(cap >> 8)`, three terms a bit. T2 superseded this (`docs/perf-log.md`,
   "T2 depth"). The circuit now sums S and cap' side by side from the mix as `64A − 480B`
   and input-only shares `F = −(cap >> 8)` and `G = cap + F`, so the capacitor no longer
   waits on S.
4. **n·P without n·freq.** A pitched channel's wrap count toggles 0/1 nearly every pass,
   and each toggle re-evaluated the n·freq product and its 16-bit staircase. The first
   wrap adds P = 2048 − freq through one AND per bit; later wraps exist only when P is at
   most 22 (44 for the wave), so their m·P is a 5- or 6-bit product.
5. **The state packed end to end.** 438 bits in 55 bytes instead of 60, with the fields most
   passes change first. That leaves 60 input slots, so 15 gather heads, vocab 15,736 (vocab is
   `257 · inputs + 256 + outputs`, a token set per input slot) and 5 fewer decoded
   positions.
6. **4-bit timer blocks**, one level more, and with it the whole capacitor as two 12-bit
   blocks of cap + S.
7. **Short-period gates.** Two or more LFSR steps, and channel 2's later wraps, need a
   period under 22 M-cycles; those cases read copies gated by that condition, which are
   0 for any pitched note, so a pass leaves them alone. Channel 1 (its frequency comes
   after the sweep) and the wave would pass the engine's 13 circuit layers with the gate.
8. **The wave sample through a one-hot position**, so a move changes two wires instead
   of every step of the 32-way select; phase and position add the wrap count.

On text.sh, per pass, the filter (capacitor 198 + 63, filter
input 191, sample 118, per side) is 1,140 of 2,639 updates, 43%. The timers' blocks take about 450 and
the mix wires 200. The filter changes on every sample while a channel plays, so the
next step for it is fewer samples' worth of passes, not cheaper ones (§2: two samples
per pass need the capacitor twice in series and the mix at the first sample's instant).

Two changes were not kept: 8-bit capacitor blocks (−3.0% instructions for one more layer, +2.7 M) and
the noise channel's timer and LFSR frozen while it is off (they are unobservable then,
since a trigger reloads both, but the lockstep compares the whole state with the
reference, which keeps ticking them; that is a change to what the gate checks, not
made here). The worker thread now asks macOS for the interactive QoS class so the
scheduler keeps it on a performance core; at the machine's load (16 to 51) the A/B could
not measure it.

## 13. The high-pass filter as presentation (measured)

The capacitor is the board's, after the chip (`docs/hardware-shaped.md` §13.5, a
project decision), so it moved out of the weights. The checkpoint's sample slots carry the chip's
digital output, each side's DAC sum times NR50's volume (`machine::apu::Mix`), and
`HighPass` in `Machine::take_samples` filters it on the host whatever computes the APU.
The state drops `capL` and `capR`, going from 438 to 390 bits and from 55 to 49 bytes,
with 55 input and 55 result slots (50 state bytes since the wave RAM rule's `jr3`, `docs/hardware-tests.md` §3.1). `ApuLockstep` compares state, mix and read after every pass. Every WAV is
byte-identical to before, on the host and in weights.

| | Before | After |
| - | - | - |
| Wires, nodes, depth | 2,319, 1,839, 13 | 2,081, 1,649, 12 |
| Dense parameters, nonzeros | 90.1 M, 150,077 | 80.2 M, 135,817 |
| Hidden, intermediate, layers, vocab | 1,024, 960, 14, 15,736 | 1,024, 896, 13, 14,188 |
| text.sh, overworld.sh updates per pass | 2,818, 3,026 | 1,904, 2,043 |

§12's table is from an earlier tree; before A4 text.sh measured 2,818 updates a pass,
of which the filter's families (`GL`/`GR` 162 a side, `cap` 140, `FL`/`FR` and
the old 16-bit `SL`/`SR`) were the largest after the mix sums `AL`/`AR`, which stay.
§2's argument against several samples per pass no longer rests on the filter.

## 14. M-cycles in, the tick's phase and the frame sequencer's step in the state

The host used to map each block's M-cycles onto 4-dot ticks (`device::Clock::ticks`),
keep the tick's phase across blocks at double speed (`Machine::apu_phase`, reset at
every STOP), and find the frame sequencer's edges from a DIV counter it rebuilt from the
control model's (`Clock::sequencer`). Now the APU takes M-cycles, the speed and the
step, and converts them itself.

- **Inputs.** `D` is M-cycles (at most 22, or 44 at double speed), and `EV` bit 3 is
  the speed, the control logic's KEY1 bit carried over. `EV` bit 1 says the frame
  sequencer's counter bit fell at the segment's first M-cycle; the control model emits
  that instant (`docs/control-in-weights.md` §22).
- **State.** The two spare bits of the 49 bytes hold `half` (one M-cycle of the current
  tick has passed at double speed) and `pend` (a step whose tick has not finished). The
  ticks are `(half + D) >> 1` at double speed and `D` at normal speed, where the phase
  resets, so a half tick left by a switch from double speed is dropped as before. Bit b
  of the ticks is a staircase on `half + D` gated by the speed, one level beside the
  write decode, so depth stays 12. The step runs before the first tick the segment
  finishes; with none it waits in `pend`. With no M-cycle the phase holds.
- **Host.** `ApuModel` counts M-cycles into the open segment, closes it at a step, a
  sample, a read, a write, or a change of speed, and keeps nothing but the sample clock.
- **Reference.** `Apu::mcycle` keeps the same phase and step, and `ApuLockstep`
  compares both fields with the rest of the state after every pass.

The sample clock is the host's 48 kHz of real time (`docs/host-in-weights.md` §2, item
16). It used to count finished ticks, which the host can no longer see. It now counts
per M-cycle, two dots of real time at double speed, so at double speed a sample can
fall half a tick earlier, in the middle of a tick, and mix the state after the ticks
finished so far. At normal speed the instants are as before. `run --wav` against the
binary before this change, 300 frames on the host: Yellow and Mario DX byte-identical,
blargg `dmg_sound` 01 (600 frames, DMG) byte-identical, Zelda DX (double speed) 185,064
of 964,420 bytes differ. All of it is the sample instants: a build whose reference
counts the sample clock per finished tick again gives Zelda DX's WAV byte for byte, so
the phase and the step placement match the old reference there. Mario DX's 300 frames
hold little sound (14,908 non-zero data bytes), so its equality says little. The other change is the reset of the phase at a STOP that does
not switch speed at double speed, which the reference no longer makes; no game in the
gates runs one.

| | §13 | §14 |
| - | - | - |
| Wires, nodes, depth | 2,081, 1,649, 12 | 2,089, 1,657, 12 |
| Hidden, intermediate, layers, vocab | 896, 896, 13, 14,188 | 896, 960, 13, 14,188 |
| Dense parameters, nonzeros | 70.2 M, 135,817 | 72.4 M, 137,587 |

The five tick bits, `ticks>0`, the step, `pend'` and `half'` are the new nodes. The
widest level grows from 804 to 841 units, which takes intermediate one quantum up.
`src/apu_circuit/changes.txt` is re-recorded from text.sh and overworld.sh on
`--weights apu`. `examples/digest`: apu `bf499b44` (from `83c6b259`). The circuit takes any
input bytes as before: the step is the OR of `EV.1` and `pend` before the tick test, so
the one model's stream audit, which feeds random APU states, sees only 0 and 1.
`the_lockstep_catches_a_wrong_tick_phase_or_waiting_step` flips `half` and then `pend`
after passes, and the lockstep stops both. Every gate holds, `dmg_sound` 01-04 with the
APU and control in weights and in lockstep among them (`docs/control-in-weights.md`
§22.4).

## 15. SameBoy's rules: 2 MHz cycles, frame-sequencer rises, PCM12 and PCM34

The reference became a port of SameBoy v1.0.5's `Core/apu.c` (commit ea667ad, Expat
licence, `licenses/SameBoy-Expat.txt`), so SameSuite's 63 APU cases pass
(`docs/hardware-tests.md` §3.12). The circuit's structure follows the port: each rule
of `Apu::write`, `div_event`, `div_secondary_event` and `run` is a case or a closed form
here. §§1-14 describe the circuit this replaced.

### 15.1 The segment

A pass is still one segment, now in three phases over the carried state:

1. The write that opens it (`write.rs`). Each register's effects are cases muxed into
   the post-write state. A second stage sets the values that read post-write fields: a
   sample re-latched after an NRx2 glitch or a trigger, and the byte a wave trigger reads
   after DMG's corruption.
2. The frame-sequencer events at its first M-cycle (`events.rs`). These are a fall,
   flagged when a reset of the counter caused it, or a rise, and the envelope steps a
   fall at double speed left waiting. A segment that opens with an event runs its write
   and its event only, as `Apu::mcycle` runs the events at an M-cycle's start: D reads as
   0 to every node that counts cycles and the sample flag as 0 to the sample clock, and
   the pass emits all D of its M-cycles as left (`DL`). The host counts those again with
   the countdowns this pass emitted, as if they had just run (`transport::Cutter`), so
   where the event falls relative to its M-cycle's cycles is the model's order, not the
   host's. Where a counted fall at double speed leaves steps
   waiting (`pend` 2: CGB, the APU on, the event not skipped), the pass emits `SE` = 1,
   the M-cycles of the next segment after which it ends, so the steps run at the first
   M-cycle of the segment after that, as the reference's `mcycle` orders them. Every
   other pass emits 0, and that segment runs on over the cycles after it. The host counts
   `SE` down beside `SD` (`transport::Cutter`, one cutter both programs run) and decides
   no end of its own; `ApuLockstep` holds the M-cycles each emitted end falls after to
   those after which the reference's `steps_wait` holds.
3. `n` APU cycles of 2 dots (`run.rs`): `D` at double speed and `2·D` at normal speed,
   at most 44. Each countdown wraps at `c + 1 + j·P ≤ n` for the squares, the wave
   channel and the noise channel's counter, as threshold tests. The tests for j ≥ 1 are
   narrowed to the frequency's low bits, which they need. The state after the cycles
   sums the reloads, as in §2. The noise LFSR steps once per rise of the counter bit NR43
   selects among the counts, through affine powers over GF(2).

`SD` counts halves (2 dots: one M-cycle at double speed, two at normal), the unit the
sample clock advances by at either speed (375 of its 2^-14 sample units a half), so a
change of speed needs no pass: the host counts it down by each M-cycle's halves, the
rate the control logic emits, and the sample falls in the M-cycle that reaches or passes
it, as the frame time does.

SameBoy runs the APU every M-cycle while the sweep counts down, so the batch is not
always exact. The sweep's calculation lands at the M-cycle of its tick `T = rtm +
calc`: `T` at normal speed, `2T − lf_div` at double. The restart hold, and channel 1
when the calculation's overflow stops it, see only the cycles before that M-cycle. The
circuit computes that M-cycle and uses it for both.

`EV` gained bit 4, a fall from a reset, and bit 5, a rise. The state is 63 bytes
(`fields::FIELDS`, 504 bits), against 51. A channel's latched sample is carried as 0
while the channel is off, because no read sees it and every start reloads it.

### 15.2 Size (measured)

| | §14 | §15 |
| - | - | - |
| Wires, nodes, depth | 2,089, 1,657, 12 | 3,553, 3,009, 24 |
| Hidden, intermediate, layers, vocab | 896, 960, 13, 14,188 | 1,344, 768, 25, 17,801 |
| Dense parameters, nonzeros | 72.4 M, 137,587 | 168.3 M, 194,786 |

The live wires need more than the 805 channels of hidden 1,024, so `apu_circuit::build`
takes the fewest blocks of 1,024 channels whose budget the wires fit. The cost a pass is in `docs/perf-log.md`, "The APU as SameBoy's port".


### 15.3 Exactness (measured)

- `random_register_traffic_matches_the_reference_on_dmg_and_cgb` drives the circuit with
  6,000 events on DMG and CGB. The events are writes to every register, reads of every
  address including PCM12 and PCM34, falls (a third of them from a reset), rises and
  both speeds. Each pass is held to the reference.
- `hwtests --filter same-suite/apu --weights apu --lockstep apu` passes all 63 cases,
  with every pass of the compiled checkpoint held to the reference.
