# Control in weights: one event per pass

This document moves the control logic (`CLAUDE.md`, "Each component in weights, and its gate") into
weights: the timer, the interrupt controller, the joypad register, OAM DMA and HDMA
control, serial, the cartridge banking registers, bus decoding of the I/O page, and the
PPU's LY/mode/STAT/VBlank timing. Sections 1 to 8 are step 1 of the phase, the
prototype that decided the shape by numbers. It covers a census of how often each piece
is touched, the timer and the interrupt flags as a circuit held in lockstep against the
reference after every pass, its size and its time on the engine, and the decision.
Section 9 projects the remaining pieces. Section 10 is the build the numbers earned,
`--weights control`, partial and labelled. Section 13 is the second round, with the
joypad, serial, OAM DMA and HDMA control and the CGB registers in weights, in-place passes, and
the PPU timing's design. Section 14 builds that design, the PPU's timing in weights.
The census and the timing harness were `examples/control_proto/`, which was throwaway and
has since been removed, so the commands below no longer run. The circuit is `src/control_circuit/`.

## 1. What the reference computes

`machine::timer` advances one M-cycle per tick. A tick first reloads TIMA from TMA and
raises IF bit 2 when the previous M-cycle overflowed, then adds 4 to the 16-bit system
counter and, on a falling edge of the counter bit TAC selects (bit 9, 3, 5 or 7, ANDed
with TAC bit 2), increments TIMA. An overflow leaves TIMA at 0 for one M-cycle with the
reload pending. Writes have side effects. A DIV write clears the counter and a TAC
write changes the selected bit, and either can make the input fall and tick TIMA; a
TIMA write cancels a pending reload.

IF and IE live in the machine. Devices raise IF bits after each block of M-cycles
(`Signals::irq`), the joypad at a frame's button change, and the CPU clears them by
dispatching (`Outcome::irq_ack`) or writing FF0F. The CPU step reads IE and IF at its
start (`step::Inputs`), and while halted with nothing pending the host advances one
M-cycle at a time without calling the core.

`machine::control::Control` now holds the three (timer, IF, IE) behind
`ControlDevice`, the contract a checkpoint fills (§10). The reference's behaviour is
unchanged; every hash in §11 holds.

## 2. The census (measured)

`control_proto <rom> --frames N --from F` runs the reference and counts, per frame,
every CPU access to the I/O page by address, every acknowledgement, every block that
raised IF bits, the halted M-cycles and the M-cycles per step:

| Workload | Steps | Halted M-cycles (spans) | FF04 reads | FF05-FF07 R/W | FF0F, FFFF R/W | IF raises (VBlank, STAT, timer) | Acks | FF44 / FF41 reads | FF00 R/W |
| - | - | - | - | - | - | - | - | - | - |
| Yellow CGB, frames 0-1,200 | 3,355 | 9,730 (41.5) | 1.97 | 0.00 | 0.02 | 0.99, 70.4, 0 | 71.4 | 38.0 / 12.8 | 7.8 / 3.4 |
| `text.sh`, frames 2,900-3,500 | 5,348 | 3,637 (0.2) | 2.00 | 0 | 0 | 1.00, 0, 0 | 1.00 | 11.6 / 2.3 | 8.0 / 3.0 |
| `overworld.sh`, frames 7,200-7,800 | 1,966 | 12,940 (0.9) | 2.00 | 0 | 0 | 1.00, 0, 0 | 1.00 | 0 / 0 | 8.0 / 3.0 |
| Crystal, frames 0-1,500 | 4,217 | 7,042 (97.3) | 1.98 | 0.00 | 0.00 | 0.99, 139.8, 0.27 | 141.0 | 24.3 / 0.04 | 7.9 / 3.0 |
| Zelda DX, frames 0-1,500 | 6,037 | 3,369 (1.0) | 0 | 0 | 0.01 | 0.98, 3.83, 0 | 4.81 | 29.8 / 0 | 9.7 / 2.9 |
| Mario DX, frames 0-1,500 | 4,456 | 7,214 (1.0) | 0 | 0.00 | 0.01 | 0.97, 1.56, 0 | 2.53 | 3.7 / 0 | 7.8 / 2.9 |
| blargg `instr_timing`, DMG | 5,865 | 0 | 0 | 1.41 reads, 1.16 writes | 0 | 0 | 0 | 1.4 / 0 | 0 |

Per frame, the timer and the interrupt flags are touched a few to about three hundred
times, against 2,000 to 6,000 CPU steps. Most frames read DIV twice (the games' random
number routines) and take one VBlank. The busy cases are the STAT interrupts: Yellow's
intro and Crystal raise and acknowledge 70 to 140 of them per frame.

## 3. The quantum: a span ending in an event

**A pass runs one span.** A span is `D` M-cycles of timer ticks, then at most one event (a write
to FF04-FF07, FF0F or FFFF, a read of one of them, an acknowledgement, or a raise of
other devices' IF bits). A pass happens only at an event or when `D` reaches the
**horizon** the previous pass returned, the M-cycles until the next TIMA reload (1 while a
reload is pending), or until the next falling edge of counter bit 12, the APU frame
sequencer's input, whichever is sooner. Between passes nothing the device holds can
change, so the host counts M-cycles and runs nothing.

The horizon goes through `Device::horizon`'s rule. The machine's scheduler cuts every
block at the smallest horizon, so the pass at the horizon runs at the exact M-cycle of
the reload and the IF bit lands after that block, as the reference's does. A halted CPU
wakes at the right M-cycle because the halted host loop advances one M-cycle per call.

**Why the span is exact.** Inside a span the only change is ticks, and the horizon
bounds `D` so that a span holds at most one TIMA overflow. With `c` the counter, `B`
the selected bit and `P = 2^(B+1)`:

- The counter after the span is `s = c + 4D` (17 bits; the 17th makes the wrap at
  65,536 a falling edge as it is in the reference). The new counter is `s mod 2^16`.
- The edges in the span number `n = ⌊s / P⌋ − ⌊c / P⌋`, which is
  `(s >> (B+1)) − (c >> (B+1))`, so with the 9 bits above `B` of each muxed by TAC
  (`x`, `y`), `n = (x − y) mod 512` exactly, since `n ≤ 256`.
- The count starts from `v = TIMA`, or from `TMA` when a reload was pending and
  `D ≥ 1` (the reload runs before that M-cycle's edge, as in `Timer::tick`). The horizon
  guarantees `v + n ≤ 256`. Below 256, TIMA is `v + n`. At 256 the overflow happened on
  the last edge; if that edge was the span's last M-cycle (`s mod P == 0`) TIMA is 0 and
  the reload pends, otherwise the reload ran one M-cycle later, at the span's end,
  so TIMA is TMA and IF bit 2 rises. No edge can follow the reload inside the span,
  because edges are at least 4 M-cycles apart.
- The horizon from the new state: 1 while the reload pends; else, when the timer is
  enabled, `(256 − TIMA)·P/4 − (c mod P)/4 + 1` (the first edge, 255 − TIMA more, the
  reload); and at most `2048 − (c >> 2) mod 2048` (counter bit 12's next falling edge).
  So `D ≤ 2048` and the input fits 12 bits.

After the ticks the event applies. A DIV write clears the counter and a TAC write sets
the rate, each incrementing TIMA when the selected input falls (`before ∧ ¬after`, as
`set_counter_or_tac` does); a TIMA write sets TIMA and cancels the reload; IF and IE
take writes, an acknowledgement clears bits, a raise sets them; a read returns the byte
from the state after the ticks.

## 4. The circuit

`src/control_circuit/circuit.rs` builds the circuit from `circuit.rs` primitives and the
APU's helpers (`sum_mod`, `sel`, `all`). Inputs are the seven state bytes `C0 C1 TIMA TMA TC IF IE`
(`TC` is TAC in bits 0-2 and the pending reload in bit 3) and the event bytes `D0 D1`
(the span), `EV` (the kind), `EA` (the address's low byte), `EX` (the value or the
bits); outputs are the next state, `RV` (the read's byte) and `H0 H1` (the horizon).

| Level | Logic |
| - | - |
| 1 | `s = c + 4D`, one staircase per bit; `D ≥ 1`; the reload this span (`R ∧ D ≥ 1`) and the one that stays pending (`R ∧ D = 0`); the four rate one-hots from TAC and from a TAC write's value; the event decodes (`EV` bit and `EA` byte per register) |
| 2 | the windows `x`, `y` (9 bits each, one step per rate); `v = TIMA` or `TMA`; `s mod P == 0`; the selected input bit before and after a TAC write |
| 3 | `v + x − y` as 9 bits (bit 8 the overflow); whether a DIV or TAC write ticks TIMA |
| 4 | the reload ran or pends; TIMA after the ticks |
| 5 | `TIMA + 1` and its overflow for a write's tick; IF bit 2 with the timer |
| 6 | every next-state field (`sel`), the four IF bits' events in one node each, `RV` |
| 7 | whether the reload comes before counter bit 12's edge, per rate |
| 8 | the horizon: the reload's or the edge's distance, 12 bits per case, gated |

| Circuit | Inputs, outputs | Wires | Nodes | Steps | Depth | Steps per level |
| - | - | - | - | - | - | - |
| timer + IF + IE | 12, 10 | 316 | 220 | 556 | 8 | 62, 199, 48, 10, 67, 54, 4, 112 |

The largest families by steps are the read mux 48, the 9-bit sum 42, TIMA' 40, the counter
40, the windows 34 and 32, the counter's event 32.

## 5. Exactness (measured)

Three checks, from the unit to the game.

- **Random streams.** `control_circuit::tests` runs 20,000 random events per console
  (writes with fast rates, near-overflow TIMA and TMA = 0xFF over-represented, reads of
  all six registers, acknowledgements, raises, advances of 1 to 3,000 M-cycles cut at the
  horizon as the scheduler cuts them) through the circuit (`Circuit::eval`) in lockstep
  with `Control`; the whole state is compared after every pass, IF and IE after every
  advance, and every read byte. Three mutants are caught: a horizon one M-cycle late, a
  reload that drops its interrupt, and a DIV read off by one. The compiled checkpoint
  runs 200,000 such events (`control_proto --bench`) in lockstep, and 175,083 passes are
  equal.
- **The fuzz the gate names.** `tests/interrupt_fuzz.rs`
  `timer_driven_interrupts_with_the_compiled_control_logic` runs random programs dense in
  HALT, EI, DI, RETI, RST and in loads that program TAC, TIMA, TMA, DIV, IE and IF, with
  the timer at the fast rates, run with IE and IF coming from the compiled control
  checkpoint (itself in lockstep with `Control`), while the CPU candidate is held to the
  reference every step. The circuit core ran 16,000 steps with 329 dispatches, 259 by
  the timer. The CPU checkpoint ran 80,000 steps with 1,897 dispatches, 1,632 by the
  timer. The
  programs are dense in HALT, but the test does not count the wake-ups the timer
  caused apart from the other dispatches.
- **Games and test ROMs**, `--weights control --lockstep control` (the model computes
  the timer and the flags, `Control` runs beside it, the first difference stops the
  run):

| Run | Frame hash | Control passes (per frame) | Equal to the reference |
| - | - | - | - |
| Yellow CGB 300, `--weights cpu,control --lockstep all` | `cfd31f3d` | 3,735 (12.5) | all |
| Yellow DMG 8, `--weights cpu,control --lockstep all` | `6872b003` | 77 | all |
| `text.sh`, `--weights cpu,memory,apu,control --lockstep all`, 600 frames | `870aa3fd` | 94,063 (156.8) | all |
| `overworld.sh`, the same | `822059c1` | 94,400 (157.3) | all |
| Yellow CGB 300, the same | `cfd31f3d` | 3,735 (12.5) | all |
| Yellow DMG 1,200 | `c941c0f4` | 186,323 (155.3) | all |
| Crystal 1,500 | `4e6a7509` | 443,727 (295.8) | all |
| Zelda DX 1,500 | `1d00d871` | 27,509 (18.3) | all |
| Mario DX 1,500 | `7dec3947` | 20,751 (13.8) | all |
| blargg `instr_timing` | Passed | 116,534 (29.1) | all |
| blargg `instr_timing`, `--weights cpu,control`, no lockstep | Passed | 116,534 | (model only) |
| blargg `cpu_instrs` | Passed | 38,578 (9.6) | all |
| blargg `dmg_sound` 01-04, `--weights apu,control --lockstep all` | `99793856`, `881f3fe7`, `e494d943`, `8e776b75` | 19,144 each | all |

`instr_timing` measures every instruction's length with TIMA, so it is the test of the
span's closed form at the overflow; it sets the timer through 4,646 writes and reads
TIMA 5,638 times. None of the games' timers raise an interrupt except Crystal's (0.27
per frame); the fuzz is what exercises timer-driven dispatch and HALT wake-up.

## 6. Size (measured)

| Checkpoint | Hidden | Intermediate | Layers | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - |
| timer + IF + IE | 1,024 | 256 | 9 | 3 | 3,350 | 17.5 M | 14,358 |

The dense figure splits into embeddings and LM head 6.9 M, attention 3.5 M and MLPs
7.1 M. It is mostly padding, since hidden is the compiler's floor of 1,024 for 316 wires and 22 positions, and
the vocabulary is `12·257 + 256 + 10` tokens. Layer 0 gathers 4 slots per head
(3 heads for 12 inputs), so the 4-slot gather is already in these numbers. The integer
fast path is on and no pass fell back.

## 7. Time (measured)

**Per pass.** The engine runs every span on the template path, incrementally from the
previous one; the integer fast path took every pass.

| Run | Passes per frame | Units per pass | µs per pass |
| - | - | - | - |
| random stream, 200,000 events, compiled, lockstep (`control_proto --bench`) | 74.4 | 659 | 2.89 (the reference's per-M-cycle ticks included) |
| `overworld.sh`, `--weights cpu,control`, `GBCFORMER_STATS=1` | 157.3 | 483 | 1.96 |

Units by layer on `overworld.sh` are 167, 46, 0, 0, 2, 1, 9, 248, 10. Nearly every pass
changes `D`, so the counter's staircases (layer 1) and the horizon (layer 8, which reads
the counter after the span) re-evaluate on every pass; the TIMA logic in between rarely
moves.

**Where the passes go.** `text.sh` and `overworld.sh` each run 157 passes per frame,
not the 12 the census's 0-to-1 count suggested. 145 of them are raises of an IF bit that
is already set.
The PPU raises STAT once a line while the game keeps the STAT interrupt disabled in IE,
so IF bit 1 is never cleared and every raise leaves the state as it was. A raise whose
bits are all already set could skip its pass, but deciding that is `bits & !IF`, the
interrupt controller's own merge, on the host; it is not done.

Joining raises and acknowledgements into one pass per CPU step does not help either
(measured, second round). A census counter in `Machine::advance` found 88,056 advances
that raised or acknowledged anything on `overworld.sh`, against 87,456 raises and 600
acknowledgements, and 87,720 against 87,120 and 600 on `text.sh`. No raise shares an
advance with an acknowledgement, so the join would run the same 146.8 passes per frame.
The first round's "about 10 per frame" was an inference that assumed the raises cluster.
They come once per line, 114 M-cycles apart, and 74% of `overworld.sh`'s M-cycles are
halted ones, each its own advance. While the PPU raises STAT from outside the
checkpoint, the count stays near 145 per frame whatever the grouping (§13).

What the second round changed is the cost of those passes. A raise or an
acknowledgement runs **in place**, over an empty span, leaving the M-cycles since the
last pass to the next one. That is exact because both touch only IF, and the ticks they
skip cannot change IF, since the timer's bit rises only at a horizon, which always ends a span.
Consecutive raises then change only the event inputs, so the engine re-evaluates neither
the counter's staircases nor the horizon. On `overworld.sh` the units per pass fell from
483 to 20 and the time per pass from 1.96 to 0.41 µs (`GBCFORMER_STATS=1`, one run at
load 20 to 30), with the same 157.3 passes per frame, all equal to the reference.

**Per frame** (`ab.sh`'s procedure on scratch copies of `text.sh` and `overworld.sh`,
600 model frames each, one binary, A B A B A B, 3 runs per side):

| Stacks (A → B) | text.sh fps (runs; median) | overworld.sh fps (runs; median) | Load |
| - | - | - | - |
| `cpu` → `cpu,control` | 56.0, 56.7, 56.5 → 55.5, 56.0, 55.9 (56.5 → 55.9) | 132.1, 132.6, 134.6 → 127.8, 129.9, 130.1 (132.6 → 129.9) | 10.1 → 7.4 |
| `cpu,memory,apu` → `cpu,memory,apu,control` | 29.0, 29.3, 28.5 → 28.7, 22.4, 28.4 (29.0 → 28.4) | 77.2, 76.3, 76.2 → 75.2, 73.5, 75.0 (76.3 → 75.0) | 7.4 → 8.1 |

The control checkpoint costs 1 to 2% of the frame rate, 0.19 ms per frame on
`text.sh` and 0.16 to 0.23 ms on `overworld.sh` at the medians. That is its 157 passes
per frame at 1.96 µs (0.31 ms of engine time), less the reference timer's 17,556
per-M-cycle ticks the device no longer runs. The 22.4 fps run is an outlier with other
work on the machine. A first build held the device behind a trait object on
every access (from `cpu` to `cpu,control`, text.sh went from 57.5 to 56.2 and
overworld.sh from 132.8 to 129.8, at load 7.0); giving the compiled checkpoint its own static arm of the machine's slot
measured text.sh better and overworld.sh unchanged, which these runs do not separate
from noise. The structural reason to keep it is that a halted CPU touches the slot on
every M-cycle.

## 8. Decision

Three placements were on the table. The census and §7 decide between them.

1. **Folded into the CPU's step circuit.** The CPU pass already outputs the step's
   M-cycle count, so the timer could be a closed form over it, and IF an output byte
   the next step reads. This fails on two counts. The CPU checkpoint does not run while
   the CPU is halted (`step.rs`), and Yellow is halted for 3,600 to 12,900 M-cycles a
   frame (Crystal in 97 separate spans), through which the timer must keep counting and
   may be what wakes the CPU; something outside the CPU pass must run then anyway. And
   the counter changes on every step, so its adder and every staircase that reads it
   would re-evaluate on all 2,000 to 6,000 steps per frame. At the 298 control units a
   real pass re-evaluates (§7) and the engine's 1.3 to 3 ns per unit
   (`docs/perf-log.md`, "Unit I"), that is about 1 to 5 ms per frame, and about eight more levels
   after the level that computes the M-cycle count (inferred, not built). It would also
   change the CPU checkpoint, whose digest is a gate.
2. **Its own checkpoint on its own thread**, as the APU. The coupling is synchronous,
   because the CPU reads IF at the start of every step, so a worker would make the CPU thread
   wait at every step or at every pass. The APU's thread paid off because its 815
   passes per frame are independent of the CPU between reads; the control logic's are
   not.
3. **Its own checkpoint, inline, one pass per event (chosen).** This runs 9.6 to 296
   passes per frame across the workloads (§5, §7). A frame needs about 12 events, two DIV
   reads, one VBlank raise and acknowledgement and 8.6 counter-bit-12 horizons. A game that leaves a
   raised bit disabled in IE adds one per raise (Yellow's `text.sh` and `overworld.sh`,
   157).
   §7 measures the cost at 1 to 2% of the frame rate.

The quantum is a span of up to 2,048 M-cycles ending in an event or at the horizon,
exact by the closed forms of §3. The size is 316 wires, depth 8, 17.5 M dense (padding)
and 14,358 nonzeros. In the coupling (built, §10) the CPU reads IE and IF from the
control device's last state, the step's acknowledgement and every block's raised bits
are events, and the machine's scheduler cuts blocks at the device's horizon.

## 9. The rest of Phase 3 (projected)

Each piece below is projected from the census (§2) and from this circuit's measured
cost; none is built. "Passes" means passes of the control checkpoint if the piece joins
it as more event kinds and state bytes, which every piece but the PPU timing can do
without touching its quantum.

| Piece | State | Events per frame (census) | Circuit (inferred) | Notes |
| - | - | - | - | - |
| Joypad register P1 | select (2 bits) | 8 to 10 reads, 3 writes, 1 button change | about 40 steps, depth 2 | The pressed buttons stay a host input, as a keyboard is. The interrupt is the falling-edge test `Joypad::set` makes. |
| Serial | SB, SC, 10-bit countdown | 0 to 2 writes (Mario DX 1 each of FF01, FF02) | about 80 steps | The countdown is one more horizon (`remaining`), so completion is a horizon pass that raises IF bit 3. The bytes sent stay a host observation, like frames. |
| OAM DMA control | page, index (8 bits) | 1 write | about 60 steps | Two passes per transfer (the FF46 write and the end, 160 M-cycles later, a horizon) if the pass outputs the copied index range instead of one copy per M-cycle; the copies are memory records already (`docs/memory-in-weights.md` §8a). The bus block (`CTL` bit 4) becomes this state's "active" bit. The scheduler's one-M-cycle blocks during a transfer exist for the host PPU's OAM reads and are a host matter. |
| HDMA control | source (12 bits), destination (9 bits), blocks left, mode | 0 to 9 writes; while an HBlank transfer runs, one block per visible line | about 150 steps (two 16-bit address adds) | The HBlank trigger is the PPU timing's event; until that moves in it is a raise-like host input. |
| Banking registers | in the memory checkpoint already | per write | none new | With memory in weights the registers are memory records and the translation is in the CPU side's circuit; the cartridge keeps a host mirror for saves and for the lockstep. KEY0/KEY1 are one byte each; SVBK and VBK are memory records already. The MBC3 clock reads the host's wall time, a legitimate host input. |
| I/O-page bus decoding | none | every access | none new | Each checkpoint decodes its own registers' addresses (this circuit's `EA` one-hots); what stays on the host is the dispatch of an access to a checkpoint, a scheduler's job. |
| PPU timing (LY, mode, STAT, VBlank and STAT interrupts, the HBlank signal) | dot (9 bits), LY, LYC, STAT, the STAT line | LY reads 4 to 38, STAT reads 0 to 13, STAT raises 0 to 140, VBlank 1 | about 300 to 500 steps, depth about 6 | Same span shape: `dot + 4D` with `D ≤ 114` crosses at most one line, so LY and the dot are one conditional subtract, and the horizon is the next mode change that matters (an enabled STAT source's rising edge, VBlank, the host renderer's mode-3 start). |

**PPU timing's cost** is the largest. While the renderer is on the host it needs the
mode-3 start of every visible line, so the horizon is at most one line. That gives at
least 144 passes per frame, plus 144 HBlank events while HDMA runs, plus the LY and STAT
reads and the STAT raises, about 150 to 450 passes per frame (inferred). When Phase 1 puts the
renderer in weights, the per-line pass becomes the scanline checkpoint's own, and the
timing can run in it.

**Total, projected.** The merged control checkpoint comes to about 1,500 to 2,000 wires, 10
levels, and 170 to 750 passes per frame. At the 1.4 to 3 µs per pass measured here (a
pass costs a fixed part plus the units it changes, §7), that is 0.3 to 2.3 ms per frame
on the CPU's thread, against 12 to 33 ms the CPU side takes today. Inline stays the
right placement; the PPU timing is the one piece that could move to the PPU's thread
once the renderer is there, since the CPU then only sees its IF bits as raises.

## 10. The build: `--weights control`

`--weights control` (alone or with `cpu`, `memory`, `apu`) compiles the circuit into
its own checkpoint and hands the timer, IF and IE to it at the hand-off; `--lockstep
control` runs `machine::control::Control` beside it. The component is partly built and
the run's label says which parts stay on the host. After the first round it read
`control still on the host: the joypad register, serial, OAM DMA and HDMA control,
KEY0/KEY1/SVBK, the PPU's LY/mode/STAT timing, and the cartridge banking registers
unless memory is in weights`; after the second (§13) it reads `control still on the
host: the PPU's LY/mode/STAT timing and HBlank signal, and the cartridge banking
registers unless memory is in weights`; after the third (§14) it reads `control still on the host: the cartridge banking registers`, and nothing once memory is in weights (`Component::still_hosted`). This section
describes the first round; §13 what the second changed.

- **ABI.** `Ctl` implements `checkpoint::SlotAbi` with 12 inputs, no request or late slots,
  10 results. Every span is one pass, settled in pass 1.
- **Device.** `ControlModel<E: Spans>` implements `ControlDevice`. It counts M-cycles
  since its last pass and runs a pass at `sync` (before a read), `write`, `ack`, `raise`,
  and in `advance` when the count reaches the horizon (an acknowledgement or a raise runs
  over an empty span and leaves the count to the next pass, §7); `horizon()` is what is left of
  it, and `advance` panics past it. `read` without a sync panics. `ie()` and `iflag()`
  return the last pass's state bytes, which no other path can change.
- **Machine.** The machine holds `ControlSlot`, which is the reference, the compiled
  checkpoint, or another device (the lockstep) behind a trait object, matched per access
  (`control!`); the first two calls are static, since a halted CPU touches the slot on
  every M-cycle (§7). The timer's address range, FF0F and FFFF route
  to it (`Dev::Control`); `Machine::read_synced` syncs it before a read; the scheduler
  adds its horizon to the block cut; `apply_signals` and `set_buttons` call `raise` with
  any bits other devices raised; `Bus::ack` calls `ack` when a dispatch acknowledged a
  bit. Under the reference these are the same operations as before.
- **Lockstep.** `ControlLockstep` ticks `Control` through every advance and compares the
  whole state after every pass, IF and IE after every advance (so a horizon that comes
  late is caught at the M-cycle the reference raised the bit), every read byte, and the
  counter the APU reads.
- **What the host still does here, labelled in `describe()`.** It counts M-cycles
  between passes, and the APU's DIV input (`Signals::div`) is the model's last counter
  plus 4 per M-cycle since, an add on a model output, as the APU's sample phase is
  (`docs/apu-in-weights.md` §11). An empty acknowledgement or raise (no bits) runs no
  pass, since it is no event.

## 11. Gates

On the committed tree, from scratch copies of `tools/gate.sh` with absolute paths (the
worktree holds no ROMs or `vendor/`; each test binary ran from the main checkout).

| Gate | Result |
| - | - |
| build without warnings; wasm32 check of the library | ok |
| `cargo test --release`, 8 binaries (lib 118, `sm83_sst` 3, `checkpoint_sst` 5, `memory_dense` 2, `e01_primitives` 2, `interrupt_fuzz` 4, `dma_records` 1, bin 5) | ok |
| `--core lockstep` DMG 8 / CGB 300 | `6872b003` / `cfd31f3d` |
| Crystal, Zelda DX, Mario DX, 1,500 frames lockstep | `4e6a7509`, `1d00d871`, `7dec3947` |
| `text.sh`, `overworld.sh` (`model` core) | `870aa3fd`, `822059c1` |
| blargg `cpu_instrs`, `instr_timing`; `dmg_sound` 01-04 | Passed; `99793856`, `881f3fe7`, `e494d943`, `8e776b75` |
| dmg-acid2, cgb-acid2 | pixel-exact |
| the CPU checkpoint's digest (`examples/digest`) | `1f4c3519`, unchanged |
| with control in weights | every row of §5's table |
| existing cores no slower | first round: not met by the plain reading, 0.3 to 1% per pass slower on `model` and `model-mem` (final pair: text.sh 2.53 → 2.55 and 6.23 → 6.25 µs, overworld.sh 3.02 → 3.05 and 6.31 → 6.33 µs; `docs/perf-log.md`, "Phase 3"). Second round: recovered by the latched lines (§13.2, §13.6) |

## 12. Reproducing

```sh
cargo build --release --all-targets
./target/release/examples/control_proto pokemon-yellow.gbc --frames 1200            # census, frames 0-1,200
./target/release/examples/control_proto pokemon-yellow.gbc --frames 3500 --from 2900 --press ...   # text.sh's presses
./target/release/examples/control_proto --shape                                      # circuit and checkpoint
./target/release/examples/control_proto --bench 200000                               # random stream on the engine, lockstep
./target/release/gbcformer run pokemon-yellow.gbc --frames 300 --weights control --lockstep control --no-save
cargo test --release --lib control_circuit
cargo test --release --test interrupt_fuzz timer_driven
```

## 13. Second round: the rest of the control logic, and the PPU timing's design

Four units, one commit each, then this
section's measurements.

### 13.1 Passes: the join does not help; in-place passes do

§7 measured that no raise shares an advance with an acknowledgement on `text.sh` or
`overworld.sh`, so joining them into one pass per CPU step saves nothing. What does
help is running an event **in place**, over an empty span, whenever the ticks since
the last pass cannot touch what the event reads or writes (`Event::needs_span`): a
raise, an acknowledgement, a button change, an HBlank, and every access to P1, SB,
FF46 (a read), HDMA, KEY0, KEY1 and SVBK. The M-cycles stay counted for the next full
span. An in-place pass changes only the event inputs, so the engine skips the
counter's staircases and the horizon. It costs 20 units and 0.41 µs against 483 units and
1.96 µs for a full span (§7).

The one option that would bring `text.sh` and `overworld.sh` near 12 passes a frame
is not built. It is an output byte of the IF bits a raise would change (IF's
complement, free in the circuit), with the host skipping a raise whose bits all fall
outside it. In its favour, the host already schedules on model outputs (the horizon,
and the halted check in `step.rs` reads IE and IF). Against it, it is `bits & !IF` with
the NOT moved into the circuit, the skip §7 says the host should not make. That is the
operator's call. With in-place passes the 145 redundant raises cost about 0.06 ms a
frame, so the call is about where the logic sits, not about speed.

### 13.2 The latched lines

`ControlDevice::lines` returns what the bus and the scheduler read between calls (IE,
IF, the counter, the horizon, whether OAM DMA runs, SVBK, KEY1), and the machine
latches it after every control call. The CPU step's IE and IF, the bus's DMA check,
the WRAM bank and the block cut are field loads again. A control read is one call,
`read(&mut self)`, which runs the pass; there is no separate sync. Numbers in
`docs/perf-log.md`, "Phase 3, second round".

### 13.3 The new event kinds

This section records the second round as built. The `HBlank` event and
`Signals::hblank` it names are historical, because the third round moved the PPU's timing into
the control logic, and the HBlank signal to HDMA became internal (§14.1).

| Piece | State bytes | Events | Span or in place | Horizon |
| - | - | - | - | - |
| Joypad P1 | `JP` (select bits 4-5), `JB` (buttons held) | `Buttons(pressed)` from the host each frame; P1 write and read | in place | none |
| Serial | `SB`, `SC` (bits 0 and 7), `SR0 SR1` (M-cycles left, 11 bits) | SB, SC writes and reads | SC write: span; the rest in place | the transfer's end |
| OAM DMA | `DR` (FF46), `DI` (next index, 160 idle) | FF46 write and read | write: span; read in place | the transfer's end |
| HDMA | `HDMA1`-`HDMA4` (source, destination), `HDMA5` (as it reads) | FF51-FF55 writes, FF55 read, `HBlank` (historical, §14.1) | in place | none |
| KEY0, KEY1, SVBK | `K0`, `K1`, `SV` | writes and reads | in place | none |

Every closed form follows §3's rule: the horizon bounds the span so that it holds at
most one of each piece's events, and that event lands on the span's last M-cycle.

- **Joypad.** Each of P1's four lines is `(¬P14 ∧ low button) ∨ (¬P15 ∧ high button)`,
  one inclusion-exclusion node, before and after a select write or a button change.
  A selected line falling sets IF bit 4 inside the circuit. The buttons are a host
  input, as a keyboard is; the machine passes them once a frame.
- **Serial.** The countdown is `rem − D`, gated by `rem ≥ 1`; `rem = D ≥ 1` is the
  completion, where SB becomes 0xFF, SC bit 7 clears, IF bit 3 rises. An SC write of 0x81
  loads 1,024 and sets an output flag, and the host records SB as the byte sent (the
  blargg ROMs' "Passed" now comes from the model). A transfer's end is one more horizon
  candidate.
- **OAM DMA.** `DI' = DI + D` while `DI < 160`; a write to FF46 sets `DR` and `DI = 0`.
  The model outputs the source page (`DR − 0x20` for E0-FF, the WRAM echo) and an
  "active" flag that gates every bus access. The host moves one byte per M-cycle,
  from the page at index `DI` plus the M-cycles since the pass. That is a host add on a
  model output, as the counter is, and it is labelled. The transfer's end is a horizon candidate,
  so a transfer costs two passes. Blocks stay one M-cycle long while it runs because
  the host PPU reads OAM at mode 3's start.
- **HDMA.** Source and destination count in 16-byte blocks (12 and 9 bits). A
  general-purpose transfer copies `n = min(blocks, 512 − destination)` blocks at once,
  with the source and destination advanced by `n` (the destination to 0 when it fills
  VRAM). An HBlank transfer copies one block per `HBlank` event and finishes at its
  last block or at the end of VRAM. HDMA5 reads back as the reference's. The pass
  outputs the copy (source, destination, blocks); the host turns blocks into bytes
  (16 each) and stall M-cycles (8 each). In this round the HBlank event was the host
  PPU's `Signals::hblank`, delivered whenever it was set, about 144 in-place passes a
  frame. Both are gone since the third round, which made the signal internal (§14.1).
- **The horizon.** The candidates are the timer's reload (one per rate), counter bit
  12's next falling edge, the serial end and the OAM DMA end. Each candidate's bits are
  gated by "valid, no reload pending, and nearest of the valid ones", with one
  comparison node per pair summed into the gate. That is the same two levels the timer's horizon
  used.

### 13.4 Size (measured, `control_proto --shape`)

| Checkpoint | Wires | Nodes | Depth | Hidden | Intermediate | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - |
| timer + IF + IE (first round) | 316 | 220 | 8 | 1,024 | 256 | 3 | 3,350 | 17.5 M | 14,358 |
| + joypad, serial, KEY0/KEY1/SVBK | 524 | 356 | 8 | 1,024 | 384 | 6 | 5,673 | 29.3 M | 21,315 |
| + OAM DMA, HDMA | 762 | 538 | 8 | 1,024 | 640 | 7 | 7,485 | 41.3 M | 31,748 |

The dense figure is still mostly padding. 28 inputs and 33 outputs make the
vocabulary `28·257 + 256 + 33`, and embeddings and the LM head are 15.3 M of the 41.3 M. The
largest families by steps are the read mux (143), HDMA1's next value (72) and HDMA5's
(53).

### 13.5 Exactness (measured)

- **Random streams.** `control_circuit::tests` runs 20,000 events per console over all
  18 control addresses, button changes, raises, acknowledgements, and advances with a
  random HBlank in one block of 8. The circuit runs in lockstep with `Control` on state,
  reads, the serial bytes, every write's and advance's bus requests (copies, stall)
  and the latched lines. Eight mutants are caught: a late horizon, a reload without its
  interrupt, a DIV read off by one, a serial completion without its interrupt, a
  button press without the joypad interrupt, a transfer that never reports its byte,
  an OAM DMA that ends a byte early, and an HDMA that copies a block too few.
- **Games and test ROMs**, the model computing the control logic and `Control`
  beside it:

| Run | Frame hash | Control passes per frame (in place) | Equal to the reference |
| - | - | - | - |
| Yellow DMG 8 / CGB 300, `--weights cpu,control --lockstep all` | `6872b003` / `cfd31f3d` | | all |
| Yellow DMG 1,200, `--weights control --lockstep control` | `c941c0f4` | | all |
| `text.sh`, `--weights cpu,memory,apu,control --lockstep all` | `870aa3fd` | 314.6 (302.0) | all |
| `overworld.sh`, the same | `822059c1` | 314.6 (302.0) | all |
| Crystal 1,500, the same | `4e6a7509` | 457.6 (444.7) | all |
| Zelda DX 1,500, the same | `1d00d871` | 227.5 (217.0) | all |
| Mario DX 1,500, the same | `7dec3947` | 179.6 (168.1) | all |
| blargg `instr_timing`, `--weights control --lockstep control` and `--weights cpu,control` | Passed | 174.0 (145.9) | all |
| blargg `cpu_instrs`, `--weights control --lockstep control` | Passed | 154.6 (145.9) | all |
| blargg `dmg_sound` 01-04, `--weights apu,control --lockstep all` | `99793856`, `881f3fe7`, `e494d943`, `8e776b75` | | all |

Per frame on `text.sh` and `overworld.sh` there are 9.6 horizons, 4 writes (3 to P1, 1 to
FF46), 10 reads (8 of P1, 2 of DIV), 1 acknowledgement, 145 raises, 1 button change
and 144 HBlanks. Only the horizons, the FF46 write and the DIV reads run a full span,
about 12.6 passes a frame. The two workloads now count the same events. In the first round they differed
by 120 and 456 raises, the joypad interrupts of frames with a button held (text.sh
holds A or Down about 120 of its 600 frames, overworld.sh Left or Right about 450),
which the circuit now raises inside the pass that changes P1 or the buttons. Every 1,500-frame game moves HDMA or OAM DMA through the
model, and the memory core's DMA records (`docs/memory-in-weights.md` §8a) now come
from the model's copy requests.

### 13.6 Time (measured)

`ab.sh`'s procedure on scratch copies of `text.sh` and `overworld.sh`, A B A B A B, 3
runs a side, load 6.5 to 7.5. With control in weights A is the first round's binary
(timer and flags in weights and the rest of the control logic on the host)
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
frame (text.sh from 57.0 to 55.9 fps at `cpu,control`, overworld.sh from 128.4 to
127.3). That is
the price of moving the joypad, serial, OAM DMA, HDMA and the CGB registers off the
host. It buys a checkpoint 2.4 times larger and 157 more passes a frame, almost all of
them in place: 144 HBlanks, 3 P1 writes, 8 P1 reads, 1 button change and 1 FF46 write.
The control engine's own stats on text.sh (`--weights cpu,control`,
`GBCFORMER_STATS=1`, load 12) account for it, with 181 units and 1.12 µs per pass over
314.6 passes, 0.35 ms a frame. Most of those units (77 and 69 in layers 1 and 2) are
the event decodes re-evaluating when consecutive in-place passes switch kind, a raise
then an HBlank. That is the next lever.
The full stack measures flat on overworld.sh (73.4 to 73.3) and 1% slower on text.sh.
On the cores without control in weights, B is behind on overworld.sh's `model` core in
every pair (0.7 to 2%) and level elsewhere. The load-independent count says the host
path is not the cause. 600 reference frames take 4.97 G instructions and 0.847 G
cycles against the binary before the first round's 4.83 G and 0.845 G, and 40 `model` frames take 36.67 G
against 36.68 G. So the `model` gap on overworld.sh is not explained by work done. It
is within what a load of 7 moves a 3-run median (the A runs alone span 133.6 to 135.1).
The first round's 0.3 to 1% per pass on `model` and `model-mem` is no longer
measurable.


### 13.7 What stays on the host

The run label now reads `control still on the host: the PPU's LY/mode/STAT timing and
HBlank signal, and the cartridge banking registers unless memory is in weights`. The
host also does these, each labelled in `describe()`:

- it counts M-cycles between passes;
- it adds them to the model's counter (the APU's DIV input) and to the model's OAM DMA
  index (the byte of each M-cycle);
- it turns HDMA blocks into bytes and stall;
- it records SB when the model flags a transfer's start;
- it cuts blocks to one M-cycle while OAM DMA runs, for its PPU;
- it hands each I/O access to the device the memory model's emitted address names;
  with memory in weights the bus decoding is the model's (`docs/host-in-weights.md`
  §3 step 2), and without it the host's map routes it (§9).

### 13.8 The PPU's timing as event kinds (designed, not built)

**State.** The M-cycle within the line (0-113, 7 bits), LY, LYC, STAT's four enables,
the STAT line, and LCDC bit 7. That is about 4 bytes. LCDC writes go to both the
control device and the PPU, which renders with the other bits.

**Span.** With `D ≤ 114` a span crosses at most one line, with `m' = m + D − 114·[m + D ≥
114]` and `LY' = LY + [m + D ≥ 114]`, wrapping from 153 to 0. These are one conditional
subtract each, no division. The bound costs nothing because the horizon is at most a
line anyway (below).

**Events.** Writes to FF40 (LCD off resets LY and the line), FF41 and FF45; reads of
FF41 and FF44. VBlank (IF bit 0 at line 144) and STAT (IF bit 1 on the STAT line's
rising edge) become internal bits, as the timer's are. The HBlank signal to HDMA
becomes internal too, so `Event::HBlank` goes away.

**Horizon candidates.**

- Every visible line's mode-3 start, where the host renderer draws line LY.
- Line 144, for VBlank and the frame swap.
- Each VBlank line's start, to keep `D ≤ 114`.
- A STAT rising edge, only while IF bit 1 is clear. When the bit is set the edge
  changes nothing, and the circuit, not the host, knows that.
- A mode-0 start, only while an HBlank HDMA runs.

**Outputs.** "Render line LY now", "frame done", and the mode, for the PPU's
palette-RAM lock in mode 3.

**Passes and time (projected from the census, §2).** 144 render passes, 1 VBlank, 10
VBlank lines, the LY and STAT reads (4 to 51 a frame), and the STAT edges with IF
bit 1 clear (0 to 140). That is about 160 to 350 full-span passes a frame. The 145
in-place raises and 144 in-place HBlanks of `text.sh` disappear. At the measured 1.96
µs per full span that is 0.3 to 0.7 ms a frame, inside the budget of about 2 ms. The circuit
is estimated at 300 to 500 steps over the same 8 levels (inferred). When Phase 1
moves the renderer into weights, the per-line render pass becomes the scanline
checkpoint's own and the timing can run in it.

**The seam in `ppu.rs`.** `Ppu::tick` keeps the renderer and loses the clock. It gets
three entry points the machine calls on the control device's flags: `render_line(ly)`
at mode 3's start, `end_frame()` at line 144, and `lcd_off()`. Its reads and writes of
FF41, FF44 and FF45 move to the control device, and its palette lock reads the mode
from the latched lines.

**Why it is not built in this round.** The seam changes `Ppu::tick`, the function that
drives the render path that was being rebuilt in `ppu.rs` at the same time. Building
the timing here meant both changes rewriting the same function. The numbers fit
inline, so the order is to land the renderer first, then move the timing behind those three
entry points in one commit with its own lockstep.

## 14. Third round: the PPU's timing in weights (built)

This round builds §13.8 with three changes found while building it. The PPU keeps
no clock. `machine::lcd::Lcd` is the reference timing. It holds the M-cycle within the line,
LY, LYC, STAT's enables, the STAT line and LCDC bit 7. It lives in `Control`, and the
control checkpoint computes it.

### 14.1 What moved

- **Routing.** FF41, FF44 and FF45 route to control. An LCDC write reaches both the
  PPU (which renders with the other bits) and control (bit 7). The PPU's `tick`, its
  STAT logic and its `Device` impl are gone.
- **The three entry points.** `Signals` carries `render: Option<u8>` (the line whose
  mode 3 began on the block's last M-cycle), `frame` (line 144 began) and `lcd_off`.
  `Machine::apply_signals` calls `Ppu::render_line(ly)`, `end_frame()` and `lcd_off()`
  before the block's copies land, so a line sees VRAM as of its mode 3, as before. The
  window's counter resets at `end_frame` instead of at line 154; no line is drawn in
  between, so this is the same.
- **The block cut.** A block must end at the M-cycle the renderer acts on. The reference's
  horizon (`Lcd::horizon`) and the model's both include a visible line's mode-3 start
  and line 144's start. `Control::advance` asserts that a render or frame lands on the
  block's last M-cycle.
- **HBlank.** The HBlank signal to HDMA is internal (`Hdma::hblank`), so
  `Signals::hblank` and `Event::HBlank` are gone. No device in the machine raises IF
  bits any more, since VBlank and STAT are the control logic's own.

### 14.2 The span, and where it differs from §13.8

- **m and LY.** `m' = m + D − 114·[m + D ≥ 114]`, `LY' = LY + [m + D ≥ 114] −
  154·[LY = 153 ∧ m + D ≥ 114]`, both gated by LCDC bit 7. §13.8 bounded `D ≤ 114`.
  The bound that matters is `m + D < 228`, a single line crossing. The horizon
  guarantees it while the LCD is on, because every candidate is in the current line or at the
  next line's mode 3. With the LCD off `m` and `LY` stay 0 and `D` runs to the timer's
  2,048 (since §16, to `MAX_SPAN` itself, because counter bit 12's edge is no longer a
  horizon).
- **The STAT line without IF.** §13.8 made the STAT edge a horizon only while IF bit 1
  is clear. That makes the horizon depend on IF, so an acknowledgement of bit 1 would
  need a span. Instead, every M-cycle where the line can rise is a horizon whatever IF
  holds. It can rise only at a line's start and at mode 0's start. The horizon covers
  both, with mode 0's start on every visible line and a line's start where
  `line(LY+1, 0) ∧ ¬line(LY, 113)` holds or VBlank runs. Line 144's second M-cycle is
  covered too when its first raised the mode-2 source, because that source falls
  there. With that, the line changes only on a span's last M-cycle, and IF bit 1 rises
  iff the line after the event is high and the stored line was low, which is one node. A STAT,
  LYC or LCDC write needs a span; acknowledgements stay in place. (§16 brings the IF
  condition back, so an acknowledgement of bit 1 is now a span.)
- **The palette lock.** The PPU locks CGB palette RAM in mode 3 and reads the mode from
  the latched lines (`Lines::mode3`). Mode 0's start is a horizon on every visible line,
  so mode 3 is constant between passes.

### 14.3 Size (measured, `control_proto --shape`)

| Checkpoint | Wires | Nodes | Depth | Hidden | Intermediate | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - |
| second round | 762 | 538 | 8 | 1,024 | 640 | 7 | 7,485 | 41.3 M | 31,748 |
| + the PPU's timing | 909 | 653 | 8 | 1,024 | 768 | 8 | 8,517 | 48.1 M | 42,696 |

The largest families by steps are the read mux (167), the PPU's horizon candidate
(76), HDMA1's next value (72) and HDMA5's (53).

### 14.4 Exactness (measured)

`control_circuit::tests` adds FF40, FF41, FF44 and FF45 to the random stream, with the
LCD on seven times in eight and LYC within the frame's lines. Five more mutants are
caught: a STAT rise without its interrupt, VBlank without its interrupt, a line never
drawn, an LY read off by one, and mode 3 without the palette lock.

| Run | Frame hash | Control passes per frame (at the horizon) | Equal to the reference |
| - | - | - | - |
| Yellow DMG 8 / CGB 300, `--weights cpu,control --lockstep all` | `6872b003` / `cfd31f3d` | 343.7 (298.3) CGB | all |
| Yellow DMG 1,200, `--weights control --lockstep control` | `c941c0f4` | 413.5 (302.0) | all |
| Crystal, Zelda DX, Mario DX 1,500, the same, before double speed | `4e6a7509`, `1d00d871`, `7dec3947` (with double speed, Zelda DX `99f51381` and Mario DX `b083771f`) | 491.8 (306.7), 413.5 (307.0), 335.2 (301.9) | all |
| blargg `instr_timing`, `--weights control --lockstep control` and `--weights cpu,control --lockstep all` | Passed | 328.9 (323.9) | all |
| blargg `cpu_instrs`, `--weights control --lockstep control` | Passed | 314.6 (307.3) | all |
| blargg `dmg_sound` 01-04, `--weights apu,control --lockstep all` | `99793856`, `881f3fe7`, `e494d943`, `8e776b75` | 312 (307.0) | all |
| dmg-acid2, cgb-acid2, `--weights control --lockstep control` and `--weights cpu,memory,ppu,control --lockstep all` | pixel-exact | | all |
| Yellow, Crystal, Zelda DX, Mario DX 300, `--weights cpu,memory,apu,ppu,control --lockstep all`, before double speed | `cfd31f3d`, `ccfe7ad3`, `7b126eb8`, `3e1f4d27` (with double speed, Zelda DX `e2bd7038` and Mario DX `ea7f86ca`) | 343.7, 460.8, 423.1, 312.4 | all, and every line pixel for pixel |
| `text.sh`, `overworld.sh`, the same | `870aa3fd`, `822059c1` | 338.3 (308.4), 324.4 (308.4) | all |

Per frame on `text.sh`, 144 passes draw a line, 1 shows the frame, 144 more are mode 0's
start, 10 are VBlank's lines and line 0, and the rest are reads and the timer's own. The
145 raises and 144 HBlanks of the second round are gone.

### 14.5 Time (measured)

The A/B procedure of §13.6: A is the round before, with the undo-log fix, B
is this round. Load 4.5 to 10.7, other work and a play window on the machine.

| Stack or core (A → B) | text.sh fps (runs; median) | overworld.sh fps (runs; median) |
| - | - | - |
| `model` | 57.3, 57.9, 57.4 → 57.3, 57.4, 57.2 (57.4 → 57.3) | 135.4, 135.1, 135.3 → 134.0, 134.5, 134.1 (135.3 → 134.1) |
| `model-mem` | 28.7, 27.9, 28.1 → 28.4, 27.6, 28.3 (28.1 → 28.3) | 76.0, 75.8, 75.6 → 76.0, 75.0, 74.6 (75.8 → 75.0) |
| `cpu,control` | 48.3, 54.6, 54.5 → 15.5, 49.7, 50.3 (54.5 → 49.7) | 123.6, 123.8, 75.0 → 103.7, 103.4, 104.7 (123.6 → 103.7) |
| `cpu,memory,apu,ppu,control` | 27.3, 27.4, 23.0 → 26.2, 26.0, 22.4 (27.3 → 26.0) | 60.8, 44.0, 61.4 → 59.1, 56.9, 53.1 (60.8 → 56.9) |

**Explaining the number.** With control in weights the timing costs about 1.6 ms a
frame. The control engine's stats on these runs give 1,747 units and 5.9 µs per pass on
text.sh (1,829 and 5.5 µs on overworld.sh), against 181 units and 1.0 µs in the second
round. Passes per frame went from 314.6, all but 12.6 in place, to 338.3, of which 308.4
are full spans. So 308 full spans at 5.9 µs is 1.8 ms, against 0.35 ms. A full span
moves the counter, `D`, `m` and LY, so about half the circuit's units re-evaluate. The
horizon's eight candidates and their pairwise comparisons all read the counter. The
`cpu,control` overworld.sh median goes from 8.1 to 9.6 ms a frame, about 1.5 ms. Two runs were
disturbed by builds on the same machine (text.sh B run 1 at 15.5 fps, overworld.sh A run
3 at 75.0). The medians stand without them.

On the cores without control in weights, `model` overworld.sh is 0.9% slower in all
three pairs, and the others tie. The reference path now ticks the timing in
`Control::advance` instead of `Ppu::advance`, and ends about 145 more blocks a frame at
the render points. That is consistent with the gap, but not profiled.

**What is next.** Half of the full spans are mode 0's start, which only HBlank HDMA,
a STAT mode-0 rise and the palette lock need. §15 moves the palette lock to the palette
port accesses, which removes them wherever neither of the other two needs them.

## 15. The palette index registers, and mode 0 only where it matters

With the palette lookup in the PPU checkpoint (`docs/ppu-in-weights.md` §10), BCPS
and OCPS move into the control logic (`machine::control::PaletteIndex`). The control
checkpoint gains two state bytes (`BP`, `OP`) and one result, `PA`, the palette byte a
BCPD or OCPD access reaches, with bit 7 set unless mode 3 locks the port. A data port
write hands the bus that index and the CPU's byte (`Signals::palette`). A read returns
the index (`ControlDevice::palette`), and the machine reads the byte from the PPU
checkpoint, or 0xFF when locked. The data ports need a span, since the lock reads the
mode. The index ports run in place.

So the mode no longer has to be constant between passes, and `Lines::mode3` is gone.
Mode 0's start is a horizon only where something acts on it, which is while an HBlank
HDMA runs (HDMA5's bit 7 clear after the event) or where the STAT line rises there
(`e0 ∧ ¬(LY = LYC ∧ eL)`). An HDMA5 write now needs a span, because starting an HBlank
transfer can make mode 0's start a horizon that an in-place pass would already have
passed.

| Checkpoint | Wires | Nodes | Depth | Hidden | Intermediate | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - |
| §14 | 909 | 653 | 8 | 1,024 | 768 | 8 | 8,517 | 48.1 M | 42,696 |
| + palette index, mode 0 where needed | 976 | 704 | 9 | 1,024 | 832 | 9 | 9,034 | 55.9 M | 44,382 |
| + the level schedule | 976 | 704 | 9 | 1,024 | 448 | 9 | 9,034 | 44.1 M | 44,475 |

The control circuit never ran `Circuit::schedule`, which the CPU, APU and PPU
circuits do. Nodes moved within their slack take the widest level from 673 to 427
units (the horizon's last level, which cannot move), so the intermediate width goes
from 832 to 448 (−11.8 M dense parameters, 93 more nonzeros
for the reordered clearing units).

Two more mutants are caught by the random stream (which now writes and reads FF68 to
FF6B): a palette write to the wrong byte, and a palette read reaching the wrong byte.

**Passes (measured).** Yellow CGB 300 from the hand-off runs 159.1 horizon passes a frame,
from 298.3. On text.sh and overworld.sh the game enables STAT's mode-0 source, so mode 0's
start stays a horizon on every visible line, 308.4 a frame as in §14, at 1,749 and
1,832 units and 6.3 and 5.6 µs a pass, about 2.0 and 1.8 ms a frame on the CPU's thread.
The §14 lever stands, since most of a full span's units are the horizon's candidates and
comparisons re-evaluating because the counter moved.

**The run's label** with every component in weights now reads, in full:

> in weights: cpu, memory, ppu, apu, control; on the host: none; lockstep: none; ppu
> still on the host: LCDC, SCX, SCY, WX and WY latched as written and handed to each
> line's pass; the host besides: presentation (each pixel's colour mapped to the
> screen's RGB, the board's high-pass filter over the APU's mix, the samples played,
> the frame pacing, battery saves written), input (the buttons, and an MBC3
> cartridge's real-time clock registers, set from the wall clock), and scheduling
> arithmetic (M-cycles counted between passes and added to the
> model's counter and OAM DMA index, the 48 kHz sample instants, HDMA blocks turned
> into bytes and stall, each I/O address routed to the checkpoint that owns it, and
> each model's outputs handed to the next pass or the next model)

So the host does more than presentation, input and scheduling arithmetic in two places.
It latches five display registers, and it runs the MBC3 clock's registers.

## 16. Fewer full spans, and a horizon that moves less

On `text.sh`, `overworld.sh` and `battle.sh` the control checkpoint ran 308.4 full
spans a frame at about 1,750 units each, all on the CPU's thread. Two things made that
cost: the passes Yellow's STAT mode-0 source added, and the horizon's candidates and
comparisons, which read the counter and re-evaluated every time it moved.

### 16.1 What changed

- **STAT rises only while IF bit 1 can be clear.** Yellow enables STAT's mode-0
  source but never enables the STAT interrupt, so IF bit 1 is set after the first rise
  and stays set. While it is set, a rise of the line changes nothing a reader sees. The
  horizon now drops mode 0's start (unless an HBlank HDMA runs), the line starts where
  the line rises, and line 144's second M-cycle when IF bit 1 is held, meaning set
  before the pass with the event neither an IF write nor an acknowledgement of bit 1 (one node,
  `IF.1 held`, at level 2). Both clearing events are full spans. An acknowledgement
  with bit 1 now needs a span (`Event::needs_span`). This is §13.8's condition, with
  the acknowledgement made a span, which §14.2 avoided.
- **One reload candidate.** The four per-rate reload candidates are one form,
  `Σ_rate [rate]·((256 − TIMA)·2^(B−1) + 1 − counter[2..=B])`, with each counter and TIMA
  bit ANDed with its rate at the level after them. With the timer off every term is 0,
  so the counter no longer moves four candidates and their comparisons.
- **No counter-bit-12 candidate.** The reference's horizon never cut blocks at the
  DIV edge. The APU takes DIV from the counter at each block's start
  (`Signals::reset`). The edge was only the span's upper bound. With the LCD on, the
  PPU's next point is at most 134 M-cycles away. With it off, the candidate is
  `MAX_SPAN` (2,048) itself, gated by LCDC bit 7 off.
- **The horizon less one.** `H0 H1` hold the horizon minus 1, so the 2,048 fits 11 bits
  and the longest span's bits are its gate. The host adds the 1 back
  (`Ctl::decode_outcome`). Each candidate's bits are as wide as its largest value
  needs: 10 for a serial transfer, 8 for the OAM DMA and the PPU.
- **One comparison per pair.** `j` beats `i` is `¬(i beats j)`. Before, the circuit
  computed each pair twice.

The invariant of §14.2, that the line changes only on a span's last M-cycle, no longer
holds while IF bit 1 is held. Its one consumer is the rise detector for IF bit 1, and
that bit is already set. `line_n` comes from the position at every full span, so the
stored line matches the reference's after each one, and the lockstep compares all of
it. The events that clear bit 1 are full spans that refresh the stored line before the
horizon looks for the next rise.

### 16.2 Passes and units (measured, `--weights control`, `GBCFORMER_STATS=1 GBCFORMER_UNITS=1`)

| Step (text.sh) | Passes per frame (full spans) | Units per pass |
| - | - | - |
| before | 338.9 (308.4) | 1,749 |
| + STAT rises only while IF bit 1 can be clear | 194.9 (164.4) | 753 |
| + one reload candidate | 194.9 (164.4) | 537 |
| + no bit-12 candidate, the horizon less one, one comparison per pair | 186.5 (156.0) | 358 |

On overworld.sh the counts went from 324.4 (308.4) passes at 1,832 units to 172.0
(156.0) at 360, and on battle.sh from 333.9 (308.4) at 1,769 to 181.5 (156.0) at 367. Of the 156 full spans a
frame, 144 draw a line, 1 shows the frame, and 10 are VBlank's line starts and line 0.
None of the remaining ones is the timer's own. The largest families per pass are now
the counter's own sum bits `s` (131 updates), `LY_t` (41) and the PPU candidate (24).

### 16.3 Size (measured, `control_proto --shape`)

| Checkpoint | Wires | Nodes | Depth | Hidden | Intermediate | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - |
| §15 | 976 | 704 | 9 | 1,024 | 448 | 44.1 M | 44,475 |
| §16 | 939 | 667 | 9 | 1,024 | 448 | 44.1 M | 45,785 |

The reload form's 52 ANDs and its 11 bits add nonzeros. The bit-12 candidate, its
comparisons and the duplicate comparisons take some away.

### 16.4 Tried and not kept

- **The counter's sum bits as parity and carry.** Each bit becomes the parity of its
  two addends and a one-step carry, so a counter bit that changes reaches one step per
  bit above it, not three. That cut 140 units a pass. It also put the counter one
  level later on the timer's path, at 11 layers and +2.6 M parameters. Measured before the
  bit-12 change, the gain was 0.08k instructions a step on text.sh.
- **VBlank as one span.** With no LYC or mode-2 source live, line 144 to line 0's
  mode 3 can be one span of up to 1,160 M-cycles. That needs a line count `⌊(m + D)/114⌋`
  of up to 10 and the frame-end distance in the PPU candidate. It saved 10 passes a
  frame but added 30 units a pass, an intermediate of 512 (+2.0 M parameters) and
  4,400 nonzeros, for about 0.01k instructions a step.

## 17. CGB double speed

### 17.1 What the hardware does, and which source this follows

Pan Docs ("CGB Registers", KEY1) gives the behaviour. STOP with KEY1 bit 0 set switches
the speed. KEY1 bit 7 then reads the new speed, bit 0 clears, and the CPU pauses. At
double speed the CPU, DIV and the timer, serial and OAM DMA run twice as fast. The PPU,
HDMA and the APU keep their real-time rates. Pan Docs ("Timer and Divider Registers")
adds that STOP resets DIV and that DIV does not tick until the pause ends. Pan Docs
("Audio details") says the APU's frame sequencer counts falling edges of DIV bit 5 at
double speed instead of bit 4, so it stays at 512 Hz. Pan Docs' VRAM DMA section gives
8 µs per 16-byte block in both speeds, which is 8 M-cycles at normal speed and 16 at
double.

The pause's length is the one place this does not follow Pan Docs, which says 2,050
M-cycles. daid's `speed_switch_timing_ly` test (gbdev/GBEmulatorShootout, MIT) switches
to double speed at line 144 on hardware and reads LY 0x85 next, about 16,400 ticks
later. SameBoy's countdown is 0x20008 T-cycles of the new speed (32,770 M-cycles).
With 2,050 the reference's LY reads land 12,321 pixels off the hardware
screenshot. The pause here is 32,772 M-cycles of the new speed
(`control::SWITCH_MCYCLES`), the value that makes the DIV and STAT screenshots
pixel-exact (table below). DIV is held through it, as Pan Docs says, and the DIV test
confirms it.

| daid `speed_switch_timing_*` (CGB, against the hardware PNG) | base | Pan Docs' 2,050 | 32,772 |
| - | - | - | - |
| div | 3,556 px differ | 0 | 0 |
| ly | not run | 12,321 | 109 (one read late at three of its transitions) |
| stat | not run | 4,810 | 0 |

The same three results hold with every component in weights in lockstep, as four
checkpoints and as one model. No even value from 32,756 to 32,776 makes ly and stat both exact.
32,774 makes ly exact and stat 233 pixels off. The PPU's timing here has whole-M-cycle
modes, so its sub-M-cycle alignment is not modelled.

Two behaviours are kept as before. STOP is one byte, and STOP without a switch armed does nothing (low-power
stop mode is not emulated, and Pan Docs notes no licensed game uses it). Whether
interrupts can fire during the pause is a TODO in Pan Docs. The IF bits the PPU raises
in it wait for the CPU. HBlank HDMA waits through the pause, as it does through HALT.

### 17.2 The reference

`Control::stop` toggles KEY1 bit 7, clears bit 0, resets DIV the way a DIV write does
(the falling edge can tick TIMA), and asks for the pause as stall M-cycles. In
`Control::advance` a pause span moves only the PPU's timing. At double speed the PPU's
timing ticks every second M-cycle; `half` is the M-cycle within the current tick, and a
LCD-off write or a switch clears it. The horizon converts the PPU's next point to
M-cycles (`2·ticks − half` at double speed), and the pause's end is a horizon.

The machine maps each block onto the APU's 4-dot ticks with `device::Clock` (speed, the
M-cycle within the APU's tick, the pause's held counter), built from `Lines::double`
and `Lines::paused`. The APU's tick phase and the frame time (`Machine::halves`, in
2-dot halves) are host arithmetic on block lengths, as the sample instants already
were. `GameBoy::run_frame` runs 35,112 halves, so a double-speed frame runs twice the
M-cycles. The APU checkpoint is unchanged and takes its ticks and DIV edges from the
host as before. The PPU checkpoint is unchanged, since the control logic calls the
renderer.

### 17.3 The circuit

The new state is `PS0 PS1`, the pause's M-cycles left (16 bits), and KEY1 bit 7. `LM` now
holds the position in the line in 2-dot halves (0 to 227), not ticks. The new inputs are the
event `EV_STOP`; `Q0 Q1`, a pause span's M-cycles (the host counts M-cycles into `Q`
instead of `D` while the last pass flagged `FX_PAUSE`); and `H0 H1`, the span in
halves at the model's speed (two an M-cycle at normal speed, one at double), which the
host computes from KEY1 bit 7. The timer, serial and OAM DMA read `D` only, so a pause
span leaves them alone with no gating. The PPU's timing reads `LM + H` with every
threshold doubled. One comparison per point covers both speeds. The horizon has two PPU
candidates, `K − m` at normal speed and `2(K − m) − half` at double speed, and a pause
candidate.

The first cut kept `LM` in ticks and compared the position once per speed in every
node. Its double-speed copies re-evaluated on every normal-speed pass, 463 control units
per fast pass on text.sh as one model, against the base's 358. With halves it is 389.
Folding the two PPU candidates into one, with per-speed gates, did not shrink the last
level (the one candidate's 9-bit sum took 153 steps), so it was not kept.

### 17.4 Size (measured, `control_proto --shape` and `examples/digest`)

| Checkpoint | Wires | Nodes | Depth | Hidden | Intermediate | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - |
| §16 | 939 | 667 | 9 | 1,024 | 448 | 44.1 M | 45,785 |
| §17 | 1,057 | 737 | 9 | 1,024 | 640 | 54.5 M | 56,779 |

The last level, the horizon's bit sums, sets the intermediate. It holds the
double-speed PPU candidate (77 steps) and the pause candidate. Depth is unchanged.

## 18. Every step f16-legal

`cargo run --release --example f16census -- control` listed 182 steps that an f16
checkpoint cannot hold (`docs/stock-llama.md` §3). It now lists none. The function is
unchanged on the domain the span contract allows. The circuit tests,
`tests/interrupt_fuzz.rs` and every control gate run in lockstep with no divergence.

- **Gates at the least weight.** `within`, the LYC compares and `m_t`'s gate weighted
  the whole form's range (8,676 and 767). `gated_ge` gates each step at the weight its
  orientation needs. Up, that weight is `xhi − θ + 1`. Down, it is `θ − xlo`, from
  `[g ≥ n] − [−x + (g − n)·(θ − xlo) ≥ 1 − θ]`.
- **The line position by byte digits.** `md = LM + H` has 13 bits. `ge_digits` reads it
  as `256·H[8..] + (LM + H[..8])`. There is one step for the high values the low digit
  cannot stop, and one step per high value where the low digit decides. That value's
  bits become literals of the step. `fm = md − 228·cross` lies in `[lo, hi]`, with
  `hi < 228`, exactly when `md` lies in that range or 228 above it. So the mode tests
  read `md` directly, and that holds for every input.
- **The horizon in byte digits.** Each candidate is a horizon whose low byte digit
  spans at most 255. The reload is `X + 2`, with X the bits of `¬counter[2..=B]` below
  the bits of `¬TIMA`. The circuit makes the PPU's distance less one into bits first,
  because its exclusive constants overlap in the declared ranges. An invalid candidate
  sits 4,096 higher, which only moves the high digit, in place of 2^18. Each pairwise compare
  therefore splits into byte digits. The output is one node per bit over each
  candidate's bits of `h − 1`, which come from a one-level increment or decrement for
  the binary candidates.

| Circuit | Wires | Nodes | Steps | Depth | Widest level | Units | Intermediate | Dense parameters |
| - | - | - | - | - | - | - | - | - |
| before (f16 emitter base) | 1,272 | 952 | 2,502 | 12 | 303 | 3,425 | 448 | 56.6 M |
| §18 | 1,225 | 905 | 2,409 | 11 | 312 | 3,195 | 448 | 46.9 M |

## 19. The line-timing schedule

On text.sh the control checkpoint ran 186.5 passes a frame, 144 of them only to say
"draw this line now" at a visible line's mode-3 start. Mode 3 starts at a fixed phase
of every line, 114 M-cycles apart at normal speed. Where nothing else can need a pass
before line 144, one pass can say when the next ones are.

### 19.1 What the model emits

A pass schedules up to `SCHEDULE` = 8 mode-3 starts when all of these hold after its
event: the LCD on at normal speed, IF bit 1 held (§16, so no STAT rise can matter), no
HBlank HDMA, the timer, serial, OAM DMA and a pause idle, no reload pending, and the
PPU's next point a mode-3 start (`to mode 3` or `to next mode 3`). Then every other
horizon candidate is invalid, so the schedule never passes one. These are the new results:

| Slot | Value |
| - | - |
| `RF` | the first start less one: the horizon's PPU distance (`hppu`'s bits) |
| `RS` | how many starts are scheduled: `g_j = [the j-th start after the first is visible]` for j = 1..8, the count's bits read from that thermometer code |
| `RP` | 114, the M-cycles between starts |
| `RM` | 40, `LM`'s byte at a start (M-cycle 20) |
| `SL0`..`SL7` | the line each start draws: `LY' + [past mode 3] + j` |

The horizon `H` is still the first start. With `RS` above 0 the next pass is the start
after the last scheduled one, `RF + 1 + 114·RS` M-cycles on, which draws its own line
as before. Depth, layers and heads are unchanged (11, 12, 5). Hidden goes from 832 to
896, the dense size from 40.6 M to 43.8 M, and the nonzeros from 53,143 to 56,241.

### 19.2 What the host does

`ControlModel` counts down to each scheduled start, cuts the machine's block there
(`Lines::horizon`) and calls the renderer with the line the model named for it, with no
pass. Any event that needs a span is a pass, and every pass replaces the schedule; the
events that move the timing all need a span (`Event::needs_span`, which lists writes to FF02, FF04
to FF07, FF0F, FF40, FF41, FF45, FF46, FF55, FF69, FF6B and FFFF, and reads of the timing
registers, an acknowledgement of IF bit 1 and STOP), so a write to LCDC, STAT or LYC,
or a speed switch, drops the rest of a schedule. A pass after `N` scheduled starts runs
from the position the model emitted for the last of them, `LM` from `RM`, `LY` from
`SL[N−1]`, and the PPU's halves counted since that start (`Span::since`), so the span
still crosses at most one line. An in-place pass keeps the last full span's position;
the host asserts that its schedule keeps the starts already passed and that none of its
starts lies behind the M-cycles elapsed.

### 19.3 Exactness

`ControlLockstep` compares the bus requests of every advance, so each scheduled start's
`render` is held to the reference's at the same M-cycle. `scheduled_mode_3_starts_match_the_reference`
runs a random stream of the traffic the schedule serves (no timer, serial, OAM DMA or
HDMA, IF bit 1 raised often) in lockstep on DMG and CGB, with most lines drawn at a
scheduled start. Four more mutants are caught: a scheduled line one off, a schedule one
start long, the first start one M-cycle late, and the position at a start one M-cycle
late. A schedule one start short is no fault, because the next pass then draws that line
itself.

On text.sh with every component in weights and control in lockstep, the control
checkpoint runs 58.4 passes a frame (from 186.5), 27.9 of them at the horizon, with 9,538 lines over
600 frames drawn by a pass and 76,862 at a scheduled start.

## 20. The MBC3 cartridge clock

Pokémon Crystal's cartridge (MBC3, type 0x10) has a real-time clock. Its registers ran in
`machine::cart` as host code. They now run in the control logic: `machine::rtc::Rtc` is
the reference inside `machine::control::Control`, and the control checkpoint computes it.
Wall-clock time stays a host input, as the buttons are.

### 20.1 What moved

The clock holds its own copies of the RAM enable (a 0000-1FFF write) and of the select (a
4000-5FFF write), so it decides which register an access reaches. It also holds the latch
(a 0x00 then 0x01 written to 6000-7FFF copies the live registers), halt (DH bit 6 stops
counting), the 9-bit day counter and its sticky carry (DH bit 7), the counting with its
carries, register writes at selects 08 to 0C (S and M reduced modulo 60, H modulo 24, as
the reference always did), and reads of the latched registers.

The host keeps these parts:

- **Bus decoding.** On a board whose header says it has a clock (`Kind::rtc`), the
  machine routes 0000-1FFF and 4000-7FFF writes to the control device as well as to the
  mapper (`Cartridge::clock_hears`). It routes A000-BFFF reads and writes there while the
  cartridge's select names a clock register (`Cartridge::clock_at`). That is the same
  decision the memory checkpoint makes when it sends such a read to the host-served VOID
  block. Every other board behaves as before.
- **The wall time.** `Machine::set_time` keeps the last Unix second it handed over. On a
  change it hands the control device the time elapsed (`ControlDevice::elapse`), written
  as whole days plus the time of day (`rtc::Elapsed`). That formatting is the wall
  clock's reading. The host applies no halt check, no carry and no wrap. Time that runs
  backwards elapses nothing.
- **Saves.** The cartridge still decides when a save is due. A clock register write with
  RAM enabled counts as one, and a clock that only ticked does not. The 48-byte BGB footer
  comes from whichever device holds the clock (`ControlDevice::clock`), so the model's
  state bytes make the footer when control runs in weights. A footer loads into the
  reference control logic before any hand-off.

The reference's `saturating_add` at u64 overflow was not hardware behaviour. The clock now
adds modulo 512 days and sets the carry when the sum reaches 512 days, however far past.
`mbc3_rtc_wraps_on_an_absurd_host_time` (renamed from `..._saturates_...`) now expects
04 00 07 89 80 after u64::MAX − 10 seconds, where it expected 15 00 07 89 80.

### 20.2 The circuit

`control_circuit::clock` builds it. The state is eight bytes, packed so the input slots
fill whole layer-0 gather groups (`llama::GROUP` = 8; 56 inputs give 7 heads, where the
first cut's 60 gave 8). `CK0` holds S with halt and the carry, `CK1` M with day bit 8 and
the RAM enable, `CK2` H with the select's code (0 for none, 1 to 5 for 08 to 0C), and
`CK3` DL. `CKL0` to `CKL3` hold the latched copy in the same layout, with `CKL1` bit 7 the
latch armed. A latch copies the live bytes across.

A clock event is `EV` bit 6 with a code in `EA`'s low three bits. The codes are 0 (RAM
enable), 2 (select), 3 (latch) and 5 (register write), each a write address's top three
bits, then 6 (read) and 7 (elapsed time). The elapsed time's days arrive in `ED0` to `ED5`
(48 bits, past `u64::MAX / 86400`), hours in `EH`, minutes in `EM` and seconds in `EX`.
Every clock event runs in place (`Event::needs_span` is false), since the clock reads
nothing the span's ticks move.

The add is one register a level. The seconds carry is `[S + s ≥ 60]`, then
`[M + m + c₁ ≥ 60]`, then `[H + h + c₂ ≥ 24]`. The day counter takes the low 9 bits of the
days with `c₃` modulo 512. The carry sets when `[D + d₀..₈ + c₃ ≥ 512]` or when any of the
days' bits 9 to 47 is set (one OR over 39 bits). Halt gates the whole add. The deepest
clock node sits at level 5, so the circuit's depth stays 11. `examples/f16census control`
reports 0 violations under T1 and T2.

### 20.3 Size (measured, `control_proto --shape`)

| Checkpoint | Wires | Nodes | Steps | Depth | Widest level | Hidden | Intermediate | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - | - | - |
| §19 | 1,567 | 1,247 | 3,145 | 11 | 384 | 896 | 512 | 5 | 10,590 | 43.77 M | 56,241 |
| §20, first cut (12 state bytes) | 1,879 | 1,399 | 3,650 | 11 | 448 | 1,024 | 640 | 8 | 15,742 | 70.01 M | 68,098 |
| §20 (8 packed state bytes) | 1,835 | 1,387 | 3,592 | 11 | 447 | 1,024 | 640 | 7 | 14,710 | 66.33 M | 66,540 |

The dense growth is mostly per input slot, not per step. Each input slot adds 257 vocab
rows to the embedding and the LM head, about 0.53 M at hidden 1,024. Each 8 input slots
add a layer-0 head, about 3.1 M across 12 layers. The 16 new input slots and the wider
hidden (896 to 1,024 for 268 more wires) account for embed+head 18.98 to 30.13 M and
attention 13.76 to 22.02 M. The intermediate goes from 512 to 640 and the MLP from 16.52 to 23.59 M. The circuit's
widest level is 447 units, under 512, so the last layer, the routing MLP with 8 more
outputs, must set it (inferred from `Model::shape`, not measured per layer).

The six day bytes cost the most input slots. They follow the design that has the model
OR the high bits. A reading of days modulo 512 plus a "512 days or more" flag would fit
the days in two bytes. That is four fewer slots (52 inputs, still 7 heads) and about
2.1 M fewer dense parameters. It was not built, since it moves part of the carry to the
host.

### 20.4 Exactness and the gates

`ControlLockstep` compares the whole state after every pass, the clock's eight bytes
included, and every read byte. `ControlLockstep::clock` also asserts the footer's clock
equals the reference's.

- `random_clock_traffic_matches_the_reference` (circuit, 20,000 events) and
  `random_clock_traffic_through_the_engine_matches_the_reference` (checkpoint, 5,000
  events) run random enables, selects (mostly 08 to 0C), latches, register writes (halt
  and the carry set and cleared, S, M and H past their ranges), reads, and elapsed times
  from 1 s to past u64::MAX's days. Control traffic and advances are mixed in.
- `the_lockstep_catches_a_wrong_clock` catches seven mutants: a minute short after time
  passes, the carry never set, time passing while halted, a latch that drops halt and
  the carry, a read off by one, a RAM disable ignored, and a seconds write not reduced
  modulo 60.
- The cartridge's clock tests moved to `machine::tests` and drive the bus. Each runs
  twice, with the reference and with the circuit in lockstep: counting and latching, the
  footer, resuming from a footer, a clock that only ticked, a timer without RAM, the day
  carry, and the absurd host time.
- Pokémon Crystal for 300 frames with every component in weights and in lockstep reaches
  `ccfe7ad3` as four checkpoints and as one model, with no divergence. Its control report
  reads `the cartridge clock: 51 writes, 7 reads, 4 wall times handed over` (5 as one
  model, since the wall time is real). So the gate's existing Crystal line already runs
  the clock's writes, latches and reads through the model, and `tools/gate.sh` needs no
  new run.

The run label's host work now names the wall-clock time an MBC3 clock counts as an input,
and each access that reaches the clock's chip as bus decoding (`stack::HOST_WORK`).

## 21. DMA transfers as model outputs

The control checkpoint decided when DMA ran, and the host expanded its decisions. A
general-purpose HDMA pass emitted a block count `CN`, and the host turned it into
`16·CN` byte copies and `8·CN` (or `16·CN`) stall M-cycles. OAM DMA emitted the page
once, and the host added the M-cycles since the pass to the model's index `DI` to name
each M-cycle's byte. At a speed switch the host read the pause from the state `PS` and
ran it as a stall. Now every byte and every stall is a model output, and the host only
places bytes at the address bits the model emitted and runs the stall the model names.

### 21.1 What the model emits

| Slot | Value |
| - | - |
| `FX` bit 6 | the pass moves one HDMA block |
| `CS0 CS1` | the block's source, bits 15-4 (the low nibble emitted as 0) |
| `CD0 CD1` | the block's destination, `8000 \| ` bits 12-4 |
| `FX` bit 7 | the span moved a run of OAM DMA bytes (§21.5) |
| `DP` | the run's source page, E0-FF mapped to the echo of WRAM |
| `DX` | the 8-byte OAM line the run is in (index bits 7-3) |
| `DM`, `DL` | a bit per byte of that line the run moved: `DM` for the span's M-cycles before its last, `DL` for its last; byte j goes to `FE00 \| DX \| j` |
| `SS0 SS1` | the M-cycles the CPU stays stopped after the pass: 8 for a block, 16 at double speed, or a speed switch's 32,772 |
| `HDMA3'` bit 7 (`GD`) | a general-purpose transfer has blocks left |

`CN` and the four general-purpose adders behind it (`gdma n`, `gdma room`, `gdma src`,
`gdma src room`, `gdma dst`) are gone. Inputs are unchanged (56 slots, 7 heads); `EV`
bit 7 is the new `Block` event.

- **HDMA.** A pass moves at most one block, from the source and destination before it,
  and steps them by one block (12 and 9 bits, wrapping). There are three exclusive ways
  a block moves: the HDMA5 write that starts a general-purpose transfer, a `Block` event
  of one, and a mode-0 start of an HBlank one. The write moves the first block and sets
  `GD` unless that block was the last or sat at the end of VRAM. Between the
  transfer's passes HDMA5 holds the blocks left after the next, bit 7 clear, as an
  HBlank transfer's does, so a `Block` event steps exactly as a mode-0 start does. The
  host (`ControlModel::pass_with`) runs `Event::Block` passes in place, with no span,
  while the state's `GD` bit is set. All of a transfer's blocks therefore land at the
  HDMA5 write, as on hardware, before the CPU runs again.
- **OAM DMA.** The first cut ran a pass every M-cycle of a transfer, one byte each.
  §21.5 replaces it with a run of bytes per pass, up to an 8-byte OAM line.
  `ControlModel::lines` no longer forces a horizon of 1, and `ControlModel::advance` no
  longer pushes a copy.
- **Stall.** One 16-bit output covers a block and the pause, which never share a pass.
  `ControlModel::transfers` adds it to `Signals::stall`, and the machine runs it as
  before.

`device::Copy` is now `Byte { source, dest }` or `Block { source, dest }`. A block's
addresses have their low four bits clear (`Copy::bytes` asserts it), and byte k moves
from `source | k` to `dest | k`. `Machine::apply_signals` enumerates those bytes and does
no address arithmetic. The reference `machine::dma::Hdma` asks for a general-purpose
transfer one block per copy, so its requests equal the model's block for block.

### 21.2 Exactness

`ControlLockstep` already compared every call's `Signals` with the reference's. Those
now hold one `Copy` per OAM DMA byte and per HDMA block, each with its emitted address
bits, and the stall summed over the call's passes. For a general-purpose transfer, that
call is the HDMA5 write, which includes all of its `Block` passes. The full state is
compared after every pass. The OAM DMA index is never stale between passes any more, so
`untimed` keeps it.

- `random_dma_traffic_matches_the_reference` (circuit, 20,000 events, CGB) and
  `random_dma_traffic_through_the_engine_matches_the_reference` (checkpoint, 4,000
  events) run general-purpose transfers of up to 128 blocks, many from near the end of
  VRAM, HBlank transfers across lines with cancels and restarts, OAM DMA (sometimes
  restarted mid-transfer, from WRAM and its echo), speed switches with each stall run as
  the machine runs it, and advances. The circuit run moved 163,780 OAM DMA bytes, ran
  39,480 `Block` passes, had 297 transfers cut short by the end of VRAM and 67 that
  started at its last block, and moved 1,864 HBlank blocks at double speed.
- `the_lockstep_catches_a_wrong_dma_byte_or_stall` catches seven mutants: a block's
  source bit 4 and destination bit 12 flipped, a stall one M-cycle short, a double-speed
  block stalled at normal speed, an OAM DMA index one off, an echo source not mapped to
  WRAM, and an HBlank block that never moves. The older list's HDMA mutant is now a
  general-purpose transfer a block short (`GD` cleared before its last block).
- Every gate hash holds with control in weights and in lockstep (the list below).

  | Gate | Hash |
  | - | - |
  | Yellow DMG 8, `--core lockstep` | `6872b003` |
  | Yellow CGB 300, `--core lockstep` | `cfd31f3d` |
  | Yellow, Crystal, Zelda DX, Mario DX 300, all five in weights and in lockstep | `cfd31f3d`, `ccfe7ad3`, `e2bd7038`, `ea7f86ca` |
  | Yellow and Crystal 300 as one model, all in lockstep | `cfd31f3d`, `ccfe7ad3` |
  | Yellow 300 as one model with the step oracle (control every 6,000 passes, 18 checked) | `cfd31f3d` |
  | `instr_timing` (cpu, control), `dmg_sound` 01-04 (apu, control), dmg-acid2 and cgb-acid2 (cpu, memory, ppu, control) | Passed, `99793856` `881f3fe7` `e494d943` `8e776b75`, pixel-exact |

  Mario DX's 300 frames stall 36,512 M-cycles for HDMA, all of it from the model's `SS`
  output, and the reference agrees at every call.

### 21.3 Size (measured, `control_proto --shape`)

| Checkpoint | Wires | Nodes | Steps | Depth | Widest level | Hidden | Intermediate | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - | - | - |
| §20 | 1,835 | 1,387 | 3,592 | 11 | 447 | 1,024 | 640 | 7 | 14,710 | 66.33 M | 66,540 |
| §21 | 1,710 | 1,262 | 3,398 | 11 | 415 | 896 | 576 | 7 | 14,712 | 55.98 M | 62,287 |

The checkpoint shrinks, because one block a pass needs only the one-block step the HBlank
path already had. The five adders that moved a general-purpose transfer's source and
destination by its whole length are gone, which removes 125 wires and 125 nodes. Hidden
follows the wire count and intermediate the widest level (inferred from `Model::shape`, as
in §20.3): 125 fewer wires bring hidden from 1,024 to 896, and the widest level of 415
units brings intermediate from 640 to 576. Three new output slots (`DX`, `SS0`, `SS1`)
less `CN` add two vocab rows. `examples/f16census control` reports 0 violations
under T1 and T2. `examples/digest` gives control `a39e22da` (from `bf493acf`) and the one
model `50e24cc2` (from `baa801d1`). The one model keeps 613.6 M dense parameters
(measured); that a larger part sets its widths is inferred. cpu `87bb33f8`, cpu-memory
`851a553c`, apu `83c6b259` and ppu `e368efef` are unchanged.

### 21.4 Time (measured)

Yellow runs one OAM DMA transfer a frame, so the control model now runs 160 more passes a
frame. On `tools/bench/text.sh --weights all --one-model` the control checkpoint ran
216.4 passes a frame, up from 58.4: 96,000 OAM DMA bytes over 600 frames, a pass each, no
general-purpose HDMA, and the same 76,862 lines drawn at a scheduled mode-3 start as
before. Interleaved against a binary built at §20, six rounds, the first three with
the old binary first and the last three with the new one first:

| Round | Order | Before (fps) | After (fps) | Load (1 min, before and after) |
| - | - | - | - | - |
| 1 | old first | 57.9 | 57.3 | 28.6, 20.7 |
| 2 | old first | 57.8 | 56.7 | 14.0, 14.1 |
| 3 | old first | 56.3 | 55.4 | 9.8, 8.0 |
| 4 | new first | 57.6 | 56.0 | 11.9, 10.2 |
| 5 | new first | 57.7 | 56.3 | 9.3, 10.9 |
| 6 | new first | 57.1 | 56.4 | 17.5, 23.3 |

The median goes from 57.65 to 56.35 fps, 2.3% slower, a little over the brief's 2%. The
rounds run 1.0% to 2.8% slower, and the new-first rounds run slower than the old-first ones
(2.8, 2.4 and 1.2% against 1.0, 1.9 and 1.6%), so the falling load in the first three
favoured the new binary. The frame hash is `870aa3fd` in every run. The machine was shared
with other jobs (the load column), so the absolute fps are contended. A transfer costs 160
control passes, each re-evaluating about 561 units (the step-oracle run's count).

The alternative, not built: a pass at an OAM DMA transfer's start emits the page, and the
next pass runs at the transfer's end or at the PPU's next point, whichever is sooner. The
host would enumerate the indices in between by concatenation, as it does an HDMA block's
low four bits, and land them as the PPU's next read requires. For the indices to be pure
concatenation each run must be aligned to a power of two, so a transfer becomes a few
runs, and the bytes stay model outputs at a few passes per transfer.

### 21.5 OAM DMA in line runs

One pass per OAM DMA M-cycle cost about 2% to 4% of text.sh and overworld.sh as one model
(§21.4), and 160 control passes a transfer. A pass now moves a run of bytes, up to the
end of the 8-byte OAM line the transfer is in. The host lands the run at the pass.

**What the model emits.** While a transfer runs, the horizon candidate `hoam` is 8 less
the low three bits of `DI'`, the end of the next byte's line. A span of D M-cycles then
moves the bytes `DI` to `DI + D − 1`, all in one line, from page `DR` as it was before the
event. The pass emits the page (`DP`), the line (`DX`, index bits 7-3) and a bit per byte
of the line: `DL` for the byte of the span's last M-cycle and `DM` for the bytes before
it. Byte j moved when `DI`'s low bits are at most j and the low bits of the index after
the span are outside 1..=j; it is the last when those low bits are j + 1 (0 for byte 7).
The host asserts that `DL` has one bit, that the masks do not overlap and that the run
has as many bytes as the span had M-cycles.

**Where a run ends.** The model ends a run at anything that could see OAM or the
transfer's position:

- the end of an 8-byte line, from `hoam`;
- every mode-3 start, the PPU's existing horizon;
- every line start, which is mode 2's start on a visible line: while a transfer runs,
  "the line start matters" (`rising ∨ dma on'`) picks the 114-M-cycle case over the next
  mode 3;
- every event: while a transfer runs, `ControlModel::pass` runs each event as a full span,
  never in place. The bytes of a pass that cannot ask the bus for anything (an
  acknowledgement, a raise, a button change, a read, the wall time) wait in
  `oam_pending` and land at the next call that can, before any M-cycle runs;
- every CPU access the transfer blocks (outside HRAM): `Machine` calls
  `ControlDevice::cut` from `Bus::read` and `Bus::write`, and `step::step` reports each
  read a memory core serves itself through `Bus::served_elsewhere`. The cut is a `Tick`
  pass over the M-cycles so far.

A blocked access reads 0xFF and an I/O write is dropped, so none of those accesses sees
OAM. The run ends there anyway: the rule is simpler to check than an argument about
what a blocked access can see.

**What the host does.** `ControlModel::transfers` turns the masks into `Copy::Byte`s at
`page << 8 | line | j` and `FE00 | line | j`: concatenation, no add. `DM`'s bytes go into
`Signals::early`, `DL`'s into `Signals::copies`. `Machine::advance_handing` lands the early
copies and hands them to the core before `apply_signals` calls the renderer, so a line
drawn at the run's last M-cycle sees every earlier byte, as with one byte an M-cycle. The
last byte lands after the line, as it did before. The first cut landed every byte of the
run after the line. `tests/oam_dma_runs.rs` caught it: 4 pixels of one row differed from
the reference machine with control in weights, and lockstep did not see it, because in
lockstep the reference renderer reads the host's OAM, which the model's requests fill.

**Exactness.** `ControlLockstep` keeps both sides' OAM DMA bytes since the last pass
boundary and compares them at every boundary: the model has passed (`elapsed` 0) with
nothing pending, and its run equals the bytes the reference moved, one an M-cycle,
since. Every other request is compared per call as before. The state is compared in
full at every pass; between passes `untimed` drops the OAM DMA index again.

- `oam_dma_runs_end_at_mode_2_mode_3_and_every_access` (circuit, DMG and CGB, 1,500
  transfers each) starts transfers at random points of the line (DMG: 248 in mode 2, 534
  in mode 3, 620 in mode 0, 98 in VBlank). It runs each beside blocked accesses (19,627
  cuts on DMG), acknowledgements, raises, button changes and short advances, in
  lockstep: 240,000 bytes in 67,699 runs.
- `random_dma_traffic_*` (circuit and checkpoint) now include blocked accesses during
  transfers.
- The DMA mutants add a run one byte to the side and a run in the next line. They replace
  the index-one-off mutant, and every mutant is caught.
- `tests/oam_dma_runs.rs` (machine, no ROM file) runs a DMG program that starts a
  transfer each frame at a line and delay that move every frame (mode 2, mode 3 and
  mode 0). Its HRAM routine reads OAM while the transfer blocks it (0xFF) and after it
  ends, and it moves 40 sprites. With `cpu,memory,ppu,control` in weights and all in
  lockstep, 30 frames and the bytes read equal the reference machine's, pixel for pixel.

**Gates** (with the runs): Yellow DMG 8 `6872b003`, CGB 300 `cfd31f3d`, `instr_timing`
Passed, `dmg_sound` 01-04 `99793856` `881f3fe7` `e494d943` `8e776b75`, both acid2
pixel-exact, Yellow, Crystal, Zelda DX and Mario DX 300 with all five in weights and in
lockstep `cfd31f3d` `ccfe7ad3` `e2bd7038` `ea7f86ca`, Yellow and Crystal 300 as one model
`cfd31f3d` `ccfe7ad3`, and Yellow one model with the step oracle `cfd31f3d` (control 11
passes checked). `cargo test --release` passes in full.

**Size** (`control_proto --shape`):

| Checkpoint | Wires | Nodes | Steps | Depth | Widest level | Hidden | Intermediate | Heads | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - | - | - |
| §21, one byte a pass | 1,710 | 1,262 | 3,398 | 11 | 415 | 896 | 576 | 7 | 14,712 | 55.98 M | 62,287 |
| §21.5, line runs | 1,760 | 1,312 | 3,467 | 11 | 426 | 960 | 576 | 7 | 14,714 | 59.98 M | 63,296 |

The masks' 24 nodes and the line-start OR add 50 wires, and hidden follows the wires
(inferred, §20.3). `f16census control` reports 0 violations. Digests: control `d9dcad71`
and one model `07373770` (with `GBCFORMER_BITS=1`, `33fa8c8f`). cpu, cpu-memory, apu and
ppu are unchanged.

**Time** (measured). `tools/bench/text.sh` and `overworld.sh` with `--weights all
--one-model`, three rounds, the three binaries interleaved with the order rotated each
round, load 3.7 to 12.1. The binaries are the base before §21 (`u2`), one byte a pass
(§21) and line runs (§21.5).

| Workload | Before §21 (fps, median) | One byte a pass | Line runs | Control passes a frame (before, byte, runs) |
| - | - | - | - | - |
| text.sh | 57.1 (57.1, 57.1, 58.9) | 55.0 (55.0, 55.1, 54.5) | 57.0 (55.9, 57.0, 58.6) | 58.4, 216.4, 77.3 |
| overworld.sh | 133.2 (129.4, 133.6, 133.2) | 131.4 (130.8, 131.4, 134.6) | 130.7 (130.7, 135.7, 126.4) | 44.0, 202.0, 62.9 |

On text.sh the runs bring the cost from 3.7% to 0.2%, within the rounds' spread. On
overworld.sh the three are within the spread of their rounds (126.4 to 135.7 for the
runs), so the medians do not separate them. Every run reached the frame hash (`870aa3fd`,
`822059c1`). A transfer is now about 19 control passes a frame on these workloads, not
160: 20 lines, less the runs a line or mode-3 start splits plus the cuts. Text stays
under 60 fps on this machine with or without the runs (57.1 before §21).

## 22. The clocks and the HALT wake as model outputs

The host still did four pieces of timing arithmetic for the control logic and the APU
(`docs/host-in-weights.md` §2, items 6 to 9, 11 and 15). It added 4 per M-cycle to the
model's counter for the APU's DIV input and found the frame sequencer's falling edge per
tick, doubled the M-cycles at normal speed for the PPU's timing, tested `IE & IF & 0x1F`
for the HALT wake, and computed `first + period·j` for the scheduled mode-3 starts. Each
is now a model output or the model's own conversion, and the host counts down to what
the model emitted.

### 22.1 What the model emits

| Slot | Value |
| - | - |
| `FE0`, `FE1` bits 0-3 | `E − 1`, for `E` the M-cycles from the pass to where the count next carries out of the frame sequencer's bit (bit 12, bit 13 at double speed). `E = 2^11 − counter[2..=12]` (`2^12 − counter[2..=13]`), so the bits are the counter's inverted, from the counter after the event; only bit 11 needs a node |
| `FE1` bit 4 (`FE_DROP`) | a DIV write or a speed switch reset a set bit (at the speed before the event): it falls at the next M-cycle |
| `FE1` bit 5 (`FE_HELD`) | a pause holds the counter after the event, so no count falls |
| `FE1` bit 6 (`FE_WAKE`) | `IE' & IF' & 0x1F != 0`, the five AND gates and their OR after the event |
| `TC'` bit 4 | the drop above, carried to the next pass while no M-cycle has passed: the counter is 0 then, so a pass in place or a second reset before the next M-cycle keeps the one drop |

`H0 H1` now carry the M-cycles the PPU's position moves, not halves. The circuit shifts
them by its own speed bit into the 13 bits of halves in one level of two-step muxes
before the line position, so `Ctl::encode` passes M-cycles only (`D`, `Q` in a pause,
and `H`).

### 22.2 What the host does

`ControlModel` keeps the edges of the last pass in M-cycles since the last full span: at
1 with `FE_DROP`, and at `E` unless `FE_HELD`. It counts down to the nearer one and the
next stop, cuts the machine's block at an edge (`Lines::horizon`) and sets
`Signals::edge` for that block, with no pass. A pass in place re-emits the same `E`
from the same counter, and an edge already behind the elapsed M-cycles is never handed
twice. The count carries at most once before the next full pass at normal speed (the
period is 2,048 M-cycles, `MAX_SPAN`), so with the drop there are at most two.

The schedule's starts are now counted down: the first at the M-cycle the model emitted,
then a countdown reloaded with the emitted period (`RP`) at each start, and the next pass
when the countdown after the last start runs out. The next pass's `since` is a counter
reset at each start. A pass in place during a schedule must keep its first start,
period, position and the lines already drawn.

`Lines::wake` is the last pass's wake bit, and `step::step` asks the bus
(`Bus::wake`) whether a halted GBC CPU wakes; a bus with no control device keeps the
IE and IF test as the default. `Lines::counter`, `Signals::div`, `device::Clock` and
`Schedule::at` are gone.

### 22.3 The reference, and the one place it moved

The reference `Control` derives the same edges from its own counter (`edge_horizon`:
the drop first, none in a pause, else the carry) and cuts its horizon there, so
`ControlLockstep`'s comparison of every advance's `Signals` holds each edge instant to
it, and its comparison of the lines holds the wake bit. The drop is part of
`ControlState` (`dropped`), so a hand-off right after a DIV reset carries it.

The APU used to find the edge itself, from the counter bit it saw at each 4-dot tick's
end. `signalled_steps_are_the_falls_the_apu_saw` runs 60,000 random advances, DIV writes
and speed switches (writes and switches where a tick begins) and finds the two equal at
every one of over a million ticks. They differ in one case,
which `a_reset_inside_the_tick_the_bit_rose_in_steps_once` pins: at double speed, a DIV
write one M-cycle into a tick in which the count had just set the bit. The bit rose and
fell inside the tick, so the old sampling missed the fall; the control logic now
signals it, as a falling edge of the counter bit does on hardware. A switch's STOP at
double speed with the APU half a tick in is the same case. On Zelda DX's first 300
frames on the host (double speed) the APU's output equals the old one byte for byte once
the sample clock counts finished ticks as before (`docs/apu-in-weights.md` §14), so
neither case occurred there; the frame hashes and the lockstep cannot see it, since
both hold to the new reference.

### 22.4 Exactness

`the_lockstep_catches_a_wrong_edge_or_wake` catches five mutants: an edge one M-cycle
late, one early, a DIV reset's drop lost, an edge while the pause holds the counter, and
a flipped wake bit. `the_lockstep_catches_a_wrong_schedule` adds a period one M-cycle
long. The DMA tests and mutants of §21.5 still pass beside them: an edge inside an OAM
DMA run cuts the machine's block with no pass, so the run still ends only where §21.5
says. Every gate hash holds with the line runs and the I/O dispatch merged in (the
table below).

  | Gate | Hash |
  | - | - |
  | Yellow DMG 8, `--core lockstep` | `6872b003` |
  | Yellow CGB 300, `--core lockstep` | `cfd31f3d` |
  | Yellow, Crystal, Zelda DX, Mario DX 300, all five in weights and in lockstep | `cfd31f3d`, `ccfe7ad3`, `e2bd7038`, `ea7f86ca` |
  | Yellow 300 as one model, all in lockstep, and with the step oracle `cpu=100000,apu=25000,control=6000,ppu=8400` (10, 10, 11, 10 passes checked) | `cfd31f3d` |
  | `instr_timing` (cpu, control), `dmg_sound` 01-04 (apu, control, all in lockstep) | Passed, `99793856` `881f3fe7` `e494d943` `8e776b75` |

### 22.5 Size and time (measured)

| Checkpoint | Wires | Nodes | Steps | Depth | Widest level | Hidden | Intermediate | Vocab | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - | - | - |
| §21 | 1,710 | 1,262 | 3,398 | 11 | 415 | 896 | 576 | 14,712 | 55.98 M | 62,287 |
| §22 without the line runs | 1,735 | 1,287 | 3,438 | 11 | 435 | 960 | 640 | 14,714 | 62.19 M | 62,687 |
| §21.5, line runs | 1,760 | 1,312 | 3,467 | 11 | 426 | 960 | 576 | 14,714 | 59.98 M | 63,296 |
| §22 with the line runs | 1,785 | 1,337 | 3,507 | 11 | 446 | 960 | 640 | 14,716 | 62.19 M | 63,681 |

The 25 new wires are the 13 half muxes, the five wake ANDs and their OR, the drop, bit 11
of the distance, `D+Q>0` and the drop carried in `TC`. On top of the line runs they take
the widest level from 426 to 446 units, past 576 intermediate, which costs 2.2 M dense
parameters; hidden was already 960. The nonzeros grow 385. `examples/f16census all`
reports 0 violations. `examples/digest`: control `7fef034b` (from `d9dcad71`); with the
APU's change (`docs/apu-in-weights.md` §14, apu `bf499b44`) the one model is `ba60eff1`
(from `e83d0c16`, `docs/perf-log.md`), still 628.6 M dense and 0.97 M nonzeros, and
`9642b3ae` with `GBCFORMER_BITS=1`. cpu `87bb33f8`, cpu-memory `096c5372` (`060e022f`
with `GBCFORMER_BITS=1`) and ppu `e368efef` are unchanged.

Time, measured on §22 before the line runs: on
`tools/bench/text.sh --weights all --one-model` the control checkpoint runs the same
216.4 passes a frame and the APU the same 816.2 segments: an edge cuts a block but runs
no pass. Six interleaved rounds against a binary built at §21 (three each order,
1-minute load 10.6 falling to 4.8) give medians of 55.55 fps before and 55.7 after,
within the spread of either side; frame hash `870aa3fd` in every run. With the line runs
the control checkpoint runs about 77 passes a frame there (§21.5); the merged binary has
not been timed.

## 23. Every event over a span, and three more outputs

**No pass shape chosen by the host.** §7's in-place passes needed a host table,
`Event::needs_span`, naming the events the ticks since the last pass cannot touch. With
lockstep off nothing checked it. Every event now runs after the span before it, which
is the order the hardware has, and the table is gone. An event that ran in place now
also moves the ticks it skipped, so a raise or an acknowledgement costs a full pass's
units. `docs/perf-log.md` ("Host rules") has the cost on `text.sh`.

**`HR`, `DD`, `CL`.** `HR` is the 2-dot halves each M-cycle lasts after the pass (2 at
normal speed, 1 at double), from `K1'` bit 7. The machine counts real time by it and
reads no KEY1 bit. `DD` is OAM DMA's destination page (0xFE) and `CL` an HDMA block's
length (16), both constants in the weights. The control device places each OAM DMA byte
at `DD`, the line and the bit the pass emitted, and each block by `CL`. Under lockstep
the lines' rate and every copy are held to the reference's, and the mutants (a speed
switch that keeps the rate, a half-length block, OAM DMA into FD00) are caught.

## 24. The transport's decisions as model outputs

`docs/minimal-runner.md` §2 classified every rule of `src/transport.rs` and found 13 the
host still decided. Each now comes from a pass's output, with the reference computing the
same thing on its own so that `ControlLockstep` compares it, or is shown to decide nothing.

**Outputs and inputs.** `RESULT` gains two slots and `INPUTS` loses six:

| Slot | Value | Replaces |
| - | - | - |
| `RT` | M-cycles from a schedule's last mode-3 start to the next pass (114) | the host reusing `RP` for the pass after the schedule |
| `NE` | the `EV` code of a further pass at once, at the same M-cycle: `EV_BLOCK` while `GD'` is set, else 0 | the host looping on the state's `GD` bit |
| `DX` bit 0 | the run's last byte lands before a line its M-cycle starts (constant 0) | the host's tie between the last byte and the line |
| `FE1` bit 4 | read as the M-cycles to a dropped bit's fall (0 or 1) | the host turning the drop flag into "1 M-cycle" |
| `SS0 SS1` | the M-cycles the GBC CPU stays stopped once the step that runs has ended, a total: `SQ` plus what the pass adds | an increment the host summed across passes and with the pause |
| `SQ0 SQ1` (input) | the M-cycles the GBC CPU stays stopped as the pass begins, the host's countdown of the last `SS` | none |
| `EX` (elapse) | k, for 2^k seconds of wall time | `ED0`-`ED5`, `EH`, `EM` (days, hours, minutes) and seconds in `EX` |

**The stop.** `SS` is a total, so the model decides how stops combine: an HDMA block adds
8 (16 at double speed), a speed switch adds its pause, each after what is owed. The host
counts the total down only through the cycles it runs with the GBC CPU stopped, after the
step's own (`transport::advance`), and hands what is left to the next pass as `SQ`. The
held bit (`CL` bit 7) still routes a block to the step's end; a stall has no route, since
every stop runs after the step. A speed switch in a step whose HBlank block was held now
stops for the block's 8 M-cycles and the pause, the model's sum
(`a_switch_after_a_held_block_runs_after_its_stall`). The reference `Control` keeps its
own count, adding each block and pause and counting down `advance`'s stopped M-cycles,
and `Lines::owed` holds the two to each other after every call. `SS` costs one
`sum_mod` over the 32 bits of `SQ` and the added stall.

**The clock.** The host hands the clock one binary digit of the elapsed seconds a pass:
2^k for each bit k set. The circuit decodes k, splits 2^k seconds into seconds, minutes,
hours, days below 512 and whether the days pass 511 by a table of 64 one-hots and an OR
per output bit, and adds them with the carries it already had. The reference adds all the
seconds at once (`Rtc::elapse`), so the lockstep compares the sum of the digits' passes
with the whole. Wall time is handed in only by `set_time`, once a frame, and only on a
cartridge with a clock when whole seconds passed; a hand-over of d seconds costs
popcount(d) passes where it cost one. A clock that ticks once a second (the scripted
clock of the proofs, or the wall clock at 60 frames a second) hands over d = 1 each
time, so the cost is the same: Crystal 600 frames as one model with
`GBCFORMER_TIME=1700000000` hands over 9 times and runs 308,450 control passes before and
after (`docs/perf-log.md`). Only a gap, such as a save loaded after a day away, takes more
passes, once: at most 17 for anything under a day and a half, at most 64 in all.

**Copies at once.** A pass at an event that asks the bus for nothing (a read, an
acknowledgement, the buttons, the wall time, a lock query) hands its run's OAM DMA bytes
to its caller, which lands them at once (`ControlDevice` methods take `landed`). No pass
keeps bytes for a later call, and the lockstep compares the bytes at every pass boundary,
reads included. A blocked access always runs a pass, at any M-cycle.

**The line and its bytes.** The reference's OAM DMA puts the bytes of a block's M-cycles
before its last into `early` and the last one after the line, as the model's run does by
`DX` bit 0. The lockstep's byte stream carries each line drawn where it fell among the
bytes, so a run whose last byte lands on the wrong side of its line diverges
(`the_lockstep_catches_a_run_s_last_byte_before_the_line_it_starts`).

**The buttons.** The buttons pass reports the change it made to the held bit (STOP's hold
of the PPU's VRAM), as STOP's pass does, and the machine and the runner release the PPU
by it rather than by the pause flag's change (`Passes::pass_holding`).

**Mutants**, each caught, by the lockstep or, for a stop total below what is owed, first by
the model-side guard that a pass's total never falls below `SQ`: a dropped bit's edge one M-cycle late; the pass
after a schedule one M-cycle late; a general-purpose transfer's further blocks never asked
for; a stop emitted as the increment rather than the total; a switch's pause that drops
the stop owed; a run's last byte before its line; a fresh OAM DMA that blocks one M-cycle
early or late (`DZ`, from the access-timing work). `tools/runner/tests/transport.rs` drives
the shared transport with a fake stream emitting values no model of today emits (a drop
of 5, a tail unlike the period, further passes at named events, a total that shrinks), so
a host that kept its old rule fails there.

## 25. Mode 0 at the dot

A CGB ends mode 3 at the dot, where a DMG rounds SCX's fine scroll up to the next M-cycle
(`machine::lcd::Lcd`, docs/hardware-tests.md §3.14). Three things moved for it.

- **The length in dots.** The PPU checkpoint, the length model and the stepper publish a
  line's length as the dots to mode 0 less 128 (`lcd::LENGTH_BASE`): 44 for the least line,
  a byte for any line. A DMG's length keeps the fine scroll rounded up (`fifo::mode0_dots`),
  a CGB's is exact. `ML` is that byte; the stepper's two-byte dots reach it through the
  wiring's `bias` (`wiring::Stepper`), with no division.
- **Two mode-0 predicates over the same position.** The control logic's position is in
  2-dot halves already (`LM`, one half an M-cycle at double speed), so no state was added.
  A tick's mode, which the STAT interrupt, the locks, the palette ports and the HBlank block
  use, is taken at its second M-cycle at double speed: `4m ≥ D + 77 + 3·dk + 2·lk`. A STAT
  read, an HDMA5 write and a HALT see the half they land at: `2p ≥ D + 77 + 5·dk + 2·lk`.
  `dk` is double speed and `lk` a CGB's line 0 after the LCD turned on, each counted only
  with the length known. Both are linear in `ML`'s bits once halved: `⌈(ML + c) / 2⌉` is
  `⌊ML / 2⌋` plus a constant and `[ML.0 ∧ dk]`. Mode 0's tick for the horizon is
  `⌊ML / 4⌋ + 52` plus two carries of `ML`'s low bits with `3·dk + 2·lk`.
- **An LCD turned on at double speed starts at M-cycle 1.** The state after the event
  (`LM'`) takes it, and the horizon's distances count from it (`m_neg` gains the turn-on).
  At double speed a CGB's line 0 reads mode 0 through M-cycle 20.
- **A speed switch moves the edges at once.** The tick's mode after a STOP that switches
  takes the new speed (`K1'` bit 7), and the reference recomputes the STAT line there
  (`Lcd::set_double`), so a line that the new edge raises rises at the switch. That is a
  rule chosen so the two agree, inferred: no source or test pins what the STAT line does
  at a switch. The random
  control stream found the old order: the locks after a switch at M-cycle 80 of a line
  whose mode 0 falls between the two speeds' edges.

Checks (measured on this branch): the control lockstep's random streams (DMG, CGB with
switches, the schedule, the mutants), a sweep of 12 STAT-only seeds of 100,000 events
with three CGB seeds at double speed (a temporary edit, not committed), `tests/
interrupt_fuzz.rs`, and the gate games, acid2 and daid's speed-switch tests with every
component in weights in lockstep.

## 26. The HBlank HDMA request at its own edge, and HDMA5 counted at the step's end

Mealybug `dma/hdma_timing-C` reads STAT, HDMA5 and DIV around one HBlank block at SCX 1
and 2, at both speeds (docs/hardware-tests.md §3.14). Three rules close it, each in the
reference and the control checkpoint.

- **The request's edge.** The HBlank block moves at the half p of a visible line with
  `2p ≥ D + 79` at normal speed (at a tick's start) and `D + 83` at double speed, two dots
  after STAT's mode 0 at normal speed and one at double (`Lcd::request_at`). In the circuit
  `past_rq` is that edge's distance, linear in `ML`'s bits like the mode-0 forms (§25), and
  the block moves at its first position. The horizon's case `to_rq` lands there: the
  request's tick, `⌊ML / 4⌋ + 52` plus a carry at normal speed and `⌊ML / 4⌋ + 53 +
  [ML.1 ∧ lk]` at double. An odd half at double speed is one M-cycle past its tick's start:
  that pass closes every horizon gate (`one_more`), so the next pass is one M-cycle on, as a
  wake's is.
- **HDMA5 counted at the step's end.** A block the request moves inside a GBC CPU step keeps
  HDMA5's count until the step ends (SameBoy v1.0.5 counts it in `GB_hdma_run`, after the
  next opcode fetch). HDMA3 bit 5 (`HP`) holds it; a pass at the step's end
  (`Event::StepEnd`, a write to FF0B) counts it, and an HDMA5 write drops it. The host runs
  that pass only while `HP` is set (`ControlModel::step_running`, the runner's
  `SpanStream::step_ended` by the wiring's `uncounted` flag), as it runs the HALT's and the
  wake's (§22, docs/minimal-runner.md).
- **A run's extra M-cycle.** Each run of the copier stops the GBC CPU one M-cycle beside its
  blocks (`dma::RUN_MCYCLES`): an HBlank block, a wake's block, and a general-purpose
  transfer's first block (its `Block` events continue the run). The stall `SS` adds it.

## 27. One HBlank block a line across the gap

§25 and §26 split STAT's mode 0 and the HBlank request into two edges, which fall an
M-cycle apart at normal speed at SCX fine 2, 3, 6 and 7 (one half at double speed at even
fine values). Nothing recorded that a line's block had already moved, so an HBlank transfer
started in that gap (`FF55 = 0x80 | n` while STAT reads mode 0) moved a block at once and the
request then moved a second on the same line, and a HALT entered there lost the line's block
(the HALT kept "began in mode 0", and the request is held off while halted). A review of
the unit before found it; `machine::tests::a_line_moves_one_hblank_block_when_the_transfer_starts_in_mode_0`
and `a_halt_between_mode_0_and_the_hblank_request_keeps_the_line_s_block` fail on that unit's
reference at fine 2 (two blocks; none) and pass now.

- **Reference.** `Hdma::line_done` is set when a block moves in the gap (`Lcd::in_gap`: STAT
  reads mode 0, the request still to come): at the HDMA5 write that starts a transfer there,
  and at a wake there. The request then moves no block, and every request clears it. A HALT in
  the gap keeps the block owed unless it moved (`Control::halt`). One block a line is the rule
  chosen; SameBoy's request and mode 0 fall together, so it says nothing of the gap, and the
  rule is inferred (`docs/hardware-tests.md` §5).
- **Circuit.** `DN` is HDMA3 bit 6 in the state. `DN'` is set by the HDMA5 start (`hnow`) or
  the wake's block (`hwake`) with the event in the gap (`in the HBlank gap`: the LCD on, a
  visible line, the read past mode 0, not past the request), and holds while the span stays
  inside the line before its request (`past the HBlank request`, `cross`), so a span that
  passes the request clears it whether or not a pass lands there. `hblock` reads `¬DN`, and
  `FL.7'` keeps the block owed for a HALT in the gap with `DN` clear. Between passes the model
  holds the state of its last pass while the reference has run on, so `untimed` leaves `DN`
  out of the comparison there; every pass boundary compares it.
- **Coverage.** The random DMA stream counts its HBlank starts and HALTs in the gap and requires
  at least one of each, and accesses to FF55, FF53 and FF54 later in a step that held a block
  (reads, new counts, new destinations), at least five of each, all in lockstep.
