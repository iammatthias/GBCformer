# Hardware-shaped circuits: what the Game Boy's own design buys the compiled model

This document explains how each compiled circuit (the GBC CPU with memory, the PPU, the
APU and the control logic) could be reshaped to follow how Nintendo and Sharp built the
hardware, and what each reshaping would buy on the two cost models of
`docs/architecture-census.md` §1. Units 1 (C4) and 3 (the banks once per pass, the first
half of C2) of the build order (§7) were built. Unit 2 (C1) and C2's one bus address were
built, measured and rejected. §2.6 records what each measured, and §13 to §21 record what
later work built or probed. The rest of §2 to §9 is not built. Every
number is labelled **measured** (a command in §11), **estimated** (arithmetic
on measured inputs, with the assumption named) or **source** (a hardware fact, cited to
§10).

The short answer, in numbers:

- **The hardware's lesson that pays most is the order of operations, not the width of
  its units.** The GBC CPU's hardware latches the opcode into IR during the previous instruction's last
  M-cycle [H2] and decodes IR. It drives one address onto one bus, and the MBC adds the
  bank on the upper ROM pins [H3]. Our GBC CPU with memory does the opposite in both
  places. It decodes the opcode straight out of a fetched 8-byte line, so 340 units read
  PC's low three bits (measured), 256 of them the decoder's (`OP0` family). That family
  alone costs 655 of the 2,087 gate updates per pass on text (measured). It translates
  nine candidate addresses in parallel, which is 4,712 units, 43% of the circuit
  (census, measured).
- **When the opcode and data bytes arrive as inputs, the GBC CPU's own logic costs 616 updates
  per pass (measured, the host-fed GBC CPU on the same text workload). The GBC CPU with
  memory costs 2,101.** The gap, about 1,480 updates, is memory plumbing, and the
  hardware shape removes most of that plumbing.
- **The hardware-shaped GBC CPU.** It combines a prefetched IR, bank keys with one
  translated bus address, and word-line selects. It projects to about 6,200 circuit units
  against 10,959, the same depth of 10, and about 1,300 updates per pass against 2,087
  (all estimated). Engine time projects to about 3.9 µs per step against 4.8, which is
  about 48 fps on text (estimated, optimistic by the census's warning). That is about
  half of the way to 3.1 µs.
- **Circuit shape alone reaches 60 fps on text only at about 620 updates per pass.**
  That figure is estimated from the census's µs split, and it is the host-fed GBC CPU's
  count. Only byte-valued channels (d4) or cuts to the engine's fixed 1.5 µs per step can
  close the rest.
- **The PPU's hardware lesson is the fetcher, which fetches what this pixel needs when it
  needs it.** Today every line fetches 21 tiles for the background and 21 for the window
  in shared units. Each of the 160 pixel rows then selects among them, and every row
  computes all 31,073 units in llama.cpp. In the hardware-shaped PPU each pixel row
  fetches its own tile, palette entry and object pixels with position-dependent heads.
  It projects to about 12,000 units, 57 heads against 125, and a widest boundary of
  about 900 live wires against 1,802 (estimated).
- **Combined target.** The target is one model as a prefix overlay, with widths rounded
  to 64 and the hardware-shaped GBC CPU and PPU. It projects to about 338 M dense
  parameters (today 2,057.9 M measured; the census's (a') 867 M), about 0.64 M nonzeros
  (today 1,092,918), and 2.3 to 3.1 llama.cpp hours to frame 300 (today 11.0 measured).
  The per-pass floor (3.6 ms of the GBC CPU's 6.3 to 8.3 ms) then decides.
- **Serialising the GBC CPU into one pass per M-cycle loses on both models.** It projects
  to 4.2 to 5.4 hours, against 2.5 to 3.3 for the hardware-shaped one-pass GBC CPU. In
  the engine, if the fixed 1.5 µs per pass does not shrink with a shallower pass, the
  fixed cost alone over 2.6 M-cycles per instruction (measured mean) passes the 3.1 µs
  budget (estimated).

## 1. Method

The two cost models are the census's. llama.cpp (and MLX, Core ML) pays dense
dimensions per pass, plus a fixed floor of about 2.6 ms and 1.0 ms of harness for the
GBC CPU. Our engine pays the gate updates and channel changes per step (census §1). The
llama.cpp projections come from `bun tools/census/project.ts`, which scales the H9
ablation components by each variant's dims (census §5.1). This study added three rows to
it. The engine projections use the census's split of the measured 4.8 µs GBC CPU step.
The gate walk takes 2.3 µs and scales with updates, the change-driven loops take 0.97 µs
and scale with wire changes, and 1.53 µs is fixed (census §5.6, `docs/harness.md` §9.4).
The walk does not shrink linearly when the cheapest updates go (`docs/harness.md` §9.1),
so every engine projection here is an upper bound on the gain.

New measurements on this branch:

- `examples/hw_shape.rs` (since removed) read the compiled
  circuits statically, in seconds, with no checkpoint compile. `--forms` gives the GBC CPU's control forms, `--fetch` PC's readers
  and the fetch families, and `--ppu` the PPU's shared and positional split, its heads and
  its live wires per boundary.
- `GBCFORMER_UNITS=1 tools/bench/text.sh --weights cpu,memory,apu,control` gave the GBC
  CPU's, APU's and control's updates per family, which closes the gap census §8 named for
  the APU and control. Frame `870aa3fd`. The µs in census mode come from the census
  build and do not measure speed.
- `GBCFORMER_UNITS=1 CORE=model tools/bench/text.sh` measured the host-fed GBC CPU on the
  same workload. Frame `870aa3fd`.

## 2. The GBC CPU, with memory

The GBC CPU makes 72% of the passes to frame 300 and takes 74% of llama.cpp's hours, and
it alone accounts for the engine's gap to 60 fps (census §3).

### 2.1 How the hardware does it

| Hardware fact | Source |
| - | - |
| One 16-bit address bus; one memory access per M-cycle | [H1] ch. 3, [H2] §5.1 |
| The next opcode is fetched during the last M-cycle of the current instruction (fetch/execute overlap) and held in IR, an 8-bit register of the register file | [H2] §5.1, fig. 5.4; [H1] ch. 4, fig. 4.3 |
| The decoder reads IR, the M-cycle state, CB mode and the interrupt line. It is three levels of NAND/NOR trees with domino logic, not a ROM. A behavioural HDL model has 97 final decoder outputs | [H6]; [H5] `sm83_decoder_pkg.vhd`, `decoder.vhd` (counts from a summary of the VHDL, not a hand count) |
| The ALU is 8-bit: two 8-bit input ports. The 4-bit ALU is documented for the Z80 [H8], not for the GBC CPU; the GBC CPU's sources contradict it | [H1] ch. 4; [H5] `alu.vhd` |
| A separate 16-bit IDU increments or decrements the address-bus value, independently of the ALU | [H1] ch. 4 |
| WZ holds operands and addresses inside an instruction (e.g. `LD A,(nn)`: Z then W read at PC, then WZ on the address bus) | [H1] p. 30; [H7] `reg_wz` sheet |
| MBC1 drives ROM A14 to A20 from its bank registers; A0 to A13 go straight to the ROM. RAM A13, A14 come from BANK2 | [H1] §13.2, §13.3 |
| MBC1 maps a written bank 0 to 1; mode 1 applies BANK2 to 0x0000 to 0x3FFF too. MBC5 has a 9-bit ROM bank and a 4-bit RAM bank. CGB SVBK maps bank 0 to 1; VBK uses bit 0 | [H1] §13, §17.1; [P1] "CGB Registers", "MBC5" |

### 2.2 How our circuit does it today

The values come from the census (measured) and `examples/hw_shape.rs` (measured, this
branch):

| | Value |
| - | - |
| GBC CPU with memory | depth 10, 10,959 units, 1,240 live wires at the widest boundary, 10 memory heads + 2 copy heads |
| Opcode decode from the line | `OP0.lo*/hi*` 256 units at level 3 (each nibble pattern picks its byte out of the line by PC's low bits and matches it in one node); `OP0=xx` 256 ANDs at level 4 |
| Readers of PC bits 0 to 2 | 310 steps, 340 units: `OP` 256, `pc+` 69, `PC'` 6, `F` 5, `W` 3, other 1 |
| Updates per pass, text | 2,087 (2,101 in this branch's run); the `OP0` family alone 655 |
| Fetch translations | `F0` 249, `F1` 265, `F2` 265 units at levels 1 to 3 |
| All translations and physical selects | 4,712 units (43%), 210 updates per pass |
| Byte selects from fetched lines | `OP1[`, `OP2[` 64 units each at level 4; `D0[` 128, `D1[` 64 at level 8; 312 updates per pass |
| Write-line merge | `WL` 640 units at level 10; 227 updates per pass |
| 16-bit adders | 1,165 units, 162 updates per pass |
| ALU | 611 units (5.6%), 78 updates per pass (3.7%) |
| The host-fed GBC CPU on the same workload | depth 5, 6,012 units, **616 updates per pass**, 95.8 (changed channel, reading layer) pairs against the GBC CPU with memory's 218.3 |

**Control forms (measured, `--forms`).** Every step that reads an opcode one-hot carries
one control form, its terms over the one-hots. A form is one line of the decode ROM.

| Circuit | Distinct forms | Plain opcode sets | Negated sets | Weighted | Single opcodes | Steps reading the one-hots | One-hot fan-out (steps): mean, median, max |
| - | - | - | - | - | - | - | - |
| host-fed, OP0 one-hots | 173 | 155 | 16 | 2 | 90 | 1,788 (2,180 units) | 19.8, 18, 258 (CB) |
| host-fed, CB one-hots | 233 | 223 | 10 | 0 | 192 | 1,957 (2,070 units) | 24.3, 20, 60 |
| memory, `OP0=xx` | 170 | 149 | 15 | 6 | 82 | 1,561 (2,174 units) | 19.2, 14, 171 |

The circuit needs about 170 distinct decode lines for the unprefixed opcodes. That is the
same order as the 97 final decoder outputs of the HDL model [H5], which also decodes per
M-cycle and so needs fewer per-instruction lines.

### 2.3 The reshaping

There are five changes, each named after the hardware structure it copies. C2 is the
census's d1, which the project adopted.

**C1. Prefetch the opcode into IR (fetch/execute overlap).** Make IR a GBC CPU state byte
like PC. A pass reads IR as an input slot and ends by fetching IR' at PC' with one head
(the line of PC', its byte picked by PC' bits 0 to 2). This is the hardware's last
M-cycle. The opcode's nibble patterns then read IR's four bits each (32 units, level 1),
and the 256 one-hots are ANDs at level 2, two levels earlier than today. The adders and
the IR' pick read PC's low bits, and the decoder does not.

- *Forwarding.* The hardware's overlapped fetch comes after the instruction's writes. The
  IR' fetch must see this pass's writes in two cases. A write can hit PC's line, which
  needs a 13-bit equality and a byte merge. A write can hit a banking register when PC'
  lies in a banked window, which needs the post-write bank as a mux over the write
  record. Forwarding costs about 60 units (estimated).
- *Interrupts and HALT.* A dispatch fetches IR' at the vector, as the hardware discards
  its fetched opcode. A halted pass fetches at PC. The HALT bug keeps PC, so IR' is the
  same byte.
- *Host-served fetches.* Today a fetch that lands where the host serves the byte reads
  VOID's zero line. The host serves the I/O page, IE, the unusable area FEA0 to FEFF,
  cartridge RAM the image does not hold, and, under OAM DMA, every region but HRAM (the
  void cases of `sm83/translate.rs`). The host has already placed the byte in the HOST
  block, because it knows PC before the pass (`fetch_void`). Under C1 the host learns PC'
  only after the pass, so it cannot place that byte in time.
- *The fetch-only pass.* It is the same circuit with `exec` off. Every field keeps its
  value, the pass takes no M-cycles and makes no writes, and it fetches IR' at PC' = PC.
  The host knows PC before this pass, so the HOST-block mechanism works unchanged, and
  the circuit needs no second fetch path. Two bus facts trigger it, and neither is a
  mapper rule. One is the tail's own void flag (IR' landed in a host-served region, a
  model output bit). The other is the host's address equality when a write that is not
  the GBC CPU's (an HDMA or GDMA copy) lands on PC's byte after IR' was fetched. How often
  Yellow needs it is not measured (§9).
- *Units.* The change is −761 (`OP0.` 256, `OP0=` 256, `F0` 249) and +32 nibbles, +256
  ANDs, about +128 for the IR' pick and +60 for forwarding, net about −285 (estimated).
- *Updates.* The `OP0` family's 655 becomes about 130, since IR's four changed bits each
  reach 16 nibble units and each changed nibble one-hot reaches 16 ANDs. The IR' pick adds
  about 115, because PC' bits 0 to 2 change about 1.8 times a pass and each reaches 64
  steps. Word lines cannot help there, since PC' arrives at the last field level.
  Forwarding adds about 20. Net about −390 (estimated).
- *Depth.* The one-hots sit at level 2, `RA0` at 3 and the data head at layer 4. The
  operand select, the ALU, the flags and the field mux then put PC' at level 8 or 9, and
  the IR' head and pick at 10. The depth stays 10 (11 layers), with the fetch moved from
  the front to the tail (estimated from the census's level list).

**C2. One bus address, translated once, with bank keys (d1).** The hardware selects one
address and lets the MBC drive the bank pins. Compute the effective banks once from
`REGS` at level 1, about 50 units. Key each cell by (bank, address). The host places
records at slot = bank bits followed by address bits, a fixed rule (a project rule).
Select each access's bus address first with the existing `RA0` mux, so every head's
query is linear in the address and bank wires. That gives one data-read head where `D0h`
and `D0r` are two today, and one write address with its neighbour for the two-write
steps.

- *Units.* The 4,712 translation units minus `F0`'s 249 (counted in C1) become about 250
  of region decodes (VOID, the I/O page, OAM DMA), net about −4,210 (estimated).
- *Updates.* The 210 updates become about 40. Wire changes fall by about 13 per pass
  (census, the translated-address wires).
- *Depth.* Selecting first costs the `RA0` level that the parallel copies save. With C1
  the one-hots sit two levels earlier, which pays for it, and the data path then leaves
  room for the IR' fetch at the tail.

**C3. One IDU.** The hardware has one 16-bit incrementer on the address bus. Today the
circuit carries `pc+1`, `pc+2`, `pc+n`, `sp±n`, `pair±`, `hl+pair`, `sp+e8` and `a16+1`
as separate adders. A pass needs PC's increment and at most one register increment or
decrement in parallel, so one operand mux plus one ±1/±2 adder replaces `pair+`,
`pair-`, `sp+` and `sp-`. The change is about −300 units and −40 updates (estimated, low
confidence). It adds a mux level on a path that has slack in the hardware-shaped
schedule (not checked).

**C4. Word-line selects.** A RAM's address decoder turns the low address bits into one
active word line. Today each byte pick's eight steps read PC+1's (or PC+2's) three low
bits directly, so each changed bit reaches 64 steps. Instead, compute the one-hot of
`(PC + k) & 7` for k = 0, 1, 2 straight from PC's bits at level 1 (24 units, no adder).
Each pick step then reads one word line, and a PC increment changes two word lines, each
read by 8 steps.

- *Updates.* The `OP1`, `OP2`, `D0` and `D1` picks' 312 become about 110 (estimated).
- *Units.* The change is +24.

**C5. Keep the 8-bit ALU, and keep the one-level adder.** The 4-bit ALU is not the
GBC CPU's ([H1], [H5]), and the ALU is 5.6% of units and 3.7% of updates. Splitting it would
add a level to save little. The circuit already splits adders where fan-in hurts (`pc+1`
at bit 7).

**Decode table (d2), rejected on the measurement.** At 173 OP0 forms plus the CB fields,
a table head's values would need at least three 64-dim heads at the decode layer. Once
the decode reads IR instead of the line (C1), the 256 ANDs cost about 64 updates. The
table would save at most those 64 and the 256 AND units, at the price of a prefilled
region in every engine and oracle. The updates came from decoding out of the line, not
from the decode itself. Delivering the forms through IR's embedding is worse, because a
one-hot input forces `kv_heads = heads` model-wide (`llama.rs`, "unless an input carries
a one-hot"), which multiplies the KV cache of 141,415 cells.

**d3 (control wires instead of one-hots), not worth it.** 173 forms in place of 256
one-hots save at most 83 live wires for 173 units (measured counts).

### 2.4 What it buys

| | Today (measured) | Hardware-shaped C1 to C4 (estimated) |
| - | - | - |
| Circuit units | 10,959 | about 6,200 |
| Own units (routing and clearing, ×1.22 as today) | 13,379 | about 7,500 |
| Depth (layers) | 10 (11) | about 10 (11) |
| Live wires, widest | 1,240 | about 1,000 |
| Memory heads | 10 + 2 copy | 9 + 2 copy (REGS ×2, OP1, OP2, D0, D1, WL ×2, IR') |
| Updates per pass, text | 2,087 | about 1,300 |
| Wire changes per pass | 144.5 | about 132 |
| Engine µs per step | 4.8 | about 3.9 (1.53 + 2.3 × 1,300/2,087 + 0.97 × 132/144.5) |
| Text fps (5,374 steps a frame) | 38.7 | about 48 |
| llama.cpp ms per GBC CPU pass, prefix overlay (`project.ts`) | 7.3 to 9.2 ((a')) | 6.3 to 8.3 |
| Own checkpoint dense, 64-rounded | about 142 M | about 89 M (hidden 1,472, 11 layers) |

**The remaining gap.** At 1,300 updates the memory plumbing still costs about 680 over
the host-fed GBC CPU's 616 (estimated). The largest remaining items are the write-line
merge (227, unchanged), the IR' pick (about 115) and the byte picks (about 110). Byte
cells for RAM (census d5, the hardware's one-byte write) remove the first and bring the
step to about 3.6 µs (estimated). Below that, only d4 (byte-valued channels, approved for
a host-fed prototype) or the engine's fixed 1.53 µs can move it.

### 2.5 What breaks

- **The step contract (`step.rs`).** `Cpu` gains IR. The reference interpreter computes
  IR' as the byte at PC' after its writes, so lockstep compares it. `Core::audit` holds
  the model's IR' fetch to the host's byte.
- **SST.** The cases give the opcode in the initial RAM at PC, and their bus log has no
  prefetch (`vendor/sm83/v1/00.json`: NOP's one cycle reads PC). The harness runs the
  fetch-only pass first, so every opcode in the 500,000 cases is a model fetch, then the
  case's pass, and checks IR' against the final RAM at the final PC (0 when the case
  lists no byte there, as `peek` already does). The cases stay checkable. Forwarding
  needs its own tests, with random writes to PC's line and to the bank registers.
- **Exactness.** C1, C3 and C4 add nothing new, since they are ordinary threshold nodes
  and one more memory head. C2 changes the key format of `docs/stock-llama.md` §4 (about
  30 key dims). Its T2 margin needs S = 512 and a re-check (census §5.6). VOID's zero
  line and the region decodes stay.
- **Digests.** The host-fed checkpoint `6bd416c1` does not change, because C1 to C4 touch
  only the memory circuit. The cpu-memory digest and the one model's (`b03d896c` on this
  branch, census §2) change at each unit, as do the GGUF, the f16 census and
  `src/sm83/changes-memory.txt`, which must be re-recorded after each circuit change.

### 2.6 Built

**Unit 1, C4 word lines (measured).** Every pick of a byte from a fetched line reads one
word line, the one-hot of the access's low three address bits (`sm83/line.rs`,
`line::words`), in place of the three bits. The fetches' word lines are those of
`(pc + k − hb) & 7` and read 1's are those of `(sp + 1) & 7`, all computed straight from
the registers at level 1. Read 0's word lines come from its sources' low bits under their
selects, at the level of the selection. They are all 0 when nothing is selected, and the
head then reads VOID's zero line. k = 0 feeds the opcode's 32 nibble nodes, so C4 also
cuts the decode's updates, which C1 deletes anyway. The table separates the two.

| | Before (measured) | After (measured) | Estimate |
| - | - | - | - |
| Circuit units | 10,959 | 11,063 | +24 |
| Depth (layers) | 10 (11) | 10 (11) | same |
| Live wires, widest | 1,240 | 1,265 | |
| Readers of PC bits 0 to 2 | 340 units | 124 (`pc+` 69, `OP1w` 16, `OP2w` 16, `OP0w` 8, others 15) | |
| Updates per pass, text, `model-mem` | 2,091 | **1,498** | about 1,890 |
| of which byte picks (`OP1`, `OP2`, `D0`, `D1` and their word lines) | 312 | **153** | about 110 |
| of which opcode decode | 702 | 348 | not estimated |
| of which operand muxes (read 0's byte enters them through its picks) | 144 | 55 | not estimated |
| Updates per pass, overworld | 1,921 | 1,372 | |
| Updates per pass, text, one model's GBC CPU stream | 2,087 | 1,480 | |
| Text fps, every component in weights, one model, GGUF (median of 3) | 38.1 | **41.6** | |
| Overworld fps, the same | 73.6 | 74.7 | |

The engine's shared layers fell from 3.85 to 3.46 µs a pass on text for 607 fewer
updates (0.64 ns each), against the census's 2.3 µs for 2,087 (1.1 ns each). The updates
that went were the decoder's and the picks' runs of identical readers, which are the
cheapest updates the block walk makes (`docs/harness.md` §9.1). The digests moved:
cpu-memory from `ffe22414` to `b63c3759`, and the one model from `b03d896c` to
`466f7c72` (2,057.9 M to 2,043.2 M dense, because the widest level fell from 6,336 to
6,272 units). The host `6bd416c1` is unchanged.

**Unit 2, C1 prefetched IR (measured, rejected).** It was built on top of unit 1 and
measured, not merged. The
decoder read IR, an input slot the host carried between steps, and each pass ended by
fetching IR' at PC'. Circuit units went from 11,063 to 11,015 (−48 against −285
estimated), at depth 10. Updates per pass on text went from 1,498 to 1,399, −99 against
−390 estimated, because unit 1's word lines had already taken 354 of the decoder's 702
updates and the IR' fetch needed a translation of PC' the estimate did not count. Text
fps fell from 41.6 to 39.7 (4.6% slower) and overworld from 75.9 to 74.4, with every
component in weights, one model, GGUF, median of 3. IR's eight bits are an input that
changes on most steps (layer 0 saw 8.4 changed channels a pass against 4.5), and the
updates that went were the decoder's runs of identical readers, the block walk's
cheapest. The branch's own §2.6 has the full table.

**C2 mux-first, one bus address (measured, rejected).** It was built on C1 and the banks
below, and not merged. The circuit selected read 0's
bus address first among its sources and translated it once, and did the same for each
write's address, as the GBC CPU drives one address onto one bus. Updates per pass on text
rose from 1,399 to 1,998 (measured). The selected address changes whenever any selected
source or the selection changes, so its one translation re-evaluates on most passes,
while the parallel translations of HL and SP change only with those registers (inferred
from the counts, not traced).

**Unit 3, the banks once per pass (measured).** This is the first half of C2, without C1
and without one bus address (`Regs::banked`, `sm83/translate.rs`). The four MAP
translations of the banked windows' first bytes (0000 under MBC1's mode, 4000, A000,
D000) read only the registers and become the effective banks: the MBC's bank pins with
MBC1's quirks and the masks, the RAM's enable, bank and the MBC3 clock, and SVBK. Every
other read translation takes its window's bank bits from them in one case, where it
had a case per mapper. VBK stays a literal pair (its bit and the CGB flag). Writes that
reach a banking register keep their cases per mapper, since they decode the MBC's
register map, not a bank. The MAP slots are these wires.

With C1 the banks measured 9,343 units (−1,672) and 1,360 updates (−39),
with the opcode fetched at the tail. Without C1 the opcode fetch is at the front.
Reading the banks moves its head from layer 2 to layer 3, and the circuit gains a level
(depth 11, 12 layers, 9,394 units, text 1,464 updates, measured). The opcode's fetch
(F0) therefore keeps reading the registers directly (`Regs::unbanked`), and F1, F2 and
the data reads, which have the slack, take the banks.

| | After unit 1 (measured) | Every translation banked (measured) | Unit 3, F0 unbanked (measured) |
| - | - | - | - |
| Circuit units | 11,063 | 9,394 | **9,517** |
| Depth (layers) | 10 (11) | 11 (12) | 10 (11) |
| Live wires, widest | 1,265 | 1,161 | 1,292 |
| Checkpoint (cpu-memory) | 175.3 M, intermediate 1,792 | 164.6 M, intermediate 1,472 | **162.3 M**, intermediate 1,600 |
| Updates per pass, text, `model-mem` | 1,498 | 1,464 | **1,470** |
| of which address translation | 210 | 178 | 183 |
| Updates per pass, overworld | 1,372 | 1,349 | 1,351 |
| (changed channel, reading layer) pairs, text | 218.1 | 220.6 | 217.1 |
| Updates per pass, text, one model's GBC CPU stream | 1,480 | | 1,452 |
| Text fps, every component in weights, one model, GGUF (median of 3) | 41.7 | | **41.4** |
| Overworld fps, the same | 75.3 | | 75.2 |

The units fall by the per-mapper cases of every read translation, but the updates move
little (text −28, all of it in the translation, from 210 to 183). A bank changes only
when a banking register is written, so the circuit rarely re-evaluated most of the cases
it replaces. The engine does not move. The shared layers took 3.45 to 3.49 µs a pass on
text on both sides, and 28 updates at the walk's 0.64 ns are under 0.02 µs, inside the
runs' spread. The unit pays off in the circuit's size (1,546 fewer units, cpu-memory from
175.3 M to 162.3 M dense, 0.25 M to 0.23 M nonzeros, the one model from 1.49 M to 1.47 M
nonzeros; the PPU's width sets the one model's 2,043.2 M dense) and in the banks as
wires for the work that follows. The MAP slots are now the model's effective banks,
which the DMA placement reads (`docs/memory-in-weights.md` §8a). The DMA fix that follows
from the project's rule (the host places records only by concatenation of values the model
emitted) costs nothing measurable, with text at 41.5 against 41.5 fps for the banks alone
against the banks with the fix. The digests moved: cpu-memory from `b63c3759` to
`438b8783`, and the one model from `466f7c72` to `02b8ca6d`. The host `6bd416c1`, apu,
control and ppu are unchanged.

## 3. The PPU

### 3.1 How the hardware does it

| Hardware fact | Source |
| - | - |
| A line is 456 dots: OAM scan 80, drawing 172 to 289, HBlank the rest; 154 lines a frame | [P1] "Rendering" |
| OAM scan picks the first 10 objects whose Y covers the line | [P1] "OAM" |
| A fetcher walks the line one tile at a time (get tile, data low, data high, sleep, push; 2 dots a step), pushing 8 pixels into a BG FIFO; objects go through an OBJ FIFO | [P1] "Pixel FIFO" |
| A tile row is two bytes, low and high bitplanes; CGB attributes (palette, bank, flips, priority) sit in VRAM bank 1 | [P1] "Tile Data", "Tile Maps" |
| Palettes are lookups: BGP, OBP0, OBP1 map 2-bit colours to shades; CGB palette RAM holds 64 bytes BG + 64 OBJ of RGB555 | [P1] "Palettes" |
| No die-level source read for this study says the PPU is counters plus a fetcher state machine; the fetcher and FIFO are Pan Docs' behavioural model | [H9] (covers "All PPU logic" as schematics) |

### 3.2 How our circuit does it today (measured, `hw_shape --ppu`, `levels --table ppu`)

The circuit has 31,073 units, depth 14, 125 heads, and 1,802 live wires at the widest
boundary (9).

| Stage | Units | Largest families |
| - | - | - |
| Shared (once per line in the engine) | 22,411 | `bg` 4,074 and `wn` 4,074 (21 tiles each: map, attribute and data heads, picks, flips), `si` 3,726 (the 10-of-40 selection), `fc` 2,560 and `fo` 2,560 (each slot's row shifted into 16 fine positions), `so` 2,160, `od` 640, `sr` 640 |
| Positional (per pixel in the engine) | 8,662 | `pb` 1,680 and `pw` 1,680 (each pixel picks its tile out of 21 by column), `oc` 1,280, `ba` 1,024, `bc` 1,024, `out` 984 (palette), `op` 640, `entry` 296 |

The heads are `bg` 29, `wn` 29, `y` 20, `o` 20 (with `od`), `pal` 17 and `sx` 10. At
the widest boundary, 768 of the 1,802 live wires are `bg` and `wn` head bits and 1,025
are shared node wires, mostly the 21 tiles' picked rows and the slots' fine buffers. From
boundary 10 to 13 the palette lines take 360 to 984 wires.

In llama.cpp every one of the 170 rows computes all 31,073 units and 125 heads. The
engine computes the shared stage once and the positional stage per pixel. The circuit
computes the hardware's whole line at once, with all 21 fetches in parallel and then a
21-way select per pixel.

### 3.3 The reshaping

Each row is one moment of the hardware's line. Pixel x's row does what the fetcher and
the FIFO do when they reach x.

- **P1. Each pixel row fetches its own tile (the fetcher).** The row computes its map
  column `(SCX + x) >> 3`, which is affine in the position wires plus a 5-bit
  staircase. Its map and attribute heads read that column's byte, and its data head
  reads the tile row. It then picks its pixel's two bits by `(SCX + x) & 7` and the X
  flip. The window does the same with `x − (WX − 7)`. That is three heads per path where
  today there are 29.
  - *Units.* About 460 per path, against the 13,556 that `bg`, `wn`, `pb`, `pw`, `bc`
    and `ba` have together, net about −12,550 (estimated).
- **P2. Object pixels per row (the OBJ FIFO).** The OAM scan stays shared, as mode 2 is
  once per line (`si`, `so`, `sr` and the `y`, `sx`, `o`, `od` heads). For each of the 10
  slots, the row tests `0 ≤ x − X + 8 < 8` and picks its colour and opaque bits from the
  slot's row by that offset and the flip. That is about 100 units a slot, plus the
  priority chain.
  - *Units.* About −5,740 (estimated), from `fc`, `fo`, `oc` and `op`.
- **P3. The palette as one read per row (the palette RAM).** The pixel's (object,
  palette, colour) is already an address in `layout::PAL`. One head per row reads its
  line, and the pick takes the colour's byte, or BGP, OBP0 or OBP1's two bits on DMG.
  - *Units and heads.* About −1,110 units (estimated), and 16 fewer heads.
- **Per tile, not per pixel, in the engine.** Eight neighbouring rows issue the same tile
  query. Within one pass the engine can answer equal queries once, because the head's
  result is a function of its query and the cache, so equal queries give equal results.
  That is the hardware's one tile row per 8 pixels, 21 tile lookups per path per line. It
  needs heads in the positional stage, which the engine does not have (a new engine
  feature, with a guard that the query words are equal). It does not memoize across
  passes.

| | Today (measured) | P1 + P2 + P3 (estimated) |
| - | - | - |
| Units | 31,073 | about 12,000 |
| Heads | 125 | about 57 |
| Live wires, widest | 1,802 | about 900 (fewer head lines live at once; no 21-tile rows; palette 64 bits where it is 984) |
| Own checkpoint dense, 64-rounded | about 711 M | about 173 M (hidden 1,472) |
| llama.cpp ms per PPU pass, prefix overlay | 18.6 to 20.0 | 8.2 to 10.2 |

**Rows stay one byte each.** Every row emits one byte token, so 160 pixel rows remain.
DMG's 2-bit shades could pack four pixels per token (40 rows). Yellow runs in CGB mode,
so that does not move frame 300, and it changes the DMG output format.

### 3.4 What breaks

The Phase 1 gate (the PPU in lockstep per line, acid2 pixel-exact, four games' hashes)
holds as written. Position-dependent queries are exact in the dense oracle, since every
row computes its own attention. The engine needs positional heads and their equal-query
guard. The PPU thread does more lookups per line, about 7 per pixel, deduplicated to
about 50 distinct per line (estimated). It runs off the GBC CPU's thread today. Whether
it keeps 8 lines a wake is not measured. The PPU checkpoint's digest, the one model's
and the GGUF change.

## 4. The APU

### 4.1 How the hardware does it

The APU has four channels: two pulse channels with 8-step duty patterns, a wave channel
of 32 4-bit samples, and noise from a 15-bit LFSR with a 7-bit mode [P1 "Audio
Registers"]. The period dividers are counters, an up counter in Pan Docs and a down
counter in blargg's draft [P1], [H10]. A 512 Hz frame sequencer on DIV bit 4 (bit 5 at
double speed) clocks length at 256 Hz, sweep at 128 Hz and envelope at 64 Hz [P1 "Audio
details"]. Each channel has a DAC. NR51 pans and NR50 sets the volume, and the mix itself
is analog [P1].

### 4.2 How our circuit does it today

The circuit has 9,447 units and depth 13, with every level at 600 to 811 units
(measured, `levels --table apu`). The timers already advance by closed forms over a
segment's ticks, which is the counter shape. Per segment on text the APU does 2,816
updates (measured, this branch's run, 489,745 passes). Of the 2,131 updates in its
printed families, 1,286 are the mix and the output: `AL`, `AR`, `GL`, `GR`, `cap`,
`mix`, `FL`, `FR`, `SL` and `SR`. Another 786 are channel timers and phases (measured,
`bun tools/census/families.ts`).

### 4.3 The reshaping

- **A1. The mix as a sum (the analog mixer).** The hardware's mixer adds four DAC
  outputs. In bits that sum is an adder tree, the 936 `mix` units and 484 `AL`/`AR`.
  With byte-valued channels (d4) it is affine and free. Panning becomes
  `relu(a − 16·(1 − p))` and volume eight gated cases. The sample bytes still need bits
  back, through a staircase per bit or an LM head that scores a value channel directly,
  `2tv − t²`. That head is exact in f32, but its logit gap of 1 at about 65,000 fails
  f16, so it needs its own check. The change is about −1,000 units and −900 updates per
  segment (a guess until d4 is prototyped).
- **A2. The timers are already hardware-shaped.** They are counters plus closed forms
  over the tick count, so there is nothing to take.
- **A3. One pass per tick, rejected.** At least 22 M-cycles fit a segment, so per-tick
  passes multiply the APU's 243,952 passes to frame 300 about tenfold or more
  (estimated), and every pass pays llama.cpp's floor.

The APU takes 3.7 to 6.8 ms a pass in the prefix overlay, and it runs on its own thread
in the engine. A1 matters for llama.cpp hours and size, not for the GBC CPU's 60 fps. It
depends on d4's norm argument (census §5.6), so it comes after the host-fed d4 prototype.

**What breaks.** A1 breaks `docs/stock-llama.md` §2's rule that every channel lies in
[−1, 1], as d4 does, and moves the APU's digest, the one model's and the GGUF.
`ApuLockstep` still holds state, sample and read after every segment, and `dmg_sound`
01 to 04 stay the phase gate.

## 5. Control

**The hardware.** Pan Docs describes the timer and interrupts this way [P1 "Timer
Obscure Behaviour", "Interrupts"]:

- DIV is the visible part of a system counter that ticks every M-cycle. The source read
  does not state its width.
- TIMA counts falling edges of a TAC-selected counter bit ANDed with the enable. A DIV
  write can make an edge.
- An overflow reloads TMA and sets IF one M-cycle later.
- Dispatch takes 5 M-cycles.

**Today.** The circuit has 3,943 units and depth 11, a flat 384 units per level
(measured). On text it does 403 updates per pass over 111,889 passes, 3.5% of the GBC
CPU's (measured). The counter family `s` is the largest at 112, then `LY_t` 31 and `LY`
24.

**The reshaping.** The circuit already advances the counter by a span's M-cycles and
counts edges in closed form, the hardware's counter in one pass. The one shape lesson
left is the split adder of `pc+1` (split at bit 7). Add the span to the counter's low bits
and carry into the high bits, so the low bits' changes stop reaching every high-bit
unit. That is about −60 of `s`'s 112 updates (estimated). Control is 50.0 M dense and
about 41 M at 64-rounding (census). Beyond rounding, control does not justify a redesign.

**What breaks.** Nothing in the exactness argument breaks. The split adder moves
control's digest and the one model's. `ControlLockstep`, `instr_timing` and the
interrupt fuzz stay the gates.

## 6. The combined target architecture

The target is one model in the census's prefix overlay (a'), so any subset stays
selectable at about its own cost. Widths are rounded to 64. The GBC CPU (C1 to C4) and
the PPU (P1 to P3) are hardware-shaped, and the APU and control stay as compiled.

| | Today (measured) | (a') prefix overlay (census, estimated) | Hardware-shaped (a') (estimated) |
| - | - | - | - |
| Hidden (the widest prefix) | 5,120 | 2,368 (the PPU's) | 1,536 (the CPU's 509 shared + about 1,000 wires; the PPU about 1,472) |
| Layers | 15 | 15 | 15 (the PPU's; the CPU leaves after 11) |
| Heads per layer | 22 | 22 | about 12 (about 90 with weights over 15 layers) |
| Intermediate | 6,336 | 6,336 | about 3,520 (about 35,400 units over 15 layers, widest level) |
| Dense parameters | 2,057.9 M | 867 M | **about 338 M** |
| Nonzeros | 1,092,918 | about 1.15 M | **about 0.64 M** (CPU about 100 k, PPU about 300 k, APU 150 k, control 53 k, one gate term per unit) |
| CPU ms per pass in llama.cpp | 29.9 | 7.3 to 9.2 | 6.3 to 8.3 |
| PPU ms per pass in llama.cpp | 38.6 | 18.6 to 20.0 | 8.2 to 10.2 |
| llama.cpp hours to frame 300 | **11.0** | 2.8 to 3.6 | **2.3 to 3.1** |
| Engine µs per GBC CPU step, text | 4.8 | 4.8 | **about 3.9** (about 3.6 with d5) |
| Text fps (engine) | 38.7 | 38.7 | **about 48** (about 52 with d5) |
| A run with only the CPU in weights | hidden 5,120, 15 layers, 157 heads | hidden 1,792, 11 layers | hidden about 1,536, 11 layers, its own 17 heads: about 89 M of tensors read |

The hours come from `bun tools/census/project.ts`, which takes the variant's dims from
the rows above. Dense sizes use `Config::params`. Nonzeros scale each part's measured
nonzeros per unit by its projected units. None of this moves llama.cpp's floor. The GBC
CPU's 3.6 ms of floor and harness is 43% to 57% of its projected 6.3 to 8.3 ms, so below
about 2 hours only fewer passes help.

**The serialised alternative.** One pass per M-cycle copies the hardware's time structure
exactly, with one access, one address, and the IDU and ALU once each. Each pass is small
(estimated hidden 1,024, 6 layers, 10 heads, 4,000 units), but there are 2.6 passes per
instruction (8,351,304 M-cycles over 3,208,968 steps on text, measured). In llama.cpp it
projects to 4.9 to 6.1 ms a pass, 12.7 to 15.9 ms per instruction, and 4.2 to 5.4 hours
against 2.5 to 3.3 for the one-pass hardware-shaped GBC CPU (`project.ts`). In the
engine, if the fixed 1.53 µs per pass does not shrink with a shallower pass, the fixed
cost alone is about 1.53 µs × 2.6 ≈ 4.0 µs per instruction, above the 3.1 µs budget
before the engine walks any gate (estimated). It loses on both models. The opposite
direction, two instructions a pass, halves the floor's share at about double the depth
(census §5.6) and is not studied here.

## 7. Build order

Each unit ends in the CLAUDE.md gates: SST 500,000 on circuit and checkpoint, engine
equal to the dense pass, `run --frames 8 --core lockstep --hardware dmg` at `6872b003`
and `run --frames 300 --core lockstep` at `cfd31f3d`, then `tools/gate.sh` and its phase
gates. Each unit on the GBC CPU with memory also ends with `model-mem` lockstep, a
re-recorded `changes-memory.txt`, the f16 census, a re-exported and re-verified GGUF, a
text and overworld A/B (median of 3) with a row in `docs/perf-log.md`, and the new
digests recorded there and in CLAUDE.md. The GBC CPU goes first because it makes 72% of
the passes.

| # | Unit | Proves for the next | Digests that move |
| - | - | - | - |
| 0 | `examples/hw_shape.rs` (since removed) and `project.ts` rows | the counts every later unit is held to | none |
| 1 | C4 word-line selects in the GBC CPU with memory | that the census's update model predicts a pick change (about −200 updates) | cpu-memory, one model; the host `6bd416c1` unchanged |
| 2 | C1 prefetched IR, with forwarding and the fetch-only pass; `step.rs`, the reference, the SST harness and lockstep gain IR | measured −99 updates and 4.6% slower on text: rejected (§2.6) | cpu-memory, one model; the host unchanged unless the host-fed GBC CPU adopts IR too (not needed) |
| 3 | C2 bank keys and one bus address (d1), with the S = 512 key margin and the bit-concatenation slot rule | the banks once per pass built (−1,546 units, no speed change); one bus address measured +599 updates and rejected (§2.6) | cpu-memory, one model, the cache layout (every saved cache state) |
| 4 | C3 one IDU, only if a count from unit 3's circuit confirms the saving | | cpu-memory, one model |
| 5 | d5 byte cells for RAM (optional, about −227 updates) | how close circuit shape gets to 3.1 µs | cpu-memory, one model, the cache layout |
| 6 | 64-channel rounding in `Model::shape`, every checkpoint | built (§15.1): nothing needs whole blocks; the parts from 631.3 M to 501.9 M, the one model from 1,235.4 M to 1,138.9 M | every checkpoint |
| 7 | Positional heads with the equal-query guard in the engine; then P3, P1 and P2 in that order (the smallest first) | the PPU's width falling below the GBC CPU's | PPU, one model |
| 8 | The prefix overlay (a') | built (§15.2, §15.3): one model 716.5 M; each stream runs its own slice at its own engine cost, its dense slice above its own checkpoint by the shared vocabulary and channels | one model |
| 9 | d4 prototype on the host-fed GBC CPU; then A1 the APU mix | whether value channels hold the norm argument | host `6bd416c1` (if adopted), APU, one model |

## 8. Choices left to the project

- **IR as model-carried state.** C1 adds IR to the GBC CPU state that the host carries
  between passes, as it carries PC. The model fills IR itself for its own next pass. When
  a write that is not the GBC CPU's lands on PC's byte, the host marks IR invalid by
  address equality (no mapper rule) and the model runs a fetch-only pass. The question is
  whether that rule, and a GBC CPU state the model writes for itself, meet "the model
  must really run the CPU". The default here is yes, because every IR byte comes from a
  model fetch, and lockstep checks it.
- **DMG output packing.** Four 2-bit shades per token would cut DMG lines to 40 rows. It
  changes the DMG output format and does nothing for Yellow in CGB mode. The default here
  is to leave it out.

The project has decided the census's other questions: bank keys, tables written by the
embedding, the d4 prototype, and stream truncation. This study recommends against the
decode table on its measurement (§2.3), so the ruling on it is not needed.

## 9. What this study does not cover

- Outside §2.6 and §13 to §21, nothing here is built. The unit counts for C1 to C4 and P1
  to P3 in §2.3 and §3.3 are arithmetic on measured families, each with its assumption
  named. Live wires are the least certain. The PPU's 900 and the GBC CPU's 1,000 are
  guesses from which head lines and family wires leave.
- The llama.cpp projections inherit the census's limits. They are not measured, and they
  use a guessed least kernel time of 20 µs.
- How often Yellow would need a fetch-only pass under C1 (a fetch from a host-served
  region, or a copy onto PC's byte) is not measured. The `--record` traffic of a Yellow
  run can count it.
- Whether the PPU thread keeps pace with positional heads is not measured.
- Core ML and MLX are not projected.

## 10. Sources

Retrieved 2026-11-04. No query sent personal data.

- **[P1] Pan Docs**, https://gbdev.io/pandocs/. Pages: "Memory Map", "CGB Registers",
  "MBC3", "MBC5", "Rendering", "OAM", "Pixel FIFO", "Tile Data", "Tile Maps",
  "Palettes", "Audio", "Audio Registers", "Audio details", "Timer and Divider
  Registers", "Timer Obscure Behaviour", "Interrupts".
- **[H1] gekkio, "Game Boy: Complete Technical Reference"**, Revision 192 (16 August
  2026), https://gekkio.fi/files/gb-docs/gbctr.pdf. Used: ch. 2.2, ch. 3, ch. 4 (fig.
  4.3), §5.1, the `LD A,(nn)` timing, §13.2, §13.3, §17.1. It covers models before the
  CGB.
- **[H2]** The same document, §5.1 "Fetch/execute overlap" and fig. 5.4: "the CPU can only
  perform one memory access per M-cycle"; "the fetch stage of an instruction always
  overlaps with the last machine cycle of the execute stage of the previous
  instruction."
- **[H3]** [H1] §13.2: "the remaining ROM pins (A14-A20) are controlled by the MBC1."
- **[H5] Gekkio/gb-research**, `sm83-cpu-core/hdl` (`alu.vhd`, `alu_stage1.vhd`,
  `decoder.vhd`, `sm83_decoder_pkg.vhd`), https://github.com/Gekkio/gb-research. The
  repository describes its HDL as involving guesswork.
- **[H6] emu-russia/dmgcpu**, `wiki/sm83/Readme.md`,
  https://github.com/emu-russia/dmgcpu: "Decoders are made as NAND/NOR trees + domino
  logic"; the floorplan (ALU upper left; registers, SP, PC and the
  incrementer/decrementer at the bottom; the sequencer on the right).
- **[H7] msinger/dmg-schematics**, `sm83` sheets (alu, decoder1 to 3, idu, reg_wz,
  register_bank, sequencer), https://github.com/msinger/dmg-schematics.
- **[H8] Ken Shirriff**, "The Z-80 has a 4-bit ALU",
  https://www.righto.com/2013/09/the-z-80-has-4-bit-alu-heres-how-it.html. About the
  Z80 only.
- **[H9] furrtek/DMG-CPU-Inside**, https://github.com/furrtek/DMG-CPU-Inside. It gives
  the APU, PPU, timer, DMA and interrupt logic as schematics and leaves the GBC CPU core out
  as "a standalone core which doesn't use the standard cell topology".
- **[H10] blargg, "Gameboy sound hardware"** (draft), read through a copy at
  https://gist.github.com/drhelius/3652407 because the gbdev wiki refused connections.
  Secondary.

## 11. Reproducing

The three `hw_shape` commands are gone with the example, which has since been removed.

```sh
cargo run --release --example hw_shape -- --forms        # control forms (§2.2), seconds
cargo run --release --example hw_shape -- --fetch        # PC's readers, fetch and translation families, heads
cargo run --release --example hw_shape -- --ppu          # PPU shared vs positional units, heads, live wires per boundary
cargo run --release --example levels -- --table cpu      # also host, ppu, apu, ctl
GBCFORMER_UNITS=1 GBCFORMER_STATS=1 tools/bench/text.sh --weights cpu,memory,apu,control 2> text.err   # GBC CPU, APU, control updates per family
GBCFORMER_UNITS=1 GBCFORMER_STATS=1 CORE=model tools/bench/text.sh 2> host.err                         # the host-fed GBC CPU's 616 updates
bun tools/census/project.ts                              # §6's llama.cpp rows and dense sizes
bun tools/census/families.ts text.err                    # §4.2's APU split and control's total
```

## 12. The PUs, widened

This section widens §3 to §5 with one engine measurement the study did not have, adds the
other places where the hardware does something more cheaply, and ranks them on both cost
models. It is the plan for §13, which records what was built.

### 12.1 The measurement that reorders the PPU

Before this work, `GBCFORMER_STATS=1 tools/bench/overworld.sh --weights cpu,memory,ppu`
(measured, load 9 to 16) shows the PPU checkpoint re-evaluating 166,834 units per pass,
163,981 of them (98%) in the positional stage, at 34.9 µs a pass and 70.2 µs a CGB line
(two passes). The same workload with `--hardware dmg` (one pass a line) re-evaluates
15,901 units at 12.3 µs a line.

The cause is the palette lookup's output nodes (`out`, 984 units at each of the 160
positions). Every step of `out.j` reads FLG bit 2, the byte the pass asks for, so the
second CGB pass of a line re-evaluates all of them at every position. That is 160 × 984 =
157,440 units, about 96% of the PPU's engine work (inferred from the counts above; the
DMG run, which never flips the bit, does a tenth of the work). §3.3 valued P3 at −1,110
units for llama.cpp. On the engine it is the largest PPU lever by an order of magnitude.

The same measurement puts a price on §3.3's per-row reshapes. The engine runs a
positional unit once per position, so a unit moved from the shared stage into the rows
costs 160 times as much there, while llama.cpp pays every unit on every row either way.
P1 and P2 cut llama.cpp's dims and raise the engine's work, and the PPU thread is already
at 7.5 to 10 ms of a 16.7 ms frame (`docs/ppu-in-weights.md` §10.4's 52 to 72 µs a line,
144 lines).

### 12.2 What else the hardware does more cheaply

The hardware facts come from the Pan Docs [P1] pages "Pixel FIFO", "OAM" and "Audio
details", retrieved 2026-11-04.

| # | Hardware structure | Today | Reshaping | llama.cpp (dims) | Engine (updates) | Rank |
| - | - | - | - | - | - | - |
| U1 | Palette RAM is a lookup addressed by (object, palette, colour) | 17 constant palette heads, 64 entry one-hots, `out` 984 units per position | **P3.** One positional head per pixel reads its palette line, and `out` picks the colour's byte | −1,110 units, −16 heads, boundaries 10 to 13 −900 wires (estimated) | about −150,000 of 167,000 per CGB pass (estimated from §12.1) | 1, with U2 |
| U2 | The fetcher reads what the pixel needs; 8 pixels share one tile fetch | heads serve one address each; a positional query is refused | **Positional heads in the engine.** Each position's instance keeps its address bits, reads again only when they or the image move, and equal (layer, line) queries in a pass share one read | none | the enabler of U1, P1 | 1, with U1 |
| U3 | The OAM scan walks 40 entries keeping a count of objects found [P1 "OAM": first ten whose Y covers the line, in OAM order] | `si`: each slot's 6 index bits as a sum over 40 entries of `[chosen_i ∧ Σchosen[..i] = s]`, 3,726 units | one-hot "slot s holds entry i" (about 355 nodes of 2 steps), index bits as free sums of them | about −3,000 units; about +350 live wires between the scan and the sort (estimated, not checked against the widest boundary) | shared stage: small | 3 |
| U4 | The pixel's colour is one 15-bit value [P1 "Palettes"] | two passes per CGB line, one per byte | one pass with 320 output rows, both bytes | half the PPU passes (83,501 to about 41,750) at about twice the rows (guess) | removes the second pass; after U1 that pass costs about 15,000 units (estimated) | 4 |
| P1 | The fetcher, per row (§3.3) | 21 tiles fetched shared, then a 21-way select per pixel | each row fetches its own tile | about −12,550 units, 58 heads to 6 | adds per-position fetch, flip and pick units; the stage is 98% of the PPU's work | 5, measure before building |
| P2 | The OBJ FIFO, per row (§3.3) | each slot's row shifted into 16 fine positions, shared | each row tests and picks its slots' pixels | about −5,740 units | about 640 positional units per position re-evaluated per line, about +100,000 per pass (estimated) | not recommended for the engine on this estimate; built after P1 measured the stage, and it cost +15,076 (§13.3) |
| A4 | The high-pass filter is a capacitor on the mixer's analog outputs [P1 "Audio details"] | the capacitor is APU state, 24 bits, two levels | the APU emits the mixer's sum; the host applies the capacitor as presentation, as it maps colours to RGB | about −2 of 14 layers (estimated); −1 measured | about −758 of 2,639 updates per APU segment (§12 of `docs/apu-in-weights.md`: capacitor and sample, per side); −914 of 2,818 measured | built (§13.5) |
| A1 | The mixer adds four DAC outputs (§4.3) | adder trees over bits | byte-valued channels (d4) | about −1,000 units (guess) | about −900 per segment (guess) | after the d4 prototype |
| C1 | DIV's counter and TIMA's edge (§5) | a span's add over the whole counter | the split adder | none | about −60 of 403 updates per control pass; control is about 0.7% of the GBC CPU thread's updates per frame (§5's 111,889 passes) | not worth building |

The duty patterns, wave RAM, length, envelope and sweep are already counters and one-hot
selects (`docs/apu-in-weights.md` §12). The hardware's 10-object limit and its Y-only
scan are already the circuit's (`vis`, `ch`). Nothing else here is cheaper in the
hardware's form.

### 12.3 Order of the build

U2 and U1 come first, together, because the engine feature has no user without P3 and P3
cannot run in the engine without it. The other units follow in the table's order, each
only after a measurement of positional units per pass. P2 was left out on the estimate
above, then built when its measured cost came to a sixth of it (§13.3). Every unit moves
the PPU's digest and the one model's, as the GBC CPU side's units of §2.6 do, so a
merge of the two needs one re-export of the one-model GGUF and one
run of every gate after the merge.

### 12.4 Choices left to the project

- **A4, the high-pass filter as presentation.** The capacitor sits after the mixer, on
  the analog outputs [P1 "Audio details"]. The source does not say whether it is a board
  part or on the chip. Moving it to the host would make the host compute part of every
  sample the speakers play, labelled like the colour-to-RGB map, and `ApuLockstep` would
  compare the pre-filter sample. The default here was to leave it in the model. The project
  decided that the capacitor is the board's, after the chip's digital output, so it is
  presentation, and §13.5 built it.

## 13. The PUs: built

### 13.1 U1 and U2: the palette as one read per pixel

**What changed.** `ppu_circuit::circuit` replaces the 17 constant palette heads, the 64
entry one-hots and the 984-unit `out` with `shown` (the showing pixel's colour, its
palette line and, on DMG, which of BGP, OBP0 and OBP1 applies, 58 units) and one memory
head whose query is that line. `out` (84 units) then picks the colour's byte from the
line. The engine serves a head whose query reads a positional wire in the positional
stage (`Positional::attach`, `Positional::memory_reads`). One instance per position keeps
its address bits from the changes routed to its layer's head bucket and reads again only
when they move or the image's generation does. Equal (layer, line) queries in one pass of
a part share one read. A query bit that is not 0 or 1, or a line the cells do not give as
one row, refuses the pass, as for the shared heads. `MemPlan::check` still checks every
head against the record pattern before handing the positional ones to the stage.

| | Before (measured) | After (measured) | §12.2 estimate |
| - | - | - | - |
| Circuit units | 31,073 | 29,935 | −1,110 |
| Memory heads | 125 | 109 (108 shared, 1 positional) | −16 |
| Live wires, boundaries 10 to 13 | 1,613, 1,196, 1,009, 1,059 | 1,250, 512, 25, 80 | −900 |
| Widest boundary (9) | 1,802 | 1,802 | unchanged |
| PPU units per pass, overworld CGB | 166,834 | 20,755 | about −150,000 |
| Positional units per pass | 163,981 | 17,902 | |
| PPU µs per line, overworld | 70.2 (load 9 to 16) | 28.6 (load 3) | |
| Head reads per PPU pass | 4.4 | 4.5 | |
| PPU checkpoint, `cpu,memory,ppu` run line | hidden 3,072, intermediate 5,504, 922.2 M, 785,126 nonzero | hidden 3,072, intermediate 5,248, 886.9 M, 771,452 nonzero | |

The two µs figures come from different loads. The units per pass do not depend on load.

**Speed (measured).** A was the study's build before this unit and B this unit, interleaved over
three rounds with `--weights all --one-model` and `GBCFORMER_STATS=1`, at load 3.5 to
8.5. Text ran at 37.5, 37.9, 37.6 against 39.0, 39.1, 37.9 fps (median 37.6 to 39.0),
and the PPU line went from 72.1 to 25.7 µs. Overworld ran at 74.7, 74.5, 72.4 against
99.9, 99.6, 99.9 fps (median 74.5 to 99.6), and the PPU line went from 71.5 to 30.3 µs.
The GBC CPU's thread bounds text, so text moves little. The PPU bounds overworld. The
machine's thread waits for the PPU worker when it reads the frame and at each palette
read, so a PPU line 2.4 times cheaper gives a third more frames (inferred from the µs per
line and the waits, not profiled).

**Gates (measured).** `tools/gate.sh` passes every hash, acid2 image, blargg ROM and
in-weights phase gate, the one model's Yellow 300 included. Its first `cargo test` failed
only on `tests/one_model.rs`'s count of the PPU stream's heads (125), which now counts the
positional head (109, `Engine::plan_size`), and the rerun of the library and the
`one_model`, `stream_audit`, `memory_dense` and `e01_primitives` tests passes.
`tests/ppu_lines.rs` and `tests/ppu_checkpoint.rs` pass.

**Digests.** The PPU moved from `6ee4beb0` to `f5a04864` (886.9 M dense, 0.84 M
nonzeros) and the one model from `b03d896c` to `7aee0707` (2,043.2 M, 1.47 M). The host
`6bd416c1`, cpu-memory `ffe22414`, APU `ebeeb39c` and control `254acbba` are unchanged.
The re-exported one-model GGUF (`export one-model --layer-shapes --align 64`, 1,526.4 M
values) loads with 0 of 2,042,992,640 values differing (`digest --gguf`).

### 13.2 P1: each pixel fetches its own tile

**What changed.** `B::fetch` replaces the 21-tile fetch and the per-pixel 21-way select.
Each position computes its map x (`At::map_x`, which is SCX + x for the background and
x + ¬WX + 8 for the window, three steps a bit). It reads its map and attribute line and
then its tile's row through three positional heads per path, and picks its two bits by
the flipped row, the pixel and the X flip. Each step also requires `[bgon, ¬winx]` or
`[bgon, winx]`, so one node per colour bit sums both paths. Two engine changes came with
it. Each layer's positional heads keep their own address-bit terms (`PartHeads::terms`
per layer), because two heads on different layers read the same row bits and one shared
table applied each change twice. Each part of the positional stage numbers its blocks
from the shared channels on (`Positional::first`), so 160 positions of about 550 channels
each fit the u16 tables, and the part count rises to whatever keeps a part within them.

| | U1 (measured) | P1 (measured) | §3.3 estimate |
| - | - | - | - |
| Circuit units | 29,935 | 16,898 | about −12,550 |
| Memory heads | 109 | 57 | 58 to 6 for the two paths |
| Widest live boundary | 1,802 (9) | 1,074 (8, the objects' row lines) | about 900 |
| Widest level | 4,602 | 3,034 | |
| PPU checkpoint | hidden 3,072, intermediate 5,248, 886.9 M, 771,452 nonzero | hidden 2,048, intermediate 3,072, 390.7 M, 624,664 nonzero | about 173 M with P2 and P3 |
| PPU units per pass, overworld | 20,755 | 35,233 | |
| PPU µs per line, overworld (`cpu,memory,ppu`) | 28.6 | 33.2 | |

**Speed (measured).** A was unit 1 and B P1, interleaved over three rounds with
`--weights all --one-model`, at load 6 to 11. Text ran at 38.5, 37.7, 38.0 against 36.6,
36.2, 37.3 fps (median 38.0 to 36.6, −3.7%), and the PPU line went from 26.6 to 39.6 µs.
Overworld ran at 97.5, 95.6, 96.8 against 93.5, 95.2, 94.3 (96.8 to 94.3, −2.6%), and the
PPU line went from 32.5 to 36.5 µs. The per-position fetch gives the PPU's thread and its
three helpers about half again as much work, and at this load the GBC CPU's thread shares
the cores with them (inferred; the GBC CPU thread's own work is unchanged). P1 is a size
unit. The one model falls from 2,043.2 M to 1,327.8 M dense parameters.

**Gates (measured).** `tools/gate.sh` passes every check, `cargo test` included (the
PPU stream's head count is now 57).

**Digests.** The PPU moved from `f5a04864` to `768a060c` and the one model from
`7aee0707` to `a4521464`. The rest are unchanged. The re-exported GGUF (986.4 M values)
loads with 0 values differing.

### 13.3 P2, U3 and the fetcher's two bytes

**P2, the OBJ FIFO per pixel.** Each position tests each slot's column v = x + 8 − X
(`hit`, one node per v over X's coarse bits and its `X & 7` one-hot) and picks the
slot's colour and opaque bits from its row by v and the X flip. The shared fine buffers
(`fc`, `fo`, 5,120 units) and the per-pixel placement (`oc`, `op`, 1,920) become 960
positional units. §12.2 estimated about +100,000 engine updates a pass for it. It
measured +15,076 (35,233 to 50,309), because a slot's row changes in few bits a line.

**U3, the OAM scan's count.** `hold{s}.{i}` holds when entry i is chosen with s chosen
before it. Each slot's OAM index bits are free sums of those 355 nodes, where `si` was a
node per bit over 40 entries. `si`'s 3,726 units become 1,380, and `so` and `bf` grow by
375 because their literals are now sums.

**The fetcher's two bytes.** The tile row's low and high bytes are picked first (16
`row_byte` nodes a path) and the pixel's bit from them, so a new row moves those nodes
instead of all 64 steps per plane at every position. The map and attribute heads sit on
layer 2 and the data head as early as its query allows. The background's path is then
one level shorter, and the circuit's depth falls from 14 to 13.

| | P1 (measured) | P2 + U3 (measured) | + two bytes (measured) |
| - | - | - | - |
| Circuit units | 16,898 | 8,847 | 8,783 |
| Depth (layers) | 14 (15) | 14 (15) | 13 (14) |
| Widest level | 3,034 | 2,400 | 2,400 |
| Widest live boundary | 1,074 | 1,074 | 978 |
| PPU checkpoint dense, nonzeros | 390.7 M, 624,664 | 367.1 M, 320,725 | 338.8 M, 318,898 |
| PPU units per pass, overworld | 35,233 | 50,309 | 42,687 |

**Equal fetch work answered once.** The reads are already answered once. Equal (layer,
line) queries in a part's pass share one read, and the PPU's 160 positions × 7 positional
heads read 3.0 lines a pass on overworld. What remains per position is gate updates of
units whose inputs differ by position (the pixel's column and fine bits) or that sit
downstream of them. Sharing a unit between the 8 pixels of a tile would need the engine
to prove, per pass, that their inputs are equal. The tile boundary moves with SCX & 7, so
the grouping changes on every scroll, and checking a unit's inputs at each position costs
what the update does. The cheap part of the idea is the hardware's own order, the
fetcher's two bytes above, which removed 15% of the stage's updates.

**Gates (measured).** `tools/gate.sh` on this work passes every check: build without
warnings, wasm32, `cargo test` (the PPU tests included), the lockstep hashes, the other
games, both benchmarks, blargg, both acid2 images with the PPU in weights, the speed
switch, `instr_timing` with `cpu,control`, `dmg_sound` 01 to 04 with `apu,control`, the
four games' 300 frames with all five in weights in lockstep, and Yellow's 300 as one
model. P2 and U3 passed `tests/ppu_lines.rs`, `tests/ppu_checkpoint.rs` and
the overworld hash on their own. The full gate ran once on the tree with the two bytes.

**Digests.** The PPU moved from `768a060c` to `1f904a28` (338.8 M, 0.38 M nonzeros) and
the one model from `a4521464` to `a5fdac3a` (1,259.1 M, 1.02 M). The rest are unchanged.

**Speed (measured).** Both A/Bs ran interleaved, three rounds each, with
`--weights all --one-model`, at load 3 to 12. P1 against the final tree gave text 36.8, 36.9, 37.8
against 38.0, 38.2, 37.8 fps (median 36.9 to 38.0), with the PPU line from 38.5 to
34.0 µs, and overworld 94.4, 93.0, 97.1 against 95.5, 93.6, 96.8 (94.4 to 95.5).
The tree before this work against the final tree gave text 38.5, 37.4, 39.5 against 37.6, 36.6, 38.0
(median 38.5 to 37.6, −2.3%), with the PPU line from 69.3 to 31.3 µs, and overworld 75.4,
76.0, 75.5 against 97.2, 96.7, 96.9 (75.5 to 96.9, +28%), with the PPU line from 70.3 to
36.4 µs. The GGUF (886.4 M values) loads with 0 values differing.

### 13.4 Where this work ends

| | Before (measured) | After (measured) |
| - | - | - |
| PPU circuit units, depth, heads | 31,073, 14, 125 | 8,783, 13, 57 |
| PPU widest live boundary | 1,802 | 978 |
| PPU checkpoint | hidden 3,072, intermediate 5,504, 15 layers, 922.2 M, 785,126 nonzero | hidden 2,048, intermediate 2,752, 14 layers, 338.8 M, 318,898 nonzero |
| One model | 2,057.9 M (`b03d896c`; 2,043.2 M after U1) | 1,259.1 M (`a5fdac3a`); 1,235.4 M (`2d6085d7`) after A4 (§13.5) |
| PPU units per pass, overworld | 166,834 | 42,687 |
| PPU line, one model, overworld | 70.3 µs | 36.4 µs |
| Text, overworld fps (one model) | 38.5, 75.5 | 37.6, 96.9 |

**What A1 needs.** A1 (the APU mixer with byte-valued channels) is not built. It rests on
d4, which had no prototype when this work ended (§15.4 records d4 on the GBC CPU
since). A prototype on the host-fed GBC CPU must first show four things: a value-node
kind in the circuit IR (`Σ coef·[form ≥ θ]` today); the norm argument with channels up to
255 (`docs/stock-llama.md` §2's eps-dominated RMSNorm assumes |x| ≤ 1, so the residual
unit falls to about 2^-13 and the scale moves into the next layer's weights, checked in
f16); value muxes `relu(v − 256·(1 − s))` exact at K = 128 within T2's partial sums; and a
readout of a value channel into a byte token (a staircase per bit, or an LM head scoring
`2tv − t²`, whose logit gap of 1 near 65,000 fails f16 and needs its own check). The
engine's integer mode already holds integers, so its guards would need ranges up to 255,
not new arithmetic. Only then can the APU's `mix`, `AL`/`AR` and filter sums become
affine.

### 13.5 A4: the high-pass filter as presentation

**The ruling.** On the hardware the high-pass filter is the board's output capacitor
after the APU chip's output, so it is presentation, as mapping the PPU's colour numbers
to the screen's RGB is. The model still computes every value the chip produces: each
side's DAC sum over the channels NR51 routes there, times NR50's volume.

**What changed.** `machine::apu` ends at that digital value (`Mix`, −480 to 480 a
side). `HighPass` multiplies it by 32 and runs the old recurrence (`filtered = in −
(cap >> 8)`, `cap += filtered`, the sample is `filtered` clamped to i16).
`Machine::take_samples` applies it once, to whatever device holds the APU slot, so the
host, the inline checkpoint and the worker share one implementation. The checkpoint
loses both 24-bit capacitors and the filter's sums (`F`, `G`, `cap`). Its `SL0 SL1 SR0
SR1` carry the mix, as 10 bits of two's complement with bit 9 repeated into the i16's top
six. `ApuLockstep` compares state, mix and read. The run's label lists the filter under
presentation. `changes.txt` was re-recorded on the new circuit.

**Audio identity (measured).** `run --wav` is byte-identical before and after for
Yellow frames 0 to 600 with the APU on the host, in weights, and in weights with
`--lockstep apu` (1,928,800 bytes, 488,096 passes equal), and for `text.sh` with the
APU on the host and in weights after the warmup's hand-off (11,251,140 bytes). Within
each workload every file equals every other: nine Yellow WAVs and six text.sh WAVs
from the old binary and two builds of the new one. One difference is possible, and no
run reached it. Mixes that `SAMPLE_CAPACITY` drops before a drain no longer move the
capacitor, as they did when it was APU state. One drain a frame (800 of 4,096) drops
none.

| | Before A4 (measured) | A4 (measured) |
| - | - | - |
| APU circuit wires, nodes, depth | 2,319, 1,839, 13 | 2,081, 1,649, 12 |
| APU widest live boundary | 771 | 697 |
| APU checkpoint | hidden 1,024, intermediate 960, 14 layers, vocab 15,736, 90.1 M, 150,077 nonzero | hidden 1,024, intermediate 896, 13 layers, vocab 14,188, 80.2 M, 135,817 nonzero |
| APU slots in, out | 60, 60 | 54, 54 |
| APU updates per segment, text, overworld (`GBCFORMER_UNITS=1`, `--weights apu`) | 2,818, 3,026 | 1,904, 2,043 |
| One model | 1,259.1 M (`a5fdac3a`) | 1,235.4 M (`2d6085d7`) |
| Text, overworld fps (one model) | 38.4, 98.0 | 39.9, 100.0 |

The A/B ran `--weights all --one-model`, interleaved over three rounds, at load 5.0 to
5.8. Text ran at 38.2, 38.4, 38.6 against 39.6, 39.9, 40.0 fps, and overworld at 97.8,
98.0, 98.2 against 99.0, 100.0, 100.1. B was faster in every pair. Why the GBC CPU's
thread gains 2 to 4% from a cheaper APU stream was not profiled. The likely cause is
shorter waits at FF10-FF3F reads and at the end of a frame's audio, and less contention
for cores (inferred).

The estimate was low because the filter's input-only shares `G` (162 updates a side)
and `F` went too, not only the capacitor and the sample. Depth fell one level, not two,
because the timers' last blocks and `RV` stay at the tail. The re-exported one-model GGUF
(`--layer-shapes --align 64`, 859.4 M values) loads with 0 of 1,235,312,640 values
differing (`digest --gguf`). The APU digest moved from `ebeeb39c` to `110ecc5d` and the
one model's from `a5fdac3a` to `2d6085d7`. The host `6bd416c1`, cpu-memory `ffe22414`,
control `254acbba` and PPU `1f904a28` are unchanged. `tools/gate.sh` passes in full.

## 14. Merged

The PU work (§13) and the GBC CPU work (§2.6, with C4 word lines, the banks once per pass,
and DMA records placed from the banks the model emits) were merged. The two change
disjoint circuits, so every per-part checkpoint keeps its parent's
digest, and only the one model moves. The measurements, from `docs/perf-log.md`
"Shape", are:

| | GBC CPU work (§2.6) | PU work (§13) | Merged |
| - | - | - | - |
| cpu-memory | `438b8783`, 162.3 M | `ffe22414` | `438b8783`, 162.3 M, 0.23 M nonzeros |
| APU | `ebeeb39c`, 90.1 M | `110ecc5d`, 80.2 M | `110ecc5d`, 80.2 M, 0.29 M nonzeros |
| PPU | `6ee4beb0`, 922.2 M | `1f904a28`, 338.8 M | `1f904a28`, 338.8 M, 0.38 M nonzeros |
| One model | `02b8ca6d`, 2,043.2 M, 1.47 M nonzeros | `2d6085d7`, 1,235.4 M, 0.99 M | `52f41ae4`, 1,235.4 M, 0.97 M |
| Units per fast pass, overworld: cpu, apu, ppu | 1,335, 3,025, 166,834 | 1,925, 2,044, 42,687 | 1,335, 2,044, 42,687 |
| Text, overworld fps (one model, GGUF) | 42.5, 76.5 | 40.0, 100.4 | 42.6, 106.6 |

The host `6bd416c1` and control `254acbba` are unchanged throughout. The one model's
dense shape equals the PU work's (hidden 4,096, intermediate 4,544, 14 layers), likely
because the PPU sets its widest levels (inferred). The cuts to the GBC CPU with memory
lower the nonzeros (0.99 M to 0.97 M), not the shape. The GGUF
(`--layer-shapes --align 64`, 843.0 M values) loads with 0 values differing.

## 15. The one model's shared dimensions

The one model's hidden size was the sum of the parts' live wires, rounded to blocks of
1,024, so every stream paid every part's width. Three units shrink it, each with the
CLAUDE.md gates, the f16 census, a re-exported GGUF and an interleaved A/B
(`docs/perf-log.md`, "Overlay unit").

### 15.1 Hidden in steps of 64

This is row 6 of §7. Every checkpoint's hidden is its used channels rounded up to 64,
since nothing needed whole blocks. The parts shrink by 10% to 31% and the one model from
1,235.4 M to 1,138.9 M dense (hidden 4,096 to 3,776). The new digests are host
`3c11d62b`, cpu-memory `1645a346`, apu `83c6b259`, control `f2676ee8`, ppu `a1ee149e` and
one model `c21b3cac`.

### 15.2 The overlay

This builds `docs/one-model.md` §3's design, with two changes found on the way. Parts
that never share a position share residual channels. Channel k of every part's own
channel plan (`Circuit::overlay_channels`, each part planned as `channels` plans it alone)
is one residual channel, so hidden is the shared channels plus the widest part's wires,
not their sum. Each part keeps the levels its own build gave it (no merged reschedule),
so its units sit where its own checkpoint places them. (Re-levelling later changed this
for the PPU's three parts, which now fill the layers the other parts leave:
`docs/perf-log.md`, "Re-levelling".)

- **The mode is per position, not per stream.** Each output slot's mask token carries
  its part's mode channel (four channels after the lanes). Every unit of a part reads
  that channel as its `up` input instead of the constant. SwiGLU's own gate,
  `silu(gate)·up`, is then exactly the unit at the part's positions (`up` = 1) and
  exactly 0 everywhere else (`up` = 0, the gate finite in f32). This replaces the
  `−BIG·(1 − mode)` gate term, which f16 could not hold (a gate's range must stay
  within 511 steps of K = 128). No mode token and no layer-0 gather are needed.
- **Routing units are mode-gated too.** The design took them as safe because they key
  on their slot's identity, `[field + slot ≥ 2]`. That fails when a field form carries
  a folded constant (`x + 1` for a wire in {−1, 0}). Over another part's wire in [0, 1]
  the form reaches 2 at slot 0. The engine's positional guard refused the PPU stream
  on exactly this, a GBC CPU output's routing unit over a PPU wire.
- **Memory heads attend to a null token outside their part's positions.** Each query
  gains `N·(1 − mode)` on null dims (RoPE pairs 22 to 30; both halves of pairs 8 to 11
  since §21), whose key is 1 only on the null token (value 0, one extra
  cache cell; `tok::null`). `N` must beat the largest logit the head's query can give
  any key over foreign channel values in [−1, 1] (`foreign_score`) by `EXP_ZERO`. That
  is 190,464 before the 1/8 scale, split over six dims of 31,744 (measured). The copy
  heads stay ungated, because reader records carry no part's mode, and only the record
  part's units and the reader tokens read the copy channels.
- **Layer-0 gathers stay as they are.** A stream holds every other part's slots at
  byte 0 or the mask, whose values are 0 (`Model::stream` asserts it for every slot),
  so another part's gather adds 0 to the channels it shares.

**The proof, per stream (`Model::stream`).** At the stream's own output positions every
other part's unit has `up` = 0, every other part's memory head attends only to the null
token, and every other part's gather reads zeros, so every shared channel holds the
stream's own value. The engine plans each stream on that network (its units' `up`
folded onto the constant, its heads' null term dropped, the layout's occupants its own)
and keeps the whole checkpoint as the dense pass. The stream's network also clears the
mode and null channels from its embedding, which none of its rows read once folded. The
engine's owner tables do not track a mask token that carries two channels, and the first
build, which kept them, decoded 72 positions a GBC CPU pass instead of 14.5 and ran text
at 33.1 fps against 42.1 (measured, `GBCFORMER_STATS=1`). The norm divides exactly as
before. Every channel still holds a value in [−1, 1] at every position, and `mean(x²)` is
`eps`-dominated at any width (`docs/stock-llama.md` §2). Each stream's plan checks the
bound over its own occupants (6,207 against 33,792 at hidden 2,112).
`tests/one_model.rs` holds the APU and control streams' fast passes to the whole
checkpoint's dense pass (3 random passes each), the APU stream to the APU's own
checkpoint (500 passes, and 2,000 in `examples/one_model --build`, since removed), and `tools/gate.sh`
runs Yellow 300 with every component in weights, in lockstep, as one model.

**What stock runtimes must know.** A runtime that runs every unit and head at every
position (no per-part slices) in f16 accumulation is not covered. Another part's gate,
summed over this part's wires, can pass 65,504 (159 rows would, by the f16-accumulation
census over every part's ranges), `inf·0` is NaN, and the null logit itself is 190,464
before scaling. With f32 sums both are exact. With per-part slices (unit 3,
`Model::stream`) neither arises, because a slice holds no other part's rows. The f16
census now checks each row over its own part's ranges, where its output counts.
llama.cpp's default Mac CPU build sums f16 products in f16 and is not covered: its whole
model gives NaN at the first GBC CPU pass (`docs/perf-log.md`, "The Mac CPU backend's
NaN", which counts 2,027 such gate rows in `7b064d51`).
llama.cpp on Metal runs every row at every position with f32 sums, which covers both,
but its flash attention holds the query in f16 and its cache holds only the cells it
decodes. A later change makes the null token decode the null cell and scales the
GGUF's queries and keys so every query entry stays finite (`docs/stock-llama.md` §4).

| | Unit 1 | Overlay |
| - | - | - |
| Hidden | 3,776 | **2,112** (789 shared, 1,292 GBC CPU wires, 4 mode, 1 null, rounded to 64) |
| Intermediate | 4,544 | 5,440 (no merged reschedule) |
| Layers, heads per layer | 14, 22 | 14, 22 |
| Vocabulary | 34,763 | 34,764 (the null token) |
| Dense | 1,138.9 M | **716.5 M** |
| Nonzero | 0.97 M | 0.97 M |
| GGUF values (`--layer-shapes --align 64`) | 780.0 M | 432.2 M |
| Digest | `c21b3cac` | `e4615c82` |

Every per-part checkpoint keeps unit 1's digest. The units in each stream's plan, against
the part's own checkpoint, are PPU 4,119 (own 4,783), GBC CPU 8,994 (9,023), APU 7,348
(7,348) and control 3,291 (3,296). The PPU gap is not traced. The reader tokens' 672
routing units, which the PPU stream's reach need not include, would account for most of
it (inferred).

### 15.3 The prefix overlay

The prefix overlay lays out the overlay so that each part needs only a prefix of the
checkpoint, and every stream, the dense oracle included, runs only that prefix.

- **Channels.** An overlay puts the wires last (`Layout::wire_base`): constants, slot
  identities, the own byte, the memory block, routes, lanes, the mode and null channels,
  then the wires. Every part's wires start at the first wire channel, so part p uses
  `Layout::prefix(p)` channels and every channel above holds only other parts' wires.
- **Layers.** Each part routes its outputs at its own depth (`Circuit::out_depth`; the
  GBC CPU at layer 10, control at 11, the APU at 12, the PPU at 13), the reader tokens at
  the record part's (the GBC CPU's), and the copy heads sit below it. A part has no unit
  or head past its routing layer.
- **Units and heads.** Each layer holds a part's nodes and clearing units as one
  contiguous range, part by part. Heads are a range per part, with the record gathers
  and copy heads shared.
- **`Model::stream`.** It cuts a part's network out: its prefix of channels (rounded to
  64), its layers, its own units and heads and the shared ones, with the overlay's folds.
  It asserts what makes the cut exact: no kept row reads a channel above the prefix and
  no kept writer writes one, no layer past the last kept holds the part's work, and every
  held slot's layer-0 value is 0. At the part's positions every left-out unit adds 0
  (`up` = 0), every left-out head adds the null token's 0, every channel above the
  prefix holds 0, and the layers past the routing layer add nothing, so the cut network
  computes the checkpoint's residual on every kept channel. The norm divides by
  `sqrt(eps)` at either width (`docs/stock-llama.md` §2).
- **The engine.** It runs each stream on its own network. That network is the stream's
  dense pass and the oracle its fallback checks against, and `Engine::streams` checks the
  stream's plan on it. `one_model::streams` builds only the parts a run selects, so
  `--weights cpu,memory --one-model` builds the GBC CPU stream alone.
- **The GGUF.** `gguf::load_slice` reads a stream's network straight from the file (the
  prefix columns of each row it keeps, its layers, its heads and units) and compares
  every value with the compiler's (0 differ). `export --layer-shapes` writes each part's
  slice as metadata (`gbcformer.stream.<part>.embedding_length`, `block_count`, `heads`,
  `ffn`), which is what a runtime needs to run one part from the one file.

**Checked.** `every_slice_is_the_checkpoint_at_its_positions` (`tests/one_model.rs`,
ignored, 273 s) runs the APU's and control's slices against the whole checkpoint's dense
pass. The residuals are bit-identical on every slice channel (except the mode and null
channels, which the slice clears from its embedding and none of its rows read) and 0 on
every channel above, the tokens are equal, and the streams' fast passes decode them. The
lockstep gates hold the GBC CPU and PPU streams to the reference.

| Stream | Its network: hidden, intermediate, layers, heads | Dense | Nonzero | Own checkpoint |
| - | - | - | - | - |
| GBC CPU | 2,112, 1,600, 11, 6 | 279.2 M | 171,211 | 1,728, 1,600, 11, 6: 137.0 M |
| PPU | 1,792, 2,752, 14, 20 | 399.2 M | 348,250 | 1,536, 2,752, 14, 20: 254.1 M |
| APU | 1,536, 1,472, 13, 11 | 225.7 M | 208,992 | 896, 896, 13, 8: 70.2 M |
| control | 1,472, 832, 12, 8 | 166.8 M | 129,850 | 832, 512, 12, 5: 40.6 M |

The one model keeps its size (716.5 M, hidden 2,112, digest `6a17e689`), because the
prefix only reorders and cuts nothing. A stream pays its own layers, units and heads, but
two shared costs remain above its own checkpoint (estimated from the shapes). One is the
vocabulary, 34,764 tokens for every part. Embedding and LM head are 146.8 M of the GBC
CPU stream's 279.2 M, which is file size only for a stock runtime, since it gathers
embedding rows. The other is the shared channels, which hold every part's slot
identities and the memory block (789 of them). The GBC CPU's prefix is the whole hidden
because the GBC CPU has the most live wires (1,292). Shared slots (`docs/one-model.md`
§3) would cut both, and per-part embedding and LM-head ranges would cut the vocabulary.

**For llama.cpp, later.** A stream runs from the GGUF's `gbcformer.stream.<part>.*` keys:
views of each tensor's first `embedding_length` columns (or copies), the listed query
heads of `attn_q` and `attn_output`, the listed runs of `ffn_gate`, `ffn_up` and
`ffn_down`, `block_count` layers, then the output norm and LM head on the prefix. The
patch is `layer-shapes.patch` plus per-ubatch slices, in the family of
`ffn-slices.patch`. Two conditions hold only on slices. Without one, another part's gate
over this part's wires can pass f16's range in f16 sums (`inf·0` is NaN), and the null
logit (190,464 before the 1/8 scale) passes it too. A slice holds neither row. A stream
run without its slice must use f32 sums and keep the null token in its cache.

### 15.4 The GBC CPU on nibble channels

The GBC CPU circuits carry bytes as two nibble channels (0..15) by default
(`docs/byte-channels.md`), and the other parts stay 0/1. On the overlay that changes
three things, each measured:

- **Hidden.** The GBC CPU has 1,050 live wires against 1,292, so the one model's hidden
  is 789 shared + 1,050 + 4 mode + 1 null, rounded to 1,856 (from 2,112), and the one
  model is 634.6 M dense (from 716.5 M), digest `e872e6c9`. The GBC CPU stream's slice is
  1,856 wide, 233.6 M dense. The norm's bound holds at that width (the GBC CPU stream's
  29,695 against 16·1,856 = 29,696). A narrower GBC CPU would have its hidden set by the
  bound (`llama::norm_squares`), not by its channels.
- **The null term.** At a GBC CPU position, another part's memory head reads a channel
  that a GBC CPU nibble wire occupies, so `foreign_score` reads each channel's
  occupants' largest magnitude. The null term is 9 dims of 30,720 against a null key of
  4, which is 1,105,920 before the 1/8 scale (from 190,464). As before, only a runtime
  without per-part slices meets it, and that runtime needs f32 sums.
- **The f16 census.** It holds a memory head's query rows over its own part's ranges, as
  it holds a unit's gate. Over another part's nibble channels a PPU head's query rows
  reach partial sums of 155,648, which, like the gate case above, arise only without
  slices.

With every component in weights as one model (`docs/perf-log.md`, "d4 the default"),
text went from 43.6 to 47.9 fps and the GBC CPU pass from 4.27 to 3.89 µs, and overworld
went from 110.3 to 122.5 fps.

## 16. Ports at rest and one record a line

### 16.1 How the hardware does it

The GBC CPU has one 16-bit address bus and one 8-bit data bus, and it drives them only for
an access ([H1] ch. 3, [H2] §5.1). A memory chip that is not selected keeps its state,
and nothing toggles its lines. A push, a call or an interrupt dispatch writes SP−1, then
SP−2, two adjacent bytes that share an 8-byte line seven times in eight ([H1] ch. 4).

### 16.2 How our circuit did it

Each memory port fell back to a fixed address on a step that did not use it: the main
read head to VOID, and both write heads to physical address 0, ROM's first line. An
access then cost two changes at every line bit that differs, one on and one off. Text's
133.4 wire changes a pass included 41.9 at the data and write ports (a census by
changed wire). A step that wrote two bytes of one line kept two records, the
second of which already holds the first's byte.

### 16.3 The replay

A reference-core replay of text and overworld reproduces the recorded table
(`src/sm83/changes-memory.txt`) to 0.01 (text `D0h` 5.94, `WL0h` 8.05, `WL1h` 3.30;
overworld 3.47, 4.83, 1.81). The replay is a scratch tool, not committed. It runs the
reference GBC CPU over the model's physical image and counts, for each head's line per
step under each idle policy, the Hamming flips of its 64 line bits. The line-bit flips
per pass are:

| Head, idle policy | Text | Overworld |
| - | - | - |
| `D0h` VOID (built) / HL / SP / last line held | 5.94 / 12.42 / 8.72 / 3.18 | 3.47 / 4.85 / 8.13 / 3.55 |
| `WL0h` ROM line 0 (before) / VOID / SP−1 / HL / held | 8.05 / **6.71** / 6.15 / 9.58 / 2.98 | 4.83 / **3.94** / 6.44 / 4.09 / 1.81 |
| `WL1h` ROM line 0 (before) / SP−2 / held | 3.30 / **1.16** / 0.77 | 1.81 / **0.61** / 0.47 |

HL as a default loses on the read port, as the "pair holding HL" revert did (perf-log,
fan-out round), because HL moves between accesses. Holding the last line wins on every
head, but a held address is state, and it has to reach the model as input.

### 16.4 Held addresses, built and rejected

This build carried RP0 and WP0 back as six input slots (`LR0a..c`, `LW0a..c`), with the
main read head and write 0 holding the last line. The heads' changes fell as replayed
(wire changes from 106.2 to 98.3 a pass on text, the shared layers −0.09 µs). But each
held slot that moves is a moved input position, which costs about 70 ns in layer 0, and
they moved 0.7 times a pass. Text was flat and overworld 1.2% slower, with two more
layer-0 heads, the GBC CPU checkpoint from 135.3 to 146.3 M dense, and the one model's
hidden from 1,856 to 1,920.

### 16.5 Kept: stateless rest positions

- Write 0's port rests on VOID, whose line is zero. Write 1's rests on SP−2, which its
  translation computes every step and whose line pushes and calls write.
- Each write's enable gates its word lines, so on a step without a write the record's
  line positions keep the old line whole.
- The main read head already rested on VOID, the best stateless choice the replay found.

Text's wire changes fall from 106.2 to 102.5 a pass (`WL0h` −1.34, `WL1h` −2.14), and
overworld's fall by 2.2. No level, layer or hidden width moved, and the checkpoint stays
at 135.3 M dense with 98,464 nonzeros (+96).

### 16.6 One record a line

When write 1 lands in write 0's line, its record already holds write 0's byte
(`same_line`'s `s0`), so `WQ.2` is now `we0 ∧ record0 ∧ ¬Σ s0` and write 0 keeps no
record. The KV cache gets one cell per line written, as a RAM writes its row once. Text
keeps 877,196 records where it kept 1,024,869 (−14.4%; 66% of its two-write steps share
a line). `CircuitMemCore::commit` and the audit expect write 0's record left out exactly
when write 1's replaces the same line.

## 17. The PPU's line timing as a schedule

Mode 2 lasts 80 dots and a line 456, so a visible line's mode 3 starts at a fixed phase,
114 M-cycles after the last at normal speed ([P1] "Rendering"). The GBC CPU sees the
position only through reads of LY and STAT and through interrupts. The control
checkpoint ran a pass at every visible line's mode-3 start only to say "draw it now",
which was 144 of text's 186.5 passes a frame. Where no interrupt, transfer or timer can
need a pass before line 144, a pass now emits the next eight starts and the line each
draws, and the host counts down to them (`docs/control-in-weights.md` §19). Text then
takes 58.4 control passes a frame, and its fps rises 1.1% without the stats clocks.

## 18. The open row serves the operands

### 18.1 How the hardware does it

A RAM reads a row into its sense amplifiers and serves any column of the open row, so the
next access to the same row needs no new row read. The GBC CPU fetches an instruction's
operand bytes from the addresses after its opcode ([H1] ch. 3, one access an M-cycle), so
they sit in the opcode's row unless the instruction straddles a row boundary. Here the
cache's 8-byte line is the row. It is not the real geometry of the cartridge ROM or WRAM.

### 18.2 How our circuit did it

Three fetch heads read the lines of PC, PC+1 and PC+2 through three translations, and each
pick read its own head. The three lines are the same one for most steps, so a line change
flipped about the same 32 bits at all three. That made 30.8 head line-bit changes a pass
on text (`OP0h` 10.43, `OP1h` 10.65, `OP2h` 9.74), each a routed change.

### 18.3 The reshaping (F1)

- The opcode head reads PC's line (unchanged).
- Fetch k's byte sits at offset o = (pc & 7) + k − hb from the line's start. That is lane
  o of PC's line when o is below 8, and lane o − 8 of the next line from 8 up. One head
  reads line(PC) + 1 (`FNh`, its query an adder on pc's line bits and one translation).
  Only lanes 0 and 1 are picked, so only those 16 bits take channels (`Circuit::unread`).
- The only line that straddles a host boundary is FFF8..FFFF, because FFFF is IE, which
  the host serves. A third head (`FXh`) supplies its lane. That head's query is FFFF's
  fetch line, the HOST block's or, under OAM DMA, the FF block's, so it is constant
  except for the DMA bit.
- Each word line is a node with one step per (pc & 7, hb) that lands on its (line, lane).
  Each fetch's host flag is the sum over its lanes of the word line and that line's flag.

The census on text (`GBCFORMER_UNITS=1`, one model) shows wire changes from 102.5 to
83.2, pairs from 191.2 to 183.6, and updates from 1,112 to 1,085. The pairs fell less
than the changes, because the picks now read the opcode line's bits a layer after the
decode. Under background load text's fps rose 4.5% with the stats clocks and 1.7% without
(`docs/perf-log.md`, "the open row"). §20 replaces the three heads and the picks, and
every fetched byte now comes from a column head on PC's line, the next line or FFFF's.

## 19. Byte-wide RAM writes (H3): probed, not built

### 19.1 The design

A RAM writes one byte. The cache writes a whole line, so every GBC CPU write reads its
line (`WL0h`, `WL1h`), merges the byte in, and emits the line as eight record positions.
H3 gives every writable region (cartridge RAM, WRAM, VRAM, OAM, HRAM, REGS, PAL, HOST) one
cell per byte, and read-only ROM keeps one cell per line.

- *Reads stay the same.* A byte cell carries its line's key and 8 × its byte's bits in
  its own lane, 0 elsewhere. A line query weighs the line's eight byte cells 1/8 each
  when their logits are equal, so the attention result is the whole line, exact in f16
  (1/8 and 8·bit are exact), and every read path (fetch, data, the PPU's lines) is the
  same circuit. A ROM line cell keeps its value at scale 1, so the byte records need
  their own token family (or the ROM records one at 1/8 with `W_o` × 8), because each
  cell kind has one value scale.
- *Writes.* The GBC CPU emits the written byte and its address, and no write reads its
  line. H5's one-record-a-line merge goes away, and two writes keep two records.
- *Positions.* Byte cells need distinct positions for the runtime's per-cell eviction:
  byte cell (n, k) at 8n + k, a ROM line at 8n, every group and record at `8·LINES`.
- *The all-eight invariant.* A line over byte cells is exact only when all eight cells
  exist. The prefill writes all eight, eviction always replaces one, and the lookup's
  guard checks eight rows present, each with the canonical key, each value 0 or 8 in its
  own lane only.

### 19.2 The probe (measured)

The probe built the step side alone, as a throwaway build. The write ports emit
byte records (each lane the byte at its word line, else 0), with no write-line heads and
no merge. A scratch host merge (`H3_PROBE_MERGE`) folds the byte into the line cell. An
engine may not do that, so the probe is a measurement, not a candidate. It ran against
F1, interleaved over three rounds, at load 2.9 to 6.4:

| | F1 | probe |
| - | - | - |
| Memory heads, intermediate, dense | 10, 1,408, 135.3 M | 8, 1,280, 127.5 M |
| Text fps, `GBCFORMER_STATS=1` | 55.5, 54.4, 54.9 (54.9) | 56.1, 56.5, 56.0 (56.1) |
| Text engine pass, µs | 2.84 to 2.91 | 2.73 to 2.76 |
| (changed channel, reading layer) pairs | 183.6 | 173.7 |
| Text fps, no clocks | 57.0, 57.3, 56.3 (57.0) | 58.0, 58.0, 57.3 (58.0) |

The step gives +1.7 to +2.2%. Two gains are not counted: the 14 line positions a pass
that byte records leave out (H2's eight positions were about +1%), and the cheaper K/V
rows (records and rows take 0.11 µs a pass; a byte record's rows are a function of its
line address, lane and byte). Full H3 projects to about +4 to +5% on text.

### 19.3 Why it waits for a ruling

- A stock runtime's cache grows by about 70,000 cells for Yellow (+50%), and its attention
  per pass grows with it.
- Equal logits are not yet proven in every order. With f16 keys the eight cells' rotated
  keys round to the same ±1 entries. In an f32 cache they differ (a scratch probe, 21 of
  21 lines checked), and their logits came out bit-identical only in sequential
  summation. The harness's rule is exactness under any summation order. (§21 resolves
  this. One key bit per RoPE pair makes every key entry exactly ±1 at every position up
  to H3's, so the eight cells' logits are one number in any order.)
- It moves every digest and the DMA, HOST and palette records, and its real proof is
  llama.cpp token for token over the byte cells, which this round could not run.

## 20. Column select: the byte picked in attention

### 20.1 How the hardware does it

A RAM's row decoder opens a row, and its column decoder, driven by the low address bits,
connects one byte of that row to the data bus. Every column drives the same bus through
its own select, and an unselected column is off the bus ([H1] ch. 3 for the GBC CPU's one
data bus). The pick is part of the memory, not of the logic that reads the byte.

### 20.2 How our circuit did it

A head read the whole 8-byte line (its 64 bits each a channel), and the logic picked the
byte. The opcode's 32 nibble one-hots were nodes with one step per line byte, each step
reading that byte's four bits and its word line. The operands' picks were 16 select
nodes over the opcode head's lanes and the next line's. Every mux that reads read 0's
byte absorbed it, 16 steps a bit over two heads. A line change flipped about 32 line
bits, each reaching 16 nibble steps, and a change in PC's low three bits moved two word
lines, each reaching 32 (`OP0` 288 updates a pass on text, the largest single family).

### 20.3 The reshaping

- **A column is a head (`Circuit::head_onto`, `Query::gated`).** There is one head per
  (line, lane) a byte can come from, its query the row's line when its word line is 1
  and VOID's zero line when it is 0. The query stays linear. Address bit b's form is
  `a_b + (2·off_b − 1)·(1 − word)`, whose sign is `a_b` with the word on and the off
  line's bit with it off. So `S·(2·form − 1)` is ±S or ±3S a bit, and the line its signs
  spell beats every other by 2S a mismatched bit, as before. The widest logit is
  3S·19 + B = 60,416, under f16's 65,504. `Params::check` holds the margins for the 3S
  queries in f32 and f16 (`docs/stock-llama.md` §4). Only one KV head is needed, because
  the selection is in the query and each column's `W_o` writes only its lane.
- **The bus.** Every column of a byte writes the same eight channels (`Circuit::bus`).
  At most one is on a line other than a zero one, so each channel is 0 or 1.
- **The pick's readers read the bus.** The opcode's nibble one-hots are one step each
  over four bus bits. OP1 and OP2 are their buses, a level earlier than the picks they
  replace. The (HL) operand's mux reads read 0's bus, two steps a bit. `OP0h`, `FNh`,
  `FXh`, `D0h` and `D0r` go, replaced by 8 opcode columns, 10 for each operand (PC's
  lanes, the next line's first two, FFFF's) and 16 for read 0.
- **No level is added.** The word lines are level 1 (fetch) or the selection's level
  (read 0), at or below the translation's, so every column sits at its old head's layer.
- **VOID's cell must exist before any record.** An off column reads VOID's line. In a
  cache without that cell it reads the best other key instead, which at a record's own
  positions is the record's tokens, and a bus adds its columns. A prefill therefore
  decodes VOID's first line right after the top cell (`tools/stock-llama/src/cache.rs`,
  `tests/memory_dense.rs`).

### 20.4 What the engine needs (`engine/plan.rs`, `engine/run.rs`)

- The engine reads a query bit's form by its sign over [−1, 2] (`circuit::address_bit`).
- Columns of one row share an address unit, the row's forms, which the head bucket
  updates once. A column reads `word ? row : off`, and the engine recognises it only
  where the word and every row form are 0 or 1 over the declared ranges that the pass's
  range guard holds.
- The engine diffs a bus driven only by columns as the sum of its drivers' lane bytes,
  one XOR a bus, and drivers whose bytes overlap refuse the pass. Every other channel
  keeps one writer.
- A head diffs only its live bits, and a pass reads each (row group, line) once.

Without the last three the built circuit was slower. OP0 alone fell from 288 to 120
updates a pass as projected, but its columns cost more than the walk saved (layer 2's
memory reads went from 80 to 341 ns, measured with a timing probe).

### 20.5 Measured

`docs/perf-log.md`, "Column select", records text +4.9% without the stats clocks (from 56.7
to 59.5 fps at load 3.4 to 4.8, and from 55.8 to 58.1 at load 5 to 6), overworld +1.4%,
and the one model from 634.7 to 594.8 M dense.

## 21. One key bit per RoPE pair

### 21.1 How the hardware does it

A RAM's address decoder reads each address line as a level, so the same pattern of bits
selects a row whatever the cycle. Nothing in the comparison depends on when it is made.

### 21.2 How our circuit did it

The 19 line-address bits sat on dims 10..20 and 42..49, and HF's rotate_half pairs dim
t with t + 32, so pairs 10..17 held a key bit on both halves. A roped entry was then
±1 ∓ sin(p·θ_t), up to about 6·10⁻⁶ at reachable positions, which is more than half an
f32 ulp of 1 and different at every position. Within the line cache that was harmless
(one cell per line, margins of 256). But H3's eight byte cells of one line sit at eight
positions, and their logits were bit-identical only in one summation order (§19.3).

### 21.3 The reshaping

- Address bit i sits on the first half of pair 12 + i (dims 12..30), committed on dim
  31, and every partner dim (44..63) is zero in every query and key
  (`memory::attn::KEY_DIMS`, `COMMIT`). Pair 12 turns by 5.6·10⁻¹⁵ rad a position, so up
  to `KEY_REACH` (8·LINES + 2, every byte position H3 could use) cos is exactly 1.0 in
  f32 and sin rounds to 0 in f16. A roped key entry is then ±1 with a ±0 partner, a roped
  query entry is ±S, ±3S or B, and every logit is the same multiple of 1,024 in any
  order. `Params::check` runs the proof (`key_pairs_turn_by_nothing`), and
  `a_cells_logit_is_one_number_in_any_order` checks eight positions of one line in three
  summation orders, in f32 and f16.
- The overlay's null target moves to both halves of pairs 11 down to 8. It needs a
  margin, not an exact logit. With equal halves its score is `2·w·k·cos` of the turn,
  which rounds to `2·w·k` in f32 at every cell position on these pairs. At a part's own
  positions its query term is exactly 0, so it never meets a key bit.
- Layer 0's gathers keep `LOW_DIMS`, and no memory layer reads them.
- Pairs 0..7 turn too fast (pair 7 by 1.4·10⁻³ rad at P), so the 20 key dims and the
  null's 8 fill pairs 8..31, and the head stays 64 wide.

### 21.4 Measured

Speed is neutral, as projected, because the engine builds cells by formula. Text went
from 58.1 to 58.2 fps without the stats clocks and from 57.9 to 58.3 with them, at load
2.5 to 3.6. Every memory head moved, so cpu-memory, the PPU and the one model have new
digests, with sizes unchanged (`docs/perf-log.md`, "One key bit per RoPE pair").

### 21.5 Since: the 24-bit layout

MBC5's 8 MiB ROMs widened the line key to 21 bits (`docs/perf-log.md`, "MBC5 at 8 MiB:
the 24-bit layout"). At the new `KEY_REACH` pair 12 turns too far, so bits 0..17 sit on
the first halves of pairs 13..30 and bits 18..20 on the second halves of pairs 29..31,
which turn by less than 10⁻²⁷ rad. One bit per pair no longer holds on those three
pairs. `key_pairs_turn_by_nothing` now checks every roped entry on a key dim bit for bit
against its unroped value, so the any-order logit argument above still holds.
