# The host's GBC logic: census and design

The goal is that every GBC-specific state transition comes from a model. The host may
load the ROM, supply the buttons and the elapsed time, manage inference (sequences,
caches, threads, and moving outputs into the next inputs), show frames, play audio and
save data. Anything else GBC-specific the host computes is a gap.

This page lists what the host still computes with every component in weights
(`--weights all --one-model`, no lockstep), says which items are hardware and which are
plumbing, and gives the design that moves each hardware item into a model. Each move is
its own commit with its own gates. The census was read on 2026-11-09.

## 1. Already moved or moving

| Item | Where it was | Where it goes |
| - | - | - |
| LCDC, SCY, SCX, WY and WX | host latches handed to each line | PPU display bytes, written as records, read by a constant head (`docs/ppu-in-weights.md` §12) |
| The MBC3 clock | `machine/cart.rs` `Rtc` | the control model; the host supplies the elapsed wall time |
| OAM DMA and HDMA transfers | the host's `DI + elapsed` add, block expansion and stall arithmetic | the control model emits each run of OAM DMA bytes (page, 8-byte line, a bit per byte), each HDMA block's address bits and every stall, one pass per run or block (`docs/control-in-weights.md` §21, §21.5) |
| I/O dispatch (items 1 to 5, part of 13) | `route`, `display_index`, `cpu_visible_during_dma`, `read_dev`'s constants, VBK's latch, `clock_at` and `clock_hears` | the memory model's translation: port blocks, constant cells and the readback table (§3 step 2); the data reads and writes of the GBC CPU take it, but the fetch bytes, the DMA copies and the shadow writes still pass through `route` (§4) |
| The frame sequencer's edges, the APU's ticks, control's halves, the HALT wake and the schedule's starts (items 6 to 9, 11, 15) | `Lines::counter` plus 4 per M-cycle, `Clock::sequencer`, `Clock::ticks`, `Machine::apu_phase`, the doubling in `Ctl::encode`, the `IE & IF` test in `step::step`, `Schedule::at` | the control model emits each edge's distance, the drop a DIV reset causes and the wake bit, and converts M-cycles to halves; the APU model takes M-cycles and the speed and keeps the tick's phase and a waiting step; the host counts down to each emitted instant (`docs/control-in-weights.md` §22, `docs/apu-in-weights.md` §14) |

## 2. The census

Each row is a host computation on the path with every component in weights. H is
hardware behaviour that must move. P is plumbing the criterion allows.

| # | Where | What the host computes | Class |
| - | - | - | - |
| 1 | `machine/mod.rs` `route`, `read_synced`, `write_unblocked` | which device an I/O address reaches, CGB-only registers dropped on DMG, an LCDC write sent to both the PPU and control | H |
| 2 | `memory::layout::display_index` | a display register's bus address to its index in the PPU's display bytes | H |
| 3 | `cpu_visible_during_dma` | I/O writes dropped and reads answered 0xFF while OAM DMA runs (the memory model already lands those writes in `FF`) | H |
| 4 | `read_dev` | the unusable area reading 0x00 and unmapped I/O reading 0xFF | H |
| 5 | `Ppu::read` | VBK's readback from a host latch (the memory model holds a VBK record) | H |
| 6 | `ControlModel::lines` | the APU's DIV input as the model's counter plus 4 per M-cycle since its pass | moved (§1) |
| 7 | `device::Clock::sequencer`, `ApuModel::begin_cycle` | the frame sequencer's DIV bit and its falling edge, per tick | moved (§1) |
| 8 | `device::Clock::ticks`, `Machine::apu_phase` | M-cycles to APU ticks at double speed, the phase across blocks | moved (§1) |
| 9 | `Ctl::encode` | M-cycles to 2-dot halves for control's PPU timing at the speed | moved (§1) |
| 10 | `machine/cart.rs` | mapper write decode, the RAM-enable rule and save content taken from the host's cartridge RAM | H |
| 11 | `step::step` | the HALT wake test `IE & IF & 0x1F` | moved (§1) |
| 12 | `sm83/abi.rs` `late_fetch` (since removed) | the HALT bug's fetch address and the HOST block for host-served fetches | H |
| 13 | `CopyFrom::at` | a DMA source the host serves takes the host's byte | H |
| 14 | `step::step`, `Machine` | the shadow machine: every fetch and write also runs on host memory, VBK and the window counter | delete once 1 to 13 are gone |
| 15 | `Schedule::at`, `ControlModel::pass` | `first + period·j` for scheduled mode-3 starts | moved (§1): the host counts down from the emitted first start and period |
| 16 | `ApuModel` | the 48 kHz sample instants | P (the host's audio clock) |
| 17 | `Machine::advance`, `ApuModel` | blocks cut at the model's horizons, segments cut at events | P |
| 18 | `Bus::ack`, buttons, serial | outputs carried to the next inputs | P |
| 19 | `machine/ppu.rs` frame buffers, `rgb`, `HighPass`, the workers | presentation and threads | P |
| 20 | `Event::needs_span` | a host table of which events need a span | G: it chooses the pass shape, and with lockstep off nothing checks it (§4) |
| 21 | `Machine::advance`, `ApuModel::end_cycle` | M-cycles to real time (2-dot halves) by the speed bit, for the frame time and the 48 kHz sample clock | H (the speed's meaning for time), left: the host's clocks need real time, and the model would have to emit the halves each block spans |

## 3. Design, in build order

1. **The frame sequencer's clock (items 6 to 9). Built** (`docs/control-in-weights.md` §22, `docs/apu-in-weights.md` §14). Control emits `FE`, the M-cycles from
   the pass to the next falling edge of DIV bit 12 (bit 13 at double speed), from the
   counter's low bits, and an FX bit when a DIV write drops the bit. The host counts
   down to it as it does for mode-3 starts and hands the APU an edge. The APU's segment
   takes M-cycles and the speed bit, and its own state keeps the tick phase, so the APU
   model converts M-cycles to ticks. Control's circuit shifts the M-cycles it is given by
   its own speed bit, so the host passes M-cycles only. `Lines::counter`,
   `Clock::sequencer`, `Clock::ticks` and `apu_phase` go.
2. **I/O dispatch from the memory model (items 1 to 5, 13).** The memory model's address
   translation already places every access. It gains one marker block per device
   (control, APU, the PPU's display bytes), constant cells for unmapped (0xFF) and
   unusable (0x00) reads, and VBK and SVBK readback from their records. The model emits
   the physical address of a host-served access, and the host dispatches on its high
   bits, by concatenation as `layout::place` places a copy. An LCDC write is two
   records, one per device.

   *The design as built.* The translation passes bus bits 0..6 through on every read, so
   the dispatch lives in the high bits only:

   - **Ports.** `layout::IO` is 16 blocks of 128 bytes, `IO + code·128 + (a & 0x7F)`.
     Code 1 is control (FF00-FF02, FF04-FF07, FF0F, STAT, LY, LYC, DMA, and on CGB KEY0,
     KEY1, HDMA and the palette index and data ports; IE at low bits 0x7F), 2 the APU
     (FF10-FF3F), 4 the cartridge clock's registers (A000-BFFF with the clock selected),
     5 its latch (6000-7FFF), and 8 + i the PPU's display byte `0x80 | i` (BGP, OBP0,
     OBP1, LCDC, SCY, SCX, WY, WX). Code 11, LCDC, is the multi-target marker: the host
     hands the byte to the display byte and to control. Port blocks hold zero tokens, so a
     host-served read adds the host's byte to a zero, as VOID does today.
   - **Constants.** Unmapped I/O reads code 0, a block of 0xFF. The unusable area reads
     VOID (zeros). VBK and SVBK read `TAB + (bank << 7) + (a & 0x7F)`, eight blocks whose
     bytes at 0x4F and 0x70 hold `0xFE | (j & 1)` and `0xF8 | j`, so their readback is a
     memory read whose high address bits are the register's bank bits. Writes to
     constants and to VOID are not recorded. VBK and SVBK writes stay records in REGS.
   - **Registers a device also hears.** A record at `REGS + SVBK` also reaches control,
     whose copy feeds the host's WRAM bank, and records at `REGS + RAMG` and `REGS + RAMB`
     reach the clock's enable and select. `layout::port` names these by full address, so
     a write reaches memory when `recorded(p)` and a device when `port(p)` is Some.
   - **The clock.** MAPPER bit 5 says the board has MBC3's clock, written at load like the
     other bits. On such a board selects 08-0C reach the clock port; on a clockless MBC3
     they read 0xFF from FF, as the cartridge does.
   - **The host.** A host-served read takes its port from RP0 or RP1, decoded in the
     request pass (RP1 is no longer audit-only). Each write goes to host memory when
     recorded (the shadow machine, item 14) and to `port(WP)` when it names one.
     `route` remains the reference machine's bus, used without memory in weights and by
     the lockstep reference. `Lockstep` asks the model where a host-served read goes.
   - **DMA (item 13).** `place` refuses a source in the I/O page or IE, whose device no
     concatenation names. A source in the clock port keeps the host's byte.
   - **Reads through the FF page.** LDH A,(n) and LD A,(C) read through `RFF`, an FF-page
     translation of their byte under their opcode, which shares the rare heads, so the
     port table reads a fetched byte or C only on those steps. Wired translations read
     the low bits gated by the I/O page.
   - **Checks.** Under memory lockstep the audit holds RP, WP and each served read's port
     to `layout::translate`, and a machine test holds `translate` to `route` at all
     65,536 addresses on ten boards. A test flips read 0's emitted port code and the
     audit stops the run. The fetch path (item 12) is unchanged: every host-served fetch
     lands in HOST.
   - **What remains.** The shadow machine (item 14) still applies every recorded write by
     bus address through the reference map, so it keeps the reference's VBK, its SVBK
     copy and the cartridge's registers. Fetches from the I/O page go through HOST (item
     12). A DMA source in the clock's port takes the host's byte, and one in the I/O page
     is refused rather than served (item 13). The clock's chip hears RAMG and RAMB on
     every board; only a board with the clock reads it.
3. **HALT and fetches (items 11, 12).** Control emits a wake bit, and the host skips GBC
   CPU passes while the model's state is halted and the bit is clear (built, §22 of
   `docs/control-in-weights.md`). The GBC CPU emits
   the fetch addresses of a host-served fetch.
4. **Saves and the shadow machine (items 10, 14).** A battery save is the model's
   cartridge RAM region, committed when the model's RAM-enable record turns off after a
   write landed there. The host's cartridge RAM, mapper writes, WRAM, HRAM, VRAM, OAM and
   VBK then have no reader with lockstep off, and the step stops writing them.

Each step keeps the rules: exact by construction, lockstep against the reference with the
first divergence stopping the run, every engine shortcut guarded, and every hash in
`tools/gate.sh` unchanged.

## 4. What remains (census of 2026-11-09)

A read of the code with every component in weights as one model and lockstep off. G is
a GBC state transition the host still computes, L is load or save-time GBC knowledge,
S is shadow state nothing reads with lockstep off, and P is plumbing.

| # | Where | What the host still decides | Class |
| - | - | - | - |
| 1 | `step::step`, `Machine::read`, `sm83/abi.rs` `late_fetch` (since removed) | the three fetch bytes read through `route`, the cartridge's banking, `wram_index`, VBK and the HALT bug's fetch rule for the HOST records | G |
| 2 | `cpu_visible_during_dma` | which accesses the transfer blocks, which decides when a DMA run is cut | G |
| 3 | `Machine` copy loop, `CopyFrom::at`, `cart.clock_at` | each DMA byte read and written through `route`; a clock-port source takes the host's byte | G |
| 4 | `read_control` | BCPD and OCPD as an indirect read: control gives the index, the PPU the byte, the host 0xFF under the mode-3 lock | G |
| 5 | `layout::port` | the REGS records' devices: SVBK to control, RAMG and RAMB to the clock | G |
| 6 | `layout::place`, `AnyCore::copied` | the bus map's ranges for DMA bytes, VRAM's bank from VBK bit 0, the RAMMASK test | G |
| 7 | `control_circuit` OAM destination, `device::Copy` | FE00 and the 16-byte HDMA block as host constants | G |
| 8 | `Event::needs_span` | which events need a span | G |
| 9 | `ControlModel::double`, `HALVES_PER_FRAME`, the APU's sample phase | M-cycles to real time by the speed bit (item 21) | G |
| 10 | `Machine::write_memory`, `machine/cart.rs`, `Session::run_frame` | battery saves from the host's cartridge RAM, by its own mapper and RAM-enable rules; a save that changed only the clock is dropped | G |
| 11 | `Machine::peek` | an observer's WRAM view through `wram_index` | G |
| 12 | `rom.rs`, `Hardware::resolve`, `Kind::from_type`, `ram_size` | header decoding | L |
| 13 | the `post_boot` functions, `Banking::new`, `memory_image` | the state the models start from | L |
| 14 | `machine/rtc.rs` footers, `set_time` | the clock's save format | L |
| 15 | `Ppu` latches and window counter, shadow VRAM, OAM and VBK | written, never read with lockstep off | S |
| 16 | `Bus::peek`, `physical`, `banking`, the audits | run only in lockstep | S |
| 17 | port decode, `Copy::bytes`, the schedule and edge countdowns, the wake bit, `ack`, buttons, serial, the LCDC fan-out, the APU's segment cuts, `HighPass`, `rgb`, the workers | carrying outputs to inputs, presentation and threads | P |

So the I/O dispatch moved for the GBC CPU's data accesses, and the fetch path, DMA
copies and saves still read the host's machine. The minimal runner
has to drop items 1 to 11 and 15, and say for 12 to 14 which load-time facts it keeps.

## 5. The host rules: what moved, and what remains for the minimal runner

The bar is the one at the top of this page. Each row below is a rule from §4. It says
what moved into a model, what was shown to be plumbing, and what is still host code
with every component in weights, as one model and with lockstep off.

| §4 item | Rule | State after §5 |
| - | - | - |
| 1 | the three fetch bytes read through `route`; the HALT bug's fetch address (`late_fetch`, since removed) for HOST records | open. The memory model reads every opcode byte from its own cells; the host's three bus reads matter only for a fetch the model flags as host-served (I/O page, IE, the unusable area, the MBC3 clock). Design: the fetch translation emits each such byte's port address, as RP does for reads, and the host reads that port and keeps the HOST record |
| 2 | `cpu_visible_during_dma` | the hardware-accuracy work's. Runs now count the accesses it blocks (`dma_blocked=` on the run line): 0 in Yellow, Crystal, Zelda DX and Mario DX over 300 frames with every component in weights |
| 3 | the DMA copy loop through `route`; a clock-port source takes the host's byte | open. Design: one memory-stream pass translates each DMA line's source and destination, as the per-line OAM DMA runs do for control |
| 4 | BCPD and OCPD | the hardware-accuracy work's. Runs count the reads (`palette_reads=`): 0 in the same four runs |
| 5 | `layout::port`'s REGS aliases (SVBK to control, RAMG and RAMB to the clock) | open. Design: the translation emits a second port field for a register write a device also hears |
| 6 | `layout::place`'s bus-map ranges, VRAM's bank, RAMMASK | open, with item 3 |
| 7 | FE00 and the 16-byte HDMA block | moved. Control emits `DD`, OAM DMA's destination page, and `CL`, an HDMA block's length. The host places each byte at the page, line and bits the model emitted; `device::Copy::Block` carries the length |
| 8 | `Event::needs_span` | removed. Every control event runs after the span before it, the hardware's order, so the host chooses no pass shape. No weights changed |
| 9 | M-cycles to real time by the speed bit | the speed's meaning moved. Control emits `HR`, the 2-dot halves an M-cycle lasts after the pass, and the machine counts real time by it. Still host: the frame period (`HALVES_PER_FRAME`) and the 48 kHz sample instants (`CPU_HZ` against the sample phase), which are the host's frame and audio clocks |
| 10 | saves from the host's cartridge RAM by its mapper rules | moved. A save is the cartridge RAM in the model's cache, due when the A000 window the model emits (`MAPS`) leaves cartridge RAM and the clock's port after a write landed in either (`sm83::Battery`). Confirmed and fixed: a game that set the clock and disabled RAM lost the save, since the clock write reaches the clock's port and never the host's cartridge (`tests/battery_saves.rs`). A cartridge the host wrote before the hand-off keeps its pending save (`Battery::seeded`). Two boards differ from the host: on MBC2 the model's cells hold a written nibble as the bus reads it back, `0xF0 | n`, where the host's cartridge kept `n`, so the save carries the high nibble set; on an MBC3 without the clock a select of 08-0C moves the window to `FF`, so the save finishes there, with the same bytes, rather than at the RAMG write. Both are pinned by the test. The BGB footer's format stays host code (L) |
| 11 | `Machine::ram_byte` through `wram_index` | moved off the host's arrays. `Session::ram_byte` reads the model's image, D000 at the bank window the model emitted (`MAPW`) |
| 12, 13, 14 | header decode, the post-boot states, the clock's save format | 13 and the header's CGB byte closed for a run from the boot ROM (§7): the model runs it from power-on, and the host reads nothing from the header for the mode and hard-codes no post-boot state. Without a boot ROM both stay, labelled, as a snapshot handed over as data. The mapper's type and sizes from the header (12) and the clock's save format (14) stay load-time facts (L) |
| 15 | the PPU's latches and window counter, shadow VRAM, OAM and VBK | deleted with the GBC CPU, memory and the PPU in weights and nothing beside them (`Machine::drop_shadow`, `Ppu::render_line`). The cartridge's writes still reach the host's cartridge, whose clock decode items 1 and 3 read |

**The runner's foundation.** The crate builds without the Game Boy: every GBC module,
and the GBC rules in `memory::layout` (the translation's specification, the port
decode, the constants' bytes, the mapper's defaults), sit behind the default `gbc`
feature. `cargo check --lib --no-default-features` builds the inference core alone.
Every exported GGUF carries `gbcformer.meta`, everything besides the weights the engine
needs (configuration, slots, layout, wire ranges, memory and overlay specs, the kept
heads and units, the streams), and `gguf::open_model` builds the model from the file
alone. `GBCFORMER_GGUF_ONLY=<model.gguf>` runs the one model with no circuit compiled.

**What the runner still needs.** The streams' transport today lives in the GBC
modules: `MemCheckpointCore`, `ControlModel`, `ApuModel`, `PpuModel` and
`Machine::advance`. A runner that links only the core must carry it, and it then holds
items 1, 3, 5, 6, 9's frame and sample clocks, and 12 to 14 until they move. The runner
is not built in this round.

## 6. The host rules' second round: fetches, DMA lines and second ports

Each unit below moved a §5 rule out of the host's code into the memory model's outputs.
Every one adds result slots that a pass decodes only when it needs them, so an ordinary
GBC CPU pass decodes what it decoded before.

**Fetched bytes the host serves (§4 item 1).** For each fetched byte the host serves (the
request's RQ bits 4 to 6), the model emits `FP0` to `FP2`: the byte's bus address,
pc + k − hb with the HALT bug's repeat included, translated as a read. Every input of
those translations is gated by "the host serves some fetched byte". On any other step,
only the 17 gates that read PC and the HALT bit change. When a fetch is flagged, one
more pass decodes the `FP` slots. The host then reads the port each one names, or for a
constant cell (the unusable area, an unmapped register, VBK's and SVBK's readback)
copies the model's own cell. It keeps the byte at `HOST | (FP & 0x7F)`, and the request
pass runs again. The host no longer computes a fetch address or reads the three bytes
at PC for the memory model. (`late_fetch` then survived only in the audit; since then the audit reads the fetched stream the host keeps in `Inputs::op`.)
`step::request` is the protocol: `StepFacts::fetch_ports`, then `Core::serve_fetch`. Checks:
- `sm83::tests::fetch_bytes_come_from_their_lines` holds every flagged byte's `FP` to
  `layout::translate`, with and without the HALT bug and OAM DMA.
- The memory audit holds it in lockstep.
- `tests/host_fetch.rs` calls into TMA, which holds RET, and halts at FFFE so the HALT
  bug fetches IE twice, under memory lockstep.
- `a_wrong_fetch_port_stops_the_audit` flips a port code and expects the audit to stop.

The four games never fetch from a port, so the extra pass never runs in them.

**DMA lines (§4 items 3 and 6).** The memory circuit has four more input slots, `DSL DSH
DDL DDH`, and six result slots, `DP` and `DQ`. The inputs hold a DMA line's source and
destination bus addresses and are 0 on every GBC CPU step. The results are those
addresses translated as reads under the registers the model holds, without OAM DMA's
block. A DMA pass (`MemCheckpointCore::dma_line`) sets the inputs, decodes only `DP`
and `DQ`, and resets them. Byte k of the line lands at `DP | k` and `DQ | k`. Several
things go:
- `layout::place`, `layout::Banks` and the copy's reads of VBK and RAMMASK from the
  image;
- the MAP slots as the DMA's banks;
- the host's read of every DMA source through `route`. `Machine::land` reads a byte only
  while the shadow machine runs, for lockstep.

A source line in the clock's port takes the host's byte while the shadow machine runs,
and is refused without it. A source in another port is refused, since one line can
name several devices. `layout::tests::a_lines_translation_places_its_bytes` holds the
line property at every address outside the I/O page. `dma_lines_are_translated_as_reads`
holds `DP` and `DQ` to the specification. Memory lockstep holds every placement.

**Second ports (§4 item 5).** A write a second device hears names that device in
`WD0` or `WD1`, flagged by WQ bits 4 and 5:
- SVBK's record reaches control's FF70;
- LCDC's display byte reaches control's FF40;
- on a board with MBC3's clock, RAMG's and RAMB's records reach the clock's enable and
  select, which have port codes 6 and 7.

`WD` comes from the write's own physical address, four equality steps a write. A
flagged write costs one more pass. `layout::port` loses its REGS aliases and the
multi-target LCDC variant, so each emitted address names one device.
`layout::second_port` is the specification, and the memory audit and the machine's
65,536-address test hold the model to it.

**MBC2 saves.** The reference cartridge now holds each MBC2 cell as the bus reads it,
`0xF0 | n`, and masks a loaded file's high nibbles. The model's save and the host's are
then byte-equal 512-byte files, as SameBoy and BGB write, and a file loads the same
whatever its high nibbles hold. SameBoy ORs 0xF0 into every read (`Core/memory.c`), and
mGBA's 256-byte packed files are a different format. Before, a loaded save of `n` read
back `n` from the model and `0xF0 | n` from the reference, and an unwritten cell read 0
from the model where the host read 0xF0. `tests/battery_saves.rs` pins both directions.

**The frame and sample clocks (§4 item 9).** Control emits `FH`, the 2-dot halves a
frame of emulated time lasts (35,112, 154 lines of 456 dots), as a constant in its weights,
as it emits `DD` and `CL`. `Session::run_frame` ends a frame when the halves it counts
by `HR` reach the `FH` the model last emitted (`Lines::frame`). The host keeps no frame
period. The APU model keeps the 48 kHz sample clock in its state (`sq`, `sr`:
`750·sq + sr` units of 2^-14 of a sample period since the last sample) and emits `SD`,
the M-cycles from the end of a segment to the next sample at its speed. The host counts
`SD` down and ends a segment with a sample at 0, so it keeps no sample phase. The
arithmetic is exact: 48,000 samples a second against 2^22 T-cycles is 750 units per
M-cycle at normal speed and 375 at double, with a sample every 16,384 = 21·750 + 634.
A segment moves r by 0, 116, 375 or 491 modulo 750, and `SD` is `21 − q + [r < 634]`, or
`42 − 2q + [r < 634] + [r < 259]` at double speed (`apu_circuit::circuit::sample_clock`).
A speed switch costs one pass over no M-cycle at the new speed, which re-emits `SD`. The
hand-off costs one pass over no M-cycle too. `ApuLockstep` holds every sample's instant
and mix to `machine::apu`, whose phase maps onto the model's exactly (2^8 of its units
to one). A control mutant that lengthens the frame by a half is caught.

## 7. The boot ROM in the weights

A run from a boot ROM you supply (`--boot cgb_boot.bin`, `Session::power_on`) starts every
device at its power-on state and the GBC CPU at 0000. Nothing is read from the
cartridge header for the console's mode, and no post-boot state is hard-coded. The boot
ROM's own writes set up the console, and the models carry the result.

- **The overlay.** Four registers join the memory model's cells: BOOTING (1 for a
  power-on run, written at load), UNMAPPED, COMPAT and KEY0. `layout::BOOT` is a 16 KB
  bank written at load: the cartridge's bank 0 with the boot ROM over 0000-00FF, and on
  CGB hardware 0200-08FF. While BOOTING is set and neither UNMAPPED nor COMPAT is, bank
  0's window (the `MAP0` slot, computed once a pass from the registers) is that bank, so
  no case reads an address bit for the boot ROM, and after FF50 it costs nothing. A
  first design carved the boot ranges out of bank 0 by address and added 35 gate updates
  to every GBC CPU pass, boot or not.
- **KEY0.** The boot ROM's FF4C write is a register record on CGB hardware while the boot
  ROM is mapped, and its second port reaches control's KEY0 (the readback). After FF50
  a KEY0 write reaches nothing (`layout::io_write_code`).
- **FF50.** The translation lands the write on COMPAT when the hardware is a CGB and
  KEY0 bit 2 is set, and on UNMAPPED otherwise: two exclusive cases over register bits,
  no extra level. Either one unmaps the boot ROM; COMPAT also leaves CGB mode, so VBK,
  SVBK, KEY1, HDMA and the palette ports route nowhere (`Banking::cgb`).
- **The PPU.** FLG bit 3 is bit 0 of the COMPAT cell as the memory model last recorded it.
  The record joins the PPU's video queue in order with the VRAM and OAM records, and the
  PPU checkpoint holds it (`PpuCheckpoint::mode`). Under PPU lockstep it is held to the
  reference's mode every line.
- **The host.** It follows the model's FF50 record (`Bus::write_memory` with the record's
  physical address): COMPAT unmaps and leaves CGB mode, UNMAPPED unmaps. Beside a
  reference it holds the model's choice to the reference rule (KEY0 bit 2 on CGB) and
  stops at a difference. The rule itself (`Machine::reference_ff50`) runs only when the
  memory is on the host.
- **Without a boot ROM.** `Machine::new` is a labelled convenience: the header's CGB byte
  picks the mode, and the post-boot registers are a snapshot handed over as data.

## 8. The mode-3 length as its own stream

Each visible line's mode-3 length is an input to the control logic (`ML`, asked by `MQ`
at every pass past M-cycle 63 of a visible line while STAT's mode-0 interrupt is on).
The PPU checkpoint emits it with the line's pixels, on its own thread, so the control
logic waited for the renderer's backlog. `ppu_circuit::circuit::build_length` computes
the length alone from the scanline circuit's OAM scan, its window test and
`mode3_length`, over the same OAM image and display registers, and threads the window's
arming. It is `one_model::Part::Length`, the fifth stream over the shared weights, and
its own checkpoint without `--one-model`.

`LengthModel` runs it on the machine's thread as each line is handed over, from the same
records the renderer follows (OAM, the display bytes, COMPAT), and publishes the length
at once. The renderer's own `ML` is held to it line by line. Under PPU lockstep the
length model is held to the reference's length and window arming. The first difference
stops the run. The host only carries records and the line number; the length comes from
the model. The costs are in `docs/perf-log.md`, "Host rules".
