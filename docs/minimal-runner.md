# The minimal inference runner

The completion bar for "the whole Game Boy Color in the model" is that every GBC-specific
state change comes from the model. The host may load the ROM, supply the buttons and the
elapsed time, manage inference (sequences, caches, threads, moving outputs into inputs),
show frames, play audio and store saves. The proof is a program that links no GBC hardware
code and reproduces the full program's results: `tools/runner`.

## 1. Design

The runner is a loop over the one model's six streams (the stepper the sixth) that knows nothing about the Game
Boy. Everything it needs to know about how the streams feed each other is data the
compiler writes into the GGUF next to the weights, the **wiring table**
(`gbcformer.wiring`, `src/wiring.rs`). A run starts from a **snapshot**, also data.

**The wiring table.** A stream plays one of six roles, named by how its passes are cut:

| Role | Its pass | The GBC part that plays it |
| - | - | - |
| steps | one step of the machine; its length in cycles is an output | the GBC CPU with its memory |
| spans | after the cycles since the last pass, at an event or at the horizon it emitted | the control logic |
| segments | the cycles and events since the last pass, closed at the events the role lists | the APU |
| lines | one row of pixels, triggered by the spans stream | the PPU |
| length | one line's mode-3 length, as the lines stream's line is queued | the PPU's length model |
| stepper | one step of the pixel FIFO, for a line the length stream's verdict marked | the PPU's pixel-FIFO stepper |

For each role the table names, by slot, what the runner moves:

- **carries**: an output fed to an input at the next pass (the state);
- **composed inputs**: an input built from a carried byte and another stream's outputs
  (the GBC CPU's `CTL` from its own `CTL'`, control's OAM DMA flag and its access locks);
- **pass sets**: the output slots each pass decodes (`sm83::mem_passes`), and the flags
  that ask for a further pass (a fetched byte the host serves, a read the host serves, a
  second port, a copy line);
- **access timing**: each data access's cycle within its step (`RC`, `WC`), whether the
  locks matter at each access, and the inputs (`AS0`, `AS1`) that hand each access its
  bus state where it differs from the step's start;
- **fields and flags**: every output the runner acts on, as little-endian slot bytes with
  a shift, a mask and a bias (`horizon = H + 1`), or a bit of a slot;
- **event codes**: the code bytes of each event a spans pass takes;
- **ports**: a physical address the model emitted names a port block by its high bits;
  the table gives each block's target stream and, for each of its 128 low addresses, the
  address byte that stream takes there;
- **the records the lines stream follows**: the memory ranges (VRAM, OAM) and the cell
  (compatibility mode) whose changed bytes reach the lines and length streams;
- **the save's footer**: on a cartridge that counts wall time, each state bit of the
  clock's registers routed to its bit of the 48-byte footer, probed through the full
  program's `decode_state` and `Rtc::footer`, then the wall time last read; the elapse
  event takes one binary digit of the elapsed seconds a pass (`clock::DIGITS` of them);
- **presentation**: the shades, the 15-bit colour widening, the blank colour, the board's
  high-pass filter's gain and shift, and the buttons' bits.

`src/gbc_wiring.rs` fills every entry from what the full program runs: a constant its own
decoder reads (`sm83::bits`, `control_circuit`'s and `apu_circuit`'s bit names), or a
probe of its function (`layout::port` at every address of every block, each event's code
bytes from `Ctl::encode`, the colours from `rgb`). Its tests hold every slot name to the
stream that reads it and the presentation entries to the host's functions over every input.

**One transport.** `src/transport.rs`, in the inference core, holds the host's side of the
streams: the countdown that cuts the spans stream into passes (horizon, scheduled starts,
edges, look-ahead spans, the stop owed), the copies a pass hands over (runs within a line,
blocks, and the blocks the pass holds to the end of the step that runs), applying a block's
signals, the block loop with its stop and the stopped cycles, a step's accesses at their cycles (the
states settled over the step's requests, the reads before a copy's end served on the way,
each prediction held at its access), the battery-save tracker and the output filter. The
full program (`control_circuit::ControlModel`, `machine::Machine`, `step::step`,
`sm83::MemCheckpointCore`) and the runner both run it, so their transport is one by
construction, with nothing copied. The passes of the spans stream go through one trait,
`transport::Passes`, whose `pass_with`, `pass_alone`, `pass_holding`, `cut`, `stop` and `advance` both
sides run, and the look-ahead (`Start::place`, `held_lock`, `bus_state`) is shared too.
What each side keeps is the decoding, and it differs by construction: `ControlModel`
reads a pass's outputs from its circuit's field layout (`Outcome`), the runner reads the
same outputs by the wiring table's slot names, and each then hands the shared code the
same values (the cut, the run, the block, the locks). The lock query is one event on both
sides, a STAT read: the full program names it `Event::Read(0xFF41)`, and the runner takes
its code from the table.

**The snapshot.** `GBCFORMER_SNAPSHOT=<dir>` makes the full program write its hand-off as
data, stamped with the digests of the wiring it ran and, under `GBCFORMER_GGUF_ONLY`, of the
GGUF's tensors (`wiring::Manifest::check_snapshot`: the runner refuses a snapshot for another
wiring or model, or one written without a GGUF): each stream's first input bytes by slot name, the memory image, the PPU stream's
image and the length stream's (bytes and token flags), the two frame buffers, and the facts
a run keeps (passes per line, the PPU's mode bits, the save's length and pending state,
whether the cartridge counts wall time, the last line's length, the frame number, how far
the last step ran past the frame's end, the clock's last reading and the filter's charge).
It leaves the ROM and boot ROM out: the runner fills those cells from `--rom` and,
at power-on, `--boot`.

**Power-on.** With `--boot` and no snapshot the runner builds the state itself, from the
ROM and the boot ROM alone, by the wiring's `PowerOn` tables (§2 says what they hold and
why). `--check DIR` holds the built state to the full program's power-on snapshot.

**The runner** (`tools/runner/src`):

- `slots.rs` resolves the table's slot names to positions once and decodes passes, each
  stream's tokens in a `sequence::Sequence` so a pass handles the positions that changed;
- `steps.rs`, `spans.rs`, `segments.rs`, `lines.rs`, `length.rs` drive one role each;
- `ports.rs` dispatches an emitted address by its high bits;
- `scheduler.rs` runs each step's accesses at their cycles (the other streams first run
  up to each, the step asks again under each access's bus state as the spans stream says
  it will be there, block copies wait for the step's end) and the blocks between steps;
- `present.rs` maps colours;
- `worker.rs` puts the segments and lines streams on threads of their own (`--threads 0`
  keeps one; with threads the memory cells' K/V rows are also written by a worker,
  `Cells::defer`, and settled before a save or the report): each takes its calls in the
  order the scheduler makes them, so the passes are the inline ones.

## 2. What it does not know, and what it keeps

`tools/runner/tests/quarantine.rs` holds the runner's source and `src/transport.rs`,
comments stripped, to three rules: no string literal that is a slot name of the model's
streams (read from the GGUF in `GBCFORMER_GGUF`; skipped without it), no hexadecimal
literal and only decimal literals from a short list (`0 1 2 3 4 8 60 1000`: indices, a
byte's bits, the scripted clock's frames a second, a batch size), and none of 33 terms of
the machine (`cpu`, `ppu`, `oam`, `hdma`, `lcdc`, `mbc`, ...; the ROM and the boot ROM
are named, since loading them is the host's job). A fourth test lists every line that
branches, each with its review (the ratchet of "What the transport still decides" below),
and a fifth holds the scanners to a line that breaks each rule. What this proves is narrow: the runner's text names no slot, no address and no GBC term. That it
holds no rule is the shared transport's job (§1) and the record comparison's (§4).
`tests/dependencies.rs` holds its crate graph to gbcformer without the `gbc`, `stock` and
`frontend` features, and its binary to no symbol of `memory::layout`, `machine`,
`reference`, `device`, `sm83`, the circuits, `one_model`, `stack`, `step`, `session`,
`lockstep` or `gbc_wiring`. `tools/gate.sh` runs both.

**What the host learns from the header at load.** Reading the cartridge header when the
ROM is loaded is the one place the host still interprets cartridge bytes. It takes:

- byte 0x147, the cartridge type: which parts the cartridge has (the mapper chip, RAM, a
  battery, a real-time clock);
- byte 0x149, the RAM size;
- the ROM's length;
- in the full program only, with `--hardware auto`, byte 0x143 bit 7, the CGB flag, which
  picks the console to run (`Hardware::resolve`); the runner takes the console from the boot
  ROM's length instead.
- in the full program only, for its labelled start without a boot ROM (`Machine::new`'s
  post-boot state, never the runner's): KEY0 from byte 0x143 and the logo (0x104-0x133) for
  the screen, as the boot ROM would leave them; and the title (0x134-0x142), printed only.

From the first three the board table (`wiring::Board`, one entry per header the full program
accepts) gives the memory model's register cells (the mapper's kind and its bank masks), the
RAM's length and fill byte, the save's length (0 without a battery) and whether the board
counts wall time. Nothing else is read from the header; the boot ROM in the weights reads the
logo and the CGB flag itself.

This describes which hardware is plugged in, not how it behaves. On a real console these
facts are wires and chips on the cartridge's board: the mapper chip soldered to it, the RAM
chip's size, whether a battery and a clock crystal are fitted. No code reads them, and the
header bytes are the cartridge's own label of that board. Every behaviour of those parts (a
bank switch, a RAM enable, the clock's counting, which bytes a save holds and when they
change) is computed by the model from the cells the table seeds; the host applies no mapper
rule after load. A header the table lacks is refused, as the full program refuses it.

**What stays as data, and why.** These facts reach the runner as tables, or as rules in the
shared transport, rather than from a pass:

- **The board.** From power-on the runner looks up the ROM's two header bytes (0x147, the
  mapper; 0x149, the RAM size) and its length in a table, one entry per header the full
  program accepts at every length from 32 KiB to 8 MiB. Each gives the memory model's
  register cells (the mapper kind, the bank masks), the RAM's length and fill byte, the
  save's length and whether it counts wall time. The boot ROM in the weights does not make
  this unnecessary: it reads the header's logo and CGB flag, but a mapper is cartridge
  hardware that no boot code configures, so the memory model needs its kind as a load-time
  fact (`docs/host-in-weights.md` §4 items 12 to 14). A header the table lacks is refused,
  as the full program refuses it (`machine::cart`).
- **The console.** The boot ROM's length names it (256 bytes DMG, 2,304 CGB), and its
  entry gives the power-on template and which offsets of bank 0 the boot ROM covers.
- **The battery save (storage).** What the host knows, and where from:
  - *Which bytes.* The save is `save_len` bytes of the model's own memory image from
    `save_base` (`steps.rs` `take_save`, `image.region`). `save_base` is where the memory
    model keeps cartridge RAM's cells (`layout::CRAM`, written into the wiring by the
    compiler: the same layout the host places records by). `save_len` is 0 without a
    battery, else the RAM's length: the board table's entry for the header's cartridge
    type and RAM size (§2 "The board", the load-time fact that also sizes the memory
    model's RAM). The bytes are what the model's records left there; the host neither
    copies nor filters them, and applies no mapper rule.
  - *When.* `transport::Battery` writes the file at the step whose emitted A000 window
    (`MAPS`, the model's own translation of RAM enable and bank) stops landing in the saved
    regions after a write landed in them. The regions (`save_regions`: cartridge RAM's 128
    KiB, the clock's port block) are each a power-of-two block aligned to its length
    (`the_saved_regions_are_fields_of_an_emitted_address`), so the test reads the high bits
    of an address the model emitted, as a port's dispatch does; it uses no cartridge type
    and no mapper rule. It decides only when the file is written: battery RAM keeps its
    bytes on the hardware whatever the enable does, no GBC state reads the tracker, and
    writing the same `save_len` bytes every frame, or only at exit, would store the bytes
    the model holds at that moment with the machine unchanged. Storing saves is the host's
    job, class (b) below.
- **Checks.** The transport checks what a pass emits (a run moves one byte a cycle, its
  last in the span's last cycle; a block is aligned to its length; a line drawn shares its
  block with nothing that reaches what it reads). The checks compute nothing.
- **The footer.** The save footer's byte layout is a file format, routed bit by bit.

### What the transport still decides

Every rule in `src/transport.rs`, with one of three
classes. **(a)** Pure transport: counting down a number the model emitted, concatenating
bits it emitted, or moving an output to an input; a guard (an `assert`) that only checks
the model is also (a). **(b)** Presentation or storage. **(c)** A GBC rule the host still
decides. The class is strict: where the host picks an order or a tie-break, the row is (c)
even when the model emits the numbers around it. A review on 2026-11-11 found 13 rows of
class (c); each is now (a) or (b), and the last column says what moved
(`docs/control-in-weights.md` §24).

| What the transport does | Where | Class |
| - | - | - |
| A block copy's byte k moves from `source \| k` to `dest \| k` | `src/transport.rs:26` | (a) |
| A run's bytes, one per bit the pass set, at `page:line \| j` to `dest:line \| j` (or a fill) | `src/transport.rs:60` | (a) |
| A run's bytes before the span's last cycle land before that cycle's line is drawn; the last cycle's byte lands on the side the pass's own bit names (`DX` bit 0) | `src/transport.rs:66` | (a), was (c): the tie is the model's bit (constant 0, after the line). The reference splits its own bytes at each block's last M-cycle, and the lockstep's byte stream carries each line drawn among them. |
| A run moves one byte a cycle, its last in the span's last | `src/transport.rs:70` | (a) guard |
| A pass after a scheduled start starts at the phase and line the model emitted for it | `src/transport.rs:120`, `:200`, `:210` | (a) |
| The locks hold while the cycle is before the `until` the model emitted | `src/transport.rs:129` | (a) |
| The bus state ahead: the held locks, else a look-ahead pass's, and the busy bit from `DZ` to `DY` | `src/transport.rs:136` | (a) |
| Horizon, scheduled starts and edges counted down from the model's numbers | `src/transport.rs:179`-`:256` | (a) |
| A dropped bit's edge falls at the distance the pass emitted | `src/transport.rs:229` | (a), was (c): `FE1` bit 4 is read as a distance field (one bit wide), not a flag the host turns into one cycle. The weights did not move for this row. |
| After a schedule's last start, the next pass falls at the tail the pass emitted (`RT`) | `src/transport.rs:252` | (a), was (c): a new output, not the period reused. |
| Cycles run with the steps stream stopped count down the stop the last pass emitted | `src/transport.rs:238` | (a) |
| A battery save is written at the step whose window stops enabling the saved regions after a write | `src/transport.rs:277` | (b) storage, was (c): see "The battery commit" above. |
| A write is to battery RAM when its address lies in the saved regions | `src/transport.rs:307` | (b) storage, was (c): it feeds only the row above. The regions are the region field of the address the model emitted, the same bits the host places records by. |
| A DMA copy's byte sharing a line with a write record not landed yet stops the run (access-timing's guard, now one function both programs call) | `src/transport.rs:302` | (a) guard |
| The board's high-pass filter over the mixed samples | `src/transport.rs:321` | (b) |
| The segments stream is cut by one `Cutter` both programs run: a segment ends where the sample countdown (`SD`, in halves, counted down by the rate the spans stream's pass emitted until reached or passed, as the frame time is) or the end countdown (`SE`) the last pass emitted reaches 0, or at a read | `src/transport.rs` `SegmentClock`, `Cutter` (line numbers above are as of 2026-11-11) | (a), was (c) twice: both cutters ended the segment one cycle after a counted fall at the fast speed by a rule of their own (now `SE`, 1 only where envelope steps wait for the next cycle), and both ran an extra pass at each change of speed because `SD` counted M-cycles at the last pass's speed (now halves, so no pass is needed). `ApuLockstep` holds every emitted end to the cycle the reference leaves steps waiting after. |
| A cycle's events open a segment, and the cycles of that segment the pass leaves (`DL`) are counted again from the cycle after the part the pass ran, with the countdowns it emitted; the close that ran the pass is then delivered once | `src/transport.rs` `Cutter::close`, `Cutter::events` | (a), was (c): the host ran each event as a pass of its own with no cycles, deciding that the events precede their cycle's cycles. The pass now takes the event with the cycles after it and runs the write and the event only (`DL` = D, every cycle-counting node and the sample clock reading D and the sample flag as 0), so that order is the model's. The pass count is unchanged. A write and a change of speed also open a segment: the encoding has one write, one event set and one speed bit per pass. |
| A cycle's events fall at the cycle the spans stream's pass emitted for them (`FE`, `FR`, the drop), the last cycle of the block that counts down to it | `src/transport.rs` `Spans::advance`, `control_circuit` `Outcome` | (a): the distances are model outputs, the host only counts to them, and `ControlLockstep`'s mutants (an edge one M-cycle late or early, a drop lost) are caught. |
| A block's signals land in one order: early copies, the line drawn, the stop, the frame, the blank, the own byte, the held copies, the copies | `src/transport.rs:400` | (a) guarded, was (c): only a line and the bytes it reads share state, and the passes order those (the row on the run's last byte). The rest reach disjoint state (the lines stream's own byte, the stop's hold, the frame shown, the screen blanked, memory), argued from the sinks, not tested; the guard stops the run when a line shares its block with any of them, or a frame with a blank. |
| After a step: the held copies land, a STOP is its pass, then the steps stream stays stopped for the total the passes emitted, counted down in blocks to 0 | `src/transport.rs:453` | (a), was (c): the total is the model's (`SQ` in, `SS` out), so it decides how stops add up; a pause's cycles are counted by the model's flag. |
| A pass's block goes to the step's end when its held bit is set; a stall has no route | `src/transport.rs:479` | (a), was (c): every stop runs after the step, so only the block has a held bit. |
| A further pass runs at once at each event code the last pass emitted (`NE`) | `src/transport.rs:554` | (a), was (c): an output moved into the next pass's input, not a loop on a state bit. |
| The renderer's line is the state's `LY`; a write's own byte is the index the pass emitted with the written value | `src/transport.rs:563`, `:572` | (a) |
| A pass that asks the bus for nothing hands its run's bytes to its caller, which lands them at once; it asks for no copy, render or further pass and leaves the stop owed | `src/transport.rs:578` | (a), was (c): no bytes wait for a later call. |
| A blocked access cuts the run with a pass | `src/transport.rs:593` | (a), was (c): always a pass, at any cycle. |
| A buttons pass's, and STOP's, outcome is the change in the state's held bit | `src/transport.rs:601`, `:609` | (a) |
| At a scheduled start the renderer draws the line the model named; at the horizon a pass; at an edge the APU steps | `src/transport.rs:622` | (a) |
| The other streams run to each access's cycle in time order, each predicted state held to the host's | `src/transport.rs:691` | (a) guard |
| The accesses' states settle over at most two asks; reads before the copy's end are served at their own cycles; a copy a write starts blocks from the cycle the model emitted (`DZ`) | `src/transport.rs:713` | (a), the last part was (c) at `:717`: the access-timing work replaced "write 0 starting the copy blocks write 1" with `DZ`; `the_lockstep_catches_a_fresh_oam_dma_blocking_a_cycle_off` is its mutant. |
| Each access's mask is `DMA_BLOCKING \| (flag ? locks : 0)`: the busy bit always, the locks where the request's own flag sets them (`RC` bit 3 for the data accesses, `WC` bit 3 for the fetched bytes) | `src/transport.rs:729` (`access_mask`, both programs) | (a): accepted as transport. The host applies the busy bit unconditionally; which bus OAM DMA blocks stays the model's, through `DB`. A later change gave the fetched bytes the locks (they had the busy bit only). |
| A step's opening: the held locks, else a pass for them where they lapsed and the first fetched byte's mask reads them, and the step asks again; a request that asks for the locks (`RQ` bit 7) gets them from a pass; the start's state is the held locks, else the asked, and the busy bit | `src/transport.rs:735`, `:757` (`start_state`, `open_step`, both programs) | (a): each pass is asked for by a flag the step emitted. One function since that change; `a_step_asks_for_the_lapsed_locks_only_where_its_fetch_reads_them` fails with the lapsed ask removed. |
| A rest (the steps stream halted) is a pass only while the state's `idle` flag is clear (HDMA5 bit 7: an HBlank transfer runs), its wake only while `kept` is set (`FL` bit 3), and a step's end only while `uncounted` is set (HDMA3 bit 5: an HBlank block waits to be counted) | `tools/runner/src/spans.rs` `rest`, `woke`, `step_ended`; `ControlModel::halt`, `woke`, `step_running` | (a) inference management, measured: with the flags ignored, so every rest and wake ran its pass, the runner proofs on Yellow from power-on to 300, Yellow 300 on DMG, Zelda DX 300 and Mario DX 300 (both double speed) kept every frame and every sample of the skipping run and every other stream's record. Only the spans stream's record grew, by the extra passes (68, 481, 687 and 571 record lines); no extra pass landed a copy, raised a signal or flipped either flag. The skipped passes are passes the model answers with no change. The step's end is gated the same way, measured in an earlier round with a runner that runs every step's end as a pass (`if true` for the `uncounted` flag), against the full program's records: `hdma_timing-C` 60, Yellow 300, Zelda DX 300 and Mario DX 300 kept every frame and every sample, and the lines, segments and length records. The spans record grew (221,821 to 1,061,737 lines on `hdma_timing-C`, 71,760 to 1,268,980 on Yellow, 131,949 to 5,998,569 on Zelda DX, 68,564 to 640,813 on Mario DX), and the steps record changed with it (Yellow 995,267 to 1,017,840 lines, Mario DX 641,382 to 664,074; on `hdma_timing-C` and Zelda DX the same count, other content), as did Zelda DX's stepper record. Inferred, not traced: each extra pass renews the access locks and the rate the spans stream holds, which the steps stream composes into its inputs and the stepper reads at a row's trigger, so those streams took other passes to the same results. The extra passes changed no frame, sample or device state the outputs show; the runner proof on `hdma_timing-C` requires the gated passes to run, as many as the full program's. |
| The wall time: one elapse pass for each binary digit of the seconds (`ControlModel::elapse`, the runner's `set_time`) | outside `src/transport.rs` | (a), was (c) at `:450`: the host split the seconds into days, hours and minutes; the clock circuit now splits each digit itself. |

Class (c) is empty. `tools/runner/tests/quarantine.rs` holds it so mechanically, in three
ways:

- The term, number and slot scanners run over `src/transport.rs` as over the runner.
- **A ratchet.** `every_branch_is_reviewed` lists every line of the runner and the transport
  that branches or picks (`if`, `match`, a loop, a filter, a minimum, `then`, `unwrap_or`,
  `contains`, `any`, `all`, `position`, `=>`), that stops on a condition (`assert!`,
  `assert_eq!`, `assert_ne!`, their `debug_` forms, `panic!`, `unreachable!`) or that
  joins conditions (`&&`, and `||` after an operand, not a closure's empty parameter list),
  506 of them (450 class a, 13 class b, 43 guards), each with the class a
  review gave it, in `tests/branches.txt`. A new or changed line fails until it is listed with
  class (a), (b) or a guard; `the_scanners_catch_a_rule` holds the scanner to the HALT-bug
  refusal it once missed (an `assert!` with `||`). What it sees is text, line by line: a
  decision written without those words (an index or a shift computed from a flag, a table
  lookup, `?` on an error) is not listed, two identical lines share one entry, and a listed
  line's class is a review's judgement. It proves that no line with those words was added or
  changed without a review, not that the runner holds no rule.
- **A property test.** `tools/runner/tests/transport.rs` drives the transport with a fake
  stream whose passes emit values no model of today emits: a drop five cycles on, a tail
  unlike the period, further passes at named events, a stop total that shrinks, a run's last
  byte before its line. A transport that kept any of the old rules fails it. It proves the
  transport follows those numbers; it says nothing of a rule written elsewhere.

## 3. Every path, and the refusals

The runner refuses nothing the full program runs, and refuses what it refuses. A step
that reads its opcode's address twice across a lock edge (the GBC CPU's HALT bug,
`docs/hardware-tests.md` §3) runs on both: the model fetches its operand 1 under that
read's own bus state, and `tools/gate.sh` holds the runner to the full program on the
HALT-bug program (`examples/halt_bug_rom.rs`). Paths the five game runs never reach are
held to the full program on test ROMs (§4): a fetched byte the host serves (an opcode
fetched from a control register and from IE through the HALT bug, `examples/host_fetch_rom.rs`),
an operand fetched under a lock the opcode fetch did not see, with the host's copy of the
locks lapsed (`examples/fetch_locks_rom.rs`),
an access OAM DMA blocks (mooneye's `oam_dma` ROMs), a read of a palette data port through
the index the pass names (gambatte's `cgbpal_m3`), a run of OAM DMA bytes with no source
(mooneye's `sources-GS` on CGB hardware) and a STOP that holds the PPU (daid's
`stop_instr`). Two cases stop both programs alike: a DMA copy whose source line is a port,
whose bytes no line pass places, and a cartridge header the board table lacks (§2).

**The bus at PC.** A core whose memory is the model's reads no host byte at PC
(`StepFacts::reads_the_bus`); the full program peeks those bytes, and a reference's data bytes
under each read's own bus state, only for a reference beside it (lockstep, the audit),
without a device sync or a cut. Bytes the host serves through the fetch ports replace the
peeked ones before the execute and the audit. So a run with a reference and one without
take the same passes (§4, the lockstep check).

## 4. Results (measured)

**On 2026-11-13**, the one model exported with `--layer-shapes --align 64`
(1,093.4 M parameters stored). Each run passed with all six records equal on both sides,
every event (no window), frames, audio and saves equal, 0 dense fallbacks. The counters
column is what each side printed, equal where the proof requires it (`EQUAL`):

| Run | Frame | Record lines (steps, lines, segments, spans, length, stepper) | Counters, full = runner |
| - | - | - | - |
| Yellow 300 | `cfd31f3d` | 995,267 / 83,499 / 251,675 / 71,760 / 41,856 / 1 | none fire |
| Zelda DX 300 | `f10b2b4a` | 4,480,602 / 81,847 / 254,128 / 131,949 / 43,389 / 1,222,014 | `revised_before` 1,077 |
| Mario DX 300 | `ea7f86ca` | 641,382 / 82,437 / 253,589 / 68,564 / 41,220 / 1 | none fire |
| Mealybug `hdma_timing-C`, 60 | `d043ab24` | 814,793 / 14,113 / 50,451 / 221,821 / 7,057 / 1 | rest, wake and step-end passes 10 |
| Mealybug `m3_bgp_change`, 4 | `c66c92d4` | 50,295 / 146 / 3,345 / 5,902 / 3,599 / 2,578,201 | `revised_before` 432, `taken_after` 428 |
| Mealybug `m3_scx_low_3_bits`, 4 | `c8c688b4` | 52,936 / 146 / 3,345 / 5,567 / 1,278 / 641,022 | `revised_before` 378, `taken_after` 375 |
| Mealybug `m3_lcdc_obj_size_change`, 4 | `f5c77755` | 47,248 / 146 / 3,345 / 5,716 / 1,371 / 885,521 | `revised_before` 245 |
| dmg-acid2 on CGB from Nintendo's boot ROM, 120 | `afa02e33` | 1,064,051 / 31,913 / 100,577 / 474,893 / 15,958 / 1 | none fire (the boot ROM writes FF50 with no row open) |

`revised_before` counts lines whose length the stepper revised before the spans stream read
it, `taken_after` lines a write it took reached after that read, `forwarded` display reads
the record queue served while a row was open, `compat_closes` rows a COMPAT record closed.
The last two need directed programs, which `tools/gate.sh` runs (`examples/display_read_rom.rs`,
`examples/compat_boot_rom.rs`, below). The HALT-bug program stops both programs with the same
refusal (`refused: the step reads its opcode's address twice ...`).

**Plain against lockstep, every event.** `tools/runner/same-passes.sh` runs the full program
as one model plain and under `--lockstep all`, every stream recorded uncapped, and compares
the records with the memory core's audit-only slots (`OP0' OP1' OP2' RV0 RV1`) left out of
both, requiring that the lockstep side decoded some and the plain side none. mooneye
`oam_dma_restart` 30 (239,595 / 3,247 / 25,076 / 21,841 / 3,248 / 1 lines, 1,144,475 audit
positions left out) and Mealybug `m3_bgp_change` 4 (50,295 / 146 / 3,345 / 5,902 / 3,599 /
2,578,201 lines, 246,255 left out): every other token, position and memory head read equal.

**On 2026-11-12**, with the change that gives the operand fetches the access locks and
moves both programs' lock asks into `transport::open_step`. The GGUF is the one model
exported as below (one-model digest `b2a06e13`, 1,391.0 M dense parameters, `digest --gguf`
0 values differing). Each run passed, all five pass records equal, frames and audio equal,
0 dense fallbacks:

| Run | Frame | Record lines (steps, lines, segments, spans, length) |
| - | - | - |
| Yellow 300 | `cfd31f3d` | 995,267 / 83,501 / 251,675 / 71,509 / 41,751 |
| mooneye `oam_dma_restart`, 120 | `7b7dbe09` | 766,275 / 16,208 / 100,298 / 35,970 / 16,208 |
| fetch-locks (`examples/fetch_locks_rom.rs`), DMG, 24 | `2817dca1` | 233,129 / 3,457 / 20,061 / 62,642 / 3,457 |
| fetch-locks, CGB, 24 | `d043ab24` | 233,129 / 6,913 / 20,060 / 62,642 / 3,457 |

The fetch-locks ROM is `tests/fetch_locks.rs`'s sweep: the PPU's lock falls between a call's
opcode and its operand, in VRAM and in OAM. It reaches the step's lapsed-lock ask in the
runner. With the runner's `Opening::fetch_reads_locks` made to return false, the proof
fails on both consoles. The steps record has 232,873 lines against 233,129 and the spans
record 62,758 against 62,642, first differing at lines 91 and 8. The frames stay equal,
since the program keeps each call's A in WRAM. With the lapsed branch removed from
`transport::open_step` itself, both programs change alike, and `tests/fetch_locks.rs` is
the check: both consoles fail at frame 0.

On DMG with the OAM corruption bug in weights (`docs/hardware-tests.md` §4.4) and a DMG's read
and write locks (§3.17), each from the one model's GGUF, six streams recorded on both sides,
every pass record, frame and sample equal, 0 dense fallbacks (`REWRITES=1` also holds the
full program's `oam_bugs` to the runner's `rewrites`):

| Run | Built at | Frame | Record lines (six streams) | OAM bug passes |
| - | - | - | - | - |
| blargg `oam_bug/rom_singles/2-causes`, DMG, 120 | 2026-11-12 | `85f58109` | 886,229 / 12,952 / 100,304 / 57,416 / 12,956 / 848 | 134 |
| fetch-locks, DMG, 24 (the routine at FE01, written again after each call) | 2026-11-12 | `2817dca1` | 232,430 / 3,457 / 20,061 / 62,810 / 3,457 / 1 | 88 |
| mooneye `ppu/lcdon_timing-GS`, DMG, 60 | 2026-11-13 | `002271da` | 482,479 / 6,893 / 50,149 / 60,858 / 6,894 / 1 | 0 |
| mooneye `ppu/lcdon_write_timing-GS`, DMG, 60 | 2026-11-13 | `002271da` | 563,191 / 6,527 / 50,149 / 128,362 / 6,528 / 1 | 0 |

`tools/gate.sh` runs the first two; blargg's ROMs claim CGB support, so `auto` would pick
CGB, where the bug never fires, and the proofs pass `--hardware dmg`. The runner refused
nothing: the OAM pass runs from the wiring table (`wiring::Steps::events`).

**On 2026-11-11**, with the transport rules above. The GGUF is the one model exported as below (sha1 `5a819cec`, 977.3 M
dense parameters, one-model digest `5c21d371`, `digest --gguf` equal to the compiled).
Every case below passed, the runner's five pass records equal to the full program's, frames,
audio and saves equal, 0 dense fallbacks (record lines: steps, lines, segments, spans,
length):

| Run | Frame | Record lines |
| - | - | - |
| Yellow 300 (and on one thread) | `cfd31f3d` | 994,981 / 83,501 / 243,959 / 71,509 / 41,751 |
| Yellow 300 from the boot ROM, power-on (and on one thread) | `0db60d9b` | 2,595,086 / 81,113 / 243,784 / 850,691 / 40,557 |
| Crystal 300 (and on one thread) | `ccfe7ad3` | 1,323,481 / 83,555 / 244,313 / 228,099 / 41,778 |
| Zelda DX 300 (and on one thread) | `f10b2b4a` | 4,480,320 / 84,003 / 244,520 / 130,284 / 42,002 |
| Mario DX 300 (and on one thread) | `ea7f86ca` | 641,043 / 82,439 / 243,977 / 68,344 / 41,220 |
| Yellow, text snapshot, frames 2900-3100 | `7cb69988` | 1,201,459 / 57,601 / 163,156 / 27,259 / 28,801 |
| Crystal from frame 200, overshoot 4 | `b85417d0` | 1,048,052 / 57,601 / 162,904 / 174,026 / 28,801 |
| host-fetch, both `cgbpal_m3`, seven mooneye OAM DMA cases, three daid STOP cases | as below | equal |

The power-on state the runner builds equals the full program's for all four games, and the
runner's tests pass. With every reference beside it (`--lockstep all`), Yellow, Crystal,
Zelda DX, Mario DX and Yellow from the boot ROM at 300 frames, both STOP ROMs, `instr_timing`
(`cpu,control`) and both benches pass, frames as above. Lockstep against plain on
`oam_dma_restart`: the steps, segments, spans and length records are equal, the steps
record's audit-only positions (now 332, 333, 334, 341 and 342) stripped. The lines record
under lockstep is the plain run's less its last line (16,207 of 16,208, the rest equal byte
for byte), and the build before the transport rules shows the same, so it predates them; it is open.
In llama.cpp b12070 on Metal (whether that build was patched, and its BLAS option, are not recorded here), Yellow ran 4,000 passes with this GGUF against the engine's
record: 4,001 events equal, 0 differing, 305,528 decoded tokens compared.

The rows below are an earlier run of the same day.

The full program ran with every component in weights as one model, from the GGUF alone
(`GBCFORMER_GGUF_ONLY`), with no lockstep, its APU and PPU on the machine's thread and the
scripted clock `GBCFORMER_TIME=1700000000`, keeping its snapshot, pass record and outputs;
the runner ran from that snapshot (or from power-on) recording the same way
(`tools/runner/proof.sh`). The GGUF is the one model exported with `--layer-shapes --align
64` (sha1 `fa47368f`, 911.5 M dense parameters, one-model digest
`dc902b72`), from a build that merges the runner with the access-timing and host-rules
work and moves the pass wrappers and the look-ahead into
`transport`; every row below ran on both sides built from it. With control running
shared code on both sides, its reference beside it is the check on that code: on that build
Crystal, Zelda DX and Mario DX 300 with every component in weights as one model and
`--lockstep all` reach `ccfe7ad3`, `f10b2b4a` and `ea7f86ca`, daid `stop_instr` (DMG and
CGB) `6872b003` and `c9a38612`, and blargg `instr_timing` with `cpu,control` passes, none
diverging.
Every case records all five streams that build had (the stepper, the sixth, came later) on both sides, and every pass of every stream ran on
the engine's fast path, 0 dense fallbacks.

| Run | Frame | Record lines (steps, lines, segments, spans, length) | Records | Frames, audio | Saves |
| - | - | - | - | - | - |
| Yellow 300 | `cfd31f3d` | 995,170 / 83,501 / 243,959 / 71,312 / 41,751 | equal | equal | none |
| Yellow 300 from the boot ROM, power-on | `0db60d9b` | 2,595,139 / 81,113 / 243,784 / 850,630 / 40,557 | equal | equal | none |
| Crystal 300 (MBC3 clock) | `ccfe7ad3` | 1,323,324 / 83,555 / 244,313 / 227,811 / 41,778 | equal | equal | 3, equal |
| Zelda DX 300 | `f10b2b4a` | 4,480,503 / 84,003 / 244,520 / 129,728 / 42,002 | equal | equal | 1, equal |
| Mario DX 300 | `ea7f86ca` | 640,786 / 82,439 / 243,977 / 68,059 / 41,220 | equal | equal | 4, equal |
| Yellow, text bench snapshot, frames 2900-3100 | `7cb69988` | 1,201,550 / 57,601 / 163,156 / 27,091 / 28,801 | equal | equal | 3, equal |
| Crystal from frame 200, overshoot 4 | `b85417d0` | 1,048,037 / 57,601 / 162,904 / 173,824 / 28,801 | equal | equal | none |

Each game ran twice, the runner on one thread and with the segments and lines streams on
their own, both equal. The test ROMs, each from the full program's frame-0 snapshot:

| ROM | Frames | Frame | Steps record lines |
| - | - | - | - |
| host-fetch (`examples/host_fetch_rom.rs`, since removed; the program is `tests/host_fetch.rs`'s) | 10 | `6872b003` | 58,536 |
| gambatte `cgbpal_m3end_1_cgb04c_out7` | 60 | `a505c00c` | 357,085 |
| gambatte `cgbpal_m3end_ds_2_cgb04c_out0` | 60 | `a505c00c` | 703,704 |
| mooneye `oam_dma_restart` | 120 | `218e2822` | 766,318 |
| mooneye `oam_dma_start` | 120 | `9c72e6c6` | 770,671 |
| mooneye `oam_dma_timing` | 120 | `218e2822` | 766,307 |
| mooneye `oam_dma/basic` | 120 | `002271da` | 758,045 |
| mooneye `oam_dma/reg_read` | 120 | `1b63984f` | 757,001 |
| mooneye `oam_dma/sources-GS`, DMG | 120 | `002271da` | 902,098 |
| mooneye `oam_dma/sources-GS`, CGB | 120 | `0807c385` | 899,581 |
| daid `stop_instr`, DMG | 120 | `6872b003` | 26,956 |
| daid `stop_instr`, CGB | 120 | `c9a38612` | 26,956 |
| daid `stop_instr_gbc_mode3`, CGB | 120 | `9582f278` | 27,476 |

All five streams' records, frames and audio equal on every row. The power-on state the
runner builds from the ROM and boot ROM alone equals the full program's power-on snapshot
for all four games. The full program with every reference beside it (`--lockstep all`) and
without took the same passes on `oam_dma_restart` (120 frames, `218e2822` both): all five
streams equal, 766,318 / 16,206 / 97,469 / 35,959 / 16,206 record lines, once the steps
record's five positions that only the audit decodes (`sm83::AUDIT_SLOTS`, positions 336,
337, 338, 345 and 346) are stripped from both sides.

**Speed.** Both programs run their plain builds (`cargo build --release`) with their own
threads. The runner starts from the bench's hand-off snapshot (frame 2,900 for text, 7,200
for overworld) with the bench's presses, and the full program runs every component in weights
as one model from the same GGUF (`GBCFORMER_GGUF_ONLY`). Each round runs them interleaved
through the heavy queue (base runner, tip runner, full program, then the reverse), three
rounds, so six runs a side. Every run reached the bench frame (`870aa3fd`, `822059c1`) with 0
dense fallbacks.

| Workload, 600 frames | Full program (fps) | Runner, step buffers kept (fps) | Runner before (fps) |
| - | - | - | - |
| text | 63.55 (63.3 to 63.9) | 63.2 (62.1 to 63.8) | 59.3 (58.8 to 59.7) |
| overworld | 131.3 (130.9 to 133.8) | 131.95 (131.0 to 133.4) | 128.05 (126.8 to 130.5) |

Medians of six with ranges, 2026-11-17, load 2.1 to 4.4. The full program compiling its own
weights read 63.1 and 63.2 on text and 131.5 and 131.6 on overworld in the same ticket. The
runner is over 60 fps on text in every run, and it is not told apart from the full program:
the ranges overlap on both workloads. The runner with its step buffers kept passes each stream's tokens in a
`sequence::Sequence`, defers the memory cells' K/V rows to a worker thread and keeps a step's
buffers, which the runner before did not (`docs/perf-log.md`, "Round 6b", has each
change's timing and the proofs run on it).

The earlier timing, the runner of 2026-11-11 against the full program of 2026-11-10 (one checkpoint,
`dc902b72`), load 24 to 55, medians of three: text 48.9 against 49.3, overworld 109.9 against
112.4. At four streams the runner ran 37.4 against 50.0 and 61.0 against 100.9.

Where the gap went, measured with `sample` and by leaving one change out at a time
(the runner of 2026-11-10, load 35 to 88, medians of three):

- The runner's threads ran at the default QoS class while the full program's ask for
  user-interactive (`qos::interactive`). On a loaded machine macOS ran the runner's main
  thread, which is busy for its whole run (it waits under 1 % of samples), behind other
  work: without the class, text fell to 12.2 fps and overworld to 29.1 (44.7 and 96.9
  with it).
- Each frame the runner waited for the segments stream to hand back its samples; the
  samples now come back as each batch ends and the frame does not wait. Without it,
  overworld fell to 72.8 (text 43.7, within noise).
- Calls to the lines stream's thread went one per wake; they go eight at a time now, the
  segments stream's batches at once. Without the batching, overworld fell to 84.3.
- The main thread spent 15 % of its samples outside the engine against the full program's
  10 %: a slot name searched and cloned every step, a dozen vectors allocated per step, the
  request cloned at every reach to an access's cycle, pass lists rebuilt per pass. They are
  reused or resolved once now (text 44.1 to 46.5 in the next bench, medians); the rest of the time
  is the same engine passes, by the pass records.

## 5. Reproducing

```sh
cargo run --release --example export -- one-model DIR --gguf-only --layer-shapes --align 64   # the GGUF with its wiring table
cargo build --release --features stock
(cd tools/runner && cargo build --release --features record)
tools/runner/proof.sh DIR/model.gguf OUT pokemon-yellow.gbc 300                     # the frame-0 hand-off
tools/runner/proof.sh DIR/model.gguf OUT pokemon-yellow.gbc 3100 --warmup 2900 ...  # a snapshot mid-run (presses as tools/bench/text.sh)
tools/runner/proof.sh DIR/model.gguf OUT pokemon-yellow.gbc 300 --boot cgb_boot.bin # power-on, no snapshot
cargo run --release --example fetch_locks_rom -- F.gb; tools/runner/proof.sh DIR/model.gguf OUT F.gb 24 --hardware cgb   # operand fetches under the locks
THREADS=0 tools/runner/proof.sh ...                                                  # the runner on one thread
PLAIN=1 REUSE=1 RUNNER=<plain build with debug assertions> tools/runner/proof.sh ...   # no pass record: outputs alone, the change-list placement checked at every pass
SCAN=1 REUSE=1 tools/runner/proof.sh ...                                           # every cell lookup also read through the stored K/V rows
(cd tools/runner && GBCFORMER_GGUF=DIR/model.gguf cargo test --release)              # dependencies and quarantine
```

`proof.sh` runs the full program from the GGUF alone with every component in weights as
one model, its APU and PPU on the machine's thread and the scripted clock
`GBCFORMER_TIME=1700000000`, keeping its snapshot, pass record and outputs; then the runner
from that snapshot (or from power-on), recording the same way. It requires a record from
at least five of the six streams on both sides, as many on each (the stepper records only where a line is marked) and compares them by sha1 and line count (each
stream writes into a FIFO that `tools/runner/digest.ts` reduces to its sha1 and a CRC-32
per line as it runs, so no raw record is kept; when two differ, the CRC lists name the
first differing record line), then
frames, audio and saves byte for byte, and fails on any dense fallback, panic or refusal.

Two modes hold what a recording proof does not (`docs/perf-log.md`, "Round 6c", has the runs).
`REUSE=1` keeps the full program's side from the recording proof in the same directory.

- `PLAIN=1` runs the runner with no `GBCFORMER_RECORD` and compares frames, audio and saves
  alone. A recording run places the whole sequence after every write record, and a plain run
  places it by its change list. Build the runner for it without the `record` feature and with
  debug assertions, under which the engine holds every pass's change list and decoded store
  to the whole sequence:
  `(cd tools/runner && CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true CARGO_TARGET_DIR=DBG cargo build --release)`.
- `SCAN=1` sets `GBCFORMER_SCAN=1` for the runner, so every lookup of a memory cell is also
  read as the softmax over the stored K/V rows, which with `--threads 1` a worker thread
  writes. The proof requires the steps stream's count of checked lookups, which the runner
  prints, to be above 0 and equal to its lookups. On Yellow's cache a frame takes about a
  minute, so use a small cartridge (`m3_bgp_change`, about three minutes).
