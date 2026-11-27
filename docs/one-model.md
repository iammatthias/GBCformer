# One model: the whole Game Boy as one Llama checkpoint

**Now.** One checkpoint of six parts (the PPU's line renderer, the GBC CPU
with its memory, the APU, the control logic, the PPU's mode-3 length and its pixel-FIFO
stepper), each a stream over the shared weights. The canonical artifact is its
per-layer-shaped GGUF (`export one-model --gguf-only --layer-shapes --align 64`): Meta's
Llama layer set with per-layer shapes, each layer storing only the heads and FFN units it
uses, and the file leaving out the rows and columns of the LM head, the embedding and each
`ffn_up` that are constant or zero (`docs/harness.md` §2). It runs in llama.cpp b12070 with
three patches (`tools/stock-llama/layer-shapes.patch`, `stream-slices.patch` and `stored-runs.patch`).
It stores 382.0 M parameters (382,008,896: 381,910,976 in its weight matrices, which `digest --gguf` counts, and 97,920 in its 51 norms of 1,920) in 769,593,632 bytes, 1.39 M of them nonzero (digest `ff86dde7`, since the parts share the input slots: `docs/perf-log.md`, "Round 13").
The current weights ran Yellow from the hand-off to frame 300 in unpatched llama.cpp on
2026-11-25 (`docs/perf-log.md`, "The proof of record on `ff86dde7`"). Older llama.cpp results
on this page and in the history sections of `docs/stock-llama.md` are earlier models'.
Padded to one uniform Llama checkpoint (hidden 1,920, intermediate 3,456, 25 layers, 26
heads, vocabulary 27,511) it is 769.3 M dense (1,544,154,592 bytes; before the parts shared
the input slots it was 1,008.5 M at 41 heads and 65,804 tokens, and before the PPU's parts
were re-levelled, `docs/perf-log.md` "Re-levelling", 2,518.2 M at intermediate
10,688 and 69 heads): that export runs in unpatched llama.cpp b12070 on Metal
and is the claim of record (`docs/stock-llama.md`, "What runs where"; Yellow from the hand-off to frame 300 in llama.cpp's default build, every pass equal to the engine's
record: `docs/perf-log.md`, "The proof of record on `ff86dde7`"), and `digest --gguf` expands the shaped file tensor by tensor against
the compiled checkpoint. The file's manifest (`gbcformer.manifest`: the wiring's schema
version and digest, the tensor digest, the vocabulary and the llama.cpp patches' digests) binds
the metadata to the tensors: the runner and `GBCFORMER_GGUF_ONLY` refuse a file without one, a
file whose wiring is another schema or not the one its manifest names, a field the build does
not know, a snapshot written for another wiring, and a file whose weights, expanded by its
placement keys, do not rebuild to the digest its metadata names (`gguf::open_model`, 12.5 s of
hashing at load). The parameters hold the rules; the
cartridge ROM, the boot ROM and every RAM are the KV cache's cells, and the boards' starting
states and the wiring table are GGUF metadata. The rest of this page is the history of how it
got here, each section labelled with the round it describes; the sizes and counts in §3 and
§6 are those rounds', not today's. §4 and §7 are current.

**History.** Until this round the Game Boy in weights was four checkpoints: the CPU with its memory
(82.6 M dense parameters, the memory as its KV cache), the APU (145.9 M, on its own
thread), the control logic (41.3 M, inline, one pass per event) and the scanline PPU
(514.8 M, on its own thread, 160 pixels at 160 output positions). The release needs
one. This document says what "one Llama model" means here, how the six circuits (the PPU, the GBC CPU with its memory, the APU, the control
logic, the PPU's mode-3 length and its pixel-FIFO stepper) become one checkpoint, how exactness and the dense oracle carry over, how lockstep
works, what it costs in size and speed, and what remains. The prototype is
`--one-model` (`src/one_model.rs`, `Engine::streams`). `examples/one_model`, which printed
the measured figures below, has since been removed.

## 1. What one model means

One model is one set of weights under one configuration (layers, hidden size,
intermediate size, heads, vocabulary) and one forward pass, the HF Llama pass that
`llama.rs` runs as the oracle. It is not one sequence. A serving stack runs many
sequences over the same weights at once, each with its own KV cache, and shares cache
prefixes between sequences (prefix caching). The Game Boy maps onto that directly:

- **The CPU stream** is the sequence that owns the memory. Every CPU step is a group
  at the cache's next position, its write records are kept tokens, and the memory is
  the KV cache, as in `docs/memory-in-weights.md`.
- **The PPU stream** is a branch of that sequence at the position `P_LY` where a line's
  mode 3 starts. "VRAM as of the scanline" is the cache prefix up to `P_LY`
  (`docs/ppu-in-weights.md` §5). The engine follows the prefix through the records the
  CPU stream hands over, as it does today.
- **The APU stream** and **the control stream** are sequences of their own. They read
  no memory, so the cache they could see does not change their tokens (§4).

The parts run on different clocks. A CPU pass is one step of 1 to 6 M-cycles, an APU
pass is a segment of up to 22 M-cycles that runs through HALT, a control pass is one
event, and a PPU pass is one line. One shared sequence would interleave those passes in one order,
put every part's slots in every group, and serialize the threads the APU and the PPU
run on now (`docs/apu-in-weights.md` §11.6, point 3). Separate sequences keep each
part's quantum, its thread, and its incremental engine state, and they are standard
serving. A thread per stream is concurrency across sequences, not a change to any
sequence's tokens. Each stream's tokens are exactly what the dense pass gives for that
stream's sequence.

**How a stream selects its circuit.** A stream selects its circuit by its tokens. Every part keeps its own input
tokens, its own output slots and its own slot identities, so a stream's sequence holds
only that part's live tokens. No separate mode token is needed. This is the role the
REC and RD tokens play for the memory. The tokens a group holds decide what the same
weights compute.

## 2. The checkpoint: concatenated circuits

`Circuit::concat` puts the six circuits side by side (`one_model::Part`; the PPU's mode-3 length joined on 2026-11-10, the stepper after it). It renumbers every wire, node and
memory head so no two parts share a wire, puts inputs and outputs in part order, and
prefixes slot names (`ppu:`, `cpu:`, `apu:`, `ctl:`). The PPU comes first,
because its position wires name output positions from the first output slot. Its six
write-record slots, which a scanline leaves zero, are left out, so the CPU's record
slots are the checkpoint's. `Circuit::schedule` then spreads the merged nodes over the
12 levels, and the stock `Model::compile_mem` compiles the result. No compiler change
was needed.

What the parts share in the compiled checkpoint, because one compile lays them out:
the constant channels, the eight own-byte channels, the 8 + 4 route channels and the
LM head, the 256 output byte tokens, the memory block (the gathered address, the kind
channels, the copy head, the family tokens, REC and RD) and therefore one KV cache
format that every memory head reads, the CPU's and the PPU's alike. Input tokens, slot identities
and the wires stay per part.

**One identity channel per gather group.** The input slots of one layer-0 gather group
share one identity channel (`Layout::identities`), since a key reads an input slot's
identity only for its group's code. Each output slot kept its own until round 9; the parts
now share them (§4, "The parts share the output identity channels"). Every input and mask
token's key is therefore bit-identical to the key it had with one channel per slot. REC's
key is not: REC sets one identity channel per group instead of one per slot, so its key on
a group's code dim is 32 times the number of groups using that dim, about a quarter of
what it was. At a record position REC must still win each input gather over the output
masks and family tokens, which score 384 on the fallback dim. It wins by at least 128,
above the 104 the exactness argument needs, where the margin was at least four times
larger before. `DenseMemory::push` asserts each record's cell, so a lost margin stops the
run.

**Every stream holds every slot.** A layer-0 head gathers its group's slots by their
identity. In a sequence without those slots it would find no key and average the lanes
of whatever tokens are there. So each stream's sequence is the checkpoint's full
sequence (121 inputs, 304 outputs), with every slot the stream does not own held at a
fixed token, byte 0 at an input and the mask at an output. Every gather then finds exactly
its keys in every stream, and every foreign circuit computes on valid inputs. The
stream writes and decodes only its own slots. The engine's `View` maps the part's own
slot list and token ids onto the checkpoint's and back, so the devices
(`MemCheckpointCore`, `ApuCheckpoint`, `ControlCheckpoint`, `PpuCheckpoint`) run
unchanged against their own ABIs.

**A stock runtime's pass needs only the stream's own slots.** The gathers group each
part's inputs on their own (`Circuit::parts`), and every output mask keys on a fallback
dim that every gather's query reads, 128 below its targets. A gather whose targets a
group leaves out attends to the group's own masks, whose values are zero, and writes 0,
as another part's byte-0 slots do. `one_model::group` derives each stream's group from
the weights and asserts every margin; `docs/stock-llama.md` §12a has the census. The
engine still keeps every slot, and the fallback leaves its record unchanged.

## 3. Size

Measured on the compiled checkpoint (`cargo run --release --example one_model --
--build`) and projected for the overlay (same example, from each part's real channel
plan and `Config::params`; not compiled):

| Checkpoint | Hidden | Intermediate | Layers | Heads | Vocab | Dense parameters |
| - | - | - | - | - | - | - |
| CPU + memory | 1,024 | 1,728 | 10 | 10 | 8,004 | 82.6 M |
| APU | 1,024 | 2,560 | 11 | 17 | 17,026 | 145.9 M |
| control | 1,024 | 640 | 9 | 7 | 7,485 | 41.3 M |
| PPU | 2,048 | 3,968 | 13 | 52 | 5,042 | 514.8 M |
| four checkpoints, summed | | | | | | 784.7 M |
| **one checkpoint, concatenated (built)** | **4,096** | **4,928** | **13** | **52** | **34,219** | **1,422.1 M** (1,057,620 nonzero) |
| overlay, exclusive head channels (projected) | 3,072 | 4,928 | 13 | 52 | 34,219 | 1,066.6 M |
| overlay, null-token heads (projected) | 2,048 | 4,928 | 13 | 52 | 34,219 | 711.0 M |
| overlay, null-token heads, shared slots (projected) | 2,048 | 4,928 | 13 | 52 | 19,691 | 651.5 M |

After the PPU timing and palette round (the PPU's palette heads, two passes per CGB
line and window outputs; the control logic's PPU timing and BCPS/OCPS) the parts are
PPU 552.9 M (14 layers, 410 memory heads), control 44.1 M, APU 98.7 M and CPU with
memory 82.6 M, 778.3 M summed. The one checkpoint is hidden 4,096, intermediate 4,864,
14 layers, 52 heads, vocabulary 34,480, for 1,501.0 M dense parameters and 1,039,106 nonzeros. The
overlay projections move to 1,125.8 M, 750.5 M and 684.7 M. The table above is the
round that built the one checkpoint (`docs/perf-log.md`, "Integration: the PPU's timing
and palettes on the one model").

Layers and heads are the maximum of the parts, not the sum. Every part's layer 0 and
routing layer coincide, and the PPU's 52 heads per layer (its memory fetches) cover the
other parts' layer-0 gathers. Intermediate is close to the widest part too, but only
because of scheduling. Unscheduled, the merged levels sum to 6,956 units at the widest
and the checkpoint is 1,831.0 M. Scheduled, every level holds about 3,900 units (the
44,360 units over 12 levels), and the channel plan's clearing units bring the widest to
4,928. Hidden is the one dimension that stays the sum. It holds 2,780 wire channels live at
once across the four circuits of that round, plus 522 shared channels.

**Why units do not overlay.** An MLP unit is one fixed row of weights; it cannot be
the CPU's node in one stream and the APU's in another. So in any layer the live units
are the sum over the parts placed there, overlay or not, and the intermediate size
comes from scheduling alone. Overlay only changes hidden (and, with shared slots, the
vocabulary). Heads sum the same way. Since the re-levelling (`docs/perf-log.md`, "Re-levelling") the PPU's three parts sit
in the layers the GBC CPU, the APU and the control logic leave (`src/level.rs`,
`one_model::PLACEMENT`), which took the widest layer from 10,688 units and 69 heads to
3,584 and 41 (`docs/perf-log.md`, "Re-levelling"), and with the routing units counted as
written to 3,456 ("Re-levelling, round 2").

**The overlay.** `docs/hardware-shaped.md` §15.2 built it and corrects this design. In
the built form the mode is per position and gates each unit through `up`, and it gates the routing
units too. Wires of parts that never run in the same stream can share residual
channels. One rule makes it exact. Every writer of a channel is either the same in
every mode or contributes exactly 0 outside its own mode.

1. A mode one-hot reaches every position, through a mode token at position 0 of each
   stream and a layer-0 gather of it.
2. Every MLP unit gains a gate term `−BIG·(1 − mode_own)`, with BIG above the row's largest
   pre-activation over its channels' ranges in every mode, so in a foreign mode its
   gate is at most −1 and silu gives exactly 0. Routing units become
   `[field + slot + mode ≥ 3]`.
3. Clearing units are gated the same way. `Segments::new` must then accept a clearing
   unit that reads its own channel and one mode channel.
4. Layer-0 gathers are attention and cannot be gated. Every part's input wire (slot j,
   bit i) must land on the same channel, which the overlay layout asserts.
5. Memory heads are attention too. Either their output channels stay out of every
   other part's pool (exclusive: 1,026 of the PPU's 1,117 planned channels hold a head
   byte at some point, which is why that variant needs hidden 3,072), or each head's
   query gains `BIGNULL·(1 − mode_own)` on a dimension whose key is 1 only on the mode
   token, so in a foreign mode it attends one-hot to the mode token, whose value is 0
   (null-token: hidden 2,048).

The per-stream engine of an overlay is the positional stage's specialisation applied to
the mode channel. It folds `mode_own = 1` into every live unit's bias and drops every
unit whose gate cannot reach 1 (all foreign units). Its hot loop is the same as the
concatenated stream's below. With null-token heads and shared slots the one checkpoint
(651.5 M) is smaller than the four it replaces (784.7 M), because the four each pay for
their own layers and vocabularies.

## 4. Exactness and the dense oracle

**The dense pass is the oracle, per stream.** A stream's pass is the dense forward of
the checkpoint over that stream's sequence, at the read position above every memory
cell, over the machine's whole memory. All streams share one sequence, with one cell per
8-byte line of physical memory (`docs/stock-llama.md` §4). Every activation stays an
exact integer. `rms_norm_eps` dominates every norm, every gather finds its keys, and every memory
head, the foreign parts' included, reads exactly one cell, because every part the
stream does not own computes on its held inputs (byte 0) exactly as on any other
input and names an address that has a cell. `tests/stream_audit.rs` checks that
claim. The merged circuit, evaluated with each part's slots at zero (the APU's also
at random states) over Yellow's memory after its boot, keeps every node at 0 or 1 and
names only addresses with a cell.

**Correction.** Until the f16 round this section said the APU and control streams
could run with no cache and the PPU stream with VRAM, OAM and ROM byte 0 alone, with a
foreign memory head then writing a convex combination that no kept unit reads. That
was false for a stock runtime, which runs every unit. A verification run found the
APU stream decoding wrong tokens on ggml-cpu, Accelerate and Metal, and about 90 k
head-positions per GBC CPU pass with queries off ±S, with the foreign heads' fractions
reaching the outputs through the norm and the routing units. In f16 a fraction is worse
than inexact, because a unit whose inputs leave their declared range can overflow to
inf, and `0·inf` in the down projection makes every channel NaN. Our engine was right only
because it prunes those heads. Every stream now sees the whole memory.

**The engine runs each stream's reach, not the whole checkpoint.** `Engine::streams`
checks the plan once (`Plan::check`, unchanged) and prunes it per stream
(`Plan::prune`). Virtual channels have one writer each, so the reach is a walk. It starts
from the routing units the stream's tokens own (and any routing unit no token gates),
plus the whole positional stage for the PPU stream, and goes through each kept unit's
read channels and each kept memory head's query back to layer 0. The engine leaves
everything else out. The CPU stream evaluates none of the PPU's memory heads (409 since
the palette round), the APU and control streams evaluate no memory head at all, so they
run with no image, and no stream evaluates the layer-0
gathers of another part's slots. Because the view holds every foreign position at one
token, a continuing pass compares only the positions the stream writes when it looks
for moved tokens and for a cached attention plan; comparing all 425 positions cost the
CPU stream about 16 k instructions a step (measured by `sample`, with `layer0_changes` at 664
samples against 226 for the CPU's own checkpoint, 200 after the change).

| Stream | Units | Memory heads | Positional stage | Part's own checkpoint |
| - | - | - | - | - |
| CPU | 9,829 | 17 | no | 9,914 units, 17 heads |
| APU | 7,999 | 0 | no | 8,078 units |
| control | 1,696 | 0 | no | 1,696 units |
| PPU | 8,368 | 278 | yes | 8,391 units, 279 heads |

(The stream counts are lower where the merged schedule needed fewer clearing units.)

**The guard for what the engine leaves out.** A left-out unit still runs in the dense
pass. It can change the stream's tokens two ways only, and `Plan::prune` refuses the
plan unless both are closed:

- **Through RMSNorm.** The sum over non-constant channels of the largest square any
  occupant can hold, by the declared wire ranges (`Model::wire_ranges`; 0/1 for every
  other channel, attention outputs being convex combinations of 0/1 values), must stay
  below half an ulp of n·2^40. For the one checkpoint it is 4,092 against 262,144. Then
  every norm divides by exactly 2^15 whatever the left-out wires hold.
- **Through a routing unit firing at a decoded position.** Every left-out routing unit
  must be gated by a token channel (it reads an output slot identity, or RD, with a
  positive weight) that the stream's decoded tokens do not carry. It then sees
  `field + 0 ≥ 2`, which is 0 because an output field bit is 0/1.

Both rest on the circuits' range declarations, the same footing as the positional
stage's pruning. That is a real assumption. The first attempt tried to avoid it with a
pure interval propagation from the weights alone (each unit's activation over its
gate's interval), which loses the mutual exclusivity the circuits are built on and bounds the
norm sum at about 10^17, refusing even the four standalone checkpoints. The declared
ranges are what every routing layer already relies on (they keep the other slots'
routing units at 0 at each decoded position of any single checkpoint), and every unit
the engine does run is still checked at run time as before.

**The parts share the output identity channels** (round 9). An output
slot's identity channel is 1 on the slot's mask token. It tells the routing units, the PPU's
position wires and the layer-0 fallback key which slot a position holds. Until round 9 each
of the 530 output slots had its own channel. Now a part's k-th output slot holds the k-th
of 163 channels (`Layout::identities`), as many as the widest part, the PPU, has output
slots. With the 38 input groups' that makes 201 identity channels where there were 568, and
hidden went from 2,560 to 2,176 (`docs/perf-log.md`, "Round 9: the parts share the output
identity channels"). A channel is therefore 1 at one slot of every part that has a k-th output.

Within a part the identities are distinct. At a position of part p every row of p reads the
values it read before, under another channel number. The change is that a row of part q can
find an identity it reads at 1 at a position of part p. Every row that reads an output
identity is silent there, whatever the identities hold:

- **MLP units.** The routing units, the family-bit units and the PPU's units that read a
  position wire each read their part's mode channel as `up`. The mode is 0 at every other
  part's position and at every record position, so the unit adds exactly 0. This is the gate
  that already silenced a part's units where another part's wires hold their channels.
- **Memory head queries.** None reads an identity today (measured on the weights): the
  PPU's positional heads read wires that its units compute from the position wires. A
  query that read one would be covered the same way as any other. Off its part's
  positions its null term `N·(1 − mode)` sends the head to the null cell, whose value is
  0. `N` leads every score the query can reach with each channel at its largest magnitude
  (`foreign_score`), and that bound takes a channel outside the wires at 1.
- **Layer-0 keys.** The fallback dim reads every output identity with one weight, so every
  mask scores what it scored. The record gathers key on their record slots' identities,
  which are GBC CPU slots. Another part's mask on the same identity now scores with them.
  In a step the record slots hold masks, every mask's layer-0 value is 0, and the gather
  writes 0 as before. A record (`memory::dense::Records`) holds REC, the family tokens,
  write 0's line masks and the CELL or RD tokens, and no other part's token, so its
  gathers weigh the tokens they weighed before, each exactly 1/len.
  `one_model::group` derives each stream's group and margins from these weights
  (`tests/stream_audit.rs`: each group is still the stream's own slots, and in the GBC
  CPU's group a record gather attends to its own record slots alone).
- **No other row.** No value row, `up` row or LM head row reads an output identity, and no
  head or unit writes one.

`tests/one_model.rs` (`every_reader_of_an_output_identity_is_gated_by_its_part`) reads
this list from the compiled weights: each unit's `up` row, each head's null term and its
margin over the query's reach, the layer of each key row, and each mask's layer-0 value.

A stream's network (`Model::stream`) folds its own units' `up` onto the constant and clears
the mode channels. At another part's position that holds one of its identities, its routing
unit then computes what it computes at its own slot. The stream never decodes that
position. An MLP writes only its own position, and no key or value row reads a channel an
MLP writes (layer 0's keys read identities and its values the lanes; the later heads' keys
and values read the gathered address and line), so nothing computed there reaches a decoded
position. The same test checks that from the weights. The whole checkpoint, which a stock runtime runs, keeps
the mode gate. The engine reads identities back to slots among the stream's own outputs
only (`Layout::own_outputs`): the owner table tracks a routing unit's owner there, and the
positional stage names each pixel's position there.

**Tested.** These checks hold the streams to dense passes.

- **The step oracle** (`GBCFORMER_STEP_ORACLE`, `Engine::check_step`). Every Nth fast
  pass of a stream, the first included, also runs through the dense pass of the
  stream's own network, its slice (`Model::stream`), over the same tokens and the same
  KV cache. The first decoded token that differs stops the run. The GBC CPU stream reads
  the machine's cache and the PPU stream its own image of VRAM, OAM and the palettes.
  The APU and control streams hold no memory, so they read a cache of line 0 alone,
  the one cell their copy heads read, since their record slots hold masks. The value is
  one cadence for every engine (`5000`) or one per component (`cpu=100000,ppu=8400`).
  It counts passes, so a GBC CPU step's second pass and a CGB line's second pass count
  too. `tools/gate.sh` runs Yellow CGB 300 as one model at
  `cpu=100000,apu=25000,control=6000,ppu=8400`. The run reaches `cfd31f3d` with 10
  passes checked in each stream, every token equal, in 187 s at load 26 to 29 (the run
  without the oracle takes 19 s). At one cadence of 50,000 the counts were 19, 5, 2
  and 2, and a check took 18.5 s on the GBC CPU stream (its slice over 141,398 cells),
  4.1 s on the PPU's, 2.4 s on the APU's and 1.7 s on control's, at load about 30.
  Since this round the dense pass splits its rows across threads, bit for bit the
  serial pass (`docs/perf-log.md`, "The step oracle").
- **The slice test** (`every_slice_is_the_checkpoint_at_its_positions`,
  `tests/one_model.rs`). At the APU's and control's output positions, the whole
  checkpoint's dense pass and the part's slice give bit-identical residuals and the
  same tokens, and the stream's fast pass decodes those tokens. It covers those two streams only, not the GBC CPU or PPU. It runs 8 random passes
  for each of the two in every `cargo test --release`, 133 s at load 20 to 26 (until this round
  2 passes, ignored by default, 230 s).
- **Every part's slice** (`every_parts_slice_is_the_checkpoint_at_its_positions`, since
  the re-levelling's second round). The same comparison for all six parts, the scanline
  PPU, the GBC CPU, the length and the stepper included: one pass each on random input
  bytes the part's own circuit accepts, over a cache of the lines it reads there and 1,024
  other lines of a random memory. The residuals are bit-identical on the slice's channels,
  0 above them, the tokens equal, and each token is the byte the part's own circuit gives,
  the GBC CPU's 14 record slots' family tokens included. The same pass with no null cell
  must differ, so the test shows it would see a head that reads off its part.
- **A cell's rows at every memory head**
  (`a_write_records_cell_is_the_caches_at_every_memory_head`). The whole checkpoint's dense
  pass over 27 write records gives the cell token, at each of the 185 memory heads (17
  layers, the last the stepper's at 22), the formula's key and the line's value, and the
  engine's row builder over the whole checkpoint's weights gives those rows at each
  memory layer. An unpatched runtime computes a cell this way, through layers past the
  GBC CPU's own 11.
- **All six streams, and the records** (`tools/gate.sh`). Mealybug
  `m3_bgp_change`, 4 frames as one model with every component in lockstep, at
  `cpu=20000,apu=4000,control=2000,ppu=20000` (the length and stepper streams share the PPU's
  cadence) and `GBCFORMER_ORACLE=500`: the GBC CPU 3 passes checked, the APU 1, control 3,
  the PPU 1, the length 1, the stepper 37, and 3 memory records through the dense record
  oracle. Every record token of the one model sat above 65,535 while its vocabulary was
  65,804. It is 27,511 since the parts share the input slots, so no token is that large,
  and the test is `the_cpu_stream_keeps_records_through_the_oracles`. The OAM rewrite passes are checked one by one on blargg `2-causes`.
- **The oracle catches a wrong result** (`tests/step_oracle_faults.rs`). The APU engine's 40
  random passes, each through the step oracle: unchanged, all agree; with one LM-head weight
  of the dense pass changed (`Engine::dense_mut`), or the first token of one fast pass
  flipped (`Engine::inject_fault`), the run stops with the oracle's own message. Both hooks exist only with feature
  `testing` (`cargo test --features testing`), so a normal build cannot mutate the weights
  under a checked plan. A check asked for a
  pass that falls back or is delegated is spent there, not on the next fast pass.
- **The whole checkpoint over a real run** (`examples/slice_full.rs`, in `tools/gate.sh`). The
  full program records one event in every few thousand of each stream from Yellow 60 and
  Mealybug `m3_bgp_change` 2, as one model from the GGUF; the whole 25-layer checkpoint, not the
  stream's slice, runs its dense pass over each recorded sequence, and every decoded token must
  equal the stream's. Up to three events per stream per run, all six streams: 2,038 tokens
  equal, the GBC CPU's record tokens among them above 65,535 (about 35 s and 12 GB a pass).
  Every stream memory of at most 20,000 cells is recorded and checked whole
  (`GBCFORMER_RECORD_CELLS`, `slice_full --cells`, which fails such an event without its
  cells): the PPU, length and stepper streams on both runs, and the GBC CPU stream on
  `m3_bgp_change`, whose 32 KiB cartridge leaves a cache of 12,758 cells, so there the CPU
  stream's dense pass reads every cell its own cache held. `slice_full` also requires six
  distinct streams by their slot lists, positions and tokens of one length, and every stream
  with a memory checked over its whole cache at least once. Yellow's GBC CPU cache (143,830
  cells) is too large to write with each event, so it passes over the cells its heads read and
  line 0, a selection the engine under test made; for that cache the whole-cache evidence is
  the llama.cpp run over the prefilled cells (§7, `docs/stock-llama.md` §12a). The proof of record is not whole-cache evidence, since its passes attend their scopes of the cache. Without the PPU's whole image the check
  failed at every pixel, which shows it reads what it is given.
- **The APU stream against the APU's own checkpoint**, over 500 random segments, all on
  the fast path (`every_stream_is_its_part_on_its_slice_of_the_weights`).
- **Lockstep.** The streams together run Yellow in lockstep (§6).

## 5. Lockstep per stream

Nothing changes. Each stream drives the device it drove before, and each device's
lockstep is the one it had. `Lockstep` holds the CPU to the reference interpreter,
memory lockstep audits the cache against the host's arrays, `ApuLockstep` compares
state, samples and reads after every pass, `ControlLockstep` compares state, reads and
lines after every pass and advance, and `PpuModel` holds every line's 160 pixels to the
reference renderer. The first divergence on any thread stops the run, as before.

## 6. Gates and speed (measured)

Yellow CGB for 300 frames with `--weights all --lockstep all --one-model` reaches frame
hash `cfd31f3d`, the same as `--core lockstep`. The run has 913,782 CPU passes, 41,750
lines equal to the reference pixel for pixel, 243,951 APU passes equal in state, sample
and read, and 49,964 control passes equal, with no dense fallback in any stream. Startup (compile, one
plan check, four pruned plans, the memory insert) took 8.8 s.

**Speed.** The machine's 1-minute load was 8 to 22 throughout (other builds
and runs), so wall-clock fps moved by 30% between identical runs and instructions
retired are the primary evidence. Each figure below is the 600 model frames alone, a
full run minus a run that stops at the hand-off, so compile time cancels. Both sides use
the same binary, `--weights all` (A, four checkpoints) against
`--weights all --one-model` (B), with no lockstep.

| Change | text.sh A → B | overworld.sh A → B |
| - | - | - |
| first build | 362.5 → 428.8 G (+18%) | 215.2 → 246.9 G (+15%) |
| layer-0 gathers pruned per stream | 368.2 → 432.1 G (+17%) | 219.0 → 249.6 G (+14%) |
| a stream compares only the positions it writes | 352.6 → 373.1 G (**+5.8%**) | 208.8 → 218.9 G (**+4.8%**) |

After the last change, on text.sh, the CPU stream went from 254.9 to 270.6 G (+6.2%,
about 4.9 k instructions a step on a step of about 80 k) and the APU stream from 78.0 to
81.1 G (+4.0%). Scheduling is not the cause. Without `Circuit::schedule` the CPU stream is
+5.6%. What `sample` still shows is the view's translation (`View::place`, the token
maps each pass), the 425-token scan in `keep_merged` per write record, and copies of the
425-token sequence, about 3% of the CPU thread together; the rest is not attributed.

Under that load the wall clock (interleaved A B, three runs a side, median) went from
27.4 to 22.3 fps on text.sh and from 61.4 to 49.2 fps on overworld.sh on the first
build, and from 23.5 to 17.7 and from 46.5 to 45.5 on an intermediate build, where A
itself ranged 20.5 to 26.9 on text.sh. These do not separate a
6% difference and are not used for the verdict. Per pass the engine evaluates the same
units in both (CPU 1,409 against 1,423 per pass, APU 3,813 against 3,896, control 181,
PPU 3,395). The target is therefore reachable, since the remaining 5 to 6% is
bookkeeping around the passes, not circuit work.

## 7. What remains for release

What this section listed before has landed. The overlay is built
(`docs/hardware-shaped.md` §15). The one model is 628.6 M dense parameters with
0.97 M nonzeros (digest `ba60eff1`), and each stream runs its own slice, so the per-pass
bookkeeping gap of §6 closed (text at 60.7 fps as one model, `docs/perf-log.md`, "Host
and engine: the machine thread's last few percent"). The dense oracle now covers every
stream's steps (§4, the step oracle). What remains:

- **The whole checkpoint over a machine's whole cache, for the GBC CPU stream.** The whole
  checkpoint is held to every stream over real runs (§4), the PPU streams
  over their whole image; the GBC CPU stream over the cells its heads read. The paragraph below
  is the earlier state. That comparison is the
  whole 628.6 M checkpoint's dense pass with every memory head over Yellow's 141,398
  cells. It costs more than the 18.5 s a check of the GBC CPU's slice, since the whole
  checkpoint adds the PPU's memory heads (not measured). The evidence for those two
  streams when this was written was the llama.cpp b12070 run on Metal of the whole one-model GGUF over Yellow's
  prefilled cells, whose 2,863 GBC CPU and 102 PPU events matched the engine's tokens
  (`docs/stock-llama.md` §12a).
- **Speed under load.** 60 fps on text needs about 3.1 µs a step. The head takes about
  3.07 µs at load 1 to 2.5 and about 3.19 µs at load 3 to 4, so the next gain has to come
  from the circuit or the memory path's shape (`docs/perf-log.md`, "Host and engine").
- **Memory footprint.** Each engine keeps its slice resident for the oracle and the
  fallback. Yellow CGB 300 with every component in weights, in lockstep, as one model
  peaked at 6.8 GB (6.3 GB resident), measured with `/usr/bin/time -l` this round.
- **The parts each phase still leaves on the host.** The host keeps the PPU's LCDC, SCX,
  SCY, WX and WY latches and the APU's sample instants and DIV edges
  (`Component::still_hosted`).

## 8. Reproducing

The `one_model` example (shapes, overlay projection, stream plans, the APU stream against
its own checkpoint: `cargo run --release --example one_model -- --build`) has since been
removed.

```sh
cargo test --release --test one_model                  # slots, one set of weights, plans, the APU stream, the APU and control slices against the whole checkpoint
./target/release/gbcformer run pokemon-yellow.gbc --frames 300 --weights all --lockstep all --one-model --no-save
GBCFORMER_STEP_ORACLE=cpu=100000,apu=25000,control=6000,ppu=8400 ./target/release/gbcformer run pokemon-yellow.gbc --frames 300 --weights all --lockstep all --one-model --no-save   # the same run, 10 passes of each stream against its dense pass
```

The other gates measured on the final binary: `--one-model` DMG 8 frames `6872b003`
(144 lines, 231 APU passes, 6,497 control passes equal); `--core lockstep` DMG 8
`6872b003` and CGB 300 `cfd31f3d`; the CPU checkpoint's digest `1f4c3519`; every test
binary ok (lib 119, bin 5, `checkpoint_sst`, `sm83_sst`, `memory_dense`, `dma_records`,
`e01_primitives`, `interrupt_fuzz`, `ppu_checkpoint`, `ppu_dma`, `one_model`); `model`
and `model-mem` no slower than the base by instructions (`docs/perf-log.md`).

**Digest after the `MAP` slots (H4).** The GBC CPU part's memory circuit names the physical
address of each banked window's first byte in twelve result slots (`MAP0a..MAPWc`,
`docs/stock-llama.md` §12a), so a stock runtime can attend only the mapped banks. The
one model's digest moved from `c9fb64ad` to `b03d896c` (2,057.9 M dense parameters,
sequence 461) and the `cpu-memory` checkpoint's from `eddbfd8c` to `ffe22414`; the GBC
CPU checkpoint (`6bd416c1`), the APU's, control's and the PPU's are unchanged. `tools/gate.sh`
passes on it, `Yellow 300 all, one model` at `cfd31f3d` included.
