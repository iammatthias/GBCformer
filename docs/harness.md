# The harness: the GGUF's weights, fast, checked against llama.cpp

Milestone 1 is built. It reads the weights from the file, serves memory reads over the
cells' K/V rows, matches llama.cpp token for token, and has measured speed. Milestone 2
(speed, §8) is built and took text from 24.3 to 36.0 fps and overworld from 29.7 to 72.8.
Milestone 3 (§9) is built. It walks the gates in blocks and took text from 34.9 to 38.7
fps and overworld from 70.0 to 73.0, against milestone 2's binary on a busier machine.
Text is still short of 60 (§9.4). The goal is the whole Game Boy Color in the model's
weights at 60 fps on an M1 Max, with every pass's tokens identical to llama.cpp running
the same file. llama.cpp cannot reach 60 fps itself. A GBC CPU pass of the current weights costs about 35 ms in
it unpatched on Metal and about 12 ms with the patches (`docs/perf-log.md`, "The proof of record on `ff86dde7`" and "Round 14"), and Yellow needs about 470,000 passes a second. So this
harness is the fast runtime, and llama.cpp is its check.

## 1. Rules

- The harness computes the weights of the file llama.cpp loads. No side computation
  replaces them.
- A memory read is attention over the K/V rows llama.cpp keeps for the cells. A fast
  path may find the one row that matters, but it must be proven equal to the attention
  result and guarded on every read, and it falls back to the full computation when the
  guard fails.
- Every pass's decoded tokens equal llama.cpp's for the same file (`stock-llama diff`).

## 2. Weights from the file (`src/gguf.rs`)

`GBCFORMER_GGUF=<model.gguf>` makes a one-model run (`one_model::streams`, and the
`stock-llama` harness's `checkpoint`) read every weight from the file. The file holds
weights only, in `--layer-shapes --align 64` form. For each layer it keeps only the query
heads and FFN units that are not all zero, padded with zero units to a multiple of 64, and
it stores `attn_q` and `attn_k` in llama.cpp's interleaved pair order. The compiler still
supplies what is not a weight (layout, slot names, wire ranges, memory spec) and the
indices of the kept heads and units (`gguf::keep`, which `examples/export.rs` also uses).
`gguf::load` expands each tensor back to the compiled layout. It undoes the interleave,
places rows and columns at their indices, requires every padding row and column to be
zero, and requires every norm weight to be `GAMMA`. It then compares the tensor bit for
bit with the compiled tensor and replaces the compiled tensor with it. One differing value
fails the load.

Measured check. `cargo run --release --example digest -- --gguf <model.gguf>` prints the
one model's digest as compiled and again after the load. Both the earlier file
(`c9fb64ad`) and the current one (`b03d896c`, after the MAP slots) give the same digest
before and after. The load reads 107 tensors, expands the file's 1,568,512,000 values to
2,057,738,240, and finds 0 that differ.

**Stored runs.** The layer-shaped file also leaves out three blocks that
are zero by construction. The export derives each from the tensors, never from the layout's
numbers, and names it in a metadata key that the loader and
`tools/stock-llama/stored-runs.patch` both read. A file without a key stores that tensor
whole, so an earlier file loads as before.

| Key | The file stores | Left out |
| - | - | - |
| `gbcformer.output.rows` (first, count), `gbcformer.output.fill` | the rows of `output.weight` from the first to the last that is not the fill row (`gguf::HeadRows`) | every row that holds only `fill` on channel 0 |
| `gbcformer.ffn_up.cols` (first, count per layer) | per layer, the run of whole 64-channel quanta that holds every channel a kept unit's `up` row reads (`gguf::up_cols`) | the zero columns of `ffn_up` |
| `gbcformer.token_embd.cols`, `gbcformer.output.cols` | the prefix of whole quanta that holds every nonzero column (`gguf::col_prefixes`) | the zero columns past it |

Two more keys per stream of the overlay store nothing and let a runtime compute less:
`gbcformer.stream.<part>.head`, the run of tokens the stream can decode
(`gguf::stream_head`), and `gbcformer.stream.<part>.width`, the prefix of channels its rows
read and write (`gguf::stream_width`).

The loader expands each run back to the compiled layout, a fill row outside the head's run
and zero columns elsewhere, and then compares every value with the compiler's as before.
`gguf::stored_params` counts what the file stores from the same derivations, and the export
asserts that it equals the summed tensor lengths.

## 3. The cells' K/V rows (`src/engine/cells.rs`)

Each engine with memory keeps, per cell (one per 8-byte line) and per layer, the f16 key
and value row of the one KV head, as llama.cpp's f16 cache stores them. That is 15 layers
× (64 + 64) halves, 3,840 bytes a cell, and 543 MB for Yellow's 141,415 cells.

**How a row is computed.** The engine computes it as llama.cpp does when it decodes the
line's write record `REC WP0a..c WP1a..c WL0..WL7` at `P` with the cell token `CELL` at
`P + 1`, then moves the cell to position `n` (`seq_add`, which re-ropes the key):

1. Layer 0's attention of the cell token over its record. The query is the same at every
   token, because it reads only the constant channels, which every embedding holds alike.
   The record tokens' keys and values come from the normed embeddings, roped at their
   positions. All of them are rounded to f16, and the softmax runs in f32.
2. The attention's output projection onto the channels that the keys and values past
   layer 0 read is added to the cell token's embedding, normed (× 2^7, the eps-dominated
   RMSNorm of `docs/stock-llama.md` §2) and rounded to f16.
3. Each layer's `W_k` and `W_v` apply to that result. The key is roped at `P + 1` and
   stored in f16, then roped by `n − P − 1` and stored in f16.

**Row groups.** Layers 1 to 13 of the one model have the same `W_k` and `W_v` rows over
the channels a key or value reads, weight for weight, so they produce the same rows for
every cell. `RowBuilder::new` groups the layers by that equality (`{0}`, `{1..13}`,
`{14}`), `rows` computes each group once, and `Cells` stores one row per group and cell
(108 MB for Yellow's cells in place of 543 MB). A read at layer `l` reads its group's row,
and `Cells::rows_by_line` expands the groups back to the 15 layers llama.cpp holds. The
build tabulates layer 0's scores of the cell token against each token's key with the same
f32 fold, and `Cells::turns` keeps each line's RoPE turn by `n − P − 1` for the line's
next record.

`RowBuilder::new` proves the shortcut in steps 2 and 3 (no MLP, no later attention) from
the weights. No MLP and no attention past layer 0 writes a channel that a key or value
past layer 0 reads. It also proves that layer 0's query reads only constants, that only
the cell token's embedding sets the committed channel, that no layer-0 head writes it, and
that the key's committed dim reads nothing else. For each record, the cell token's best
layer-0 score must beat any cached cell's bound `|q|·|k0(CELL)|` by 104, so the cells
weigh exactly 0 at layer 0.

Measured check (`examples/cells_state.rs`, since removed). The example parses llama.cpp's saved cache for
Yellow (`yellow-cache-B.state`, 141,415 cells, written by the Metal prefill) and builds
every cell's rows again from the line its value spells. Of the 141,415 cells, 0 differ as
f16 numbers in any of the 15 layers' key or value, and all 141,415 hold the canonical key
at every memory layer. Every cell differs in some bit, the sign of a zero, in the partner
dims of rotated key pairs (llama.cpp writes −0 where this code computes +0, and the
reverse). No dot product or weighted sum can tell ±0 apart, so the guard compares rows as
numbers (`cells::same`).

## 4. A read

A memory head's query is the lattice point of its line, with `±S` on the 21 address dims,
`B` on the committed dim and 0 elsewhere. `MemPlan::check` proves the pattern from `W_q`,
and the engine's per-pass guard requires each address bit's form to be 0 or 1.

**The lookup and its guard** (`Cells::read`). `Cells::read` finds the row by the line's
index. It serves the row's value only when all of these hold:

- (a) every row in the cache holds the canonical key of its own line. The code checks this
  as it writes each row, and the count of irregular rows must be 0. So every other row
  differs from the query's line in at least one address bit and scores at least
  `2·S·scale = 256` below the matching row, where f32 `exp` is exactly 0 (which needs 104);
- (b) the row found holds the canonical key of the line read. At a memory layer this is
  the comparison made when the row was written, kept as the row's flag, which (a) makes
  true for every row;
- (c) its value is exactly 0 or 1 on every dim. The code derives the line it spells when
  it writes the row and keeps that line beside it.

Then the softmax over every row weighs that row exactly 1 and every other row exactly 0,
and the read is that row's value. The group's own tokens at the memory layers hold no
committed bit (§3), so with address entries within ±1 they score at least `B·scale = 256`
below the cell of the line they name (`Params::check`, the uncommitted margin). That
bound on the address entries comes from the gathers' design. The comparison with
llama.cpp checks it, and no weight proof does.

The fallback is the full scan. It scores the query (roped at `P`, rounded to f16) against
every row's key, takes the softmax in f32 and mixes the values. A result that is not
exactly 0 or 1 on every dim refuses the pass. Every read path goes through the rows: the
heads of every stream (`Engine::memory_reads`), a DMA copy's source and destination, and
the PPU's palette reads (`Engine::read_line`, through the copy heads' layers). The byte
image remains only for the record bookkeeping (undo log, banking, the PPU's video record
queue, the memory audit).

`GBCFORMER_SCAN=1` runs the full scan beside every lookup and stops at the first
difference.

Measured checks:

| check | reads | lookups the scan checked | fallbacks | differences |
| - | - | - | - | - |
| SST through the CPU-memory checkpoint, all 500,000 cases, 0 failed (`GBCFORMER_SCAN=1 cargo test --release --features stock --test checkpoint_sst memory_checkpoint_matches_singlesteptests`, 1,187 s) | 3,994,124 | 3,994,124 | 0 | 0 |
| Yellow, five frames, every component in weights and in lockstep, one model (`GBCFORMER_SCAN=1 gbcformer run … --frames 5 --weights all --lockstep all --one-model`), CPU stream | 59,908 | 59,908 | 0 | 0 |
| same, PPU stream | 1,392 | 1,392 | 0 | 0 |

The five frames are boot frames with little memory traffic. The text workload (§6) serves
all 8.2 million of its reads with the lookup and 0 fallbacks, but it does not run the scan
beside each.

## 5. Token for token against llama.cpp

`stock-llama play none` built from this branch, with `GBCFORMER_GGUF` set, records
every pass (`GBCFORMER_RECORD`). `stock-llama diff` compares that record with llama.cpp's
record of the same file.

| record | events compared | differ | decoded tokens |
| - | - | - | - |
| harness, five frames, scan oracle on, against the engine's record `rec-engine-B` | 56,419 | 0 | 3,599,265 |
| harness against a fresh llama.cpp `play` (`stock-llama-B`, Metal, b12070 with the layer-shapes patch, scoped attention, own groups, 600 passes from the saved cache, inside the GPU lock) | 601 | 0 | 40,068 |

The lines each memory head read (the records' `heads` field, which `diff` does not
compare) equal the engine record's in every stream, 597,780 GBC CPU head lines and 36,000
PPU head lines.

## 6. Speed

The workload is `tools/bench/text.sh --weights all --one-model` (Oak's speech, 600 model
frames after 2,900 labelled reference frames, frame `870aa3fd` in every run), with every
component in weights as a stream of the one checkpoint, `GBCFORMER_STATS=1` on both
sides, an M1 Max and a release build. Load averages are `uptime`'s before each run, and
other jobs ran throughout.

**Harness against the engine it replaces** (interleaved A, B, A, B, A, B). A is
the build before the harness (the image index, compiled weights) with only the layer-0 fix below
applied. B is the harness (weights from the GGUF, every read over the cells' rows).

| side | fps per run | median | CPU step, µs | load |
| - | - | - | - | - |
| A, image index | 24.2, 23.6, 23.6 | 23.6 | 7.68, 7.89, 7.89 | 2.8 to 6.2 |
| B, harness | 24.0, 23.8, 23.2 | **23.8** | 7.76, 7.82, 8.02 | 2.4 to 6.5 |

There is no measurable difference, since the gap is inside the run-to-run spread. Every
read in B is a lookup (GBC CPU 6,925,025, PPU 1,297,629, 0 full scans). B built 1,166,267
rows from the weights (141,398 at insertion in 0.9 s, then one per write record), and no
pass fell back.

**A regression in the build before, fixed here.** That build alone measures 1.4 fps on this
workload (one run, load 3.2; 70.9 µs per 72-position GBC CPU pass, 12.6 ms per
67-position pass). The MAP slots made layer 0's attention replan twice per host-served
step (the late data slots change their keys), and each replan compared every decoded
position's logits with the first's over every key of every head (about 13 ms). The
changes in `engine/layer0.rs` are all exact. A position whose query has the first
position's query id (bit-identical vectors) has bit-identical logits, so the engine does
not compare it again. The engine memoises a plan by every position's (key id, query id)
and the decoded positions, of which the plan is a function, so a key set seen before
reuses its plan. A moved key whose logits equal the old key's in every head keeps the
plan. There are still 0.010 replans per pass, and each now costs a hash lookup.

**Where the time goes** (B, run 2, 25.2 s for the 600 frames).

| thread | passes | µs per pass | busy | share of the wall |
| - | - | - | - | - |
| CPU (the machine's thread; control runs here too) | 3,224,368 steps | 7.82 per step: engine 5.0 (layer 0 0.20, shared layers 4.60, decode 0.15), the rest outside the engine (the step protocol, the write records and their rows, the host machine) | 25.2 s | 100%: the limiter |
| PPU | 172,800 (two per line) | 107.6 (shared layers 106.2, most of it the per-pixel stage's 161,000 units) | 18.6 s | 74% |
| APU | 489,745 | 6.4 | 3.1 s | 12% |
| control (on the CPU's thread) | 111,889 | 2.5 | 0.3 s | inside the CPU's |

At 60 fps the 600 frames take 10 s. The GBC CPU's thread must then spend at most 3.1 µs
per step (2.5 times less than now), and the PPU at most 58 µs per pass (1.9 times less).

## 7. Closing the gap to 60 fps

Milestone 1 listed three levers with estimates, and §8 measured them. The PPU's per-pixel
stage in parts (§8.2) took the PPU off the critical path, and overworld runs above 60 fps
(72.8). The write records' rows (§8.1, §8.4) and the reads cut (§8.3) took the step from
7.7 to 5.2 µs, more than the estimate. The GBC CPU's shared layers (§8.3) did not move,
because their gate walk is a scalar gather-scatter near its issue limit.

§8.6 measured what stands between text and 60 fps. Text needs 5,374 GBC CPU steps a
frame, so 60 fps needs at most 3.1 µs a step, and the step costs 5.2. The engine's pass is
4.56 of that (layer 0 0.19, shared layers 4.19, decode 0.14), and 0.6 is outside it. Of
the shared layers, the gate walk (`Marks::layer`) is about 3.0 µs, 2,097 gate updates a
pass at 3 to 5 cycles each. Even with nothing outside the engine, the pass alone is 1.5
times the budget. These changes could close the gap, and none of them is a lever of this
milestone:

- **Fewer gate updates.** The 2,097 updates a pass are what the one model's weights give
  for the 211 (changed channel, reading layer) pairs a step moves, and the engine
  evaluates each exactly once. Fewer updates would need a circuit (and so a GGUF) whose
  GBC CPU stream moves less per step, which is a compiler change. Milestone 3 measured the
  engine-side version (§9.2). Exact tables for units with identical inputs absorb 946 of
  2,154 updates a pass, and they lost, because those were the decoders' full blocks, the
  cheapest updates the walk makes.
- **A vectorisable layout of the gate walk.** Milestone 3 built this (§9.2). It renumbers
  units so a channel's readers form blocks, updates four gates in each NEON update, and
  left-packs the sends. It took about 0.55 µs off the step.
- **Memoising whole GBC CPU passes.** A cache keyed by a pass's input tokens and the lines
  its heads read could replay a repeated step's tokens, since the pass is a function of
  those. It would replace evaluating the weights on that pass, which the project's rules
  forbid, so it is not built.

The scoped attention of `docs/stock-llama.md` §12a (`Circuit::reach`, the MAP-routed
windows) bounds which rows a head can match. The lookup does not need it, since it is one
index read, but the full scan would. Scoping the fallback scan to those rows would cut a
guard failure's cost from the whole cache to about 2,100 rows (7,300 for the GBC CPU),
under the same guard. No read fell back in any run here, so it would buy nothing
measurable yet.

## 8. Milestone 2: toward 60 fps

This section uses the workload and machine of §6, with `tools/bench/overworld.sh
--weights all --one-model` beside it. Every comparison is interleaved A, B, A, B, A, B
against the binary before the lever, with `GBCFORMER_STATS=1` on both sides and load
averages from `uptime` before each run.

### 8.1 The GBC CPU step outside the engine

**The number explained** (`sample` of the milestone-1 binary on text.sh, 12 s of the
GBC CPU's thread, scaled from the 8.96 µs per step under sampling to the 7.82 unsampled):

| piece | share of the CPU's thread | µs per step |
| - | - | - |
| the engine's pass (layer 0, shared layers, decode) | 62.6% | 5.0 |
| write records: `Cells::put` for the CPU's writes (0.32 a step) | 29.1% | 2.3 |
| of which `RowBuilder::rows` | 21.8% | 1.7 |
| of which `Cells::regular` (twice a put, 13 layers × 64 dims) | 6.2% | 0.5 |
| write records for DMA copies (`Engine::copy`) | 2.2% | 0.2 |
| the step protocol (`MemCheckpointCore::decode`, `fill_into`, the view) | about 3% | 0.25 |
| the host machine's advance and the control pass | about 1.5% | 0.1 |
| `GBCFORMER_STATS` clock reads (on both sides of every comparison) | about 2.5% | 0.2 |

A put cost about 6.5 µs. It computed thirteen identical layers' projections, each a serial
f32 fold, and layer 0's two live heads' scores over 16 keys of 64 dims, also serial folds.
It also took a `sincos` per rotated pair per distinct layer, allocated a Vec per
intermediate, and checked the regularity of the old and the new row at 13 layers.

**The cut** (`RowBuilder::group_rows`, §3's row groups). The 13 layers are one group
(proven at build time from the weights, bit for bit), the layer-0 scores are a table, the
cells keep the turn per line, the regularity is a flag per row, and nothing allocates. The
code still computes the rows from the weights for every record. The GBC CPU step fell by
1.7 µs of the 2.5 the records cost. §8.4 covers the write-record memo.

Checks. `examples/cells_state` against llama.cpp's `yellow-cache-B.state` finds 0 of
141,415 cells that differ as f16 numbers in any of the 15 layers' key or value (rows
expanded from the groups). In Yellow five frames with every component in weights and in
lockstep, one model and `GBCFORMER_SCAN=1`, the full scan checked 59,908 GBC CPU and 1,392
PPU lookups and found 0 differences. That run's record against milestone 1's llama.cpp
record (`stock-llama diff`) has 601 events, 0 differing, and 40,068 decoded tokens.

| workload | side | fps per run | median | CPU step, µs | load |
| - | - | - | - | - | - |
| text | A, milestone 1 | 23.5, 23.6, 23.8 | 23.6 | 7.92, 7.87, 7.82 | 2.6 to 3.0 |
| text | B, row groups | 30.4, 30.0, 30.6 | **30.4** | 6.13, 6.20, 6.08 | 2.7 to 3.2 |
| overworld | A, milestone 1 | 29.4, 29.7, 29.2 | 29.4 | 17.2, 17.0, 17.3 | 2.5 to 3.0 |
| overworld | B, row groups | 30.1, 30.7, 30.1 | **30.1** | 16.8, 16.5, 16.8 | 2.4 to 2.8 |

On text the GBC CPU's thread is still the limiter (engine 4.85 µs of 6.1, the PPU 91%
busy). On overworld the PPU is the limiter. Its 108 µs passes, 288 a frame, cap it near
32 fps, and the GBC CPU's thread waited 11 to 13 s of the 20 for it at palette reads (its
GBC CPU step counts the wait).

### 8.2 The PPU's per-pixel stage in parts on helper threads

This came before the GBC CPU's shared layers (§8.3). After §8.1 the PPU thread bounds
overworld (the GBC CPU's thread waited 11 to 13 s of 20 for it there) and comes next on
text, so a gain on the GBC CPU's side would not show in either until the PPU passes
shrink.

A position's positional units read only the shared channels and its own block of
channels (`Positional`), so positions are independent once the shared layers have run.
`Positional::split` cuts the 160 positions into parts of consecutive blocks
(`GBCFORMER_POS_PARTS`, by default 4), each with its own layers, and each part keeps its
own state, marks and buckets (`PosPart`). `Positional::advance` hands part `i` to helper
thread `i − 1` of an `engine::pool::Pool`, runs part 0 on the PPU's thread, and then
joins. The parts run the same units, the same updates and the same range guards per part,
and the engine merges the stale positions after the join. A helper spins 50 µs for the
next pass and then parks (passes come 16 at a time). This is native only, with one part
where there are no threads.

Checks. `cargo test --release --test ppu_checkpoint` (the engine against the dense pass
and the circuit's colours) has 7 passed. dmg-acid2 and cgb-acid2 are pixel-exact with
`--weights cpu,memory,ppu,control --lockstep ppu`. Yellow five frames in lockstep with the
scan oracle give 144 lines equal to the reference colour for colour, and the record
against llama.cpp's has 601 events, 0 differing, and 40,068 decoded tokens.

| workload | side | fps per run | median | PPU pass, µs | CPU step, µs | load |
| - | - | - | - | - | - | - |
| overworld | A, §8.1 | 30.5, 30.3, 29.1 | 30.3 | 107.7 | 16.5 to 17.3 (counts the wait) | 3.1 to 4.7 |
| overworld | B, four parts | 68.8, 69.2, 67.8 | **68.8** | 36.7 | 7.3 to 7.5 | 3.4 to 5.3 |
| text | A, §8.1 | 30.5, 31.2, 30.5 | 30.5 | 101.9 | 5.97 to 6.11 | 4.1 to 5.6 |
| text | B, four parts | 30.7, 30.9, 30.9 | **30.9** | 36.1 | 6.01 to 6.07 | 3.6 to 4.8 |

Overworld passes 60 fps. Text does not move, because its GBC CPU step (5,374 steps a
frame) is the limiter. The GBC CPU's engine pass reads 5.02 µs in B against 4.87 in A on
text (run 2 of each), within the spread of the loaded machine. In B the GBC CPU's thread
waits 47 ms for the PPU on text and 1.5 s on overworld.

### 8.3 The GBC CPU's shared layers

**The number explained** (`sample` of §8.2's binary on text.sh, the GBC CPU's thread at
about 6.0 µs a step, where 1% is about 0.06 µs):

| piece | share | µs per step |
| - | - | - |
| `Marks::layer`, the gate walk of the circuit layers | 51.5% | 3.1 |
| of which the plain units' update (load, add, store, activation change, staged send) | about 28% | 1.7 |
| of which the fused pairs' update | about 12% | 0.7 |
| of which the send loop and the per-change setup | about 7% | 0.4 |
| `Engine::advance` itself (routing the changes, the head bucket, the layer loop) | 7.6% | 0.46 |
| `Marks::drain_wires` | 8.4% | 0.5 |
| `Marks::route_layer` | 4.4% | 0.27 |
| the memory heads' reads (`Cells::serve`, 2.1 a pass) | 4.2% | 0.25 |
| layer 0 | 2.4% | 0.15 |
| write records (`Cells::put`, `RowBuilder::group_rows`) | 8.2% | 0.5 |
| the host machine and the control pass | 3.2% | 0.2 |
| `GBCFORMER_STATS` clock reads | 2.0% | 0.12 |
| the step protocol (`MemCheckpointCore::decode`, `fill_merged`, `fill_into`) | about 4.5% | 0.27 |

The plain units' loop compiles to 16 instructions an entry (`otool`). They are two loads
of the entry, the gate's load, multiply-add and store, five for the activation change, the
staged store and its index, and the loop. The loop runs at about 3 to 5 cycles an update,
near what a scalar gather-scatter can issue. NEON has no gather or scatter, and a
channel's readers are not contiguous units, so the walk cannot be vectorised as laid out.

**Reads cut.** A memory read compared its row's key with the line's canonical key and
checked every value dim, about 120 ns a read. At a memory layer the key comparison is the
one the code makes when it writes the row (`Cells::regular`, all true while the irregular
count is 0), and the code derives the line a value row spells when it writes the row. The
read is now the index, the flag and that line. The guard and the result are the same. The
code still checks every condition, once per write instead of once per read.

| workload | side | fps per run | median | CPU step, µs | load |
| - | - | - | - | - | - |
| text | A, §8.2 | 30.7, 31.2, 31.1 | 31.1 | 6.05, 5.97, 5.97 | 3.3 to 4.2 |
| text | B, reads cut | 32.4, 32.2, 32.3 | **32.3** | 5.75, 5.78, 5.75 | 3.9 |

**The gate walk unrolled by two, rejected.** This variant ran the plain and pair loops of
`Marks::layer` over two entries an iteration (the same updates in the same order, with a
remainder loop). One pilot pair on text after §8.4 measured 35.4 fps before against 33.8
after (5.26 against 5.51 µs a step, load 2.4 to 3.1). It was slower, so it was reverted
without the three-round A/B. Laying the units out so a channel's readers are contiguous
(which a NEON update would need) or as dense columns is a layout change, not one
experiment, and was not started.

### 8.4 The write-record memo

A cell's rows are a function of its write record's tokens, the byte address and the line
(`RowBuilder::group_rows`). After §8.1 the records still cost about 0.5 µs a step (8% of
the GBC CPU's thread, §8.3's table), and on text most records repeat the tokens of a
record seen before. Of the 1,024,869 write records the GBC CPU kept, 888,081 hit the memo
below.

`Cells` keeps the rows built for recent records in a direct-mapped table of 2^16 entries
keyed by those tokens (`Memo`, about 70 MB a memory engine), with what `put` derives from
them (the canonical-key flag and each value row's line). The guard has two parts:

- a hit serves only rows that `group_rows` built from the weights for a record with the
  same tokens, found by exact equality of the byte address and all 8 bytes of the line
  (the table index is a hash, but the comparison is not);
- every 64th hit builds the rows from the weights again and stops the run if they differ
  in any bit.

For a record the memo does not hold, the code builds its rows from the weights as before.
Since round 5b (`docs/perf-log.md`) the memo and each cell keep what a lookup reads of the
rows apart from the rows, and in a session a worker thread writes the GBC CPU stream's
rows into their cells in record order and runs the 64th-hit rebuild. The worker may be 4
batches of 512 records behind, so a failed rebuild stops the run fewer than 3,072 records
after the write: at a later hand-over, at the settle before `run` reports and before
`play` writes its last save, or when the cells are dropped.
This is not the whole-pass memo that §7 rules out. No pass's tokens come from it,
only the cache rows of a record whose rows the weights already gave.

Checks. Yellow five frames in lockstep with the scan oracle check 59,908 GBC CPU and 1,392
PPU lookups with 0 differences, and have 648 memo hits, 10 of them rebuilt and compared.
The record against llama.cpp's has 601 events, 0 differing. On text, 888,081 of the
1,166,267 puts hit (the other puts include the 141,398 lines of the cartridge's
insertion), and 13,876 hits were rebuilt and compared, none differing.

| workload | side | fps per run | median | CPU step, µs | load |
| - | - | - | - | - | - |
| text | A, §8.3 | 32.4, 32.1, 32.5 | 32.4 | 5.74, 5.80, 5.73 | 3.5 to 4.7 |
| text | B, memo | 35.0, 35.1, 34.7 | **35.0** | 5.31, 5.31, 5.36 | 3.9 to 4.9 |
| overworld | A, §8.3 | 71.6, 71.5, 71.7 | 71.6 | 7.05, 7.06, 7.05 | 2.3 to 3.0 |
| overworld | B, memo | 73.5, 73.5, 73.9 | **73.5** | 6.87, 6.87, 6.84 | 2.8 to 3.1 |

The text B side ran with `GBCFORMER_MEMO_BITS=16`, the size the code then read from the
environment, which the commit fixed as the constant. A single pilot at 2^14 entries showed
no gain against its A and was not repeated. The PPU stream's `Cells` has the same table
and gets almost nothing from it, because its few records rarely repeat. Sizing the table
per engine was a follow-up, done in §9.3.

### 8.5 The host machine and control off the GBC CPU's thread, not built

§8.3's profile puts the host machine's advance and the control pass at 3.2% of the GBC
CPU's thread, about 0.2 µs a step. Of that, the DMA copies' write records (0.9%) belong to
the GBC CPU stream's engine and would stay. The machine and control advance between every
two GBC CPU steps, and the step's inputs (IE, IF, the control reads) come from that
advance. A worker would therefore need a handoff each way every step, which costs about as
much as the work it would move. The measurement says it does not pay, so it was not built.

### 8.6 Milestone 2 against milestone 1

A is milestone 1 and B is milestone 2 (§8.1 to §8.4), interleaved over three
rounds.

| workload | A fps | B fps | median A, B | load |
| - | - | - | - | - |
| text | 24.3, 24.2, 24.3 | 36.2, 36.0, 35.4 | 24.3, **36.0** | 1.9 to 3.1 |
| overworld | 29.7, 29.7, 29.7 | 72.8, 72.2, 72.8 | 29.7, **72.8** | 2.1 to 2.8 |

**Where the time goes after milestone 2** (B, text, run 2, 600 frames in 16.7 s):

| thread | µs per pass | busy | share of the wall |
| - | - | - | - |
| CPU (steps; control and the host machine here too) | 5.17 a step: engine 4.56 (layer 0 0.19, shared layers 4.19, decode 0.14), 0.6 outside it | 16.7 s | 100%: the limiter |
| PPU (two passes a line, four parts) | 33.6 | 5.8 s | 35% |
| APU | 6.15 | 3.0 s | 18% |
| control (on the CPU's thread) | 2.43 | 0.27 s | inside the CPU's |

A `sample` of the GBC CPU's thread (B, text) splits it as `Marks::layer` 57.5%,
`Engine::advance`'s own loop 9.2%, `Marks::drain_wires` 9.4%, `Marks::route_layer` 4.9%,
layer 0 2.5%, the step protocol about 5%, the write records 2.1%, the host machine and
control 2.0%, and `GBCFORMER_STATS` clocks about 2.7%. The memory reads fell below 0.6%.

**Checks on the final commit** (which adds only the pool's join on a panicking
caller to B). `tools/gate.sh` is green, with every step `ok` (load 2.2 to 5.4 at its
start). Yellow five frames in lockstep with the scan oracle check 59,908 GBC CPU and 1,392
PPU lookups with 0 differences, and that record against milestone 1's llama.cpp record has
601 events, 0 differing, and 40,068 decoded tokens.

## 9. Milestone 3: the gate walk's layout

This section uses the workloads, machine and method of §8, interleaved A, B, A, B, A, B
against the binary before the lever, with `GBCFORMER_STATS=1` on both sides and load
averages from `uptime` before each run. Other jobs ran throughout (an MLX build,
llama.cpp GPU runs), so loads are higher than in §8.

### 9.1 The number explained

**A replay of the walk.** To measure walk layouts without the rest of the run, a
throwaway build recorded the GBC CPU stream's per-layer change lists for 20,000 passes
(from pass 200,000 of text.sh, with the gates at the first of them) and replayed them
through each walk, best of 7. The replay reproduces the walk's cost in the run. The
scalar walk took 3.1 to 3.3 µs a pass in it (2,154 gate updates a pass), against §8.6's
estimate of about 3.0. Under Instruments' CPU Counters (CPU Bottlenecks, with the replay
alone on the process, of the one-lane-vector walk rejected below), the cycles split as 51%
useful, 24% stalled in the back end, 22% discarded (mispredicted) and 3% starved for
instructions. The inference from that and from the rejected variants is that neither cache
misses nor the update arithmetic alone sets the walk's cost, and that a layout which only
halves the updates does not halve it (see the groups below).

**How a channel's readers are laid out.** Per pass on text, the GBC CPU stream's 2,154
updates come from 211 changes. Most hot channels have many readers that read exactly the
same channels (the opcode decoder has three bits with 256 readers each at layer 2, 623 of
the layer's updates). So ordering a layer's units by their read channels puts each
channel's readers into runs of consecutive units. Under the measured change rates 88% of
the updates fall in runs of four or more, and 81% do under the static order the engine
uses (a unit's read channels, compared as sequences).

### 9.2 Blocks of consecutive readers, four gates at a time

`Template::pack` (at the end of `Template::check` and `Plan::prune`) renumbers each
circuit layer's units to the ones that write, in `IntLayer::packed_order`. The order is a
bijection on the writing units (`IntLayer::permuted` asserts it), so every gate, send and
channel stays the same, and only `Template::unit_ids` keeps the old names, for the update
census. The GBC CPU stream's layers keep 9,344 of their 88,704 unit slots.

`IntLayer::with_blocks` then lays each channel's readers out as blocks. Readers at most
four units apart share a block, with weight 0 for the units between (their gates do not
move, so they send nothing). A group too sparse to pay (two vectors cost about what three
lone updates do) stays as lone entries. The build checks that every gate entry is in
exactly one block lane or lone entry with its weight. `Marks::blocks` updates a block four
gates at a time with NEON (`vmlaq`), computes the four activation changes (plain units and
fused pairs selected by a per-unit mask), and left-packs the lanes that changed into the
send list with one table shuffle (`PACK`). It updates lone entries as before.
`IntLayer::start_padded` pads the gates by three for a block's last vector. Where there is
no NEON (wasm), the same block runs one lane at a time. The scalar walk stays for the
update census (`GBCFORMER_UNITS`), and
`engine::tests::block_walk_sends_what_the_scalar_walk_sends` checks that the two walks
produce the same sends and gates on random packed layers, layer call by layer call (it
fails when a pair lane's sign is flipped).

Checks. `cargo test --release --test checkpoint_sst` (the engine against the dense pass,
the checkpoint against all 500,000 SST cases) has 7 passed. In Yellow five frames with
every component in weights and in lockstep, one model and `GBCFORMER_SCAN=1`, the scan
checked 59,908 GBC CPU and 1,392 PPU lookups with 0 differences. That record against
milestone 1's llama.cpp record has 601 events, 0 differing, and 40,068 decoded tokens. No
weight changed, and the GGUF and the digests are as before.

| workload | side | fps per run | median | CPU step, µs | shared layers, µs | load |
| - | - | - | - | - | - | - |
| text | A, milestone 2 | 34.9, 34.9, 34.8 | 34.9 | 5.33, 5.33, 5.34 | 4.32 | 3.4 to 4.1 |
| text | B, blocks | 38.8, 38.7, 38.7 | **38.7** | 4.79, 4.80, 4.81 | 3.78 | 3.6 to 4.6 |
| overworld | A, milestone 2 | 70.4, 69.3, 70.0 | 70.0 | 7.18, 7.29, 7.22 | 4.22 | 3.8 to 5.1 |
| overworld | B, blocks | 72.1, 73.0, 73.7 | **73.0** | 7.01, 6.92, 6.86 | 3.74 | 4.0 to 5.7 |

The gate updates are the same 2,097 a pass. The walk does them in about 0.55 µs less.

The variants below were rejected. Each was exact, and each measured slower or no faster.

| variant | measured | why it lost |
| - | - | - |
| the unit compaction alone, scalar walk | text pilot, shared layers 4.21 against 4.24 µs | the walk is not bound by where the gates sit |
| every reader as a one-lane vector (no lone entries), greedy windows | 37.3 fps against the blocks' 37.4 (three rounds) | more vectors (824 a pass, 2.6 used lanes each) |
| windows aligned to four units | text pilot, shared layers 3.92 against 3.73 µs | more vectors |
| the layer's vectors listed flat first, one loop | replay 3,479 against 2,928 ns a pass | the list stores cost more than the loop exits |
| the next change's gates and weights touched ahead | replay 3,572 against 3,177 ns | |
| sends staged at fixed slots with a lane bitmap | replay 2,947 against 3,029 ns, within noise | |
| **groups**: units with the same inputs evaluated as one exact table over the inputs' declared ranges (built from the integer weights, the range check as guard) | 36.3, 36.3, 36.2 fps against the blocks' 38.4, 38.4, 38.6 | they absorb 946 of 2,154 updates a pass, but those were the decoder's full blocks, the cheapest updates the walk makes; the table bookkeeping cost more |

**The GBC CPU pass on two threads, not built.** The layers run in order, and a layer's
work averages about 0.2 µs (the largest has 623 of the 2,097 updates). A hand-off between
threads per layer costs about as much, and the machine's other cores are already busy
with the PPU, the APU, the positional parts and other jobs.

### 9.3 The write-record memo per engine, and its guard

**Sized per engine.** §8.4's memo was 2^16 entries (about 70 MB) in every memory engine
from the start, and the cartridge's insertion filled it with lines that are rarely
written again as they were inserted. In this milestone the insertion builds its rows
without the memo (`Cells::fill`), and the memo starts at 2^10 entries (about 1 MB) with
the first write record. It grows four times over each time it has missed as many records
as it holds, up to 2^16 (`Cells::grow` moves the entries to their new slots). On text the
GBC CPU stream's memo reaches 2^16 entries and serves 885,914 records (888,081 before; the
difference is the inserted lines it no longer holds). The PPU stream's memo stays at 2^12
entries, about 4 MB instead of 70, and serves 36 (55 before). The guard is the same as in
§8.4.

Checks. In Yellow five frames with every component in weights and in lockstep, one model
and `GBCFORMER_SCAN=1`, the scan checked 59,908 GBC CPU and 1,392 PPU lookups with 0
differences, and the record against llama.cpp's has 601 events, 0 differing, and 40,068
decoded tokens. That run has no memo hit now, because its 648 hits in §8.4 were records
equal to inserted lines. Text's 13,842 rebuilt hits, none differing, exercise the guard.

| workload | side | fps per run | median | CPU step, µs | load |
| - | - | - | - | - | - |
| text | A, §9.2 | 37.9, 38.0, 32.8 | 37.9 | 4.91, 4.89, 5.67 | 4.0 to 4.9 |
| text | B, sized memo | 37.9, 37.4, 38.3 | **37.9** | 4.91, 4.98, 4.86 | 4.4 to 5.2 |
| overworld | A, §9.2 | 72.6, 73.4, 64.6 | 72.6 | 6.96, 6.89, 7.82 | 4.4 to 5.5 |
| overworld | B, sized memo | 73.3, 72.1, 72.1 | **72.1** | 6.89, 7.01, 7.00 | 4.3 to 5.2 |

Speed did not change either way (one run on each side fell under load). The gain is the
PPU stream's 66 MB.

**The guard compares every hit by its tokens and rebuilds every 64th.** The project's rule
for the memory is "guarded on every read". A memo hit is not a read, since no pass's
attention comes from it. It is the rows of a write record, which
`RowBuilder::group_rows` computes from the weights and the record's tokens alone. On every
hit, the exact comparison of the hit's byte address and all 8 line bytes with the record
the rows were built for ties a served row to the weights (the slot is a hash, but the
comparison is not). Rebuilding the rows on every hit as well would test that `group_rows`
is a pure function of those tokens, which the code makes true by construction (it has no
state besides the weights and the record's line turn, itself a function of the line).
Every 64th hit still runs that test and stops the run on a difference. Rebuilding every
hit was measured and costs more than the 1% the brief allowed:

| side | fps per run | CPU step, µs | load |
| - | - | - | - |
| every 64th hit rebuilt | 38.4, 37.6 | 4.85, 4.94 | 4.8 to 5.6 |
| every hit rebuilt (`MEMO_CHECK = 1`, a pilot build) | 34.1, 34.7 | 5.46, 5.36 | 4.3 to 4.7 |

Rebuilding every hit costs about 10% of the GBC CPU's thread (0.5 µs a step, 885,914
rebuilds of about 1.8 µs each), so the guard stays as it was. That pilot rebuilt and
compared all 885,914 hits of the text workload, and none differed.

### 9.4 Milestone 3 against milestone 2, and what stands between text and 60 fps

Against milestone 2's last binary the levers measure as in §9.2's table, text
from 34.9 to 38.7 fps and overworld from 70.0 to 73.0. §9.3's memo sizing changed no
speed.

**Where the GBC CPU's thread spends its time after milestone 3** (text, `sample` of the
final binary, 5,009 samples of the thread). Its unsampled step is about 4.8 µs, of which
the engine's pass is about 4.1 (layer 0 0.19, shared layers 3.78, decode 0.14).

| piece | share of the CPU's thread | µs per step |
| - | - | - |
| `Marks::blocks`, the gate walk | about 48% | 2.3 |
| of which the block updates (NEON) | 27.6% | 1.3 |
| of which the lone entries | about 9% | 0.45 |
| of which the send loop | about 6% | 0.3 |
| of which each change's setup | about 4% | 0.2 |
| `Engine::advance`'s own loop (routing the layer-0 changes, the head bucket, the layer loop) | about 8% | 0.4 |
| `Marks::drain_wires` | about 7.6% | 0.37 |
| `Marks::route_layer` | about 4% | 0.2 |
| layer 0, decode | about 7% | 0.33 |
| write records (`Engine::write_record`, `Cells::put`) | 2.8% | 0.14 |
| the host machine and the control pass | about 2.5% | 0.12 |
| the rest (the step protocol, `fill_merged` and `fill_into`, `GBCFORMER_STATS` clocks) | about 20% | 1.0 |

**The gap.** Text needs 5,374 GBC CPU steps a frame, so 60 fps needs at most 3.1 µs a
step, and the step is 4.8. If everything but the walk stays where it is (about 2.5 µs),
the walk would have to fall from 2.3 to about 0.8 µs, three times less again. The
counters of §9.1 show why layout levers give less and less. About half the cycles are
already useful work at a high issue rate, a fifth are lost to mispredicted branches on
loops whose trip counts vary with every change (blocks, lone entries, reader bits), and
the circuit fixes the work per pass (211 changes, 2,097 updates, about 190 sends). No
single piece outside the walk is worth 0.2 µs to a cheap change. `drain_wires`,
`advance`'s loop and `route_layer` are each 0.2 to 0.4 µs of the same kind of short
data-dependent loops, and moving the host machine and control off the thread was measured
not to pay (§8.5).

Two changes could still close the gap. Neither is built.

- **A GBC CPU stream that moves fewer channels per step.** This is a compiler change.
  Fewer changes and updates per pass is the only lever that shrinks every piece above at
  once. It needs every gate and digest rule of `CLAUDE.md` and a re-exported, re-verified
  GGUF.
- **Memoising whole GBC CPU passes by their input tokens and the lines their heads read**
  (§7). The project rules it out, and it is not built.

**Checks on the final commit.** `tools/gate.sh` is green, with every step `ok`
(load 7.1 to 14.7 at its start). Yellow five frames in lockstep with the scan oracle check
59,908 GBC CPU and 1,392 PPU lookups with 0 differences, and that record against
milestone 1's llama.cpp record has 601 events, 0 differing, and 40,068 decoded tokens.
`examples/cells_state` against llama.cpp's `yellow-cache-B.state` finds 0 of 141,415
cells that differ as f16 numbers in any of the 15 layers' key or value. The one model's
digest is `b03d896c`, the GGUF's.
