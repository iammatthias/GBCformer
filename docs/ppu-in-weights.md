# PPU in weights: the scanline checkpoint

This document puts the PPU's rendering in weights (`CLAUDE.md`, "Each component in weights, and its gate").
Sections 1 to 5 are the step-1 prototype, a one-scanline circuit built on the memory
checkpoint's head primitive, held in lockstep against `machine::ppu` line by line, with
the cost of several shapes of the same logic measured on the engine
(`examples/ppu_proto/`, throwaway and since removed, so the commands below no longer run). Section 6 is the shape decision, the pixels as 160
output positions of one pass per line (§3b, §4a), which needed the position wires and
the engine's per-position stage. Section 9 is the build, the scanline checkpoint as the
machine's renderer on its own thread, with its own VRAM and OAM image fed by the CPU
side's records (`src/ppu_circuit/`, `--weights ppu`).

## 1. What the prototype is

`examples/ppu_proto/scanline.rs` builds a `Circuit` (the IR of `circuit.rs`, memory
heads included) whose inputs are the bytes `render_line` reads and whose outputs are
pixels before the palette lookup:

| Slot | Bytes |
| - | - |
| inputs | `LCDC SCX SCY LY WX WY WLN FLG`; `WLN` is the PPU's `window_line`, `FLG` bit 0 its `window_armed` and bit 1 CGB mode (the register-head form of §5 would read the CGB bit from `REGS + MAPPER` instead) |
| outputs | one byte per pixel in the `machine::ppu::Pixel` encoding: colour index in bits 0-1, palette in bits 2-4 (CGB: the map or OAM palette; DMG: OBP1 selected in bit 2), bit 5 set when an object won |
| memory | VRAM (both banks) and OAM at their fixed physical addresses (`memory::layout`), read by `Circuit::head` queries |

`machine::ppu::render_line` now computes the same `Pixel` per pixel (`Ppu::pick`) before
its palette lookup (then `Ppu::shade`, now the free function `machine::ppu::colour`), and calls an optional `shadow` function with a
`LineTrace` (registers, window state, VRAM, OAM, pixels) for every visible line. The
reference renderer's frames are unchanged (DMG 8 `6872b003`, CGB 300 `cfd31f3d`, both
acid2 ROMs pixel-exact, dmg_sound 01 `99793856`).

The circuit follows `render_line` stage by stage, all of it 0/1 wires and threshold
nodes:

| Level or layer | Logic |
| - | - |
| level 1 | background column adders `(SCX >> 3) + k mod 32` for the 21 tile columns a scrolled line spans, row `SCY + LY`, the fine-scroll one-hot `SCX & 7`; the window as a background whose `SCX' = ¬WX + 8` (its map x is `x − (WX − 7)`), so its columns are `(¬WX >> 3) + 1 + k` and its fine shift `¬WX & 7`; `LY == WY`; `WX ≤ 166`; "background on" (`LCDC.0 ∨ CGB`) |
| layer 1 | 40 constant-address heads read every object's Y |
| level 2 | window enabled; per object, `0 ≤ LY + 16 − Y < 8 + 8·LCDC.2` (two steps) |
| level 3 | the first ten visible objects (`10·vis_i − Σ_{j<i} vis_j + 10·LCDC.1 ≥ 11`, one step each); `x ≥ WX − 7` per pixel |
| level 4 | slot k's OAM index, bit by bit, as a sum over objects of `[chosen_i ∧ #chosen before i == k]`; slot k exists (`Σ chosen ≥ k + 1`) |
| layer 4 | ten heads read each slot's X |
| level 5 | the priority order: for slots p < q, `[X_p ≤ X_q] ∧ valid_q` on DMG, `valid_q` on CGB, one step each (45 nodes); a slot's rank is an affine sum of these |
| level 6 | the sorted slot's OAM index bits: `Σ_k [valid_k ∧ rank_k == r ∧ bit]` |
| layer 6 | 40 heads read the sorted slots' Y, X, tile and attribute bytes |
| level 7 | the object's row (`LY + 16 − Y`, flipped by attr.6 over 8 or 16 rows, the flip folded into the adder's staircase), the tile's bit 0 giving way to the row's bit 3 for 8x16 objects, the CGB bank, the `X & 7` one-hot |
| layers 2–5 | the background and window tiles: map and attribute heads (layers 2 and 3), tile-row address nodes (signed addressing by `LCDC.4`, the Y flip folded into the row bits), two data heads per tile (layers 4 and 5); placed so no layer holds more than 52 heads |
| layer 7 | 20 heads read each slot's two row bytes |
| level 8 | each slot's row shifted by `X & 7` into 16 fine positions: two colour bits and an opaque bit per position |
| level 9 | per pixel and slot: the slot covers the pixel (its coarse `X >> 3` is the pixel's tile column or the next one) and is opaque there |
| level 10 | per pixel and slot: the slot is the first opaque one in priority order (`10·opaque_s − Σ_{t<s} opaque_t ≥ 10`) |
| level 11 | per pixel: the winning slot's colour, palette (CGB attr.0-2 or DMG attr.4) and priority bit; "an object is present" is the affine sum of the winners |
| levels 6–7, 12 | per pixel: the background colour and attribute bits (a mux over the fine shift, the horizontal flip and background-or-window), then the composition: object shows unless (DMG) its priority bit is set over a non-zero background, or (CGB) `LCDC.0`, a non-zero background and either priority bit |

Three shapes share every stage but the per-pixel ones:

- **Wide(n).** Pixels `0..n` are outputs and every screen x is a constant, with one pass
  per line (`n = 160`). Smaller `n` compile a smaller checkpoint whose per-pixel-stage cost
  extrapolates.
- **Tile.** Pixels `8·X..8·X + 7` are outputs with `X` an input slot, at 20 passes per
  line, with two background and two window tiles fetched per pass.
- **One.** Pixel `X` is the output with `X` an input slot, at 160 passes per line, with
  one tile per layer fetched and objects composed directly by `X_s − x`.

## 2. Exactness (measured)

`cargo run --release --example ppu_proto -- pokemon-yellow.gbc --frames N [--hardware dmg|cgb] [--tile | --one | --pixels n] [--engine] [--stride k]`
installs the shadow, re-renders every visible line (every k-th with `--stride`) by
`Circuit::eval_mem` over the line's VRAM and OAM, and panics at the first pixel that
differs.

| Shape | Run | Lines equal to the reference |
| - | - | - |
| Wide(160) | Yellow, DMG, frames 0–600 | 84,950 of 84,950 |
| Wide(160) | Yellow, CGB, frames 0–600 | 84,950 of 84,950 |
| Wide(160) | Yellow, CGB, text.sh's presses, frames 2300–3500 (text boxes: the window) | 172,800 of 172,800 |
| Wide(160) | Yellow, DMG, the same presses and frames | 172,800 of 172,800 |
| Wide(16), Wide(32), Tile | Yellow, CGB, frames 0–600, every second line (the §4 runs) | 42,475 of 42,475 each |
| Tile | Yellow, CGB, frames 2300–3500 with the presses, every 7th line | 24,685 of 24,685 |
| One | Yellow, CGB, frames 0–600, every 23rd line; frames 2300–3500 with the presses, every 47th line | 3,693 of 3,693; 3,676 of 3,676 |

The 600 boot frames cover the copyright and intro screens (object-heavy, scrolling),
the title and the start of the game, all with `LCDC.4` set; the text.sh run covers
Oak's speech, drawn with `LCDC.4` clear (signed tile addressing) and with the window.
That run found the one defect of the prototype. For tiles 128..255 under signed
addressing the row address is `tile·16`, so address bit 11 is `tile.7` in both modes
(the first build had `LCDC.4 ∧ tile.7`). The first diverging pixel was frame 2300, line
66, pixel 32 on CGB and line 90, pixel 54 on DMG, and the §3 circuit is the fixed one.
This is the kind of defect the lockstep exists for, and the reason the §2 runs before
any build must include every `LCDC` mode a game uses.

## 3. Size (measured on the IR, projected on the checkpoint)

The circuit statistics come from `ppu_proto`, and the checkpoint statistics from
`Model::compile_mem` where it fit in memory (the dense weights are f32, and the Wide(160)
checkpoint would be about 8 B dense parameters, 32 GB, so it was not compiled; see §4).

| Shape | Wires | Nodes | Steps | Depth | Heads (max per layer) | Steps at the widest level | Nonzeros (steps and heads, counted on the IR) |
| - | - | - | - | - | - | - | - |
| Wide(160) | 9,769 | 7,481 | 67,968 | 12 | 278 (52) | 20,000 | about 578 k |
| Tile | 2,291 | 1,211 | 10,499 | 12 | 126 (40) | 3,840 | about 122 k |
| One | 1,432 | 416 | 5,004 | 11 | 118 (40) | 1,961 | about 103 k |

(After the signed-addressing fix of §2. The §4 timings ran the circuit before it, which
had one more one-step node per fetched tile, 42, 4 and 2 wires more, no change to the
heads, the depth or the widest level.)

The CPU circuit is 1,916 wires, 1,636 nodes, depth 9, 16 heads (unit H).

**Hidden width.** The layout is `n_const + positions + 8 + wires + 37 + 12`, rounded up to
a multiple of 1,024 (`llama::Layout::used`). The CPU alone sits at 2,041 of 2,048. One
checkpoint holding both, as the design requires:

| Shape added to the CPU | Wires | Positions | Hidden |
| - | - | - | - |
| Wide(160) | 11,685 | 66 + 168 | 12,288 |
| Tile | 4,207 | 66 + 17 | 5,120 |
| One | 3,348 | 66 + 10 | 4,096 |

**Dense parameters** scale as `layers · (4·a·d + 3·d·i)` with `a = 32 · max heads per
layer` and `i` the widest level's MLP units (`Config::params`; a step whose form can
exceed its threshold is two units, so `i` is about 1.6× the IR's step count, as Wide(32)'s
4,000 steps compiled to 6,400 units and Tile's 3,840 to 5,120). The table projects the merged
checkpoint at 13 layers (depth 12 of the PPU, one more than the CPU's 10), with the
standalone checkpoints the prototype did compile beside them (§4):

| Shape | `d` | `a` | `i` | Dense parameters, merged | Compared with unit H (293.7 M) | Standalone, compiled |
| - | - | - | - | - | - | - |
| Wide(160) | 12,288 | 52·32 = 1,664 | about 32,000 | about 16 B | 55× | not compiled: about 8 B at hidden 10,240, 32 GB of f32 |
| Tile | 5,120 | 41·32 = 1,312 | 5,120 | about 1.4 B | 4.6× | 854.7 M (hidden 3,072) |
| One | 4,096 | 41·32 = 1,312 | 3,840 | about 0.9 B | 3× | 433.2 M (hidden 2,048) |

**Depth.** The PPU is 12 levels (13 layers) against the CPU's 9 (10 layers); merged, the
checkpoint takes the deeper one. The engine's reader mask (`Readers = u16`, one bit per
circuit layer, the routing layer and the head bucket) allows 14 circuit layers, so
depth 12 leaves two levels of slack for either circuit.

Nonzeros add the PPU's count to unit H's 203,941, giving about 780 k (Wide, from the IR
count), 503 k (Tile, compiled), 470 k (One, compiled). The ratio of dense to nonzero is
what the wide shape costs. A 160-pixel line is 160 copies of the pixel logic in distinct
wires, every wire is a residual channel, and every channel meets every MLP unit. At
32,000 units the wide shape also touches the engine's limit of 32,768 units per layer
(`IntLayer`), which a level split would relieve.

**Channel overlay** (CPU and PPU wires sharing residual channels, gated by a mode wire)
does not change the picture. The PPU's own wires set the hidden size in every shape
(9,769 against the CPU's 1,916 for Wide), so overlaying the CPU's channels onto the
PPU's saves at most one 1,024 block. It would also need a new plan rule in
`engine::Template::check` (today one writer per channel) and a gate term in every step
of both circuits. The saving does not justify building it.

### 3a. Re-projected under the scheduled compiler

The compiler now schedules nodes across levels (`Circuit::schedule`), gathers four input
slots per layer-0 head and puts the copy head on the emptiest layer (`docs/perf-log.md`,
"Size: the level schedule and the grouped gather"). `build` and `build_on` schedule every
scanline shape, so §4's timings predate the schedule. `ppu_proto --project` prints each
shape from `Model::config` without building weights; "with the CPU" builds the scanline
on top of `sm83::build_mem()` in one circuit and schedules the whole.

| Shape | Alone: hidden, `i`, heads, dense | With the CPU: hidden, `i`, heads, layers, dense | Wires live at most (merged) |
| - | - | - | - |
| CPU only | 2,048, 1,536, 10, 153.4 M | | 931 |
| Wide(16) | 5,120, 3,776, 52, 1.25 B | 6,144, 3,904, 52, 13, 1.59 B | 1,592 |
| Wide(32) | 5,120, 6,400, 52, 1.77 B | 7,168, 6,400, 52, 13, 2.55 B | 1,929 |
| Wide(160) | 10,240, 32,000, 52, 13.8 B | 12,288, 32,000, 52, 13, 16.7 B | 3,902 |
| Tile | 3,072, 3,776, 40, 689 M | 5,120, 3,904, 50, 13, 1.31 B | 1,397 |
| One | 2,048, 3,776, 40, 425 M | 4,096, 3,904, 50, 12, 975 M | 1,342 |

The schedule barely moves the scanline. Its widest level (3,744 units at level 4 in every
shape, 32,000 at level 11 in Wide(160)) is rigid, because those nodes read a head's byte
one layer down and feed a head's query one layer up. Its 40 to 52 memory heads per layer
set `a`, not the gather. Merged, layer 1 holds the CPU's 10 register heads beside the
scanline's 40. So the levers for the PPU are, in order, channel reuse (a wire's channel
freed after its last reader; the merged One and Tile need 1,342 and 1,397 live wires, so
hidden 2,048 instead of 4,096 and 5,120, about 490 M and 525 M dense parameters at the
same `i` and heads), fewer scanline heads per layer (fetches spread over the layers or
packed several bytes to a head), and splitting the rigid level's nodes by the step.

**With channel reuse** (`docs/perf-log.md`, "Size and speed: channel reuse and the
changed-pair placement"). `ppu_proto --project` now runs `Circuit::fit` on each shape with
the channel budget of the fewest 1,024-channel blocks the search reaches, and the layout
reuses dead wires' channels. Heads and depth are unchanged; `i` includes the clearing units.

| Shape | Alone: hidden, `i`, dense (was) | With the CPU: hidden, `i`, dense (was) | Wires live at most (merged, after `fit`) |
| - | - | - | - |
| CPU only | 1,024, 1,728, 82.6 M (153.4 M) | | 868 |
| Wide(16) | 2,048, 4,032, 519 M (1.25 B) | 2,048, 4,544, 582 M (1.59 B) | 1,592 |
| Wide(32) | 2,048, 6,400, 709 M (1.77 B) | 2,048, 6,400, 730 M (2.55 B) | 1,851 |
| Wide(160) | 4,096, 32,448, 5.58 B (13.8 B) | 5,120, 32,448, 7.03 B (16.7 B) | 3,902 |
| Tile | 1,024, 4,032, 240 M (689 M) | 2,048, 4,288, 555 M (1.31 B) | 1,397 |
| One | 1,024, 3,776, 213 M (425 M) | 2,048, 4,032, 497 M (975 M) | 1,342 |

The merged One and Tile land at hidden 2,048 as §3a predicted, at about half their dense
parameters; the clearing units widen `i` by up to 384 units. What sets their size now is the rigid level-4 width (3,744 scanline units plus
the CPU's) and the 50 heads at layer 1, not the residual.

### 3b. The positions shape

**Pos** runs one pass per line, with the pixels at 160 output positions that share the
per-pixel weights. Each position reads its own x from 160 position wires (`Circuit::positions`).
Wire k is the slot identity of output position k, so it is 1 there and 0 everywhere
else, and costs no wire channel. The tile column one-hot, the fine-pixel one-hot, x and
the column's value are affine in those wires and cost no level. The shared stages are
Wide's (21 + 21 fetched tiles, the ten objects and their fine buffers); the pixel stage
is Wide's with the constants read from the position:

- the two tiles a pixel can come from (its column's and the next) are selected from the
  fetched row by the column one-hot, one step per column, 84 wires;
- the background colour and attribute bits select the fine position by the fine-pixel
  one-hot over the fine scroll;
- the object stage places each slot's fine buffer by `X_s >> 3 − column`, the column
  read from the position.

At position x, every step whose position literal is 0 is constant 0, and what is left is
exactly Wide's pixel x. The engine uses that (§9.2).

| Shape | Wires (live at most) | Depth | Hidden | Intermediate | Heads | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - |
| Pos alone | 3,812 (1,117) | 12 | 2,048 | 3,968 | 52 | 514.8 M | 691,571 |
| One alone, for comparison | 1,432 (446) | 11 | 1,024 | 3,776 | 40 | 212.7 M | 269,246 |
| Wide(160) alone | 9,769 (3,522) | 12 | 4,096 | 32,448 | 52 | 5.58 B | about 578 k (IR) |

Pos is one tenth of Wide(160) and 2.4 times One. Units per level are 696, 162, 1,070,
3,753, 2,351, 3,440, 2,618, 3,752, 2,008, 10, 1,330, 137. The widest is level 4, which is the
shared object selection that sets every shape's width (§3a), so the pixel stage costs
the dense checkpoint no extra MLP width. The 52 heads per layer are Wide's shared
fetches, and with hidden 2,048 the attention is 177 M of the 515 M.

## 4. Time (measured on the engine)

The engine runs the compiled scanline as it runs the CPU: the circuit layers once per
pass, incrementally, heads served from the index when their query or the image changed,
every pass checked against the reference pixels. Two engine changes were needed, both
checks the CPU checkpoint never exercised. `MemPlan` refused more than 32 memory heads
(a `u32` dirty mask), now a bitset of as many `u64` words as the heads need; and a
layer-0 head slot that another layer fills but layer 0 leaves empty (the PPU has 41
heads at layer 1 against 12 at layer 0) made `plan_attention` refuse every pass, since a
live head with no keys has no softmax. A head whose `W_o` is zero now counts as dead,
which is exact, since it changes no channel whatever it attends to. The CPU checkpoint is
byte-identical (`examples/digest` `fe67fe79`); the A/B is in §7.

The compiled Wide(160) checkpoint does not fit, because 8 B f32 dense parameters are 32 GB
(the compile reached 24 GB of allocations before it was stopped). The wide shape was
compiled and timed at 16 and 32 pixels, with every shared stage (21 + 21 tiles, 40
objects, the ten slots) at full size, so the two points give the per-pixel-stage slope.

The runs are Yellow on CGB, frames 0 to 600, every second line (42,475 lines), with
`--engine`, single runs with another play window open (1-minute load 2.9 to 6.8
at the section starts). Each line's tokens equalled the reference pixels; no pass took the dense
fallback.

| Shape | Checkpoint (standalone) | Passes per line | Units per pass | Index reads per pass | µs per pass | µs per line | ms per frame (144 lines) |
| - | - | - | - | - | - | - | - |
| Tile | hidden 3,072, intermediate 5,120, 41 heads, 854.7 M dense, 298,965 nonzeros | 20 | 1,955 | 9.5 | 7.12 | 142.4 | 20.5 |
| Wide(16) | hidden 5,120, intermediate 5,120, 52 heads, 1,515.7 M dense, 363,595 nonzeros | 1 | 2,471 | 91.8 | 15.57 | 15.6 | 2.2 |
| Wide(32) | hidden 5,120, intermediate 6,400, 52 heads, 1,771.4 M dense, 460,587 nonzeros | 1 | 2,789 | 91.8 | 20.31 | 20.3 | 2.9 |
| Wide(160), projected from the two points | §3's 16 B dense merged | 1 | about 5,300 | about 92 | about 58 | about 58 | about 8.4 |
| One (every 23rd line, 3,693 lines) | hidden 2,048, intermediate 3,840, 41 heads, 433.2 M dense, 266,304 nonzeros | 160 | 2,728 | 0.7 | 7.73 | 1,236.5 | 178 |

**Explaining the numbers.** The tile shape's units are the object composition, `oc`
(the winning slot's colour per pixel, 40 steps over `X_s >> 3 − J`) and `op` (opaque per
slot and pixel) re-evaluate on every pass because the tile index `J` changes on every
pass (census: `oc*` 153 updates each, `op*` 57 each, 1,800 of the 1,955 units). That is
the per-tile shape's structural cost, because the sprite placement cannot be shared across the
20 passes of a line. The wide shapes change only `LY` between passes, so their units are
the line-to-line changes: the row adder, the slot index bits `si*`, the sorted index
`so*`, the fine buffers `fc*`/`fo*` (about 30 updates each), and 92 index reads (the tile
rows and the slot bytes whose addresses moved; the data heads re-read because their
row changed, not because the image did). The 16 extra pixels from Wide(16) to Wide(32)
add 318 units and 4.7 µs, 15 ns per unit against the CPU engine's 2.4 ns, because these units
are 40-step nodes with ten-term gate rows, and each pixel adds a decoded position. The
projection to 160 pixels is linear in that slope and is an estimate, not a
measurement; a Wide(64) point (about 10 GB to compile) would tighten it.

The 92 index reads per line were measured on frames that write VRAM and OAM rarely
(7,821 changed bytes over 600 frames, 13 per frame, DMA included). The engine
invalidates heads by the image's generation, so the line that follows a VRAM or OAM
write re-reads all 278 heads, about 12 µs more at 45 ns each. Yellow writes VRAM and
OAM in VBlank, so that is the first line or two of a frame, not every line; a game
that writes mid-frame pays it on every line after a write until the invalidation is
per address.

Against the budget, with 5,350 CPU steps per frame on text.sh:

| CPU pass | CPU alone | CPU + PPU, one thread | CPU + PPU, two threads |
| - | - | - | - |
| 7.15 µs (unit H) | 38.3 ms, 26 fps | 46.7 ms, 21 fps | 38.3 ms, 26 fps |
| 3.1 µs (the 60 fps target) | 16.6 ms, 60 fps | 25.0 ms, 40 fps | 16.6 ms, 60 fps |

The wide PPU costs about half a frame of a second core and nothing of the first; on one
thread it costs a third of the frame rate at either CPU speed. The tile shape's 20.5 ms
per frame exceeds a frame on its own. The per-pixel
shape is out on either thread, since 7.73 µs × 23,040 passes is 178 ms per frame, and its
units are the ten slots' direct `X_s − x` composition (`sc*`, `op*`, 127 updates each)
re-evaluated on every pass because `x` changes on every pass.

### 4a. Per position (measured)

A positions shape keeps one residual per output position, so what a pass costs is what
each position's units cost when the line changes, times 160, plus the shared stages
once. One at a fixed x measures that for One's pixel stage: `GBC_AT=37 ppu_proto
--one --engine` renders only pixel 37 of every line, so the engine's incremental state
is one position's from line to line. The run covers Yellow CGB frames 0 to 600, every line, at a 1-minute load of
about 9:

| Run | Units per pass | Index reads per pass | µs per line |
| - | - | - | - |
| One, x = 37 | 973 | 23.8 | 6.02 |
| One, x = 85 | 974 | 23.8 | 6.07 |

The census splits the 973 units into about 760 shared (`bgrow` 41, the slot index
`si*` 16 to 17 each, the sorted index `so*` 15, the object rows `sr*` 24, `vis*` 4) and
about 215 for the pixel (`op*` 9 to 11 each, `sc*` 7 to 9, the tile address `bg` 24,
`wn` 8, `px` 2). One's pixel stage composes each slot directly by `X_s − x` from the
slot's row bytes, which change on most lines, so 160 positions of it would be about
760 + 160 × 215 ≈ 35,000 units per line, about 215 µs at this run's 6.2 ns per unit,
31 ms per frame (inferred from the measured parts). Wide's pixel stage is cheap from
line to line because it reads the slots' fine buffers, which are shared, at 318 units
per 16 pixels (§4), 20 per pixel.

So the positions shape (§3b) takes Wide's pixel stage, and the engine runs each position
specialised to its x (§9.2), where each position's units are Wide's pixel x's.
Measured with `ppu_proto --pos --engine` over Yellow frames 0 to 3,499 with text.sh's
presses (the boot, the title, Oak's speech in the window), every line, both modes,
1-minute load 15 to 16 (other runs on the machine):

| Mode | Lines, all equal to the reference | Units per pass | Index reads per pass | µs per line | ms per frame (144 lines) |
| - | - | - | - | - | - |
| CGB | 500,596 | 4,099 | 86.5 | 20.02 | 2.9 |
| DMG | 497,178 | 4,278 | 86.6 | 17.16 | 2.5 |

No pass took the dense fallback. About 2,300 of the 4,100 units are the positional
stage's (the per-layer counts put them on the routing layer's line), the rest the
shared stages'. That is about 14 positional units per position and line, under Wide's
20 (most likely because Yellow's lines repeat more tile rows than §4's boot frames;
not separated).

## 5. Two threads, and what "as of the scanline" means in the model

The reference renders a line at dot 80 (`MODE3_START`), reading VRAM, OAM and the
registers at that moment. In the memory checkpoint, every CPU write is a kept token at
the step's cache position, so "VRAM as of the scanline" is "the newest write per address
among the tokens at positions up to the step that crossed dot 80". That is already what
attention computes when the scanline pass sits at that cache position. Causal
attention excludes later tokens and the recency term picks the newest write before
it, with the same margins as the CPU's reads (`memory::attn::Params`). In a stock stack
the scanline pass is a branch of the sequence, the cache prefix up to position `P_LY`
plus the scanline's own tokens, while the CPU's sequence continues from `P_LY` with its
next step. Prefix sharing between two sequences is what a serving stack calls prefix
caching, so nothing non-standard is needed.

Two threads then fall out, and each keeps the engine exact:

- The CPU thread runs CPU passes and keeps write records as today. When its block
  crosses dot 80 of a visible line it emits a **scanline request** with the registers the
  line reads (`LCDC SCX SCY LY WX WY`, the window counters) and the cache position
  `P_LY`.
- The PPU thread owns a second `Engine` state over the same weights (its own
  incremental deltas line to line, which is where the per-pass cost in §4 comes from)
  and its own VRAM and OAM `Image`. It receives the CPU thread's write records to VRAM
  and OAM (address, byte, position) over a channel and applies those with position
  `≤ P_LY` before rendering line `LY`. Its heads read that image. The image is exact by
  construction, since it holds the same records the cache holds, filtered by position, which
  is what the attention over the prefix computes. A snapshot copy (16.5 KB per line) is
  the alternative and costs more memory traffic than the records do (the §4 runs saw
  13 changed VRAM and OAM bytes per frame over Yellow's boot, DMA included; membench
  puts all of Yellow's writes at 249 to 429 bytes per frame).
- The dense oracle (`GBCFORMER_ORACLE`) samples lines and runs `forward_cached` over the
  cache **bounded at `P_LY`** (a visible-prefix argument the function does not take
  today; every call passes `kv.len()`) and compares the decoded pixels with the
  engine's. That is the guard for the position filter, as the sampled oracle is the
  guard for the index.
- In lockstep the PPU thread compares its 160 pixels with the reference line
  (`LineTrace`) and stops at the first difference, as the CPU lockstep does.

Two prerequisites the memory design left to the host, both needed before any in-model
PPU can see what Yellow draws. The first is now built:

1. **OAM DMA and HDMA copies are records** (built, unit J, `docs/memory-in-weights.md`
   §8a). Each copied byte is a reader record at the source and a write record at the
   destination, so the cache's OAM and VRAM equal the host's after every DMA. The
   memory audit shows it. With the audit and no records, lockstep-mem CGB 300 stopped
   at Yellow's first OAM DMA, and with the records every lockstep-mem gate passes with
   VRAM and OAM compared after each DMA and every writable region at each frame end.
   A PPU thread applying records up to `P_LY` sees DMA'd sprites the same way it sees
   CPU writes. The cost is one record per byte, 156 per frame in Yellow (perf-log "DMA
   copies as records"). The DMA schedule (which bytes, when) is still host control
   logic, a Phase 3 item.
2. **Palette RAM** (CGB, 128 bytes behind `FF68-FF6B`) is PPU state the host owns. The
   prototype outputs palette indices and leaves the lookup to the host, which the
   Phase 1 gate allows ("colour indices + palette"). Moving the lookup into the model
   costs two heads per pixel on the wide shape (320 at one layer) or two per pass on
   the others, once palette RAM is in the cache.

## 6. Decision

**Build the positions shape (§3b) as a checkpoint of its own, on its own thread.** The
rule was the smallest checkpoint that renders a frame in about 12 ms on a thread of its
own:

| Shape | Passes per line | Dense parameters (alone) | ms per frame on its thread | Source |
| - | - | - | - | - |
| One | 160 | 212.7 M | 178 | §4, measured |
| Tile | 20 | 239.7 M | 20.5 | §4, measured |
| Positions with One's pixel stage | 1 | about 213 M (inferred) | about 31 | §4a, measured per position, multiplied |
| Wide(160) | 1 | 5.58 B | about 8.4 | §4, projected; does not compile in 64 GB |
| **Pos** | **1** | **514.8 M** | **2.5 (DMG) to 2.9 (CGB)** | **§4a, measured, every line of 3,500 frames** |

Pos meets the time with room at a tenth of Wide(160)'s size. Tile and One are smaller
and miss the time; Wide(160) meets it at 5.58 B. What made Pos possible is two pieces
the prototype lacked: position wires in the circuit IR and the engine's per-position
stage (§9.2), which runs each position's units specialised to its x so that a pass
costs what Wide's would.

The two-thread design of §5 stands, with one change. The PPU is a checkpoint of its
own, not a mode of the memory checkpoint. It holds its own VRAM and OAM image and
follows the CPU side's write and DMA records, which is the "image filtered by
position" of §5. A merge into the CPU's checkpoint would cost the CPU side the PPU's
heads and width on every step for no gain the separate checkpoint lacks (the records
are already the cache's), and one engine state cannot run on two threads
(`docs/apu-in-weights.md` §11.6 makes the same case for the APU).

The first decision, from §3 and §4 before the positions shape existed, was to stop
after step 1, because the only shape with a real-time path was Wide(160), at about 16 B merged.
That decision's analysis (width caps paid in passes, fewer wires per pixel) is in the
history of this file, and the numbers it rests on are §3 and §4 above.

## 7. Gates

Run on the committed tree (`tools/gate.sh`, plus the memory-core hashes the gate script
leaves to `--core lockstep-mem`), with the host checkpoint byte-identical
(`examples/digest` `fe67fe79`, 187.4 M parameters, 68,004 nonzeros):

| Gate | Result |
| - | - |
| build, no warnings; wasm32 check of the library | ok |
| `cargo test --release` (SST for the circuit, the checkpoint and the memory engine, `tests/memory_dense`, E01, interrupt fuzz) | ok |
| `run --frames 8 --core lockstep --hardware dmg` / `lockstep-mem` | `6872b003` / `6872b003` |
| `run --frames 300 --core lockstep` / `lockstep-mem` | `cfd31f3d` / `cfd31f3d` (host served 11,688 of 902,094 steps) |
| Crystal, Zelda DX, Mario DX, 1,500 frames lockstep | `4e6a7509`, `1d00d871`, `7dec3947` |
| text.sh, overworld.sh (`model` and `model-mem`, every A/B run) | `870aa3fd`, `822059c1` |
| blargg `cpu_instrs`, `instr_timing`, `dmg_sound` 01–04 | Passed; `99793856`, `881f3fe7`, `e494d943`, `8e776b75` |
| dmg-acid2, cgb-acid2 | pixel-exact |
| `model` core no slower (ab.sh, 3 runs) | text.sh 2.43 → 2.43 µs per pass, overworld.sh 2.88 → 2.86 (perf-log, "Phase 1 prototype") |
| `model-mem` core no slower (ab.sh, 3 runs) | text.sh 6.94 → 7.01 µs, overworld.sh 6.83 → 6.92: 1.0 to 1.3% slower in 3 of 3 pairs, units and index reads per pass unchanged; recorded as the cost of the wider head mask, not a tie |

## 8. Reproducing

```sh
cargo build --release --all-targets
./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 600 --hardware dmg          # Wide(160), every line
./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 600 --hardware cgb
./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 120 --hardware cgb --tile --stride 3 --engine
./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 120 --hardware cgb --pixels 16 --engine
./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 300 --hardware cgb --one --stride 23 --engine
GBC_AT=37 ./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 600 --hardware cgb --one --engine   # One at a fixed x (§4a)
GBC_NOIR=1 ./target/release/examples/ppu_proto pokemon-yellow.gbc --frames 3500 --hardware cgb --pos --engine \
    $(for f in $(seq 2300 40 3500); do echo --press $f:a:4 --press $((f+20)):down:4; done)          # Pos (§4a); GBC_NOIR skips the slow IR check
./target/release/examples/ppu_proto --project                                                    # every shape's checkpoint from Model::config
```

## 9. The build: the scanline checkpoint as the machine's renderer

`--weights ppu` (with `cpu` and `memory`) compiles the positions shape into its own
checkpoint and draws every visible line with it; `--lockstep ppu` holds every line's 160
pixels to `machine::ppu`'s, the first difference stopping the run. Three things stay on
the host, named in every run's label. They are the palette lookup (BGP, OBP0, OBP1 and CGB
palette RAM stay host state), the window line counter (two bits of PPU state the line's
inputs carry), and the LY/mode/STAT timing that decides when a line is drawn, which the
control phase owns. §10 moves the first two into the checkpoint, and `docs/control-in-weights.md` §14 moves the timing into the control checkpoint.

### 9.1 Shape

`src/ppu_circuit/` holds the circuit (`circuit.rs`, the Pos shape of §3b moved from the
prototype with no knobs), the checkpoint and the frame drawing (`mod.rs`) and the thread
(`worker.rs`). `examples/ppu_proto` builds Pos from it and keeps its own copies of the
other shapes as the measurement instrument of §4.

- **Inputs and outputs.** The eight bytes of §1, then 160 output slots `PX0..PX159` that
  all route the same eight wires, then the six write-record slots the memory compiler
  expects, which a scanline leaves zero. One pass decodes the 160 pixel positions.
- **The image.** `PpuCheckpoint::new` inserts an image holding VRAM and OAM as the
  machine's memory has them at the hand-off, and ROM byte 0 (the copy head's query is
  the dead address 0). It holds no other ROM, because the scanline reads nothing else.
- **`PpuModel`** draws a line, looks its pixels up in the line's palettes into its back
  frame, and under lockstep compares them with the reference's first.

### 9.2 The per-position stage

The engine's template path evaluates the circuit layers once per pass, which is exact
because every position holds the same wires. A position wire breaks that on purpose, so
`engine::Positional` takes its readers out:

- **Which units.** A circuit unit is positional when its gate reads an output slot's
  identity or a channel a positional unit writes, and so is every unit writing a channel
  a positional unit writes (a node's units move together). A routing unit is positional
  when it reads a positional channel. The template keeps the rest unchanged; a checkpoint
  with no position wires has no stage, and the CPU checkpoint's digest is unchanged
  (`1f4c3519`).
- **Specialised per position.** For each output position whose field is positional, each
  positional unit is re-emitted with that position's slot channel (1) folded into its
  bias and the other slots' (0) dropped. A unit whose gate cannot reach 1 over its
  inputs' declared ranges (`Model::wire_ranges`, the circuit's `ranges`) is left out at
  that position, since its activation is 0 there whatever the inputs. At position x what is
  left is Wide's pixel x (§3b).
- **The guards.** That pruning holds only while the wires it read stay in their declared
  ranges. Every shared channel a positional unit reads, pruned or not, gets its declared
  range as its range check, so a pass that takes one out of range refuses and runs dense;
  the stage checks its own channels the same way. The plan is refused when a memory
  head's query reads a positional wire (the index serves one address per head, a
  positional query has one per position), and when a positional routing unit could fire
  at an output position the stage does not run (its slot identity 0 there). A refused
  pass clears every pending change, so the fresh pass after it starts clean.
- **The pass.** After the shared layers, the pass's changes to the shared channels the
  stage reads go to each specialised position's units, layer by layer, as the template
  does; the positional routing units write per-position route totals that the decode adds
  to the template's. Its channels are the shared virtual ones, then one block per
  position of that position's positional and route channels (u16 indices, so at most
  65,536 in all, which the plan checks).
- **What it costs other checkpoints.** One more reader bit (the mask stayed a `u16` then, 12
  circuit layers, the routing layer, the head bucket and the positional bucket; §10 widens it to `u32`),
  one option checked per decoded position, and one per memory write record for the
  video queue.

`tests/ppu_checkpoint.rs` holds the engine to the dense pass over the KV cache (the
stock-stack oracle) and to the circuit, on lines over random VRAM, OAM and registers with
random writes between them.

### 9.3 The records, and the order that filters them

§5's "VRAM as of the scanline" is the newest record per address among those kept before
the line's mode 3. The memory checkpoint already keeps every CPU write and every DMA byte
as a record (`docs/memory-in-weights.md` §8a); `engine::Memory::video` is a queue the CPU
engine appends each VRAM and OAM record to as it keeps it. The CPU's thread drains the
queue when the host PPU reaches mode 3 of a visible line and hands the records and the
line over together, so the PPU's image applies exactly the records kept before that
point and none after. The host's order is the same. A step's writes are kept during the
step, before `Machine::advance` reaches mode 3, and the DMA bytes a block copied reach the
memory checkpoint before the next block runs (`Machine::advance_handing`), as the host
applies them after the block; a general-purpose HDMA's bytes, which a bus write lands,
reach it before the step's advance. Handing them over once per step instead, as the
first build did, left a line drawn later in the same advance (or in the HDMA stall)
without them. `tests/ppu_dma.rs` (general-purpose HDMA over tile 0, back to back)
stopped at frame 0, line 16 before the change. The lockstep runs below compare every line with
the reference renderer reading the host's VRAM and OAM at the same point, so an ordering
mistake shows as a pixel difference.

The stack rule stays: `ppu` needs `memory`, because the records are the memory
checkpoint's. Without memory in weights there are no records to follow, and a host-fed
image would be a second source of VRAM bytes beside the model's.

### 9.4 Its own thread

`worker::Worker` runs `PpuModel` on a thread named `ppu`. The machine's thread sends the
records and the line and runs on; it waits only at VBlank, where the PPU's front frame
is the worker's back frame as of every line handed over. A lockstep divergence panics on
the worker, and the machine's thread re-raises it at its next send or wait
(`a_divergence_on_the_ppu_thread_stops_the_caller`). `GBCFORMER_PPU_THREAD=0` keeps the
checkpoint inline; wasm32 always does.

### 9.5 Size (measured)

`cargo run --release --example shape -- --ppu` reports 3,812 wires (1,117 live at most),
1,364 nodes, depth 12 and 278 memory heads. The checkpoint has hidden 2,048,
intermediate 3,968, 13 layers, 52 heads, vocabulary 5,042, 514.8 M dense parameters and
691,571 nonzeros, with the integer fast path on. Of the dense parameters, attention is 177 M (52 heads per layer,
from the shared fetches), the MLPs 317 M, the embedding and head 21 M. The positional
stage adds nothing to the dense checkpoint, because it is how the engine runs the same
weights.

The checkpoints in a full stack today are the CPU and memory 82.6 M, the PPU 514.8 M, the
APU 606.4 M, plus the control checkpoint (`docs/control-in-weights.md`).

### 9.6 Gates (measured)

On the final binary, from scratch copies of `tools/gate.sh`, `tools/acid2.sh` and the
bench scripts with absolute paths (the worktree holds no ROMs or `vendor/`):

| Gate | Result |
| - | - |
| 300 frames, `--weights cpu,memory,apu,ppu --lockstep all`, against `--core lockstep` | Yellow CGB `cfd31f3d` (41,750 lines equal), Yellow DMG 8 frames `6872b003`, Crystal `ccfe7ad3` (41,777), Zelda DX `7b126eb8` (40,756), Mario DX `3e1f4d27` (39,220); no dense fallback |
| dmg-acid2, cgb-acid2, `--weights cpu,memory,ppu`, model only and `--lockstep ppu` | pixel-exact |
| text.sh, overworld.sh with `cpu,memory,apu,ppu` | `870aa3fd`, `822059c1` (overworld.sh with the PPU in lockstep: 86,400 lines equal) |
| `tests/ppu_checkpoint.rs` | the engine equals the dense pass and the circuit; a divergence on the PPU thread stops the caller |
| `tests/ppu_dma.rs` | lines drawn during general-purpose HDMA see its bytes (failed before §9.3's per-block hand-off) |
| build without warnings, wasm32 check of the library | ok |
| `cargo test --release`, all 10 binaries | ok |
| `--core lockstep` DMG 8, CGB 300; Crystal, Zelda DX, Mario DX 1,500 frames | `6872b003`, `cfd31f3d`; `4e6a7509`, `1d00d871`, `7dec3947` |
| text.sh, overworld.sh on the `model` core | `870aa3fd`, `822059c1` |
| blargg `cpu_instrs`, `instr_timing`, `dmg_sound` 01–04 | Passed; `99793856`, `881f3fe7`, `e494d943`, `8e776b75` |
| the CPU checkpoint's digest | `1f4c3519`, unchanged |
| `model`, `model-mem` no slower than the base | `model-mem` ties; `model` ties on text.sh and is 1.3% behind on overworld.sh at the median (`docs/perf-log.md`, "Phase 1: the scanline checkpoint") |

### 9.7 Speed (measured)

On its own thread the checkpoint draws a line in 12 to 30 µs (the four-game runs, load
5 to 13), 2 to 4 ms per frame. In the full stack, `cpu,memory,apu` against
`cpu,memory,apu,ppu` (medians of 3), text.sh went from 27.7 to 27.4 fps and
overworld.sh from 73.9 to 63.0 fps (`docs/perf-log.md`). The CPU side is what holds the stack under 60 fps on text.sh;
on overworld.sh the PPU thread costs 15%, competing for a core on a loaded machine
(inferred, not profiled).

## 10. The palettes and the window in weights

This section moves the palette lookup and the window's line counter into the
checkpoint (§9 left both on the host). Each output position now carries a pixel's final
colour, CGB's RGB555 or DMG's 2-bit shade. The host only maps that colour to the
screen's RGB (`machine::ppu::rgb`, a fixed table with no register in it).

### 10.1 Where each piece went

- **Palette bytes as records.** The checkpoint's image has a palette region
  (`memory::layout::PAL`, 131 bytes: background palette RAM, object palette RAM, then
  BGP, OBP0 and OBP1; `Image::with_palettes`). The CPU side's image has no tokens there.
  A write to BGP, OBP0 or OBP1 and every palette RAM byte the bus stores become records
  in the same queue as the VRAM and OAM records (`ppu_circuit::keep_palette`), in bus
  order. A line sees exactly the palette records kept before its mode 3, as for VRAM.
- **The index registers in control.** BCPS and OCPS (auto-increment, the mode-3 lock)
  are control logic (`machine::control::PaletteIndex`, two more state bytes in the
  control checkpoint, `docs/control-in-weights.md` §15). A BCPD or OCPD write gives
  the bus the byte's index (`Signals::palette`) unless mode 3 locks it. A read gives the
  index, and the byte comes from whoever holds the palettes: the reference PPU's copy,
  or the checkpoint (`Renderer::palette_read`, a round trip to the PPU thread). A BGP,
  OBP0 or OBP1 read takes the same path.
- **A palette read from the checkpoint.** The byte is what a reader record at
  `PAL + index` reads, the newest record for the address. The engine's memory index
  serves it, which the margin argument proves is the attention result
  (`Params::check` at insert), and at the memory's oracle cadence the reader record
  runs through the dense pass over the KV cache and must agree
  (`PpuCheckpoint::palette`, as `Engine::copy` samples a DMA source). The reader
  record itself is not the read path, because it is a dense pass over the whole cache
  (about 50,000 tokens) that takes 3 to 5 s. Under `--lockstep ppu` every read is held to the host's
  palette byte and the first difference stops the run; the report counts the reads,
  the oracle checks and the reads equal. In 300 frames under `--weights all
  --lockstep all`, Yellow reads 17 palette bytes, Crystal 3, Link's Awakening DX and
  Super Mario Bros. Deluxe none; `tools/bench/battle.sh` reads 35. The CPU's thread
  waits 2.2 ms in all for Yellow's 17 (`docs/perf-log.md`).
- **The lookup.** 131 constant-address heads read the palette bytes. They sit on
  layers 8 to 10, which had no heads, so the widest layer stays at 52 heads and
  attention adds no dense parameters. Per position, one node per (object, palette,
  colour) entry (64) one-hots the pixel; 15 colour nodes sum `[CGB ∧ entry ∧ bit]` over
  the entries, plus the DMG shade's two bits from BGP, OBP0 or OBP1; 8 output nodes
  pick the low or the high byte by `FLG` bit 2.
- **Two passes on CGB.** A byte token cannot hold 15 bits. A CGB line runs the
  checkpoint twice, once per byte, and the second pass changes only `FLG` bit 2, so
  only the 8 output nodes per position re-evaluate. A DMG line runs once.
- **The window.** The checkpoint outputs the counter and the arming after the line
  (`WLN'`, `WAR'`, shared, not positional). The model threads them into the next line's
  inputs, starting from the machine's at the hand-off. The frame's reset is a function
  of LY inside the circuit, where line 0 starts from 0 and unarmed. That is the reference's
  reset at line 144 and at LCD off, because line 0 is the next line drawn after either.
  It stays in the PPU checkpoint, not control, because it changes only when a line is drawn,
  and it reads LCDC, WX and WY as of that line, which only the line's pass has.
- **Lockstep.** `--lockstep ppu` holds every line's 160 colours and the window's state
  after the line to the reference renderer's (`machine::ppu::Drawn`).

### 10.2 What stays on the host

After this round LCDC, SCX, SCY, WX and WY were latched by the host as the CPU wrote
them and handed to each line's pass as inputs. §12 moves them into the checkpoint. The
colour-to-RGB map is presentation.

### 10.3 Size (measured, `examples/shape --ppu`)

| Checkpoint | Wires | Nodes | Depth | Layers | Heads | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - |
| §9 | 3,812 | 1,364 | 12 | 13 | 52 | 514.8 M | 691,571 |
| §10 | 4,967 | 1,471 | 15 | 16 | 52 | 628.9 M | 714,071 |
| §10 with two folds | 4,958 | 1,462 | 14 | 15 | 52 | 590.9 M | 712,895 |
| §10 with three folds | 4,952 | 1,456 | 13 | 14 | 52 | 552.9 M | 715,340 |

Three more levels (the one-hot, the colour, the byte) are three more layers at hidden
2,048 and intermediate 3,968, or 114 M dense parameters. Three folds take two back. The
sprite winner (`wn0..9`) sat alone on a level. Slot s wins when it is opaque and no
earlier slot is, a conjunction of literals, so each colour, palette and priority step
carries it and the level goes (−38.0 M). The pixel byte (`compose`) fed only the 64
entry one-hots, so each entry is now the sum, over the exclusive cases where its layer
shows, of that case with the layer's five fields equal to the entry's. The one-hots sit
on the compose level (−38.0 M). A tile row's Y flip was an 8-step staircase per bit.
It is now `row ⊕ attr.6`, two steps over two wires, on the same level. The depth is back at
13, so §10's wider reader mask (`engine::Readers` as `u32`) is reverted and
`engine.rs` is as before. The CPU checkpoint's digest is unchanged (`1f4c3519`).

### 10.4 Gates and speed (measured)

With the three folds, from a copy of `tools/gate.sh` with absolute paths, every gate
passes, the new in-weights lines included. Yellow CGB 300, Crystal, Zelda DX and Mario DX
300 frames with all five in weights and in lockstep reach `cfd31f3d`, `ccfe7ad3`,
`7b126eb8` and `3e1f4d27`, every line equal colour for colour (41,750 on Yellow), and
dmg-acid2 and cgb-acid2 are pixel-exact with `cpu,memory,ppu,control --lockstep all`.
`tests/ppu_checkpoint.rs` holds the engine to the dense pass and the circuit on both
bytes and the window state; with 14 layers its dense passes take 999 s.

A line costs 52 µs on text.sh and 72 µs on overworld.sh on the PPU thread (6,312 and
10,185 units a pass, two passes a CGB line), against 21 and 47 µs in §9.7. The difference is the second
pass and the lookup's three levels. The thread hides it. The machine's thread no longer
waits at VBlank. The worker gets 8 lines a wake, VBlank only asks for the frame, and
`Session::frame` waits for it when the frame is read; the worker runs at the
interactive QoS class on macOS. The full stack (`cpu,memory,apu,ppu,control`, no
lockstep) measured 27.1 fps on text.sh and 64.4 on overworld.sh in one run at load 4.5,
against 27.3 and 60.8 before this round (`docs/perf-log.md`, "Phase 1 and 3: the host's
last PPU work").


## 11. Line heads

The memory now holds one cell per 8-byte line (`memory::layout::LINE`), and a head
reads a whole line (`Circuit::head`). The scanline circuit asks for the lines it needs
and picks bytes in its own nodes, so it needs 125 heads instead of 409, with depth 14 as
before.

| Heads | Before | Lines | Pick |
| - | - | - | - |
| Object Y | 40 | 20 constant lines: line i holds entries 2i and 2i + 1 | none |
| Slot X for the sort | 10 | 10: the line of entries `index & !1` | index bit 0, inside the X comparisons (four `le` nodes per slot pair, one level before `before`) |
| Sorted slot Y, X, tile, attributes | 40 | 10 | sorted bit 0: inside the row, 8x16 bit, bank and `X & 7` nodes; X >> 3, tile and attribute bits are two-step nodes on the same level |
| Object row | 20 | 10: rows `row & !3` to `row | 3`, both planes | row bits 0, 1, one node level before the fine shift |
| Background and window map and attributes | 84 | 16: a 32-column map row is four lines, read from column `c0 & !7` on, the line bits wrapping at 32 | tile t takes byte `(c0 & 7) + t` of the four, one eight-way node per bit |
| Background and window tile rows | 84 | 42 | the flipped row's bits 0, 1, one node level before the column pick |
| Palettes | 131 | 17 constant lines | none |

Before, layers 1 to 10 held 40, 42, 42, 52, 42, 37, 40, 38, 38, 38 heads. After, layers
1 to 12 hold 20, 16, 4, 18, 17, 17, 10, 10, 4, 3, 3, 3. The widest layer
is layer 1's Y lines, which the sort's critical path needs there.

The object row's pick costs a level the sprite path cannot spare. Fusing it into the
fine shift would multiply that stage's 3,840 units by four, so the pick keeps its
level and the output byte's choice moved into the colour nodes instead. Each output bit
sums the colour steps of bit j under `¬FLG.2` and of bit j + 8 under `FLG.2`. The
background's picks sit on levels its path had spare.

The circuit grew from 1,681 nodes and 13,386 steps to 3,070 nodes and 22,185 steps,
mostly the map's eight-way picks (5,040 units). Even so, the widest level fell from
3,744 units to 3,124. Slot index bit 0 no longer feeds a query, so `Circuit::schedule`
moves its nodes (700 units) off level 4, which held the old widest level. The f16 census
is clean before and after, with no T1 violation and no T2 gate or down row over its
bound.

`tests/ppu_lines.rs` holds the circuit to `machine::ppu`'s renderer over random VRAM (both
banks on CGB), OAM, palettes and per-line registers, on DMG and CGB, and compares every
pixel of both colour bytes and the window state after the line.

## 12. The display registers in weights

Before this change the host latched LCDC, SCY, SCX, WY and WX as the CPU wrote them and
handed the five bytes to each line's pass as inputs. That was the last PPU state the
host held. Now they are display bytes in the checkpoint's image, as BGP, OBP0 and OBP1
already were.

- **Where they sit.** `memory::layout::PAL` grows from 131 to 136 bytes. The five
  registers follow OBP1 from `PAL_REGS` in the order LCDC, SCY, SCX, WY, WX, so BGP's line
  (`PAL + 128`) holds BGP, OBP0, OBP1 and then them.
- **Writes.** A bus write to any of them is a record in the queue the palette writes
  and the CPU side's VRAM and OAM records share (`Ppu::display_write`,
  `ppu_circuit::keep_display`). A line sees exactly the records kept before its mode 3,
  which is when the host latch was read before.
- **Reads in the circuit.** One constant-address head at layer 1 reads BGP's line, and
  the circuit takes bytes 3 to 7 as the registers. The inputs are now `LY WLN FLG`. The
  sprite scan's Y lines are layer-1 heads already, so the registers arrive no later than
  the objects they are compared with, and the depth stays 13.
- **CPU reads.** A read of LCDC, SCY, SCX, WY or WX takes the path a BGP read takes
  (`Renderer::display_read`): the newest record at `PAL + index`, from the engine's
  memory index, sampled by the dense oracle at the memory's cadence. Under `--lockstep
  ppu` every read is held to the reference PPU's byte. Yellow reads them rarely
  (measured with a temporary counter on the reference core over a whole bench run:
  LCDC 849 times and WX 15 times in text.sh's 3,500 frames). Over 300 frames in
  lockstep Yellow made 39 display byte reads and the CPU's thread waited 8.97 ms for
  them in all; Crystal made 15 and waited 1.93 ms.
- **Lockstep.** The reference PPU still keeps its own registers; under lockstep it draws
  each line with them and hands its colours, window state and registers over in
  `Drawn`. A wrong register record shows as a pixel difference on the next line or as a
  read difference (`a_corrupt_display_register_record_stops_the_run_on_the_ppu_thread`).

### 12.1 Size (measured, `examples/shape --ppu`)

| Checkpoint | Wires | Nodes | Depth | Layers | Heads (widest layer) | Vocabulary | Dense parameters | Nonzeros |
| - | - | - | - | - | - | - | - | - |
| before | 5,310 | 1,438 | 13 | 14 | 20 | 6,090 | 254.1 M | 318,898 |
| after | 5,334 | 1,438 | 13 | 14 | 21 | 4,805 | 252.9 M | 317,621 |

Five fewer input slots drop 1,285 tokens from the vocabulary, which saves more than the
twenty-first head on layer 1 costs.

### 12.2 What stays on the host

Nothing of the PPU's state. The host maps each colour to the screen's RGB
(presentation), and the control checkpoint says when a line is drawn.

## 13. The pixel FIFO reference: writes during mode 3

Through §12 the reference drew a line at mode 3's start from the registers of that
moment. Games and the outside tests write LCDC, SCX, SCY, WX, WY, BGP, OBP0 and OBP1
during mode 3: Yellow 8 times in 300 CGB frames, Crystal 10, Zelda DX 1,656,
text.sh about 135 and overworld.sh about 1,500 over their runs (measured with a
temporary counter, `--weights none`). Mealybug Tearoom's `m3_*` tests, cgb-acid-hell
and daid's `ppu_scanline_bgp` judge exactly those writes. The reference now draws each
line with a pixel FIFO and every write at its time.

### 13.1 The model

`machine::fifo` is one visible line's mode 3 as a pure function: `draw(start, writes)`
takes the line's start (display bytes, window state, VRAM, OAM and the objects the OAM
scan chose) and the display writes made during it, each at its time, and returns the
160 colours, the window's state after the line and mode 3's length. It ports SameBoy
v1.0.5 (`Core/display.c`: `GB_display_run` from `mode_3_start`,
`advance_fetcher_state_machine`, `render_pixel_if_possible`), for the DMG and for CGB
hardware in CGB mode or compatibility mode:

- the background and window fetcher, a step a dot (tile index, low and high bitplanes,
  push), into an 8-pixel FIFO, with the 8 junk pixels and the SCX fine-scroll discard;
- objects fetched as the line reaches their X, the fetcher finishing its tile first;
  overlaid by OAM index in CGB mode and first-fetched otherwise; fetched with OBJ_EN
  clear on CGB hardware;
- the window started when the line reaches WX (WX 0 and the DMG's WX - 6 cases), its
  rows counted per start, so a line can start it twice;
- every register read at the dot SameBoy reads it: LCDC, SCX, SCY and WX at fetch,
  BGP, OBP0, OBP1, BG_EN and OBJ_EN as each pixel leaves the FIFO;
- SameBoy's glitches for these consoles: the DMG's window pixel insertion (issue 278),
  the CGB's WX glitch, the tile-select glitch of every CGB revision but D.

Time is in half-dots from the end of the M-cycle STAT's mode 3 begins in (`fifo::Time`),
because double-speed writes land on half-dots. A write's changes come from SameBoy's
conflict maps (`sm83_cpu.c`, `cycle_write`), with these findings from the tests. Where
Mealybug ships two DMG images, DMG-blob and DMG-B, the census judges DMG-blob and the
rules follow it:

| Register | Change | Source |
| - | - | - |
| all | a write landing on the time a dot's code runs comes before it | SameBoy's `GB_STATE_MACHINE` resumes a state only once its time has passed; m3_lcdc_obj_en_change_variant |
| DMG SCY | 2 T-cycles before the write's end, not SameBoy's 1 | Mealybug's PPU notes ("SCY": immediate on the DMG, 2 T-cycles later on the CGB); m3_scy_change |
| DMG LCDC | BG_MAP, TILE_SEL, WIN_MAP and OBJ_SIZE take the new value 2 T-cycles before the write's end | m3_lcdc_bg_map_change, _tile_sel_change, _tile_sel_win_change, _win_map_change, _obj_size_change and _obj_size_change_scx |
| DMG LCDC | WIN_EN cleared also lands 2 T-cycles early, and the window pixel insertion is disabled then, not a T-cycle later | m3_lcdc_win_en_change_multiple_wx: on lines 2, 22 and 30 the next window tile's fetch (at 28, 84 and 84 half-dots, the first stage of writes ending at 32, 88, 88) sees WIN_EN clear and fetches background; on lines 8 and 36 the window's start check at 28 and 84 sees it clear and the window never starts; and no pixel is inserted between the stages (lines 3, 31) |
| DMG LCDC | BG_EN cleared at x 0 lands 2 T-cycles early, as SameBoy has OBJ_EN cleared there; BG_EN set lands 1 T-cycle early, at the second stage, not SameBoy's 2 | m3_lcdc_bg_en_change: x 0 of lines 19 to 22 (both images; fitted on that one band), and the DMG-blob image, which shows the background a pixel later than the DMG-B image where BG_EN is set |
| DMG WX | the window started at WX - 6 does not set the LCD a pixel back (SameBoy: "only on some DMGs") | m3_lcdc_win_en_change_multiple_wx lines 16 and 44: DMG-blob draws the window from WX - 6, DMG-B a pixel sooner, both with all 160 columns, which SameBoy's step back leaves one short of |

One constant ties our line to SameBoy's: a write ending an M-cycle lands `WRITE_OFFSET`
= 8 half-dots after that M-cycle's end in the FIFO's time. It was fixed on m3_bgp_change
and holds for every other case (7 and 8 both pass every SameBoy-passing DMG case; 8 is
one single-speed M-cycle). No double-speed case in the census writes during mode 3, so
whether the constant is in dots or in M-cycles is not tested.

### 13.2 In the machine

`Ppu::render_line` opens the line at mode 3's start: it snapshots the display bytes and
the window, runs the OAM scan (`fifo::scan`), publishes the length the line has with no
write and hands the renderer `Renderer::open`. A display write while the line is open
joins its writes at its time (`Ppu::set_clock`, from `Machine::halves`) and revises the
published length when it moves it. The line is drawn (`close`) at the first of: the next
line, VBlank, LCD off, STOP or compatibility mode changing. Every register write until
then joins the line, and the FIFO's time alone decides what it reaches: a write landing at
or past `2·dots + fifo::REACH` (5 half-dots, one past the 4 a DMG palette write reaches
back) changes nothing (`a_write_past_the_end_changes_nothing`). Up to §14.2 the line was
also drawn at the first write at `2·dots + 4` or later, though such a write can still reach
the last pixel's palette (measured on 30,000 random lines: offsets 0 to 4 change a line,
none from 5 on); that early close is gone. CGB palette RAM cannot be written during mode 3,
so its writes never join a line. VRAM or OAM written while the line is open is kept as of
its start.

The renderer's records split at the open: those kept before it are the line's image,
those after wait until the line is handed over (`Renderer::open`, the `cut` in
`ppu_circuit::{Inline, Worker}`), so a VRAM write after mode 3 that the memory
checkpoint keeps before the host closes the line cannot reach the line.

The window arms when LY equals WY at a line's start, as before. SameBoy also asks for the
window enabled there and checks again at each LCDC or WY write (`wy_check`); with that rule
the 48 PPU cases of the census (Mealybug, acid, daid, mooneye ppu) give the same verdicts,
case for case, so the rule the checkpoint already computes stays.

The DMG hand-off now leaves VRAM as the DMG boot ROM does (`Ppu::boot_logo`): the
header's logo in tiles 1 to 24, the ® mark in tile 25 and the logo's map rows.
Mealybug's tests draw the ® tile. `examples/boot_state` (since removed) against SameBoy's DMG boot ROM
finds no VRAM byte different.

### 13.3 Against the line renderer it replaced

`ppu::tests::the_fifo_draws_a_quiet_line_as_the_line_renderer_did` draws 3,000 random
lines with no write on DMG, CGB and compatibility mode with both. Colours, the window's
rows and the length are equal except in these classes, each SameBoy's and each a case the
line renderer did not model:

- a DMG's window starts with LCDC.0 clear (white, its rows still counted);
- WX below 7 or at 166, and a DMG armed with the window disabled, which inserts a pixel;
- CGB hardware fetches objects with OBJ_EN clear, so they lengthen mode 3;
- objects under a fine scroll or on a line the window starts on: the lengths are 1 to 6
  dots apart. Pan Docs' "Mode 3 length" sides with the FIFO on both: "an OBJ with an OAM X
  position of 0 (thus, completely off the left side of the screen) always incurs a 11-dot
  penalty, regardless of `SCX`", and the tile an object's penalty counts in "is affected
  by horizontal scrolling and/or the window". With one object, the FIFO's penalty equals
  Pan Docs' formula at every X and SCX but X 0. Mooneye's sprite table, at SCX 0 with no
  window, holds the FIFO (`mode_3_grows_as_mooneyes_sprite_table_says` runs it on DMG and
  CGB). No gbmicrotest PPU case (`examples/microtest`, since removed; 178 sprite, window, SCX and
  HBlank tests) passes on one rule and fails on the other.

On the write-free lines of the gate games and benches the FIFO and the line renderer
draw the same colours on every line. The lengths differ on lines with an object at X 0
under a fine scroll (Yellow parks objects there) and on lines where the window starts
with objects: 316 lines of text.sh and overworld.sh each, 13,797 of battle.sh. These are
the lines §14's first step must bring the checkpoint to.

### 13.3a The census

`docs/hardware-tests/after-ppu-fifo.tsv` is the runner on this reference (342 applicable
cases). Against the reference before the FIFO, 30 cases go from FAIL to PASS and none the other way:
the 15 DMG Mealybug rows SameBoy passes, the seven DMG rows SameBoy fails that
the early DMG SCY and LCDC address bits turn (`m3_lcdc_bg_map_change`, `_tile_sel_change`,
`_tile_sel_win_change`, `_win_map_change`, `_obj_size_change`, `_obj_size_change_scx`,
`m3_scy_change`), cgb-acid-hell,
and CGB Mealybug rows the census now marks NA. `m3_lcdc_bg_en_change` and
`m3_lcdc_win_en_change_multiple_wx` on DMG pass since the LCDC and WX rules of §13.1 that
follow the DMG-blob images (232 and 45 pixels off before). Against the DMG-B images, which
Mealybug ships for these two tests only and the census does not judge, they are 228 and 3
pixels off: the BG_EN set a pixel sooner, the window started at WX - 6 a pixel sooner. Left on DMG: daid
`ppu_scanline_bgp`, which waits on LY 153's LYC timing and
the HALT wake (access timing's, `examples/microtest`).

### 13.4 What it does to the gate hashes

None moved. `--weights none` and `--core lockstep` both give Yellow DMG 8 `6872b003`, CGB
300 `cfd31f3d`, Crystal `ccfe7ad3`, Zelda DX `f10b2b4a`, Mario DX `ea7f86ca`, text.sh
`870aa3fd` and overworld.sh `822059c1`; `--core lockstep-mem` DMG 8 `6872b003`. battle.sh
gives `9a57a127` with the FIFO and without it alike, so the `14d7468e` in
`CLAUDE.md` moved before this work.

Through §14.1 the PPU checkpoint drew a line at once from its start, so `--lockstep ppu`
stopped at the first line with a write that changes a pixel or the length. §14.2 built the
checkpoint's side, and every line is drawn by the model and compared.

## 14. Design: the checkpoint draws writes during mode 3

Status: built (§14.1, §14.1a, §14.2b, §14.2c). §13's reference is the target, and `--lockstep ppu`
holds on every line: no line is refused or left uncompared.

### 14.1 First step: quiet lines equal to the FIFO

The scanline circuit (§3b to §12) and the length stream (`circuit::build_length`) still
compute the line renderer's results, so on a line with no write they differ from §13 in
the classes of §13.3. In the gate games and benches only the length differs, on lines with
an object at X 0 under a fine scroll and on lines where the window starts with objects.
The circuit changes, all in the shared stages (no positional unit moves):

| Class | Circuit change |
| - | - |
| object at X 0 | it counts as fine position 0 of tile 0, whatever SCX: penalty 5 if first in that tile (`fifo::length_rule::x0_rule`, 20,000 random object sets, all equal) |
| objects on a window line | an object whose leftmost pixel is at or past the window's start (`X - 8 >= WX - 7`) counts in the window's grid, fine position `(X - WX - 1) mod 8`, in tiles of their own (`fifo::length_rule::window_rule`, 20,000 random lines, all equal) |
| DMG, LCDC.0 clear, window armed and enabled | the window still starts (rows counted, 6 dots), and draws white |
| WX 0 to 6, WX 166 | SameBoy's start points and the DMG's carried WX 166 line (`fifo::Window::carried`, one more FLG bit) |
| DMG, window armed and disabled | the pixel inserted at WX (SameBoy issue 278) |
| CGB hardware, OBJ_EN clear | objects still fetched, so they lengthen mode 3 (no game line hit it; SameBoy's) |

Gates: `tests/ppu_lines.rs` holds the circuit to `fifo::draw` with no write (it drives
`Ppu` with no clock today, so its writes land inside lines); `ppu_checkpoint`, `ppu_dma`;
every gate game and bench with `--lockstep ppu`, 0 dense fallbacks; the census unchanged.

### 14.1a The window's edge cases, built

The three window classes `fifo::QUIRKS` kept out of the checkpoint are in both the line
circuit and the length circuit now, and the label is gone: no lockstep run skips a line.
`fifo::length_rule` holds each rule as a closed form against `fifo::draw` (200,000 random
lines for the length and the window's state, 60,000 for every background and window
pixel), and the circuits compute those closed forms (`circuit::window_cases`). The rules,
each read off `fifo::State` on a line with no write:

- **The window's start** (`State::window`). It starts at the line's check when enabled,
  armed and not carried, at WX 0 to 165, or 166 on CGB hardware (`166 + cgb`). At WX 1 to 6
  and at WX 0 with SCX's fine scroll f = 0 it draws as at WX 7 to 165: column `x + 7 - WX`.
  At WX 0 with f > 0 it starts at position -16, before the junk row, and draws column
  `x + 8 + f` from x = 0, one dot later (the DMG's `WindowOn` dot, the CGB's glitched push).
  A DMG at WX 166 never starts it, but counts a row at position 159.
- **The pixel inserted at WX.** Rows are pushed at positions `-f + 8n`, so a push meets
  `pos + 7 == WX` exactly when `(WX & 7) + f == 7`, at `P = WX - 7`. There a DMG armed with
  the window disabled inserts one pixel (issue 278, `push`), and so does a line fetching
  the window from its first tile (`insert_bg_pixel`, which needs `wx_triggered`). The pixel
  at P is colour 0 and every pixel after it shows the stream one place later. A carried
  line meets WX 247 at its junk row, `P = -f`.
- **The carried line** (`window_after`, `Window::carried`, FLG bit 5, `WAR'` bit 1). A DMG
  line that ends armed and enabled at WX 166 counts one more row and carries the window:
  the next line draws it from x = 0 at column `x + 8 + f`, from the row before its counter,
  and counts no row of its own.
- **Mode 3's length.** 172 dots, f, and 6 when the window starts (7 at WX 0 with f > 0).
  Past an inserted pixel an object counts in the grid a dot later, `X - 9 - P`; on a window
  line the row is already in the FIFO, so the 9 positions from P are one tile at fine
  position `X - 8 - P`.

Two classes the old label did not name surfaced in the pixel rule, both CGB hardware with
the window enabled and armed (`update_wx_glitch`): a T1 step run where `pos + 7 == WX`
(WX 247, 248, 250, 252: the first background tile takes tile 0 and attributes 0, or loses
a plane, as the fetcher's steps from position -15 meet it, or all at -16 when an object at
X 0 is fetched), and at WX 0 with f > 0 the window's column tile 1 (and tile 2 at f = 7
with an object at X 1) has tile 0 and attributes 0, its planes kept only with no object at
X 0 or 1 (`fifo::length_rule::glitch`). Both are in the line circuit too: a per-line tile-0
head reads the stale tile's row.

Size (`examples/shape --ppu`, the digest): the line checkpoint stays at 14 layers and goes
from hidden 1792, intermediate 2816, 299.9M parameters and 354,578 nonzeros to hidden 1984,
intermediate 2880, 337.3M parameters and 377,005 nonzeros; the length checkpoint from 13
layers, 98.7M parameters and 139,344 nonzeros to 14 layers, 105.7M and 148,926. The growth is not broken down by node
family on the final circuit, and the length checkpoint's extra layer is not traced to its node. Digests: ppu `84e854e6` to
`6f77b44e`, one model `13878ff8` to `7902814e`; f16census 0 violations.

Gates (measured): `ppu_length` 3,000 lines (355 at WX below 7 or 166, 43 issue 278, 256
carried); `ppu_lines` 600 lines (287, 47, 76, and 170 with the CGB's WX glitch); Yellow,
Crystal, Zelda DX and Mario DX 300 frames with all five in weights and in lockstep, and
Yellow as one model, at their hashes, every PPU line equal and 0 dense fallbacks.

### 14.2 Second step: lines with writes

Measured so far: Zelda DX's mid-line writes are SCX (1,377 in 300 frames) and SCY (276);
Yellow and Crystal write each register once or twice in 300 frames; the benches write
SCY, BGP, OBP0 and OBP1 (§13). Still to measure before building: event lines per frame,
mean and maximum, inside each bench's measured window (frames 2900-3500, 7200-7800,
14880-15480) on this build.

Two shapes were weighed:

1. **Selection in the line pass.** The pass takes a short list of (time, register, value)
   writes and selects, per pixel, the value in force at the dot the pixel leaves the FIFO,
   and per fetched tile, the value at its fetch dots. That is exact only while the writes
   leave the line's dot timeline unchanged. By the FIFO's structure they do for BGP, OBP0,
   OBP1 (read as a pixel pops), SCY (read at fetch, which no FIFO state waits on), SCX once
   the fine-scroll discard is over (coarse SCX only names the next tile), and LCDC's
   BG_MAP, TILE_SEL, WIN_MAP and OBJ_SIZE bits (addresses only). They do not for SCX during
   the discard, WX, WIN_EN or OBJ_EN (the DMG's aborted fetch). It needs each pixel's output dot and each tile's fetch dots in the
   pass: the §3 object and window sums carried per position.
2. **A FIFO stepper.** A small checkpoint whose pass is one step of `fifo::Line` (a dot,
   or a stage of an object fetch), with the FIFO's state as input and output slots, as the
   APU threads its state per segment (`docs/apu-in-weights.md`). The pass takes the next
   pending write (time, register, value) and its progress through its conflict stages, and
   says whether it consumed it; the host only moves the queue on when the model says so.
   One pixel leaves per pass at most. It is the reference's step function compiled, so it
   is exact by construction for every register, and a line with no write never runs it.

The plan is the stepper first, because it is exact for every write and costs nothing on
quiet lines; the selection pass is the fast path for the timing-free registers above if
the stepper's measured cost misses the bench. Before building the stepper:

- refactor `fifo::Line` into an explicit step function over a state struct (the object
  fetch's sleeps become stages), with the reference driven by it, so the circuit and the
  reference share one spec;
- build one step (fetcher, pixel pop, palette, no objects) in the circuit IR and read its
  depth and units (`examples/shape`); a step must fit the layer budget, or the step is
  split, which doubles the passes;
- measure µs per pass on the engine and multiply by the steps of an event line (about 180
  to 300).

### 14.2b The stepper and the judge, built

The stepper (`ppu_circuit::stepper`) is the second shape: one pass is one `fifo::State::step`,
and a start pass computes `State::new`. A line is the stepper's only when a write can change
it; the length stream decides that per write, so a quiet line, and a line whose writes all
miss it, still takes the scanline checkpoint's one pass.

**The queue.** `fifo::drive` hands a line its writes as a strict queue: the head write is
offered until a pass says its last change applied (`Out::finished`). That is time order for
writes at least an M-cycle apart, because a write's changes span less than one M-cycle
(`drive_is_time_order`, 30,000 random lines in every mode and at both speeds, every pass's
state equal to the old driver's, which picked the write whose next change lands first). The
driver asserts the spacing, and needs no conflict map of its own: the stepper and the
runner move the queue only as a pass's output says.

**The judge** (`fifo::judge`, computed by the length circuit). A write cannot change the
line when it targets WY (read at the line's start), BGP, OBP0 or OBP1 in CGB mode (which
draws from palette RAM), or carries the value the line started with (but WX on the DMG,
where any write reaches the WX comparator), or lands at or past `2·dots + REACH` of the
line with no write. The length circuit takes the write as three inputs (`MWH`, the machine's
2-dot halves since mode 3 began; `MWI`, `MWV`) and outputs the verdict (`MJ`: the line is
marked; the write is a register's, which the stepper's queue takes) and the write's time in
half-dots, `4·H + 8` (`MWT0`, `MWT1`). The stepper's queue carries that time, so the host
computes none: it only counts halves since the line's trigger, as it counts the spans
stream's countdown. `a_line_the_judge_passes_draws_as_with_no_write` (60,000 random lines)
holds the rule sound; `ppu_length` holds the circuit to it on 3,000 lines and 12,000 writes;
under `--lockstep ppu` every verdict and time is held to the reference's, and a mutant
length circuit whose verdict is zero stops the run
(`a_judge_that_passes_a_changing_write_stops_the_run`).

**The window.** The stepper's start pass takes the window after the line before, as the
scanline circuit does, and applies the frame start and the LY = WY arming in weights
(`fifo::Window::starting`, which the reference uses too).

**In the machine.** `Ppu` keeps a line open until it closes (§13.2), with a renderer drawing
alone as with the reference beside it, and hands each display write its halves since the
open (`Renderer::display_write`, `LineWrite`). The length model judges it on the machine's
thread. At the close a marked line goes to the PPU worker, whose `PpuModel` draws it with
the stepper over the same records the scanline checkpoint follows, every state slot carried
from a pass's outputs to the next pass's inputs byte for byte; its colours, window and
length replace the scanline pass's. The control logic reads a line's length at the least
mode-0 start, often before the line ends: while it has not read it, a marked write the
stepper takes makes a revision due (`Lengths::reissue`), which the worker draws with the
writes so far and publishes before the read returns. A revision after the read is lost, in
the reference and the model alike. The line after a stepped one waits for the window the
stepper left. Under `--lockstep ppu` every stepper pass's state is held to the reference
FIFO's trace, and the revised length to the reference's at the read. The refusal is gone:
every line is drawn by the model and compared.

**The one model and the runner.** The stepper is the one model's sixth stream
(`one_model::Part::Stepper`). The wiring table gains its role (`wiring::Stepper`: its slots,
its flag and report bits, its carries, the length's shift) and the length's judge slots. The
runner (`tools/runner/src/rows.rs`, `stepper.rs`) opens a row at each trigger, judges each
own-memory write while it is open, redraws a marked row's length through the stepper while
the spans stream has yet to read it, and at the close posts the stepper's row in place of
the lines stream's. `tools/stock-llama` runs the stepper's passes in llama.cpp like any
stream's.

### 14.2c Mode 0's SCX rounding, the sizes and the census, as merged

**SCX's rounding (`fifo::mode0_dots`).** A3 (`docs/hardware-tests.md` §3.12) ends a
DMG's mode 3 at the next M-cycle boundary after SCX's fine-scroll discard: the dots to mode
0 are `dots − f + ⌈f/4⌉·4`, f = SCX & 7 at mode 3's start. A CGB ends mode 3 at the dot,
so its dots are the FIFO's (§3.14 there). The published length is those dots less 128
(`fifo::length`, `lcd::LENGTH_BASE`), which the control logic places by halves. The
FIFO's own dots stay unrounded, because the line's pixels, its window and the judge read
them. The rule applies to the reference's published length, its revision on a mid-line
write and `Drawn::length`; to `ML` in the length and scanline circuits; and to the
stepper's `DOTS`, from the fine scroll its start pass latches, which the wiring turns into
the length by its bias. The line renderer the FIFO replaced (`Ppu::mode3_dots`) rounded
the fine scroll on every console, so `the_fifo_draws_a_quiet_line_as_the_line_renderer_did`
compares it with the DMG rounding of the FIFO's dots.

**The write queue's order.** `fifo::drive` offers the head write until a pass reports its
last change applied (`Out::finished`). That is time order, because the writes are at
least an M-cycle apart and a write's changes span less than one (§14.2b).

**Sizes** (`examples/digest`, `examples/shape`, measured on the merged build):

| Checkpoint | Layers | Hidden | Intermediate | Dense | Nonzeros |
|---|---:|---:|---:|---:|---:|
| ppu (scanline) | 14 | 1,984 | 2,880 | 337.3M | 0.43M |
| ppu length (the judge) | 15 | | | 114.1M | 151,468 |
| ppu stepper | 9 | 2,176 | 5,312 | 482.9M | 0.62M (350,961 in `shape`) |
| one model, six streams | 25 | 2,752 | 10,624 | 3,178.5M | 1.84M |

The stepper's stream is what widens the one model. The build before it, without it,
was hidden 2,176, intermediate 5,376 and 1,387.2M dense. The stepper's 9 layers overlay
the first 9 of the one model, and its units add to theirs there: those layers go from
1,472-5,376 units to 2,560-10,624. The widest, layer 6, goes from 5,248 to 10,624.
Hidden grows from 2,176 to 2,752. Those widths are left for the optimization phase.

**Stepped lines on the benches.** Measured on the merged build with all five in lockstep:

| Workload | Lines the reference marked, warmup included (`stepped_lines`) | Lines the stepper drew in the model's 600 frames | Stepper passes |
|---|---:|---:|---:|
| text.sh | 8 | 5 | 4,385 |
| overworld.sh | 20 | 0 | 0 |
| battle.sh | 31 | 11 | 9,757 |

`stepped_lines` in `run`'s summary counts the machine's marked lines from frame 0, so it
includes the reference's warmup frames. The model's count is the length stream's
`lines marked` line.

**Census.** 310 of the 322 applicable cases pass in the reference and with all five in
weights, case for case (`docs/hardware-tests.md` §2.2). Every applicable Mealybug DMG
`ppu/` row passes, as do cgb-acid-hell and daid's `ppu_scanline_bgp` on both consoles.
