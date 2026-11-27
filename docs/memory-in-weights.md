# Memory in weights: the Game Boy's memory as the Llama model's KV cache

**Superseded in part (f16 round, `docs/stock-llama.md` §4 and §5).** The f16 round made the
memory one cell per physical address, with the address as its position. A write evicts
the old cell and decodes a first-write record, so the recency pair, the snapshot kind,
the window and compaction below (§3, §8) are gone. Every memory head's value was the
byte's two nibble one-hots, which replaced the class heads, and one KV head serves every
head. A later change ("sm83: memory circuit reads and writes whole lines")
replaced the byte cells with one cell per 8-byte line at the line's index, whose value is
the line's 64 bits (`docs/stock-llama.md` §4, "Chosen: one cell per 8-byte line", with
the measurements in §12a). The sections below describe the design as it stood before
the f16 round.

Design A (DEC14) integrated with the CPU checkpoint. Instruction fetch, data reads and
writes go through attention heads inside the model, and the host keeps byte arrays only
as a shadow for lockstep and as the mirror the host devices still read. This document
is the design of record for E20 and is kept current as units land. Each section says
whether it is **built** or **designed**. DEC16 records where the built form departs
from DEC15 and why.

Status by unit:

| Unit | State | Evidence |
| - | - | - |
| A. One head dim and one rope_theta for the whole model (32, 1e38) | built | `docs/perf-log.md` row 43; checkpoint_sst, sm83_sst, e01 and interrupt_fuzz pass; lockstep hashes `6872b003`, `cfd31f3d`, text.sh `870aa3fd`, overworld.sh `822059c1` unchanged; dense params 171.7M to 187.4M, nonzeros 68036 to 68004 |
| B. Memory records: the model computes a token's cache key and value from tokens | built | `src/memory/record.rs` and `examples/prefill.rs` (both since removed), `cargo test --release --lib memory::`; §3's table |
| C. Address translation and the fixed layout in the circuit | built | `memory::layout::translate` equals `Machine::physical` on all five mappers, every bus address, both access kinds, random register traffic and OAM DMA; the one-level circuit equals the spec on 20,000 random samples |
| D. Fetch and data heads, write records, in the SM83 circuit and the checkpoint | built | `build_mem()` passes 500,000 SST cases (one aliased case skipped); the dense pass with a KV cache equals the circuit on random steps and every kept key equals `memory::attn::key` (`tests/memory_dense`) |
| E. Engine `Plan::Memory`, indexed fast path, compaction, `--core model-mem` | built | engine equals `forward_cached` with the cache; memory engine SST (§9); lockstep-mem hashes `6872b003`, `cfd31f3d` and §9's table |
| G. One translation per address source, nested page cases, snapshot keyed on read | built | units per pass 4,314 to 2,093 and compaction 50 ms to 0.1 ms (§9, perf-log "Unit G"); every gate of E holds, host checkpoint byte-identical |
| H. Write translations by a level floor (hidden 2048 again), routing stamps only on mask-token owners, memory heads read only when their address or the image changed | built, five commits | 1,916 wires, hidden 3072 to 2048, 440.5M to 293.7M dense parameters; text.sh 8.64 to 7.15 µs per pass and overworld.sh 8.04 to 6.96 chained over six interleaved A/Bs, index reads per pass 16 to 6.3 (§9, perf-log "Unit H"); every gate of E holds, host checkpoint byte-identical and no slower |
| I. The circuit reshaped around what changes per step: translation cases per bit, CB decode by field, the rare bus by nibble one-hots with a second data head, fetch adders split at bit 7, region wires | built, six commits (a seventh was measured no faster and reverted) | units per fast pass 2,082 to 1,387 on text.sh and 1,931 to 1,385 on overworld.sh; 1,726 wires, 293.7M to 266.1M dense parameters, 203,941 to 147,525 nonzeros; text.sh 7.07 to 5.95 µs per pass and overworld.sh 6.94 to 5.88 chained over five interleaved A/Bs (26.3 to 30.4 and 72.9 to 81.5 fps at the loads measured) (§9, perf-log "Unit I"); every gate of E holds, host checkpoint byte-identical and no slower |
| J. DMA copies as copy-mode records: every OAM DMA, HDMA and GDMA byte is a reader record at the source and a write record at the destination; the memory audit holds the model's image to the host's arrays after each DMA and at each frame end | built, two commits: the audit, which stops lockstep-mem CGB 300 at the first OAM DMA, then the records | lockstep-mem DMG 8, CGB 300 and the Crystal, Zelda DX and Mario DX 1,500-frame hashes unchanged with the image audit on; `tests/dma_records`; §8a and perf-log "DMA copies as records" |

## 1. The stock-stack view

Everything below is expressible to a stock Llama inference stack with a KV cache:

- `position_ids` are an input. Every token of one record, and every token of one CPU
  step, carries the same position, the number of tokens kept in the cache so far.
- After a record or a step, every token but the kept value tokens is evicted from the
  cache. A kept token sits mid-group (the last of a record, the `WM0` position of a
  step), so this is selective KV eviction, the operation StreamingLLM and H2O perform,
  not a prefix rollback. Compaction (§8) is the prefix rollback.
- Loading a cartridge is a prefill of one record per ROM byte, then one per writable byte.
- Compaction is a rollback to the ROM prefix followed by a prefill of snapshot records.

The dense pass with a KV cache (`Model::forward_cached`, HF `past_key_values`
semantics) is the oracle. A full dense pass over a million-token cache runs every layer
on every token and costs seconds (perf-log, "Memory prototypes"), so the oracle is
applied to sampled records (`GBCFORMER_ORACLE=N`) and sampled steps
(`GBCFORMER_STEP_ORACLE=N`, every decoded token of the step), and the fast path
is guarded by the margin check computed in the read's own f32 arithmetic
(`memory::attn::Params::check`).

## 2. Token stream and vocabulary (built)

A **memory record** is a group of five tokens at one position: `REC`, three address
bytes, and a value token. Only the value token is kept.

| Kind | Tokens | Key terms (see §3) | Kept token's role |
| - | - | - | - |
| ROM | `REC WP0a WP0b WP0c ROMV` | address, committed | a cartridge byte; never aged, never overwritten |
| SNAP | `REC WP0a WP0b WP0c SNAPV` | address, committed, snapshot penalty | a RAM byte at load or after a compaction; loses to any later write |
| WRITE | the CPU step's own `WM0` (and `WM1`) positions | address, committed, recency | a CPU write; the newest wins by RoPE recency |
| read | `REC WP0a WP0b WP0c RD` | (none kept) | copy mode: `RD` decodes to the `SNAPV` token of the byte at the address |

`REC` carries every input slot's identity, so the input gather heads of layer 0 see an
exact zero in a record (a gather head with no key would attend uniformly). `WP0a WP0b
WP0c` are **per-slot output tokens** (`tok::family(Fam::Addr { write: 0, byte })`),
carrying the slot identity of the `WP0x` result slot and the value bits. The CPU step
decodes the same tokens into its `WP0a..c` result slots, so a ROM record is a CPU
write record with the CPU's other positions evicted.

The kept value tokens `ROMV`, `SNAPV`, `WM0`, `WM1` carry the value bits plus kind
channels the memory heads' key projection reads (§3). `WM1` also carries `second`,
which steers the address gathers toward `WP1a..c`. Memory tokens carry no input-slot
identity, so the CPU's layer-0 gather heads see a zero key for them; §7 makes this a
plan check.

The vocabulary has 8,004 tokens: inputs `20·257`, shared outputs 256, output masks 46,
per-slot outputs `256·10` (`WP0a..c`, `WP1a..c`, `WM0`, `WM1`, `ROMV`, `SNAPV`), `REC`
and `RD`. The host checkpoint has 5,423.

**Banking registers are memory bytes** at §4's `REGS`, not one-token records as DEC15
said. A register write is an ordinary write record, a compaction snapshots it like any
other byte, and the translation interprets the raw written byte as the mapper does
(`max(v & 0x1F, 1)`, `v & 0x0F == 0x0A`), so no value transform follows the write.

## 3. How the model computes a cache key and value from tokens (built)

Only the value token is kept, so only its K and V at the memory layers matter. Both
come from the model's own layers, with no hand-written K/V:

1. **Layer-0 gather.** Three heads copy the address bytes into 24 address channels of
   every position in the group. Head `b` keys on `slot_id(WP0b)` through RoPE dim 15
   and on `slot_id(WP1b)` through dim 31, and its query is `(32 − 32·second, 32·second)`
   on those two dims, linear in the embedding. At every position but `WM1` the `WP0b`
   token scores 181 and the `WP1b` token 0, at `WM1` the reverse, and f32 `exp`
   underflows to exactly 0 beyond a gap of 104, so the copy is bit-exact. It is exact
   because the previous records' address tokens were evicted, so at most one token in
   the cache carries each `slot_id(WPwb)`.
2. **Key at a memory layer.** `W_k` reads the 22 gathered address channels as
   `(2·b − 1)` (weight `2·S'` on the channel and `−S'` on the constant), giving the ±1
   pattern of `memory::attn::key` in RoPE pairs 5..15, plus the kind channels from the
   value token's embedding: `committed` (+1, every value token), recency (−1 in pair
   2's second dim, `WM*` only), snapshot penalty (−1 in dim 3, `SNAPV` only). RoPE at
   the token's position then produces the stored key. Every memory head has the same
   `W_k` and `W_v`, so one key and value per token serves them all; the dense oracle
   asserts the heads agree before storing (`memory::dense`).
3. **Value.** `W_v` copies the value token's eight own-bit channels. `W_o` writes the
   head's output into the eight wires of the layer that read it.
4. **Query at a reading position.** `W_q` reads the 22 wires of the physical address
   the circuit computed (affine forms over wires, constants folded), as `±S`, plus `M`
   in pair 2 (recency), `B` in dim 3 (penalty) and `B₂` on `committed`. With
   `B₂ ≥ range + gap`, a step position whose residual holds the same address wires but
   whose token lacks `committed` (every non-value token, every mask token) loses to
   any real memory token. So a step's reads cannot see its own undecoded write
   positions, and an `INC (HL)` reads the old value.

**No memory head sits at layer 0.** A kept token's key at layer 0 would come from its
embedding alone, before layer 0 gathers its address, and differ from the key the later
heads compute; the dense oracle caught exactly this when the register heads were tried
at layer 0. The register heads therefore sit at layer 1, which costs one level of depth.

The margins are the prototype's (`Params::with_gap(256)`). A match beats a one-bit
mismatch, a newer write beats an older one by at least 256 logits per position across
the window, a write beats a snapshot, and every gap is at least 104 where f32 `exp`
is exactly 0. `Params::check(total + window)` evaluates them at the longest position
the cache reaches. The gather heads' gap of 181 is position-free (pair 15's angle is
below 1e-29 at 4.4 M positions).

The project's f32 convention forms angles in f64 and casts `cos`/`sin` to f32
(`llama::rope`). HF forms `position·inv_freq` in f32, which at position 2 M and
pair 2 (1.78e-5 rad per position) carries a phase error near 4e-6 rad; the margin
check does not cover that stack, and the document says so rather than claiming it.

### Prefill measurements (unit B)

Measured by `cargo run --release --example prefill -- <rom>` on the standalone record
checkpoint (`memory::record::compile`, two layers, three gather heads and one memory
head, 5.1 M dense parameters), which carries the same heads the CPU checkpoint now holds:

| Cartridge | Records | Prefill (formula path) | Oracle: records bit-identical, reads exact | KV cache (256 B per kept token) |
| - | - | - | - | - |
| Pokémon Yellow, 1 MB + 82,207 RAM | 1,130,783 | 0.834 s, 0.74 µs per record | 277 of 277 at 120 ms each; 35 of 35 at 108 ms | 289 MB |
| Pokémon Crystal, 2 MB + 82,207 RAM | 2,179,359 | 1.21 s, 0.56 µs per record | 533 of 533 at 218 ms each; 67 of 67 at 212 ms | 558 MB |

Inside the CPU checkpoint, the engine inserts Yellow's machine state as 1,131,049
records (the fixed layout adds the marker blocks and the registers) in 0.67 s, 290 MB.

## 4. The physical address space (built)

A fixed 24-bit layout (`memory::layout`, 22 bits until MBC5's 8 MiB ROMs) gives the translation circuit constant region
bases and passes the low address bits through unchanged in every read region (no
selection below bit 7):

| Region | Physical base | Size | Bus source |
| - | - | - | - |
| ROM | `0x000000` | up to 8 MB | `0000–7FFF`, bank from the mapper |
| Cartridge RAM | `0x800000` | up to 128 KB | `A000–BFFF`, bank from the mapper, masked to the RAM size |
| WRAM | `0x820000` | 32 KB | `C000–FDFF`, bank 1–7 at `D000` from SVBK |
| VRAM | `0x828000` | 16 KB | `8000–9FFF`, bank from VBK on CGB |
| OAM | `0x82C000` | 160 B | `FE00–FE9F` |
| HRAM | `0x82C100` | 127 B | `FF80–FFFE` |
| FF | `0x82C180` | 128 B of `0xFF` | disabled cartridge RAM, unmapped ROM, any read blocked by OAM DMA; `FF \| (a & 0x7F)` |
| VOID | `0x82C200` | 128 B of `0x00` | the unusable area `FEA0–FEFF` and an idle read port; writes that reach nothing; `VOID \| (a & 0x7F)`; a write landing here is not recorded |
| Registers | `0x82C280` | 15 B | `ROMB0 ROMB1 RAMB RAMG MODE SVBK VBK MAPPER MASK0 RAMMASK BOOTING UNMAPPED COMPAT KEY0 MASK1` |
| HOST | `0x82C400` | 128 B | an opcode byte fetched from a host-served address: the fetch translation lands at `HOST \| (a & 0x7F)`, and the host keeps the bus byte there as a write record before the request pass runs again (`MemoryDenoiser::keep_host`) |
| IO | `0x82C800` | 16 × 128 B | the I/O page and IE, `IO + code·128 + (a & 0x7F)` with the port code of `layout::io_code`: block 0 holds `0xFF` (unmapped registers), the others are zero ports the host serves from the device the code names (`layout::port`: control, the APU, the display bytes, LCDC to both, the MBC3 clock and its latch); not recorded |
| TAB | `0x82D000` | 8 × 128 B | VBK and SVBK readback on CGB: `TAB + (bank << 7) + (a & 0x7F)` holds `0xFE \| (bank & 1)` at `0x4F` and `0xF8 \| bank` at `0x70` |

`MAPPER` holds the mapper one-hot, a has-clock bit (MBC3 with a timer), a has-RAM bit and the CGB bit; `MASK0` is
`banks − 1` and `MASK1` bit 0 its bit 8 (MBC5's 512 banks); `RAMMASK` is `(ram_len − 1) >> 9`. Tokens exist only for the loaded ROM
length, the cartridge's RAM size and the fixed regions, so a query must never name an
address without a token (it would attend uniformly and write non-integers into the
residual). The translation guarantees it by masking banks, and `Image::read` asserts
it. ROM sizes that are not a power of two, or above 8 MB, are rejected at insertion.
Every region above the ROM sits 0x60_0000 higher than in the 22-bit layout, so its low
bits are unchanged and its region bit is 23 instead of 21 (`docs/perf-log.md`, "MBC5 at
8 MiB: the 24-bit layout").

The markers are 128-byte blocks because the low seven bus bits pass through into them;
DEC15's one-byte markers would have needed a selection on bits 0..6 in every read.

## 5. Reads and writes inside the GBC CPU circuit (built)

In the numbering, layer ℓ's attention runs before its MLP, and that MLP produces
circuit level ℓ+1. Wires of level ℓ are readable by layer ℓ's attention.

| Where | What |
| - | - |
| level 1 | the low seven bits of `PC+1−hb` and `PC+2−hb` and their carries (under the HALT bug the stream is PC, PC, PC+1, so no immediate mux); `sp±1`, `sp±2`; the six region wires of PC, HL and SP (`translate::Region::Wired`) |
| layer 1 attention | **register heads**: ten constant-address memory heads read the banking registers; the copy head (§8) |
| level 2 | the high nine bits of `PC+1`, `PC+2` from PC's high bits and the carry; the physical address of fetch 0 and of the stable read sources HL and SP (`sm83::translate`, one level: cases over the region wires and bank literals, one node per bit, the cases of one bit disjoint; the FE and FF pages as one case plus the bits each range adds); the region wires of SP+1, SP−1, SP−2 |
| layer 2 attention | **fetch head 0** reads `OP0` and, through eight class heads, its 256 opcode one-hots; a host-served fetch reads the HOST record the host kept |
| level 3 | physical addresses of fetches 1 and 2 (region tests over the address bits, since their heads must sit at layer 3) and of SP+1; nibble one-hots of `OP0`; opcode-field one-hots (`x`, `y`, `z`); the write translations of HL, SP−1 and SP−2 (a level floor, `Circuit::node_at`, keeps them off level 2) |
| layer 3 attention | **fetch heads 1 and 2** read `OP1 OP2`; every consumer of those bytes reads at level 4 or later (superseded: since then OP1 and OP2 are bytes of the opcode head's line unless they cross into the next, which one head reads, `docs/hardware-shaped.md` §18) |
| level 4 | the 256 opcode one-hots; one selector per read source from the field one-hots; the `rare` bus address (a16, BC or DE when the nibble one-hots name one of their opcodes, else 0); physical FF00+n |
| level 5 | `RP0` selected among the stable physical sources with VOID as the default (`translate::select`); the rare translation, gated by its selector and VOID otherwise (`translate::read_address_gated`); `RP1 = sp+1`; the bus address `RA0` for the host; the CB fields (`cbxy`, `cbz`, `cbw`: 48 wires in place of 256 one-hots) |
| layer 5 attention | **data heads**: three memory heads; `D0` is the sum of the stable head and the rare head, exactly one of which reads a real byte while the other reads VOID's zero (`translate::either` gives the evidence slots `main + rare − VOID`); `D1` reads `sp+1`; the host's late byte is added when a read lands in VOID |
| levels 6..9 | operand selection, ALU, zero flags, final muxes; the write translations of the isolated FF00+n, `rare0` (BC, DE, FF00+C, a16; region wires at level 6, the translation at 7) and a16+1; `WP0 WP1` selected among them; `WM0 WM1` (MBC2 RAM holds a nibble whose reads return `0xF0 \| nibble`, so its record holds `v \| 0xF0`) |
| level 10 | routing |

Depth grows from 5 to 9 levels and from 6 to 10 layers, with 1,726 wires and 1,438
nodes (unit G's
1,996 wires passed the 2,048 channels of hidden 2048 by 73; unit H removed the 64 copy
wires and a duplicate SP+1 adder; unit I's CB fields removed 248 and its second data
head, split adders and region wires added 58). Every wire has its own residual channel, so the
hidden size is 1024 times the smallest `n_const` whose channels hold the layout:
constants, 66 slot identities, 8 own bits, the wires, 37 memory channels and 12 routes.
Request slots stay independent of the data bytes. The circuit's `Circuit::eval_mem` takes a
memory oracle (a byte array) so `CircuitMemCore` keeps running SST and lockstep without
a checkpoint.

**What a step costs the engine.** The engine re-evaluates a gate once per changed
channel it reads, so a step's cost per pass is the number of its inputs that flipped,
and a wire's cost is the number of steps that read it. Unit I (perf-log "Unit I")
reshaped the circuit around that count, keeping depth 9. A translation's cases are
disjoint per bit rather than partitioning the domain (the forced-low bank rule touches
bit 14 alone, so a bank register bit now reaches seven steps, not every switchable-ROM
step of nine translations); the region tests of a source with a level to spare are six
wires (`Region::Wired`), so a jump re-evaluates a step once instead of once per flipped
address bit; the `rare` bus tests the opcode by its two nibble one-hots instead of eight
bits, which moves its translation to the data heads' layer and makes the rare read a
second head whose translation lands in VOID unless selected, `d0` being the sum of the
two; the fetch adders split at bit 7 so a change in PC's low bits reaches a carry and
not nine high bits each; and the CB opcode decodes by field. Each change is one commit
with its census before and after.

**Why one translation per source.** The engine re-evaluates a gate when a channel it
reads changes. A translation of the *selected* bus address flips between 0 and an
address on every step that accesses memory, and every step of the translation that
reads a changed bit updates. The three translations and the request mux made 62% of the
gate updates of a text-box pass. Translating each stable source once (HL, SP, SP−1,
SP−2, FF00+C, FF00+n) and selecting the 22 physical bits, the host flag and the record
flag by one wire per source leaves a translation's nodes still until its own source
changes. The sources read from the instruction bytes or seldom used (a16, BC, DE) share
one bus address that is zero unless one of their opcodes executes, so their translation
moves only on those steps. The circuit must stay 0/1 on copy-mode inputs too (a record
runs it with every input slot zero, §8); it does whenever the banking registers are
consistent, which `sm83::tests::copy_mode_inputs_keep_every_node_in_range` checks.

**Result slots added.** The step adds `OP0' OP1' OP2'` (the bytes the model fetched), `RP0a..c
RP1a..c` (physical read addresses), `RV0 RV1` (the bytes the model read), `WP0a..c
WP1a..c` (physical write addresses), `WM0 WM1` (the bytes recorded). `WA* WV* WQ` stay
for the host mirror; `WQ` bits 2 and 3 say a record is kept; `RQ` bits 2..6 say the host
serves read 0, read 1, fetch 0, 1, 2. 47 positions become 66.

**Writes.** A CPU write at bus address `a` with value `v` is one of:

| Bus range | Model | Host |
| - | - | - |
| `0000–7FFF` | register record at `REGS + r` by mapper and sub-range (MBC1 `RAMG ROMB0 ROMB1 MODE`; MBC2 by `a.8`; MBC3 `RAMG ROMB0 RAMB`, the latch to the clock's latch port on a board with the clock (VOID otherwise), and `RAMG` and `RAMB` records also reach the clock (`layout::port`); MBC5 `RAMG ROMB0 ROMB1 RAMB`; ROM-only nothing) | mirror (the cartridge keeps its registers for saves and for `Machine::physical` in lockstep) |
| `A000–BFFF`, `C000–FDFF`, `8000–9FFF`, `FE00–FE9F`, `FF80–FFFE` | write record at the physical address, unless RAM is disabled or MBC3 selects the clock | mirror (PPU renders from its VRAM/OAM; battery saves read `cart.ram`) |
| `FF4F` (VBK), `FF70` (SVBK) on CGB | register record | mirror (host devices read them) |
| `FF00–FF7F` other, `FFFF`, `FEA0–FEFF` | nothing | the write, as today |

The OAM-DMA bus rule (reads return `0xFF` and writes are dropped outside HRAM while a
DMA runs) is host control logic until the PPU moves in. The host passes it as `CTL`
bit 4; the circuit sends every blocked access to the FF block, so the model serves the
`0xFF` itself and drops the record. The bytes a DMA moves are records too (§8a).

## 6. Host/model boundary (built)

The boundary comes from `machine::route` and `Machine::fixed_physical`:

| Bus address | Served by | Lockstep check (`Core::audit`) |
| - | - | - |
| ROM, cartridge RAM, WRAM, VRAM, OAM, HRAM | model | `RP` equals `Machine::fixed_physical`; `RV` equals the bus byte; `OP'` equals the bytes the host read at PC |
| disabled cartridge RAM, unmapped ROM, anything but HRAM during OAM DMA | model, FF block | the host reads `0xFF` |
| unmapped I/O, `FEA0–FEFF`, VBK and SVBK readback | model, constant cells (IO block 0, VOID, TAB) | `RV` equals the bus byte |
| I/O registers, `FFFF`, the MBC3 clock | host, from the device the emitted `RP` names (`layout::port`), through the late slots `D0 D1`; a write goes to the device its `WP` names; a fetch through a HOST record the host keeps before the pass | `RP` and `WP` equal `Machine::fixed_physical`, and the port the host served from equals that address's port; `OP'` equals the bytes the host read at PC |
| a DMA's source and destination bytes | model, copy-mode records (§8a); the host decides which bytes and when | the byte the model read equals the byte the host moved; `Core::audit_memory` compares VRAM and OAM after each DMA and every writable region at each frame end |

The host runs the reference byte arrays behind the checkpoint only as the shadow and
the device mirror. `--core model-mem` reports the share of steps the host served, the
way `--warmup` labels reference frames. It reports 2,059 of 68,159 steps in Yellow's
boot and 11,688 of 902,094 in 300 CGB frames (I/O-page reads of the joypad, LY and
timers). A step with a
host-served fetch or read takes a second (or third) pass; every other step is one pass,
where the host checkpoint needed two for 24% of gameplay steps.

## 7. Engine plan variant (built)

`Plan::Memory` is the template plus the memory heads. The plan makes these build-time
checks, and refuses the checkpoint when one fails:

- attention after layer 0 is live only in the declared memory heads;
- every memory head's query is recovered from `W_q` (wire terms and a constant per
  address bit) and the whole head is re-emitted through `llama::memory_head` and
  compared weight for weight, so every head carries the record pattern of one `Params`;
- the three address-gather heads of layer 0 feed channels no layer and no head-table
  row reads, so the fast path leaves them out (their output differs at `WM1`, which
  would otherwise break the shared layer-0 output);
- the copy head's query is the gathered address, which no CPU pass sets (every CPU pass
  holds mask tokens at the `WP` positions), so the engine serves it as address 0;
- no memory head sits at layer 0.

At insertion, `Params::check(tokens + window)` holds the margins at the longest
position the cache can reach before a compaction. At run time every queried address
must have a token (`Image::read`), a query form outside 0/1 refuses the pass, and wires
stay in {−1, 0, 1} as today.

On the fast path the circuit layers run as the template does. Before each layer's MLP
the engine reads the layer's memory heads from the index (the newest byte per physical
address, which the margin argument proves equals the attention result) and routes the
changed output bits like any other change. A kept write updates the index and appends
the token's key and value, by the formula §3 proves equal to the layers' output, to the
f32 cache the oracle and the dense fallback read. Any refused pass runs
`forward_cached` over the cache for that step and counts in `Stats`;
`GBCFORMER_ORACLE=N` recomputes every Nth kept record's key and value with the dense
pass over the step's own tokens and compares bit for bit.

## 8. Compaction (built)

When the window fills (29,444 write tokens at gap 256), the cache rolls back to the ROM
prefix and every writable byte (cartridge RAM, WRAM, VRAM, OAM, HRAM, the two marker
blocks, the registers) is re-emitted as a snapshot record. In the model this is copy
mode, a read record `REC WP0a WP0b WP0c RD`, whose copy head (layer 1, query from the
gathered address) reads the byte and whose `RD` routing decodes it as the `SNAPV`
token (`tests/memory_dense::reader_records_read_the_cache_through_the_model`). The
engine shortcuts each record as the index read plus the formula, and the sampled oracle
runs the read record through `forward_cached` and compares the decoded token and the
kept key and value. §9 has the measured cost and rate.

The engine's cache (`llama::KvCache`) holds the snapshot as one (address, byte) per
record and forms its key by the formula at the record's own index when the dense pass
reads it, so a compaction costs the copy of the writable regions (0.1 ms) and the
oracle pays the RoPE once per call. A compaction runs between steps, when the window
cannot take another step's two records, so every record of a step sits at the step's
position; the stock-stack view is unchanged (a rollback to the ROM prefix and a prefill
between two steps).

## 8a. DMA copies (built)

OAM DMA (160 bytes into OAM), CGB HDMA (16 bytes into VRAM per HBlank) and GDMA (up to
2 KB at once) are copies the host performs on its byte arrays (`Machine::apply_signals`).
Each byte is also a record in the model's memory, in copy mode. A reader record
`REC WP0a WP0b WP0c RD` sits at the source, and its copy head reads the byte from the
model's own memory. The write record `REC WP0a WP0b WP0c WM0` follows at the
destination, kept at the next position with the write key, so it beats every earlier record for the address
by recency as a CPU write does. Each byte is its own record group between CPU steps, in
the order the host copied them.

- **Placement (a project rule).** The weights compute every effective bank, and the
  host places a record only by bit concatenation (region, bank, offset) of values the
  model emitted; the host never applies a mapper rule. The machine logs each byte as
  the bus addresses it moved between and the byte (`memory::Copied`), and
  `GameBoy::step` hands the log to the core after the advance (`Core::copied`). The
  memory core places both addresses with `layout::place`. An address in a banked window
  is the window's first byte, from the last pass's MAP slots (`Evidence::map`), joined
  with the address's offset in the window (its low seven bits when the window's byte is
  in the FF or VOID block). A VRAM address takes its bank from bit 0 of the VBK record
  the model kept (never written on DMG). An address in an unbanked region is the
  region's base joined with the offset. The DMA block rule is off for the DMA's own
  accesses. `layout::tests::placing_by_concatenation_equals_the_translation` holds
  `place` to `translate` at all 65,536 addresses under 300 random registers of every
  mapper. Before this, the machine translated the copies itself under the host's own
  banking (`layout::translate` in `Machine::apply_signals`), a host mapper rule.
- **The guard.** A MAP slot names the banks under the registers before its step's
  writes. When the step kept a record in REGS (a banking register's write), no pass has
  emitted the banks the following copies land under, and a copy that needs a window
  stops the run (`MemCheckpointCore::windows`, `layout::place`). So does a copy from or
  into cartridge RAM smaller than its window, whose offset the RAM's size would mask
  (RAMMASK's record; no gate cartridge has one). VRAM, OAM, HRAM and WRAM bank 0 need
  no window. A GDMA fires inside the step's own write to FF55 and an OAM DMA byte
  inside the advance, so both land after the step's records. One step cannot write
  FF55 and a memory byte, since its two writes are adjacent addresses and FF54 and
  FF56 are I/O. Under memory lockstep each placement is also held to `translate` under
  the registers the model holds, and the byte the model read to the byte the host
  moved. With memory on the host (no memory core), nothing is placed.
- **Engine.** `Engine::copy` serves the reader from the index, as compaction does (§8),
  and keeps the write record through `keep`, so it gets the formula key, the
  compaction check and the sampled oracle. When the oracle samples the write record,
  the reader record runs through `forward_cached` too and must decode to `SNAPV` of
  the byte.
- **What the host still does, labelled.** Which bytes to copy and when (FF46, HDMA1-5,
  the HBlank schedule) is control logic. With `--weights control` it comes from the
  control checkpoint (`docs/control-in-weights.md` §13.3); the host still performs
  the copies and adds the M-cycles since the pass to the OAM DMA index. The run
  report gives the byte count. A source in the clock's port takes the host's byte and
  is counted apart; a source in the I/O page or IE is refused (`layout::place`), since
  no concatenation names its device. None occurred in any gate run.
- **Audit.** Under memory lockstep the byte the model read must equal the byte the host
  moved. `Core::audit_memory` holds the model's image to the host's VRAM (both banks)
  and OAM after every step that copied, and to every writable region (cartridge RAM,
  WRAM, VRAM, OAM, HRAM) at each frame end. With the audit and no records,
  lockstep-mem CGB 300 stops at Yellow's first OAM DMA: `memory audit:
  0x22c000 holds 0x00 in the model's memory and 0xa0 on the host`.

Phase 3 can build on this interface. The control logic in weights would emit the
(source, destination) pairs that `memory::Copied` carries today, and the host's byte
would leave the interface.

Copies are 156 bytes per frame in Yellow (one OAM DMA, plus HDMA in the CGB intro).
They raise the record rate, and so the compaction rate, by the copied share. The copies are 46,912 of
124,621 records over CGB 300's frames, 238,272 of 1,895,674 in Crystal, 232,480 of
1,282,471 in Zelda DX and 457,024 of 943,315 in Mario DX, whose HDMA traffic nearly
doubles its records. The fps cost is in §9 and perf-log "DMA copies as records".

## 9. Where the time goes (measured)

The numbers come from `tools/bench/text.sh` and `tools/bench/overworld.sh` run with
`CORE=model-mem` against `CORE=model`, interleaved, median of 3 (`docs/perf-log.md`,
"Memory in weights", holds every run and the load):

| Workload | model | model-mem, unit E | model-mem, unit G | model-mem, unit H | model-mem, unit I |
| - | - | - | - | - | - |
| text.sh | 63.9 fps, 2.34 µs per pass, 512 units per pass, 4.0M passes | 17.0 fps, 10.95 µs, 4,314 units, 3.22M passes (one per step) | 21.9 fps, 8.49 µs, 2,093 units (A/B against E at load 9 to 19: 15.0 fps, 12.39 µs) | 26.0 fps, 7.15 µs, 2,082 units, 6.3 index reads (six chained A/Bs at load 4 to 17, base 8.64 µs) | 30.4 fps, 6.12 µs, 1,416 units, 6.5 index reads (five chained A/Bs at load 3 to 8, base 7.07 µs) |
| overworld.sh | 146.7 fps, 2.87 µs, 589 units | 47.9 fps, 10.56 µs, 3,490 units | 62.9 fps, 8.03 µs, 1,938 units (A/B against E: 44.7 fps, 11.30 µs) | 72.6 fps, 6.96 µs, 1,931 units, 5.4 index reads (base 8.04 µs) | 81.5 fps, 6.20 µs, 1,414 units, 5.6 index reads (base 6.94 µs) |

Unit I's clean split of the final binary on text.sh puts the fixed cost per pass at
about 1.1 µs (the 42-unit host-served passes) and the units with the wire changes that
drive them at about 4.5 µs; a removed unit returned 1.3 to 1.5 ns in each of its A/Bs,
below the 3.2 ns average, because the pass also pays per changed wire, which the census
does not count (perf-log "Unit I"). Unit G removed 2,221 units and 3.9 µs
per pass. Unit E's units were the three translations of muxed bus addresses (`WP0`
1,131, `RP0` 718, `WP1` 451 gate updates per pass) and the request mux (`RA0` 387), 62%
of the pass. Unit G's census on text.sh lists the `rare` bus and its translations (`rare`
195, `RRARE` 113, `WRARE0` 81, where each `rare` node reads the eight `OP0` bits), the
fetch path (`pc+2` 145, `pc+1` 110, `pc+` 93, `F0 F1 F2` 84, 85, 81) and the one-hots
from the fetched bytes (`op0` 90, `cb*` 75, `op1` 64, `op*` 58). A pass with 42 units costs 2.7
to 4.1 µs under the same load, so the fixed cost per pass (16 index reads, 66 positions,
10 layers of marks, the attention plan and the decode of 46 positions) is now the larger
share and the next target; the perf log ("Unit G") has the runs and the load.

- The prefill takes 0.16 to 0.20 s for 1,131,049 records (was 0.67 s), with 290 MB of KV.
- Each step has four more MLP layers than the host checkpoint, 19 more positions and 17
  memory heads (ten registers, three fetches, three data since unit I split read 0 into a
  stable and a rare head, the copy head) of which the engine reads 6.5 per pass in the
  text box and 5.6 in the overworld, since unit H skips a head whose query channels
  and image are unchanged; one pass per step except host-served ones.
- Compaction runs once per 29,443 write records and takes 0.1 ms (was 42 to 51 ms).
- The KV cache holds 256 B per token on the engine for the ROM prefix and the writes,
  and 3 B per snapshot record (§8). A stock stack would hold K/V for every layer and head, which is
  why the per-head oracle exists.

## 10. What blocks moving the PPU in next

`docs/ppu-in-weights.md` is the Phase 1 prototype built on this list: a one-scanline
circuit in lockstep with `machine::ppu`, three shapes measured, and the two-thread
design. The points below stand.

- **One model.** The PPU's circuit must live in the same weights and attend to the same
  cache, since a second checkpoint cannot see this one's KV cache. That means one
  checkpoint with mode-selected circuits (CPU step, PPU scanline, copy mode), the way
  `REC`/`RD` already select copy mode inside this one.
- **Read bandwidth.** A memory head reads one byte per pass. A scanline reads about 100
  VRAM and OAM bytes, so the PPU needs either that many heads or that many passes; the
  engine serves a head as one index lookup, so heads are cheap on the fast path but
  each costs 32 dims of `W_q/W_k/W_v/W_o` in the dense parameters.
- Memory heads cannot sit at layer 0 (§3), so a PPU circuit that reads VRAM pays at
  least one layer before its first byte arrives.
- **Bus rules the host still owns.** These are mode-3 VRAM blocking (not emulated today, so no
  blocker yet), OAM DMA (the `CTL` bit) and the VBK/SVBK mirrors of the shadow machine.
  The I/O page's dispatch is the model's: each access lands in its device's port block
  and the host serves it from there (`docs/host-in-weights.md` §3 step 2).
- OAM DMA and HDMA copies are records (§8a, unit J), and the image audit shows the
  cache's OAM and VRAM equal to the host's after every DMA. The DMA schedule itself is
  still host control logic (Phase 3).
