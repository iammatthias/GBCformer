# Hardware tests outside the reference

Lockstep holds the model to the Rust reference devices, so a gap the two share passes
unnoticed. This page records outside test ROMs, each judged by its own pass signal, and
the accuracy work they drive.

## 1. The harness

`tools/hwtests/fetch.sh` fetches the ROMs into `vendor/hwtests/` (gitignored), each
download checked against a pinned sha256 or commit:

- `gbtr/`: c-sp/game-boy-test-roms v7.0, prebuilt. It holds the mooneye-test-suite
  (MIT), SameSuite (MIT), Mealybug Tearoom (MIT), blargg's ROMs, dmg-acid2, cgb-acid2
  and cgb-acid-hell, with the screenshots each suite judges by.
- `daid/`: testroms/daid of gbdev/GBEmulatorShootout (MIT) at commit `38b926bd`.
- `sameboy/`: SameBoy v1.0.5's own boot ROMs (MIT), the open boot ROM substitute.

`cargo run --release --example hwtests -- [--filter TEXT] [--jobs N] [--weights LIST]
[--lockstep LIST] [--one-model]` runs every case and prints one line per case: `PASS`,
`FAIL`, `REFUSED` or `NA`, the suite, the ROM, the console, what decided it, the mark
(`APPLIES` or `NA`), SameBoy's verdict and the area (§2.3). The judges are the suites'
own:

| Suite | Signal | When |
|---|---|---|
| mooneye, SameSuite, Mealybug `dma/` | B, C, D, E, H, L = 3, 5, 8, 13, 21, 34 | at the first `LD B,B`, within 120 s |
| Mealybug `ppu/`, acid2, cgb-acid-hell | the screen equals the suite's image | the frame the first `LD B,B` ran in |
| blargg bundles, daid | the screen equals the suite's image | after the emulated seconds c-sp lists (daid: 60 frames) |
| blargg singles | serial "Passed", or cartridge RAM signature `DE B0 61` at A001 and status 0 at A000 | within 60 s |

Screens compare in the suites' common palette: DMG shades as 000000, 555555, AAAAAA
and FFFFFF, RGB555 expanded as `c << 3 | c >> 2`, each channel within 4 (blargg's images
carry FDFDFD for white). The consoles are DMG-CPU C and CPU CGB E.

Each case is a ROM on one console, and it has one of two marks. `NA` means the ROM's own
name or documentation leaves that console out: the case is listed with its reason and
source and is not run. `APPLIES` means it runs under its judge. The rules are tables in
the runner (`SUFFIXES`, `NOT_ON`, `APPLIES`), and each row names its source:

- A model suffix, in mooneye's scheme, which SameSuite and Mealybug share. Mooneye's
  README ("Test naming") defines it: single models (`dmg`, `mgb`, `sgb`, `sgb2`,
  `cgb`, `agb`, `ags`), SoC revisions (`0`, then letters) and groups. The README gives the
  groups as:

  > * G = dmg+mgb
  > * S = sgb+sgb2
  > * C = cgb+agb+ags
  > * A = agb+ags

  So `-dmgABC`, `-dmgABCmgb` and `-GS` cover DMG-CPU C;
  `-C`, `-cgb`, `-cgbABCDE` and `-cgbDE` cover CPU CGB E; `-dmg0`, `-mgb`, `-sgb`,
  `-sgb2`, `-S`, `-A`, `-cgb0`, `-cgb0B`, `-cgb0BC` and `-cgbB` cover neither. A suffix
  missing from the table stops the runner.
- A ROM without a suffix applies to both consoles. Every unsuffixed mooneye acceptance
  test's source header lists CGB among its verified passes (mooneye-test-suite
  `8d742b9d55`), and SameSuite's apu README says CPU-CGB-E passes all its tests.
- A console its documentation names. blargg's `oam_bug` readme says it verifies the bug
  on DMG, and the sound readmes say each set is for DMG or for CGB. c-sp's blargg notes
  call `interrupt_time` a CGB-only ROM. Mealybug's seven `*_change2` ROMs have only a
  CPU CGB C image.

The mark decides only whether a case runs. The suites a console runs stay as before:
mooneye's `emulator-only` and blargg's `cpu_instrs` and `mem_timing` singles run on DMG
only, SameSuite on CGB only. A Mealybug ROM with no image for a console and no statement
leaving that console out has no judge there, so it is not listed. Mealybug's CGB images
are the CPU CGB D ones, and Mealybug publishes none for CGB E, the revision run here. No
source shows CGB E drawing them as CGB D does, and SameBoy's verdict as CGB E is unknown
for every one, so the Mealybug `ppu/` cases on CGB are `NA`: an expected image for
hardware we do not model is not a bar. The two `dma/*-C` ROMs judge by registers and
stay `APPLIES`.

Mealybug's DMG rows are all judged against one revision, the DMG-blob images
(`{name}_dmg_blob.png`). The suite's README publishes its DMG results as that one set
(`expected/DMG-blob/`). The c-sp bundle also ships `_dmg_b.png` images, but only for
`m3_lcdc_bg_en_change` and `m3_lcdc_win_en_change_multiple_wx`, so no DMG-B census can
cover the console. SameBoy's DMG model is DMG-B, and SameBoy fails both of those rows
against the blob images. To pass them, the reference departs from SameBoy in two places
(`machine::fifo`, `docs/ppu-in-weights.md` §13.1). First, an LCDC write's first stage
already clears WIN_EN, and BG_EN too when the write lands at x 0. Second, a window started
at WX = x + 6 does not step the LCD's x back. SameBoy's own comment says that step-back
happens "only on some DMGs". Against the two DMG-B images the reference is 228 and 3
pixels off. The two rules touch only LCDC writes during mode 3 and the WX − 6 start. After
the change no mooneye, SameSuite or daid DMG row changed verdict in the reference census
(§3 has the census), so no other suite in the census expects behaviour the blob choice
contradicts. That one reference passing every such row is the evidence.

`REFUSED` means the machine refused to run the case: every one in the baseline is a
DMG cartridge on CGB, which needs compatibility mode (item 5).

## 2. Baseline, the reference before this work

The reference alone (`--weights none`), before the hardware-accuracy work
began. Per ROM: `docs/hardware-tests/baseline.tsv`.

| Suite | Pass | Fail | Refused |
|---|---:|---:|---:|
| mooneye | 60 | 34 | 58 |
| SameSuite | 3 | 63 | 3 |
| Mealybug | 1 | 25 | 20 |
| blargg bundles | 7 | 7 | 0 |
| blargg singles | 33 | 16 | 0 |
| acid | 2 | 1 | 1 |
| daid | 2 | 3 | 3 |

### 2.1 Failures by cause

Each failure is put under the one cause that explains it. Causes outside the nine items
are named so they are not lost.

- **CGB compatibility mode (item 5).** All 85 refused cases.
- **Access timing inside an instruction (not among the items).** The step contract
  (`step.rs`) commits an instruction's reads and writes at the step's boundaries, so a
  test that places an access at one M-cycle of an instruction sees it at another.
  blargg `mem_timing` 01-03 report it directly (`F0:2-3`: LDH A,(n) reads at cycle 2,
  expected 3). The same cause explains mooneye's `add_sp_e_timing`, `call_timing`,
  `call_timing2`, `call_cc_timing`, `call_cc_timing2`, `jp_timing`, `jp_cc_timing`,
  `ld_hl_sp_e_timing`, `pop_timing`, `push_timing`, `ret_timing`, `ret_cc_timing`,
  `reti_timing`, `rst_timing`, `di_timing-GS`, `halt_ime1_timing2-GS`,
  `oam_dma_start`, `oam_dma_restart`, `oam_dma_timing`, `tima_write_reloading`,
  `tma_write_reloading` and `interrupts/ie_push`. It is a gap the GBC CPU circuit
  shares with the reference: fixing it changes the step contract and the circuit. §4.1
  has the design.
- **PCM12 and PCM34 (not among the items).** CGB's FF76 and FF77 read the channels'
  digital outputs. The bus routes them to nothing (0xFF). 43 of SameSuite's 60 APU
  tests name FF76 or FF77 eight or more times (a byte search); they fail at the first
  such read whatever the APU does.
- **Mode-3 length (item 1).** mooneye `hblank_ly_scx_timing-GS` and
  `intr_2_mode0_timing_sprites`.
- **Mid-line register changes (not among the items).** Mealybug's `m3_*` tests and
  daid's `ppu_scanline_bgp` change BGP, LCDC, SCX, SCY, WX or OBP0 during mode 3. The
  renderer draws a whole line at mode 3's start with the registers of that moment, so
  variable mode-3 length alone does not flip these.
- **PPU interrupt and LCD-on timing.** mooneye `intr_1_2_timing-GS`,
  `intr_2_0_timing`, `intr_2_oam_ok_timing`, `lcdon_timing-GS`,
  `lcdon_write_timing-GS`, `stat_lyc_onoff`; blargg `oam_bug` 1 ("Turning LCD on
  starts too early in scanline").
- **The OAM corruption bug (not among the items).** blargg `oam_bug` 2, 4, 5, 7, 8: on
  DMG, 16-bit INC/DEC and some reads with an address in FE00-FEFF during mode 2 corrupt
  OAM.
- **DMG wave RAM (item 6).** blargg `dmg_sound` 09, 10 and 12, and on CGB
  `cgb_sound` 09.
- **Ordinary STOP (item 3).** daid `stop_instr` on DMG.
- **Speed-switch LY (item 4).** daid `speed_switch_timing_ly`, 109 pixels.
- **Boot state.** mooneye `boot_div-dmgABCmgb` (DIV's phase at the hand-off) and
  `serial/boot_sclk_align-dmgABCmgb` (the serial clock's phase).
- **Others.** mooneye `oam_dma/reg_read` (FF46 reads back), MBC1 multicart
  (`multicart_rom_8Mb`), SameSuite `dma/gdma_addr_mask`, `hdma_lcd_off` and
  `hdma_mode0`, Mealybug `hdma_during_halt-C` and `hdma_timing-C`, cgb-acid-hell (2
  pixels).

### 2.2 The census now

As of one HBlank block a line across the gap between STAT's
mode 0 and the request (§5; the same cases as the build before it, case for case, none gained or
lost), and before it (the OAM corruption bug in
weights, HDMA through a HALT, the CGB compatibility-mode registers, the STAT edges and a
DMG's read and write locks of §3.14-§3.17, then a CGB's mode 0 at the dot and
`hdma_timing-C`'s HDMA rules of §3.14), measured by `verify.sh`: the reference column with every component on the
host, the in-weights column with all five in weights and all five in lockstep. blargg's
`7-timing_effect` on DMG is NA now (§3.14), so 321 cases apply and 127 are NA. The two
columns match case for case, and no lockstep, guard or refusal stopped a run. The result is
**321 of 321 in the declared census, post-boot**: each case starts at the hand-off from the
state the host sets up without a boot ROM, as `examples/hwtests` does unless it is given one
(§2.4 pins the census).

| Suite | Pass / applicable | In weights | NA |
|---|---:|---:|---:|
| mooneye | 152 / 152 | 152 / 152 | 42 |
| SameSuite | 69 / 69 | 69 / 69 | 7 |
| Mealybug | 26 / 26 | 26 / 26 | 41 |
| blargg bundles | 14 / 14 | 14 / 14 | 4 |
| blargg singles | 48 / 48 | 48 / 48 | 33 |
| acid | 4 / 4 | 4 / 4 | 0 |
| daid | 8 / 8 | 8 / 8 | 0 |
| all | 321 / 321 | 321 / 321 | 127 |

Against the pixel FIFO census's 310 (below), 10 cases go from FAIL to PASS in both columns and none the
other way: mooneye `ppu/lcdon_timing-GS` and `lcdon_write_timing-GS` on DMG (§3.17), blargg `oam_bug` (the bundle) and its singles `2-causes`, `4-scanline_timing`,
`5-timing_bug` and `8-instr_effect` (§3.14, §4.4), Mealybug `dma/hdma_during_halt-C`
(§3.14), and mooneye `misc/boot_hwio-C` and `misc/bits/unused_hwio-C` (§3.15); the build
with the OAM corruption bug in weights had 320 of 321 in both columns, and Mealybug `dma/hdma_timing-C` on CGB passes since
(§3.14). Before §3.17, both columns were 318 of 321.

The census with the pixel FIFO and its stepper
(`docs/ppu-in-weights.md` §13, §14), merged with the APU as
SameBoy's port (§3.12 "The APU") and the access timing's A3. The reference column
is the runner with every component on the host, per case in
`docs/hardware-tests/after-ppu-fifo-green.tsv`. The in-weights column runs all five
components in weights with all five in lockstep (`--weights all --lockstep all`), per
case in `docs/hardware-tests/in-weights-ppu-fifo-green.tsv`. Both were measured on
the merged build. The two columns match case for case, and no lockstep or guard stopped a run.
The census before the marks had 342 cases; 322 of them apply. The other 20 are the
Mealybug `ppu/` cases on CGB, which are `NA`. The 126 `NA` cases also include the (ROM,
console) pairs the runner used to drop without listing. SameBoy's column counts its
passes among the applicable cases; the rest are SameBoy fails or `unknown`.

| Suite | Pass / applicable | In weights | NA | SameBoy pass | SameBoy unknown |
|---|---:|---:|---:|---:|---:|
| mooneye | 148 / 152 | 148 / 152 | 42 | 96 | 56 |
| SameSuite | 69 / 69 | 69 / 69 | 7 | 67 | 2 |
| Mealybug | 24 / 26 | 24 / 26 | 41 | 15 | 2 |
| blargg bundles | 13 / 14 | 13 / 14 | 4 | 5 | 9 |
| blargg singles | 44 / 49 | 44 / 49 | 32 | 48 | 1 |
| acid | 4 / 4 | 4 / 4 | 0 | 3 | 1 |
| daid | 8 / 8 | 8 / 8 | 0 | 8 | 0 |
| all | 310 / 322 | 310 / 322 | 126 | 242 | 71 |

Against A3 (224 in both columns), 86 cases go from FAIL to PASS and none the other way:
the 60 SameSuite APU cases, from the APU port, and 26 PPU mid-scanline cases, from the
pixel FIFO. The 26 are the 23 applicable Mealybug DMG `ppu/` rows, cgb-acid-hell, and
daid `ppu_scanline_bgp` on both consoles. Nine of the Mealybug rows are ones SameBoy
fails too; they pass on the DMG-blob rules of `docs/ppu-in-weights.md` §13.1.

The 12 applicable cases that still fail, the same 12 in both columns, by cause:

- **The OAM corruption bug** (6). blargg `oam_bug` (the bundle) and its singles
  `2-causes`, `4-scanline_timing`, `5-timing_bug`, `7-timing_effect` and
  `8-instr_effect`, all on DMG. Built since (§3.14, §4.4): all pass, and `7-timing_effect`
  is NA.
- **The LCD-on locks** (2). mooneye `ppu/lcdon_timing-GS` and `lcdon_write_timing-GS` on
  DMG need a read lock and a write lock that differ by an M-cycle at mode 3's edges. The
  locks are one pair per region (§3.12).
- **HDMA around HALT** (2). Mealybug `dma/hdma_during_halt-C` and `hdma_timing-C` on CGB.
  SameBoy moves no HBlank block while the GBC CPU is halted, and moves one at the wake in
  some cases (§3.12). Both pass since (§3.14).
- **`misc/boot_hwio-C`** (1, mooneye, CGB). Measured on the reference, frame 300: the ROM
  reports `MISMATCH AT $FF68, EXPECTED C8, GOT FF`. BCPS reads `0xFF` where the CGB boot
  ROM leaves `0xC8`. The cause in the hand-off state is not traced.
- **`misc/bits/unused_hwio-C`** (1, mooneye, CGB). Measured on the reference, frame 300:
  the ROM reports `TEST FAILED` at `$FF76` (PCM12) and shows the read with bits 3-0 set.
  §3.12 expected this case and `boot_hwio-C` to pass once the APU port, which builds
  PCM12 and PCM34, reached this branch. It has, and both still fail, so that inference
  was wrong. The cause is not traced. Both pass since (§3.15).

The earlier columns: A3 (§3.12) added 11 to both columns and lost none, per case in
`docs/hardware-tests/after-a3.tsv` and `in-weights-a3.tsv`: mooneye `di_timing-GS`,
`halt_ime1_timing2-GS`, `ppu/intr_1_2_timing-GS`, `ppu/hblank_ly_scx_timing-GS`,
`ppu/stat_lyc_onoff` on both consoles, `emulator-only/mbc1/multicart_rom_8Mb` and
`misc/ppu/vblank_stat_intr-C`, and SameSuite `dma/gdma_addr_mask`, `hdma_lcd_off` and
`hdma_mode0`. The in-weights census matches the reference case for case, with no
lockstep stop or refusal.

Before A3, the reference column was A2's build (§3.11): 43 cases flipped to pass and none
to fail against 170. The in-weights column runs the GBC CPU, memory, PPU and control in
weights with all four in lockstep, from A2's operand fetches in the memory checkpoint
per case in `docs/hardware-tests/in-weights-access-timing.tsv` with the NA
marks above. Against the build that added WC bit 3 (168), 45 cases flip to
pass and none to fail. Among them are mooneye `emulator-only/mbc5/rom_32Mb` and
`rom_64Mb` on DMG, which the 22-bit layout refused for exceeding 2 MiB and the 24-bit
layout passes (`docs/perf-log.md`, "MBC5 at 8 MiB"). It matches the reference case for
case. The last 12 to join it, against the merged build before them (201), are mooneye
`add_sp_e_timing`, `call_cc_timing`, `call_timing`, `jp_cc_timing`, `jp_timing` and
`ld_hl_sp_e_timing` on both consoles, whose operands the memory checkpoint now fetches
at their M-cycles (§3.11). No lockstep or guard stopped a run.

The NA cases by rule, counted in (ROM, console) cases:

- A model suffix that leaves the console out. That gives 42 mooneye cases: the
  `-dmg0`, `-mgb`, `-sgb`, `-sgb2`, `-S`, `-A` and `-cgb0` ROMs on both consoles, the
  `-GS` and `-dmgABC*` ROMs on CGB, and the `-C` and `-cgb*` ROMs on DMG. It also gives
  7 SameSuite cases (`-A`, `-cgb0`, `-cgb0B`, `-cgb0BC`, `-cgbB` on CGB) and 2 Mealybug
  cases (the `dma/*-C` ROMs on DMG).
- blargg's documentation names the console. That gives 4 bundle cases (`oam_bug` and
  `dmg_sound` on CGB, `cgb_sound` and `interrupt_time` on DMG) and 32 singles
  (`oam_bug` and `dmg_sound` on CGB, `cgb_sound` on DMG).
- Mealybug's `*_change2` ROMs, with only a CPU CGB C image, give 14 cases (7 ROMs on
  both consoles).
- Mealybug's `ppu/` ROMs on CGB, judged only by CPU CGB D images, give 25 cases.

SameBoy's verdicts are in `docs/hardware-tests/sameboy.tsv`, written by
`tools/hwtests/sameboy.sh`. Its sources, best first:

- SameBoy itself publishes no per-ROM pass list. Its CI (`.github/workflows/sanity.yml`
  at v1.0.5) runs five ROMs and checks their screenshots against fixed hashes. That is a
  regression check, not a pass verdict, and two of the five (`dmg_sound-2`,
  `oam_bug-2`) are not census files. Its features page says it passes all of mooneye's
  suite and blargg's ROMs, but names no ROM or console. The census uses neither.
- GBEmulatorShootout (gbdev.io/GBEmulatorShootout, updated 2026-06-04) supplies 251 of
  the rows. Each Shootout test runs on one model: DMG unless its definition in
  `testroms/*.py` at `38b926bd` says CGB or SGB. A verdict covers that console only, so
  an unsuffixed mooneye ROM's CGB case is `unknown`. The page does not name SameBoy's
  version. v1.0.3 was the newest release on that date (inferred from GitHub's release
  dates). The script stops unless SameBoy's cells add up to the page header's 253/264.
- c-sp's blargg notes say SameBoy made the expected `dmg_sound` and `cgb_sound` bundle
  screenshots. That gives PASS for those two bundles on their console.
- SameSuite's apu README at `f71b4b3c37` says SameBoy as CGB-E passes all but
  `channel_4_freq_change` and `channel_1_sweep_restart_2`. This is a suite-level
  statement, and the census uses it only for `channel_1_freq_change_timing-cgbDE`,
  which the Shootout does not run. The README and the Shootout disagree on those two
  tests: the Shootout, the newer source, has SameBoy passing both, and the census takes
  its verdict.
- No bundle verdict is inferred from its singles. SameBoy was not built here: its
  Makefile needs RGBDS for the boot ROMs, and RGBDS is not installed.

Before this census, `after-default.tsv` (153 pass) and `after-timing.tsv` (159 pass)
recorded the default path and the access-timing prototype; `baseline.tsv` is §2.

### 2.3 The work queue

As of the census of §2.2 no applicable case fails; mooneye's `lcdon` cases
pass since §3.17 and Mealybug's `hdma_timing-C` since §3.14.

No applicable case fails in either column of §2.2.

No applicable case fails here that SameBoy is recorded as failing. Nine pass here that
SameBoy fails, all Mealybug DMG `ppu/` rows (`m3_lcdc_*` and `m3_scy_change`), on the
DMG-blob rules of `docs/ppu-in-weights.md` §13.1.

### 2.4 The pinned census

The demonstrated result is **321 of 321 in the declared census, post-boot**, in both
columns. The census starts each case at the hand-off from the state the host sets up
without a boot ROM, as `examples/hwtests` does without `--boot-dmg` or `--boot-cgb`; with
them it runs the boot ROM first on the case's own stack, the memory checkpoint mapping it.
The model's own boot runs are held to the reference at the hand-off apart (§4.3).

- **Run.** The census run by the verify script, dated 2026-11-14:
  the reference column with every component on the host, the in-weights column with all
  five components in weights and all five in lockstep, separate checkpoints, and the declared
  sample below as the one model from the GGUF (62 of 62 pass). Before the step's fetched
  stream changed, the same two columns gave the same 321 of 321.
- **Checkpoint digests** (`cargo run --release --example digest`, sha1): cpu `bebf5d93`, cpu-memory
  `6acacbed`, apu `d8d3a06e`, control `6e225675`, ppu `79706b58`, ppu-stepper `4fdab521`, and the
  one model `eda1ea0b` (the layer-shaped GGUF stores 1,093.4 M parameters, 3,184.8 M as the
  padded uniform checkpoint, 1.85 M nonzeros).
- **As one model** (measured). The weights column stays separate
  checkpoints in lockstep: the census as the one model does not fit the machine at three
  shards. On shard 0 of 45 (10 cases, 6 applicable, all passing in every run), one process
  each: separate checkpoints 99 s, the one model compiled 280 s, the one model from the GGUF
  (`GBCFORMER_GGUF_ONLY`) 255 s; one case's peak resident memory is 6.8 GB with separate
  checkpoints and 29.9 GB as the one model (`/usr/bin/time -l`, `tim01_div_trigger`). Three
  shards of the one model would need about 90 GB of the machine's 64 GB, and one shard about
  2.6 times the separate column's 48 minutes at three shards, about 6.2 hours (estimated from
  that ratio). The verify script runs instead a declared sample: each suite's ten slowest
  passing cases in the separate column (`hwtests` prints each case's seconds), as the one
  model from the GGUF with every component in lockstep, one process (`hwtests --cases`):
  62 cases (acid 4, daid 8, and 10 each of blargg, blargg-single, Mealybug,
  mooneye and SameSuite), 62 pass, 0 fail, 0 refused, in 80 minutes.
- **Test-suite artifacts.** The ROM of every case, by sha1, in
  `docs/hardware-tests/artifacts.tsv` (288 files, hashes only; `tools/hwtests/fetch.sh` pins
  the archives they come from by sha256).
- **Exclusions.** Every NA case with its class, in `docs/hardware-tests/exclusions.tsv`
  (written from the census by `examples/hwtests`, which prints each NA case's class before its
  reason). 127 cases, in three classes:

| Class | Cases | What they are |
| - | - | - |
| wrong hardware | 87 | A model suffix that leaves the console out (51), the sound ROMs of the other console (26), `oam_bug` on CGB (9), `interrupt_time` on DMG (1) |
| missing oracle | 39 | Mealybug `ppu/` on CGB, whose images are CPU CGB D's (25), and the seven `*_change2` ROMs on both consoles, whose only image is CPU CGB C's (14) |
| broken artifact | 1 | blargg `oam_bug/rom_singles/7-timing_effect` on DMG, whose text routine overruns cartridge RAM (§3.14) |

**Outside the demonstrated result.** gbmicrotest (391 of 513 in an uncommitted probe,
`docs/perf-log.md`) and Mealybug's PPU set on CPU CGB E, for which no expected images
exist, are not part of the 321, and no claim here covers what they test.

## 3. What changed, item by item

Every rule below is on by default, in the reference and in the checkpoints that own
it; `--accuracy` is gone. So is access timing's switch (§4.1): every run lands each
access at its M-cycle.

### 3.1 Item 6, DMG wave RAM: reference and APU checkpoint

- While channel 3 plays, a DMG access reaches wave RAM only in the tick the channel read
  it, on that tick's last 2 dots (`Wave::just_read`); otherwise a read sees 0xFF and a
  write drops. CGB reaches the playing byte at any time, as before.
- A trigger's first read comes 6 dots after a period (SameBoy reloads its countdown 3
  units later). cgb_sound 09 reads it.
- On DMG a trigger within one timer unit of a read overwrites wave RAM's first bytes
  from the byte about to be read (`Wave::corrupt`, SameBoy's majority variant).
- The APU checkpoint carries the flag as state bit `jr3` and computes all three. The
  flag's equality reads the terms the new timer sums, so the circuit stays 12 levels
  deep (13 layers); a first version that compared the summed timer took it to 17.
- blargg `dmg_sound` 09, 10, 12, `cgb_sound` 09 and both sound bundles now pass, and
  with `--weights apu --lockstep apu` all 24 sound singles pass with the checkpoint in
  lockstep.

### 3.2 Item 5, CGB compatibility mode: reference and PPU checkpoint

- `Hardware::resolve` no longer refuses CGB for a DMG-only game. The machine holds
  `compat` (`Machine::cgb` is CGB mode), so VBK's bank, SVBK, KEY1, HDMA and the palette
  ports route nowhere, as on hardware.
- `ppu::Mode::Compat` draws as DMG does and looks each shade up in CGB palette RAM:
  background palette 0 for BGP, object palettes 0 and 1 for OBP0 and OBP1.
- Without a boot ROM the PPU starts from the palettes the CGB boot ROM loads for a
  title it does not recognise (background 7FFF 1BEF 6180 0000, objects 7FFF 421F 1CF2
  0000). `examples/boot_state` (below) against the real CGB boot ROM shows these palette bytes
  equal for dmg-acid2; a recognised Nintendo title gets its own palette, which only
  running the boot ROM gives. The GBC CPU starts from the compatibility hand-off
  registers, which the boot ROM run also matches.
- The PPU checkpoint takes FLG bit 3: the line draws as DMG, and a second palette
  head, beside the DMG one, reads the CGB palette line that BGP, OBP0 or OBP1 maps to.
  `tests/ppu_lines` draws a third of its scenes in compatibility mode with random
  palette RAM, and dmg-acid2 on CGB draws 7,556 lines equal to the reference with
  `--weights cpu,memory,ppu --lockstep all`.
- KEY0 as the boot ROM writes it is the memory model's (`Reg::Key0`, which its FF50
  translation reads) and the control logic's (its readback, reached by the record's
  second port); after FF50 a KEY0 write reaches nothing. With a boot ROM the memory
  model's COMPAT cell says whether the console left CGB mode, and the PPU checkpoint's
  FLG bit 3 is that cell as the memory model last recorded it. Without a boot ROM
  compatibility mode comes from the header, a labelled snapshot (§4.2).

### 3.3 Item 7, the boot ROM: reference

- `--boot FILE` (and `Session::power_on`) starts at power-on with the file mapped over
  0000-00FF, and on CGB also 0200-08FF. The boot ROM unmaps itself with its FF50 write,
  and the KEY0 it wrote picks compatibility mode then. The file is one you supply: it is
  read, never stored. `.gitignore` excludes `*_boot.bin`, and `tools/link-data.sh` links
  one into a worktree as it links ROMs.
- `examples/boot_state <boot.bin> <rom> [dmg|cgb]` ran the boot ROM to its hand-off and
  printed every difference from the hard-coded start. The example has since been
  removed. Against Nintendo's CGB boot ROM the
  GBC CPU's registers match for Yellow and for a DMG cartridge. The differences that
  remain are DIV (the boot's length), LY and STAT (the hand-off lands at LY 0x93-0x98,
  the hard-coded start at LY 0), wave RAM (the boot ROM leaves 00 FF pairs), SC, BCPS
  and the CGB-mode object palettes (0 after the boot, white hard-coded). The hard-coded
  start keeps these values, so no gate hash moves; a run with `--boot` has them right.
- The runner takes `--boot-cgb FILE --boot-dmg FILE` and judges each case from the
  hand-off. With Nintendo's CGB boot ROM and SameBoy's DMG one, every verdict equals the
  hard-coded start's but one: mooneye `boot_hwio-dmgABCmgb` fails with SameBoy's DMG
  boot ROM, whose I/O state differs from Nintendo's.
- The memory checkpoint maps the boot ROM itself (`docs/host-in-weights.md` §7): BOOTING,
  UNMAPPED, COMPAT and KEY0 are registers in its cells, bank 0 while it is mapped is
  `layout::BOOT` (bank 0 with the boot ROM over it), and its FF50 translation picks UNMAPPED or COMPAT. A run from
  power-on with every component in weights reads nothing from the header and hard-codes
  no post-boot state; the host follows the model's FF50 record.

### 3.4 Item 3, ordinary STOP: reference and control checkpoint

STOP without a speed switch armed resets DIV and holds the GBC CPU and the counter
until a selected joypad line falls (`Lines::paused`). A DMG's PPU stops and its screen
blanks white; a CGB's PPU runs on but cannot read VRAM and draws black, unless STOP came
in mode 3. daid's `stop_instr` on DMG and CGB and `stop_instr_gbc_mode3` pass, and pass
with all five components in weights and in lockstep.

The control checkpoint keeps three bits in `TC`: bit 5 `stopped`, bit 6 `blocked` (a
CGB's PPU kept from VRAM) and bit 7 the console, a CGB, fixed at load. STOP without KEY1
armed resets DIV as the switch does, drops the frame sequencer's bit the same way and
sets `stopped`; a joypad interrupt clears both bits. While stopped, the span counts in
`Q` as a speed switch's pause does (`FX` bit 5 is now the CPU held, by either), so the
timer, serial and OAM DMA hold, no HBlank block moves, no frame-sequencer edge falls
(`FE_HELD`), and a pause the STOP interrupts holds too. A DMG's STOP sets `FX` bit 4 (the
screen blanks) and holds the PPU's position by gating `H`; a CGB's runs on. The PPU
checkpoint takes FLG bit 4, `blocked`, and draws 0 at every pixel then. The reference
gained one fix on the way: its frame sequencer edge kept falling during ordinary STOP
though DIV held; `edge_horizon` now holds it as a pause does.

### 3.5 Item 1, mode-3 length: reference and PPU and control checkpoints

`Ppu::mode3_dots` gives each line's mode 3 when it is drawn: 172 dots, plus SCX's fine
scroll, 6 when the window shows, and per fetched object 6 more, the first object in each
background tile also waiting out `5 - min(5, (X + SCX) mod 8)`. The values give
the 21 rows of mooneye's `intr_2_mode0_timing_sprites` table that
`ppu::tests::mode_3_grows_as_mooneyes_sprite_table_says` holds them to. The control logic moves
mode 0 (STAT's mode, the mode-0 interrupt, the HBlank HDMA block) to `20 + dots / 4`.
The mooneye test still fails at its first case (one object at X 0, 183 dots, mode 0 from
M-cycle 65): the poll it expects to see mode 0 reads at M-cycle 64 of the line, because
the step contract serves `LDH A,(STAT)` at the instruction's start; hardware reads in its
third M-cycle. It waits on the access timing of §4.1.

In weights the PPU checkpoint emits the length as a shared output, `ML` (dots over 4),
from its OAM scan: each slot's X, whether it is fetched (X below 168), the tile and fine
position of X + SCX, pairwise tile equality and order, and a first-in-tile flag per slot
and fine position, summed with SCX and the window into the dots, whose bits 2-8 are the
byte. `PpuModel` holds it to the reference's under lockstep.

The control checkpoint takes `ML` as a late input. A pass whose span ends past M-cycle 63
of a visible line asks for it (`MQ`, its request), and decodes again with it; before
that the length cannot matter, since mode 3 lasts 43 M-cycles at the least. With `ML`
known, mode 0 begins at `20 + ML`; while it is not, at 63, so a horizon toward mode 0
stops there and the pass there asks. The host reads the length from `ppu::Lengths`, a
board each drawn line's length reaches: the reference publishes it as it draws, the PPU
checkpoint on its thread after its pass, and a reader that finds lines not yet drawn
first asks the PPU's worker to hand over the lines it queued. In Yellow 300 with all
five in weights, 21.7 passes a frame ask, and they wait 0.29 ms in all over 300 frames.

### 3.6 Item 2, the access locks: reference, control and memory checkpoints

The GBC CPU reads 0xFF from VRAM in mode 3 and from OAM in modes 2 and 3, and its writes
there drop. mooneye `intr_2_oam_ok_timing` passes on DMG and CGB with the locks.

The control checkpoint emits the locks after each pass (`LK` bits 0-1: VRAM, OAM) and how
long they hold at the least (`MX`), then the locks that follow (bits 2-3) and until when
(`MY`): mode 2 holds to mode 3's start, mode 3 to mode 0's (63 while the length is not
known, and then no second span), mode 0 to the line's end and the next line's mode 2 to
its mode 3. The host keeps them as `Lines::locks` while the M-cycles since the pass are
inside a span, and has none past it. The memory checkpoint reads them as CTL bits 5 and
6, with bit 7 saying they are current. Its translations send a locked VRAM or OAM access
to the FF block, so a read gives 0xFF and a write keeps no record. A step that reaches
VRAM or the FE page by its bus address while the locks lapsed sets RQ bit 7, and the
host asks the control model for them (a pass in place, as a STAT read is) and hands them
over in the late slot `LK`. A host-memory run asks the same way (`Machine::ppu_locked`),
and under lockstep the control model's locks are held to the reference's at every latch.
Yellow 300 with all five in weights asks 0 times: every VRAM and OAM access fell inside a
span a pass had emitted.

### 3.7 Item 8, a serial link partner: reference and control checkpoint

`io::Link::pair()` is a cable with two ends; `Session::link` plugs one into each of two
games, and `Session::run_linked_frame` steps whichever game is behind in emulated time.
Each end shows the other its SB as written. The side clocking a transfer (SC 0x81)
swaps the two bytes when its 8 bits are done; the other side takes its byte at its next
advance and, if it waited on the external clock (SC 0x80), completes and raises the
serial interrupt. Without a cable an internal-clock transfer still shifts in 0xFF and an
external-clock one never ends.

In weights the control checkpoint takes `SI`, the byte the other end shows (0xFF with no
cable, the line high), and shifts it in when its transfer ends; it emits `SX` bit 0 and
`SO`, the byte it shifted out, which the host puts in the other end's inbox. A byte in
this end's inbox is offered as a write event to FF03, which no bus access reaches: it
lands in SB while no transfer of this end's counts (`SX` bit 1, the host then takes it)
and completes a transfer waiting on the external clock. The host moves bytes between the
two machines' outputs and inputs; which byte crosses and when is the model's.
`tests/serial_link.rs` checks both ends on the reference and with the control logic in
weights. A cable with the control logic in lockstep is refused, since both sides would
drive it.

### 3.8 Item 4, the speed-switch LY: reference and control checkpoint

daid's `speed_switch_timing_ly` differed in one value of 128: its read 46, at the last
M-cycle of line 0x86, read 0x86 where hardware reads 0x87. LY shows the next line from
the line's last M-cycle (dot 452) on; at single speed a read rarely lands there, at
double speed every other one does. With that rule (`Lcd::set_early_ly`) the screenshot
is pixel-exact, and blargg `oam_bug` 1 (`lcd_sync`) passes too. The pause length
(`SWITCH_MCYCLES`) was not the cause and is unchanged. The control checkpoint has the rule:
its LY read takes the next line (`LY+1`, 0 after 153) when the position in the line is
halves 226-227, and `tools/speed-switch.sh --weights cpu,control --lockstep all` is
pixel-exact on all three. The rule is always on. It moves one gate hash: Zelda DX CGB 300
goes from `e2bd7038` to `f10b2b4a`, because the game reads LY in a line's last M-cycle;
Zelda DX at 1,500 frames keeps `99f51381`, and Yellow, Crystal and Mario DX keep theirs.

### 3.9 Host rules into weights: the DMA-blocked access, BCPD and OCPD

- **The access OAM DMA blocks.** The memory model already sent every access but HRAM's
  under OAM DMA to a constant 0xFF cell; it now sends them to their own block,
  `layout::BLOCKED` (I/O code 3, 0xFF, no record), and the host cuts the transfer's run
  when a model-emitted read or write address lands there (`Bus::blocked`). The host's
  `cpu_visible_during_dma` stays only in the reference bus that serves host memory.
- **BCPD and OCPD.** A data port read during mode 3 gives the control model's own 0xFF
  (`RV`, `PaletteRead::Byte`); outside it the model names the palette byte's index and
  the PPU checkpoint's memory serves it. The host only carries the index.
- **An HDMA block inside a GBC CPU step.** Hardware runs HDMA after an opcode fetch
  (SameBoy's `GB_hdma_run`), so a block the step's M-cycles move lands at the step's
  end. The host used to decide that: it kept every `Copy::Block` back while a step ran.
  Now the control logic decides it. The machine tells it when a step starts and ends
  (`ControlDevice::step_running`). The circuit reads that as the input `IN`, and sets
  `CL` bit 7 on a block the pass moves while a step runs. The host lands such a block,
  `Signals::held`, at the step's end. It no longer filters copies by kind. The
  reference `Control` holds its blocks the same way. `dma_stream` toggles `IN` at
  random, and two new mutants (a held block landed at once, a block outside a step
  held) stop the lockstep. The block's stall goes with it (`Signals::held_stall`), so
  the host counts it from the step's end because the model said so. The machine stops
  the run if a stall arrives inside a step without being held.

### 3.10 A CGB's OAM DMA from E000-FFFF: reference and control checkpoint

Hardware (SameBoy's `GB_dma_run`): a DMG's transfer from E000-FFFF reads the echo of work
RAM, a CGB's writes 0xFF to OAM. `OamDma` now does both, by console, and the copy contract
has a source-less byte, `Copy::Fill`. The control checkpoint emits `DF`, the byte every
OAM DMA byte of the pass is when no source drives it (0xFF on a CGB with `DR` at E0 or
above, else 0), and the memory model reads it from its own FF block. A machine test holds
OAM after a transfer from E000 to the echo on DMG and to 0xFF on CGB, and the random DMA
traffic in `control_circuit`'s tests writes E3 to FF46 on both consoles, in lockstep.

The earlier report that "an I/O-page source stops the run" does not hold up by test.
`layout::place` refuses a source in the I/O page, but no OAM DMA source reaches it: on a
DMG the echo rule sends FE and FF to DE and DF, and on a CGB no source is read. mooneye
`oam_dma/sources-GS` run on CGB with all five components in weights and in lockstep goes
its 300 frames with no stop (frame `0807c385`).

### 3.11 A2: GBC CPU, interrupt, OAM DMA, timer, serial and boot timing

Each rule below is in the reference and in the checkpoint that owns it, and the
in-weights lockstep games hold their hashes unless a row says otherwise.

- **OAM DMA blocks by bus.** A running transfer blocks OAM always, never the FF page
  (I/O, HRAM, IE), and below FE00 only the bus its source sits on: a VRAM source the
  VRAM bus, a CGB's C000-DFFF source the work RAM bus, any other source every bus but
  VRAM's (SameBoy v1.0.5 `is_addr_in_dma_use`). A blocked read gives 0xFF and a blocked
  write drops; SameBoy's read of the transfer's own byte on a conflict is not modelled.
  `layout::Dma` is the one table. The control checkpoint emits the buses as `DB` (the
  transfer's in the low nibble, those blocked after the pass in the high), the host
  carries it to the memory checkpoint's new input `DB`, and each data access's
  translation reads its own bus's blocking. FF46 now reads back during a transfer
  (mooneye `oam_dma/reg_read`).
- **OAM DMA starts two M-cycles after the write.** The M-cycle after the FF46 write still
  reaches OAM and the next is byte 0's (mooneye `oam_dma_start`); a write while a
  transfer blocks keeps the bus blocked through those two M-cycles and moves no byte
  then (SameBoy `dma_restarting`). The control checkpoint counts `DI` 254 and 255 before
  byte 0, keeps the restart in `FL` bit 0 and emits `DZ`, the M-cycles before a fresh
  start blocks, beside `DY`. WC bit 3 no longer adds DMA to write 1's predicted state:
  a transfer started within a step blocks after its last access.
- **TIMA's reload cycle.** A TIMA write in the M-cycle the reload ran is lost, and a TMA
  write then reaches TIMA too (mooneye-gb `tima_write_cycle`, `tma_write_cycle`). The
  control checkpoint keeps that M-cycle in `FL` bit 1 for a pass in place.
- **Operand fetches at their M-cycles.** Operand k of an instruction is read at M-cycle k,
  so an operand that crosses the end or start of OAM DMA reads what the bus shows then
  (mooneye `call_timing`, `call_cc_timing`, `jp_timing`, `jp_cc_timing`,
  `add_sp_e_timing`, `ld_hl_sp_e_timing`). The step's access cycles carry the operand
  count, held to SingleStepTests' bus log for every case; the GBC CPU circuits emit it as
  bit 7 of `RC` (one or more) and of `WC` (two), read off the reference's table. A core
  that takes its bytes from the bus reads each operand at its M-cycle and asks again when
  one changed. The memory checkpoint takes each operand's bus state where it differs from
  the step's start as a third access byte, `FS` (operand 1 low, 2 high, AS0's nibble
  format): the shared access loop (`transport::settle_accesses`) settles the fetch slots
  at M-cycles 1 and 2 with the data accesses, from the control checkpoint's look-ahead
  alone. The next line's translation (`FN`, and the served operands' ports `FP1`, `FP2`)
  is unblocked and unlocked; each operand's column words are held off when its own
  state's buses block the line it reads or its own access locks hold it (VRAM in mode 3,
  FE00-FE9F in modes 2 and 3), and its byte is then the bus plus 0xFF, an affine form
  (`F{k}B`, `F{k}L`).
- **Operand fetches take the access locks at their M-cycles.** A fetch is a read: SameBoy
  v1.0.5's operand fetches are `cycle_read(gb, gb->pc++)`, which goes through
  `GB_read_memory`, `read_vram` (`vram_read_blocked`) and `read_high_memory`
  (`oam_read_blocked`) with no exception for PC. The reference already read operand k
  with `Bus::read` at M-cycle k. The memory checkpoint fetched each operand under the step-start locks,
  because `transport.rs` gave the fetch slots only the DMA bit (`if locks { all } else {
  busy }`); now the model emits the fetches' lock mask (WC bit 3), reads `FS` bits 0 and
  1, and `tests/fetch_locks.rs` runs `LD A,n` from VRAM and OAM across the lock's start,
  in weights with everything in lockstep, against the reference. The host re-reads an overridden operand at its M-cycle for the
  audit and a reference beside the model. The six tests pass with memory in weights too. The
  runner takes the fetch slots' cycles from the wiring (`Steps::fetch_cycles`).
- **The HALT bug's two reads.** Under the HALT bug the GBC CPU reads the opcode's address
  twice, at M-cycles 0 and 1, the second read standing in for operand 1 (Pan Docs, "halt": "pc fails to be normally
  incremented", so "the byte after the halt [is] read a second time"; SameBoy v1.0.5 `Core/sm83_cpu.c`, the run mode's
  `cycle_read(gb, gb->pc++)` and then `gb->pc--` under the bug, each `cycle_read` under
  its own M-cycle's bus state). A lock that rises between them (code running from OAM as a
  line's OAM read lock rises) gives two different bytes. The step's inputs are the fetched
  stream (`Inputs::op`: fetch k at `Cpu::fetch_address`, pc, pc and pc + 1 under the bug),
  so each read keeps its own byte: the reference interprets the stream as fetched, the GBC
  CPU circuit's immediates are OP1 and OP2 with or without the bug, and the memory
  checkpoint fetches operand 1 at pc under its own state (the `OP1` bus, `FS`), which it
  already did. `tests/halt_bug_locks.rs` runs `HALT; LD A,$5A; RET` from OAM on a DMG with
  IME clear and an interrupt pending at a different point of a line each call and logs A:
  over the last 256 calls the reference reads 94 calls with no edge (0x3E), 2 where OAM's
  read lock rose between the two reads (0xFF) and 160 where the opcode's own read was
  locked (0x01, through RST 38H), and the checkpoints in lockstep, in weights alone and as
  one model alone reach the same counts. `tools/gate.sh` holds the runner's passes on the
  same program to the one model's (`examples/halt_bug_rom.rs`). Until this change the step
  held one byte for both reads, and every path refused the step. With memory on the host
  (`--weights cpu`) the host reads the fetched bytes at `Cpu::fetch_address`, so the bug's
  fetch rule sits in the host there, read from the CTL bit the model emitted; with memory in
  weights the model fetches its own bytes and emits each host-served fetch's port.
- **The dispatch's vector.** The vector is chosen after the high byte's push: with SP 0
  that push writes IE, the bits PC's high byte leaves enabled choose the vector, and none
  left jumps to 0000 with nothing acknowledged (mooneye `interrupts/ie_push`, SameBoy).
  IF changes during the dispatch's M-cycles still do not move the vector.
- **HALT with an interrupt pending.** HALT with IME on (EI's delay included) and an
  interrupt pending leaves PC on the HALT, so the dispatch returns into it (SameBoy
  `halt`; SameSuite `interrupt/ei_delay_halt`).
- **Line 153 and the LY=LYC comparison.** LY reads 0 through line 153 on DMG and from its
  second M-cycle on CGB. The comparison sees no line in a line's last M-cycle, and on line
  153 sees 153 in its first M-cycle, none in its second and 0 from its third; line 153's
  last M-cycle reads mode 0 (aappleby's gbmicrotest: all 32 `line_153_*` and 45 of 49 `lyc*`; the 4 left are the HALT wake below).
  The control checkpoint takes the same positions, and stops at line 153's third M-cycle
  where LYC 0 can raise the STAT line.
- **Serial.** An internal-clock transfer ends at the eighth falling edge of the
  counter's bit 8 after its start, 1,024 M-cycles less the counter's bits 2-8 (mooneye
  `serial/boot_sclk_align-dmgABCmgb`).
- **STAT's LY=LYC bit with the LCD off.** The comparison stops while the LCD is off: the
  bit keeps its value, LYC writes leave it, and it alone can hold the STAT line; turning
  the LCD on compares line 0 (mooneye `ppu/stat_lyc_onoff`). The control checkpoint
  keeps it as `ST` bit 2.
- **Boot state.** SC reads 0x7E after the CGB boot ROM in both modes (Nintendo's CGB boot
  ROM run to its hand-off). In compatibility mode the system counter starts at 0x2678:
  mooneye `misc/boot_div-cgbABCDE` reads DIV 0x27 just after it ticks 34 M-cycles after
  the hand-off. The boot ROM run here hands over with DIV 0xF2 for that cartridge, so the
  test, verified on CGB A-E, is the source.
- **HALT's wake cycle.** Built in §3.12: the wake takes no M-cycle, and the control
  logic delays it on a CGB and for a DMG's late interrupt bits.

### 3.12 A3: the HALT wake, LCD-on, mode 0's SCX rounding, HDMA, MBC1M, VBK and line 144

Each rule below is in the reference and in the checkpoint that owns it.

- **The HALT wake (control logic and GBC CPU).** A wake takes no M-cycle of its own:
  the GBC CPU circuits' wake case of `MC` is gone, and the control logic says when a
  halted GBC CPU wakes (`Lines::wake`, `FE_WAKE`). SameBoy's `GB_cpu_run` samples IE &
  IF at the start of a CGB's halted M-cycle and halfway through a DMG's. So on a CGB an
  interrupt that becomes pending wakes the GBC CPU one M-cycle later, and on a DMG in
  the same M-cycle, unless every bit pending rose in that M-cycle's second half: the
  timer's, an internal serial transfer's (by analogy with the timer, not traced) and
  STAT's from the mode-2 source. The control checkpoint keeps the wait in `FL` bit 2
  (`ControlState::waking`), and its horizon is 1 while it waits. Sources: mooneye
  `di_timing-GS`, `halt_ime1_timing2-GS`, `ppu/intr_1_2_timing-GS` on DMG, daid's
  `speed_switch_timing_ly` on CGB, gbmicrotest `int_timer_halt`, `int_oam_halt`,
  `vblank_int_halt_a`, `lyc_int_halt_a`, mooneye `intr_2_*`.
- **The first line after the LCD turns on (control logic).** Line 0 after an LCDC write
  turns the LCD on reads mode 0 until mode 3, raises no mode interrupt then and leaves
  OAM open (SameBoy's `GB_display_run` LCD-on path). The control checkpoint keeps it in
  `ST` bit 1. mooneye `ppu/stat_lyc_onoff` passes on both consoles, and gbmicrotest's
  `lcdon_*` go from 32 to 37 of 40.
- **Mode 0 after SCX's fine scroll (PPU).** The pixels SCX discards end mode 3 at the
  next M-cycle boundary: 1-4 pixels one M-cycle later, 5-7 two (mooneye
  `hblank_ly_scx_timing-GS`, a DMG test). It applies on CGB too, where no source says
  either way; the four CGB gate games keep their hashes. `Ppu::mode3_dots` and the PPU checkpoint's `ML` round the
  fine scroll up to 4 dots.

  On the pixel FIFO (`machine::fifo`, `docs/ppu-in-weights.md` §13) the rule moved to where
  the dots become M-cycles, which is what the GBC CPU observes (STAT's mode 0, the HBlank
  interrupt, HDMA): `fifo::mode0(dots, fine) = ⌊(dots − f + ⌈f/4⌉·4) / 4⌋`, with f = SCX & 7
  at mode 3's start. The FIFO's dots stay unrounded: the line's pixels, its window and the
  judge of a mid-line write (`fifo::judge`) read them. On a line with no write the FIFO's
  dots equal the closed form A3 rounded (`fifo::length_rule`), so the two rules agree there.
  A plain ceiling of the dots would not do: mooneye's sprite table expects an object that
  adds 7 dots to add one M-cycle. Where it applies: the reference's published length, its
  revision on a mid-line write and `Drawn::length`; the length and scanline circuits' `ML`;
  the stepper's `DOTS` (the dots to mode 0, `fifo::mode0_dots`, from the fine scroll its
  start pass latches). After an SCX write during mode 3 the line's start value still applies
  (inferred: the discard happens at the line's start; no census case writes SCX's low bits
  before mode 0 and reads STAT on the same line). Measured on the merge of the access
  timing's A3 into the pixel FIFO: `hblank_ly_scx_timing-GS` passes in the reference and with all
  five in weights under `--lockstep all`; `ppu_length` (the length circuit against
  `fifo::mode0`), `ppu_checkpoint`, `ppu_lines` and every applicable Mealybug PPU row pass.
  A textual merge that kept A3's term inside the length circuit's dots put the circuit one
  M-cycle above the FIFO wherever f is neither 0 nor 4, and failed
  `hblank_ly_scx_timing-GS` in the reference.
- **HDMA (control logic).** An HBlank transfer started while STAT reads mode 0 (the LCD
  off, or a visible line past mode 3) moves its first block at once, and a cancel reads
  back the count it wrote (SameBoy's HDMA5 write; SameSuite `dma/hdma_lcd_off`,
  `hdma_mode0`, `gdma_addr_mask`).
- **MBC1M (memory).** An MBC1 cartridge above 256 KiB whose bank 0x10 repeats bank 0's
  logo is a multicart, fixed at load in `MASK1` bit 1 (SameBoy's guess, plus one condition of ours: a logo
  of one repeated byte is no logo, which keeps a blank test ROM standard MBC1). Its bank
  register's 4 low bits reach the ROM, the high register sits above them, and RAM is
  never banked. The memory circuit's MBC1 cases split by the bit (mooneye
  `emulator-only/mbc1/multicart_rom_8Mb`).
- **Line 144's mode-2 interrupt on CGB (control logic).** On a CGB it rises in line
  143's last M-cycle, one M-cycle before VBlank's (mooneye `misc/ppu/vblank_stat_intr-C`;
  `vblank_stat_intr-GS` has them together on DMG).
- **VBK in compatibility mode (memory and PPU).** CGB hardware reads VBK as
  `0xFE | bank` in either mode (SameBoy; mooneye `misc/boot_hwio-C`).

- **FF72-FF75 (control logic and memory).** FF72 and FF73 read back on CGB hardware in
  either mode, FF74 in CGB mode only, and FF75 keeps bits 4-6 and reads `0x8F | v`
  (Pan Docs, "Undocumented registers"; mooneye `misc/bits/unused_hwio-C`). The control
  checkpoint holds them as state bytes `UR0`-`UR2` and `FL` bits 4-6. The memory
  circuit's port table gains a class for CGB hardware beside CGB mode, and the host's
  bus table (`route`) the same.

What is left of A3's cases, with the reason measured or read:

- **mooneye `lcdon_timing-GS` and `lcdon_write_timing-GS`.** Both need a read lock and a
  write lock that differ by an M-cycle at mode 3's edges. The expected values read OAM
  as locked in a line's last M-cycle and VRAM as locked at M-cycle 19, while an OAM write
  at M-cycle 19 still lands (SameBoy `display.c`: on DMG the VRAM read lock starts at OAM
  scan index 37, the OAM write lock opens there, and the OAM read lock starts 3 dots
  into the line). The locks are one pair per region today (`LK`, the access nibbles), so
  this is a change to the lock transport, not built. Moving line 0's start by 1 or 2
  M-cycles, the other reading of "the PPU is late by 2 T-cycles", drops gbmicrotest's
  `lcdon_*` cases from 37 to 16 of 40 passing. Built since as two pairs of locks (§3.17).
- **Mealybug `dma/hdma_during_halt-C` and `hdma_timing-C`.** §3.14: both pass.
- **mooneye `misc/boot_hwio-C` and `misc/bits/unused_hwio-C`.** FF72-FF75 and VBK are in.
  The APU merge alone did not close them: both run in compatibility mode, where the bus and
  the memory model's port table routed PCM12, PCM34, BCPS and OCPS to nothing. SameBoy
  v1.0.5 `memory.c` gates the first four on `GB_is_cgb` (the hardware) and BCPD and OCPD on
  `cgb_mode`, and `boot_hwio-C` reads C8 at FF68 and D0 at FF6A. Built in §3.15.
- **blargg `oam_bug`.** Not built in A3; §3.14 and §4.4 have it since.
- **gbmicrotest and SCX.** `int_hblank_halt_scx*` expects the halted wake's SCX pattern
  two dots away from mooneye's `hblank_ly_scx_timing-GS`. In whole M-cycles both cannot
  hold; the census follows mooneye, and 4 of the 8 microtest cases pass either way.
  gbmicrotest overall goes from 363 to 386 of 513 passing.

### 3.13 The APU: SameBoy's rules in 2 MHz cycles, reference and APU checkpoint

SameSuite's APU tests read PCM12 and PCM34 one M-cycle apart at double speed, so they
see the channels at 2-dot resolution. The reference counted 4-dot ticks and passed 3 of
the 63 applicable cases.

The reference APU (`machine::apu`) is now a port of SameBoy v1.0.5's `Core/apu.c`
(commit ea667ad, Expat licence, `licenses/SameBoy-Expat.txt`) for CPU CGB E:

- Time counts in APU cycles of 2 dots: two an M-cycle at normal speed, one at double.
  The squares count at 1 MHz through `lf_div`. A CPU access sees every M-cycle before
  it, as SameBoy's `cycle_read` and `cycle_write` do, which the per-access timing of
  §4.1 already gives.
- The frame sequencer takes the DIV bit's fall (`GB_apu_div_event`) and its rise
  (`GB_apu_div_secondary_event`, which arms the envelope clocks). At double speed it
  defers a fall's envelope steps by one M-cycle (`pending_envelope_tick`). A fall from a
  DIV write or STOP is SameBoy's `during_div_write`. Power-on skips the first event when
  the bit is set (`skip_div_event`).
- The control logic now signals the rise (`Signals::rise`) and whether a fall came from a
  reset (`Signals::reset`). Its checkpoint emits the rise's distance as `FR0 FR1`,
  which the host counts down to as it does the fall's.
- Each rule SameSuite tests is SameBoy's:
  - the trigger's start delay, index step and suppressed first sample;
  - the NRx4 frequency change's step back and the NRx3 reload of a countdown that just
    reloaded;
  - NRx2's zombie mode, and the length glitch on enabling length;
  - the sweep's delayed calculation, its reload timer, the restart hold and the negate
    check on NR10 writes;
  - the wave channel's start delay and its reads after its length stops it with the DAC
    on;
  - the noise channel's 14-bit counter, its alignment, the trigger's counter phase and
    NR43's width and shift glitches.
- PCM12 and PCM34 (FF76, FF77, CGB) read each channel's latched output while it plays.
  The bus and the memory model's port table route them to the APU (`layout::io_code`).
  The mix converts the same outputs.

DMG keeps its own rules from before: wave RAM access and corruption, and the length
counters writable while off. Elsewhere it follows the CPU CGB E paths. Every DMG sound
case passes (blargg `dmg_sound` 01-12 and the bundle).

**Verdicts.** The reference passes all 63 applicable SameSuite APU cases, including
`channel_4_freq_change` and `channel_1_sweep_restart_2`. The 7 NA ones name other CGB
revisions. No other case changes verdict (`docs/hardware-tests/after-apu.tsv`: 231
pass, SameSuite 65 of 69). The gate hashes on the host are unchanged.

The APU checkpoint computes the same rules (`docs/apu-in-weights.md` §15).
`hwtests --filter same-suite/apu --weights apu --lockstep apu` passes all 63 with every
pass held to the reference.

**Known differences from SameBoy.** No applicable case exercises any of them:

- SameBoy's paths for CPU CGB C and older, which its DMG also takes. These are the
  two-step NRx2 glitch, the NR10 corruption, NR43's pre-write glitch and the noise
  start's ≤C phase adjustments. They are SameBoy's model of instance-specific behaviour,
  much of it marked non-deterministic in its source. Every DMG sound case passes
  without them.
- SameBoy's DMG-only additions marked "measured on one DMG": the duty latch, the clock
  that survives length expiry, the noise feedback latch, the delayed noise start and the
  pulled-high wave bus.
- Two wave reads take a byte through the GBC CPU's bus in SameBoy: NR30's write reads at
  the program counter's low nibble, and a read pending after a length stop reads at the
  last address on the bus. The reference reads the byte the channel's position selects.
  SameBoy's choice would need the GBC CPU's address bus as an APU input per cycle.
- SameBoy's envelope volume is a `uint8_t`. An envelope step after a direction flip can
  take it past 15. The reference wraps it at 4 bits, as the chip's counter has 4 bits.
### 3.14 A4: HDMA through a HALT and the OAM corruption bug

The HALT rule is in the reference and in the control checkpoint, and the OAM bug in the
reference and in weights.

- **No HBlank block while the GBC CPU is halted (control logic).** SameBoy's
  `GB_display_run` sets `hdma_on` at mode 0 only when the GBC CPU is neither halted nor
  stopped, and its `halt` keeps `allow_hdma_on_wake`, whether STAT read a mode other than 0
  as HALT ran. At the wake (`GB_cpu_run`, both the dispatch and the plain wake) one block
  moves when that flag is set and STAT reads mode 0. Mealybug's `dma/hdma_during_halt-C`
  checks it: HALT outside mode 0 with a 2-block transfer armed, a wake at line 3's mode 2,
  HDMA5 still `01` there. The GBC CPU model's `halted` output reaches the control logic as
  two events (`ControlDevice::halt`, before the HALT's M-cycle, and `woke`, before the wake
  step's). The control checkpoint keeps the HALT in `FL` bits 3 and 7 (`FL_HALT`,
  `FL_HALT_ALLOW`), and its HBlank block waits on bit 3. The host runs the HALT's pass only
  while the model's state says an HBlank transfer runs (HDMA5 bit 7 clear), and the wake's
  only while the state keeps the HALT, so a game without HBlank HDMA pays no pass.
- **Mealybug `dma/hdma_timing-C` (built, reference and control checkpoint).** The test
  reads STAT, HDMA5 and DIV around one HBlank block at SCX 1 and 2, at both speeds. Three
  causes show, each tried alone in the reference first:
  - The stall. SameBoy's `GB_hdma_run` advances 2 + 16 x 2 + 2 T-cycles at normal speed
    and 4 + 16 x 4 at double, 9 and 17 M-cycles for an HBlank block (and for a whole
    general-purpose transfer, not per block), where `block_mcycles` gives 8 and 16. With one
    more M-cycle per HBlank block every DIV read and the mode-2 reads at normal speed match.
  - STAT's mode 0 on CGB. Rounding SCX's fine scroll down instead of up
    (`Ppu::mode3_dots`) on CGB matches the first STAT read of each group; mooneye
    `hblank_ly_scx_timing-GS` holds the round-up on DMG, so it would be a CGB-only rule.
  - HDMA5. `Hdma::hblank` counts the block down at the mode-0 tick and holds only its
    copies and stall to the step's end. SameBoy counts it in `GB_hdma_run`, after the next
    opcode fetch, and its request comes 2 dots after STAT's mode 0, so SCX 2 delays it an
    M-cycle past SCX 1 while STAT reads the same. Matching that needs the count held with
    the copies and a request edge of its own, two dots off STAT's.
  The case needs all three, and they are built: the stall one M-cycle longer per run
  (`dma::RUN_MCYCLES`), HDMA5 counted as the step ends (`Hdma::count`, the control
  checkpoint's `HP` bit and its step-end pass), the request at its own edge, `2p ≥ D + 79`
  at normal speed and `D + 83` at double (`Lcd::request_at`), and STAT's mode 0 at the dot
  (the next bullet; `docs/control-in-weights.md` §25, §26). `hdma_timing-C` passes on the
  reference and with all five components in weights and in lockstep, and the runner's proof
  on it (`tools/gate.sh`) runs its step-end passes, as many as the full program's. Yellow,
  Crystal and Mario DX move HBlank blocks in their 300 frames (`hdma_mcycles=` 578, 288 and
  36,798) and keep their hashes.
- **A CGB's mode 0 at the dot (built, reference and control, PPU and stepper checkpoints).**
  A CGB ends mode 3 at the dot: STAT's mode 0 shows from the half p of the line with
  `2p ≥ D + 77` at normal speed and `D + 82` at double speed, D the dots to mode 0, two
  dots later on line 0 after the LCD turned on; a DMG keeps the fine scroll rounded up to
  4 (`Lcd`, `fifo::mode0_dots`). An LCD turned on at double speed starts at M-cycle 1. At a
  speed switch the edges move at once and the STAT line is recomputed there (inferred, a
  rule with no source or test behind it, chosen so the reference and the model agree).
  The PPU checkpoint, its length model and the stepper publish lengths in dots, and the
  control checkpoint places both the tick's mode (the STAT interrupt, the locks, the
  palette ports, the HBlank block) and a read's mode (STAT, HDMA5, HALT) by halves
  (`docs/control-in-weights.md` §25). Yellow, Crystal, Zelda DX and Mario DX 300 keep
  `cfd31f3d`, `ccfe7ad3`, `f10b2b4a` and `ea7f86ca` on the reference and with every
  component in weights in lockstep as one model; the reference census is unchanged case
  for case, daid's three speed-switch screens stay pixel-exact, and `hdma_mode0`,
  `hdma_lcd_off` and `hdma_during_halt-C` pass in weights.
- **The OAM corruption bug (reference, GBC CPU, memory and control checkpoints).** On a
  DMG, an FE page address on the bus while the PPU's OAM scan reads a row rewrites that row
  and its neighbours (SameBoy v1.0.5 `memory.c`: `GB_trigger_oam_bug` for a write or an
  M-cycle without an access, `GB_trigger_oam_bug_read` for a read, the DMG patterns;
  `ppu::corrupt_oam`). The row is 8 (m + 1) at M-cycle m of a visible line's mode 2
  (`ControlDevice::oam_row`; rows 0 and 0xA0 are never corrupted). 8 x m and 8 x (m + 2)
  were tried: each fails blargg `4-scanline_timing` at one end of the 19-M-cycle window,
  where 8 x (m + 1) passes both. The addresses an M-cycle without an access puts on the bus
  come from SameBoy's `cycle_oam_bug` sites (INC rr and DEC rr, JR, PUSH, CALL and RST
  before their writes, LD SP,HL, and a dispatch's PC + 1 and SP). Every read, write and
  fetch of FE00-FEFF triggers at its own M-cycle, once: SameBoy's `cycle_read` sends each
  opcode and operand fetch through `GB_read_memory`, and `read_high_memory` calls
  `GB_trigger_oam_bug_read` there while OAM is write-blocked on a DMG (`memory.c`
  548-550). A dispatch reads PC too (`GB_cpu_run`: `cycle_read(gb, gb->pc++)` before
  `cycle_oam_bug(gb, GB_REGISTER_PC)`); a wake from HALT fetches nothing. The reference
  used to read all three bytes at PC at a step's start and the operands again at their
  M-cycles, so an operand in the FE page triggered twice; `step::opcode_bytes` now reads
  only the opcode there and peeks the rest. §4.4 has the model's side. blargg `oam_bug`
  singles 1 to 6 and 8 and the bundle pass, on the reference and in weights.

  `7-timing_effect` is NA on DMG: its text overruns its own memory on any console. It
  prints the OAM of each corrupted iteration through this routine (ROM 0x43E7, copied to
  C3E7 with the test's code, which runs from C000):

  ```
  43E7  E5        PUSH HL
  43E8  F5        PUSH AF
  43E9  FA 83 D8  LD A,($D883)
  43EC  6F        LD L,A
  43ED  FA 84 D8  LD A,($D884)
  43F0  67        LD H,A
  43F1  23        INC HL
  43F2  36 00     LD (HL),$00
  43F4  7D        LD A,L
  43F5  EA 83 D8  LD ($D883),A
  43F8  7C        LD A,H
  43F9  EA 84 D8  LD ($D884),A
  43FC  2B        DEC HL
  43FD  F1        POP AF
  43FE  77        LD (HL),A
  43FF  E1        POP HL
  4400  C9        RET
  ```

  The pointer in D883-D884 starts at A004 and is incremented with no bound. The header's
  0147-0149 read `03 00 02`: MBC1 with RAM and battery, 32 KiB of ROM, 8 KiB of cartridge
  RAM, so A000-BFFF. The whole text is 10,525 bytes: with the two stores replaced by
  instructions of the same length and M-cycles (43F2 `36 00` to `18 00`, 43FE `77` to
  `23`), the pointer ends at C921 and the test passes at frame 1324 with its CRC
  (`check_crc $7D792E7C`) right, so its corruption results and timing are the hardware's
  (measured with a probe example, not committed). Past BFFF the stores land in work RAM:
  from frame 773 the routine writes text over the test's code at C000 (measured: C000
  `C3 E6 C2` becomes `20 2D 2D`), and at frame 788 the GBC CPU jumps into the overwritten
  code and the test restarts. The overrun follows from the ROM's bytes, so any console
  that prints the same text, which a right CRC implies, runs over the same code. SameBoy
  v1.0.5 does not finish it either: its tester in DMG mode (`sameboy_tester --dmg`, built
  outside the repository for this check; the census's SameBoy column does not use it)
  logs "A stack overflow has probably occurred. (SP = $d5a4; backtrace size = 512)" and
  "Game probably stuck with blank screen.", and its last screen is blank. The bundle
  `oam_bug.gb` runs test 7 too, keeps its text on the screen, and reports `07:ok` in the
  expected DMG screenshot the test collection ships (`oam_bug-dmg.png`), which the census
  holds the screen equal to on the reference and in weights. Yellow on DMG triggers
  the bug 0 times in 3,000 frames (`oam_bugs=` in the run line).

  To reproduce (the probe that measured it was not committed; these steps restate what it
  did, with the gbcformer binary instead of the probe's example):

  1. Copy the single to a scratch directory and patch the two stores in the copy, at
     file offsets 0x43F2 and 0x43FE (the ROM is 32 KiB with no banking, so a file offset
     is the address):
     `printf '\x18\x00' | dd of=7-timing_effect.gb bs=1 seek=$((0x43F2)) conv=notrunc` and
     `printf '\x23' | dd of=7-timing_effect.gb bs=1 seek=$((0x43FE)) conv=notrunc`.
  2. Run the patched copy on the reference:
     `gbcformer run 7-timing_effect.gb --hardware dmg --frames 1400 --no-save --dump out.ppm`.
     Its screen reports the test passed (frame 1324 in the probe's run).
  3. Run the unpatched single for 800 frames and read C000-C002 after each one (the probe
     read them with `Session::ram_byte`): from frame 773 they
     read `20 2D 2D`, text bytes, where the test's code had `C3 E6 C2`.
  4. SameBoy: build `sameboy_tester` from SameBoy v1.0.5 (`make tester`) outside this
     repository and run `sameboy_tester --dmg 7-timing_effect.gb`. It logs the stack
     overflow and blank-screen lines quoted above.

### 3.15 Last 12: CGB hardware registers in compatibility mode (memory checkpoint)

BCPS, OCPS, PCM12 and PCM34 answer on CGB hardware in either mode; BCPD and OCPD stay CGB
mode's (SameBoy v1.0.5 `memory.c`: `GB_IO_BGPI`, `GB_IO_OBPI`, `GB_IO_PCM12` and
`GB_IO_PCM34` return 0xFF only when `!GB_is_cgb(gb)`, `GB_IO_BGPD` and `GB_IO_OBPD` when
`!gb->cgb_mode`). The host's bus table (`route`) and the memory circuit's port table
(`layout::io_code`) both move the four to the CGB-hardware class. The post-boot palette
indices are the ones Nintendo's CGB boot ROM leaves, measured by running it (`--boot
cgb_boot.bin`): C8 and D0 in compatibility mode, as mooneye `misc/boot_hwio-C` expects, and
C0 and 40 in CGB mode. The machine test
`hardware_follows_the_header_and_cgb_runs_dmg_only_games_in_compatibility_mode` reads them
and fails without the change. `boot_hwio-C` and `unused_hwio-C` pass in the reference and
with `--weights all --lockstep all`.

### 3.16 Last 12: STAT edges inside a span (control checkpoint)

The control checkpoint passes at every M-cycle where the STAT line can rise (§3.12's
horizons) and takes a rise as the line high after the pass and low before it. Two
cases broke that, each a divergence from `Control` and none a reference change:

- **A drop and a rise inside one span.** In a line's last M-cycle the LY=LYC compare
  stops and mode 2's source replaces mode 0's, so a line held by mode 0 (mode 2's source
  off), or by the LYC match, drops there and rises again at the next line's start when a
  source is on there: LYC on the next line, line 144's mode 1, or on a DMG its mode 2.
  LYC 153 holds the line only in line 153's first M-cycle, and line 0's mode 2 raises it
  again. With IF bit 1 cleared before the drop, one span covered both, and the pass at
  the rise saw the line high on both sides. The model now also passes at the drop
  (M-cycle 113, or 153.1). Found by the seed-7 control stream
  (`the_lockstep_catches_a_wrong_edge_or_wake`, pass 148,546, IF 25 against 27), whose
  traffic changed when the OAM-row read joined it; the directed test
  (`a_stat_line_dropped_in_the_last_m_cycle_rises_at_the_next_line`) fails on the build
  before this work too, so the hole predates it.
- **A drop inside a span, then a rise at a register write.** A STAT write that turns a
  source on raises the line after the ticks dropped it; the pass compared the line after
  the write with the line at the last pass, both high. IF bit 1 now rises at the ticks
  when the line is high after them and was low at the last pass, and at the event when
  it is high after the event and was low after the ticks. Found by a STAT-only random
  stream (`random_stat_traffic_matches_the_reference`: enables, LYC on this line, the next,
  0, 143, 144 or 153, IF bit 1 cleared at random M-cycles).

With the extra passes at line 143 and 153.1 disabled, those five cases of the directed
test fail; a sweep of 12 seeds of 100,000 STAT events each passes (run once).

### 3.17 Last 12: a DMG's read and write locks (reference, control checkpoint, transport)

mooneye `ppu/lcdon_timing-GS` reads LY, STAT, OAM and VRAM, and `lcdon_write_timing-GS`
writes OAM and VRAM, at fixed M-cycles after the LCD turns on, over lines 0 to 2. Their
tables, in a line's M-cycles m (mode 2 from m = 0, mode 3 from 20, mode 0 from 63 at the
least), have reads and writes locked differently at two M-cycles on lines past the first:

| | m = 113 of the line before | m = 0-18 | m = 19 | m = 20-62 | m = 63-112 |
|---|---|---|---|---|---|
| OAM read | locked | locked | locked | locked | open |
| OAM write | open | locked | open | locked | open |
| VRAM read | open | open | locked | locked | open |
| VRAM write | open | open | open | locked | open |

SameBoy v1.0.5 `display.c` has the same edges on a DMG: `oam_read_blocked` rises 3 dots
into a line, a dot before STAT shows mode 2 and `oam_write_blocked` rises; at OAM scan
index 37 `vram_read_blocked` rises and `oam_write_blocked` falls (`!GB_is_cgb`); mode 3
sets all four and mode 0 clears them. The line after the LCD turns on has no mode 2 and
neither M-cycle-19 edge (the tables: a VRAM read at m = 19 of that line is open). Two
rules beyond the tables:

- **Built, but inferred: a DMG's OAM read lock in line 153's last M-cycle.** The lock
  rises there before line 0 of the next frame, because SameBoy runs the same loop for
  line 0. The reference and the control checkpoint both have it. No ROM in the census
  reads OAM at that M-cycle, so no test checks it.
- **Not built: SameBoy's early CGB OAM lock.** On a CGB, SameBoy locks OAM reads and
  writes from the line's start and has no early VRAM read lock. No census case checks
  this, so neither the reference nor any checkpoint has it: a CGB keeps the modes' locks
  for both.

- **Reference.** `Lcd::locks` gives reads' VRAM and OAM in bits 0-1 and writes' in bits
  2-3 (`step::WRITE_SHIFT`), and `Lines::locks` carries all four. `Machine::ppu_locked`
  takes the pair of its access's kind.
- **Control checkpoint.** `LK` carries the four locks after the event in bits 0-3 and the
  four after those in bits 4-7, with `MX` and `MY` as before; on a DMG mode 2 now holds to
  M-cycle 19, mode 0 of lines 0-142 to 113, and line 153 to 113.
- **Transport.** Each access takes the pair of its kind: the wiring names the write slots
  (`Steps::write_slots`, `write_shift`), and `transport::StateBits::slot_locks`,
  `bus_state` and `Accesses::ahead` pick by slot. The step-start state (CTL bits 5-6, `LK`
  late) stays the reads' pair, so a write slot whose pair differs at its M-cycle gets its
  own access nibble. The GBC CPU and memory circuits are unchanged.

Both mooneye cases pass on the reference and with all five in weights and in lockstep.
aappleby's gbmicrotest (the result byte at FF82 after 30 frames on DMG, read by a probe
not committed) goes from 387 to 391 of 513: `oam_read_l1_a`, `oam_read_l1_f`,
`oam_write_l1_c` and `vram_read_l1_b` pass, and none fails that passed. `Lcd::locks` has a
unit test over a line's M-cycles on both consoles
(`reads_and_writes_lock_by_the_mode_and_a_dmg_s_edges`).

## 4. Designs left

### 4.1 Access timing inside an instruction

**Status: built, the default in every run.** `step::step` runs the devices to each
access's M-cycle before it lands. A core that takes its bytes from the bus (the
reference) gets operand k at M-cycle k; a core with its own memory
(`StepFacts::timed_accesses`, the memory checkpoint) gets each access's bus state at its
M-cycle (`transport::settle_accesses`). `Core::access_cycles`, `Bus::timed` and
`GBCFORMER_ACCESS_TIMING` below are the prototype's names, which the code no longer has;
the measurements below are the prototype's.

The largest cause in §2.1. A step's reads happen before its M-cycles run and its
writes after. The design: the GBC CPU circuit emits, per requested read and per write,
the M-cycle of the instruction it falls in (a 3-bit field per slot, a linear sum of the
opcode one-hots and the taken-branch wire, so one level and no new heads); the host runs
the devices to that M-cycle before serving the read, and on to the next before
committing each write, then the rest of the step. The memory checkpoint's own reads of
VRAM and OAM take the access locks at that M-cycle, which `Lines::locks` already carries
as spans from the last pass (§3.6). SingleStepTests list every case's bus activity per
M-cycle, so the new outputs are held to them for all 500,000 cases.

**The reference half, built.** `Core::access_cycles` gives the M-cycle each access
falls in (`AccessCycles`); the reference interpreter records it as it runs, and
`sst::check` holds it to the bus log: `gbcformer sst` passes all 500,000 cases with it.
`step::step` runs the devices up to each access first when the bus is timed
(`Bus::timed`, `Bus::tick`), and the machine is timed with `GBCFORMER_ACCESS_TIMING`
set and the GBC CPU on the host; no checkpoint emits the cycles yet, so a core in
weights runs untimed.

**Verdicts** (the runner on the reference, 342 cases, `docs/hardware-tests/after-timing.tsv`):
153 pass untimed and 159 timed. Timing flips 14 to pass: blargg `mem_timing` and
`mem_timing-2` (both bundles on DMG and CGB, and the six singles), mooneye `pop_timing`
(DMG and CGB) and `boot_div-dmgABCmgb`, and SameSuite `channel_3_wave_ram_locked_write`.
It flips 8 to fail: mooneye `intr_2_mode0_timing`, `intr_2_mode3_timing` and
`intr_2_oam_ok_timing` (DMG and CGB) and daid's `speed_switch_timing_ly` and
`_stat`. Those passed with the reads at the step's start, so the rules they hold (LY's
early increment, §3.8, and the mode boundaries) were likely tuned an access-time early
and must move back once reads land at their M-cycles; that is inferred, not traced. The
retuning, then the circuit's outputs, are what is left. The call, jump, return and RST
timing tests still fail timed; their cause is not traced yet.

**The memory checkpoint's own accesses at their M-cycle.** The memory circuit
translates each data access under the bus state at that access's M-cycle, not the
step's start. Two early input slots carry the state, `AS0` for reads 0 and 1 and `AS1`
for writes 0 and 1. Each access gets a nibble: bit 0 is VRAM locked, bit 1 OAM locked,
bit 2 OAM DMA blocking, and bit 3 means the nibble replaces the step-start state (CTL
bits 4-6 or the late `LK`). Each access's lock and DMA wires are one level-1 node per
bit, `[base − 2·ov ≥ 1] + [n + ov ≥ 2]`. They feed that access's translations only:

- read 0: HL, SP, the rare bus and the FF page;
- read 1: SP+1;
- write 0: HL, SP−1, FF00+n and the rare bus;
- write 1: SP−2 and a16+1.

The opcode fetch keeps the step-start state, and each operand fetch takes its own (`FS`,
§3.11). Write 1's record keeps write 0's byte only when
both writes keep a record (`fields::same_line`), since a lock or OAM DMA can drop one
and not the other. The circuit also emits two request bits:

- RC bit 3: the step reaches VRAM or the FE page by a data access's bus address.
- WC bit 3: PC's line or the next reaches VRAM or the FE page, so the locks at each
  operand fetch's M-cycle matter (§3.11). It replaced the bit that said write 0 starts
  OAM DMA, which no host read once a transfer started within a step blocked only after
  its last access.

The host's part is `step::step` and `Timeline`. It runs only for cores that translate
their own accesses (`StepFacts::timed_accesses`). For each access with an M-cycle above 0, it
asks `Bus::ahead` for the state at that M-cycle under the access's mask
(`transport::Asked::masks`): OAM DMA always, and the locks where the model's RC bit 3
(the data accesses) or WC bit 3 (the operand fetches) is set. `transport.rs` only ANDs
each mask with the state it transports; no access kind is exempt there. An access whose state differs from the step's start gets an override nibble.
The host then runs the request protocol again under the nibbles, fetches included,
until they settle (at most twice). When the start's OAM DMA ends before an overridden
access, the devices first run to that access, so the transfer's last bytes are records
before the pass reads OAM. Every access before that one is blocked or in HRAM. At each
access's M-cycle, the host checks the bus's actual state against the bits it predicted
and stops the run at a difference, naming the access.

Each write record lands at its write's M-cycle. The memory core decodes the step's
outcome with both records, then keeps neither (`MemCheckpointCore::execute_on`). When
`step::step` reaches write i's M-cycle, it calls `Core::land_write(i)`, and the core keeps
that record then. So a line drawn or a DMA byte copied in the ticks before the write
sees the old byte. A pass that decodes a second device's port overwrites the step's
tokens, so the core keeps a copy of them for the records first. `Core::execute`, which
runs no bus (SingleStepTests, the dense-pass tests), keeps both records at once. The
next request stops the run if a record was never kept.
`tests/checkpoint_sst.rs` `a_write_record_lands_at_its_write_not_at_the_outcome` runs a
PUSH BC whose two writes fall in different lines and reads the model's image after the
outcome and after each write. Before this change the outcome kept both records at
once, and the test's first check fails.

One case still lands a byte late. When both writes keep a record in the same 8-byte
line, write 1's record holds write 0's byte and replaces it (`fields::same_line`), so
write 0's byte lands at write 1's M-cycle, one M-cycle late. No reader can see that.
A line drawn between the two writes needs mode 3 to start there, which locks write 1's
VRAM or OAM access, and then write 1 keeps no record and write 0 keeps its own. The
timeline holds each access's predicted lock to the bus's lock at that M-cycle. An OAM
DMA byte copied between them means the transfer blocks write 1, unless write 1 is in
HRAM. No transfer reads HRAM: a DMG source from E000 up reads C000-DFFF, and a CGB
writes 0xFF (`machine::dma`). The memory core holds both cases: a DMA copy whose source
or destination line holds a record not kept yet stops the run (`AnyCore::copied`).

With the control logic in weights, `Bus::ahead` uses only what the model emitted:

- OAM DMA blocks while the pass's new `DY` output (160 − DI′, 0 when idle) exceeds the
  M-cycles since the pass plus the access's.
- The locks come from the span (`LK`, `MX`, `MY`) that holds at that M-cycle.
- Past the spans, one more control pass over the span to the access's M-cycle gives the
  locks, and its state is dropped. The spans lapse mostly in mode 3 at its 63rd M-cycle,
  where a pass does not yet know the line's length. A pass in place first was tried:
  307 of 312 on Yellow 300 still lapsed.

The reference `Control` looks ahead on copies of its `Lcd` and `OamDma`. The control
lockstep holds the model's prediction to the reference's.

The GBC CPU's lockstep changes nothing the devices do. It used to serve every read the
reference requested from the live bus. A read that OAM DMA blocks then cut the control
model's OAM run at that read's M-cycle, where a plain run cuts once after the step. So
a POP from WRAM under a transfer ran one control pass more in lockstep. Now the host
serves only the reads the model asks it for. The reference gets every other byte from
the side-effect-free `Bus::peek`, under that read's own bus state
(`Lockstep::execute_on`). `tests/oam_dma_runs.rs`
`lockstep_leaves_the_control_passes_as_a_plain_run_has_them` compares the control
passes of a run with the GBC CPU and memory in lockstep against a plain run. Its first
version passed on the old code: POP's first read fell on the M-cycle where an OAM line's
run ends, so the control model had just run a pass there, and the extra cut found 0
M-cycles elapsed (traced). One NOP before the POP moves the read off that M-cycle. On
the old code, 12 frames then run 30,452 control passes in lockstep against 30,432
plain, and on the new code 30,432 both. The memory audit holds each
access's physical address and byte to `Machine::fixed_physical` and the peek under
that access's state, and the fetches under the step's start. Inside a step, the ticks
hand each block's DMA copies to the core before the next block runs (`Bus::tick`), and
the frame's memory audit follows. `tests/oam_dma_runs.rs` polls OAM through a transfer's
end and the PPU's OAM lock with all four in weights and in lockstep. At least one of
its steps is asked again under its accesses' own state.

Measured with all five in weights and in lockstep:

| Game, 300 frames | Frame hash | Steps asked again | Lookahead passes |
|---|---|---|---|
| Yellow | `cfd31f3d` | 0 | 314 |
| Yellow, one model | `cfd31f3d` | 0 | 314 |
| Crystal | `ccfe7ad3` | 204 | 596 |
| Zelda DX | `f10b2b4a` | 66 | 54 |
| Mario DX | `ea7f86ca` | 0 | 13 |

`tools/bench/text.sh` with all five in weights asks 0 steps again and runs 763
lookahead passes in 3,208,966 steps, 1.27 per frame. The control passes stay at 77.3
per frame. The memory core's passes are unchanged: 14,800 result passes and
3,248,876 in all.

### 4.2 The boot ROM in the memory checkpoint

**Status: built.** The memory checkpoint maps the boot ROM while it is mapped
(`tests/boot_overlay.rs` runs one under memory lockstep); the gate's power-on runs and
the census with `--boot-dmg`/`--boot-cgb` boot with memory in weights.

The boot ROM becomes a region of the model's KV cells, written at load like the
cartridge. A banking register `BOOT` (1 bit) joins `memory::layout::Reg`; while it is
set, the read translation sends 0000-00FF (and 0200-08FF on CGB) to the boot region,
and the FF50 write is a record that clears it. The host places the boot bytes only at
the bits the model emits, as it places every record. This moves the cpu-memory and
one-model digests; the boot ROM's bytes never enter the weights, only the cache.

### 4.3 The boot ROM with the model running it, measured

With Nintendo's CGB boot ROM, `run pokemon-yellow.gbc --frames 400 --boot cgb_boot.bin
--weights cpu,apu,control --lockstep all` runs the boot ROM's 2,044,767 steps in the GBC
CPU checkpoint, its start-up sound in the APU checkpoint and its timing in the control
checkpoint, all in lockstep, and reaches frame hash `0db60d9b`, the reference's. A DMG
cartridge goes through the same boot ROM into compatibility mode: dmg-acid2 on CGB with
`--warmup 210` and all five components in weights after the hand-off reaches `9c7eee04`,
the hash of the hard-coded compatibility start, so the boot ROM's palettes and the
default ones draw the same image. `tools/gate.sh` runs both when `cgb_boot.bin` is
present and skips them otherwise. With the boot overlay (§4.2) both run with memory in
weights too: Yellow from power-on with all five in weights reaches `0db60d9b`, in
lockstep and as one model.

**Held to the reference at the hand-off.** `examples/boot_proof` runs one boot ROM and
cartridge twice from power-on to the FF50 write: on the reference alone, and as the one model
with every component in lockstep. At the hand-off it holds the model's memory image to the
host's arrays, then the whole hand-off state to the reference run's: the GBC CPU's registers
and flags, the emulated time, FF00-FF7F and IE as the GBC CPU reads them, the 136 display
bytes and the window state, and cartridge RAM, WRAM, VRAM, OAM and HRAM byte for byte.
`tools/gate.sh` runs it for each boot ROM it finds. Measured on 2026-11-14, every case
equal:

| Boot ROM | Cartridge, mode | Hand-off (halves) | Memory bytes compared |
| - | - | - | - |
| Nintendo's CGB (supplied) | Yellow, CGB | 7,206,102 | 82,207 |
| Nintendo's CGB (supplied) | dmg-acid2, compatibility | 7,240,054 | 49,439 |
| SameBoy's CGB | Yellow, CGB | 6,542,862 | 82,207 |
| SameBoy's CGB | dmg-acid2, compatibility | 6,578,184 | 49,439 |
| SameBoy's DMG | Yellow, DMG | 2,940,554 | 82,207 |

Nintendo's DMG boot ROM is not here; SameBoy's stands in for the DMG case, and the gate prints
the skip.

### 4.4 The OAM corruption bug in weights

On a DMG, the GBC CPU corrupts OAM when it puts an address in FE00-FEFF on the bus while
the PPU scans OAM in mode 2 (§3.14 has the rule and its sources). Each piece below is a
model output; the host runs the passes they ask for and decides nothing about the bug.

- **Events (GBC CPU, memory checkpoint).** The memory circuits emit a bit for each access
  that puts an FE page address on the bus that OAM DMA does not block: reads 0 and 1
  (`ACK` bits 5 and 6), writes 0 and 1 (`WQ` bits 6 and 7), the M-cycles without an
  access (`MC` bits 3 and 4, that M-cycle in bits 5 and 6, `decode::idle_bits`), and the
  fetches (`CTL'` bits 5 to 7, `decode::fetch_fe_bits`): the opcode on every step but a
  wake, and operand k at PC + k − hb when the step fetches it, each under its own OAM DMA
  state. The JR target's page test reads PC, the offset and the HALT bug rather than the
  PC adder's late bits, which keeps the GBC CPU checkpoint at depth 6 and the memory
  checkpoint at 10 (f16census). SingleStepTests holds the bits to the reference
  interpreter's on all 500,000 cases in the 0/1 and nibble circuits, and the memory core's
  audit holds them to the bus on every lockstep step.
- **Row (control checkpoint).** A read of FF0A, which no bus access reaches, answers the
  row the OAM scan reads at that M-cycle: 8 (m + 1) at M-cycle m of a visible line's mode
  2 on a DMG, past the first line after the LCD turned on, else 0 (`OAM_ROW_PORT`, the
  `OAM scan` node; `ControlDevice::oam_row`). The host asks at every event. An earlier
  version skipped the read on CGB hardware and while the latched locks showed OAM
  unlocked: both were rules about the bug in host code, and both are gone, since the
  control logic answers 0 there itself.
- **Rewrite (memory checkpoint).** When the row is not 0, the OAM pass reads the event
  byte `OB` (the row, bit 0 set for a read) and the rows it can touch, and emits the new
  line and up to five rows' addresses (`OT`, `OW0`-`OW4`, `OL0`-`OL7`,
  `oam_bug::memory_pass`), each kept as a line record (`tok::OAM_RECORDS` on). The PPU
  checkpoint follows them as it follows any OAM record. The pass leaves the step's inputs
  and decoded tokens as they were, so the step's later passes and records read the same
  tokens with or without it.
- **Order.** The full program (`step::step`) and the runner run the events in M-cycle
  order: the fetches' first (the opcode's at the step's start, operand k's at M-cycle k),
  each read's before it is served, the idle M-cycles' after the result pass, each write's
  before its record lands. The row query and the rewrite are one shared function,
  `transport::bus_event`.
- **Runner.** The wiring table names the event flags, the event byte, the pass's slots
  and its records with their addresses (`wiring::Steps::events`), and the row query's
  event code (`wiring::Spans::row_query`). Its ratchet rows for the events are class (a):
  flags and cycles a pass emitted, bits concatenated, a pass the emitted row asks for.
  The runner refuses a rewrite of a row that shares a line with a write record of the
  same step not landed yet, as it refuses such a copy.

The cost lands on every step of the memory checkpoint, DMG or CGB, since the checkpoint
is shared, while the bug fires only on DMG: the memory circuit is 1,713 nodes and 7,361
steps at depth 10 with the events and the OAM pass (f16census), of which the fetch
events are 7 nodes and 10 steps. On CGB every event costs a control
pass that answers 0.

## 5. Open, as of this branch

Every limit known at this branch, with its evidence. The census passing does not show that
this list is complete: a behaviour no census ROM exercises and no outside source pins can
be wrong in the reference and the model alike, since lockstep checks only their agreement.
The HALT bug's two reads across a lock edge, refused until this branch, are built (§3, "The
HALT bug's two reads").

### 5.1 Hardware behaviour

- **`boot_div-cgbABCDE` from the boot ROM (measured; the cause is bounded, not traced).** The
  census starts from the post-boot state, where mooneye `misc/boot_div-cgbABCDE` passes. Run
  with `--boot-cgb cgb_boot.bin` (Nintendo's CGB boot ROM, its revision not checked), it fails
  on the reference and with all five in weights in lockstep alike. `examples/boot_state` on
  that ROM measures the reason the test sees: the boot ROM runs 3,620,026 M-cycles to its
  hand-off here and leaves DIV at 0xF2, where the test (verified on CGB A to E) needs the
  0x26 the hard-coded start holds. DIV ticks every 64 M-cycles and wraps every 16,384, so
  the boot runs 3,328 M-cycles short of hardware's, modulo 16,384 (or 13,056 long). The
  boot ROM's instructions and its waits are the reference's and the model's alike (the boot
  proofs of §4.3 hold the model to the reference at the hand-off), so the difference sits in
  the power-on state the boot ROM starts from or in the timing of what it waits on (the LCD's
  first frames); which one is not traced. `boot_hwio-C` and `boot_regs-cgb` pass that way.
- **SameBoy's early CGB OAM lock (not built, §3.17).** On a CGB, SameBoy v1.0.5 locks
  OAM reads and writes from the line's start rather than from mode 2, and has no early VRAM
  read lock. Neither the reference nor any checkpoint has it: a CGB keeps the modes' locks
  for both, as a DMG does. Its scope is CGB hardware, in CGB mode and with a DMG cartridge
  in compatibility mode; a DMG is not affected. No census case reads or writes OAM in the
  M-cycles where the two differ, and which CGB revisions SameBoy's rule describes is not
  established. Measured: the census passes without it; not measured: hardware.
- **A fetched byte the host serves is read at the step's opening (an approximation, not a
  measured failure).** Code fetched from an address the host serves (an I/O register, IE,
  the unusable area, the MBC3 clock's registers) has each such byte read when the step opens
  (`step::request`, `StepFacts::fetch_ports`), not at its own M-cycle, so under the HALT bug the
  two reads of such an address are one read, and an operand fetched there misses a change
  within the step. The bytes in the model's own cells (ROM, every RAM, OAM, VRAM) are each
  read at their own M-cycle under their own bus state. No census ROM or game runs code from
  the I/O page.
- **Where an HDMA transfer runs off its destination (an open question, not a measured
  failure).** The reference and the control checkpoint keep the destination as an offset into
  VRAM (HDMA3's low five bits, Pan Docs: "only bits 12-4 are respected") and end a transfer
  after the block at offset 0x1FF0. SameBoy v1.0.5 keeps HDMA3's whole byte
  (`Core/memory.c`, `hdma_current_dest |= value << 8`), writes each byte at the
  destination's low 13 bits and ends a transfer when the 16-bit destination wraps to 0
  (`GB_hdma_run`: `gb->hdma_current_dest == 0`), so from 0x9FF0 it goes on writing at
  VRAM's start until 0xFFF0. No census ROM runs a transfer off VRAM's end. Which rule
  hardware follows is not established; pinned by a ROM that starts an HBlank transfer at
  0x9FE0 with more blocks and reads VRAM 0x8000 and HDMA5 after.
- **Inferred rules no outside source pins.** Lockstep holds the model to the reference, so
  a rule written into both at once is checked only for agreement. Each of these is labelled
  inferred where it is built, and each would be pinned by the trace named.
  - *A speed switch raising the STAT line* (`Control::stop`, `Lcd::set_double`; control
    circuit `K1'.7`, `docs/control-in-weights.md` §25). The STAT line after a STOP switch
    takes the new speed's mode-0 edge. Pinned by a hardware or SameBoy trace of a switch
    about 80 M-cycles into a line whose mode-0 edge falls between the two speeds' edges,
    with STAT's interrupt enables set.
  - *The HBlank request's edge* (`Lcd::request_at`: `2p ≥ D + 79`, `D + 83` at double
    speed; the control circuit's `past_rq` and `at_rq`). Fitted to Mealybug
    `dma/hdma_timing-C`, whose rows are SCX 1 and 2; SameBoy v1.0.5's `hdma_on` comes a dot
    later. SCX 0 and 3-7, lines lengthened by objects or the window, and a CGB's line 0
    after the LCD turned on are extrapolated. Pinned by `hdma_timing-C`-style DIV and HDMA5
    reads at those SCX values and with objects on the line, or SameBoy at those values with
    its one-dot offset taken out.
  - *One HBlank block a line across the gap between STAT's mode 0 and the request*
    (`Lcd::in_gap`, `Hdma::line_done`; the control circuit's `DN`, HDMA3 bit 6). At SCX fine
    2, 3, 6 and 7 at normal speed (one M-cycle) and at even fine values at double speed (one
    half), STAT reads mode 0 before the request. An HBlank transfer started there moves its
    first block at once and the line's request then moves none; a HALT entered there keeps
    the line's block owed, and the wake moves it. Before, the request moved a second block
    on the same line, and a HALT in the gap lost the line's block. SameBoy's request and
    mode 0 fall together, so it gives no answer here: one block a line is the rule chosen,
    inferred. `machine::tests::a_line_moves_one_hblank_block_when_the_transfer_starts_in_mode_0`
    and `a_halt_between_mode_0_and_the_hblank_request_keeps_the_line_s_block` hold it on the
    reference, and the random DMA stream holds the circuit to it. Pinned by a ROM that polls
    STAT for mode 0 at SCX 2, writes FF55 = 0x80 | n and reads HDMA5 on the next line, and
    one that HALTs in that M-cycle, on hardware.
  - *A DMG's OAM read lock in line 153's last M-cycle* (§3.17). The lock rises there,
    before line 0 of the next frame, because SameBoy runs the same loop for line 0. The
    reference and the control checkpoint both have it; no census ROM reads OAM in that
    M-cycle. Pinned by a DMG ROM that reads OAM at line 153's last M-cycle.

  An HBlank block held to a GBC CPU step's end counts in HDMA5 as the step ends, and
  whether it ended the transfer is the block's own fact, kept from the block (`Hdma::count`,
  the control circuit's `HE`, HDMA4 bit 0), as SameBoy decides it in the same run. Until
  this change the count read the destination at the step's end, which a write to HDMA3 or
  HDMA4 later in the step could change (review 3, finding 3;
  `machine::dma::tests::a_held_block_off_the_end_of_vram_ends_the_transfer_after_a_destination_write`
  and its converse).

### 5.2 Premises the guards hold

Each guard stops a run where a premise of the step contract or the transport fails. No
census case and no game run has tripped one; that every legal execution avoids them is not
proved.

- *One held HBlank request a GBC CPU step* (`Hdma::hblank`; the control circuit's `HP'`
  counts one): a step's in-step M-cycles are at most 6, and two requests are a line apart.
- *Coincident signals in one block* (`transport::Signals::apply`): a line drawn shares its
  block with no own byte, stop, blank, frame or block copy, and a frame not with a blank.
- *Staged writes and OAM rewrites* (`sm83::abi` `MemCheckpointCore`, the runner's
  `steps.rs`): an OAM corruption rewrite shares no line with a write record of the step not
  landed yet, and a DMA copy's byte no line with one (`transport::staged_in_line`).
- *The stepper's mode-3 length within a byte* (the runner's `stepper.rs`).

### 5.3 Inference management

- *The step-end and rest passes the runner skips* (`docs/minimal-runner.md` §2, the row on
  rests, wakes and step ends). A rest is a pass only while the state's `idle` flag is clear,
  a wake only while `kept` is set, a step's end only while `uncounted` is set. Measured:
  with every one run as a pass, every frame and sample and the lines, segments and length
  records stayed equal on the runs that document lists, while the spans and steps records
  (and Zelda DX's stepper record) changed. That the skipped passes change no hidden state a
  later output can show is inferred from those runs, not traced.

### 5.4 Verification scope

- *The census.* Both columns are post-boot. Its weights column is the components as
  separate checkpoints in lockstep; as the one model only a declared sample runs (§2.4, each
  suite's ten slowest cases), since the one model's census does not fit the machine at three
  shards. The boot proofs (§4.3) hold the boot ROM's run to the reference at the hand-off,
  not to hardware.
- *The whole checkpoint against each stream* (`examples/slice_full.rs`, `docs/one-model.md`
  §4): every stream memory of at most 20,000 cells whole (the PPU family's, and the GBC CPU's
  on `m3_bgp_change`), Yellow's GBC CPU cache over the cells each event read.
- *llama.cpp.* On the closure weights (one model `47ceed72`) two bounded runs held every
  pass to the engine's record: 100,000 passes of `m3_bgp_change` (85,288 stepper passes)
  and 20,000 passes of Yellow to frame 1; each verify runs 4,000 passes split and plain. The
  full 300-frame Yellow run has since finished in unpatched llama.cpp on Metal, on `45bfb056` and then on the current weights, `ff86dde7`, every
  pass equal to the engine's record (`docs/perf-log.md`, "The first proof" and "The proof of record on `ff86dde7`"). A DMG
  trace through the OAM corruption bug is owed.
- *The ratchet* (`tools/runner/tests/quarantine.rs`): it lists every line of the runner and
  the transport that branches, asserts, panics or joins conditions with `&&` or `||`, by
  its text. It does not see a decision written without those words (an index computed from
  a flag, a table lookup), and a listed line's class is a review's judgement.

### 5.5 Representation

- The canonical artifact, the one model's per-layer-shaped GGUF, runs in llama.cpp with
  three patches, a speed-up only (`docs/stock-llama.md`); the padded uniform export runs in
  unpatched llama.cpp b12070 on Metal, built with llama.cpp's own default options, and is the claim of record (`docs/stock-llama.md`, "What runs where").
- The cartridge ROM, the boot ROM and every RAM are KV cache cells, not parameters; the
  board's starting state and the wiring between the streams are metadata in the GGUF,
  checked against the build that reads them (`docs/one-model.md`).

