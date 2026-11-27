#!/bin/zsh
# Runs every gate from CLAUDE.md against one binary and prints each hash beside its
# expected value: first on the host, then with each component in weights and in
# lockstep. Fast failures come first. BIN overrides the binary under test; the tests
# always run in this checkout. The dmg_sound hashes are the reference's frames (the
# tests report on screen).
#
#   tools/gate.sh            every gate
#   tools/gate.sh --quick    build, wasm, DMG 8 and CGB 300 only
#
# A caller that already ran a step hands over its result instead:
#   GATE_NO_TEST=1   skips `cargo test --release --features testing`, which the caller ran
#   GGUF=<file>      the one model's GGUF the caller exported, in place of the export
#   STOCK_BIN=<gbcformer built with --features stock> and RUNNER_BIN=<gbcformer-runner built
#                    with --features record>, in place of building both into target-stock
#   RUNNER_CHECKED_BIN=<gbcformer-runner built plain with CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true>,
#                    in place of building it into target-stock/checked
# Every ok or FAIL line ends with the seconds since the previous one.
set -u
cd "${0:A:h}/.."
BIN=${BIN:-./target/release/gbcformer}
rc=0
quick=${1:-}
tmp=$(mktemp -d)

say() { print -r -- "$@"; }
last=$SECONDS
took() { REPLY="[$(( SECONDS - last ))s]"; last=$SECONDS; }
fail() { took; say "FAIL  $* $REPLY"; rc=1; }
pass() { took; say "ok    $* $REPLY"; }
# Nonzero when any engine ran a pass on a dense fallback (an engine prints it on two lines, both counted).
fallbacks() { grep -Eho '[0-9]+ (on the )?dense fallback' "$@" | awk '{s+=$1} END {print s+0}'; }

# One `run` whose stdout line carries frame_sha1=…; args after the expected prefix go to the
# binary. The run reports its engines, and a pass on a dense fallback fails it.
check_hash() {
  local name=$1 want=$2; shift 2
  local out got fb
  if ! out=$(GBCFORMER_STATS=1 "$BIN" run "$@" 2>$tmp/err); then
    fail "$name: run exited $? ($(tail -1 $tmp/err))"
    return
  fi
  got=$(print -r -- "$out" | sed -n 's/.*frame_sha1=\([0-9a-f]*\).*/\1/p' | tail -1) fb=$(fallbacks $tmp/err)
  if [[ $got == $want* && $fb == 0 ]]; then pass "$name: $got"; else fail "$name: got ${got:-<none>}, want $want…, $fb passes on the dense fallback"; fi
}

say "== merge-conflict markers in tracked files"
if markers=$(tools/conflict-markers.sh); then pass "no conflict markers"; else fail "conflict markers:"; print -r -- "$markers" | head -20; fi

say "== cited commits outside this history"
if cites=$(tools/dead-citations.sh); then pass "no dead citations"; else fail "dead citations:"; print -r -- "$cites" | head -20; fi

say "== build (no warnings)"
build=$(cargo build --release --all-targets 2>&1)
if print -r -- "$build" | grep -q "^warning\|^error"; then fail "build: $(print -r -- "$build" | grep -c '^warning\|^error') warnings or errors"; print -r -- "$build" | grep -A3 '^warning\|^error' | head -40; else pass "build"; fi

say "== wasm32 check (lib, the gbc feature without the frontend)"
if cargo check --lib --no-default-features --features gbc --target wasm32-unknown-unknown 2>&1 | grep -q '^warning\|^error'; then fail "wasm32"; else pass "wasm32"; fi

say "== the inference core alone (no gbc feature)"
if cargo check --lib --no-default-features 2>&1 | grep -q '^warning\|^error'; then fail "inference core without gbc"; else pass "inference core without gbc"; fi

say "== lockstep hashes"
check_hash "DMG 8 lockstep" 6872b003 pokemon-yellow.gbc --frames 8 --core lockstep --hardware dmg --no-save
check_hash "CGB 300 lockstep" cfd31f3d pokemon-yellow.gbc --frames 300 --core lockstep --no-save
if [[ $quick == --quick ]]; then exit $rc; fi

say "== cargo test --release --features testing"
# Without the SingleStepTests cases the SST tests skip, which is not a pass of this gate.
[[ -d vendor/sm83/v1 ]] || fail "no vendor/sm83/v1: the SST tests would skip"
if [[ ${GATE_NO_TEST:-0} == 1 ]]; then say "skip  cargo test: GATE_NO_TEST=1, the caller ran it"
elif cargo test --release --features testing 2>&1 | grep -q "^test result: FAILED\|^error\|panicked"; then fail "cargo test"; else pass "cargo test"; fi

say "== checkpoint digests (tools/digests.txt)"
if digests=$(cargo run --release --example digest -- --check tools/digests.txt 2>&1) && print -r -- "$digests" | grep -q '^digests ok'; then pass "$(print -r -- "$digests" | grep '^digests ok')"; else fail "digests: $(print -r -- "$digests" | grep 'MOVED\|panicked\|^error' | head -3)"; fi

say "== other games, 1,500 frames lockstep"
check_hash "Crystal" 4e6a7509 pokemon-crystal.gbc --frames 1500 --core lockstep --no-save
check_hash "Zelda DX" 99f51381 legend-of-zelda-links-awakening.gbc --frames 1500 --core lockstep --no-save
check_hash "Mario DX" b083771f super-mario-bros-deluxe.gbc --frames 1500 --core lockstep --no-save

say "== benchmarks (model core)"
for b in text:870aa3fd overworld:822059c1; do
  name=${b%%:*} want=${b##*:}
  out=$(BIN=$BIN GBCFORMER_STATS=1 tools/bench/$name.sh 2>$tmp/err)
  got=$(print -r -- "$out" | sed -n 's/.*frame_sha1=\([0-9a-f]*\).*/\1/p' | tail -1) fb=$(fallbacks $tmp/err)
  if [[ $got == $want* && $fb == 0 ]]; then pass "$name.sh: $got"; else fail "$name.sh: got ${got:-<none>}, want $want…, $fb passes on the dense fallback"; fi
done

say "== blargg"
for r in cpu_instrs instr_timing; do
  if "$BIN" run vendor/blargg/$r.gb --frames 4000 --core ref --no-save 2>/dev/null | grep -q "Passed"; then pass "$r"; else fail "$r: no Passed on serial"; fi
done
check_hash "dmg_sound 01" 99793856 vendor/blargg/dmg_sound/01-registers.gb --frames 2000 --core ref --hardware dmg --no-save
check_hash "dmg_sound 02" 881f3fe7 vendor/blargg/dmg_sound/02-len_ctr.gb --frames 2000 --core ref --hardware dmg --no-save
check_hash "dmg_sound 03" e494d943 vendor/blargg/dmg_sound/03-trigger.gb --frames 2000 --core ref --hardware dmg --no-save
check_hash "dmg_sound 04" 8e776b75 vendor/blargg/dmg_sound/04-sweep.gb --frames 2000 --core ref --hardware dmg --no-save

say "== acid2"
# Both images pixel-exact; extra args name the stack.
acid2() {
  local name=$1; shift
  BIN=$BIN tools/acid2.sh "$@" > $tmp/acid 2>&1
  if [[ $(grep -c pixel-exact $tmp/acid) == 2 ]]; then pass "$name pixel-exact"; else fail "$name: $(grep -v pixel-exact $tmp/acid | head -2)"; fi
}
acid2 "acid2"

say "== speed switch"
# daid's speed_switch_timing div, ly and stat against their expected pixel counts.
if [[ ! -d vendor/daid ]]; then
  say "skip  speed switch: no vendor/daid (speed_switch_timing_*.gbc and .png, tools/speed-switch.sh)"
elif ! command -v magick >/dev/null; then
  say "skip  speed switch: ImageMagick (magick) not installed"
elif BIN=$BIN tools/speed-switch.sh > $tmp/speed 2>&1; then
  pass "speed switch: $(tr '\n' ';' < $tmp/speed)"
else
  fail "speed switch: $(grep -v 'as expected' $tmp/speed | tr '\n' ';')"
fi

# The phase gates with each component in weights and its reference beside it, the
# first divergence stopping the run (CLAUDE.md, "Each component in weights, and its gate").
say "== in weights, lockstep"
if "$BIN" run vendor/blargg/instr_timing.gb --frames 4000 --weights cpu,control --lockstep all --no-save 2>/dev/null | grep -q "Passed"; then pass "instr_timing cpu,control"; else fail "instr_timing cpu,control: no Passed on serial"; fi
check_hash "dmg_sound 01 apu,control" 99793856 vendor/blargg/dmg_sound/01-registers.gb --frames 2000 --weights apu,control --lockstep all --hardware dmg --no-save
check_hash "dmg_sound 02 apu,control" 881f3fe7 vendor/blargg/dmg_sound/02-len_ctr.gb --frames 2000 --weights apu,control --lockstep all --hardware dmg --no-save
check_hash "dmg_sound 03 apu,control" e494d943 vendor/blargg/dmg_sound/03-trigger.gb --frames 2000 --weights apu,control --lockstep all --hardware dmg --no-save
check_hash "dmg_sound 04 apu,control" 8e776b75 vendor/blargg/dmg_sound/04-sweep.gb --frames 2000 --weights apu,control --lockstep all --hardware dmg --no-save
acid2 "acid2 cpu,memory,ppu,control" --weights cpu,memory,ppu,control --lockstep all
all=(--weights cpu,memory,apu,ppu,control --lockstep all --no-save)
check_hash "Yellow 300 all" cfd31f3d pokemon-yellow.gbc --frames 300 $all
check_hash "Crystal 300 all" ccfe7ad3 pokemon-crystal.gbc --frames 300 $all
check_hash "Zelda DX 300 all" f10b2b4a legend-of-zelda-links-awakening.gbc --frames 300 $all
check_hash "Mario DX 300 all" ea7f86ca super-mario-bros-deluxe.gbc --frames 300 $all
# The same stack as one checkpoint, each component a stream of it (docs/one-model.md);
# lockstep stops the run at the first line, sample or step that differs.
check_hash "Yellow 300 all, one model" cfd31f3d pokemon-yellow.gbc --frames 300 $all --one-model
# The same run with the step oracle (docs/one-model.md §4): sampled fast passes of every
# stream also run through the stream's dense pass over the same KV cache, and the first
# decoded token that differs stops the run. Each stream must report its checks, in the
# order the run prints them: the GBC CPU, the APU, control, the PPU and its length (the
# length stream shares the PPU's cadence).
oracle=cpu=100000,apu=25000,control=6000,ppu=8400
if out=$(GBCFORMER_STEP_ORACLE=$oracle "$BIN" run pokemon-yellow.gbc --frames 300 $all --one-model 2>&1); then
  got=$(print -r -- "$out" | sed -n 's/.*frame_sha1=\([0-9a-f]*\).*/\1/p' | tail -1)
  checks=(${(f)"$(print -r -- "$out" | sed -n 's/.*; \([0-9]*\) fast passes checked by the step oracle.*/\1/p')"})
  if [[ $got == cfd31f3d* && ${#checks} == 5 ]]; then pass "Yellow 300 all, one model, step oracle $oracle: $got, passes checked ${(j:, :)checks}"; else fail "Yellow 300 one model, step oracle: got ${got:-<none>}, want cfd31f3d…, ${#checks} of 5 streams checked"; fi
else
  fail "Yellow 300 one model, step oracle: run exited ($(print -r -- "$out" | grep -m1 'step oracle:\|panicked'))"
fi
# Yellow draws no line with the pixel-FIFO stepper, so all six streams are sampled together
# on Mealybug m3_bgp_change, whose every line the stepper draws (about 900 passes a line),
# and the memory records (every record token above 65,535 in the one model, whose
# vocabulary passes 2^16) through the dense record oracle (`GBCFORMER_ORACLE`).
m3=vendor/hwtests/gbtr/mealybug-tearoom-tests/ppu/m3_bgp_change.gb
if [[ ! -f $m3 ]]; then
  say "skip  the stepper's step oracle: no $m3"
elif out=$(GBCFORMER_STEP_ORACLE=cpu=20000,apu=4000,control=2000,ppu=20000 GBCFORMER_ORACLE=500 "$BIN" run $m3 --frames 4 $all --one-model 2>&1); then
  checks=(${(f)"$(print -r -- "$out" | sed -n 's/.*; \([0-9]*\) fast passes checked by the step oracle.*/\1/p')"})
  stepper=$(print -r -- "$out" | sed -n 's/^ppu stepper: .*; \([0-9]*\) fast passes checked by the step oracle.*/\1/p')
  records=$(print -r -- "$out" | sed -n 's/.* records kept, \([0-9]*\) dense oracle checks.*/\1/p' | head -1)
  if [[ ${#checks} == 6 && ${#${(M)checks:#0}} == 0 && ${stepper:-0} -gt 0 && ${records:-0} -gt 0 ]]; then pass "m3_bgp_change 4, one model, the step oracle on all six streams: passes checked ${(j:, :)checks}; $records records through the dense record oracle"; else fail "m3_bgp_change 4, one model: step-oracle checks ${(j:, :)checks} (want six above 0), stepper ${stepper:-none}, record oracle ${records:-none}"; fi
else
  fail "m3_bgp_change 4 one model, step oracle: run exited ($(print -r -- "$out" | grep -m1 'step oracle:\|panicked'))"
fi
# The OAM corruption bug's rewrite passes are rare, so with the step oracle on each one is
# checked whatever the cadence (`MemoryDenoiser::check_next`): blargg's 2-causes on DMG.
ob=vendor/hwtests/gbtr/blargg/oam_bug/rom_singles/2-causes.gb
if [[ ! -f $ob ]]; then
  say "skip  the OAM rewrite passes' step oracle: no $ob"
elif out=$(GBCFORMER_STEP_ORACLE=cpu=200000 "$BIN" run $ob --frames 120 --hardware dmg $all --one-model 2>&1); then
  n=$(print -r -- "$out" | sed -n 's/^cpu engine: .*, \([0-9]*\) OAM rewrite passes among them.*/\1/p') bugs=$(print -r -- "$out" | grep -o 'oam_bugs=[0-9]*' | cut -d= -f2)
  if [[ ${n:-0} -gt 0 && ${bugs:-0} -gt 0 ]]; then pass "2-causes 120 DMG, one model, the step oracle on $n OAM rewrite passes of $bugs"; else fail "2-causes 120 DMG one model: ${n:-no} OAM rewrite passes checked, oam_bugs ${bugs:-none}"; fi
else
  fail "2-causes 120 DMG one model, step oracle: run exited ($(print -r -- "$out" | grep -m1 'step oracle:\|panicked'))"
fi

# Nintendo's CGB boot ROM, if you supply it (never in the repository; tools/link-data.sh links it), run from
# power-on with all five in weights, the memory model mapping it: Yellow, then a DMG
# cartridge through it into compatibility mode, in lockstep and as one model alone.
say "== boot ROM"
if [[ -f cgb_boot.bin ]]; then
  check_hash "Yellow 400 from the CGB boot ROM, all" 0db60d9b pokemon-yellow.gbc --frames 400 --boot cgb_boot.bin $all
  check_hash "dmg-acid2 on CGB from the boot ROM, all" 9c7eee04 vendor/acid2/dmg-acid2.gb --hardware cgb --frames 300 --boot cgb_boot.bin $all
  check_hash "dmg-acid2 on CGB from the boot ROM, one model, no lockstep" 9c7eee04 vendor/acid2/dmg-acid2.gb --hardware cgb --frames 300 --boot cgb_boot.bin --weights all --one-model --no-save
else
  say "skip  no cgb_boot.bin"
fi
# SameBoy's open boot ROMs (MIT, tools/hwtests/fetch.sh), so the boot path is gated without
# Nintendo's file.
sb=vendor/hwtests/sameboy
if [[ -f $sb/cgb_boot.bin && -f $sb/dmg_boot.bin ]]; then
  check_hash "Yellow 400 from SameBoy's CGB boot ROM, all" 0db60d9b pokemon-yellow.gbc --frames 400 --boot $sb/cgb_boot.bin $all
  check_hash "dmg-acid2 from SameBoy's DMG boot ROM, all" 52bc2177 vendor/acid2/dmg-acid2.gb --hardware dmg --frames 300 --boot $sb/dmg_boot.bin $all
else
  say "skip  no vendor/hwtests/sameboy (tools/hwtests/fetch.sh)"
fi
# The boot ROM run by the one model with every component in lockstep, to the FF50 hand-off,
# then the whole hand-off state (the GBC CPU, the time, every I/O register, the display
# bytes, every RAM) held to a reference-only run of the same boot ROM (`examples/boot_proof`):
# DMG, CGB, and a DMG cartridge on a CGB in compatibility mode.
say "== boot proofs"
boot_proof() {
  local name=$1; shift
  if out=$(cargo run --release --example boot_proof -- "$@" --one-model 2>&1) && print -r -- "$out" | grep -q "^boot proof: the hand-off in weights equals"; then pass "boot proof, $name: $(print -r -- "$out" | grep '^hand-off at')"; else fail "boot proof, $name: $(print -r -- "$out" | grep -m3 '^differs\|^boot proof\|panicked' | tr '\n' ';')"; fi
}
if [[ -f cgb_boot.bin ]]; then
  boot_proof "Nintendo's CGB boot ROM, Yellow" cgb_boot.bin pokemon-yellow.gbc
  boot_proof "Nintendo's CGB boot ROM, dmg-acid2 in compatibility mode" cgb_boot.bin vendor/acid2/dmg-acid2.gb
else
  say "skip  boot proofs with Nintendo's CGB boot ROM: no cgb_boot.bin (a file you supply, never in the repository)"
fi
if [[ -f dmg_boot.bin ]]; then
  boot_proof "Nintendo's DMG boot ROM, Yellow on DMG" dmg_boot.bin pokemon-yellow.gbc
else
  say "skip  boot proof with Nintendo's DMG boot ROM: no dmg_boot.bin (a file you supply, never in the repository); SameBoy's DMG boot ROM stands in"
fi
if [[ -f $sb/cgb_boot.bin && -f $sb/dmg_boot.bin ]]; then
  boot_proof "SameBoy's CGB boot ROM, Yellow" $sb/cgb_boot.bin pokemon-yellow.gbc
  boot_proof "SameBoy's CGB boot ROM, dmg-acid2 in compatibility mode" $sb/cgb_boot.bin vendor/acid2/dmg-acid2.gb
  boot_proof "SameBoy's DMG boot ROM, Yellow on DMG" $sb/dmg_boot.bin pokemon-yellow.gbc
else
  say "skip  boot proofs with SameBoy's boot ROMs: no vendor/hwtests/sameboy (tools/hwtests/fetch.sh)"
fi

# The minimal runner (docs/minimal-runner.md): its quarantine and dependency tests against
# the one model's GGUF, then the runner held to the full program pass record by pass record,
# from the frame-0 hand-off, from a snapshot mid-run (the text bench's warmup) and, with the
# CGB boot ROM, from power-on. Recording needs the stock and record features, built apart.
say "== minimal runner"
if [[ -n ${GGUF:-} ]]; then
  mkdir -p $tmp/model; ln -s ${GGUF:A} $tmp/model/model.gguf; export_log="GGUF=$GGUF"
else
  export_log=$(cargo run --release --example export -- one-model $tmp/model --gguf-only --layer-shapes --align 64 2>&1)
fi
if [[ ! -f $tmp/model/model.gguf ]]; then
  fail "export: $(print -r -- "$export_log" | tail -1)"
else
  if (cd tools/runner && GBCFORMER_GGUF=$tmp/model/model.gguf cargo test --release 2>&1) | grep -q "^test result: FAILED\|^error\|panicked"; then fail "runner tests (quarantine, dependencies)"; else pass "runner tests (quarantine, dependencies)"; fi
  stock=${STOCK_BIN:-} runner=${RUNNER_BIN:-} checked=${RUNNER_CHECKED_BIN:-}
  if [[ -z $stock || -z $runner ]]; then
    rec=$PWD/target-stock stock=$PWD/target-stock/release/gbcformer runner=$PWD/target-stock/release/gbcformer-runner
    CARGO_TARGET_DIR=$rec cargo build --release --features stock --bin gbcformer >/dev/null 2>&1 && (cd tools/runner && CARGO_TARGET_DIR=$rec cargo build --release --features record >/dev/null 2>&1) || stock=
  fi
  # The runner as it is timed (no record feature), with debug assertions, in its own directory.
  if [[ -z $checked ]]; then
    checked=$PWD/target-stock/checked/release/gbcformer-runner
    (cd tools/runner && CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true CARGO_TARGET_DIR=${checked:h:h} cargo build --release >/dev/null 2>&1) || checked=
  fi
  if [[ ! -x $stock || ! -x $runner ]]; then
    fail "the recording builds"
  else
    proof() {
      local name=$1 as=${AS:+, $AS}; shift
      if FULL=$stock RUNNER=${WITH:-$runner} tools/runner/proof.sh $tmp/model/model.gguf $tmp/proof-$name "$@" > $tmp/proof 2>&1; then pass "runner, $name$as: $(grep -c '^ok' $tmp/proof) checks"; else fail "runner, $name$as: $(grep '^FAIL' $tmp/proof | head -3 | tr '\n' ';')"; fi
    }
    proof "Yellow 300" pokemon-yellow.gbc 300
    P=(); for f in $(seq 2300 40 3100); do P+=(--press "${f}:a:4" --press "$((f+20)):down:4"); done
    proof "Yellow from the warmed-up text snapshot, frames 2900-3100" pokemon-yellow.gbc 3100 --warmup 2900 $P
    # A recording run places the whole sequence after every write record; the timed build
    # places it by its change list. That build with debug assertions, which hold every pass's
    # change list and decoded store to the whole sequence, with no record: its frames, samples
    # and saves against the full program's from the two proofs above.
    if [[ -x $checked ]]; then
      AS="plain with debug assertions" WITH=$checked PLAIN=1 REUSE=1 proof "Yellow 300" pokemon-yellow.gbc 300
      AS="plain with debug assertions" WITH=$checked PLAIN=1 REUSE=1 proof "Yellow from the warmed-up text snapshot, frames 2900-3100" pokemon-yellow.gbc 3100 --warmup 2900 $P
    else
      fail "the runner's build with debug assertions"
    fi
    OVERSHOOT=1 proof "Crystal from a snapshot 4 halves past frame 200's end, frames 200-400" pokemon-crystal.gbc 400 --warmup 200
    # A lockstep run takes the passes a plain run takes: every stream's pass record equal,
    # every event, with every component in lockstep. The memory core's audit-only slots are
    # left out of both records (`tools/runner/same-passes.sh`).
    same_passes() {
      local name=$1; shift
      if FULL=$stock tools/runner/same-passes.sh $tmp/model/model.gguf $tmp/lsp-$name "$@" > $tmp/lsp 2>&1; then pass "same passes plain and in lockstep, $name: $(grep 'records equal\|left out' $tmp/lsp | tr '\n' ';')"; else fail "same passes, $name: $(grep '^FAIL' $tmp/lsp | head -3 | tr '\n' ';')"; fi
      rm -rf $tmp/lsp-$name
    }
    same_passes oam_dma_restart vendor/hwtests/gbtr/mooneye-test-suite/acceptance/oam_dma_restart.gb 30
    same_passes m3_bgp_change vendor/hwtests/gbtr/mealybug-tearoom-tests/ppu/m3_bgp_change.gb 4
    # Every record, uncapped; a marked line's length revised before the control logic reads it
    # and a taken write after it, as many lines on both sides.
    STREAMS=6 EQUAL="revised_before taken_after" proof "Mealybug m3_bgp_change, the stepper drawing every line" vendor/hwtests/gbtr/mealybug-tearoom-tests/ppu/m3_bgp_change.gb 4
    # The same case with every cell lookup also read through the softmax over the stored K/V
    # rows, which the runner's worker thread writes (GBCFORMER_SCAN=1 for the runner alone).
    AS="every lookup through the stored rows" SCAN=1 REUSE=1 STREAMS=6 EQUAL="revised_before taken_after" proof "Mealybug m3_bgp_change, the stepper drawing every line" vendor/hwtests/gbtr/mealybug-tearoom-tests/ppu/m3_bgp_change.gb 4
    # A display register read back in the line that wrote it: the byte from the record queue.
    if cargo run --release --example display_read_rom -- $tmp/display_read.gb > /dev/null 2>&1; then
      EQUAL="forwarded" proof "a display read forwarded from the record queue, DMG" $tmp/display_read.gb 20 --hardware dmg
    else
      fail "the display-read ROM"
    fi
    # A boot ROM of our own that sets compatibility mode while line 10 is open: the COMPAT
    # record closes the row on both sides (the runner at its next own-memory write).
    if cargo run --release --example compat_boot_rom -- $tmp/compat_boot.bin $tmp/compat_cart.gb > /dev/null 2>&1; then
      EQUAL="compat_closes" proof "a COMPAT record with a row open, a DMG cartridge on CGB" $tmp/compat_cart.gb 30 --boot $tmp/compat_boot.bin
    else
      fail "the compatibility boot ROM"
    fi
    # The OAM corruption bug on DMG, where it fires (blargg's ROMs claim CGB support, so
    # auto would pick CGB): the same rewrite passes on both sides, more than none.
    REWRITES=1 proof "blargg oam_bug 2-causes on DMG, the OAM bug firing" vendor/hwtests/gbtr/blargg/oam_bug/rom_singles/2-causes.gb 120 --hardware dmg
    if cargo run --release --example fetch_locks_rom -- $tmp/fetch_locks.gb > /dev/null 2>&1; then
      REWRITES=1 proof "fetch-locks on DMG, fetches under the locks and the OAM bug" $tmp/fetch_locks.gb 24 --hardware dmg
    else
      fail "the fetch-locks ROM"
    fi
    # Every stream's passes from real runs held to the whole one-model checkpoint's dense pass
    # (not the stream's slice): Yellow 60 and m3_bgp_change 2, one event in every few thousand
    # recorded, three per stream checked (`examples/slice_full.rs`). Every stream memory of at
    # most 20,000 cells comes whole, the PPU family's on both and the GBC CPU's on
    # m3_bgp_change (12,758 cells), which slice_full requires; Yellow's GBC CPU cache passes
    # over the cells its heads read.
    sf=$tmp/slice; rm -rf $sf; mkdir -p $sf
    for c in "yellow pokemon-yellow.gbc 60 4000" "m3 vendor/hwtests/gbtr/mealybug-tearoom-tests/ppu/m3_bgp_change.gb 2 3000"; do
      set -- ${=c}; mkdir -p $sf/$1
      GBCFORMER_GGUF_ONLY=$tmp/model/model.gguf GBCFORMER_APU_THREAD=0 GBCFORMER_PPU_THREAD=0 GBCFORMER_RECORD=$sf/$1 GBCFORMER_RECORD_CELLS=20000 GBCFORMER_RECORD_WINDOW=1 GBCFORMER_RECORD_EVERY=$4 $stock run $2 --frames $3 --weights all --one-model --no-save > $sf/$1.log 2>&1 || fail "the $1 record run exited $?"
    done
    cargo run --release --example halt_bug_rom -- $tmp/halt_bug.gb > /dev/null 2>&1 || fail "the HALT-bug ROM"
    if out=$(cargo run --release --example slice_full -- $sf/yellow $sf/m3 --passes 3 --cells 20000 2>/dev/null); then pass "every stream against the whole checkpoint over real runs: $(print -r -- "$out" | grep -o '[0-9]* tokens equal' | awk '{s+=$1} END {print s}') tokens equal"; else fail "the whole checkpoint over real runs: $(print -r -- "$out" | grep -m3 'FAIL' | tr '\n' ';')"; fi
    rm -rf $sf
    # The HALT bug across a lock edge (docs/hardware-tests.md §5): the opcode's two reads under
    # their own bus states, the runner's passes held to the one model's
    # (tests/halt_bug_locks.rs holds the one model's log to the reference's).
    proof "the HALT bug across a lock edge, DMG" $tmp/halt_bug.gb 60 --hardware dmg
    # A DMG's read and write locks differ at M-cycles 19 and 113 (docs/hardware-tests.md §3.17).
    for r in lcdon_timing-GS lcdon_write_timing-GS; do
      LOCKPAIRS=1 proof "mooneye $r on DMG, the read and write locks" vendor/hwtests/gbtr/mooneye-test-suite/acceptance/ppu/$r.gb 60 --hardware dmg
    done
    # An HBlank transfer through a HALT runs the spans stream's rest and wake passes
    # (docs/hardware-tests.md §3.14), as many in the runner as in the full program.
    HALTS=1 proof "Mealybug hdma_during_halt-C, the rest and wake passes" vendor/hwtests/gbtr/mealybug-tearoom-tests/dma/hdma_during_halt-C.gb 40
    # HBlank blocks inside steps, each counted in HDMA5 by a pass at the step's end (§3.14).
    HALTS=1 proof "Mealybug hdma_timing-C, the step-end passes" vendor/hwtests/gbtr/mealybug-tearoom-tests/dma/hdma_timing-C.gb 60
    if [[ -f cgb_boot.bin ]]; then proof "Yellow 300 from power-on" pokemon-yellow.gbc 300 --boot cgb_boot.bin; else say "skip  runner from power-on: no cgb_boot.bin"; fi
  fi
fi
rm -rf "$tmp"
exit $rc
