#!/bin/zsh
# A profile-guided build of gbcformer (docs/perf-log.md, "Round 5b").
# The plain `cargo build --release` stays the default build, and the one the tests and the gate use.
#
#   tools/build-pgo.sh [cargo build targets, default --bin gbcformer]
#
# 1. Builds the binary with rustc's instrumentation (-Cprofile-generate) in $PGO_DIR/gen.
# 2. Runs the text, overworld and battle benches on it, as one model with everything in weights.
# 3. Merges the counts with the toolchain's llvm-profdata (rustup component add llvm-tools).
# 4. Builds the targets with -Cprofile-use in $PGO_DIR/use, and holds the result's text frame
#    to the instrumented build's.
# The binary is $PGO_DIR/use/release/gbcformer (PGO_DIR defaults to target/pgo). It needs the ROM.
# CARGO_BUILD_JOBS limits both builds. The thread interleaving of the profile runs differs from
# run to run, so two profiles, and the two binaries built from them, are not bit-identical.
set -eu
cd "${0:A:h}/.."
D=${${PGO_DIR:-target/pgo}:A}
targets=("$@"); (( $# )) || targets=(--bin gbcformer)
PD=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata
[ -x $PD ] || { print -u2 "no $PD: rustup component add llvm-tools"; exit 2; }
frame() { grep -o 'frame_sha1=[0-9a-f]*' $1 | tail -1 | cut -d= -f2; }

RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-generate=$D/raw" cargo build --release --target-dir $D/gen --bin gbcformer
# Build scripts and proc-macros ran instrumented too; their counts are not the game's.
rm -rf $D/raw
for w in text overworld battle; do
  BIN=$D/gen/release/gbcformer tools/bench/$w.sh --weights all --one-model > $D/$w.log 2>&1
  print "profile run $w: frame $(frame $D/$w.log)"
done
$PD merge -o $D/gbcformer.profdata $D/raw
RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-use=$D/gbcformer.profdata" cargo build --release --target-dir $D/use $targets
BIN=$D/use/release/gbcformer tools/bench/text.sh --weights all --one-model > $D/check.log 2>&1
[ -n "$(frame $D/check.log)" ] && [ "$(frame $D/check.log)" = "$(frame $D/text.log)" ] || { print -u2 "the PGO build's text frame $(frame $D/check.log) is not the instrumented build's $(frame $D/text.log)"; exit 1; }
print "$D/use/release/gbcformer: text frame $(frame $D/check.log), $(grep -o '^[0-9.]* emulated fps' $D/check.log)"
