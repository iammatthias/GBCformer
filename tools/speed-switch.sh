#!/bin/sh
# Runs daid's CGB speed-switch tests and compares each last frame with the hardware
# screenshot pixel for pixel: all three pixel-exact since LY reads the next line in a
# line's last M-cycle (docs/hardware-tests.md §3.8; ly was 109 pixels off before).
# Exits 1 when a run fails or a count drifts either way. Needs ImageMagick and, in vendor/daid/
# (gitignored), speed_switch_timing_{div,ly,stat}.gbc and .png from testroms/daid in
# github.com/gbdev/GBEmulatorShootout (MIT).
#
# Arguments after the script name go to every `run`, so a stack can be named:
#   tools/speed-switch.sh --weights all --lockstep all --one-model
set -eu
cd "$(dirname "$0")/.."
bin=${BIN:-target/release/gbcformer}
dir=${DAID:-vendor/daid}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
status=0
for case in div:0 ly:0 stat:0; do
  t=${case%%:*} want=${case##*:}
  if ! "$bin" run "$dir/speed_switch_timing_$t.gbc" --frames 60 --no-save --dump "$out/$t.ppm" "$@" > "$out/$t.log" 2>&1; then
    echo "$t: run failed ($(tail -1 "$out/$t.log"))"
    status=1
    continue
  fi
  magick "$out/$t.ppm" "$out/$t.png"
  differ=$(magick compare -metric AE "$out/$t.png" "$dir/speed_switch_timing_$t.png" null: 2>&1 | cut -d" " -f1 || true)
  if [ "$differ" = "$want" ]; then
    echo "$t: $differ pixels differ, as expected"
  else
    echo "$t: $differ pixels differ, want $want"
    status=1
  fi
done
exit $status
