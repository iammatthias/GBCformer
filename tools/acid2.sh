#!/bin/sh
# Runs dmg-acid2 and cgb-acid2 and compares the last frame with each reference image
# pixel for pixel. Needs ImageMagick and, in vendor/acid2/ (gitignored), the released
# ROMs and reference PNGs:
#   dmg-acid2.gb, reference-dmg.png  from github.com/mattcurrie/dmg-acid2
#   cgb-acid2.gbc, reference-cgb.png from github.com/mattcurrie/cgb-acid2
# The DMG reference is greyscale; its four greys are mapped onto the host's shades,
# lightest first: white -> E0F8D0, AAAAAA -> 88C070, 555555 -> 346856, black -> 081820.
#
# Arguments after the script name go to every `run`, so a stack can be named:
#   tools/acid2.sh --weights cpu,memory,ppu,control --lockstep all
set -eu
cd "$(dirname "$0")/.."
bin=${BIN:-target/release/gbcformer}
dir=vendor/acid2
out=$(mktemp -d)
status=0

check() {
  name=$1 rom=$2 hardware=$3 reference=$4
  shift 4
  if ! "$bin" run "$dir/$rom" --frames 60 --hardware "$hardware" --no-save --dump "$out/$name.ppm" "$@" > "$out/$name.log" 2>&1; then
    echo "$name: run failed ($(tail -1 "$out/$name.log"))"
    status=1
    return
  fi
  magick "$out/$name.ppm" "$out/$name.png"
  differ=$(magick compare -metric AE "$out/$name.png" "$reference" null: 2>&1 | cut -d" " -f1 || true)
  if [ "$differ" = "0" ]; then
    echo "$name: pixel-exact"
  else
    echo "$name: $differ pixels differ (output $out/$name.png)"
    status=1
  fi
}

magick "$dir/reference-dmg.png" -colorspace sRGB \
  -fill '#081820' -opaque '#000000' -fill '#346856' -opaque '#555555' \
  -fill '#88C070' -opaque '#AAAAAA' -fill '#E0F8D0' -opaque '#FFFFFF' "PNG24:$out/reference-dmg-shades.png"
check dmg-acid2 dmg-acid2.gb dmg "$out/reference-dmg-shades.png" "$@"
check cgb-acid2 cgb-acid2.gbc cgb "$dir/reference-cgb.png" "$@"
exit $status
