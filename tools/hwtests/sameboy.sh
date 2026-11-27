#!/bin/sh
# SameBoy's published verdicts as docs/hardware-tests/sameboy.tsv, one row per ROM and
# console: the ROM's path under vendor/hwtests, the console, PASS, FAIL or unknown, and
# the source. The runner (examples/hwtests.rs) joins it to each case; a case with no row
# has SameBoy's verdict unknown.
#
#   tools/hwtests/sameboy.sh [PAGE.html]
#
# The main source is the GBEmulatorShootout results page (gbdev.io/GBEmulatorShootout),
# fetched live unless PAGE.html is given. Each of its tests runs on one model, DMG unless
# its definition in testroms/*.py at SHOOTOUT names model=CGB or SGB, so a verdict covers
# that console only. The page is untrusted data: only its cell classes and test names are
# read, and the script stops unless SameBoy's pass count matches the page's header.
# Two rows come from elsewhere, each named in its source column (FIXED below).
set -eu
cd "$(dirname "$0")/../.."
SHOOTOUT=38b926bdbc26993d1b4c43e97979ecc66287bf02
out=docs/hardware-tests/sameboy.tsv
vendor=vendor/hwtests
[ -d "$vendor/gbtr" ] || { echo "no $vendor/gbtr; run tools/hwtests/fetch.sh" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ $# -ge 1 ]; then cp "$1" "$tmp/page.html"; else curl -sSfL -o "$tmp/page.html" https://gbdev.io/GBEmulatorShootout/; fi
mkdir "$tmp/defs"
for f in acid blargg daid mealybug mooneye samesuite; do
  curl -sSfL -o "$tmp/defs/$f.py" "https://raw.githubusercontent.com/gbdev/GBEmulatorShootout/$SHOOTOUT/testroms/$f.py"
done

# name -> model, from the uncommented Test(...) lines; a name the page shows with
# "(DMG)" or "(GBC)" carries its model itself.
cat "$tmp"/defs/*.py | sed -n 's/^ *Test("\([^"]*\)".*/\1	&/p' | awk -F'\t' '{
  m = "dmg"; if ($2 ~ /model=CGB/) m = "cgb"; else if ($2 ~ /model=SGB/) m = "sgb"; print $1 "\t" m }' > "$tmp/models.tsv"

# One line per page row: the test's name, then SameBoy's cell. Cells are split on every
# <td, so an unclassed "No result" cell keeps its place.
tr '\n' ' ' < "$tmp/page.html" | sed -E 's/<img[^>]*>//g' | sed 's/<tr>/\
<tr>/g' | awk '
  /class=.emulator./ && !col {
    n = split($0, h, "<th class=.emulator.>")
    for (i = 2; i <= n; i++) if (h[i] ~ />SameBoy</) { col = i - 1; hdr = h[i]; sub(/.*SameBoy<\/a> \(/, "", hdr); sub(/\).*/, "", hdr) }
    print "#header\t" hdr; next
  }
  /<th class=.test.>/ {
    name = $0; sub(/.*<th class=.test.>/, "", name); sub(/<\/th>.*/, "", name)
    gsub(/<span class="tooltiptext">[^<]*<\/span>/, "", name); gsub(/<[^>]*>/, "", name); gsub(/&#8203;/, "", name)
    n = split($0, c, "<td")
    cell = c[col + 1]; v = "unknown"
    if (cell ~ /class=.PASS./) v = "PASS"; else if (cell ~ /class=.FAIL./) v = "FAIL"; else if (cell ~ /class=.UNKNOWN./) v = "UNKNOWN"; else if (cell ~ /class=.INFO./) v = "INFO"
    print name "\t" v
  }' > "$tmp/rows.tsv"

hdr=$(sed -n 's/^#header\t//p' "$tmp/rows.tsv")
updated=$(tr '\n' ' ' < "$tmp/page.html" | sed -n 's/.*Updated On<br>\([^<]*\)<.*/\1/p')
# The header counts an INFO cell as passed and every cell with a result.
pass=$(awk -F'\t' '$2 == "PASS" || $2 == "INFO"' "$tmp/rows.tsv" | wc -l | tr -d ' ')
judged=$(awk -F'\t' '$1 != "#header" && $2 != "unknown"' "$tmp/rows.tsv" | wc -l | tr -d ' ')
[ "$pass/$judged" = "$hdr" ] || { echo "SameBoy's cells count $pass/$judged, the header says $hdr" >&2; exit 1; }

# The page's name for a ROM -> its path under vendor/hwtests, the first candidate that exists.
resolve() {
  n=$1
  case $n in
    mooneye/*) set -- "gbtr/mooneye-test-suite/${n#mooneye/}" ;;
    samesuite/*) set -- "gbtr/same-suite/${n#samesuite/}" ;;
    mealybug-tearoom-tests/*) set -- "gbtr/$n" ;;
    daid/*) set -- "$n" ;;
    acid/*) b=${n#acid/}; set -- "gbtr/${b%.*}/$b" ;;
    blargg/*/*)
      d=${n#blargg/}; d=${d%%/*}; f=${n##*/}; spaced=$(printf '%s' "$f" | tr _ ' ')
      set -- "gbtr/blargg/$d/individual/$f" "gbtr/blargg/$d/rom_singles/$f" "gbtr/blargg/$d/individual/$spaced" "gbtr/blargg/$d/rom_singles/$spaced" ;;
    blargg/*) b=${n#blargg/}; set -- "gbtr/blargg/$b" "gbtr/blargg/${b%.gb}/$b" ;;
    *) set -- ;;
  esac
  for p in "$@"; do [ -f "$vendor/$p" ] && { printf '%s' "$p"; return; }; done
}

{
  printf 'rom\tconsole\tsameboy\tsource\n'
  grep -v '^#' "$tmp/rows.tsv" | while IFS="$(printf '\t')" read -r name verdict || [ -n "$name" ]; do
    case $verdict in PASS | FAIL) ;; *) continue ;; esac
    rom=$(resolve "${name% (*}")
    if [ -z "$rom" ]; then echo "not in vendor/hwtests: $name" >&2; continue; fi
    case $name in
      *"(DMG)") model=dmg ;;
      *"(GBC)" | *"(CGB"*) model=cgb ;;
      *) model=$(awk -F'\t' -v n="$name" '$1 == n { print $2; exit }' "$tmp/models.tsv") ;;
    esac
    if [ -z "$model" ]; then echo "no definition for $name" >&2; touch "$tmp/failed"; continue; fi
    printf '%s\t%s\t%s\tGBEmulatorShootout %s, SameBoy release then current (v1.0.3, inferred)\n' "$rom" "$model" "$verdict" "$updated"
  done
  # FIXED: verdicts the Shootout does not give.
  printf 'gbtr/blargg/dmg_sound/dmg_sound.gb\tdmg\tPASS\tc-sp game-boy-test-roms v7.0 blargg howto: the expected screenshot was made with SameBoy\n'
  printf 'gbtr/blargg/cgb_sound/cgb_sound.gb\tcgb\tPASS\tc-sp game-boy-test-roms v7.0 blargg howto: the expected screenshot was made with SameBoy\n'
  printf 'gbtr/same-suite/apu/channel_1/channel_1_freq_change_timing-cgbDE.gb\tcgb\tPASS\tSameSuite apu README at f71b4b3c37: SameBoy as CGB-E passes all but channel_4_freq_change and channel_1_sweep_restart_2 (suite-level statement)\n'
} > "$tmp/out.tsv"
[ ! -e "$tmp/failed" ] || exit 1
mv "$tmp/out.tsv" "$out"
echo "$out: $(($(wc -l < "$out") - 1)) rows; SameBoy $hdr on the page updated $updated" >&2
