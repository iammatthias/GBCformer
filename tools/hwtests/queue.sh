#!/bin/sh
# The hardware-test work queue from a census TSV (examples/hwtests.rs output): the
# applicable cases SameBoy passes and the reference fails, grouped by area, as Markdown;
# then, with --unknown, the applicable fails SameBoy has no published verdict for.
#
#   tools/hwtests/queue.sh [--unknown] [docs/hardware-tests/after-a3.tsv]
set -eu
want=PASS
[ "${1:-}" = "--unknown" ] && { want=unknown; shift; }
tsv=${1:-docs/hardware-tests/after-a3.tsv}
# Columns: verdict, suite, ROM, console, detail, mark, SameBoy, area.
awk -F'\t' -v want="$want" '
  $1 != "PASS" && $1 != "NA" && $6 == "APPLIES" && $7 == want {
    rom = $3; sub(/^gbtr\/[^\/]*\//, "", rom)
    n[$8]++; items[$8] = items[$8] sprintf("  - `%s` (%s, %s)\n", rom, $2, $4); total++
  }
  END {
    split("GBC CPU and timing|interrupts|OAM DMA|timer|serial|PPU timing|OAM bug|DMA (HDMA)|APU|PPU mid-scanline|boot state|cartridge (MBC)|I/O registers|PPU rendering", order, "|")
    for (i = 1; i in order; i++) { a = order[i]; if (a in n) { printf "- **%s** (%d)\n%s", a, n[a], items[a]; delete n[a] } }
    for (a in n) { printf "- **%s** (%d), an area missing from the order above\n%s", a, n[a], items[a] }
    printf "\n%d cases.\n", total
  }' "$tsv"
