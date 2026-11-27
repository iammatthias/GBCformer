#!/bin/sh
# Prints the hill-climb's metric, the projected hours for Yellow to reach frame 300 with
# every pass in the runtime. The bench measures each stream's ms per pass over PASSES
# (default 300) passes of `play` from the saved cache state, takes the median of RUNS
# (default 3) runs, and multiplies it by the stream's passes to frame 300 (the engine's
# record, frozen below). It diffs every run's decoded tokens against the engine's record.
# The error count is the sum of differing events, mirror differences and failed runs,
# and any error fails the bench.
#
#   tools/stock-llama/bench.sh <stock-llama binary> <model.gguf> [label]
#
# Extra environment reaches the binary (STOCK_*). The bench takes the GPU through a lock
# directory, so two benches never overlap.
set -eu
BIN=$1
MODEL=$2
LABEL=${3:-$(basename "$BIN")}
HERE=$(cd "$(dirname "$0")" && pwd)
W=${BENCH_WORK:-$HERE/.build}
ROM=$HERE/../../pokemon-yellow.gbc
RUNS=${RUNS:-3}
PASSES=${PASSES:-300}
LOCK=/tmp/gbcformer-gpu.lock
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
trap 'rm -rf "$OUT"; rmdir "$LOCK" 2>/dev/null || true' EXIT
export STOCK_GPU_LAYERS=99
for r in $(seq "$RUNS"); do
    rm -rf "$OUT/rec" && mkdir -p "$OUT/rec"
    echo "load before run $r: $(uptime | sed 's/.*load averages*: //')" >&2
    GBCFORMER_RECORD=$OUT/rec STOCK_STATE=$W/yellow-cache.state STOCK_GROUP=${GROUP:-own} \
        "$BIN" play "$MODEL" "$ROM" --frames 300 --max-passes "$PASSES" --lanes 255 > "$OUT/play-$r.log" 2>&1 || echo "FAILED run $r" >> "$OUT/errors"
    "$BIN" diff "$OUT/rec" "$W/rec-engine" > "$OUT/diff-$r.log" 2>&1 || echo "FAILED diff $r" >> "$OUT/errors"
    grep -E '^(Cpu|Ppu|Apu|Control|Length): ' "$OUT/play-$r.log" | sed -E 's/^([A-Za-z]+): .* ([0-9.]+) ms per pass .*/\1 \2/' >> "$OUT/ms"
    grep -E 'MIRROR DIFFERS|GROUP ROW DIFFERS|SLICE ROW DIFFERS|SLICE NOT RUN|SLICE UNKNOWN|MASK ROW DIFFERS|WRITE CELL DIFFERS|SWAP CELL DIFFERS|SWAP UNCHECKED|SWAP NOT HELD|READ NOT HELD|SCOPE MISSES LINE|NARROW SCOPE NOT HELD|READ ROW DIFFERS|READ DIFFERS|MAP SLOT NOT A BYTE|panicked' "$OUT/play-$r.log" >> "$OUT/errors" || true
    grep -E '^all streams' "$OUT/diff-$r.log" | sed -E 's/.* ([0-9]+) differ.*/\1/' | awk '$1 > 0 { print "differ " $1 }' >> "$OUT/errors"
    grep -qE '^all streams' "$OUT/diff-$r.log" || echo "no diff summary in run $r" >> "$OUT/errors"
done
errors=$(cat "$OUT/errors" 2>/dev/null | wc -l | tr -d ' ')
events=$(grep -hE '^all streams' "$OUT"/diff-*.log | sed -E 's/all streams: ([0-9]+) events.*/\1/' | paste -sd+ - | bc)
# Decoded passes to frame 300 per stream: the fill events of the engine's record
# (`GBCFORMER_RECORD=<dir> stock-llama play none <rom> --frames 300`, one model 45bfb056). The
# stepper has none in these frames. The run also decodes the GBC CPU's 69,514 write records and
# about 12,000 reader records, which this figure leaves out.
awk -v label="$LABEL" -v errors="$errors" -v events="$events" -v runs="$RUNS" '
    BEGIN { n["Cpu"] = 925752; n["Ppu"] = 83498; n["Apu"] = 251674; n["Control"] = 71759; n["Length"] = 41855 }
    { v[$1] = v[$1] " " $2 }
    END {
        total = 0
        for (s in n) {
            k = split(v[s], a, " "); asort_n(a, k); med = a[int((k + 1) / 2)]
            ms[s] = med; total += n[s] * med / 3.6e6
        }
        printf "%s: %.1f hours to frame 300 (Cpu %.1f, Ppu %.1f, Apu %.1f, Control %.1f, Length %.1f ms per pass, median of %d); %d events compared, %d errors\n", label, total, ms["Cpu"], ms["Ppu"], ms["Apu"], ms["Control"], ms["Length"], runs, events, errors
    }
    function asort_n(a, k,   i, j, t) { for (i = 2; i <= k; i++) for (j = i; j > 1 && a[j - 1] + 0 > a[j] + 0; j--) { t = a[j]; a[j] = a[j - 1]; a[j - 1] = t } }
' "$OUT/ms"
[ "$errors" = 0 ] || { cat "$OUT/errors" >&2; exit 1; }
