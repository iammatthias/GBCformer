#!/bin/bash
# The harness's stops, each shown to fire on a deliberate fault, under `play replay`: no
# model and no GPU. Every pass decodes the tokens of the engine's record, so a run here
# shows that a check stops the run, never that a model's tokens are right. One clean
# replay first, which must end with exit 0 and no error line; then one replay per fault,
# which must print its FAULT INJECTED line and then stop at the named check with exit 1.
#
#   tools/stock-llama/faults.sh <stock-llama binary> <engine's record dir> <rom> [out dir] [frames]
#
# Not shown here, because a replay keeps no sequences and reads no cells: STOCK_FAULT_SEQUENCE
# (`SCOPE SEQUENCE DIFFERS`) and a fault without the engine's record, which need a model.
set -u
BIN=$1 REF=$2 ROM=$3 OUT=${4:-$(mktemp -d)} FRAMES=${5:-2}
ERR='REFERENCE DIFFERS|REFERENCE ENDS|MIRROR DIFFERS|GROUP ROW DIFFERS|SWAP NOT HELD|READ NOT HELD|SCOPE MISSES LINE|NARROW SCOPE NOT HELD|WINDOWS SCOPE NOT HELD|SCOPE SEQUENCE DIFFERS|CHECK NOT HELD|READ DIFFERS|MAP SLOT NOT A BYTE|panicked'
mkdir -p "$OUT"
rc=0
run() { # name, the line the run must stop at (empty: a clean run), settings
    local name=$1 want=$2; shift 2
    env STOCK_REFERENCE="$REF" "$@" nice -n 19 "$BIN" play replay "$ROM" --frames "$FRAMES" > "$OUT/$name.log" 2>&1
    local code=$? errors
    errors=$(grep -cE "$ERR" "$OUT/$name.log")
    if [ -z "$want" ]; then
        if [ "$code" = 0 ] && [ "$errors" = 0 ]; then echo "ok    $name: exit 0, no error line, $(grep -cE '^frame [0-9]+ sha1' "$OUT/$name.log") frames"; else echo "FAIL  $name: exit $code, $errors error lines"; rc=1; fi
    elif [ "$code" = 1 ] && grep -q 'FAULT INJECTED' "$OUT/$name.log" && grep -qE "$want" "$OUT/$name.log"; then
        echo "ok    $name: stopped with exit 1 at: $(grep -E "$want" "$OUT/$name.log" | head -1 | cut -c1-230)"
    else
        echo "FAIL  $name: exit $code, fault lines $(grep -c 'FAULT INJECTED' "$OUT/$name.log"), wanted a line matching: $want"; rc=1
    fi
}
run clean ''
run scope 'SCOPE MISSES LINE' STOCK_FAULT_SCOPE=2000
run map 'NARROW SCOPE NOT HELD.*its lines were placed under' STOCK_FAULT_MAP=200
run probe 'NARROW SCOPE NOT HELD.*changed when it decoded again over the lines its PC names' STOCK_FAULT_PROBE=0
run named 'NARROW SCOPE NOT HELD.*the addresses the pass names changed' STOCK_FAULT_NAMED=0
run read 'READ NOT HELD' STOCK_FAULT_READ=0
echo "logs: $OUT"
exit $rc
