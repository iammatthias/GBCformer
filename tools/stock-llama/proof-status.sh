#!/bin/bash
# Where the proof run (tools/stock-llama/proof.sh) stands: its chunks and the last
# checkpoint, the frame, the passes, the time over all chunks, the ETA
# from each stream's passes left (the reference record's count) at its ms per pass so far,
# the loans so far (the own lines the following streams lent to the cache, which the ETA
# leaves out), the errors so far, and on completion the final frame hash against the
# reference and the comparison's result.
#
#   tools/stock-llama/proof-status.sh [work dir]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-${PROOF_WORK:-$HERE/.build-proof}}
LOG=$W/proof.log
REF=${PROOF_REFERENCE:-$HERE/.build-proof/rec-engine-300}
WANT=cfd31f3d
FRAMES=${FRAMES:-300}
[ -f "$LOG" ] || { echo "no log at $LOG"; exit 1; }
pid=$(cat "$W/proof.pid" 2>/dev/null)
chunks=$(grep -c '^chunk [0-9]* waits' "$LOG")
chunk_at=$(grep -nE '^chunk [0-9]* waits' "$LOG" | tail -1 | cut -d: -f1)
this() { tail -n +"${chunk_at:-1}" "$LOG"; }
if [ -f "$W/proof.exit" ]; then
    state="finished, exit $(cat "$W/proof.exit")"
elif [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    if ! this | grep -q '^lock taken'; then
        state="chunk $chunks waits for the GPU lock (pid $pid)"
    elif this | grep -q '^checkpoint found' && ! this | grep -q '^resumed:'; then
        state="chunk $chunks runs the host again to its checkpoint, with no model (pid $pid)"
    else
        state="chunk $chunks running (pid $pid)"
    fi
elif [ -f "$W/proof.paused" ]; then
    state="PAUSED at a checkpoint after chunk $(cat "$W/proof.paused"); proof.sh with the same settings continues"
else
    state="NOT RUNNING and no exit recorded (pid ${pid:-?} gone); proof.sh with the same settings continues from the last checkpoint"
fi
echo "state: $state"
echo "log: $LOG"
# Which claim the run makes (proof.sh's PROOF_PURE), and which llama.cpp the harness found.
grep -E '^mode: ' "$LOG" | tail -1
grep -E '^llama\.cpp (unpatched|patched|PARTLY)' "$LOG" | tail -1

# Decoded passes per stream in the reference, counted once.
counts=$W/reference-passes.txt
if [ ! -s "$counts" ] && [ -d "$REF" ]; then
    for f in "$REF"/*.jsonl; do
        # A stream is named by its first two slots: the PPU's lines and its stepper both start with LY.
        name=$(head -1 "$f" | jq -r '.slots[0:2] | join(",")')
        echo "$name $(grep -c '"kind":"fill"' "$f")"
    done | sed -E 's/^A,F /Cpu /; s/^LY,WLN /Ppu /; s/^S0,S1 /Apu /; s/^C0,C1 /Control /; s/^MLY,MFLG /Length /; s/^LY,FLG /Stepper /' > "$counts"
fi

# The time every chunk held the GPU lock, the open chunk up to now: the prefill, each
# resume's run of the host to its checkpoint and the checkpoints included.
end=$(this | sed -nE 's/^play exited .* unix ([0-9]+)$/\1/p' | tail -1)
now=${end:-$(date +%s)}
e=$(awk -v now="$now" '
    /^lock taken / { match($0, /unix [0-9]+/); start = substr($0, RSTART + 5, RLENGTH - 5) + 0 }
    /^play exited / { match($0, /unix [0-9]+/); total += substr($0, RSTART + 5, RLENGTH - 5) - start; start = 0 }
    END { if (start) total += now - start; print total + 0 }' "$LOG")
printf 'chunks: %d; elapsed: %dh %02dm %02ds with the GPU lock over all of them (the prefill, each resume and the checkpoints included)\n' "$chunks" $((e / 3600)) $((e % 3600 / 60)) $((e % 60))
# The live ledger: every event is decoded by llama.cpp in exactly the stretch of a chunk after it goes
# live. A chunk goes live at a checkpoint, and only a live chunk saves one, so the stretches join with no
# gap when each go-live event is a checkpoint an earlier chunk logged.
awk '
    /^checkpoint: event [0-9]+,/ { e = $3 + 0; saved[e] = 1; if (e > reach) reach = e }
    /^resumed: chunk [0-9]+ goes live after event [0-9]+/ {
        e = $8 + 0; n++
        if (!(e in saved)) { bad++; printf "LEDGER GAP: chunk %s goes live after event %d, which no earlier chunk logged as a checkpoint\n", $3, e }
        if (e < reach) redone += reach - e
    }
    END { printf "live ledger: %d resumes, each at a checkpoint a live chunk saved%s; live from event 0 to the log'"'"'s end with no gap%s; at least %d events decoded twice after a kill\n", n, bad ? " EXCEPT " bad : "", bad ? " NOT shown" : "", redone }' "$LOG"
grep -E '^resumed: ' "$LOG" | tail -1 | cut -c1-200
grep -E '^checkpoint: ' "$LOG" | tail -1
grep -E '^stopped at a checkpoint' "$LOG" | tail -1
grep -E '^prefill: ' "$LOG" | tail -1
frame=$(grep -E '^frame [0-9]+ sha1' "$LOG" | tail -1)
echo "last frame: ${frame:-none yet} (of $FRAMES)"
grep -E '^progress: ' "$LOG" | tail -1

# The last counts block: one line per stream.
for s in Cpu Ppu Apu Control Length Stepper; do grep -E "^$s: [0-9]+ passes" "$LOG" | tail -1; done > "$W/.status-counts"
if [ -s "$W/.status-counts" ]; then
    awk -v counts="$counts" -v now="$now" -v ended="$([ -f "$W/proof.exit" ] && echo 1)" '
        BEGIN { while ((getline l < counts) > 0) { split(l, a, " "); total[a[1]] = a[2] } }
        {
            name = $1; sub(":", "", name)
            match($0, /[0-9]+ passes,/); done = substr($0, RSTART, RLENGTH) + 0
            match($0, /[0-9.]+ ms per pass/); ms = substr($0, RSTART, RLENGTH) + 0
            match($0, /[0-9]+ passes equal the reference/); eq = substr($0, RSTART, RLENGTH) + 0
            lp = 0; ll = 0; ls = 0
            if (match($0, /[0-9]+ passes with [0-9]+ own lines lent to the cache(, [0-9.]+ s)?/)) { split(substr($0, RSTART, RLENGTH), w, " "); lp = w[1]; ll = w[4]; ls = w[11] }
            lent[name] = sprintf("%d passes lent %d own lines, %.1f s in those loans", lp, ll, ls); loans += ls
            left = total[name] - done; if (left < 0) left = 0
            rem += left * ms / 1000; sum += done; all += total[name]; equal += eq
            printf "  %-8s %9d of %9d passes, %6.1f ms per pass, %9d equal the reference\n", name, done, total[name], ms, eq
        }
        END {
            printf "passes: %d of %d (%.1f%%), %d equal the reference\n", sum, all, all ? 100 * sum / all : 0, equal
            for (n in lent) printf "  %-8s %s\n", n, lent[n]
            printf "loans: %.1f s so far, outside the ms per pass of each stream and so outside the ETA\n", loans
            if (ended == "") printf "remaining %d\n", rem
        }' "$W/.status-counts" > "$W/.status-out"
    grep -v "^remaining " "$W/.status-out"
    rem=$(sed -nE "s/^remaining ([0-9]+)/\1/p" "$W/.status-out")
    if [ -n "$rem" ]; then
        printf "ETA: %.1f hours of passes left at each stream's ms per pass so far" "$(echo "$rem / 3600" | bc -l)"
        # A date only while a chunk runs: a paused run finishes that long after it continues.
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then printf ", finishing about %s if this chunk runs to the end\n" "$(date -r $((now + rem)) "+%Y-%m-%d %H:%M %Z")"; else printf "\n"; fi
    fi
    rm -f "$W/.status-out"
    grep -E '^cache: ' "$LOG" | tail -1 | sed -nE 's/.*(swaps: [0-9]+).*(swap_checks: .*) \}/cells put back: \1, \2/p'
fi
rm -f "$W/.status-counts"

ERR='REFERENCE DIFFERS|REFERENCE ENDS|ROW HASH DIFFERS|STATE CELL DIFFERS|MIRROR DIFFERS|GROUP ROW DIFFERS|SLICE ROW DIFFERS|SLICE NOT RUN|SLICE UNKNOWN|MASK ROW DIFFERS|WRITE CELL DIFFERS|SWAP CELL DIFFERS|SWAP UNCHECKED|SWAP NOT HELD|READ NOT HELD|SCOPE MISSES LINE|NARROW SCOPE NOT HELD|WINDOWS SCOPE NOT HELD|SCOPE SEQUENCE DIFFERS|CHECK NOT HELD|READ ROW DIFFERS|READ DIFFERS|MAP SLOT NOT A BYTE|RECORD DIFFERS|RESUME DIFFERS|LEDGER GAP|panicked|ERROR LINES'
errors=$(grep -cE "$ERR" "$LOG")
echo "errors so far: $errors"
[ "$errors" = 0 ] || grep -E "$ERR" "$LOG" | head -5
# Chunks that could not start or save: no check failed in them (proof.sh, OPS).
OPS='RESUME REFUSED|RESUME FAILED|CHECKPOINT NOT SAVED'
ops=$(grep -cE "$OPS" "$LOG")
[ "$ops" = 0 ] || { echo "chunks refused or checkpoints not saved, which fail no check: $ops"; grep -E "$OPS" "$LOG" | tail -2 | cut -c1-200; }

if [ -f "$W/proof.exit" ]; then
    last=$(grep -E "^frame $FRAMES sha1" "$LOG" | tail -1)
    hash=$(echo "$last" | awk '{print $4}')
    want=$(sed -nE "s/^frame $FRAMES sha1 ([0-9a-f]+) .*/\1/p" "$(dirname "$REF")/record-engine.log" 2>/dev/null | tail -1)
    echo "final frame $FRAMES hash: ${hash:-none}"
    # tools/gate.sh's hash is frame 300's; a shorter run has only the engine's record to equal.
    [ "$FRAMES" = 300 ] || WANT=${want:0:8}
    echo "the engine's record of the same run: ${want:-not found, no record-engine.log beside the reference}; tools/gate.sh's one-model Yellow 300: cfd31f3d…"
    if [ -n "$hash" ] && [ -n "$WANT" ] && [ "${hash:0:8}" = "$WANT" ] && { [ -z "$want" ] || [ "$hash" = "$want" ]; }; then
        echo "frame hash: EQUAL"
    else
        echo "frame hash: DIFFERS or missing"
    fi
    grep -E '^record |^RECORD DIFFERS' "$LOG"
    if [ "$(cat "$W/proof.exit")" = 0 ] && [ "$errors" = 0 ] && [ -n "$WANT" ] && [ "${hash:0:8}" = "$WANT" ]; then
        echo "PROOF: every decoded pass equal the engine's record over $chunks chunks, and frame $FRAMES's hash equals $WANT"
    else
        echo "PROOF: NOT shown"
    fi
fi
