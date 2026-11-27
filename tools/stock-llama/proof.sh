#!/bin/bash
# The end-to-end proof (docs/slice-stack.md, "The end-to-end run"): Pokémon Yellow (CGB)
# from the hand-off state (no boot ROM) to frame FRAMES (default 300) with every component a stream of the one
# checkpoint and every pass decoded by llama.cpp, the cartridge prefill included. Every
# decoded pass is held, as it decodes, to the engine's record of the same run
# (`STOCK_REFERENCE`), and the host references run beside it (`--lockstep all`); the
# first difference stops the run, and so does the first failed check of the cache: a
# mirror that differs from a stream's image, or a cell put back after a loan that differs
# from the cell found or is never compared. The script exits 0 only when play did and the
# log holds no error line. The GPU lock is held for a whole chunk and released on any exit. Each pass runs its stream's slice of the model, which needs llama.cpp built
# with stream-slices.patch (`run.sh setup`); the run refuses a build without it unless
# STOCK_STREAMS says otherwise (0 runs the whole model for every pass, at about twice the hours).
#
# PROOF_PURE=1 is the run of the claim of record (docs/stock-llama.md, "What runs where" and §13; the
# proof of record built llama.cpp with its own default options): llama.cpp b12070 with no
# patch, the uniform file (`export one-model <dir> --gguf-only`) and the whole model every
# pass. The harness then refuses a patched or partly patched llama.cpp and a file written
# for one (STOCK_PURE=1), and PROOF_BIN must be a harness built against the unpatched tree.
#
#   tools/stock-llama/proof.sh            # detach with nohup; tools/stock-llama/proof-status.sh reads the log
#
# PROOF_WORK (default .build-proof) holds the log, the run's record and its prefilled
# cache. PROOF_BIN is the harness (`run.sh <work dir> <rom> setup` builds target-metal),
# PROOF_MODEL the one model's GGUF (`run.sh .build <rom> export` writes
# .build/one-model), and PROOF_REFERENCE the engine's record (`GBCFORMER_RECORD=<dir>
# <bin> play none <rom> --frames 300`). PROOF_STATE loads a saved prefilled cache
# instead of prefilling.
#
# The run goes in chunks (docs/stock-llama.md §14). play saves a checkpoint to
# PROOF_WORK/checkpoints every PROOF_CHECKPOINT_SECS seconds (default 300): the cache and
# what the harness held. Running the same command again continues from the newest one:
# the host runs again from frame 0 with every pass answered from the engine's record and
# no model, must hold what the checkpoint holds, and llama.cpp decodes from there. A chunk
# ends at a checkpoint, with nothing lost, when PROOF_WORK/stop exists (touch it), when
# this script gets TERM or INT, or after PROOF_CHUNK_SECS seconds; a chunk that is killed
# loses what it decoded since its last checkpoint. PROOF_FRESH=1 drops the checkpoints and
# starts at frame 0. The prefill has checkpoints of its own, between its lane batches,
# and a chunk continues it the same way.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${PROOF_WORK:-$HERE/.build-proof}
BIN=${PROOF_BIN:-$HERE/target-metal/release/stock-llama}
MODEL=${PROOF_MODEL:-$HERE/.build/one-model/model.gguf}
REF=${PROOF_REFERENCE:-$HERE/.build-proof/rec-engine-300}
ROM=${PROOF_ROM:-$HERE/../../pokemon-yellow.gbc}
FRAMES=${FRAMES:-300}
LOCK=/tmp/gbcformer-gpu.lock
LOG=$W/proof.log
CKPT=$W/checkpoints
ERR='REFERENCE DIFFERS|REFERENCE ENDS|ROW HASH DIFFERS|STATE CELL DIFFERS|MIRROR DIFFERS|GROUP ROW DIFFERS|SLICE ROW DIFFERS|SLICE NOT RUN|SLICE UNKNOWN|MASK ROW DIFFERS|WRITE CELL DIFFERS|SWAP CELL DIFFERS|SWAP UNCHECKED|SWAP NOT HELD|READ NOT HELD|SCOPE MISSES LINE|NARROW SCOPE NOT HELD|WINDOWS SCOPE NOT HELD|SCOPE SEQUENCE DIFFERS|CHECK NOT HELD|READ ROW DIFFERS|READ DIFFERS|MAP SLOT NOT A BYTE|RECORD DIFFERS|RESUME DIFFERS|panicked'
# Lines that say a chunk could not start or save, which fail no check: a checkpoint of
# another build or setting is refused, a state does not load, a checkpoint is not written.
# The chunk ends or goes on, the proof is neither failed nor finished, and the right
# command continues from the last checkpoint.
OPS='RESUME REFUSED|RESUME FAILED|CHECKPOINT NOT SAVED'
mkdir -p "$W"
if [ "${PROOF_FRESH:-0}" = 1 ]; then
    rm -rf "$CKPT" "$LOG" "$W/proof.exit" "$W/proof.paused" "$W/reference-passes.txt"
fi
if [ -f "$W/proof.exit" ]; then
    echo "the proof in $W finished with exit $(cat "$W/proof.exit"); PROOF_FRESH=1 starts another" >&2
    exit 1
fi
# One chunk at a time in a work directory.
old=$(cat "$W/proof.pid" 2>/dev/null)
if [ -n "$old" ] && [ "$old" != $$ ] && kill -0 "$old" 2>/dev/null; then
    echo "a chunk of this proof is running (pid $old)" >&2
    exit 1
fi
echo $$ > "$W/proof.pid"
rm -f "$W/proof.paused" "$W/stop"
if ls "$CKPT"/ckpt-*.json >/dev/null 2>&1 || [ -f "$CKPT/prefill.json" ]; then
    resumed=1
else
    resumed=0
    : > "$LOG"
fi
touch "$LOG"
chunk=$(($(grep -c '^chunk [0-9]* waits' "$LOG") + 1))
echo "chunk $chunk waits for the GPU lock at $(date)" >> "$LOG"
child=
# A signal ends the chunk at its next checkpoint: play takes it and stops there.
trap 'if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; else exit 130; fi' INT TERM
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
if [ -n "${PROOF_STATE:-}" ]; then
    STATE=$PROOF_STATE
else
    STATE=$W/yellow-cache.state
    [ "$resumed" = 1 ] || rm -f "$STATE" "$STATE.cells"
fi
# The engines write the run's record from frame 0 in every chunk; a resumed chunk's
# rewritten lines are held to the digest its checkpoint keeps of the lines before it.
mkdir -p "$W/rec-run"
[ "$resumed" = 1 ] || { rm -rf "$W/rec-run" && mkdir -p "$W/rec-run"; }
[ -n "${PROOF_CHUNK_SECS:-}" ] && export STOCK_CHUNK_SECS=$PROOF_CHUNK_SECS
from=$(($(wc -l < "$LOG") + 1))
echo "lock taken at $(date), unix $(date +%s); load $(uptime | sed 's/.*load averages*: //')" >> "$LOG"
if [ "${PROOF_PURE:-0}" = 1 ]; then
    STREAMS=0
    PURE=1
    echo "mode: PURE: llama.cpp unpatched, the uniform file, the whole model every pass" >> "$LOG"
else
    STREAMS=${STOCK_STREAMS:-require}
    PURE=0
    echo "mode: patched llama.cpp, the trimmed file, STOCK_STREAMS=$STREAMS" >> "$LOG"
fi
echo "command: STOCK_GPU_LAYERS=99 STOCK_GROUP=own STOCK_PURE=$PURE STOCK_STREAMS=$STREAMS STOCK_STATE=$STATE STOCK_REFERENCE=$REF GBCFORMER_RECORD=$W/rec-run STOCK_CHECKPOINT=$CKPT STOCK_CHECKPOINT_SECS=${PROOF_CHECKPOINT_SECS:-300} STOCK_STOP=$W/stop STOCK_CHUNK_SECS=${PROOF_CHUNK_SECS:-} $BIN play $MODEL $ROM --frames $FRAMES --lanes 255" >> "$LOG"
STOCK_GPU_LAYERS=99 STOCK_GROUP=own STOCK_PURE=$PURE STOCK_STREAMS=$STREAMS STOCK_STATE=$STATE STOCK_REFERENCE=$REF GBCFORMER_RECORD=$W/rec-run \
    STOCK_CHECKPOINT=$CKPT STOCK_CHECKPOINT_SECS=${PROOF_CHECKPOINT_SECS:-300} STOCK_STOP=$W/stop \
    "$BIN" play "$MODEL" "$ROM" --frames "$FRAMES" --lanes 255 >> "$LOG" 2>&1 &
child=$!
# `wait` returns early when the trap runs; the chunk ends when play has.
wait "$child"
code=$?
while kill -0 "$child" 2>/dev/null; do
    wait "$child"
    code=$?
done
child=
echo "play exited $code at $(date), unix $(date +%s)" >> "$LOG"
rmdir "$LOCK" 2>/dev/null
errors=$(grep -cE "$ERR" "$LOG")
if [ "$code" = 0 ] && [ "$errors" = 0 ] && tail -n +"$from" "$LOG" | grep -q '^stopped at a checkpoint'; then
    echo "chunk $chunk paused at a checkpoint; the same command continues" >> "$LOG"
    # The chunk's last checkpoint is kept apart (hard links), so a state that later fails to
    # load costs the chunks after it and not the run: copy a pair back into checkpoints/.
    last=$(ls "$CKPT"/ckpt-*.json 2>/dev/null | tail -1)
    if [ -n "$last" ]; then
        mkdir -p "$W/archive"
        for f in "$last" "${last%.json}.state"; do
            [ -f "$f" ] && ln -f "$f" "$W/archive/" 2>/dev/null
        done
    fi
    echo "$chunk" > "$W/proof.paused"
    exit 0
fi
if [ "$code" != 0 ] && [ "$errors" = 0 ]; then
    # Killed, crashed or refused with no failed check: nothing is concluded, and the last
    # checkpoint stands.
    why=$(tail -n +"$from" "$LOG" | grep -E "$OPS" | head -1 | cut -c1-120)
    echo "chunk $chunk ended with exit $code and no failed check${why:+ ($why)}; the right command continues from the last checkpoint" >> "$LOG"
    exit "$code"
fi
# An error line fails the proof whatever play's exit code was.
if [ "$code" = 0 ] && [ "$errors" != 0 ]; then
    echo "ERROR LINES in the log of a play that exited 0" >> "$LOG"
    code=4
fi
# The write records (`keep`), which the runtime does not decode, held to the reference
# too, after the lock is released: kind, sequence, positions and tokens, stream by stream.
if [ "$code" = 0 ]; then
    for f in "$W"/rec-run/*.jsonl; do
        # The reference's record of the same stream: the one with the same first line.
        r=
        for c in "$REF"/*.jsonl; do
            if [ "$(head -1 "$c")" = "$(head -1 "$f")" ]; then r=$c; break; fi
        done
        if [ -z "$r" ]; then
            echo "RECORD DIFFERS: $(basename "$f") names a stream the reference lacks" >> "$LOG"
            code=3
            continue
        fi
        n=$(tail -n +2 "$f" | wc -l | tr -d ' ')
        if tail -n +2 "$f" | jq -c '[.kind,.tokens,.positions,.out]' | cmp -s - <(tail -n +2 "$r" | awk -v n="$n" 'NR > n { exit } { print }' | jq -c '[.kind,.tokens,.positions,.out]'); then
            echo "record $(basename "$f"): $n events equal the reference's first $n (it holds $(($(wc -l < "$r") - 1)))" >> "$LOG"
        else
            echo "RECORD DIFFERS: $(basename "$f") against $(basename "$r")" >> "$LOG"
            code=3
        fi
    done
fi
echo "proof done, exit $code" >> "$LOG"
echo "$code" > "$W/proof.exit"
