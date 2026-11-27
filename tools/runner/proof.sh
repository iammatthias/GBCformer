#!/bin/zsh
# proof.sh <model.gguf> <out dir> <rom> <frames> [full program flags]: the full program with
# every component in weights as one model, from the GGUF alone, and the runner from the
# full program's hand-off snapshot (with --warmup, a snapshot mid-run), compared pass record
# by pass record (every stream, sha1 and line count), frame by frame, sample by sample
# and save by save. `--boot FILE` among the flags starts both at power-on, the runner with
# no snapshot. Prints one `ok` or `FAIL` line per check; exits 1 on any failure.
#
# FULL (default ./target/release/gbcformer, built with --features stock) and RUNNER
# (default tools/runner/target/release/gbcformer-runner, built with --features record)
# name the binaries; THREADS=0 keeps the runner on one thread. Every stream must write a
# record and every pass must be fast.
#
# PLAIN=1 runs the runner with no GBCFORMER_RECORD (RUNNER may then be a build without the
# record feature) and compares the outputs alone. A recording run places the whole sequence
# after every write record; a plain run places it by its change list, which a build with
# debug assertions holds to the whole sequence at every pass. SCAN=1 sets GBCFORMER_SCAN=1 for
# the runner, so every lookup of a memory cell is also read through the softmax over the
# stored K/V rows (with threads, the rows the worker wrote), and requires the count the
# runner prints for its first stream to be above 0 and equal to its lookups.
set -u
cd "${0:A:h}/../.."
model=$1 out=$2 rom=$3 frames=$4; shift 4
FULL=${FULL:-./target/release/gbcformer}
RUNNER=${RUNNER:-tools/runner/target/release/gbcformer-runner}
# STREAMS: the streams that must record; the stepper records only where a line is marked.
STREAMS=${STREAMS:-5}
rc=0
fail() { print -r -- "FAIL  $*"; rc=1; }
pass() { print -r -- "ok    $*"; }
boot=() presses=() warm=0
for ((i = 1; i <= $#; i++)); do
  case ${@[i]} in
    --boot) boot=(--boot ${@[i+1]}) ;;
    --press) presses+=(--press ${@[i+1]}) ;;
    --warmup) warm=${@[i+1]} ;;
  esac
done
# No raw record is kept: each stream writes into a FIFO that tools/runner/digest.ts reduces to
# `sha1 lines name` (the comparison) and a CRC-32 per line (to locate a difference). `listen`
# opens them before a run; `settle` closes those a run never opened, digests any record a
# stream wrote under another name, and writes <dir>.sha1 sorted by name.
NAMES=(stream-0-A stream-1-LY stream-2-S0 stream-3-C0 stream-4-MLY stream-5-LY)
typeset -a readers
listen() {
  local n
  readers=()
  for n in $NAMES; do
    mkfifo $1/$n.jsonl
    { cat $1/$n.jsonl | bun tools/runner/digest.ts $1/$n $n.jsonl } &
    readers+=($!)
  done
}
settle() {
  local n f fd
  for n in $NAMES; do exec {fd}<>$1/$n.jsonl; exec {fd}>&-; done
  wait $readers
  for f in $1/stream-*.jsonl(N.); do bun tools/runner/digest.ts ${f:r} ${f:t} < $f; rm $f; done
  rm -f $1/stream-*.jsonl(Np)
  cat $1/*.sha1(N) | sort -k3 > $1.sha1
}
# The streams whose record holds a line.
recorded() { awk '$2 > 0' $1.sha1 | wc -l | tr -d ' ' }
# The first record line two sides' CRC lists differ at, per stream whose sha1s differ.
locate() {
  local s n a b at
  while read s n; do
    a=$full/rec/${n:r}.lines b=$run/rec/${n:r}.lines
    if [[ -f $a && -f $b ]]; then
      at=$(cmp $a $b 2>&1 | grep -o 'char [0-9]*' | cut -d' ' -f2)
      [[ -n $at ]] && print -r -- "      $n: first differing record line $(( (at - 1) / 4 + 1 ))" || print -r -- "      $n: one record a prefix of the other"
    else
      print -r -- "      $n: no line list on the full side (reused); rerun without REUSE to locate"
    fi
  done < <(diff $full/rec.sha1 $run/rec.sha1 | awk '/^[<>]/ {print $2, $4}' | sort -u -k2,2)
}
full=$out/full run=$out/runner
# REUSE=1 keeps the full program's side from an earlier run into the same directory: valid
# only while its binary and the model are byte for byte the ones that ran.
if [[ ${REUSE:-0} == 1 && -f $full/rec.sha1 && -d $full/out ]]; then
  print -r -- "      the full program's records, outputs and snapshot reused"
else
  # HALTS=1 needs the full program's count of passes at a HALT, its wake or a step's end, a stats line.
  (( ${HALTS:-0} )) && export GBCFORMER_STATS=1
  rm -rf $full; mkdir -p $full/rec; listen $full/rec
  GBCFORMER_GGUF_ONLY=$model GBCFORMER_APU_THREAD=0 GBCFORMER_PPU_THREAD=0 GBCFORMER_TIME=1700000000 \
    GBCFORMER_SNAPSHOT=$full/snap GBCFORMER_RECORD=$full/rec GBCFORMER_OUTPUTS=$full/out \
    $FULL run $rom --frames $frames --weights all --one-model --no-save "$@" > $full.log 2>&1 || fail "full program exited $? ($(tail -1 $full.log))"
  settle $full/rec
fi
nf=$(recorded $full/rec)
rm -rf $run; mkdir -p $run/rec
plain=${PLAIN:-0} scan=${SCAN:-0}
renv=()
if (( plain )); then nr=$nf; else listen $run/rec; renv+=(GBCFORMER_RECORD=$run/rec); fi
(( scan )) && renv+=(GBCFORMER_SCAN=1)
if (( ${#boot} )); then start=($boot); else start=(--snapshot $full/snap); fi
# The scripted clock reads a second every 60 frames from frame 0; the warmup's frames count.
env $renv $RUNNER $model --rom $rom $start --frames $frames --time 1700000000 --outputs $run/out --threads ${THREADS:-1} $presses > $run.log 2>&1 || fail "runner exited $? ($(tail -1 $run.log))"
if (( plain )); then
  [[ -z $(ls $run/rec) ]] && pass "the runner wrote no pass record" || fail "the runner wrote a pass record in a plain run"
else
  settle $run/rec
  nr=$(recorded $run/rec)
  [[ $nf -ge $STREAMS && $nr == $nf ]] && pass "$nf streams recorded on both sides" || fail "records written: full $nf, runner $nr, want $STREAMS or more, equal"
fi
hf=$(grep -o 'frame_sha1=........' $full.log) hr=$(grep -o 'frame_sha1=........' $run.log)
[[ -n $hf && $hf == $hr ]] && pass "last frame $hf" || fail "last frame: full ${hf:-none}, runner ${hr:-none}"
if (( plain )); then :; elif diff -q $full/rec.sha1 $run/rec.sha1 > /dev/null; then pass "pass records equal: $(cut -d' ' -f2 $full/rec.sha1 | tr '\n' ' ')"; else fail "pass records differ"; paste $full/rec.sha1 $run/rec.sha1; locate; fi
for f in frames.txt audio.raw $(ls $full/out $run/out 2>/dev/null | grep '^save-' | sort -u); do
  cmp -s $full/out/$f $run/out/$f && pass "equal $f ($(wc -c < $run/out/$f | tr -d ' ') bytes)" || fail "$f differs"
done
# A frame's hash is one line: as many lines as frames run, on both sides.
lf=$(wc -l < $full/out/frames.txt | tr -d ' ') lr=$(wc -l < $run/out/frames.txt | tr -d ' ')
[[ $lr == $((frames - warm)) && $lf == $lr ]] && pass "$lr frames compared, $(ls $run/out | grep -c '^save-') saves" || fail "frames written: full $lf, runner $lr, want $((frames - warm))"
if (( scan )); then
  line=$(grep '^steps: ' $run.log)
  by=$(print -r -- $line | grep -o '[0-9]* by lookup' | cut -d' ' -f1) checked=$(print -r -- $line | grep -o '[0-9]* lookups checked' | cut -d' ' -f1)
  [[ ${checked:-0} -gt 0 && $checked == $by ]] && pass "cell lookups read again through the softmax over the stored rows: $checked of $by" || fail "lookups checked by the scan: ${checked:-none} of ${by:-none}, want all and above 0"
fi
# REWRITES=1: a DMG case whose OAM corruption bug must fire, as many rewrite passes on both
# sides (the full program's `oam_bugs`, the runner's `rewrites`).
if (( ${REWRITES:-0} )); then
  bf=$(grep -o 'oam_bugs=[0-9]*' $full.log | cut -d= -f2) br=$(grep -o 'rewrites=[0-9]*' $run.log | cut -d= -f2)
  [[ ${bf:-0} -gt 0 && $bf == $br ]] && pass "OAM bug passes: full $bf, runner $br" || fail "OAM bug passes: full ${bf:-none}, runner ${br:-none}, want equal and above 0"
fi
# HALTS=1: a case whose HALTs run the spans stream's rest and wake passes, as many on both
# sides (the full program's "passes at a HALT, its wake or a step's end", the runner's `rest_passes`).
if (( ${HALTS:-0} )); then
  hf=$(grep -o '[0-9]* passes at a HALT, its wake or a step.s end' $full.log | cut -d" " -f1) hr=$(grep -o 'rest_passes=[0-9]*' $run.log | cut -d= -f2)
  [[ ${hf:-0} -gt 0 && $hf == $hr ]] && pass "rest and wake passes: full $hf, runner $hr" || fail "rest and wake passes: full ${hf:-none}, runner ${hr:-none}, want equal and above 0"
fi
# LOCKPAIRS=1: a DMG case whose accesses land where the reads' and the writes' lock pairs
# differ (the runner's `lock_pairs_differ`), so the access's kind picks its pair.
if (( ${LOCKPAIRS:-0} )); then
  lp=$(grep -o 'lock_pairs_differ=[0-9]*' $run.log | cut -d= -f2)
  [[ ${lp:-0} -gt 0 ]] && pass "accesses where the lock pairs differ: $lp" || fail "accesses where the lock pairs differ: ${lp:-none}, want above 0"
fi
# EQUAL="name ...": counters both programs print as name=N, each above 0 and equal on both
# sides. ABOVE="name ...": above 0 on both sides, where the two count one branch at different
# points (the runner closes a row at a COMPAT record lazily, at its next own-memory write).
for n in ${=EQUAL:-}; do
  a=$(grep -o "$n=[0-9]*" $full.log | tail -1 | cut -d= -f2) b=$(grep -o "$n=[0-9]*" $run.log | tail -1 | cut -d= -f2)
  [[ ${a:-0} -gt 0 && $a == $b ]] && pass "$n: full $a, runner $b" || fail "$n: full ${a:-none}, runner ${b:-none}, want equal and above 0"
done
for n in ${=ABOVE:-}; do
  a=$(grep -o "$n=[0-9]*" $full.log | tail -1 | cut -d= -f2) b=$(grep -o "$n=[0-9]*" $run.log | tail -1 | cut -d= -f2)
  [[ ${a:-0} -gt 0 && ${b:-0} -gt 0 ]] && pass "$n: full $a, runner $b" || fail "$n: full ${a:-none}, runner ${b:-none}, want both above 0"
done
if faults=$(grep -hE '(^|[^0-9])[1-9][0-9]* (on the )?dense fallback|panicked|refused' $full.log $run.log); then fail "faults: ${faults[1,200]}"; else pass "no dense fallback, panic or refusal"; fi
if (( warm )); then
  over=$(jq .overshoot $full/snap/snapshot.json 2>/dev/null)
  # OVERSHOOT=1 requires a hand-off past the frame's end, so the runner's clock offset is exercised.
  if (( ${OVERSHOOT:-0} && ${over:-0} == 0 )); then fail "the snapshot's overshoot is 0: the case exercises no clock offset"; else pass "warmup $warm, snapshot overshoot $over"; fi
fi
exit $rc
