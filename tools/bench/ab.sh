#!/bin/zsh
# Interleaved A/B of two binaries on text.sh and overworld.sh: A B A B A B per workload,
# reporting every run's emulated fps and µs per pass, the medians and the load average.
#
#   tools/bench/ab.sh <baseline binary> <candidate binary> [runs=3]
set -eu
cd "${0:A:h}/../.."
a=$1 b=$2 runs=${3:-3}

one() {
  # fps from "N emulated fps", µs from "N µs per pass", frame hash from stdout.
  local bin=$1 bench=$2 out
  out=$(GBCFORMER_STATS=1 BIN=$bin tools/bench/$bench.sh 2>&1)
  local fps=$(print -r -- "$out" | sed -n 's/^\([0-9.]*\) emulated fps.*/\1/p')
  local us=$(print -r -- "$out" | sed -n 's/.*; \([0-9.]*\) µs per pass.*/\1/p')
  local hash=$(print -r -- "$out" | sed -n 's/.*frame_sha1=\([0-9a-f]\{8\}\).*/\1/p')
  print -r -- "$fps $us $hash"
}

median() { print -l -- "$@" | sort -n | sed -n "$(( ($# + 1) / 2 ))p"; }

print -r -- "load before: $(uptime | sed 's/.*load averages: //')"
for bench in text overworld; do
  typeset -a fa fb ua ub
  fa=() fb=() ua=() ub=()
  for i in $(seq $runs); do
    for side in A B; do
      bin=$a; [[ $side == B ]] && bin=$b
      read fps us hash <<< "$(one $bin $bench)"
      print -r -- "$bench $side run $i: $fps fps, $us µs/pass, frame $hash"
      if [[ $side == A ]]; then fa+=($fps) ua+=($us); else fb+=($fps) ub+=($us); fi
    done
  done
  print -r -- "$bench median: A $(median $fa) fps / $(median $ua) µs, B $(median $fb) fps / $(median $ub) µs"
done
print -r -- "load after: $(uptime | sed 's/.*load averages: //')"
