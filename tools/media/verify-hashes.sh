#!/bin/zsh
# verify-hashes.sh <frames.txt> [demo dir]: a run's frame hashes (the runner's
# `--outputs DIR` writes DIR/frames.txt, line N+1 the sha1 of frame N) held to the demo's.
# The demo keeps the hash of every 1,000th frame and of the last one
# (frame-hashes-every-1000.txt, FRAME HASH lines) and the count and sha1 of the whole list
# (frame-hashes.sha1). A run of the whole log must match both. A shorter run is held to
# the kept hashes it reaches. Exits 1 at any difference.
set -eu
list=$1 demo=${2:-${0:A:h}/demo}
read -r want_n want_sha < $demo/frame-hashes.sha1
n=$(wc -l < $list | tr -d ' ')
awk 'NR == FNR { want[$1 + 1] = $2; next }
  FNR in want { seen++; if ($1 != want[FNR]) { bad++; if (bad == 1) first = FNR - 1 } }
  END { if (bad) { printf "FAIL  %d of %d kept hashes differ, the first at frame %d\n", bad, seen, first; exit 1 }
        printf "ok    %d kept hashes equal\n", seen }' $demo/frame-hashes-every-1000.txt $list
if (( n > want_n )); then
  print "FAIL  $n frames, the demo has $want_n"; exit 1
elif (( n < want_n )); then
  print "ok    $n of the demo's $want_n frames: the whole list's sha1 is not checked on a shorter run"
else
  sha=$(shasum $list | cut -c1-40)
  [[ $sha == $want_sha ]] || { print "FAIL  the list's sha1 is $sha, want $want_sha"; exit 1 }
  print "ok    $n frames, the list's sha1 $sha"
fi
