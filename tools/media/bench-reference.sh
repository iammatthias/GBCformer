#!/bin/zsh
# bench-reference.sh <bench.sh's out dir>: each bench scene's press script on the reference
# interpreter alone, every frame's hash to <out dir>/<scene>-reference/frames.txt, and the
# comparison with the runner's frames.txt of the same scene.
cd ${0:A:h}/../..
out=$1 rc=0
for s in text:3500 overworld:7800 battle:15480; do
  name=${s%:*} n=${s#*:}
  P=(); while read -r p; do P+=(--press $p); done < $out/$name.presses
  GBCFORMER_OUTPUTS=$out/$name-reference GBCFORMER_TIME=1700000000 nice -n 19 ./target/release/gbcformer run pokemon-yellow.gbc --frames $n --core ref --no-save $P > $out/$name-reference.log 2>&1
  if [[ $(wc -l < $out/$name/frames.txt) -eq $n ]] && cmp -s $out/$name/frames.txt $out/$name-reference/frames.txt; then
    print "ok    $name: $n frames, every frame's hash equal on the reference and the runner"
  else
    print "FAIL  $name: frame hashes differ or are missing"; rc=1
  fi
done
exit $rc
