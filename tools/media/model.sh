#!/bin/zsh
# model.sh <model.gguf> <input log> <frames> <out dir> [FROM:TO ...]: the input log played
# from power-on by the minimal runner (the one model's GGUF, your ROM and boot ROM) and by
# the reference interpreter, every frame's hash and every sample from both compared.
# ROM and BOOT name the files (default pokemon-yellow.gbc and cgb_boot.bin in the checkout).
# <out dir>/model holds the runner's outputs (frames.txt, audio.raw), <out dir>/pixels the
# raw frames of each FROM:TO range, <out dir>/runner.log its report, <out dir>/reference
# the interpreter's frames.txt. REFERENCE=0 leaves the interpreter out.
# Exits 1 at a difference, a dense fallback or a refusal.
cd ${0:A:h}/../..
G=$1 log=$2 n=$3 out=$4; shift 4
rom=${ROM:-pokemon-yellow.gbc} boot=${BOOT:-cgb_boot.bin}
clips=(); for r in "$@"; do clips+=(--clip $r); done
mkdir -p $out
nice -n 19 tools/runner/target/release/gbcformer-runner $G --rom $rom --boot $boot --frames $n --presses $log --time 1700000000 --outputs $out/model --pixels $out/pixels $clips > $out/runner.log 2>&1
rc=$?
tail -8 $out/runner.log
(( rc )) && { print "FAIL  runner exited $rc"; exit 1 }
lines=$(wc -l < $out/model/frames.txt | tr -d ' ')
[[ $lines == $n ]] || { print "FAIL  the runner wrote $lines of $n frames"; exit 1 }
if grep -E '(^|[^0-9])[1-9][0-9]* dense fallback|panicked|refused' $out/runner.log; then print "FAIL  fallback, panic or refusal"; exit 1; fi
print "ok    $n frames, no dense fallback, panic or refusal (list sha1 $(shasum $out/model/frames.txt | cut -c1-40))"
(( ${REFERENCE:-1} )) || exit 0
presses=(); while read -r p; do presses+=(--press $p); done < $log
GBCFORMER_OUTPUTS=$out/reference GBCFORMER_TIME=1700000000 nice -n 19 ./target/release/gbcformer run $rom --frames $n --core ref --hardware cgb --boot $boot --no-save $presses > $out/reference.log 2>&1 || { tail -3 $out/reference.log; print "FAIL  reference run"; exit 1 }
if ! cmp -s $out/reference/frames.txt $out/model/frames.txt; then
  print "FAIL  frames differ: first at frame $(( $(cmp $out/reference/frames.txt $out/model/frames.txt | grep -o 'line [0-9]*' | cut -d' ' -f2) - 1 ))"; exit 1
fi
print "ok    every frame's hash equal on the reference and the model"
if cmp -s $out/reference/audio.raw $out/model/audio.raw; then
  print "ok    every sample equal on the reference and the model ($(( $(stat -f %z $out/model/audio.raw) / 4 )) stereo samples)"
  rm -f $out/reference/audio.raw $out/reference/save-*.bin(N)
else
  print "FAIL  samples differ: $(cmp $out/reference/audio.raw $out/model/audio.raw)"; exit 1
fi
