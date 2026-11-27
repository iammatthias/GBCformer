#!/bin/zsh
# bench.sh <model.gguf> <out dir>: the scenes of the three benchmark workloads
# (tools/bench/text.sh, overworld.sh, battle.sh) with every frame from the model. The
# bench scripts run their first frames on the reference interpreter; here the minimal
# runner plays each press script from frame 0, from the full program's hand-off snapshot
# (the labelled post-boot state, written with no frame run), and the last frame's hash
# must be the workload's. <out dir>/<scene>/ holds the runner's outputs, <out dir>/pixels
# the scene's 600 frames.
cd ${0:A:h}/../..
G=$1 out=$2
mkdir -p $out
GBCFORMER_GGUF_ONLY=$G GBCFORMER_SNAPSHOT=$out/snap nice -n 19 ./target/release/gbcformer run pokemon-yellow.gbc --frames 1 --weights all --one-model --no-save > $out/snapshot.log 2>&1 || { print "FAIL  snapshot"; exit 1 }
{ for f in $(seq 2300 40 3500); do print "${f}:a:4\n$((f+20)):down:4"; done } > $out/text.presses
{ for f in $(seq 2300 40 7000); do print "${f}:a:4\n$((f+20)):down:4"; done
  for f in $(seq 7200 64 7800); do print "${f}:left:24\n$((f+32)):right:24"; done } > $out/overworld.presses
{ for f in $(seq 2300 40 5660); do print "${f}:a:4\n$((f+20)):down:4"; done
  print "5720:right:64\n5800:up:100\n5960:right:40\n6080:down:100\n6190:left:80\n6280:down:20\n6380:right:84\n6470:up:120"
  for f in $(seq 6880 40 11160); do print "${f}:a:4"; done
  print "11180:down:12\n11210:right:28\n11250:up:4"
  for f in $(seq 11280 40 13400); do print "${f}:a:4"; done
  for f in $(seq 13440 40 13520); do print "${f}:b:4"; done
  print "13560:down:100"
  for f in $(seq 13700 40 15480); do print "${f}:a:4"; done } > $out/battle.presses
rc=0
scene() { # name, frames, expected hash
  nice -n 19 tools/runner/target/release/gbcformer-runner $G --rom pokemon-yellow.gbc --snapshot $out/snap --frames $2 --presses $out/$1.presses --time 1700000000 --outputs $out/$1 --pixels $out/pixels --clip $(($2 - 600)):$2 > $out/$1.log 2>&1
  local sha=$(grep -o 'frame_sha1=[0-9a-f]*' $out/$1.log | cut -d= -f2)
  if [[ $sha == $3* ]] && ! grep -qE '(^|[^0-9])[1-9][0-9]* dense fallback|panicked|refused' $out/$1.log; then
    print "ok    $1: $2 frames from frame 0 in the runner, last frame $sha, $(grep -o '[0-9.]* fps' $out/$1.log)"
  else
    print "FAIL  $1: last frame ${sha:-none}, want $3"; rc=1
  fi
}
scene text 3500 870aa3fd
scene overworld 7800 822059c1
# docs/perf-log.md records d3534543 for this workload; CLAUDE.md still says 14d7468e.
scene battle 15480 d3534543
exit $rc
