#!/bin/zsh
# story.sh <manifest.tsv> <work dir> <dest dir> <model.gguf>: the clips and stills a
# manifest names, cut from the frames model.sh or bench.sh wrote, each with a sidecar .json.
#
# A manifest line is tab-separated:
#   name  run  range  first  count  kind  shows  caption
# run is the directory under <work dir> (run2, bench/..), range the FROM-TO of its pixel
# file, first the first frame (as the run numbers frames) and count the frames. kind is
# `clip` (MP4 at 4x with the model's samples), `gif` (MP4 and a GIF at 2x), `readme`
# (a GIF at 2x alone), `readme50` (the same with every frame kept, 2 cs each), `still` (a PNG at 4x) or `still3` (a PNG at 3x).
set -eu
manifest=$1 W=$2 D=$3 G=$4
here=${0:A:h}
mkdir -p $D
# The checkout's commit, which the sidecar names.
commit=$(git -C $here rev-parse --short HEAD)
gguf=$(shasum $G | cut -c1-40)
digest=$(grep '^one-model ' $here/../digests.txt | cut -d' ' -f2)
print "| file | frames | seconds | bytes | what it shows | suggested caption |\n| - | - | - | - | - | - |" > $D/INDEX.md
while IFS=$'\t' read -r name run range first count kind shows caption; do
  [[ -z $name || $name == \#* ]] && continue
  dir=$W/$run
  raw=$dir/pixels/$range.bgr0
  [[ -f $raw ]] || raw=$dir/../pixels/$range.bgr0
  from=${range%-*}
  frames=$dir/model/frames.txt audio=$dir/model/audio.raw start=0
  [[ -f $frames ]] || { frames=$dir/frames.txt audio=$dir/audio.raw; start=$(jq .frame $dir/../snap/snapshot.json) }
  export FIRST=$((first - from)) COUNT=$count
  case $kind in
    clip) AUDIO=$audio BEFORE=$(cat ${raw:r}.samples) $here/clip.sh $raw $D/$name > /dev/null; files=($name.mp4) ;;
    gif) AUDIO=$audio BEFORE=$(cat ${raw:r}.samples) GIF=2 $here/clip.sh $raw $D/$name > /dev/null; files=($name.mp4 $name.gif) ;;
    readme) MP4=0 GIF=2 $here/clip.sh $raw $D/$name > /dev/null; files=($name.gif) ;;
    readme50) MP4=0 GIF=2 GIF_ALL=1 $here/clip.sh $raw $D/$name > /dev/null; files=($name.gif) ;;
    still) STILL=4 COUNT=1 $here/clip.sh $raw $D/$name > /dev/null; files=($name.png) ;;
    still3) STILL=3 COUNT=1 $here/clip.sh $raw $D/$name > /dev/null; files=($name.png) ;;
  esac
  [[ $kind == still* ]] && count=1
  hashes=$(sed -n "$((first - start + 1)),$((first - start + count))p" $frames)
  [[ $(print -r -- $hashes | wc -l) -eq $count ]] || { print "FAIL  $name: the run holds no hash for its frames"; exit 1 }
  if [[ $run == bench/* ]]; then
    start_state="the full program's hand-off snapshot at frame 0 (the labelled post-boot state, no boot ROM), every frame from 0 played by the runner"
    log=$dir/../${run:t}.presses
  else
    start_state="power-on, the state built by the runner from the ROM and a CGB boot ROM (sha1 $(shasum $here/../../cgb_boot.bin | cut -c1-40), a SameBoy build by its logo), every frame from 0 played by the runner"
    log=$dir/input.log
  fi
  jq -n --arg name $name --arg digest $digest --arg gguf $gguf --arg commit $commit --arg run $run --arg log_sha1 "$(shasum $log | cut -c1-40)" \
    --argjson first $first --argjson count $count --arg hashes_sha1 "$(print -r -- $hashes | shasum | cut -c1-40)" --arg last "$(print -r -- $hashes | tail -1)" \
    --arg start $start_state --arg shows $shows --arg caption $caption --arg kind $kind --arg files "$files" \
    --arg command "gbcformer-runner model.gguf --rom pokemon-yellow.gbc $([[ $run == bench/* ]] && print -- '--snapshot SNAPSHOT' || print -- '--boot cgb_boot.bin') --frames $(wc -l < $frames | tr -d ' ' | xargs -I{} expr {} + $start) --presses $(basename $log) --time 1700000000 --outputs OUT --pixels PIXELS --clip ${range/-/:}" \
    '{name: $name, files: ($files | split(" ")), game: "Pokémon Yellow (CGB mode)", weights_digest: $digest, gguf_sha1: $gguf, commit: $commit,
      program: "the minimal runner (tools/runner): the one model GGUF, no Game Boy hardware rules in the host", command: $command,
      start: $start, input_log: "a recorded input log, FRAME:BUTTON:HOLD lines", input_log_sha1: $log_sha1,
      first_frame: $first, frames: $count, seconds: ($count / 59.7275 * 100 | round / 100),
      frame_hashes_sha1: $hashes_sha1, last_frame_sha1: $last,
      checked: "every frame of the run has the hash the reference interpreter gives for the same input log",
      every_frame_decoded_by_the_model: "yes, via the runner", encoding: (if $kind == "clip" or $kind == "gif" then "MP4 H.264 yuv420p 59.7275 fps, nearest-neighbour 4x, AAC 128k from the model'"'"'s own samples" else "" end) + (if $kind == "gif" or $kind == "readme" then " GIF 2x, every second frame, delays 3,3,4 cs (29.86 fps, 0.5% fast)" else "" end) + (if $kind == "readme50" then "GIF 2x, every frame kept, 2 cs each (50 fps: plays 16% slower than the console'"'"'s 59.73 fps)" else "" end) + (if ($kind | startswith("still")) then "PNG, nearest-neighbour" else "" end),
      shows: $shows, caption: $caption}' > $D/$name.json
  for f in $files; do
    print "| $f | $count | $(( count * 100000 / 5972750 )).$(( count * 1000000 / 5972750 % 10 )) | $(stat -f %z $D/$f) | $shows | $caption |" >> $D/INDEX.md
  done
  print "$name: $files"
done < $manifest
