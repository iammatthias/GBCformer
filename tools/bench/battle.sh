#!/bin/zsh
# Battle workload: scripted presses from power-on through Oak's lab to the rival battle on the reference core
# (warmup, frames 0-14879), then 600 model frames of one battle round: the FIGHT menu, Thundershock and Tackle
# with their animations, a critical hit and both HP bars draining.
# Extra arguments go to run; --core ${CORE:-model} is added unless they give --weights or --lockstep.
cd "${0:A:h}/../.."
core=(--core ${CORE:-model})
for a in "$@"; do [[ $a == --weights || $a == --lockstep ]] && core=(); done
P=()
for f in $(seq 2300 40 5660); do P+=(--press "${f}:a:4" --press "$((f+20)):down:4"); done
P+=(--press 5720:right:64 --press 5800:up:100 --press 5960:right:40)
P+=(--press 6080:down:100 --press 6190:left:80 --press 6280:down:20)
P+=(--press 6380:right:84 --press 6470:up:120)
for f in $(seq 6880 40 11160); do P+=(--press "${f}:a:4"); done
P+=(--press 11180:down:12 --press 11210:right:28 --press 11250:up:4)
for f in $(seq 11280 40 13400); do P+=(--press "${f}:a:4"); done
for f in $(seq 13440 40 13520); do P+=(--press "${f}:b:4"); done
P+=(--press 13560:down:100)
for f in $(seq 13700 40 15480); do P+=(--press "${f}:a:4"); done
exec "${BIN:-./target/release/gbcformer}" run pokemon-yellow.gbc --frames 15480 --warmup 14880 "${core[@]}" --no-save "${P[@]}" "$@"
