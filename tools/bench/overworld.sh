#!/bin/zsh
# Overworld workload: scripted presses into Red's bedroom on the reference core (warmup), then 600 model frames walking.
# Extra arguments go to run; --core ${CORE:-model} is added unless they give --weights or --lockstep.
cd "${0:A:h}/../.."
core=(--core ${CORE:-model})
for a in "$@"; do [[ $a == --weights || $a == --lockstep ]] && core=(); done
P=()
for f in $(seq 2300 40 7000); do P+=(--press "${f}:a:4" --press "$((f+20)):down:4"); done
for f in $(seq 7200 64 7800); do P+=(--press "${f}:left:24" --press "$((f+32)):right:24"); done
exec "${BIN:-./target/release/gbcformer}" run pokemon-yellow.gbc --frames 7800 --warmup 7200 "${core[@]}" --no-save "${P[@]}" "$@"
