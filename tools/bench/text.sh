#!/bin/zsh
# Text-box workload: Oak's speech. Frames 0-2899 on the reference core (warmup), 600 model frames advancing dialogue with A.
# Extra arguments go to run; --core ${CORE:-model} is added unless they give --weights or --lockstep.
cd "${0:A:h}/../.."
core=(--core ${CORE:-model})
for a in "$@"; do [[ $a == --weights || $a == --lockstep ]] && core=(); done
P=()
for f in $(seq 2300 40 3500); do P+=(--press "${f}:a:4" --press "$((f+20)):down:4"); done
exec "${BIN:-./target/release/gbcformer}" run pokemon-yellow.gbc --frames 3500 --warmup 2900 "${core[@]}" --no-save "${P[@]}" "$@"
