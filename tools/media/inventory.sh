#!/bin/zsh
# inventory.sh <model.gguf> <out dir>: the runs behind the earlier media, on the
# full program with every part in weights as one model read from the GGUF alone, each
# part's reference beside it in lockstep (the first difference stops the run). Frames go
# to <out dir>/<name>/frame_NNNNN.ppm, each run's log to <out dir>/<name>.log. A run whose
# frame directory exists is skipped.
cd ${0:A:h}/../..
G=$1 V=$2
mkdir -p $V
go() {
  local name=$1; shift
  [[ -e $V/$name ]] && { print -r -- "$name: skipped, $V/$name exists"; return }
  GBCFORMER_GGUF_ONLY=$G nice -n 19 ./target/release/gbcformer run "$@" --weights all --lockstep all --one-model --no-save --record $V/$name > $V/$name.log 2>&1
  print -r -- "$name: exit $? $(grep -o 'frame_sha1=[0-9a-f]*' $V/$name.log) $(grep -ciE '(^|[^0-9])[1-9][0-9]* (on the )?dense fallback|panicked|refused|diverg' $V/$name.log) fault lines"
}
oak=(--press 2300:a:4 --press 2420:a:4)
go yellow-dmg pokemon-yellow.gbc --frames 2820 --hardware dmg $oak
go yellow-cgb pokemon-yellow.gbc --frames 2900 --hardware cgb $oak --wav $V/yellow-cgb.wav
go zelda legend-of-zelda-links-awakening.gbc --frames 300
go crystal pokemon-crystal.gbc --frames 300
go mario super-mario-bros-deluxe.gbc --frames 300
