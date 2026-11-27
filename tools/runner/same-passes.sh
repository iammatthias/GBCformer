#!/bin/zsh
# same-passes.sh <model.gguf> <out dir> <rom> <frames> [run flags]: the full program as one
# model from the GGUF alone, plain and with every component in lockstep (`--lockstep all`),
# every stream's pass record compared event by event, none capped. Under the GBC CPU and
# memory lockstep the memory core also decodes the slots only its audit reads
# (`MemCheckpointCore::without_audit_slots`: OP0' OP1' OP2' RV0 RV1); the comparison leaves
# those slots out of both records (their tokens and positions, `digest.ts` DROP) and requires
# that the lockstep side decoded some and the plain side none. Every other token, position
# and memory head read must be equal. Prints one `ok` or `FAIL` line per check; exits 1 on any.
#
# FULL (default ./target/release/gbcformer, built with --features stock) names the binary.
set -u
cd "${0:A:h}/../.."
model=$1 out=$2 rom=$3 frames=$4; shift 4
FULL=${FULL:-./target/release/gbcformer}
AUDIT="OP0',OP1',OP2',RV0,RV1"
NAMES=(stream-0-A stream-1-LY stream-2-S0 stream-3-C0 stream-4-MLY stream-5-LY)
rc=0
fail() { print -r -- "FAIL  $*"; rc=1; }
pass() { print -r -- "ok    $*"; }
for mode in plain lockstep; do
  d=$out/$mode; rm -rf $d; mkdir -p $d
  readers=()
  for n in $NAMES; do
    mkfifo $d/$n.jsonl
    { cat $d/$n.jsonl | DROP=$AUDIT bun tools/runner/digest.ts $d/$n $n.jsonl } &
    readers+=($!)
  done
  ls=(); [[ $mode == lockstep ]] && ls=(--lockstep all)
  GBCFORMER_GGUF_ONLY=$model GBCFORMER_APU_THREAD=0 GBCFORMER_PPU_THREAD=0 GBCFORMER_RECORD=$d \
    $FULL run $rom --frames $frames --weights all --one-model $ls --no-save "$@" > $out/$mode.log 2>&1 || fail "the $mode run exited $? ($(tail -1 $out/$mode.log))"
  for n in $NAMES; do exec {fd}<>$d/$n.jsonl; exec {fd}>&-; done
  wait $readers
  rm -f $d/stream-*.jsonl(Np)
  cat $d/*.sha1(N) | sort -k3 > $out/$mode.sha1
done
hp=$(grep -o 'frame_sha1=........' $out/plain.log) hl=$(grep -o 'frame_sha1=........' $out/lockstep.log)
[[ -n $hp && $hp == $hl ]] && pass "last frame $hp" || fail "last frame: plain ${hp:-none}, lockstep ${hl:-none}"
dp=$(cat $out/plain/stream-0-A.dropped 2>/dev/null) dl=$(cat $out/lockstep/stream-0-A.dropped 2>/dev/null)
[[ ${dp:-x} == 0 && ${dl:-0} -gt 0 ]] && pass "audit-only positions left out: plain $dp, lockstep $dl" || fail "audit-only positions: plain ${dp:-none}, lockstep ${dl:-none}, want 0 and above 0"
if diff -q $out/plain.sha1 $out/lockstep.sha1 > /dev/null; then pass "pass records equal, every event: $(cut -d' ' -f2 $out/plain.sha1 | tr '\n' ' ')"; else fail "pass records differ"; paste $out/plain.sha1 $out/lockstep.sha1; fi
if faults=$(grep -hE '(^|[^0-9])[1-9][0-9]* (on the )?dense fallback|panicked|refused' $out/plain.log $out/lockstep.log); then fail "faults: ${faults[1,200]}"; else pass "no dense fallback, panic or refusal"; fi
exit $rc
