#!/bin/sh
# Sets up the llama.cpp check (docs/stock-llama.md §12): it builds llama.cpp b12070 with
# layer-shapes.patch, stream-slices.patch and stored-runs.patch on Metal and the harness
# against it, writes the f16 exports, records the engine's passes and plays the one model
# in llama.cpp against that record. This is the patched build with -DGGML_BLAS=OFF, a speed-up
# only. The claim of record runs unpatched llama.cpp in its own default build
# (docs/stock-llama.md, "What runs where"), which proof.sh with PROOF_PURE=1 drives.
#
#   tools/stock-llama/run.sh <work dir> <rom> <step>
#
# Steps: setup, export, engine, play-metal. play-metal saves the prefilled cache to
# <work dir>/yellow-cache.state and loads it on later runs. It records its passes under
# <work dir>/rec-play-metal and `diff` holds them to rec-engine. PASSES bounds the play
# (default 300), GROUP=own decodes the reduced groups.
set -eu
W=$1
ROM=$2
STEP=$3
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LLAMA=$W/llama.cpp
M=$HERE/target-metal/release/stock-llama
mkdir -p "$W"
export STOCK_GPU_LAYERS=99

case $STEP in
setup)
    [ -d "$LLAMA" ] || git clone --depth 1 --branch b12070 https://github.com/ggml-org/llama.cpp.git "$LLAMA"
    # Each patch once: a tree that already carries one refuses its dry run.
    for p in layer-shapes stream-slices stored-runs; do
        if (cd "$LLAMA" && patch -p1 -N -s --dry-run < "$HERE/$p.patch" >/dev/null 2>&1); then
            (cd "$LLAMA" && patch -p1 -N -s < "$HERE/$p.patch")
        fi
    done
    cmake -S "$LLAMA" -B "$LLAMA/build-metal" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF >/dev/null
    cmake --build "$LLAMA/build-metal" --config Release -j 4 --target llama >/dev/null
    (cd "$HERE" && env CARGO_TARGET_DIR="$HERE/target-metal" LLAMA_CPP_DIR="$LLAMA" LLAMA_CPP_BUILD="$LLAMA/build-metal" cargo build --release)
    ;;
export)
    for c in cpu cpu-memory one-model; do (cd "$REPO" && cargo run --release --example export -- "$c" "$W/$c"); done
    ;;
engine)
    rm -rf "$W/rec-engine" && mkdir -p "$W/rec-engine"
    GBCFORMER_RECORD=$W/rec-engine "$M" play none "$ROM" --frames "${FRAMES:-5}"
    ;;
play-metal)
    rm -rf "$W/rec-$STEP" && mkdir -p "$W/rec-$STEP"
    GBCFORMER_RECORD=$W/rec-$STEP STOCK_STATE=$W/yellow-cache.state STOCK_GROUP=${GROUP:-own} \
        "$M" play "$W/one-model/model.gguf" "$ROM" --frames "${FRAMES:-5}" --max-passes "${PASSES:-300}" --lanes 255
    "$M" diff "$W/rec-$STEP" "$W/rec-engine"
    ;;
*)
    echo "unknown step $STEP" >&2
    exit 2
    ;;
esac
