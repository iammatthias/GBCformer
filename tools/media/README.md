# Media tooling

These scripts play a recorded input log in the model and cut clips from the model's
frames. The clips are not in the repository.

Two rules hold for every clip:

- Every frame is decoded by the model, in the minimal runner (`tools/runner`). The
  reference interpreter plays the same log, and every frame's hash from both is compared.
  It supplies no frame of a clip.
- The inputs are button presses by frame, from power-on. Nothing edits memory.

## The demo: 193,603 frames of Pokémon Yellow, from power-on to Mt. Moon

`demo/` holds a recorded input log and the hashes to check a replay against.

| File | Contents |
| - | - |
| `demo/input.log` | 10,713 presses, one `FRAME:BUTTON:HOLD` a line, the `--press` format (sha1 `f6b9dbbf…`) |
| `demo/checkpoints.txt` | Places along the way: the frames run so far, the sha1 of the last of them, a name, where the game is |
| `demo/frame-hashes-every-1000.txt` | `FRAME HASH` for frames 0, 1,000, … 193,000 and the last frame, 193,602 |
| `demo/frame-hashes.sha1` | The count of frames and the sha1 of the whole list of frame hashes (`88bb3acf…`) |

The log starts a new game, takes the Boulder Badge at frame 152,100 and ends inside
Mt. Moon, 54 minutes of game time. A frame's hash is the sha1 of its 160 × 144 pixels as
red, green and blue bytes.

### What you need

- Your own Pokémon Yellow ROM (sha1 `cc7d0326…`), as `pokemon-yellow.gbc` in the checkout.
- That exact CGB boot ROM, as `cgb_boot.bin` in the checkout. The log was recorded from
  power-on with SameBoy's CGB boot ROM as the project owner obtained it from the Internet
  Archive, hash verified: 2,304 bytes, sha1 `80e3ee82416c1a74827a5459fde32f04bf51a1f6`. The
  repository does not include it, and you need a file with that sha1 to replay the log.
  **The hashes hold only with that file.** The game's random numbers follow the time since
  power-on, so another boot ROM gives another game for the same presses. The SameBoy v1.0.5
  boot ROM that `tools/hwtests/fetch.sh` fetches (sha1 `11390377…`) is a different build:
  with it the frame hashes differ from frame 16 on, and the game goes another way. The
  `cgb_boot.bin` of SameBoy's releases 0.12.3 to 1.0.5 is not this file either.
- The log cannot be replayed without a boot ROM. The runner's other start, a snapshot of
  the state after the boot, holds the ROM and is not in the repository, and the program's
  start with no boot ROM reaches the game at another time.
- The one model's GGUF, `ff86dde7…`. Building it takes about 25 GB of memory.

  ```sh
  cargo run --release --example export -- one-model model --gguf-only --layer-shapes --align 64
  shasum model/model.gguf        # 5dce2fc1212166f05d50e03c6dd5915573fd4e5e
  ```

- `zsh`. The clip scripts also need `ffmpeg`, `jq` and Bun.

### Replay and check

```sh
cargo build --release
(cd tools/runner && cargo build --release)

tools/runner/target/release/gbcformer-runner model/model.gguf --rom pokemon-yellow.gbc --boot cgb_boot.bin \
  --frames 193603 --presses tools/media/demo/input.log --time 1700000000 --outputs out
tools/media/verify-hashes.sh out/frames.txt
```

The runner prints its passes, its dense fallbacks (0) and its frames a second. On an M1 Max the
whole log takes about 42 minutes at 77 frames a second. `verify-hashes.sh` holds `out/frames.txt` to the kept hashes
and, for the whole log, to the list's sha1. A shorter run works too: `--frames 20000`
reaches Viridian City in about five minutes and is held to the 20 kept hashes it reaches.

`model.sh` does the same and also plays the log on the reference interpreter, then
compares every frame's hash and every sample from both:

```sh
tools/media/model.sh model/model.gguf tools/media/demo/input.log 193603 out
```

## Cut clips

```sh
tools/media/model.sh <model.gguf> <input log> <frames> <out dir> [FROM:TO ...]
bun tools/media/verify-pixels.ts <out dir>/model/frames.txt 0 <out dir>/pixels/*.bgr0
tools/media/story.sh <manifest.tsv> <work dir> <dest dir> <model.gguf>
```

The runner has two flags for this, both opt-in:

- `--presses FILE` reads one `--press` from each line of FILE.
- `--pixels DIR --clip FROM:TO` writes the frames shown from FROM up to TO as raw pixels,
  with the count of samples played before the first.

`verify-pixels.ts` hashes each written frame again and holds it to its line of
`frames.txt`, so a clip shows the frames whose hashes were compared.

Each manifest line names a clip, the run and frame range it comes from, its kind and its
caption. `story.sh` writes the MP4, GIF or PNG and a sidecar `.json` with the weights
digest, the GGUF's sha1, the commit, the command, the input log's sha1, the frame range
and the sha1 of the clip's frame-hash list. `clip.sh` does the encoding and states the
frame rates in its header. `sheet.sh` makes a contact sheet of a range. `story.txt`,
`stills.txt` and `readme.txt` are the manifests of the clips cut from the demo log
(`story.sh` reads them with `|` turned into tabs).

## Other scripts

- `bench.sh` renders the scenes of the three benchmark workloads with every frame from
  the runner, from frame 0, and `bench-reference.sh` compares their frame hashes with the
  reference interpreter's.
- `inventory.sh` and `inventory-encode.sh` render older media with the full program in
  lockstep.
