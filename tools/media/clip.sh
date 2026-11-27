#!/bin/zsh
# clip.sh <frames.bgr0> <out base>: a range of raw frames (the runner's `--pixels`,
# 160x144, four bytes a pixel) as <out base>.mp4, .gif and .png. Set by environment:
#   FIRST, COUNT  the frames of the file to take (default: all)
#   AUDIO, BEFORE the runner's audio.raw (48 kHz stereo s16le) and the samples played
#                 before the file's first frame (its .samples file): the MP4 gets AAC 128k
#   SCALE         the MP4's integer scale (default 4, 640x576), nearest-neighbour
#   GIF=N         also a GIF at scale N; MP4=0 leaves the MP4 out
#   STILL=N       instead, frame FIRST alone as a PNG at scale N
#
# MP4: H.264 yuv420p +faststart at 59.7275 fps.
# GIF: every second frame at 29.86 fps. A GIF's delays are whole centiseconds, so the
# frames alternate 3, 3 and 4 cs (33.3 ms on average against the 33.49 ms of two frames):
# it plays 0.5% fast. Full rate would need 1.67 cs a frame, which no GIF holds.
# GIF_ALL=1 keeps every frame instead, each for 2 cs (50 fps): it plays 16% slow.
set -eu
raw=$1 out=$2 scale=${SCALE:-4} first=${FIRST:-0}
bytes=$((160 * 144 * 4))
count=${COUNT:-$(( $(stat -f %z $raw) / bytes - first ))}
# Scratch files go to CLIP_TMP (default the system's temporary directory), not beside the output.
tmp=$(mktemp -d ${CLIP_TMP:-${TMPDIR:-/tmp}}/clip.XXXXXX)
part=$tmp/frames.bgr0
tail -c +$(( first * bytes + 1 )) $raw | head -c $(( count * bytes )) > $part
video=(-f rawvideo -pix_fmt bgr0 -s 160x144 -framerate 59.7275 -i $part)
if (( ${STILL:-0} )); then
  ffmpeg -nostdin -v error -y $video -frames:v 1 -vf scale=$((160 * STILL)):$((144 * STILL)):flags=neighbor $out.png
elif (( ${MP4:-1} )); then
  x=scale=$((160 * scale)):$((144 * scale)):flags=neighbor
  if [[ -n ${AUDIO:-} ]]; then
    # 48,000 samples a second over 59.7275 frames a second (803.65 a frame), four bytes a sample pair.
    from=$(( BEFORE + (first * 4800000000 + 2986375) / 5972750 ))
    samples=$(( (count * 4800000000 + 2986375) / 5972750 ))
    tail -c +$(( from * 4 + 1 )) $AUDIO | head -c $(( samples * 4 )) > $tmp/audio.s16
    ffmpeg -nostdin -v error -y $video -f s16le -ar 48000 -ac 2 -i $tmp/audio.s16 -vf $x -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart $out.mp4
  else
    ffmpeg -nostdin -v error -y $video -vf $x -c:v libx264 -pix_fmt yuv420p -movflags +faststart $out.mp4
  fi
fi
if (( ${GIF:-0} )); then
  pick="select='not(mod(n,2))',setpts=N/(29.86375*TB)" rate=29.86375
  (( ${GIF_ALL:-0} )) && pick="setpts=N/(50*TB)" rate=50
  ffmpeg -nostdin -v error -y $video -vf "$pick,scale=$((160 * GIF)):$((144 * GIF)):flags=neighbor,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -r $rate $out.gif
fi
rm -r $tmp
print "$count frames"
