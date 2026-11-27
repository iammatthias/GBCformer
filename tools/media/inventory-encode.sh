#!/bin/zsh
# inventory-encode.sh <inventory.sh's out dir> <dest dir>: the earlier media
# under their names, from the frames inventory.sh recorded. MP4s are H.264 yuv420p
# +faststart at 59.7275 fps, nearest-neighbour. The two GIFs keep every second frame at
# 29.86 fps, as the old ones did; their delays alternate 3, 3 and 4 cs.
set -eu
V=$1 D=$2
mkdir -p $D
x4=scale=640:576:flags=neighbor
seq() { print -r -- -framerate 59.7275 -start_number $2 -i $V/$1/frame_%05d.ppm }
mp4=(-c:v libx264 -pix_fmt yuv420p -movflags +faststart)
gif() { # run, first frame, frames, name
  ffmpeg -v error -y $(seq $1 $2) -frames:v $3 $=mp4 -vf $x4 $D/$4.mp4
  ffmpeg -v error -y $(seq $1 $2) -vf "trim=end_frame=$3,select='not(mod(n,2))',setpts=N/(29.86375*TB),$x4,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -r 29.86375 $D/$4.gif
}
still() { magick $V/$1/frame_$2.ppm -scale $3 $D/$4.png }
gif yellow-dmg 850 1556 pikachu-intro
still yellow-dmg 02000 640x576 shot-2000-title
still yellow-dmg 02800 640x576 shot-2800-oak
ffmpeg -v error -y $(seq yellow-dmg 0) $=mp4 -vf $x4 $D/boot-to-oak.mp4
gif yellow-cgb 900 1800 pikachu-intro-color
still yellow-cgb 02000 640x576 shot-color-2000
ffmpeg -v error -y $(seq yellow-cgb 0) -i $V/yellow-cgb.wav $=mp4 -vf $x4 -c:a aac -b:a 128k -ar 48000 $D/boot-to-oak-color-sound.mp4
still yellow-cgb 02720 480x432 full-stack-oak
ffmpeg -v error -y $(seq zelda 0) $(seq crystal 0) $(seq mario 0) -filter_complex "[0][1][2]hstack=3,scale=960:288:flags=neighbor" $=mp4 $D/three-games-lockstep.mp4
ls -l $D | awk '{print $5, $9}'
