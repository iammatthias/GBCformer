#!/bin/zsh
# sheet.sh <frames.bgr0> <out.png> [EVERY]: a contact sheet, one tile every EVERY frames
# (default: 48 tiles over the file), eight tiles a row at half size.
raw=$1 n=$(( $(stat -f %z $1) / 92160 ))
every=${3:-$(( (n + 47) / 48 ))}
ffmpeg -v error -y -f rawvideo -pix_fmt bgr0 -s 160x144 -i $raw -vf "select='not(mod(n,$every))',scale=80:72:flags=neighbor,tile=8x$(( (n / every + 8) / 8 ))" -frames:v 1 $2
print "$n frames, a tile every $every"
