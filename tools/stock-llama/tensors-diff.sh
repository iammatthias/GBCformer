#!/bin/zsh
# usage: tools/stock-llama/tensors-diff.sh <a.txt> <b.txt>
# Two STOCK_TENSORS files (src/shim.c) compared tensor by tensor: the same decode on two backends or two builds.
# f32 tensors pair by op, shape and the count of that pair so far, since node numbers differ between backends.
# Prints each pair whose hashes differ with both sides' largest value and sum, then the totals.
key() { awk '/ f32 / { for (i=1;i<=NF;i++) { if ($i=="f32") { op=$(i-1); shape=$(i+1) } if ($i=="hash") h=$(i+1); if ($i=="top") t=$(i+1); if ($i=="sum") s=$(i+1) } k=op " " shape; name=$0; sub(/ [A-Z_0-9]+ f32 .*/, "", name); print k "#" (++c[k]) "\t" h "\t" t "\t" s "\t" name }' $1; }
awk -F'\t' 'NR==FNR { h[$1]=$2; t[$1]=$3; s[$1]=$4; next }
  { n++; if (!($1 in h)) { miss++; next }
    if (h[$1]==$2) same++; else { d++; print "differs: " $1 " (" $5 "): largest " t[$1] " vs " $3 ", sum " s[$1] " vs " $4 } }
  END { print n " f32 tensors in the second file: " same+0 " with the same hash, " d+0 " differ, " miss+0 " without a pair" }' <(key $1) <(key $2)
