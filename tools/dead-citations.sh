#!/bin/sh
# Lists every commit hash the tracked text cites that this history does not contain, and
# exits 1 on any. A cited commit must be an ancestor of HEAD.
#
#   - A hex token of 8 or more characters (a digest, a frame hash, a file's sha1, a 40-character
#     upstream commit) is checked only when it names a commit object here, so digests pass.
#   - A 7-character token is how this repository cites its own commits, so one with a letter
#     and a digit, or one in backticks or after "commit", must name a commit of this history.
#     Upstream commits cited at 7 or 8 characters (llama.cpp, SameBoy, SingleStepTests and the
#     compressors' sources) are named in UPSTREAM.
#   - The word branch before a name in backticks must name a public branch (BRANCHES).
set -eu
cd "$(git rev-parse --show-toplevel)"

BRANCHES='main'
UPSTREAM='00f113ec ea667ad 38b926bd f9c30210 9f30e09 2da5c4e 6009704 3652407'
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git grep -I -n -w -o -E '[0-9a-f]{7,40}' -- . ':!tools/digests.txt' ':!*.lock' ':!tools/media/demo' >"$tmp/tokens" || true
git grep -I -n -o -E '(`[0-9]{7}`|commit [0-9]{7})' -- . ':!*.lock' >"$tmp/digits" || true

# file:line:token -> candidates worth resolving
awk -F: -v up="$UPSTREAM" '
  BEGIN { n = split(up, u, " "); for (i = 1; i <= n; i++) allow[u[i]] = 1 }
  {
    t = $3
    if (FILENAME ~ /digits$/) gsub(/[^0-9]/, "", t)
    for (p in allow) if (index(t, p) == 1) next
    if (FILENAME ~ /digits$/) { print $1 ":" $2 ":" t ":short"; next }
    if (t ~ /^0b[01]+$/) next
    if (length(t) == 7) { if (t ~ /[a-f]/ && t ~ /[0-9]/) print $1 ":" $2 ":" t ":short" }
    else print $1 ":" $2 ":" t ":long"
  }' "$tmp/tokens" "$tmp/digits" | sort -u >"$tmp/cand"

cut -d: -f3 "$tmp/cand" | sort -u >"$tmp/uniq"
git cat-file --batch-check='%(objecttype) %(objectname)' <"$tmp/uniq" 2>/dev/null >"$tmp/types" || true
paste -d' ' "$tmp/uniq" "$tmp/types" >"$tmp/resolved"

dead=0
while read -r tok kind sha; do
  case $kind in
    commit) git merge-base --is-ancestor "$sha" HEAD 2>/dev/null || echo "$tok notin" ;;
    *) echo "$tok missing" ;;
  esac
done <"$tmp/resolved" >"$tmp/verdict"

while IFS=: read -r file line tok len; do
  v=$(awk -v t="$tok" '$1 == t { print $2 }' "$tmp/verdict")
  if [ "$v" = notin ] || { [ "$v" = missing ] && [ "$len" = short ]; }; then
    echo "$file:$line: $tok (not a commit of this history)"
    dead=$((dead + 1))
  fi
done <"$tmp/cand"

git grep -I -n -o -E 'branch `[A-Za-z0-9._/-]+`' -- . ':!*.lock' >"$tmp/branches" || true
while IFS=: read -r file line text; do
  name=${text#branch \`}; name=${name%\`}
  if ! printf '%s\n' $BRANCHES | grep -qxF "$name"; then
    echo "$file:$line: branch $name (not a public branch)"
    dead=$((dead + 1))
  fi
done <"$tmp/branches"

echo "dead citations: $dead"
[ "$dead" -eq 0 ]
