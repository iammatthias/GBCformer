#!/bin/sh
# Prints every merge-conflict marker line, as file:line: text, and exits 1 if there is
# one: a line starting "<<<<<<< ", "||||||| " or ">>>>>>> " anywhere, or a bare
# "=======" between a "<<<<<<< " line and its ">>>>>>> " (outside one it is a Markdown
# heading rule). Checks the files named, or every tracked text file.
#
#   tools/conflict-markers.sh
#   tools/conflict-markers.sh docs/perf-log.md
set -u
cd "$(dirname "$0")/.."
program='
  FNR == 1 { open = 0 }
  /^<<<<<<< / { open = 1; found = 1; print FILENAME ":" FNR ": " $0; next }
  /^>>>>>>> / { open = 0; found = 1; print FILENAME ":" FNR ": " $0; next }
  /^\|\|\|\|\|\|\| / { found = 1; print FILENAME ":" FNR ": " $0; next }
  open && /^=======$/ { found = 1; print FILENAME ":" FNR ": " $0 }
  END { exit found }
'
if [ $# -gt 0 ]; then
  LC_ALL=C awk "$program" "$@"
  exit $?
fi
# xargs exits 123 when any batch of files had a marker.
git grep -I --name-only -z -e '' | LC_ALL=C xargs -0 awk "$program"
[ $? -eq 0 ]
