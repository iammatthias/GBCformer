#!/bin/sh
# Links the main checkout's ROMs (*.gb, *.gbc), boot ROMs (*_boot.bin) and vendor/ into the current git worktree,
# so tools/ and the tests run there unchanged. Creates links only: an existing file or
# link of the same name is left alone. .gitignore keeps the links untracked.
set -eu
cd "$(git rev-parse --show-toplevel)"
main=$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)
[ "$main" = "$(pwd -P)" ] && exit 0
for f in "$main"/*.gb "$main"/*.gbc "$main"/*_boot.bin "$main"/vendor; do
  [ -e "$f" ] || continue
  t=$(basename "$f")
  [ -e "$t" ] || [ -L "$t" ] || ln -s "$f" "$t"
done
