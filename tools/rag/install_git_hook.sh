#!/usr/bin/env bash
# Install the RAG auto-update post-commit hook into THIS repo's git hooks dir.
# Idempotent: if a post-commit hook already exists, appends our call rather than clobbering it.
# Git hooks live in .git/ (not version-controlled), so run this once per clone/worktree.
set -e
cd "$(git rev-parse --show-toplevel)"
HOOKS="$(git rev-parse --git-path hooks)"
SRC="tools/git-hooks/post-commit"
DST="$HOOKS/post-commit"
mkdir -p "$HOOKS"

if [ ! -f "$DST" ]; then
  cp "$SRC" "$DST"; chmod +x "$DST"
  echo "installed $DST"
elif grep -q "update_indexes.py" "$DST"; then
  echo "already installed ($DST already calls update_indexes.py)"
else
  # append our block to the existing hook (skip the shebang line of our source)
  { echo ""; echo "# --- RAG auto-update (appended by tools/rag/install_git_hook.sh) ---"; tail -n +2 "$SRC"; } >> "$DST"
  chmod +x "$DST"
  echo "appended RAG auto-update block to existing $DST"
fi
echo "Done. New commits touching research/findings|docs|CLAUDE|ROADMAP|README will auto-refresh both RAG indexes."
