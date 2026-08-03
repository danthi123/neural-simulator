#!/usr/bin/env bash
# Periodic RAG refresh entry point. Unlike the repository post-commit hook, this
# also notices papers/textbooks added directly to the sibling source catalog.
set -u

CANONICAL=${SIM_CANONICAL_REPO:-$HOME/Projects/sim}
BRANCH=${SIM_RAG_REFRESH_BRANCH:-main}

repo=""
candidate=""
while IFS=' ' read -r key value; do
  case "$key" in
    worktree)
      candidate=$value
      ;;
    branch)
      if [ "$value" = "refs/heads/$BRANCH" ]; then
        repo=$candidate
        break
      fi
      ;;
  esac
done < <(git -C "$CANONICAL" worktree list --porcelain 2>/dev/null)

ROOT=${SIM_RAG_ROOT:-$(dirname "$CANONICAL")/rag_index}
LOG="$ROOT/_autoupdate.log"
mkdir -p "$ROOT"

if [ -z "$repo" ] || [ ! -d "$repo" ]; then
  printf '[periodic] BLOCKED: no worktree owns branch=%s (canonical=%s)\n' \
    "$BRANCH" "$CANONICAL" >> "$LOG"
  exit 1
fi

PY=${SIM_RAG_PYTHON:-$CANONICAL/.venv-rag/bin/python}
UPD=${SIM_RAG_UPDATER:-$repo/tools/rag/update_indexes.py}
if [ ! -x "$PY" ]; then
  printf '[periodic] BLOCKED: RAG interpreter missing: %s\n' "$PY" >> "$LOG"
  exit 1
fi
if [ ! -f "$UPD" ]; then
  printf '[periodic] BLOCKED: updater missing: %s\n' "$UPD" >> "$LOG"
  exit 1
fi

if ! git -C "$repo" diff --quiet -- \
    ':(glob)research/findings/*.md' ':(glob)research/findings/**/*.md' \
    ':(glob)docs/*.md' ':(glob)docs/plans/*.md' \
    CLAUDE.md ROADMAP.md README.md GAP_CLOSURE_MISSION.md \
  || ! git -C "$repo" diff --cached --quiet -- \
    ':(glob)research/findings/*.md' ':(glob)research/findings/**/*.md' \
    ':(glob)docs/*.md' ':(glob)docs/plans/*.md' \
    CLAUDE.md ROADMAP.md README.md GAP_CLOSURE_MISSION.md \
  || [ -n "$(git -C "$repo" ls-files --others --exclude-standard -- \
    ':(glob)research/findings/*.md' ':(glob)research/findings/**/*.md' \
    ':(glob)docs/*.md' ':(glob)docs/plans/*.md' \
    CLAUDE.md ROADMAP.md README.md GAP_CLOSURE_MISSION.md)" ]; then
  printf '[periodic] DEFER: indexed project prose has uncommitted changes: %s\n' "$repo" >> "$LOG"
  exit 0
fi

printf '[periodic] START: branch=%s source=%s\n' "$BRANCH" "$repo" >> "$LOG"
SIM_REPO="$repo" "$PY" "$UPD" >> "$LOG" 2>&1
status=$?
printf '[periodic] EXIT: status=%s source=%s\n' "$status" "$repo" >> "$LOG"
exit "$status"
