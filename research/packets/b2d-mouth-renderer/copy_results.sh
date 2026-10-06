#!/usr/bin/env bash
# copy_results.sh stub|mouth -- bring landed shards into the worktree for committing.
#   stub : the pool pulls b2d0929 results into the primary checkout ($P/$SH/b2d0929); copy them into $W.
#   mouth: the GPU jobs write into the workstation's export ($TREE/$SH/b2d0929-mouth); copy them into $P (where the
#          scorers read) and into $W. Attempt-1 directories moved aside by the re-run scripts come along.
# PIN.txt is never copied (the worktree's own, written by repin.py apply, is the record).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -n "${PIN_P:-}" ] || { echo "no PIN_P recorded yet (HANDOFF step 1)" >&2; exit 2; }
A1=research/findings/raw/_load_bearing/_b2d0929_attempt1
case "${1:-}" in
  stub)
    mkdir -p "$W/$SH/b2d0929"
    rsync -a --exclude PIN.txt --exclude 'b2d_score*.json' "$P/$SH/b2d0929/" "$W/$SH/b2d0929/"
    [ -d "$P/$A1/b2d0929" ] && mkdir -p "$W/$A1/b2d0929" && rsync -a "$P/$A1/b2d0929/" "$W/$A1/b2d0929/"
    ;;
  mouth)
    mkdir -p "$P/$SH/b2d0929-mouth" "$W/$SH/b2d0929-mouth"
    rsync -a --exclude PIN.txt "$TREE/$SH/b2d0929-mouth/" "$P/$SH/b2d0929-mouth/"
    rsync -a --exclude PIN.txt "$TREE/$SH/b2d0929-mouth/" "$W/$SH/b2d0929-mouth/"
    if [ -d "$TREE/$A1/b2d0929-mouth" ]; then
      mkdir -p "$P/$A1/b2d0929-mouth" "$W/$A1/b2d0929-mouth"
      rsync -a "$TREE/$A1/b2d0929-mouth/" "$P/$A1/b2d0929-mouth/"
      rsync -a "$TREE/$A1/b2d0929-mouth/" "$W/$A1/b2d0929-mouth/"
    fi
    ;;
  *) echo "usage: $0 stub|mouth" >&2; exit 2 ;;
esac
echo "copied $1: $(find "$W/$SH/b2d0929$([ "$1" = mouth ] && echo -mouth)" -name lb.json 2>/dev/null | wc -l) lb.json in the worktree"
