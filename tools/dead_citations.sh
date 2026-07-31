#!/usr/bin/env bash
# Which findings do the STEERING docs cite that do not exist anywhere in the repo?
#
# Measured 2026-07-31: 427 distinct citations, 43 of them DEAD. A session reads CLAUDE.md / the board / the
# roadmap first, follows a citation, and lands on nothing -- then re-derives. This is failure class 8
# (stale-pointer) in its most literal form, and unlike a stale VERDICT it is trivially checkable.
#
# NOTE ON THE FIRST VERSION OF THIS CHECK: it looked only in research/findings/ and reported 224 dead. Most of
# those live in docs/plans/. The number was wrong by 5x because the check was wrong, which is exactly the failure
# mode this repo keeps hitting -- so it searches the WHOLE repo now.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
grep -ohE '20[0-9]{2}-[0-9]{2}-[0-9]{2}-[A-Za-z0-9._-]+\.md' \
     CLAUDE.md GAP_CLOSURE_MISSION.md ROADMAP.md docs/plans/*.md 2>/dev/null | sort -u > /tmp/_cited.$$
TOTAL=$(wc -l < /tmp/_cited.$$); DEAD=0
while read -r f; do
  find . -name "$f" -not -path "./.git/*" -print -quit 2>/dev/null | grep -q . || { echo "  DEAD: $f"; DEAD=$((DEAD+1)); }
done < /tmp/_cited.$$
rm -f /tmp/_cited.$$
echo "  => $DEAD dead of $TOTAL cited"
[ "$DEAD" -eq 0 ]
