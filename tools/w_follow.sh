#!/usr/bin/env bash
# w_follow.sh -- bring the WORKSTATION follower checkout up to published main (run at Claude session start on W).
# The workstation is a follower + compute node (owner 2026-09-30); nothing else fast-forwards it, and wspool results
# land here untracked, so a plain `git merge --ff-only` collides with the committed copies once the server lands them.
# Untracked files identical to the incoming version are removed; differing ones and the locally modified coordination
# files are copied to $ASIDE first. W-only provenance rows (runs.jsonl) are re-appended after the fast-forward.
# Never discards data: everything it removes is byte-identical to what main brings in, or saved under $ASIDE.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ "$(git symbolic-ref --short HEAD)" = main ] || { echo "w_follow: not on main, refusing"; exit 2; }
ASIDE="${ASIDE:-$HOME/Projects/sim-aside/$(date +%F)-follow}"
RUNS=research/findings/raw/_provenance/runs.jsonl
timeout 120 git fetch -q origin main
NEW=$(git rev-parse FETCH_HEAD)
git merge-base --is-ancestor HEAD "$NEW" || { echo "w_follow: local main has commits not on origin/main, refusing"; exit 3; }
[ "$(git rev-parse HEAD)" = "$NEW" ] && { echo "w_follow: already at ${NEW:0:9}"; exit 0; }
rm_n=0; aside_n=0
while IFS= read -r f; do
  [ -e "$f" ] || continue
  if [ "$(git hash-object -- "$f")" = "$(git rev-parse "$NEW:$f")" ]; then rm -f -- "$f"; rm_n=$((rm_n+1))
  else mkdir -p "$ASIDE/$(dirname "$f")"; mv -- "$f" "$ASIDE/$f"; aside_n=$((aside_n+1)); fi
done < <(comm -12 <(git ls-files --others --exclude-standard | sort) <(git diff --name-only --diff-filter=A HEAD "$NEW" | sort))
saved_runs=""
while IFS= read -r f; do
  mkdir -p "$ASIDE/$(dirname "$f")"; cp -p -- "$f" "$ASIDE/$f"; git checkout -- "$f"
  [ "$f" = "$RUNS" ] && saved_runs="$ASIDE/$f"
done < <(comm -12 <(git diff --name-only | sort) <(git diff --name-only HEAD "$NEW" | sort))
git merge --ff-only -q "$NEW"
if [ -n "$saved_runs" ]; then
  python3 - "$RUNS" "$saved_runs" <<'EOF'
import sys
p, saved = sys.argv[1], sys.argv[2]
have = set(open(p).read().splitlines())
extra = [l for l in open(saved).read().splitlines() if l and l not in have]
open(p, "a").write("".join(x + "\n" for x in extra))
print(f"w_follow: re-appended {len(extra)} W-only provenance rows")
EOF
fi
echo "w_follow: now at $(git log --oneline -1 | cut -c1-80); removed $rm_n identical untracked, moved $aside_n to $ASIDE"
