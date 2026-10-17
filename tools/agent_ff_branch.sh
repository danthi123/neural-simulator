#!/usr/bin/env bash
# agent_ff_branch.sh research/<branch> <commit> -- the ONE way the driver agent (Hermes) moves a research branch FORWARD
# to a commit that already contains it (typically a claude-build/* fix Claude made on top of the branch tip).
#
# Owner 2026-10-16 ("let Hermes fast-forward research branches, not main"): Hermes's approvals deny any typed
# `git merge`, so a fast-forward of its own branch onto a Claude fix waited for a Claude session (STOP #12b that day).
# This script only ever fast-forwards; it never merges, never writes main, never pushes:
#   1. the branch must be a local research/* branch; the target must be a commit that DESCENDS from the branch tip
#      (a true fast-forward; anything else is refused and nothing changes);
#   2. a branch checked out in a worktree is fast-forwarded THERE (`merge --ff-only`), and only when that worktree has
#      no staged or unstaged change to a tracked file and no merge in progress; a branch checked out nowhere has its
#      ref moved with the old value checked (`update-ref <ref> <new> <old>`);
#   3. it prints the old and new tips, so the move can be quoted in a RESULT line.
# Exit 0 = moved, or already at the target; 2 = refused (reason printed; report it as NEEDS-CLAUDE); else git error.
set -uo pipefail
refuse() { echo "⛔ agent_ff_branch: $*" >&2; exit 2; }

b=${1:-} target=${2:-}
[[ "$b" =~ ^research/[A-Za-z0-9._/-]+$ ]] && [ -n "$target" ] || refuse "usage: agent_ff_branch.sh research/<branch> <commit>"
case "$b" in *..*|*/.*|*.lock|*/|*//*) refuse "bad branch name $b" ;; esac
old=$(git rev-parse -q --verify "refs/heads/$b^{commit}") || refuse "no local branch $b"
new=$(git rev-parse -q --verify "$target^{commit}") || refuse "no commit $target"
[ "$old" = "$new" ] && { echo "$b already at ${new:0:9}"; exit 0; }
git merge-base --is-ancestor "$old" "$new" || refuse "${new:0:9} does not contain $b @ ${old:0:9}: not a fast-forward"

wt=$(git worktree list --porcelain | awk -v ref="branch refs/heads/$b" '/^worktree /{w=substr($0,10)} $0==ref{print w}')
if [ -n "$wt" ]; then
  cd "$wt" || exit 1
  [ "$(git branch --show-current)" = "$b" ] || refuse "$wt is not on $b"
  [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ] && refuse "a merge is in progress in $wt"
  git diff --quiet && git diff --cached --quiet || refuse "$wt has uncommitted changes to tracked files; commit or drop them first"
  git merge -q --ff-only "$new" >/dev/null 2>&1 || refuse "git refused the fast-forward in $wt (an untracked file in the way?); nothing changed"
else
  git update-ref -m "agent_ff_branch: fast-forward" "refs/heads/$b" "$new" "$old" || refuse "the ref moved under us; nothing changed"
fi
echo "$b fast-forwarded ${old:0:9} -> ${new:0:9}${wt:+ (worktree $wt)}"
