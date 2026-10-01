#!/usr/bin/env bash
# agent_merge.sh research/<branch> -- the ONE way the driver agent (Hermes) merges its own work into main.
#
# Owner ruling 2026-10-01 ("Routine merges"): Hermes merges routine work itself -- results data, docs, worklogs, new
# tests and new findings marked PROVISIONAL -- after two MiMo Pro reviewers approve it (runbook RULING routine-merges).
# Everything else stays Claude's. This script enforces the path half of that ruling and makes the merge go through
# every gate:
#   1. runs in the main checkout, on main, with a clean index;
#   2. tools/agent_merge_policy.py refuses any reserved path (sim code, gates, tools, preregs, HANDOFFs, scorers,
#      edits to existing runners / tests / findings, any deletion or rename);
#   3. merges with --no-ff --no-commit, then concludes with `git commit`, so the full pre-commit suite runs on the
#      merge result, including the uid-99 backstop that re-checks the STAGED result with the same policy (the step-2
#      three-dot check is only a pre-screen: with several merge bases or renames on main the result can differ).
#      A conflict or a failing gate aborts the merge and leaves main as it was.
# Exit 0 = merged; 2 = refused (reason printed); anything else = a git error. Never bypass with --no-verify.
set -uo pipefail
refuse() { echo "⛔ agent_merge: $*" >&2; exit 2; }

b=${1:-}
[[ "$b" =~ ^research/[A-Za-z0-9._/-]+$ ]] || refuse "usage: agent_merge.sh research/<branch>"
ROOT=$(git rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1
[ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ] || refuse "run in the main checkout, not a worktree"
[ "$(git branch --show-current)" = main ] || refuse "the main checkout is not on main"
git rev-parse -q --verify "refs/heads/$b^{commit}" >/dev/null || refuse "no local branch $b"
git diff --cached --quiet || refuse "the index is not clean; commit or unstage first"
[ -f .git/MERGE_HEAD ] && refuse "a merge is already in progress"

full=$(git rev-parse "refs/heads/$b^{commit}") || exit 1   # pin ONE commit: the policy checks it, the merge uses it
sha=${full:0:9}
python3 tools/agent_merge_policy.py "$b" "$full" || exit 2

if ! git merge --no-ff --no-commit "$full"; then
  git merge --abort 2>/dev/null
  refuse "merging $b conflicts with main; merge aborted, main unchanged"
fi
if ! git commit -q -m "Merge $b @ $sha (agent_merge.sh: routine merge, owner ruling 2026-10-01)"; then
  git merge --abort 2>/dev/null
  refuse "the gates refused the merge of $b; merge aborted, main unchanged"
fi
echo "merged $b @ $sha into main as $(git rev-parse --short HEAD)"
