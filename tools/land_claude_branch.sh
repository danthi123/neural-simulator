#!/usr/bin/env bash
# land_claude_branch.sh claude/<topic> <sha> -- the owner lands a CLOUD Claude session's work on the server's main.
#
# Owner 2026-10-01: cloud Claude sessions (credits until 2026-11-05) do the work Hermes may not do (confirming a PASS,
# default flips, preregs, scorers, sim code, packet specs). A cloud session can reach only GitHub, so it pushes a
# `claude/<topic>` branch there; the homelab publisher mirrors those branches into a read-only bare repo the hermes
# container can read ($CLAUDE_IMPORT_REPO). The OWNER runs this during a check-in, with the sha the cloud session
# reported, as uid 99 with SIM_CLAUDE_REVIEW=1 (docs/CLOUD_CLAUDE_SESSIONS.md has the exact command):
#   1. runs in the main checkout, on main, with a clean index;
#   2. fetches the branch and refuses unless it is EXACTLY the reported sha (a later push cannot swap the content);
#   3. prints what it brings in, merges with --no-ff --no-commit and concludes with `git commit`, so every gate runs
#      on the merge result (the cloud sandbox cannot run the gates that need server data; this is where they run).
#      A conflict or a failing gate aborts and leaves main as it was.
# Without SIM_CLAUDE_REVIEW=1 the uid-99 pre-commit backstop refuses any non-routine change, so the driver agent
# cannot use this to land reserved work. Exit 0 = landed; 2 = refused (reason printed); else a git error.
set -uo pipefail
refuse() { echo "⛔ land_claude_branch: $*" >&2; exit 2; }

b=${1:-} want=${2:-}
[[ "$b" =~ ^claude/[A-Za-z0-9._-]+$ ]] || refuse "usage: land_claude_branch.sh claude/<topic> <sha>"
[[ "$want" =~ ^[0-9a-f]{7,40}$ ]] || refuse "give the sha the cloud session reported (7-40 hex)"
REPO=${CLAUDE_IMPORT_REPO:-/opt/host-status/claude-import.git}
[[ "$REPO" =~ ^/[A-Za-z0-9._/-]+$ ]] || refuse "CLAUDE_IMPORT_REPO must be a plain absolute path"
ROOT=$(git rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1
[ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ] || refuse "run in the main checkout, not a worktree"
[ "$(git branch --show-current)" = main ] || refuse "the main checkout is not on main"
git diff --cached --quiet || refuse "the index is not clean; commit or unstage first"
[ -f .git/MERGE_HEAD ] && refuse "a merge is already in progress"

ref="refs/claude-import/${b#claude/}"
# The mirror is owned by another user on purpose (read-only for uid 99). For a local-path fetch git starts
# upload-pack with a cleaned environment, so `-c safe.directory` must be repeated inside --upload-pack.
git -c safe.directory="$REPO" fetch -q --no-tags --upload-pack="git -c safe.directory='$REPO' upload-pack" \
  "$REPO" "+refs/heads/$b:$ref" || refuse "could not fetch $b from $REPO"
full=$(git rev-parse "$ref^{commit}") || exit 1
[[ "$full" == "$want"* ]] || refuse "$b is at ${full:0:12}, not the reported $want; ask the cloud session which sha it pushed"
git merge-base --is-ancestor "$full" HEAD && { echo "$b @ ${full:0:9} is already on main"; exit 0; }

echo "landing $b @ ${full:0:9} (merge base $(git merge-base HEAD "$full" | cut -c1-9)):"
git diff --stat=120 "HEAD...$full" | tail -25
if ! git merge --no-ff --no-commit "$full" >/dev/null; then
  git merge --abort 2>/dev/null
  refuse "$b conflicts with main; merge aborted, main unchanged (ask a cloud session to merge main into $b and push again)"
fi
if ! git commit -q -m "Merge $b @ ${full:0:9} (land_claude_branch.sh: cloud Claude work landed by the owner)"; then
  git merge --abort 2>/dev/null
  refuse "the gates refused $b; merge aborted, main unchanged (the gate output above says why)"
fi
echo "landed $b @ ${full:0:9} on main as $(git rev-parse --short HEAD)"
