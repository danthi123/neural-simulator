#!/usr/bin/env bash
# Push to BOTH remotes and VERIFY, rather than assert. Exit non-zero if either remote does not end up at HEAD.
#
# WHY (2026-07-16): the habit this replaces was
#     git push -q origin main 2>&1 | tail -1; git push -q gitea main 2>&1 | tail -1; echo pushed
# in which `echo pushed` runs UNCONDITIONALLY, `-q` suppresses the output, and `| tail -1` eats the error. It
# reports success whether or not the push happened. It was reported as "pushed both remotes" ~20 times in one
# session on nothing but faith; a `git ls-remote` audit later showed it had been true -- which is LUCK, not
# verification. Same shape as the day's other defects: a claim nothing checked.
#
# Usage: tools/push_both.sh [branch]   (default: current branch)
set -uo pipefail
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
LOCAL=$(git rev-parse "$BRANCH")
rc=0
for R in origin gitea; do
  GIT_TERMINAL_PROMPT=0 timeout 120 git push "$R" "$BRANCH" >/dev/null 2>&1
  # Ground truth: ask the REMOTE what it has. A cached remote-tracking ref would happily agree with a failed push.
  REMOTE=$(GIT_TERMINAL_PROMPT=0 timeout 60 git ls-remote "$R" "refs/heads/$BRANCH" 2>/dev/null | awk '{print $1}')
  if [ "$REMOTE" = "$LOCAL" ]; then
    echo "  OK   $R -> ${LOCAL:0:8}"
  else
    echo "  FAIL $R -> got '${REMOTE:0:8}' want '${LOCAL:0:8}'  (unreachable? auth? rejected?)"
    rc=1
  fi
done
[ $rc -eq 0 ] && echo "verified: both remotes at ${LOCAL:0:8}" || echo "PUSH NOT VERIFIED -- do not report this as pushed"
exit $rc
