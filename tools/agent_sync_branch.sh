#!/usr/bin/env bash
# agent_sync_branch.sh research/<branch> -- the ONE way the driver agent (Hermes) brings main INTO its own branch.
#
# Owner 2026-10-01 ("resolve that new gap"): Hermes may not run `git merge`, so every fix landed on main left older
# research branches carrying the old copy (that day a stale tools/biology_check.py blocked every commit on six of
# them, and pool provisioning refuses a revision that does not contain published main). The direction is fixed:
# main -> research/<branch>. It never writes main and never pushes.
#   1. the branch must be a local research/* branch; main is pinned to ONE commit;
#   2. it runs in the branch's own worktree (or a temporary one it removes afterwards); files you had STAGED are
#      unstaged for the merge and staged again afterwards, whatever happens;
#   3. it merges with --no-ff --no-commit and concludes with `git commit`, so every gate runs on the merge result;
#   4. it resolves a conflict ONLY in two coordination files: research/coordination/LOCAL_LLM_RUNBOOK.md takes main's
#      copy (main's runbook is the live one) and research/FAILURE_LOG.md keeps both sides' lines. Any other
#      conflict, a local change the merge would overwrite, or a failing gate aborts and leaves the branch as it was.
# Exit 0 = merged, or already contains main; 2 = refused (reason printed; report it as NEEDS-CLAUDE); else git error.
set -uo pipefail
refuse() { echo "⛔ agent_sync_branch: $*" >&2; exit 2; }

b=${1:-}
[[ "$b" =~ ^research/[A-Za-z0-9._/-]+$ ]] || refuse "usage: agent_sync_branch.sh research/<branch>"
case "$b" in *..*|*/.*|*.lock|*/|*//*) refuse "bad branch name $b" ;; esac
git rev-parse -q --verify "refs/heads/$b^{commit}" >/dev/null || refuse "no local branch $b"
main=$(git rev-parse -q --verify "refs/heads/main^{commit}") || refuse "no local main"
if git merge-base --is-ancestor "$main" "refs/heads/$b"; then
  echo "$b already contains main @ ${main:0:9}"; exit 0
fi
top=$(cd "$(git rev-parse --git-common-dir)/.." && pwd) || exit 1

wt=$(git worktree list --porcelain | awk -v ref="branch refs/heads/$b" '/^worktree /{w=substr($0,10)} $0==ref{print w}')
tmp="" staged="" merging=0
finish() {
  if [ "$merging" = 1 ]; then git -C "$wt" merge --abort 2>/dev/null; fi
  if [ -n "$staged" ] && [ -s "$staged" ]; then
    git -C "$wt" add --pathspec-from-file="$staged" --pathspec-file-nul \
      || echo "⚠ agent_sync_branch: could not restage every file listed in $staged; restage them by hand" >&2
  fi
  [ -n "$staged" ] && rm -f "$staged"
  if [ -n "$tmp" ]; then git -C "$top" worktree remove "$tmp" || echo "⚠ agent_sync_branch: left $tmp" >&2; fi
}
trap finish EXIT
if [ -z "$wt" ]; then
  tmp="$top/.claude/worktrees/_sync-$(printf '%s' "${b#research/}" | tr / -)"
  [ -e "$tmp" ] && { t=$tmp; tmp=""; refuse "$t already exists; remove it first"; }
  git -C "$top" worktree add -q "$tmp" "$b" || { tmp=""; refuse "could not create a worktree for $b"; }
  wt=$tmp
fi
cd "$wt" || exit 1
[ "$(git branch --show-current)" = "$b" ] || refuse "$wt is not on $b"
[ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ] && refuse "a merge is already in progress in $wt"

staged=$(mktemp) || exit 1
git diff --cached --name-only --no-renames -z > "$staged" || exit 1   # a rename = its delete + its add
if [ -s "$staged" ]; then
  both=$(comm -12 <(tr '\0' '\n' < "$staged" | sort) <(git diff --name-only | sort))
  if [ -n "$both" ]; then : > "$staged"; refuse "these files have both staged and unstaged edits; stage or drop the unstaged part first: $both"; fi
  git reset -q || exit 1
fi

merging=1
if ! git merge -q --no-ff --no-commit "$main" >/dev/null 2>&1; then
  mapfile -d '' -t conflicted < <(git diff --name-only --diff-filter=U -z)
  if [ ${#conflicted[@]} -eq 0 ]; then
    merging=0; refuse "git refused to merge main into $b (a local file would be overwritten?); nothing changed"
  fi
  bad=()
  for f in "${conflicted[@]}"; do
    case "$f" in
      research/coordination/LOCAL_LLM_RUNBOOK.md)
        git checkout --theirs -- "$f" && git add -- "$f" || bad+=("$f") ;;
      research/FAILURE_LOG.md)
        if git cat-file -e ":2:$f" 2>/dev/null && git cat-file -e ":3:$f" 2>/dev/null; then
          base=$(mktemp); ours=$(mktemp); theirs=$(mktemp)
          { git show ":1:$f" 2>/dev/null || true; } > "$base"
          git show ":2:$f" > "$ours" && git show ":3:$f" > "$theirs" \
            && git merge-file -p --union "$ours" "$base" "$theirs" > "$f" && git add -- "$f" || bad+=("$f")
          rm -f "$base" "$ours" "$theirs"
        else bad+=("$f"); fi ;;
      *) bad+=("$f") ;;
    esac
  done
  [ ${#bad[@]} -eq 0 ] || refuse "merging main into $b conflicts in: ${bad[*]}; aborted, branch unchanged"
fi
if ! git commit -q -m "Merge main @ ${main:0:9} into $b (agent_sync_branch.sh)"; then
  refuse "the gates refused the merge of main into $b; aborted, branch unchanged"
fi
merging=0
echo "synced $b: merged main @ ${main:0:9} as $(git rev-parse --short HEAD)"
