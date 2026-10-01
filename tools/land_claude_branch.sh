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
#   Before the merge it clears two kinds of stray state the live checkout carries (owner 2026-10-01: the first two
#   landings of claude/session1-backlog failed on these before any gate ran):
#   a. UNTRACKED files the branch ADDS (results synced into the checkout before their branch landed, e.g. 377 iw2 raw
#      files). Byte-identical to the branch's blob: moved aside to $LAND_BACKUP_ROOT/land-backup-<UTC stamp>/<path>
#      (default /opt/data/cache/scratch). Any that DIFFER: refuse, listing them, and touch nothing.
#   b. Uncommitted APPENDED lines in the union-merged run ledgers research/findings/raw/_provenance/*.jsonl (running
#      jobs keep appending): the appended tail is saved (also into the backup dir), the file restored to HEAD, the merge
#      made, and the tail re-appended UNSTAGED -- whatever the outcome. A ledger change that is not a pure append:
#      refuse. On any refusal after (a), the moved files go back, so main's checkout is left as it was.
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

BK="${LAND_BACKUP_ROOT:-/opt/data/cache/scratch}/land-backup-$(date -u +%Y%m%dT%H%M%SZ)"
moved=() ledgers=() ledger_n=() landed=0
cleanup() {   # runs on EVERY exit from here on: re-append ledger tails always; put moved files back unless landed
  local f p
  local i
  for i in ${ledgers[@]+"${!ledgers[@]}"}; do      # the offset is the PRE-merge HEAD length, saved at set-aside
    if [ -e "${ledgers[i]}" ]; then
      tail -c +"$(( ledger_n[i] + 1 ))" "$BK/provenance/${ledgers[i]}" >> "${ledgers[i]}"
    else
      cp "$BK/provenance/${ledgers[i]}" "${ledgers[i]}"        # the restore never happened: put it back whole
    fi
  done
  if [ "$landed" != 1 ]; then
    for p in ${moved[@]+"${moved[@]}"}; do
      [ -e "$p" ] || { mkdir -p "$(dirname "$p")"; mv "$BK/files/$p" "$p"; }
    done
    if [ ${#moved[@]} -gt 0 ]; then echo "   put the ${#moved[@]} moved file(s) back" >&2; fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP     # a signal still runs cleanup (the ledger tails go back whatever happens)

# (a) untracked files the branch adds: identical -> moved aside; any different -> refuse, nothing touched
same=() differ=()
while IFS= read -r -d '' p; do
  { [ -e "$p" ] || [ -L "$p" ]; } || continue
  git ls-files --error-unmatch -- "$p" >/dev/null 2>&1 && continue      # tracked on main: an ordinary merge
  if [ -f "$p" ] && [ "$(git hash-object -- "$p")" = "$(git rev-parse "$full:$p")" ]; then
    same+=("$p")
  else
    differ+=("$p")
  fi
done < <(git diff -z --name-only --diff-filter=A "HEAD...$full")
if [ ${#differ[@]} -gt 0 ]; then
  printf '   differs: %s\n' "${differ[@]:0:40}" >&2
  [ ${#differ[@]} -gt 40 ] && echo "   ... and $(( ${#differ[@]} - 40 )) more" >&2
  refuse "${#differ[@]} untracked file(s) in main's checkout would be overwritten by $b with DIFFERENT content; nothing moved, main unchanged (move or delete them, then land again)"
fi

# (b) run ledgers: only pure appends may be set aside. Each modified ledger is MOVED whole into the backup dir (a job
# still appending through an open handle keeps writing to that moved copy, so no line is lost), HEAD's version is
# restored, and at exit everything past HEAD's length in the moved copy is appended back, unstaged.
led=() led_n=()
for f in $(git diff --name-only -- 'research/findings/raw/_provenance/*.jsonl'); do
  n=$(git cat-file -s "HEAD:$f" 2>/dev/null) || refuse "$f is modified but not in HEAD; resolve it by hand"
  if [ ! -f "$f" ] || ! cmp -s <(git cat-file blob "HEAD:$f") <(head -c "$n" "$f"); then
    refuse "$f has uncommitted changes that are not pure appends; resolve it by hand (main unchanged)"
  fi
  led+=("$f"); led_n+=("$n")
done
for i in ${led[@]+"${!led[@]}"}; do
  f=${led[i]}
  mkdir -p "$BK/provenance/$(dirname "$f")" && mv "$f" "$BK/provenance/$f" || refuse "could not set $f aside"
  ledgers+=("$f"); ledger_n+=("${led_n[i]}")
  git checkout -q -- "$f" || refuse "could not restore $f from HEAD"
  echo "   $f: $(( $(wc -c < "$BK/provenance/$f") - $(wc -c < "$f") )) appended byte(s) set aside; re-appended after the merge"
done

if [ ${#same[@]} -gt 0 ]; then
  for p in ${same[@]+"${same[@]}"}; do
    mkdir -p "$BK/files/$(dirname "$p")" && mv "$p" "$BK/files/$p" && moved+=("$p")
  done
  echo "   moved ${#moved[@]} untracked file(s) identical to $b's aside to $BK/files"
fi
if ! git merge --no-ff --no-commit "$full" >/dev/null; then
  git diff --name-only --diff-filter=U | sed 's/^/   conflict: /' >&2
  git merge --abort 2>/dev/null
  refuse "$b conflicts with main; merge aborted, main unchanged (ask a cloud session to merge main into $b and push again)"
fi
if ! git commit -q -m "Merge $b @ ${full:0:9} (land_claude_branch.sh: cloud Claude work landed by the owner)"; then
  git merge --abort 2>/dev/null
  refuse "the gates refused $b; merge aborted, main unchanged (the gate output above says why)"
fi
landed=1
echo "landed $b @ ${full:0:9} on main as $(git rev-parse --short HEAD)"
