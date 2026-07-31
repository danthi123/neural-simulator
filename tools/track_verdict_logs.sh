#!/usr/bin/env bash
# Track the run logs that carry a VERDICT, past the blanket `*.log` ignore.
#
# WHY (2026-07-31, owner decision). `research/findings/raw/` holds ~3,960 run logs and a blanket `*.log`
# rule in .gitignore keeps every one of them out of git. That means HALF THE FAILURE RECORD has no backup —
# on a machine whose only other copy (the E: drive) was wiped during the Linux migration. It is not a
# theoretical exposure: four logs recovered today from Windows-mangled filenames were the SOLE copies of
# their runs, and one carried `MOAT PASS = True`.
#
# Tracking all 3,960 would commit stdout noise. Tracking NONE loses conclusions. So the rule is content-based:
# a log that states an OUTCOME is part of the record and gets committed; a log that is pure progress output
# stays ignored. Measured at adoption: 1,419 of 3,963 logs carry a marker, 26 MB total — negligible against
# a 13 GB pack, and it is exactly the half that says what happened.
#
# .gitignore cannot express "ignore unless the contents say X", so this is a script rather than a rule.
# Run it after a batch of runs lands, then commit. Idempotent: already-tracked files are skipped by git.
#
#   bash tools/track_verdict_logs.sh          # report what WOULD be added
#   bash tools/track_verdict_logs.sh --add    # stage them
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# A VERDICT marker: the run said what it concluded. Deliberately narrow — a log merely mentioning "GO" in
# passing does not qualify; these are the shapes our runners actually print as their conclusion line.
MARKERS='MOAT PASS|^\[RESULTS\]|=> *(GO|⛔ *NO-GO|NO-GO)|VERDICT:|⛔ EVERY ARM|HONEST NEGATIVE|6-seed GO'

mapfile -t HITS < <(grep -rlE "$MARKERS" --include='*.log' research/findings/raw/ 2>/dev/null | sort)
if [ "${#HITS[@]}" -eq 0 ]; then echo "no verdict-carrying logs found"; exit 0; fi

UNTRACKED=()
for f in "${HITS[@]}"; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || UNTRACKED+=("$f")
done

printf 'verdict-carrying logs : %d\n' "${#HITS[@]}"
printf 'already tracked       : %d\n' $(( ${#HITS[@]} - ${#UNTRACKED[@]} ))
printf 'NOT yet tracked       : %d\n' "${#UNTRACKED[@]}"

if [ "${#UNTRACKED[@]}" -eq 0 ]; then echo "=> nothing to do; the verdict half of the record is backed up"; exit 0; fi
printf '   %s\n' "${UNTRACKED[@]:0:5}"
[ "${#UNTRACKED[@]}" -gt 5 ] && printf '   ... and %d more\n' $(( ${#UNTRACKED[@]} - 5 ))

if [ "${1:-}" = "--add" ]; then
  # -f is REQUIRED: these are ignored by the blanket rule, and that rule stays for the noisy majority.
  printf '%s\0' "${UNTRACKED[@]}" | xargs -0 -n 200 git add -f
  echo "=> staged ${#UNTRACKED[@]} log(s). Commit them, then push both remotes."
else
  echo "=> re-run with --add to stage them"
fi
