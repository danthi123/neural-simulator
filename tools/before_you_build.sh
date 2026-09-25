#!/usr/bin/env bash
# ONE COMMAND to run BEFORE the first lever against any defect. Costs ~30 seconds.
#
# WHY: on 2026-07-28 a 497-line research gate for the identical defect, on the identical substrate,
# with a ranked 6-mechanism ladder, was TWO DAYS OLD — and a full day was spent re-deriving it. Later
# the same session built a threshold-based fix AFTER its own findings doc had already measured
# thresholds inert (0/27 winners). Two different knowledge failures, one cause: not asking.
#
#   bash tools/before_you_build.sh "the slot competition ignores the cue"
set -uo pipefail

# RESOLVE LOCATIONS BEFORE ANY `cd` (research/corpus-check-shared-log, 2026-09-25). A prior propagation fix
# put a relative `dirname "$0"` lookup AFTER a `cd`, so a call made with a relative path from a different cwd
# silently resolved the wrong tree and logged nothing. Both of these are computed from argv0/$PWD as this
# process actually started, once, before anything below is allowed to change directory.
_ORIG_CWD="$PWD"
_SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

cd "$_SCRIPT_DIR/.."
Q="${*:-}"
[ -z "$Q" ] && { echo "usage: bash tools/before_you_build.sh \"<the defect in one line>\""; exit 2; }

echo "════ 1. HAS THIS ALREADY BEEN SCOPED / TRIED / REFUTED? (our own findings) ════"
bash tools/rag/search.sh "$Q" 5 --corpus finding 2>/dev/null | grep -vi "^LLM is" || echo "  (rag unavailable)"

echo
echo "════ 2. IS THERE A RESEARCH GATE / SCOPE DOC ALREADY? ════"
# RECURSIVE (2026-07-31). This step hunts for an existing scope doc, and 24 of the 42 findings that a FLAT
# glob could not see are named `_*_scoping.md` -- they sit in research/findings/raw/. The check built to find
# prior scoping work was blind to most of it, which is precisely how a scoped question gets re-derived.
find research/findings -name '*research-gate*.md' -o -name '*scop*.md' 2>/dev/null \
  | xargs -r ls -t 2>/dev/null | head -5 | sed 's/^/  /' || true

echo
echo "════ 3. THIS ARC'S OWN EXCLUSIONS — things already measured NOT to be the cause ════"
echo "  (the corpus check covers PRIOR findings; it does NOT cover the current arc. Read these.)"
grep -ohE "^\*\*⛔[^*]{0,110}|REFUTED[^.]{0,90}|EXCLUDED[^.]{0,90}" \
  $(find research/findings -name '*.md' | xargs -r ls -t 2>/dev/null | head -3) 2>/dev/null \
  | sort -u | head -12 | sed 's/^/  /'

echo
echo "════ 4. LEVER COUNT — >=2 levers against ONE defect means the research gate FIRES ════"
echo "  Cheapness of the next test is NOT an exemption (6 levers / ~4 GPU-h were spent on one defect"
echo "  without the gate ever subjectively firing). Count them in the findings doc."
echo
echo "Proceed only after reading what the above surfaced — a hit is a POINTER, not a paraphrase."

# RECORD THAT THIS CHECK HAPPENED (2026-07-31). Until now this script was purely advisory: it printed the
# priors and nothing bound running it to launching anything. On 2026-07-31 a nine-hour eight-cell crux was
# launched against a question whose answer was banked three weeks earlier with its root cause named, and
# this script -- which returns those four priors in 0.63 s -- was not run until after the write-up.
# The heartbeat flagged the missing check ~15 times that day and was read past every time, so REPORTING is
# demonstrably insufficient for this class. The record below is what `gates/corpus_check_required` reads.
#
# ONE SHARED LOG (research/corpus-check-shared-log, 2026-09-25). The original version of this record wrote to
# `$PWD/research/queue/.corpus_checks.jsonl` -- a path INSIDE whichever worktree happened to run the check, so
# a check made in one worktree was invisible to a run launched from another (or from the pool/GPU lanes, which
# operate their own checkouts). `git rev-parse --git-common-dir` names the ONE `.git` directory every worktree
# of this repo shares, regardless of which checkout invokes it -- so writing there, instead of under any
# worktree's own `research/`, makes the log genuinely shared without needing a new sync mechanism. Resolved
# from `$_SCRIPT_DIR` (this script's OWN real location), not `$PWD`, so the answer does not depend on the
# caller's cwd either.
if [ -n "${SIM_CORPUS_CHECK_LOG:-}" ]; then
  _CC_LOG="$SIM_CORPUS_CHECK_LOG"
else
  _CC_COMMON_DIR="$(git -C "$_SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  if [ -z "$_CC_COMMON_DIR" ]; then
    echo "ERROR: could not resolve the git common dir from $_SCRIPT_DIR -- refusing to log a corpus check" \
         "nowhere (set SIM_CORPUS_CHECK_LOG to override)." >&2
    exit 1
  fi
  _CC_LOG="$_CC_COMMON_DIR/corpus_checks_shared.jsonl"
fi
[ -z "$_CC_LOG" ] && { echo "ERROR: resolved an empty corpus-check log path -- refusing to log nowhere." >&2; exit 1; }
mkdir -p "$(dirname "$_CC_LOG")"
printf '{"when": %s, "iso": "%s", "query": %s, "cwd": "%s"}\n' \
  "$(date +%s)" "$(date -Iseconds)" "$(printf '%s' "$Q" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$_ORIG_CWD" \
  >> "$_CC_LOG" 2>/dev/null || true
echo "  [recorded] corpus check logged to $_CC_LOG"
