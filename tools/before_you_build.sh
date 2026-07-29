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
cd "$(dirname "$0")/.."
Q="${*:-}"
[ -z "$Q" ] && { echo "usage: bash tools/before_you_build.sh \"<the defect in one line>\""; exit 2; }

echo "════ 1. HAS THIS ALREADY BEEN SCOPED / TRIED / REFUTED? (our own findings) ════"
.venv-rag/bin/python tools/rag/rag_search.py "$Q" 5 --corpus finding 2>/dev/null | grep -vi "^LLM is" || echo "  (rag unavailable)"

echo
echo "════ 2. IS THERE A RESEARCH GATE / SCOPE DOC ALREADY? ════"
ls -t research/findings/*research-gate*.md research/findings/*scope*.md 2>/dev/null | head -5 | sed 's/^/  /' || true

echo
echo "════ 3. THIS ARC'S OWN EXCLUSIONS — things already measured NOT to be the cause ════"
echo "  (the corpus check covers PRIOR findings; it does NOT cover the current arc. Read these.)"
grep -ohE "^\*\*⛔[^*]{0,110}|REFUTED[^.]{0,90}|EXCLUDED[^.]{0,90}" \
  $(ls -t research/findings/*.md | head -3) 2>/dev/null | sort -u | head -12 | sed 's/^/  /'

echo
echo "════ 4. LEVER COUNT — >=2 levers against ONE defect means the research gate FIRES ════"
echo "  Cheapness of the next test is NOT an exemption (6 levers / ~4 GPU-h were spent on one defect"
echo "  without the gate ever subjectively firing). Count them in the findings doc."
echo
echo "Proceed only after reading what the above surfaced — a hit is a POINTER, not a paraphrase."
