#!/usr/bin/env bash
# deep_research.sh — the ONE command to run AT A WALL (>=2 levers against one defect), before the next lever.
#
# WHY (2026-08-09, owner-flagged recurrence — "your lack of proper research when faced with walls keeps
# happening"). before_you_build.sh does the LOCAL record check only; nothing forced the EXTERNAL-literature
# check the owner asks for, and the "≥2 levers ⇒ research fires" rule was printed, never enforced. On the
# teacher-loop forgetting wall, FIVE mechanism levers ran before any deep research — which then took one query
# each to find the project's OWN Complementary-Learning-Systems design + Phase-1.4 (103% retention) AND the
# external SOTA (PS-SNN pattern separation, EWC, van de Ven replay). gates/deep-research-at-wall now BLOCKS a
# 3rd+ lever in a lane until an external source is logged; this script does BOTH halves and records them.
#
#   bash tools/deep_research.sh "the wall in one line, e.g. teacher-loop catastrophic forgetting plateau"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="${*:-}"
[ -z "$Q" ] && { echo "usage: bash tools/deep_research.sh \"<the wall in one line>\""; exit 2; }

echo "################################################################"
echo "# DEEP RESEARCH AT A WALL — do BOTH halves, then record a source"
echo "# wall: $Q"
echo "################################################################"
echo
echo "==== HALF 1/2 — LOCAL RECORD (has the record already SOLVED or CHARACTERISED this?) ===="
bash "$ROOT/tools/before_you_build.sh" "$Q" || true
echo
echo "==== HALF 2/2 — EXTERNAL LITERATURE (the half that keeps getting skipped) ===="
cat <<'EOF'
  Run a REAL external search NOW (a bash script cannot call the MCP tools — YOU must):
    • bio-research MCP: mcp__plugin_bio-research_consensus__search  (Semantic Scholar + PubMed + Scopus + ArXiv)
      and/or mcp__plugin_bio-research_pubmed__search_articles  (neuro/bio)
    • or WebSearch / a paper PDF for the ML side (EWC, continual learning, SNN methods)
  READ the top hit(s) — a title is a pointer, not the finding (owner: read sources in depth, do not cite abstracts).
  Ask specifically: "what is the PROVEN mechanism for this class of problem?" — it is usually already named.
  THEN record what you found (a NON-EMPTY source is required):
    bash tools/record_external_search.sh "<query>" "<paper / arxiv / doi / (Author, YEAR) / 'none-found: <why>'>"
EOF
echo
echo "Only after BOTH halves + a recorded external source does gates/deep-research-at-wall clear for this lane."
