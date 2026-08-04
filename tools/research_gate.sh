#!/usr/bin/env bash
# research_gate.sh — did you actually consult a PRIMARY SOURCE, or only our own findings?
#
# WHY (2026-07-29, owner caught this class for the SECOND time; first was 2026-07-06 →
# memory feedback_read_sources_in_depth_not_skim "actually READ the section"). The failure is not
# skipping research — it is running a RAG query, getting `(finding)` hits, reading THOSE, and
# stopping. Our findings docs summarise primary sources in one line; a one-line summary is how
# "BTSP (Bittner & Magee 2017)" got cited all session without the paper being opened, and how a
# whole session on PLACE FIELDS never opened O'Keefe-Nadel — which, when finally read, produced a
# mechanism, a confirmed prediction, and TWO corrections to already-committed conclusions.
#
#   bash tools/research_gate.sh "<question>"
#
# FIRST VERSION WAS WRONG, and the way it was wrong is the actual lesson. I built it to fail when the
# RAG returned NO primary source -- then tested it on the exact query that had failed me, and it
# PASSED: that query returned 3 primary hits, including a catalog entry naming
# "O&N Ch 4.7 (pp. 190-217)" -- the precise chapter I needed, sitting in output I had already seen.
# The corpus was working. I read hit [1] (a finding), acted on it, and never looked at hits [2]-[5].
#
# So the check is NOT "did the RAG surface a source" but "was the source PUT IN FRONT OF YOU rather
# than left at position 4 of 5". This version re-prints every primary-source hit at the END, after
# the raw results, with the canonical path and a ready-to-run read command -- the same trick
# device_check.sh uses, because a finding buried mid-scroll is a finding you skip.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
Q="${1:?usage: research_gate.sh \"<question>\"}"
N="${2:-8}"
OUT=$(mktemp)
# THE BLENDED QUERY ALONE MADE THIS GATE UNSATISFIABLE (defect found 2026-07-30, by the gate firing at me).
# An un-scoped query reranks our OWN findings to the top and crowds the primary corpora out of the top-N
# entirely, so PRIM=0 regardless of how much source reading has actually been done. The gate then printed
# "re-query the primary corpora explicitly" and exited 1 -- advice it never took ITSELF, so running it again
# returned the same zero forever. A check that cannot be satisfied by doing the right thing is worse than no
# check: it trains you to ignore the alert (rule 8), which is precisely the failure it exists to prevent. It
# also inverted the file's own stated purpose -- "was the source PUT IN FRONT OF YOU" -- by leaving the primary
# corpora unqueried. Proven concretely: the blended query returned 0 primary hits for a question whose
# --corpus kandel query returned Figure 10-15, the passage that reframed the whole gap#5 residual.
# So the gate now runs the per-corpus queries ITSELF and counts across all of them.
bash "$ROOT/tools/rag/search.sh" "$Q" "$N" 2>/dev/null | tee "$OUT"
for C in kandel catalog paper; do
  echo
  echo "────── --corpus $C (the primary corpora, queried directly so they cannot be crowded out) ──────"
  bash "$ROOT/tools/rag/search.sh" "$Q" 4 --corpus "$C" 2>/dev/null | tee -a "$OUT"
done
echo
# RELEVANCE FLOOR, not a hit count (defect found 2026-07-31 by the codebase bug hunt, in THIS file, hours after
# I "fixed" it). Counting primary hits made the gate UNFAILABLE: the RAG returns its top-N for ANY string, so the
# nonsense query "purple monkey dishwasher quantum bicycle" scored 18 primary hits and PASSED. That is the same
# defect as the original -- a check that cannot discriminate -- merely inverted: it went from never passing to
# never failing, and both are worthless. Neither state can be detected by running the gate on a question you
# believe in; only a NEGATIVE control exposes it.
# The reranker score DOES discriminate: on a real question the best primary hits score about -2.4 (kandel),
# -1.7 (catalog), +0.6 (paper); on the nonsense control they collapse to -6.7, -10.2, -7.9. So require at least
# one primary hit at or above a floor, rather than merely present.
FLOOR="${RESEARCH_GATE_FLOOR:--5.0}"
PRIM=$(awk -v floor="$FLOOR" '
  /\((kandel|paper|catalog|textbook)\)/ {
    for (i = 1; i <= NF; i++) if ($i ~ /^-?[0-9]+\.[0-9]+$/) { if ($i + 0 >= floor + 0) n++; break }
  } END { print n + 0 }' "$OUT")
PRIM_ANY=$(grep -cE '\((kandel|paper|catalog|textbook)\)' "$OUT" || true)
OURS=$(grep -cE '\((finding|plan|doc)\)' "$OUT" || true)
echo "──────────────────────────────────────────────────────────────────────"
echo "  primary-source hits ABOVE the relevance floor ($FLOOR): $PRIM   (of $PRIM_ANY returned)"
echo "  our-own-writing hits (finding/plan/doc):             $OURS"
if [ "$PRIM" -gt 0 ]; then
  echo
  echo "  📖 PRIMARY SOURCES SURFACED BY THIS QUERY — READ THESE, NOT JUST OUR FINDINGS:"
  grep -E '\((kandel|paper|catalog|textbook)\)' "$OUT" | sed 's/^/     /'
  echo
  echo "     Canonical copies (single-column, grep-clean, NO -a needed):"
  ls -d ~/Projects/sim-catalog/references/textbooks/*/ 2>/dev/null | sed 's/^/       /'
  echo
  echo "     Read a passage with:  grep -n -A15 \"<phrase>\" <that path>/*.txt"
  echo "     ⚠️  A catalog hit often names the exact chapter/pages (e.g. \"O&N Ch 4.7, pp. 190-217\")."
  echo "         That pointer IS the assignment -- open it before building."
fi
if [ "$PRIM" -eq 0 ]; then
  echo
  [ "${PRIM_ANY:-0}" -gt 0 ] && echo "  ⚠️  $PRIM_ANY primary hit(s) returned but ALL scored below $FLOOR — the RAG returns top-N for any"
  [ "${PRIM_ANY:-0}" -gt 0 ] && echo "     string, so these are NOISE, not evidence. Rephrase the question toward the biology."
  echo "  ⛔ NO RELEVANT PRIMARY SOURCE IN THESE RESULTS. Our findings cite sources in ONE LINE; that is not"
  echo "     reading them. Re-query the primary corpora explicitly before building:"
  echo
  echo "       .venv-rag/bin/python tools/rag/rag_search.py \"$Q\" 8 --corpus kandel"
  echo "       .venv-rag/bin/python tools/rag/rag_search.py \"$Q\" 8 --corpus catalog"
  echo "       .venv-rag/bin/python tools/rag/rag_search.py \"$Q\" 8 --corpus paper"
  echo
  echo "  Then READ the section in the CANONICAL copy under:"
  echo "       ~/Projects/sim-catalog/references/textbooks/<name>/*.txt      (single-column, greps clean)"
  echo "     NOT the two-column ISO-8859 WIP extractions in .catalog-work/ (those need grep -a)."
  touch "$ROOT/research/.last_research_gate" "$ROOT/research/.last_research_gate_empty"
  rm -f "$OUT"; exit 1
fi
echo "  ✔ a primary source is represented — READ IT (a rerank hit is a pointer, not a paraphrase)."
rm -f "$ROOT/research/.last_research_gate_empty"
touch "$ROOT/research/.last_research_gate"   # lets tools/workflow_check.sh verify this actually ran
rm -f "$OUT"; exit 0
