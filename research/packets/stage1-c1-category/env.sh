# Stage-1 C1 category-inheritance packet: `source` this at the top of EVERY HANDOFF command block (the Bash tool keeps
# no state between calls). Sets the paths and the pins recorded in PINS.tsv (the PREREG row and the LAST GATE row).
P=/home/dant123/Projects/sim
W=$P/.claude/worktrees/stage1-c1-category
K=research/packets/stage1-c1-category
PY=$P/.venv/bin/python
BR=research/stage1-c1-category
PREREG=research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md
OUT=research/findings/raw/_stage1_c1_category/gate
JOBS=$P/research/queue/stage1_c1_category_jobs.tsv
PREREG_SHA=""; GATE_SHA=""
if [ -f "$W/$K/PINS.tsv" ]; then
  PREREG_SHA=$(awk -F'\t' '$1=="PREREG"{p=$2} END{print p}' "$W/$K/PINS.tsv")
  GATE_SHA=$(awk -F'\t' '$1=="GATE"{p=$2} END{print p}' "$W/$K/PINS.tsv")
fi
