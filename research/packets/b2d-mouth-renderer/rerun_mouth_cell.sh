#!/usr/bin/env bash
# rerun_mouth_cell.sh <seed> <row> "<cause>" [--dry-run]
# B2d Amendment 1 A1.2 re-run for ONE mouth-arm (b2d0929-mouth) cell that tools/b2d_mouth_score.py lists under
# `rerun_cells` (state MISSING / C-UNCONFIRMED / M-UNCONFIRMED). There is one GPU, so the re-run is a second attempt
# on the workstation: move attempt 1 aside in the export ($TREE), in $P and in $W, log one row in
# research/coordination/b2d_mouth_reruns.tsv, and queue the cell's unchanged line from the mouth job file on the GPU.
# A cell that NEVER RAN (no file at all in the export, e.g. its memory guard refused) is re-queued instead, at most twice
# (A1.4 c), with nothing moved. Refuses a second re-run of the same cell and a cell whose line is queued or running.
# CLAUDE ONLY (review 2026-09-30): it queues with tools/gpu_queue.sh add, which the local model's settings deny and
# its runbook forbids (HANDOFF step 5); the local model writes NEEDS-CLAUDE instead of running this script.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SEED=${1:?seed}; ROW=${2:?row}; CAUSE=${3:?cause text}; DRY=${4:-}
[ -n "${PIN_P:-}" ] || { echo "REFUSED: no PIN_P recorded" >&2; exit 2; }
LEDGER=$W/research/coordination/b2d_mouth_reruns.tsv
CELL="b2d0929-mouth/s$SEED/$ROW"
OUT="$SH/b2d0929-mouth/s$SEED/$ROW/lb.json"
LINE=$(grep -F -- "--out $OUT" "$K/jobs/mouth_$P7.txt" | head -1)
[ -n "$LINE" ] || { echo "REFUSED: no line for $OUT in jobs/mouth_$P7.txt" >&2; exit 2; }
if grep -qF -- "--out $OUT" "$P/research/queue/gpu.queue" 2>/dev/null || pgrep -f -- "--out $OUT" >/dev/null; then
  echo "REFUSED: $CELL is still queued or running" >&2; exit 2; fi
[ -f "$LEDGER" ] || printf 'cell\tfirst_host\tfirst_cause_and_verdict\trerun_host\trerun_command\trerun_start_time\trerun_result\n' > "$LEDGER"
# NEVER RAN (A1.4 c): the cell directory holds no file at all in the export (e.g. the job's memory guard refused before
# any arm was built). That is not an attempt: re-queue the unchanged line, at most twice, nothing to move aside.
if [ -z "$(ls -A "$TREE/$SH/b2d0929-mouth/s$SEED/$ROW" 2>/dev/null)" ]; then
  n=$(awk -F'\t' -v c="$CELL" '$1==c && $3 ~ /^never-ran/ {n++} END {print n+0}' "$LEDGER")
  [ "$n" -lt 2 ] || { echo "REFUSED: $CELL never ran and was already re-queued twice (A1.4 c): it stays MISSING" >&2; exit 2; }
  echo "cell $CELL never ran (no file in the export): re-queue $((n + 1)) of 2"
  [ "$DRY" = "--dry-run" ] && { echo "(dry run: nothing logged or queued)"; exit 0; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$CELL" "workstation" "never-ran: $CAUSE" "workstation" "$LINE" "$(date -u +%FT%TZ)" "requeued" >> "$LEDGER"
  cd "$W" && bash tools/gpu_queue.sh add "$LINE" </dev/null && echo "re-queued on the GPU"
  exit $?
fi
if awk -F'\t' -v c="$CELL" '$1==c && $3 !~ /^never-ran/ {f=1} END {exit !f}' "$LEDGER"; then
  echo "REFUSED: $CELL already re-run once (the same class C or M twice is FAIL; any other pair stays not final, so the arm's INCOMPLETE is final)" >&2; exit 2; fi
A1="research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929-mouth/s$SEED"
echo "cell $CELL: second attempt on the workstation GPU"
[ "$DRY" = "--dry-run" ] && { echo "(dry run: nothing moved, logged or queued)"; exit 0; }
for root in "$TREE" "$P" "$W"; do   # $W too: copy_results.sh may already hold attempt 1 there (rsync, no --delete)
  if [ -d "$root/$SH/b2d0929-mouth/s$SEED/$ROW" ]; then mkdir -p "$root/$A1" && rm -rf "$root/$A1/$ROW" && mv "$root/$SH/b2d0929-mouth/s$SEED/$ROW" "$root/$A1/"; fi
done
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$CELL" "workstation" "$CAUSE" "workstation" "$LINE" "$(date -u +%FT%TZ)" "pending" >> "$LEDGER"
cd "$W" && bash tools/gpu_queue.sh add "$LINE" </dev/null && echo "queued on the GPU; yield the local model (HANDOFF step 6) if it holds the card"
