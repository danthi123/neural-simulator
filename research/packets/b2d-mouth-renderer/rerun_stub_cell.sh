#!/usr/bin/env bash
# rerun_stub_cell.sh <seed> <faculty> <node> "<cause>" [--dry-run]
# B2d A1.4 (a)+(b) for ONE stub-arm (b2d0929) cell the scorer lists under `not_defined`, exactly as the
# pre-registration's Step 7 writes it:
#   (a) move attempt 1 aside, in the primary checkout AND on the first host (so pool_sync never brings it back), to
#       research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929/s<seed>/<faculty>/;
#   (b) ONE re-run by direct ssh on <node>, a node provisioned with P that is NOT the first host, with the job line
#       unchanged from `cd` to `--out` and `bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 --` put in front of
#       its `env`; logged as one row of research/coordination/b2d_reruns.tsv (the file tools/b2d_score.py --reruns
#       reads: first_host / rerun_host decide FAIL = class C on two different hosts).
# Refuses: a second re-run of the same cell (one per cell), <node> equal to the first host, a cell with no job line, a
# node without ~/derisk-pool/revisions/<P>, or a cell whose first attempt is still running on its node.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SEED=${1:?seed}; FAC=${2:?faculty}; NODE=${3:?node}; CAUSE=${4:?cause text from b2d_score not_defined}; DRY=${5:-}
[ -n "${PIN_P:-}" ] || { echo "REFUSED: no PIN_P recorded" >&2; exit 2; }
LEDGER=$W/research/coordination/b2d_reruns.tsv
CELL="b2d0929/s$SEED/$FAC"
OUT="$SH/b2d0929/s$SEED/$FAC/lb.json"
LINE=$(grep -F -- "--out $OUT" "$W/research/coordination/b2d_jobs.txt" | head -1)
[ -n "$LINE" ] || { echo "REFUSED: no job line for $OUT in b2d_jobs.txt" >&2; exit 2; }
case "$LINE" in "cd ~/derisk-pool/revisions/$PIN_P && "*) ;; *) echo "REFUSED: job line not pinned to $PIN_P" >&2; exit 2 ;; esac
if [ -f "$LEDGER" ] && awk -F'\t' -v c="$CELL" '$1==c {f=1} END {exit !f}' "$LEDGER"; then
  echo "REFUSED: $CELL already has a re-run in $(basename "$LEDGER") (one per cell, A1.4 b)" >&2; exit 2; fi
FIRST=$(grep -E "^\[pool-dispatch\] \S+ \S+ <- .*--out $OUT( |$)" "$P/research/queue/dispatch.log" 2>/dev/null \
        | tail -1 | awk '{print $3}')
[ -n "$FIRST" ] || { echo "REFUSED: no dispatch record for $OUT (a never-ran line is re-QUEUED instead, A1.4 c)" >&2; exit 2; }
[ "$FIRST" != "$NODE" ] || { echo "REFUSED: $NODE is the first host of $CELL; pick another P-provisioned node" >&2; exit 2; }
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -F "$SSHCFG" "$NODE" "test -d ~/derisk-pool/revisions/$PIN_P" \
  || { echo "REFUSED: $NODE unreachable or not provisioned with $PIN_P" >&2; exit 2; }
if ssh -n -o BatchMode=yes -o ConnectTimeout=10 -F "$SSHCFG" "$FIRST" "pgrep -f -- '--out $OUT' >/dev/null" 2>/dev/null; then
  echo "REFUSED: the first attempt of $CELL is still running on $FIRST" >&2; exit 2; fi
RUN=${LINE#"cd ~/derisk-pool/revisions/$PIN_P && "}
RUN=${RUN/ && env / \&\& bash tools/mem_ok.sh 8 2 \&\& bash tools/memcap.sh 12 -- env }
case "$RUN" in *"bash tools/memcap.sh 12 -- env "*) ;; *) echo "REFUSED: could not insert the memory guard" >&2; exit 2 ;; esac
A1="research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929/s$SEED"
LOG="research/findings/raw/_load_bearing/_b2d0929_reruns/s${SEED}_$FAC.log"
echo "cell $CELL: first host $FIRST -> re-run on $NODE"
echo "  cmd: cd ~/derisk-pool/revisions/$PIN_P && $RUN"
[ "$DRY" = "--dry-run" ] && { echo "(dry run: nothing moved, logged or started)"; exit 0; }
# (a) move attempt 1 aside: primary checkout, then the first host (best effort there; E-stranded cells are handled by
#     the prereg's stranded-results rule, which this script does not touch)
if [ -d "$P/$SH/b2d0929/s$SEED/$FAC" ]; then mkdir -p "$P/$A1" && mv "$P/$SH/b2d0929/s$SEED/$FAC" "$P/$A1/"; fi
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -F "$SSHCFG" "$FIRST" \
  "cd ~/derisk-pool/revisions/$PIN_P 2>/dev/null && if [ -d $SH/b2d0929/s$SEED/$FAC ]; then mkdir -p $A1 && mv $SH/b2d0929/s$SEED/$FAC $A1/; fi" \
  || echo "  (first host $FIRST unreachable: its copy stays; do NOT start $FIRST before this re-run lands -- prereg stranded-results rule)"
# (b) log, then start the re-run detached on the node
[ -f "$LEDGER" ] || printf 'cell\tfirst_host\tfirst_cause_and_verdict\trerun_host\trerun_command\trerun_start_time\trerun_result\n' > "$LEDGER"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$CELL" "$FIRST" "$CAUSE" "$NODE" "cd ~/derisk-pool/revisions/$PIN_P && $RUN" \
  "$(date -u +%FT%TZ)" "pending" >> "$LEDGER"
ssh -n -o BatchMode=yes -F "$SSHCFG" "$NODE" \
  "cd ~/derisk-pool/revisions/$PIN_P && mkdir -p $(dirname "$LOG") && nohup bash -c $(printf %q "$RUN") > $LOG 2>&1 < /dev/null &" \
  && echo "started on $NODE; the result lands via pool_sync at $P/$OUT; log on the node: $LOG"
