#!/usr/bin/env bash
# Local rule v2 (prereg brain/research/2026-11-27-local-rule-v2-prereg.md), CPU fp32 on a pool node. One process at a time,
# resumable: each (tag, seed, arm) is one json unit written atomically; existing units are skipped. Any unit run that exits
# non-zero (OOM under memcap, a bug) STOPS this script: a crash never becomes a verdict.
#   bash run_node.sh <01b|04b> dev                 seed 7: ref (no_sleep, gradient) + 9 v2 configs (lr x tau), then the pick
#                                                  (the pick refuses unless all 11 units exist, one model, one commit)
#   bash run_node.sh <01b|04b> six <seed> [seed..] the 7 arms at the frozen pick (python reads PICK: refuses NEGATIVE-DEV,
#                                                  another model, or another commit)
#   bash run_node.sh <01b|04b> score               verdict over the six seeds (any machine; no model; checks units vs PICK)
# Order: 01b dev (pool40) -> copy pool40's dev/pick.json to pool41/42 (same path) or set PICK=<path> -> 01b six
# (42 43 44 on pool41, 100 101 102 on pool42) -> 01b score (copy both six/ dirs to one machine) -> the same with 04b.
# After tools/mem_ok.sh.
set -u
R=~/brain-lane/route_repo
SCALE=$1; PHASE=$2; shift 2
case $SCALE in
  01b) MODEL_FILE=rwkv7-g1d-0.1b-20260129-ctx8192.pth; CAP=${MEMCAP_GB:-8} ;;
  04b) MODEL_FILE=rwkv7-g1d-0.4b-20260210-ctx8192.pth; CAP=${MEMCAP_GB:-13} ;;
  *) echo "scale must be 01b or 04b" >&2; exit 2 ;;
esac
export COMMIT=${COMMIT:-$(cat $R/COMMIT 2>/dev/null)}
[ -n "$COMMIT" ] || { echo "COMMIT empty: set COMMIT or write $R/COMMIT" >&2; exit 1; }
export POOL_NODE=1 HOST=$(hostname) BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
MODEL=$BRAIN_MODEL_DIR/$MODEL_FILE
OUT=${OUT:-$HOME/brain-lane/out/local-rule-v2/$SCALE}
PICK=${PICK:-$OUT/dev/pick.json}
mkdir -p "$OUT/dev" "$OUT/six"
TIMEV=(); [ -x /usr/bin/time ] && TIMEV=(/usr/bin/time -v)   # "Maximum resident set size" in each log
cd $R || exit 1
run() {  # dir seed tag arms [extra args]; skip when every unit exists (the python also skips per unit); abort on rc != 0
  local d=$1 s=$2 tag=$3 arms=$4; shift 4
  local need=0; for a in ${arms//,/ }; do [ -s "$OUT/$d/${tag}_s${s}_${a}.json" ] || need=1; done
  [ $need = 0 ] && { echo "skip $tag s$s"; return 0; }
  echo "start $(date -u +%FT%TZ) $HOST $COMMIT $SCALE $tag s$s $arms $*" >> "$OUT/$d/done.txt"
  bash tools/memcap.sh $CAP -- ${TIMEV[@]+"${TIMEV[@]}"} "$PY" -u -m brain.experiments.local_rule_v2 run --model "$MODEL" \
    --seed "$s" --arms "$arms" --tag "$tag" --out-dir "$OUT/$d" "$@" > "$OUT/$d/${tag}_s${s}.log" 2>&1
  local rc=$?
  echo "rc $rc $(date -u +%FT%TZ) $tag s$s" >> "$OUT/$d/done.txt"
  [ $rc = 0 ] || { echo "ABORT: $tag s$s exited $rc (see $OUT/$d/${tag}_s${s}.log); fix and re-run (resumes)" >&2; exit $rc; }
}
case $PHASE in
  dev)
    run dev 7 ref no_sleep,gradient
    for lr in 0.1 0.3 1.0; do for tau in 0.9 0.99 0.999; do
      run dev 7 "lr${lr}_tau${tau}" v2 --lr $lr --tau $tau
    done; done
    "$PY" -m brain.experiments.local_rule_v2 pick --dir "$OUT/dev" --out "$OUT/dev/pick.json" || exit 1 ;;
  six)
    [ -s "$PICK" ] || { echo "no $PICK: run dev first and copy its pick.json" >&2; exit 1; }
    [ $# -gt 0 ] || { echo "six needs seeds" >&2; exit 2; }
    for s in "$@"; do
      run six "$s" six no_sleep,gradient,v2,v2_shuffled,noproj,sham,sham_tok --pick "$PICK"
    done ;;
  score)
    [ -s "$PICK" ] || { echo "no $PICK" >&2; exit 1; }
    "$PY" -m brain.experiments.local_rule_v2 score --dir "$OUT/six" --tag six --pick "$PICK" --out "$OUT/six/verdict.json" ;;
  *) echo "phase must be dev, six or score" >&2; exit 2 ;;
esac
