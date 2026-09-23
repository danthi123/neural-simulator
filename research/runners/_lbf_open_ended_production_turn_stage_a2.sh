#!/usr/bin/env bash
# AMENDMENT-2 staging for the open-ended production-turn probe (lane research/open-ended-production-turn-lb, fix round
# 2026-09-23; PREREG amendment 2 = commit 4f9cc6b3e). Runs DETACHED on the local box (numpy -- the oe_routed_full
# reply-equality check against `default` is only meaningful on the SAME machine/env as the default arms), each lane
# under tools/memcap.sh and started only when tools/mem_ok.sh passes. Lanes, in launch order:
#   I  flag-off identity: 4 dumps (pre 4c141b8e8 / post 8c5d7b03a x default / oe_off) + a pre-vs-pre control, compare
#   H  default mode, host_oracle arm, seeds 42 43 44 100 101 102
#   F1 oe_routed_full (true open-ended configuration), seeds 42 43 44
#   F2 oe_routed_full, seeds 100 101 102
# then the harvest (bash research/runners/_lbf_open_ended_production_turn_harvest_a2.sh), which can be re-run.
#   bash research/runners/_lbf_open_ended_production_turn_stage_a2.sh <pre_tree> <post_tree>
# where <pre_tree>/<post_tree> are clean `git archive` extractions of 4c141b8e8 / 8c5d7b03a.
set -uo pipefail
cd "$(dirname "$0")/../.."
PRE=${1:?pre tree (git archive of 4c141b8e8)}
POST=${2:?post tree (git archive of 8c5d7b03a)}
A2=research/findings/raw/_load_bearing/_oe_production_turn/a2
LOG=$A2/logs
ID=$A2/flag_off_identity
mkdir -p "$LOG" "$ID" "$A2/default" "$A2/oe_routed_full"
# REFUSE a degraded brain: data/corpus is untracked (not in a worktree checkout or a git archive). Without it the
# one-brain XEDGE build fails and the webapp silently degrades to standalone organs, so these arms would not be
# comparable to the original arms (whose worktree had data -> /home/dant123/Projects/sim/data). Caught 2026-09-23.
for t in . "$PRE" "$POST"; do
  [ -f "$t/data/corpus/tinystories.txt" ] || { echo "[stage_a2] REFUSED: $t has no data/corpus (symlink data -> the main checkout's data/)" >&2; exit 2; }
done
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
# NOT readlink -f: resolving the venv's python symlink lands on the system /usr/bin/python3.11 and drops the venv
# (first launch 2026-09-23: every identity dump died "No module named 'fastapi'"; the compares correctly read UNDEFINED).
PYABS="$(pwd)/.venv/bin/python"
ENVS=(env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2)
PROBE=(.venv/bin/python -u -m research.runners._lbf_open_ended_production_turn_probe)
ident=research/runners/_lbf_oe_route_flag_off_identity.py

wait_mem() {  # block until starting a job of $1 GB leaves the default mem_ok safety margin
  until bash tools/mem_ok.sh "$1" >/dev/null 2>&1; do sleep 60; done
}

lane_identity() {
  for spec in "pre:$PRE:default" "post:$POST:default" "pre:$PRE:oe_off" "post:$POST:oe_off" "prectl:$PRE:default"; do
    IFS=: read -r tag tree es <<< "$spec"
    bash tools/memcap.sh 7 -- "${ENVS[@]}" "$PYABS" -u "$ident" dump --repo "$tree" --env-set "$es" \
      --out "$(pwd)/$ID/${tag}_${es}.json" >> "$LOG/identity.log" 2>&1
  done
  for es in default oe_off; do
    "$PYABS" "$ident" compare --a "$ID/pre_${es}.json" --b "$ID/post_${es}.json" --a-sha 4c141b8e8 --b-sha 8c5d7b03a \
      --out "$ID/verdict_${es}.json" >> "$LOG/identity.log" 2>&1
  done
  "$PYABS" "$ident" compare --a "$ID/pre_default.json" --b "$ID/prectl_default.json" --a-sha 4c141b8e8 \
    --b-sha 4c141b8e8 --out "$ID/verdict_control_pre_vs_pre.json" >> "$LOG/identity.log" 2>&1
}

if [ "${3:-}" = "identity-only" ]; then        # re-run just lane I (e.g. after the first launch's venv bug)
  wait_mem 5
  lane_identity
  exit 0
fi
wait_mem 5
lane_identity &
P_I=$!
sleep 180
wait_mem 6
bash tools/memcap.sh 8 -- "${ENVS[@]}" "${PROBE[@]}" --mode default --arms host_oracle \
  --seeds 42,43,44,100,101,102 --k 40 --out-dir "$A2/default" > "$LOG/H_host_oracle.log" 2>&1 < /dev/null &
P_H=$!
sleep 180
wait_mem 6
bash tools/memcap.sh 8 -- "${ENVS[@]}" "${PROBE[@]}" --mode oe_routed_full --seeds 42,43,44 --k 40 \
  --out-dir "$A2/oe_routed_full" > "$LOG/F1_oe_routed_full.log" 2>&1 < /dev/null &
P_F1=$!
sleep 180
wait_mem 6
bash tools/memcap.sh 8 -- "${ENVS[@]}" "${PROBE[@]}" --mode oe_routed_full --seeds 100,101,102 --k 40 \
  --out-dir "$A2/oe_routed_full" > "$LOG/F2_oe_routed_full.log" 2>&1 < /dev/null &
P_F2=$!
wait "$P_I" "$P_H" "$P_F1" "$P_F2"
echo "[stage_a2] all lanes finished $(date -Is)" >> "$LOG/stage.log"
bash research/runners/_lbf_open_ended_production_turn_harvest_a2.sh >> "$LOG/harvest.log" 2>&1
