#!/usr/bin/env bash
# AMENDMENT-2 harvest for the open-ended production-turn probe (idempotent; re-run until every aggregate reads 6 seeds).
#  1. gathers the worker JSONs the amendment-2 scorer needs into a2/<mode>/:
#       seed 42 default / oe_unfixed_taught arms  <- this checkout's committed _oe_production_turn/<mode>/ (in-sample)
#       seeds 43..102 default / oe_unfixed_taught <- the ORIGINAL staging worktree (launched 13:20 under the original
#                                                    rule; the worker protocol is unchanged, only the scorer is new)
#  2. re-scores every seed with the amendment-2 scorer, then aggregates per mode (oe_routed_full also gets the exact
#     reply-equality check against default).
#   bash research/runners/_lbf_open_ended_production_turn_harvest_a2.sh [original_worktree_root]
set -uo pipefail
cd "$(dirname "$0")/../.."
ORIG=${1:-/home/dant123/Projects/sim/.claude/worktrees/wf_6cf1082d-b06-2}
BASE=research/findings/raw/_load_bearing/_oe_production_turn
A2=$BASE/a2
SEEDS=42,43,44,100,101,102
PY=(.venv/bin/python -u -m research.runners._lbf_open_ended_production_turn_probe)
for mode in default oe_unfixed_taught; do
  mkdir -p "$A2/$mode"
  for arm in intact intact_rebuild lesion; do
    src="$BASE/$mode/${mode}_s42_${arm}.json"
    [ -f "$src" ] && cp -n "$src" "$A2/$mode/"
    [ -f "$src.prov.json" ] && cp -n "$src.prov.json" "$A2/$mode/"
    for s in 43 44 100 101 102; do
      src="$ORIG/$BASE/$mode/${mode}_s${s}_${arm}.json"
      [ -f "$src" ] && cp -n "$src" "$A2/$mode/"
      [ -f "$src.prov.json" ] && cp -n "$src.prov.json" "$A2/$mode/"
    done
  done
done
for mode in default oe_routed_full oe_unfixed_taught; do
  SIM_BACKEND=numpy "${PY[@]}" --rescore --mode "$mode" --seeds "$SEEDS" --out-dir "$A2/$mode"
  shopt -s nullglob
  V=("$A2/$mode/${mode}"_s*_verdict.json)
  shopt -u nullglob
  [ "${#V[@]}" -gt 0 ] || { echo "[harvest] $mode: no verdicts yet"; continue; }
  CMP=()
  [ "$mode" = "oe_routed_full" ] && CMP=(--compare-dir "$A2/default")
  SIM_BACKEND=numpy "${PY[@]}" --aggregate "${V[@]}" --out "$A2/${mode}_aggregate.json" "${CMP[@]}" > /dev/null
  echo "[harvest] $mode: ${#V[@]} seed verdict(s) -> $A2/${mode}_aggregate.json"
done
