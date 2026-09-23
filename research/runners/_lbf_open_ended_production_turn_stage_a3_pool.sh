#!/usr/bin/env bash
# AMENDMENT-3 staging for the open-ended production-turn probe, on the MINI-PC POOL (never locally: the local box is
# RAM-bound and each session is one full numpy brain). PREREG amendment 3 = commit eefdd666a.
#   1. bash tools/pool_provision.sh --isolated --revision <SHA> pool41 pool42      (once)
#   2. bash research/runners/_lbf_open_ended_production_turn_stage_a3_pool.sh <FULL_SHA> [--smoke]
#   3. bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh <FULL_SHA>   (idempotent; re-run)
# One pool job = ONE session = one fresh full brain (build, 13 teach turns, K=8 asks) -> one worker JSON under
#   research/findings/raw/_load_bearing/_oe_production_turn/a3/default/default_s<seed>_<arm>_n<j>.json
# in the node's isolated revision dir. Jobs are idempotent (an existing output is kept), so a re-queue is harmless.
# ONE FULL BRAIN PER NODE (15 GB nodes): every job takes the node-level full-brain locks before starting. The D1
# affect-marker lane's lock is shared on purpose, so this lane and that one never co-reside on a node.
# --smoke: queue ONE non-governed session (seed 7, intact, n0) into a3_smoke/ to check the pipeline on a node.
set -uo pipefail
cd "$(dirname "$0")/../.."
SHA=${1:?full commit sha of the isolated pool revision}
[ ${#SHA} -eq 40 ] || { echo "need the FULL 40-char sha (the revision dir name)" >&2; exit 2; }
MODE=default
LOCKS='flock ~/derisk-pool/.d1_affect_marker_fullbrain.lock flock ~/derisk-pool/.fullbrain.lock'
ENVS='env SIM_POOL_HOST=$(hostname) SIM_BACKEND=numpy OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3'
RUN=".venv/bin/python -u -m research.runners._lbf_open_ended_production_turn_probe --a3-session --mode $MODE"
CHECKED="PREREG amendment 3 (eefdd666a) committed before this job; record (before_you_build): 2026-09-21 single-turn \
field-diff reads open-ended-generation NOT load-bearing, the synthetic distributional ruler 6-seed robust; no \
production-turn run with independent noise-stream sessions exists (the a2 default seeds 43+ were killed before \
writing any output). Primary evidence for the a3 GO."
q() {  # $1 seed, $2 arm, $3 session, $4 out-dir
  bash tools/pool_queue.sh add "cd ~/derisk-pool/revisions/$SHA && $LOCKS $ENVS $RUN --seed $1 --arm $2 --session $3 --out-dir $4" \
    --checked "$CHECKED seed $1 $2 n$3." || echo "[stage_a3] QUEUE FAILED: seed $1 $2 n$3" >&2
}
if [ "${2:-}" = "--smoke" ]; then
  q 7 intact 0 research/findings/raw/_load_bearing/_oe_production_turn/a3_smoke/$MODE
  exit 0
fi
OUT=research/findings/raw/_load_bearing/_oe_production_turn/a3/$MODE
for s in 42 43 44 100 101 102; do
  for j in 0 1 2 3; do
    q "$s" intact "$j" "$OUT"
    q "$s" lesion "$j" "$OUT"
  done
  q "$s" intact_rebuild 0 "$OUT"
done
bash tools/pool_queue.sh list | head -3
