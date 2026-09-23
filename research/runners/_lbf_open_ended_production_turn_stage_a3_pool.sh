#!/usr/bin/env bash
# AMENDMENT-3 staging for the open-ended production-turn probe, on the MINI-PC POOL (never locally: the local box is
# RAM-bound and each session is one full numpy brain). PREREG amendment 3 = commit eefdd666a.
#   1. bash tools/pool_provision.sh --isolated --revision <SHA> pool41 pool42      (once)
#   2. bash research/runners/_lbf_open_ended_production_turn_stage_a3_pool.sh <FULL_SHA> --smoke
#   3. wait for the smoke job to land ON THE NODE, THEN HARVEST IT TO THIS LOCAL CHECKOUT:
#        bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh <FULL_SHA> smoke
#      (round-6 review, 2026-09-23: the check below reads $SMOKE_FILE from THIS checkout, not from the pool
#      node -- the smoke job's output only reaches here after this harvest step. Skipping it means step 4 below
#      will always REFUSE with "no smoke output", even once the node-side job has long finished.)
#   4. THEN: bash .../stage_a3_pool.sh <FULL_SHA>   (no 2nd arg: queues the 54 governed jobs, but ONLY after
#      checking the harvested smoke output completed cleanly -- round-5 review fix, 2026-09-23: this used to be
#      advisory-only, so the 54 governed jobs (hours of pool time) could start before, or without, anyone
#      checking the smoke. REFUSES (exit 3) if the smoke file is missing or looks incomplete/errored;
#      `--skip-smoke-check` overrides for someone who already verified health another way.)
#   5. bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh <FULL_SHA>   (idempotent; re-run)
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
SMOKE_OUT=research/findings/raw/_load_bearing/_oe_production_turn/a3_smoke/$MODE
SMOKE_FILE="$SMOKE_OUT/${MODE}_s7_intact_n0.json"
if [ "${2:-}" = "--smoke" ]; then
  q 7 intact 0 "$SMOKE_OUT"
  exit 0
fi
# ROUND-5 REVIEW FIX (2026-09-23): "the non-governed smoke (seed 7) was queued ahead of the 54 governed jobs but
# does not gate them ... someone must check the smoke before the bulk burns hours." The smoke was advisory only --
# nothing stopped queuing the 54 governed jobs (many hours of pool time) before it had even returned. This is now
# a real gate: no smoke file (or one that never completed / errored) REFUSES the governed queue outright. This
# checks completion and shape, NOT the scientific verdict (a smoke has no lesion arm to score) -- a slow-but-valid
# smoke session should not be treated as a failure.
if [ "${2:-}" != "--skip-smoke-check" ]; then
  if [ ! -f "$SMOKE_FILE" ]; then
    echo "[stage_a3] REFUSED: no smoke output at $SMOKE_FILE (in THIS LOCAL CHECKOUT)." >&2
    echo "[stage_a3] Run '$0 $SHA --smoke' first, wait for the job to finish ON THE POOL NODE, then pull it here:" >&2
    echo "[stage_a3]   bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh $SHA smoke" >&2
    echo "[stage_a3] (round-6 review, 2026-09-23: this check only ever reads the LOCAL checkout -- a smoke job" >&2
    echo "[stage_a3] that finished on the node but was never harvested here will refuse forever.) Then re-run" >&2
    echo "[stage_a3] this command (no 2nd arg)." >&2
    echo "[stage_a3] Override only if you have already manually verified pipeline health another way:" >&2
    echo "[stage_a3]   $0 $SHA --skip-smoke-check" >&2
    exit 3
  fi
  if ! .venv/bin/python -c "
import json, sys
d = json.load(open('$SMOKE_FILE'))
replies = d.get('replies') or []
ok = bool(replies) and not any('error' in r for r in replies) and d.get('draw_counter', {}).get('n_calls', 0) > 0
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    echo "[stage_a3] REFUSED: smoke output at $SMOKE_FILE exists but did not complete cleanly (no replies, an" >&2
    echo "[stage_a3] errored reply, or the draw was never reached). Inspect it before staging the 54 governed" >&2
    echo "[stage_a3] jobs; do not assume the pipeline is healthy. Override: $0 $SHA --skip-smoke-check" >&2
    exit 3
  fi
  echo "[stage_a3] smoke check OK ($SMOKE_FILE) -- staging the 54 governed jobs."
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
