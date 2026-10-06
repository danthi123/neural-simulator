#!/usr/bin/env bash
# make_jobs.sh -- print the GPU job lines of the Stage-1 C3 `lived` cupy replication (HANDOFF.md, this directory).
#   bash docs/drafts/gpu-s1-lived-cupy/make_jobs.sh dev    # 2 lines: dev seed 7, intact_a + intact_b (smoke + null)
#   bash docs/drafts/gpu-s1-lived-cupy/make_jobs.sh gate   # 42 lines: 6 gate seeds x the 7 `lived` arms
# Each line is handed to `bash tools/gpu_remote.sh add --rev $GATE '<line>'`, which prefixes `cd <per-job worktree at
# GATE> && `: the job runs the Stage-1 gate's own runner at the gate's own pin, on cupy, one brain process at a time.
# Arms = the runner's arms_for("lived") at GATE (no informational arms exist for lived). The out dir is NEVER the gate's
# (research/findings/raw/_stage1_exit/gate): this is a different network at the same seed (docs/ENGINE_REFERENCE.md,
# the cross-backend seed trap), reported beside the gate, never mixed into it.
set -euo pipefail
GATE=c238ff325a38e40a68b1f9751822630ff729377b
PREREG=research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md
OUT=research/findings/raw/_stage1_exit/gpu_lived_cupy
ARMS="intact_a intact_b lesion_recall lesion_recall.rep0 lesion_episodic lesion_episodic.rep0 permuted"
line() {   # line <seed> <arm> <out dir>
  local d="$3/s$1"
  printf '%s\n' "mkdir -p $d && for _w in \$(seq 90); do bash tools/mem_ok.sh 8 4 >/dev/null 2>&1 && break; sleep 60; done && bash tools/mem_ok.sh 8 4 && bash tools/memcap.sh 12 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u -m research.runners.stage1_exit_gate --shard --backend cupy --seed $1 --group lived --arm $2 --allow-gate-seed --prereg $PREREG --out $d/lived__$2.json"
}
case "${1:-}" in
  dev)  line 7 intact_a "$OUT/dev"; line 7 intact_b "$OUT/dev" ;;
  gate) for s in 42 43 44 100 101 102; do for a in $ARMS; do line "$s" "$a" "$OUT"; done; done ;;
  *) echo "usage: make_jobs.sh dev|gate" >&2; exit 2 ;;
esac
