#!/usr/bin/env bash
# CUPY 6-seed GATE verify for the confidence-forthcomingness 100k scale-invariance recalibration
# (board #94 / #108 R3, 2026-09-02). Runs BOTH:
#   (1) the 100k #94 re-verify  -> must now vary_lesion_all_GO=True (the fix)
#   (2) the 15k no-regression re-verify -> must STILL vary_lesion_all_GO=True (the shipped operating point)
# on the ONE brain-loading GPU at a time (this is queued via tools/gpu_queue.sh; never run concurrently with
# another heavy job). This script lives ONLY on branch research/confidence-forthcoming-100k-recalibration, so a
# controller running it from main is proof the branch was merged (the fix in metacog_production_organ.py /
# rf_phasor_composer.py is then present too) -- the queued command self-guards on this file's existence.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 3

OUT=research/findings/raw/_confidence_100k_recalib
V100K=research/findings/raw/_flip108_r3_100k_honesty_reverify/verify_margin_norm_recalibration_100k.py
V15K=research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration.py
SEEDS="42 43 44 100 101 102"

rc=0
echo "== (1) 100k #94 re-verify (6-seed cupy) -- the FIX (expect vary_lesion_all_GO=True) =="
SIM_BACKEND=cupy .venv/bin/python "$V100K" --seeds $SEEDS \
    --out "$OUT/verify_100k_6seed_cupy.json" || rc=1

echo "== (2) 15k no-regression re-verify (6-seed cupy) -- expect vary_lesion_all_GO=True (unchanged) =="
SIM_BACKEND=cupy .venv/bin/python "$V15K" --seeds $SEEDS \
    --out "$OUT/verify_15k_6seed_noregress_cupy.json" || rc=1

echo "== DONE (rc=$rc) -- read $OUT/verify_100k_6seed_cupy.json + verify_15k_6seed_noregress_cupy.json =="
exit $rc
