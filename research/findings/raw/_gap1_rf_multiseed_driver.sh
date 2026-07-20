#!/usr/bin/env bash
# gap#1 RF-phase-encode 6-seed generalization gate. Provenance-recorded config (the seed-42 ckpt lost its training cmd).
# Retrain a matched ssmU6 set (43 already trained), then per seed: M1 reference + RF-phase-encode MAIN.
# Anti-cheat spot-checks (memoryless + scramble) on one blind seed (100). Batched so GPU VRAM never oversubscribes.
set -u
cd /home/dant123/Projects/sim
RAW=research/findings/raw
PY=".venv/bin/python"
CKPT_PREFIX=bridges/wkv_ckpt/wkv_ssmU6_v1000_d128
DEV="42 43 44"; BLIND="100 101 102"; ALL="$DEV $BLIND"
TRAINCFG="--vocab 1000 --d-model 128 --n-sentences 40000 --epochs 12 --recurrence ssm --dual-nonneg --uniform-decay --corpus data/corpus/tinystories_train.txt"
GATECFG="--ssm-state --use-ssm-readout --n-eval 200"

echo "=== [1/3] retrain matched set (seed 43 exists; train 42 44 100 101 102 concurrently) ==="
for s in 42 44 100 101 102; do
  SIM_BACKEND=cupy $PY -u -m research.runners._emerge_wkv_lm_derisk --seeds $s $TRAINCFG \
    --save-ssm $CKPT_PREFIX > $RAW/_gap1_train_ssmU6_seed$s.log 2>&1 &
done
wait; echo "training done: $(ls $CKPT_PREFIX*.npz 2>/dev/null | wc -l) checkpoints"

echo "=== [2/3] M1 reference per seed (batch of 6) ==="
for s in $ALL; do
  SIM_BACKEND=cupy $PY -u -m research.runners._emerge_wkv_onbridge_derisk \
    --ssm ${CKPT_PREFIX}_seed$s.npz $GATECFG --seed $s \
    --json $RAW/_gap1_ms_M1_seed$s.json > $RAW/_gap1_ms_M1_seed$s.log 2>&1 &
done
wait; echo "M1 done"

echo "=== [3/3] RF-phase-encode MAIN per seed (batch of 6) ==="
for s in $ALL; do
  SIM_BACKEND=cupy $PY -u -m research.runners._emerge_wkv_onbridge_derisk \
    --ssm ${CKPT_PREFIX}_seed$s.npz $GATECFG --rf-phase-encode --seed $s \
    --json $RAW/_gap1_ms_RF_seed$s.json > $RAW/_gap1_ms_RF_seed$s.log 2>&1 &
done
wait; echo "RF-MAIN done"

echo "=== anti-cheat spot-checks on blind seed 100 (memoryless + scramble) ==="
SIM_BACKEND=cupy $PY -u -m research.runners._emerge_wkv_onbridge_derisk \
  --ssm ${CKPT_PREFIX}_seed100.npz $GATECFG --rf-phase-encode --ssm-memoryless --seed 100 \
  --json $RAW/_gap1_ms_RFmemless_seed100.json > $RAW/_gap1_ms_RFmemless_seed100.log 2>&1 &
SIM_BACKEND=cupy $PY -u -m research.runners._emerge_wkv_onbridge_derisk \
  --ssm ${CKPT_PREFIX}_seed100.npz $GATECFG --rf-phase-encode --rf-scramble --seed 100 \
  --json $RAW/_gap1_ms_RFscramble_seed100.json > $RAW/_gap1_ms_RFscramble_seed100.log 2>&1 &
wait; echo "ALL MULTISEED DONE"