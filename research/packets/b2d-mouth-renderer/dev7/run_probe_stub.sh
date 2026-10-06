#!/usr/bin/env bash
# Dev seed 7, CPU: the mood-render probe with the stub renderer (LB_ARM_RENDERER unset), three arms in turn.
set -uo pipefail
Q=${1:?revision}
cd "$HOME/derisk-pool/revisions/$Q" || exit 2
mkdir -p research/findings/raw/_b2d_mood_render_probe/dev7_stub
for arm in intact_a intact_b lesion; do
  until bash tools/mem_ok.sh 8 4 >/dev/null 2>&1; do sleep 30; done
  bash tools/memcap.sh 12 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BRAIN_CHAT_SEED=7 \
    .venv/bin/python -u -m research.runners._b2d_mood_render_probe --arm $arm \
    --out research/findings/raw/_b2d_mood_render_probe/dev7_stub/$arm.json \
    > research/findings/raw/_b2d_mood_render_probe/dev7_stub/$arm.log 2>&1
  echo "$(date +%T) stub probe arm=$arm rc=$?"
done
echo ALL-DONE
