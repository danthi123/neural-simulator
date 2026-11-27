#!/usr/bin/env bash
# E1 spiking twin on the workstation RTX 3090 (prereg: brain/research/2026-11-27-spiking-twin-e1-prereg.md).
# Every GPU job goes through tools/gpu_queue.sh (one job at a time, yields VRAM) under tools/memcap.sh.
# Stages, run one at a time and in order; each is idempotent (units resume / skip):
#   (first) write OUT/KERNEL: the owner's / reviewer's training-kernel decision (prereg section 7 item 1). prep and
#           smoke refuse without it, so no GPU time is spent before the decision. "in-repo" = measure the per-token
#           loop only (the gate is expected to stop it); anything else names an off-the-shelf kernel + its parity test.
#   bash launch_3090.sh prep     # tokenise fineweb (first 0.3B tokens) + WT103 val, teacher-written chat pools (queued)
#   bash launch_3090.sh smoke    # 200 optimizer steps of arm a, seed 7, in OUT/smoke: measures tok/s and the time of
#                                #   one a2 refill (queued)
#   bash launch_3090.sh gate     # reads the smoke tok/s; queues seed-7 dev (a, a2, b train; a, a2, b, c eval) only
#                                #   if tok/s >= MIN_TOK_S (projected seed-7 total <= about 30 GPU-h); else stops
#   bash launch_3090.sh score7   # dev reading (not a claim)
#   bash launch_3090.sh six      # six seeds; refuses unless OUT/REGISTERED exists (pre-data review done) and the
#                                #   seed-7 dev verdict is not VOID
#   bash launch_3090.sh score    # the registered six-seed verdict -> OUT/verdict.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO"; export PYTHONPATH="$REPO"   # every module call below runs THIS checkout's harness, from any cwd
PY=/home/dant123/Projects/sim/.venv/bin/python
OUT="${E1_OUT:-$HERE/run}"   # ~0.6 GB token arrays + 21 fp32 twins (~0.4-0.8 GB each) + one live resume ckpt per
                             # training job (weights + AdamW, ~1.2-2.3 GB; removed when its twin is final):
                             # about 15-20 GB in all [E]; outside git (.gitignore)
MIN_TOK_S="${MIN_TOK_S:-10000}"
MEM_GB="${MEM_GB:-16}"
SEEDS_SIX="42 43 44 100 101 102"
mkdir -p "$OUT"
[ -f "$HERE/.gitignore" ] || printf 'run/\n' > "$HERE/.gitignore"
mod="-m brain.experiments.spiking_twin_e1"
job() {   # job '<args to the module>' : one queued GPU job, in this checkout, capped
  local cmd="cd $REPO && BRAIN_ALLOW_W3090=1 OMP_NUM_THREADS=4 PYTHONPATH=. bash tools/memcap.sh $MEM_GB -- $PY -u $mod $1 >> $OUT/log.txt 2>&1"
  bash "$REPO/tools/gpu_queue.sh" add "$cmd"
}
need_kernel() { [ -s "$OUT/KERNEL" ] || { echo "refuse: write the training-kernel decision to $OUT/KERNEL first"; exit 7; }; }
case "${1:-}" in
  prep)
    need_kernel
    bash "$REPO/tools/mem_ok.sh" "$MEM_GB" || { echo "mem_ok refuses: not enough free RAM"; exit 3; }
    job "prep --out-dir $OUT --device cuda" ;;
  smoke)
    need_kernel
    for f in fineweb.npy chat_train.npy; do [ -f "$OUT/$f" ] || { echo "run prep first ($f missing)"; exit 2; }; done
    mkdir -p "$OUT/smoke"
    for f in fineweb.npy chat_train.npy; do ln -sf "$OUT/$f" "$OUT/smoke/$f"; done
    job "train --arm a --seed 7 --out-dir $OUT/smoke --device cuda --precision bf16 --max-steps 200" ;;
  gate)
    tps=$($PY -c "import json;print(int(json.load(open('$OUT/smoke/train_a_s7.partial.json'))['tok_per_s']))")
    rs=$($PY -c "import json;print(json.load(open('$OUT/smoke/train_a_s7.partial.json'))['refill_s'])")
    $PY $mod estimate --tok-per-s "$tps" --refill-s "$rs" | tee "$OUT/estimate_measured.json"
    if [ "$tps" -lt "$MIN_TOK_S" ]; then
      echo "GATE STOP: measured $tps tok/s < $MIN_TOK_S. Do not launch; get an off-the-shelf WKV7 training kernel"
      echo "(prereg section 7, pre-data review item 1) or re-plan the token budget in a NEW prereg."
      exit 4
    fi
    for arm in a a2 b; do job "train --arm $arm --seed 7 --out-dir $OUT --device cuda --precision bf16"; done
    for arm in a a2 b c; do job "eval --arm $arm --seed 7 --out-dir $OUT --device cuda"; done ;;
  score7)
    $PY $mod score --out-dir "$OUT" --seeds 7 --out "$OUT/verdict_dev7.json" ;;
  six)
    [ -f "$OUT/REGISTERED" ] || { echo "refuse: prereg not registered (touch $OUT/REGISTERED after the pre-data review)"; exit 5; }
    v=$($PY -c "import json;print(json.load(open('$OUT/verdict_dev7.json'))['verdict'])")
    [ "$v" != "VOID" ] || { echo "refuse: seed-7 dev verdict is VOID; fix the instrument first"; exit 6; }
    for s in $SEEDS_SIX; do
      for arm in a a2 b; do job "train --arm $arm --seed $s --out-dir $OUT --device cuda --precision bf16"; done
      for arm in a a2 b c; do job "eval --arm $arm --seed $s --out-dir $OUT --device cuda"; done
    done ;;
  score)
    $PY $mod score --out-dir "$OUT" --out "$OUT/verdict.json" ;;
  *) sed -n 2,18p "$0"; exit 1 ;;
esac
