#!/usr/bin/env bash
# E1 spiking twin on the workstation RTX 3090 (prereg: brain/research/2026-11-27-spiking-twin-e1-prereg.md).
# Every GPU job goes through tools/gpu_queue.sh (one job at a time, yields VRAM) under tools/memcap.sh.
# Stages, run one at a time and in order; each is idempotent (units resume / skip):
#   bash launch_3090.sh kernel   # writes OUT/KERNEL = the owner-approved training kernel (prereg section 7 item 1:
#                                #   fla-core 0.6.0 chunk_rwkv7, off the shelf, from FLA_LIB) and queues its GPU parity
#                                #   test (brain/tests/test_spiking_twin_fla.py, BRAIN_REQUIRE_GPU=1 so a skip FAILS).
#                                #   Only a passing run writes OUT/PARITY_OK, keyed to the sha256 of the twin code and
#                                #   the test. prep and smoke refuse without KERNEL, and for kernel=fla also without a
#                                #   PARITY_OK whose hashes match the current files (a code change re-requires parity).
#                                #   KERNEL_CHOICE=loop writes the in-repo loop decision instead (no parity; the gate is
#                                #   then expected to stop). WKV_DTYPE=bf16 hands bf16 tensors to the kernel (default
#                                #   fp32); the parity test covers both. S1_REPEAT=1 (prereg 7.1 A2, once only): after
#                                #   a rerun whose only failure was the S1 mean rule, keeps that run in parity/rerun_draw1/
#                                #   and reruns with BRAIN_S1_REPEAT=1 (S1-mean then decided by draw 2 + pooled 128).
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
FLA_LIB="${FLA_LIB:-/home/dant123/Projects/sim/_scratch/pylib_fla}"   # fla-core 0.6.0 (--no-deps) + einops; twin only
MEM_GB="${MEM_GB:-16}"
SEEDS_SIX="42 43 44 100 101 102"
mkdir -p "$OUT"
[ -f "$HERE/.gitignore" ] || printf 'run/\n' > "$HERE/.gitignore"
mod="-m brain.experiments.spiking_twin_e1"
job() {   # job '<args to the module>' : one queued GPU job, in this checkout, capped
  local cmd="cd $REPO && BRAIN_ALLOW_W3090=1 OMP_NUM_THREADS=4 PYTHONPATH=.${2:-} bash tools/memcap.sh $MEM_GB -- $PY -u $mod $1 >> $OUT/log.txt 2>&1"
  bash "$REPO/tools/gpu_queue.sh" add "$cmd"
}
kv() { sed -n "s/^$1=//p" "$OUT/KERNEL" | head -1; }
code_hash() {   # everything the parity verdict depends on: twin, cortex math, E1 harness, the test, the fla install
  local rec="$FLA_LIB/fla_core-0.6.0.dist-info/RECORD"
  [ -s "$rec" ] || { echo "missing-$rec"; return; }
  (cd "$REPO" && sha256sum brain/spiking_twin.py brain/plastic_cortex.py brain/experiments/spiking_twin_e1.py \
     brain/tests/test_spiking_twin_fla.py brain/experiments/spiking_twin_fla_bias_diag.py "$rec" | sha256sum | cut -c1-64); }
need_kernel() {
  [ -s "$OUT/KERNEL" ] || { echo "refuse: run 'bash launch_3090.sh kernel' first (writes $OUT/KERNEL)"; exit 7; }
  if [ "$(kv kernel)" = "fla" ]; then
    [ -s "$OUT/PARITY_OK" ] || { echo "refuse: fla parity has not passed (no $OUT/PARITY_OK; see $OUT/parity/log.txt)"; exit 8; }
    [ "$(sed -n 's/^code=//p' "$OUT/PARITY_OK")" = "$(code_hash)" ] || {
      echo "refuse: PARITY_OK is for other twin/test code; re-run 'bash launch_3090.sh kernel'"; exit 8; }
  fi
}
train_args() {   # extra train flags + PYTHONPATH suffix from OUT/KERNEL
  if [ "$(kv kernel)" = "fla" ]; then WKV_FLAGS="--wkv fla --wkv-dtype $(kv wkv_dtype)"; WKV_PATH=":$FLA_LIB"
  else WKV_FLAGS=""; WKV_PATH=""; fi
}
case "${1:-}" in
  kernel)
    choice="${KERNEL_CHOICE:-fla}"; dt="${WKV_DTYPE:-fp32}"
    rm -f "$OUT/PARITY_OK"
    if [ "$choice" = "loop" ]; then
      printf 'kernel=loop\nwkv_dtype=fp32\ndecision=in-repo per-token loop (measure only; the gate is expected to stop)\n' > "$OUT/KERNEL"
      echo "KERNEL = loop"; exit 0
    fi
    [ "$choice" = "fla" ] || { echo "KERNEL_CHOICE must be fla or loop"; exit 1; }
    case "$dt" in fp32|bf16) ;; *) echo "WKV_DTYPE must be fp32 or bf16"; exit 1 ;; esac
    ver=$(PYTHONPATH="$FLA_LIB" $PY -c "import importlib.metadata as m;print(m.version('fla-core'))" 2>/dev/null || true)
    [ "$ver" = "0.6.0" ] || { echo "refuse: fla-core 0.6.0 not found in $FLA_LIB (got '$ver')"; exit 9; }
    { echo "kernel=fla"; echo "wkv_dtype=$dt"; echo "fla_core=$ver"; echo "fla_lib=$FLA_LIB"
      echo "op=fla.ops.rwkv7.chunk_rwkv7 (training only: student + teacher targets inside train; eval/prep/T=1 = loop)"
      echo "f32_dot=ieee (TRITON_F32_DEFAULT, set by brain.spiking_twin.fla_import; prereg 7.1 amendment A2)"
      echo "decision=owner-approved off-the-shelf training kernel, 2026-11-27 (prereg section 7 item 1)"
      echo "parity=brain/tests/test_spiking_twin_fla.py (bars in its docstring and the prereg); throughput gate unchanged"
      echo "written=$(date -Iseconds)"; } > "$OUT/KERNEL"
    mkdir -p "$OUT/parity"
    rep_env=""
    if [ "${S1_REPEAT:-0}" = "1" ]; then   # A2: one repeat, only after a rerun that failed the S1 mean rule alone
      P="$OUT/parity"
      [ ! -e "$P/rerun_draw1" ] || { echo "refuse: the A2 repeat has already run ($P/rerun_draw1 exists)"; exit 10; }
      $PY -c "import json,sys;d=json.load(open('$P/teacher_on_stat.json'));f=d['fails'];sys.exit(0 if d.get('amendment')=='A2' and 'repeat' not in d and f and all('fails the A2 rule' in x for x in f) else 1)" \
        && grep -Eq '(^|[^0-9])1 failed' "$P/log.txt" \
        || { echo "refuse: the A2 repeat needs an A2 rerun whose only failure is the S1 mean rule"; exit 10; }
      mkdir "$P/rerun_draw1" && cp -p "$P"/*.json "$P/log.txt" "$P/rerun_draw1/"
      rep_env="BRAIN_S1_REPEAT=1 "
    fi
    ch=$(code_hash)
    cmd="cd $REPO && ${rep_env}BRAIN_ALLOW_W3090=1 BRAIN_REQUIRE_GPU=1 BRAIN_PARITY_OUT=$OUT/parity OMP_NUM_THREADS=4 PYTHONPATH=.:$FLA_LIB bash tools/memcap.sh $MEM_GB -- $PY -m pytest -q -p no:cacheprovider -rA brain/tests/test_spiking_twin_fla.py > $OUT/parity/log.txt 2>&1 && ! grep -qi skipped $OUT/parity/log.txt && printf 'code=$ch\\npassed=%s\\n' \"\$(date -Iseconds)\" > $OUT/PARITY_OK"
    bash "$REPO/tools/gpu_queue.sh" add "$cmd"
    echo "KERNEL = fla ($ver, wkv_dtype $dt); parity test queued -> $OUT/parity/log.txt; PARITY_OK only on a pass" ;;
  prep)
    need_kernel
    bash "$REPO/tools/mem_ok.sh" "$MEM_GB" || { echo "mem_ok refuses: not enough free RAM"; exit 3; }
    job "prep --out-dir $OUT --device cuda" ;;
  smoke)
    need_kernel
    for f in fineweb.npy chat_train.npy; do [ -f "$OUT/$f" ] || { echo "run prep first ($f missing)"; exit 2; }; done
    mkdir -p "$OUT/smoke"
    for f in fineweb.npy chat_train.npy; do ln -sf "$OUT/$f" "$OUT/smoke/$f"; done
    train_args
    job "train --arm a --seed 7 --out-dir $OUT/smoke --device cuda --precision bf16 --max-steps 200 $WKV_FLAGS" "$WKV_PATH" ;;
  gate)
    need_kernel
    # the smoke must have been measured under the kernel in OUT/KERNEL (records carry 'wkv' only for a non-loop kernel)
    sk=$($PY -c "import json;w=json.load(open('$OUT/smoke/train_a_s7.partial.json')).get('wkv');print('loop' if w is None else w['kernel']+':'+w['dtype'])")
    if [ "$(kv kernel)" = "fla" ]; then want="fla:$(kv wkv_dtype)"; else want="loop"; fi
    [ "$sk" = "$want" ] || { echo "refuse: smoke was measured with kernel '$sk' but OUT/KERNEL says '$want'; re-run smoke"; exit 8; }
    tps=$($PY -c "import json;print(int(json.load(open('$OUT/smoke/train_a_s7.partial.json'))['tok_per_s']))")
    rs=$($PY -c "import json;print(json.load(open('$OUT/smoke/train_a_s7.partial.json'))['refill_s'])")
    $PY $mod estimate --tok-per-s "$tps" --refill-s "$rs" | tee "$OUT/estimate_measured.json"
    if [ "$tps" -lt "$MIN_TOK_S" ]; then
      echo "GATE STOP: measured $tps tok/s < $MIN_TOK_S. Do not launch; get an off-the-shelf WKV7 training kernel"
      echo "(prereg section 7, pre-data review item 1) or re-plan the token budget in a NEW prereg."
      exit 4
    fi
    train_args
    for arm in a a2 b; do job "train --arm $arm --seed 7 --out-dir $OUT --device cuda --precision bf16 $WKV_FLAGS" "$WKV_PATH"; done
    for arm in a a2 b c; do job "eval --arm $arm --seed 7 --out-dir $OUT --device cuda"; done ;;
  score7)
    $PY $mod score --out-dir "$OUT" --seeds 7 --out "$OUT/verdict_dev7.json" ;;
  six)
    [ -f "$OUT/REGISTERED" ] || { echo "refuse: prereg not registered (touch $OUT/REGISTERED after the pre-data review)"; exit 5; }
    v=$($PY -c "import json;print(json.load(open('$OUT/verdict_dev7.json'))['verdict'])")
    [ "$v" != "VOID" ] || { echo "refuse: seed-7 dev verdict is VOID; fix the instrument first"; exit 6; }
    need_kernel; train_args
    for s in $SEEDS_SIX; do
      for arm in a a2 b; do job "train --arm $arm --seed $s --out-dir $OUT --device cuda --precision bf16 $WKV_FLAGS" "$WKV_PATH"; done
      for arm in a a2 b c; do job "eval --arm $arm --seed $s --out-dir $OUT --device cuda"; done
    done ;;
  score)
    $PY $mod score --out-dir "$OUT" --out "$OUT/verdict.json" ;;
  *) sed -n 2,26p "$0"; exit 1 ;;
esac
