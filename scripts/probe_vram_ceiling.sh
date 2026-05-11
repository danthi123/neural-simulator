#!/usr/bin/env bash
# VRAM ceiling probe — find hardware limit on RTX 3090 24 GB.
# Per user (2026-05-10): we know encoding scaling helps but haven't
# measured the actual VRAM ceiling. This sweep uses perf_benchmark
# (each probe ~3-5 min) to map the boundary.
#
# Phase A: encoding scaling alone (vocab=8, n_motor=2000)
# Phase B: motor scaling alone (vocab=8, n_lang=4096)
# Phase C: combined push
#
# Captures wall clock + VRAM peak per config.
# OOM caught via try-or-exit; logs go to research/findings/raw/perf/.

set +e  # don't bail on individual OOM — that's the data we want
mkdir -p research/findings/raw/perf/vram_ceiling

echo "=== VRAM ceiling probe started $(date +%Y-%m-%dT%H:%M:%S) ==="

# Wait for GPU to free up
while true; do
  RUNNING=$(curl -s "http://localhost:8765/api/runs/launch" 2>/dev/null \
    | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(sum(1 for r in d.get('runs',[]) if r.get('running')))" 2>/dev/null)
  if [ "$RUNNING" = "0" ] || [ -z "$RUNNING" ]; then
    # Also check nvidia-smi
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$GPU_USED" ] && [ "$GPU_USED" -lt 5000 ]; then
      break
    fi
  fi
  echo "[wait] $(date +%H:%M:%S)  GPU busy (used=${GPU_USED:-?} MB, running=${RUNNING:-?}); sleep 60s..."
  sleep 60
done
echo "[OK] GPU free, starting probes."

probe() {
  local name="$1"
  local n_lang="$2"
  local n_motor="$3"
  local vocab="${4:-8}"
  local steps="${5:-500}"
  local out="research/findings/raw/perf/vram_ceiling/${name}.json"

  echo "--- probe: ${name} (n_lang=${n_lang}, n_motor=${n_motor}, vocab=${vocab}) ---"

  python -m research.runners.perf_benchmark \
    --steps "${steps}" \
    --vocab-size "${vocab}" \
    --n-lang-input "${n_lang}" \
    --n-motor-per-action "${n_motor}" \
    --n-motor-fs-per-action "$((n_motor / 10))" \
    --out "${out}" 2>&1 | tail -20

  if [ -f "${out}" ]; then
    echo "[OK] ${name} succeeded"
  else
    echo "[OOM or crash] ${name} FAILED — VRAM ceiling found"
  fi
}

# ─── Phase A: encoding scaling alone ───────────────────────────────────
echo ""
echo "=== Phase A: encoding scaling ==="
probe "A1_lang16k_motor2k" 16384 2000 8 500
probe "A2_lang32k_motor2k" 32768 2000 8 500
probe "A3_lang65k_motor2k" 65536 2000 8 500
probe "A4_lang131k_motor2k" 131072 2000 8 500

# ─── Phase B: motor scaling alone ──────────────────────────────────────
echo ""
echo "=== Phase B: motor scaling ==="
probe "B1_lang4k_motor4k" 4096 4000 8 500
probe "B2_lang4k_motor8k" 4096 8000 8 500
probe "B3_lang4k_motor16k" 4096 16000 8 500
probe "B4_lang4k_motor32k" 4096 32000 8 500

# ─── Phase C: combined ─────────────────────────────────────────────────
echo ""
echo "=== Phase C: combined ==="
probe "C1_lang16k_motor4k" 16384 4000 8 500
probe "C2_lang32k_motor4k" 32768 4000 8 500
probe "C3_lang16k_motor8k" 16384 8000 8 500
probe "C4_lang32k_motor8k" 32768 8000 8 500

# ─── Phase D: max-vocab at max-encoding ────────────────────────────────
echo ""
echo "=== Phase D: max vocab at max encoding ==="
# These use bigger vocab + max viable arch
probe "D1_lang16k_motor2k_vocab128" 16384 2000 128 500
probe "D2_lang32k_motor2k_vocab256" 32768 2000 256 500
probe "D3_lang32k_motor4k_vocab256" 32768 4000 256 500

echo ""
echo "=== VRAM ceiling probe done $(date +%Y-%m-%dT%H:%M:%S) ==="
echo ""
echo "Summary of successful probes:"
for f in research/findings/raw/perf/vram_ceiling/*.json; do
  if [ -f "$f" ]; then
    name=$(basename "$f" .json)
    python -c "
import json
d = json.load(open('$f'))
print(f'  {d.get(\"n_lang_input\", \"?\"):>6}lang x {d.get(\"n_motor_per_action\", \"?\"):>5}mot x vocab{d.get(\"vocab_size\", \"?\"):>3}  ->  VRAM {d.get(\"vram_peak_mb\", 0):>5.0f} MB, {d.get(\"steps_per_sec\", 0):>6.1f} steps/sec'
" 2>/dev/null
  fi
done
