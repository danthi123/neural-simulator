#!/usr/bin/env bash
# Multi-seed vocab-ceiling sweep driver (overnight Thread A, step 2).
# Runs the full BrainConversationalAgent capability matrix + anti-cheats across
# seeds 42-47 at V=320 (D=128 and D=256) and V=128 (D=128).
# GPU: SIM_BACKEND=cupy. Each cell writes its own raw JSON.
set -u
export SIM_BACKEND=cupy
cd "$(git rev-parse --show-toplevel)"
RAW=research/findings/raw
LOG=$RAW/_vocab_ceiling_multiseed_run.log
: > "$LOG"

run_cell () {
  local V=$1 seed=$2 D=$3 tag=$4
  local out=$RAW/_vocab_ceiling_${tag}.json
  echo "[$(date +%H:%M:%S)] START V=$V seed=$seed D=$D -> $out" | tee -a "$LOG"
  python -m research.runners.vocab_ceiling_probe --V "$V" --seed "$seed" --D "$D" --out "$out" \
      >> "$LOG" 2>&1
  # extract the one-line verdict from the JSON
  python - "$out" <<'PY' 2>>"$LOG" | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
m = d["matrix"]
ab = m["abstention"]; sc = m["shuffled_control"]; cl = m["embedded_clause"]; ta = m["two_attribute"]
print(f"[DONE] V={d['V']} seed={d['seed']} D={d['D']} verdict={d['verdict']} "
      f"abstain={ab['correct']}/{ab['attempted']} clause={cl['correct']}/{cl['attempted']} "
      f"2attr={ta['correct']}/{ta['attempted']} shuffled_false_hits={sc['false_hits']} "
      f"failing={d['failing_caps']}")
PY
}

# ---- V=320, D=128, all 6 seeds (re-confirm 42-44 + new 45-47) ----
for s in 42 43 44 45 46 47; do
  run_cell 320 $s 128 "V320_s${s}_D128"
done

# ---- V=320, D=256 arm, all 6 seeds (clause D-floor map) ----
for s in 42 43 44 45 46 47; do
  run_cell 320 $s 256 "V320_s${s}_D256"
done

# ---- V=128, D=128, all 6 seeds (intermediate rung) ----
for s in 42 43 44 45 46 47; do
  run_cell 128 $s 128 "V128_s${s}_D128"
done

echo "[$(date +%H:%M:%S)] ALL CELLS COMPLETE" | tee -a "$LOG"
