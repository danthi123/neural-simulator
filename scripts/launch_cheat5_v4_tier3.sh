#!/usr/bin/env bash
# Cheat #5 v4 Tier 3 launcher — full validation, 6 seeds × 30K pretraining + 1800 eval.
#
# Wall-clock estimate: ~14h total at 4-concurrent (single-process baseline 2.27 step/s,
# 4-concurrent ~1.25 step/s/run → 31800 steps / 1.25 = ~7h per seed wall-time).
# We launch 4 first; when first 4 finish, fire the remaining 2 (4+2 = 6).
#
# Run AFTER tier 2 returns mean sum ≤ 4.5 per the v4 decision matrix
# (see scripts/analyze_cheat5_v4.py).
#
# Pre-flight checks (do these before invoking this script):
#   - Tier 1 PASSED (commit 6bcecff)
#   - Tier 2 mean sum ≤ 4.5 with both phases ≤ 4
#   - Webapp running at http://localhost:8765
#   - GPU has the budget (no other heavy runs)

set -euo pipefail

SEEDS=(42 43 44 100 101 102)
WEBAPP="${WEBAPP:-http://localhost:8765}"

launch_one() {
    local seed=$1
    local response
    response=$(curl -s -X POST -H "Content-Type: application/json" \
        -d "{\"preset\":\"flagship\",\"seed\":${seed},\"extra_args\":[
            \"--bg-lateral-inhibition\",\"--bg-cross-projections\",\"--cross-projection-weight\",\"0.0\",
            \"--developmental-pretraining\",\"--pretraining-n-goals\",\"10\",\"--pretraining-steps-per-goal\",\"3000\"
        ]}" \
        "${WEBAPP}/api/runs/launch")
    echo "$response" | python -c "import json,sys; d=json.load(sys.stdin); print(f'  tier3 seed=${seed}: run_id={d.get(\"run_id\")} pid={d.get(\"pid\")}')"
}

count_running() {
    curl -s "${WEBAPP}/api/runs/launch" | python -c "
import json, sys
d = json.load(sys.stdin)
running = [r for r in d['runs'] if r['running']]
print(len(running))
"
}

echo "=== Cheat #5 v4 Tier 3 — full validation (6 seeds × 30K pretraining) ==="
echo "Launching first batch of 4 seeds (within concurrency knee)..."
for seed in "${SEEDS[@]:0:4}"; do
    launch_one "$seed"
done

echo
echo "Waiting for first batch to drain to ≤ 2 running before launching the remaining 2..."
until [ "$(count_running)" -le "2" ]; do
    sleep 300  # 5 min poll, this is overnight scale
done

echo "Capacity available; launching remaining 2 seeds..."
for seed in "${SEEDS[@]:4:2}"; do
    launch_one "$seed"
done

echo
echo "=== All 6 tier-3 seeds launched. Run 'python scripts/analyze_cheat5_v4.py --tier 3' once they all finish. ==="
