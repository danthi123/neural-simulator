#!/usr/bin/env bash
# Dispatch the consolidation dendritic operating-point sweep across the mini-PC pool, DETACHED (survives this session).
# 240 configs x 6 seeds = 1440 cells, round-robin across pool40/41/42, each node runs 12 concurrent timeout-guarded
# subprocesses. RESUME-SAFE: re-running skips cells whose JSON already exists (a node reboot resumes cleanly).
# Prereq: tools/pool_provision.sh has run (venv + numpy/scipy + code present at ~/derisk-pool/sim).
# Usage:  bash tools/pool_opsweep_dispatch.sh            # launch on all 3 nodes, detached
#         bash tools/pool_opsweep_dispatch.sh --status   # collect progress
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=derisk-pool/sim
OUT=research/findings/raw/consol_opsweep
SEEDS="42 43 44 100 101 102"
NODES=(pool40 pool41 pool42)
NCFG=$(.venv/bin/python -m research.runners._consol_dendritic_opsweep --list-configs 2>/dev/null | tail -1)

if [ "${1:-}" = "--status" ]; then
  tot=0
  for h in "${NODES[@]}"; do
    n=$(ssh "$h" "ls $REMOTE/$OUT/op*_seed*.json 2>/dev/null | wc -l" 2>/dev/null)
    done=$(ssh "$h" "ls $REMOTE/$OUT/QUEUE_DONE_* 2>/dev/null | wc -l" 2>/dev/null)
    cand=$(ssh "$h" "grep -l '\"candidate\": true' $REMOTE/$OUT/op*.json 2>/dev/null | wc -l" 2>/dev/null)
    echo "$h: $n json, candidates=$cand, sentinel=$done"
    tot=$((tot + ${n:-0}))
  done
  echo "TOTAL: $tot / $((NCFG * 6)) cells"
  exit 0
fi

# ---- build the per-node run-cell script (heredoc) --------------------------------------------------------------------
RUNCELL=$(cat <<'EOS'
#!/usr/bin/env bash
cd ~/derisk-pool/sim
export SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=research/findings/raw/consol_opsweep
mkdir -p "$OUT" logs
run_one(){
  local ci="$1" seed="$2"
  local f; f="$OUT/op$(printf %03d "$ci")_seed${seed}.json"
  [ -s "$f" ] && return 0                       # RESUME: skip completed cells (incl. dendritic-done/linear-timed-out)
  timeout 2700 ~/simvenv/bin/python -m research.runners._consol_dendritic_opsweep \
    --config-index "$ci" --seed "$seed" --out "$OUT" >> logs/opsweep.log 2>&1
}
export -f run_one; export OUT
xargs -P 12 -n 2 bash -c 'run_one "$0" "$1"' < cells.txt
touch "$OUT/QUEUE_DONE_$(hostname).txt"
EOS
)

# ---- shard cells round-robin + launch each node detached -------------------------------------------------------------
tmp=$(mktemp -d)
i=0
for s in $SEEDS; do
  for ci in $(seq 0 $((NCFG - 1))); do
    node=$((i % 3)); echo "$ci $s" >> "$tmp/cells_$node.txt"; i=$((i + 1))
  done
done
echo "sharded $i cells across ${#NODES[@]} nodes ($(wc -l < "$tmp/cells_0.txt") each)"
for k in 0 1 2; do
  h="${NODES[$k]}"
  echo "=== launching $h ==="
  ssh "$h" "mkdir -p ~/$REMOTE" || { echo "  SSH FAIL"; continue; }
  scp -q "$tmp/cells_$k.txt" "$h:~/$REMOTE/cells.txt"
  printf '%s\n' "$RUNCELL" | ssh "$h" "cat > ~/$REMOTE/run_cell.sh && chmod +x ~/$REMOTE/run_cell.sh"
  ssh "$h" "cd ~/$REMOTE && setsid nohup bash run_cell.sh >/dev/null 2>&1 & echo '  launched detached pid '\$!"
done
rm -rf "$tmp"
echo "ALL NODES LAUNCHED. Collect with:  bash tools/pool_opsweep_dispatch.sh --status"
