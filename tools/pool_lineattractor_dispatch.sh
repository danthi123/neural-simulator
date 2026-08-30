#!/usr/bin/env bash
# Dispatch the consolidation CONTINUOUS-ATTRACTOR (line/bump ring/CANN) de-risk across the mini-PC pool, DETACHED.
# Successor to pool_opsweep_dispatch.sh: the point-plateau opsweep returned NO-GO (343 cells, 0 separation), so this
# sweeps the ring-attractor geometry that is its named surpass. 216 configs; a 2-SEED SCREEN first (42,43) to LOCATE
# candidate operating points cheaply -- 6-seed the winners after. RESUME-SAFE (skips cells whose JSON exists).
# Its kill step also stops any running opsweep run_cell/xargs, handing the pool over (the opsweep verdict is banked).
# Usage:  bash tools/pool_lineattractor_dispatch.sh            # launch on all 3 nodes, detached
#         bash tools/pool_lineattractor_dispatch.sh --status   # collect progress
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=derisk-pool/sim
OUT=research/findings/raw/consol_lineattractor
SEEDS="42 43"
NODES=(pool40 pool41 pool42)
NCFG=$(.venv/bin/python -m research.runners._consol_dendritic_lineattractor_derisk --list-configs 2>/dev/null | tail -1)

if [ "${1:-}" = "--status" ]; then
  tot=0; nseeds=$(echo $SEEDS | wc -w)
  for h in "${NODES[@]}"; do
    n=$(ssh "$h" "ls $REMOTE/$OUT/op*_seed*.json 2>/dev/null | wc -l" 2>/dev/null)
    cand=$(ssh "$h" "grep -l '\"candidate\": true' $REMOTE/$OUT/op*.json 2>/dev/null | wc -l" 2>/dev/null)
    echo "$h: ${n:-0} json, candidates=${cand:-0}"
    tot=$((tot + ${n:-0}))
  done
  echo "TOTAL: $tot / $((NCFG * nseeds)) cells (2-seed screen)"
  exit 0
fi

RUNCELL=$(cat <<'EOS'
#!/usr/bin/env bash
cd ~/derisk-pool/sim
export SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=research/findings/raw/consol_lineattractor
mkdir -p "$OUT" logs
run_one(){
  local ci="$1" seed="$2"
  local f; f="$OUT/op$(printf %03d "$ci")_seed${seed}.json"
  [ -s "$f" ] && return 0                       # RESUME: skip completed cells
  timeout 5400 ~/simvenv/bin/python -m research.runners._consol_dendritic_lineattractor_derisk \
    --config-index "$ci" --seed "$seed" --out "$OUT" >> logs/lineattractor.log 2>&1
}
export -f run_one; export OUT
xargs -P 12 -n 2 bash -c 'run_one "$0" "$1"' < cells.txt
touch "$OUT/QUEUE_DONE_$(hostname).txt"
EOS
)

tmp=$(mktemp -d)
i=0
for s in $SEEDS; do
  for ci in $(seq 0 $((NCFG - 1))); do
    node=$((i % 3)); echo "$ci $s" >> "$tmp/cells_$node.txt"; i=$((i + 1))
  done
done
echo "sharded $i cells across ${#NODES[@]} nodes ($(wc -l < "$tmp/cells_0.txt") each), timeout 5400s/cell"
for k in 0 1 2; do
  h="${NODES[$k]}"
  echo "=== launching $h (kills any opsweep/lineattractor run_cell first -> hands pool over) ==="
  ssh "$h" "for pass in 1 2; do for pg in \$(ps -eo pgid,args | grep -E '[r]un_cell.sh|[x]args -P 12|[_]consol_dendritic' | awk '{print \$1}' | sort -u); do kill -9 -\$pg 2>/dev/null; done; sleep 1; done; mkdir -p ~/$REMOTE; exit 0" || { echo "  SSH FAIL (connection)"; continue; }
  scp -q "$tmp/cells_$k.txt" "$h:~/$REMOTE/cells.txt"
  printf '%s\n' "$RUNCELL" | ssh "$h" "cat > ~/$REMOTE/run_cell.sh && chmod +x ~/$REMOTE/run_cell.sh"
  ssh -f -n "$h" "cd ~/$REMOTE && setsid bash run_cell.sh </dev/null >run_cell.out 2>&1 & exit 0"
  echo "  launched $h (shard $(wc -l < "$tmp/cells_$k.txt") cells)"
done
rm -rf "$tmp"
echo "ALL NODES LAUNCHED. Collect with:  bash tools/pool_lineattractor_dispatch.sh --status"
