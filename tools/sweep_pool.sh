#!/bin/bash
# sweep_pool.sh — runner-AGNOSTIC parameter sweep on the mini-PC pool, HEADLESS, ZERO Claude/agent tokens.
# The usage-saving path for parameter TUNING (grid search): the standing rule is "sweeps run on non-Claude
# machinery, not subagents". Multi-SEED validation the controller already fans out directly; this is for
# GRIDS over parameter VALUES — it expands the grid, dispatches round-robin across pool40/41/42 detached,
# and writes a manifest. Retrieve results later with the printed rsync lines. (CPU/numpy runners only — the
# pool is CPU; GPU sweeps stay local.)
#
# Usage:
#   tools/sweep_pool.sh <out_dir_rel> '<runner module + FIXED args, with {V} where the swept value goes>' \
#       'AXIS_NAME=v1,v2,v3' ['AXIS2=a,b' ...]
#
# Example (sweep the mouth read window over 3 values, 6-seed each, one job per value):
#   tools/sweep_pool.sh raw/mouth_win \
#     'research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk --seeds 42,43,44,100,101,102 --sub-read-window {V}' \
#     'win=120,240,360'
# -> dispatches 3 jobs (win=120/240/360), each writing <out_dir_rel>/<tag>.json on its node.
#
# NOTE: a single {V} axis is substituted; extra axes are cartesian-producted and appended as --AXIS VALUE.
set -e
OUT_REL="$1"; TEMPLATE="$2"; shift 2
[ -z "$OUT_REL" ] || [ -z "$TEMPLATE" ] && { echo "see header for usage" >&2; exit 1; }
NODES=(${POOL_NODES:-pool40 pool41 pool42})
# SINGLE-NODE RESILIENCE (2026-09-03, matches tools/pool_queue.sh): drop unreachable nodes so one
# down node (e.g. pool40 offline) does NOT round-robin ~1/3 of the sweep onto a dead host. Respects a
# POOL_NODES override too. Refuses only if NO node is reachable.
_live=(); for _n in "${NODES[@]}"; do
  timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=6 "$_n" true >/dev/null 2>&1 && _live+=("$_n") \
    || echo "  (sweep_pool: skipping unreachable node $_n)" >&2
done
[ ${#_live[@]} -gt 0 ] && NODES=("${_live[@]}") || { echo "sweep_pool: no reachable pool node" >&2; exit 1; }
REMOTE="derisk-pool/sim"

# Build the grid: first axis fills {V} (the template's inline slot); further axes append as --name value.
AXES=("$@")
[ ${#AXES[@]} -eq 0 ] && { echo "need at least one AXIS=v1,v2 sweep axis" >&2; exit 1; }

# expand cartesian product of all axes -> lines of "tagpart1_tagpart2  arg-suffix  vfill"
python3 - "$TEMPLATE" "$OUT_REL" "${AXES[@]}" <<'PY' > /tmp/_sweep_jobs.txt
import sys, itertools
template, out_rel = sys.argv[1], sys.argv[2]
axes = [a.split("=",1) for a in sys.argv[3:]]
names = [n for n,_ in axes]
valsets = [v.split(",") for _,v in axes]
for combo in itertools.product(*valsets):
    tag = "_".join("%s%s" % (n, v) for n,v in zip(names, combo))
    vfill = combo[0]                      # first axis fills {V}
    suffix = " ".join("--%s %s" % (n, v) for n,v in zip(names[1:], combo[1:]))
    outp = "%s/%s.json" % (out_rel.rstrip("/"), tag)
    cmd = template.replace("{V}", vfill)
    print("\t".join([tag, cmd, suffix, outp]))
PY

N=$(wc -l < /tmp/_sweep_jobs.txt)
echo "== sweep: $N jobs across ${#NODES[@]} nodes (headless, no agent tokens) =="
i=0
while IFS=$'\t' read -r tag cmd suffix outp; do
  node="${NODES[$((i % ${#NODES[@]}))]}"; i=$((i+1))
  full="mkdir -p \$(dirname $outp); SIM_BACKEND=numpy OMP_NUM_THREADS=4 .venv/bin/python -u -m $cmd $suffix --json $outp"
  ssh -f -n -o BatchMode=yes "$node" "cd ~/$REMOTE && setsid bash -c '{ $full; } > sweep_${tag}.out 2>&1; printf \"%s\t%s\t%s\n\" \"\$?\" \"\$(date +%H:%M:%S)\" \"$tag\" >> job_status.log' </dev/null >/dev/null 2>&1 & exit 0" 2>/dev/null \
    && echo "  $node <- $tag" || echo "  $node <- $tag  (SSH FAIL)"
  echo "rsync -e ssh $node:$REMOTE/$outp research/findings/raw/" >> /tmp/_sweep_retrieve.txt
done < /tmp/_sweep_jobs.txt
echo "== retrieve later: =="; cat /tmp/_sweep_retrieve.txt 2>/dev/null; rm -f /tmp/_sweep_retrieve.txt
echo "== progress: ssh pool4X 'tail ~/$REMOTE/job_status.log' =="
