#!/usr/bin/env bash
# b2c_smoke.sh -- Battery B2c's integrity smoke (declared, no weight): two FLIPCAND lines at seed 7 (not a battery
# seed), run by direct ssh on one F2-provisioned node under a HARD 8 GB cap (the lines' declared mem_gb).
#
# Prereg: research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md ("Integrity
# smoke"). It checks only that the arms build with the pair ON, that `da_tag_capture` appears without an error,
# n_managed_blocks, `sleep_replay_capture` at slp_recall, and that each line completes under memcap 8. No criterion
# reads it. Output goes to research/findings/raw/_load_bearing/_b2c0925_smoke/, which no aggregate reads (it is not
# under _shards/). The job text is the flipcand line of b2c0925_jobs.txt with the seed and the output dir swapped, so
# the smoke runs exactly what the battery will run.
#
# Usage:  bash research/coordination/b2c_smoke.sh [--print] [node]        (default node: pool1)
#         --print shows the two remote commands and runs nothing.
# Exit:   0 = both lines completed; otherwise the first failing line's exit status.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
F2=fd29040db19987819461693aaf385977e45840ef
PRINT=0; [ "${1:-}" = "--print" ] && { PRINT=1; shift; }
NODE=${1:-pool1}
JOBS=research/coordination/b2c0925_jobs.txt
SSHCFG=research/queue/.pool_ssh_config
SSH=(ssh -n); [ -f "$SSHCFG" ] && SSH=(ssh -n -F "$SSHCFG")
rc=0
for ROW in sleep-replay causal-whatif; do
  line=$(grep -F "/_shards/b2c0925-flipcand/s42/$ROW/" "$JOBS" | head -1)
  [ -n "$line" ] || { echo "⛔ no flipcand s42 line for $ROW in $JOBS" >&2; exit 2; }
  envpart=${line#*" && env "}; envpart=${envpart%%" .venv/bin/python -u -m "*}   # the line's own env prefix
  out_dir=research/findings/raw/_load_bearing/_b2c0925_smoke/s7/$ROW
  cmd="cd ~/derisk-pool/revisions/$F2 && mkdir -p $out_dir && bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 8 --"
  cmd+=" env $envpart .venv/bin/python -u -m research.runners.load_bearing_fraction --only $ROW --seed 7"
  cmd+=" --repeats 2 --out $out_dir/lb.json"
  echo "[b2c-smoke] $NODE: $cmd"
  [ "$PRINT" -eq 1 ] && continue
  "${SSH[@]}" "$NODE" "$cmd"; r=$?
  echo "[b2c-smoke] $ROW exit $r"
  [ "$r" -ne 0 ] && [ "$rc" -eq 0 ] && rc=$r
done
exit "$rc"
