#!/usr/bin/env bash
# b2c_smoke.sh -- Battery B2c's integrity smoke (declared, no weight): two FLIPCAND lines at seed 7 (not a battery
# seed), run by direct ssh on one F2-provisioned node under a HARD 8 GB cap (the lines' declared mem_gb).
#
# Prereg: research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md ("Integrity
# smoke"). No criterion reads it (this is declared, not gated), but a failure here is worth stopping for before
# wave 1. FIX ROUND (review of this branch, "the assert_flipped_defaults guard is dropped" / "the checks... are
# not automated" / "the outputs stay on pool1 because nothing pulls them back"):
#   - the pinned `.venv/bin/python tools/assert_flipped_defaults.py &&` guard now runs right after `cd`, exactly as
#     every battery job line does (b2c_make_jobs.sh);
#   - stdout+stderr of each remote run is captured and grepped for "DA tag-and-capture tick failed"
#     (continuous_engine.py's idle-tick handler only LOGS this, never raises, so no JSON field can see it -- this
#     is the only way the smoke or the eventual tools/b2c_score.py can catch it);
#   - the remote out_dir is rsync'd back to this checkout (research/findings/raw/_load_bearing/_b2c0925_smoke/,
#     which stays outside _shards/ so no aggregate reads it), THEN research/coordination/b2c_smoke_check.py scores
#     the pulled lb.json + intact arm file(s) against the declared checks: `da_tag_capture` present with no `error`
#     key at ANY depth (top-level or nested under `observe`), `n_managed_blocks`, and for the sleep-replay row
#     `sleep_replay_capture.n_epochs` >= 1 at `slp_recall` once a block is managed. The LTM tier the arm actually
#     built against is read from the pulled sidecar and reported (not gated).
#
# Usage:  bash research/coordination/b2c_smoke.sh [--print] [node]        (default node: pool1)
#         --print shows the two remote commands and runs nothing (and pulls/checks nothing).
# Exit:   0 = both lines completed AND every automated check passed; otherwise the first failing line's status
#         (an ssh/exit failure, a logged tick-failure, a failed pull, or a b2c_smoke_check.py failure all count).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
F2=fd29040db19987819461693aaf385977e45840ef
PRINT=0; [ "${1:-}" = "--print" ] && { PRINT=1; shift; }
NODE=${1:-pool1}
JOBS=research/coordination/b2c0925_jobs.txt
SSHCFG=research/queue/.pool_ssh_config
SSH_ARGS=(); [ -f "$SSHCFG" ] && SSH_ARGS=(-F "$SSHCFG")
SSH=(ssh -n "${SSH_ARGS[@]}")                       # -n: these commands read no stdin
RSYNC_E="ssh -o BatchMode=yes -o ConnectTimeout=6 ${SSH_ARGS[*]}"   # rsync manages its own stdin: no -n here
LOCAL_BASE=research/findings/raw/_load_bearing/_b2c0925_smoke
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
CHECKER=research/coordination/b2c_smoke_check.py
rc=0
for ROW in sleep-replay causal-whatif; do
  line=$(grep -F "/_shards/b2c0925-flipcand/s42/$ROW/" "$JOBS" | head -1)
  [ -n "$line" ] || { echo "⛔ no flipcand s42 line for $ROW in $JOBS" >&2; exit 2; }
  envpart=${line#*" && env "}; envpart=${envpart%%" .venv/bin/python -u -m "*}   # the line's own env prefix
  out_dir=research/findings/raw/_load_bearing/_b2c0925_smoke/s7/$ROW
  cmd="cd ~/derisk-pool/revisions/$F2 && .venv/bin/python tools/assert_flipped_defaults.py && mkdir -p $out_dir"
  cmd+=" && bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 8 --"
  cmd+=" env $envpart .venv/bin/python -u -m research.runners.load_bearing_fraction --only $ROW --seed 7"
  cmd+=" --repeats 2 --out $out_dir/lb.json"
  echo "[b2c-smoke] $NODE: $cmd"
  [ "$PRINT" -eq 1 ] && continue

  logf=$(mktemp "${TMPDIR:-/tmp}/b2c_smoke_${ROW}.XXXXXX.log")
  "${SSH[@]}" "$NODE" "$cmd" >"$logf" 2>&1; r=$?
  cat "$logf"
  echo "[b2c-smoke] $ROW remote exit $r"
  if grep -qF "DA tag-and-capture tick failed" "$logf"; then
    echo "⛔ [b2c-smoke] $ROW: idle-tick error only LOGGED on the remote (webapp/continuous_engine.py) --" \
         "no JSON field sees this; treated as a fail" >&2
    [ "$r" -eq 0 ] && r=1
  fi
  rm -f "$logf"

  local_dir="$LOCAL_BASE/s7/$ROW"; mkdir -p "$local_dir"
  if ! timeout 30 rsync -q -e "$RSYNC_E" "$NODE:derisk-pool/revisions/$F2/$out_dir/" "$local_dir/" 2>/dev/null; then
    echo "⛔ [b2c-smoke] $ROW: rsync pull from $NODE failed" >&2
    [ "$rc" -eq 0 ] && rc=1
    continue
  fi
  if [ ! -f "$local_dir/lb.json" ]; then
    echo "⛔ [b2c-smoke] $ROW: pull ran but no lb.json landed at $local_dir" >&2
    [ "$rc" -eq 0 ] && rc=1
    continue
  fi

  check_out=$("$PY" "$CHECKER" "$local_dir" "$ROW"); check_rc=$?
  echo "[b2c-smoke] $ROW checks: $check_out"
  if [ "$r" -ne 0 ]; then
    [ "$rc" -eq 0 ] && rc=$r
  elif [ "$check_rc" -ne 0 ]; then
    [ "$rc" -eq 0 ] && rc=1
  fi
done
exit "$rc"
