#!/usr/bin/env bash
# status.sh -- ONE read-only snapshot for weekend local-model supervision (see
# research/coordination/LOCAL_LLM_RUNBOOK.md). NEVER MODIFIES ANYTHING: every external call here is a `status`/
# `depth`/read-only query with a short timeout, so this is always safe to run, including concurrently with a live
# run. Reuses tools/gpu_queue.sh, tools/pool_queue.sh, tools/aws_budget.sh and tools/battery_status.py rather than
# re-deriving their logic. Output is kept to <= 40 lines by design (short per-section summaries, bounded lists).
#
#   tools/status.sh
#
# Env overrides (tests only -- never set these in production; each is honoured by the underlying tool it names):
#   GPU_QUEUE_DIR, POOL_QUEUE_PATH, POOL_NODES, POOL_SSH_CONFIG, AWS_DAILY_CAP_USD, SIM_ENGINE_PYTHON,
#   HANDOFF_BATTERIES_TSV, STATUS_SSH (ssh binary override, for a stub in tests)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${SIM_ENGINE_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
TSV="${HANDOFF_BATTERIES_TSV:-$ROOT/research/coordination/handoff_batteries.tsv}"
NODES="${POOL_NODES:-pool40 pool41 pool42}"
SSH_BIN="${STATUS_SSH:-ssh}"
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")

echo "== status $(date -u '+%F %T UTC') =="

# ---- GPU (tools/gpu_queue.sh owns the shared queue's location + nvidia-smi timeouts) ----------------------------
GPU_OUT=$(timeout 12 bash "$ROOT/tools/gpu_queue.sh" status 2>/dev/null)
echo "-- GPU --"
printf '%s\n' "$GPU_OUT" | grep -E '^(state:|current:|queued:|dispatcher:)' | sed 's/^/  /'
GPU_LOG=$(printf '%s\n' "$GPU_OUT" | sed -n 's/^shared queue: //p')/gpu_queue.log
GPU_TEXT=$(printf '%s\n' "$GPU_OUT" | grep '^current:')

# ---- Pool: queue depth (tools/pool_queue.sh) + running/unreachable per node (direct read-only ssh, short timeout)
echo "-- Pool --"
QDEPTH=$(timeout 10 bash "$ROOT/tools/pool_queue.sh" depth 2>/dev/null); QDEPTH=${QDEPTH:-?}
echo "  queue depth: $QDEPTH"
POOL_TEXT=""; POOL_FAIL_ARR=()
remote_probe() {  # printed via stdin (process substitution) so no fragile nested-quote nesting; no `-n` needed
  # since this is not inside a while-read loop, ssh consuming its own stdin here is harmless (see
  # tools/gates/ssh_stdin_in_read_loop.py's scope: it only guards a `while`/`until read` loop's shared fd 0).
  cat <<'REMOTE'
r=$(pgrep -fc '[r]esearch\.runners' 2>/dev/null | head -1); r=${r:-0}
f=0
if [ -f "$HOME/derisk-pool/sim/job_status.log" ]; then
  f=$(awk -F'\t' -v now="$(date +%s)" '
    $1=="v2" { if ($3!=0 && (now-$2)<=86400) c++; next }
    # non-v2 lines carry no timestamp, so they cannot be placed in the last 24 h: not counted (2026-09-25:
    # counting them reported 94 'failures' on pool42 whose real 24 h failures were 6 deliberate SIGTERMs)
    END { print c+0 }' "$HOME/derisk-pool/sim/job_status.log")
fi
echo "$r $f"
REMOTE
}
for n in $NODES; do
  R=$(timeout 8 "$SSH_BIN" "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=5 "$n" 'bash -s' < <(remote_probe) 2>/dev/null)
  if [ -z "$R" ]; then
    echo "  $n: unreachable"
  else
    set -- $R
    run="${1:-0}"; fail="${2:-0}"
    echo "  $n: running=$run fail24h=$fail"
    [ "$run" -gt 0 ] && POOL_TEXT="$POOL_TEXT $n:running"
    [ "$fail" -gt 0 ] && POOL_FAIL_ARR+=("  $n: $fail non-zero job(s) in job_status.log (last 24h)")
  fi
done

# ---- AWS spend vs cap + instance count (tools/aws_budget.sh; describe-instances is a read-only AWS call) --------
echo "-- AWS --"
AWS_OUT=$(timeout 15 bash "$ROOT/tools/aws_budget.sh" status 2>/dev/null)
printf '%s\n' "$AWS_OUT" | head -1 | sed 's/^/  /'
N_INST=$(printf '%s\n' "$AWS_OUT" | grep -cE '^\s+i-')
[ "${N_INST:-0}" -gt 0 ] && printf '%s\n' "$AWS_OUT" | grep -E '^\s+i-' | sed 's/^/  /' || echo "  (no project instances)"

# ---- local-llm unit ------------------------------------------------------------------------------------------
echo "-- local-llm --"
# `is-active` PRINTS a state word (active/inactive/failed/...) on BOTH a zero AND non-zero exit -- unlike most
# commands, so `cmd || echo inactive` would append a SECOND "inactive" line whenever the real state already was
# inactive (caught by running this once: output had "inactive" twice). Only the truly-empty case (unit unknown
# to this user's systemd, e.g. never installed) needs the fallback.
LLM_STATE=$(timeout 5 systemctl --user is-active local-llm 2>/dev/null)
[ -z "$LLM_STATE" ] && LLM_STATE="not-installed"
echo "  local-llm: $LLM_STATE"

# ---- Batteries (tools/battery_status.py; "running" text = GPU current job + pool nodes with running>0) ----------
echo "-- Batteries --"
RUNNING_TEXT="$GPU_TEXT $POOL_TEXT"
if [ -f "$TSV" ]; then
  "$PY" "$ROOT/tools/battery_status.py" --tsv "$TSV" --root "$ROOT" --running-text "$RUNNING_TEXT" 2>/dev/null \
    | awk -F'\t' '{printf "  %-20s %s/%s  %s\n", $1, $2, $3, $4}'
else
  echo "  (no battery registry at $TSV)"
fi

# ---- Recent failures: GPU log DONE(rc!=0) in the last 24h + pool job_status non-zero (gathered above) -----------
echo "-- Recent failures (24h) --"
NOW=$(date +%s); ANY=0
if [ -f "$GPU_LOG" ]; then
  while IFS= read -r line; do
    ts="${line:0:19}"
    epoch=$(date -d "$ts" +%s 2>/dev/null) || continue
    age=$((NOW - epoch))
    if [ "$age" -ge 0 ] && [ "$age" -le 86400 ]; then echo "  $line" | cut -c1-100; ANY=1; fi
  done < <(grep -E 'DONE\(rc=[1-9]' "$GPU_LOG" 2>/dev/null | tail -20)
fi
if [ "${#POOL_FAIL_ARR[@]}" -gt 0 ]; then printf '%s\n' "${POOL_FAIL_ARR[@]}"; ANY=1; fi
# `if` (not `test && echo`) deliberately: a read-only status report must exit 0 on the common/healthy case too
# (the exact bug fixed 2026-09-09 in tools/gpu_queue.sh's `status` -- see tests/test_gpu_queue_status_exit.py).
if [ "$ANY" -eq 0 ]; then echo "  (none)"; fi
exit 0
