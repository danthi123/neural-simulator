#!/usr/bin/env bash
# tools/dispatcher_selftest.sh — the compute dispatchers get the same discipline as the gates: a selftest
# that FAILS IN THE FAILING DIRECTION, so a broken dispatcher fails loudly instead of idling the fleet.
#
# EARNED 2026-08-01. Two silent-broken-dispatcher bugs cost ~an hour while every liveness signal said healthy:
#   (a) lane_dispatch running_count did `pgrep -c PAT || echo 0` -- pgrep -c ALREADY prints 0 and exits 1 when
#       nothing matches, so the fallback double-emitted "0\n0", crashing `$(( SLOTS - N ))` at 0-running (the
#       startup case). The dispatcher never dispatched a single job, invisibly.
#   (b) pool jobs were staged with bare `python` -- the nodes have NONE (only .venv/bin/python) -- so they
#       dispatched and produced nothing. The staging checks all shelled out to .venv/bin/python themselves, so
#       the bad command passed validation and died on the node.
# Both are now guarded (lane_dispatch startup self-check; pool_queue.sh interpreter guard). This asserts it.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0; ok(){ printf '  ok   %s\n' "$*"; }; bad(){ printf '  FAIL %s\n' "$*"; FAIL=1; }

echo "── dispatcher self-test ──"

# (a) running_count must return ONE clean integer at 0-running, and the FREE arithmetic must not crash.
N=$(pgrep -fc '[.]venv/bin/python .*-m research' 2>/dev/null); N="${N:-0}"
if [[ "$N" =~ ^[0-9]+$ ]] && FREE=$(( 3 - N )) 2>/dev/null; then ok "running_count clean int ($N); FREE=$FREE (no crash)"
else bad "running_count not a clean int at 0-running: [$N] -> \$(( SLOTS - N )) would crash"; fi

# (b1) pool_queue.sh must REJECT a bare-python research command (the nodes have no bare python).
if bash tools/pool_queue.sh add 'SIM_BACKEND=numpy python -u -m research.runners._selftest_fake --seeds 42' --checked selftest >/dev/null 2>&1; then
  bad "pool_queue ACCEPTED a bare-python command (would dispatch + produce nothing)"
else ok "pool_queue rejects a bare-python command"; fi

# (b2) the interpreter guard must NOT fire on a venv command (it may be refused later for the fake module -- a
#      DIFFERENT refusal -- but it must get PAST the interpreter check).
out=$(bash tools/pool_queue.sh add 'SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._selftest_fake --seeds 42' --checked selftest 2>&1 || true)
if printf '%s' "$out" | grep -q "BARE 'python'"; then bad "interpreter guard wrongly fired on a .venv/bin/python command"
else ok "interpreter guard passes a venv command through to the module check"; fi

# (c) lane_dispatch parses and its startup self-check line exists (the loud-fail-on-broken-count guard).
if bash -n tools/lane_dispatch.sh && grep -q "DISPATCHER SELF-CHECK FAILED" tools/lane_dispatch.sh; then
  ok "lane_dispatch parses + carries the startup self-check"
else bad "lane_dispatch missing its startup self-check or fails to parse"; fi

# (d) Generic pool enqueue must emit the timestamped format consumed by the
# pool dispatcher. This previously emitted a GPU-style bare command.
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
POOL_QUEUE_PATH="$T/pool.queue" bash tools/queue_add.sh \
  pool 'printf pool-format-ok' selftest >/dev/null 2>&1
POP=$(POOL_QUEUE_PATH="$T/pool.queue" POOL_RUNNING_PATH="$T/pool.running" \
  bash tools/pool_autodispatch.sh --pop-once 2>"$T/pop.err")
if [ "$POP" = "printf pool-format-ok" ] && \
   grep -q 'printf pool-format-ok  #checked:selftest' "$T/pool.queue.claims"; then
  ok "generic pool enqueue, dispatcher, and claim record agree on format"
else bad "pool producer/consumer mismatch: dispatcher recovered [$POP]"; fi

# (e) A malformed direct edit must be preserved and refused rather than
# counting forever as work in transit while every node remains idle.
printf '%s\n' 'printf never-dispatch  #checked:selftest' > "$T/pool.queue"
POP=$(POOL_QUEUE_PATH="$T/pool.queue" POOL_RUNNING_PATH="$T/pool.running" \
  bash tools/pool_autodispatch.sh --pop-once 2>"$T/malformed.err")
if [ -z "$POP" ] && [ ! -s "$T/pool.queue" ] && \
   grep -q 'never-dispatch' "$T/pool.queue.malformed" && \
   grep -q 'BLOCKED + quarantined' "$T/malformed.err"; then
  ok "malformed pool records fail loudly and are preserved"
else bad "malformed pool record was lost, hidden, or dispatchable"; fi

[ "$FAIL" = 0 ] && { echo "DISPATCHER SELFTEST PASS"; exit 0; } || { echo "DISPATCHER SELFTEST FAIL"; exit 1; }
