#!/usr/bin/env bash
# aws_brain_sanity_check.sh — POST-PROVISION SANITY CHECK. Builds the tiny-demo brain (the same one
# `webapp.server._build_chat_brain('tiny-demo', ...)` / the regression battery build) BOTH locally (the
# reference) and on the just-provisioned remote, then ABORTS unless neuron+synapse counts match EXACTLY.
#
# WHY A COUNT COMPARISON, NOT JUST "did it import" (2026-09-22 finding: "AWS env built a DEGENERATE
# 2-neuron/0-synapse brain ... exercised=0"). aws_provision.sh/aws_cpu_provision.sh already assert their
# specific imports succeed (cupy-sees-a-device, webapp.server imports) -- necessary but NOT sufficient: a fixed
# import gap does not prove no OTHER silent-degenerate path exists on a freshly-provisioned box (a different
# missing package, a CoreSimConfig default that resolves differently under a different Python version, a
# region-manager wiring step that no-ops instead of raising). This is the general instrument underneath all of
# those: compare the ACTUAL measured substrate size against a KNOWN-GOOD reference.
#
# The reference is always built SIM_BACKEND=numpy, even when checking a GPU (cupy) lane: neuron/synapse COUNTS
# are a structural property of the same seed+config (which regions, how many neurons per region, which
# populations get wired) -- the array backend changes how the numbers are COMPUTED at runtime, not the TOPOLOGY
# that gets built. Building a cupy reference locally would also mean launching CUDA work outside
# `tools/gpu_queue.sh`, which this project's compute discipline reserves for genuine dispatched runs.
#
# Usage: tools/aws_brain_sanity_check.sh '<ssh -i key -o ... user@host>' [remote_backend=numpy] [remote_dir=~/sim]
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
SSH="${1:?usage: tools/aws_brain_sanity_check.sh '<ssh command>' [remote_backend] [remote_dir]}"
REMOTE_BACKEND="${2:-numpy}"
# REMOTE_DIR (2026-09-23): the checkout lives at ~/sim on the single-instance CPU/GPU verify lanes but at
# ~/derisk-pool/sim on a pool node (mini-PC or the AWS-as-extra-pool-node lane, tools/aws_pool_node.sh) --
# parametrize instead of forking this script. Default is unchanged for every existing caller.
REMOTE_DIR="${3:-~/sim}"
REF_CACHE="$ROOT/research/queue/.brain_build_reference.json"
LOCAL_OUT=$(mktemp); REMOTE_OUT=$(mktemp)
trap 'rm -f "$LOCAL_OUT" "$REMOTE_OUT"' EXIT

_json_get() {   # _json_get <file> <key> -- prints the value or "" if unparseable/missing
  python3 -c "
import json, sys
try:
    d = json.load(open('$1'))
except Exception:
    print(''); sys.exit(0)
print(d.get('$2', ''))
" 2>/dev/null
}

echo "[sanity] computing the LOCAL reference (tiny-demo brain, SIM_BACKEND=numpy)…"
# PYBIN resolution: a worktree (this script may well run from one -- worktrees are the default build
# environment here) does NOT carry its own .venv, only the shared main checkout does. Same git-common-dir trick
# `tools/gates/lane_starvation.py`'s `_shared_queue_root()` already uses for the analogous "worktree vs shared
# checkout" problem.
PYBIN="$ROOT/.venv/bin/python"
if [ ! -x "$PYBIN" ]; then
  _COMMON_DIR=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || _COMMON_DIR=""
  if [ -n "$_COMMON_DIR" ]; then
    _SHARED_VENV="$(dirname "$_COMMON_DIR")/.venv/bin/python"
    [ -x "$_SHARED_VENV" ] && PYBIN="$_SHARED_VENV"
  fi
fi
[ -x "$PYBIN" ] || PYBIN=python3
LOCAL_BUILT=0
if bash "$ROOT/tools/mem_ok.sh" 8 >/tmp/aws_sanity_mem_ok.log 2>&1; then
  if bash "$ROOT/tools/memcap.sh" 8 -- "$PYBIN" -m tools.brain_build_sanity >"$LOCAL_OUT" 2>/tmp/aws_sanity_local.err; then
    LOCAL_BUILT=1
  fi
  # memcap prints its own banner line first; brain_build_sanity's JSON is always the LAST line it emits. A
  # non-zero exit (brain_build_sanity's own ok=false path) still leaves valid JSON on that last line, so read
  # it regardless of LOCAL_BUILT -- only an EMPTY/unparseable tail should fall through to the cache.
  tail -1 "$LOCAL_OUT" > "${LOCAL_OUT}.tail"; mv "${LOCAL_OUT}.tail" "$LOCAL_OUT"
fi
if [ "$(_json_get "$LOCAL_OUT" ok)" = "True" ]; then
  cp "$LOCAL_OUT" "$REF_CACHE"
elif [ -f "$REF_CACHE" ]; then
  echo "[sanity] local reference build unavailable ($( [ "$LOCAL_BUILT" = 1 ] && cat "$LOCAL_OUT" || echo "mem_ok refused, see /tmp/aws_sanity_mem_ok.log" )) -- falling back to the CACHED reference"
  cp "$REF_CACHE" "$LOCAL_OUT"
else
  echo "[sanity] ⛔ no local reference available (mem_ok refused / local build failed, AND no cached $REF_CACHE)" >&2
  echo "         cannot verify the remote brain size without a trusted reference -- refusing." >&2
  exit 1
fi
LOCAL_N=$(_json_get "$LOCAL_OUT" n_neurons); LOCAL_S=$(_json_get "$LOCAL_OUT" n_synapses)
if [ -z "$LOCAL_N" ] || [ -z "$LOCAL_S" ]; then
  echo "[sanity] ⛔ local reference is unparseable/incomplete: $(cat "$LOCAL_OUT")" >&2
  exit 1
fi
echo "[sanity] local reference: n_neurons=$LOCAL_N n_synapses=$LOCAL_S"

echo "[sanity] building the SAME brain REMOTELY (SIM_BACKEND=$REMOTE_BACKEND, dir=$REMOTE_DIR)…"
$SSH "cd $REMOTE_DIR && SIM_BACKEND=$REMOTE_BACKEND .venv/bin/python -m tools.brain_build_sanity" \
  > "$REMOTE_OUT" 2>/tmp/aws_sanity_remote.err
tail -1 "$REMOTE_OUT" > "${REMOTE_OUT}.tail"; mv "${REMOTE_OUT}.tail" "$REMOTE_OUT"
if [ "$(_json_get "$REMOTE_OUT" ok)" != "True" ]; then
  echo "[sanity] ⛔ the REMOTE brain build FAILED or produced no parseable JSON: $(cat "$REMOTE_OUT")" >&2
  echo "[sanity]    remote stderr tail:" >&2; tail -20 /tmp/aws_sanity_remote.err >&2
  exit 1
fi
REMOTE_N=$(_json_get "$REMOTE_OUT" n_neurons); REMOTE_S=$(_json_get "$REMOTE_OUT" n_synapses)
echo "[sanity] remote result:  n_neurons=$REMOTE_N n_synapses=$REMOTE_S"

if [ "$REMOTE_N" != "$LOCAL_N" ] || [ "$REMOTE_S" != "$LOCAL_S" ]; then
  echo "[sanity] ⛔ DEGENERATE REMOTE BUILD — local n_neurons=$LOCAL_N n_synapses=$LOCAL_S vs" >&2
  echo "           remote n_neurons=$REMOTE_N n_synapses=$REMOTE_S. This is the exact 2026-09-22 failure" >&2
  echo "           mode (\"2-neuron/0-synapse\"). DO NOT dispatch the battery/6-seed run on this box." >&2
  exit 1
fi
echo "[sanity] ✓ remote brain matches the local reference exactly (n_neurons=$REMOTE_N n_synapses=$REMOTE_S)."
