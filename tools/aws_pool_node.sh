#!/usr/bin/env bash
# aws_pool_node.sh — treat ONE AWS r7i.4xlarge (16 vCPU / 128 GiB, on-demand) as an EXTRA mini-PC-pool node,
# fed by tools/pool_autodispatch.sh from the SAME research/queue/pool.queue, with NO change to pool40/41/42's
# behaviour (see tools/pool_autodispatch.sh's "AWS-AS-EXTRA-POOL-NODE" header comment for the shared plumbing:
# a repo-local, gitignored ssh config the dispatcher/provisioner/sync/queue scripts now honour via -F, and a
# gitignored extra-nodes file the dispatcher re-reads every cycle).
#
# WHY (2026-09-23, owner-approved AWS spend up to $50/day): pool41/pool42 are RAM-bound at 15 GB each -- pool
# jobs grow to 4-6 GB, so each node runs ~2 while 20+ jobs queue. One r7i.4xlarge (128 GB) adds real headroom
# without touching the existing nodes or their dispatch logic at all -- node_is_idle's own nproc/load/
# MemAvailable/reservation checks are REUSED unmodified; this script's only job is to make the node NAMEABLE
# and RESOLVABLE the same way pool40/41/42 already are.
#
#   bash tools/aws_pool_node.sh up [node-name]      # launch + provision + verify, THEN wire it in (default: pool1)
#   bash tools/aws_pool_node.sh down [node-name]    # unwire FIRST, pull results, THEN terminate + delete the SG
#   bash tools/aws_pool_node.sh status [node-name]
#
# STATE: research/queue/.aws_<node-name> (e.g. .aws_pool1) -- its OWN file, separate from the single-instance
# `.aws_gpu` lane, so both can run at once and tools/aws_idle_stop.sh (which already globs
# `research/queue/.aws_*`) finds its key without any change to that script. `down` appends a "TORN DOWN" record
# to the SAME file rather than deleting it, preserving the audit trail (mirrors `.aws_gpu`'s own durability
# intent: state lives in a file, not in memory).
#
# ORDERING, both directions, is the load-bearing part of this script (see cmd_up/cmd_down below for why):
#   up:   launch -> provision (via a TEMPORARY ssh alias, using the instance's raw IP+key) -> verify
#         non-degenerate -> ONLY THEN write the PERSISTENT ssh Host entry + register for dispatch.
#         A node that fails provisioning or the sanity check is NEVER wired in, so the dispatcher can never
#         send it work.
#   down: unregister for dispatch FIRST (no job can land on a node about to disappear) -> pull whatever it
#         already produced -> terminate + delete its SG -> mark the state file torn down.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
REGION="${AWS_REGION:-us-east-1}"
TYPE="${AWS_POOL_NODE_TYPE:-r7i.4xlarge}"

KNOWN_HOSTS="${POOL_SSH_KNOWN_HOSTS:-$ROOT/research/queue/.pool_ssh_known_hosts}"

_state_get() { awk -F= -v k="^$1=" '$0 ~ k {print substr($0, index($0,"=")+1)}' "$STATE" 2>/dev/null | tail -1; }

_write_host_block() {   # _write_host_block <file> <alias> <ip> <key>  -- appends (or replaces) one Host block
  local file="$1" alias="$2" ip="$3" key="$4"
  mkdir -p "$(dirname "$file")"
  if [ -f "$file" ] && grep -q "^Host $alias\$" "$file" 2>/dev/null; then
    # Replace an existing block for this alias (a re-`up` after a torn-down node re-launched with a new IP).
    awk -v a="Host $alias" '
      $0==a {skip=1}
      skip && /^Host / && $0!=a {skip=0}
      !skip {print}
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
  fi
  {
    echo "Host $alias"
    echo "  HostName $ip"
    echo "  User ubuntu"
    echo "  IdentityFile $key"
    echo "  StrictHostKeyChecking accept-new"
    echo "  UserKnownHostsFile $KNOWN_HOSTS"
  } >> "$file"
}

_remove_host_block() {   # _remove_host_block <file> <alias>
  local file="$1" alias="$2"
  [ -f "$file" ] || return 0
  awk -v a="Host $alias" '
    $0==a {skip=1; next}
    skip && /^Host / {skip=0}
    !skip {print}
  ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

if [ "${1:-}" = "--write-host-block" ]; then
  # TEST SEAM (2026-09-23): exercises _write_host_block's add-then-replace logic directly, with NO AWS/ssh
  # call, so it can be unit-tested without a real launch (this repo's build-lane rule: "do not launch a real
  # instance" for this feature). Bypasses the node-name validation below, since arg 2 here is a FILE path.
  [ "$#" -eq 5 ] || { echo "usage: $0 --write-host-block <file> <alias> <ip> <key>" >&2; exit 2; }
  _write_host_block "$2" "$3" "$4" "$5"; exit 0
fi
if [ "${1:-}" = "--remove-host-block" ]; then
  [ "$#" -eq 3 ] || { echo "usage: $0 --remove-host-block <file> <alias>" >&2; exit 2; }
  _remove_host_block "$2" "$3"; exit 0
fi

CMD="${1:-status}"
NODE_NAME="${2:-pool1}"
case "$NODE_NAME" in
  ""|*[!a-zA-Z0-9_-]*) echo "⛔ invalid node name: '$NODE_NAME' (letters/digits/-/_ only)" >&2; exit 2 ;;
esac
# AWS_POOL_NODE_STATE_FILE override exists for tests (mirrors AWS_GPU_STATE_FILE / AWS_CPU_STATE_FILE
# elsewhere in this repo's AWS tooling) -- keeps them from writing into the shared repo's research/queue/.
STATE="${AWS_POOL_NODE_STATE_FILE:-$ROOT/research/queue/.aws_${NODE_NAME}}"
SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-$ROOT/research/queue/.pool_extra_nodes}"

cmd_up() {
  [ -f "$STATE" ] && ! grep -q '^# TORN DOWN' "$STATE" && {
    echo "⛔ $NODE_NAME already recorded live in $STATE — 'down' it first, or pick a different node-name." >&2
    exit 1
  }
  # Informative early exit (the AUTHORITATIVE refusal is inside aws_cpu_launch.sh itself, which this calls
  # next -- see tools/aws_budget.sh's own docstring; not duplicated here to avoid two gates drifting apart).
  bash "$ROOT/tools/aws_budget.sh" check "$TYPE" || {
    echo "⛔ aws_pool_node.sh up: refused by tools/aws_budget.sh (daily cap) — see above" >&2; exit 1; }

  echo "[aws-pool-node] launching $TYPE (state file $STATE)…"
  AWS_CPU_STATE_FILE="$STATE" bash "$ROOT/tools/aws_cpu_launch.sh" || {
    echo "⛔ aws_pool_node.sh up: aws_cpu_launch.sh failed — see above (no ssh entry written, nothing registered)" >&2
    exit 1
  }

  IID=$(_state_get instance); KEY=$(_state_get key)
  [ -n "$IID" ] && [ -n "$KEY" ] || { echo "⛔ $STATE has no instance/key after launch" >&2; exit 1; }
  IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
        --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
  [ -n "$IP" ] && [ "$IP" != "None" ] || { echo "⛔ instance $IID has no public IP" >&2; exit 1; }
  echo "[aws-pool-node] instance=$IID ip=$IP"

  # PROVISION VIA A TEMPORARY ALIAS (the "only THEN write the ssh Host entry" ordering). A throwaway ssh
  # config, scoped to this one call, lets tools/pool_provision.sh -- unmodified in its own node-selection
  # logic, just honouring POOL_SSH_CONFIG like every other pool script now does -- build the SAME
  # ~/derisk-pool/sim layout + venv (identical pinned numpy/scipy/etc versions) + corpus/LTM sync + local-vs-
  # remote non-degenerate-brain sanity compare it already runs for pool40/41/42, with ZERO special-casing for
  # "this one is on AWS". If ANY of that fails, the instance is left launched-but-unwired: `down` (or manual
  # cleanup via tools/aws_gpu.sh-style terminate against $STATE) is the recovery path, not a silent retry.
  STAGING_ALIAS="aws-pool-staging-$$"
  TMP_CONFIG=$(mktemp)
  trap 'rm -f "$TMP_CONFIG"' RETURN
  _write_host_block "$TMP_CONFIG" "$STAGING_ALIAS" "$IP" "$KEY"

  echo "[aws-pool-node] provisioning ~/derisk-pool/sim on $NODE_NAME via staging alias…"
  if ! POOL_SSH_CONFIG="$TMP_CONFIG" bash "$ROOT/tools/pool_provision.sh" "$STAGING_ALIAS"; then
    echo "⛔ aws_pool_node.sh up: pool_provision.sh FAILED on $NODE_NAME (instance $IID stays LIVE + UNWIRED)." >&2
    echo "   Inspect, or tear down: bash tools/aws_pool_node.sh down $NODE_NAME" >&2
    exit 1
  fi

  echo "[aws-pool-node] non-degenerate-brain sanity check (pool_provision.sh's own compare already ran above;"
  echo "                re-verifying via aws_brain_sanity_check.sh for the SAME instrument aws_cpu_provision uses)…"
  SSH_DIRECT="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ubuntu@$IP"
  if ! bash "$ROOT/tools/aws_brain_sanity_check.sh" "$SSH_DIRECT" numpy '~/derisk-pool/sim'; then
    echo "⛔ aws_pool_node.sh up: sanity check FAILED on $NODE_NAME (instance $IID stays LIVE + UNWIRED)." >&2
    echo "   Inspect, or tear down: bash tools/aws_pool_node.sh down $NODE_NAME" >&2
    exit 1
  fi

  # ONLY NOW: wire the PERSISTENT alias in, and register for dispatch. Everything above used the throwaway
  # staging alias; the dispatcher/provisioner/sync scripts will only ever see "$NODE_NAME" from here on.
  if [ ! -f "$SSH_CONFIG" ]; then
    mkdir -p "$(dirname "$SSH_CONFIG")"
    { echo "# Generated by tools/aws_pool_node.sh -- repo-local, gitignored. NEVER edit ~/.ssh/config for this."
      echo "Include ~/.ssh/config"
      echo; } > "$SSH_CONFIG"
  fi
  _write_host_block "$SSH_CONFIG" "$NODE_NAME" "$IP" "$KEY"
  mkdir -p "$(dirname "$EXTRA_NODES_FILE")"; touch "$EXTRA_NODES_FILE"
  grep -qxF "$NODE_NAME" "$EXTRA_NODES_FILE" 2>/dev/null || echo "$NODE_NAME" >> "$EXTRA_NODES_FILE"
  echo "[aws-pool-node] ✓ $NODE_NAME is LIVE, provisioned, and registered in $EXTRA_NODES_FILE."
  echo "                tools/pool_autodispatch.sh picks it up within one poll cycle (no restart needed)."
}

cmd_down() {
  [ -f "$STATE" ] || { echo "no state recorded for $NODE_NAME at $STATE — nothing to tear down"; exit 0; }
  if grep -q '^# TORN DOWN' "$STATE"; then
    echo "$NODE_NAME is already torn down (see $STATE)"; exit 0
  fi
  IID=$(_state_get instance); REGION_S=$(_state_get region); SG=$(_state_get sg)
  REGION_S="${REGION_S:-$REGION}"

  # 1. UNREGISTER FIRST — no new job can be dispatched to a node that is about to disappear. This is a pure
  #    local file edit (no network round-trip), so by the time step 2 below even starts, the dispatcher's
  #    NEXT cycle (which re-reads this file every time, see pool_autodispatch.sh) will no longer offer it work.
  if [ -f "$EXTRA_NODES_FILE" ]; then
    grep -vxF "$NODE_NAME" "$EXTRA_NODES_FILE" > "$EXTRA_NODES_FILE.tmp" 2>/dev/null || true
    mv "$EXTRA_NODES_FILE.tmp" "$EXTRA_NODES_FILE"
  fi
  echo "[aws-pool-node] $NODE_NAME removed from $EXTRA_NODES_FILE (dispatcher will not target it again)."

  # 2. PULL RESULTS while the node (and its persistent ssh alias, untouched so far) is still reachable.
  echo "[aws-pool-node] pulling any results $NODE_NAME already produced…"
  POOL_NODES="$NODE_NAME" bash "$ROOT/tools/pool_sync.sh" || \
    echo "  (pool_sync reported an issue — continuing with teardown; nothing is lost from the AWS side until terminate)" >&2

  # 3. TERMINATE + delete the SG — only after (1) and (2).
  if [ -n "$IID" ]; then
    echo "[aws-pool-node] terminating $IID…"
    aws ec2 terminate-instances --instance-ids "$IID" --region "$REGION_S" \
      --query 'TerminatingInstances[].CurrentState.Name' --output text
  fi
  if [ -n "$SG" ]; then
    echo "[aws-pool-node] deleting security group $SG (best-effort; AWS can take a few seconds to release it)…"
    aws ec2 delete-security-group --group-id "$SG" --region "$REGION_S" >/dev/null 2>&1 || \
      echo "  (SG delete failed/pending — retry later: aws ec2 delete-security-group --group-id $SG --region $REGION_S)" >&2
  fi

  # 4. Remove the now-stale persistent ssh Host entry, and mark the state file torn down (never delete it —
  #    same durability intent as .aws_gpu: the record of what ran and when survives).
  _remove_host_block "$SSH_CONFIG" "$NODE_NAME"
  { echo "# TORN DOWN $(date '+%F %T %Z')"; cat "$STATE"; } > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
  echo "[aws-pool-node] ✓ $NODE_NAME torn down."
}

cmd_status() {
  if [ ! -f "$STATE" ]; then echo "$NODE_NAME: not launched (no $STATE)"; exit 0; fi
  if grep -q '^# TORN DOWN' "$STATE"; then
    echo "$NODE_NAME: TORN DOWN ($(grep '^# TORN DOWN' "$STATE" | head -1 | sed 's/^# //'))"
    exit 0
  fi
  IID=$(_state_get instance); REGION_S=$(_state_get region); REGION_S="${REGION_S:-$REGION}"
  read -r EC2_STATE EC2_IP <<<"$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
      --query 'Reservations[].Instances[].[State.Name,PublicIpAddress]' --output text 2>/dev/null)"
  echo "$NODE_NAME: instance=$IID state=${EC2_STATE:-unknown} ip=${EC2_IP:-none}"
  if grep -qxF "$NODE_NAME" "$EXTRA_NODES_FILE" 2>/dev/null; then
    echo "  registered for dispatch: yes ($EXTRA_NODES_FILE)"
  else
    echo "  registered for dispatch: no"
  fi
  if [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    OUT=$(timeout 10 ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME" \
      "echo \$(nproc) \$(pgrep -c -f '^[^ ]*/?python[0-9.]* .*-m [r]esearch\.runners' 2>/dev/null | head -1) \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo)" 2>/dev/null)
    if [ -n "$OUT" ]; then
      set -- $OUT
      echo "  cores=${1:-?} running_runners=${2:-?} MemAvailable=${3:-?}GB"
    else
      echo "  (unreachable via ssh -F $SSH_CONFIG $NODE_NAME)"
    fi
  else
    echo "  (no ssh Host entry yet -- not provisioned/wired)"
  fi
}

case "$CMD" in
  up)     cmd_up ;;
  down)   cmd_down ;;
  status) cmd_status ;;
  *) echo "usage: aws_pool_node.sh {up|down|status} [node-name=pool1]" >&2; exit 2 ;;
esac
