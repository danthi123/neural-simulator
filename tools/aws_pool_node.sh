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
#   bash tools/aws_pool_node.sh down [node-name] [--force]   # drain, THEN terminate + delete the SG
#   bash tools/aws_pool_node.sh status [node-name]
#
# STATE: research/queue/.aws_<node-name> (e.g. .aws_pool1) -- its OWN file, separate from the single-instance
# `.aws_gpu` lane, so both can run at once and tools/aws_idle_stop.sh (which already globs
# `research/queue/.aws_*`) finds its key without any change to that script. `down` appends a "TORN DOWN" record
# to the SAME file rather than deleting it, preserving the audit trail (mirrors `.aws_gpu`'s own durability
# intent: state lives in a file, not in memory). Gitignored (research/queue/.aws_pool*) -- it holds an instance
# id, SG id and a local key PATH (never key material), but a broad `git add research/queue` should not commit it.
#
# ORDERING, both directions, is the load-bearing part of this script (see cmd_up/cmd_down below for why):
#   up:   launch -> provision (via a TEMPORARY ssh alias, using the instance's raw IP+key) -> verify
#         non-degenerate -> pre-provision every git revision the LIVE queue already references (best-effort,
#         --isolated, so revision-pinned jobs are runnable on arrival rather than stranding on this node until
#         someone remembers) -> ONLY THEN write the PERSISTENT ssh Host entry + register for dispatch.
#         A node that fails provisioning or the sanity check is NEVER wired in -- and is auto-torn-down
#         (terminate + delete SG) rather than left live-and-unwired for aws_idle_stop.sh to eventually notice.
#   down: unregister for dispatch FIRST (no job can land on a node about to disappear) -> DRAIN (wait for any
#         runner already in flight there to finish, since the root volume is DeleteOnTermination=true and a
#         killed-mid-run job's output is unrecoverable) -> pull whatever it produced -> refuse to terminate if
#         the sync failed or a runner is still (somehow) running, unless --force -> terminate + delete its SG
#         -> mark the state file torn down.
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

_running_runners() {   # _running_runners <ssh-alias-or-command-prefix...> -- prints a count, "" if unreachable
  # Same bracketed pgrep pattern as pool_autodispatch.sh's node_is_idle (an un-bracketed one self-matches the
  # ssh command carrying it). Used by `down`'s drain wait and its refuse-while-running guard.
  timeout 10 "$@" "pgrep -c -f '^[^ ]*/?python[0-9.]* .*-m [r]esearch\.runners' 2>/dev/null | head -1" 2>/dev/null
}

_terminate_and_delete_sg() {   # _terminate_and_delete_sg <instance-id> <sg-id> <region>
  local iid="$1" sg="$2" region="$3"
  if [ -n "$iid" ]; then
    echo "[aws-pool-node] terminating $iid…"
    aws ec2 terminate-instances --instance-ids "$iid" --region "$region" \
      --query 'TerminatingInstances[].CurrentState.Name' --output text
  fi
  if [ -n "$sg" ]; then
    echo "[aws-pool-node] deleting security group $sg (best-effort; AWS can take a few seconds to release it)…"
    aws ec2 delete-security-group --group-id "$sg" --region "$region" >/dev/null 2>&1 || \
      echo "  (SG delete failed/pending — retry later: aws ec2 delete-security-group --group-id $sg --region $region)" >&2
  fi
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
# --force (down only): skip the drain-refusal guard (runners still running / sync failed). Any position after
# the node-name; `down testnode --force` is the documented form.
FORCE=0
for _a in "${@:3}"; do [ "$_a" = "--force" ] && FORCE=1; done
# AWS_POOL_NODE_STATE_FILE override exists for tests (mirrors AWS_GPU_STATE_FILE / AWS_CPU_STATE_FILE
# elsewhere in this repo's AWS tooling) -- keeps them from writing into the shared repo's research/queue/.
STATE="${AWS_POOL_NODE_STATE_FILE:-$ROOT/research/queue/.aws_${NODE_NAME}}"
SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-$ROOT/research/queue/.pool_extra_nodes}"
POOL_QUEUE_FILE="${POOL_QUEUE_PATH:-$ROOT/research/queue/pool.queue}"

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

  # AUTO-TEARDOWN ON ANY FAILURE FROM HERE ON (2026-09-23 fix round). Before this, a provision/sanity failure
  # left the instance LIVE + UNWIRED with only aws_idle_stop.sh's ~20min idle-STOP (not terminate) as a
  # backstop -- billing continued and the SG/instance leaked until someone noticed. `_up_failed` terminates +
  # deletes the SG + marks the state file torn down (so a retried `up NODE_NAME` is not blocked by
  # "already recorded live") before propagating the failure.
  REGION_S=$(_state_get region); REGION_S="${REGION_S:-$REGION}"; SG=$(_state_get sg)
  _up_failed() {
    local reason="$1"
    echo "⛔ aws_pool_node.sh up: $reason — auto-tearing-down $NODE_NAME (instance $IID) rather than leaving it live+unwired." >&2
    _terminate_and_delete_sg "$IID" "$SG" "$REGION_S"
    { echo "# TORN DOWN $(date '+%F %T %Z') (auto, up failed: $reason)"; cat "$STATE"; } > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
    exit 1
  }

  # PROVISION VIA A TEMPORARY ALIAS (the "only THEN write the ssh Host entry" ordering). A throwaway ssh
  # config, scoped to this one call, lets tools/pool_provision.sh -- unmodified in its own node-selection
  # logic, just honouring POOL_SSH_CONFIG like every other pool script now does -- build the SAME
  # ~/derisk-pool/sim layout + venv (identical pinned numpy/scipy/etc versions) + corpus/LTM sync + local-vs-
  # remote non-degenerate-brain sanity compare it already runs for pool40/41/42, with ZERO special-casing for
  # "this one is on AWS". If ANY of that fails, `_up_failed` above tears the instance back down.
  STAGING_ALIAS="aws-pool-staging-$$"
  TMP_CONFIG=$(mktemp)
  # BUGFIX (fix round): a RETURN trap only fires when a function returns via `return`/falling off the end --
  # NOT via `exit`, which is how every failure path here leaves the function. TMP_CONFIG leaked on every
  # provision/sanity failure. EXIT fires unconditionally (normal exit, `exit N`, or an unhandled error), so it
  # is cleared here explicitly on function return AND fires on any `exit` below.
  trap 'rm -f "$TMP_CONFIG"' EXIT
  _write_host_block "$TMP_CONFIG" "$STAGING_ALIAS" "$IP" "$KEY"

  echo "[aws-pool-node] provisioning ~/derisk-pool/sim on $NODE_NAME via staging alias…"
  if ! POOL_SSH_CONFIG="$TMP_CONFIG" bash "$ROOT/tools/pool_provision.sh" "$STAGING_ALIAS"; then
    _up_failed "pool_provision.sh FAILED on $NODE_NAME"
  fi

  echo "[aws-pool-node] non-degenerate-brain sanity check (pool_provision.sh's own compare already ran above;"
  echo "                re-verifying via aws_brain_sanity_check.sh for the SAME instrument aws_cpu_provision uses)…"
  # UserKnownHostsFile pinned to the repo-local file (2026-09-23 fix round): without it this direct ssh call
  # (bypassing -F/POOL_SSH_CONFIG entirely) wrote the AWS IP into the user's DEFAULT ~/.ssh/known_hosts via
  # accept-new -- a recycled AWS IP later reused by an unrelated host then fails ssh with a host-key mismatch
  # that has nothing to do with this project. The persistent Host block below already scopes to $KNOWN_HOSTS;
  # this direct call should too.
  SSH_DIRECT="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS -o ConnectTimeout=15 ubuntu@$IP"
  if ! bash "$ROOT/tools/aws_brain_sanity_check.sh" "$SSH_DIRECT" numpy '~/derisk-pool/sim'; then
    _up_failed "sanity check FAILED on $NODE_NAME"
  fi

  # ONLY NOW: wire the PERSISTENT alias in, and register for dispatch. Everything above used the throwaway
  # staging alias; the dispatcher/provisioner/sync scripts will only ever see "$NODE_NAME" from here on.
  if [ ! -f "$SSH_CONFIG" ]; then
    mkdir -p "$(dirname "$SSH_CONFIG")"
    { echo "# Generated by tools/aws_pool_node.sh -- repo-local, gitignored. NEVER edit ~/.ssh/config for this."
      echo "Include ~/.ssh/config"
      # Include the system config too (2026-09-23 fix round): `-F <this file>` makes ssh skip
      # /etc/ssh/ssh_config (and its own ssh_config.d/*.conf Include) ENTIRELY, not just ~/.ssh/config -- so
      # once ANY AWS pool node is upped, pool40/41/42 connections silently stopped seeing system-wide ssh
      # settings (proxy config, host-wide ciphers, etc). Include both, matching ssh's own default read order.
      echo "Include /etc/ssh/ssh_config"
      echo; } > "$SSH_CONFIG"
  fi
  _write_host_block "$SSH_CONFIG" "$NODE_NAME" "$IP" "$KEY"

  # PRE-PROVISION every git revision the LIVE queue already references (2026-09-23 fix round, best-effort).
  # pool.queue's revision-pinned jobs (`cd ~/derisk-pool/revisions/<sha> && ...`, from `pool_provision.sh
  # --isolated`) would otherwise all target a node whose ONLY provisioned tree is ~/derisk-pool/sim -- the
  # dispatcher-side revision check (pool_autodispatch.sh) now skips this node for those jobs rather than losing
  # them, but that just leaves them permanently unable to use the new capacity. Do it here, once, up front.
  # POOL_PROVISION_ALLOW_STALE=1: a queued revision is very likely BEHIND current origin/main by the time this
  # runs (queued a while ago) -- that is exactly the "deliberate reproduction of an old state" case the stale-
  # source guard's override exists for, not a mistake to refuse. Best-effort: a failure here does not fail
  # `up` (the node is still usable for non-pinned / current-HEAD work); it just leaves that job queued for
  # another node, same as any other missing-revision case.
  if [ -f "$POOL_QUEUE_FILE" ]; then
    QUEUED_SHAS=$(grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' "$POOL_QUEUE_FILE" 2>/dev/null | sed 's#.*/##' | sort -u)
    for _sha in $QUEUED_SHAS; do
      echo "[aws-pool-node] pre-provisioning queued revision $_sha on $NODE_NAME…"
      if ! POOL_SSH_CONFIG="$SSH_CONFIG" POOL_PROVISION_ALLOW_STALE=1 \
           bash "$ROOT/tools/pool_provision.sh" --revision "$_sha" --isolated "$NODE_NAME"; then
        echo "  ⛔ WARNING: could not provision revision $_sha on $NODE_NAME -- jobs pinned to it will stay queued for another node." >&2
      fi
    done
  fi

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

  # 2. DRAIN — wait for any runner already in flight on this node to finish (2026-09-23 fix round). The root
  #    volume is DeleteOnTermination=true, so a job killed mid-run by `terminate` loses its output with no
  #    requeue; unregistering (step 1) stops NEW jobs landing but does nothing for one already running. Bounded
  #    wait (AWS_POOL_DRAIN_TIMEOUT_S, default 30 min matches the slowest routine pool job) so `down` cannot
  #    hang forever on a stuck runner -- --force (or the timeout) is the way out of that case.
  if [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    DRAIN_TIMEOUT="${AWS_POOL_DRAIN_TIMEOUT_S:-1800}"
    DRAIN_START=$(date +%s)
    while :; do
      N_RUNNING=$(_running_runners ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME")
      [ -z "${N_RUNNING:-}" ] && N_RUNNING=0   # unreachable -- nothing to drain FOR, do not hang on it
      [ "$N_RUNNING" -eq 0 ] 2>/dev/null && break
      if [ $(( $(date +%s) - DRAIN_START )) -ge "$DRAIN_TIMEOUT" ]; then
        echo "  ⛔ drain timed out after ${DRAIN_TIMEOUT}s with $N_RUNNING runner(s) still running on $NODE_NAME." >&2
        [ "$FORCE" = 1 ] || { echo "     Refusing to terminate (their output would be lost). Re-run with --force to override." >&2; exit 1; }
        break
      fi
      echo "  [aws-pool-node] draining $NODE_NAME: $N_RUNNING runner(s) still in flight, waiting…"
      sleep "${AWS_POOL_DRAIN_POLL_S:-20}"
    done
  fi

  # 3. PULL RESULTS while the node (and its persistent ssh alias, untouched so far) is still reachable.
  echo "[aws-pool-node] pulling any results $NODE_NAME already produced…"
  SYNC_OK=1
  POOL_NODES="$NODE_NAME" bash "$ROOT/tools/pool_sync.sh" || SYNC_OK=0
  if [ "$SYNC_OK" = 0 ]; then
    echo "  ⛔ pool_sync reported an issue pulling results from $NODE_NAME." >&2
    if [ "$FORCE" = 1 ]; then
      echo "     --force set: continuing with teardown anyway (unsynced results on this instance WILL be lost)." >&2
    else
      echo "     Refusing to terminate (the root volume is DeleteOnTermination=true; unpulled results would be" >&2
      echo "     destroyed). Fix connectivity and retry, or re-run with --force to accept the loss." >&2
      exit 1
    fi
  fi
  # RE-CHECK for anything that started (or was still exiting) between the drain wait and here -- best-effort,
  # same unreachable-means-nothing-to-drain-for treatment as step 2.
  if [ "$FORCE" != 1 ] && [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    N_RUNNING=$(_running_runners ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME")
    if [ -n "${N_RUNNING:-}" ] && [ "$N_RUNNING" -gt 0 ] 2>/dev/null; then
      echo "  ⛔ $N_RUNNING runner(s) still running on $NODE_NAME. Refusing to terminate. Re-run with --force to override." >&2
      exit 1
    fi
  fi

  # 4. TERMINATE + delete the SG — only after (1), (2) and (3) all either succeeded or were force-overridden.
  _terminate_and_delete_sg "$IID" "$SG" "$REGION_S"

  # 5. Remove the now-stale persistent ssh Host entry, and mark the state file torn down (never delete it —
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
