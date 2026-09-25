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
#   bash tools/aws_pool_node.sh start [node-name]   # start a STOPPED node (2026-09-25), wait for running+ssh,
#                                                    # then rewrite .pool_ssh_config's Host block to its NEW
#                                                    # public ip (atomically, with a .bak) -- budget-gated
#                                                    # (tools/aws_budget.sh check) before the start-instances
#                                                    # call. Already-running is a cheap Host-block refresh only.
#   bash tools/aws_pool_node.sh refresh [node-name]  # CHEAP: rewrite the Host block ONLY if this node is
#                                                    # already running and its recorded ip is stale/missing --
#                                                    # never starts a stopped instance, never budget-gated.
#                                                    # tools/pool_sync.sh calls this once, automatically, when
#                                                    # an AWS-managed node it tries to sync from is unreachable.
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
  # ATOMIC, UNDER flock (2026-09-25 review, MEDIUM: "_write_host_block not atomic"). The OLD version did TWO
  # separate filesystem operations with no lock at all: an awk-filter-then-`mv` (removing any existing block for
  # this alias), followed by a SEPARATE `>>` append of the new one. A concurrent READER (any `ssh -F "$file"`
  # call, which every pool script makes constantly) could observe the file in between those two steps, with the
  # old block already gone and the new one not yet written -- reproduced: 10/1482 snapshots had the block
  # MISSING entirely, 2/1482 PARTIAL. Two concurrent WRITERS (e.g. a routine pool_sync stale-ip refresh racing
  # an operator's `start`) could each read the SAME pre-edit content and then race their two `mv`s -- reproduced:
  # a DUPLICATE Host block with the STALE ip listed first in 27/30 trials (ssh uses the first match it finds).
  # FIX: hold `flock` on "$file.lock" around the ENTIRE read-modify-write, and build the WHOLE new content
  # (filtered old content, if any, plus the new block) in one `mktemp` file in the SAME directory as $file, then
  # `mv` it ONCE -- one atomic filesystem operation instead of two, and no writer can interleave with another
  # while the lock is held (a second writer waits, then re-reads the FIRST writer's already-updated content, so
  # its own alias-replace logic still finds and replaces cleanly instead of duplicating).
  local file="$1" alias="$2" ip="$3" key="$4"
  local dir; dir="$(dirname "$file")"
  mkdir -p "$dir"
  local lock="$file.lock" tmp
  exec 8>"$lock"
  # -w 30 (2026-09-25 review, LOW/INFO): an unbounded flock can stall pool_autodispatch.sh's MAIN dispatch loop
  # forever -- its _maybe_refresh_stale_aws_node calls `refresh`, which calls this, from inside node_is_idle on
  # every cycle. 30s is generous next to this function's own sub-second normal runtime; a self-heal that fails
  # to acquire the lock in 30s (dispatcher already treats it as best-effort/silent) beats hanging dispatch.
  flock -w 30 8 || { echo "⛔ _write_host_block: timed out waiting for the lock on $lock" >&2; exec 8>&-; return 1; }
  tmp=$(mktemp "$dir/.host_block.XXXXXX") || { flock -u 8; exec 8>&-; return 1; }
  # GUARD EVERY STEP (2026-09-25 review, MEDIUM regression): the prior version ran the filter/copy step, the
  # append, and the `mv` as three UNCHECKED statements -- a failed awk (or a full-disk cat/append) was silently
  # ignored and the `mv` still ran, swapping the real config for a truncated/wrong one while still returning 0.
  # Chain every step with `&&` so ANY failure aborts BEFORE the mv, leaves the original file untouched, and
  # returns non-zero.
  {
    if [ -f "$file" ] && grep -q "^Host $alias\$" "$file" 2>/dev/null; then
      # Replace an existing block for this alias (a re-`up` after a torn-down node re-launched with a new IP).
      awk -v a="Host $alias" '
        $0==a {skip=1}
        skip && /^Host / && $0!=a {skip=0}
        !skip {print}
      ' "$file"
    elif [ -f "$file" ]; then
      cat "$file"
    fi
  } > "$tmp" && {
    echo "Host $alias"
    echo "  HostName $ip"
    echo "  User ubuntu"
    echo "  IdentityFile $key"
    echo "  StrictHostKeyChecking accept-new"
    echo "  UserKnownHostsFile $KNOWN_HOSTS"
  } >> "$tmp" && mv "$tmp" "$file" || {
    rm -f "$tmp"
    flock -u 8; exec 8>&-
    echo "⛔ _write_host_block: failed to rewrite $file -- original content left untouched" >&2
    return 1
  }
  flock -u 8
  exec 8>&-
}

_backup_ssh_config() {   # _backup_ssh_config <file> -- best-effort ONE-PRIOR-VERSION .bak before any rewrite by
  # `start`/`refresh` (2026-09-25). Mirrors this script's existing "never delete, mark instead" durability
  # intent (the state file's own "# TORN DOWN" convention) at the scale a "keep a backup" ask calls for -- a
  # single last-good copy an owner can diff/restore from by hand, not a version history. Never touches
  # ~/.ssh/config (this is always the repo-local POOL_SSH_CONFIG path, never that file). Silent no-op when the
  # file does not exist yet (a first-ever write has nothing to back up).
  local file="$1"
  [ -f "$file" ] && cp -p "$file" "$file.bak" 2>/dev/null
}

_remove_host_block() {   # _remove_host_block <file> <alias>
  # Same atomic-under-flock treatment as _write_host_block (2026-09-25 review, MEDIUM) -- a reader must never
  # observe a half-rewritten config, and a concurrent _write_host_block/_remove_host_block on the same file must
  # never race each other (they share the SAME "$file.lock").
  local file="$1" alias="$2"
  [ -f "$file" ] || return 0
  local dir; dir="$(dirname "$file")"
  local lock="$file.lock" tmp
  exec 8>"$lock"
  # -w 30: same reasoning as _write_host_block's own flock -w 30 above.
  flock -w 30 8 || { echo "⛔ _remove_host_block: timed out waiting for the lock on $lock" >&2; exec 8>&-; return 1; }
  tmp=$(mktemp "$dir/.host_block.XXXXXX") || { flock -u 8; exec 8>&-; return 1; }
  # GUARD (2026-09-25 review, MEDIUM regression, same fix as _write_host_block above): a failed awk must never
  # be followed by an unconditional `mv` that would swap the real config for a truncated/wrong one.
  awk -v a="Host $alias" '
    $0==a {skip=1; next}
    skip && /^Host / {skip=0}
    !skip {print}
  ' "$file" > "$tmp" && mv "$tmp" "$file" || {
    rm -f "$tmp"
    flock -u 8; exec 8>&-
    echo "⛔ _remove_host_block: failed to rewrite $file -- original content left untouched" >&2
    return 1
  }
  flock -u 8
  exec 8>&-
}

_running_runners() {   # _running_runners <ssh-alias-or-command-prefix...> -- prints a count, "" if unreachable
  # Same bracketed pgrep pattern as pool_autodispatch.sh's node_is_idle (an un-bracketed one self-matches the
  # ssh command carrying it). Used by `down`'s drain wait and its refuse-while-running guard.
  timeout 10 "$@" "pgrep -c -f '^[^ ]*/?python[0-9.]* .*-m [r]esearch\.runners' 2>/dev/null | head -1" 2>/dev/null
}

_terminate_and_delete_sg() {   # _terminate_and_delete_sg <instance-id> <sg-id> <region>
  local iid="$1" sg="$2" region="$3"
  # Returns NON-ZERO when terminate-instances fails (re-review round 4, MEDIUM): a
  # throttled/credential/API failure used to be ignored, and the caller then recorded a still-running instance as
  # TORN DOWN and removed its Host block -- a cost leak plus a false record.
  if [ -n "$iid" ]; then
    echo "[aws-pool-node] terminating $iid…"
    local out rc
    out=$(aws ec2 terminate-instances --instance-ids "$iid" --region "$region" \
      --query 'TerminatingInstances[].CurrentState.Name' --output text 2>&1); rc=$?
    echo "$out"
    if [ "$rc" -ne 0 ]; then
      echo "⛔ terminate-instances did not confirm termination of $iid (rc=$rc). NOT marking it torn down." >&2
      echo "   Retry: aws ec2 terminate-instances --instance-ids $iid --region $region" >&2
      return 1
    fi
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
  # PROPAGATE THE REAL EXIT CODE (2026-09-25 review, MEDIUM fix verification): this used to be an unconditional
  # `exit 0` regardless of _write_host_block's own return status, so a test (or any other caller) could not tell
  # a guarded failure from success through this seam at all -- only by inspecting file content, which does not
  # distinguish "refused, original untouched" from "silently wrote nothing".
  _write_host_block "$2" "$3" "$4" "$5"; exit $?
fi
if [ "${1:-}" = "--remove-host-block" ]; then
  [ "$#" -eq 3 ] || { echo "usage: $0 --remove-host-block <file> <alias>" >&2; exit 2; }
  _remove_host_block "$2" "$3"; exit $?
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

  # AUTO-TEARDOWN ON ANY FAILURE FROM HERE ON (2026-09-23 fix round; extended fix round #2 to cover the
  # no-public-IP and interrupted-mid-`up` cases too). Before this, a provision/sanity failure left the instance
  # LIVE + UNWIRED with only aws_idle_stop.sh's ~20min idle-STOP (not terminate) as a backstop -- billing
  # continued and the SG/instance leaked until someone noticed. `_up_failed` terminates + deletes the SG + marks
  # the state file torn down (so a retried `up NODE_NAME` is not blocked by "already recorded live") before
  # propagating the failure. Defined (and its state read) BEFORE the no-public-IP check below, which used to
  # `exit 1` directly and skip this teardown entirely -- an instance with no public IP still costs money and
  # still needs its SG cleaned up.
  REGION_S=$(_state_get region); REGION_S="${REGION_S:-$REGION}"; SG=$(_state_get sg)
  _up_failed() {
    local reason="$1"
    echo "⛔ aws_pool_node.sh up: $reason — auto-tearing-down $NODE_NAME (instance $IID) rather than leaving it live+unwired." >&2
    if _terminate_and_delete_sg "$IID" "$SG" "$REGION_S"; then
      { echo "# TORN DOWN $(date '+%F %T %Z') (auto, up failed: $reason)"; cat "$STATE"; } > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
    else
      echo "⛔ auto-teardown of $IID did NOT confirm; the state file stays live so idle-stop and a later down can find it." >&2
    fi
    exit 1
  }
  # INTERRUPTED MID-`up` (2026-09-23 fix round #2, LOW): Ctrl-C during the long pre-provision loop below used to
  # leave a live, unregistered instance with a persistent Host block and no cleanup -- aws_idle_stop.sh only
  # STOPs an idle instance, never terminates one, so an interrupted `up` billed indefinitely until someone
  # noticed and ran `down` by hand. Route SIGINT/SIGTERM through the SAME auto-teardown as any other failure.
  trap '_up_failed "interrupted (SIGINT/SIGTERM)"' INT TERM

  IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
        --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
  [ -n "$IP" ] && [ "$IP" != "None" ] || _up_failed "instance $IID has no public IP"
  echo "[aws-pool-node] instance=$IID ip=$IP"

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

  # 1b. A STOPPED instance is started, synced, THEN terminated (2026-09-23 fix round #2). aws_idle_stop.sh only
  #     STOPs an idle AWS pool node (never terminates), so `down` can be called against a node that is live in
  #     AWS but unreachable over ssh purely because its network interface is down while stopped -- the OLD code
  #     read that identically to "genuinely gone" and (see 2/3 below) would have terminated it unsynced. Start it
  #     back up first so the rest of this function's drain/sync verification has something real to check.
  if [ -n "$IID" ]; then
    EC2_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
        --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
    case "$EC2_STATE" in
      stopped)
        echo "[aws-pool-node] $NODE_NAME's instance $IID is STOPPED — starting it to sync + terminate cleanly…"
        aws ec2 start-instances --instance-ids "$IID" --region "$REGION_S" >/dev/null 2>&1
        START_TIMEOUT="${AWS_POOL_START_TIMEOUT_S:-180}"; START_BEGIN=$(date +%s)
        RUNNING_STATE_SEEN=0
        while :; do
          _POLL_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
              --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
          if [ "$_POLL_STATE" = "running" ]; then RUNNING_STATE_SEEN=1; break; fi
          if [ $(( $(date +%s) - START_BEGIN )) -ge "$START_TIMEOUT" ]; then
            echo "  ⛔ $NODE_NAME's instance did not reach 'running' within ${START_TIMEOUT}s of starting it (last state: ${_POLL_STATE:-unknown})." >&2
            [ "$FORCE" = 1 ] || { echo "     Refusing to terminate an instance we could not verify/sync. Re-run with --force to accept the loss." >&2; exit 1; }
            break
          fi
          sleep "${AWS_POOL_START_POLL_S:-10}"
        done
        if [ "$RUNNING_STATE_SEEN" = 1 ]; then
          # RE-READ THE PUBLIC IP AND REWRITE THE HOST BLOCK (2026-09-23 fix round #3, re-review HIGH: "the
          # stopped-instance branch does not work in practice"). EC2 assigns a NEW public IPv4 on every
          # stop/start (there is no Elastic IP anywhere in this feature), but the persistent Host block still
          # carries the IP from the last `up` (or the previous `down` that started it) -- so probing "$NODE_NAME"
          # against it always hit the STALE address and timed out after ${AWS_POOL_START_TIMEOUT_S:-180}s, `down`
          # exited 1 leaving the instance RUNNING (a cost leak `down` itself created, only re-stopped once
          # aws_idle_stop's next cycle noticed), and the refusal text ("re-run with --force to accept the loss")
          # was exactly the unsynced-termination case HIGH #4 exists to prevent. Re-read the CURRENT
          # PublicIpAddress and rewrite the Host block BEFORE the reachability probe below -- exactly like `up`
          # does for a freshly-launched instance (see _write_host_block's call further down this file).
          NEW_IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
              --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
          KEY_S=$(_state_get key)
          if [ -n "$NEW_IP" ] && [ "$NEW_IP" != "None" ] && [ -n "$KEY_S" ]; then
            _write_host_block "$SSH_CONFIG" "$NODE_NAME" "$NEW_IP" "$KEY_S"
            echo "  [aws-pool-node] $NODE_NAME's Host block updated to its current IP $NEW_IP."
          else
            echo "  ⛔ could not re-read a public IP (or the recorded key) for restarted instance $IID -- the" >&2
            echo "     Host block still carries the OLD IP; the reachability probe below will likely time out." >&2
          fi
          PROBE_TIMEOUT="${AWS_POOL_START_TIMEOUT_S:-180}"; PROBE_BEGIN=$(date +%s)
          while :; do
            if [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null && \
               timeout 8 ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME" true 2>/dev/null; then
              echo "  [aws-pool-node] $NODE_NAME is back up and reachable."
              break
            fi
            if [ $(( $(date +%s) - PROBE_BEGIN )) -ge "$PROBE_TIMEOUT" ]; then
              echo "  ⛔ $NODE_NAME did not become reachable within ${PROBE_TIMEOUT}s of restarting it." >&2
              [ "$FORCE" = 1 ] || { echo "     Refusing to terminate an instance we could not verify/sync. Re-run with --force to accept the loss." >&2; exit 1; }
              break
            fi
            sleep "${AWS_POOL_START_POLL_S:-10}"
          done
        fi
        ;;
      terminated|shutting-down)
        # Already CONFIRMED gone at AWS's side (a successful describe-instances says so) -- there is nothing left
        # to drain, sync or terminate. Mark torn down and stop.
        echo "[aws-pool-node] $NODE_NAME's instance $IID is already '$EC2_STATE' at AWS -- nothing to sync/terminate." >&2
        [ "$FORCE" = 1 ] || { echo "     Re-run with --force to mark it torn down anyway (no data can be recovered either way)." >&2; exit 1; }
        _remove_host_block "$SSH_CONFIG" "$NODE_NAME"
        { echo "# TORN DOWN $(date '+%F %T %Z') (instance already $EC2_STATE at AWS)"; cat "$STATE"; } > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
        echo "[aws-pool-node] ✓ $NODE_NAME marked torn down."
        exit 0
        ;;
      "")
        # EMPTY describe-instances is UNKNOWN, NOT gone (2026-09-23 fix round #3, re-review MEDIUM: "empty
        # describe-instances = UNKNOWN"). A transient AWS API/credential/throttle failure on a STILL-RUNNING
        # instance reads EXACTLY like this too -- the old code treated it identically to a confirmed-terminated
        # instance and, with --force, marked the state file torn down and removed the Host block WITHOUT ever
        # calling terminate-instances: a still-live instance recorded as gone, its EBS volume and SG leaking
        # indefinitely (aws_idle_stop.sh only STOPs, never terminates, so nothing ever cleans it up after that).
        # Never take the "already gone" shortcut here: fall through to the normal drain/sync/terminate path below
        # instead. Its own --force handling already governs each of ITS refusals, but step 5's
        # `_terminate_and_delete_sg` call always runs regardless -- so --force may skip a refusal along the way,
        # but the actual terminate-instances call is never skipped, and the state file is only ever marked torn
        # down AFTER that real attempt (never in place of it).
        echo "  ⛔ describe-instances returned EMPTY for $IID -- instance state UNKNOWN (a transient AWS API/credential/throttle failure on a still-running instance reads identically), not assumed gone." >&2
        ;;
    esac
  fi

  # 2. DRAIN — wait for any runner already in flight on this node to finish (2026-09-23 fix round). The root
  #    volume is DeleteOnTermination=true, so a job killed mid-run by `terminate` loses its output with no
  #    requeue; unregistering (step 1) stops NEW jobs landing but does nothing for one already running. Bounded
  #    wait (AWS_POOL_DRAIN_TIMEOUT_S, default 30 min matches the slowest routine pool job) so `down` cannot
  #    hang forever on a stuck runner -- --force (or the timeout) is the way out of that case.
  #
  #    BUGFIX (fix round #2): unreachable used to read as "0 runners" -- nothing to drain FOR -- which let an
  #    unreachable node (a real ssh/network hiccup, or an instance aws_idle_stop stopped mid-cycle) sail through
  #    the drain guard it exists to provide. Unreachable is UNKNOWN, not zero: retry until the SAME bounded
  #    timeout, then refuse (like any other undrained state) unless --force.
  if [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    DRAIN_TIMEOUT="${AWS_POOL_DRAIN_TIMEOUT_S:-1800}"
    DRAIN_START=$(date +%s)
    while :; do
      N_RUNNING=$(_running_runners ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME")
      if [ -z "${N_RUNNING:-}" ]; then
        if [ $(( $(date +%s) - DRAIN_START )) -ge "$DRAIN_TIMEOUT" ]; then
          echo "  ⛔ $NODE_NAME unreachable for ${DRAIN_TIMEOUT}s during drain — runner state UNKNOWN (not assumed zero)." >&2
          [ "$FORCE" = 1 ] || { echo "     Refusing to terminate (cannot verify nothing is still running). Re-run with --force to override." >&2; exit 1; }
          break
        fi
        echo "  [aws-pool-node] $NODE_NAME unreachable during drain (runner count UNKNOWN), retrying…"
        sleep "${AWS_POOL_DRAIN_POLL_S:-20}"
        continue
      fi
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

  # 3. PULL job_status.log FIRST (2026-09-23 fix round #2, MEDIUM lost-job observability): the crash/idle
  #    detectors (tools/workflow_check.sh) read job_status.log LIVE over ssh -- once this node is terminated
  #    (DeleteOnTermination=true), that history is gone forever, so a crash on an AWS node was never surfaced
  #    anywhere. Best-effort/non-fatal (diagnostic, not the correctness gate below) -- a missing/unreachable log
  #    must not block teardown by itself.
  if [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    mkdir -p "$ROOT/research/queue/aws_pool_node_logs"
    if timeout 30 scp -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 \
        "$NODE_NAME:~/derisk-pool/sim/job_status.log" \
        "$ROOT/research/queue/aws_pool_node_logs/${NODE_NAME}.job_status.log" 2>/dev/null; then
      echo "[aws-pool-node] pulled $NODE_NAME's job_status.log (crash/verdict history preserved before terminate)."
    else
      echo "  (no job_status.log pulled from $NODE_NAME — none written yet, or already unreachable)" >&2
    fi
  fi

  # 4. PULL RESULTS, STRICTLY, while the node (and its persistent ssh alias, untouched so far) is still
  #    reachable. BUGFIX (fix round #2, the re-review's HIGH #4): pool_sync.sh's plain/default mode ALWAYS
  #    exits 0 (by design — it is also the systemd-timer's best-effort cadence call, which must never abort on
  #    one bad node) and swallows a per-revision rsync failure with `|| continue`, so SYNC_OK above was
  #    STRUCTURALLY UNABLE to ever read 0 for the exact cases this guard exists for ("unreachable ... or an
  #    instance already STOPPED"). --strict (POOL_SYNC_STRICT=1) is pool_sync.sh's new opt-in mode that DOES
  #    fail loudly on those cases; only `down` (not the timer) turns it on.
  echo "[aws-pool-node] pulling any results $NODE_NAME already produced (strict)…"
  SYNC_OK=1
  POOL_NODES="$NODE_NAME" POOL_SYNC_STRICT=1 bash "$ROOT/tools/pool_sync.sh" || SYNC_OK=0
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
  # same unreachable-is-UNKNOWN treatment as step 2 (an unreachable re-check here does not itself block, since
  # the strict sync above already gated on reachability; this only catches a runner that started IN BETWEEN).
  if [ "$FORCE" != 1 ] && [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    N_RUNNING=$(_running_runners ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME")
    if [ -n "${N_RUNNING:-}" ] && [ "$N_RUNNING" -gt 0 ] 2>/dev/null; then
      echo "  ⛔ $N_RUNNING runner(s) still running on $NODE_NAME. Refusing to terminate. Re-run with --force to override." >&2
      exit 1
    fi
  fi

  # 5. TERMINATE + delete the SG — only after (1)-(4) all either succeeded or were force-overridden. If AWS does not
  #    confirm termination, keep the state file live and the Host block in place, and exit 1 (retry `down`).
  if ! _terminate_and_delete_sg "$IID" "$SG" "$REGION_S"; then
    echo "[aws-pool-node] $NODE_NAME NOT torn down — re-run: bash tools/aws_pool_node.sh down $NODE_NAME" >&2
    exit 1
  fi

  # 6. Remove the now-stale persistent ssh Host entry, and mark the state file torn down (never delete it —
  #    same durability intent as .aws_gpu: the record of what ran and when survives).
  _remove_host_block "$SSH_CONFIG" "$NODE_NAME"
  { echo "# TORN DOWN $(date '+%F %T %Z')"; cat "$STATE"; } > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
  echo "[aws-pool-node] ✓ $NODE_NAME torn down."
}

cmd_start() {
  # `start` -- start a STOPPED node (e.g. one tools/aws_idle_stop.sh stopped, or a `down`-in-progress node the
  # owner wants back without a fresh `up`), wait for running+ssh, and rewrite its .pool_ssh_config Host block
  # to whatever new public IP EC2 hands it (there is no Elastic IP anywhere in this AWS-pool-node feature, so
  # every stop/start gets a DIFFERENT ip -- see `down`'s own "stopped-then-reachable" handling above, which
  # this mirrors for the standalone case). 2026-09-25, incident-driven: pool1 was idle-stopped and only came
  # back after the owner started it BY HAND and edited .pool_ssh_config BY HAND (a manual step this command
  # replaces). Budget-gated (tools/aws_budget.sh check) before any spend-incurring start-instances call --
  # never gated when the instance is already running (no new spend, nothing to refuse).
  [ -f "$STATE" ] || {
    echo "⛔ $NODE_NAME: no state file $STATE -- nothing to start (never launched via aws_pool_node.sh, or the record was moved)" >&2
    exit 1
  }
  if grep -q '^# TORN DOWN' "$STATE"; then
    echo "⛔ $NODE_NAME is TORN DOWN ($STATE) -- there is no instance left to start; use 'up' to launch a fresh one." >&2
    exit 1
  fi
  IID=$(_state_get instance); REGION_S=$(_state_get region); REGION_S="${REGION_S:-$REGION}"; KEY_S=$(_state_get key)
  [ -n "$IID" ] || { echo "⛔ $STATE has no instance=" >&2; exit 1; }
  [ -n "$KEY_S" ] && [ -f "$KEY_S" ] || { echo "⛔ $NODE_NAME: no usable key= in $STATE" >&2; exit 1; }

  EC2_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
      --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
  case "$EC2_STATE" in
    running)
      echo "[aws-pool-node] $NODE_NAME's instance $IID is already running -- refreshing its Host block only (no restart, no budget check: no new spend)."
      ;;
    terminated|shutting-down)
      echo "⛔ $NODE_NAME's instance $IID is '$EC2_STATE' at AWS -- cannot start a terminated instance; 'up' launches a fresh one." >&2
      exit 1 ;;
    stopping)
      # WAIT FOR 'stopped' FIRST (2026-09-25 review, LOW): AWS refuses start-instances on an instance that is
      # still mid-'stopping' (e.g. aws_idle_stop.sh's own stop-instances call landed moments ago). The OLD code
      # fell straight through to the `*)` refusal below and gave up immediately instead of waiting out a
      # transition that, unlike 'terminated', WILL resolve on its own.
      echo "[aws-pool-node] $NODE_NAME's instance $IID is 'stopping' -- waiting for it to reach 'stopped' before starting it…"
      STOP_WAIT_TIMEOUT="${AWS_POOL_STOP_WAIT_TIMEOUT_S:-180}"; STOP_WAIT_BEGIN=$(date +%s)
      while :; do
        EC2_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
            --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
        [ "$EC2_STATE" = "stopped" ] && break
        if [ $(( $(date +%s) - STOP_WAIT_BEGIN )) -ge "$STOP_WAIT_TIMEOUT" ]; then
          echo "⛔ $NODE_NAME's instance did not finish stopping within ${STOP_WAIT_TIMEOUT}s (last state: ${EC2_STATE:-unknown}) -- cannot start it yet, try again shortly." >&2
          exit 1
        fi
        sleep "${AWS_POOL_START_POLL_S:-10}"
      done
      # Genuinely 'stopped' now -- fall through to the SAME start-instances + wait-for-running path as below.
      ;&
    stopped)
      TYPE_S=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
          --query 'Reservations[].Instances[].InstanceType' --output text 2>/dev/null)
      bash "$ROOT/tools/aws_budget.sh" check "${TYPE_S:-$TYPE}" || {
        echo "⛔ aws_pool_node.sh start: refused by tools/aws_budget.sh (daily cap) — see above" >&2; exit 1; }
      echo "[aws-pool-node] starting $NODE_NAME's instance $IID…"
      # CAPTURE RC + STDERR (2026-09-25 review, LOW): the old `>/dev/null 2>&1` discarded start-instances'
      # own failure (throttled/credential/quota error) entirely and then waited the FULL timeout with no
      # explanation of why the instance never reached 'running'. Fail fast, with the real reason.
      START_OUT=$(aws ec2 start-instances --instance-ids "$IID" --region "$REGION_S" 2>&1); START_RC=$?
      if [ "$START_RC" -ne 0 ]; then
        echo "⛔ start-instances failed for $NODE_NAME's instance $IID (rc=$START_RC):" >&2
        echo "$START_OUT" >&2
        exit 1
      fi
      START_TIMEOUT="${AWS_POOL_START_TIMEOUT_S:-180}"; START_BEGIN=$(date +%s)
      while :; do
        EC2_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
            --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
        [ "$EC2_STATE" = "running" ] && break
        if [ $(( $(date +%s) - START_BEGIN )) -ge "$START_TIMEOUT" ]; then
          echo "⛔ $NODE_NAME's instance did not reach 'running' within ${START_TIMEOUT}s of starting it (last state: ${EC2_STATE:-unknown})." >&2
          exit 1
        fi
        sleep "${AWS_POOL_START_POLL_S:-10}"
      done
      ;;
    pending)
      # WAIT FOR 'running' (2026-09-25 review, LOW): a start is already in flight (e.g. a concurrent `start`
      # call, or the owner's own console action) -- this must wait it out, not refuse a state that will
      # resolve to exactly what `start` wants on its own.
      echo "[aws-pool-node] $NODE_NAME's instance $IID is already 'pending' (a start is already in flight) -- waiting for it to reach 'running'…"
      START_TIMEOUT="${AWS_POOL_START_TIMEOUT_S:-180}"; START_BEGIN=$(date +%s)
      while :; do
        EC2_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
            --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
        [ "$EC2_STATE" = "running" ] && break
        if [ $(( $(date +%s) - START_BEGIN )) -ge "$START_TIMEOUT" ]; then
          echo "⛔ $NODE_NAME's instance did not reach 'running' within ${START_TIMEOUT}s (was already 'pending'; last state: ${EC2_STATE:-unknown})." >&2
          exit 1
        fi
        sleep "${AWS_POOL_START_POLL_S:-10}"
      done
      ;;
    *)
      echo "⛔ $NODE_NAME's instance $IID is in state '${EC2_STATE:-unknown}' -- refusing to start (only 'running'/'stopped'/'stopping'/'pending' are handled; describe-instances may be UNKNOWN/unreachable)." >&2
      exit 1 ;;
  esac

  CUR_IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
      --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
  [ -n "$CUR_IP" ] && [ "$CUR_IP" != "None" ] || { echo "⛔ $NODE_NAME: instance running but no public IP on record" >&2; exit 1; }

  # ATOMIC REWRITE, WITH A BACKUP (the build ask, verbatim): back up the config, then let _write_host_block's
  # own tmp-file-then-`mv` (same filesystem, so `mv` is atomic) replace it -- a reader never observes a
  # half-written config either way.
  _backup_ssh_config "$SSH_CONFIG"
  _write_host_block "$SSH_CONFIG" "$NODE_NAME" "$CUR_IP" "$KEY_S"
  echo "[aws-pool-node] $NODE_NAME's Host block set to $CUR_IP (backup: $SSH_CONFIG.bak)."

  echo "[aws-pool-node] waiting for ssh…"
  PROBE_TIMEOUT="${AWS_POOL_START_TIMEOUT_S:-180}"; PROBE_BEGIN=$(date +%s)
  while :; do
    if timeout 8 ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=6 "$NODE_NAME" true 2>/dev/null; then
      echo "[aws-pool-node] ✓ $NODE_NAME is running and reachable at $CUR_IP."
      exit 0
    fi
    if [ $(( $(date +%s) - PROBE_BEGIN )) -ge "$PROBE_TIMEOUT" ]; then
      echo "⛔ $NODE_NAME did not become ssh-reachable within ${PROBE_TIMEOUT}s of starting it (Host block carries its current ip, $CUR_IP -- may just need more time, or a security-group/network issue)." >&2
      exit 1
    fi
    sleep "${AWS_POOL_START_POLL_S:-10}"
  done
}

cmd_refresh() {
  # `refresh` -- CHEAP, read-mostly: if this node has a RUNNING instance whose current public ip differs from
  # (or is missing from) its .pool_ssh_config Host block, rewrite that block (atomically, with a backup);
  # otherwise a no-op. Never starts a stopped instance (that is `start`'s job, which is budget-gated and can
  # take minutes) and never calls tools/aws_budget.sh (nothing here spends money). Built so OTHER entry points
  # that discover a stale Host block for a node that is actually up (e.g. tools/pool_sync.sh, wired in
  # 2026-09-25) can self-heal in one cheap call instead of needing a human to notice and re-run `start`/`up`.
  [ -f "$STATE" ] || { echo "$NODE_NAME: no state file $STATE -- nothing to refresh"; exit 0; }
  if grep -q '^# TORN DOWN' "$STATE"; then
    echo "$NODE_NAME: torn down ($STATE) -- nothing to refresh"; exit 0
  fi
  IID=$(_state_get instance); REGION_S=$(_state_get region); REGION_S="${REGION_S:-$REGION}"; KEY_S=$(_state_get key)
  [ -n "$IID" ] || { echo "$NODE_NAME: $STATE has no instance= -- nothing to refresh"; exit 0; }

  EC2_STATE=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
      --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
  if [ "$EC2_STATE" != "running" ]; then
    echo "$NODE_NAME: instance $IID is '${EC2_STATE:-unknown}', not running -- nothing to refresh (use 'start' to bring it up)"
    exit 0
  fi
  CUR_IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION_S" \
      --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
  if [ -z "$CUR_IP" ] || [ "$CUR_IP" = "None" ]; then
    echo "⛔ $NODE_NAME: instance $IID is running but describe-instances gave no public IP -- cannot refresh" >&2
    exit 1
  fi

  RECORDED_IP=""
  if [ -f "$SSH_CONFIG" ] && grep -q "^Host $NODE_NAME\$" "$SSH_CONFIG" 2>/dev/null; then
    RECORDED_IP=$(awk -v a="Host $NODE_NAME" '
      $0==a {f=1; next}
      f && /^[[:space:]]*HostName[[:space:]]/ {print $2; exit}
      f && /^Host / {exit}
    ' "$SSH_CONFIG" 2>/dev/null)
  fi
  if [ "$RECORDED_IP" = "$CUR_IP" ]; then
    echo "$NODE_NAME: Host block already current ($CUR_IP)"; exit 0
  fi
  [ -n "$KEY_S" ] && [ -f "$KEY_S" ] || { echo "⛔ $NODE_NAME: no usable key= in $STATE -- cannot write a usable Host block" >&2; exit 1; }

  _backup_ssh_config "$SSH_CONFIG"
  _write_host_block "$SSH_CONFIG" "$NODE_NAME" "$CUR_IP" "$KEY_S"
  echo "$NODE_NAME: Host block refreshed ${RECORDED_IP:-<none>} -> $CUR_IP"
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
  up)      cmd_up ;;
  down)    cmd_down ;;
  status)  cmd_status ;;
  start)   cmd_start ;;
  refresh) cmd_refresh ;;
  *) echo "usage: aws_pool_node.sh {up|down|status|start|refresh} [node-name=pool1]" >&2; exit 2 ;;
esac
