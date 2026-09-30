#!/bin/bash
# tools/agent_container/server-stack/probe_from_pristine.sh -- run the network-isolation probes
# (bin/net_probe.sh, baked into the image) from a THROWAWAY, PRISTINE container on the sim-agent network. Runs ON
# THE SERVER as the owner, called by update.sh (every weekly cycle) and verify_isolation.sh:
#
#   bash .../probe_from_pristine.sh [--image sim-agent:local] [--network apps_sim-agent-net] -- <net_probe.sh args>
#
# WHY NOT `docker exec` INTO sim-agent (2026-09-30 security review, HIGH #1): that container's home is
# model-writable (planted binaries, rc files) and its /tmp is shared with whatever the model left running, so a
# verdict computed there is the model's to choose. This container starts from the image alone: no bind mount, no
# volume, read-only root, private tmpfs, every capability dropped, no-new-privileges, the image's non-root uid,
# bounded memory/pids, and the DNS the stack fragment pins. It sits on the same bridge and subnet, so
# sim-agent.nft treats its packets exactly like the agent's. The host-side nft drop counters
# (verify_isolation.sh) stay the authoritative signal; this is the vantage point nothing in the agent container
# can influence.
set -uo pipefail
IMAGE="${SIM_AGENT_IMAGE:-sim-agent:local}"
NETWORK="${SIM_AGENT_NETWORK:-apps_sim-agent-net}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --network) NETWORK="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "probe_from_pristine: unknown argument '$1' (net_probe.sh arguments go after --)" >&2; exit 2 ;;
  esac
done
exec docker run --rm --pull never --name "sim-agent-probe-$$" \
  --network "$NETWORK" --dns 192.168.0.1 --dns 1.1.1.1 \
  --read-only --tmpfs /tmp:size=16m,mode=1777,noexec,nosuid,nodev \
  --cap-drop ALL --security-opt no-new-privileges:true --user 61000:61000 \
  --memory 256m --pids-limit 256 --cpus 1 \
  --label sim-agent.role=isolation-probe \
  -e PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --entrypoint /bin/bash "$IMAGE" /usr/local/lib/agent-container/bin/net_probe.sh "$@"
