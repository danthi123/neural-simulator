#!/usr/bin/env bash
# tools/agent_container/server-stack/verify_isolation.sh -- the owner's post-apply check of the sim-agent
# isolation, run ON THE SERVER (srv-host-0) after the firewall is installed and the stack is up, and again after
# any change to sim-agent.nft, fw.nft, Docker, Gitea or the stack fragment:
#
#   bash /home/dant123/data/sim-agent/src/tools/agent_container/server-stack/verify_isolation.sh
#
# Read-only: it lists rules/units/networks/datasets, reads the container's config with `docker inspect`, runs
# two fresh PATH-pinned read-only execs into the container (`id -u`, /proc/self/status), and probes the network
# from a THROWAWAY PRISTINE container (probe_from_pristine.sh); it changes nothing. `sudo` is used only for
# `nft list` (reading the ruleset needs CAP_NET_ADMIN). Exit 0 = every check passed.
#
# WHAT IS TRUSTED (2026-09-30 security review, HIGH #1): nothing computed inside the long-lived sim-agent
# container. Its home is model-writable (planted binaries, rc files) and its /tmp is shared with whatever the model
# left running, so the probes run from a container started from the image alone on the same bridge, and the
# host's own nft drop counters must grow during them (a hard check, the authoritative signal). The container's
# isolation settings come from `docker inspect`, i.e. from Docker, not from the container.
#
# Probe verdicts (bin/net_probe.sh): must-fail targets PASS only on a TIMEOUT (sim-agent.nft drops silently); a
# connection or a refusal FAILS. must-pass targets PASS on a connection, and on a refusal with a note (the path is
# open, nothing listening); a timeout FAILS.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${SIM_AGENT_CONTAINER:-sim-agent}"
IMAGE="${SIM_AGENT_IMAGE:-sim-agent:local}"
BRIDGE=br-simagent
SUBNET=172.30.30.0/24
NETWORK=apps_sim-agent-net
GITEA_API="${SIM_AGENT_GITEA_API:-https://git.dant123.com/api/v1}"
GITEA_REPO="${SIM_AGENT_GITEA_REPO:-dant123/neural-simulator}"
GITEA_DATASET="${SIM_AGENT_GITEA_DATASET:-data/appdata/gitea}"
WORKSTATION_SSH="${SIM_AGENT_WORKSTATION_SSH:-dant123@192.168.0.68:22}"
# Every `docker exec` into the container pins PATH and names its binary absolutely (2026-09-30 security review).
EXEC_ENV_PATH="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
FAILS=0
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }

echo "== host side =="
if [ "$(systemctl is-active sim-agent-fw.service 2>/dev/null)" = active ]; then pass "sim-agent-fw.service active"
else fail "sim-agent-fw.service is not active (sudo systemctl status sim-agent-fw.service)"; fi
if [ "$(systemctl is-enabled sim-agent-fw.service 2>/dev/null)" = enabled ]; then pass "sim-agent-fw.service enabled (loads at boot, before docker)"
else fail "sim-agent-fw.service is not enabled"; fi
if [ "$(systemctl is-active sim-agent-fw-selfwan.timer 2>/dev/null)" = active ]; then pass "sim-agent-fw-selfwan.timer active"
else fail "sim-agent-fw-selfwan.timer is not active"; fi
if [ -n "$(systemctl show -p OnFailure --value sim-agent-fw-selfwan.service 2>/dev/null)" ]; then
  pass "sim-agent-fw-selfwan.service alerts on failure (OnFailure=)"
else fail "sim-agent-fw-selfwan.service has no OnFailure= -- a failed WAN refresh would be silent (install the current unit)"; fi
if [ "$(systemctl is-enabled nftables.service 2>/dev/null)" = enabled ]; then
  fail "nftables.service is ENABLED -- /etc/nftables.conf flushes the whole ruleset (Docker's + this one) at boot"
else pass "nftables.service not enabled (it would flush every table)"; fi
table="$(sudo nft list table inet sim_agent 2>/dev/null)"
if [ -n "$table" ]; then pass "table inet sim_agent is loaded"; else fail "table inet sim_agent is NOT loaded"; fi
n_wan="$(sudo nft -j list set inet sim_agent self_wan 2>/dev/null | python3 -c '
import json, sys
try:
    sets = [o["set"] for o in json.load(sys.stdin)["nftables"] if "set" in o]
    print(len(sets[0].get("elem", [])) if sets else 0)
except Exception:
    print(0)')"
if [ "${n_wan:-0}" -gt 0 ]; then pass "self_wan set holds $n_wan address(es) (the hairpin is closed)"
else fail "self_wan set is EMPTY -- the container can hairpin into this box's own HTTPS services (sudo /usr/local/sbin/sim-agent-fw-selfwan.sh)"; fi
if ip -br link show "$BRIDGE" >/dev/null 2>&1; then pass "bridge interface $BRIDGE exists"
else fail "no interface $BRIDGE -- the network was created without com.docker.network.bridge.name (recreate it)"; fi
net_sub="$(docker network inspect -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}' "$NETWORK" 2>/dev/null)"
net_br="$(docker network inspect -f '{{index .Options "com.docker.network.bridge.name"}}' "$NETWORK" 2>/dev/null)"
if [ "$net_sub" = "$SUBNET" ] && [ "$net_br" = "$BRIDGE" ]; then pass "$NETWORK = $SUBNET on $BRIDGE"
else fail "$NETWORK is '$net_sub' on '$net_br' (want $SUBNET on $BRIDGE)"; fi
if [ "$(docker network inspect -f '{{.EnableIPv6}}' "$NETWORK" 2>/dev/null)" = false ]; then pass "$NETWORK has IPv6 disabled"
else fail "$NETWORK does not report EnableIPv6=false"; fi

# Gitea (2026-09-30 security review, HIGH #3 + MEDIUM): with Actions on, a workflow file on any branch the deploy key
# can push runs as a container on the shared `proxy` network through a socket proxy that allows POST+EXEC.
# Unreadable counts as a failure: this check must never pass by not running.
has_actions="$(curl -fsS -m 10 "$GITEA_API/repos/$GITEA_REPO" 2>/dev/null | python3 -c '
import json, sys
try:
    v = json.load(sys.stdin).get("has_actions")
    print({True: "true", False: "false"}.get(v, "unknown"))
except Exception:
    print("unknown")')"
case "$has_actions" in
  false) pass "Gitea Actions are OFF for $GITEA_REPO (has_actions=false)" ;;
  true)  fail "Gitea Actions are ON for $GITEA_REPO -- untick Actions (repo Settings) BEFORE the deploy key exists" ;;
  *)     fail "could not read has_actions from $GITEA_API/repos/$GITEA_REPO -- this check did not run" ;;
esac
# The deploy key can push unlimited research/draft-* data; the gitea dataset shares the `data` pool with every
# other stack, so it must carry a quota (step 3 of docs/AGENT_HARNESS_SETUP.md).
gq="$(zfs get -Hp -o value quota "$GITEA_DATASET" 2>/dev/null | head -1)"
if [[ "$gq" =~ ^[0-9]+$ ]] && [ "$gq" -gt 0 ]; then pass "$GITEA_DATASET has a quota ($((gq / 1073741824)) GiB)"
else fail "$GITEA_DATASET has NO quota -- sudo zfs set quota=<N>G $GITEA_DATASET (before the deploy key)"; fi

if [ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != running ]; then
  fail "$CONTAINER is not running -- start it (cd ~/apps && docker compose up -d sim-agent), then re-run"
  echo "== $FAILS check(s) FAILED =="; exit 1
fi
nets="$(docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$CONTAINER" 2>/dev/null | xargs)"
if [ "$nets" = "$NETWORK" ]; then pass "$CONTAINER is attached to $NETWORK only"
else fail "$CONTAINER networks: '$nets' (want exactly $NETWORK)"; fi
ports="$(docker port "$CONTAINER" 2>/dev/null)"
if [ -z "$ports" ]; then pass "no published ports"; else fail "published ports: $ports"; fi

echo "== the container's configuration (docker inspect: Docker's record, not the container's) =="
inspect_out="$(docker inspect "$CONTAINER" 2>/dev/null | python3 -c '
import json, sys
try:
    c = json.load(sys.stdin)[0]
except Exception:
    print("FAIL could not read docker inspect"); sys.exit(0)
h, mounts = c.get("HostConfig") or {}, c.get("Mounts") or []
def say(ok, good, bad):
    print(("PASS " + good) if ok else ("FAIL " + bad))
say("ALL" in (h.get("CapDrop") or []), "every capability dropped (cap_drop ALL)", "cap_drop does not include ALL")
say(not h.get("CapAdd"), "no capabilities added", "capabilities added: %s" % h.get("CapAdd"))
say(h.get("Privileged") is False, "not privileged", "the container is PRIVILEGED")
say(h.get("ReadonlyRootfs") is True, "root filesystem is read-only", "the root filesystem is writable")
say(any(o.startswith("no-new-privileges") and not o.endswith(":false") for o in (h.get("SecurityOpt") or [])),
    "no-new-privileges set", "no-new-privileges is not set")
say(h.get("PidMode", "") != "host" and h.get("IpcMode", "") != "host" and h.get("UsernsMode", "") != "host",
    "no host pid/ipc/userns namespace", "shares a host namespace")
say(sorted(h.get("Dns") or []) == sorted(["192.168.0.1", "1.1.1.1"]), "DNS pinned to 192.168.0.1 + 1.1.1.1",
    "DNS is %s (want the two pinned resolvers)" % h.get("Dns"))
dests = sorted(m.get("Destination") for m in mounts)
say(dests == ["/home/agent", "/work/sim"] and all(m.get("Type") == "bind" for m in mounts),
    "exactly two bind mounts (/work/sim, /home/agent)", "mounts are %s" % dests)
socks = [m.get("Source") for m in mounts if "docker.sock" in (m.get("Source") or "") + (m.get("Destination") or "")]
say(not socks, "no docker.sock mounted", "a docker.sock is mounted: %s" % socks)
')"
while read -r verdict msg; do
  case "$verdict" in PASS) pass "$msg" ;; FAIL) fail "$msg" ;; esac
done <<< "$inspect_out"
[ -n "$inspect_out" ] || fail "could not read docker inspect $CONTAINER"
# Two fresh execs, PATH pinned and binaries absolute: `id` reads the kernel's credentials; grep reads its OWN
# /proc/self/status. Neither runs a shell or touches the shared /tmp.
uid="$(docker exec -e "$EXEC_ENV_PATH" "$CONTAINER" /usr/bin/id -u 2>/dev/null)"
if [ "$uid" = 61000 ]; then pass "runs as uid 61000"; else fail "runs as uid '${uid:-?}' (want 61000)"; fi
status="$(docker exec -e "$EXEC_ENV_PATH" "$CONTAINER" /bin/grep -E '^(CapEff|NoNewPrivs):' /proc/self/status 2>/dev/null)"
if printf '%s\n' "$status" | grep -qE '^CapEff:[[:space:]]*0+$'; then pass "no effective capabilities in a fresh process"
else fail "effective capabilities: $(printf '%s' "$status" | grep ^CapEff)"; fi
if printf '%s\n' "$status" | grep -qE '^NoNewPrivs:[[:space:]]*1$'; then pass "no_new_privs set in a fresh process"
else fail "no_new_privs is not set"; fi

# --- network probes from a pristine container, bracketed by the host's own drop counters ------------------------
# counters -> one "<packets>\t<comment>" line per commented counter rule of table inet sim_agent
counters() {
  sudo nft -j list table inet sim_agent 2>/dev/null | python3 -c '
import json, sys
try:
    for o in json.load(sys.stdin)["nftables"]:
        r = o.get("rule")
        if r and r.get("comment"):
            for e in r.get("expr", []):
                if isinstance(e, dict) and "counter" in e:
                    print("%d\t%s" % (e["counter"]["packets"], r["comment"]))
except Exception:
    pass'
}

# addresses that only exist at runtime: other stacks' socket proxies and containers
ips_of() { docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' "$@" 2>/dev/null | xargs; }
dyn_fail=""
for ip in $(ips_of gitea-docker-socket-proxy homepage-dockerproxy); do dyn_fail+=" $ip:2375"; done
for ip in $(ips_of gitea); do dyn_fail+=" $ip:3000 $ip:22"; done
for ip in $(ips_of caddy); do dyn_fail+=" $ip:80 $ip:443"; done

MUST_FAIL="172.30.30.1:22 172.30.30.1:111 172.30.30.1:445 172.30.30.1:2049
192.168.0.10:22 192.168.0.10:111 192.168.0.10:139 192.168.0.10:445 192.168.0.10:2049 192.168.0.10:3493
192.168.0.10:1880 192.168.0.10:18443 192.168.0.10:11000 192.168.0.10:8554
192.168.0.1:80 192.168.0.1:443 192.168.0.5:53 192.168.0.5:80 192.168.0.68:80 192.168.0.68:445
172.17.0.1:22 172.20.0.1:22 8.8.8.8:53 1.1.1.1:80 $dyn_fail"
MUST_PASS="192.168.0.1:53 1.1.1.1:53 192.168.0.68:22 192.168.0.68:8080 192.168.0.10:2222"

before="$(counters)"
echo "== from a pristine probe container on $NETWORK (image $IMAGE, no mounts; probes run in parallel, ~20 s) =="
# shellcheck disable=SC2086
if bash "$HERE/probe_from_pristine.sh" --image "$IMAGE" --network "$NETWORK" -- \
     --must-fail "$(echo $MUST_FAIL)" --must-pass "$MUST_PASS" --dns-name github.com \
     --https "https://github.com https://claude.ai" --hairpin "git.dant123.com" \
     --ssh-pubkey-only "$WORKSTATION_SSH" --no-ipv6; then
  pass "every probe from the pristine container passed"
else
  fail "the pristine probe container reported failures (lines above)"
fi
after="$(counters)"

echo "== drop counters (host side, authoritative: each must have grown during the probes) =="
counter_out="$(python3 - "$before" "$after" <<'PY'
import sys
def parse(text):
    out = {}
    for line in text.splitlines():
        n, _, comment = line.partition("\t")
        if n.isdigit():
            out[comment] = int(n)
    return out
b, a = parse(sys.argv[1]), parse(sys.argv[2])
for comment in ("sim-agent -> host: nothing", "LAN, other containers, this host", "everything else",
                "hairpin to this site's own services"):
    if comment not in a:
        print("FAIL counter '%s' not found in table inet sim_agent" % comment)
    elif a[comment] > b.get(comment, 0):
        print("PASS '%s' dropped %d packet(s) during the probes" % (comment, a[comment] - b.get(comment, 0)))
    else:
        print("FAIL '%s' did not grow (%d -> %d): the probes were not dropped by this table" % (
            comment, b.get(comment, 0), a[comment]))
PY
)"
while read -r verdict msg; do
  case "$verdict" in PASS) pass "$msg" ;; FAIL) fail "$msg" ;; esac
done <<< "$counter_out"
[ -n "$counter_out" ] || fail "could not compare the drop counters"

if [ "$FAILS" -eq 0 ]; then echo "== ALL CHECKS PASSED =="; else echo "== $FAILS check(s) FAILED =="; fi
[ "$FAILS" -eq 0 ]
