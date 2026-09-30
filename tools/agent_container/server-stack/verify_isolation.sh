#!/usr/bin/env bash
# tools/agent_container/server-stack/verify_isolation.sh -- the owner's post-apply check of the sim-agent
# isolation, run ON THE SERVER (srv-host-0) after the firewall is installed and the stack is up, and again after
# any change to sim-agent.nft, fw.nft, Docker, or the stack fragment:
#
#   bash /home/dant123/data/sim-agent/src/tools/agent_container/server-stack/verify_isolation.sh
#
# Read-only: it lists rules/units/networks and runs connection probes INSIDE the sim-agent container
# (`docker exec`); it changes nothing. `sudo` is used only for `nft list` (reading the ruleset needs
# CAP_NET_ADMIN). Exit 0 = every check passed.
#
# Probe verdicts (from inside the container):
#   must-fail targets PASS only on a TIMEOUT (sim-agent.nft drops silently). A completed connection, or a
#   refusal (the packet reached a host that answered), FAILS.
#   must-pass targets PASS on a connection; a refusal PASSES with a note (the path is open but nothing is
#   listening, e.g. llama-swap stopped for gaming); a timeout FAILS (an allowed path is being dropped).
set -uo pipefail
CONTAINER="${SIM_AGENT_CONTAINER:-sim-agent}"
BRIDGE=br-simagent
SUBNET=172.30.30.0/24
NETWORK=apps_sim-agent-net
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
nets="$(docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$CONTAINER" 2>/dev/null | xargs)"
if [ "$nets" = "$NETWORK" ]; then pass "$CONTAINER is attached to $NETWORK only"
else fail "$CONTAINER networks: '$nets' (want exactly $NETWORK)"; fi
ports="$(docker port "$CONTAINER" 2>/dev/null)"
if [ -z "$ports" ]; then pass "no published ports"; else fail "published ports: $ports"; fi
if [ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != running ]; then
  fail "$CONTAINER is not running -- start it (cd ~/apps && docker compose up -d sim-agent), then re-run"
  echo "== $FAILS check(s) FAILED =="; exit 1
fi

# addresses that only exist at runtime: other stacks' socket proxies and containers, and this site's WAN name
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

echo "== from inside $CONTAINER (probes run in parallel, ~5 s) =="
# shellcheck disable=SC2086
docker exec -i -e "$EXEC_ENV_PATH" "$CONTAINER" /bin/bash -s -- "$(echo $MUST_FAIL)" "$MUST_PASS" <<'INNER'
fails=0
probe() { timeout "$3" bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$1" "$2" 2>/dev/null; echo $?; }
out=$(mktemp -d)
for t in $1; do ( printf '%s %s\n' "$t" "$(probe "${t%:*}" "${t##*:}" 3)" > "$out/f_${t//[:.]/_}" ) & done
for t in $2; do ( printf '%s %s\n' "$t" "$(probe "${t%:*}" "${t##*:}" 5)" > "$out/p_${t//[:.]/_}" ) & done
wait
for f in "$out"/f_*; do read -r t rc < "$f"
  case "$rc" in 124) echo "  PASS  blocked: $t (dropped)";;
                0) echo "  FAIL  NOT blocked: $t accepted a connection"; fails=$((fails+1));;
                *) echo "  FAIL  NOT dropped: $t answered with a refusal/unreachable (rc=$rc)"; fails=$((fails+1));; esac
done
for f in "$out"/p_*; do read -r t rc < "$f"
  case "$rc" in 0) echo "  PASS  reachable: $t";;
                124) echo "  FAIL  unreachable: $t (an allowed path is being dropped)"; fails=$((fails+1));;
                *) echo "  PASS  path open to $t, but nothing listening right now (rc=$rc) -- fine if that service is down";; esac
done
rm -rf "$out"
if getent hosts github.com >/dev/null; then echo "  PASS  DNS resolves github.com"; else echo "  FAIL  DNS lookup failed"; fails=$((fails+1)); fi
for url in https://github.com https://claude.ai; do
  if curl -sS -o /dev/null -m 10 "$url"; then echo "  PASS  HTTPS egress: $url"; else echo "  FAIL  no HTTPS egress to $url"; fails=$((fails+1)); fi
done
# this site's own public name -> the WAN address -> hairpin back into Caddy: must be dropped
for name in git.dant123.com; do
  if curl -sS -o /dev/null -m 6 "https://$name" 2>/dev/null; then
    echo "  FAIL  hairpin OPEN: https://$name answered (the self_wan set is empty or stale)"; fails=$((fails+1))
  else echo "  PASS  hairpin closed: https://$name"; fi
done
# /proc/net/if_inet6: addr ifindex plen scope flags name; scope 20 = link-local (harmless: sim-agent.nft drops
# every v6 packet from this bridge), 10 = host. Anything else on a non-lo interface is a routable v6 address.
v6="$(awk '$6 != "lo" && $4 != "20" && $4 != "10"' /proc/net/if_inet6 2>/dev/null)"
if [ -z "$v6" ]; then echo "  PASS  no routable IPv6 address"
else echo "  FAIL  the container has routable IPv6 addresses"; fails=$((fails+1)); fi
if [ -S /var/run/docker.sock ] || [ -S /run/docker.sock ]; then echo "  FAIL  a docker socket is mounted"; fails=$((fails+1))
else echo "  PASS  no docker socket inside"; fi
if [ "$(id -u)" = 61000 ]; then echo "  PASS  runs as uid 61000"; else echo "  FAIL  runs as uid $(id -u)"; fails=$((fails+1)); fi
if grep -q '^CapEff:[[:space:]]*0*$' /proc/self/status; then echo "  PASS  no effective capabilities"
else echo "  FAIL  effective capabilities: $(grep ^CapEff /proc/self/status)"; fails=$((fails+1)); fi
if grep -q '^NoNewPrivs:[[:space:]]*1' /proc/self/status; then echo "  PASS  no_new_privs set"; else echo "  FAIL  no_new_privs not set"; fails=$((fails+1)); fi
if ( : > /opt/.rw_test ) 2>/dev/null; then rm -f /opt/.rw_test; echo "  FAIL  root filesystem is writable"; fails=$((fails+1))
else echo "  PASS  root filesystem is read-only"; fi
exit "$fails"
INNER
inner=$?
FAILS=$((FAILS + inner))

echo "== drop counters (they should have grown during the probes) =="
sudo nft list table inet sim_agent 2>/dev/null | grep -E 'counter packets [1-9]' | sed 's/^[[:space:]]*/  /' || true

if [ "$FAILS" -eq 0 ]; then echo "== ALL CHECKS PASSED =="; else echo "== $FAILS check(s) FAILED =="; fi
[ "$FAILS" -eq 0 ]
