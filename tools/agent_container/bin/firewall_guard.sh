#!/bin/bash
# tools/agent_container/bin/firewall_guard.sh -- fail-closed start check, run by entrypoint.sh BEFORE anything
# else (so also before every `docker compose run ... smoke_test.sh`). Baked into the read-only image.
#
# On the server, the stack fragment sets SIM_AGENT_REQUIRE_FIREWALL=1, SIM_AGENT_FW_PROBE=<gateway>:22 and
# SIM_AGENT_FW_WAN_NAME=<a name that resolves to this site's own public address>. Two probes, both of which must be
# DROPPED (a timeout), or the container refuses to start:
#   1. the host's own sshd seen through the bridge gateway, which fw.nft alone WOULD accept (`iifname "br-*"
#      accept`). sim-agent.nft drops it: proves the table is loaded.
#   2. this site's WAN address on 443 (2026-09-30 security review, MEDIUM): every *.dant123.com name resolves to it
#      and the router hairpins it back to Caddy with a LAN-looking source. sim-agent.nft drops it only once the
#      runtime-filled self_wan set holds that address, which is best-effort at boot (sim-agent-fw.service's
#      ExecStartPost) and otherwise a minute later (the timer). The container auto-starts with a persistent,
#      model-writable home, so it must not run while that window is open. The address is resolved here, at
#      runtime, and never printed.
#   timeout  -> dropped                                               -> next check / start
#   connect  -> the host or the hairpin accepted it                   -> exit 1, refuse to start
#   refused / unreachable -> answered unfiltered, or the network is broken -> exit 1, refuse
#   the WAN name does not resolve to a public IPv4 (DNS not up yet)   -> exit 1, refuse (the restart policy retries)
# Why here and not only in systemd: Docker merely WANTS sim-agent-fw.service (a broken firewall edit must never
# keep every other stack down), so if the unit fails at boot the container itself must refuse to run.
# Without SIM_AGENT_REQUIRE_FIREWALL=1 (a workstation-local build/test) the check is skipped.
#
# PATH is reset below and every interpreter is absolute (2026-09-30 security review, HIGH): the environment this
# runs in must never be able to substitute its own `timeout` or `bash` and turn "connected" into "dropped".
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
if [ "${SIM_AGENT_REQUIRE_FIREWALL:-0}" != "1" ]; then
  echo "firewall_guard: not required here (SIM_AGENT_REQUIRE_FIREWALL != 1) -- skipped"
  exit 0
fi
probe="${SIM_AGENT_FW_PROBE:-}"
host="${probe%:*}"
port="${probe##*:}"
if ! [[ "$host" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || ! [[ "$port" =~ ^[0-9]{1,5}$ ]]; then
  echo "firewall_guard: REFUSING TO START -- SIM_AGENT_FW_PROBE='$probe' is not <ipv4>:<port>" >&2
  exit 1
fi
wait_s="${SIM_AGENT_FW_PROBE_TIMEOUT_S:-3}"
/usr/bin/timeout "$wait_s" /bin/bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$host" "$port" 2>/dev/null
rc=$?
case "$rc" in
  124)
    echo "firewall_guard: OK -- $host:$port was dropped (sim-agent.nft is loaded)" ;;
  0)
    echo "firewall_guard: REFUSING TO START -- $host:$port ACCEPTED a connection, so the sim-agent firewall is" \
         "not loaded on the host. Owner: sudo systemctl status sim-agent-fw.service" >&2
    exit 1 ;;
  *)
    echo "firewall_guard: REFUSING TO START -- $host:$port was refused/unreachable (rc=$rc) instead of dropped:" \
         "the firewall is not filtering this bridge, or the network is down. Owner: sudo systemctl status" \
         "sim-agent-fw.service; sudo nft list table inet sim_agent" >&2
    exit 1 ;;
esac

# --- 2. the hairpin: this site's own WAN address on 443 must be dropped too -------------------------------------
wan_name="${SIM_AGENT_FW_WAN_NAME:-}"
if ! [[ "$wan_name" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]]; then
  echo "firewall_guard: REFUSING TO START -- SIM_AGENT_FW_WAN_NAME='$wan_name' is not a host name (the stack" \
       "fragment sets it; without it the hairpin check cannot run)" >&2
  exit 1
fi
is_public_ipv4() {
  local a b
  [[ "$1" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.[0-9]{1,3}\.[0-9]{1,3}$ ]] || return 1
  a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}
  [ "$a" -eq 0 ] || [ "$a" -eq 10 ] || [ "$a" -eq 127 ] || [ "$a" -ge 224 ] && return 1
  [ "$a" -eq 100 ] && [ "$b" -ge 64 ] && [ "$b" -le 127 ] && return 1
  [ "$a" -eq 169 ] && [ "$b" -eq 254 ] && return 1
  [ "$a" -eq 172 ] && [ "$b" -ge 16 ] && [ "$b" -le 31 ] && return 1
  [ "$a" -eq 192 ] && [ "$b" -eq 168 ] && return 1
  return 0
}
wan=()
while read -r ip _; do
  if is_public_ipv4 "$ip"; then
    case " ${wan[*]} " in *" $ip "*) ;; *) wan+=("$ip") ;; esac
  fi
done < <(/usr/bin/getent ahostsv4 "$wan_name" 2>/dev/null)
if [ "${#wan[@]}" -eq 0 ]; then
  echo "firewall_guard: REFUSING TO START -- $wan_name did not resolve to a public IPv4 address (DNS not up yet?)," \
       "so the hairpin check cannot run. The restart policy retries." >&2
  exit 1
fi
for ip in "${wan[@]}"; do
  /usr/bin/timeout "$wait_s" /bin/bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$ip" 443 2>/dev/null
  rc=$?
  if [ "$rc" -ne 124 ]; then
    echo "firewall_guard: REFUSING TO START -- this site's WAN address (from $wan_name) on 443 was not dropped" \
         "(rc=$rc): the self_wan set is empty or stale, so the hairpin into this box's own services is open." \
         "Owner: sudo /usr/local/sbin/sim-agent-fw-selfwan.sh; sudo nft list set inet sim_agent self_wan" >&2
    exit 1
  fi
done
echo "firewall_guard: OK -- the WAN address(es) of $wan_name on 443 were dropped (${#wan[@]} checked; the hairpin is closed)"
exit 0
