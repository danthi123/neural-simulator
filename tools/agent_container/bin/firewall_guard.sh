#!/usr/bin/env bash
# tools/agent_container/bin/firewall_guard.sh -- fail-closed start check, run by entrypoint.sh BEFORE anything
# else (so also before every `docker compose run ... smoke_test.sh`). Baked into the read-only image.
#
# On the server, the stack fragment sets SIM_AGENT_REQUIRE_FIREWALL=1 and SIM_AGENT_FW_PROBE=<gateway>:22 -- the
# host's own sshd seen through the bridge gateway, which fw.nft alone WOULD accept (`iifname "br-*" accept`).
# sim-agent.nft drops it silently, so the ONLY acceptable outcome is a TIMEOUT:
#   timeout  -> dropped: sim-agent.nft is loaded                       -> exit 0, start
#   connect  -> the host accepted it: sim-agent.nft is NOT loaded      -> exit 1, refuse to start
#   refused / unreachable -> the packet was answered unfiltered, or the network is broken -> exit 1, refuse
# Why here and not only in systemd: Docker merely WANTS sim-agent-fw.service (a broken firewall edit must never
# keep every other stack down), so if the unit fails at boot the container itself must refuse to run.
# Without SIM_AGENT_REQUIRE_FIREWALL=1 (a workstation-local build/test) the check is skipped.
set -uo pipefail
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
timeout "$wait_s" bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$host" "$port" 2>/dev/null
rc=$?
case "$rc" in
  124)
    echo "firewall_guard: OK -- $host:$port was dropped (sim-agent.nft is loaded)"
    exit 0 ;;
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
