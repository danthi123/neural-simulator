#!/usr/bin/env bash
# tools/agent_container/egress.sh -- sim-agent container EGRESS FIREWALL, homelab-server. M1 (2026-09-29
# security review): the harness must not be started before this runs. The OWNER runs this with sudo, ON THE
# SERVER, after `docker compose up` has created the `agent-net` bridge with its FIXED subnet (compose.yaml's
# `ipam.config`, set 2026-09-29 for exactly this reason -- a script here can no longer guess a Docker-assigned
# subnet that might change on a recreate). Edit the CONFIGURATION block below for this deployment, then:
#
#   sudo bash tools/agent_container/egress.sh              # apply (idempotent -- safe to re-run after any edit)
#   sudo bash tools/agent_container/egress.sh --dry-run     # print the exact commands, apply NOTHING (no root needed)
#
# WHAT WAS WRONG BEFORE (docs/AGENT_HARNESS_SETUP.md's old step (d), marked "OPTIONAL" -- this replaces it, and
# the harness setup doc now makes running this MANDATORY, not optional):
#   - every rule used `iptables -I DOCKER-USER ...` (insert at the TOP), issued ACCEPT-then-DROP -- each new
#     insert pushes the previous ones down, so issuing DROP LAST put DROP ABOVE every ACCEPT, blocking
#     everything. THE FIX: issue the catch-all DROP FIRST (so it starts at position 1), then issue every ACCEPT
#     via the SAME `-I ... 1` -- each ACCEPT then pushes DROP one step further down, so the final chain has
#     every ACCEPT above the DROP. Order matters here in a way it does not for `-A` (append).
#   - port 443 was open to ANY destination, which includes this very server's own reverse-proxied services
#     (Nextcloud, Home Assistant, Gitea, Vikunja) and the rest of the LAN, not just the internet APIs it was
#     meant for. THE FIX: exclude RFC1918 (+ loopback/link-local) destinations from the 443 rule explicitly.
#   - the gitea rule matched `-d 127.0.0.1`, which is the CONTAINER's own loopback from ITS OWN netns
#     perspective when a DOCKER-USER rule evaluates it -- it can never match gitea's real address. THE FIX: a
#     real LAN address (or the server's own address, if gitea is local to it) + real port, both configurable
#     below.
#   - nothing governed traffic from the container straight to THIS HOST's own listening services (host-network
#     things like sshd) -- DOCKER-USER only sees FORWARDED traffic, not traffic addressed to the host itself
#     arriving over the bridge interface. THE FIX: an INPUT DROP for the bridge interface, with explicit ACCEPT
#     for only the host ports the container legitimately needs (the workstation model port is reached over an
#     OUTBOUND connection FROM the container, so it is a DOCKER-USER/FORWARD concern, not an INPUT one -- INPUT
#     here is about the container talking to services bound directly on homelab-server itself, if any).
#   - nothing was idempotent or persisted across a reboot. THE FIX: every rule this script adds carries a
#     `--comment sim-agent-egress` tag so a re-run flushes exactly its own prior rules (never touching anything
#     else on this shared, multi-service box) before reapplying; persistence is the accompanying systemd unit
#     (tools/agent_container/sim-agent-egress.service, installed by the owner -- see its own header).
#
# A SECOND ordering bug (2026-09-29 RE-review, MEDIUM-1) was in THIS script's own FIRST version: the generic
# 443-ACCEPT rule was executed AFTER the RFC1918-exclusion DROP loop, which (same `-I ... 1` semantics as above)
# put the generic ACCEPT ABOVE every private-range DROP in the final chain -- silently defeating the RFC1918
# exclusion entirely. See the 443 block below for the fix and the exact reasoning.
set -uo pipefail

# ============================================================================ CONFIGURATION (edit for this box)
SUBNET="${SIM_AGENT_EGRESS_SUBNET:-172.30.30.0/24}"          # MUST match compose.yaml's agent-net ipam subnet
BRIDGE_IF="${SIM_AGENT_EGRESS_BRIDGE_IF:-br-sim-agent}"       # `docker network inspect sim-agent_agent-net
                                                               #   --format '{{.Id}}'` -> the bridge is
                                                               #   `br-<first 12 chars of that id>`; set this to
                                                               #   the real value after the network is created.
WORKSTATION_IP="${SIM_AGENT_EGRESS_WORKSTATION_IP:-192.168.0.68}"
WORKSTATION_MODEL_PORT="${SIM_AGENT_EGRESS_MODEL_PORT:-8080}"   # the local-llm server's port (tools/local_llm/llm.sh)
GITEA_HOST="${SIM_AGENT_EGRESS_GITEA_HOST:-192.168.0.10}"       # the LAN address gitea actually answers on
                                                                  # (homelab-server's own address if gitea is
                                                                  # local to this box, as it is here)
GITEA_PORT="${SIM_AGENT_EGRESS_GITEA_PORT:-2222}"               # gitea's ssh clone port
DNS_RESOLVER="${SIM_AGENT_EGRESS_DNS:-192.168.0.1}"             # this LAN's actual resolver (router/pi-hole/...)
# Host ports (on homelab-server ITSELF, arriving over the bridge) the container legitimately needs -- almost
# certainly NONE (it reaches the workstation and gitea over the LAN, not the docker host's own loopback
# services); leave empty unless a real need is identified. Space-separated "tcp:PORT" / "udp:PORT" entries.
HOST_PORTS_NEEDED="${SIM_AGENT_EGRESS_HOST_PORTS:-}"
# ============================================================================================================

IPTABLES="${SIM_AGENT_EGRESS_IPTABLES:-iptables}"   # TEST-ONLY override; NEVER set in production
COMMENT="sim-agent-egress"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY-RUN: %s\n' "$*"
  else
    "$@"
  fi
}

if [ "$DRY_RUN" = 0 ] && [ "$(id -u)" != "0" ]; then
  echo "egress.sh: must run as root (sudo) to modify iptables -- use --dry-run to preview without root." >&2
  exit 1
fi

# RFC1918 + loopback + link-local -- excluded from the 443 rule so "443 for cloud APIs" can never also mean
# "443 to this server's own reverse-proxied LAN services or anything else on the LAN".
PRIVATE_RANGES=(10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 127.0.0.0/8 169.254.0.0/16)

echo "== sim-agent egress firewall =="
echo "   subnet=$SUBNET bridge=$BRIDGE_IF workstation=$WORKSTATION_IP:$WORKSTATION_MODEL_PORT"
echo "   gitea=$GITEA_HOST:$GITEA_PORT dns=$DNS_RESOLVER  $([ "$DRY_RUN" = 1 ] && echo '(DRY RUN -- nothing applied)')"

# ---- idempotent reset: remove every rule THIS script previously added (tagged with --comment), never anything
#      else on this shared box (gitea/nextcloud/home-assistant/etc. each own their own, untouched, rules). ----
for chain in DOCKER-USER INPUT; do
  while :; do
    line=$(run "$IPTABLES" -L "$chain" --line-numbers -n 2>/dev/null | awk -v c="$COMMENT" '$0 ~ c {print $1; exit}')
    [ -n "$line" ] || break
    run "$IPTABLES" -D "$chain" "$line"
  done
done

# ---- DOCKER-USER: DROP-all FIRST (position 1), then every ACCEPT via the SAME -I 1 -- see the header comment
#      on why insertion ORDER, not append order, is what makes this end up correct. --------------------------
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -m comment --comment "$COMMENT" -j DROP

# established/related return traffic for any connection the container itself initiated (in reverse order of
# the list below, since each -I 1 pushes the previous ones -- and DROP -- further down; the RELATIVE order
# among these ACCEPTs does not matter, only that all of them land above DROP).
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -m conntrack --ctstate ESTABLISHED,RELATED \
    -m comment --comment "$COMMENT" -j ACCEPT
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -d "$WORKSTATION_IP" -p tcp --dport "$WORKSTATION_MODEL_PORT" \
    -m comment --comment "$COMMENT" -j ACCEPT
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -d "$WORKSTATION_IP" -p tcp --dport 22 \
    -m comment --comment "$COMMENT" -j ACCEPT
# CTORIGDST, NOT -d (2026-09-29 re-review, M4): if gitea is reached through any port-forward/DNAT on this
# server (e.g. it is itself a container whose published port maps to a different internal address), the
# packet's destination has ALREADY been rewritten by the time DOCKER-USER evaluates it -- DOCKER-USER sees the
# POST-DNAT destination, never what the container actually dialed. A plain `-d "$GITEA_HOST"` rule would then
# match nothing (the real destination it sees is gitea's rewritten internal address, which can also change
# across a recreate), silently leaving this ACCEPT dead and gitea access dependent on some other rule. `-m
# conntrack --ctorigdst/--ctorigdstport` matches the connection's ORIGINAL destination -- what the client
# dialed -- regardless of any NAT applied afterward, so this keeps matching "gitea, as the owner configured it"
# even if its actual on-server address moves.
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -p tcp -m conntrack --ctorigdst "$GITEA_HOST" --ctorigdstport "$GITEA_PORT" \
    -m comment --comment "$COMMENT" -j ACCEPT
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -d "$DNS_RESOLVER" -p udp --dport 53 \
    -m comment --comment "$COMMENT" -j ACCEPT
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -d "$DNS_RESOLVER" -p tcp --dport 53 \
    -m comment --comment "$COMMENT" -j ACCEPT
# 443 ONLY to non-RFC1918 destinations (cloud APIs, the Claude Code installer/updater). iptables has no
# single-rule "not in this list of ranges" -- the equivalent is one DROP rule per excluded range, each of which
# must land ABOVE the generic 443 ACCEPT in the FINAL chain so it is evaluated first.
#
# ORDERING BUG FIXED (2026-09-29 re-review, MEDIUM-1): this used to EXECUTE the generic ACCEPT LAST (after the
# private-range DROP loop) -- but `-I ... 1` means the LAST rule EXECUTED ends up FIRST in the chain, so the
# generic ACCEPT landed ABOVE every private-range DROP, matching (and thus silently accepting) 443 traffic to
# this very server's own reverse-proxied LAN services before any DROP rule ever got a chance to fire (confirmed
# with --dry-run: the printed EXECUTION order does not by itself show this -- it is the -I semantics applied to
# that order that put ACCEPT on top). THE FIX: execute the generic ACCEPT FIRST, then the private-range DROPs
# -- each DROP's `-I 1` now pushes the already-inserted ACCEPT further down, so the final chain (top to bottom)
# reads private-range DROPs, then the generic ACCEPT, exactly as intended.
run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -p tcp --dport 443 \
    -m comment --comment "$COMMENT" -j ACCEPT
for r in "${PRIVATE_RANGES[@]}"; do
  run "$IPTABLES" -I DOCKER-USER 1 -s "$SUBNET" -d "$r" -p tcp --dport 443 \
      -m comment --comment "$COMMENT" -j DROP
done

# ---- INPUT: block the container from reaching homelab-server's OWN listening services over the bridge,
#      except any host port explicitly declared needed above (almost certainly none). ------------------------
for entry in $HOST_PORTS_NEEDED; do
  proto="${entry%%:*}"; port="${entry##*:}"
  run "$IPTABLES" -I INPUT 1 -i "$BRIDGE_IF" -p "$proto" --dport "$port" -m comment --comment "$COMMENT" -j ACCEPT
done
run "$IPTABLES" -I INPUT 1 -i "$BRIDGE_IF" -m comment --comment "$COMMENT" -j DROP

echo "== done. Persist across reboot: install+enable tools/agent_container/sim-agent-egress.service =="
