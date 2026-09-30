#!/bin/bash
# tools/agent_container/bin/net_probe.sh -- the sim-agent network-isolation probes, baked into the read-only image
# and run ONLY from a throwaway PRISTINE container (server-stack/probe_from_pristine.sh: `docker run --rm` of the
# image on apps_sim-agent-net, no bind mounts, read-only root, no capabilities). Never run inside the long-lived
# sim-agent container: a model there can plant binaries and rc files in its writable home and race files in its
# /tmp, so a verdict computed there is the model's to choose (2026-09-30 security review, HIGH #1). The pristine
# container sits on the same bridge and subnet, so sim-agent.nft treats its packets exactly like the agent's, and
# nothing a rogue process in the agent container does can reach into it.
#
#   net_probe.sh [--must-fail "<ip:port> ..."] [--must-pass "<ip:port> ..."] [--dns-name <name>]
#                [--https "<url> ..."] [--hairpin "<name> ..."] [--ssh-pubkey-only "<user>@<ip>:<port> ..."]
#                [--no-ipv6]
#
# Verdicts:
#   must-fail   PASS only on a TIMEOUT (sim-agent.nft drops silently). A connection, or a refusal (the packet
#               reached something that answered), FAILS.
#   must-pass   PASS on a connection; a refusal PASSES with a note (the path is open, nothing is listening, e.g.
#               llama-swap stopped for gaming); a timeout FAILS (an allowed path is being dropped).
#   hairpin     `https://<name>` must TIME OUT (the WAN address is in self_wan). An answer, a refusal or a name that
#               does not resolve FAILS (the last one tested nothing).
#   ssh-pubkey-only   the sshd at <ip>:<port> must offer no password/keyboard-interactive method to <user> (read
#               from the server's own "Permission denied (<methods>)" reply to a `none` auth request; no key is
#               ever offered). No reply at all FAILS: the check did not run.
#   no-ipv6     this container has no routable IPv6 address.
# Exit 0 = every check passed, 1 = at least one failed. Every binary is named absolutely; PATH is reset.
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
TIMEOUT=/usr/bin/timeout
CURL=/usr/bin/curl
SSH=/usr/bin/ssh
GETENT=/usr/bin/getent

MUST_FAIL="" MUST_PASS="" DNS_NAME="" HTTPS_URLS="" HAIRPIN="" SSH_PUBKEY="" NO_IPV6=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --must-fail) MUST_FAIL="$2"; shift 2 ;;
    --must-pass) MUST_PASS="$2"; shift 2 ;;
    --dns-name) DNS_NAME="$2"; shift 2 ;;
    --https) HTTPS_URLS="$2"; shift 2 ;;
    --hairpin) HAIRPIN="$2"; shift 2 ;;
    --ssh-pubkey-only) SSH_PUBKEY="$2"; shift 2 ;;
    --no-ipv6) NO_IPV6=1; shift ;;
    *) echo "net_probe: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

FAILS=0
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }

valid_target() { [[ "$1" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}:[0-9]{1,5}$ ]]; }
# tcp_probe <ip> <port> <seconds> -> prints 0 connected, 124 timed out, anything else refused/unreachable
tcp_probe() { "$TIMEOUT" "$3" /bin/bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$1" "$2" 2>/dev/null; echo $?; }

out="$(/usr/bin/mktemp -d)"
n=0
for t in $MUST_FAIL; do
  valid_target "$t" || { fail "malformed must-fail target '$t'"; continue; }
  n=$((n + 1)); ( printf '%s %s\n' "$t" "$(tcp_probe "${t%:*}" "${t##*:}" 3)" > "$out/f_$n" ) &
done
for t in $MUST_PASS; do
  valid_target "$t" || { fail "malformed must-pass target '$t'"; continue; }
  n=$((n + 1)); ( printf '%s %s\n' "$t" "$(tcp_probe "${t%:*}" "${t##*:}" 5)" > "$out/p_$n" ) &
done
wait
for f in "$out"/f_*; do
  [ -e "$f" ] || continue
  read -r t rc < "$f"
  case "$rc" in
    124) pass "blocked: $t (dropped)" ;;
    0)   fail "NOT blocked: $t accepted a connection" ;;
    *)   fail "NOT dropped: $t answered with a refusal/unreachable (rc=$rc)" ;;
  esac
done
for f in "$out"/p_*; do
  [ -e "$f" ] || continue
  read -r t rc < "$f"
  case "$rc" in
    0)   pass "reachable: $t" ;;
    124) fail "unreachable: $t (an allowed path is being dropped)" ;;
    *)   pass "path open to $t, but nothing listening right now (rc=$rc) -- fine if that service is down" ;;
  esac
done
/bin/rm -rf "$out"

if [ -n "$DNS_NAME" ]; then
  if "$GETENT" hosts "$DNS_NAME" >/dev/null 2>&1; then pass "DNS resolves $DNS_NAME"
  else fail "DNS lookup of $DNS_NAME failed"; fi
fi

for url in $HTTPS_URLS; do
  if "$CURL" -sS -o /dev/null -m 10 "$url" 2>/dev/null; then pass "HTTPS egress: $url"
  else fail "no HTTPS egress to $url"; fi
done

for name in $HAIRPIN; do
  "$CURL" -sS -o /dev/null -m 6 "https://$name" 2>/dev/null
  rc=$?
  case "$rc" in
    28) pass "hairpin closed: https://$name (timed out)" ;;
    0)  fail "hairpin OPEN: https://$name answered (the self_wan set is empty or stale)" ;;
    6)  fail "hairpin not tested: $name did not resolve" ;;
    *)  fail "hairpin not dropped: https://$name failed fast (curl rc=$rc) instead of timing out" ;;
  esac
done

for spec in $SSH_PUBKEY; do
  if ! [[ "$spec" =~ ^([a-z_][a-z0-9_-]*)@([0-9]{1,3}(\.[0-9]{1,3}){3}):([0-9]{1,5})$ ]]; then
    fail "malformed ssh target '$spec' (want user@ip:port)"; continue
  fi
  user="${BASH_REMATCH[1]}" host="${BASH_REMATCH[2]}" port="${BASH_REMATCH[4]}"
  reply="$("$TIMEOUT" 20 "$SSH" -F /dev/null -o BatchMode=yes -o PreferredAuthentications=none \
           -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=8 -o LogLevel=ERROR -p "$port" "$user@$host" true \
           2>&1 </dev/null)"
  methods="$(printf '%s\n' "$reply" | sed -n 's/.*Permission denied (\([^)]*\)).*/\1/p' | tail -1)"
  if [ -z "$methods" ]; then
    fail "sshd $host:$port did not state its auth methods (unreachable or no reply) -- the check did not run"
  elif [[ ",$methods," == *",password,"* || ",$methods," == *",keyboard-interactive,"* ]]; then
    fail "sshd $host:$port offers '$methods' to $user: a password login is possible (PasswordAuthentication no)"
  else
    pass "sshd $host:$port offers only '$methods' to $user"
  fi
done

if [ "$NO_IPV6" = 1 ]; then
  # /proc/net/if_inet6: addr ifindex plen scope flags name; scope 20 = link-local, 10 = host
  v6="$(/usr/bin/awk '$6 != "lo" && $4 != "20" && $4 != "10"' /proc/net/if_inet6 2>/dev/null)"
  if [ -z "$v6" ]; then pass "no routable IPv6 address"; else fail "routable IPv6 address(es) on this network"; fi
fi

echo "net_probe: $FAILS failure(s)"
[ "$FAILS" -eq 0 ]
