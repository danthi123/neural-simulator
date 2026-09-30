#!/usr/bin/env bash
# /usr/local/sbin/sim-agent-fw-selfwan.sh -- fill sim-agent.nft's `self_wan` set with this site's own public
# IPv4 address(es), so the container's "HTTPS to the public internet" rule can never hairpin back into this
# box's own *.dant123.com services. Run by sim-agent-fw-selfwan.timer (every 5 min + at boot) and by
# sim-agent-fw.service's ExecReload (a reload recreates the table and empties the set).
# Source of truth: neural-simulator repo, tools/agent_container/server-stack/host/usr/local/sbin/.
#
# Why this is needed: every *.dant123.com name resolves to the WAN address (no split-horizon DNS here) and the
# router hairpins that traffic back to Caddy with a LAN-looking source address, which Caddy's private-range
# rules then treat as a LAN client. DDNS can change the address, so it is resolved at runtime and never written
# into any file (this repo is public).
#
# Safe by design: only PUBLIC IPv4 answers are kept; if resolution yields nothing, the current set is left
# UNTOUCHED (a DNS hiccup must not empty it) and the script exits non-zero so the timer shows the failure.
#
#   sudo /usr/local/sbin/sim-agent-fw-selfwan.sh             # refresh now
#   sudo /usr/local/sbin/sim-agent-fw-selfwan.sh --dry-run   # print what it would load, change nothing
set -uo pipefail
NAMES="${SIM_AGENT_SELFWAN_NAMES:-dant123.com}"     # names that resolve to this site's WAN address
NFT="${SIM_AGENT_SELFWAN_NFT:-/usr/sbin/nft}"        # TEST-ONLY override; NEVER set in production
GETENT="${SIM_AGENT_SELFWAN_GETENT:-getent}"         # TEST-ONLY override; NEVER set in production
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

is_public_ipv4() {
  local ip="$1" a b c d
  [[ "$ip" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
  a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}; c=${BASH_REMATCH[3]}; d=${BASH_REMATCH[4]}
  for o in "$a" "$b" "$c" "$d"; do [ "$o" -le 255 ] || return 1; done
  [ "$a" -eq 0 ] && return 1                                   # this-network
  [ "$a" -eq 10 ] && return 1                                  # RFC1918
  [ "$a" -eq 100 ] && [ "$b" -ge 64 ] && [ "$b" -le 127 ] && return 1   # CGNAT
  [ "$a" -eq 127 ] && return 1                                 # loopback
  [ "$a" -eq 169 ] && [ "$b" -eq 254 ] && return 1             # link-local
  [ "$a" -eq 172 ] && [ "$b" -ge 16 ] && [ "$b" -le 31 ] && return 1    # RFC1918
  [ "$a" -eq 192 ] && [ "$b" -eq 168 ] && return 1             # RFC1918
  [ "$a" -ge 224 ] && return 1                                 # multicast + reserved
  return 0
}

addrs=()
for name in $NAMES; do
  while read -r ip _; do
    [ -n "$ip" ] || continue
    if is_public_ipv4 "$ip"; then
      case " ${addrs[*]} " in *" $ip "*) ;; *) addrs+=("$ip") ;; esac
    else
      echo "sim-agent-fw-selfwan: ignoring non-public answer for $name" >&2
    fi
  done < <("$GETENT" ahostsv4 "$name" 2>/dev/null)
done

if [ "${#addrs[@]}" -eq 0 ]; then
  echo "sim-agent-fw-selfwan: no public IPv4 resolved for [$NAMES] -- leaving the self_wan set unchanged" >&2
  exit 1
fi

elements=$(IFS=,; echo "${addrs[*]}")
script="flush set inet sim_agent self_wan
add element inet sim_agent self_wan { ${elements} }"

if [ "$DRY_RUN" = 1 ]; then
  printf 'DRY-RUN: %s\n' "$script"
  exit 0
fi
# One nft transaction: the flush and the add apply atomically, so the set is never observed empty.
if ! printf '%s\n' "$script" | "$NFT" -f -; then
  echo "sim-agent-fw-selfwan: nft failed (is sim-agent-fw.service active?)" >&2
  exit 1
fi
echo "sim-agent-fw-selfwan: self_wan set holds ${#addrs[@]} address(es)"
