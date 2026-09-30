#!/usr/bin/env bash
# tools/pool_nodes.sh — sourced helper: the single source of truth for the DEFAULT mini-PC pool node
# list (the Python half is tools/pool_nodes.py; the data both read is tools/pool_nodes.conf).
#
#   source "$ROOT/tools/pool_nodes.sh"       # sets $POOL_NODES_DEFAULT
#   NODES="${POOL_NODES:-$POOL_NODES_DEFAULT}"
#
# Precedence is UNCHANGED and stays the caller's own job: an explicit POOL_NODES env var or explicit CLI
# node arguments must still win over $POOL_NODES_DEFAULT, and research/queue/.pool_extra_nodes (AWS/extra
# nodes) is still appended by each caller exactly where it already was -- this file only replaces the
# hardcoded "pool40 pool41 pool42" literal, nothing about how overrides or extras are layered on top.
POOL_NODES_CONF_FILE="${POOL_NODES_CONF_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pool_nodes.conf}"

pool_nodes_default() {
  # Re-read fresh every call (this file is sourced by short-lived scripts, not a daemon, so there is
  # nothing to cache and nothing that can go stale between calls).
  [ -f "$POOL_NODES_CONF_FILE" ] || return 0
  grep -vE '^[[:space:]]*(#|$)' "$POOL_NODES_CONF_FILE" 2>/dev/null | tr -s '[:space:]' ' ' | sed -e 's/^ *//' -e 's/ *$//'
}

POOL_NODES_DEFAULT="$(pool_nodes_default)"
