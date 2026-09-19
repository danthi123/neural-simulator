#!/usr/bin/env bash
# PreToolUse(Bash) guard — BLOCK launching a known memory-heavy full-brain runner WITHOUT a memory
# cap. This is the ENFORCED (not remembered) half of the 2026-09-18 OOM fix: an uncapped
# onebrain_regression_battery full-brain build ballooned to ~28GB and the global OOM-killer took the
# Claude session with it. Heavy full-brain runners must go through tools/memcap.sh (kernel cgroup cap).
#
# Reads the PreToolUse JSON on stdin; exit 2 blocks the call and feeds stderr back to Claude.
# Deliberately NARROW: only the runners that build the whole integrated brain / many organs, so it
# never blocks ordinary commands. A launch that already contains "memcap.sh" is allowed through.
set -uo pipefail
cmd=$(jq -r '.tool_input.command // ""' 2>/dev/null || echo "")
[ -z "$cmd" ] && exit 0
# Runners that build the full integrated brain or multiple organs (RAM-heavy). Extend as needed.
HEAVY='onebrain_regression_battery|_onebrain_11organ_pool_flip_regression|webapp\.server|brain_chat|_rank2_integrated_loop|_integrated_.*_battery|onebrain_.*production.*--(grow|battery)'
if echo "$cmd" | grep -qE "$HEAVY"; then
  if ! echo "$cmd" | grep -q 'memcap\.sh'; then
    echo "⛔ memcap_guard: this launches a memory-heavy full-brain runner WITHOUT a cap." >&2
    echo "   Wrap it:  tools/memcap.sh <gb> -- <your command>   (gate first with tools/mem_ok.sh <need_gb>)." >&2
    echo "   Why: 2026-09-18 an uncapped onebrain battery ballooned to ~28GB -> global OOM killed the session." >&2
    echo "   If this is genuinely light or must run uncapped, note why and re-run with an explicit small cap." >&2
    exit 2
  fi
fi
exit 0
