#!/usr/bin/env bash
# PreToolUse(Bash) guard — BLOCK a python LAUNCH of a known memory-heavy full-brain runner WITHOUT a memory cap.
# This is the ENFORCED half of the 2026-09-18 OOM fix: an uncapped onebrain_regression_battery full-brain build
# ballooned to ~28GB and the global OOM-killer took the Claude session with it. Heavy full-brain runners must go
# through tools/memcap.sh (kernel cgroup cap).
#
# Reads the PreToolUse JSON on stdin; exit 2 blocks the call and feeds stderr back to Claude.
# NARROW + LAUNCH-SPECIFIC: only trips when python actually EXECUTES a heavy runner (a `-m research.runners.X`
# module, a `webapp.server` / brain_chat build, or the runner's .py) — NOT when a runner name merely appears as a
# grep/sed/cat/git argument (that false-positive is why v1 blocked a read-only inspection). A launch that already
# contains "memcap.sh" is allowed through.
set -uo pipefail
cmd=$(jq -r '.tool_input.command // ""' 2>/dev/null || echo "")
[ -z "$cmd" ] && exit 0

# Remote execution (ssh to a cloud instance) runs on the REMOTE box, not this one -> no local OOM -> exempt.
echo "$cmd" | grep -qE '\bssh\b[^|;&]*(ubuntu|ec2-user|root)@' && exit 0

# Must be an actual python execution...
echo "$cmd" | grep -qE '(^|[^a-zA-Z_])(python[0-9.]*|\.venv/bin/python)([[:space:]]|$)' || exit 0

# ...of a heavy full-brain runner in a MODULE-LAUNCH form (`-m research.runners.X` / `-m webapp.server` /
# a brain_chat( call in python -c). Deliberately NOT the bare `X.py` path or a bare name, so grepping/reading
# the runner FILE never trips this (the false-positive that kept blocking read-only inspection).
HEAVY_LAUNCH='(-m[[:space:]]+research\.runners\.(onebrain_regression_battery|_onebrain_11organ_pool_flip_regression|load_bearing_fraction|_rank2_integrated_loop[a-z_]*)|-m[[:space:]]+webapp\.server|brain_chat[[:space:]]*\()'
if echo "$cmd" | grep -qE "$HEAVY_LAUNCH"; then
  if ! echo "$cmd" | grep -q 'memcap\.sh'; then
    echo "⛔ memcap_guard: this LAUNCHES a memory-heavy full-brain runner WITHOUT a cap." >&2
    echo "   Wrap it:  tools/memcap.sh <gb> -- <your command>   (gate first with tools/mem_ok.sh <need_gb>)." >&2
    echo "   Why: 2026-09-18 an uncapped onebrain battery ballooned to ~28GB -> global OOM killed the session." >&2
    exit 2
  fi
fi
exit 0
