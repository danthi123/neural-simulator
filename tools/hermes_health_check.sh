#!/usr/bin/env bash
# hermes_health_check.sh — POST-REBOOT / POST-SYSTEM-UPDATE sanity gate for Hermes autonomous mode.
#
#   bash tools/hermes_health_check.sh
#
# Run this (or `~/Desktop/hermes-sim.sh check`) BEFORE trusting an overnight autonomous run right
# after a CachyOS package update or any reboot. A routine system update can silently break exactly
# the things autonomous mode depends on: a rebuilt llama.cpp that dropped the DFlash2 flag, a
# relocated/evicted model cache, a systemd unit that failed to survive, a shell-hook allowlist that
# needs the gateway to run once before it self-populates. Every check below is READ-ONLY — it makes
# no changes, starts nothing, loads no GPU work. Safe to run any time, as often as you like.
#
# Exit 0 = all green. Exit 1 = at least one ✗ line above needs attention before trusting overnight
# mode.
set -uo pipefail
REPO=/home/dant123/Projects/sim
HERMES="${HERMES_BIN:-/home/dant123/.local/bin/hermes}"
LLAMA="${QWEN_LLAMA_SERVER:-/home/dant123/.unsloth/llama.cpp/llama-server}"
TARGET_GGUF="${QWEN_TARGET_GGUF:-/home/dant123/.cache/huggingface/hub/models--sdkyuan--qwen3.8-27B-qat-q2_0-gguf/snapshots/a5885499d443cbf4a7998001508ddb3b279eeb5f/qwen38-27b-qat-q2_0.gguf}"
PORT="${QWEN_PORT:-8033}"
FAIL=0
pass() { printf '  \xe2\x9c\x93 %s\n' "$1"; }
fail() { printf '  \xe2\x9c\x97 %s\n' "$1"; FAIL=1; }
note() { printf '  \xe2\x84\xb9 %s\n' "$1"; }

hbin() { if [ -x "$HERMES" ]; then echo "$HERMES"; else command -v hermes 2>/dev/null; fi; }

echo "════ Hermes autonomous-mode health check (post-reboot / post-update gate) ════"

echo
echo "-- llama.cpp / DFlash2 (qwen_serve.sh refuses to launch without this flag) --"
if [ -x "$LLAMA" ]; then
  pass "llama-server present + executable: $LLAMA"
  if "$LLAMA" --help 2>&1 | grep -q 'draft-dflash'; then
    pass "--spec-type draft-dflash supported"
  else
    fail "llama-server has NO --spec-type draft-dflash -- a system update likely replaced the build. qwen_serve.sh will refuse to launch (by design). Rebuild the DFlash2-capable llama.cpp before trusting overnight mode."
  fi
else
  fail "llama-server missing/not executable at $LLAMA (override with QWEN_LLAMA_SERVER=...)"
fi

echo
echo "-- local Qwen target model --"
if [ -e "$TARGET_GGUF" ]; then
  pass "target GGUF present: $TARGET_GGUF"
else
  fail "target GGUF missing at $TARGET_GGUF -- qwen_serve.sh will fall back to a ~9GB -hf download (needs network + time on first use)"
fi

echo
echo "-- GPU --"
if command -v nvidia-smi >/dev/null 2>&1 && timeout 8 nvidia-smi >/dev/null 2>&1; then
  pass "nvidia-smi responsive: $(timeout 8 nvidia-smi --query-gpu=memory.used,memory.total,driver_version --format=csv,noheader 2>/dev/null)"
else
  fail "nvidia-smi not responsive -- driver not loaded post-update/reboot? try: nvidia-smi"
fi

echo
echo "-- systemd services (reboot-resilience: enabled + linger = survives reboot without login) --"
linger=$(loginctl show-user "$(id -un)" -p Linger 2>/dev/null | cut -d= -f2)
if [ "$linger" = "yes" ]; then
  pass "systemd linger enabled for $(id -un) (user services start at boot without a login)"
else
  fail "systemd linger NOT enabled -- user services may not survive reboot until you log in. fix: sudo loginctl enable-linger $(id -un)"
fi

if systemctl --user is-active qwen-supervisor.service >/dev/null 2>&1; then
  pass "qwen-supervisor.service active"
else
  fail "qwen-supervisor.service not active -- start: systemctl --user start qwen-supervisor.service"
fi
if systemctl --user is-enabled qwen-supervisor.service >/dev/null 2>&1; then
  pass "qwen-supervisor.service enabled (auto-starts on boot)"
else
  fail "qwen-supervisor.service not enabled for boot -- fix: systemctl --user enable qwen-supervisor.service"
fi

h="$(hbin)"
gw_running=0
if [ -n "$h" ]; then
  gw_status=$(timeout 15 "$h" gateway status 2>&1)
  if echo "$gw_status" | grep -qi "gateway service is running"; then
    pass "hermes gateway service running (required for the cron heartbeat to tick at all)"
    gw_running=1
  else
    fail "hermes gateway NOT running -- cron jobs will not fire. fix: bash tools/hermes_autonomous.sh on"
  fi
else
  fail "hermes binary not found on PATH / at \$HERMES_BIN -- cannot check the gateway or hooks"
fi

echo
echo "-- pre_llm_call live-state hook (must be registered AND allowlisted to actually fire) --"
if [ -n "$h" ]; then
  doctor_out=$(timeout 20 "$h" hooks doctor 2>&1)
  if echo "$doctor_out" | grep -q "pre_llm_call"; then
    block=$(printf '%s\n' "$doctor_out" | sed -n '/\[pre_llm_call\]/,/^$/p')
    if printf '%s\n' "$block" | grep -q "not allowlisted"; then
      if [ "$gw_running" -eq 1 ]; then
        fail "pre_llm_call hook registered but NOT allowlisted, and the gateway IS running -- something is wrong; restart the gateway to re-trigger registration: bash tools/hermes_autonomous.sh off && bash tools/hermes_autonomous.sh on. If it persists, check hooks_auto_accept: true is set in ~/.hermes/config.yaml."
      else
        note "pre_llm_call hook not yet allowlisted -- EXPECTED before the gateway has run a session; hooks_auto_accept auto-approves it the first time the gateway/cron actually fires. Not counted as a failure; re-run this check after 'bash tools/hermes_autonomous.sh on'."
      fi
    else
      pass "pre_llm_call hook (hook_live_state_context.py) registered and allowlisted"
    fi
  else
    fail "pre_llm_call hook not found by 'hermes hooks doctor' -- apply hermes-parity/config.hooks.snippet.yaml's hooks: block to ~/.hermes/config.yaml"
  fi
else
  fail "cannot check hooks -- hermes binary not found"
fi

echo
echo "-- git pre-commit gate + durable-state parity --"
# Retry once: the parity check's first step is `git rev-parse`, which can fail transiently right after a
# reboot (git/filesystem/services still settling) and self-correct seconds later. A health GATE must not
# false-fail on that — but a real, persistent break fails both attempts, so this masks nothing.
if bash "$REPO/tools/hermes_parity_check.sh" >/tmp/hermes_health_parity.$$ 2>&1; then
  pass "tools/hermes_parity_check.sh: OK"
elif sleep 2 && bash "$REPO/tools/hermes_parity_check.sh" >/tmp/hermes_health_parity.$$ 2>&1; then
  pass "tools/hermes_parity_check.sh: OK (passed on retry — first attempt hit a transient, e.g. a post-reboot git hiccup)"
else
  fail "tools/hermes_parity_check.sh FAILED twice -- run it directly for detail: bash tools/hermes_parity_check.sh"
fi
rm -f /tmp/hermes_health_parity.$$

echo
echo "-- Qwen endpoint (checked only if currently up -- idle is normal) --"
if pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1; then
  if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    pass "Qwen endpoint reachable at http://127.0.0.1:$PORT/v1"
  else
    fail "llama-server process present but the endpoint is NOT reachable on port $PORT -- check research/queue/qwen_server.log"
  fi
else
  note "Qwen not currently running (normal when idle -- the supervisor brings it up on demand)"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "RESULT: ALL GREEN -- safe to trust an overnight autonomous run."
else
  printf 'RESULT: \xe2\x9c\x97 one or more checks failed -- fix the \xe2\x9c\x97 lines above BEFORE trusting overnight autonomous mode.\n'
fi
exit "$FAIL"
