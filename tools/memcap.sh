#!/usr/bin/env bash
# memcap.sh — run a command under a HARD, kernel-enforced cgroup memory cap, so a ballooning job
# dies in ITS OWN cgroup instead of triggering a GLOBAL OOM that takes down the Claude session or
# the live training.
#
# WHY (2026-09-18, the incident this exists to prevent): an UNCAPPED flip no-regression battery
# spawned a full-brain build that ballooned to ~28 GB RSS. On a 46 GB box, alongside the running
# decisive training (~9 GB) + the session + desktop, that tripped the global OOM-killer, which
# killed the battery AND the Claude session (oom_score_adj=200 => highly killable). The training
# survived and completed, but ~7 h of oversight were lost. Advisory swap-monitoring did NOT prevent
# it — only a hard cap does.
#
# Usage:
#   tools/memcap.sh <max_gb> [--swap <gb>] -- <command...>
# Examples:
#   tools/memcap.sh 20 -- .venv/bin/python -m research.runners.onebrain_regression_battery --flag X
#   tools/memcap.sh 12 --swap 0 -- .venv/bin/python -u -m research.runners._some_heavy_derisk
#
# Behavior: the command runs in a transient systemd --user scope with MemoryMax=<max_gb>G and
# MemorySwapMax=<swap>G (default 0 -> no swap, so it can't thrash swap either). If it exceeds the
# cap, the KERNEL OOM-kills THIS scope only. MemoryHigh is set to 90% of the cap to throttle
# (reclaim pressure) before the hard kill, giving a softer landing. Exit code is the command's.
set -uo pipefail

MAX_GB="${1:?usage: tools/memcap.sh <max_gb> [--swap <gb>] -- <command...>}"; shift
SWAP_GB=0
if [ "${1:-}" = "--swap" ]; then SWAP_GB="${2:?--swap needs a value}"; shift 2; fi
[ "${1:-}" = "--" ] && shift
[ -n "${1:-}" ] || { echo "memcap: no command given" >&2; exit 2; }

# Daemons started outside the login session (the gpu_queue dispatcher runs from a boot-time unit) inherit no
# XDG_RUNTIME_DIR, so `systemd-run --user` cannot find the user bus and every memcapped GPU-queue job died
# instantly with rc=3 (2026-09-23: all 6 queued battery seeds failed in the same second). The user manager's
# bus lives at the standard path whenever the user is lingering/logged in — point at it.
if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
# Fall back to running UNCAPPED with a loud warning only if systemd --user is unavailable, rather
# than silently dropping the cap (a silent no-cap is exactly the failure mode we're closing).
if ! systemctl --user is-system-running >/dev/null 2>&1 && ! systemd-run --user --scope --quiet true >/dev/null 2>&1; then
  echo "[memcap] ⛔ systemd --user unavailable — cannot enforce a cap. REFUSING to run uncapped." >&2
  echo "[memcap]    (run the job yourself only if you accept the OOM risk; this guard exists on purpose)" >&2
  exit 3
fi

HIGH_GB=$(( MAX_GB * 9 / 10 )); [ "$HIGH_GB" -lt 1 ] && HIGH_GB=1
echo "[memcap] MemoryMax=${MAX_GB}G MemoryHigh=${HIGH_GB}G MemorySwapMax=${SWAP_GB}G :: $*" >&2
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${MAX_GB}G" -p MemoryHigh="${HIGH_GB}G" -p MemorySwapMax="${SWAP_GB}G" \
  -- "$@"
