#!/usr/bin/env bash
# install_aws_guard_timer.sh — install + enable the systemd --user timer that runs the AWS budget guard
# (tools/aws_budget.sh enforce, THEN tools/aws_idle_stop.sh — see aws-guard.service for why that order)
# every 10 minutes. Owner-approved 2026-09-23:
# on-demand AWS instances for 6-seed CPU batteries, cap enforced by tooling not memory. User-level, this
# project only — matches the existing convention of tools/systemd/pool-sync.{service,timer} and
# tools/systemd/gpu-queue-autofill.{service,timer}, already installed the same way on this box.
#
# Idempotent: safe to re-run (copies + `daemon-reload` + `enable --now`, all no-ops if already current).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

for f in aws-guard.service aws-guard.timer; do
  cp "$ROOT/tools/systemd/$f" "$UNIT_DIR/$f"
  echo "[install_aws_guard_timer] installed $UNIT_DIR/$f"
done

systemctl --user daemon-reload
systemctl --user enable --now aws-guard.timer

echo "[install_aws_guard_timer] done. Timer status:"
systemctl --user list-timers | grep -i aws-guard || echo "  (not yet listed — check 'systemctl --user status aws-guard.timer')"
