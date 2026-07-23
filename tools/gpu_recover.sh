#!/usr/bin/env bash
# gpu_recover.sh — attempt a NO-REBOOT recovery of the NVIDIA GPU (RTX 3090) after a crash.
#
# IMPORTANT: this ONLY helps for a driver/process glitch (GPU core still alive). For a HUNG CORE
# (kernel log shows "_scrubWaitAndSave: Timed out" / "API_GPU_ATTACHED_SANITY_CHECK failed" — the 2026-07-22 crash),
# a module reload CANNOT revive a wedged core and this script REFUSES; reboot instead (lmtrain-resume.service auto-resumes
# the training on boot). See docs/GPU_CRASH_RECOVERY.md.
set -uo pipefail

echo "=== gpu_recover: assessing GPU state ==="
if nvidia-smi -L >/dev/null 2>&1; then
  echo "nvidia-smi RESPONDS — the GPU is not off the bus. If a training just failed, likely a wedged process, not a core hang."
  echo "Recovery may not even be needed. Current state:"; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
  RESPONSIVE=1
else
  echo "nvidia-smi does NOT respond (GPU off the bus / hung)."
  RESPONSIVE=0
fi

# REFUSE if the kernel log shows a hung core (reboot-only condition).
if sudo dmesg 2>/dev/null | tail -200 | grep -qiE "API_GPU_ATTACHED_SANITY_CHECK failed|_scrubWaitAndSave: Timed out|has fallen off the bus"; then
  echo ""
  echo "!!! HUNG CORE detected in the kernel log (sanity-check failed / scrub timeout / fell off the bus)."
  echo "!!! A module reload or PCIe rescan CANNOT recover this. REBOOT is required:"
  echo "      sudo reboot     # lmtrain-resume.service auto-resumes the training on boot"
  echo "Refusing to attempt a no-reboot recovery (it would likely hang). Exiting."
  exit 2
fi

echo ""
read -r -p "Attempt no-reboot recovery (stop lactd, free /dev/nvidia*, reload nvidia modules, restart lactd)? [y/N] " ans
[[ "${ans:-N}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

echo "=== 1. stop lactd (releases its /dev/nvidia* handles) ==="
sudo systemctl stop lactd 2>/dev/null; sleep 1

echo "=== 2. kill any process holding /dev/nvidia* ==="
PIDS=$(sudo lsof -t /dev/nvidia* 2>/dev/null | sort -u)
[ -n "$PIDS" ] && { echo "killing: $PIDS"; sudo kill "$PIDS" 2>/dev/null; sleep 3; sudo kill -9 $PIDS 2>/dev/null; } || echo "(none)"

echo "=== 3. unload nvidia kernel modules ==="
for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem nvidia; do sudo modprobe -r "$m" 2>/dev/null; done
lsmod | grep -q "^nvidia" && { echo "!!! nvidia module still loaded (a process still holds it). Reboot needed."; sudo systemctl start lactd; exit 3; }

echo "=== 4. reload nvidia + restart lactd ==="
sudo modprobe nvidia && sudo modprobe nvidia_uvm && sudo modprobe nvidia_modeset
sudo systemctl start lactd

echo "=== 5. verify ==="
if nvidia-smi -L >/dev/null 2>&1; then
  echo "RECOVERED:"; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
  echo "Now resume training:  cd $(cd "$(dirname "$0")/.." && pwd) && .venv/bin/python3 -m research.runners.lm_train_run resume --root bridges/lmtrain/run3"
else
  echo "STILL DOWN after reload — the core is wedged. REBOOT: sudo reboot (lmtrain-resume.service auto-resumes)."
  exit 4
fi
