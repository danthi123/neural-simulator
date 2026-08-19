#!/bin/bash
# One-command control for the run3 LM training + its GPU-crash/auto-resume watchdog.
#   train.sh pause    -> stop training + stand down the watchdog, free the GPU (zero work lost; for gaming etc.)
#   train.sh resume   -> re-arm the watchdog + resume training from the last checkpoint
#   train.sh status   -> is it running? progress, GPU, pause/stranded state
# Safe to run repeatedly (idempotent). Uses PID-based kills only (never `pkill -f`, which can self-kill).
set -u
ROOT=/home/dant123/Projects/sim/bridges/lmtrain/run3
SVC=lmtrain-resume.service
WD=gpu-train-watchdog.service

_step(){ tail -1 "$ROOT/progress.jsonl" 2>/dev/null | grep -oE '"step": [0-9]+' | grep -oE '[0-9]+' | head -1; }
_gpu(){ nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1; }
_ntrain(){ ps -eo args | grep -c '[l]m_train_run'; }

case "${1:-status}" in
  pause|stop)
    touch "$ROOT/PAUSE"                         # watchdog + service both respect this
    sudo systemctl stop "$WD"  2>/dev/null      # no auto-reboot/-resume while paused
    sudo systemctl stop "$SVC" 2>/dev/null      # graceful SIGTERM -> checkpoint -> exit
    for i in $(seq 1 8); do                     # clear any stragglers so the GPU is fully freed
      PIDS=$(ps -eo pid,args | grep '[l]m_train_run' | awk '{print $1}')
      [ -z "$PIDS" ] && break
      for P in $PIDS; do kill "$P" 2>/dev/null; done
      sleep 3
    done
    for P in $(ps -eo pid,args | grep '[l]m_train_run' | awk '{print $1}'); do kill -9 "$P" 2>/dev/null; done
    sleep 1
    echo "PAUSED  | training procs: $(_ntrain)  | checkpoint safe @ step $(_step)  | GPU: $(_gpu)"
    ;;
  resume|unpause|start)
    rm -f "$ROOT/PAUSE"
    sudo rm -f /var/tmp/gpu_train_watchdog/STRANDED   # clear any prior circuit-breaker latch
    sudo systemctl reset-failed "$SVC" 2>/dev/null
    sudo systemctl start "$SVC" 2>/dev/null
    sudo systemctl start "$WD"  2>/dev/null
    sleep 6
    echo "RESUMING  | lmtrain=$(systemctl is-active "$SVC")  watchdog=$(systemctl is-active "$WD")"
    grep -iE 'RESUME @|error' "$ROOT/boot_resume.log" 2>/dev/null | tail -1
    echo "(torch compiles a few min before the GPU ramps — watch: tail -f $ROOT/progress.jsonl)"
    ;;
  status)
    echo "lmtrain=$(systemctl is-active "$SVC")  watchdog=$(systemctl is-active "$WD")  | train procs: $(_ntrain)  | GPU: $(_gpu)"
    echo "PAUSE: $([ -f "$ROOT/PAUSE" ] && echo SET || echo none)   STRANDED: $([ -f /var/tmp/gpu_train_watchdog/STRANDED ] && echo 'YES (GPU dead - manual fix)' || echo no)"
    echo "recent progress:"; tail -2 "$ROOT/progress.jsonl" 2>/dev/null | grep -oE '"step": [0-9]+.*"val_ppl": [0-9.]+' | sed 's/^/  /'
    ;;
  *)
    echo "usage: train.sh {pause|resume|status}"; exit 1 ;;
esac
