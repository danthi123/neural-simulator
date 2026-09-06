#!/usr/bin/env bash
# Memory/compute-cap heartbeat body — emits ONE line per call. Re-armed as a COMMITTED script (not inline
# Monitor text) so a logic fix is live on the next cycle with NO re-arm, exactly like heartbeat_cmd.sh folds
# parallel_audit.py.
#
# THE FIX (2026-09-05): the cap verdict now keys on ACTUAL memory risk — RAM-available + swap THRASH RATE —
# NOT raw swap_used, which is a LAGGING indicator that false-alarms on benign idle-swap. Earned: after the KDE
# baloo indexer was suspended, ~13 GB of its idle pages sat in swap with 28 GB RAM free and swap I/O at ~60
# pages/s (quiescent); the old `swap_used > threshold` verdict read THROTTLE-swap for hours, which both blocked
# launches forever (swap won't drain without RAM pressure) and corroded the guard (a false alarm trains the
# reader to ignore it — the project's own silent-failure rule 8). See memory project_baloo_indexer_memory_hog.
# Real risk = RAM actually low, OR swap being actively churned (thrash). Idle pages parked in swap are free.
cd /home/dant123/Projects/sim
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr '\n' '|')
load=$(awk '{print $1}' /proc/loadavg)
ram_avail_g=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
swap_used_g=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024/1024)}' /proc/meminfo)
# swap THRASH rate: pages swapped in+out over a 1s window (4 KB/page). Quiescent parked swap ~= tens of pages/s;
# genuine thrash is thousands/s sustained. This is the signal that distinguishes danger from benign idle-swap.
r1=$(awk '/pswpin|pswpout/{s+=$2} END{print s}' /proc/vmstat); sleep 1
r2=$(awk '/pswpin|pswpout/{s+=$2} END{print s}' /proc/vmstat)
swap_io=$(( r2 - r1 ))
runners=$(pgrep -fc "research.runners" 2>/dev/null | head -1); runners=${runners:-0}
gpu_q=$(grep -cve '^[[:space:]]*$' research/queue/gpu.queue 2>/dev/null); gpu_q=${gpu_q:-0}
# VERDICT — 20-core box. THROTTLE only on REAL risk:
if   [ "${ram_avail_g:-99}" -lt 4 ]; then cap="THROTTLE-RAM"                                   # genuine RAM exhaustion
elif [ "$swap_io" -gt 1500 ] && [ "${swap_used_g:-0}" -gt 4 ]; then cap="THROTTLE-swap-thrash" # active churn, not parked
elif awk -v l="$load" 'BEGIN{exit !(l>32)}'; then cap="THROTTLE-load"                          # >1.6x cores
else cap="OK"; fi
par=$(.venv/bin/python tools/parallel_audit.py 2>/dev/null | grep -oiE "SATURATED|UNDER-PARALLELIZED|SERIAL" | head -1); par=${par:-?}
# swap_used shown for context but LABELLED benign when it is parked-not-churned, so the number never reads as alarm on its own
sw_note=""; [ "${swap_used_g:-0}" -gt 4 ] && [ "$swap_io" -le 1500 ] && sw_note="(parked/idle)"
echo "HB $(date +%H:%M) gpu=[$gpu] load=$load ram_avail=${ram_avail_g}G swap_used=${swap_used_g}G${sw_note} swap_io=${swap_io}pg/s gpu_q=$gpu_q runners=$runners cap=$cap parallel=$par"
# unpushed safety (the only copy is this disk; branch-aware)
cur_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
up_sha=$(git ls-remote origin "refs/heads/${cur_branch}" 2>/dev/null | cut -f1); [ -z "$up_sha" ] && up_sha=$(git ls-remote origin refs/heads/main 2>/dev/null | cut -f1)
ahead=$(git rev-list --count "${up_sha}"..HEAD 2>/dev/null || echo 0)
[ "$ahead" != "0" ] && echo "⛔ $ahead COMMIT(S) UNPUSHED — run: bash tools/push_both.sh"
