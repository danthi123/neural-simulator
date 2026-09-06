#!/usr/bin/env bash
# vllm_sleep_pilot_serve.sh — PILOT launch/stop of Qwen3.8-27B on vLLM 0.27.1, Sleep-Mode-enabled, sized for
# the ~19-20GB budget (24GB 3090 minus ~3.5-5GB monitors/desktop, no 2nd GPU). This is a PILOT of vLLM Sleep
# Mode (`/sleep`+`/wake_up`, ~3-6s in-place VRAM release) as a possible faster alternative to qwen_serve.sh's
# kill+cold-reload dance for sharing the one 3090 between the local model and brain experiments.
#
# tools/qwen_serve.sh (llama.cpp, Q4_K_M GGUF, port 8033) is UNTOUCHED and remains the proven fallback — this
# script is additive, a separate port (18020), a separate venv, a separate model checkout. Nothing here
# perturbs the sim's own .venv or qwen_serve.sh's state files.
#
#   bash tools/vllm_sleep_pilot_serve.sh up        # launch vllm serve (detached), wait until ready
#   bash tools/vllm_sleep_pilot_serve.sh down      # stop the server, free ALL its VRAM
#   bash tools/vllm_sleep_pilot_serve.sh status    # running? endpoint reachable? sleeping? VRAM used?
#   bash tools/vllm_sleep_pilot_serve.sh sleep [1|2]   # POST /sleep?level=N  (default 1)
#   bash tools/vllm_sleep_pilot_serve.sh wake          # POST /wake_up
#   bash tools/vllm_sleep_pilot_serve.sh restart
#
# ⚠️ GPU SHARING GUARD (hard requirement, this pilot's owner-approved scope): `up` REFUSES to launch while any
# `research.runners` / `webapp` python process is GPU-resident (same detection pattern as
# tools/gpu_queue.sh:gpu_resident_brain_pids — a `research.runners`/`webapp` python cmdline in nvidia-smi's
# compute-apps pid list), because loading a second ~15GB model onto a card a brain job already owns is the
# OOM/bus-off failure mode this project has hit before (CLAUDE.md, docs/GPU_CRASH_RECOVERY.md). Set
# VLLM_PILOT_FORCE=1 to override (only after you have personally confirmed via `nvidia-smi` the card is free —
# never set this from an automated loop). Prefer: queue this script's `up` through the controller once
# `tools/gpu_queue.sh status` shows the card idle, or run it by hand after confirming `nvidia-smi`.
#
# ⚠️ KNOWN vLLM 0.27.1 BUG on Ampere (RTX 3090 = compute capability 8.6, same class as the reported RTX A5000):
# CUDA-graph capture for the Gated-DeltaNet hybrid-attention kernel HANGS INDEFINITELY on Ampere at startup
# (vllm-project/vllm#52682, open, no merged fix as of this pilot — 2026-09-06 WebSearch). This script therefore
# ALWAYS passes --enforce-eager. Do not remove it to chase throughput until that issue is closed upstream, or
# `up` may wedge (no crash, no error — the process just parks in graph capture forever; `_wait_ready`'s
# timeout below will fire, but VRAM is not released until you `down` it, so watch for a stuck `up`).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"                       # reuse the queue dir for sentinels/pidfiles (already gitignored)
PIDF="$STATE/vllm_pilot_server.pid"
LOG="$STATE/vllm_pilot_server.log"

# --- configurable (env overrides) ------------------------------------------------------------------------------------
# The venv lives in the syv-ai/qwen38-27b-rtx3090 checkout (a separate, already-cloned reference project for
# exactly this model+card combo — see research/findings/2026-09-06-vllm-sleep-mode-pilot-*.md for how it was
# found and what it contributed), NOT in sim's own .venv — keeps vLLM's pinned torch/triton/vllm stack from
# ever touching the sim's CuPy/numpy dependency set.
VLLM_VENV="${VLLM_VENV:-/home/dant123/Projects/qwen38-27b-rtx3090/venv}"
VLLM_PY="$VLLM_VENV/bin/python"
# Base (non-"-fast") W4A16-AutoRound requantization: already downloaded+requantized locally by that reference
# project, so `up` needs NO new HF download. The "-fast" sibling (int4 lm_head+MTP) is speculative-decoding-
# oriented and untested here; override VLLM_MODEL_DIR to try it.
VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-/home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound}"
SERVED_NAME="${VLLM_SERVED_NAME:-qwen3.8-27b-pilot}"
PORT="${VLLM_PORT:-18020}"                          # matches the reference repo's own convention; no collision with qwen_serve.sh's 8033
HOSTADDR="${VLLM_HOST:-127.0.0.1}"                  # loopback by default — this is a pilot, not a shared service
API_KEY="${VLLM_API_KEY:-}"                         # empty = no auth (fine on loopback-only); set to require a bearer token
# ~19-20GB budget out of a 24GB (24576 MiB) card: 0.80 -> ~19.66GB overall ceiling (weights + activation peak +
# KV pool), a SAFETY BOUND, not the KV sizing knob (see KV_MEM_BYTES below). Raise toward 0.83 ONLY after
# confirming via `nvidia-smi` that desktop/monitor usage is at the low end (~3.5GB) right before this specific
# `up`; the ceiling is a fraction of TOTAL device memory, not of currently-free, so it does not auto-adjust to
# what else is resident.
GPU_UTIL="${VLLM_GPU_UTIL:-0.80}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-81920}"              # ~80k ctx — middle of the 64-100k target; see the findings doc's VRAM-fit math
KV_DTYPE="${VLLM_KV_DTYPE:-fp8}"                    # Q8 KV. Untested against this hybrid checkpoint on stock 0.27.1 — see findings doc Plan B (auto/bf16) if this errors
# `--kv-cache-memory-bytes` (confirmed present, stock, `vllm serve --help=all`) PINS the KV pool size instead of
# deriving it from gpu_memory_utilization's run-to-run-noisy activation-peak profiling (the syv-ai/qwen38-27b-
# rtx3090 reference project's gotcha #16 measured a 1.09 vs 1.96 GiB profiled-peak swing between cold-cache
# starts of the SAME config — pinning bytes removes that variance from the number that matters most here).
# Default 2.8 GiB, from the findings doc's math: ~19GB total ceiling - ~15.0GB (14.26 GiB language-model-only
# weights, per that project's docs/gotchas.md #9, + ~0.4 GiB non-weight overhead) - ~1.2GB activation-peak
# safety margin (eager-mode forward pass, no captured-graph memory to reserve) ≈ 2.8GB, which at fp8 KV and
# 32,768 bytes/token (16 full-attention layers × 2 (K+V) × 4 kv_heads × 256 head_dim × 1 byte — the config.json
# numbers, see the findings doc) gives ≈ 91k tokens of context, comfortably inside the 64-100k target.
KV_MEM_BYTES="${VLLM_KV_CACHE_BYTES:-3006477107}"   # 2.8 GiB
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"              # single-user pilot: maximize per-request context over concurrency
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"                   # escape hatch, e.g. VLLM_EXTRA_ARGS='--trust-remote-code'

endpoint() { echo "http://$HOSTADDR:$PORT"; }
running()  { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; }
ready()    { curl -sf -m 4 "$(endpoint)/health" >/dev/null 2>&1; }
auth_hdr() { [ -n "$API_KEY" ] && echo "-H Authorization:\ Bearer\ $API_KEY" || true; }
# timeout 8: mirrors qwen_serve.sh / gpu_queue.sh — when the 3090 falls off the bus nvidia-smi HANGS rather
# than erroring, which would otherwise hang this script's own status/guard calls forever.
vram()     { command -v nvidia-smi >/dev/null 2>&1 && timeout 8 nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo n/a; }
is_sleeping() { curl -sf -m 4 "$(endpoint)/is_sleeping" 2>/dev/null; }

# Ground truth for "is a brain-loading GPU process resident RIGHT NOW?" — same pattern as
# tools/gpu_queue.sh:gpu_resident_brain_pids(), duplicated here (not sourced) so this script has no load-bearing
# dependency on gpu_queue.sh's internals changing shape.
gpu_resident_brain_pids() {
  local p
  for p in $(timeout 8 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \r'); do
    [ -n "$p" ] || continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE 'python.*(research\.runners|webapp)' && echo "$p"
  done
}

case "${1:-status}" in
  up)
    if running; then
      if ready; then echo "[vllm-pilot] already up + ready at $(endpoint)"; exit 0; fi
      echo "[vllm-pilot] a server is already loading (pid $(cat "$PIDF")) -> not double-launching; check $LOG"; exit 1
    fi
    if [ "${VLLM_PILOT_FORCE:-0}" != "1" ]; then
      brains="$(gpu_resident_brain_pids)"
      if [ -n "$brains" ]; then
        echo "[vllm-pilot] REFUSING to launch — GPU-resident brain process(es) found: $brains"
        echo "[vllm-pilot] loading a 2nd ~15GB model onto a card a brain job owns risks OOM/bus-off."
        echo "[vllm-pilot] wait for tools/gpu_queue.sh status to show idle, confirm via nvidia-smi, or set VLLM_PILOT_FORCE=1 to override (only by hand, only after you've checked)."
        exit 1
      fi
    fi
    [ -x "$VLLM_PY" ] || { echo "[vllm-pilot] ERROR: no venv python at $VLLM_PY (run: cd \$(dirname \$(dirname \$VLLM_PY)) && uv venv --python 3.14 venv && uv pip install --python venv/bin/python vllm==0.27.1 ...)"; exit 1; }
    [ -d "$VLLM_MODEL_DIR" ] || { echo "[vllm-pilot] ERROR: model dir not found: $VLLM_MODEL_DIR"; exit 1; }
    echo "[vllm-pilot] launching vllm serve (Sleep Mode ON, --enforce-eager for the Ampere GDN CUDA-graph hang, text-only)…  log: $LOG"
    echo "[vllm-pilot]   model=$VLLM_MODEL_DIR  ctx=$MAX_LEN  kv=$KV_DTYPE  gpu_util=$GPU_UTIL  port=$PORT"
    a=(serve "$VLLM_MODEL_DIR" --served-model-name "$SERVED_NAME" --host "$HOSTADDR" --port "$PORT" \
       --gpu-memory-utilization "$GPU_UTIL" --kv-cache-memory-bytes "$KV_MEM_BYTES" \
       --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
       --kv-cache-dtype "$KV_DTYPE" --language-model-only --enforce-eager --enable-sleep-mode)
    [ -n "$API_KEY" ] && a+=(--api-key "$API_KEY")
    [ -n "$EXTRA_ARGS" ] && a+=($EXTRA_ARGS)
    # $VLLM_VENV/bin/vllm is the console-script entry point installed alongside $VLLM_PY (confirmed present +
    # `serve --help=all` inspected for every flag used above — verified, not assumed, 2026-09-06).
    setsid env VLLM_SERVER_DEV_MODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
      "$VLLM_VENV/bin/vllm" "${a[@]}" </dev/null >>"$LOG" 2>&1 &
    echo $! > "$PIDF"
    echo "[vllm-pilot] pid $(cat "$PIDF"); waiting for /health (first start = torch.compile warmup, can take minutes)…"
    for _ in $(seq 1 240); do ready && { echo "[vllm-pilot] READY at $(endpoint)  (VRAM: $(vram))"; exit 0; }; running || { echo "[vllm-pilot] process exited during startup — see $LOG:"; tail -30 "$LOG"; exit 1; }; sleep 3; done
    echo "[vllm-pilot] not ready after 12 min — still running (pid $(cat "$PIDF")); this may be the Ampere CUDA-graph hang (#52682) if --enforce-eager was overridden out via VLLM_EXTRA_ARGS. Check $LOG and consider \`$0 down\`."; exit 1 ;;
  down)
    if running; then p=$(cat "$PIDF"); kill -TERM "$p" 2>/dev/null; for _ in 1 2 3 4 5 6 7 8; do kill -0 "$p" 2>/dev/null || break; sleep 1; done; kill -KILL "$p" 2>/dev/null; fi
    pkill -f "bin/vllm serve $VLLM_MODEL_DIR" 2>/dev/null || true
    rm -f "$PIDF"; echo "[vllm-pilot] down (VRAM: $(vram))" ;;
  restart) "$0" down; sleep 2; "$0" up ;;
  status)
    if running; then
      echo "[vllm-pilot] UP (pid $(cat "$PIDF")) ready=$(ready && echo yes || echo no) endpoint=$(endpoint)"
      ready && echo "[vllm-pilot] is_sleeping: $(is_sleeping)"
    else echo "[vllm-pilot] down"; fi
    echo "[vllm-pilot] VRAM: $(vram)" ;;
  sleep)
    ready || { echo "[vllm-pilot] not ready — nothing to sleep"; exit 1; }
    lvl="${2:-1}"
    echo "[vllm-pilot] POST /sleep?level=$lvl  (VRAM before: $(vram))"
    curl -sf -m 30 -X POST "$(endpoint)/sleep?level=$lvl" && echo
    sleep 1
    echo "[vllm-pilot] is_sleeping: $(is_sleeping)   VRAM after: $(vram)" ;;
  wake)
    echo "[vllm-pilot] POST /wake_up  (VRAM before: $(vram))"
    curl -sf -m 60 -X POST "$(endpoint)/wake_up" && echo
    sleep 1
    echo "[vllm-pilot] is_sleeping: $(is_sleeping)   VRAM after: $(vram)" ;;
  *) echo "usage: bash tools/vllm_sleep_pilot_serve.sh {up|down|restart|status|sleep [1|2]|wake}"; exit 2 ;;
esac
