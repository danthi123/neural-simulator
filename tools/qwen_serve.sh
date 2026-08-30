#!/usr/bin/env bash
# qwen_serve.sh — one-command launch/stop of the local Qwen3.8-27B (Q2) target + DFlash2 speculative drafter
# on an OpenAI-compatible endpoint, for Hermes to use as the fallback dev-agent brain when Claude usage runs out.
#
#   bash tools/qwen_serve.sh up        # download-on-first-run + launch llama-server (detached), wait until ready
#   bash tools/qwen_serve.sh down      # stop the server, free ALL its VRAM
#   bash tools/qwen_serve.sh status    # running? endpoint reachable? VRAM used?
#   bash tools/qwen_serve.sh restart   # down then up
#
# The VRAM-aware supervisor (tools/qwen_supervisor.sh) calls up/down automatically around local GPU runs;
# you rarely call this directly except the very first `up` (which triggers the ~9 GB HF auto-download).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"                       # reuse the queue dir for sentinels/pidfiles (already gitignored)
PIDF="$STATE/qwen_server.pid"
LOG="$STATE/qwen_server.log"

# --- configurable (env overrides) ------------------------------------------------------------------------------------
LLAMA="${QWEN_LLAMA_SERVER:-/home/dant123/.unsloth/llama.cpp/llama-server}"   # unsloth build has --spec-type draft-dflash
# Target: prefer the already-downloaded local GGUF (fast, no HF re-resolution); fall back to -hf auto-download.
TARGET_GGUF="${QWEN_TARGET_GGUF:-/home/dant123/.cache/huggingface/hub/models--sdkyuan--qwen3.8-27B-qat-q2_0-gguf/snapshots/a5885499d443cbf4a7998001508ddb3b279eeb5f/qwen38-27b-qat-q2_0.gguf}"
TARGET_HF="${QWEN_TARGET_HF:-sdkyuan/qwen3.8-27B-qat-q2_0-gguf}"              # fallback if the local GGUF is absent
DRAFT_HF="${QWEN_DRAFT_HF:-HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF:Q2_K_S}"  # the DFlash2 drafter (535 MiB, auto-downloads)
PORT="${QWEN_PORT:-8033}"
HOSTADDR="${QWEN_HOST:-127.0.0.1}"
CTX="${QWEN_CTX:-163840}"
NGL="${QWEN_NGL:-99}"
SPEC_NMAX="${QWEN_SPEC_NMAX:-3}"                                              # DFlash2 draft block size

endpoint() { echo "http://$HOSTADDR:$PORT/v1"; }
running()  { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; }
ready()    { curl -sf -m 4 "http://$HOSTADDR:$PORT/health" >/dev/null 2>&1 || curl -sf -m 4 "$(endpoint)/models" >/dev/null 2>&1; }
# timeout 8: when the 3090 falls off the bus, nvidia-smi HANGS rather than erroring (documented GPU-crash mode).
# Without the cap, a status echo here hangs `down`/`up`, whose caller (loop.py) then TimeoutExpires and crashes.
# Mirrors gpu_queue.sh:freevram. (M5)
vram()     { command -v nvidia-smi >/dev/null 2>&1 && timeout 8 nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo n/a; }

case "${1:-status}" in
  up)
    # Target via -hf (resolves from the HF cache — the GGUF is already downloaded, so NO re-download — and,
    # crucially, keeps the HF machinery active so the -hfd DRAFT resolves; mixing local --model with -hfd made
    # the draft path resolve to '' and the server exited on "failed to load draft model, ''").
    _launch() {   # $1 = draft|nodraft
      local a=(-hf "$TARGET_HF" --jinja --reasoning-budget -1 --ctx-size "$CTX" --host "$HOSTADDR" --port "$PORT" \
               -ngl "$NGL" --flash-attn on --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
               --repeat-penalty 1.0 --no-mmproj)
      [ "$1" = draft ] && a+=(-hfd "$DRAFT_HF" --spec-type draft-dflash --spec-draft-n-max "$SPEC_NMAX")
      setsid "$LLAMA" "${a[@]}" </dev/null >>"$LOG" 2>&1 &
      echo $! > "$PIDF"
    }
    _wait_ready() {   # returns 0 ready, 1 exited/timeout
      for _ in $(seq 1 "${1:-600}"); do ready && return 0; running || return 1; sleep 2; done; return 1
    }
    # M7: NEVER launch a second server while one is alive (loading OR ready). A `running`-but-not-`ready`
    # process is MID-LOAD; launching again would allocate the 27B a SECOND time during load -> OOM / card off
    # the bus. The old guard was `running && ready` (it fell through while loading). If one is loading, WAIT
    # for it; never double-launch.
    if running; then
      if ready; then echo "[qwen] already up + ready at $(endpoint)"; exit 0; fi
      echo "[qwen] a server is already loading (pid $(cat "$PIDF")) -> waiting for ready, NOT launching a second"
      if _wait_ready 300; then echo "[qwen] became ready at $(endpoint)  (VRAM: $(vram))"; exit 0; fi
      echo "[qwen] existing server still not ready after wait — leaving it in place (no double-launch)"; exit 1
    fi
    [ -x "$LLAMA" ] || { echo "[qwen] ERROR: llama-server not found/executable at $LLAMA (set QWEN_LLAMA_SERVER)"; exit 1; }
    # DFlash2 support check — refuse to launch a build that lacks the drafter (would silently fall back / error)
    if ! "$LLAMA" --help 2>&1 | grep -q 'draft-dflash'; then
      echo "[qwen] ERROR: $LLAMA has no --spec-type draft-dflash (needs a llama.cpp built after 2026-08-27). Aborting."; exit 1
    fi
    echo "[qwen] launching WITH DFlash2 drafter (drafter auto-downloads if absent, ~535 MiB)…  log: $LOG"
    _launch draft
    echo "[qwen] pid $(cat "$PIDF"); waiting (first run may download)…"
    if _wait_ready 600; then echo "[qwen] READY (DFlash2) at $(endpoint)  (VRAM: $(vram))"; exit 0; fi
    # Drafter failed? DFlash2 is a SPEED optimization, not correctness — fall back to target-only so Hermes still
    # gets a working brain. (Only retry if the failure was draft-related and the server actually exited.)
    if ! running && grep -qi "draft model" "$LOG" 2>/dev/null; then
      echo "[qwen] ⚠ drafter failed to load — retrying TARGET-ONLY (no DFlash2; a bit slower, fully functional)…"
      "$0" down >/dev/null 2>&1; sleep 1
      _launch nodraft
      if _wait_ready 600; then echo "[qwen] READY (target-only, no DFlash2) at $(endpoint)  (VRAM: $(vram))"; exit 0; fi
    fi
    echo "[qwen] server failed/timed out — see $LOG:"; tail -8 "$LOG"; exit 1 ;;
  down)
    if running; then p=$(cat "$PIDF"); kill -TERM "$p" 2>/dev/null; for _ in 1 2 3 4 5; do kill -0 "$p" 2>/dev/null || break; sleep 1; done; kill -KILL "$p" 2>/dev/null; fi
    # belt-and-suspenders: any stray llama-server on our port
    pkill -f "llama-server .*--port $PORT" 2>/dev/null || true
    rm -f "$PIDF"; echo "[qwen] down (VRAM: $(vram))" ;;
  restart) "$0" down; sleep 2; "$0" up ;;
  status)
    if running; then echo "[qwen] UP (pid $(cat "$PIDF")) ready=$(ready && echo yes || echo no) endpoint=$(endpoint)"; else echo "[qwen] down"; fi
    echo "[qwen] VRAM: $(vram)" ;;
  *) echo "usage: bash tools/qwen_serve.sh {up|down|restart|status}"; exit 2 ;;
esac
