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
TARGET_HF="${QWEN_TARGET_HF:-sdkyuan/qwen3.8-27B-qat-q2_0-gguf}"              # the 27B Q2 target (auto-downloads via -hf)
DRAFT_HF="${QWEN_DRAFT_HF:-HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF:Q2_K_S}"  # the DFlash2 block-diffusion drafter
PORT="${QWEN_PORT:-8033}"
HOSTADDR="${QWEN_HOST:-127.0.0.1}"
CTX="${QWEN_CTX:-32768}"
NGL="${QWEN_NGL:-99}"
SPEC_NMAX="${QWEN_SPEC_NMAX:-3}"                                              # DFlash2 draft block size

endpoint() { echo "http://$HOSTADDR:$PORT/v1"; }
running()  { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; }
ready()    { curl -sf -m 4 "http://$HOSTADDR:$PORT/health" >/dev/null 2>&1 || curl -sf -m 4 "$(endpoint)/models" >/dev/null 2>&1; }
vram()     { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo n/a; }

case "${1:-status}" in
  up)
    if running && ready; then echo "[qwen] already up + ready at $(endpoint)"; exit 0; fi
    [ -x "$LLAMA" ] || { echo "[qwen] ERROR: llama-server not found/executable at $LLAMA (set QWEN_LLAMA_SERVER)"; exit 1; }
    # DFlash2 support check — refuse to launch a build that lacks the drafter (would silently fall back / error)
    if ! "$LLAMA" --help 2>&1 | grep -q 'draft-dflash'; then
      echo "[qwen] ERROR: $LLAMA has no --spec-type draft-dflash (needs a llama.cpp built after 2026-08-27). Aborting."; exit 1
    fi
    echo "[qwen] launching (first run auto-downloads ~9 GB from HF)…  log: $LOG"
    setsid "$LLAMA" \
      -hf "$TARGET_HF" -hfd "$DRAFT_HF" \
      --spec-type draft-dflash --spec-draft-n-max "$SPEC_NMAX" \
      --jinja --reasoning-budget -1 --ctx-size "$CTX" \
      --host "$HOSTADDR" --port "$PORT" -ngl "$NGL" --flash-attn on \
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
      --no-mmproj \
      </dev/null >>"$LOG" 2>&1 &
    echo $! > "$PIDF"
    echo "[qwen] pid $(cat "$PIDF"); waiting for endpoint (first run can take minutes for the download)…"
    for _ in $(seq 1 600); do ready && { echo "[qwen] READY at $(endpoint)  (VRAM: $(vram))"; exit 0; }; running || { echo "[qwen] server exited early — see $LOG:"; tail -5 "$LOG"; exit 1; }; sleep 2; done
    echo "[qwen] timed out waiting for ready (still downloading? check $LOG)"; exit 1 ;;
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
