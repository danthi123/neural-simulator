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
# CUTOVER 2026-09-06 (Q2 -> Q4_K_M): served TARGET-ONLY via local --model. Q4_K_M (Unsloth Dynamic UD-Q4_K_M,
# ~16GB weights) keeps ~92-95% quality vs Q2's degraded code/reasoning, and fits the ~19-20GB monitor budget at
# 32k ctx + Q8 KV (VALIDATED 2026-09-06: 19.3GB, coherent). No Q4 DFlash2 drafter exists -> target-only (DRAFT_HF
# empty). REVERT to the old Q2+DFlash2 path with env: QWEN_TARGET_GGUF=<q2 gguf> QWEN_TARGET_HF=sdkyuan/... \
#   QWEN_DRAFT_HF=HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF:Q2_K_S QWEN_CTX=163840 QWEN_KV_TYPE=f16
TARGET_GGUF="${QWEN_TARGET_GGUF:-/home/dant123/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf}"
TARGET_HF="${QWEN_TARGET_HF:-unsloth/Qwen3.8-27B-GGUF:Q4_K_M}"               # fallback if the local GGUF is absent
DRAFT_HF="${QWEN_DRAFT_HF:-}"                                                # empty = target-only (no Q4 DFlash2 drafter)
PORT="${QWEN_PORT:-8033}"
HOSTADDR="${QWEN_HOST:-127.0.0.1}"
CTX="${QWEN_CTX:-32768}"                                                     # 32k fits the ~19-20GB budget (65k=20.3GB, tight)
KV_TYPE="${QWEN_KV_TYPE:-q8_0}"                                              # Q8 KV cache — keeps VRAM in budget (f16 to revert)
NGL="${QWEN_NGL:-99}"
SPEC_NMAX="${QWEN_SPEC_NMAX:-3}"                                              # DFlash2 draft block size (only if DRAFT_HF set)

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
      local a=()
      if [ "$1" = draft ] && [ -n "$DRAFT_HF" ]; then
        # Q2+DFlash2 path: -hf (NOT local --model) keeps HF machinery active so -hfd resolves; local --model + -hfd
        # made the draft path resolve to '' and the server exited (documented). Drafter only when DRAFT_HF is set.
        a=(-hf "$TARGET_HF" -hfd "$DRAFT_HF" --spec-type draft-dflash --spec-draft-n-max "$SPEC_NMAX")
      elif [ -f "$TARGET_GGUF" ]; then
        a=(--model "$TARGET_GGUF")            # Q4 target-only: local file on disk (fast, no HF re-resolution)
      else
        a=(-hf "$TARGET_HF")                  # target-only fallback: auto-download from HF
      fi
      a+=(--jinja --reasoning-budget -1 --ctx-size "$CTX" --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" \
          --host "$HOSTADDR" --port "$PORT" -ngl "$NGL" --flash-attn on --parallel 1 --temp 1.0 --top-p 0.95 \
          --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --no-mmproj)
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
    if [ -n "$DRAFT_HF" ]; then
      # DFlash2 support check — refuse to launch a build that lacks the drafter (would silently fall back / error)
      if ! "$LLAMA" --help 2>&1 | grep -q 'draft-dflash'; then
        echo "[qwen] ERROR: $LLAMA has no --spec-type draft-dflash (needs a llama.cpp built after 2026-08-27). Aborting."; exit 1
      fi
      echo "[qwen] launching WITH DFlash2 drafter (drafter auto-downloads if absent, ~535 MiB)…  log: $LOG"
      _launch draft
      echo "[qwen] pid $(cat "$PIDF"); waiting (first run may download)…"
      if _wait_ready 600; then echo "[qwen] READY (DFlash2) at $(endpoint)  (VRAM: $(vram))"; exit 0; fi
      # Drafter failed? DFlash2 is a SPEED optimization, not correctness — fall through to target-only so Hermes
      # still gets a working brain. (Only retry if the failure was draft-related and the server actually exited.)
      if ! running && grep -qi "draft model" "$LOG" 2>/dev/null; then
        echo "[qwen] ⚠ drafter failed to load — retrying TARGET-ONLY (no DFlash2; a bit slower, fully functional)…"
        "$0" down >/dev/null 2>&1; sleep 1
      fi
    fi
    # TARGET-ONLY launch (the Q4 default path, or the Q2-drafter-failed fallback)
    echo "[qwen] launching TARGET-ONLY (model=$(basename "$TARGET_GGUF"), ctx=$CTX, KV=$KV_TYPE)…  log: $LOG"
    _launch nodraft
    echo "[qwen] pid $(cat "$PIDF"); waiting (first run may download)…"
    if _wait_ready 600; then echo "[qwen] READY (target-only) at $(endpoint)  (VRAM: $(vram))"; exit 0; fi
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
