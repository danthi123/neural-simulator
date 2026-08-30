#!/usr/bin/env bash
# hermes_local_setup.sh — point Hermes at the LOCAL Qwen (llama-server) OpenAI endpoint. Run ONCE (idempotent).
#
# Reliable path is Hermes' own wizard (auto-detects the model via GET /v1/models). This script backs up your
# Hermes config, ATTEMPTS the scripted config-set as a convenience, then tells you the exact wizard values in case
# the scripted keys differ across Hermes versions. Safe to re-run.
#
#   1. bash tools/qwen_serve.sh up        # (or hermes_takeover.sh on) so the endpoint is live for auto-detect
#   2. bash tools/hermes_local_setup.sh   # this script
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES="${HERMES_BIN:-/home/dant123/.local/bin/hermes}"
HHOME="${HERMES_HOME:-$HOME/.hermes}"
PORT="${QWEN_PORT:-8033}"
BASEURL="http://127.0.0.1:${PORT}/v1"
KEY="${QWEN_API_KEY:-sk-local-qwen}"    # llama-server ignores it unless --api-key is set; Hermes just needs one present
TS="$(date +%s 2>/dev/null || echo backup)"

echo "[hermes-setup] backing up Hermes config…"
[ -f "$HHOME/config.yaml" ] && cp -a "$HHOME/config.yaml" "$HHOME/config.yaml.bak.$TS" && echo "  -> $HHOME/config.yaml.bak.$TS"
[ -f "$HHOME/.env" ] && cp -a "$HHOME/.env" "$HHOME/.env.bak.$TS" && echo "  -> $HHOME/.env.bak.$TS"

echo "[hermes-setup] attempting scripted config (best-effort; non-fatal if a key name differs)…"
"$HERMES" config set model.base_url "$BASEURL" 2>/dev/null && echo "  set model.base_url=$BASEURL" || echo "  (could not set model.base_url via CLI — use the wizard below)"
"$HERMES" config set model.provider custom 2>/dev/null || "$HERMES" config set model.provider openai 2>/dev/null || true
# key into ~/.hermes/.env (Hermes reads secrets from there)
if [ -f "$HHOME/.env" ] && ! grep -q "OPENAI_API_KEY=" "$HHOME/.env" 2>/dev/null; then echo "OPENAI_API_KEY=$KEY" >> "$HHOME/.env"; fi
[ -f "$HHOME/.env" ] || echo "OPENAI_API_KEY=$KEY" > "$HHOME/.env"

echo
echo "[hermes-setup] VERIFY the endpoint is reachable (Hermes will auto-detect the model here):"
curl -sf -m 5 "$BASEURL/models" 2>/dev/null | head -c 300 || echo "  (endpoint not up yet — run qwen_serve.sh up first, then re-run this)"
echo
echo "[hermes-setup] If Hermes doesn't pick up the local model, run the wizard ONCE and enter these EXACT values:"
echo "     hermes setup     ->  Model & Provider  ->  Custom OpenAI-compatible endpoint"
echo "       API base URL : $BASEURL"
echo "       API key      : $KEY"
echo "       Model        : (accept the auto-detected one)"
echo "       Context      : (leave blank = auto)"
echo "[hermes-setup] then launch:  hermes"
echo "[hermes-setup] revert anytime:  cp $HHOME/config.yaml.bak.$TS $HHOME/config.yaml"
