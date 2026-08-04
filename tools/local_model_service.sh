#!/usr/bin/env bash
# Run the prepared local model while holding the shared GPU lease.
# Keep this process alive for the lifetime of the service; do not use
# `docker compose up -d` for the GPU-backed local model.
set -euo pipefail

ACTION="${1:-up}"
CLUB_ROOT="${CLUB_3090_ROOT:-$HOME/Projects/club-3090}"
ENV_FILE="${CLUB_3090_ENV:-$CLUB_ROOT/.env}"
COMPOSE_FILE="${CLUB_3090_COMPOSE:-$CLUB_ROOT/models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml}"
LEASE_PATH="${SIM_GPU_LEASE_PATH:-/tmp/sim-local-model-gpu0.lock}"

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

case "$ACTION" in
  up)
    exec 9>"$LEASE_PATH"
    if ! flock -n 9; then
      echo "[local-model] GPU lease is busy; an experiment or another model service owns GPU 0" >&2
      exit 75
    fi
    cleanup() {
      compose down >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM
    compose up
    ;;
  down)
    compose down
    ;;
  ps)
    compose ps
    ;;
  *)
    echo "usage: $0 {up|down|ps}" >&2
    exit 2
    ;;
esac
