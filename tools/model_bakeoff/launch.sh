#!/usr/bin/env bash
# launch.sh -- start the throwaway bake-off container on homelab-server (docs: tools/model_bakeoff/README.md).
#
# Runs on the server HOST as dant123, but never from the agent-writable workspace (migration plan section 10): pipe
# it from a reviewed checkout instead, e.g. from the workstation:
#   ssh homelab-server 'bash -s -- preflight'            < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- run screen1 --parallel 3' < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- judge screen1'        < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- report screen1'       < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- stop screen1'         < tools/model_bakeoff/launch.sh
# `run` is detached (log: <bakeoff>/runs/<batch>.log, read it as the agent); the others run in the foreground.
#
# The container gets ONLY: the bake-off tools and per-task bases (read-only), its runs dir and HERMES_HOME
# (read-write), the OpenRouter key file (read-only), the project's Python environment and the reference library
# (read-only, at their real paths). No workspace, no ssh keys, no RAG index, no Hermes data. Same image as Hermes,
# run as the agent's uid, capped at 6 CPUs / 10 GB, removed when it exits.
set -euo pipefail
cmd=${1:?usage: launch.sh preflight|run <batch> [args]|judge <batch>|report <batch>|stop <batch>}; shift
HD=/home/dant123/data/hermes          # Hermes's data dir on the host (= /opt/data in the Hermes container)
BKH=$HD/bakeoff                       # written by prepare.sh as the agent
WS=/home/dant123/Projects/neural-simulator
IMG=$(docker inspect -f '{{.Config.Image}}' hermes)

case "$cmd" in
  stop) docker stop "sim-bakeoff-${1:?batch}" ; exit ;;
  run)  batch=${1:?batch}; shift; mode=(-d --name "sim-bakeoff-$batch")
        inner="python3 run.py --batch $batch $* >> /bk/runs/$batch.log 2>&1" ;;
  preflight) mode=(-i); inner="python3 run.py --preflight" ;;
  judge)  mode=(-i); inner="python3 judge.py --batch ${1:?batch}" ;;
  report) mode=(-i); inner="python3 report.py --batch ${1:?batch} --out /bk/runs/${1}/report.md" ;;
  *) echo "unknown command: $cmd"; exit 2 ;;
esac
[[ "${inner}" =~ ^[A-Za-z0-9\ ./_=,:\>\&-]+$ ]] || { echo "refusing unexpected characters in: $inner"; exit 2; }

docker run --rm "${mode[@]}" --user 99:100 --cpus 6 --memory 10g --pids-limit 4096 \
  -e HERMES_HOME=/bk/home -e BAKEOFF_ROOT=/bk -e BAKEOFF_SUITE="${BAKEOFF_SUITE:-suite.json}" -e PATH=/opt/hermes/.venv/bin:/usr/local/bin:/usr/bin:/bin \
  -v "$BKH/tools:/bk/tools:ro" -v "$BKH/bases:/bk/bases:ro" \
  -v "$BKH/runs:/bk/runs" -v "$BKH/home:/bk/home" -v "$HD/.env:/bk/home/.env:ro" \
  -v "$WS/.venv:$WS/.venv:ro" -v "$WS/.python:$WS/.python:ro" \
  -v /home/dant123/Projects/sim-catalog:/home/dant123/Projects/sim-catalog:ro \
  -w /bk/tools --entrypoint /bin/sh "$IMG" -c "$inner"
