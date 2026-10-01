#!/usr/bin/env bash
# launch.sh -- the model bake-off's containers on homelab-server (docs: tools/model_bakeoff/README.md).
#
# Runs on the server HOST as dant123, but never from the agent-writable workspace (migration plan section 10): pipe
# it from a reviewed checkout instead, e.g. from the workstation:
#   ssh homelab-server 'bash -s -- init'                       < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- prepare <rev>'              < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- preflight'                  < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- run screen1 --parallel 3'   < tools/model_bakeoff/launch.sh   (detached)
#   ssh homelab-server 'bash -s -- log screen1'                < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- report screen1'             < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- fetch screen1' > results.tgz < tools/model_bakeoff/launch.sh
#   ssh homelab-server 'bash -s -- stop screen1' | 'bash -s -- destroy'
#
# Storage: ONE Docker volume, `sim-bakeoff` (tools/ bases/ runs/ home/), on Docker's own storage: outside backup.sh's
# roots and outside sanoid, so ~20 GB of disposable clones never reach the nightly backup or the offsite copy.
# `destroy` removes it when the bake-off is over.
#
# A model run's container gets ONLY: the tools and per-task bases (read-only sub-paths of the volume), its runs dir
# and HERMES_HOME (read-write), the OpenRouter key file (read-only), the project's Python environment and the
# reference library (read-only, at their real paths). No workspace, no ssh keys, no RAG index, no Hermes data.
# Same image as Hermes, the agent's uid, 6 CPUs / 10 GB, removed when it exits. `prepare` alone sees the workspace
# (read-only, no network): it builds the bases, each holding history only up to its task's base commit.
set -euo pipefail
cmd=${1:?usage: launch.sh init|prepare <rev>|preflight|run <batch> [args]|log|judge|report|fetch|stop <batch>|destroy}; shift
HD=/home/dant123/data/hermes          # Hermes's data dir on the host (= /opt/data in the Hermes container)
WS=/home/dant123/Projects/neural-simulator
VOL=sim-bakeoff
IMG=$(docker inspect -f '{{.Config.Image}}' hermes)
mnt() { echo "type=volume,src=$VOL,dst=/bk/$1,volume-subpath=$1${2:+,$2}"; }
COMMON=(--rm --user 99:100 --cpus 6 --memory 10g --pids-limit 4096
  -e HERMES_HOME=/bk/home -e BAKEOFF_ROOT=/bk -e BAKEOFF_SUITE="${BAKEOFF_SUITE:-suite.json}"
  -e PATH=/opt/hermes/.venv/bin:/usr/local/bin:/usr/bin:/bin
  --mount "$(mnt tools readonly)" --mount "$(mnt bases readonly)" --mount "$(mnt runs)" --mount "$(mnt home)"
  -v "$HD/.env:/bk/home/.env:ro"
  -v "$WS/.venv:$WS/.venv:ro" -v "$WS/.python:$WS/.python:ro"
  -v /home/dant123/Projects/sim-catalog:/home/dant123/Projects/sim-catalog:ro
  -w /bk/tools --entrypoint /bin/sh)

case "$cmd" in
  init)
    docker volume create "$VOL" >/dev/null
    docker run --rm --user 0 -v "$VOL:/bk" --entrypoint /bin/sh "$IMG" -c \
      'mkdir -p /bk/tools /bk/bases /bk/runs /bk/home && touch /bk/home/.env && chown -R 99:100 /bk'
    echo "volume $VOL ready" ;;
  prepare)
    rev=${1:?rev}; shift; [[ "$rev" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "bad rev"; exit 2; }
    # No network; the workspace and Hermes's config read-only. git's ownership check is relaxed for this read-only use
    # through a global config in the throwaway container's own HOME (env-only config does not reach the clone's
    # upload-pack helper).
    docker run --rm --user 99:100 --network none -e HOME=/tmp \
      -e BAKEOFF_DIR=/bk -e BAKEOFF_REV="$rev" -e BAKEOFF_SUITE="${BAKEOFF_SUITE:-suite.json}" \
      -v "$VOL:/bk" -v "$WS:$WS:ro" -v "$HD/config.yaml:/bk-hermes-config.yaml:ro" -e BAKEOFF_HERMES_CONFIG=/bk-hermes-config.yaml \
      --entrypoint /bin/sh "$IMG" -c \
      "git config --global --add safe.directory '*' && git -C $WS show $rev:tools/model_bakeoff/prepare.sh > /tmp/prepare.sh && bash /tmp/prepare.sh ${*}" ;;
  run)
    batch=${1:?batch}; shift
    inner="python3 run.py --batch $batch $* >> /bk/runs/$batch.log 2>&1"
    [[ "$inner" =~ ^[A-Za-z0-9\ ./_=,:\>\&-]+$ ]] || { echo "refusing unexpected characters"; exit 2; }
    docker run -d --name "sim-bakeoff-$batch" "${COMMON[@]}" "$IMG" -c "$inner" ;;
  preflight) docker run -i "${COMMON[@]}" "$IMG" -c "python3 run.py --preflight" ;;
  judge)  b=${1:?batch}; [[ "$b" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2; docker run -i "${COMMON[@]}" "$IMG" -c "python3 judge.py --batch $b" ;;
  report) b=${1:?batch}; [[ "$b" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
          docker run -i "${COMMON[@]}" "$IMG" -c "python3 report.py --batch $b --out /bk/runs/$b/report.md" ;;
  log)    b=${1:?batch}; [[ "$b" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
          docker run --rm -v "$VOL:/bk:ro" --entrypoint /bin/sh "$IMG" -c "tail -n ${2:-40} /bk/runs/$b.log" ;;
  fetch)  b=${1:?batch}; [[ "$b" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2   # results only (no clones): a tar.gz on stdout
          docker run --rm -v "$VOL:/bk:ro" --entrypoint /bin/sh "$IMG" -c \
            "cd /bk/runs && tar -czf - --exclude='*/repo' --exclude='*/home' --exclude='*/sim-catalog' $b $b.log" ;;
  stop)    docker stop "sim-bakeoff-${1:?batch}" ;;
  destroy) docker volume rm "$VOL" ;;
  *) echo "unknown command: $cmd"; exit 2 ;;
esac
