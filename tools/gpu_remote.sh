#!/usr/bin/env bash
# gpu_remote.sh -- queue and watch GPU jobs from the project's home (the server, no GPU) on the workstation's 3090.
#
#   bash tools/gpu_remote.sh add '<cmd>'             # queue <cmd> at THIS checkout's committed HEAD (cmd runs from the repo root)
#   bash tools/gpu_remote.sh status                  # the workstation's queue: dispatcher, current job, depth
#   bash tools/gpu_remote.sh log [N]                 # last N (default 30) lines of the workstation's gpu_queue.log
#   bash tools/gpu_remote.sh yield [--max-hours N]   # ask the local model to yield the 3090 to the queue (llm.sh yield)
#   bash tools/gpu_remote.sh provision <sha>         # run <sha>'s own pool_provision.sh --isolated on the workstation
#   bash tools/gpu_remote.sh --dry-run <subcommand> ...   # print the script that would run on the workstation, run nothing
#
# WHY (2026-09-30, docs/plans/2026-09-30-home-to-server-MIGRATION.md section 3). The project's home moved to a server
# with no GPU; the 3090, its queue (tools/gpu_queue.sh, one daemon, one singleton lock) and the local model stay on the
# workstation. gpu_add.sh/gpu_queue.sh are single-host by design, and nothing else creates a checkout of a server
# revision there, so this is the one bridge. It adds no queue of its own:
#   add        fetches this repo on the workstation (remote `server`), makes a per-job worktree of HEAD under
#              ~/derisk-pool/revisions/<sha>-<epoch> WITHOUT research/findings/raw (as pool_provision.sh excludes it;
#              the results come back through `POOL_NODES=workstation tools/pool_sync.sh`, the pool-node layout),
#              symlinks .venv and data to the workstation's, then execs THAT worktree's tools/local_llm/gpu_add.sh
#              "cd <worktree> && <cmd>" -- so the guard runs at the job's own revision, into the workstation's queue.
#              One worktree per job, because gpu_add.sh refuses a checkout holding untracked outputs.
#   provision  pool_provision.sh needs rsync, the pool-node keys and the corpus; the workstation has all three.
# The job travels base64-encoded (the workstation's login shell is fish), like the pool dispatcher's JOB_B64.
#
# Env: SIM_GPU_HOST (ssh host, default `workstation`), SIM_GPU_HOST_REPO (the checkout there, default
# $HOME/Projects/sim, expanded on the workstation).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${SIM_GPU_HOST:-workstation}
WREPO=${SIM_GPU_HOST_REPO:-'$HOME/Projects/sim'}
DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }

usage() { sed -n '4,9p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

# mkwt <sha> <remote worktree path>: the remote lines that make a clean, sparse (no research/findings/raw) worktree.
mkwt() {
  cat <<EOF
R="$WREPO"
W="$2"
git -C "\$R" fetch -q server
git -C "\$R" worktree add -q --no-checkout --detach "\$W" $1
git -C "\$W" sparse-checkout set --no-cone '/*' '!/research/findings/raw/'
git -C "\$W" reset -q --hard
ln -s "\$R/.venv" "\$W/.venv"
ln -s "\$R/data" "\$W/data"
EOF
}

run_remote() {   # stdin: the remote script
  if [ "$DRY" = 1 ]; then echo "# ssh -o BatchMode=yes $HOST bash -s <<'SCRIPT'"; cat; echo "SCRIPT"; return 0; fi
  ssh -o BatchMode=yes "$HOST" bash -s
}

cmd_add() {
  local cmd="${1:-}"
  [ -n "$cmd" ] || { echo "usage: gpu_remote.sh add '<cmd>'" >&2; exit 2; }
  # The job runs the COMMITTED revision, so a local code edit would silently not be in it. Results and coordination
  # state (research/findings, research/coordination, research/queue) are data the job does not run, so they may differ.
  if ! git -C "$ROOT" diff --quiet HEAD -- . ':!research/findings' ':!research/coordination' ':!research/queue'; then
    echo "⛔ REFUSED: $ROOT has uncommitted changes to tracked code; commit them first (the job runs HEAD):" >&2
    git -C "$ROOT" diff --stat HEAD -- . ':!research/findings' ':!research/coordination' ':!research/queue' | tail -5 >&2
    exit 1
  fi
  if [ -z "$(git -C "$ROOT" branch --contains HEAD 2>/dev/null | grep -v 'HEAD detached')" ]; then
    echo "⛔ REFUSED: HEAD is on no branch, so the workstation cannot fetch it; commit on a branch first." >&2
    exit 1
  fi
  local sha b64
  sha=$(git -C "$ROOT" rev-parse HEAD)
  b64=$(printf '%s' "$cmd" | base64 -w0)
  {
    echo 'set -euo pipefail'
    mkwt "$sha" "\$HOME/derisk-pool/revisions/$sha-\$(date +%s)"
    echo "JOB=\$(printf %s '$b64' | base64 -d)"
    echo "echo \"gpu_remote: worktree \$W (revision $sha)\""
    # A refused job leaves nothing behind: its worktree was made for it a moment ago and nothing ran in it.
    echo 'bash "$W/tools/local_llm/gpu_add.sh" "cd $W && $JOB" && exit 0'
    echo 'rc=$?; git -C "$R" worktree remove --force "$W"; exit "$rc"'
  } | run_remote
}

cmd_provision() {
  local sha="${1:-}"
  [[ "$sha" =~ ^[0-9a-f]{7,40}$ ]] || { echo "usage: gpu_remote.sh provision <commit sha>" >&2; exit 2; }
  {
    echo 'set -euo pipefail'
    mkwt "$sha" "\$HOME/Projects/sim-worktrees/prov-$sha-\$(date +%s)"
    echo 'cd "$W"'
    echo "exec bash tools/pool_provision.sh --isolated --revision $sha"
  } | run_remote
}

case "${1:-}" in
  add) shift; cmd_add "$@" ;;
  status) printf '%s\n' "bash \"$WREPO/tools/gpu_queue.sh\" status" | run_remote ;;
  log)
    n="${2:-30}"; [[ "$n" =~ ^[0-9]+$ ]] || usage
    printf '%s\n' "tail -n $n \"$WREPO/research/queue/gpu_queue.log\"" | run_remote ;;
  yield)
    shift; args=""
    if [ "${1:-}" = "--max-hours" ]; then
      [[ "${2:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || usage
      args="--max-hours $2"
    elif [ -n "${1:-}" ]; then usage; fi
    printf '%s\n' "bash \"$WREPO/tools/local_llm/llm.sh\" yield $args" | run_remote ;;
  provision) shift; cmd_provision "$@" ;;
  *) usage ;;
esac
