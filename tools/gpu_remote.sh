#!/usr/bin/env bash
# gpu_remote.sh -- queue and watch GPU jobs from the project's home (the server, no GPU) on the workstation's 3090.
#
#   bash tools/gpu_remote.sh add '<cmd>'             # queue <cmd> at THIS checkout's committed HEAD (cmd runs from the repo root)
#   bash tools/gpu_remote.sh add --rev <sha> '<cmd>' # queue <cmd> at an older committed revision on a branch (a gate pin)
#   bash tools/gpu_remote.sh status                  # the workstation's queue: dispatcher, current job, depth
#   bash tools/gpu_remote.sh log [N]                 # last N (default 30) lines of the workstation's gpu_queue.log
#   bash tools/gpu_remote.sh events [N]              # last N (default 200) START/DONE lines of that log + a `now: <epoch> <wall clock>` line
#   bash tools/gpu_remote.sh yield [--max-hours N]   # ask the local model to yield the 3090 to the queue (llm.sh yield)
#   bash tools/gpu_remote.sh provision <sha>         # run <sha>'s own pool_provision.sh --isolated on the workstation
#   bash tools/gpu_remote.sh --dry-run <subcommand> ...   # print the script that would run on the workstation, run nothing
#
# WHY (2026-09-30, docs/plans/2026-09-30-home-to-server-MIGRATION.md section 3). The project's home moved to a server
# with no GPU; the 3090, its queue (tools/gpu_queue.sh, one daemon, one singleton lock) and the local model stay on the
# workstation. gpu_add.sh/gpu_queue.sh are single-host by design, and nothing else creates a checkout of a server
# revision there, so this is the one bridge. It adds no queue of its own:
#   add        pushes HEAD's commit to the workstation's repo (as refs/server-jobs/<sha>), makes a per-job worktree of it under
#              ~/derisk-pool/revisions/<sha>-<epoch> WITHOUT research/findings/raw (as pool_provision.sh excludes it;
#              the results come back through `POOL_NODES=workstation tools/pool_sync.sh`, the pool-node layout),
#              symlinks .venv and data to the workstation's, then execs THAT worktree's tools/local_llm/gpu_add.sh
#              "cd <worktree> && <cmd>" -- so the guard runs at the job's own revision, into the workstation's queue.
#              One worktree per job, because gpu_add.sh refuses a checkout holding untracked outputs.
#   provision  pool_provision.sh needs rsync, the pool-node keys and the corpus; the workstation has all three.
# The job travels base64-encoded (the workstation's login shell is fish), like the pool dispatcher's JOB_B64.
# The revision travels by PUSH over this same ssh connection (2026-09-30 security re-review, CRITICAL): the workstation
# used to `git fetch server` with its own passphrase-free key to dant123@srv-host-0, a key that also opened a root
# shell on the server (passwordless sudo) to anything running as dant123 on the workstation, GPU jobs included. With
# the push, the workstation holds no credential to the server at all, and that key can leave its disk.
#
# Env: SIM_GPU_HOST (ssh host, default `workstation`), SIM_GPU_HOST_REPO (the checkout there, default
# $HOME/Projects/sim, expanded on the workstation).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${SIM_GPU_HOST:-workstation}
WREPO=${SIM_GPU_HOST_REPO:-'$HOME/Projects/sim'}
DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }

usage() { sed -n '4,11p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

# mkwt <sha> <remote worktree path>: the remote lines that make a clean, sparse (no research/findings/raw) worktree.
mkwt() {
  cat <<EOF
R="$WREPO"
W="$2"
git -C "\$R" worktree add -q --no-checkout --detach "\$W" $1
git -C "\$W" sparse-checkout set --no-cone '/*' '!/research/findings/raw/'
git -C "\$W" reset -q --hard
ln -s "\$R/.venv" "\$W/.venv"
ln -s "\$R/data" "\$W/data"
EOF
}

push_revision() {   # push_revision <full sha>: put the commit into the workstation's repo before the remote script uses it
  local url="$HOST:${WREPO#\$HOME/}"
  if [ "$DRY" = 1 ]; then echo "# git push $url $1:refs/server-jobs/$1"; return 0; fi
  GIT_SSH_COMMAND="ssh -o BatchMode=yes" git -C "$ROOT" push -q "$url" "$1:refs/server-jobs/$1" \
    || { echo "⛔ could not push revision $1 to $url" >&2; exit 1; }
}

run_remote() {   # stdin: the remote script
  if [ "$DRY" = 1 ]; then echo "# ssh -o BatchMode=yes $HOST bash -s <<'SCRIPT'"; cat; echo "SCRIPT"; return 0; fi
  ssh -o BatchMode=yes "$HOST" bash -s
}

cmd_add() {
  # --rev <sha> (2026-10-02, cloud Claude session 4): run the job at a pinned, already-committed revision instead of
  # HEAD -- a gate's pin (e.g. the Stage-1 gate's c238ff325), which predates this bridge, so its own tree has no
  # gpu_remote.sh to run from and moving a branch to it would move the gate's branch. The working tree is then
  # irrelevant (nothing uncommitted can reach the job), so only the revision is checked: it must be a commit, on a
  # branch (local or remote-tracking), and carry its own tools/local_llm/gpu_add.sh (the guard runs at the job's
  # revision, as for HEAD).
  local rev=""
  if [ "${1:-}" = "--rev" ]; then
    [[ "${2:-}" =~ ^[0-9a-f]{7,40}$ ]] || { echo "usage: gpu_remote.sh add --rev <commit sha> '<cmd>'" >&2; exit 2; }
    rev="$2"; shift 2
  fi
  local cmd="${1:-}"
  [ -n "$cmd" ] || { echo "usage: gpu_remote.sh add [--rev <sha>] '<cmd>'" >&2; exit 2; }
  local sha b64
  if [ -n "$rev" ]; then
    sha=$(git -C "$ROOT" rev-parse -q --verify "$rev^{commit}") || { echo "⛔ REFUSED: no such commit here: $rev" >&2; exit 2; }
    if [ -z "$(git -C "$ROOT" branch -a --contains "$sha" 2>/dev/null | grep -v 'HEAD detached')" ]; then
      echo "⛔ REFUSED: $sha is on no branch (local or remote-tracking): its results would name a revision nobody can check out." >&2
      exit 1
    fi
    if ! git -C "$ROOT" cat-file -e "$sha:tools/local_llm/gpu_add.sh" 2>/dev/null; then
      echo "⛔ REFUSED: $sha has no tools/local_llm/gpu_add.sh, so its job could not pass the guard at its own revision." >&2
      exit 1
    fi
  else
    # The job runs the COMMITTED revision, so a local code edit would silently not be in it. Results and coordination
    # state (research/findings, research/coordination, research/queue) are data the job does not run, so they may differ.
    if ! git -C "$ROOT" diff --quiet HEAD -- . ':!research/findings' ':!research/coordination' ':!research/queue'; then
      echo "⛔ REFUSED: $ROOT has uncommitted changes to tracked code; commit them first (the job runs HEAD):" >&2
      git -C "$ROOT" diff --stat HEAD -- . ':!research/findings' ':!research/coordination' ':!research/queue' | tail -5 >&2
      exit 1
    fi
    if [ -z "$(git -C "$ROOT" branch --contains HEAD 2>/dev/null | grep -v 'HEAD detached')" ]; then
      echo "⛔ REFUSED: HEAD is on no branch (a commit on no branch is eventually garbage-collected on the server, and the job's results would name a revision nobody can check out); commit on a branch first." >&2
      exit 1
    fi
    sha=$(git -C "$ROOT" rev-parse HEAD)
  fi
  b64=$(printf '%s' "$cmd" | base64 -w0)
  push_revision "$sha"
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
  local full   # the push needs the full id; the remote script keeps the sha as given (it names the revision dirs)
  full=$(git -C "$ROOT" rev-parse -q --verify "$sha^{commit}") || { echo "⛔ no such commit here: $sha" >&2; exit 2; }
  push_revision "$full"
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
  events)
    # READ-ONLY. The START / DONE lines only, found by grep ON the workstation: a job that writes hundreds of thousands of
    # output lines after its START pushes the START out of any `tail -n` window (tools/stall_watch.py needs it). The
    # trailing `now:` line is the workstation's epoch AND wall clock, so the reader can tell the zone the log's stamps are in.
    n="${2:-200}"; [[ "$n" =~ ^[0-9]+$ ]] || usage
    run_remote <<EOF
grep -aE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} (START|DONE\(rc=-?[0-9]+\)): ' "$WREPO/research/queue/gpu_queue.log" | tail -n $n
echo "now: \$(date +%s) \$(date '+%F %T')"
EOF
    ;;
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
