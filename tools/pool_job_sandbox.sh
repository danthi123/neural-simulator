#!/usr/bin/env bash
# tools/pool_job_sandbox.sh -- the POOL-NODE half of a sandboxed pool job (2026-09-30).
#
# WHY. The 2026-09-30 Fable security review of the sim-agent harness found that tools/pool_autodispatch.sh ran every
# pool job over ssh as the node's login user -- passwordless sudo, full network -- so a job line written by an agent
# (the gateway's `pool-submit`, or the workstation's local-model driver running `tools/pool_queue.sh add`) was a LAN
# foothold: the SMB guest shares, rpcbind/NFS, upsd and the published ports on the server all trust the LAN. This
# file is the recipe tools/agent_gateway_submit.py's build_bwrap_argv already applies to gpu-submit (bubblewrap
# built FROM EMPTY, --unshare-all, the checkout read-only, one writable output area), applied to pool jobs.
#
# HOW IT RUNS. Never on the workstation in production. tools/pool_autodispatch.sh base64-encodes THIS FILE into the
# remote launch command -- the same WRAPPER_B64 slot its legacy four-line wrapper uses -- and the node runs it with
# `bash` (script on stdin), as its login user, OUTSIDE the sandbox, from ~/derisk-pool/sim. Everything is inside
# functions and `main` is called on the last line, so bash has parsed the whole script before any child runs.
#
#   POOL_SANDBOX_ACTION=run    (default) decode JOB_B64, build the sandbox, run the job in it, write the SAME
#                              `v2<TAB>epoch<TAB>rc<TAB>JOB_B64` row to job_status.log the legacy wrapper writes
#                              (check_fast_fail, workflow_check and pool_queue's liveness scan read it unchanged)
#   POOL_SANDBOX_ACTION=probe  print `POOL_SANDBOX_READY bwrap=<version>` if this node can start the sandbox (bwrap
#                              installed, unprivileged user namespaces allowed, no-new-privs set inside), else
#                              `POOL_SANDBOX_NOT_READY reason=<...>`. The dispatcher's readiness check.
#
# WHAT THE JOB SEES (built from an empty root; nothing else exists):
#   - namespaces: --unshare-all (net, pid, ipc, uts, cgroup, user). NO NETWORK AT ALL -- only a loopback device --
#     so no LAN, no internet, no DNS; no view of other processes; no abstract unix sockets (they belong to the net
#     namespace); /run does not exist, so neither the systemd user bus nor any other host socket is reachable.
#   - no new privileges: bwrap sets PR_SET_NO_NEW_PRIVS unconditionally, so setuid binaries (sudo, su, pkexec) cannot
#     raise privileges; the job's uid is the node user's, mapped, with no capabilities; /etc/sudoers is not there.
#   - --clearenv + an allow-list: PATH, HOME, USER/LOGNAME, LANG, SHELL, TMPDIR, the dispatcher's POOL_JOB_ID /
#     POOL_JOB_MEM_GB and thread-count variables (so provenance and the node's accounting read what they read today),
#     POOL_SANDBOX=1 (recorded in every provenance sidecar's env block, since it starts with POOL_).
#   - read-only: /usr (+ /bin, /sbin, /lib* as the node's own merged-usr symlinks), six /etc entries (localtime,
#     passwd, group, nsswitch.conf, ld.so.cache, alternatives -- Ubuntu's /usr/bin/awk is an alternatives link),
#     a fresh /proc (this pid namespace only) and /dev (null/zero/random/tty only), each revision the job line pins
#     (~/derisk-pool/revisions/<sha>: source, .venv, data/corpus), ~/Projects/sim-data (the LTM bundles), and the
#     shared ~/derisk-pool/sim checkout ONLY for a line that pins no revision.
#   - writable: a fresh tmpfs /tmp, a fresh tmpfs over the node's HOME (so ~/.ssh, ~/.config, every other job's
#     files and every other revision are simply absent -- HOME keeps its real path so `cd ~/derisk-pool/...` and
#     Path.home()/Projects/sim-data resolve as today), and ONLY the output area:
#       owner line:  <revision>/research/findings/raw  (runner --out paths + the provenance door's _provenance/)
#       agent line:  <revision>/research/findings/raw/_agent_jobs/<id>  (a line carrying AGENT_JOB_SRC=sim-agent,
#                    i.e. built by tools/agent_gateway_submit.py; <id> must be the gateway's sha12-hex8 shape)
#     Everything else in the revision is read-only, so a job cannot edit the source other jobs at that revision run.
#
# MEMCAP / MEM_OK INSIDE (pinned revisions carry their own, unmodified tools/memcap.sh and tools/mem_ok.sh). memcap
# needs `systemd-run --user`, which cannot reach the user manager from inside. So when the line calls memcap.sh N
# (the largest N if several), THIS wrapper puts the whole sandbox in the same transient scope memcap would have made
# (MemoryMax=N G, MemoryHigh 90 %, MemorySwapMax from --swap, default 0), from outside, and ro-binds two shims at
# /opt/pool-sandbox/bin, first on PATH: `systemd-run --user --scope ... -- cmd` execs cmd (the outer scope is the
# cap; it refuses with memcap's own rc 3 if there is no outer scope), and `systemctl --user` answers mem_ok's three
# queries from a snapshot of the node's running run-*.scope units taken just before launch. No outer scope possible
# (systemd --user unavailable) -> refused with rc 3, exactly what memcap itself does today.
# An AGENT line always gets the outer scope (MemoryMax=8G unless it declares memcap, TasksMax=512) under a 12 h
# wall-clock `timeout`, like gpu-submit's resource bounds.
#
# REFUSALS write `[pool-sandbox] REFUSED ...` to autodispatch.out and a v2 row with rc 126 (rc 3 for the memcap
# case), and never fall back to running the job unsandboxed. tools/pool_autodispatch.sh's check_fast_fail reports
# rc 126 loudly.
set -uo pipefail

SB_BWRAP="${POOL_SANDBOX_BWRAP:-bwrap}"   # TEST-ONLY override; the dispatcher's remote command never sets it
SB_MOUNT=/opt/pool-sandbox                # where the per-job shim directory appears inside the sandbox
SB_AGENT_MEM_GB=8                         # agent lines: default MemoryMax when the line declares no memcap
SB_AGENT_TASKS_MAX=512
SB_AGENT_TIMEOUT=12h
SB_RC_REFUSED=126

SB_ARGV=(); SB_OUTER=(); SB_DIR=""; SB_CHDIR=""; SB_AGENT=0; SB_AGENT_ID=""; SB_WHY=""; SB_RC=0; SB_WRITABLE=()
SB_SKIPPED=()

sb_refuse() { SB_WHY="$2"; SB_RC="$1"; return 1; }

# sb_safe_dir <dir> -- the directory exists, and neither it nor any parent up to (excluding) / is a symlink. Paths
# bound into the sandbox are checked this way so state planted by an earlier, unsandboxed job cannot redirect a bind.
sb_safe_dir() {
  local p="$1"
  [ -d "$p" ] || return 1
  while [ -n "$p" ] && [ "$p" != "/" ]; do
    [ -L "$p" ] && return 1
    p="${p%/*}"
  done
  return 0
}

# sb_base_argv <home> -- everything that does not depend on the job.
sb_base_argv() {
  local home="$1" d f user host
  user=$(id -un 2>/dev/null) || user="uid$(id -u)"
  host=$(hostname 2>/dev/null) || host="pool-node"
  SB_ARGV=(
    --unshare-all --die-with-parent --new-session
    --hostname "$host"
    --clearenv
    --setenv PATH "$SB_MOUNT/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"
    --setenv HOME "$home" --setenv USER "$user" --setenv LOGNAME "$user"
    --setenv LANG "${LANG:-C.UTF-8}" --setenv SHELL /bin/bash --setenv TMPDIR /tmp
    --setenv PYTHONDONTWRITEBYTECODE 1
    --setenv POOL_SANDBOX 1
  )
  # The dispatcher's per-job variables (tools/pool_autodispatch.sh remote_launch_command): only a plain token
  # charset is passed through.
  for d in POOL_JOB_ID POOL_JOB_MEM_GB OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS; do
    f="${!d:-}"
    [[ -n $f && $f =~ ^[A-Za-z0-9._-]+$ ]] && SB_ARGV+=(--setenv "$d" "$f")
  done
  SB_ARGV+=(--proc /proc --dev /dev --tmpfs /tmp --ro-bind /usr /usr)
  for d in bin sbin lib lib32 lib64 libx32; do
    if [ -L "/$d" ]; then SB_ARGV+=(--symlink "$(readlink "/$d")" "/$d")
    elif [ -d "/$d" ]; then SB_ARGV+=(--ro-bind "/$d" "/$d"); fi
  done
  for f in localtime passwd group nsswitch.conf ld.so.cache alternatives; do
    SB_ARGV+=(--ro-bind-try "/etc/$f" "/etc/$f")
  done
  # A fresh, empty HOME at the node's real HOME path; the job's read-only and writable areas are bound under it.
  SB_ARGV+=(--tmpfs "$home")
}

# sb_bind_checkout <dir> -- read-only checkout + its writable output area (owner: raw/, agent: raw/_agent_jobs/<id>).
sb_bind_checkout() {
  local dir="$1" raw="$1/research/findings/raw" out
  sb_safe_dir "$dir" || sb_refuse "$SB_RC_REFUSED" "checkout $dir is missing or reached through a symlink" || return 1
  [ -d "$raw" ] || mkdir -p -- "$raw" 2>/dev/null
  sb_safe_dir "$raw" || sb_refuse "$SB_RC_REFUSED" "$raw is missing or reached through a symlink" || return 1
  SB_ARGV+=(--ro-bind "$dir" "$dir")
  if [ "$SB_AGENT" = 1 ]; then
    out="$raw/_agent_jobs/$SB_AGENT_ID"
    [ -d "$raw/_agent_jobs" ] || mkdir -- "$raw/_agent_jobs" 2>/dev/null
    [ -d "$out" ] || mkdir -- "$out" 2>/dev/null
    sb_safe_dir "$out" || sb_refuse "$SB_RC_REFUSED" "agent output dir $out is missing or reached through a symlink" \
      || return 1
    [ "$(realpath -e -- "$out" 2>/dev/null)" = "$(realpath -e -- "$raw" 2>/dev/null)/_agent_jobs/$SB_AGENT_ID" ] \
      || sb_refuse "$SB_RC_REFUSED" "agent output dir $out does not resolve inside $raw" || return 1
  else
    out="$raw"
  fi
  SB_ARGV+=(--bind "$out" "$out")
  SB_WRITABLE+=("$out")
  [ -n "$SB_CHDIR" ] || SB_CHDIR="$dir"
}

# sb_memcap <job> -- "<max_gb> <swap_gb>" over every `memcap.sh N [--swap S]` in the line, or nothing.
sb_memcap() {
  printf '%s\n' "$1" | grep -oE 'memcap\.sh[[:space:]]+[0-9]+([[:space:]]+--swap[[:space:]]+[0-9]+)?' \
    | awk '{n = $2 + 0; if (n > m) m = n; if ($3 == "--swap" && $4 + 0 > s) s = $4 + 0}
           END {if (m > 0) print m, s + 0}'
}

sb_systemd_user_ok() {
  if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then export XDG_RUNTIME_DIR="/run/user/$(id -u)"; fi
  systemctl --user is-system-running >/dev/null 2>&1 || systemd-run --user --scope --quiet true >/dev/null 2>&1
}

# sb_write_shims -- the per-job directory ro-bound at $SB_MOUNT: bin/systemd-run, bin/systemctl, scopes.tsv.
sb_write_shims() {
  local u
  mkdir -p -- "$1/bin" || return 1
  # mem_ok.sh's committed-but-unused headroom: the same query it makes, answered from outside, now.
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user list-units --type=scope --state=running --no-legend 'run-*.scope' 2>/dev/null </dev/null \
      | awk '{print $1}' | while read -r u; do
          printf '%s\t%s\t%s\n' "$u" "$(systemctl --user show "$u" -p MemoryMax --value 2>/dev/null </dev/null)" \
                                     "$(systemctl --user show "$u" -p MemoryCurrent --value 2>/dev/null </dev/null)"
        done > "$1/scopes.tsv"
  else
    : > "$1/scopes.tsv"
  fi
  cat > "$1/bin/systemd-run" <<'SHIM'
#!/bin/bash
# pool-sandbox shim (tools/pool_job_sandbox.sh): memcap.sh's `systemd-run --user --scope -p ... -- cmd` inside the
# sandbox. The memory cap is the transient scope the node wrapper put AROUND the whole sandbox.
user=0; scope=0
while [ $# -gt 0 ]; do
  case "$1" in
    --user) user=1 ;;
    --scope) scope=1 ;;
    --quiet|-q) ;;
    -p|--property) shift ;;
    -p*|--property=*|--unit=*|--description=*) ;;
    --) shift; break ;;
    *) echo "[pool-sandbox] systemd-run: option '$1' is not available inside the pool sandbox" >&2; exit 1 ;;
  esac
  shift
done
if [ "$user" != 1 ] || [ "$scope" != 1 ]; then
  echo "[pool-sandbox] systemd-run: only --user --scope (memcap.sh) is available inside the pool sandbox" >&2; exit 1
fi
if [ "${POOL_SANDBOX_MEMCAP_ENFORCED:-0}" != 1 ]; then
  echo "[pool-sandbox] ⛔ a memory cap was requested but this sandbox has no outer scope -- refusing to run uncapped" >&2
  exit 3
fi
[ $# -gt 0 ] || exit 0
echo "[pool-sandbox] memcap: enforced by the scope around the sandbox (MemoryMax=${POOL_SANDBOX_OUTER_MEMORY_MAX:-?})" >&2
exec "$@"
SHIM
  cat > "$1/bin/systemctl" <<'SHIM'
#!/bin/bash
# pool-sandbox shim (tools/pool_job_sandbox.sh): the three `systemctl --user` queries memcap.sh/mem_ok.sh make,
# answered from the snapshot of the node's running run-*.scope units the wrapper took just before launch.
snap=/opt/pool-sandbox/scopes.tsv
[ "${1:-}" = "--user" ] || { echo "[pool-sandbox] systemctl is not available inside the pool sandbox" >&2; exit 1; }
shift
case "${1:-}" in
  is-system-running)
    if [ "${POOL_SANDBOX_MEMCAP_ENFORCED:-0}" = 1 ]; then echo running; exit 0; fi
    echo offline; exit 1 ;;
  list-units)
    awk -F'\t' 'NF {print $1 " loaded active running pool-sandbox-snapshot"}' "$snap" 2>/dev/null; exit 0 ;;
  show)
    unit="${2:-}"; shift 2 2>/dev/null; prop=""
    while [ $# -gt 0 ]; do case "$1" in -p) prop="${2:-}"; shift 2 ;; -p*) prop="${1#-p}"; shift ;; *) shift ;; esac; done
    awk -F'\t' -v u="$unit" -v p="$prop" '$1 == u {print (p == "MemoryMax" ? $2 : (p == "MemoryCurrent" ? $3 : "")); f = 1}
                                          END {exit !f}' "$snap" 2>/dev/null
    exit ;;
esac
echo "[pool-sandbox] systemctl --user ${1:-}: not available inside the pool sandbox" >&2
exit 1
SHIM
  chmod 0555 "$1/bin/systemd-run" "$1/bin/systemctl"
}

# sb_prepare <home> <job> -- fills SB_OUTER + SB_ARGV (and SB_DIR); returns 1 with SB_RC/SB_WHY on refusal.
sb_prepare() {
  local home="$1" job="$2" pool="$1/derisk-pool" revs r n ids cap swap high
  SB_AGENT=0; SB_AGENT_ID=""; SB_CHDIR=""; SB_WRITABLE=(); SB_OUTER=(); SB_DIR=""
  command -v "$SB_BWRAP" >/dev/null 2>&1 \
    || sb_refuse "$SB_RC_REFUSED" "bubblewrap ('bwrap') is not installed on this node" || return 1
  sb_safe_dir "$home" || sb_refuse "$SB_RC_REFUSED" "HOME ($home) is not a plain directory" || return 1
  case "$job" in *AGENT_JOB_SRC=sim-agent*) SB_AGENT=1 ;; esac
  revs=$(printf '%s\n' "$job" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | sed 's#.*/##' | sort -u)
  n=$(printf '%s\n' "$revs" | grep -c .)
  if [ "$SB_AGENT" = 1 ]; then
    [ "$n" = 1 ] || sb_refuse "$SB_RC_REFUSED" "an agent line must pin exactly one revision (found $n)" || return 1
    ids=$(printf '%s\n' "$job" | grep -oE 'research/findings/raw/_agent_jobs/[^[:space:]/;&|]+' | sed 's#.*/##' | sort -u)
    [ "$(printf '%s\n' "$ids" | grep -c .)" = 1 ] && [[ $ids =~ ^[0-9a-f]{12}-[0-9a-f]{8}$ ]] \
      || sb_refuse "$SB_RC_REFUSED" "an agent line must name exactly one research/findings/raw/_agent_jobs/<sha12>-<hex8> directory" \
      || return 1
    SB_AGENT_ID="$ids"
  fi
  sb_base_argv "$home"
  if [ "$n" = 0 ]; then
    sb_bind_checkout "$pool/sim" || return 1
  else
    # Every pinned revision that exists on this node. A mention of one that does not (e.g. inside the --checked
    # reason, which pop_job prefixes as POOL_CHECKED_REASON=...) is skipped, not fatal; an agent line's single
    # revision must exist. If none exists the line cannot run at all.
    SB_SKIPPED=()
    for r in $revs; do
      if [ "$SB_AGENT" != 1 ] && [ ! -e "$pool/revisions/$r" ]; then SB_SKIPPED+=("$r"); continue; fi
      sb_bind_checkout "$pool/revisions/$r" || return 1
    done
    [ -n "$SB_CHDIR" ] || sb_refuse "$SB_RC_REFUSED" "none of the revisions this line pins is provisioned here ($revs)" \
      || return 1
  fi
  SB_ARGV+=(--ro-bind-try "$home/Projects/sim-data" "$home/Projects/sim-data")
  # Outer resource scope: the line's own memcap, and always for an agent line.
  read -r cap swap <<< "$(sb_memcap "$job")"
  if [ "$SB_AGENT" = 1 ] && [ -z "${cap:-}" ]; then cap=$SB_AGENT_MEM_GB; swap=0; fi
  if [ -n "${cap:-}" ]; then
    sb_systemd_user_ok || sb_refuse 3 "a memory cap is required (memcap.sh ${cap} or an agent line) but systemd --user is unavailable -- refusing to run uncapped, as memcap.sh itself does" || return 1
    high=$(( cap * 9 / 10 )); [ "$high" -lt 1 ] && high=1
    SB_OUTER=(systemd-run --user --scope --quiet -p "MemoryMax=${cap}G" -p "MemoryHigh=${high}G" -p "MemorySwapMax=${swap:-0}G")
    [ "$SB_AGENT" = 1 ] && SB_OUTER+=(-p "TasksMax=$SB_AGENT_TASKS_MAX")
    SB_OUTER+=(--)
    SB_ARGV+=(--setenv POOL_SANDBOX_MEMCAP_ENFORCED 1 --setenv POOL_SANDBOX_OUTER_MEMORY_MAX "${cap}G")
  fi
  [ "$SB_AGENT" = 1 ] && SB_OUTER=(timeout --kill-after=30s "$SB_AGENT_TIMEOUT" "${SB_OUTER[@]}")
  mkdir -p -- "$pool/.sandbox-jobs" 2>/dev/null
  SB_DIR=$(mktemp -d "$pool/.sandbox-jobs/job.XXXXXXXX" 2>/dev/null) \
    || sb_refuse "$SB_RC_REFUSED" "could not create the per-job shim directory under $pool/.sandbox-jobs" || return 1
  sb_write_shims "$SB_DIR" || sb_refuse "$SB_RC_REFUSED" "could not write the sandbox shims" || return 1
  SB_ARGV+=(--ro-bind "$SB_DIR" "$SB_MOUNT")
  SB_ARGV+=(--chdir "$SB_CHDIR")
  return 0
}

sb_run() {
  local job rc
  job=$(printf '%s' "${JOB_B64:-}" | base64 -d) || exit 125
  if sb_prepare "$HOME" "$job"; then
    {
      printf '[pool-sandbox] sandboxed: cwd=%s writable=%s agent=%s outer=%s skipped-revisions=%s\n' \
        "$SB_CHDIR" "${SB_WRITABLE[*]}" "$SB_AGENT" "${SB_OUTER[*]:-none}" "${SB_SKIPPED[*]:-none}"
      "${SB_OUTER[@]}" "$SB_BWRAP" "${SB_ARGV[@]}" -- /bin/bash -c "$job" </dev/null
    } > autodispatch.out 2>&1
    rc=$?
  else
    rc=$SB_RC
    printf '[pool-sandbox] ⛔ REFUSED (rc=%s, job NOT run): %s\n' "$rc" "$SB_WHY" > autodispatch.out
  fi
  [ -n "$SB_DIR" ] && rm -rf -- "$SB_DIR"
  printf "v2\t%s\t%s\t%s\n" "$(date +%s)" "$rc" "$JOB_B64" >> job_status.log
}

sb_probe() {
  local out ver
  command -v "$SB_BWRAP" >/dev/null 2>&1 || { echo "POOL_SANDBOX_NOT_READY reason=bwrap-not-installed"; return 0; }
  ver=$("$SB_BWRAP" --version 2>/dev/null | awk '{print $NF; exit}')
  sb_base_argv "$HOME"
  out=$("$SB_BWRAP" "${SB_ARGV[@]}" --chdir / -- /bin/sh -c \
        'grep -q "^NoNewPrivs:[[:space:]]*1" /proc/self/status && echo SANDBOX_INSIDE_OK' </dev/null 2>&1)
  case "$out" in
    *SANDBOX_INSIDE_OK*) echo "POOL_SANDBOX_READY bwrap=${ver:-unknown}" ;;
    *) echo "POOL_SANDBOX_NOT_READY reason=bwrap-failed: $(printf '%s' "$out" | head -1 | tr -cd '[:print:]' | cut -c1-160)" ;;
  esac
}

main() {
  case "${POOL_SANDBOX_ACTION:-run}" in
    run) sb_run ;;
    probe) sb_probe ;;
    *) echo "pool_job_sandbox.sh: unknown POOL_SANDBOX_ACTION '${POOL_SANDBOX_ACTION}'" >&2; exit 2 ;;
  esac
}
main; exit $?
