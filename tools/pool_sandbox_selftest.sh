#!/usr/bin/env bash
# tools/pool_sandbox_selftest.sh -- the LIVE check of the pool-job sandbox on one node (2026-09-30).
#
#   bash tools/pool_sandbox_selftest.sh <node> <revision-sha>                 # the isolation probe
#   bash tools/pool_sandbox_selftest.sh <node> <revision-sha> --job '<line>'  # also run one real job line sandboxed
#
# Runs through the EXACT launch tools/pool_autodispatch.sh uses for a sandboxed dispatch (--render-remote-command
# ... sandbox, i.e. tools/pool_job_sandbox.sh as the remote wrapper), NOT through the queue, so it works before the
# dispatcher is switched to POOL_SANDBOX=on and never jumps the live queue. <revision-sha> must already be provisioned
# on <node> (pool_provision.sh --isolated). It checks, from INSIDE the sandbox a job at that revision gets:
#   (b) no network: 192.168.0.10:445 (the server's SMB), 1.1.1.1:443 and DNS all fail;
#   (c) no privilege: NoNewPrivs=1 and `sudo -n true` fails;
#   (d) no writes outside the output area: the revision tree, /usr and the node's real HOME are unwritable, and
#       ~/.ssh and ~/derisk-pool/sim are not visible; only research/findings/raw/ is writable;
#   (a) the report lands on the node and tools/pool_sync.sh pulls it back here.
# With --job, the given line (e.g. a queued dev-seed-7 runner line, its `#checked:` text removed) runs the same way
# and its v2 status row must read rc=0; its outputs come back with the same pool_sync pull.
# Exit 0 only if every check passes. Read-only on the node except the files these jobs write.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${1:?usage: pool_sandbox_selftest.sh <node> <revision-sha> [--job '<line>']}"
REV="${2:?usage: pool_sandbox_selftest.sh <node> <revision-sha> [--job '<line>']}"
[[ $REV =~ ^[0-9a-f]{7,40}$ ]] || { echo "revision must be a hex sha" >&2; exit 2; }
EXTRA_JOB=""
[ "${3:-}" = "--job" ] && EXTRA_JOB="${4:?--job needs a line}"
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")
FAIL=0
say() { printf '%s\n' "$*"; }
check() { if [ "$2" = 1 ]; then say "  PASS  $1"; else say "  FAIL  $1"; FAIL=1; fi; }

say "== readiness ($NODE)"
probe=$(bash "$ROOT/tools/pool_autodispatch.sh" --sandbox-probe "$NODE")
say "  $probe"
case "$probe" in POOL_SANDBOX_READY*) ;; *) say "not ready -- see tools/pool_job_sandbox.sh (header: node setup)"; exit 1 ;; esac

# run_sandboxed <job> -- launch exactly as a sandboxed dispatch, then wait for its v2 row; prints its rc.
run_sandboxed() {
  local job="$1" b64 rendered rc="" i
  b64=$(printf '%s' "$job" | base64 -w0)
  rendered=$(bash "$ROOT/tools/pool_autodispatch.sh" --render-remote-command "$job" sandbox) || return 1
  ssh -n "${SSH_F[@]}" -o BatchMode=yes "$NODE" "$rendered" 2>/dev/null
  for i in $(seq 1 "${SELFTEST_WAIT_S:-1800}"); do
    rc=$(timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$NODE" \
         "$(printf 'grep -F -- %q ~/derisk-pool/sim/job_status.log 2>/dev/null | tail -1' "$b64")" 2>/dev/null \
         | awk -F'\t' -v b="$b64" '$1 == "v2" && $4 == b {print $3}')
    [ -n "$rc" ] && break
    sleep 1
  done
  printf '%s' "${rc:-timeout}"
}

TS=$(date +%Y%m%dT%H%M%S)
OUT_REL="research/findings/raw/_pool_sandbox_selftest/${NODE}_${TS}.json"
read -r -d '' PROBE <<'PY'
import json, os, socket, subprocess
r = {}
def conn(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close(); return "OPEN"
    except OSError as e:
        return "blocked: %s" % e.strerror
r["lan_smb"] = conn("192.168.0.10", 445)
r["internet"] = conn("1.1.1.1", 443)
try:
    socket.getaddrinfo("example.com", 443); r["dns"] = "RESOLVED"
except OSError as e:
    r["dns"] = "blocked: %s" % e
try:
    r["sudo_rc"] = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode
except OSError:
    r["sudo_rc"] = "absent"
r["nnp"] = [l.split()[1] for l in open("/proc/self/status") if l.startswith("NoNewPrivs")][0]
def can_write(p):
    try:
        open(p, "w").write("x"); os.unlink(p); return True
    except OSError:
        return False
r["write_revision"] = can_write("tools/.selftest_planted")
r["write_usr"] = can_write("/usr/.selftest_planted")
r["write_raw"] = can_write("research/findings/raw/.selftest_planted")
r["ssh_visible"] = os.path.exists(os.path.expanduser("~/.ssh"))
r["shared_checkout_visible"] = os.path.exists(os.path.expanduser("~/derisk-pool/sim"))
r["pids"] = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
r["pool_sandbox_env"] = os.environ.get("POOL_SANDBOX")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(r, open(OUT, "w"), indent=1)
PY
CODE="OUT = '$OUT_REL'
$PROBE"
JOB="cd ~/derisk-pool/revisions/$REV && touch ~/.selftest_home_write && .venv/bin/python -c $(printf '%q' "$CODE")"
# `touch ~/...` inside writes the sandbox's tmpfs HOME, never the node's; checked from outside below.

say "== isolation probe ($NODE, revision $REV)"
rc=$(run_sandboxed "$JOB")
check "probe job ran sandboxed (v2 rc=$rc)" "$([ "$rc" = 0 ] && echo 1)"
home_leak=$(timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes "$NODE" 'test -e ~/.selftest_home_write && echo LEAK' 2>/dev/null)
check "a write to ~ inside the sandbox did not reach the node's HOME" "$([ -z "$home_leak" ] && echo 1)"

if [ -n "$EXTRA_JOB" ]; then
  say "== real job ($NODE)"
  rc=$(run_sandboxed "$EXTRA_JOB")
  check "real job line exited rc=$rc" "$([ "$rc" = 0 ] && echo 1)"
fi

say "== pull (tools/pool_sync.sh --node $NODE)"
POOL_NODES="$NODE" bash "$ROOT/tools/pool_sync.sh" --node "$NODE" | tail -3
LOCAL="$ROOT/$OUT_REL"
check "(a) the probe report was pulled back to $OUT_REL" "$([ -f "$LOCAL" ] && echo 1)"
if [ -f "$LOCAL" ]; then
  eval "$(python3 - "$LOCAL" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ok = lambda b: "1" if b else "0"
print("NET=%s" % ok(r["lan_smb"].startswith("blocked") and r["internet"].startswith("blocked") and r["dns"].startswith("blocked")))
print("PRIV=%s" % ok(r["nnp"] == "1" and r["sudo_rc"] != 0))
print("RO=%s" % ok(not r["write_revision"] and not r["write_usr"] and r["write_raw"]))
print("HIDDEN=%s" % ok(not r["ssh_visible"] and not r["shared_checkout_visible"] and max(r["pids"]) < 20))
PY
)"
  say "  report: $(tr -d '\n' < "$LOCAL" | cut -c1-400)"
  check "(b) no network: LAN SMB, internet and DNS all blocked" "$NET"
  check "(c) no-new-privs set and sudo fails" "$PRIV"
  check "(d) revision and /usr read-only, research/findings/raw writable" "$RO"
  check "(d) ~/.ssh, ~/derisk-pool/sim and the node's other processes invisible" "$HIDDEN"
fi
[ "$FAIL" = 0 ] && say "SELFTEST PASS ($NODE)" || say "SELFTEST FAIL ($NODE)"
exit "$FAIL"
