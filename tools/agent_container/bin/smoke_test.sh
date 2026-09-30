#!/usr/bin/env bash
# tools/agent_container/bin/smoke_test.sh -- run ON THE SERVER via:
#   cd ~/apps && docker compose run --rm --no-deps sim-agent smoke_test.sh
# (tools/agent_container/update.sh runs exactly this against every freshly built image BEFORE swapping it in,
# and rolls the image tag back if it fails.) Build verification only: does NOT start the long-running service,
# does not touch any other container/network/config. Reports every check, exits non-zero if any FAILed; a check
# that depends on an owner step not yet done (repo seeding) SKIPs with a clear reason instead.
#
# Section 7 probes the network FROM INSIDE (the run container sits on the same br-simagent bridge, so
# sim-agent.nft applies to it): destinations that must be dropped, and the few that must work. It runs only
# when SIM_AGENT_REQUIRE_FIREWALL=1 (set by the server stack fragment); SMOKE_SKIP_NET=1 skips it explicitly.
# Every target is overridable (SMOKE_MUST_FAIL / SMOKE_MUST_PASS, space-separated host:port lists).
set -uo pipefail
FAIL=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAIL=1; }
skip() { echo "SKIP: $*"; }

echo "== smoke test: $(date -u '+%F %T') =="

# 1. non-root user
if [ "$(id -u)" != "0" ]; then pass "running as non-root (uid $(id -u), $(whoami))"; else fail "running as root"; fi

# 2. tmux present and a session was created by entrypoint.sh
if command -v tmux >/dev/null 2>&1; then pass "tmux present ($(tmux -V))"; else fail "tmux not found"; fi
if tmux has-session -t agent 2>/dev/null; then pass "tmux session 'agent' exists"; else fail "tmux session 'agent' was not created by entrypoint.sh"; fi

# 3. claude CLI (installed by entrypoint.sh's native installer into the home volume)
if command -v claude >/dev/null 2>&1 && ver=$(claude --version 2>&1); then
  pass "claude --version -> $ver"
else
  fail "claude --version did not run (see ~/entrypoint.log -- the first-boot native install needs HTTPS egress to claude.ai)"
fi

# 4. python imports: numpy (baked venv) always; the project package only if the repo bind mount is seeded (an
#    OWNER step in docs/AGENT_HARNESS_SETUP.md -- not done yet on a fresh deploy, so this SKIPs rather than
#    fails when /work/sim is empty).
if /opt/venv-sim/bin/python -c "import numpy; print(numpy.__version__)" >/tmp/numpy_ver 2>/tmp/numpy_err; then
  pass "numpy import ok (venv-sim, $(cat /tmp/numpy_ver))"
else
  fail "numpy import failed: $(cat /tmp/numpy_err)"
fi
if /opt/venv-rag/bin/python -c "import llama_index.core" >/tmp/rag_err 2>&1; then
  pass "llama_index.core import ok (venv-rag)"
else
  fail "llama_index.core import failed: $(cat /tmp/rag_err)"
fi
if [ -d /work/sim/sim ] && [ -n "$(ls -A /work/sim/sim 2>/dev/null)" ]; then
  if /opt/venv-sim/bin/python -c "import sys; sys.path.insert(0, '/work/sim'); import sim" >/tmp/sim_err 2>&1; then
    pass "import sim (project package) ok"
  else
    fail "import sim failed: $(cat /tmp/sim_err)"
  fi
else
  skip "import sim -- /work/sim is not seeded yet (owner step in docs/AGENT_HARNESS_SETUP.md: clone from gitea)"
fi

# 5. read-only root filesystem: a write outside the declared writable paths must fail.
if ( : > /opt/ro_write_test ) 2>/dev/null; then
  fail "root filesystem is WRITABLE (wrote /opt/ro_write_test) -- read_only: true is not taking effect"
  rm -f /opt/ro_write_test 2>/dev/null || true
else
  pass "root filesystem is read-only (write to /opt refused)"
fi
# ...and the writable paths ARE writable.
if ( : > /tmp/rw_write_test ) 2>/dev/null; then pass "/tmp is writable (tmpfs)"; rm -f /tmp/rw_write_test; else fail "/tmp is not writable"; fi
if ( : > "$AGENT_HOME/rw_write_test" ) 2>/dev/null; then pass "home bind mount is writable"; rm -f "$AGENT_HOME/rw_write_test"; else fail "home bind mount is not writable (owner: chown 61000:61000 \${DATA_ROOT}/sim-agent/home)"; fi
if ( : > /work/sim/rw_write_test ) 2>/dev/null; then pass "repo bind mount is writable"; rm -f /work/sim/rw_write_test; else fail "repo bind mount is not writable (owner: chown 61000:61000 \${DATA_ROOT}/sim-agent/repo)"; fi

# 6. other tools present
for t in git ssh jq node npm python3; do
  if command -v "$t" >/dev/null 2>&1; then pass "$t present"; else fail "$t not found"; fi
done

# 7. network isolation, from inside (see the header). A must-fail target PASSES only on a timeout (sim-agent.nft
#    drops silently); a connection OR a refusal means the packet reached something unfiltered.
tcp_probe() {   # tcp_probe <host> <port> <seconds> -> 0 connected, 124 timed out, other = refused/unreachable
  timeout "$3" bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$1" "$2" 2>/dev/null
}
if [ "${SMOKE_SKIP_NET:-0}" = "1" ] || [ "${SIM_AGENT_REQUIRE_FIREWALL:-0}" != "1" ]; then
  skip "network isolation probes (not on the server stack, or SMOKE_SKIP_NET=1)"
else
  # this host via the bridge gateway and via its LAN address (sshd, rpcbind, smb, nfs), a published port
  # (caddy 18443), another stack's subnet, the router, the workstation's SMB, a non-allowed resolver, plain HTTP
  MUST_FAIL="${SMOKE_MUST_FAIL:-172.30.30.1:22 172.30.30.1:445 192.168.0.10:22 192.168.0.10:111 192.168.0.10:445 192.168.0.10:2049 192.168.0.10:18443 172.20.0.1:22 192.168.0.1:443 192.168.0.68:445 8.8.8.8:53 1.1.1.1:80}"
  # allowed paths that do not depend on the workstation: the DNS resolvers and gitea ssh via this host's 2222.
  # The workstation's 22/8080 are allowed too, but it may be asleep or gaming when the weekly rebuild runs, and a
  # miss here rolls the image back -- server-stack/verify_isolation.sh (owner-run) checks those instead.
  MUST_PASS="${SMOKE_MUST_PASS:-192.168.0.1:53 1.1.1.1:53 192.168.0.10:2222}"
  for t in $MUST_FAIL; do
    tcp_probe "${t%:*}" "${t##*:}" 3; rc=$?
    case "$rc" in
      124) pass "blocked: $t (dropped)" ;;
      0)   fail "NOT blocked: $t ACCEPTED a connection -- sim-agent.nft is missing or wrong" ;;
      *)   fail "NOT dropped: $t was refused/unreachable (rc=$rc) -- the packet was answered unfiltered" ;;
    esac
  done
  for t in $MUST_PASS; do
    if tcp_probe "${t%:*}" "${t##*:}" 5; then pass "reachable: $t"; else fail "unreachable: $t (an allowed path is broken, or that service is down)"; fi
  done
  if getent hosts github.com >/dev/null 2>&1; then pass "DNS resolves github.com"; else fail "DNS lookup of github.com failed"; fi
  if tcp_probe github.com 443 5; then pass "HTTPS egress to github.com:443"; else fail "no HTTPS egress to github.com:443"; fi
fi

echo "== smoke test done: $([ "$FAIL" -eq 0 ] && echo ALL PASS/SKIP || echo SOME FAILURES) =="
exit "$FAIL"
