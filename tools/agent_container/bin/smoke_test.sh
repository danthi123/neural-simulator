#!/bin/bash
# tools/agent_container/bin/smoke_test.sh -- run ON THE SERVER via:
#   cd ~/apps && docker compose run --rm --no-deps sim-agent smoke_test.sh
# (tools/agent_container/update.sh runs exactly this against every freshly built image BEFORE swapping it in,
# and rolls the image tag back if it fails.) Build verification only: does NOT start the long-running service,
# does not touch any other container/network/config. Reports every check, exits non-zero if any FAILed; a check
# that depends on an owner step not yet done (repo seeding) SKIPs with a clear reason instead.
#
# FUNCTIONAL checks only. This container mounts the model-writable home, so nothing here is evidence about
# isolation: the network probes run from a pristine container instead (bin/net_probe.sh via
# server-stack/probe_from_pristine.sh, from update.sh and verify_isolation.sh).
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

# 3. claude CLI (installed by entrypoint.sh's native installer into the home bind mount). Called by its install
#    path: the image PATH deliberately excludes the writable home. A FUNCTIONAL check only -- that binary is
#    model-writable, so this line proves the install runs, never anything about isolation.
if [ -x "$AGENT_HOME/.local/bin/claude" ] && ver=$("$AGENT_HOME/.local/bin/claude" --version 2>&1); then
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

# 7. network isolation is NOT probed here (2026-09-30 security review, HIGH #1): this container mounts the
#    model-writable home, so a verdict computed in it is the model's to choose. update.sh probes from a pristine
#    container every cycle (server-stack/probe_from_pristine.sh -> bin/net_probe.sh), and so does the owner's
#    verify_isolation.sh.
skip "network isolation (probed from a pristine container by update.sh and verify_isolation.sh, not from here)"

echo "== smoke test done: $([ "$FAIL" -eq 0 ] && echo ALL PASS/SKIP || echo SOME FAILURES) =="
exit "$FAIL"
