#!/usr/bin/env bash
# tools/agent_container/bin/smoke_test.sh -- run via:
#   docker compose -f tools/agent_container/compose.yaml run --rm sim-agent smoke_test.sh
# Build-verification only (step 7 of the container build): does NOT start the long-running service, does not
# touch any other container/network/config on the server. Exits non-zero (and says why) on the first real
# failure; a check that depends on an owner-applied step not yet done (repo seeding, ssh keys) SKIPs with a
# clear reason instead of failing the whole smoke test.
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
  fail "claude --version did not run (see ~/entrypoint.log -- likely no network egress to claude.ai yet; see docs/AGENT_HARNESS_SETUP.md's optional egress-restriction step)"
fi

# 4. python imports: numpy (baked venv) always; the project package only if the repo volume is seeded (an
#    OWNER step, docs/AGENT_HARNESS_SETUP.md 6e -- not done yet on a fresh deploy, so this SKIPs rather than
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
  skip "import sim -- /work/sim is not seeded yet (owner step: docs/AGENT_HARNESS_SETUP.md 6e, clone from gitea or a git bundle)"
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
if ( : > "$AGENT_HOME/rw_write_test" ) 2>/dev/null; then pass "agent-home volume is writable"; rm -f "$AGENT_HOME/rw_write_test"; else fail "agent-home volume is not writable"; fi
if ( : > /work/sim/rw_write_test ) 2>/dev/null; then pass "repo volume is writable"; rm -f /work/sim/rw_write_test; else fail "repo volume is not writable"; fi

# 6. other tools present
for t in git ssh jq node npm python3; do
  if command -v "$t" >/dev/null 2>&1; then pass "$t present"; else fail "$t not found"; fi
done

echo "== smoke test done: $([ "$FAIL" -eq 0 ] && echo ALL PASS/SKIP || echo SOME FAILURES) =="
exit "$FAIL"
