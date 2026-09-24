"""Both directions for .claude/hooks/require_memcap_for_brain_builds.py: an uncapped full-brain build is blocked, and
capped runs, job strings handed to the queues, light tests and text that merely mentions the runners pass."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".claude", "hooks",
                    "require_memcap_for_brain_builds.py")
_spec = importlib.util.spec_from_file_location("require_memcap_for_brain_builds", HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

BLOCK = [
    # the 2026-09-24 processes, verbatim shapes
    "/home/u/sim/.venv/bin/python -m pytest -q tests/test_webapp_server.py -k 'multiref or xedge or wave3' -p no:cacheprovider",
    ".venv/bin/python -m pytest -q tests/test_webapp_server.py::test_brain_chat_xedge_curiosity_d6_no_regression_on_ordinary_turns",
    "cd /home/u/wt && SIM_NO_PROVENANCE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_webapp_server.py tests/test_other.py",
    "pytest tests/test_webapp_server.py",
    ".venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns confirm",
    "SIM_BACKEND=numpy python3 -m research.runners.load_bearing_fraction --seeds 42",
    "bash tools/mem_ok.sh 12 3 && .venv/bin/python -m pytest tests/test_webapp_server.py",
]
ALLOW = [
    "bash tools/mem_ok.sh 12 3 && bash tools/memcap.sh 12 -- .venv/bin/python -m pytest -q tests/test_webapp_server.py",
    "bash tools/memcap.sh 8 -- .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {}",
    "systemd-run --user --scope -p MemoryMax=12G .venv/bin/python -m pytest tests/test_webapp_server.py",
    ".venv/bin/python -m pytest -q tests/test_webapp_server.py --collect-only",
    ".venv/bin/python -m research.runners.load_bearing_fraction --help",
    "timeout 60 .venv/bin/python -m research.runners.onebrain_regression_battery --selftest",
    "SIM_BACKEND=numpy .venv/bin/python -m research.runners.load_bearing_fraction --selftest",
    ".venv/bin/python -m pytest -q tests/test_guard_protected_delete.py tests/test_lbf_row_registry_hook.py",
    "bash tools/pool_queue.sh add --checked 'cd ~/sim && .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker'",
    "bash tools/gpu_queue.sh add \"cd /x && .venv/bin/python -m pytest tests/test_webapp_server.py\"",
    "git commit -q -F - <<'EOF'\nran .venv/bin/python -m pytest tests/test_webapp_server.py uncapped\nEOF",
    "grep -n onebrain_regression_battery research/runners/*.py",
    "echo 'python -m research.runners.load_bearing_fraction'",
    "cat research/runners/load_bearing_fraction.py | head",
]


@pytest.mark.parametrize("cmd", BLOCK)
def test_blocks_uncapped_full_brain_builds(cmd):
    assert hook.problems(cmd), cmd


@pytest.mark.parametrize("cmd", ALLOW)
def test_allows_capped_runs_and_text(cmd):
    assert hook.problems(cmd) == [], cmd


def test_process_exit_codes():
    for cmd, code in ((BLOCK[0], 2), (ALLOW[0], 0)):
        r = subprocess.run([sys.executable, HOOK], input=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}}),
                           text=True, capture_output=True)
        assert r.returncode == code, (cmd, r.stderr)
