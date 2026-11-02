"""attribution-required REGRESSION mode: a MODIFIED runner that drops its tools.lab attribution call is caught.

The incident (2026-09-23): the D3 affect-pool verify runner was clean at 0f1f35ff1 (it called `lever` and
`attributable_to`); the v2 gate amendment cd5288018 removed both, and pre-commit let it through because the gate
only saw ADDED files.
"""
import subprocess

import pytest

from tools.gates import attribution_required as g

_RUNNER = "research/runners/_onebrain_affect_pool_verify.py"


def _show(rev):
    try:
        return subprocess.run(["git", "show", f"{rev}:{_RUNNER}"], capture_output=True, text=True,
                              check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"{rev} not in this clone")


def test_selftest_fails_in_the_failing_direction():
    assert g.selftest() == []


def test_the_real_incident_is_caught_and_the_fix_is_clean():
    clean, dropped = _show("0f1f35ff1"), _show("cd5288018")
    assert g._regressed(_RUNNER, clean, dropped), "the cd5288018 modification must be flagged"
    with open(_RUNNER) as f:
        head = f.read()
    assert g._regressed(_RUNNER, dropped, head) == []
    assert g._check_one(_RUNNER, head) == []


def test_standalone_mode_still_checks_nothing():
    assert g.check(None) == []
