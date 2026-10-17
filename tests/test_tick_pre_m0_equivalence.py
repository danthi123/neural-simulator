"""One-brain M0 re-review item 6 (2026-10-13): the restructured idle tick, on the battery clock (a caller-supplied
`now`, single-threaded), runs exactly the pre-M0 sequence. Pinned against a golden captured from the pre-M0 source
(commit 2e7392f12) by tests/_capture_tick_pre_m0_golden.py: same faculty calls in the same order (affect reads,
mouth builds, wanders with their IOR-biased gains, D5 / sleep / homeostasis steps), same inner-life records, same
wander and D5 budgets, same IOR adaptation, same ideation counter. Stub organs, no brain.
"""
import json
import os
import shutil
import subprocess

import pytest

from tests import _capture_tick_pre_m0_golden as G

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def tick_env(monkeypatch):
    for k, v in G.ENV.items():
        monkeypatch.setenv(k, v)
    for k in G.UNSET:
        monkeypatch.delenv(k, raising=False)


def _golden():
    with open(G.GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)["trace"]


def _diff(a, b):
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def test_battery_clock_tick_matches_the_pre_m0_golden(tick_env):
    golden = _golden()
    assert len(golden["log"]) > 50 and sum(len(v) for v in golden["inner"].values()) == 15   # the scenario has teeth
    now = G.scenario(G.load(os.path.join(_ROOT, "webapp", "continuous_engine.py"), "ce_branch"))
    assert now == golden, _diff(now, golden)


def test_the_golden_is_the_pre_m0_source(tick_env, tmp_path):
    """When the pre-M0 commit is reachable (a full clone), recompute the golden from it."""
    if shutil.which("git") is None:
        pytest.skip("no git")
    r = subprocess.run(["git", "show", "%s:webapp/continuous_engine.py" % G.PRE_M0_COMMIT], cwd=_ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("pre-M0 commit %s not in this clone" % G.PRE_M0_COMMIT)
    src = tmp_path / "ce_pre_m0.py"
    src.write_text(r.stdout, encoding="utf-8")
    assert G.scenario(G.load(str(src), "ce_pre_m0")) == _golden()
