"""Regression test for the FAILURE_LOG 2026-09-23 row that `gates/coverage` could not resolve to any enforcement
point (its gate cell named a source file + function, `research/runners/load_bearing_fraction.py measure_faculty`,
which the coverage gate's parser does not recognize as a module or a runnable check).

Incident: a brain that FAILED TO LOAD returns per-turn `{'_error': ...}` dicts, not `None` -- which `compare()`
then read as "fields absent in both arms" (verdict `not-exercised`), a silent false negative. An AWS D1 run
missing the `experiment` package read not-exercised on all 10 arms in seconds; only reading the raw arm JSON
exposed it. The fix: any `_error` in an intact/lesion arm now yields verdict `arm-build-failed`, with the error
text recorded on the result.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")


def test_arm_error_surfaces_as_build_failure_not_not_exercised(monkeypatch):
    from research.runners import load_bearing_fraction as LB

    def _fake_spawn_arm(env, turn_labels, out_path):
        # A brain that failed to load: every turn in the group carries an `_error` dict, never None.
        return {t: {"_error": "failed to load brain ... No module named experiment"} for t in turn_labels}

    monkeypatch.setattr(LB, "_spawn_arm", _fake_spawn_arm)
    res = LB.measure_faculty("curiosity-followup", out_dir="/tmp", intact_cache={})
    assert res["verdict"] == "arm-build-failed"
    assert "No module named experiment" in res["arm_error"]


def test_a_clean_arm_still_reads_its_ordinary_verdict_not_build_failed(monkeypatch):
    """Guard against the fix being over-broad: an arm with no `_error` anywhere must NOT be misread as a build
    failure just because the check now runs."""
    from research.runners import load_bearing_fraction as LB

    calls = {"n": 0}

    def _fake_spawn_arm(env, turn_labels, out_path):
        calls["n"] += 1
        # call order is intact_a, intact_b, THEN lesioned (see measure_faculty) -- only the 3rd (lesion) silences.
        curious = calls["n"] != 3
        return {t: {"abstained": True, "curiosity": {"curious": curious, "on": True}} for t in turn_labels}

    monkeypatch.setattr(LB, "_spawn_arm", _fake_spawn_arm)
    res = LB.measure_faculty("curiosity-followup", out_dir="/tmp", intact_cache={})
    assert res["verdict"] != "arm-build-failed"
    assert res.get("arm_error") is None
