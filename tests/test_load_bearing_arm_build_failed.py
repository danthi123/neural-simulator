"""Regression pin for the 2026-09-23 silent false negative: a brain that FAILED TO LOAD returns per-turn
{'_error': ...} arm dicts (not None), which compared as "fields absent in both arms" and read NOT-EXERCISED. An AWS run
missing the `experiment` package read not-exercised on all 10 arms in seconds. measure_faculty must now report
arm-build-failed and record the error. The failing direction is tested first."""
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import load_bearing_fraction as lbf  # noqa: E402

_ERR = {"_error": "HTTPException: 400: failed to load brain 'tiny-demo': ModuleNotFoundError: No module named 'experiment'"}


def _fake_spawn(result):
    def spawn(env, turn_labels, out_path):
        return {label: dict(result) for label in turn_labels}
    return spawn


def test_errored_arms_read_arm_build_failed_not_not_exercised(monkeypatch, tmp_path):
    monkeypatch.setattr(lbf, "_spawn_arm", _fake_spawn(_ERR))
    res = lbf.measure_faculty("affect-marker-spiking-wta", str(tmp_path), repeats=1, intact_cache={}, seed=42)
    assert res["verdict"] == "arm-build-failed", res
    assert "experiment" in res.get("arm_error", "")


def test_clean_arms_are_not_misclassified_as_build_failures(monkeypatch, tmp_path):
    # Identical clean arms: whatever the verdict is, it must NOT be a build failure.
    monkeypatch.setattr(lbf, "_spawn_arm", _fake_spawn({"answer": "ok", "abstained": False}))
    res = lbf.measure_faculty("affect-marker-spiking-wta", str(tmp_path), repeats=1, intact_cache={}, seed=42)
    assert res["verdict"] != "arm-build-failed", res
    assert "arm_error" not in res
