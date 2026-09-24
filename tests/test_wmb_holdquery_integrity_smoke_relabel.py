"""Regression pin for the 2026-09-24 wm-binding-advanced hold-query INTEGRITY-SMOKE relabel.

Adversarial re-review (journal key v2:7512414c9a65ead57a205e7a2e166bcf476472d9dfbc61180205447d18d460df, label
'rereview:wm-binding') found: "I deleted `res["verdict"] = "integrity-smoke"` and `--selftest` still printed
VERDICT: PASS. tests/ has no test for it either." The hold-query probe (LB_WMB_HOLDQUERY_PROBE) is
PASS-BY-CONSTRUCTION: its reply is a host template whose only input is the buffer the lesion disables, so once
the route is reached a reply change is guaranteed. `measure_faculty`'s `_wmb_on` tail (research/runners/
load_bearing_fraction.py) is what keeps that predetermined result OUT of run()'s load-bearing numerator and
denominator: it unconditionally forces `res["verdict"] = "integrity-smoke"` and `res["load_bearing"] = None`
after the pre-registered adequacy gate, regardless of what the underlying treatment/control comparison read.
That force was covered by no test and no gate mutation, so a later edit could silently drop it and both
`--selftest` and every existing check would keep passing while a pass-by-construction probe re-entered the #1
metric as a real GO. This test builds the wmb tail with fully synthetic (non-brain) arms and asserts the
relabel's effect directly, at both the single-faculty level and through `run()`'s own exercised-set filter. It
FAILS the moment the three relabel lines (`integrity_smoke` / `integrity_smoke_verdict` / the `verdict` and
`load_bearing` overwrite) are removed, because the identical synthetic arms below make the PRE-relabel verdict
read "pass" (a real, non-null value that trips every assertion below)."""
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import load_bearing_fraction as lbf  # noqa: E402


def _fake_spawn(result):
    def spawn(env, turn_labels, out_path):
        return {label: dict(result) for label in turn_labels}
    return spawn


def _patch_wmb_on(monkeypatch):
    monkeypatch.setattr(lbf, "LB_WMB_HOLDQUERY", True)
    # Every arm (intact, intact-rebuild, lesion, and the wmb1 specificity controls) returns the SAME reply on
    # every turn label -- a clean null control and a "pass" (unchanged) treatment/lesion comparison BEFORE the
    # relabel runs. If the relabel is removed, this reads verdict="pass", load_bearing=False -- not None -- so
    # the assertions below catch the regression instead of vacuously passing on a build failure.
    monkeypatch.setattr(lbf, "_spawn_arm", _fake_spawn({"answer": "ok", "abstained": False}))


def test_wmb_holdquery_tail_forces_integrity_smoke_not_a_verdict(monkeypatch, tmp_path):
    _patch_wmb_on(monkeypatch)
    res = lbf.measure_faculty("wm-binding-advanced", str(tmp_path), repeats=1, intact_cache={}, seed=42)
    # The relabel: verdict must be the sentinel, never a countable "regressed"/"pass", and load_bearing must be
    # unset. Removing `res["verdict"] = "integrity-smoke"` / `res["load_bearing"] = None` makes this read
    # verdict="pass", load_bearing=False (identical synthetic arms) -- a real value that fails these asserts.
    assert res["verdict"] == "integrity-smoke", res
    assert res["load_bearing"] is None, res
    assert res.get("integrity_smoke") is True, res
    # The smoke's own (pre-relabel) outcome is still recorded, just not under the counted key.
    assert res.get("integrity_smoke_verdict") is not None, res


def test_wmb_holdquery_is_excluded_from_runs_load_bearing_fraction(monkeypatch, tmp_path):
    """Integration-level pin: run()'s own exercised filter (`verdict in ("regressed", "pass")`) must exclude
    the hold-query probe from n_exercised / n_load_bearing / load_bearing_fraction. wm-binding-advanced's
    FACULTY_LESIONS kind is "neural-lesion" (a coverable kind), so without the relabel this probe's guaranteed
    "pass" would count as a real not-load-bearing result in the denominator -- the over-credit the review named."""
    _patch_wmb_on(monkeypatch)
    report = lbf.run(out_dir=str(tmp_path), only=["wm-binding-advanced"], repeats=1, seed=42)
    per = report["per_faculty"][0]
    assert per["verdict"] == "integrity-smoke", per
    assert report["counts"]["n_coverable_env_lesion"] == 1, report["counts"]
    assert report["counts"]["n_exercised"] == 0, report["counts"]
    assert report["counts"]["n_load_bearing"] == 0, report["counts"]
    assert report["counts"]["n_not_load_bearing"] == 0, report["counts"]
    assert report["load_bearing_fraction"] is None, report
    assert "wm-binding-advanced" not in report["load_bearing_faculties"]
    assert "wm-binding-advanced" not in report["not_load_bearing_faculties"]
