"""Pin for AMENDMENT C (2026-09-24) to research/findings/2026-09-24-wm-binding-ordinary-content-probe-
PREREGISTRATION.md.

AMENDMENT B rescoped the PROSE a T=true/"regressed" GO on the ordinary-content probe (LB_WMB_CONTENT_PROBE) must
carry: it shows the organ's own w_k->w_k recurrence shapes comprehension's read of a host-reinjected, host-timed,
host-targeted drive into a POSITIONAL focus pool (`CAND_POOLS[0]`) -- NOT that the organ's held referent CONTENT
reaches an ordinary reply. But it left the COUNTING unchanged: `measure_wmb_content` still reported
`res["faculty"] = "wm-binding-advanced"`, so a GO would land in THAT faculty's row of `load_bearing_fraction`'s
numerator and `load_bearing_faculties` list -- silently over-crediting the held-content-binding claim the prose
now explicitly disclaims. AMENDMENT C fixes the counting to match the prose: the result is reported under the
distinct key `_WMC_FACULTY_KEY` ("wm-binding-recurrence-drive"), so it can still enter the harness's aggregate
fraction (the lesion is still `FACULTY_LESIONS["wm-binding-advanced"]`'s neural-lesion kind -- this is a
reporting-label fix, not a re-scoring) without ever appearing under "wm-binding-advanced" itself.

This test builds fully synthetic (non-brain) arms that reproduce a genuine 6/6-style GO (`_wmc_gate` reads
"regressed"/True on both content sessions, exactly the `_wmc_selftest_checks` "wmc gate: both contents change"
case) and asserts the reported identity directly, at both the single-faculty level and through `run()`'s own
`load_bearing_faculties` list. It fails the moment `_WMC_FACULTY_KEY` (or the `res["faculty"]` assignment in
`measure_wmb_content`) is reverted to "wm-binding-advanced", because the synthetic GO below is real (not
UNDEFINED) and would then show up under that name."""
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import load_bearing_fraction as lbf  # noqa: E402


def _wmc_synthetic_spawn(env, turn_labels, out_path):
    """Fake `_spawn_arm` for the content probe's two turn-groups (['wmc_intro','wmc_drive'] and
    ['wmcx_intro','wmcx_drive']). Varies by `env` (is BRAIN_MULTIREF_LESION armed? BRAIN_MULTIREF_LESION_SCOPE is
    present on EVERY arm per `_WMC_BASE_ENV`, so it alone cannot distinguish intact from lesion) to reproduce a
    genuine T=true GO on BOTH content sessions: intro is in-scope on every arm (R1), the drive is ordinary on
    every arm (R2), the confined lesion kills the intro hold (L), the intact rebuild is clean (N), the intact
    reply follows its own content and not the other's (C), the lesion reproduces (R), and the intact vs lesion
    drive answer differs (T) -- adequate on every pre-registered condition, so `_wmc_gate` returns (True,
    "regressed"), not an UNDEFINED probe-inadequate verdict."""
    lesioned = env.get("BRAIN_MULTIREF_LESION") == "1"
    out = {}
    for label in turn_labels:
        if label in ("wmc_intro", "wmcx_intro"):
            mr = {"kind": "maintain", "n_referents": 2, "hold_alive_min": (0.0 if lesioned else 0.06)}
            if lesioned:
                mr["lesion_scope"] = "recur"
            out[label] = {"answer": "ok", "multiref": mr}
        elif label == "wmc_drive":
            out[label] = {"answer": ("who watched whom?" if lesioned else "the wolf watched the owl"),
                          "multiref": {"kind": "other"}}
        elif label == "wmcx_drive":
            out[label] = {"answer": ("who watched whom?" if lesioned else "the dog watched the owl"),
                          "multiref": {"kind": "other"}}
    return out


def _patch_wmc_on(monkeypatch):
    monkeypatch.setattr(lbf, "LB_WMB_CONTENT", True)
    monkeypatch.setattr(lbf, "_spawn_arm", _wmc_synthetic_spawn)


def test_content_probe_go_is_reported_under_the_distinct_faculty_key(monkeypatch, tmp_path):
    _patch_wmc_on(monkeypatch)
    res = lbf.measure_faculty("wm-binding-advanced", str(tmp_path), repeats=1, intact_cache={}, seed=42)
    # A genuine GO, not an UNDEFINED probe-inadequate reading -- proves the synthetic arms exercise the real gate.
    assert res["load_bearing"] is True, res
    assert res["verdict"] == "regressed", res
    # AMENDMENT C: the reported identity is the distinct key, never "wm-binding-advanced".
    assert res["faculty"] == "wm-binding-recurrence-drive", res
    assert res["faculty"] == lbf._WMC_FACULTY_KEY, res
    assert res["faculty"] != "wm-binding-advanced", res
    assert res.get("source_faculty_lesion_key") == "wm-binding-advanced", res
    assert res.get("counted_faculty_key") == "wm-binding-recurrence-drive", res


def test_content_probe_go_never_enters_wm_binding_advanceds_row_via_run(monkeypatch, tmp_path):
    """Integration-level pin: run()'s aggregation keys `load_bearing_faculties` /
    `not_load_bearing_faculties` off each row's OWN `res["faculty"]`. A real GO here (kind="neural-lesion",
    verdict="regressed", load_bearing=True) DOES enter the aggregate numerator/denominator per AMENDMENT B (this
    amendment does not change that), but must be attributed to "wm-binding-recurrence-drive", never
    "wm-binding-advanced"."""
    _patch_wmc_on(monkeypatch)
    report = lbf.run(out_dir=str(tmp_path), only=["wm-binding-advanced"], repeats=1, seed=42)
    per = report["per_faculty"][0]
    assert per["faculty"] == "wm-binding-recurrence-drive", per
    assert per["load_bearing"] is True, per
    # still counted in the aggregate fraction (AMENDMENT B: the lesion-kind wiring is unchanged) --
    assert report["counts"]["n_coverable_env_lesion"] == 1, report["counts"]
    assert report["counts"]["n_exercised"] == 1, report["counts"]
    assert report["counts"]["n_load_bearing"] == 1, report["counts"]
    assert report["load_bearing_fraction"] == 1.0, report
    # -- but never under the "wm-binding-advanced" name.
    assert "wm-binding-recurrence-drive" in report["load_bearing_faculties"]
    assert "wm-binding-advanced" not in report["load_bearing_faculties"]
    assert "wm-binding-advanced" not in report["not_load_bearing_faculties"]
