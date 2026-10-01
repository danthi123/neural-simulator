"""tools/b2b_score.py: the B2b scorer's A1.2 rules and A1.5 states, on synthetic shard trees (no real data)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import b2b_score as bs  # noqa: E402

HOST = "node-x"
PRIMARY = "a" * 64
HOST_SHA = {HOST: PRIMARY}


def _prov(extra_env=None, sha=bs.F, host=HOST, **kw):
    env = {"SIM_BACKEND": "numpy"}
    env.update(extra_env or {})
    d = {"git_sha": sha, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
         "source_manifest_verified_at_exit": True, "env": env, "host": host}
    d.update(kw)
    return d


def _cell(tmp, fac="row-a", verdict="regressed", lb=True, null_clean=True, kind="neural-lesion",
          lb_prov=None, arm_prov=None, unreliable=False, turn=None):
    d = tmp / fac
    d.mkdir(parents=True, exist_ok=True)
    arm_env = {"BRAIN_CHAT_SEED": "42"}
    for name in ("intact_a_t.json", "intact_b_t.json", "lesion_%s.json" % fac.replace("-", "_"),
                 "lesion_%s.json.rep0" % fac.replace("-", "_")):
        (d / name).write_text(json.dumps({"t": turn or {"answer": "x"}}))
        (d / (name + ".prov.json")).write_text(json.dumps(arm_prov or _prov(arm_env)))
    if lb:
        rep = {"per_faculty": [{"faculty": fac, "kind": kind, "verdict": verdict,
                                "load_bearing": verdict == "regressed", "null_control_clean": null_clean}]}
        if unreliable:
            rep["UNRELIABLE"] = True
        (d / "lb.json").write_text(json.dumps(rep))
        (d / "lb.json.prov.json").write_text(json.dumps(lb_prov or _prov()))
    return str(d)


def test_good_cell_is_defined(tmp_path):
    r = bs.judge_dir(_cell(tmp_path), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] is None and r["fails"] == [] and r["host"] == HOST


def test_rule1_wrong_sha_and_rule3_brain_key_are_class_e(tmp_path):
    r = bs.judge_dir(_cell(tmp_path / "a", lb_prov=_prov(sha="deadbeef")), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] == "E" and any("git_sha" in x for x in r["fails"])
    r = bs.judge_dir(_cell(tmp_path / "b", lb_prov=_prov({"BRAIN_X": "1"})), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] == "E" and any("BRAIN keys" in x for x in r["fails"])


def test_rule2_arm_sidecar_and_rule3_arm_brain_key(tmp_path):
    bad = _prov({"BRAIN_CHAT_SEED": "42", "BRAIN_OTHER": "1"})
    r = bs.judge_dir(_cell(tmp_path, arm_prov=bad), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] == "E" and any("arm BRAIN keys" in x for x in r["fails"])


def test_rule2_missing_arm_sidecar(tmp_path):
    d = _cell(tmp_path)
    os.remove(os.path.join(d, "intact_a_t.json.prov.json"))
    r = bs.judge_dir(d, "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] == "E" and any("no sidecar" in x for x in r["fails"])


def test_rule4_unknown_or_mismatched_corpus_host(tmp_path):
    r = bs.judge_dir(_cell(tmp_path / "a", lb_prov=_prov(host="elsewhere")), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] == "E" and any("rule4" in x for x in r["fails"])
    r = bs.judge_dir(_cell(tmp_path / "b"), "row-a", bs.F, {HOST: "b" * 64}, PRIMARY)
    assert r["class"] == "E" and any("rule4" in x for x in r["fails"])


def test_rule5_classes(tmp_path):
    assert bs.judge_dir(_cell(tmp_path / "a", verdict="arm-build-failed"), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] == "C"
    assert bs.judge_dir(_cell(tmp_path / "b", verdict="unmapped-x"), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] == "C"
    assert bs.judge_dir(_cell(tmp_path / "c", null_clean=False), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] == "C"
    assert bs.judge_dir(_cell(tmp_path / "d", unreliable=True), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] == "C"
    assert bs.judge_dir(_cell(tmp_path / "e", verdict="probe-inadequate:x"), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] == "I"
    assert bs.judge_dir(_cell(tmp_path / "f", verdict="pass"), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] is None
    assert bs.judge_dir(_cell(tmp_path / "g", verdict="not-exercised"), "row-a", bs.F, HOST_SHA, PRIMARY)["class"] is None


def test_missing_lb_json_is_class_e(tmp_path):
    r = bs.judge_dir(_cell(tmp_path, lb=False), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] == "E" and "E: no lb.json" in r["fails"]


def test_thin_row_is_not_judged_on_rule5(tmp_path):
    r = bs.judge_dir(_cell(tmp_path, kind="thin", verdict="not-covered:thin"), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] is None


def test_multiref_error_turns_are_counted(tmp_path):
    r = bs.judge_dir(_cell(tmp_path, turn={"multiref": {"error": "boom"}}), "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["n_multiref_turns"] == 4 and r["multiref_error_turns"] == 4 and r["class"] is None


def test_a1_5_final_states():
    ok, c, i, e = {"class": None}, {"class": "C"}, {"class": "I"}, {"class": "E"}
    assert bs.final_state(ok, e, "pool1", "pool41") == "DEFINED"
    assert bs.final_state(c, c, "pool1", "pool41") == "FAIL"
    assert bs.final_state(c, c, "pool1", "pool1") == "C-UNCONFIRMED"   # a same-host re-run cannot make a FAIL
    assert bs.final_state(c, e, "pool1", "pool41") == "C-UNCONFIRMED"
    assert bs.final_state(c, None, None, None) == "C-UNCONFIRMED"
    assert bs.final_state(i, c, "pool1", "pool41") == "UNDECIDED"
    assert bs.final_state(e, None, None, None) == "MISSING"


def test_cross_check_flags_any_number_that_differs():
    res = {"robust_core_n": 1, "union_n": 1, "robust_core": ["r"], "mean_fraction": 1.0, "sd_fraction_pop": 0.0,
           "per_seed": {str(s): {"n_exercised": 1, "n_load_bearing": 1} for s in bs.SEEDS},
           "per_faculty": {"r": {"n_load_bearing": 6}}}
    agg = {"provenance": {"pin": bs.F, "n_invalid": 0}, "robust_core_n": 1, "union_n": 1, "robust_core": ["r"],
           "mean_fraction": 1.0, "sd_fraction": 0.0, "incomplete_faculties": [],
           "per_seed": {str(s): {"n_exercised": 1, "n_load_bearing": 1} for s in bs.SEEDS},
           "per_faculty": {"r": {"n_load_bearing": 6}}}
    assert bs.cross_check(agg, res) == []
    agg["per_faculty"]["r"]["n_load_bearing"] = 5
    assert any("n_load_bearing" in m for m in bs.cross_check(agg, res))


def test_organ_error_turns_are_reported(tmp_path):
    r = bs.judge_dir(_cell(tmp_path, turn={"comprehension": {"on": True, "error": "ValueError: shape"}}),
                     "row-a", bs.F, HOST_SHA, PRIMARY)
    assert r["class"] is None                      # reported, not a validity rule under the B2b prereg
    assert len(r["organ_errors"]) == 4 and r["organ_errors"][0]["organ"] == "comprehension"
