"""tools/answer_rate.py: the REPORTED flip check (Claude review 2026-10-06; wm-focus-bind abstained on 2 of 4 sessions with
the flag on, where the flag-off default answered all 4, on 6/6 seeds). Fixtures only; no brain."""
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
AR = importlib.import_module("answer_rate")

ANS = {"answer": "The dog chases cat.", "abstained": False}
ABS = {"answer": "I don't know about that.", "abstained": True}


def _arm(d, name, probe, intro=None):
    os.makedirs(d, exist_ok=True)
    doc = {"x_intro": intro or {"answer": "ok", "abstained": False}, "x_ask": probe}
    with open(os.path.join(str(d), name), "w") as f:
        json.dump(doc, f)
    with open(os.path.join(str(d), name + ".prov.json"), "w") as f:
        json.dump({"host": "node-x"}, f)


def test_classify_flag_text_fallback_and_empty():
    assert AR.classify(ANS) == "answered"
    assert AR.classify(ABS) == "abstained"
    assert AR.classify({"answer": "I do not know about that.", "abstained": False}) == "abstained-text-only"
    assert AR.classify({"answer": "I haven't learned about wolf yet: tell me?"}) == "abstained-text-only"   # no flag at all
    assert AR.classify({"answer": "  ", "abstained": False}) == "empty" and AR.classify(None) == "empty"
    assert AR.classify({"answer": "I don't know what you mean by that cat", "abstained": False}) == "answered"


def test_rate_counts():
    r = AR.rate([ANS, ABS, ABS, {"answer": "", "abstained": False}])
    assert (r["n"], r["answered"], r["abstained"], r["empty"], r["answer_rate"]) == (4, 1, 2, 1, 0.25)
    assert AR.rate([])["answer_rate"] is None


def test_report_flags_a_flip_that_hurts_on_any_seed_and_only_then():
    ok = {"on": [ANS] * 4, "off": [ANS] * 4}
    better = {"on": [ANS] * 4, "off": [ANS, ABS, ABS, ABS]}                       # the flip answers MORE: not a harm
    hurt = {"on": [ANS, ANS, ABS, ABS], "off": [ANS] * 4}                          # the wm-focus-bind shape
    rep = AR.report({"42": ok, "43": better})
    assert rep["status"] == "OK" and rep["flags"] == [] and rep["n_measured"] == 2
    rep = AR.report({"42": ok, "43": hurt, "44": ok})
    assert rep["flags"] == ["FLIP-HURTS-ANSWERS"] and rep["hurt_seeds"] == ["43"] and rep["status"] == "FLIP-HURTS-ANSWERS"
    assert rep["per_seed"]["43"]["on"]["answer_rate"] == 0.5 and rep["per_seed"]["43"]["off"]["answer_rate"] == 1.0
    assert "FLIP-HURTS-ANSWERS" in AR.render(rep) and "Claude ruling" in AR.render(rep)


def test_a_seed_missing_a_side_is_unmeasured_never_ok():
    rep = AR.report({"42": {"on": [ANS], "off": []}, "43": {"on": [ANS], "off": [ANS]}})
    assert rep["status"] == "UNMEASURED" and rep["unmeasured_seeds"] == ["42"] and rep["flags"] == []
    rep = AR.report({"42": {"on": [ANS], "off": []}, "43": {"on": [ABS], "off": [ANS]}})
    assert rep["flags"] == ["FLIP-HURTS-ANSWERS"]                                  # a harm on a measured seed still fires


def test_prefixed_layout_reproduces_the_wmf_finding(tmp_path):
    """on_intact_* / off_intact_* arm files, four sessions per seed; flag-on abstains on 2 of 4 on every seed."""
    for seed in (42, 43, 44, 100, 101, 102):
        d = tmp_path / ("s%d" % seed) / "wm-binding-advanced"
        for ses, on_probe in (("a1", ANS), ("a2", ABS), ("b1", ANS), ("b2", ABS)):
            _arm(d, "on_intact_a_wmf_%s.json" % ses, on_probe)
            _arm(d, "on_intact_b_wmf_%s.json" % ses, on_probe)
            _arm(d, "off_intact_a_wmf_%s.json" % ses, ANS)
            _arm(d, "off_lesion_wmf_%s.json" % ses, ABS)                             # lesion arms are not read
            _arm(d, "on_lesion_wmf_%s.json" % ses, ABS)
        with open(d / "lb.json", "w") as f:
            json.dump({}, f)
    dirs = sorted(str(p) for p in tmp_path.glob("s*/wm-binding-advanced"))
    rep = AR.report(AR.prefixed_per_seed(dirs))
    assert rep["flags"] == ["FLIP-HURTS-ANSWERS"] and sorted(rep["hurt_seeds"]) == ["100", "101", "102", "42", "43", "44"]
    s = rep["per_seed"]["42"]
    assert (s["on"]["n"], s["on"]["answered"], s["on"]["abstained"]) == (8, 4, 4)
    assert (s["off"]["n"], s["off"]["answered"]) == (4, 4)
    assert AR.main(["prefixed", str(tmp_path / "s*" / "wm-binding-advanced")]) == 0          # reported, exit 0
    assert AR.main(["prefixed", str(tmp_path / "s*" / "wm-binding-advanced"), "--strict"]) == 4
    assert AR.main(["prefixed", str(tmp_path / "nope*")]) == 2


def test_the_probe_is_the_last_answer_bearing_turn(tmp_path):
    _arm(tmp_path, "intact_a_x.json", ABS, intro=ANS)                               # intro answered, probe abstains
    assert AR.classify(AR.probe_body(str(tmp_path / "intact_a_x.json"))) == "abstained"
    (tmp_path / "bad.json").write_text("{not json")
    assert AR.probe_body(str(tmp_path / "bad.json")) is None and AR.probe_body(str(tmp_path / "missing.json")) is None


def test_paired_tags_per_seed_and_missing_flag_off_cell(tmp_path):
    for s in AR.SEEDS:
        _arm(tmp_path / "flip" / ("s%d" % s) / "fac", "intact_a_q.json", ANS if s != 100 else ABS)
        _arm(tmp_path / "flip" / ("s%d" % s) / "fac", "intact_b_q.json", ANS)
        _arm(tmp_path / "off" / ("s%d" % s) / "fac", "intact_a_q.json", ANS)
        _arm(tmp_path / "off" / ("s%d" % s) / "fac", "intact_b_q.json", ANS)
    rep = AR.report(AR.paired_per_seed(str(tmp_path), "flip", "off", "fac"))
    assert rep["hurt_seeds"] == ["100"] and rep["n_measured"] == 6
    rep = AR.report(AR.paired_per_seed(str(tmp_path), "flip", "absent-tag", "fac"))
    assert rep["status"] == "UNMEASURED" and rep["n_measured"] == 0
    assert AR.main(["paired", "--base", str(tmp_path), "--on-tag", "flip", "--off-tag", "off", "--faculty", "fac",
                    "--strict"]) == 4
    assert AR.main(["paired", "--base", str(tmp_path), "--on-tag", "flip", "--off-tag", "absent", "--faculty", "fac",
                    "--strict"]) == 3


# ── wired into the registered flip scorer, reported beside the verdict ──────────────────────────────────────────
def _iw2_fixture(tmp_path):
    t = importlib.import_module("test_integration_wave2_anchors")
    sc = importlib.import_module("tools.iw2_score")
    t._std(tmp_path, "base", "affect-coloring", "emo", "warm", "flat")
    t._std(tmp_path, "flip", "affect-coloring", "emo", "warm", "flat")
    t._fill(tmp_path, tag="flip", env={}, arm_env={})
    return t, sc


def test_iw2_B_reports_the_answer_rate_and_never_changes_the_verdict(tmp_path):
    t, sc = _iw2_fixture(tmp_path)
    res, code = sc.score_B("base", t.PIN, "flip", t.PIN, ["false-belief-chat"], base=str(tmp_path))
    assert code == sc.PASS and res["answer_rate"]["status"] == "UNMEASURED" and res["flags"] == []   # stated, not 'fine'
    # flag-off cells (forced-off run): the false-belief probe answers on every seed
    t._fill(tmp_path, tag="alloff", env={}, arm_env={})
    res, code = sc.score_B("base", t.PIN, "flip", t.PIN, ["false-belief-chat"], base=str(tmp_path), off_tag="alloff")
    assert code == sc.PASS and res["answer_rate"]["status"] == "OK" and res["flags"] == []
    # the flip abstains on seed 43 where the off run answers: the registered F1 verdict FAILs on its own bar, and the
    # reported flag rides along either way
    t._write_cell(str(tmp_path), "flip", 43, "tom-false-belief", "tom_fb", {"ia": t._ABSTAIN, "le": t._ABSTAIN}, arm_env={})
    res, code = sc.score_B("base", t.PIN, "flip", t.PIN, ["false-belief-chat"], base=str(tmp_path), off_tag="alloff")
    assert res["flags"] == ["FLIP-HURTS-ANSWERS"] and res["answer_rate"]["per_packet"]["false-belief-chat"]["hurt_seeds"] == ["43"]
    assert res["verdict"] == "FAIL" and code == sc.FAIL


def test_iw2_B_verdict_is_identical_with_and_without_the_report(tmp_path):
    t, sc = _iw2_fixture(tmp_path)
    t._fill(tmp_path, tag="alloff", env={}, arm_env={})
    t._write_cell(str(tmp_path), "flip", 100, "tom-false-belief", "tom_fb", {"ia": t._ABSTAIN, "le": t._ABSTAIN}, arm_env={})
    a, ca = sc.score_B("base", t.PIN, "flip", t.PIN, ["false-belief-chat"], base=str(tmp_path))
    b, cb = sc.score_B("base", t.PIN, "flip", t.PIN, ["false-belief-chat"], base=str(tmp_path), off_tag="alloff")
    assert (a["verdict"], ca, a["regressions"], a["undefined"]) == (b["verdict"], cb, b["regressions"], b["undefined"])
