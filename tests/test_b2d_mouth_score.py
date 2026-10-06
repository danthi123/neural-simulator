"""tools/b2d_mouth_score.py: the B2d mouth-channel arm's scorer, on synthetic shard trees (no brain build).

Governing text: research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md ("Mouth-channel arm")
and its "Amendment 1" section.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import b2d_mouth_score as M  # noqa: E402

PIN = "e" * 40
HOST = "ws"
FAC = "affect-coloring"
TURN = "emo"
FIELDS_ROW = {"fields": {"affect.valence_sign": {"class": "N"}, "affect.tone_token": {"class": "N"}},
              "kept": ["affect.valence_sign", "affect.tone_token"], "probe_turn": TURN, "sd_only": False}


def _prov(env=None, sha=PIN, host=HOST):
    e = {"SIM_BACKEND": "cupy", "LB_ARM_RENDERER": "production"}
    e.update(env or {})
    return {"git_sha": sha, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "env": e, "host": host}


def _turn(answer, calls=1, renderer=M.QWEN_RENDERER_NAME, mode="production", extra=None):
    t = {"answer": answer, "abstained": False, "recalled_svo": ["wolf", "bite", "apple"], "renderer": renderer,
         "affect": {"on": True, "valence_sign": "+"},
         "_arm_renderer": {"mode": mode, "qwen_generate_calls": calls}}
    t.update(extra or {})
    return {TURN: t}


def _cell(base, seed, arms, tag="b2d0929-mouth", fac=FAC, lb_prov=None, arm_prov=None, drop=()):
    d = os.path.join(base, tag, "s%d" % seed, fac)
    os.makedirs(d, exist_ok=True)
    lb = {"per_faculty": [{"faculty": fac, "kind": "neural-lesion", "verdict": "regressed", "load_bearing": True,
                           "turn": TURN, "diffs": [], "null_control_clean": True}]}
    json.dump(lb, open(os.path.join(d, "lb.json"), "w"))
    json.dump(lb_prov or _prov(), open(os.path.join(d, "lb.json.prov.json"), "w"))
    names = {"intact_a": "intact_a_emo.json", "intact_b": "intact_b_emo.json", "lesion": "lesion_affect_coloring.json",
             "lesion.rep0": "lesion_affect_coloring.json.rep0"}
    for k, fn in names.items():
        if k in drop:
            continue
        json.dump(arms[k], open(os.path.join(d, fn), "w"))
        json.dump(arm_prov or _prov({"BRAIN_CHAT_SEED": str(seed)}), open(os.path.join(d, fn + ".prov.json"), "w"))
    return d


def _judge(d):
    return M.judge_mouth_cell(d, FAC, PIN, FIELDS_ROW)


def test_a_mouth_reply_that_changes_under_the_lesion_is_reply_load_bearing(tmp_path):
    arms = {"intact_a": _turn("The wolf happily bites the apple!"), "intact_b": _turn("The wolf happily bites the apple!"),
            "lesion": _turn("The wolf bites the apple."), "lesion.rep0": _turn("The wolf bites the apple.")}
    r = _judge(_cell(str(tmp_path), 42, arms))
    assert r["state"] == "DEFINED" and r["s2"][0] == "regressed" and r["s2"][1]["answer_only"] is True


def test_the_same_mouth_reply_is_a_pass(tmp_path):
    t = _turn("The wolf bites the apple.")
    r = _judge(_cell(str(tmp_path), 42, {k: t for k in M.ARM_KEYS}))
    assert r["state"] == "DEFINED" and r["s2"][0] == "pass"


def test_a_turn_the_mouth_never_rendered_is_mouth_silent_never_zero_or_pass(tmp_path):
    t = _turn("I don't know about that.", calls=0)
    r = _judge(_cell(str(tmp_path), 42, {k: t for k in M.ARM_KEYS}))
    assert r["state"] == "MOUTH-SILENT" and "s2" not in r
    assert M.final_state(r, None) == "MOUTH-SILENT"


def test_a_mouth_that_speaks_only_in_one_arm_is_defined(tmp_path):
    arms = {"intact_a": _turn("The wolf bites the apple.", calls=1), "intact_b": _turn("The wolf bites the apple.", 1),
            "lesion": _turn("I don't know about that.", calls=0), "lesion.rep0": _turn("I don't know about that.", 0)}
    r = _judge(_cell(str(tmp_path), 42, arms))
    assert r["state"] == "DEFINED" and r["s2"][0] == "regressed"


def test_a_stub_resolved_arm_is_not_the_mouth(tmp_path):
    t = _turn("the wolf bites the apple")
    arms = {k: t for k in M.ARM_KEYS}
    arms["lesion"] = _turn("the wolf bites the apple", renderer="template-stub (GPU-free)")
    r = _judge(_cell(str(tmp_path), 42, arms))
    assert r["state"] == "M" and M.final_state(r, None) == "M-UNCONFIRMED"
    assert M.final_state(r, dict(r)) == "FAIL"


def test_a_missing_mode_readout_is_not_the_mouth(tmp_path):
    t = _turn("x")
    del t[TURN]["_arm_renderer"]
    r = _judge(_cell(str(tmp_path), 42, {k: t for k in M.ARM_KEYS}))
    assert r["state"] == "M"


def test_numpy_or_stub_mode_sidecars_are_invalid(tmp_path):
    t = _turn("x")
    arms = {k: t for k in M.ARM_KEYS}
    r = _judge(_cell(str(tmp_path / "a"), 42, arms, lb_prov=_prov({"SIM_BACKEND": "numpy"})))
    assert r["state"] == "E" and any("SIM_BACKEND=numpy" in f for f in r["fails"])
    r = _judge(_cell(str(tmp_path / "b"), 42, arms, arm_prov=_prov({"LB_ARM_RENDERER": "stub",
                                                                      "BRAIN_CHAT_SEED": "42"})))
    assert r["state"] == "E" and any("LB_ARM_RENDERER=stub" in f for f in r["fails"])
    r = _judge(_cell(str(tmp_path / "c"), 42, arms, lb_prov=_prov(sha="d" * 40)))
    assert r["state"] == "E"
    r = _judge(_cell(str(tmp_path / "d"), 42, arms, arm_prov=_prov({"BRAIN_CHAT_RENDERER": "qwen"})))
    assert r["state"] == "E"


def test_a_missing_rep0_is_class_e_rule6(tmp_path):
    t = _turn("x")
    r = _judge(_cell(str(tmp_path), 42, {k: t for k in M.ARM_KEYS}, drop=("lesion.rep0",)))
    assert r["state"] == "E" and any("rule6" in f for f in r["fails"])


def test_null_noise_is_undefined_never_zero(tmp_path):
    arms = {"intact_a": _turn("A."), "intact_b": _turn("B."), "lesion": _turn("C."), "lesion.rep0": _turn("C.")}
    r = _judge(_cell(str(tmp_path), 42, arms))
    assert r["state"] == "DEFINED" and r["s2"][0] == "UNDEFINED:null-noise"


def test_an_organ_error_on_the_probe_turn_is_class_c(tmp_path):
    t = _turn("x", extra={"affect": {"on": True, "error": "ValueError: boom"}})
    r = _judge(_cell(str(tmp_path), 42, {k: t for k in M.ARM_KEYS}))
    assert r["state"] == "C"


def _fields_file(tmp_path, pin=PIN):
    p = tmp_path / "fields.json"
    json.dump({"pin": pin, "cited_blobs_at_pin": {}, "rows": {FAC: FIELDS_ROW}}, open(p, "w"))
    return str(p)


def test_row_readings_and_battery_verdict(tmp_path):
    base = str(tmp_path / "shards")
    lb_arms = {"intact_a": _turn("Warm."), "intact_b": _turn("Warm."), "lesion": _turn("Flat."),
               "lesion.rep0": _turn("Flat.")}
    for s in (42, 43):
        _cell(base, s, lb_arms)
    res = M.score(base, "b2d0929-mouth", PIN, [42, 43], _fields_file(tmp_path), rows=[FAC],
                  code_root=str(tmp_path))
    assert res["per_row"][FAC]["reading"] == "MOUTH-LOAD-BEARING" and res["verdict"] == "COMPLETE"
    silent = {k: _turn("I don't know.", calls=0) for k in M.ARM_KEYS}
    _cell(base, 43, silent)
    res = M.score(base, "b2d0929-mouth", PIN, [42, 43], _fields_file(tmp_path), rows=[FAC], code_root=str(tmp_path))
    assert res["per_row"][FAC]["reading"] == "MOUTH-PARTIAL" and res["per_row"][FAC]["n_mouth_silent"] == 1
    res = M.score(base, "b2d0929-mouth", PIN, [42, 43, 44], _fields_file(tmp_path), rows=[FAC],
                  code_root=str(tmp_path))
    assert res["verdict"] == "INCOMPLETE" and res["rerun_cells"] == ["s44/%s" % FAC]
    assert res["per_row"][FAC]["reading"] == "PENDING"


def test_a_fields_pin_mismatch_is_a_failed_precondition(tmp_path):
    base = str(tmp_path / "shards")
    _cell(base, 42, {k: _turn("x") for k in M.ARM_KEYS})
    res = M.score(base, "b2d0929-mouth", PIN, [42], _fields_file(tmp_path, pin="a" * 40), rows=[FAC],
                  code_root=str(tmp_path))
    assert res["verdict"].startswith("UNDEFINED")


def test_the_scorer_never_imports_the_harness(tmp_path):
    base = str(tmp_path / "shards")
    _cell(base, 42, {k: _turn("x") for k in M.ARM_KEYS})
    code = ("import sys; sys.path.insert(0, %r); import b2d_mouth_score as M; "
            "M.score(%r, 'b2d0929-mouth', %r, [42], %r, rows=[%r], code_root=%r); "
            "bad=[m for m in sys.modules if m.endswith(('load_bearing_fraction','onebrain_regression_battery')) "
            "or m.startswith('webapp')]; print(bad); sys.exit(1 if bad else 0)"
            % (os.path.join(ROOT, "tools"), base, PIN, _fields_file(tmp_path), FAC, str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_the_required_renderer_name_is_the_production_mouths():
    src = open(os.path.join(ROOT, "research", "runners", "brain_chat_tui.py")).read()
    assert 'name = "%s"' % M.QWEN_RENDERER_NAME in src
