"""tools/s2p_score.py: the session-2 packets' registered scorer (selection rule, organ step O, step R's op check)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import s2p_score as S  # noqa: E402

PK = "fb-spiking-witness"


def _w(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f)


def _sweep(d, passing):
    """passing: {op: set(seeds that pass)}; every (op, seed) file for ops 0..7 is written."""
    for op in range(8):
        for s in S.DEV_SEEDS:
            _w(d, "op%d_s%d.json" % (op, s), {"seed": s, "op_index": op, "all_pass": s in passing.get(op, set())})


def test_select_takes_the_first_index_passing_every_dev_seed(tmp_path):
    _sweep(str(tmp_path), {2: set(S.DEV_SEEDS) - {5}, 4: set(S.DEV_SEEDS), 6: set(S.DEV_SEEDS)})
    res, code = S.score_select(PK, str(tmp_path))
    assert (res["verdict"], res["op"], code) == ("SELECTED", 4, S.PASS)


def test_select_none_when_no_index_passes_all(tmp_path):
    _sweep(str(tmp_path), {op: set(S.DEV_SEEDS) - {1} for op in range(8)})
    res, code = S.score_select(PK, str(tmp_path))
    assert (res["verdict"], res["op"], code) == ("NONE", None, S.FAIL)


def test_select_never_skips_a_missing_cell(tmp_path):
    _sweep(str(tmp_path), {4: set(S.DEV_SEEDS)})
    os.remove(os.path.join(str(tmp_path), "op1_s3.json"))
    res, code = S.score_select(PK, str(tmp_path))
    assert (res["verdict"], code) == ("UNDEFINED", S.UNDEF) and "op1_s3" in res["missing"]


def _organ_dir(d, op, fail_seed=None, prereg=None):
    for s in S.GATE_SEEDS:
        _w(d, "s%d.json" % s, {"seed": s, "op_index": op, "all_pass": s != fail_seed,
                               "prereg": prereg or S.PACKETS[PK]["prereg"],
                               "cases": {"false_belief": {"pass": s != fail_seed}}})
        _w(d, "s%d.json.prov.json" % s, {"git_sha": "x"})
    _w(d, "selection.json", {"packet": PK, "verdict": "SELECTED", "op": op})


def test_O_pass_fail_undefined(tmp_path, monkeypatch):
    monkeypatch.setattr(S.B, "sidecar_fails", lambda pj, pin, is_lb: [])
    d = str(tmp_path)
    _organ_dir(d, 4)
    assert S.score_O(PK, d, os.path.join(d, "selection.json"), "pin")[0]["verdict"] == "PASS"
    _organ_dir(d, 4, fail_seed=101)
    res, code = S.score_O(PK, d, os.path.join(d, "selection.json"), "pin")
    assert (res["verdict"], code) == ("FAIL", S.FAIL) and res["per_seed"]["101"]["failed_cases"] == ["false_belief"]
    _organ_dir(d, 4, prereg="other.md")
    assert S.score_O(PK, d, os.path.join(d, "selection.json"), "pin")[0]["verdict"] == "UNDEFINED"


def test_O_rejects_a_cell_at_another_operating_point(tmp_path, monkeypatch):
    monkeypatch.setattr(S.B, "sidecar_fails", lambda pj, pin, is_lb: [])
    d = str(tmp_path)
    _organ_dir(d, 4)
    _w(d, "s44.json", {"seed": 44, "op_index": 5, "all_pass": True, "prereg": S.PACKETS[PK]["prereg"]})
    res, _ = S.score_O(PK, d, os.path.join(d, "selection.json"), "pin")
    assert res["verdict"] == "UNDEFINED" and any("op_index" in f for f in res["per_seed"]["44"]["fails"])


def test_R_sidecar_must_carry_the_selected_operating_point(monkeypatch):
    monkeypatch.setattr(S.B, "sidecar_fails", lambda pj, pin, is_lb: [])
    spec = S.PACKETS[PK]
    f = S._allowing_op(spec["flags"] + spec["lesion_flags"], spec["flags"], spec["op_env"], 4)
    env_ok = {"BRAIN_FALSE_BELIEF_CHAT": "1", "BRAIN_FALSE_BELIEF_SPIKING_WITNESS": "1",
              "BRAIN_FALSE_BELIEF_WITNESS_OP": "4"}
    assert f({"env": env_ok}, "pin", True) == []
    assert f({"env": dict(env_ok, BRAIN_FALSE_BELIEF_WITNESS_OP="5")}, "pin", True)
    assert f({"env": dict(env_ok, BRAIN_FALSE_BELIEF_SPIKING_WITNESS="0")}, "pin", True)


def test_R_without_selection_is_undefined(tmp_path):
    res, code = S.score_R(PK, "s2p-R-fb-x", "pin", str(tmp_path / "nope.json"))
    assert (res["verdict"], code) == ("UNDEFINED", S.UNDEF)


def test_every_packet_names_a_prereg_and_a_control():
    for k, p in S.PACKETS.items():
        assert p["prereg"].startswith("research/findings/2026-10-01-") and "PREREGISTRATION" in p["prereg"], k
        assert p["control"]["row"] and p["op_env"].startswith("BRAIN_") and p["lesion_flags"], k


def test_flip_tool_apply_then_revert_round_trips(tmp_path, monkeypatch):
    import shutil
    import s2p_flip as FL
    for rel in ("webapp/false_belief_chat.py", "research/runners/lbf_rows/fb_spiking_witness.py",
                "webapp/reasoning_transitive_chat.py", "research/runners/lbf_rows/transitive_spiking_reach.py"):
        os.makedirs(os.path.dirname(str(tmp_path / rel)), exist_ok=True)
        shutil.copy(os.path.join(FL.ROOT, rel), str(tmp_path / rel))
    monkeypatch.setattr(FL, "ROOT", str(tmp_path))
    before = {r: open(str(tmp_path / r)).read() for r in ("webapp/false_belief_chat.py",
                                                          "research/runners/lbf_rows/fb_spiking_witness.py")}
    assert FL.main(["apply", "fb-spiking-witness=4", "transitive-spiking-reach=0"]) == 0
    st = {p: v for p, v in FL.status().items() if p != "word-role-parse"}
    assert st["fb-spiking-witness"] == {"on": True, "op": 4, "rows_lifted": True}
    assert st["transitive-spiking-reach"]["on"] is True
    assert FL.main(["apply", "fb-spiking-witness=4"]) == 2          # already on: refuses, writes nothing
    assert FL.main(["revert", "fb-spiking-witness", "transitive-spiking-reach"]) == 0
    assert {r: open(str(tmp_path / r)).read() for r in before} == before


# ── ENGAGEMENT (RULING iw2-merged (b), session 7): a transitive-spiking-reach R cell counts only when its arms record
#    that the route ran, the lesion flag matched the arm, and no exception was swallowed. ───────────────────────────
def _eng_arms(tmp_path, lesion_trace, intact_trace=None):
    import json as _j
    intact_trace = intact_trace or {"route": "spiking_reach", "lesioned": False, "error": None}
    for name, t in (("intact_a_x.json", intact_trace), ("intact_b_x.json", intact_trace),
                    ("lesion_x.json", lesion_trace), ("lesion_x.json.rep0", lesion_trace)):
        (tmp_path / name).write_text(_j.dumps({"t1": {"answer": "a"}, "probe": {"answer": "b", "transitive_trace": t}}))
    (tmp_path / "lb.json").write_text("{}")
    return str(tmp_path)


def test_engagement_recorded_passes(tmp_path):
    import s2p_score as S2
    eng = S2.PACKETS["transitive-spiking-reach"]["engagement"]
    d = _eng_arms(tmp_path, {"route": "spiking_reach", "lesioned": True, "error": None})
    assert S2.engagement_state(d, eng) is None


@pytest.mark.parametrize("lesion_trace,why", [
    ({"route": "spiking_reach", "lesioned": True, "error": "ValueError: organ crashed"}, "error"),
    ({"route": "chase", "lesioned": True, "error": None}, "route"),
    ({"route": "spiking_reach", "lesioned": False, "error": None}, "lesion-flag"),
])
def test_engagement_failures_are_undefined(tmp_path, lesion_trace, why):
    import s2p_score as S2
    eng = S2.PACKETS["transitive-spiking-reach"]["engagement"]
    assert S2.engagement_state(_eng_arms(tmp_path, lesion_trace), eng).startswith(why)


def test_engagement_missing_record_is_undefined(tmp_path):
    import json as _j
    import s2p_score as S2
    eng = S2.PACKETS["transitive-spiking-reach"]["engagement"]
    for name in ("intact_a_x.json", "lesion_x.json"):
        (tmp_path / name).write_text(_j.dumps({"probe": {"answer": "b"}}))
    assert S2.engagement_state(str(tmp_path), eng).startswith("no-record")
