"""research/runners/c3_flip_compare.py: the C3 L+P flip reader (RULING c3-flip-1029b) on synthetic shards."""
import copy
import json
import os

from research.runners import c3_flip_compare as C

SEEDS = (42, 43)


def _shard(group, answers):
    resp = {}
    for label, (ans, abst, svo) in answers.items():
        resp[label] = {"answer": ans, "abstained": abst, "recalled_svo": svo, "curiosity": None}
    return {"group": group, "arm": "intact_a", "responses": resp, "timings_s": {"x": 1.0}, "wall_s": 3.0}


BASE = {"t1": ("the fox chases the hare", False, ["fox", "chase", "hare"]),
        "p1": ("I don't know about that. My curiosity is piqued about owls.", True, None)}


def _write(root, seed, group, shard, flags):
    d = os.path.join(root, "s%d" % seed)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s__intact_a.json" % group)
    with open(p, "w") as f:
        json.dump(shard, f)
    env = {"SIM_BACKEND": "numpy"}
    if flags:
        env.update({k: "1" for k in C.FLAGS})
    with open(p + ".prov.json", "w") as f:
        json.dump({"git_sha": "f" * 40, "env": env}, f)


def _battery(tmp_path, edit_on=None, on_flags=True, off_flags=False, skip=None):
    on_root, off_root = str(tmp_path / "on"), str(tmp_path / "off")
    for s in SEEDS:
        for g in C.GROUPS:
            off = _shard(g, BASE)
            on = copy.deepcopy(off)
            on["timings_s"] = {"x": 9.0}                   # excluded: never compared
            if edit_on:
                edit_on(s, g, on)
            if skip != (s, g):
                _write(on_root, s, g, on, on_flags)
            _write(off_root, s, g, off, off_flags)
    return C.score(on_root, off_root, SEEDS)


def test_identical_is_go(tmp_path):
    r = _battery(tmp_path)
    assert r["status"] == "GO" and r["go"] is True
    assert all(c["pass"] and c["n_other_leaf_diffs"] == 0 for c in r["cells"].values())


def test_curiosity_clause_on_both_abstained_turn_is_allowed(tmp_path):
    def edit(s, g, on):
        on["responses"]["p1"]["answer"] = "I don't know about that."
    r = _battery(tmp_path, edit)
    assert r["status"] == "GO"
    assert all(c["both_abstained"] == ["p1"] for c in r["cells"].values())


def test_answer_change_on_answered_turn_fails(tmp_path):
    def edit(s, g, on):
        if (s, g) == (43, "infer"):
            on["responses"]["t1"]["answer"] = "the fox chases the deer"
    r = _battery(tmp_path, edit)
    assert r["status"] == "NO-GO" and r["go"] is False
    assert r["cells"]["43/infer"]["answer_diff"] == ["t1"]


def test_lost_answer_is_flip_hurts_answers(tmp_path):
    def edit(s, g, on):
        if (s, g) == (42, "emo"):
            on["responses"]["t1"].update(answer="I don't know about that.", abstained=True)
    r = _battery(tmp_path, edit)
    assert r["status"] == "NO-GO"
    c = r["cells"]["42/emo"]
    assert c["hurts"] == ["t1"] and c["abstained_diff"] == ["t1"] and r["flags"] == [C.HURTS]


def test_svo_change_fails_and_other_leaf_is_reported(tmp_path):
    def edit(s, g, on):
        if (s, g) == (42, "conv"):
            on["responses"]["t1"]["recalled_svo"] = ["fox", "chase", "deer"]
        if (s, g) == (43, "inner"):
            on["responses"]["t1"]["curiosity"] = {"curiosity_topic_stop": {"applied": True}}
    r = _battery(tmp_path, edit)
    assert r["status"] == "NO-GO" and r["cells"]["42/conv"]["svo_diff"] == ["t1"]
    inner = r["cells"]["43/inner"]
    assert inner["pass"] and inner["n_other_leaf_diffs"] >= 1 and inner["on_topic_stop_traces"] == 1


def test_missing_cell_is_undefined(tmp_path):
    r = _battery(tmp_path, skip=(43, "emo"))
    assert r["status"] == "UNDEFINED" and r["go"] is False


def test_unengaged_flags_are_undefined(tmp_path):
    assert _battery(tmp_path, on_flags=False)["status"] == "UNDEFINED"


def test_flags_on_the_off_side_are_undefined(tmp_path):
    assert _battery(tmp_path, off_flags=True)["status"] == "UNDEFINED"
