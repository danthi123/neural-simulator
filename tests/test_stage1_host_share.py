"""Stage-1 host share (packet T22): turn tables, label resolution, job lines and the aggregate on synthetic shards.

No brain is built: the shards are synthetic, in the instrument's own event format (research/runners/host_share.py).
"""
import json
import os

import pytest

from research.runners import host_share as H
from research.runners import stage1_host_share as T

SEEDS = (42, 43, 44, 100, 101, 102)


def _resp(patient, abstained=False):
    r = H._fake_resp(patient, abstained=abstained)
    r["source"] = H.EXPECTED_SOURCE
    return r


def _write(tmp, seed, group, arm, *, corpus=True, ign_follow=True, unresolved=False, drop_turn=None):
    specs = {t[0]: {"target": t[3], "cue": t[2]} for t in T.S1_GROUPS[group]["sessions"][0]}
    responses, events = {}, {}
    for lab, sp in specs.items():
        if lab == drop_turn:
            continue
        cand = sp["cue"][0] + "x" if sp["cue"] else None
        responses[lab] = _resp(cand) if sp["target"] else _resp(None, abstained=True)
        if sp["target"] and arm in ("intact_b", "ignition:intervene"):
            ev = {"seam": "ignition", "mode": "observe" if arm == "intact_b" else "intervene",
                  "host_committed": cand, "candidates": [cand], "rates": [0.33, 0.0], "thr": 0.2}
            if arm == "ignition:intervene":
                ev["applied"] = True
                if ign_follow:
                    responses[lab] = _resp(None, abstained=True)
            events[lab] = [ev]
    d = {"seed": seed, "group": group, "arm": arm, "responses": responses, "events": events, "snapshots": {},
         "seeds_used": {"value_choice": seed, "bg": seed}, "pythonhashseed_ok": True, "corpus_present": corpus,
         "unresolved_turns": [["x", "y"]] if unresolved else []}
    p = H.shard_path(str(tmp), seed, group, arm)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(d, open(p, "w"))


def _fill(tmp, **kw):
    for s in SEEDS:
        for g in T.S1_GROUPS:
            for a in T.arms_for(g):
                _write(tmp, s, g, a, **kw)


def test_selftest():
    assert T.selftest(verbose=False)[0]


def test_turns_replay_the_stage1_sessions():
    word = [t[1] for t in T.S1_GROUPS["s1_word"]["sessions"][0]]
    lived = [t[1] for t in T.S1_GROUPS["s1_lived"]["sessions"][0]]
    assert word == ["what does the bear chase", "the otter eats the clam", "the bear chases the otter",
                    "what does the bear chase", "what does it eat", "what does the otter eat",
                    "what does the otter chase", "what does the clam eat", "what does the dog chase"]
    assert lived[2] == "the wolf bites the apple" and lived[-4:] == [
        "what does the wolf bite", "did we discuss the wolf", "what does the wolf chase", "what does the dog chase"]
    assert T.S1_GROUPS["s1_lived"]["env"] == {"BRAIN_EPISODIC_STORE": "1"}
    targets = [t[0] for g in T.S1_GROUPS.values() for t in g["sessions"][0] if t[3]]
    assert set(targets) == set(T.PROBE_ROLE) and len(targets) == 8
    assert T.stage1_label("s1w_p_bear_chase") == "p_bear_chase"


def test_job_lines():
    jl = T.job_lines("a" * 40)
    assert len(jl) == 132
    assert len({c for c, _ in jl}) == 132
    assert all(c.startswith("cd ~/derisk-pool/revisions/" + "a" * 40 + " && ") for c, _ in jl)
    assert all("bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy" in c and "-m research.runners.stage1_host_share" in c
               for c, _ in jl)
    dev = T.job_lines("a" * 40, seeds=(7,), out_dir=T.DEV_OUT_DIR)
    assert len(dev) == 22 and not any("--allow-gate-seed" in c for c, _ in dev)
    with pytest.raises(ValueError):
        T.job_lines("abc")


def test_aggregate_valid_host_owned_ignition(tmp_path):
    _fill(tmp_path)
    rep = T.aggregate(str(tmp_path), SEEDS)
    assert rep["verdict"] == "MEASUREMENT-VALID", rep["validity"]
    for c in ("C2", "C3"):
        cr = rep["criteria"][c]
        assert cr["seams"]["ignition"]["class"].startswith("HOST-OWNED")
        assert cr["seams"]["k-select"]["class"] == "UNDEFINED"
        assert cr["credit"] == "PARTIAL (host decision: ignition)"
        assert cr["sham_noise_rate"] == 0.0
    assert "recall" in H.GROUPS and "s1_word" not in H.GROUPS
    md = T.report_markdown(rep)
    assert "C2 (`s1_word`" in md and "PARTIAL (host decision: ignition)" in md


def test_aggregate_neural_return_is_not_host(tmp_path):
    _fill(tmp_path, ign_follow=False)
    rep = T.aggregate(str(tmp_path), SEEDS)
    # F = 0 everywhere -> no F = 1 event -> V4 fails: not a valid measurement, and no credit is read
    assert rep["verdict"] == "INVALID-OR-INCOMPLETE"
    assert rep["criteria"]["C2"]["credit"].startswith("UNDEFINED")


def test_corpus_missing_invalidates(tmp_path):
    _fill(tmp_path)
    _write(tmp_path, 42, "s1_lived", "intact_a", corpus=False)
    rep = T.aggregate(str(tmp_path), SEEDS)
    assert rep["verdict"] == "INVALID-OR-INCOMPLETE"
    assert rep["validity"]["V7_corpus_missing"] == ["s42/s1_lived/intact_a"]


def test_unresolved_turn_invalidates(tmp_path):
    _fill(tmp_path)
    _write(tmp_path, 43, "s1_word", "organ-c:substitute", unresolved=True)
    assert T.aggregate(str(tmp_path), SEEDS)["verdict"] == "INVALID-OR-INCOMPLETE"


def test_missing_shard_invalidates(tmp_path):
    _fill(tmp_path)
    os.remove(H.shard_path(str(tmp_path), 101, "s1_word", "metacog:intervene"))
    rep = T.aggregate(str(tmp_path), SEEDS)
    assert rep["verdict"] == "INVALID-OR-INCOMPLETE" and rep["validity"]["missing_shards"]


def test_devcheck(tmp_path):
    for g in T.S1_GROUPS:
        for a in T.arms_for(g):
            _write(tmp_path, 7, g, a)
    ok, checks, live, _ = T.devcheck(str(tmp_path), [7])
    assert ok, checks
    assert live["C2"]["ignition"] == 1 and live["C2"]["k-select"] == 0


def test_stage1_agreement(tmp_path):
    _fill(tmp_path / "hs")
    for s in SEEDS:
        for g, sg in (("s1_word", "word"), ("s1_lived", "lived")):
            src = json.load(open(H.shard_path(str(tmp_path / "hs"), s, g, "intact_a")))
            resp = {T.stage1_label(k): v for k, v in src["responses"].items()}
            p = H.shard_path(str(tmp_path / "s1"), s, sg, "intact_a")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump({"responses": resp}, open(p, "w"))
    rep = T.aggregate(str(tmp_path / "hs"), SEEDS, stage1_dir=str(tmp_path / "s1"))
    a = rep["stage1_agreement"]
    assert a["n"] == 48 and a["n_agree"] == 48
