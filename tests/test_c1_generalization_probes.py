"""CPU tests for research/runners/c1_generalization_probes.py (the C1 override + graded dev probes, RULING
c1-moat-ruled-1013). No brain and no organ is built."""
import json
import os

import pytest

from research.runners import c1_generalization_probes as P


def test_selftest_passes():
    assert P.selftest(verbose=False)


def _write(root, seed, shards):
    for probe, shard in shards.items():
        p = P.shard_path(root, seed, probe)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(shard, f)


def test_score_reads_files_and_passes_a_perfect_brain(tmp_path):
    for s in P.DEV_SEEDS:
        _write(str(tmp_path), s, P.synthetic_seed("perfect"))
    out = P.score(str(tmp_path))
    assert out["override"]["verdict"] == "PASS" and out["override"]["n_pass"] == 8
    assert out["graded"]["verdict"] == "PASS" and out["graded"]["n_pass"] == 8
    assert all(c["ok"] for c in out["override"]["preconditions"])


def test_a_template_seed_fails_override_and_a_flat_seed_fails_graded(tmp_path):
    for s in P.DEV_SEEDS:
        _write(str(tmp_path), s, P.synthetic_seed({4: "template", 6: "graded_flat"}.get(s, "perfect")))
    out = P.score(str(tmp_path))
    assert out["override"]["verdict"] == "FAIL" and any(w.startswith("seed 4: O1") for w in out["override"]["why"])
    assert out["graded"]["verdict"] == "FAIL" and any(w.startswith("seed 6:") for w in out["graded"]["why"])


def test_a_missing_graded_shard_is_undefined_for_both_probes(tmp_path):
    for s in P.DEV_SEEDS:
        sh = P.synthetic_seed("perfect")
        if s == 8:
            sh.pop("graded")
        _write(str(tmp_path), s, sh)
    out = P.score(str(tmp_path))
    assert out["override"]["verdict"] == "UNDEFINED" and out["override"]["undefined_seeds"] == [8]
    assert out["graded"]["verdict"] == "UNDEFINED" and out["graded"]["undefined_seeds"] == [8]


def test_a_failed_precondition_downgrades_an_asserted_verdict():
    res = P._earn({"verdict": "PASS", "per_seed": {1: {"defined": False, "undefined": ["override/hop_not_recorded"]}}},
                  "t")
    assert res["verdict"] == "UNDEFINED" and res["verdict_unearned"] == "PASS"


def test_gate_seeds_are_refused():
    with pytest.raises(SystemExit):
        P.pool_job_lines("0" * 40, seeds=(1, 42))
    with pytest.raises(SystemExit):
        P.run_shard(100, "override", "/nonexistent/x.json")
    with pytest.raises(SystemExit):
        P.organ_feed(44, "graded")


def test_run_shard_removes_the_c1_envs_so_the_defaults_apply(monkeypatch, tmp_path):
    seen = []

    class _P:
        returncode = 1

    def _fake_run(cmd, env=None, cwd=None):
        seen.append((env, cmd))
        return _P()

    for k in P.C1_ENVS_REMOVED:
        monkeypatch.setenv(k, "0")
    monkeypatch.setattr(P.subprocess, "run", _fake_run)
    assert P.run_shard(7, "graded", str(tmp_path / "g.json")) == 1
    env, cmd = seen[0]
    assert not any(k in env for k in P.C1_ENVS_REMOVED)
    assert env["PYTHONHASHSEED"] == "0" and env["SIM_BACKEND"] == "numpy"
    assert cmd[cmd.index("--probe") + 1] == "graded" and "--worker" in cmd


def test_turn_tables_hold_the_registered_design():
    ov = P.TURNS["override"]
    teach = [t for t in P.CF.CAT_BRIDGE_TURNS if t[2] == "teach"]
    assert ov[:len(teach)] == teach                                  # the cat_bridge context, verbatim
    labels = [l for l, _m, _r in ov]
    assert labels.index("p_bat_swim_pre") < labels.index("t_bat_swim_told") < labels.index("p_bat_swim_post")
    assert P.expected_svo("the bat swims in the pond")[1] == "swim"  # competes with `swim lake` under the asked verb
    # graded: constant number of told facts per target, k of them fish traits (frames trout AND salmon carry, minus
    # the queried property), taught in descending k, shared traits first
    frames = {}
    for _l, m, r in P.TURNS["graded"]:
        if r == "teach":
            a, v, p = P.expected_svo(m)
            frames.setdefault(a, []).append((v, p))
    fish = (set(frames["trout"]) & set(frames["salmon"])) - {("swim", "lake")}
    for a, k in P.SHARED_TRAITS.items():
        own = frames[a][:3]
        assert len(own) == 3 and sum(f in fish for f in own) == k, (a, own)
        assert all(f in fish for f in own[:k]), "shared traits come first"
        assert ("swim", "lake") not in own
    unshared = [f for a in P.SHARED_TRAITS for f in frames[a][:3] if f not in fish]
    assert len(unshared) == len(set(unshared)), "an unshared frame belongs to one target only"
    order = [P.expected_svo(m)[0] for _l, m, r in P.TURNS["graded"] if r == "teach"]
    assert order.index("eel") < order.index("toad") < order.index("newt")    # descending k
    assert frames["eel"][3] == ("swim", "river")                             # the strongest-guess override, last


def test_teach_sentences_reach_the_organ_as_expected_by_the_production_extractor():
    """The instrument: every teach sentence's B3 extraction + lemma (what `_maybe_acquire` stores and the organ
    observes) equals `expected_svo`, which the DEFINED precondition checks against the trace."""
    b3 = pytest.importorskip("research.runners.b3_noncontradiction_production_organ")
    from research.runners.lexical_lemma import lemma_verb
    for probe in P.PROBES:
        for _l, m, r in P.TURNS[probe]:
            if r != "teach":
                continue
            a, v, p, pol = b3.extract_polar_assertion(m)
            assert (a, lemma_verb(v), p) == P.expected_svo(m) and pol == "AFFIRM", m


def test_pool_lines_are_dev_only_and_carry_the_lane_fields():
    lines = P.pool_job_lines("b" * 40)
    assert len(lines) == 16
    for c, w in lines:
        assert "revisions/" + "b" * 40 in c and "--out %s/" % P.OUT_DIR in c
        assert "mem_gb=9" in w and "max_age_h=72" in w and "c1-moat-ruled-1013" in w
        seed = int(c.split("--seed ")[1].split()[0])
        assert seed in P.DEV_SEEDS
