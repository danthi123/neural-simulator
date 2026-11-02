"""Stage-2 exit gate, amendment A6 (DRAFT, docs/drafts/stage2-exit/AMENDMENT-A6.md): the C2 permuted-control arm's
acquisition precondition, selected by `--amendment-level`, plus the dev-preview options of `pool_job_lines`. No brain
build: synthetic shards (docs/drafts/stage2-exit/bar_sanity_s2_common.py) through the registered aggregate, plus the
committed 10-26 gate shards where they are checked out."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import stage2_exit_gate as SG

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "docs", "drafts", "stage2-exit"))
import bar_sanity_s2_common as C  # noqa: E402

_GATE = os.path.join(_REPO, "research", "findings", "raw", "_stage2_exit", "gate")
_SHA = "41d76bdbe9b58d4362b9d480f301f1db13550570"


def _infer_tree(td, over=None, realistic=False):
    data = C.make_group(np.random.default_rng(0), "infer", over=over, realistic=realistic)
    C.write_tree(td, {"infer": data})
    return sorted(data)


def _agg(td, seeds, level=None):
    kw = {} if level is None else {"amendment_level": level}
    return SG.aggregate(td, seeds=seeds, informational=False, **kw)


def test_default_level_is_a5_and_explicit_a5_is_identical(tmp_path):
    assert SG.AMENDMENT_LEVEL == "A5" and SG.AMENDMENT_LEVELS == ("A5", "A6")
    td = str(tmp_path)
    seeds = _infer_tree(td, over={"permuted": "permuted+refused"})
    a, b = _agg(td, seeds), _agg(td, seeds, "A5")
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert "amendment_level" not in a
    g = a["groups"]["infer"]
    assert g["verdict"] == "FAIL" and any("not exposure-driven" in w for w in g["why"])        # the 10-26 reading
    assert "permuted_acquisition" not in g["seeds"][seeds[0]]


def test_a6_refused_permuted_teach_is_undefined_never_fail(tmp_path):
    td = str(tmp_path)
    seeds = _infer_tree(td, over={"permuted": "permuted+refused"})
    agg = _agg(td, seeds, "A6")
    g = agg["groups"]["infer"]
    assert agg["amendment_level"] == "A6"
    assert g["verdict"] == "INCOMPLETE" and g["n_defined"] == 0
    row = g["seeds"][seeds[0]]
    assert row["tracks"] is None and row["tracks_as_scored_pre_A6"] is False
    assert row["permuted_refused"] == ["t_heron_hunt"]
    assert row["permuted_acquisition"]["t_heron_hunt"] == {"refused": True, "carried": False}
    assert row["permuted_acquisition"]["t_frog_eat"]["refused"] is False
    assert any("(A6)" in r for r in row["undefined_reasons"])
    assert agg["verdict"].startswith("UNDEFINED") and not agg["GO"]
    assert any("amendment A6" in r for r in agg["undefined_reasons"])
    assert "C2 permuted-exposure acquisition (amendment A6" in SG.report(agg)


def test_a6_exposure_blind_brain_that_took_the_teach_still_fails(tmp_path):
    td = str(tmp_path)
    seeds = _infer_tree(td, over={"permuted": "permuted+blind"})
    g = _agg(td, seeds, "A6")["groups"]["infer"]
    assert g["verdict"] == "FAIL" and g["n_defined"] == len(seeds) and g["n_permuted_tracks"] == 0
    assert any("not exposure-driven" in w for w in g["why"])
    assert g["seeds"][seeds[0]]["permuted_refused"] == []


def test_a6_an_echo_that_drops_the_triple_is_not_a_refusal(tmp_path):
    """The permuted arm answering like the intact brain (echoing the INTACT triples on the permuted teach turns, no
    abstain) is scored as having received the teach: it stays FAIL, never UNDEFINED."""
    td = str(tmp_path)
    seeds = _infer_tree(td, over={"permuted": "intact"})
    g = _agg(td, seeds, "A6")["groups"]["infer"]
    row = g["seeds"][seeds[0]]
    assert g["verdict"] == "FAIL" and row["permuted_refused"] == []
    assert row["permuted_acquisition"]["t_heron_hunt"] == {"refused": False, "carried": False}


def test_a6_refusal_never_masks_a_fail_another_clause_decides(tmp_path):
    td = str(tmp_path)
    seeds = _infer_tree(td, over={"permuted": "permuted+refused", "intact_a": "one_hop", "intact_b": "one_hop"})
    g = _agg(td, seeds, "A6")["groups"]["infer"]
    row = g["seeds"][seeds[0]]
    assert g["verdict"] == "FAIL" and g["n_defined"] == len(seeds)
    assert row["tracks"] is None and not any("(A6)" in r for r in row["undefined_reasons"])
    assert any("capability absent" in w for w in g["why"])
    td2 = str(tmp_path / "inert")
    seeds = _infer_tree(td2, over={"permuted": "permuted+refused", "lesion_chain_route": "intact"})
    g = _agg(td2, seeds, "A6")["groups"]["infer"]
    assert g["verdict"] == "FAIL" and any("not load-bearing" in w for w in g["why"])


def test_a6_capable_brain_passes_and_matches_a5(tmp_path):
    td = str(tmp_path)
    seeds = _infer_tree(td, realistic=True)
    a5, a6 = _agg(td, seeds, "A5"), _agg(td, seeds, "A6")
    assert a5["groups"]["infer"]["verdict"] == a6["groups"]["infer"]["verdict"] == "PASS"
    assert a6["groups"]["infer"]["seeds"][seeds[0]]["permuted_refused"] == []


def test_a6_touches_no_other_group_and_no_bar():
    assert SG.A6_INFER_HOP_TEACHES == ("t_heron_hunt", "t_frog_eat")
    assert (SG.ASK_RATIO_BAR, SG.ASK_GAP_FLOOR, SG.ASK_MOAT_MIN_DEFINED) == (2.0, 2.0 / 3.0, 1)
    assert [l for l, _m, r in SG.GROUPS["infer"]["permuted_turns"] if r == "teach"][:2] == list(SG.A6_INFER_HOP_TEACHES)


def test_unknown_level_is_refused(tmp_path):
    with pytest.raises(ValueError):
        SG.aggregate(str(tmp_path), seeds=(1,), amendment_level="A7")


def test_the_10_26_gate_reads_as_filed_at_a5_and_fails_on_c1_at_a6():
    """Ruling (Claude, 2026-11-01), criterion-only variant: at A6 C2 is INCOMPLETE (the refused permuted teach), and the
    STAGE is FAIL because C1 FAILs on defined data. An undefined C2 never hides a defined C1 failure."""
    if not os.path.exists(os.path.join(_GATE, "s42", "infer__permuted.json")):
        pytest.skip("10-26 gate shards not checked out")
    a5 = SG.aggregate(_GATE)
    assert a5["verdict"] == "FAIL" and a5["criteria"] == {"C1": "FAIL", "C2": "FAIL", "C3": "FAIL", "C4": "FAIL", "C5": "PASS"}
    assert "stage_rule" not in a5 and "criteria_failed" not in a5
    a6 = SG.aggregate(_GATE, amendment_level="A6")
    assert a6["criteria"]["C2"] == "INCOMPLETE" and a6["groups"]["infer"]["n_defined"] == 0
    assert {k: v for k, v in a6["criteria"].items() if k != "C2"} == {"C1": "FAIL", "C3": "FAIL", "C4": "FAIL", "C5": "PASS"}
    assert a6["verdict"] == "FAIL" and a6["GO"] is False
    assert "C1" in a6["criteria_failed"] and "C2" not in a6["criteria_failed"]
    assert a6["groups"]["conv"]["verdict"] == "FAIL" and a6["groups"]["conv"]["n_defined"] == 6
    assert any("amendment A6" in r for r in a6["undefined_reasons"])      # the unmet precondition is still recorded
    assert "Stage rule at A6 (criterion-only" in SG.report(a6)
    for s, row in a6["groups"]["infer"]["seeds"].items():
        assert row["permuted_refused"] == ["t_heron_hunt"], s


# ── the stage rule at A6 (Claude's ruling 2026-11-01: criterion-only) on synthetic six-seed x five-group brains ─────
def _stage(td, over_by_group=None, conv_random=False, level="A6"):
    rng = np.random.default_rng(0)
    data = C.make_stage(rng, over_by_group)
    if conv_random:
        data["conv"] = C.make_group(rng, "conv", degenerate="random")
    C.write_tree(td, data)
    seeds = sorted(next(iter(data.values())))
    return SG.aggregate(td, seeds=seeds, informational=False, amendment_level=level)


_REFUSED = {"permuted": "permuted+refused"}
_C1_GAP = {"intact_a": "no_elaboration", "intact_b": "no_elaboration"}


def test_a6_stage_fails_when_c1_fails_and_c2_is_incomplete(tmp_path):
    agg = _stage(str(tmp_path), {"conv": _C1_GAP, "infer": _REFUSED})
    assert agg["criteria"]["C1"] == "FAIL" and agg["criteria"]["C2"] == "INCOMPLETE"
    assert agg["verdict"] == "FAIL" and agg["GO"] is False and agg["criteria_failed"] == ["C1"]
    assert any("amendment A6" in r for r in agg["undefined_reasons"])


def test_a6_stage_is_undefined_when_only_c2_is_incomplete(tmp_path):
    agg = _stage(str(tmp_path), {"infer": _REFUSED})
    assert agg["criteria"]["C2"] == "INCOMPLETE"
    assert all(v == "PASS" for k, v in agg["criteria"].items() if k != "C2")
    assert agg["verdict"].startswith("UNDEFINED") and agg["GO"] is False and agg["criteria_failed"] == []


def test_a6_stage_passes_the_capable_brain_and_a5_keeps_its_rule(tmp_path):
    agg = _stage(str(tmp_path / "a6"))
    assert agg["verdict"] == "PASS" and agg["GO"] is True
    a5 = _stage(str(tmp_path / "a5"), {"conv": _C1_GAP, "infer": _REFUSED}, level="A5")
    assert a5["verdict"] == "FAIL" and a5["criteria"]["C2"] == "FAIL" and "stage_rule" not in a5


# ── pool lines: the gate lines unchanged; the dev-preview options ────────────────────────────────────────────────────
def _emit_sha(**kw):
    txt = "".join("%s\t%s\n" % (c, w) for c, w in SG.pool_job_lines(_SHA, **kw))
    return hashlib.sha256(txt.encode()).hexdigest()


def test_gate_pool_lines_are_byte_identical_to_the_pre_a6_runner():
    # sha256 of `--emit-pool-jobs 41d76bdbe... [--seeds 1,2,3]` stdout at runner 25e360740 (before this change)
    assert _emit_sha() == "55034ef2bafa9af30685c2ac90c41d20af5f01f7e1f5a6ec28d34656ae326b1c"
    assert _emit_sha(seeds=(1, 2, 3)) == "b4e804df56b35f699cafae0cda6d1b13893d14b5d557410803363725229b3ccf"


def test_dev_preview_lines_carry_the_env_the_groups_and_no_gate_seed_flags():
    lines = SG.pool_job_lines(_SHA, seeds=(1, 2, 3), groups=("inner", "emo"),
                              extra_env={"SIM_STABLE_INTEGRATION": "1", "SIM_STABLE_RECAL": "1"},
                              out_dir="research/findings/raw/_stage2_preview_1101/stable", reason="RULING x: y")
    assert len(lines) == 3 * (len(SG.gate_arms_for("inner")) + len(SG.gate_arms_for("emo"))) == 36
    for c, w in lines:
        assert "CUDA_VISIBLE_DEVICES= SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL=1 .venv/bin/python" in c
        assert "--allow-gate-seed" not in c and "--prereg" not in c
        assert "--out research/findings/raw/_stage2_preview_1101/stable/s" in c
        assert w.startswith("RULING x: y; 41d76bdbe seed ") and w.endswith("mem_gb=9 max_age_h=240")
        assert "--group inner" in c or "--group emo" in c
    gate = SG.pool_job_lines(_SHA, seeds=(42,), groups=("inner",), extra_env={"X": "1"})
    assert all("--allow-gate-seed --prereg %s" % SG.PREREG in c for c, _w in gate)       # a gate seed keeps its guard


def test_committed_stable_preview_lines_match_the_emitter():
    p = os.path.join(_REPO, "research", "coordination", "stage2-stable-preview-pool-lines.tsv")
    if not os.path.exists(p):
        pytest.skip("lines file not present")
    lines = SG.pool_job_lines(_SHA, seeds=(1, 2, 3), groups=("inner", "emo"),
                              extra_env={"SIM_STABLE_INTEGRATION": "1", "SIM_STABLE_RECAL": "1"},
                              out_dir="research/findings/raw/_stage2_preview_1101/stable",
                              reason="RULING stage2-retake-1101: stable+recal dev preview incl. lesion arms")
    with open(p) as f:
        assert f.read() == "".join("%s\t%s\n" % (c, w) for c, w in lines)
