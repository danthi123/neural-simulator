"""Stage-2 exit gate, amendment A4 (docs/drafts/stage2-exit/AMENDMENT-A4.md, Claude 2026-10-13): the C3 `ask` group's
order confound and failed-acquisition moat, and the C1 / C3 named-gap reporting rule. No brain build: synthetic shards in
the worker's format through the registered scorer, plus the committed dev-seed-7 shards where they are checked out."""
import json
import os

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import stage2_exit_gate as SG

_DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "findings", "raw",
                    "_stage2_exit", "dev", "s7")

# the registered `ask` turn table exactly as the prereg froze it (2026-09-30); A4 only names its blocks
_REGISTERED_ASK = [
    ("t_yak", "the yak eats the grass"), ("t_moose", "the moose eats the leaf"), ("t_goat", "the goat eats the hay"),
    ("g_eel", "what does the eel chase"), ("m_yak", "what does the yak chase"), ("g_crab", "what does the crab chase"),
    ("m_moose", "what does the moose chase"), ("g_newt", "what does the newt chase"), ("m_goat", "what does the goat chase"),
    ("lp_ibex_1", "what does the ibex chase"), ("t_ibex", "the ibex chases the marmot"), ("lp_ibex_2", "what does the ibex chase"),
    ("u_toad_1", "what does the toad chase"), ("c_dog_1", "what does the dog chase"), ("u_toad_2", "what does the toad chase"),
    ("c_dog_2", "what does the dog chase"), ("u_toad_3", "what does the toad chase"), ("c_dog_3", "what does the dog chase"),
    ("u_toad_4", "what does the toad chase"),
]


def _dev(name):
    p = os.path.join(_DEV, name)
    if not os.path.exists(p):
        pytest.skip("dev-7 shard %s not present" % name)
    with open(p) as f:
        return json.load(f)


# ── (b) the reversed-order arm ───────────────────────────────────────────────────────────────────────────────────────
def test_the_registered_ask_order_is_unchanged_by_the_block_refactor():
    assert [(l, m) for l, m, _r in SG.GROUPS["ask"]["turns"]] == _REGISTERED_ASK
    assert [(l, m) for l, m, _r in SG.turns_for("ask", 42, "intact_a")] == _REGISTERED_ASK
    assert [(l, m) for l, m, _r in SG.turns_for("ask", 42, "lesion_graded.rep0")] == _REGISTERED_ASK


def test_reversed_arm_reverses_the_probe_blocks_keeps_each_block_and_the_teach_block_first():
    rev = SG.turns_for("ask", 42, "reversed")
    assert sorted(rev) == sorted(SG.GROUPS["ask"]["turns"])                      # the same turns, nothing added or lost
    assert rev == SG._ASK_TEACH + SG._ASK_UNLEARNABLE + SG._ASK_LEARNABLE + SG._ASK_DISCRIM
    assert SG.GROUPS["ask"]["turns"] == SG._ASK_TEACH + SG._ASK_DISCRIM + SG._ASK_LEARNABLE + SG._ASK_UNLEARNABLE
    labels = [t[0] for t in rev]
    assert labels.index("t_ibex") == labels.index("lp_ibex_1") + 1 == labels.index("lp_ibex_2") - 1
    assert max(labels.index(t) for t in ("t_yak", "t_moose", "t_goat")) < min(labels.index(l) for l in ("m_yak", "m_moose", "m_goat"))


def test_reversed_is_a_counted_production_arm_only_in_ask_and_da_lesion_is_informational():
    assert SG.arm_spec("ask", "reversed") == ({}, None, "order")
    assert "reversed" in SG.arms_for("ask", with_informational=False)
    assert "da_lesion" in SG.arms_for("ask") and "da_lesion" not in SG.arms_for("ask", with_informational=False)
    assert SG.arm_spec("ask", "da_lesion") == ({"BRAIN_CURIOSITY_DA_LESION": "1"}, None, "informational")
    for g in SG.GROUPS:
        if g != "ask":
            assert SG.order_arms(g) == () and "reversed" not in SG.arms_for(g)


def test_pool_lines_carry_one_reversed_and_one_da_lesion_line_per_seed():
    lines = SG.pool_job_lines("e" * 40)
    rev = [c for c, _w in lines if "--group ask --arm reversed " in c]
    dal = [c for c, _w in lines if "--group ask --arm da_lesion " in c]
    assert len(rev) == len(dal) == len(SG.GATE_SEEDS)
    assert all(SG.shard_path(SG.GATE_OUT_DIR, s, "ask", "reversed") in " ".join(rev) for s in SG.GATE_SEEDS)
    no_info = SG.pool_job_lines("e" * 40, informational=False)
    assert any("--arm reversed " in c for c, _w in no_info) and not any("--arm da_lesion " in c for c, _w in no_info)


def _tree(td, over=None, unacquired=(), seeds=(1, 2), drop=()):
    kinds = {"intact_a": "intact", "intact_b": "intact", "lesion_curiosity": "lesion", "lesion_curiosity.rep0": "lesion",
             "lesion_graded": "graded_lesion", "lesion_graded.rep0": "graded_lesion", "permuted": "permuted",
             "reversed": "intact", "da_lesion": "intact"}
    kinds.update(over or {})
    for seed in seeds:
        for arm, kind in kinds.items():
            p = SG.shard_path(td, seed, "ask", arm)
            if arm in drop:
                if os.path.exists(p):
                    os.remove(p)
                continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            d = dict(SG._fake_shard("ask", SG._synthetic_ask(kind, unacquired=unacquired), arm=arm), seed=seed, arm=arm,
                     corpus_present=True)
            with open(p, "w") as f:
                json.dump(d, f)
    return SG.aggregate(td, seeds=seeds)


def test_a_brain_correct_only_in_the_registered_order_fails_capability_absent(tmp_path):
    """The order confound (HANDOFF #16): the default-ON DA/engagement threshold silences the asks after two, so the
    position of a block, not the topic's history, can decide a clause. Every clause must hold in BOTH orders."""
    g = _tree(str(tmp_path), over={"reversed": "no_lp"})["groups"]["ask"]
    assert g["verdict"] == "FAIL" and g["n_correct_by_order"] == {"registered": 2, "reversed": 0}
    assert any("capability absent on 2 of 2" in w and "reversed 0 of 2" in w for w in g["why"]), g["why"]
    assert g["sub_clauses_n_true"]["reversed_lp_unlearnable"] == 0 and g["sub_clauses_n_true"]["lp_unlearnable"] == 2
    # and the mirror: correct only in the reversed order
    g = _tree(str(tmp_path), over={"intact_a": "no_lp", "intact_b": "no_lp"})["groups"]["ask"]
    assert g["verdict"] == "FAIL" and g["n_correct_by_order"] == {"registered": 0, "reversed": 2}


def test_a_missing_reversed_arm_is_undefined_never_a_fail(tmp_path):
    agg = _tree(str(tmp_path), drop=("reversed",))
    g = agg["groups"]["ask"]
    assert g["verdict"] == "INCOMPLETE" and g["n_defined"] == 0
    assert any("reversed arm missing" in r for r in g["seeds"][1]["undefined_reasons"])
    assert any(m.endswith("ask__reversed.json") for m in agg["missing_shards"])


def test_the_reversed_arm_must_be_a_valid_measurement_like_every_counted_arm(tmp_path):
    td = str(tmp_path)
    _tree(td)
    p = SG.shard_path(td, 2, "ask", "reversed")
    d = json.load(open(p))
    d["sources"] = ["tiny-demo"]
    json.dump(d, open(p, "w"))
    g = SG.aggregate(td, seeds=(1, 2))["groups"]["ask"]
    assert g["verdict"] == "INCOMPLETE" and any("reversed: unexpected brain source" in r for r in g["seeds"][2]["undefined_reasons"])


# ── (a) a taught moat whose teach was not acquired ────────────────────────────────────────────────────────────────────
def test_teach_acquired_reads_the_teach_echo_and_rejects_the_role_binding_abstain():
    f = ["goat", "eat", "hay"]
    assert SG.teach_acquired({"answer": "the goat eats the hay", "abstained": False, "recalled_svo": f}, f)
    assert not SG.teach_acquired({"answer": SG._ROLE_ABSTAIN, "abstained": True, "recalled_svo": None}, f)
    assert not SG.teach_acquired({"answer": "x", "abstained": True, "recalled_svo": f}, f)            # abstained
    assert not SG.teach_acquired({"answer": "x", "abstained": False, "recalled_svo": ["hay", "eat", "goat"]}, f)
    assert not SG.teach_acquired({"_error": "boom"}, f) and not SG.teach_acquired(None, f)


def test_dev7_goat_teach_was_not_acquired_so_m_goat_is_undefined_and_the_rate_is_over_yak_and_moose():
    """Reported, not tuned to: dev seed 7 abstained on `the goat eats the hay` (role binding)."""
    ia = _dev("ask__intact_a.json")
    acq = SG.acquired_topics(ia)
    assert acq == {"yak": True, "moose": True, "goat": False, "ibex": True}
    s = SG.score_ask(ia)
    assert s["moats_undefined"] == ["m_goat"] and s["n_moats_defined"] == 2 and s["moat_rate_defined"]
    assert s["moat_items"]["m_goat"] == {"topic": "goat", "teach_acquired": False, "asked": False, "counted": False}
    lg = _dev("ask__lesion_graded.json")
    assert SG.effective_scored_labels("ask", "lesion_graded", ia, lg) == ["m_yak", "m_moose"]


def test_an_unacquired_moat_is_never_counted_as_novel():
    """The gap set stays the three never-taught topics; an unacquired moat that the brain (correctly) asks about joins
    neither rate."""
    r = SG._synthetic_ask("intact", unacquired=("t_goat",))
    r["g_newt"] = SG._fake(SG._AB, True)                                   # one real gap not asked
    s = SG.score_ask(SG._fake_shard("ask", r))
    assert s["ask_rate_gap"] == pytest.approx(2.0 / 3.0) and s["ask_rate_moat"] == 0.0
    assert s["asked_by_label"]["m_goat"] is True and s["moats_undefined"] == ["m_goat"]


def test_an_unacquired_moat_drops_out_of_the_graded_lesion_scored_turns_only():
    ia = SG._fake_shard("ask", SG._synthetic_ask("intact", unacquired=("t_goat",)))
    lg = SG._fake_shard("ask", SG._synthetic_ask("graded_lesion"), arm="lesion_graded")
    assert SG.effective_scored_labels("ask", "lesion_graded", ia, lg) == ["m_yak", "m_moose"]
    assert SG.effective_scored_labels("ask", "lesion_curiosity", ia, lg) == SG.SCORED_LABELS["ask"]["lesion_curiosity"]
    for g, d in SG.SCORED_LABELS.items():          # every other group: identity
        if g != "ask":
            for les, labs in d.items():
                assert SG.effective_scored_labels(g, les, {}, {}) == labs


def test_no_acquired_moat_in_any_counted_arm_makes_the_seed_undefined_with_the_a4_precondition(tmp_path):
    for arm_over in ({}, {"reversed": "intact"}):
        agg = _tree(str(tmp_path), over=arm_over, unacquired=("t_yak", "t_moose", "t_goat"))
        g = agg["groups"]["ask"]
        assert g["verdict"] == "INCOMPLETE" and g["n_defined"] == 0
        pre = [c for c in agg["preconditions"] if "A4 (a)" in c["name"]]
        assert len(pre) == 1 and pre[0]["ok"] is False


def test_the_permuted_arm_reads_its_own_teaching_acquisition(tmp_path):
    """Permuted teaching acquires eel/crab/newt (its own moats); none acquired there -> UNDEFINED, not 'not
    exposure-driven' (an acquisition gap is not a curiosity result)."""
    td = str(tmp_path)
    agg = _tree(td)
    assert agg["groups"]["ask"]["verdict"] == "PASS"
    for seed in (1, 2):
        p = SG.shard_path(td, seed, "ask", "permuted")
        d = json.load(open(p))
        for l in ("t_yak", "t_moose", "t_goat"):
            d["responses"][l] = SG._fake(SG._ROLE_ABSTAIN, True)
        json.dump(d, open(p, "w"))
    g = SG.aggregate(td, seeds=(1, 2))["groups"]["ask"]
    assert g["verdict"] == "INCOMPLETE" and any("permuted: taught moat acquired on 0 of 3" in r
                                                for r in g["seeds"][1]["undefined_reasons"])


# ── (c) the DA-lesion arm and the reporting rules ────────────────────────────────────────────────────────────────────
def test_the_da_lesion_arm_is_reported_and_never_moves_the_verdict(tmp_path):
    for kind in ("intact", "lesion", "no_lp"):
        g = _tree(str(tmp_path), over={"da_lesion": kind})["groups"]["ask"]
        assert g["verdict"] == "PASS"
        assert set(g["seeds"][1]["informational"]["da_lesion"]) >= {"discriminates", "lp_learnable", "lp_unlearnable"}


def test_named_gaps_are_printed_for_a_c1_or_c3_fail_only(tmp_path):
    agg = _tree(str(tmp_path), over={"reversed": "no_lp"})
    assert agg["groups"]["ask"]["named_gaps"] == ["global DA engagement threshold; no per-topic learning-progress signal"]
    assert all(d["named_gaps"] == [] for g, d in agg["groups"].items() if g != "ask")     # INCOMPLETE: nothing named
    rep = SG.report(agg)
    assert "Named gaps (amendment A4" in rep and "global DA engagement threshold" in rep
    assert "  - reversed order: gap" in rep and "  - da_lesion (informational, never counted)" in rep
    agg = _tree(str(tmp_path))
    assert agg["groups"]["ask"]["named_gaps"] == [] and "Named gaps" not in SG.report(agg)
    assert SG.A4_NAMED_GAPS["conv"] == ("wh-locative question parse",
                                        "locative frame storage (the teach echo drops the preposition)",
                                        "discourse elaboration for 'tell me more'")


def test_the_report_reads_an_aggregate_written_before_a4():
    """`--report` on an older aggregate.json (no named_gaps / orders keys) still renders."""
    agg = {"verdict": "FAIL", "criteria": {"C1": "FAIL"}, "missing_shards": [],
           "groups": {g: {"criterion": SG.GROUPS[g]["criterion"], "verdict": "FAIL", "n_correct": 0, "n_defined": 0,
                          "n_lesion_effect": {}, "n_reproduced": {}, "n_permuted_tracks": 0, "why": [],
                          "sub_clauses_n_true": {}, "seeds": {}} for g in SG.GROUPS}}
    assert "Stage-2 exit verdict: FAIL" in SG.report(agg)


def test_no_bar_constant_moved():
    assert SG.ASK_RATIO_BAR == 2.0 and SG.ASK_GAP_FLOOR == pytest.approx(2.0 / 3.0)
    assert SG.SCORED_LABELS["ask"] == {"lesion_curiosity": ["g_eel", "g_crab", "g_newt", "lp_ibex_1", "u_toad_1"],
                                       "lesion_graded": ["m_yak", "m_moose", "m_goat"]}
    assert SG.ASK_MOAT_MIN_DEFINED == 1
