"""Tests for the curiosity learning-progress trace de-risk (docs/drafts/curiosity-lp/PLAN.md).

Pure-logic checks (no bridge): the battery world, the turn orders, the scorer and the aggregate's all-8-seeds rule,
the pre-registered bars and the Stage-2 turn order copied verbatim. The bridge itself is exercised by the runner's
--design / --battery modes under the brain lock, not here.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners._curiosity_lp_trace_derisk as M  # noqa: E402


def test_selftest_passes():
    ok, checks = M.selftest()
    assert ok, {k: v for k, v in checks.items() if not v}


def test_stage2_ask_turn_order_is_verbatim():
    labels = [t[0] for t in M.ASK_TURNS]
    assert labels == ["t_yak", "t_moose", "t_goat", "g_eel", "m_yak", "g_crab", "m_moose", "g_newt", "m_goat",
                      "lp_ibex_1", "t_ibex", "lp_ibex_2", "u_toad_1", "c_dog_1", "u_toad_2", "c_dog_2", "u_toad_3",
                      "c_dog_3", "u_toad_4"]
    assert [t[1] for t in M.ASK_PERMUTED_TURNS[:3]] == ["the eel eats the grass", "the crab eats the leaf",
                                                         "the newt eats the hay"]
    assert M.ASK_PERMUTED_TURNS[3:] == M.ASK_PROBES


def test_world_gaps_and_moats_abstain_controls_recall():
    w = M.BatteryWorld()
    for _l, msg, _k in M.ASK_TURNS[:3]:
        w.store(M.parse_teach(msg))
    for label in M.GAPS + M.MOATS + ("lp_ibex_1", "u_toad_1"):
        msg = dict((t[0], t[1]) for t in M.ASK_TURNS)[label]
        assert w.recall(*M.parse_probe(msg)) is None, label
    assert w.recall("dog", "chase") == "cat"
    vocab = w.vocab()
    assert {"yak", "moose", "goat", "grass", "leaf", "hay", "dog", "chase"} <= vocab
    assert not ({"eel", "crab", "newt", "ibex", "toad"} & vocab)


def test_perm_order_keeps_constrained_families_in_order():
    for s in M.BATTERY_SEEDS:
        lab = [t[0] for t in M.perm_order_turns(s)]
        assert lab[:3] == ["t_yak", "t_moose", "t_goat"]
        assert lab.index("lp_ibex_1") < lab.index("t_ibex") < lab.index("lp_ibex_2")
        assert [l for l in lab if l in M.TOAD] == list(M.TOAD)
        assert sorted(lab) == sorted(t[0] for t in M.ASK_TURNS)


def test_bars_and_counted_arms_preregistered():
    assert set(M.COUNTED_ARMS) <= set(M.ARMS)
    assert {"shared_slice", "lesion_reset"} <= set(M.ARMS) - set(M.COUNTED_ARMS)
    for k in ("B1_first_novel_asked", "B1b_every_first_gap_asked", "B2_stop_by_4th", "B3_answered_not_reasked",
              "B4_moats_silent", "B5_lesion_continues", "B6_perm_order", "B7_perm_exposure", "B8_null",
              "B9_drive_lesion"):
        assert k in M.BARS
    assert M.BATTERY_SEEDS == tuple(range(1, 9)) and M.DESIGN_SEED not in M.BATTERY_SEEDS
    assert all(1 <= s <= 15 for s in M.BATTERY_SEEDS + (M.DESIGN_SEED,))


def test_aggregate_fails_when_any_seed_misses_a_bar():
    good = {k: True for k in M.BAR_KEYS}
    good["toad_asks"] = {"intact": 1, "lesion_habituation": 4, "intact_b": 1}
    per = {s: dict(good) for s in M.BATTERY_SEEDS}
    assert M.aggregate(per)["verdict"] == "GO"
    per[5] = dict(good, B2_stop_by_4th=False)
    agg = M.aggregate(per)
    assert agg["verdict"] == "NO-GO" and agg["bars"]["B2_stop_by_4th"]["n_true"] == 7


def test_no_ou_flag_is_additive_and_default_off():
    """RULING c3-methods-1018: one additive flag, default off. Flag-off executes no new cfg write, so
    flag-off output stays byte-identical to the frozen method-1 battery (the artifact gains the `no_ou`
    key only when the flag is ON)."""
    import inspect
    import subprocess
    import sys

    for fn in (M.build_lp_bridge, M.run_arm, M.run_battery, M.CuriosityLPOrgan.__init__):
        p = inspect.signature(fn).parameters["no_ou"]
        assert p.default is False, fn.__name__
    out = subprocess.run([sys.executable, "-m", "research.runners._curiosity_lp_trace_derisk", "--help"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "--no-ou" in out.stdout
