"""Tests for the curiosity LP slope-gate de-risk (C3 method 3; RULING c3-methods-1018b/c).

Pure-logic checks (no bridge): method 1's battery world and turn orders are reused verbatim, the slope/veto gate's
EMA math (flat slope accumulates, progress clears, decline doubles, cap holds, gate lesion never moves), the
B5-retargeted scorer is method 1's own B1-B9 code on a rekeyed shard dict, the aggregate keeps the all-8-seeds
rule, and the module does not leak into webapp/ or the production organ. The bridge itself is exercised by the
runner's --battery mode under the brain lock, not here.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners._curiosity_lp_slope_gate_derisk as M  # noqa: E402


def test_selftest_passes():
    ok, checks = M.selftest()
    assert ok, {k: v for k, v in checks.items() if not v}


def test_stage2_ask_turn_order_is_verbatim():
    labels = [t[0] for t in M.M1.ASK_TURNS]
    assert labels == ["t_yak", "t_moose", "t_goat", "g_eel", "m_yak", "g_crab", "m_moose", "g_newt", "m_goat",
                      "lp_ibex_1", "t_ibex", "lp_ibex_2", "u_toad_1", "c_dog_1", "u_toad_2", "c_dog_2", "u_toad_3",
                      "c_dog_3", "u_toad_4"]
    assert [t[1] for t in M.M1.ASK_PERMUTED_TURNS[:3]] == ["the eel eats the grass", "the crab eats the leaf",
                                                            "the newt eats the hay"]
    assert M.M1.ASK_PERMUTED_TURNS[3:] == M.M1.ASK_PROBES


def test_bars_and_counted_arms_b5_retargeted():
    assert set(M.COUNTED_ARMS) <= set(M.ARMS)
    assert {"shared_slice", "lesion_reset"} <= set(M.ARMS) - set(M.COUNTED_ARMS)
    assert "lesion_slope_gate" in M.ARMS and M.ARMS["lesion_slope_gate"] == {"slope_gate_off": True}
    assert "lesion_habituation" not in M.ARMS
    for k in ("B1_first_novel_asked", "B1b_every_first_gap_asked", "B2_stop_by_4th", "B3_answered_not_reasked",
              "B4_moats_silent", "B5_lesion_continues", "B6_perm_order", "B7_perm_exposure", "B8_null",
              "B9_drive_lesion"):
        assert k in M.BARS
    assert "lesion_slope_gate" in M.BARS["B5_lesion_continues"]
    assert M.BATTERY_SEEDS == tuple(range(1, 9)) and M.M1.DESIGN_SEED not in M.BATTERY_SEEDS


def test_preregistered_constants_are_frozen():
    assert (M.ALPHA_FAST, M.ALPHA_SLOW) == (0.5, 0.125)
    assert (M.SLOPE_EPS_HZ, M.VETO_STEP_HZ, VETO_DECLINE := M.VETO_STEP_DECLINE_HZ) == (5.0, 10.0, 20.0)
    assert M.VETO_MAX_HZ == 120.0 and M.M3_TAU_MS == 200.0
    assert M.W_TOPIC_ASK == 80
    assert M.build_slope_gate_bridge.__kwdefaults__ == {"tau_hab_ms": M.M3_TAU_MS, "ou_process": False}


def test_gate_math():
    org = M.CuriositySlopeGateOrgan(7, w_topic_ask=M.W_TOPIC_ASK)
    # first sample initializes both EMAs (slope 0) and takes one veto step: the first probe is still asked
    assert org.gate_update("t", 35.0) == 0.0
    assert org._gate("t")["veto"] == M.VETO_STEP_HZ
    # flat series (probed, never answered) keeps accumulating
    for _ in range(3):
        org.gate_update("t", 35.0)
    assert org._gate("t")["veto"] == 4 * M.VETO_STEP_HZ
    # a teach jump (DA phasic-above-tonic Hz + the last want) is positive slope: veto-protected
    assert org.gate_update("t", 35.0 + 500.0) > M.SLOPE_EPS_HZ
    assert org._gate("t")["veto"] == 0.0
    # a declining series crosses the negative eps and vetoes at double step
    org.gate_update("d", 100.0)
    org.gate_update("d", 100.0)
    before = org._gate("d")["veto"]
    slope_down = org.gate_update("d", 35.0)
    assert slope_down is not None and slope_down < -M.SLOPE_EPS_HZ
    assert org._gate("d")["veto"] == before + M.VETO_STEP_DECLINE_HZ
    # the cap holds
    for _ in range(20):
        org.gate_update("c", 35.0)
    assert org._gate("c")["veto"] == M.VETO_MAX_HZ
    # B5's arm: the gate lesion never moves the veto
    org_l = M.CuriositySlopeGateOrgan(7, w_topic_ask=M.W_TOPIC_ASK, slope_gate_off=True)
    assert org_l.gate_update("t", 35.0) is None
    assert org_l._gate("t")["veto"] == 0.0


def test_gate_state_is_declared_in_header_and_recorded_shape():
    org = M.CuriositySlopeGateOrgan(7, w_topic_ask=M.W_TOPIC_ASK)
    org.gate_update("t", 12.0)
    snap = org.gate_snapshot()
    assert set(snap["t"]) == {"fast", "slow", "veto", "last_want", "n"}
    doc = M.__doc__ or ""
    assert "DECLARED HOST SCAFFOLD" in doc and "fast/slow LP state" in doc
    assert "RULING c3-methods-1018b" in doc and "RULING c3-methods-1018c" in doc


def test_scorer_is_method1_code_with_b5_retargeted():
    def fake(asked_labels, turns=None):
        turns = turns or M.M1.ASK_TURNS
        r = {}
        for l, _m, k in turns:
            if k == "teach":
                r[l] = {"kind": k, "asked": None}
            elif l == "lp_ibex_2":
                r[l] = {"kind": k, "asked": None, "recalled": "marmot"}
            elif l.startswith("c_dog"):
                r[l] = {"kind": k, "asked": None, "recalled": "cat"}
            else:
                r[l] = {"kind": k, "asked": l in asked_labels}
        return {"responses": r, "turn_order": [t[0] for t in turns]}

    from research.runners._curiosity_lp_trace_derisk import FIRST_GAP_PROBES, TOAD

    good = fake(set(FIRST_GAP_PROBES))
    no_stop = fake(set(FIRST_GAP_PROBES) | set(TOAD))
    shards = {"intact": good, "intact_b": good, "lesion_slope_gate": no_stop, "lesion_drive": fake(set()),
              "perm_order": good, "perm_exposure": fake({"m_yak", "m_moose", "m_goat", "lp_ibex_1", "u_toad_1"},
                                                        M.M1.ASK_PERMUTED_TURNS)}
    sc = M.score_seed(shards)
    assert all(sc[k] for k in M.BAR_KEYS)
    assert "lesion_slope_gate" in sc["toad_asks"] and "lesion_habituation" not in sc["toad_asks"]
    # identical to method 1's own scorer on the rekeyed shard dict (same B1-B9 code)
    rekeyed = dict(shards, lesion_habituation=no_stop)
    assert M.M1.score_seed(rekeyed) == M.M1.score_seed({**{k: v for k, v in shards.items()
                                                           if k != "lesion_slope_gate"},
                                                        "lesion_habituation": shards["lesion_slope_gate"]})
    # B2/B5 discriminate: a brain that never stops fails B2 and B5_differs
    stopless = {k: (no_stop if k not in ("intact_b", "lesion_drive") else
                    (fake(set()) if k == "lesion_drive" else good))
                for k in shards}
    sc2 = M.score_seed(stopless)
    assert not sc2["B2_stop_by_4th"] and not sc2["B5_differs"]


def test_aggregate_all_8_seeds_rule_and_b5_attribution():
    from research.runners._curiosity_lp_trace_derisk import FIRST_GAP_PROBES, TOAD

    def fake(asked_labels, turns=None):
        turns = turns or M.M1.ASK_TURNS
        r = {}
        for l, _m, k in turns:
            if k == "teach":
                r[l] = {"kind": k, "asked": None}
            elif l == "lp_ibex_2":
                r[l] = {"kind": k, "asked": None, "recalled": "marmot"}
            elif l.startswith("c_dog"):
                r[l] = {"kind": k, "asked": None, "recalled": "cat"}
            else:
                r[l] = {"kind": k, "asked": l in asked_labels}
        return {"responses": r, "turn_order": [t[0] for t in turns]}

    shards = {"intact": fake(set(FIRST_GAP_PROBES)), "intact_b": fake(set(FIRST_GAP_PROBES)),
              "lesion_slope_gate": fake(set(FIRST_GAP_PROBES) | set(TOAD)),
              "lesion_drive": fake(set()), "perm_order": fake(set(FIRST_GAP_PROBES)),
              "perm_exposure": fake({"m_yak", "m_moose", "m_goat", "lp_ibex_1", "u_toad_1"},
                                    M.M1.ASK_PERMUTED_TURNS)}
    per = {s: M.score_seed(shards) for s in M.BATTERY_SEEDS}
    agg = M.aggregate(per)
    assert agg["verdict"] == "GO"
    assert agg["bars"]["B5_attribution"]["treatment"] > 0 and agg["bars"]["B5_attribution"]["holds"]
    per[5] = dict(per[5], B2_stop_by_4th=False)
    agg2 = M.aggregate(per)
    assert agg2["verdict"] == "NO-GO" and agg2["bars"]["B2_stop_by_4th"]["n_true"] == 7
    agg3 = M.aggregate({s: per[s] for s in M.BATTERY_SEEDS[:7]})
    assert agg3["verdict"] == "NO-GO"


def test_module_never_imported_by_webapp_or_production_organ():
    """RULING c3-methods-1018b: a new module, never imported by webapp/ or the production organ, no sim/ edit."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, ".."))
    needle = "_curiosity_lp_slope_gate"
    for sub in ("webapp", "sim"):
        hits = []
        for base, _dirs, files in os.walk(os.path.join(root, sub)):
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(base, fn)
                    with open(p, encoding="utf-8", errors="replace") as f:
                        if needle in f.read():
                            hits.append(os.path.relpath(p, root))
        assert not hits, (sub, hits)
    prod = os.path.join(root, "research", "runners", "spiking_novelty_habituation_organ.py")
    with open(prod, encoding="utf-8", errors="replace") as f:
        assert needle not in f.read()