import numpy as np
from research.runners.order_intrinsic_core import decode_position_sweep

def test_decode_position_sweep_argmax_and_abstain():
    # per-position pool-firing-rate dicts (query_position output shape)
    per_pos = [
        {"A": 0.50, "B": 0.10, "C": 0.05},   # pos0 -> A (clears floor)
        {"A": 0.08, "B": 0.40, "C": 0.06},   # pos1 -> B
        {"A": 0.02, "B": 0.03, "C": 0.02},   # pos2 -> below floor 0.10
    ]
    decoded, conf, abstained = decode_position_sweep(per_pos, floor=0.10)
    assert decoded == ["A", "B", None]        # pos2 abstains, no confab
    assert abstained == [2]
    assert conf[0] == 0.50 and conf[1] == 0.40
    # deterministic + tie-break stable (first max)
    d2, _, _ = decode_position_sweep(
        [{"A": 0.2, "B": 0.2}], floor=0.0)
    assert d2 == ["A"]
    # empty -> empty
    assert decode_position_sweep([], floor=0.1) == ([], [], [])

from research.runners.order_intrinsic_core import control_max_floor

def test_control_max_floor_is_control_max_operating_point():
    enc = [0.50, 0.42, 0.61]          # encoded (intended) top-rates
    ctl = [0.20, 0.31, 0.18, 0.27]    # control (permuted/random) top-rates
    f = control_max_floor(enc, ctl)
    assert f == 0.31                  # the SAME operating criterion
                                      # that produced prior floors (control-max)
    assert control_max_floor([0.9], []) == 0.0   # no controls -> 0.0

from research.runners.order_intrinsic_core import order_intrinsic_verdict

def test_order_intrinsic_verdict_reuses_g1_bars():
    # true-order decoded correct; permuted-order decoded scrambled;
    # gate cleared -> PASS via UNMODIFIED g1_verdict (>=10% + >=0.5)
    v = order_intrinsic_verdict(
        true_decoded=[1, 2, 3], intended=[1, 2, 3],
        perm_decoded=[[2, 1, 3], [3, 2, 1]], gate_cleared=True)
    assert v["GATE"] == "PASS" and v["true_score"] == 1.0
    # gate not cleared -> FAIL regardless
    assert order_intrinsic_verdict([1,2,3],[1,2,3],[[2,1,3]],
                                   gate_cleared=False)["GATE"] == "FAIL"
    # true == permuted (no order learned) -> FAIL
    assert order_intrinsic_verdict([2,1,3],[1,2,3],[[2,1,3]],
                                   gate_cleared=True)["GATE"] == "FAIL"

from research.runners.order_intrinsic_core import aggregate_multiseed

def test_aggregate_multiseed_requires_all_seeds_pass():
    # per-seed list of per-prop verdict dicts (from order_intrinsic_verdict)
    seed_ok = [{"GATE":"PASS"},{"GATE":"PASS"}]
    seed_bad = [{"GATE":"PASS"},{"GATE":"FAIL"}]
    assert aggregate_multiseed([seed_ok, seed_ok, seed_ok])["GATE"] == "PASS"
    assert aggregate_multiseed([seed_ok, seed_bad, seed_ok])["GATE"] == "FAIL"
    assert aggregate_multiseed([seed_ok, seed_ok])["GATE"] == "FAIL"  # <3 seeds


# --- review-recommended adversarial coverage: lock the paramount
#     anti-cheat properties so a future regression cannot silently
#     break this terminal-verdict-gating core ---

def test_encoded_distribution_cannot_tune_the_floor():
    # THE paramount anti-cheat property: the floor is control-MAX
    # ONLY; the encoded (intended) distribution must never move it.
    ctl = [0.20, 0.31, 0.18, 0.27]
    base = control_max_floor([0.50, 0.42, 0.61], ctl)
    assert control_max_floor([1e9, 1e9, 1e9], ctl) == base
    assert control_max_floor([1e-9], ctl) == base
    assert control_max_floor([], ctl) == base
    assert base == 0.31

def test_verdict_no_permuted_controls_is_fail():
    # no permuted-ORDER contrast -> no order-evidence -> FAIL even
    # with a perfect true decode + gate cleared
    v = order_intrinsic_verdict([1, 2, 3], [1, 2, 3],
                                perm_decoded=[], gate_cleared=True)
    assert v["GATE"] == "FAIL"

def test_verdict_none_slots_penalised_not_free_pass():
    # an abstained (None) slot is a mismatch, not a clean stop:
    # 2/3 correct -> below the 0.5 floor only if <0.5; here 2/3>0.5
    # but a single correct (1/3) must be FAIL
    v_partial = order_intrinsic_verdict([1, None, 3], [1, 2, 3],
                                        [[2, 1, 3]], gate_cleared=True)
    assert abs(v_partial["true_score"] - (2.0 / 3.0)) < 1e-9
    v_one = order_intrinsic_verdict([1, None, None], [1, 2, 3],
                                    [[2, 1, 3]], gate_cleared=True)
    assert v_one["true_score"] < 0.5 and v_one["GATE"] == "FAIL"

def test_decode_at_floor_boundary_abstains():
    # rate EXACTLY at floor abstains (<= floor) -- the no-confab moat
    d, _, ab = decode_position_sweep([{"A": 0.20}], floor=0.20)
    assert d == [None] and ab == [0]

def test_aggregate_mixed_empty_seed_cannot_manufacture_pass():
    # a >=3-seed run where one seed contributed ZERO props must FAIL
    # (a zero-prop seed vacuously "all-pass" -- the closed hole)
    seed_ok = [{"GATE": "PASS"}, {"GATE": "PASS"}]
    r = aggregate_multiseed([seed_ok, seed_ok, []])
    assert r["GATE"] == "FAIL" and r["all_seeds_have_props"] is False
