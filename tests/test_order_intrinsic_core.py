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
