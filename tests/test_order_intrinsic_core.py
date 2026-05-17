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
