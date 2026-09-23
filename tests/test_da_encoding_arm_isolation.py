"""Order-independence of the DA-encoding natural-drive arms (review fix 3, 2026-09-23).

The v2 runner's arms depended on run order: the spiking write-gain reader
(`_da_write_gain_spiking_derisk._READERS`) is built lazily inside the FIRST arm that needs it, and later reads draw
OU noise from a global RNG stream the reader snapshot does not restore. So the same arm, fed the same DA trace, got
different write gains depending on whether it ran first or second in the process.

`test_v2_runner_is_order_dependent` pins the defect on the pre-fix runner (it asserts the gains DIFFER).
`test_v3_run_arm_is_order_independent` is the fix's proof: the same assertion with `==` passes on the v3 runner.
Synthetic DA traces (no DA workspace build) keep this to four composer builds per test."""
import os

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")


def _trace(condition, fact_da, other_da):
    from research.runners._da_encoding_natural_drive_persistence import conversation
    return [{"da": (fact_da if fi is not None else other_da), "fact": fi} for (_t, fi) in conversation(condition)]


# DA away from the gain clamps (1.0 floor / 3.0 ceiling) so read noise shows in the gain
TR_A = _trace("salient", 0.70, 0.56)
TR_B = _trace("neutral", 0.58, 0.54)


def _gains_first_and_second(run_a, run_b, key):
    from research.runners import _da_write_gain_spiking_derisk as W
    W._READERS.clear()
    first = run_a()
    run_b()
    W._READERS.clear()
    run_b()
    second = run_a()
    return first[key], second[key], first, second


def test_v2_runner_is_order_dependent():
    """The defect, pinned: on the PRE-FIX runner, arm A's write gains change with its position in the process."""
    from research.runners import _da_encoding_natural_drive_persistence as V2
    g1, g2, _, _ = _gains_first_and_second(
        lambda: V2.run_arm(7, "salient", "intact", TR_A, 1.0, 1.5),
        lambda: V2.run_arm(7, "neutral", "intact", TR_B, 1.0, 1.5), "write_gains")
    assert g1 != g2, "expected the v2 runner to be order-dependent (the reviewed defect); got identical gains"


def test_v3_run_arm_is_order_independent():
    """The fix: arm state is reset at arm start, so arm A is identical whether it runs first or second."""
    from research.runners import _da_encoding_natural_drive_synaptic as V3
    g1, g2, a1, a2 = _gains_first_and_second(
        lambda: V3.run_arm(7, "salient", "intact", TR_A, V3.PRIMARY),
        lambda: V3.run_arm(7, "neutral", "intact", TR_B, V3.PRIMARY), "write_gains_used")
    assert len(g1) == 4
    assert g1 == g2
    assert [t["a"] for t in a1["turn_log"]] == [t["a"] for t in a2["turn_log"]]
    assert a1["blocks"] == a2["blocks"]
    assert a1["delayed_24h"] == a2["delayed_24h"]
    assert a1["gamma"] == a2["gamma"]
