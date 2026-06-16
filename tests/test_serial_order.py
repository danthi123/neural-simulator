"""Unit tests for the de-risked serial-order (sentence-generation de-templating) mechanisms (CYCLE 104-106).

Pure-numpy, CPU-fast (no bridge): pins the competitive-queuing serial-order generator (`CQSerialOrder`) and the
frame-conditioned variant (`FrameCQ`) so the de-risked GO mechanisms don't silently regress. The learned primacy
gradient (~2.4 vs 0 after 12 learn steps) dominates the WTA noise (0.25), so the emitted order is deterministic.
"""
import numpy as np

from research.runners._phaseB_serial_order_cq_derisk import CQSerialOrder
from research.runners._phaseB_serial_order_multiframe_derisk import FrameCQ, FRAMES


def test_cq_learns_svo_primacy_and_order():
    """After learning the SVO order from the teacher, the primacy gradient ranks agent > action > patient, and
    the choice-WTA read-out emits the fillers in that order."""
    cq = CQSerialOrder(seed=42)
    for _ in range(12):
        cq.learn([0, 1, 2])
    assert cq.prim[0] > cq.prim[1] > cq.prim[2]                  # learned primacy gradient (deterministic)
    fillers = {0: 5, 1: 9, 2: 3}
    assert cq.emit(fillers, [0, 1, 2], np.random.default_rng(0)) == [5, 9, 3]


def test_cq_untrained_does_not_fix_order():
    """An untrained CQ has ~flat primacy, so it does NOT reliably emit the SVO order -- the learning is what
    produces the order (mirrors the de-risk's no-learning control)."""
    cq = CQSerialOrder(seed=42)                                  # untrained
    fillers = {0: 5, 1: 9, 2: 3}
    orders = {tuple(cq.emit(fillers, [0, 1, 2], np.random.default_rng(s))) for s in range(8)}
    assert len(orders) > 1                                       # not pinned to one order without learning


def test_framecq_learns_distinct_frame_orders():
    """The frame-conditioned CQ learns DIFFERENT orders for F0 vs F1 and orders the SAME fact differently by
    frame (the cross-frame separation that is the seed of syntax)."""
    cq = FrameCQ(seed=42)
    for _ in range(12):
        for f, order in FRAMES.items():
            cq.learn(f, order)
    fillers = {0: 5, 1: 9, 2: 3}
    o0 = cq.emit(0, fillers, np.random.default_rng(0))           # F0 = [0,1,2] -> [5,9,3]
    o1 = cq.emit(1, fillers, np.random.default_rng(0))           # F1 = [2,0,1] -> [3,5,9]
    assert o0 == [fillers[r] for r in FRAMES[0]]
    assert o1 == [fillers[r] for r in FRAMES[1]]
    assert o0 != o1                                              # frame-conditioned: same fact, different order
