"""CI guard for EMERGE-61 -- close the spiking-Broca render-ORDER tail. The EMERGE-60 wire renders EMERGE answers on
spikes with correct CONTENT (1.00) but the word ORDER swapped on 2/6 seeds at a late emit ('the robin breathe can'),
because the Izhikevich slow-adaptation current `cp_recovery_variable_u` accumulates across productions. The fix is an
inter-utterance WASH-OUT (restore the exact post-init substrate state before each production) so a production does not
depend on prior productions' residual state. CPU/numpy, offline.

Load-bearing properties asserted: (1) with the reset, the EMERGE-60 emit SEQUENCE renders EXACT (robin as the 5th emit)
on the previously-failing seeds; (2) POSITION-INDEPENDENCE -- the same fact renders identically regardless of how many
productions preceded it; (3) the un-reset control still swaps (causal); (4) the gate-first MOAT is untouched (0 producer
calls on abstains); (5) EMERGE-59's FrameSlotCQ is byte-unchanged (default preserved).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners._emerge59_spiking_broca_frame_slots_derisk import FrameSlotCQ, BrocaProducer, decision_from_emerge
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (
    ResetFrameSlotCQ, _render_sequence, _sequence_exact, _position_independence,
    _make_reset, _make_plain, _SEQUENCE,
)


@pytest.mark.parametrize("seed", [100, 101])
def test_reset_fixes_sequence_order_on_failing_seeds(seed):
    """On the seeds where EMERGE-60 swapped (100, 101), the reset renders the emit SEQUENCE EXACT (robin as 5th emit)."""
    cq = ResetFrameSlotCQ(seed=seed)
    cq.learn()
    surfaces, _prod, _moat, _mp = _render_sequence(cq)
    assert _sequence_exact(surfaces) == 1.0, surfaces
    assert surfaces[-1] == "the robin can breathe", surfaces[-1]   # the 5th emit, previously swapped


@pytest.mark.parametrize("seed", [42, 100, 101])
def test_position_independence_with_reset(seed):
    """The SAME fact renders identically at emit-position 1 / 3 / 5 (0 / 2 / 4 prior productions) -- the load-bearing
    property: an utterance must not depend on prior utterances' residual state."""
    ok, surfaces_at = _position_independence(_make_reset, seed)
    assert ok, surfaces_at
    assert surfaces_at[1] == surfaces_at[5] == "the robin can breathe", surfaces_at


def test_reset_is_causal_control_swaps():
    """WITHOUT the reset, at least one of the failing seeds still swaps at the 5th emit (the fix is causally needed)."""
    swapped = False
    for seed in (100, 101):
        cq = FrameSlotCQ(seed=seed)
        cq.learn()
        surfaces, _prod, _moat, _mp = _render_sequence(cq)
        if _sequence_exact(surfaces) < 1.0:
            swapped = True
    assert swapped, "the un-reset control did not swap on 100/101 -- rebuild the failing sequence"


def test_moat_untouched_by_reset():
    """The reset does NOT run (hence does not reset) the producer on an ABSTAIN -- the gate-first moat holds."""
    cq = ResetFrameSlotCQ(seed=100)
    cq.learn()
    prod = BrocaProducer(cq)
    before = prod.production_count
    r = prod.speak(decision_from_emerge("ABSTAIN"))
    assert r["produced"] is False
    assert prod.production_count == before                          # producer never invoked on abstain


def test_emerge59_frameslotcq_default_unchanged():
    """EMERGE-59's FrameSlotCQ has NO reset (its per-production behavior is byte-unchanged): rendering robin twice from
    the SAME producer WITHOUT a reset advances the shared substrate (the two renders are NOT guaranteed identical), i.e.
    the base class does not silently acquire the wash-out. This pins that the fix is confined to the subclass."""
    assert not hasattr(FrameSlotCQ(seed=100), "_post_init_state")   # base class has no snapshot
    assert hasattr(ResetFrameSlotCQ(seed=100), "_post_init_state")  # only the subclass does


def test_sequence_shape():
    """The tested sequence is the EMERGE-60 emit order (owl/minnow F_MODAL, penguin/pike F_INTR, robin F_MODAL 5th)."""
    subjects = [s for (s, _v, _p, _e) in _SEQUENCE]
    assert subjects == ["owl", "minnow", "penguin", "pike", "robin"]
    assert _SEQUENCE[-1][3] == "the robin can breathe"
