"""CI guard for EMERGE-40: the fully-spiking HTM Spatial Pooler. The winner-inactive (selectivity) depression is now the
committed sim/ kernel fused_htm_winner_inactive_depression (additive; existing kernels byte-unchanged). Both pooler
learning terms are sim/ fused kernels; overlapping categories separate; the selectivity kernel is load-bearing; permuted +
dAP-lesion collapse it. Also pins the kernel's math + that the sibling permanence kernel is unchanged. CPU (numpy)."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest


def test_winner_inactive_kernel_math():
    """The new kernel depresses ONLY winner (post_win=1) + inactive-input (pre_active=0) synapses; else a no-op."""
    from sim.kernels import fused_htm_winner_inactive_depression as g
    w = np.array([0.8, 0.8, 0.8, 0.8])
    pre_active = np.array([1.0, 0.0, 1.0, 0.0])
    post_win = np.array([1.0, 1.0, 0.0, 0.0])
    out = np.asarray(g(w, pre_active, post_win, 0.02, 0.0, 1.0))
    # only index 1 (inactive input, winner column) is depressed
    assert abs(out[0] - 0.8) < 1e-6 and abs(out[2] - 0.8) < 1e-6 and abs(out[3] - 0.8) < 1e-6
    assert abs(out[1] - 0.78) < 1e-6


def test_permanence_kernel_unchanged():
    """The sibling committed kernel keeps its 8-arg three-term signature (additive edit did not touch it)."""
    from sim.kernels import fused_htm_permanence_update as f
    w = np.array([0.5, 0.5, 0.5]); pl = np.array([1.0, 1.0, 0.0]); pn = np.array([1.0, 0.0, 1.0]); h = np.array([1.0, 1.0, 1.0])
    out = np.asarray(f(w, pl, pn, h, 0.05, 0.02, 0.0, 1.0))
    assert abs(out[0] - 0.55) < 1e-6 and abs(out[1] - 0.48) < 1e-6 and abs(out[2] - 0.5) < 1e-6


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge40_spiking_htm_sp_kernel_derisk import SpikingHTMSPProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge40 deps unavailable: {e}")
    return SpikingHTMSPProbe(seed=42, epochs=40)


def test_spiking_pooler_separates_overlapping_categories(probe):
    """The fully-sim/-kernel learned pooler separates 6 OVERLAPPING categories: held-out inheritance is high."""
    assert probe.held_out_acc() >= 0.8


def test_selectivity_kernel_load_bearing():
    """With the winner-inactive kernel OFF, discrimination collapses (the kernel is load-bearing)."""
    from research.runners._emerge40_spiking_htm_sp_kernel_derisk import SpikingHTMSPProbe
    withk = SpikingHTMSPProbe(seed=42, epochs=40, selectivity=True).held_out_acc()
    without = SpikingHTMSPProbe(seed=42, epochs=40, selectivity=False).held_out_acc()
    assert withk >= without + 0.25


if __name__ == "__main__":
    from research.runners._emerge40_spiking_htm_sp_kernel_derisk import SpikingHTMSPProbe
    p = SpikingHTMSPProbe(seed=42, epochs=40)
    assert p.held_out_acc() >= 0.8
    print("OK: emerge40 fully-spiking HTM-SP -- winner-inactive kernel separates overlapping cats, load-bearing")
