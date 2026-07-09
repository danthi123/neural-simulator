"""CI guard for the R-iii on-substrate dendritic-dAP CA3 completion SURPASS (CYCLE 1068): the two-compartment
dAP plateau completes a hand-installed CA3 attractor where the linear point-neuron cannot, riding the right
ensemble structure. GPU-gated (the result is cupy-validated; the plateau ignition regime is backend-sensitive)."""
import os

import pytest


def _gpu():
    try:
        from sim.backend import get_backend, is_gpu_backend
        get_backend()
        return is_gpu_backend()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="on-substrate dAP completion is cupy-validated (GPU-gated)")
def test_dendritic_dap_completes_where_linear_fails_seed42():
    from research.runners._riii_onsubstrate_readout_test import run_seed
    kw = dict(k_thresh=6.0, w_high=15.0, w_low=1.5, plateau_strength=300.0, two_comp=True, apical_R=50.0, apical_gc=2.0)
    plateau = run_seed(42, coincidence=True, **kw)
    linear = run_seed(42, coincidence=False, **kw)
    flat = run_seed(42, coincidence=True, flat=True, **kw)
    scramble = run_seed(42, coincidence=True, scramble=True, **kw)
    assert plateau["heldout"] > 0.30, plateau                         # dendritic dAP completes the held-out members
    assert linear["heldout"] < 0.20, linear                           # linear point-neuron fails at the same attractor
    assert plateau["heldout"] - linear["heldout"] > 0.30, (plateau, linear)   # plateau load-bearing
    assert flat["heldout"] < 0.20, flat                               # completion needs the installed attractor
    assert plateau["heldout"] - scramble["heldout"] > 0.20, (plateau, scramble)  # rides the RIGHT structure
    assert plateau["nonens"] < 0.20, plateau                          # specific (non-ensemble stays silent)
