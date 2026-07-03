"""CPU tests for EMERGE-82 -- the on-bridge spiking LSM (recurrent BrainRegion on a real SimulationBridge).

Light structural tests (the full 6-seed on-bridge sweep is validated by `--derisk`); a slow smoke builds a tiny recurrent
region on a real SimulationBridge and confirms it is genuinely spiking + `final_state` reads the region's real firing.
CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge82_onbridge_lsm_derisk as m82


def test_mirrors_reservoir_api_and_reuses_harness():
    from research.runners._emerge78_reservoir_form_to_role_derisk import _slot_data, _fit_slots
    # the on-bridge LSM composes with the EMERGE-78 harness (drop-in for the reservoir)
    assert hasattr(m82.OnBridgeLSM, "final_state")
    assert m82._fit_slots is _fit_slots and m82._slot_data is _slot_data
    assert m82._INTERNAL_DENSITY > 0.0                              # a genuinely RECURRENT region


def test_uses_brain_region_framework_recurrence():
    # the reservoir is a recurrent BrainRegion (internal_density > 0), not a hand-wired numpy matrix
    assert m82._N_POOL >= 100 and m82._EXC_W > 0 and m82._INH_W > 0


@pytest.mark.slow
def test_tiny_onbridge_region_is_genuinely_spiking():
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder
    enc = Encoder({"the", "to", "on", "can"})
    lsm = m82.OnBridgeLSM(enc.dim, seed=42, n=80)                   # a tiny recurrent region on a real SimulationBridge
    U = enc.encode(["the", "dog", "zeps", "to", "the", "pond"])
    state = lsm.final_state(U)
    assert state.shape == (lsm.n,)
    assert lsm._last_mean_spikes > 0.3                             # the bridge region actually spikes
    # region-silence (zero the drive) -> the region's real firing carries no signal
    s = lsm.final_state(U, silence=True)
    assert lsm._last_mean_spikes == 0.0
