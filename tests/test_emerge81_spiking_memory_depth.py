"""CPU tests for EMERGE-81 -- the spiking liquid-state machine memory-depth characterization.

Light structural tests (the heavy 3-seed spiking sweep is validated by `--derisk`); a slow single-seed smoke asserts the
spiking pool holds a distal 1-bit cue past a fixed +-2 window. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest

import research.runners._emerge81_spiking_memory_depth_derisk as m81


def test_reuses_emerge79_task_and_emerge80_spiking_lsm():
    from research.runners._emerge80_spiking_lsm_port_derisk import SpikingLSM
    import research.runners._emerge79_reservoir_variable_distance_derisk as m79
    # EMERGE-81 composes the EMERGE-79 distal task with the EMERGE-80 spiking pool (no reimplementation)
    assert m81.SpikingLSM is SpikingLSM
    assert m81.m79 is m79
    assert m81._TEST_DISTS[0] == 0 and m81._TEST_DISTS[-1] >= 12


@pytest.mark.slow
def test_seed42_spiking_pool_holds_distal_cue():
    d = m81._one(42)
    assert d["marks_ok"] is True
    assert d["mean_spikes"] > 0.5                                   # the pool is genuinely spiking
    # the spiking LSM holds the distal cue well past the +-2 window at a mid distance (6 fillers)
    assert d["by_distance"][6]["spiking_lsm"] >= 0.85
    assert d["by_distance"][6]["window2"] <= 0.65                   # the fixed window is blind at distance
    assert d["mark_lesion_acc"] <= 0.65                             # role genuinely mark-determined
