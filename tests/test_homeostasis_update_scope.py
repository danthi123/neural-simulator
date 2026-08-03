import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners._homeostatic_spiking_reward_plasticity_derisk import build_bridge
from sim.backend import get_backend, to_host


def test_homeostasis_update_scope_preserves_unscoped_thresholds():
    bridge, cfg = build_bridge(seed=40, n=20)
    xp, _ = get_backend()
    mask = np.zeros(int(cfg.num_neurons), dtype=bool)
    mask[: int(cfg.num_neurons) // 2] = True
    cfg.enable_homeostasis = True
    cfg.enable_ou_process = False
    cfg.homeostasis_target_rate = 0.5
    cfg.homeostasis_ema_alpha = 1.0
    cfg.homeostasis_threshold_adapt_rate = 1.0
    bridge.cp_homeostasis_update_neuron_mask = xp.asarray(mask)
    bridge.cp_neuron_activity_ema[:] = 0.25
    thresholds_before = np.asarray(
        to_host(bridge.cp_neuron_firing_thresholds)
    ).copy()
    activity_before = np.asarray(to_host(bridge.cp_neuron_activity_ema)).copy()

    bridge.cp_external_input_current[:] = 0.0
    bridge._run_one_simulation_step()

    thresholds_after = np.asarray(to_host(bridge.cp_neuron_firing_thresholds))
    activity_after = np.asarray(to_host(bridge.cp_neuron_activity_ema))
    assert np.any(thresholds_after[mask] != thresholds_before[mask])
    assert np.any(activity_after[mask] != activity_before[mask])
    np.testing.assert_array_equal(
        thresholds_after[~mask], thresholds_before[~mask]
    )
    np.testing.assert_array_equal(activity_after[~mask], activity_before[~mask])
