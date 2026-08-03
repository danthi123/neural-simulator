import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._vocal_action_selector_gate import (
    SelectorConfig,
    build_selector_bridge,
)
from sim.backend import to_host


def _advance_once(bridge):
    bridge._run_one_simulation_step()


def test_ou_mask_scopes_injected_current_without_changing_ou_state():
    config = SelectorConfig(ou_sigma_pA=0.0)
    masked = build_selector_bridge(seed=19, config=config)
    silent = build_selector_bridge(seed=19, config=config)

    n = int(masked.core_config.num_neurons)
    split = n // 2
    masked.cp_ou_current[:] = np.float32(500.0)
    silent.cp_ou_current[:] = np.float32(500.0)
    masked.ou_decay_factor = np.float32(1.0)
    silent.ou_decay_factor = np.float32(1.0)
    masked.cp_ou_neuron_mask[:] = False
    masked.cp_ou_neuron_mask[:split] = True
    silent.cp_ou_neuron_mask[:] = False

    _advance_once(masked)
    _advance_once(silent)

    masked_ou = np.asarray(to_host(masked.cp_ou_current))
    silent_ou = np.asarray(to_host(silent.cp_ou_current))
    masked_v = np.asarray(to_host(masked.cp_membrane_potential_v))
    silent_v = np.asarray(to_host(silent.cp_membrane_potential_v))
    np.testing.assert_array_equal(masked_ou, silent_ou)
    assert np.any(
        masked_v[:split] != silent_v[:split]
    )
    np.testing.assert_array_equal(
        masked_v[split:],
        silent_v[split:],
    )


def test_ou_none_mask_preserves_legacy_all_neuron_injection():
    config = SelectorConfig(ou_sigma_pA=0.0)
    legacy = build_selector_bridge(seed=23, config=config)
    explicit_all = build_selector_bridge(seed=23, config=config)

    legacy.cp_ou_current[:] = np.float32(500.0)
    explicit_all.cp_ou_current[:] = np.float32(500.0)
    legacy.ou_decay_factor = np.float32(1.0)
    explicit_all.ou_decay_factor = np.float32(1.0)
    legacy.cp_ou_neuron_mask = None
    explicit_all.cp_ou_neuron_mask[:] = True

    _advance_once(legacy)
    _advance_once(explicit_all)

    np.testing.assert_array_equal(
        np.asarray(to_host(legacy.cp_membrane_potential_v)),
        np.asarray(to_host(explicit_all.cp_membrane_potential_v)),
    )
