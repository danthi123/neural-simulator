"""Ensure cp_synapse_plastic_mask survives save_checkpoint / load_checkpoint.

Needed for G3 (persistence). Without this, a checkpointed mid-training run
would lose the frozen-reservoir property on reload and plasticity would leak
into hidden->hidden weights.
"""
import numpy as np
import pytest


def test_plastic_mask_roundtrip_through_checkpoint(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("h5py")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 6
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 101
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = True
    cfg.enable_watts_strogatz = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    plan = {
        "plastic": {
            "pre_indices": [0, 0],
            "post_indices": [3, 4],
            "initial_weights": np.array([0.2, 0.3], dtype=np.float32),
            "plastic": True,
            "count": 2,
        },
        "fixed": {
            "pre_indices": [1, 1, 2],
            "post_indices": [3, 4, 5],
            "initial_weights": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "plastic": False,
            "count": 3,
        },
    }
    bridge.inject_explicit_wiring(plan)

    # Snapshot the expected mask
    mask_before = cp.asnumpy(bridge.cp_synapse_plastic_mask).copy()
    assert mask_before.sum() == 2  # two plastic

    # Save checkpoint
    ckpt_path = tmp_path / "mask_ckpt.simstate.h5"
    assert bridge.save_checkpoint(str(ckpt_path)) is not False

    # Fresh bridge, load
    bridge2 = SimulationBridge(
        core_config=CoreSimConfig(), viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge2.load_checkpoint(str(ckpt_path))

    assert bridge2.cp_synapse_plastic_mask is not None, \
        "Checkpoint should restore cp_synapse_plastic_mask"
    mask_after = cp.asnumpy(bridge2.cp_synapse_plastic_mask)
    assert np.array_equal(mask_after, mask_before), \
        f"Mask drifted across checkpoint: before={mask_before}, after={mask_after}"


def test_absent_mask_load_back_compat(tmp_path):
    """Older checkpoints without the mask should load with mask = None."""
    pytest.importorskip("cupy")
    pytest.importorskip("h5py")
    import cupy as cp
    import h5py

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 4
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.seed = 1
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.enable_stdp = True
    cfg.enable_watts_strogatz = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # No explicit wiring → no mask
    assert bridge.cp_synapse_plastic_mask is None

    ckpt = tmp_path / "no_mask_ckpt.simstate.h5"
    bridge.save_checkpoint(str(ckpt))

    # Confirm the H5 file has no mask dataset
    with h5py.File(ckpt, "r") as h:
        assert "cp_synapse_plastic_mask" not in h

    bridge2 = SimulationBridge(
        core_config=CoreSimConfig(), viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge2.load_checkpoint(str(ckpt))
    assert bridge2.cp_synapse_plastic_mask is None
