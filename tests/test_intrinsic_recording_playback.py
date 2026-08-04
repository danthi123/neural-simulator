"""End-to-end recording/playback coverage for region intrinsic current."""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import to_host  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402


def _bridge(*, intrinsic_current_pA: float = 0.0) -> SimulationBridge:
    region = BrainRegion(
        name="gpi",
        n_neurons=8,
        exc_fraction=0.0,
        internal_density=0.2,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type="IZH2007_GPI_OUTPUT",
        intrinsic_current_pA=intrinsic_current_pA,
    )
    config = CoreSimConfig(
        num_neurons=8,
        seed=37,
        enable_brain_region_framework=True,
        brain_regions=[region],
        region_pathways=[],
        enable_ou_process=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_homeostasis=False,
        enable_stdp=False,
        enable_structural_plasticity=False,
        enable_reward_modulation=False,
        enable_inhibitory_stdp=False,
        enable_nmda=False,
        enable_gabab=False,
        enable_step_megakernel_v2=False,
    )
    return SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(
            enable_profiling=False,
            recording_mode="streaming",
            streaming_async_write=False,
        ),
    )


def test_intrinsic_current_survives_recording_playback_initialization(tmp_path):
    recording = _bridge(intrinsic_current_pA=100.0)
    recording._initialize_simulation_data()
    assert recording.is_initialized

    path = tmp_path / "intrinsic-current.simrec.h5"
    assert recording.start_recording_to_file(str(path)) is True
    recording.stop_recording()

    playback = _bridge()
    loaded = playback._prepare_loaded_recording_metadata(str(path))
    assert loaded is not None
    h5_file = loaded["h5_file_obj_for_playback"]
    try:
        config_snapshot = loaded["config_snapshot"]
        assert type(config_snapshot["core_config"]["seed"]) is int
        assert playback._apply_config_and_initial_state_from_recording(
            config_snapshot, h5_file["initial_state"]
        ) is True
    finally:
        h5_file.close()

    np.testing.assert_array_equal(
        to_host(playback.cp_intrinsic_current_pA),
        np.full(8, 100.0, dtype=np.float32),
    )
