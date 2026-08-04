"""Focused coverage for optional region-scoped intrinsic current."""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    NeuronModel,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import to_host  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402


_READ_ONLY = {
    "enable_ou_process": False,
    "enable_hebbian_learning": False,
    "enable_short_term_plasticity": False,
    "enable_homeostasis": False,
    "enable_stdp": False,
    "enable_structural_plasticity": False,
    "enable_reward_modulation": False,
    "enable_inhibitory_stdp": False,
    "enable_nmda": False,
    "enable_gabab": False,
    "enable_step_megakernel_v2": False,
}


def _region(name: str, n: int, intrinsic_current_pA: float = 0.0) -> BrainRegion:
    return BrainRegion(
        name=name,
        n_neurons=n,
        exc_fraction=0.0,
        internal_density=0.2,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type="IZH2007_GPI_OUTPUT",
        intrinsic_current_pA=intrinsic_current_pA,
    )


def _build(regions, *, model=NeuronModel.IZHIKEVICH.name, seed=7):
    regions = list(regions)
    cfg = CoreSimConfig(
        num_neurons=sum(int(region.n_neurons) for region in regions),
        neuron_model_type=model,
        seed=seed,
        dt_ms=0.05 if model == NeuronModel.HODGKIN_HUXLEY.name else 1.0,
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=[],
        **_READ_ONLY,
    )
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def _capture(bridge, steps: int):
    raster = []
    for _ in range(steps):
        bridge._run_one_simulation_step()
        raster.append(to_host(bridge.cp_firing_states).copy())
    return (
        np.stack(raster),
        to_host(bridge.cp_membrane_potential_v).copy(),
        to_host(bridge.cp_recovery_variable_u).copy(),
    )


def test_brain_region_intrinsic_current_defaults_off():
    assert BrainRegion(name="gpi", n_neurons=4).intrinsic_current_pA == 0.0
    bridge = _build([_region("gpi", 8)])
    assert bridge.cp_intrinsic_current_pA is None


def test_intrinsic_current_is_scoped_and_survives_external_clear():
    bridge = _build([_region("gpi", 6, 100.0), _region("control", 4)])
    expected = np.concatenate(
        [np.full(6, 100.0, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    )
    np.testing.assert_array_equal(to_host(bridge.cp_intrinsic_current_pA), expected)

    intrinsic_before = to_host(bridge.cp_intrinsic_current_pA).copy()
    bridge.cp_external_input_current[:] = 321.0
    bridge.clear_tag_drive()
    np.testing.assert_array_equal(
        to_host(bridge.cp_external_input_current), np.zeros(10, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        to_host(bridge.cp_intrinsic_current_pA), intrinsic_before
    )

    bridge.clear_simulation_state_and_gpu_memory()
    assert bridge.cp_intrinsic_current_pA is None


def test_intrinsic_current_matches_equivalent_external_drive():
    intrinsic = _build([_region("gpi", 12, 100.0)])
    external = _build([_region("gpi", 12)], seed=7)
    external.cp_external_input_current[:] = 100.0

    intrinsic_result = _capture(intrinsic, 200)
    external_result = _capture(external, 200)
    for intrinsic_array, external_array in zip(intrinsic_result, external_result):
        np.testing.assert_array_equal(intrinsic_array, external_array)
    assert int(intrinsic_result[0].sum()) > 0
    np.testing.assert_array_equal(
        to_host(intrinsic.cp_external_input_current), np.zeros(12, dtype=np.float32)
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_intrinsic_current_raises(bad_value):
    with pytest.raises(ValueError, match="must be finite"):
        _build([_region("gpi", 4, bad_value)])


def test_intrinsic_current_must_fit_float32():
    with pytest.raises(ValueError, match="representable as float32"):
        _build([_region("gpi", 4, 1e40)])


@pytest.mark.parametrize(
    "model", [NeuronModel.HODGKIN_HUXLEY.name, NeuronModel.ADEX.name]
)
def test_non_izhikevich_intrinsic_current_raises(model):
    with pytest.raises(ValueError, match="supports only IZHIKEVICH"):
        _build([_region("gpi", 4, 100.0)], model=model)


def test_intrinsic_current_checkpoint_round_trip_and_old_checkpoint(tmp_path):
    bridge = _build([_region("gpi", 10, 100.0)])
    expected = to_host(bridge.cp_intrinsic_current_pA).copy()
    checkpoint = tmp_path / "intrinsic-current.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True

    restored = _build([_region("gpi", 10)])
    assert restored.load_checkpoint(str(checkpoint)) is True
    np.testing.assert_array_equal(to_host(restored.cp_intrinsic_current_pA), expected)

    with h5py.File(checkpoint, "r+") as h5f:
        h5f["cp_intrinsic_current_pA"][0] = np.nan
    malformed = _build([_region("gpi", 10)])
    assert malformed.load_checkpoint(str(checkpoint)) is False

    with h5py.File(checkpoint, "r+") as h5f:
        h5f["cp_intrinsic_current_pA"][:] = expected
        del h5f["cp_intrinsic_current_pA"]

    legacy = _build([_region("gpi", 10, 100.0)])
    assert legacy.load_checkpoint(str(checkpoint)) is True
    assert legacy.cp_intrinsic_current_pA is None


def test_checkpoint_cannot_bypass_intrinsic_current_model_guard(tmp_path):
    bridge = _build([_region("gpi", 4, 100.0)])
    checkpoint = tmp_path / "wrong-model.simstate.h5"
    assert bridge.save_checkpoint(str(checkpoint)) is True
    with h5py.File(checkpoint, "r+") as h5f:
        h5f.attrs["neuron_model_type"] = NeuronModel.HODGKIN_HUXLEY.name

    restored = _build([_region("gpi", 4)])
    assert restored.load_checkpoint(str(checkpoint)) is False
    assert restored.cp_intrinsic_current_pA is None


def test_recording_initial_state_captures_intrinsic_current_only_as_static_state():
    bridge = _build([_region("gpi", 5, 100.0)])
    without_intrinsic = _build([_region("gpi", 5)])
    snapshot = bridge._capture_initial_state_for_recording()
    np.testing.assert_array_equal(
        snapshot["cp_intrinsic_current_pA"], np.full(5, 100.0, dtype=np.float32)
    )
    assert bridge._estimate_frame_size_bytes() == without_intrinsic._estimate_frame_size_bytes()
