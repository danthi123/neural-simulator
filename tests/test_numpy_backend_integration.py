"""End-to-end integration test for SIM_BACKEND=numpy.

Verifies the major user-facing code paths run on the NumPy backend without
crashing or producing degenerate results. Acts as a regression guard:
if a future CuPy-only call sneaks into bridge.py / connectivity.py /
kernels.py / bio_three_factor.py / chat_repl, this test should catch it.

NOT a performance test. NumPy backend is intentionally slow at production
scale; these tests use toy architectures.

Mark as @pytest.mark.slow because each test builds + initializes a bridge
(several seconds even at toy scale).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def numpy_backend(monkeypatch):
    """Force SIM_BACKEND=numpy for the duration of a single test.

    Resets the cached backend so the test sees a clean numpy resolution.
    """
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    xp, name = get_backend("numpy")
    assert name == "numpy"
    yield xp
    _reset_cache_for_tests()


@pytest.mark.slow
def test_bridge_constructs_and_runs_steps_on_numpy(numpy_backend):
    """SimulationBridge initializes a 50-neuron Watts-Strogatz network +
    runs 100 simulation steps end-to-end under NumPy backend."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig

    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.dt = 1.0
    viz = VisualizationConfig()
    rt = RuntimeState()
    gpu = GPUConfig()

    bridge = SimulationBridge(
        core_config=cfg, viz_config=viz, runtime_state=rt, gpu_config=gpu
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized

    # Run 100 steps; ensure no exception
    for _ in range(100):
        bridge._run_one_simulation_step()

    # Sanity: cp_connections should be scipy.sparse (not cupyx)
    import scipy.sparse as sp
    assert isinstance(bridge.cp_connections, sp.csr_matrix)
    import numpy as np
    assert isinstance(bridge.cp_connections.data, np.ndarray)


@pytest.mark.slow
def test_bridge_brain_region_framework_on_numpy(numpy_backend):
    """Brain region framework (multiple regions + pathways) runs under NumPy."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="cortex", n_neurons=30, exc_fraction=0.8,
                     internal_density=0.1),
        BrainRegion(name="motor", n_neurons=15, exc_fraction=0.8,
                     internal_density=0.1),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="cortex", to_region="motor",
                       density=0.1, weight_mean=1.0, weight_jitter=0.1),
    ]
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    assert bridge.core_config.num_neurons == 45  # 30 + 15

    # Run 20 steps
    for _ in range(20):
        bridge._run_one_simulation_step()


@pytest.mark.slow
def test_bridge_checkpoint_save_load_on_numpy(numpy_backend, tmp_path):
    """save_checkpoint + load_checkpoint round-trip under NumPy."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig

    cfg = CoreSimConfig()
    cfg.num_neurons = 30
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized

    # Run a few steps
    for _ in range(20):
        bridge._run_one_simulation_step()

    # Save + load
    ckpt = tmp_path / "test.simstate.h5"
    bridge.save_checkpoint(str(ckpt))
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0

    # Build a fresh bridge + load
    bridge2 = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge2._initialize_simulation_data()
    bridge2.load_checkpoint(str(ckpt))
    # Bridge state should be loaded
    assert bridge2.core_config.num_neurons == 30


@pytest.mark.slow
def test_chat_repl_tier1_toy_runs_on_numpy(numpy_backend):
    """chat_repl tier1 toy arch (n_lang=64, n_motor=16) trains + chat_inference
    returns sensible result under NumPy backend.

    Toy arch (not Tier 1 production) keeps the test fast (~3-5 sec).
    Doesn't check accuracy (4 events isn't enough for binding); just
    verifies the pipeline runs end-to-end without crashing.
    """
    from research.runners.bio_three_factor import run_three_factor

    bridge, _ = run_three_factor(
        seed=42, n_events_per_direction=2, biological=True,
        n_lang_input=64, n_motor_per_action=16, n_motor_fs_per_action=4,
        enable_motor_fs=True, enable_nmda=False,
        apply_topographic_bias=True, embodied_hebbian=True,
        synonym_mode=False, verbose=False,
    )
    # Bridge built + trained; basic sanity
    assert bridge.core_config.num_neurons == (64 + 16 * 4 + 64 + 4 * 4)
    assert bridge.is_initialized

    # Now try chat_inference (the W->A inference path)
    from research.runners.chat_repl import chat_inference
    result = chat_inference(bridge, "north")
    # Predicted action should be one of N/E/S/W
    assert result["predicted_action"] in ("N", "E", "S", "W")
    # Delta counts should be a dict with 4 keys
    assert set(result["delta_counts"].keys()) == {"N", "E", "S", "W"}


@pytest.mark.slow
def test_lineage_save_load_under_numpy(numpy_backend, tmp_path):
    """BridgeLineage save+load works under NumPy backend.

    Uses a minimal bridge to keep the test fast.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.lineage import BridgeLineage

    cfg = CoreSimConfig()
    cfg.num_neurons = 30
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()

    lineage = BridgeLineage("numpy_test", root=tmp_path)
    lineage.save(bridge, tier="numpy-test-tier",
                  arch={"mode": "numpy_test", "n_neurons": 30})
    assert lineage.exists()
    meta = lineage.read_metadata()
    assert meta.current_tier == "numpy-test-tier"
    assert meta.arch["mode"] == "numpy_test"

    # Reload by building a fresh bridge + load_checkpoint
    bridge2 = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge2._initialize_simulation_data()
    bridge2.load_checkpoint(str(lineage.current_path))
    assert bridge2.core_config.num_neurons == 30


def test_backend_default_when_cupy_available():
    """Without SIM_BACKEND env var, default resolves to cupy on hosts
    that have it. (Skipped on CPU-only hosts.)"""
    try:
        import cupy  # noqa: F401
    except (ImportError, RuntimeError):
        pytest.skip("cupy unavailable; default-backend test is GPU-only")
    # Clear cache, clear env var, re-resolve
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    if "SIM_BACKEND" in os.environ:
        del os.environ["SIM_BACKEND"]
    xp, name = get_backend()
    assert name == "cupy"


def test_backend_respects_env_var_forced_numpy(monkeypatch):
    """SIM_BACKEND=numpy env var forces numpy resolution even if cupy
    is available."""
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    xp, name = get_backend()
    assert name == "numpy"
    import numpy as np
    assert xp is np
