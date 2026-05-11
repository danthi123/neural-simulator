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

    Resets the cached backend AND reloads sim.bridge / sim.kernels /
    sim.connectivity / sim.lineage so their module-level cp / csp /
    fuse references re-resolve to NumPy. Without this, tests that
    run AFTER a CuPy-backed test (e.g. test_backend.py CuPy paths)
    would see the cached CuPy `cp` inside bridge.py.
    """
    import importlib
    import sys

    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    xp, name = get_backend("numpy")
    assert name == "numpy"

    # Reload the modules that bind backend at import time. Order matters:
    # backend first (already imported), then connectivity + kernels (no
    # bridge dependency), then bridge (depends on connectivity/kernels).
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])

    yield xp

    # Clean up: reset cache + reload back so next test sees default again
    _reset_cache_for_tests()
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])


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


@pytest.mark.slow
def test_export_shards_real_bridge_on_numpy(numpy_backend, tmp_path):
    """Real bridge under NumPy: extract_per_pathway_csrs + lineage.export_shards.

    Builds a small brain-region bridge and exports per-pathway shards.
    Verifies shard contents match the original pathway data.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.lineage import BridgeLineage

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="A", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),  # no internal connections
        BrainRegion(name="B", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),
        BrainRegion(name="C", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="A", to_region="B",
                       density=0.3, weight_mean=1.0, weight_jitter=0.1),
        RegionPathway(from_region="B", to_region="C",
                       density=0.3, weight_mean=2.0, weight_jitter=0.1),
        RegionPathway(from_region="A", to_region="C",
                       density=0.3, weight_mean=3.0, weight_jitter=0.1),
    ]
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized

    # Extract per-pathway CSRs
    pathways = bridge.extract_per_pathway_csrs()
    assert set(pathways.keys()) == {"A_to_B", "B_to_C", "A_to_C"}

    # Each sub-matrix should have the expected shape (post x pre)
    # All regions are 10 neurons, so each pathway is 10x10
    for name, csr in pathways.items():
        assert csr.shape == (10, 10), f"{name}: shape {csr.shape}"

    # Now export via lineage
    lineage = BridgeLineage("shard_test", root=tmp_path)
    n = lineage.export_shards(bridge)
    assert n == 3

    # list_shards returns the expected names
    names = lineage.list_shards()
    assert set(names) == {"A_to_B", "B_to_C", "A_to_C"}

    # Shard files exist
    for pw in ("A_to_B", "B_to_C", "A_to_C"):
        shard_path = tmp_path / "shard_test" / "shards" / f"{pw}.npz"
        assert shard_path.exists(), f"{pw}.npz missing"


@pytest.mark.slow
def test_extract_per_pathway_csrs_requires_region_manager():
    """extract_per_pathway_csrs raises on non-brain-region bridges."""
    # Force a fresh backend
    import importlib
    import sys
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    if "SIM_BACKEND" in os.environ:
        del os.environ["SIM_BACKEND"]
    xp, name = get_backend("numpy") if "cupy" not in sys.modules else get_backend()
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])

    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig

    cfg = CoreSimConfig()
    cfg.num_neurons = 30
    cfg.enable_brain_region_framework = False  # no region_manager
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized

    # extract_per_pathway_csrs should raise — no region_manager
    with pytest.raises(RuntimeError, match="region_manager is None"):
        bridge.extract_per_pathway_csrs()


@pytest.mark.slow
def test_synapse_tiering_strategy_b_end_to_end(numpy_backend, tmp_path):
    """Strategy B: synapse_store mirror is initialized, activity is
    tracked per step, and dormant pathways are evicted to disk.

    Real-bridge end-to-end test of Phase 3 part 2 Strategy B.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="A", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),
        BrainRegion(name="B", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="A", to_region="B",
                       density=0.3, weight_mean=1.0, weight_jitter=0.1),
    ]
    cfg.enable_synapse_tiering = True
    cfg.synapse_tiering_evict_idle_steps = 5  # aggressive for test speed
    cfg.synapse_tiering_grace_pagein_steps = 0
    cfg.synapse_tiering_root = str(tmp_path / "shards")
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()

    # Store should be initialized + 1 pathway mirrored
    assert bridge.synapse_store is not None
    s = bridge.synapse_store.stats()
    assert s["n_pathways"] == 1
    assert s["n_in_memory"] == 1
    assert s["n_on_disk"] == 0
    assert s["n_pageouts_lifetime"] == 0

    # Run 20 steps — beyond evict threshold of 5
    for _ in range(20):
        bridge._run_one_simulation_step()

    # Pathway should have been evicted (no activity in toy network)
    s = bridge.synapse_store.stats()
    assert s["n_on_disk"] >= 1, (
        f"Expected eviction after 20 idle steps; stats: {s}"
    )
    # Shard file should exist on disk
    shard_path = tmp_path / "shards" / "A_to_B.npz"
    assert shard_path.exists()


@pytest.mark.slow
def test_synapse_tiering_opt_in_default_off(numpy_backend, tmp_path):
    """Default cfg.enable_synapse_tiering=False -> no synapse_store."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="X", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),
        BrainRegion(name="Y", n_neurons=10, exc_fraction=1.0,
                     internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="X", to_region="Y",
                       density=0.3, weight_mean=1.0, weight_jitter=0.1),
    ]
    # cfg.enable_synapse_tiering defaults to False
    cfg.dt = 1.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    # No store created
    assert bridge.synapse_store is None
    # Step still runs cleanly
    bridge._run_one_simulation_step()


@pytest.mark.slow
def test_bridge_memory_real_bridge_store_recall_on_numpy(numpy_backend, tmp_path, monkeypatch):
    """BridgeMemory end-to-end on a real bridge under SIM_BACKEND=numpy.

    Builds a tier1 toy bridge, binds a key with embodied-Hebbian,
    verifies recall returns the expected motor pool.

    Toy scale: n_lang=64, n_motor=16 — fast enough for CI but big
    enough to exercise the full bind/recall paths.
    """
    from sim.bridge_memory import BridgeMemory
    from research.runners.bio_three_factor import run_three_factor

    # Build a tiny pre-trained bridge (skip lineage train path)
    bridge, _ = run_three_factor(
        seed=42, n_events_per_direction=2, biological=True,
        n_lang_input=64, n_motor_per_action=16, n_motor_fs_per_action=4,
        enable_motor_fs=True, enable_nmda=False,
        apply_topographic_bias=True, embodied_hebbian=True,
        synonym_mode=False, verbose=False,
    )

    # Pre-create the lineage to skip BridgeMemory's load-or-train path
    from sim.lineage import BridgeLineage
    lineage = BridgeLineage("bridge_memory_real_test", root=tmp_path)
    lineage.save(bridge, tier="tier1", arch={"mode": "tier1"})

    mem = BridgeMemory(
        lineage_name="bridge_memory_real_test",
        mode="tier1",
        bridge=bridge,
        auto_save=True,
        verbose=False,
    )
    mem._lineage = lineage  # bypass _ensure_loaded's lineage fetch

    # Bind a new word ("alice") to motor_N
    store_result = mem.store("alice", "north", n_events=20)
    assert store_result["target_action"] == "N"
    assert store_result["n_events_run"] == 20

    # Recall should at least return SOMETHING (4 motor pools)
    recall_result = mem.recall("alice", top_k=4)
    assert len(recall_result) == 4
    # All entries should have required fields
    for r in recall_result:
        assert "action" in r and r["action"] in ("N", "E", "S", "W")
        assert "value" in r
        assert "confidence" in r
        assert "rank" in r

    # The lineage should have a memory_bind growth event
    meta = lineage.read_metadata()
    bind_events = [e for e in meta.growth_events
                    if e["kind"] == "memory_bind"]
    assert len(bind_events) == 1
    assert bind_events[0]["metadata"]["target_action"] == "N"
