"""Tests for bridge.set_pathway_weights() — post-build CSR weight overwrite.

Used by Cluster K v2 to apply Gabor pre-init to V1 simple cells, but
generally useful for any pathway whose initial weights need to be
computed post-bridge-init (e.g. loading checkpointed pathways).
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_minimal_bridge():
    """Build a tiny bridge with two regions and one pathway for testing."""
    pytest.importorskip("cupy")
    from sim.regions import BrainRegion, RegionPathway
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge

    regions = [
        BrainRegion(name="A", n_neurons=10, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0),
        BrainRegion(name="B", n_neurons=10, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0),
    ]
    pathways = [
        RegionPathway(from_region="A", to_region="B",
                      density=0.5, weight_mean=2.0, weight_jitter=0.0,
                      plastic=False),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 0.5
    cfg.seed = 42

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def test_set_pathway_weights_roundtrip():
    """After set_pathway_weights, the new weights are observable in CSR."""
    bridge = _make_minimal_bridge()
    import cupy as cp

    # Find any (pre, post) pair with an existing edge
    coo = bridge.cp_connections.tocoo()
    pre = int(coo.row[0].get())
    post = int(coo.col[0].get())

    # Set a distinctive weight
    new_weight = 99.5
    n_updated = bridge.set_pathway_weights(
        pathway_name="test_A_to_B",
        pre_indices=np.array([pre], dtype=np.int64),
        post_indices=np.array([post], dtype=np.int64),
        weights=np.array([new_weight], dtype=np.float32),
    )
    assert n_updated == 1

    # Read back
    coo2 = bridge.cp_connections.tocoo()
    found = False
    for i in range(int(coo2.nnz)):
        if int(coo2.row[i].get()) == pre and int(coo2.col[i].get()) == post:
            assert float(coo2.data[i].get()) == pytest.approx(new_weight, abs=1e-3)
            found = True
            break
    assert found, f"({pre},{post}) edge not found after set_pathway_weights"


def test_set_pathway_weights_missing_edge_raises():
    """Calling set_pathway_weights with a (pre,post) that doesn't exist
    should raise ValueError when add_missing=False (default)."""
    bridge = _make_minimal_bridge()

    # Find a (pre, post) pair that does NOT exist in CSR
    coo = bridge.cp_connections.tocoo()
    existing_pairs = set()
    for i in range(int(coo.nnz)):
        existing_pairs.add(
            (int(coo.row[i].get()), int(coo.col[i].get()))
        )

    # Pick a definitely-non-existent pair
    missing_pre = None
    missing_post = None
    for p in range(20):
        for q in range(20):
            if (p, q) not in existing_pairs and p < 10 and q >= 10:
                missing_pre = p
                missing_post = q
                break
        if missing_pre is not None:
            break

    assert missing_pre is not None, "Could not find a missing edge for test"

    with pytest.raises(ValueError, match=r"not found|missing"):
        bridge.set_pathway_weights(
            pathway_name="test_missing",
            pre_indices=np.array([missing_pre], dtype=np.int64),
            post_indices=np.array([missing_post], dtype=np.int64),
            weights=np.array([5.0], dtype=np.float32),
        )


def test_set_pathway_weights_multiple_edges():
    """Update multiple edges in one call; count should match."""
    bridge = _make_minimal_bridge()

    coo = bridge.cp_connections.tocoo()
    n_to_update = min(5, int(coo.nnz))
    pres = np.array([int(coo.row[i].get()) for i in range(n_to_update)],
                    dtype=np.int64)
    posts = np.array([int(coo.col[i].get()) for i in range(n_to_update)],
                     dtype=np.int64)
    new_weights = np.array([10.0 + i for i in range(n_to_update)],
                           dtype=np.float32)

    n_updated = bridge.set_pathway_weights(
        pathway_name="test_multi",
        pre_indices=pres,
        post_indices=posts,
        weights=new_weights,
    )
    assert n_updated == n_to_update

    # Verify all updates landed
    coo2 = bridge.cp_connections.tocoo()
    pair_to_data = {}
    for i in range(int(coo2.nnz)):
        pair_to_data[(int(coo2.row[i].get()), int(coo2.col[i].get()))] = (
            float(coo2.data[i].get())
        )
    for i in range(n_to_update):
        assert pair_to_data[(int(pres[i]), int(posts[i]))] == pytest.approx(
            float(new_weights[i]), abs=1e-3
        )
