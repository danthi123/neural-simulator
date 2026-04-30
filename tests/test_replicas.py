"""Unit tests for the batched-replica framework (Session E.3).

See docs/plans/2026-04-24-batched-replica-framework.md.

The framework runs B independent replicas in a single bridge process via
block-diagonal embedding: a B×N super-network where each block is an
independent G9-style reservoir. Per-step compute amortizes across all B
replicas (one sparse matmul, one neuron-dynamics kernel, etc.).

Default OFF: when no replicas are configured, the bridge runs as a
single population unchanged.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- Task 1: dataclasses + manager ----------

def test_replica_config_defaults():
    from sim.replicas import ReplicaConfig

    rc = ReplicaConfig(replica_id=0)
    assert rc.replica_id == 0
    assert rc.seed_offset == 0
    assert rc.neuromodulator_overrides == {}


def test_replica_config_with_overrides():
    from sim.replicas import ReplicaConfig

    rc = ReplicaConfig(
        replica_id=3,
        seed_offset=42,
        neuromodulator_overrides={"ne_sensitivity": 60.0, "ne_threshold": 0.6},
    )
    assert rc.replica_id == 3
    assert rc.seed_offset == 42
    assert rc.neuromodulator_overrides["ne_sensitivity"] == 60.0


def test_replica_manager_allocates_blocks():
    from sim.replicas import ReplicaConfig, ReplicaManager

    replicas = [ReplicaConfig(replica_id=0), ReplicaConfig(replica_id=1)]
    mgr = ReplicaManager(replicas, neurons_per_replica=100)
    mgr.initialize()
    assert mgr.n_replicas() == 2
    assert mgr.total_neurons() == 200
    assert mgr.indices(0) == list(range(0, 100))
    assert mgr.indices(1) == list(range(100, 200))


def test_replica_manager_replica_for_neuron():
    """Reverse lookup: which replica does a given global neuron index belong to?"""
    from sim.replicas import ReplicaConfig, ReplicaManager

    replicas = [ReplicaConfig(replica_id=i) for i in range(3)]
    mgr = ReplicaManager(replicas, neurons_per_replica=50)
    mgr.initialize()
    assert mgr.replica_for_neuron(0) == 0
    assert mgr.replica_for_neuron(49) == 0
    assert mgr.replica_for_neuron(50) == 1
    assert mgr.replica_for_neuron(149) == 2
    with pytest.raises(IndexError):
        mgr.replica_for_neuron(150)


def test_replica_manager_unknown_replica_raises():
    from sim.replicas import ReplicaConfig, ReplicaManager

    mgr = ReplicaManager([ReplicaConfig(replica_id=0)], neurons_per_replica=10)
    mgr.initialize()
    with pytest.raises(KeyError):
        mgr.indices(99)


def test_replica_manager_indices_dict_for_neuromod_groups():
    """Returns {f'replica:N': [int]} for nm_mgr.set_group_indices() and
    set_replica_indices()."""
    from sim.replicas import ReplicaConfig, ReplicaManager

    replicas = [ReplicaConfig(replica_id=i) for i in range(3)]
    mgr = ReplicaManager(replicas, neurons_per_replica=10)
    mgr.initialize()
    d = mgr.replica_indices_dict()
    assert d[0] == list(range(0, 10))
    assert d[1] == list(range(10, 20))
    assert d[2] == list(range(20, 30))


# ---------- Task 2: block-diagonal wiring plan ----------

def test_replicate_wiring_plan_basic():
    from sim.replicas import replicate_wiring_plan

    template = {
        "input_to_hidden": {
            "pre_indices": [0, 1, 0],
            "post_indices": [2, 2, 3],
            "initial_weights": [1.0, 1.0, 0.5],
            "plastic": False,
            "conn_type": "E_TO_MIX",
            "count": 3,
        },
    }
    replicated = replicate_wiring_plan(
        template, n_replicas=2, neurons_per_replica=4,
    )
    g = replicated["input_to_hidden"]
    assert g["count"] == 6
    # Replica 0 indices unshifted
    assert g["pre_indices"][:3] == [0, 1, 0]
    assert g["post_indices"][:3] == [2, 2, 3]
    # Replica 1 indices shifted by neurons_per_replica=4
    assert g["pre_indices"][3:] == [4, 5, 4]
    assert g["post_indices"][3:] == [6, 6, 7]
    # Weights tiled exactly (no jitter without seed strategy)
    assert g["initial_weights"][:3] == [1.0, 1.0, 0.5]
    assert g["initial_weights"][3:] == [1.0, 1.0, 0.5]
    # Other metadata preserved
    assert g["plastic"] is False
    assert g["conn_type"] == "E_TO_MIX"


def test_replicate_wiring_plan_handles_multiple_groups():
    from sim.replicas import replicate_wiring_plan

    template = {
        "input_to_hidden": {
            "pre_indices": [0],
            "post_indices": [1],
            "initial_weights": [1.0],
            "plastic": False,
            "conn_type": "E_TO_MIX",
            "count": 1,
        },
        "hidden_to_motor": {
            "pre_indices": [1],
            "post_indices": [2],
            "initial_weights": [0.5],
            "plastic": True,
            "conn_type": "MIXED",
            "count": 1,
        },
    }
    replicated = replicate_wiring_plan(
        template, n_replicas=3, neurons_per_replica=3,
    )
    assert "input_to_hidden" in replicated
    assert "hidden_to_motor" in replicated
    assert replicated["input_to_hidden"]["count"] == 3
    assert replicated["hidden_to_motor"]["count"] == 3
    # Replica 2 input_to_hidden synapse should be at indices (0+2*3, 1+2*3) = (6, 7)
    assert replicated["input_to_hidden"]["pre_indices"][2] == 6
    assert replicated["input_to_hidden"]["post_indices"][2] == 7
    # plastic flag preserved per-group
    assert replicated["input_to_hidden"]["plastic"] is False
    assert replicated["hidden_to_motor"]["plastic"] is True


def test_replicate_wiring_plan_zero_replicas_yields_empty():
    """0 replicas -> empty plan (each group has count=0)."""
    from sim.replicas import replicate_wiring_plan

    template = {
        "g1": {"pre_indices": [0], "post_indices": [1],
                "initial_weights": [1.0], "plastic": False,
                "conn_type": "E_TO_MIX", "count": 1},
    }
    plan = replicate_wiring_plan(template, n_replicas=0, neurons_per_replica=10)
    assert plan["g1"]["count"] == 0
    assert plan["g1"]["pre_indices"] == []


# ---------- Task 3: per-replica seed-jitter ----------

def test_replicate_with_seeds_produces_different_weights_per_replica():
    """Different seed_offsets -> different weight realizations per replica."""
    from sim.replicas import (
        ReplicaConfig, replicate_wiring_plan_with_seeds,
    )
    template = {
        "input_to_hidden": {
            "pre_indices": [0, 1, 2],
            "post_indices": [3, 3, 3],
            "initial_weights": [1.0, 1.0, 1.0],
            "plastic": False,
            "conn_type": "E_TO_MIX",
            "count": 3,
        },
    }
    replicas = [
        ReplicaConfig(replica_id=0, seed_offset=42),
        ReplicaConfig(replica_id=1, seed_offset=43),
    ]
    plan = replicate_wiring_plan_with_seeds(
        template, replicas=replicas,
        neurons_per_replica=4, weight_jitter=0.2,
    )
    g = plan["input_to_hidden"]
    block0_weights = g["initial_weights"][:3]
    block1_weights = g["initial_weights"][3:]
    assert block0_weights != block1_weights
    # All weights should still be positive and roughly near 1.0
    for w in block0_weights + block1_weights:
        assert 0.4 < w < 1.6


def test_replicate_with_seeds_zero_jitter_matches_template():
    """With weight_jitter=0, weights are tiled exactly from template."""
    from sim.replicas import (
        ReplicaConfig, replicate_wiring_plan_with_seeds,
    )
    template = {
        "g": {
            "pre_indices": [0],
            "post_indices": [1],
            "initial_weights": [0.7],
            "plastic": False,
            "conn_type": "E_TO_MIX",
            "count": 1,
        },
    }
    plan = replicate_wiring_plan_with_seeds(
        template,
        replicas=[ReplicaConfig(replica_id=0, seed_offset=0),
                   ReplicaConfig(replica_id=1, seed_offset=1)],
        neurons_per_replica=2, weight_jitter=0.0,
    )
    assert plan["g"]["initial_weights"] == [0.7, 0.7]


# ---------- Task 4: NeuromodulatorManager scope="replica:N" ----------

def test_neuromod_scope_replica_resolves_to_block_indices():
    """NeuromodulatorManager.set_replica_indices() lets ModulatorTarget(scope='replica:N')
    resolve to that block's neuron indices."""
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np

    from sim.neuromodulators import (
        NeuromodulatorConfig, NeuromodulatorManager, ModulatorTarget,
    )
    from sim.replicas import ReplicaConfig, ReplicaManager

    # Build a 3-replica setup (4 neurons each, 12 total)
    rmgr = ReplicaManager(
        [ReplicaConfig(replica_id=i) for i in range(3)],
        neurons_per_replica=4,
    )
    rmgr.initialize()

    nm = NeuromodulatorConfig(
        name="ne",
        baseline=0.0,
        targets=[
            ModulatorTarget(target_type="excitability_drive",
                             scope="replica:1",  # only replica 1 affected
                             sensitivity=10.0),
        ],
    )
    mgr = NeuromodulatorManager([nm], dt_ms=1.0)
    mgr.initialize(n_neurons=12, cp_module=cp)
    mgr.set_replica_indices(rmgr.replica_indices_dict())
    mgr.set_concentration("ne", 1.0)

    drive = mgr.compute_excitability_drive_per_neuron()
    drive_np = cp.asnumpy(drive)
    # Replica 0 (neurons 0..3): 0 drive
    assert (drive_np[:4] == 0).all()
    # Replica 1 (neurons 4..7): +10 pA each
    assert np.allclose(drive_np[4:8], 10.0)
    # Replica 2 (neurons 8..11): 0 drive
    assert (drive_np[8:] == 0).all()


# ---------- T-end-to-end: 3-replica G9 reservoir runs cleanly ----------

def test_three_replica_g9_reservoir_runs_50_steps():
    """End-to-end: build a 3-replica block-diagonal G9 reservoir using
    replicate_wiring_plan, run it through the existing bridge for 50 steps.
    Validates the pipeline end-to-end at the wiring level (without yet
    needing Tasks 5+6+7+8 for full per-replica neuromodulator integration).
    """
    pytest.importorskip("cupy")
    import cupy as cp
    import numpy as np

    from sim import (
        SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.replicas import (
        ReplicaConfig, ReplicaManager, replicate_wiring_plan_with_seeds,
    )
    from research.runners.g9_runner import _build_g9_plan

    # Build a single-replica G9 plan
    base_cfg, base_plan = _build_g9_plan(seed=42)
    n_per_replica = base_cfg.num_neurons

    # Replicate to 3 blocks
    replicas = [ReplicaConfig(replica_id=i, seed_offset=42 + i) for i in range(3)]
    replicated_plan = replicate_wiring_plan_with_seeds(
        base_plan, replicas=replicas,
        neurons_per_replica=n_per_replica,
        weight_jitter=0.2,
    )

    # Build a giant cfg
    rmgr = ReplicaManager(replicas, neurons_per_replica=n_per_replica)
    rmgr.initialize()
    base_cfg.num_neurons = rmgr.total_neurons()

    sb = SimulationBridge(
        core_config=base_cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(base_cfg.max_synaptic_delay_ms / base_cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)

    # Set up per-replica inhibitory traits.
    new_traits = np.zeros(base_cfg.num_neurons, dtype=np.int32)
    layout = base_plan["layout"]
    for r in replicas:
        shift = r.replica_id * n_per_replica
        for i in layout["hidden_inh_idx"]:
            new_traits[i + shift] = 1
    sb.cp_traits = cp.asarray(new_traits)
    sb._cached_inhibitory_mask = None

    sb.inject_explicit_wiring(replicated_plan, output_inhibitory_indices=None)

    # Run 50 steps; should not crash.
    for _ in range(50):
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_step += 1

    # Sanity: total synapses = 3 * single-replica synapses
    expected_synapses = 3 * (
        base_plan["input_to_hidden"]["count"]
        + base_plan["hidden_recurrent"]["count"]
        + base_plan["hidden_to_motor"]["count"]
    )
    assert int(sb.cp_connections.nnz) == expected_synapses, (
        f"replicated wiring should have {expected_synapses} synapses, "
        f"got {int(sb.cp_connections.nnz)}"
    )
    sb.clear_simulation_state_and_gpu_memory()
