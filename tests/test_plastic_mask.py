"""Test that SimulationBridge.cp_synapse_plastic_mask gates STDP writes.

Strategy: build two small networks that are identical except one has all
synapses plastic, the other has all fixed. Drive both with identical Poisson
stimulus so STDP fires in both. Compare weight deltas. If the mask works,
the plastic net moves weights while the fixed net keeps them stable.
"""
import numpy as np
import pytest


def _build_tiny_net(seed, all_plastic):
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import (CoreSimConfig, StimulusPattern, StimulusChannel,
                            NeuronGroup, ExperimentConfig, ExperimentPhase,
                            ReadoutConfig)
    from sim.enums import (NeuronModel, StimulusPatternType,
                           ExperimentPhaseType, NeuronGroupRole)
    from experiment import ExperimentEngine

    cfg = CoreSimConfig()
    cfg.num_neurons = 10   # 5 pre, 5 post
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.inhibitory_trait_indices = []
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_watts_strogatz = False
    cfg.stdp_a_plus = 0.05
    cfg.stdp_a_minus = 0.04
    cfg.stdp_w_min = 0.0
    cfg.stdp_w_max = 2.0
    cfg.propagation_strength = 3.0
    cfg.inhibitory_propagation_strength = 1.0
    cfg.ou_std_current_pA = 0.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    pre_idx = list(range(5))
    post_idx = list(range(5, 10))
    # All-to-all pre->post
    pre, post = [], []
    for i in pre_idx:
        for j in post_idx:
            pre.append(i); post.append(j)
    w = np.full(len(pre), 0.5, dtype=np.float32)

    plan = {
        "conn": {
            "pre_indices": pre,
            "post_indices": post,
            "initial_weights": w,
            "plastic": all_plastic,
            "count": len(pre),
        },
    }
    bridge.inject_explicit_wiring(plan)

    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0

    engine = ExperimentEngine(cfg.num_neurons, cfg.dt_ms)
    ecfg = ExperimentConfig()
    ecfg.neuron_groups = [
        NeuronGroup(name="pre", role=NeuronGroupRole.INPUT.name, neuron_indices=pre_idx),
        NeuronGroup(name="post", role=NeuronGroupRole.OUTPUT.name, neuron_indices=post_idx),
    ]
    ecfg.readout = ReadoutConfig(rate_window_ms=100, spike_count_window_ms=100,
                                 rate_group_names=["pre", "post"])
    ecfg.phases = [ExperimentPhase(name="x",
                                   phase_type=ExperimentPhaseType.TRAINING.name,
                                   duration_ms=1e9)]
    engine.load_experiment(ecfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    # Poisson stimulus on both pre and post (drives both to fire reliably,
    # ensuring STDP pair events).
    rates_pre = [30.0] * len(pre_idx)
    rates_post = [30.0] * len(post_idx)
    pat_pre = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=1000.0, spike_duration_ms=2.0,
        rate_vector_hz=rates_pre,
    )
    pat_post = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=1000.0, spike_duration_ms=2.0,
        rate_vector_hz=rates_post,
    )
    ch_pre = StimulusChannel(name="c_pre", pattern=pat_pre,
                             target_neuron_indices=pre_idx,
                             onset_ms=0, duration_ms=2000, enabled=True)
    ch_post = StimulusChannel(name="c_post", pattern=pat_post,
                              target_neuron_indices=post_idx,
                              onset_ms=0, duration_ms=2000, enabled=True)
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([ch_pre, ch_post], engine.group_manager, cp)
    engine.phase_start_ms = 0.0

    return bridge, cp


def test_plastic_mask_freezes_fixed_synapses():
    pytest.importorskip("cupy")
    import cupy as cp

    bridge_fixed, _ = _build_tiny_net(seed=7, all_plastic=False)
    bridge_plastic, _ = _build_tiny_net(seed=7, all_plastic=True)

    w0_fixed = cp.asnumpy(bridge_fixed.cp_connections.data).copy()
    w0_plastic = cp.asnumpy(bridge_plastic.cp_connections.data).copy()
    assert np.allclose(w0_fixed, 0.5)
    assert np.allclose(w0_plastic, 0.5)

    # Mask presence
    assert bridge_fixed.cp_synapse_plastic_mask is not None
    assert bridge_plastic.cp_synapse_plastic_mask is None

    # Drive both sims for 500 ms so STDP pairs occur many times.
    for step in range(500):
        bridge_fixed._run_one_simulation_step()
        bridge_fixed.runtime_state.current_time_step += 1
        bridge_fixed.runtime_state.current_time_ms = (
            bridge_fixed.runtime_state.current_time_step * 1.0
        )
        bridge_plastic._run_one_simulation_step()
        bridge_plastic.runtime_state.current_time_step += 1
        bridge_plastic.runtime_state.current_time_ms = (
            bridge_plastic.runtime_state.current_time_step * 1.0
        )

    w1_fixed = cp.asnumpy(bridge_fixed.cp_connections.data)
    w1_plastic = cp.asnumpy(bridge_plastic.cp_connections.data)

    # Fixed: every weight exactly unchanged.
    assert np.allclose(w1_fixed, 0.5, atol=1e-6), (
        f"Fixed weights changed: max diff = "
        f"{np.abs(w1_fixed - 0.5).max():.6f}"
    )

    # Plastic: at least some weight moved meaningfully.
    max_change = float(np.abs(w1_plastic - 0.5).max())
    assert max_change > 0.01, (
        f"Plastic weights didn't move (max |dW|={max_change:.6f}); "
        f"test setup is wrong"
    )


def test_no_mask_means_all_plastic():
    """When inject_explicit_wiring is called with no plastic=False populations,
    cp_synapse_plastic_mask should stay None (back-compat with existing paths)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 3
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 5
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

    wiring_plan = {
        "pop_a": {"pre_indices": [0], "post_indices": [1],
                  "initial_weights": np.array([0.3], dtype=np.float32),
                  "plastic": True, "count": 1},
        "pop_b": {"pre_indices": [0], "post_indices": [2],
                  "initial_weights": np.array([0.3], dtype=np.float32),
                  "plastic": True, "count": 1},
    }
    bridge.inject_explicit_wiring(wiring_plan)
    assert bridge.cp_synapse_plastic_mask is None


def test_mask_aligned_with_csr_order():
    """When populations have a mix of plastic and non-plastic synapses, the
    mask must align with cp_connections.data's internal CSR order."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 6
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 11
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

    # Plastic: (0,3), (0,4). Fixed: (1,3), (1,4). Verify CSR-aligned mask
    # labels each (pre, post) correctly.
    plan = {
        "plastic": {
            "pre_indices": [0, 0],
            "post_indices": [3, 4],
            "initial_weights": np.array([0.2, 0.3], dtype=np.float32),
            "plastic": True,
            "count": 2,
        },
        "fixed": {
            "pre_indices": [1, 1],
            "post_indices": [3, 4],
            "initial_weights": np.array([0.4, 0.5], dtype=np.float32),
            "plastic": False,
            "count": 2,
        },
    }
    bridge.inject_explicit_wiring(plan)

    coo = bridge.cp_connections.tocoo(copy=False)
    pre_h = cp.asnumpy(coo.row)
    post_h = cp.asnumpy(coo.col)
    mask_h = cp.asnumpy(bridge.cp_synapse_plastic_mask)

    # For every synapse: if pre=0 then plastic, if pre=1 then fixed.
    for pre, post, is_plastic in zip(pre_h, post_h, mask_h):
        if pre == 0:
            assert is_plastic, f"Synapse (0,{post}) should be plastic"
        elif pre == 1:
            assert not is_plastic, f"Synapse (1,{post}) should be fixed"
