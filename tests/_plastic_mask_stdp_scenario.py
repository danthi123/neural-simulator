"""Backend-hermetic STDP scenario for cp_synapse_plastic_mask (Vikunja #203 follow-up, 2026-09-08).

Runs `SimulationBridge` with real STDP dynamics and reports whether a `plastic=False` pathway's
weights stay frozen while a `plastic=True` pathway's weights move.

WHY A SUBPROCESS, NOT AN IN-PROCESS FIXTURE (the actual root cause this file fixes): `sim/bridge.py`
resolves its numpy-vs-cupy backend ONCE, at MODULE level (`cp, _backend_name = get_backend()`,
sim/bridge.py:44) -- the FIRST time `sim.bridge` is imported anywhere in a process. Every later
`SimulationBridge` built in that SAME process reuses that same `cp` binding, no matter what
`SIM_BACKEND` is set to afterward (an in-process env-var override or fixture cannot undo it once
`sim.bridge` is already in `sys.modules`). `tests/test_enforce_plastic_mask.py` does a module-level
`os.environ.setdefault("SIM_BACKEND", "numpy")` and, being collected alphabetically before
`test_plastic_mask.py`, is typically the first thing in the whole pytest session to import
`sim.bridge` -- permanently pinning the WHOLE test process to the numpy backend before
`test_plastic_mask.py`'s own `import cupy as cp` line ever runs.

That matters here because this exact tiny seeded network (10 Izhikevich neurons, seed=7, Poisson
drive) produces ZERO spikes over 500 steps under the numpy backend (verified empirically: both the
all-plastic and all-fixed bridges show `cp_last_spike_time` stuck at the -1000 sentinel throughout --
a cross-backend RNG/heterogeneity divergence at this tiny scale, the same class of effect as the
2026-06-09 N9 CuPy-vs-numpy firing-margin finding, not a plasticity bug), while it fires and produces
real STDP weight motion under cupy. So `test_plastic_mask_freezes_fixed_synapses`, which explicitly
targets cupy (`pytest.importorskip("cupy")` + `import cupy as cp`), silently degraded into "did
anything fire at all" whenever an earlier-collected test file happened to import `sim.bridge` first
under numpy -- and its failure was misread in FAILURE_LOG.md as evidence that "the STDP-update path
is NOT covered [by cp_synapse_plastic_mask]". It is covered (sim/bridge.py:10241, unconditional,
matching BDSP/BTSP) -- confirmed by direct code read. The failure was a process-backend-stickiness
instrument artifact, not a plasticity-gating regression.

Running this scenario in its OWN subprocess with SIM_BACKEND=cupy set in that subprocess's
environment BEFORE `sim.bridge` is ever imported guarantees the real cupy backend regardless of
what any sibling test file did earlier in the parent pytest process.

run(seed, all_plastic) returns a dict: {fired_any, mask_present, w0, w1, max_change}.
"""
import json
import os
import sys

import numpy as np


def run(seed, all_plastic):
    # Imports are INSIDE the function (not module-level) so the caller's SIM_BACKEND env var,
    # set before spawning this subprocess, is honored on sim.bridge's first-ever import.
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import (CoreSimConfig, StimulusPattern, StimulusChannel,
                             NeuronGroup, ExperimentConfig, ExperimentPhase,
                             ReadoutConfig)
    from sim.enums import (NeuronModel, StimulusPatternType,
                            ExperimentPhaseType, NeuronGroupRole)
    from sim.backend import get_backend
    from experiment import ExperimentEngine

    cp, _backend_name = get_backend()
    assert _backend_name == "cupy", (
        "this scenario must run under the cupy backend (got %r) -- invoke it as a "
        "subprocess with SIM_BACKEND=cupy set BEFORE sim.bridge's first import, "
        "since sim/bridge.py binds its backend once at module level" % _backend_name
    )

    def _to_host(arr):
        return arr.get() if hasattr(arr, "get") else np.asarray(arr)

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
    pre, post = [], []
    for i in pre_idx:
        for j in post_idx:
            pre.append(i)
            post.append(j)
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

    w0 = _to_host(bridge.cp_connections.data).copy()
    mask_present = bridge.cp_synapse_plastic_mask is not None

    fired_any = False
    for _step in range(500):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * 1.0
        if bool(_to_host(bridge.cp_prev_firing_states).any()):
            fired_any = True

    w1 = _to_host(bridge.cp_connections.data)
    max_change = float(np.abs(w1 - w0).max())

    return {
        "fired_any": fired_any,
        "mask_present": mask_present,
        "w0_allclose_0p5": bool(np.allclose(w0, 0.5)),
        "w1_allclose_w0": bool(np.allclose(w1, w0, atol=1e-6)),
        "max_change": max_change,
    }


if __name__ == "__main__":
    _seed = 7
    _all_plastic = False
    for a in sys.argv[1:]:
        if a.startswith("--seed="):
            _seed = int(a.split("=", 1)[1])
        elif a == "--all-plastic":
            _all_plastic = True
    print(json.dumps(run(_seed, _all_plastic)))
