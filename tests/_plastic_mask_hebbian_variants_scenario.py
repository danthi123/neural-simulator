"""Backend-hermetic Hebbian-VARIANT scenario for cp_synapse_plastic_mask (Vikunja #203 residual 2,
2026-09-18 follow-up to the STDP freeze test in `tests/_plastic_mask_stdp_scenario.py`).

The main Hebbian LTP/decay/clip block (sim/bridge.py ~9950-9159) is guarded by
`self._hebbian_plastic_mask_enforced(cfg)` (bridge.py:1250) so `plastic=False` synapses are frozen --
but that guard is applied at THREE separate write sites that differ per rule variant:

  * rate-window Hebbian   (cfg.hebbian_rate_window=True, cfg.hebbian_bcm=0.0)  -- mask applied to
    `delta_weights` at bridge.py:10066-10069, inside the `elif _rate_win:` branch.
  * BCM                   (cfg.hebbian_rate_window=True, cfg.hebbian_bcm>0.0)  -- mask applied to
    `_dw_b` at bridge.py:10013-10014, the BCM sub-branch of the same `elif _rate_win:` block.
  * branchless Hebbian    (cfg.enable_branchless_plasticity=True)             -- a SEPARATE code path,
    `_apply_branchless_hebbian` (bridge.py:1214), mask applied to `delta_all` at bridge.py:1240-1247.

Each of these was "structurally guarded but not individually SHA-tested" per Vikunja #203 residual 2:
a code read shows the guard present at all three write sites, but no regression test previously froze
a `plastic=False` synapse under each variant specifically and confirmed a companion `plastic=True`
synapse still moves (the false-freeze control, same discipline as the STDP freeze test).

WHY A SUBPROCESS, NOT AN IN-PROCESS FIXTURE: identical reasoning to
`tests/_plastic_mask_stdp_scenario.py` -- `sim/bridge.py` resolves numpy-vs-cupy ONCE at module level
on its first-ever import in the process, so a sibling test file that imports `sim.bridge` first under
numpy would permanently pin this process to numpy, and this exact tiny seeded network produces ZERO
spikes over several hundred steps under numpy at this scale (2026-09-08 finding). Running in a fresh
subprocess with SIM_BACKEND=cupy set before `sim.bridge`'s first import is immune to that.

TEST-ONLY CONFIG CHOICES (not production defaults, and not a sim/ change -- these are scenario-local
CoreSimConfig overrides, same spirit as the STDP scenario boosting stdp_a_plus/a_minus far above
production values for a clean signal in a short window):
  - `hebbian_learning_rate` raised well above the 0.0005 production default so a handful of
    coincidences over a few hundred steps produce an unambiguous weight change.
  - `hebbian_coactivity_thresh` lowered near zero for the rate-window/BCM variants. The production
    value (0.25) gates rule SPECIFICITY (a separate, already-studied concern); this test's job is only
    to confirm the mask zeroes the delta at the write site, so decoupling "did masking work" from
    "was the specificity threshold biologically tuned" avoids a false negative from under-firing at
    this synthetic tiny-network scale.
  - `hebbian_bcm_theta_alpha` lowered well below the 0.001 default so the BCM sliding threshold
    theta_M stays near zero for the whole run. Without this, BCM's homeostatic design would let
    theta_M rise to track <y^2> and could self-cancel net drift by the end of the run (LTP early,
    LTD once theta_M catches up) -- a real dynamical property of BCM, not a masking bug, but one that
    would make the "plastic control moves" assertion flaky if theta_M were left at its default rate.
    Keeping the run in the LTP-dominated regime isolates the property this test actually checks.

run(seed, all_plastic, variant) returns a dict: {fired_any, mask_present, w0_allclose_0p5,
w1_allclose_w0, max_change}, mirroring the STDP scenario's return shape exactly.
"""
import json
import os
import sys

import numpy as np

_VARIANTS = ("rate_window", "bcm", "branchless")


def run(seed, all_plastic, variant):
    assert variant in _VARIANTS, "unknown variant %r (expected one of %s)" % (variant, _VARIANTS)

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
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_watts_strogatz = False
    cfg.propagation_strength = 3.0
    cfg.inhibitory_propagation_strength = 1.0
    cfg.ou_std_current_pA = 0.0

    # Force-enable the additive, default-off Hebbian plastic-mask enforcement directly via cfg
    # (equivalent to BRAIN_ENFORCE_PLASTIC_MASK=1 -- see _hebbian_plastic_mask_enforced, bridge.py:1250).
    cfg.enforce_plastic_mask_in_hebbian = True

    # Boosted vs. the 0.0005 production default -- see module docstring.
    cfg.hebbian_learning_rate = 0.05
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = 2.0
    cfg.hebbian_symmetric = False

    if variant == "rate_window":
        cfg.hebbian_rate_window = True
        cfg.hebbian_bcm = 0.0
        cfg.hebbian_coactivity_decay = 0.9
        cfg.hebbian_coactivity_thresh = 1e-6
        cfg.enable_branchless_plasticity = False
    elif variant == "bcm":
        cfg.hebbian_rate_window = True
        cfg.hebbian_bcm = 0.5
        cfg.hebbian_coactivity_decay = 0.9
        cfg.hebbian_coactivity_thresh = 1e-6
        cfg.hebbian_bcm_pre_floor = 1e-6
        cfg.hebbian_bcm_theta_alpha = 2e-4  # kept low -- see module docstring
        cfg.enable_branchless_plasticity = False
    elif variant == "branchless":
        cfg.hebbian_rate_window = False
        cfg.hebbian_bcm = 0.0
        cfg.enable_branchless_plasticity = True

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
    _variant = "rate_window"
    for a in sys.argv[1:]:
        if a.startswith("--seed="):
            _seed = int(a.split("=", 1)[1])
        elif a == "--all-plastic":
            _all_plastic = True
        elif a.startswith("--variant="):
            _variant = a.split("=", 1)[1]
    print(json.dumps(run(_seed, _all_plastic, _variant)))
