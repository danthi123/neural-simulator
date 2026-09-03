"""Read-isolation scenario for BRAIN_ENFORCE_PLASTIC_MASK / enforce_plastic_mask_in_hebbian.

Self-contained, backend-agnostic (numpy or cupy) scenario that reproduces the confirmed read-isolation bug:
a RegionPathway(plastic=False) recurrent pathway WITHOUT a named zeroed plasticity_gate drifts under the runtime
Hebbian rule purely from being read (co-activity), because the Hebbian LTP/decay/clip path historically consulted
only cp_plasticity_rate_gain, never cp_synapse_plastic_mask (the comprehension organ measured 13.8->56.1 max-weight-
delta over 30 reads). This module is COPIED VERBATIM into a pre-fix git worktree by the selftest so the exact same
scenario runs against pre-fix and fixed code; therefore it must reference only stable, long-lived engine APIs and must
NOT touch the new config field / env var when enforce=False (so it runs unchanged on pre-fix code).

run(enforce, seed) returns a dict: {sha, drift_series, final_max_delta, fired_any}.
  - sha: sha256 over the full post-run substrate state (weights + membrane V + recovery U), for byte-identical checks.
  - drift_series: max |w - w0| over the non-plastic recurrent pathway after each of N_READS reads.
  - final_max_delta: drift_series[-1].
"""
import hashlib
import os

import numpy as np

N_NEURONS = 6
N_READS = 30
STEPS_PER_READ = 10
W0 = 13.8            # initial non-plastic recurrent weight (matches the comprehension-organ read-drift start)
W_MAX = 56.1         # hebbian_max_weight (matches the measured drift ceiling 56.1)
DRIVE_PA = 4000.0    # strong constant external drive so every recurrent neuron co-fires each step


def _to_host(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def run(enforce, seed=42):
    # Import inside the function so SIM_BACKEND (set by the caller before spawning) is honored on first import.
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = N_NEURONS
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.inhibitory_trait_indices = []
    cfg.enable_watts_strogatz = False
    # Isolate the plain Hebbian rule: everything else off so the ONLY weight motion is Hebbian LTP on the
    # non-plastic recurrent pathway (the exact bug surface).
    cfg.enable_hebbian_learning = True
    cfg.enable_stdp = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.hebbian_learning_rate = 0.02
    cfg.hebbian_max_weight = W_MAX
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 0.0   # isolate potentiation; decay would confound the drift measurement
    cfg.ou_std_current_pA = 0.0

    # Only touch the NEW knobs when enforce is requested, so enforce=False runs unchanged on pre-fix code.
    if enforce:
        cfg.enforce_plastic_mask_in_hebbian = True
        os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "1"
    else:
        os.environ.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Recurrent all-to-all (no self-loops) among all neurons, declared NON-PLASTIC (plastic=False) and WITHOUT
    # any named plasticity_gate -> cp_synapse_plastic_mask is built, cp_plasticity_rate_gain stays None. This is
    # exactly the process-shared read-only organ configuration that drifts under the current Hebbian rule.
    pre, post = [], []
    for i in range(N_NEURONS):
        for j in range(N_NEURONS):
            if i != j:
                pre.append(i)
                post.append(j)
    w = np.full(len(pre), W0, dtype=np.float32)
    plan = {
        "recurrent_fixed": {
            "pre_indices": pre,
            "post_indices": post,
            "initial_weights": w,
            "plastic": False,
            "count": len(pre),
        },
    }
    bridge.inject_explicit_wiring(plan)

    w0_host = _to_host(bridge.cp_connections.data).copy()

    drift_series = []
    fired_any = False
    for _read in range(N_READS):
        for _s in range(STEPS_PER_READ):
            # Re-assert strong drive every step so all recurrent neurons fire and co-activity Hebbian pairs form.
            if bridge.cp_external_input_current is not None:
                bridge.cp_external_input_current[:] = DRIVE_PA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            if bool(_to_host(bridge.cp_prev_firing_states).any()):
                fired_any = True
        w_now = _to_host(bridge.cp_connections.data)
        drift_series.append(float(np.abs(w_now - w0_host).max()))

    # Full-state SHA over the substrate quantities a Hebbian run can touch.
    h = hashlib.sha256()
    for attr in ("cp_connections", "cp_membrane_potential_v", "cp_recovery_variable_u"):
        obj = getattr(bridge, attr, None)
        if obj is None:
            continue
        data = obj.data if attr == "cp_connections" else obj
        arr = _to_host(data).astype(np.float64)  # float64 canonicalizes numpy/cupy float32 bit noise consistently
        h.update(np.ascontiguousarray(arr).tobytes())

    return {
        "sha": h.hexdigest(),
        "drift_series": drift_series,
        "final_max_delta": drift_series[-1] if drift_series else 0.0,
        "fired_any": fired_any,
    }


if __name__ == "__main__":
    import json
    import sys
    _enforce = "--enforce" in sys.argv
    _seed = 42
    for a in sys.argv[1:]:
        if a.startswith("--seed="):
            _seed = int(a.split("=", 1)[1])
    print(json.dumps(run(_enforce, _seed)))
