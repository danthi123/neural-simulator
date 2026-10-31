"""Vikunja #203 residual probe: which runtime write sites STILL move a synapse declared
plastic=False (cp_synapse_plastic_mask=False, no named plasticity_gate) after the
Hebbian-path fix (merge 509c7137b, flag BRAIN_ENFORCE_PLASTIC_MASK /
cfg.enforce_plastic_mask_in_hebbian, default OFF)?

Each arm isolates ONE rule that writes cp_connections.data and measures max|w - w0| on the
frozen pathway over 30 reads, flag OFF vs flag ON (ON = the Hebbian enforcement is armed).
CPU/numpy only; no GPU.

Read sites surveyed by grep over sim/bridge.py (all writes to cp_connections.data):
  STDP fused path (:1268-1295)            mask-gated (always)
  inhibitory STDP (:1350-1397)            mask-gated (always)
  Hebbian LTP (:1429-1454, :10853-11028) mask-gated ONLY behind the flag
  BDSP (:11436-11456) / BTSP (:11554-11664) mask-gated (always)
  reward-modulated C2 (:11893-11949)      gain-gated only; eligibility itself is mask-gated (:11781)
  synaptic scaling 5b (:12287-12301)      gain-gated only -> NOT mask-gated
  structural elimination C3 (:11993)      not mask-gated at all
  bookkeeping: _remap_gate_indices_after_rebuild reads the DEAD attr `cp_plastic_mask`
"""
import json
import os
import sys

import numpy as np

N_NEURONS = 6
N_READS = 30
STEPS_PER_READ = 20
W0 = 20.0
DRIVE_PA = 4000.0


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _cfg(seed, rule):
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
    cfg.ou_std_current_pA = 0.0
    cfg.enable_hebbian_learning = False
    cfg.enable_stdp = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_synaptic_scaling = False
    cfg.enable_reward_modulation = False
    if rule == "hebbian":
        cfg.enable_hebbian_learning = True
        cfg.hebbian_learning_rate = 0.02
        cfg.hebbian_weight_decay = 0.0
        cfg.hebbian_max_weight = 60.0
        cfg.hebbian_min_weight = 0.0
    elif rule == "scaling":
        cfg.enable_synaptic_scaling = True
        cfg.homeostasis_target_rate = 0.8
        cfg.synaptic_scaling_rate = 0.05
    elif rule == "structural":
        cfg.enable_structural_plasticity = True
        cfg.struct_plast_update_interval_steps = 10
        cfg.struct_plast_elimination_rate = 0.05
        cfg.struct_plast_weight_threshold = 5.0   # W0=20 is BELOW nothing; use a low frozen weight arm below
        cfg.struct_plast_target_density = 0.9
    elif rule == "reward":
        cfg.enable_reward_modulation = True
        cfg.reward_learning_rate = 0.5
    return cfg


def run(rule, seed=42, enforce=False, seed_eligibility=False, frozen_w0=W0):
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    os.environ["SIM_BACKEND"] = "numpy"
    if enforce:
        os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "1"
    else:
        os.environ.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)
    cfg = _cfg(seed, rule)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)

    pre, post = [], []
    for i in range(N_NEURONS):
        for j in range(N_NEURONS):
            if i != j:
                pre.append(i)
                post.append(j)
    plan = {
        "frozen_recurrent": {
            "pre_indices": pre, "post_indices": post,
            "initial_weights": np.full(len(pre), frozen_w0, dtype=np.float32),
            "plastic": False, "count": len(pre),
        }
    }
    bridge.inject_explicit_wiring(plan)
    mask = _host(bridge.cp_synapse_plastic_mask)
    w0 = _host(bridge.cp_connections.data).copy()
    n_frozen = int(mask.size)

    if seed_eligibility:
        bridge.cp_eligibility_trace[:] = 0.5
        bridge.reward_prediction_error = 1.0

    series = []
    fired_any = False
    steps = 0
    for _read in range(N_READS):
        for _s in range(STEPS_PER_READ):
            if bridge.cp_external_input_current is not None:
                bridge.cp_external_input_current[:] = DRIVE_PA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            steps += 1
            if bool(_host(bridge.cp_prev_firing_states).any()):
                fired_any = True
        w = _host(bridge.cp_connections.data)
        series.append(float(np.abs(w[:n_frozen] - w0[:n_frozen]).max()))

    return {
        "rule": rule, "enforce": enforce, "steps": steps, "fired_any": fired_any,
        "frozen_max_delta": series[-1], "first_delta": series[0],
        "nnz_final": int(bridge.cp_connections.nnz),
        "mask_size_final": int(bridge.cp_synapse_plastic_mask.size if bridge.cp_synapse_plastic_mask is not None else -1),
    }


def mask_alignment_after_rebuild(seed=42):
    """Does cp_synapse_plastic_mask still label the SAME (pre,post) pairs after a CSR-reordering
    rebuild (set_pathway_weights(add_missing=True))?  The remap helper reads the DEAD attribute
    `cp_plastic_mask`, so the live mask is never remapped."""
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel
    os.environ["SIM_BACKEND"] = "numpy"
    cfg = _cfg(seed, "none")
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    pre, post, w = [], [], []
    for i in range(N_NEURONS):
        for j in range(N_NEURONS):
            if i != j:
                pre.append(i); post.append(j); w.append(20.0)
    bridge.inject_explicit_wiring({"frozen_recurrent": {
        "pre_indices": pre, "post_indices": post,
        "initial_weights": np.asarray(w, dtype=np.float32), "plastic": False, "count": len(pre)}})
    mask_before = _host(bridge.cp_synapse_plastic_mask).copy()
    coo_before = bridge._get_cached_coo()
    pairs_before = list(zip(coo_before.row.tolist(), coo_before.col.tolist()))
    added = bridge.set_pathway_weights("frozen_recurrent", {0: 1}, add_missing=True)
    coo_after = bridge._get_cached_coo()
    pairs_after = list(zip(coo_after.row.tolist(), coo_after.col.tolist()))
    mask_after = _host(bridge.cp_synapse_plastic_mask)
    # compare mask value per (pre,post) coordinate before vs after
    before_map = {p: bool(m) for p, m in zip(pairs_before, mask_before)}
    mismatch = sum(1 for p, m in zip(pairs_after, mask_after) if before_map.get(p, True) != bool(m))
    return {"synapses_added": int(added), "nnz_before": len(pairs_before), "nnz_after": len(pairs_after),
            "mask_size_after": int(mask_after.size), "mask_coord_mismatches": int(mismatch),
            "dead_attr_exists": hasattr(bridge, "cp_plastic_mask") and getattr(bridge, "cp_plastic_mask", None) is not None}


if __name__ == "__main__":
    out = {"seed": 42, "arms": []}
    for rule in ("hebbian", "scaling", "structural", "reward"):
        for enforce in (False, True):
            out["arms"].append(run(rule, enforce=enforce, seed_eligibility=(rule == "reward")))
    out["rebuild_mask_alignment"] = mask_alignment_after_rebuild()
    print(json.dumps(out, indent=2))
