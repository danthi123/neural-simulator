"""Scenario for the Vikunja #203 RESIDUAL: write sites other than the Hebbian rule that move a synapse
declared plastic=False / plastic_internal=False.

Six isolated rules, each a single weight-write site or bookkeeping path in sim/bridge.py:
  scaling   - homeostatic synaptic scaling (bridge.py "5b"): data *= post_scales
  structural- structural-plasticity ELIMINATION (bridge.py "C3"): data[eliminate_mask] = 0.0
  reward    - the reward-modulated CLIP (bridge.py "C2"): clip(data, w_min, w_max) when no named gain array exists
  rebuild   - mask bookkeeping: a CSR-reordering rebuild (set_pathway_weights(add_missing=True)) leaves
              cp_synapse_plastic_mask misaligned, so the freeze silently applies to the wrong synapses
  compaction- the DEFERRED CSR compaction path after structural elimination (_compact_synapse_arrays), a second
              re-ordering the mask was never carried through
  combined  - scaling + structural elimination + reward modulation on ONE bridge, i.e. all three masked write
              sites live at once (the integration case the individual rules cannot show)
  hebbian   - the Hebbian LTP + decay + clip itself, armed ONLY through cfg.enforce_plastic_mask_beyond_hebbian
              (the 2026-09-02 fix's own flag, cfg.enforce_plastic_mask_in_hebbian, left OFF in BOTH arms). This is
              the rule the 2026-10-31 comprehension-battery instrument found to be the ONLY mover of the organ's
              frozen synapses: arming the umbrella field must arm the Hebbian sites too, or the freeze covers every
              site except the ones that actually write.

Each run builds a small bridge (6 neurons; 8 for `compaction`, whose eliminated set must sit BETWEEN surviving
populations for the re-packing to shift any index at all) with TWO measured populations:
  frozen  : all-to-all among neurons 0..3, declared plastic=False, NO named plasticity_gate  -> must not move
  control : all-to-all among neurons 4..5, declared plastic=True                            -> MUST move
The control population is what makes a zero in `frozen_max_delta` meaningful: it proves the rule actually ran.

run(rule, enforce, seed) -> {sha, frozen_max_delta, control_max_delta, frozen_n, control_n, fired_any}
`enforce=False` touches NOTHING new, so the identical file runs unchanged on pre-fix code (the byte-identity
selftest copies it into a pre-fix git worktree).
"""
import hashlib
import os

import numpy as np

N_NEURONS = 6
N_READS = 30
STEPS_PER_READ = 20
DRIVE_PA = 4000.0


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _base_cfg(rule, seed):
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 8 if rule == "compaction" else N_NEURONS
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
    if rule == "scaling":
        cfg.enable_synaptic_scaling = True
        cfg.homeostasis_target_rate = 0.9
        cfg.synaptic_scaling_rate = 0.05
    elif rule == "structural":
        cfg.enable_structural_plasticity = True
        cfg.struct_plast_update_interval_steps = 10
        cfg.struct_plast_elimination_rate = 0.05
        cfg.struct_plast_weight_threshold = 5.0
        cfg.struct_plast_target_density = 0.2
    elif rule == "reward":
        cfg.enable_reward_modulation = True
        cfg.current_reward_signal = 1.0
        cfg.reward_baseline = 0.0
        cfg.reward_learning_rate = 0.0
    elif rule == "combined":
        # three masked write sites live simultaneously on one bridge
        cfg.enable_reward_modulation = True
        cfg.current_reward_signal = 1.0
        cfg.reward_baseline = 0.0
        cfg.reward_learning_rate = 0.0
        cfg.enable_structural_plasticity = True
        cfg.struct_plast_update_interval_steps = 10
        cfg.struct_plast_elimination_rate = 1.0    # every weak plastic synapse is removed each interval
        cfg.struct_plast_weight_threshold = 5.0
        cfg.struct_plast_target_density = 0.0     # no formation: elimination only
        cfg.enable_synaptic_scaling = True
        cfg.homeostasis_target_rate = 0.9
        cfg.synaptic_scaling_rate = 0.05
    elif rule == "compaction":
        # the DEFERRED CSR compaction path: elimination removes synapses, then _compact_synapse_arrays re-packs
        # every per-synapse array. This is a second re-ordering the mask must survive (the 2026-07-31 fix
        # compacted five arrays and left the mask out).
        cfg.enable_structural_plasticity = True
        cfg.struct_plast_update_interval_steps = 10
        cfg.struct_plast_elimination_rate = 1.0     # every weak synapse is removed each interval
        cfg.struct_plast_weight_threshold = 5.0     # w0 = 0.02 is 'weak' for both populations
        cfg.struct_plast_target_density = 0.0       # no formation: elimination only
        cfg.enable_synaptic_scaling = True          # so the control population moves (positive control)
        cfg.homeostasis_target_rate = 0.9
        cfg.synaptic_scaling_rate = 0.05
    elif rule == "hebbian":
        cfg.enable_hebbian_learning = True
        cfg.hebbian_learning_rate = 0.1
        cfg.hebbian_weight_decay = 1e-4
    return cfg


def _pairs(*ranges):
    pre, post = [], []
    for lo, hi in ranges:
        for i in range(lo, hi):
            for j in range(lo, hi):
                if i != j:
                    pre.append(i)
                    post.append(j)
    return pre, post


def _mask_alignment(bridge, frozen_pairs, control_pairs):
    """How many DECLARED (pre,post) pairs the live cp_synapse_plastic_mask currently labels WRONG.

    The mask is positional; the pairs are coordinates. A rebuild re-sorts the synapses, so a mask that is not
    remapped by coordinate mislabels synapses even though every entry is still a True/False.
    """
    nnz = int(bridge.cp_connections.nnz)
    _m = bridge._ensure_gate_capacity("cp_synapse_plastic_mask", nnz, fill=False, dtype=np.bool_)
    mask = np.zeros(nnz, dtype=np.bool_) if _m is None else _host(_m)[:nnz]
    coo = bridge._get_cached_coo()
    coords = list(zip(_host(coo.row).tolist(), _host(coo.col).tolist()))
    expected = {p: False for p in frozen_pairs}
    expected.update({p: True for p in control_pairs})
    return sum(1 for i, c in enumerate(coords) if c in expected and bool(mask[i]) != expected[c])


def _coords(bridge):
    coo = bridge._get_cached_coo()
    return _host(coo.row).tolist(), _host(coo.col).tolist()


def _new_pair_stats(bridge, declared_pairs):
    """(n_synapses not in declared_pairs, how many of them the live mask labels PLASTIC).

    Structurally-formed or rebuild-added synapses are not named by any wiring plan, so under enforcement they must
    read as plastic (like the gain default of 1.0 = open gate). Without enforcement the mask array does not exist,
    so every such synapse reads as frozen -- the failure this metric measures.
    """
    nnz = int(bridge.cp_connections.nnz)
    _m = bridge._ensure_gate_capacity("cp_synapse_plastic_mask", nnz, fill=False, dtype=np.bool_)
    mask = np.zeros(nnz, dtype=np.bool_) if _m is None else _host(_m)[:nnz]
    coo = bridge._get_cached_coo()
    coords = list(zip(_host(coo.row).tolist(), _host(coo.col).tolist()))
    new = [i for i, c in enumerate(coords) if c not in declared_pairs]
    return len(new), sum(1 for i in new if bool(mask[i]))


def run(rule, enforce=False, seed=42, w0=None):
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    w0 = {
        "scaling": 20.0,
        "structural": 0.02,
        "compaction": 10.0,     # strong enough to survive elimination: the probe measures scaling drift + the mask surviving the compaction re-pack
        "reward": 100.0,      # above hebbian_max_weight -> the clip itself is the mover
        "combined": 100.0,    # clipped, then weak -> the three sites in sequence
        "rebuild": 20.0,
        "hebbian": 20.0,      # above hebbian_max_weight: LTP, decay and the clip all move it
    }[rule] if w0 is None else w0

    # numpy by DEFAULT, not by fiat: an explicit SIM_BACKEND=cupy at the call site must reach the engine, or a
    # "cupy run" silently executes on the CPU and the backend confirmation is fake (2026-10-31: the first GPU
    # job queued for this follow-up reported "NumPy backend" for every one of its 84 runs).
    os.environ.setdefault("SIM_BACKEND", "numpy")
    if rule == "hebbian":
        # Deliberately NOT the umbrella env var: this rule measures whether cfg.enforce_plastic_mask_beyond_hebbian
        # ALONE reaches the Hebbian write sites. With the env var set, both flags are on and the rule proves nothing.
        os.environ.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)
    elif enforce:
        os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "1"
    else:
        os.environ.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)

    cfg = _base_cfg(rule, seed)
    if rule == "hebbian":
        cfg.enforce_plastic_mask_beyond_hebbian = enforce
        cfg.enforce_plastic_mask_in_hebbian = False   # the 2026-09-02 flag stays OFF in both arms
    if rule == "compaction":
        # default 1000 steps would never trigger inside a 600-step probe
        _gpu = GPUConfig()
        _gpu.struct_plast_compaction_interval = 5
    else:
        _gpu = GPUConfig()
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=_gpu)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    fp, fq = _pairs((0, 4))
    cp_, cq = _pairs((4, 6))
    dp, dq = [], []
    if rule == "compaction":
        # 8 neurons, THREE populations so the eliminated set sits BETWEEN surviving ones: a tail-only removal
        # would not shift any index and the alignment instrument would read 0 in both arms (vacuous).
        # frozen rows 0,1 and 4,5 (weak, declared plastic=False) | doomed rows 2,3 (weak, plastic -> removed)
        # | control rows 6,7 (strong, plastic -> survives and is scaled)
        fp, fq = _pairs((0, 2), (4, 6))
        dp, dq = _pairs((2, 4))
        cp_, cq = _pairs((6, 8))
    wiring = {
        "frozen_recurrent": {"pre_indices": fp, "post_indices": fq,
                             "initial_weights": np.full(len(fp), w0, dtype=np.float32),
                             "plastic": False, "count": len(fp)},
        "control_recurrent": {"pre_indices": cp_, "post_indices": cq,
                              "initial_weights": np.full(len(cp_), (10.0 if rule == "compaction" else w0),
                                                         dtype=np.float32),
                              "plastic": True, "count": len(cp_)},
    }
    if dp:
        wiring["doomed_recurrent"] = {"pre_indices": dp, "post_indices": dq,
                                      "initial_weights": np.full(len(dp), 0.02, dtype=np.float32),
                                      "plastic": True, "count": len(dp)}
    bridge.inject_explicit_wiring(wiring)
    mask = _host(bridge.cp_synapse_plastic_mask)
    frozen_idx = np.where(~mask)[0]
    control_idx = np.where(mask)[0]
    declared = list(zip(fp, fq)) + list(zip(cp_, cq)) + list(zip(dp, dq))
    plastic_pairs = list(zip(cp_, cq)) + list(zip(dp, dq))   # doomed edges are plastic too, just weak
    alignment_at_build = _mask_alignment(bridge, list(zip(fp, fq)), plastic_pairs)
    new_at_build = _new_pair_stats(bridge, declared)

    if rule == "rebuild":
        # CSR-reordering rebuild: add edges that do not exist yet, so every per-synapse array is re-sorted.
        bridge.set_pathway_weights("frozen_recurrent", [0, 1], [0, 1], [18.0, 18.0], add_missing=True)
        cfg.enable_synaptic_scaling = True
        mask = _host(bridge.cp_synapse_plastic_mask)
        frozen_idx = np.where(~mask)[0]
        control_idx = np.where(mask)[0]
        alignment_after_rebuild = _mask_alignment(bridge, list(zip(fp, fq)), plastic_pairs)
        new_after_rebuild = _new_pair_stats(bridge, declared)
    else:
        alignment_after_rebuild = alignment_at_build
        new_after_rebuild = new_at_build

    w0_host = _host(bridge.cp_connections.data).copy()
    coords0 = list(zip(*_coords(bridge)))
    # weights are keyed by (pre,post) COORDINATE, not position: elimination, formation and CSR re-sorting move
    # synapses, so a positional snapshot would compare a synapse against some other synapse's starting weight.
    w0_coord = {c: float(w0_host[i]) for i, c in enumerate(coords0)}
    frozen_set = set(zip(fp, fq))
    control_set = set(zip(cp_, cq))

    def _max_delta(pop_set):
        w = _host(bridge.cp_connections.data)
        vals = [abs(w[i] - w0_coord.get(c, w[i])) for i, c in enumerate(zip(*_coords(bridge)))
                if c in pop_set and i < w.size]
        return float(max(vals)) if vals else 0.0

    fired_any = False
    series = []
    for _read in range(N_READS):
        for _s in range(STEPS_PER_READ):
            if bridge.cp_external_input_current is not None:
                bridge.cp_external_input_current[:] = DRIVE_PA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            if bool(_host(bridge.cp_prev_firing_states).any()):
                fired_any = True
        series.append(_max_delta(frozen_set))

    w_final = _host(bridge.cp_connections.data)
    h = hashlib.sha256()
    for attr in ("cp_connections", "cp_membrane_potential_v", "cp_recovery_variable_u"):
        obj = getattr(bridge, attr, None)
        if obj is None:
            continue
        arr = _host(obj.data if attr == "cp_connections" else obj).astype(np.float64)
        h.update(np.ascontiguousarray(arr).tobytes())

    # AFTER the sha: the deferred CSR COMPACTION path (structural elimination -> _compact_synapse_arrays) is a
    # second re-ordering the mask must survive. Measured last so it cannot perturb the hashed state.
    alignment_at_end = _mask_alignment(bridge, list(zip(fp, fq)), plastic_pairs)
    new_at_end = _new_pair_stats(bridge, declared)

    return {
        "rule": rule, "enforce": enforce, "seed": seed, "sha": h.hexdigest(),
        "frozen_max_delta": series[-1], "first_delta": series[0], "fired_any": fired_any,
        "mask_mislabelled_at_build": alignment_at_build,
        "mask_mislabelled_after_rebuild": alignment_after_rebuild,
        "mask_mislabelled_at_end": alignment_at_end,
        "new_synapses_total": new_after_rebuild[0], "new_synapses_labelled_plastic": new_after_rebuild[1],
        "new_synapses_total_at_end": new_at_end[0], "new_synapses_labelled_plastic_at_end": new_at_end[1],
        "control_max_delta": _max_delta(control_set),
        "frozen_n": int(frozen_idx.size), "control_n": int(control_idx.size),
        "nnz_final": int(bridge.cp_connections.nnz),
    }


if __name__ == "__main__":
    import json
    import sys
    _rule = sys.argv[1] if len(sys.argv) > 1 else "scaling"
    _enforce = "--enforce" in sys.argv
    _seed = 42
    for a in sys.argv[2:]:
        if a.startswith("--seed="):
            _seed = int(a.split("=", 1)[1])
    print(json.dumps(run(_rule, _enforce, _seed)))
