"""(de-risk A, mechanism sanity) Confirm an INHIBITORY-TRAIT FS pool wired concept->FS (E_TO_I) and
FS->concept (I_TO_E) produces conductance-based SHUNTING (divisive) inhibition on the core bridge --
i.e. firing FS neurons raise g_i and the term g_i*(E_i - V) divides down concept firing. This is the
substrate the divisive-normalization cleanup rests on.

Critical detail discovered in bridge.py (lines 5046-5070): the g_e vs g_i routing depends on whether the
PRESYNAPTIC neuron is INHIBITORY (cp_traits in inhibitory_trait_indices), NOT on the wiring plan's
informational conn_type string. So the prior WTA probe (enable_inhibitory_neurons defaulted False) added
its "I_TO_E" weights to g_e = EXCITATION, which is why WTA "hurt" (0/45). Here we set the FS pool to an
inhibitory trait + enable_inhibitory_neurons so it truly shunts.

Test: drive a small concept pool, measure its firing WITHOUT the FS pool active vs WITH (FS pool driven by
the concept pool). If the FS pool shunts, concept firing drops, and -- the divisive signature -- the
suppression is multiplicative (a strongly-driven concept stays relatively higher than a weakly-driven one,
preserving rank, unlike subtractive inhibition which would floor both).

  python -m research.findings.raw._divnorm_mechanism_sanity
"""
from __future__ import annotations
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

INH_TRAIT = 1


def build(seed, M, n_fs, w_concept_in, w_c_to_fs, w_fs_to_c, enable_inh):
    """M concept neurons [0,M), n_fs FS neurons [M, M+n_fs). Concept neurons get external drive;
    concept->FS (E_TO_I) pools activity; FS->concept (I_TO_E) shunts. FS neurons carry inhibitory trait."""
    N = M + n_fs
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 2 if enable_inh else 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0
    if enable_inh:
        cfg.enable_inhibitory_neurons = True
        cfg.inhibitory_trait_indices = [INH_TRAIT]

    concept = np.arange(0, M); fs = np.arange(M, M + n_fs)
    pre, post, w, ctype = [], [], [], []
    # concept -> FS (every concept excites every FS: pools total population activity)
    for c in range(M):
        for j in range(n_fs):
            pre.append(int(concept[c])); post.append(int(fs[j])); w.append(float(w_c_to_fs)); ctype.append("E_TO_I")
    # FS -> concept (every FS inhibits every concept: divisive feedback)
    for j in range(n_fs):
        for c in range(M):
            pre.append(int(fs[j])); post.append(int(concept[c])); w.append(float(w_fs_to_c)); ctype.append("I_TO_E")
    plan = {"net": {"pre_indices": pre, "post_indices": post,
                    "initial_weights": np.array(w, dtype=np.float32), "plastic": False,
                    "conn_type": "E_TO_E", "count": len(pre)}}
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # Force FS neurons to the inhibitory trait BEFORE the first step (mask is cached on first step).
    if enable_inh:
        xp, _ = get_backend()
        tr = bridge.cp_traits
        tr[:] = 0
        tr[xp.asarray(fs, dtype=tr.dtype)] = INH_TRAIT
        bridge.cp_traits = tr
        bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring(plan)
    xp, _ = get_backend()
    return bridge, xp.asarray(concept, dtype=xp.int64), xp.asarray(fs, dtype=xp.int64)


def run(bridge, concept_idx, fs_idx, drives, run_steps=200, reset_steps=20):
    xp, _ = get_backend()
    M = len(drives)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[concept_idx] = xp.asarray(np.asarray(drives, dtype=np.float32))
    bridge.cp_external_input_current[:] = cur
    acc_c = xp.zeros(M, dtype=xp.float64)
    acc_fs = 0.0
    for _ in range(run_steps):
        bridge._run_one_simulation_step()
        acc_c += bridge.cp_firing_states[concept_idx].astype(xp.float64)
        acc_fs += float(bridge.cp_firing_states[fs_idx].astype(xp.float64).sum())
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc_c) / run_steps, acc_fs / run_steps / max(1, len(fs_idx))


def main():
    M = 8; n_fs = 12
    # graded drives so we can see whether suppression preserves rank (divisive) or floors all (subtractive)
    drives = np.linspace(400, 2000, M)

    # baseline: FS pool present but FS->concept weight zero (no feedback)
    b0, ci, fi = build(42, M, n_fs, 0, 0.0, 0.0, enable_inh=True)
    r0, fs0 = run(b0, ci, fi, drives)
    print(f"[sanity] NO feedback (w_fs_to_c=0): concept rates {np.round(r0,3)}  fs_rate={fs0:.3f}")

    # with divisive feedback, inhibitory trait ON (true shunting). Need the FS pool to actually FIRE:
    # sweep concept->FS drive strength and FS->concept feedback strength.
    for w_cfs in (50.0, 150.0):
        for w_fs in (5.0, 20.0, 50.0):
            b1, ci, fi = build(42, M, n_fs, 0, w_cfs, w_fs, enable_inh=True)
            r1, fs1 = run(b1, ci, fi, drives)
            ratio = r1 / np.maximum(r0, 1e-9)
            print(f"[sanity] divisive (inh trait, w_cfs={w_cfs}, w_fs={w_fs}): concept {np.round(r1,3)}  "
                  f"fs={fs1:.3f}  supp_ratio {np.round(ratio,2)}  rank_preserved={bool(np.all(np.diff(r1)>=-1e-9))}")

    # CONTROL: same wiring but enable_inh False -> 'I_TO_E' weights add to g_e (the prior-probe bug); should NOT shunt
    b2, ci, fi = build(42, M, n_fs, 0, 8.0, 5.0, enable_inh=False)
    r2, fs2 = run(b2, ci, fi, drives)
    print(f"[sanity] CONTROL no-inh-trait (w_fs=5 -> excitation): concept {np.round(r2,3)}  fs={fs2:.3f}")


if __name__ == "__main__":
    main()
