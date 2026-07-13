"""ON-BRIDGE NODE-PERTURBATION cheap-first de-risk (the riskiest assumption of the on-bridge NP realization): can the
sim's OWN reward-modulated three-factor machinery deliver NODE-PERTURBATION credit WITHOUT a host weight-write?

THE MECHANISM the on-bridge NP realization rests on: perturb the POST (hidden) region with an intrinsic-noise current
xi during a settle window -> the post neurons fire MORE where xi>0 -> the pre->post STDP builds MORE eligibility on the
synapses whose post fired more (pre-before-post coincidence) -> so the eligibility trace ~ xi x (pre activity) = exactly
the node-perturbation eligibility. Then a global reward = -dL scales it (the sim's `enable_reward_modulation` three-factor
update). NO host weight-write; the sim's committed plasticity does the NP step.

THE RISKIEST ASSUMPTION (this probe): does `cp_eligibility_trace` on the pre->post pathway actually CORRELATE with the
injected post perturbation xi? Build a 2-region (pre -> post) bridge with STDP + reward-modulation ON; drive pre; run a
settle with +xi vs -xi current on post; read the pre->post eligibility for each; check corr( elig(+xi) - elig(-xi),
per-post-neuron xi ). GO = a clear positive correlation (the perturbation shapes the eligibility) -> the on-bridge NP
realization is viable. NO permanent `sim/` edit (uses public config + arrays).

Run: SIM_BACKEND=numpy python -m research.runners._np_onbridge_eligibility_probe --seeds 42 43 44
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def build_probe_bridge(seed, n_pre=24, n_post=16, fwd_w=8.0):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(); cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.actual_seed_used = int(seed); cfg.ou_std_current_pA = 0.0
    cfg.enable_brain_region_framework = True
    # the three-factor machinery under test: STDP builds eligibility, reward-modulation applies it.
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.current_reward_signal = 0.0                 # keep reward 0 during the probe (we read the ELIGIBILITY, not the update)
    for flag in ("enable_hebbian_learning", "enable_homeostasis", "enable_structural_plasticity",
                 "enable_short_term_plasticity", "enable_nmda", "enable_input_divisive_norm", "enable_bdsp"):
        setattr(cfg, flag, False)
    cfg.stdp_w_max = 50.0
    regions = [
        BrainRegion(name="pre", n_neurons=n_pre, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="post", n_neurons=n_post, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [RegionPathway(from_region="pre", to_region="post", density=1.0,
                              weight_mean=float(fwd_w), weight_jitter=0.5, plastic=True)]
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb, cfg


def run(seed, settle=60, pre_drive=600.0, xi_pA=120.0, post_bias=0.0, fwd_w=8.0):
    from sim.backend import to_host, from_host
    sb, cfg = build_probe_bridge(seed, fwd_w=fwd_w)
    rm = sb.region_manager
    idx_pre = np.asarray(list(rm.indices("pre")), int)
    idx_post = np.asarray(list(rm.indices("post")), int)
    n = int(cfg.num_neurons)
    # pre->post pathway mask over the cached COO (row in pre, col in post)
    coo = sb._get_cached_coo()
    row = np.asarray(to_host(coo.row)).astype(int); col = np.asarray(to_host(coo.col)).astype(int)
    pre_set = set(idx_pre.tolist()); post_set = set(idx_post.tolist())
    mask = np.array([(r in pre_set and c in post_set) for r, c in zip(row, col)])
    col_of = col[mask]                                            # post-neuron (col) of each pre->post synapse

    rng = np.random.default_rng(seed * 17 + 3)
    xi = (rng.standard_normal(len(idx_post))).astype(np.float32)   # per-post-neuron perturbation pattern (sign matters)

    def _reset():
        if getattr(sb, "cp_izh_c_reset", None) is not None:
            sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
        else:
            sb.cp_membrane_potential_v[:] = -65.0
        sb.cp_recovery_variable_u[:] = 0.0
        if getattr(sb, "cp_firing_states", None) is not None:
            sb.cp_firing_states[:] = False
        if getattr(sb, "cp_eligibility_trace", None) is not None:
            sb.cp_eligibility_trace[:] = 0.0
        for a in ("cp_conductance_g_e", "cp_conductance_g_i"):
            arr = getattr(sb, a, None)
            if arr is not None:
                arr[:] = 0.0

    def _elig_after(sign):
        _reset()
        drive = np.zeros(n, dtype=np.float32)
        drive[idx_pre] = pre_drive
        drive[idx_post] += post_bias                             # standing drive so post fires (STDP needs post spikes)
        drive[idx_post] += sign * xi_pA * xi                     # the node perturbation = intrinsic-noise current on post
        dev = from_host(drive)
        fired_pre = 0.0; fired_post = 0.0
        for _ in range(settle):
            sb.cp_external_input_current[:] = dev
            sb._run_one_simulation_step()
            fs = np.asarray(to_host(sb.cp_firing_states)).astype(float)
            fired_pre += fs[idx_pre].mean(); fired_post += fs[idx_post].mean()
        _elig_after.rates = (fired_pre / settle, fired_post / settle)
        el = np.asarray(to_host(sb.cp_eligibility_trace)).astype(float)[:len(mask)]   # align to cp_connections.data/COO order (nnz)
        el_m = el[mask]
        # per-post-neuron mean eligibility over its incoming pre->post synapses
        per_post = np.array([el_m[col_of == p].mean() if np.any(col_of == p) else 0.0 for p in idx_post])
        return per_post

    e_plus = _elig_after(+1.0); e_minus = _elig_after(-1.0)
    d = e_plus - e_minus                                          # the perturbation-driven eligibility difference
    # correlation of the eligibility-difference with the perturbation pattern xi (per post neuron)
    if d.std() > 1e-9 and xi.std() > 1e-9:
        corr = float(np.corrcoef(d, xi)[0, 1])
    else:
        corr = 0.0
    rates = getattr(_elig_after, "rates", (0.0, 0.0))
    return {"seed": seed, "corr_elig_xi": round(corr, 3),
            "elig_plus_mean": round(float(e_plus.mean()), 4), "elig_minus_mean": round(float(e_minus.mean()), 4),
            "d_std": round(float(d.std()), 5), "pre_rate": round(rates[0], 3), "post_rate": round(rates[1], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--pre-drive", type=float, default=600.0); ap.add_argument("--xi-pA", type=float, default=120.0)
    ap.add_argument("--post-bias", type=float, default=0.0); ap.add_argument("--fwd-w", type=float, default=8.0)
    a = ap.parse_args()
    corrs = []
    for s in a.seeds:
        r = run(s, a.settle, a.pre_drive, a.xi_pA, a.post_bias, a.fwd_w)
        corrs.append(r["corr_elig_xi"])
        print(f"[elig-probe seed={s}] corr(elig_diff, xi)={r['corr_elig_xi']:+.3f} "
              f"elig+={r['elig_plus_mean']:.4f} elig-={r['elig_minus_mean']:.4f} d_std={r['d_std']:.5f} "
              f"| pre_rate={r['pre_rate']:.3f} post_rate={r['post_rate']:.3f}", flush=True)
    mc = float(np.mean(corrs))
    ngo = sum(1 for c in corrs if c > 0.3)
    print(f"[elig-probe] mean corr {mc:+.3f} | {ngo}/{len(corrs)} seeds corr>0.3 "
          f"-> {'GO (STDP eligibility IS shaped by the perturbation -> on-bridge NP viable)' if ngo >= max(1, len(corrs)-1) else 'no (eligibility not perturbation-correlated -> need a different eligibility route)'}", flush=True)


if __name__ == "__main__":
    main()
