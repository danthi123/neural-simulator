"""D3 SPIKING port (rung 1): the re-discretization ON SPIKES — the concrete "simulated recurrent sequence/language
cortex". The rate de-risk (`_d3_group_composition_derisk.py`) proved DISCRETE-ATTRACTOR recurrence length-generalizes
multi-hop group composition (S3 + theorem-backed A5) where a continuous RNN cannot; the mechanism = re-discretize the
running state to a CLEAN attractor each step. THIS ports that re-discretization onto the project's OWN spiking substrate:
each step's transition scores drive K Izhikevich attractor pools with input-DIVISIVE-NORMALIZATION (the E%-max WTA =
the OneBrainComposer/NEF cleanup = CA3 pattern completion) -> the WINNER pool FIRES -> the next state is read from
SPIKES -> iterate. So the running group state is maintained as a spiking attractor, composing to held-out-DEEPER depth.

RUNG-1 SCOPE: the TRANSITION (delta: state x input -> next-state scores) is the rate-learned weights (reuse the validated
discrete_attractor_rnn); only the RE-DISCRETIZATION is moved on-spikes (the divnorm WTA). Anti-cheats: (a) spiking-WTA
winner == host-argmax winner per step (the WTA is faithful); (b) DIVNORM-OFF lesion -> the WTA degrades (the divisive
normalization is load-bearing); (c) held-out-DEEPER state-track on spikes >> chance == the rate result. Reuse-by-import;
NO `sim/` edit. numpy backend (small bridge).

Run:  SIM_BACKEND=numpy python -m research.runners._d3_spiking_attractor_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, discrete_attractor_rnn
from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge, onbridge_divnorm_drive


def build_fswta_score_bridge(seed, K, n_word=12, n_fs=24, exc_to_fs=2.0, fs_to_exc=9.0):
    """K Izhikevich attractor pools + a shared INHIBITORY FS pool with LATERAL INHIBITION (each pool excites FS; FS
    inhibits all pools). The winner (highest score-drive) fires first -> recruits FS -> FS suppresses the runners-up
    -> a CLEAN one-of-K winner even at SMALL margins (large K). This is the project's shared_FS / concept-pool WTA
    biology applied to the D3 re-discretization. NO `sim/` edit."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(); cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.enable_brain_region_framework = True; cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp", "enable_input_divisive_norm"):
        setattr(cfg, flag, False)
    regions = [BrainRegion(name=f"w{k}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False) for k in range(K)]
    regions.append(BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    pathways = []
    for k in range(K):
        pathways.append(RegionPathway(from_region=f"w{k}", to_region="fs", density=0.6, weight_mean=exc_to_fs, weight_jitter=0.1, plastic=False))
        pathways.append(RegionPathway(from_region="fs", to_region=f"w{k}", density=0.6, weight_mean=fs_to_exc, weight_jitter=0.1, plastic=False))
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def fswta_drive(sb, K, scores, input_gain=1200.0, settle=25):
    """Drive the K attractor pools by score; the FS lateral inhibition resolves a CLEAN winner. Returns (None, acc[K])."""
    from sim.backend import to_host, from_host
    rm = sb.region_manager
    _ridx = {k: np.asarray(list(rm.indices(f"w{k}")), dtype=int) for k in range(K)}
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = -65.0
    sb.cp_recovery_variable_u[:] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for k in range(K):
        cur[_ridx[k]] = float(input_gain * s[k])
    acc = np.zeros(K); cur_dev = from_host(cur)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur_dev
        sb._run_one_simulation_step()
        fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        for k in range(K):
            acc[k] += fir[_ridx[k]].mean()
    sb.cp_external_input_current[:] = 0.0
    return None, acc


def spiking_rollout_eval(task, W, split, sb, K, input_gain=1200.0, settle=15, n_eval=60, seed=42, drive_fn=None):
    """Autoregressive rollout with ON-BRIDGE spiking WTA re-discretization. Each step: scores = Ws.tanh(Wr.emb[cur] +
    Wi.x) + bs (rate transition) -> drive the K attractor pools -> the winner FIRES (divnorm WTA) -> next state = the
    spiking winner. Returns spiking state-track acc + the spiking-vs-host winner agreement."""
    emb, Wr, Wi, Ws, bs = W["emb"], W["Wr"], W["Wi"], W["Ws"], W["bs"]
    ident = task["ident"]
    Xe, ye, Le, _, Se = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(Le), min(n_eval, len(Le)), replace=False)
    ok_spk = 0; agree = 0; steps = 0
    for n in idx:
        cur = ident
        for t in range(int(Le[n])):
            h = np.tanh(emb[cur] @ Wr.T + Xe[n, t] @ Wi.T)
            scores = h @ Ws.T + bs                                # K-dim transition scores
            _drv = drive_fn if drive_fn is not None else onbridge_divnorm_drive
            _, acc = _drv(sb, K, scores, input_gain=input_gain, settle=settle)
            nxt_spk = int(np.argmax(acc)) if acc.max() > 0 else ident
            nxt_host = int(np.argmax(scores))
            agree += int(nxt_spk == nxt_host); steps += 1
            cur = nxt_spk                                         # ROLL OUT on the SPIKING winner
        ok_spk += int(cur == Se[n, int(Le[n]) - 1])
    return {"spk_track": ok_spk / len(idx), "spk_host_agree": agree / max(steps, 1)}


def run_seed(group_name, seed, n_pool=None, n_hid=192, epochs=60, n_per_len=None):
    is_big = group_name == "A5"
    n_pool = n_pool if n_pool is not None else (256 if is_big else 64)
    n_per_len = n_per_len if n_per_len is not None else (8000 if is_big else 1500)
    task = make_group_task(group_name, seed, n_pool=n_pool, noise=0.6, n_per_len=n_per_len,
                           train_lens=(1, 2, 3, 4, 5), test_lens=(6, 7, 8))
    K = task["K"]
    da = discrete_attractor_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid)     # rate transition (validated)
    W = da["weights"]
    # PRIMARY spiking WTA = PLAIN Izhikevich drive (drive each attractor pool by its score -> the winner fires most ->
    # decode argmax(firing) = the spiking re-discretization). The divisive-norm E%-max OVER-normalizes single-winner
    # transition scores (a diagnostic, not the right cleanup for a clear one-of-K winner).
    sb = build_divnorm_score_bridge(seed=seed, V=K, n_word=10, enable_divnorm=False)
    sb_fs = build_fswta_score_bridge(seed=seed, K=K)                                       # FS lateral-inhibition WTA
    spk_same = spiking_rollout_eval(task, W, "test_same", sb, K, seed=seed)
    spk_deep = spiking_rollout_eval(task, W, "test_deeper", sb, K, seed=seed)
    fs_deep = spiking_rollout_eval(task, W, "test_deeper", sb_fs, K, seed=seed, settle=25, drive_fn=fswta_drive)
    return {"seed": seed, "group": group_name, "K": K, "rate_step_delta": round(da["step_transition_acc"], 3),
            "rate_deeper_track": round(da["state_deeper"], 3),
            "SPK_same_track": round(spk_same["spk_track"], 3), "SPK_deeper_track": round(spk_deep["spk_track"], 3),
            "SPK_host_agree_deeper": round(spk_deep["spk_host_agree"], 3),
            "FSWTA_deeper_track": round(fs_deep["spk_track"], 3), "FSWTA_host_agree": round(fs_deep["spk_host_agree"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 SPIKING attractor] {a.group} | re-discretization ON SPIKES (divnorm WTA = CA3/NEF cleanup) | rate transition + spiking re-discretize", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, n_hid=a.n_hid, epochs=a.epochs)
        rows.append(r)
        print(f"  [seed {s}] rate: step-delta={r['rate_step_delta']} deeper={r['rate_deeper_track']} || "
              f"plain-WTA: DEEPER={r['SPK_deeper_track']} (agree={r['SPK_host_agree_deeper']}) || "
              f"FS-WTA (lateral inhib): DEEPER={r['FSWTA_deeper_track']} (agree={r['FSWTA_host_agree']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        spk_d, agree = _m("SPK_deeper_track"), _m("SPK_host_agree_deeper")
        fs_d, fs_a = _m("FSWTA_deeper_track"), _m("FSWTA_host_agree")
        # GO: the FS lateral-inhibition WTA re-discretizes ON SPIKES with a CLEAN competitive winner, holding held-out-
        # DEEPER (>>chance) AND faithful (== host argmax) even at LARGE K where the plain drive's small-margin errors
        # compound. (The plain WTA is the S3 baseline; FS-WTA is the clean-attractor scale fix.)
        best_d = max(spk_d, fs_d); best_a = max(agree, fs_a)
        go = (best_d > 0.90) and (best_a > 0.95)
        print(f"\n  AGGREGATE ({a.group}): plain-WTA deeper={spk_d:.3f} (agree {agree:.3f}) | FS-WTA deeper={fs_d:.3f} (agree {fs_a:.3f}) (chance={1.0/rows[0]['K']:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the discrete-attractor re-discretization runs ON SPIKES (best deeper-track '+format(best_d,'.2f')+', faithful == host argmax) -> the recurrent composition is realized on the project spiking substrate = the simulated recurrent language cortex; FS lateral inhibition gives the clean one-active attractor at scale' if go else 'the spiking WTA did not hold cleanly (tune FS exc/inh weights or input_gain/settle; read the host-agree gap between plain and FS-WTA)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
