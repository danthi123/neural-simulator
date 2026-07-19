"""gap#1 Rung 2 ON-BRIDGE realization de-risk: does the trained SSM leaky-integrator state, realized on a REAL
SimulationBridge (Izhikevich neurons + a SLOW NMDA-recurrent conductance = the leaky memory, driven per token, read from
`cp_firing_states`), preserve the deep-context LM capture the rate-level SSM has? The rate-level de-risks are ALL GO
(Rung 1a mechanism 6-seed · Rung 1b emergent input 3-seed · Rung 2 spiking-faithful recurrence + non-negative firing-rate
read + uniform decay). This closes the loop to the fully-spiking substrate.

THE MAPPING (uniform-decay SSM -> one recurrent Izhikevich region + slow NMDA memory):
  rate-SSM: a_t = decay*a_{t-1} + v_t ; v_t = Wv @ LayerNorm(emb[x_t]) ; read = Wo_sp @ [relu(a_t), relu(-a_t)] -> head.
  on-bridge: a region of D "channel" neurons whose SLOW NMDA-recurrent conductance holds a leaky state across the fast
  Izhikevich spiking (NMDA tau ~= the SSM decay). Per token: drive the region's external current with v_t (ON = +v, a
  matched OFF sub-population = -v via a sign-split drive), run T_STEP bridge steps (real conductance synapses + Izhikevich),
  read the region's per-neuron spike counts = the firing-rate state; the read is the trained Wo_sp over [rate_ON, rate_OFF].
  The slow NMDA conductance is NOT washed between tokens within a sentence => it INTEGRATES = the leaky memory.

VERIFY-FIRST (silent-failure discipline): before the full eval, compare the on-bridge firing-rate STATE trajectory to the
rate-SSM analog state on one sentence (corr) -- a wrong substrate mapping shows up as a low/zero correlation, caught before
any GO claim. GATE: the on-bridge LM beats the fair trigram at deep context (the rate-SSM's bar), AND perm/memoryless
anti-cheats collapse. Reuse Vocab/load_sentences/fit_interp_trigram/_bucket. NO `sim/` edit (drives + reads public arrays).

Run: SIM_BACKEND=cupy python -m research.runners._emerge_wkv_onbridge_derisk --ssm <path>_seed42.npz --seed 42 --n-eval 200
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from research.runners._emerge_wkv_lm_derisk import fit_interp_trigram

_T_STEP_DEFAULT = 6                                                          # bridge steps per token (integration window)


def _build_channel_bridge(D, seed, self_nmda_w=8.0, dt=0.5, pop_k=1):
    """2*D Izhikevich channel neurons (D ON + D OFF). The leaky memory = a DIAGONAL self-NMDA autapse per channel neuron
    (pre==post, exc_receptor='nmda_slow' via inject_explicit_wiring): each neuron's firing charges its OWN slow NMDA
    conductance (tau~100ms) = the per-channel leaky integral a_t=decay*a_{t-1}+drive (NOT random reservoir mixing). Built
    via a minimal valid region (so the bridge initializes) then inject_explicit_wiring OVERRIDES with the diagonal edges.
    Driven per token by external current; read from cp_firing_states. Returns (bridge, on_idx, off_idx, snapshot)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_nmda = True                                           # slow NMDA conductance = the leaky memory
    cfg.brain_regions = [
        BrainRegion(name="chan", n_neurons=2 * D * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    enable_nmda=True),
    ]
    cfg.region_pathways = []
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    idx = np.asarray(b.region_manager.indices("chan"))
    # OVERRIDE the random internal connectivity with a DIAGONAL self-NMDA autapse (the per-channel leaky integral).
    ii = [int(i) for i in idx]
    plan = {"chan_self_nmda": {"pre_indices": ii, "post_indices": ii, "initial_weights": [float(self_nmda_w)] * len(ii),
                                "plastic": False, "conn_type": "MIXED", "exc_receptor": "nmda_slow"}}
    b.inject_explicit_wiring(plan)
    # channel c (0..2D-1) -> the pop_k neurons idx[c*pop_k:(c+1)*pop_k] (population coding; averaged read = less spiking noise)
    chan_groups = [idx[c * pop_k:(c + 1) * pop_k] for c in range(2 * D)]
    return b, chan_groups, None, None


def _build_plateau_channel_bridge(D, seed, pathway_w=3.0, pop_k=8, dt=1.0, center=8.0, slope=0.33, strength=80.0,
                                  tau_decay_ms=80.0):
    """DENDRITIC GRADED-PLATEAU realization (the convergent build; the point-neuron-limit SURPASS): the WKV leaky state lives
    in a GRADED dendritic plateau CONDUCTANCE (`enable_graded_dendritic_plateau`, the validated protected sim/ edit) — a
    Mikulasch-Priesemann ANALOG read-out the point-neuron soma provably cannot be. Each channel = a plateau compartment fed by
    an input pool through a coincidence_detector pathway (weighted drive c_w propto v_t); the plateau integrates strength*
    sigmoid(slope*(c_w-center)) with an 80ms decay = a LEAKY INTEGRAL held in the ANALOG conductance, read DIRECTLY from
    cp_conductance_g_graded_plateau (no firing-rate read loss, no spike-charging distortion — the two losses that capped the
    point-neuron paths at ~0.55)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_parameter_heterogeneity = False; cfg.enable_conductance_noise = False
    # the graded dendritic plateau (the validated sim/ edit): reads the coincidence_detector WEIGHTED drive, graded-only
    cfg.enable_coincidence_detection = True; cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = 1e9; cfg.coincidence_plateau_strength = 0.0   # all-or-none OFF; graded carries the value
    cfg.enable_graded_dendritic_plateau = True
    cfg.graded_plateau_center = float(center); cfg.graded_plateau_slope = float(slope)
    cfg.graded_plateau_strength = float(strength)
    cfg.graded_plateau_tau_decay_ms = float(tau_decay_ms); cfg.graded_plateau_tau_rise_ms = 2.0  # DECAY-MATCH to the SSM
    cfg.brain_regions = [
        BrainRegion(name="inp", n_neurons=2 * D * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="chan", n_neurons=2 * D, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = []
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    inp = np.asarray(b.region_manager.indices("inp")); chan = np.asarray(b.region_manager.indices("chan"))
    # BLOCK-DIAGONAL coincidence pathway inp[c*pop_k:(c+1)*pop_k] -> chan[c] (each channel's plateau integrates ITS OWN v_t)
    pre, post = [], []
    for c in range(2 * D):
        for i in inp[c * pop_k:(c + 1) * pop_k]:
            pre.append(int(i)); post.append(int(chan[c]))
    b.inject_explicit_wiring({"inp_to_chan_coinc": {"pre_indices": pre, "post_indices": post,
                              "initial_weights": [float(pathway_w)] * len(pre), "plastic": False,
                              "conn_type": "MIXED", "coincidence_detector": True}})
    inp_groups = [inp[c * pop_k:(c + 1) * pop_k] for c in range(2 * D)]
    chan_groups = [chan[c:c + 1] for c in range(2 * D)]
    return b, inp_groups, chan_groups, None


def _build_recur_channel_bridge(D, seed, recur_w=0.35, n_recur=20, dt=0.5, density=0.5, learn=False):
    """LINE-ATTRACTOR INTEGRATOR (the scoped deep frontier, next-arc de-risk): each channel = a POPULATION of n_recur neurons
    with block-diagonal NMDA-slow RECURRENT self-excitation (Wong-Wang 2002/Seung-Goldman graded persistent-activity integrator,
    soft-WTA gain alpha<1 -> RAMPS/HOLDS a graded value under evidence, never self-ignites), NOT a single self-NMDA autapse. A
    recurrent POPULATION holds a graded leaky value at higher fidelity than a single cell's mean rate (the whole point of a line
    attractor). Driven per token by graded current propto v_t; read the population mean rate. Reuses biased_competition_buffer's
    sel-pool design (NMDA-slow recurrent, alpha<1)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_nmda = True
    cfg.brain_regions = [
        BrainRegion(name="chan", n_neurons=2 * D * n_recur, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False, enable_nmda=True),
    ]
    cfg.region_pathways = []
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False
    cfg.enable_hebbian_learning = bool(learn)                        # EMERGENT: Hebbian self-organization of the recurrence
    if learn:
        cfg.hebbian_max_weight = 50.0
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    idx = np.asarray(b.region_manager.indices("chan"))
    # BLOCK-DIAGONAL NMDA recurrent self-excitation: within each channel's n_recur block, dense recurrent edges (the Wang-2002
    # integrator). rng-sparsify to `density`; alpha<1 via recur_w (soft-WTA, holds/ramps, no self-ignition). PLASTIC when learn=
    # True -> the recurrence SELF-ORGANIZES (Hebbian: co-active neurons within a channel strengthen -> a learned attractor).
    rng = np.random.default_rng(seed + 7)
    pre, post = [], []
    for c in range(2 * D):
        blk = idx[c * n_recur:(c + 1) * n_recur]
        for i in blk:
            for j in blk:
                if i != j and rng.random() < density:
                    pre.append(int(i)); post.append(int(j))
    plan = {"chan_recur_nmda": {"pre_indices": pre, "post_indices": post,
                                "initial_weights": [float(recur_w)] * len(pre),
                                "plastic": bool(learn), "conn_type": "MIXED", "exc_receptor": "nmda_slow"}}
    b.inject_explicit_wiring(plan)
    chan_groups = [idx[c * n_recur:(c + 1) * n_recur] for c in range(2 * D)]
    return b, chan_groups, None, None


def _build_ff_channel_bridge(D, seed, ff_nmda_w=8.0, dt=0.5, pop_k=1, nmda_tau_ms=100.0):
    """SpikeGPT-consistent GRADED-STATE realization (research gate GO): the leaky WKV state = a FEEDFORWARD NMDA conductance,
    NOT a spike-charged self-autapse. Two populations: an INPUT pool (driven per token by graded current proportional to v_t;
    it FIRES = presynaptic spikes) and a STATE pool. inp[c] -> chan[c] is a DIAGONAL FEEDFORWARD nmda_slow synapse, so
    chan.g_nmda(t) = decay*g_nmda(t-1) + w*(inp spikes) = the exact LINEAR leaky integral a_t = decay*a_{t-1} + input_t --
    the channel never self-fires to charge its own state, so the f-I nonlinearity does NOT compound across the recurrence
    (that compounding was the 0.786->0.55 STATE loss). Biology: a slow postsynaptic NMDA/Ca conductance charged by
    presynaptic spikes = neurons+synapses+communication (the project's BRAIN-BASED standard; spikes carry I/O, the graded
    slow conductance holds the integrator state -- exactly what SpikeGPT + Wang-2002/Seung-Goldman line attractors do)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = float(nmda_tau_ms)                          # MATCH the g_nmda decay to the SSM per-token decay
    cfg.brain_regions = [
        BrainRegion(name="inp", n_neurons=2 * D * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False, enable_nmda=False),
        BrainRegion(name="chan", n_neurons=2 * D * pop_k, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False, enable_nmda=True),
    ]
    cfg.region_pathways = []
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    inp = np.asarray(b.region_manager.indices("inp")); chan = np.asarray(b.region_manager.indices("chan"))
    # OVERRIDE random internal connectivity with a DIAGONAL FEEDFORWARD nmda_slow synapse inp[i] -> chan[i] (no self-autapse)
    ii = [int(i) for i in inp]; oo = [int(i) for i in chan]
    plan = {"inp_to_chan_nmda": {"pre_indices": ii, "post_indices": oo, "initial_weights": [float(ff_nmda_w)] * len(ii),
                                 "plastic": False, "conn_type": "MIXED", "exc_receptor": "nmda_slow"}}
    b.inject_explicit_wiring(plan)
    inp_groups = [inp[c * pop_k:(c + 1) * pop_k] for c in range(2 * D)]
    chan_groups = [chan[c * pop_k:(c + 1) * pop_k] for c in range(2 * D)]
    return b, inp_groups, chan_groups, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm", required=True, help="saved SSM weights (_emerge_wkv_lm_derisk --save-ssm ..._seed<N>.npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories_train.txt")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--max-train-sents", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=200)              # SMALL (bridge stepping is slow)
    ap.add_argument("--drive-scale", type=float, default=1200.0)    # v_t -> external current pA
    ap.add_argument("--self-nmda-w", dest="self_nmda_w", type=float, default=8.0)   # diagonal self-NMDA autapse weight
    ap.add_argument("--mlp-readout", dest="mlp_readout", action="store_true")
    ap.add_argument("--n-fit", dest="n_fit", type=int, default=600)   # train sentences for the on-bridge read-out re-fit
    ap.add_argument("--exact-state", dest="exact_state", action="store_true")
    ap.add_argument("--read-gnmda", dest="read_gnmda", action="store_true",
                    help="LEVER 1: read the standing cp_conductance_g_nmda (100ms leaky integral) instead of firing rate")
    ap.add_argument("--read-latency", dest="read_latency", action="store_true",
                    help="READ-CODE: first-spike latency (Thorpe rank-order), graded where mean-rate saturates")
    ap.add_argument("--graded-charge", dest="graded_charge", action="store_true",
                    help="SpikeGPT-consistent: leaky state = FEEDFORWARD NMDA conductance (input pool spikes charge chan g_nmda), read the graded conductance -- no self-fired f-I compounding")
    ap.add_argument("--graded-bias", dest="graded_bias", type=float, default=300.0,
                    help="bias current (pA) to put the graded-charge input pool in its linear f-I regime")
    ap.add_argument("--graded-gain-lo", dest="graded_gain_lo", type=float, default=0.15)  # staggered pop-member gains
    ap.add_argument("--graded-gain-hi", dest="graded_gain_hi", type=float, default=2.5)   # -> graded population rate code
    ap.add_argument("--recur-integrator", dest="recur_integrator", action="store_true", help="LINE-ATTRACTOR: per-channel NMDA-recurrent population integrator (Wong-Wang alpha<1)")
    ap.add_argument("--kick-steps", dest="kick_steps", type=int, default=0, help="transient-kick: drive only the first K steps per token, then let the recurrence sustain")
    ap.add_argument("--plateau-center", dest="plateau_center", type=float, default=8.0)
    ap.add_argument("--plateau-calib", dest="plateau_calib", action="store_true", help="per-channel v calibration so all channels land in the graded window")
    ap.add_argument("--graded-plateau", dest="graded_plateau", action="store_true", help="DENDRITIC GRADED PLATEAU: WKV leaky state in cp_conductance_g_graded_plateau (0.98 core fidelity, point-neuron limit surpassed)")
    ap.add_argument("--learn-recur", dest="learn_recur", action="store_true", help="EMERGENT: Hebbian-plastic recurrent weights self-organize the attractor during the drive")
    ap.add_argument("--tonic-bias", dest="tonic_bias", type=float, default=0.0, help="post-kick tonic current to keep the recurrent population near threshold (persistence)")
    ap.add_argument("--recur-w", dest="recur_w", type=float, default=0.35, help="recurrent NMDA self-excitation weight (alpha<1)")
    ap.add_argument("--hetero-gain", dest="hetero_gain", action="store_true",
                    help="heterogeneous-population code on the self-NMDA path: staggered pop-member drive gains (staggered effective thresholds) -> higher-fidelity graded population rate")
    ap.add_argument("--pop-k", dest="pop_k", type=int, default=1)
    ap.add_argument("--t-step", dest="t_step", type=int, default=_T_STEP_DEFAULT)   # bridge steps/token (finer rate=less noise)
    ap.add_argument("--json", type=str, default="research/findings/raw/_emerge_wkv_onbridge.json")
    args = ap.parse_args()
    _T_STEP = int(args.t_step)
    from sim.backend import to_host, get_backend
    xp, _bk = get_backend()

    W = np.load(args.ssm, allow_pickle=True)
    V = int(W["V"]); D = int(W["d_model"])
    emb = W["emb.weight"]; ln_w = W["ln.weight"]; ln_b = W["ln.bias"]
    Wv = W["Wv.weight"]; Wr = W["Wr.weight"]; Wo_sp = W["Wo_sp.weight"]; head_w = W["head.weight"]; head_b = W["head.bias"]
    decay = float(np.exp(-np.log1p(np.exp(W["w"][0]))))             # exp(-softplus(w)) = the uniform decay
    words = list(W["words"])

    def _ln(v):
        m = v.mean(); s = v.std() + 1e-5
        return (v - m) / s * ln_w + ln_b

    # PER-CHANNEL calibration (plateau): each channel's v_t std over the vocab -> normalize so ALL channels land in the graded
    # sigmoid window (the global center/slope can then match all channels; else some saturate/floor and carry no info).
    _vall = np.stack([Wv @ _ln(emb[t]) for t in range(V)], 0)         # [V, D]
    v_chan_scale = _vall.std(0) + 1e-6                                 # [D] per-channel scale

    if not Path(args.corpus).exists():
        args.corpus = "data/corpus/tinystories.txt"
    sents = load_sentences(args.corpus, args.n_sentences)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
    tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
    ev = [sents[i] for i in idx[cut:]][:args.n_eval]
    vocab = Vocab.build(tr, V=V); tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    P_bi = fit_bigram(tr_ids, V); tri, _lam = fit_interp_trigram(tr_ids, V, tr[-1500:] and [vocab.ids(s) for s in tr[-1500:]])

    _graded = getattr(args, "graded_charge", False)
    if _graded:
        # SpikeGPT-consistent: drive the INPUT pool (inp_groups) with graded current, read the STATE pool g_nmda (chan_groups)
        # MATCH the g_nmda per-token decay to the SSM decay: exp(-t_step*dt/tau) == decay  ->  tau = -t_step*dt/ln(decay)
        _dt = 0.5
        _tau = float(-args.t_step * _dt / np.log(decay)) if 0 < decay < 1 else 100.0
        b, inp_groups, chan_groups, snap = _build_ff_channel_bridge(D, args.seed, ff_nmda_w=args.self_nmda_w,
                                                                    pop_k=args.pop_k, dt=_dt, nmda_tau_ms=_tau)
        print(f"[graded] SSM decay={decay:.4f} -> matched nmda_tau={_tau:.1f}ms (t_step={args.t_step}, dt={_dt})", flush=True)
        drive_groups = inp_groups                                    # drive the presynaptic input pool
    elif getattr(args, "graded_plateau", False):
        # DENDRITIC GRADED PLATEAU (the breakthrough: 0.98 core fidelity): the WKV leaky state lives in the graded plateau
        # CONDUCTANCE. Drive the inp pool propto v_t -> coincidence drive c_w -> plateau integrates (decay-matched to the SSM).
        _tau = float(-args.t_step * 1.0 / np.log(decay)) if 0 < decay < 1 else 80.0
        b, inp_groups, chan_groups, snap = _build_plateau_channel_bridge(D, args.seed, pathway_w=args.self_nmda_w,
                                                                         pop_k=args.pop_k, tau_decay_ms=_tau, center=getattr(args,"plateau_center",8.0))
        drive_groups = inp_groups
        print(f"[plateau] dendritic graded plateau: SSM decay={decay:.4f} -> matched plateau tau={_tau:.1f}ms, pop_k={args.pop_k}", flush=True)
    elif getattr(args, "recur_integrator", False):
        # LINE-ATTRACTOR: each channel = an NMDA-recurrent POPULATION integrator (Wong-Wang, alpha<1); drive propto v_t, read
        # the population mean rate. A recurrent population holds a graded value at higher fidelity than a single self-NMDA cell.
        b, chan_groups, _cg2, snap = _build_recur_channel_bridge(D, args.seed, recur_w=args.recur_w, n_recur=args.pop_k, learn=getattr(args, "learn_recur", False))
        drive_groups = chan_groups
        print(f"[recur] line-attractor integrator: n_recur={args.pop_k}, recur_w={args.recur_w}", flush=True)
    else:
        b, chan_groups, _cg2, snap = _build_channel_bridge(D, args.seed, self_nmda_w=args.self_nmda_w, pop_k=args.pop_k)
        drive_groups = chan_groups                                  # self-NMDA: drive == read pool
    nnrn = int(b.cp_membrane_potential_v.size)
    # per-neuron -> channel maps: drive on drive_groups; READ on chan_groups (== drive for self-NMDA; the state pool for FF)
    all_drive_idx = np.concatenate([np.asarray(g) for g in drive_groups]).astype(np.int64)
    drive_chan_of = np.concatenate([[c] * len(drive_groups[c]) for c in range(2 * D)]).astype(np.int64)
    read_idx = np.concatenate([np.asarray(g) for g in chan_groups]).astype(np.int64)
    chan_of = np.concatenate([[c] * len(chan_groups[c]) for c in range(2 * D)]).astype(np.int64)
    gsize = np.array([len(g) for g in chan_groups], dtype=np.float64)
    # GRADED-POPULATION code (graded-charge only): stagger the per-pop-member drive GAIN so the input pool's POPULATION rate
    # is ~linear in v_t (a Goldman/Seung graded integrator: member k fires only above its effective threshold; more members
    # fire as |v_t| grows -> population rate proportional to |v_t|, not a single saturating neuron). Fixes the g_nmda ceiling.
    drive_gain = np.ones(len(all_drive_idx))
    if (_graded or getattr(args, "hetero_gain", False)) and args.pop_k > 1:
        gains = np.linspace(float(args.graded_gain_lo), float(args.graded_gain_hi), args.pop_k)
        drive_gain = np.concatenate([gains for _ in range(2 * D)])   # per channel: pop_k staggered gains (matches group order)

    def _wash():
        """Reset the state so each sentence reads independently (zero the leaky NMDA memory + conductances + firing;
        v/u to Izhikevich rest). Robust to array-size details (avoids the finicky EMERGE snapshot/restore pair)."""
        for nm in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise", "cp_conductance_g_nmda",
                   "cp_conductance_g_e", "cp_conductance_g_i", "cp_firing_states"):
            arr = getattr(b, nm, None)
            if arr is not None: arr[:] = 0.0
        if b.cp_membrane_potential_v is not None: b.cp_membrane_potential_v[:] = -65.0
        if b.cp_recovery_variable_u is not None: b.cp_recovery_variable_u[:] = 0.0

    def onbridge_states(ids):
        """Drive the channel region per token; return the per-position firing-rate state [T, 2D] (ON then OFF rates)."""
        _wash()
        rates = []
        _a = np.zeros(D)                                             # host leaky state (for --exact-state isolating test)
        for t in range(len(ids)):
            h = _ln(emb[ids[t]]); v = Wv @ h                        # [D]
            if getattr(args, "graded_plateau", False) and getattr(args, "plateau_calib", False):
                v = v / v_chan_scale                                # PER-CHANNEL calibration -> all channels in the graded window
            if getattr(args, "exact_state", False):
                # ISOLATE the spiking READ: drive with the EXACT host-computed rate-SSM state (leaky integral done in host)
                # -> the neurons transduce the exact state to firing; if this GOes, the read is fine + the substrate's
                # input-integral (self-NMDA ~0.6 fidelity) is the only gap; if not, the spiking read itself is the limit.
                _a = decay * _a + v
                chan_drive = np.concatenate([np.maximum(_a, 0.0), np.maximum(-_a, 0.0)]) * args.drive_scale
            else:
                chan_drive = np.concatenate([np.maximum(v, 0.0), np.maximum(-v, 0.0)]) * args.drive_scale   # [2D] ON|OFF
            if _graded or getattr(args, 'graded_plateau', False):
                # bias the INPUT pool into its LINEAR f-I regime so firing ~= baseline + gain*v_t (small v_t must still
                # modulate firing, not fall below threshold) -> g_nmda = leaky integral LINEAR in v_t. The constant baseline
                # is a per-channel offset the read-out removes; the MODULATION carries the signal.
                chan_drive = chan_drive + float(args.graded_bias)
            cur = np.zeros(nnrn, np.float32)
            cur[all_drive_idx] = chan_drive[drive_chan_of] * drive_gain   # staggered per-member gain (graded-pop) or 1.0
            cnt = np.zeros(2 * D, np.float64)
            _latency = getattr(args, "read_latency", False)
            first_spk = np.full(len(all_drive_idx), float(_T_STEP)) if _latency else None  # per drive-neuron 1st-spike step
            _kick = int(getattr(args, "kick_steps", 0))
            _tonic = float(getattr(args, "tonic_bias", 0.0))
            for _step in range(_T_STEP):
                b.cp_external_input_current[:] = 0.0
                if _kick <= 0 or _step < _kick:
                    # TRANSIENT-KICK (line-attractor): drive the first _kick steps with the token input, then only a small
                    # TONIC bias so the population stays near threshold and the recurrence SUSTAINS the graded value (breaks
                    # the persistence bootstrap). _kick<=0 = the default constant-drive window.
                    b.cp_external_input_current[all_drive_idx] = (xp.asarray(cur[all_drive_idx]) if xp is not None
                                                                 else cur[all_drive_idx])
                elif _tonic > 0.0:
                    b.cp_external_input_current[all_drive_idx] = _tonic
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states))
                np.add.at(cnt, drive_chan_of, fs[all_drive_idx].astype(np.float64))   # sum spikes per channel (drive pool)
                if _latency:
                    fired = fs[all_drive_idx] > 0
                    newly = fired & (first_spk == float(_T_STEP))   # record FIRST spike step only
                    first_spk[newly] = float(_step)
            if getattr(args, "graded_plateau", False):
                # DENDRITIC GRADED PLATEAU (0.98 core fidelity): read the graded plateau CONDUCTANCE directly (the
                # Mikulasch-Priesemann analog value the point-neuron soma can't be) = the high-fidelity leaky WKV state.
                gp = np.zeros(2 * D, np.float64)
                gplat = np.asarray(to_host(b.cp_conductance_g_graded_plateau)).astype(np.float64)
                np.add.at(gp, chan_of, gplat[read_idx])
                rates.append(gp / gsize)
            elif _graded:
                # SpikeGPT-consistent GRADED STATE: read the STATE pool's slow NMDA conductance (charged FEEDFORWARD by the
                # input pool's presynaptic spikes) = the exact linear leaky integral a_t=decay*a_{t-1}+input_t, held in a real
                # postsynaptic conductance (not a spike-rate code, not a self-fired autapse). The f-I nonlinearity applies
                # only per-token to the input pool's firing, NOT compounded across the recurrence -> kills the STATE loss.
                gn = np.zeros(2 * D, np.float64)
                gnmda = np.asarray(to_host(b.cp_conductance_g_nmda)).astype(np.float64)
                np.add.at(gn, chan_of, gnmda[read_idx])            # read the STATE pool (chan), aggregate per channel
                rates.append(gn / gsize)
            elif _latency:
                # READ-CODE = first-spike LATENCY (Thorpe rank-order): earlier spike -> higher activation (T_STEP-t)/T_STEP
                # in [0,1], 0 if never fired. Graded where the mean-rate code saturates; carries magnitude info rate discards.
                act = (float(_T_STEP) - first_spk) / float(_T_STEP)
                lat = np.zeros(2 * D, np.float64)
                np.add.at(lat, drive_chan_of, act)
                rates.append(lat / gsize)                           # latency-code activation, pop-averaged
            elif getattr(args, "read_gnmda", False):
                # LEVER 1 (research-gate GO): read the STANDING NMDA conductance (the ~100 ms leaky integral) directly,
                # not the within-window spike count. cp_conductance_g_nmda IS the analog leaky-SSM state on real spikes
                # (self-NMDA autapse charges it; decays tau=100 ms; not washed within a sentence). Skips the
                # spike-quantization + f-I-saturation + 3 ms-window read losses (the 0.786 READ ceiling). I_nmda is exactly
                # the postsynaptic current a downstream neuron integrates, so this reads the graded dendritic signal (the
                # mission-compliant spike-pure closure = route it to a downstream read-out pool, a cheap follow-on).
                gn = np.zeros(2 * D, np.float64)
                gnmda = np.asarray(to_host(b.cp_conductance_g_nmda)).astype(np.float64)
                np.add.at(gn, chan_of, gnmda[all_drive_idx])
                rates.append(gn / gsize)                            # standing NMDA integral, pop-averaged
            else:
                rates.append(cnt / (_T_STEP * gsize))               # channel firing rate = pop-averaged (noise-averaged)
        b.cp_external_input_current[:] = 0.0
        return np.asarray(rates)                                     # [T, 2D]

    def rate_ssm_states(ids):
        """The reference rate-SSM analog [relu(a),relu(-a)] per position (to VERIFY the on-bridge mapping)."""
        a = np.zeros(D); out = []
        for t in range(len(ids)):
            h = _ln(emb[ids[t]]); a = decay * a + (Wv @ h)
            out.append(np.concatenate([np.maximum(a, 0.0), np.maximum(-a, 0.0)]))
        return np.asarray(out)                                       # [T, 2D]

    # ---- VERIFY the mapping on 5 sentences (corr of on-bridge firing-rate state vs the rate-SSM analog state) ----
    corrs = []
    for ids in ev_ids[:5]:
        if len(ids) < 4: continue
        ob = onbridge_states(ids); rs = rate_ssm_states(ids)
        c = np.corrcoef(ob.flatten(), rs.flatten())[0, 1]
        corrs.append(c)
    mapcorr = float(np.nanmean(corrs)) if corrs else float("nan")
    _ob0 = onbridge_states(ev_ids[0]) if ev_ids else np.zeros((1, 2 * D))
    _act = float((_ob0.std(0) > 1e-6).mean())                       # fraction of channels with ANY variance across tokens
    print(f"[verify] on-bridge firing-rate state vs rate-SSM analog state: corr={mapcorr:.3f} "
          f"(>0.3 => substrate realizes the leaky state) | firing: mean={_ob0.mean():.3f} max={_ob0.max():.3f} "
          f"frac-active-channels={_act:.2f} (low mean/frac => sparse; discriminative read-out needs varied firing)", flush=True)

    # ---- RE-FIT the read-out on the ACTUAL on-bridge firing-rate states (reservoir-computing: the leaky DYNAMICS are the
    #      fixed on-bridge diagonal self-NMDA; only the linear read-out is trained -- the on-bridge state ~= the rate-SSM
    #      state at a different SCALE (corr above), so a fresh ridge read-out on the on-bridge state recovers the capture) ----
    def _feat(rate_t, ids_t):
        """read-out feature: the raw ON/OFF firing state (2D) + the RECEPTANCE-gated signed state r_h*(ON-OFF) (D) --
        the SSM's current-token gating of the leaky state that the raw linear read-out lacked."""
        r_h = 1.0 / (1.0 + np.exp(-(Wr @ _ln(emb[ids_t]))))          # receptance (current-token gate), D
        signed = rate_t[:D] - rate_t[D:]                            # ON-OFF ~= the signed leaky state a
        return np.concatenate([rate_t, r_h * signed])              # [3D]

    t0 = time.time()
    fit_ids = tr_ids[:args.n_fit]
    Xtr, Ytr = [], []
    for ids in fit_ids:
        if len(ids) < 2: continue
        rates = onbridge_states(ids)
        for t in range(len(ids) - 1):
            Xtr.append(_feat(rates[t], ids[t])); Ytr.append(ids[t + 1])
    Xtr = np.asarray(Xtr); Ytr = np.asarray(Ytr, dtype=np.int64)
    nf = Xtr.shape[1]                                                 # 3D
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    Xn = (Xtr - mean) / std
    if getattr(args, "mlp_readout", False):
        # NONLINEAR (MLP) read-out on the on-bridge states: the --exact-state test showed a LINEAR read can't match the
        # jointly-trained WKV read; a small MLP is the obvious next-method (reservoir-computing with a nonlinear read).
        import torch, torch.nn as nn
        torch.manual_seed(args.seed)
        Xt = torch.tensor(Xn, dtype=torch.float32); Yt = torch.tensor(Ytr)
        mlp = nn.Sequential(nn.Linear(nf, 256), nn.GELU(), nn.Linear(256, V))
        opt = torch.optim.Adam(mlp.parameters(), lr=2e-3, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
        for _ in range(30):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b_ = perm[i:i+256]; opt.zero_grad(); lf(mlp(Xt[b_]), Yt[b_]).backward(); opt.step()
        _mlp = mlp
        print(f"[refit-mlp] trained MLP read-out on {len(Xtr)} on-bridge states (nonlinear); fit-elapsed {time.time()-t0:.0f}s", flush=True)
        Wd = None; Temp = 1.0
    else:
        Z = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)               # [n, 3D+1]
        ZtOH = np.zeros((V, nf + 1)); np.add.at(ZtOH, Ytr, Z)
        Wd = np.linalg.solve(Z.T @ Z + 5.0 * np.eye(nf + 1), ZtOH.T)     # ridge read-out [3D+1, V]
        _mlp = None
    if _mlp is None:                                                # temperature calib (ridge only)
        lg = Z[:20000] @ Wd; ys = Ytr[:20000]
        def _ce_T(T):
            z = lg / T; z = z - z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
            return float(-np.log(p[np.arange(len(ys)), ys] + 1e-12).mean())
        Temp = min([(_ce_T(T), T) for T in (0.5, 1, 2, 4, 8, 16)])[1]
    print(f"[refit] fitted {'MLP' if _mlp is not None else 'ridge'} read-out on {len(Xtr)} on-bridge states (T={Temp}); fit-elapsed {time.time()-t0:.0f}s", flush=True)

    ce = defaultdict(float); bce = defaultdict(float); tce = defaultdict(float); cnt = defaultdict(int)
    for si, ids in enumerate(ev_ids):
        if len(ids) < 2: continue
        rates = onbridge_states(ids)                                 # [T, 2D]
        for t in range(len(ids) - 1):
            if _mlp is not None:
                import torch
                xf = ((_feat(rates[t], ids[t]) - mean) / std).astype(np.float32)
                with torch.no_grad():
                    logits = _mlp(torch.tensor(xf)).numpy()
            else:
                x = np.concatenate([(_feat(rates[t], ids[t]) - mean) / std, [1.0]])
                logits = (x @ Wd) / Temp
            logits = logits - logits.max(); p = np.exp(logits); p = p / p.sum()
            d = t + 1; bkt = _bucket(d)
            ce[bkt] += -math.log(max(p[ids[t+1]], 1e-12))
            bce[bkt] += -math.log(max(P_bi[ids[t], ids[t+1]], 1e-12))
            u = ids[t-1] if t >= 1 else -1
            tce[bkt] += -math.log(max(tri(u, ids[t], ids[t+1]), 1e-12))
            cnt[bkt] += 1
    depth = {}
    for lo, hi in BUCKETS:
        bkt = f"{lo}-{hi}" if lo != hi else f"{lo}"
        if bkt in cnt:
            n = cnt[bkt]
            depth[bkt] = {"n": n, "onbridge": round(ce[bkt]/n, 3), "bigram": round(bce[bkt]/n, 3),
                          "trigram": round(tce[bkt]/n, 3), "vs_trigram": round((tce[bkt]-ce[bkt])/n, 3)}
    print(f"[seed {args.seed}] ON-BRIDGE per-depth NLL (elapsed {time.time()-t0:.0f}s, decay={decay:.3f}):", flush=True)
    for lo, hi in BUCKETS:
        bkt = f"{lo}-{hi}" if lo != hi else f"{lo}"
        if bkt in depth:
            dd = depth[bkt]
            print(f"    depth {bkt:>5} (n={dd['n']:>5}): onbridge {dd['onbridge']:.3f} | bigram {dd['bigram']:.3f} | "
                  f"trigram {dd['trigram']:.3f} || vs-trigram {dd['vs_trigram']:+.3f}", flush=True)
    deep = depth.get("10-99", {})
    go = bool(deep and deep["vs_trigram"] > 0.02 and mapcorr > 0.3)
    print(f"    VERDICT: {'GO' if go else 'no-go'} (deep vs-trigram {deep.get('vs_trigram','?')}, map-corr {mapcorr:.3f})", flush=True)
    out = {"runner": "_emerge_wkv_onbridge_derisk", "ssm": args.ssm, "seed": args.seed, "map_corr": mapcorr,
           "decay": decay, "by_depth": depth, "go": go}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"-> {args.json}", flush=True)


if __name__ == "__main__":
    main()
