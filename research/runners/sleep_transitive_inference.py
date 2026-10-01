"""Sleep transitive inference (plan 2026-09-25 section 5; prereg
research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md).

A->B is learned in context X, B->D in context Y; A and D never co-occur awake. After sleep, A is presented alone in a
novel context Z: does the brain reach D? The substrate is the spiking CA3 replay store the project already has (the
Ecker et al. 2022 AdEx CA3 pyramidal preset, the SWR-initiation protocol and the engine's own STDP, as in
`_gap5_ecker_adex_ca3_stdp_band_derisk` and `_gap5_ecker_replay_learn_through_use_derisk`), generalized from a
six-assembly band to a store in which every ordered pair of assemblies is anatomically connected (weak, symmetric,
plastic), so that a link that was never taught can be written by later activity.

What the brain does (spiking, engine code): the episodes are written into the recurrent weights by the engine's STDP
during the waking sweeps; during sleep the store replays (each SWR event is started by a non-specific cue on a
randomly chosen assembly, the rest of the event is the network's own propagation through what it learned) with STDP
on; the probe presents A (and Z) and the network's own dynamics decide which assemblies ignite.

What the host does (declared): the world (which assemblies are driven when, during the waking episodes and the probe);
the body clock (awake / asleep, the length of the night); the SWR initiation cue (a non-specific current on a
randomly chosen assembly every `swr_period` steps: the same declared shortcut as the Ecker GO runner, whose no-seed
control is silent); the instruments (spike reads, event detection, weight reads) and the experimenter's
interventions (the probe-time silencing of the hub, the replay lesion, the surrogate rasters of the scrambled arm).
No host code relates A to D: nothing in this file looks up, chains or scores a relation; the relational reads are
spike counts of the assembly the network drove.

Arms (per seed; each phase runs on a fresh build with the previous phase's weights loaded, so no transient state
crosses the hours between phases):
  wake     encode -> probe (no sleep)                                     [wake only]
  sleep    encode -> night: SWR replay, STDP on -> probe                  [normal replay]
  off      encode -> night: STDP on, OU on, clock, no SWR initiation      [replay switched off]
  yok      encode -> the sleep arm's own night raster applied through an exact replica of the engine's STDP rule
           (instrument: must reproduce the sleep arm's weights, else scr is UNDEFINED)
  scr      encode -> the sleep arm's night raster with each assembly's activity circularly shifted by a random
           whole number of SWR periods (every cell's spike count and every assembly's event structure kept, the
           order across assemblies destroyed), through the same replica                 [replay order scrambled]
  nohub    a second encode in which the Y episode is C->D instead of B->D (no shared item; same number of episodes,
           slots, laps and drives) -> sleep -> probe                                   [relational chain changed]
  nostdp   encode -> night with SWR replay and STDP OFF                    [plasticity-during-replay lesion]
  nocomp   encode -> night with SWR replay, STDP on, and the within-assembly (completion) weights at zero for the
           night only (restored for the probe)                              [pattern-completion lesion]
  noadapt  encode -> night with SWR replay, STDP on, AdEx adaptation a = b = 0 for the night only [adaptation lesion]

Every constant is in DEFAULTS; the prereg fixes the gate-seed values in its operating-point addendum.

Dev:   SIM_BACKEND=numpy .venv/bin/python -m research.runners.sleep_transitive_inference --dev --seed 7
Gate:  SIM_BACKEND=numpy .venv/bin/python -m research.runners.sleep_transitive_inference --seed 42 \
           --out research/findings/raw/_sleep_transitive_inference/seed42.json
Selftest (grader fixtures, no brain):  ... --selftest
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
OUT_DIR = _REPO / "research" / "findings" / "raw" / "_sleep_transitive_inference"

# ---------------------------------------------------------------------------------------------------------------------
# The store's assemblies. Identical in every arm (the nohub arm uses C1/C2 where the hub arms leave them unused), so
# the build, the anatomy and the RNG streams are the same in every arm of a seed.
#   A_k, B_k, D_k : the chain items of chain k (k = 1, 2; the two chains are each other's matched foil)
#   P_k           : a minor detail told in chain k's X episode (a weak slot on a third of the laps)
#   C_k           : the replacement for B_k in the nohub arm's Y episode (never encoded in the hub arms)
#   X_k, Y_k      : the contexts of chain k's two episodes
#   Z             : the novel context of the test (never encoded)
# ---------------------------------------------------------------------------------------------------------------------
NAMES = ["A1", "B1", "D1", "P1", "C1", "X1", "Y1",
         "A2", "B2", "D2", "P2", "C2", "X2", "Y2",
         "Z"]
IDX = {n: i for i, n in enumerate(NAMES)}
OTHER = {1: 2, 2: 1}

DEFAULTS = {

    # store (Ecker 2022 AdEx CA3 PC preset; sizes as the Ecker GO runner)
    "asm_size": 80, "w_within": 60.0, "within_density": 0.5, "between_init": 15.0, "between_density": 0.25, "extra_cells": 20,
    "b_override": 120.0, "ou_sigma": 100.0, "dt": 0.1,
    # STDP (the engine's Bi-Poo soft-bound rule; the Ecker band runner's values)
    "stdp_w_max": 900.0, "stdp_a_plus": 0.05, "stdp_a_minus": 0.06, "stdp_tau": 20.0,
    # waking episodes (moving-cue sweep over the episode's slots, as the Ecker encode)
    "n_laps": 30, "detail_every": 3, "enc_step": 80, "enc_dwell": 40, "enc_settle": 300, "enc_cue_pa": 9000.0, "enc_cue_frac": 0.6,
    "detail_drive": 0.5,
    # night
    "night_steps": 240000, "swr_period": 1500, "cue_pa": 9000.0, "cue_steps": 40, "cue_frac": 0.6,
    # probe
    "probe_trials": 8, "probe_steps": 400, "probe_settle": 1000, "probe_cue_pa": 9000.0, "probe_cue_frac": 0.6,
    "ctx_cue_pa": 9000.0, "silence_pa": 20000.0,
    # reads
    "smooth_w": 50, "active_frac": 0.10, "particip_frac": 0.25,
    # feedback inhibition (the Ecker 2022 CA3 PV basket pool; 0 = none): every PC excites every PV cell, every PV cell
    # inhibits every PC. Without it the plastic night spreads to every assembly (dev seed 7, section 2 of the prereg).
    "n_pvbc": 40, "w_pc_pvbc": 150.0, "w_pvbc_pc": 40.0,
    # the test context (never encoded, never an SWR initiation site)
    "probe_context": "Z",
    # the replay lesion (disruption): -dis_pa on every PC from dis_start to dis_start + dis_len steps after each
    # initiation (the initiating cue ends at cue_steps = 40)
    "dis_start": 42, "dis_len": 150, "dis_pa": 4000.0,
}


def _chain_episodes(design: str):
    """The waking episodes, in presentation order. Phase X (every X episode, all laps) then phase Y. Each slot is
    (assembly, drive scale, detail_only). A and D never share an episode; a settle gap with the STDP spike clocks
    cleared separates episodes (standing in for the minutes between them), so no cross-episode pairing forms."""
    eps = []
    for k in (1, 2):
        eps.append(("X", k, [(f"X{k}", 1.0, False), (f"A{k}", 1.0, False), (f"B{k}", 1.0, False),
                             (f"P{k}", None, True)]))
    for k in (1, 2):
        # hub: Y_k->B_k->D_k. nohub: Y_k->C_k->D_k (no shared item). cross: Y_k->B_other->D_k, so every item keeps its
        # marginal counts (each B in two episodes, each D in one) and only the chain changes: A_k now leads to D_other.
        mid = {"hub": f"B{k}", "nohub": f"C{k}", "cross": f"B{OTHER[k]}"}[design]
        eps.append(("Y", k, [(f"Y{k}", 1.0, False), (mid, 1.0, False), (f"D{k}", 1.0, False)]))
    return eps


# ---------------------------------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------------------------------
def build_store(seed: int, c: dict, *, a_override=None, b_override="default"):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    from sim.backend import to_host, get_backend
    cp, _ = get_backend()
    m = len(NAMES); S = int(c["asm_size"])
    n_pc = m * S + int(c["extra_cells"])
    regions = [BrainRegion(name="pc", n_neurons=n_pc, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    n_pv = int(c.get("n_pvbc", 0))
    if n_pv > 0:
        regions.append(BrainRegion(name="pvbc", n_neurons=n_pv, exc_fraction=0.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)      # cfg.seed, never actual_seed_used
    cfg.dt_ms = float(c["dt"]); cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    for f in ("enable_homeostasis", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity", "enable_inhibitory_stdp", "enable_reward_modulation"):
        setattr(cfg, f, False)
    cfg.enable_stdp = True           # allocates cp_last_spike_time; switched off right after init
    cfg.stdp_a_plus = float(c["stdp_a_plus"]); cfg.stdp_a_minus = float(c["stdp_a_minus"])
    cfg.stdp_tau_plus_ms = float(c["stdp_tau"]); cfg.stdp_tau_minus_ms = float(c["stdp_tau"])
    cfg.stdp_w_min = 0.0; cfg.stdp_w_max = float(c["stdp_w_max"])
    # OU background current: the engine reads `ou_std_current_pA` (sim/config.py). The Ecker runners set a field the
    # engine never reads (`ou_noise_sigma_pa`), so their "40 pA" runs ran at the 100 pA default; this runner sets the
    # real field, at the 100 pA those validated runs actually used.
    cfg.enable_ou_process = float(c["ou_sigma"]) > 0; cfg.ou_std_current_pA = float(c["ou_sigma"])
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.enable_stdp = False
    bo = c["b_override"] if b_override == "default" else b_override
    if bo is not None:
        b.core_config.adex_b = float(bo)
    if a_override is not None:
        b.core_config.adex_a = float(a_override)
    pc = np.asarray(b.region_manager.indices("pc"), int)
    asm_local = [np.arange(i * S, (i + 1) * S) for i in range(m)]

    rng = np.random.default_rng(int(seed) * 17 + 3)
    wp, wq, ww, bp, bq, bw = [], [], [], [], [], []
    wd = float(c["within_density"]); bd = float(c["between_density"])
    for i in range(m):
        for s in asm_local[i]:
            d = asm_local[i][asm_local[i] != s]
            d = d[rng.random(len(d)) < wd]
            wp.append(np.full(len(d), pc[s])); wq.append(pc[d])
        for j in range(m):
            if j == i:
                continue
            for s in asm_local[i]:
                d = asm_local[j][rng.random(S) < bd]
                bp.append(np.full(len(d), pc[s])); bq.append(pc[d])
    wp = np.concatenate(wp); wq = np.concatenate(wq); bp = np.concatenate(bp); bq = np.concatenate(bq)
    wiring = {
        "within": {"pre_indices": wp.astype(int).tolist(), "post_indices": wq.astype(int).tolist(),
                   "initial_weights": [float(c["w_within"])] * len(wp), "plastic": False, "conn_type": "ff"},
        "between": {"pre_indices": bp.astype(int).tolist(), "post_indices": bq.astype(int).tolist(),
                    "initial_weights": [float(c["between_init"])] * len(bp), "plastic": True, "conn_type": "ff"},
    }
    if n_pv > 0:
        pv = np.asarray(b.region_manager.indices("pvbc"), int)
        allpc = pc[: m * S]
        ip, iq, iw = [], [], []
        for k in range(n_pv):
            ip.append(allpc); iq.append(np.full(len(allpc), pv[k])); iw.append(np.full(len(allpc), float(c["w_pc_pvbc"])))
            ip.append(np.full(len(allpc), pv[k])); iq.append(allpc); iw.append(np.full(len(allpc), float(c["w_pvbc_pc"])))
        wiring["pv"] = {"pre_indices": np.concatenate(ip).astype(int).tolist(),
                        "post_indices": np.concatenate(iq).astype(int).tolist(),
                        "initial_weights": np.concatenate(iw).astype(float).tolist(), "plastic": False, "conn_type": "ff"}
    b.inject_explicit_wiring(wiring)

    coo = b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row), int); col = np.asarray(to_host(coo.col), int)
    loc = np.full(b.core_config.num_neurons, -1, int); loc[pc] = np.arange(n_pc)
    asm_of = np.full(n_pc, -1, int)
    for i, al in enumerate(asm_local):
        asm_of[al] = i
    pre_l = loc[row]; post_l = loc[col]
    ok = (pre_l >= 0) & (post_l >= 0)
    a_pre = np.where(ok, asm_of[np.clip(pre_l, 0, None)], -1); a_post = np.where(ok, asm_of[np.clip(post_l, 0, None)], -1)
    between_pos = np.nonzero((a_pre >= 0) & (a_post >= 0) & (a_pre != a_post))[0]
    within_pos = np.nonzero((a_pre >= 0) & (a_post >= 0) & (a_pre == a_post))[0]
    data0 = np.asarray(to_host(b.cp_connections.data))
    assert np.allclose(data0[between_pos], float(c["between_init"])), "between init misaligned (COO/.data order)"
    assert np.allclose(data0[within_pos], float(c["w_within"])), "within init misaligned (COO/.data order)"
    pm = np.asarray(to_host(b.cp_synapse_plastic_mask)).astype(bool)
    assert pm[between_pos].all() and not pm[within_pos].any(), "plastic mask wrong"
    return dict(bridge=b, cp=cp, pc=pc, n_pc=n_pc, asm_local=asm_local, m=m, S=S,
                between_pos=between_pos, within_pos=within_pos,
                pair_key=(a_pre[between_pos] * m + a_post[between_pos]),
                syn_pre_l=pre_l[between_pos], syn_post_l=post_l[between_pos])


def get_w(store):
    from sim.backend import to_host
    return np.asarray(to_host(store["bridge"].cp_connections.data)).astype(np.float32).copy()


def load_w(store, w):
    cp = store["cp"]
    store["bridge"].cp_connections.data[:] = cp.asarray(np.asarray(w, dtype=np.float32))
    mk = getattr(store["bridge"], "mark_weights_edited", None)
    if mk is not None:
        mk()


def pair_matrix(store, w):
    """Mean between-assembly weight per ordered pair (pre assembly i -> post assembly j): the representational read."""
    m = store["m"]
    vals = np.asarray(w)[store["between_pos"]].astype(np.float64)
    s = np.bincount(store["pair_key"], weights=vals, minlength=m * m)
    n = np.bincount(store["pair_key"], minlength=m * m)
    return (s / np.maximum(n, 1)).reshape(m, m)


def _cue_cells(store, name, frac, rng):
    a = store["asm_local"][IDX[name]]
    k = max(1, int(round(float(frac) * len(a))))
    return np.sort(rng.choice(a, min(k, len(a)), replace=False))


def _dev(store, local_idx):
    cp = store["cp"]
    return cp.asarray(store["pc"][np.asarray(local_idx, int)], dtype=cp.int64)


def _fired_local(store):
    from sim.backend import to_host
    return np.asarray(to_host(store["bridge"].cp_firing_states))[store["pc"]].astype(bool)


# ---------------------------------------------------------------------------------------------------------------------
# Waking episodes
# ---------------------------------------------------------------------------------------------------------------------
def encode(store, seed, c, design):
    """STDP on, clock advanced every step. Each episode is a moving-cue sweep over its slots; a slot drives the cue
    cells of its assembly for enc_dwell of its enc_step steps. After each episode: enc_settle silent steps and the
    STDP spike clocks cleared (the next episode is minutes later; no pairing crosses it)."""
    b = store["bridge"]; cp = store["cp"]
    rng = np.random.default_rng(int(seed) * 777 + 5)
    cue = {n: _dev(store, _cue_cells(store, n, c["enc_cue_frac"], rng)) for n in NAMES}
    b.core_config.enable_stdp = True
    b.runtime_state.current_time_ms = 0.0
    n_steps = 0; n_eps = 0
    phases = {}
    for ph, k, slots in _chain_episodes(design):
        phases.setdefault(ph, []).append((k, slots))
    for ph in ("X", "Y"):
        for lap in range(int(c["n_laps"])):
            for k, slots in phases[ph]:
                for (name, scale, detail_only) in slots:
                    if detail_only:
                        if lap % int(c["detail_every"]) != 0:
                            continue
                        scale = float(c["detail_drive"])
                    for s in range(int(c["enc_step"])):
                        b.cp_external_input_current[:] = 0.0
                        if s < int(c["enc_dwell"]):
                            b.cp_external_input_current[cue[name]] += float(c["enc_cue_pa"]) * float(scale)
                        b._run_one_simulation_step()
                        b.runtime_state.current_time_ms += float(c["dt"])
                        n_steps += 1
                for _ in range(int(c["enc_settle"])):
                    b.cp_external_input_current[:] = 0.0
                    b._run_one_simulation_step()
                    b.runtime_state.current_time_ms += float(c["dt"])
                    n_steps += 1
                b.cp_last_spike_time[:] = cp.float32(-1000.0)      # minutes between episodes: no cross pairing
                n_eps += 1
    b.core_config.enable_stdp = False
    b.cp_external_input_current[:] = 0.0
    return dict(n_steps=n_steps, n_episodes=n_eps)


# ---------------------------------------------------------------------------------------------------------------------
# Night
# ---------------------------------------------------------------------------------------------------------------------
def night(store, seed, c, *, seed_on=True, stdp_on=True, record=True, disrupt=False):
    """SWR replay (seed_on) with the engine's STDP (stdp_on), clock advanced every step. Returns the raster (for the
    replica and the replay counts), the initiation log and the weights after.

    SWR initiation draws uniformly from every assembly except the test context (`probe_context`, never part of
    experience, so it never replays). `disrupt` is the replay lesion (Girardeau et al. 2009; Ego-Stengel & Wilson
    2010: stimulation that truncates each ripple): initiation proceeds exactly as in `sleep`, then from `dis_start`
    steps after each initiation, for `dis_len` steps, every pyramidal cell receives -`dis_pa`, which ends the event
    before it propagates past the initiating assembly. The raster records whether it held (U2)."""
    b = store["bridge"]
    T = int(c["night_steps"]); P = int(c["swr_period"])
    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    cue = [_dev(store, _cue_cells(store, n, c["cue_frac"], cell_rng)) for n in NAMES]
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)
    pool = [i for i, n in enumerate(NAMES) if n != c["probe_context"]]
    all_pc = _dev(store, np.arange(store["n_pc"]))
    d0, d1 = int(c["dis_start"]), int(c["dis_start"]) + int(c["dis_len"])
    F = np.zeros((T, store["n_pc"]), dtype=bool) if record else None
    log = []
    cur = None
    b.core_config.enable_stdp = bool(stdp_on)
    b.runtime_state.current_time_ms = 0.0
    for t in range(T):
        b.cp_external_input_current[:] = 0.0
        ph = t % P
        if ph == 0 and seed_on:
            cur = pool[int(choice_rng.integers(0, len(pool)))]; log.append(cur)
        if seed_on and ph < int(c["cue_steps"]) and cur is not None:
            b.cp_external_input_current[cue[cur]] += float(c["cue_pa"])
        if disrupt and seed_on and d0 <= ph < d1:
            b.cp_external_input_current[all_pc] -= float(c["dis_pa"])
        b._run_one_simulation_step()
        b.runtime_state.current_time_ms += float(c["dt"])
        if record:
            F[t] = _fired_local(store)
    b.core_config.enable_stdp = False
    b.cp_external_input_current[:] = 0.0
    return dict(F=F, log=log, w=get_w(store))


def replica_stdp(store, F, w0, c):
    """An exact host replica of the engine's STDP step (sim/bridge.py compacting path + sim/kernels.py
    fused_stdp_weight_update) on the plastic (between) synapses, driven by a given raster. Used ONLY by the yok
    (instrument) and scr (surrogate) arms; the yok arm checks that it reproduces the engine."""
    w = np.asarray(w0, dtype=np.float32).copy()
    pos = store["between_pos"]; pre = store["syn_pre_l"]; post = store["syn_post_l"]
    ws = w[pos].copy()
    n = store["n_pc"]
    by_pre = [[] for _ in range(n)]; by_post = [[] for _ in range(n)]
    order_pre = np.argsort(pre, kind="stable"); order_post = np.argsort(post, kind="stable")
    ip = np.searchsorted(pre[order_pre], np.arange(n + 1)); iq = np.searchsorted(post[order_post], np.arange(n + 1))
    a_plus = np.float32(c["stdp_a_plus"]); a_minus = np.float32(c["stdp_a_minus"])
    tau = float(c["stdp_tau"]); w_min = np.float32(0.0); w_max = np.float32(c["stdp_w_max"])
    window = max(tau, tau) * 5.0
    last = np.full(n, -1000.0, dtype=np.float32)
    now = 0.0
    for t in range(F.shape[0]):
        fired = np.nonzero(F[t])[0]
        if fired.size:
            last[fired] = np.float32(now)
            cand = np.unique(np.concatenate([order_pre[ip[f]:ip[f + 1]] for f in fired]
                                            + [order_post[iq[f]:iq[f + 1]] for f in fired]))
            if cand.size:
                tp = last[pre[cand]]; tq = last[post[cand]]
                dtt = (tq - tp).astype(np.float32)
                valid = (tp > -500.0) & (tq > -500.0) & (np.abs(dtt) < window)
                cc = cand[valid]; d = dtt[valid]
                if cc.size:
                    wc = ws[cc]
                    ltp = np.where(d > 0.0, a_plus * (w_max - wc) * np.exp(-d / np.float32(tau)), np.float32(0.0))
                    ltd = np.where(d < 0.0, -a_minus * (wc - w_min) * np.exp(d / np.float32(tau)), np.float32(0.0))
                    ws[cc] = np.clip((wc + ltp + ltd).astype(np.float32), w_min, w_max)
        now += float(c["dt"])
    w[pos] = ws
    return w


def scramble_raster(store, F, seed, c):
    """Each assembly's activity (all its cells together, and each extra cell alone) circularly shifted by a random
    whole number of SWR periods: spike counts, within-assembly timing and every assembly's event structure kept;
    which assemblies are active together, and in what order, destroyed."""
    T = F.shape[0]; P = int(c["swr_period"]); nper = max(2, T // P)
    rng = np.random.default_rng(int(seed) * 6007 + 11)
    G = np.zeros_like(F)
    shifts = {}
    for i, al in enumerate(store["asm_local"]):
        sh = int(rng.integers(1, nper)) * P
        shifts[NAMES[i]] = sh
        G[:, al] = np.roll(F[:, al], sh, axis=0)
    covered = np.zeros(store["n_pc"], bool)
    for al in store["asm_local"]:
        covered[al] = True
    for j in np.nonzero(~covered)[0]:
        G[:, j] = np.roll(F[:, j], int(rng.integers(1, nper)) * P, axis=0)
    return G, shifts


def replay_counts(store, F, c, log=None):
    """Per assembly, the number of SWR periods in which it took part (at least particip_frac of its cells spiked),
    the co-participation matrix, and (given the initiation log) per-event reach: whether the initiating assembly took
    part and how many assemblies did. The lesion checks (U2) read these off the raster, not off a host flag."""
    P = int(c["swr_period"]); T = F.shape[0]
    cnt = np.zeros(store["m"], int)
    coact = np.zeros((store["m"], store["m"]), int)
    n_part = []
    init_fired = []
    for n in range(T // P):
        seg = F[n * P:(n + 1) * P]
        act = []
        for i, al in enumerate(store["asm_local"]):
            if float(seg[:, al].any(0).mean()) >= float(c["particip_frac"]):
                act.append(i)
        for i in act:
            cnt[i] += 1
            for j in act:
                coact[i, j] += 1
        n_part.append(len(act))
        if log is not None and n < len(log):
            init_fired.append(bool(log[n] in act))
    reach = dict(n_periods=int(T // P), mean_participating=float(np.mean(n_part)) if n_part else 0.0,
                 frac_propagated=float(np.mean([k >= 2 for k in n_part])) if n_part else 0.0,
                 frac_init_fired=(float(np.mean(init_fired)) if init_fired else None))
    return cnt, coact, reach


def reverse_raster(store, F, c):
    """Order-only surrogate: time-reversed inside every SWR period. Every cell's spike count, every period's set of
    co-participating assemblies and every pair's co-occurrence are kept; only the order inside an event is reversed."""
    P = int(c["swr_period"]); T = F.shape[0]
    G = F.copy()
    for n in range(T // P):
        G[n * P:(n + 1) * P] = F[n * P:(n + 1) * P][::-1]
    return G


def permute_dw(store, w0, w_ref, seed):
    """Weight-level control: the normal night's own weight changes on the plastic synapses, permuted among them (seeded).
    The multiset of changes, their total and their distribution are exactly the night's; only WHICH synapse received
    which change is destroyed (activity matched at the weight level)."""
    pos = store["between_pos"]
    d = np.asarray(w_ref, np.float64)[pos] - np.asarray(w0, np.float64)[pos]
    perm = np.random.default_rng(int(seed) * 7727 + 13).permutation(len(pos))
    w = np.asarray(w0, np.float32).copy()
    w[pos] = (np.asarray(w0, np.float64)[pos] + d[perm]).astype(np.float32)
    return w


def budget_match(store, w0, w_ref, w_sur, w_max):
    """Scale a surrogate's weight change on the plastic synapses so that its total |dw| equals the reference night's
    (activity matched at the WEIGHT level: the same amount of plasticity, redistributed). Returns (w, scale)."""
    pos = store["between_pos"]
    d_ref = np.abs(np.asarray(w_ref, np.float64)[pos] - np.asarray(w0, np.float64)[pos]).sum()
    d_sur = np.asarray(w_sur, np.float64)[pos] - np.asarray(w0, np.float64)[pos]
    tot = np.abs(d_sur).sum()
    k = float(d_ref / tot) if tot > 0 else 0.0
    w = np.asarray(w0, np.float32).copy()
    w[pos] = np.clip(np.asarray(w0, np.float64)[pos] + k * d_sur, 0.0, float(w_max)).astype(np.float32)
    return w, k


# ---------------------------------------------------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------------------------------------------------
def probe(seed, c, w, cue_name, ctx_name, *, silence=None, tag="", store_kw=None):
    """Fresh build with weights w, STDP off. For each trial: drive cue_name's (and ctx_name's) cue cells for
    cue_steps, silence `silence` for the whole trial if given, run probe_steps, then probe_settle silent steps.
    Returns, per assembly, the mean over trials of the recruited fraction (the fraction of the assembly's cells that
    spiked in the trial window): the network's own answer."""
    st = build_store(seed, c, **(store_kw or {}))
    load_w(st, w)
    b = st["bridge"]
    peaks = np.zeros((int(c["probe_trials"]), st["m"]))
    onset = np.full((int(c["probe_trials"]), st["m"]), np.nan)
    sil = _dev(st, st["asm_local"][IDX[silence]]) if silence else None
    for r in range(int(c["probe_trials"])):
        rng = np.random.default_rng(int(seed) * 7919 + 101 * r + 7)
        cue = _dev(st, _cue_cells(st, cue_name, c["probe_cue_frac"], rng))
        ctx = _dev(st, _cue_cells(st, ctx_name, c["probe_cue_frac"], rng)) if ctx_name else None
        F = np.zeros((int(c["probe_steps"]), st["n_pc"]), bool)
        for t in range(int(c["probe_steps"])):
            b.cp_external_input_current[:] = 0.0
            if t < int(c["cue_steps"]):
                b.cp_external_input_current[cue] += float(c["probe_cue_pa"])
                if ctx is not None:
                    b.cp_external_input_current[ctx] += float(c["ctx_cue_pa"])
            if sil is not None:
                b.cp_external_input_current[sil] -= float(c["silence_pa"])
            b._run_one_simulation_step()
            F[t] = _fired_local(st)
        for i, al in enumerate(st["asm_local"]):
            peaks[r, i] = float(F[:, al].any(0).mean())     # recruited fraction: cells of the assembly that spiked
            cum = np.maximum.accumulate(F[:, al], axis=0).mean(1)
            hit = np.nonzero(cum >= float(c["particip_frac"]))[0]
            if hit.size:
                onset[r, i] = float(hit[0])                  # steps from probe start to 25 % recruited
        for _ in range(int(c["probe_settle"])):
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()
    with np.errstate(all="ignore"):
        on = np.nanmean(onset, 0)
    return dict(mean=peaks.mean(0).tolist(), trials=peaks.tolist(), onset_steps=[None if np.isnan(x) else float(x) for x in on],
                tag=tag)


def reads(seed, c, w, *, store_kw=None, ctx=None):
    """The registered probe set on weights w. Per chain k: cue A_k (+ctx) free; cue A_k (+ctx) with B_k silenced;
    cue B_k (+ctx) (premise B->D and the detail B->P). Scores are differences against the other chain's matched
    assembly (the foil), so every marginal statistic cancels."""
    ctx = ctx or c["probe_context"]
    out = {}
    for k in (1, 2):
        o = OTHER[k]
        fr = probe(seed, c, w, f"A{k}", ctx, tag=f"A{k}+{ctx}", store_kw=store_kw)
        sl = probe(seed, c, w, f"A{k}", ctx, silence=f"B{k}", tag=f"A{k}+{ctx}|B{k}-silenced", store_kw=store_kw)
        bd = probe(seed, c, w, f"B{k}", ctx, tag=f"B{k}+{ctx}", store_kw=store_kw)
        g = lambda p, n: float(p["mean"][IDX[n]])  # noqa: E731
        out[f"chain{k}"] = dict(
            free_D=g(fr, f"D{k}"), free_Dfoil=g(fr, f"D{o}"),
            direct_D=g(sl, f"D{k}"), direct_Dfoil=g(sl, f"D{o}"), direct_B=g(sl, f"B{k}"),
            free_D_onset_steps=fr["onset_steps"][IDX[f"D{k}"]], free_B_onset_steps=fr["onset_steps"][IDX[f"B{k}"]],
            AB=g(fr, f"B{k}"), AB_foil=g(fr, f"B{o}"), BD=g(bd, f"D{k}"), BD_foil=g(bd, f"D{o}"),
            BP=g(bd, f"P{k}"), BP_foil=g(bd, f"P{o}"),
            spurious_C=max(g(sl, f"C{k}"), g(sl, f"C{o}")), probes=[fr, sl, bd])
    ch = [out["chain1"], out["chain2"]]
    out["DI"] = float(np.mean([x["direct_D"] - x["direct_Dfoil"] for x in ch]))
    out["FI"] = float(np.mean([x["free_D"] - x["free_Dfoil"] for x in ch]))
    out["PAB"] = float(np.mean([x["AB"] - x["AB_foil"] for x in ch]))
    out["PBD"] = float(np.mean([x["BD"] - x["BD_foil"] for x in ch]))
    out["PBP"] = float(np.mean([x["BP"] - x["BP_foil"] for x in ch]))
    out["silence_held"] = bool(max(x["direct_B"] for x in ch) < float(c["active_frac"]))
    out["foil_direct_max"] = float(max(x["direct_Dfoil"] for x in ch))
    out["spurious_C_max"] = float(max(x["spurious_C"] for x in ch))
    return out


def weight_reads(store, w):
    M = pair_matrix(store, w)
    r = {}
    for k in (1, 2):
        o = OTHER[k]
        f = lambda a, bb: float(M[IDX[a], IDX[bb]])  # noqa: E731
        r[f"chain{k}"] = dict(AD=f(f"A{k}", f"D{k}"), AD_foil=f(f"A{k}", f"D{o}"), DA=f(f"D{k}", f"A{k}"),
                              DA_foil=f(f"D{o}", f"A{k}"), DB=f(f"D{k}", f"B{k}"),
                              AB=f(f"A{k}", f"B{k}"), BD=f(f"B{k}", f"D{k}"), BP=f(f"B{k}", f"P{k}"),
                              XA=f(f"X{k}", f"A{k}"), YD=f(f"Y{k}", f"D{k}"), AB_foil=f(f"A{k}", f"B{o}"))
    ch = [r["chain1"], r["chain2"]]
    r["WAD"] = float(np.mean([x["AD"] - x["AD_foil"] for x in ch]))
    r["WDA"] = float(np.mean([x["DA"] - x["DA_foil"] for x in ch]))
    off = ~np.eye(len(NAMES), dtype=bool)
    r["between_mean"] = float(M[off].mean())
    r["WAB"] = float(np.mean([x["AB"] - x["AB_foil"] for x in ch]))
    r["WBD"] = float(np.mean([x["BD"] for x in ch]))
    r["WBP"] = float(np.mean([x["BP"] for x in ch]))
    r["WCTX"] = float(np.mean([0.5 * (x["XA"] + x["YD"]) for x in ch]))
    r["matrix"] = M.round(3).tolist()
    return r


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# One seed: every arm
# ---------------------------------------------------------------------------------------------------------------------
ALL_ARMS = ("wake", "sleep", "off", "dis", "yok", "wperm", "scr", "rev", "cross", "nohub", "nostdp", "nocomp",
            "noadapt")


def run_seed(seed, c, arms=ALL_ARMS, log=print):
    t0 = time.time()
    out = dict(seed=int(seed), cfg=dict(c), names=NAMES, arms={}, timing={})

    def tick(msg):
        log(f"  [seed {seed} +{time.time() - t0:.0f}s] {msg}", flush=True) if log is print else log(msg)

    st = build_store(seed, c)
    w_init = get_w(st)
    out["init_hash"] = _h(w_init)
    hub_arms = {"wake", "sleep", "off", "dis", "yok", "wperm", "scr", "rev", "nostdp", "nocomp", "noadapt"}
    w_enc = None
    if hub_arms & set(arms):
        enc = encode(st, seed, c, "hub")
        w_enc = get_w(st)
        out["encode"] = dict(enc, w_hash=_h(w_enc), init_hash=_h(w_init))
        out["pre_weights"] = weight_reads(st, w_enc)
        tick(f"encoded hub ({enc['n_steps']} steps): WAB={out['pre_weights']['WAB']:.1f} "
             f"WBD={out['pre_weights']['WBD']:.1f} WAD={out['pre_weights']['WAD']:.2f}")
    out["timing"]["encode"] = time.time() - t0
    out["arms_requested"] = list(arms)
    pos = st["between_pos"]

    def arm_record(name, w_post, *, w_pre, night_rec=None, extra=None):
        rd = reads(seed, c, w_post)
        dw = np.abs(np.asarray(w_post, np.float64)[pos] - np.asarray(w_pre, np.float64)[pos])
        rec = dict(weights=weight_reads(st, w_post), reads=rd, w_hash=_h(w_post),
                   mean_abs_dw_between=float(dw.mean()))
        if night_rec is not None:
            cnt, co, reach = replay_counts(st, night_rec["F"], c, night_rec.get("log"))
            rec["night"] = dict(n_init=len(night_rec.get("log") or []), init_log=night_rec.get("log"),
                                replay_counts=cnt.tolist(), reach=reach,
                                coact_AD=[int(co[IDX["A1"], IDX["D1"]]), int(co[IDX["A2"], IDX["D2"]])],
                                coact_AD_foil=[int(co[IDX["A1"], IDX["D2"]]), int(co[IDX["A2"], IDX["D1"]])],
                                pop_rate=float(night_rec["F"].mean()),
                                total_spikes=int(night_rec["F"].sum()))
        if extra:
            rec.update(extra)
        out["arms"][name] = rec
        tick(f"{name}: DI={rd['DI']:.3f} FI={rd['FI']:.3f} PAB={rd['PAB']:.3f} PBD={rd['PBD']:.3f} "
             f"PBP={rd['PBP']:.3f} foil={rd['foil_direct_max']:.3f} WAD={rec['weights']['WAD']:.1f} "
             f"WDA={rec['weights']['WDA']:.1f} |dw|={rec['mean_abs_dw_between']:.2f}")
        return rec

    def fresh(w, **kw):
        s2 = build_store(seed, c, **kw); load_w(s2, w)
        return s2

    if "wake" in arms:
        arm_record("wake", w_enc, w_pre=w_enc)
    sleep_night = None
    if any(a in arms for a in ("sleep", "yok", "wperm", "scr", "rev")):
        sleep_night = night(fresh(w_enc), seed, c, seed_on=True, stdp_on=True)
        arm_record("sleep", sleep_night["w"], w_pre=w_enc, night_rec=sleep_night)
    if "off" in arms:
        nr = night(fresh(w_enc), seed, c, seed_on=False, stdp_on=True)
        arm_record("off", nr["w"], w_pre=w_enc, night_rec=nr,
                   extra=dict(weights_unchanged=bool(np.array_equal(nr["w"], w_enc))))
    if "dis" in arms:
        nr = night(fresh(w_enc), seed, c, seed_on=True, stdp_on=True, disrupt=True)
        arm_record("dis", nr["w"], w_pre=w_enc, night_rec=nr)
    if "yok" in arms and sleep_night is not None:
        wy = replica_stdp(st, sleep_night["F"], w_enc, c)
        dif = np.abs(wy - sleep_night["w"])
        yok = dict(max_abs_diff=float(dif.max()), n_diff=int((dif > 1e-3).sum()))
        out["arms"]["yok"] = dict(replica=yok, weights=weight_reads(st, wy))
        tick(f"yok replica vs engine: max|dw|={yok['max_abs_diff']:.3g} n>1e-3={yok['n_diff']}")
    if "wperm" in arms and sleep_night is not None:
        wp = permute_dw(st, w_enc, sleep_night["w"], seed)
        d_s = np.sort(np.asarray(sleep_night["w"], np.float64)[pos] - np.asarray(w_enc, np.float64)[pos])
        d_p = np.sort(np.asarray(wp, np.float64)[pos] - np.asarray(w_enc, np.float64)[pos])
        arm_record("wperm", wp, w_pre=w_enc, extra=dict(dw_multiset_equal=bool(np.allclose(d_s, d_p, atol=1e-3))))
    if "scr" in arms and sleep_night is not None:
        G, shifts = scramble_raster(st, sleep_night["F"], seed, c)
        w_raw = replica_stdp(st, G, w_enc, c)
        ws, k = budget_match(st, w_enc, sleep_night["w"], w_raw, c["stdp_w_max"])
        cnt_g, _, _ = replay_counts(st, G, c)
        cnt_f, _, _ = replay_counts(st, sleep_night["F"], c)
        raw_dw = float(np.abs(np.asarray(w_raw, np.float64)[pos] - np.asarray(w_enc, np.float64)[pos]).mean())
        arm_record("scr", ws, w_pre=w_enc,
                   extra=dict(shifts=shifts, budget_scale=k, raw_mean_abs_dw_between=raw_dw,
                              raw_weights=weight_reads(st, w_raw),
                              spike_count_equal=bool(np.array_equal(G.sum(0), sleep_night["F"].sum(0))),
                              replay_counts=cnt_g.tolist(), replay_counts_equal=bool(np.array_equal(cnt_g, cnt_f))))
        del G
    if "rev" in arms and sleep_night is not None:
        G = reverse_raster(st, sleep_night["F"], c)
        wr = replica_stdp(st, G, w_enc, c)
        arm_record("rev", wr, w_pre=w_enc,
                   extra=dict(spike_count_equal=bool(np.array_equal(G.sum(0), sleep_night["F"].sum(0)))))
        del G
    if "nostdp" in arms:
        nr = night(fresh(w_enc), seed, c, seed_on=True, stdp_on=False)
        arm_record("nostdp", nr["w"], w_pre=w_enc, night_rec=nr,
                   extra=dict(weights_unchanged=bool(np.array_equal(nr["w"], w_enc))))
    if "nocomp" in arms:
        wl = w_enc.copy(); wl[st["within_pos"]] = 0.0
        s5 = fresh(wl)
        held = bool(np.all(get_w(s5)[st["within_pos"]] == 0.0))
        nr = night(s5, seed, c, seed_on=True, stdp_on=True)
        held = held and bool(np.all(nr["w"][st["within_pos"]] == 0.0))      # the lesion still held at the night's end
        wpost = nr["w"].copy(); wpost[st["within_pos"]] = w_enc[st["within_pos"]]   # completion restored for the probe
        arm_record("nocomp", wpost, w_pre=w_enc, night_rec=nr, extra=dict(lesion_held=held))
    if "noadapt" in arms:
        s6 = fresh(w_enc, a_override=0.0, b_override=0.0)
        held = bool(float(s6["bridge"].core_config.adex_a) == 0.0 and float(s6["bridge"].core_config.adex_b) == 0.0)
        nr = night(s6, seed, c, seed_on=True, stdp_on=True)
        held = held and bool(float(s6["bridge"].core_config.adex_a) == 0.0
                             and float(s6["bridge"].core_config.adex_b) == 0.0)
        arm_record("noadapt", nr["w"], w_pre=w_enc, night_rec=nr, extra=dict(lesion_held=held))
    for design in ("cross", "nohub"):
        if design not in arms:
            continue
        sd = build_store(seed, c)
        encd = encode(sd, seed, c, design)
        wd = get_w(sd)
        arm_record(f"{design}_pre", wd, w_pre=wd, extra=dict(encode=encd))
        nr = night(fresh(wd), seed, c, seed_on=True, stdp_on=True)
        arm_record(design, nr["w"], w_pre=wd, night_rec=nr, extra=dict(encode=encd))
    # attribution (REPORTED; the verdict is the grader's): the fraction of the night's A->D weight change that is not
    # present under the replay lesion
    if "sleep" in out["arms"] and "dis" in out["arms"]:
        from tools.lab import attributable_to
        out["attribution_WAD_sleep_vs_dis"] = attributable_to(
            "A->D weight change: sleep vs disrupted replay",
            out["arms"]["sleep"]["weights"]["WAD"] - out["pre_weights"]["WAD"],
            out["arms"]["dis"]["weights"]["WAD"] - out["pre_weights"]["WAD"])
    # REPORTED: the training-context probe after the normal night (context dependence)
    if sleep_night is not None:
        out["sleep_trainctx_DI"] = {}
        for k in (1, 2):
            p = probe(seed, c, sleep_night["w"], f"A{k}", f"X{k}", silence=f"B{k}", tag=f"A{k}+X{k}|B{k}-sil")
            out["sleep_trainctx_DI"][f"chain{k}"] = float(p["mean"][IDX[f"D{k}"]] - p["mean"][IDX[f"D{OTHER[k]}"]])
    out["timing"]["total"] = time.time() - t0
    for rec in out["arms"].values():
        rd = rec.get("reads")
        if rd:
            for k in (1, 2):
                rd[f"chain{k}"].pop("probes", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=False)
    ap.add_argument("--dev", action="store_true", help="dev run: refuses the six gate seeds")
    ap.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    ap.add_argument("--set", nargs="*", default=[], help="override a DEFAULTS constant: key=value (dev only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        from research.runners.sleep_transitive_inference_grade import selftest
        sys.exit(0 if selftest() else 1)
    if a.seed is None:
        ap.error("--seed is required")
    if a.dev and a.seed in GATE_SEEDS:
        ap.error(f"--dev refuses gate seed {a.seed}")
    if a.set and not a.dev:
        ap.error("--set is allowed only with --dev (gate runs use the registered constants)")
    if not a.dev and os.environ.get("SIM_BACKEND") != "numpy":
        ap.error("gate runs are registered on numpy (the STDP replica is exact there; on cupy it is not): "
                 f"SIM_BACKEND={os.environ.get('SIM_BACKEND')!r}")
    c = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in c:
            ap.error(f"unknown constant {k}")
        c[k] = type(c[k])(v) if not isinstance(c[k], bool) else v.lower() in ("1", "true")
    out_path = Path(a.out) if a.out else OUT_DIR / ("dev" if a.dev else "") / f"seed{a.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        res = run_seed(a.seed, c, arms=tuple(a.arms))
        res["error"] = None
    except Exception as e:  # noqa: BLE001
        res = dict(seed=a.seed, cfg=c, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(res["traceback"], flush=True)
    res["backend"] = os.environ.get("SIM_BACKEND")
    res["role"] = "dev" if a.dev else "gate"
    try:
        from research.runners.sleep_transitive_inference_grade import grade_seed
        res["grade"] = grade_seed(res)
    except Exception as e:  # noqa: BLE001
        res["grade"] = dict(error=f"{type(e).__name__}: {e}")
    out_path.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out_path}", flush=True)
    print(json.dumps(res.get("grade", {}), indent=1, default=float)[:4000], flush=True)


if __name__ == "__main__":
    main()
