"""D pragmatics -- Leg 2 v2 CONVERGENCE wall, an UNTRIED lever: a region-scoped firing-rate HOMEOSTAT
(Diehl-Cook adaptive threshold) run over a SETTLING/EXPOSURE phase on the competing critic/actor populations,
to EQUALIZE intrinsic per-neuron excitability -> LOWER the heterogeneity noise floor -> let the tiny DA-learned
value/policy differential (~0.01-0.02) separate the aligned utterance.

THIS IS A SMOKE / DE-RISK. It does NOT modify the committed runner
(research/runners/_pragmatic_success_readback_leg2_v2_derisk.py) -- it imports its helpers by reference and only
adds (a) a homeostat-scoped bridge builder + exposure phase and (b) the mechanistic noise measurement. NO sim/ edit.

THE WALL (measured, 2026-08-08): the DA-trained value/policy DIFFERENTIAL is real but tiny (success values
~0.027-0.052; gaps ~0.01-0.02) and swamped by per-neuron heterogeneity noise; the committed 6-seed critic-argmax
sits at 0.556 and actor-WTA at 0.500 (near chance 0.333). The oracle-weight precondition is RESOLVED (1.0/6).

THE HYPOTHESIS: an SNR problem -- signal < intrinsic-heterogeneity noise. A per-neuron rate homeostat over a
symmetric exposure phase raises the thresholds of intrinsically-hot cells and lowers the cold ones, equalizing
excitability so the per-assembly mean-rate bias shrinks; the learned differential can then win.

VERIFIED SUBSTRATE ANCHORS (confirmed this session, NO sim/ edit):
  - BrainRegion.enable_homeostasis=True region-scopes the homeostat (mask sim/bridge.py:2085-2097). Only masked
    regions USE adapted thresholds for spike detection (sim/bridge.py:8895-8898) and only they matter for the read.
  - The GLOBAL defaults are deliberately slow (adapt_rate 0.0005, ema_alpha 0.0002): on a brief probe the default
    NEVER engages. We RAISE both and run an EXPOSURE phase (the operating point IS the mechanism).
  - IMPORTANT operating-point facts found this session: cp_izh_vpeak=+35mV (baseline spike detection), but the
    homeostat threshold array inits uniform in [-55,-30] and the update kernel (sim/kernels.py:1314) CLIPS to
    [-55,-30]. So a masked neuron can NEVER be made as quiet as baseline; the homeostat only tunes within a 25mV
    band near vt. Whether it can equalize under this clip is exactly what the mechanistic probe measures.
  - The substrate (izh heterogeneity, weights, threshold-array VALUES) is drawn identically with/without the homeo
    mask (threshold array is allocated unconditionally for Izhikevich; heterogeneity applies globally). The ONLY
    difference the mask makes is spike-detection threshold + adaptation for utter/crit -> a clean comparison.

Usage:
  # CHEAP mechanistic + selectivity probe (no training) across the adapt-rate x exposure sweep:
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v2_homeostat_derisk --mech \
      --seeds 42 44 100 --json .../homeo_mech.json
  # FULL critic-value-separability sweep (trains, ~75s/seed/condition):
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v2_homeostat_derisk --sweep \
      --seeds 42 44 100 --json .../homeo_sweep.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _snapshot_state, _restore_state, SETTLE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection  # noqa: E402
from research.runners._self_schema_region_derisk import WS_LOOP_GATE  # noqa: E402
from research.runners._pragmatic_success_coincidence_derisk import (  # noqa: E402
    ITEM, DET, K, BELIEF_TOTAL, INTENT_PA, W_SYN, K_THR, GAIN, PLATEAU,
)
# reuse the committed leg2-v2 helpers + constants VERBATIM (choice/eval/commit/reward + belief sources)
import research.runners._pragmatic_success_readback_leg2_v2_derisk as V2  # noqa: E402
from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    UTT_ITEM, UTT_FS_N, UTT_FS_W, FS_UTT_W, UTT_DRIVE_PA, W_ORACLE, W_I2U_INIT, W_I2U_JIT, W_OTHER,
    CRIT_ITEM, W_I2C_INIT, CRIT_GATE, CRIT_READ_GAIN, SPEAK_GATE, SETTLE_MS, READ_UTT,
    REWARD_GAIN, EPSILON, N_TRAIN, LR, ELIG_TAU,
    _belief_sources, _aligned_utts, _choose_utterance, _commit_action, _evaluate_success,
    _deliver_reward, _readout_policy,
)

# ── homeostat operating point (the lever) ─────────────────────────────────────────────────────────────────────
HOMEO_TARGET_RATE = 0.02     # Diehl-Cook target (== engine default homeostasis_target_rate)
HOMEO_EMA_ALPHA = 0.02       # RAISED from the 0.0002 default (tau ~50 steps) so the rate estimate tracks exposure
EXPOSE_DRIVE_PA = INTENT_PA  # intent drive during exposure (same operating point as the decision)


def build_speaker_bridge_homeo(seed, oracle=False, homeo=True, adapt_rate=0.05, ema_alpha=HOMEO_EMA_ALPHA,
                               expose_steps=2000, expose_drive=EXPOSE_DRIVE_PA, freeze_after=True,
                               target_rate=HOMEO_TARGET_RATE, thresh_max=-30.0):
    """Replicates V2.build_speaker_bridge (same wiring, same seeded substrate) but (when homeo=True) sets
    BrainRegion.enable_homeostasis on the competing readout populations (utter + crit), RAISES the adapt-rate +
    ema-alpha, and runs a SYMMETRIC cycled EXPOSURE phase so the homeostat equalizes intrinsic excitability BEFORE
    the decision. homeo=False reproduces the committed build byte-for-byte (verified).

    Returns (bridge, xp, idx, snap, diag) where diag records whether the homeostat engaged (ema/threshold path)."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="intent", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="utter", n_neurons=UTT_ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False,
                    enable_homeostasis=bool(homeo)),
        BrainRegion(name="utter_fs", n_neurons=UTT_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="crit", n_neurons=CRIT_ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False,
                    enable_homeostasis=bool(homeo)),
        BrainRegion(name="belief", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="success", n_neurons=DET * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="utter", to_region="utter_fs", density=0.6, weight_mean=UTT_FS_W,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="utter_fs", to_region="utter", density=0.6, weight_mean=FS_UTT_W,
                      weight_jitter=0.0, plastic=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process", "enable_nmda"):
        setattr(cfg, f, False)                              # GLOBAL homeostasis stays OFF -> region mask drives it
    cfg.enable_parameter_heterogeneity = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(LR)
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.reward_eligibility_tau_ms = float(ELIG_TAU)
    cfg.reward_eligibility_from_coactivity = True
    cfg.reward_coactivity_trace_tau_ms = float(ELIG_TAU)
    cfg.reward_coactivity_scale = 0.2
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0
    cfg.enable_coincidence_detection = True
    cfg.coincidence_k_threshold = float(K_THR)
    cfg.coincidence_gain = float(GAIN)
    cfg.coincidence_plateau_strength = float(PLATEAU)
    # homeostat operating point (only bites the enable_homeostasis regions via the per-region mask)
    cfg.homeostasis_target_rate = float(target_rate)
    cfg.homeostasis_threshold_adapt_rate = float(adapt_rate)
    cfg.homeostasis_ema_alpha = float(ema_alpha)
    # RAISE the threshold clip ceiling (cfg value, NO sim edit): the default -30 caps thresholds far below
    # vpeak(+35), so under the strong INTENT drive the masked neurons stay saturated (~5x target) and the homeostat
    # has no headroom to quiet/equalize them. Raising thresh_max gives it room to reach target. Init threshold
    # array becomes uniform in [thresh_min, thresh_max] but the RNG-draw count is unchanged -> downstream
    # heterogeneity draws are identical.
    cfg.homeostasis_threshold_max = float(thresh_max)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    rm = bridge.region_manager
    intent = np.asarray(rm.indices("intent"), dtype=np.int64)
    utter = np.asarray(rm.indices("utter"), dtype=np.int64)
    crit = np.asarray(rm.indices("crit"), dtype=np.int64)
    belief = np.asarray(rm.indices("belief"), dtype=np.int64)
    suc = np.asarray(rm.indices("success"), dtype=np.int64)
    intent_k = {k: intent[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    utter_k = {k: utter[k * UTT_ITEM:(k + 1) * UTT_ITEM] for k in range(K)}
    crit_k = {k: crit[k * CRIT_ITEM:(k + 1) * CRIT_ITEM] for k in range(K)}
    belief_k = {k: belief[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    suc_k = {k: suc[k * DET:(k + 1) * DET] for k in range(K)}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    rng = np.random.default_rng(seed * 17 + 3)             # SAME stream as the committed builder
    for t in range(K):
        for u in range(K):
            pre = np.repeat(intent_k[t], UTT_ITEM)
            post = np.tile(utter_k[u], ITEM)
            if oracle:
                w = np.full(pre.shape[0], (W_ORACLE if t == u else W_OTHER), dtype=np.float32)
                union[f"i2u_{t}_{u}"] = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
                                        "initial_weights": w, "plastic": False, "conn_type": "E_TO_E",
                                        "count": int(pre.size)}
            else:
                w = (W_I2U_INIT + rng.normal(0.0, W_I2U_JIT, pre.shape[0])).astype(np.float32)
                union[f"i2u_{t}_{u}"] = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
                                        "initial_weights": np.clip(w, 0.1, None), "plastic": True,
                                        "plasticity_gate": SPEAK_GATE, "conn_type": "E_TO_E", "count": int(pre.size)}
            if not oracle:
                cpre = np.repeat(intent_k[t], CRIT_ITEM)
                cpost = np.tile(crit_k[u], ITEM)
                cw = np.full(cpre.shape[0], W_I2C_INIT, dtype=np.float32)
                union[f"i2c_{t}_{u}"] = {"pre_indices": cpre.astype(np.int64), "post_indices": cpost.astype(np.int64),
                                        "initial_weights": cw, "plastic": True, "plasticity_gate": CRIT_GATE,
                                        "conn_type": "E_TO_E", "count": int(cpre.size)}
    for k in range(K):
        d1 = _dense_projection(belief_k[k], suc_k[k], W_SYN, WS_LOOP_GATE); d1["coincidence_detector"] = True
        d2 = _dense_projection(intent_k[k], suc_k[k], W_SYN, WS_LOOP_GATE); d2["coincidence_detector"] = True
        union[f"bel2suc_{k}"] = d1
        union[f"itn2suc_{k}"] = d2

    inh = list(rm.inhibitory_indices("utter_fs"))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    if not oracle:
        bridge.set_plasticity_gate(SPEAK_GATE, 1.0)
        bridge.set_plasticity_gate(CRIT_GATE, 1.0)

    idx_dev = {"intent": {k: xp.asarray(intent_k[k]) for k in range(K)},
               "utter": {k: xp.asarray(utter_k[k]) for k in range(K)},
               "crit": {k: xp.asarray(crit_k[k]) for k in range(K)},
               "belief": {k: xp.asarray(belief_k[k]) for k in range(K)},
               "suc_all": xp.asarray(suc)}

    diag = {"homeo": bool(homeo), "adapt_rate": float(adapt_rate), "ema_alpha": float(ema_alpha),
            "expose_steps": int(expose_steps), "expose_drive": float(expose_drive), "freeze_after": bool(freeze_after)}

    comp_mask = None
    if homeo:
        # boolean mask over the competing readout neurons (utter + crit) for measuring adaptation
        comp = np.concatenate([utter, crit])
        comp_mask = xp.asarray(comp)
        # ── EXPOSURE: freeze actor/critic plasticity, cycle intent SYMMETRICALLY, let the homeostat equalize ──
        if not oracle:
            bridge.set_plasticity_gate(SPEAK_GATE, 0.0)
            bridge.set_plasticity_gate(CRIT_GATE, 0.0)
        bridge.core_config.current_reward_signal = 0.0
        thr0 = to_host(bridge.cp_neuron_firing_thresholds[comp_mask]).copy()
        for s in range(expose_steps):
            t = s % K                                       # cycle intents -> each assembly is 'aligned' equally
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[idx_dev["intent"][t]] = xp.float32(expose_drive)
            bridge._run_one_simulation_step()
        thr1 = to_host(bridge.cp_neuron_firing_thresholds[comp_mask]).copy()
        ema1 = to_host(bridge.cp_neuron_activity_ema[comp_mask]).copy()
        diag["thr_before"] = {"mean": float(thr0.mean()), "std": float(thr0.std())}
        diag["thr_after"] = {"mean": float(thr1.mean()), "std": float(thr1.std())}
        diag["ema_after"] = {"mean": float(ema1.mean()), "std": float(ema1.std()), "target": float(target_rate),
                             "frac_at_clip_max": float((thr1 >= thresh_max - 1e-4).mean()),
                             "frac_at_clip_min": float((thr1 <= -54.9999).mean())}
        diag["thresh_max"] = float(thresh_max)
        # restore actor/critic plasticity for the (subsequent) training
        if not oracle:
            bridge.set_plasticity_gate(SPEAK_GATE, 1.0)
            bridge.set_plasticity_gate(CRIT_GATE, 1.0)
        if freeze_after:
            bridge.core_config.homeostasis_threshold_adapt_rate = 0.0   # freeze thresholds for the decision
            bridge.core_config.homeostasis_ema_alpha = 0.0

    # clear any eligibility/coactivity built during exposure, then the standard quiescent settle + snapshot
    if getattr(bridge, "cp_eligibility_trace", None) is not None:
        bridge.cp_eligibility_trace[:] = 0.0
    if getattr(bridge, "cp_reward_coactivity_trace", None) is not None:
        bridge.cp_reward_coactivity_trace[:] = 0.0
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)
    return bridge, xp, idx_dev, snap, diag


# ── measurements ──────────────────────────────────────────────────────────────────────────────────────────────

def measure_noise(bridge, xp, idx, snap, n_reps=1):
    """The MECHANISTIC check: under SYMMETRIC drive (untrained/equal afferents), the K=3 utter assemblies SHOULD
    have equal mean rates; any spread is intrinsic per-neuron heterogeneity. Measure, per intent, the assembly
    mean rates; report the cross-assembly CV (std/mean) averaged over intents = the heterogeneity noise the winner
    must overcome. Also the per-neuron rate CV inside the competing populations."""
    saved = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    per_intent_cv = []
    utt_neuron_rates = []
    crit_neuron_rates = []
    for t in range(K):
        _, rates, V = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None, read_crit=True)
        m = float(np.mean(rates))
        per_intent_cv.append(float(np.std(rates) / (m + 1e-12)))
        # per-neuron rates over the read window for the competing populations (re-run a read to grab neuron-level)
    # neuron-level: one symmetric drive (intent 0), read per-neuron firing over READ_UTT
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    acc_utt = None; acc_crit = None
    for s in range(SETTLE_MS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["intent"][0]] = xp.float32(INTENT_PA)
        bridge._run_one_simulation_step()
        if s >= SETTLE_MS - READ_UTT:
            fu = np.concatenate([to_host(bridge.cp_firing_states[idx["utter"][u]]).astype(np.float64) for u in range(K)])
            fc = np.concatenate([to_host(bridge.cp_firing_states[idx["crit"][u]]).astype(np.float64) for u in range(K)])
            acc_utt = fu if acc_utt is None else acc_utt + fu
            acc_crit = fc if acc_crit is None else acc_crit + fc
    ur = acc_utt / READ_UTT
    cr = acc_crit / READ_UTT
    bridge.core_config.reward_learning_rate = saved
    return {
        "assembly_rate_cv_mean": float(np.mean(per_intent_cv)),       # cross-assembly mean-rate CV (the winner noise)
        "assembly_rate_cv_per_intent": [round(x, 4) for x in per_intent_cv],
        "utter_neuron_rate_mean": float(ur.mean()), "utter_neuron_rate_cv": float(ur.std() / (ur.mean() + 1e-12)),
        "crit_neuron_rate_mean": float(cr.mean()), "crit_neuron_rate_cv": float(cr.std() / (cr.mean() + 1e-12)),
    }


def oracle_selectivity(bridge, xp, idx, snap):
    """SELECTIVITY-COLLAPSE control: with a W_ORACLE differential afferent, does the WTA winner still track it?
    If the homeostat washed out all differential response, even a strong afferent won't move the winner -> acc
    drops from 1.0 (the current oracle value) -> selectivity collapsed."""
    choice = _readout_policy(bridge, xp, idx, snap)
    return float(np.mean([choice[t] == t for t in range(K)]))


def critic_probe_homeo(seed, bridge, xp, idx, snap, n_train=N_TRAIN):
    """The finding's metric under the homeostat: v2b-train (localize-credit + executed epsilon-greedy), then read
    actor-WTA winner vs learned critic value V(intent,u). Mirrors V2.critic_value_probe_seed's training loop
    VERBATIM (the loop hardcodes _commit_action; it does not branch on LOCALIZE_CREDIT)."""
    belief_src = _belief_sources(seed)
    aligned = _aligned_utts(belief_src)
    belief_by_u = {ui: belief_src[u] for ui, u in enumerate(V2.UTTS)}
    rng = np.random.default_rng(seed * 71 + 13)             # SAME stream as V2.critic_value_probe_seed
    for _ in range(n_train):
        t = int(rng.integers(K))
        greedy, _, V = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None, read_crit=True)
        winner = int(rng.integers(K)) if (rng.random() < EPSILON) else greedy
        _commit_action(bridge, xp, idx, snap, t, winner)
        success = _evaluate_success(bridge, xp, idx, t, belief_by_u[winner])
        _deliver_reward(bridge, xp, REWARD_GAIN * (success - float(V[winner])))
    rows, actor_hits, critic_hits = {}, 0, 0
    for t in range(K):
        _, rates, V = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None, read_crit=True)
        uw, vw = int(np.argmax(rates)), int(np.argmax(V))
        actor_hits += int(uw == aligned[t]); critic_hits += int(vw == aligned[t])
        rows[str(t)] = {"aligned": int(aligned[t]), "utter_rate": [round(float(x), 5) for x in rates],
                        "critic_V": [round(float(x), 5) for x in V], "actor_wta": uw, "critic_argmax": vw}
    return {"seed": int(seed), "per_intent": rows, "actor_wta_acc": actor_hits / K,
            "critic_argmax_acc": critic_hits / K, "chance": 1.0 / K}


# ── condition sweep ──────────────────────────────────────────────────────────────────────────────────────────
def _conditions():
    """Each: (label, homeo, adapt_rate, expose_steps, freeze_after, clip_max)."""
    return [
        ("no_homeo",              False, 0.0,    0,    True,  -30.0),  # committed NEGATIVE baseline (must reproduce)
        # default clip (-30): the homeostat is clip-starved (ema stuck ~5x target) -> cannot equalize
        ("homeo_a05_long_c30",    True,  0.05,   3000, True,  -30.0),
        # RAISED clip (+35 == vpeak): give the homeostat real headroom to reach target under the strong drive
        ("homeo_a05_long_c35",    True,  0.05,   6000, True,   35.0),
        ("homeo_a2_long_c35",     True,  0.2,    6000, True,   35.0),
        ("homeo_a2_xlong_c35",    True,  0.2,    12000,True,   35.0),
        ("homeo_a2_xlong_c35_cont", True,0.2,    12000,False,  35.0),  # NOT frozen -> collapse control
    ]


def run_mech(seeds, out_path):
    """CHEAP first pass: per condition x seed, build (+ expose), measure noise + oracle selectivity. No training."""
    conds = _conditions()
    results = {}
    for label, homeo, ar, es, fz, cm in conds:
        results[label] = {}
        for sd in seeds:
            t0 = time.time()
            b, xp, idx, snap, diag = build_speaker_bridge_homeo(sd, oracle=False, homeo=homeo, adapt_rate=ar,
                                                                expose_steps=es, freeze_after=fz, thresh_max=cm)
            noise = measure_noise(b, xp, idx, snap)
            # selectivity: rebuild in ORACLE mode with the SAME homeo/exposure (cycled -> per-intent differential preserved)
            bo, xpo, idxo, snapo, diago = build_speaker_bridge_homeo(sd, oracle=True, homeo=homeo, adapt_rate=ar,
                                                                     expose_steps=es, freeze_after=fz, thresh_max=cm)
            osel = oracle_selectivity(bo, xpo, idxo, snapo)
            results[label][str(sd)] = {"noise": noise, "oracle_selectivity_acc": osel, "diag": diag,
                                       "elapsed_s": round(time.time() - t0, 1)}
            print(f"  [{label} seed {sd}] assembly_cv={noise['assembly_rate_cv_mean']:.4f} "
                  f"utt_neuron_cv={noise['utter_neuron_rate_cv']:.4f} oracle_sel={osel:.3f} "
                  f"thr_std {diag.get('thr_before',{}).get('std','-')}->{diag.get('thr_after',{}).get('std','-')} "
                  f"ema_after={diag.get('ema_after',{}).get('mean','-')} ({results[label][str(sd)]['elapsed_s']}s)",
                  flush=True)
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"probe": "homeostat_mechanistic", "seeds": list(seeds), "results": results}, f, indent=2, default=str)
    print(f"[homeo-mech] wrote {out_path}", flush=True)


def run_sweep(seeds, out_path, conditions=None, n_train=N_TRAIN):
    """FULL pass: per condition x seed, build (+ expose), then critic_probe (trains). Emits actor/critic argmax +
    the noise metric so we can check improvement TRACKS noise dropping."""
    conds = [c for c in _conditions() if (conditions is None or c[0] in conditions)]
    results = {}
    for label, homeo, ar, es, fz, cm in conds:
        results[label] = {}
        for sd in seeds:
            t0 = time.time()
            b, xp, idx, snap, diag = build_speaker_bridge_homeo(sd, oracle=False, homeo=homeo, adapt_rate=ar,
                                                                expose_steps=es, freeze_after=fz, thresh_max=cm)
            noise = measure_noise(b, xp, idx, snap)
            probe = critic_probe_homeo(sd, b, xp, idx, snap, n_train=n_train)
            results[label][str(sd)] = {"actor_wta_acc": probe["actor_wta_acc"],
                                       "critic_argmax_acc": probe["critic_argmax_acc"],
                                       "assembly_rate_cv_mean": noise["assembly_rate_cv_mean"],
                                       "utter_neuron_rate_cv": noise["utter_neuron_rate_cv"],
                                       "per_intent": probe["per_intent"], "diag": diag,
                                       "elapsed_s": round(time.time() - t0, 1)}
            print(f"  [{label} seed {sd}] actor={probe['actor_wta_acc']:.3f} critic={probe['critic_argmax_acc']:.3f} "
                  f"assembly_cv={noise['assembly_rate_cv_mean']:.4f} ({results[label][str(sd)]['elapsed_s']}s)",
                  flush=True)
        # per-condition means
        accs = [results[label][str(sd)] for sd in seeds]
        results[label]["_mean"] = {
            "actor_wta_acc": float(np.mean([a["actor_wta_acc"] for a in accs])),
            "critic_argmax_acc": float(np.mean([a["critic_argmax_acc"] for a in accs])),
            "assembly_rate_cv_mean": float(np.mean([a["assembly_rate_cv_mean"] for a in accs])),
        }
        print(f"  == [{label}] MEAN actor={results[label]['_mean']['actor_wta_acc']:.3f} "
              f"critic={results[label]['_mean']['critic_argmax_acc']:.3f} "
              f"assembly_cv={results[label]['_mean']['assembly_rate_cv_mean']:.4f}", flush=True)
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"probe": "homeostat_critic_sweep", "seeds": list(seeds), "n_train": n_train,
                   "chance": 1.0 / K, "results": results}, f, indent=2, default=str)
    print(f"[homeo-sweep] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 44, 100])
    ap.add_argument("--mech", action="store_true", help="cheap mechanistic+selectivity probe (no training)")
    ap.add_argument("--sweep", action="store_true", help="full critic-value sweep (trains)")
    ap.add_argument("--conditions", type=str, nargs="+", default=None, help="subset of condition labels for --sweep")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--json", type=str, default="/tmp/homeo.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)
    if args.mech:
        run_mech(args.seeds, args.json)
    elif args.sweep:
        run_sweep(args.seeds, args.json, conditions=args.conditions, n_train=args.n_train)
    else:
        print("pass --mech or --sweep", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
