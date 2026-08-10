"""EPISODIC / WTA-readout de-risk: does a region-scoped firing-rate HOMEOSTAT (Diehl-Cook adaptive threshold) +
an E%-max/gamma de-latch make a shared-inhibition WTA select on the cue-specific AFFERENT advantage instead of
latching on the assembly's INTRINSIC strength?

DESIGN: research/findings/2026-08-10-neural-WTA-afferent-winner-common-mode-removal-research-gate.md
THE DEFECT (characterized across four arcs): a shared-inhibition winner-take-all latches on the assembly with the
largest INTRINSIC strength (per-neuron threshold heterogeneity + the shared-FS first-igniter latch), NOT the
assembly the cue-specific AFFERENT points at. The pragmatics oracle probe (intent[t]->utter[t]=30, others=1) in
the LATCH regime scored ~0.167 (below 1/3 chance) -- "a forced source afferent does NOT move the winner."
THE SURPASS (tested here, RUNNER-SIDE, NO sim/ edit):
  (1) a region-scoped per-neuron firing-rate HOMEOSTAT on the competing readout assemblies (BrainRegion
      enable_homeostasis=True on `utter` only), run over an ENCODING/SETTLING exposure phase with a RAISED
      adapt-rate (the global default 0.0005 is deliberately too slow) -> raises the thresholds of always-firing
      intrinsic-strong cells (shrinks the per-assembly magnitude common-mode), PER-NEURON (what a pooled divide
      cannot do); and
  (2) an E%-max feedforward basket (de Almeida-Idiart-Lisman 2009; the built ca3_ff_basket pattern) as a
      gamma-paced de-latch replacing the hard first-igniter latch.

ARMS (all on the SAME substrate + seed; oracle afferent-swap probe; winner read NEURALLY = which utterance pool's
cells fire most in the late window):
  - REMOVER-OFF  : the current latched WTA (hard shared-FS latch, vpeak thresholds, NO homeostasis). The
                   no-remover CONTROL -- MUST stay latched/low (~0.17). A flip here = a wiring artifact -> reject.
  - EDELATCH-ONLY: E%-max feedforward basket, NO homeostasis (vpeak thresholds). Isolates the de-latch/E%-max from
                   the homeostat (the design's "homeostat OFF / gamma ON -> the structural winner returns" lesion).
  - REMOVER-ON   : E%-max basket + region-scoped homeostat (raised adapt-rate, settling exposure). The full
                   surpass. SWEPT over adapt-rate x settling-steps (the operating point IS the mechanism).

METRIC = afferent-follow rate: over the K driven intents, the fraction where the NEURAL winner (the utterance
assembly with the most late-window spikes) is the one holding the afferent advantage (intent[t]->utter[t]=30).
Also reported (graded, finer than the K-quantized follow rate): mean rank of the afferent-target assembly, the
afferent-target vs off-target rate ratio, and a SELECTIVITY-COLLAPSE flag (winner ~0 firing, or no margin -> a
too-fast homeostat drove everything to target rate and nothing wins).

SMOKE = seed 42, CPU numpy, single seed (6 seeds are the NEXT step after review). Determinism check FIRST
(build-twice cp_neuron_firing_thresholds hash via cfg.seed; refuse to run if unseeded).

Usage:
  SIM_BACKEND=numpy PYTHONPATH=$PWD python -u -m research.runners._wta_afferent_winner_homeostat_derisk \
      --seed 42 --json research/findings/raw/_wta_afferent/homeostat_smoke_s42.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
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
from sim.enums import NeuronModel, NeuronType  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _snapshot_state, _restore_state, SETTLE_STEPS,
)
from research.runners._pragmatic_success_coincidence_derisk import K, ITEM, INTENT_PA  # noqa: E402

# ── speaker geometry (LATCH regime = the documented negative; v1 leg2: UTT_FS_W=6, FS_UTT_W=16, W_ORACLE=30) ────
UTT_ITEM = 60            # utterance assembly size (== v1 leg2)
UTT_FS_N = 40            # shared FS pool
UTT_FS_W = 6.0           # utter -> FS      (LATCH)
FS_UTT_W = 16.0          # FS -> utter      (LATCH: hard mutual inhibition, single winner)
W_ORACLE = 30.0          # oracle afferent advantage: intent[t]->utter[t]=30, others=1 (the decisive anti-cheat)
W_OTHER = 1.0
# READOUT tonic = 0: the DOCUMENTED latch negative regime (v2 build, oracle probe: seed42=0.0, mean 0.222 -- the
# afferent does NOT move the latched winner). tonic=900 instead HELPS the afferent (mean 0.778), masking the defect.
UTT_DRIVE_PA = 0.0       # tonic drive to utterances DURING READOUT (0 -> the afferent alone; reproduces the latch)
EXPO_DRIVE_PA = 900.0    # uniform common drive DURING the homeostat EXPOSURE phase (exposes intrinsic excitability)

# ── E%-max feedforward basket (de-latch vehicle; ca3_ff_basket pattern _riii...:80-102) ───────────────────────
FFB_N = 40               # E%-max basket size
FFB_FF_W = 5.0           # intent -> basket (FEEDFORWARD: total afferent volley sets HOW MANY utter cells fire)
FFB_FB_W = 2.0           # utter  -> basket (WEAK FEEDBACK: E%-max within-cycle competition, WHICH fire)
FFB_INH_W = 16.0         # basket -> utter (fast-GABA inhibition; gamma-paced -> de-latch)

# ── homeostat (remover) ───────────────────────────────────────────────────────────────────────────────────────
HOMEO_TARGET_RATE = 0.02     # Diehl-Cook target (== global default)
HOMEO_EMA_ALPHA = 0.05       # RAISED from the 0.0002 global default so the EMA tracks within a short settling phase
HOMEO_THR_MIN = -55.0        # == global default bounds
HOMEO_THR_MAX = -30.0

# ── readout ───────────────────────────────────────────────────────────────────────────────────────────────────
READ_MS = 40             # settle+read window per intent
READ_LAST = 25           # count late-window spikes over the last READ_LAST steps


def _base_cfg(seed):
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)   # ⭐ seeds the substrate (NOT actual_seed_used) -- see CLAUDE.md seed trap
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process", "enable_nmda",
              "enable_reward_modulation", "enable_synaptic_scaling"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True   # per-neuron intrinsic heterogeneity (the intrinsic-strength source)
    # homeostat operating point (only ENGAGES where a region opts in via enable_homeostasis)
    cfg.homeostasis_target_rate = float(HOMEO_TARGET_RATE)
    cfg.homeostasis_ema_alpha = float(HOMEO_EMA_ALPHA)
    cfg.homeostasis_threshold_min = float(HOMEO_THR_MIN)
    cfg.homeostasis_threshold_max = float(HOMEO_THR_MAX)
    cfg.homeostasis_threshold_adapt_rate = 0.0    # set per-arm
    return cfg


def build_probe(seed, arm, adapt_rate=0.0, w_oracle=W_ORACLE):
    """ONE minimal bridge for the oracle afferent-swap probe: intent[K] --(FIXED oracle)--> utter[K]-WTA, plus the
    arm's inhibition. arm in {off, edelatch, on}:
       off      -> hard shared-FS latch, vpeak thresholds, NO homeostasis (the documented negative).
       edelatch -> E%-max feedforward basket, vpeak thresholds, NO homeostasis.
       on       -> E%-max feedforward basket + region-scoped homeostat on `utter` (adapt_rate).
    Intrinsic strength (per-neuron cfg.seed heterogeneity) is FIXED across arms/intents; the afferent advantage
    (intent[t]->utter[t]=W_ORACLE) is SWAPPED across the K intents (each intent -> a different advantaged assembly).
    """
    xp, _ = get_backend()
    use_ffb = arm in ("edelatch", "on")
    use_homeo = arm == "on"

    regions = [
        BrainRegion(name="intent", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="utter", n_neurons=UTT_ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False,
                    enable_homeostasis=bool(use_homeo)),
    ]
    pathways = []
    if use_ffb:
        regions.append(BrainRegion(name="utter_ffb", n_neurons=FFB_N, exc_fraction=0.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                                   plastic_internal=False,
                                   izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(RegionPathway(from_region="intent", to_region="utter_ffb", density=0.40,
                                      weight_mean=FFB_FF_W, weight_jitter=0.2, plastic=False))  # FEEDFORWARD
        pathways.append(RegionPathway(from_region="utter", to_region="utter_ffb", density=0.40,
                                      weight_mean=FFB_FB_W, weight_jitter=0.2, plastic=False))  # WEAK FEEDBACK (E%-max)
        pathways.append(RegionPathway(from_region="utter_ffb", to_region="utter", density=1.0,
                                      weight_mean=FFB_INH_W, weight_jitter=0.2, plastic=False))  # gamma inhibition
        inh_region = "utter_ffb"
    else:
        regions.append(BrainRegion(name="utter_fs", n_neurons=UTT_FS_N, exc_fraction=0.0, internal_density=0.0,
                                   enable_nmda=False))
        pathways.append(RegionPathway(from_region="utter", to_region="utter_fs", density=0.6,
                                      weight_mean=UTT_FS_W, weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region="utter_fs", to_region="utter", density=0.6,
                                      weight_mean=FS_UTT_W, weight_jitter=0.0, plastic=False))
        inh_region = "utter_fs"

    cfg = _base_cfg(seed)
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    if use_homeo:
        cfg.homeostasis_threshold_adapt_rate = float(adapt_rate)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    rm = bridge.region_manager
    intent = np.asarray(rm.indices("intent"), dtype=np.int64)
    utter = np.asarray(rm.indices("utter"), dtype=np.int64)
    intent_k = {k: intent[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    utter_k = {k: utter[k * UTT_ITEM:(k + 1) * UTT_ITEM] for k in range(K)}

    # scope the homeostat UPDATE to the utter assemblies only (the USE mask already restricts it, but this keeps
    # the threshold UPDATE region-scoped too -- clean per the design "region-scoped").
    if use_homeo and bridge.cp_homeostasis_neuron_mask is not None:
        bridge.cp_homeostasis_update_neuron_mask = bridge.cp_homeostasis_neuron_mask

    # FIXED oracle wiring: intent[t]->utter[t]=W_ORACLE, intent[t]->utter[u!=t]=W_OTHER (all-to-all, non-plastic).
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for t in range(K):
        for u in range(K):
            pre = np.repeat(intent_k[t], UTT_ITEM)
            post = np.tile(utter_k[u], ITEM)
            w = np.full(pre.shape[0], (float(w_oracle) if t == u else W_OTHER), dtype=np.float32)
            union[f"i2u_{t}_{u}"] = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
                                    "initial_weights": w, "plastic": False, "conn_type": "E_TO_E",
                                    "count": int(pre.size)}
    inh = list(rm.inhibitory_indices(inh_region))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {"intent": {k: xp.asarray(intent_k[k]) for k in range(K)},
           "utter": {k: xp.asarray(utter_k[k]) for k in range(K)},
           "utter_all": xp.asarray(utter)}
    return bridge, xp, idx, snap


def run_exposure(bridge, xp, idx, settling_steps):
    """ENCODING/SETTLING exposure: drive ALL utterance assemblies with a UNIFORM common current (no afferent
    advantage present) so the homeostat equalizes the per-neuron INTRINSIC excitability (raises always-firing
    intrinsic-strong cells) WITHOUT ever suppressing any specific afferent-target. Adaptation is active here."""
    for _ in range(settling_steps):
        bridge.cp_external_input_current[:] = 0.0
        for u in range(K):
            bridge.cp_external_input_current[idx["utter"][u]] = xp.float32(EXPO_DRIVE_PA)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0


def read_winner(bridge, xp, idx, snap, intent_t, freeze_adapt=True):
    """Drive intent[t] (one-hot afferent, delivered through the FIXED oracle synapses) + uniform tonic to the
    utterances; let the WTA settle; the NEURAL winner = the utterance assembly with the most late-window spikes
    (which motor pool fired = the body acting on motor output). Adaptation FROZEN during the read (a pure
    measurement of the settled thresholds; order-independent across intents)."""
    saved_rate = bridge.core_config.homeostasis_threshold_adapt_rate
    if freeze_adapt:
        bridge.core_config.homeostasis_threshold_adapt_rate = 0.0
    _restore_state(bridge, snap)          # quiescent membrane; adapted thresholds PERSIST (not in the snapshot)
    bridge.cp_external_input_current[:] = 0.0
    acc = np.zeros(K)
    for s in range(READ_MS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["intent"][intent_t]] = xp.float32(INTENT_PA)
        for u in range(K):
            bridge.cp_external_input_current[idx["utter"][u]] = xp.float32(UTT_DRIVE_PA)
        bridge._run_one_simulation_step()
        if s >= READ_MS - READ_LAST:
            for u in range(K):
                acc[u] += float(to_host(bridge.cp_firing_states[idx["utter"][u]].astype(xp.float64).sum()))
    bridge.core_config.homeostasis_threshold_adapt_rate = saved_rate
    rates = acc / (READ_LAST * UTT_ITEM)
    return int(np.argmax(rates)), rates


def afferent_follow(bridge, xp, idx, snap):
    """Oracle afferent-swap: for each intent t the afferent advantage sits on utter[t] (swapped across the K
    intents). Winner should be t. Returns follow-rate + graded diagnostics + collapse flag."""
    winners, all_rates, target_ranks, ratios = [], [], [], []
    for t in range(K):
        w, rates = read_winner(bridge, xp, idx, snap, t)
        winners.append(w)
        all_rates.append([round(float(x), 5) for x in rates])
        order = np.argsort(-rates)                          # descending
        target_ranks.append(int(np.where(order == t)[0][0]) + 1)   # 1 = winner
        off = [rates[u] for u in range(K) if u != t]
        ratios.append(float(rates[t] / (np.mean(off) + 1e-9)))
    follow = float(np.mean([winners[t] == t for t in range(K)]))
    # selectivity collapse: winner fires ~nothing (nothing wins), OR top1-top2 margin ~0 across intents.
    top_rates = [max(r) for r in all_rates]
    margins = []
    for r in all_rates:
        srt = sorted(r, reverse=True)
        margins.append((srt[0] - srt[1]) / (srt[0] + 1e-9))
    mean_top = float(np.mean(top_rates))
    mean_margin = float(np.mean(margins))
    collapsed = bool(mean_top < 1e-3 or mean_margin < 0.02)
    return {"afferent_follow": follow, "winners": winners, "rates": all_rates,
            "mean_target_rank": float(np.mean(target_ranks)), "mean_target_ratio": float(np.mean(ratios)),
            "mean_top_rate": mean_top, "mean_margin": mean_margin, "selectivity_collapsed": collapsed}


def determinism_check(seed):
    """Build twice at cfg.seed; hash cp_neuron_firing_thresholds. Identical => the substrate is actually seeded."""
    def _hash(arm):
        b, _, _, _ = build_probe(seed, arm)
        thr = to_host(b.cp_neuron_firing_thresholds)
        return hashlib.sha1(np.ascontiguousarray(thr).tobytes()).hexdigest()
    h1, h2 = _hash("off"), _hash("off")
    return h1, h2, (h1 == h2)


def run(seed, adapt_rates, settling_steps_list, out_path):
    t0 = time.time()
    print(f"[wta-homeostat] seed={seed} K={K} ITEM={ITEM} UTT_ITEM={UTT_ITEM} W_ORACLE={W_ORACLE} "
          f"tonic={UTT_DRIVE_PA} chance={1.0/K:.3f}", flush=True)

    h1, h2, ok = determinism_check(seed)
    print(f"[wta-homeostat] DETERMINISM cp_neuron_firing_thresholds: {h1[:12]} vs {h2[:12]} -> "
          f"{'SEEDED (identical)' if ok else 'NOT SEEDED (DIFFER) -- ABORT'}", flush=True)
    if not ok:
        raise SystemExit("substrate not seeded (cp_neuron_firing_thresholds differ across builds) -- refuse to run")

    results = {"seed": int(seed), "chance": 1.0 / K, "determinism_ok": ok,
               "operating_point": {"W_ORACLE": W_ORACLE, "W_OTHER": W_OTHER, "UTT_DRIVE_PA": UTT_DRIVE_PA,
                                   "UTT_FS_W": UTT_FS_W, "FS_UTT_W": FS_UTT_W,
                                   "FFB_FF_W": FFB_FF_W, "FFB_FB_W": FFB_FB_W, "FFB_INH_W": FFB_INH_W,
                                   "HOMEO_TARGET_RATE": HOMEO_TARGET_RATE, "HOMEO_EMA_ALPHA": HOMEO_EMA_ALPHA,
                                   "HOMEO_THR_BOUNDS": [HOMEO_THR_MIN, HOMEO_THR_MAX],
                                   "READ_MS": READ_MS, "READ_LAST": READ_LAST}}

    # ── arm 1: REMOVER-OFF (no-remover control -- MUST stay latched/low) ──
    b, xp, idx, snap = build_probe(seed, "off")
    off = afferent_follow(b, xp, idx, snap)
    results["remover_off"] = off
    print(f"[wta-homeostat] REMOVER-OFF   follow={off['afferent_follow']:.3f} "
          f"tgt_rank={off['mean_target_rank']:.2f} tgt/off_ratio={off['mean_target_ratio']:.2f} "
          f"top_rate={off['mean_top_rate']:.4f} margin={off['mean_margin']:.3f} "
          f"collapsed={off['selectivity_collapsed']} winners={off['winners']}", flush=True)

    # ── arm 2: EDELATCH-ONLY (E%-max, no homeostat -> isolates the de-latch) ──
    b, xp, idx, snap = build_probe(seed, "edelatch")
    ed = afferent_follow(b, xp, idx, snap)
    results["edelatch_only"] = ed
    print(f"[wta-homeostat] EDELATCH-ONLY follow={ed['afferent_follow']:.3f} "
          f"tgt_rank={ed['mean_target_rank']:.2f} tgt/off_ratio={ed['mean_target_ratio']:.2f} "
          f"top_rate={ed['mean_top_rate']:.4f} margin={ed['mean_margin']:.3f} "
          f"collapsed={ed['selectivity_collapsed']} winners={ed['winners']}", flush=True)

    # ── arm 3: REMOVER-ON sweep (adapt-rate x settling) ──
    print(f"\n[wta-homeostat] REMOVER-ON sweep (E%-max basket + region homeostat on `utter`):", flush=True)
    print(f"{'adapt_rate':>11} {'settling':>9} {'follow_ON':>10} {'tgt_rank':>9} {'tgt/off':>8} "
          f"{'top_rate':>9} {'margin':>7} {'collapsed':>10}", flush=True)
    sweep = []
    for ar in adapt_rates:
        for ss in settling_steps_list:
            b, xp, idx, snap = build_probe(seed, "on", adapt_rate=ar)
            run_exposure(b, xp, idx, ss)
            on = afferent_follow(b, xp, idx, snap)
            thr = to_host(b.cp_neuron_firing_thresholds)
            umask = to_host(b.cp_homeostasis_neuron_mask) if b.cp_homeostasis_neuron_mask is not None else None
            uthr = thr[umask] if umask is not None else thr
            cell = {"adapt_rate": ar, "settling_steps": ss, **on,
                    "utter_thr_mean": float(np.mean(uthr)), "utter_thr_std": float(np.std(uthr))}
            sweep.append(cell)
            print(f"{ar:>11} {ss:>9} {on['afferent_follow']:>10.3f} {on['mean_target_rank']:>9.2f} "
                  f"{on['mean_target_ratio']:>8.2f} {on['mean_top_rate']:>9.4f} {on['mean_margin']:>7.3f} "
                  f"{str(on['selectivity_collapsed']):>10}", flush=True)
    results["remover_on_sweep"] = sweep

    # ── DECISIVE disambiguator: afferent-magnitude CROSSOVER (design anti-cheat) ──
    # The adapt-rate axis above shows whether the homeostatic ADAPTATION moves the winner. This axis asks the
    # cosmetic-vs-real question: sweep the afferent advantage DOWN from 30x. If REMOVER-ON follows only when the
    # afferent is HUGE (needs-huge-afferent), the "win" is cosmetic (a low-threshold WTA + a swamping afferent),
    # NOT common-mode removal. If the homeostat lowers the CROSSOVER drive vs OFF/EDELATCH, it is a real remover.
    # W=1.0 is the NO-AFFERENT control (all intent->utter weights equal) -> every arm MUST fall to ~chance; a
    # REMOVER-ON follow above chance at W=1.0 is a wiring artifact (the winner set by something other than afferent).
    w_list = [30.0, 10.0, 3.0, 1.5, 1.0]
    ss_x = settling_steps_list[-1]
    print(f"\n[wta-homeostat] AFFERENT-MAGNITUDE CROSSOVER (settling={ss_x}; follow / target-rank):", flush=True)
    print(f"{'W_oracle':>9} {'OFF':>16} {'EDELATCH':>16} {'ON@0.0005':>16} {'ON@0.5':>16}", flush=True)
    crossover = []
    for w in w_list:
        row = {"w_oracle": w}
        cells = {}
        for tag, arm, ar in (("off", "off", 0.0), ("edelatch", "edelatch", 0.0),
                             ("on_slow", "on", 0.0005), ("on_fast", "on", 0.5)):
            b, xp, idx, snap = build_probe(seed, arm, adapt_rate=ar, w_oracle=w)
            if arm == "on":
                run_exposure(b, xp, idx, ss_x)
            r = afferent_follow(b, xp, idx, snap)
            cells[tag] = r
            row[tag] = {"follow": r["afferent_follow"], "tgt_rank": r["mean_target_rank"],
                        "collapsed": r["selectivity_collapsed"], "top_rate": r["mean_top_rate"]}
        crossover.append(row)
        def _c(t):
            r = cells[t]
            return f"{r['afferent_follow']:.2f}/{r['mean_target_rank']:.2f}{'*' if r['selectivity_collapsed'] else ''}"
        print(f"{w:>9} {_c('off'):>16} {_c('edelatch'):>16} {_c('on_slow'):>16} {_c('on_fast'):>16}", flush=True)
    results["afferent_crossover"] = crossover
    print("  (cells = follow/target-rank; * = selectivity collapsed; chance follow=0.33, chance rank=2.0)", flush=True)

    # ── honest read ──
    off_f = off["afferent_follow"]
    best = max((c for c in sweep if not c["selectivity_collapsed"]), key=lambda c: c["afferent_follow"], default=None)
    verdict = {}
    verdict["no_remover_control_latched"] = bool(off_f <= 1.0 / K + 1e-9)   # stays at/below chance
    verdict["edelatch_alone_moved"] = bool(ed["afferent_follow"] > off_f + 1e-9)
    if best is not None:
        verdict["best_on_follow"] = best["afferent_follow"]
        verdict["best_on_cell"] = {"adapt_rate": best["adapt_rate"], "settling_steps": best["settling_steps"]}
        verdict["homeostat_moved_over_off"] = bool(best["afferent_follow"] > off_f + 1e-9)
        verdict["homeostat_moved_over_edelatch"] = bool(best["afferent_follow"] > ed["afferent_follow"] + 1e-9)
    else:
        verdict["best_on_follow"] = None
        verdict["homeostat_moved_over_off"] = False
        verdict["all_on_cells_collapsed"] = True
    # crossover-derived interpretation (the decisive attribution)
    def _cx(w, tag):
        for r in crossover:
            if r["w_oracle"] == w:
                return r[tag]
        return None
    off_w30 = _cx(30.0, "off");  off_w10 = _cx(10.0, "off");  off_w1 = _cx(1.0, "off")
    on_slow_w30 = _cx(30.0, "on_slow"); on_fast_w30 = _cx(30.0, "on_fast")
    verdict["off_latch_follows_at_moderate_W(10x)"] = bool(off_w10 and off_w10["follow"] >= 1.0 - 1e-9)
    verdict["off_negative_is_W30_overdrive_artifact"] = bool(
        off_w30 and off_w10 and off_w30["follow"] < off_w10["follow"] - 1e-9)
    verdict["no_afferent_control_at_chance(W1)"] = bool(off_w1 and abs(off_w1["follow"] - 1.0 / K) < 1e-6)
    verdict["homeostat_adapt_rate_gradient_at_W30"] = bool(
        on_slow_w30 and on_fast_w30 and abs(on_fast_w30["follow"] - on_slow_w30["follow"]) > 1e-9)
    results["read"] = verdict
    print(f"\n[wta-homeostat] READ: no-remover control latched(<=chance)={verdict['no_remover_control_latched']} "
          f"(off@W30={off_f:.3f}); OFF-latch-follows@W10={verdict['off_latch_follows_at_moderate_W(10x)']}; "
          f"W30-negative-is-overdrive-artifact={verdict['off_negative_is_W30_overdrive_artifact']}; "
          f"no-afferent-control@chance={verdict['no_afferent_control_at_chance(W1)']}; "
          f"homeostat-adapt-gradient@W30={verdict['homeostat_adapt_rate_gradient_at_W30']}", flush=True)

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[wta-homeostat] wrote {out_path} ({results['elapsed_seconds']}s)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapt-rates", type=float, nargs="+", default=[0.0005, 0.005, 0.05, 0.5])
    ap.add_argument("--settling", type=int, nargs="+", default=[150, 600])
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wta_afferent/homeostat_smoke_s42.json")
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)
    run(args.seed, args.adapt_rates, args.settling, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
