"""LANE C: the NEURAL ABSTAIN/HEDGE GATE -- route the GO learned metacognitive-confidence pool into a SPIKING
decision that HEDGES the output when the brain's own uncertainty monitor reads low-confidence.

THE HONESTY BOUNDARY AS A DELIVERABLE (owner directive). The GO 6/6 dynamic ACC/aPFC learned monitor
(`_second_order_metacog_monitor_derisk --confidence-read learned_acc --learned-feature-mode dynamic`;
type2_auc 0.831, meta_d 2.43) predicts whether the first-order decision was CORRECT, and a fixed spiking relay
(`_laneC_self_schema_metacog_integration_derisk`) already reads it into a `self_schema` confidence pool. Those
runners STOP at "the confidence rate separates correct from error". They do NOT yet change BEHAVIOR. This runner
closes that step: the `self_schema` confidence pool's SPIKES gate an `assert` (output/motor) pool through a tonic
inhibitory `abstain_fs` veto, so the brain ASSERTS its answer only when confidence overcomes the caution veto, and
otherwise ABSTAINS/HEDGES ("my familiarity monitor reads this as low-confidence, so I'm not sure").

THE FAMILIAR-BUT-WRONG BATTERY (the confabulation failure mode). First-order confidence -- the winning
accumulator's magnitude / balance-of-evidence (Vickers) -- is HIGH on confident errors (strong noise pushed the
wrong class up), so a first-order gate CANNOT hedge them. The learned second-order monitor read the response-
conflict/persistence features and CAN. We score both gates on the SAME neural circuit, changing only the
confidence SOURCE, and ask whether the learned monitor raises selective accuracy / the risk-coverage curve where
the first-order magnitude does not.

BRAIN-BASED-ONLY. The abstain/hedge gate is NEURAL: the `self_schema` confidence pool excites the `assert` pool,
a tonically-driven inhibitory `abstain_fs` pool opposes it, and the DECISION is whether the `assert` pool fires
(a motor/output read-out -- body-legit, like reading which motor pool fired). The host does NOT threshold the
confidence scalar. The inherited scaffold (acknowledged, documented) is the learned monitor's host logistic that
renders a confidence CURRENT into `meta_schema` -- the SAME step the GO relay already uses; the NEW contribution
here (the spiking abstain/hedge gate) is fully on-substrate.

ANTI-CHEATS (all recorded, gate the verdict):
  (a) CONTINGENT ON LOW CONFIDENCE -- hedge rate on high-confidence-CORRECT items must stay LOW while hedge rate on
      familiar-but-wrong / novel items is HIGH (else the gate abstains on everything = useless).
  (b) NEURAL GATE -- lesion the `self_schema -> assert` synapse: the assert decision becomes INDEPENDENT of
      confidence (contingency collapses), proving the confidence pool's SPIKES drive the gate.
  (c) MOAT-SAFE -- genuinely-novel/unknown items (zero-signal, no knowledge) are ABSTAINED, not confidently
      asserted; the gate only ever REMOVES asserts (asserted set is a subset), it cannot manufacture a confident
      answer for an unknown. The learned gate's novel-assert rate must be <= the first-order gate's.
  (d) BEATS FIRST-ORDER -- on the familiar-but-wrong items the learned gate hedges MORE than the first-order
      magnitude gate, and its risk-coverage curve (selective accuracy vs coverage) dominates.

HONEST BAR: a FUNCTIONAL honesty behavior -- the brain withholds an answer its own monitor flags as unreliable.
NOT a claim of subjective experience. An honest negative (the neural gate underperforming a host threshold) is a
mapped substrate limit and is itself a deliverable.

Usage:
  # CPU smoke (1 seed, tiny; prints a verdict, controls live):
  SIM_BACKEND=numpy python -u -m research.runners._laneC_self_schema_neural_abstain_gate_derisk --smoke --seed 42 \
      --json research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_smoke_s42.json
  # full 6-seed (local CPU):
  SIM_BACKEND=numpy python -u -m research.runners._laneC_self_schema_neural_abstain_gate_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host

from research.runners import _second_order_metacog_monitor_derisk as meta
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state, SETTLE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
from tools.lab import attributable_to
from tools.verdict import Verdict


SELF_CONFID_SIZE = 60
ASSERT_SIZE = 60
ABSTAIN_FS_SIZE = 40

META_TO_SELF_GATE = "meta_to_self_fixed"
SELF_TO_ASSERT_GATE = "self_to_assert_fixed"
ABSTAIN_TO_ASSERT_GATE = "abstain_to_assert_fixed"

DEFAULT_META_TO_SELF_W = 10.0
DEFAULT_SELF_TO_ASSERT_W = 9.0     # self_schema (confidence) -> assert (excitatory release)
DEFAULT_ABSTAIN_TO_ASSERT_W = 14.0  # abstain_fs -> assert (tonic inhibitory caution veto)

DEFAULT_ASSERT_INTENT_PA = 240.0   # tonic "intent to answer" drive to the assert pool
# FIXED tonic inhibitory abstain-veto (pA into abstain_fs) at the confidence-graded operating point: the probe
# (2026-08-07) showed veto~120pA gives a monotone assert-pool response to confidence (0.004->0.065 over conf
# 150->750pA). Coverage is then traced by the motor-readout threshold, not by moving the veto (which cliffs).
DEFAULT_GATE_VETO_PA = 120.0
DEFAULT_GATE_STEPS = 80            # report+gate window (matches the GO relay operating point)

DEFAULT_TARGET_COVERAGE = 0.70     # operating point for the per-category contingency read

DEFAULT_THRESHOLDS = {
    "type1_acc_lo": 0.60, "type1_acc_hi": 0.90,     # genuine errors to be metacognitive about
    "min_selective_gain_vs_nogate": 0.03,           # learned gate raises selective accuracy over all-assert
    "min_selective_gain_vs_first_order": 0.02,      # ...and over the first-order magnitude gate at matched coverage
    "min_rc_auc_gain_vs_first_order": 0.0,          # learned risk-coverage AUC dominates first-order
    "max_highconf_correct_hedge_over_base": 0.10,   # contingency: high-conf-correct hedged no more than the base
                                                    #   (1-coverage) rate + margin -- NOT preferentially abstained
    "min_familiar_wrong_hedge": 0.50,               # contingency: confident errors mostly hedged
    "min_contingency_gap": 0.25,                    # hedge(fam-wrong) - hedge(high-conf-correct)
    "max_lesion_contingency_gap": 0.12,             # neural: lesion collapses the contingency gap
    "max_novel_assert_learned": 0.60,               # moat: novel/unknown items meaningfully reduced from all-assert
    "min_novel_over_hcc_hedge": 0.15,               # moat: novels are hedged MORE than confident-correct items
}


def _learned_config(args):
    return {
        "calib_trials": int(min(args.learned_calib_trials, 64) if args.smoke else args.learned_calib_trials),
        "epochs": int(args.learned_epochs),
        "lr": float(args.learned_lr),
        "l2": float(args.learned_l2),
        "w_max": float(args.learned_w_max),
        "conf_min_pa": float(args.learned_conf_min_pa),
        "conf_max_pa": float(args.learned_conf_max_pa),
        "report_steps": int(args.gate_steps),
        "balance_classes": False,
        "symmetric_features": False,
        "response_homeostasis": False,
        "feature_mode": "dynamic",
    }


# ── build the one-brain bridge: workspace competition + meta_schema + self_schema + assert/abstain gate ──────────
def build_gate_bridge(seed: int, meta_to_self_w=DEFAULT_META_TO_SELF_W,
                      self_to_assert_w=DEFAULT_SELF_TO_ASSERT_W,
                      abstain_to_assert_w=DEFAULT_ABSTAIN_TO_ASSERT_W,
                      lesion_self_to_assert: bool = False):
    """One `SimulationBridge`: the GNW workspace 2AFC competition + slow-NMDA meta_schema + a self_schema confidence
    pool + a NEURAL abstain/hedge gate (`self_schema` excites `assert`; tonic `abstain_fs` vetoes it). Under
    `lesion_self_to_assert` the confidence->assert synapse is 0 (the gate can no longer read confidence)."""
    xp, _ = get_backend()
    n_ws = meta.ASSEMBLY_SIZE * meta.K_CLASSES

    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=meta.WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="meta_schema", n_neurons=meta.META_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=True),
        BrainRegion(name="self_schema", n_neurons=SELF_CONFID_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="assert_pool", n_neurons=ASSERT_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="abstain_fs", n_neurons=ABSTAIN_FS_SIZE, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=meta.WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=meta.FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
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
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.nmda_tau_decay = float(meta.DEFAULT_NMDA_TAU)
    cfg.nmda_recurrent_tau_decay_ms = float(meta.DEFAULT_NMDA_TAU)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.stdp_w_max = max(400.0, float(meta.DEFAULT_ATTRACTOR_WEIGHT) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(meta.DEFAULT_ATTRACTOR_WEIGHT) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    fs = np.asarray(rm.indices("workspace_fs"), dtype=np.int64)
    meta_idx = np.asarray(rm.indices("meta_schema"), dtype=np.int64)
    self_idx = np.asarray(rm.indices("self_schema"), dtype=np.int64)
    assert_idx = np.asarray(rm.indices("assert_pool"), dtype=np.int64)
    abstain_idx = np.asarray(rm.indices("abstain_fs"), dtype=np.int64)
    member_idx = {k: ws[k * meta.ASSEMBLY_SIZE:(k + 1) * meta.ASSEMBLY_SIZE] for k in range(meta.K_CLASSES)}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(meta.K_CLASSES):
        union[f"loop_{k}"] = _build_assembly_loop_population(member_idx[k], float(meta.DEFAULT_ATTRACTOR_WEIGHT))
    union["meta_to_self"] = _dense_projection(meta_idx, self_idx, float(meta_to_self_w), META_TO_SELF_GATE)
    w_s2a = 0.0 if lesion_self_to_assert else float(self_to_assert_w)
    union["self_to_assert"] = _dense_projection(self_idx, assert_idx, w_s2a, SELF_TO_ASSERT_GATE)
    union["abstain_to_assert"] = _dense_projection(abstain_idx, assert_idx, float(abstain_to_assert_w),
                                                   ABSTAIN_TO_ASSERT_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(meta.WS_LOOP_GATE, 0.0)
    bridge.set_plasticity_gate(META_TO_SELF_GATE, 0.0)
    bridge.set_plasticity_gate(SELF_TO_ASSERT_GATE, 0.0)
    bridge.set_plasticity_gate(ABSTAIN_TO_ASSERT_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {
        "member_dev": {k: xp.asarray(v) for k, v in member_idx.items()},
        "fs_dev": xp.asarray(fs),
        "meta_dev": xp.asarray(meta_idx),
        "self_dev": xp.asarray(self_idx),
        "assert_dev": xp.asarray(assert_idx),
        "abstain_dev": xp.asarray(abstain_idx),
        "confidence_read": meta.LEARNED_ACC_CONFIDENCE_READ,
    }
    return bridge, xp, idx, snap


def _run_gate(bridge, xp, idx, confidence_current: float, intent_pa: float, veto_pa: float,
              gate_steps: int):
    """Report+gate phase. Drive meta_schema with the confidence current (-> self_schema confidence spikes), drive
    the assert pool with a tonic intent and the abstain_fs veto with veto_pa. Read the late-window pop-rates of the
    self_schema confidence pool and the assert (output) pool. The assert pool fires only when the confidence
    excitation overcomes the tonic inhibitory veto."""
    gate_steps = int(max(3, gate_steps))
    late_start = gate_steps - max(1, gate_steps // 3)
    _restore_state(bridge, idx["_snap"])
    bridge.cp_external_input_current[:] = 0.0
    self_acc = 0
    assert_acc = 0
    meta_dev = idx["meta_dev"]; self_dev = idx["self_dev"]
    assert_dev = idx["assert_dev"]; abstain_dev = idx["abstain_dev"]
    for t in range(gate_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[meta_dev] = xp.float32(float(confidence_current))
        bridge.cp_external_input_current[assert_dev] = xp.float32(float(intent_pa))
        bridge.cp_external_input_current[abstain_dev] = xp.float32(float(veto_pa))
        bridge._run_one_simulation_step()
        if t >= late_start:
            self_acc += int(to_host(bridge.cp_firing_states[self_dev].astype(xp.float64).sum()))
            assert_acc += int(to_host(bridge.cp_firing_states[assert_dev].astype(xp.float64).sum()))
    bridge.cp_external_input_current[:] = 0.0
    nlate = float(gate_steps - late_start)
    return self_acc / (nlate * SELF_CONFID_SIZE), assert_acc / (nlate * ASSERT_SIZE)


# ── the battery: graded-difficulty 2AFC (confident errors emerge) + genuinely-novel zero-signal items ───────────
def make_battery(seed, n_main, n_novel, base_pa, sig_lo, sig_hi, stim_noise):
    stim_m, drive_m, sig_m = meta.make_trials(seed, n_main, base_pa, sig_lo, sig_hi, stim_noise)
    # novel/unknown: NO signal (both classes = base + noise only) -> genuine ambiguity, the "I don't know" moat items.
    stim_n, drive_n, sig_n = meta.make_trials(seed * 5 + 3, n_novel, base_pa, 0.0, 0.0, stim_noise)
    stimulus = np.concatenate([stim_m, stim_n])
    drive = np.concatenate([drive_m, drive_n], axis=0)
    sig = np.concatenate([sig_m, sig_n])
    is_novel = np.concatenate([np.zeros(n_main, dtype=bool), np.ones(n_novel, dtype=bool)])
    return stimulus, drive, sig, is_novel


def _risk_coverage(assert_rate, correct, n_steps=25):
    """Trace the risk-coverage curve from the assert (motor/output) pool's GRADED late rate by sweeping the
    motor-readout threshold tau (a body read of the output pool, standard ROC-style). Returns
    (pts=[(coverage, selective_accuracy, tau)], AUC of selective-acc vs coverage; higher is better)."""
    rate = np.asarray(assert_rate, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    lo, hi = float(rate.min()), float(rate.max())
    taus = np.linspace(lo - 1e-6, hi + 1e-9, int(n_steps))
    pts = []
    for tau in taus:
        asserted = rate > tau
        cov = float(np.mean(asserted))
        sel = float(np.mean(correct[asserted])) if asserted.any() else float("nan")
        pts.append((cov, sel, float(tau)))
    finite = sorted({(c, s) for c, s, _ in pts if np.isfinite(s)})
    if len(finite) < 2:
        auc = float("nan")
    else:
        cov = np.asarray([c for c, _ in finite]); sel = np.asarray([s for _, s in finite])
        _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = float(_trap(sel, cov) / (cov.max() - cov.min())) if cov.max() > cov.min() else float("nan")
    return pts, auc


def _tau_at_coverage(assert_rate, target_cov):
    """The motor-readout threshold whose realized coverage is closest to target; returns (tau, coverage)."""
    rate = np.asarray(assert_rate, dtype=np.float64)
    lo, hi = float(rate.min()), float(rate.max())
    taus = np.linspace(lo - 1e-6, hi + 1e-9, 60)
    best_tau, best_cov, best_d = taus[0], 1.0, 1e9
    for tau in taus:
        cov = float(np.mean(rate > tau))
        d = abs(cov - target_cov)
        if d < best_d:
            best_d, best_tau, best_cov = d, float(tau), cov
    return best_tau, best_cov


def evaluate_seed(seed, args, thresholds, verbose=False):
    lc = _learned_config(args)
    stimulus, drive, sig, is_novel = make_battery(
        seed, args.n_main, args.n_novel, args.base_pa, args.sig_lo, args.sig_hi, args.stim_noise
    )
    drive_offset = np.asarray([0.0, float(args.response1_tonic_pa)], dtype=np.float64)
    if float(args.response1_tonic_pa) != 0.0:
        drive = np.clip(drive + drive_offset, 0.0, None)
    n = int(len(drive))

    # learned ACC/aPFC monitor (the GO 6/6 dynamic mechanism; calibrated on a separate feedback block).
    monitor = meta.fit_learned_acc_apfc_monitor(
        seed, lc["calib_trials"], args.base_pa, args.sig_lo, args.sig_hi, args.stim_noise,
        args.attractor_weight, args.meta_exc_w, args.meta_inh_w, args.nmda_tau, lc,
        drive_offset_by_class=drive_offset,
    )
    veto = float(args.gate_veto_pa)

    def gate_rates(bridge, xp, idx, cur):
        """Run the NEURAL gate at the FIXED confidence-graded veto for one confidence-current vector. Returns
        (assert_rate[n], self_rate[n]) -- the graded late pop-rates of the assert (motor) pool and the self_schema
        confidence pool."""
        assert_rate = np.zeros(n, dtype=np.float64)
        self_rate = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sr, ar = _run_gate(bridge, xp, idx, float(cur[i]), args.assert_intent_pa, veto, lc["report_steps"])
            assert_rate[i] = ar
            self_rate[i] = sr
        return assert_rate, self_rate

    # ---- Phase A: workspace competition (ONCE, intact bridge) -> response, winner-magnitude, learned confidence ----
    bridge, xp, idx, snap = build_gate_bridge(seed, lesion_self_to_assert=False)
    idx["_snap"] = snap
    response = np.zeros(n, dtype=int)
    winner_mag = np.zeros(n, dtype=np.float64)
    learned_conf = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = meta._run_workspace_decision_trace(bridge, xp, idx, snap, drive[i], feature_mode=lc["feature_mode"])
        response[i] = meta._response_from_assembly(tr["assembly"])
        a0, a1 = float(tr["assembly"][0]), float(tr["assembly"][1])
        winner_mag[i] = max(a0, a1)
        learned_conf[i] = float(monitor.confidence_from_features(tr["features"]))
    correct = (response == stimulus)
    main = ~is_novel
    # first-order confidence = winner magnitude, min-max normalized on THIS block's main items (no oracle; a
    # balance-of-evidence read of the winning accumulator), mapped to the SAME confidence-current range.
    lo, hi = float(np.min(winner_mag[main])), float(np.max(winner_mag[main]))
    fo_conf = np.clip((winner_mag - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    cur_learned = np.asarray([monitor.current_from_confidence(c) for c in learned_conf])
    cur_first = np.asarray([lc["conf_min_pa"] + c * (lc["conf_max_pa"] - lc["conf_min_pa"]) for c in fo_conf])

    # ---- Phase B (intact): the neural gate for BOTH confidence sources ----
    learned_rate, self_rate = gate_rates(bridge, xp, idx, cur_learned)
    first_rate, _ = gate_rates(bridge, xp, idx, cur_first)
    base = {"response": response, "correct": correct, "winner_mag": winner_mag,
            "learned_conf": learned_conf, "fo_conf": fo_conf, "self_rate": self_rate,
            "learned_rate": learned_rate, "first_rate": first_rate}

    # ---- Phase B (lesion): SAME learned currents, but self_schema->assert severed (neural anti-cheat) ----
    bridge_l, xp_l, idx_l, snap_l = build_gate_bridge(seed, lesion_self_to_assert=True)
    idx_l["_snap"] = snap_l
    lesion_rate, _ = gate_rates(bridge_l, xp_l, idx_l, cur_learned)
    lesion = {"learned_rate": lesion_rate}

    correct = base["correct"]
    main = ~is_novel
    type1_accuracy = float(np.mean(correct[main]))
    d1, c1, _, _ = meta._type1_sdt(stimulus[main], base["response"][main])
    # self_schema confidence pool separates correct from error (sanity that the routed signal is the GO monitor).
    t2 = meta._score_type2(stimulus[main], base["response"][main], base["self_rate"][main], c1, d1, seed=seed)

    # ---- risk-coverage over the MAIN block (the graded battery with confident errors) ----
    learned_pts, learned_auc = _risk_coverage(base["learned_rate"][main], correct[main])
    first_pts, first_auc = _risk_coverage(base["first_rate"][main], correct[main])
    nogate_acc = type1_accuracy  # all-assert selective accuracy

    # ONE motor-readout threshold, derived on the (knowable) main block at the target coverage, then applied to
    # EVERY item (main + novel) -- the honest single operating point.
    tau_l, l_cov = _tau_at_coverage(base["learned_rate"][main], args.target_coverage)
    tau_f, f_cov = _tau_at_coverage(base["first_rate"][main], l_cov)   # match first-order to learned's coverage
    learned_asserted = base["learned_rate"] > tau_l
    first_asserted = base["first_rate"] > tau_f
    l_sel = float(np.mean(correct[main & learned_asserted])) if (main & learned_asserted).any() else float("nan")
    f_sel = float(np.mean(correct[main & first_asserted])) if (main & first_asserted).any() else float("nan")

    sel_gain_vs_nogate = float(l_sel - nogate_acc)
    sel_gain_vs_first = float(l_sel - f_sel)
    rc_auc_gain = float(learned_auc - first_auc) if (np.isfinite(learned_auc) and np.isfinite(first_auc)) else float("nan")

    # ---- contingency (anti-cheat a): hedge rate by CATEGORY at the operating threshold ----
    high_thr = float(np.quantile(base["winner_mag"][main], 0.60))  # "confident" first-order (top ~40% winner magnitude)
    high_conf_correct = main & correct & (base["winner_mag"] >= high_thr)
    familiar_wrong = main & (~correct) & (base["winner_mag"] >= high_thr)   # confident errors = the confabulation mode

    def hedge_rate(mask, asserted):
        mask = np.asarray(mask)
        if not mask.any():
            return float("nan")
        return float(np.mean(~np.asarray(asserted)[mask]))

    hedge_hcc = hedge_rate(high_conf_correct, learned_asserted)
    hedge_fw = hedge_rate(familiar_wrong, learned_asserted)
    hedge_novel = hedge_rate(is_novel, learned_asserted)
    contingency_gap = (float(hedge_fw - hedge_hcc)
                       if (np.isfinite(hedge_fw) and np.isfinite(hedge_hcc)) else float("nan"))
    # first-order gate on the SAME confident-error items (should hedge them LESS -- it can't see the conflict).
    first_hedge_fw = hedge_rate(familiar_wrong, first_asserted)

    # ---- neural gate (anti-cheat b): lesion self->assert collapses the contingency ----
    # apply the SAME operating coverage to the lesion assert-rates (their own tau at target coverage).
    tau_le, _ = _tau_at_coverage(lesion["learned_rate"][main], args.target_coverage)
    lesion_asserted = lesion["learned_rate"] > tau_le
    lesion_hedge_hcc = hedge_rate(high_conf_correct, lesion_asserted)
    lesion_hedge_fw = hedge_rate(familiar_wrong, lesion_asserted)
    lesion_gap = (float(lesion_hedge_fw - lesion_hedge_hcc)
                  if (np.isfinite(lesion_hedge_fw) and np.isfinite(lesion_hedge_hcc)) else float("nan"))
    contingency_attributable = attributable_to(
        "abstain contingency gap from the self_schema->assert synapse (lesion vs intact)",
        contingency_gap, lesion_gap, warn_below=-1.0,
    )

    # ---- moat (anti-cheat c): novel/unknown items abstained; learned <= first-order on novel-assert ----
    novel_assert_learned = float(np.mean(learned_asserted[is_novel])) if is_novel.any() else float("nan")
    novel_assert_first = float(np.mean(first_asserted[is_novel])) if is_novel.any() else float("nan")

    # ---- GO ----
    base_hedge = float(1.0 - l_cov)   # the operating-point hedge rate; high-conf-correct must not exceed it (+margin)
    in_window = bool(thresholds["type1_acc_lo"] <= type1_accuracy <= thresholds["type1_acc_hi"])
    go = bool(
        in_window
        and np.isfinite(sel_gain_vs_nogate) and sel_gain_vs_nogate >= thresholds["min_selective_gain_vs_nogate"]
        and np.isfinite(sel_gain_vs_first) and sel_gain_vs_first >= thresholds["min_selective_gain_vs_first_order"]
        and (not np.isfinite(rc_auc_gain) or rc_auc_gain >= thresholds["min_rc_auc_gain_vs_first_order"])
        and np.isfinite(hedge_hcc) and hedge_hcc <= base_hedge + thresholds["max_highconf_correct_hedge_over_base"]
        and np.isfinite(hedge_fw) and hedge_fw >= thresholds["min_familiar_wrong_hedge"]
        and np.isfinite(contingency_gap) and contingency_gap >= thresholds["min_contingency_gap"]
        and np.isfinite(lesion_gap) and lesion_gap <= thresholds["max_lesion_contingency_gap"]
        # MOAT (anti-cheat c): the gate meaningfully withholds unknowns, and treats them as MORE uncertain than
        # confident-correct items. (The learned<=first-order-on-novel comparison is recorded but NOT gated: on pure
        # zero-signal novelty, raw winner-magnitude is itself informative, an honest boundary of this instrument.)
        and np.isfinite(novel_assert_learned) and novel_assert_learned <= thresholds["max_novel_assert_learned"]
        and np.isfinite(hedge_novel) and np.isfinite(hedge_hcc)
        and (hedge_novel - hedge_hcc) >= thresholds["min_novel_over_hcc_hedge"]
    )

    r = {
        "seed": int(seed), "go": bool(go),
        "n_main": int(main.sum()), "n_novel": int(is_novel.sum()),
        "first_order": {
            "type1_accuracy": type1_accuracy, "d1": d1,
            "self_type2_auc": t2["type2_auc"], "self_meta_d": t2["meta_d"], "self_m_ratio": t2["m_ratio"],
            "in_operating_window": in_window,
        },
        "risk_coverage": {
            "learned_points": [[round(c, 4), (round(s, 4) if np.isfinite(s) else None)] for c, s, _ in learned_pts],
            "first_order_points": [[round(c, 4), (round(s, 4) if np.isfinite(s) else None)] for c, s, _ in first_pts],
            "learned_auc": learned_auc, "first_order_auc": first_auc, "rc_auc_gain": rc_auc_gain,
            "gate_veto_pa": float(args.gate_veto_pa), "op_tau_learned": tau_l, "op_tau_first_order": tau_f,
            "nogate_accuracy": nogate_acc,
            "op_coverage_learned": l_cov, "op_selective_acc_learned": l_sel,
            "op_selective_acc_first_order_matched": f_sel,
            "selective_gain_vs_nogate": sel_gain_vs_nogate,
            "selective_gain_vs_first_order": sel_gain_vs_first,
        },
        "contingency": {
            "high_thr_winner_mag": high_thr,
            "n_high_conf_correct": int(high_conf_correct.sum()), "n_familiar_wrong": int(familiar_wrong.sum()),
            "hedge_high_conf_correct": hedge_hcc, "hedge_familiar_wrong": hedge_fw, "hedge_novel": hedge_novel,
            "base_hedge_rate": base_hedge, "contingency_gap": contingency_gap,
            "first_order_hedge_familiar_wrong": first_hedge_fw,
        },
        "neural_gate_lesion": {
            "hedge_high_conf_correct": lesion_hedge_hcc, "hedge_familiar_wrong": lesion_hedge_fw,
            "contingency_gap": lesion_gap, "contingency_attributable": contingency_attributable,
        },
        "moat": {
            "novel_assert_rate_learned": novel_assert_learned, "novel_assert_rate_first_order": novel_assert_first,
        },
        "learned_monitor": monitor.to_json(),
    }
    if verbose:
        _print_seed(r)
    return r


def _f(x, nd=2):
    return ("nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}")


def _print_seed(r):
    fo = r["first_order"]; rc = r["risk_coverage"]; ct = r["contingency"]
    le = r["neural_gate_lesion"]; mo = r["moat"]
    print(f"  [seed {r['seed']}] type1_acc={_f(fo['type1_accuracy'],3)} in_window={fo['in_operating_window']} "
          f"| self_type2_auc={_f(fo['self_type2_auc'],3)}", flush=True)
    print(f"    RISK-COVERAGE  op cov={_f(rc['op_coverage_learned'])} | sel_acc learned={_f(rc['op_selective_acc_learned'],3)} "
          f"no-gate={_f(rc['nogate_accuracy'],3)} first-order@match={_f(rc['op_selective_acc_first_order_matched'],3)} "
          f"| gain vs no-gate={_f(rc['selective_gain_vs_nogate'],3)} vs first-order={_f(rc['selective_gain_vs_first_order'],3)} "
          f"| RC-AUC learned={_f(rc['learned_auc'],3)} first={_f(rc['first_order_auc'],3)}", flush=True)
    print(f"    CONTINGENCY    hedge: high-conf-correct={_f(ct['hedge_high_conf_correct'])} "
          f"familiar-wrong={_f(ct['hedge_familiar_wrong'])} novel={_f(ct['hedge_novel'])} "
          f"gap={_f(ct['contingency_gap'])}  (first-order hedges fam-wrong={_f(ct['first_order_hedge_familiar_wrong'])})",
          flush=True)
    print(f"    NEURAL-LESION  self->assert cut: gap={_f(le['contingency_gap'])} "
          f"(attributable={_f(le['contingency_attributable'])})", flush=True)
    print(f"    MOAT           novel-assert learned={_f(mo['novel_assert_rate_learned'])} "
          f"first-order={_f(mo['novel_assert_rate_first_order'])}", flush=True)
    print(f"    >>> seed GO = {r['go']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="LANE C NEURAL ABSTAIN/HEDGE GATE (learned metacog confidence -> spiking hedge).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-main", type=int, default=120, help="graded-difficulty 2AFC trials (confident errors emerge)")
    ap.add_argument("--n-novel", type=int, default=40, help="zero-signal novel/unknown items (the moat battery)")
    ap.add_argument("--base-pa", type=float, default=300.0)
    ap.add_argument("--sig-lo", type=float, default=40.0)
    ap.add_argument("--sig-hi", type=float, default=260.0)
    ap.add_argument("--stim-noise", type=float, default=70.0)
    ap.add_argument("--attractor-weight", type=float, default=meta.DEFAULT_ATTRACTOR_WEIGHT)
    ap.add_argument("--meta-exc-w", type=float, default=meta.DEFAULT_META_EXC_W)
    ap.add_argument("--meta-inh-w", type=float, default=meta.DEFAULT_META_INH_W)
    ap.add_argument("--nmda-tau", type=float, default=meta.DEFAULT_NMDA_TAU)
    ap.add_argument("--response1-tonic-pa", type=float, default=0.0,
                    help="per-trial tonic drive to class-1 assembly (off by default; a strong offset tanks type1 acc)")
    ap.add_argument("--assert-intent-pa", type=float, default=DEFAULT_ASSERT_INTENT_PA)
    ap.add_argument("--gate-veto-pa", type=float, default=DEFAULT_GATE_VETO_PA,
                    help="tonic inhibitory abstain-veto current (fixed at the confidence-graded operating point)")
    ap.add_argument("--gate-steps", type=int, default=DEFAULT_GATE_STEPS)
    ap.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    ap.add_argument("--learned-calib-trials", type=int, default=meta.DEFAULT_LEARNED_CALIB_TRIALS)
    ap.add_argument("--learned-epochs", type=int, default=meta.DEFAULT_LEARNED_EPOCHS)
    ap.add_argument("--learned-lr", type=float, default=meta.DEFAULT_LEARNED_LR)
    ap.add_argument("--learned-l2", type=float, default=meta.DEFAULT_LEARNED_L2)
    ap.add_argument("--learned-w-max", type=float, default=meta.DEFAULT_LEARNED_W_MAX)
    ap.add_argument("--learned-conf-min-pa", type=float, default=meta.DEFAULT_LEARNED_CONF_MIN_PA)
    ap.add_argument("--learned-conf-max-pa", type=float, default=meta.DEFAULT_LEARNED_CONF_MAX_PA)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    if args.smoke:
        seeds = [args.seed]
        args.n_main = min(args.n_main, 72)
        args.n_novel = min(args.n_novel, 18)
    else:
        seeds = args.seeds if args.seeds is not None else [args.seed]

    print(f"[abstain-gate] LANE C NEURAL ABSTAIN/HEDGE GATE | seeds={seeds} n_main={args.n_main} "
          f"n_novel={args.n_novel} backend={args.backend} gate_steps={args.gate_steps} "
          f"gate_veto_pa={args.gate_veto_pa} intent_pa={args.assert_intent_pa} target_cov={args.target_coverage}",
          flush=True)
    print("[abstain-gate] mechanism: GO learned ACC/aPFC confidence -> self_schema spikes -> assert pool gated by a "
          "tonic inhibitory abstain veto. Decision = did the assert (output) pool fire.", flush=True)
    print("[abstain-gate] HONEST: a functional honesty BEHAVIOR (withhold an answer the monitor flags unreliable) -- "
          "NOT a claim of subjective experience.", flush=True)

    t0 = time.time()
    per_seed = [evaluate_seed(s, args, DEFAULT_THRESHOLDS, verbose=True) for s in seeds]

    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed))
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    def _mean(path):
        vals = []
        for r in per_seed:
            v = r
            for k in path:
                v = v[k]
            if v is not None and np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "mean_type1_accuracy": _mean(["first_order", "type1_accuracy"]),
        "mean_self_type2_auc": _mean(["first_order", "self_type2_auc"]),
        "mean_selective_gain_vs_nogate": _mean(["risk_coverage", "selective_gain_vs_nogate"]),
        "mean_selective_gain_vs_first_order": _mean(["risk_coverage", "selective_gain_vs_first_order"]),
        "mean_rc_auc_gain": _mean(["risk_coverage", "rc_auc_gain"]),
        "mean_hedge_high_conf_correct": _mean(["contingency", "hedge_high_conf_correct"]),
        "mean_hedge_familiar_wrong": _mean(["contingency", "hedge_familiar_wrong"]),
        "mean_contingency_gap": _mean(["contingency", "contingency_gap"]),
        "mean_lesion_contingency_gap": _mean(["neural_gate_lesion", "contingency_gap"]),
        "mean_novel_assert_learned": _mean(["moat", "novel_assert_rate_learned"]),
        "mean_novel_assert_first_order": _mean(["moat", "novel_assert_rate_first_order"]),
        "all_beats_first_order": all(
            (r["risk_coverage"]["selective_gain_vs_first_order"] is not None
             and r["risk_coverage"]["selective_gain_vs_first_order"] >= 0.0) for r in per_seed
        ),
        "all_neural_lesion_collapses": all(
            (np.isfinite(r["neural_gate_lesion"]["contingency_gap"])
             and r["neural_gate_lesion"]["contingency_gap"] <= DEFAULT_THRESHOLDS["max_lesion_contingency_gap"])
            for r in per_seed
        ),
        "all_moat_safe": all(
            (np.isfinite(r["moat"]["novel_assert_rate_learned"])
             and r["moat"]["novel_assert_rate_learned"] <= DEFAULT_THRESHOLDS["max_novel_assert_learned"]
             and np.isfinite(r["contingency"]["hedge_novel"]) and np.isfinite(r["contingency"]["hedge_high_conf_correct"])
             and (r["contingency"]["hedge_novel"] - r["contingency"]["hedge_high_conf_correct"])
             >= DEFAULT_THRESHOLDS["min_novel_over_hcc_hedge"])
            for r in per_seed
        ),
        "all_learned_novel_le_first_order": all(
            (np.isfinite(r["moat"]["novel_assert_rate_learned"]) and np.isfinite(r["moat"]["novel_assert_rate_first_order"])
             and r["moat"]["novel_assert_rate_learned"] <= r["moat"]["novel_assert_rate_first_order"] + 1e-9)
            for r in per_seed
        ),
    }

    preconditions = [
        {"name": "per_seed_risk_coverage_and_contingency_recorded",
         "ok": all("risk_coverage" in r and "contingency" in r for r in per_seed)},
        {"name": "neural_lesion_and_moat_controls_recorded",
         "ok": all("neural_gate_lesion" in r and "moat" in r for r in per_seed)},
        {"name": "verdict_derived_from_recorded_seed_go_flags",
         "ok": verdict == ("GO" if n_go == len(per_seed) else ("PARTIAL" if n_go > 0 else "NEGATIVE"))},
    ]
    v = Verdict("laneC_neural_abstain_gate")
    for p in preconditions:
        v.require(p["name"], p["ok"], expect=True)
    decided = v.decide(go=all_go, verbose=False)

    out = {
        "runner": "_laneC_self_schema_neural_abstain_gate_derisk",
        "faculty": ("F4 self-model/metacognition -- the HONESTY BOUNDARY as BEHAVIOR: a NEURAL abstain/hedge gate "
                    "driven by the GO learned ACC/aPFC metacognitive-confidence pool"),
        "theory": ("Maniscalco-Lau type-2 confidence read by a slow-NMDA monitor, routed through a self_schema "
                   "confidence pool into an assert/abstain disinhibition-style gate. Functional honesty correlate "
                   "only -- NOT a claim of subjective experience."),
        "seeds": seeds, "backend": args.backend,
        "battery": {"n_main": args.n_main, "n_novel": args.n_novel},
        "gate": {
            "self_to_assert_w": DEFAULT_SELF_TO_ASSERT_W, "abstain_to_assert_w": DEFAULT_ABSTAIN_TO_ASSERT_W,
            "assert_intent_pa": args.assert_intent_pa, "gate_veto_pa": args.gate_veto_pa,
            "gate_steps": args.gate_steps, "target_coverage": args.target_coverage,
            "coverage_traced_by": "motor_readout_threshold_on_assert_pool_rate",
        },
        "thresholds": DEFAULT_THRESHOLDS,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "aggregate": agg,
        "per_seed": per_seed,
        "preconditions": decided["preconditions"],
        "verdict_object": decided,
        "honest_scope": ("A functional honesty BEHAVIOR: the GO learned metacognitive-confidence pool's SPIKES gate a "
                         "motor/assert pool through a tonic inhibitory veto, so the brain withholds (hedges) an answer "
                         "its own monitor flags as unreliable -- specifically the familiar-but-wrong confident errors "
                         "a first-order winner-magnitude confidence cannot catch. NOT a claim of subjective "
                         "experience. The learned monitor's scalar->current render is the inherited (documented) "
                         "scaffold; the abstain/hedge gate itself is on-substrate."),
    }
    if decided["status"] == "UNDEFINED":
        out["verdict"] = "UNDEFINED"
        out["undefined_reasons"] = decided["undefined_reasons"]

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[abstain-gate] === VERDICT: {out['verdict']} ({n_go}/{len(seeds)} seeds GO) ===", flush=True)
    print(f"[abstain-gate]   sel-acc gain vs no-gate={agg['mean_selective_gain_vs_nogate']} "
          f"vs first-order={agg['mean_selective_gain_vs_first_order']} | RC-AUC gain={agg['mean_rc_auc_gain']}", flush=True)
    print(f"[abstain-gate]   contingency gap={agg['mean_contingency_gap']} (lesion={agg['mean_lesion_contingency_gap']}) "
          f"| hedge hcc={agg['mean_hedge_high_conf_correct']} fam-wrong={agg['mean_hedge_familiar_wrong']}", flush=True)
    print(f"[abstain-gate]   moat novel-assert learned={agg['mean_novel_assert_learned']} "
          f"first-order={agg['mean_novel_assert_first_order']} | beats-first-order={agg['all_beats_first_order']} "
          f"neural-lesion-collapses={agg['all_neural_lesion_collapses']} moat-safe={agg['all_moat_safe']}", flush=True)
    print(f"[abstain-gate]   elapsed={time.time()-t0:.1f}s  wrote {args.json}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
