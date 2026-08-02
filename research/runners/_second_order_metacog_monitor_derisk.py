"""LANE C: the SECOND-ORDER METACOGNITION MONITOR -- a slow-NMDA `meta_schema` region whose confidence
read-out TRACKS FIRST-ORDER CORRECTNESS, scored in the field-standard type-2 SDT / meta-d' currency.

THE OWNER REFRAME (context): a genuinely-conversing, self-aware sim-brain that can say "my familiarity monitor
reads this as novel, so I'm uncertain" -- an HONEST FUNCTIONAL read-out of its own knowing. Metacognition is the
second-order faculty: not "is the stimulus A or B" (first-order / type-1), but "was my A-vs-B decision CORRECT"
(second-order / type-2). This de-risks the roadmap's F4 self-model/metacognition faculty with the *canonical*
metacognition instrument -- Maniscalco & Lau type-2 SDT -- rather than the OLD per-regime abstention-gate ROUTING
approach (`per_regime_monitor_runner.py`), which was a decisive FAIL on 2026-05-20 (an encoding-regime mismatch,
NOT a monitor failure). This is a DIFFERENT mechanism (a downstream confidence read-out, not a routing gate), so
it does not re-run concluded-negative work.

HONEST BAR (carried into every line): this is a FUNCTIONAL metacognition CORRELATE -- a region whose graded rate
predicts whether the brain's own first-order decision was right. It is NOT, and makes no claim of, subjective
experience / phenomenal consciousness (OPEN, arguably untestable -- Chalmers hard problem). We build + measure the
correlate (meta-d' > 0, M-ratio near the metacognitive ideal, dissociable from type-1 sensitivity); we never
assert the experience.

THE MECHANISM (reuse-by-import, template = `_self_schema_region_derisk.py`, NO `sim/` edit). ONE spiking
`SimulationBridge`, three regions:
  1. `workspace` (exc, NMDA) -- K=2 first-order MEMBER assemblies (the two stimulus classes of a 2AFC
     discrimination), each a self-recurrent NMDA accumulator loop (GNW Rung-1 `_build_assembly_loop_population`) at
     a MODERATE attractor weight so the sustained rate is GRADED by the input evidence (an input-driven persistent
     accumulator, not an all-or-none latch -- graded evidence is what metacognition monitors). They share ONE
     inhibitory `workspace_fs` pool (GNW Rung-2 mutual inhibition = the competition). The first-order DECISION =
     which assembly wins the late-window rate; the WINNING MARGIN = the balance-of-evidence (Vickers) = the
     internal confidence signal.
  2. `workspace_fs` (inhib) -- the shared WTA / competition pool.
  3. `meta_schema` (exc, **slow NMDA**) -- the SECOND-ORDER MONITOR. A single pool that reads the first-order
     competition through TWO fixed projections: (a) EXCITATION from the whole workspace (the winner, whose sustained
     rate is graded by evidence quality, dominates because the loser is suppressed); (b) feed-forward INHIBITION
     from `workspace_fs` (scaled by the TOTAL competitive drive -- an ambiguous trial keeps BOTH assemblies partly
     active, driving fs harder, subtracting more from meta). The slow-NMDA time constant is what lets `meta_schema`
     INTEGRATE the settled balance-of-evidence rather than the transient onset. Its late-window RATE = the graded
     CONFIDENCE the trial's decision was correct.

TYPE-2 SDT (Maniscalco & Lau 2012), computed on the spiking read-outs (host-side scoring, NOT a host confidence
signal -- the confidence IS the `meta_schema` rate):
  * TYPE-1 (first-order sensitivity): from the 2AFC hit/false-alarm rates -> d' and criterion c.
  * TYPE-2 (metacognitive sensitivity): confidence = the `meta_schema` late rate; the model-free type-2 ROC AUC =
    P(confidence on a CORRECT trial > confidence on an ERROR trial); meta-d' = the type-1 d' an ideal SDT observer
    would need to produce the observed type-2 AUC at the fitted type-1 criterion; M-ratio = meta-d' / d' (=1 is the
    metacognitive ideal, <1 is inefficient monitoring).

WHY THIS IS NOT A TRIVIAL "inject X, read X". The confidence is READ from a genuine spiking WTA competition whose
margin the brain itself computes; the value the monitor adds is a SECOND-ORDER signal DISSOCIABLE from the
first-order decision -- lesioning the monitor's access COLLAPSES meta-d' to ~0 while leaving d' (the first-order
accuracy) UNCHANGED (the type-1/type-2 dissociation that DEFINES metacognition). The anti-cheats below license the
claim.

GO GATE (6-seed; the monitor's confidence tracks first-order correctness, in the metacognition currency):
  * type1_accuracy in the OPERATING WINDOW [0.60, 0.90] -- the first-order task has genuine ERRORS to be
    metacognitive about (not a ceiling where meta-d' is undefined, not chance).
  * type2_auc >= 0.65   (chance 0.50) -- confidence separates correct from error trials.
  * meta_d > 0 AND m_ratio >= 0.60    -- efficient monitoring, near the metacognitive ideal, in d' units.
ANTI-CHEATS (all must hold):
  (1) META-LESION / DOMAIN DISSOCIATION -- sever the monitor's ACCESS to the first-order competition (workspace->meta
      and fs->meta weights 0; the workspace WTA still runs, so the first-order BRAIN state + decision are unchanged)
      -> type2_auc -> chance AND meta_d -> ~0, WHILE type1 d' / accuracy are UNCHANGED. This is the type-1/type-2
      dissociation: metacognition is a SEPARATE downstream faculty from the first-order decision.
  (2) PERMUTED-CONFIDENCE -- pair each trial's TRUE correctness with a PERMUTED confidence (decorrelated from the
      trial) -> type2_auc -> 0.5, meta_d -> 0. Proves the monitor tracks the ACTUAL trial's evidence.
  (3) MONITOR-IS-ABOUT-CORRECTNESS-NOT-STIMULUS -- confidence separates CORRECT from ERROR even WITHIN a fixed
      stimulus class (per-class type-2 AUC > chance) -> the monitor reports HOW-SURE-I-WAS, not WHICH stimulus.

Usage:
  # CPU smoke (1 seed, tiny -- proves it runs, controls live, prints a verdict):
  python -u -m research.runners._second_order_metacog_monitor_derisk --smoke --seed 42 \
      --json research/findings/raw/_second_order_metacog_smoke.json --backend numpy
  # full 6-seed (local CPU):
  python -u -m research.runners._second_order_metacog_monitor_derisk --seeds 42 43 44 100 101 102 \
      --n-trials 160 --json research/findings/raw/_second_order_metacog_6seed.json --backend numpy
"""
from __future__ import annotations

import argparse
import json
import math
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

# reuse-by-import: the validated GNW spiking machinery + template scorers.
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    SETTLE_STEPS, DRIVE_STEPS, FREE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
from research.runners._self_schema_region_derisk import _auc, _spearman
from tools.lab import attributable_to


# ── geometry ───────────────────────────────────────────────────────────────────────────────────────────────
K_CLASSES = 2                # 2AFC first-order discrimination (chance type-1 accuracy = 0.5)
ASSEMBLY_SIZE = 80           # per-class workspace accumulator assembly
WORKSPACE_FS_N = 50          # shared inhibitory competition pool (Rung-2)
META_SIZE = 80               # the second-order monitor pool (slow-NMDA)

WS_LOOP_GATE = "workspace_loop_fixed"
META_GATE = "meta_read_fixed"
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0        # shared mutual inhibition (Rung-2 competition)

# MODERATE recurrent accumulator weight: the assembly AMPLIFIES its input into a graded sustained rate but does NOT
# saturate into an all-or-none latch, so the winning margin stays GRADED by evidence quality (what the monitor reads).
DEFAULT_ATTRACTOR_WEIGHT = 12.0
# the monitor's two read projections (fixed, non-plastic; zeroed under the meta-lesion).
DEFAULT_META_EXC_W = 1.4      # workspace member -> meta (excitation, winner-dominated balance-of-evidence)
DEFAULT_META_INH_W = 2.2      # workspace_fs -> meta (feed-forward inhibition scaled by total competition)

# slow NMDA on the monitor: NR2B-like long decay so meta INTEGRATES the settled competition (Wang persistent NMDA).
DEFAULT_NMDA_TAU = 150.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation; |err| < 1.2e-9 over the open interval)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _type1_sdt(stimulus, response):
    """Type-1 SDT for a 2AFC (class 1 = 'signal'). Returns (d1, c1, hr, far) with a log-linear (0.5-count)
    correction so extreme rates don't blow up z()."""
    stimulus = np.asarray(stimulus).astype(int)
    response = np.asarray(response).astype(int)
    n_s1 = int((stimulus == 1).sum())          # 'signal' present
    n_s0 = int((stimulus == 0).sum())          # 'noise'
    n_hit = int(((stimulus == 1) & (response == 1)).sum())
    n_fa = int(((stimulus == 0) & (response == 1)).sum())
    hr = (n_hit + 0.5) / (n_s1 + 1.0) if n_s1 > 0 else 0.5
    far = (n_fa + 0.5) / (n_s0 + 1.0) if n_s0 > 0 else 0.5
    z_hr, z_far = _norm_ppf(hr), _norm_ppf(far)
    d1 = float(z_hr - z_far)
    c1 = float(-0.5 * (z_hr + z_far))
    return d1, c1, float(hr), float(far)


def _theoretical_type2_auc(meta_d: float, c1: float, seed: int = 0, n: int = 120000) -> float:
    """The type-2 ROC AUC of an ideal equal-variance SDT observer with sensitivity `meta_d` and decision criterion
    `c1`, where confidence = |x - c1| (distance from the criterion = the balance-of-evidence). Monte-Carlo,
    deterministic given `seed`. Monotone increasing in meta_d -> invertible."""
    if meta_d <= 0.0:
        return 0.5
    rng = np.random.default_rng(seed)
    n2 = n // 2
    x_s0 = rng.normal(-meta_d / 2.0, 1.0, n2)   # stimulus class 0
    x_s1 = rng.normal(+meta_d / 2.0, 1.0, n2)   # stimulus class 1
    x = np.concatenate([x_s0, x_s1])
    stim = np.concatenate([np.zeros(n2, dtype=int), np.ones(n2, dtype=int)])
    resp = (x > c1).astype(int)
    correct = (resp == stim)
    conf = np.abs(x - c1)
    return _auc(conf, correct)


def _meta_d_from_auc(observed_auc: float, c1: float, d1: float, seed: int = 0):
    """Invert the observed model-free type-2 AUC into meta-d' (SDT d' units): the meta_d whose IDEAL type-2 AUC (at
    the fitted type-1 criterion c1) matches the observed. Grid + linear interpolation over a monotone curve.
    Returns (meta_d, m_ratio)."""
    if not np.isfinite(observed_auc) or observed_auc <= 0.5:
        return 0.0, 0.0
    grid = np.linspace(0.0, 5.0, 51)
    curve = np.array([_theoretical_type2_auc(md, c1, seed=seed + i) for i, md in enumerate(grid)])
    # enforce monotone (Monte-Carlo jitter) via cumulative max, then invert.
    curve = np.maximum.accumulate(curve)
    obs = float(np.clip(observed_auc, 0.5, curve.max()))
    meta_d = float(np.interp(obs, curve, grid))
    m_ratio = float(meta_d / d1) if abs(d1) > 1e-6 else 0.0
    return meta_d, m_ratio


def _score_type2(stimulus, response, confidence, c1, d1, seed=0):
    """The full type-2 scoring bundle on one block: model-free type-2 AUC + meta-d' + M-ratio."""
    correct = (np.asarray(response).astype(int) == np.asarray(stimulus).astype(int))
    t2_auc = _auc(np.asarray(confidence, dtype=np.float64), correct)
    meta_d, m_ratio = _meta_d_from_auc(t2_auc, c1, d1, seed=seed)
    return {"type2_auc": float(t2_auc), "meta_d": float(meta_d), "m_ratio": float(m_ratio),
            "conf_correct_mean": float(np.mean(np.asarray(confidence)[correct])) if correct.any() else None,
            "conf_error_mean": float(np.mean(np.asarray(confidence)[~correct])) if (~correct).any() else None}


# ── build the one-brain bridge (workspace competition + slow-NMDA meta monitor) ──────────────────────────────
def build_metacog_bridge(seed: int = 42, lesion_meta: bool = False,
                         attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                         meta_exc_w: float = DEFAULT_META_EXC_W, meta_inh_w: float = DEFAULT_META_INH_W,
                         nmda_tau: float = DEFAULT_NMDA_TAU):
    """One `SimulationBridge`: `workspace` (K accumulator assemblies + shared inhibition) + slow-NMDA `meta_schema`
    monitor. The monitor reads the first-order competition via a fixed workspace->meta excitation + fs->meta
    feed-forward inhibition; under `lesion_meta` both read weights are 0 (severs the monitor's ACCESS while the
    workspace competition -- the first-order decision -- runs unchanged). Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()

    n_ws = ASSEMBLY_SIZE * K_CLASSES
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="meta_schema", n_neurons=META_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=True),
    ]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
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
    # slow NMDA -> the monitor INTEGRATES the settled balance-of-evidence (long NR2B-like decay; Wang persistent NMDA).
    cfg.nmda_tau_decay = float(nmda_tau)
    cfg.nmda_recurrent_tau_decay_ms = float(nmda_tau)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    # Per-neuron HETEROGENEITY (static, deterministic given the seed) DESYNCHRONIZES the pools so their POPULATION
    # firing RATE is a smoothly-graded function of the injected current (a proper graded rate code) -- essential for
    # a graded confidence read-out. Seeded from cfg.seed so the substrate is deterministic per seed.
    cfg.enable_parameter_heterogeneity = True
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    fs = np.asarray(rm.indices("workspace_fs"), dtype=np.int64)
    meta = np.asarray(rm.indices("meta_schema"), dtype=np.int64)
    member_idx = {k: ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE] for k in range(K_CLASSES)}

    w_exc = 0.0 if lesion_meta else float(meta_exc_w)
    w_inh = 0.0 if lesion_meta else float(meta_inh_w)
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(K_CLASSES):
        union[f"loop_{k}"] = _build_assembly_loop_population(member_idx[k], float(attractor_weight))
    # the monitor's two read projections (fixed; sign of fs->meta comes from fs being in the inhibitory index set).
    union["workspace_to_meta"] = _dense_projection(ws, meta, w_exc, META_GATE)
    union["fs_to_meta"] = _dense_projection(fs, meta, w_inh, META_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    bridge.set_plasticity_gate(META_GATE, 0.0)

    # settle to a true quiescent rest, snapshot (each trial restores this).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {
        "member_dev": {k: xp.asarray(v) for k, v in member_idx.items()},
        "meta_dev": xp.asarray(meta),
    }
    return bridge, xp, idx, snap


# ── one trial: drive the 2AFC competition -> read (first-order winner, monitor confidence rate) ──────────────
def _run_trial(bridge, xp, idx, snap, drive_pa):
    """Restore quiescence -> drive each class assembly with drive_pa[k] (graded evidence: correct class stronger,
    noisy) -> the WTA competition settles into a graded winner -> free-run -> read the late-window mean rate of each
    workspace assembly (-> first-order winner) + the meta_schema monitor (-> confidence). Returns dict."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    def _set_drive():
        for k in range(K_CLASSES):
            bridge.cp_external_input_current[idx["member_dev"][k]] = xp.float32(float(drive_pa[k]))

    # (1) drive the competition (a pulse; the moderate accumulators amplify + hold).
    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        _set_drive()
        bridge._run_one_simulation_step()

    # (2) keep a small holding drive (input-driven accumulator, not a self-latch); read the late window.
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    asm_acc = {k: 0 for k in range(K_CLASSES)}
    meta_acc = 0
    meta_dev = idx["meta_dev"]
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        _set_drive()
        bridge._run_one_simulation_step()
        if t >= late_start:
            for k in range(K_CLASSES):
                asm_acc[k] += int(to_host(bridge.cp_firing_states[idx["member_dev"][k]].astype(xp.float64).sum()))
            meta_acc += int(to_host(bridge.cp_firing_states[meta_dev].astype(xp.float64).sum()))
    nlate = float(FREE_STEPS - late_start)
    return {
        "assembly": {k: asm_acc[k] / (nlate * ASSEMBLY_SIZE) for k in range(K_CLASSES)},
        "meta": meta_acc / (nlate * META_SIZE),
    }


# ── the per-trial first-order 2AFC generator (graded difficulty -> genuine errors) ──────────────────────────
def make_trials(seed, n_trials, base_pa, sig_lo, sig_hi, stim_noise):
    """Sample T independent 2AFC trials. Each: a true class s in {0,1}; a signal strength `sig` in [sig_lo, sig_hi]
    (the DIFFICULTY, sampled per trial); the correct class gets base+sig, the other gets base; both get independent
    Gaussian noise (sd `stim_noise`). On weak-signal / high-noise trials the wrong class can win -> genuine ERRORS
    for the monitor to be metacognitive about. Signal + noise are drawn INDEPENDENTLY of the confidence read-out.
    Returns (stimulus, drive_pa[T,K], sig)."""
    rng = np.random.default_rng(seed * 101 + 7)
    stimulus = rng.integers(0, K_CLASSES, size=n_trials).astype(int)
    sig = rng.uniform(float(sig_lo), float(sig_hi), size=n_trials)
    drive = np.zeros((n_trials, K_CLASSES), dtype=np.float64)
    for i in range(n_trials):
        s = int(stimulus[i])
        for k in range(K_CLASSES):
            drive[i, k] = float(base_pa) + (float(sig[i]) if k == s else 0.0) \
                + float(rng.normal(0.0, float(stim_noise)))
        drive[i] = np.clip(drive[i], 0.0, None)
    return stimulus, drive, sig


# ── evaluate one seed (intact + all anti-cheats) ─────────────────────────────────────────────────────────────
def evaluate_seed(seed, n_trials, base_pa, sig_lo, sig_hi, stim_noise,
                  attractor_weight, meta_exc_w, meta_inh_w, nmda_tau, thresholds, verbose=False):
    stimulus, drive, sig = make_trials(seed, n_trials, base_pa, sig_lo, sig_hi, stim_noise)

    def run_block(bridge, xp, idx, snap):
        """Run T trials; return (response[T], confidence[T]) -- first-order winner + monitor rate."""
        response = np.zeros(n_trials, dtype=int)
        confidence = np.zeros(n_trials, dtype=np.float64)
        for i in range(n_trials):
            r = _run_trial(bridge, xp, idx, snap, drive[i])
            asm = r["assembly"]
            response[i] = int(max(asm, key=asm.get))
            confidence[i] = r["meta"]
        return response, confidence

    # ---- INTACT: the monitor reads the real first-order competition ----
    bridge, xp, idx, snap = build_metacog_bridge(seed=seed, lesion_meta=False, attractor_weight=attractor_weight,
                                                 meta_exc_w=meta_exc_w, meta_inh_w=meta_inh_w, nmda_tau=nmda_tau)
    response, confidence = run_block(bridge, xp, idx, snap)
    type1_accuracy = float(np.mean(response == stimulus))
    d1, c1, hr, far = _type1_sdt(stimulus, response)
    t2 = _score_type2(stimulus, response, confidence, c1, d1, seed=seed)

    # confidence should be MONOTONE in the trial's true signal strength (a graded evidence read-out).
    conf_vs_sig_spearman = _spearman(sig, confidence)

    # ---- ANTI-CHEAT (3) MONITOR-IS-ABOUT-CORRECTNESS-NOT-STIMULUS: confidence separates correct/error WITHIN class ----
    correct = (response == stimulus)
    per_class_t2_auc = {}
    for k in range(K_CLASSES):
        m = stimulus == k
        if m.sum() >= 4 and correct[m].any() and (~correct[m]).any():
            per_class_t2_auc[k] = float(_auc(confidence[m], correct[m]))
        else:
            per_class_t2_auc[k] = None
    valid_pc = [v for v in per_class_t2_auc.values() if v is not None]
    within_class_ok = bool(valid_pc and min(valid_pc) >= thresholds["within_class_t2_auc"])

    # ---- ANTI-CHEAT (1) META-LESION / DOMAIN DISSOCIATION: sever the monitor's access -> meta collapses, d' intact ----
    bridge_l, xp_l, idx_l, snap_l = build_metacog_bridge(seed=seed, lesion_meta=True, attractor_weight=attractor_weight,
                                                         meta_exc_w=meta_exc_w, meta_inh_w=meta_inh_w, nmda_tau=nmda_tau)
    response_l, confidence_l = run_block(bridge_l, xp_l, idx_l, snap_l)
    type1_accuracy_l = float(np.mean(response_l == stimulus))
    d1_l, c1_l, _, _ = _type1_sdt(stimulus, response_l)
    t2_l = _score_type2(stimulus, response_l, confidence_l, c1_l, d1_l, seed=seed)
    # ATTRIBUTION (tools.lab, gap#5 lesson: measuring both arms != attributing the difference): what FRACTION of
    # the intact metacognitive sensitivity (meta-d') is NOT present once the monitor's ACCESS to the first-order
    # competition is severed (the meta-lesion). ~100% => the second-order signal is produced by the monitor, not a
    # first-order artifact. warn_below=-1.0: a small residual meta-d' under lesion should not trip a spurious flag.
    meta_d_attributable_lesion = attributable_to("meta-d' from monitor access (meta-lesion vs intact)",
                                                  t2["meta_d"], t2_l["meta_d"], warn_below=-1.0)
    lesion_meta_collapsed = bool(t2_l["type2_auc"] <= thresholds["chance_type2_auc"]
                                 and t2_l["meta_d"] <= thresholds["collapse_meta_d"])
    # DOMAIN DISSOCIATION: first-order sensitivity UNCHANGED when the second-order monitor is lesioned.
    domain_dissociation_ok = bool(abs(d1 - d1_l) <= thresholds["max_d1_shift"]
                                  and abs(type1_accuracy - type1_accuracy_l) <= thresholds["max_acc_shift"])

    # ---- ANTI-CHEAT (2) PERMUTED-CONFIDENCE: TRUE correctness paired with a permuted confidence -> collapse ----
    rng = np.random.default_rng(seed * 777 + 13)
    perm = rng.permutation(n_trials)
    conf_perm = confidence[perm]
    t2_perm = _score_type2(stimulus, response, conf_perm, c1, d1, seed=seed)
    # ATTRIBUTION: the fraction of intact meta-d' NOT present once the confidence is decorrelated from the trial
    # (permuted). ~100% => the monitor tracks the ACTUAL trial's evidence, not a stimulus/response-set artifact.
    meta_d_attributable_permuted = attributable_to("meta-d' from true trial pairing (permuted vs intact)",
                                                    t2["meta_d"], t2_perm["meta_d"], warn_below=-1.0)
    permuted_collapsed = bool(t2_perm["type2_auc"] <= thresholds["chance_type2_auc"]
                              and t2_perm["meta_d"] <= thresholds["collapse_meta_d"])

    # ---- GO (per-seed) ----
    in_window = bool(thresholds["type1_acc_lo"] <= type1_accuracy <= thresholds["type1_acc_hi"])
    go_type2 = bool(t2["type2_auc"] >= thresholds["type2_auc"])
    go_meta = bool(t2["meta_d"] > 0.0 and t2["m_ratio"] >= thresholds["m_ratio"])
    go = bool(in_window and go_type2 and go_meta
              and lesion_meta_collapsed and domain_dissociation_ok and permuted_collapsed and within_class_ok)

    r = {
        "seed": int(seed), "n_trials": int(n_trials),
        "intact": {
            "type1_accuracy": type1_accuracy, "d1": d1, "c1": c1, "hr": hr, "far": far,
            "type2_auc": t2["type2_auc"], "meta_d": t2["meta_d"], "m_ratio": t2["m_ratio"],
            "conf_correct_mean": t2["conf_correct_mean"], "conf_error_mean": t2["conf_error_mean"],
            "conf_vs_signal_spearman": conf_vs_sig_spearman, "in_operating_window": in_window,
        },
        "within_class_correctness": {
            "per_class_type2_auc": {str(k): v for k, v in per_class_t2_auc.items()},
            "min_per_class_type2_auc": (float(min(valid_pc)) if valid_pc else None),
            "within_class_ok": within_class_ok,
        },
        "meta_lesion": {
            "type1_accuracy": type1_accuracy_l, "d1": d1_l,
            "type2_auc": t2_l["type2_auc"], "meta_d": t2_l["meta_d"], "m_ratio": t2_l["m_ratio"],
            "meta_d_attributable": meta_d_attributable_lesion,
            "collapsed": lesion_meta_collapsed, "domain_dissociation_ok": domain_dissociation_ok,
        },
        "permuted_confidence": {
            "type2_auc": t2_perm["type2_auc"], "meta_d": t2_perm["meta_d"],
            "meta_d_attributable": meta_d_attributable_permuted, "collapsed": permuted_collapsed,
        },
        "go_components": {"in_operating_window": in_window, "type2": go_type2, "meta": go_meta,
                          "meta_lesion_collapses": lesion_meta_collapsed,
                          "domain_dissociation": domain_dissociation_ok,
                          "permuted_collapses": permuted_collapsed, "within_class_correctness": within_class_ok},
        "go": go,
    }
    if verbose:
        _print_seed(r)
    return r


def _print_seed(r):
    it = r["intact"]; wl = r["meta_lesion"]; pp = r["permuted_confidence"]; wc = r["within_class_correctness"]
    print(f"  [seed {r['seed']}] INTACT type1_acc={it['type1_accuracy']:.3f} (chance .5) d'={it['d1']:+.2f} "
          f"| type2_auc={it['type2_auc']:.3f} (chance .5) meta_d={it['meta_d']:.2f} M-ratio={it['m_ratio']:.2f} "
          f"| in_window={it['in_operating_window']}", flush=True)
    print(f"           conf correct/error mean = {it['conf_correct_mean']}/{it['conf_error_mean']}  "
          f"conf~signal spearman={it['conf_vs_signal_spearman']:+.2f}", flush=True)
    print(f"    WITHIN-CLASS  per-class type2_auc={wc['per_class_type2_auc']} min={wc['min_per_class_type2_auc']} "
          f"ok={wc['within_class_ok']}", flush=True)
    print(f"    META-LESION   type1_acc={wl['type1_accuracy']:.3f} d'={wl['d1']:+.2f} type2_auc={wl['type2_auc']:.3f} "
          f"meta_d={wl['meta_d']:.2f}  collapsed={wl['collapsed']}  domain_dissociation={wl['domain_dissociation_ok']}",
          flush=True)
    print(f"    PERMUTED      type2_auc={pp['type2_auc']:.3f} meta_d={pp['meta_d']:.2f}  collapsed={pp['collapsed']}",
          flush=True)
    print(f"    >>> seed GO = {r['go']}  {r['go_components']}", flush=True)


DEFAULT_THRESHOLDS = {
    "type1_acc_lo": 0.60, "type1_acc_hi": 0.90,   # operating window: genuine errors, not ceiling/chance
    "type2_auc": 0.65,                            # confidence separates correct from error
    "m_ratio": 0.60,                              # metacognitive efficiency near the ideal (=1)
    "within_class_t2_auc": 0.55,                  # confidence tracks correctness WITHIN a fixed stimulus class
    "chance_type2_auc": 0.58,                     # lesion/permuted type2 AUC must drop to ~chance (0.5) + margin
    "collapse_meta_d": 0.35,                      # lesion/permuted meta_d must collapse to ~0
    "max_d1_shift": 0.30,                         # domain dissociation: type-1 d' UNCHANGED under meta-lesion
    "max_acc_shift": 0.06,                        # domain dissociation: type-1 accuracy UNCHANGED under meta-lesion
}


def main():
    ap = argparse.ArgumentParser(description="LANE C second-order METACOGNITION MONITOR (type-2 SDT / meta-d') de-risk.")
    ap.add_argument("--seed", type=int, default=42, help="single seed (used by --smoke)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed list (overrides --seed)")
    ap.add_argument("--n-trials", type=int, default=160, help="2AFC trials per block")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed smoke (fewer trials)")
    ap.add_argument("--base-pa", type=float, default=300.0, help="baseline drive to both class assemblies")
    ap.add_argument("--sig-lo", type=float, default=40.0, help="min per-trial signal strength (hardest trials)")
    ap.add_argument("--sig-hi", type=float, default=260.0, help="max per-trial signal strength (easiest trials)")
    ap.add_argument("--stim-noise", type=float, default=70.0, help="per-trial Gaussian drive noise sd")
    ap.add_argument("--attractor-weight", type=float, default=DEFAULT_ATTRACTOR_WEIGHT,
                    help="moderate recurrent accumulator weight (graded, not latched)")
    ap.add_argument("--meta-exc-w", type=float, default=DEFAULT_META_EXC_W, help="workspace->meta excitation weight")
    ap.add_argument("--meta-inh-w", type=float, default=DEFAULT_META_INH_W, help="workspace_fs->meta inhibition weight")
    ap.add_argument("--nmda-tau", type=float, default=DEFAULT_NMDA_TAU, help="slow NMDA decay tau (ms) for the monitor")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default="research/findings/raw/_second_order_metacog_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    if args.smoke:
        seeds = [args.seed]
        n_trials = min(args.n_trials, 64)
    else:
        seeds = args.seeds if args.seeds is not None else [args.seed]
        n_trials = args.n_trials

    print(f"[metacog] LANE C second-order METACOGNITION MONITOR | seeds={seeds} n_trials={n_trials} "
          f"backend={args.backend} K={K_CLASSES} base_pa={args.base_pa} sig[{args.sig_lo},{args.sig_hi}] "
          f"stim_noise={args.stim_noise} attractor_w={args.attractor_weight} "
          f"meta_exc/inh={args.meta_exc_w}/{args.meta_inh_w} nmda_tau={args.nmda_tau}", flush=True)
    print(f"[metacog] regions: workspace({ASSEMBLY_SIZE}x{K_CLASSES} accumulators + NMDA) + shared inhibition + "
          f"slow-NMDA meta_schema({META_SIZE})", flush=True)
    print("[metacog] instrument: type-2 SDT (Maniscalco-Lau meta-d') on spiking read-outs -- confidence IS the "
          "meta_schema rate, NOT a host signal.", flush=True)
    print("[metacog] HONEST: a FUNCTIONAL metacognition correlate (the monitor rate predicts first-order "
          "correctness) -- NOT a claim of subjective experience.", flush=True)

    t0 = time.time()
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, n_trials, args.base_pa, args.sig_lo, args.sig_hi, args.stim_noise,
                                      args.attractor_weight, args.meta_exc_w, args.meta_inh_w, args.nmda_tau,
                                      DEFAULT_THRESHOLDS, verbose=True))

    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed))
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    def _mean(key_path):
        vals = []
        for r in per_seed:
            v = r
            for k in key_path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "mean_type1_accuracy": _mean(["intact", "type1_accuracy"]),
        "mean_d1": _mean(["intact", "d1"]),
        "mean_type2_auc": _mean(["intact", "type2_auc"]),
        "mean_meta_d": _mean(["intact", "meta_d"]),
        "mean_m_ratio": _mean(["intact", "m_ratio"]),
        "all_in_window": all(r["intact"]["in_operating_window"] for r in per_seed),
        "all_meta_lesion_collapse": all(r["meta_lesion"]["collapsed"] for r in per_seed),
        "all_domain_dissociation": all(r["meta_lesion"]["domain_dissociation_ok"] for r in per_seed),
        "all_permuted_collapse": all(r["permuted_confidence"]["collapsed"] for r in per_seed),
        "all_within_class_ok": all(r["within_class_correctness"]["within_class_ok"] for r in per_seed),
    }

    out = {
        "runner": "_second_order_metacog_monitor_derisk",
        "faculty": "F4 self-model/metacognition (LANE C second-order METACOGNITION MONITOR, slow-NMDA meta_schema)",
        "theory": ("Maniscalco & Lau type-2 SDT / meta-d' + balance-of-evidence confidence (Vickers) read by a "
                   "slow-NMDA downstream monitor (Wang persistent-activity NMDA) -- FUNCTIONAL correlate only"),
        "seeds": seeds, "n_trials": n_trials, "backend": args.backend,
        "thresholds": DEFAULT_THRESHOLDS,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "aggregate": agg,
        "per_seed": per_seed,
        "honest_scope": ("A functional metacognition correlate: a slow-NMDA meta_schema region reads the brain's OWN "
                         "first-order 2AFC competition (the WTA winning margin = balance-of-evidence) and emits a "
                         "graded confidence rate that predicts whether the first-order decision was CORRECT (meta-d' "
                         ">0, M-ratio near the metacognitive ideal). The monitor is DISSOCIABLE from first-order "
                         "sensitivity (meta-lesion collapses meta-d' while d' is unchanged) and collapses under a "
                         "permuted confidence. NOT a claim of subjective experience (phenomenal consciousness is "
                         "OPEN, arguably untestable)."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[metacog] === VERDICT: {verdict} ({n_go}/{len(seeds)} seeds GO) ===", flush=True)
    print(f"[metacog]   mean type1_acc={agg['mean_type1_accuracy']:.3f} d'={agg['mean_d1']:+.2f} | "
          f"mean type2_auc={agg['mean_type2_auc']:.3f} meta_d={agg['mean_meta_d']:.2f} "
          f"M-ratio={agg['mean_m_ratio']:.2f}", flush=True)
    print(f"[metacog]   anti-cheats: meta-lesion collapses={agg['all_meta_lesion_collapse']} | "
          f"domain-dissociation(d' intact)={agg['all_domain_dissociation']} | "
          f"permuted collapses={agg['all_permuted_collapse']} | "
          f"within-class correctness={agg['all_within_class_ok']} | in-window={agg['all_in_window']}", flush=True)
    print(f"[metacog]   elapsed={time.time()-t0:.1f}s  wrote {args.json}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
