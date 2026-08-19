"""GNW NEURAL SWAP INTENTION — make the swap DECISION (WHEN to swap) NEURAL: a spiking NOVELTY/MISMATCH+SALIENCE
detector, comparing the incoming PROPOSAL to the currently-held workspace content, itself INITIATES the swap when a
salient new input MISMATCHES the held thought. Its spikes drive the recurrence-depression eviction of the incumbent;
the existing neural vacancy gate then admits the newcomer. So: salient-mismatch input -> NEURAL trigger -> NEURAL
eviction -> NEURAL admission — a fully self-driven swap, NO host `if`.

WHAT THIS CLOSES. The recurrence-weaken swap (2026-08-19-gnw-recurrence-weaken-swap-GO) EVICTS neurally; the neural
vacancy gate (2026-08-19-gnw-neural-vacancy-gate-GO) ADMITS neurally. Both findings' honest limit was the SAME last
host piece: the swap TRIGGER — the "swap-now" STD boost — was a FIXED-duration top-down command (`std.set_boost(inc,
BOOST)` for `boost_steps`), a host DECISION of WHEN to swap. Per BRAIN-BASED-ONLY, that decision between sensation and
action is a shortcut. This runner replaces it with a spiking mismatch/salience population whose FIRING sets the boost:
the DECISION to swap is now made by neurons.

THE MECHANISM (novelty/mismatch comparator; disinhibition veto; NO `sim/` edit; explicit wiring; reuse-by-import the
neural-vacancy-gate substrate — workspace/norm_pool/thal/occ/gate — + the MultiLoopSTD eviction effector):
  mm_k  (per-pattern mismatch/salience detector, EXCITATORY): receives the sensory PROPOSAL for content k as an
        excitatory drive ("k is proposed, with salience == the drive strength"). A WEAK (non-salient) proposal leaves
        mm_k sub-threshold; a STRONG (salient) proposal pushes it over the ignition knee — UNLESS it is vetoed:
  pred_k (per-pattern prediction/held-content interneuron, INHIBITORY): excited by workspace pattern_k
        (`pattern_k -> pred_k`, E_TO_I). When content k is CURRENTLY HELD in the workspace, pred_k fires and TONICALLY
        INHIBITS mm_k (`pred_k -> mm_k`, I_TO_E). So mm_k fires IFF (k is proposed with salience) AND (k is NOT the
        held content) == a SALIENT MISMATCH between the proposal and the current thought. Proposing the SAME content
        that is held (a MATCH) -> pred_k vetoes mm_k -> no mismatch -> no swap. Proposing DIFFERENT content weakly
        (non-salient) -> mm_k sub-threshold -> no swap.
  TRIGGER: the mismatch population's firing RATE sets the per-spike utilization boost on the workspace recurrence
        (`eff_boost = min(MAX_BOOST, BOOST_GAIN * mm_rate_window)`) — a salience/novelty read-out that raises the
        release probability U of the held loop (the Mongillo/Tsodyks resource variable), so the CURRENTLY-FIRING
        (incumbent) coalition depletes its own loop below the sustain knee and SELF-EVICTS. It is applied to every
        coalition loop, but only the FIRING one depletes; the moment the challenger is admitted and HELD, pred fires,
        mm goes silent, the boost falls to 0 and the new thought holds (the trigger SELF-TERMINATES when the proposal
        becomes the held content). NO host read of any workspace/collapse state gates the boost — the swap DECISION is
        the mismatch population's spikes.

Once mm has triggered the eviction, the neural VACANCY GATE (occ occupancy interneurons -> gate_k disinhibition ->
gate_k -> pattern_k) admits the proposal into the freed slot, UNCHANGED from the vacancy-gate finding. So the whole
chain is neural: mismatch/salience (WHEN) -> STD eviction (OUT) -> disinhibitory vacancy gate (IN).

Biology: hippocampal/cortical NOVELTY-MISMATCH comparison — a match-mismatch detector fires to novel (unpredicted)
input and is suppressed by a matching prediction (Lisman & Grace 2005, Neuron 46:703 "hippocampal-VTA novelty loop";
Vinogradova 2001, Prog.Neurobiol 45:523, CA1/CA3 comparator; Kumaran & Maguire 2007, J.Neurosci 27:8517, hippocampal
match-mismatch). Salience/novelty drives a phasic destabilization of working memory (Dehaene & Changeux 2011, Neuron
70:200 — an ignited workspace state must be destabilizable and "spontaneously replaced by another"). Predictive-coding
mismatch = feedforward drive minus a matching top-down prediction (Bastos et al. 2012, Neuron 76:695). Disinhibitory
gating: Chevalier & Deniau 1990, TINS 13:277; Pi et al. 2013, Nature 503:521. Eviction: Mongillo, Barak & Tsodyks
2008, Science 319:1543. Corpus-first (`before_you_build.sh`) + the source check were run and logged BEFORE building.

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed) — per seed, ALL of:
  NEURAL-TRIGGERED SWAP — a SALIENT MISMATCH proposal makes mm fire -> the STD boost is set by mm's rate (no host
        `if`) -> incumbent self-evicts -> vacancy gate admits the newcomer: win_pre=A & n_pre=1, then A DROPS TO
        BASELINE (old_residual_post NOT ignited) & B ignites & win_post=B & n_post=1.
  SPECIFICITY (the crux — this is what makes it a DECISION, not an always-swap):
        - a NON-SALIENT mismatch proposal (weak drive) does NOT make mm fire -> boost stays ~0 -> the incumbent HOLDS
          (no swap).
        - a SALIENT MATCH proposal (propose the content already held) is VETOED by pred -> mm silent -> no swap ->
          the incumbent HOLDS.
        Report swap-rate salient-mismatch vs non-salient vs match (expect ~1 / ~0 / ~0).
  TRIGGER-LESION DISSOCIATION — silence the mismatch population (give mm NO proposal drive; the SAME proportional
        boost read-out is untouched) -> mm never fires -> boost stays 0 -> a salient new input does NOT trigger a swap
        -> the incumbent HOLDS. The neural signal, not a host command, decides.
  REVERSIBLE — a two-swap A->B->A on ONE continuous substrate, BOTH swaps TRIGGERED by the neural mismatch detector.
  NO-HOST-RESET / NO-HOST-TRIGGER — the swap headline is a CONTINUOUS run (host_workspace_reset_calls==0); the only
        host writes are thal tonic + the sub-threshold sensory PROPOSAL (world/body). The STD boost is set ONLY by
        `BOOST_GAIN * mm_rate` (grep-checkable: no `if collapsed` / no fixed `set_boost(inc, BOOST)` command).
  DETERMINISM — build twice at one seed -> identical seed-derived Izhikevich-param hash.

NOT-A-WALL: any residual is QUANTIFIED (does mm fire? does the challenger latch under the transient boost? swap rate)
so the next mechanism is mapped.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_swap_intention_derisk --calibrate --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_swap_intention_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_neural_swap_intention_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_swap_intention_derisk --six-seed \
      --json research/findings/raw/_gnw_neural_swap_intention_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

# reuse-by-import: the swap substrate geometry / split recurrence / dense pops / stepping / spiking reads / ignition
# constants (NO re-derivation), the neural-vacancy-gate dis-inhibitory admission, and the STD eviction effector.
from research.runners._gnw_active_overwrite_derisk import (
    _pattern_geometry, _rec_population_split,
    _ws_step, _drive, _read_private_rates, _instant_private_rate, _margin,
    N_PATTERNS, PATTERN_SIZE, WORKSPACE_N, NORM_N, THAL_N,
    W_SHARED, WS_NORM_W, NORM_WS_W, THAL_TONIC_PA, THAL_WS_W, STRONG_PA, STD_TAU_D, OU_NOISE_PA,
)
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._gnw_rung1_ignition_curve_derisk import DRIVE_STEPS, SETTLE_STEPS, WS_LOOP_GATE
from research.runners._gnw_rung2_competitive_access_derisk import _ignited, IGNITE_FRAC, SOLO_PLATEAU
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore
from research.runners._gnw_recurrence_weaken_swap_derisk import MultiLoopSTD
# reuse the exact neural vacancy-gate constants (occ/gate) so the ADMISSION half is byte-for-byte the GO finding.
from research.runners._gnw_neural_vacancy_gate_derisk import (
    OCC_N, GATE_PER, W_WS_OCC, W_OCC_GATE, W_GATE_WS,
)

IGNITE_THRESH = IGNITE_FRAC * SOLO_PLATEAU     # a swapped-OUT / vetoed coalition reads BELOW this (0.1667)

# ── the neural mismatch/salience detector (the NEW trigger; calibrated on seed 42, --calibrate; frozen) ────────────
MM_PER = 60                    # neurons per per-pattern mismatch/salience detector mm_k
W_MM_REC = 0.0                 # mm_k -> mm_k (E_TO_E): OFF. A measured sweep (w=0..32) showed within-assembly
                               # recurrence saturates the salient response (~0.17, refractory-limited) while RAISING
                               # the non-salient response — i.e. it NARROWS the salience gap; the pure feed-forward
                               # detector (w=0) gives the cleanest salient/non-salient separation (0.13 vs 0.04),
                               # so mm is a feed-forward salience/mismatch comparator (no intra-assembly attractor).
PRED_PER = 40                  # neurons per per-pattern prediction interneuron pred_k (INHIBITORY; the match veto)
W_PAT_PRED = 8.0               # pattern_k -> pred_k (E_TO_I): the held-content read (mirrors ws->occ=8)
W_PRED_MM = 20.0               # pred_k -> mm_k (I_TO_E): the MATCH veto (mirrors occ->gate=20; >40 inverts via
                               # Izhikevich depolarization-block relief, so 20 is the clean veto)
SALIENT_PA = 5000.0            # a SALIENT proposal drive into mm_k AND gate_k: fires mm_k (~0.14) when NOT vetoed
                               # (a mismatch) — the ignition-strength drive; == STRONG_PA.
NONSALIENT_PA = 600.0          # a NON-SALIENT proposal: leaves mm_k near rest (~0.03) even when NOT vetoed -> the
                               # boost stays below the collapse knee -> no trigger.

# the swap TRIGGER: the mismatch population's firing rate sets the per-spike utilization boost (the "swap-now"
# intention is now NEURAL — mm's spikes, not a fixed host schedule). eff_boost = min(MAX_BOOST, BOOST_GAIN*mm_window).
BOOST_GAIN = 1.0               # rate-to-boost gain (calibrated: a salient mm ~0.14 -> ~0.14 boost, above the ~0.10
                               # collapse knee; a non-salient mm ~0.035 -> ~0.035 boost, below the ~0.06 hold line).
MAX_BOOST = 0.16               # saturation of the salience->release-probability modulation (caps a strongly-firing
                               # mm so it does not over-deplete; a held loop collapses at boost>=~0.10).
BOOST_WINDOW = 5               # trailing window over which the mm rate is averaged (smooths the period-3 oscillation)

# swap operating point (the eviction/admission halves are the vacancy-gate GO point, reused unchanged).
ESTABLISH_PA = 8000.0          # incumbent-establishment DIRECT drive (the "before" = attending to thought A). Above
                               # STRONG_PA because the appended mm/pred/occ/gate pools shift the workspace Izhikevich
                               # params per seed (RNG-prefix quirk) — 8000x35 clears the near-threshold ignition
                               # boundary uniformly on every seed. NOT load-bearing (it only sets the "before" state).
EVICT_STEPS = 320              # FIXED window the proposal is presented (NOT vacancy-gated). The neural chain (mm ->
                               # boost -> collapse -> gate -> admit) runs inside it; a longer window than the vacancy
                               # gate's 260 because the boost ramps with mm's firing rather than stepping on instantly.
REIGNITE_HOLD = 150
W_REC = W_SHARED               # 34.0 uniform recurrence (supra-critical disjoint = the inhibition-resistant incumbent)

# timing-read thresholds (MEASUREMENT ONLY — never gate the trigger; the boost is a pure function of mm's rate).
TIMING_WINDOW = 9              # trailing window (3 ignition periods) to smooth the period-3 oscillation for timing

_RESTORE_CALLS = {"n": 0}


def _counted_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


# ── build: the neural-vacancy-gate swap substrate + the NEW mismatch/salience detector (explicit wiring; NO sim/) ──
def build(seed=42, w_rec=W_REC, heterogeneity=True, ou_noise_pA=OU_NOISE_PA,
          w_pat_pred=W_PAT_PRED, w_pred_mm=W_PRED_MM, w_mm_rec=W_MM_REC, pred_lesion=False):
    """workspace (exc, NMDA; N_PATTERNS disjoint supra-critical cliques) + norm_pool (inh) + thal (exc tonic) + occ
    (inh occupancy) + gate (K per-pattern admission relays, exc) — the EXACT neural-vacancy-gate substrate — PLUS the
    NEW mismatch trigger: mm (K per-pattern salience/mismatch detectors, exc) + pred (K per-pattern held-content
    interneurons, inh). pred_lesion=True zeroes pred->mm (the match veto is removed: mm fires even for a matching
    proposal -> used to prove the veto is what enforces match-specificity). Returns a dict of handles."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="thal", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="occ", n_neurons=OCC_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="gate", n_neurons=GATE_PER * N_PATTERNS, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="mm", n_neurons=MM_PER * N_PATTERNS, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="pred", n_neurons=PRED_PER * N_PATTERNS, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(w_rec) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(w_rec) * 4.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, PATTERN_SIZE, 0)   # overlap=0 -> disjoint supra-critical
    ws_used = np.unique(np.concatenate(patterns)).astype(np.int64)
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)
    thal_idx = np.asarray(rm.indices("thal"), dtype=np.int64)
    occ_idx = np.asarray(rm.indices("occ"), dtype=np.int64)
    gate_idx = np.asarray(rm.indices("gate"), dtype=np.int64)
    mm_idx = np.asarray(rm.indices("mm"), dtype=np.int64)
    pred_idx = np.asarray(rm.indices("pred"), dtype=np.int64)
    gate_slices = [gate_idx[k * GATE_PER:(k + 1) * GATE_PER] for k in range(N_PATTERNS)]
    mm_slices = [mm_idx[k * MM_PER:(k + 1) * MM_PER] for k in range(N_PATTERNS)]
    pred_slices = [pred_idx[k * PRED_PER:(k + 1) * PRED_PER] for k in range(N_PATTERNS)]

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    # the swap substrate (identical to build_swap_bridge's disjoint uniform-recurrence config)
    union_plan["workspace_rec"] = _rec_population_split(patterns, privates, float(w_rec), float(w_rec))
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(WS_NORM_W), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, float(NORM_WS_W), "I_TO_E")
    union_plan["thal2ws"] = _dense_pop(thal_idx, ws_used, float(THAL_WS_W), "E_TO_E")
    # the NEURAL dis-inhibitory vacancy gate (ADMISSION half — unchanged from the vacancy-gate GO)
    union_plan["ws2occ"] = _dense_pop(ws_used, occ_idx, float(W_WS_OCC), "E_TO_I")
    for k in range(N_PATTERNS):
        union_plan[f"occ2gate{k}"] = _dense_pop(occ_idx, gate_slices[k], float(W_OCC_GATE), "I_TO_E")
        union_plan[f"gate{k}2ws"] = _dense_pop(gate_slices[k], patterns[k], float(W_GATE_WS), "E_TO_E")
    # the NEW mismatch/salience TRIGGER: pred_k reads the held content, vetoes mm_k on a match
    wpm = 0.0 if pred_lesion else float(w_pred_mm)
    for k in range(N_PATTERNS):
        if float(w_mm_rec) > 0.0:
            union_plan[f"mm_rec{k}"] = _dense_pop(mm_slices[k], mm_slices[k], float(w_mm_rec), "E_TO_E")
        union_plan[f"pat2pred{k}"] = _dense_pop(patterns[k], pred_slices[k], float(w_pat_pred), "E_TO_I")
        union_plan[f"pred2mm{k}"] = _dense_pop(pred_slices[k], mm_slices[k], wpm, "I_TO_E")

    inh = list(norm_idx) + list(occ_idx) + list(pred_idx)   # occ + pred are inhibitory
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    thal_dev = xp.asarray(thal_idx)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[thal_dev] = xp.float32(THAL_TONIC_PA)
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "w_rec": float(w_rec), "w_pat_pred": float(w_pat_pred), "w_pred_mm": float(wpm),
               "w_mm_rec": float(w_mm_rec), "pred_lesion": bool(pred_lesion), "n_ws_used": int(ws_used.size),
               "mm_per": int(MM_PER), "pred_per": int(PRED_PER), "occ_n": int(occ_idx.size)}
    return {
        "bridge": bridge, "xp": xp,
        "patterns": [xp.asarray(p) for p in patterns], "privates": [xp.asarray(p) for p in privates],
        "patterns_host": [p.astype(np.int64) for p in patterns], "ws_used": ws_used,
        "thal": thal_dev, "occ": xp.asarray(occ_idx),
        "gate_slices": [xp.asarray(g) for g in gate_slices],
        "mm_slices": [xp.asarray(m) for m in mm_slices], "mm_all": xp.asarray(mm_idx),
        "pred_slices": [xp.asarray(p) for p in pred_slices],
        "snap": snap, "handles": handles,
    }


def _izh_hash(bridge):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest() if parts else ""


def _pop_rate(S, idx_dev):
    b, xp = S["bridge"], S["xp"]
    return float(to_host(b.cp_firing_states[idx_dev].astype(xp.float64).mean()))


def _set_boost_all(std, val):
    for k in range(N_PATTERNS):
        std.set_boost(k, float(val))


# ── one single-move swap whose TRIGGER is the neural mismatch detector (NO host `if` in the trigger path) ──────────
def run_intention_swap(S, std, *, incumbent=0, proposed=1, proposal_pa=SALIENT_PA, boost_gain=BOOST_GAIN,
                       evict_steps=EVICT_STEPS, reignite_hold=REIGNITE_HOLD, trigger_lesion=False, isolate=True):
    """Ignite incumbent A (hold). Present a sub-threshold sensory PROPOSAL for `proposed` (drives gate_proposed AND
    mm_proposed every step). The DECISION to swap is the mismatch detector's firing: each step
    `eff_boost = min(MAX_BOOST, boost_gain * mm_rate_window)` sets the STD boost on the workspace recurrence — nothing
    reads a workspace/collapse state. If the proposal is a SALIENT MISMATCH, mm fires -> boost -> the incumbent
    self-evicts -> the neural vacancy gate admits `proposed` -> the newcomer holds & mm self-silences (pred veto).
      trigger_lesion=True -> give mm NO proposal drive (the population is silenced; the boost read-out is UNCHANGED)
        -> mm never fires -> boost stays 0 -> a salient new input does NOT trigger a swap (the incumbent holds). The
        neural signal, not a host command, decides.
      isolate=False -> a CONTINUOUS run (0 restore calls) = the swap HEADLINE."""
    bridge, xp, thal = S["bridge"], S["xp"], S["thal"]
    patterns, privates = S["patterns"], S["privates"]
    if isolate:
        _counted_restore(bridge, S["snap"])
        std.reset()

    # (1) establish the incumbent -> it holds on its supra-critical recurrent loop (mm silent: no proposal yet).
    _drive(bridge, xp, thal, THAL_TONIC_PA, std, [(patterns[incumbent], ESTABLISH_PA)], n=DRIVE_STEPS)
    pre = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_pre, _m_pre, n_pre = _margin(pre)

    # (2) present the proposal; the mismatch detector's rate SETS the boost (the neural swap decision).
    gate_dev = S["gate_slices"][proposed]
    mm_dev = S["mm_slices"][proposed]
    mm_all = S["mm_all"]
    mm_drive = 0.0 if trigger_lesion else float(proposal_pa)
    mm_hist, a_hist, b_hist = [], [], []
    xA_min, boost_max, mm_peak = 1.0, 0.0, 0.0
    a_vacate_step, b_ignite_step, coign_steps, trigger_step = -1, -1, 0, -1
    for t in range(int(evict_steps)):
        mm_rate = _pop_rate(S, mm_all)                                  # the SALIENCE/MISMATCH read (spiking)
        mm_hist.append(mm_rate)
        mm_win = float(np.mean(mm_hist[-BOOST_WINDOW:]))
        eff_boost = min(MAX_BOOST, float(boost_gain) * mm_win)          # the NEURAL swap trigger (no host `if`)
        _set_boost_all(std, eff_boost)
        _ws_step(bridge, xp, thal, THAL_TONIC_PA, std,
                 drive_map=[(gate_dev, float(proposal_pa)), (mm_dev, mm_drive)])
        # READ-ONLY instruments (never gate the trigger):
        a_hist.append(_instant_private_rate(bridge, xp, privates, incumbent))
        b_hist.append(_instant_private_rate(bridge, xp, privates, proposed))
        aw = float(np.mean(a_hist[-TIMING_WINDOW:])); bw = float(np.mean(b_hist[-TIMING_WINDOW:]))
        xA_min = min(xA_min, std.x_mean(incumbent))
        boost_max = max(boost_max, eff_boost); mm_peak = max(mm_peak, mm_rate)
        if trigger_step < 0 and eff_boost >= 0.10:
            trigger_step = t
        if a_vacate_step < 0 and t >= TIMING_WINDOW and aw < IGNITE_THRESH:
            a_vacate_step = t
        if b_ignite_step < 0 and bw > IGNITE_THRESH and proposed != incumbent:
            b_ignite_step = t
        if proposed != incumbent and _ignited(aw) and _ignited(bw):
            coign_steps += 1
    std.clear_boost()

    # (3) identity read (free-run, no proposal -> gate closes & mm silent; the held coalition sustains on its loop).
    post = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_post, _m_post, n_post = _margin(post)
    hold = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std, n_free=int(reignite_hold))
    win_hold, _m_hold, n_hold = _margin(hold)

    old_res = float(post[incumbent]); new_rate = float(post[proposed])
    old_res_hold = float(hold[incumbent]); new_hold = float(hold[proposed])
    swapped = bool(proposed != incumbent and win_pre == incumbent and n_pre == 1 and (not _ignited(old_res))
                   and _ignited(new_rate) and win_post == proposed and n_post == 1)
    reignite_ok = bool(_ignited(new_hold) and win_hold == proposed and n_hold == 1 and (not _ignited(old_res_hold)))
    # held == the incumbent is STILL the sole ignited winner (no swap). When proposed==incumbent (a MATCH probe) the
    # "new" and "old" indices coincide, so the challenger-not-ignited clause only applies to a genuine mismatch probe.
    held = bool(win_post == incumbent and n_post == 1 and _ignited(old_res)
                and (proposed == incumbent or not _ignited(new_rate)))
    timing_ok = bool(swapped and a_vacate_step >= 0 and b_ignite_step > a_vacate_step and coign_steps == 0)
    return {
        "pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
        "hold_rates": [float(r) for r in hold],
        "winner_pre": int(win_pre), "winner_post": int(win_post), "winner_hold": int(win_hold),
        "n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post), "n_ignited_hold": int(n_hold),
        "old_residual_post": old_res, "old_residual_hold": old_res_hold,
        "new_rate_post": new_rate, "new_rate_hold": new_hold, "xA_min": float(xA_min),
        "mm_peak": float(mm_peak), "boost_max": float(boost_max), "trigger_step": int(trigger_step),
        "a_vacate_step": int(a_vacate_step), "b_ignite_step": int(b_ignite_step), "coign_steps": int(coign_steps),
        "swapped": swapped, "reignite_ok": reignite_ok, "held": held, "timing_ok": timing_ok,
        "trigger_lesion": bool(trigger_lesion), "proposed": int(proposed), "incumbent": int(incumbent),
    }


# ── two-swap reversibility: A -> B -> A on ONE continuous substrate, BOTH swaps neural-mismatch-triggered ──────────
def run_two_swap(S, std, *, a=0, b=1, proposal_pa=SALIENT_PA, boost_gain=BOOST_GAIN, evict_steps=EVICT_STEPS,
                 reignite_hold=REIGNITE_HOLD, recover_steps=None):
    _counted_restore(S["bridge"], S["snap"]); std.reset()
    if recover_steps is None:
        recover_steps = int(3 * STD_TAU_D)
    s1 = run_intention_swap(S, std, incumbent=a, proposed=b, proposal_pa=proposal_pa, boost_gain=boost_gain,
                            evict_steps=evict_steps, reignite_hold=reignite_hold, isolate=False)
    xA_after_s1 = std.x_mean(a)
    for _ in range(int(recover_steps)):
        _ws_step(S["bridge"], S["xp"], S["thal"], THAL_TONIC_PA, std)
    xA_recovered = std.x_mean(a)
    s2 = run_intention_swap(S, std, incumbent=b, proposed=a, proposal_pa=proposal_pa, boost_gain=boost_gain,
                            evict_steps=evict_steps, reignite_hold=reignite_hold, isolate=False)
    s1_evicted_A = bool(s1["swapped"])
    recovered = bool(xA_recovered > 0.85)
    s2_brought_A_back = bool(s2["winner_hold"] == a and s2["n_ignited_hold"] == 1
                             and _ignited(s2["new_rate_hold"]) and not _ignited(s2["old_residual_hold"]))
    reversible = bool(s1_evicted_A and recovered and s2_brought_A_back)
    return {"swap1": s1, "swap2": s2, "xA_after_swap1": float(xA_after_s1), "xA_recovered": float(xA_recovered),
            "recover_steps": int(recover_steps), "reversible": reversible, "s1_evicted_A": s1_evicted_A,
            "recovered": recovered, "s2_brought_A_back": s2_brought_A_back}


# ── one seed: headline neural-triggered swap + specificity (salient/non-salient/match) + trigger lesion + reverse ──
def evaluate_seed(seed, *, proposal_pa=SALIENT_PA, nonsalient_pa=NONSALIENT_PA, boost_gain=BOOST_GAIN,
                  evict_steps=EVICT_STEPS, reignite_hold=REIGNITE_HOLD, w_rec=W_REC, heterogeneity=True, verbose=True):
    S = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    ws_used, pats_host = S["ws_used"], S["patterns_host"]
    b_, xp = S["bridge"], S["xp"]
    # ⚠ construct ALL STD instances NOW on the freshly-built substrate (each snapshots base recurrence at build).
    std = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_ns = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_match = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_les = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_rev = MultiLoopSTD(b_, xp, ws_used, pats_host)

    # HEADLINE: SALIENT MISMATCH (propose B while A held) -> mm fires -> neural trigger -> swap. CONTINUOUS.
    restore_before = _RESTORE_CALLS["n"]
    headline = run_intention_swap(S, std, incumbent=0, proposed=1, proposal_pa=proposal_pa, boost_gain=boost_gain,
                                  evict_steps=evict_steps, reignite_hold=reignite_hold, isolate=False)
    host_workspace_reset_calls = int(_RESTORE_CALLS["n"] - restore_before)

    # SPECIFICITY 1 — NON-SALIENT mismatch (weak proposal for B): mm sub-threshold -> no trigger -> A holds.
    nonsalient = run_intention_swap(S, std_ns, incumbent=0, proposed=1, proposal_pa=nonsalient_pa,
                                    boost_gain=boost_gain, evict_steps=evict_steps, reignite_hold=reignite_hold,
                                    isolate=True)
    # SPECIFICITY 2 — SALIENT MATCH (propose the HELD content A): pred vetoes mm -> no trigger -> A holds.
    match = run_intention_swap(S, std_match, incumbent=0, proposed=0, proposal_pa=proposal_pa, boost_gain=boost_gain,
                               evict_steps=evict_steps, reignite_hold=reignite_hold, isolate=True)
    # TRIGGER-LESION DISSOCIATION — salient mismatch but mm silenced (no proposal drive into mm): no swap, A holds.
    lesion = run_intention_swap(S, std_les, incumbent=0, proposed=1, proposal_pa=proposal_pa, boost_gain=boost_gain,
                                evict_steps=evict_steps, reignite_hold=reignite_hold, trigger_lesion=True, isolate=True)
    # REVERSIBILITY — two-swap A->B->A, both neural-mismatch-triggered.
    two = run_two_swap(S, std_rev, a=0, b=1, proposal_pa=proposal_pa, boost_gain=boost_gain, evict_steps=evict_steps,
                       reignite_hold=reignite_hold)

    # ── anti-cheats ──
    swap_ok = bool(headline["swapped"])
    reignite_ok = bool(headline["reignite_ok"])
    timing_ok = bool(headline["timing_ok"])
    # SPECIFICITY (the crux): the salient mismatch swaps; the non-salient and the match HOLD (mm did not fire).
    nonsalient_holds = bool(not nonsalient["swapped"] and nonsalient["held"])
    match_holds = bool(not match["swapped"] and match["held"])
    specificity_ok = bool(swap_ok and nonsalient_holds and match_holds)
    # TRIGGER LESION: silence mm -> the salient mismatch does NOT trigger a swap (the neural signal decides).
    lesion_holds = bool(not lesion["swapped"] and lesion["held"])
    trigger_load_bearing = bool(swap_ok and lesion_holds)
    reversible = bool(two["reversible"])
    swap_attr = attributable_to("neural-mismatch-triggered swap (headline vs trigger-silenced lesion)",
                                float(swap_ok), float(lesion["swapped"]), warn_below=0.0)

    # DETERMINISM (substrate-integrity anti-cheat).
    h1 = _izh_hash(b_)
    S2 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    seed_deterministic = bool(_izh_hash(S2["bridge"]) == h1 and h1 != "")

    seed_go = bool(swap_ok and reignite_ok and timing_ok and specificity_ok and trigger_load_bearing
                   and nonsalient_holds and match_holds and lesion_holds and reversible
                   and host_workspace_reset_calls == 0 and seed_deterministic)

    v = Verdict("GNW neural swap intention (seed %d)" % seed)
    v.require("incumbent ignites confidently (n_pre==1, winner A) [precondition]",
              bool(headline["n_ignited_pre"] == 1 and headline["winner_pre"] == 0), expect=True)
    v.require("NEURAL-TRIGGERED swap: salient mismatch -> mm fires -> old->baseline & new ignites (n_post==1, B)",
              swap_ok, expect=True)
    v.require("SPECIFICITY: non-salient mismatch does NOT swap (mm sub-threshold) -> incumbent holds",
              nonsalient_holds, expect=True)
    v.require("SPECIFICITY: salient MATCH does NOT swap (pred vetoes mm) -> incumbent holds", match_holds, expect=True)
    v.require("TRIGGER load-bearing: silence mm -> salient input does NOT trigger a swap -> incumbent holds",
              lesion_holds, expect=True)
    v.require("TIMING: challenger ignites only AFTER incumbent vacates (windowed, zero co-ignition)", timing_ok,
              expect=True)
    v.require("admitted coalition re-ignites and HOLDS", reignite_ok, expect=True)
    v.require("REVERSIBLE two-swap A->B->A (both neural-mismatch-triggered)", reversible, expect=True)
    v.require("no host workspace reset in the swap headline (continuous run)",
              host_workspace_reset_calls == 0, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating; STD targets ONLY the incumbent E->E loop, driven by mm")
    v.disabled("additive_substrate_hash",
               why="N/A: RNG-prefix property does not hold on this engine; determinism (build-twice) is used instead")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"proposal_pa": float(proposal_pa), "nonsalient_pa": float(nonsalient_pa),
                            "boost_gain": float(boost_gain), "max_boost": float(MAX_BOOST),
                            "boost_window": int(BOOST_WINDOW), "evict_steps": int(evict_steps),
                            "establish_pa": float(ESTABLISH_PA), "reignite_hold": int(reignite_hold),
                            "w_rec": float(w_rec), "w_pat_pred": float(W_PAT_PRED), "w_pred_mm": float(W_PRED_MM),
                            "mm_per": int(MM_PER), "pred_per": int(PRED_PER),
                            "w_ws_occ": float(W_WS_OCC), "w_occ_gate": float(W_OCC_GATE), "w_gate_ws": float(W_GATE_WS),
                            "occ_n": int(OCC_N), "gate_per": int(GATE_PER),
                            "U_baseline": 0.0, "tau_D": float(STD_TAU_D), "heterogeneity": bool(heterogeneity)},
        "go_gate": {"swap_ok": swap_ok, "reignite_ok": reignite_ok, "timing_ok": timing_ok,
                    "specificity_ok": specificity_ok, "nonsalient_holds": nonsalient_holds,
                    "match_holds": match_holds, "trigger_load_bearing": trigger_load_bearing,
                    "lesion_holds": lesion_holds, "reversible": reversible,
                    "no_host_reset": bool(host_workspace_reset_calls == 0), "seed_deterministic": seed_deterministic},
        "specificity": {
            "salient_mismatch_swapped": swap_ok, "nonsalient_swapped": bool(nonsalient["swapped"]),
            "match_swapped": bool(match["swapped"]),
            "salient_mm_peak": headline["mm_peak"], "salient_boost_max": headline["boost_max"],
            "nonsalient_mm_peak": nonsalient["mm_peak"], "nonsalient_boost_max": nonsalient["boost_max"],
            "match_mm_peak": match["mm_peak"], "match_boost_max": match["boost_max"],
        },
        "anti_cheats": {"specificity_ok": specificity_ok, "trigger_load_bearing": trigger_load_bearing,
                        "nonsalient_non_trigger": nonsalient_holds, "match_veto": match_holds,
                        "trigger_lesion_holds": lesion_holds, "reversible_two_swap": reversible,
                        "no_host_workspace_reset": bool(host_workspace_reset_calls == 0),
                        "seed_deterministic": seed_deterministic, "swap_attributable_fraction": swap_attr},
        "residual": {
            "headline": {"winner_pre": headline["winner_pre"], "winner_post": headline["winner_post"],
                         "n_pre": headline["n_ignited_pre"], "n_post": headline["n_ignited_post"],
                         "old_residual_post": headline["old_residual_post"],
                         "new_ignited": bool(_ignited(headline["new_rate_post"])),
                         "new_rate_post": headline["new_rate_post"], "xA_min": headline["xA_min"],
                         "mm_peak": headline["mm_peak"], "boost_max": headline["boost_max"],
                         "trigger_step": headline["trigger_step"], "a_vacate_step": headline["a_vacate_step"],
                         "b_ignite_step": headline["b_ignite_step"], "coign_steps": headline["coign_steps"],
                         "old_residual_hold": headline["old_residual_hold"], "new_rate_hold": headline["new_rate_hold"]},
            "nonsalient": {"swapped": nonsalient["swapped"], "held": nonsalient["held"],
                           "winner_post": nonsalient["winner_post"], "n_post": nonsalient["n_ignited_post"],
                           "old_residual_post": nonsalient["old_residual_post"],
                           "new_rate_post": nonsalient["new_rate_post"], "mm_peak": nonsalient["mm_peak"],
                           "boost_max": nonsalient["boost_max"]},
            "match": {"swapped": match["swapped"], "held": match["held"], "winner_post": match["winner_post"],
                      "n_post": match["n_ignited_post"], "old_residual_post": match["old_residual_post"],
                      "new_rate_post": match["new_rate_post"], "mm_peak": match["mm_peak"],
                      "boost_max": match["boost_max"]},
            "trigger_lesion": {"swapped": lesion["swapped"], "held": lesion["held"],
                               "winner_post": lesion["winner_post"], "n_post": lesion["n_ignited_post"],
                               "old_residual_post": lesion["old_residual_post"],
                               "new_rate_post": lesion["new_rate_post"], "mm_peak": lesion["mm_peak"],
                               "boost_max": lesion["boost_max"]},
            "reversibility": {"reversible": two["reversible"], "s1_evicted_A": two["s1_evicted_A"],
                              "recovered": two["recovered"], "s2_brought_A_back": two["s2_brought_A_back"],
                              "xA_recovered": two["xA_recovered"], "recover_steps": two["recover_steps"],
                              "swap1_swapped": two["swap1"]["swapped"],
                              "swap2_winner_hold": two["swap2"]["winner_hold"],
                              "swap2_n_hold": two["swap2"]["n_ignited_hold"]},
        },
        "host_workspace_reset_calls": int(host_workspace_reset_calls),
        "substrate_hash": h1, "std_weight_writes": int(std.n_writes),
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        hd = headline
        print(f"[neural-swap-intent seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    HEADLINE(salient mismatch): win {hd['winner_pre']}->{hd['winner_post']} n {hd['n_ignited_pre']}->"
              f"{hd['n_ignited_post']} old_res={hd['old_residual_post']:.3f} new={hd['new_rate_post']:.3f} "
              f"mm_peak={hd['mm_peak']:.3f} boost_max={hd['boost_max']:.3f} trig@{hd['trigger_step']} "
              f"a_vac@{hd['a_vacate_step']} b_ig@{hd['b_ignite_step']} swapped={hd['swapped']} timing={hd['timing_ok']}",
              flush=True)
        print(f"    SPECIFICITY: non-salient swapped={nonsalient['swapped']} (mm_peak={nonsalient['mm_peak']:.3f} "
              f"boost={nonsalient['boost_max']:.3f} held={nonsalient['held']}) | "
              f"match swapped={match['swapped']} (mm_peak={match['mm_peak']:.3f} boost={match['boost_max']:.3f} "
              f"held={match['held']})", flush=True)
        print(f"    TRIGGER-LESION(mm silenced): swapped={lesion['swapped']} held={lesion['held']} "
              f"mm_peak={lesion['mm_peak']:.3f} boost={lesion['boost_max']:.3f} load_bearing={trigger_load_bearing}",
              flush=True)
        print(f"    REVERSE: reversible={two['reversible']} xA_recovered={two['xA_recovered']:.3f} "
              f"s1={two['swap1']['swapped']} s2_hold_win={two['swap2']['winner_hold']} | det={seed_deterministic} "
              f"resets={host_workspace_reset_calls}", flush=True)
    return result


# ── calibration: verify the mismatch/salience primitive on one seed (no swap; just the detector's ON/OFF) ─────────
def run_calibrate(seed, args):
    print(f"[neural-swap-intent calibrate] seed={seed} — mismatch/salience detector primitive", flush=True)
    S = build(seed=seed)
    b, xp = S["bridge"], S["xp"]
    read = int(args.evict_steps) // 6 if args.evict_steps else 45

    def _mm_rate_when(hold_pattern, propose_k, propose_pa):
        _full_restore(b, S["snap"])
        if hold_pattern is not None:
            _drive(b, xp, S["thal"], THAL_TONIC_PA, None, [(S["patterns"][hold_pattern], ESTABLISH_PA)], n=DRIVE_STEPS)
        rates = []
        for _ in range(read):
            _ws_step(b, xp, S["thal"], THAL_TONIC_PA, None,
                     drive_map=[(S["mm_slices"][propose_k], float(propose_pa))])
            rates.append(_pop_rate(S, S["mm_all"]))
        return float(np.mean(rates[-read // 2:]))

    salient_mismatch = _mm_rate_when(0, 1, SALIENT_PA)     # A held, propose B strongly -> mm SHOULD fire
    salient_match = _mm_rate_when(0, 0, SALIENT_PA)        # A held, propose A strongly -> pred vetoes -> mm silent
    nonsalient_mm = _mm_rate_when(0, 1, NONSALIENT_PA)     # A held, propose B weakly -> mm sub-threshold -> silent
    # the meaningful lines: salient FIRES mm (drives boost over the ~0.10 collapse knee at gain 1.0); the match is
    # VETOED by pred (near zero); the non-salient stays low enough that its boost stays under the ~0.06 hold line.
    ok = bool(salient_mismatch > 0.10 and salient_match < 0.02 and nonsalient_mm < 0.05)
    print(f"  A-held + SALIENT MISMATCH (propose B) -> mm_rate={salient_mismatch:.3f}  (want >0.10: fires)", flush=True)
    print(f"  A-held + SALIENT MATCH   (propose A) -> mm_rate={salient_match:.3f}  (want <0.02: pred veto)", flush=True)
    print(f"  A-held + NON-SALIENT     (propose B) -> mm_rate={nonsalient_mm:.3f}  (want <0.05: sub-threshold)",
          flush=True)
    print(f"  MISMATCH PRIMITIVE {'HOLDS' if ok else 'FAILS'}", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_neural_swap_intention_derisk", "mode": "calibrate", "seed": seed,
                       "salient_mismatch": salient_mismatch, "salient_match": salient_match,
                       "nonsalient_mm": nonsalient_mm, "primitive_ok": ok}, f, indent=2)
    return 0 if ok else 1


def run_smoke(seed, args):
    r = evaluate_seed(seed, proposal_pa=args.proposal_pa, nonsalient_pa=args.nonsalient_pa, boost_gain=args.boost_gain,
                      evict_steps=args.evict_steps, reignite_hold=args.reignite_hold, w_rec=args.w_rec,
                      heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_neural_swap_intention_derisk", "mode": "smoke", "seed": seed, "result": r}, f,
                  indent=2, default=str)
    print(f"\n[neural-swap-intent smoke] wrote {args.json}  seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[neural-swap-intent six-seed] seeds={seeds} @ salient={args.proposal_pa} nonsalient={args.nonsalient_pa} "
          f"boost_gain={args.boost_gain} evict={args.evict_steps}", flush=True)
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, proposal_pa=args.proposal_pa, nonsalient_pa=args.nonsalient_pa,
                                      boost_gain=args.boost_gain, evict_steps=args.evict_steps,
                                      reignite_hold=args.reignite_hold, w_rec=args.w_rec,
                                      heterogeneity=not args.no_heterogeneity, verbose=True))
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_swap = sum(1 for r in per_seed if r["go_gate"]["swap_ok"])
    n_spec = sum(1 for r in per_seed if r["go_gate"]["specificity_ok"])
    n_ns = sum(1 for r in per_seed if r["go_gate"]["nonsalient_holds"])
    n_match = sum(1 for r in per_seed if r["go_gate"]["match_holds"])
    n_lb = sum(1 for r in per_seed if r["go_gate"]["trigger_load_bearing"])
    n_les = sum(1 for r in per_seed if r["go_gate"]["lesion_holds"])
    n_timing = sum(1 for r in per_seed if r["go_gate"]["timing_ok"])
    n_reig = sum(1 for r in per_seed if r["go_gate"]["reignite_ok"])
    n_rev = sum(1 for r in per_seed if r["go_gate"]["reversible"])
    n_nores = sum(1 for r in per_seed if r["go_gate"]["no_host_reset"])
    n_det = sum(1 for r in per_seed if r["go_gate"]["seed_deterministic"])
    # swap rates (the specificity headline): salient vs non-salient vs match
    salient_swap_rate = sum(1 for r in per_seed if r["specificity"]["salient_mismatch_swapped"]) / len(seeds)
    nonsalient_swap_rate = sum(1 for r in per_seed if r["specificity"]["nonsalient_swapped"]) / len(seeds)
    match_swap_rate = sum(1 for r in per_seed if r["specificity"]["match_swapped"]) / len(seeds)
    pooled_go = bool(n_go >= 5 and n_swap >= 5 and n_spec >= 5 and n_ns == 6 and n_match == 6 and n_lb >= 5
                     and n_les == 6 and n_timing >= 5 and n_reig >= 5 and n_rev >= 5 and n_nores == 6 and n_det == 6)
    verdict = "GO" if pooled_go else ("PARTIAL" if n_swap >= 1 else "NO-GO")

    v = Verdict("GNW neural swap intention: 6-seed aggregate")
    v.require("neural-triggered swap on >=5/6", bool(n_swap >= 5), expect=True)
    v.require("SPECIFICITY on >=5/6 (salient swaps; non-salient + match hold)", bool(n_spec >= 5), expect=True)
    v.require("NON-SALIENT does NOT trigger a swap on 6/6", bool(n_ns == 6), expect=True)
    v.require("MATCH does NOT trigger a swap (pred veto) on 6/6", bool(n_match == 6), expect=True)
    v.require("trigger load-bearing on >=5/6 (silence mm -> no swap)", bool(n_lb >= 5), expect=True)
    v.require("trigger-lesion holds on 6/6 (silenced mm -> incumbent holds)", bool(n_les == 6), expect=True)
    v.require("TIMING correct on >=5/6", bool(n_timing >= 5), expect=True)
    v.require("re-ignites and holds on >=5/6", bool(n_reig >= 5), expect=True)
    v.require("reversible two-swap on >=5/6", bool(n_rev >= 5), expect=True)
    v.require("no host workspace reset on 6/6", bool(n_nores == 6), expect=True)
    v.require("determinism on 6/6", bool(n_det == 6), expect=True)
    v.disabled("homeostasis", why="frozen base weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating; STD targets ONLY the incumbent E->E recurrence")
    v.disabled("additive_substrate_hash",
               why="N/A: RNG-prefix property does not hold on this engine; determinism (build-twice) is used instead")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_neural_swap_intention_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "swap_rates": {"salient_mismatch": salient_swap_rate, "nonsalient_mismatch": nonsalient_swap_rate,
                              "salient_match": match_swap_rate},
               "counts": {"seed_go": n_go, "swap_ok": n_swap, "specificity_ok": n_spec, "nonsalient_holds": n_ns,
                          "match_holds": n_match, "trigger_load_bearing": n_lb, "lesion_holds": n_les,
                          "timing_ok": n_timing, "reignite_ok": n_reig, "reversible": n_rev,
                          "no_host_reset": n_nores, "seed_deterministic": n_det, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[neural-swap-intent six-seed] verdict={verdict} seed_go {n_go}/6 swap {n_swap}/6 spec {n_spec}/6 "
          f"non_salient_holds {n_ns}/6 match_holds {n_match}/6 load_bearing {n_lb}/6 lesion_holds {n_les}/6 "
          f"timing {n_timing}/6 reignite {n_reig}/6 reversible {n_rev}/6 no_reset {n_nores}/6 det {n_det}/6", flush=True)
    print(f"[neural-swap-intent six-seed] SWAP RATES  salient={salient_swap_rate:.2f}  "
          f"non_salient={nonsalient_swap_rate:.2f}  match={match_swap_rate:.2f}  -> POOLED_GO={pooled_go}", flush=True)
    print(f"[neural-swap-intent six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW neural swap intention: a spiking mismatch/salience detector decides "
                                             "WHEN to swap the held workspace thought (self-driven swap trigger).")
    ap.add_argument("--calibrate", action="store_true", help="verify the mismatch/salience detector primitive")
    ap.add_argument("--smoke", action="store_true", help="full single-seed evaluation")
    ap.add_argument("--six-seed", action="store_true", help="42/43/44/100/101/102 at the frozen operating point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proposal-pa", type=float, default=SALIENT_PA, help="SALIENT proposal drive (pA)")
    ap.add_argument("--nonsalient-pa", type=float, default=NONSALIENT_PA, help="NON-SALIENT proposal drive (pA)")
    ap.add_argument("--boost-gain", type=float, default=BOOST_GAIN, help="mm-rate -> STD-boost gain")
    ap.add_argument("--evict-steps", type=int, default=EVICT_STEPS)
    ap.add_argument("--reignite-hold", type=int, default=REIGNITE_HOLD)
    ap.add_argument("--w-rec", type=float, default=W_REC)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_neural_swap_intention.json")
    args = ap.parse_args()

    if args.calibrate:
        return run_calibrate(args.seed, args)
    if args.six_seed:
        return run_six_seed(args)
    if args.smoke:
        return run_smoke(args.seed, args)
    r = evaluate_seed(args.seed, proposal_pa=args.proposal_pa, nonsalient_pa=args.nonsalient_pa,
                      boost_gain=args.boost_gain, evict_steps=args.evict_steps, reignite_hold=args.reignite_hold,
                      w_rec=args.w_rec, heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_neural_swap_intention_derisk", "mode": "single", "result": r}, f,
                  indent=2, default=str)
    print(f"[neural-swap-intent] wrote {args.json} seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
