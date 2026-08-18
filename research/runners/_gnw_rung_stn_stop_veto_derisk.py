"""GNW STN->GPi hyperdirect REACTIVE STOP-SIGNAL veto: the conflict-gated GLOBAL BRAKE on the workspace.

The brain-based replacement for the banked HOST "active-clear FS quench" shortcut (rung-2c do-not-retread:
"active-clear FS quench works but is a HOST shortcut"). A neural conflict monitor reads the workspace's OWN
spiking ignition margin, and — when the commit is UNRELIABLE (low margin = conflict = the workspace failed to
reach single-content access, holding two co-active contents) — a hyperdirect ACC->STN->GPi loop fires a broad
inhibitory STOP pulse that KILLS ALL workspace content (workspace -> EMPTY, n_ignited -> 0), flipping the
delivered verdict from a wrong/ambiguous COMMIT to ABSTAIN. On a CONFIDENT commit (high margin, single content)
the veto stays silent and the correct content broadcasts.

WHY THIS SIDESTEPS THE Rung-2b/2c EVICTION HORN (do NOT re-hit it): Rung-2b (SFA) and Rung-2c (salience
dis-inhibition) tried to EVICT one attractor to SWAP IN a more-salient challenger (n_ignited stays 1) — that
needs a metastable tip-point the dense weight-30 recurrent attractor does not have (inhibition-resistant:
co-ignite or self-extinguish; the metastable middle is EMPTY — see
`research/findings/2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY.md` and
`research/findings/2026-08-14-gnw-rung2b-sfa-workspace-eviction-BOUNDARY.md`). The STN stop is a GLOBAL reactive
brake: SELF-EXTINCTION is the DESIRED outcome, so it uses exactly the regime the eviction attempts fell into
("above the eviction inhibition, NEITHER ignites") as its stopping mechanism. It never needs a metastable state.

BIOLOGY (grounded): Frank 2006 hyperdirect "hold-your-horses"; Aron & Poldrack 2006 / Wessel & Aron 2017 broad
fast STN GLOBAL reactive stop; Wei-Rubin-Wang 2015 STN-GPe stopping dynamics. Presets exist in-repo
(`sim/enums.py` IZH2007_STN_BURST + IZH2007_GPI_OUTPUT; `sim/profiles.py` STN-GPe loop) — the substrate here uses
the validated GENERIC_UNSTRUCTURED Izhikevich workspace pool for the assemblies and dense hand-wired ACC/STN/GPi
pools (reuse-by-import; NO `sim/` edit; explicit-wiring only, the Rung-2b/2c pattern).

MECHANISM (explicit wiring on the built workspace; reuse-by-import of the Rung-1 assembly-loop + the _p1_2
K-slot workspace harness + the Rung-2 competitive harness):
  (a) ACC conflict unit — a spiking pool whose afferent drive ENCODES the NEURAL conflict: the host reads the
      workspace's OWN late-window per-slot spike rates (the instrument), forms margin = winner_rate -
      runnerup_rate, and injects i_acc = conflict_gain * max(0, MARGIN_REF - margin) * CURRENT_SCALE into the ACC
      pool. This is a SENSOR on the spiking ignition margin, NOT a host "this-answer-is-wrong" ground-truth flag
      (verified: a margin-SCRAMBLE that feeds a confident margin to a conflict trial breaks the abort).
  (b) STN pool — excited by ACC (ACC->STN, glutamatergic hyperdirect); its effective drive scales with conflict.
  (c) STN->GPi excitatory + GPi->workspace BROAD inhibitory (all slots) = the timed reactive STOP pulse. GPi is
      near-silent at baseline (phasic-stop model: the hyperdirect STN->GPi INCREASES GPi firing -> broad
      inhibition), so at zero conflict there is no brake and the workspace holds its content.

GO GATE (6 seeds 42/43/44/100/101/102, ALL must hold):
  (1) ABORT — HIGH-conflict/low-margin commit: the STN->GPi pulse drives n_ignited -> 0 and the DELIVERED verdict
      flips wrong/ambiguous-COMMIT -> ABSTAIN.
  (2) SELECTIVITY — LOW-conflict/high-margin confident commit: the veto does NOT fire (i_acc ~ 0), the correct
      commit BROADCASTS normally (n_ignited stays >= 1).
  (3) CAUSAL — STN-lesion (STN->GPi weight = 0): the wrong commit is NOT aborted (broadcasts the error);
      attributable_to(intact_empty, lesion_empty) high on all 6.
  (4) SIGNATURE — post-abort workspace is EMPTY (n_ignited == 0), distinguishing a GLOBAL stop from Rung-2
      eviction's single-challenger swap (which leaves n_ignited == 1).

ANTI-CHEATS (each asserted in code):
  (1) SIGNAL-DRIVEN not host reset: i_acc = f(ACC conflict); pulse_zero_at_zero_conflict AND
      pulse_scales_with_conflict (a current into a spiking pool timed as a pulse, NOT a `_restore_state`).
  (2) NEURAL SENSOR (load-bearing): host-margin-SCRAMBLE (feed the confident margin to the conflict trial) breaks
      the abort.
  (3) CONFLICT-OFF reproduces the negative: conflict_gain = 0 -> the wrong commit broadcasts uncorrected.
  (4) STN-lesion load-bearing (GO gate 3).
  (5) 0 `_restore_state`/`_full_restore` calls in the continuous abort headline (the emptying is NEURAL, not a
      host wash-out).
  (6) determinism: build-twice-at-one-seed identical hash of the seed-derived Izhikevich params (cfg.seed, NOT
      actual_seed_used).
  (7) verdict is a clean GO/NO-GO, not UNDEFINED — all validity preconditions (commit ignites on both trials,
      the neural margin distinguishes conflict from confident, the pulse is gated, the lesion changes the
      outcome) are checked BEFORE scoring.

Usage (CPU cheap-first):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung_stn_stop_veto_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_stn_stop_veto_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung_stn_stop_veto_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_stn_stop_veto_6seed.json
"""
from __future__ import annotations

import argparse
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

# reuse-by-import: Rung-1 assembly-loop builder + protocol constants (the validated ignition recipe).
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, DEFAULT_ATTRACTOR_WEIGHT, DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS,
)
# reuse-by-import: the Rung-2 competitive harness (ignition read + the K-slot geometry constants).
from research.runners._gnw_rung2_competitive_access_derisk import (
    _ignited, IGNITE_FRAC, SOLO_PLATEAU, ASSEMBLY_SIZE, WS_LOOP_GATE,
)
# reuse-by-import: the Rung-2b determinism hash + the Rung-2c dense-population + restore-accounting helpers.
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
# reuse-by-import: the _p1_2 built-workspace harness (K-slot workspace + the full-state wash-out + drive constants).
from research.runners._p1_2_workspace_deliberation_loop_derisk import (
    _full_snapshot, _full_restore, K_SLOTS, IGNITE_PA, DISTRACTOR_FRAC,
)

# ── geometry: K self-recurrent workspace slots + ACC -> STN -> GPi hyperdirect stop loop ──────────────────────
N_SLOTS = K_SLOTS                       # 4 candidate slots (reuse the _p1_2 workspace geometry)
WORKSPACE_N = N_SLOTS * ASSEMBLY_SIZE + 20
# Feedforward Izhikevich pools cap at ~0.15-0.30 firing (refractory), so the hyperdirect chain is lossy; a LARGE
# GPi pool makes the chain-delivered inhibition reach the workspace-silencing g_i range even at that rate (probed).
ACC_N = 80                              # ACC conflict unit (cortical; excitatory hyperdirect source)
STN_N = 120                             # subthalamic nucleus (glutamatergic)
GPI_N = 200                             # GPi/SNr output gate (GABAergic; broad workspace inhibition)

ACC_STN_W = 25.0                        # ACC -> STN excitation (hyperdirect; tuned for chain transmission)
STN_GPI_W = 25.0                        # STN -> GPi excitation
# GPi -> workspace BROAD inhibition (per-synapse; effective g_i ~ N_gpi * gpi_rate * w * inhib_prop). 8.0 is the
# moderate operating point where the effector delivers a real g_i (~30) yet the dense recurrent attractor HOLDS
# (inhibition-resistant); the effector-residual sweep spans weaker (survivor) -> stronger (destabilizes upward).
GPI_WS_W = 8.0

# drive amplitudes. Heterogeneity (seed-derived, ON for the determinism hash) makes the bare IGNITE_PA=2500
# sit at the per-slot ignition KNEE — some slots then miss (probed: het amp=2500 ignites only ~1/4 slots, amp>=4500
# ignites all). STRONG_PA is well ABOVE every slot's knee so per-slot ignition is robust (probed all 6 seeds);
# WEAK_PA is well BELOW it so a competing/distractor slot does NOT self-ignite (probed max slot rate <=0.05 << the
# 0.167 ignite threshold on all 6 seeds).
STRONG_PA = 2.0 * IGNITE_PA             # 5000 pA: robust per-slot ignition under heterogeneity
WEAK_PA = 1.6 * IGNITE_PA * DISTRACTOR_FRAC   # 1200 pA: safely sub-knee (a slot that does NOT self-ignite)

# conflict sensor: margin = winner_rate - runnerup_rate (late-window per-slot). deficit = max(0, REF - margin).
MARGIN_REF = SOLO_PLATEAU * 0.5         # 1/6: halfway between single-content (~1/3) and co-ignition (~0)
CONFLICT_CURRENT_SCALE = 18000.0        # i_acc = conflict_gain * deficit * SCALE (deficit ~1/6 -> i_acc ~3000 pA)
PULSE_DURATION = 60                     # steps the ACC->STN->GPi stop pulse is driven

# OU noise DESYNCHRONIZES the deterministic period-3 attractor into a proper ASYNC rate attractor (the Rung-2/2b/2c
# fix). A synchronous attractor produces a synchronous REBOUND volley when the stop-inhibition releases, which
# re-latches the loop; an async rate attractor has no synchronized rebound, so a silenced workspace stays empty.
OU_NOISE_PA = 30.0
# GPi is tonically active in vivo (high baseline rate; Aron & Poldrack 2006); the hyperdirect STN->GPi INCREASES
# it phasically. A residual tonic GPi inhibition (baseline drive into GPi) raises the workspace's re-ignition
# threshold so a just-cleared workspace cannot spontaneously re-latch, WITHOUT extinguishing an already-sustained
# confident commit (bistability with a raised floor). 0.0 = phasic-only (tested first).
GPI_TONIC_PA = 0.0

READ_FREE_STEPS = 45                    # free steps to let the commit settle before reading the margin
POST_FREE_STEPS = FREE_STEPS            # free steps after the stop pulse to read the settled (empty?) workspace

# restore-call accounting: the CONTINUOUS abort headline MUST make ZERO restore calls (anti-cheat 5).
_RESTORE_CALLS = {"n": 0}


def _counted_full_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


def build_stn_stop_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                          stn_lesion: bool = False, acc_stn_w: float = ACC_STN_W, stn_gpi_w: float = STN_GPI_W,
                          gpi_ws_w: float = GPI_WS_W, heterogeneity: bool = True, ou_noise_pA: float = OU_NOISE_PA,
                          gpi_tonic_pA: float = GPI_TONIC_PA):
    """One `workspace` region (N_SLOTS dense self-recurrent assemblies) + ACC/STN/GPi pools wired as the
    hyperdirect stop loop: ACC -> STN (E), STN -> GPi (E), GPi -> ALL workspace neurons (broad I).
      stn_lesion=True -> STN->GPi weight 0 (the CAUSAL anti-cheat: STN fires but cannot excite GPi -> no brake).
    Deterministic clean ignition (heterogeneity seed-derived from cfg.seed for the determinism hash; OU off).
    Returns (bridge, xp, slots_dev, acc_dev, snap, handles)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    acc = BrainRegion(name="acc", n_neurons=ACC_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    stn = BrainRegion(name="stn", n_neurons=STN_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    gpi = BrainRegion(name="gpi", n_neurons=GPI_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False)
    regions = [workspace, acc, stn, gpi]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []            # ALL inter-region wiring is explicit (sub-slice precision)
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)  # explicit: heterogeneity is seed-derived (determinism hash)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False      # FOOT-GUN: synaptic-scaling clip slams the frozen attractor weights
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:               # OU desynchronizes the attractor -> no synchronous rebound on stop release
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
    ws = rm.indices("workspace")
    slots = [np.asarray(ws[i * ASSEMBLY_SIZE:(i + 1) * ASSEMBLY_SIZE], dtype=np.int64) for i in range(N_SLOTS)]
    ws_all = np.asarray(ws[:N_SLOTS * ASSEMBLY_SIZE], dtype=np.int64)   # GPi inhibits every assembly neuron
    acc_idx = np.asarray(rm.indices("acc"), dtype=np.int64)
    stn_idx = np.asarray(rm.indices("stn"), dtype=np.int64)
    gpi_idx = np.asarray(rm.indices("gpi"), dtype=np.int64)

    eff_weight = float(attractor_weight)
    stn_gpi_eff = 0.0 if stn_lesion else float(stn_gpi_w)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    for i, s in enumerate(slots):
        union_plan[f"workspace_loop_{i}"] = _build_assembly_loop_population(s, eff_weight)   # gated WS_LOOP_GATE
    # the hyperdirect stop loop (dense all-to-all frozen populations).
    union_plan["acc2stn"] = _dense_pop(acc_idx, stn_idx, float(acc_stn_w), "E_TO_E")
    union_plan["stn2gpi"] = _dense_pop(stn_idx, gpi_idx, stn_gpi_eff, "E_TO_E")
    union_plan["gpi2ws"] = _dense_pop(gpi_idx, ws_all, float(gpi_ws_w), "I_TO_E")   # broad workspace inhibition

    inh = list(gpi_idx)                 # GPi is the only inhibitory (GABAergic) source; ACC/STN/workspace excite
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "stn_lesion": bool(stn_lesion), "attractor_weight": eff_weight,
               "acc_stn_w": float(acc_stn_w), "stn_gpi_w": float(stn_gpi_eff), "gpi_ws_w": float(gpi_ws_w),
               "heterogeneity": bool(heterogeneity), "ou_noise_pA": float(ou_noise_pA),
               "gpi_tonic_pA": float(gpi_tonic_pA), "n_slots": N_SLOTS}
    slots_dev = [xp.asarray(s) for s in slots]
    return bridge, xp, slots_dev, xp.asarray(acc_idx), xp.asarray(gpi_idx), snap, handles


def _read_slot_rates_free(bridge, xp, slots_dev, n_free: int):
    """Free-run `n_free` steps (zero external current) and return the LATE-window (last third) per-slot mean
    per-neuron firing rate. Leaves the workspace evolving in place (so a later stop pulse acts on the live
    state) — this is the spiking-ignition read, the SENSOR the ACC conflict unit is a function of."""
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(slots_dev)
    for t in range(n_free):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            for i, s_dev in enumerate(slots_dev):
                counts[i] += int(to_host(bridge.cp_firing_states[s_dev].astype(xp.float64).sum()))
    denom = float((n_free - late_start) * ASSEMBLY_SIZE)
    return [c / denom for c in counts]


def _margin(rates):
    """margin = top - runnerup over the K slot rates (order-invariant). Returns (winner_idx, margin, n_ignited)."""
    order = sorted(range(len(rates)), key=lambda i: rates[i], reverse=True)
    top = rates[order[0]]
    second = rates[order[1]] if len(order) > 1 else 0.0
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    return int(order[0]), float(top - second), n_ign


def _verdict_label(rates):
    """The DELIVERED verdict read from the settled workspace: ABSTAIN if empty, else COMMIT to the argmax slot
    (n_ignited > 1 is an AMBIGUOUS commit — the conflict state)."""
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    if n_ign == 0:
        return "ABSTAIN", 0
    winner = int(np.argmax(rates))
    return (f"COMMIT_slot{winner}" if n_ign == 1 else f"AMBIGUOUS_COMMIT_slot{winner}"), n_ign


def run_stop_trial(bridge, xp, slots_dev, acc_dev, snap, drives, *, conflict_gain, current_scale, margin_ref,
                   pulse_duration, do_stop=True, isolate=True, margin_override=None, gpi_dev=None):
    """ONE continuous stop trial: (1) DRIVE the slots to ignition; (2) free-run + READ the late-window per-slot
    rates = the commit + the ignition margin (the ACC sensor); (3) inject i_acc = conflict_gain *
    max(0, margin_ref - margin) * current_scale into the ACC pool for `pulse_duration` steps (ACC->STN->GPi->
    broad workspace inhibition = the reactive STOP), then (4) free-run + READ the settled (empty?) workspace.
    CONTINUOUS: with isolate=False there is NO `_restore` anywhere (the emptying is purely neural). margin_override
    feeds a DIFFERENT margin to the sensor (the NEURAL-SENSOR scramble anti-cheat)."""
    bridge.cp_external_input_current[:] = 0.0
    if isolate:
        _counted_full_restore(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    # (1) DRIVE the candidate slots.
    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        for s_dev, d in zip(slots_dev, drives):
            if d > 0.0:
                bridge.cp_external_input_current[s_dev] = xp.float32(d)
        bridge._run_one_simulation_step()

    # (2) COMMIT + READ the ignition margin (the workspace's OWN spiking read).
    bridge.cp_external_input_current[:] = 0.0
    pre_rates = _read_slot_rates_free(bridge, xp, slots_dev, READ_FREE_STEPS)
    winner_pre, margin, n_pre = _margin(pre_rates)
    sensed_margin = float(margin if margin_override is None else margin_override)
    deficit = max(0.0, float(margin_ref) - sensed_margin)
    i_acc = float(conflict_gain) * deficit * float(current_scale)     # the ACC conflict-unit drive (signal-driven)

    # (3) the reactive STOP pulse: ACC -> STN -> GPi -> broad workspace inhibition. Record GPi + workspace firing
    #     over the last third of the pulse (documents the effector genuinely delivers inhibition).
    gpi_during = []
    ws_during = []
    if do_stop:
        pdur = int(pulse_duration)
        late = pdur - max(1, pdur // 3)
        for t in range(pdur):
            bridge.cp_external_input_current[:] = 0.0
            if i_acc > 0.0:
                bridge.cp_external_input_current[acc_dev] = xp.float32(i_acc)   # conflict>0: drive the ACC unit
            bridge._run_one_simulation_step()
            if t >= late:
                if gpi_dev is not None:
                    gpi_during.append(float(to_host(bridge.cp_firing_states[gpi_dev].astype(xp.float64).mean())))
                ws_during.append(float(np.mean([to_host(bridge.cp_firing_states[s].astype(xp.float64).mean())
                                                for s in slots_dev])))

    # (4) POST-READ the settled workspace (empty on a successful abort).
    bridge.cp_external_input_current[:] = 0.0
    post_rates = _read_slot_rates_free(bridge, xp, slots_dev, POST_FREE_STEPS)
    _wp, _m, n_post = _margin(post_rates)
    verdict_pre, _ = _verdict_label(pre_rates)
    verdict_post, _ = _verdict_label(post_rates)
    aborted = bool(n_pre >= 1 and n_post == 0)
    return {
        "drives": [float(d) for d in drives], "pre_rates": [float(r) for r in pre_rates],
        "post_rates": [float(r) for r in post_rates], "margin": float(margin), "sensed_margin": sensed_margin,
        "deficit": float(deficit), "i_acc": float(i_acc), "n_ignited_pre": int(n_pre),
        "n_ignited_post": int(n_post), "winner_pre": int(winner_pre), "delivered_pre": verdict_pre,
        "delivered_post": verdict_post, "aborted": aborted,
        "gpi_rate_during": (float(np.mean(gpi_during)) if gpi_during else None),
        "ws_rate_during": (float(np.mean(ws_during)) if ws_during else None),
    }


def _confident_drives():
    return [STRONG_PA] + [WEAK_PA] * (N_SLOTS - 1)          # slot0 strong (single content), rest weak


def _conflict_drives():
    return [STRONG_PA, STRONG_PA] + [WEAK_PA] * (N_SLOTS - 2)  # slots 0,1 co-active competitors (co-ignition)


EFFECTOR_SWEEP_W = [8.0, 16.0, 24.0, 32.0, 40.0]   # GPi->workspace strengths for the effector-residual sweep


def evaluate_seed(seed, *, conflict_gain, current_scale, margin_ref, pulse_duration,
                  acc_stn_w, stn_gpi_w, gpi_ws_w, heterogeneity, verbose=True):
    """Build the STN-stop bridge and measure the GO gate + all anti-cheats at ONE frozen operating point."""
    conf_drives = _confident_drives()
    conf_drivesf = [float(x) for x in conf_drives]
    conflict_drives = _conflict_drives()

    bridge, xp, slots_dev, acc_dev, gpi_dev, snap, handles = build_stn_stop_bridge(
        seed=seed, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w, gpi_ws_w=gpi_ws_w, heterogeneity=heterogeneity)

    # ── GO(2) SELECTIVITY: a CONFIDENT commit (veto ON) must NOT fire the brake -> broadcasts ─────────────────
    confident = run_stop_trial(bridge, xp, slots_dev, acc_dev, snap, conf_drivesf, conflict_gain=conflict_gain,
                               current_scale=current_scale, margin_ref=margin_ref, pulse_duration=pulse_duration,
                               do_stop=True, isolate=True, gpi_dev=gpi_dev)
    selectivity = bool(confident["i_acc"] <= 1e-6 and confident["n_ignited_post"] >= 1
                       and confident["delivered_post"].startswith("COMMIT"))

    # ── GO(1)+(4) ABORT + SIGNATURE: a HIGH-conflict commit (veto ON) empties the workspace -> ABSTAIN ────────
    abort = run_stop_trial(bridge, xp, slots_dev, acc_dev, snap, [float(x) for x in conflict_drives],
                           conflict_gain=conflict_gain, current_scale=current_scale, margin_ref=margin_ref,
                           pulse_duration=pulse_duration, do_stop=True, isolate=True, gpi_dev=gpi_dev)
    abort_ok = bool(abort["n_ignited_pre"] >= 1 and abort["n_ignited_post"] == 0
                    and abort["delivered_post"] == "ABSTAIN")
    signature_empty = bool(abort["n_ignited_post"] == 0)

    # ── ANTI-CHEAT 1: pulse SIGNAL-DRIVEN — sweep the conflict level (slot1 weak->equal) and record i_acc ──────
    #    i_acc must be 0 at the confident end (zero conflict) and rise monotonically with conflict.
    slot1_sweep = list(np.linspace(WEAK_PA, STRONG_PA, 6))
    sweep_i_acc, sweep_margin = [], []
    for d1 in slot1_sweep:
        dv = [STRONG_PA, float(d1)] + [WEAK_PA] * (N_SLOTS - 2)
        r = run_stop_trial(bridge, xp, slots_dev, acc_dev, snap, dv, conflict_gain=conflict_gain,
                           current_scale=current_scale, margin_ref=margin_ref, pulse_duration=pulse_duration,
                           do_stop=False, isolate=True)                       # do_stop=False: just read the pulse
        sweep_i_acc.append(r["i_acc"]); sweep_margin.append(r["margin"])
    pulse_zero_at_zero_conflict = bool(sweep_i_acc[0] <= 1e-6)
    pulse_scales_with_conflict = bool(all(sweep_i_acc[i + 1] >= sweep_i_acc[i] - 1e-6
                                          for i in range(len(sweep_i_acc) - 1)) and sweep_i_acc[-1] > 1e-6)

    # ── ANTI-CHEAT 2: NEURAL-SENSOR scramble — feed the CONFIDENT margin to the conflict trial -> abort breaks ─
    scramble = run_stop_trial(bridge, xp, slots_dev, acc_dev, snap, [float(x) for x in conflict_drives],
                              conflict_gain=conflict_gain, current_scale=current_scale, margin_ref=margin_ref,
                              pulse_duration=pulse_duration, do_stop=True, isolate=True,
                              margin_override=confident["margin"])
    scramble_breaks_abort = bool(not scramble["aborted"] and scramble["n_ignited_post"] >= 1)

    # ── ANTI-CHEAT 3: CONFLICT-OFF — conflict_gain=0 -> no pulse -> the wrong commit broadcasts uncorrected ────
    conflict_off = run_stop_trial(bridge, xp, slots_dev, acc_dev, snap, [float(x) for x in conflict_drives],
                                  conflict_gain=0.0, current_scale=current_scale, margin_ref=margin_ref,
                                  pulse_duration=pulse_duration, do_stop=True, isolate=True)
    conflict_off_broadcasts = bool(not conflict_off["aborted"] and conflict_off["n_ignited_post"] >= 1)

    # ── GO(3) CAUSAL: STN-lesion (STN->GPi weight 0) fails to abort the wrong commit -> broadcasts the error ───
    bridge_l, xp_l, slots_l, acc_l, gpi_l, snap_l, _ = build_stn_stop_bridge(
        seed=seed, stn_lesion=True, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w, gpi_ws_w=gpi_ws_w,
        heterogeneity=heterogeneity)
    lesion = run_stop_trial(bridge_l, xp_l, slots_l, acc_l, snap_l, [float(x) for x in conflict_drives],
                            conflict_gain=conflict_gain, current_scale=current_scale, margin_ref=margin_ref,
                            pulse_duration=pulse_duration, do_stop=True, isolate=True, gpi_dev=gpi_l)
    lesion_broadcasts = bool(not lesion["aborted"] and lesion["n_ignited_post"] >= 1)
    # attribution: what fraction of the workspace-emptying is caused by the intact STN->GPi (vs the lesion)?
    intact_empty = float(abort["n_ignited_pre"] - abort["n_ignited_post"])   # slots emptied, intact
    lesion_empty = float(lesion["n_ignited_pre"] - lesion["n_ignited_post"]) # slots emptied, lesion
    stn_attribution = attributable_to("workspace-emptying via STN->GPi", intact_empty, lesion_empty,
                                      warn_below=0.8)
    causal_ok = bool(lesion_broadcasts and (stn_attribution is not None) and stn_attribution >= 0.8)

    # ── ANTI-CHEAT 5: CONTINUOUS abort headline — ZERO restore calls (any emptying is neural, not a host wash-out)
    bridge_c, xp_c, slots_c, acc_c, gpi_c, snap_c, _ = build_stn_stop_bridge(
        seed=seed, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w, gpi_ws_w=gpi_ws_w, heterogeneity=heterogeneity)
    restore_before = _RESTORE_CALLS["n"]
    cont = run_stop_trial(bridge_c, xp_c, slots_c, acc_c, snap_c, [float(x) for x in conflict_drives],
                          conflict_gain=conflict_gain, current_scale=current_scale, margin_ref=margin_ref,
                          pulse_duration=pulse_duration, do_stop=True, isolate=False, gpi_dev=gpi_c)   # 0 restores
    continuous_no_restore = bool(_RESTORE_CALLS["n"] == restore_before)
    continuous_abort = bool(cont["aborted"] and cont["n_ignited_post"] == 0)

    # ── EFFECTOR-RESIDUAL SWEEP: vary the GPi->workspace strength on the CONFLICT abort and record n_ignited_post.
    #    QUANTIFIES the residual (the effector arm): no GPi strength drives n_ignited -> 0 (weak = survivor/hold;
    #    strong = the dense recurrent attractor DESTABILIZES upward, n rises) — the Rung-2c inhibition-resistance
    #    boundary, now for the global-clear case. This is a MEASUREMENT, not a validity precondition.
    residual = []
    for w in EFFECTOR_SWEEP_W:
        b_s, xp_s, sl_s, ac_s, gp_s, sn_s, _ = build_stn_stop_bridge(
            seed=seed, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w, gpi_ws_w=float(w), heterogeneity=heterogeneity)
        r_s = run_stop_trial(b_s, xp_s, sl_s, ac_s, sn_s, [float(x) for x in conflict_drives],
                             conflict_gain=conflict_gain, current_scale=current_scale, margin_ref=margin_ref,
                             pulse_duration=pulse_duration, do_stop=True, isolate=True, gpi_dev=gp_s)
        residual.append({"gpi_ws_w": float(w), "n_ignited_post": int(r_s["n_ignited_post"]),
                         "gpi_rate_during": r_s["gpi_rate_during"], "ws_rate_during": r_s["ws_rate_during"],
                         "post_rates": r_s["post_rates"]})
    min_n_post_over_sweep = min(r["n_ignited_post"] for r in residual)
    ever_empty = bool(min_n_post_over_sweep == 0)

    # ── ANTI-CHEAT 6: determinism — build twice at one seed, hash the seed-derived Izhikevich params ──────────
    h1 = _threshold_hash(bridge, xp)
    bridge2, xp2, _, _, _, _, _ = build_stn_stop_bridge(
        seed=seed, acc_stn_w=acc_stn_w, stn_gpi_w=stn_gpi_w, gpi_ws_w=gpi_ws_w, heterogeneity=heterogeneity)
    h2 = _threshold_hash(bridge2, xp2)
    seed_deterministic = bool(h1 == h2 and h1 != "")

    # ── VALIDITY preconditions (checked BEFORE scoring; a failure -> UNDEFINED, NOT a false negative). These are
    #    the conditions under which the effector test is INTERPRETABLE: the sensor works and there is a commit to
    #    abort. The ABORT itself (n->0) is the MEASURED outcome, NOT a precondition — its failure is the negative,
    #    exactly the Rung-2b/2c split (encoding the outcome as a require would wrongly mark a valid NO-GO UNDEFINED).
    commit_ignites_confident = bool(confident["n_ignited_pre"] >= 1)
    commit_ignites_conflict = bool(abort["n_ignited_pre"] >= 1)
    margin_distinguishes = bool(abort["margin"] < confident["margin"] - 1e-6 and abort["margin"] < margin_ref
                                and confident["margin"] >= margin_ref)
    pulse_gated = bool(pulse_zero_at_zero_conflict and pulse_scales_with_conflict)
    gpi_fires = bool((abort["gpi_rate_during"] or 0.0) > 0.02)   # the effector genuinely delivers a GPi pulse

    seed_go = bool(abort_ok and selectivity and causal_ok and signature_empty
                   and scramble_breaks_abort and conflict_off_broadcasts
                   and continuous_no_restore and continuous_abort and seed_deterministic)

    v = Verdict("STN->GPi reactive stop-veto @ frozen operating point (seed %d)" % seed)
    v.require("commit ignites on the confident trial", commit_ignites_confident, expect=True)
    v.require("commit ignites on the conflict trial", commit_ignites_conflict, expect=True)
    v.require("neural margin distinguishes conflict (low) from confident (high)", margin_distinguishes, expect=True)
    v.require("pulse is conflict-gated (0 at zero conflict, scales with conflict)", pulse_gated, expect=True)
    v.require("SELECTIVITY: confident commit does NOT fire the veto (broadcasts)", selectivity, expect=True)
    v.require("effector delivers a GPi pulse (chain transmits)", gpi_fires, expect=True)
    v.require("host-margin-SCRAMBLE breaks the abort (neural sensor load-bearing)", scramble_breaks_abort,
              expect=True)
    v.require("CONFLICT-OFF reproduces the negative (gain=0 broadcasts)", conflict_off_broadcasts, expect=True)
    v.require("continuous abort headline makes 0 restore calls", continuous_no_restore, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.knob("ACC conflict drive (conflict trial i_acc)", requested=conflict_gain * abort["deficit"] * current_scale,
           applied=abort["i_acc"])
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("short_term_plasticity", why="not needed for a reactive global brake; banked eviction foot-gun")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"conflict_gain": float(conflict_gain), "current_scale": float(current_scale),
                            "margin_ref": float(margin_ref), "pulse_duration": int(pulse_duration),
                            "acc_stn_w": float(acc_stn_w), "stn_gpi_w": float(stn_gpi_w), "gpi_ws_w": float(gpi_ws_w),
                            "heterogeneity": bool(heterogeneity)},
        "go_gate": {
            "abort": abort_ok, "selectivity": selectivity, "causal": causal_ok, "signature_empty": signature_empty},
        "anti_cheats": {
            "pulse_zero_at_zero_conflict": pulse_zero_at_zero_conflict,
            "pulse_scales_with_conflict": pulse_scales_with_conflict,
            "neural_sensor_scramble_breaks_abort": scramble_breaks_abort,
            "conflict_off_broadcasts": conflict_off_broadcasts,
            "stn_lesion_broadcasts": lesion_broadcasts,
            "continuous_no_restore": continuous_no_restore,
            "seed_deterministic": seed_deterministic},
        "effector_residual_sweep": residual,
        "min_n_ignited_post_over_sweep": int(min_n_post_over_sweep),
        "effector_ever_empties_workspace": ever_empty,
        "measurements": {
            "confident": confident, "abort": abort, "scramble": scramble, "conflict_off": conflict_off,
            "lesion": lesion, "continuous_headline": cont,
            "gpi_rate_during_abort": abort["gpi_rate_during"], "ws_rate_during_abort": abort["ws_rate_during"],
            "sweep_slot1_drive": [float(x) for x in slot1_sweep],
            "sweep_i_acc": [float(x) for x in sweep_i_acc], "sweep_margin": [float(x) for x in sweep_margin],
            "stn_attribution": (None if stn_attribution is None else float(stn_attribution)),
            "intact_empty": intact_empty, "lesion_empty": lesion_empty,
            "threshold_hash": h1},
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        print(f"[stn-stop seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    ABORT: conflict n_ign {abort['n_ignited_pre']}->{abort['n_ignited_post']} "
              f"delivered {abort['delivered_pre']}->{abort['delivered_post']} margin={abort['margin']:.3f} "
              f"i_acc={abort['i_acc']:.0f} | SELECTIVITY: confident n_ign "
              f"{confident['n_ignited_pre']}->{confident['n_ignited_post']} i_acc={confident['i_acc']:.0f} "
              f"delivered={confident['delivered_post']}", flush=True)
        print(f"    CAUSAL: lesion n_ign {lesion['n_ignited_pre']}->{lesion['n_ignited_post']} "
              f"broadcasts={lesion_broadcasts} attribution={stn_attribution} | SIGNATURE empty={signature_empty} "
              f"| GPi_rate_during={abort['gpi_rate_during']}", flush=True)
        print(f"    anti-cheats: pulse_gated={pulse_gated} scramble_breaks={scramble_breaks_abort} "
              f"conflict_off_broadcasts={conflict_off_broadcasts} cont_no_restore={continuous_no_restore} "
              f"det={seed_deterministic}", flush=True)
        print(f"    EFFECTOR RESIDUAL (n_post vs GPi->ws strength): "
              f"{[(r['gpi_ws_w'], r['n_ignited_post']) for r in residual]} "
              f"min_n_post={min_n_post_over_sweep} ever_empty={ever_empty}", flush=True)
    return result


def run_smoke(seed, args):
    """ONE-seed diagnostic + a small grid over (current_scale, gpi_ws_w, pulse_duration) to find a working point."""
    print(f"[stn-stop smoke] seed={seed} — probing confident/conflict ignition + the stop pulse", flush=True)
    grid = []
    for cs in (args.current_scale, args.current_scale * 1.5):
        for gpi in (args.gpi_ws_w, args.gpi_ws_w + 20.0):
            for pd in (args.pulse_duration, args.pulse_duration + 20):
                r = evaluate_seed(seed, conflict_gain=args.conflict_gain, current_scale=cs, margin_ref=args.margin_ref,
                                  pulse_duration=pd, acc_stn_w=args.acc_stn_w, stn_gpi_w=args.stn_gpi_w,
                                  gpi_ws_w=gpi, heterogeneity=not args.no_heterogeneity, verbose=True)
                grid.append(r)
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_rung_stn_stop_veto_derisk", "mode": "smoke", "grid": grid}, f, indent=2)
    print(f"\n[stn-stop smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW STN->GPi hyperdirect reactive stop-signal veto de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42, help="single seed (smoke)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--conflict-gain", type=float, default=1.0)
    ap.add_argument("--current-scale", type=float, default=CONFLICT_CURRENT_SCALE)
    ap.add_argument("--margin-ref", type=float, default=MARGIN_REF)
    ap.add_argument("--pulse-duration", type=int, default=PULSE_DURATION)
    ap.add_argument("--acc-stn-w", type=float, default=ACC_STN_W)
    ap.add_argument("--stn-gpi-w", type=float, default=STN_GPI_W)
    ap.add_argument("--gpi-ws-w", type=float, default=GPI_WS_W)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1-seed grid probe to find/confirm the operating point")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_stn_stop_veto_6seed.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    print(f"[stn-stop] N_SLOTS={N_SLOTS} margin_ref={args.margin_ref:.3f} current_scale={args.current_scale} "
          f"pulse={args.pulse_duration} acc->stn={args.acc_stn_w} stn->gpi={args.stn_gpi_w} "
          f"gpi->ws={args.gpi_ws_w} het={not args.no_heterogeneity} backend={args.backend}\n", flush=True)

    if args.smoke:
        return run_smoke(args.seed, args)

    results = []
    for seed in args.seeds:
        results.append(evaluate_seed(seed, conflict_gain=args.conflict_gain, current_scale=args.current_scale,
                                     margin_ref=args.margin_ref, pulse_duration=args.pulse_duration,
                                     acc_stn_w=args.acc_stn_w, stn_gpi_w=args.stn_gpi_w, gpi_ws_w=args.gpi_ws_w,
                                     heterogeneity=not args.no_heterogeneity, verbose=True))

    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)
    any_undefined = any(r["verdict"] == "UNDEFINED" for r in results)

    summary = {
        "runner": "_gnw_rung_stn_stop_veto_derisk", "mode": "6seed",
        "seeds": list(args.seeds), "backend": args.backend,
        "all_go": bool(all_go and not any_undefined), "n_go": n_go, "n_seeds": len(results),
        "any_undefined": any_undefined,
        "operating_point": results[0]["operating_point"] if results else {},
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if summary["all_go"] else ("UNDEFINED" if any_undefined else "NO-GO")
    print(f"\n{'=' * 100}", flush=True)
    print(f"  STN->GPi REACTIVE STOP-VETO VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    for r in results:
        g = r["go_gate"]; a = r["anti_cheats"]
        print(f"    seed {r['seed']}: {r['verdict']:9s} abort={g['abort']} select={g['selectivity']} "
              f"causal={g['causal']} empty={g['signature_empty']} | scramble={a['neural_sensor_scramble_breaks_abort']} "
              f"conflict_off={a['conflict_off_broadcasts']} det={a['seed_deterministic']}", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if summary["all_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
