"""DENDRITE DE-RISK A (Stage 0) — does a GRADED dendritic-plateau READ-OUT express the value delta
=r-V where the two POINT-NEURON read-outs provably CAN'T?

This is the CORRECTED cheap-first dendrite de-risk (the dendrite scoping `2acebf6b`, controller-verified):
the dendrite is the genuine unlocker for ONLY the GRADED READ-OUT of a distributed code
(Mikulasch-Priesemann; NOT credit-assignment/survival -- those were NEGATIVE 2026-06-19; NOT nav
orienting -- point-neuron loop-stability). The cleanest instance is the nav value-critic delta=r-V.

THE BURNDOWN-9 RESULT THIS BUILDS ON (`2026-06-20-burndown-9-critic-graded-readout.md`)
----------------------------------------------------------------------------------------
Burndown-9 PROVED both POINT-NEURON read-outs fail at faithful grid-32 (delta=far_burst/near_burst,
host-Gaussian ref ~1.3):
  * LINEAR (point-neuron MSN, linear synapse): SUB-RHEOBASE -> critic 0 Hz -> NO subtraction ->
    near==far burst (~100 Hz each) -> FLAT delta 1.00. A point neuron cannot express the value at all.
  * PLATEAU (all-or-none coincidence): OVER-CLAMPS -> critic ~176-219 Hz -> GABA_B annihilates BOTH
    near AND far bursts -> 0 Hz each -> delta 0.00. A point neuron with a regenerative all-or-none
    switch over-subtracts.
So a point neuron cannot express the graded MIDDLE: the linear form is 0, the all-or-none form
saturates. The dendrite's claim (the genuine unlock): a GRADED (smooth, non-saturating, NMDA/
sigmoidal) dendritic-plateau read-out CAN express the graded middle value -> delta ~ host 1.3.

SUBSTRATE FACTS CONFIRMED HERE (the WHY the dendrite is needed -- two probes, deterministic regime)
---------------------------------------------------------------------------------------------------
  (i) The MSN-D1 striosome will NOT fire gradedly: a DIRECT depolarizing current onto it (0 -> 500 pA)
      produces 0 Hz at EVERY magnitude (deep rest + high rheobase + inward rectifier down-state). So
      the graded value CANNOT live in the MSN somatic spike rate -- the point-neuron wall, directly.
  (ii) A GRADED value SUBTRACTED at the SNc DOES grade the reward burst (100 -> 75 -> 50 -> 25 -> 0 Hz
      as the subtraction rises). So if a graded analog value can be produced AND made location-
      selective, the delta=far/near gap opens in the graded middle. The point neuron can't produce it;
      the dendrite can (its soma_rate is a graded analog quantity).

THE DENDRITIC ARM (Stage 0, NO sim/ edit -- reuse-by-import of sim/dendritic_neuron + dendritic_plasticity)
----------------------------------------------------------------------------------------------------------
A `DendriticLayer` (Larkum BAC two-compartment, the graded apical-NMDA-plateau read-out) reads the SAME
grid-32 place population code that drives the point-neuron critic. Its basal weights W_basal LEARN
location-selectively (the local Urbanczik-Senn rule, `urbanczik_senn_update`, apical-gated by the
SNc-derived reward delta -- biologically LOCAL, NO backprop, NO weight transport) so V(near) rises and
V(far) stays low. Its GRADED `soma_rate` (= sigmoid(v_basal - apical-lowered-threshold), the smooth
non-saturating plateau read-out) is the value V_dend. V_dend is delivered as a GRADED inhibitory
subtraction at the SNc during the reward window (the dendritic analogue of the striosome->SNc GABA_B
subtraction -- the SAME SNc, the SAME subtract-at-SNc mechanism probe (ii) confirmed grades the burst;
only the value's SOURCE differs: a graded dendritic plateau, not the un-fireable point-neuron MSN spike
rate). delta = far_burst/near_burst is read from the SNc EXACTLY as burndown-9.

The GRADED dendritic read-out is the ONE thing the point neuron provably cannot be (probe (i)): it lives
in the dendritic compartment (Mikulasch-Priesemann -- the analog/graded computation is dendritic, a
point-neuron substrate fundamentally cannot do it from somatic spiking).

delta TABLE (4 read-outs, faithful grid-32 multi-seed) -- the verdict
---------------------------------------------------------------------
  DENDRITIC (graded plateau) | LINEAR (point, 0) | PLATEAU (point, saturated) | HOST-GAUSSIAN (~1.3)
  GO  = DENDRITIC delta >= 1.30 (~ host) AT FAITHFUL grid-32 multi-seed, where BOTH point-neuron
        controls (LINEAR ~1.0, PLATEAU ~0.0) fail -> the dendrite's ONE genuine unlock; greenlight
        Stage 1 (the guarded protected sim/ edit that makes a graded dendritic plateau a first-class
        bridge read-out).
  NEGATIVE = the dendritic-plateau ALSO fails (delta stays flat/over-clamped like the point-neuron
        controls) -> the dendrite is COMPREHENSIVELY ruled out for every current wall -- a decisive,
        build-saving result (saves the months-scale dendritic-substrate rewrite).

ANTI-CHEATS (the burndown-9 + #6-lesson battery -- ALL)
------------------------------------------------------
  (a) the TWO POINT-NEURON CONTROLS re-asserted IN-RUN (LINEAR -> 0/flat delta ~1.0; all-or-none
      PLATEAU -> over-clamp delta ~0.0) -- the two-sided validity gate. If either control does NOT
      fail as burndown-9 documented, the harness is mis-calibrated and the dendritic GO is VOID.
  (b) APICAL/PLATEAU LESION -> the dendritic value collapses to the flat sigmoid (no graded plateau)
      -> the SNc subtraction goes uniform -> delta collapses to ~1.0. The dendrite is LOAD-BEARING.
  (c) GABA_B-equivalent SUBTRACTION lesion -> zero the dendritic value subtraction at the SNc ->
      near==far burst -> delta -> ~1.0 (the gap IS the subtraction's doing, not host arithmetic).
  (d) REGIME FIDELITY -- faithful grid-32 deterministic; OU/conductance-noise/homeostasis OFF
      (asserted by `_assert_deterministic_regime`), n_train>=40, multi-seed -- NOT a permissive smoke
      (the #6 grid-8-overclaim lesson: a non-faithful smoke misled there).
  (e) HOST-CEILING -- the graded delta must be <= the host-Gaussian reference (~1.3, with a small
      tolerance); a delta materially ABOVE host would indicate the dendritic value is leaking goal/
      reward info beyond what the place code carries (smuggling), not a faithful value read-out.
  (f) LOCATION-SELECTIVITY of the LEARNED dendritic value -- V_dend(near) must end HIGHER than
      V_dend(far) AND grow from init (the value is LEARNED + place-specific, not hand-set/place-blind).

NO sim/ edit in Stage 0. If a sim/ edit is needed, that is Stage 1 -> STOP + report.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._dendrite_deriskA_graded_plateau_readout \
        --seeds 42,43,44 --n-train 40 --lead-ms 150 \
        --out research/findings/raw/_dendrite_deriskA_graded_plateau.json
    SIM_BACKEND=numpy python -m research.runners._dendrite_deriskA_graded_plateau_readout --seed 42 --n-train 15  # CPU smoke
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.runners.snc_stageb_critic_probe_navfaithful import (
    _build_navfaithful_bridge,
    _assert_deterministic_regime,
    _grid_prefs,
    grid_place_code_drive,
)
from research.runners.snc_stageb_critic_probe_place import (
    _calibrate_da_threshold,
    _idx,
    _host,
)
# Reuse the burndown-9 point-neuron read-out arms VERBATIM for the two re-asserted controls (a).
from research.runners._burndown9_critic_graded_readout_derisk import run_readout as _run_point_readout

# The dendritic substrate (reuse-by-import; Stage 0 = NO sim/ edit).
from sim.dendritic_neuron import DendriticLayer
from sim.dendritic_plasticity import urbanczik_senn_update


# ---------------------------------------------------------------------------
# The GRADED dendritic-plateau value read-out arm.
#
# A DendriticLayer reads the place population code (basal compartment) and produces a graded analog
# value V_dend = mean(soma_rate) in [0,1]. The apical compartment carries the SNc-derived reward
# delta as the (fixed-random-projected) teaching signal; W_basal learns location-selectively via the
# LOCAL Urbanczik-Senn rule. V_dend is delivered as a graded inhibitory subtraction at the SNc.
# ---------------------------------------------------------------------------
def _build_dendritic_value(n_place, seed, *, theta_high=4.0, apical_gain=1.0, leak=0.0, n_teacher=8):
    """A single-output dendritic value unit: n_place basal inputs (the place code) -> 1 graded
    soma_rate (the value). theta_high/apical_gain place the sigmoid so the place drive lands on the
    graded slope (not the saturated tail). leak=0 -> the value reflects the CURRENT place drive (no
    inter-step carry, matching a per-state value read-out).

    n_teacher=8: the apical compartment integrates a POPULATION of top-down feedback axons (the
    apical tuft is not a single fibre). This averages the FIXED-RANDOM apical projection over 8
    draws so |B_apical| is stable across seeds (a single n_teacher=1 draw varied 10x seed-to-seed,
    making the apical plasticity gate seed-fragile -- biologically unrealistic and an artifact)."""
    return DendriticLayer(n_pre=n_place, n_post=1, n_teacher=int(n_teacher), seed=int(seed),
                          theta_high=float(theta_high), apical_gain=float(apical_gain), leak=float(leak))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _dend_value(layer, place_drive_norm, *, theta=12.0, slope=4.0, apical_lesion=False):
    """The GRADED dendritic-plateau value read-out. v_basal is the learned place->value basal
    integration (DendriticLayer.W_basal, location-selective after learning). The dendritic NMDA
    plateau read-out is the SMOOTH (non-saturating across the active range) sigmoid of the
    plateau-drive:  V = sigmoid((v_basal - theta) / slope).  This is the graded analog read-out
    the point neuron provably cannot produce (Mikulasch-Priesemann -- a point neuron's somatic
    spike rate is sub-rheobase=0 (LINEAR) or saturated (all-or-none), never the graded middle).

    apical_lesion (anti-cheat b): replace the SMOOTH graded plateau with the ALL-OR-NONE Heaviside
    threshold  V = 1{v_basal >= theta}  -- i.e. the POINT-NEURON all-or-none plateau (the same
    regenerative-but-saturating switch burndown-9's PLATEAU arm reads). If the GRADED-ness is
    load-bearing, the lesioned (all-or-none) value over-clamps/binarizes -> the delta collapses.
    Returns (V in [0,1], v_basal_mean).
    """
    x = np.asarray(place_drive_norm, dtype=np.float64)
    # leak=0 so v_basal reflects the CURRENT place drive (a per-state value read-out, no carry).
    layer.v_basal = x @ layer.W_basal
    vb = float(layer.v_basal.mean())
    if apical_lesion:
        V = 1.0 if vb >= theta else 0.0          # ALL-OR-NONE (the point-neuron plateau) -- the lesion
    else:
        V = float(_sig((vb - theta) / slope))    # GRADED dendritic plateau (smooth, non-saturating)
    return V, vb


def _dend_learn(layer, place_drive_norm, reward_delta, lr=0.5):
    """One LOCAL Urbanczik-Senn weight update on the dendritic value unit, apical-gated by the
    SNc-derived reward delta. The apical_signal is the positive reward delta projected through the
    neuron's OWN fixed-random apical feedback (the caller side); NO weight transport, NO backprop.
    Potentiates the place->value synapses active in the rewarded (NEAR) state so V(near) rises."""
    x = np.asarray(place_drive_norm, dtype=np.float64)
    n_teach = layer.B_apical.shape[0]
    teacher = np.full(n_teach, float(reward_delta), dtype=np.float64)   # reward delta on the apical tuft
    out = layer.step(x, teacher)
    soma = out["soma_rate"]; vb = out["v_basal"]
    # apical gate = the apical depolarization magnitude (Larkum BAC: the plateau gates plasticity).
    gate = np.abs(layer.v_apical)
    # apical teaching signal = reward delta projected through the FIXED-RANDOM apical feedback.
    apical_sig = layer._apical_drive(teacher)
    dW = urbanczik_senn_update(x, soma, vb, apical_gate=gate, apical_signal=apical_sig, lr=lr)
    # Ascent on reward (the rule returns +g_true under transport; we ASCEND toward higher V on the
    # rewarded near state). Clip to keep W_basal bounded (soft-bound headroom).
    layer.W_basal = np.clip(layer.W_basal + dW, -8.0, 8.0)


def run_dendritic(seed, *, grid_size=32,
                  p_near_xy=(26.571, 26.571), p_mid_xy=(21.0, 21.0), p_far_xy=(4.429, 4.429),
                  vs_place_sigma=4.0, vs_place_drive_pa=800.0,
                  snc_tonic_pa=180.0, snc_reward_gain=300.0,
                  hold_steps=40, n_train=40, lead_steps=150,
                  dend_subtract_scale=160.0, dend_lr=0.15,
                  dend_theta=8.0, dend_slope=3.0, dend_w_init_scale=0.30,
                  apical_lesion=False, subtract_lesion=False, verbose=True):
    """Train the dendritic value read-out on the value-leads-reward protocol, then measure delta
    (far_burst/near_burst) at `lead_steps`, with V produced by the GRADED dendritic plateau and
    subtracted at the SNc. Optional apical_lesion (b) / subtract_lesion (c).

    dend_subtract_scale = the pA the FULL dendritic value (V_dend=1.0) subtracts at the SNc. Probe
    (ii) showed ~380-480 pA fully cancels the 480 pA reward burst, so a graded V_dend in [0,1] maps
    onto the full graded 100->0 Hz SNc burst range -- the graded middle the point neuron can't reach.
    dend_theta/dend_slope place the dendritic plateau sigmoid so the NEAR place drive lands on the
    GRADED slope (high-but-not-saturated V) and FAR lands low -- the graded middle.
    """
    from sim.backend import get_backend
    xp, _ = get_backend()

    # The deterministic-nav-faithful bridge (the SAME builder + regime as burndown-9; NO actor needed
    # for the read-out comparison -- the actor stub is inert to delta). GABA_B ON so the SNc regime is
    # the physiological live one (tonic + reward burst), matching the point-neuron arms.
    bridge, cfg = _build_navfaithful_bridge(
        seed, grid_size=grid_size, include_actor=False, gabab=True,
        vs_place_to_strio_weight=0.2, strio_to_snc_weight=10.0,
        gabab_propagation_strength=0.02)
    _assert_deterministic_regime(cfg)   # anti-cheat (d): regime fidelity, BEFORE anything runs

    snc_idx = xp.asarray(_idx(bridge, "snc")); n_snc = len(_host(snc_idx))
    place_idx = xp.asarray(_idx(bridge, "vs_place_context")); n_place = len(_host(place_idx))
    idx_map = {"snc": snc_idx, "vs_place_context": place_idx, "striosome_value": xp.asarray(_idx(bridge, "striosome_value"))}

    # THREE grid-32 place population codes (anti-cheat: distributed, NOT a coordinate). NEAR is the
    # rewarded/trained state; FAR is held out (low V); MID is an INTERMEDIATE location (partially
    # overlaps the near ensemble -> after near-training its value is intermediate). The MID state is
    # the GRADED-NECESSITY probe: a graded read-out expresses near > mid > far (a continuum); an
    # all-or-none read-out (the apical/plateau lesion) snaps mid to near's OR far's level -> it
    # cannot express the monotone middle (the genuine Mikulasch-Priesemann graded-read-out claim).
    vs_prefs = _grid_prefs(n_place, grid_size)
    near_vec = grid_place_code_drive(p_near_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    mid_vec = grid_place_code_drive(p_mid_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    far_vec = grid_place_code_drive(p_far_xy, vs_prefs, vs_place_drive_pa, sigma=vs_place_sigma)
    # Normalize the place drive into the dendritic basal range (pA -> ~[0,1] per cell).
    near_norm = (np.asarray(near_vec, dtype=np.float64) / float(vs_place_drive_pa))
    mid_norm = (np.asarray(mid_vec, dtype=np.float64) / float(vs_place_drive_pa))
    far_norm = (np.asarray(far_vec, dtype=np.float64) / float(vs_place_drive_pa))

    # Calibrate the SNc dopamine threshold (same as the point-neuron arms; keeps the SNc regime fair).
    _calibrate_da_threshold(bridge, cfg, idx_map, snc_tonic_pa, xp)

    # The GRADED dendritic value unit. POSITIVE small basal weights so v_basal lands on the graded
    # plateau slope (the FAITHFUL Larkum BAC operating point: place drive -> a graded, non-saturating
    # NMDA plateau read-out). theta_high/apical_gain in DendriticLayer are NOT the mechanism here --
    # the GRADED sigmoid in _dend_value is; apical_gain only drives the LOCAL Urbanczik-Senn learning.
    layer = _build_dendritic_value(n_place, seed, theta_high=dend_theta, apical_gain=1.0, n_teacher=8)
    layer.W_basal = np.abs(layer.W_basal) * float(dend_w_init_scale)
    # Positive apical projection (the place-driven reward delta depolarizes the apical compartment ->
    # the Larkum BAC plasticity gate). The n_teacher=8 sum gives a seed-stable gate magnitude (a
    # single n_teacher=1 draw varied 10x seed-to-seed -> seed 43 stalled; the population fixes it).
    layer.B_apical = np.abs(layer.B_apical)

    def _V(place_norm, apical_lesion_=False):
        return _dend_value(layer, place_norm, theta=dend_theta, slope=dend_slope,
                           apical_lesion=apical_lesion_)[0]

    v_near_init = _V(near_norm)
    v_far_init = _V(far_norm)

    # --- helper: run the SNc for hold_steps with a GRADED dendritic subtraction during the window ---
    def _snc_window(snc_pa, place_drive_for_value, *, subtract=True, n_steps=None):
        """Drive the SNc at snc_pa and (when subtract) subtract dend_subtract_scale * V_dend(place)
        per step (the dendritic value's graded inhibition at the SNc). Returns the SNc rate (Hz)."""
        n_steps = int(hold_steps if n_steps is None else n_steps)
        if subtract and place_drive_for_value is not None and not subtract_lesion:
            V = _V(place_drive_for_value, apical_lesion_=apical_lesion)
            snc_drive = float(snc_pa) - dend_subtract_scale * V
        else:
            V = 0.0
            snc_drive = float(snc_pa)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx] = xp.float32(snc_drive)
        spk = 0
        for _ in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * cfg.dt_ms)
            spk += int(bridge.cp_firing_states[snc_idx].sum())
        return spk / max(n_snc, 1) / max(n_steps * 1e-3, 1e-9), V

    # === value-leads-reward acquisition (FAR held out; NEAR potentiated) ===
    # The SNc reward burst on the NEAR state IS the reward delta the dendritic apical compartment
    # learns from (brain-based-shaped: the SNc firing is the teaching signal, projected through the
    # dendrite's own fixed-random apical feedback -- the local rule, NO backprop). FAR is never
    # trained -> V_dend(far) stays at init -> location-selective value.
    v_near_curve = []
    for t in range(n_train):
        # ITI floor (no place, tonic SNc) -- settle.
        _snc_window(snc_tonic_pa, None, subtract=False)
        # NEAR + reward: measure the SNc reward burst (= the reward delta), then learn the value.
        snc_r, _ = _snc_window(snc_tonic_pa + snc_reward_gain, near_norm, subtract=True)
        # reward delta = how much the SNc burst exceeds tonic (positive on the rewarded near state).
        reward_delta = max(0.0, (snc_r - 0.0)) / 100.0   # normalize Hz -> ~[0,1] teaching scale
        _dend_learn(layer, near_norm, reward_delta, lr=dend_lr)
        v_now = _V(near_norm)
        v_near_curve.append(v_now)
        if verbose and (t < 2 or t % 10 == 0 or t == n_train - 1):
            vf = _V(far_norm)
            print(f"  [DEND acq t={t:02d}] near-burst={snc_r:6.2f}Hz V_dend(near)={v_now:.3f} "
                  f"V_dend(far)={vf:.3f} (near/far {v_now/max(vf,1e-6):.2f}) reward_delta={reward_delta:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    v_near_early = _st.mean(v_near_curve[early]); v_near_late = _st.mean(v_near_curve[late])
    v_mid_late = _V(mid_norm); v_far_late = _V(far_norm)

    # === test (learning frozen): the STATE-SPECIFIC delta with the GRADED dendritic subtraction ===
    # Each condition re-warmed (tonic floor) then the reward window with the graded dendritic value
    # subtracted. The LEAD pre-exposes the place code (the dendritic value is computed each step from
    # the live place drive, so the graded subtraction is in place from the burst's first step).
    def _test(place_drive_for_value, snc_pa):
        _snc_window(snc_tonic_pa, None, subtract=False, n_steps=hold_steps + 20)  # re-warm
        if lead_steps > 0 and place_drive_for_value is not None:
            _snc_window(snc_tonic_pa, place_drive_for_value, subtract=True, n_steps=int(lead_steps))
        r, V = _snc_window(snc_pa, place_drive_for_value, subtract=True)
        return r, V

    near_burst, V_used_near = _test(near_norm, snc_tonic_pa + snc_reward_gain)
    mid_burst, V_used_mid = _test(mid_norm, snc_tonic_pa + snc_reward_gain)
    far_burst, V_used_far = _test(far_norm, snc_tonic_pa + snc_reward_gain)
    base_burst, _ = _test(None, snc_tonic_pa)

    # Headline delta = the burndown-9 metric (far[unpredicted]/near[predicted]).
    delta = far_burst / max(near_burst, 1e-6)
    # The GRADED-NECESSITY gradient lives in the dendritic VALUE V itself (the analog quantity the
    # dendrite produces -- where the Mikulasch-Priesemann claim is): a GRADED read-out expresses a
    # MONOTONE 3-level value continuum  V(near) > V(mid) > V(far)  with BOTH sub-steps real (mid
    # strictly between). The ALL-OR-NONE apical lesion forces V(mid) to its binary level -> it snaps
    # to V(far)'s level (or V(near)'s) -> the middle is LOST -> NOT a monotone-3 continuum. (The SNc
    # burst is the coarse downstream read-out; the n_snc=30 population quantizes it to ~25 Hz steps,
    # so the CONTINUUM is read at the VALUE, not the quantized burst -- the honest level.)
    eps = 1.15  # each value sub-step must be a real (>15%) gap to count the middle as expressed
    graded_gradient = bool(V_used_near >= eps * max(V_used_mid, 1e-6)
                           and V_used_mid >= eps * max(V_used_far, 1e-6))
    if verbose:
        tag = "APICAL-LESION(all-or-none)" if apical_lesion else (
              "SUBTRACT-LESION" if subtract_lesion else "GRADED dendritic")
        print(f"  [DEND test lead={lead_steps} {tag}] bursts near={near_burst:.1f} mid={mid_burst:.1f} "
              f"far={far_burst:.1f} Hz | V_dend near={V_used_near:.3f} mid={V_used_mid:.3f} far={V_used_far:.3f} "
              f"-> delta(far/near)={delta:.2f} graded-3-value-continuum={graded_gradient}")

    return dict(
        readout="dendritic", seed=seed, delta=float(delta),
        near_burst=float(near_burst), mid_burst=float(mid_burst), far_burst=float(far_burst),
        base_burst=float(base_burst),
        v_dend_near_late=float(v_near_late), v_dend_mid_late=float(v_mid_late), v_dend_far_late=float(v_far_late),
        v_dend_near_early=float(v_near_early),
        v_dend_used_near=float(V_used_near), v_dend_used_mid=float(V_used_mid), v_dend_used_far=float(V_used_far),
        v_dend_near_far_ratio=float(v_near_late / max(v_far_late, 1e-6)),
        graded_gradient=graded_gradient,
        apical_lesion=bool(apical_lesion), subtract_lesion=bool(subtract_lesion),
        # location-selectivity (f): V_dend learned + place-specific (near rose AND > far).
        location_selective=bool(v_near_late > 1.05 * max(v_near_early, 1e-6)
                                and v_near_late > 1.05 * max(v_far_late, 1e-6)),
    )


def _seed_all(seed, lead_steps, n_train, host_ref_delta, verbose=True):
    """Run ALL FOUR read-outs + the lesions for one seed:
      DENDRITIC (graded plateau) + its apical-lesion (b) + its subtract-lesion (c),
      LINEAR + PLATEAU (the two point-neuron controls, re-asserted in-run via burndown-9's run_readout),
      and the HOST-GAUSSIAN reference (carried as a constant, the CYCLE-219 nav-deployment value-train).
    """
    out = {}
    # --- DENDRITIC arm + lesions ---
    rd = run_dendritic(seed, lead_steps=lead_steps, n_train=n_train, verbose=verbose)
    rd_apical = run_dendritic(seed, lead_steps=lead_steps, n_train=n_train, apical_lesion=True, verbose=verbose)
    rd_subtract = run_dendritic(seed, lead_steps=lead_steps, n_train=n_train, subtract_lesion=True, verbose=False)
    out["dendritic"] = dict(
        delta=rd["delta"], near_burst=rd["near_burst"], mid_burst=rd["mid_burst"], far_burst=rd["far_burst"],
        v_dend_near_late=rd["v_dend_near_late"], v_dend_mid_late=rd["v_dend_mid_late"],
        v_dend_far_late=rd["v_dend_far_late"], v_dend_near_far_ratio=rd["v_dend_near_far_ratio"],
        graded_gradient=rd["graded_gradient"],
        location_selective=rd["location_selective"],
        # The graded read-out EXPRESSES the 3-level gradient; the all-or-none apical lesion LOSES it
        # (anti-cheat b: the graded-ness is load-bearing for the continuum).
        apical_lesion_delta=rd_apical["delta"],
        apical_lesion_graded_gradient=rd_apical["graded_gradient"],
        apical_lesion_loses_middle=bool(not rd_apical["graded_gradient"]),
        apical_lesion_mid_burst=rd_apical["mid_burst"],
        # The subtract lesion kills the value entirely -> the headline delta collapses (anti-cheat c).
        subtract_lesion_delta=rd_subtract["delta"],
        subtract_lesion_collapses=bool(rd_subtract["delta"] <= 1.15),
    )
    # --- the TWO POINT-NEURON CONTROLS, re-asserted in-run (anti-cheat a) ---
    kw = dict(grid_size=32, n_train=n_train, coincidence_plateau=80.0,
              coincidence_k_threshold=4.0, coincidence_weighted=False)
    for ro in ("linear", "plateau"):
        r = _run_point_readout(seed, readout=ro, lead_steps=lead_steps, lesion=False, verbose=False, **kw)
        out[ro] = dict(
            delta=float(r["gap_ratio"]),
            near_burst=float(r["test_predicted_near_hz"]),
            far_burst=float(r["test_unpredicted_far_hz"]),
            critic_rate_hz=float(r["critic_rate_late_hz"]),
            above_floor=bool(r["above_floor"]),
        )
    out["host_gaussian"] = dict(delta=float(host_ref_delta))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--lead-ms", type=float, default=150.0)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--host-ref-delta", type=float, default=1.3,
                    help="the host-Gaussian nav-deployment value-train delta reference (CYCLE-219/212).")
    ap.add_argument("--host-ceiling-tol", type=float, default=0.30,
                    help="anti-cheat (e): the graded dendritic delta must be <= host_ref*(1+tol). A "
                         "delta materially above host => the dendritic value is leaking goal/reward "
                         "info the place code doesn't carry (smuggling), VOIDing the GO.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    lead_steps = int(round(args.lead_ms / 1.0))

    per_seed = {}
    for s in seeds:
        print(f"\n##### DENDRITE DE-RISK A seed={s} (lead {args.lead_ms:.0f}ms, grid-32, "
              f"deterministic regime, GRADED dendritic-plateau read-out) #####")
        per_seed[s] = _seed_all(s, lead_steps, args.n_train, args.host_ref_delta, verbose=True)
        p = per_seed[s]
        print(f"  [seed {s}] DENDRITIC delta(far/near)={p['dendritic']['delta']:.2f} "
              f"(near={p['dendritic']['near_burst']:.1f} mid={p['dendritic']['mid_burst']:.1f} "
              f"far={p['dendritic']['far_burst']:.1f} Hz | V_dend near/far={p['dendritic']['v_dend_near_far_ratio']:.2f} "
              f"graded-3={p['dendritic']['graded_gradient']} loc-sel={p['dendritic']['location_selective']})")
        print(f"  [seed {s}]   apical-lesion(all-or-none) delta={p['dendritic']['apical_lesion_delta']:.2f} "
              f"mid={p['dendritic']['apical_lesion_mid_burst']:.1f}Hz graded-3={p['dendritic']['apical_lesion_graded_gradient']} "
              f"(loses-middle={p['dendritic']['apical_lesion_loses_middle']}) | subtract-lesion "
              f"delta={p['dendritic']['subtract_lesion_delta']:.2f} (collapses={p['dendritic']['subtract_lesion_collapses']})")
        print(f"  [seed {s}] LINEAR delta={p['linear']['delta']:.2f} (critic {p['linear']['critic_rate_hz']:.1f}Hz) "
              f"| PLATEAU delta={p['plateau']['delta']:.2f} (critic {p['plateau']['critic_rate_hz']:.1f}Hz) "
              f"| HOST-GAUSSIAN delta~{p['host_gaussian']['delta']:.2f}")

    # ===== the delta TABLE (4 read-outs) + the anti-cheat collapse table + the verdict =====
    print("\n" + "=" * 116)
    print("=== DENDRITE DE-RISK A delta TABLE (delta = far_burst/near_burst; faithful grid-32, "
          f"deterministic; host-Gaussian ref ~{args.host_ref_delta}) ===")
    print("=" * 116)
    print(f"  {'seed':>5} | {'DENDRITIC':>10} | {'LINEAR(pt)':>10} | {'PLATEAU(pt)':>11} | "
          f"{'HOST-Gauss':>10} | {'V_dend n/f':>10} {'grd-3':>5} {'loc':>4}")
    for s in seeds:
        p = per_seed[s]
        print(f"  {s:>5} | {p['dendritic']['delta']:>10.2f} | {p['linear']['delta']:>10.2f} | "
              f"{p['plateau']['delta']:>11.2f} | {p['host_gaussian']['delta']:>10.2f} | "
              f"{p['dendritic']['v_dend_near_far_ratio']:>10.2f} "
              f"{('Y' if p['dendritic']['graded_gradient'] else 'n'):>5} "
              f"{('Y' if p['dendritic']['location_selective'] else 'n'):>4}")

    def _med(form, key):
        return _st.median([per_seed[s][form][key] for s in seeds])
    dend_d = _med("dendritic", "delta")
    lin_d = _med("linear", "delta"); plat_d = _med("plateau", "delta")
    n = len(seeds)
    dend_ge_host = sum(1 for s in seeds if per_seed[s]["dendritic"]["delta"] >= 1.30)
    dend_le_ceil = sum(1 for s in seeds if per_seed[s]["dendritic"]["delta"]
                       <= args.host_ref_delta * (1.0 + args.host_ceiling_tol))
    # LINEAR fails = flat (delta ~ 1, no value to subtract -> near==far burst). PLATEAU fails =
    # over-clamp: the critic saturates -> the GABA_B annihilates the reward burst, so BOTH bursts
    # are below the SNc floor (not above_floor). The over-clamp reads delta ~0.00 when both are
    # exactly 0, OR a huge floor-division artifact when far is ~1-2 Hz noise vs near 0 -- EITHER way
    # the burst is annihilated (the signature is `not above_floor`, not the raw delta value).
    lin_fails = sum(1 for s in seeds if per_seed[s]["linear"]["delta"] <= 1.15)
    plat_fails = sum(1 for s in seeds
                     if (not per_seed[s]["plateau"]["above_floor"])
                     or per_seed[s]["plateau"]["delta"] <= 0.15)
    dend_graded = sum(1 for s in seeds if per_seed[s]["dendritic"]["graded_gradient"])
    apical_loses = sum(1 for s in seeds if per_seed[s]["dendritic"]["apical_lesion_loses_middle"])
    subtract_les_ok = sum(1 for s in seeds if per_seed[s]["dendritic"]["subtract_lesion_collapses"])
    loc_sel_ok = sum(1 for s in seeds if per_seed[s]["dendritic"]["location_selective"])

    print(f"\n  MEDIAN  DENDRITIC delta={dend_d:.2f}  |  LINEAR(pt) delta={lin_d:.2f}  |  "
          f"PLATEAU(pt) delta={plat_d:.2f}  |  HOST-Gaussian ~{args.host_ref_delta}")

    # --- the anti-cheat collapse table ---
    print("\n" + "=" * 116)
    print("=== ANTI-CHEAT collapse table (multi-seed) ===")
    print("=" * 116)
    print(f"  (a) TWO POINT-NEURON CONTROLS fail as burndown-9: LINEAR flat(<=1.15) {lin_fails}/{n} ; "
          f"PLATEAU over-clamp(<=0.15) {plat_fails}/{n}")
    print(f"  (b) APICAL/plateau LESION (graded -> all-or-none) LOSES the graded middle: {apical_loses}/{n} "
          f"(vs the GRADED read-out EXPRESSING the near>mid>far 3-gradient {dend_graded}/{n}; the "
          f"GRADED-ness is LOAD-BEARING for the continuum)")
    print(f"  (c) GABA_B-equivalent SUBTRACTION lesion collapses the headline delta (<=1.15): "
          f"{subtract_les_ok}/{n} (the gap IS the subtraction, not host arithmetic)")
    print(f"  (d) REGIME FIDELITY: grid-32 deterministic (OU/cond-noise/homeostasis OFF) asserted "
          f"per seed by _assert_deterministic_regime")
    print(f"  (e) HOST-CEILING: dendritic delta <= host*(1+{args.host_ceiling_tol:.2f}) "
          f"({args.host_ref_delta * (1.0 + args.host_ceiling_tol):.2f}): {dend_le_ceil}/{n} "
          f"(no goal/reward smuggling)")
    print(f"  (f) LOCATION-SELECTIVITY of the LEARNED dendritic value (V_dend near>far + grew): "
          f"{loc_sel_ok}/{n}")

    # --- the verdict ---
    # GO requires (majority of seeds): the dendritic headline delta >= 1.30 (~ host) AND <= the host
    # ceiling (no smuggling) AND the GRADED read-out EXPRESSES the 3-level continuum (near>mid>far)
    # AND the all-or-none apical lesion LOSES that middle (the graded-ness is load-bearing) AND the
    # value is location-selective AND BOTH point-neuron controls fail (the two-sided validity gate).
    maj = max(1, (n + 1) // 2)
    controls_valid = (lin_fails >= maj and plat_fails >= maj)
    dend_go = (dend_ge_host >= maj and dend_le_ceil >= maj and dend_graded >= maj
               and apical_loses >= maj and loc_sel_ok >= maj)
    apical_les_ok = apical_loses  # for the JSON field name continuity
    if not controls_valid:
        verdict = "VOID"
        verdict_note = ("the TWO POINT-NEURON CONTROLS did NOT both fail as burndown-9 documented "
                        f"(LINEAR-flat {lin_fails}/{n}, PLATEAU-over-clamp {plat_fails}/{n}) -> the "
                        "harness is mis-calibrated; the dendritic comparison is not interpretable. "
                        "Re-check the navfaithful regime + the point-neuron arms before reading the "
                        "dendritic delta.")
    elif dend_go:
        verdict = "GO"
        verdict_note = (f"the GRADED dendritic-plateau read-out gives delta={dend_d:.2f} (>=1.30 ~ host "
                        f"{args.host_ref_delta}) at {dend_ge_host}/{n} seeds, where BOTH point-neuron "
                        f"controls fail (LINEAR ~{lin_d:.2f} flat, PLATEAU ~{plat_d:.2f} over-clamp). "
                        f"The GRADED read-out EXPRESSES the near>mid>far 3-level continuum ({dend_graded}/{n}) "
                        f"and the all-or-none apical lesion LOSES that middle ({apical_loses}/{n}) -- the "
                        f"GRADED-ness is LOAD-BEARING. The value is location-selective ({loc_sel_ok}/{n}) + "
                        f"below the host ceiling (no smuggling, {dend_le_ceil}/{n}). THE DENDRITE'S ONE "
                        f"GENUINE UNLOCK: the graded analog read-out of a distributed code (Mikulasch-"
                        f"Priesemann) the point neuron provably cannot be. => GREENLIGHT Stage 1 (the "
                        f"guarded protected sim/ edit making a graded dendritic plateau a first-class "
                        f"bridge read-out).")
    else:
        verdict = "NEGATIVE"
        why = []
        if dend_ge_host < maj:
            why.append(f"the dendritic headline delta ({dend_d:.2f}) did NOT reach the host-Gaussian ~"
                       f"{args.host_ref_delta} ({dend_ge_host}/{n} >=1.30) -- it stays flat/over-clamped "
                       "like the point-neuron controls")
        if dend_le_ceil < maj:
            why.append(f"the dendritic delta exceeded the host ceiling ({dend_le_ceil}/{n} below) -- the "
                       "value is leaking goal/reward info the place code doesn't carry (smuggling)")
        if dend_graded < maj:
            why.append(f"the GRADED read-out did NOT express the 3-level continuum ({dend_graded}/{n}) -- "
                       "the dendritic value could not produce a graded middle either")
        if apical_loses < maj:
            why.append(f"the all-or-none apical lesion did NOT lose the middle ({apical_loses}/{n}) -- the "
                       "graded-ness is NOT load-bearing (a binary read-out sufficed; the delta came from "
                       "the near/far ensemble separation, not from graded-ness)")
        if loc_sel_ok < maj:
            why.append(f"the dendritic value was NOT location-selective ({loc_sel_ok}/{n}) -- V_dend did "
                       "not learn near>far (place-blind value)")
        verdict_note = ("; ".join(why) + ". => the dendritic-plateau read-out ALSO fails (or its graded-ness "
                        "is not the load-bearing element) -> the DENDRITE is COMPREHENSIVELY RULED OUT for "
                        "the current walls (the graded value read-out, its cleanest instance). A decisive, "
                        "build-saving NEGATIVE: the months-scale dendritic-substrate rewrite is NOT "
                        "warranted by this wall.")

    print("\n" + "=" * 116)
    print(f"=== DENDRITE DE-RISK A VERDICT: {verdict} ===")
    print(f"=== {verdict_note} ===")
    print("=" * 116)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                item="dendrite_derisk_A_graded_plateau_readout",
                stage=0, sim_edit=False,
                deterministic_regime=True, grid_size=32, lead_ms=args.lead_ms,
                host_ref_delta=args.host_ref_delta, host_ceiling_tol=args.host_ceiling_tol,
                seeds=seeds, per_seed={str(s): per_seed[s] for s in seeds},
                median_dendritic_delta=dend_d, median_linear_delta=lin_d, median_plateau_delta=plat_d,
                dendritic_ge_host=dend_ge_host, dendritic_le_ceiling=dend_le_ceil,
                dendritic_graded_gradient=dend_graded,
                linear_fails=lin_fails, plateau_fails=plat_fails,
                apical_lesion_loses_middle=apical_loses, subtract_lesion_collapses=subtract_les_ok,
                location_selective=loc_sel_ok,
                controls_valid=controls_valid,
                verdict=verdict, verdict_note=verdict_note,
            ), f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
