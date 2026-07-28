"""gap#5 candidate #3 (RANK-1 timing method) -- the RAMP-PHASE-ADVANCE replay READOUT: order the DISCRETE
forward-asymmetric stored CA3 chain by TIMING (theta phase), not by IGNITION, on the real spiking substrate. NO `sim/` edit.

2026-07-23. Spec: research/findings/2026-07-23-gap5-phase-precession-research-gate-Buzsaki-ramp-mechanism.md
(Buzsaki *Rhythms of the Brain* pp.315-320; Kamondi et al. 1998 single-cell dendritic-ramp pacemaker; Zugaro 2005;
Skaggs 1996 / Dragoi-Buzsaki 2006 theta compression). The two IGNITION readouts are banked NEGATIVE (spontaneous
bistable 1/6; DG-detonator max_ev=0 across 32 configs at 32x drive) -> per THE LAW those are verdicts on the IGNITION
METHOD, and the through-line is to READ the stored chain's ORDER as TIMING against a theta reference.

THE MECHANISM (Kamondi single-cell pacemaker; four ingredients, NO `sim/` edit):
  1. THETA-PACED perisomatic (BASKET) inhibition onto the CA3 assemblies = the PHASE REFERENCE clock ("the trough is
     the attractor"). Theta drives the CA3 INHIBITORY basket (ca3_pv_basket), NOT the excitatory cells: it quiets the
     basket at the cue phase (disinhibits the assembly) and drives it mid-cycle (re-inhibits = the reset). CRUX: the
     DECOUPLED store SPARES members from the basket (sel_inhib_spare=0.0 -> basket->member synapses zeroed, theta inert
     on the assemblies), so this readout sets sel_inhib_spare>0 so the theta-modulated basket actually REACHES the cells.
  2. A per-assembly DEPOLARIZING RAMP seeded on the FIRST assembly: a graded, SUB-THRESHOLD external current that RISES
     over ~1s of sim time (NOT a strong burst/detonator). It makes assembly-0's members oscillate slightly ahead of the
     theta trough (fire earliest in the disinhibited window); the frozen forward-asymmetric CA3 recurrent links then hand
     the depolarization FORWARD to assembly-1, which fires at the next theta phase, then assembly-2 (theta compression).
  3. INTRINSIC-FATIGUE self-avoidance on the CA3-exc slice (Izhikevich d_increment/a + the de-latch
     coincidence_plateau_self_regen=0) so the just-fired assembly self-fatigues -> forward-only sweep + transient bursts
     that re-seed each theta cycle (Ecker 2022) instead of a single latched ON state.
  4. READOUT = the PHASE ORDER of the 3 assemblies against the theta reference: each assembly's CIRCULAR-MEAN spike phase
     (weighted by per-step member firing), and whether the phases occupy MONOTONICALLY-ADVANCING positions in the correct
     FORWARD order (assembly-0 earliest, then 1, then 2, within a forward half-cycle). This is TIMING/order, NOT a discrete
     ignition event. Secondary: per-theta-cycle forward/reverse order tally via `_detect_sequence_events` (theta compression).

GO GATE (verify, don't assert -- the runner PRINTS its verdict; the caller reads THAT line):
  - phase_order_forward: all 3 assemblies phase-locked (resultant length R >= R_floor, member-spike weight > 0) AND
    lag(1) and lag(2) monotonically advance after assembly-0 within a forward half-cycle (0 < lag1 < lag2 < pi).
  - support: per-cycle forward_frac > reverse_frac + 0.15 and forward_frac >= 1.5x chance.
Anti-cheats (each WIRED AND INVOKED):
  (1) SHUFFLED-STORE (`_scramble_between_weights`: permute the between-edge multiset) -> the directional chain is gone ->
      forward phase order collapses (order is in the learned weights, not imposed by ramp/theta timing).
  (2) REVERSE-ASYMMETRY-LESION (`_symmetrize_between_weights`: flatten between-edges to the mean -> adj_fwd==adj_rev) ->
      the forward DIRECTION is destroyed -> forward phase order fails (the forward WEIGHT ASYMMETRY is load-bearing).
  (3) BASKET-OFF (no theta on the basket -> NO phase reference clock) -> the assemblies do not phase-lock (R collapses)
      -> phase order undefined/collapses (the theta-basket reference is load-bearing, not a host argmax).
  + FROZEN plasticity byte-verified across every rest phase (order rides the STORED frozen chain + the substrate's own
    theta + weights + u-fatigue, NOT rest-phase re-encoding); numpy-reference guard: NO host per-step per-assembly
    silencing / argmax inside the loop (the order emerges from the substrate).

HONEST NOTE: the ramp + theta are host-injected currents standing in for a slow dendritic depolarizing envelope + a
septal-paced FS pool (a de-risk of the READOUT; a partial/negative -- e.g. the ramp fires only assembly-0 with no
forward hand-off, or no phase-locking -- is a real, honestly-reported result diagnosing the next method).

CPU-smoke (seed 42; proves it RUNS + all controls live + produces a verdict):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_ramp_phase_advance_readout_derisk \
      --seeds 42 --n-ca3 2000 --rest-steps 2200 \
      --out research/findings/raw/gap5_r4/ramp_phase_advance_seed42.json
Fast RUNS-check (small store; NOT a verdict):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_ramp_phase_advance_readout_derisk \
      --seeds 42 --n-ca3 600 --rest-steps 700 --theta-period 100 --ramp-rise 400 --out /tmp/ramp_smoke.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
# the DECOUPLED forward-asymmetric encode (6/6-GO weight store) + the per-cycle order diagnostic + weight-lesion controls
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    _prepare_sequence, _detect_sequence_events, _scramble_between_weights, _symmetrize_between_weights,
)
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402
# the RANK-1 rest building blocks (freeze/silence/OU) reused verbatim
from research.runners._gap5_spontaneous_reactivation_derisk import _hard_silence, _configure_ou  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "ramp_phase_advance_readout.json"


# ----------------------------------------------------------------------------------------------------------------------
# _rest_ramp_phase: freeze plasticity + hard-silence (verify reset) + de-latch + crank Izhikevich adaptation, then run
# REST while (a) if basket_on, modulating the CA3 INHIBITORY BASKET with a theta oscillation (the phase-reference clock:
# disinhibit the assembly at the cue phase, re-inhibit mid-cycle), and (b) applying a rising SUB-THRESHOLD depolarizing
# RAMP onto assembly-0 (the single-cell pacemaker seed). NO burst detonator; NO host per-step per-assembly silence/argmax.
# ----------------------------------------------------------------------------------------------------------------------
def _rest_ramp_phase(prep, rest_steps, seed, *, basket_on, theta_period, theta_depth, basket_baseline,
                     ramp_max_pa, ramp_rise_steps, ramp_settle, ramp_cell_frac,
                     self_regen_read, d_abs, a_abs, adapt, verbose=False):
    """Returns dict(F, weights_frozen, apical_rest_max, apical_n_latched, basket_n, ramp_n, basket_on)."""
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    assert bridge.core_config.enable_hebbian_learning is False
    # DE-LATCH the plateau during the READ (0 = transient -> discrete bursts that re-seed each theta cycle + hand off).
    bridge.core_config.coincidence_plateau_self_regen = float(self_regen_read)

    _hard_silence(bridge)
    # DENDRITIC-RESET verification (no latched plateau at rest-start over the assembly union)
    apical_max = None; n_latched = 0
    if getattr(bridge, "cp_v_apical", None) is not None:
        va = np.asarray(to_host(bridge.cp_v_apical))[prep["ca3_arr_host"][prep["assembly_local"]]]
        apical_max = float(np.max(va)); n_latched = int((va > float(DECOUPLED_CFG["plateau_v_hold"])).sum())

    _configure_ou(bridge, None, seed)   # NO non-specific background -> the RAMP is the SOLE depolarizing source

    ca3_arr_host = prep["ca3_arr_host"]
    assemblies_local = prep["assemblies_local"]
    rm = bridge.region_manager
    exc_glob = ca3_arr_host[prep["ca3_exc_local"]]
    exc_dev = cp.asarray(exc_glob, dtype=cp.int64)

    # THE THETA TARGET (the phase reference): the CA3 INHIBITORY BASKET (ca3_pv_basket).
    basket_glob = None; basket_n = 0
    try:
        _b = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
        basket_glob = cp.asarray(_b, dtype=cp.int64); basket_n = int(len(_b))
    except Exception:
        basket_glob = None

    # crank Izhikevich spike-frequency adaptation on the CA3-exc slice (intrinsic-fatigue self-avoidance; Ecker 2022)
    if adapt and getattr(bridge, "cp_izh_d_increment", None) is not None:
        bridge.cp_izh_d_increment[exc_dev] = cp.float32(d_abs)
        bridge.cp_izh_a[exc_dev] = cp.float32(a_abs)

    # -- RAMP cell set: a subset (ramp_cell_frac) of ASSEMBLY-0's OWN cells get the rising sub-threshold depolarization. --
    a0_loc = assemblies_local[0]
    k_ramp = max(1, int(round(ramp_cell_frac * len(a0_loc))))
    rrng = np.random.default_rng(int(seed) * 77003 + 23)
    ramp_loc = np.sort(rrng.choice(a0_loc, min(k_ramp, len(a0_loc)), replace=False))
    ramp_dev = cp.asarray(ca3_arr_host[ramp_loc], dtype=cp.int64)
    k_ramp = int(len(ramp_loc))

    if verbose:
        print(f"      [ramp basket_on={basket_on} k_ramp={k_ramp} ramp_max={ramp_max_pa} rise={ramp_rise_steps} "
              f"settle={ramp_settle} theta_period={theta_period} depth={theta_depth} basket_n={basket_n} "
              f"self_regen={self_regen_read} adapt={adapt}]", flush=True)

    # FROZEN-PLASTICITY GUARD (order rides the STORED frozen chain + the substrate's own u-fatigue, NOT rest re-encoding).
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    two_pi = 2.0 * np.pi
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        # THETA on the BASKET (the phase reference). theta_mod: 0 at phase 0 (disinhibit), 1 mid-cycle (re-inhibit reset).
        if basket_on and basket_glob is not None:
            phase = (t % theta_period) / theta_period
            theta_mod = (1.0 - np.cos(two_pi * phase)) / 2.0
            # signed Tsodyks disinhibition: basket BELOW baseline at phase 0 (quiet -> assembly disinhibited), ABOVE
            # baseline mid-cycle (fires -> assembly re-inhibited = reset). With sel_inhib_spare>0 the basket->member
            # synapses exist so this modulation genuinely reaches + gates the assembly cells.
            bridge.cp_external_input_current[basket_glob] += float(basket_baseline) + float(theta_depth) * (2.0 * theta_mod - 1.0)
        # RISING SUB-THRESHOLD RAMP onto assembly-0 (the single-cell pacemaker seed; NOT a burst).
        if t >= ramp_settle:
            frac = min(1.0, (t - ramp_settle) / max(1.0, float(ramp_rise_steps)))
            bridge.cp_external_input_current[ramp_dev] += float(ramp_max_pa) * frac
        bridge._run_one_simulation_step()          # NO external per-assembly silence / argmax (numpy-reference guard)
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
    bridge.core_config.enable_ou_process = False
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    weights_frozen = bool(np.array_equal(w_before, w_after))
    return dict(F=F, weights_frozen=weights_frozen, apical_rest_max=apical_max, apical_n_latched=n_latched,
                basket_n=basket_n, ramp_n=k_ramp, basket_on=basket_on)


# ----------------------------------------------------------------------------------------------------------------------
# _phase_order: THE READOUT. Each assembly's CIRCULAR-MEAN spike phase against the theta reference (weighted by per-step
# member firing count), its resultant length R (phase-lock concentration in [0,1]), and its total member-spike weight.
# Forward-order-correctness: relative to assembly-0's mean phase, lag(k) = (mean_phase[k] - mean_phase[0]) mod 2pi; the
# order is FORWARD iff the lags monotonically advance within a forward half-cycle (0 < lag1 < lag2 < pi), with every
# assembly phase-locked (R >= R_floor, weight > 0). settle steps are excluded (pre-ramp baseline).
# ----------------------------------------------------------------------------------------------------------------------
def _phase_order(F, assemblies_local, theta_period, settle, R_floor=0.05):
    T, _ = F.shape
    ts = np.arange(T)
    ang = 2.0 * np.pi * ((ts % theta_period) / theta_period)   # per-step theta phase angle
    keep = ts >= settle
    per = []
    for A in assemblies_local:
        cnt = F[:, A].sum(1).astype(float) * keep               # per-step member firing count (post-settle)
        W = float(cnt.sum())
        if W <= 0.0:
            per.append(dict(mean_phase=None, R=0.0, weight=0.0))
            continue
        C = float((cnt * np.cos(ang)).sum()); S = float((cnt * np.sin(ang)).sum())
        mp = float(np.arctan2(S, C) % (2.0 * np.pi))
        R = float(np.hypot(C, S) / W)
        per.append(dict(mean_phase=mp, R=R, weight=W))
    # forward-order-correctness (needs all 3 locked + firing)
    locked = [p for p in per if p["mean_phase"] is not None and p["R"] >= R_floor and p["weight"] > 0.0]
    all_locked = (len(locked) == len(assemblies_local))
    lags = None; monotone = False; phase_order_forward = False
    if per[0]["mean_phase"] is not None:
        mp0 = per[0]["mean_phase"]
        lags = []
        for p in per:
            if p["mean_phase"] is None:
                lags.append(None)
            else:
                lags.append(float((p["mean_phase"] - mp0) % (2.0 * np.pi)))
        # lag[0] == 0 by construction; check lag[1], lag[2] advance forward within a half-cycle
        if all(l is not None for l in lags) and len(lags) == 3:
            monotone = (0.0 < lags[1] < lags[2] < np.pi)
        phase_order_forward = bool(all_locked and monotone)
    mean_R = float(np.mean([p["R"] for p in per]))
    n_active = int(sum(1 for p in per if p["weight"] > 0.0))
    return dict(
        per_assembly=per,
        mean_phase_cycfrac=[None if p["mean_phase"] is None else round(p["mean_phase"] / (2.0 * np.pi), 4) for p in per],
        R=[round(p["R"], 4) for p in per],
        weight=[round(p["weight"], 1) for p in per],
        lags_cycfrac=None if lags is None else [None if l is None else round(l / (2.0 * np.pi), 4) for l in lags],
        all_locked=bool(all_locked), monotone_advance=bool(monotone),
        phase_order_forward=bool(phase_order_forward), mean_R=mean_R, n_active=n_active,
    )


def _weight_diag(prep):
    return dict(w_within=prep["w_within"], w_adj_fwd=prep.get("w_adj_fwd"), w_adj_rev=prep.get("w_adj_rev"),
                ratio_adj=(float(prep.get("w_adj_fwd", 0.0)) / max(abs(float(prep.get("w_adj_rev", 0.0))), 1e-6)),
                n_between_fwd=prep.get("n_between_fwd"), n_between_rev=prep.get("n_between_rev"),
                assembly_sizes=[int(len(a)) for a in prep["assemblies"]])


def _run_arm(prep, rest_steps, seed, a, basket_on, tag):
    cp, _ = get_backend()
    r = _rest_ramp_phase(prep, rest_steps, seed, basket_on=basket_on, theta_period=a.theta_period,
                         theta_depth=a.theta_depth, basket_baseline=a.basket_baseline, ramp_max_pa=a.ramp_max_pa,
                         ramp_rise_steps=a.ramp_rise, ramp_settle=a.ramp_settle, ramp_cell_frac=a.ramp_cell_frac,
                         self_regen_read=a.self_regen_read, d_abs=a.d_abs, a_abs=a.a_abs, adapt=True,
                         verbose=(tag == "GO"))
    ph = _phase_order(r["F"], prep["assemblies_local"], a.theta_period, a.ramp_settle, R_floor=a.r_floor)
    seq = _detect_sequence_events(r["F"], prep["assemblies_local"], W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k,
                                  active_frac=a.active_frac, onset_frac=a.onset_frac)
    return r, ph, seq


def one_seed(seed, cfg, a):
    t0 = time.time()
    out = {"seed": seed}

    # -- BUILD the DECOUPLED forward-asymmetric store (with sel_inhib_spare>0 so theta-basket reaches the assemblies) --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode_decoupled"] = _weight_diag(prep)
    print(f"  [seed {seed}] DECOUPLED store: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
          f"adj_rev={prep['w_adj_rev']:.2f} ratio={out['encode_decoupled']['ratio_adj']:.2f}x "
          f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) sel_spare={cfg.get('sel_inhib_spare')} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- MAIN: theta on the basket + rising ramp on assembly-0 --
    r_go, ph_go, seq_go = _run_arm(prep, a.rest_steps, seed, a, basket_on=True, tag="GO")
    out["go"] = dict(phase=ph_go, forward_frac=seq_go["forward_frac"], reverse_frac=seq_go["reverse_frac"],
                     n_multi=seq_go["n_multi"], chance_forward=seq_go["chance_forward"], duty=seq_go["duty_cycle"],
                     per_asm_active=seq_go["per_asm_active"], pop_rate=seq_go["pop_rate"],
                     weights_frozen=r_go["weights_frozen"], basket_n=r_go["basket_n"], ramp_n=r_go["ramp_n"])
    print(f"  [seed {seed}] MAIN (theta-BASKET + ramp): phases(cyc)={ph_go['mean_phase_cycfrac']} "
          f"R={ph_go['R']} wt={ph_go['weight']} lags={ph_go['lags_cycfrac']} locked={ph_go['all_locked']} "
          f"mono={ph_go['monotone_advance']} FWDorder={ph_go['phase_order_forward']} | "
          f"per-cycle FWD={seq_go['forward_frac']:.3f} REV={seq_go['reverse_frac']:.3f} "
          f"chance={seq_go['chance_forward']:.3f} n_multi={seq_go['n_multi']} act={seq_go['per_asm_active']} "
          f"basket_n={r_go['basket_n']} frozen={r_go['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 1: SHUFFLED-STORE (fresh encode + permute the between-edge multiset) -> directional chain gone --
    prep_sh = _prepare_sequence(seed, cfg, do_encode=True)
    n_sh = _scramble_between_weights(prep_sh, seed)
    r_sh, ph_sh, seq_sh = _run_arm(prep_sh, a.rest_steps, seed, a, basket_on=True, tag="SHUFFLED")
    out["shuffled_store"] = dict(n_between_shuffled=n_sh, phase=ph_sh, forward_frac=seq_sh["forward_frac"],
                                 reverse_frac=seq_sh["reverse_frac"], n_multi=seq_sh["n_multi"],
                                 per_asm_active=seq_sh["per_asm_active"], weights_frozen=r_sh["weights_frozen"])
    print(f"  [seed {seed}] SHUFFLED-STORE ({n_sh} edges): phases={ph_sh['mean_phase_cycfrac']} R={ph_sh['R']} "
          f"lags={ph_sh['lags_cycfrac']} FWDorder={ph_sh['phase_order_forward']} | per-cycle FWD={seq_sh['forward_frac']:.3f} "
          f"REV={seq_sh['reverse_frac']:.3f} act={seq_sh['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 2: REVERSE-ASYMMETRY-LESION (fresh encode + symmetrize between-edges to the mean) -> direction gone --
    prep_sym = _prepare_sequence(seed, cfg, do_encode=True)
    n_sym = _symmetrize_between_weights(prep_sym)
    r_sym, ph_sym, seq_sym = _run_arm(prep_sym, a.rest_steps, seed, a, basket_on=True, tag="REVERSE-ASYM")
    out["reverse_asymmetry_lesion"] = dict(n_between_symmetrized=n_sym, phase=ph_sym, forward_frac=seq_sym["forward_frac"],
                                           reverse_frac=seq_sym["reverse_frac"], n_multi=seq_sym["n_multi"],
                                           per_asm_active=seq_sym["per_asm_active"], weights_frozen=r_sym["weights_frozen"])
    print(f"  [seed {seed}] REVERSE-ASYM-LESION ({n_sym} edges): phases={ph_sym['mean_phase_cycfrac']} R={ph_sym['R']} "
          f"lags={ph_sym['lags_cycfrac']} FWDorder={ph_sym['phase_order_forward']} | per-cycle FWD={seq_sym['forward_frac']:.3f} "
          f"REV={seq_sym['reverse_frac']:.3f} act={seq_sym['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 3: BASKET-OFF (no theta on the basket -> NO phase reference) -> assemblies do not phase-lock (R collapses) --
    r_bo, ph_bo, seq_bo = _run_arm(prep, a.rest_steps, seed, a, basket_on=False, tag="BASKET-OFF")
    out["basket_off"] = dict(phase=ph_bo, forward_frac=seq_bo["forward_frac"], reverse_frac=seq_bo["reverse_frac"],
                             n_multi=seq_bo["n_multi"], per_asm_active=seq_bo["per_asm_active"],
                             pop_rate=seq_bo["pop_rate"], weights_frozen=r_bo["weights_frozen"])
    print(f"  [seed {seed}] BASKET-OFF (no clock): phases={ph_bo['mean_phase_cycfrac']} R={ph_bo['R']} "
          f"mean_R={ph_bo['mean_R']:.3f} lags={ph_bo['lags_cycfrac']} FWDorder={ph_bo['phase_order_forward']} | "
          f"per-cycle FWD={seq_bo['forward_frac']:.3f} act={seq_bo['per_asm_active']} pop={seq_bo['pop_rate']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- PER-SEED VERDICT (verify, don't assert) --
    chance = max(seq_go["chance_forward"], 1e-6)
    fwd = seq_go["forward_frac"]; rev = seq_go["reverse_frac"]
    # PRIMARY: circular-mean phase order forward + all assemblies phase-locked + all 3 firing
    main_phase_order = bool(ph_go["phase_order_forward"] and ph_go["n_active"] == 3)
    # SUPPORT: per-theta-cycle order tally leans forward above chance
    per_cycle_support = (fwd > rev + 0.15 and fwd >= 1.5 * chance and seq_go["n_multi"] >= 2)
    # ANTI-CHEATS collapse
    shuffled_collapses = (not ph_sh["phase_order_forward"]) or (seq_sh["forward_frac"] <= max(0.67 * fwd, 1.5 * chance))
    reverse_collapses = (not ph_sym["phase_order_forward"]) or (seq_sym["forward_frac"] <= max(0.67 * fwd, 1.5 * chance))
    # basket-off: no phase reference -> either order fails OR the phase-lock concentration collapses vs MAIN
    basket_off_collapses = (not ph_bo["phase_order_forward"]) or (ph_bo["mean_R"] <= 0.6 * max(ph_go["mean_R"], 1e-6))
    frozen_ok = bool(r_go["weights_frozen"] and r_sh["weights_frozen"] and r_sym["weights_frozen"]
                     and r_bo["weights_frozen"])
    dendrite_reset_ok = (r_go["apical_rest_max"] is None
                         or r_go["apical_rest_max"] <= float(DECOUPLED_CFG["plateau_v_hold"]) + 1e-3)

    seed_go = bool(main_phase_order and per_cycle_support and shuffled_collapses and reverse_collapses
                   and basket_off_collapses and frozen_ok and dendrite_reset_ok)
    out["checks"] = dict(main_phase_order=main_phase_order, per_cycle_support=per_cycle_support,
                         shuffled_collapses=shuffled_collapses, reverse_collapses=reverse_collapses,
                         basket_off_collapses=basket_off_collapses, frozen_ok=frozen_ok,
                         dendrite_reset_ok=dendrite_reset_ok, n_active=ph_go["n_active"],
                         mean_R_main=round(ph_go["mean_R"], 4), mean_R_basketoff=round(ph_bo["mean_R"], 4))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000, help="the decoupled store only completes at 2000 (RANK-1 finding)")
    ap.add_argument("--n-mem", type=int, default=3)
    # THETA (the phase reference): a modest oscillatory drive onto the CA3 INHIBITORY BASKET
    ap.add_argument("--theta-period", type=int, default=125, help="steps per theta cycle; dt=1.0ms -> 125ms ~= 8 Hz")
    ap.add_argument("--theta-depth", type=float, default=400.0, help="basket theta modulation amplitude (pA)")
    ap.add_argument("--basket-baseline", type=float, default=0.0, help="tonic basket drive floor (pA)")
    # THE CRUX SWITCH: let theta-on-basket REACH the assembly cells (the DECOUPLED store spares them by default -> inert)
    ap.add_argument("--sel-inhib-spare", type=float, default=20.0,
                    help="basket->member inhibitory weight so theta-on-basket reaches the assembly cells (0.0 = inert)")
    # THE RAMP (the single-cell pacemaker seed): a rising SUB-THRESHOLD depolarization on assembly-0 (NOT a burst)
    ap.add_argument("--ramp-max-pa", type=float, default=1000.0, help="peak ramp current on assembly-0 (sub-threshold; pA)")
    ap.add_argument("--ramp-rise", type=int, default=1000, help="steps over which the ramp rises to max (~1s at dt=1.0ms)")
    ap.add_argument("--ramp-settle", type=int, default=60, help="silent settle steps before the ramp begins")
    ap.add_argument("--ramp-cell-frac", type=float, default=1.0, help="fraction of assembly-0's cells the ramp drives")
    # READOUT substrate (de-latch + cranked intrinsic-fatigue self-avoidance)
    ap.add_argument("--self-regen-read", type=float, default=0.0, help="plateau self-regen during READ (0 = transient de-latch)")
    ap.add_argument("--d-abs", type=float, default=40.0, help="cranked Izhikevich per-spike u-kick on CA3-exc (fatigue)")
    ap.add_argument("--a-abs", type=float, default=0.008, help="cranked Izhikevich recovery rate a on CA3-exc")
    ap.add_argument("--rest-steps", type=int, default=2200, help="settle + ~1s ramp rise + ~10 theta cycles for phase read")
    ap.add_argument("--r-floor", type=float, default=0.05, help="phase-lock resultant-length floor for 'locked'")
    # per-cycle order diagnostic knobs (secondary readout via _detect_sequence_events)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.4)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--active-frac", type=float, default=0.12)
    ap.add_argument("--onset-frac", type=float, default=0.08)
    # store knobs (default = the 6/6-GO DECOUPLED store; exposed so the JSON records exactly what was tested)
    ap.add_argument("--within-events", type=int, default=None)
    ap.add_argument("--within-refresh", type=int, default=None)
    ap.add_argument("--chain-fwd", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cfg = dict(DECOUPLED_CFG)
    cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = int(a.n_mem)
    cfg["sel_inhib_spare"] = float(a.sel_inhib_spare)   # CRUX: let theta-on-basket reach the assembly cells
    if a.within_events is not None:
        cfg["within_events"] = int(a.within_events)
    if a.within_refresh is not None:
        cfg["within_refresh"] = int(a.within_refresh)
    if a.chain_fwd is not None:
        cfg["chain_fwd"] = int(a.chain_fwd)

    _, backend = get_backend()
    print(f"[gap5-ramp-phase] RAMP-PHASE-ADVANCE READOUT (candidate #3, timing method) on the DECOUPLED "
          f"forward-asymmetric store (within-lr {cfg['btsp_lr']} + chain-lr {cfg['chain_btsp_lr']}) | "
          f"n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} sel_spare={cfg['sel_inhib_spare']} theta_period={a.theta_period} "
          f"depth={a.theta_depth} ramp_max={a.ramp_max_pa} ramp_rise={a.ramp_rise} ramp_frac={a.ramp_cell_frac} "
          f"self_regen={a.self_regen_read} d_abs={a.d_abs} rest_steps={a.rest_steps} seeds={a.seeds} backend={backend}",
          flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        go = n_go >= max(1, (len(per) + 1) // 2)   # smoke gate; the FULL-RUN GO bar is >=5/6 (stated in the verdict)
        # aggregate the load-bearing numbers
        mainpo = sum(1 for p in per if p["checks"]["main_phase_order"])
        mfwd = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mrev = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance_forward"] for p in per]))
        mR = float(np.mean([p["checks"]["mean_R_main"] for p in per]))
        mRbo = float(np.mean([p["checks"]["mean_R_basketoff"] for p in per]))
        nact = [p["go"]["per_asm_active"] for p in per]
        basket_ok = sum(1 for p in per if p["go"]["basket_n"] >= 1)
        if go:
            verdict = (f"RAMP-PHASE-ADVANCE GO {n_go}/{len(per)} -- a rising SUB-THRESHOLD ramp on assembly-0 + THETA on "
                       f"the CA3 basket reads the stored forward-asymmetric chain as MONOTONICALLY-ADVANCING theta "
                       f"phases in the correct forward order (phase-order-forward on {mainpo}/{len(per)}; per-cycle "
                       f"forward_frac {mfwd:.3f} vs reverse {mrev:.3f} vs chance {mch:.3f}; mean phase-lock R {mR:.3f}). "
                       f"SHUFFLED-STORE + REVERSE-ASYM-LESION collapse the forward order, BASKET-OFF collapses the "
                       f"phase-lock (R {mRbo:.3f} << {mR:.3f}). => the TIMING readout surpasses the ignition roadblock; "
                       f"run the full 6-seed (bar >=5/6).")
        elif basket_ok >= 1:
            verdict = (f"HONEST NEGATIVE {n_go}/{len(per)} -- the theta-basket reference is live (basket_n>=1 on "
                       f"{basket_ok}/{len(per)}) and the ramp drives assembly-0, but the ramp-phase-advance readout did "
                       f"NOT produce clean forward-ordered phases (phase-order-forward {mainpo}/{len(per)}; per-asm "
                       f"active {nact}; per-cycle forward_frac {mfwd:.3f} vs chance {mch:.3f}; mean R {mR:.3f}). "
                       f"Per THE LAW this is a verdict on THIS timing-method's operating point, NOT on the phase-order "
                       f"CAPABILITY -- read the per-asm-active + R numbers to scope the next move (weak forward hand-off "
                       f"-> stronger sel_inhib_spare / adj-fwd, or a per-cycle re-seed ramp, or a wider theta window).")
        else:
            verdict = (f"INCONCLUSIVE {n_go}/{len(per)} -- the CA3 basket target was not found (basket_n=0); the store "
                       f"must build ca3_pv_basket (ca3_fb_inhib set). Re-check n_ca3/DECOUPLED_CFG before concluding.")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_ramp_phase_advance_readout", "mechanism": "candidate #3 Kamondi ramp single-cell pacemaker",
               "GO": go, "n_go": n_go, "seeds": a.seeds,
               "decoupled_cfg": {k: cfg[k] for k in sorted(cfg)},          # every store knob recorded
               "ramp_theta_cfg": dict(theta_period=a.theta_period, theta_depth=a.theta_depth,
                                      basket_baseline=a.basket_baseline, sel_inhib_spare=a.sel_inhib_spare,
                                      ramp_max_pa=a.ramp_max_pa, ramp_rise=a.ramp_rise, ramp_settle=a.ramp_settle,
                                      ramp_cell_frac=a.ramp_cell_frac, self_regen_read=a.self_regen_read,
                                      d_abs=a.d_abs, a_abs=a.a_abs, n_ca3=a.n_ca3, n_mem=a.n_mem,
                                      rest_steps=a.rest_steps, r_floor=a.r_floor, window=a.window, ev_floor=a.ev_floor,
                                      ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[gap5-ramp-phase] VERDICT: {verdict}\n[gap5-ramp-phase] wrote {a.out}\n" + "=" * 118,
          flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
