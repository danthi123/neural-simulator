"""gap#5 candidate #3 DENDRITIC version -- the RAMP-PHASE-ADVANCE replay READOUT with a real APICAL COMPARTMENT.

2026-07-24. This is the DENDRITIC sibling of `_gap5_ramp_phase_advance_readout_derisk.py` (the POINT-neuron version
that FAILED: R~=0.05, no phase-locking; finding
`2026-07-24-gap5-candidate3-kamondi-ramp-6seed-NEGATIVE-point-neuron-limit.md`). The point-neuron negative diagnosed
the Kamondi (1998) single-cell pacemaker as intrinsically DENDRITIC: a ~1s APICAL depolarizing ramp drives a
voltage-dependent dendritic oscillation slightly FASTER than the somatic theta, which advances the spike phase each
cycle. A point soma cannot host that oscillation; a TWO-COMPARTMENT cell can.

THE ONE HYPOTHESIS THIS TESTS: does giving each CA3 cell a real APICAL compartment (the already-guarded
`enable_two_compartment_dap` two-compartment dAP substrate, which the DECOUPLED store ALREADY builds) + applying the
~1s ramp to the APICAL (not the soma) + KEEPING the dendritic bistability ON during the read (self_regen=0.15, NOT the
point-neuron de-latch self_regen=0) restore phase-locking (R substantially above 0.05) and a monotonic forward phase
order? If yes -> the point-neuron-limit hypothesis is CONFIRMED (the dendrite is load-bearing). If no -> an honest
negative that says precisely which dendritic sub-mechanism failed.

WHAT IS ADAPTED FROM THE POINT-NEURON RUNNER (everything else is IDENTICAL):
  1. self_regen_read: 0.0 (point-neuron de-latch) -> 0.15 (the 2026-07-18 completion GO value -> the apical stays
     BISTABLE-competent during the read, so it can host the voltage-dependent plateau/oscillation dynamics).
  2. ramp TARGET: the SOMA (cp_external_input_current) -> the APICAL compartment (cp_v_apical). The ramp is injected as
     an apical current-equivalent each step: cp_v_apical[ramp] += (dt/apical_tau) * ramp_apical_mv * frac, which is
     exactly `+ R*I_ext` in the apical ODE `dv` (bridge.py:7171) -- so R cancels and `ramp_apical_mv` IS the nominal
     peak steady-state apical depolarization (mV) the ramp drives. NO `sim/` EDIT (the apical ODE already integrates
     cp_v_apical every step when the store's coincidence two-comp block runs; we only pre-add the drive from the runner).
  The DECOUPLED forward-asymmetric store (within~208, adj_fwd~38, adj_rev~5), the theta-paced BASKET reference, the
  forward-asymmetric CA3 links, the intrinsic-fatigue self-avoidance, and ALL anti-cheats are UNCHANGED.

The DECOUPLED store dendritic operating point (== the 2026-07-18 completion GO_CFG, the starting point per the research
gate): plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=1.0 (soma->apical back-coupling, weak), apical_gc_read=5.0
(apical->soma read, strong asymmetric), apical_R=50.0, plateau_v_hold=-35.0, apical_E_rest=-65.0. These are set by the
store builder (`_build(..., two_comp=True, ...)`); this runner does NOT change them (only self_regen during the read is
restored to 0.15 instead of the point-neuron 0).

GO GATE (verify, don't assert -- the runner PRINTS its verdict; the caller reads THAT line):
  - main_phase_order: all 3 assemblies phase-locked (per-assembly resultant R >= r_floor=0.15, i.e. SUBSTANTIALLY above
    the point-neuron 0.05) AND lag(1),lag(2) monotonically advance forward (0 < lag1 < lag2 < pi), all 3 firing.
  - support: per-theta-cycle forward_frac >= 1.5x chance AND forward_frac > reverse_frac + 0.2.
Anti-cheats (each WIRED AND INVOKED; reused from the point-neuron runner):
  (1) SHUFFLED-STORE (permute the between-edge multiset) -> the directional chain is gone -> forward phase order collapses.
  (2) REVERSE-ASYMMETRY-LESION (flatten between-edges to the mean, adj_fwd==adj_rev) -> the forward DIRECTION is destroyed.
  (3) BASKET-OFF (no theta on the basket -> NO phase reference clock) -> the assemblies do not phase-lock (R collapses).
  (4) NO-CUE (the apical ramp OFF -> no pacemaker seed on assembly-0) -> no forward-ordered lead / phase order collapses.
  (5) FROZEN plasticity byte-verified across every rest phase (order rides the STORED frozen chain + the substrate's own
      theta + weights + dendritic dynamics, NOT rest re-encoding).
  + DENDRITIC (the confirmation control): POINT-NEURON arm (self_regen=0 + ramp-on-SOMA == the point-neuron runner) ->
    R stays ~0.05 (the dendrite is load-bearing). This is the SCIENTIFIC confirmation of the point-neuron-limit hypothesis.

HONEST NOTE: the ramp + theta are host-injected currents standing in for a slow dendritic depolarizing envelope + a
septal-paced FS pool (a de-risk of the READOUT). NO host-computed phase ordering; the phase advance must EMERGE from the
apical dendritic dynamics + the neural basket reset. A partial/negative is a real, honestly-reported result diagnosing
the next dendritic sub-mechanism.

Seed-42 GPU smoke (the run this file was built for):
  SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._gap5_ramp_dendritic_phase_advance_derisk \
      --seeds 42 --n-ca3 2000 --rest-steps 2200 \
      --out research/findings/raw/gap5_r4/ramp_dendritic_seed42.json
CPU RUNS-check (small store; NOT a verdict):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_ramp_dendritic_phase_advance_derisk \
      --seeds 42 --n-ca3 600 --rest-steps 700 --theta-period 100 --ramp-rise 400 --out /tmp/ramp_dend_smoke.json
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
# the READOUT (circular-mean phase order) + the store weight-diag, reused VERBATIM from the point-neuron runner
from research.runners._gap5_ramp_phase_advance_readout_derisk import _phase_order, _weight_diag  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "ramp_dendritic_phase_advance_readout.json"


# ----------------------------------------------------------------------------------------------------------------------
# _rest_ramp_dendritic_phase: the DENDRITIC read. IDENTICAL to the point-neuron `_rest_ramp_phase` EXCEPT (1) the
# dendritic bistability is KEPT ON during the read (self_regen_read=0.15, not the point-neuron 0) so the apical stays
# BISTABLE-competent, and (2) the rising ~1s ramp is injected into the APICAL compartment cp_v_apical (the single-cell
# pacemaker seed), not the soma. `ramp_target="soma"` + self_regen_read=0 reproduces the point-neuron control arm.
# ----------------------------------------------------------------------------------------------------------------------
def _rest_ramp_dendritic_phase(prep, rest_steps, seed, *, basket_on, ramp_on, ramp_target,
                               theta_period, theta_depth, basket_baseline,
                               ramp_apical_mv, ramp_soma_pa, ramp_rise_steps, ramp_settle, ramp_cell_frac,
                               self_regen_read, d_abs, a_abs, adapt, verbose=False):
    """Returns dict(F, weights_frozen, apical_rest_max, apical_n_latched, basket_n, ramp_n, basket_on, two_comp,
    has_apical, apical_read_max, apical_read_mean, ramp_target)."""
    cp, _ = get_backend()
    bridge = prep["bridge"]
    cfg = bridge.core_config
    cfg.enable_hebbian_learning = False
    assert cfg.enable_hebbian_learning is False
    # KEEP the dendritic bistability ON during the read (self_regen_read=0.15 = the completion GO value). For the
    # point-neuron control arm the caller passes self_regen_read=0.0 (transient de-latch) + ramp_target="soma".
    cfg.coincidence_plateau_self_regen = float(self_regen_read)

    two_comp = bool(getattr(cfg, "enable_two_compartment_dap", False))
    _hard_silence(bridge)
    # DENDRITIC-RESET verification (no latched plateau at rest-start over the assembly union)
    apical_max = None; n_latched = 0
    has_apical = getattr(bridge, "cp_v_apical", None) is not None
    if has_apical:
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

    # -- RAMP cell set: a subset (ramp_cell_frac) of ASSEMBLY-0's OWN cells get the rising depolarization. --
    a0_loc = assemblies_local[0]
    k_ramp = max(1, int(round(ramp_cell_frac * len(a0_loc))))
    rrng = np.random.default_rng(int(seed) * 77003 + 23)
    ramp_loc = np.sort(rrng.choice(a0_loc, min(k_ramp, len(a0_loc)), replace=False))
    ramp_dev = cp.asarray(ca3_arr_host[ramp_loc], dtype=cp.int64)
    k_ramp = int(len(ramp_loc))

    # apical injection scaling: v_apical += (dt/apical_tau) * ramp_apical_mv * frac == `+R*I_ext` in the apical ODE dv,
    # so R cancels and ramp_apical_mv is the nominal PEAK steady-state apical depolarization (mV). (bridge.py:7171/7184.)
    apical_tau = float(getattr(cfg, "apical_tau_ms", 15.0))
    dt_ms = float(cfg.dt_ms)
    apical_inject_per_mv = dt_ms / max(apical_tau, 1e-6)

    if verbose:
        print(f"      [dend-ramp target={ramp_target} ramp_on={ramp_on} basket_on={basket_on} k_ramp={k_ramp} "
              f"apical_mv={ramp_apical_mv} soma_pa={ramp_soma_pa} rise={ramp_rise_steps} settle={ramp_settle} "
              f"theta_period={theta_period} depth={theta_depth} basket_n={basket_n} self_regen={self_regen_read} "
              f"two_comp={two_comp} has_apical={has_apical} adapt={adapt}]", flush=True)

    # FROZEN-PLASTICITY GUARD (order rides the STORED frozen chain + the substrate's own dendritic dynamics, NOT rest re-encode)
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    two_pi = 2.0 * np.pi
    # track the apical DEPOLARIZATION over the ramp cells (diagnostic: did the ramp actually depolarize the apical?)
    apical_read_max = -1e9; apical_read_sum = 0.0; apical_read_n = 0
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        # THETA on the BASKET (the phase reference). theta_mod: 0 at phase 0 (disinhibit), 1 mid-cycle (re-inhibit reset).
        if basket_on and basket_glob is not None:
            phase = (t % theta_period) / theta_period
            theta_mod = (1.0 - np.cos(two_pi * phase)) / 2.0
            bridge.cp_external_input_current[basket_glob] += float(basket_baseline) + float(theta_depth) * (2.0 * theta_mod - 1.0)
        # RISING RAMP (the single-cell pacemaker seed). APICAL (dendritic) by default; SOMA for the point-neuron control.
        if ramp_on and t >= ramp_settle:
            frac = min(1.0, (t - ramp_settle) / max(1.0, float(ramp_rise_steps)))
            if ramp_target == "apical" and has_apical:
                bridge.cp_v_apical[ramp_dev] += cp.float32(apical_inject_per_mv * float(ramp_apical_mv) * frac)
            else:  # "soma" (the point-neuron control) -- exactly the point-neuron runner's ramp
                bridge.cp_external_input_current[ramp_dev] += float(ramp_soma_pa) * frac
        bridge._run_one_simulation_step()          # NO external per-assembly silence / argmax (numpy-reference guard)
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
        if has_apical and t >= ramp_settle:
            _va = np.asarray(to_host(bridge.cp_v_apical))[ca3_arr_host[ramp_loc]]
            apical_read_max = max(apical_read_max, float(_va.max())); apical_read_sum += float(_va.mean()); apical_read_n += 1
    bridge.core_config.enable_ou_process = False
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    weights_frozen = bool(np.array_equal(w_before, w_after))
    return dict(F=F, weights_frozen=weights_frozen, apical_rest_max=apical_max, apical_n_latched=n_latched,
                basket_n=basket_n, ramp_n=k_ramp, basket_on=basket_on, two_comp=two_comp, has_apical=has_apical,
                apical_read_max=(None if apical_read_n == 0 else round(apical_read_max, 2)),
                apical_read_mean=(None if apical_read_n == 0 else round(apical_read_sum / apical_read_n, 2)),
                ramp_target=ramp_target)


def _run_arm(prep, rest_steps, seed, a, *, basket_on, ramp_on, ramp_target, self_regen_read, tag):
    r = _rest_ramp_dendritic_phase(
        prep, rest_steps, seed, basket_on=basket_on, ramp_on=ramp_on, ramp_target=ramp_target,
        theta_period=a.theta_period, theta_depth=a.theta_depth, basket_baseline=a.basket_baseline,
        ramp_apical_mv=a.ramp_apical_mv, ramp_soma_pa=a.ramp_soma_pa, ramp_rise_steps=a.ramp_rise,
        ramp_settle=a.ramp_settle, ramp_cell_frac=a.ramp_cell_frac, self_regen_read=self_regen_read,
        d_abs=a.d_abs, a_abs=a.a_abs, adapt=(not a.no_adapt), verbose=(tag == "GO"))
    ph = _phase_order(r["F"], prep["assemblies_local"], a.theta_period, a.ramp_settle, R_floor=a.r_floor)
    seq = _detect_sequence_events(r["F"], prep["assemblies_local"], W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k,
                                  active_frac=a.active_frac, onset_frac=a.onset_frac)
    return r, ph, seq


def _arm_line(seed, name, r, ph, seq, t0):
    print(f"  [seed {seed}] {name}: phases(cyc)={ph['mean_phase_cycfrac']} R={ph['R']} wt={ph['weight']} "
          f"lags={ph['lags_cycfrac']} locked={ph['all_locked']} mono={ph['monotone_advance']} "
          f"FWDorder={ph['phase_order_forward']} meanR={ph['mean_R']:.3f} | per-cycle FWD={seq['forward_frac']:.3f} "
          f"REV={seq['reverse_frac']:.3f} chance={seq['chance_forward']:.3f} n_multi={seq['n_multi']} "
          f"act={seq['per_asm_active']} apical(mean/max)={r['apical_read_mean']}/{r['apical_read_max']} "
          f"basket_n={r['basket_n']} frozen={r['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)


def one_seed(seed, cfg, a):
    t0 = time.time()
    out = {"seed": seed}

    # -- BUILD the DECOUPLED forward-asymmetric store (with sel_inhib_spare>0 so theta-basket reaches the assemblies) --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode_decoupled"] = _weight_diag(prep)
    _two = bool(getattr(prep["bridge"].core_config, "enable_two_compartment_dap", False))
    print(f"  [seed {seed}] DECOUPLED store: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
          f"adj_rev={prep['w_adj_rev']:.2f} ratio={out['encode_decoupled']['ratio_adj']:.2f}x "
          f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) sel_spare={cfg.get('sel_inhib_spare')} "
          f"two_comp={_two} ({time.time()-t0:.0f}s)", flush=True)
    out["two_compartment_dap"] = _two

    # -- MAIN (DENDRITIC): theta on the basket + rising ramp on assembly-0's APICAL + dendritic bistability ON --
    r_go, ph_go, seq_go = _run_arm(prep, a.rest_steps, seed, a, basket_on=True, ramp_on=True, ramp_target="apical",
                                   self_regen_read=a.self_regen_read, tag="GO")
    out["go"] = dict(phase=ph_go, forward_frac=seq_go["forward_frac"], reverse_frac=seq_go["reverse_frac"],
                     n_multi=seq_go["n_multi"], chance_forward=seq_go["chance_forward"], duty=seq_go["duty_cycle"],
                     per_asm_active=seq_go["per_asm_active"], pop_rate=seq_go["pop_rate"],
                     weights_frozen=r_go["weights_frozen"], basket_n=r_go["basket_n"], ramp_n=r_go["ramp_n"],
                     apical_read_mean=r_go["apical_read_mean"], apical_read_max=r_go["apical_read_max"],
                     apical_rest_max=r_go["apical_rest_max"], two_comp=r_go["two_comp"], has_apical=r_go["has_apical"])
    _arm_line(seed, "MAIN dendritic (theta-BASKET + APICAL ramp)", r_go, ph_go, seq_go, t0)

    # -- CONTROL 1: SHUFFLED-STORE (fresh encode + permute the between-edge multiset) -> directional chain gone --
    prep_sh = _prepare_sequence(seed, cfg, do_encode=True)
    n_sh = _scramble_between_weights(prep_sh, seed)
    r_sh, ph_sh, seq_sh = _run_arm(prep_sh, a.rest_steps, seed, a, basket_on=True, ramp_on=True, ramp_target="apical",
                                   self_regen_read=a.self_regen_read, tag="SHUFFLED")
    out["shuffled_store"] = dict(n_between_shuffled=n_sh, phase=ph_sh, forward_frac=seq_sh["forward_frac"],
                                 reverse_frac=seq_sh["reverse_frac"], n_multi=seq_sh["n_multi"],
                                 per_asm_active=seq_sh["per_asm_active"], weights_frozen=r_sh["weights_frozen"])
    _arm_line(seed, f"SHUFFLED-STORE ({n_sh} edges)", r_sh, ph_sh, seq_sh, t0)

    # -- CONTROL 2: REVERSE-ASYMMETRY-LESION (fresh encode + symmetrize between-edges to the mean) -> direction gone --
    prep_sym = _prepare_sequence(seed, cfg, do_encode=True)
    n_sym = _symmetrize_between_weights(prep_sym)
    r_sym, ph_sym, seq_sym = _run_arm(prep_sym, a.rest_steps, seed, a, basket_on=True, ramp_on=True, ramp_target="apical",
                                      self_regen_read=a.self_regen_read, tag="REVERSE-ASYM")
    out["reverse_asymmetry_lesion"] = dict(n_between_symmetrized=n_sym, phase=ph_sym, forward_frac=seq_sym["forward_frac"],
                                           reverse_frac=seq_sym["reverse_frac"], n_multi=seq_sym["n_multi"],
                                           per_asm_active=seq_sym["per_asm_active"], weights_frozen=r_sym["weights_frozen"])
    _arm_line(seed, f"REVERSE-ASYM-LESION ({n_sym} edges)", r_sym, ph_sym, seq_sym, t0)

    # -- CONTROL 3: BASKET-OFF (no theta on the basket -> NO phase reference) -> assemblies do not phase-lock --
    r_bo, ph_bo, seq_bo = _run_arm(prep, a.rest_steps, seed, a, basket_on=False, ramp_on=True, ramp_target="apical",
                                   self_regen_read=a.self_regen_read, tag="BASKET-OFF")
    out["basket_off"] = dict(phase=ph_bo, forward_frac=seq_bo["forward_frac"], reverse_frac=seq_bo["reverse_frac"],
                             n_multi=seq_bo["n_multi"], per_asm_active=seq_bo["per_asm_active"],
                             pop_rate=seq_bo["pop_rate"], weights_frozen=r_bo["weights_frozen"])
    _arm_line(seed, "BASKET-OFF (no clock)", r_bo, ph_bo, seq_bo, t0)

    # -- CONTROL 4: NO-CUE (the APICAL ramp OFF -> no pacemaker seed on assembly-0) -> no forward-ordered lead --
    r_nc, ph_nc, seq_nc = _run_arm(prep, a.rest_steps, seed, a, basket_on=True, ramp_on=False, ramp_target="apical",
                                   self_regen_read=a.self_regen_read, tag="NO-CUE")
    out["no_cue"] = dict(phase=ph_nc, forward_frac=seq_nc["forward_frac"], reverse_frac=seq_nc["reverse_frac"],
                         n_multi=seq_nc["n_multi"], per_asm_active=seq_nc["per_asm_active"],
                         pop_rate=seq_nc["pop_rate"], weights_frozen=r_nc["weights_frozen"])
    _arm_line(seed, "NO-CUE (apical ramp off)", r_nc, ph_nc, seq_nc, t0)

    # -- CONTROL 5 (the DENDRITE-is-load-bearing confirmation): POINT-NEURON arm (self_regen=0 + ramp-on-SOMA == the
    #    point-neuron runner). If the dendritic MAIN locks (R high) but this stays R~=0.05, the apical is load-bearing. --
    r_pt, ph_pt, seq_pt = _run_arm(prep, a.rest_steps, seed, a, basket_on=True, ramp_on=True, ramp_target="soma",
                                   self_regen_read=0.0, tag="POINT-NEURON")
    out["point_neuron_control"] = dict(phase=ph_pt, forward_frac=seq_pt["forward_frac"], reverse_frac=seq_pt["reverse_frac"],
                                       n_multi=seq_pt["n_multi"], per_asm_active=seq_pt["per_asm_active"],
                                       pop_rate=seq_pt["pop_rate"], weights_frozen=r_pt["weights_frozen"])
    _arm_line(seed, "POINT-NEURON control (self_regen=0, ramp-on-SOMA)", r_pt, ph_pt, seq_pt, t0)

    # -- PER-SEED VERDICT (verify, don't assert) --
    chance = max(seq_go["chance_forward"], 1e-6)
    fwd = seq_go["forward_frac"]; rev = seq_go["reverse_frac"]
    # PRIMARY: circular-mean phase order forward + all assemblies phase-locked (R>=r_floor, substantially above 0.05) + all 3 firing
    main_phase_order = bool(ph_go["phase_order_forward"] and ph_go["n_active"] == 3)
    R_substantially_above_point = bool(ph_go["mean_R"] >= a.r_go_floor)      # mean-R substantially above the point-neuron 0.055
    # SUPPORT: per-theta-cycle order tally leans forward above chance
    per_cycle_support = bool(fwd > rev + 0.2 and fwd >= 1.5 * chance and seq_go["n_multi"] >= 2)
    # ANTI-CHEATS collapse
    shuffled_collapses = bool((not ph_sh["phase_order_forward"]) or (seq_sh["forward_frac"] <= max(0.67 * fwd, 1.5 * chance)))
    reverse_collapses = bool((not ph_sym["phase_order_forward"]) or (seq_sym["forward_frac"] <= max(0.67 * fwd, 1.5 * chance)))
    basket_off_collapses = bool((not ph_bo["phase_order_forward"]) or (ph_bo["mean_R"] <= 0.6 * max(ph_go["mean_R"], 1e-6)))
    nocue_collapses = bool(not ph_nc["phase_order_forward"])
    point_neuron_collapses = bool((not ph_pt["phase_order_forward"]) or (ph_pt["mean_R"] <= 0.6 * max(ph_go["mean_R"], 1e-6)))
    frozen_ok = bool(r_go["weights_frozen"] and r_sh["weights_frozen"] and r_sym["weights_frozen"]
                     and r_bo["weights_frozen"] and r_nc["weights_frozen"] and r_pt["weights_frozen"])
    dendrite_reset_ok = (r_go["apical_rest_max"] is None
                         or r_go["apical_rest_max"] <= float(DECOUPLED_CFG["plateau_v_hold"]) + 1e-3)
    apical_present = bool(r_go["two_comp"] and r_go["has_apical"])

    seed_go = bool(main_phase_order and R_substantially_above_point and per_cycle_support
                   and shuffled_collapses and reverse_collapses and basket_off_collapses and nocue_collapses
                   and point_neuron_collapses and frozen_ok and dendrite_reset_ok and apical_present)
    out["checks"] = dict(main_phase_order=main_phase_order, R_substantially_above_point=R_substantially_above_point,
                         per_cycle_support=per_cycle_support, shuffled_collapses=shuffled_collapses,
                         reverse_collapses=reverse_collapses, basket_off_collapses=basket_off_collapses,
                         nocue_collapses=nocue_collapses, point_neuron_collapses=point_neuron_collapses,
                         frozen_ok=frozen_ok, dendrite_reset_ok=dendrite_reset_ok, apical_present=apical_present,
                         n_active=ph_go["n_active"], mean_R_main=round(ph_go["mean_R"], 4),
                         mean_R_basketoff=round(ph_bo["mean_R"], 4), mean_R_nocue=round(ph_nc["mean_R"], 4),
                         mean_R_pointneuron=round(ph_pt["mean_R"], 4))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  mean_R: main={ph_go['mean_R']:.3f} "
          f"pt-neuron={ph_pt['mean_R']:.3f} basket-off={ph_bo['mean_R']:.3f} no-cue={ph_nc['mean_R']:.3f}  "
          f"checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000, help="the decoupled store only completes at 2000 (RANK-1 finding)")
    ap.add_argument("--n-mem", type=int, default=3)
    # THETA (the phase reference): a modest oscillatory drive onto the CA3 INHIBITORY BASKET (unchanged from point-neuron)
    ap.add_argument("--theta-period", type=int, default=125, help="steps per theta cycle; dt=1.0ms -> 125ms ~= 8 Hz")
    ap.add_argument("--theta-depth", type=float, default=400.0, help="basket theta modulation amplitude (pA)")
    ap.add_argument("--basket-baseline", type=float, default=0.0, help="tonic basket drive floor (pA)")
    ap.add_argument("--sel-inhib-spare", type=float, default=20.0,
                    help="basket->member inhibitory weight so theta-on-basket reaches the assembly cells (0.0 = inert)")
    # THE APICAL RAMP (the single-cell pacemaker seed): a rising depolarization on assembly-0's APICAL compartment.
    ap.add_argument("--ramp-apical-mv", type=float, default=30.0,
                    help="nominal PEAK steady-state APICAL depolarization the ramp drives (mV; -65 rest, -35 v_hold)")
    ap.add_argument("--ramp-soma-pa", type=float, default=1000.0,
                    help="peak SOMA ramp current for the point-neuron control arm (pA; == the point-neuron runner)")
    ap.add_argument("--ramp-rise", type=int, default=1000, help="steps over which the ramp rises to max (~1s at dt=1.0ms)")
    ap.add_argument("--ramp-settle", type=int, default=60, help="silent settle steps before the ramp begins")
    ap.add_argument("--ramp-cell-frac", type=float, default=1.0, help="fraction of assembly-0's cells the ramp drives")
    # READOUT substrate: the dendritic bistability KEPT ON (self_regen_read=0.15 = completion GO) + cranked fatigue.
    ap.add_argument("--self-regen-read", type=float, default=0.15,
                    help="plateau self-regen during READ (0.15 = completion GO, the apical stays bistable; 0 = point-neuron de-latch)")
    ap.add_argument("--no-adapt", action="store_true", help="disable the cranked Izhikevich intrinsic-fatigue self-avoidance")
    ap.add_argument("--d-abs", type=float, default=40.0, help="cranked Izhikevich per-spike u-kick on CA3-exc (fatigue)")
    ap.add_argument("--a-abs", type=float, default=0.008, help="cranked Izhikevich recovery rate a on CA3-exc")
    ap.add_argument("--rest-steps", type=int, default=2200, help="settle + ~1s ramp rise + ~10 theta cycles for phase read")
    ap.add_argument("--r-floor", type=float, default=0.15,
                    help="per-assembly phase-lock resultant-length floor for 'locked' (0.15 = SUBSTANTIALLY above the point-neuron 0.05)")
    ap.add_argument("--r-go-floor", type=float, default=0.15,
                    help="mean-R gate: the MAIN arm's mean phase-lock R must be >= this (substantially above the point-neuron 0.055)")
    # per-cycle order diagnostic knobs (secondary readout via _detect_sequence_events; unchanged)
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
    print(f"[gap5-ramp-DENDRITIC] APICAL-RAMP-PHASE-ADVANCE READOUT (candidate #3 DENDRITIC; two-compartment dAP) on the "
          f"DECOUPLED forward-asymmetric store | n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} "
          f"sel_spare={cfg['sel_inhib_spare']} theta_period={a.theta_period} depth={a.theta_depth} "
          f"ramp_apical_mv={a.ramp_apical_mv} ramp_rise={a.ramp_rise} ramp_frac={a.ramp_cell_frac} "
          f"self_regen_read={a.self_regen_read} d_abs={a.d_abs} rest_steps={a.rest_steps} r_floor={a.r_floor} "
          f"seeds={a.seeds} backend={backend}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        go = n_go >= max(1, (len(per) + 1) // 2)   # smoke gate; the FULL-RUN GO bar is >=5/6 (stated in the verdict)
        mainpo = sum(1 for p in per if p["checks"]["main_phase_order"])
        mfwd = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mrev = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance_forward"] for p in per]))
        mR = float(np.mean([p["checks"]["mean_R_main"] for p in per]))
        mRbo = float(np.mean([p["checks"]["mean_R_basketoff"] for p in per]))
        mRpt = float(np.mean([p["checks"]["mean_R_pointneuron"] for p in per]))
        nact = [p["go"]["per_asm_active"] for p in per]
        apic = sum(1 for p in per if p["checks"]["apical_present"])
        basket_ok = sum(1 for p in per if p["go"]["basket_n"] >= 1)
        if not (apic >= 1):
            verdict = (f"INCONCLUSIVE {n_go}/{len(per)} -- the two-compartment APICAL substrate was NOT present "
                       f"(apical_present {apic}/{len(per)}); the DECOUPLED store did not build enable_two_compartment_dap "
                       f"+ cp_v_apical. Re-check _build(two_comp=True) before concluding.")
        elif go:
            verdict = (f"DENDRITIC RAMP-PHASE-ADVANCE GO {n_go}/{len(per)} -- with a real APICAL compartment, a rising ~1s "
                       f"depolarizing ramp on assembly-0's APICAL (dendritic bistability KEPT ON, self_regen={a.self_regen_read}) "
                       f"+ THETA on the CA3 basket reads the stored forward-asymmetric chain as MONOTONICALLY-ADVANCING "
                       f"theta phases in the correct forward order (phase-order-forward {mainpo}/{len(per)}; per-cycle "
                       f"forward_frac {mfwd:.3f} vs reverse {mrev:.3f} vs chance {mch:.3f}; mean phase-lock R {mR:.3f} "
                       f">> the point-neuron 0.055). SHUFFLED + REVERSE-ASYM + NO-CUE collapse the forward order; "
                       f"BASKET-OFF collapses the phase-lock (R {mRbo:.3f}); and the POINT-NEURON control (self_regen=0, "
                       f"ramp-on-soma) stays R {mRpt:.3f} ~ the point-neuron 0.055 => THE DENDRITIC COMPARTMENT IS "
                       f"LOAD-BEARING; the point-neuron-limit hypothesis is CONFIRMED. Run the full 6-seed (bar >=5/6).")
        else:
            verdict = (f"HONEST NEGATIVE {n_go}/{len(per)} -- the two-compartment APICAL substrate is present "
                       f"(apical_present {apic}/{len(per)}, basket_n>=1 on {basket_ok}/{len(per)}) and the ramp targets the "
                       f"apical, but the DENDRITIC ramp-phase-advance readout did NOT produce clean forward-ordered "
                       f"phase-locking (phase-order-forward {mainpo}/{len(per)}; per-asm active {nact}; per-cycle "
                       f"forward_frac {mfwd:.3f} vs chance {mch:.3f}; MAIN mean R {mR:.3f} vs point-neuron control {mRpt:.3f}). "
                       f"Per THE LAW this is a verdict on THIS dendritic operating point, NOT on the phase-order CAPABILITY "
                       f"-- read the per-seed apical_read(mean/max) + per-asm-active + mean_R numbers to scope the next "
                       f"move (e.g. the bistable plateau LATCHES so it can't reset each theta cycle -> a NON-latching "
                       f"sub-threshold dendritic oscillation / stronger basket->soma->apical reset / lower self_regen).")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_ramp_dendritic_phase_advance_readout",
               "mechanism": "candidate #3 Kamondi ramp single-cell pacemaker, DENDRITIC (two-compartment apical)",
               "GO": go, "n_go": n_go, "seeds": a.seeds, "sim_edit": False,
               "decoupled_cfg": {k: cfg[k] for k in sorted(cfg)},
               "ramp_theta_cfg": dict(theta_period=a.theta_period, theta_depth=a.theta_depth,
                                      basket_baseline=a.basket_baseline, sel_inhib_spare=a.sel_inhib_spare,
                                      ramp_apical_mv=a.ramp_apical_mv, ramp_soma_pa=a.ramp_soma_pa, ramp_rise=a.ramp_rise,
                                      ramp_settle=a.ramp_settle, ramp_cell_frac=a.ramp_cell_frac,
                                      self_regen_read=a.self_regen_read, no_adapt=bool(a.no_adapt),
                                      d_abs=a.d_abs, a_abs=a.a_abs, n_ca3=a.n_ca3, n_mem=a.n_mem,
                                      rest_steps=a.rest_steps, r_floor=a.r_floor, r_go_floor=a.r_go_floor,
                                      window=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac,
                                      onset_frac=a.onset_frac),
               "point_neuron_reference": {"mean_R": 0.055, "note": "seed-42 point-neuron NEGATIVE (ramp-on-soma, self_regen=0)"},
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[gap5-ramp-DENDRITIC] VERDICT: {verdict}\n[gap5-ramp-DENDRITIC] wrote {a.out}\n" + "=" * 118,
          flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
