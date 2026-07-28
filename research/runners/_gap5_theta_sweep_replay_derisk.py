"""gap#5 READOUT (RANK 1) — the CUED THETA-DISINHIBITION SWEEP: read a stored forward-asymmetric CA3 chain's ORDER as a
theta-paced forward SWEEP (phase-ordered replay), on the real spiking substrate, NO `sim/` edit.

2026-07-23 (research-gate synthesis /tasks/wex1n8jx9 — "Tsodyks cued theta-disinhibition sweep", RANK 1). The ENCODE
is SOLVED: the DECOUPLED forward-asymmetric store (`DECOUPLED_CFG`, within ~206 + adj_fwd ~38 / adj_rev ~5, ratio
~7.65x) is 6/6-GO. The forward-asymmetric WEIGHTS *are* the order; there is NOTHING extra to "encode as phase". The
open piece was the READOUT: two IGNITION readouts failed — (A) spontaneous bistable ignition = the correct HONEST
NEGATIVE (recall is always CUED, never self-igniting); (B) one-shot DG-detonator ev=0 (no theta disinhibition ramp /
E%-max window). And three prior external-inhibition attempts put theta on the WRONG target (the CA3 *excitatory* cells,
`_gap5_spiking_gamma_replay_derisk._rest_with_gamma`'s `theta_ramp` -> `exc_dev_all`, which "KILLED all firing") and/or
used GAMMA as the reset carrier (McLelland-Paulsen 2009: gamma CANNOT reset). Theta is the reset; gamma only selects
within a theta cycle.

THE MECHANISM (RANK 1; four ingredients, three already built, NO `sim/` edit):
  1. THETA as a MODEST oscillatory drive onto the CA3 INHIBITORY BASKET (`ca3_pv_basket`, which the store already builds
     via ca3_fb_inhib=20) — NOT the exc cells. Disinhibits the exc chain at the theta PEAK (the asymmetric weights sweep
     forward to the next assembly) and RE-inhibits at the TROUGH (reset). Tsodyks-Skaggs-Sejnowski-McNaughton 1996 Eq.10
     (theta on the interneuron pool at modest depth). THIS IS THE SINGLE LOAD-BEARING CORRECTION vs the pinned failure.
  2. A per-theta-onset DG-DETONATOR CUE to assembly-0 (recall is always cued; Henze-Wittner-Buzsaki 2002 single granule
     cell = CA3 detonator). Aligned to theta_period. Folded from `_gap5_dg_detonator_ignition_derisk._rest_and_detonate`.
  3. INTRINSIC-FATIGUE self-avoidance on the CA3-exc slice (Izhikevich `cp_izh_d_increment`/`cp_izh_a` + the de-latch
     `coincidence_plateau_self_regen=0`) so the just-fired assembly self-fatigues -> forward-only sweep (Ecker 2022).
  4. Read order via `_detect_sequence_events` (already phase-compatible: per-assembly ONSET time within each event
     window -> forward_frac / reverse_frac / n_multi / per_asm_active).

GO GATE (verify, don't assert — the runner PRINTS its verdict; the caller reads THAT line):
  - forward_frac >= 1.5x chance AND forward_frac > reverse_frac + 0.2 AND n_multi >= 2 (discrete ordered forward bursts).
  - discreteness: per_asm_active toward [1,1,1] (NOT [3,3,3] co-ignition, NOT [1,0,0] no-handoff).
Anti-cheats (each WIRED AND INVOKED — a control written-but-never-called is the silent-failure mode):
  (1) NO-THETA (constant basket inhibition, no oscillation) -> no reset carrier -> ordered sweep collapses.  [theta reset
      is load-bearing, not a host argmax]
  (2) THETA-ON-EXC (the pinned prior failure: theta INHIBITION on the exc cells, not the basket) -> collapses.  [isolates
      the target correction]
  (3) REVERSE-ASYMMETRY-LESION (`_symmetrize_between_weights`: flatten between-edges to the mean -> adj_fwd==adj_rev) ->
      forward DIRECTION destroyed.  [the forward WEIGHT ASYMMETRY carries the order]
  (4) PERMUTED-STORE (`_scramble_between_weights`: permute the between-edge multiset) -> order collapses EVEN with the
      full theta clock.  [order is in the learned weights, not imposed by cue timing / self-avoidance]
  (5) NO-CUE (no per-theta detonator) -> no reliable sweep initiation (= method A, n_multi ~ 0).  [the cue is the seed]
  + FROZEN plasticity byte-verified across every rest phase (order rides the STORED frozen chain, not rest-phase
    re-encoding); numpy-reference guard: NO host per-step per-assembly silencing / argmax inside the loop (order emerges
    from the substrate's own theta + weights + adaptation).

HONEST NOTE: a targeted cue is a legitimate biological ignition (DG->CA3 mossy detonator); the theta/fatigue are
host-injected currents standing in for a septal pacemaker + a gamma-FS pool (the RUNG-3 endpoint = a LEARNED spiking
oscillator; additive/default-off if ever built, NOT needed to de-risk this readout). This is a de-risk: a
partial/negative (cued ignition but no forward hand-off; or co-ignition) is a real, honestly-reported result.

CPU-smoke (proves it RUNS + all controls live + produces a verdict; NOT a GO/negative claim):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_theta_sweep_replay_derisk \
      --seeds 42 --n-ca3 600 --within-events 4 --within-refresh 2 --chain-fwd 4 --rest-steps 260 \
      --theta-period 100 --det-settle 40 --out research/findings/raw/gap5_r4/theta_sweep_smoke.json
Full run (CPU, local; the store completes at n_ca3=2000):
  SIM_BACKEND=numpy nohup .venv/bin/python -m research.runners._gap5_theta_sweep_replay_derisk \
      --seeds 42 43 44 100 101 102 --n-ca3 2000 --rest-steps 1500 \
      --out research/findings/raw/gap5_r4/theta_sweep_6seed.json &
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
# the DECOUPLED forward-asymmetric encode (6/6-GO weight store) + its config + the ordered-replay diagnostic + the
# weight-lesion controls (reuse-by-import; NO `sim/` edit anywhere)
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    _prepare_sequence, _detect_sequence_events, _scramble_between_weights, _symmetrize_between_weights,
)
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402
# the RANK-1 rest building blocks (freeze/silence/OU) reused verbatim
from research.runners._gap5_spontaneous_reactivation_derisk import _hard_silence, _configure_ou  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "theta_sweep_replay.json"


# ----------------------------------------------------------------------------------------------------------------------
# _rest_theta_sweep: freeze plasticity + hard-silence (verify reset) + de-latch + crank Izhikevich adaptation, then run
# REST while (a) modulating the CA3 INHIBITORY BASKET with a theta oscillation (disinhibit exc at peak -> the asymmetric
# weights sweep forward; re-inhibit at trough -> reset), and (b) firing a per-theta-onset DG-detonator cue into
# assembly-0. theta_target in {"basket" (RANK 1), "exc" (the pinned wrong-target control), "none" (NO-THETA: constant)}.
# cue=True/False (NO-CUE control). NO host per-step per-assembly silence / argmax (numpy-reference guard).
# ----------------------------------------------------------------------------------------------------------------------
def _rest_theta_sweep(prep, rest_steps, seed, *, theta_target, cue, theta_period, theta_depth, basket_baseline,
                      theta_exc_pa, det_frac, det_pa, det_dur, det_settle, self_regen_read, d_abs, a_abs, adapt,
                      theta_region=None, cue_kind="assembly", verbose=False):
    """Returns dict(F, weights_frozen, apical_rest_max, apical_n_latched, n_cues, k_det, basket_n, theta_target)."""
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    assert bridge.core_config.enable_hebbian_learning is False
    # DE-LATCH the plateau during the READ (0 = transient -> discrete + able to hand off; the load-bearing knob).
    bridge.core_config.coincidence_plateau_self_regen = float(self_regen_read)

    _hard_silence(bridge)
    # DENDRITIC-RESET verification (no latched plateau at rest-start over the assembly union)
    apical_max = None; n_latched = 0
    if getattr(bridge, "cp_v_apical", None) is not None:
        va = np.asarray(to_host(bridge.cp_v_apical))[prep["ca3_arr_host"][prep["assembly_local"]]]
        apical_max = float(np.max(va)); n_latched = int((va > float(DECOUPLED_CFG["plateau_v_hold"])).sum())

    _configure_ou(bridge, None, seed)   # NO non-specific background -> the CUE is the SOLE ignition source (keeps the
    #                                     NO-CUE control a genuine silence test; recall is cued, not noise-driven)

    ca3_arr_host = prep["ca3_arr_host"]
    assemblies_local = prep["assemblies_local"]
    rm = bridge.region_manager
    exc_glob = ca3_arr_host[prep["ca3_exc_local"]]
    exc_dev = cp.asarray(exc_glob, dtype=cp.int64)
    # THE THETA TARGET (the correction): the CA3 INHIBITORY BASKET (mode "basket"/"none"). theta_region selects WHICH
    # basket: default ca3_pv_basket (the store's feedback basket, ca3_fb_inhib=20; its ->member synapses are sparsed to
    # sel_inhib_spare). RANK-2: pass theta_region="ca3_ff_basket" -- the E%-max FEEDFORWARD basket (built only if cfg
    # ca3_ff_inhib set), a SEPARATE region NOT subject to the ca3_pv_basket->member sparing, so theta reaches the
    # assembly WITHOUT touching sel_inhib_spare.
    _target_region = theta_region or "ca3_pv_basket"
    basket_glob = None; basket_n = 0
    try:
        _b = np.asarray(list(rm.indices(_target_region)), dtype=np.int64)
        basket_glob = cp.asarray(_b, dtype=cp.int64); basket_n = int(len(_b))
    except Exception:
        basket_glob = None

    # crank Izhikevich spike-frequency adaptation on the CA3-exc slice (the intrinsic-fatigue self-avoidance transition
    # driver, Ecker 2022; the just-fired assembly self-fatigues so the stored forward chain drives the next).
    if adapt and getattr(bridge, "cp_izh_d_increment", None) is not None:
        bridge.cp_izh_d_increment[exc_dev] = cp.float32(d_abs)
        bridge.cp_izh_a[exc_dev] = cp.float32(a_abs)

    # -- DETONATOR CUE cell set: a SPARSE random subset (det_frac) of assembly-0's OWN cells (== _rest_and_detonate). --
    det_dev = None; k_det = 0
    if cue:
        aidx = 0
        a_loc = assemblies_local[aidx]
        k_det = max(1, int(round(det_frac * len(a_loc))))
        drng = np.random.default_rng(int(seed) * 77003 + 19)
        if cue_kind == "assembly":
            sel_loc = np.sort(drng.choice(a_loc, min(k_det, len(a_loc)), replace=False))
        else:   # "shuffled": same COUNT but random NON-assembly CA3-exc cells (destroys targeting)
            member = set()
            for a in assemblies_local:
                member.update(int(x) for x in a)
            nonmember_loc = np.asarray([int(i) for i in prep["ca3_exc_local"] if int(i) not in member], dtype=np.int64)
            sel_loc = np.sort(drng.choice(nonmember_loc, min(k_det, len(nonmember_loc)), replace=False))
        k_det = int(len(sel_loc))
        det_dev = cp.asarray(ca3_arr_host[sel_loc], dtype=cp.int64)

    if verbose:
        print(f"      [sweep target={theta_target} cue={cue}({cue_kind}) k_det={k_det} theta_period={theta_period} "
              f"depth={theta_depth} basket_n={basket_n} self_regen={self_regen_read} adapt={adapt}]", flush=True)

    # FROZEN-PLASTICITY GUARD (order rides the STORED frozen chain + the substrate's own u-fatigue, NOT rest re-encoding).
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    n_cues = 0
    two_pi = 2.0 * np.pi
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = (t % theta_period) / theta_period
        theta_mod = (1.0 - np.cos(two_pi * phase)) / 2.0                 # 0 at phase 0 (cue), 1 mid-cycle (reset)
        if theta_target == "basket" and basket_glob is not None:
            # SIGNED Tsodyks disinhibition (Eq.10): drive the basket BELOW baseline at the cue onset (phase 0 ->
            # -theta_depth: the FS basket quiets -> the assembly is DISINHIBITED -> the asymmetric weights sweep forward)
            # and ABOVE baseline mid-cycle (+theta_depth: the basket fires -> the assembly is RE-inhibited = the reset).
            # This modulates BOTH below and above the basket's own ca3-feedback baseline, so with sel_inhib_spare>0 (the
            # basket->member synapses exist) the theta now genuinely REACHES + disinhibits the assembly cells.
            bridge.cp_external_input_current[basket_glob] += float(basket_baseline) + float(theta_depth) * (2.0 * theta_mod - 1.0)
        elif theta_target == "exc":
            # THE PINNED PRIOR FAILURE (control): theta as INHIBITION on the CA3 EXCITATORY cells, not the basket.
            bridge.cp_external_input_current[exc_dev] += -float(theta_exc_pa) * theta_mod
        elif theta_target == "none" and basket_glob is not None:
            # NO-THETA (control): CONSTANT basket drive (baseline, no oscillation) -> no reset carrier -> no ordered sweep.
            bridge.cp_external_input_current[basket_glob] += float(basket_baseline)
        # per-theta-onset DG-detonator CUE to assembly-0 (the seed): drive for det_dur steps starting at each onset.
        if det_dev is not None and t >= det_settle:
            phase_step = (t - det_settle) % theta_period
            if phase_step < det_dur:
                bridge.cp_external_input_current[det_dev] += float(det_pa)   # sparse -> the within-attractor completes
                if phase_step == 0:
                    n_cues += 1
        bridge._run_one_simulation_step()          # NO external per-assembly silence / argmax (numpy-reference guard)
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
    bridge.core_config.enable_ou_process = False
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    weights_frozen = bool(np.array_equal(w_before, w_after))
    return dict(F=F, weights_frozen=weights_frozen, apical_rest_max=apical_max, apical_n_latched=n_latched,
                n_cues=n_cues, k_det=k_det, basket_n=basket_n, theta_target=theta_target)


def _seq(F, assemblies_local, det):
    r = _detect_sequence_events(F, assemblies_local, **det)
    return r


def _weight_diag(prep):
    return dict(w_within=prep["w_within"], w_adj_fwd=prep.get("w_adj_fwd"), w_adj_rev=prep.get("w_adj_rev"),
                ratio_adj=(float(prep.get("w_adj_fwd", 0.0)) / max(abs(float(prep.get("w_adj_rev", 0.0))), 1e-6)),
                n_between_fwd=prep.get("n_between_fwd"), n_between_rev=prep.get("n_between_rev"),
                assembly_sizes=[int(len(a)) for a in prep["assemblies"]])


def one_seed(seed, cfg, a):
    t0 = time.time()
    out = {"seed": seed}
    det = dict(W=a.window, ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac)
    sweep_kw = dict(theta_period=a.theta_period, theta_depth=a.theta_depth, basket_baseline=a.basket_baseline,
                    theta_exc_pa=a.theta_exc_pa, det_frac=a.det_frac, det_pa=a.det_pa, det_dur=a.det_dur,
                    det_settle=a.det_settle, self_regen_read=a.self_regen_read, d_abs=a.d_abs, a_abs=a.a_abs, adapt=True,
                    theta_region=a.go_theta_region)   # None=ca3_pv_basket (RANK 1); "ca3_ff_basket"=RANK-2 E%-max basket

    # -- BUILD the DECOUPLED forward-asymmetric store (the store under test; reused frozen across the readout arms) --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode_decoupled"] = _weight_diag(prep)
    al = prep["assemblies_local"]
    print(f"  [seed {seed}] DECOUPLED store: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
          f"adj_rev={prep['w_adj_rev']:.2f} ratio={out['encode_decoupled']['ratio_adj']:.2f}x "
          f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) ({time.time()-t0:.0f}s)", flush=True)

    # -- GO ARM: theta on the BASKET + cue on -> cued theta-disinhibition sweep --
    r_go = _rest_theta_sweep(prep, a.rest_steps, seed, theta_target="basket", cue=True, verbose=True, **sweep_kw)
    s_go = _seq(r_go["F"], al, det)
    out["go"] = {k: s_go[k] for k in ("n_events", "n_multi", "n_full", "forward_frac", "reverse_frac", "mean_tau",
                                      "chance_forward", "duty_cycle", "pop_rate", "per_asm_active")}
    out["go"].update(weights_frozen=r_go["weights_frozen"], apical_rest_max=r_go["apical_rest_max"],
                     apical_n_latched=r_go["apical_n_latched"], n_cues=r_go["n_cues"], basket_n=r_go["basket_n"])
    print(f"  [seed {seed}] GO (theta-BASKET + cue): ev={s_go['n_events']} multi={s_go['n_multi']} "
          f"FWD={s_go['forward_frac']:.3f} REV={s_go['reverse_frac']:.3f} chance={s_go['chance_forward']:.3f} "
          f"tau={s_go['mean_tau']:+.3f} duty={s_go['duty_cycle']:.3f} act={s_go['per_asm_active']} "
          f"cues={r_go['n_cues']} basket_n={r_go['basket_n']} frozen={r_go['weights_frozen']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 1: NO-THETA (constant basket inhibition, no oscillation) -> no reset carrier -> order collapses --
    r_nt = _rest_theta_sweep(prep, a.rest_steps, seed, theta_target="none", cue=True, **sweep_kw)
    s_nt = _seq(r_nt["F"], al, det)
    out["no_theta"] = dict(n_multi=s_nt["n_multi"], forward_frac=s_nt["forward_frac"], reverse_frac=s_nt["reverse_frac"],
                           duty_cycle=s_nt["duty_cycle"], per_asm_active=s_nt["per_asm_active"],
                           weights_frozen=r_nt["weights_frozen"])
    print(f"  [seed {seed}] NO-THETA (constant): multi={s_nt['n_multi']} FWD={s_nt['forward_frac']:.3f} "
          f"REV={s_nt['reverse_frac']:.3f} duty={s_nt['duty_cycle']:.3f} act={s_nt['per_asm_active']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 2: THETA-ON-EXC (the pinned prior failure: theta inhibition on the exc cells, not the basket) --
    r_ex = _rest_theta_sweep(prep, a.rest_steps, seed, theta_target="exc", cue=True, **sweep_kw)
    s_ex = _seq(r_ex["F"], al, det)
    out["theta_on_exc"] = dict(n_multi=s_ex["n_multi"], forward_frac=s_ex["forward_frac"],
                               reverse_frac=s_ex["reverse_frac"], pop_rate=s_ex["pop_rate"],
                               per_asm_active=s_ex["per_asm_active"], weights_frozen=r_ex["weights_frozen"])
    print(f"  [seed {seed}] THETA-ON-EXC (wrong target): multi={s_ex['n_multi']} FWD={s_ex['forward_frac']:.3f} "
          f"pop={s_ex['pop_rate']:.4f} act={s_ex['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 3: NO-CUE (theta on the basket, NO detonator) -> no reliable sweep initiation (= method A) --
    r_nc = _rest_theta_sweep(prep, a.rest_steps, seed, theta_target="basket", cue=False, **sweep_kw)
    s_nc = _seq(r_nc["F"], al, det)
    out["no_cue"] = dict(n_multi=s_nc["n_multi"], forward_frac=s_nc["forward_frac"], pop_rate=s_nc["pop_rate"],
                         per_asm_active=s_nc["per_asm_active"], weights_frozen=r_nc["weights_frozen"])
    print(f"  [seed {seed}] NO-CUE: multi={s_nc['n_multi']} FWD={s_nc['forward_frac']:.3f} pop={s_nc['pop_rate']:.4f} "
          f"act={s_nc['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 4: PERMUTED-STORE (fresh encode + permute the between-edge multiset) -> order collapses w/ full clock --
    prep_sc = _prepare_sequence(seed, cfg, do_encode=True)
    n_sc = _scramble_between_weights(prep_sc, seed)
    r_sc = _rest_theta_sweep(prep_sc, a.rest_steps, seed, theta_target="basket", cue=True, **sweep_kw)
    s_sc = _seq(r_sc["F"], prep_sc["assemblies_local"], det)
    out["permuted_store"] = dict(n_between_shuffled=n_sc, n_multi=s_sc["n_multi"], forward_frac=s_sc["forward_frac"],
                                 reverse_frac=s_sc["reverse_frac"], per_asm_active=s_sc["per_asm_active"],
                                 weights_frozen=r_sc["weights_frozen"])
    print(f"  [seed {seed}] PERMUTED-STORE ({n_sc} edges): multi={s_sc['n_multi']} FWD={s_sc['forward_frac']:.3f} "
          f"REV={s_sc['reverse_frac']:.3f} act={s_sc['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 5: REVERSE-ASYMMETRY-LESION (fresh encode + symmetrize between-edges to the mean) -> direction gone --
    prep_sym = _prepare_sequence(seed, cfg, do_encode=True)
    n_sym = _symmetrize_between_weights(prep_sym)
    r_sym = _rest_theta_sweep(prep_sym, a.rest_steps, seed, theta_target="basket", cue=True, **sweep_kw)
    s_sym = _seq(r_sym["F"], prep_sym["assemblies_local"], det)
    out["reverse_asymmetry_lesion"] = dict(n_between_symmetrized=n_sym, n_multi=s_sym["n_multi"],
                                           forward_frac=s_sym["forward_frac"], reverse_frac=s_sym["reverse_frac"],
                                           per_asm_active=s_sym["per_asm_active"], weights_frozen=r_sym["weights_frozen"])
    print(f"  [seed {seed}] REVERSE-ASYM-LESION ({n_sym} edges symmetrized): multi={s_sym['n_multi']} "
          f"FWD={s_sym['forward_frac']:.3f} REV={s_sym['reverse_frac']:.3f} act={s_sym['per_asm_active']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- PER-SEED VERDICT (verify, don't assert) --
    chance = max(s_go["chance_forward"], 1e-6)
    fwd = s_go["forward_frac"]; rev = s_go["reverse_frac"]
    forward_ordered = (fwd >= 1.5 * chance and fwd > rev + 0.2 and s_go["n_multi"] >= 2)
    # discreteness: per_asm_active leaning to ~[1,1,1] (single-assembly bursts), not [3,3,3] co-ignition
    pa = s_go["per_asm_active"]; nev = max(s_go["n_multi"], 1)
    discrete = (s_go["duty_cycle"] <= 0.55 and all(x <= 1.6 * nev for x in pa))
    theta_lesion_load_bearing = (s_nt["forward_frac"] <= fwd - 0.15 or s_nt["n_multi"] == 0)
    wrong_target_collapses = (s_ex["forward_frac"] <= max(0.67 * fwd, 1.5 * chance) or s_ex["n_multi"] == 0)
    cue_lesion_load_bearing = (s_nc["n_multi"] == 0 or s_nc["forward_frac"] <= max(0.67 * fwd, 1.5 * chance))
    permuted_collapses = (s_sc["forward_frac"] <= max(0.67 * fwd, 1.5 * chance) or s_sc["n_multi"] == 0)
    reverse_lesion_collapses = (s_sym["forward_frac"] <= max(0.67 * fwd, 1.5 * chance) or s_sym["n_multi"] == 0)
    frozen_ok = bool(r_go["weights_frozen"] and r_nt["weights_frozen"] and r_ex["weights_frozen"]
                     and r_nc["weights_frozen"] and r_sc["weights_frozen"] and r_sym["weights_frozen"])
    dendrite_reset_ok = (r_go["apical_rest_max"] is None
                         or r_go["apical_rest_max"] <= float(DECOUPLED_CFG["plateau_v_hold"]) + 1e-3)

    seed_go = bool(forward_ordered and discrete and theta_lesion_load_bearing and wrong_target_collapses
                   and cue_lesion_load_bearing and permuted_collapses and reverse_lesion_collapses
                   and frozen_ok and dendrite_reset_ok)
    out["checks"] = dict(forward_ordered=forward_ordered, discrete=discrete,
                         theta_lesion_load_bearing=theta_lesion_load_bearing,
                         wrong_target_collapses=wrong_target_collapses, cue_lesion_load_bearing=cue_lesion_load_bearing,
                         permuted_collapses=permuted_collapses, reverse_lesion_collapses=reverse_lesion_collapses,
                         frozen_ok=frozen_ok, dendrite_reset_ok=dendrite_reset_ok)
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-ca3", type=int, default=2000, help="the decoupled store only completes at 2000 (RANK-1 finding)")
    ap.add_argument("--n-mem", type=int, default=3)
    # THETA (the correction): a modest oscillatory drive onto the CA3 INHIBITORY BASKET
    ap.add_argument("--theta-period", type=int, default=220, help="steps per theta cycle (one sequence); dt=0.5ms -> ~110ms")
    ap.add_argument("--theta-depth", type=float, default=400.0, help="basket theta modulation amplitude (pA); disinhibits exc at the cue onset, re-inhibits mid-cycle")
    ap.add_argument("--basket-baseline", type=float, default=0.0, help="tonic basket drive floor (pA)")
    ap.add_argument("--theta-exc-pa", type=float, default=800.0, help="THETA-ON-EXC control: inhibition amplitude on the exc cells (the pinned prior failure)")
    # CUE (the seed): a per-theta-onset DG-detonator into assembly-0
    ap.add_argument("--det-frac", type=float, default=0.15, help="fraction of assembly-0's cells the DG detonator cue drives (sparse)")
    ap.add_argument("--det-pa", type=float, default=3000.0, help="cue drive strength (pA)")
    ap.add_argument("--det-dur", type=int, default=12, help="cue pulse duration (steps) per theta onset")
    ap.add_argument("--det-settle", type=int, default=60, help="silent settle steps before the first cue (baseline-silence window)")
    # READOUT substrate (de-latch + cranked intrinsic-fatigue self-avoidance)
    ap.add_argument("--self-regen-read", type=float, default=0.0, help="plateau self-regen during the READ (0 = transient de-latch -> discrete + hand-off)")
    ap.add_argument("--d-abs", type=float, default=40.0, help="cranked Izhikevich per-spike u-kick on CA3-exc (fatigue self-avoidance, Ecker 2022)")
    ap.add_argument("--a-abs", type=float, default=0.008, help="cranked Izhikevich recovery rate a on CA3-exc")
    ap.add_argument("--rest-steps", type=int, default=1500)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.4)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--active-frac", type=float, default=0.12, help="ordered-replay: per-assembly peak ACTIVE frac")
    ap.add_argument("--onset-frac", type=float, default=0.08, help="ordered-replay: per-assembly ONSET frac")
    # store knobs (default = the 6/6-GO DECOUPLED store; exposed so the JSON records exactly what was tested)
    ap.add_argument("--within-events", type=int, default=None)
    ap.add_argument("--within-refresh", type=int, default=None)
    ap.add_argument("--chain-fwd", type=int, default=None)
    # RANK-2 E%-max FEEDFORWARD ca3_ff_basket (STEP 2): a SEPARATE basket NOT subject to the ca3_pv_basket->member
    # sparing, so theta reaches the assembly WITHOUT touching the GO store's sel_inhib_spare. Set --ca3-ff-inhib to build
    # it (threaded through _prepare_sequence -> _build; default None = byte-identical), and --go-theta-region ca3_ff_basket
    # to target theta onto it. de Almeida-Idiart-Lisman 2009.
    ap.add_argument("--ca3-ff-inhib", type=float, default=None, help="RANK-2: build the E-pct-max ca3_ff_basket FEEDFORWARD basket (weight); None = not built (RANK 1)")
    ap.add_argument("--ca3-ff-n", type=int, default=None, help="RANK-2: ca3_ff_basket size (default 0.25*n_ca3)")
    ap.add_argument("--go-theta-region", type=str, default=None, choices=["ca3_pv_basket", "ca3_ff_basket"],
                    help="which basket the GO/NO-THETA/NO-CUE/PERMUTED/REVERSE arms target theta onto; default (unset)=ca3_pv_basket (RANK 1), ca3_ff_basket=RANK 2")
    ap.add_argument("--sel-inhib-spare", type=float, default=None,
                    help="THE CRUX SWITCH (research synthesis RANK-2 note): the DECOUPLED store spares assembly members "
                         "from the CA3 basket (sel_inhib_spare=0.0 -> basket->member synapses zeroed), so THETA-ON-BASKET "
                         "cannot reach the assembly cells (verified: theta_depth 0->160 inert, per_asm_active=[0,0,0] at "
                         "n_ca3=2000). Set >0 (e.g. 20, matching ca3_fb_inhib) so the theta-modulated basket inhibition "
                         "actually REACHES the assembly cells. None = keep 0.0 (byte-identical to the GO store).")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cfg = dict(DECOUPLED_CFG)
    cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = int(a.n_mem)
    if a.within_events is not None:
        cfg["within_events"] = int(a.within_events)
    if a.within_refresh is not None:
        cfg["within_refresh"] = int(a.within_refresh)
    if a.chain_fwd is not None:
        cfg["chain_fwd"] = int(a.chain_fwd)
    if a.sel_inhib_spare is not None:
        cfg["sel_inhib_spare"] = float(a.sel_inhib_spare)   # crux switch: let theta-on-basket reach the assembly cells
    if a.ca3_ff_inhib is not None:
        cfg["ca3_ff_inhib"] = float(a.ca3_ff_inhib)         # RANK-2: build the E%-max ca3_ff_basket (threaded to _build)
    if a.ca3_ff_n is not None:
        cfg["ca3_ff_n"] = int(a.ca3_ff_n)

    _, backend = get_backend()
    print(f"[gap5-theta-sweep] CUED THETA-DISINHIBITION SWEEP (RANK 1) on the DECOUPLED forward-asymmetric store "
          f"(within-lr {cfg['btsp_lr']} + chain-lr {cfg['chain_btsp_lr']} + freeze={cfg['freeze_between_refresh']}) | "
          f"n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} theta_period={a.theta_period} depth={a.theta_depth} "
          f"det_frac={a.det_frac} det_pa={a.det_pa} det_dur={a.det_dur} settle={a.det_settle} "
          f"self_regen_read={a.self_regen_read} d_abs={a.d_abs} a_abs={a.a_abs} rest_steps={a.rest_steps} "
          f"seeds={a.seeds} backend={backend}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        go = n_go >= max(1, (len(per) + 1) // 2)      # smoke gate; the FULL-RUN GO bar is >=5/6 (stated in the verdict)
        mf = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mr = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance_forward"] for p in per]))
        mnt = float(np.mean([p["no_theta"]["forward_frac"] for p in per]))
        mex = float(np.mean([p["theta_on_exc"]["forward_frac"] for p in per]))
        mnc_multi = float(np.mean([p["no_cue"]["n_multi"] for p in per]))
        msc = float(np.mean([p["permuted_store"]["forward_frac"] for p in per]))
        msym = float(np.mean([p["reverse_asymmetry_lesion"]["forward_frac"] for p in per]))
        n_sym_ok = sum(1 for p in per if p["go"]["basket_n"] >= 1)
        if go:
            verdict = (f"CUED-THETA-SWEEP GO {n_go}/{len(per)} -- a per-theta-onset CUE + THETA on the CA3 INHIBITORY "
                       f"BASKET reads the stored forward-asymmetric chain as a DISCRETE forward SWEEP (forward_frac "
                       f"{mf:.3f} vs reverse {mr:.3f} vs chance {mch:.3f}); NO-THETA collapses ({mnt:.3f}), THETA-ON-EXC "
                       f"collapses ({mex:.3f}), NO-CUE gives ~0 ordered events ({mnc_multi:.1f}), PERMUTED-STORE ({msc:.3f}) "
                       f"and REVERSE-ASYM-LESION ({msym:.3f}) collapse. => the theta target-correction + per-cycle cue "
                       f"surpass the ev=0 ignition roadblock; run/confirm the full 6-seed (full-run bar >=5/6).")
        elif n_sym_ok >= 1:
            verdict = (f"HONEST NEGATIVE {n_go}/{len(per)} -- the basket theta target is live (basket_n>=1 on "
                       f"{n_sym_ok}/{len(per)}) and the cue fires, but the cued theta-disinhibition sweep did NOT cleanly "
                       f"produce discrete forward-ordered bursts (forward_frac {mf:.3f} vs chance {mch:.3f}; NO-THETA "
                       f"{mnt:.3f}). => scopes the residual: tune theta_period/theta_depth/basket_baseline, det alignment, "
                       f"self_regen_read/d_abs, or add the RANK-2 E%-max ff_basket layer so ONE assembly bursts per gamma "
                       f"slot. A partial on the theta-sweep rung is a real, honestly-reported result.")
        else:
            verdict = (f"INCONCLUSIVE {n_go}/{len(per)} -- the CA3 basket target was not found (basket_n=0); the store "
                       f"must build ca3_pv_basket (ca3_fb_inhib set). Re-check n_ca3/DECOUPLED_CFG before concluding.")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_theta_sweep_replay", "mechanism": "RANK 1 cued theta-disinhibition sweep",
               "GO": go, "n_go": n_go, "seeds": a.seeds,
               "decoupled_cfg": {k: cfg[k] for k in sorted(cfg)},          # every store knob recorded
               "theta_cfg": dict(theta_period=a.theta_period, theta_depth=a.theta_depth,
                                 basket_baseline=a.basket_baseline, theta_exc_pa=a.theta_exc_pa,
                                 det_frac=a.det_frac, det_pa=a.det_pa, det_dur=a.det_dur, det_settle=a.det_settle,
                                 self_regen_read=a.self_regen_read, d_abs=a.d_abs, a_abs=a.a_abs,
                                 n_ca3=a.n_ca3, n_mem=a.n_mem, rest_steps=a.rest_steps, window=a.window,
                                 ev_floor=a.ev_floor, ev_k=a.ev_k, active_frac=a.active_frac, onset_frac=a.onset_frac),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[gap5-theta-sweep] VERDICT: {verdict}\n[gap5-theta-sweep] wrote {a.out}\n" + "=" * 118,
          flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
