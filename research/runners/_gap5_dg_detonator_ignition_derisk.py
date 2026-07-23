"""gap#5 READOUT branch (B) — TARGETED DG-DETONATOR ignition of a single CA3 assembly, then FORWARD transition.

2026-07-23 (scoped by research/findings/2026-07-23-gap5-decoupled-lr-encode-GO-readout-reactivation-roadblock.md,
"Next" step 2, branch B). The DECOUPLED-lr forward-asymmetric encode is a 6/6-GO WEIGHT store (within ~206
reactivation-scale + adj_fwd ~38 / adj_rev ~5, ratio ~7.65x). The open piece is the spiking READOUT: on the
weak-between decoupled store, NON-SPECIFIC background (OU/Poisson) does NOT ignite a single within~206 assembly
(ev=0, every noise level, both the bistable-latch readout and the intrinsic-fatigue de-latch readout). The roadblock
finding's diagnosis: strong between-links -> diffuse co-fire (all 3 assemblies co-igniting); weak -> no reactivation
at all. So SPONTANEOUS discrete single-assembly ignition from noise is the wall.

THE BRANCH-B MECHANISM (research gate 2026-07-23; Kandel Principles Ch 54, hippocampal circuit): biology does NOT
ignite a CA3 replay from diffuse cortical noise — it ignites it with a **DG mossy-fiber DETONATOR synapse**. A dentate
granule cell forms a few very large "detonator" boutons (thorny excrescences) onto CA3 pyramidal cells, each so strong
that a SINGLE granule-cell spike reliably discharges its CA3 targets (Henze-Wittner-Buzsaki 2002 Nat Neurosci "Single
granule cells reliably discharge targets in the hippocampal CA3"; Bischofberger; McNaughton-Morris 1987
detonator/autoassociation; Kandel 6e Ch 54). A sparse DG input thus DETONATES a small set of CA3 cells, and the CA3
recurrent auto-associator (the within-attractor we built at ~206) COMPLETES the rest of the assembly. This is a
TARGETED ignition, not diffuse noise — exactly what a weak-between store needs.

THE TEST (reuse-by-import of the DECOUPLED encode `_prepare_sequence` + the bistable rest/detect building blocks
`_hard_silence`/`_configure_ou`/`_detect_events`/`_shuffle_within_weights`/`_noise_label` + the ordered-replay
diagnostic `_detect_sequence_events`; NO `sim/` edit): build the decoupled forward-asymmetric store, FREEZE
plasticity, hard-silence (verify dendritic reset), then during REST inject a periodic TARGETED DETONATOR — drive a
SPARSE random subset (det_frac) of assembly-0's OWN cells at det_pa for det_dur steps, every det_period, with the
plateau DE-LATCHED (self_regen_read, transient so each ignition is a discrete event) and Izhikevich spike-frequency
adaptation cranked (the intrinsic-fatigue transition driver, Ecker 2022). Does the sparse detonator IGNITE the whole
assembly-0 DISCRETELY via within-attractor completion, then TRANSITION FORWARD (A->B->C) along the adj_fwd links?

GO GATE (verify, don't assert — the runner PRINTS its own verdict; the caller must read that printed VERDICT line):
  - DISCRETE ignition: detonator produces assembly-0-specific events (member_frac >= min_frac AND >> random), the net
    rests silent between detonations (duty <= 0.40) -> ev>=1, discrete.
  - FORWARD transition: the ignited assembly hands off in order (forward_frac >= 2x chance AND > reverse_frac,
    n_multi >= 2) -> "then transitions forward".
  - CLEAN vs DIFFUSE: the decoupled store's ordered forward replay beats the symmetric positive control's DIFFUSE
    co-fire (forward order, low cross-onset), distinguishing single-assembly ignition from whole-net co-ignition.
Controls (each WIRED AND INVOKED in the run path; a control written-but-never-called is the silent-failure mode):
  - NO-DETONATOR (same encoded bridge, detonator OFF) -> SILENT, ev=0  [ACID: the detonator is the ignition source;
    retires the self-sustaining artifact]
  - SHUFFLED-DETONATOR (drive a random NON-assembly set of the same size, same schedule) -> no assembly-0-specific
    ignition  [retires "any strong drive ignites the assembly"; isolates the TARGETING]
  - NO-ENCODE (fresh bridge, store skipped, same detonator) -> only the driven cells fire, no completion  [retires
    "the detonator IS the event"; the completion needs the learned store]
  - SHUFFLED-WITHIN (scramble the within-assembly recurrent weights, same detonator) -> no completion  [isolates the
    learned within-attractor]
  - SYMMETRIC-STORE positive control (freeze_between_refresh=False -> strong SYMMETRIC between-links, same detonator)
    -> the detonator DOES ignite (readout works on this substrate) but co-fires DIFFUSELY, no forward order  [proves
    the readout ignites + doubles as the direction-lesion: symmetric between-links => no ordered transition]
  - LATCH-ON diagnostic (self_regen_read = bistable, same detonator) -> the detonated assembly stays latched, not
    discrete / no transition  [the de-latch is load-bearing for discreteness + hand-off]
  - FROZEN plasticity byte-verified across every rest phase; DENDRITIC-RESET verified (no latched plateau at rest-start)

HONEST NOTE: a targeted detonator is a legitimate biological ignition (DG->CA3 mossy detonator), NOT a host shortcut —
the ignition is a synaptic current into real CA3 cells that then complete via their own recurrent weights, and every
control isolates the LEARNED store/targeting. A "reactivation" with NO detonator (self-sustaining) is the retracted
artifact; a "reactivation" of a RANDOM set (shuffled detonator) is non-specific. This is a de-risk: a partial/negative
(ignites but does not transition; or ignites diffusely) is a real, honestly-reported result that scopes the next rung.

CPU-smoke (proves it RUNS + controls live + produces a verdict; NOT a GO/negative claim):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_dg_detonator_ignition_derisk \
      --seeds 42 --n-ca3 600 --within-events 4 --within-refresh 2 --chain-fwd 4 --rest-steps 120 \
      --det-period 40 --det-settle 20 --det-pa 1500 --out research/findings/raw/gap5_r4/dg_detonator_smoke.json
Full run (GPU, the store only completes at n_ca3=2000):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_dg_detonator_ignition_derisk \
      --seeds 42 43 44 100 101 102 --n-ca3 2000 --rest-steps 1500 \
      --out research/findings/raw/gap5_r4/dg_detonator_6seed.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
# the DECOUPLED forward-asymmetric encode (6/6-GO weight store) + its config + the ordered-replay diagnostic
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    _prepare_sequence, _detect_sequence_events, _scramble_between_weights,
)
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402
# the RANK-1 bistable rest/detect building blocks (reuse-by-import; the readout that reactivated the symmetric store)
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    _hard_silence, _configure_ou, _detect_events, _shuffle_within_weights, _noise_label, GO_CFG as _SPONT_GO_CFG,
)

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "dg_detonator_ignition.json"


# ----------------------------------------------------------------------------------------------------------------------
# _rest_and_detonate: freeze plasticity + hard-silence (verify reset) + DE-LATCH plateau + crank Izhikevich adaptation,
# then run REST while injecting a PERIODIC TARGETED DETONATOR (a sparse subset of ONE assembly's cells driven strongly).
# Mirrors _gap5_intrinsic_fatigue_replay_derisk._rest_with_fatigue's readout substrate (de-latch + adaptation) but the
# ignition source is the DG-detonator, not Poisson noise. NO host per-step assembly silence / argmax-next (numpy-ref guard).
# ----------------------------------------------------------------------------------------------------------------------
def _rest_and_detonate(prep, det_spec, rest_steps, seed, self_regen_read, adapt, d_abs, a_abs,
                       det_period, det_settle, apical_gc_read=None, verbose=False):
    """det_spec: ("assembly", aidx, det_frac, det_pa, det_dur) | ("shuffled", aidx, det_frac, det_pa, det_dur) | ("none",).
    Returns dict(F, weights_frozen, apical_rest_max, apical_n_latched, n_detonations, k_det, d0, a0)."""
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    assert bridge.core_config.enable_hebbian_learning is False
    # DE-LATCH: the plateau self-regen is read live each step (bridge.py:7399); low -> each ignition is a TRANSIENT event
    # (discrete + able to hand off), high (== plateau_self_regen) -> a bistable latch that sticks ON (the LATCH-ON diag).
    bridge.core_config.coincidence_plateau_self_regen = float(self_regen_read)
    if apical_gc_read is not None:
        bridge.core_config.apical_g_couple_to_soma = float(apical_gc_read)   # read live at bridge.py:7111

    _hard_silence(bridge)
    # DENDRITIC-RESET verification (no latched plateau at rest-start over the assembly union) -- retires the _hard_silence bug
    apical_max = None; n_latched = 0
    if getattr(bridge, "cp_v_apical", None) is not None:
        va = np.asarray(to_host(bridge.cp_v_apical))[prep["ca3_arr_host"][prep["assembly_local"]]]
        apical_max = float(np.max(va)); n_latched = int((va > float(_SPONT_GO_CFG["plateau_v_hold"])).sum())

    _configure_ou(bridge, None, seed)   # NO non-specific background -> the DETONATOR is the SOLE ignition source (keeps
    #                                     the NO-DETONATOR acid a genuine silence test)

    ca3_arr_host = prep["ca3_arr_host"]
    assemblies_local = prep["assemblies_local"]
    exc_glob = ca3_arr_host[prep["ca3_exc_local"]]
    exc_dev = cp.asarray(exc_glob, dtype=cp.int64)

    # crank Izhikevich spike-frequency adaptation on the CA3-exc slice (the intrinsic-fatigue transition driver, Ecker 2022;
    # the just-fired assembly self-fatigues so the stored forward chain drives the next). adapt=False = the ExpIF control.
    d0 = a0 = None
    if getattr(bridge, "cp_izh_d_increment", None) is not None:
        d0 = float(to_host(bridge.cp_izh_d_increment[exc_dev]).mean())
        a0 = float(to_host(bridge.cp_izh_a[exc_dev]).mean())
        if adapt:
            bridge.cp_izh_d_increment[exc_dev] = cp.float32(d_abs)
            bridge.cp_izh_a[exc_dev] = cp.float32(a_abs)

    # -- DETONATOR cell set: a SPARSE random subset (det_frac) of a SINGLE assembly's OWN cells (the mossy-fiber detonator
    #    targets; a few strong synapses ignite the assembly, then CA3 recurrence completes). "shuffled" = same COUNT but
    #    random NON-assembly CA3-exc cells (destroys the targeting -> should NOT ignite the assembly). "none" = OFF. --
    det_kind = det_spec[0]
    det_dev = None; k_det = 0; det_pa = 0.0; det_dur = 0
    if det_kind in ("assembly", "shuffled"):
        aidx = int(det_spec[1]); det_frac = float(det_spec[2]); det_pa = float(det_spec[3]); det_dur = int(det_spec[4])
        a_loc = assemblies_local[aidx]
        k_det = max(1, int(round(det_frac * len(a_loc))))
        drng = np.random.default_rng(int(seed) * 77003 + 19)
        if det_kind == "assembly":
            sel_loc = np.sort(drng.choice(a_loc, min(k_det, len(a_loc)), replace=False))
        else:
            member = set()
            for a in assemblies_local:
                member.update(int(x) for x in a)
            nonmember_loc = np.asarray([int(i) for i in prep["ca3_exc_local"] if int(i) not in member], dtype=np.int64)
            sel_loc = np.sort(drng.choice(nonmember_loc, min(k_det, len(nonmember_loc)), replace=False))
        k_det = int(len(sel_loc))
        det_dev = cp.asarray(ca3_arr_host[sel_loc], dtype=cp.int64)
        if verbose:
            print(f"      [detonate={det_kind} a{det_spec[1]}] k_det={k_det} det_pa={det_pa:g} det_dur={det_dur} "
                  f"period={det_period} settle={det_settle} self_regen_read={self_regen_read} adapt={adapt}", flush=True)

    # FROZEN-PLASTICITY GUARD: capture weights, VERIFY byte-unchanged across rest (order rides the STORED frozen chain +
    # the substrate's own u-fatigue, NOT rest-phase re-encoding -- retires the Wang confound).
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    n_detonations = 0
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0
        if det_dev is not None and t >= det_settle:
            phase = (t - det_settle) % det_period
            if phase < det_dur:
                bridge.cp_external_input_current[det_dev] = det_pa       # TARGETED detonator (sparse -> the rest completes)
                if phase == 0:
                    n_detonations += 1
        bridge._run_one_simulation_step()          # NO external inhibition / argmax / per-assembly silence (numpy-ref guard)
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
    bridge.core_config.enable_ou_process = False
    w_after = np.asarray(to_host(bridge.cp_connections.data))
    weights_frozen = bool(np.array_equal(w_before, w_after))
    return dict(F=F, weights_frozen=weights_frozen, apical_rest_max=apical_max, apical_n_latched=n_latched,
                n_detonations=n_detonations, k_det=k_det, d0=d0, a0=a0)


def _score(F, assemblies_local, assembly_local_by_idx, aidx, seed, W, ev_floor, ev_k, min_frac, active_frac, onset_frac):
    """Score BOTH the single-assembly discrete-ignition (via _detect_events on assembly aidx, with the NEXT assembly as
    the cross-assembly co-fire reference) AND the forward-transition order (via _detect_sequence_events)."""
    a_test = assembly_local_by_idx[aidx]
    a_other = assembly_local_by_idx[(aidx + 1) % len(assembly_local_by_idx)] if len(assembly_local_by_idx) > 1 else None
    ev = _detect_events(F, a_test, seed, other_local=a_other, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    seq = _detect_sequence_events(F, assemblies_local, W=W, ev_floor=ev_floor, ev_k=ev_k,
                                  active_frac=active_frac, onset_frac=onset_frac)
    return ev, seq


def _weight_diag(prep):
    return dict(w_within=prep["w_within"], w_adj_fwd=prep.get("w_adj_fwd"), w_adj_rev=prep.get("w_adj_rev"),
                ratio_adj=(float(prep.get("w_adj_fwd", 0.0)) / max(abs(float(prep.get("w_adj_rev", 0.0))), 1e-6)),
                n_between_fwd=prep.get("n_between_fwd"), n_between_rev=prep.get("n_between_rev"),
                assembly_sizes=[int(len(a)) for a in prep["assemblies"]])


def one_seed(seed, cfg, a):
    t0 = time.time()
    out = {"seed": seed}
    W, ev_floor, ev_k, min_frac = a.window, a.ev_floor, a.ev_k, a.min_frac
    af, onf = a.active_frac, a.onset_frac
    aidx = int(a.assembly_idx)
    det_frac, det_dur = float(a.det_frac), int(a.det_dur)
    self_regen_read, d_abs, a_abs = float(a.self_regen_read), float(a.d_abs), float(a.a_abs)
    det_period, det_settle = int(a.det_period), int(a.det_settle)

    def _rd(prep, det_spec, verbose=False):
        return _rest_and_detonate(prep, det_spec, a.rest_steps, seed, self_regen_read, adapt=True, d_abs=d_abs,
                                  a_abs=a_abs, det_period=det_period, det_settle=det_settle,
                                  apical_gc_read=a.apical_gc_read, verbose=verbose)

    # -- BUILD the DECOUPLED forward-asymmetric store (the store under test) --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode_decoupled"] = _weight_diag(prep)
    al = prep["assemblies_local"]
    print(f"  [seed {seed}] DECOUPLED store: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
          f"adj_rev={prep['w_adj_rev']:.2f} ratio={out['encode_decoupled']['ratio_adj']:.2f}x "
          f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) ({time.time()-t0:.0f}s)", flush=True)

    # -- GO: sweep the detonator strength (det_pa) on the SAME frozen decoupled bridge (weights frozen -> reuse-safe). --
    go_runs = {}; best_pa, best, best_ev, best_seq = None, None, None, None
    for pa in a.det_pa:
        r = _rd(prep, ("assembly", aidx, det_frac, pa, det_dur), verbose=(best_pa is None))
        ev, seq = _score(r["F"], al, al, aidx, seed, W, ev_floor, ev_k, min_frac, af, onf)
        rec = dict(det_pa=pa, k_det=r["k_det"], n_detonations=r["n_detonations"],
                   n_events=ev["n_events"], n_specific=ev["n_specific"], member_frac=ev["member_frac"],
                   random_frac=ev["random_frac"], cross_frac=ev["cross_frac"], specificity=ev["specificity"],
                   duty_cycle=ev["duty_cycle"], pop_rate=ev["pop_rate"], assembly_rest_frac=ev["assembly_rest_frac"],
                   forward_frac=seq["forward_frac"], reverse_frac=seq["reverse_frac"], n_multi=seq["n_multi"],
                   chance_forward=seq["chance_forward"], per_asm_active=seq["per_asm_active"],
                   weights_frozen=r["weights_frozen"], apical_rest_max=r["apical_rest_max"],
                   apical_n_latched=r["apical_n_latched"])
        go_runs[f"det_pa={pa:g}"] = rec
        print(f"  [seed {seed}] GO det_pa={pa:>7g}: ev={ev['n_events']:>3} spec={ev['n_specific']:>3} "
              f"memb={ev['member_frac']:.3f} rand={ev['random_frac']:.3f} cross={ev['cross_frac']:.3f} "
              f"duty={ev['duty_cycle']:.3f} | FWD={seq['forward_frac']:.3f} REV={seq['reverse_frac']:.3f} "
              f"chance={seq['chance_forward']:.3f} multi={seq['n_multi']} act={seq['per_asm_active']} "
              f"frozen={r['weights_frozen']} latched={r['apical_n_latched']} ({time.time()-t0:.0f}s)", flush=True)
        score = (ev["n_specific"], seq["forward_frac"], ev["specificity"])
        if best is None or score > best:
            best, best_pa, best_ev, best_seq = score, pa, ev, seq
    out["go_runs"] = go_runs; out["best_det_pa"] = best_pa
    go_ev, go_seq = best_ev, best_seq

    # -- CONTROL 1: NO-DETONATOR (acid) -- same frozen bridge, detonator OFF -> MUST be SILENT (the detonator is the source)
    r_nd = _rd(prep, ("none",))
    nd_ev, _ = _score(r_nd["F"], al, al, aidx, seed, W, ev_floor, ev_k, min_frac, af, onf)
    out["no_detonator"] = dict(n_events=nd_ev["n_events"], n_specific=nd_ev["n_specific"],
                               assembly_rest_frac=nd_ev["assembly_rest_frac"], pop_rate=nd_ev["pop_rate"],
                               duty_cycle=nd_ev["duty_cycle"], weights_frozen=r_nd["weights_frozen"])
    print(f"  [seed {seed}] NO-DETONATOR (acid): ev={nd_ev['n_events']} spec={nd_ev['n_specific']} "
          f"assembly_rest={nd_ev['assembly_rest_frac']:.4f} pop={nd_ev['pop_rate']:.5f} "
          f"frozen={r_nd['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 2: SHUFFLED-DETONATOR -- random NON-assembly cells, same schedule -> MUST NOT ignite assembly-0 specifically
    prep_shd = _prepare_sequence(seed, cfg, do_encode=True)
    r_shd = _rest_and_detonate(prep_shd, ("shuffled", aidx, det_frac, best_pa, det_dur), a.rest_steps, seed,
                               self_regen_read, adapt=True, d_abs=d_abs, a_abs=a_abs, det_period=det_period,
                               det_settle=det_settle, apical_gc_read=a.apical_gc_read)
    shd_ev, shd_seq = _score(r_shd["F"], prep_shd["assemblies_local"], prep_shd["assemblies_local"], aidx, seed,
                             W, ev_floor, ev_k, min_frac, af, onf)
    out["shuffled_detonator"] = dict(k_det=r_shd["k_det"], n_events=shd_ev["n_events"], n_specific=shd_ev["n_specific"],
                                     member_frac=shd_ev["member_frac"], random_frac=shd_ev["random_frac"],
                                     specificity=shd_ev["specificity"], forward_frac=shd_seq["forward_frac"])
    print(f"  [seed {seed}] SHUFFLED-DETONATOR (random non-assembly, k={r_shd['k_det']}): ev={shd_ev['n_events']} "
          f"spec={shd_ev['n_specific']} memb={shd_ev['member_frac']:.3f} rand={shd_ev['random_frac']:.3f} "
          f"spec_margin={shd_ev['specificity']:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 3: NO-ENCODE -- fresh bridge, store skipped, same assembly detonator -> only driven cells fire, no completion
    prep_ne = _prepare_sequence(seed, cfg, do_encode=False)
    r_ne = _rest_and_detonate(prep_ne, ("assembly", aidx, det_frac, best_pa, det_dur), a.rest_steps, seed,
                              self_regen_read, adapt=True, d_abs=d_abs, a_abs=a_abs, det_period=det_period,
                              det_settle=det_settle, apical_gc_read=a.apical_gc_read)
    ne_ev, _ = _score(r_ne["F"], prep_ne["assemblies_local"], prep_ne["assemblies_local"], aidx, seed,
                      W, ev_floor, ev_k, min_frac, af, onf)
    out["no_encode"] = dict(w_within=prep_ne["w_within"], n_events=ne_ev["n_events"], n_specific=ne_ev["n_specific"],
                            member_frac=ne_ev["member_frac"], random_frac=ne_ev["random_frac"])
    print(f"  [seed {seed}] NO-ENCODE (w_within={prep_ne['w_within']:.2f}): ev={ne_ev['n_events']} "
          f"spec={ne_ev['n_specific']} memb={ne_ev['member_frac']:.3f} rand={ne_ev['random_frac']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- CONTROL 4: SHUFFLED-WITHIN -- fresh encoded bridge, scramble within-assembly recurrent weights, same detonator ->
    #    the within basin is destroyed -> the detonator can NOT complete -> no assembly-specific event
    prep_sw = _prepare_sequence(seed, cfg, do_encode=True)
    n_sw = _shuffle_within_weights(prep_sw, seed)
    r_sw = _rest_and_detonate(prep_sw, ("assembly", aidx, det_frac, best_pa, det_dur), a.rest_steps, seed,
                              self_regen_read, adapt=True, d_abs=d_abs, a_abs=a_abs, det_period=det_period,
                              det_settle=det_settle, apical_gc_read=a.apical_gc_read)
    sw_ev, _ = _score(r_sw["F"], prep_sw["assemblies_local"], prep_sw["assemblies_local"], aidx, seed,
                      W, ev_floor, ev_k, min_frac, af, onf)
    out["shuffled_within"] = dict(n_within_shuffled=n_sw, n_events=sw_ev["n_events"], n_specific=sw_ev["n_specific"],
                                  member_frac=sw_ev["member_frac"], random_frac=sw_ev["random_frac"])
    print(f"  [seed {seed}] SHUFFLED-WITHIN ({n_sw} edges): ev={sw_ev['n_events']} spec={sw_ev['n_specific']} "
          f"memb={sw_ev['member_frac']:.3f} rand={sw_ev['random_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # -- POSITIVE CONTROL: SYMMETRIC store (freeze_between_refresh=False -> strong SYMMETRIC between-links), same detonator
    #    -> the detonator DOES ignite (readout works on this substrate) but co-fires DIFFUSELY (high cross, no forward order).
    #    Doubles as the DIRECTION-LESION: symmetric between-weights => no ordered transition. --
    cfg_sym = {**cfg, "freeze_between_refresh": False}
    prep_sym = _prepare_sequence(seed, cfg_sym, do_encode=True)
    out["encode_symmetric"] = _weight_diag(prep_sym)
    r_sym = _rest_and_detonate(prep_sym, ("assembly", aidx, det_frac, best_pa, det_dur), a.rest_steps, seed,
                               self_regen_read, adapt=True, d_abs=d_abs, a_abs=a_abs, det_period=det_period,
                               det_settle=det_settle, apical_gc_read=a.apical_gc_read)
    sym_ev, sym_seq = _score(r_sym["F"], prep_sym["assemblies_local"], prep_sym["assemblies_local"], aidx, seed,
                             W, ev_floor, ev_k, min_frac, af, onf)
    out["symmetric_readout"] = dict(w_within=prep_sym["w_within"], w_adj_fwd=prep_sym["w_adj_fwd"],
                                    w_adj_rev=prep_sym["w_adj_rev"], n_events=sym_ev["n_events"],
                                    n_specific=sym_ev["n_specific"], member_frac=sym_ev["member_frac"],
                                    cross_frac=sym_ev["cross_frac"], duty_cycle=sym_ev["duty_cycle"],
                                    forward_frac=sym_seq["forward_frac"], reverse_frac=sym_seq["reverse_frac"],
                                    n_multi=sym_seq["n_multi"], per_asm_active=sym_seq["per_asm_active"])
    print(f"  [seed {seed}] SYM-CTRL (freeze OFF, within={prep_sym['w_within']:.1f} adj_fwd={prep_sym['w_adj_fwd']:.2f} "
          f"adj_rev={prep_sym['w_adj_rev']:.2f}): ev={sym_ev['n_events']} spec={sym_ev['n_specific']} "
          f"memb={sym_ev['member_frac']:.3f} cross={sym_ev['cross_frac']:.3f} | FWD={sym_seq['forward_frac']:.3f} "
          f"multi={sym_seq['n_multi']} act={sym_seq['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- DIAGNOSTIC: LATCH-ON (self_regen_read = bistable, same frozen decoupled bridge, same detonator) -> the detonated
    #    assembly latches ON, not discrete / no transition (the de-latch is load-bearing). Reported, not a hard GO gate. --
    r_latch = _rest_and_detonate(prep, ("assembly", aidx, det_frac, best_pa, det_dur), a.rest_steps, seed,
                                 float(cfg["plateau_self_regen"]), adapt=True, d_abs=d_abs, a_abs=a_abs,
                                 det_period=det_period, det_settle=det_settle, apical_gc_read=a.apical_gc_read)
    latch_ev, latch_seq = _score(r_latch["F"], al, al, aidx, seed, W, ev_floor, ev_k, min_frac, af, onf)
    out["latch_on_diag"] = dict(n_events=latch_ev["n_events"], duty_cycle=latch_ev["duty_cycle"],
                                forward_frac=latch_seq["forward_frac"], per_asm_active=latch_seq["per_asm_active"])
    print(f"  [seed {seed}] LATCH-ON diag (self_regen_read={cfg['plateau_self_regen']}): ev={latch_ev['n_events']} "
          f"duty={latch_ev['duty_cycle']:.3f} FWD={latch_seq['forward_frac']:.3f} act={latch_seq['per_asm_active']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- PER-SEED VERDICT --
    chance = max(go_seq["chance_forward"], 1e-6)
    discrete_ignition = (go_ev["n_specific"] >= 1 and go_ev["duty_cycle"] <= 0.40)
    assembly_specific = (go_ev["member_frac"] >= min_frac and go_ev["member_frac"] > 2.0 * (go_ev["random_frac"] + 1e-6))
    forward_transition = (go_seq["forward_frac"] >= 2.0 * chance and go_seq["forward_frac"] > go_seq["reverse_frac"]
                          and go_seq["n_multi"] >= 2)
    no_detonator_silent = (nd_ev["n_specific"] == 0 and nd_ev["assembly_rest_frac"] < 0.05)          # ACID
    shuffled_detonator_retired = (shd_ev["n_specific"] == 0
                                  or shd_ev["member_frac"] < 0.5 * max(go_ev["member_frac"], 1e-6))
    noencode_retired = (ne_ev["n_specific"] == 0 or ne_ev["member_frac"] < 0.5 * max(go_ev["member_frac"], 1e-6))
    shuffled_within_retired = (sw_ev["n_specific"] == 0 or sw_ev["member_frac"] < 0.5 * max(go_ev["member_frac"], 1e-6))
    frozen_ok = bool(go_runs[f"det_pa={best_pa:g}"]["weights_frozen"] and r_nd["weights_frozen"]
                     and r_shd["weights_frozen"] and r_ne["weights_frozen"] and r_sw["weights_frozen"]
                     and r_sym["weights_frozen"])
    dendrite_reset_ok = (go_runs[f"det_pa={best_pa:g}"]["apical_rest_max"] is None
                         or go_runs[f"det_pa={best_pa:g}"]["apical_rest_max"] <= float(cfg["plateau_v_hold"]) + 1e-3)
    readout_ignites_symmetric = (sym_ev["n_events"] >= 1)                                             # positive control fires
    # CLEAN vs DIFFUSE (diagnostic): the decoupled store's forward order beats the symmetric store's diffuse co-fire.
    cleaner_than_symmetric = bool(go_seq["forward_frac"] > sym_seq["forward_frac"]
                                  or go_ev["cross_frac"] < sym_ev["cross_frac"])

    seed_go = bool(discrete_ignition and assembly_specific and forward_transition and no_detonator_silent
                   and shuffled_detonator_retired and noencode_retired and shuffled_within_retired
                   and frozen_ok and dendrite_reset_ok and readout_ignites_symmetric)
    out["checks"] = dict(discrete_ignition=discrete_ignition, assembly_specific=assembly_specific,
                         forward_transition=forward_transition, no_detonator_silent=no_detonator_silent,
                         shuffled_detonator_retired=shuffled_detonator_retired, noencode_retired=noencode_retired,
                         shuffled_within_retired=shuffled_within_retired, frozen_ok=frozen_ok,
                         dendrite_reset_ok=dendrite_reset_ok, readout_ignites_symmetric=readout_ignites_symmetric,
                         cleaner_than_symmetric=cleaner_than_symmetric)
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']}  best_det_pa={best_pa:g} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-ca3", type=int, default=2000, help="the decoupled store only completes at 2000 (RANK-1 finding)")
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--assembly-idx", type=int, default=0, help="the within~206 attractor to detonate (0)")
    # DETONATOR (the branch-B mechanism)
    ap.add_argument("--det-frac", type=float, default=0.15, help="fraction of the assembly's cells the DG detonator drives (sparse; a few strong mossy synapses)")
    ap.add_argument("--det-pa", type=float, nargs="+", default=[1500.0, 3000.0, 6000.0], help="detonator drive strength sweep (pA)")
    ap.add_argument("--det-dur", type=int, default=15, help="detonator pulse duration (steps) per detonation")
    ap.add_argument("--det-period", type=int, default=150, help="steps between detonation onsets (long enough for an A->B->C sweep + return to silence)")
    ap.add_argument("--det-settle", type=int, default=50, help="silent settle steps before the first detonation (baseline-silence window)")
    # READOUT substrate (reused from the intrinsic-fatigue readout: de-latch + cranked adaptation)
    ap.add_argument("--self-regen-read", type=float, default=0.0, help="plateau self-regen during the READ (0 = transient de-latch -> discrete + hand-off; the load-bearing knob)")
    ap.add_argument("--d-abs", type=float, default=40.0, help="cranked Izhikevich per-spike u-kick on CA3-exc (transition driver, Ecker 2022)")
    ap.add_argument("--a-abs", type=float, default=0.008, help="cranked Izhikevich recovery rate a on CA3-exc (slower fatigue recovery)")
    ap.add_argument("--apical-gc-read", type=float, default=None, help="WEAKEN apical->soma read during rest (bridge.py:7111); None = build value (byte-identical)")
    ap.add_argument("--rest-steps", type=int, default=1500)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.5)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--min-frac", type=float, default=0.30, help="assembly-active fraction for a 'specific' ignition event")
    ap.add_argument("--active-frac", type=float, default=0.12, help="ordered-replay: per-assembly peak ACTIVE frac")
    ap.add_argument("--onset-frac", type=float, default=0.08, help="ordered-replay: per-assembly ONSET frac")
    # store knobs (default = the 6/6-GO DECOUPLED store; exposed so the JSON records exactly what was tested)
    ap.add_argument("--within-events", type=int, default=None)
    ap.add_argument("--within-refresh", type=int, default=None)
    ap.add_argument("--chain-fwd", type=int, default=None)
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

    _, backend = get_backend()
    print(f"[gap5-detonate] DG-DETONATOR ignition (branch B) on the DECOUPLED forward-asymmetric store "
          f"(within-lr {cfg['btsp_lr']} + chain-lr {cfg['chain_btsp_lr']} + freeze={cfg['freeze_between_refresh']}) | "
          f"n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} assy~{max(6, int(cfg['assembly_frac']*cfg['n_ca3']))} "
          f"det_frac={a.det_frac} det_pa={a.det_pa} det_dur={a.det_dur} period={a.det_period} settle={a.det_settle} "
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
        go = n_go >= max(1, (len(per) + 1) // 2)      # smoke gate; the FULL-RUN GO bar is >=5/6 (stated below)
        n_sym = sum(1 for p in per if p["symmetric_readout"]["n_events"] >= 1)
        mg = [p["go_runs"][f"det_pa={p['best_det_pa']:g}"] for p in per]
        mm = float(np.mean([g["member_frac"] for g in mg])); mr = float(np.mean([g["random_frac"] for g in mg]))
        mc = float(np.mean([g["cross_frac"] for g in mg])); md = float(np.mean([g["duty_cycle"] for g in mg]))
        mf = float(np.mean([g["forward_frac"] for g in mg])); mrev = float(np.mean([g["reverse_frac"] for g in mg]))
        mch = float(np.mean([g["chance_forward"] for g in mg]))
        mnd = float(np.mean([p["no_detonator"]["assembly_rest_frac"] for p in per]))
        msymf = float(np.mean([p["symmetric_readout"]["forward_frac"] for p in per]))
        if go:
            verdict = (f"DETONATOR-IGNITION GO {n_go}/{len(per)} -- a TARGETED DG-detonator (sparse subset of "
                       f"assembly-0's cells) IGNITES the whole assembly DISCRETELY via within-attractor completion "
                       f"(member_frac {mm:.3f} vs random {mr:.3f} / cross {mc:.3f}, duty {md:.3f}) then TRANSITIONS "
                       f"FORWARD (forward_frac {mf:.3f} vs reverse {mrev:.3f} vs chance {mch:.3f}); NO-DETONATOR "
                       f"SILENT (assembly rest {mnd:.4f}), shuffled-detonator + no-encode + shuffled-within collapse, "
                       f"symmetric positive control ignites (readout works, its diffuse forward {msymf:.3f}). => the "
                       f"targeted detonator surpasses the ev=0 noise-ignition roadblock; run the 6-seed GPU confirm "
                       f"(full-run bar >=5/6).")
        elif n_sym >= 1:
            verdict = (f"HONEST NEGATIVE {n_go}/{len(per)} -- the DG-detonator IGNITES on this substrate (symmetric "
                       f"positive control fires {n_sym}/{len(per)}) but the DECOUPLED store did NOT cleanly "
                       f"ignite-discretely-AND-transition-forward (member_frac {mm:.3f} duty {md:.3f} forward_frac "
                       f"{mf:.3f} vs chance {mch:.3f}). => scopes the residual: {'discrete ignition works but forward transition weak' if mm >= 0.30 else 'the sparse detonator does not complete a single within~206 assembly'}. "
                       f"Per THE LAW: RESEARCH GATE the next rung (sharper within-attractor with feedback inhibition so "
                       f"ONE assembly bursts discretely; or tune det_frac/det_pa/self_regen_read/d_abs); a partial on "
                       f"the targeted-ignition rung is a real, honestly-reported result.")
        else:
            verdict = (f"INCONCLUSIVE {n_go}/{len(per)} -- the DG-detonator did NOT ignite even the strong-between "
                       f"SYMMETRIC positive control at this scale/det_pa (sym forward {msymf:.3f}). Re-check n_ca3=2000 "
                       f"+ det_frac/det_pa/det_dur before concluding anything about the decoupled store.")
    else:
        go = False; n_go = 0; n_sym = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_dg_detonator_ignition", "branch": "B (targeted DG-detonator ignition)",
               "GO": go, "n_go": n_go, "n_symmetric_ignites": (n_sym if err is None else 0), "seeds": a.seeds,
               "decoupled_cfg": {k: cfg[k] for k in sorted(cfg)},            # every store knob recorded
               "detonator_cfg": dict(det_frac=a.det_frac, det_pa=a.det_pa, det_dur=a.det_dur, det_period=a.det_period,
                                     det_settle=a.det_settle, self_regen_read=a.self_regen_read, d_abs=a.d_abs,
                                     a_abs=a.a_abs, apical_gc_read=a.apical_gc_read, assembly_idx=a.assembly_idx,
                                     n_ca3=a.n_ca3, n_mem=a.n_mem, rest_steps=a.rest_steps, window=a.window,
                                     ev_floor=a.ev_floor, ev_k=a.ev_k, min_frac=a.min_frac, active_frac=a.active_frac,
                                     onset_frac=a.onset_frac),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[gap5-detonate] VERDICT: {verdict}\n[gap5-detonate] wrote {a.out}\n" + "=" * 118, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
