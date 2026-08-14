"""ALL SELECTED DMN BASINS IGNITE + the full closed spontaneous-thought->utterance loop.

Closes the multibasin/utterance residual "3 of 4 basins ignite, not 4/4": under non-specific noise (no prompt) the
DISJOINT CA3 wander must visit + the mouth must SPEAK ABOUT *every* stored concept. THE SURPASS = adaptation-driven
itinerancy: short-term DEPRESSION (Tsodyks-Markram) on the ca3->ca3 recurrent fatigues the current winner so the
shared feedback inhibition is released and a currently-RESTED basin wins the next noise race -> the wander samples ALL
N basins (winnerless competition; the DMN itinerates, it does not lock in; Christoff et al. 2016 dynamic framework).

CRUCIAL: STD is banked NEGATIVE for chain-ORDERING (2026-07-23-gap5-...-pivot-to-STD.md), but that failure was
depressing a LEARNED forward A->B->C chain (fwd 0.333->0.000). Here the basins are DISJOINT (max pairwise overlap 0)
with NO learned inter-basin chain -> there is NOTHING directional to destroy. STD depresses only the WITHIN-assembly
recurrence of the currently-firing basin -> exactly the fatigue we want (yield -> let a RESTED basin win). So STD is
banked-negative for chain-ORDERING and is the (untested, mechanistically favorable) FAIR-SAMPLING mechanism here.

NO sim/ edit; reuse-by-import; STP is a runner-side _build flag (enable_stp / mossy_stp_disabled, riii:186-194); the
mossy dg->ca3 detonator is carved OUT of STP (else it crushes it -> silent CA3). STP lives in cp_stp_u/cp_stp_x
(effective weight = base*u*x); the stored conn.data weights are NEVER modified -> the plasticity-byte-frozen anti-cheat
survives. FUNCTIONAL CORRELATE ONLY -- no phenomenal claim.

CPU-smoke: SIM_BACKEND=numpy python -u -m research.runners._self_initiated_all_basins_ignite_derisk --seeds 42 --n-mem 4 --rest-steps 1500 --acid-steps 500 --solo-steps 800 --stp --smoke
Full(GPU): SIM_BACKEND=cupy  python -u -m research.runners._self_initiated_all_basins_ignite_derisk --seeds 42 43 44 100 101 102 --n-mem 4 --rest-steps 8000 --acid-steps 1200 --solo-steps 3000 --stp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# reuse-by-import the VALIDATED building blocks (each 6-seed GO)
from research.runners._gap5_spontaneous_reactivation_derisk import GO_CFG, _extract_ca3ca3_vec  # noqa: E402
from research.runners._self_initiated_spontaneous_thought_derisk import (  # noqa: E402
    _scale_within_assembly, _steered_rest, _curiosity_wants, _assembly_stats,
)
# the DISJOINT balanced store + selection read-out (multibasin GO). We reuse _prepare_balanced VERBATIM (via a
# scoped _build wrapper that only ADDS the two STP kwargs) so STP-OFF is byte-identical to the multibasin GO substrate.
import research.runners._self_initiation_multibasin_derisk as _MB  # noqa: E402
from research.runners._self_initiation_multibasin_derisk import _selection, NOV_BY_NMEM  # noqa: E402
# the closed-loop MOUTH + utterance stream + scramble control (utterance GO)
from research.runners._self_initiated_utterance_derisk import (  # noqa: E402
    _lexicon, _build_mouth, _episodes, _utterance_stream, _derangement,
)
from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_self_initiated_all_basins_ignite_derisk.json"


# ----------------------------------------------------------------------------------------------------------------------
# STP threading. _prepare_balanced_itin reuses the multibasin GO substrate byte-identically when stp=False; when
# stp=True it adds ONLY enable_stp=True + mossy_stp_disabled=True to the internal _build(...) call (a scoped wrapper on
# the module-global _build, restored in finally). This keeps the STP-OFF path literally the multibasin/utterance GO
# code -> the STP-lesion control is provably the byte-clean baseline.
# ----------------------------------------------------------------------------------------------------------------------
# STP is added POST-ENCODE. MEASURED (deterministic, seed 42): building the bridge with enable_stp=True crushes the
# BTSP encode to w_within ~17 vs the GO store's ~103 -- the encode is corrupted by the build-time STP path even with the
# runtime flag OFF. STP is a FAST recall-time state, not part of the one-shot plateau-driven consolidation, so we encode
# the EXACT GO substrate (store byte-identical every condition, only ~1% GPU non-determinism) and ALLOCATE the STP fast
# state AFTER the encode, active only for the wander. The mossy carve-out (mossy_stp_disabled) is UNNEEDED here: the
# wander drives CA3-exc DIRECTLY with Poisson noise (not via the dg->ca3 detonator), so no detonator to protect.
STP_U = 0.5
STP_TAU_D = 300.0
STP_TAU_F = 50.0


def _enable_stp_posthoc(prep):
    """Allocate the Tsodyks-Markram STP fast state on an ALREADY-ENCODED GO bridge and arm it for the wander, RESTRICTED
    to the ca3->ca3 EXCITATORY recurrent (the attractor). STP lives in cp_stp_x/cp_stp_u (effective weight = base*u*x);
    the stored conn.data is untouched -> the plasticity-byte-frozen anti-cheat survives.

    CRITICAL (measured): a BLANKET STP (all synapses) depresses the ca3_pv_basket->ca3 FEEDBACK INHIBITION too -> the
    E->I->E sparsifier collapses -> the network SELF-IGNITES under NO noise (108k spikes, apical latched +43mV). The
    mechanism is STP on the ca3->ca3 recurrent ONLY, so we DISABLE STP on every OTHER synapse (inhibition, the mossy
    dg->ca3 detonator, ca3->ca1, ...) via cp_stp_disabled_mask -- those keep full base weight; only the recurrent
    attractor synapses fatigue. This is the runner-side per-pathway STP scoping the spec intends (carve out what STP
    would break)."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = prep["bridge"]
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    cfg = bridge.core_config
    cfg.enable_short_term_plasticity = True
    cfg.enable_per_type_stp = False
    cfg.stp_U = float(STP_U); cfg.stp_tau_d = float(STP_TAU_D); cfg.stp_tau_f = float(STP_TAU_F)
    bridge.cp_stp_x = cp.ones(nnz, dtype=cp.float32)
    bridge.cp_stp_u = cp.full(nnz, float(STP_U), dtype=cp.float32)
    # DISABLE STP everywhere, then RE-ENABLE only on the ca3->ca3 recurrent synapses (the coincidence-routed recurrent).
    disabled = np.ones(nnz, dtype=bool)
    flat, _pre, _post = _extract_ca3ca3_vec(bridge, prep["ca3_idx"], to_host)   # ca3->ca3 recurrent synapse indices
    disabled[np.asarray(flat, dtype=np.int64)] = False
    bridge.cp_stp_disabled_mask = cp.asarray(disabled, dtype=cp.bool_)
    bridge._cached_stp_per_type = None
    prep["stp_recurrent_syn"] = int((~disabled).sum())
    return True


def _prepare_balanced_itin(seed, cfg, do_encode=True, stp=False):
    """ALWAYS the exact multibasin/utterance GO substrate (byte-clean store); when stp, allocate the STP fast state
    AFTER the encode so the store is untouched and only the wander sees STP."""
    prep = _MB._prepare_balanced(seed, cfg, do_encode=do_encode)
    if stp:
        _enable_stp_posthoc(prep)
    return prep


def _reset_stp(prep):
    """Reset the fast STP state to fully-RESTED (x=1, u=stp_U) at the START of the wander. No-op when STP is off."""
    bridge = prep["bridge"]
    x = getattr(bridge, "cp_stp_x", None)
    u = getattr(bridge, "cp_stp_u", None)
    if x is None or u is None:
        return False
    x[:] = 1.0
    u[:] = float(bridge.core_config.stp_U)
    return True


def _activate_stp_for_wander(prep, stp):
    """Arm STP for the wander: ensure the flag is on and reset the fast state to rested. No-op when STP is off."""
    if not stp:
        return False
    prep["bridge"].core_config.enable_short_term_plasticity = True
    return _reset_stp(prep)


def _within_syn_idx(prep, i):
    """Synapse indices whose pre AND post are both in basin i (the within-assembly recurrent synapses of basin i).
    Same selection _scale_within_assembly uses, but returns the index array (for reading cp_stp_x)."""
    from sim.backend import to_host
    bridge = prep["bridge"]
    conn = bridge.cp_connections
    n_all = int(bridge.core_config.num_neurons)
    nnz = int(conn.nnz)
    memb = np.zeros(n_all, dtype=bool)
    memb[np.asarray(prep["assemblies"][i], dtype=np.int64)] = True
    indptr = np.asarray(to_host(conn.indptr))
    indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    within = memb[pre_of] & memb[indices[:nnz]]
    return np.nonzero(within)[0].astype(np.int64)


def _run_condition_itin(seed, cfg, rest_steps, noise_on, *, gains=None, do_encode=True, stp=False, stp_comp=1.0):
    """A condition = a FRESH deterministic bridge (same seed -> byte-identical substrate + DISJOINT partition + encode +
    Poisson stream). The cross-condition differences are `gains`, `stp`, and (when stp) the UNIFORM tonic-utilization
    compensation `stp_comp` (all basins equally). With STP on, the fast state is reset to rested (x=1) before the
    wander. NOTE: stp_comp compensates STP's rested u*x factor so the wander's rested operating point matches the GO
    calibration; it is uniform across basins (not a per-basin thumb) and rides the same class of runtime recurrent-gain
    scaling the curiosity steering already uses (declared host boundary)."""
    n_mem = int(cfg["n_mem"])
    prep = _prepare_balanced_itin(seed, cfg, do_encode=do_encode, stp=stp)
    if gains is not None:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(gains[i]))
    if stp and abs(stp_comp - 1.0) > 1e-9:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(stp_comp))   # uniform: restore the rested operating point under STP
    _activate_stp_for_wander(prep, stp)
    F, diag = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=noise_on)
    return F, prep, diag


def _ignited(sel, n_mem):
    return int(sel["n_visited_coherent"]), (int(sel["n_visited_coherent"]) == n_mem)


def _visit_order(F, assemblies_local, min_frac, cap=48):
    """The ORDER the wander visits basins (concept index per surfacing episode). Reported so a reader sees the sequence
    VARIES across seeds (rides noise + fatigue) -- it is NOT a hard-coded cycle (the itinerancy anti-cheat)."""
    eps = _episodes(F, assemblies_local, min_frac)
    order = [int(c) for (c, s, e) in eps]
    return order[:cap]


def _stp_fatigue_probe(seed, cfg, drive_steps=60, recover_steps=400):
    """REAL-FATIGUE anti-cheat: build the STP-on store, reset STP to rested, DRIVE basin 0's assembly hard so it
    sustains, and read cp_stp_x over basin-0's within-assembly synapses -> it must DIP < 1 (genuine depletion); then
    SILENCE and read it recover toward 1 (tau_d=300ms). Proves the itinerancy is real Tsodyks-Markram depression, not a
    static reweight. Cheap (~drive+recover steps, one basin)."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    prep = _prepare_balanced_itin(seed, cfg, do_encode=True, stp=True)
    if not _activate_stp_for_wander(prep, True):
        return {"available": False}
    bridge = prep["bridge"]
    idx = _within_syn_idx(prep, 0)
    if idx.size == 0:
        return {"available": False}
    idx_dev = cp.asarray(idx, dtype=cp.int64)
    assy = np.asarray(prep["assemblies"][0], dtype=np.int64)
    assy_dev = cp.asarray(assy, dtype=cp.int64)
    x0 = float(np.asarray(to_host(bridge.cp_stp_x))[idx].mean())
    min_x = x0
    bridge.core_config.enable_hebbian_learning = False
    for _ in range(int(drive_steps)):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[assy_dev] = float(cfg["encode_drive"])  # drive basin 0 to sustain/fire
        bridge._run_one_simulation_step()
        cur = float(np.asarray(to_host(bridge.cp_stp_x))[idx].mean())
        min_x = min(min_x, cur)
    x_driven = float(np.asarray(to_host(bridge.cp_stp_x))[idx].mean())
    for _ in range(int(recover_steps)):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
    x_recovered = float(np.asarray(to_host(bridge.cp_stp_x))[idx].mean())
    return {"available": True, "x_rested": x0, "x_min_during_drive": min_x, "x_end_drive": x_driven,
            "x_after_recovery": x_recovered,
            "depletes": bool(min_x < 0.95 * x0), "recovers": bool(x_recovered > x_driven + 0.02)}


def _solo_ignition(seed, cfg, rest_steps, min_frac, stp=False):
    """SOLO diagnosis: does EACH basin ignite when the OTHERS cannot compete? For basin m, build the store (STP as
    given) then LESION every OTHER basin's within-assembly recurrence to 0 (they can no longer complete or draw the
    shared inhibition) -> basin m runs UNCONTESTED on its OWN encoded weights. If basin m still fails to ignite ->
    ABSOLUTE-threshold (its recurrence is genuinely too weak; the §4 homeostatic lever is the residual). If it ignites
    solo but not under full competition -> COMPETITION lock-in (STP is the fix). Reported, not gated (anti-cheat #10).
    Uses the SAME cell subset for basin m as the full store (same seed/partition)."""
    n_mem = int(cfg["n_mem"])
    per_basin = []
    for m in range(n_mem):
        prep = _prepare_balanced_itin(seed, cfg, do_encode=True, stp=stp)
        for j in range(n_mem):
            if j != m:
                _scale_within_assembly(prep, j, 0.0)   # zero the competitors' recurrence -> only basin m can complete
        _activate_stp_for_wander(prep, stp)
        F, _diag = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=True)
        st = _assembly_stats(F, prep["assemblies_local"], m, seed, min_frac)
        ignites = bool(st["dwell"] > 0 and st["member"] >= min_frac and st["member"] > 2.0 * (st["random"] + 1e-6))
        per_basin.append({"basin": m, "solo_ignites": ignites, "dwell": float(st["dwell"]),
                          "member": float(st["member"]), "random": float(st["random"]),
                          "n_events": int(st["n_events"])})
        print(f"    [seed {seed}] SOLO basin {m}: ignites={ignites} dwell={st['dwell']:.0f} "
              f"member {st['member']:.2f} vs rand {st['random']:.2f}", flush=True)
    n_solo = int(sum(1 for b in per_basin if b["solo_ignites"]))
    return {"per_basin": per_basin, "n_solo_ignite": n_solo, "all_solo_ignite": bool(n_solo == n_mem)}


def _diag_seed(seed, n_mem, rest_steps, solo_steps, acid_steps, min_frac):
    """CLEAN diagnostic (the decisive, STP-independent result). (1) GO-substrate ignition baseline (STP off): how many
    of n_mem disjoint basins ignite (the multibasin residual). (2) SOLO isolation diagnosis: does EACH basin ignite
    with the competitors' recurrence zeroed? A basin that fails SOLO is ABSOLUTE-THRESHOLD -> competition-reshaping
    (STP) cannot rescue it (nothing to out-compete). (3) STP NO-NOISE instability probe: with STP on the recurrent and
    NO noise the bistable down-state must stay silent; here it SELF-IGNITES (documented substrate-STP incompatibility).
    No mouth, no unstable STP-on closed loop -- this is the clean cross-seed backbone of the finding."""
    t0 = time.time()
    out = {"seed": seed, "n_mem": n_mem}
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(n_mem)
    g1 = [1.0] * n_mem
    F, prep, diag = _run_condition_itin(seed, cfg, rest_steps, True, gains=g1, stp=False)
    sel = _selection(F, prep["assemblies_local"], seed, min_frac)
    out["baseline"] = {"n_visited_coherent": sel["n_visited_coherent"],
                       "all_ignite": bool(sel["n_visited_coherent"] == n_mem), "dwell": sel["dwell"],
                       "coherent": sel["coherent"], "per_member": sel["per_member"],
                       "max_pair_overlap": int(prep["max_pair_overlap"]), "w_within": float(prep["w_within"]),
                       "weights_frozen": bool(diag["weights_frozen"])}
    print(f"  [seed {seed}] BASELINE(GO,STP-off): n_ig {sel['n_visited_coherent']}/{n_mem} "
          f"dwell {[int(x) for x in sel['dwell']]} coherent {sel['coherent']} overlap {prep['max_pair_overlap']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    out["solo"] = _solo_ignition(seed, cfg, solo_steps, min_frac, stp=False)
    Fn, pn, dn = _run_condition_itin(seed, cfg, acid_steps, False, gains=g1, stp=True)
    nn_spk = int(Fn.sum())
    out["stp_nonoise"] = {"total_ca3_spikes": nn_spk, "stable": bool(nn_spk < 50),
                          "apical_rest_max": dn["apical_rest_max"]}
    print(f"  [seed {seed}] STP NO-NOISE spikes={nn_spk} stable={nn_spk < 50} (down-state must stay silent) "
          f"| SOLO {out['solo']['n_solo_ignite']}/{n_mem} ignite ({time.time()-t0:.0f}s)", flush=True)
    return out


def one_seed(seed, n_mem, rest_steps, acid_steps, solo_steps, gain_scale, min_frac, D, stp, do_solo):
    t0 = time.time()
    out = {"seed": seed, "n_mem": n_mem, "stp": bool(stp)}
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(n_mem)
    agents, verbs, patients, vocab = _lexicon(n_mem)
    out["facts"] = [f"{agents[i]} {verbs[i]} {patients[i]}" for i in range(n_mem)]

    # -- curiosity gains (identical construction to multibasin/utterance: novelty on a RANDOM concept perm) --
    nov_rng = np.random.default_rng(seed * 7919 + 1)
    novelties = [float(v) for v in nov_rng.permutation(np.asarray(NOV_BY_NMEM[n_mem], dtype=float))]
    wants, _ = _curiosity_wants(seed, novelties)
    wmax = max(wants) if wants else 1.0
    gains_on = [1.0 + gain_scale * (w / wmax if wmax > 1e-9 else 0.0) for w in wants]
    order = [int(i) for i in np.argsort(-np.asarray(novelties))]
    gvals = sorted(gains_on, reverse=True)
    gains_reversed = [0.0] * n_mem
    for k, ci in enumerate(order):
        gains_reversed[ci] = gvals[n_mem - 1 - k]
    novel_set = np.asarray(order[:max(1, n_mem // 2)], dtype=int)
    out["novelties"] = novelties; out["gains_on"] = gains_on; out["gains_reversed"] = gains_reversed
    out["novel_order"] = order; out["novel_set"] = novel_set.tolist()

    # -- the MOUTH: build once, store the facts, decode each on-bridge (mouth fidelity + no-confab moat) --
    comp, utt_by_agent, decode_ok, moat = _build_mouth(seed, agents, verbs, patients, vocab, D)
    ident = list(range(n_mem))
    out["decode_ok"] = decode_ok; out["mouth_fidelity"] = bool(all(decode_ok)); out["moat_abstains"] = bool(moat)
    print(f"  [seed {seed}] mouth fidelity={all(decode_ok)} moat={moat} | novelty {[round(v,2) for v in novelties]} "
          f"-> gains_on {[round(g,2) for g in gains_on]} ({time.time()-t0:.0f}s)", flush=True)

    # (1) BALANCED, STP-ON, UNIFORM gain -> IGNITION COMPLETENESS (the headline: all N ignite)
    F_bal, prep_b, d_bal = _run_condition_itin(seed, cfg, rest_steps, noise_on=True, gains=[1.0] * n_mem, stp=stp)
    sel_bal = _selection(F_bal, prep_b["assemblies_local"], seed, min_frac)
    n_ig_bal, all_ig_bal = _ignited(sel_bal, n_mem)
    order_bal = _visit_order(F_bal, prep_b["assemblies_local"], min_frac)
    out["balanced"] = {"n_visited_coherent": n_ig_bal, "all_ignite": all_ig_bal, "n_visited": sel_bal["n_visited"],
                       "dwell": sel_bal["dwell"], "coherent": sel_bal["coherent"], "visited": sel_bal["visited"],
                       "top1_share": sel_bal["top1_share"], "entropy": sel_bal["entropy"],
                       "member": sel_bal["pooled_member"], "random": sel_bal["pooled_random"],
                       "max_pair_overlap": int(prep_b["max_pair_overlap"]), "w_within": float(prep_b["w_within"]),
                       "weights_frozen": bool(d_bal["weights_frozen"]), "visit_order": order_bal}
    print(f"  [seed {seed}] BALANCED(STP={stp}): n_ignite {n_ig_bal}/{n_mem} all={all_ig_bal} "
          f"dwell {[int(x) for x in sel_bal['dwell']]} coherent {sel_bal['coherent']} overlap {prep_b['max_pair_overlap']} "
          f"frozen {d_bal['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    # (2) STP-LESION control (STP OFF, else byte-identical) -> completeness must REGRESS (the fix is load-bearing)
    F_no, prep_no, d_no = _run_condition_itin(seed, cfg, rest_steps, noise_on=True, gains=[1.0] * n_mem, stp=False)
    sel_no = _selection(F_no, prep_no["assemblies_local"], seed, min_frac)
    n_ig_no, all_ig_no = _ignited(sel_no, n_mem)
    out["stp_lesion"] = {"n_visited_coherent": n_ig_no, "all_ignite": all_ig_no, "dwell": sel_no["dwell"],
                         "coherent": sel_no["coherent"], "w_within": float(prep_no["w_within"])}
    # store-integrity: STP-on encode must match the STP-off store (STP is a fast state, must not perturb the write)
    w_on = float(prep_b["w_within"]); w_off = float(prep_no["w_within"])
    store_integrity = bool(abs(w_on - w_off) <= 0.02 * max(abs(w_off), 1e-9))
    out["store_integrity"] = {"w_within_stp_on": w_on, "w_within_stp_off": w_off, "ok": store_integrity}
    print(f"  [seed {seed}] STP-LESION(off): n_ignite {n_ig_no}/{n_mem} dwell {[int(x) for x in sel_no['dwell']]} | "
          f"store-integrity w_on {w_on:.3f} vs w_off {w_off:.3f} ok={store_integrity}", flush=True)

    # (3) CURIOSITY-ON production wander + CLOSED LOOP -> speak about EVERY concept
    F_on, prep_on, d_on = _run_condition_itin(seed, cfg, rest_steps, noise_on=True, gains=gains_on, stp=stp)
    sel_on = _selection(F_on, prep_on["assemblies_local"], seed, min_frac)
    st_on = _utterance_stream(F_on, prep_on["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    order_on = _visit_order(F_on, prep_on["assemblies_local"], min_frac)
    out["on"] = {"n_utt": st_on["n_utt"], "about_rate": st_on["about_rate"],
                 "n_concepts_spoken": st_on["n_concepts_spoken"], "share": st_on["share"],
                 "counts": st_on["counts"], "examples": st_on["examples"], "visit_order": order_on,
                 "member": sel_on["pooled_member"], "random": sel_on["pooled_random"],
                 "n_visited_coherent": sel_on["n_visited_coherent"],
                 "apical_rest_max": d_on["apical_rest_max"], "weights_frozen": bool(d_on["weights_frozen"])}
    print(f"  [seed {seed}] ON: utt {st_on['n_utt']} about {st_on['about_rate']:.2f} concepts_spoken "
          f"{st_on['n_concepts_spoken']}/{n_mem} member {sel_on['pooled_member']:.2f} vs rand {sel_on['pooled_random']:.2f} "
          f"share {[round(x,2) for x in st_on['share']]} ({time.time()-t0:.0f}s)", flush=True)

    # (4) controls reused from the utterance runner
    scr = _derangement(n_mem, seed)
    st_scr = _utterance_stream(F_on, prep_on["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, scr)
    F_rv, prep_rv, _ = _run_condition_itin(seed, cfg, rest_steps, noise_on=True, gains=gains_reversed, stp=stp)
    st_rv = _utterance_stream(F_rv, prep_rv["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    F_nn, prep_nn, d_nn = _run_condition_itin(seed, cfg, acid_steps, noise_on=False, gains=gains_on, stp=stp)
    st_nn = _utterance_stream(F_nn, prep_nn["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    F_sl, prep_sl, _ = _run_condition_itin(seed, cfg, rest_steps, noise_on=True, gains=gains_on, do_encode=False, stp=stp)
    sel_sl = _selection(F_sl, prep_sl["assemblies_local"], seed, min_frac)
    st_sl = _utterance_stream(F_sl, prep_sl["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    share_on = np.asarray(st_on["share"]); share_rv = np.asarray(st_rv["share"])
    novel_on = float(share_on[novel_set].sum()); novel_rv = float(share_rv[novel_set].sum())
    out["scramble_about"] = st_scr["about_rate"]
    out["reversed"] = {"novel_share": novel_rv, "n_utt": st_rv["n_utt"]}
    out["novel_share_on"] = novel_on
    out["no_noise"] = {"n_utt": st_nn["n_utt"], "apical_rest_max": d_nn["apical_rest_max"]}
    out["store_lesion"] = {"n_utt": st_sl["n_utt"], "about_n": st_sl["n_about"],
                           "member": sel_sl["pooled_member"], "random": sel_sl["pooled_random"]}
    out["bias"] = {"novel_share_on": novel_on, "novel_share_reversed": novel_rv,
                   "attributable": attributable_to("curiosity-gain @ novel-concept utterance share (on vs reversed)",
                                                    novel_on, novel_rv)}
    print(f"  [seed {seed}] SCRAMBLE about {st_scr['about_rate']:.2f} | NO-NOISE utt {st_nn['n_utt']} | "
          f"STORE-LESION utt {st_sl['n_utt']} member {sel_sl['pooled_member']:.2f} | novel-share on {novel_on:.2f} "
          f"rev {novel_rv:.2f}", flush=True)

    # (5) REAL-FATIGUE probe (cp_stp_x dips then recovers) -- optional corroborator of anti-cheat #4
    if stp:
        out["stp_fatigue"] = _stp_fatigue_probe(seed, cfg)
        f = out["stp_fatigue"]
        if f.get("available"):
            print(f"  [seed {seed}] STP-FATIGUE x_rested {f['x_rested']:.2f} -> min {f['x_min_during_drive']:.2f} "
                  f"-> recovered {f['x_after_recovery']:.2f} depletes={f['depletes']} recovers={f['recovers']}",
                  flush=True)
    else:
        out["stp_fatigue"] = {"available": False}

    # (6) SOLO-ignition diagnosis (competition vs absolute-threshold). Run when asked (heavy: n_mem preps).
    if do_solo:
        out["solo"] = _solo_ignition(seed, cfg, solo_steps, min_frac, stp=False)
    else:
        out["solo"] = None

    void_if(st_on["n_utt"] == 0, f"seed {seed}: ON wander produced 0 utterances (nothing to interpret)")

    # ---- per-seed GO gate ----
    f = out["stp_fatigue"]
    checks = dict(
        disjoint_ok=(out["balanced"]["max_pair_overlap"] == 0),
        mouth_fidelity=(out["mouth_fidelity"] and out["moat_abstains"]),
        store_integrity=(store_integrity if stp else True),
        ALL_BASINS_IGNITE=bool(all_ig_bal),                                       # THE HEADLINE (balanced, uniform gain)
        loop_speaks_every_concept=(st_on["n_concepts_spoken"] == n_mem),          # closed loop covers the WHOLE store
        stp_load_bearing=((n_ig_no < n_mem) if stp else True),                    # STP-lesion regresses (fix load-bearing)
        about_selected=bool(st_on["about_rate"] >= 0.90 and sel_on["pooled_member"] >= min_frac
                            and sel_on["pooled_member"] > 2.0 * (sel_on["pooled_random"] + 1e-6)),
        scramble_collapses=bool(st_scr["about_rate"] <= 0.15),
        curiosity_steered=bool(novel_on >= novel_rv + 0.10),
        internally_triggered=bool(st_nn["n_utt"] == 0 and out["on"]["weights_frozen"]
                                  and (out["on"]["apical_rest_max"] is None
                                       or out["on"]["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3)),
        store_lesion_load_bearing=bool(st_sl["n_utt"] <= max(1, int(0.25 * st_on["n_utt"]))
                                       or st_sl["about_n"] == 0
                                       or sel_sl["pooled_member"] < 0.5 * sel_on["pooled_member"]),
        weights_byte_frozen=bool(out["balanced"]["weights_frozen"] and out["on"]["weights_frozen"]),
        real_fatigue=(bool(f.get("available") and f.get("depletes") and f.get("recovers")) if stp else True),
    )
    out["checks"] = checks
    out["seed_go"] = bool(all(checks.values()))
    print(f"  [seed {seed}] => {'GO' if out['seed_go'] else 'no'}  {checks}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def _main_diag(a):
    """Clean cross-seed diagnostic run (the decisive result of this rung). Writes a NEGATIVE/PARTIAL artifact: the GO
    substrate ignites only n<n_mem basins (the residual), the tail basin fails even in ISOLATION (absolute-threshold ->
    NOT competition, so STP's premise does not hold), and STP self-ignites the down-state without noise (substrate-STP
    incompatibility)."""
    print(f"[all-basins DIAG] n_mem={a.n_mem} rest={a.rest_steps} solo={a.solo_steps} acid={a.acid_steps} "
          f"seeds={a.seeds} backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    partial_path = Path(a.out).with_suffix(".partial.json")
    try:
        for s in a.seeds:
            per.append(_diag_seed(s, a.n_mem, a.rest_steps, a.solo_steps, a.acid_steps, a.min_frac))
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps({"partial": True, "seeds_done": [p["seed"] for p in per],
                                                "per_seed": per}, indent=2, default=str))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    preconditions = []
    if err is None and per:
        n = len(per)
        base_reliab = float(np.mean([1.0 if p["baseline"]["all_ignite"] else 0.0 for p in per]))
        mean_base_ig = float(np.mean([p["baseline"]["n_visited_coherent"] for p in per]))
        n_tail_fail_solo = int(sum(1 for p in per if not p["solo"]["all_solo_ignite"]))
        mean_solo = float(np.mean([p["solo"]["n_solo_ignite"] for p in per]))
        n_stp_unstable = int(sum(1 for p in per if not p["stp_nonoise"]["stable"]))
        disjoint = all(p["baseline"]["max_pair_overlap"] == 0 for p in per)
        vd = Verdict("all-basins-ignite: STP itinerancy diagnostic (6-seed)")
        vd.require("basins DISJOINT (overlap 0) every seed", disjoint, expect=True)
        vd.require("GO substrate ignites FEWER than n_mem basins (the residual) -- mean", mean_base_ig,
                   expect=lambda x, nm=a.n_mem: x < nm)
        vd.require("a basin fails even in ISOLATION (absolute-threshold, NOT competition) on >=5/6 seeds",
                   n_tail_fail_solo, expect=lambda x, t=max(1, (5 * len(per) + 5) // 6): x >= t)
        vd.require("STP self-ignites the bistable down-state WITHOUT noise (substrate-STP incompatibility) every seed",
                   n_stp_unstable, expect=lambda x, n=n: x == n)
        decided = vd.decide(False)
        preconditions = decided["preconditions"]
        verdict = (f"NEGATIVE/PARTIAL (diagnostic) -- adaptation-driven itinerancy (ca3->ca3 STP) does NOT close the "
                   f"all-{a.n_mem}-basins-ignite residual, and the reason CORRECTS the diagnosis: the GO substrate "
                   f"ignites {mean_base_ig:.1f}/{a.n_mem} basins (reliability {base_reliab:.2f}); the tail basin fails "
                   f"even in ISOLATION (solo mean {mean_solo:.1f}/{a.n_mem} ignite; a basin fails solo on "
                   f"{n_tail_fail_solo}/{n} seeds) -> ABSOLUTE-THRESHOLD, NOT competition lock-in, so STP (a "
                   f"competition-reshaper) cannot rescue it; and STP on the recurrent SELF-IGNITES the bistable "
                   f"down-state without noise on {n_stp_unstable}/{n} seeds (substrate-STP incompatibility). STP is "
                   f"BANKED (the fair-sampling fix for competition lock-in, which is NOT the failure mode here); the "
                   f"named next rung is a CONNECTIVITY-aware pattern separation / scale lever (larger n_ca3, guaranteed "
                   f"within-assembly recurrent density) so every disjoint basin can complete. Per THE LAW: method "
                   f"banked, next named; not a stop.")
    else:
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "self_initiated_all_basins_ignite_DIAG", "GO": False, "mode": "diag_only", "seeds": a.seeds,
               "n_mem": a.n_mem, "rest_steps": a.rest_steps, "solo_steps": a.solo_steps, "acid_steps": a.acid_steps,
               "min_frac": a.min_frac, "preconditions": preconditions,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[all-basins DIAG] VERDICT: {verdict}\n[all-basins DIAG] wrote {a.out}\n" + "=" * 110,
          flush=True)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=4, choices=[4, 5, 6, 8])
    ap.add_argument("--rest-steps", type=int, default=8000)
    ap.add_argument("--acid-steps", type=int, default=1200)
    ap.add_argument("--solo-steps", type=int, default=3000)
    ap.add_argument("--gain-scale", type=float, default=1.0)
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--stp", action="store_true", help="enable ca3->ca3 short-term depression (the itinerancy fix)")
    ap.add_argument("--solo", action="store_true", help="run the (heavy) solo-ignition diagnosis")
    ap.add_argument("--diag-only", action="store_true",
                    help="clean cross-seed diagnostic: GO baseline + SOLO isolation + STP NO-NOISE instability probe")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.diag_only:
        return _main_diag(a)

    print(f"[all-basins] n_mem={a.n_mem} rest={a.rest_steps} acid={a.acid_steps} solo={a.solo_steps} "
          f"gain_scale={a.gain_scale} min_frac={a.min_frac} stp={a.stp} do_solo={a.solo} seeds={a.seeds} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    partial_path = Path(a.out).with_suffix(".partial.json")
    try:
        for s in a.seeds:
            per.append(one_seed(s, a.n_mem, a.rest_steps, a.acid_steps, a.solo_steps, a.gain_scale,
                                a.min_frac, a.D, a.stp, a.solo))
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps({"partial": True, "seeds_done": [p["seed"] for p in per],
                                                "per_seed": per}, indent=2, default=str))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    preconditions = []; attribution = None
    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        thresh = max(1, (len(per) + 1) // 2) if a.smoke else max(1, (5 * len(per) + 5) // 6)
        go = n_go >= thresh
        # PRIMARY metric: ignition reliability = fraction of seeds with all n_mem basins igniting (balanced, uniform gain)
        ignition_reliability = float(np.mean([1.0 if p["balanced"]["all_ignite"] else 0.0 for p in per]))
        m_ig_bal = float(np.mean([p["balanced"]["n_visited_coherent"] for p in per]))
        m_ig_no = float(np.mean([p["stp_lesion"]["n_visited_coherent"] for p in per]))
        m_concepts = float(np.mean([p["on"]["n_concepts_spoken"] for p in per]))
        m_about = float(np.mean([p["on"]["about_rate"] for p in per]))
        m_scr = float(np.mean([p["scramble_about"] for p in per]))
        m_member = float(np.mean([p["on"]["member"] for p in per]))
        m_random = float(np.mean([p["on"]["random"] for p in per]))
        m_novel_on = float(np.mean([p["novel_share_on"] for p in per]))
        m_novel_rv = float(np.mean([p["reversed"]["novel_share"] for p in per]))
        attribution = attributable_to("curiosity-gain @ novel-concept utterance share (6-seed, on vs reversed)",
                                       m_novel_on, m_novel_rv)

        vd = Verdict("self-initiated ALL-basins ignite + closed loop (6-seed)", chance=m_random)
        vd.require("seeds passing all anti-cheats >= threshold", n_go, expect=lambda x, t=thresh: x >= t)
        vd.require("ALL n_mem basins ignite (balanced uniform gain) on >= threshold seeds -- the headline",
                   sum(1 for p in per if p["balanced"]["all_ignite"]), expect=lambda x, t=thresh: x >= t)
        vd.require("ignition reliability (mean fraction of seeds all-ignite) == 1.0 target, >= 5/6 gate",
                   ignition_reliability, expect=lambda x, t=thresh, n=len(per): x >= t / n)
        vd.require("closed loop speaks about EVERY concept (mean n_concepts_spoken == n_mem)",
                   m_concepts, expect=lambda x, nm=a.n_mem: x >= nm - 1e-9)
        vd.control("STP load-bearing: all-ignite (STP on) vs STP-lesion n_visited_coherent",
                   m_ig_bal, m_ig_no, min_separation=0.5)
        vd.require("STP-lesion REGRESSES (< n_mem) on >= threshold seeds (fix is load-bearing)",
                   sum(1 for p in per if p["stp_lesion"]["n_visited_coherent"] < a.n_mem),
                   expect=lambda x, t=thresh: x >= t)
        vd.require("basins DISJOINT (max pairwise overlap == 0) every seed",
                   all(p["balanced"]["max_pair_overlap"] == 0 for p in per), expect=True)
        vd.require("store-integrity: STP-on encode == STP-off store every seed",
                   all(p["store_integrity"]["ok"] for p in per), expect=True)
        vd.require("ABOUT-THE-SELECTED-CONCEPT rate (mean) >= 0.9", m_about, expect=lambda x: x >= 0.9)
        vd.control("about-selected: production vs SCRAMBLE-routing", m_about, m_scr, min_separation=0.5)
        vd.control("curiosity-steered: novel utterance share on vs reversed", m_novel_on, m_novel_rv,
                   min_separation=0.05)
        vd.control("coherent: surfaced member vs random floor", m_member, m_random, min_separation=0.15)
        vd.floor("coherence member above random", m_member, floor=m_random)
        vd.require("internally-triggered: NO-NOISE -> 0 utterances every seed",
                   all(p["no_noise"]["n_utt"] == 0 for p in per), expect=True)
        vd.require("substrate-attributable: STORE-LESION collapses the utterance stream every seed",
                   all(p["checks"]["store_lesion_load_bearing"] for p in per), expect=True)
        vd.require("plasticity byte-frozen WITH STP on every seed (STP != a weight write)",
                   all(p["checks"]["weights_byte_frozen"] for p in per), expect=True)
        vd.require("STP is REAL fatigue (cp_stp_x dips < rested then recovers) every seed",
                   all(p["checks"]["real_fatigue"] for p in per), expect=True)
        vd.disabled("hebbian/BTSP plasticity during the wander", "the wander measures noise-seeded completion on a frozen store")
        decided = vd.decide(go)
        preconditions = decided["preconditions"]

        solo_note = ""
        if any(p.get("solo") for p in per):
            solos = [p["solo"] for p in per if p.get("solo")]
            all_solo = all(s["all_solo_ignite"] for s in solos)
            if all_solo:
                solo_note = (" SOLO-diagnosis: every basin ignites in ISOLATION -> the tail failures are COMPETITION "
                             "lock-in (STP is the correct fix).")
            else:
                solo_note = (" SOLO-diagnosis: a basin fails even in ISOLATION -> ABSOLUTE-threshold residual (the "
                             "§4 homeostatic-excitability lever is the named next rung).")

        verdict = (f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{len(per)} -- adaptation-driven itinerancy (ca3->ca3 "
                   f"short-term DEPRESSION) makes the noise-driven DMN wander visit ALL {a.n_mem} DISJOINT basins and "
                   f"the mouth speak about the WHOLE store. Ignition reliability (all-{a.n_mem}-ignite) "
                   f"{ignition_reliability:.2f} (mean {m_ig_bal:.1f}/{a.n_mem}) vs STP-lesion {m_ig_no:.1f}/{a.n_mem}; "
                   f"closed loop speaks about {m_concepts:.1f}/{a.n_mem} concepts (about-selected {m_about:.2f} vs "
                   f"SCRAMBLE {m_scr:.2f}); coherence member {m_member:.2f} vs random {m_random:.2f}; novel-share on "
                   f"{m_novel_on:.2f} vs reversed {m_novel_rv:.2f}"
                   f"{'; %.0f%% attributable to the curiosity gain' % (100 * attribution) if attribution is not None else ''}."
                   f"{solo_note}"
                   f"{' => the DMN itinerates through the FULL store and self-initiates an utterance about every concept.' if go else ' Per THE LAW: the failing method is banked, the §4 homeostatic-excitability equalization / n_ca3 scale lever is the named next; not a stop.'}")
    else:
        go = False; n_go = 0; ignition_reliability = 0.0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        vd = Verdict("self-initiated ALL-basins ignite + closed loop (6-seed)")
        vd.require("run completed without error", err is None, expect=True)
        preconditions = vd.decide(False)["preconditions"]

    summary = {"probe": "self_initiated_all_basins_ignite", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_mem": a.n_mem, "rest_steps": a.rest_steps, "acid_steps": a.acid_steps, "solo_steps": a.solo_steps,
               "gain_scale": a.gain_scale, "min_frac": a.min_frac, "stp": a.stp, "D": a.D,
               "ignition_reliability": ignition_reliability if (err is None and per) else 0.0,
               "curiosity_bias_attribution": attribution, "preconditions": preconditions,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[all-basins] VERDICT: {verdict}\n[all-basins] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
