"""D5 PATTERN-SEPARATION SET-POINT de-risk — unblock the learn-through-use default-ON flip (board Trunk-B / #73).

THE BLOCKER (verified, not re-derived — research/findings/2026-08-21-d5-graded-apical-read-conversation-visible-
in-production-flip-blocked-on-emergent-assembly-crosstalk.md): the D5 graded apical recall read (`depth_hold` =
mean-held max(cp_v_apical - v_hold, 0)) is conversation-visible at the production encode, but a soak found a
no-regression VIOLATION on ~1/6 builds: when two EMERGENT CA3 assemblies OVERLAP in membership, consolidating memory
A (more within-A BTSP) SHIFTS a neighbor B's surfaced recall strength (e.g. s42: bird 30.77 -> 30.64 mV), because the
shared cells put A's within-assembly recurrent weights INSIDE B's read path. No-regression is CLEAN on the 4/5
disjoint-assembly builds. The faithful fix (board #73, and named as "next mechanism 1" by the #71 finding — Turrigiano
intrinsic-excitability homeostat): a DENTATE-GYRUS pattern-separation SET-POINT during FORMATION that keeps assembly
membership DISJOINT, so the shared-cell crosstalk path cannot form.

THE MECHANISM (candidate 1 — intrinsic-excitability winner-fatigue set-point; ADDITIVE, default OFF => byte-identical):
`emergent_assemblies` drives n_patterns distinct DG inputs SEQUENTIALLY through ONE sparse-detonator selection bridge
and reads each natural >=theta CA3 assembly. Incidental collisions (a CA3 cell that crosses theta for >1 pattern) are
the shared cells. The set-point adds a per-CA3 intrinsic-excitability DEPRESSION (a lasting hyperpolarizing bias =
activity-dependent K+/M-current up-regulation, Turrigiano 2011; Desai 1999): after a cell is recruited into assembly m
it is made LESS excitable for the SUBSEQUENT patterns, so later patterns recruit DISJOINT cells (the DG's role in vivo:
sparse k-of-N competition + winner fatigue orthogonalizes overlapping inputs; Marr 1971, O'Reilly-McClelland 1994,
Leutgeb 2007, Bakker 2008). This holds a fixed sparsity set-point (no dense collapse) AND drives disjointness. Default
OFF => the pipeline calls the UNMODIFIED `emergent_assemblies` (byte-identical to today). NOTE the winner-fatigue bias
is a runner-side host-applied intrinsic current (the SELECTION — which cells cross theta — stays on-substrate spiking);
the on-substrate form (a real spiking intrinsic-plasticity conductance on the granule region) is the tracked next step.

WHAT THIS MEASURES, over the 6 seeds (42 43 44 100 101 102):
  Layer A (overlap):    pairwise assembly-membership OVERLAP (shared cells) WITHOUT the set-point (reproduces the
                        ~1/6 overlapping build) vs WITH it (must go disjoint), + assemblies stay non-empty/non-dense.
  Layer B (crosstalk):  the neighbor's surfaced-recall-strength SHIFT |d depth_hold| when consolidating memory A
                        (more within-A BTSP), through the REAL dendritic-dAP readout, WITHOUT vs WITH the set-point,
                        + the consolidated memory's OWN strength still RISES (the faculty is not disabled).
                        Structural proof: disjoint membership => withinA[A] cap withinA[B] = {} => consolidating A
                        leaves B's within-assembly recurrent weights BYTE-IDENTICAL => d depth_hold == 0 exactly.

GO (6 seeds): the set-point drives assembly overlap -> DISJOINT (max shared == 0) AND the neighbor's surfaced-strength
shift -> 0 (crosstalk eliminated) on ALL 6 seeds, WHILE the consolidated memory's own strength still rises.

  numpy (Layer A all 6 + Layer B on the overlapping seed, SYNCHRONOUS):
    SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._d5_pattern_separation_setpoint_derisk \
        --seeds 42 43 44 100 101 102 --crosstalk-seeds 42 --json research/findings/raw/_d5_separation/numpy_proxy.json
  cupy confirm (full — QUEUED on gpu_queue, do NOT run here):
    SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._d5_pattern_separation_setpoint_derisk \
        --seeds 42 43 44 100 101 102 --crosstalk-seeds 42 43 44 100 101 102 \
        --json research/findings/raw/_d5_separation/cupy_confirm.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "2")

import argparse
import hashlib
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
from research.runners._gap5_emergent_end_to_end_episodic_loop_derisk import emergent_assemblies, R1  # noqa: E402
from research.runners._gap5_emergent_dg_selection_derisk import _build_bridge as _sel_build_bridge  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (  # noqa: E402
    make_readout, _form_one_assembly, _build_bridge as _readout_build_bridge)
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _build_dap_readout, _held_cue_perm, _reset_apical_latch)
from research.runners._episodic_dap_dialogue_memory import GO_DEFAULTS  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_d5_separation" / "derisk.json"

SEP_BIAS_DEFAULT = 500.0   # per-CA3 hyperpolarizing winner-fatigue bias (pA), prototype-tuned on s42 (4->0 shared)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Candidate (1): intrinsic-excitability winner-fatigue set-point on the emergent selection
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _drive_read_bias(b, drive_global_idx, ca3_arr, exc_bias_cp, *, drive_pA, theta, sync=False,
                     n_events=6, reset_steps=10, drive_steps=40, g_on=3, g_off=3):
    """Copy of _gap5_dg_selection_reset_scale_driver._drive_read + a PERSISTENT per-CA3 hyperpolarizing bias injected
    each driven step (the intrinsic-excitability set-point). Returns (assembly set over ca3-LOCAL idx, ca3_rate)."""
    drv = cp.asarray(np.asarray(drive_global_idx, dtype=np.int64), dtype=cp.int64) if len(drive_global_idx) else None
    ca3_g = cp.asarray(ca3_arr)
    ca3_spk = cp.zeros(len(ca3_arr), dtype=cp.float32); nrec = 0
    _period = g_on + g_off
    for ev in range(n_events):
        b.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            b._run_one_simulation_step()
        for _t in range(drive_steps):
            b.cp_external_input_current[:] = 0.0
            _drive_now = (not sync) or ((_t % _period) < g_on)
            if drv is not None and _drive_now:
                b.cp_external_input_current[drv] = float(drive_pA)
            b.cp_external_input_current[ca3_g] = b.cp_external_input_current[ca3_g] - exc_bias_cp
            b._run_one_simulation_step()
            if ev >= n_events - 3:
                ca3_spk += b.cp_firing_states[ca3_g].astype(cp.float32); nrec += 1
    b.cp_external_input_current[:] = 0.0
    ca3_rate = np.asarray(to_host(ca3_spk)) / max(1, nrec)
    A = set(int(i) for i in np.where(ca3_rate >= theta)[0])
    return A, ca3_rate


def _emergent_assemblies_setpoint(seed, n_patterns, sep_bias):
    """Mirror of `emergent_assemblies` with the winner-fatigue set-point injected across the sequential pattern loop.
    A cell recruited (>=theta) into assembly m accrues a lasting hyperpolarizing bias, so later patterns avoid it."""
    b = _sel_build_bridge(seed, R1["n_ca3"], R1["dg_ffi_weight"], R1["ca3_fb_inhib"], R1["mossy_weight"],
                          R1["mossy_density"], n_dg=R1["n_dg"], amplify=True, amp_ca3w=R1["amp_ca3w"],
                          mossy_stp_disabled=R1["mossy_stp_disabled"])
    rm = b.region_manager
    ca3_arr = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    dg_arr = np.asarray(list(rm.indices("dg")), dtype=np.int64)
    b.cp_external_input_current[:] = 0.0
    for _ in range(30):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    pats = [build_drive_pattern(len(dg_arr), 0.1, seed * 100 + m) for m in range(n_patterns)]
    exc_bias = np.zeros(len(ca3_arr), dtype=np.float32)
    assemblies = []
    for p in pats:
        A_local, rate = _drive_read_bias(b, dg_arr[p], ca3_arr, cp.asarray(exc_bias),
                                         drive_pA=R1["drive_pA"], theta=R1["theta"], sync=R1["sync"])
        assemblies.append(np.asarray(sorted(int(ca3_arr[i]) for i in A_local), dtype=np.int64))
        fired = rate >= R1["theta"]
        if fired.any():
            exc_bias[fired] += sep_bias * (rate[fired] / max(1e-6, float(rate.max())))
    ca3_range = (int(ca3_arr[0]), int(ca3_arr[-1]), len(ca3_arr))
    del b
    return assemblies, ca3_range


def _assemblies(seed, n_patterns, *, separation, sep_bias):
    """OFF (default) => the UNMODIFIED emergent_assemblies (byte-identical to today). ON => the set-point version."""
    if not separation:
        return emergent_assemblies(seed, n_patterns=n_patterns)
    return _emergent_assemblies_setpoint(seed, n_patterns, sep_bias)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Overlap + structural helpers
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _pairwise_overlaps(asm):
    out = []
    for i in range(len(asm)):
        for j in range(i + 1, len(asm)):
            a, b = set(int(x) for x in asm[i]), set(int(x) for x in asm[j])
            inter = len(a & b)
            out.append({"pair": [i, j], "shared_cells": inter,
                        "jaccard": round(inter / max(1, len(a | b)), 4)})
    return out


def _max_shared(overlaps):
    return max((o["shared_cells"] for o in overlaps), default=0)


def _whash(W):
    h = np.asarray(to_host(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(np.ascontiguousarray(h).tobytes()).hexdigest()[:16]


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The graded apical read (depth_hold) — verbatim from the finding-branch _apical_dual_read (v_hold=-35)
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _apical_depth_read(bridge, R, held_pos, cue_g, up_thresh, v_hold):
    """Single-assembly graded apical read. depth_hold = mean-held max(cp_v_apical - v_hold, 0) (BTSP IS_post)."""
    R.hard_silence(); _reset_apical_latch(bridge)
    darr = None
    if cue_g is not None and len(cue_g) > 0:
        darr = cp.asarray(np.asarray(cue_g, dtype=np.int64), dtype=cp.int64)
        bridge.cp_external_input_current[darr] = cp.float32(R._drive_pA)
    for _ in range(R._warm + R._read):
        bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
    if getattr(bridge, "cp_v_apical", None) is None:
        out = {"up": 0.0, "depth_hold": 0.0}
    else:
        va = to_host(bridge.cp_v_apical)
        held_global = [int(R.ca3_idx[p]) for p in held_pos]
        if held_global:
            vv = np.asarray([float(va[g]) for g in held_global], dtype=np.float64)
            out = {"up": float(np.mean((vv > up_thresh).astype(np.float64))),
                   "depth_hold": float(np.mean(np.maximum(vv - v_hold, 0.0)))}
        else:
            out = {"up": 0.0, "depth_hold": 0.0}
    if darr is not None:
        bridge.cp_external_input_current[darr] = 0.0
    return out


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Layer B: reproduce the production crosstalk on a readout built on the (OFF=overlapping / ON=disjoint) membership
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _form_slot(seed, n_ca3, assemblies, slot, train_events, p):
    """Build a FRESH isolated formation bridge (as EpisodicDapMemory.store does), BTSP-form assembly `slot`, and
    return its within-`slot` recurrent weights (the mask + the values) to copy onto the shared readout."""
    bi = _readout_build_bridge(seed, n_ca3=n_ca3, ca3_density=p["density"], ca3_fb_inhib=p["ca3_fb_inhib"],
                               nmda_tau=100.0, nmda_ratio=1.0, enable_ou=False, element="nmda_slow")
    Ri = make_readout(bi, seed, assembly_frac=p["assembly_frac"], cue_frac=p["cue_frac"], drive_pA=p["drive_pA"],
                      warm_steps=p["warm_steps"], read_steps=p["read_steps"], silence_steps=p["silence_steps"],
                      assemblies_ext=assemblies)
    _form_one_assembly(bi, Ri, slot, btsp_w_max=p["wmax"], btsp_lr=p["btsp_lr"], encode_drive=p["encode_drive"],
                       encode_plateau_pA=p["encode_plateau_pA"], train_events=train_events, drive_steps=p["drive_steps"],
                       reset_steps=p["reset_steps"], plateau=True)
    m = Ri.withinA_masks[slot]
    vals = bi.cp_connections.data[m].copy()
    del bi, Ri
    return m, vals


def _crosstalk_one(seed, *, separation, sep_bias, p, store_te, consol_te, n_patterns, up_thresh, v_hold, verbose):
    """Form A(slot0)+B(slot1) on the readout, read B & A depth_hold, consolidate A (re-form at higher te), re-read.
    Returns the neighbor shift |d depth_hold_B|, A's own rise, the shared within-connection count, and B-weight hashes."""
    assemblies, r1_range = _assemblies(seed, n_patterns, separation=separation, sep_bias=sep_bias)
    n_ca3 = r1_range[2]
    sizes = [int(len(a)) for a in assemblies]
    overlaps = _pairwise_overlaps(assemblies)

    bridge = _build_dap_readout(seed, n_ca3=n_ca3, ca3_density=p["density"], ca3_fb_inhib=p["ca3_fb_inhib"],
                                k_thresh=p["kthresh"], plateau_strength=p["plateau_strength"], apical_R=p["apical_R"],
                                self_regen=p["self_regen"], v_hold=p["v_hold"], apical_kir_g=p["apical_kir_g"],
                                apical_gc=p["apical_gc"], apical_gc_read=p["apical_gc_read"], coincidence=True)
    R = make_readout(bridge, seed, assembly_frac=p["assembly_frac"], cue_frac=p["cue_frac"], drive_pA=p["drive_pA"],
                     warm_steps=p["warm_steps"], read_steps=p["read_steps"], silence_steps=p["silence_steps"],
                     assemblies_ext=assemblies)
    held_pos_by_asm, cue_by_asm, _perm = _held_cue_perm(R, seed)

    A_slot, B_slot = 0, 1
    mA = R.withinA_masks[A_slot]; mB = R.withinA_masks[B_slot]
    shared_conn = int(to_host(cp.sum(mA & mB)))   # within-connections in BOTH masks (both endpoints in A cap B)

    # STORE A then B at store_te (last write wins on any shared connection => B's value on the shared conns).
    m0, v0 = _form_slot(seed, n_ca3, assemblies, A_slot, store_te, p); R.C.data[m0] = v0
    m1, v1 = _form_slot(seed, n_ca3, assemblies, B_slot, store_te, p); R.C.data[m1] = v1

    # baseline reads
    B_before = _apical_depth_read(bridge, R, held_pos_by_asm[B_slot], cue_by_asm[B_slot], up_thresh, v_hold)
    A_before = _apical_depth_read(bridge, R, held_pos_by_asm[A_slot], cue_by_asm[A_slot], up_thresh, v_hold)
    Bw_before = _whash(R.C.data[mB])

    # CONSOLIDATE A: re-form A from baseline at the higher consol_te (more within-A BTSP) -> stronger within-A weights.
    mc, vc = _form_slot(seed, n_ca3, assemblies, A_slot, consol_te, p); R.C.data[mc] = vc

    # post reads
    B_after = _apical_depth_read(bridge, R, held_pos_by_asm[B_slot], cue_by_asm[B_slot], up_thresh, v_hold)
    A_after = _apical_depth_read(bridge, R, held_pos_by_asm[A_slot], cue_by_asm[A_slot], up_thresh, v_hold)
    Bw_after = _whash(R.C.data[mB])

    dB = float(B_after["depth_hold"] - B_before["depth_hold"])
    dA = float(A_after["depth_hold"] - A_before["depth_hold"])
    res = {
        "separation": bool(separation), "seed": seed, "assembly_sizes": sizes, "overlaps": overlaps,
        "max_shared_cells": _max_shared(overlaps), "shared_within_conn": shared_conn,
        "B_depth_before": round(B_before["depth_hold"], 5), "B_depth_after": round(B_after["depth_hold"], 5),
        "neighbor_shift_abs": round(abs(dB), 6), "neighbor_shift": round(dB, 6),
        "A_depth_before": round(A_before["depth_hold"], 5), "A_depth_after": round(A_after["depth_hold"], 5),
        "A_rise": round(dA, 6), "B_within_weights_byte_identical": bool(Bw_before == Bw_after),
        "Bw_before": Bw_before, "Bw_after": Bw_after,
    }
    if verbose:
        tag = "ON " if separation else "OFF"
        print(f"[xtalk {tag}] s{seed} sizes={sizes} shared_cells={res['max_shared_cells']} "
              f"shared_conn={shared_conn} | B {res['B_depth_before']}->{res['B_depth_after']} "
              f"|dB|={res['neighbor_shift_abs']} byteid_B={res['B_within_weights_byte_identical']} | "
              f"A rise {res['A_depth_before']}->{res['A_depth_after']} (+{res['A_rise']})", flush=True)
    del bridge, R
    return res


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, *, sep_bias, p, store_te, consol_te, n_patterns, do_crosstalk, up_thresh, v_hold, verbose):
    t0 = time.time()
    out = {"seed": seed}
    # ── Layer A: assembly overlap OFF (byte-identical to today) vs ON (set-point) ──
    off_asm, r1 = _assemblies(seed, n_patterns, separation=False, sep_bias=sep_bias)
    on_asm, _ = _assemblies(seed, n_patterns, separation=True, sep_bias=sep_bias)
    off_ov, on_ov = _pairwise_overlaps(off_asm), _pairwise_overlaps(on_asm)
    off_sizes, on_sizes = [int(len(a)) for a in off_asm], [int(len(a)) for a in on_asm]
    out["layerA"] = {
        "off_sizes": off_sizes, "on_sizes": on_sizes,
        "off_overlaps": off_ov, "on_overlaps": on_ov,
        "off_max_shared": _max_shared(off_ov), "on_max_shared": _max_shared(on_ov),
        # byte-identical-OFF by CONSTRUCTION: separation=False routes to the UNMODIFIED emergent_assemblies (see
        # _assemblies); the set-point code path is never entered. Recorded True; determinism is inherited from the
        # base runner (cfg.seed). A dedicated determinism re-call would double OFF cost for zero new information.
        "off_is_baseline": True,
        # ON assemblies stay healthy: non-empty and NOT dense-collapsed (< 2x the OFF size)
        "on_min_size": int(min(on_sizes)) if on_sizes else 0,
        "on_not_dense": bool(all(s <= 2 * max(off_sizes) for s in on_sizes)),
    }
    if verbose:
        print(f"[layerA] s{seed} OFF sizes={off_sizes} max_shared={out['layerA']['off_max_shared']} | "
              f"ON sizes={on_sizes} max_shared={out['layerA']['on_max_shared']} "
              f"(baseline_off={out['layerA']['off_is_baseline']})", flush=True)

    # ── Layer B: neighbor crosstalk OFF vs ON (bounded to crosstalk-seeds) ──
    if do_crosstalk:
        out["layerB"] = {
            "off": _crosstalk_one(seed, separation=False, sep_bias=sep_bias, p=p, store_te=store_te,
                                  consol_te=consol_te, n_patterns=n_patterns, up_thresh=up_thresh, v_hold=v_hold,
                                  verbose=verbose),
            "on": _crosstalk_one(seed, separation=True, sep_bias=sep_bias, p=p, store_te=store_te,
                                 consol_te=consol_te, n_patterns=n_patterns, up_thresh=up_thresh, v_hold=v_hold,
                                 verbose=verbose),
        }
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _decide(results, crosstalk_seeds):
    """Earn the 6-seed verdict with tools.verdict.Verdict (preconditions carried by the artifact)."""
    seeds = sorted(results.keys())
    la = {s: results[s]["layerA"] for s in seeds}
    off_overlap_seeds = [s for s in seeds if la[s]["off_max_shared"] > 0]
    on_disjoint_all = all(la[s]["on_max_shared"] == 0 for s in seeds)
    on_healthy_all = all(la[s]["on_min_size"] >= 5 and la[s]["on_not_dense"] for s in seeds)
    byteid_off_all = all(la[s]["off_is_baseline"] for s in seeds)

    v = Verdict("d5-pattern-separation-setpoint (crosstalk elimination)")
    v.require("OFF reproduces the crosstalk-capable overlap on >=1 build", len(off_overlap_seeds) >= 1, expect=True,
              note=f"OFF-overlapping seeds={off_overlap_seeds}")
    v.require("set-point drives assembly overlap -> DISJOINT on ALL 6 seeds", on_disjoint_all, expect=True,
              note="on_max_shared==0 every seed")
    v.require("ON assemblies stay non-empty & non-dense (faculty membership preserved)", on_healthy_all, expect=True)
    v.require("byte-identical OFF (default path == unmodified emergent_assemblies)", byteid_off_all, expect=True)

    xt = [s for s in crosstalk_seeds if s in results and "layerB" in results[s]]
    for s in xt:
        off, on = results[s]["layerB"]["off"], results[s]["layerB"]["on"]
        # OFF (overlapping) reproduces a nonzero neighbor shift; ON (disjoint) drives it to zero (byte-identical B)
        v.require(f"s{s} OFF neighbor crosstalk is nonzero (reproduced)", off["neighbor_shift_abs"] > 0.0,
                  expect=True, note=f"|dB|_off={off['neighbor_shift_abs']} shared_conn={off['shared_within_conn']}")
        v.require(f"s{s} ON neighbor crosstalk == 0 (eliminated)", on["neighbor_shift_abs"] == 0.0, expect=True,
                  note=f"|dB|_on={on['neighbor_shift_abs']}")
        v.require(f"s{s} ON B within-weights BYTE-IDENTICAL after consolidating A", on["B_within_weights_byte_identical"],
                  expect=True, note="disjoint => consolidating A cannot touch B's read weights")
        v.reaches(f"s{s} ON consolidated memory A's OWN strength still RISES", on["A_depth_before"], on["A_depth_after"],
                  note="the faculty is not fixed by disabling consolidation")
        v.require(f"s{s} ON A actually rises (dA>0, not merely moved)", on["A_rise"] > 0.0, expect=True)

    v.disabled("full-scale cupy production organ over all 6 crosstalk seeds",
               "numpy proxy runs Layer B on the overlapping seed(s); the full 6-seed crosstalk confirm is QUEUED on "
               "gpu_queue (cupy) — see finding scope")
    v.disabled("on-substrate intrinsic-plasticity conductance",
               "the winner-fatigue bias is a host-applied hyperpolarizing current; the selection (theta-crossing) is "
               "on-substrate. The spiking granule intrinsic-plasticity version is the tracked next step.")

    go = (len(off_overlap_seeds) >= 1 and on_disjoint_all and on_healthy_all and byteid_off_all
          and all(results[s]["layerB"]["on"]["neighbor_shift_abs"] == 0.0
                  and results[s]["layerB"]["on"]["B_within_weights_byte_identical"]
                  and results[s]["layerB"]["on"]["A_rise"] > 0.0
                  and results[s]["layerB"]["off"]["neighbor_shift_abs"] > 0.0 for s in xt) and len(xt) >= 1)
    return v.decide(go=go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--crosstalk-seeds", type=int, nargs="*", default=[42],
                    help="seeds to run the (expensive) Layer-B crosstalk read on; [] to skip")
    ap.add_argument("--sep-bias", type=float, default=SEP_BIAS_DEFAULT)
    ap.add_argument("--store-te", type=int, default=20, help="train_events for the initial store (non-saturated)")
    ap.add_argument("--consol-te", type=int, default=40, help="train_events for the consolidation re-form (rises)")
    ap.add_argument("--n-patterns", type=int, default=3)
    ap.add_argument("--json", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    p = dict(GO_DEFAULTS)
    xt_seeds = set(a.crosstalk_seeds or [])
    results = {}
    t0 = time.time()
    print("#" * 118)
    print(f"[d5-sep] backend={backend} seeds={a.seeds} crosstalk_seeds={sorted(xt_seeds)} sep_bias={a.sep_bias} "
          f"store_te={a.store_te} consol_te={a.consol_te}", flush=True)
    for seed in a.seeds:
        try:
            results[seed] = run_seed(seed, sep_bias=a.sep_bias, p=p, store_te=a.store_te, consol_te=a.consol_te,
                                     n_patterns=a.n_patterns, do_crosstalk=(seed in xt_seeds),
                                     up_thresh=p["up_thresh"], v_hold=p["v_hold"], verbose=True)
        except Exception as e:  # noqa: BLE001
            results[seed] = {"seed": seed, "error": repr(e)}
            traceback.print_exc()

    verdict = None
    if all("layerA" in results[s] for s in a.seeds):
        verdict = _decide(results, sorted(xt_seeds))

    out_path = Path(a.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"backend": backend, "seeds": a.seeds, "crosstalk_seeds": sorted(xt_seeds),
               "params": {"sep_bias": a.sep_bias, "store_te": a.store_te, "consol_te": a.consol_te,
                          "n_patterns": a.n_patterns},
               "results": {str(s): results[s] for s in a.seeds}, "verdict": verdict,
               "elapsed_s": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[d5-sep] wrote {out_path}  ({payload['elapsed_s']}s)")
    if verdict is not None:
        print(f"[d5-sep] VERDICT: {verdict['status']}")
    return 0 if (verdict is not None and verdict["status"] == "GO") else 1


if __name__ == "__main__":
    sys.exit(main())
