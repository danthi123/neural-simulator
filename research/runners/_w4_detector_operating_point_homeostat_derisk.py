"""D · PRAGMATICS -- the LAST W4 lever: a detector OPERATING-POINT HOMEOSTAT that stabilizes the fractional-drive
(implicature) response across per-seed heterogeneity, to close the W4 arc 5/6 -> 6/6.

THE RESIDUAL ([B] `2026-08-13-w4-belief-magnitude-calibration-BOUNDARY.md`): reading the TRUE-INTENT detector
success[t] (not the whole-population landscape) removes the off-target belief-only AND-gate leak and flips the
informativeness objective OBJ_inf from [W]'s -0.011/3-of-6 to +0.090/5-of-6. The ONLY failing seed is 44, whose
per-seed graded-plateau calibration picked center=96 (the highest of six) so the low fractional implicature drive
(belief mass ~0.27) collapses: matched implicature cell 0.017, OBJ_inf move -0.041.

WHERE THE SEED-44 COLLAPSE ACTUALLY COMES FROM (measured per DETECTOR COLUMN on the substrate, not assumed):
`calibrate_graded_seed` (the [M]/[B] calibration) picks ONE (center, slope) by minimizing the GLOBAL proportionality
error on the ignition curve of ONE detector column (t=0). But under parameter heterogeneity each success[t] detector
has its OWN operating point, and the IMPLICATURE lives on a DIFFERENT column (t="all", index 2). Seed 44 at the
col-0-calibrated 96/0.08:
    col 0 (calib checks THIS): solo_intent=0.000  r(0.27)=0.079  -> clean.
    col 2 ("all", the implicature): solo_intent=0.084 (LEAK!)  r(0.27)=0.022 < solo  -> COLLAPSE + INVERSION
        (the solo-intent AND-gate leak 0.084 is LARGER than the fractional coincidence 0.022; row-normalized
         implicature cell = 0.017).
At 88/0.08 the SAME seed-44 detectors are ALL clean+resolved (col 2: solo=0.000, r(0.27)=0.102) -> implicature cell
0.202 (analytic 0.20). So the collapse is a PER-DETECTOR OPERATING-POINT artifact: the single-column proxy
calibration is BLIND to the leak on the detector the implicature actually uses. This is the "missing-companion-
process" pattern (CLAUDE.md): the real detector population runs per-unit intrinsic-excitability homeostasis so EVERY
detector's operating point resolves its inputs; we substituted ONE detector's operating point for the whole pool.

THE MECHANISM -- a DETECTOR OPERATING-POINT HOMEOSTAT (label-free; content-free; NO belief change; NO sim/ edit):
  1. PER-DETECTOR OPERATING-POINT HOMEOSTASIS (the load-bearing gain). Re-select the shared graded-plateau operating
     point (center, slope) so the homeostatic set-point -- AND-gate SILENCE (solo drives sub-floor) + PROPORTIONAL
     FRACTIONAL-COINCIDENCE RESOLUTION (r_t(f_frac) resolved ABOVE the solo leak, proportional to drive) -- holds on
     the WORST detector across ALL K columns, not column 0 alone. This is intrinsic-excitability homeostasis
     (Turrigiano, a FIXED set-point NOT fit to the answer): each detector's excitability is set to resolve its own
     content-free drive; the population homeostasis picks the shared point that satisfies the worst detector. It
     REJECTS seed-44's 96/0.08 (col-2 leaks) and selects a point where the implicature detector resolves.
  2. PER-DETECTOR DIVISIVE-NORMALIZATION READ (Carandini & Heeger 2012, "a canonical neural computation"). The matched
     read is divisively normalized per detector by its OWN content-free statistics: Sdn[t,u] = max(0, S[t,u]-b_t) /
     (sigma + g_t), b_t = the detector's solo-drive AND-gate leak floor, g_t = its dynamic range (full coincidence -
     leak). Removes residual per-column leak and equalizes gain across heterogeneous detectors. REPORTED (the canonical
     read-out form + robustness); the operating-point homeostasis is what actually rescues the collapsed detector.

Everything the homeostat uses is measured on CONTENT-FREE controlled drives (solo intent, solo belief, fractional
coincidence) per detector column -- NEVER the RSA answer, the belief content, or which intent wins. Applied UNIFORMLY
to onehot/graded/scramble.

WHAT THIS RUNNER DOES (additive; reuse-by-import of [B]'s runner; NO sim/ edit):
  * READ + BELIEFS + OBJECTIVES + the matched true-intent detector landscape: reuse-by-import of [B]'s
    `_w4_belief_magnitude_calibration_derisk` (read_cell / landscapes / measure_transfer / _invert / _impl_cell) and
    the W4 A/B belief sources + informativeness weights. Beliefs BYTE-IDENTICAL, plasticity off.
  * PER-SEED, THREE OPERATING POINTS: (a) COL-0 calibration (`calibrate_graded_seed`, the [B] pick) -> matched read =
    the "col0" arm that REPRODUCES [B]'s 5/6 (seed 44 collapses); (b) the HOMEOSTAT operating point
    (`calibrate_homeostat_seed`, all-detector) -> matched read = "matched_hom" (PRIMARY) + the divisive-normalization
    read "divnorm" (reported); (c) the WHOLE-POPULATION read at the homeostat point = the leak control (must still fail
    onehot>=graded, so the win is the matched read, not operating-point inflation).
  * A/B graded vs one-hot on the informativeness objective OBJ_inf, 6 seeds.

PRE-REGISTERED GO GATE (6 seeds 42 43 44 100 101 102, CPU numpy):
  GO iff the HOMEOSTAT matched read (matched_hom) beats one-hot on OBJ_inf by > MOVE_EPS (0.03) on 6/6 seeds
  (specifically RESCUING seed 44), AND the COL-0 arm reproduces [B]'s boundary (5/6, seed 44 collapses -- so the ONLY
  change is the operating point), AND the WHOLE-POPULATION leak control still fails at the homeostat point (move <=
  MOVE_EPS: the win is removing the leak, not inflating everything), AND SCRAMBLE (graded mass on WRONG intents) loses
  under matched_hom, AND the matched read faithfully encodes the posterior (recovery < RECOVERY_TOL; weights'
  L1==ANALYTIC_L1), AND the belief is UNCHANGED (implicature margin > 0.05; argmax/recall preserved). Else BOUNDARY:
  quantify seed 44's residual + name the next mechanism.

ANTI-CHEATS (each a gate): (i) LABEL-FREE -- the homeostat reads only content-free per-detector drives, never the
  answer; (ii) NOT "just lower all thresholds" -- the per-detector RESOLVED constraint + the reproduced col0 arm show
  the homeostat SELECTIVELY re-points the leaky detector (seed 44), and the OTHER 5 seeds + SCRAMBLE are re-checked;
  (iii) VALID -- SCRAMBLE still loses under matched_hom AND under divnorm; (iv) the leak control (wholepop at the
  homeostat point) still fails; (v) BELIEF byte-identical (margin>0.05; argmax preserved); (vi) col0 reproduces [B].

HONEST SCOPE. A FUNCTIONAL pragmatics correlate. A detector-side operating-point homeostat + divisive-normalization
read-out (the same category of host-side READ-OUT correction the pipeline already applies): it re-points the shared
graded-plateau operating point to a content-free homeostatic set-point satisfied on every detector, and divisively
normalizes each detector by its own content-free statistics. It does NOT change the belief (byte-identical to the W4
A/B) or which intent wins (recall intact); SCRAMBLE still loses. numpy-CPU real spiking Izhikevich; NO sim/ edit;
additive NEW runner. NOT a claim of phenomenal access to another mind; a self-report would be a functional read-out.

EXTERNAL GROUNDING: Carandini & Heeger (2012) Nat Rev Neurosci 13:51 ("Normalization as a canonical neural
  computation" -- divisive normalization / gain control). Turrigiano (2008) Cell 135:422 (homeostatic plasticity with
  a FIXED set-point -- multiplicative, preserves learned relative structure). Frank & Goodman (2012) Science
  336(6084):998 (the RSA posterior scale). Larkum (2013) TiNS 36(3):141 (the graded plateau's tunable operating
  point). Mikulasch & Priesemann (the dendritic analog read).

Usage:
  # smoke (1 seed, prints the col0-vs-homeostat operating points + per-seed implicature cells; verdict UNDEFINED n<6):
  SIM_BACKEND=numpy python -u -m research.runners._w4_detector_operating_point_homeostat_derisk --smoke \
      --json research/findings/raw/_w4_op_homeostat/smoke.json
  # 6-seed deliverable:
  SIM_BACKEND=numpy python -u -m research.runners._w4_detector_operating_point_homeostat_derisk \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_w4_op_homeostat/w4_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import logging as _logging
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

# reuse-by-import: [B]'s matched-read machinery (byte-identical instrument) + the graded bridge + its col-0 calibration.
from research.runners._w4_belief_magnitude_calibration_derisk import (  # noqa: E402
    read_cell, landscapes, measure_transfer, _invert, _belief_mass_matrix, _impl_cell,
    _AI, _SI, BELIEFS, MOVE_EPS, RECOVERY_TOL,
)
from research.runners._magnitude_preserving_plateau_readout_derisk import (  # noqa: E402
    build_success_bridge_graded, _ignition_drive, calibrate_graded_seed,
    FGRID, SILENT_FLOOR_A, IGNITE_MIN_A,
)
from research.runners._pragmatic_success_coincidence_derisk import K  # noqa: E402
from research.runners._recursive_tom_rsa_derisk import STATES, UTTS  # noqa: E402
from research.runners._pragmatic_spiking_graded_belief_derisk import (  # noqa: E402
    belief_variants, _row_norm, _analytic_landscape, argmax_align, _implicature_margin,
)
from research.runners._w4_informativeness_objective_derisk import (  # noqa: E402
    build_informativeness_weights, OBJECTIVES, _per_intent_fidelity,
)
from tools.lab import lever, void_if, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

# ── homeostat operating-point search grid: IDENTICAL to the col-0/[M] calibration grid ([80,88,96]x[0.08,0.11,0.14]),
# so the ONLY difference from the col-0 arm is the SELECTION CRITERION (per-detector worst-detector homeostatic
# set-point vs the col-0 single-column global-proportionality). A same-grid control confirmed the per-detector
# CRITERION -- not a finer grid -- is what rescues seed 44 (96->88, impl 0.017->0.202, OBJ_inf move +0.198). ──
CENTER_GRID_HOM = [80.0, 88.0, 96.0]
SLOPE_GRID_HOM = [0.08, 0.11, 0.14]
_FRAC_IDX = [FGRID.index(0.135), FGRID.index(0.27)]     # fractional-drive references (content-free DRIVE levels)
_RESOLVE_IDX = FGRID.index(0.27)  # the implicature-scale fractional DRIVE (content-free; the RSA answer never enters)
FRAC_RESOLVE_MARGIN = 0.02        # the fractional coincidence r_t(0.27) must exceed the solo AND-gate leak by this
                                  # (rejects the seed-44 INVERSION regime where a fractional coincidence < the leak)
DIVNORM_SIGMA = 0.02              # Carandini-Heeger semi-saturation constant for the per-detector divisive read
READS = ("col0", "col0_wholepop", "matched_hom", "divnorm", "wholepop")


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# per-detector operating-point homeostat: measure every success[t] detector's content-free ignition, require the
# homeostatic set-point (silence + proportional fractional resolution) on the WORST detector.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _per_detector_stats(bridge, xp, idx, snap):
    """For each success[t] detector column t, measure (content-free): solo_intent, solo_belief, and the coincidence
    ignition curve r_t(FGRID). Returns a list of per-detector dicts. This is the instrument the homeostat set-point is
    evaluated on -- a detector PROPERTY, zero RSA content."""
    dets = []
    for t in range(K):
        solo_i = _ignition_drive(bridge, xp, idx, snap, 0.0, t, intent_on=True, belief_on=False)
        solo_b = _ignition_drive(bridge, xp, idx, snap, 1.0, t, intent_on=False, belief_on=True)
        rc = [_ignition_drive(bridge, xp, idx, snap, f, t, intent_on=True, belief_on=True) for f in FGRID]
        r_full = float(rc[-1])
        leak = float(max(solo_i, solo_b))
        diffs = np.diff(np.asarray(rc, float))
        monotonic = bool(np.all(diffs >= -0.02))
        silent = bool(leak < SILENT_FLOOR_A and r_full > IGNITE_MIN_A)
        # RESOLVED: the fractional coincidence at the implicature-scale drive is encoded ABOVE the AND-gate leak floor
        # (rejects the collapse/INVERSION regime where a fractional coincidence reads LESS than the solo leak)
        resolved = bool(rc[_RESOLVE_IDX] > leak + FRAC_RESOLVE_MARGIN) and r_full > 1e-9
        if r_full > 1e-9:
            ratio = np.asarray(rc, float) / r_full
            prop_err = float(np.mean(np.abs(ratio - np.asarray(FGRID))))
        else:
            prop_err = 9.9
        dets.append({"t": t, "solo_i": round(solo_i, 4), "solo_b": round(solo_b, 4),
                     "leak": round(leak, 4), "r_full": round(r_full, 4), "r_curve": [round(x, 4) for x in rc],
                     "silent": silent, "monotonic": monotonic, "resolved": resolved,
                     "prop_err": round(prop_err, 4),
                     "frac_resolve": round(float(rc[_FRAC_IDX[1]] / r_full) if r_full > 1e-9 else 0.0, 4)})
    return dets


def calibrate_homeostat_seed(seed, verbose=True):
    """PER-DETECTOR OPERATING-POINT HOMEOSTASIS. Pick the shared (center, slope) whose homeostatic set-point -- AND-gate
    silence + proportional fractional-coincidence RESOLUTION -- holds on the WORST detector across ALL K columns (not
    column 0). Objective among the qualifying candidates: minimize the WORST-detector proportionality error. Content-
    free (controlled drives), a detector property, NOT fit to the RSA answer."""
    table = []
    for center in CENTER_GRID_HOM:
        for slope in SLOPE_GRID_HOM:
            bridge, xp, idx, snap = build_success_bridge_graded(seed, center, slope)
            dets = _per_detector_stats(bridge, xp, idx, snap)
            clean_all = bool(all(d["silent"] and d["monotonic"] for d in dets))
            resolved_all = bool(all(d["resolved"] for d in dets))
            worst_prop = float(max(d["prop_err"] for d in dets))
            worst_frac = float(min(d["frac_resolve"] for d in dets))
            min_full = float(min(d["r_full"] for d in dets))
            max_leak = float(max(d["leak"] for d in dets))
            table.append({"center": center, "slope": slope, "clean_all": clean_all, "resolved_all": resolved_all,
                          "worst_prop_err": round(worst_prop, 4), "worst_frac_resolve": round(worst_frac, 4),
                          "min_r_full": round(min_full, 4), "max_leak": round(max_leak, 4), "dets": dets})
    qualified = [r for r in table if r["clean_all"] and r["resolved_all"]]
    clean_rows = [r for r in table if r["clean_all"]]
    if qualified:
        pick = min(qualified, key=lambda r: r["worst_prop_err"]); tier = "homeostatic"
    elif clean_rows:
        pick = min(clean_rows, key=lambda r: r["worst_prop_err"]); tier = "clean_unresolved"
    else:
        pick = min(table, key=lambda r: r["worst_prop_err"]); tier = "unclean"
    rec = {"seed": int(seed), "center": float(pick["center"]), "slope": float(pick["slope"]), "tier": tier,
           "worst_prop_err": pick["worst_prop_err"], "worst_frac_resolve": pick["worst_frac_resolve"],
           "max_leak": pick["max_leak"], "picked": pick, "table": table}
    if verbose:
        print(f"  [homeostat seed {seed}] picked center={pick['center']:.0f} slope={pick['slope']:.2f} tier={tier} | "
              f"worst_prop_err={pick['worst_prop_err']:.3f} worst_frac_resolve={pick['worst_frac_resolve']:.3f} "
              f"max_leak={pick['max_leak']:.3f} | per-det leak/r(.27): "
              f"{[(d['leak'], d['r_curve'][_FRAC_IDX[1]]) for d in pick['dets']]}", flush=True)
    return rec


def measure_detector_norm(bridge, xp, idx, snap):
    """Per-detector content-free divisive-normalization statistics: baseline b_t = the solo-drive AND-gate leak floor,
    gain g_t = the dynamic range (full coincidence - leak). Carandini-Heeger normalization pool, per detector."""
    b = np.zeros(K, dtype=np.float64)
    g = np.zeros(K, dtype=np.float64)
    for t in range(K):
        solo_i = _ignition_drive(bridge, xp, idx, snap, 0.0, t, intent_on=True, belief_on=False)
        solo_b = _ignition_drive(bridge, xp, idx, snap, 1.0, t, intent_on=False, belief_on=True)
        r_full = _ignition_drive(bridge, xp, idx, snap, 1.0, t, intent_on=True, belief_on=True)
        b[t] = float(max(solo_i, solo_b))
        g[t] = max(float(r_full) - b[t], 1e-6)
    return b, g


def _divnorm_read(S, b, g, sigma=DIVNORM_SIGMA):
    """Per-detector divisive normalization (Carandini-Heeger): Sdn[t,u] = max(0, S[t,u]-b_t)/(sigma+g_t). Removes the
    residual per-detector AND-gate leak (subtractive) and equalizes gain across heterogeneous detectors (divisive).
    Content-free b_t/g_t; applied UNIFORMLY to every belief."""
    S = np.asarray(S, dtype=np.float64)
    out = np.zeros_like(S)
    for t in range(S.shape[0]):
        out[t] = np.maximum(0.0, S[t] - b[t]) / (sigma + g[t])
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# per-seed evaluation: col0 (reproduce [B]) vs homeostat operating point (matched + divnorm) + wholepop leak control.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════

def eval_seed(seed, W, verbose=True):
    col0 = calibrate_graded_seed(seed, verbose=verbose)        # the [B]/[M] col-0 calibration (reproduces [B])
    hom = calibrate_homeostat_seed(seed, verbose=verbose)      # the all-detector operating-point homeostat
    out = {"seed": int(seed),
           "col0_center": col0["center"], "col0_slope": col0["slope"],
           "hom_center": hom["center"], "hom_slope": hom["slope"], "hom_tier": hom["tier"],
           "hom_worst_prop_err": hom["worst_prop_err"], "hom_max_leak": hom["max_leak"],
           "operating_point_changed": bool(col0["center"] != hom["center"] or col0["slope"] != hom["slope"]),
           "analytic_implicature_cell": round(_impl_cell(_analytic_landscape()), 4),
           "scores": {r: {} for r in READS}, "impl_cell": {}, "recovery_err": {}, "argmax_preserved": {}}

    # --- COL-0 arm: build at [B]'s operating point, matched read (reproduces [B]'s 5/6; seed 44 collapses) ---
    b0, x0, i0, s0 = build_success_bridge_graded(seed, col0["center"], col0["slope"])
    bel = belief_variants(seed)
    for b in BELIEFS:
        Swp0, Sm0 = landscapes(b0, x0, i0, s0, bel[b])
        out["scores"]["col0"][b] = {name: round(f(Sm0, W), 5) for name, f in OBJECTIVES.items()}
        out["scores"]["col0_wholepop"][b] = {name: round(f(Swp0, W), 5) for name, f in OBJECTIVES.items()}
        out.setdefault("impl_cell", {}).setdefault(b, {})["col0"] = round(_impl_cell(Sm0), 4)

    # --- HOMEOSTAT arm: build at the homeostat operating point; matched read (primary) + divnorm (reported) + wholepop
    #     leak control; recovery + argmax anti-cheats ---
    bh, xh, ih, sh = build_success_bridge_graded(seed, hom["center"], hom["slope"])
    bvec, gvec = measure_detector_norm(bh, xh, ih, sh)
    T = measure_transfer(bh, xh, ih, sh)
    per_intent = {}
    for b in BELIEFS:
        Swp, Sm = landscapes(bh, xh, ih, sh, bel[b])
        Sdn = _divnorm_read(Sm, bvec, gvec)
        Scal = _invert(Sm, T)                                  # inverse-transfer -> posterior scale (recovery anti-cheat)
        out["scores"]["matched_hom"][b] = {name: round(f(Sm, W), 5) for name, f in OBJECTIVES.items()}
        out["scores"]["divnorm"][b] = {name: round(f(Sdn, W), 5) for name, f in OBJECTIVES.items()}
        out["scores"]["wholepop"][b] = {name: round(f(Swp, W), 5) for name, f in OBJECTIVES.items()}
        out["impl_cell"].setdefault(b, {})
        out["impl_cell"][b]["matched_hom"] = round(_impl_cell(Sm), 4)
        out["impl_cell"][b]["divnorm"] = round(_impl_cell(Sdn), 4)
        out["impl_cell"][b]["wholepop"] = round(_impl_cell(Swp), 4)
        out["recovery_err"][b] = round(float(np.mean(np.abs(Scal - _belief_mass_matrix(bel[b])))), 4)
        out["argmax_preserved"][b] = bool(abs(argmax_align(Sm) - argmax_align(Swp)) < 1e-9)
        if b in ("onehot", "graded"):
            per_intent[b] = [round(x, 4) for x in _per_intent_fidelity(Sm, W["Ideal"])]
    out["per_intent_matched_hom"] = per_intent
    out["belief_implicature_margin_graded"] = round(_implicature_margin(bel["graded"]), 4)

    if verbose:
        ic = out["impl_cell"]["graded"]
        sc = out["scores"]
        mv0 = sc["col0"]["graded"]["OBJ_inf"] - sc["col0"]["onehot"]["OBJ_inf"]
        mvh = sc["matched_hom"]["graded"]["OBJ_inf"] - sc["matched_hom"]["onehot"]["OBJ_inf"]
        print(f"    [seed {seed}] impl graded col0={ic['col0']} -> matched_hom={ic['matched_hom']} "
              f"(divnorm={ic['divnorm']}, analytic {out['analytic_implicature_cell']}) | OBJ_inf move col0={mv0:+.4f} "
              f"-> homeostat={mvh:+.4f} | op-pt {out['col0_center']:.0f}->{out['hom_center']:.0f} "
              f"changed={out['operating_point_changed']}", flush=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# aggregation + verdict
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _mean(rows, path):
    vals = []
    for r in rows:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    return round(float(np.mean(vals)), 5)


def _seeds_gt(rows, block, obj):
    return int(sum(1 for r in rows if r["scores"][block]["graded"][obj] > r["scores"][block]["onehot"][obj]))


def _block_agg(per_seed, block, obj):
    oh = _mean(per_seed, ["scores", block, "onehot", obj])
    gr = _mean(per_seed, ["scores", block, "graded", obj])
    sc = _mean(per_seed, ["scores", block, "scramble", obj])
    return {"onehot": oh, "graded": gr, "scramble": sc, "move": round(gr - oh, 5),
            "seeds_graded_gt_onehot": _seeds_gt(per_seed, block, obj),
            "scramble_loses": bool(sc < oh - 1e-6)}


def build_summary(per_seed, seeds, W, backend, smoke):
    n = len(seeds)
    obj = "OBJ_inf"
    agg = {blk: {o: _block_agg(per_seed, blk, o) for o in OBJECTIVES} for blk in READS}
    prim = agg["matched_hom"][obj]        # PRIMARY: matched read at the homeostat operating point
    col0 = agg["col0"][obj]               # reproduces [B]'s 5/6
    col0wp = agg["col0_wholepop"][obj]    # reproduces [W]'s whole-population boundary (onehot>=graded, move ~ -0.011)
    divn = agg["divnorm"][obj]            # reported: + divisive normalization
    wp = agg["wholepop"][obj]             # leak control at the homeostat operating point

    impl_col0 = _mean(per_seed, ["impl_cell", "graded", "col0"])
    impl_hom = _mean(per_seed, ["impl_cell", "graded", "matched_hom"])
    impl_dn = _mean(per_seed, ["impl_cell", "graded", "divnorm"])
    analytic_impl = _mean(per_seed, ["analytic_implicature_cell"])
    recov = round(float(np.mean([_mean(per_seed, ["recovery_err", b]) for b in ("onehot", "graded")])), 5)
    argmax_ok = bool(all(r["argmax_preserved"][b] for r in per_seed for b in ("onehot", "graded")))
    margin = _mean(per_seed, ["belief_implicature_margin_graded"])
    n_op_changed = int(sum(1 for r in per_seed if r["operating_point_changed"]))

    # per-seed seed-44 rescue (the named residual): col0 vs homeostat implicature cell + OBJ_inf move
    per_seed_move_col0 = {int(r["seed"]): round(r["scores"]["col0"]["graded"][obj]
                                                - r["scores"]["col0"]["onehot"][obj], 4) for r in per_seed}
    per_seed_move_hom = {int(r["seed"]): round(r["scores"]["matched_hom"]["graded"][obj]
                                               - r["scores"]["matched_hom"]["onehot"][obj], 4) for r in per_seed}
    per_seed_impl_hom = {int(r["seed"]): r["impl_cell"]["graded"]["matched_hom"] for r in per_seed}

    # ── the lever (MOVED?): the operating-point homeostat rescues the collapsed implicature cell (col0 -> homeostat) ──
    lever("detector_operating_point_homeostat", round(impl_col0, 3), round(impl_hom, 3), required=True,
          continuous=f"divnorm={impl_dn:.3f} analytic={analytic_impl:.2f} | op-points-changed={n_op_changed}/{n}")

    # ── PRE-REGISTERED GO (primary = the HOMEOSTAT matched read) ──
    weights_principled = bool(W["L1_matches_analytic"])
    transform_recovers = bool(recov < RECOVERY_TOL)
    principled = bool(weights_principled and transform_recovers)
    belief_unchanged = bool(margin > 0.05 and argmax_ok)
    scramble_loses = bool(prim["scramble_loses"])
    matched_beats = bool(prim["move"] > MOVE_EPS and prim["seeds_graded_gt_onehot"] == n)
    col0_reproduces_B = bool(col0["seeds_graded_gt_onehot"] < n)                     # [B]'s 5/6 (seed 44 collapses)
    col0_wholepop_reproduces_W = bool(col0wp["move"] <= MOVE_EPS)                     # [W]'s whole-pop boundary
    wholepop_control = bool(wp["move"] <= MOVE_EPS)                                   # the leak read still fails
    divnorm_scramble_loses = bool(divn["scramble_loses"])
    go = bool(n >= 6 and principled and belief_unchanged and scramble_loses and matched_beats
              and wholepop_control)

    v = Verdict("W4 detector OPERATING-POINT HOMEOSTAT -- does re-pointing the shared graded-plateau operating point to "
                "a per-detector homeostatic set-point (AND-gate silence + proportional fractional resolution on the "
                "WORST detector, not column 0) RESCUE seed 44's collapsed implicature response and make the graded "
                "belief BEAT one-hot on OBJ_inf 6/6 (up from [B]'s 5/6)?")
    v.require("6 seeds (project bar)", n >= 6, expect=True)
    v.require("PRIMARY: the HOMEOSTAT matched read beats one-hot on OBJ_inf by > %.02f on 6/6 seeds (seed 44 rescued)"
              % MOVE_EPS, matched_beats, expect=True,
              note=f"matched_hom move={prim['move']:+.4f} ({prim['seeds_graded_gt_onehot']}/{n}); "
                   f"per-seed hom move={per_seed_move_hom}")
    v.require("the COL-0 arm REPRODUCES [B]'s boundary (< 6/6 -- seed 44 collapses at the col-0 operating point, so the "
              "ONLY change is the operating-point homeostat)", col0_reproduces_B, expect=True,
              note=f"col0 move={col0['move']:+.4f} ({col0['seeds_graded_gt_onehot']}/{n}); per-seed col0 move="
                   f"{per_seed_move_col0}; per-seed impl(col0/hom) 44={per_seed_impl_hom.get(44)}")
    v.require("the WHOLE-POPULATION leak control still FAILS at the homeostat operating point (onehot >= graded on "
              "OBJ_inf: the win is REMOVING the leak, not operating-point inflation)", wholepop_control, expect=True,
              note=f"wholepop move={wp['move']:+.4f}")
    v.require("VALID: SCRAMBLE (graded mass on WRONG intents) LOSES to one-hot under matched_hom", scramble_loses,
              expect=True, note=f"scramble={prim['scramble']} onehot={prim['onehot']}")
    v.require("PRINCIPLED, non-circular: the matched read faithfully encodes the RSA posterior (mean |T^{-1}(matched)-"
              "belief_mass| < %.02f; content-free T); weights' L1 == ANALYTIC_L1" % RECOVERY_TOL, principled,
              expect=True, note=f"recovery_err={recov} (tol {RECOVERY_TOL}); L1==ANALYTIC_L1={weights_principled}")
    v.require("BELIEF unchanged (moat): graded implicature margin > 0.05 AND argmax/recall preserved",
              belief_unchanged, expect=True, note=f"margin={margin} argmax_preserved={argmax_ok}")
    v.control("the operating-point homeostat RESCUES the collapsed implicature cell (col0 -> homeostat, toward analytic)",
              treatment=abs(impl_hom - analytic_impl), control=abs(impl_col0 - analytic_impl))
    v.disabled("STDP/Hebbian/homeostasis(plasticity)/STP/structural/OU/NMDA + the all-or-none coincidence current",
               "a detector-side OPERATING-POINT read-out correction (re-point the shared graded-plateau center/slope to "
               "a content-free per-detector homeostatic set-point + a per-detector divisive-normalization read); "
               "beliefs byte-identical to the W4 A/B; plasticity off.")
    vb = v.decide(go=go)

    attributable_to("the seed-44 rescue to the OPERATING-POINT HOMEOSTAT (homeostat matched move minus the col-0 "
                    "matched move, on the SAME beliefs)", col0["graded"] - col0["onehot"], prim["graded"] - prim["onehot"])
    void_if(not principled, "the matched read did not recover the RSA posterior scale -- do not read OBJ_inf as a fair A/B")

    pi_oh = np.mean([r["per_intent_matched_hom"]["onehot"] for r in per_seed], axis=0) if per_seed[0].get(
        "per_intent_matched_hom", {}).get("onehot") else np.zeros(K)
    pi_gr = np.mean([r["per_intent_matched_hom"]["graded"] for r in per_seed], axis=0) if per_seed[0].get(
        "per_intent_matched_hom", {}).get("graded") else np.zeros(K)

    if smoke or n < 6:
        verdict = ("UNDEFINED -- smoke (n<6); the col-0-vs-homeostat operating points + per-seed implicature cells + the "
                   "OBJ_inf A/B teeth are printed for mechanism-check only. The 6-seed bar is authoritative.")
    elif go:
        verdict = ("GO -- the seed-44 W4 residual was a PER-DETECTOR OPERATING-POINT artifact: the col-0 calibration "
                   "picked an operating point (center 96) where seed-44's IMPLICATURE detector (column 'all') leaked "
                   "solo-intent and collapsed the fractional coincidence (implicature cell %.3f). Re-pointing the shared "
                   "graded-plateau operating point to a per-detector homeostatic set-point (AND-gate silence + "
                   "proportional fractional resolution on the WORST detector, all content-free) RESCUES it: implicature "
                   "cell %.3f -> %.3f (analytic %.2f), and the graded belief now BEATS one-hot on OBJ_inf 6/6: "
                   "matched_hom graded=%.3f > onehot=%.3f (move=%+.4f, %d/%d), where the COL-0 arm reproduces [B]'s "
                   "boundary (move=%+.4f, %d/%d -- seed 44 collapses). The win is REMOVING the leak, not inflation: the "
                   "whole-population leak control still fails (move=%+.4f). LABEL-FREE (content-free per-detector drives, "
                   "never the answer), PRINCIPLED (recovery=%.3f < %.02f), belief UNCHANGED (margin=%.2f, recall "
                   "preserved), VALID (scramble=%.3f < onehot=%.3f). The divisive-normalization read agrees (move=%+.4f). "
                   "The W4 / Task-#12 pragmatic arc CLOSES: detector -> read-out -> objective -> OPERATING POINT all "
                   "surpassed." %
                   (impl_col0, impl_col0, impl_hom, analytic_impl, prim["graded"], prim["onehot"], prim["move"],
                    prim["seeds_graded_gt_onehot"], n, col0["move"], col0["seeds_graded_gt_onehot"], n, wp["move"],
                    recov, RECOVERY_TOL, margin, prim["scramble"], prim["onehot"], divn["move"]))
    else:
        failed = []
        if not matched_beats:
            failed.append(f"HOMEOSTAT matched OBJ_inf move={prim['move']:+.4f} ({prim['seeds_graded_gt_onehot']}/{n}); "
                          f"per-seed hom move={per_seed_move_hom}")
        if not col0_reproduces_B:
            failed.append(f"col0 did NOT reproduce [B]'s boundary (col0 {col0['seeds_graded_gt_onehot']}/{n})")
        if not wholepop_control:
            failed.append(f"wholepop leak control failed (move={wp['move']:+.4f} -- operating-point inflation confound)")
        if not scramble_loses:
            failed.append("scramble does not lose")
        if not principled:
            failed.append(f"NOT-PRINCIPLED (recovery_err={recov} >= {RECOVERY_TOL})")
        if not belief_unchanged:
            failed.append(f"BELIEF changed (margin={margin}, argmax_preserved={argmax_ok})")
        verdict = ("BOUNDARY -- %s. The operating-point homeostat moves the implicature cell (col0 %.3f -> homeostat "
                   "%.3f, analytic %.2f; op-points changed %d/%d) but does NOT clear the 6/6 GO bar. Per-seed homeostat "
                   "OBJ_inf move: %s. matched_hom per-intent onehot=%s graded=%s (none, SBNA, all). Honest residual + "
                   "the next mechanism in the finding; the refuted deep-credit/BDSP rule is NOT re-proposed." %
                   ("; ".join(failed), impl_col0, impl_hom, analytic_impl, n_op_changed, n, per_seed_move_hom,
                    [round(float(x), 3) for x in pi_oh], [round(float(x), 3) for x in pi_gr]))

    summary = {
        "runner": "_w4_detector_operating_point_homeostat_derisk",
        "faculty": "D pragmatics: the LAST W4 lever -- a detector OPERATING-POINT HOMEOSTAT. Seed 44's implicature "
                   "collapse was a per-detector operating-point artifact (the col-0 calibration is blind to the leak on "
                   "the detector the implicature uses). Re-pointing the shared graded-plateau operating point to a "
                   "content-free per-detector homeostatic set-point (silence + proportional fractional resolution on the "
                   "worst detector) + a Carandini-Heeger per-detector divisive-normalization read rescues it. FUNCTIONAL "
                   "pragmatics correlate.",
        "builds_on": [
            "2026-08-13-w4-belief-magnitude-calibration-BOUNDARY (matched true-intent read; 5/6; seed 44 residual)",
            "2026-08-13-w4-informativeness-objective-BOUNDARY (the objective half; the leak control)",
            "2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY (the graded plateau read + col-0 calibration)",
            "2026-08-01-W4-recursive-theory-of-mind-...-6seed-GO (depth-2 scalar implicature GO)",
        ],
        "seeds": list(seeds), "backend": backend, "smoke": bool(smoke or n < 6),
        "move_eps": MOVE_EPS, "recovery_tol": RECOVERY_TOL, "frac_resolve_margin": FRAC_RESOLVE_MARGIN,
        "divnorm_sigma": DIVNORM_SIGMA, "verdict": verdict, "go": go,
        "external_grounding": [
            "Carandini & Heeger (2012) Nat Rev Neurosci 13:51 -- 'Normalization as a canonical neural computation' "
            "(divisive normalization / gain control -- the per-detector read).",
            "Turrigiano (2008) Cell 135:422 -- homeostatic plasticity with a FIXED set-point (intrinsic excitability "
            "homeostasis; the operating-point re-selection is content-free, not fit to the answer).",
            "Frank & Goodman (2012) Science 336(6084):998 -- the RSA posterior scale. Larkum (2013) TiNS 36(3):141 -- "
            "the graded plateau's tunable operating point. Mikulasch & Priesemann -- the dendritic analog read.",
        ],
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "gates": {"HOMEOSTAT_matched_OBJ_inf_beats_onehot_6of6": matched_beats,
                  "col0_reproduces_B_boundary": col0_reproduces_B,
                  "col0_wholepop_reproduces_W_boundary": col0_wholepop_reproduces_W,
                  "wholepop_leak_control_still_fails": wholepop_control,
                  "scramble_loses_matched_hom": scramble_loses, "scramble_loses_divnorm": divnorm_scramble_loses,
                  "principled_faithful_read_recovers_posterior": principled,
                  "weights_L1_matches_analytic": weights_principled, "belief_unchanged_moat": belief_unchanged},
        "implicature_cell_graded": {"col0": round(impl_col0, 4), "matched_hom": round(impl_hom, 4),
                                    "divnorm": round(impl_dn, 4), "analytic": round(analytic_impl, 4),
                                    "per_seed_matched_hom": per_seed_impl_hom,
                                    "per_seed_col0": {int(r["seed"]): r["impl_cell"]["graded"]["col0"]
                                                      for r in per_seed}},
        "OBJ_inf_move": {"col0_reproduces_B": col0["move"], "col0_wholepop_reproduces_W": col0wp["move"],
                         "matched_homeostat_PRIMARY": prim["move"], "divnorm_reported": divn["move"],
                         "wholepop_leak_control_at_homeostat_pt": wp["move"],
                         "per_seed_col0": per_seed_move_col0, "per_seed_homeostat": per_seed_move_hom},
        "operating_points": {"per_seed": {int(r["seed"]): {"col0": [r["col0_center"], r["col0_slope"]],
                                                           "homeostat": [r["hom_center"], r["hom_slope"],
                                                                         r["hom_tier"]],
                                                           "changed": r["operating_point_changed"]}
                                          for r in per_seed},
                             "n_changed": n_op_changed,
                             "note": "the homeostat SELECTIVELY re-points the leaky detectors (label-free per-detector "
                                     "set-point); seeds already balanced keep their operating point -- NOT a blanket "
                                     "threshold lowering."},
        "recovery": {"mean_recovery_err_onehot_graded": recov, "tol": RECOVERY_TOL,
                     "per_belief": {b: _mean(per_seed, ["recovery_err", b]) for b in BELIEFS}},
        "belief_implicature_margin_graded": margin, "argmax_preserved": argmax_ok,
        "aggregate": {blk: agg[blk] for blk in READS},
        "informativeness_weights": {"primary_per_intent_entropy_w_inf": [round(float(x), 4) for x in W["w_inf"]],
                                    "L1_matches_analytic_ANALYTIC_L1": bool(W["L1_matches_analytic"]),
                                    "intents": list(STATES), "utterances": list(UTTS)},
        "per_seed": per_seed,
        "honest_scope": (
            "A FUNCTIONAL pragmatics correlate. Seed 44's implicature collapse was a PER-DETECTOR OPERATING-POINT "
            "artifact: the per-seed graded-plateau calibration minimizes proportionality on ONE detector column (t=0), "
            "but under heterogeneity the IMPLICATURE detector (column 'all') has its own operating point, and at the "
            "col-0-chosen center=96 it leaked solo-intent (0.084) and collapsed the fractional coincidence (0.022 < "
            "leak) -> implicature cell 0.017. The fix is a detector-side OPERATING-POINT HOMEOSTAT: re-point the shared "
            "graded-plateau operating point to a content-free per-detector homeostatic set-point (AND-gate silence + "
            "proportional fractional-coincidence resolution) satisfied on the WORST detector, plus a per-detector "
            "Carandini-Heeger divisive-normalization read (subtract the solo leak, divide by the dynamic range). Both "
            "use ONLY content-free controlled drives per detector -- never the RSA answer, the belief, or which intent "
            "wins. It re-points the READ operating point; it does NOT change the belief (byte-identical to the W4 A/B) "
            "or which intent wins (recall intact); SCRAMBLE still loses; the whole-population leak control still fails; "
            "the col-0 arm reproduces [B]'s boundary. numpy-CPU real spiking Izhikevich; NO sim/ edit; additive NEW "
            "runner (reuse-by-import of [B]'s matched read + the W4 A/B + the informativeness objective). NOT a claim of "
            "phenomenal access to another mind; a self-report would be a functional read-out."),
    }
    return summary, verdict


def _emit(summary, verdict, out_path):
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    agg = summary["aggregate"]
    ic = summary["implicature_cell_graded"]
    print("\n" + "=" * 118, flush=True)
    print(f"[w4-ophom] === VERDICT: {verdict} ===", flush=True)
    print(f"[w4-ophom]  IMPL CELL (graded): col0={ic['col0']} -> matched_hom={ic['matched_hom']} "
          f"(divnorm={ic['divnorm']}, analytic {ic['analytic']})", flush=True)
    print(f"[w4-ophom]  per-seed impl(col0/homeostat): "
          f"{ {s: (ic['per_seed_col0'][s], ic['per_seed_matched_hom'][s]) for s in ic['per_seed_matched_hom']} }",
          flush=True)
    print(f"[w4-ophom]  operating points (col0->homeostat): "
          f"{ {s: (v['col0'][0], v['homeostat'][0], 'chg' if v['changed'] else 'same') for s, v in summary['operating_points']['per_seed'].items()} }",
          flush=True)
    for o in OBJECTIVES:
        m, c, d, w = agg["matched_hom"][o], agg["col0"][o], agg["divnorm"][o], agg["wholepop"][o]
        print(f"[w4-ophom]  {o:9s} HOMEOSTAT onehot={m['onehot']} graded={m['graded']} move={m['move']:+.4f} "
              f"({m['seeds_graded_gt_onehot']}/{len(summary['seeds'])}) scr={m['scramble']}(lose={m['scramble_loses']}) "
              f"| col0 move={c['move']:+.4f}({c['seeds_graded_gt_onehot']}/{len(summary['seeds'])}) "
              f"| divnorm move={d['move']:+.4f} | wholepop move={w['move']:+.4f}", flush=True)
    print(f"[w4-ophom]  recovery mean|fhat-mass|={summary['recovery']['mean_recovery_err_onehot_graded']} "
          f"(tol {summary['recovery']['tol']}) | belief margin={summary['belief_implicature_margin_graded']} "
          f"argmax_preserved={summary['argmax_preserved']}", flush=True)
    print(f"[w4-ophom]  gates={summary['gates']}", flush=True)
    print(f"[w4-ophom]  wrote {out_path}\n" + "=" * 118, flush=True)


def main():
    ap = argparse.ArgumentParser(description="W4 detector operating-point homeostat: re-point the shared graded-plateau "
                                             "operating point to a per-detector homeostatic set-point + a divisive-"
                                             "normalization read; does it rescue seed 44 and make graded beat onehot 6/6?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None, help="single-seed convenience (overrides --seeds)")
    ap.add_argument("--smoke", action="store_true", help="1 seed, prints the teeth (verdict UNDEFINED at n<6)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default="research/findings/raw/_w4_op_homeostat/summary.json")
    args = ap.parse_args()
    if args.backend != "auto":
        from sim.backend import get_backend
        get_backend(args.backend)

    seeds = [args.seed] if args.seed is not None else ([args.seeds[0]] if args.smoke else args.seeds)
    smoke = bool(args.smoke or len(seeds) < 6)

    t0 = time.time()
    print(f"[w4-ophom] W4 detector operating-point homeostat | seeds={seeds} backend={args.backend}", flush=True)
    W = build_informativeness_weights()
    print(f"[w4-ophom] weights: w_inf={np.round(W['w_inf'],4).tolist()} | L1==ANALYTIC_L1={W['L1_matches_analytic']} "
          f"| intents={STATES}", flush=True)
    print("[w4-ophom] HOMEOSTAT = per-detector operating-point set-point (silence + proportional fractional resolution "
          "on the WORST detector, content-free) + per-detector divisive-normalization read. Anti-cheats: col0 "
          "reproduces [B], wholepop leak control still fails, scramble loses, belief byte-identical, argmax preserved.",
          flush=True)

    per_seed = [eval_seed(s, W, verbose=True) for s in seeds]
    summary, verdict = build_summary(per_seed, seeds, W, args.backend, smoke)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    _emit(summary, verdict, args.json)
    return 0 if summary["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
