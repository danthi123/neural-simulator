"""AFFECT MERGE — per-region OU-noise seam CLOSES the mapped affect boundary (Gate-B one-brain, 2026-08-13).

Rung-2b (`2026-08-13-per-region-param-het-cluster-GO.md`) MIGRATED metacog + pragmatic onto the shared spiking
pool byte-identically via `cfg.per_region_parameter_heterogeneity`, but AFFECT stayed an honest partial (BOUNDARY):
its mood-ladder read runs with `enable_ou_process=True` (OU noise is a `size=n` per-step GLOBAL draw -> a region's
noise slice is position-shifted) AND drives the global neuromodulator subsystem. The finding MEASURED the OU seam
(OU-on co-resident delta ~1.5e2 vs OU-off 0.0) and NAMED per-region OU + per-region neuromod scoping as the next
rungs.

THIS RUNG lands `cfg.per_region_ou_seed` (the guarded `sim/` edit: each region draws its per-step OU noise from
its OWN persistent host RNG stream, name-keyed via zlib.crc32, so a region's OU realization is invariant to
co-residents) and MIGRATES the REAL affect production organ (`affect_production_organ.AffectProductionOrgan`,
whose read is the sign-aware neural ladder differential rate(aff_pos_readout) - rate(aff_neg_readout) through the
`affect_out` gate) onto a co-resident pool.

WHAT THIS VERIFIES (SIM_BACKEND=numpy, so cp == numpy -> bit-exact):

 (1) BYTE-IDENTITY (flags ON). The affect organ is built STANDALONE vs with INERT (density-0, unwired) co-resident
     regions PREPENDED -- the whole brain (rf + faculties + ladder) shifts to a NON-ZERO offset, the exact
     perturbation a shared pool introduces, while the pads consume NO build_wiring_plan RNG (wiring byte-identical)
     and add NO cross-synapse. With per_region_ou_seed + per_region_parameter_heterogeneity +
     per_region_threshold_heterogeneity ON, the organ's PRODUCTION READ (`read_differential`) is BYTE-IDENTICAL
     merged-vs-co-resident (max delta 0.0) for BOTH a positive and a negative appraisal. 6/6 EXPECTED (GO).

 (2) FLAG LOAD-BEARING (flags OFF control). SAME comparison with the flags OFF -> the read DIVERGES (the OU +
     init seams are position-shifted). Confirms the flags are the fix, not a no-op.

 (3) FACULTY ALIVE (merged). On the co-resident pool the sign is preserved: a positive appraisal holds a POSITIVE
     ladder differential and a negative appraisal a NEGATIVE one (the graded staggered-bistable ladder still
     represents signed valence under co-residence).

HONEST FINDING vs the mapped boundary: the neuromodulator subsystem was NAMED as a second open seam, but it is
MEASURED here NOT to be a divergence source -- it reads region firing position-independently (rm.indices) and sets
concentrations by name, so given byte-identical firing (which per-region OU + param/threshold deliver) the neuromod
effects are byte-identical too. So affect closes on the OU seam alone (plus the already-landed param/threshold
seams); no per-region neuromod scoping was needed. That refines the rung-2b boundary to a GO.

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._per_region_ou_affect_merge_verify \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_per_region_ou_affect_merge_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.regions import BrainRegion

# Appraisals read on each build: one positive, one negative (both signs must be byte-identical + alive).
_APPR_POS = 0.6
_APPR_NEG = -0.6
_ALIVE_TOL = 1e-4   # |differential| must exceed this (with correct sign) for the faculty to count as alive


def _pads():
    """INERT (density-0, unwired, exc-only) co-resident regions PREPENDED to shift the whole affect brain to a
    non-zero offset. Distinct names so they never collide with rf / faculty / ladder region names."""
    return [BrainRegion(name="cx_merge_pad0", n_neurons=64, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="cx_merge_pad1", n_neurons=80, exc_fraction=1.0, internal_density=0.0)]


def _build_organ(seed, coresident, flags_on):
    from research.runners import affect_production_organ as AP
    from research.runners import _stageA_full_integration_derisk as SA
    from sim.backend import get_backend
    organ = AP.AffectProductionOrgan(seed=int(seed))
    organ.bridge, organ.comp, organ.idx, organ.snap = SA.build_one_brain(
        int(seed), with_faculties=True, co_resident_affect_ladder=True,
        coresident_regions=(_pads() if coresident else None),
        per_region_param_het=flags_on, per_region_thresh=flags_on, per_region_ou=flags_on)
    organ.xp, _ = get_backend()
    organ._built = True
    return organ


def run_seed(seed: int, verbose=True) -> dict:
    # ---- flags ON: standalone vs co-resident, both signs -> byte-identical + faculty alive ----
    solo = _build_organ(seed, coresident=False, flags_on=True)
    cor = _build_organ(seed, coresident=True, flags_on=True)
    s_pos = solo.read_differential(_APPR_POS)["differential"]
    c_pos = cor.read_differential(_APPR_POS)["differential"]
    s_neg = solo.read_differential(_APPR_NEG)["differential"]
    c_neg = cor.read_differential(_APPR_NEG)["differential"]
    d_on = max(abs(s_pos - c_pos), abs(s_neg - c_neg))
    byte_id = bool(d_on == 0.0)
    alive = bool(c_pos > _ALIVE_TOL and c_neg < -_ALIVE_TOL)   # merged read: positive holds +, negative holds -

    # ---- flags OFF control: same comparison (positive appraisal) -> diverges (flags load-bearing) ----
    solo_off = _build_organ(seed, coresident=False, flags_on=False)
    cor_off = _build_organ(seed, coresident=True, flags_on=False)
    d_off = abs(solo_off.read_differential(_APPR_POS)["differential"]
                - cor_off.read_differential(_APPR_POS)["differential"])

    go = bool(byte_id and alive)
    res = {"seed": int(seed), "maxdelta_on": float(d_on), "maxdelta_off": float(d_off),
           "byte_identical": byte_id, "alive": alive, "go": go,
           "solo_pos": float(s_pos), "cor_pos": float(c_pos),
           "solo_neg": float(s_neg), "cor_neg": float(c_neg)}
    if verbose:
        print(f"  [seed {seed}] AFFECT on-delta={d_on:.3e}(off={d_off:.3e}) "
              f"pos={c_pos:+.5f} neg={c_neg:+.5f} alive={alive} -> byte_id={byte_id} GO={go}", flush=True)
    return res


def _gate(n_go, n):
    return "GO" if ((n >= 6 and n_go >= 5) or (n < 6 and n_go == n)) else "BOUNDARY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== AFFECT MERGE — per-region OU-noise seam verify ===")
    results = [run_seed(s) for s in seeds]
    n = len(results)
    n_bi = sum(r["byte_identical"] for r in results)
    n_alive = sum(r["alive"] for r in results)
    n_go = sum(r["go"] for r in results)
    worst_off = max((r["maxdelta_off"] for r in results), default=0.0)

    print("\n=== VERDICT (affect merge) ===")
    print(f"  byte-identical (flags ON): {n_bi}/{n}   faculty alive: {n_alive}/{n}   GO: {n_go}/{n} -> {_gate(n_go, n)}")
    print(f"  flag-OFF control worst divergence: {worst_off:.3e} (>0 -> the flags are load-bearing)")

    from tools.verdict import Verdict
    v = Verdict("affect merge via per_region_ou_seed")
    v.require("affect_read_byte_identical_on", n_bi, expect=n,
              note="affect production read (sign-aware ladder differential) max delta 0.0 merged-vs-co-resident, "
                   "flags ON, both signs, all seeds")
    v.require("affect_faculty_alive_merged", n_alive, expect=n,
              note="on the co-resident pool a positive appraisal holds a positive differential, negative a negative")
    v.control("flags_load_bearing", treatment=worst_off, control=0.0, min_separation=0.0,
              note="with the flags OFF the read DIVERGES (position-shifted OU + init) -> not a no-op")
    decided = v.decide(go=(n_bi == n and n_alive == n and worst_off > 0.0), verbose=False)
    payload = {"mode": "per_region_ou_affect_merge", "n_seeds": n, "results": results,
               "n_byte_identical": n_bi, "n_alive": n_alive, "n_go": n_go, "worst_off_delta": worst_off,
               "verdict": decided["status"], "affect_verdict": _gate(n_go, n),
               "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
               "note": ("per_region_ou_seed (+ the landed param/threshold seams) makes the affect production read "
                        "invariant to co-residence on one co-stepped pool (byte-identical); the neuromodulator "
                        "subsystem is measured NOT to diverge -> affect closes on the OU seam alone (rung-2b "
                        "boundary refined to a GO)")}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
