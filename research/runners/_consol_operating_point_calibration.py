"""Consolidation OPERATING-POINT CALIBRATION on the PHYSICALLY VALID substrate (2026-07-25).

WHY: the 2026-07-25 consolidation arc was run at `comp_apical_R=50.0` — a 333x miscalibration of a pA->mV units
constant (engine default 0.15) that parked `v_apical` at ~2e5 mV and, via `apical_g_couple_to_soma=5.0`, drove EVERY
soma. That produced a 93%-active "dense CA1 code" which the arc mis-diagnosed as a fundamental boundary. See
`research/findings/2026-07-25-CRITICAL-apical-R-333x-miscalibration-...md`. EVERY constant in that arc (core threshold,
btsp_lr, tag drive, the MSN phenotype) was fitted to that artifact's ~100x inflated activity and is therefore invalid.

This probe does NOT sweep from those constants. It MEASURES the valid substrate's own activity statistics and DERIVES
the operating point from them:

  1. physiological check   -- v_apical must sit within -90..+50 mV (else the substrate is invalid, stop).
  2. activity statistics   -- the per-cell CA1 spike-count distribution under an isolated tag: percentiles + the cell
                              count at each candidate threshold. This is what `core_thr_frac` must be derived FROM.
  3. derived core threshold-- pick the absolute spike threshold whose core size lands in a target band (default 8-20
                              cells), reported per fact, rather than inheriting 0.25*40=10 from the artifact regime.
  4. code quality at it    -- Jaccard + magnitude-free cosine specificity of the resulting cores (the separability the
                              write would have to exploit), with the raw per-fact masses (the MASS-artifact triad).

Output is a calibration record, NOT a GO/NO-GO: it fixes the constants a subsequent write experiment must use.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_operating_point_calibration --seed 42
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, CONSOLIDATED_FACTS)
from research.runners._consol_direct_weight_probe import BASE, _fire_under_tag, _jac
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
V_PHYSIO = (-90.0, 50.0)


def _cos_spec(F):
    U = F / np.maximum(np.linalg.norm(F, axis=1), 1e-12)[:, None]
    out = []
    for i in range(N):
        m = float(np.mean([float(U[i] @ U[j]) for j in range(N) if j != i]))
        out.append(round(1.0 / m, 2) if m > 1e-12 else float("inf"))
    return out


def run(seed, apical_R=0.15, gc_read=0.5, tag_drive=1500.0, steps=40,
        core_lo=8, core_hi=20, hippo_izh=""):
    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15,
             comp_kir_g=3.0, comp_v_hold=-50.0, comp_apical_R=float(apical_R), comp_gc_read=float(gc_read))
    if hippo_izh:      # default "" = the substrate's own hippocampal pyramidal phenotype (MSN was an artifact patch)
        a.update(hippo_izh_type=str(hippo_izh), hippo_izh_regions="dg,ca3,ca1")
    b = build_substrate(seed, SimpleNamespace(**a))
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    rm = b.region_manager
    ca1 = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)

    # (1) PHYSIOLOGICAL CHECK — the gate everything else depends on
    va = to_host(b.cp_v_apical) if getattr(b, "cp_v_apical", None) is not None else None
    v_ok = bool(va is not None and va.min() >= V_PHYSIO[0] and va.max() <= V_PHYSIO[1])

    # (2) ACTIVITY STATISTICS on the valid substrate
    F = np.stack([_fire_under_tag(b, t, ca1, drive=float(tag_drive), steps=int(steps))[0] for t in tags])
    counts_at = {int(t): [int((F[i] > t).sum()) for i in range(N)] for t in (0, 1, 2, 3, 5, 8, 10, 15)}
    pct = {p: [round(float(np.percentile(F[i], p)), 2) for i in range(N)] for p in (50, 90, 95, 99)}

    # (3) DERIVE the core threshold from the measured distribution (target band), not from the artifact's 0.25*40
    derived, best = None, None
    for t in range(1, 40):
        sizes = [int((F[i] > t).sum()) for i in range(N)]
        if all(core_lo <= s <= core_hi for s in sizes):
            derived = t; best = sizes; break
    if derived is None:                       # fall back to the threshold closest to the band midpoint
        mid = 0.5 * (core_lo + core_hi)
        t = min(range(1, 40), key=lambda t: abs(np.mean([(F[i] > t).sum() for i in range(N)]) - mid))
        derived, best = t, [int((F[i] > t).sum()) for i in range(N)]

    # (4) CODE QUALITY at the derived threshold (+ the mass triad)
    cores = {i: ca1[F[i] > derived] for i in range(N)}
    jac = float(np.mean([_jac(cores[i], cores[j]) for i in range(N) for j in range(i + 1, N)]))
    G = np.stack([(F[i] > derived).astype(np.float64) for i in range(N)])
    return dict(seed=int(seed), thr_hash=thr_hash, backend=BACKEND,
                apical_R=float(apical_R), gc_read=float(gc_read), hippo_izh=hippo_izh or "(default pyramidal)",
                v_apical_range=[round(float(va.min()), 2), round(float(va.max()), 2)] if va is not None else None,
                v_apical_physiological=v_ok,
                total_spikes=[round(float(F[i].sum()), 1) for i in range(N)],
                cells_above=counts_at, percentiles=pct,
                derived_core_threshold=int(derived), derived_core_sizes=best,
                core_jaccard=round(jac, 4), core_cosine_specificity=_cos_spec(G),
                rate_cosine_specificity=_cos_spec(F))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apical-R", type=float, default=0.15)
    ap.add_argument("--gc-read", type=float, default=0.5)
    ap.add_argument("--tag-drive", type=float, default=1500.0)
    ap.add_argument("--core-lo", type=int, default=8)
    ap.add_argument("--core-hi", type=int, default=20)
    ap.add_argument("--hippo-izh", type=str, default="")
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed, args.apical_R, args.gc_read, args.tag_drive,
            core_lo=args.core_lo, core_hi=args.core_hi, hippo_izh=args.hippo_izh)
    Path(f"{args.out}/opcalib_R{args.apical_R:g}_gc{args.gc_read:g}_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    print(f"[seed {args.seed}] backend={r['backend']} thr_hash={r['thr_hash']} R={r['apical_R']} gc_read={r['gc_read']} phenotype={r['hippo_izh']}")
    print(f"  (1) v_apical range={r['v_apical_range']} mV  PHYSIOLOGICAL={r['v_apical_physiological']}"
          + ("" if r['v_apical_physiological'] else "   <-- INVALID SUBSTRATE, stop here"))
    print(f"  (2) total CA1 spikes/window={r['total_spikes']}   percentiles(50/90/95/99)={ {k: v for k, v in r['percentiles'].items()} }")
    print(f"      cells above N spikes: " + "  ".join(f"{k}:{v}" for k, v in r['cells_above'].items()))
    print(f"  (3) DERIVED core threshold={r['derived_core_threshold']} spikes -> core sizes={r['derived_core_sizes']}  (target band {args.core_lo}-{args.core_hi})")
    print(f"  (4) core Jaccard={r['core_jaccard']}  core cosine-specificity={r['core_cosine_specificity']}  rate cosine-specificity={r['rate_cosine_specificity']}")
    print("OPERATING-POINT-CALIBRATION DONE", flush=True)


if __name__ == "__main__":
    main()
