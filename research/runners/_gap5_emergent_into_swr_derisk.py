"""gap#5 — feed the EMERGENT-DG-selected CA3 assemblies into the CLOSED SWR generative-replay readout, and test whether
the CA1 specificity SURVIVES (the last open piece of gap#5: the SWR stack was validated on PRE-ASSIGNED disjoint random
assemblies; the emergence bar requires the assemblies be SELECTED FROM EXPERIENCE by the DG/mossy front end).

RUN (GPU, phase-2 Schaffer STP-off is REQUIRED for the readout to fire — the CLOSED mechanism):
  SIM_BACKEND=cupy SWR_PHASE2_NOSTP=1 python -m research.runners._gap5_emergent_into_swr_derisk --seeds 42 43 44 100 101 102

Pipeline (VERIFY-FIRST at every stage — never feed a downstream run whose upstream positive control failed):
  (1) INDEX-SPACE match: the emergent `_build_bridge(amplify,n_ca3=2000)` bridge and the SWR `run()`-style `_build`
      bridge must share IDENTICAL ca3 GLOBAL indices, else the cross-bridge `assemblies_ext` globals are invalid.
  (2) SELECT n_mem emergent assemblies on the emergent bridge; convert CA3-LOCAL -> GLOBAL via ca3_idx[local].
      GATE the feed on SPARSE (|A| in [3,60]) + SEPARATED (pairwise binary cos < 0.4) -- if the selection does not
      produce usable assemblies (silent, or a whole-network avalanche), print BLOCKED and STOP (do not fabricate).
  (3) FEED the selected globals as `assemblies_ext=` to the reproduced SWR-CLOSED `run()` (learned Schaffer + E%-max
      read); measure ca1_match/ca1_cross. ANTI-CHEATS: no-learn (dense-random Schaffer) collapses to a near-tie
      (learned Schaffer load-bearing); shuffled-assembly control drops match.

NO sim/ edit. Reuse-by-import of the SWR runner `run()` (cross-bridge `assemblies_ext` hook) + the emergent `_select`.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._riii_ca3_synchronous_assembly_derisk import run as swr_run  # noqa: E402
from research.runners._riii_ca3_coincidence_completion_derisk import _build  # noqa: E402
from research.runners._gap5_emergent_dg_selection_derisk import _build_bridge, _select  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402

cp, _ = get_backend()

# THE SWR-CLOSED config (2026-07-19 6/6 GO; = the bistable-completion ENCODING base + the SWR sparse+synchronous
# overrides + learned Schaffer + E%-max read). Firing REQUIRES env SWR_PHASE2_NOSTP=1 (phase-2 Schaffer STP-off).
# `assembly_frac`/`swr_disjoint` are DROPPED here because the assemblies are GIVEN via assemblies_ext.
SWR_CLOSED = dict(
    n_ca3=2000, ca3_density=0.05, encode_drive=3000.0, no_sync=False,
    coact_thresh=0.02, hebb_lr=4.0, lam_dep_wi=1.0, hebb_max=150.0, ca3_fb_inhib=30.0, k_thresh=15.0,
    recall_k_thresh=30.0, recall_drive=1200, recall_steps=150, bistable=True, nmda_recurrent=False,
    enable_ou=False, selective_inhib=True, structural_sep=1, plateau_self_regen=0.15, apical_kir_g=3.0,
    apical_gc=1.0, apical_gc_read=5.0, swr_schaffer_hi=80.0, swr_schaffer_lo=0.0,
)

# emergent-DG selection config (amplify: mossy detonator + coincidence recurrent + bistability keystone).
EMERG = dict(dg_ffi_weight=6.0, ca3_fb_inhib=20.0, mossy_weight=200.0, mossy_density=0.10, amplify=True, amp_ca3w=4.0)


def _cos(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return float(u @ v / (nu * nv)) if nu > 1e-9 and nv > 1e-9 else 0.0


def _check_index_space(seed, n_ca3=2000):
    be = _build_bridge(seed, n_ca3=n_ca3, **EMERG)
    ca3_e = list(be.region_manager.indices("ca3"))
    br = _build(seed, n_ca3=n_ca3, ca3w=6.0, ca3_density=SWR_CLOSED["ca3_density"], coincidence=True, two_comp=True,
                train=True, hebb_max=SWR_CLOSED["hebb_max"], ca3_fb_inhib=SWR_CLOSED["ca3_fb_inhib"],
                coact_thresh=0.02, hebb_lr=4.0, enable_ou=False, k_thresh=15.0, plateau_self_regen=0.15,
                apical_kir_g=3.0, apical_gc=1.0, apical_gc_read=5.0)
    ca3_r = list(br.region_manager.indices("ca3"))
    return (ca3_e == ca3_r), ca3_e, be


def _select_emergent(be, n_mem, seed, drive_pA=200.0, theta=0.3, use_dg_direct=True, stp_off=False):
    """Select n_mem emergent assemblies; return list of GLOBAL ca3 index arrays + diagnostics."""
    rm = be.region_manager
    ca3_idx = list(rm.indices("ca3"))
    ca3_arr = np.asarray(ca3_idx, dtype=np.int64)
    dg_arr = np.asarray(list(rm.indices("dg")), dtype=np.int64)
    lang = list(rm.indices("language_input"))
    if stp_off:
        be.core_config.enable_short_term_plasticity = False
    if use_dg_direct:
        drive_arr = dg_arr
        pats = [build_drive_pattern(len(dg_arr), 0.1, seed * 100 + m) for m in range(n_mem)]
    else:
        drive_arr = np.asarray(lang, dtype=np.int64)
        pats = [build_drive_pattern(len(lang), 0.1, seed * 100 + m) for m in range(n_mem)]
    sel = [_select(be, drive_arr, ca3_arr, dg_arr, p, sync=True, drive_pA=drive_pA, theta=theta) for p in pats]
    A_local = [sorted(s[0]) for s in sel]
    rates = [s[1] for s in sel]
    sizes = [len(a) for a in A_local]
    # separation on the thresholded binary assembly vectors
    bins = [(r >= theta).astype(float) for r in rates]
    pc = [_cos(bins[i], bins[j]) for i in range(n_mem) for j in range(i + 1, n_mem)]
    sep = float(np.mean(pc)) if pc else 0.0
    A_global = [np.asarray([ca3_idx[i] for i in a], dtype=np.int64) for a in A_local]
    return A_global, sizes, sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--ca1-topk", type=float, default=0.1)
    ap.add_argument("--drive-pA", type=float, default=200.0)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--sel-stp-off", action="store_true", help="disable STP during selection (mossy detonation lever)")
    ap.add_argument("--out", type=str, default="research/findings/raw/_gap5_emergent_into_swr.json")
    args = ap.parse_args()
    if not os.environ.get("SWR_PHASE2_NOSTP"):
        print("[WARN] SWR_PHASE2_NOSTP not set -> the SWR readout will NOT fire / anti-cheat will not collapse. "
              "Re-run with SWR_PHASE2_NOSTP=1.", flush=True)

    rows = []
    for s in args.seeds:
        same, _, be = _check_index_space(s, n_ca3=SWR_CLOSED["n_ca3"])
        print(f"[seed {s}] index-space ca3-globals identical(emergent vs run): {same}", flush=True)
        if not same:
            print(f"[seed {s}] ABORT: index spaces differ -> cross-bridge globals invalid.", flush=True)
            rows.append({"seed": s, "index_match": False, "status": "index-mismatch"})
            continue
        A_global, sizes, sep = _select_emergent(be, args.n_mem, s, drive_pA=args.drive_pA, theta=args.theta,
                                                 stp_off=args.sel_stp_off)
        usable = all(3 <= sz <= 60 for sz in sizes) and sep < 0.4
        print(f"[seed {s}] emergent selection: sizes={sizes} sep_cos={sep:.3f} usable={usable}", flush=True)
        if not usable:
            reason = ("silent (all-empty: mossy STP-capped)" if all(sz == 0 for sz in sizes)
                      else f"avalanche/too-dense (sizes {sizes}, sep {sep:.2f})" if any(sz > 60 for sz in sizes)
                      else f"not sparse+separated (sizes {sizes}, sep {sep:.2f})")
            print(f"[seed {s}] BLOCKED: emergent selection did not produce usable assemblies -> {reason}. "
                  f"NOT feeding garbage to SWR (verify-first).", flush=True)
            rows.append({"seed": s, "index_match": True, "sizes": sizes, "sep_cos": sep, "usable": False,
                         "status": f"selection-blocked:{reason}"})
            continue
        # (3) FEED: emergent globals -> SWR run() (learned Schaffer ON) + no-learn anti-cheat (learned Schaffer OFF)
        on = swr_run(s, read_ca1=True, swr_learn_schaffer=True, swr_ca1_topk=args.ca1_topk,
                     assemblies_ext=[a.tolist() for a in A_global], **SWR_CLOSED)
        off = swr_run(s, read_ca1=True, swr_learn_schaffer=False, swr_ca1_topk=args.ca1_topk,
                      assemblies_ext=[a.tolist() for a in A_global], **SWR_CLOSED)
        # shuffled-assembly control: same-size RANDOM (non-DG-selected) assemblies -> match should drop
        rng = np.random.default_rng(s * 999 + 7)
        ca3_all = list(be.region_manager.indices("ca3"))
        shuf = [np.asarray(sorted(rng.choice(ca3_all, len(a), replace=False)), dtype=np.int64) for a in A_global]
        shf = swr_run(s, read_ca1=True, swr_learn_schaffer=True, swr_ca1_topk=args.ca1_topk,
                      assemblies_ext=[a.tolist() for a in shuf], **SWR_CLOSED)
        r = {"seed": s, "index_match": True, "sizes": sizes, "sep_cos": sep, "usable": True,
             "held_cue": on.get("held_cue", 0.0),
             "on_match": on.get("ca1_match", 0.0), "on_cross": on.get("ca1_cross", 0.0),
             "off_match": off.get("ca1_match", 0.0), "off_cross": off.get("ca1_cross", 0.0),
             "shuf_match": shf.get("ca1_match", 0.0), "shuf_cross": shf.get("ca1_cross", 0.0), "status": "fed"}
        rows.append(r)
        print(f"[seed {s}] SWR-ON match={r['on_match']:.3f} cross={r['on_cross']:.3f} "
              f"(ratio {r['on_match']/(r['on_cross']+1e-9):.1f}x) | no-learn OFF match={r['off_match']:.3f} "
              f"cross={r['off_cross']:.3f} | shuffled match={r['shuf_match']:.3f}", flush=True)

    fed = [r for r in rows if r.get("status") == "fed"]
    print("=" * 90)
    if fed:
        m_on = float(np.mean([r["on_match"] for r in fed])); c_on = float(np.mean([r["on_cross"] for r in fed]))
        c_off = float(np.mean([r["off_cross"] for r in fed])); m_shuf = float(np.mean([r["shuf_match"] for r in fed]))
        n_go = sum(int(r["on_match"] >= 0.6 and r["on_cross"] <= 0.3 and r["on_match"] >= 3 * r["on_cross"]) for r in fed)
        anticheat = c_off >= 0.7
        print(f"[emergent->SWR] {len(fed)} fed | ca1_match {m_on:.3f} cross {c_on:.3f} ratio {m_on/(c_on+1e-9):.1f}x | "
              f"no-learn cross {c_off:.3f} | shuffled match {m_shuf:.3f} | per-seed-GO {n_go}/{len(fed)}")
        print(f"  VERDICT: {'GO' if (n_go >= max(1, int(0.83*len(fed))) and anticheat) else 'BOUNDARY'}")
    else:
        print("[emergent->SWR] NO seed produced usable emergent assemblies -> Phase-2 BLOCKED on the emergent selection "
              "(the committed selection does not fire a sparse+separated CA3 assembly). See per-seed status.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows, "swr_closed": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                            for k, v in SWR_CLOSED.items()}}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
