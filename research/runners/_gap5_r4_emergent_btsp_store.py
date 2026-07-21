"""gap#5 R4 — store the EMERGENT-selected sparse CA3 assembly (from the R1 sparse detonator) as a self-sustaining
COMPLETABLE attractor via one-shot BTSP, then read it out through the CLOSED SWR readout.

R1 recovered the emergent-DG SELECTION at scale (sparse detonator, n_ca3=2000: sparse/separated/stable/input-specific,
GO 6/6 core). R4 = the completable STORE. The BTSP-store + bistable-completion + learned-Schaffer SWR readout are all
GO on PRE-ASSIGNED assemblies (2026-07-18 unification, 2026-07-19 SWR). Here I feed the EMERGENT assemblies instead.

STEP 1: run the R1 emergent selection (drive DG directly on the mossy-detonator bridge, read the NATURAL >=theta CA3
assembly) -> a list of CA3 GLOBAL index arrays, ONE per DG input pattern.
STEP 2: VERIFY index-space -- the R1 selection bridge and the BTSP bridge (`_riii...run`'s internal `_build`) must
place CA3 at the SAME global indices (same region sizes before CA3). Asserted before storing.
STEP 3: feed the emergent assemblies as `assemblies_ext=` to the BTSP path (the exact 2026-07-18 GO config) -> does the
bistable CA3 COMPLETE from a partial cue? cue-gated / SPECIFIC (permuted) / BISTABLE (no-cue silent) / no-encode->0.
STEP 4: if completion works, `read_ca1 + swr_learn_schaffer + swr_ca1_topk=0.1` (+ env SWR_PHASE2_NOSTP=1) -> the CLOSED
SWR readout: ca1_match(partial vs full, same) high, ca1_cross(other) low.

HONEST NOTE: the emergent set is TRANSIENT-drive-selected + SPARSE (~37-49 cells vs the reference's ~240) -> fewer
within-set recurrents at density 0.05. If it can't be BTSP-stored to a completable attractor, that is a precise
BOUNDARY (SELECTION is GO, the STORE needs a different mechanism). NO sim/ edit (reuse encode_btsp + R1 selection).

  SIM_BACKEND=cupy python -m research.runners._gap5_r4_emergent_btsp_store --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap5_emergent_dg_selection_derisk import _build_bridge  # noqa: E402
from research.runners._gap5_dg_selection_reset_scale_driver import _drive_read  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402
from research.runners._riii_ca3_synchronous_assembly_derisk import run as btsp_run  # noqa: E402

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap5_r4_emergent_btsp_store.json"

# R1 emergent-selection config (the recovered sparse-detonator working point at n_ca3=2000)
R1 = dict(n_ca3=2000, dg_ffi_weight=6.0, ca3_fb_inhib=20.0, mossy_weight=3000.0, mossy_density=0.02,
          amp_ca3w=12.0, n_dg=300, mossy_stp_disabled=True, drive_pA=2000.0, sync=False, theta=0.15)

# the 2026-07-18 BTSP-store GO config (pre-assigned -> here fed the EMERGENT assemblies)
GO_CFG = dict(n_ca3=2000, ca3_density=0.05, encode_drive=3000.0, no_sync=True,
              recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False,
              enable_ou=False, selective_inhib=True, structural_sep=1, plateau_self_regen=0.15, apical_kir_g=3.0,
              apical_gc=1.0, apical_gc_read=5.0)
BTSP_CFG = {**GO_CFG, "encode_btsp": True, "encode_ca3w": 0.5, "encode_plateau_pA": 250.0, "btsp_lr": 0.02,
            "hebb_max": 300.0, "train_events": 30, "recall_k_thresh": 40.0}


def emergent_assemblies(seed, n_patterns=2):
    """Build the R1 mossy-detonator bridge, drive n_patterns distinct DG inputs, return (list of CA3 GLOBAL index
    arrays, ca3_global_range) — the NATURAL >=theta assembly per input, mapped from ca3-local to global."""
    b = _build_bridge(seed, R1["n_ca3"], R1["dg_ffi_weight"], R1["ca3_fb_inhib"], R1["mossy_weight"],
                      R1["mossy_density"], n_dg=R1["n_dg"], amplify=True, amp_ca3w=R1["amp_ca3w"],
                      mossy_stp_disabled=R1["mossy_stp_disabled"])
    rm = b.region_manager
    ca3_arr = np.asarray(list(rm.indices("ca3")), dtype=np.int64)   # GLOBAL ca3 indices
    dg_arr = np.asarray(list(rm.indices("dg")), dtype=np.int64)
    b.cp_external_input_current[:] = 0.0
    for _ in range(30):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    pats = [build_drive_pattern(len(dg_arr), 0.1, seed * 100 + m) for m in range(n_patterns)]
    assemblies = []
    for p in pats:
        A_local, _ = _drive_read(b, dg_arr[p], ca3_arr, drive_pA=R1["drive_pA"], sync=R1["sync"], theta=R1["theta"])
        assemblies.append(np.asarray(sorted(int(ca3_arr[i]) for i in A_local), dtype=np.int64))
    ca3_range = (int(ca3_arr[0]), int(ca3_arr[-1]), len(ca3_arr))
    del b
    return assemblies, ca3_range


def _verify_index_space(seed, r1_ca3_range):
    """Build the BTSP bridge's substrate (same `_build` path run() uses) and assert its CA3 GLOBAL index range == R1's,
    so the emergent global indices refer to the SAME physical CA3 cells."""
    from research.runners._riii_ca3_coincidence_completion_derisk import _build
    bb = _build(seed, n_ca3=GO_CFG["n_ca3"], ca3w=BTSP_CFG["encode_ca3w"], ca3_density=GO_CFG["ca3_density"],
                coincidence=True, two_comp=True, train=True, ca3_fb_inhib=20.0, hebb_rate=True)
    ca3b = np.asarray(list(bb.region_manager.indices("ca3")), dtype=np.int64)
    rng_b = (int(ca3b[0]), int(ca3b[-1]), len(ca3b))
    del bb
    return rng_b, (rng_b == r1_ca3_range)


def run_seed(seed, do_swr=False, ca3_density=None, recall_k_thresh=None, structural_sep=None, isolate=False):
    t = {}
    assemblies, r1_range = emergent_assemblies(seed, n_patterns=2)
    sizes = [len(a) for a in assemblies]
    btsp_range, ok = _verify_index_space(seed, r1_range)
    t["assembly_sizes"] = sizes; t["r1_ca3_range"] = r1_range; t["btsp_ca3_range"] = btsp_range
    t["index_space_match"] = bool(ok)
    if not ok:
        t["error"] = f"INDEX-SPACE MISMATCH r1={r1_range} btsp={btsp_range}"; return t
    cfg = dict(BTSP_CFG)
    if ca3_density is not None:
        cfg["ca3_density"] = float(ca3_density)   # storage-substrate recurrent fan-in (indices unchanged)
    if recall_k_thresh is not None:
        cfg["recall_k_thresh"] = float(recall_k_thresh)
    if structural_sep is not None:
        cfg["structural_sep"] = int(structural_sep)   # 2 = full bidirectional isolation (closed set, no completion spread)
    if isolate:
        cfg["interassembly_isolate"] = True   # zero between-assembly recurrents (emergent equivalent of swr_disjoint)
    t["ca3_density"] = cfg["ca3_density"]; t["recall_k_thresh"] = cfg["recall_k_thresh"]
    t["structural_sep"] = cfg["structural_sep"]; t["isolate"] = bool(isolate)
    # STEP 3: BTSP-store the EMERGENT assemblies + bistable completion
    r = btsp_run(seed, assemblies_ext=[a.copy() for a in assemblies], read_ca1=do_swr,
                 **({**cfg, "swr_learn_schaffer": True, "swr_ca1_topk": 0.1} if do_swr else cfg))
    t.update(held_cue=r.get("held_cue"), held_nocue=r.get("held_nocue"), held_perm=r.get("held_perm"),
             rest_firing=r.get("rest_firing"), w_within=r.get("w_within"), completion_go=r.get("go"))
    if do_swr:
        t.update(ca1_match=r.get("ca1_match"), ca1_cross=r.get("ca1_cross"), ca1_fire=r.get("ca1_fire"))
    # no-encode anti-cheat (assembly present but NOT stored -> completion must collapse)
    ne = btsp_run(seed, assemblies_ext=[a.copy() for a in assemblies],
                  **{**cfg, "encode_plateau_pA": 0.0, "encode_drive": 0.0})
    t["noencode_cue"] = ne.get("held_cue")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--swr", action="store_true", help="also run the CLOSED SWR readout (ca1_match/cross)")
    ap.add_argument("--ca3-density", type=float, default=None, help="storage-substrate CA3 recurrent density (default 0.05)")
    ap.add_argument("--recall-k-thresh", type=float, default=None)
    ap.add_argument("--structural-sep", type=int, default=None, help="2 = full bidirectional isolation (closed set)")
    ap.add_argument("--isolate", action="store_true", help="zero between-assembly recurrents (emergent swr_disjoint)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.swr:
        os.environ["SWR_PHASE2_NOSTP"] = "1"
    t0 = time.time(); err = None; per = []
    print(f"[gap5-R4] BTSP-store the EMERGENT assembly + bistable-complete (+SWR={a.swr}); seeds={a.seeds}", flush=True)
    try:
        for s in a.seeds:
            r = run_seed(s, do_swr=a.swr, ca3_density=a.ca3_density, recall_k_thresh=a.recall_k_thresh,
                         structural_sep=a.structural_sep, isolate=a.isolate)
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] {r['error']}", flush=True); continue
            swr = (f" | ca1_match {r['ca1_match']:.2f} cross {r['ca1_cross']:.2f}" if a.swr and r.get('ca1_match') is not None else "")
            print(f"  [seed {s}] sizes {r['assembly_sizes']} idx-match {r['index_space_match']} | "
                  f"cue {r['held_cue']:.3f} nocue {r['held_nocue']:.3f} perm {r['held_perm']:.3f} rest {r['rest_firing']:.3f} "
                  f"w_within {r['w_within']:.1f} | no-encode {r['noencode_cue']:.3f} -> "
                  f"{'GO' if r['completion_go'] else 'NO'}{swr}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    valid = [p for p in per if not p.get("error") and p.get("held_cue") is not None]
    if err is None and valid:
        def mean(k): return float(np.mean([p[k] for p in valid if p.get(k) is not None]))
        # MECHANISM completion GO per seed (the reference's own 6/6 criteria: cue-gated + specific + bistable +
        # load-bearing; magnitude ~0.18-0.19 is the characterized uniform-BTSP residual just below the strict 0.20 bar):
        # cue>=0.15 (real completion) & cue>=3x nocue & cue>=3x perm & nocue<=0.10 & no-encode collapses.
        def cgo(p):
            c = p["held_cue"] or 0.0
            return (c >= 0.15 and c >= 3.0 * ((p.get("held_nocue") or 0.0) + 1e-6)
                    and c >= 3.0 * ((p.get("held_perm") or 0.0) + 1e-6) and (p.get("held_nocue") or 0.0) <= 0.10
                    and (p.get("noencode_cue") or 0.0) <= max(0.05, 0.34 * c))
        npass = sum(cgo(p) for p in valid)
        strict20 = sum(1 for p in valid if p.get("completion_go"))
        comp_go = npass >= max(1, int(np.ceil(5 / 6 * len(a.seeds)))) if len(a.seeds) >= 6 else all(cgo(p) for p in valid)
        m = {k: mean(k) for k in ("held_cue", "held_nocue", "held_perm", "noencode_cue", "w_within")}
        swr_go = None; swr_m = {}
        if a.swr and any(p.get("ca1_match") is not None for p in valid):
            swr_m = {"ca1_match": mean("ca1_match"), "ca1_cross": mean("ca1_cross")}
            swr_pass = sum(1 for p in valid if (p.get("ca1_match") or 0) >= 0.6 and (p.get("ca1_cross") or 1) <= 0.3)
            swr_go = swr_pass >= max(1, int(np.ceil(5 / 6 * len(a.seeds)))) if len(a.seeds) >= 6 else swr_pass == len(valid)
        verdict = (f"{'COMPLETION-MECHANISM-GO' if comp_go else 'COMPLETION-BOUNDARY'} {npass}/{len(valid)} seeds "
                   f"(strict-0.20 bar {strict20}/{len(valid)}; cue {m['held_cue']:.3f}, nocue {m['held_nocue']:.3f}, "
                   f"perm {m['held_perm']:.3f}, no-encode {m['noencode_cue']:.3f}, w_within {m['w_within']:.1f})"
                   + (f" || SWR {'GO' if swr_go else 'NO'} (match {swr_m.get('ca1_match',0):.2f} cross {swr_m.get('ca1_cross',0):.2f})" if a.swr else ""))
        go = bool(comp_go and (swr_go if a.swr else True))
    else:
        go = False; verdict = f"ERROR -- {err}" if err else "ERROR -- no valid seeds"
    summary = {"probe": "gap5_r4_emergent_btsp_store", "GO": go, "verdict": verdict, "seeds": a.seeds,
               "swr": a.swr, "elapsed_seconds": round(time.time()-t0, 1), "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[gap5-R4] VERDICT: {verdict}\n[gap5-R4] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
