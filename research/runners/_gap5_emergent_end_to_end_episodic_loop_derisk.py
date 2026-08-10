"""gap#5 END-TO-END EMERGENT EPISODIC LOOP — compose the three individually-GO pieces into ONE loop on ONE spiking
substrate: (a) emergent-DG SELECTION picks a sparse pattern-separated CA3 assembly from a novel DG volley; (b) BTSP
one-shot plateau-gated plasticity FORMS that assembly's within-CA3 slow-NMDA reverberatory attractor in its OWN
temporally-isolated encode episode; (c) a PARTIAL cue of that assembly COMPLETES it via the somatic slow-NMDA
reverberatory + FS-basket readout. This is an INTEGRATION build — it REPLACES the PRE-ASSIGNED assemblies tonight's
formation runner used with EMERGENTLY-SELECTED ones. No piece is re-derived: it composes the committed runners.

WHY THIS IS GENUINELY UN-COMPOSED (STEP 1 checked): tonight's BTSP-forms-slow-NMDA GO (cee2ff124) formed the attractor
on PRE-ASSIGNED (random-permutation) assemblies; its own "Honest scope" names emergent SELECTION as the open front-end.
The one prior end-to-end chain (2026-07-21 R4) read completion through the DENDRITIC/bistable path built on the
2026-07-18 config whose "learned CLOSED" claim was later RETRACTED (self-sustaining + Wang confound). So "emergent
selection -> tonight's slow-NMDA reverberatory formation + completion" is the untested composition this runner closes
or bounds.

THE SEAM (where confounds live): the two pieces were de-risked at DIFFERENT scales. Emergent-DG SELECTION is 6/6-core GO
ONLY at n_ca3=2000 (the sparse detonator; the n_ca3=400 GO does NOT reproduce -- 2026-07-21 record correction). The
slow-NMDA formation/readout was validated at n_ca3=400 with 72-cell pre-assigned assemblies. Composing forces a common
scale (n_ca3=2000) and hands the readout SPARSE emergent assemblies (~20-58 cells) instead of 72. A ~40-cell assembly
at density 0.12 has ~half the within-assembly fan-in of the 72-cell reference, so the reverberatory attractor may not
form strongly at the reference density. Fan-in is a substrate connectivity parameter (density) that leaves CA3 indices
unchanged, so this runner SWEEPS ca3_density to find (or bound) the operating point -- per THE LAW, not a stop.

PIPELINE per seed:
  1. emergent_assemblies(seed): build the R1 recovered-at-scale sparse-detonator bridge (n_ca3=2000, d0.02/w3000,
     acw12, drv2000, theta0.15, mossy_stp_disabled), drive n_patterns distinct DG inputs, read the NATURAL >=theta CA3
     assembly per input -> a list of CA3 GLOBAL-index arrays. The membership is DG-SELECTED, NOT a hand-set mask.
  2. index-space verify: the readout/formation bridge (tonight's _build_bridge at n_ca3=2000) places CA3 at the SAME
     global indices (same region sizes) -> the emergent indices refer to the same physical CA3 cells. Asserted.
  3. tonight's run_seed(assemblies_ext=<emergent>) runs the FULL committed instrument on the emergent membership:
     hand-install cross-check, BTSP isolated-episode formation, no-plateau lesion, no-encoding, recurrence-zero,
     permuted cue, silent-rest nocue, cross_dw/nonmem_dw, OU off/on. ONLY the membership differs from tonight's GO.

ANTI-CHEATS (the mandatory suite):
  #1 EMERGENT membership: the assemblies come from the DG volley, NOT a hand-set/pre-assigned set. Teeth: (a) sizes are
     the emergent ~20-58 (NOT the readout's 0.18*N pre-assigned size); (b) mossy-LESION (mossy_weight=0) -> assemblies
     collapse to ~empty (the DG->CA3 detonation is load-bearing = the membership is DG-derived); (c) low Jaccard vs the
     random-permutation set the readout would have used.
  #2 BTSP formation genuine: btsp_noplateau lesion -> no attractor (inherited from tonight's run_seed, 0/6).
  #3 completion cue-specific: permuted cue ~0, silent-rest nocue<=0.10, no-encoding -> 0 (inherited).
  #4 temporally-separated encode: cross_dw ~ 0 across the emergently-selected assemblies (inherited; isolated episodes).
  #5 plasticity FROZEN at recall, OU controllable (inherited).

GO (6-seed 42/43/44/100/101/102): an EMERGENTLY-SELECTED + BTSP-FORMED assembly completes from a partial cue,
cue-specifically -- held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10, at some
(density,wmax) working point. A PARTIAL (n/6) or a quantified integration-seam NEGATIVE (which piece's output doesn't
match the next piece's required input, and the fix) is a first-class honest deliverable. SIM_BACKEND=cupy.
  Run: SIM_BACKEND=cupy python -m research.runners._gap5_emergent_end_to_end_episodic_loop_derisk \
         --seeds 42 43 44 100 101 102 --densities 0.12 0.35 0.5 --both-ou \
         --out research/findings/raw/_gap5_e2e/e2e_6seed.json
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
from research.runners._gap5_emergent_dg_selection_derisk import _build_bridge as _sel_build_bridge  # noqa: E402
from research.runners._gap5_dg_selection_reset_scale_driver import _drive_read  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (  # noqa: E402
    run_seed as btsp_run_seed, _build_bridge as _readout_build_bridge)

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap5_e2e" / "e2e.json"

# R1 recovered-at-scale sparse-detonator emergent-selection config (the 2026-07-21 GO working point, n_ca3=2000).
R1 = dict(n_ca3=2000, dg_ffi_weight=6.0, ca3_fb_inhib=20.0, mossy_weight=3000.0, mossy_density=0.02,
          amp_ca3w=12.0, n_dg=300, mossy_stp_disabled=True, drive_pA=2000.0, sync=False, theta=0.15)

# readout/formation constants matched to tonight's committed slow-NMDA GO (only n_ca3 lifts 400 -> 2000 for the seam).
READ = dict(ca3_fb_inhib=60.0, nmda_tau=100.0, nmda_ratio=1.0, drive_pA=300.0, warm_steps=200, read_steps=200,
            assembly_frac=0.18, cue_frac=0.5, btsp_lr=0.05, encode_drive=700.0, encode_plateau_pA=250.0,
            train_events=40, element="nmda_slow")


def emergent_assemblies(seed, n_patterns=3, mossy_weight=None):
    """Mirror of _gap5_r4_emergent_btsp_store.emergent_assemblies with a mossy_weight knob (for the mossy-LESION
    anti-cheat). Build the R1 sparse-detonator bridge, drive n_patterns distinct DG inputs, return (list of CA3
    GLOBAL-index arrays = the NATURAL >=theta assembly per input, ca3 global range). The membership is DG-SELECTED
    (mossy detonator), NOT a hand-set mask."""
    mw = R1["mossy_weight"] if mossy_weight is None else float(mossy_weight)
    b = _sel_build_bridge(seed, R1["n_ca3"], R1["dg_ffi_weight"], R1["ca3_fb_inhib"], mw,
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


def _readout_ca3_range(seed, n_ca3, ca3_density):
    """Build tonight's readout/formation bridge and return its CA3 global index range (for the index-space verify)."""
    b = _readout_build_bridge(seed, n_ca3=n_ca3, ca3_density=ca3_density, ca3_fb_inhib=READ["ca3_fb_inhib"],
                              nmda_tau=READ["nmda_tau"], nmda_ratio=READ["nmda_ratio"], enable_ou=False,
                              element=READ["element"])
    ca3 = np.asarray(list(b.region_manager.indices("ca3")), dtype=np.int64)
    rng = (int(ca3[0]), int(ca3[-1]), len(ca3))
    del b
    return rng


def _jacc(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return len(a & b) / max(1, len(a | b))


def run_one_seed(seed, *, n_patterns, densities, wmax_grid, ou_modes, check_lesion=True, verbose=True):
    t = {"seed": seed}
    # ---- STEP 1: EMERGENT SELECTION ---------------------------------------------------------------------------------
    assemblies, r1_range = emergent_assemblies(seed, n_patterns=n_patterns)
    sizes = [len(a) for a in assemblies]
    t["assembly_sizes"] = sizes
    t["r1_ca3_range"] = r1_range
    if min(sizes) == 0:
        t["error"] = f"EMERGENT SELECTION produced an EMPTY assembly (sizes={sizes}) -- selection seam"; return t

    # ---- anti-cheat #1: the membership is DG-selected, not a hand-set/pre-assigned mask -----------------------------
    n_ca3 = r1_range[2]
    # (a) sizes are the emergent ~20-58, NOT the readout's pre-assigned 0.18*N mask
    preassigned_size = max(6, int(READ["assembly_frac"] * n_ca3))
    t["preassigned_size"] = preassigned_size
    t["emergent_not_preassigned_size"] = bool(max(sizes) < 0.5 * preassigned_size)
    # (b) low Jaccard vs the random-permutation set the readout would have used at this seed
    rng = np.random.default_rng(seed)
    perm_idx = rng.permutation(n_ca3)
    lo = r1_range[0]
    default_asm = [set(int(lo + perm_idx[a * preassigned_size:(a + 1) * preassigned_size][k])
                       for k in range(preassigned_size)) for a in range(n_patterns)]
    t["jaccard_vs_preassigned"] = [round(_jacc(assemblies[i], default_asm[i]), 4) for i in range(n_patterns)]
    # (c) mossy-LESION -> assemblies collapse (DG->CA3 detonation load-bearing = membership is DG-derived)
    if check_lesion:
        les, _ = emergent_assemblies(seed, n_patterns=n_patterns, mossy_weight=0.0)
        t["lesion_sizes"] = [len(a) for a in les]
        t["mossy_lesion_collapses"] = bool(sum(len(a) for a in les) <= max(1, 0.2 * sum(sizes)))
    emergent_ok = (t["emergent_not_preassigned_size"]
                   and all(j <= 0.34 for j in t["jaccard_vs_preassigned"])
                   and (t.get("mossy_lesion_collapses", True)))
    t["anticheat1_emergent_membership"] = bool(emergent_ok)

    # ---- STEP 2+3: BTSP FORMATION + slow-NMDA COMPLETION on the emergent membership (tonight's instrument) ----------
    rows = []
    for density in densities:
        # index-space verify (same region sizes -> match; density leaves CA3 indices unchanged)
        rr_range = _readout_ca3_range(seed, n_ca3, density)
        if rr_range != r1_range:
            t.setdefault("index_mismatch", []).append({"density": density, "readout_range": rr_range})
            continue
        for ou in ou_modes:
            rr = btsp_run_seed(seed, n_ca3=n_ca3, ca3_density=density, assembly_frac=READ["assembly_frac"],
                               cue_frac=READ["cue_frac"], ca3_fb_inhib=READ["ca3_fb_inhib"], nmda_tau=READ["nmda_tau"],
                               drive_pA=READ["drive_pA"], warm_steps=READ["warm_steps"], read_steps=READ["read_steps"],
                               enable_ou=ou, element=READ["element"], btsp_w_max_grid=wmax_grid, btsp_lr=READ["btsp_lr"],
                               encode_drive=READ["encode_drive"], encode_plateau_pA=READ["encode_plateau_pA"],
                               train_events=READ["train_events"], handinstall_W=wmax_grid,
                               assemblies_ext=assemblies, verbose=verbose)
            for r in rr:
                r["density"] = density
            rows.append(rr)
    t["index_space_match"] = bool(any(True for _ in rows)) and ("index_mismatch" not in t)
    flat = [r for rr in rows for r in rr]
    t["rows"] = flat

    # ---- verdict per seed: a BTSP working point that completes cue-specifically (+ genuine + lesion has teeth) -------
    def go_row(r):
        return bool(r.get("arm") == "btsp" and (r.get("held_cue") or 0) >= 0.20
                    and (r.get("held_cue") or 0) >= 3 * ((r.get("held_perm") or 0) + 1e-6)
                    and (r.get("held_cue") or 0) >= 3 * ((r.get("held_nocue") or 0) + 1e-6)
                    and (r.get("held_nocue") or 0) <= 0.10 and bool(r.get("genuine_formation")))
    go_rows_off = [r for r in flat if r.get("arm") == "btsp" and r.get("enable_ou") is False and go_row(r)]
    go_rows_on = [r for r in flat if r.get("arm") == "btsp" and r.get("enable_ou") is True and go_row(r)]
    t["seed_go_ou_off"] = bool(go_rows_off)
    t["seed_go_ou_on"] = bool(go_rows_on)
    # best (density,wmax) working point by held_cue for the record
    btsp_rows = [r for r in flat if r.get("arm") == "btsp"]
    if btsp_rows:
        best = max(btsp_rows, key=lambda r: ((r.get("held_cue") or 0), -abs((r.get("held_perm") or 0))))
        t["best"] = {k: best.get(k) for k in ("density", "btsp_w_max", "enable_ou", "held_cue", "held_perm",
                                              "held_nocue", "w_within", "cross_dw", "genuine_formation",
                                              "recurrence_zero_held_cue", "no_encoding_held_cue")}
    # no-plateau teeth (max over the collected no-plateau arms; should stay ~0)
    npl = [r.get("held_cue") or 0 for r in flat if r.get("arm") == "btsp_noplateau"]
    t["noplateau_max_cue"] = float(max(npl)) if npl else None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-patterns", type=int, default=3, help="number of emergently-selected co-stored assemblies")
    ap.add_argument("--densities", type=float, nargs="+", default=[0.12, 0.35, 0.5],
                    help="CA3 recurrent density sweep for the formation fan-in (indices unchanged)")
    ap.add_argument("--wmax", type=float, nargs="+", default=[2500.0, 5000.0, 9000.0])
    ap.add_argument("--ou", action="store_true")
    ap.add_argument("--both-ou", action="store_true")
    ap.add_argument("--no-lesion", action="store_true", help="skip the mossy-lesion anti-cheat (faster smoke)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    ou_modes = [False, True] if a.both_ou else [bool(a.ou)]
    t0 = time.time(); err = None; per = []
    print(f"[gap5-E2E] emergent-DG SELECTION -> BTSP FORMATION -> slow-NMDA COMPLETION | seeds={a.seeds} "
          f"n_patterns={a.n_patterns} densities={a.densities} wmax={a.wmax} ou={ou_modes}", flush=True)
    try:
        for s in a.seeds:
            r = run_one_seed(s, n_patterns=a.n_patterns, densities=a.densities, wmax_grid=a.wmax, ou_modes=ou_modes,
                             check_lesion=(not a.no_lesion), verbose=True)
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] ERROR {r['error']}", flush=True); continue
            b = r.get("best", {})
            print(f"  [seed {s}] sizes {r['assembly_sizes']} emergent-membership-OK={r['anticheat1_emergent_membership']} "
                  f"(jacc {r['jaccard_vs_preassigned']} lesion {r.get('lesion_sizes')}) || BEST d{b.get('density')} "
                  f"wmax{b.get('btsp_w_max')} ou{int(bool(b.get('enable_ou')))} cue={b.get('held_cue')} "
                  f"perm={b.get('held_perm')} nocue={b.get('held_nocue')} w_in={b.get('w_within')} "
                  f"crossdW={b.get('cross_dw')} genuine={b.get('genuine_formation')} | GO off={r['seed_go_ou_off']} "
                  f"on={r['seed_go_ou_on']} | noplateau_max={r.get('noplateau_max_cue')} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    valid = [p for p in per if not p.get("error")]
    n_off = sum(1 for p in valid if p.get("seed_go_ou_off"))
    n_on = sum(1 for p in valid if p.get("seed_go_ou_on"))
    n_emergent = sum(1 for p in valid if p.get("anticheat1_emergent_membership"))
    if err is None and valid:
        go = bool(n_off >= max(1, int(np.ceil(5 / 6 * len(a.seeds)))) and n_emergent == len(valid))
        verdict = (f"{'END-TO-END-GO' if go else 'INTEGRATION-SEAM' if not go else 'PARTIAL'} "
                   f"emergent-membership {n_emergent}/{len(valid)} | BTSP-completion GO off {n_off}/{len(valid)}, "
                   f"on {n_on}/{len(valid)} (seed-level: an EMERGENTLY-SELECTED + BTSP-FORMED assembly completes "
                   f"cue-specifically from a partial cue)")
    else:
        go = False; verdict = f"ERROR -- {err}" if err else "ERROR -- no valid seeds"
    summary = {"probe": "gap5_emergent_end_to_end_episodic_loop", "GO": go, "verdict": verdict, "seeds": a.seeds,
               "n_go_ou_off": n_off, "n_go_ou_on": n_on, "n_emergent_membership": n_emergent,
               "densities": a.densities, "wmax": a.wmax, "ou_modes": ou_modes,
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[gap5-E2E] VERDICT: {verdict}\n[gap5-E2E] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
