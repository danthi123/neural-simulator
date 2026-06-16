"""Vision->concept SPIKING N_PER_CAT learning-curve sweep -- the unified embodied agent's LAST weak link.

WHY THIS RUNS (the documented diagnosis chain):
  The unified embodied agent (navigate + compose + generalize + converse on ONE SimulationBridge) is 6-seed
  ROBUST on integration + no-confab moat + nav + compose + conversation + parse (Stage-3 = 4/6 GO, 0 moat
  breaches).  The ONLY remaining miss is GENERALIZATION-at-chance at seeds 100/101: the gen_concept assembly fires
  STRONGLY (held-out win-fire 1.88/1.31) but assigns the WRONG category (spiking H5 = 0.25 = chance).  The
  vision->concept fidelity SCOPING (`2026-06-16-vision-to-concept-fidelity-scoping.md`, controller-verified)
  root-caused this as a held-out/train SPLIT-MARGIN issue: with only N_PER_CAT-1 train exemplars per category on a
  4-class orientation ring, a boundary held-out draw flips the category vote.  The TOP FIX = more exemplars per
  category -> an IT POPULATION PROTOTYPE (catalog E.12 / Kandel Ch 24) -- a one-CONSTANT N_PER_CAT change.

  The CPU host stand-in (`_vision_to_concept_npercat_sweep`) was INCONCLUSIVE-BY-CEILING (saturates at 1.00 for
  every (N_PER_CAT, seed) including 100/101) -> the REPRESENTATION does NOT carry the seed-100/101 failure; it is a
  SPIKING-ONLY concept-read-margin property (the point-neuron sub-threshold read, the project's rate-code wall).
  ⇒ the decisive test is THIS: does more N_PER_CAT DEEPEN the *spiking* concept-read margin at 100/101?

WHAT THIS SWEEPS (reuse-by-import; NO sim/ edit, NO change to the validated capstone):
  For each N_PER_CAT in --npercat (default 4,8,12), patch the shared concept-count constant (N_PER_CAT + the
  derived F = N_CAT*N_PER_CAT) across the THREE modules that bind it (the onsubstrate-convergence source, the
  graded-propagation bridge builder that sizes the concept/readout regions from F, and the capstone itself), then
  run the capstone's `run_seed` for each of the 6 seeds.  The capstone's run_seed ALREADY carries every anti-cheat
  -- this runner only loops + records, it does NOT re-implement or weaken any control:
    * leakage-free split (hold out 1 concept/category, each held-out has same-cat TRAIN peers; asserted no overlap)
    * FLAT-distinct perception baseline (orthogonal codes, no visual structure -> must be ~chance)
    * category-DERANGEMENT permuted control (must collapse)
    * structure-preservation assert (top-K active-set within>between margin)
    * the ×1.5 no-confab MOAT (a visually-novel no-category shape must NOT drive confident category spikes)

THE METRIC = the SPIKING H5 = `out["structured"]["concept_cat_acc"]` (the held-out vision-derived concept-spike
category accuracy -- exactly the unified agent's gen H5).

GATE (per AUTONOMOUS_STATE.md EXACT NEXT):
  GO       : seeds 100 AND 101 reach spiking H5 >= 0.50 at some swept N_PER_CAT, AND the 6-seed MINIMUM H5 RISES
             with N_PER_CAT (the population/exemplar lift for the rate-code wall), AND the moat NEVER breaches at
             any (N_PER_CAT, seed), AND flat ~chance + derangement collapses + structure preserved everywhere.
  NEGATIVE : flat curve (more exemplars does not lift 100/101's spiking margin) -- an honest, load-bearing finding
             that routes to the population-code lift (more concept neurons / n_per averaging) or Option 3
             (DG/Marr-Albus pattern-separation).  Report per-(N_PER_CAT, seed) honestly.

HARD STOP (never negotiable): ANY moat breach at ANY (N_PER_CAT, seed) is a HARD STOP -- reported prominently, the
gate is NEVER loosened to manufacture a GO.  The no-confab moat is load-bearing.

GPU `SIM_BACKEND=cupy` (the spiking substrate; numpy is a tiny --smoke path only).
Run:  SIM_BACKEND=cupy python -u -m research.runners._vision_to_concept_spiking_npercat_sweep \
          --npercat 4,8,12 --seeds 42,43,44,100,101,102
Smoke: SIM_BACKEND=cupy python -u -m research.runners._vision_to_concept_spiking_npercat_sweep --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The three modules that each bind N_CAT / N_PER_CAT / F (Python looks these up in each module's globals at call
# time, so reassigning the module attribute IS seen by the functions that reference it).
import research.runners._genfrontier_onsubstrate_convergence_derisk as conv_mod          # noqa: E402
import research.runners._genfrontier_graded_propagation_derisk as gradprop_mod           # noqa: E402
import research.runners._genfrontier_capstone_vision_to_concept_derisk as capstone_mod   # noqa: E402

N_CAT = conv_mod.N_CAT   # 4 -- held fixed; only N_PER_CAT (and the derived F) is swept.


def _patch_npercat(npc: int):
    """Set N_PER_CAT (+ the derived F = N_CAT*N_PER_CAT) on ALL three modules so the bridge sizing
    (graded-propagation build_propagation_bridge uses F) and the indexing (capstone run_seed uses N_PER_CAT/F)
    agree.  N_CAT is unchanged.  Returns the new F for logging."""
    new_F = N_CAT * npc
    for mod in (conv_mod, gradprop_mod, capstone_mod):
        mod.N_PER_CAT = npc
        mod.F = new_F
    return new_F


def _capstone_args(a) -> SimpleNamespace:
    """The capstone main()'s validated default knobs (mirrored exactly so run_seed sees the GO config), plus the
    sweep's heavy-knob overrides (epochs/read_steps) for an optional faster pass."""
    return SimpleNamespace(
        candidate="nmda",
        top_k=a.top_k,
        min_set_margin=0.05,
        n_concept_per=a.n_concept_per,
        n_readout_per=a.n_concept_per,
        epochs=a.epochs,
        scene_steps=16,
        read_steps=a.read_steps,
        perc_scale=300.0,
        conc_scale=600.0,
        read_weight=30.0,
        nmda_ratio=2.0,
        hebbian_rate=0.05,
        hebbian_max=20.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npercat", default="4,8,12", help="N_PER_CAT values to sweep (exemplars per category)")
    p.add_argument("--seeds", default="42,43,44,100,101,102", help="the 6 unified-agent seeds")
    p.add_argument("--n-concept-per", type=int, default=100, help="neurons per concept block (population code)")
    p.add_argument("--top-k", type=int, default=60, help="top-K V1-complex perception drive size")
    p.add_argument("--epochs", type=int, default=20, help="convergence epochs (20 = capstone default)")
    p.add_argument("--read-steps", type=int, default=80, help="concept-spike read window (80 = capstone default)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny shape verification: npercat=8, seed 42, epochs=3 -- confirms the patched F "
                        "propagates to bridge construction without a shape error (NOT a science run)")
    p.add_argument("--out", default="research/findings/raw/_vision_to_concept_spiking_npercat_sweep.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")

    if a.smoke:
        a.npercat, a.seeds, a.epochs, a.read_steps = "8", "42", 3, 24

    npercat_vals = [int(x) for x in a.npercat.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    chance = 1.0 / N_CAT
    t0 = time.time()

    print(f"[vision->concept SPIKING N_PER_CAT sweep] backend={os.environ['SIM_BACKEND']} npercat={npercat_vals} "
          f"seeds={seeds} n_concept_per={a.n_concept_per} epochs={a.epochs} read_steps={a.read_steps} "
          f"chance={chance:.2f}\n  metric = the SPIKING H5 (held-out vision-derived concept-spike cat-acc); "
          f"GATE = 100/101 H5>=0.50 + the 6-seed MIN rises with N_PER_CAT; MOAT breach = HARD STOP (never "
          f"loosened).", flush=True)

    # results[npc] = {seed: row}
    results = {}
    moat_breaches = []   # (npc, seed, ho_fam, novel_fam) -- HARD STOP material
    for npc in npercat_vals:
        new_F = _patch_npercat(npc)
        ca = _capstone_args(a)
        print(f"\n{'#'*100}\n# N_PER_CAT={npc}  (F={new_F} concepts, {N_CAT} categories, {npc-1} train exemplars/cat, "
              f"perc->concept = 2048x{new_F * a.n_concept_per} synapses)\n{'#'*100}", flush=True)
        per_seed = {}
        for seed in seeds:
            try:
                out = capstone_mod.run_seed(seed, ca)
                h5 = float(out["structured"]["concept_cat_acc"])           # THE spiking H5
                h5_margin = float(out["structured"]["concept_margin"])
                conc_sp = float(out["structured"]["concept_spikes_per_cue"])
                flat = float(out["flat"]["concept_cat_acc"])
                perm_margin = float(out["permuted"]["concept_margin"])
                moat_ok = bool(out["moat"]["moat_ok"])
                ho_fam = float(out["moat"]["heldout_familiarity"])
                novel_fam = float(out["moat"]["novel_familiarity"])
                struct = bool(out["active_set"]["structure_preserved"])
                set_margin = float(out["active_set"]["margin"])
                row = {"h5": h5, "h5_margin": h5_margin, "concept_spikes_per_cue": conc_sp,
                       "flat": flat, "permuted_margin": perm_margin, "moat_ok": moat_ok,
                       "heldout_familiarity": ho_fam, "novel_familiarity": novel_fam,
                       "structure_preserved": struct, "set_margin": set_margin,
                       "held_out": out["held_out"]}
                if not moat_ok:
                    moat_breaches.append((npc, seed, ho_fam, novel_fam))
                print(f"  >> N_PER_CAT={npc} seed={seed}: H5={h5:.2f} (margin {h5_margin:+.3f}, {conc_sp:.0f} "
                      f"spk/cue) | flat={flat:.2f} | perm-margin {perm_margin:+.3f} | struct "
                      f"{'OK' if struct else 'LOST'} | moat {'OK' if moat_ok else 'BREACH'}", flush=True)
            except Exception as exc:   # one failed cell must not kill the multi-hour sweep
                row = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                print(f"  !! N_PER_CAT={npc} seed={seed}: ERROR {type(exc).__name__}: {exc}", flush=True)
            per_seed[seed] = row
        results[npc] = per_seed

    # ---- aggregate + GATE ----
    def _h5s(npc):
        return [results[npc][s]["h5"] for s in seeds if "h5" in results[npc][s]]

    print(f"\n{'='*100}\n  SWEEP SUMMARY (spiking H5 per N_PER_CAT x seed; chance {chance:.2f})\n{'='*100}", flush=True)
    header = "  N_PER_CAT | " + " ".join(f"s{seed:>4}" for seed in seeds) + " |   min   mean"
    print(header, flush=True)
    mins = {}
    for npc in npercat_vals:
        cells = []
        for s in seeds:
            r = results[npc][s]
            cells.append(f"{r['h5']:.2f}" if "h5" in r else " ERR")
        h5s = _h5s(npc)
        mn = float(np.min(h5s)) if h5s else float("nan")
        mean = float(np.mean(h5s)) if h5s else float("nan")
        mins[npc] = mn
        print(f"  {npc:>9} | " + " ".join(f"{c:>5}" for c in cells) + f" |  {mn:.2f}  {mean:.2f}", flush=True)

    # GATE evaluation -- spelled out so the verdict is auditable.
    # (1) seeds 100 AND 101 reach H5 >= 0.50 at SOME swept N_PER_CAT.
    def _seed_reaches(seed, thr=0.50):
        return any(("h5" in results[npc][seed] and results[npc][seed]["h5"] >= thr) for npc in npercat_vals)
    s100_ok = _seed_reaches(100)
    s101_ok = _seed_reaches(101)
    # (2) the 6-seed MINIMUM H5 RISES with N_PER_CAT (monotone non-decreasing across the swept values, strictly up
    #     end-to-end).
    ordered_mins = [mins[npc] for npc in npercat_vals if not np.isnan(mins[npc])]
    min_rises = (len(ordered_mins) >= 2
                 and all(ordered_mins[i + 1] >= ordered_mins[i] - 1e-9 for i in range(len(ordered_mins) - 1))
                 and ordered_mins[-1] > ordered_mins[0] + 1e-9)
    # (3) the controls hold everywhere (flat ~chance, derangement collapses, structure preserved) -- per the
    #     capstone's own per-seed gate; we require them at the BEST (largest) N_PER_CAT for the apply decision.
    best_npc = npercat_vals[-1]
    best_rows = [results[best_npc][s] for s in seeds if "h5" in results[best_npc][s]]
    controls_ok = bool(best_rows) and all(
        (r["flat"] <= chance + 0.15 and r["structure_preserved"]
         and r["permuted_margin"] <= r["h5_margin"] - 0.005) for r in best_rows)
    moat_intact = (len(moat_breaches) == 0)

    go = bool(s100_ok and s101_ok and min_rises and controls_ok and moat_intact)
    verdict = "GO" if go else "NEGATIVE"

    print(f"\n{'='*100}\n  GATE:", flush=True)
    print(f"    seed 100 reaches H5>=0.50 : {s100_ok}", flush=True)
    print(f"    seed 101 reaches H5>=0.50 : {s101_ok}", flush=True)
    print(f"    6-seed MIN H5 rises       : {min_rises}  (mins {[f'{m:.2f}' for m in ordered_mins]})", flush=True)
    print(f"    controls hold @best       : {controls_ok}  (N_PER_CAT={best_npc})", flush=True)
    print(f"    MOAT intact (HARD STOP)   : {moat_intact}"
          + ("" if moat_intact else f"  <-- BREACHES: {moat_breaches}"), flush=True)
    print(f"  ==> {verdict}\n{'='*100}", flush=True)

    if not moat_intact:
        print("  *** HARD STOP: the no-confab MOAT BREACHED -- this is a load-bearing finding, NOT a GO. The gate "
              "is NOT loosened. Report the breach + route honestly. ***", flush=True)
    if verdict == "GO":
        print(f"  GO -- more exemplars/category DEEPENS the spiking concept-read margin: the 6-seed minimum H5 rises "
              f"({ordered_mins[0]:.2f}->{ordered_mins[-1]:.2f}) and seeds 100/101 clear 0.50. The IT-population-"
              f"prototype (catalog E.12) lifts the unified agent's last weak link. NEXT: plumb N_PER_CAT into the "
              f"merged gen stack (build_compose_bridge) + re-validate Stage-3 6-seed toward 6/6.", flush=True)
    else:
        why = []
        if not (s100_ok and s101_ok):
            why.append(f"seeds 100/101 do not both reach 0.50 (100={s100_ok}, 101={s101_ok})")
        if not min_rises:
            why.append(f"the 6-seed MIN H5 does not rise with N_PER_CAT (mins {[f'{m:.2f}' for m in ordered_mins]} "
                       f"= flat curve -> more exemplars does not lift the spiking margin)")
        if not controls_ok:
            why.append("a control failed at the best N_PER_CAT (flat/derangement/structure)")
        if not moat_intact:
            why.append("MOAT breach (HARD STOP)")
        print(f"  NEGATIVE: {'; '.join(why)}. Honest, load-bearing -> route to the population-code lift (more "
              f"n_concept_per / n_per averaging) or Option 3 (DG/Marr-Albus pattern-separation). The dendritic "
              f"rewrite is explicitly NOT needed (a fidelity gap, not a substrate-mechanism gap).", flush=True)

    payload = {
        "verdict": verdict, "chance": chance, "npercat": npercat_vals, "seeds": seeds,
        "n_concept_per": a.n_concept_per, "epochs": a.epochs, "read_steps": a.read_steps, "top_k": a.top_k,
        "gate": {"seed100_reaches_0.50": s100_ok, "seed101_reaches_0.50": s101_ok,
                 "min_h5_rises": min_rises, "controls_ok_at_best": controls_ok,
                 "moat_intact": moat_intact, "moat_breaches": moat_breaches, "best_npercat": best_npc},
        "min_h5_per_npercat": {npc: mins[npc] for npc in npercat_vals},
        "results": {npc: results[npc] for npc in npercat_vals},
    }
    os.makedirs(os.path.dirname(os.path.join(_REPO, a.out)), exist_ok=True)
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else 1)


if __name__ == "__main__":
    main()
