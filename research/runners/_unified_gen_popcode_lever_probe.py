"""Unified-agent generalization: the POPULATION-CODE lever vs the merged-bridge 100/101 co-residence compression.

THE DIAGNOSIS THIS TESTS (finding `2026-06-16-vision-to-concept-spiking-npercat-sweep.md`):
  The unified embodied agent's only Stage-3 misses are GENERALIZATION-at-chance at seeds 100/101 (merged gen H5 =
  0.25). The N_PER_CAT sweep + a zero-GPU cheap-first analysis localized this DECISIVELY:
    * IDENTICAL held-out split (merged uses the same seed*31+5 split as the standalone capstone),
    * IDENTICAL convergence training config (epochs/hebbian_max=20/scales/OU-off/STDP-off — `_train_merged_convergence`),
    * IDENTICAL read method (`_read_gen_spikes` calls the standalone's own `read_heldout_spikes` + category-mean),
    * OU is off in BOTH reads,
  yet STANDALONE 100/101 = 0.75 (moat intact) vs MERGED 100/101 = 0.25 — and seed 42 is byte-identical (0.75) in both.
  ⇒ the ONLY remaining difference is the merged gen_concept's PER-INDEX NEURON HETEROGENEITY (it sits at global
  indices ~8000+, drawing different Izhikevich a/b/c/d jitter than the standalone's base-2048), tipping the MARGINAL
  100/101 category read. The exemplar lever (more N_PER_CAT) is REJECTED — it lifts accuracy but BREAKS the no-confab
  moat at N=8 (a HARD STOP). The MOAT-SAFE lever is the POPULATION CODE: more gen_n_concept_per averages the
  category-mean over MORE heterogeneous neurons → lower read variance → the thin-but-real 100/101 margin wins again,
  WITHOUT broadening the category cores (so the moat contrast is preserved).

THE TEST: build the MERGED bridge (co_resident_generalization=True — gen at the high global indices, the failing
condition) at gen_n_concept_per in {100 (baseline, reproduces 0.25), 300 (the lever)}, for seeds {100,101 (the
failures), 42 (the robust control)}, and read the Stage-1 generalization check (`_gen_check`: H5 concept-cat spike
acc + the 0.6× no-confab MOAT) on the CLEAN build state (no episode — `_gen_check` runs on the clean state by design).

GATE:
  GO       : at gen_n_concept_per=300, seeds 100 AND 101 recover H5 >= 0.50 (the read averages out the heterogeneity),
             AND the no-confab MOAT ABSTAINS for EVERY (seed, n_concept_per) (HARD STOP — never loosened), AND seed
             42's H5 + moat are preserved.
  NEGATIVE : the lever does not recover 100/101 (the co-residence compression is a SYSTEMATIC bias, not read-variance)
             -> an honest, load-bearing finding routing to per-region heterogeneity control or accepting it as
             seed-variance. Report per-(seed, n_concept_per) honestly.

HARD STOP: ANY moat breach at ANY (seed, n_concept_per) fails outright; the gate is NEVER loosened.

Reuse-by-import ONLY (build_merged_nav_conv_bridge + navigate_unified_episode._gen_check). NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._unified_gen_popcode_lever_probe \
          --seeds 100,101,42 --nconcept 100,300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend                                              # noqa: E402
from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge  # noqa: E402
from research.runners.navigate_unified_episode import _gen_check                  # noqa: E402


def _run_cell(seed: int, n_concept_per: int):
    """Build the merged bridge (gen co-resident) at this gen_n_concept_per, read the clean-state generalization
    check. Returns the _gen_check dict + a couple of build facts."""
    xp, _ = get_backend()
    bridge, h = build_merged_nav_conv_bridge(
        seed=seed, co_resident_rf=True, co_resident_perception=True, enable_spiking_wta_readout=True,
        co_resident_generalization=True, gen_n_concept_per=int(n_concept_per))
    gen = h["gen"]
    r = _gen_check(bridge, h, seed, xp)
    r["gen_n_concept_per"] = int(n_concept_per)
    r["conc_base"] = int(gen.get("conc_base", -1))
    r["n_neurons"] = int(bridge.core_config.num_neurons)
    del bridge
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="100,101,42", help="100/101 = the merged failures; 42 = the robust control")
    p.add_argument("--nconcept", default="100,300", help="gen_n_concept_per values (100 baseline, 300 the lever)")
    p.add_argument("--out", default="research/findings/raw/_unified_gen_popcode_lever.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    seeds = [int(s) for s in a.seeds.split(",")]
    nconcepts = [int(n) for n in a.nconcept.split(",")]
    t0 = time.time()

    print(f"[gen pop-code lever] backend={os.environ['SIM_BACKEND']} seeds={seeds} gen_n_concept_per={nconcepts}\n"
          f"  Q: does more gen_n_concept_per (read-side population averaging) recover the MERGED 100/101 H5 (the "
          f"co-residence/heterogeneity compression) WITHOUT breaking the no-confab moat? (the exemplar lever is "
          f"REJECTED — it breaks the moat.)\n  GATE: 100/101 H5>=0.50 at the largest n_concept_per + MOAT abstains "
          f"EVERY cell (HARD STOP).", flush=True)

    results = {}     # results[seed][n] = cell
    moat_breaches = []
    for seed in seeds:
        results[seed] = {}
        for n in nconcepts:
            try:
                r = _run_cell(seed, n)
                h5 = float(r["h5_concept_cat_acc"]); moat = bool(r["moat_abstains"])
                results[seed][n] = r
                if not moat:
                    moat_breaches.append((seed, n, r["heldout_win_fire"], r["novel_win_fire"]))
                print(f"  >> seed={seed} n_concept_per={n}: H5={h5:.2f} margin {r['h5_margin']:+.3f} | "
                      f"held-out win-fire {r['heldout_win_fire']:.2f} vs novel {r['novel_win_fire']:.2f} "
                      f"(gate {r['gate_thresh']:.2f}) -> MOAT {'ABSTAIN' if moat else 'CONFAB(BREACH)'} | "
                      f"H6 {r['h6_hybrid_recall_acc']:.2f}", flush=True)
            except Exception as exc:
                results[seed][n] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                print(f"  !! seed={seed} n_concept_per={n}: ERROR {type(exc).__name__}: {exc}", flush=True)
            # incremental save (each cell is a ~10-15 min merged build; partial progress must survive)
            _pp = os.path.join(_REPO, a.out)
            os.makedirs(os.path.dirname(_pp), exist_ok=True)
            with open(_pp, "w") as fh:
                json.dump({"verdict": "IN_PROGRESS", "seeds": seeds, "nconcepts": nconcepts,
                           "results": {str(s): {str(k): v for k, v in results[s].items()} for s in results}},
                          fh, indent=2, default=str)

    # ---- GATE ----
    largest = max(nconcepts)
    def _h5(seed, n):
        c = results.get(seed, {}).get(n, {})
        return c.get("h5_concept_cat_acc") if "h5_concept_cat_acc" in c else None
    recover_100 = (_h5(100, largest) is not None and _h5(100, largest) >= 0.50) if 100 in seeds else True
    recover_101 = (_h5(101, largest) is not None and _h5(101, largest) >= 0.50) if 101 in seeds else True
    moat_intact = (len(moat_breaches) == 0)
    ctrl_42 = True
    if 42 in seeds and _h5(42, largest) is not None:
        ctrl_42 = _h5(42, largest) >= 0.50
    go = bool(recover_100 and recover_101 and moat_intact and ctrl_42)
    verdict = "GO" if go else "NEGATIVE"

    print(f"\n{'='*100}\n  SUMMARY (merged-bridge H5 per seed x gen_n_concept_per):", flush=True)
    print("  seed | " + " ".join(f"n={n:>4}" for n in nconcepts), flush=True)
    for seed in seeds:
        cells = []
        for n in nconcepts:
            h5 = _h5(seed, n)
            cells.append(f"{h5:.2f}" if h5 is not None else " ERR")
        print(f"  {seed:>4} | " + " ".join(f"{c:>5}" for c in cells), flush=True)
    print(f"\n  GATE:", flush=True)
    print(f"    seed 100 recovers H5>=0.50 @n={largest} : {recover_100}", flush=True)
    print(f"    seed 101 recovers H5>=0.50 @n={largest} : {recover_101}", flush=True)
    print(f"    seed 42 control preserved @n={largest}  : {ctrl_42}", flush=True)
    print(f"    MOAT intact (HARD STOP)                 : {moat_intact}"
          + ("" if moat_intact else f"  <-- BREACHES {moat_breaches}"), flush=True)
    print(f"  ==> {verdict}\n{'='*100}", flush=True)
    if not moat_intact:
        print("  *** HARD STOP: no-confab MOAT BREACHED — load-bearing finding, NOT a GO. Gate NOT loosened. ***",
              flush=True)
    if verdict == "GO":
        print(f"  GO — the POPULATION-CODE lever (more gen_n_concept_per) recovers the merged 100/101 generalization "
              f"WITHOUT breaking the moat: the co-residence/heterogeneity compression was read-variance, averaged out "
              f"by a larger concept population. NEXT: set gen_n_concept_per={largest} in build_compose_bridge + "
              f"re-validate Stage-3 6-seed toward clean 6/6.", flush=True)
    else:
        print(f"  NEGATIVE — the lever did not recover 100/101 (H5 100={_h5(100,largest)}, 101={_h5(101,largest)}) "
              f"or a control failed. The co-residence compression is likely a SYSTEMATIC heterogeneity bias, not "
              f"read-variance -> route to per-region heterogeneity control (homogenize the gen_concept Izhikevich "
              f"params) or accept it as documented seed-variance. Honest, load-bearing.", flush=True)

    payload = {"verdict": verdict, "seeds": seeds, "nconcepts": nconcepts, "largest": largest,
               "gate": {"recover_100": recover_100, "recover_101": recover_101, "ctrl_42": ctrl_42,
                        "moat_intact": moat_intact, "moat_breaches": moat_breaches},
               "results": {str(s): {str(k): v for k, v in results[s].items()} for s in results}}
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else 1)


if __name__ == "__main__":
    main()
