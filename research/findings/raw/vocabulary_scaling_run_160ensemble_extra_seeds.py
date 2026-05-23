"""160-concept ensemble vocab-scaling -- 2-additional-seeds extension.

Cheap pre-registered disambiguation step for the 160-ensemble BOUNDARY
result. The decisive run (seeds 42, 43, 44) had 4 of 5 bridges PASS
the multi-seed-mean criterion at every load; bridgeD_spatial uniquely
missed. The mandatory smell-test established the substrate side is
sound (per-bridge captured density uniform across bridges; recognition
perfect; symbol-input geometry identical across bridges -- so the
vocabulary-structure hypothesis is refuted) and seed 43 was anomalous
for both bridgeC (collapse to 0.523 at L=5) and bridgeD. This
extension adds seeds 45 and 46 across all 5 bridges to test whether
bridgeD's miss is robust at a larger sample or seed-43-anomaly that
averages out.

PRE-REGISTERED reading (fixed; never tuned):
- ROBUST_MISS: bridgeD continues to miss at the 5-seed sample (per-
  bridge multi-seed-mean still < 0.80 at some load on bridgeD). The
  per-category scaling limit at this tier is real; the failure mode
  is not seed-anomaly.
- ANOMALY_WASHES_OUT: bridgeD's mean rises above 0.80 at all loads
  when averaged across 5 seeds. The original miss was seed-43-anomaly
  that averages out at larger sample; the K=16 PASS recipe extends
  per-bridge to all 5 categories at the 160-concept tier (subject to
  a fresh dedicated adversarial review before any capability claim).

The reviewed 160-ensemble runner is byte-unchanged; this extension
script calls `run_one_bridge_seed` directly (the reviewed function)
for the 2 new seeds × 5 bridges = 10 new bridge-seed combinations
and combines the result with the 15 existing cells (loaded from the
decisive run's JSON). The frozen 0.80 bar is unchanged. The K=16
PASS recipe is the runner's recipe.

CPU/GPU: the runner builds + trains + captures on GPU per (bridge,
seed); estimated wall-clock about 35 min per new bridge-seed × 10 =
roughly 6 hours total. Kill-safe per-bridge per-seed cache (the
runner's existing cache pattern); a re-launch resumes from the next
uncached new (bridge, seed). Plain ASCII.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The reviewed 160-ensemble runner -- imported byte-unchanged.
from research.findings.raw.vocabulary_scaling_run_160ensemble import (
    BRIDGE_NAMES, run_one_bridge_seed,
)
# The frozen bar + the K=16 PASS recipe constants.
from research.findings.raw.vocabulary_scaling_run import BAR, LOADS

NEW_SEEDS = [45, 46]
DECISIVE_JSON = os.path.join(
    _HERE, "vocabulary_scaling_run_160ensemble_full.json")
OUT_JSON = os.path.join(
    _HERE, "vocabulary_scaling_run_160ensemble_5seeds.json")


def main():
    print("=== 160-ensemble extension: +2 seeds (45, 46) across 5 bridges ===",
          flush=True)
    print(f"frozen bar {BAR} (not moved); K=16 PASS recipe unchanged; "
          f"reuses reviewed runner's run_one_bridge_seed", flush=True)

    if not os.path.exists(DECISIVE_JSON):
        print(f"ERROR: decisive 3-seed JSON not found: {DECISIVE_JSON}")
        return 2

    with open(DECISIVE_JSON, "r", encoding="utf-8") as f:
        decisive = json.load(f)
    existing_cells = decisive["cell_results"]
    existing_seeds = sorted(set(c["seed"] for c in existing_cells))
    print(f"loaded {len(existing_cells)} existing cell results from the "
          f"decisive run (seeds {existing_seeds})", flush=True)

    # Run the 10 new bridge-seed combinations.
    new_cells = []
    t0 = time.time()
    for bridge_name in BRIDGE_NAMES:
        for seed in NEW_SEEDS:
            t_cell = time.time()
            cell = run_one_bridge_seed(bridge_name, seed, smoke=False)
            new_cells.append(cell)
            print(f"  [{bridge_name}/{seed}] cell complete in "
                  f"{time.time() - t_cell:.1f}s (total elapsed "
                  f"{time.time() - t0:.1f}s)", flush=True)

    all_cells = existing_cells + new_cells
    all_seeds = sorted(set(c["seed"] for c in all_cells))
    assert all_seeds == [42, 43, 44, 45, 46], (
        f"unexpected seed set: {all_seeds}")

    # --- Aggregate per (bridge, load) over all 5 seeds ---------------
    print(f"\n=== PER-BRIDGE 5-SEED AGGREGATE ===", flush=True)
    print("              " +
          "  ".join(f"L={load} (int mean over 5 seeds)" for load in LOADS),
          flush=True)
    per_bridge_agg = {}
    all_pass = True
    for bridge_name in BRIDGE_NAMES:
        per_bridge_agg[bridge_name] = {}
        row = [f"  {bridge_name:>18}:"]
        for load in LOADS:
            ints = [c["per_load"][str(load)]["integrated_accuracy"]
                    for c in all_cells
                    if c["bridge"] == bridge_name]
            m = float(np.mean(ints)) if ints else float("nan")
            per_bridge_agg[bridge_name][load] = {
                "mean_integrated": m,
                "per_seed_integrated": ints,
            }
            row.append(f"{m:.4f} {'>=' if m >= BAR else '<'}{BAR}")
            if m < BAR:
                all_pass = False
        print("  ".join(row), flush=True)

    # --- Pre-registered reading --------------------------------------
    print(f"\n=== READING ===", flush=True)
    bridge_d_means = [per_bridge_agg["bridgeD_spatial"][l]["mean_integrated"]
                      for l in LOADS]
    bridge_d_all_pass = all(m >= BAR for m in bridge_d_means)
    if all_pass:
        reading = "ANOMALY_WASHES_OUT_K16_EXTENDS_TO_ALL_5_BRIDGES"
        print(f"  At 5-seed sample, EVERY (bridge, load) cell clears the "
              f"frozen {BAR} bar. The original bridgeD miss was seed-43-"
              f"anomaly that averages out at larger sample. The K=16 PASS "
              f"recipe extends per-bridge to all 5 categories at the "
              f"160-concept tier. Subject to a fresh dedicated adversarial "
              f"review before any capability claim.", flush=True)
    elif bridge_d_all_pass:
        reading = "OTHER_BRIDGE_MISSES_AT_5_SEEDS"
        print(f"  bridgeD now clears at 5 seeds but some other bridge "
              f"misses -- per-bridge breakdown above. The 3-seed miss was "
              f"sample-size-dependent.", flush=True)
    else:
        reading = "BRIDGED_ROBUST_MISS"
        print(f"  bridgeD continues to miss at the 5-seed sample "
              f"(per-load means {[round(m,4) for m in bridge_d_means]}). "
              f"The per-category scaling limit at this tier is real; the "
              f"failure mode is not seed-anomaly.", flush=True)

    out = {
        "seeds": all_seeds, "bridges": BRIDGE_NAMES, "loads": LOADS,
        "bar": BAR, "n_seeds": len(all_seeds),
        "new_seeds_added": NEW_SEEDS,
        "cell_results": all_cells,
        "per_bridge_aggregate": {b: {str(l): v for l, v in d.items()}
                                 for b, d in per_bridge_agg.items()},
        "reading": reading,
        "verdict": ("ENSEMBLE_160CONCEPT_K16_5SEED_PASS" if all_pass
                    else "ENSEMBLE_160CONCEPT_K16_5SEED_BELOW_BAR"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
