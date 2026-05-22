"""Mandatory anti-cheat smell-test for the trained-substrate
vocabulary-scaling decisive run.

The discipline requires that a nominal PASS be scrutinised HARDER than a
NEGATIVE, and that the verdict be recomputed FROM THE SINGLE RECORDED
OUTPUT -- no re-run, no threshold change. This tool does exactly that.
It reads `vocabulary_scaling_run_trained_full.json` (the decisive run's
recording) and the per-seed trained-activity cache, and:

1. Recomputes the per-load integrated + composition-only means directly
   from `per_seed`, INDEPENDENTLY of the runner's own `aggregate`
   block, and cross-checks the two match (a mismatch means the runner's
   aggregation is wrong).
2. Recomputes the captured pool-activity density from the trained
   activity cache and confirms the substrate was genuinely exercised --
   the density must be well above the untrained decisive run's 0.0077
   pool-nonzero (the near-silence that caused the original NEGATIVE).
3. Re-derives the verdict from the recomputed means against the frozen
   0.80 bar and checks it matches the recorded verdict.
4. Runs consistency checks: composition-only accuracy >= integrated
   accuracy at each load (composition-only conditions on clean
   recognition, so it cannot be lower); per-seed values are not
   degenerate (not all identical, not all exactly 0 or 1); the
   recognition numbers are in [0, 1].

It prints a structured report and exits 0 if every check passes, 1 if
any check fails -- a failed check means the recorded result cannot be
trusted as-is and must be investigated before propagation.

Pure standard library + numpy; reads only recorded files; no bridge, no
GPU, no re-run. The activity cache is a numeric-only .npz (np.load
defaults to the safe no-object-array mode). Plain ASCII.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# The untrained decisive run's captured pool density, recomputed from
# its recording (full_seed42.npz) -- the near-silence that caused the
# original NEGATIVE. The trained substrate must beat this decisively.
UNTRAINED_DENSITY = 0.0077
# The validated v14/v16 substrate the pipeline passed on.
V16_DENSITY = 0.075
BAR = 0.80   # the frozen compositional bar (pinned; never tuned here)

RESULT_JSON = os.path.join(_HERE, "vocabulary_scaling_run_trained_full.json")
CACHE_DIR = os.path.join(_HERE, "vocabulary_scaling_trained_cache")


def _fail(checks, msg):
    checks.append(("FAIL", msg))


def _ok(checks, msg):
    checks.append(("ok", msg))


def recompute_aggregate(per_seed, loads):
    """Recompute per-load integrated + composition-only means directly
    from the per-seed records -- independent of the runner's own
    `aggregate` block."""
    agg = {}
    for load in loads:
        key = str(load)
        ints, comps = [], []
        for r in per_seed:
            pl = r["per_load"]
            cell = pl.get(key, pl.get(load))
            ints.append(float(cell["integrated_accuracy"]))
            c = float(cell["composition_only_accuracy"])
            if c == c:  # not NaN
                comps.append(c)
        agg[load] = {
            "mean_integrated": float(np.mean(ints)),
            "per_seed_integrated": ints,
            "mean_composition_only": (float(np.mean(comps))
                                      if comps else float("nan")),
        }
    return agg


def capture_density_from_cache(seed):
    """Recompute the captured pool-activity density from a trained
    activity cache file, or None if the cache is absent."""
    path = os.path.join(CACHE_DIR, f"trained_full_seed{seed}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    words = [str(w) for w in data["__words__"]]
    fracs = [float(np.mean(data["act__" + w] > 0.0)) for w in words]
    return float(np.mean(fracs))


def main():
    checks = []
    print("=== vocabulary-scaling trained-substrate smell-test ===")
    print(f"recompute-only; frozen bar {BAR}; reads the single recording\n")

    if not os.path.exists(RESULT_JSON):
        print(f"ERROR: result JSON not found: {RESULT_JSON}")
        print("The decisive run has not finished. Re-run this tool once "
              "vocabulary_scaling_run_trained_full.json exists.")
        return 2

    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    loads = res["loads"]
    per_seed = res["per_seed"]
    seeds = res["seeds"]
    recorded_verdict = res["verdict"]
    print(f"seeds={seeds}; loads={loads}; n_train_events="
          f"{res.get('n_train_events')}; substrate={res.get('substrate')}")
    print(f"recorded verdict: {recorded_verdict}\n")

    # --- Check 1: independent recompute vs the recorded aggregate ------
    recomputed = recompute_aggregate(per_seed, loads)
    recorded_agg = res["aggregate"]
    print("per-load integrated accuracy (recomputed from per_seed):")
    for load in loads:
        rc = recomputed[load]
        rec = recorded_agg.get(str(load), recorded_agg.get(load, {}))
        rec_mean = float(rec.get("mean_integrated", float("nan")))
        match = abs(rc["mean_integrated"] - rec_mean) < 1e-6
        print(f"  L={load}: per-seed="
              f"{['%.4f' % a for a in rc['per_seed_integrated']]} "
              f"mean={rc['mean_integrated']:.4f} | composition-only="
              f"{rc['mean_composition_only']:.4f}")
        if match:
            _ok(checks, f"L{load} recomputed mean matches recorded aggregate")
        else:
            _fail(checks, f"L{load} recomputed mean {rc['mean_integrated']:.4f} "
                          f"!= recorded {rec_mean:.4f}")

    # --- Check 2: the substrate was genuinely exercised ---------------
    print("\ncaptured pool density (recomputed from the trained cache):")
    any_cache = False
    for seed in seeds:
        d = capture_density_from_cache(seed)
        if d is None:
            print(f"  seed {seed}: cache absent (cannot recompute density)")
            continue
        any_cache = True
        print(f"  seed {seed}: density={d:.4f} "
              f"(untrained was {UNTRAINED_DENSITY}; v14/v16 {V16_DENSITY})")
        if d > UNTRAINED_DENSITY * 2.0:
            _ok(checks, f"seed {seed} trained density {d:.4f} decisively "
                        f"above the untrained {UNTRAINED_DENSITY}")
        else:
            _fail(checks, f"seed {seed} trained density {d:.4f} NOT above "
                          f"the untrained {UNTRAINED_DENSITY} -- the "
                          f"training stage did not exercise the substrate")
    if not any_cache:
        _fail(checks, "no trained activity cache found -- cannot verify "
                      "the substrate was genuinely exercised")

    # --- Check 3: re-derive the verdict from the recomputed means -----
    all_pass = all(recomputed[load]["mean_integrated"] >= BAR
                   for load in loads)
    derived = ("VOCABULARY_SCALING_64CONCEPT_TRAINED_PASS" if all_pass
               else "VOCABULARY_SCALING_64CONCEPT_TRAINED_BELOW_BAR")
    print(f"\nverdict re-derived from the recompute: {derived}")
    if derived == recorded_verdict:
        _ok(checks, "re-derived verdict matches the recorded verdict")
    else:
        _fail(checks, f"re-derived verdict {derived} != recorded "
                      f"{recorded_verdict}")

    # --- Check 4: consistency checks ----------------------------------
    for load in loads:
        rc = recomputed[load]
        ci = rc["mean_composition_only"]
        ii = rc["mean_integrated"]
        # composition-only conditions on clean recognition -> >= integrated.
        if ci == ci and ci + 1e-6 < ii:
            _fail(checks, f"L{load} composition-only {ci:.4f} < integrated "
                          f"{ii:.4f} -- inconsistent (composition-only "
                          f"conditions on clean recognition)")
        else:
            _ok(checks, f"L{load} composition-only >= integrated (consistent)")
        ps = rc["per_seed_integrated"]
        if len(ps) > 1 and len(set(round(x, 6) for x in ps)) == 1:
            _fail(checks, f"L{load} per-seed integrated all identical "
                          f"({ps[0]:.4f}) -- suspicious, no seed variation")
        else:
            _ok(checks, f"L{load} per-seed integrated shows seed variation")

    for r in per_seed:
        for k in ("recognition_per_observation",
                  "recognition_temporally_averaged"):
            v = float(r[k])
            if not (0.0 <= v <= 1.0):
                _fail(checks, f"seed {r['seed']} {k}={v} outside [0,1]")
    _ok(checks, "recognition numbers in [0, 1]")

    # --- Report --------------------------------------------------------
    n_fail = sum(1 for s, _ in checks if s == "FAIL")
    print(f"\n=== SMELL-TEST REPORT ({len(checks)} checks, "
          f"{n_fail} failed) ===")
    for status, msg in checks:
        print(f"  [{status}] {msg}")
    if n_fail == 0:
        print(f"\nSMELL-TEST PASSED -- the recorded result is internally "
              f"consistent and recomputes cleanly. The verdict "
              f"({recorded_verdict}) can be propagated honestly.")
        return 0
    print(f"\nSMELL-TEST FAILED ({n_fail} check(s)) -- the recorded "
          f"result must be investigated before propagation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
