"""Structural-effect check for the ACh-staged-recurrence variant.

The variant returned NEGATIVE with zero change in pool firing on
recurrence install (0.0024 -> 0.0023). That is ambiguous: either the
installed recurrence had no supra-threshold seed to amplify (valid
NEGATIVE) or the installed recurrence is functionally INERT (would
make the variant verdict VOID, not NEGATIVE). The SPEAR arc's
adversarial review caught exactly this defect class.

This check resolves it empirically. Drive a SUBSET (30%) of a concept
pool's excitatory neurons with a strong supra-threshold current and
measure the WHOLE pool's firing rate, BEFORE and AFTER installing the
staged recurrence. If the recurrence transmits, the driven subset
recurrently excites the undriven majority and the whole-pool rate
rises. If before == after, the recurrence is inert.

Decision rule (fixed): after-install whole-pool rate >= 1.3x the
before-install rate (or absolute rise > 0.05) -> recurrence ACTIVE,
the variant NEGATIVE is valid. Otherwise -> recurrence INERT, the
variant verdict is VOID and the install must be fixed.

Reuse-by-import: build_variant_bridge + install_staged_recurrence from
the variant script. No protected module modified.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from importlib import util as _iu
_v_path = os.path.join(_HERE, "ach_staged_recurrence_variant.py")
_spec = _iu.spec_from_file_location("_achv", _v_path)
_achv = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_achv)
build_variant_bridge = _achv.build_variant_bridge
install_staged_recurrence = _achv.install_staged_recurrence

import research.runners.concept_pool_demo as cpd

SEED = 42
VARIANT_CACHE = "research/findings/raw/unified_per_regime/phase1_ca1variant/seed42.simstate.h5"
PROBE_POOL = "adjective_pool_BIG"
DRIVE_PA = 200.0
DRIVE_FRACTION = 0.30
STIM_STEPS = 100


def drive_subset_measure_pool(bridge, pool, drive_fraction, drive_pA, stim_steps,
                                rng):
    """Drive a random subset of the pool's excitatory neurons at
    drive_pA; measure the mean firing rate of the WHOLE pool."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    all_idx = list(rm.indices(pool))
    inh = set(rm.inhibitory_indices(pool))
    exc = np.array([i for i in all_idx if i not in inh], dtype=np.int64)
    n_drive = max(1, int(len(exc) * drive_fraction))
    driven = rng.choice(exc, size=n_drive, replace=False)
    driven_gpu = cp.asarray(driven, dtype=cp.int64)
    whole_gpu = cp.asarray(np.array(all_idx, dtype=np.int64))

    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    accum = 0.0
    for _ in range(stim_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[driven_gpu] = float(drive_pA)
        bridge._run_one_simulation_step()
        firing = bridge.cp_firing_states
        accum += float(cp.sum(firing[whole_gpu].astype(cp.float32)))

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    return accum / (stim_steps * len(all_idx)), n_drive, len(all_idx)


def main():
    print("=== ACh-staged-recurrence structural-effect check ===")
    print(f"probe pool={PROBE_POOL}; drive {DRIVE_FRACTION:.0%} of exc neurons "
          f"at {DRIVE_PA} pA")

    bridge = build_variant_bridge(SEED)
    bridge.load_checkpoint(VARIANT_CACHE)

    rng = np.random.default_rng(SEED)
    # Use a FIXED driven-subset RNG so before/after drive the same neurons.
    drive_rng_seed = 12345

    before, n_drive, n_pool = drive_subset_measure_pool(
        bridge, PROBE_POOL, DRIVE_FRACTION, DRIVE_PA, STIM_STEPS,
        np.random.default_rng(drive_rng_seed))
    print(f"  BEFORE install: whole-pool rate={before:.4f} "
          f"(drove {n_drive}/{n_pool} exc neurons)")

    concept_pools = (
        ["noun_pool_%s" % n for n in cpd.NOUN_NAMES]
        + ["verb_pool_%s" % v for v in cpd.VERB_NAMES]
        + ["adjective_pool_%s" % a for a in cpd.ADJECTIVE_NAMES]
    )
    n_edges, n_attempted = install_staged_recurrence(
        bridge, concept_pools, _achv.RECUR_DENSITY, _achv.RECUR_WEIGHT, rng)
    print(f"  installed {n_attempted} recurrent edges; "
          f"set_pathway_weights returned n_updated={n_edges}")

    after, _, _ = drive_subset_measure_pool(
        bridge, PROBE_POOL, DRIVE_FRACTION, DRIVE_PA, STIM_STEPS,
        np.random.default_rng(drive_rng_seed))
    print(f"  AFTER install:  whole-pool rate={after:.4f}")

    ratio = after / before if before > 1e-9 else float("inf")
    abs_rise = after - before
    active = (ratio >= 1.3) or (abs_rise > 0.05)
    print(f"\n=== STRUCTURAL-EFFECT VERDICT ===")
    print(f"  before={before:.4f}  after={after:.4f}  ratio={ratio:.2f}  "
          f"abs_rise={abs_rise:+.4f}")
    if active:
        print("  --> RECURRENCE ACTIVE: the installed recurrent excitation "
              "transmits (driven subset spreads activity to the undriven "
              "majority). The staged-recurrence variant NEGATIVE is VALID.")
        verdict = "RECURRENCE_ACTIVE"
    else:
        print("  --> RECURRENCE INERT: installing the recurrence did not "
              "change the pool's response to a supra-threshold drive. The "
              "staged-recurrence variant verdict is VOID -- the install did "
              "not produce a functional recurrent pathway. Must fix the "
              "install before any conclusion.")
        verdict = "RECURRENCE_INERT"

    out = {
        "seed": SEED, "probe_pool": PROBE_POOL, "drive_pA": DRIVE_PA,
        "drive_fraction": DRIVE_FRACTION, "n_recurrent_edges": n_attempted,
        "set_pathway_weights_returned": n_edges,
        "before_rate": before, "after_rate": after,
        "ratio": ratio, "abs_rise": abs_rise, "verdict": verdict,
    }
    with open("research/findings/raw/ach_recurrence_structural_check.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/ach_recurrence_structural_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
