"""Multi-seed verification of the multitag drive-rescue on DLPFC.

Per Direction D single-seed finding: at drive=3000 pA (vs n=102's
baseline 1500), multitag on DLPFC-extension PASSes 8/8 FULL on
seed 42. This probe verifies multi-seed (seeds 42, 43, 44) at the
rescue drive value.

If multi-seed-mean FULL >= 0.80 at 3000 pA: MULTITAG_RESCUE_VALIDATED
(scale-sensitivity bound from n=102 is multi-seed compensable; warrants
new VALIDATED pillar n=103).
If below 0.80: characterisation refinement; single-seed rescue was
seed-specific.

Reuses the validated multitag mechanism + the dlpfc-extension's
substrate builder byte-unchanged. ~10-15 min GPU.
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

from research.findings.raw.multitag_dlpfc_drive_sweep import (
    test_multitag_at_drive, N_WORDS_FOR_ORTHOGONAL,
)
from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import _ALL_CONCEPTS
from research.findings.raw.mode_unification_with_hippo_dlpfc_probe import (
    _build_bridge_with_hippo_and_dlpfc,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(
    _HERE, "multitag_dlpfc_rescue_multiseed.json")

RESCUE_DRIVE_PA = 3000.0
SEEDS = [42, 43, 44]


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== multitag drive-rescue multi-seed verification ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Rescue drive: {RESCUE_DRIVE_PA} pA (vs n=102 baseline "
          f"1500 pA which gave multi-seed 0.708)", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)

    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]

    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        cache_path = os.path.join(
            _HERE, "mode_unification_with_hippo_dlpfc_cache",
            f"bridge_full_seed{seed}.simstate.h5")
        bridge = _build_bridge_with_hippo_and_dlpfc(
            seed=seed, enable_adjective=True, verbose=False)
        bridge.load_checkpoint(cache_path)
        for g in ("language_input_to_motor",
                  "language_input_to_noun_pool",
                  "language_input_to_verb_pool",
                  "language_input_to_adjective_pool",
                  "motor_to_language_output",
                  "noun_pool_to_language_output",
                  "verb_pool_to_language_output",
                  "adjective_pool_to_language_output",
                  "lang_to_dlpfc_wm"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass

        r = test_multitag_at_drive(bridge, RESCUE_DRIVE_PA, valid_concepts)
        r["seed"] = seed
        seed_results.append(r)
        print(f"  seed {seed} drive={RESCUE_DRIVE_PA}: FULL "
              f"{r['n_full']}/{r['n_total']} = {r['full_acc']:.3f}",
              flush=True)
    total_min = (time.time() - t0) / 60

    full_accs = [r['full_acc'] for r in seed_results]
    full_mean = float(np.mean(full_accs))
    partial_accs = [r['partial_acc'] for r in seed_results]
    partial_mean = float(np.mean(partial_accs))

    print(f"\n=== MULTI-SEED RESULT (drive={RESCUE_DRIVE_PA} pA) ===",
          flush=True)
    print(f"  FULL mean = {full_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in full_accs)}]",
          flush=True)
    print(f"  PARTIAL mean = {partial_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in partial_accs)}]",
          flush=True)
    print(f"  Wall-clock: {total_min:.2f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if full_mean >= 0.80:
        verdict = "MULTITAG_DRIVE_RESCUE_VALIDATED"
        print(f"  Multi-seed FULL mean {full_mean:.3f} >= 0.80 -- the"
              f" scale-sensitivity bound from n=102 IS COMPENSABLE "
              f"AT MULTI-SEED via doubling the stim drive. Compared "
              f"to n=102 (0.708) the same substrate at the rescue "
              f"drive gives {full_mean:.3f} (+{full_mean - 0.708:.3f})."
              f" Pending fresh adversarial review for VALIDATED "
              f"pillar n=103.", flush=True)
    else:
        verdict = "MULTITAG_DRIVE_RESCUE_SEED_SPECIFIC"
        print(f"  Multi-seed FULL mean {full_mean:.3f} < 0.80 -- "
              f"single-seed rescue (seed 42 at 1.000) was seed-"
              f"specific; multi-seed verification doesn't confirm. "
              f"Characterisation extension to n=102; deeper analysis "
              f"would be needed.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "rescue_drive_pA": RESCUE_DRIVE_PA,
        "seeds": SEEDS, "per_seed": seed_results,
        "multi_seed_full_mean": full_mean,
        "multi_seed_full_per_seed": full_accs,
        "multi_seed_partial_mean": partial_mean,
        "multi_seed_partial_per_seed": partial_accs,
        "n102_baseline_drive_pA": 1500.0,
        "n102_baseline_full_mean": 0.708,
        "rescue_improvement_vs_n102": full_mean - 0.708,
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
