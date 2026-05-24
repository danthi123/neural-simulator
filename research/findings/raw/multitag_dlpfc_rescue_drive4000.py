"""Final drive sweep point: 4000 pA at multi-seed (between 3000 and
5000 single-seed best; multi-seed at 3000 was 0.792 just below bar)."""
from __future__ import annotations
import json, os, sys, time
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
    _HERE, "multitag_dlpfc_rescue_drive4000.json")
RESCUE_DRIVE_PA = 4000.0
SEEDS = [42, 43, 44]


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== multitag drive=4000 pA multi-seed ===", flush=True)
    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]
    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
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
        print(f"  seed {seed}: FULL {r['n_full']}/{r['n_total']}"
              f" = {r['full_acc']:.3f}", flush=True)
    full_accs = [r['full_acc'] for r in seed_results]
    partial_accs = [r['partial_acc'] for r in seed_results]
    full_mean = float(np.mean(full_accs))
    partial_mean = float(np.mean(partial_accs))
    print(f"\nMulti-seed FULL mean: {full_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in full_accs)}]",
          flush=True)
    print(f"Multi-seed PARTIAL mean: {partial_mean:.3f}", flush=True)
    print(f"Wall-clock: {(time.time()-t0)/60:.2f} min", flush=True)

    if full_mean >= 0.80:
        verdict = "MULTITAG_RESCUE_VALIDATED_AT_4000PA"
        print(f"\n  VALIDATED at drive=4000 pA multi-seed: "
              f"{full_mean:.3f} >= 0.80", flush=True)
    else:
        verdict = "BOUND_PERSISTS_AT_4000PA"
        print(f"\n  Multi-seed {full_mean:.3f} < 0.80 at drive=4000;"
              f" the bound persists across drive sweep range",
              flush=True)

    out = {
        "drive_pA": RESCUE_DRIVE_PA, "seeds": SEEDS,
        "per_seed": seed_results,
        "multi_seed_full_mean": full_mean,
        "multi_seed_full_per_seed": full_accs,
        "multi_seed_partial_mean": partial_mean,
        "comparison_n102_baseline_1500pA": 0.708,
        "comparison_rescue_3000pA_multiseed": 0.792,
        "verdict": verdict,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
