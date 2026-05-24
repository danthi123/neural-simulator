"""Direction D: stim-drive sweep on multitag-on-DLPFC-extension.

Per pillar n=102 BOUNDARY: multitag on DLPFC-extension substrate
DEGRADES to 70.8% multi-seed (vs OPTION 3 / HIPPO-OPTION3 at 91.7%);
biology-translatable bound: dlpfc_wm NMDA bistability pulls cortical
drive 3.09x sparser; scale-sensitive readout is affected.

This probe tests whether INCREASING the stim drive (compensating for
the reduced cortical activity) can RESCUE multitag PASS on n=98.

Sweep stim_drive_pA in {1500 (baseline n=102), 3000, 5000} on seed
42 only (fastest single-seed comparison). If 3000 or 5000 PASSes
(FULL >= 0.80): the scale-sensitivity bound IS compensable; biology-
translatable refinement to n=102. If neither: deeper bound; the
n=98 substrate fundamentally can't support multitag at strong drive
either (perhaps because dlpfc_wm pulls drive irrespective of input
magnitude).

CHEAP probe (~10-15 min GPU); no protected/frozen/moat modified.
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

from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_stim, cosine_to_word,
    _ALL_CONCEPTS,
)
from research.findings.raw.mode_unification_with_hippo_dlpfc_probe import (
    _build_bridge_with_hippo_and_dlpfc,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "multitag_dlpfc_drive_sweep.json")

DRIVES_TO_TEST = [1500.0, 3000.0, 5000.0]
SEED = 42
PAIRS_STR = ("apple:big,dog:small,cat:hot,river:cold,"
             "big:hot,small:cold,apple:cat,dog:river")
N_LANG_INPUT = 2048
N_PER_POOL = 200
N_FS_PER_POOL = 24
N_WORDS_FOR_ORTHOGONAL = 16
ENCODING_STEPS = 500
TOP_K = 100
DRIVE_STEPS = 100
SPARSITY = 0.05
BALANCED_TEACHER_PA = 500.0
TOP_N = 2


def test_multitag_at_drive(bridge, drive_pA, valid_concepts):
    """Run multitag readout at given drive_pA; return per-cue results."""
    pairs = []
    for ps in PAIRS_STR.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            pairs.append((a, b))

    rm = bridge.region_manager
    region_filter = []
    for kind, names in [
            ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
            ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
            ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]:
        for n in names:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass

    # Encode pairs (encoding drive unchanged; only stim drive varies)
    encoded_tags = []
    for a, b in pairs:
        tag = f"{a}_{b}"
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=ENCODING_STEPS,
            drive_pA=200.0, sparsity=SPARSITY,
            n_lang_input=N_LANG_INPUT,
            n_words_for_orthogonal=N_WORDS_FOR_ORTHOGONAL,
            region_filter=region_filter, top_k=TOP_K,
            balanced_teacher_pA=BALANCED_TEACHER_PA,
            verbose=False,
        )
        encoded_tags.append(tag)

    cue_to_associates = {}
    for tag in encoded_tags:
        a, b = tag.split("_")
        cue_to_associates.setdefault(a, []).append(b)
        cue_to_associates.setdefault(b, []).append(a)

    n_full = 0; n_partial = 0
    per_cue = []
    test_cues = [c for c in cue_to_associates
                  if len(cue_to_associates[c]) >= 2]

    for cue in test_cues:
        matching_tags = [tag for tag in encoded_tags
                         if cue in tag.split("_")]
        agg_cos = {w: 0.0 for w in valid_concepts}
        for tag in matching_tags:
            pattern, n_lang_out = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=drive_pA,
                stim_steps=DRIVE_STEPS)
            for w in valid_concepts:
                if w == cue:
                    continue
                cos = cosine_to_word(
                    pattern, w, n_lang_out,
                    n_words_for_orthogonal=N_WORDS_FOR_ORTHOGONAL,
                    sparsity=SPARSITY)
                agg_cos[w] = agg_cos.get(w, 0.0) + cos

        ranked = sorted(((w, c) for w, c in agg_cos.items() if w != cue),
                         key=lambda x: x[1], reverse=True)
        topN = [w for w, c in ranked[:TOP_N]]
        expected = cue_to_associates[cue]
        in_top = [w for w in expected if w in topN]
        full = (len(in_top) == len(expected))
        partial = (len(in_top) >= 1)
        if full: n_full += 1
        if partial: n_partial += 1
        per_cue.append({
            "cue": cue, "expected": expected, "topN": topN,
            "full": full, "partial": partial,
        })

    n_total = len(test_cues)
    return {
        "drive_pA": drive_pA,
        "n_total": n_total,
        "n_full": n_full, "n_partial": n_partial,
        "full_acc": n_full / n_total,
        "partial_acc": n_partial / n_total,
        "per_cue": per_cue,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== multitag stim-drive sweep on DLPFC-extension (Direction D) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Drives to test: {DRIVES_TO_TEST} pA; seed={SEED} only "
          f"(fast comparison)", flush=True)
    print(f"  Tests whether higher stim drive RESCUES multitag PASS "
          f"on n=98 substrate (rescues the scale-sensitivity bound "
          f"from pillar n=102).", flush=True)

    cache_path = os.path.join(
        _HERE, "mode_unification_with_hippo_dlpfc_cache",
        f"bridge_full_seed{SEED}.simstate.h5")
    print(f"\n  Building DLPFC-extension substrate + loading cache "
          f"({cache_path})...", flush=True)
    bridge = _build_bridge_with_hippo_and_dlpfc(
        seed=SEED, enable_adjective=True, verbose=False)
    bridge.load_checkpoint(cache_path)
    # Re-freeze plasticity gates.
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

    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]

    results = []
    t0 = time.time()
    for drive in DRIVES_TO_TEST:
        print(f"\n--- testing drive={drive} pA ---", flush=True)
        r = test_multitag_at_drive(bridge, drive, valid_concepts)
        results.append(r)
        print(f"  drive={drive}: FULL {r['n_full']}/{r['n_total']} "
              f"= {r['full_acc']:.3f}; PARTIAL {r['n_partial']}/"
              f"{r['n_total']} = {r['partial_acc']:.3f}", flush=True)
        for pc in r['per_cue']:
            print(f"    cue={pc['cue']:8} expected={'+'.join(pc['expected']):14}"
                  f" top-{TOP_N}={','.join(pc['topN']):16} "
                  f"{'FULL' if pc['full'] else ('PARTIAL' if pc['partial'] else 'MISS')}",
                  flush=True)
    total_min = (time.time() - t0) / 60
    print(f"\nTotal wall-clock: {total_min:.2f} min", flush=True)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  drive_pA  FULL_acc  PARTIAL_acc", flush=True)
    for r in results:
        print(f"  {r['drive_pA']:.0f}      "
              f"{r['full_acc']:.3f}     {r['partial_acc']:.3f}",
              flush=True)

    print(f"\n=== INTERPRETATION ===", flush=True)
    base = results[0]
    print(f"  baseline (1500 pA): FULL {base['full_acc']:.3f}",
          flush=True)
    best_drive = max(results, key=lambda r: r['full_acc'])
    if best_drive['full_acc'] >= 0.80:
        verdict = "MULTITAG_RESCUE_BY_HIGHER_DRIVE"
        print(f"  At drive={best_drive['drive_pA']} pA: FULL "
              f"{best_drive['full_acc']:.3f} >= 0.80. The scale-"
              f"sensitivity bound from pillar n=102 IS COMPENSABLE "
              f"by stronger stim drive. dlpfc_wm reduces baseline "
              f"cortical activity, but stronger engram stim can "
              f"recover multitag readout.", flush=True)
    else:
        verdict = "BOUND_PERSISTS_ACROSS_DRIVE_VALUES"
        print(f"  Best drive {best_drive['drive_pA']} pA only reaches "
              f"FULL {best_drive['full_acc']:.3f} < 0.80. The "
              f"scale-sensitivity bound is NOT simply compensable "
              f"by stronger drive; deeper substrate-level "
              f"perturbation (dlpfc_wm changes cortex dynamics more "
              f"than just baseline activity level).", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "drives_tested": DRIVES_TO_TEST,
        "results": results,
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
