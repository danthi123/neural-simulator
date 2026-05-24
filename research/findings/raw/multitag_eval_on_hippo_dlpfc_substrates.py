"""Multitag pair-retrieval test on the HIPPO-OPTION3 and DLPFC-
extension substrates (pillars n=97 + n=98).

Per the n=100 multitag REPLICATION finding on OPTION 3: confirms the
validated 2026-05-14 multitag mechanism (encoded via lang_input +
balanced_teacher; region_filter=concept pools; readout via stim_tag +
lang_output cosine) PASSes on the OPTION 3 substrate at 91.7% multi-
seed. This probe extends the validation to the two substrates with
hippo + dlpfc additions:
- HIPPO-OPTION3 (n=97): + hippocampus EC/DG/CA3/CA1 + SWR pathways
- DLPFC-extension (n=98): + hippocampus + dlpfc_wm NMDA bistable

The OPTION 3 cache loads cleanly into the validated multitag runner
because multitag_eval uses build_concept_bridge from concept_pool_demo
(matching architecture). The HIPPO/DLPFC caches were built via a
different builder (_build_bridge_with_hippo_and_dlpfc) so their caches
CANNOT be loaded into a build_concept_bridge build. This probe uses
the dlpfc-extension's builder + loads its cache + then runs the
validated multitag mechanism on the augmented substrate.

PRE-REGISTERED reading (matches the original validated bar):
- MULTITAG_PASS_ON_EXTENSIONS: multi-seed-mean FULL pass >= 0.80 on
  BOTH hippo and dlpfc substrates. The substrate extensions preserve
  the validated retrieval primitive.
- MULTITAG_DEGRADED_ON_EXTENSIONS: below bar on either; biology-
  translatable characterisation of which extension breaks multitag.

Reuses every primitive byte-unchanged: encode_concept_pair, stim
mechanism, readout from compose_concept_engram (validated 2026-05-14);
substrate builders from mode_unification_with_hippo_dlpfc_probe and
mode_unification_with_hippo_probe (validated n=97/n=98).

No protected/frozen/moat module modified; no autograd; no-confab moat
must stay 7/7 green.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_stim, cosine_to_word,
    _ALL_CONCEPTS,
)
# Substrate builders (validated n=97/n=98).
from research.findings.raw.mode_unification_with_hippo_probe import (
    _build_bridge_with_hippo,
)
from research.findings.raw.mode_unification_with_hippo_dlpfc_probe import (
    _build_bridge_with_hippo_and_dlpfc,
)
from sim.backend import get_backend, is_gpu_backend


SUBSTRATES = {
    "HIPPO_OPTION3_n97": {
        "builder": _build_bridge_with_hippo,
        "cache_dir": os.path.join(
            _HERE, "mode_unification_with_hippo_cache"),
    },
    "DLPFC_extension_n98": {
        "builder": _build_bridge_with_hippo_and_dlpfc,
        "cache_dir": os.path.join(
            _HERE, "mode_unification_with_hippo_dlpfc_cache"),
    },
}

SEEDS = [42, 43, 44]
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

OUT_JSON = os.path.join(
    _HERE, "multitag_eval_on_hippo_dlpfc_substrates.json")


def run_one_substrate_one_seed(substrate_name, seed):
    cfg = SUBSTRATES[substrate_name]
    builder = cfg["builder"]
    cache_dir = cfg["cache_dir"]
    cache_path = os.path.join(cache_dir, f"bridge_full_seed{seed}.simstate.h5")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"substrate cache missing: {cache_path}")

    print(f"\n--- {substrate_name} / seed={seed} ---", flush=True)
    # Build the substrate.
    bridge = builder(seed=seed, enable_adjective=True, verbose=False)
    bridge.load_checkpoint(cache_path)
    # Re-freeze plasticity gates so the multitag encoding doesn't
    # drift weights.
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

    # Parse pairs.
    pairs = []
    for ps in PAIRS_STR.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            pairs.append((a, b))

    # Build region_filter for concept pools (the validated multitag
    # mechanism).
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

    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]

    # Encode all pairs (validated mechanism).
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
    print(f"  encoded {len(encoded_tags)} pairs", flush=True)

    # For each cue word in >= 2 tags, multitag-aggregate cosines + check.
    cue_to_associates = {}
    for tag in encoded_tags:
        a, b = tag.split("_")
        cue_to_associates.setdefault(a, []).append(b)
        cue_to_associates.setdefault(b, []).append(a)

    n_full = 0
    n_partial = 0
    per_cue = []
    test_cues = [c for c in cue_to_associates if len(cue_to_associates[c]) >= 2]

    for cue in test_cues:
        # Find all tags containing this cue.
        matching_tags = [tag for tag in encoded_tags
                         if cue in tag.split("_")]
        # Aggregate cosines.
        agg_cos = {w: 0.0 for w in valid_concepts}
        for tag in matching_tags:
            pattern = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=1500.0,
                stim_steps=DRIVE_STEPS, readout_steps=50)
            for w in valid_concepts:
                if w == cue:
                    continue
                cos = cosine_to_word(
                    pattern, w, _WORD_TO_IDX[w],
                    N_WORDS_FOR_ORTHOGONAL, len(pattern),
                    sparsity=SPARSITY)
                agg_cos[w] = agg_cos.get(w, 0.0) + cos

        # Rank by aggregated cosine; pick top-N (excluding cue).
        ranked = sorted(((w, c) for w, c in agg_cos.items() if w != cue),
                         key=lambda x: x[1], reverse=True)
        topN = [w for w, c in ranked[:TOP_N]]
        expected = cue_to_associates[cue]

        in_top = [w for w in expected if w in topN]
        full = (len(in_top) == len(expected))
        partial = (len(in_top) >= 1)
        if full:
            n_full += 1
        if partial:
            n_partial += 1

        per_cue.append({
            "cue": cue,
            "expected": expected,
            "topN": topN,
            "full": full,
            "partial": partial,
        })
        print(f"  cue={cue:8} expected={'+'.join(expected):14} "
              f"top-{TOP_N}={','.join(topN):16} "
              f"{'FULL' if full else ('PARTIAL' if partial else 'MISS')}",
              flush=True)

    n_total = len(test_cues)
    full_acc = n_full / n_total
    partial_acc = n_partial / n_total
    print(f"  [{substrate_name}/{seed}] FULL: {n_full}/{n_total}"
          f" = {full_acc:.3f}; PARTIAL: {n_partial}/{n_total}"
          f" = {partial_acc:.3f}", flush=True)
    return {
        "substrate": substrate_name, "seed": seed,
        "n_total_cues": n_total,
        "n_full": n_full, "n_partial": n_partial,
        "full_accuracy": full_acc,
        "partial_accuracy": partial_acc,
        "per_cue": per_cue,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== multitag eval on HIPPO-OPTION3 + DLPFC-extension "
          "substrates ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  pairs={PAIRS_STR}", flush=True)
    print(f"  recipe: encoding_steps={ENCODING_STEPS}; balanced_"
          f"teacher_pA={BALANCED_TEACHER_PA}; top_k={TOP_K}; top_n="
          f"{TOP_N}", flush=True)

    all_results = []
    for substrate_name in SUBSTRATES:
        for seed in SEEDS:
            r = run_one_substrate_one_seed(substrate_name, seed)
            all_results.append(r)

    # Multi-seed aggregate per substrate.
    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    agg = {}
    for substrate in SUBSTRATES:
        rs = [r for r in all_results if r["substrate"] == substrate]
        full_accs = [r["full_accuracy"] for r in rs]
        partial_accs = [r["partial_accuracy"] for r in rs]
        full_mean = float(np.mean(full_accs))
        partial_mean = float(np.mean(partial_accs))
        agg[substrate] = {
            "full_mean": full_mean,
            "full_per_seed": full_accs,
            "partial_mean": partial_mean,
            "partial_per_seed": partial_accs,
        }
        verdict = "PASS" if full_mean >= 0.80 else "BELOW BAR"
        print(f"  {substrate}: FULL mean={full_mean:.3f} per-seed="
              f"{full_accs} ({verdict})", flush=True)
        print(f"  {substrate}: PARTIAL mean={partial_mean:.3f} "
              f"per-seed={partial_accs}", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "seeds": SEEDS, "pairs": PAIRS_STR,
        "encoding_steps": ENCODING_STEPS,
        "balanced_teacher_pA": BALANCED_TEACHER_PA,
        "top_k": TOP_K, "top_n": TOP_N,
        "per_substrate_per_seed": all_results,
        "aggregate": agg,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
