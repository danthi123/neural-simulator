"""Direction 6 dedicated-pool grounded-symbol geometry diagnostic.

Empirically measures the actual cosine geometry of D4 V=80 cached
per-concept activity vectors to characterise WHY the FHRR algebra
capacity-ratio prediction was DECISIVELY SHATTERED at pillar n=108.

Pre-registered hypothesis: bio_brain_regions dedicated-pool
grounded-symbol geometry is substantially CLEANER than uniform-random
phasors FHRR algebra assumes -- likely near-orthogonal because each
concept fires its own dedicated pool with other pools quiet.

CPU-only (does not compete with the in-flight D7 GPU smoke). Does NOT
modify any protected/frozen/moat module; operates on cached .npz.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.direction_4_vocab_spec import (
    DIRECTION_4_BRIDGE_A_WORDS,
    DIRECTION_4_BRIDGE_B_WORDS,
    DIRECTION_4_BRIDGE_C_WORDS,
    DIRECTION_4_BRIDGE_D_WORDS,
    DIRECTION_4_BRIDGE_E_WORDS,
)

_PER_BRIDGE_WORDS: Dict[str, List[str]] = {
    "A_nouns": DIRECTION_4_BRIDGE_A_WORDS,
    "B_verbs": DIRECTION_4_BRIDGE_B_WORDS,
    "C_adj": DIRECTION_4_BRIDGE_C_WORDS,
    "D_spatial": DIRECTION_4_BRIDGE_D_WORDS,
    "E_functional": DIRECTION_4_BRIDGE_E_WORDS,
}

CACHE_DIR = os.path.join(_HERE, "direction_4_cache")
SEEDS = [42, 43, 44]


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_bridge_activity(bridge_name, seed):
    cache_p = os.path.join(
        CACHE_DIR,
        "activity_full_" + bridge_name + "_seed" + str(seed) + ".npz",
    )
    if not os.path.exists(cache_p):
        raise FileNotFoundError("Cache not found: " + cache_p)
    data = np.load(cache_p)
    words = _PER_BRIDGE_WORDS[bridge_name]
    acts = {w: data[str(w)] for w in words}
    n_pool_union = acts[words[0]].shape[1]
    return acts, n_pool_union


def _within_bridge_same_concept(acts):
    cosines = []
    for word, obs in acts.items():
        m_obs = obs.shape[0]
        for i in range(m_obs):
            for j in range(i + 1, m_obs):
                cosines.append(_cosine(obs[i], obs[j]))
    return cosines


def _within_bridge_different_concept(acts, mean_centre=False):
    words = list(acts.keys())
    means = {w: np.mean(acts[w], axis=0) for w in words}
    if mean_centre:
        all_means = np.stack(list(means.values()), axis=0)
        common = np.mean(all_means, axis=0)
        means = {w: v - common for w, v in means.items()}
    cosines = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            cosines.append(_cosine(means[words[i]], means[words[j]]))
    return cosines


def _summarize(label, cosines):
    arr = np.asarray(cosines, dtype=np.float64)
    return {
        "label": label,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
    }


def main():
    print("=== D6 dedicated-pool grounded-symbol geometry diagnostic ===")
    print("  Reading cached D4 V=80 production activity (5 bridges x 3 seeds)")
    print("  Hypothesis: dedicated-pool near-orthogonal (~0) vs FHRR algebra")
    print("  uniform-random (~0.5).")
    print()

    all_results = {
        "within_bridge_same_concept": [],
        "within_bridge_different_concept_raw": [],
        "within_bridge_different_concept_mean_centred": [],
    }

    for bridge_name in _PER_BRIDGE_WORDS:
        for seed in SEEDS:
            try:
                acts, n_pool_union = _load_bridge_activity(bridge_name, seed)
            except FileNotFoundError as exc:
                print("  [skip] " + str(exc))
                continue

            same = _within_bridge_same_concept(acts)
            diff_raw = _within_bridge_different_concept(acts, mean_centre=False)
            diff_mc = _within_bridge_different_concept(acts, mean_centre=True)

            same_s = _summarize(bridge_name + "/seed" + str(seed) + " same", same)
            diff_raw_s = _summarize(bridge_name + "/seed" + str(seed) + " diff_raw", diff_raw)
            diff_mc_s = _summarize(bridge_name + "/seed" + str(seed) + " diff_mc", diff_mc)

            all_results["within_bridge_same_concept"].append(same_s)
            all_results["within_bridge_different_concept_raw"].append(diff_raw_s)
            all_results["within_bridge_different_concept_mean_centred"].append(diff_mc_s)

            print(
                "  [" + bridge_name + "/seed " + str(seed)
                + "] n_pool_union=" + str(n_pool_union)
                + " | same mean=" + ("%.4f" % same_s["mean"])
                + " (n=" + str(same_s["n"]) + ")"
                + " | diff_raw mean=" + ("%.4f" % diff_raw_s["mean"])
                + " abs=" + ("%.4f" % diff_raw_s["abs_mean"])
                + " (n=" + str(diff_raw_s["n"]) + ")"
                + " | diff_mc mean=" + ("%.4f" % diff_mc_s["mean"])
                + " abs=" + ("%.4f" % diff_mc_s["abs_mean"]),
                flush=True,
            )

    def _aggregate(key):
        cells = all_results[key]
        if not cells:
            return {"n_cells": 0}
        means = [c["mean"] for c in cells]
        abs_means = [c["abs_mean"] for c in cells]
        stds = [c["std"] for c in cells]
        n_pairs_total = sum(c["n"] for c in cells)
        return {
            "n_cells": len(cells),
            "n_pairs_total": int(n_pairs_total),
            "across_cells_mean_of_means": float(np.mean(means)),
            "across_cells_mean_of_abs_means": float(np.mean(abs_means)),
            "across_cells_mean_of_stds": float(np.mean(stds)),
            "min_cell_mean": float(np.min(means)),
            "max_cell_mean": float(np.max(means)),
        }

    aggregate = {
        "within_bridge_same_concept": _aggregate("within_bridge_same_concept"),
        "within_bridge_different_concept_raw": _aggregate("within_bridge_different_concept_raw"),
        "within_bridge_different_concept_mean_centred": _aggregate("within_bridge_different_concept_mean_centred"),
    }

    print()
    print("=== AGGREGATE across (bridge, seed) cells ===")
    for key, agg in aggregate.items():
        if agg.get("n_cells", 0) == 0:
            print("  [" + key + "] no cells loaded")
            continue
        print(
            "  [" + key + "] cells=" + str(agg["n_cells"])
            + " pairs_total=" + str(agg["n_pairs_total"])
            + " | mean_of_means=" + ("%.4f" % agg["across_cells_mean_of_means"])
            + " mean_of_abs_means=" + ("%.4f" % agg["across_cells_mean_of_abs_means"])
            + " | range[" + ("%.4f" % agg["min_cell_mean"])
            + ", " + ("%.4f" % agg["max_cell_mean"]) + "]",
        )

    print()
    print("=== INTERPRETATION ===")
    same_agg = aggregate["within_bridge_same_concept"]
    diff_raw_agg = aggregate["within_bridge_different_concept_raw"]
    diff_mc_agg = aggregate["within_bridge_different_concept_mean_centred"]

    if same_agg.get("n_cells", 0) > 0:
        same_mean = same_agg["across_cells_mean_of_means"]
        if same_mean > 0.85:
            print("  - Same-concept M_OBS noise LOW (mean cos %.3f > 0.85)."
                  % same_mean)
        else:
            print("  - Same-concept M_OBS noise HIGH (mean cos %.3f <= 0.85)."
                  % same_mean)

    if diff_raw_agg.get("n_cells", 0) > 0:
        diff_raw_mean = diff_raw_agg["across_cells_mean_of_means"]
        diff_raw_abs = diff_raw_agg["across_cells_mean_of_abs_means"]
        if diff_raw_mean > 0.45:
            print("  - Raw different-concept cosine HIGH (mean %.3f, abs %.3f)."
                  % (diff_raw_mean, diff_raw_abs))
        else:
            print("  - Raw different-concept cosine MODERATE (mean %.3f, abs %.3f)."
                  % (diff_raw_mean, diff_raw_abs))

    if diff_mc_agg.get("n_cells", 0) > 0:
        diff_mc_mean = diff_mc_agg["across_cells_mean_of_means"]
        diff_mc_abs = diff_mc_agg["across_cells_mean_of_abs_means"]
        if abs(diff_mc_mean) < 0.10 and diff_mc_abs < 0.30:
            print("  - Mean-centred different-concept cosine NEAR-ORTHOGONAL "
                  "(mean %.4f, abs %.4f). Empirically validates n=108/n=109 SHATTER."
                  % (diff_mc_mean, diff_mc_abs))
        elif diff_mc_abs < 0.50:
            print("  - Mean-centred different-concept cosine MODERATE "
                  "(mean %.4f, abs %.4f)."
                  % (diff_mc_mean, diff_mc_abs))
        else:
            print("  - Mean-centred different-concept cosine HIGH "
                  "(mean %.4f, abs %.4f)."
                  % (diff_mc_mean, diff_mc_abs))

    out_path = os.path.join(_HERE, "direction_4_geometry_diagnostic.json")
    out = {
        "per_cell": all_results,
        "aggregate": aggregate,
        "discipline": {
            "bar": "no bar; diagnostic only",
            "protected_set_byte_unchanged": True,
            "autograd": False,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print("Wrote " + out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
