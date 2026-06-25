"""C0 — the zero-GPU SUBSTRATE-vs-SCALE split for the foundational-curriculum generalization miss.

THE QUESTION (from `_curriculum_gen_miss_DEEP_scoping.md`): the validated stream-cortex generalization 0.91
was a PURE-NUMPY EXACT-COUNT pipeline at 64 concepts / 8 categories (`_phaseB_online_stream_cortex_derisk.py`).
The Step-1 320-concept run gets gen ~0.15 on a SPIKING bridge (a noisy population read of learned weights,
corr(M,C)=0.756). The two priors blamed (1) the yardstick, then (2) the vocab -- BOTH refuted. The DEEP gate
found the dominant cause is most likely the SPIKING READ-OUT FIDELITY AT SCALE, not the vocab. The single
decisive control with ZERO GPU is C0:

  Run the SAME numpy EXACT-COUNT pipeline (M[a,b]+=1, double_center(log1p(M*100)), heldout_generalization)
  at the STEP-1 SCALE -- 320 concepts / 40 categories / n_hub=300 (the Step-1 bridge value) -- and see whether
  the numpy-exact-count gen STAYS HIGH or COLLAPSES.

  * numpy-320 STAYS HIGH (gen >= ~0.6, Pearson >= +0.40)  => the SPIKING READ-OUT is the dominant cause
        (the substrate, not scale/metric). Lever: on-bridge fidelity (n_per / windows / n_hub).
  * numpy-320 COLLAPSES (gen ~0.15, Pearson ~+0.07)       => it is SCALE / metric-granularity (40-cat chance
        + thin hub), NOT the spiking substrate. Addressable by curriculum granularity, not a substrate wall.

This harness REUSES THE VALIDATED PIPELINE BODY VERBATIM (the same `run_seed` loop from
`_phaseB_online_stream_cortex_derisk.py`) but parameterized on `n_hub` and the vocab/cat_ids, so the ONLY
changed axes vs the validated 0.91 are (i) #concepts 64->320, (ii) #categories 8->40, (iii) n_hub 500->300.
Substrate stays NUMPY EXACT-COUNT. NO sim/ edit, NO GPU, NO bridge. Numpy host, ~minutes CPU.

ANTI-CHEATS (per the DEEP scoping SS4):
  - The gen reference is the INDEPENDENT a-priori TAXONOMY_40x8 category blocks (NEVER corpus-derived).
  - Report gen on BOTH the 64 (reproduce-0.91 anchor) AND the 320 arm, plus ratio_vs_chance AND
    Pearson(cos, S_true) (the chance-INDEPENDENT tell -- the load-bearing number).
  - Provenance: print the validated/Step-1 prior numbers alongside.

Run:  SIM_BACKEND=numpy python -u -m research.runners._curriculum_gen_C0_substrate_vs_scale
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse the VALIDATED pipeline pieces verbatim (the exact-count substrate + normalization + gen metric):
from research.runners._phaseB_online_stream_cortex_derisk import (  # noqa: E402
    WINDOW,
    EMA_ALPHA,
    double_center,
    load_token_stream,
)
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim,
    _pearson_vs_Strue,
    heldout_generalization,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
)
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8  # noqa: E402

SEEDS = (42, 43, 44)


def run_seed_numpy_exact(seed, stories, vocab, cat_ids, n_hub):
    """The VALIDATED `run_seed` body, byte-for-byte, but parameterized on `n_hub` (validated hardcodes 500).
    Pure NUMPY exact-count: M[a,b] += 1 (corr(M,C)=1.0 by construction). Returns gen + Pearson(cos,S_true).
    """
    rng = np.random.RandomState(seed)
    targets = list(vocab)
    target_set = set(targets)
    Nt = len(targets)
    S_true = (np.asarray(cat_ids)[:, None] == np.asarray(cat_ids)[None, :]).astype(np.float64)
    # STEP 0 (pick hubs): top-N frequent context words (the stream's running global frequency).
    gfreq = Counter()
    for toks in stories:
        gfreq.update(toks)
    hubs = [w for w, _ in gfreq.most_common() if w not in STOPLIST and w not in target_set][:n_hub]
    hub_idx = {w: i for i, w in enumerate(hubs)}
    keep = target_set | set(hubs)
    tgt_row = {w: i for i, w in enumerate(targets)}

    # ONLINE LEARNING: stream kept tokens, WM window, Hebbian M[target, hub] += 1 (EXACT integer count).
    M = np.zeros((Nt, n_hub), dtype=np.float64)
    freq = np.zeros(n_hub, dtype=np.float64)
    n_updates = 0
    story_order = rng.permutation(len(stories))
    for si in story_order:
        kept = [t for t in stories[si] if t in keep]
        for c in range(len(kept)):
            w = kept[c]
            lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1)
            ctx = set(kept[lo:hi]) - {w}
            for u in kept[lo:hi]:
                if u in hub_idx:
                    freq[hub_idx[u]] += EMA_ALPHA * (1.0 - freq[hub_idx[u]])
            if w in target_set:
                for u in ctx:
                    if u in hub_idx:
                        M[tgt_row[w], hub_idx[u]] += 1.0
                        n_updates += 1
    assert n_updates > 0
    # The concept CODE: log-domain double-centering of the EXACT-count M (the validated normalization).
    code = double_center(np.log1p(M * 100.0))
    pear = _pearson_vs_Strue(_cos_sim(code), S_true)
    gen, ch = heldout_generalization(code, np.asarray(cat_ids))
    return {
        "seed": seed,
        "gen": float(gen),
        "chance": float(ch),
        "ratio_vs_chance": float(gen / ch) if ch > 0 else None,
        "pearson_cos_Strue": float(pear),
        "n_hub": int(n_hub),
        "n_concepts": int(Nt),
        "n_categories": int(len(set(np.asarray(cat_ids).tolist()))),
        "n_updates": int(n_updates),
    }


def run_arm(label, taxonomy, n_hub, stories):
    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(taxonomy)
    print(f"\n[{label}] {len(vocab)} concepts x {len(cat_names)} categories | n_hub={n_hub} "
          f"| chance {1.0/len(cat_names):.4f}", flush=True)
    rows = [run_seed_numpy_exact(s, stories, vocab, cat_ids, n_hub) for s in SEEDS]
    for r in rows:
        print(f"    seed {r['seed']}: gen {r['gen']:.3f} (ch {r['chance']:.4f}, "
              f"ratio {r['ratio_vs_chance']:.2f}x) | Pearson(cos,S_true) {r['pearson_cos_Strue']:+.3f} "
              f"| {r['n_updates']} exact-count updates", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    summ = {
        "label": label,
        "n_concepts": rows[0]["n_concepts"],
        "n_categories": rows[0]["n_categories"],
        "n_hub": n_hub,
        "chance": rows[0]["chance"],
        "gen_mean": m("gen"),
        "ratio_vs_chance_mean": m("ratio_vs_chance"),
        "pearson_cos_Strue_mean": m("pearson_cos_Strue"),
        "per_seed": rows,
    }
    print(f"  MEAN ({len(SEEDS)} seeds): gen {summ['gen_mean']:.3f} "
          f"(ratio {summ['ratio_vs_chance_mean']:.2f}x chance) | "
          f"Pearson(cos,S_true) {summ['pearson_cos_Strue_mean']:+.3f}", flush=True)
    return summ


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("=" * 100, flush=True)
    print("  C0 — SUBSTRATE-vs-SCALE split (zero GPU): SAME numpy EXACT-COUNT pipeline at 64 vs 320 scale.",
          flush=True)
    print("  numpy-320 STAYS HIGH (gen>=~0.6, Pearson>=+0.40) => the SPIKING READ-OUT is the cause "
          "(substrate).", flush=True)
    print("  numpy-320 COLLAPSES (gen~0.15, Pearson~+0.07)    => it is SCALE/metric, NOT the substrate.",
          flush=True)
    print("=" * 100, flush=True)
    stories = load_token_stream()
    print(f"  loaded {len(stories)} stories from TinyStories", flush=True)

    # ARM 1 (the reproduce-0.91 anchor): VALIDATED config -- TAXONOMY_8x8 (64), n_hub=500.
    arm64 = run_arm("numpy-64 (reproduce-0.91 anchor: TAXONOMY_8x8, n_hub=500)", TAXONOMY_8x8, 500, stories)

    # ARM 2 (the Step-1 scale): SAME numpy exact-count -- TAXONOMY_40x8 (320), n_hub=300 (the bridge value).
    arm320 = run_arm("numpy-320 (Step-1 scale: TAXONOMY_40x8, n_hub=300)", TAXONOMY_40x8, 300, stories)

    # ---- VERDICT (let C0 decide; do NOT predict) ----
    g64, p64 = arm64["gen_mean"], arm64["pearson_cos_Strue_mean"]
    g320, p320 = arm320["gen_mean"], arm320["pearson_cos_Strue_mean"]
    reproduced = abs(g64 - 0.91) <= 0.05 and p64 >= 0.45
    stays_high = (g320 >= 0.60) and (p320 >= 0.40)
    collapses = (g320 <= 0.25) or (p320 <= 0.15)
    if stays_high:
        verdict = ("SUBSTRATE-READ-OUT: numpy-exact-count STAYS HIGH at 320 (gen {:.3f}, Pearson {:+.3f}) "
                   "=> the SPIKING READ-OUT FIDELITY is the dominant cause of the on-bridge gen miss, NOT "
                   "scale/metric. Lever: on-bridge read-out fidelity (n_per / windows / n_hub) -- the "
                   "documented rate-code-wall lever -- OR the honest on-bridge fidelity boundary."
                   ).format(g320, p320)
    elif collapses:
        verdict = ("SCALE/METRIC: numpy-exact-count ALSO COLLAPSES at 320 (gen {:.3f}, Pearson {:+.3f}) "
                   "=> the gen miss is SCALE / metric-granularity (40-cat chance {:.4f} + thin hub 300/320), "
                   "NOT the spiking substrate. The spiking substrate is EXONERATED. Addressable by "
                   "curriculum/metric granularity, not a substrate wall."
                   ).format(g320, p320, arm320["chance"])
    else:
        verdict = ("MIXED: numpy-320 lands between the bars (gen {:.3f}, Pearson {:+.3f}) -- a PARTIAL "
                   "scale cost on the exact-count substrate. Both scale AND read-out contribute; the "
                   "residual on-bridge gap (numpy-320 minus bridge-320) isolates the spiking cost."
                   ).format(g320, p320)
    print("\n" + "=" * 100, flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print("=" * 100, flush=True)
    print(f"  ANCHOR reproduced 0.91: {reproduced} (numpy-64 gen {g64:.3f}, Pearson {p64:+.3f})", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {
        "experiment": "C0_substrate_vs_scale",
        "substrate": "pure-numpy-exact-count (M[a,b]+=1, corr(M,C)=1.0 by construction) -- ZERO GPU/bridge",
        "pipeline": "double_center(log1p(M*100)) + heldout_generalization (byte-identical to validated runner)",
        "seeds": list(SEEDS),
        "anchor_reproduced_0.91": bool(reproduced),
        "numpy_64": {
            "gen_mean": g64, "pearson_cos_Strue_mean": p64,
            "chance": arm64["chance"], "ratio_vs_chance_mean": arm64["ratio_vs_chance_mean"],
            "n_concepts": arm64["n_concepts"], "n_categories": arm64["n_categories"], "n_hub": arm64["n_hub"],
        },
        "numpy_320": {
            "gen_mean": g320, "pearson_cos_Strue_mean": p320,
            "chance": arm320["chance"], "ratio_vs_chance_mean": arm320["ratio_vs_chance_mean"],
            "n_concepts": arm320["n_concepts"], "n_categories": arm320["n_categories"], "n_hub": arm320["n_hub"],
        },
        "verdict": verdict,
        "stays_high": bool(stays_high),
        "collapses": bool(collapses),
        "go_bars": {
            "stays_high_substrate": "gen>=0.60 AND Pearson>=+0.40",
            "collapses_scale": "gen<=0.25 OR Pearson<=+0.15",
        },
        "provenance_priors": {
            "validated_numpy_64": {"gen": 0.91, "pearson": 0.513, "chance": 0.125,
                                   "source": "_phaseB_online_stream_cortex_derisk.py / CYCLE-94"},
            "step1_320_bridge_all": {"gen": 0.153, "pearson": 0.070, "chance": 0.0333, "corr_MC": 0.756,
                                     "source": "_curriculum_step1_320_real_corpus_seed42.json (spiking bridge)"},
            "step1_320_bridge_coherent_remeasure": {"gen": 0.167, "pearson": 0.082, "note": "prompt-reported"},
            "step1_320_bridge_content_remeasure": {"gen": 0.125, "note": "prompt-reported; content-filter REFUTED"},
            "onbridge_64_curated": {"gen": 0.45, "note": "_phaseB_onbridge_stream_cortex_derisk.py 30K windows -- "
                                    "numpy->bridge alone costs ~0.46 of gen on the validated vocab"},
            "refuted_hypotheses": ["yardstick-swap (0.153->0.167)", "content-vocab (->0.125)"],
        },
    }
    path = os.path.join(_REPO, "research", "findings", "raw", "_curriculum_gen_C0_substrate_vs_scale.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}\n", flush=True)


if __name__ == "__main__":
    main()
