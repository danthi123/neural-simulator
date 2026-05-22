"""Pattern-separation grounding probe -- the routing from the shortcut-2
NEGATIVE.

Biologization shortcut 2 (ground the composition symbol in the
substrate's own activity) is a terminal NEGATIVE: the substrate's
consolidated concept representations overlap by a mean pairwise
similarity of about 0.45, and FHRR/VSA composition requires
near-orthogonal atomic symbols, so the grounded symbols crosstalk.

The project's own validated biology supplies the candidate fix. The
brain orthogonalises overlapping representations by PATTERN SEPARATION,
and the project has a validated pattern-separation result -- the
hippocampal dentate gyrus (catalog D.12): a validation run measured an
input cosine of 0.80 reduced to a dentate-gyrus cosine of 0.218.

Pattern separation is an expansion (a projection into a
higher-dimensional space) followed by sparsification (a small fraction
of cells active -- a k-winners-take-all). The nonlinearity of the
sparsification is what decorrelates: two overlapping inputs activate
mostly-disjoint sparse sets in the expanded space.

This probe asks, cheaply: does dentate-gyrus-style pattern separation
of the substrate's overlapping concept representations produce
near-orthogonal symbols that FHRR-compose?

Pipeline:
1. Load the real consolidated concept activity vectors (16 concepts,
   from the activity cache -- the same vectors whose derived symbols
   overlap by 0.45).
2. Apply a dentate-gyrus transform: a fixed random expansion, then a
   k-winners-take-all sparsification. The transform is fixed and
   deterministic -- the same activity always maps to the same separated
   code; the orthogonality, if it appears, emerges from the
   expansion + sparsification, it is not assigned.
3. Derive a phasor symbol from each separated code.
4. Measure the mean pairwise similarity of the separated symbols
   (against the 0.45 baseline and the D.12-measured ~0.2 target) and
   run the FHRR compositional task with them.
5. Also test recognition: a noisy per-observation activity, separated
   and derived, must still map to its own concept's separated symbol.

PRE-REGISTERED reading (fixed; never tuned):
- If the separated symbols have a mean pairwise similarity well below
  0.45 (toward the D.12 ~0.2 range) AND the FHRR composition clears the
  frozen 0.80 bar at loads {2,3,5} AND recognition of a noisy
  observation stays at or above 0.80 -> pattern-separation grounding is
  REACHABLE; proceed to the real build (route substrate activity
  through the validated dentate gyrus before the symbol).
- Otherwise -> the honest finding is which of those three fails, and
  shortcut 2 is closed accordingly (separation that breaks recognition
  is the classic separation-versus-completion tension; separation that
  does not orthogonalise enough means the substrate's concepts are
  irreducibly entangled).

Standalone numpy, ENGINEERING ceiling-clarification (non-load-bearing).
Reuses the activity cache + deriver pattern by import. No
protected/frozen/moat module touched. No automatic differentiation.
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

from research.findings.raw.activity_level_integration import (
    capture_seed, CACHE_DIR, K_VOCAB,
)
from research.runners.unified_per_regime_monitor_runner import (
    _direct_pool_target,
)

SEEDS = [42, 43, 44]
DG_EXPANSION = 4.0          # dentate-gyrus expansion ratio (DG has ~4x EC cells)
DG_SPARSITY = 0.02          # ~2% of dentate-gyrus cells active (biological)
N_DIM = 512                 # FHRR phasor dimension
LOADS = [2, 3, 5]
N_TRIALS = 200
BAR = 0.80
DG_SEED = 20260522          # fixed -- the dentate-gyrus projection is fixed
DERIV_SEED = 777            # fixed -- the symbol derivation is fixed
SEPARATION_TARGET = 0.30    # separated mean similarity must fall below this


def dg_separate(activity, e_matrix, k):
    """Dentate-gyrus pattern separation: expand by a fixed random
    projection, then keep the k strongest units (k-winners-take-all),
    zeroing the rest. A fixed, deterministic transform."""
    a = np.maximum(np.asarray(activity, dtype=np.float64), 0.0)
    norm = np.linalg.norm(a)
    expanded = e_matrix @ (a / (norm + 1e-9))
    if k < expanded.shape[0]:
        cutoff = np.partition(expanded, -k)[-k]
        expanded = np.where(expanded >= cutoff, expanded, 0.0)
    return expanded


def make_deriver(n_dim, d_in, seed):
    """Fixed random complex projection: a (normalised) code -> a phasor
    symbol's phases in [0,1)."""
    rng = np.random.default_rng(seed)
    w_re = rng.normal(0.0, 1.0, size=(n_dim, d_in))
    w_im = rng.normal(0.0, 1.0, size=(n_dim, d_in))

    def derive(code):
        c = np.asarray(code, dtype=np.float64)
        norm = np.linalg.norm(c)
        z = w_re @ (c / (norm + 1e-9)) + 1j * (w_im @ (c / (norm + 1e-9)))
        return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)

    return derive


def phase_sim(a, b):
    return float(np.mean(np.cos(2.0 * np.pi * (a - b))))


def bind(a, b):
    return np.mod(a + b, 1.0)


def unbind(a, b):
    return np.mod(a - b, 1.0)


def bundle(phase_list):
    z = np.sum([np.exp(2j * np.pi * p) for p in phase_list], axis=0)
    return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)


def mean_pairwise_sim(symbols):
    n = len(symbols)
    sims = [phase_sim(symbols[a], symbols[b])
            for a in range(n) for b in range(n) if a != b]
    return float(np.mean(sims))


def run_one_seed(seed):
    print(f"\n--- seed {seed} ---")
    cache_path = os.path.join(CACHE_DIR, f"full_seed{seed}.npz")
    obs, clean, _slices, _all_pools, words = capture_seed(
        seed, cache_path, 16)
    d_act = obs[words[0]].shape[1]

    # Consolidated concept activity = K-averaged real activity per word.
    consolidated = {w: obs[w][:K_VOCAB].mean(axis=0) for w in words}

    # Fixed dentate-gyrus transform + fixed symbol deriver.
    d_dg = int(DG_EXPANSION * d_act)
    k = max(1, int(DG_SPARSITY * d_dg))
    e_matrix = np.random.default_rng(DG_SEED).normal(
        0.0, 1.0, size=(d_dg, d_act))
    derive_raw = make_deriver(N_DIM, d_act, DERIV_SEED)
    derive_sep = make_deriver(N_DIM, d_dg, DERIV_SEED + 1)

    # Baseline: symbols derived from raw activity (the shortcut-2 case).
    raw_sym = {w: derive_raw(consolidated[w]) for w in words}
    raw_mean_sim = mean_pairwise_sim([raw_sym[w] for w in words])

    # Pattern-separated: symbols derived from the dentate-gyrus code.
    sep_code = {w: dg_separate(consolidated[w], e_matrix, k) for w in words}
    sep_sym = {w: derive_sep(sep_code[w]) for w in words}
    sep_mean_sim = mean_pairwise_sim([sep_sym[w] for w in words])

    # Recognition: a noisy per-observation activity, separated + derived,
    # must still match its own concept's separated symbol.
    n_rec_correct = n_rec_total = 0
    widx = {w: i for i, w in enumerate(words)}
    for w in words:
        for i in range(16):
            obs_sym = derive_sep(dg_separate(obs[w][i], e_matrix, k))
            sims = [phase_sim(obs_sym, sep_sym[ww]) for ww in words]
            n_rec_correct += int(int(np.argmax(sims)) == widx[w])
            n_rec_total += 1
    rec_acc = n_rec_correct / n_rec_total

    # FHRR composition with the pattern-separated symbols.
    cue_words = [w for w in words
                 if _direct_pool_target(w).startswith(("noun_pool_",
                                                        "verb_pool_"))]
    filler_words = [w for w in words
                    if _direct_pool_target(w).startswith("adjective_pool_")]
    qrng = np.random.default_rng(seed + 1)
    per_load = {}
    for load in LOADS:
        n_ok = n_tot = 0
        for _ in range(N_TRIALS):
            cues = list(qrng.choice(cue_words, size=load, replace=False))
            fills = list(qrng.choice(filler_words, size=load, replace=True))
            facts = list(zip(cues, fills))
            composite = bundle([bind(sep_sym[c], sep_sym[f])
                                for (c, f) in facts])
            for (c, f) in facts:
                recovered = unbind(composite, sep_sym[c])
                sims = {fw: phase_sim(recovered, sep_sym[fw])
                        for fw in filler_words}
                if max(sims, key=sims.get) == f:
                    n_ok += 1
                n_tot += 1
        per_load[load] = n_ok / n_tot

    print(f"  raw-derived symbols: mean pairwise similarity={raw_mean_sim:.3f}")
    print(f"  pattern-separated symbols: mean pairwise similarity="
          f"{sep_mean_sim:.3f}")
    print(f"  noisy-observation recognition (separated): {rec_acc:.3f}")
    for load in LOADS:
        print(f"  L={load}: FHRR composition (separated symbols)="
              f"{per_load[load]:.4f}")
    return {
        "seed": seed,
        "raw_mean_similarity": raw_mean_sim,
        "separated_mean_similarity": sep_mean_sim,
        "recognition_acc": rec_acc,
        "composition": {str(k): v for k, v in per_load.items()},
    }


def main():
    print("=== pattern-separation grounding probe ===")
    print(f"seeds {SEEDS}; DG expansion={DG_EXPANSION}x, sparsity="
          f"{DG_SPARSITY}; FHRR N_dim={N_DIM}; loads={LOADS}; bar={BAR}")

    results = [run_one_seed(s) for s in SEEDS]

    raw_sim = float(np.mean([r["raw_mean_similarity"] for r in results]))
    sep_sim = float(np.mean([r["separated_mean_similarity"] for r in results]))
    rec = float(np.mean([r["recognition_acc"] for r in results]))
    comp = {load: float(np.mean([r["composition"][str(load)]
                                 for r in results]))
            for load in LOADS}

    print(f"\n=== MULTI-SEED AGGREGATE ===")
    print(f"  raw-derived mean similarity:       {raw_sim:.3f}")
    print(f"  pattern-separated mean similarity: {sep_sim:.3f} "
          f"(target < {SEPARATION_TARGET})")
    print(f"  noisy-observation recognition:     {rec:.3f} (target >= {BAR})")
    for load in LOADS:
        print(f"  L={load}: composition {comp[load]:.4f} "
              f"({'>=' if comp[load] >= BAR else '<'} {BAR})")

    separated_ok = sep_sim < SEPARATION_TARGET
    recognition_ok = rec >= BAR
    composition_ok = all(comp[load] >= BAR for load in LOADS)

    print(f"\n=== VERDICT ===")
    if separated_ok and recognition_ok and composition_ok:
        verdict = "PATTERN_SEPARATION_GROUNDING_REACHABLE"
        print("  Pattern separation orthogonalises the substrate's "
              "concept representations, recognition survives, and the "
              "separated symbols clear the frozen 0.80 compositional bar. "
              "Grounding the symbol via pattern separation is REACHABLE -- "
              "proceed to the real build (route substrate activity through "
              "the validated dentate gyrus before the symbol).")
    else:
        verdict = "PATTERN_SEPARATION_GROUNDING_INSUFFICIENT"
        fails = []
        if not separated_ok:
            fails.append("the separation does not orthogonalise enough")
        if not recognition_ok:
            fails.append("separation breaks recognition (the separation-"
                          "versus-completion tension)")
        if not composition_ok:
            fails.append("the separated symbols still do not compose")
        print(f"  Pattern-separation grounding is insufficient: "
              f"{'; '.join(fails)}. Shortcut 2 is closed accordingly -- "
              f"the composition layer keeps oracle-assigned symbols, "
              f"honestly framed.")

    out = {
        "seeds": SEEDS, "dg_expansion": DG_EXPANSION,
        "dg_sparsity": DG_SPARSITY, "n_dim": N_DIM, "loads": LOADS,
        "n_trials": N_TRIALS, "bar": BAR,
        "separation_target": SEPARATION_TARGET,
        "per_seed": results,
        "aggregate": {
            "raw_mean_similarity": raw_sim,
            "separated_mean_similarity": sep_sim,
            "recognition_acc": rec,
            "composition": {str(k): v for k, v in comp.items()},
        },
        "verdict": verdict,
    }
    with open("research/findings/raw/pattern_separation_grounding_probe.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/pattern_separation_grounding_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
