"""Dentate-gyrus orthogonality sweep -- the smell-test of the
fully-biologized grounded composition NEGATIVE.

The fully-biologized grounded compositional pipeline failed because the
attractor clean-up is degenerate over the dentate-gyrus pattern-
separated grounded symbols (0.19 mean pairwise similarity at the
biological DG setting of 4x expansion, 2% sparsity). The honest
question this smell-test answers: is that NEGATIVE genuine, or an
artifact of a too-weak DG setting? Could a stronger DG orthogonalise
the substrate's concept representations enough for the attractor?

Recognition in the pipeline is done by temporal averaging in the raw
activity space; it does NOT pass through the dentate gyrus. So a
stronger DG does not cost recognition -- it only affects the stored
symbol. This sweep therefore varies the DG transform (expansion ratio
and sparsity) freely, well past biological values, and measures, for
each setting: the stored-symbol mean pairwise similarity, and whether
the attractor clean-up can identify a clean stored symbol as itself.

If some setting drives the similarity low enough that the attractor
discriminates the symbols, the pipeline NEGATIVE is fixable. If even
the most aggressive setting cannot, the NEGATIVE is genuine and the
substrate's concept representations are irreducibly entangled under
pattern separation.

Standalone numpy, reuses the activity cache. No protected/frozen/moat
module touched. No automatic differentiation.
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
from research.findings.raw.pattern_separation_grounding_probe import (
    dg_separate, make_deriver, DG_SEED,
)
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, phase_similarity,
)
from research.runners.resonate_fire_fhrr import (
    ResonateFireTPAM, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)

SEEDS = [42, 43, 44]
EXPANSIONS = [4, 8, 16, 32]
SPARSITIES = [0.02, 0.01, 0.005, 0.002]
N_DIM = 512
DERIV_SEED = 90909
# The attractor clean-up was validated on random near-orthogonal
# symbols (mean |similarity| ~ 1/sqrt(512) ~ 0.044). A stored symbol
# set must be at least this orthogonal for the attractor.
ATTRACTOR_ORTHOGONALITY_NEED = 0.05


def run_one_seed(seed):
    obs, _clean, _slices, _pools, words = capture_seed(
        seed, os.path.join(CACHE_DIR, f"full_seed{seed}.npz"), 16)
    d_act = obs[words[0]].shape[1]
    consolidated = {w: obs[w][:K_VOCAB].mean(axis=0) for w in words}
    n_words = len(words)

    grid = {}
    for expn in EXPANSIONS:
        for spars in SPARSITIES:
            d_dg = int(expn * d_act)
            k = max(1, int(spars * d_dg))
            e_matrix = np.random.default_rng(DG_SEED).normal(
                0.0, 1.0, size=(d_dg, d_act))
            deriver = make_deriver(N_DIM, d_dg, DERIV_SEED)
            syms = [phases_to_spikes(deriver(
                dg_separate(consolidated[w], e_matrix, k))) for w in words]
            sim = float(np.mean([phase_similarity(syms[a], syms[b])
                                 for a in range(n_words)
                                 for b in range(n_words) if a != b]))
            tpam = ResonateFireTPAM(syms)
            n_id = 0
            for i in range(n_words):
                z, _ = tpam.settle_annealed(
                    syms[i], ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH,
                    ANNEAL_ITERS, fast=True)
                n_id += int(int(np.argmax(np.abs(tpam.s.conj().T @ z))) == i)
            grid[(expn, spars)] = {"similarity": sim,
                                   "attractor_self_id": n_id,
                                   "n_words": n_words}
    return grid


def main():
    print("=== dentate-gyrus orthogonality sweep ===")
    print(f"seeds {SEEDS}; expansions {EXPANSIONS}; sparsities {SPARSITIES}")
    per_seed = [run_one_seed(s) for s in SEEDS]

    print(f"\n  expansion x sparsity -> mean stored-symbol similarity | "
          f"attractor self-ID (mean /16)")
    best_sim = 1.0
    any_attractor_ok = False
    grid_out = {}
    for expn in EXPANSIONS:
        for spars in SPARSITIES:
            sims = [g[(expn, spars)]["similarity"] for g in per_seed]
            ids = [g[(expn, spars)]["attractor_self_id"] for g in per_seed]
            nws = per_seed[0][(expn, spars)]["n_words"]
            msim = float(np.mean(sims))
            mid = float(np.mean(ids))
            best_sim = min(best_sim, msim)
            if mid >= 0.9 * nws:
                any_attractor_ok = True
            grid_out[f"exp{expn}_spars{spars}"] = {
                "mean_similarity": msim, "mean_attractor_self_id": mid,
                "n_words": nws}
            print(f"    exp{expn:>2}x spars{spars:.3f}: similarity={msim:.3f}"
                  f"  attractor self-ID={mid:.1f}/{nws}")

    print(f"\n=== VERDICT ===")
    print(f"  best (lowest) stored-symbol similarity reached, any DG "
          f"setting: {best_sim:.3f}")
    print(f"  attractor clean-up needs similarity below ~"
          f"{ATTRACTOR_ORTHOGONALITY_NEED}")
    if any_attractor_ok:
        verdict = "DG_CAN_ORTHOGONALISE_FOR_ATTRACTOR"
        print(f"  Some DG setting orthogonalises the substrate's concept "
              f"symbols enough for the attractor clean-up to discriminate "
              f"them -- the fully-biologized pipeline NEGATIVE is fixable "
              f"with that setting.")
    else:
        verdict = "DG_CANNOT_ORTHOGONALISE_FOR_ATTRACTOR"
        print(f"  No DG setting -- not even {max(EXPANSIONS)}x expansion at "
              f"{min(SPARSITIES)*100:.1f}% sparsity, far past biological "
              f"values -- orthogonalises the substrate's concept "
              f"representations enough: the similarity floors at "
              f"{best_sim:.3f} and the attractor stays degenerate at every "
              f"setting. The fully-biologized pipeline NEGATIVE is genuine; "
              f"the substrate's concept representations are irreducibly "
              f"entangled under pattern separation, below the attractor's "
              f"orthogonality requirement.")

    out = {
        "seeds": SEEDS, "expansions": EXPANSIONS, "sparsities": SPARSITIES,
        "attractor_orthogonality_need": ATTRACTOR_ORTHOGONALITY_NEED,
        "grid": grid_out, "best_similarity": best_sim,
        "verdict": verdict,
    }
    with open("research/findings/raw/dg_orthogonality_sweep.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/dg_orthogonality_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
