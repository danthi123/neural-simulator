"""L1 scale-capacity axis: does the online nonneg similarity-matching rule keep extracting structure as the
number of CATEGORIES grows (64 -> 512 concepts)? Online LOCAL rules can degrade with scale (the lateral has
to separate more components). The real build targets ~2048 concepts, so this is the last cheap axis before
the build (synthetic capacity only -- real-data-noise-at-scale remains a build concern).

Synthetic (build_concept_hub_counts) at a HARDER calibration (weaker per-category signal -> host ceiling in
the realistic ~+0.5 range, NOT the easy +0.95) so the test is not trivially easy. Scale n_cat in {8,16,32}
(per_cat=8 -> 64/128/256 concepts). The KEY metric is the extraction FRACTION (learner / offline PCA): if
it stays high as categories grow, the rule has capacity headroom; if it collapses, scale is a wall to flag
BEFORE the weeks-commit. nonneg + centered (the brain-correct learner). CPU/numpy, compute-light. NO sim/ edits.

Run: python -u -m research.runners._l1_scale_capacity_check --scales 8,16,32
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import ppmi_matrix, pca_lowrank_sim  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners._l1_centered_online_pca_probe import center_cols  # noqa: E402
from research.runners._l1_nonneg_simmatch_check import simmatch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", default="8,16,32", help="n_cat values (per_cat=8 -> 64/128/256 concepts)")
    p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=150)
    # HARDER calibration: weak per-category signal, more shared/background noise -> realistic ~+0.5 host
    p.add_argument("--n-common", type=int, default=300); p.add_argument("--n-sig-per-cat", type=int, default=8)
    p.add_argument("--lam-common", type=float, default=40.0); p.add_argument("--lam-sig", type=float, default=2.2)
    p.add_argument("--lam-bg", type=float, default=0.5)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--out", default="research/findings/raw/_l1_scale_capacity_check.json")
    args = p.parse_args()
    scales = [int(s) for s in args.scales.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[L1 scale-capacity] harder calibration (lam_sig={args.lam_sig}, n_common={args.n_common}); "
          f"nonneg+centered online simmatch", flush=True)

    results, fracs = [], []
    for n_cat in scales:
        Nc = n_cat * args.per_cat
        k = n_cat + 24
        ps, gens, offs, hosts, perms = [], [], [], [], []
        for s in seeds:
            C, labels, S_true, _ = build_concept_hub_counts(
                n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
                args.lam_common, args.lam_sig, args.lam_bg, s)
            Xc = center_cols(ppmi_matrix(C, args.host_alpha))
            offline = _pearson_vs_Strue(pca_lowrank_sim(ppmi_matrix(C, args.host_alpha), k), S_true)
            host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=args.host_alpha)
            host_p, _, _, _ = score(host_sim, labels)
            codes = simmatch(Xc, k, args.epochs, 0.010, 0.030, 30, s, nonneg=True, spike_output=False)
            pe = _pearson_vs_Strue(_cos_sim(codes), S_true)
            g, ch = heldout_generalization(codes, labels)
            rng = np.random.RandomState(s * 2718281 + 1); perm = rng.permutation(labels)
            S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
            ps.append(pe); gens.append(g); offs.append(offline); hosts.append(host_p)
            perms.append(_pearson_vs_Strue(_cos_sim(codes), S_perm))
        pm, om, hm, gm, prm = (float(np.mean(ps)), float(np.mean(offs)), float(np.mean(hosts)),
                               float(np.mean(gens)), float(np.mean(perms)))
        frac = pm / om if om > 1e-9 else 0.0
        fracs.append(frac)
        rec = {"n_cat": n_cat, "n_concepts": Nc, "k": k, "learner_pearson": pm, "learner_seeds": ps,
               "offline_pca": om, "host_ppmi_svd": hm, "extraction_fraction": frac, "gen_mean": gm,
               "perm_mean": prm, "chance": ch}
        results.append(rec)
        print(f"  [n_cat={n_cat:3d} ({Nc:4d} concepts, k={k})] learner={pm:+.3f}  offline={om:+.3f}  "
              f"host={hm:+.3f}  frac={frac:.0%}  gen={gm:.3f}  perm={prm:+.3f}", flush=True)

    frac_drop = fracs[0] - fracs[-1]
    min_frac = min(fracs)
    clean = all(abs(r["perm_mean"]) <= 0.15 for r in results)
    if min_frac >= 0.70 and frac_drop <= 0.15 and clean:
        verdict = "SCALE_CAPACITY_HOLDS_GO"
        why = (f"the extraction fraction stays high as categories grow ({[f'{f:.0%}' for f in fracs]}, min "
               f"{min_frac:.0%}, drop {frac_drop:+.0%}) -> the online nonneg rule has capacity headroom to "
               f"scale; no scale wall at this range. With all FOUR axes (rule, input-spiking, learning-nonneg, "
               f"scale) GO at the smoke level, the L1 spiking similarity-matching cortex is comprehensively "
               f"de-risked -- only the bridge assembly + real-data-noise-at-2048 remain the build itself.")
    elif min_frac >= 0.50:
        verdict = "SCALE_CAPACITY_PARTIAL"
        why = (f"the extraction fraction degrades with scale ({[f'{f:.0%}' for f in fracs]}, drop "
               f"{frac_drop:+.0%}) but stays above half -> a soft scale cost; the build should budget more "
               f"epochs / larger k at 2048, and watch real-data-noise-at-scale.")
    else:
        verdict = "SCALE_CAPACITY_DEGRADES"
        why = (f"the extraction fraction collapses with scale ({[f'{f:.0%}' for f in fracs]}) -> the online "
               f"local rule loses capacity as categories grow -> a scale wall to address (deeper architecture / "
               f"hierarchical pools) BEFORE the 2048 commit.")
    print(f"\n{'='*92}\n  SCALE VERDICT: {verdict}\n  {why}\n{'='*92}", flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "fractions": fracs, "results": results, "seeds": seeds}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
