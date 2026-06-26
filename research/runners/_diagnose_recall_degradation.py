#!/usr/bin/env python
"""Offline diagnosis of the 1454-concept recall crash (0.917@8K -> 0.208@150K).

NO retraining, NO GPU: load the SAVED M (learned weights) + codes from
bridges/firstchat/brain1454_seed42.npz, re-derive the grounded phasor codes under
several NORMALIZATIONS, and re-measure recall+moat on each. If a different
normalization of the SAME weights recovers recall, the over-training problem is a
read-out/normalization issue (instant fix); if NO normalization recovers it, the
weights M themselves degraded (-> fewer windows / higher n_per / multi-bridge).
"""
import sys
import numpy as np

from research.runners._curriculum_step1_320_real_corpus import (
    double_center, measure_recall_and_moat)

# NOTE: this .npz is OUR OWN artifact (written by the curriculum runner's --save-codes this session);
# allow_pickle=True is required only for the vocab/cat_names dtype=object string arrays. Trusted source.
NPZ = "bridges/firstchat/brain1454_seed42.npz"


def ground(code, proj, vocab):
    g = {}
    for i, w in enumerate(vocab):
        z = proj @ code[i].astype(np.complex128)
        g[w] = (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)
    return g


def main():
    d = np.load(NPZ, allow_pickle=True)
    vocab = list(d["vocab"]); cat_ids = d["cat_ids"]; cat_names = list(d["cat_names"])
    M = d["M"].astype(np.float64); D = int(d["D"]); seed = int(d["seed"])
    n_hub = M.shape[1]
    grounded_saved = {vocab[i]: d["grounded"][i] for i in range(len(vocab))}

    # weight-saturation signature: fraction of M near its max (frequent pairs pinned at the cap?)
    mmax = M.max()
    frac_hi = float((M > 0.9 * mmax).mean())
    frac_nz = float((M > 1e-6).mean())
    print(f"[M stats] max {mmax:.3f} mean {M.mean():.4f} | frac>0.9max {frac_hi:.4f} | "
          f"frac nonzero {frac_nz:.4f}", flush=True)

    # 1) reproduce the saved-brain recall on the saved grounded (sanity: should ~match 0.208)
    r0 = measure_recall_and_moat(grounded_saved, vocab, cat_ids, cat_names, seed, 24, D)
    print(f"[SAVED grounded]            recall {r0['recall']:.3f}  moat-fa {r0['false_accept']}", flush=True)

    # a fresh, valid random projection (recall depends on code STRUCTURE, preserved by any random proj;
    # used identically across all norms so the comparison isolates the normalization)
    rng = np.random.RandomState(12345)
    proj = (rng.randn(D, n_hub) + 1j * rng.randn(D, n_hub)) / np.sqrt(n_hub)

    # 2) control + 3) alternative normalizations of the SAME saved weights M
    norms = {
        "saved: double_center(log1p(M*100))": double_center(np.log1p(M * 100.0)),
        "log1p(M) (no *100)":                 double_center(np.log1p(M)),
        "log1p(M*10)":                        double_center(np.log1p(M * 10.0)),
        "M raw (double_center)":              double_center(M.copy()),
        "sqrt(M)":                            double_center(np.sqrt(M)),
        "row-L2-normalized M":                double_center(M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)),
    }
    for name, c in norms.items():
        g = ground(c, proj, vocab)
        r = measure_recall_and_moat(g, vocab, cat_ids, cat_names, seed, 24, D)
        print(f"[{name:36s}] recall {r['recall']:.3f}  moat-fa {r['false_accept']}", flush=True)

    print("\nVERDICT: if any alt-norm recall >> 0.208 with moat-fa 0 -> normalization fix (re-derive+save, "
          "no retrain). If all ~0.208 -> M degraded (weights), needs fewer windows / higher n_per / multi-bridge.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
