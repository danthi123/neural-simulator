"""Substantiate the within-category-NEGATIVE finding's central claim DIRECTLY on the 320 stream codes:
the cortex codes' category structure is carried by a LEARNED read-out direction, NOT by raw nearest-neighbor
proximity.

The within-category-error probe (2026-06-17-within-category-error-signature-NEGATIVE.md) found the conversational
binder's recall errors are only weakly within-category (~5% vs 2.2% chance), and crucially raw-no-bind ~ bound —
so the structure is not in the raw nearest-neighbor geometry. It cited the perception->concept arc (held-out
cat-acc 0.92 via a LEARNED read-out) as where generalization lives. This runner closes that loop ON THE SAME 320
codes, not by analogy: compare

  * RAW kNN category accuracy  — leave-one-out nearest neighbour by cosine: does a code's nearest other code share
    its category? (the raw-proximity signal the binder cleanup actually uses)
  * LEARNED linear read-out cat-acc — a ridge one-hot category read-out, k-fold cross-validated (a downstream
    region learning to read the category direction)
  * DERANGED control — the learned read-out on shuffled category labels must collapse to chance (proves it learns
    REAL categories, not folds/overfitting).

Chance = 1/40 = 2.5%. PREDICTION (substantiates the negative): LEARNED >> RAW, and DERANGED ~ chance. That is:
the category structure IS in the codes, but a LEARNED projection extracts it where raw distance cannot.

Run (CPU; closed-form ridge, no sklearn):
  SIM_BACKEND=numpy python -m research.runners._genfrontier_learned_vs_raw_category_readout --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories


def _unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def raw_knn_cat_acc(codes, cat):
    """Leave-one-out nearest-neighbour (cosine) category accuracy over all 320 codes."""
    X = _unit(codes)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argmax(S, axis=1)
    return float(np.mean(cat[nn] == cat))


def learned_cat_acc(codes, cat, n_cat, seed, folds=5, ridge=1.0):
    """k-fold CV accuracy of a closed-form ridge one-hot category read-out."""
    rng = np.random.default_rng(seed * 13 + 7)
    Nc = codes.shape[0]
    X = _unit(codes)
    Y = np.eye(n_cat)[cat]
    order = rng.permutation(Nc)
    fold = np.array_split(order, folds)
    correct = 0
    for f in range(folds):
        te = fold[f]
        tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        Xt, Yt = X[tr], Y[tr]
        W = np.linalg.solve(Xt.T @ Xt + ridge * np.eye(X.shape[1]), Xt.T @ Yt)  # (D, n_cat)
        pred = np.argmax(X[te] @ W, axis=1)
        correct += int(np.sum(pred == cat[te]))
    return correct / Nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--readout", choices=["neural", "host"], default="host")
    ap.add_argument("--out", default="research/findings/raw/_genfrontier_learned_vs_raw_category_readout.json")
    a = ap.parse_args()

    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    cat = np.asarray(cat_ids)
    n_cat = len(set(cat_ids))
    chance = 1.0 / n_cat
    suffix = "neural_seed" if a.readout == "neural" else "seed"

    print(f"[learned-vs-raw category read-out] {n_cat} categories | chance {100*chance:.1f}%\n"
          "  PREDICTION (substantiates the within-category NEGATIVE): LEARNED >> RAW, DERANGED ~ chance.\n",
          flush=True)
    rows = []
    for seed in a.seeds:
        cpath = os.path.join(_REPO, "research", "findings", "raw",
                             f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
        if not os.path.exists(cpath):
            print(f"  [seed {seed}] SKIP — no codes at {cpath}", flush=True)
            continue
        codes = np.load(cpath)
        raw = raw_knn_cat_acc(codes, cat)
        learned = learned_cat_acc(codes, cat, n_cat, seed)
        rng = np.random.default_rng(seed * 991 + 3)
        derange = cat.copy(); rng.shuffle(derange)
        deranged = learned_cat_acc(codes, derange, n_cat, seed)
        rows.append({"seed": seed, "raw_knn": raw, "learned": learned, "deranged": deranged})
        print(f"  [seed {seed}] RAW kNN {100*raw:5.1f}% | LEARNED {100*learned:5.1f}% | "
              f"DERANGED {100*deranged:4.1f}% (chance {100*chance:.1f}%)", flush=True)

    if rows:
        mr = float(np.mean([r["raw_knn"] for r in rows]))
        ml = float(np.mean([r["learned"] for r in rows]))
        md = float(np.mean([r["deranged"] for r in rows]))
        # substantiated if learned clearly beats raw AND beats deranged (which should sit at chance).
        ok = (ml >= 2.0 * max(mr, chance)) and (ml >= 3.0 * max(md, chance)) and (md <= 2.0 * chance)
        verdict = "SUBSTANTIATED" if ok else "NOT-SUBSTANTIATED"
    else:
        mr = ml = md = float("nan"); verdict = "NO-CODES"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"chance": chance, "raw_mean": mr, "learned_mean": ml, "deranged_mean": md,
                   "verdict": verdict, "rows": rows}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if verdict == "SUBSTANTIATED":
        print(f"  SUBSTANTIATED: LEARNED read-out {100*ml:.1f}% >> RAW kNN {100*mr:.1f}% (DERANGED {100*md:.1f}% "
              f"~ chance {100*chance:.1f}%). The 320 stream codes' category structure IS present but is carried by "
              "a LEARNED read-out direction, NOT raw nearest-neighbour proximity — exactly why the conversational "
              "binder's raw-cleanup errors are near-random (the within-category NEGATIVE) yet the perception→concept "
              "arc generalizes (cat-acc 0.92 via a learned read-out). The two findings are one mechanism.", flush=True)
    elif verdict == "NOT-SUBSTANTIATED":
        print(f"  NOT-SUBSTANTIATED: LEARNED {100*ml:.1f}% vs RAW {100*mr:.1f}% vs DERANGED {100*md:.1f}% — the "
              "learned-read-out advantage is not clear-cut on these codes; revisit the mechanistic claim.", flush=True)
    else:
        print("  NO CODES — run the 320 stream cortex first.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
