"""Aggregate the grow-to-context (corpus-structured sparse pool) vocab-scale sweep: the scaling table (HTM + synapse
fraction vs vocab), the window=1 vs window=8 comparison, and the anti-cheat collapse (lesion/permute). Reads all
research/findings/raw/_htmsparse/corpus_*.json."""
import json
import glob
import os
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.dirname(__file__))


def load(pat):
    rows = []
    for f in sorted(glob.glob(os.path.join(D, pat))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        sp = d.get("sparse", {})
        rows.append({"file": os.path.basename(f), "n_subj": d.get("n_subj"), "vocab": d.get("vocab"),
                     "window": d.get("window"), "htm": sp.get("htm"), "syn": sp.get("synapses"),
                     "dense": d.get("dense", {}).get("synapses"), "ratio": d.get("synapse_ratio"),
                     "best_ngram": d.get("best_ngram"), "GO": d.get("GO")})
    return rows


def table(rows, label):
    by_n = defaultdict(list)
    for r in rows:
        if r["htm"] is not None:
            by_n[r["n_subj"]].append(r)
    if not by_n:
        print(f"\n[{label}] (no results yet)")
        return
    print(f"\n=== {label} ===")
    print(f"{'n_subj':>6} {'vocab':>6} {'seeds':>5} {'HTM(mean)':>10} {'n-gram':>7} {'sparse_syn':>11} "
          f"{'dense_syn':>12} {'ratio':>8}")
    for n in sorted(by_n):
        rs = by_n[n]
        htm = np.mean([r["htm"] for r in rs])
        ng = np.mean([r["best_ngram"] for r in rs if r["best_ngram"] is not None])
        syn = int(np.mean([r["syn"] for r in rs if r["syn"]]))
        dense = int(np.mean([r["dense"] for r in rs if r["dense"]]))
        ratio = np.mean([r["ratio"] for r in rs if r["ratio"] is not None])
        print(f"{n:>6} {rs[0]['vocab']:>6} {len(rs):>5} {htm:>10.3f} {ng:>7.3f} {syn:>11} {dense:>12} {ratio:>8.4f}")


def controls(pat, label, expect):
    rows = load(pat)
    if not rows:
        print(f"\n[{label}] (no results yet)")
        return
    htm = np.mean([r["htm"] for r in rows if r["htm"] is not None])
    ng = np.mean([r["best_ngram"] for r in rows if r["best_ngram"] is not None])
    print(f"\n=== {label} ({len(rows)} seeds) ===  HTM {htm:.3f} vs n-gram floor {ng:.3f}  [{expect}]")


if __name__ == "__main__":
    table(load("corpus_n*_s*.json"), "window=8 scaling (batch 1)")
    table([r for r in load("corpus_w1_n*_s*.json") if "lesion" not in r["file"]
           and "permute" not in r["file"] and "parity" not in r["file"]], "window=1 scaling (batch 2, CONFOUND-FREE)")
    table(load("corpus_w1_parity_n*_s*.json"), "window=1 dense-parity (measured corpus==dense)")
    controls("corpus_w1_lesion_n8_s*.json", "window=1 dAP-LESION anti-cheat", "should COLLAPSE to ~n-gram floor")
    controls("corpus_w1_permute_n8_s*.json", "window=1 PERMUTE anti-cheat", "should COLLAPSE to ~n-gram floor")
