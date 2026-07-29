#!/usr/bin/env python3
"""Sparse-pool CAPACITY sweep: the re-route's decisive gate. Writes INCREMENTALLY (untended-safe).

THE TWO QUESTIONS, and why they must be asked together:
 1. **What is the shared pool's REAL capacity?** The banked "~32 concepts per sparse bridge" figure
    (2026-05-15 G.20 SHIPPED, 5 bridges x 32 = 160) is NOT a measured pool limit — it is an identity of
    the cue encoder. `orthogonal_drive_pattern` (sim/text_embeddings.py:194-196) lays each cue in a
    NON-OVERLAPPING band with `n_active = sparsity*N` and `stride = N//n_cues`, so its guard reduces to
    `n_cues <= 1/sparsity`, INDEPENDENT of layer size — exactly 33 at the default sparsity=0.03. Growing
    the layer cannot help (verified: n_active 246->614->1229 as n_lang_input 8192->20480->40960). The
    encoder refused before the pool was ever asked. Pass `--sparsity <= 1/n` to get past it.
 2. **Do COMPOSED FACTS store, not just concepts?** The banked result stores INDEPENDENT concepts, which
    overlap only by chance (~5 of 100 neurons). Consolidation needs facts that SHARE constituents, which
    overlap structurally (measured 27.9 mean / 69 max at n=32) — 5.7x more. Storing 64 unrelated concepts
    says nothing about storing 64 facts that share their words. This is the actual open question.

ANTI-CHEAT / READING RULES baked in:
 * Lower sparsity means fewer active input neurons per cue, hence WEAKER drive. If discrimination falls
   at large n, that must be checked against the drive change BEFORE calling it a capacity limit — the
   exact misattribution (instrument property read as substrate property) that produced finding (1).
   Each row therefore records `sparsity` and `n_active` alongside the score.
 * composed-vs-independent is run at MATCHED n and MATCHED sparsity, so the only difference is overlap.
 * Results append to the JSON after EVERY config, so an interrupted run still yields everything finished.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="research/findings/raw/sparse_capacity/sweep.json")
    ap.add_argument("--seeds", type=str, default="42,43")
    ap.add_argument("--n-list", type=str, default="32,64,128,256")
    ap.add_argument("--budget-s", type=float, default=9000.0)
    ap.add_argument("--train-events", type=int, default=200)
    args = ap.parse_args()

    from research.runners.concept_pool_sparse_distributed import (
        build_sparse_pool_bridge, generate_sparse_patterns, apply_sparse_topographic_prior)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    rows, t0 = [], time.time()

    def flush(status):
        json.dump({"status": status, "elapsed_s": round(time.time() - t0, 1),
                   "rows": rows, "argv": sys.argv}, open(args.out, "w"), indent=1)

    flush("started")
    # Priority order: the composed-vs-independent CONTRAST at every n comes before extra seeds, so an
    # interrupted run still answers the actual question rather than over-replicating one condition.
    plan = [(n, cv, s) for s in seeds for n in ns for cv in (0, max(8, int(round((6 * n) ** (1 / 3.0)) + 3)))]
    for (n, cv, seed) in plan:
        if time.time() - t0 > args.budget_s:
            flush("budget_exhausted")
            print("BUDGET EXHAUSTED after %d configs" % len(rows), flush=True)
            break
        sparsity = min(0.03, 0.8 / float(n))          # honour n_cues <= 1/sparsity with headroom
        row = dict(n=n, composed_vocab=cv, seed=seed, sparsity=round(sparsity, 5))
        try:
            pats = generate_sparse_patterns(n_concepts=n, n_pool=2000, pattern_size=100,
                                            seed=seed, composed_vocab=cv)
            # record the ACTUAL overlap so composed-vs-independent is verified, not assumed
            import itertools
            import numpy as np
            P = [set(p) for p in pats]
            ov = [len(a & b) for a, b in itertools.islice(itertools.combinations(P, 2), 4000)]
            row["mean_overlap"] = round(float(np.mean(ov)), 2)
            row["max_overlap"] = int(max(ov))
            b = build_sparse_pool_bridge(seed=seed, n_lang_input=8192, n_shared_pool=2000,
                                         n_shared_fs=300, verbose=False)
            res = apply_sparse_topographic_prior(
                b, n_concepts=n, n_lang_input=8192, sparse_patterns=pats, sparsity=sparsity)
            row.update({k: v for k, v in (res or {}).items() if isinstance(v, (int, float, str, bool))})
            row["ok"] = True
        except Exception as e:
            row["ok"] = False
            row["error"] = "%s: %s" % (type(e).__name__, e)
            row["trace"] = traceback.format_exc()[-800:]
        rows.append(row)
        flush("running")
        print("done n=%d composed=%d seed=%d -> %s" % (n, cv, seed, row.get("error", "ok")), flush=True)
    else:
        flush("complete")
    flush(rows and "complete" or "empty")
    print("WROTE %s (%d rows)" % (args.out, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
