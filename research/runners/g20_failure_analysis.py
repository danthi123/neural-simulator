"""Cross-benchmark failure-correlation analysis (research, CPU-only).

A recurring failure pattern appeared across EVERY 320 benchmark
(functional-bridge / verb words underperform). This asks: is failure
concept-INTRINSIC (the same concepts fail across independent
benchmarks -> a structural property) or benchmark-specific noise?
And if intrinsic, which structural feature predicts it
(bridge / vocab-index / sparse-pattern overlap / orthogonal-drive
overlap)?

Pure analysis of already-committed JSON + the deterministic
generate_sparse_patterns / orthogonal_drive_pattern. No GPU, no
implementation -- this INFORMS the flagged recovery + any future
design; it does not build anything.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.runners.concept_pool_sparse_distributed import (
    generate_sparse_patterns,
)
from research.runners.g20_multibridge import read_vocab_file
from sim.text_embeddings import orthogonal_drive_pattern

V = "research/findings/raw/g11_bg"
NAMES = ["bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
         "bridgeD_spatial", "bridgeE_functional"]
SEED, NPOOL, KPAT, NLANG, NCUES, SPARS = 42, 2000, 100, 8192, 64, 0.007


def load_vocabs():
    return {n: read_vocab_file(f"{V}/g20_{n}_vocab64.txt") for n in NAMES}


def structural_features(vocabs):
    """Per (bridge, word): vocab idx, sparse-pattern max-overlap,
    orthogonal-drive max-overlap. All bridges share seed-42 patterns,
    so pattern/drive structure is indexed by vocab position."""
    pats = generate_sparse_patterns(NCUES, NPOOL, KPAT, SEED)
    S = [set(p) for p in pats]
    pat_maxov = [max(len(S[i] & S[j]) for j in range(NCUES) if j != i)
                 for i in range(NCUES)]
    D = [set(np.where(orthogonal_drive_pattern(
            cue_idx=i, n_cues=NCUES, n_neurons=NLANG,
            drive_max_pA=200.0, sparsity=SPARS)[0:NLANG] > 0)[0].tolist())
         for i in range(NCUES)]
    drv_maxov = [max((len(D[i] & D[j]) for j in range(NCUES) if j != i),
                     default=0) for i in range(NCUES)]
    feat = {}
    for n, vocab in vocabs.items():
        for idx, w in enumerate(vocab):
            if idx >= NCUES:
                break
            feat[(n, w)] = {"bridge": n, "idx": idx,
                            "pat_maxov": pat_maxov[idx],
                            "drv_maxov": drv_maxov[idx]}
    return feat


def main():
    vocabs = load_vocabs()
    word2bridge = {}
    for n, voc in vocabs.items():
        for w in voc:
            word2bridge.setdefault(w, n)
    feat = structural_features(vocabs)

    # Collect per-concept fail records: (target-side) the concept that
    # should have been retrieved but wasn't, per independent benchmark.
    target_fail = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # w->bench->[fail,total]

    xb = json.load(open(f"{V}/g20_xbridge_bench_320.json"))["rows"]
    for r in xb:  # B is the target retrieved via query of A
        b = r["b"]
        target_fail[b]["pair"][1] += 1
        if not r["genuine"]:
            target_fail[b]["pair"][0] += 1

    sb = json.load(open(f"{V}/g20_sentence_bench_320.json"))["rows"]
    for r in sb:  # verb + obj are targets
        for tw, key in ((r["verb"], "v_in_topk"),
                        (r["obj"], "o_in_topk")):
            target_fail[tw]["sentence"][1] += 1
            if not r[key]:
                target_fail[tw]["sentence"][0] += 1

    ib = json.load(open(f"{V}/g20_interference_bench_320.json"))["rows"]
    for r in ib:  # b is the target after load
        b = r["b"]
        target_fail[b]["interf"][1] += 1
        if not r["genuine"]:
            target_fail[b]["interf"][0] += 1

    # --- bridge-level failure rate ---
    bridge_fail = defaultdict(lambda: [0, 0])
    for w, benches in target_fail.items():
        br = word2bridge.get(w)
        if br is None:
            continue
        for bench, (f, t) in benches.items():
            bridge_fail[br][0] += f
            bridge_fail[br][1] += t
    print("=== target-failure rate by bridge (all 3 benchmarks) ===")
    for n in NAMES:
        f, t = bridge_fail[n]
        print(f"  {n:20s}: {f}/{t} = "
              f"{100*f/t:.1f}%" if t else f"  {n}: no data")

    # --- repeat offenders: fail in >= 2 independent benchmarks ---
    print("\n=== repeat offenders (target-fail in >=2 benchmarks) ===")
    repeat = []
    for w, benches in sorted(target_fail.items()):
        failed_in = [bn for bn, (f, t) in benches.items() if f > 0]
        if len(failed_in) >= 2:
            fe = feat.get((word2bridge.get(w), w), {})
            repeat.append(w)
            print(f"  {w:10s} [{word2bridge.get(w,'?'):18s}] "
                  f"fails={failed_in} idx={fe.get('idx','?')} "
                  f"pat_ov={fe.get('pat_maxov','?')} "
                  f"drv_ov={fe.get('drv_maxov','?')}")
    if not repeat:
        print("  NONE -- failures are benchmark-specific (noise), "
              "not concept-intrinsic.")

    # --- structural correlation: failed-once vs never-failed ---
    failed_any, never = [], []
    for (br, w), fe in feat.items():
        bn = target_fail.get(w)
        if not bn:
            continue
        any_fail = any(f > 0 for f, t in bn.values())
        (failed_any if any_fail else never).append(fe)

    def mean(lst, k):
        v = [x[k] for x in lst]
        return sum(v) / len(v) if v else 0.0

    print("\n=== structural predictor (failed-any vs never-failed) ===")
    print(f"  n failed-any={len(failed_any)}  n never={len(never)}")
    for k in ("pat_maxov", "drv_maxov", "idx"):
        print(f"  mean {k}: failed={mean(failed_any,k):.2f}  "
              f"never={mean(never,k):.2f}")

    # --- verdict ---
    fb = bridge_fail["bridgeE_functional"]
    vb = bridge_fail["bridgeB_verbs"]
    cleanest = min((n for n in NAMES if bridge_fail[n][1]),
                   key=lambda n: bridge_fail[n][0] / bridge_fail[n][1])
    print("\n=== VERDICT ===")
    print(f"  cleanest bridge: {cleanest} "
          f"({bridge_fail[cleanest][0]}/{bridge_fail[cleanest][1]})")
    print(f"  functional: {fb[0]}/{fb[1]} | verbs: {vb[0]}/{vb[1]}")
    pa = mean(failed_any, "pat_maxov")
    pn = mean(never, "pat_maxov")
    print(f"  pattern-overlap predictive? failed {pa:.2f} vs "
          f"never {pn:.2f} -> "
          f"{'YES' if pa > pn + 0.5 else 'NO (not the driver)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
