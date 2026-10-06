"""Why the emotion-category organ learns its word meanings from labeling episodes, not from corpus co-occurrence
(packet s2-c4-build, PLAN.md section 3). Banked METHODS, measured, numpy only, ~1 min:

  A  leave-one-out label propagation of four category channels over the DR-2 learned co-occurrence graph (the exact
     construction that sources the production valence map: PPMI over hubs, cosine kNN, seed-clamped propagation),
     seeded from a category seed list; grid n_hub {500,1000,2000} x window {2,4,8} x knn {8,12,20} x hops {1,2} x
     {raw, per-channel gain}. Selection rule (written without reading the Stage-2 situations): the best macro LOO
     accuracy over seed words that are NOT Stage-2 words. Reported: the four situations' and the permuted arm's
     category margins after the gate's own two-exposure EMA (decay 0.4), i.e. what a perfect read-out of that map
     would name.
  B  conditioning on innate triggers in text: each affect word's co-occurrence (same / +-1 sentence) with the trigger
     words of each channel (organ TRIGGERS + common variants), per-channel base-rate gain; same report.

    python -m research.runners._emotion_category_corpus_probe --out research/findings/raw/_s2_c4_build/corpus_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._affect_distributional_tag_derisk import (WARRINER, load_stories, build_cooccurrence,  # noqa: E402
                                                                codes_from_cooccurrence, affinity_knn, propagate)
from research.runners.emotion_category_dev import SITUATIONS, SITUATION_AFFECT_WORDS, SITUATION_SWAP, situation_text  # noqa: E402

CATS = ("joy", "fear", "anger", "sadness")
SEEDS = {
    "joy": "happy joy laugh fun smile cheer glad delighted thrilled excited ecstatic overjoyed elated joyful cheerful "
           "pleased wonderful amazing fantastic awesome love play dance sing hug proud lucky",
    "fear": "scared afraid fear scary terrified nervous anxious worried monster ghost dark spider snake storm witch",
    "anger": "angry mad furious anger hate fight frustrated annoyed cruel mean nasty jealous",
    "sadness": "sad cry lonely lost miserable heartbroken devastated despair grief depressed hopeless disappointed sorry "
               "broken lose",
}
US = {
    "joy": "candy cake toy treat present prize party play game sweet birthday park presents toys treats games cookie",
    "fear": "monster ghost snake spider dark storm thunder danger fire witch dragon shadow lion tiger bear monsters",
    "anger": "took stole steal broke push pushed hit unfair tease teased bully grabbed snatched ruined rude mine",
    "sadness": "lost gone miss missed goodbye died die dead leave left away sick moved",
}
STAGE2_WORDS = set("happy joy joyful glad delighted pleased cheerful elated afraid scared fear fearful anxious frightened "
                   "terrified angry anger mad furious rage irritated annoyed sad sadness sorrow grief lonely unhappy "
                   "miserable down cry".split())
AFFECT = {w for w, (v, _a) in WARRINER.items() if abs(v - 5.0) >= 2.0}


def margins(vec):
    """The gate's sequence through a perfect read-out of `vec` (word -> 4-vector or None): per situation (intact, then
    permuted) the EMA's margin of the expected category over the best other one."""
    out = {}
    for swap in (False, True):
        ema = np.zeros(4)
        for c in CATS:
            vs = [vec(w) for w in re.findall(r"[a-z]+", situation_text(c, swap)) if w in AFFECT]
            vs = [v for v in vs if v is not None]
            a = np.mean(vs, 0) if vs else np.zeros(4)
            for _ in range(2):
                ema = 0.4 * ema + 0.6 * a
            want = CATS.index(SITUATION_SWAP[c] if swap else c)
            out[("permuted_" if swap else "") + c] = round(float(ema[want] - max(ema[k] for k in range(4) if k != want)), 4)
    return out


def method_a(stories):
    lab = {w: c for c, ws in SEEDS.items() for w in ws.split()}
    rows = []
    for n_hub in (500, 1000, 2000):
        for window in (2, 4, 8):
            vocab, C = build_cooccurrence(stories, n_hub, window, 5)
            codes = codes_from_cooccurrence(C)
            n, idx = len(vocab), {w: i for i, w in enumerate(vocab)}
            for knn in (8, 12, 20):
                P = affinity_knn(codes, knn)
                Y = np.zeros((n, 4)); seeded = np.zeros(n, bool)
                for w, c in lab.items():
                    if w in idx:
                        Y[idx[w], CATS.index(c)] = 1.0; seeded[idx[w]] = True
                for hops in (1, 2):
                    F = {}
                    for w in lab:
                        if w in idx:
                            m = seeded.copy(); m[idx[w]] = False
                            F[w] = np.array([propagate(P, m, Y[:, k], hops)[idx[w]] for k in range(4)])
                    allf = np.array([[propagate(P, seeded, Y[:, k], hops)[i] for k in range(4)]
                                     for i in range(n) if not seeded[i]])
                    gain = 1.0 / (allf.mean(0) + 1e-9)
                    for norm in ("raw", "gain"):
                        def vec(w, norm=norm, F=F, gain=gain):
                            if w not in F:
                                return np.eye(4)[CATS.index(lab[w])] if w in lab else None
                            f = F[w] * (gain if norm == "gain" else 1.0)
                            return f / (f.sum() + 1e-12)
                        per = {c: [] for c in CATS}
                        for w in F:
                            if w not in STAGE2_WORDS:
                                per[lab[w]].append(CATS[int(np.argmax(vec(w)))] == lab[w])
                        macro = float(np.mean([np.mean(v) for v in per.values()]))
                        mg = margins(vec)
                        rows.append({"n_hub": n_hub, "window": window, "knn": knn, "hops": hops, "norm": norm,
                                     "macro_loo_nonstage2": round(macro, 4), "margins": mg,
                                     "n_situations_right": sum(v > 0 for v in mg.values())})
    best = max(rows, key=lambda r: r["macro_loo_nonstage2"])
    return {"rows": rows, "selected_by_rule": best,
            "n_configs_all_8_right": sum(r["n_situations_right"] == 8 for r in rows),
            "max_min_margin": max(min(r["margins"].values()) for r in rows)}


def method_b(text_stories, radius):
    us = {w: c for c, ws in US.items() for w in ws.split()}
    cnt = defaultdict(lambda: np.zeros(4)); usfreq = np.zeros(4)
    for s in text_stories:
        sents = [re.findall(r"[a-z]+", x) for x in re.split(r"[.!?]", s.lower())]
        usc = [np.array([sum(1 for t in x if us.get(t) == c) for c in CATS], float) for x in sents]
        for v in usc:
            usfreq += v
        for i, x in enumerate(sents):
            aw = {w for w in x if w in AFFECT and w not in us}
            if aw:
                ctx = sum(usc[max(0, i - radius): i + radius + 1])
                for w in aw:
                    cnt[w] += ctx
    gain = usfreq.sum() / (usfreq + 1e-9)

    def vec(w):
        f = cnt[w] * gain
        return f / f.sum() if f.sum() > 0 else None
    words = {w: (None if vec(w) is None else [round(float(x), 3) for x in vec(w)])
             for c in CATS for w in SITUATION_AFFECT_WORDS[c]}
    return {"radius": radius, "margins": margins(vec), "situation_word_vectors": words}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    stories = load_stories(60000)
    with open(os.path.join(_REPO, "data", "corpus", "tinystories.txt"), encoding="utf-8", errors="ignore") as fh:
        raw = fh.read().split("<|endoftext|>")
    res = {"probe": "research/runners/_emotion_category_corpus_probe.py", "categories": CATS,
           "A_loo_label_propagation": method_a(stories),
           "B_trigger_conditioning": [method_b(raw, r) for r in (0, 1)]}
    a_sel = res["A_loo_label_propagation"]["selected_by_rule"]
    print("A: selected", {k: a_sel[k] for k in ("n_hub", "window", "knn", "hops", "norm", "macro_loo_nonstage2",
                                                "n_situations_right")}, "min margin", min(a_sel["margins"].values()))
    print("A: configs with all 8 right:", res["A_loo_label_propagation"]["n_configs_all_8_right"], "of",
          len(res["A_loo_label_propagation"]["rows"]), "; best min margin", res["A_loo_label_propagation"]["max_min_margin"])
    for b in res["B_trigger_conditioning"]:
        print("B radius %d:" % b["radius"], b["margins"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
