"""Design evidence for the closed-class fix (dev seed 7 only): WHERE the single-offset referent lexicon's noun drive
comes from, and how well joint (-1,+1) frames would separate nouns from closed-class words and adjectives.

Two host analyses, both DESIGN evidence (they choose a mechanism; they are not a result about any mechanism):
  1. SIDE DECOMPOSITION of the trained lexicon (`lexicon_spiking_frame_category.get_lexicon(seed=7)`): for a word's
     K sampled occurrences, the learned CN-minus-CX mean weight summed over its active afferents, split into the
     LEFT offsets (-2,-1) and the RIGHT offsets (+1,+2). Uses the circuit's installed weights, not its spikes.
  2. FRAME PROXY: curriculum-only evidence tables built from the SAME seed words the teacher uses (38 nouns, 37
     non-nouns), over the same 100 context words: per joint frame (word at -1, word at +1), and per single offset.
     A test word's score is the mean over its K sampled occurrences. Reported: the fraction of held-out fixture
     NOUN / VERB / ADJ words and of closed-class words with a positive score under each code.
Closed-class list here = the animacy module's `_STOP` minus the D6 `_STOP` (what the D6 learned path can be asked).

  bash tools/mem_ok.sh 2 && bash tools/memcap.sh 2 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_frame_proxy --corpus /path/to/tinystories.txt \
      --json research/findings/raw/_lexicon_closed_class/frame_proxy_s7.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEV_SEED = 7


def run(corpus, seed=DEV_SEED):
    from research.runners import lexicon_spiking_frame_category as L
    from research.runners import d6_multiref_wm_production_organ as D6
    from research.runners._comprehension_learned_animacy_cue_derisk import _STOP as ANIM_STOP
    from research.runners._lexicon_closed_class_parse_diag import FIXTURE, _sha256

    for k in ("BRAIN_LEARNED_REFERENT_LEXICON", "BRAIN_LEARNED_REFERENT_LESION", "BRAIN_LEARNED_REFERENT_JUNCTION"):
        os.environ.pop(k, None)
    L._LEXICON = None
    lex = L.get_lexicon(seed=seed, corpus_path=corpus)
    env, C = lex.env, lex.env.C
    fx = json.load(open(FIXTURE))["pos"]
    excl = set(D6._REFERENT_NOUNS) | set(L.HAND_NOUN_SEEDS) | set(L.NONNOUN_SEEDS)
    closed = sorted(w for w in (set(ANIM_STOP) - set(D6._STOP)) if w in env.pos)

    # 1. side decomposition of the trained single-offset lexicon
    Wd = lex.W.reshape(env.n_frame, 2, L.N_CAT).mean(axis=2)
    marg = Wd[:, 0] - Wd[:, 1]

    def sides(w):
        occ = env.occurrences(w, lex.k_occ, lex.seed)
        x = np.zeros(env.n_frame)
        for f in occ:
            x[f] += 1
        per = [float(x[pj * C:(pj + 1) * C] @ marg[pj * C:(pj + 1) * C]) / len(occ) for pj in range(4)]
        return per[0] + per[1], per[2] + per[3]

    # 2. frame proxy (curriculum-only evidence)
    tc = env.tok_ctx
    rng_all = np.random.default_rng(seed)

    def neigh(i):
        g = lambda j: int(tc[j]) if 0 <= j < len(tc) else -1  # noqa: E731
        return g(i - 2), g(i - 1), g(i + 1), g(i + 2)

    def table(words):
        pair, single, n = collections.Counter(), collections.Counter(), 0
        for w in words:
            p = env.pos[w]
            pick = rng_all.choice(p, size=min(len(p), 256), replace=False)
            for i in pick:
                l2, l1, r1, r2 = neigh(int(i))
                n += 1
                if l1 >= 0 and r1 >= 0:
                    pair[(l1, r1)] += 1
                for off, c in ((-2, l2), (-1, l1), (1, r1), (2, r2)):
                    if c >= 0:
                        single[(off, c)] += 1
        return pair, single, n

    nouns = [w for w in L.HAND_NOUN_SEEDS if w in env.pos]
    non = [w for w in L.NONNOUN_SEEDS if w in env.pos]
    pN, sN, nN = table(nouns)
    pX, sX, nX = table(non)

    def proxy(w):
        rng = np.random.default_rng((L._stable_int(w) * 1000003 + seed) & 0x7FFFFFFF)
        p = env.pos[w]
        pick = rng.choice(p, size=lex.k_occ, replace=len(p) < lex.k_occ)
        sp = ss = 0.0
        for i in pick:
            l2, l1, r1, r2 = neigh(int(i))
            if l1 >= 0 and r1 >= 0:
                sp += pN[(l1, r1)] / nN - pX[(l1, r1)] / nX
            for off, c in ((-2, l2), (-1, l1), (1, r1), (2, r2)):
                if c >= 0:
                    ss += sN[(off, c)] / nN - sX[(off, c)] / nX
        return sp / len(pick), ss / len(pick)

    groups = collections.defaultdict(list)
    for w, tag in fx.items():
        if w not in excl and w in env.pos:
            groups[tag].append(w)
    groups["CLOSED"] = closed
    summary = {}
    for tag, ws in sorted(groups.items()):
        pr = np.array([proxy(w) for w in ws])
        sd = np.array([sides(w) for w in ws])
        summary[tag] = {"n": len(ws), "frame_positive_frac": float((pr[:, 0] > 0).mean()),
                        "single_offset_positive_frac": float((pr[:, 1] > 0).mean()),
                        "learned_left_margin_mean": float(sd[:, 0].mean()),
                        "learned_right_margin_mean": float(sd[:, 1].mean())}
    probe = ["when", "what", "who", "before", "most", "today", "wonderful", "crazy", "east", "owl", "marble",
             "basket", "room", "leaves", "anne", "sally"]
    words = {w: {"frame": proxy(w)[0], "single": proxy(w)[1], "left": sides(w)[0], "right": sides(w)[1],
                 "decision": lex.decide(w)[0][0]} for w in probe if w in env.pos}
    out = {"seed": seed, "summary": summary, "probe_words": words, "corpus_path": corpus}
    out["corpus_sha256"], out["corpus_bytes"] = _sha256(corpus)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    r = run(a.corpus)
    print(json.dumps(r["summary"], indent=1))
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(r, open(dst, "w"), indent=1, default=str)
        print("wrote", dst)


if __name__ == "__main__":
    main()
