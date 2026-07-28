"""EMERGENT-POS open corpus-fact mining: mine SVO facts from TinyStories WITHOUT a hand-curated noun list,
using an EMERGENT part-of-speech filter (EMERGE-62 closed-class + EMERGE-73 attributive-adjective signals)
to keep noun-VERB-noun and DROP the adjective-object noise ("tim saw big"). Broadens what the brain learns
from experience beyond the curated-noun set.

The raw open mining is noisy (objects land on adjectives; subjects are names). Two emergent signals clean it:
  * CLOSED class (EMERGE-62): high running-frequency + high context-coverage words (the/a/was/...) -> not content.
  * ATTRIBUTIVE-ADJECTIVE (EMERGE-73): attribscore[w] = fraction of w's occurrences in the DET _ NOUN slot
    (preceded by a closed word AND followed by a content word) -> high for adjectives (big/little), low for nouns.
A word is a mineable NOUN iff it is open-class AND low-attribscore. Mine noun-VERB-noun with this filter and
compare the fact quality (fraction of facts whose object is a real noun, not an adjective) to the raw open mining
+ the curated-noun baseline. numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from collections import Counter

from research.runners._emergent_vocab_breadth_scale_derisk import discover_vocab
from research.runners._realcorpus_cancellation_derisk import _ANIMALS
from research.runners._realcorpus_learn_corpus_facts_derisk import VERBS, DET, NOUNS_EXTRA
from research.runners.corpus_stream import load_token_stream_multi

# a small KNOWN-adjective set for VALIDATION ONLY (measuring how many mined objects are adjectives) -- NOT used to filter.
_KNOWN_ADJ = {"big", "little", "small", "good", "bad", "happy", "sad", "old", "new", "red", "blue", "hot",
              "cold", "long", "tall", "nice", "pretty", "scared", "loud", "soft", "hard", "fast", "slow", "funny"}


def _closed_and_attrib(stories, vocab):
    """EMERGE-62 closed-class (freq x coverage) + EMERGE-73 attributive-adjective score, from the corpus stream."""
    vset = set(vocab)
    freq = Counter(); cover = {w: set() for w in vocab}
    attr_num = Counter(); attr_den = Counter()
    for si, st in enumerate(stories):
        for i, w in enumerate(st):
            if w in vset:
                freq[w] += 1; cover[w].add(si % 200)
                attr_den[w] += 1
                prev_closed = i > 0 and st[i - 1] in DET
                next_content = i + 1 < len(st) and st[i + 1] in vset and st[i + 1] not in DET
                if prev_closed and next_content:
                    attr_num[w] += 1                       # w in the DET _ NOUN slot (attributive position)
    N = sum(freq.values()) + 1e-9
    # closed-class: high running-frequency AND high context-coverage (EMERGE-62 Goldilocks)
    closed = set()
    for w in vocab:
        rf = freq[w] / N
        cov = len(cover[w]) / 200.0
        if rf > 0.004 and cov > 0.5:
            closed.add(w)
    attrib = {w: (attr_num[w] / attr_den[w] if attr_den[w] >= 20 else 0.0) for w in vocab}
    return closed, attrib


def _mine(toks, is_noun, verbs):
    facts = Counter()
    n = len(toks)
    i = 0
    while i < n - 4:
        if is_noun(toks[i]) and toks[i + 1] in verbs:
            j = i + 2
            while j < n and toks[j] in DET:
                j += 1
            if j < n and is_noun(toks[j]):
                facts[(toks[i], toks[i + 1], toks[j])] += 1
        i += 1
    return facts


def run_seed(seed, stories, K, top_n=40, attrib_thresh=0.45):
    vocab, gfreq = discover_vocab(stories, K)
    vset = set(vocab)
    verbs = [v for v in VERBS if v in vset]
    toks = [t for st in stories for t in st]
    closed, attrib = _closed_and_attrib(stories, vocab)

    # EMERGENT noun: open-class (not closed, not a verb) AND low attributive-adjective score
    def emergent_noun(w):
        return w in vset and w not in closed and w not in verbs and attrib.get(w, 0.0) < attrib_thresh
    def raw_open(w):
        return w in vset and w not in verbs                      # raw: any non-verb content word (noisy)
    curated = (_ANIMALS | NOUNS_EXTRA) & vset
    def curated_noun(w):
        return w in curated

    def obj_adj_frac(facts):                                     # fraction of mined objects that are KNOWN adjectives (noise)
        objs = [o for (s, v, o), c in facts.most_common(top_n)]
        return float(np.mean([o in _KNOWN_ADJ for o in objs])) if objs else 0.0

    fe = _mine(toks, emergent_noun, verbs)
    fr = _mine(toks, raw_open, verbs)
    fc = _mine(toks, curated_noun, verbs)
    return {"seed": seed, "n_vocab": len(vocab), "n_closed": len(closed),
            "emergent": {"n": sum(fe.values()), "obj_adj": obj_adj_frac(fe),
                         "sample": [f"{s} {v} {o}" for (s, v, o), c in fe.most_common(6)]},
            "raw_open": {"n": sum(fr.values()), "obj_adj": obj_adj_frac(fr)},
            "curated": {"n": sum(fc.values()), "obj_adj": obj_adj_frac(fc)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--attrib-thresh", type=float, default=0.45)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[emergent-POS open mining] K={a.K} attrib_thresh={a.attrib_thresh}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, attrib_thresh=a.attrib_thresh)
        recs.append(r)
        print(f"  [seed {s}] EMERGENT-POS: {r['emergent']['n']} facts, obj-adj-noise={r['emergent']['obj_adj']:.2f} "
              f"(e.g. {r['emergent']['sample'][:4]}) || RAW-open obj-adj={r['raw_open']['obj_adj']:.2f} | "
              f"CURATED obj-adj={r['curated']['obj_adj']:.2f}", flush=True)

    def m(arm): return float(np.mean([r[arm]["obj_adj"] for r in recs]))
    em, raw, cur = m("emergent"), m("raw_open"), m("curated")
    # GO: emergent-POS filtering CLEANS the open mining (obj-adjective noise near the curated-baseline, well below raw)
    go = all(r["emergent"]["obj_adj"] <= 0.10 for r in recs) and all(r["emergent"]["obj_adj"] < r["raw_open"]["obj_adj"] for r in recs)
    print(f"\n  AGGREGATE: obj-adjective-noise EMERGENT-POS={em:.2f} | RAW-open={raw:.2f} | CURATED={cur:.2f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL'} -- the EMERGENT POS filter (closed-class + attributive-adjective, "
          f"no hand noun list) {'CLEANS the OPEN mining (adjective-object noise {:.2f} ~ curated {:.2f}, vs raw {:.2f}) -> the brain learns clean facts from experience over the OPEN vocab'.format(em,cur,raw) if go else 'reduces but does not fully clean the adjective-object noise'}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "PARTIAL", "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
