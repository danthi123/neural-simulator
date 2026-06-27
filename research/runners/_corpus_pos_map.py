#!/usr/bin/env python
"""Deep-knowledge build (taxonomy expansion): build a corpus POS-map. Tagger-only spaCy over a capped sample
-> {content-word: dominant POS (NOUN/VERB/ADJ)} + full-corpus frequency. These are the candidates for a
POS-DRIVEN concept-admission path BEYOND the g20 taxonomy (~2012) cap: the trainer's new admission reads this
to admit the top-N corpus content words with POS-based cat_ids (the proposer noun/verb pools; gen demoted).
Host-side curriculum prep (legitimate -- preparing the syllabus). CPU.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import spacy

CONTENT_POS = {"NOUN", "VERB", "ADJ"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-paths", default="data/corpus/tinystories.txt,data/corpus/simplewiki.txt")
    ap.add_argument("--max-pieces", type=int, default=120000, help="stories/articles to POS-tag (dominant POS is sample-stable)")
    ap.add_argument("--out", default="research/findings/raw/_corpus_pos_map.json")
    a = ap.parse_args()
    paths = [p for p in a.corpus_paths.split(",") if p]

    # full-corpus frequency (cheap regex; same tokenizer as the stream cortex)
    gfreq = Counter()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                gfreq.update(re.findall(r"[a-z]+", line.lower()))
    print(f"[posmap] full-corpus freq: {len(gfreq)} word types", flush=True)

    # tagger-only spaCy (disable parser/ner/lemmatizer -> fast POS over a sample)
    # keep tok2vec + tagger + attribute_ruler (attribute_ruler sets tok.pos_ from tag_); drop only the heavy/unused.
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    pieces = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for s in fh.read().replace("<|endoftext|>", "\n").split("\n"):
                s = s.strip()
                if s and len(s) < 100000:
                    pieces.append(s)
    pieces = pieces[:a.max_pieces]
    print(f"[posmap] POS-tagging {len(pieces)} pieces (tagger-only) ...", flush=True)
    pos_counts = defaultdict(Counter)
    for doc in nlp.pipe(pieces, batch_size=256):
        for tok in doc:
            pos_counts[tok.text.lower()][tok.pos_] += 1   # count ALL POS per word (not just content)

    # A word is a CONTENT concept iff its DOMINANT POS (over ALL taggings) is content. This excludes function
    # words that get an occasional content mis-tag ("in" tagged NOUN a few times) -- the earlier 'in:NOUN' bug.
    out = {}
    for w, pc in pos_counts.items():
        dom = pc.most_common(1)[0][0]
        if dom in CONTENT_POS:
            out[w] = {"pos": dom, "freq": int(gfreq.get(w, 0))}
    ranked = sorted(out.items(), key=lambda kv: -kv[1]["freq"])
    print(f"[posmap] {len(ranked)} content words tagged | top 20: "
          + ", ".join(f'{w}:{d["pos"]}' for w, d in ranked[:20]), flush=True)
    for thr in (50, 100, 300, 1000):
        print(f"  content words freq>={thr}: {sum(1 for _, d in ranked if d['freq'] >= thr)}", flush=True)
    json.dump(dict(ranked), open(a.out, "w", encoding="utf-8"))
    print(f"[posmap] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
