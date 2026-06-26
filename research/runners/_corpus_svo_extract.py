#!/usr/bin/env python
"""Stage 1a (deep-knowledge build): extract REAL (subject, verb, object) facts from the corpus with spaCy,
restricted to a brain's learned vocab. Replaces _make_svo_facts's uniform-random sampling -- the GAP-2 root
cause (research/findings/2026-06-26-deep-knowledge-brain-fluency-research.md) -- with corpus-attested facts.

This is host-side CURRICULUM preprocessing (legitimate per BRAIN-BASED-ONLY: preparing the syllabus, like
rendering a retinal image the neural retina then receives); the brain still STORES/RECALLS/GENERALIZES the
facts via spikes/binding. Anti-cheat: every kept fact is corpus-ATTESTED (>= --min-count occurrences) and
logged with a source sentence, so a fact is provably from the corpus, not invented.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np
import spacy

OBJ_DEPS = {"dobj", "dative", "attr", "oprd"}


def load_vocab(npz_path):
    # our own artifact (this session's curriculum runner); allow_pickle only for vocab dtype=object
    d = np.load(npz_path, allow_pickle=True)
    return set(str(w).lower() for w in d["vocab"])


def vform(tok, vocab):
    """The in-vocab surface for a token: prefer the lemma, fall back to the raw lowercased text.
    (The brain's vocab is raw corpus tokens via re.findall[a-z]+, so 'chase' the lemma may be absent
    while 'chased' is present -- accept whichever the brain actually learned.)"""
    lem = tok.lemma_.lower()
    if lem in vocab:
        return lem
    txt = tok.text.lower()
    if txt in vocab:
        return txt
    return None


def extract(corpus_path, vocab, max_sentences, nlp):
    counts, attest = Counter(), {}
    # the corpus is one blob delimited by <|endoftext|> (NOT newlines); split into stories (each short,
    # well under spaCy's 1M-char parser limit). Skip any pathological >50k-char chunk defensively.
    with open(corpus_path, encoding="utf-8") as fh:
        text = fh.read()
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip() and len(s) < 50000]
    n_sent = 0
    for doc in nlp.pipe(stories, batch_size=128):
        for sent in doc.sents:
            n_sent += 1
            for tok in sent:
                if tok.dep_ != "nsubj" or tok.head.pos_ not in ("VERB", "AUX"):
                    continue
                a = vform(tok, vocab)
                v = tok.head
                vl = vform(v, vocab)
                if a is None or vl is None:
                    continue
                for c in v.children:
                    p = None
                    if c.dep_ in OBJ_DEPS:
                        p = vform(c, vocab)
                    elif c.dep_ == "prep":
                        po = next((x for x in c.children if x.dep_ == "pobj"), None)
                        if po is not None:
                            p = vform(po, vocab)
                    if p and p != a:
                        counts[(a, vl, p)] += 1
                        attest.setdefault((a, vl, p), sent.text.strip()[:90])
        if n_sent >= max_sentences:
            break
    return counts, attest, n_sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="bridges/firstchat/brain1454_w7000_seed42.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--max-sentences", type=int, default=30000)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    vocab = load_vocab(a.npz)
    print(f"[svo] brain vocab: {len(vocab)} words", flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"]) if False else spacy.load("en_core_web_sm")
    print(f"[svo] spaCy {spacy.__version__} loaded; parsing up to {a.max_sentences} sentences of {a.corpus} ...",
          flush=True)
    counts, attest, n_sent = extract(a.corpus, vocab, a.max_sentences, nlp)
    kept = [(t, c) for t, c in counts.most_common() if c >= a.min_count]
    print(f"[svo] parsed {n_sent} sentences -> {len(counts)} distinct in-vocab triples, "
          f"{len(kept)} with count>={a.min_count}", flush=True)
    print(f"[svo] TOP {a.top_n} corpus-attested facts (are they MEANINGFUL?):", flush=True)
    for (aa, vv, pp), c in kept[:a.top_n]:
        print(f"    {c:4d}x  ({aa}, {vv}, {pp})   e.g. \"{attest[(aa,vv,pp)]}\"", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump([{"agent": aa, "action": vv, "patient": pp, "count": c,
                        "attest": attest[(aa, vv, pp)]} for (aa, vv, pp), c in kept], fh, indent=1)
        print(f"[svo] wrote {len(kept)} attested facts -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
