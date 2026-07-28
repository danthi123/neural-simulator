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


def extract(corpus_path, vocab, max_sentences, nlp, keep_prep=False):
    """Extract (subject, verb, object) facts. When keep_prep=True, ALSO retain the PREPOSITION that introduced an
    oblique object (Tier 0.1: "go to the park" keeps 'to' -> a typed GOAL role downstream, instead of collapsing
    the oblique into a bare patient and discarding the preposition). The preposition is recorded in a parallel
    `preps` dict keyed by the (a, vl, p) triple; the default (keep_prep=False) output is byte-identical."""
    counts, attest = Counter(), {}
    preps = {}     # (a, vl, p) -> the preposition string that introduced p (None for a direct object)
    # the corpus is one blob delimited by <|endoftext|> (NOT newlines); split into stories (each short,
    # well under spaCy's 1M-char parser limit). Skip any pathological >50k-char chunk defensively.
    with open(corpus_path, encoding="utf-8") as fh:
        text = fh.read()
    # robust to both corpora: TinyStories is <|endoftext|>-delimited (few newlines), Simple-Wiki is one article
    # PER LINE. Normalize both to newline-split, then chunk any long article well under spaCy's 1M parser limit.
    pieces = []
    for s in text.replace("<|endoftext|>", "\n").split("\n"):
        s = s.strip()
        if not s:
            continue
        if len(s) <= 100000:
            pieces.append(s)
        else:
            pieces.extend(s[i:i + 100000] for i in range(0, len(s), 100000))
    n_sent = 0
    for doc in nlp.pipe(pieces, batch_size=128):
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
                    p, prep = None, None
                    if c.dep_ in OBJ_DEPS:
                        p = vform(c, vocab)
                    elif c.dep_ == "prep":
                        po = next((x for x in c.children if x.dep_ == "pobj"), None)
                        if po is not None:
                            p = vform(po, vocab)
                            prep = c.text.lower()       # Tier 0.1: KEEP the preposition (was discarded)
                    if p and p != a:
                        counts[(a, vl, p)] += 1
                        attest.setdefault((a, vl, p), sent.text.strip()[:90])
                        if keep_prep:
                            preps.setdefault((a, vl, p), prep)
        if n_sent >= max_sentences:
            break
    return (counts, attest, preps, n_sent) if keep_prep else (counts, attest, n_sent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="bridges/firstchat/brain1454_w7000_seed42.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--max-sentences", type=int, default=30000)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--typed-roles", action="store_true",
                    help="Tier 0.1: keep the preposition + emit a typed oblique role (GOAL/RECIPIENT/LOCATION/...) "
                         "per the verb-frame lexicon, instead of collapsing obliques into a bare patient.")
    a = ap.parse_args()

    vocab = load_vocab(a.npz)
    print(f"[svo] brain vocab: {len(vocab)} words", flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"]) if False else spacy.load("en_core_web_sm")
    print(f"[svo] spaCy {spacy.__version__} loaded; parsing up to {a.max_sentences} sentences of {a.corpus} ...",
          flush=True)

    if a.typed_roles:
        from research.runners.argstructure_composer import VERB_PREP_ROLE, FRAME_ROLES
        counts, attest, preps, n_sent = extract(a.corpus, vocab, a.max_sentences, nlp, keep_prep=True)
        kept = [(t, c) for t, c in counts.most_common() if c >= a.min_count]
        print(f"[svo] parsed {n_sent} sentences -> {len(counts)} distinct triples, {len(kept)} with "
              f"count>={a.min_count}; assigning typed oblique roles by (verb, prep)", flush=True)
        recs = []
        for (aa, vv, pp), c in kept:
            prep = preps.get((aa, vv, pp))
            # which typed role does this object fill? A prep maps via VERB_PREP_ROLE; a DIRECT object (prep=None)
            # fills the verb-frame's FIRST internal-argument content role (Bock & Levelt: the verb lemma projects
            # its argument frame -- a transitive verb's direct object is the `patient`, but a ditransitive verb like
            # `give`/`send`/`put` has NO `patient` slot, so its direct object ('mom gave a hug') is the THEME, not a
            # patient the frame can't render). An unknown (verb,prep) with a single oblique role -> that role.
            if prep is not None:
                role = VERB_PREP_ROLE.get((vv, prep))
                if role is None:
                    obliques = [r for r in FRAME_ROLES.get(vv, []) if r not in ("agent", "action", "patient")]
                    role = obliques[0] if len(obliques) == 1 else None
            else:
                # direct object -> the frame's first non-(agent/action) CONTENT role (patient for transitive,
                # THEME for ditransitive go-by-the-frame), so the stored role is one the frame actually renders.
                frame_obj = [r for r in FRAME_ROLES.get(vv, FRAME_ROLES["_default"]) if r not in ("agent", "action")]
                role = frame_obj[0] if frame_obj else "patient"
            rec = {"agent": aa, "action": vv, "count": c, "attest": attest[(aa, vv, pp)], "prep": prep}
            rec[role if role else "patient"] = pp
            recs.append(rec)
        print(f"[svo] TOP {a.top_n} typed-role corpus facts:", flush=True)
        for r in recs[:a.top_n]:
            obl = {k: v for k, v in r.items() if k not in ("agent", "action", "count", "attest", "prep")}
            print(f"    {r['count']:4d}x  ({r['agent']}, {r['action']}, {obl})  prep={r['prep']}  "
                  f"e.g. \"{r['attest']}\"", flush=True)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(recs, fh, indent=1)
            print(f"[svo] wrote {len(recs)} typed-role facts -> {a.out}", flush=True)
        return 0

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
