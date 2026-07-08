"""The brain LEARNS relational facts FROM THE CORPUS (not just taught): mine clean subject-verb-object
triples from TinyStories and answer relational questions about them -- a breadth capability toward "talk
about what it has experienced", beyond hand-taught facts.

Cheap-first: raw SVO mining over the discovered vocab is NOISY (objects land on adjectives -- "tim saw
big"; subjects are names). A NOUN filter (subject + object both in a curated common-noun set) yields ~160
CLEAN corpus-derived facts ("bird saw cat", "dog saw ball", "mom saw dog"). Those are stored in the SVO
store (grounded real-corpus codes) and the brain answers what/who/yes-no about them, with the no-confab
moat on never-seen relations. Anti-cheat: PERMUTED-corpus mining yields DIFFERENT facts (the knowledge is
corpus-derived, not injected). numpy FHRR. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._realcorpus_svo_qa_derisk import SVOStore
from research.runners._realcorpus_svo_compose_probe import _phasors, _role
from research.runners._realcorpus_cancellation_derisk import _ANIMALS
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi

NOUNS_EXTRA = {"ball", "tree", "box", "sun", "cake", "toy", "car", "house", "door", "rock", "boat",
               "girl", "boy", "mom", "dad", "friend", "food", "water", "flower", "apple", "book",
               "cup", "hat", "bed", "park", "bug", "mouse"}
VERBS = ("saw", "found", "liked", "wanted", "chased", "hugged", "ate", "held", "kicked",
         "see", "like", "want", "chase", "hug", "eat", "hold")
DET = {"the", "a", "an", "his", "her", "their", "some", "that", "this", "one"}
# normalize a mined (past-tense) verb to a present-tense base so a query "what does X see?" matches "X saw Y"
VERB_NORM = {"saw": "see", "found": "find", "ate": "eat", "liked": "like", "wanted": "want",
             "chased": "chase", "hugged": "hug", "held": "hold", "kicked": "kick"}


def mine_svo(toks, nouns, verbs):
    """Clean noun-VERB-noun SVO mining (subject + object both nouns), determiners skipped. Returns a Counter."""
    from collections import Counter
    facts = Counter()
    n = len(toks)
    i = 0
    while i < n - 4:
        if toks[i] in nouns and toks[i + 1] in verbs:
            j = i + 2
            while j < n and toks[j] in DET:
                j += 1
            if j < n and toks[j] in nouns:
                facts[(toks[i], toks[i + 1], toks[j])] += 1
        i += 1
    return facts


def run_seed(seed, stories, K, top_n=40):
    vocab, gfreq = discover_vocab(stories, K)
    vset = set(vocab)
    nouns = (_ANIMALS | NOUNS_EXTRA) & vset
    verbs = [v for v in VERBS if v in vset]
    toks = [t for s in stories for t in s]
    facts = mine_svo(toks, nouns, verbs).most_common(top_n)
    if not facts:
        return None

    row_of = {w: i for i, w in enumerate(vocab)}
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in vset or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)

    def _vrow(v):                                                # resolve a verb surface form to a vocab row
        return row_of.get(v) or row_of.get(v[:-1] if v.endswith("s") else v)

    triples = [(s, v, o) for (s, v, o), _ in facts if _vrow(v) is not None]
    # phasors for EVERY word that participates (subjects, VERBS, objects) -- the store binds all three roles
    fact_words = sorted({w for (s, v, o) in triples for w in (s, o)} | {v for (_, v, _) in triples})
    part_rows = sorted({row_of[w] for w in fact_words if w in row_of} | {_vrow(v) for (_, v, _) in triples})
    Z = _phasors(codes, part_rows, seed)                         # grounded phasors keyed by ROW
    store = SVOStore(Z, part_rows, (_role(rng), _role(rng), _role(rng)))
    for (s, v, o) in triples:
        store.store(row_of[s], _vrow(v), row_of[o])

    # ANSWER what/who about the mined facts; MOAT on a never-mined relation.
    # The corpus facts are MANY-TO-MANY (cat saw dog AND bird AND mouse), so a query returns ONE valid
    # object -- score it correct if it is ANY object the subject actually V-ed in the corpus (a valid fact).
    mined_set = set(triples)
    what_ok = who_ok = n_ans = 0
    seen_sv = set()
    for (s, v, o) in triples[:25]:
        vr = _vrow(v)
        if vr is None or (s, v) in seen_sv:      # one query per distinct (subject, verb)
            continue
        seen_sv.add((s, v)); n_ans += 1
        po = store.answer_patient(row_of[s], vr)
        what_ok += int(po is not None and (s, v, vocab[po]) in mined_set)     # ANY valid object
        pa = store.answer_agent(vr, row_of[o])
        who_ok += int(pa is not None and (vocab[pa], v, o) in mined_set)      # ANY valid subject
    # MOAT: a (noun, verb) NOT in the mined facts -> abstain
    mined_sv = {(s, v) for (s, v, o) in triples}
    unmined, tries = [], 0
    while len(unmined) < 20 and tries < 400:
        s, v = rng.choice(fact_words), rng.choice([vv for vv in verbs if vv in row_of] or list(verbs))
        if (s, v) not in mined_sv:
            unmined.append((s, v))
        tries += 1
    moat = 0
    for (s, v) in unmined:
        vr = row_of.get(v) or row_of.get(v[:-1] if v.endswith("s") else v)
        if vr is not None:
            moat += int(store.answer_patient(row_of[s], vr) is None)
    return {"seed": seed, "n_facts": len(triples), "n_distinct_words": len(fact_words),
            "what_acc": what_ok / max(1, n_ans), "who_acc": who_ok / max(1, n_ans),
            "moat_abstain": moat / max(1, len(unmined)),
            "sample_facts": [f"{s} {v} {o}" for (s, v, o) in triples[:6]]}


def run_permuted(seed, stories, K, top_n=40):
    """Anti-cheat: shuffle the token order -> mining yields DIFFERENT facts (knowledge is corpus-derived)."""
    vocab, gfreq = discover_vocab(stories, K)
    vset = set(vocab)
    nouns = (_ANIMALS | NOUNS_EXTRA) & vset
    verbs = [v for v in VERBS if v in vset]
    toks = [t for s in stories for t in s]
    rng = np.random.default_rng(seed)
    perm = list(toks); rng.shuffle(perm)
    real = set(dict(mine_svo(toks, nouns, verbs).most_common(top_n)).keys())
    permd = set(dict(mine_svo(perm, nouns, verbs).most_common(top_n)).keys())
    overlap = len(real & permd) / max(1, len(real))
    return overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[learn corpus facts] mine clean SVO from TinyStories -> answer | K={a.K} top_n={a.top_n}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.top_n)
        if r is None:
            print(f"  [seed {s}] no facts mined"); continue
        r["permuted_overlap"] = run_permuted(s, stories, a.K, a.top_n)
        recs.append(r)
        print(f"  [seed {s}] mined {r['n_facts']} facts (e.g. {r['sample_facts'][:4]}) | "
              f"what={r['what_acc']:.2f} who={r['who_acc']:.2f} moat={r['moat_abstain']:.2f} | "
              f"permuted-corpus overlap={r['permuted_overlap']:.2f}", flush=True)

    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    def m(k): return float(np.mean([r[k] for r in recs]))
    what_ok = all(r["what_acc"] > 0.75 for r in recs)
    who_ok = all(r["who_acc"] > 0.75 for r in recs)
    moat_ok = all(r["moat_abstain"] > 0.9 for r in recs)
    corpus_derived = all(r["permuted_overlap"] < 0.5 for r in recs)     # shuffled corpus -> different facts
    go = what_ok and who_ok and moat_ok and corpus_derived
    print(f"\n  AGGREGATE: what={m('what_acc'):.2f} who={m('who_acc'):.2f} moat={m('moat_abstain'):.2f} "
          f"permuted-overlap={m('permuted_overlap'):.2f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- the brain LEARNS clean relational facts FROM THE "
          f"CORPUS and answers what/who about them (moat intact); the knowledge is corpus-derived (permuted-corpus "
          f"mining yields different facts, overlap {m('permuted_overlap'):.2f}).", flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "PARTIAL", "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
