"""KNOWLEDGE-half of breadth, TAXONOMIC is-a from DEFINITIONAL syntax: learn is-a categories from
"X is a Y" statements (not co-occurrence clustering), and do inheritance over them.

The multi-level negative (2026-07-08) showed co-occurrence CLUSTERING gives FLAT categories, not a
taxonomic is-a hierarchy -- because children's-story co-occurrence lacks the is-a signal. But an
ENCYCLOPEDIC corpus (WikiText) carries it explicitly: "X is a species/song/novel/church" (real
taxonomic superordinates). This mechanism MINES those "X is a Y" statements to learn TAXONOMIC
categories (subjects grouped by their superordinate), then does property inheritance over the LEARNED
is-a groups (a held-out subject of a super inherits a property taught to other subjects of that super).

⇒ the brain learns "X is a <category>" from definitional text and reasons (inherits) over the taxonomy
-- a DIFFERENT, explicitly-taxonomic category signal than distributional clustering.

Mechanism: (1) mine (subject, superordinate) is-a pairs from "X is a/an [adj]* HEAD-NOUN"; (2) group
subjects by superordinate, keep supers with >=4 subjects (a category); (3) learn co-occurrence CODES
for the subjects via the stream cortex; (4) inheritance: teach a distinct property to SOME subjects of
each super, a HELD-OUT subject inherits its super's property (argmax, via the code similarity to its
super-siblings). Anti-cheat: super-label DERANGEMENT collapses it. Reuse-by-import. NO sim/ edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from collections import Counter, defaultdict

from research.runners._emergent_vocab_breadth_scale_derisk import (
    learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows, _inherit_accuracy, _build_splits
from research.runners.corpus_stream import load_token_stream_multi

_STOP = set(("the a an of in on at to and or but with for is was are were be been being that this it "
             "he she they we you as by from very more most also had has have not no yes who which what "
             "there their his her its our your all some any one two his her".split()))


def mine_isa(stories, min_subjects=4):
    """Mine (subject, superordinate) is-a pairs from 'X is a/an [adj]* HEAD-NOUN'. Returns {super: [subjects]}.

    Head-noun heuristic: the LAST content word before a sentence/PP boundary (skips leading adjectives).
    NOUN filter: a real superordinate NOUN also appears as a SUBJECT of 'is a' somewhere (nouns fill both
    roles); adjectives/participles (located/american) do not -> filters the noise supers structurally."""
    _BOUND = set("in of from with by for on at as and or that which who".split())
    pairs = Counter()
    subjects_seen = set()                       # words that appear as SUBJECT of 'is a' (real nouns)
    for toks in stories:
        for i in range(len(toks) - 4):
            if toks[i + 1] == "is" and toks[i + 2] in ("a", "an"):
                subj = toks[i]
                if subj not in _STOP and len(subj) > 2:
                    subjects_seen.add(subj)
                # head noun = the last content word before a boundary word, in the next up-to-4 positions
                head = None
                for w in toks[i + 3:i + 7]:
                    if w in _BOUND:
                        break
                    if w not in _STOP and len(w) > 2:
                        head = w                # keep advancing to the LAST pre-boundary content word
                if head and subj not in _STOP and len(subj) > 2:
                    pairs[(subj, head)] += 1
    by_super = defaultdict(set)
    for (subj, sup), c in pairs.items():
        by_super[sup].add(subj)
    # keep supers that (a) have >= min_subjects distinct subjects AND (b) are themselves real nouns
    # (appear as a subject of 'is a' -> filters adjectives/participles like located/american)
    return {sup: sorted(subs) for sup, subs in by_super.items()
            if len(subs) >= min_subjects and sup in subjects_seen}


def run_seed(seed, stories, isa_groups, args):
    # vocab = all subjects across the usable is-a groups (+ their supers as context is fine)
    subjects = sorted({s for subs in isa_groups.values() for s in subs})
    if len(subjects) < 8:
        return None
    # context hubs = frequent content words not in the subject set
    freq = Counter(w for toks in stories for w in toks)
    hubs = []
    subj_set = set(subjects)
    for w, _ in freq.most_common():
        if w in STOPLIST or w in subj_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    # learn co-occurrence codes for the subjects (as the "vocab")
    codes_full, _ = learn_stream_codes(seed, stories, subjects, hubs, window=WINDOW)
    U = _unit_rows(codes_full)
    row_of = {w: i for i, w in enumerate(subjects)}

    # usable supers: >= 4 subjects that HAVE codes
    usable = {}
    for sup, subs in isa_groups.items():
        rows = [row_of[s] for s in subs if s in row_of]
        if len(rows) >= 4:
            usable[sup] = rows
    if len(usable) < 2:
        return None
    sup_ids = sorted(usable.keys())
    rng = np.random.RandomState(seed)
    P = rng.randn(len(sup_ids), 64)

    taught_by, heldout_q, _ = _build_splits({s: usable[s] for s in sup_ids}, sup_ids, rng)
    ho_acc, n_ho = _inherit_accuracy(codes_full, taught_by, heldout_q, P, sup_ids)

    # DERANGE super labels
    all_rows = [r for rs in usable.values() for r in rs]
    der_accs = []
    for _ in range(5):
        pool = list(all_rows); rng.shuffle(pool)
        der = {}; i = 0
        for s in sup_ids:
            n = len(usable[s]); der[s] = pool[i:i + n]; i += n
        d_t, d_h, _ = _build_splits(der, sup_ids, rng)
        da, _ = _inherit_accuracy(codes_full, d_t, d_h, P, sup_ids)
        der_accs.append(da)

    return {"seed": seed, "n_supers": len(sup_ids), "chance": 1.0 / len(sup_ids),
            "isa_inherit_acc": ho_acc, "deranged_acc": float(np.mean(der_accs)), "n_heldout": n_ho,
            "example_supers": {s: [subjects[r] for r in usable[s][:5]] for s in sup_ids[:4]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/wikitext.txt")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--min-subjects", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    isa_groups = mine_isa(stories, min_subjects=a.min_subjects)
    print(f"[isa-definitional inheritance] corpus={a.corpus_path} stories={len(stories)} "
          f"| mined {len(isa_groups)} is-a supers with >={a.min_subjects} subjects", flush=True)
    print(f"  example is-a groups: " + " ; ".join(f"{sup}<-{sorted(subs)[:5]}" for sup, subs in list(isa_groups.items())[:5]), flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, isa_groups, a)
        if r is None:
            print(f"  [seed {s}] too few usable is-a supers -- skip", flush=True); continue
        recs.append(r)
        print(f"  [seed {s}] IS-A inherit (held-out subject -> super property)={r['isa_inherit_acc']:.3f} | "
              f"deranged={r['deranged_acc']:.3f} | chance={r['chance']:.3f} (supers={r['n_supers']}, nq={r['n_heldout']})",
              flush=True)
    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    def m(k): return float(np.mean([r[k] for r in recs]))
    ho, der, ch = m("isa_inherit_acc"), m("deranged_acc"), m("chance")
    bc = all(r["isa_inherit_acc"] - r["chance"] > a.margin for r in recs)
    bd = all(r["isa_inherit_acc"] - r["deranged_acc"] > a.margin for r in recs)
    go = bc and bd
    print(f"\n  AGGREGATE ({len(recs)} seeds): IS-A inherit={ho:.3f} | deranged={der:.3f} | chance={ch:.3f}", flush=True)
    print(f"  beats_chance={bc} | beats_deranged={bd}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- a held-out subject of a DEFINITIONALLY-LEARNED is-a category "
          f"{'INHERITS its superordinate property (above chance + derangement) -> taxonomic reasoning from is a statements' if go else 'does NOT clearly inherit'}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "aggregate": {"isa": ho, "deranged": der, "chance": ch},
                   "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
