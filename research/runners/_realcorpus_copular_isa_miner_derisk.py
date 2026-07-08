"""COPULAR is-a mining (the untested taxonomy angle): the distributional approaches (centroid / stacked-pooler
co-occurrence) are data-gated -- co-occurrence gives FLAT categories, not a nested is-a hierarchy. This mines
the is-a signal DIRECTLY from EXPLICIT definitional sentences ("a cat is an animal", "a robin is a bird") via
copular patterns, using the emergent closed-class (EMERGE-62) to keep noun->noun pairs. Tests whether an
ENCYCLOPEDIC corpus (WikiText) yields a CLEAN is-a graph where a children's-story corpus (TinyStories) does not.

Metrics: n is-a pairs mined; the graph's transitivity/multi-level depth; and (light) whether the mined is-a
graph supports inheritance (a child inherits a superordinate property via the discovered is-a) vs a deranged
is-a control. Cheap-first first look. numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from research.runners.corpus_stream import load_token_stream_multi

_STOP = {"the", "a", "an", "this", "that", "these", "those", "it", "he", "she", "they", "there", "one",
         "which", "who", "what", "his", "her", "its", "their", "such", "any", "no", "some", "each", "not",
         "very", "more", "most", "only", "also", "first", "same", "other", "well"}
_DET = {"a", "an", "the"}


def _is_content_noun(w):
    return w.isalpha() and len(w) >= 3 and w not in _STOP and w.islower()


_ADJ_LIKE = ("ly", "ed", "ing", "ous", "ive", "ful", "less", "able", "ible")


def _np_head(st, j, n):
    """From position j (just after 'a'/'an'), skip attributive adjectives and return the HEAD NOUN (skip up to
    2 adjective-like words: common adjectival/participial/adverbial suffixes). 'unk' -> reject."""
    for _ in range(3):
        if j >= n:
            return None
        w = st[j]
        if w == "unk":
            return None
        if w.endswith(_ADJ_LIKE):                               # attributive adjective / participle / adverb -> skip
            j += 1
            continue
        return w                                                # the head noun
    return None


def mine_isa(stories, max_pairs=100000):
    """Mine copular is-a pairs '<child> is a/an [adj]* <parentNOUN>' -- take the NP HEAD noun (skip attributive
    adjectives), filter 'unk', keep noun->noun only; 'a kind/type of Y' -> Y."""
    pairs = Counter()
    for st in stories:
        n = len(st)
        for i in range(1, n - 2):
            if st[i] == "is" and st[i + 1] in ("a", "an"):
                child = st[i - 1]
                if child == "unk" or not _is_content_noun(child):
                    continue
                if st[i + 2] in ("kind", "type", "sort", "form", "member", "species", "genus") and \
                        i + 4 < n and st[i + 3] == "of":
                    parent = _np_head(st, i + 4, n)
                else:
                    parent = _np_head(st, i + 2, n)
                if parent and _is_content_noun(parent) and child != parent:
                    pairs[(child, parent)] += 1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/wikitext.txt")
    ap.add_argument("--min-count", type=int, default=2)
    a = ap.parse_args()
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[copular is-a miner] corpus={a.corpus_path} ({len(stories)} segments)", flush=True)
    pairs = mine_isa(stories)
    kept = {p: c for p, c in pairs.items() if c >= a.min_count}
    # build the is-a graph (child -> set of parents) on the min-count-cleared pairs
    isa = defaultdict(set)
    for (c, p) in kept:
        isa[c].add(p)
    # multi-level depth: is there a chain child -> parent -> grandparent?
    chains = 0
    for c in isa:
        for p in isa[c]:
            if p in isa:                                        # p also has a parent -> a 2-level chain
                chains += 1
    # HUB parents = genuine superordinates (>=3 DISTINCT children each). A real taxonomy has hubs with many members.
    n_children_of = Counter()
    for (c, p) in kept:
        n_children_of[p] += 1
    hubs = {p for p, k in n_children_of.items() if k >= 3}
    hub_pairs = {(c, p): kept[(c, p)] for (c, p) in kept if p in hubs}
    print(f"  mined {sum(pairs.values())} raw hits -> {len(pairs)} distinct pairs; >={a.min_count}x: {len(kept)} pairs, "
          f"{len(isa)} children, {len(n_children_of)} parents", flush=True)
    print(f"  HUB superordinates (>=3 distinct children): {len(hubs)} -> {len(hub_pairs)} hub-pairs; chains: {chains}", flush=True)
    print(f"  top hub parents: {[f'{p}({k})' for p, k in n_children_of.most_common(12) if p in hubs]}", flush=True)
    print(f"  sample hub is-a pairs: {[f'{c}->{p}' for (c, p) in list(hub_pairs)[:15]]}", flush=True)
    # VERDICT: a usable taxonomy needs genuine HUB superordinates (>=3 children) whose children/parents are real
    # nouns -- and a human read of the sample must show CLEAN noun->noun is-a (not adjective/adverb noise).
    go = len(hubs) >= 8 and len(hub_pairs) >= 40
    print(f"\n  VERDICT: {'PROMISING (inspect the sample for clean noun->noun is-a)' if go else 'DATA-GATE HOLDS'} -- "
          f"copular is-a mining from this corpus {'yields HUB superordinates with multiple members -> candidate for the stacked-pooler taxonomy (pending a clean-pair audit)' if go else 'does NOT yield enough clean HUB superordinates; the taxonomy data-gate persists even for explicit NP-head copular extraction'}.", flush=True)


if __name__ == "__main__":
    main()
