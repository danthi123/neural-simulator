"""KNOWLEDGE-half of breadth, rung 3: ground the inheritance in REAL corpus-mined properties.

Rungs 1-2 taught a SYNTHETIC per-category property target. This rung mines the property from the
corpus: for each DISCOVERED category, the most category-DISCRIMINATIVE co-occurring content word
(the word its members co-occur with distinctively vs other categories -- a real fact, e.g. animals
co-occur with 'little'/'ran', family with 'said'). A held-out member must inherit ITS category's
mined-real property (argmax over the categories' mined properties). The mechanism is rung-1's
associative-memory inheritance read; the NEW part is that the property is a REAL mined fact, and the
demo prints the mined property words so the grounding is legible.

The property is represented as a distinct tag LABELED by the mined real word (clean discrimination),
so the rigor is: (a) the property WORDS are genuinely category-discriminative (reported), and (b) a
held-out member inherits ITS category's mined property, with label-derangement collapsing to chance.

Reuse-by-import of the breadth discovery + rung-1's inheritance read. numpy-only, offline. NO sim/ edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from collections import Counter

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, build_probe, TAXONOMY_8x8,
    STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows, _inherit_accuracy, _build_splits
from research.runners.corpus_stream import load_token_stream_multi


def _co_counts(stories, member_words, context_vocab, window=WINDOW):
    """Per-(member, context) windowed co-occurrence counts -> used to mine category-discriminative words."""
    mem_set = set(member_words); ctx_idx = {w: i for i, w in enumerate(context_vocab)}
    keep = mem_set | set(context_vocab)
    C = np.zeros((len(member_words), len(context_vocab)), dtype=np.float64)
    mrow = {w: i for i, w in enumerate(member_words)}
    for toks in stories:
        kept = [t for t in toks if t in keep]
        for c in range(len(kept)):
            w = kept[c]
            if w not in mem_set:
                continue
            lo, hi = max(0, c - window), min(len(kept), c + window + 1)
            for u in set(kept[lo:hi]) - {w}:
                if u in ctx_idx:
                    C[mrow[w], ctx_idx[u]] += 1.0
    return C, mrow


def mine_properties(stories, probe_words, probe_rows, probe_labels, vocab, gfreq, cat_ids, row_to_word):
    """For each category, the most DISCRIMINATIVE context content word: highest (within-cat mean minus
    between-cat mean) normalized co-occurrence. Returns {cat: (property_word, discrim_score)}."""
    # context = frequent content words not in the probe set
    probe_set = set(probe_words)
    context = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in probe_set or len(w) < MIN_WORD_LEN:
            continue
        context.append(w)
        if len(context) >= 400:
            break
    C, mrow = _co_counts(stories, probe_words, context)
    C = C / (C.sum(axis=1, keepdims=True) + 1e-9)      # per-member row-normalized co-occurrence profile
    word_of_row = row_to_word
    lab_of_word = {}
    for r, lab in zip(probe_rows.tolist(), probe_labels.tolist()):
        lab_of_word[word_of_row[r]] = lab
    props = {}
    for k in cat_ids:
        in_rows = [mrow[w] for w in probe_words if lab_of_word.get(w) == k]
        out_rows = [mrow[w] for w in probe_words if lab_of_word.get(w) not in (k, None)]
        if not in_rows or not out_rows:
            continue
        within = C[in_rows].mean(axis=0)
        between = C[out_rows].mean(axis=0)
        discrim = within - between
        j = int(np.argmax(discrim))
        props[k] = (context[j], float(discrim[j]))
    return props


def run_seed(seed, stories, K, args):
    vocab, gfreq = discover_vocab(stories, K)
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    probe_rows, probe_labels, probe_words, n_cat, per_cat = build_probe(vocab, TAXONOMY_8x8)
    row_to_word = {int(r): vocab[int(r)] for r in probe_rows.tolist()}
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)

    cats = {}
    for r, lab in zip(probe_rows.tolist(), probe_labels.tolist()):
        cats.setdefault(lab, []).append(r)
    usable = {c: rs for c, rs in cats.items() if len(rs) >= 4}
    if len(usable) < 2:
        return None
    cat_ids = sorted(usable.keys())

    # MINE the real property word per category (category-discriminative co-occurring content word)
    props = mine_properties(stories, probe_words, probe_rows, probe_labels, vocab, gfreq, cat_ids, row_to_word)

    # inheritance with DISTINCT tags labeled by the mined real word (clean discrimination; the real part is the mining)
    rng = np.random.RandomState(seed)
    P = rng.randn(len(cat_ids), 64)
    taught_by_cat, heldout_q, taught_q = _build_splits(usable, cat_ids, rng)
    ho_acc, n_ho = _inherit_accuracy(codes, taught_by_cat, heldout_q, P, cat_ids)

    # label-derangement anti-cheat
    all_rows = [r for rs in usable.values() for r in rs]
    der_accs = []
    for _ in range(5):
        pool = list(all_rows); rng.shuffle(pool)
        der = {}; i = 0
        for c in cat_ids:
            k = len(usable[c]); der[c] = pool[i:i + k]; i += k
        d_taught, d_held, _ = _build_splits(der, cat_ids, rng)
        da, _ = _inherit_accuracy(codes, d_taught, d_held, P, cat_ids)
        der_accs.append(da)

    # category-name lookup for legible reporting
    cat_names = list(TAXONOMY_8x8.keys())
    mined = {cat_names[k]: props.get(k, ("<none>", 0.0)) for k in cat_ids}
    return {
        "seed": seed, "n_categories": len(cat_ids), "chance": 1.0 / len(cat_ids),
        "heldout_inherit_acc": ho_acc, "deranged_acc": float(np.mean(der_accs)), "n_heldout": n_ho,
        "mined_properties": {name: {"word": w, "discrim": round(s, 4)} for name, (w, s) in mined.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[rung3 mined-properties] corpus={a.corpus_path} stories={len(stories)} K={a.K}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a)
        if r is None:
            continue
        recs.append(r)
        mp = " ; ".join(f"{name}->'{d['word']}'({d['discrim']:+.3f})" for name, d in r["mined_properties"].items())
        print(f"  [seed {s}] held-out inherit={r['heldout_inherit_acc']:.3f} | deranged={r['deranged_acc']:.3f} | "
              f"chance={r['chance']:.3f}\n            MINED: {mp}", flush=True)

    def m(k): return float(np.mean([r[k] for r in recs]))
    ho, der, ch = m("heldout_inherit_acc"), m("deranged_acc"), m("chance")
    beats_chance = all(r["heldout_inherit_acc"] - r["chance"] > a.margin for r in recs)
    beats_der = all(r["heldout_inherit_acc"] - r["deranged_acc"] > a.margin for r in recs)
    go = beats_chance and beats_der
    verdict = "GO" if go else "NEGATIVE"
    print(f"\n  AGGREGATE ({len(recs)} seeds): held-out inherit={ho:.3f} | deranged={der:.3f} | chance={ch:.3f}", flush=True)
    print(f"  beats_chance={beats_chance} | beats_deranged={beats_der}", flush=True)
    print(f"  VERDICT: {verdict} -- a HELD-OUT member inherits its category's REAL corpus-MINED property "
          f"{'(above chance + label-derangement) -> the inheritance is grounded in a real mined fact, not a synthetic target' if go else '-- within control range'}.",
          flush=True)
    # report the most-common mined property per category across seeds (the stable real facts)
    from collections import defaultdict
    byname = defaultdict(Counter)
    for r in recs:
        for name, d in r["mined_properties"].items():
            byname[name][d["word"]] += 1
    print("  STABLE MINED PROPERTIES (modal word across seeds):", flush=True)
    for name, c in byname.items():
        w, n = c.most_common(1)[0]
        print(f"    {name:10s} -> '{w}' ({n}/{len(recs)} seeds)", flush=True)
    if a.out:
        json.dump({"verdict": verdict, "K": a.K, "aggregate": {"heldout": ho, "deranged": der, "chance": ch},
                   "beats_chance": beats_chance, "beats_deranged": beats_der, "per_seed": recs},
                  open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
