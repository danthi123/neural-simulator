"""KNOWLEDGE-half of breadth, rung 1 (rate, cheap-first): does a category DISCOVERED from a
REAL broad corpus support HELD-OUT property inheritance?

The breadth thread established (2026-07-08) that the emergent stream cortex learns genuine
frequency-independent category structure over a 1024-word real-corpus (TinyStories) discovered
vocab, matching the batch co-occurrence ceiling. That is the vocab-STRUCTURE half. This rung
tests the KNOWLEDGE half: can EMERGE-style inheritance (teach a class property to some members
-> a HELD-OUT member inherits it via the shared category structure -- EMERGE-30/42, but there
over SYNTHETIC streams) ride the REAL-corpus-discovered categories?

Mechanism (rate-first, the cheapest rung of the ladder; the spiking wire-in reuses EMERGE-42):
  * codes = the learned co-occurrence codes for the top-K TinyStories vocab (the breadth mechanism).
  * A discovered category C = the a-priori-category probe words (e.g. animals) present in the vocab.
    Split C into TAUGHT (a subset) + HELDOUT (the rest) -- the held-out members are NEVER taught.
  * TEACH property p (a fixed random target vector) to the TAUGHT members: a Hebbian associative
    memory M = sum_taught outer(unit(code_m), p). (No held-out member enters M.)
  * RECALL for a query word q: p_hat = M^T @ unit(code_q); inheritance score = cosine(p_hat, p).
  * INHERITANCE works iff a HELD-OUT same-category member recalls p (high score) via its code's
    similarity to the taught members -- the shared discovered category structure carries it.

Anti-cheats (the load-bearing controls):
  (A) WRONG-CATEGORY: a member of a DIFFERENT discovered category must NOT inherit p (score ~ chance).
  (B) SCRAMBLED corpus: co-occurrence destroyed (unigram freq preserved) -> the category dissolves ->
      held-out inheritance must COLLAPSE to the wrong-category level.
  (C) MEMORIZATION floor: the taught members trivially recall p (a ceiling sanity check, not the claim).

GO = held-out same-category inheritance CLEARLY exceeds BOTH wrong-category (A) and scrambled (B),
multi-seed. Reuse-by-import of the breadth runner; numpy-only, offline.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, build_probe, TAXONOMY_8x8,
    STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi


def _unit_rows(C):
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)


def _inherit_accuracy(codes, taught_by_cat, queries, P, cat_ids):
    """MULTI-property inheritance discrimination. Each category c has its OWN property vector P[c].
    Build ONE associative memory over all taught members (each bound to ITS category's property);
    a query INHERITS the property whose category its code is most similar to. Returns the fraction of
    (query, true_cat) pairs whose recalled property argmaxes to the TRUE category (== nearest-taught-
    category by code similarity). Chance = 1/len(cat_ids)."""
    U = _unit_rows(codes)
    D = codes.shape[1]
    M = np.zeros((D, P.shape[1]), dtype=np.float64)
    for ci, rows in taught_by_cat.items():
        idx = cat_ids.index(ci)
        for r in rows:
            M += np.outer(U[r], P[idx])
    correct = 0
    for q, true_c in queries:
        phat = U[q] @ M                     # (n_prop,)
        scores = P @ phat                   # alignment with each category's property
        pred = cat_ids[int(np.argmax(scores))]
        correct += int(pred == true_c)
    return correct / max(1, len(queries)), len(queries)


def _build_splits(rows_by_cat, cat_ids, rng):
    """taught_by_cat + held-out/taught query lists (each query = (row, true_cat))."""
    taught_by_cat, heldout_q, taught_q = {}, [], []
    for c in cat_ids:
        rows = list(rows_by_cat[c]); rng.shuffle(rows)
        n_taught = max(2, len(rows) // 2)
        taught_by_cat[c] = rows[:n_taught]
        heldout_q += [(q, c) for q in rows[n_taught:]]
        taught_q += [(m, c) for m in rows[:n_taught]]
    return taught_by_cat, heldout_q, taught_q


def run_seed(seed, stories, K, hubs, vocab, probe_rows, probe_labels, args):
    rng = np.random.RandomState(seed)

    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    codes_scr, _ = learn_stream_codes(seed, stories, vocab, hubs, scramble=True, window=WINDOW)

    # group probe rows by category; keep categories with >= 4 members (>=2 taught + >=2 held-out)
    cats = {}
    for r, lab in zip(probe_rows.tolist(), probe_labels.tolist()):
        cats.setdefault(lab, []).append(r)
    usable = {c: rs for c, rs in cats.items() if len(rs) >= 4}
    if len(usable) < 2:
        return None
    cat_ids = sorted(usable.keys())
    P = rng.randn(len(cat_ids), 64)         # DISTINCT property vector per category (near-orthogonal)

    # TRUE grouping
    taught_by_cat, heldout_q, taught_q = _build_splits(usable, cat_ids, rng)
    ho_acc, n_ho = _inherit_accuracy(codes, taught_by_cat, heldout_q, P, cat_ids)
    mem_acc, _ = _inherit_accuracy(codes, taught_by_cat, taught_q, P, cat_ids)
    # SCRAMBLED-corpus (secondary control; weak for bag-of-words category structure -- noted)
    scr_acc, _ = _inherit_accuracy(codes_scr, taught_by_cat, heldout_q, P, cat_ids)

    # PRIMARY anti-cheat: LABEL-DERANGEMENT -- reassign the SAME probe words to RANDOM categories, keeping
    # the same per-category sizes, then teach+test on the REAL codes. If inheritance rides GENUINE discovered
    # category structure, a held-out member of a RANDOM group cannot classify to it -> collapse to chance.
    all_rows = [r for rs in usable.values() for r in rs]
    der_accs = []
    for _ in range(args.n_derange):
        pool = list(all_rows); rng.shuffle(pool)
        deranged = {}
        i = 0
        for c in cat_ids:
            k = len(usable[c]); deranged[c] = pool[i:i + k]; i += k
        d_taught, d_held, _ = _build_splits(deranged, cat_ids, rng)
        da, _ = _inherit_accuracy(codes, d_taught, d_held, P, cat_ids)
        der_accs.append(da)
    der_acc = float(np.mean(der_accs))
    chance = 1.0 / len(cat_ids)

    return {
        "seed": seed, "n_categories_used": len(cat_ids),
        "heldout_inherit_acc": ho_acc,        # HELD-OUT members inherit their OWN category's property
        "deranged_label_acc": der_acc,        # PRIMARY anti-cheat: random grouping -> collapse to chance
        "scrambled_heldout_acc": scr_acc,     # secondary control (weak for bag-of-words structure)
        "memorization_ceiling_acc": mem_acc,  # taught members (ceiling)
        "chance": chance, "n_heldout": n_ho,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="held-out inheritance accuracy must exceed chance AND deranged-labels by this margin (GO bar)")
    ap.add_argument("--n-derange", type=int, default=5, help="label-derangement repeats (averaged)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[realcorpus-inheritance rung1] corpus={a.corpus_path} stories={len(stories)} "
          f"tokens={sum(len(s) for s in stories)} K={a.K}", flush=True)

    vocab, gfreq = discover_vocab(stories, a.K)
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    probe_rows, probe_labels, probe_words, n_cat, per_cat = build_probe(vocab, TAXONOMY_8x8)
    print(f"  discovered vocab={len(vocab)} hubs={len(hubs)} | probe={probe_rows.size}w/{n_cat}cat "
          f"per_cat={per_cat}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, hubs, vocab, probe_rows, probe_labels, a)
        if r is None:
            print(f"  [seed {s}] too few usable categories (need >=2 with >=4 members) -- skip", flush=True)
            continue
        recs.append(r)
        print(f"  [seed {s}] HELD-OUT inherit-acc={r['heldout_inherit_acc']:.3f} | "
              f"DERANGED-labels={r['deranged_label_acc']:.3f} | scrambled={r['scrambled_heldout_acc']:.3f} | "
              f"mem-ceiling={r['memorization_ceiling_acc']:.3f} | chance={r['chance']:.3f} "
              f"(n_ho={r['n_heldout']}, cats={r['n_categories_used']})", flush=True)

    if not recs:
        print("  VERDICT: NOT-EVALUABLE (no usable categories at this K/probe)", flush=True)
        if a.out:
            json.dump({"verdict": "NOT-EVALUABLE", "per_seed": recs}, open(a.out, "w"), indent=2)
        return

    def m(k): return float(np.mean([r[k] for r in recs]))
    ho, der, scr, ch = (m("heldout_inherit_acc"), m("deranged_label_acc"),
                        m("scrambled_heldout_acc"), m("chance"))
    beats_chance = all(r["heldout_inherit_acc"] - r["chance"] > a.margin for r in recs)
    beats_deranged = all(r["heldout_inherit_acc"] - r["deranged_label_acc"] > a.margin for r in recs)
    go = beats_chance and beats_deranged
    verdict = "GO" if go else "NEGATIVE"
    print(f"\n  AGGREGATE ({len(recs)} seeds): HELD-OUT inherit-acc={ho:.3f} | DERANGED-labels={der:.3f} "
          f"(primary control) | scrambled={scr:.3f} (secondary) | mem-ceiling={m('memorization_ceiling_acc'):.3f} "
          f"| chance={ch:.3f}", flush=True)
    print(f"  beats_chance (all seeds, margin {a.margin})={beats_chance} | "
          f"beats_deranged_labels={beats_deranged}", flush=True)
    _inh_msg = 'INHERITS ITS OWN category property' if go else 'does NOT clearly inherit'
    _ctrl_msg = ('above chance AND the LABEL-DERANGEMENT control (random grouping collapses it) -> the GENUINE '
                 'discovered category carries the inheritance; the KNOWLEDGE half rides real-corpus breadth'
                 if go else 'within control range')
    print(f"  VERDICT: {verdict} -- a HELD-OUT member of a REAL-corpus-DISCOVERED category "
          f"{_inh_msg} (argmax over per-category properties), {_ctrl_msg}.", flush=True)
    if a.out:
        json.dump({"verdict": verdict, "K": a.K,
                   "aggregate": {"heldout_acc": ho, "deranged_acc": der, "scrambled_acc": scr, "chance": ch},
                   "beats_chance": beats_chance, "beats_deranged": beats_deranged, "per_seed": recs},
                  open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
