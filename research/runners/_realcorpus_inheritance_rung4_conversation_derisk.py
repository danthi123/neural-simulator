"""KNOWLEDGE-half of breadth, rung 4: TALK to the brain about a broad, REAL-corpus-discovered vocab.

The mission payoff of the breadth->knowledge arc. The corpus supplies the VOCAB + CATEGORIES (the
emergent breadth mechanism); the FACTS are TAUGHT explicitly (as EMERGE's inheritance consoles do,
NOT mined from a shallow corpus -- rung 3's caveat). Then the brain ANSWERS a yes/no question about a
HELD-OUT real word it was never told about, by inheriting the taught class property over the
discovered category -- with the no-confab MOAT abstaining on words outside the discovered vocab.

Flow (per property taught):
  * DISCOVER categories from TinyStories (breadth) -> the co-occurrence codes.
  * TEACH a class property to SOME members of a category ("dogs and cats can MOVE") -> a Hebbian
    associative memory binds those members' codes to the property tag.
  * ASK about a HELD-OUT member of that category ("can a FISH move?") -> YES, inherited via the shared
    category structure (fish was never taught).
  * ASK about a member of a DIFFERENT category ("can a HOUSE move?") -> NO (houses inherit the places
    property, not the taught move-property).
  * ASK about an UNKNOWN word ("can a ZZZ move?") -> "I don't know" (the moat: zzz is not in the
    discovered vocab, so there is no code to reason over).

Gates (6-seed): INHERIT-YES (held-out same-cat -> yes) AND REJECT-NO (other-cat -> no) clearly beat
chance; MOAT (unknown -> abstain) is 1.00 by construction; label-DERANGEMENT collapses inherit-yes.
Rate-level (reuse rung-1); the spiking realization is rung-2. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, build_probe, TAXONOMY_8x8,
    STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows
from research.runners.corpus_stream import load_token_stream_multi

CAT_NAMES = list(TAXONOMY_8x8.keys())


class RealCorpusConsole:
    """Discover categories from a real corpus; teach class properties explicitly; answer yes/no with a moat."""

    def __init__(self, seed, stories, K, emergent=False, n_clusters=10):
        vocab, gfreq = discover_vocab(stories, K)
        self.vocab = vocab
        self.row_of = {w: i for i, w in enumerate(vocab)}
        target_set = set(vocab)
        hubs = []
        for w, _ in gfreq.most_common():
            if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
                continue
            hubs.append(w)
            if len(hubs) >= N_HUB:
                break
        codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
        self.U = _unit_rows(codes)
        if emergent:
            # PROBE-FREE: DISCOVER categories by clustering the codes (no hand-labeled probe). Each cluster is
            # named by its most-central member word (legible). Categories are the brain's own groupings.
            from research.runners._realcorpus_inheritance_emergent_clusters_derisk import _kmeans
            labels = _kmeans(codes, n_clusters, seed)
            from collections import Counter
            cnt = Counter(labels.tolist())
            by_cat = {}
            for c in sorted(cnt):
                if cnt[c] < 4:
                    continue
                rows = [i for i in range(len(vocab)) if labels[i] == c]
                cent = self.U[rows].mean(0)
                central = vocab[rows[int(np.argmax(self.U[rows] @ cent))]]      # most-central word names the cluster
                by_cat[f"~{central}"] = [vocab[i] for i in rows]
            self.members = by_cat
        else:
            pr, pl, pw, nc, per = build_probe(vocab, TAXONOMY_8x8)
            self.cat_of_word = {vocab[int(r)]: CAT_NAMES[int(lab)] for r, lab in zip(pr.tolist(), pl.tolist())}
            by_cat = {}
            for w, c in self.cat_of_word.items():
                by_cat.setdefault(c, []).append(w)
            self.members = {c: ws for c, ws in by_cat.items() if len(ws) >= 4}
        self.rng = np.random.RandomState(seed)
        self.D = 64
        # multi-category argmax (the proven rung-1 mechanism): every usable category gets a DISTINCT property
        # tag; "does <word> have <the property taught to category C>?" == does the word's recalled property
        # argmax to C. This avoids the single-property scalar-of-P saturation (everything-says-yes).
        self.cat_ids = sorted(self.members.keys())
        self.P = {c: self.rng.randn(self.D) for c in self.cat_ids}
        self.taught_by_cat = {}     # populated by teach()
        self.M = None

    def teach(self, taught_by_cat):
        """Bind each category's property tag to ITS taught members' codes (one associative memory over all)."""
        self.taught_by_cat = taught_by_cat
        M = np.zeros((self.U.shape[1], self.D))
        for c, members in taught_by_cat.items():
            for w in members:
                if w in self.row_of:
                    M += np.outer(self.U[self.row_of[w]], self.P[c])
        self.M = M

    def _predict_cat(self, word):
        phat = self.U[self.row_of[word]] @ self.M
        scores = {c: float(self.P[c] @ phat) for c in self.cat_ids}
        return max(scores, key=scores.get)

    def ask(self, category, word):
        """'does <word> have <the property taught to `category`>?' -> yes / no / I-don't-know (moat)."""
        if word not in self.row_of:
            return "idk"                       # the no-confab moat: not in the discovered vocab
        return "yes" if self._predict_cat(word) == category else "no"


def _splits(members, cat_ids, rng):
    """taught (half) + held-out (half) per category."""
    taught_by_cat, held_by_cat = {}, {}
    for c in cat_ids:
        ws = list(members[c]); rng.shuffle(ws)
        n = max(2, len(ws) // 2)
        taught_by_cat[c] = ws[:n]; held_by_cat[c] = ws[n:]
    return taught_by_cat, held_by_cat


def _coherence(con, c):
    """mean within-category pairwise cosine of the members' codes (how tight the category is)."""
    rows = [con.row_of[w] for w in con.members[c] if w in con.row_of]
    if len(rows) < 2:
        return 0.0
    U = con.U[rows]
    S = U @ U.T
    iu = np.triu_indices(len(rows), 1)
    return float(S[iu].mean())


def run_seed(seed, stories, K, verbose=False, emergent=False, n_clusters=10):
    con = RealCorpusConsole(seed, stories, K, emergent=emergent, n_clusters=n_clusters)
    if len(con.cat_ids) < 2:
        return None
    # pick pos = the MOST COHERENT category (the one co-occurrence actually forms); neg = 2nd most coherent.
    coh = {c: _coherence(con, c) for c in con.cat_ids}
    cats_by_coh = sorted(con.cat_ids, key=lambda c: -coh[c])
    pos, neg = cats_by_coh[0], cats_by_coh[1]
    taught_by_cat, held_by_cat = _splits(con.members, con.cat_ids, con.rng)
    con.teach(taught_by_cat)                       # teach ALL categories their distinct property
    held = held_by_cat[pos]

    yes_correct = np.mean([con.ask(pos, w) == "yes" for w in held])              # held-out pos -> yes (inherit)
    no_correct = np.mean([con.ask(pos, w) == "no" for w in con.members[neg]])    # neg-cat -> no
    moat = con.ask(pos, "zzzqqx") == "idk"                                       # unknown -> I don't know

    # DERANGE anti-cheat: teach with SHUFFLED category labels -> held-out pos members no longer predict pos
    con_d = RealCorpusConsole(seed, stories, K, emergent=emergent, n_clusters=n_clusters)
    allw = [w for c in con_d.cat_ids for w in con_d.members[c]]
    con_d.rng.shuffle(allw)
    der_taught = {}; i = 0
    for c in con_d.cat_ids:
        n = len(taught_by_cat[c]); der_taught[c] = allw[i:i + n]; i += n
    con_d.teach(der_taught)
    yes_deranged = np.mean([con_d.ask(pos, w) == "yes" for w in held])

    if verbose:
        print(f"    TAUGHT (explicit facts): " + " ; ".join(f"{c}<-{taught_by_cat[c]}" for c in (pos, neg)))
        for w in held[:3]:
            print(f"    Q: does a '{w}' have the {pos} property? ({pos}, HELD-OUT) -> {con.ask(pos, w).upper()}")
        for w in con.members[neg][:2]:
            print(f"    Q: does a '{w}' have the {pos} property? ({neg}) -> {con.ask(pos, w).upper()}")
        print(f"    Q: does a 'zzzqqx' have the {pos} property? (unknown) -> {con.ask(pos,'zzzqqx').upper()}  [moat]")

    return {"seed": seed, "pos_cat": pos, "neg_cat": neg, "n_held": len(held),
            "pos_coherence": round(coh[pos], 3), "inherit_yes_acc": float(yes_correct),
            "reject_no_acc": float(no_correct), "moat_ok": bool(moat),
            "deranged_yes_acc": float(yes_deranged)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--demo", action="store_true", help="print a scripted conversation transcript (seed 42)")
    ap.add_argument("--emergent", action="store_true", help="PROBE-FREE: discover categories by clustering (no probe)")
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[rung4 conversation] corpus={a.corpus_path} stories={len(stories)} K={a.K} "
          f"{'EMERGENT-clusters (probe-free)' if a.emergent else 'probe'}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, verbose=(a.demo and s == seeds[0]), emergent=a.emergent, n_clusters=a.n_clusters)
        if r is None:
            continue
        recs.append(r)
        print(f"  [seed {s}] pos={r['pos_cat']}(coh {r['pos_coherence']})/neg={r['neg_cat']} | "
              f"INHERIT-yes {r['inherit_yes_acc']:.3f} | REJECT-no {r['reject_no_acc']:.3f} | "
              f"MOAT {int(r['moat_ok'])} | deranged-yes {r['deranged_yes_acc']:.3f}", flush=True)

    def m(k): return float(np.mean([r[k] for r in recs]))
    yes, no, der = m("inherit_yes_acc"), m("reject_no_acc"), m("deranged_yes_acc")
    moat = all(r["moat_ok"] for r in recs)
    yes_ok = all(r["inherit_yes_acc"] > 0.5 for r in recs)
    no_ok = all(r["reject_no_acc"] > 0.5 for r in recs)
    beats_der = all(r["inherit_yes_acc"] - r["deranged_yes_acc"] > 0.25 for r in recs)
    go = yes_ok and no_ok and moat and beats_der
    verdict = "GO" if go else "NEGATIVE"
    print(f"\n  AGGREGATE ({len(recs)} seeds): INHERIT-yes {yes:.3f} | REJECT-no {no:.3f} | "
          f"MOAT {int(moat)} | deranged-yes {der:.3f}", flush=True)
    print(f"  inherit>0.5 all={yes_ok} | reject>0.5 all={no_ok} | moat all={moat} | beats_deranged(>.25)={beats_der}",
          flush=True)
    _verdict_msg = ('ANSWERS a yes/no question about a HELD-OUT real word by inheriting a TAUGHT class property over '
                    'the REAL-corpus-discovered category, rejects other-category words, and ABSTAINS on unknown words '
                    '(no-confab moat)' if go else 'does NOT cleanly answer')
    print(f"  VERDICT: {verdict} -- the brain {_verdict_msg}.", flush=True)
    if a.out:
        json.dump({"verdict": verdict, "K": a.K,
                   "aggregate": {"inherit_yes": yes, "reject_no": no, "moat": moat, "deranged_yes": der},
                   "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
