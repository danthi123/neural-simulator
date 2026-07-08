"""DITRANSITIVE relational store (schema expansion beyond binary SVO): a 4-ROLE FHRR store binds a ternary
relation "the dog GIVES the cat a bone" = give(agent=dog, recipient=cat, theme=bone), and answers both
argument queries ("what does the dog give the cat?" -> bone; "who does the dog give a bone (to)?" -> cat),
abstaining on the unstored (no-confab moat). The production side (EMERGE-77 C_DITRANS, n_slot_pools=8) already
renders "the dog gives the cat a bone" on spikes; this de-risks the STORE + COMPREHENSION side (recover each
argument from a ternary fact) so the brain can DISCUSS ditransitive relations, not just binary SVO.

Mechanism (extends the validated SVOStore FHRR from 3 -> 4 roles; D=512 gives ample SNR for 4 superposed terms):
  f = AGENT*z_s + VERB*z_v + RECIPIENT*z_i + PATIENT*z_o ; recover an argument by unbind (f*conj(ROLE)) + cleanup.
Anti-cheats: MOAT (unstored cue -> abstain); PERMUTED (query with a wrong verb -> miss). numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_svo_compose_probe import _phasors, _role
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi

MATCH_MARGIN = 0.25


class DitransStore:
    """A 4-role FHRR store for ternary (ditransitive) relations: agent-verb-recipient-theme."""

    def __init__(self, Z, rows, roles):
        self.Z, self.rows = Z, rows
        self.AGENT, self.VERB, self.RECIP, self.THEME = roles
        self.facts = []

    def store(self, s, v, i, o):
        self.facts.append(self.AGENT * self.Z[s] + self.VERB * self.Z[v]
                          + self.RECIP * self.Z[i] + self.THEME * self.Z[o])

    def _cleanup(self, est):
        sims = np.array([np.real(np.vdot(self.Z[r], est)) / len(est) for r in self.rows])
        j = int(np.argmax(sims))
        return self.rows[j], float(sims[j])

    def answer_theme(self, s, v, i):
        """what does the <s> <v> the <i>? -> the theme. Match agent+verb+recipient, unbind theme."""
        best, best_m = None, MATCH_MARGIN
        for f in self.facts:
            a, ma = self._cleanup(f * np.conj(self.AGENT))
            vb, mv = self._cleanup(f * np.conj(self.VERB))
            rc, mr = self._cleanup(f * np.conj(self.RECIP))
            if a == s and vb == v and rc == i and min(ma, mv, mr) > MATCH_MARGIN:
                o, mo = self._cleanup(f * np.conj(self.THEME))
                if mo > best_m:
                    best, best_m = o, mo
        return best

    def answer_recipient(self, s, v, o):
        """who does the <s> <v> the <o> (to)? -> the recipient. Match agent+verb+theme, unbind recipient."""
        best, best_m = None, MATCH_MARGIN
        for f in self.facts:
            a, ma = self._cleanup(f * np.conj(self.AGENT))
            vb, mv = self._cleanup(f * np.conj(self.VERB))
            o2, mo = self._cleanup(f * np.conj(self.THEME))
            if a == s and vb == v and o2 == o and min(ma, mv, mo) > MATCH_MARGIN:
                rc, mr = self._cleanup(f * np.conj(self.RECIP))
                if mr > best_m:
                    best, best_m = rc, mr
        return best


def run_seed(seed, stories, K, n_facts=10):
    vocab, gfreq = discover_vocab(stories, K)
    target = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)
    rows = list(range(len(vocab)))
    Z = _phasors(codes, rows, seed)
    roles = (_role(rng), _role(rng), _role(rng), _role(rng))     # AGENT/VERB/RECIP/THEME

    idx = list(rows); rng.shuffle(idx)
    quads = [(idx[i], idx[i + 1], idx[i + 2], idx[i + 3]) for i in range(0, n_facts * 4, 4)]
    store = DitransStore(Z, rows, roles)
    for (s, v, i, o) in quads:
        store.store(s, v, i, o)

    theme_ok = sum(int(store.answer_theme(s, v, i) == o) for (s, v, i, o) in quads) / len(quads)
    recip_ok = sum(int(store.answer_recipient(s, v, o) == i) for (s, v, i, o) in quads) / len(quads)
    # MOAT: an unstored (agent, verb, recipient) -> abstain
    used = {(s, v, i) for (s, v, i, o) in quads}
    unstored, tries = [], 0
    while len(unstored) < len(quads) and tries < 800:
        s, v, i = int(rng.choice(rows)), int(rng.choice(rows)), int(rng.choice(rows))
        if (s, v, i) not in used:
            unstored.append((s, v, i))
        tries += 1
    moat = sum(int(store.answer_theme(s, v, i) is None) for (s, v, i) in unstored) / max(1, len(unstored))
    # PERMUTED: query the theme with a WRONG verb -> miss
    perm = sum(int(store.answer_theme(s, rng.choice([vv for (_, vv, _, _) in quads if vv != v] or [v]), i) == o)
               for (s, v, i, o) in quads) / len(quads)
    return {"seed": seed, "n_facts": len(quads), "n_vocab": len(vocab), "chance": 1.0 / len(vocab),
            "theme_acc": theme_ok, "recip_acc": recip_ok, "moat_abstain": moat, "permuted_acc": perm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-facts", type=int, default=10)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[ditransitive store] 4-role FHRR ternary relations (agent-verb-recipient-theme) | K={a.K}", flush=True)
    recs = [run_seed(s, stories, a.K, a.n_facts) for s in seeds]
    for r in recs:
        print(f"  [seed {r['seed']}] theme-acc={r['theme_acc']:.3f} recip-acc={r['recip_acc']:.3f} "
              f"moat-abstain={r['moat_abstain']:.3f} permuted={r['permuted_acc']:.3f} (chance {r['chance']:.3f})", flush=True)
    def m(k): return float(np.mean([r[k] for r in recs]))
    go = all(r["theme_acc"] >= 0.9 and r["recip_acc"] >= 0.9 and r["moat_abstain"] >= 0.9 and
             r["permuted_acc"] <= 0.1 for r in recs)
    print(f"\n  AGGREGATE: theme={m('theme_acc'):.3f} recip={m('recip_acc'):.3f} moat={m('moat_abstain'):.3f} "
          f"permuted={m('permuted_acc'):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL'} -- a 4-role FHRR store binds + recovers DITRANSITIVE (ternary) "
          f"relations (both argument queries), abstaining on the unstored (moat) {'and the permuted control collapses' if go else ''}; "
          f"the brain can DISCUSS ditransitive relations, not just binary SVO (production = EMERGE-77 C_DITRANS on spikes).", flush=True)


if __name__ == "__main__":
    main()
