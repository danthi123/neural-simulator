"""RELATIONAL/SVO question-answering over the brain's OWN real-corpus codes (6-seed de-risk): STORE a set
of subject-verb-object facts, then ANSWER "what did <subject> <verb>?" by matching the cue against the
stored facts and reading the object -- with the no-confab MOAT abstaining on an UNSTORED relation.

Builds on the SVO compose PROBE (6-seed GO: real-corpus co-occurrence codes bind/unbind cleanly). This is
the actual capability: a persistent multi-fact relational memory + query-by-cue + moat, the who/what+moat
pattern (the project's RFPhasorComposer design) over the brain's DISCOVERED concept codes.

Mechanism (self-contained numpy FHRR; reuse `_phasors`/`_role`):
  * each concept's real-corpus code -> a unit phasor (fixed random complex projection).
  * roles AGENT/VERB/PATIENT = fixed random unit phasors.
  * STORE each fact as a bound vector f = AGENT*z_s + VERB*z_v + PATIENT*z_o (kept in a list).
  * ANSWER "what did <subj> <verb>?": scan the stored facts, find the one whose agent-slot cleans up to
    <subj> AND verb-slot to <verb>, read its patient-slot -> the object. NO stored (s,v,o) label is read
    (recovery is purely by unbind+cleanup); the labels are ground-truth for scoring only.
  * MOAT: an UNSTORED (subj,verb) matches NO fact -> "I don't know" (gate-first, no-confab).
Gates (6-seed(-blind)): ANSWER (stored -> correct object) >> chance AND >> permuted; MOAT (unstored ->
abstain) = 1.0. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._realcorpus_svo_compose_probe import _phasors, _role, D
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi

MATCH_MARGIN = 0.25          # min cleanup margin for a slot to count as a confident match (moat threshold)


class SVOStore:
    """A persistent relational store over real-corpus codes: bind SVO facts, answer by cue, abstain on miss."""

    def __init__(self, Z, rows, roles):
        self.Z = Z; self.rows = rows
        self.AGENT, self.VERB, self.PATIENT = roles
        self.facts = []                                # list of bound vectors

    def store(self, s, v, o):
        self.facts.append(self.AGENT * self.Z[s] + self.VERB * self.Z[v] + self.PATIENT * self.Z[o])

    def _cleanup(self, est):
        scores = np.array([np.real(np.vdot(self.Z[r], est)) for r in self.rows]) / D
        i = int(np.argmax(scores))
        srt = np.sort(scores)
        return self.rows[i], float(srt[-1] - srt[-2])

    def answer_patient(self, subj, verb):
        """'what did <subj> <verb>?' -> the object, or None (moat) if no stored fact matches the cue."""
        best_o, best_score = None, -1.0
        for f in self.facts:
            a, ma = self._cleanup(f * np.conj(self.AGENT))
            vb, mv = self._cleanup(f * np.conj(self.VERB))
            if a == subj and vb == verb and ma > MATCH_MARGIN and mv > MATCH_MARGIN:
                o, mo = self._cleanup(f * np.conj(self.PATIENT))
                if mo > best_score:
                    best_o, best_score = o, mo
        return best_o                                   # None -> abstain (no-confab moat)

    def contains(self, subj, verb, obj):
        """Verify a SPECIFIC fact: is '<subj> <verb> <obj>' stored? (all THREE roles clean up to the cue in
        ONE stored fact). The correct yes/no over MANY-TO-MANY facts (a subject with several objects)."""
        for f in self.facts:
            a, ma = self._cleanup(f * np.conj(self.AGENT))
            vb, mv = self._cleanup(f * np.conj(self.VERB))
            o, mo = self._cleanup(f * np.conj(self.PATIENT))
            if a == subj and vb == verb and o == obj and min(ma, mv, mo) > MATCH_MARGIN:
                return True
        return False

    def answer_agent(self, verb, obj):
        """'who <verb> <obj>?' -> the subject (agent), or None (moat) if no stored fact matches the cue.
        The reverse query -- unbind the AGENT slot after matching verb + patient."""
        best_a, best_score = None, -1.0
        for f in self.facts:
            vb, mv = self._cleanup(f * np.conj(self.VERB))
            o, mo = self._cleanup(f * np.conj(self.PATIENT))
            if vb == verb and o == obj and mv > MATCH_MARGIN and mo > MATCH_MARGIN:
                a, ma = self._cleanup(f * np.conj(self.AGENT))
                if ma > best_score:
                    best_a, best_score = a, ma
        return best_a                                   # None -> abstain (no-confab moat)


def run_seed(seed, stories, K, n_facts=12):
    vocab, gfreq = discover_vocab(stories, K)
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)
    rows = list(range(len(vocab)))
    Z = _phasors(codes, rows, seed)
    roles = (_role(rng), _role(rng), _role(rng))

    idx = list(rows); rng.shuffle(idx)
    triples = [(idx[i], idx[i + 1], idx[i + 2]) for i in range(0, n_facts * 3, 3)]
    store = SVOStore(Z, rows, roles)
    for (s, v, o) in triples:
        store.store(s, v, o)

    # ANSWER: each stored (subj,verb) -> its object
    ans_correct = sum(int(store.answer_patient(s, v) == o) for (s, v, o) in triples)
    ans_acc = ans_correct / len(triples)

    # MOAT: an UNSTORED (subj,verb) -> abstain. Build cues from concepts NOT used as (subj with that verb):
    used = {(s, v) for (s, v, o) in triples}
    unstored, tries = [], 0
    while len(unstored) < len(triples) and tries < 500:
        s, v = rng.choice(rows), rng.choice(rows)
        if (int(s), int(v)) not in used:
            unstored.append((int(s), int(v)))
        tries += 1
    moat_abstain = sum(int(store.answer_patient(s, v) is None) for (s, v) in unstored)
    moat = moat_abstain / max(1, len(unstored))

    # PERMUTED anti-cheat: ask with the correct subj but a WRONG verb (not paired with it) -> should abstain/miss
    perm_correct = 0
    for (s, v, o) in triples:
        wrong_v = rng.choice([vv for (_, vv, _) in triples if vv != v] or [v])
        perm_correct += int(store.answer_patient(s, wrong_v) == o)
    perm_acc = perm_correct / len(triples)

    return {"seed": seed, "n_facts": len(triples), "n_vocab": len(vocab), "chance": 1.0 / len(vocab),
            "answer_acc": ans_acc, "moat_abstain": moat, "permuted_acc": perm_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[SVO relational QA] corpus={a.corpus_path} K={a.K} n_facts={a.n_facts}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.n_facts)
        recs.append(r)
        print(f"  [seed {s}] answer_acc={r['answer_acc']:.3f} | MOAT abstain={r['moat_abstain']:.3f} | "
              f"permuted={r['permuted_acc']:.3f} | chance={r['chance']:.4f} (V={r['n_vocab']})", flush=True)

    def m(k): return float(np.mean([r[k] for r in recs]))
    ans, moat, perm = m("answer_acc"), m("moat_abstain"), m("permuted_acc")
    ans_ok = all(r["answer_acc"] > 0.75 for r in recs)
    moat_ok = all(r["moat_abstain"] > 0.9 for r in recs)
    perm_ok = all(r["answer_acc"] - r["permuted_acc"] > 0.4 for r in recs)
    go = ans_ok and moat_ok and perm_ok
    print(f"\n  AGGREGATE ({len(recs)} seeds): answer_acc={ans:.3f} | MOAT abstain={moat:.3f} | permuted={perm:.3f}", flush=True)
    print(f"  answer>0.75 all={ans_ok} | moat>0.9 all={moat_ok} | beats_permuted all={perm_ok}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- the brain {'STORES + ANSWERS relational SVO facts over its OWN '
             'real-corpus codes (what-did-X-verb -> the object), and ABSTAINS on an unstored relation (no-confab moat)'
             if go else 'does NOT cleanly answer relational SVO'}.", flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "aggregate": {"answer": ans, "moat": moat, "permuted": perm},
                   "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
