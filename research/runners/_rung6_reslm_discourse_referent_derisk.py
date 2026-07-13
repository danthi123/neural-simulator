"""RUNG 6 (cheap-first) of the open-generation ladder -- EMERGENT DISCOURSE COHERENCE: does the reslm GENERATOR's OWN
reservoir state carry the discourse referent, so a recurring subject can be produced as a pronoun that RESOLVES to its
antecedent -- emergently, from a training stream, NOT a hand-wired discourse module?

WHY (the emergence bar, not whack-a-mole): the project ALREADY has cross-sentence coherence, but on the COMPOSER/VSA
SCAFFOLD (`2026-06-17-cross-sentence-coherence-derisk.md`: a hand-wired referent buffer + pronominalize + `referent_at`
spiking unbind). Per the emergence bar that is a scaffold to REPLACE. Rung 6's mission-critical version asks whether
discourse coherence EMERGES in the learned reslm generator: after "SUBJ VERB . it", is the ANTECEDENT (which subject
"it" is) DECODABLE FROM THE RESERVOIR'S OWN STATE -- i.e. does the generator track who we're talking about, the way an
LM learns coreference from text? This is the reslm-generation analog of the D3 referent-tracking task
(`2026-07-09-D3-language-reference-tracking-GO.md`), which established that a FIXED reservoir FADES (EMERGE-83) while a
DISCRETE ATTRACTOR tracks unbounded referents. So the honest cheap-first question: does the reslm reservoir track the
referent at SHORT range (within its fading-memory window), and where does it need the D3 attractor?

THE TASK (a 2-clause possession/agent discourse over the 16 A->W words): "SUBJ VERB1 . it VERB2" where the pronoun "it"
refers to SUBJ. The reslm reads the clause stream; after the token "it" its reservoir state S is read by a trained
read-out to RECOVER the antecedent SUBJ (one of 8). Referent-tracking accuracy = does the reservoir carry who "it" is.
A `--gap G` inserts G distractor tokens between clause 1 and "it" (a DISTRACTOR intervening clause with a DIFFERENT
subject) -> sweeps the fading-memory range (the reservoir should hold short gaps, fade at long -> the D3-attractor need).

ANTI-CHEATS (the reservoir must BEAT all):
  - MEMORYLESS BAG: a bag-of-prefix read-out over the same positions (no order, no recurrent memory) -> cannot bind
    "it" to a specific earlier subject beyond frequency -> ~chance. (The reservoir's recurrent state must beat it.)
  - SHUFFLED-ANTECEDENT: randomize which subject is the true antecedent (break the SUBJ<->it link) -> collapses to
    chance (proves the read tracks the REAL referent, not a positional artifact).
  - CHANCE = 1/8 = 0.125 (8 subjects).
GATE (>=3, ideally 6 seeds): reservoir referent-track >> bag AND >> shuffled AND >> chance at gap=0; report the
gap-sweep (where the reservoir fades = the D3-attractor boundary). numpy-CPU (single run, contention-safe).

Run:
  SIM_BACKEND=numpy python -m research.runners._rung6_reslm_discourse_referent_derisk --seed 42 --gap 0
  SIM_BACKEND=numpy python -m research.runners._rung6_reslm_discourse_referent_derisk --seeds 42 43 44 100 101 102 --gap 0
Reuse-by-import; NO `sim/` edit.
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time

import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, ReservoirStates, _standardize_fit
from research.runners._emerge67_neural_spell_wirein_derisk import _AW_SUBJECTS, _AW_VERBS, _AW_CONTENT

_IT = "it"
_DOT = "."
_VOCAB_WORDS = list(_AW_CONTENT) + [_IT, _DOT]


def _make_discourse(rng, n, gap, shuffle_antecedent=False):
    """Each item: a token stream 'SUBJ VERB1 [. DSUBJ DVERB]*gap . it' + the TRUE antecedent (SUBJ, or a shuffled one).
    The read target is the antecedent subject-id; the read is taken from the reservoir state AFTER the final 'it'."""
    items = []
    ants = []
    for _ in range(n):
        subj = _AW_SUBJECTS[rng.integers(len(_AW_SUBJECTS))]
        v1 = _AW_VERBS[rng.integers(len(_AW_VERBS))]
        toks = [subj, v1]
        for _g in range(gap):                                 # distractor intervening clause(s) with a DIFFERENT subject
            d = _AW_SUBJECTS[rng.integers(len(_AW_SUBJECTS))]
            dv = _AW_VERBS[rng.integers(len(_AW_VERBS))]
            toks += [_DOT, d, dv]
        toks += [_DOT, _IT]
        items.append(toks)
        ants.append(subj)
    if shuffle_antecedent:                                    # ANTI-CHEAT: break the SUBJ<->it link
        perm = rng.permutation(len(ants)); ants = [ants[i] for i in perm]
    return items, ants


def _state_after_it(res, vocab, items):
    """Reservoir state (running-cumulative feature) at the FINAL token ('it') of each stream."""
    feats = []
    for toks in items:
        S = res.per_token_states(vocab.encode_seq(toks))
        feats.append(np.asarray(S[-1]))                       # state after 'it'
    return np.asarray(feats)


def _bag_after_it(vocab, items):
    """MEMORYLESS control: normalized bag-of-token-ids over the whole stream (no order, no recurrence)."""
    V = vocab.size; out = []
    for toks in items:
        v = np.zeros(V)
        for w in toks:
            v[vocab.id(w)] += 1.0
        out.append(v / max(1, len(toks)))
    return np.asarray(out)


def _fit_readout(X, y, n_cls, l2=1.0):
    """Ridge multiclass (closed-form) antecedent read-out -> predict subject-id from the feature."""
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.eye(n_cls)[y]
    W = np.linalg.solve(Xa.T @ Xa + l2 * np.eye(Xa.shape[1]), Xa.T @ Y)
    return W


def _acc(W, X, y):
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    return float(np.mean(np.argmax(Xa @ W, 1) == y))


def run(seed, gap, n_train=480, n_eval=160, n_pool=160):
    rng = np.random.default_rng(seed)
    vocab = Vocab(list(_VOCAB_WORDS))
    res = ReservoirStates(in_dim=vocab.size, seed=seed, n=n_pool)
    s2i = {s: i for i, s in enumerate(_AW_SUBJECTS)}
    tr_items, tr_ants = _make_discourse(rng, n_train, gap)
    ev_items, ev_ants = _make_discourse(rng, n_eval, gap)
    ytr = np.array([s2i[a] for a in tr_ants]); yev = np.array([s2i[a] for a in ev_ants])
    # RESERVOIR referent-track
    Xtr = _state_after_it(res, vocab, tr_items); Xev = _state_after_it(res, vocab, ev_items)
    mean, std = _standardize_fit([(Xtr, None)]) if False else (Xtr.mean(0), Xtr.std(0) + 1e-6)
    W = _fit_readout((Xtr - mean) / std, ytr, len(_AW_SUBJECTS)); res_acc = _acc(W, (Xev - mean) / std, yev)
    # BAG control
    Btr = _bag_after_it(vocab, tr_items); Bev = _bag_after_it(vocab, ev_items)
    Wb = _fit_readout(Btr, ytr, len(_AW_SUBJECTS)); bag_acc = _acc(Wb, Bev, yev)
    # SHUFFLED-ANTECEDENT control (reservoir feature, broken link)
    sh_items, sh_ants = _make_discourse(np.random.default_rng(seed * 7 + 1), n_train, gap, shuffle_antecedent=True)
    yshtr = np.array([s2i[a] for a in sh_ants]); Xshtr = _state_after_it(res, vocab, sh_items)
    she_items, she_ants = _make_discourse(np.random.default_rng(seed * 7 + 2), n_eval, gap, shuffle_antecedent=True)
    yshev = np.array([s2i[a] for a in she_ants]); Xshev = _state_after_it(res, vocab, she_items)
    shm, shs = Xshtr.mean(0), Xshtr.std(0) + 1e-6
    Wsh = _fit_readout((Xshtr - shm) / shs, yshtr, len(_AW_SUBJECTS)); sh_acc = _acc(Wsh, (Xshev - shm) / shs, yshev)
    go = (res_acc > 0.5) and (res_acc > bag_acc + 0.15) and (sh_acc < 0.30)
    print(f"[rung6 seed={seed} gap={gap}] reservoir_referent_track={res_acc:.3f}  bag={bag_acc:.3f}  "
          f"shuffled={sh_acc:.3f}  chance=0.125 -> {'GO' if go else 'no'}")
    return dict(seed=seed, gap=gap, reservoir=round(res_acc, 3), bag=round(bag_acc, 3),
                shuffled=round(sh_acc, 3), chance=0.125, go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    t0 = time.time()
    results = [run(s, a.gap) for s in seeds]
    if len(results) > 1:
        gos = sum(1 for r in results if r["go"]); print(f"[rung6 gap={a.gap}] {gos}/{len(results)} seeds GO")
    if a.out:
        json.dump(dict(results=results, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
