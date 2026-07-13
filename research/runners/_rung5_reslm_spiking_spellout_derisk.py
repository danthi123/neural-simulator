"""RUNG 5 of the open-generation ladder -- OPEN-VOCAB SPIKING SPELL-OUT: wire the emergent reservoir-LM's next-token
prediction through the VALIDATED EMERGE-67 A->W spiking read-out, so the generator SPELLS its predicted token ON SPIKES
(decoded from `language_output` firing), not as a host token string.

WHY (the ladder): Rungs 1-4 are GO (emergent next-token beats bigram, WM-latch distal context, novel-subject
generalization, order-decisive recombination). The reslm's OUTPUT is still a host word string (`vocab.word(argmax(W@x))`).
Rung 5 makes that output SPIKING: feed each predicted token to EMERGE-67's `NeuralSpell.spell(word)` (drive the word's
concept pool on a real `SimulationBridge` -> accumulate `language_output` spikes -> cosine-decode the spoken word). This
is the EXPRESSIVENESS rung (spell what the generator already predicts, on spikes) -- NOT a scale rung, so the reservoir's
Ueda scale-ceiling (`2026-07-12-reslm-batched-scale-CONFOUND-FREE-...md`) does NOT block it. It COMPOSES two already-
LEARNED components (the reslm's learned next-token + the A->W's learned spelling) -- on the emergence ladder, not a
hand-built capability.

CHEAP-FIRST (the vocab-overlap crux): EMERGE-67's A->W spells a BOUNDED 16-word content vocab (owl/penguin/.../fly/swim/
walks/...), rebound onto the 16 validated concept pools + GPU-trained-once + cached. So Rung 5's cheap-first trains the
reslm on a tiny corpus over THOSE 16 words (subject->verb next-token) -> every predicted token is in the A->W vocab ->
every prediction is spike-spellable. "Open vocab" (V=200) is the named LEVER = more A->W bridges (EMERGE-68's multi-bridge
pattern), a scale/data follow-on, NOT this rung.

THE GATE (>=3, ideally 6 seeds 42/43/44/100/101/102):
  (a) RESLM PREDICTS (Rung-1 sanity, numpy-CPU): the reslm learns subject->verb; rollout next-token accuracy high, all
      predictions IN the A->W 16-word vocab.
  (b) SPIKE-SPELL FIDELITY (the Rung-5 claim, GPU A->W): for each predicted token w_pred, NeuralSpell.spell(w_pred)
      decodes w_pred from `language_output` spikes -- fidelity >= 0.90.
  (c) GENUINELY SPIKING (lesion): NeuralSpell(lesion_pool_out=True) (zero the pool->language_output pathway) collapses
      the decode (a host lookup would be unaffected).
  (d) END-TO-END: the composed (reslm predict -> brain spell) reproduces the correct verb surface on spikes.
GO bar: reslm next-token acc high + all-in-vocab (a); spike-spell fidelity >= 0.90 (b); lesion collapses (c); >=3 seeds.
Reuse-by-import; NO `sim/` edit.

Run:
  # gaming-safe reslm-side smoke (numpy-CPU, no GPU): does the reslm learn subject->verb + predict in-vocab?
  SIM_BACKEND=numpy python -m research.runners._rung5_reslm_spiking_spellout_derisk --reslm-only --seed 42
  # full Rung-5 de-risk (needs the cupy A->W engine cache from EMERGE-67):
  SIM_BACKEND=cupy python -m research.runners._rung5_reslm_spiking_spellout_derisk --derisk --seeds 42 43 44 100 101 102
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

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, ReservoirStates, train_readout, _cache, _standardize_fit,
)
# EMERGE-67's 16-word content vocab (the A->W-spellable set) -- import so the corpus + A->W agree exactly.
from research.runners._emerge67_neural_spell_wirein_derisk import _AW_SUBJECTS, _AW_VERBS, _AW_CONTENT

# A deterministic subject->verb next-token map over the 16 A->W words (8 subjects, 8 verbs, bijection).
SUBJ_VERB = {s: v for s, v in zip(_AW_SUBJECTS, _AW_VERBS)}   # owl->fly, penguin->swim, ... crow->rests


def _corpus(reps):
    """A tiny 'SUBJ VERB' next-token corpus over the 16 A->W words (subject->verb bijection, `reps` copies).
    Sentences are LISTS of word-tokens (the reslm's `vocab.encode_seq`/`ids` iterate `for w in s`; a joined string
    would iterate CHARACTERS -> all <unk>)."""
    sents = []
    for _ in range(reps):
        for s in _AW_SUBJECTS:
            sents.append([s, SUBJ_VERB[s]])
    return sents


def _reslm_predict(seed, n_pool=120, reps=40, epochs=12, lr=0.02):
    """Build a reslm over the 16-word corpus, train the read-out, roll out one next-token per subject.
    Returns (vocab, predictions dict subj->predicted_verb, next_token_acc, all_in_vocab)."""
    sents = _corpus(reps)
    vocab = Vocab(list(_AW_CONTENT))                      # FIXED 16-word vocab (== the A->W set)
    res = ReservoirStates(in_dim=vocab.size, seed=seed, n=n_pool)
    cache = _cache(res, vocab, sents)
    mean, std = _standardize_fit(cache)
    W = train_readout(cache, vocab.size, epochs, lr, np.random.default_rng(seed * 13 + 1), mean, std)
    preds = {}
    for s in _AW_SUBJECTS:
        toks = res.rollout(vocab, W, mean, std, seed_token=s, n_gen=1)
        preds[s] = toks[1] if len(toks) > 1 else None
    correct = sum(1 for s in _AW_SUBJECTS if preds[s] == SUBJ_VERB[s])
    acc = correct / len(_AW_SUBJECTS)
    in_vocab = all(preds[s] in _AW_CONTENT for s in _AW_SUBJECTS)
    return vocab, preds, acc, in_vocab


def run_reslm_only(seed):
    vocab, preds, acc, in_vocab = _reslm_predict(seed)
    print(f"[rung5 reslm-only seed={seed}] next-token acc={acc:.3f}  all-in-vocab={in_vocab}")
    for s in _AW_SUBJECTS:
        print(f"    {s:>8} -> pred={preds[s]:<6} (target {SUBJ_VERB[s]})  {'OK' if preds[s]==SUBJ_VERB[s] else 'x'}")
    return dict(seed=seed, next_token_acc=round(acc, 3), all_in_vocab=bool(in_vocab))


def run_derisk(seed):
    """Full Rung-5: reslm predicts -> A->W spells each prediction ON SPIKES -> gate. Needs the cupy A->W engine."""
    from research.runners._emerge67_neural_spell_wirein_derisk import NeuralSpell
    vocab, preds, acc, in_vocab = _reslm_predict(seed)
    ns = NeuralSpell(load=True)                            # cached A->W engine (GPU-trained once)
    ns_les = NeuralSpell(load=True, lesion_pool_out=True)  # lesion: zero pool->language_output
    n = len(_AW_SUBJECTS); ok = les_ok = e2e = 0
    for s in _AW_SUBJECTS:
        w = preds[s]
        if ns.spell(w) == w:                              # (b) spike-spell fidelity
            ok += 1
        if ns_les.spell(w) == w:                          # (c) lesion should NOT decode
            les_ok += 1
        if preds[s] == SUBJ_VERB[s] and ns.spell(w) == w:  # (d) end-to-end correct surface on spikes
            e2e += 1
    fid = ok / n; les_fid = les_ok / n; e2e_acc = e2e / n
    go = (acc >= 0.9) and in_vocab and (fid >= 0.9) and (les_fid <= 0.3)
    print(f"[rung5 derisk seed={seed}] reslm_acc={acc:.3f} in_vocab={in_vocab} | spike_spell_fid={fid:.3f} "
          f"lesion_fid={les_fid:.3f} e2e={e2e_acc:.3f} -> {'GO' if go else 'NO-GO'}")
    return dict(seed=seed, next_token_acc=round(acc, 3), all_in_vocab=bool(in_vocab),
                spike_spell_fidelity=round(fid, 3), lesion_fidelity=round(les_fid, 3),
                e2e_acc=round(e2e_acc, 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reslm-only", action="store_true", help="numpy-CPU reslm-side smoke (gaming-safe, no GPU)")
    ap.add_argument("--derisk", action="store_true", help="full Rung-5 (needs cupy A->W engine)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    fn = run_reslm_only if a.reslm_only else run_derisk
    t0 = time.time()
    results = [fn(s) for s in seeds]
    if a.derisk and len(results) > 1:
        gos = sum(1 for r in results if r.get("go"))
        print(f"[rung5 derisk] {gos}/{len(results)} seeds GO")
    if a.out:
        json.dump(dict(results=results, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
