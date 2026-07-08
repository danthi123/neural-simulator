"""Train a broad-vocab concept-pool A->W on the breadth reasoner's OWN discovered vocab (TinyStories
animals + verbs), so the full pipeline speaks the reasoner's actual words ("the fish can run") on
spikes -- not just the v16 overlap.

*** BLOCKED (2026-07-08, root-caused engineering limitation, NOT a mechanism wall) ***
The concept-pool A->W architecture is VOCAB-WELDED: (1) the naive per-word train (no topographic bias)
gives ~random A->W (1/16 spell); (2) `apply_concept_topographic_bias` (the A->W enabler) is welded to
the concept-pool's specific 16-word vocab -- it iterates ALL pool words ('north'/'apple'/... ) and
requires each in `word_to_idx`, so a custom breadth vocab raises KeyError. A clean broad-vocab A->W
needs DECOUPLING the vocab from the concept-pool functions (a real refactor of concept_pool_demo), or
editing its shared NOUN_VOCAB/VERB_VOCAB (affects all concept-pool runners). Documented follow-on.
The DELIVERED fluent-speech capability is the v16-DEFAULT path (full-frame speech GO + the complete
talkable loop GO, over the v16 vocab which includes dog/cat) -- see the 2026-07-08 findings.
Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import os
import time

from research.runners.concept_pool_demo import (
    build_concept_bridge, train_word_to_pool, apply_concept_topographic_bias,
)

# the breadth vocab: 4 animals + 4 verbs + the 2 FUNCTION words (the/can) so the WHOLE frame speaks on
# spikes. noun/verb/adjective pools all get the (decoupled) topographic bias + reciprocal readout.
ANIMALS = ["dog", "cat", "bird", "fish"]
VERBS = ["run", "jump", "walk", "eat"]
FUNCTION = ["the", "can"]
VOCAB = ANIMALS + VERBS + FUNCTION
POOLS = ([f"noun_pool_{n}" for n in ("APPLE", "RIVER", "DOG", "CAT")]
         + [f"verb_pool_{n}" for n in ("GO", "COME", "STOP", "LOOK")]
         + [f"adjective_pool_{n}" for n in ("BIG", "SMALL")])   # the -> adj_BIG, can -> adj_SMALL
WORD_TO_POOL = dict(zip(VOCAB, POOLS))
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
N_LANG = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--out-dir", default="bridges/breadth_aw")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"seed{a.seed}.simstate.h5")

    print(f"[breadth A->W train] {len(VOCAB)} words: {VOCAB}", flush=True)
    bridge = build_concept_bridge(seed=a.seed, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                  weak_dynamics=True, enable_adjective=True,
                                  enable_direct_verb_to_motor=True, verbose=False)
    # the v16 recipe's KEY enabler: topographic bias on lang_input->pool (+ reciprocal pool->lang_output),
    # with the breadth vocab's orthogonal codes -- WITHOUT this the A->W spelling is ~random (root-caused).
    apply_concept_topographic_bias(bridge, n_lang_input=N_LANG, topographic_factor=3.0, off_target_factor=0.3,
                                   sparsity=0.05, orthogonal_codes=True, n_words_for_orthogonal=len(VOCAB),
                                   word_to_idx=WORD_TO_IDX, word_to_pool_override=WORD_TO_POOL, verbose=False)
    t0 = time.time()
    for w in VOCAB:
        tw = time.time()
        train_word_to_pool(bridge, w, WORD_TO_POOL[w], n_events=a.n_events,
                           n_lang_input=N_LANG, n_lang_output=N_LANG, sparsity=0.05,
                           orthogonal_codes=True, n_words_for_orthogonal=len(VOCAB),
                           word_to_idx=WORD_TO_IDX, verbose=False)
        print(f"  trained '{w}' -> {WORD_TO_POOL[w]} ({time.time()-tw:.0f}s)", flush=True)
    bridge.save_checkpoint(out)
    print(f"[breadth A->W train] complete ({time.time()-t0:.0f}s); saved {out}", flush=True)


if __name__ == "__main__":
    main()
