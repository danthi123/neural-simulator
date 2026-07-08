"""Train a THIRD breadth-vocab concept-pool A->W (BRIDGE-3) over the 3sg VERB SURFACES (eats/chases/...)
+ "a" + broader animal subjects, so the talkable brain can SPEAK its RELATIONAL answer ("the dog chases
the cat") FULLY ON SPIKES -- the C_TRANS slot order is EMERGE-74-proven; the residual is spelling the 3sg
verb form, which BRIDGE-1/2 do not cover (both at 16-word capacity). EMERGE-68 multi-bridge dispatch route.

Frequent inflected forms are lexically stored (Pinker's words-and-rules: regular high-frequency 3sg forms
are memorized whole, not always rule-derived), so a dedicated A->W bridge for the core-SVO 3sg surfaces is
biologically defensible. BRIDGE-1: 8 animals + 6 bare verbs + the/can. BRIDGE-2: 15 object nouns. BRIDGE-3
(this): 9 core 3sg verb surfaces + "a" + 6 more animal subjects. Same recipe as BRIDGE-1/2 (decoupled
topographic bias + per-word train). NO `sim/` edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import os
import time

from research.runners.concept_pool_demo import (
    build_concept_bridge, train_word_to_pool, apply_concept_topographic_bias,
)

# 9 core-SVO 3sg verb surfaces (base VERB_NORM present-tense + -s/-es) + "a" + 6 broader animal subjects.
# The 3sg surfaces the console's relational answer needs (VERB_NORM bases: see/eat/chase/hug/like/want/find/hold/kick).
VOCAB = ["eats", "chases", "sees", "likes", "wants", "hugs", "finds", "holds", "kicks",   # 9 3sg verbs
         "a", "wolf", "rabbit", "lion", "fox", "mouse", "owl"]                             # a + 6 subjects (16 total)
POOLS = ([f"noun_pool_{n}" for n in ("APPLE", "RIVER", "DOG", "CAT")]        # 4
         + [f"adjective_pool_{n}" for n in ("BIG", "SMALL", "HOT", "COLD")]  # 4
         + [f"verb_pool_{n}" for n in ("GO", "COME", "STOP", "LOOK")]        # 4
         + [f"motor_{a}" for a in ("N", "E", "S", "W")])                     # 4  (16 pools for 16 words)
WORD_TO_POOL = dict(zip(VOCAB, POOLS))
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
N_LANG = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--out-dir", default="bridges/breadth_aw3")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"seed{a.seed}.simstate.h5")

    print(f"[breadth A->W BRIDGE-3 train] {len(VOCAB)} words: {VOCAB}", flush=True)
    bridge = build_concept_bridge(seed=a.seed, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                  weak_dynamics=True, enable_adjective=True,
                                  enable_direct_verb_to_motor=True, verbose=False)
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
    print(f"[breadth A->W BRIDGE-3 train] complete ({time.time()-t0:.0f}s); saved {out}", flush=True)


if __name__ == "__main__":
    main()
