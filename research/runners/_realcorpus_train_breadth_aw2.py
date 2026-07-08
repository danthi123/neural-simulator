"""Train a SECOND breadth-vocab concept-pool A->W (BRIDGE-2) over 15 more real-corpus-discovered nouns
(objects + characters, not just animals), so the talkable brain can SPEAK a broader vocab -- the
EMERGE-68 two-bridge dispatch route (one concept-pool bridge caps at 16 words).

BRIDGE-1 (`_realcorpus_train_breadth_aw`, bridges/breadth_aw/seed42.simstate.h5): 8 animals + 6 verbs +
the/can. BRIDGE-2 (this): 15 common object/character nouns present in the TinyStories top-256. The two
bridges are dispatched by word (UnifiedNeuralSpell / EMERGE-68) so the console speaks ~23 nouns + verbs
on spikes instead of 8 animals. Same recipe as BRIDGE-1 (decoupled topographic bias + per-word train).
NO `sim/` edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import os
import time

from research.runners.concept_pool_demo import (
    build_concept_bridge, train_word_to_pool, apply_concept_topographic_bias,
)

# 15 object/character nouns discovered in the TinyStories top-256 (verified present, disjoint from BRIDGE-1)
VOCAB = ["ball", "tree", "box", "sun", "cake", "toy", "car", "house", "door", "rock", "boat",
         "girl", "boy", "mom", "dad"]
POOLS = ([f"noun_pool_{n}" for n in ("APPLE", "RIVER", "DOG", "CAT")]        # 4
         + [f"adjective_pool_{n}" for n in ("BIG", "SMALL", "HOT", "COLD")]  # 4
         + [f"verb_pool_{n}" for n in ("GO", "COME", "STOP", "LOOK")]        # 4
         + [f"motor_{a}" for a in ("N", "E", "S")])                          # 3  (15 pools for 15 words)
WORD_TO_POOL = dict(zip(VOCAB, POOLS))
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
N_LANG = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--out-dir", default="bridges/breadth_aw2")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"seed{a.seed}.simstate.h5")

    print(f"[breadth A->W BRIDGE-2 train] {len(VOCAB)} words: {VOCAB}", flush=True)
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
    print(f"[breadth A->W BRIDGE-2 train] complete ({time.time()-t0:.0f}s); saved {out}", flush=True)


if __name__ == "__main__":
    main()
