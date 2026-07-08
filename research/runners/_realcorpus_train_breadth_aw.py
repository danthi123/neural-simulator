"""Train a broad-vocab concept-pool A->W on the breadth reasoner's OWN discovered vocab (TinyStories
animals + verbs), so the full pipeline speaks the reasoner's actual words ("the fish can run") on
spikes -- not just the v16 overlap. ~17 min GPU. Saves bridges/breadth_aw/seed<N>.simstate.h5.

16 words -> the 16 concept pools: 8 animals -> noun+adj pools, 8 verbs -> verb+motor pools. Orthogonal
codes indexed in this vocab (so the A->W decode is well-defined). Reuse-by-import. NO sim/ edit.
Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import os
import time

from research.runners.concept_pool_demo import build_concept_bridge, train_word_to_pool

# the breadth vocab (TinyStories-frequent animals + verbs the emergent reasoner discovers)
ANIMALS = ["dog", "cat", "bird", "fish", "frog", "bear", "mouse", "duck"]
VERBS = ["run", "jump", "walk", "eat", "sleep", "play", "look", "sing"]
VOCAB = ANIMALS + VERBS
# 8 animals -> 4 noun + 4 adjective pools; 8 verbs -> 4 verb + 4 motor pools
POOLS = ([f"noun_pool_{n}" for n in ("APPLE", "RIVER", "DOG", "CAT")]
         + [f"adjective_pool_{n}" for n in ("BIG", "SMALL", "HOT", "COLD")]
         + [f"verb_pool_{n}" for n in ("GO", "COME", "STOP", "LOOK")]
         + [f"motor_{a}" for a in ("N", "E", "S", "W")])
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
