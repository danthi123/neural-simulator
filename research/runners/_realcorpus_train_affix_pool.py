"""Train a MICRO concept-pool A->W over the BOUND MORPHEMES {s, es, ed, ing, ies} as spellable units -- the
affix half of PRODUCTIVE regular inflection ON SPIKES. Pinker-Ullman declarative/procedural: the procedural
system (Broca + basal ganglia) composes a productive inflection by concatenating a bound-morpheme AFFIX to a
lexically-retrieved STEM. Here the affix is its own spellable A->W pool, so a NOVEL 3sg ("wug" -> "wugs")
is produced as spell(stem) + spell("-s") ON SPIKES -- never a stored whole-form lexeme, never `emerge_v3`.

Same concept-pool recipe as BRIDGE-1/2/3 (decoupled topographic bias + per-word train). The affix set covers
3sg (-s / -es / -ies), past (-ed), and gerund (-ing). NO `sim/` edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import os
import time

from research.runners.concept_pool_demo import (
    build_concept_bridge, train_word_to_pool, apply_concept_topographic_bias,
)

# the English regular bound morphemes, as spellable units (surface graphemes).
VOCAB = ["s", "es", "ed", "ing", "ies"]
POOLS = [f"noun_pool_{n}" for n in ("APPLE", "RIVER", "DOG", "CAT")] + ["adjective_pool_BIG"]   # 5 pools for 5 affixes
WORD_TO_POOL = dict(zip(VOCAB, POOLS))
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
N_LANG = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--out-dir", default="bridges/affix_aw")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"seed{a.seed}.simstate.h5")

    print(f"[affix A->W train] {len(VOCAB)} bound morphemes: {VOCAB}", flush=True)
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
        print(f"  trained affix '{w}' -> {WORD_TO_POOL[w]} ({time.time()-tw:.0f}s)", flush=True)
    bridge.save_checkpoint(out)
    print(f"[affix A->W train] complete ({time.time()-t0:.0f}s); saved {out}", flush=True)


if __name__ == "__main__":
    main()
