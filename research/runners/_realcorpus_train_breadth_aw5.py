"""Train a FIFTH breadth-vocab concept-pool A->W (BRIDGE-5) over 16 more high-frequency conversational content
words -- descriptive adjectives (happy/sad/little/good/bad), everyday nouns (day/friend/home/food/water/book),
and common verbs (said/went/made/help/love) -- broadening the speakable-on-spikes vocab past BRIDGE-1/2/3/4's
61 words. Each word gets its own dedicated ~200-neuron pool (word-specific by construction = the full-fidelity
Rank-2 path the whitened-decode ceiling proof named). Dispatched by `_realcorpus_multi_bridge_speaker.py`
(first-bridge-wins on any overlap). NO `sim/` edit. SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse, os, time
from research.runners.concept_pool_demo import build_concept_bridge, train_word_to_pool, apply_concept_topographic_bias

VOCAB = ["happy", "sad", "little", "good", "bad", "day", "friend", "home",
         "food", "water", "book", "said", "went", "made", "help", "love"]        # 16 words
POOLS = ([f"noun_pool_{n}" for n in ("APPLE", "RIVER", "DOG", "CAT")]
         + [f"adjective_pool_{n}" for n in ("BIG", "SMALL", "HOT", "COLD")]
         + [f"verb_pool_{n}" for n in ("GO", "COME", "STOP", "LOOK")]
         + [f"motor_{a}" for a in ("N", "E", "S", "W")])
WORD_TO_POOL = dict(zip(VOCAB, POOLS)); WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
N_LANG = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--out-dir", default="bridges/breadth_aw5")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"seed{a.seed}.simstate.h5")
    print(f"[breadth A->W BRIDGE-5 train] {len(VOCAB)} words: {VOCAB}", flush=True)
    bridge = build_concept_bridge(seed=a.seed, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                  weak_dynamics=True, enable_adjective=True, enable_direct_verb_to_motor=True, verbose=False)
    apply_concept_topographic_bias(bridge, n_lang_input=N_LANG, topographic_factor=3.0, off_target_factor=0.3,
                                   sparsity=0.05, orthogonal_codes=True, n_words_for_orthogonal=len(VOCAB),
                                   word_to_idx=WORD_TO_IDX, word_to_pool_override=WORD_TO_POOL, verbose=False)
    t0 = time.time()
    for w in VOCAB:
        tw = time.time()
        train_word_to_pool(bridge, w, WORD_TO_POOL[w], n_events=a.n_events, n_lang_input=N_LANG, n_lang_output=N_LANG,
                           sparsity=0.05, orthogonal_codes=True, n_words_for_orthogonal=len(VOCAB), word_to_idx=WORD_TO_IDX, verbose=False)
        print(f"  trained '{w}' -> {WORD_TO_POOL[w]} ({time.time()-tw:.0f}s)", flush=True)
    bridge.save_checkpoint(out)
    print(f"[breadth A->W BRIDGE-5 train] complete ({time.time()-t0:.0f}s); saved {out}", flush=True)


if __name__ == "__main__":
    main()
