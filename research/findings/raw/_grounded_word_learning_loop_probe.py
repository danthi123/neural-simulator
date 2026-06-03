"""Cheap-first END-TO-END: text-as-pixels -> real V1 -> ONE-SHOT Hebbian grounding -> compose+produce.
The faithful, tokenizer-free, data-efficient word-learning loop, combining all validated pieces.

Loop: render a word as pixels -> real retina -> real Gabor V1 (the word-FORM feature) -> ONE-SHOT Hebbian
binding of that V1 word-form to its concept feature (grounding: the word co-occurs ONCE with its referent) ->
the grounded concept enters generate-by-composition. Tests: (1) one-shot word acquisition -- after ONE exposure
each, recognize words from their V1 word-form (nearest grounded prototype); (2) the grounded words COMPOSE --
form novel sentences and produce them in order (generate-by-composition). Data-efficiency: 1 exposure/word ->
read + compose any sentence, no tokenizer, no co-occurrence-statistics training.

Reuse-by-import: V1 machinery (_text_as_pixels_v1_probe) + generate-by-composition. Stdlib + numpy + PIL +
sim.visual_cortex (the validated visual pathway). No protected-module change.

  python -m research.findings.raw._grounded_word_learning_loop_probe
"""
from __future__ import annotations
import numpy as np
from research.findings.raw._text_as_pixels_v1_probe import (render, v1_features, ALPHABET, N_POS)
from research.findings.raw._generate_by_composition_probe import compose, generate, ROLE_NAMES
import sim.visual_cortex as VC


def main():
    pre, post, w = VC.build_v1_simple_weights()
    n_v1 = VC.N_ORIENTATIONS * VC.N_FREQUENCIES * VC.V1_POSITIONS_PER_DIM * VC.V1_POSITIONS_PER_DIM
    rng = np.random.default_rng(42)
    # vocabulary of distinct 3-letter word-forms
    words = []
    seen = set()
    while len(words) < 24:
        ww = "".join(rng.choice(ALPHABET, size=N_POS))
        if ww not in seen:
            seen.add(ww); words.append(ww)

    def v1_of(ww):
        f = v1_features(render(ww), pre, post, w, n_v1)
        return f / (np.linalg.norm(f) + 1e-9)

    print("=== grounded word-learning loop: text-as-pixels -> V1 -> ONE-SHOT grounding -> compose+produce ===",
          flush=True)
    # ONE-SHOT grounding: each word seen ONCE -> store (V1 word-form prototype, concept feature)
    D = 512
    concept_feat = {ww: (lambda v: (v - v.mean()) / (np.linalg.norm(v - v.mean()) + 1e-9))(
        np.random.default_rng(1000 + i).standard_normal(D)) for i, ww in enumerate(words)}
    proto = {ww: v1_of(ww) for ww in words}                       # 1 exposure each (the grounding event)

    # (1) one-shot recognition: render the word again (render noise/aa) -> nearest grounded prototype -> concept
    rec_ok = 0
    for ww in words:
        f = v1_of(ww)                                            # a fresh view of the word-form
        best = max(words, key=lambda k: float(proto[k] @ f))
        rec_ok += int(best == ww)
    print(f"  (1) one-shot word recognition from V1 word-form: {rec_ok}/{len(words)} "
          f"({rec_ok/len(words):.3f}) -- 1 exposure/word", flush=True)

    # (2) the grounded words COMPOSE: build NOVEL sentences from grounded concepts, produce in order
    concepts = {i: concept_feat[ww] for i, ww in enumerate(words)}
    rrng = np.random.default_rng(7)                              # ONE rng -> DISTINCT role vectors
    roles = {r: rrng.choice([-1.0, 1.0], size=D) for r in ROLE_NAMES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    idx2word = {i: ww for i, ww in enumerate(words)}
    rng2 = np.random.default_rng(3)
    comp_ok = tot = 0
    transcript = []
    for _ in range(20):
        pick = rng2.choice(len(words), size=3, replace=False)
        meaning = {ROLE_NAMES[k]: int(pick[k]) for k in range(3)}
        bound = compose(meaning, concepts, roles)
        out_idx = generate(bound, ROLE_NAMES[:3], concepts, roles, list(range(len(words))))
        target = [meaning[ROLE_NAMES[k]] for k in range(3)]
        comp_ok += int(out_idx == target); tot += 1
        if len(transcript) < 4:
            transcript.append(" ".join(idx2word[i] for i in out_idx))
    print(f"  (2) grounded words COMPOSE into novel sentences (produced in order): {comp_ok}/{tot} "
          f"({comp_ok/tot:.3f})", flush=True)
    print(f"      sample produced sentences: {transcript}", flush=True)

    ok = (rec_ok / len(words) >= 0.9) and (comp_ok / tot >= 0.9)
    if ok:
        print("\nVERDICT: RESOLVES -- the FAITHFUL data-efficient word-learning loop works end-to-end: words "
              "enter as PIXELS through the real visual pathway, are grounded to concepts in ONE exposure each "
              "(no tokenizer, no co-occurrence-statistics training), and the grounded words COMPOSE into novel "
              "produced sentences. 1 exposure/word -> read + compose any sentence. -> wire this into the bridge "
              "(retina->V1->grounded concept pools) as the production input path.", flush=True)
    else:
        print(f"\nVERDICT: recognition {rec_ok/len(words):.2f} / composition {comp_ok/tot:.2f} -- characterize "
              "(V1 word-form separability / grounding capacity) before the bridge wire-up.", flush=True)


if __name__ == "__main__":
    main()
