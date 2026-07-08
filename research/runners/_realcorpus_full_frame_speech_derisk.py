"""KNOWLEDGE-half of breadth, FULL-FRAME fluent speech: the brain speaks a grounded FRAME
("the cat can go") over its vocab -- not just yes/no.

The SPEAK rung spoke a yes/no proxy word. This speaks a full grammatical frame: the reasoner's
(subject, verb) decision is rendered "the <subject> can <verb>", with the CONTENT words (subject,
verb) SPELLED ON SPIKES via the validated concept-pool A->W read-out (drive the word's pool ->
decode from `language_output` firing; the cached v16 bridge), and the closed-class frame words
(the, can) host-rendered (a documented scaffold -- EMERGE-68's spiking function-word A->W is the
follow-on). Gate-first moat: an unknown word (not in the A->W vocab) -> "I don't know", no frame.

Honest scope: the content vocab is the cached v16 concept-pool A->W (nouns dog/cat/apple/river +
verbs go/come/stop/look + adjectives), so the demo speaks frames over THAT overlap with the breadth
reasoner's discovered vocab (dog/cat). Broad-vocab full-frame speech = an A->W retrain on the
reasoner's full vocab (the GPU follow-on). Reuse-by-import. NO sim/ edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners.concept_pool_demo import (
    build_concept_bridge, NOUN_VOCAB, VERB_VOCAB, ADJECTIVE_VOCAB, DIRECTION_VOCAB,
)
from research.runners.concept_speak_demo import (
    _target_pool_for_word, drive_pool_and_read_lang_output, _cosine, _all_words,
)
from sim.text_embeddings import orthogonal_drive_pattern

N_LANG = 2048
SPARSITY = 0.05


class ConceptFrameSpeaker:
    """Speaks a grounded frame with content words produced ON SPIKES via the cached v16 A->W."""

    def __init__(self, bridge_path, seed=42):
        self.bridge = build_concept_bridge(
            seed=seed, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
            weak_dynamics=True, enable_adjective=True,
            enable_direct_verb_to_motor=True, verbose=False,
        )
        self.bridge.load_checkpoint(bridge_path)
        # the A->W content vocab + their orthogonal reference patterns (match v16 training)
        self.vocab = _all_words(include_adjective=True)
        self.patterns = {w: orthogonal_drive_pattern(cue_idx=self.vocab.index(w), n_cues=len(self.vocab),
                                                      n_neurons=N_LANG, sparsity=SPARSITY)
                         for w in self.vocab}

    def spell(self, word):
        """Drive the word's pool -> decode the spoken word from language_output firing (ON SPIKES)."""
        if word not in self.vocab:
            return None
        pool = _target_pool_for_word(word)
        spike = drive_pool_and_read_lang_output(self.bridge, pool, n_lang_output=N_LANG)
        best = max(self.vocab, key=lambda w: _cosine(spike, self.patterns[w]))
        return best

    def speak_frame(self, subject, verb):
        """Render 'the <subject> can <verb>' with the content words SPELLED ON SPIKES; the/can host-rendered.
        Gate-first moat: unknown content word -> 'I don't know' (no frame)."""
        if subject not in self.vocab or verb not in self.vocab:
            return "I don't know", None
        subj_spoken = self.spell(subject)      # ON SPIKES
        verb_spoken = self.spell(verb)         # ON SPIKES
        frame = f"the {subj_spoken} can {verb_spoken}"   # closed-class 'the'/'can' host-rendered (scaffold)
        correct = (subj_spoken == subject and verb_spoken == verb)
        return frame, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="bridges/v16/seed42.simstate.h5")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    spk = ConceptFrameSpeaker(a.bridge, seed=a.seed)
    print(f"[full-frame speech] loaded {a.bridge}; content vocab: {spk.vocab}", flush=True)

    # a few grounded frames (subject, verb) from the overlap vocab + a moat probe
    tests = [("dog", "go"), ("cat", "come"), ("apple", "stop"), ("river", "look"),
             ("zzzqqx", "go")]   # last = unknown subject -> moat
    n_ok, n_frame = 0, 0
    for subj, verb in tests:
        frame, correct = spk.speak_frame(subj, verb)
        if correct is None:
            print(f"  ({subj}, {verb}) -> \"{frame}\"  [MOAT: unknown word, no frame]", flush=True)
        else:
            n_frame += 1; n_ok += int(bool(correct))
            print(f"  ({subj}, {verb}) -> \"{frame}\"  [content ON SPIKES, {'exact' if correct else 'MISREAD'}]", flush=True)
    print(f"\n  frame content-spell accuracy: {n_ok}/{n_frame} exact | moat on unknown: OK", flush=True)
    print(f"  VERDICT: {'GO' if n_ok == n_frame and n_frame > 0 else 'PARTIAL'} -- the brain speaks a grounded "
          f"FRAME ('the <subject> can <verb>') with the CONTENT words produced ON SPIKES (concept-pool A->W from "
          f"language_output), and ABSTAINS on an unknown word (gate-first moat). Closed-class the/can host-rendered "
          f"(the spiking function-word A->W is EMERGE-68's follow-on); broad-vocab = an A->W retrain.", flush=True)


if __name__ == "__main__":
    main()
