"""Multi-bridge A->W dispatch speaker (EMERGE-68 route, generalized to N bridges): SPEAK a broad vocab on
spikes by routing each word to whichever concept-pool bridge holds it. One concept-pool bridge caps at 16
words; three bridges (BRIDGE-1 = 8 animals + 6 bare verbs + the/can; BRIDGE-2 = 15 object nouns; BRIDGE-3 =
9 3sg verb surfaces + "a" + 6 more animal subjects) cover the RELATIONAL C_TRANS answer's whole vocab (the
subject, the 3sg verb, the object, the/a) so "the dog chases the cat" renders FULLY on spikes.

Exposes the SAME interface (`spell`, `speak_frame`, `vocab`) as `ConceptFrameSpeaker` so the unified console
uses it transparently. numpy or cupy. NO `sim/` edit.
"""
from __future__ import annotations

from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB as V1, WORD_TO_POOL as P1
from research.runners._realcorpus_train_breadth_aw2 import VOCAB as V2, WORD_TO_POOL as P2
from research.runners._realcorpus_train_breadth_aw3 import VOCAB as V3, WORD_TO_POOL as P3

BRIDGE1 = "bridges/breadth_aw/seed42.simstate.h5"
BRIDGE2 = "bridges/breadth_aw2/seed42.simstate.h5"
BRIDGE3 = "bridges/breadth_aw3/seed42.simstate.h5"

# (path, vocab, word_to_pool) per bridge; earlier bridges WIN on overlap (there is none by construction).
DEFAULT_BRIDGES = [(BRIDGE1, V1, P1), (BRIDGE2, V2, P2), (BRIDGE3, V3, P3)]


class MultiBridgeFrameSpeaker:
    """Dispatches spell/speak_frame across N A->W bridges by word membership."""

    def __init__(self, bridges=None, seed=42):
        bridges = bridges if bridges is not None else DEFAULT_BRIDGES
        self.speakers = [ConceptFrameSpeaker(p, seed=seed, vocab=v, word_to_pool=wp) for (p, v, wp) in bridges]
        self.vocab, self._of = [], {}
        for spk, (_p, v, _wp) in zip(self.speakers, bridges):
            for w in v:
                if w not in self._of:                # first bridge holding w wins
                    self._of[w] = spk
                    self.vocab.append(w)

    def spell(self, word):
        """Spell a word ON SPIKES via whichever bridge holds it (else None)."""
        spk = self._of.get(word)
        return spk.spell(word) if spk is not None else None

    def speak_frame(self, subject, verb):
        """'the <subject> can <verb>' (the F_MODAL property frame) with content ON SPIKES via dispatch."""
        if subject not in self._of or verb not in self._of:
            return "I don't know", None
        subj_spoken, verb_spoken = self.spell(subject), self.spell(verb)
        the_spoken = self.spell("the") if "the" in self._of else "the"
        can_spoken = self.spell("can") if "can" in self._of else "can"
        frame = f"{the_spoken} {subj_spoken} {can_spoken} {verb_spoken}"
        return frame, (subj_spoken == subject and verb_spoken == verb)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    spk = MultiBridgeFrameSpeaker(seed=a.seed)
    print(f"[multi-bridge A->W] combined vocab ({len(spk.vocab)}): {spk.vocab}", flush=True)
    n_ok = 0
    for w in spk.vocab:
        got = spk.spell(w)
        ok = (got == w)
        n_ok += int(ok)
        print(f"  spell '{w}' -> '{got}'  {'OK' if ok else 'MISREAD'}", flush=True)
    print(f"\n  all-word spell accuracy: {n_ok}/{len(spk.vocab)}", flush=True)
    print(f"  VERDICT: {'GO' if n_ok >= len(spk.vocab) - 3 else 'PARTIAL'} -- the brain speaks a "
          f"{len(spk.vocab)}-word vocab across {len(spk.speakers)} A->W bridges (dispatched by word).", flush=True)


if __name__ == "__main__":
    main()
