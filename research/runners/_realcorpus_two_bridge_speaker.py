"""Two-bridge A->W dispatch speaker (EMERGE-68 route): SPEAK a broader vocab on spikes by routing each
word to whichever concept-pool bridge holds it. One concept-pool bridge caps at 16 words; two bridges
(BRIDGE-1 = 8 animals + 6 verbs + the/can; BRIDGE-2 = 15 object/character nouns) roughly triple the
speakable vocab. Exposes the SAME interface (`spell`, `speak_frame`, `vocab`) as `ConceptFrameSpeaker`
so the unified console can use it transparently. numpy or cupy. NO `sim/` edit.
"""
from __future__ import annotations

from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB as V1, WORD_TO_POOL as P1
from research.runners._realcorpus_train_breadth_aw2 import VOCAB as V2, WORD_TO_POOL as P2

BRIDGE1 = "bridges/breadth_aw/seed42.simstate.h5"
BRIDGE2 = "bridges/breadth_aw2/seed42.simstate.h5"


class TwoBridgeFrameSpeaker:
    """Dispatches spell/speak_frame across two A->W bridges by word membership."""

    def __init__(self, bridge1=BRIDGE1, bridge2=BRIDGE2, seed=42):
        self.s1 = ConceptFrameSpeaker(bridge1, seed=seed, vocab=V1, word_to_pool=P1)
        self.s2 = ConceptFrameSpeaker(bridge2, seed=seed, vocab=V2, word_to_pool=P2)
        self.vocab = list(V1) + [w for w in V2 if w not in V1]
        self._of = {}
        for w in V1:
            self._of[w] = self.s1
        for w in V2:
            self._of.setdefault(w, self.s2)      # BRIDGE-1 wins on overlap (there is none by construction)

    def spell(self, word):
        """Spell a word ON SPIKES via whichever bridge holds it (else None)."""
        spk = self._of.get(word)
        return spk.spell(word) if spk is not None else None

    def speak_frame(self, subject, verb):
        """'the <subject> can <verb>' with content ON SPIKES via dispatch; the/can host-rendered
        (matches ConceptFrameSpeaker's contract for a vocab lacking the/can on a given bridge)."""
        if subject not in self.vocab or verb not in self.vocab:
            return "I don't know", None
        subj_spoken = self.spell(subject)
        verb_spoken = self.spell(verb)
        the_spoken = self.spell("the") if "the" in self.vocab else "the"
        can_spoken = self.spell("can") if "can" in self.vocab else "can"
        frame = f"{the_spoken} {subj_spoken} {can_spoken} {verb_spoken}"
        correct = (subj_spoken == subject and verb_spoken == verb)
        return frame, correct


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    spk = TwoBridgeFrameSpeaker(seed=a.seed)
    print(f"[two-bridge A->W] combined vocab ({len(spk.vocab)}): {spk.vocab}", flush=True)
    n_ok = 0
    for w in spk.vocab:
        got = spk.spell(w)
        ok = (got == w)
        n_ok += int(ok)
        print(f"  spell '{w}' -> '{got}'  {'OK' if ok else 'MISREAD'}", flush=True)
    print(f"\n  all-word spell accuracy: {n_ok}/{len(spk.vocab)}", flush=True)
    print(f"  VERDICT: {'GO' if n_ok >= len(spk.vocab) - 2 else 'PARTIAL'} -- the brain speaks a "
          f"{len(spk.vocab)}-word vocab across two A->W bridges (dispatched by word).", flush=True)


if __name__ == "__main__":
    main()
