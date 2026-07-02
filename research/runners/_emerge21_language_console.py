"""EMERGE-21 / toward-language — the END-TO-END console: a small EMERGENT sequence cortex the owner can CUE, unifying
the whole toward-language chain (EMERGE-15..20) into one interactive artifact on the real spiking `SimulationBridge`.
The owner cues a word; the cortex PRODUCES a grounded word sequence, GENERALIZES to a similar (untrained) cue, and
ABSTAINS ("I don't know") for a truly-novel/ungrounded cue (the intrinsic no-confab moat). NO transformer, NO external
model, NO `sim/` edit -- the brain does the language.

Capabilities shown (all validated GO in EMERGE-15..20):
  - GENERATE (production): cue "dog" -> rolls out the grounded continuation "chased ball home" (autoregressive
    excitability-replay, EMERGE-16).
  - GENERALIZE: cue "wolf" (never trained, but SIMILAR to dog via shared canine micro-columns) -> generates the
    canine continuation (EMERGE-17/18/19).
  - MOAT / ABSTAIN: cue "zzz" (a novel word, code disjoint from everything) -> ABSTAINS (EMERGE-20; the moat is
    intrinsic -- no learned pathway -> no production, so it cannot confabulate).

Reuse-by-import (`_emerge14` + `_emerge18` `SeqGenLearner`); NO `sim/` edit. Run: `python -m research.runners._emerge21_language_console`
for the scripted transcript (default), or `--interactive` to cue words yourself. CPU numpy-backend.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from collections import Counter
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, coincidence_predict
from research.runners._emerge18_sequence_generalization_derisk import SeqGenLearner


def build_corpus():
    """A tiny grounded corpus. Canine/feline families share micro-columns (overlapping codes -> generalization); the
    grounded facts are the trained sentences; novel words have disjoint codes (-> abstain)."""
    word2cols = {
        "dog": [0, 1, 2, 3], "wolf": [0, 1, 2, 4], "fox": [0, 1, 2, 5],           # canines share [0,1,2]
        "cat": [6, 7, 8, 9], "lion": [6, 7, 8, 10],                                # felines share [6,7,8]
        "chased": [11, 12, 13, 14], "ball": [15, 16, 17, 18],                      # shared middle
        "home": [19, 20, 21, 22], "away": [23, 24, 25, 26],                        # branch words
        "zzz": [40, 41, 42, 43], "qqq": [44, 45, 46, 47],                          # NOVEL -- disjoint codes (must abstain)
    }
    family_branch = {"dog": "home", "wolf": "home", "fox": "home", "cat": "away", "lion": "away"}
    trained = ["dog", "cat"]                                                       # the grounded (trained) subjects
    sentences = {s: [s, "chased", "ball", family_branch[s]] for s in family_branch}
    M = 1 + max(c for cols in word2cols.values() for c in cols)
    return word2cols, family_branch, trained, sentences, M


class LanguageCortex:
    """A trained emergent sequence cortex the owner can cue. Wraps the EMERGE-18 SeqGenLearner (multi-micro-column words,
    high-order sequences) + an autoregressive GENERATE rollout + the intrinsic-moat abstain."""

    def __init__(self, seed=42, epochs=80, k_win=4, act_th=3):
        self.word2cols, self.fb, self.trained, self.sentences, self.M = build_corpus()
        self.nE = 24
        self.b, self.cells_idx, self.row, self.col = build_pool_bridge(self.M, self.nE, seed, act_th=act_th)
        self.lr = SeqGenLearner(self.b, self.row, self.col, self.cells_idx, self.word2cols, self.nE, self.M,
                                k_win=k_win, act_th=act_th)
        self.words = list(self.word2cols)
        for _ in range(epochs):
            for s in self.trained:
                self.lr.train_sentence(self.sentences[s])

    def _sdr(self, w):
        return set(c * self.nE + 0 for c in self.word2cols[w])

    def _predict_word(self, active, exclude):
        primed = coincidence_predict(self.b, self.cells_idx, active, self.M * self.nE, self.nE)
        if not primed:
            return None, set()
        pc = Counter(int(i) // self.nE for i in primed)
        cand = [w for w in self.words if w not in exclude]
        scores = {w: sum(pc.get(c, 0) for c in self.word2cols[w]) for w in cand}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return None, set()
        winners = set(i for i in primed if int(i) // self.nE in set(self.word2cols[best]))
        return best, winners

    def respond(self, cue, max_len=4):
        """Autoregressively GENERATE a continuation from the cue word; return the produced word list (starting with the
        cue), or [cue, '<abstain>'] if the cortex has no grounded pathway for it (the intrinsic moat)."""
        out = [cue]
        active = self._sdr(cue)                                                    # the cue fires its identity SDR
        seen = set(self.word2cols[cue])
        for _ in range(max_len):
            nxt, winners = self._predict_word(active, exclude=set(out))
            if nxt is None:
                break
            out.append(nxt)
            active = winners if winners else self._sdr(nxt)
            if nxt in ("home", "away"):                                            # a branch word ends the utterance
                break
        if len(out) == 1:                                                          # nothing produced -> ABSTAIN (moat)
            return out + ["<I don't know>"]
        return out


def _transcript(cortex):
    print("\n=== EMERGE-21 language console -- a small emergent spiking cortex you can cue (no transformer) ===")
    print(f"  trained (grounded) on: {[' '.join(cortex.sentences[s]) for s in cortex.trained]}")
    print(f"  vocab: {cortex.words}\n")
    demo = [
        ("dog", "GROUNDED (trained): produces the learned fact"),
        ("cat", "GROUNDED (trained)"),
        ("wolf", "GENERALIZE: never trained, but SIMILAR to dog -> canine continuation"),
        ("fox", "GENERALIZE (similar to dog)"),
        ("lion", "GENERALIZE (similar to cat)"),
        ("zzz", "MOAT: novel/ungrounded -> ABSTAINS (no confabulation)"),
        ("qqq", "MOAT: novel/ungrounded -> ABSTAINS"),
    ]
    for cue, note in demo:
        out = cortex.respond(cue)
        print(f"  cue '{cue}'  ->  {' '.join(out)}    [{note}]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--interactive", action="store_true")
    a = ap.parse_args()
    print("[emerge21] building + training the emergent language cortex (a few seconds)...", flush=True)
    cortex = LanguageCortex(seed=a.seed, epochs=a.epochs)
    _transcript(cortex)
    if a.interactive:
        print("Interactive: type a word to cue the cortex (from the vocab above), or 'quit'.")
        while True:
            try:
                cue = input("cue> ").strip()
            except EOFError:
                break
            if cue in ("quit", "exit", ""):
                break
            if cue not in cortex.word2cols:
                print(f"  (unknown token '{cue}'; the cortex has no code for it -> would abstain)")
                continue
            print("  -> " + " ".join(cortex.respond(cue)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
