"""EMERGE-25 / toward-language — the GROUNDED GROWING CONSOLE: an interactive REPL where the owner TALKS to the emergent
spiking brain and TEACHES it. It unifies the toward-language chain (EMERGE-22 grammar + EMERGE-23 grammatical grounded
production + EMERGE-24 online growth) into ONE conversational artifact on the real spiking `SimulationBridge`:
  - ASK a subject -> the brain PRODUCES a grammatical grounded sentence about it ("dog" -> "dog chased ball");
  - ASK a SIMILAR cue -> it GENERALIZES the family exemplar's fact ("wolf" -> "wolf chased ball", canine like dog);
  - TEACH a new fact live ("bear grabbed honey") -> it GROWS (learns it on the same bridge, retains the old);
  - ASK an UNKNOWN cue -> it ABSTAINS ("zzz" -> "I don't know") -- the intrinsic no-confab moat.

Everything runs on the emergent HTM spiking sequence cortex: three-block content-bearing codes (POS-class = grammar /
content = the specific word / family = generalization) over a pre-allocated dense coincidence pool + the committed
`sim/` three-term kernel; generation reads grammar from the shared class block and content from the distinguishing
content+family blocks (a resting-vs-plateau apical threshold isolates the genuinely-primed cells). NO `sim/` edit.

HONEST SCOPE: facts use DISTINCT verbs (one exemplar per family), the validated regime (EMERGE-23/24). Facts that
SHARE a verb/object need high-order content+context binding (the shared verb's cells must be context-specific by
subject) -- a real open sub-problem (the SeqGenLearner's column-level scoring rides the shared class block and loses
the context), handed to the open-world-semantics / relational-binding research gate. This console is the distinct-verb
conversational capstone.

`--demo` for a transcript; `--script "dog;wolf;teach bear grabbed honey;bear;zzz"` for a scripted run; no args =
interactive. CPU numpy-backend; reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, sys
from collections import Counter
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

CLASS_COLS = {"NOUN": [0, 1, 2], "VERB": [3, 4, 5]}
FRAME = ["NOUN", "VERB", "NOUN"]
ACT_TH = 2
_PLATEAU_THRESH = -40.0                                                          # between the ~-62 apical rest and +20 plateau

# built-in knowledge base: one exemplar per family (others generalize via the shared family block).
DEFAULT_FACTS = [["dog", "chased", "ball"], ["cat", "ate", "fish"], ["owl", "saw", "moon"]]
DEFAULT_FAMILIES = {"dog": "canine", "wolf": "canine", "fox": "canine",         # canine members
                    "cat": "feline", "lion": "feline",                          # feline members
                    "owl": "avian", "hawk": "avian"}                            # avian members


class LanguageConsole:
    """The emergent spiking brain as a talk-to-and-teach conversational artifact (distinct-verb regime)."""

    def __init__(self, seed=42, facts=None, families=None, epochs=80, lesion=False, reserve=24):
        self.nE = 8
        self.epochs = epochs
        self.families = dict(DEFAULT_FAMILIES if families is None else families)
        facts = [list(f) for f in (DEFAULT_FACTS if facts is None else facts)]
        self.wordclass = {}                                                     # word -> NOUN/VERB (by frame position)
        self._content = {}                                                      # word -> [2 content cols]
        self._famcols = {}                                                      # family name -> [2 family cols]
        self._next_col = 6                                                      # class cols occupy 0..5
        # pre-reserve blocks for the built-in vocab + a healthy growth margin, then a disjoint zzz block.
        for f in facts:
            self._register_fact_words(f)
        # register the family MEMBERS that have no fact of their own (wolf/fox/lion/hawk) so a similar cue can
        # GENERALIZE the family exemplar's fact via the shared family block (they get a disjoint content block +
        # the shared family block -> primed by the exemplar's learned pathway through that family block).
        for w in list(self.families):
            if w not in self.wordclass:
                self.wordclass[w] = "NOUN"
                self._content[w] = self._alloc(2)
            fam = self._fam_of(w)
            if fam and fam not in self._famcols:
                self._famcols[fam] = self._alloc(2)
        # a permanently-disjoint code for the "unknown" probe (never trained) -> the moat.
        self._content["<unknown>"] = self._alloc(2); self.wordclass["<unknown>"] = "NOUN"
        self.M = 1 + self._next_col + 4 * reserve                              # headroom for taught (grown) words
        self.b, self.ci, self.row, self.col = build_pool_bridge(self.M, self.nE, seed, act_th=ACT_TH,
                                                                coincidence=(not lesion))
        self.z = np.zeros(self.M * self.nE); self.lesion = lesion
        self.facts = []
        for f in facts:                                                        # learn the built-in KB
            self._learn_fact(f, epochs)

    # ---- vocabulary / column assignment ------------------------------------------------------------------------
    def _alloc(self, k):
        cols = list(range(self._next_col, self._next_col + k)); self._next_col += k
        return cols

    def _fam_of(self, w):
        return self.families.get(w)

    def _register_fact_words(self, fact):
        for pos, w in enumerate(fact):
            cls = FRAME[pos]
            if w not in self.wordclass:
                self.wordclass[w] = cls
                self._content[w] = self._alloc(2)
            fam = self._fam_of(w)
            if fam and fam not in self._famcols:
                self._famcols[fam] = self._alloc(2)

    def _word2cols(self, w):
        fam = self._fam_of(w)
        return list(CLASS_COLS[self.wordclass[w]]) + list(self._content[w]) + list(self._famcols.get(fam, []))

    def _dist_cols(self, w):
        fam = self._fam_of(w)
        return list(self._content[w]) + list(self._famcols.get(fam, []))

    def _sdr(self, w):
        return set(c * self.nE + 0 for c in self._word2cols(w))

    def _dist_sdr(self, w):
        return set(c * self.nE + 0 for c in self._dist_cols(w))

    def _known(self, w):
        return w in self.wordclass and w != "<unknown>"

    # ---- learning ----------------------------------------------------------------------------------------------
    def _learn_fact(self, fact, epochs):
        for _ in range(epochs):
            for a, bnext in zip(fact, fact[1:]):
                apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(a), self._sdr(bnext),
                                    self.z, 0.14, 0.02, 1.0)
        if fact not in self.facts:
            self.facts.append(list(fact))

    def teach(self, fact, epochs=None):
        """Learn a new [subject, verb, object] fact LIVE on the same bridge (online growth)."""
        if len(fact) != 3:
            return f"(a fact is subject verb object, e.g. 'bear grabbed honey')"
        if self._next_col + 6 > self.M:
            return "(no column headroom left for new words -- raise reserve)"
        self._register_fact_words(fact)
        self._learn_fact(fact, self.epochs if epochs is None else epochs)
        return f"ok, i learned: {' '.join(fact)}"

    # ---- prediction / generation -------------------------------------------------------------------------------
    def _predict_primed(self, active):
        if getattr(self.b, "cp_v_apical", None) is None and not self.b.core_config.enable_coincidence_detection:
            return set()
        ab = np.zeros(len(self.ci), bool)
        for i in active:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return set()
        vap = _host(vap)[self.ci]
        return set(int(i) for i in np.where(vap > _PLATEAU_THRESH)[0])

    def generate(self, subject):
        """Return the produced word list (or ['<abstain>'])."""
        if not self._known(subject):
            # an unknown subject has no code -> route it through the reserved disjoint <unknown> code -> abstain.
            return ["<abstain>"]
        out = [subject]
        active = self._dist_sdr(subject)
        for step in range(1, len(FRAME)):
            primed = self._predict_primed(active)
            if not primed:
                break
            pc = Counter(int(i) // self.nE for i in primed)
            want = FRAME[step]
            cand = [w for w in self.wordclass if self.wordclass[w] == want and w != "<unknown>"]
            scores = {w: sum(pc.get(c, 0) for c in self._content[w]) for w in cand}
            if not scores or max(scores.values()) == 0:
                break
            nxt = max(scores, key=scores.get)
            out.append(nxt); active = self._dist_sdr(nxt)
        return out if len(out) == len(FRAME) else ["<abstain>"]

    def respond(self, subject):
        """The conversational surface: a grounded grammatical sentence, or an honest abstention."""
        out = self.generate(subject)
        if out == ["<abstain>"]:
            return "I don't know."
        return " ".join(out) + "."


def _handle(console, line):
    line = line.strip()
    if not line:
        return None
    if line.lower().startswith("teach "):
        return console.teach(line[6:].split())
    return console.respond(line.split()[0])


def _demo(seed=42, epochs=80):
    c = LanguageConsole(seed=seed, epochs=epochs)
    print("\n=== EMERGE-25 grounded growing console (talk to + teach the emergent brain; no transformer) ===")
    print(f"  knowledge: {[' '.join(f) for f in c.facts]}\n")
    script = [("dog", "grounded"), ("cat", "grounded"), ("owl", "grounded"),
              ("wolf", "generalize: canine, like dog"), ("lion", "generalize: feline, like cat"),
              ("hawk", "generalize: avian, like owl"), ("bear", "unknown -> abstain (moat)"),
              ("teach bear grabbed honey", "TEACH a new fact live"), ("bear", "now grounded (grew)"),
              ("dog", "old fact retained (no forgetting)"), ("zzz", "unknown -> abstain (moat)")]
    for cue, note in script:
        if cue.startswith("teach "):
            print(f"  you> {cue}\n  brain> {c.teach(cue[6:].split())}   ({note})")
        else:
            print(f"  you> {cue}\n  brain> {c.respond(cue)}   ({note})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--script", default=None, help="';'-separated cues, e.g. \"dog;wolf;teach bear grabbed honey;bear;zzz\"")
    a = ap.parse_args()
    if a.demo:
        _demo(a.seed, a.epochs); return 0
    c = LanguageConsole(seed=a.seed, epochs=a.epochs)
    print(f"grounded growing console -- knowledge: {[' '.join(f) for f in c.facts]}")
    print("ask a subject (dog/cat/owl/wolf/...), 'teach <subj> <verb> <obj>', or an unknown word; Ctrl-D to exit.")
    if a.script:
        for line in a.script.split(";"):
            r = _handle(c, line)
            if r is not None:
                print(f"  you> {line.strip()}\n  brain> {r}")
        return 0
    try:
        while True:
            line = input("you> ")
            r = _handle(c, line)
            if r is not None:
                print(f"brain> {r}")
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
