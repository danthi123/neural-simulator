"""A TALKABLE console for the EMERGENT generation ladder (closes the arc to full capacity: the reslm generator was
validated but wired into NO console the owner can talk to). The owner types a word; the EMERGENT reservoir-LM (Rung-1,
an on-substrate spiking reservoir + a local next-token read-out, NO backprop) GENERATES its predicted next token from
its own learned dynamics and the console shows it. A no-confab moat: a word the generator never learned as a context ->
"I don't know." Reuse-by-import of the validated Rung-5 reslm (`_rung5_reslm_spiking_spellout_derisk._reslm_predict`
machinery); NO `sim/` edit.

HONEST SCOPE (per the deep-research gate's shortlist #3): the corpus is EMERGE-67's bounded 16-word subject->verb
bijection (owl->fly, penguin->swim, ...), so the "conversation" is an interactive demo of the emergent generator over a
toy grammar -- it EXPOSES where the ladder is thin (bounded vocab + toy grammar), NOT fluent open-domain speech. The
generation is genuinely EMERGENT (the reslm LEARNED the subject->verb map from the stream; it is not a host lookup --
`--lesion` zeroes the read-out and the prediction collapses). ESCALATIONS (validated, separate): the predicted token is
spellable ON SPIKES via `_rung5_..._derisk --derisk` (GPU A->W read-out, 6-seed GO); novel-referent tracking via the
Rung-6c `HebbianBinder`; open vocab (V=200) = more A->W bridges (EMERGE-68).

Run:
  SIM_BACKEND=numpy python -m research.runners._reslm_talkable_console --smoke        # scripted GPU-free verification
  SIM_BACKEND=numpy python -m research.runners._reslm_talkable_console                 # interactive REPL
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys

import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, ReservoirStates, train_readout, _cache, _standardize_fit
from research.runners._emerge67_neural_spell_wirein_derisk import _AW_SUBJECTS, _AW_CONTENT
from research.runners._rung5_reslm_spiking_spellout_derisk import SUBJ_VERB, _corpus


class ReslmConsole:
    """Holds a trained emergent reslm; `generate(word)` rolls out the next token from the reservoir's own dynamics."""

    def __init__(self, seed=42, n_pool=200, reps=120, epochs=25, lr=0.02, lesion=False):
        self.vocab = Vocab(list(_AW_CONTENT))
        self.res = ReservoirStates(in_dim=self.vocab.size, seed=seed, n=n_pool)
        cache = _cache(self.res, self.vocab, _corpus(reps))
        self.mean, self.std = _standardize_fit(cache)
        self.W = train_readout(cache, self.vocab.size, epochs, lr, np.random.default_rng(seed * 13 + 1), self.mean, self.std)
        if lesion:                                             # anti-cheat: a zero read-out -> the prediction is not a host lookup
            self.W = np.zeros_like(self.W)
        self.subjects = set(_AW_SUBJECTS)

    def generate(self, word):
        """Return (predicted_next_token, known). A word the generator never saw as a context (not a learned subject)
        -> the no-confab moat abstains (known=False)."""
        w = word.strip().lower()
        if w not in self.subjects:                             # the moat: only learned contexts drive a grounded generation
            return None, False
        toks = self.res.rollout(self.vocab, self.W, self.mean, self.std, seed_token=w, n_gen=1)
        return (toks[1] if len(toks) > 1 else None), True

    def reply(self, word):
        pred, known = self.generate(word)
        if not known:
            return f"I don't know what '{word.strip()}' does -- I only learned about: {', '.join(sorted(self.subjects))}."
        return f"the {word.strip()} {pred}." if pred else f"(the {word.strip()} ... I couldn't generate a continuation.)"


# the learned subjects are EMERGE-67's 8 birds (_AW_SUBJECTS); pick 4 of them + 2 never-learned words for the moat.
_SCRIPT = [
    (_AW_SUBJECTS[0], "known"), (_AW_SUBJECTS[3], "known"), (_AW_SUBJECTS[5], "known"), (_AW_SUBJECTS[7], "known"),
    ("dragon", "moat"),                    # never-learned subject -> the moat abstains
    ("banana", "moat"),                    # not even an animal -> the moat abstains
]


def run_smoke(seed=42):
    con = ReslmConsole(seed=seed)
    print(f"[reslm-console smoke seed={seed}] the EMERGENT generator answers over its learned 16-word subject->verb grammar:", flush=True)
    ok_known = ok_moat = n_known = n_moat = 0
    for word, kind in _SCRIPT:
        pred, known = con.generate(word)
        line = con.reply(word)
        if kind == "known":
            n_known += 1
            hit = known and pred == SUBJ_VERB.get(word)
            ok_known += int(hit)
            print(f"    you> {word:<9} brain> {line}   [{'OK' if hit else 'x -- expected ' + SUBJ_VERB.get(word, '?')}]", flush=True)
        else:
            n_moat += 1
            ok_moat += int(not known)
            print(f"    you> {word:<9} brain> {line}   [{'OK moat' if not known else 'x -- LEAKED'}]", flush=True)
    # anti-cheat: a lesioned read-out must NOT reproduce the learned subject->verb map (proves it is emergent, not a lookup)
    lcon = ReslmConsole(seed=seed, lesion=True)
    lesion_hits = sum(1 for w in _AW_SUBJECTS if lcon.generate(w)[0] == SUBJ_VERB.get(w))
    emergent = lesion_hits <= max(1, len(_AW_SUBJECTS) // 4)   # lesion collapses to ~chance -> the generation is emergent
    go = (ok_known == n_known) and (ok_moat == n_moat) and emergent
    print(f"    -> known-correct {ok_known}/{n_known}  moat-held {ok_moat}/{n_moat}  lesion-collapse {lesion_hits}/{len(_AW_SUBJECTS)} (emergent={emergent}) "
          f"-> {'GO -- the emergent generator is talkable, grounded, moat-safe' if go else 'no'}", flush=True)
    print(f"    (escalations: spell the prediction ON SPIKES via _rung5_..._derisk --derisk [GPU A->W, 6-seed GO]; "
          f"novel-referent tracking via the Rung-6c HebbianBinder; open vocab = more A->W bridges.)", flush=True)
    return go


def repl(seed=42):
    con = ReslmConsole(seed=seed)
    print("EMERGENT-generator console (type a subject, e.g. 'owl'; Ctrl-D to quit). Learned subjects: "
          + ", ".join(sorted(con.subjects)), flush=True)
    while True:
        try:
            word = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not word:
            continue
        print("brain> " + con.reply(word), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if run_smoke(a.seed) else 1)
    repl(a.seed)


if __name__ == "__main__":
    main()
