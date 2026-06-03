"""Learned voice-invariant SVO parser (conjunctive position*voice coding) -- the faithful COMPREHEND piece
for the integrated conversation loop, replacing hand-coded position->role rules.

Validated science (research/findings/raw/_vsa_parser_voice_probe.py): syntactic role assignment that is
VOICE-INVARIANT -- understanding that "dog chase cat" (active) and "cat is chased by dog" (passive) have the
SAME agent (dog) -- REQUIRES conjunctive position*voice coding (mixed selectivity). Position-only (P) and
additive position+voice (PV) both score 0.000; the conjunctive (PxV) interaction scores 1.000, because the
voice flip (agent<->patient) is an interaction, not an additive effect. Biology: a brain parsing syntax must
CONJOIN word-position with a syntactic-voice cue; it cannot read roles off position alone.

This module packages that readout (closed-form least-squares over the 6 (position, voice) combinations) plus
voice detection + light morphology (so real passive sentences with inflected verbs parse) into a reusable
parser. The fully-spiking version (conjunctive coding in the substrate's distributed codes) is a further
step; this is the learned-not-hand-coded, voice-invariant comprehend piece. numpy only; no protected import.

  from research.runners.conjunctive_parser import ConjunctiveParser
  ConjunctiveParser().parse("cat is chased by dog", vocab)  # -> {agent: dog, action: chase, patient: cat}
"""
from __future__ import annotations
import numpy as np

ROLES = ["agent", "action", "patient"]

# small irregular past-participle / past-tense map (real parsers have full morphology; this covers the
# common irregulars so passive sentences with irregular verbs normalize to the base vocab verb).
_IRREGULAR = {
    "held": "hold", "seen": "see", "saw": "see", "eaten": "eat", "ate": "eat", "given": "give",
    "gave": "give", "made": "make", "found": "find", "taken": "take", "took": "take", "chosen": "choose",
    "caught": "catch", "brought": "bring", "thrown": "throw", "threw": "throw", "sent": "send",
}


def detect_voice(tokens):
    """Passive iff the sentence has the 'BE ... by' frame (N is/was/are/were V by N)."""
    low = [t.lower() for t in tokens]
    return any(b in low for b in ("is", "was", "are", "were", "be", "been")) and "by" in low


def _normalize(tok, vocab):
    """Map an (possibly inflected) token to its base vocab form, or None if it is not a content word."""
    if tok in vocab:
        return tok
    if tok in _IRREGULAR and _IRREGULAR[tok] in vocab:
        return _IRREGULAR[tok]
    for suf in ("ed", "d", "ing", "es", "s"):                 # regular inflections
        if tok.endswith(suf):
            base = tok[:-len(suf)]
            if base in vocab:
                return base
            if base + "e" in vocab:                           # chased -> chas + e -> chase
                return base + "e"
    return None


def _features(pos, passive):
    """Conjunctive position*voice (PxV) feature vector for a content word at content-position pos."""
    p = [0.0, 0.0, 0.0]
    p[pos] = 1.0
    v = 1.0 if passive else 0.0
    px = [pi * v for pi in p]
    return np.array(p + [v] + px + [1.0])


def fit_readout():
    """Closed-form least-squares readout: (position, voice) features -> role one-hot, over all 6 combos."""
    X, Y = [], []
    for passive in (False, True):
        truth = ["patient", "action", "agent"] if passive else ["agent", "action", "patient"]
        for pos in range(3):
            X.append(_features(pos, passive))
            y = [0.0, 0.0, 0.0]
            y[ROLES.index(truth[pos])] = 1.0
            Y.append(y)
    W, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return W


class ConjunctiveParser:
    """Parse a sentence (active or passive) to a {role: word} meaning, voice-invariantly."""

    def __init__(self):
        self.W = fit_readout()

    def parse(self, text, vocab):
        """text -> {role: base_word} (voice-invariant), or None if it is not a 3-content-word sentence."""
        toks = (text or "").strip().rstrip("?.").split()
        passive = detect_voice(toks)
        content = []
        for t in toks:
            base = _normalize(t.lower(), vocab)
            if base is not None:
                content.append(base)
        if len(content) != 3:
            return None
        out = {}
        for pos, w in enumerate(content):
            pred = _features(pos, passive) @ self.W
            out[ROLES[int(np.argmax(pred))]] = w
        return out
