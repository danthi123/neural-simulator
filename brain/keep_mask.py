# Capability: Lifelong learning
# Record: brain/research/2026-11-23-kp-identity-prereg.md (K-P-U turn rule, section 1); brain/research/2026-11-21-sleep-language-keep-prereg.md (K-P)
# Expected-fail: yes, named in the prereg: masking the Assistant-role positions may give back K-P's language win, and the self-name/refusal costs may sit in the weights, not the keep set
"""K-P-U: which positions of a keep text the sleep keep KL counts (Brain(keep_bank="prior_u")).

Turn rule (structural, no content words; declared scaffold, brain/SCAFFOLDS.md): a line that opens with a speaker label
(the start of the text, or right after "\\n", then at most 3 words and ":") opens a turn. A line with no label continues
the turn that is open. A turn is a frame turn when its label equals the frame label ("User" for frame "User:"). Every
other labelled turn (Assistant, Bot, AI, ...), its label tokens and its continuation, is masked.

Token -> character offsets use the same per-token decode walk as `_scratch/kpgate_w/drift_read.roles()` (txt += decode([i])).
A token belongs to the turn open at its LAST character (so a "\\n\\n" token before a label stays in the old turn and a
"\\nAssistant" token opens the new one). Logit position i predicts token i+1, so position i is kept when token i+1 is in a
frame turn; the last position (which predicts past the text) follows the last token's turn.
"""
from __future__ import annotations

import re

LABEL_MAX_WORDS = 3   # the host cap of the turn rule (SCAFFOLDS.md)
_LABEL = re.compile(r"(?m)^([^\s:]+(?:[ \t]+[^\s:]+){0,%d}):" % (LABEL_MAX_WORDS - 1))


def turn_labels(text: str) -> list:
    """[(char_start, label)] of every labelled line in `text`, in order."""
    return [(m.start(), m.group(1)) for m in _LABEL.finditer(text)]


def char_frame_mask(text: str, frame: str) -> list:
    """Per character of `text`: True when it sits in a frame turn. Characters before the first label belong to no
    labelled turn and are masked (a bank text always opens with the frame label, so this never happens there)."""
    want = frame.rstrip(":").strip()
    out = [False] * len(text)
    labs = turn_labels(text)
    for k, (start, lab) in enumerate(labs):
        end = labs[k + 1][0] if k + 1 < len(labs) else len(text)
        if lab == want:
            for c in range(start, end):
                out[c] = True
    return out


def frame_turn_positions(ids, decode, frame: str) -> list:
    """Logit positions (0..len(ids)-1) whose predicted token sits in a frame turn. ids: token ids of the text as the
    sleep loop encodes it; decode: one-token decode (cortex.pipe.decode([i]))."""
    if not ids:
        return []
    txt, ends = "", []
    for i in ids:
        txt += decode([i])
        ends.append(len(txt))
    cm = char_frame_mask(txt, frame)
    tok = []
    for j, e in enumerate(ends):
        start = ends[j - 1] if j else 0
        tok.append(cm[e - 1] if e > start else (tok[-1] if tok else False))   # an empty decode follows its neighbour
    n = len(ids)
    return [i for i in range(n) if (tok[i + 1] if i + 1 < n else tok[-1])]
