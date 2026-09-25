"""Build the TOKEN-LEVEL (per-battery-turn, per-occurrence) ground-truth POS fixture for the closed-class parse
diagnostic (AMENDMENT 2, research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md, review
issue G2-3). Mirrors the existing spaCy-built fixtures (`_lexicon_build_pos_gt_fixture.py`,
`closed_class_inventory_nltk_english.json`): the tagger runs ONCE, here, to produce a STATIC committed JSON; no
runtime code in `research/runners/_lexicon_closed_class_parse_diag.py` imports nltk (matching this repo's convention
that spaCy/NLTK are build-time-only tools, never a production runtime dependency -- see requirements.txt's commented
`spacy` line).

WHY THIS EXISTS. `_lexicon_closed_class_parse_diag.load_gt()` classifies a word by its TYPE-level dominant POS (one
tag for every occurrence of the word form, anywhere in the corpus): 'today' is NOUN in
`research/fixtures/lexicon_referent_pos_gt.json` (the fixture's dominant reading), 'leaves' is credited a NOUN even
in "Sally leaves the room" (a verb use), and the POS maps hold only NOUN/VERB/ADJ, so every modal ('might'), wh-word
('who', 'what'), indefinite pronoun ('something', 'everyone', 'anybody') and adverb ('never', 'ever', 'honestly',
'absolutely') NOT in the 198-word NLTK stopword list falls to UNKNOWN and is listed, not counted -- exactly the class
of word a closed-class-admission mistake would hide in. This fixture instead tags each of the 112 battery turns'
OWN words IN THEIR OWN SENTENCE (`nltk.pos_tag`, averaged-perceptron, Penn Treebank tagset), so 'leaves' resolves
per-occurrence (VBZ in "Sally leaves the room") and modals/wh-words/pronouns/adverbs resolve to NON directly from
their own tag instead of falling through two word-list lookups to UNKNOWN.

HONEST RESIDUAL: the averaged-perceptron tagger is itself imperfect (verified: it mistags 'bite' NN in "what does
the wolf bite", a verb use -- the tagger's own error, not this fixture's design; left as UNKNOWN handling would not
fix a wrong TAG, only a MISSING one). This is a second, independent instrument from the type-level maps, not a
strictly superior one; `_lexicon_closed_class_parse_diag.py` uses it for per-turn ADJUDICATION (where a wrong
type-level tag can hide or manufacture a mismatch) and keeps the type-level maps for the cross-turn `queried` report
(unchanged scope).

    /home/dant123/Projects/sim/.venv/bin/python -m research.runners._lexicon_closed_class_token_pos_fixture
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT = os.path.join(_REPO, "research", "fixtures", "lexicon_referent_pos_gt_tokenlevel.json")

# Penn Treebank tag -> the diag instrument's 3-valued class. NN*/PRP-less nominal tags -> NOUN; every closed-class /
# verbal / adjectival / adverbial tag a real word can carry -> NON; anything else (foreign words, symbols, list
# markers, interjections used as pure exclamation) -> UNKNOWN, listed not counted (same discipline as load_gt()).
_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_NON_TAGS = {
    # verbs
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    # adjectives
    "JJ", "JJR", "JJS",
    # closed-class: determiners, pronouns, wh-words, modals, prepositions, conjunctions, particles, adverbs
    "DT", "PDT", "WDT", "PRP", "PRP$", "WP", "WP$", "MD", "IN", "CC", "TO", "EX", "RB", "RBR", "RBS", "WRB",
    "UH", "POS", "RP",
}
# PENN TREEBANK CONVENTION OVERRIDE. The PTB tagset has no indefinite-pronoun tag, so its OWN annotation guidelines
# tag "something"/"everyone"/"anybody"/... as NN (verified: nltk's averaged-perceptron tagger reproduces this on
# every one of these words in the battery). Raw PTB tags therefore do NOT fix review issue G2-3 for this word class
# (it moves them UNKNOWN -> wrongly NOUN, not UNKNOWN -> NON) -- an explicit, small, closed override is needed. This
# is an INSTRUMENT word list (ground truth), never read by the mechanism under test, the same status as the NLTK
# stopword inventory already in `research/fixtures/closed_class_inventory_nltk_english.json`.
_INDEFINITE_PRONOUN_OVERRIDE = {
    "something", "someone", "somebody", "anything", "anyone", "anybody",
    "everything", "everyone", "everybody", "nothing", "noone", "nobody",
}


def build(turns):
    """turns: iterable of (label, text). Returns {label: {word_lower: class}} (first tag wins on a repeat word)."""
    from nltk import pos_tag
    import re
    word_re = re.compile(r"[A-Za-z']+")
    out = {}
    tag_counts = {}
    for label, text in turns:
        raw = word_re.findall(text or "")
        tagged = pos_tag(raw)  # pre-tokenized: no nltk tokenizer mismatch against D6._WORD_RE
        per_word = {}
        for w, tag in tagged:
            lw = w.lower()
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if lw in _INDEFINITE_PRONOUN_OVERRIDE:
                cls = "NON"
            else:
                cls = "NOUN" if tag in _NOUN_TAGS else ("NON" if tag in _NON_TAGS else "UNKNOWN")
            if lw not in per_word:      # first occurrence in the turn wins (turns are short; rare to repeat)
                per_word[lw] = {"tag": tag, "class": cls}
        out[label] = per_word
    return out, tag_counts


def main():
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL
    turns = [(label, t[1]) for label, t in _TURN_BY_LABEL.items()]
    per_turn, tag_counts = build(turns)
    unknown_tags = sorted({w_info["tag"] for pw in per_turn.values() for w_info in pw.values()
                           if w_info["class"] == "UNKNOWN"})
    out = {
        "source": "nltk averaged_perceptron_tagger (pos_tag), per-turn, tokenized with D6._WORD_RE ([A-Za-z']+)",
        # no simulation runs here (a build-time dictionary tagging pass over fixed battery text); recorded for the
        # device-and-cost gate, matching _lexicon_build_pos_gt_fixture.py's own committed-fixture convention.
        "backend": "numpy", "device": "cpu",
        "n_turns": len(per_turn), "n_labels": len(turns),
        "noun_tags": sorted(_NOUN_TAGS), "non_tags": sorted(_NON_TAGS),
        "unknown_tags_seen": unknown_tags, "tag_counts": tag_counts,
        "turns": per_turn,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"n_turns": out["n_turns"], "unknown_tags_seen": unknown_tags,
                      "wrote": os.path.relpath(OUT, _REPO)}, indent=1))


if __name__ == "__main__":
    main()
