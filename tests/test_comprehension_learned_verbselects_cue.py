"""CI guard — comprehension cue-lexicon conversion (2026-08-27): the open-vocab VERB_SELECTS patient-slot cue
that lifts the D4/D6/D3/T1-6/D2 comprehension organs' 8-verb hand-table vocab ceiling is LEARNED from real
TinyStories co-occurrence (6-seed GO, `research/findings/raw/_comprehension_learned_verbselects_cue_6seed.json`)
and SPIKING-REALIZED via gap#3-A1's F_anim/F_inanim coincidence pools
(`_comprehension_learned_verbselects_cue_derisk.py`), wired into `comprehension_production_organ.py` behind
`BRAIN_LEARNED_VERB_SELECTS` (default OFF). Mirrors `test_comprehension_learned_animacy_cue.py` exactly, one
level up (the VERB, not a noun, is the extended cue).

Pins: (1) the flag defaults OFF and is byte-identical to the pre-existing hand-table-only scope; (2) a
held-out verb the hand VERB_SELECTS table lacks becomes COMPETENT + judged only when the flag is ON; (3)
lesioning the F_anim/F_inanim coupling (`BRAIN_LEARNED_VERB_SELECTS_LESION=1`) reverts that verb to
byte-identical flag-OFF behaviour; (4) the moat holds — a genuinely off-graph (never-seen) verb still
abstains, flag on or off; (5) a REGRESSION GUARD (verify-go, 2026-08-27): with the flag ON, an EXISTING
hand-covered verb's INFLECTED surface form must still lemmatize to its correct hand-table base, never to a
learned-lexicon read of the inflected form itself (caught: "pushed" resolved to itself unlemmatized, "bites"
resolved to "bit", "kicked" resolved to itself -- all because the learned lexicon's corpus-wide vocabulary
covers those surface tokens too, and was consulted before the hand table was tried exhaustively). CPU/numpy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.comprehension_production_organ as CO
from research.runners._comprehension_learned_verbselects_cue_derisk import LearnedVerbSelectsLexicon

HELD_OUT_VERB = "clean"   # not in the hand VERB_SELECTS table; in the learned lexicon's corpus vocab
TEXT = "the dog clean the cup"   # dog (hand-table animate) + clean (held-out inanim-patient verb) + cup (hand-table inanimate)


def _clear_flags():
    os.environ.pop("BRAIN_LEARNED_VERB_SELECTS", None)
    os.environ.pop("BRAIN_LEARNED_VERB_SELECTS_LESION", None)


def test_held_out_verb_not_in_hand_table():
    assert HELD_OUT_VERB not in CO.VERB_SELECTS


def test_learned_lexicon_classifies_held_out_verb():
    lex = LearnedVerbSelectsLexicon(seed=42)
    assert lex.classify(HELD_OUT_VERB) == "inanimate_patient"


def test_learned_lexicon_abstains_off_graph():
    lex = LearnedVerbSelectsLexicon(seed=42)
    assert lex.classify("zzznotarealword999") is None      # genuinely off the learned graph -> abstain (moat)


def test_learned_lexicon_lesion_collapses_to_abstain():
    lex = LearnedVerbSelectsLexicon(seed=42)
    assert lex.classify(HELD_OUT_VERB) is not None
    lex.set_lesion(True)
    assert lex.classify(HELD_OUT_VERB) is None                # coupling zeroed -> every verb abstains
    lex.set_lesion(False)
    assert lex.classify(HELD_OUT_VERB) is not None             # un-lesioning restores it


def test_flag_off_byte_identical_to_pre_existing_scope():
    _clear_flags()
    organ = CO.get_organ(seed=42)
    tr = CO.extract_transitive(TEXT)
    assert tr == ("dog", HELD_OUT_VERB, "cup")
    assert organ.competent(*tr) is False                     # unrecognized (hand table lacks 'clean') -> out of scope
    assert organ.judge(TEXT) is None                          # byte-identical: unchanged turn


def test_flag_on_extends_coverage_load_bearing():
    _clear_flags()
    os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
    try:
        organ = CO.get_organ(seed=42)
        tr = CO.extract_transitive(TEXT)
        assert organ.competent(*tr) is True                  # now covered via the learned lexicon
        j = organ.judge(TEXT)
        assert j is not None and j["in_scope"] is True
        assert j["comprehended"] is True                      # a well-formed transitive resolves cleanly
    finally:
        _clear_flags()


def test_flag_on_lesioned_reverts_byte_identical():
    _clear_flags()
    os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
    os.environ["BRAIN_LEARNED_VERB_SELECTS_LESION"] = "1"
    try:
        organ = CO.get_organ(seed=42)
        tr = CO.extract_transitive(TEXT)
        assert organ.competent(*tr) is False                  # the load-bearing lesion: coverage reverts
        assert organ.judge(TEXT) is None                       # matches the flag-OFF result exactly
    finally:
        _clear_flags()


def test_moat_genuinely_oov_still_abstains_flag_on():
    _clear_flags()
    os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
    try:
        organ = CO.get_organ(seed=42)
        text = "the wug blickets the glorp"                   # verb AND both nouns unknown -> fully OOV
        j = organ.judge(text)
        assert j is not None and j["comprehended"] is False    # low margin, honestly uncomprehended
        rt = organ.repair_target(text)
        assert rt["kind"] == "oov"
        assert set(rt["oov_tokens"]) >= {"wug", "blickets", "glorp"}
    finally:
        _clear_flags()


def test_flag_on_does_not_break_existing_inflected_hand_verbs():
    """REGRESSION GUARD (verify-go, 2026-08-27): with the flag ON, an inflected surface form of an EXISTING
    hand-table verb must lemmatize to the SAME correct base it does with the flag off -- never to the
    learned lexicon's own (unrelated) read of the raw inflected token. Caught: "pushed"->"pushed" (should be
    "push"), "bites"->"bit" (should be "bite"), "kicked"->"kicked" (should be "kick") -- all three surface
    forms happen to be real corpus words with their own learned scores, and the pre-fix `_lemma_verb` tried
    the learned lexicon (via `_verb_selects_of` on each candidate, in order) before it had exhausted the hand
    table. Fixed by trying every candidate against the hand table FIRST, in full, before ever falling back to
    the learned lexicon."""
    _clear_flags()
    try:
        assert CO._lemma_verb("pushed") == "push"
        assert CO._lemma_verb("bites") == "bite"
        assert CO._lemma_verb("kicked") == "kick"
        os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
        assert CO._lemma_verb("pushed") == "push"
        assert CO._lemma_verb("bites") == "bite"
        assert CO._lemma_verb("kicked") == "kick"
        # every hand-table verb's -s/-ed/-ing inflection should be unaffected by the flag, both directions
        for base, inflected in (("chase", "chases"), ("eat", "eats"), ("carry", "carries"),
                                 ("grab", "grabs"), ("watch", "watches")):
            os.environ.pop("BRAIN_LEARNED_VERB_SELECTS", None)
            off = CO._lemma_verb(inflected)
            os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
            on = CO._lemma_verb(inflected)
            assert off == on == base, (base, inflected, off, on)
    finally:
        _clear_flags()
