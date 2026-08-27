"""CI guard — comprehension cue-lexicon conversion (2026-08-27): the open-vocab ANIMACY cue that lifts the
D4/D6/D3/T1-6/D2 comprehension organs' ~19-noun hand-table vocab ceiling is LEARNED from real TinyStories
co-occurrence (6-seed GO, `research/findings/raw/_comprehension_learned_animacy_cue_6seed.json`) and
SPIKING-REALIZED via gap#3-A1's F_anim/F_inanim coincidence pools (`_comprehension_learned_animacy_spiking.py`),
wired into `comprehension_production_organ.py` behind `BRAIN_LEARNED_ANIMACY_CUE` (default OFF).

Pins: (1) the flag defaults OFF and is byte-identical to the pre-existing hand-table-only scope; (2) a
held-out animate word the hand ANIMACY table lacks becomes COMPETENT + judged only when the flag is ON;
(3) lesioning the F_anim/F_inanim coupling (`BRAIN_LEARNED_ANIMACY_LESION=1`) reverts that word to
byte-identical flag-OFF behaviour; (4) the moat holds — a genuinely off-graph (never-seen) token still
abstains, flag on or off. CPU/numpy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.comprehension_production_organ as CO
from research.runners._comprehension_learned_animacy_spiking import LearnedAnimacyLexicon

HELD_OUT_ANIMATE = "monkey"   # not in the hand ANIMACY table; in the learned lexicon's corpus vocab
TEXT = "the monkey carries the cup"   # monkey (held-out animate) + carry (asymm verb) + cup (hand-table inanimate)


def _clear_flags():
    os.environ.pop("BRAIN_LEARNED_ANIMACY_CUE", None)
    os.environ.pop("BRAIN_LEARNED_ANIMACY_LESION", None)


def test_held_out_word_not_in_hand_table():
    assert HELD_OUT_ANIMATE not in CO.ANIMACY


def test_learned_lexicon_classifies_held_out_word():
    lex = LearnedAnimacyLexicon(seed=42)
    assert lex.classify(HELD_OUT_ANIMATE) == "animate"


def test_learned_lexicon_abstains_off_graph():
    lex = LearnedAnimacyLexicon(seed=42)
    assert lex.classify("zzznotarealword999") is None      # genuinely off the learned graph -> abstain (moat)


def test_learned_lexicon_lesion_collapses_to_abstain():
    lex = LearnedAnimacyLexicon(seed=42)
    assert lex.classify(HELD_OUT_ANIMATE) is not None
    lex.set_lesion(True)
    assert lex.classify(HELD_OUT_ANIMATE) is None            # coupling zeroed -> every word abstains
    lex.set_lesion(False)
    assert lex.classify(HELD_OUT_ANIMATE) is not None        # un-lesioning restores it


def test_flag_off_byte_identical_to_pre_existing_scope():
    _clear_flags()
    organ = CO.get_organ(seed=42)
    tr = CO.extract_transitive(TEXT)
    assert tr == (HELD_OUT_ANIMATE, "carry", "cup")
    assert organ.competent(*tr) is False                    # unrecognized (hand table lacks 'monkey') -> out of scope
    assert organ.judge(TEXT) is None                         # byte-identical: unchanged turn


def test_flag_on_extends_coverage_load_bearing():
    _clear_flags()
    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    try:
        organ = CO.get_organ(seed=42)
        tr = CO.extract_transitive(TEXT)
        assert organ.competent(*tr) is True                  # now covered via the learned lexicon
        j = organ.judge(TEXT)
        assert j is not None and j["in_scope"] is True
        assert j["comprehended"] is True                     # a well-formed transitive resolves cleanly
    finally:
        _clear_flags()


def test_flag_on_lesioned_reverts_byte_identical():
    _clear_flags()
    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    os.environ["BRAIN_LEARNED_ANIMACY_LESION"] = "1"
    try:
        organ = CO.get_organ(seed=42)
        tr = CO.extract_transitive(TEXT)
        assert organ.competent(*tr) is False                 # the load-bearing lesion: coverage reverts
        assert organ.judge(TEXT) is None                      # matches the flag-OFF result exactly
    finally:
        _clear_flags()


def test_moat_genuinely_oov_still_abstains_flag_on():
    _clear_flags()
    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    try:
        organ = CO.get_organ(seed=42)
        text = "the wug blickets the glorp"                  # verb AND both nouns unknown -> fully OOV
        j = organ.judge(text)
        assert j is not None and j["comprehended"] is False   # low margin, honestly uncomprehended
        rt = organ.repair_target(text)
        assert rt["kind"] == "oov"
        assert set(rt["oov_tokens"]) >= {"wug", "glorp"}
    finally:
        _clear_flags()
