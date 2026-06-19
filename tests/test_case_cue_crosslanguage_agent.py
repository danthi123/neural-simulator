"""CI guard for the production wire-in of the Phase-2 CASE-aware multi-cue role-COMPETITION parser into
`BrainConversationalAgent`, behind the default-OFF `enable_case_competition` flag.

De-risk GO: 2026-06-19-case-cue-crosslanguage-derisk.md (case path 5/6 seeds; the cross-linguistic dissociation
6/6; all anti-cheat controls; moat 0 breaches). Wire-in: 2026-06-19-case-cue-crosslanguage-agent-wirein.md.

The Phase-2 win: adding a CASE cue (nominative particle -> +1 agent, accusative -> -1 patient) to the validated
role competition reads thematic roles by the case PARTICLE on a FREE-word-order CASE-MARKED toy (Japanese-style
ga/wo) where word-position cannot -- AND the SAME parser is language-general: case goes silent on an English
(unmarked) sentence so it falls back to position+semantics. The full LEARNED-weight cross-linguistic flip
(English w_case -> floor, Japanese w_case -> top) is the de-risk's headline on the three-factor LEARNER; the
production agent uses the validated INSTALL-path (fixed case-language validities), so this guard asserts the
install-path dissociation: case is what makes the free-order sentence solvable (a case-FREE Japanese-toy item is
content-ambiguous -> the plain multicue path abstains; the case particle resolves it), and case is silent on
English.

What this asserts (CPU/numpy-runnable; the agent uses the rf composer with an explicit vocab so the denoise64
cache is NOT needed):

  * CAPABILITY (flag ON): the agent answers who/what CORRECTLY on a FREE-WORD-ORDER object-fronted CASE-MARKED
    sentence 'wolf wo dog ga chase' -- the case particle (dog+ga = nominative -> agent, wolf+wo = accusative ->
    patient) OVERRIDES the surface position, so it stores dog=agent / wolf=patient and who_does('chase','wolf')
    == 'dog', what_does('dog','chase') == 'wolf'. A position-only read of the same surface (after particle strip:
    [wolf, dog]) would map wolf -> agent and get it BACKWARDS.

  * THE CROSS-LINGUISTIC DISSOCIATION in the production path:
      - case DECIDES on the case-marked free-order toy (above);
      - case is SILENT on English (no particles) -> the SAME case-aware parser falls back to position+semantics,
        so a canonical English 'dog eat apple' still reads dog=agent / apple=patient;
      - case is what makes free order solvable: a case-FREE Japanese-toy item (two animate nouns + a symmetric
        verb, object-front) is content-ambiguous -> a plain position+semantic parser ABSTAINS, while the case
        particle resolves the identical item -- the install-path signature of the dissociation.

  * NO-REGRESSION on clean canonical case (flag ON): 'dog ga wolf wo chase' (canonical SOV, case-marked) stores
    dog=agent / wolf=patient. The case cue does not break the canonical order.

  * MOAT (flag ON, never weakened): (a) an unstored fact -> abstain (None); (b) an UNMARKED ambiguous transitive
    (two animate nouns + a symmetric verb, NO case particles) -> parse_decisive reports decisive=False (case
    silent + animacy ties + symmetric verb -> no decisive content cue); the case-MARKED counterpart -> decisive
    True. Zero confabulation.

  * FLAG-OFF byte-identity: with the flag default-OFF the case parser is never constructed and hear() takes the
    unchanged path; the existing test_brain_conversational_agent.py passes verbatim (that file is the full
    byte-identity guard). Here we additionally assert the parser is not built and the flag is OFF, AND that a
    plain MultiCueRoleParser stays un-regressed alongside (the global-isolation guard).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners.brain_conversational_agent import BrainConversationalAgent

# rf composer with an explicit vocab -> no denoise64 cache needed. The Japanese-toy lexicon: ALL nouns animate
# (so animacy/verb-fit TIE -> the case particle is the SOLE reliable role cue, the canonical Competition-Model
# Japanese test) + symmetric transitive verbs. English nouns (animate agents + inanimate patients) are included
# so the case-aware parser's English fallback (case silent) has a decisive semantic cue.
JP_NOUNS = ["dog", "cat", "fox", "bird", "wolf", "bear", "owl", "frog"]   # all animate
EN_PATIENTS = ["apple", "ball", "rock", "book"]                          # inanimate (for the English fallback)
VERBS = ["chase", "watch", "follow", "greet", "eat"]                     # chase/watch/follow/greet symmetric; eat asym
VOCAB = {w: None for w in JP_NOUNS + EN_PATIENTS + VERBS}
SEED = 42  # the validated GO seed


def _on_agent():
    """A BrainConversationalAgent with the CASE-aware role-competition ON (rf composer, explicit vocab, default
    Japanese-style ga/wo case lexicon)."""
    return BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB,
                                    enable_case_competition=True, case_verbs=VERBS)


def test_case_resolves_free_word_order_object_fronted():
    """Flag ON: the agent comprehends a FREE-WORD-ORDER object-fronted CASE-MARKED 'wolf wo dog ga chase'
    correctly -- the case particle overrides position (dog+ga = agent, wolf+wo = patient) -- and answers who/what
    via the right roles, where a position-only read of the surface would invert them."""
    a = _on_agent()
    roles = a.hear("wolf wo dog ga chase")                # object-fronted + case: dog=agent, wolf=patient
    assert roles["agent"] == "dog"
    assert roles["patient"] == "wolf"
    assert roles["action"] == "chase"
    assert a.who_does("chase", "wolf") == "dog"           # CORRECT agent on free word order (by case)
    assert a.what_does("dog", "chase") == "wolf"          # CORRECT patient on free word order (by case)
    # NOT the inverted (position-only) answer: a surface-position read ([wolf, dog]) would store wolf=agent, so
    # who_does('chase','dog') would have returned 'wolf'; the case path returns None there.
    assert a.who_does("chase", "dog") is None


def test_case_silent_on_english_falls_back_to_position_semantics():
    """THE DISSOCIATION (English arm): the SAME case-aware parser, on an English (unmarked) canonical sentence
    'dog eat apple', has the case cue SILENT (no particles) and falls back to position+semantics -> still reads
    dog=agent / apple=patient. Same code, two languages: case present -> case decides; case absent -> the other
    cues decide."""
    a = _on_agent()
    roles = a.hear("dog eat apple")                       # English, no ga/wo -> case silent
    assert roles["agent"] == "dog"
    assert roles["patient"] == "apple"
    assert a.who_does("eat", "apple") == "dog"
    assert a.what_does("dog", "eat") == "apple"


def test_case_is_what_makes_free_order_solvable():
    """THE DISSOCIATION (case-load-bearing arm): case is exactly what makes a free-order Japanese-toy item
    solvable. A case-FREE item (two animate nouns + a symmetric verb, object-front) is content-ambiguous, so the
    case-aware parser with NO particle reports decisive=False (position-only can't, semantics tie); the IDENTICAL
    item WITH the ga/wo particles is decisive and reads correctly. The case particle's presence flips
    abstain <-> decide -- the install-path signature of the cross-linguistic dissociation."""
    a = _on_agent()
    parser = a._ensure_case_parser()
    # case-FREE, two animate + symmetric verb, object-front surface [wolf, dog]: no decisive content cue -> abstain
    _roles_uc, decisive_uc = parser.parse_decisive(["wolf", "dog", "chase"])
    assert decisive_uc is False                           # case silent + animacy ties + symmetric -> moat
    # the IDENTICAL item WITH case particles: decisive AND content-correct (dog+ga = agent, wolf+wo = patient)
    roles_c, decisive_c = parser.parse_decisive("wolf wo dog ga chase")
    assert decisive_c is True
    assert roles_c["agent"] == "dog"
    assert roles_c["patient"] == "wolf"


def test_case_no_regression_on_canonical_sov():
    """Flag ON: a clean canonical SOV case-marked sentence 'dog ga wolf wo chase' still comprehends correctly --
    the case cue does not break the canonical (toy-majority) order."""
    a = _on_agent()
    roles = a.hear("dog ga wolf wo chase")                # canonical SOV, case-marked
    assert roles["agent"] == "dog"
    assert roles["patient"] == "wolf"
    assert a.who_does("chase", "wolf") == "dog"
    assert a.what_does("dog", "chase") == "wolf"


def test_case_moat_abstains_on_unstored_fact():
    """Flag ON: the no-confab moat -- a query about an unstored fact abstains (None). Zero confabulation."""
    a = _on_agent()
    a.hear("wolf wo dog ga chase")                        # only this fact stored
    assert a.what_does("fox", "chase") is None            # fox has no fact -> abstain
    assert a.who_does("watch", "bear") is None            # no watch fact at all -> abstain


def test_case_moat_unmarked_ambiguous_not_decisive():
    """Flag ON: an UNMARKED ambiguous transitive (two animate nouns + the symmetric verb 'chase', NO case
    particles) -> the content gate reports decisive=False (case silent, animacy ties, verb symmetric), so the
    caller can ABSTAIN rather than confabulate. The case-MARKED counterpart reports decisive=True."""
    a = _on_agent()
    parser = a._ensure_case_parser()
    _roles, decisive = parser.parse_decisive(["dog", "wolf", "chase"])      # unmarked, both animate + symmetric
    assert decisive is False                              # content cannot break the tie -> moat
    _roles2, decisive2 = parser.parse_decisive("dog ga wolf wo chase")     # case-marked -> decisive
    assert decisive2 is True


def test_flag_off_parser_not_built_and_default_path():
    """Flag default-OFF: the case parser is never constructed and hear() takes the unchanged path. The full
    byte-identity guard is test_brain_conversational_agent.py; here we assert the flag is OFF + the parser absent.
    (Built with the rf composer + explicit vocab so it needs no cache.)"""
    a = BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB)   # flag default OFF
    assert a.enable_case_competition is False
    assert a._case_parser is None                         # parser never built when the flag is off
    a.hear("dog eat apple")                               # canonical English -> the default (position) path
    assert a.what_does("dog", "eat") == "apple"


def test_enable_case_requires_verbs():
    """enable_case_competition=True without case_verbs is a clear construction error (the lexical front-end needs
    the known-verb set to find the sentence's verb)."""
    with pytest.raises(ValueError):
        BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB,
                                 enable_case_competition=True)   # missing case_verbs


def test_case_parser_does_not_contaminate_plain_multicue():
    """Global-isolation guard: building + using the CASE-aware parser must NOT permanently mutate the Phase-1
    module globals (CUES/SEMANTIC_CUES), so a co-resident plain MultiCueRoleParser stays un-regressed (its moat
    _semantic_contrast would otherwise KeyError on a missing ev['case']). Build the case parser, use it, THEN
    build + use a plain MultiCueRoleParser and confirm it reads + abstains correctly."""
    import research.runners._phaseB_multicue_competition_spiking_derisk as P1
    from research.runners.multicue_role_parser import MultiCueRoleParser

    case_cues_before = tuple(P1.CUES)
    cp = BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB,
                                  enable_case_competition=True, case_verbs=VERBS)
    cp.hear("wolf wo dog ga chase")                       # exercise the case path
    assert tuple(P1.CUES) == case_cues_before             # globals restored after the case-parser call
    assert "case" not in P1.CUES                          # the plain Phase-1 cue set is 4-cue (no case)

    mc = MultiCueRoleParser(known_verbs=VERBS, seed=SEED)  # plain parser built AFTER the case parser
    en = mc.parse("apple eat dog")                        # English object-front (animate+inanimate) -> semantics
    assert en["agent"] == "dog" and en["patient"] == "apple"
    _r, dec = mc.parse_decisive(["dog", "chase", "cat"])  # two animate + symmetric -> not decisive (moat intact)
    assert dec is False
