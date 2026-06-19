"""CI guard for the production wire-in of the SPIKING multi-cue role-COMPETITION parser into
`BrainConversationalAgent`, behind the default-OFF `enable_multicue_competition` flag.

De-risk GO: 2026-06-19-multicue-competition-spiking-derisk.md (install path 5/6 seeds; all anti-cheat controls;
moat 0 breaches on every seed). Wire-in: 2026-06-19-multicue-competition-agent-wirein.md.

What this asserts (CPU/numpy-runnable; the agent uses the rf composer with an explicit vocab so the denoise64
cache is NOT needed):

  * CAPABILITY (flag ON): the agent answers who/what CORRECTLY on a DEGRADED (object-fronted) sentence
    'apple eat dog' -- it stores dog=agent / apple=patient (content overrides position), so who_does('eat','apple')
    == 'dog' and what_does('dog','eat') == 'apple'. This is exactly the input where the default POSITION-ONLY path
    assigns the roles BACKWARDS (agent=apple / patient=dog) -- shown explicitly via the parser's position-by-
    construction ground-truth map, the load-bearing contrast.

  * NO-REGRESSION on clean canonical (flag ON): 'cat eat ball' (canonical SVO) still stores cat=agent / ball=
    patient -> who_does('eat','ball') == 'cat', what_does('cat','eat') == 'ball'. The multi-cue parser does not
    break the native case.

  * MOAT (flag ON, never weakened): (a) an unstored fact -> abstain (None); (b) an all-ambiguous transitive (two
    animate nouns + a symmetric verb, e.g. 'dog chase cat') -> parse_decisive reports decisive=False (the content
    gate refuses to confabulate a role assignment). Zero confabulation.

  * FLAG-OFF byte-identity: with the flag default-OFF the multi-cue parser is never constructed and hear() takes
    the unchanged path; the existing test_brain_conversational_agent.py passes verbatim (that file is the full
    byte-identity guard). Here we additionally assert the parser is not built and the flag is OFF.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners.brain_conversational_agent import BrainConversationalAgent, _GT

# rf composer with an explicit vocab -> no denoise64 cache needed. Animate agents + inanimate patients so the
# semantic cues (animacy + verb-selectional-fit) are decisive on a degraded order.
NOUNS = ["dog", "cat", "fox", "bird", "apple", "ball", "rock", "book"]
VERBS = ["eat", "chase", "push", "carry"]
VOCAB = {w: None for w in NOUNS + VERBS}
SEED = 42  # the validated GO seed


def _on_agent():
    """A BrainConversationalAgent with the multi-cue role-competition ON (rf composer, explicit vocab)."""
    return BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB,
                                    enable_multicue_competition=True, multicue_verbs=VERBS)


def _position_only_roles(words):
    """The DEFAULT position-only parser's decision, which is purely position-by-construction (_GT maps
    position*2+voice -> role; pos0=agent, pos1=action, pos2=patient). This is what the agent stores WITHOUT the
    flag -- the load-bearing contrast (it inverts an object-fronted sentence)."""
    return {_GT[pos * 2]: w for pos, w in enumerate(words)}


def test_default_position_only_parser_inverts_object_fronted():
    """The load-bearing CONTRAST: the default position-only parser assigns the object-fronted 'apple eat dog'
    BACKWARDS (agent=apple / patient=dog). This is the failure the multi-cue competition fixes."""
    r = _position_only_roles(["apple", "eat", "dog"])
    assert r["agent"] == "apple"      # WRONG: apple is the patient
    assert r["patient"] == "dog"      # WRONG: dog is the agent
    assert r["action"] == "eat"


def test_multicue_resolves_object_fronted_degraded_input():
    """Flag ON: the agent comprehends the DEGRADED object-fronted 'apple eat dog' correctly (content overrides
    position) and answers who/what via the right roles -- where the default path (above) gets it backwards."""
    a = _on_agent()
    roles = a.hear("apple eat dog")                       # object-fronted; content -> dog=agent, apple=patient
    assert roles["agent"] == "dog"
    assert roles["patient"] == "apple"
    assert roles["action"] == "eat"
    assert a.who_does("eat", "apple") == "dog"            # CORRECT agent on degraded order
    assert a.what_does("dog", "eat") == "apple"           # CORRECT patient on degraded order
    # the default would have stored apple=agent/dog=patient, so on the default path who_does('eat','apple') would
    # have been None and who_does('eat','dog') would have returned 'apple' (a DIFFERENT non-None answer); the
    # multi-cue path returns the content-correct answer.
    assert a.who_does("eat", "dog") is None               # NOT the inverted (default-path) answer


def test_multicue_no_regression_on_clean_canonical():
    """Flag ON: a clean canonical SVO ('cat eat ball') still comprehends correctly -- the multi-cue parser does
    not break the native word order."""
    a = _on_agent()
    roles = a.hear("cat eat ball")                        # canonical SVO
    assert roles["agent"] == "cat"
    assert roles["patient"] == "ball"
    assert a.who_does("eat", "ball") == "cat"
    assert a.what_does("cat", "eat") == "ball"


def test_multicue_moat_abstains_on_unstored_fact():
    """Flag ON: the no-confab moat -- a query about an unstored fact abstains (None). Zero confabulation."""
    a = _on_agent()
    a.hear("apple eat dog")                               # only this fact stored
    assert a.what_does("rock", "eat") is None             # rock has no fact -> abstain
    assert a.who_does("chase", "bird") is None            # no chase fact at all -> abstain


def test_multicue_moat_ambiguous_sentence_not_decisive():
    """Flag ON: an all-ambiguous transitive (two animate nouns + the symmetric verb 'chase') -> the content gate
    reports decisive=False, so the caller can ABSTAIN rather than confabulate a role assignment. The decisive
    counterpart ('cat eat apple', animate+inanimate) reports decisive=True."""
    a = _on_agent()
    parser = a._ensure_multicue_parser()
    _roles, decisive = parser.parse_decisive(["dog", "chase", "cat"])   # both animate + symmetric verb
    assert decisive is False                              # content cannot break the tie -> moat
    _roles2, decisive2 = parser.parse_decisive(["cat", "eat", "apple"]) # animate + inanimate + asymmetric verb
    assert decisive2 is True                              # content decisive


def test_flag_off_parser_not_built_and_default_path():
    """Flag default-OFF: the multi-cue parser is never constructed and hear() takes the unchanged path. The full
    byte-identity guard is test_brain_conversational_agent.py; here we assert the flag is OFF + the parser absent.
    (Built with the rf composer + explicit vocab so it needs no cache.)"""
    a = BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB)   # flag default OFF
    assert a.enable_multicue_competition is False
    assert a._multicue_parser is None                     # parser never built when the flag is off
    a.hear("dog eat apple")                               # canonical -> the default (position) path
    assert a.what_does("dog", "eat") == "apple"


def test_enable_multicue_requires_verbs():
    """enable_multicue_competition=True without multicue_verbs is a clear construction error (the lexical
    front-end needs the known-verb set to find the sentence's verb)."""
    with pytest.raises(ValueError):
        BrainConversationalAgent(seed=SEED, composer_kind="rf", concepts=VOCAB,
                                 enable_multicue_competition=True)   # missing multicue_verbs
