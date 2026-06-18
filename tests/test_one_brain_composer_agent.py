"""CI GUARD (roadmap phase 2, the real "one brain"): BrainConversationalAgent(composer_kind="onebrain") must keep
answering the core who/what/yes-no/moat matrix on the production OneBrainComposer -- the WHOLE pipeline (comprehend ->
store -> query -> abstain) on ONE persistent co-resident bridge, the agent delegating comprehension to the composer's
on-bridge parser (one parser on the one brain).

Why this test exists: the OneBrainComposer is the integrated one-brain conversational composer (2026-06-18-one-brain-
composer-A3-GO.md). Without a guard it silently bit-rots as the agent / composer / bridge code evolves. This pins the
core capability + the no-confab moat.

HONEST SCOPE: affirmative facts (who / what / affirmative yes-no + abstention). Negation (a bound polarity tag = a 4th
role) + the richer caps (describe / reason_chain / elaborate) are documented follow-ons, NOT asserted here.

GPU-only (the on-bridge parser trains on the CuPy substrate); skips gracefully without GPU / when the concept cache is
absent (like the other on-brain agent tests).
"""
import os

import pytest

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gpu_backend(),
                                reason="the OneBrainComposer's on-bridge parser needs the CuPy/GPU substrate")

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]


def _build(seed):
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    a = BrainConversationalAgent(seed=seed, composer_kind="onebrain", concepts={w: None for w in VOCAB})
    a.hear("dog go north", polarity="AFFIRM")
    a.hear("cat come east", polarity="AFFIRM")
    a.hear("bird look south", polarity="AFFIRM")
    a.hear("west stop river", voice="passive", polarity="AFFIRM")   # passive frame -> agent=river (voice-invariant)
    return a


def test_onebrain_agent_matrix_and_moat():
    try:
        a = _build(42)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")

    # who / what on the persistent on-bridge store
    assert a.what_does("dog", "go") == "north"
    assert a.who_does("go", "north") == "dog"
    assert a.what_does("cat", "come") == "east"
    # voice-invariant comprehension: the passively-heard "west stop river" stores (agent=river, action=stop,
    # patient=west) -- the passive frame flips 1st<->3rd -- so it queries back as river-stop-west
    assert a.what_does("river", "stop") == "west"
    assert a.who_does("stop", "west") == "river"
    # affirmative yes/no
    assert a.is_it_true("dog", "go", "north") == "yes"
    assert a.is_it_true("bird", "look", "south") == "yes"

    # the no-confab moat: an unheard cue abstains (what_does -> None), an unheard fact abstains (is_it_true -> unknown)
    assert a.what_does("apple", "stop") is None, "moat breach: unstored cue not abstained"
    assert a.is_it_true("cat", "go", "west") in ("unknown", "no"), "moat breach: unstored fact not abstained"


def test_onebrain_negation_yes_no():
    """Negation: a fact heard with polarity='NEGATE' (a bound 4th polarity role) -> is_it_true 'no'; an affirmative
    fact -> 'yes'; an unstored fact -> 'unknown' (the moat). who/what read the stored subject-verb-object regardless of
    polarity (only the yes/no answer flips), matching the rf composer's semantics."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="onebrain", concepts={w: None for w in VOCAB})
        a.hear("dog go north", polarity="AFFIRM")
        a.hear("cat come east", polarity="NEGATE")     # asserts: cat does NOT come east
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert a.is_it_true("dog", "go", "north") == "yes", "affirmative fact must answer yes"
    assert a.is_it_true("cat", "come", "east") == "no", "negated fact must answer no"
    assert a.is_it_true("dog", "go", "south") == "unknown", "moat breach: unstored fact not abstained"
    # who/what still read the stored SVO of the negated fact (only the polarity/yes-no flips)
    assert a.what_does("cat", "come") == "east"


def test_onebrain_describe_and_reason():
    """The richer caps via the agent: `describe` (generation -- render the stored fact for an agent, None on an unknown
    agent = no confabulation) and `reason_chain` (multi-hop -- each action's patient becomes the next hop's agent,
    abstaining the moment a hop has no fact)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="onebrain", concepts={w: None for w in VOCAB})
        a.hear("dog go cat")        # dog -go-> cat
        a.hear("cat go north")      # cat -go-> north
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert a.describe("dog") == "dog go cat", "describe must render the stored fact"
    assert a.describe("bird") is None, "moat breach: describe must not confabulate an unknown agent"
    assert a.reason_chain("dog", ["go", "go"]) == "north", "multi-hop: dog -go-> cat -go-> north"
    assert a.reason_chain("dog", ["go", "come"]) is None, "moat: no (cat, come) fact -> abstain at hop 2"


def test_onebrain_default_path_unaffected():
    """The additive wiring must not change the default ('rf') agent: it has no `hear` on its composer, so it builds the
    agent's own parser and uses parse+store (the byte-unchanged path). A construction smoke (no GPU run needed)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    import inspect
    src = inspect.getsource(BrainConversationalAgent.hear)
    assert "self.composer.hear" in src and "self.parser.parse" in src, \
        "hear() must keep BOTH the delegation path (onebrain) and the parse+store path (rf/rate default)"
