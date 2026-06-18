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


def test_onebrain_clause_parity_with_rf_oracle():
    """Recursive embedded clause: a fact whose patient is an SVO clause ('dog go (cat look south)') stores + decodes on
    the OneBrainComposer == the RFPhasorComposer numpy oracle == ground truth, via BOTH query_patient (the decoded
    inner clause sentence) AND render_fact (the outer fact with the inner clause filling the patient slot). This brings
    the rf composer's recursive-clause feature to parity on the one-brain path (toward retiring the legacy numpy
    production runtime while keeping numpy as the oracle). The on-bridge decode is a chained register->register unbind
    (outer patient -> a Q register -> the 3 clause roles -> cleanup)."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.rf_phasor_composer import RFPhasorComposer, Clause
    clause = Clause(agent="cat", action="look", patient="south")   # all of dog/go/cat/look/south are in VOCAB
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB)
        oracle = RFPhasorComposer(seed=42, D=64, vocab=VOCAB)       # same seed/D/period -> identical codes
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    c.store("dog", "go", clause)
    oracle.store("dog", "go", clause)
    # query_patient: the decoded inner clause sentence
    got = c.query_patient("dog", "go")
    assert got == oracle.query_patient("dog", "go") == "cat look south", \
        f"clause query_patient {got!r} != oracle {oracle.query_patient('dog', 'go')!r} != truth 'cat look south'"
    # render_fact: the outer fact with the clause in the patient slot
    gotr = c.render_fact("dog")
    assert gotr == oracle.render_fact("dog") == "dog go cat look south", \
        f"clause render_fact {gotr!r} != oracle {oracle.render_fact('dog')!r} != truth 'dog go cat look south'"
    # the no-confab moat still holds for an unstored cue (abstain before any clause decode)
    assert c.query_patient("apple", "stop") is None, "moat breach: unstored cue not abstained"


def test_onebrain_agent_clause_fact():
    """The agent path: hear_clause_fact stores an embedded-clause fact on the OneBrainComposer; what_does decodes the
    inner clause + describe renders the outer fact; an unknown agent still abstains (the moat). Uses the agent's own
    core_sim_composition.Clause (a DISTINCT namedtuple from the rf module's -- the duck-typed _is_clause spans both)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    from research.runners.core_sim_composition import Clause
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="onebrain", concepts={w: None for w in VOCAB})
        a.hear_clause_fact("dog", "go", Clause(agent="cat", action="look", patient="south"))
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert a.what_does("dog", "go") == "cat look south", "agent must decode the embedded clause patient"
    assert a.describe("dog") == "dog go cat look south", "agent must render the outer fact with the inner clause"
    assert a.what_does("bird", "go") is None, "moat breach: unknown agent not abstained"


def test_onebrain_reconsolidation_parity():
    """Reconsolidation (prediction-error-gated in-place fact update) on the OneBrainComposer == the RFPhasorComposer
    numpy oracle. A corrective utterance reactivates the cued fact and -- only above the labilization gate --
    REWRITES the patient in place (no contradictory duplicate); a re-statement restabilizes; a never-stored cue
    abstains (the no-confab moat). The in-place rewrite re-composes the fact and overwrites the same store block.
    Brings the rf composer's reconsolidation to parity on the one-brain path (toward retiring the numpy runtime)."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.rf_phasor_composer import RFPhasorComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB)
        oracle = RFPhasorComposer(seed=42, D=64, vocab=VOCAB)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for (a, v, p) in facts:
        c.store(a, v, p); oracle.store(a, v, p)
    # (1) a CORRECTION ('actually, dog go south') -> rewrite in place (== oracle), no duplicate
    r = c.update_on_mismatch("dog", "go", "south")
    ro = oracle.update_on_mismatch("dog", "go", "south")
    assert r["action"] == ro["action"] == "rewrite", f"correction must rewrite: onebrain {r} vs oracle {ro}"
    assert c.query_patient("dog", "go") == "south", "rewritten fact must read the new patient"
    assert c.count_facts("dog", "go") == 1, "rewrite must not append a contradictory duplicate"
    # (2) a RE-STATEMENT ('cat come east' again) -> PE below the gate -> restabilize unchanged
    r2 = c.update_on_mismatch("cat", "come", "east")
    assert r2["action"] == "restabilize", f"a re-statement must restabilize, not rewrite: {r2}"
    assert c.query_patient("cat", "come") == "east" and c.count_facts("cat", "come") == 1
    # (3) the moat: a NEVER-stored cue abstains (no fabricated trace)
    rm = c.update_on_mismatch("bird", "go", "west")
    assert rm["action"] == "abstain" and c.count_facts("bird", "go") == 0, "moat breach: unstored cue not abstained"


def test_onebrain_batched_equals_per_block():
    """A5 lever 1: the BATCHED read (default, read all blocks in 3 windows) == the per-block oracle (enable_batched
    toggled off) on the production OneBrainComposer -- answer-identical, just faster (the de-risk: 7.3x)."""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for (a, v, p) in facts:
        c.store(a, v, p)                                  # store() resolves roles directly (no parser needed)
    for (a, v, p) in facts:
        c.enable_batched = True
        bat = (c.query_patient(a, v), c.query_agent(v, p), c.ask_yes_no(a, v, p))
        c.enable_batched = False
        per = (c.query_patient(a, v), c.query_agent(v, p), c.ask_yes_no(a, v, p))
        assert bat == per == (p, a, "yes"), f"batched {bat} != per-block {per} != truth for {(a, v, p)}"
    # moat parity (absent cue)
    c.enable_batched = True
    assert c.query_patient("apple", "stop") is None
    c.enable_batched = False
    assert c.query_patient("apple", "stop") is None


def test_onebrain_default_path_unaffected():
    """The additive wiring must not change the default ('rf') agent: it has no `hear` on its composer, so it builds the
    agent's own parser and uses parse+store (the byte-unchanged path). A construction smoke (no GPU run needed)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    import inspect
    src = inspect.getsource(BrainConversationalAgent.hear)
    assert "self.composer.hear" in src and "self.parser.parse" in src, \
        "hear() must keep BOTH the delegation path (onebrain) and the parse+store path (rf/rate default)"
