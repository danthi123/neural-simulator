"""CI guard for the COMMUNICABLE-BRAIN Stage B wire-in (the production-agent opt-in + talkativeness-Q
persistence). Fast, CPU-only (numpy): asserts the DEFAULT-OFF byte-identity contract (the orchestrator is never
constructed, the existing API is untouched, converse() raises when OFF), the opt-in plumbing through
BrainConversationalAgent + MultiTurnAgent, and the talkativeness-Q round-trip through the developed-brain bundle
(save_developed_brain / load_developed_brain). The HEAVY end-to-end communicable gate (the PPMI brain + the
proposer + the spiking speak race + the Stage A invariants in composition) is the runner
`research/runners/_communicable_turn_stageB_wirein.py` (GO 3-seed) -- this test guards the wire-in SURFACE so a
regression of the default path / the persistence path is caught in CI without the ~96s/seed brain build.

Default-OFF byte-identity is the load-bearing claim: a default agent must behave EXACTLY as before, which the
existing tests/test_brain_conversational_agent.py + tests/test_multi_turn_agent.py already pin; this adds the
no-build + opt-in-surface assertions the wire-in introduced.
"""
import os
import tempfile
import shutil

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.multi_turn_agent import MultiTurnAgent

NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat", "go", "come", "north", "south"]


def _agent(**kw):
    return BrainConversationalAgent(seed=42, concepts={w: None for w in VOCAB}, composer_kind="rf",
                                    enable_neural_render=False, **kw)


# ---- (1) DEFAULT-OFF byte-identity: the orchestrator is NEVER built; the existing API is unchanged -------------
def test_default_off_does_not_build_orchestrator():
    a = _agent()
    assert a.communicable_mode is False
    assert a._communicable is None            # the CommunicableTurn is NEVER constructed at the default
    assert a._communicable_brain is None
    # the existing comprehend/store/recall/abstain path is byte-unchanged
    a.composer.kb = []
    a.hear("dog go north")
    a.hear("cat come south", polarity="AFFIRM")
    assert a.what_does("dog", "go") == "north"
    assert a.who_does("come", "south") == "cat"
    assert a.what_does("bird", "eat") is None          # the no-confab moat (abstain)
    assert a.is_it_true("cat", "come", "south") == "yes"


def test_converse_raises_when_off():
    a = _agent()
    with pytest.raises(RuntimeError):
        a.converse("hi")                       # the orchestrator is intentionally not built at the default
    # speak_value_Q() is safe even when off (no orchestrator) -> {}
    assert a.speak_value_Q() == {}


def test_multiturn_default_off_passthrough():
    mt = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42)
    assert mt.communicable_mode is False
    assert mt.agent._communicable is None
    with pytest.raises(RuntimeError):
        mt.converse("hi")
    assert mt.speak_value_Q() == {}


# ---- (2) OPT-IN plumbing: the flag/runtime-enable sets the state but DOES NOT build until first use ------------
def test_opt_in_flag_defers_build():
    a = _agent(communicable_mode=True, communicable_draw="host")
    assert a.communicable_mode is True
    assert a._communicable is None             # lazy: built on the first converse()/feedback()/speak_value_Q-after-build
    assert a._communicable_draw == "host"


def test_runtime_enable():
    a = _agent()
    assert a.communicable_mode is False
    a.enable_communicable_mode(draw="host", speak_value_Q={"dog": 0.7})
    assert a.communicable_mode is True
    assert a._communicable_draw == "host"
    # the seeded Q is available even before the orchestrator is built (the getter fallback)
    assert abs(a.speak_value_Q().get("dog", 0.0) - 0.7) < 1e-9


def test_seed_Q_roundtrips_through_getter():
    """A seeded talkativeness Q is returned by speak_value_Q() before any turn (so a save-before-any-turn still
    round-trips a restored Q)."""
    a = _agent(communicable_mode=True, communicable_draw="host", speak_value_Q={"cat": 0.4, "fish": 0.9})
    Q = a.speak_value_Q()
    assert abs(Q.get("cat", 0.0) - 0.4) < 1e-9
    assert abs(Q.get("fish", 0.0) - 0.9) < 1e-9


# ---- (3) Q-PERSISTENCE through the developed-brain bundle (the manifest field + the json file) -----------------
def test_speak_value_Q_persists_through_bundle():
    """save_developed_brain writes speak_value_Q.json + the manifest n_speak_value_Q; load_developed_brain restores
    it onto the rebuilt agent. (A directly-seeded Q stands in for a feedback-learned one -- the persistence path is
    identical; the LEARNED-from-feedback Q is exercised end-to-end by the Stage B runner.)"""
    from research.runners.developed_brain_io import save_developed_brain, load_developed_brain
    a = _agent(communicable_mode=True, communicable_draw="host", speak_value_Q={"dog": 0.6, "cat": 0.3})
    a.composer.kb = []
    a.composer.store("dog", "go", "north", polarity="AFFIRM")    # a fact so the bundle has knowledge to reload
    tmp = tempfile.mkdtemp(prefix="commB_ci_")
    try:
        bundle = os.path.join(tmp, "brain")
        manifest = save_developed_brain(a, bundle, seed=42, composer_kind="rf")
        assert manifest["n_speak_value_Q"] == 2
        assert os.path.exists(os.path.join(bundle, "speak_value_Q.json"))
        a2, _m = load_developed_brain(bundle, communicable_mode=True, communicable_draw="host")
        Q2 = a2.speak_value_Q()
        assert abs(Q2.get("dog", 0.0) - 0.6) < 1e-6
        assert abs(Q2.get("cat", 0.0) - 0.3) < 1e-6
        # the reloaded brain still recalls its fact + abstains (the moat survives the bundle round-trip)
        assert a2.what_does("dog", "go") == "north"
        assert a2.what_does("cat", "eat") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_non_communicable_bundle_has_no_Q_file():
    """A NON-communicable agent's bundle is byte-unchanged: no speak_value_Q.json, manifest n_speak_value_Q == 0."""
    from research.runners.developed_brain_io import save_developed_brain
    a = _agent()                               # communicable_mode OFF
    a.composer.kb = []
    a.composer.store("dog", "go", "north", polarity="AFFIRM")
    tmp = tempfile.mkdtemp(prefix="commB_ci_noq_")
    try:
        bundle = os.path.join(tmp, "brain")
        manifest = save_developed_brain(a, bundle, seed=42, composer_kind="rf")
        assert manifest["n_speak_value_Q"] == 0
        assert not os.path.exists(os.path.join(bundle, "speak_value_Q.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
