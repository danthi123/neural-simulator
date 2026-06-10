"""STEP 2b acceptance gate (b) — the conversational capability matrix + no-confab moat with the RF composer
CO-RESIDENT on the merged bridge.

This is the STEP-2a gate (`tests/test_nav_conv_merged_agent.py`) re-run with `co_resident_composer=True`: fact
storage / retrieval / question answering / abstention / negation / clauses / generation are now computed on a SLICE
of the SAME merged navigation+conversation bridge (the `rf` region), via the owner-approved sliced `rf_kick`, instead
of on a separate per-op bridge. The VERBATIM conversational assertions (incl. the three `is None` no-confab
assertions) MUST still pass.

Like the STEP-2a gate, this skips off-GPU (the merged bridge + the RF complex matvec at production D=128 run on CuPy;
numpy is a tiny-smoke/CI path only). On GPU — the acceptance environment — it does NOT skip.

Anti-cheat: the composer's binding actually ran on the merged bridge's rf slice (a silent fallback to a standalone
composer would fail) — the composer is a `MergedRFComposer` bound to the merged bridge, and after a store the merged
bridge carries the complex bind synapses (`cp_rf_w_re is not None`).
"""
import pytest

from sim.backend import is_gpu_backend
from research.runners.core_sim_composition import Clause


pytestmark = pytest.mark.skipif(
    not is_gpu_backend(),
    reason="STEP 2b co-resident composer runs the RF complex matvec at production D=128 on the merged bridge; "
           "GPU (CuPy) is the acceptance environment.")


@pytest.fixture(scope="module")
def agent():
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    return MergedNavConvAgent(seed=42, co_resident_composer=True)


# --- anti-cheat: the composer's binding ran on the merged bridge's rf slice (not a standalone fallback) -----------
def test_composer_is_co_resident_on_merged_bridge(agent):
    from research.runners.nav_conv_merged_bridge import MergedRFComposer
    region_names = agent._merged_bridge.region_manager.region_indices_dict()
    assert "rf" in region_names                                  # the composer slice co-resides on the merged bridge
    assert "cortex_N" in region_names                            # ... alongside navigation
    assert "parse_conj" in region_names                          # ... and the parser
    assert isinstance(agent.composer, MergedRFComposer)
    assert agent.composer._merged is agent._merged_bridge
    # a store drives the rf slice -> the merged bridge now carries the complex bind synapses
    agent.composer.kb = []
    agent.hear("dog go north")
    assert agent._merged_bridge.cp_rf_w_re is not None


# --- the VERBATIM conversational acceptance assertions (mirror tests/test_nav_conv_merged_agent.py) ---------------
def test_comprehend_store_and_qa(agent):
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("cat come south")
    assert agent.what_does("dog", "go") == "north"
    assert agent.what_does("cat", "come") == "south"
    assert agent.who_does("go", "north") == "dog"
    assert agent.what_does("river", "look") is None     # abstention (no-confab moat)


def test_voice_invariant_comprehension(agent):
    assert agent.parser.parse(["dog", "go", "north"], "active")["agent"] == "dog"
    assert agent.parser.parse(["north", "go", "dog"], "passive")["agent"] == "dog"


def test_negation_yes_no(agent):
    agent.composer.kb = []
    agent.hear("dog go north", polarity="AFFIRM")
    agent.hear("cat come south", polarity="NEGATE")
    assert agent.is_it_true("dog", "go", "north") == "yes"
    assert agent.is_it_true("cat", "come", "south") == "no"
    assert agent.is_it_true("apple", "stop", "west") == "unknown"


def test_embedded_clause(agent):
    agent.composer.kb = []
    agent.hear_clause_fact("dog", "look", Clause("cat", "go", "south"))
    agent.hear("apple stop west")
    assert agent.what_does("dog", "look") == "cat go south"
    assert agent.what_does("apple", "stop") == "west"


def test_dialogue_planning_elaborate(agent):
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("dog come south")
    associates = set(agent._assoc_graph().get("dog", {}))
    assert agent.elaborate("dog") in associates
    assert agent.elaborate("river") is None            # unconnected topic (no-confab moat)


def test_generation_describe(agent):
    agent.composer.kb = []
    agent.hear("dog go north")
    assert agent.describe("dog") == "dog go north"
    assert agent.describe("river") is None
