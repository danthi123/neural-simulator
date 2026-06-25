"""STEP 2a acceptance gate (b) — the conversational capability matrix + the no-confab moat on the MERGED bridge.

`docs/plans/2026-06-10-nav-conv-merge-implementation-design.md` §3 STEP 2a: the `MergedNavConvAgent` shim runs
comprehension (the PARSER) and dialogue planning (the dlPFC `elaborate`) on the MERGED nav+conv `SimulationBridge`,
with fact storage/retrieval delegated to a SEPARATE-bridge production `RFPhasorComposer`. This file runs the SAME
assertions as `tests/test_brain_conversational_agent.py` against the merged shim — they are the VERBATIM acceptance
assertions and MUST pass, especially the three `is None` no-confab assertions.

NO silent skip masking a fallback (design §3 anti-cheat): the production RF composer uses random PHASOR codes (NOT the
denoise64 cache), so the merged agent never raises FileNotFoundError — this file does NOT skip on a missing cache (a
silent skip would hide a non-functional merge). The ONLY skip is the legitimate environment gate: the merged bridge's
Hebbian parser is GPU-validated (the original suite's spiking-cleanup tests skip off-GPU for the same reason), so this
file skips off-GPU. On GPU (CuPy) — the acceptance environment — it does NOT skip.

The merged bridge (~2900 neurons) is built ONCE (module-scoped fixture); the composer kb is cleared between tests."""
import pytest

from sim.backend import is_gpu_backend
from research.runners.core_sim_composition import Clause


pytestmark = pytest.mark.skipif(
    not is_gpu_backend(),
    reason="MergedNavConvAgent's Hebbian parser is GPU-validated (numpy-backend KeyError in the parser bridge); "
           "the merged bridge + dlPFC run on CuPy. Run on GPU (the acceptance environment).")


# STEP 2a runs the composer on a SEPARATE per-op bridge (co_resident_composer=False), so co_resident_composer_kind is
# inert here. We still parametrize over it as a regression guard: the 2026-06-25 Closure-1 flip changed the kind DEFAULT
# (rf -> onebrain), and this asserts the STEP-2a agent constructs + passes the full matrix + the no-confab moat under
# BOTH the new onebrain default and the retained rf oracle.
@pytest.fixture(scope="module", params=["onebrain", "rf"], ids=["kind=onebrain", "kind=rf"])
def agent(request):
    """Build the merged nav+parser+dlPFC bridge + the separate RFPhasorComposer ONCE per composer-kind. NO try/except
    skip on a missing denoise64 cache: the RF composer uses random phasor codes, so this never raises FileNotFoundError —
    a silent skip would hide a non-functional merge (design §3 anti-cheat)."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    return MergedNavConvAgent(seed=42, co_resident_composer_kind=request.param)


# --- anti-cheat: the agent's parser + dlPFC actually run on the merged bridge (design §3) -----------------------
def test_merged_bridge_holds_nav_and_conv(agent):
    """The shim's parser + dlPFC run on the MERGED bridge holding BOTH brains — not a standalone fallback."""
    region_names = agent._merged_bridge.region_manager.region_indices_dict()
    # navigation region AND conversational regions co-reside on the ONE merged bridge
    assert "cortex_N" in region_names                     # a navigation region
    assert "parse_conj" in region_names                   # the parser slice
    assert "dlpfc_wm" in region_names                     # the dlPFC slice
    # elaborate drives the MERGED bridge's dlPFC context (not a throwaway dlPFC bridge)
    assert agent._dlpfc_ctx.bridge is agent._merged_bridge


# --- the VERBATIM conversational acceptance assertions (mirror tests/test_brain_conversational_agent.py) --------
def test_comprehend_store_and_qa(agent):
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("cat come south")
    assert agent.what_does("dog", "go") == "north"
    assert agent.what_does("cat", "come") == "south"
    assert agent.who_does("go", "north") == "dog"
    assert agent.what_does("river", "look") is None     # abstention (no-confab moat)


def test_voice_invariant_comprehension(agent):
    """The learned parser assigns the SAME agent in active and passive frames (the active<->passive role flip)."""
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
    """Dialogue planning: the dlPFC spiking content-selection Control (on the MERGED bridge's dlPFC slice) brings up an
    on-topic associate from the agent's own facts; abstains on an unconnected topic."""
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("dog come south")
    associates = set(agent._assoc_graph().get("dog", {}))
    assert agent.elaborate("dog") in associates       # an on-topic concept, chosen on the merged dlPFC slice
    assert agent.elaborate("river") is None            # unconnected topic (no-confab moat)


def test_generation_describe(agent):
    """Generation: the agent produces a sentence about a known subject from its spiking memory; abstains (None) on an
    unknown subject (no confabulation)."""
    agent.composer.kb = []
    agent.hear("dog go north")
    assert agent.describe("dog") == "dog go north"
    assert agent.describe("river") is None


def test_elaborate_cache_invalidates_on_graph_change(agent):
    """The dlPFC Control is cached on the graph CONTENT, so a new fact set of the SAME size cannot reuse a stale
    Control (regression guard for the cache-key fix)."""
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("dog come south")
    assert agent.elaborate("dog") in agent._assoc_graph().get("dog", {})
    agent.composer.kb = []                              # same length (2 facts), different graph
    agent.hear("cat stop west")
    agent.hear("cat look east")
    assert agent.elaborate("dog") is None              # dog absent from the new graph
    assert agent.elaborate("cat") in agent._assoc_graph().get("cat", {})
