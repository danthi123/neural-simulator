"""Phase-2 on-brain conversational test set: the BrainConversationalAgent runs the whole loop -- comprehend
(Hebbian parser bridge) -> store/recall/compose (composer bridge) -> who/what Q&A, abstention, negation, clauses,
and voice-invariant comprehension -- all spiking on SimulationBridge neurons, no bolted-on numpy simulator.

Builds two real bridges (parser ~126 neurons + composer ~6400); a module-scoped agent is built ONCE and the KB is
cleared between tests. Runs on the available backend (GPU when present); skips if the substrate concept-code cache
is absent."""
import pytest

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.core_sim_composition import Clause


@pytest.fixture(scope="module")
def agent():
    try:
        return BrainConversationalAgent(seed=42)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present")


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
    """Dialogue planning: the dlPFC spiking content-selection Control brings up an on-topic associate from the
    agent's own facts; abstains on an unconnected topic."""
    agent.composer.kb = []
    agent.hear("dog go north")
    agent.hear("dog come south")
    associates = set(agent._assoc_graph().get("dog", {}))
    assert agent.elaborate("dog") in associates       # an on-topic concept, chosen on the dlPFC bridge
    assert agent.elaborate("river") is None            # unconnected topic


def test_generation_describe(agent):
    """Generation: the agent produces a sentence about a known subject from its spiking memory; abstains (None) on
    an unknown subject (no confabulation)."""
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


def test_spiking_cleanup_agent_qa():
    """Cheat-B conversion at the AGENT level: enable_spiking_cleanup routes the agent's composer cleanup through the
    FULLY-on-bridge spiking path (matched filter on the complex synapse + Izhikevich WTA, argmax-over-firing). The
    whole comprehend/store/QA loop + the no-confab moat must work, identical to the numpy default (== numpy at the
    composer parity, 27/27 multi-seed). GPU-only: the Hebbian BridgeParser is GPU-validated."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("BridgeParser is GPU-validated (numpy-backend KeyError in the parser bridge)")
    try:
        sp = BrainConversationalAgent(seed=42, enable_spiking_cleanup=True)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present")
    sp.hear("dog go north")
    sp.hear("cat come south")
    assert sp.what_does("dog", "go") == "north"          # cleanup runs on the substrate (spikes)
    assert sp.who_does("go", "north") == "dog"
    assert sp.what_does("river", "look") is None         # no-confab moat preserved under the spiking cleanup


def test_substrate_store_agent_qa():
    """Cheat-C conversion at the AGENT level: enable_substrate_store holds the agent's fact memory in the SUBSTRATE
    (per-fact trigger->readout complex weights), retrieved via firing. Combined with enable_spiking_cleanup, BOTH the
    fact memory AND the cleanup run on the substrate -- the fully-on-substrate conversational path. The whole
    comprehend/store/QA loop + the no-confab moat must work. GPU-only: the Hebbian BridgeParser is GPU-validated."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("BridgeParser is GPU-validated (numpy-backend KeyError in the parser bridge)")
    try:
        ag = BrainConversationalAgent(seed=42, enable_spiking_cleanup=True, enable_substrate_store=True)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present")
    ag.hear("dog go north")
    ag.hear("cat come south")
    assert ag.what_does("dog", "go") == "north"          # memory AND cleanup both on the substrate
    assert ag.who_does("go", "north") == "dog"
    assert ag.what_does("river", "look") is None         # no-confab moat preserved


def test_neural_render_describe():
    """Sentence-generation de-templating (CYCLE 105 Step 2): enable_neural_render routes describe()'s word
    ORDERING through the de-risked spiking competitive-queuing serial-order generator (NeuralSerialOrderRenderer)
    instead of the host f-string `f"{agent} {action} {patient}"`. The recalled sentence + the no-confab moat must
    be IDENTICAL to the f-string default (the order is now produced by spiking rate ranking, not a host literal).
    GPU-only (the ordering runs on a spiking bridge; the BridgeParser is GPU-validated)."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("the neural renderer + BridgeParser are GPU-validated")
    try:
        ag = BrainConversationalAgent(seed=42, enable_neural_render=True)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present")
    ag.hear("dog go north")
    assert ag.describe("dog") == "dog go north"          # word order produced by the spiking CQ read-out
    assert ag.describe("river") is None                  # no-confab moat preserved (abstain BEFORE ordering)


def test_neural_render_clause():
    """CYCLE 107: enable_neural_render also de-templates the GENERATION (describe) path's nested-CLAUSE order --
    the inner clause's SVO order ('cat go south') is produced by the spiking CQ, not a host f-string, AND the outer
    'dog look <clause>' order is neural. The Q&A path (query_patient/what_does) is intentionally left host for this
    step (a separate follow-on). The no-confab moat must hold. GPU-only."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("the neural renderer + BridgeParser are GPU-validated")
    try:
        ag = BrainConversationalAgent(seed=42, enable_neural_render=True)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present")
    ag.hear_clause_fact("dog", "look", Clause("cat", "go", "south"))
    assert ag.describe("dog") == "dog look cat go south"     # outer SVO + inner clause BOTH neural-ordered
    assert ag.describe("river") is None                       # no-confab moat preserved (abstain before ordering)


def test_learned_assoc_graph_agent():
    """Cheat-D conversion at the AGENT level: enable_learned_assoc makes dialogue planning (elaborate) spread over a
    SUBSTRATE-LEARNED association graph -- a sparse Hebbian recurrent (the CA3 autoassociator / _D_sparse_heteroassoc
    mechanism) -- instead of the Python co-occurrence recompute. The graph is LEARNED from heard facts (store_fact
    co-fires the concepts -> the recurrent grows), and elaborate brings up a true co-occurring associate. GPU-only."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("BridgeParser is GPU-validated; LearnedAssocGraph also builds a bridge")
    try:
        ag = BrainConversationalAgent(seed=42, enable_learned_assoc=True)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present")
    for s in ["dog go north", "cat run south", "dog look apple"]:
        ag.hear(s)
    assert ag._learned_assoc is not None                                  # substrate-learned, NOT a Python recompute
    assert set(ag._assoc_graph().get("dog", {})) & {"go", "north", "look", "apple"}   # learned = true co-occurrences
    assert ag.elaborate("dog") in {"go", "north", "look", "apple"}        # dialogue planning -> a true associate
