"""Unit + smoke tests for the content-selection / dialogue-control layer.

All controller-logic tests use small hand-built (synthetic) association graphs so
they are fast and deterministic and need no spiking bridge / GPU. Plain ASCII only.
"""

from research.runners.content_selection import (
    ContextBuffer,
    relevance,
    SaidTrace,
    select_candidate,
    ContentSelectionController,
    build_association_graph,
)
from research.runners.content_selection_eval import (
    BaselineSelector,
    on_topic,
    non_repetition,
    turn_to_turn_coherence,
    topic_progression,
    run_dialogue,
)


# --- Task 1: Context buffer -------------------------------------------------

def test_context_buffer_decays_and_adds():
    cb = ContextBuffer(decay=0.5)
    cb.update(["apple"])                 # apple enters at weight 1.0
    cb.update(["big"])                   # apple fades to 0.5, big enters at 1.0
    w = cb.weights()
    assert abs(w["big"] - 1.0) < 1e-9
    assert abs(w["apple"] - 0.5) < 1e-9


def test_context_buffer_reinforces_repeat():
    cb = ContextBuffer(decay=0.5)
    cb.update(["apple"]); cb.update(["apple"])   # 1.0 -> 0.5 then +1.0 = 1.5
    assert abs(cb.weights()["apple"] - 1.5) < 1e-9


# --- Task 2: Relevance score ------------------------------------------------

def test_relevance_sums_association_to_active_context():
    graph = {"apple": {"big": 2.0, "cat": 1.0}, "big": {"apple": 2.0}}
    context = {"apple": 1.0}             # we've been talking about apple
    # 'big' is associated with apple at 2.0 -> relevance 2.0; 'cat' at 1.0 -> 1.0
    assert abs(relevance("big", context, graph) - 2.0) < 1e-9
    assert abs(relevance("cat", context, graph) - 1.0) < 1e-9
    assert abs(relevance("apple", context, graph) - 0.0) < 1e-9   # not associated w/ itself here


def test_relevance_weights_by_context_strength():
    graph = {"apple": {"big": 2.0}, "river": {"big": 1.0}}
    context = {"apple": 0.5, "river": 1.0}   # river is more active
    # big: 0.5*2.0 + 1.0*1.0 = 2.0
    assert abs(relevance("big", context, graph) - 2.0) < 1e-9


# --- Task 3: Inhibition-of-return (said trace) ------------------------------

def test_said_trace_decays():
    st = SaidTrace(decay=0.5)
    st.mark("big")                       # big -> 1.0
    st.step()                            # fade -> 0.5
    assert abs(st.activation("big") - 0.5) < 1e-9
    assert st.activation("apple") == 0.0


# --- Task 4: Candidate selection --------------------------------------------

def test_select_prefers_relevant_unsaid():
    graph = {"apple": {"big": 2.0, "cat": 1.0}}
    context = {"apple": 1.0}
    said = {}                            # nothing said yet
    # big (2.0) beats cat (1.0)
    assert select_candidate(["big", "cat"], context, graph, said, lam=1.0) == "big"


def test_select_inhibits_repeats():
    graph = {"apple": {"big": 2.0, "cat": 1.0}}
    context = {"apple": 1.0}
    said = {"big": 2.0}                  # big was just said, strongly
    # big score 2.0 - 1.0*2.0 = 0.0 ; cat score 1.0 - 0 = 1.0 -> cat wins
    assert select_candidate(["big", "cat"], context, graph, said, lam=1.0) == "cat"


# --- Task 5: ContentSelectionController -------------------------------------

def test_controller_walks_coherently_without_repeating():
    # apple links to big and cat; big links to hot. A coherent walk from apple should visit
    # big/cat (apple's associates) without repeating, staying connected.
    graph = {"apple": {"big": 2.0, "cat": 1.5}, "big": {"hot": 2.0, "apple": 2.0},
             "cat": {"apple": 1.5}, "hot": {"big": 2.0}}
    ctrl = ContentSelectionController(graph, decay=0.7, said_decay=0.6, lam=1.0)
    said = [ctrl.turn(["apple"]) for _ in range(3)]   # 3 elaboration turns from topic 'apple'
    assert said[0] == "big"               # apple's strongest associate
    assert len(set(said)) == 3            # no repeats across the 3 turns
    assert "apple" not in said            # doesn't just echo the topic


# --- Task 6: No-control baseline --------------------------------------------

def test_baseline_ignores_context_and_repetition():
    # Baseline = retrieval-only: pick the globally strongest associate of the *last input*,
    # with NO context buffer and NO inhibition -> it will happily repeat.
    graph = {"apple": {"big": 2.0, "cat": 1.0}}
    b = BaselineSelector(graph, seed=0)
    out = [b.turn(["apple"]) for _ in range(3)]
    assert out == ["big", "big", "big"]   # repeats because no inhibition / context progression


# --- Task 7: Coherence metrics ----------------------------------------------

# A directed chain graph: apple -> big (2.0), big -> hot (4.0). Edges are looked up
# as graph[prev][cur] (same orientation as relevance()).
_CHAIN_GRAPH = {"apple": {"big": 2.0}, "big": {"hot": 4.0}, "hot": {}}
_COHERENT_WALK = ["apple", "big", "hot"]      # non-repeating, each turn advances
_PARKED_WALK = ["big", "big", "big"]          # parks on one concept


def test_non_repetition_perfect_and_repeating():
    assert abs(non_repetition(_COHERENT_WALK) - 1.0) < 1e-9     # no repeats
    assert abs(non_repetition(_PARKED_WALK) - (1.0 / 3.0)) < 1e-9   # 2 of 3 turns are repeats


def test_topic_progression_catches_parking():
    # A coherent advancing walk introduces a new concept every turn -> 1.0.
    assert abs(topic_progression(_COHERENT_WALK) - 1.0) < 1e-9
    # A parked walk introduces a new concept only on turn 1 -> near 0 (the honesty guard).
    assert abs(topic_progression(_PARKED_WALK) - (1.0 / 3.0)) < 1e-9


def test_turn_to_turn_coherence_consecutive_pairs():
    # consecutive edges: (apple->big)=2.0, (big->hot)=4.0 -> mean 3.0
    assert abs(turn_to_turn_coherence(_COHERENT_WALK, _CHAIN_GRAPH) - 3.0) < 1e-9


def test_on_topic_running_window():
    # i=1 (big): window [apple] -> assoc(apple->big)=2.0, mean 2.0
    # i=2 (hot): window [apple,big] -> assoc(apple->hot)=0.0, assoc(big->hot)=4.0, mean 2.0
    # overall mean = (2.0 + 2.0) / 2 = 2.0
    assert abs(on_topic(_COHERENT_WALK, _CHAIN_GRAPH) - 2.0) < 1e-9


def test_parked_walk_high_coherence_low_progression():
    # The honesty guard in action: parking on a concept that strongly self-associates can score
    # high turn-to-turn coherence yet near-zero progression. Use a self-loop graph.
    self_loop = {"big": {"big": 5.0}}
    assert abs(turn_to_turn_coherence(_PARKED_WALK, self_loop) - 5.0) < 1e-9   # high coherence
    assert abs(topic_progression(_PARKED_WALK) - (1.0 / 3.0)) < 1e-9           # low progression


# --- Task 8: build_association_graph from substrate tags --------------------

def test_build_association_graph_symmetric_edges():
    graph = build_association_graph(["apple_big", "dog_river"])
    assert graph == {
        "apple": {"big": 1.0},
        "big": {"apple": 1.0},
        "dog": {"river": 1.0},
        "river": {"dog": 1.0},
    }


def test_build_association_graph_skips_malformed():
    # 'weirdtag' has no underscore -> skipped, not crashed. 'a_b_c' is not a 2-concept
    # tag -> skipped. An empty side ('_x' / 'x_') -> skipped.
    graph = build_association_graph(["apple_big", "weirdtag", "a_b_c", "_x", "y_"])
    assert graph == {
        "apple": {"big": 1.0},
        "big": {"apple": 1.0},
    }


def test_build_association_graph_accumulates_strength():
    # The same edge appearing in multiple tags accumulates strength (e.g. apple_big twice -> 2.0).
    graph = build_association_graph(["apple_big", "apple_big"])
    assert abs(graph["apple"]["big"] - 2.0) < 1e-9
    assert abs(graph["big"]["apple"] - 2.0) < 1e-9


# --- Task 9: Integration smoke (controller on a real-shaped graph) ----------

def _connected_component(graph, start):
    """All concepts reachable from `start` via association edges (undirected reachability)."""
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nbr in graph.get(node, {}):
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen


def test_run_dialogue_nonempty_nonrepeating_in_component():
    # apple - big - hot chain plus apple - cat branch. All four are in apple's component.
    graph = build_association_graph(["apple_big", "big_hot", "apple_cat"])
    transcript = run_dialogue(graph, topic="apple", n_turns=5)
    assert len(transcript) > 0                         # non-empty
    assert len(transcript) == len(set(transcript))     # non-repeating
    component = _connected_component(graph, "apple")
    assert all(c in component for c in transcript)      # stays within topic's component
    assert "apple" not in transcript                   # elaborates, doesn't echo the topic


def test_run_dialogue_isolated_topic_returns_empty():
    # A topic with no associations has an empty component (besides itself); the controller has
    # nothing relevant to say and must not fabricate off-topic content.
    graph = build_association_graph(["dog_river"])      # apple absent entirely
    transcript = run_dialogue(graph, topic="apple", n_turns=3)
    assert transcript == []
