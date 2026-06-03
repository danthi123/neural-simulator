"""Coherence evaluation for the content-selection / dialogue-control layer.

Contains:
  - BaselineSelector: a no-control, retrieval-only selector (no context buffer, no
    inhibition) that the ContentSelectionController is compared against.
  - Four coherence metrics over a dialogue transcript (Task 7).
  - run_dialogue: a multi-turn dialogue runner that drives a selector for n_turns
    elaboration turns from a topic concept and returns the transcript (Task 9).
  - main(): the decisive multi-seed coherence eval (Task 10) -- left as a TODO for
    the human controller to fill in and run; see the marked section at the bottom.

Reuse-by-import only; does not modify any sim/ / experiment/ module or any bridge.
Plain ASCII, Python 3 + NumPy.

Manual (non-pytest) smoke against a real substrate bridge
---------------------------------------------------------
To inspect a real dialogue transcript by hand (the transcript honesty guard),
load a saved substrate bridge, build its association graph from its engram tags,
and run a dialogue. A loaded bridge exposes its tags via
``bridge.list_engram_tags()`` (a list of dicts with a ``name`` key; see
``research/runners/compose_concept_chat.py``). Sketch::

    from research.runners.content_selection import build_association_graph
    from research.runners.content_selection_eval import run_dialogue
    # ... load a saved bridge here (reuse-by-import; do not modify it) ...
    tag_names = [t["name"] for t in bridge.list_engram_tags()]
    graph = build_association_graph(tag_names)
    print(run_dialogue(graph, topic="apple", n_turns=6))
"""

import numpy as np

from research.runners.content_selection import ContentSelectionController


class BaselineSelector:
    """No-control, retrieval-only baseline.

    Each turn it returns the globally strongest associate of the *last input*
    concept(s), with NO context buffer and NO inhibition-of-return, so it has
    nothing to drive topic progression or to prevent repetition. This is a fair
    retrieval-only selector (the strongest single-step associative retrieval),
    not a crippled strawman -- it simply lacks the Control layer.

    Ties in association strength are broken by a seeded RNG for determinism.
    """

    def __init__(self, graph, seed: int = 0):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

    def turn(self, user_concepts):
        # Aggregate the outgoing association strength of every candidate over the
        # current input concept(s). No context buffer: only this turn's input counts.
        scores: dict[str, float] = {}
        for c in user_concepts:
            for associate, strength in self.graph.get(c, {}).items():
                scores[associate] = scores.get(associate, 0.0) + float(strength)
        if not scores:
            return None
        best_score = max(scores.values())
        # Among all associates tied at the max strength, pick one with a seeded RNG.
        tied = sorted(k for k, v in scores.items() if v == best_score)
        if len(tied) == 1:
            return tied[0]
        idx = int(self.rng.integers(0, len(tied)))
        return tied[idx]


# --- Coherence metrics (Task 7) ---------------------------------------------
#
# Each metric takes a dialogue transcript (the list of concepts said, in order)
# and -- where relevant -- the association graph, and returns a single float.
# Association strength between two concepts a (earlier) and b (later) is read as
# graph[a][b], the same orientation as content_selection.relevance().


def _assoc(graph, a, b):
    """Association strength of the directed edge a -> b (0.0 if absent)."""
    return float(graph.get(a, {}).get(b, 0.0))


def on_topic(transcript, graph):
    """Mean association-strength of each said concept to the concepts said just before it
    (a running window). Higher = stays on topic.

    For each turn i >= 1, average the association strength from every preceding concept
    (the running window = the whole prefix transcript[0:i]) to the current concept
    transcript[i]; then average those per-turn means over all i >= 1. Returns 0.0 for
    transcripts shorter than 2 turns."""
    if transcript is None or len(transcript) < 2:
        return 0.0
    per_turn = []
    for i in range(1, len(transcript)):
        window = transcript[:i]
        cur = transcript[i]
        per_turn.append(sum(_assoc(graph, prev, cur) for prev in window) / len(window))
    return float(sum(per_turn) / len(per_turn))


def non_repetition(transcript):
    """1 - (repeated_turns / total_turns). Higher = fewer repeats.

    A turn is a "repeated turn" if its concept has already appeared earlier in the
    transcript. Returns 1.0 for an empty transcript (nothing repeated)."""
    if not transcript:
        return 1.0
    seen = set()
    repeated = 0
    for c in transcript:
        if c in seen:
            repeated += 1
        else:
            seen.add(c)
    return float(1.0 - repeated / len(transcript))


def turn_to_turn_coherence(transcript, graph):
    """Mean association-strength between consecutive said concepts. Higher = adjacent turns
    relate. Returns 0.0 for transcripts shorter than 2 turns."""
    if transcript is None or len(transcript) < 2:
        return 0.0
    pairs = [_assoc(graph, transcript[i - 1], transcript[i]) for i in range(1, len(transcript))]
    return float(sum(pairs) / len(pairs))


def topic_progression(transcript):
    """Fraction of turns that introduce a not-yet-said concept. Higher = keeps advancing.

    A controller that parks on one concept scores high coherence but LOW progression, which is
    exactly what this metric is designed to catch. Returns 0.0 for an empty transcript."""
    if not transcript:
        return 0.0
    seen = set()
    introduced = 0
    for c in transcript:
        if c not in seen:
            introduced += 1
            seen.add(c)
    return float(introduced / len(transcript))
