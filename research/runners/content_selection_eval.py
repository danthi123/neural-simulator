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
