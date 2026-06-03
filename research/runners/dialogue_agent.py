"""Interactive coherent-dialogue agent -- the content-selection Control layer (Milestone 1, VALIDATED)
turned into a demonstrable back-and-forth conversation.

A user converses by providing topics and follow-ups; the agent responds with coherent, context-tracking,
non-repeating associations across the WHOLE conversation, including topic shifts. It is a thin wrapper
over the validated ContentSelectionController -- the value is that it makes the proven Control mechanism
usable as an interactive dialogue (vs Milestone 1's single-topic elaboration eval).

Reuse-by-import only (the validated controller + an association graph); no protected-module edits.
Plain ASCII, Python 3.

  python -m research.runners.dialogue_agent     # scripted demo on a synthetic multi-topic graph
"""
from __future__ import annotations

from research.runners.content_selection import ContentSelectionController


class DialogueAgent:
    """A conversational wrapper over ContentSelectionController.

    A user turn is either:
      - a concept name  -> set it as the current focus and respond with a coherent associate;
      - 'more' or ''    -> continue the current focus, responding with the next coherent associate.

    The controller's context buffer and said-trace persist across the whole conversation, so coherence
    carries across topic shifts and the agent does not repeat itself.
    """

    _YESNO_WORDS = ("is", "are", "related", "associated", "?", "does", "do")
    _LINK_WORDS = ("link", "links", "common", "both", "connect", "share", "and")

    def __init__(self, graph, **ctrl_kwargs):
        self.graph = graph
        self.ctrl = ContentSelectionController(graph, **ctrl_kwargs)
        self.focus = None
        self._nodes = set(graph) | {a for v in graph.values() for a in v}

    def _known(self, text):
        """Known concepts (association-graph nodes) mentioned in the text, in order of appearance."""
        return [w for w in text.replace("?", " ").split() if w in self._nodes]

    def _assoc(self, x, y):
        return float(self.graph.get(x, {}).get(y, 0.0) or self.graph.get(y, {}).get(x, 0.0))

    def _answer_yes_no(self, x, y):
        s = self._assoc(x, y)
        if s > 0:
            return f"Yes -- {x} and {y} are associated (strength {s:.1f})."
        return f"No -- {x} and {y} are not directly associated."

    def _answer_common(self, x, y):
        common = sorted(set(self.graph.get(x, {})) & set(self.graph.get(y, {})))
        if common:
            return f"{x} and {y} are both associated with: {', '.join(common)}."
        return f"{x} and {y} have no association in common."

    def respond(self, user_input):
        """Return the agent's response (a string). A concept name elaborates that topic (returns the
        chosen associate); 'more'/'' continues the current topic; a yes/no or common-link question is
        answered from the association substrate. Mentioned concepts feed the Control context so the
        conversation stays coherent across questions and topics."""
        text = (user_input or "").strip().lower()
        if text in ("more", ""):                                  # follow-up on the current topic
            if self.focus is None:
                return None
            return self.ctrl.turn([self.focus])
        known = self._known(text)
        if len(known) >= 2 and any(w in text.split() for w in self._LINK_WORDS):
            self.ctrl.ctx.update(known[:2])                       # the asked-about concepts enter context
            return self._answer_common(known[0], known[1])
        if len(known) >= 2 and any(w in text.split() for w in self._YESNO_WORDS):
            self.ctrl.ctx.update(known[:2])
            return self._answer_yes_no(known[0], known[1])
        if len(known) >= 1:                                       # a topic -> set focus, elaborate
            self.focus = known[0]
            return self.ctrl.turn([self.focus])
        return "I don't know about that."


def run_conversation(graph, script, **kwargs):
    """Drive a scripted conversation. `script` is a list of user inputs (concept names or 'more').
    Returns a list of (user_input, agent_response) pairs."""
    agent = DialogueAgent(graph, **kwargs)
    return [(u, agent.respond(u)) for u in script]


def main():
    # Reuse the synthetic multi-topic association graph from the Milestone-1 eval for a richer demo.
    from research.runners.content_selection_eval import _synthetic_multi_topic_graph

    graph = _synthetic_multi_topic_graph()
    # A varied conversation mixing topic elaboration, follow-ups, yes/no questions, and a common-link
    # question -- all answered from the association substrate with Control-driven coherence.
    script = [
        "rain", "more",                       # elaborate a topic
        "is rain related to storm?",          # yes/no question  -> Yes
        "what links cloud and storm?",        # common-link question -> their shared associates
        "is apple related to rain?",          # yes/no question  -> No (different topics)
        "apple", "more",                      # shift topic + elaborate
        "song", "more",                       # shift topic + elaborate
    ]
    convo = run_conversation(graph, script)

    print("Interactive coherent-dialogue agent -- scripted demo (Control + KB question-answering)")
    print("(a concept = topic; 'more' = continue; 'is X related to Y' / 'what links X and Y' = questions)\n")
    for user, agent in convo:
        print(f"  user : {user}")
        print(f"  agent: {agent}")
    print("\n  -> the agent answers association questions AND elaborates topics coherently, tracking "
          "context across both -- content-selection Control as an interactive question-answering dialogue.")


if __name__ == "__main__":
    main()
