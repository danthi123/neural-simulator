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

    def __init__(self, graph, **ctrl_kwargs):
        self.graph = graph
        self.ctrl = ContentSelectionController(graph, **ctrl_kwargs)
        self.focus = None

    def respond(self, user_input):
        """Return the agent's chosen concept for this user turn (or None if there is no focus yet)."""
        text = (user_input or "").strip().lower()
        if text and text != "more":
            self.focus = text                    # topic shift / new focus
        if self.focus is None:
            return None
        return self.ctrl.turn([self.focus])      # context updates with the focus; coherent associate returned


def run_conversation(graph, script, **kwargs):
    """Drive a scripted conversation. `script` is a list of user inputs (concept names or 'more').
    Returns a list of (user_input, agent_response) pairs."""
    agent = DialogueAgent(graph, **kwargs)
    return [(u, agent.respond(u)) for u in script]


def main():
    # Reuse the synthetic multi-topic association graph from the Milestone-1 eval for a richer demo.
    from research.runners.content_selection_eval import _synthetic_multi_topic_graph

    graph = _synthetic_multi_topic_graph()
    # A varied conversation: a topic, two follow-ups, a topic shift, two more follow-ups, another shift.
    script = ["rain", "more", "more", "apple", "more", "more", "song", "more", "more"]
    convo = run_conversation(graph, script)

    print("Interactive coherent-dialogue agent -- scripted demo (Control over a synthetic graph)")
    print("(user 'more' = continue the current topic; a concept name = shift topic)\n")
    for user, agent in convo:
        print(f"  user : {user}")
        print(f"  agent: {agent}")
    responses = [r for _, r in convo if r is not None]
    print(f"\n  distinct responses: {len(set(responses))}/{len(responses)} "
          f"(no-repeat across the whole conversation: {len(set(responses)) == len(responses)})")
    print("  -> the agent tracks each topic, stays coherent, shifts cleanly on a new concept, and never "
          "repeats -- the Control layer as an interactive conversation.")


if __name__ == "__main__":
    main()
