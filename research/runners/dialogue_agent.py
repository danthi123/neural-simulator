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
    _LINK_WORDS = ("link", "links", "common", "both", "connect", "share")
    _DESCRIBE_WORDS = ("tell", "describe", "about", "know")

    def __init__(self, graph, controller=None, **ctrl_kwargs):
        """`controller` lets a caller inject an alternative content-selection backend with the same
        interface (`.turn(concepts)` + `.ctx.update(concepts)`) -- e.g. the validated faithful
        SpikingSpreadingController, so the SAME interactive dialogue runs on the spiking substrate. When
        omitted, the structured (numpy) ContentSelectionController is used (fast default)."""
        self.graph = graph
        self.ctrl = controller if controller is not None else ContentSelectionController(graph, **ctrl_kwargs)
        self.focus = None
        self._nodes = set(graph) | {a for v in graph.values() for a in v}

    def _known(self, text):
        """Known concepts (association-graph nodes) mentioned in the text, in order of appearance."""
        return [w for w in text.replace("?", " ").split() if w in self._nodes]

    def _assoc(self, x, y):
        return float(self.graph.get(x, {}).get(y, 0.0) or self.graph.get(y, {}).get(x, 0.0))

    def _answer_yes_no(self, x, y, negated=False):
        s = self._assoc(x, y)
        associated = s > 0
        if negated:                                  # user asked "is X NOT related to Y"
            if associated:
                return f"Actually, {x} and {y} ARE associated (strength {s:.1f})."
            return f"Right -- {x} and {y} are not associated."
        if associated:
            return f"Yes -- {x} and {y} are associated (strength {s:.1f})."
        return f"No -- {x} and {y} are not directly associated."

    def _describe(self, x, top_k=3):
        """A multi-fact summary: the top-k strongest associates of x, as a sentence."""
        assoc = sorted(self.graph.get(x, {}).items(), key=lambda kv: -kv[1])[:top_k]
        names = [a for a, _ in assoc]
        if not names:
            return f"I don't know anything about {x}."
        if len(names) == 1:
            return f"{x} is associated with {names[0]}."
        return f"{x} is associated with {', '.join(names[:-1])} and {names[-1]}."

    def _answer_common(self, x, y):
        common = sorted(set(self.graph.get(x, {})) & set(self.graph.get(y, {})))
        if common:
            return f"{x} and {y} are both associated with: {', '.join(common)}."
        return f"{x} and {y} have no association in common."

    def _elaborate(self):
        """One elaboration turn on the current focus. Prefers the controller's latency read
        (`turn_latency`: focused 1-hop, robust on connected graphs) when it exposes one -- this is how the
        spiking backend stays on-topic on a richly-connected association graph where the rate read would
        over-spread. The structured controller (no `turn_latency`) falls back to its `turn`."""
        turn_fn = getattr(self.ctrl, "turn_latency", None) or self.ctrl.turn
        return turn_fn([self.focus])

    def respond(self, user_input):
        """Return the agent's response (a string). A concept name elaborates that topic (returns the
        chosen associate); 'more'/'' continues the current topic; a yes/no or common-link question is
        answered from the association substrate. Mentioned concepts feed the Control context so the
        conversation stays coherent across questions and topics."""
        text = (user_input or "").strip().lower()
        words = text.replace("?", " ").split()
        if text in ("more", ""):                                  # follow-up on the current topic
            if self.focus is None:
                return None
            return self._elaborate()
        known = self._known(text)
        if known and any(w in words for w in self._DESCRIBE_WORDS):   # "tell me about X" -> multi-fact
            self.focus = known[0]
            self.ctrl.ctx.update([known[0]])
            return self._describe(known[0])
        if len(known) >= 2 and any(w in words for w in self._LINK_WORDS):
            self.ctrl.ctx.update(known[:2])                       # the asked-about concepts enter context
            return self._answer_common(known[0], known[1])
        if len(known) >= 2 and any(w in words for w in self._YESNO_WORDS):
            self.ctrl.ctx.update(known[:2])
            negated = "not" in words or "n't" in text             # "is X NOT related to Y"
            return self._answer_yes_no(known[0], known[1], negated)
        if len(known) >= 1:                                       # a topic -> set focus, elaborate
            if known[0] != self.focus:                            # explicit topic shift: strongly refocus
                if hasattr(self.ctrl, "_reset_wm"):               # spiking backend: clear the persistent WM
                    self.ctrl._reset_wm()                         #  latches so the prior topic doesn't bleed in
                for _ in range(3):                                # (PFC attention reorienting to the new topic
                    self.ctrl.ctx.update([known[0]])              #  so it dominates the accumulated context)
            self.focus = known[0]
            return self._elaborate()
        return "I don't know about that."


def run_conversation(graph, script, **kwargs):
    """Drive a scripted conversation. `script` is a list of user inputs (concept names or 'more').
    Returns a list of (user_input, agent_response) pairs."""
    agent = DialogueAgent(graph, **kwargs)
    return [(u, agent.respond(u)) for u in script]


def repl(graph, controller=None):
    """Live interactive shell: the user types, the agent responds, until 'quit'. Pass `controller` to run
    the dialogue on an alternative backend (e.g. the faithful SpikingSpreadingController)."""
    agent = DialogueAgent(graph, controller=controller)
    backend = type(agent.ctrl).__name__
    print(f"Dialogue agent -- live ({backend}). Try:  <concept> | more | tell me about <X> |")
    print("  is <X> related to <Y> | is <X> not related to <Y> | what links <X> and <Y> | quit\n")
    while True:
        try:
            u = input("user : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("quit", "exit"):
            break
        print(f"agent: {agent.respond(u)}")


def main():
    import argparse
    from research.runners.content_selection_eval import _synthetic_multi_topic_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--repl", action="store_true", help="live interactive dialogue shell")
    ap.add_argument("--spiking", action="store_true",
                    help="run the REPL on the faithful SpikingSpreadingController backend (validated "
                         "spiking content-selection; slower per turn)")
    a = ap.parse_args()
    graph = _synthetic_multi_topic_graph()
    if a.repl:
        controller = None
        if a.spiking:
            from research.runners.content_selection_spiking import SpikingSpreadingController
            controller = SpikingSpreadingController(graph, seed=42)
        repl(graph, controller=controller)
        return

    # Scripted demo showing every capability: multi-fact describe, topic + follow-up, yes/no, negation,
    # common-link, and a coherent topic shift -- all on the validated Control substrate.
    script = [
        "tell me about rain",                 # multi-fact answer
        "rain", "more",                       # topic elaboration + follow-up
        "is rain related to storm?",          # yes/no -> Yes
        "is rain not related to apple?",      # negation -> Right, not associated
        "what links cloud and storm?",        # common-link -> shared associates
        "apple", "more",                      # shift topic + elaborate
    ]
    convo = run_conversation(graph, script)
    print("Interactive dialogue agent -- scripted demo (multi-fact + negation + Q&A + coherent topics)\n")
    for user, agent in convo:
        print(f"  user : {user}")
        print(f"  agent: {agent}")
    print("\n  -> multi-fact answers, negation, association Q&A, and coherent topic elaboration -- a "
          "usable conversational agent on the validated Control.  (run with --repl for a live shell.)")


if __name__ == "__main__":
    main()
