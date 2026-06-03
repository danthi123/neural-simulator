"""Integrated conversation loop (milestone 1, numpy): comprehend -> decide-what-to-say (content-selection
Control) -> produce (generate-by-composition). The agent hears SVO statements (binds them into a knowledge
base), answers direct factual questions with PRODUCED sentences, AND -- the new capability from this
session's validated content-selection Control -- ELABORATES on a topic: it walks its associative memory
(the Control over an association graph built from its OWN KB), bringing up the most relevant unsaid fact
each turn and producing it as an ordered sentence, coherent + non-repeating + progressing.

This unifies the project's three validated conversational abilities into one fluid loop:
  comprehend (SVO parse)  +  decide-what-to-say (content-selection Control)  +  produce (generate-by-
  composition).
The existing _integrated_conversation_loop_demo only did factual Q&A (react to direct questions); the
Control adds the missing DIALOGUE-PLANNING -- deciding what to say when the user just raises a topic.

Staged (this = milestone 1, numpy, fastest to a tangible end-to-end agent): reuses the validated
production (generate-by-composition) + the structured content-selection Control. Milestone 2 swaps in the
spiking SpikingSpreadingController (same `.turn` API) for the biology-faithful version -- the KB-graph +
produce wiring is unchanged.

Honest scope: numpy substrate + a simple position-based SVO parser (the faithful pieces are the validated
spiking Control + the Hebbian conjunctive parser); this milestone demonstrates the INTEGRATION + dialogue
planning. Reuse-by-import only; no protected-module change.

  python -m research.runners.integrated_conversation_loop          # scripted demo
  python -m research.runners.integrated_conversation_loop --repl   # live shell
"""
from __future__ import annotations

from research.findings.raw._generate_by_composition_probe import build_world, compose, generate
from research.runners.content_selection import build_association_graph, ContentSelectionController
from research.runners.conjunctive_parser import ConjunctiveParser

WORDS = ["dog", "cat", "bird", "child", "ball", "apple", "river", "sun",          # 0-7 nouns
         "chase", "eat", "see", "hold", "want", "give", "watch", "find",          # 8-15 verbs
         "small", "red", "happy", "fast", "big", "cold"]                          # 16-21 adj/patient
W2I = {w: i for i, w in enumerate(WORDS)}
ROLES3 = ["agent", "action", "patient"]


class ConversationalAgent:
    """Comprehend -> bind into KB -> answer questions (produce) -> AND elaborate topics via the
    content-selection Control (decide-what-to-say), producing composed sentences."""

    def __init__(self, seed=42, D=1024, controller_factory=None):
        self.concepts, self.roles = build_world(len(WORDS), D, seed)
        self.words = list(self.concepts.keys())     # concept keys = word indices 0..N-1
        self.kb = []                                # list of {role: word_index} facts (the memory)
        self.focus = None                           # current elaboration topic (a word string)
        self.ctrl = None                            # content-selection Control for the current focus
        self._elaborated = set()                    # ids of facts already elaborated for this focus
        # the Control backend (decide-what-to-say): default = structured ContentSelectionController (fast).
        # Inject a factory building a SpikingSpreadingController for the faithful spiking dialogue planning
        # (milestone 2) -- the .turn API is identical, so the KB-graph + produce wiring is unchanged.
        self._controller_factory = controller_factory or (lambda graph: ContentSelectionController(graph))
        self.parser = ConjunctiveParser()           # learned, voice-invariant COMPREHEND (milestone 3a)
        self._vocab = set(W2I)

    # --- PRODUCE (generate-by-composition) ---
    def _say(self, fact):
        bound = compose(fact, self.concepts, self.roles)
        idxs = generate(bound, ROLES3, self.concepts, self.roles, self.words)
        return " ".join(WORDS[i] for i in idxs)

    # --- the agent's KB-derived association graph (what it knows) ---
    def _kb_graph(self):
        pairs = []
        for f in self.kb:
            ws = [WORDS[f[r]] for r in ROLES3]
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    pairs.append(f"{ws[i]}_{ws[j]}")
        return build_association_graph(pairs) if pairs else {}

    def _facts_about(self, word):
        wi = W2I.get(word)
        return [f for f in self.kb if wi in (f["agent"], f["action"], f["patient"])]

    def _fact_linking(self, a_word, b_word):
        """An unsaid KB fact whose words include both a_word and b_word."""
        ai, bi = W2I.get(a_word), W2I.get(b_word)
        for f in self.kb:
            vals = (f["agent"], f["action"], f["patient"])
            if ai in vals and bi in vals and id(f) not in self._elaborated:
                return f
        return None

    # --- DECIDE WHAT TO SAY (content-selection Control) + produce ---
    def _set_topic(self, topic):
        self.focus = topic
        self._elaborated = set()
        self.ctrl = self._controller_factory(self._kb_graph())

    def _elaborate(self):
        """Pick the most relevant unsaid fact about the focus (Control over the KB graph) and produce it.
        The Control guarantees the pick is on-topic + not recently said; we map it to an unsaid fact that
        links the focus to the picked concept, and produce that fact as a sentence."""
        if self.focus is None:
            return "(no topic yet)"
        if not self._facts_about(self.focus):
            return f"(i don't know about {self.focus})"
        # prefer the controller's latency read (focused 1-hop, robust on the connected KB graph) if present
        turn_fn = getattr(self.ctrl, "turn_latency", None) or self.ctrl.turn
        for _ in range(len(WORDS)):                  # walk the Control's ranked associates until one maps
            pick = turn_fn([self.focus])             # to an unsaid fact (concept-level IoR is the Control's)
            if pick is None:
                break
            fact = self._fact_linking(self.focus, pick)
            if fact is not None:
                self._elaborated.add(id(fact))
                return self._say(fact)
        return f"(that's all i know about {self.focus})"

    # --- COMPREHEND + dispatch ---
    def _answer_question(self, toks):
        if toks[0] == "tell":                         # "tell me about <noun>"
            noun = toks[-1]
            facts = self._facts_about(noun)
            if not facts:
                return f"(i don't know about {noun})"
            return " ; ".join(self._say(f) for f in facts)
        content = [w for w in toks[1:] if w in W2I]
        if toks[0] == "what" and len(content) >= 2:   # agent + action known -> want patient
            ag, ac = W2I[content[0]], W2I[content[1]]
            for f in self.kb:
                if f["agent"] == ag and f["action"] == ac:
                    return self._say(f)
            return f"(i don't know what {WORDS[ag]} {WORDS[ac]})"
        if toks[0] == "who" and len(content) >= 2:    # action + patient known -> want agent
            ac, pa = W2I[content[0]], W2I[content[1]]
            for f in self.kb:
                if f["action"] == ac and f["patient"] == pa:
                    return self._say(f)
            return f"(i don't know who {WORDS[ac]} {WORDS[pa]})"
        return "(i didn't understand the question)"

    def hear(self, text):
        toks = (text or "").strip().rstrip("?").lower().split()
        if not toks:
            return "(i didn't understand)"
        if toks[0] in ("what", "who", "tell"):                         # QUESTION -> retrieve + produce
            return self._answer_question(toks)
        if toks[0] in ("more", "and") and self.focus is not None:      # CONTINUE elaborating
            return self._elaborate()
        meaning = self.parser.parse(text, self._vocab)                 # STATEMENT (voice-invariant) -> bind
        if meaning is not None:
            fact = {r: W2I[meaning[r]] for r in ROLES3}
            if fact in self.kb:                                        # active+passive of the same fact -> dedup
                return "i already knew: " + self._say(fact)
            self.kb.append(fact)
            return "ok -- i learned: " + self._say(fact)
        content = [t for t in toks if t in W2I]
        if len(content) == 1:                                          # TOPIC -> elaborate via Control
            self._set_topic(content[0])
            return self._elaborate()
        return "(i didn't understand)"


def make_spiking_agent(seed=42, D=1024):
    """Milestone 2: a ConversationalAgent whose DECIDE-WHAT-TO-SAY runs on the validated SPIKING content-
    selection Control (SpikingSpreadingController -- spiking working memory + spreading-activation relevance
    + latency read). Slower per topic (it builds a spiking bridge when the topic changes) but the dialogue
    planning is faithful spiking. The loop, the KB-graph, and the production are unchanged."""
    from research.runners.content_selection_spiking import SpikingSpreadingController
    return ConversationalAgent(
        seed=seed, D=D,
        controller_factory=lambda graph: SpikingSpreadingController(graph, seed=seed))


def run_conversation(script, seed=42, agent=None):
    """Drive a scripted conversation; return [(user, agent), ...]."""
    a = agent if agent is not None else ConversationalAgent(seed=seed)
    return [(u, a.hear(u)) for u in script]


def repl(seed=42, agent=None):
    a = agent if agent is not None else ConversationalAgent(seed=seed)
    print("Integrated conversational agent -- live. Teach SVO facts ('dog chase cat'), ask questions")
    print("  ('what does dog chase' | 'who chase cat' | 'tell me about cat'), or raise a topic ('dog' |")
    print("  'more') to have the agent ELABORATE its memory. 'quit' to exit.\n")
    while True:
        try:
            u = input("user : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("quit", "exit"):
            break
        print(f"agent: {a.hear(u)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repl", action="store_true", help="live interactive shell")
    ap.add_argument("--spiking", action="store_true",
                    help="run dialogue planning on the faithful SPIKING content-selection Control (slower)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    agent = make_spiking_agent(args.seed) if args.spiking else None
    if args.repl:
        repl(args.seed, agent=agent)
        return
    if args.spiking:                                 # shorter script (spiking is slower per topic)
        print("=== integrated loop on the SPIKING content-selection Control (milestone 2) ===\n")
        script = ["dog chase cat", "dog eat apple", "child hold ball",
                  "what does dog chase", "dog", "more", "more", "child"]
        for user, ag in run_conversation(script, agent=agent):
            print(f"  user : {user}")
            print(f"  agent: {ag}")
        print("\n  -> the same conversational loop, dialogue planning computed by SPIKING spreading-"
              "activation + latency.")
        return

    print("=== integrated conversation loop: comprehend -> decide-what-to-say (Control) -> produce ===\n")
    script = [
        "dog chase cat", "dog eat apple", "child hold ball", "bird see river",   # teach facts
        "what does dog chase",          # factual Q&A -> "dog chase cat"
        "who hold ball",                # -> "child hold ball"
        "dog",                          # TOPIC -> Control elaborates a dog fact (produced sentence)
        "more",                         # -> the OTHER dog fact (coherent, non-repeating)
        "more",                         # -> "that's all i know about dog"
        "child",                        # topic shift -> a child fact
        "tell me about cat",            # -> all cat facts, each produced
    ]
    for user, agent in run_conversation(script):
        print(f"  user : {user}")
        print(f"  agent: {agent}")
    print("\n  -> the agent comprehends statements, answers factual questions with PRODUCED sentences,")
    print("     AND elaborates a topic by walking its associative memory (content-selection Control)")
    print("     -- a coherent, progressing conversation unifying the three validated abilities.")


if __name__ == "__main__":
    main()
