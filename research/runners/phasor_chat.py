"""PhasorChat -- a conversational agent on the biology-grounded phasor substrate.

Type statements and questions in plain words; the agent learns facts and answers them, with the cognition
running on the validated phasor substrate: concept codes LEARNED by spike-timing plasticity
(PhasorAssociativeMemory), facts stored + queried by phasor binding/unbinding with the resonator decode and
abstention (NestedCompositionAgent). It handles the structures the non-invertible production binding cannot:
attributed entities ("cold river") and embedded clauses ("dog eat (cat chase ball)").

Scope (honest): a simple rule-based parser is the LANGUAGE FRONT-END (not claimed biological); the COGNITION
-- memory, composition, nesting, abstention -- is on the biology-grounded substrate. Fixed vocabulary (the
words it knows), declared up front, like the project's tier vocabularies. This is a conversational MEMORY +
composition agent, not free-form generation -- a step toward the goal, honestly scoped.

  python -m research.runners.phasor_chat            # scripted conversation, then an interactive REPL if a TTY
"""
from __future__ import annotations
import sys

from research.runners.phasor_associative_memory import PhasorAssociativeMemory
from research.runners.nested_composition_agent import NestedCompositionAgent, Clause


class PhasorChat:
    """Learn facts from typed statements and answer typed questions, on the phasor learned-code substrate."""

    def __init__(self, nouns, verbs, adjs, D=2048, seed=42):
        self.nouns, self.verbs, self.adjs = set(nouns), set(verbs), set(adjs)
        self.mem = PhasorAssociativeMemory(D=D, seed=seed)
        for w in list(nouns) + list(verbs) + list(adjs):
            self.mem.learn(w)
        learned = {w: self.mem.code(w) for w in list(nouns) + list(verbs) + list(adjs)}
        self.agent = NestedCompositionAgent(list(nouns), list(verbs), list(adjs), D=D, seed=seed,
                                            external_codes=learned)

    def _kind(self, w):
        if w in self.nouns:
            return "n"
        if w in self.verbs:
            return "v"
        if w in self.adjs:
            return "a"
        return None

    def _parse_patient(self, toks):
        """Classify the patient tokens into a flat noun / (adj,noun) / ((adj,adj),noun) / Clause."""
        kinds = [self._kind(t) for t in toks]
        if kinds == ["n"]:
            return toks[0]
        if kinds == ["a", "n"]:
            return (toks[0], toks[1])
        if kinds == ["a", "a", "n"]:
            return ((toks[0], toks[1]), toks[2])
        if kinds == ["n", "v", "n"]:
            return Clause(toks[0], toks[1], toks[2])                       # embedded clause
        if kinds == ["n", "v", "a", "n"]:
            return Clause(toks[0], toks[1], (toks[2], toks[3]))            # clause with attributed patient
        return None

    def say(self, text):
        """Process one conversational turn; return the agent's reply."""
        toks = text.lower().replace("?", " ").replace(".", " ").split()
        if not toks:
            return ""
        unknown = [t for t in toks if t not in {"what", "does", "who"} and self._kind(t) is None]
        if unknown:
            return f"I don't know the word{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}."
        # questions
        if toks[:2] == ["what", "does"] and len(toks) >= 4 and self._kind(toks[2]) == "n" and self._kind(toks[3]) == "v":
            ans = self.agent.query_patient(toks[2], toks[3])
            return ans if ans is not None else f"I don't know what {toks[2]} {toks[3]}s."
        if toks[0] == "who" and len(toks) >= 3 and self._kind(toks[1]) == "v" and self._kind(toks[2]) == "n":
            ans = self.agent.query_agent(toks[1], toks[2])
            return ans if ans is not None else f"I don't know who {toks[1]}s {toks[2]}."
        # statement: <noun> <verb> <patient...>
        if len(toks) >= 3 and self._kind(toks[0]) == "n" and self._kind(toks[1]) == "v":
            patient = self._parse_patient(toks[2:])
            if patient is not None:
                self.agent.learn(toks[0], toks[1], patient)
                return "ok"
        return "I didn't understand that."


def _scripted(chat, turns):
    for t in turns:
        reply = chat.say(t)
        tag = "  you:" if not t.startswith(("what", "who")) else "  you?"
        print(f"{tag} {t}\n   bot: {reply}", flush=True)


def main():
    nouns = ["dog", "cat", "ball", "river", "bird", "child"]
    verbs = ["chase", "see", "eat", "hold", "want"]
    adjs = ["big", "red", "cold", "small"]
    chat = PhasorChat(nouns, verbs, adjs)
    print("=== PhasorChat: conversation on the biology-grounded phasor substrate ===\n", flush=True)
    _scripted(chat, [
        "dog chase cat",                 # flat fact
        "bird see cold river",           # attributed entity
        "child hold big red ball",       # two attributes
        "dog eat cat chase ball",        # embedded clause
        "what does dog chase",           # -> cat
        "what does bird see",            # -> cold river
        "child hold what",               # not understood (front-end is simple)
        "what does child hold",          # -> big red ball
        "what does dog eat",             # -> cat chase ball (embedded clause)
        "who chase cat",                 # -> dog
        "what does cat want",            # -> abstain (never told)
        "dog chase zebra",               # unknown word
    ])
    print("\n  -> a conversational agent that learns facts (incl. attributed entities and embedded clauses)", flush=True)
    print("     and answers + abstains, with the cognition on codes LEARNED by spike-timing plasticity.", flush=True)
    if sys.stdin.isatty():
        print("\n  (interactive: type statements/questions, 'quit' to exit)", flush=True)
        while True:
            try:
                line = input("  you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line in ("quit", "exit"):
                break
            print(f"  bot> {chat.say(line)}", flush=True)


if __name__ == "__main__":
    main()
